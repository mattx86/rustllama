//! Reference scalar CPU kernels used by `rustllama-models`.
//!
//! Phase 1 ships the math we need for a Llama-family forward pass: gemm with
//! F16 weights × F32 activations, RMSNorm, RoPE (neox / half-split variant
//! used by HF-converted GGUFs), embedding lookup, softmax, SwiGLU, and a
//! grouped-query attention block that handles both prefill and decode.
//!
//! All kernels are naive scalar loops. SIMD (AVX2 / AVX-512) and the SYCL
//! GPU equivalents come in later phases; correctness first.

pub mod delta_net;
pub mod mlx_affine;
pub mod mxfp;
pub mod mxfp_kv;
pub mod nvfp4;
pub mod hadamard;
pub mod q4_0_kv;
pub mod turboquant;

use half::f16;
use rustllama_tensor::{as_bytes, as_slice_f16, as_slice_f32, Dtype, Tensor};

/// `C = W · X` where `W: [M, K]` (row-major, F16) and `X: [K, N]` (col-major
/// for `N == 1` is a vector, otherwise row-major). Output `C: [M, N]`
/// row-major, F32. Activations accumulate in F32.
pub fn gemm_f16_w_f32_a(w: &[f16], x: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    debug_assert_eq!(w.len(), m * k);
    debug_assert_eq!(x.len(), k * n);
    debug_assert_eq!(c.len(), m * n);
    for i in 0..m {
        let w_row = &w[i * k..(i + 1) * k];
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += w_row[p].to_f32() * x[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
}

/// Matvec dispatcher that picks the right kernel based on the weight
/// tensor's dtype. Centralizes the per-dtype branch so the model forward
/// pass stays dtype-agnostic.
///
/// `#[inline]` here lets the compiler resolve the dispatch match at the
/// call site when `w.dtype` is statically known (rare from
/// `rustllama-models` but free when it does happen) and, more
/// importantly, removes the cross-crate function-call boundary from
/// the per-layer per-token hot path on Llama forward.
#[inline]
pub fn matvec_tensor(w: &Tensor, x: &[f32], out: &mut [f32], m: usize, k: usize) {
    // F16/F32 carry their own row-count auto-dispatchers.
    match w.dtype {
        Dtype::F16 => return matvec_f16_w_f32_a(as_slice_f16(w), x, out, m, k),
        Dtype::F32 => return matvec_f32(as_slice_f32(w), x, out, m, k),
        _ => {}
    }
    // Row-parallel fast path for the byte-addressed row-major
    // kernels: every quant dtype stores rows contiguously
    // (`Dtype::byte_size(k)` bytes per output row), so the M axis
    // splits across the rayon pool with the same chunking heuristic
    // as the F16 path. Per-row accumulation stays inside the serial
    // kernel — results are bit-identical to the serial call. Before
    // this, every quant matvec (including the 248K-row Q6_K LM head)
    // ran on one core.
    let crossover = parallel_matvec_crossover();
    // MLX affine is a self-describing blob (packed codes + f16 sidecar),
    // NOT a fixed-bpw row-major byte stream, so it can't use the generic
    // `byte_size(k)`-per-row split below. It has its own row-band parallel
    // kernel (each output row reads its own packed row + sidecar groups).
    if w.dtype == Dtype::MlxAffineRaw {
        if crossover > 0 && m >= crossover {
            let n_threads = rayon::current_num_threads().max(1);
            let chunk_rows = ((m + 4 * n_threads - 1) / (4 * n_threads)).max(64).min(m);
            mlx_affine::matvec_mlx_affine_blob_w_f32_a_parallel(
                as_bytes(w), x, out, m, k, chunk_rows,
            );
        } else {
            mlx_affine::matvec_mlx_affine_blob_w_f32_a(as_bytes(w), x, out, m, k);
        }
        return;
    }
    if crossover > 0 && m >= crossover {
        let serial: Option<fn(&[u8], &[f32], &mut [f32], usize, usize)> = match w.dtype {
            Dtype::Bf16Raw => Some(matvec_bf16_w_f32_a),
            Dtype::Q8_0Raw => Some(matvec_q8_0_w_f32_a),
            Dtype::Q4_0Raw => Some(matvec_q4_0_w_f32_a),
            Dtype::Q5_0Raw => Some(matvec_q5_0_w_f32_a),
            Dtype::Q4_1Raw => Some(matvec_q4_1_w_f32_a),
            Dtype::Q5_1Raw => Some(matvec_q5_1_w_f32_a),
            Dtype::Q2_KRaw => Some(matvec_q2_k_w_f32_a),
            Dtype::Q8_KRaw => Some(matvec_q8_k_w_f32_a),
            Dtype::Q3_KRaw => Some(matvec_q3_k_w_f32_a),
            Dtype::Q4_KRaw => Some(matvec_q4_k_w_f32_a),
            Dtype::Q5_KRaw => Some(matvec_q5_k_w_f32_a),
            Dtype::Q6_KRaw => Some(matvec_q6_k_w_f32_a),
            Dtype::IQ4_XSRaw => Some(matvec_iq4_xs_w_f32_a),
            Dtype::IQ4_NLRaw => Some(matvec_iq4_nl_w_f32_a),
            Dtype::IQ3_SRaw => Some(matvec_iq3_s_w_f32_a),
            Dtype::IQ3_XXSRaw => Some(matvec_iq3_xxs_w_f32_a),
            Dtype::IQ2_XXSRaw => Some(matvec_iq2_xxs_w_f32_a),
            Dtype::IQ2_XSRaw => Some(matvec_iq2_xs_w_f32_a),
            Dtype::IQ2_SRaw => Some(matvec_iq2_s_w_f32_a),
            Dtype::IQ1_SRaw => Some(matvec_iq1_s_w_f32_a),
            Dtype::IQ1_MRaw => Some(matvec_iq1_m_w_f32_a),
            Dtype::Nvfp4Raw => Some(nvfp4::matvec_nvfp4_w_f32_a),
            Dtype::Mxfp4Raw => Some(mxfp::matvec_mxfp4_w_f32_a),
            Dtype::Mxfp6Raw => Some(mxfp::matvec_mxfp6_w_f32_a),
            Dtype::Mxfp8Raw => Some(mxfp::matvec_mxfp8_w_f32_a),
            Dtype::PQ2_0Raw => Some(matvec_pq2_0_w_f32_a),
            // Decode/single-activation path stays on the exact
            // reference kernel: measured, the fastdot 4-accumulator
            // variant gives NO decode speedup (decode is memory-
            // bandwidth bound — each weight is read once per token and
            // the FMA hides behind the DRAM wait), so exactness wins.
            // fastdot's benefit is prefill-only (weights reused across
            // a chunk → compute-bound → ILP helps).
            Dtype::PTQ1_0Raw => Some(matvec_ptq1_0_w_f32_a),
            _ => None,
        };
        if let Some(serial_fn) = serial {
            use rayon::iter::{IndexedParallelIterator, ParallelIterator};
            use rayon::slice::ParallelSliceMut;
            let row_bytes = w.dtype.byte_size(k as u64) as usize;
            let w_bytes = as_bytes(w);
            let n_threads = rayon::current_num_threads().max(1);
            let chunk_rows = ((m + 4 * n_threads - 1) / (4 * n_threads)).max(64).min(m);
            out.par_chunks_mut(chunk_rows)
                .enumerate()
                .for_each(|(ci, oc)| {
                    let r0 = ci * chunk_rows;
                    let nr = oc.len();
                    serial_fn(&w_bytes[r0 * row_bytes..(r0 + nr) * row_bytes], x, oc, nr, k);
                });
            return;
        }
    }
    matvec_tensor_serial(w, x, out, m, k)
}

/// Batched tensor matvec: `out[t][i] = W[i] · xs[t]` for all
/// `t in 0..n_rows`. PTQ1_0 routes to the decode-once batched kernel
/// ([`matvec_ptq1_0_w_f32_a_batched`] — bitwise-equal to per-row
/// calls with ~n_rows× less trit extraction and weight traffic);
/// every other dtype falls back to per-row [`matvec_tensor`] calls,
/// which is exactly the pre-batched behavior.
pub fn matvec_tensor_batched(
    w: &Tensor,
    xs: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n_rows: usize,
) {
    if w.dtype == Dtype::PTQ1_0Raw {
        let w_bytes = as_bytes(w);
        matvec_ptq1_0_w_f32_a_batched(w_bytes, xs, out, m, k, n_rows);
        return;
    }
    for t in 0..n_rows {
        matvec_tensor(
            w,
            &xs[t * k..(t + 1) * k],
            &mut out[t * m..(t + 1) * m],
            m,
            k,
        );
    }
}

/// Fully-serial tensor matvec — never enters the rayon pool. For
/// callers that already run INSIDE a rayon task (e.g. the MoE q★
/// split-exec CPU branch) where nested parallelism would oversubscribe
/// against the concurrent GPU branch.
pub fn matvec_tensor_serial(w: &Tensor, x: &[f32], out: &mut [f32], m: usize, k: usize) {
    match w.dtype {
        Dtype::F16 => matvec_f16_w_f32_a_serial(as_slice_f16(w), x, out, m, k),
        Dtype::F32 => matvec_f32_serial(as_slice_f32(w), x, out, m, k),
        Dtype::Bf16Raw => matvec_bf16_w_f32_a(as_bytes(w), x, out, m, k),
        Dtype::Q8_0Raw => matvec_q8_0_w_f32_a(as_bytes(w), x, out, m, k),
        Dtype::Q4_0Raw => matvec_q4_0_w_f32_a(as_bytes(w), x, out, m, k),
        Dtype::Q5_0Raw => matvec_q5_0_w_f32_a(as_bytes(w), x, out, m, k),
        Dtype::Q4_1Raw => matvec_q4_1_w_f32_a(as_bytes(w), x, out, m, k),
        Dtype::Q5_1Raw => matvec_q5_1_w_f32_a(as_bytes(w), x, out, m, k),
        Dtype::Q2_KRaw => matvec_q2_k_w_f32_a(as_bytes(w), x, out, m, k),
        Dtype::Q8_KRaw => matvec_q8_k_w_f32_a(as_bytes(w), x, out, m, k),
        Dtype::Q3_KRaw => matvec_q3_k_w_f32_a(as_bytes(w), x, out, m, k),
        Dtype::Q4_KRaw => matvec_q4_k_w_f32_a(as_bytes(w), x, out, m, k),
        Dtype::Q5_KRaw => matvec_q5_k_w_f32_a(as_bytes(w), x, out, m, k),
        Dtype::Q6_KRaw => matvec_q6_k_w_f32_a(as_bytes(w), x, out, m, k),
        Dtype::IQ4_XSRaw => matvec_iq4_xs_w_f32_a(as_bytes(w), x, out, m, k),
        Dtype::IQ4_NLRaw => matvec_iq4_nl_w_f32_a(as_bytes(w), x, out, m, k),
        Dtype::IQ3_SRaw => matvec_iq3_s_w_f32_a(as_bytes(w), x, out, m, k),
        Dtype::IQ3_XXSRaw => matvec_iq3_xxs_w_f32_a(as_bytes(w), x, out, m, k),
        Dtype::IQ2_XXSRaw => matvec_iq2_xxs_w_f32_a(as_bytes(w), x, out, m, k),
        Dtype::IQ2_XSRaw => matvec_iq2_xs_w_f32_a(as_bytes(w), x, out, m, k),
        Dtype::IQ2_SRaw => matvec_iq2_s_w_f32_a(as_bytes(w), x, out, m, k),
        Dtype::IQ1_SRaw => matvec_iq1_s_w_f32_a(as_bytes(w), x, out, m, k),
        Dtype::IQ1_MRaw => matvec_iq1_m_w_f32_a(as_bytes(w), x, out, m, k),
        Dtype::Nvfp4Raw => nvfp4::matvec_nvfp4_w_f32_a(as_bytes(w), x, out, m, k),
        Dtype::Mxfp4Raw => mxfp::matvec_mxfp4_w_f32_a(as_bytes(w), x, out, m, k),
        Dtype::Mxfp6Raw => mxfp::matvec_mxfp6_w_f32_a(as_bytes(w), x, out, m, k),
        Dtype::Mxfp8Raw => mxfp::matvec_mxfp8_w_f32_a(as_bytes(w), x, out, m, k),
        Dtype::PQ2_0Raw => matvec_pq2_0_w_f32_a(as_bytes(w), x, out, m, k),
        Dtype::PTQ1_0Raw => matvec_ptq1_0_w_f32_a(as_bytes(w), x, out, m, k),
        // MLX affine packed blob — serial (nested-rayon-safe) variant.
        Dtype::MlxAffineRaw => {
            mlx_affine::matvec_mlx_affine_blob_w_f32_a(as_bytes(w), x, out, m, k)
        }
        other => panic!("matvec_tensor: unsupported dtype {other:?}"),
    }
}

#[cfg(test)]
mod matvec_parallel_parity {
    use super::*;
    use rustllama_tensor::{Device, Storage};

    /// Rows past the default parallel gate (M ≥ 256) so the parallel
    /// path actually engages.
    const M: usize = 512;

    fn quant_tensor(dtype: Dtype, bytes: Vec<u8>, k: usize) -> Tensor {
        Tensor {
            device: Device::Cpu,
            dtype,
            shape: vec![M as u64, k as u64],
            strides: vec![k as i64, 1],
            storage: Storage::CpuOwned(bytes.into()),
            name: "parity".into(),
        }
    }

    fn assert_bits_eq(par: &[f32], ser: &[f32]) {
        for (i, (p, s)) in par.iter().zip(ser.iter()).enumerate() {
            assert_eq!(
                p.to_bits(),
                s.to_bits(),
                "row {i}: parallel {p} vs serial {s}"
            );
        }
    }

    /// The auto-dispatched parallel F32 path must be bit-identical to
    /// the serial kernel: only the M axis splits, per-row accumulation
    /// is the same code.
    #[test]
    fn f32_parallel_bit_identical() {
        let k = 64usize;
        let w: Vec<f32> = (0..M * k)
            .map(|i| ((i * 37 + 11) % 251) as f32 * 0.013 - 1.6)
            .collect();
        let x: Vec<f32> = (0..k).map(|i| ((i * 53 + 7) % 97) as f32 * 0.021 - 1.0).collect();
        let mut par = vec![0.0f32; M];
        let mut ser = vec![0.0f32; M];
        matvec_f32(&w, &x, &mut par, M, k);
        matvec_f32_serial(&w, &x, &mut ser, M, k);
        assert_bits_eq(&par, &ser);
    }

    /// Q8_0 through the tensor dispatcher: the byte-level row-parallel
    /// wrapper must match `matvec_tensor_serial` bit for bit.
    #[test]
    fn q8_0_tensor_parallel_bit_identical() {
        let k = 64usize; // 2 blocks per row
        let row_bytes = Dtype::Q8_0Raw.byte_size(k as u64) as usize;
        let mut bytes = vec![0u8; M * row_bytes];
        for (r, row) in bytes.chunks_mut(row_bytes).enumerate() {
            for (b, blk) in row.chunks_mut(34).enumerate() {
                let d = f16::from_f32(0.01 + 0.005 * ((r + b) % 7) as f32);
                blk[..2].copy_from_slice(&d.to_le_bytes());
                for (j, q) in blk[2..].iter_mut().enumerate() {
                    *q = ((r * 31 + b * 17 + j * 7) % 256) as u8;
                }
            }
        }
        let w = quant_tensor(Dtype::Q8_0Raw, bytes, k);
        let x: Vec<f32> = (0..k).map(|i| ((i * 29 + 3) % 83) as f32 * 0.017 - 0.7).collect();
        let mut par = vec![0.0f32; M];
        let mut ser = vec![0.0f32; M];
        matvec_tensor(&w, &x, &mut par, M, k);
        matvec_tensor_serial(&w, &x, &mut ser, M, k);
        assert_bits_eq(&par, &ser);
    }

    /// PQ2_0 (PrismML ternary, 34-byte group-128 blocks, f16 d at
    /// offset 0): arbitrary code bytes are valid; stamp a sane d.
    #[test]
    fn pq2_0_tensor_parallel_bit_identical() {
        let k = 256usize; // 2 blocks per row
        let row_bytes = Dtype::PQ2_0Raw.byte_size(k as u64) as usize;
        assert_eq!(row_bytes, 68);
        let mut bytes = vec![0u8; M * row_bytes];
        for (bi, blk) in bytes.chunks_mut(34).enumerate() {
            let d = f16::from_f32(0.006 + 0.002 * (bi % 5) as f32);
            blk[..2].copy_from_slice(&d.to_le_bytes());
            for (j, q) in blk[2..].iter_mut().enumerate() {
                *q = ((bi * 37 + j * 11 + 3) % 256) as u8;
            }
        }
        let w = quant_tensor(Dtype::PQ2_0Raw, bytes, k);
        let x: Vec<f32> = (0..k).map(|i| ((i * 31 + 7) % 97) as f32 * 0.015 - 0.7).collect();
        let mut par = vec![0.0f32; M];
        let mut ser = vec![0.0f32; M];
        matvec_tensor(&w, &x, &mut par, M, k);
        matvec_tensor_serial(&w, &x, &mut ser, M, k);
        assert_bits_eq(&par, &ser);
    }

    /// PTQ1_0 (PrismML ternary, 28-byte group-128 blocks, f16 d at
    /// offset 26): arbitrary qs/qh bytes decode to valid trits.
    #[test]
    fn ptq1_0_tensor_parallel_bit_identical() {
        let k = 256usize; // 2 blocks per row
        let row_bytes = Dtype::PTQ1_0Raw.byte_size(k as u64) as usize;
        assert_eq!(row_bytes, 56);
        let mut bytes = vec![0u8; M * row_bytes];
        for (bi, blk) in bytes.chunks_mut(28).enumerate() {
            for (j, q) in blk[..26].iter_mut().enumerate() {
                *q = ((bi * 53 + j * 7 + 1) % 256) as u8;
            }
            let d = f16::from_f32(0.005 + 0.003 * (bi % 4) as f32);
            blk[26..28].copy_from_slice(&d.to_le_bytes());
        }
        let w = quant_tensor(Dtype::PTQ1_0Raw, bytes, k);
        let x: Vec<f32> = (0..k).map(|i| ((i * 43 + 5) % 89) as f32 * 0.017 - 0.8).collect();
        let mut par = vec![0.0f32; M];
        let mut ser = vec![0.0f32; M];
        matvec_tensor(&w, &x, &mut par, M, k);
        matvec_tensor_serial(&w, &x, &mut ser, M, k);
        assert_bits_eq(&par, &ser);
    }

    /// The batched PTQ1_0 kernel's contract is BITWISE equality with
    /// N independent single-row calls — that is what lets the hybrid
    /// prefill hoist projections into batched calls without changing
    /// model output by a single bit. Covers: n=1 passthrough, small
    /// serial m, large parallel m (crosses the rayon gate), several
    /// n_rows, and multi-block k.
    #[test]
    fn ptq1_0_batched_bit_identical_to_single_calls() {
        for (m, k, n_rows) in [
            (7usize, 128usize, 3usize),
            (33, 256, 5),
            (300, 384, 4), // m past the parallel crossover
            (12, 512, 1),  // n=1 passthrough arm
            (21, 256, 8),  // exactly one 8-token tile
            (21, 256, 9),  // tile + 1 tail token
            (40, 128, 17), // two tiles + 1 tail
            (2, 128, 4),   // minimal row pair, one token quad
            (5, 256, 4),   // row quad + odd-row tail
            (6, 128, 13),  // quad + pair; triples + tail
            (9, 128, 3),   // 2 quads + 1 row; one token triple
            (8, 256, 6),   // quads only; 2 triples
            (11, 128, 7),  // quads + pair + single; triples + tail
        ] {
            let row_bytes = (k / 128) * 28;
            let mut bytes = vec![0u8; m * row_bytes];
            for (bi, blk) in bytes.chunks_mut(28).enumerate() {
                for (j, q) in blk[..26].iter_mut().enumerate() {
                    *q = ((bi * 31 + j * 11 + 3) % 256) as u8;
                }
                let d = f16::from_f32(0.004 + 0.002 * (bi % 5) as f32);
                blk[26..28].copy_from_slice(&d.to_le_bytes());
            }
            let xs: Vec<f32> = (0..n_rows * k)
                .map(|i| ((i * 37 + 11) % 97) as f32 * 0.013 - 0.6)
                .collect();
            let mut batched = vec![0.0f32; n_rows * m];
            // Force BITWISE mode explicitly: the public
            // `matvec_ptq1_0_w_f32_a_batched` now defaults to fastdot
            // (best-settings bake-in), which is tolerance-gated not
            // bit-identical. The bit-identity contract lives on the
            // bitwise path; fastdot has its own tolerance test.
            matvec_ptq1_0_w_f32_a_batched_with_mode(&bytes, &xs, &mut batched, m, k, n_rows, false);
            let mut single = vec![0.0f32; m];
            for t in 0..n_rows {
                matvec_ptq1_0_w_f32_a(&bytes, &xs[t * k..(t + 1) * k], &mut single, m, k);
                assert_bits_eq(&batched[t * m..(t + 1) * m], &single);
            }
        }
    }

    /// Fastdot mode is NOT bitwise (block sums kept in vector lanes,
    /// d folded as a broadcast-fmadd, one hsum per dot) — the gates
    /// are (a) tolerance against the reference kernel and (b) exact
    /// determinism across repeated runs.
    /// Fast single-row (decode-path) matvec: tolerance vs the bitwise
    /// reference kernel + exact determinism. Reassociated 4-acc sum,
    /// so not bitwise — same gate philosophy as batched fastdot.
    #[test]
    fn ptq1_0_row_fast_tolerance_and_determinism() {
        for (m, k) in [(64usize, 128usize), (300, 5120), (128, 384)] {
            let row_bytes = (k / 128) * 28;
            let mut bytes = vec![0u8; m * row_bytes];
            for (bi, blk) in bytes.chunks_mut(28).enumerate() {
                for (j, q) in blk[..26].iter_mut().enumerate() {
                    *q = ((bi * 23 + j * 7 + 1) % 256) as u8;
                }
                let d = half::f16::from_f32(0.003 + 0.0015 * (bi % 6) as f32);
                blk[26..28].copy_from_slice(&d.to_le_bytes());
            }
            let x: Vec<f32> = (0..k).map(|i| ((i * 31 + 5) % 89) as f32 * 0.012 - 0.5).collect();
            let mut reference = vec![0.0f32; m];
            matvec_ptq1_0_w_f32_a(&bytes, &x, &mut reference, m, k);
            let mut fast = vec![0.0f32; m];
            matvec_ptq1_0_w_f32_a_fast(&bytes, &x, &mut fast, m, k);
            let scale = reference.iter().fold(0f32, |a, &v| a.max(v.abs())).max(1e-3);
            for (i, (&r, &f)) in reference.iter().zip(fast.iter()).enumerate() {
                assert!((r - f).abs() <= 1e-3 * scale, "row-fast m={m} k={k} row {i}: ref={r} fast={f}");
            }
            let mut fast2 = vec![0.0f32; m];
            matvec_ptq1_0_w_f32_a_fast(&bytes, &x, &mut fast2, m, k);
            assert_bits_eq(&fast, &fast2);
        }
    }

    #[test]
    fn ptq1_0_batched_fastdot_tolerance_and_determinism() {
        for (m, k, n_rows) in [
            (7usize, 128usize, 3usize),
            (33, 256, 6),
            (300, 384, 4),
            (9, 512, 7),
            (2, 128, 3),
            (5, 256, 13),
            (3, 128, 4),   // one 3-row fastfold group, one quad
            (7, 256, 9),   // 3+3+1 rows; quads + tail
            (6, 384, 8),   // two 3-row groups; quads only
        ] {
            let row_bytes = (k / 128) * 28;
            let mut bytes = vec![0u8; m * row_bytes];
            for (bi, blk) in bytes.chunks_mut(28).enumerate() {
                for (j, q) in blk[..26].iter_mut().enumerate() {
                    *q = ((bi * 29 + j * 13 + 5) % 256) as u8;
                }
                let d = f16::from_f32(0.003 + 0.002 * (bi % 7) as f32);
                blk[26..28].copy_from_slice(&d.to_le_bytes());
            }
            let xs: Vec<f32> = (0..n_rows * k)
                .map(|i| ((i * 41 + 17) % 101) as f32 * 0.011 - 0.55)
                .collect();
            let mut reference = vec![0.0f32; n_rows * m];
            matvec_ptq1_0_w_f32_a_batched_with_mode(
                &bytes, &xs, &mut reference, m, k, n_rows, false,
            );
            let mut fast = vec![0.0f32; n_rows * m];
            matvec_ptq1_0_w_f32_a_batched_with_mode(
                &bytes, &xs, &mut fast, m, k, n_rows, true,
            );
            let scale = reference
                .iter()
                .fold(0f32, |a, &v| a.max(v.abs()))
                .max(1e-3);
            for (i, (&r, &f)) in reference.iter().zip(fast.iter()).enumerate() {
                assert!(
                    (r - f).abs() <= 1e-3 * scale,
                    "fastdot m={m} k={k} n={n_rows} elem {i}: ref={r} fast={f}"
                );
            }
            let mut fast2 = vec![0.0f32; n_rows * m];
            matvec_ptq1_0_w_f32_a_batched_with_mode(
                &bytes, &xs, &mut fast2, m, k, n_rows, true,
            );
            assert_bits_eq(&fast, &fast2);
        }
    }

    /// Q6_K (the LM-head dtype on the target model): 210-byte blocks
    /// (ql 128 + qh 64 + scales 16 + f16 d), k = 256. Quants/scales
    /// are a deterministic byte pattern; the trailing `d` is a normal
    /// f16 per block.
    #[test]
    fn q6_k_tensor_parallel_bit_identical() {
        let k = 256usize; // 1 block per row
        let row_bytes = Dtype::Q6_KRaw.byte_size(k as u64) as usize;
        assert_eq!(row_bytes, 210);
        let mut bytes = vec![0u8; M * row_bytes];
        for (r, blk) in bytes.chunks_mut(row_bytes).enumerate() {
            for (j, q) in blk[..208].iter_mut().enumerate() {
                *q = ((r * 13 + j * 5 + 1) % 256) as u8;
            }
            let d = f16::from_f32(0.008 + 0.003 * (r % 5) as f32);
            blk[208..210].copy_from_slice(&d.to_le_bytes());
        }
        let w = quant_tensor(Dtype::Q6_KRaw, bytes, k);
        let x: Vec<f32> = (0..k).map(|i| ((i * 41 + 13) % 89) as f32 * 0.019 - 0.8).collect();
        let mut par = vec![0.0f32; M];
        let mut ser = vec![0.0f32; M];
        matvec_tensor(&w, &x, &mut par, M, k);
        matvec_tensor_serial(&w, &x, &mut ser, M, k);
        assert_bits_eq(&par, &ser);
    }
}

#[cfg(test)]
mod prism_ternary_parity {
    use super::*;

    /// Deterministic pseudo-random block bytes with a sane f16 scale
    /// stamped at `d_off`. Arbitrary code bytes are valid for both
    /// Prism codecs (the base-3 extraction always yields trits 0..=2
    /// and the 2-bit decode covers all four codes).
    fn gen_blocks(n_blocks: usize, block_bytes: usize, d_off: usize, seed: usize) -> Vec<u8> {
        let mut v = vec![0u8; n_blocks * block_bytes];
        for (bi, blk) in v.chunks_mut(block_bytes).enumerate() {
            for (j, b) in blk.iter_mut().enumerate() {
                *b = ((seed + bi * 131 + j * 17 + 5) % 256) as u8;
            }
            let d = f16::from_f32(0.004 + 0.0025 * ((bi + seed) % 6) as f32);
            blk[d_off..d_off + 2].copy_from_slice(&d.to_le_bytes());
        }
        v
    }

    fn gen_x(k: usize) -> Vec<f32> {
        (0..k).map(|i| ((i * 29 + 13) % 101) as f32 * 0.019 - 0.9).collect()
    }

    /// matvec == dequant-to-f32 then f32 matvec, for both codecs.
    #[test]
    fn matvec_matches_dequant_reference() {
        let (m, k) = (7usize, 384usize); // 3 blocks per row
        for (name, block_bytes, d_off) in [("pq2_0", 34usize, 0usize), ("ptq1_0", 28, 26)] {
            let w = gen_blocks(m * k / 128, block_bytes, d_off, 42);
            let x = gen_x(k);
            let mut dq = vec![0f32; m * k];
            let mut fused = vec![0f32; m];
            match name {
                "pq2_0" => {
                    rustllama_gguf::dequant::dequant_pq2_0(&w, &mut dq);
                    matvec_pq2_0_w_f32_a(&w, &x, &mut fused, m, k);
                }
                _ => {
                    rustllama_gguf::dequant::dequant_ptq1_0(&w, &mut dq);
                    matvec_ptq1_0_w_f32_a(&w, &x, &mut fused, m, k);
                }
            }
            let mut reference = vec![0f32; m];
            matvec_f32_serial(&dq, &x, &mut reference, m, k);
            for i in 0..m {
                let rel = (fused[i] - reference[i]).abs() / (reference[i].abs() + 1e-4);
                assert!(
                    rel < 1e-4,
                    "{name} row {i}: fused {} vs reference {}",
                    fused[i],
                    reference[i]
                );
            }
        }
    }

    /// Embed lookup == whole-table dequant row slices.
    #[test]
    fn embed_lookup_matches_dequant() {
        let (vocab, d) = (11usize, 256usize);
        let ids = [0i32, 4, 10, 4];
        for (name, block_bytes, d_off) in [("pq2_0", 34usize, 0usize), ("ptq1_0", 28, 26)] {
            let table = gen_blocks(vocab * d / 128, block_bytes, d_off, 7);
            let mut full = vec![0f32; vocab * d];
            let mut looked = vec![0f32; ids.len() * d];
            match name {
                "pq2_0" => {
                    rustllama_gguf::dequant::dequant_pq2_0(&table, &mut full);
                    embed_lookup_pq2_0(&table, &ids, &mut looked, d);
                }
                _ => {
                    rustllama_gguf::dequant::dequant_ptq1_0(&table, &mut full);
                    embed_lookup_ptq1_0(&table, &ids, &mut looked, d);
                }
            }
            for (i, &id) in ids.iter().enumerate() {
                let want = &full[(id as usize) * d..(id as usize + 1) * d];
                let got = &looked[i * d..(i + 1) * d];
                assert_eq!(got, want, "{name} row {id}");
            }
        }
    }

    /// The SIMD dispatch (AVX2 where available) must agree with the
    /// scalar kernel to FP-reduction-order tolerance on random-byte
    /// blocks across several shapes, including single-block rows and
    /// larger multi-block rows.
    #[test]
    fn ptq1_0_simd_matches_scalar() {
        for (m, k, seed) in [(3usize, 128usize, 1usize), (5, 640, 9), (2, 5120, 23)] {
            let w = gen_blocks(m * k / 128, 28, 26, seed);
            let x = gen_x(k);
            let mut dispatched = vec![0f32; m];
            let mut scalar = vec![0f32; m];
            matvec_ptq1_0_w_f32_a(&w, &x, &mut dispatched, m, k);
            matvec_ptq1_0_w_f32_a_scalar(&w, &x, &mut scalar, m, k);
            for i in 0..m {
                let rel = (dispatched[i] - scalar[i]).abs() / (scalar[i].abs() + 1e-4);
                assert!(
                    rel < 1e-4,
                    "m={m} k={k} row {i}: dispatched {} vs scalar {}",
                    dispatched[i],
                    scalar[i]
                );
            }
        }
    }

    /// Golden PTQ1_0 round-trip: pack a known ternary pattern with the
    /// ceiling base-3 encoder (ported from the PrismML fork's
    /// quantize_row_ptq1_0_ref) and assert the decode-side extraction
    /// recovers every trit exactly. This is the test that would fail
    /// if the stage geometry, digit order, or extraction trick ever
    /// drifted from the reference.
    #[test]
    fn ptq1_0_golden_round_trip() {
        const QK: usize = 128;
        let src: Vec<f32> = (0..QK).map(|i| ((i * 7 + 2) % 3) as f32 - 1.0).collect();
        // --- reference encoder (fork ggml-quants.c:2205-2253) ---
        let mut blk = [0u8; 28];
        let d = 1.0f32;
        blk[26..28].copy_from_slice(&f16::from_f32(d).to_le_bytes());
        let stages = [32usize, 16, 8];
        let mut xp = 0usize; // encoder's advancing x pointer
        let mut j = 0usize;
        for &c in &stages {
            while j + c <= 24 {
                for mm in 0..c {
                    let mut q = 0u16;
                    for n in 0..5 {
                        let xi = (src[xp + mm + n * c].round() as i32 + 1) as u16;
                        q = q * 3 + xi;
                    }
                    blk[j + mm] = ((q * 256 + 242) / 243) as u8;
                }
                xp += 5 * c;
                j += c;
            }
        }
        for h in 0..2usize {
            let mut q = 0u16;
            for mm in 0..4 {
                let xi = (src[xp + h + mm * 2].round() as i32 + 1) as u16;
                q = q * 3 + xi;
            }
            q *= 3; // shift first value to the most significant trit
            blk[24 + h] = ((q * 256 + 242) / 243) as u8;
        }
        // --- decode and compare ---
        let mut out = vec![0f32; QK];
        rustllama_gguf::dequant::dequant_ptq1_0(&blk, &mut out);
        assert_eq!(out, src, "PTQ1_0 ceiling-pack round trip");
        // matvec against the same block must agree with the dot product.
        let x = gen_x(QK);
        let mut mv = vec![0f32; 1];
        matvec_ptq1_0_w_f32_a(&blk, &x, &mut mv, 1, QK);
        let dot: f32 = src.iter().zip(&x).map(|(a, b)| a * b).sum();
        assert!((mv[0] - dot).abs() < 1e-4, "matvec {} vs dot {}", mv[0], dot);
    }

    /// Golden PQ2_0: hand-pack one block covering all four codes
    /// (including the +2 level) and check exact decode.
    #[test]
    fn pq2_0_golden_all_codes() {
        const QK: usize = 128;
        let mut blk = [0u8; 34];
        let d = 0.5f32;
        blk[..2].copy_from_slice(&f16::from_f32(d).to_le_bytes());
        // Element j gets code j % 4 → values (-1, 0, 1, 2) * d.
        for j in 0..QK {
            blk[2 + j / 4] |= ((j % 4) as u8) << ((j % 4) * 2);
        }
        // The line above packs code j%4 at slot j%4 of byte j/4 —
        // i.e. every byte is 0b11100100 (codes 0,1,2,3 in slots
        // 0..3), so element j decodes to ((j % 4) - 1) * d.
        let mut out = vec![0f32; QK];
        rustllama_gguf::dequant::dequant_pq2_0(&blk, &mut out);
        for j in 0..QK {
            let want = ((j % 4) as f32 - 1.0) * d;
            assert_eq!(out[j], want, "element {j}");
        }
    }
}

/// PQ2_0 (PrismML Bonsai) weight matvec with on-the-fly dequant.
/// Scalar for v1 (2-bit shift+mask decode is cheap; an AVX2 path is
/// a planned follow-up once the 27B end-to-end profile says where
/// the time goes).
///
/// Block layout (34 bytes per 128 weights):
///   { d: f16, qs: [u8; 32] }  — little-endian 2-bit codes,
///   weight = d * (code - 1); code 3 (+2d) is part of the codec.
///
/// Requires `k % 128 == 0`.
pub fn matvec_pq2_0_w_f32_a(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    const BLOCK_BYTES: usize = 34;
    const QK: usize = 128;
    debug_assert_eq!(k % QK, 0, "PQ2_0 matvec requires k % 128 == 0");
    let blocks_per_row = k / QK;
    debug_assert_eq!(w_bytes.len(), m * blocks_per_row * BLOCK_BYTES);
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(out.len(), m);

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d = f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32();
            let qs = &w_bytes[off + 2..off + 2 + 32];
            let xb = &x[b * QK..(b + 1) * QK];
            // Accumulate in the integer-code domain, scale once per
            // block: sum(d*(q-1)*x) = d * (sum(q*x) - sum(x)).
            // Doing it directly keeps it simple and exact in f32.
            let mut sum = 0.0f32;
            for j in 0..QK {
                let q = ((qs[j / 4] >> ((j % 4) * 2)) & 0x3) as i32 - 1;
                sum += q as f32 * xb[j];
            }
            acc += d * sum;
        }
        out[i] = acc;
    }
}

/// PTQ1_0 (PrismML Bonsai) weight matvec with on-the-fly dequant.
/// Dispatches to AVX2 (16-bit multiply trit extraction, 16 weights
/// per SIMD step) → scalar.
///
/// Block layout (28 bytes per 128 weights):
///   { qs: [u8; 24], qh: [u8; 2], d: f16 }
/// qs packs 5 trits/byte in ceiling base-3 fixed point, chunk-staged
/// {32,16,8} (resolves to a 16-byte then an 8-byte chunk); qh packs
/// 4 trits/byte with the first trit pre-shifted to the top digit.
/// Element order and extraction mirror `dequant_ptq1_0` exactly.
///
/// Requires `k % 128 == 0`.
pub fn matvec_ptq1_0_w_f32_a(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime feature detection above.
            unsafe { matvec_ptq1_0_w_f32_a_avx2(w_bytes, x, out, m, k) };
            return;
        }
    }
    matvec_ptq1_0_w_f32_a_scalar(w_bytes, x, out, m, k);
}

/// Batched PTQ1_0 matvec over `n_rows` activation rows:
/// `out[t][i] = W[i] · xs[t]` with `xs` row-major `[n_rows, k]` and
/// `out` row-major `[n_rows, m]`.
///
/// This is the CPU prefill kernel the per-token fallback lacked: the
/// per-token loop re-streamed every weight byte AND re-ran the trit
/// extraction once per token. Here each weight row is decoded ONCE
/// into an f32 trit buffer (±1.0/0.0 — exactly representable, decoded
/// in the same element order the single-row kernels walk), then all
/// `n_rows` dot products replay the single kernel's exact FP
/// accumulation grouping against that buffer. Extraction cost and
/// weight DRAM traffic drop by ~n_rows×.
///
/// BITWISE CONTRACT: `out[t]` equals `matvec_ptq1_0_w_f32_a(w, xs[t],
/// ..)` exactly, per element. The extraction is integer-side, so
/// hoisting it cannot change FP results; the replay preserves the
/// AVX2 kernel's lane grouping (five 16-lane stages as lo/hi fmadd
/// pairs, five 8-lane stages, the same horizontal-sum shuffle order,
/// eight scalar tail adds, then `acc += d·sum` per block) and the
/// scalar kernel's sequential order on the non-AVX2 path. Pinned by
/// `ptq1_0_batched_bit_identical_to_single_calls`.
///
/// Parallelism: rayon over weight-row chunks (same crossover lever as
/// [`matvec_tensor`]). Workers write a transposed `[m, n_rows]`
/// scratch (disjoint per-row chunks), scattered to the caller's
/// `[n_rows, m]` layout in one serial pass at the end.
std::thread_local! {
    /// Per-thread single-row decode scratch for the fast decode
    /// matvec (grow-only i8 trits + f32 scales).
    static PTQ1_0_ROW_SCRATCH: std::cell::RefCell<(Vec<i8>, Vec<f32>)> =
        const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
}

/// Fastdot single-row matvec (`RUSTLLAMA_TERNARY_FASTDOT=1`): decode
/// each weight row to i8 once (SIMD), then dot with the 4-accumulator
/// fast kernel. This is the DECODE path's fast variant — decode is
/// single-token (one activation vector, m output rows), so it can't
/// use the batched kernel's token tiling; the win comes purely from
/// breaking the per-block FMA latency chain. Reassociated sums →
/// tolerance-gated vs `matvec_ptq1_0_w_f32_a`, not bitwise (which is
/// why it is gated, and why the bitwise batched tests still compare
/// against the untouched reference kernel).
pub fn matvec_ptq1_0_w_f32_a_fast(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            const BLOCK_BYTES: usize = 28;
            let blocks_per_row = k / 128;
            let row_bytes = blocks_per_row * BLOCK_BYTES;
            PTQ1_0_ROW_SCRATCH.with(|cell| {
                let (trits, dvals) = &mut *cell.borrow_mut();
                for i in 0..m {
                    decode_ptq1_0_row_trits_i8(
                        &w_bytes[i * row_bytes..(i + 1) * row_bytes],
                        blocks_per_row,
                        trits,
                        dvals,
                    );
                    // SAFETY: avx2+fma verified above.
                    out[i] = unsafe {
                        ptq1_0_row_dot_i8_avx2_fast(trits, dvals, x, blocks_per_row)
                    };
                }
            });
            return;
        }
    }
    // AArch64 NEON (baseline on every aarch64 CPU — no runtime detect):
    // the ARM decode fast path for the DGX Spark / Grace Blackwell target.
    #[cfg(target_arch = "aarch64")]
    {
        const BLOCK_BYTES: usize = 28;
        let blocks_per_row = k / 128;
        let row_bytes = blocks_per_row * BLOCK_BYTES;
        PTQ1_0_ROW_SCRATCH.with(|cell| {
            let (trits, dvals) = &mut *cell.borrow_mut();
            for i in 0..m {
                decode_ptq1_0_row_trits_i8(
                    &w_bytes[i * row_bytes..(i + 1) * row_bytes],
                    blocks_per_row,
                    trits,
                    dvals,
                );
                // SAFETY: NEON is always available on aarch64.
                out[i] =
                    unsafe { ptq1_0_row_dot_i8_neon_fast(trits, dvals, x, blocks_per_row) };
            }
        });
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    matvec_ptq1_0_w_f32_a_scalar(w_bytes, x, out, m, k);
}

pub fn matvec_ptq1_0_w_f32_a_batched(
    w_bytes: &[u8],
    xs: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n_rows: usize,
) {
    matvec_ptq1_0_w_f32_a_batched_with_mode(
        w_bytes,
        xs,
        out,
        m,
        k,
        n_rows,
        ternary_fastdot_enabled(),
    );
}

/// Reordered-accumulation ternary dot (per-128-block VECTOR
/// accumulator + broadcast d-fold, one horizontal sum per dot).
/// Removes the per-block scalar epilogue that bound the bitwise
/// replay at ~28-31 G mul-adds/s; measured coherent (text-identical
/// on the real 27B) and a real prefill win. **Default ON** (best
/// setting baked in per the productionization pass); it is only a
/// PREFILL/batched-path change and a no-op for non-ternary models.
/// NOT bitwise-equal to the reference kernel (tighter FP summation
/// order) — deterministic for fixed inputs, tolerance-gated against
/// the reference. Disable with `RUSTLLAMA_TERNARY_FASTDOT=0`.
fn ternary_fastdot_enabled() -> bool {
    use std::sync::OnceLock;
    static C: OnceLock<bool> = OnceLock::new();
    *C.get_or_init(|| {
        std::env::var("RUSTLLAMA_TERNARY_FASTDOT")
            .map(|v| !(v == "0" || v.eq_ignore_ascii_case("false")))
            .unwrap_or(true)
    })
}

pub(crate) fn matvec_ptq1_0_w_f32_a_batched_with_mode(
    w_bytes: &[u8],
    xs: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n_rows: usize,
    fast: bool,
) {
    const BLOCK_BYTES: usize = 28;
    const QK: usize = 128;
    debug_assert_eq!(k % QK, 0, "PTQ1_0 matvec requires k % 128 == 0");
    debug_assert_eq!(xs.len(), n_rows * k);
    debug_assert_eq!(out.len(), n_rows * m);
    debug_assert_eq!(w_bytes.len(), m * (k / QK) * BLOCK_BYTES);
    if n_rows == 0 || m == 0 {
        return;
    }
    if n_rows == 1 {
        matvec_ptq1_0_w_f32_a(w_bytes, xs, out, m, k);
        return;
    }
    let blocks_per_row = k / QK;
    let row_bytes = blocks_per_row * BLOCK_BYTES;

    PTQ1_0_BATCH_SCRATCH.with(|cell| {
        let out_t = &mut *cell.borrow_mut();
        if out_t.len() < m * n_rows {
            out_t.resize(m * n_rows, 0.0);
        }
        let out_t = &mut out_t[..m * n_rows];

        #[cfg(target_arch = "x86_64")]
        let use_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        #[cfg(not(target_arch = "x86_64"))]
        let use_avx2 = false;

        let body = |i0: usize, rows_out: &mut [f32], scr: &mut PtqRowScratch| {
            let rows_here = rows_out.len() / n_rows;
            let mut di = 0usize;
            // Register blocking ladder (all arms replay each
            // (row, token) pair in the single-kernel FP order —
            // bitwise-identical outputs; the blocking only changes
            // L2 reuse and chain count):
            //   4 rows x 3 tokens  — 12 acc + 4 trit vecs = 16 ymm,
            //                        (R + 4T)/(R*T) = 1.33 B/mul-add
            //   2 rows x 4 tokens  — row tail >= 2
            //   1 row  x 8 tokens  — last row
            #[cfg(target_arch = "x86_64")]
            if use_avx2 && fast {
                // Fastdot ladder: 3 rows x 4 tokens primary (d is
                // pre-folded into the trit vector per step, so each
                // (row, token) pair costs ONE register: 12 totals +
                // 3 scales + 1 temp = 16 ymm, 1.58 B/mul-add), then
                // 2 rows x 3 tokens, then 1 row x 6, then singles.
                while di + 3 <= rows_here {
                    let i = i0 + di;
                    for r in 0..3 {
                        decode_ptq1_0_row_trits_i8(
                            &w_bytes[(i + r) * row_bytes..(i + r + 1) * row_bytes],
                            blocks_per_row,
                            &mut scr.trits[r],
                            &mut scr.dvals[r],
                        );
                    }
                    let rows3 = &mut rows_out[di * n_rows..(di + 3) * n_rows];
                    let mut t = 0usize;
                    while t + 4 <= n_rows {
                        let mut outs = [0.0f32; 12];
                        // SAFETY: avx2+fma verified via use_avx2.
                        unsafe {
                            ptq1_0_rows3_dot_i8_avx2_t4_fastfold(
                                scr,
                                &xs[t * k..(t + 4) * k],
                                &mut outs,
                                blocks_per_row,
                                k,
                            );
                        }
                        for r in 0..3 {
                            rows3[r * n_rows + t..r * n_rows + t + 4]
                                .copy_from_slice(&outs[r * 4..(r + 1) * 4]);
                        }
                        t += 4;
                    }
                    while t < n_rows {
                        let x = &xs[t * k..(t + 1) * k];
                        for r in 0..3 {
                            // SAFETY: avx2+fma verified via use_avx2.
                            unsafe {
                                rows3[r * n_rows + t] = ptq1_0_row_dot_i8_avx2_fast(
                                    &scr.trits[r],
                                    &scr.dvals[r],
                                    x,
                                    blocks_per_row,
                                );
                            }
                        }
                        t += 1;
                    }
                    di += 3;
                }
                while di + 2 <= rows_here {
                    let i = i0 + di;
                    for r in 0..2 {
                        decode_ptq1_0_row_trits_i8(
                            &w_bytes[(i + r) * row_bytes..(i + r + 1) * row_bytes],
                            blocks_per_row,
                            &mut scr.trits[r],
                            &mut scr.dvals[r],
                        );
                    }
                    let (left, right) = rows_out[di * n_rows..].split_at_mut(n_rows);
                    let right = &mut right[..n_rows];
                    let mut t = 0usize;
                    while t + 3 <= n_rows {
                        let mut oa = [0.0f32; 3];
                        let mut ob = [0.0f32; 3];
                        // SAFETY: avx2+fma verified via use_avx2.
                        unsafe {
                            ptq1_0_rows2_dot_i8_avx2_t3_fast(
                                &scr.trits[0], &scr.dvals[0],
                                &scr.trits[1], &scr.dvals[1],
                                &xs[t * k..(t + 3) * k],
                                &mut oa, &mut ob,
                                blocks_per_row, k,
                            );
                        }
                        left[t..t + 3].copy_from_slice(&oa);
                        right[t..t + 3].copy_from_slice(&ob);
                        t += 3;
                    }
                    while t < n_rows {
                        let x = &xs[t * k..(t + 1) * k];
                        // SAFETY: avx2+fma verified via use_avx2.
                        unsafe {
                            left[t] = ptq1_0_row_dot_i8_avx2_fast(
                                &scr.trits[0], &scr.dvals[0], x, blocks_per_row,
                            );
                            right[t] = ptq1_0_row_dot_i8_avx2_fast(
                                &scr.trits[1], &scr.dvals[1], x, blocks_per_row,
                            );
                        }
                        t += 1;
                    }
                    di += 2;
                }
                while di < rows_here {
                    let i = i0 + di;
                    decode_ptq1_0_row_trits_i8(
                        &w_bytes[i * row_bytes..(i + 1) * row_bytes],
                        blocks_per_row,
                        &mut scr.trits[0],
                        &mut scr.dvals[0],
                    );
                    let row_out = &mut rows_out[di * n_rows..(di + 1) * n_rows];
                    let mut t = 0usize;
                    while t + 6 <= n_rows {
                        let mut o6 = [0.0f32; 6];
                        // SAFETY: avx2+fma verified via use_avx2.
                        unsafe {
                            ptq1_0_row_dot_i8_avx2_t6_fast(
                                &scr.trits[0], &scr.dvals[0],
                                &xs[t * k..(t + 6) * k],
                                &mut o6,
                                blocks_per_row, k,
                            );
                        }
                        row_out[t..t + 6].copy_from_slice(&o6);
                        t += 6;
                    }
                    while t < n_rows {
                        let x = &xs[t * k..(t + 1) * k];
                        // SAFETY: avx2+fma verified via use_avx2.
                        unsafe {
                            row_out[t] = ptq1_0_row_dot_i8_avx2_fast(
                                &scr.trits[0], &scr.dvals[0], x, blocks_per_row,
                            );
                        }
                        t += 1;
                    }
                    di += 1;
                }
                return;
            }
            #[cfg(target_arch = "x86_64")]
            if use_avx2 {
                while di + 4 <= rows_here {
                    for r in 0..4 {
                        let i = i0 + di + r;
                        decode_ptq1_0_row_trits_i8(
                            &w_bytes[i * row_bytes..(i + 1) * row_bytes],
                            blocks_per_row,
                            &mut scr.trits[r],
                            &mut scr.dvals[r],
                        );
                    }
                    let rows4 = &mut rows_out[di * n_rows..(di + 4) * n_rows];
                    let mut t = 0usize;
                    while t + 3 <= n_rows {
                        let mut outs = [0.0f32; 12];
                        // SAFETY: avx2+fma verified via use_avx2.
                        unsafe {
                            ptq1_0_rows4_dot_decoded_i8_avx2_t3(
                                scr,
                                &xs[t * k..(t + 3) * k],
                                &mut outs,
                                blocks_per_row,
                                k,
                            );
                        }
                        for r in 0..4 {
                            rows4[r * n_rows + t..r * n_rows + t + 3]
                                .copy_from_slice(&outs[r * 3..(r + 1) * 3]);
                        }
                        t += 3;
                    }
                    while t < n_rows {
                        let x = &xs[t * k..(t + 1) * k];
                        for r in 0..4 {
                            rows4[r * n_rows + t] = ptq1_0_row_dot_decoded_i8(
                                &scr.trits[r],
                                &scr.dvals[r],
                                x,
                                blocks_per_row,
                            );
                        }
                        t += 1;
                    }
                    di += 4;
                }
                while di + 2 <= rows_here {
                    let i = i0 + di;
                    for r in 0..2 {
                        decode_ptq1_0_row_trits_i8(
                            &w_bytes[(i + r) * row_bytes..(i + r + 1) * row_bytes],
                            blocks_per_row,
                            &mut scr.trits[r],
                            &mut scr.dvals[r],
                        );
                    }
                    let (left, right) = rows_out[di * n_rows..].split_at_mut(n_rows);
                    let right = &mut right[..n_rows];
                    let mut t = 0usize;
                    while t + 4 <= n_rows {
                        let (ta, rest) = scr.trits.split_at(1);
                        let (da, drest) = scr.dvals.split_at(1);
                        // SAFETY: avx2+fma verified via use_avx2.
                        unsafe {
                            ptq1_0_rows2_dot_decoded_i8_avx2_t4(
                                &ta[0],
                                &da[0],
                                &rest[0],
                                &drest[0],
                                &xs[t * k..(t + 4) * k],
                                &mut left[t..t + 4],
                                &mut right[t..t + 4],
                                blocks_per_row,
                                k,
                            );
                        }
                        t += 4;
                    }
                    while t < n_rows {
                        let x = &xs[t * k..(t + 1) * k];
                        left[t] = ptq1_0_row_dot_decoded_i8(
                            &scr.trits[0], &scr.dvals[0], x, blocks_per_row,
                        );
                        right[t] = ptq1_0_row_dot_decoded_i8(
                            &scr.trits[1], &scr.dvals[1], x, blocks_per_row,
                        );
                        t += 1;
                    }
                    di += 2;
                }
            }
            // Last row — and the whole loop on non-AVX2 hosts.
            while di < rows_here {
                let i = i0 + di;
                decode_ptq1_0_row_trits_i8(
                    &w_bytes[i * row_bytes..(i + 1) * row_bytes],
                    blocks_per_row,
                    &mut scr.trits[0],
                    &mut scr.dvals[0],
                );
                let row_out = &mut rows_out[di * n_rows..(di + 1) * n_rows];
                let mut t = 0usize;
                #[cfg(target_arch = "x86_64")]
                if use_avx2 {
                    while t + 8 <= n_rows {
                        // SAFETY: avx2+fma verified via use_avx2.
                        unsafe {
                            ptq1_0_row_dot_decoded_i8_avx2_t8(
                                &scr.trits[0],
                                &scr.dvals[0],
                                &xs[t * k..(t + 8) * k],
                                &mut row_out[t..t + 8],
                                blocks_per_row,
                                k,
                            );
                        }
                        t += 8;
                    }
                }
                while t < n_rows {
                    let x = &xs[t * k..(t + 1) * k];
                    // AArch64: route the FAST batched path through the NEON
                    // decoded dot (tolerance-gated, like the AVX2 fast
                    // ladder); keep the bitwise scalar dot for the non-fast
                    // path so `ptq1_0_batched_bit_identical_to_single_calls`
                    // still holds. Decode already happened once per row
                    // above, so this is the ARM prefill win.
                    #[cfg(target_arch = "aarch64")]
                    {
                        row_out[t] = if fast {
                            // SAFETY: NEON is always available on aarch64.
                            unsafe {
                                ptq1_0_row_dot_i8_neon_fast(
                                    &scr.trits[0], &scr.dvals[0], x, blocks_per_row,
                                )
                            }
                        } else {
                            ptq1_0_row_dot_decoded_i8(
                                &scr.trits[0], &scr.dvals[0], x, blocks_per_row,
                            )
                        };
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    {
                        row_out[t] = ptq1_0_row_dot_decoded_i8(
                            &scr.trits[0], &scr.dvals[0], x, blocks_per_row,
                        );
                    }
                    t += 1;
                }
                di += 1;
            }
        };

        if m >= parallel_matvec_crossover() {
            use rayon::prelude::*;
            let threads = rayon::current_num_threads().max(1);
            let chunk_rows = m.div_ceil(4 * threads).max(16).min(m);
            out_t
                .par_chunks_mut(chunk_rows * n_rows)
                .enumerate()
                .for_each_init(
                    PtqRowScratch::default,
                    |scr, (ci, rows_out)| {
                        body(ci * chunk_rows, rows_out, scr);
                    },
                );
        } else {
            let mut scr = PtqRowScratch::default();
            body(0, out_t, &mut scr);
        }

        // Scatter transposed [m, n_rows] → caller [n_rows, m].
        for i in 0..m {
            for t in 0..n_rows {
                out[t * m + i] = out_t[i * n_rows + t];
            }
        }
    });
}

std::thread_local! {
    /// Grow-only transposed-output scratch for the batched PTQ1_0
    /// kernel (`[m, n_rows]` f32). TLS so steady-state prefill does
    /// zero allocations; ~10 MB peak for the 27B's largest tensor at
    /// a 512-token chunk.
    static PTQ1_0_BATCH_SCRATCH: std::cell::RefCell<Vec<f32>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Fastdot core: per 128-block, accumulate sixteen 8-lane fmadds
/// into a BLOCK vector sum (elements 120..128 are just the 16th
/// step), fold the block scale with one broadcast-fmadd into a
/// running vector total, and horizontal-sum ONCE per dot at the end.
/// Different (tighter) FP summation order than the reference kernel:
/// deterministic, tolerance-gated, NOT bitwise.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn ptq1_0_row_dot_i8_avx2_fast(
    trits: &[i8],
    dvals: &[f32],
    x: &[f32],
    blocks_per_row: usize,
) -> f32 {
    use std::arch::x86_64::*;
    const QK: usize = 128;
    // Four independent accumulators break the per-block 16-deep FMA
    // dependency chain (~1 fmadd / 4-5 cyc latency-bound → ~throughput
    // bound), and four block-total accumulators break the cross-block
    // chain. This is the single-token DECODE latency fix — the same
    // ILP the batched kernel got from token tiling, which decode
    // (one activation vector) can't use. Reassociated sum → fastdot
    // tolerance-gated, not bitwise.
    let mut vt = [_mm256_setzero_ps(); 4];
    for b in 0..blocks_per_row {
        let tb = trits.as_ptr().add(b * QK);
        let xb = x.as_ptr().add(b * QK);
        let mut vs = [_mm256_setzero_ps(); 4];
        for q in 0..4 {
            let base = q * 32;
            vs[q] = _mm256_fmadd_ps(load_trits8_ps(tb.add(base)), _mm256_loadu_ps(xb.add(base)), vs[q]);
            vs[q] = _mm256_fmadd_ps(load_trits8_ps(tb.add(base + 8)), _mm256_loadu_ps(xb.add(base + 8)), vs[q]);
            vs[q] = _mm256_fmadd_ps(load_trits8_ps(tb.add(base + 16)), _mm256_loadu_ps(xb.add(base + 16)), vs[q]);
            vs[q] = _mm256_fmadd_ps(load_trits8_ps(tb.add(base + 24)), _mm256_loadu_ps(xb.add(base + 24)), vs[q]);
        }
        let vsum = _mm256_add_ps(_mm256_add_ps(vs[0], vs[1]), _mm256_add_ps(vs[2], vs[3]));
        vt[b & 3] = _mm256_fmadd_ps(_mm256_set1_ps(dvals[b]), vsum, vt[b & 3]);
    }
    let vtotal = _mm256_add_ps(_mm256_add_ps(vt[0], vt[1]), _mm256_add_ps(vt[2], vt[3]));
    let hi128 = _mm256_extractf128_ps::<1>(vtotal);
    let lo128 = _mm256_castps256_ps128(vtotal);
    let s128 = _mm_add_ps(lo128, hi128);
    let s64 = _mm_add_ps(s128, _mm_movehl_ps(s128, s128));
    let s32 = _mm_add_ss(s64, _mm_shuffle_ps::<1>(s64, s64));
    _mm_cvtss_f32(s32)
}

/// AArch64 NEON fastdot: the ARM analogue of
/// [`ptq1_0_row_dot_i8_avx2_fast`] for the decode path. Per 128-block:
/// four independent `float32x4` accumulators break the FMA dependency
/// chain (NEON's `fmla` is throughput-bound like AVX2's), each fed 4
/// lanes at a time; fold the block scale `d` with `vfmaq_n` into a
/// running vector total; horizontal-sum ONCE at the very end.
///
/// Reassociated summation like the AVX2 fast path (4-lane groups vs
/// 8-lane), so it is tolerance-gated against the reference kernel, NOT
/// bitwise — validated by `ptq1_0_row_fast_tolerance_and_determinism`
/// (which runs this path on aarch64). NEON is baseline on aarch64, so no
/// runtime feature detection is needed. Trits are i8 ±1/0; widen to f32
/// and multiply-accumulate with the activations.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn ptq1_0_row_dot_i8_neon_fast(
    trits: &[i8],
    dvals: &[f32],
    x: &[f32],
    blocks_per_row: usize,
) -> f32 {
    use std::arch::aarch64::*;
    const QK: usize = 128;
    let mut total = vdupq_n_f32(0.0);
    for b in 0..blocks_per_row {
        let tb = trits.as_ptr().add(b * QK);
        let xb = x.as_ptr().add(b * QK);
        let mut vs = [vdupq_n_f32(0.0); 4];
        // 128 elements, 8 per iteration (two f32x4), 16 iterations.
        let mut e = 0usize;
        let mut acc_i = 0usize;
        while e < QK {
            // Load 8 i8 trits, widen i8 -> i16 -> i32 -> f32 (two f32x4).
            let t8 = vld1_s8(tb.add(e));
            let t16 = vmovl_s8(t8);
            let t_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(t16)));
            let t_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(t16)));
            let x_lo = vld1q_f32(xb.add(e));
            let x_hi = vld1q_f32(xb.add(e + 4));
            vs[acc_i & 3] = vfmaq_f32(vs[acc_i & 3], t_lo, x_lo);
            acc_i += 1;
            vs[acc_i & 3] = vfmaq_f32(vs[acc_i & 3], t_hi, x_hi);
            acc_i += 1;
            e += 8;
        }
        let vsum = vaddq_f32(vaddq_f32(vs[0], vs[1]), vaddq_f32(vs[2], vs[3]));
        // total += vsum * d[b] (scalar broadcast fmla).
        total = vfmaq_n_f32(total, vsum, dvals[b]);
    }
    vaddvq_f32(total)
}

/// Fastfold, 3 rows x 4 tokens: `d` is folded into the trit vector
/// per (row, step) — `tvd = trits * d_row` — and fmadd'd straight
/// into 12 running totals, so a (row, token) pair costs one register
/// instead of two and no per-block boundary work remains at all.
/// Same tolerance/determinism gates as the rest of fastdot; the
/// d-prefold is yet another (benign) summation-order change.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn ptq1_0_rows3_dot_i8_avx2_t4_fastfold(
    scr: &PtqRowScratch,
    xs: &[f32],
    out: &mut [f32; 12],
    blocks_per_row: usize,
    k: usize,
) {
    use std::arch::x86_64::*;
    const QK: usize = 128;
    debug_assert_eq!(xs.len(), 4 * k);
    let tp: [*const i8; 3] = [
        scr.trits[0].as_ptr(),
        scr.trits[1].as_ptr(),
        scr.trits[2].as_ptr(),
    ];
    let mut vtotal = [_mm256_setzero_ps(); 12]; // [row*4 + t]
    for b in 0..blocks_per_row {
        let boff = b * QK;
        let xoff = b * QK;
        let d = [
            _mm256_set1_ps(scr.dvals[0][b]),
            _mm256_set1_ps(scr.dvals[1][b]),
            _mm256_set1_ps(scr.dvals[2][b]),
        ];
        for step in 0..16 {
            let base = step * 8;
            for r in 0..3 {
                let tvd = _mm256_mul_ps(load_trits8_ps(tp[r].add(boff + base)), d[r]);
                for t in 0..4 {
                    vtotal[r * 4 + t] = _mm256_fmadd_ps(
                        tvd,
                        _mm256_loadu_ps(xs.as_ptr().add(t * k + xoff + base)),
                        vtotal[r * 4 + t],
                    );
                }
            }
        }
    }
    for (p, vt) in vtotal.iter().enumerate() {
        let hi128 = _mm256_extractf128_ps::<1>(*vt);
        let lo128 = _mm256_castps256_ps128(*vt);
        let s128 = _mm_add_ps(lo128, hi128);
        let s64 = _mm_add_ps(s128, _mm_movehl_ps(s128, s128));
        let s32 = _mm_add_ss(s64, _mm_shuffle_ps::<1>(s64, s64));
        out[p] = _mm_cvtss_f32(s32);
    }
}

/// Fastdot, 1 row x 6 tokens: 6 block sums + 6 running totals + trit
/// + x = 14 ymm.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn ptq1_0_row_dot_i8_avx2_t6_fast(
    trits: &[i8],
    dvals: &[f32],
    xs: &[f32],
    out: &mut [f32; 6],
    blocks_per_row: usize,
    k: usize,
) {
    use std::arch::x86_64::*;
    const QK: usize = 128;
    debug_assert_eq!(xs.len(), 6 * k);
    let mut vtotal = [_mm256_setzero_ps(); 6];
    for b in 0..blocks_per_row {
        let tb = trits.as_ptr().add(b * QK);
        let xoff = b * QK;
        let mut vsum = [_mm256_setzero_ps(); 6];
        for step in 0..16 {
            let base = step * 8;
            let tv = load_trits8_ps(tb.add(base));
            for (t, vs) in vsum.iter_mut().enumerate() {
                *vs = _mm256_fmadd_ps(
                    tv,
                    _mm256_loadu_ps(xs.as_ptr().add(t * k + xoff + base)),
                    *vs,
                );
            }
        }
        let d = _mm256_set1_ps(dvals[b]);
        for t in 0..6 {
            vtotal[t] = _mm256_fmadd_ps(d, vsum[t], vtotal[t]);
        }
    }
    for (t, vt) in vtotal.iter().enumerate() {
        let hi128 = _mm256_extractf128_ps::<1>(*vt);
        let lo128 = _mm256_castps256_ps128(*vt);
        let s128 = _mm_add_ps(lo128, hi128);
        let s64 = _mm_add_ps(s128, _mm_movehl_ps(s128, s128));
        let s32 = _mm_add_ss(s64, _mm_shuffle_ps::<1>(s64, s64));
        out[t] = _mm_cvtss_f32(s32);
    }
}

/// Fastdot, 2 rows x 3 tokens: 6 block sums + 6 running totals +
/// 2 trit + 1 x = 15 ymm.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn ptq1_0_rows2_dot_i8_avx2_t3_fast(
    trits_a: &[i8],
    dvals_a: &[f32],
    trits_b: &[i8],
    dvals_b: &[f32],
    xs: &[f32],
    out_a: &mut [f32; 3],
    out_b: &mut [f32; 3],
    blocks_per_row: usize,
    k: usize,
) {
    use std::arch::x86_64::*;
    const QK: usize = 128;
    debug_assert_eq!(xs.len(), 3 * k);
    let mut vtotal = [_mm256_setzero_ps(); 6]; // [row*3 + t]
    for b in 0..blocks_per_row {
        let ta = trits_a.as_ptr().add(b * QK);
        let tbp = trits_b.as_ptr().add(b * QK);
        let xoff = b * QK;
        let mut vsum = [_mm256_setzero_ps(); 6];
        for step in 0..16 {
            let base = step * 8;
            let tva = load_trits8_ps(ta.add(base));
            let tvb = load_trits8_ps(tbp.add(base));
            for t in 0..3 {
                let xv = _mm256_loadu_ps(xs.as_ptr().add(t * k + xoff + base));
                vsum[t] = _mm256_fmadd_ps(tva, xv, vsum[t]);
                vsum[3 + t] = _mm256_fmadd_ps(tvb, xv, vsum[3 + t]);
            }
        }
        let da = _mm256_set1_ps(dvals_a[b]);
        let db = _mm256_set1_ps(dvals_b[b]);
        for t in 0..3 {
            vtotal[t] = _mm256_fmadd_ps(da, vsum[t], vtotal[t]);
            vtotal[3 + t] = _mm256_fmadd_ps(db, vsum[3 + t], vtotal[3 + t]);
        }
    }
    for t in 0..3 {
        for (row, dst) in [(0usize, &mut *out_a), (1usize, &mut *out_b)] {
            let v = vtotal[row * 3 + t];
            let hi128 = _mm256_extractf128_ps::<1>(v);
            let lo128 = _mm256_castps256_ps128(v);
            let s128 = _mm_add_ps(lo128, hi128);
            let s64 = _mm_add_ps(s128, _mm_movehl_ps(s128, s128));
            let s32 = _mm_add_ss(s64, _mm_shuffle_ps::<1>(s64, s64));
            dst[t] = _mm_cvtss_f32(s32);
        }
    }
}

/// Per-worker decode scratch for the register-blocked batched
/// PTQ1_0 kernel: up to four rows' i8 trits + per-block scales,
/// grow-only.
#[derive(Default)]
struct PtqRowScratch {
    trits: [Vec<i8>; 4],
    dvals: [Vec<f32>; 4],
}

/// 4-row x 3-token register-blocked i8 replay: 12 private
/// accumulator chains + 4 trit vectors = exactly the 16 ymm budget;
/// each x vector feeds four fmadds and each trit vector three. Every
/// (row, token) accumulator sees the single-token replay's exact op
/// sequence -- bitwise-identical outputs. `out` is `[row][token]`
/// row-major (4x3).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn ptq1_0_rows4_dot_decoded_i8_avx2_t3(
    scr: &PtqRowScratch,
    xs: &[f32],
    out: &mut [f32; 12],
    blocks_per_row: usize,
    k: usize,
) {
    use std::arch::x86_64::*;
    const QK: usize = 128;
    debug_assert_eq!(xs.len(), 3 * k);
    let tp: [*const i8; 4] = [
        scr.trits[0].as_ptr(),
        scr.trits[1].as_ptr(),
        scr.trits[2].as_ptr(),
        scr.trits[3].as_ptr(),
    ];
    let mut acc = [0.0f32; 12];
    for b in 0..blocks_per_row {
        let boff = b * QK;
        let xoff = b * QK;
        let mut va = [_mm256_setzero_ps(); 12];
        for step in 0..15 {
            let base = step * 8;
            let tv = [
                load_trits8_ps(tp[0].add(boff + base)),
                load_trits8_ps(tp[1].add(boff + base)),
                load_trits8_ps(tp[2].add(boff + base)),
                load_trits8_ps(tp[3].add(boff + base)),
            ];
            for t in 0..3 {
                let xv = _mm256_loadu_ps(xs.as_ptr().add(t * k + xoff + base));
                for r in 0..4 {
                    va[r * 3 + t] = _mm256_fmadd_ps(tv[r], xv, va[r * 3 + t]);
                }
            }
        }
        for r in 0..4 {
            for t in 0..3 {
                let v = va[r * 3 + t];
                let hi128 = _mm256_extractf128_ps::<1>(v);
                let lo128 = _mm256_castps256_ps128(v);
                let s128 = _mm_add_ps(lo128, hi128);
                let s64 = _mm_add_ps(s128, _mm_movehl_ps(s128, s128));
                let s32 = _mm_add_ss(s64, _mm_shuffle_ps::<1>(s64, s64));
                let mut sum = _mm_cvtss_f32(s32);
                for e in 120..QK {
                    sum += *tp[r].add(boff + e) as f32 * xs[t * k + xoff + e];
                }
                acc[r * 3 + t] += scr.dvals[r][b] * sum;
            }
        }
    }
    *out = acc;
}

/// Decode one PTQ1_0 weight row into `trits` (i8 -1/0/+1, element
/// order identical to the kernels' walk) and per-block scales into
/// `dvals`. Integer-side only; i8 keeps the decoded stream at 1
/// byte/weight (the f32 buffer was 4x the L2 traffic).
fn decode_ptq1_0_row_trits_i8(
    row: &[u8],
    blocks_per_row: usize,
    trits: &mut Vec<i8>,
    dvals: &mut Vec<f32>,
) {
    trits.clear();
    trits.resize(blocks_per_row * 128, 0);
    dvals.clear();
    dvals.resize(blocks_per_row, 0.0);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: runtime feature detection above.
            unsafe { decode_ptq1_0_row_trits_i8_avx2(row, blocks_per_row, trits, dvals) };
            return;
        }
    }
    decode_ptq1_0_row_trits_i8_scalar(row, blocks_per_row, trits, dvals);
}

fn decode_ptq1_0_row_trits_i8_scalar(
    row: &[u8],
    blocks_per_row: usize,
    trits: &mut [i8],
    dvals: &mut [f32],
) {
    const BLOCK_BYTES: usize = 28;
    const QK: usize = 128;
    const QS_BYTES: usize = 24;
    const STAGES: [usize; 3] = [32, 16, 8];
    const POW3: [u8; 6] = [1, 3, 9, 27, 81, 243];
    for b in 0..blocks_per_row {
        let off = b * BLOCK_BYTES;
        let qs = &row[off..off + QS_BYTES];
        let qh = &row[off + QS_BYTES..off + QS_BYTES + 2];
        dvals[b] = f16::from_le_bytes([row[off + 26], row[off + 27]]).to_f32();
        let tb = &mut trits[b * QK..(b + 1) * QK];
        let mut e = 0usize;
        let mut j = 0usize;
        for &c in STAGES.iter() {
            while j + c <= QS_BYTES {
                for n in 0..5 {
                    for mm in 0..c {
                        let q = qs[j + mm].wrapping_mul(POW3[n]);
                        tb[e] = ((((q as u16) * 3) >> 8) as i32 - 1) as i8;
                        e += 1;
                    }
                }
                j += c;
            }
        }
        for n in 0..4 {
            for h in 0..2 {
                let q = qh[h].wrapping_mul(POW3[n]);
                tb[e] = ((((q as u16) * 3) >> 8) as i32 - 1) as i8;
                e += 1;
            }
        }
        debug_assert_eq!(e, QK);
    }
}

/// AVX2 trit decode: the single kernel's `vpmullw` multiply-high
/// extraction (`((q * 3^n) & 0xFF) * 3 >> 8 - 1`), 16 (chunk 1) or
/// 8 (chunk 2) trits per digit stage, packed i16 -> i8 and stored in
/// the same element order as the scalar walk. Integer math is
/// identical, so the decoded bytes are EQUAL to the scalar decode —
/// this is pure decode speed (VTune: the scalar decode was ~24% of
/// batched-kernel CPU).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2")]
unsafe fn decode_ptq1_0_row_trits_i8_avx2(
    row: &[u8],
    blocks_per_row: usize,
    trits: &mut [i8],
    dvals: &mut [f32],
) {
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 28;
    const QK: usize = 128;
    const QS_BYTES: usize = 24;
    const POW3: [u8; 6] = [1, 3, 9, 27, 81, 243];
    let mask_ff = _mm256_set1_epi16(0xFF);
    let three = _mm256_set1_epi16(3);
    let one = _mm256_set1_epi16(1);
    for b in 0..blocks_per_row {
        let off = b * BLOCK_BYTES;
        let qs = row.as_ptr().add(off);
        let qh = &row[off + QS_BYTES..off + QS_BYTES + 2];
        dvals[b] = f16::from_le_bytes([row[off + 26], row[off + 27]]).to_f32();
        let tb = trits.as_mut_ptr().add(b * QK);
        // Chunk 1: qs[0..16], 5 stages x 16 trits -> elements 0..80.
        let c1 = _mm256_cvtepu8_epi16(_mm_loadu_si128(qs as *const __m128i));
        for n in 0..5 {
            let scaled = _mm256_and_si256(
                _mm256_mullo_epi16(c1, _mm256_set1_epi16(POW3[n] as i16)),
                mask_ff,
            );
            let trit16 = _mm256_sub_epi16(
                _mm256_srli_epi16::<8>(_mm256_mullo_epi16(scaled, three)),
                one,
            );
            let lo = _mm256_castsi256_si128(trit16);
            let hi = _mm256_extracti128_si256::<1>(trit16);
            _mm_storeu_si128(
                tb.add(n * 16) as *mut __m128i,
                _mm_packs_epi16(lo, hi),
            );
        }
        // Chunk 2: qs[16..24], 5 stages x 8 trits -> elements 80..120.
        let c2 = _mm_cvtepu8_epi16(_mm_loadl_epi64(qs.add(16) as *const __m128i));
        let mask_ff128 = _mm_set1_epi16(0xFF);
        let three128 = _mm_set1_epi16(3);
        let one128 = _mm_set1_epi16(1);
        for n in 0..5 {
            let scaled = _mm_and_si128(
                _mm_mullo_epi16(c2, _mm_set1_epi16(POW3[n] as i16)),
                mask_ff128,
            );
            let trit16 = _mm_sub_epi16(
                _mm_srli_epi16::<8>(_mm_mullo_epi16(scaled, three128)),
                one128,
            );
            _mm_storel_epi64(
                tb.add(80 + n * 8) as *mut __m128i,
                _mm_packs_epi16(trit16, trit16),
            );
        }
        // qh tail: 8 trits, scalar (elements 120..128).
        let mut e = 120usize;
        for n in 0..4 {
            for h in 0..2 {
                let q = qh[h].wrapping_mul(POW3[n]);
                *tb.add(e) = ((((q as u16) * 3) >> 8) as i32 - 1) as i8;
                e += 1;
            }
        }
    }
}

/// Dot one decoded row against one activation row, replaying the
/// single-kernel FP accumulation exactly (the i8 trits convert to
/// the identical -1.0/0.0/+1.0 f32 values in-register).
#[inline]
fn ptq1_0_row_dot_decoded_i8(
    trits: &[i8],
    dvals: &[f32],
    x: &[f32],
    blocks_per_row: usize,
) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime feature detection above.
            return unsafe {
                ptq1_0_row_dot_decoded_i8_avx2(trits, dvals, x, blocks_per_row)
            };
        }
    }
    ptq1_0_row_dot_decoded_i8_scalar(trits, dvals, x, blocks_per_row)
}

fn ptq1_0_row_dot_decoded_i8_scalar(
    trits: &[i8],
    dvals: &[f32],
    x: &[f32],
    blocks_per_row: usize,
) -> f32 {
    const QK: usize = 128;
    let mut acc = 0.0f32;
    for b in 0..blocks_per_row {
        let tb = &trits[b * QK..(b + 1) * QK];
        let xb = &x[b * QK..(b + 1) * QK];
        let mut sum = 0.0f32;
        for e in 0..QK {
            sum += tb[e] as f32 * xb[e];
        }
        acc += dvals[b] * sum;
    }
    acc
}

/// Load 8 i8 trits and widen to the exact f32 values the old f32
/// buffer held.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn load_trits8_ps(ptr: *const i8) -> core::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let b = _mm_loadl_epi64(ptr as *const __m128i);
    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b))
}

/// Single-row AVX2 replay over i8 trits: fifteen 8-lane fmadds per
/// 128-block in linear element order (the same flattened sequence as
/// the original lo/hi-stage walk), the single kernel's horizontal-sum
/// shuffle order, eight scalar tail adds, then `acc += d*sum` -- the
/// identical FP op sequence with identical values, so results are
/// bit-equal to `matvec_ptq1_0_w_f32_a_avx2`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn ptq1_0_row_dot_decoded_i8_avx2(
    trits: &[i8],
    dvals: &[f32],
    x: &[f32],
    blocks_per_row: usize,
) -> f32 {
    use std::arch::x86_64::*;
    const QK: usize = 128;
    let mut acc = 0.0f32;
    for b in 0..blocks_per_row {
        let tb = trits.as_ptr().add(b * QK);
        let xb = x.as_ptr().add(b * QK);
        let mut vacc = _mm256_setzero_ps();
        for step in 0..15 {
            let base = step * 8;
            vacc = _mm256_fmadd_ps(
                load_trits8_ps(tb.add(base)),
                _mm256_loadu_ps(xb.add(base)),
                vacc,
            );
        }
        let hi128 = _mm256_extractf128_ps::<1>(vacc);
        let lo128 = _mm256_castps256_ps128(vacc);
        let s128 = _mm_add_ps(lo128, hi128);
        let s64 = _mm_add_ps(s128, _mm_movehl_ps(s128, s128));
        let s32 = _mm_add_ss(s64, _mm_shuffle_ps::<1>(s64, s64));
        let mut sum = _mm_cvtss_f32(s32);
        for e in 120..QK {
            sum += *tb.add(e) as f32 * *xb.add(e);
        }
        acc += dvals[b] * sum;
    }
    acc
}

/// 8-token tiled i8 replay (row-tail path): one trit load feeds 8
/// private accumulator chains; per-token FP order preserved.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn ptq1_0_row_dot_decoded_i8_avx2_t8(
    trits: &[i8],
    dvals: &[f32],
    xs: &[f32],
    out: &mut [f32],
    blocks_per_row: usize,
    k: usize,
) {
    use std::arch::x86_64::*;
    const QK: usize = 128;
    debug_assert_eq!(xs.len(), 8 * k);
    debug_assert_eq!(out.len(), 8);
    let mut acc = [0.0f32; 8];
    for b in 0..blocks_per_row {
        let tb = trits.as_ptr().add(b * QK);
        let xoff = b * QK;
        let mut vacc = [_mm256_setzero_ps(); 8];
        for step in 0..15 {
            let base = step * 8;
            let tv = load_trits8_ps(tb.add(base));
            for (t, va) in vacc.iter_mut().enumerate() {
                *va = _mm256_fmadd_ps(
                    tv,
                    _mm256_loadu_ps(xs.as_ptr().add(t * k + xoff + base)),
                    *va,
                );
            }
        }
        for (t, va) in vacc.iter().enumerate() {
            let hi128 = _mm256_extractf128_ps::<1>(*va);
            let lo128 = _mm256_castps256_ps128(*va);
            let s128 = _mm_add_ps(lo128, hi128);
            let s64 = _mm_add_ps(s128, _mm_movehl_ps(s128, s128));
            let s32 = _mm_add_ss(s64, _mm_shuffle_ps::<1>(s64, s64));
            let mut sum = _mm_cvtss_f32(s32);
            for e in 120..QK {
                sum += *tb.add(e) as f32 * xs[t * k + xoff + e];
            }
            acc[t] += dvals[b] * sum;
        }
    }
    out.copy_from_slice(&acc);
}

/// 2-row x 4-token register-blocked i8 replay: per 8-lane step, two
/// trit loads and four x loads feed EIGHT fmadds (8 private chains).
/// Each (row, token) accumulator sees exactly the single-token
/// replay's op sequence -- bitwise-identical outputs; the blocking
/// only improves L2 reuse (x rows read once per two weight rows) and
/// keeps eight FMA chains in flight.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn ptq1_0_rows2_dot_decoded_i8_avx2_t4(
    trits_a: &[i8],
    dvals_a: &[f32],
    trits_b: &[i8],
    dvals_b: &[f32],
    xs: &[f32],
    out_a: &mut [f32],
    out_b: &mut [f32],
    blocks_per_row: usize,
    k: usize,
) {
    use std::arch::x86_64::*;
    const QK: usize = 128;
    debug_assert_eq!(xs.len(), 4 * k);
    debug_assert_eq!(out_a.len(), 4);
    debug_assert_eq!(out_b.len(), 4);
    let mut acc_a = [0.0f32; 4];
    let mut acc_b = [0.0f32; 4];
    for b in 0..blocks_per_row {
        let ta = trits_a.as_ptr().add(b * QK);
        let tbp = trits_b.as_ptr().add(b * QK);
        let xoff = b * QK;
        let mut va = [_mm256_setzero_ps(); 4];
        let mut vb = [_mm256_setzero_ps(); 4];
        for step in 0..15 {
            let base = step * 8;
            let tva = load_trits8_ps(ta.add(base));
            let tvb = load_trits8_ps(tbp.add(base));
            for t in 0..4 {
                let xv = _mm256_loadu_ps(xs.as_ptr().add(t * k + xoff + base));
                va[t] = _mm256_fmadd_ps(tva, xv, va[t]);
                vb[t] = _mm256_fmadd_ps(tvb, xv, vb[t]);
            }
        }
        for t in 0..4 {
            let hi128 = _mm256_extractf128_ps::<1>(va[t]);
            let lo128 = _mm256_castps256_ps128(va[t]);
            let s128 = _mm_add_ps(lo128, hi128);
            let s64 = _mm_add_ps(s128, _mm_movehl_ps(s128, s128));
            let s32 = _mm_add_ss(s64, _mm_shuffle_ps::<1>(s64, s64));
            let mut sum = _mm_cvtss_f32(s32);
            for e in 120..QK {
                sum += *ta.add(e) as f32 * xs[t * k + xoff + e];
            }
            acc_a[t] += dvals_a[b] * sum;

            let hi128 = _mm256_extractf128_ps::<1>(vb[t]);
            let lo128 = _mm256_castps256_ps128(vb[t]);
            let s128 = _mm_add_ps(lo128, hi128);
            let s64 = _mm_add_ps(s128, _mm_movehl_ps(s128, s128));
            let s32 = _mm_add_ss(s64, _mm_shuffle_ps::<1>(s64, s64));
            let mut sum = _mm_cvtss_f32(s32);
            for e in 120..QK {
                sum += *tbp.add(e) as f32 * xs[t * k + xoff + e];
            }
            acc_b[t] += dvals_b[b] * sum;
        }
    }
    out_a.copy_from_slice(&acc_a);
    out_b.copy_from_slice(&acc_b);
}

/// AVX2 PTQ1_0 matvec. The trit extraction runs 16 bytes at a time in
/// u16 lanes: `trit = (((q · 3ⁿ) & 0xFF) · 3) >> 8` maps directly onto
/// `vpmullw` / `vpand` / `vpsrlw`, producing 16 trits per digit stage
/// in the exact element order the scalar walk uses (digit-major within
/// each chunk). Per-block sums are reduced and scaled by `d` exactly
/// like the scalar kernel; only the FP add order inside a 16-lane step
/// differs.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn matvec_ptq1_0_w_f32_a_avx2(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 28;
    const QK: usize = 128;
    const QS_BYTES: usize = 24;
    const POW3: [u8; 6] = [1, 3, 9, 27, 81, 243];
    debug_assert_eq!(k % QK, 0, "PTQ1_0 matvec requires k % 128 == 0");
    let blocks_per_row = k / QK;
    debug_assert_eq!(w_bytes.len(), m * blocks_per_row * BLOCK_BYTES);
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(out.len(), m);

    let mask_ff = _mm256_set1_epi16(0xFF);
    let three = _mm256_set1_epi16(3);
    let one = _mm256_set1_epi16(1);

    // Widen 16 i16 trit values (−1/0/+1) to two f32x8 and fmadd them
    // against 16 activations.
    #[inline(always)]
    unsafe fn fmadd_trits16(acc: __m256, trits: __m256i, x_ptr: *const f32) -> __m256 {
        let lo_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(trits));
        let hi_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(trits));
        let lo_f = _mm256_cvtepi32_ps(lo_i32);
        let hi_f = _mm256_cvtepi32_ps(hi_i32);
        let acc = _mm256_fmadd_ps(lo_f, _mm256_loadu_ps(x_ptr), acc);
        _mm256_fmadd_ps(hi_f, _mm256_loadu_ps(x_ptr.add(8)), acc)
    }

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let qs = &w_bytes[off..off + QS_BYTES];
            let qh = &w_bytes[off + QS_BYTES..off + QS_BYTES + 2];
            let d = f16::from_le_bytes([w_bytes[off + 26], w_bytes[off + 27]]).to_f32();
            let xb = &x[b * QK..(b + 1) * QK];

            let mut vacc = _mm256_setzero_ps();

            // Chunk 1: qs[0..16], 5 digit stages × 16 lanes → elements 0..80.
            let c1 = _mm256_cvtepu8_epi16(_mm_loadu_si128(qs.as_ptr() as *const __m128i));
            for n in 0..5 {
                let scaled = _mm256_and_si256(
                    _mm256_mullo_epi16(c1, _mm256_set1_epi16(POW3[n] as i16)),
                    mask_ff,
                );
                let trit = _mm256_sub_epi16(
                    _mm256_srli_epi16::<8>(_mm256_mullo_epi16(scaled, three)),
                    one,
                );
                vacc = fmadd_trits16(vacc, trit, xb.as_ptr().add(n * 16));
            }

            // Chunk 2: qs[16..24], 5 digit stages × 8 lanes → elements 80..120.
            // Widen the 8 bytes into the low 8 u16 lanes; the high 8
            // lanes are zeros, whose trit decodes to 0−1 = −1, so they
            // must NOT touch the accumulator — use the 8-wide half only.
            let c2 = _mm256_cvtepu8_epi16(_mm_loadl_epi64(
                qs.as_ptr().add(16) as *const __m128i
            ));
            for n in 0..5 {
                let scaled = _mm256_and_si256(
                    _mm256_mullo_epi16(c2, _mm256_set1_epi16(POW3[n] as i16)),
                    mask_ff,
                );
                let trit16 = _mm256_sub_epi16(
                    _mm256_srli_epi16::<8>(_mm256_mullo_epi16(scaled, three)),
                    one,
                );
                let lo_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(trit16));
                let lo_f = _mm256_cvtepi32_ps(lo_i32);
                vacc = _mm256_fmadd_ps(
                    lo_f,
                    _mm256_loadu_ps(xb.as_ptr().add(80 + n * 8)),
                    vacc,
                );
            }

            // Horizontal sum of the SIMD accumulator.
            let hi128 = _mm256_extractf128_ps::<1>(vacc);
            let lo128 = _mm256_castps256_ps128(vacc);
            let s128 = _mm_add_ps(lo128, hi128);
            let s64 = _mm_add_ps(s128, _mm_movehl_ps(s128, s128));
            let s32 = _mm_add_ss(s64, _mm_shuffle_ps::<1>(s64, s64));
            let mut sum = _mm_cvtss_f32(s32);

            // qh: 2 bytes × 4 digit stages → elements 120..128 (scalar).
            let mut e = 120usize;
            for n in 0..4 {
                for h in 0..2 {
                    let q = qh[h].wrapping_mul(POW3[n]);
                    let trit = (((q as u16) * 3) >> 8) as i32 - 1;
                    sum += trit as f32 * xb[e];
                    e += 1;
                }
            }
            acc += d * sum;
        }
        out[i] = acc;
    }
}

fn matvec_ptq1_0_w_f32_a_scalar(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    const BLOCK_BYTES: usize = 28;
    const QK: usize = 128;
    const QS_BYTES: usize = 24;
    const QH_BYTES: usize = 2;
    const STAGES: [usize; 3] = [32, 16, 8];
    const POW3: [u8; 6] = [1, 3, 9, 27, 81, 243];
    debug_assert_eq!(k % QK, 0, "PTQ1_0 matvec requires k % 128 == 0");
    let blocks_per_row = k / QK;
    debug_assert_eq!(w_bytes.len(), m * blocks_per_row * BLOCK_BYTES);
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(out.len(), m);

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let qs = &w_bytes[off..off + QS_BYTES];
            let qh = &w_bytes[off + QS_BYTES..off + QS_BYTES + QH_BYTES];
            let d = f16::from_le_bytes([w_bytes[off + 26], w_bytes[off + 27]]).to_f32();
            let xb = &x[b * QK..(b + 1) * QK];
            let mut sum = 0.0f32;
            let mut e = 0usize;
            let mut j = 0usize;
            for &c in STAGES.iter() {
                while j + c <= QS_BYTES {
                    for n in 0..5 {
                        for mm in 0..c {
                            let q = qs[j + mm].wrapping_mul(POW3[n]);
                            let trit = (((q as u16) * 3) >> 8) as i32 - 1;
                            sum += trit as f32 * xb[e];
                            e += 1;
                        }
                    }
                    j += c;
                }
            }
            for n in 0..4 {
                for h in 0..QH_BYTES {
                    let q = qh[h].wrapping_mul(POW3[n]);
                    let trit = (((q as u16) * 3) >> 8) as i32 - 1;
                    sum += trit as f32 * xb[e];
                    e += 1;
                }
            }
            debug_assert_eq!(e, QK);
            acc += d * sum;
        }
        out[i] = acc;
    }
}

// ======================================================================
// AArch64 NEON grid-codebook IQ matvecs (IQ1_S / IQ1_M / IQ2_XXS / IQ2_XS
// / IQ2_S / IQ3_XXS / IQ3_S).
//
// These formats dequant through 256..2048-entry packed grids, far too big
// for an in-register `vqtbl` lookup (unlike IQ4_NL / IQ4_XS, whose 16-entry
// codebook *does* fit). NEON has no gather instruction either, so — exactly
// as the task brief prescribes — each grid entry is fetched scalar (the same
// index math as the `_scalar` reference) and only the per-run arithmetic
// (sign application + the `db·grid·x` FMA over 8 lanes) is vectorized. The
// x86 paths lean on `_mm_i32gather_*`; on ARM that part stays scalar, so the
// win here is modest, but it closes NEON parity for every quant format.
// All are bit-close (abs<1e-3 || rel<1e-5) to the scalar reference, checked
// under qemu by the `neon_parity` tests.
// ======================================================================

/// Widen a `uint8x8` (8 bytes) to two `float32x4` (unsigned). The 8-wide
/// twin of [`u8x16_to_f32x4x4`], for the 8-weight IQ grid runs.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn u8x8_to_f32x4x2(v: std::arch::aarch64::uint8x8_t) -> [std::arch::aarch64::float32x4_t; 2] {
    use std::arch::aarch64::*;
    let w16 = vmovl_u8(v);
    [
        vcvtq_f32_u32(vmovl_u16(vget_low_u16(w16))),
        vcvtq_f32_u32(vmovl_u16(vget_high_u16(w16))),
    ]
}

/// Widen a signed `int8x8` (8 bytes) to two `float32x4`. The IQ1 grids
/// store signed `i8` points (value + sign folded into the codebook), so
/// their NEON widen is signed.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn s8x8_to_f32x4x2(v: std::arch::aarch64::int8x8_t) -> [std::arch::aarch64::float32x4_t; 2] {
    use std::arch::aarch64::*;
    let w16 = vmovl_s8(v);
    [
        vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16))),
        vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16))),
    ]
}

/// Expand an 8-bit IQ sign byte to two `float32x4` of ±1.0 (lane `j` is
/// `-1.0` iff bit `j` is set — the `KMASK_IQ2XS = [1,2,4,8,16,32,64,128]`
/// convention the scalar reference uses). Shared by every sign-table IQ
/// grid kernel (IQ2_* / IQ3_*).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn iq_signs8_to_f32x4x2(sign_byte: u8) -> [std::arch::aarch64::float32x4_t; 2] {
    use std::arch::aarch64::*;
    let lo_bits = [1u32, 2, 4, 8];
    let hi_bits = [16u32, 32, 64, 128];
    let lo_mask = vld1q_u32(lo_bits.as_ptr());
    let hi_mask = vld1q_u32(hi_bits.as_ptr());
    let sbv = vdupq_n_u32(sign_byte as u32);
    let one = vdupq_n_f32(1.0);
    let neg = vdupq_n_f32(-1.0);
    // `(sbv & bit) == bit` → bit set → pick -1.0, else +1.0.
    let lo = vceqq_u32(vandq_u32(sbv, lo_mask), lo_mask);
    let hi = vceqq_u32(vandq_u32(sbv, hi_mask), hi_mask);
    [vbslq_f32(lo, neg, one), vbslq_f32(hi, neg, one)]
}

/// AArch64 NEON IQ2_XXS matvec. Scalar grid gather (8-byte `u8` points per
/// index) + `KSIGNS_IQ2XS` sign expansion, then an 8-lane FMA per run.
/// Mirrors [`matvec_iq2_xxs_w_f32_a_scalar`].
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn matvec_iq2_xxs_w_f32_a_neon(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use rustllama_gguf::dequant::{IQ2XXS_GRID, KSIGNS_IQ2XS};
    use std::arch::aarch64::*;
    const BLOCK_BYTES: usize = 66;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;
    let grid_bytes: &[u8] = bytemuck::cast_slice(&IQ2XXS_GRID);
    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d = f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32();
            let qs = &w_bytes[off + 2..off + 2 + 64];
            let xptr = x.as_ptr().add(b * QK_K);
            for ib32 in 0..8 {
                let aux0 = u32::from_le_bytes([
                    qs[8 * ib32],
                    qs[8 * ib32 + 1],
                    qs[8 * ib32 + 2],
                    qs[8 * ib32 + 3],
                ]);
                let aux1 = u32::from_le_bytes([
                    qs[8 * ib32 + 4],
                    qs[8 * ib32 + 5],
                    qs[8 * ib32 + 6],
                    qs[8 * ib32 + 7],
                ]);
                let db = d * (0.5 + (aux1 >> 28) as f32) * 0.25;
                let aux8 = aux0.to_le_bytes();
                for l in 0..4 {
                    let grid_idx = aux8[l] as usize;
                    let gf = u8x8_to_f32x4x2(vld1_u8(grid_bytes.as_ptr().add(grid_idx * 8)));
                    let sf = iq_signs8_to_f32x4x2(
                        KSIGNS_IQ2XS[((aux1 >> (7 * l)) & 127) as usize] as u8,
                    );
                    let x_off = ib32 * 32 + l * 8;
                    let coef_lo = vmulq_n_f32(vmulq_f32(gf[0], sf[0]), db);
                    let coef_hi = vmulq_n_f32(vmulq_f32(gf[1], sf[1]), db);
                    acc0 = vfmaq_f32(acc0, coef_lo, vld1q_f32(xptr.add(x_off)));
                    acc1 = vfmaq_f32(acc1, coef_hi, vld1q_f32(xptr.add(x_off + 4)));
                }
            }
        }
        out[i] = vaddvq_f32(vaddq_f32(acc0, acc1));
    }
}

/// AArch64 NEON IQ2_XS matvec. Grid index = low 9 bits of each `qs` u16,
/// sign index = high 7 bits → `KSIGNS_IQ2XS`; per-sub-block low/high nibble
/// scales. Mirrors [`matvec_iq2_xs_w_f32_a_scalar`].
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn matvec_iq2_xs_w_f32_a_neon(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use rustllama_gguf::dequant::{IQ2XS_GRID, KSIGNS_IQ2XS};
    use std::arch::aarch64::*;
    const BLOCK_BYTES: usize = 74;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;
    let grid_bytes: &[u8] = bytemuck::cast_slice(&IQ2XS_GRID);
    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d = f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32();
            let qs = &w_bytes[off + 2..off + 2 + 64];
            let scales = &w_bytes[off + 2 + 64..off + 2 + 64 + 8];
            let xptr = x.as_ptr().add(b * QK_K);
            for ib32 in 0..8 {
                let scale_byte = scales[ib32];
                let db_lo = d * (0.5 + (scale_byte & 0x0F) as f32) * 0.25;
                let db_hi = d * (0.5 + (scale_byte >> 4) as f32) * 0.25;
                for l in 0..4 {
                    let q = u16::from_le_bytes([qs[8 * ib32 + 2 * l], qs[8 * ib32 + 2 * l + 1]]);
                    let grid_idx = (q & 511) as usize;
                    let sign_idx = (q >> 9) as usize;
                    let gf = u8x8_to_f32x4x2(vld1_u8(grid_bytes.as_ptr().add(grid_idx * 8)));
                    let sf = iq_signs8_to_f32x4x2(KSIGNS_IQ2XS[sign_idx] as u8);
                    let db = if l < 2 { db_lo } else { db_hi };
                    let x_off = ib32 * 32 + l * 8;
                    let coef_lo = vmulq_n_f32(vmulq_f32(gf[0], sf[0]), db);
                    let coef_hi = vmulq_n_f32(vmulq_f32(gf[1], sf[1]), db);
                    acc0 = vfmaq_f32(acc0, coef_lo, vld1q_f32(xptr.add(x_off)));
                    acc1 = vfmaq_f32(acc1, coef_hi, vld1q_f32(xptr.add(x_off + 4)));
                }
            }
        }
        out[i] = vaddvq_f32(vaddq_f32(acc0, acc1));
    }
}

/// AArch64 NEON IQ2_S matvec. 10-bit grid index (`qs_lo` byte + 2 high bits
/// from `qh`), explicit per-run sign byte (no `KSIGNS` indirection). Mirrors
/// [`matvec_iq2_s_w_f32_a_scalar`].
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn matvec_iq2_s_w_f32_a_neon(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use rustllama_gguf::dequant::IQ2S_GRID;
    use std::arch::aarch64::*;
    const BLOCK_BYTES: usize = 82;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;
    let grid_bytes: &[u8] = bytemuck::cast_slice(&IQ2S_GRID);
    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d = f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32();
            let qs_lo = &w_bytes[off + 2..off + 2 + 32];
            let signs = &w_bytes[off + 2 + 32..off + 2 + 64];
            let qh = &w_bytes[off + 2 + 64..off + 2 + 64 + 8];
            let scales = &w_bytes[off + 2 + 64 + 8..off + 2 + 64 + 8 + 8];
            let xptr = x.as_ptr().add(b * QK_K);
            for ib32 in 0..8 {
                let scale_byte = scales[ib32];
                let db_lo = d * (0.5 + (scale_byte & 0x0F) as f32) * 0.25;
                let db_hi = d * (0.5 + (scale_byte >> 4) as f32) * 0.25;
                let qs_off = ib32 * 4;
                let qh_byte = qh[ib32];
                for l in 0..4 {
                    let high_bits = ((qh_byte as usize) << (8 - 2 * l)) & 0x300;
                    let grid_idx = (qs_lo[qs_off + l] as usize) | high_bits;
                    let gf = u8x8_to_f32x4x2(vld1_u8(grid_bytes.as_ptr().add(grid_idx * 8)));
                    let sf = iq_signs8_to_f32x4x2(signs[qs_off + l]);
                    let db = if l < 2 { db_lo } else { db_hi };
                    let x_off = ib32 * 32 + l * 8;
                    let coef_lo = vmulq_n_f32(vmulq_f32(gf[0], sf[0]), db);
                    let coef_hi = vmulq_n_f32(vmulq_f32(gf[1], sf[1]), db);
                    acc0 = vfmaq_f32(acc0, coef_lo, vld1q_f32(xptr.add(x_off)));
                    acc1 = vfmaq_f32(acc1, coef_hi, vld1q_f32(xptr.add(x_off + 4)));
                }
            }
        }
        out[i] = vaddvq_f32(vaddq_f32(acc0, acc1));
    }
}

/// AArch64 NEON IQ3_XXS matvec. 4-byte `u8` grid points, two indices per
/// run combined into one 8-lane vector (grid1 → lanes 0..4, grid2 →
/// 4..8); `KSIGNS_IQ2XS` sign expansion. Mirrors
/// [`matvec_iq3_xxs_w_f32_a_scalar`].
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn matvec_iq3_xxs_w_f32_a_neon(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use rustllama_gguf::dequant::{IQ3XXS_GRID, KSIGNS_IQ2XS};
    use std::arch::aarch64::*;
    const BLOCK_BYTES: usize = 98;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;
    let grid_bytes: &[u8] = bytemuck::cast_slice(&IQ3XXS_GRID);
    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d = f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32();
            let qs_grid = &w_bytes[off + 2..off + 2 + 64];
            let qs_sas = &w_bytes[off + 2 + 64..off + 2 + 96];
            let xptr = x.as_ptr().add(b * QK_K);
            for ib32 in 0..8 {
                let aux32 = u32::from_le_bytes([
                    qs_sas[4 * ib32],
                    qs_sas[4 * ib32 + 1],
                    qs_sas[4 * ib32 + 2],
                    qs_sas[4 * ib32 + 3],
                ]);
                let db = d * (0.5 + (aux32 >> 28) as f32) * 0.5;
                let qs_off = 8 * ib32;
                for l in 0..4 {
                    let grid1_idx = qs_grid[qs_off + 2 * l] as usize;
                    let grid2_idx = qs_grid[qs_off + 2 * l + 1] as usize;
                    let mut buf = [0u8; 8];
                    buf[0..4].copy_from_slice(&grid_bytes[grid1_idx * 4..grid1_idx * 4 + 4]);
                    buf[4..8].copy_from_slice(&grid_bytes[grid2_idx * 4..grid2_idx * 4 + 4]);
                    let gf = u8x8_to_f32x4x2(vld1_u8(buf.as_ptr()));
                    let sf = iq_signs8_to_f32x4x2(
                        KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize] as u8,
                    );
                    let x_off = ib32 * 32 + l * 8;
                    let coef_lo = vmulq_n_f32(vmulq_f32(gf[0], sf[0]), db);
                    let coef_hi = vmulq_n_f32(vmulq_f32(gf[1], sf[1]), db);
                    acc0 = vfmaq_f32(acc0, coef_lo, vld1q_f32(xptr.add(x_off)));
                    acc1 = vfmaq_f32(acc1, coef_hi, vld1q_f32(xptr.add(x_off + 4)));
                }
            }
        }
        out[i] = vaddvq_f32(vaddq_f32(acc0, acc1));
    }
}

/// AArch64 NEON IQ3_S matvec. 9-bit grid index (`qs` byte + 1 high bit from
/// `qh`), explicit per-run sign byte; `pair`/`sub` scale structure as the
/// scalar path. grid1 → lanes 0..4, grid2 → 4..8. Mirrors
/// [`matvec_iq3_s_w_f32_a_scalar`].
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn matvec_iq3_s_w_f32_a_neon(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use rustllama_gguf::dequant::IQ3S_GRID;
    use std::arch::aarch64::*;
    const BLOCK_BYTES: usize = 110;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;
    let grid_bytes: &[u8] = bytemuck::cast_slice(&IQ3S_GRID);
    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d = f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32();
            let qs = &w_bytes[off + 2..off + 2 + 64];
            let qh = &w_bytes[off + 2 + 64..off + 2 + 64 + 8];
            let signs = &w_bytes[off + 2 + 64 + 8..off + 2 + 64 + 8 + 32];
            let scales = &w_bytes[off + 2 + 64 + 8 + 32..off + 2 + 64 + 8 + 32 + 4];
            let xptr = x.as_ptr().add(b * QK_K);
            let mut qs_cur = 0usize;
            let mut signs_cur = 0usize;
            for pair in 0..4 {
                let ib32 = pair * 2;
                let scale_byte = scales[pair];
                let db1 = d * (1.0 + 2.0 * ((scale_byte & 0x0F) as f32));
                let db2 = d * (1.0 + 2.0 * ((scale_byte >> 4) as f32));
                let x_off1 = ib32 * 32;
                let x_off2 = (ib32 + 1) * 32;
                let qh_byte1 = qh[ib32];
                let qh_byte2 = qh[ib32 + 1];
                for l in 0..4 {
                    let g1_idx = qs[qs_cur + 2 * l] as usize
                        | (((qh_byte1 as usize) << (8 - 2 * l)) & 0x100);
                    let g2_idx = qs[qs_cur + 2 * l + 1] as usize
                        | (((qh_byte1 as usize) << (7 - 2 * l)) & 0x100);
                    let mut buf = [0u8; 8];
                    buf[0..4].copy_from_slice(&grid_bytes[g1_idx * 4..g1_idx * 4 + 4]);
                    buf[4..8].copy_from_slice(&grid_bytes[g2_idx * 4..g2_idx * 4 + 4]);
                    let gf = u8x8_to_f32x4x2(vld1_u8(buf.as_ptr()));
                    let sf = iq_signs8_to_f32x4x2(signs[signs_cur + l]);
                    let xo = x_off1 + l * 8;
                    let coef_lo = vmulq_n_f32(vmulq_f32(gf[0], sf[0]), db1);
                    let coef_hi = vmulq_n_f32(vmulq_f32(gf[1], sf[1]), db1);
                    acc0 = vfmaq_f32(acc0, coef_lo, vld1q_f32(xptr.add(xo)));
                    acc1 = vfmaq_f32(acc1, coef_hi, vld1q_f32(xptr.add(xo + 4)));
                }
                qs_cur += 8;
                signs_cur += 4;
                for l in 0..4 {
                    let g1_idx = qs[qs_cur + 2 * l] as usize
                        | (((qh_byte2 as usize) << (8 - 2 * l)) & 0x100);
                    let g2_idx = qs[qs_cur + 2 * l + 1] as usize
                        | (((qh_byte2 as usize) << (7 - 2 * l)) & 0x100);
                    let mut buf = [0u8; 8];
                    buf[0..4].copy_from_slice(&grid_bytes[g1_idx * 4..g1_idx * 4 + 4]);
                    buf[4..8].copy_from_slice(&grid_bytes[g2_idx * 4..g2_idx * 4 + 4]);
                    let gf = u8x8_to_f32x4x2(vld1_u8(buf.as_ptr()));
                    let sf = iq_signs8_to_f32x4x2(signs[signs_cur + l]);
                    let xo = x_off2 + l * 8;
                    let coef_lo = vmulq_n_f32(vmulq_f32(gf[0], sf[0]), db2);
                    let coef_hi = vmulq_n_f32(vmulq_f32(gf[1], sf[1]), db2);
                    acc0 = vfmaq_f32(acc0, coef_lo, vld1q_f32(xptr.add(xo)));
                    acc1 = vfmaq_f32(acc1, coef_hi, vld1q_f32(xptr.add(xo + 4)));
                }
                qs_cur += 8;
                signs_cur += 4;
            }
        }
        out[i] = vaddvq_f32(vaddq_f32(acc0, acc1));
    }
}

/// AArch64 NEON IQ1_S matvec. The IQ1 grid stores signed `i8` points; each
/// 8-point run is `dl·(grid + delta)·x`, with the per-sub-block `delta`
/// broadcast and added before the FMA. Mirrors
/// [`matvec_iq1_s_w_f32_a_scalar`].
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn matvec_iq1_s_w_f32_a_neon(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use rustllama_gguf::dequant::IQ1S_DELTA;
    use rustllama_gguf::iq1_grid::IQ1S_GRID;
    use std::arch::aarch64::*;
    const BLOCK_BYTES: usize = 50;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;
    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d = f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32();
            let qs = &w_bytes[off + 2..off + 2 + 32];
            let qh_bytes = &w_bytes[off + 34..off + 34 + 16];
            let xptr = x.as_ptr().add(b * QK_K);
            let mut x_off = 0usize;
            for ib32 in 0..8 {
                let qh = u16::from_le_bytes([qh_bytes[ib32 * 2], qh_bytes[ib32 * 2 + 1]]);
                let dl = d * (2.0 * ((qh >> 12) & 7) as f32 + 1.0);
                let delta = if qh & 0x8000 != 0 {
                    -1.0 - IQ1S_DELTA
                } else {
                    -1.0 + IQ1S_DELTA
                };
                let delta_v = vdupq_n_f32(delta);
                for l in 0..4 {
                    let idx = qs[4 * ib32 + l] as usize | ((((qh >> (3 * l)) & 7) as usize) << 8);
                    let grid = IQ1S_GRID[idx].to_le_bytes();
                    let gf = s8x8_to_f32x4x2(vld1_s8(grid.as_ptr() as *const i8));
                    let coef_lo = vmulq_n_f32(vaddq_f32(gf[0], delta_v), dl);
                    let coef_hi = vmulq_n_f32(vaddq_f32(gf[1], delta_v), dl);
                    acc0 = vfmaq_f32(acc0, coef_lo, vld1q_f32(xptr.add(x_off + 8 * l)));
                    acc1 = vfmaq_f32(acc1, coef_hi, vld1q_f32(xptr.add(x_off + 8 * l + 4)));
                }
                x_off += 32;
            }
        }
        out[i] = vaddvq_f32(vaddq_f32(acc0, acc1));
    }
}

/// AArch64 NEON IQ1_M matvec. Like IQ1_S but the super-block scale `d` is
/// reassembled from the four packed scale words and each 8-point lane
/// carries its own `(dl, delta)`. Mirrors [`matvec_iq1_m_w_f32_a_scalar`].
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn matvec_iq1_m_w_f32_a_neon(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use rustllama_gguf::dequant::IQ1S_DELTA;
    use rustllama_gguf::iq1_grid::IQ1S_GRID;
    use std::arch::aarch64::*;
    const BLOCK_BYTES: usize = 56;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;
    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let qs = &w_bytes[off..off + 32];
            let qh = &w_bytes[off + 32..off + 32 + 16];
            let scales_bytes = &w_bytes[off + 48..off + 48 + 8];
            let mut sc = [0u16; 4];
            for ii in 0..4 {
                sc[ii] = u16::from_le_bytes([scales_bytes[ii * 2], scales_bytes[ii * 2 + 1]]);
            }
            let d_bits: u16 = (sc[0] >> 12)
                | ((sc[1] >> 8) & 0x00F0)
                | ((sc[2] >> 4) & 0x0F00)
                | (sc[3] & 0xF000);
            let d = f16::from_bits(d_bits).to_f32();
            let xptr = x.as_ptr().add(b * QK_K);
            let mut x_off = 0usize;
            for ib in 0..8 {
                let s_word = sc[ib / 2];
                let shift0 = 6 * (ib % 2);
                let shift1 = 6 * (ib % 2) + 3;
                let dl1 = d * (2.0 * ((s_word >> shift0) & 0x7) as f32 + 1.0);
                let dl2 = d * (2.0 * ((s_word >> shift1) & 0x7) as f32 + 1.0);
                let qh0 = qh[ib * 2];
                let qh1 = qh[ib * 2 + 1];
                let delta = |bit_set: bool| {
                    if bit_set {
                        -1.0 - IQ1S_DELTA
                    } else {
                        -1.0 + IQ1S_DELTA
                    }
                };
                let qs_chunk = &qs[ib * 4..ib * 4 + 4];
                let lanes = [
                    (dl1, delta(qh0 & 0x08 != 0), qs_chunk[0] as usize | (((qh0 & 0x07) as usize) << 8)),
                    (dl1, delta(qh0 & 0x80 != 0), qs_chunk[1] as usize | ((((qh0 >> 4) & 0x07) as usize) << 8)),
                    (dl2, delta(qh1 & 0x08 != 0), qs_chunk[2] as usize | (((qh1 & 0x07) as usize) << 8)),
                    (dl2, delta(qh1 & 0x80 != 0), qs_chunk[3] as usize | ((((qh1 >> 4) & 0x07) as usize) << 8)),
                ];
                for (l, (dl, delta_val, idx)) in lanes.iter().copied().enumerate() {
                    let grid = IQ1S_GRID[idx].to_le_bytes();
                    let gf = s8x8_to_f32x4x2(vld1_s8(grid.as_ptr() as *const i8));
                    let delta_v = vdupq_n_f32(delta_val);
                    let coef_lo = vmulq_n_f32(vaddq_f32(gf[0], delta_v), dl);
                    let coef_hi = vmulq_n_f32(vaddq_f32(gf[1], delta_v), dl);
                    acc0 = vfmaq_f32(acc0, coef_lo, vld1q_f32(xptr.add(x_off + 8 * l)));
                    acc1 = vfmaq_f32(acc1, coef_hi, vld1q_f32(xptr.add(x_off + 8 * l + 4)));
                }
                x_off += 32;
            }
        }
        out[i] = vaddvq_f32(vaddq_f32(acc0, acc1));
    }
}

/// IQ3_S weight matvec with on-the-fly dequant. Dispatches to AVX-512
/// (gather-based, 16 weights/iter) → scalar.
///
/// Block layout (110 bytes per 256 weights):
///   { d: f16, qs: [u8; 64], qh: [u8; 8], signs: [u8; 32], scales: [u8; 4] }
///
/// Requires `k % 256 == 0`.
pub fn matvec_iq3_s_w_f32_a(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    const BLOCK_BYTES: usize = 110;
    const QK_K: usize = 256;
    debug_assert_eq!(k % QK_K, 0, "IQ3_S matvec requires k % 256 == 0");
    let blocks_per_row = k / QK_K;
    debug_assert_eq!(w_bytes.len(), m * blocks_per_row * BLOCK_BYTES);
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(out.len(), m);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx2") {
            // SAFETY: runtime feature detection above (gather is AVX2).
            unsafe { matvec_iq3_s_w_f32_a_avx512(w_bytes, x, out, m, k) };
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime feature detection above.
            unsafe { matvec_iq3_s_w_f32_a_avx2(w_bytes, x, out, m, k) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64.
        unsafe { matvec_iq3_s_w_f32_a_neon(w_bytes, x, out, m, k) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    matvec_iq3_s_w_f32_a_scalar(w_bytes, x, out, m, k);
}

fn matvec_iq3_s_w_f32_a_scalar(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use rustllama_gguf::dequant::{IQ3S_GRID, KMASK_IQ2XS};
    const BLOCK_BYTES: usize = 110;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;

    let grid_bytes: &[u8] = bytemuck::cast_slice(&IQ3S_GRID);

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d = f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32();
            let qs = &w_bytes[off + 2..off + 2 + 64];
            let qh = &w_bytes[off + 2 + 64..off + 2 + 64 + 8];
            let signs = &w_bytes[off + 2 + 64 + 8..off + 2 + 64 + 8 + 32];
            let scales = &w_bytes[off + 2 + 64 + 8 + 32..off + 2 + 64 + 8 + 32 + 4];
            let x_block = &x[b * QK_K..(b + 1) * QK_K];

            let mut qs_cur = 0usize;
            let mut signs_cur = 0usize;
            for pair in 0..4 {
                let ib32 = pair * 2;
                let scale_byte = scales[pair];
                let db1 = d * (1.0 + 2.0 * ((scale_byte & 0x0F) as f32));
                let db2 = d * (1.0 + 2.0 * ((scale_byte >> 4) as f32));
                let x_off1 = ib32 * 32;
                let x_off2 = (ib32 + 1) * 32;
                let qh_byte1 = qh[ib32];
                let qh_byte2 = qh[ib32 + 1];

                for l in 0..4 {
                    let g1_idx = qs[qs_cur + 2 * l] as usize
                        | (((qh_byte1 as usize) << (8 - 2 * l)) & 0x100);
                    let g2_idx = qs[qs_cur + 2 * l + 1] as usize
                        | (((qh_byte1 as usize) << (7 - 2 * l)) & 0x100);
                    let g1 = &grid_bytes[g1_idx * 4..g1_idx * 4 + 4];
                    let g2 = &grid_bytes[g2_idx * 4..g2_idx * 4 + 4];
                    let sign_byte = signs[signs_cur + l];
                    let x_lo = &x_block[x_off1 + l * 8..x_off1 + l * 8 + 4];
                    let x_hi = &x_block[x_off1 + l * 8 + 4..x_off1 + l * 8 + 8];
                    for j in 0..4 {
                        let s_lo = if sign_byte & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                        let s_hi = if sign_byte & KMASK_IQ2XS[j + 4] != 0 { -1.0 } else { 1.0 };
                        acc += db1 * (g1[j] as f32) * s_lo * x_lo[j];
                        acc += db1 * (g2[j] as f32) * s_hi * x_hi[j];
                    }
                }
                qs_cur += 8;
                signs_cur += 4;

                for l in 0..4 {
                    let g1_idx = qs[qs_cur + 2 * l] as usize
                        | (((qh_byte2 as usize) << (8 - 2 * l)) & 0x100);
                    let g2_idx = qs[qs_cur + 2 * l + 1] as usize
                        | (((qh_byte2 as usize) << (7 - 2 * l)) & 0x100);
                    let g1 = &grid_bytes[g1_idx * 4..g1_idx * 4 + 4];
                    let g2 = &grid_bytes[g2_idx * 4..g2_idx * 4 + 4];
                    let sign_byte = signs[signs_cur + l];
                    let x_lo = &x_block[x_off2 + l * 8..x_off2 + l * 8 + 4];
                    let x_hi = &x_block[x_off2 + l * 8 + 4..x_off2 + l * 8 + 8];
                    for j in 0..4 {
                        let s_lo = if sign_byte & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                        let s_hi = if sign_byte & KMASK_IQ2XS[j + 4] != 0 { -1.0 } else { 1.0 };
                        acc += db2 * (g1[j] as f32) * s_lo * x_lo[j];
                        acc += db2 * (g2[j] as f32) * s_hi * x_hi[j];
                    }
                }
                qs_cur += 8;
                signs_cur += 4;
            }
        }
        out[i] = acc;
    }
}

/// IQ3_S matvec — AVX-512 path. Strategy per 16-weight half-sub-block:
///
///   - Build 4 9-bit grid indices in an `__m128i` (low byte from `qs`,
///     high bit from `qh`).
///   - `_mm_i32gather_epi32::<4>(grid_ptr_i32, indices)` fetches 4 grid
///     entries (16 packed-i8 grid points = 16 bytes) into `__m128i`.
///   - `_mm512_cvtepu8_epi32` widens those 16 bytes to 16 i32 lanes.
///   - Build a 16-bit k-mask from two sign bytes; `_mm512_mask_sub_epi32`
///     negates lanes where the sign bit is set.
///   - Convert to f32, multiply by the sub-scale `dl` broadcast, FMA
///     into one of 4 round-robin accumulators.
///
/// Two iterations per sub-block (32 weights), 8 sub-blocks per super-block.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f")]
unsafe fn matvec_iq3_s_w_f32_a_avx512(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use rustllama_gguf::dequant::IQ3S_GRID;
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 110;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;
    let grid_i32_ptr = IQ3S_GRID.as_ptr() as *const i32;

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();
        let mut acc2 = _mm512_setzero_ps();
        let mut acc3 = _mm512_setzero_ps();

        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d = f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32();
            let qs = &w_bytes[off + 2..off + 2 + 64];
            let qh = &w_bytes[off + 2 + 64..off + 2 + 64 + 8];
            let signs = &w_bytes[off + 2 + 64 + 8..off + 2 + 64 + 8 + 32];
            let scales = &w_bytes[off + 2 + 64 + 8 + 32..off + 2 + 64 + 8 + 32 + 4];
            let xptr = x.as_ptr().add(b * QK_K);

            let mut qs_cur = 0usize;
            let mut signs_cur = 0usize;
            let mut x_off = 0usize;
            for pair in 0..4 {
                let scale_byte = scales[pair];
                let db1 = d * (1.0 + 2.0 * ((scale_byte & 0x0F) as f32));
                let db2 = d * (1.0 + 2.0 * ((scale_byte >> 4) as f32));

                // Each pair drives 2 ib32 sub-blocks (one with db1, one
                // with db2). Each sub-block is 32 weights = 2 half-sub-
                // blocks of 16 weights each, processed below.
                for sub in 0..2 {
                    let qh_byte = qh[pair * 2 + sub];
                    let db = if sub == 0 { db1 } else { db2 };
                    let db_v = _mm512_set1_ps(db);

                    // Two half-sub-blocks (16 weights each) per sub-block.
                    // h=0 covers l=0,1 (qh bits 0..3, qs[qs_cur+0..4]);
                    // h=1 covers l=2,3 (qh bits 4..7, qs[qs_cur+4..8]).
                    for h in 0..2 {
                        let bit_base = h * 4;
                        let qs_base = qs_cur + h * 4;
                        // Pre-shift the 4 qh bits into bit-8 positions on
                        // the scalar side; cheaper than a lane-wise variable
                        // shift in SIMD.
                        let highs: [i32; 4] = [
                            (((qh_byte >> bit_base) & 1) as i32) << 8,
                            (((qh_byte >> (bit_base + 1)) & 1) as i32) << 8,
                            (((qh_byte >> (bit_base + 2)) & 1) as i32) << 8,
                            (((qh_byte >> (bit_base + 3)) & 1) as i32) << 8,
                        ];
                        let high_v = _mm_loadu_si128(highs.as_ptr() as *const __m128i);
                        let qs_v = _mm_set_epi32(
                            qs[qs_base + 3] as i32,
                            qs[qs_base + 2] as i32,
                            qs[qs_base + 1] as i32,
                            qs[qs_base] as i32,
                        );
                        let idx_v = _mm_or_si128(qs_v, high_v);
                        // Gather 4 codebook entries; each i32 is 4 packed
                        // i8 grid points.
                        let packed = _mm_i32gather_epi32::<4>(grid_i32_ptr, idx_v);
                        // 16 i8 → 16 i32 (lifted as u8 / zero-extended;
                        // grid values are small non-negative i8).
                        let grid_v = _mm512_cvtepu8_epi32(packed);

                        // Build sign mask: 2 sign bytes → 16-bit k-mask.
                        // Bit `j` of signs[signs_cur+sb] controls lane sb*8+j.
                        let sb_lo = signs[signs_cur + h * 2] as u16;
                        let sb_hi = signs[signs_cur + h * 2 + 1] as u16;
                        let sign_mask: __mmask16 = sb_lo | (sb_hi << 8);
                        let grid_signed = _mm512_mask_sub_epi32(
                            grid_v,
                            sign_mask,
                            _mm512_setzero_si512(),
                            grid_v,
                        );
                        let grid_f = _mm512_cvtepi32_ps(grid_signed);
                        let term = _mm512_mul_ps(grid_f, db_v);
                        let x_v = _mm512_loadu_ps(xptr.add(x_off));
                        let slot = (pair + sub * 2 + h) & 3;
                        match slot {
                            0 => acc0 = _mm512_fmadd_ps(term, x_v, acc0),
                            1 => acc1 = _mm512_fmadd_ps(term, x_v, acc1),
                            2 => acc2 = _mm512_fmadd_ps(term, x_v, acc2),
                            _ => acc3 = _mm512_fmadd_ps(term, x_v, acc3),
                        }
                        x_off += 16;
                    }
                    qs_cur += 8;
                    signs_cur += 4;
                }
            }
        }
        let total = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
        out[i] = _mm512_reduce_add_ps(total);
    }
}

/// IQ3_S matvec — AVX2 path. 8 weights per YMM iteration (vs the
/// AVX-512 path's 16). Same gather + cvtepu8 + sign-blendv recipe;
/// sign application uses `vpcmpgtd` (per-lane cmpgt → byte-mask) +
/// `vpblendvb` instead of AVX-512's k-mask sub.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn matvec_iq3_s_w_f32_a_avx2(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use rustllama_gguf::dequant::IQ3S_GRID;
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 110;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;
    let grid_i32_ptr = IQ3S_GRID.as_ptr() as *const i32;
    let bit_mask = _mm256_set_epi32(128, 64, 32, 16, 8, 4, 2, 1);
    let zero_i = _mm256_setzero_si256();

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = [_mm256_setzero_ps(); 4];

        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d = f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32();
            let qs = &w_bytes[off + 2..off + 2 + 64];
            let qh = &w_bytes[off + 2 + 64..off + 2 + 64 + 8];
            let signs = &w_bytes[off + 2 + 64 + 8..off + 2 + 64 + 8 + 32];
            let scales = &w_bytes[off + 2 + 64 + 8 + 32..off + 2 + 64 + 8 + 32 + 4];
            let xptr = x.as_ptr().add(b * QK_K);

            let mut qs_cur = 0usize;
            let mut signs_cur = 0usize;
            let mut x_off = 0usize;
            for pair in 0..4 {
                let scale_byte = scales[pair];
                let db1 = d * (1.0 + 2.0 * ((scale_byte & 0x0F) as f32));
                let db2 = d * (1.0 + 2.0 * ((scale_byte >> 4) as f32));

                for sub in 0..2 {
                    let qh_byte = qh[pair * 2 + sub];
                    let db = if sub == 0 { db1 } else { db2 };
                    let db_v = _mm256_set1_ps(db);

                    for h in 0..2 {
                        let bit_base = h * 4;
                        let qs_base = qs_cur + h * 4;
                        // 4 codebook lookups (16 weights) per half-sub-
                        // block — split into 2 YMM iterations of 8
                        // weights (= 2 codebook entries) each.
                        for ch in 0..2 {
                            let lookup_base = ch * 2;
                            let highs: [i32; 2] = [
                                (((qh_byte >> (bit_base + lookup_base)) & 1) as i32) << 8,
                                (((qh_byte >> (bit_base + lookup_base + 1)) & 1) as i32) << 8,
                            ];
                            let idx_v = _mm_set_epi32(
                                0,
                                0,
                                (qs[qs_base + lookup_base + 1] as i32) | highs[1],
                                (qs[qs_base + lookup_base] as i32) | highs[0],
                            );
                            // Gather 2 codebook entries (mask the upper
                            // 2 lanes off so we don't read past the
                            // grid in pathological cases). Each entry
                            // is an i32 = 4 packed-i8 grid points.
                            let packed = _mm_mask_i32gather_epi32::<4>(
                                _mm_setzero_si128(),
                                grid_i32_ptr,
                                idx_v,
                                _mm_set_epi32(0, 0, -1, -1),
                            );
                            // Low 8 bytes → 8 i32 lanes.
                            let grid_v = _mm256_cvtepu8_epi32(packed);

                            // Sign byte: one byte for these 8 weights.
                            // Bit j of the byte controls lane j.
                            let sign_byte = signs[signs_cur + h * 2 + ch] as i32;
                            let sb_v = _mm256_set1_epi32(sign_byte);
                            let masked = _mm256_and_si256(sb_v, bit_mask);
                            let is_set = _mm256_cmpgt_epi32(masked, zero_i);
                            let negated = _mm256_sub_epi32(zero_i, grid_v);
                            let grid_signed = _mm256_blendv_epi8(grid_v, negated, is_set);

                            let grid_f = _mm256_cvtepi32_ps(grid_signed);
                            let term = _mm256_mul_ps(grid_f, db_v);
                            let x_v = _mm256_loadu_ps(xptr.add(x_off));
                            let slot = (pair + sub * 2 + h + ch) & 3;
                            acc[slot] = _mm256_fmadd_ps(term, x_v, acc[slot]);
                            x_off += 8;
                        }
                    }
                    qs_cur += 8;
                    signs_cur += 4;
                }
            }
        }
        // Reduce 4 YMMs → scalar.
        let s01 = _mm256_add_ps(acc[0], acc[1]);
        let s23 = _mm256_add_ps(acc[2], acc[3]);
        let total = _mm256_add_ps(s01, s23);
        let lo = _mm256_castps256_ps128(total);
        let hi = _mm256_extractf128_ps(total, 1);
        let sum128 = _mm_add_ps(lo, hi);
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0b01));
        out[i] = _mm_cvtss_f32(sum32);
    }
}

/// IQ2_XXS weight matvec with on-the-fly dequant. Scalar-only — the
/// codebook + sign-index indirection has the same SIMD-unfriendliness
/// as IQ3_S, and an extra `KSIGNS_IQ2XS` lookup on top.
///
/// Block layout (66 bytes per 256 weights):
///   { d: f16, qs: [u16; 32] }
///
/// Sub-block decode is described in [`rustllama_gguf::dequant::dequant_iq2_xxs`].
/// Dispatches AVX-512 → scalar.
///
/// Requires `k % 256 == 0`.
pub fn matvec_iq2_xxs_w_f32_a(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    const BLOCK_BYTES: usize = 66;
    const QK_K: usize = 256;
    debug_assert_eq!(k % QK_K, 0, "IQ2_XXS matvec requires k % 256 == 0");
    let blocks_per_row = k / QK_K;
    debug_assert_eq!(w_bytes.len(), m * blocks_per_row * BLOCK_BYTES);
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(out.len(), m);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx2") {
            // SAFETY: runtime feature detection above.
            unsafe { matvec_iq2_xxs_w_f32_a_avx512(w_bytes, x, out, m, k) };
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime feature detection above.
            unsafe { matvec_iq2_xxs_w_f32_a_avx2(w_bytes, x, out, m, k) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64.
        unsafe { matvec_iq2_xxs_w_f32_a_neon(w_bytes, x, out, m, k) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    matvec_iq2_xxs_w_f32_a_scalar(w_bytes, x, out, m, k);
}

fn matvec_iq2_xxs_w_f32_a_scalar(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use rustllama_gguf::dequant::{IQ2XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
    const BLOCK_BYTES: usize = 66;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;

    let grid_bytes: &[u8] = bytemuck::cast_slice(&IQ2XXS_GRID);

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d = f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32();
            let qs = &w_bytes[off + 2..off + 2 + 64];
            let x_block = &x[b * QK_K..(b + 1) * QK_K];

            for ib32 in 0..8 {
                let aux0 = u32::from_le_bytes([
                    qs[8 * ib32],
                    qs[8 * ib32 + 1],
                    qs[8 * ib32 + 2],
                    qs[8 * ib32 + 3],
                ]);
                let aux1 = u32::from_le_bytes([
                    qs[8 * ib32 + 4],
                    qs[8 * ib32 + 5],
                    qs[8 * ib32 + 6],
                    qs[8 * ib32 + 7],
                ]);
                let db = d * (0.5 + (aux1 >> 28) as f32) * 0.25;
                let aux8 = aux0.to_le_bytes();
                for l in 0..4 {
                    let grid_idx = aux8[l] as usize;
                    let grid = &grid_bytes[grid_idx * 8..grid_idx * 8 + 8];
                    let signs = KSIGNS_IQ2XS[((aux1 >> (7 * l)) & 127) as usize];
                    let x_off = ib32 * 32 + l * 8;
                    for j in 0..8 {
                        let s = if signs & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                        acc += db * (grid[j] as f32) * s * x_block[x_off + j];
                    }
                }
            }
        }
        out[i] = acc;
    }
}

/// IQ3_XXS weight matvec with on-the-fly dequant. Dispatches AVX-512
/// → AVX2 → scalar by runtime feature detection.
///
/// Block layout (98 bytes per 256 weights):
///   { d: f16, qs: [u8; 96] }
/// where `qs[0..64]` are 256 × 8-bit grid indices and `qs[64..96]` are
/// 8 little-endian u32 words — one per 32-weight sub-block — each
/// packing the sub-block scale (high 4 bits) and four 7-bit sign-table
/// indices (low 28 bits).
///
/// Requires `k % 256 == 0`.
pub fn matvec_iq3_xxs_w_f32_a(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    const BLOCK_BYTES: usize = 98;
    const QK_K: usize = 256;
    debug_assert_eq!(k % QK_K, 0, "IQ3_XXS matvec requires k % 256 == 0");
    let blocks_per_row = k / QK_K;
    debug_assert_eq!(w_bytes.len(), m * blocks_per_row * BLOCK_BYTES);
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(out.len(), m);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx2") {
            // SAFETY: runtime feature detection above.
            unsafe { matvec_iq3_xxs_w_f32_a_avx512(w_bytes, x, out, m, k) };
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime feature detection above.
            unsafe { matvec_iq3_xxs_w_f32_a_avx2(w_bytes, x, out, m, k) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64.
        unsafe { matvec_iq3_xxs_w_f32_a_neon(w_bytes, x, out, m, k) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    matvec_iq3_xxs_w_f32_a_scalar(w_bytes, x, out, m, k);
}

fn matvec_iq3_xxs_w_f32_a_scalar(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use rustllama_gguf::dequant::{IQ3XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
    const BLOCK_BYTES: usize = 98;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;

    let grid_bytes: &[u8] = bytemuck::cast_slice(&IQ3XXS_GRID);

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d = f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32();
            // qs region: 96 bytes split into grid_indices[0..64] +
            // scales_and_signs[64..96] (eight u32 words).
            let qs_grid = &w_bytes[off + 2..off + 2 + 64];
            let qs_sas = &w_bytes[off + 2 + 64..off + 2 + 96];
            let x_block = &x[b * QK_K..(b + 1) * QK_K];

            for ib32 in 0..8 {
                let aux32 = u32::from_le_bytes([
                    qs_sas[4 * ib32],
                    qs_sas[4 * ib32 + 1],
                    qs_sas[4 * ib32 + 2],
                    qs_sas[4 * ib32 + 3],
                ]);
                let db = d * (0.5 + (aux32 >> 28) as f32) * 0.5;
                let qs_off = 8 * ib32;
                for l in 0..4 {
                    let grid1_idx = qs_grid[qs_off + 2 * l] as usize;
                    let grid2_idx = qs_grid[qs_off + 2 * l + 1] as usize;
                    let grid1 = &grid_bytes[grid1_idx * 4..grid1_idx * 4 + 4];
                    let grid2 = &grid_bytes[grid2_idx * 4..grid2_idx * 4 + 4];
                    let signs = KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize];
                    let x_off = ib32 * 32 + l * 8;
                    for j in 0..4 {
                        let s_lo = if signs & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                        let s_hi = if signs & KMASK_IQ2XS[j + 4] != 0 { -1.0 } else { 1.0 };
                        acc += db * (grid1[j] as f32) * s_lo * x_block[x_off + j];
                        acc += db * (grid2[j] as f32) * s_hi * x_block[x_off + j + 4];
                    }
                }
            }
        }
        out[i] = acc;
    }
}

/// IQ3_XXS matvec — AVX-512 path. Same `gather + cvtepu8 + k-mask-sub`
/// shape as IQ2_XXS, but the grid is `[u32; …]` (4-byte entries) instead
/// of `[u64; …]`, so we gather 8 grid entries per ib32 sub-block via
/// `i32gather_epi32` and pack them into two ZMM lanes (lo16 + hi16).
/// Sign-byte mapping is identical to IQ2_XXS: 4 sign indices per ib32,
/// each KSIGNS_IQ2XS lookup feeding 8 lanes of the per-half mask.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f")]
unsafe fn matvec_iq3_xxs_w_f32_a_avx512(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use rustllama_gguf::dequant::{IQ3XXS_GRID, KSIGNS_IQ2XS};
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 98;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;
    let grid_i32_ptr = IQ3XXS_GRID.as_ptr() as *const i32;

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();
        let mut acc2 = _mm512_setzero_ps();
        let mut acc3 = _mm512_setzero_ps();

        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d = f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32();
            let qs_grid = &w_bytes[off + 2..off + 2 + 64];
            let qs_sas = &w_bytes[off + 2 + 64..off + 2 + 96];
            let xptr = x.as_ptr().add(b * QK_K);

            for ib32 in 0..8 {
                let aux32 = u32::from_le_bytes([
                    qs_sas[4 * ib32],
                    qs_sas[4 * ib32 + 1],
                    qs_sas[4 * ib32 + 2],
                    qs_sas[4 * ib32 + 3],
                ]);
                let db = d * (0.5 + (aux32 >> 28) as f32) * 0.5;
                let db_v = _mm512_set1_ps(db);

                let g = &qs_grid[8 * ib32..8 * ib32 + 8];
                let idx_v = _mm256_set_epi32(
                    g[7] as i32, g[6] as i32, g[5] as i32, g[4] as i32,
                    g[3] as i32, g[2] as i32, g[1] as i32, g[0] as i32,
                );
                let s0 = KSIGNS_IQ2XS[(aux32 & 127) as usize] as u16;
                let s1 = KSIGNS_IQ2XS[((aux32 >> 7) & 127) as usize] as u16;
                let s2 = KSIGNS_IQ2XS[((aux32 >> 14) & 127) as usize] as u16;
                let s3 = KSIGNS_IQ2XS[((aux32 >> 21) & 127) as usize] as u16;
                let mask_lo: __mmask16 = s0 | (s1 << 8);
                let mask_hi: __mmask16 = s2 | (s3 << 8);

                let packed = _mm256_i32gather_epi32::<4>(grid_i32_ptr, idx_v);
                let lo_half = _mm256_extracti128_si256::<0>(packed);
                let hi_half = _mm256_extracti128_si256::<1>(packed);
                let grid_lo = _mm512_cvtepu8_epi32(lo_half);
                let grid_hi = _mm512_cvtepu8_epi32(hi_half);

                let zero = _mm512_setzero_si512();
                let grid_lo_s = _mm512_mask_sub_epi32(grid_lo, mask_lo, zero, grid_lo);
                let grid_hi_s = _mm512_mask_sub_epi32(grid_hi, mask_hi, zero, grid_hi);

                let term_lo = _mm512_mul_ps(_mm512_cvtepi32_ps(grid_lo_s), db_v);
                let term_hi = _mm512_mul_ps(_mm512_cvtepi32_ps(grid_hi_s), db_v);

                let x_v_lo = _mm512_loadu_ps(xptr.add(ib32 * 32));
                let x_v_hi = _mm512_loadu_ps(xptr.add(ib32 * 32 + 16));

                // Alternate among 4 accumulators to keep the FP pipeline
                // dense across the 8 ib32 sub-blocks (same trick the
                // IQ2_XXS path uses).
                let slot = ib32 & 3;
                match slot {
                    0 => {
                        acc0 = _mm512_fmadd_ps(term_lo, x_v_lo, acc0);
                        acc1 = _mm512_fmadd_ps(term_hi, x_v_hi, acc1);
                    }
                    1 => {
                        acc2 = _mm512_fmadd_ps(term_lo, x_v_lo, acc2);
                        acc3 = _mm512_fmadd_ps(term_hi, x_v_hi, acc3);
                    }
                    2 => {
                        acc0 = _mm512_fmadd_ps(term_lo, x_v_lo, acc0);
                        acc1 = _mm512_fmadd_ps(term_hi, x_v_hi, acc1);
                    }
                    _ => {
                        acc2 = _mm512_fmadd_ps(term_lo, x_v_lo, acc2);
                        acc3 = _mm512_fmadd_ps(term_hi, x_v_hi, acc3);
                    }
                }
            }
        }
        let total = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
        out[i] = _mm512_reduce_add_ps(total);
    }
}

/// IQ3_XXS matvec — AVX2 path. Mirrors the IQ2_XXS AVX2 layout: gather
/// 8 grid u32 entries into a `__m256i`, split into four 8-byte chunks,
/// widen each to `__m256i<i32>` via `cvtepu8_epi32`, then negate per
/// chunk using the chunk's sign byte broadcast + the standard 8-bit
/// `(1, 2, 4, 8, 16, 32, 64, 128)` mask.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn matvec_iq3_xxs_w_f32_a_avx2(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use rustllama_gguf::dequant::{IQ3XXS_GRID, KSIGNS_IQ2XS};
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 98;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;
    let grid_i32_ptr = IQ3XXS_GRID.as_ptr() as *const i32;
    let bit_mask = _mm256_set_epi32(128, 64, 32, 16, 8, 4, 2, 1);
    let zero_i = _mm256_setzero_si256();

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = [_mm256_setzero_ps(); 4];

        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d = f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32();
            let qs_grid = &w_bytes[off + 2..off + 2 + 64];
            let qs_sas = &w_bytes[off + 2 + 64..off + 2 + 96];
            let xptr = x.as_ptr().add(b * QK_K);

            for ib32 in 0..8 {
                let aux32 = u32::from_le_bytes([
                    qs_sas[4 * ib32],
                    qs_sas[4 * ib32 + 1],
                    qs_sas[4 * ib32 + 2],
                    qs_sas[4 * ib32 + 3],
                ]);
                let db = d * (0.5 + (aux32 >> 28) as f32) * 0.5;
                let g = &qs_grid[8 * ib32..8 * ib32 + 8];
                let idx_v = _mm256_set_epi32(
                    g[7] as i32, g[6] as i32, g[5] as i32, g[4] as i32,
                    g[3] as i32, g[2] as i32, g[1] as i32, g[0] as i32,
                );
                let sb = [
                    KSIGNS_IQ2XS[(aux32 & 127) as usize] as i32,
                    KSIGNS_IQ2XS[((aux32 >> 7) & 127) as usize] as i32,
                    KSIGNS_IQ2XS[((aux32 >> 14) & 127) as usize] as i32,
                    KSIGNS_IQ2XS[((aux32 >> 21) & 127) as usize] as i32,
                ];
                let packed = _mm256_i32gather_epi32::<4>(grid_i32_ptr, idx_v);
                let lo16 = _mm256_extracti128_si256::<0>(packed);
                let hi16 = _mm256_extracti128_si256::<1>(packed);
                let chunks_bytes: [__m128i; 4] = [
                    lo16,
                    _mm_srli_si128::<8>(lo16),
                    hi16,
                    _mm_srli_si128::<8>(hi16),
                ];
                let db_v = _mm256_set1_ps(db);
                for c in 0..4 {
                    let grid_v = _mm256_cvtepu8_epi32(chunks_bytes[c]);
                    let sb_v = _mm256_set1_epi32(sb[c]);
                    let masked = _mm256_and_si256(sb_v, bit_mask);
                    let is_set = _mm256_cmpgt_epi32(masked, zero_i);
                    let negated = _mm256_sub_epi32(zero_i, grid_v);
                    let grid_signed = _mm256_blendv_epi8(grid_v, negated, is_set);
                    let grid_f = _mm256_cvtepi32_ps(grid_signed);
                    let term = _mm256_mul_ps(grid_f, db_v);
                    let x_v = _mm256_loadu_ps(xptr.add(ib32 * 32 + c * 8));
                    acc[c] = _mm256_fmadd_ps(term, x_v, acc[c]);
                }
            }
        }
        let s01 = _mm256_add_ps(acc[0], acc[1]);
        let s23 = _mm256_add_ps(acc[2], acc[3]);
        let total = _mm256_add_ps(s01, s23);
        let lo = _mm256_castps256_ps128(total);
        let hi = _mm256_extractf128_ps(total, 1);
        let sum128 = _mm_add_ps(lo, hi);
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0b01));
        out[i] = _mm_cvtss_f32(sum32);
    }
}

/// IQ2_XXS matvec — AVX-512 path. Same gather + cvtepu8 + mask-sub
/// pattern as IQ2_S / IQ2_XS, but the index sources differ: per ib32
/// sub-block we read two `u32` words, the first holding 4 packed
/// 8-bit grid indices and the second holding four 7-bit sign-table
/// indices plus a 4-bit sub-scale in its top nibble. One ZMM-iteration
/// per ib32 covers all 32 weights split as 2 × 16; sign bytes from
/// [`KSIGNS_IQ2XS`] become the per-lane k-masks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f")]
unsafe fn matvec_iq2_xxs_w_f32_a_avx512(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use rustllama_gguf::dequant::{IQ2XXS_GRID, KSIGNS_IQ2XS};
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 66;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;
    let grid_i64_ptr = IQ2XXS_GRID.as_ptr() as *const i64;

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();
        let mut acc2 = _mm512_setzero_ps();
        let mut acc3 = _mm512_setzero_ps();

        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d = f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32();
            let qs = &w_bytes[off + 2..off + 2 + 64];
            let xptr = x.as_ptr().add(b * QK_K);

            for ib32 in 0..8 {
                let aux0 = u32::from_le_bytes([
                    qs[8 * ib32],
                    qs[8 * ib32 + 1],
                    qs[8 * ib32 + 2],
                    qs[8 * ib32 + 3],
                ]);
                let aux1 = u32::from_le_bytes([
                    qs[8 * ib32 + 4],
                    qs[8 * ib32 + 5],
                    qs[8 * ib32 + 6],
                    qs[8 * ib32 + 7],
                ]);
                let db = d * (0.5 + (aux1 >> 28) as f32) * 0.25;
                let db_v = _mm512_set1_ps(db);
                let aux8 = aux0.to_le_bytes();
                let idx_v = _mm_set_epi32(
                    aux8[3] as i32,
                    aux8[2] as i32,
                    aux8[1] as i32,
                    aux8[0] as i32,
                );
                let s0 = KSIGNS_IQ2XS[((aux1 >> 0) & 127) as usize] as u16;
                let s1 = KSIGNS_IQ2XS[((aux1 >> 7) & 127) as usize] as u16;
                let s2 = KSIGNS_IQ2XS[((aux1 >> 14) & 127) as usize] as u16;
                let s3 = KSIGNS_IQ2XS[((aux1 >> 21) & 127) as usize] as u16;
                let mask_lo: __mmask16 = s0 | (s1 << 8);
                let mask_hi: __mmask16 = s2 | (s3 << 8);

                let packed = _mm256_i32gather_epi64::<8>(grid_i64_ptr, idx_v);
                let lo_half = _mm256_extracti128_si256::<0>(packed);
                let hi_half = _mm256_extracti128_si256::<1>(packed);
                let grid_lo = _mm512_cvtepu8_epi32(lo_half);
                let grid_hi = _mm512_cvtepu8_epi32(hi_half);

                let zero = _mm512_setzero_si512();
                let grid_lo_s = _mm512_mask_sub_epi32(grid_lo, mask_lo, zero, grid_lo);
                let grid_hi_s = _mm512_mask_sub_epi32(grid_hi, mask_hi, zero, grid_hi);

                let term_lo = _mm512_mul_ps(_mm512_cvtepi32_ps(grid_lo_s), db_v);
                let term_hi = _mm512_mul_ps(_mm512_cvtepi32_ps(grid_hi_s), db_v);

                let x_v_lo = _mm512_loadu_ps(xptr.add(ib32 * 32));
                let x_v_hi = _mm512_loadu_ps(xptr.add(ib32 * 32 + 16));

                let slot = ib32 & 3;
                match slot {
                    0 => {
                        acc0 = _mm512_fmadd_ps(term_lo, x_v_lo, acc0);
                        acc1 = _mm512_fmadd_ps(term_hi, x_v_hi, acc1);
                    }
                    1 => {
                        acc2 = _mm512_fmadd_ps(term_lo, x_v_lo, acc2);
                        acc3 = _mm512_fmadd_ps(term_hi, x_v_hi, acc3);
                    }
                    2 => {
                        acc0 = _mm512_fmadd_ps(term_lo, x_v_lo, acc0);
                        acc1 = _mm512_fmadd_ps(term_hi, x_v_hi, acc1);
                    }
                    _ => {
                        acc2 = _mm512_fmadd_ps(term_lo, x_v_lo, acc2);
                        acc3 = _mm512_fmadd_ps(term_hi, x_v_hi, acc3);
                    }
                }
            }
        }
        let total = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
        out[i] = _mm512_reduce_add_ps(total);
    }
}

/// IQ2_XXS matvec — AVX2 path. Same shape as IQ2_S AVX2: gather 4
/// u64 entries from a `__m128i` of indices built from `aux0`'s bytes,
/// split into 4 × 8-weight chunks. Sign bytes come from the
/// [`KSIGNS_IQ2XS`] table indexed by the 4 7-bit fields of `aux1`;
/// the sub-scale lives in `aux1 >> 28` and is the same `db` for all
/// 32 weights of the ib32 sub-block.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn matvec_iq2_xxs_w_f32_a_avx2(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use rustllama_gguf::dequant::{IQ2XXS_GRID, KSIGNS_IQ2XS};
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 66;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;
    let grid_i64_ptr = IQ2XXS_GRID.as_ptr() as *const i64;
    let bit_mask = _mm256_set_epi32(128, 64, 32, 16, 8, 4, 2, 1);
    let zero_i = _mm256_setzero_si256();

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = [_mm256_setzero_ps(); 4];

        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d = f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32();
            let qs = &w_bytes[off + 2..off + 2 + 64];
            let xptr = x.as_ptr().add(b * QK_K);

            for ib32 in 0..8 {
                let aux0 = u32::from_le_bytes([
                    qs[8 * ib32],
                    qs[8 * ib32 + 1],
                    qs[8 * ib32 + 2],
                    qs[8 * ib32 + 3],
                ]);
                let aux1 = u32::from_le_bytes([
                    qs[8 * ib32 + 4],
                    qs[8 * ib32 + 5],
                    qs[8 * ib32 + 6],
                    qs[8 * ib32 + 7],
                ]);
                let db = d * (0.5 + (aux1 >> 28) as f32) * 0.25;
                let aux8 = aux0.to_le_bytes();
                let idx_v = _mm_set_epi32(
                    aux8[3] as i32,
                    aux8[2] as i32,
                    aux8[1] as i32,
                    aux8[0] as i32,
                );
                let sb = [
                    KSIGNS_IQ2XS[((aux1 >> 0) & 127) as usize] as i32,
                    KSIGNS_IQ2XS[((aux1 >> 7) & 127) as usize] as i32,
                    KSIGNS_IQ2XS[((aux1 >> 14) & 127) as usize] as i32,
                    KSIGNS_IQ2XS[((aux1 >> 21) & 127) as usize] as i32,
                ];
                let packed = _mm256_i32gather_epi64::<8>(grid_i64_ptr, idx_v);
                let lo16 = _mm256_extracti128_si256::<0>(packed);
                let hi16 = _mm256_extracti128_si256::<1>(packed);
                let chunks_bytes: [__m128i; 4] = [
                    lo16,
                    _mm_srli_si128::<8>(lo16),
                    hi16,
                    _mm_srli_si128::<8>(hi16),
                ];
                let db_v = _mm256_set1_ps(db);
                for c in 0..4 {
                    let grid_v = _mm256_cvtepu8_epi32(chunks_bytes[c]);
                    let sb_v = _mm256_set1_epi32(sb[c]);
                    let masked = _mm256_and_si256(sb_v, bit_mask);
                    let is_set = _mm256_cmpgt_epi32(masked, zero_i);
                    let negated = _mm256_sub_epi32(zero_i, grid_v);
                    let grid_signed = _mm256_blendv_epi8(grid_v, negated, is_set);
                    let grid_f = _mm256_cvtepi32_ps(grid_signed);
                    let term = _mm256_mul_ps(grid_f, db_v);
                    let x_v = _mm256_loadu_ps(xptr.add(ib32 * 32 + c * 8));
                    acc[c] = _mm256_fmadd_ps(term, x_v, acc[c]);
                }
            }
        }
        let s01 = _mm256_add_ps(acc[0], acc[1]);
        let s23 = _mm256_add_ps(acc[2], acc[3]);
        let total = _mm256_add_ps(s01, s23);
        let lo = _mm256_castps256_ps128(total);
        let hi = _mm256_extractf128_ps(total, 1);
        let sum128 = _mm_add_ps(lo, hi);
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0b01));
        out[i] = _mm_cvtss_f32(sum32);
    }
}

/// IQ2_XS weight matvec with on-the-fly dequant. Dispatches AVX-512 →
/// scalar.
///
/// Block layout (74 bytes per 256 weights):
///   { d: f16, qs: [u16; 32], scales: [u8; 8] }
///
/// Sub-block decode is described in [`rustllama_gguf::dequant::dequant_iq2_xs`].
///
/// Requires `k % 256 == 0`.
pub fn matvec_iq2_xs_w_f32_a(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    const BLOCK_BYTES: usize = 74;
    const QK_K: usize = 256;
    debug_assert_eq!(k % QK_K, 0, "IQ2_XS matvec requires k % 256 == 0");
    let blocks_per_row = k / QK_K;
    debug_assert_eq!(w_bytes.len(), m * blocks_per_row * BLOCK_BYTES);
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(out.len(), m);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx2") {
            // SAFETY: runtime feature detection above.
            unsafe { matvec_iq2_xs_w_f32_a_avx512(w_bytes, x, out, m, k) };
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime feature detection above.
            unsafe { matvec_iq2_xs_w_f32_a_avx2(w_bytes, x, out, m, k) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64.
        unsafe { matvec_iq2_xs_w_f32_a_neon(w_bytes, x, out, m, k) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    matvec_iq2_xs_w_f32_a_scalar(w_bytes, x, out, m, k);
}

fn matvec_iq2_xs_w_f32_a_scalar(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use rustllama_gguf::dequant::{IQ2XS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
    const BLOCK_BYTES: usize = 74;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;

    let grid_bytes: &[u8] = bytemuck::cast_slice(&IQ2XS_GRID);

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d = f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32();
            let qs = &w_bytes[off + 2..off + 2 + 64];
            let scales = &w_bytes[off + 2 + 64..off + 2 + 64 + 8];
            let x_block = &x[b * QK_K..(b + 1) * QK_K];

            for ib32 in 0..8 {
                let scale_byte = scales[ib32];
                let db_lo = d * (0.5 + (scale_byte & 0x0F) as f32) * 0.25;
                let db_hi = d * (0.5 + (scale_byte >> 4) as f32) * 0.25;
                for l in 0..4 {
                    let q = u16::from_le_bytes([qs[8 * ib32 + 2 * l], qs[8 * ib32 + 2 * l + 1]]);
                    let grid_idx = (q & 511) as usize;
                    let sign_idx = (q >> 9) as usize;
                    let grid = &grid_bytes[grid_idx * 8..grid_idx * 8 + 8];
                    let signs = KSIGNS_IQ2XS[sign_idx];
                    let db = if l < 2 { db_lo } else { db_hi };
                    let x_off = ib32 * 32 + l * 8;
                    for j in 0..8 {
                        let s = if signs & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                        acc += db * (grid[j] as f32) * s * x_block[x_off + j];
                    }
                }
            }
        }
        out[i] = acc;
    }
}

/// IQ2_XS matvec — AVX-512 path. Same gather-based recipe as IQ2_S
/// but the per-element sign indices go through the
/// [`rustllama_gguf::dequant::KSIGNS_IQ2XS`] table (7-bit index →
/// 8-bit pattern) before becoming the k-mask. The grid index is just
/// the low 9 bits of each `qs` u16 — no `qh` indirection.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f")]
unsafe fn matvec_iq2_xs_w_f32_a_avx512(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use rustllama_gguf::dequant::{IQ2XS_GRID, KSIGNS_IQ2XS};
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 74;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;
    let grid_i64_ptr = IQ2XS_GRID.as_ptr() as *const i64;

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();
        let mut acc2 = _mm512_setzero_ps();
        let mut acc3 = _mm512_setzero_ps();

        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d = f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32();
            let qs = &w_bytes[off + 2..off + 2 + 64];
            let scales = &w_bytes[off + 2 + 64..off + 2 + 64 + 8];
            let xptr = x.as_ptr().add(b * QK_K);

            for ib32 in 0..8 {
                let scale_byte = scales[ib32];
                let db_lo = d * (0.5 + (scale_byte & 0x0F) as f32) * 0.25;
                let db_hi = d * (0.5 + (scale_byte >> 4) as f32) * 0.25;
                let base = 8 * ib32;
                let q0 = u16::from_le_bytes([qs[base], qs[base + 1]]);
                let q1 = u16::from_le_bytes([qs[base + 2], qs[base + 3]]);
                let q2 = u16::from_le_bytes([qs[base + 4], qs[base + 5]]);
                let q3 = u16::from_le_bytes([qs[base + 6], qs[base + 7]]);

                let idx_v = _mm_set_epi32(
                    (q3 & 511) as i32,
                    (q2 & 511) as i32,
                    (q1 & 511) as i32,
                    (q0 & 511) as i32,
                );
                let s0 = KSIGNS_IQ2XS[(q0 >> 9) as usize] as u16;
                let s1 = KSIGNS_IQ2XS[(q1 >> 9) as usize] as u16;
                let s2 = KSIGNS_IQ2XS[(q2 >> 9) as usize] as u16;
                let s3 = KSIGNS_IQ2XS[(q3 >> 9) as usize] as u16;
                let mask_lo: __mmask16 = s0 | (s1 << 8);
                let mask_hi: __mmask16 = s2 | (s3 << 8);

                let packed = _mm256_i32gather_epi64::<8>(grid_i64_ptr, idx_v);
                let lo_half = _mm256_extracti128_si256::<0>(packed);
                let hi_half = _mm256_extracti128_si256::<1>(packed);
                let grid_lo = _mm512_cvtepu8_epi32(lo_half);
                let grid_hi = _mm512_cvtepu8_epi32(hi_half);

                let zero = _mm512_setzero_si512();
                let grid_lo_s = _mm512_mask_sub_epi32(grid_lo, mask_lo, zero, grid_lo);
                let grid_hi_s = _mm512_mask_sub_epi32(grid_hi, mask_hi, zero, grid_hi);

                let term_lo =
                    _mm512_mul_ps(_mm512_cvtepi32_ps(grid_lo_s), _mm512_set1_ps(db_lo));
                let term_hi =
                    _mm512_mul_ps(_mm512_cvtepi32_ps(grid_hi_s), _mm512_set1_ps(db_hi));

                let x_v_lo = _mm512_loadu_ps(xptr.add(ib32 * 32));
                let x_v_hi = _mm512_loadu_ps(xptr.add(ib32 * 32 + 16));

                let slot = ib32 & 3;
                match slot {
                    0 => {
                        acc0 = _mm512_fmadd_ps(term_lo, x_v_lo, acc0);
                        acc1 = _mm512_fmadd_ps(term_hi, x_v_hi, acc1);
                    }
                    1 => {
                        acc2 = _mm512_fmadd_ps(term_lo, x_v_lo, acc2);
                        acc3 = _mm512_fmadd_ps(term_hi, x_v_hi, acc3);
                    }
                    2 => {
                        acc0 = _mm512_fmadd_ps(term_lo, x_v_lo, acc0);
                        acc1 = _mm512_fmadd_ps(term_hi, x_v_hi, acc1);
                    }
                    _ => {
                        acc2 = _mm512_fmadd_ps(term_lo, x_v_lo, acc2);
                        acc3 = _mm512_fmadd_ps(term_hi, x_v_hi, acc3);
                    }
                }
            }
        }
        let total = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
        out[i] = _mm512_reduce_add_ps(total);
    }
}

/// IQ2_XS matvec — AVX2 path. Same shape as IQ2_S AVX2 (gather 4
/// u64 entries, split into 4 × 8-weight chunks) but sign bytes come
/// from the [`KSIGNS_IQ2XS`] table indexed by the high 7 bits of each
/// `qs` u16.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn matvec_iq2_xs_w_f32_a_avx2(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use rustllama_gguf::dequant::{IQ2XS_GRID, KSIGNS_IQ2XS};
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 74;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;
    let grid_i64_ptr = IQ2XS_GRID.as_ptr() as *const i64;
    let bit_mask = _mm256_set_epi32(128, 64, 32, 16, 8, 4, 2, 1);
    let zero_i = _mm256_setzero_si256();

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = [_mm256_setzero_ps(); 4];

        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d = f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32();
            let qs = &w_bytes[off + 2..off + 2 + 64];
            let scales = &w_bytes[off + 2 + 64..off + 2 + 64 + 8];
            let xptr = x.as_ptr().add(b * QK_K);

            for ib32 in 0..8 {
                let scale_byte = scales[ib32];
                let db_lo = d * (0.5 + (scale_byte & 0x0F) as f32) * 0.25;
                let db_hi = d * (0.5 + (scale_byte >> 4) as f32) * 0.25;
                let base = 8 * ib32;
                let q0 = u16::from_le_bytes([qs[base], qs[base + 1]]);
                let q1 = u16::from_le_bytes([qs[base + 2], qs[base + 3]]);
                let q2 = u16::from_le_bytes([qs[base + 4], qs[base + 5]]);
                let q3 = u16::from_le_bytes([qs[base + 6], qs[base + 7]]);
                let idx_v = _mm_set_epi32(
                    (q3 & 511) as i32,
                    (q2 & 511) as i32,
                    (q1 & 511) as i32,
                    (q0 & 511) as i32,
                );
                let sb = [
                    KSIGNS_IQ2XS[(q0 >> 9) as usize] as i32,
                    KSIGNS_IQ2XS[(q1 >> 9) as usize] as i32,
                    KSIGNS_IQ2XS[(q2 >> 9) as usize] as i32,
                    KSIGNS_IQ2XS[(q3 >> 9) as usize] as i32,
                ];
                let packed = _mm256_i32gather_epi64::<8>(grid_i64_ptr, idx_v);
                let lo16 = _mm256_extracti128_si256::<0>(packed);
                let hi16 = _mm256_extracti128_si256::<1>(packed);
                let chunks_bytes: [__m128i; 4] = [
                    lo16,
                    _mm_srli_si128::<8>(lo16),
                    hi16,
                    _mm_srli_si128::<8>(hi16),
                ];
                let dbs = [db_lo, db_lo, db_hi, db_hi];
                for c in 0..4 {
                    let grid_v = _mm256_cvtepu8_epi32(chunks_bytes[c]);
                    let sb_v = _mm256_set1_epi32(sb[c]);
                    let masked = _mm256_and_si256(sb_v, bit_mask);
                    let is_set = _mm256_cmpgt_epi32(masked, zero_i);
                    let negated = _mm256_sub_epi32(zero_i, grid_v);
                    let grid_signed = _mm256_blendv_epi8(grid_v, negated, is_set);
                    let grid_f = _mm256_cvtepi32_ps(grid_signed);
                    let term = _mm256_mul_ps(grid_f, _mm256_set1_ps(dbs[c]));
                    let x_v = _mm256_loadu_ps(xptr.add(ib32 * 32 + c * 8));
                    acc[c] = _mm256_fmadd_ps(term, x_v, acc[c]);
                }
            }
        }
        let s01 = _mm256_add_ps(acc[0], acc[1]);
        let s23 = _mm256_add_ps(acc[2], acc[3]);
        let total = _mm256_add_ps(s01, s23);
        let lo = _mm256_castps256_ps128(total);
        let hi = _mm256_extractf128_ps(total, 1);
        let sum128 = _mm_add_ps(lo, hi);
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0b01));
        out[i] = _mm_cvtss_f32(sum32);
    }
}

/// IQ2_S weight matvec with on-the-fly dequant. Dispatches AVX-512 →
/// scalar.
///
/// Block layout (82 bytes per 256 weights):
///   { d: f16, qs: [u8; 64], qh: [u8; 8], scales: [u8; 8] }
///
/// Sub-block decode is described in [`rustllama_gguf::dequant::dequant_iq2_s`].
///
/// Requires `k % 256 == 0`.
/// IQ1_S weight matvec with on-the-fly dequant. Dispatches to AVX-512
/// → scalar by runtime feature detection.
///
/// Layout (50 bytes per 256 weights):
///   { d: f16, qs: [u8; 32], qh: [u16; 8] }
///
/// See [`rustllama_gguf::dequant::dequant_iq1_s`] for the full
/// per-sub-block decode formula. The AVX-512 path does the index
/// computation in scalar (4 indices per ib32 sub-block, each just a
/// shift + OR + byte read), then gathers four u64 grid entries
/// into a single __m256i, widens 32 i8 → 32 f32 across two ZMM
/// lanes, folds the per-sub-block `dl*grid + dl*delta` term, and
/// FMAs against the matching 32 elements of x.
///
/// Requires `k % 256 == 0`.
pub fn matvec_iq1_s_w_f32_a(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    const BLOCK_BYTES: usize = 50;
    const QK_K: usize = 256;
    debug_assert_eq!(k % QK_K, 0, "IQ1_S matvec requires k % 256 == 0");
    let blocks_per_row = k / QK_K;
    debug_assert_eq!(w_bytes.len(), m * blocks_per_row * BLOCK_BYTES);
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(out.len(), m);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: runtime feature detection above.
            unsafe { matvec_iq1_s_w_f32_a_avx512(w_bytes, x, out, m, k) };
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime feature detection above. AVX2 fallback
            // for hosts without AVX-512 (Skylake-era Xeons, all current
            // Intel client SKUs from Tiger Lake onwards lost AVX-512).
            unsafe { matvec_iq1_s_w_f32_a_avx2(w_bytes, x, out, m, k) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64.
        unsafe { matvec_iq1_s_w_f32_a_neon(w_bytes, x, out, m, k) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    matvec_iq1_s_w_f32_a_scalar(w_bytes, x, out, m, k);
}

fn matvec_iq1_s_w_f32_a_scalar(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use rustllama_gguf::dequant::IQ1S_DELTA;
    use rustllama_gguf::iq1_grid::IQ1S_GRID;
    const BLOCK_BYTES: usize = 50;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d = f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32();
            let qs = &w_bytes[off + 2..off + 2 + 32];
            let qh_bytes = &w_bytes[off + 34..off + 34 + 16];
            let x_block = &x[b * QK_K..(b + 1) * QK_K];
            let mut x_off = 0usize;
            for ib32 in 0..8 {
                let qh = u16::from_le_bytes([qh_bytes[ib32 * 2], qh_bytes[ib32 * 2 + 1]]);
                let dl = d * (2.0 * ((qh >> 12) & 7) as f32 + 1.0);
                let delta = if qh & 0x8000 != 0 {
                    -1.0 - IQ1S_DELTA
                } else {
                    -1.0 + IQ1S_DELTA
                };
                for l in 0..4 {
                    let idx = qs[4 * ib32 + l] as usize
                        | ((((qh >> (3 * l)) & 7) as usize) << 8);
                    let grid = IQ1S_GRID[idx].to_le_bytes();
                    for j in 0..8 {
                        let g = grid[j] as i8 as f32;
                        acc += dl * (g + delta) * x_block[x_off + 8 * l + j];
                    }
                }
                x_off += 32;
            }
        }
        out[i] = acc;
    }
}

/// IQ1_S matvec — AVX-512 path. Per ib32 sub-block: 4 scalar reads
/// of `IQ1S_GRID[idx]` packed into a 256-bit lane, widened to 32 f32
/// via two `cvtepi8_epi32` + `cvtepi32_ps` chains, scaled with
/// `fmadd(dl, grid, dl*delta)` to fold the per-sub-block delta into
/// a single instruction, then FMA'd against the matching 32 elements
/// of x. Two ZMM accumulators per row keep the FP pipelines busy
/// across the 8 ib32 sub-blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f")]
unsafe fn matvec_iq1_s_w_f32_a_avx512(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use rustllama_gguf::dequant::IQ1S_DELTA;
    use rustllama_gguf::iq1_grid::IQ1S_GRID;
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 50;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();

        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d = f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32();
            let qs = &w_bytes[off + 2..off + 2 + 32];
            let qh_bytes = &w_bytes[off + 34..off + 34 + 16];
            let xptr = x.as_ptr().add(b * QK_K);

            for ib32 in 0..8 {
                let qh = u16::from_le_bytes([qh_bytes[ib32 * 2], qh_bytes[ib32 * 2 + 1]]);
                let dl_scalar = d * (2.0 * ((qh >> 12) & 7) as f32 + 1.0);
                let delta_scalar = if qh & 0x8000 != 0 {
                    -1.0 - IQ1S_DELTA
                } else {
                    -1.0 + IQ1S_DELTA
                };
                // Precompute `dl * delta` so the f32 fmadd folds the
                // per-weight `dl * (grid + delta)` into one FMA.
                let dl_delta_scalar = dl_scalar * delta_scalar;

                // Scalar gather: 4 u64 grid entries, packed into one
                // __m256i (32 i8 = 32 packed grid coordinates).
                let i0 = qs[4 * ib32] as usize | ((((qh >> 0) & 7) as usize) << 8);
                let i1 = qs[4 * ib32 + 1] as usize | ((((qh >> 3) & 7) as usize) << 8);
                let i2 = qs[4 * ib32 + 2] as usize | ((((qh >> 6) & 7) as usize) << 8);
                let i3 = qs[4 * ib32 + 3] as usize | ((((qh >> 9) & 7) as usize) << 8);
                let grid_packed = _mm256_setr_epi64x(
                    IQ1S_GRID[i0] as i64,
                    IQ1S_GRID[i1] as i64,
                    IQ1S_GRID[i2] as i64,
                    IQ1S_GRID[i3] as i64,
                );
                // Split: low 16 bytes (entries 0,1), high 16 bytes
                // (entries 2,3). Each 16-byte half widens to 16 i32
                // via cvtepi8_epi32, then to 16 f32 via cvtepi32_ps.
                let grid_lo = _mm256_castsi256_si128(grid_packed);
                let grid_hi = _mm256_extracti128_si256::<1>(grid_packed);
                let grid_lo_f = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(grid_lo));
                let grid_hi_f = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(grid_hi));

                let dl_v = _mm512_set1_ps(dl_scalar);
                let dl_delta_v = _mm512_set1_ps(dl_delta_scalar);
                // value = dl * grid + (dl * delta)
                let val_lo = _mm512_fmadd_ps(dl_v, grid_lo_f, dl_delta_v);
                let val_hi = _mm512_fmadd_ps(dl_v, grid_hi_f, dl_delta_v);

                let xv0 = _mm512_loadu_ps(xptr.add(ib32 * 32));
                let xv1 = _mm512_loadu_ps(xptr.add(ib32 * 32 + 16));
                acc0 = _mm512_fmadd_ps(val_lo, xv0, acc0);
                acc1 = _mm512_fmadd_ps(val_hi, xv1, acc1);
            }
        }
        out[i] = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
    }
}

/// IQ1_S matvec — AVX2 path. Splits the same 4-grid-entry pack used
/// by the AVX-512 path into four ymm-worth-of-8-f32s, each fed
/// through one fused `fmadd(dl, grid, dl*delta)` plus an FMA against
/// the matching 8 elements of x. Two ymm accumulators per row stay
/// dense across the 8 ib32 sub-blocks. Performance trails the
/// AVX-512 variant by roughly the SIMD-width ratio (16 → 8 f32) plus
/// the per-entry widen cost, but on Tiger-Lake-and-later client SKUs
/// — which lack AVX-512 — this beats the scalar fallback by 4–6×.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn matvec_iq1_s_w_f32_a_avx2(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use rustllama_gguf::dequant::IQ1S_DELTA;
    use rustllama_gguf::iq1_grid::IQ1S_GRID;
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 50;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();

        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d = f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32();
            let qs = &w_bytes[off + 2..off + 2 + 32];
            let qh_bytes = &w_bytes[off + 34..off + 34 + 16];
            let xptr = x.as_ptr().add(b * QK_K);

            for ib32 in 0..8 {
                let qh = u16::from_le_bytes([qh_bytes[ib32 * 2], qh_bytes[ib32 * 2 + 1]]);
                let dl_scalar = d * (2.0 * ((qh >> 12) & 7) as f32 + 1.0);
                let delta_scalar = if qh & 0x8000 != 0 {
                    -1.0 - IQ1S_DELTA
                } else {
                    -1.0 + IQ1S_DELTA
                };
                let dl_delta_scalar = dl_scalar * delta_scalar;
                let dl_v = _mm256_set1_ps(dl_scalar);
                let dl_delta_v = _mm256_set1_ps(dl_delta_scalar);

                // 4 grid entries → 4 vectors of 8 f32. Each entry is 8
                // packed signed bytes; widen one at a time via the
                // 64-bit-load → cvtepi8_epi32 → cvtepi32_ps chain.
                let i0 = qs[4 * ib32] as usize | ((((qh >> 0) & 7) as usize) << 8);
                let i1 = qs[4 * ib32 + 1] as usize | ((((qh >> 3) & 7) as usize) << 8);
                let i2 = qs[4 * ib32 + 2] as usize | ((((qh >> 6) & 7) as usize) << 8);
                let i3 = qs[4 * ib32 + 3] as usize | ((((qh >> 9) & 7) as usize) << 8);

                let g0 = _mm_cvtsi64_si128(IQ1S_GRID[i0] as i64);
                let g1 = _mm_cvtsi64_si128(IQ1S_GRID[i1] as i64);
                let g2 = _mm_cvtsi64_si128(IQ1S_GRID[i2] as i64);
                let g3 = _mm_cvtsi64_si128(IQ1S_GRID[i3] as i64);
                let g0f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g0));
                let g1f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g1));
                let g2f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g2));
                let g3f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g3));

                // value = dl * grid + (dl * delta), all lanes share
                // (dl, delta) because both come from this ib32's qh.
                let v0 = _mm256_fmadd_ps(dl_v, g0f, dl_delta_v);
                let v1 = _mm256_fmadd_ps(dl_v, g1f, dl_delta_v);
                let v2 = _mm256_fmadd_ps(dl_v, g2f, dl_delta_v);
                let v3 = _mm256_fmadd_ps(dl_v, g3f, dl_delta_v);

                let xv0 = _mm256_loadu_ps(xptr.add(ib32 * 32));
                let xv1 = _mm256_loadu_ps(xptr.add(ib32 * 32 + 8));
                let xv2 = _mm256_loadu_ps(xptr.add(ib32 * 32 + 16));
                let xv3 = _mm256_loadu_ps(xptr.add(ib32 * 32 + 24));
                acc0 = _mm256_fmadd_ps(v0, xv0, acc0);
                acc1 = _mm256_fmadd_ps(v1, xv1, acc1);
                acc0 = _mm256_fmadd_ps(v2, xv2, acc0);
                acc1 = _mm256_fmadd_ps(v3, xv3, acc1);
            }
        }
        // Horizontal reduce acc0+acc1.
        let sum = _mm256_add_ps(acc0, acc1);
        let hi = _mm256_extractf128_ps::<1>(sum);
        let lo = _mm256_castps256_ps128(sum);
        let q = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(q);
        let sums = _mm_add_ps(q, shuf);
        let shuf = _mm_movehl_ps(shuf, sums);
        let sums = _mm_add_ss(sums, shuf);
        out[i] = _mm_cvtss_f32(sums);
    }
}

/// IQ1_M weight matvec with on-the-fly dequant. Dispatches to AVX-512
/// → scalar by runtime feature detection. Block layout 56 bytes
/// per 256 weights; see [`rustllama_gguf::dequant::dequant_iq1_m`]
/// for the full per-sub-block decode formula (no standalone `d` —
/// it's reassembled from the top nibbles of the 4 packed scale words).
///
/// Requires `k % 256 == 0`.
pub fn matvec_iq1_m_w_f32_a(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    const BLOCK_BYTES: usize = 56;
    const QK_K: usize = 256;
    debug_assert_eq!(k % QK_K, 0, "IQ1_M matvec requires k % 256 == 0");
    let blocks_per_row = k / QK_K;
    debug_assert_eq!(w_bytes.len(), m * blocks_per_row * BLOCK_BYTES);
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(out.len(), m);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: runtime feature detection above.
            unsafe { matvec_iq1_m_w_f32_a_avx512(w_bytes, x, out, m, k) };
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime feature detection above.
            unsafe { matvec_iq1_m_w_f32_a_avx2(w_bytes, x, out, m, k) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64.
        unsafe { matvec_iq1_m_w_f32_a_neon(w_bytes, x, out, m, k) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    matvec_iq1_m_w_f32_a_scalar(w_bytes, x, out, m, k);
}

fn matvec_iq1_m_w_f32_a_scalar(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use rustllama_gguf::dequant::IQ1S_DELTA;
    use rustllama_gguf::iq1_grid::IQ1S_GRID;
    const BLOCK_BYTES: usize = 56;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let qs = &w_bytes[off..off + 32];
            let qh = &w_bytes[off + 32..off + 32 + 16];
            let scales_bytes = &w_bytes[off + 48..off + 48 + 8];
            let mut sc = [0u16; 4];
            for ii in 0..4 {
                sc[ii] = u16::from_le_bytes([scales_bytes[ii * 2], scales_bytes[ii * 2 + 1]]);
            }
            let d_bits: u16 = (sc[0] >> 12)
                | ((sc[1] >> 8) & 0x00F0)
                | ((sc[2] >> 4) & 0x0F00)
                | (sc[3] & 0xF000);
            let d = f16::from_bits(d_bits).to_f32();
            let x_block = &x[b * QK_K..(b + 1) * QK_K];
            let mut x_off = 0usize;
            for ib in 0..8 {
                let s_word = sc[ib / 2];
                let shift0 = 6 * (ib % 2);
                let shift1 = 6 * (ib % 2) + 3;
                let dl1 = d * (2.0 * ((s_word >> shift0) & 0x7) as f32 + 1.0);
                let dl2 = d * (2.0 * ((s_word >> shift1) & 0x7) as f32 + 1.0);
                let qh0 = qh[ib * 2];
                let qh1 = qh[ib * 2 + 1];
                let delta = |bit_set: bool| {
                    if bit_set {
                        -1.0 - IQ1S_DELTA
                    } else {
                        -1.0 + IQ1S_DELTA
                    }
                };
                let qs_chunk = &qs[ib * 4..ib * 4 + 4];
                let idx_l = [
                    qs_chunk[0] as usize | (((qh0 & 0x07) as usize) << 8),
                    qs_chunk[1] as usize | ((((qh0 >> 4) & 0x07) as usize) << 8),
                    qs_chunk[2] as usize | (((qh1 & 0x07) as usize) << 8),
                    qs_chunk[3] as usize | ((((qh1 >> 4) & 0x07) as usize) << 8),
                ];
                let lanes = [
                    (dl1, delta(qh0 & 0x08 != 0), idx_l[0]),
                    (dl1, delta(qh0 & 0x80 != 0), idx_l[1]),
                    (dl2, delta(qh1 & 0x08 != 0), idx_l[2]),
                    (dl2, delta(qh1 & 0x80 != 0), idx_l[3]),
                ];
                for (l, (dl, delta_val, idx)) in lanes.iter().copied().enumerate() {
                    let grid = IQ1S_GRID[idx].to_le_bytes();
                    for j in 0..8 {
                        let g = grid[j] as i8 as f32;
                        acc += dl * (g + delta_val) * x_block[x_off + 8 * l + j];
                    }
                }
                x_off += 32;
            }
        }
        out[i] = acc;
    }
}

/// IQ1_M matvec — AVX-512 path. Same pack-4-grid-entries-into-an-i256
/// recipe as the IQ1_S AVX-512 path, with extra scalar prep to
/// reassemble the per-16-weight `(dl, delta)` pairs from the packed
/// scale words. Each 16-weight half-of-sub-block uses its own
/// `dl_delta_v` broadcast so the `fmadd(dl, grid, dl*delta)` fold
/// still holds.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f")]
unsafe fn matvec_iq1_m_w_f32_a_avx512(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use rustllama_gguf::dequant::IQ1S_DELTA;
    use rustllama_gguf::iq1_grid::IQ1S_GRID;
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 56;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();

        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let qs = &w_bytes[off..off + 32];
            let qh = &w_bytes[off + 32..off + 32 + 16];
            let scales_bytes = &w_bytes[off + 48..off + 48 + 8];
            let mut sc = [0u16; 4];
            for ii in 0..4 {
                sc[ii] = u16::from_le_bytes([scales_bytes[ii * 2], scales_bytes[ii * 2 + 1]]);
            }
            let d_bits: u16 = (sc[0] >> 12)
                | ((sc[1] >> 8) & 0x00F0)
                | ((sc[2] >> 4) & 0x0F00)
                | (sc[3] & 0xF000);
            let d = f16::from_bits(d_bits).to_f32();
            let xptr = x.as_ptr().add(b * QK_K);

            for ib in 0..8 {
                let s_word = sc[ib / 2];
                let shift0 = 6 * (ib % 2);
                let shift1 = 6 * (ib % 2) + 3;
                let dl1 = d * (2.0 * ((s_word >> shift0) & 0x7) as f32 + 1.0);
                let dl2 = d * (2.0 * ((s_word >> shift1) & 0x7) as f32 + 1.0);
                let qh0 = qh[ib * 2];
                let qh1 = qh[ib * 2 + 1];
                let delta = |bit_set: bool| {
                    if bit_set {
                        -1.0 - IQ1S_DELTA
                    } else {
                        -1.0 + IQ1S_DELTA
                    }
                };
                let qs_chunk = &qs[ib * 4..ib * 4 + 4];
                let i0 = qs_chunk[0] as usize | (((qh0 & 0x07) as usize) << 8);
                let i1 = qs_chunk[1] as usize | ((((qh0 >> 4) & 0x07) as usize) << 8);
                let i2 = qs_chunk[2] as usize | (((qh1 & 0x07) as usize) << 8);
                let i3 = qs_chunk[3] as usize | ((((qh1 >> 4) & 0x07) as usize) << 8);

                // Lane l=0 (weights 0..7):   dl1, delta(qh0 & 0x08)
                // Lane l=1 (weights 8..15):  dl1, delta(qh0 & 0x80)
                // Lane l=2 (weights 16..23): dl2, delta(qh1 & 0x08)
                // Lane l=3 (weights 24..31): dl2, delta(qh1 & 0x80)
                let d0 = delta(qh0 & 0x08 != 0);
                let d1 = delta(qh0 & 0x80 != 0);
                let d2 = delta(qh1 & 0x08 != 0);
                let d3 = delta(qh1 & 0x80 != 0);

                let grid_packed = _mm256_setr_epi64x(
                    IQ1S_GRID[i0] as i64,
                    IQ1S_GRID[i1] as i64,
                    IQ1S_GRID[i2] as i64,
                    IQ1S_GRID[i3] as i64,
                );
                let grid_lo = _mm256_castsi256_si128(grid_packed);
                let grid_hi = _mm256_extracti128_si256::<1>(grid_packed);
                let grid_lo_f = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(grid_lo));
                let grid_hi_f = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(grid_hi));

                // Per-16-weight scale/delta broadcasts. The lo half
                // packs lanes 0+1 (dl1 with deltas d0/d1) — they
                // share dl1 but differ in delta. Build the
                // `dl*grid + dl*delta` term using two per-lane
                // delta broadcasts laid out matching the i32 lanes
                // produced by cvtepi8_epi32.
                //
                // Layout after cvtepi8_epi32 on grid_lo (16 bytes of
                // packed grid entries 0,1): lanes 0..7 = entry-0
                // bytes 0..7 (weights 0..7), lanes 8..15 = entry-1
                // bytes 0..7 (weights 8..15). So we need a mask-style
                // delta vector with d0 in lanes 0..7 and d1 in 8..15.
                let dl_lo = _mm512_set1_ps(dl1);
                let dl_hi = _mm512_set1_ps(dl2);
                let dl_delta_lo = _mm512_setr_ps(
                    dl1 * d0, dl1 * d0, dl1 * d0, dl1 * d0,
                    dl1 * d0, dl1 * d0, dl1 * d0, dl1 * d0,
                    dl1 * d1, dl1 * d1, dl1 * d1, dl1 * d1,
                    dl1 * d1, dl1 * d1, dl1 * d1, dl1 * d1,
                );
                let dl_delta_hi = _mm512_setr_ps(
                    dl2 * d2, dl2 * d2, dl2 * d2, dl2 * d2,
                    dl2 * d2, dl2 * d2, dl2 * d2, dl2 * d2,
                    dl2 * d3, dl2 * d3, dl2 * d3, dl2 * d3,
                    dl2 * d3, dl2 * d3, dl2 * d3, dl2 * d3,
                );
                let val_lo = _mm512_fmadd_ps(dl_lo, grid_lo_f, dl_delta_lo);
                let val_hi = _mm512_fmadd_ps(dl_hi, grid_hi_f, dl_delta_hi);

                let xv0 = _mm512_loadu_ps(xptr.add(ib * 32));
                let xv1 = _mm512_loadu_ps(xptr.add(ib * 32 + 16));
                acc0 = _mm512_fmadd_ps(val_lo, xv0, acc0);
                acc1 = _mm512_fmadd_ps(val_hi, xv1, acc1);
            }
        }
        out[i] = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
    }
}

/// IQ1_M matvec — AVX2 path. Same recipe as the IQ1_S AVX2 variant
/// but with per-lane `(dl, delta)` reassembly from the packed scale
/// words. Entry l=0/1 share `dl1` with their own delta picks; entry
/// l=2/3 share `dl2`. Four ymm vectors per ib32 sub-block; two ymm
/// accumulators per row.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn matvec_iq1_m_w_f32_a_avx2(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use rustllama_gguf::dequant::IQ1S_DELTA;
    use rustllama_gguf::iq1_grid::IQ1S_GRID;
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 56;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();

        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let qs = &w_bytes[off..off + 32];
            let qh = &w_bytes[off + 32..off + 32 + 16];
            let scales_bytes = &w_bytes[off + 48..off + 48 + 8];
            let mut sc = [0u16; 4];
            for ii in 0..4 {
                sc[ii] = u16::from_le_bytes([scales_bytes[ii * 2], scales_bytes[ii * 2 + 1]]);
            }
            let d_bits: u16 = (sc[0] >> 12)
                | ((sc[1] >> 8) & 0x00F0)
                | ((sc[2] >> 4) & 0x0F00)
                | (sc[3] & 0xF000);
            let d = f16::from_bits(d_bits).to_f32();
            let xptr = x.as_ptr().add(b * QK_K);

            for ib in 0..8 {
                let s_word = sc[ib / 2];
                let shift0 = 6 * (ib % 2);
                let shift1 = 6 * (ib % 2) + 3;
                let dl1 = d * (2.0 * ((s_word >> shift0) & 0x7) as f32 + 1.0);
                let dl2 = d * (2.0 * ((s_word >> shift1) & 0x7) as f32 + 1.0);
                let qh0 = qh[ib * 2];
                let qh1 = qh[ib * 2 + 1];
                let delta = |bit_set: bool| {
                    if bit_set {
                        -1.0 - IQ1S_DELTA
                    } else {
                        -1.0 + IQ1S_DELTA
                    }
                };
                let d0 = delta(qh0 & 0x08 != 0);
                let d1 = delta(qh0 & 0x80 != 0);
                let d2 = delta(qh1 & 0x08 != 0);
                let d3 = delta(qh1 & 0x80 != 0);

                let qs_chunk = &qs[ib * 4..ib * 4 + 4];
                let i0 = qs_chunk[0] as usize | (((qh0 & 0x07) as usize) << 8);
                let i1 = qs_chunk[1] as usize | ((((qh0 >> 4) & 0x07) as usize) << 8);
                let i2 = qs_chunk[2] as usize | (((qh1 & 0x07) as usize) << 8);
                let i3 = qs_chunk[3] as usize | ((((qh1 >> 4) & 0x07) as usize) << 8);

                let g0 = _mm_cvtsi64_si128(IQ1S_GRID[i0] as i64);
                let g1 = _mm_cvtsi64_si128(IQ1S_GRID[i1] as i64);
                let g2 = _mm_cvtsi64_si128(IQ1S_GRID[i2] as i64);
                let g3 = _mm_cvtsi64_si128(IQ1S_GRID[i3] as i64);
                let g0f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g0));
                let g1f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g1));
                let g2f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g2));
                let g3f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g3));

                let dl1_v = _mm256_set1_ps(dl1);
                let dl2_v = _mm256_set1_ps(dl2);
                let dl1d0 = _mm256_set1_ps(dl1 * d0);
                let dl1d1 = _mm256_set1_ps(dl1 * d1);
                let dl2d2 = _mm256_set1_ps(dl2 * d2);
                let dl2d3 = _mm256_set1_ps(dl2 * d3);

                let v0 = _mm256_fmadd_ps(dl1_v, g0f, dl1d0);
                let v1 = _mm256_fmadd_ps(dl1_v, g1f, dl1d1);
                let v2 = _mm256_fmadd_ps(dl2_v, g2f, dl2d2);
                let v3 = _mm256_fmadd_ps(dl2_v, g3f, dl2d3);

                let xv0 = _mm256_loadu_ps(xptr.add(ib * 32));
                let xv1 = _mm256_loadu_ps(xptr.add(ib * 32 + 8));
                let xv2 = _mm256_loadu_ps(xptr.add(ib * 32 + 16));
                let xv3 = _mm256_loadu_ps(xptr.add(ib * 32 + 24));
                acc0 = _mm256_fmadd_ps(v0, xv0, acc0);
                acc1 = _mm256_fmadd_ps(v1, xv1, acc1);
                acc0 = _mm256_fmadd_ps(v2, xv2, acc0);
                acc1 = _mm256_fmadd_ps(v3, xv3, acc1);
            }
        }
        let sum = _mm256_add_ps(acc0, acc1);
        let hi = _mm256_extractf128_ps::<1>(sum);
        let lo = _mm256_castps256_ps128(sum);
        let q = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(q);
        let sums = _mm_add_ps(q, shuf);
        let shuf = _mm_movehl_ps(shuf, sums);
        let sums = _mm_add_ss(sums, shuf);
        out[i] = _mm_cvtss_f32(sums);
    }
}

/// Decode IQ1_S rows by id and write to `out` as F32. Thin wrapper
/// over [`rustllama_gguf::dequant::dequant_iq1_s`].
pub fn embed_lookup_iq1_s(table_bytes: &[u8], ids: &[i32], out: &mut [f32], d: usize) {
    const BLOCK_BYTES: usize = 50;
    const QK_K: usize = 256;
    assert_eq!(d % QK_K, 0, "IQ1_S embed lookup requires d % 256 == 0");
    let blocks_per_row = d / QK_K;
    let row_bytes = blocks_per_row * BLOCK_BYTES;
    for (i, &id) in ids.iter().enumerate() {
        let row_start = (id as usize) * row_bytes;
        let dst = &mut out[i * d..(i + 1) * d];
        let row = &table_bytes[row_start..row_start + row_bytes];
        rustllama_gguf::dequant::dequant_iq1_s(row, dst);
    }
}

/// Decode IQ1_M rows by id and write to `out` as F32. Thin wrapper
/// over [`rustllama_gguf::dequant::dequant_iq1_m`].
pub fn embed_lookup_iq1_m(table_bytes: &[u8], ids: &[i32], out: &mut [f32], d: usize) {
    const BLOCK_BYTES: usize = 56;
    const QK_K: usize = 256;
    assert_eq!(d % QK_K, 0, "IQ1_M embed lookup requires d % 256 == 0");
    let blocks_per_row = d / QK_K;
    let row_bytes = blocks_per_row * BLOCK_BYTES;
    for (i, &id) in ids.iter().enumerate() {
        let row_start = (id as usize) * row_bytes;
        let dst = &mut out[i * d..(i + 1) * d];
        let row = &table_bytes[row_start..row_start + row_bytes];
        rustllama_gguf::dequant::dequant_iq1_m(row, dst);
    }
}

pub fn matvec_iq2_s_w_f32_a(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    const BLOCK_BYTES: usize = 82;
    const QK_K: usize = 256;
    debug_assert_eq!(k % QK_K, 0, "IQ2_S matvec requires k % 256 == 0");
    let blocks_per_row = k / QK_K;
    debug_assert_eq!(w_bytes.len(), m * blocks_per_row * BLOCK_BYTES);
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(out.len(), m);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx2") {
            // SAFETY: runtime feature detection above.
            unsafe { matvec_iq2_s_w_f32_a_avx512(w_bytes, x, out, m, k) };
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime feature detection above.
            unsafe { matvec_iq2_s_w_f32_a_avx2(w_bytes, x, out, m, k) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64.
        unsafe { matvec_iq2_s_w_f32_a_neon(w_bytes, x, out, m, k) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    matvec_iq2_s_w_f32_a_scalar(w_bytes, x, out, m, k);
}

fn matvec_iq2_s_w_f32_a_scalar(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use rustllama_gguf::dequant::{IQ2S_GRID, KMASK_IQ2XS};
    const BLOCK_BYTES: usize = 82;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;

    let grid_bytes: &[u8] = bytemuck::cast_slice(&IQ2S_GRID);

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d = f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32();
            // qs is 64 bytes split as qs_lo[0..32] (indices) + signs[32..64].
            let qs_lo = &w_bytes[off + 2..off + 2 + 32];
            let signs = &w_bytes[off + 2 + 32..off + 2 + 64];
            let qh = &w_bytes[off + 2 + 64..off + 2 + 64 + 8];
            let scales = &w_bytes[off + 2 + 64 + 8..off + 2 + 64 + 8 + 8];
            let x_block = &x[b * QK_K..(b + 1) * QK_K];

            for ib32 in 0..8 {
                let scale_byte = scales[ib32];
                let db_lo = d * (0.5 + (scale_byte & 0x0F) as f32) * 0.25;
                let db_hi = d * (0.5 + (scale_byte >> 4) as f32) * 0.25;
                let qs_off = ib32 * 4;
                let qh_byte = qh[ib32];
                for l in 0..4 {
                    let high_bits = ((qh_byte as usize) << (8 - 2 * l)) & 0x300;
                    let grid_idx = (qs_lo[qs_off + l] as usize) | high_bits;
                    let sign_byte = signs[qs_off + l];
                    let grid = &grid_bytes[grid_idx * 8..grid_idx * 8 + 8];
                    let db = if l < 2 { db_lo } else { db_hi };
                    let x_off = ib32 * 32 + l * 8;
                    for j in 0..8 {
                        let s = if sign_byte & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                        acc += db * (grid[j] as f32) * s * x_block[x_off + j];
                    }
                }
            }
        }
        out[i] = acc;
    }
}

/// IQ2_S matvec — AVX-512 path. Processes one ib32 sub-block (32
/// weights) per outer iteration via two 16-wide ZMM operations:
///
///   - 4 10-bit grid indices (low 8 bits from `qs[ib32*4..ib32*4+4]`,
///     high 2 bits from successive bit-pairs of `qh[ib32]`) packed
///     into `__m128i`.
///   - `_mm256_i32gather_epi64::<8>` gathers 4 codebook entries (each
///     is a u64 = 8 packed-i8 grid points = 8 weights).
///   - Split the 32-byte result into two 16-byte halves; widen each
///     via `_mm512_cvtepu8_epi32` → 2 ZMMs of 16 i32 weights.
///   - Two sign bytes per ZMM build a 16-bit k-mask; mask-sub-negate
///     applies the sign in i32 space.
///   - Convert to f32, multiply by per-sub-block `db` broadcast
///     (low nibble for first ZMM, high for second), FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f")]
unsafe fn matvec_iq2_s_w_f32_a_avx512(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use rustllama_gguf::dequant::IQ2S_GRID;
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 82;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;
    let grid_i64_ptr = IQ2S_GRID.as_ptr() as *const i64;

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();
        let mut acc2 = _mm512_setzero_ps();
        let mut acc3 = _mm512_setzero_ps();

        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d = f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32();
            let qs_lo = &w_bytes[off + 2..off + 2 + 32];
            let signs = &w_bytes[off + 2 + 32..off + 2 + 64];
            let qh = &w_bytes[off + 2 + 64..off + 2 + 64 + 8];
            let scales = &w_bytes[off + 2 + 64 + 8..off + 2 + 64 + 8 + 8];
            let xptr = x.as_ptr().add(b * QK_K);

            for ib32 in 0..8 {
                let scale_byte = scales[ib32];
                let db_lo = d * (0.5 + (scale_byte & 0x0F) as f32) * 0.25;
                let db_hi = d * (0.5 + (scale_byte >> 4) as f32) * 0.25;
                let qs_off = ib32 * 4;
                let qh_byte = qh[ib32];

                // 4 grid indices, with bits 8-9 supplied by qh's bit-pairs.
                let highs: [i32; 4] = [
                    (((qh_byte >> 0) & 3) as i32) << 8,
                    (((qh_byte >> 2) & 3) as i32) << 8,
                    (((qh_byte >> 4) & 3) as i32) << 8,
                    (((qh_byte >> 6) & 3) as i32) << 8,
                ];
                let idx_v = _mm_set_epi32(
                    (qs_lo[qs_off + 3] as i32) | highs[3],
                    (qs_lo[qs_off + 2] as i32) | highs[2],
                    (qs_lo[qs_off + 1] as i32) | highs[1],
                    (qs_lo[qs_off] as i32) | highs[0],
                );

                let packed = _mm256_i32gather_epi64::<8>(grid_i64_ptr, idx_v);
                let lo_half = _mm256_extracti128_si256::<0>(packed);
                let hi_half = _mm256_extracti128_si256::<1>(packed);
                let grid_lo = _mm512_cvtepu8_epi32(lo_half);
                let grid_hi = _mm512_cvtepu8_epi32(hi_half);

                let s0 = signs[qs_off] as u16;
                let s1 = signs[qs_off + 1] as u16;
                let s2 = signs[qs_off + 2] as u16;
                let s3 = signs[qs_off + 3] as u16;
                let mask_lo: __mmask16 = s0 | (s1 << 8);
                let mask_hi: __mmask16 = s2 | (s3 << 8);

                let zero = _mm512_setzero_si512();
                let grid_lo_s = _mm512_mask_sub_epi32(grid_lo, mask_lo, zero, grid_lo);
                let grid_hi_s = _mm512_mask_sub_epi32(grid_hi, mask_hi, zero, grid_hi);

                let term_lo =
                    _mm512_mul_ps(_mm512_cvtepi32_ps(grid_lo_s), _mm512_set1_ps(db_lo));
                let term_hi =
                    _mm512_mul_ps(_mm512_cvtepi32_ps(grid_hi_s), _mm512_set1_ps(db_hi));

                let x_v_lo = _mm512_loadu_ps(xptr.add(ib32 * 32));
                let x_v_hi = _mm512_loadu_ps(xptr.add(ib32 * 32 + 16));

                // Round-robin across 4 accumulators to keep FMA pipelined.
                let slot = ib32 & 3;
                match slot {
                    0 => {
                        acc0 = _mm512_fmadd_ps(term_lo, x_v_lo, acc0);
                        acc1 = _mm512_fmadd_ps(term_hi, x_v_hi, acc1);
                    }
                    1 => {
                        acc2 = _mm512_fmadd_ps(term_lo, x_v_lo, acc2);
                        acc3 = _mm512_fmadd_ps(term_hi, x_v_hi, acc3);
                    }
                    2 => {
                        acc0 = _mm512_fmadd_ps(term_lo, x_v_lo, acc0);
                        acc1 = _mm512_fmadd_ps(term_hi, x_v_hi, acc1);
                    }
                    _ => {
                        acc2 = _mm512_fmadd_ps(term_lo, x_v_lo, acc2);
                        acc3 = _mm512_fmadd_ps(term_hi, x_v_hi, acc3);
                    }
                }
            }
        }
        let total = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
        out[i] = _mm512_reduce_add_ps(total);
    }
}

/// IQ2_S matvec — AVX2 path. 8 weights per YMM iteration (one
/// codebook entry per chunk vs the AVX-512 path's two entries
/// packed in one ZMM). Per ib32 sub-block: 4 codebook entries
/// gathered as one `_mm256_i32gather_epi64`, then split into 4 ×
/// 8-byte chunks and widened individually.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn matvec_iq2_s_w_f32_a_avx2(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use rustllama_gguf::dequant::IQ2S_GRID;
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 82;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;
    let grid_i64_ptr = IQ2S_GRID.as_ptr() as *const i64;
    let bit_mask = _mm256_set_epi32(128, 64, 32, 16, 8, 4, 2, 1);
    let zero_i = _mm256_setzero_si256();

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = [_mm256_setzero_ps(); 4];

        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d = f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32();
            let qs_lo = &w_bytes[off + 2..off + 2 + 32];
            let signs = &w_bytes[off + 2 + 32..off + 2 + 64];
            let qh = &w_bytes[off + 2 + 64..off + 2 + 64 + 8];
            let scales = &w_bytes[off + 2 + 64 + 8..off + 2 + 64 + 8 + 8];
            let xptr = x.as_ptr().add(b * QK_K);

            for ib32 in 0..8 {
                let scale_byte = scales[ib32];
                let db_lo = d * (0.5 + (scale_byte & 0x0F) as f32) * 0.25;
                let db_hi = d * (0.5 + (scale_byte >> 4) as f32) * 0.25;
                let qs_off = ib32 * 4;
                let qh_byte = qh[ib32];

                let highs: [i32; 4] = [
                    (((qh_byte >> 0) & 3) as i32) << 8,
                    (((qh_byte >> 2) & 3) as i32) << 8,
                    (((qh_byte >> 4) & 3) as i32) << 8,
                    (((qh_byte >> 6) & 3) as i32) << 8,
                ];
                let idx_v = _mm_set_epi32(
                    (qs_lo[qs_off + 3] as i32) | highs[3],
                    (qs_lo[qs_off + 2] as i32) | highs[2],
                    (qs_lo[qs_off + 1] as i32) | highs[1],
                    (qs_lo[qs_off] as i32) | highs[0],
                );
                let packed = _mm256_i32gather_epi64::<8>(grid_i64_ptr, idx_v);
                let lo16 = _mm256_extracti128_si256::<0>(packed);
                let hi16 = _mm256_extracti128_si256::<1>(packed);

                // 4 chunks of 8 weights each. Chunks 0-1 use db_lo
                // (l=0,1), chunks 2-3 use db_hi (l=2,3).
                let chunks_bytes: [__m128i; 4] = [
                    lo16,
                    _mm_srli_si128::<8>(lo16),
                    hi16,
                    _mm_srli_si128::<8>(hi16),
                ];
                let dbs = [db_lo, db_lo, db_hi, db_hi];
                for c in 0..4 {
                    let grid_v = _mm256_cvtepu8_epi32(chunks_bytes[c]);
                    let sign_byte = signs[qs_off + c] as i32;
                    let sb_v = _mm256_set1_epi32(sign_byte);
                    let masked = _mm256_and_si256(sb_v, bit_mask);
                    let is_set = _mm256_cmpgt_epi32(masked, zero_i);
                    let negated = _mm256_sub_epi32(zero_i, grid_v);
                    let grid_signed = _mm256_blendv_epi8(grid_v, negated, is_set);
                    let grid_f = _mm256_cvtepi32_ps(grid_signed);
                    let term = _mm256_mul_ps(grid_f, _mm256_set1_ps(dbs[c]));
                    let x_v = _mm256_loadu_ps(xptr.add(ib32 * 32 + c * 8));
                    acc[c] = _mm256_fmadd_ps(term, x_v, acc[c]);
                }
            }
        }
        let s01 = _mm256_add_ps(acc[0], acc[1]);
        let s23 = _mm256_add_ps(acc[2], acc[3]);
        let total = _mm256_add_ps(s01, s23);
        let lo = _mm256_castps256_ps128(total);
        let hi = _mm256_extractf128_ps(total, 1);
        let sum128 = _mm_add_ps(lo, hi);
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0b01));
        out[i] = _mm_cvtss_f32(sum32);
    }
}

/// Q6_K weight matvec with on-the-fly dequant.
///
/// Super-block layout (210 bytes per 256 weights):
///   { ql: [u8; 128], qh: [u8; 64], scales: [i8; 16], d: f16 }
///
/// Per-weight: 4 low bits in `ql`, 2 high bits in `qh` (4 weights packed per
/// byte). Combined to a 6-bit value and recentered to signed by subtracting
/// 32. Each 16-weight sub-block has its own signed-i8 scale; the 16 scales
/// are organized so that one super-block is processed as two halves of 128
/// weights each, with sub-strands picking scales at strided positions
/// (matches ggml's `dequantize_row_q6_K` layout).
///
/// Requires `k % 256 == 0`.
pub fn matvec_q6_k_w_f32_a(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    const BLOCK_BYTES: usize = 210;
    const QK_K: usize = 256;
    debug_assert_eq!(k % QK_K, 0, "Q6_K matvec requires k % 256 == 0");
    let blocks_per_row = k / QK_K;
    debug_assert_eq!(w_bytes.len(), m * blocks_per_row * BLOCK_BYTES);
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(out.len(), m);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: runtime feature detection.
            unsafe { matvec_q6_k_w_f32_a_avx512(w_bytes, x, out, m, k) };
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime feature detection above.
            unsafe { matvec_q6_k_w_f32_a_avx2(w_bytes, x, out, m, k) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64.
        unsafe { matvec_q6_k_w_f32_a_neon(w_bytes, x, out, m, k) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    matvec_q6_k_w_f32_a_scalar(w_bytes, x, out, m, k);
}

/// Build 16 Q6_K codes (one strand, one l-chunk) as unsigned `float32x4`
/// lanes `[0..4, 4..8, 8..12, 12..16]`. Combines the 4 low bits from `ql`
/// (low or high nibble per `is_high_nibble`) with the 2 high bits pulled
/// from `qh` at a per-strand bit position (`neg_shift = -shift` fed to
/// `vshlq_u8`, since NEON's logical shift takes a signed per-lane count)
/// into the 6-bit `[0,63]` value. The caller recenters by -32. Mirrors
/// `build_q_strand` in the AVX2 path.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn build_q6k_strand_neon(
    ql_ptr: *const u8,
    qh_ptr: *const u8,
    neg_shift: std::arch::aarch64::int8x16_t,
    is_high_nibble: bool,
    mask_lo: std::arch::aarch64::uint8x16_t,
    mask_2bit: std::arch::aarch64::uint8x16_t,
) -> [std::arch::aarch64::float32x4_t; 4] {
    use std::arch::aarch64::*;
    let ql = vld1q_u8(ql_ptr);
    let ql_nibble = if is_high_nibble {
        vshrq_n_u8::<4>(ql)
    } else {
        vandq_u8(ql, mask_lo)
    };
    let qh = vld1q_u8(qh_ptr);
    // (qh >> shift) & 0x03, then promote to bits 4..5 and OR with the nibble.
    let qh_bits = vandq_u8(vshlq_u8(qh, neg_shift), mask_2bit);
    let qh_top = vshlq_n_u8::<4>(qh_bits);
    u8x16_to_f32x4x4(vorrq_u8(ql_nibble, qh_top))
}

/// AArch64 NEON Q6_K matvec. Same 4-strand × 2-half super-block layout as
/// the AVX2 path (see [`matvec_q6_k_w_f32_a_scalar`] / the AVX2 strand
/// specs): each 256-weight super-block splits into two 128-weight halves
/// (`n`), each half into 4 strands (`q1..q4`) picking the ql nibble, the
/// qh 2-bit field, the x window, and the signed-i8 sub-scale. Each strand
/// covers 32 outputs in two 16-lane chunks whose scale flips at the l=16
/// boundary (is=0 vs is=1). NEON baseline; same tolerance contract as the
/// AVX2 path.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn matvec_q6_k_w_f32_a_neon(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use std::arch::aarch64::*;
    const BLOCK_BYTES: usize = 210;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;
    let mask_lo = vdupq_n_u8(0x0F);
    let mask_2bit = vdupq_n_u8(0x03);
    let thirtytwo = vdupq_n_f32(32.0);
    // (ql_off, is_high_nibble, qh_shift, x_off, scale_offset) per strand.
    let strand_specs: [(usize, bool, i32, usize, usize); 4] = [
        (0, false, 0, 0, 0),   // q1
        (32, false, 2, 32, 1), // q2
        (0, true, 4, 64, 2),   // q3
        (32, true, 6, 96, 3),  // q4
    ];
    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = [vdupq_n_f32(0.0); 8];
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let ql_ptr = w_bytes.as_ptr().add(off);
            let qh_ptr = w_bytes.as_ptr().add(off + 128);
            let scales_ptr = w_bytes.as_ptr().add(off + 192) as *const i8;
            let d = f16::from_le_bytes([w_bytes[off + 208], w_bytes[off + 209]]).to_f32();
            let xptr = x.as_ptr().add(b * QK_K);
            let mut ai = 0usize;
            for n in 0..2 {
                let ql_base = ql_ptr.add(64 * n);
                let qh_base = qh_ptr.add(32 * n);
                let scales_base = scales_ptr.add(n * 8);
                let x_base = xptr.add(n * 128);
                for &(ql_off, hi, shift_v, x_off, scale_offset) in strand_specs.iter() {
                    let neg_shift = vdupq_n_s8(-(shift_v as i8));
                    // l<16 uses scale_a, l>=16 uses scale_b.
                    let d_sa = d * (*scales_base.add(scale_offset * 2)) as f32;
                    let d_sb = d * (*scales_base.add(scale_offset * 2 + 1)) as f32;
                    for (chunk, d_s) in [(0usize, d_sa), (16usize, d_sb)] {
                        let q = build_q6k_strand_neon(
                            ql_base.add(ql_off + chunk),
                            qh_base.add(chunk),
                            neg_shift,
                            hi,
                            mask_lo,
                            mask_2bit,
                        );
                        let dsv = vdupq_n_f32(d_s);
                        for c in 0..4 {
                            let xv = vld1q_f32(x_base.add(x_off + chunk + c * 4));
                            let coef = vmulq_f32(dsv, vsubq_f32(q[c], thirtytwo));
                            acc[ai & 7] = vfmaq_f32(acc[ai & 7], coef, xv);
                            ai += 1;
                        }
                    }
                }
            }
        }
        let mut s = vdupq_n_f32(0.0);
        for v in acc.iter() {
            s = vaddq_f32(s, *v);
        }
        out[i] = vaddvq_f32(s);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f")]
unsafe fn matvec_q6_k_w_f32_a_avx512(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 210;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;
    let mask_nibble = _mm512_set1_epi32(0x0F);
    let mask_2bit = _mm512_set1_epi32(0x03);
    let thirtytwo = _mm512_set1_epi32(32);

    // 16 outputs at a time. Loads 16 ql bytes + 16 qh bytes, extracts
    // the right nibble + 2-bit field, combines to 6-bit, subtracts 32
    // for the signed range, returns a ZMM i32x16.
    #[inline(always)]
    unsafe fn build_q_strand_avx512(
        ql_ptr: *const u8,
        qh_ptr: *const u8,
        shift: __m128i,
        is_high_nibble: bool,
        nibble_mask: __m512i,
        bit_mask: __m512i,
        offset_32: __m512i,
    ) -> __m512i {
        let ql_raw = _mm_loadu_si128(ql_ptr as *const __m128i);
        let ql_lane = _mm512_cvtepu8_epi32(ql_raw);
        let ql_nibble = if is_high_nibble {
            _mm512_and_si512(_mm512_srli_epi32(ql_lane, 4), nibble_mask)
        } else {
            _mm512_and_si512(ql_lane, nibble_mask)
        };
        let qh_raw = _mm_loadu_si128(qh_ptr as *const __m128i);
        let qh_lane = _mm512_cvtepu8_epi32(qh_raw);
        let qh_bits = _mm512_and_si512(_mm512_srl_epi32(qh_lane, shift), bit_mask);
        let qh_top = _mm512_slli_epi32(qh_bits, 4);
        _mm512_sub_epi32(_mm512_or_si512(ql_nibble, qh_top), offset_32)
    }

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        // 8 accumulators per row: 4 strands × 2 lanes (l=0..16 and
        // l=16..32) — half the count of the AVX2 path because the
        // ZMM lane is 2x wider.
        let mut acc = [_mm512_setzero_ps(); 8];

        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let ql_ptr = w_bytes.as_ptr().add(off);
            let qh_ptr = w_bytes.as_ptr().add(off + 128);
            let scales_ptr = w_bytes.as_ptr().add(off + 192) as *const i8;
            let d_bits = u16::from_le_bytes([w_bytes[off + 208], w_bytes[off + 209]]);
            let d_scalar = f16::from_bits(d_bits).to_f32();
            let xptr = x.as_ptr().add(b * QK_K);

            for n in 0..2 {
                let ql_base = ql_ptr.add(64 * n);
                let qh_base = qh_ptr.add(32 * n);
                let scales_base = scales_ptr.add(n * 8);
                let x_base = xptr.add(n * 128);

                // Same 4-strand layout as the AVX2 path.
                let strand_specs: [(usize, bool, i32, usize, usize); 4] = [
                    (0, false, 0, 0, 0),   // q1
                    (32, false, 2, 32, 1), // q2
                    (0, true, 4, 64, 2),   // q3
                    (32, true, 6, 96, 3),  // q4
                ];

                for (strand_idx, &(ql_off, hi, shift_v, x_off, scale_offset)) in
                    strand_specs.iter().enumerate()
                {
                    let shift = _mm_cvtsi32_si128(shift_v);
                    // Scale changes at the l=16 boundary (is=1 vs is=0).
                    let scale_a = (*scales_base.add(scale_offset * 2)) as f32;
                    let scale_b = (*scales_base.add(scale_offset * 2 + 1)) as f32;
                    let d_sa = _mm512_set1_ps(d_scalar * scale_a);
                    let d_sb = _mm512_set1_ps(d_scalar * scale_b);

                    let acc_base = strand_idx * 2;
                    // Lane 0: l = 0..16  with scale_a
                    let q0 = build_q_strand_avx512(
                        ql_base.add(ql_off),
                        qh_base,
                        shift,
                        hi,
                        mask_nibble,
                        mask_2bit,
                        thirtytwo,
                    );
                    let dq0 = _mm512_mul_ps(d_sa, _mm512_cvtepi32_ps(q0));
                    let xv0 = _mm512_loadu_ps(x_base.add(x_off));
                    acc[acc_base] = _mm512_fmadd_ps(dq0, xv0, acc[acc_base]);

                    // Lane 1: l = 16..32 with scale_b
                    let q1 = build_q_strand_avx512(
                        ql_base.add(ql_off + 16),
                        qh_base.add(16),
                        shift,
                        hi,
                        mask_nibble,
                        mask_2bit,
                        thirtytwo,
                    );
                    let dq1 = _mm512_mul_ps(d_sb, _mm512_cvtepi32_ps(q1));
                    let xv1 = _mm512_loadu_ps(x_base.add(x_off + 16));
                    acc[acc_base + 1] = _mm512_fmadd_ps(dq1, xv1, acc[acc_base + 1]);
                }
            }
        }

        let mut sum = acc[0];
        for v in &acc[1..] {
            sum = _mm512_add_ps(sum, *v);
        }
        out[i] = _mm512_reduce_add_ps(sum);
    }
}

fn matvec_q6_k_w_f32_a_scalar(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    const BLOCK_BYTES: usize = 210;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;
    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let ql = &w_bytes[off..off + 128];
            let qh = &w_bytes[off + 128..off + 128 + 64];
            let scales = &w_bytes[off + 192..off + 192 + 16];
            let d = f16::from_le_bytes([w_bytes[off + 208], w_bytes[off + 209]]).to_f32();
            let x_block = &x[b * QK_K..(b + 1) * QK_K];

            for n in 0..2 {
                for l in 0..32 {
                    let is = l / 16 + n * 8;
                    let q1 = ((ql[64 * n + l] & 0x0F) as i32
                        | (((qh[32 * n + l] >> 0) & 0x03) as i32) << 4)
                        - 32;
                    let q2 = ((ql[64 * n + l + 32] & 0x0F) as i32
                        | (((qh[32 * n + l] >> 2) & 0x03) as i32) << 4)
                        - 32;
                    let q3 = ((ql[64 * n + l] >> 4) as i32
                        | (((qh[32 * n + l] >> 4) & 0x03) as i32) << 4)
                        - 32;
                    let q4 = ((ql[64 * n + l + 32] >> 4) as i32
                        | (((qh[32 * n + l] >> 6) & 0x03) as i32) << 4)
                        - 32;
                    let s0 = (scales[is] as i8) as f32;
                    let s1 = (scales[is + 2] as i8) as f32;
                    let s2 = (scales[is + 4] as i8) as f32;
                    let s3 = (scales[is + 6] as i8) as f32;
                    let base = n * 128 + l;
                    acc += d * s0 * q1 as f32 * x_block[base];
                    acc += d * s1 * q2 as f32 * x_block[base + 32];
                    acc += d * s2 * q3 as f32 * x_block[base + 64];
                    acc += d * s3 * q4 as f32 * x_block[base + 96];
                }
            }
        }
        out[i] = acc;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn matvec_q6_k_w_f32_a_avx2(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 210;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;
    let mask_nibble = _mm256_set1_epi32(0x0F);
    let mask_2bit = _mm256_set1_epi32(0x03);
    let thirtytwo = _mm256_set1_epi32(32);

    // Process 8 outputs at a time (one AVX2 lane). For a given (n, l_chunk,
    // strand), extract 8 nibbles from ql at the right offset, 8 2-bit values
    // from qh at the right shift, combine to 6-bit, subtract 32, convert,
    // scale, FMA against the matching 8-wide x window.
    //
    // shift_count is the `__m128i`-wrapped shift amount for qh extraction
    // (0, 2, 4, or 6 — fixed per strand within the SIMD loop, supplied at
    // runtime so we use `_mm256_srl_epi32`, not the immediate-only `_srli`).
    #[inline(always)]
    unsafe fn build_q_strand(
        ql_ptr: *const u8,
        qh_ptr: *const u8,
        shift: __m128i,
        is_high_nibble: bool,
        nibble_mask: __m256i,
        bit_mask: __m256i,
        offset_32: __m256i,
    ) -> __m256i {
        // 8 ql bytes → 8 i32 lanes (zero-extended), then either low or high nibble.
        let ql_lane = _mm256_cvtepu8_epi32(_mm_loadl_epi64(ql_ptr as *const __m128i));
        let ql_nibble = if is_high_nibble {
            _mm256_and_si256(_mm256_srli_epi32(ql_lane, 4), nibble_mask)
        } else {
            _mm256_and_si256(ql_lane, nibble_mask)
        };
        // 8 qh bytes → 8 i32 lanes, shifted to get the right 2-bit field, then masked.
        let qh_lane = _mm256_cvtepu8_epi32(_mm_loadl_epi64(qh_ptr as *const __m128i));
        let qh_bits = _mm256_and_si256(_mm256_srl_epi32(qh_lane, shift), bit_mask);
        let qh_top = _mm256_slli_epi32(qh_bits, 4); // promote bits 0-1 → bits 4-5
        // Combine + recenter: (nibble | (qh<<4)) - 32
        _mm256_sub_epi32(_mm256_or_si256(ql_nibble, qh_top), offset_32)
    }

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        // 16 accumulators: per super-block we process 8 lanes-of-8 per half
        // (q1 lo half, q1 hi half, q2 lo half, q2 hi half, ...). Eight gives
        // enough FMA parallelism while keeping the reduction simple.
        let mut acc = [_mm256_setzero_ps(); 16];

        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let ql_ptr = w_bytes.as_ptr().add(off);
            let qh_ptr = w_bytes.as_ptr().add(off + 128);
            let scales_ptr = w_bytes.as_ptr().add(off + 192) as *const i8;
            let d_bits =
                u16::from_le_bytes([w_bytes[off + 208], w_bytes[off + 209]]);
            let d_scalar = f16::from_bits(d_bits).to_f32();
            let xptr = x.as_ptr().add(b * QK_K);

            for n in 0..2 {
                let ql_base = ql_ptr.add(64 * n);
                let qh_base = qh_ptr.add(32 * n);
                let scales_base = scales_ptr.add(n * 8);
                let x_base = xptr.add(n * 128);

                // 4 strands (q1..q4); each handles 32 outputs = 2 lanes × 8.
                // Per strand, scales[is*2 + strand_idx] for the two halves of l.
                //
                // Strand layout:
                //   q1: ql low nibble at offset 0,  qh shift 0,  x at offset 0
                //   q2: ql low nibble at offset 32, qh shift 2,  x at offset 32
                //   q3: ql high nibble at offset 0, qh shift 4,  x at offset 64
                //   q4: ql high nibble at offset 32, qh shift 6, x at offset 96
                let strand_specs: [(usize, bool, i32, usize, usize); 4] = [
                    (0, false, 0, 0, 0),  // q1
                    (32, false, 2, 32, 1), // q2
                    (0, true, 4, 64, 2),  // q3
                    (32, true, 6, 96, 3), // q4
                ];

                for (strand_idx, &(ql_off, hi, shift_v, x_off, scale_offset)) in
                    strand_specs.iter().enumerate()
                {
                    let shift = _mm_cvtsi32_si128(shift_v);
                    // Lane 1: l = 0..8 → is=0 (scale at scales[0 + scale_offset*2])
                    // Lane 2: l = 8..16 → is=0  (same scale)
                    // Lane 3: l = 16..24 → is=1 (scale at scales[1 + scale_offset*2])
                    // Lane 4: l = 24..32 → is=1
                    let scale_a = (*scales_base.add(scale_offset * 2)) as f32;
                    let scale_b = (*scales_base.add(scale_offset * 2 + 1)) as f32;
                    let d_sa = _mm256_set1_ps(d_scalar * scale_a);
                    let d_sb = _mm256_set1_ps(d_scalar * scale_b);

                    // 4 SIMD lanes worth (32 outputs total) per strand.
                    let lane_specs = [
                        (0usize, d_sa),  // l = 0..8
                        (8, d_sa),       // l = 8..16
                        (16, d_sb),      // l = 16..24
                        (24, d_sb),      // l = 24..32
                    ];

                    let acc_base = strand_idx * 4;
                    for (lane_idx, &(l_offset, d_s)) in lane_specs.iter().enumerate() {
                        let q = build_q_strand(
                            ql_base.add(ql_off + l_offset),
                            qh_base.add(l_offset),
                            shift,
                            hi,
                            mask_nibble,
                            mask_2bit,
                            thirtytwo,
                        );
                        let dq = _mm256_mul_ps(d_s, _mm256_cvtepi32_ps(q));
                        let xv = _mm256_loadu_ps(x_base.add(x_off + l_offset));
                        acc[acc_base + lane_idx] = _mm256_fmadd_ps(dq, xv, acc[acc_base + lane_idx]);
                    }
                }
            }
        }

        // Reduce 16 → 1 then horizontal sum.
        let mut sum = acc[0];
        for v in &acc[1..] {
            sum = _mm256_add_ps(sum, *v);
        }
        let mut sum128 =
            _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        out[i] = _mm_cvtss_f32(sum128);
    }
}

/// IQ4_XS codebook — non-linear 4-bit values biased toward zero.
/// Mirrors `KVALUES_IQ4NL` in [`rustllama_gguf::dequant`]. Duplicated
/// here so the kernel layer has no compile-time dependency on the
/// GGUF crate's internal constants.
pub const KVALUES_IQ4XS: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// IQ4_XS weight matvec with on-the-fly dequant.
///
/// Super-block layout (136 bytes per 256 weights):
///   { d: f16, scales_h: u16, scales_l: [u8; 4], qs: [u8; 128] }
///
/// Each weight is a 4-bit index into [`KVALUES_IQ4XS`]; eight 6-bit
/// signed sub-scales (-32..=31) are split across `scales_h` (high 2
/// bits per sub-block) and `scales_l` (low 4 bits per sub-block,
/// packed two-per-byte). Dequant: `value = d * sub_scale * kvalues[idx]`.
///
/// Requires `k % 256 == 0` (one full super-block per chunk).
pub fn matvec_iq4_xs_w_f32_a(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    const BLOCK_BYTES: usize = 136;
    const QK_K: usize = 256;
    debug_assert_eq!(k % QK_K, 0, "IQ4_XS matvec requires k % 256 == 0");
    let blocks_per_row = k / QK_K;
    debug_assert_eq!(w_bytes.len(), m * blocks_per_row * BLOCK_BYTES);
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(out.len(), m);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: runtime feature detection.
            unsafe { matvec_iq4_xs_w_f32_a_avx512(w_bytes, x, out, m, k) };
            return;
        }
        if is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && is_x86_feature_detected!("ssse3")
        {
            // SAFETY: runtime feature detection (PSHUFB needs SSSE3).
            unsafe { matvec_iq4_xs_w_f32_a_avx2(w_bytes, x, out, m, k) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64.
        unsafe { matvec_iq4_xs_w_f32_a_neon(w_bytes, x, out, m, k) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    matvec_iq4_xs_w_f32_a_scalar(w_bytes, x, out, m, k);
}

/// Widen an `int8x16` (16 signed bytes) to four `float32x4` — the signed
/// twin of [`u8x16_to_f32x4x4`], used by the IQ4 codebook kernels (here)
/// and the MXFP4/NVFP4 E2M1 kernels (in the `mxfp`/`nvfp4` submodules)
/// where the looked-up codebook values are signed i8.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub(crate) unsafe fn s8x16_to_f32x4x4(
    v: std::arch::aarch64::int8x16_t,
) -> [std::arch::aarch64::float32x4_t; 4] {
    use std::arch::aarch64::*;
    let lo16 = vmovl_s8(vget_low_s8(v));
    let hi16 = vmovl_s8(vget_high_s8(v));
    [
        vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16))),
        vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16))),
        vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16))),
        vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16))),
    ]
}

/// AArch64 NEON IQ4_XS matvec. The 4-bit codes index the signed `i8`
/// codebook [`KVALUES_IQ4XS`] — a `vqtbl1q_s8` 16-way table lookup (the
/// ARM analogue of the AVX2 PSHUFB path). 8 sub-blocks of 32 weights, each
/// with a 6-bit signed sub-scale (`lo4` from `scales_l`, `hi2` from
/// `scales_h`, bias 32). Low nibbles → outputs 0..15, high → 16..31. NEON
/// baseline; same tolerance contract as the AVX2 path.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn matvec_iq4_xs_w_f32_a_neon(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use std::arch::aarch64::*;
    const BLOCK_BYTES: usize = 136;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;
    let codebook = vld1q_s8(KVALUES_IQ4XS.as_ptr());
    let mask_lo = vdupq_n_u8(0x0F);
    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = [vdupq_n_f32(0.0); 4];
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d = f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32();
            let scales_h = u16::from_le_bytes([w_bytes[off + 2], w_bytes[off + 3]]);
            let scales_l = &w_bytes[off + 4..off + 8];
            let qs = w_bytes.as_ptr().add(off + 8);
            let xb = x.as_ptr().add(b * QK_K);
            for ib in 0..8 {
                let lo_nibble = if ib % 2 == 0 {
                    scales_l[ib / 2] & 0x0F
                } else {
                    scales_l[ib / 2] >> 4
                };
                let hi_bits = ((scales_h >> (2 * ib)) & 0x03) as u8;
                let ls = ((lo_nibble | (hi_bits << 4)) as i8 - 32) as f32;
                let sdv = vdupq_n_f32(d * ls);
                let qb16 = vld1q_u8(qs.add(ib * 16));
                // vqtbl1q_s8 takes a uint8x16 index (0..15 selects a lane).
                let lo_f = s8x16_to_f32x4x4(vqtbl1q_s8(codebook, vandq_u8(qb16, mask_lo)));
                let hi_f = s8x16_to_f32x4x4(vqtbl1q_s8(codebook, vshrq_n_u8::<4>(qb16)));
                let x_off = ib * 32;
                for c in 0..4 {
                    let xl = vld1q_f32(xb.add(x_off + c * 4));
                    let xh = vld1q_f32(xb.add(x_off + 16 + c * 4));
                    acc[c] = vfmaq_f32(acc[c], vmulq_f32(sdv, lo_f[c]), xl);
                    acc[c] = vfmaq_f32(acc[c], vmulq_f32(sdv, hi_f[c]), xh);
                }
            }
        }
        let s = vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3]));
        out[i] = vaddvq_f32(s);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma,ssse3")]
unsafe fn matvec_iq4_xs_w_f32_a_avx2(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 136;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;

    // 16-byte codebook in XMM. PSHUFB will use this as the table.
    // PSHUFB(table, indices) returns `table[indices[i] & 0x0F]` per
    // byte lane — a 1-cycle 16-way lookup, ideal for this codebook.
    let codebook = _mm_setr_epi8(
        KVALUES_IQ4XS[0],
        KVALUES_IQ4XS[1],
        KVALUES_IQ4XS[2],
        KVALUES_IQ4XS[3],
        KVALUES_IQ4XS[4],
        KVALUES_IQ4XS[5],
        KVALUES_IQ4XS[6],
        KVALUES_IQ4XS[7],
        KVALUES_IQ4XS[8],
        KVALUES_IQ4XS[9],
        KVALUES_IQ4XS[10],
        KVALUES_IQ4XS[11],
        KVALUES_IQ4XS[12],
        KVALUES_IQ4XS[13],
        KVALUES_IQ4XS[14],
        KVALUES_IQ4XS[15],
    );
    let mask_low = _mm_set1_epi8(0x0F);

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        // 4 × 8-wide YMM accumulators — one full 32-output sub-block
        // worth. Folded together at the end.
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d_bits = u16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]);
            let d = f16::from_bits(d_bits).to_f32();
            let scales_h = u16::from_le_bytes([w_bytes[off + 2], w_bytes[off + 3]]);
            let scales_l = [
                w_bytes[off + 4],
                w_bytes[off + 5],
                w_bytes[off + 6],
                w_bytes[off + 7],
            ];

            for ib in 0..8 {
                let lo_nibble = if ib % 2 == 0 {
                    scales_l[ib / 2] & 0x0F
                } else {
                    scales_l[ib / 2] >> 4
                };
                let hi_bits = ((scales_h >> (2 * ib)) & 0x03) as u8;
                let ls = (lo_nibble | (hi_bits << 4)) as i8 - 32;
                let sub_d = _mm256_set1_ps(d * (ls as f32));

                let q_ptr = w_bytes.as_ptr().add(off + 8 + ib * 16);
                let q = _mm_loadu_si128(q_ptr as *const __m128i);

                // Build 16-byte nibble-index vectors (one per nibble).
                // `srli_epi16(q, 4)` shifts within each i16 (allows
                // cross-byte spill into the next byte's low nibble),
                // but the final `& 0x0F` per-byte mask zeroes any
                // spillover so the result is correct per-byte.
                let lo_idx = _mm_and_si128(q, mask_low);
                let hi_idx = _mm_and_si128(_mm_srli_epi16(q, 4), mask_low);

                // 16-way PSHUFB lookup → 16 i8 codebook values per half.
                let lo_vals_i8 = _mm_shuffle_epi8(codebook, lo_idx);
                let hi_vals_i8 = _mm_shuffle_epi8(codebook, hi_idx);

                // Sign-extend 16 i8 → 16 i32 across two YMM (8 lanes
                // per YMM). `_mm256_cvtepi8_epi32` reads the LOWER 8
                // bytes of its XMM input; shifting the XMM up by 8
                // bytes gets us the upper half.
                let lo_upper = _mm_srli_si128(lo_vals_i8, 8);
                let hi_upper = _mm_srli_si128(hi_vals_i8, 8);
                let lo_first = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo_vals_i8));
                let lo_second = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo_upper));
                let hi_first = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi_vals_i8));
                let hi_second = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi_upper));

                // Scale each by sub_d before FMA-ing with x.
                let lo_first = _mm256_mul_ps(lo_first, sub_d);
                let lo_second = _mm256_mul_ps(lo_second, sub_d);
                let hi_first = _mm256_mul_ps(hi_first, sub_d);
                let hi_second = _mm256_mul_ps(hi_second, sub_d);

                let x_ptr = x.as_ptr().add(b * QK_K + ib * 32);
                let x0 = _mm256_loadu_ps(x_ptr);
                let x1 = _mm256_loadu_ps(x_ptr.add(8));
                let x2 = _mm256_loadu_ps(x_ptr.add(16));
                let x3 = _mm256_loadu_ps(x_ptr.add(24));

                acc0 = _mm256_fmadd_ps(lo_first, x0, acc0);
                acc1 = _mm256_fmadd_ps(lo_second, x1, acc1);
                acc2 = _mm256_fmadd_ps(hi_first, x2, acc2);
                acc3 = _mm256_fmadd_ps(hi_second, x3, acc3);
            }
        }

        let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        let mut sum128 =
            _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        out[i] = _mm_cvtss_f32(sum128);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f")]
unsafe fn matvec_iq4_xs_w_f32_a_avx512(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 136;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;

    // Pre-build the codebook as an i32 ZMM vector. VPERMD permutes
    // 16 × i32 lanes by 16 indices in a single instruction — much
    // faster than VPGATHERDD (latency 12+ vs VPERMD's 3, throughput 1).
    let codebook = _mm512_setr_epi32(
        KVALUES_IQ4XS[0] as i32,
        KVALUES_IQ4XS[1] as i32,
        KVALUES_IQ4XS[2] as i32,
        KVALUES_IQ4XS[3] as i32,
        KVALUES_IQ4XS[4] as i32,
        KVALUES_IQ4XS[5] as i32,
        KVALUES_IQ4XS[6] as i32,
        KVALUES_IQ4XS[7] as i32,
        KVALUES_IQ4XS[8] as i32,
        KVALUES_IQ4XS[9] as i32,
        KVALUES_IQ4XS[10] as i32,
        KVALUES_IQ4XS[11] as i32,
        KVALUES_IQ4XS[12] as i32,
        KVALUES_IQ4XS[13] as i32,
        KVALUES_IQ4XS[14] as i32,
        KVALUES_IQ4XS[15] as i32,
    );
    let nibble_mask = _mm512_set1_epi32(0x0F);

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        // Two 16-wide accumulators: one for the low-nibble half (lanes
        // 0..16 of each sub-block), one for the high-nibble half
        // (lanes 16..32). Folded together at the end.
        let mut acc_lo = _mm512_setzero_ps();
        let mut acc_hi = _mm512_setzero_ps();

        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d_bits = u16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]);
            let d = f16::from_bits(d_bits).to_f32();
            let scales_h = u16::from_le_bytes([w_bytes[off + 2], w_bytes[off + 3]]);
            let scales_l = [
                w_bytes[off + 4],
                w_bytes[off + 5],
                w_bytes[off + 6],
                w_bytes[off + 7],
            ];

            for ib in 0..8 {
                // Per-sub-block 6-bit signed scale (low 4 from scales_l,
                // high 2 from scales_h, then subtract 32 for the
                // signed bias).
                let lo_nibble = if ib % 2 == 0 {
                    scales_l[ib / 2] & 0x0F
                } else {
                    scales_l[ib / 2] >> 4
                };
                let hi_bits = ((scales_h >> (2 * ib)) & 0x03) as u8;
                let ls = (lo_nibble | (hi_bits << 4)) as i8 - 32;
                let sub_d = _mm512_set1_ps(d * (ls as f32));

                // Load 16 q-bytes for this sub-block.
                let q_ptr = w_bytes.as_ptr().add(off + 8 + ib * 16);
                let q_raw = _mm_loadu_si128(q_ptr as *const __m128i);

                // Widen to 16 × i32, then mask out the high/low nibbles.
                let q_wide = _mm512_cvtepu8_epi32(q_raw);
                let lo_idx = _mm512_and_si512(q_wide, nibble_mask);
                let hi_idx = _mm512_and_si512(_mm512_srli_epi32(q_wide, 4), nibble_mask);

                // Codebook lookup via VPERMD (one instruction, lat 3,
                // tput 1 on Ice Lake+).
                let lo_vals_i32 = _mm512_permutexvar_epi32(lo_idx, codebook);
                let hi_vals_i32 = _mm512_permutexvar_epi32(hi_idx, codebook);

                // Convert to f32 and scale by sub_d.
                let lo_vals = _mm512_mul_ps(_mm512_cvtepi32_ps(lo_vals_i32), sub_d);
                let hi_vals = _mm512_mul_ps(_mm512_cvtepi32_ps(hi_vals_i32), sub_d);

                // Load 16 × x for each half and FMA.
                let x_ptr = x.as_ptr().add(b * QK_K + ib * 32);
                let x_lo = _mm512_loadu_ps(x_ptr);
                let x_hi = _mm512_loadu_ps(x_ptr.add(16));
                acc_lo = _mm512_fmadd_ps(lo_vals, x_lo, acc_lo);
                acc_hi = _mm512_fmadd_ps(hi_vals, x_hi, acc_hi);
            }
        }

        out[i] = _mm512_reduce_add_ps(_mm512_add_ps(acc_lo, acc_hi));
    }
}

fn matvec_iq4_xs_w_f32_a_scalar(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    const BLOCK_BYTES: usize = 136;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;
    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d = f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32();
            let scales_h =
                u16::from_le_bytes([w_bytes[off + 2], w_bytes[off + 3]]);
            let scales_l = &w_bytes[off + 4..off + 8];
            let qs = &w_bytes[off + 8..off + 8 + 128];
            let x_block = &x[b * QK_K..(b + 1) * QK_K];

            // 8 sub-blocks of 32 weights each. Per sub-block: low 4
            // bits of the scale from scales_l, high 2 bits from
            // scales_h, signed bias of 32.
            for ib in 0..8 {
                let lo_nibble = if ib % 2 == 0 {
                    scales_l[ib / 2] & 0x0F
                } else {
                    scales_l[ib / 2] >> 4
                };
                let hi_bits = ((scales_h >> (2 * ib)) & 0x03) as u8;
                let ls = ((lo_nibble | (hi_bits << 4)) as i8 - 32) as f32;
                let sub_d = d * ls;
                let q_off = ib * 16;
                let x_off = ib * 32;
                for j in 0..16 {
                    let q = qs[q_off + j];
                    let lo = (q & 0x0F) as usize;
                    let hi = (q >> 4) as usize;
                    acc += sub_d * (KVALUES_IQ4XS[lo] as f32) * x_block[x_off + j];
                    acc += sub_d * (KVALUES_IQ4XS[hi] as f32) * x_block[x_off + 16 + j];
                }
            }
        }
        out[i] = acc;
    }
}

/// IQ4_NL weight matvec with on-the-fly dequant.
///
/// Block layout (18 bytes per 32 weights):
///   { d: f16, qs: [u8; 16] }
/// Each `qs` byte's low/high nibbles index [`KVALUES_IQ4XS`]. Output
/// value: `d * KVALUES_IQ4XS[idx]`. Simpler than IQ4_XS — single
/// per-block scale, no sub-block hierarchy.
///
/// Requires `k % 32 == 0`.
pub fn matvec_iq4_nl_w_f32_a(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    const BLOCK_BYTES: usize = 18;
    const QK: usize = 32;
    debug_assert_eq!(k % QK, 0, "IQ4_NL matvec requires k % 32 == 0");
    let blocks_per_row = k / QK;
    debug_assert_eq!(w_bytes.len(), m * blocks_per_row * BLOCK_BYTES);
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(out.len(), m);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: runtime feature detection.
            unsafe { matvec_iq4_nl_w_f32_a_avx512(w_bytes, x, out, m, k) };
            return;
        }
        if is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && is_x86_feature_detected!("ssse3")
        {
            // SAFETY: runtime feature detection (PSHUFB needs SSSE3).
            unsafe { matvec_iq4_nl_w_f32_a_avx2(w_bytes, x, out, m, k) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64.
        unsafe { matvec_iq4_nl_w_f32_a_neon(w_bytes, x, out, m, k) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    matvec_iq4_nl_w_f32_a_scalar(w_bytes, x, out, m, k);
}

/// AArch64 NEON IQ4_NL matvec. Single per-block f16 scale `d`; each `qs`
/// byte's low/high nibbles index the signed `i8` codebook
/// [`KVALUES_IQ4XS`] via `vqtbl1q_s8`. Low nibbles → outputs 0..15, high
/// → 16..31. NEON baseline; same tolerance contract as the AVX2 path.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn matvec_iq4_nl_w_f32_a_neon(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use std::arch::aarch64::*;
    const BLOCK_BYTES: usize = 18;
    const QK: usize = 32;
    let blocks_per_row = k / QK;
    let codebook = vld1q_s8(KVALUES_IQ4XS.as_ptr());
    let mask_lo = vdupq_n_u8(0x0F);
    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = [vdupq_n_f32(0.0); 4];
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let dv = vdupq_n_f32(f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32());
            let qb16 = vld1q_u8(w_bytes.as_ptr().add(off + 2));
            // vqtbl1q_s8 takes a uint8x16 index (0..15 selects a lane).
            let lo_f = s8x16_to_f32x4x4(vqtbl1q_s8(codebook, vandq_u8(qb16, mask_lo)));
            let hi_f = s8x16_to_f32x4x4(vqtbl1q_s8(codebook, vshrq_n_u8::<4>(qb16)));
            let xlo = x.as_ptr().add(b * QK);
            let xhi = xlo.add(16);
            for c in 0..4 {
                let xl = vld1q_f32(xlo.add(c * 4));
                let xh = vld1q_f32(xhi.add(c * 4));
                acc[c] = vfmaq_f32(acc[c], vmulq_f32(dv, lo_f[c]), xl);
                acc[c] = vfmaq_f32(acc[c], vmulq_f32(dv, hi_f[c]), xh);
            }
        }
        let s = vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3]));
        out[i] = vaddvq_f32(s);
    }
}

fn matvec_iq4_nl_w_f32_a_scalar(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    const BLOCK_BYTES: usize = 18;
    const QK: usize = 32;
    let blocks_per_row = k / QK;
    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d = f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32();
            let qs = &w_bytes[off + 2..off + 18];
            let x_chunk = &x[b * QK..(b + 1) * QK];
            for j in 0..16 {
                let q = qs[j];
                let lo = (q & 0x0F) as usize;
                let hi = (q >> 4) as usize;
                acc += d * (KVALUES_IQ4XS[lo] as f32) * x_chunk[j];
                acc += d * (KVALUES_IQ4XS[hi] as f32) * x_chunk[j + 16];
            }
        }
        out[i] = acc;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma,ssse3")]
unsafe fn matvec_iq4_nl_w_f32_a_avx2(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 18;
    const QK: usize = 32;
    let blocks_per_row = k / QK;

    let codebook = _mm_setr_epi8(
        KVALUES_IQ4XS[0], KVALUES_IQ4XS[1], KVALUES_IQ4XS[2], KVALUES_IQ4XS[3],
        KVALUES_IQ4XS[4], KVALUES_IQ4XS[5], KVALUES_IQ4XS[6], KVALUES_IQ4XS[7],
        KVALUES_IQ4XS[8], KVALUES_IQ4XS[9], KVALUES_IQ4XS[10], KVALUES_IQ4XS[11],
        KVALUES_IQ4XS[12], KVALUES_IQ4XS[13], KVALUES_IQ4XS[14], KVALUES_IQ4XS[15],
    );
    let mask_low = _mm_set1_epi8(0x0F);

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d_bits = u16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]);
            let d = _mm256_set1_ps(f16::from_bits(d_bits).to_f32());

            let q_ptr = w_bytes.as_ptr().add(off + 2);
            let q = _mm_loadu_si128(q_ptr as *const __m128i);
            let lo_idx = _mm_and_si128(q, mask_low);
            let hi_idx = _mm_and_si128(_mm_srli_epi16(q, 4), mask_low);
            let lo_vals_i8 = _mm_shuffle_epi8(codebook, lo_idx);
            let hi_vals_i8 = _mm_shuffle_epi8(codebook, hi_idx);

            // 16 i8 codebook values → 2 × 8-wide f32 each via sign-extend.
            let lo_upper = _mm_srli_si128(lo_vals_i8, 8);
            let hi_upper = _mm_srli_si128(hi_vals_i8, 8);
            let lo_first = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo_vals_i8)), d);
            let lo_second = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo_upper)), d);
            let hi_first = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi_vals_i8)), d);
            let hi_second = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi_upper)), d);

            let x_ptr = x.as_ptr().add(b * QK);
            let x0 = _mm256_loadu_ps(x_ptr);
            let x1 = _mm256_loadu_ps(x_ptr.add(8));
            let x2 = _mm256_loadu_ps(x_ptr.add(16));
            let x3 = _mm256_loadu_ps(x_ptr.add(24));

            acc0 = _mm256_fmadd_ps(lo_first, x0, acc0);
            acc1 = _mm256_fmadd_ps(lo_second, x1, acc1);
            acc2 = _mm256_fmadd_ps(hi_first, x2, acc2);
            acc3 = _mm256_fmadd_ps(hi_second, x3, acc3);
        }

        let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        let mut sum128 = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        out[i] = _mm_cvtss_f32(sum128);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f")]
unsafe fn matvec_iq4_nl_w_f32_a_avx512(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 18;
    const QK: usize = 32;
    let blocks_per_row = k / QK;

    let codebook = _mm512_setr_epi32(
        KVALUES_IQ4XS[0] as i32, KVALUES_IQ4XS[1] as i32,
        KVALUES_IQ4XS[2] as i32, KVALUES_IQ4XS[3] as i32,
        KVALUES_IQ4XS[4] as i32, KVALUES_IQ4XS[5] as i32,
        KVALUES_IQ4XS[6] as i32, KVALUES_IQ4XS[7] as i32,
        KVALUES_IQ4XS[8] as i32, KVALUES_IQ4XS[9] as i32,
        KVALUES_IQ4XS[10] as i32, KVALUES_IQ4XS[11] as i32,
        KVALUES_IQ4XS[12] as i32, KVALUES_IQ4XS[13] as i32,
        KVALUES_IQ4XS[14] as i32, KVALUES_IQ4XS[15] as i32,
    );
    let nibble_mask = _mm512_set1_epi32(0x0F);

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc_lo = _mm512_setzero_ps();
        let mut acc_hi = _mm512_setzero_ps();

        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d_bits = u16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]);
            let d = _mm512_set1_ps(f16::from_bits(d_bits).to_f32());

            let q_ptr = w_bytes.as_ptr().add(off + 2);
            let q_raw = _mm_loadu_si128(q_ptr as *const __m128i);
            let q_wide = _mm512_cvtepu8_epi32(q_raw);
            let lo_idx = _mm512_and_si512(q_wide, nibble_mask);
            let hi_idx = _mm512_and_si512(_mm512_srli_epi32(q_wide, 4), nibble_mask);

            let lo_vals = _mm512_mul_ps(
                _mm512_cvtepi32_ps(_mm512_permutexvar_epi32(lo_idx, codebook)), d);
            let hi_vals = _mm512_mul_ps(
                _mm512_cvtepi32_ps(_mm512_permutexvar_epi32(hi_idx, codebook)), d);

            let x_ptr = x.as_ptr().add(b * QK);
            let x_lo = _mm512_loadu_ps(x_ptr);
            let x_hi = _mm512_loadu_ps(x_ptr.add(16));
            acc_lo = _mm512_fmadd_ps(lo_vals, x_lo, acc_lo);
            acc_hi = _mm512_fmadd_ps(hi_vals, x_hi, acc_hi);
        }

        out[i] = _mm512_reduce_add_ps(_mm512_add_ps(acc_lo, acc_hi));
    }
}

/// Q5_K weight matvec with on-the-fly dequant.
///
/// Super-block layout (176 bytes per 256 weights):
///   { d: f16, dmin: f16, scales: [u8; 12], qh: [u8; 32], qs: [u8; 128] }
///
/// Identical to Q4_K plus the `qh` array: for each output position `l` in
/// `0..32` (within one group of 64 outputs), the 5th bit of the low/high
/// half comes from `qh[l]`'s bit position `group*2` / `group*2+1`. The
/// 4 groups all read the same 32 `qh` bytes but at different bit offsets.
pub fn matvec_q5_k_w_f32_a(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    const BLOCK_BYTES: usize = 176;
    const QK_K: usize = 256;
    debug_assert_eq!(k % QK_K, 0, "Q5_K matvec requires k % 256 == 0");
    let blocks_per_row = k / QK_K;
    debug_assert_eq!(w_bytes.len(), m * blocks_per_row * BLOCK_BYTES);
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(out.len(), m);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: runtime feature detection.
            unsafe { matvec_q5_k_w_f32_a_avx512(w_bytes, x, out, m, k) };
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime feature detection above.
            unsafe { matvec_q5_k_w_f32_a_avx2(w_bytes, x, out, m, k) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64.
        unsafe { matvec_q5_k_w_f32_a_neon(w_bytes, x, out, m, k) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    matvec_q5_k_w_f32_a_scalar(w_bytes, x, out, m, k);
}

/// AArch64 NEON Q5_K matvec. Identical to the Q5_K scalar / Q4_K NEON
/// structure (4 groups of 64 outputs, each 32 low + 32 high nibbles with
/// per-sub-block `d*q - m`), plus the 5th bit: for group `g`, the low
/// half's extra bit is `qh[l]` bit `g*2`, the high half's is bit `g*2+1`
/// (the same 32 `qh` bytes read at different bit offsets per group). The
/// extracted bit is promoted to 0x10 and OR'd into the nibble before the
/// unsigned widen. NEON baseline; same tolerance contract as the AVX2 path.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn matvec_q5_k_w_f32_a_neon(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use std::arch::aarch64::*;
    const BLOCK_BYTES: usize = 176;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;
    let mask_lo = vdupq_n_u8(0x0F);
    let one = vdupq_n_u8(1);
    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = [vdupq_n_f32(0.0); 4];
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d = f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32();
            let dmin = f16::from_le_bytes([w_bytes[off + 2], w_bytes[off + 3]]).to_f32();
            let (sc, mn) = unpack_q4k_scales(&w_bytes[off + 4..off + 16]);
            let qh = w_bytes.as_ptr().add(off + 16);
            let qs = w_bytes.as_ptr().add(off + 48);
            let xb = x.as_ptr().add(b * QK_K);
            for group in 0..4 {
                let qptr = qs.add(group * 32);
                let d_lo = d * sc[group * 2] as f32;
                let m_lo = vdupq_n_f32(dmin * mn[group * 2] as f32);
                let d_hi = d * sc[group * 2 + 1] as f32;
                let m_hi = vdupq_n_f32(dmin * mn[group * 2 + 1] as f32);
                let neg_bit_lo = vdupq_n_s8(-((group * 2) as i8));
                let neg_bit_hi = vdupq_n_s8(-((group * 2 + 1) as i8));
                let xlo = xb.add(group * 64);
                let xhi = xb.add(group * 64 + 32);
                let mut ai = 0usize;
                for half in 0..2 {
                    let base = half * 16;
                    let qb16 = vld1q_u8(qptr.add(base));
                    let qh16 = vld1q_u8(qh.add(base));
                    // 5th bit → 0x10 at each lane: ((qh >> bit) & 1) << 4.
                    let bit_lo = vshlq_n_u8::<4>(vandq_u8(vshlq_u8(qh16, neg_bit_lo), one));
                    let bit_hi = vshlq_n_u8::<4>(vandq_u8(vshlq_u8(qh16, neg_bit_hi), one));
                    let qlo_f = u8x16_to_f32x4x4(vorrq_u8(vandq_u8(qb16, mask_lo), bit_lo));
                    let qhi_f = u8x16_to_f32x4x4(vorrq_u8(vshrq_n_u8::<4>(qb16), bit_hi));
                    for c in 0..4 {
                        let idx = base + c * 4;
                        let xl = vld1q_f32(xlo.add(idx));
                        let xh = vld1q_f32(xhi.add(idx));
                        let coef_lo = vsubq_f32(vmulq_n_f32(qlo_f[c], d_lo), m_lo);
                        let coef_hi = vsubq_f32(vmulq_n_f32(qhi_f[c], d_hi), m_hi);
                        acc[ai & 3] = vfmaq_f32(acc[ai & 3], coef_lo, xl);
                        acc[(ai + 1) & 3] = vfmaq_f32(acc[(ai + 1) & 3], coef_hi, xh);
                        ai += 2;
                    }
                }
            }
        }
        let s = vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3]));
        out[i] = vaddvq_f32(s);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f")]
unsafe fn matvec_q5_k_w_f32_a_avx512(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 176;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;
    let mask_low_nibble = _mm512_set1_epi32(0x0F);
    let one = _mm512_set1_epi32(1);

    // Same nibble-split as Q4_K but with the 5th bit pulled in from qh.
    // Each 16-q-byte half yields (lo, hi) ZMM i32x16 lanes of bare nibbles.
    #[inline(always)]
    unsafe fn unpack_16_q_bytes_avx512(
        qs_ptr: *const u8,
        mask: __m512i,
    ) -> (__m512i, __m512i) {
        let raw = _mm_loadu_si128(qs_ptr as *const __m128i);
        let widened = _mm512_cvtepu8_epi32(raw);
        let lo = _mm512_and_si512(widened, mask);
        let hi = _mm512_and_si512(_mm512_srli_epi32(widened, 4), mask);
        (lo, hi)
    }

    // Load 16 qh bytes, extract bit at `bit_pos`, shift to position 4 so
    // it OR's directly into the 5th bit of a nibble.
    #[inline(always)]
    unsafe fn qh_high_bits_avx512(
        qh_ptr: *const u8,
        shift_count: __m128i,
        one: __m512i,
    ) -> __m512i {
        let raw = _mm_loadu_si128(qh_ptr as *const __m128i);
        let widened = _mm512_cvtepu8_epi32(raw);
        let shifted = _mm512_srl_epi32(widened, shift_count);
        let bit = _mm512_and_si512(shifted, one);
        _mm512_slli_epi32(bit, 4)
    }

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();
        let mut acc2 = _mm512_setzero_ps();
        let mut acc3 = _mm512_setzero_ps();

        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d_bits = u16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]);
            let dmin_bits = u16::from_le_bytes([w_bytes[off + 2], w_bytes[off + 3]]);
            let d_scalar = f16::from_bits(d_bits).to_f32();
            let dmin_scalar = f16::from_bits(dmin_bits).to_f32();
            let (sc, mn) = unpack_q4k_scales(&w_bytes[off + 4..off + 16]);
            let qh_ptr = w_bytes.as_ptr().add(off + 16);
            let qs_ptr = w_bytes.as_ptr().add(off + 48);
            let xptr = x.as_ptr().add(b * QK_K);

            for group in 0..4 {
                let group_qs = qs_ptr.add(group * 32);
                let group_x = xptr.add(group * 64);
                let shift_lo = _mm_cvtsi32_si128((group * 2) as i32);
                let shift_hi = _mm_cvtsi32_si128((group * 2 + 1) as i32);

                let (lo_a, hi_a) = unpack_16_q_bytes_avx512(group_qs, mask_low_nibble);
                let (lo_b, hi_b) = unpack_16_q_bytes_avx512(group_qs.add(16), mask_low_nibble);

                // qh high bits for each 16-lane half.
                let qhi_lo_a = qh_high_bits_avx512(qh_ptr, shift_lo, one);
                let qhi_lo_b = qh_high_bits_avx512(qh_ptr.add(16), shift_lo, one);
                let qhi_hi_a = qh_high_bits_avx512(qh_ptr, shift_hi, one);
                let qhi_hi_b = qh_high_bits_avx512(qh_ptr.add(16), shift_hi, one);

                // 5-bit values: nibble | (qh_bit << 4).
                let q_lo_a = _mm512_or_si512(lo_a, qhi_lo_a);
                let q_lo_b = _mm512_or_si512(lo_b, qhi_lo_b);
                let q_hi_a = _mm512_or_si512(hi_a, qhi_hi_a);
                let q_hi_b = _mm512_or_si512(hi_b, qhi_hi_b);

                let d_lo = _mm512_set1_ps(d_scalar * sc[group * 2] as f32);
                let m_lo = _mm512_set1_ps(dmin_scalar * mn[group * 2] as f32);
                let d_hi = _mm512_set1_ps(d_scalar * sc[group * 2 + 1] as f32);
                let m_hi = _mm512_set1_ps(dmin_scalar * mn[group * 2 + 1] as f32);

                let lo_a_f = _mm512_fmsub_ps(_mm512_cvtepi32_ps(q_lo_a), d_lo, m_lo);
                let lo_b_f = _mm512_fmsub_ps(_mm512_cvtepi32_ps(q_lo_b), d_lo, m_lo);
                let hi_a_f = _mm512_fmsub_ps(_mm512_cvtepi32_ps(q_hi_a), d_hi, m_hi);
                let hi_b_f = _mm512_fmsub_ps(_mm512_cvtepi32_ps(q_hi_b), d_hi, m_hi);

                let x_lo_a = _mm512_loadu_ps(group_x);
                let x_lo_b = _mm512_loadu_ps(group_x.add(16));
                let x_hi_a = _mm512_loadu_ps(group_x.add(32));
                let x_hi_b = _mm512_loadu_ps(group_x.add(48));

                acc0 = _mm512_fmadd_ps(lo_a_f, x_lo_a, acc0);
                acc1 = _mm512_fmadd_ps(lo_b_f, x_lo_b, acc1);
                acc2 = _mm512_fmadd_ps(hi_a_f, x_hi_a, acc2);
                acc3 = _mm512_fmadd_ps(hi_b_f, x_hi_b, acc3);
            }
        }

        let total = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
        out[i] = _mm512_reduce_add_ps(total);
    }
}

fn matvec_q5_k_w_f32_a_scalar(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    const BLOCK_BYTES: usize = 176;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;
    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d = f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32();
            let dmin = f16::from_le_bytes([w_bytes[off + 2], w_bytes[off + 3]]).to_f32();
            let (sc, mn) = unpack_q4k_scales(&w_bytes[off + 4..off + 16]);
            let qh = &w_bytes[off + 16..off + 48];
            let qs = &w_bytes[off + 48..off + 48 + 128];
            let x_block = &x[b * QK_K..(b + 1) * QK_K];

            for group in 0..4 {
                let q_chunk = &qs[group * 32..(group + 1) * 32];
                let d_lo = d * sc[group * 2] as f32;
                let m_lo = dmin * mn[group * 2] as f32;
                let d_hi = d * sc[group * 2 + 1] as f32;
                let m_hi = dmin * mn[group * 2 + 1] as f32;
                let bit_lo = group * 2;
                let bit_hi = group * 2 + 1;
                let x_lo = &x_block[group * 64..group * 64 + 32];
                let x_hi = &x_block[group * 64 + 32..(group + 1) * 64];
                for l in 0..32 {
                    let q_byte = q_chunk[l];
                    let qh_byte = qh[l];
                    let lo = (q_byte & 0x0F) as u32;
                    let hi = (q_byte >> 4) as u32;
                    let bit_lo_set = ((qh_byte >> bit_lo) & 1) as u32;
                    let bit_hi_set = ((qh_byte >> bit_hi) & 1) as u32;
                    let lo_full = lo | (bit_lo_set << 4);
                    let hi_full = hi | (bit_hi_set << 4);
                    acc += (d_lo * lo_full as f32 - m_lo) * x_lo[l];
                    acc += (d_hi * hi_full as f32 - m_hi) * x_hi[l];
                }
            }
        }
        out[i] = acc;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn matvec_q5_k_w_f32_a_avx2(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 176;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;
    let mask_low_nibble = _mm256_set1_epi32(0x0F);
    let one = _mm256_set1_epi32(1);

    // Same nibble-unpacking helper as Q4_K. Inline for clarity.
    #[inline(always)]
    unsafe fn unpack_32_q_bytes(
        qs_ptr: *const u8,
        mask: __m256i,
    ) -> (
        __m256i, __m256i, __m256i, __m256i,
        __m256i, __m256i, __m256i, __m256i,
    ) {
        let l07 = _mm256_and_si256(
            _mm256_cvtepu8_epi32(_mm_loadl_epi64(qs_ptr as *const __m128i)),
            mask,
        );
        let l815 = _mm256_and_si256(
            _mm256_cvtepu8_epi32(_mm_loadl_epi64(qs_ptr.add(8) as *const __m128i)),
            mask,
        );
        let l1623 = _mm256_and_si256(
            _mm256_cvtepu8_epi32(_mm_loadl_epi64(qs_ptr.add(16) as *const __m128i)),
            mask,
        );
        let l2431 = _mm256_and_si256(
            _mm256_cvtepu8_epi32(_mm_loadl_epi64(qs_ptr.add(24) as *const __m128i)),
            mask,
        );
        let h07 = _mm256_and_si256(
            _mm256_srli_epi32(
                _mm256_cvtepu8_epi32(_mm_loadl_epi64(qs_ptr as *const __m128i)),
                4,
            ),
            mask,
        );
        let h815 = _mm256_and_si256(
            _mm256_srli_epi32(
                _mm256_cvtepu8_epi32(_mm_loadl_epi64(qs_ptr.add(8) as *const __m128i)),
                4,
            ),
            mask,
        );
        let h1623 = _mm256_and_si256(
            _mm256_srli_epi32(
                _mm256_cvtepu8_epi32(_mm_loadl_epi64(qs_ptr.add(16) as *const __m128i)),
                4,
            ),
            mask,
        );
        let h2431 = _mm256_and_si256(
            _mm256_srli_epi32(
                _mm256_cvtepu8_epi32(_mm_loadl_epi64(qs_ptr.add(24) as *const __m128i)),
                4,
            ),
            mask,
        );
        (l07, l815, l1623, l2431, h07, h815, h1623, h2431)
    }

    // Extract a single bit from each qh byte at `bit_pos`, then shift to
    // position 4 so it can be OR'd directly into the 5th bit of a nibble.
    // `_mm256_srl_epi32` takes a runtime shift count packed in a __m128i
    // (only the low 64 bits matter); this is the correct form when the
    // bit position varies per group at runtime.
    #[inline(always)]
    unsafe fn qh_high_bits(qh_ptr: *const u8, shift_count: __m128i, one: __m256i) -> __m256i {
        let qh_lane = _mm256_cvtepu8_epi32(_mm_loadl_epi64(qh_ptr as *const __m128i));
        let shifted = _mm256_srl_epi32(qh_lane, shift_count);
        let bit = _mm256_and_si256(shifted, one);
        _mm256_slli_epi32(bit, 4)
    }

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = [_mm256_setzero_ps(); 8];

        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d_bits = u16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]);
            let dmin_bits = u16::from_le_bytes([w_bytes[off + 2], w_bytes[off + 3]]);
            let d_scalar = f16::from_bits(d_bits).to_f32();
            let dmin_scalar = f16::from_bits(dmin_bits).to_f32();
            let (sc, mn) = unpack_q4k_scales(&w_bytes[off + 4..off + 16]);
            let qh_ptr = w_bytes.as_ptr().add(off + 16);
            let qs_ptr = w_bytes.as_ptr().add(off + 48);
            let xptr = x.as_ptr().add(b * QK_K);

            for group in 0..4 {
                let group_qs = qs_ptr.add(group * 32);
                let group_x = xptr.add(group * 64);
                let shift_lo = _mm_cvtsi32_si128((group * 2) as i32);
                let shift_hi = _mm_cvtsi32_si128((group * 2 + 1) as i32);

                let (l07, l815, l1623, l2431, h07, h815, h1623, h2431) =
                    unpack_32_q_bytes(group_qs, mask_low_nibble);

                // qh high bits for each of the 32 output positions across both halves.
                let lo_hi_07 = qh_high_bits(qh_ptr, shift_lo, one);
                let lo_hi_815 = qh_high_bits(qh_ptr.add(8), shift_lo, one);
                let lo_hi_1623 = qh_high_bits(qh_ptr.add(16), shift_lo, one);
                let lo_hi_2431 = qh_high_bits(qh_ptr.add(24), shift_lo, one);
                let hi_hi_07 = qh_high_bits(qh_ptr, shift_hi, one);
                let hi_hi_815 = qh_high_bits(qh_ptr.add(8), shift_hi, one);
                let hi_hi_1623 = qh_high_bits(qh_ptr.add(16), shift_hi, one);
                let hi_hi_2431 = qh_high_bits(qh_ptr.add(24), shift_hi, one);

                // 5-bit values: nibble | (qh_bit << 4). No subtraction needed
                // (Q5_K is unsigned 5-bit + dmin offset).
                let q_lo_07 = _mm256_or_si256(l07, lo_hi_07);
                let q_lo_815 = _mm256_or_si256(l815, lo_hi_815);
                let q_lo_1623 = _mm256_or_si256(l1623, lo_hi_1623);
                let q_lo_2431 = _mm256_or_si256(l2431, lo_hi_2431);
                let q_hi_07 = _mm256_or_si256(h07, hi_hi_07);
                let q_hi_815 = _mm256_or_si256(h815, hi_hi_815);
                let q_hi_1623 = _mm256_or_si256(h1623, hi_hi_1623);
                let q_hi_2431 = _mm256_or_si256(h2431, hi_hi_2431);

                let d_lo = _mm256_set1_ps(d_scalar * sc[group * 2] as f32);
                let m_lo = _mm256_set1_ps(dmin_scalar * mn[group * 2] as f32);
                let d_hi = _mm256_set1_ps(d_scalar * sc[group * 2 + 1] as f32);
                let m_hi = _mm256_set1_ps(dmin_scalar * mn[group * 2 + 1] as f32);

                // value = d_lo * q - m_lo  (combined via fmsub: q*d_lo - m_lo)
                let lo_v_07 = _mm256_fmsub_ps(_mm256_cvtepi32_ps(q_lo_07), d_lo, m_lo);
                let lo_v_815 = _mm256_fmsub_ps(_mm256_cvtepi32_ps(q_lo_815), d_lo, m_lo);
                let lo_v_1623 = _mm256_fmsub_ps(_mm256_cvtepi32_ps(q_lo_1623), d_lo, m_lo);
                let lo_v_2431 = _mm256_fmsub_ps(_mm256_cvtepi32_ps(q_lo_2431), d_lo, m_lo);
                let hi_v_07 = _mm256_fmsub_ps(_mm256_cvtepi32_ps(q_hi_07), d_hi, m_hi);
                let hi_v_815 = _mm256_fmsub_ps(_mm256_cvtepi32_ps(q_hi_815), d_hi, m_hi);
                let hi_v_1623 = _mm256_fmsub_ps(_mm256_cvtepi32_ps(q_hi_1623), d_hi, m_hi);
                let hi_v_2431 = _mm256_fmsub_ps(_mm256_cvtepi32_ps(q_hi_2431), d_hi, m_hi);

                let x_lo_07 = _mm256_loadu_ps(group_x);
                let x_lo_815 = _mm256_loadu_ps(group_x.add(8));
                let x_lo_1623 = _mm256_loadu_ps(group_x.add(16));
                let x_lo_2431 = _mm256_loadu_ps(group_x.add(24));
                let x_hi_07 = _mm256_loadu_ps(group_x.add(32));
                let x_hi_815 = _mm256_loadu_ps(group_x.add(40));
                let x_hi_1623 = _mm256_loadu_ps(group_x.add(48));
                let x_hi_2431 = _mm256_loadu_ps(group_x.add(56));

                acc[0] = _mm256_fmadd_ps(lo_v_07, x_lo_07, acc[0]);
                acc[1] = _mm256_fmadd_ps(lo_v_815, x_lo_815, acc[1]);
                acc[2] = _mm256_fmadd_ps(lo_v_1623, x_lo_1623, acc[2]);
                acc[3] = _mm256_fmadd_ps(lo_v_2431, x_lo_2431, acc[3]);
                acc[4] = _mm256_fmadd_ps(hi_v_07, x_hi_07, acc[4]);
                acc[5] = _mm256_fmadd_ps(hi_v_815, x_hi_815, acc[5]);
                acc[6] = _mm256_fmadd_ps(hi_v_1623, x_hi_1623, acc[6]);
                acc[7] = _mm256_fmadd_ps(hi_v_2431, x_hi_2431, acc[7]);
            }
        }

        let s01 = _mm256_add_ps(acc[0], acc[1]);
        let s23 = _mm256_add_ps(acc[2], acc[3]);
        let s45 = _mm256_add_ps(acc[4], acc[5]);
        let s67 = _mm256_add_ps(acc[6], acc[7]);
        let total = _mm256_add_ps(_mm256_add_ps(s01, s23), _mm256_add_ps(s45, s67));
        let mut sum128 =
            _mm_add_ps(_mm256_castps256_ps128(total), _mm256_extractf128_ps(total, 1));
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        out[i] = _mm_cvtss_f32(sum128);
    }
}

/// Q2_K weight matvec with on-the-fly dequant. Dispatches to AVX-512 /
/// AVX2 / scalar by runtime feature detection.
///
/// Block layout (84 bytes per 256 weights):
///   { scales: [u8; 16], qs: [u8; 64], d: f16, dmin: f16 }
///
/// Sub-block decode: see [`rustllama_gguf::dequant::dequant_q2_k`]. Each
/// 16-weight sub-block has its own `(scale, min)` 4-bit pair: a scale
/// byte's low nibble is `scale`, high nibble is `min`. Per-weight value
/// is `d * scale * q - dmin * min` with `q ∈ [0, 3]`. 16 sub-blocks
/// per 256-weight super-block.
///
/// Requires `k % 256 == 0`.
pub fn matvec_q2_k_w_f32_a(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    const BLOCK_BYTES: usize = 84;
    const QK_K: usize = 256;
    debug_assert_eq!(k % QK_K, 0, "Q2_K matvec requires k % 256 == 0");
    let blocks_per_row = k / QK_K;
    debug_assert_eq!(w_bytes.len(), m * blocks_per_row * BLOCK_BYTES);
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(out.len(), m);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: runtime feature detection above.
            unsafe { matvec_q2_k_w_f32_a_avx512(w_bytes, x, out, m, k) };
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime feature detection above.
            unsafe { matvec_q2_k_w_f32_a_avx2(w_bytes, x, out, m, k) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64.
        unsafe { matvec_q2_k_w_f32_a_neon(w_bytes, x, out, m, k) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    matvec_q2_k_w_f32_a_scalar(w_bytes, x, out, m, k);
}

/// AArch64 NEON Q2_K matvec. Mirrors the scalar layout: each 256-weight
/// super-block is 16 sub-blocks of 16 weights (2 q-chunks × 4 shifts × 2
/// halves A/B). A sub-block's 2-bit codes are `(q >> shift) & 3`; its
/// scale byte gives `dl = d*(sc & 0xF)` and `ml = dmin*(sc >> 4)`, and the
/// contribution is `(dl*q - ml)*x`. The runtime `shift` (0/2/4/6) feeds
/// `vshlq_u8` as a negative per-lane count. NEON baseline; same tolerance
/// contract as the AVX2 path.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn matvec_q2_k_w_f32_a_neon(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use std::arch::aarch64::*;
    const BLOCK_BYTES: usize = 84;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;
    let three = vdupq_n_u8(0x03);
    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = [vdupq_n_f32(0.0); 4];
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let scales = &w_bytes[off..off + 16];
            let qs = w_bytes.as_ptr().add(off + 16);
            let d = f16::from_le_bytes([w_bytes[off + 80], w_bytes[off + 81]]).to_f32();
            let dmin = f16::from_le_bytes([w_bytes[off + 82], w_bytes[off + 83]]).to_f32();
            let xb = x.as_ptr().add(b * QK_K);
            let mut x_off = 0usize;
            let mut is = 0usize;
            for chunk in 0..2 {
                let qc = qs.add(chunk * 32);
                for shift in [0i8, 2, 4, 6] {
                    let neg = vdupq_n_s8(-shift);
                    // Sub-block A: q[0..16] >> shift.
                    let sc_a = scales[is];
                    let dl_a = d * (sc_a & 0xF) as f32;
                    let ml_a = vdupq_n_f32(dmin * (sc_a >> 4) as f32);
                    let qa = u8x16_to_f32x4x4(vandq_u8(vshlq_u8(vld1q_u8(qc), neg), three));
                    for c in 0..4 {
                        let xv = vld1q_f32(xb.add(x_off + c * 4));
                        acc[c] = vfmaq_f32(acc[c], vsubq_f32(vmulq_n_f32(qa[c], dl_a), ml_a), xv);
                    }
                    x_off += 16;
                    is += 1;
                    // Sub-block B: q[16..32] >> shift.
                    let sc_b = scales[is];
                    let dl_b = d * (sc_b & 0xF) as f32;
                    let ml_b = vdupq_n_f32(dmin * (sc_b >> 4) as f32);
                    let qb16 = u8x16_to_f32x4x4(vandq_u8(vshlq_u8(vld1q_u8(qc.add(16)), neg), three));
                    for c in 0..4 {
                        let xv = vld1q_f32(xb.add(x_off + c * 4));
                        acc[c] = vfmaq_f32(acc[c], vsubq_f32(vmulq_n_f32(qb16[c], dl_b), ml_b), xv);
                    }
                    x_off += 16;
                    is += 1;
                }
            }
        }
        let s = vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3]));
        out[i] = vaddvq_f32(s);
    }
}

fn matvec_q2_k_w_f32_a_scalar(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    const BLOCK_BYTES: usize = 84;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let scales = &w_bytes[off..off + 16];
            let qs = &w_bytes[off + 16..off + 16 + 64];
            let d = f16::from_le_bytes([w_bytes[off + 80], w_bytes[off + 81]]).to_f32();
            let dmin = f16::from_le_bytes([w_bytes[off + 82], w_bytes[off + 83]]).to_f32();

            let x_block = &x[b * QK_K..(b + 1) * QK_K];
            let mut x_off = 0usize;
            let mut is = 0usize;
            for chunk in 0..2 {
                let q = &qs[chunk * 32..chunk * 32 + 32];
                for shift in [0u32, 2, 4, 6] {
                    // Sub-block A: q[0..16] >> shift
                    let sc_a = scales[is];
                    let dl_a = d * (sc_a & 0xF) as f32;
                    let ml_a = dmin * (sc_a >> 4) as f32;
                    for l in 0..16 {
                        let q_val = ((q[l] >> shift) & 3) as f32;
                        acc += (dl_a * q_val - ml_a) * x_block[x_off + l];
                    }
                    x_off += 16;
                    is += 1;
                    // Sub-block B: q[16..32] >> shift
                    let sc_b = scales[is];
                    let dl_b = d * (sc_b & 0xF) as f32;
                    let ml_b = dmin * (sc_b >> 4) as f32;
                    for l in 0..16 {
                        let q_val = ((q[l + 16] >> shift) & 3) as f32;
                        acc += (dl_b * q_val - ml_b) * x_block[x_off + l];
                    }
                    x_off += 16;
                    is += 1;
                }
            }
        }
        out[i] = acc;
    }
}

/// Q2_K matvec — AVX-512 path. Processes one 16-weight sub-block per
/// inner iteration:
///   - Load 16 `qs` bytes, widen to `[i32; 16]`, shift+mask to get the
///     2-bit value `q ∈ [0, 3]`.
///   - Convert to f32, FMA with broadcast `dl = d * (scale & 0xF)`
///     using `_mm512_fmsub_ps(dl, q_f, ml)` to fold the per-sub-block
///     `ml = dmin * (scale >> 4)` subtraction.
///   - FMA the resulting value into the running accumulator.
///
/// Four FMA accumulators round-robin per j-iteration so the FP pipeline
/// stays full despite the per-iteration scale broadcast.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f")]
unsafe fn matvec_q2_k_w_f32_a_avx512(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 84;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;
    let three = _mm512_set1_epi32(3);

    // Inner kernel for one 16-weight sub-block at a const SHIFT. The
    // shift is const-generic so the AVX-512 srli intrinsic's compile-
    // time shift requirement is met.
    unsafe fn lane_512<const SHIFT: u32>(
        qs_ptr: *const u8,
        xptr: *const f32,
        q_off: usize,
        x_off: usize,
        dl: f32,
        ml: f32,
        three: __m512i,
    ) -> __m512 {
        use std::arch::x86_64::*;
        let qs_raw = _mm_loadu_si128(qs_ptr.add(q_off) as *const __m128i);
        let qs_w = _mm512_cvtepu8_epi32(qs_raw);
        let q_val = _mm512_and_si512(_mm512_srli_epi32(qs_w, SHIFT), three);
        let q_f = _mm512_cvtepi32_ps(q_val);
        let dl_v = _mm512_set1_ps(dl);
        let ml_v = _mm512_set1_ps(ml);
        // value = dl * q - ml (fused into one instruction).
        let value_f = _mm512_fmsub_ps(dl_v, q_f, ml_v);
        let x_v = _mm512_loadu_ps(xptr.add(x_off));
        _mm512_mul_ps(value_f, x_v)
    }

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();
        let mut acc2 = _mm512_setzero_ps();
        let mut acc3 = _mm512_setzero_ps();

        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let scales = &w_bytes[off..off + 16];
            let qs_ptr = w_bytes.as_ptr().add(off + 16);
            let d = f16::from_le_bytes([w_bytes[off + 80], w_bytes[off + 81]]).to_f32();
            let dmin = f16::from_le_bytes([w_bytes[off + 82], w_bytes[off + 83]]).to_f32();
            let xptr = x.as_ptr().add(b * QK_K);

            let mut q_cursor = 0usize;
            let mut x_off = 0usize;
            let mut is = 0usize;
            let mut slot: usize = 0;
            for _chunk in 0..2 {
                for j in 0..4 {
                    // Two 16-weight halves per (chunk, j) — sub-block A
                    // reads q[0..16] at this shift, sub-block B reads
                    // q[16..32] at this shift.
                    for half in 0..2 {
                        let q_byte_off = q_cursor + half * 16;
                        let sc = scales[is];
                        is += 1;
                        let dl = d * (sc & 0xF) as f32;
                        let ml = dmin * (sc >> 4) as f32;
                        let term = match j {
                            0 => lane_512::<0>(qs_ptr, xptr, q_byte_off, x_off, dl, ml, three),
                            1 => lane_512::<2>(qs_ptr, xptr, q_byte_off, x_off, dl, ml, three),
                            2 => lane_512::<4>(qs_ptr, xptr, q_byte_off, x_off, dl, ml, three),
                            _ => lane_512::<6>(qs_ptr, xptr, q_byte_off, x_off, dl, ml, three),
                        };
                        match slot & 3 {
                            0 => acc0 = _mm512_add_ps(acc0, term),
                            1 => acc1 = _mm512_add_ps(acc1, term),
                            2 => acc2 = _mm512_add_ps(acc2, term),
                            _ => acc3 = _mm512_add_ps(acc3, term),
                        }
                        slot += 1;
                        x_off += 16;
                    }
                }
                q_cursor += 32;
            }
        }
        let total = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
        out[i] = _mm512_reduce_add_ps(total);
    }
}

/// Q2_K matvec — AVX2 path. Each 16-weight sub-block is split into
/// two 8-weight halves; YMM width naturally matches the 8-element
/// stride. Same `fmsub(dl, q, ml)` recipe as the AVX-512 path.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn matvec_q2_k_w_f32_a_avx2(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 84;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;
    let three = _mm256_set1_epi32(3);

    // Process 8 weights at a given const SHIFT, sized for AVX2's
    // 8-wide YMM lanes. The 8 bytes loaded via `_mm_loadl_epi64` are
    // widened to `[i32; 8]` via `_mm256_cvtepu8_epi32`.
    unsafe fn lane_256<const SHIFT: i32>(
        qs_ptr: *const u8,
        xptr: *const f32,
        q_off: usize,
        x_off: usize,
        dl_v: __m256,
        ml_v: __m256,
        three: __m256i,
    ) -> __m256 {
        use std::arch::x86_64::*;
        let qs_raw = _mm_loadl_epi64(qs_ptr.add(q_off) as *const __m128i);
        let qs_w = _mm256_cvtepu8_epi32(qs_raw);
        let q_val = _mm256_and_si256(_mm256_srli_epi32(qs_w, SHIFT), three);
        let q_f = _mm256_cvtepi32_ps(q_val);
        let value_f = _mm256_fmsub_ps(dl_v, q_f, ml_v);
        let x_v = _mm256_loadu_ps(xptr.add(x_off));
        _mm256_mul_ps(value_f, x_v)
    }

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        // Eight accumulators for round-robin pipelining.
        let mut acc = [_mm256_setzero_ps(); 8];

        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let scales = &w_bytes[off..off + 16];
            let qs_ptr = w_bytes.as_ptr().add(off + 16);
            let d = f16::from_le_bytes([w_bytes[off + 80], w_bytes[off + 81]]).to_f32();
            let dmin = f16::from_le_bytes([w_bytes[off + 82], w_bytes[off + 83]]).to_f32();
            let xptr = x.as_ptr().add(b * QK_K);

            let mut q_cursor = 0usize;
            let mut x_off = 0usize;
            let mut is = 0usize;
            let mut slot: usize = 0;
            for _chunk in 0..2 {
                for j in 0..4 {
                    for half in 0..2 {
                        let q_byte_off = q_cursor + half * 16;
                        let sc = scales[is];
                        is += 1;
                        let dl_v = _mm256_set1_ps(d * (sc & 0xF) as f32);
                        let ml_v = _mm256_set1_ps(dmin * (sc >> 4) as f32);
                        // 16-weight sub-block → two 8-wide YMM lanes.
                        for byte_off_lo in [0usize, 8] {
                            let term = match j {
                                0 => lane_256::<0>(
                                    qs_ptr,
                                    xptr,
                                    q_byte_off + byte_off_lo,
                                    x_off,
                                    dl_v,
                                    ml_v,
                                    three,
                                ),
                                1 => lane_256::<2>(
                                    qs_ptr,
                                    xptr,
                                    q_byte_off + byte_off_lo,
                                    x_off,
                                    dl_v,
                                    ml_v,
                                    three,
                                ),
                                2 => lane_256::<4>(
                                    qs_ptr,
                                    xptr,
                                    q_byte_off + byte_off_lo,
                                    x_off,
                                    dl_v,
                                    ml_v,
                                    three,
                                ),
                                _ => lane_256::<6>(
                                    qs_ptr,
                                    xptr,
                                    q_byte_off + byte_off_lo,
                                    x_off,
                                    dl_v,
                                    ml_v,
                                    three,
                                ),
                            };
                            acc[slot & 7] = _mm256_add_ps(acc[slot & 7], term);
                            slot += 1;
                            x_off += 8;
                        }
                    }
                }
                q_cursor += 32;
            }
        }

        let s01 = _mm256_add_ps(acc[0], acc[1]);
        let s23 = _mm256_add_ps(acc[2], acc[3]);
        let s45 = _mm256_add_ps(acc[4], acc[5]);
        let s67 = _mm256_add_ps(acc[6], acc[7]);
        let total = _mm256_add_ps(_mm256_add_ps(s01, s23), _mm256_add_ps(s45, s67));
        let lo = _mm256_castps256_ps128(total);
        let hi = _mm256_extractf128_ps(total, 1);
        let sum128 = _mm_add_ps(lo, hi);
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0b01));
        out[i] = _mm_cvtss_f32(sum32);
    }
}

/// Decode Q2_K rows by id and write to `out` as F32. Thin wrapper over
/// [`rustllama_gguf::dequant::dequant_q2_k`] applied row-by-row.
pub fn embed_lookup_q2_k(table_bytes: &[u8], ids: &[i32], out: &mut [f32], d: usize) {
    const BLOCK_BYTES: usize = 84;
    const QK_K: usize = 256;
    assert_eq!(d % QK_K, 0, "Q2_K embed lookup requires d % 256 == 0");
    let blocks_per_row = d / QK_K;
    let row_bytes = blocks_per_row * BLOCK_BYTES;
    for (i, &id) in ids.iter().enumerate() {
        let row_start = (id as usize) * row_bytes;
        let dst = &mut out[i * d..(i + 1) * d];
        let row = &table_bytes[row_start..row_start + row_bytes];
        rustllama_gguf::dequant::dequant_q2_k(row, dst);
    }
}

/// Q4_K weight matvec with on-the-fly dequant.
///
/// Super-block layout (144 bytes per 256 weights):
///   { d: f16, dmin: f16, scales: [u8; 12], qs: [u8; 128] }
///
/// The 12-byte `scales` packs 8 (sc, min) 6-bit pairs (see
/// `unpack_q4k_scales` for the bit shuffle, copied from ggml's
/// `get_scale_min_k4`). Each pair scales 32 consecutive outputs.
///
/// Layout for outputs within a super-block: 4 groups of 64 outputs, each
/// group consumes 32 q-bytes split as 32 low nibbles (scaled by sc[2k]/
/// min mn[2k]) followed by 32 high nibbles (scaled by sc[2k+1]/min
/// mn[2k+1]).
///
/// Q3_K weight matvec with on-the-fly dequant. Dispatches to AVX-512 /
/// AVX2 / scalar by runtime feature detection.
///
/// Block layout (110 bytes per 256 weights):
///   { hmask: [u8; 32], qs: [u8; 64], scales: [u8; 12], d: f16 }
///
/// Sub-block decode is described in [`rustllama_gguf::dequant::dequant_q3_k`].
///
/// Requires `k % 256 == 0`.
pub fn matvec_q3_k_w_f32_a(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    const BLOCK_BYTES: usize = 110;
    const QK_K: usize = 256;
    debug_assert_eq!(k % QK_K, 0, "Q3_K matvec requires k % 256 == 0");
    let blocks_per_row = k / QK_K;
    debug_assert_eq!(w_bytes.len(), m * blocks_per_row * BLOCK_BYTES);
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(out.len(), m);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: runtime feature detection above.
            unsafe { matvec_q3_k_w_f32_a_avx512(w_bytes, x, out, m, k) };
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime feature detection above.
            unsafe { matvec_q3_k_w_f32_a_avx2(w_bytes, x, out, m, k) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64.
        unsafe { matvec_q3_k_w_f32_a_neon(w_bytes, x, out, m, k) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    matvec_q3_k_w_f32_a_scalar(w_bytes, x, out, m, k);
}

/// AArch64 NEON Q3_K matvec. Mirrors the scalar layout: the 6-bit signed
/// sub-scales are reconstructed by the same `aux` bit-shuffle, then each
/// 256-weight super-block is 16 half-sub-blocks of 16 weights (2 q-chunks
/// × 4 shifts × 2 halves A/B). The 2-bit low field is `(qs >> shift) & 3`;
/// the per-weight high bit comes from `hmask` (bit `chunk*4 + j`): when the
/// bit is NOT set the value drops by 4, so `value = low2 - (set ? 0 : 4)`
/// (range -4..3). Scale is `d_all*(scale - 32)`. NEON baseline; same
/// tolerance contract as the AVX2 path.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn matvec_q3_k_w_f32_a_neon(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use std::arch::aarch64::*;
    const BLOCK_BYTES: usize = 110;
    const QK_K: usize = 256;
    const KMASK1: u32 = 0x03030303;
    const KMASK2: u32 = 0x0f0f0f0f;
    let blocks_per_row = k / QK_K;
    let three = vdupq_n_u8(0x03);
    let four = vdupq_n_u8(4);
    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = [vdupq_n_f32(0.0); 4];
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let hmask = w_bytes.as_ptr().add(off);
            let qs = w_bytes.as_ptr().add(off + 32);
            let sc = &w_bytes[off + 96..off + 108];
            let d_all = f16::from_le_bytes([w_bytes[off + 108], w_bytes[off + 109]]).to_f32();

            // Reconstruct the 16 signed sub-scales (same bit-shuffle as scalar).
            let mut aux = [0u32; 4];
            aux[0] = u32::from_le_bytes([sc[0], sc[1], sc[2], sc[3]]);
            aux[1] = u32::from_le_bytes([sc[4], sc[5], sc[6], sc[7]]);
            aux[2] = u32::from_le_bytes([sc[8], sc[9], sc[10], sc[11]]);
            let tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
            aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
            aux[0] = (aux[0] & KMASK2) | ((tmp & KMASK1) << 4);
            aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
            let aux_bytes: [u8; 16] = bytemuck::cast(aux);
            let scales: [i8; 16] = aux_bytes.map(|x| x as i8);

            let xb = x.as_ptr().add(b * QK_K);
            let mut x_off = 0usize;
            let mut is = 0usize;
            for chunk in 0..2 {
                let qc = qs.add(chunk * 32);
                for j in 0..4 {
                    let neg = vdupq_n_s8(-((j * 2) as i8));
                    let mbit = vdupq_n_u8(1u8 << (chunk * 4 + j));
                    // Sub-block A: qs[0..16], hmask[0..16].
                    let dl_a = d_all * (scales[is] as f32 - 32.0);
                    let lo_a = vandq_u8(vshlq_u8(vld1q_u8(qc), neg), three);
                    // 4 where hmask bit NOT set, else 0.
                    let hs_a = vandq_u8(vmvnq_u8(vtstq_u8(vld1q_u8(hmask), mbit)), four);
                    let lo_af = u8x16_to_f32x4x4(lo_a);
                    let hs_af = u8x16_to_f32x4x4(hs_a);
                    for c in 0..4 {
                        let xv = vld1q_f32(xb.add(x_off + c * 4));
                        let val = vsubq_f32(lo_af[c], hs_af[c]);
                        acc[c] = vfmaq_f32(acc[c], vmulq_n_f32(val, dl_a), xv);
                    }
                    x_off += 16;
                    is += 1;
                    // Sub-block B: qs[16..32], hmask[16..32].
                    let dl_b = d_all * (scales[is] as f32 - 32.0);
                    let lo_b = vandq_u8(vshlq_u8(vld1q_u8(qc.add(16)), neg), three);
                    let hs_b = vandq_u8(vmvnq_u8(vtstq_u8(vld1q_u8(hmask.add(16)), mbit)), four);
                    let lo_bf = u8x16_to_f32x4x4(lo_b);
                    let hs_bf = u8x16_to_f32x4x4(hs_b);
                    for c in 0..4 {
                        let xv = vld1q_f32(xb.add(x_off + c * 4));
                        let val = vsubq_f32(lo_bf[c], hs_bf[c]);
                        acc[c] = vfmaq_f32(acc[c], vmulq_n_f32(val, dl_b), xv);
                    }
                    x_off += 16;
                    is += 1;
                }
            }
        }
        let s = vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3]));
        out[i] = vaddvq_f32(s);
    }
}

fn matvec_q3_k_w_f32_a_scalar(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    const BLOCK_BYTES: usize = 110;
    const QK_K: usize = 256;
    const KMASK1: u32 = 0x03030303;
    const KMASK2: u32 = 0x0f0f0f0f;
    let blocks_per_row = k / QK_K;

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let hmask = &w_bytes[off..off + 32];
            let qs = &w_bytes[off + 32..off + 32 + 64];
            let sc = &w_bytes[off + 32 + 64..off + 32 + 64 + 12];
            let d_all = f16::from_le_bytes([w_bytes[off + 108], w_bytes[off + 109]]).to_f32();

            let mut aux = [0u32; 4];
            aux[0] = u32::from_le_bytes([sc[0], sc[1], sc[2], sc[3]]);
            aux[1] = u32::from_le_bytes([sc[4], sc[5], sc[6], sc[7]]);
            aux[2] = u32::from_le_bytes([sc[8], sc[9], sc[10], sc[11]]);
            let tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
            aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
            aux[0] = (aux[0] & KMASK2) | ((tmp & KMASK1) << 4);
            aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
            let aux_bytes: [u8; 16] = bytemuck::cast(aux);
            let scales: [i8; 16] = aux_bytes.map(|x| x as i8);

            let x_block = &x[b * QK_K..(b + 1) * QK_K];

            let mut q_cursor = 0usize;
            let mut x_off = 0usize;
            let mut m_bit: u8 = 1;
            let mut is = 0usize;
            for _chunk in 0..2 {
                let mut shift: u32 = 0;
                for _j in 0..4 {
                    let dl = d_all * (scales[is] as f32 - 32.0);
                    is += 1;
                    for l in 0..16 {
                        let lo = ((qs[q_cursor + l] >> shift) & 3) as i32;
                        let hi_sub = if hmask[l] & m_bit != 0 { 0 } else { 4 };
                        acc += dl * ((lo - hi_sub) as f32) * x_block[x_off + l];
                    }
                    x_off += 16;
                    let dl = d_all * (scales[is] as f32 - 32.0);
                    is += 1;
                    for l in 0..16 {
                        let lo = ((qs[q_cursor + l + 16] >> shift) & 3) as i32;
                        let hi_sub = if hmask[l + 16] & m_bit != 0 { 0 } else { 4 };
                        acc += dl * ((lo - hi_sub) as f32) * x_block[x_off + l];
                    }
                    x_off += 16;
                    shift += 2;
                    m_bit = m_bit.wrapping_shl(1);
                }
                q_cursor += 32;
            }
        }
        out[i] = acc;
    }
}

/// Q3_K matvec — AVX-512 path. Processes one 16-weight half-sub-block
/// per inner iteration:
///   - Load 16 `qs` bytes, widen to `[i32; 16]`, shift+mask to get the
///     low 2 bits of each weight (`low2 ∈ 0..=3`).
///   - Load 16 `hmask` bytes, widen, AND with the current `m_bit`
///     broadcast, compare to zero → 16-lane mask register `hi_set`.
///   - Reconstruction: `value = low2 - 4 + (hi_set ? 4 : 0)` (range
///     -4..3); convert to f32 and FMA against the matching 16-element
///     x lane.
///
/// Four FMA accumulators round-robin per j-iteration so the FP pipeline
/// stays full despite the per-iteration scale broadcast.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f")]
unsafe fn matvec_q3_k_w_f32_a_avx512(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 110;
    const QK_K: usize = 256;
    const KMASK1: u32 = 0x03030303;
    const KMASK2: u32 = 0x0f0f0f0f;
    let blocks_per_row = k / QK_K;
    let three = _mm512_set1_epi32(3);
    let four = _mm512_set1_epi32(4);
    let zero = _mm512_setzero_si512();

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();
        let mut acc2 = _mm512_setzero_ps();
        let mut acc3 = _mm512_setzero_ps();

        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let hmask_ptr = w_bytes.as_ptr().add(off);
            let qs_ptr = w_bytes.as_ptr().add(off + 32);
            let sc = &w_bytes[off + 32 + 64..off + 32 + 64 + 12];
            let d_all = f16::from_le_bytes([w_bytes[off + 108], w_bytes[off + 109]]).to_f32();
            let xptr = x.as_ptr().add(b * QK_K);

            let mut aux = [0u32; 4];
            aux[0] = u32::from_le_bytes([sc[0], sc[1], sc[2], sc[3]]);
            aux[1] = u32::from_le_bytes([sc[4], sc[5], sc[6], sc[7]]);
            aux[2] = u32::from_le_bytes([sc[8], sc[9], sc[10], sc[11]]);
            let tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
            aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
            aux[0] = (aux[0] & KMASK2) | ((tmp & KMASK1) << 4);
            aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
            let aux_bytes: [u8; 16] = bytemuck::cast(aux);
            let scales: [i8; 16] = aux_bytes.map(|x| x as i8);

            // Process one 16-weight half-sub-block at a given const SHIFT.
            // Helper closure that takes the (q_off, h_off, scale, m_bit) and
            // emits one fma into the chosen accumulator. SHIFT must be a
            // const-generic so the intrinsic's compile-time shift requirement
            // is satisfied — that's why this is split out into a const-fn
            // dispatch by `j` below.
            unsafe fn lane_512<const SHIFT: u32>(
                qs_ptr: *const u8,
                hmask_ptr: *const u8,
                xptr: *const f32,
                q_off: usize,
                h_off: usize,
                x_off: usize,
                dl: f32,
                m_bit_v: __m512i,
                three: __m512i,
                four: __m512i,
                zero: __m512i,
            ) -> __m512 {
                let qs_raw = _mm_loadu_si128(qs_ptr.add(q_off) as *const __m128i);
                let qs_w = _mm512_cvtepu8_epi32(qs_raw);
                let low2 = _mm512_and_si512(_mm512_srli_epi32(qs_w, SHIFT), three);
                let hm_raw = _mm_loadu_si128(hmask_ptr.add(h_off) as *const __m128i);
                let hm_w = _mm512_cvtepu8_epi32(hm_raw);
                let masked = _mm512_and_si512(hm_w, m_bit_v);
                let hi_set = _mm512_cmpneq_epi32_mask(masked, zero);
                let correction = _mm512_mask_mov_epi32(zero, hi_set, four);
                let value = _mm512_sub_epi32(_mm512_add_epi32(low2, correction), four);
                let value_f = _mm512_cvtepi32_ps(value);
                let dl_v = _mm512_set1_ps(dl);
                let term = _mm512_mul_ps(value_f, dl_v);
                let x_v = _mm512_loadu_ps(xptr.add(x_off));
                _mm512_fmadd_ps(term, x_v, _mm512_setzero_ps())
            }

            let mut q_cursor = 0usize;
            let mut x_off = 0usize;
            let mut is = 0usize;
            for chunk in 0..2 {
                for j in 0..4 {
                    let m_bit_val: i32 = 1i32 << (chunk * 4 + j);
                    let m_bit_v = _mm512_set1_epi32(m_bit_val);

                    // Two 16-weight halves per (chunk, j).
                    for half in 0..2 {
                        let q_byte_off = q_cursor + half * 16;
                        let h_byte_off = half * 16;
                        let scale = scales[is] as f32;
                        is += 1;
                        let dl = d_all * (scale - 32.0);

                        let term = match j {
                            0 => lane_512::<0>(
                                qs_ptr, hmask_ptr, xptr, q_byte_off, h_byte_off,
                                x_off, dl, m_bit_v, three, four, zero,
                            ),
                            1 => lane_512::<2>(
                                qs_ptr, hmask_ptr, xptr, q_byte_off, h_byte_off,
                                x_off, dl, m_bit_v, three, four, zero,
                            ),
                            2 => lane_512::<4>(
                                qs_ptr, hmask_ptr, xptr, q_byte_off, h_byte_off,
                                x_off, dl, m_bit_v, three, four, zero,
                            ),
                            _ => lane_512::<6>(
                                qs_ptr, hmask_ptr, xptr, q_byte_off, h_byte_off,
                                x_off, dl, m_bit_v, three, four, zero,
                            ),
                        };
                        // Four-way round-robin keeps independent FMA chains
                        // so the CPU's two FMA units stay busy.
                        let slot = (j * 2 + half) & 3;
                        match slot {
                            0 => acc0 = _mm512_add_ps(acc0, term),
                            1 => acc1 = _mm512_add_ps(acc1, term),
                            2 => acc2 = _mm512_add_ps(acc2, term),
                            _ => acc3 = _mm512_add_ps(acc3, term),
                        }
                        x_off += 16;
                    }
                }
                q_cursor += 32;
            }
        }
        let total = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
        out[i] = _mm512_reduce_add_ps(total);
    }
}

/// Q3_K matvec — AVX2 path. Processes 8 weights per iteration (one
/// quarter of a 16-weight half-sub-block). Same reconstruction recipe
/// as the AVX-512 variant; the only twist is that the 8-wide cmpgt on
/// the hmask result gives an `__m256i` byte-mask (all-1s or 0) instead
/// of an AVX-512 k-mask, so we just AND with the broadcasted `4` to
/// build the correction vector.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn matvec_q3_k_w_f32_a_avx2(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 110;
    const QK_K: usize = 256;
    const KMASK1: u32 = 0x03030303;
    const KMASK2: u32 = 0x0f0f0f0f;
    let blocks_per_row = k / QK_K;
    let three = _mm256_set1_epi32(3);
    let four = _mm256_set1_epi32(4);
    let zero_i = _mm256_setzero_si256();

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = [_mm256_setzero_ps(); 8];

        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let hmask_ptr = w_bytes.as_ptr().add(off);
            let qs_ptr = w_bytes.as_ptr().add(off + 32);
            let sc = &w_bytes[off + 32 + 64..off + 32 + 64 + 12];
            let d_all = f16::from_le_bytes([w_bytes[off + 108], w_bytes[off + 109]]).to_f32();
            let xptr = x.as_ptr().add(b * QK_K);

            let mut aux = [0u32; 4];
            aux[0] = u32::from_le_bytes([sc[0], sc[1], sc[2], sc[3]]);
            aux[1] = u32::from_le_bytes([sc[4], sc[5], sc[6], sc[7]]);
            aux[2] = u32::from_le_bytes([sc[8], sc[9], sc[10], sc[11]]);
            let tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
            aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
            aux[0] = (aux[0] & KMASK2) | ((tmp & KMASK1) << 4);
            aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
            let aux_bytes: [u8; 16] = bytemuck::cast(aux);
            let scales: [i8; 16] = aux_bytes.map(|x| x as i8);

            // Same const-generic shift dispatch as the AVX-512 path,
            // sized for AVX2's 8-wide YMM lanes.
            unsafe fn lane_256<const SHIFT: i32>(
                qs_ptr: *const u8,
                hmask_ptr: *const u8,
                xptr: *const f32,
                q_off: usize,
                h_off: usize,
                x_off: usize,
                dl: f32,
                m_bit_v: __m256i,
                three: __m256i,
                four: __m256i,
                zero_i: __m256i,
            ) -> __m256 {
                let qs_8 = _mm_loadl_epi64(qs_ptr.add(q_off) as *const __m128i);
                let qs_w = _mm256_cvtepu8_epi32(qs_8);
                let low2 = _mm256_and_si256(_mm256_srli_epi32(qs_w, SHIFT), three);
                let hm_8 = _mm_loadl_epi64(hmask_ptr.add(h_off) as *const __m128i);
                let hm_w = _mm256_cvtepu8_epi32(hm_8);
                let masked = _mm256_and_si256(hm_w, m_bit_v);
                let hi_set = _mm256_cmpgt_epi32(masked, zero_i);
                let correction = _mm256_and_si256(hi_set, four);
                let value = _mm256_sub_epi32(_mm256_add_epi32(low2, correction), four);
                let value_f = _mm256_cvtepi32_ps(value);
                let dl_v = _mm256_set1_ps(dl);
                let term = _mm256_mul_ps(value_f, dl_v);
                let x_v = _mm256_loadu_ps(xptr.add(x_off));
                _mm256_fmadd_ps(term, x_v, _mm256_setzero_ps())
            }

            let mut q_cursor = 0usize;
            let mut x_off = 0usize;
            let mut is = 0usize;
            for chunk in 0..2 {
                for j in 0..4 {
                    let m_bit_val: i32 = 1i32 << (chunk * 4 + j);
                    let m_bit_v = _mm256_set1_epi32(m_bit_val);

                    for half in 0..2 {
                        let q_byte_off = q_cursor + half * 16;
                        let h_byte_off = half * 16;
                        let scale = scales[is] as f32;
                        is += 1;
                        let dl = d_all * (scale - 32.0);

                        // 16 weights = 2 × 8 SIMD chunks.
                        for sub in 0..2 {
                            let term = match j {
                                0 => lane_256::<0>(
                                    qs_ptr, hmask_ptr, xptr, q_byte_off + sub * 8,
                                    h_byte_off + sub * 8, x_off, dl, m_bit_v,
                                    three, four, zero_i,
                                ),
                                1 => lane_256::<2>(
                                    qs_ptr, hmask_ptr, xptr, q_byte_off + sub * 8,
                                    h_byte_off + sub * 8, x_off, dl, m_bit_v,
                                    three, four, zero_i,
                                ),
                                2 => lane_256::<4>(
                                    qs_ptr, hmask_ptr, xptr, q_byte_off + sub * 8,
                                    h_byte_off + sub * 8, x_off, dl, m_bit_v,
                                    three, four, zero_i,
                                ),
                                _ => lane_256::<6>(
                                    qs_ptr, hmask_ptr, xptr, q_byte_off + sub * 8,
                                    h_byte_off + sub * 8, x_off, dl, m_bit_v,
                                    three, four, zero_i,
                                ),
                            };
                            let slot = (j * 4 + half * 2 + sub) & 7;
                            acc[slot] = _mm256_add_ps(acc[slot], term);
                            x_off += 8;
                        }
                    }
                }
                q_cursor += 32;
            }
        }

        // Horizontal reduce 8 accumulators → scalar.
        let s01 = _mm256_add_ps(acc[0], acc[1]);
        let s23 = _mm256_add_ps(acc[2], acc[3]);
        let s45 = _mm256_add_ps(acc[4], acc[5]);
        let s67 = _mm256_add_ps(acc[6], acc[7]);
        let s0123 = _mm256_add_ps(s01, s23);
        let s4567 = _mm256_add_ps(s45, s67);
        let total = _mm256_add_ps(s0123, s4567);
        // Lane-sum reduce of the YMM total.
        let lo = _mm256_castps256_ps128(total);
        let hi = _mm256_extractf128_ps(total, 1);
        let sum128 = _mm_add_ps(lo, hi);
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0b01));
        out[i] = _mm_cvtss_f32(sum32);
    }
}

/// Requires `k % 256 == 0` (i.e., k aligned to one full super-block).
pub fn matvec_q4_k_w_f32_a(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    const BLOCK_BYTES: usize = 144;
    const QK_K: usize = 256;
    debug_assert_eq!(k % QK_K, 0, "Q4_K matvec requires k % 256 == 0");
    let blocks_per_row = k / QK_K;
    debug_assert_eq!(w_bytes.len(), m * blocks_per_row * BLOCK_BYTES);
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(out.len(), m);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: runtime feature detection.
            unsafe { matvec_q4_k_w_f32_a_avx512(w_bytes, x, out, m, k) };
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime feature detection above.
            unsafe { matvec_q4_k_w_f32_a_avx2(w_bytes, x, out, m, k) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64.
        unsafe { matvec_q4_k_w_f32_a_neon(w_bytes, x, out, m, k) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    matvec_q4_k_w_f32_a_scalar(w_bytes, x, out, m, k);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f")]
unsafe fn matvec_q4_k_w_f32_a_avx512(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 144;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;
    let mask_low_nibble = _mm512_set1_epi32(0x0F);

    // Per 32-byte q chunk: split into 2 halves (16 bytes each), then per
    // half extract low and high nibbles into ZMM i32x16 lanes. So one
    // group yields 4 ZMM lanes (2 low + 2 high), each carrying 16
    // unpacked q values matched against 16 x values. Compare to AVX2:
    // 8 YMM lanes of 8. Same data, half the lane count.
    #[inline(always)]
    unsafe fn unpack_16_q_bytes_avx512(
        qs_ptr: *const u8,
        mask: __m512i,
    ) -> (__m512i, __m512i) {
        let raw = _mm_loadu_si128(qs_ptr as *const __m128i);
        let widened = _mm512_cvtepu8_epi32(raw);
        let lo = _mm512_and_si512(widened, mask);
        let hi = _mm512_and_si512(_mm512_srli_epi32(widened, 4), mask);
        (lo, hi)
    }

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        // 4 independent accumulators (2 low + 2 high per group, summed
        // over the 4 groups). Keeps FMA pipeline full like the AVX2
        // version's 8-acc layout, just wider.
        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();
        let mut acc2 = _mm512_setzero_ps();
        let mut acc3 = _mm512_setzero_ps();

        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d_bits = u16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]);
            let dmin_bits = u16::from_le_bytes([w_bytes[off + 2], w_bytes[off + 3]]);
            let d_scalar = f16::from_bits(d_bits).to_f32();
            let dmin_scalar = f16::from_bits(dmin_bits).to_f32();
            let (sc, mn) = unpack_q4k_scales(&w_bytes[off + 4..off + 16]);
            let qs_ptr = w_bytes.as_ptr().add(off + 16);
            let xptr = x.as_ptr().add(b * QK_K);

            for group in 0..4 {
                let group_qs = qs_ptr.add(group * 32);
                let group_x = xptr.add(group * 64);

                // Two halves of 16 q-bytes each → 2 × (lo, hi) ZMM lanes.
                let (lo_a, hi_a) = unpack_16_q_bytes_avx512(group_qs, mask_low_nibble);
                let (lo_b, hi_b) = unpack_16_q_bytes_avx512(group_qs.add(16), mask_low_nibble);

                let d_lo = _mm512_set1_ps(d_scalar * sc[group * 2] as f32);
                let m_lo = _mm512_set1_ps(dmin_scalar * mn[group * 2] as f32);
                let d_hi = _mm512_set1_ps(d_scalar * sc[group * 2 + 1] as f32);
                let m_hi = _mm512_set1_ps(dmin_scalar * mn[group * 2 + 1] as f32);

                // d_lo * q - m_lo for each nibble lane.
                let lo_a_f = _mm512_fmsub_ps(_mm512_cvtepi32_ps(lo_a), d_lo, m_lo);
                let lo_b_f = _mm512_fmsub_ps(_mm512_cvtepi32_ps(lo_b), d_lo, m_lo);
                let hi_a_f = _mm512_fmsub_ps(_mm512_cvtepi32_ps(hi_a), d_hi, m_hi);
                let hi_b_f = _mm512_fmsub_ps(_mm512_cvtepi32_ps(hi_b), d_hi, m_hi);

                let x_lo_a = _mm512_loadu_ps(group_x);
                let x_lo_b = _mm512_loadu_ps(group_x.add(16));
                let x_hi_a = _mm512_loadu_ps(group_x.add(32));
                let x_hi_b = _mm512_loadu_ps(group_x.add(48));

                acc0 = _mm512_fmadd_ps(lo_a_f, x_lo_a, acc0);
                acc1 = _mm512_fmadd_ps(lo_b_f, x_lo_b, acc1);
                acc2 = _mm512_fmadd_ps(hi_a_f, x_hi_a, acc2);
                acc3 = _mm512_fmadd_ps(hi_b_f, x_hi_b, acc3);
            }
        }

        let total = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
        out[i] = _mm512_reduce_add_ps(total);
    }
}

/// Unpack the 12-byte K-quant scales array into 8 (sc, min) 6-bit pairs.
/// Mirrors ggml's `get_scale_min_k4`. Shared between Q4_K / Q5_K kernels.
#[inline(always)]
fn unpack_q4k_scales(scales: &[u8]) -> ([u8; 8], [u8; 8]) {
    debug_assert_eq!(scales.len(), 12);
    let mut sc = [0u8; 8];
    let mut mn = [0u8; 8];
    for j in 0..8 {
        if j < 4 {
            sc[j] = scales[j] & 0x3F;
            mn[j] = scales[j + 4] & 0x3F;
        } else {
            sc[j] = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
            mn[j] = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        }
    }
    (sc, mn)
}

fn matvec_q4_k_w_f32_a_scalar(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    const BLOCK_BYTES: usize = 144;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;
    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d = f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32();
            let dmin = f16::from_le_bytes([w_bytes[off + 2], w_bytes[off + 3]]).to_f32();
            let (sc, mn) = unpack_q4k_scales(&w_bytes[off + 4..off + 16]);
            let qs = &w_bytes[off + 16..off + 16 + 128];
            let x_block = &x[b * QK_K..(b + 1) * QK_K];

            // 4 groups of 64 outputs, 32 q-bytes each.
            for group in 0..4 {
                let q_chunk = &qs[group * 32..(group + 1) * 32];
                let d_lo = d * sc[group * 2] as f32;
                let m_lo = dmin * mn[group * 2] as f32;
                let d_hi = d * sc[group * 2 + 1] as f32;
                let m_hi = dmin * mn[group * 2 + 1] as f32;
                let x_lo = &x_block[group * 64..group * 64 + 32];
                let x_hi = &x_block[group * 64 + 32..(group + 1) * 64];
                for l in 0..32 {
                    let q = q_chunk[l];
                    acc += (d_lo * (q & 0x0F) as f32 - m_lo) * x_lo[l];
                    acc += (d_hi * (q >> 4) as f32 - m_hi) * x_hi[l];
                }
            }
        }
        out[i] = acc;
    }
}

/// Widen a `uint8x16` (16 bytes) to four `float32x4` (unsigned). Shared
/// across this module's NEON quant kernels and the `mlx_affine` submodule.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub(crate) unsafe fn u8x16_to_f32x4x4(
    v: std::arch::aarch64::uint8x16_t,
) -> [std::arch::aarch64::float32x4_t; 4] {
    use std::arch::aarch64::*;
    let lo16 = vmovl_u8(vget_low_u8(v));
    let hi16 = vmovl_u8(vget_high_u8(v));
    [
        vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo16))),
        vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo16))),
        vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi16))),
        vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi16))),
    ]
}

/// AArch64 NEON Q4_K matvec (weight Q4_K_M, activation f32). Mirrors the
/// scalar/AVX2 math: per 256-weight super-block, 4 groups of 64 outputs;
/// each 32-byte q-chunk yields 32 low nibbles (→ x_lo, scale d_lo/min
/// m_lo) and 32 high nibbles (→ x_hi, d_hi/m_hi). Coefficient
/// `(d*q - m)` computed in f32 and `fmla`'d against the activations.
/// NEON baseline; tolerance contract like the AVX2 path.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn matvec_q4_k_w_f32_a_neon(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use std::arch::aarch64::*;
    const BLOCK_BYTES: usize = 144;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;
    let mask_lo = vdupq_n_u8(0x0F);
    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = [vdupq_n_f32(0.0); 4];
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d = f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32();
            let dmin = f16::from_le_bytes([w_bytes[off + 2], w_bytes[off + 3]]).to_f32();
            let (sc, mn) = unpack_q4k_scales(&w_bytes[off + 4..off + 16]);
            let qs = w_bytes.as_ptr().add(off + 16);
            let xb = x.as_ptr().add(b * QK_K);
            for group in 0..4 {
                let qptr = qs.add(group * 32);
                let d_lo = d * sc[group * 2] as f32;
                let m_lo = vdupq_n_f32(dmin * mn[group * 2] as f32);
                let d_hi = d * sc[group * 2 + 1] as f32;
                let m_hi = vdupq_n_f32(dmin * mn[group * 2 + 1] as f32);
                let xlo = xb.add(group * 64);
                let xhi = xb.add(group * 64 + 32);
                // 32 q-bytes as 2×16; each 16 → 16 lo + 16 hi nibbles.
                let mut ai = 0usize;
                for half in 0..2 {
                    let base = half * 16;
                    let qb16 = vld1q_u8(qptr.add(base));
                    let qlo_f = u8x16_to_f32x4x4(vandq_u8(qb16, mask_lo));
                    let qhi_f = u8x16_to_f32x4x4(vshrq_n_u8::<4>(qb16));
                    for c in 0..4 {
                        let idx = base + c * 4;
                        let xl = vld1q_f32(xlo.add(idx));
                        let xh = vld1q_f32(xhi.add(idx));
                        let coef_lo = vsubq_f32(vmulq_n_f32(qlo_f[c], d_lo), m_lo);
                        let coef_hi = vsubq_f32(vmulq_n_f32(qhi_f[c], d_hi), m_hi);
                        acc[ai & 3] = vfmaq_f32(acc[ai & 3], coef_lo, xl);
                        acc[(ai + 1) & 3] = vfmaq_f32(acc[(ai + 1) & 3], coef_hi, xh);
                        ai += 2;
                    }
                }
            }
        }
        let s = vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3]));
        out[i] = vaddvq_f32(s);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn matvec_q4_k_w_f32_a_avx2(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 144;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;
    let mask_low_nibble = _mm256_set1_epi32(0x0F);

    // Unpack 32 q-bytes (one Q4_K sub-block group) into two __m256i lanes
    // of 8 nibbles each, packed as i32 for cvtepi32_ps.
    //
    // Returns (low_07, low_815, low_1623, low_2431, hi_07, hi_815, hi_1623, hi_2431):
    // 4 vectors of low nibbles (32 outputs) and 4 vectors of high nibbles.
    #[inline(always)]
    unsafe fn unpack_32_q_bytes(
        qs_ptr: *const u8,
        mask: __m256i,
    ) -> (
        __m256i, __m256i, __m256i, __m256i,
        __m256i, __m256i, __m256i, __m256i,
    ) {
        let low_07 =
            _mm256_and_si256(_mm256_cvtepu8_epi32(_mm_loadl_epi64(qs_ptr as *const __m128i)), mask);
        let low_815 = _mm256_and_si256(
            _mm256_cvtepu8_epi32(_mm_loadl_epi64(qs_ptr.add(8) as *const __m128i)),
            mask,
        );
        let low_1623 = _mm256_and_si256(
            _mm256_cvtepu8_epi32(_mm_loadl_epi64(qs_ptr.add(16) as *const __m128i)),
            mask,
        );
        let low_2431 = _mm256_and_si256(
            _mm256_cvtepu8_epi32(_mm_loadl_epi64(qs_ptr.add(24) as *const __m128i)),
            mask,
        );
        let hi_07 = _mm256_and_si256(
            _mm256_srli_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64(qs_ptr as *const __m128i)), 4),
            mask,
        );
        let hi_815 = _mm256_and_si256(
            _mm256_srli_epi32(
                _mm256_cvtepu8_epi32(_mm_loadl_epi64(qs_ptr.add(8) as *const __m128i)),
                4,
            ),
            mask,
        );
        let hi_1623 = _mm256_and_si256(
            _mm256_srli_epi32(
                _mm256_cvtepu8_epi32(_mm_loadl_epi64(qs_ptr.add(16) as *const __m128i)),
                4,
            ),
            mask,
        );
        let hi_2431 = _mm256_and_si256(
            _mm256_srli_epi32(
                _mm256_cvtepu8_epi32(_mm_loadl_epi64(qs_ptr.add(24) as *const __m128i)),
                4,
            ),
            mask,
        );
        (
            low_07, low_815, low_1623, low_2431,
            hi_07, hi_815, hi_1623, hi_2431,
        )
    }

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        // 8 independent accumulators (one per sub-block lane of 8).
        let mut acc = [_mm256_setzero_ps(); 8];

        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d_bits = u16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]);
            let dmin_bits = u16::from_le_bytes([w_bytes[off + 2], w_bytes[off + 3]]);
            let d_scalar = f16::from_bits(d_bits).to_f32();
            let dmin_scalar = f16::from_bits(dmin_bits).to_f32();
            let (sc, mn) = unpack_q4k_scales(&w_bytes[off + 4..off + 16]);
            let qs_ptr = w_bytes.as_ptr().add(off + 16);
            let xptr = x.as_ptr().add(b * QK_K);

            for group in 0..4 {
                let group_qs = qs_ptr.add(group * 32);
                let group_x = xptr.add(group * 64);

                let (l07, l815, l1623, l2431, h07, h815, h1623, h2431) =
                    unpack_32_q_bytes(group_qs, mask_low_nibble);

                // Per-group scales for low / high halves.
                let d_lo = _mm256_set1_ps(d_scalar * sc[group * 2] as f32);
                let m_lo = _mm256_set1_ps(dmin_scalar * mn[group * 2] as f32);
                let d_hi = _mm256_set1_ps(d_scalar * sc[group * 2 + 1] as f32);
                let m_hi = _mm256_set1_ps(dmin_scalar * mn[group * 2 + 1] as f32);

                // Reconstruct (d_lo * q - m_lo) for each low-nibble lane.
                let lo_07 = _mm256_fmsub_ps(_mm256_cvtepi32_ps(l07), d_lo, m_lo);
                let lo_815 = _mm256_fmsub_ps(_mm256_cvtepi32_ps(l815), d_lo, m_lo);
                let lo_1623 = _mm256_fmsub_ps(_mm256_cvtepi32_ps(l1623), d_lo, m_lo);
                let lo_2431 = _mm256_fmsub_ps(_mm256_cvtepi32_ps(l2431), d_lo, m_lo);
                let hi_07 = _mm256_fmsub_ps(_mm256_cvtepi32_ps(h07), d_hi, m_hi);
                let hi_815 = _mm256_fmsub_ps(_mm256_cvtepi32_ps(h815), d_hi, m_hi);
                let hi_1623 = _mm256_fmsub_ps(_mm256_cvtepi32_ps(h1623), d_hi, m_hi);
                let hi_2431 = _mm256_fmsub_ps(_mm256_cvtepi32_ps(h2431), d_hi, m_hi);

                // Load 64 x values: 32 for low-nibble outputs, 32 for high.
                let x_lo_07 = _mm256_loadu_ps(group_x);
                let x_lo_815 = _mm256_loadu_ps(group_x.add(8));
                let x_lo_1623 = _mm256_loadu_ps(group_x.add(16));
                let x_lo_2431 = _mm256_loadu_ps(group_x.add(24));
                let x_hi_07 = _mm256_loadu_ps(group_x.add(32));
                let x_hi_815 = _mm256_loadu_ps(group_x.add(40));
                let x_hi_1623 = _mm256_loadu_ps(group_x.add(48));
                let x_hi_2431 = _mm256_loadu_ps(group_x.add(56));

                acc[0] = _mm256_fmadd_ps(lo_07, x_lo_07, acc[0]);
                acc[1] = _mm256_fmadd_ps(lo_815, x_lo_815, acc[1]);
                acc[2] = _mm256_fmadd_ps(lo_1623, x_lo_1623, acc[2]);
                acc[3] = _mm256_fmadd_ps(lo_2431, x_lo_2431, acc[3]);
                acc[4] = _mm256_fmadd_ps(hi_07, x_hi_07, acc[4]);
                acc[5] = _mm256_fmadd_ps(hi_815, x_hi_815, acc[5]);
                acc[6] = _mm256_fmadd_ps(hi_1623, x_hi_1623, acc[6]);
                acc[7] = _mm256_fmadd_ps(hi_2431, x_hi_2431, acc[7]);
            }
        }

        // Reduce 8 → 1 then horizontal sum.
        let s01 = _mm256_add_ps(acc[0], acc[1]);
        let s23 = _mm256_add_ps(acc[2], acc[3]);
        let s45 = _mm256_add_ps(acc[4], acc[5]);
        let s67 = _mm256_add_ps(acc[6], acc[7]);
        let s0123 = _mm256_add_ps(s01, s23);
        let s4567 = _mm256_add_ps(s45, s67);
        let total = _mm256_add_ps(s0123, s4567);
        let mut sum128 =
            _mm_add_ps(_mm256_castps256_ps128(total), _mm256_extractf128_ps(total, 1));
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        out[i] = _mm_cvtss_f32(sum128);
    }
}

/// Embed-lookup dispatcher. Handles every `*Raw` quant variant so that
/// size-aware load-time dispatch (which prefers raw storage for tensors
/// above the L3 threshold) doesn't force the embedding table into F16
/// just to keep this kernel happy.
pub fn embed_lookup_tensor(table: &Tensor, ids: &[i32], out: &mut [f32], d: usize) {
    match table.dtype {
        Dtype::F16 => embed_lookup_f16_to_f32(as_slice_f16(table), ids, out, d),
        Dtype::F32 => embed_lookup_f32(as_slice_f32(table), ids, out, d),
        Dtype::Bf16Raw => embed_lookup_bf16(as_bytes(table), ids, out, d),
        Dtype::Q8_0Raw => embed_lookup_q8_0(as_bytes(table), ids, out, d),
        Dtype::Q4_0Raw => embed_lookup_q4_0(as_bytes(table), ids, out, d),
        Dtype::Q5_0Raw => embed_lookup_q5_0(as_bytes(table), ids, out, d),
        Dtype::Q4_1Raw => embed_lookup_q4_1(as_bytes(table), ids, out, d),
        Dtype::Q5_1Raw => embed_lookup_q5_1(as_bytes(table), ids, out, d),
        Dtype::Q2_KRaw => embed_lookup_q2_k(as_bytes(table), ids, out, d),
        Dtype::Q8_KRaw => embed_lookup_q8_k(as_bytes(table), ids, out, d),
        Dtype::Q3_KRaw => embed_lookup_q3_k(as_bytes(table), ids, out, d),
        Dtype::Q4_KRaw => embed_lookup_q4_k(as_bytes(table), ids, out, d),
        Dtype::Q5_KRaw => embed_lookup_q5_k(as_bytes(table), ids, out, d),
        Dtype::Q6_KRaw => embed_lookup_q6_k(as_bytes(table), ids, out, d),
        Dtype::IQ4_XSRaw => embed_lookup_iq4_xs(as_bytes(table), ids, out, d),
        Dtype::IQ4_NLRaw => embed_lookup_iq4_nl(as_bytes(table), ids, out, d),
        Dtype::IQ3_SRaw => embed_lookup_iq3_s(as_bytes(table), ids, out, d),
        Dtype::IQ3_XXSRaw => embed_lookup_iq3_xxs(as_bytes(table), ids, out, d),
        Dtype::IQ2_XXSRaw => embed_lookup_iq2_xxs(as_bytes(table), ids, out, d),
        Dtype::IQ2_XSRaw => embed_lookup_iq2_xs(as_bytes(table), ids, out, d),
        Dtype::IQ2_SRaw => embed_lookup_iq2_s(as_bytes(table), ids, out, d),
        Dtype::IQ1_SRaw => embed_lookup_iq1_s(as_bytes(table), ids, out, d),
        Dtype::IQ1_MRaw => embed_lookup_iq1_m(as_bytes(table), ids, out, d),
        Dtype::PQ2_0Raw => embed_lookup_pq2_0(as_bytes(table), ids, out, d),
        Dtype::PTQ1_0Raw => embed_lookup_ptq1_0(as_bytes(table), ids, out, d),
        // MLX-quantized token-embedding table: dequant only the requested
        // rows straight out of the packed blob (table stays packed).
        Dtype::MlxAffineRaw => {
            mlx_affine::embed_lookup_mlx_affine_blob(as_bytes(table), ids, out, d)
        }
        other => panic!("embed_lookup_tensor: unsupported dtype {other:?}"),
    }
}

/// Decode IQ3_S rows by id and write to `out` as F32. Scalar
/// implementation; SIMD is a follow-up because the codebook-indirect
/// load doesn't vectorize cleanly.
pub fn embed_lookup_iq3_s(table_bytes: &[u8], ids: &[i32], out: &mut [f32], d: usize) {
    const BLOCK_BYTES: usize = 110;
    const QK_K: usize = 256;
    assert_eq!(d % QK_K, 0, "IQ3_S embed lookup requires d % 256 == 0");
    let blocks_per_row = d / QK_K;
    let row_bytes = blocks_per_row * BLOCK_BYTES;
    for (i, &id) in ids.iter().enumerate() {
        let row_start = (id as usize) * row_bytes;
        // Slice the row out of the table and dequant it into the row's
        // destination. Equivalent to the loop in `dequant_iq3_s` but
        // walking only the rows referenced by `ids`.
        let dst = &mut out[i * d..(i + 1) * d];
        let row = &table_bytes[row_start..row_start + row_bytes];
        rustllama_gguf::dequant::dequant_iq3_s(row, dst);
    }
}

/// Decode IQ3_XXS rows by id and write to `out` as F32. Same
/// row-slice-and-dequant pattern as [`embed_lookup_iq3_s`].
/// 98-byte super-blocks.
pub fn embed_lookup_iq3_xxs(table_bytes: &[u8], ids: &[i32], out: &mut [f32], d: usize) {
    const BLOCK_BYTES: usize = 98;
    const QK_K: usize = 256;
    assert_eq!(d % QK_K, 0, "IQ3_XXS embed lookup requires d % 256 == 0");
    let blocks_per_row = d / QK_K;
    let row_bytes = blocks_per_row * BLOCK_BYTES;
    for (i, &id) in ids.iter().enumerate() {
        let row_start = (id as usize) * row_bytes;
        let dst = &mut out[i * d..(i + 1) * d];
        let row = &table_bytes[row_start..row_start + row_bytes];
        rustllama_gguf::dequant::dequant_iq3_xxs(row, dst);
    }
}

/// Decode IQ2_XXS rows by id and write to `out` as F32. Same pattern
/// as [`embed_lookup_iq3_s`]: scalar dequant of just the referenced
/// rows. 66-byte super-blocks.
pub fn embed_lookup_iq2_xxs(table_bytes: &[u8], ids: &[i32], out: &mut [f32], d: usize) {
    const BLOCK_BYTES: usize = 66;
    const QK_K: usize = 256;
    assert_eq!(d % QK_K, 0, "IQ2_XXS embed lookup requires d % 256 == 0");
    let blocks_per_row = d / QK_K;
    let row_bytes = blocks_per_row * BLOCK_BYTES;
    for (i, &id) in ids.iter().enumerate() {
        let row_start = (id as usize) * row_bytes;
        let dst = &mut out[i * d..(i + 1) * d];
        let row = &table_bytes[row_start..row_start + row_bytes];
        rustllama_gguf::dequant::dequant_iq2_xxs(row, dst);
    }
}

/// Decode IQ2_XS rows by id and write to `out` as F32. 74-byte
/// super-blocks; otherwise the same row-slice-and-dequant pattern as
/// [`embed_lookup_iq2_xxs`].
pub fn embed_lookup_iq2_xs(table_bytes: &[u8], ids: &[i32], out: &mut [f32], d: usize) {
    const BLOCK_BYTES: usize = 74;
    const QK_K: usize = 256;
    assert_eq!(d % QK_K, 0, "IQ2_XS embed lookup requires d % 256 == 0");
    let blocks_per_row = d / QK_K;
    let row_bytes = blocks_per_row * BLOCK_BYTES;
    for (i, &id) in ids.iter().enumerate() {
        let row_start = (id as usize) * row_bytes;
        let dst = &mut out[i * d..(i + 1) * d];
        let row = &table_bytes[row_start..row_start + row_bytes];
        rustllama_gguf::dequant::dequant_iq2_xs(row, dst);
    }
}

/// Decode IQ2_S rows by id and write to `out` as F32. 82-byte
/// super-blocks; otherwise the same row-slice-and-dequant pattern as
/// [`embed_lookup_iq2_xxs`].
pub fn embed_lookup_iq2_s(table_bytes: &[u8], ids: &[i32], out: &mut [f32], d: usize) {
    const BLOCK_BYTES: usize = 82;
    const QK_K: usize = 256;
    assert_eq!(d % QK_K, 0, "IQ2_S embed lookup requires d % 256 == 0");
    let blocks_per_row = d / QK_K;
    let row_bytes = blocks_per_row * BLOCK_BYTES;
    for (i, &id) in ids.iter().enumerate() {
        let row_start = (id as usize) * row_bytes;
        let dst = &mut out[i * d..(i + 1) * d];
        let row = &table_bytes[row_start..row_start + row_bytes];
        rustllama_gguf::dequant::dequant_iq2_s(row, dst);
    }
}

/// Decode a single IQ4_NL row by id and write it into `out` as F32.
/// Same codebook lookup as [`embed_lookup_iq4_xs`] but with the
/// 32-element block layout (single scale per block, no sub-blocks).
pub fn embed_lookup_iq4_nl(table_bytes: &[u8], ids: &[i32], out: &mut [f32], d: usize) {
    const QK: usize = 32;
    assert_eq!(d % QK, 0, "IQ4_NL embed lookup requires d % 32 == 0");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: runtime feature detection.
            unsafe { embed_lookup_iq4_nl_avx512(table_bytes, ids, out, d) };
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("ssse3") {
            // SAFETY: runtime feature detection.
            unsafe { embed_lookup_iq4_nl_avx2(table_bytes, ids, out, d) };
            return;
        }
    }
    embed_lookup_iq4_nl_scalar(table_bytes, ids, out, d);
}

fn embed_lookup_iq4_nl_scalar(table_bytes: &[u8], ids: &[i32], out: &mut [f32], d: usize) {
    const BLOCK_BYTES: usize = 18;
    const QK: usize = 32;
    let blocks_per_row = d / QK;
    let row_bytes = blocks_per_row * BLOCK_BYTES;
    for (i, &id) in ids.iter().enumerate() {
        let row_start = (id as usize) * row_bytes;
        let dst = &mut out[i * d..(i + 1) * d];
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d_scale =
                f16::from_le_bytes([table_bytes[off], table_bytes[off + 1]]).to_f32();
            let qs = &table_bytes[off + 2..off + 18];
            let dst_chunk = &mut dst[b * QK..(b + 1) * QK];
            for j in 0..16 {
                let q = qs[j];
                let lo = (q & 0x0F) as usize;
                let hi = (q >> 4) as usize;
                dst_chunk[j] = d_scale * (KVALUES_IQ4XS[lo] as f32);
                dst_chunk[j + 16] = d_scale * (KVALUES_IQ4XS[hi] as f32);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,ssse3")]
unsafe fn embed_lookup_iq4_nl_avx2(table_bytes: &[u8], ids: &[i32], out: &mut [f32], d: usize) {
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 18;
    const QK: usize = 32;
    let blocks_per_row = d / QK;
    let row_bytes = blocks_per_row * BLOCK_BYTES;

    let codebook = _mm_setr_epi8(
        KVALUES_IQ4XS[0], KVALUES_IQ4XS[1], KVALUES_IQ4XS[2], KVALUES_IQ4XS[3],
        KVALUES_IQ4XS[4], KVALUES_IQ4XS[5], KVALUES_IQ4XS[6], KVALUES_IQ4XS[7],
        KVALUES_IQ4XS[8], KVALUES_IQ4XS[9], KVALUES_IQ4XS[10], KVALUES_IQ4XS[11],
        KVALUES_IQ4XS[12], KVALUES_IQ4XS[13], KVALUES_IQ4XS[14], KVALUES_IQ4XS[15],
    );
    let mask_low = _mm_set1_epi8(0x0F);

    for (i, &id) in ids.iter().enumerate() {
        let row_start = (id as usize) * row_bytes;
        let dst = &mut out[i * d..(i + 1) * d];
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d_bits = u16::from_le_bytes([table_bytes[off], table_bytes[off + 1]]);
            let scale = _mm256_set1_ps(f16::from_bits(d_bits).to_f32());

            let q_ptr = table_bytes.as_ptr().add(off + 2);
            let q = _mm_loadu_si128(q_ptr as *const __m128i);
            let lo_idx = _mm_and_si128(q, mask_low);
            let hi_idx = _mm_and_si128(_mm_srli_epi16(q, 4), mask_low);
            let lo_vals_i8 = _mm_shuffle_epi8(codebook, lo_idx);
            let hi_vals_i8 = _mm_shuffle_epi8(codebook, hi_idx);

            let lo_upper = _mm_srli_si128(lo_vals_i8, 8);
            let hi_upper = _mm_srli_si128(hi_vals_i8, 8);
            let dst_ptr = dst.as_mut_ptr().add(b * QK);
            _mm256_storeu_ps(
                dst_ptr,
                _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo_vals_i8)), scale),
            );
            _mm256_storeu_ps(
                dst_ptr.add(8),
                _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo_upper)), scale),
            );
            _mm256_storeu_ps(
                dst_ptr.add(16),
                _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi_vals_i8)), scale),
            );
            _mm256_storeu_ps(
                dst_ptr.add(24),
                _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi_upper)), scale),
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f")]
unsafe fn embed_lookup_iq4_nl_avx512(
    table_bytes: &[u8],
    ids: &[i32],
    out: &mut [f32],
    d: usize,
) {
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 18;
    const QK: usize = 32;
    let blocks_per_row = d / QK;
    let row_bytes = blocks_per_row * BLOCK_BYTES;

    let codebook = _mm512_setr_epi32(
        KVALUES_IQ4XS[0] as i32, KVALUES_IQ4XS[1] as i32,
        KVALUES_IQ4XS[2] as i32, KVALUES_IQ4XS[3] as i32,
        KVALUES_IQ4XS[4] as i32, KVALUES_IQ4XS[5] as i32,
        KVALUES_IQ4XS[6] as i32, KVALUES_IQ4XS[7] as i32,
        KVALUES_IQ4XS[8] as i32, KVALUES_IQ4XS[9] as i32,
        KVALUES_IQ4XS[10] as i32, KVALUES_IQ4XS[11] as i32,
        KVALUES_IQ4XS[12] as i32, KVALUES_IQ4XS[13] as i32,
        KVALUES_IQ4XS[14] as i32, KVALUES_IQ4XS[15] as i32,
    );
    let nibble_mask = _mm512_set1_epi32(0x0F);

    for (i, &id) in ids.iter().enumerate() {
        let row_start = (id as usize) * row_bytes;
        let dst = &mut out[i * d..(i + 1) * d];
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d_bits = u16::from_le_bytes([table_bytes[off], table_bytes[off + 1]]);
            let scale = _mm512_set1_ps(f16::from_bits(d_bits).to_f32());

            let q_raw = _mm_loadu_si128(table_bytes.as_ptr().add(off + 2) as *const __m128i);
            let q_wide = _mm512_cvtepu8_epi32(q_raw);
            let lo_idx = _mm512_and_si512(q_wide, nibble_mask);
            let hi_idx = _mm512_and_si512(_mm512_srli_epi32(q_wide, 4), nibble_mask);

            let lo_vals = _mm512_mul_ps(
                _mm512_cvtepi32_ps(_mm512_permutexvar_epi32(lo_idx, codebook)),
                scale,
            );
            let hi_vals = _mm512_mul_ps(
                _mm512_cvtepi32_ps(_mm512_permutexvar_epi32(hi_idx, codebook)),
                scale,
            );

            let dst_ptr = dst.as_mut_ptr().add(b * QK);
            _mm512_storeu_ps(dst_ptr, lo_vals);
            _mm512_storeu_ps(dst_ptr.add(16), hi_vals);
        }
    }
}

/// Decode a single IQ4_XS row by id and write it into `out` as F32.
/// Mirrors [`rustllama_gguf::dequant::dequant_iq4_xs`], walking only
/// the rows referenced by `ids`.
pub fn embed_lookup_iq4_xs(table_bytes: &[u8], ids: &[i32], out: &mut [f32], d: usize) {
    const QK_K: usize = 256;
    assert_eq!(d % QK_K, 0, "IQ4_XS embed lookup requires d % 256 == 0");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: runtime feature detection.
            unsafe { embed_lookup_iq4_xs_avx512(table_bytes, ids, out, d) };
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("ssse3") {
            // SAFETY: runtime feature detection.
            unsafe { embed_lookup_iq4_xs_avx2(table_bytes, ids, out, d) };
            return;
        }
    }
    embed_lookup_iq4_xs_scalar(table_bytes, ids, out, d);
}

fn embed_lookup_iq4_xs_scalar(table_bytes: &[u8], ids: &[i32], out: &mut [f32], d: usize) {
    const BLOCK_BYTES: usize = 136;
    const QK_K: usize = 256;
    let blocks_per_row = d / QK_K;
    let row_bytes = blocks_per_row * BLOCK_BYTES;
    for (i, &id) in ids.iter().enumerate() {
        let row_start = (id as usize) * row_bytes;
        let dst = &mut out[i * d..(i + 1) * d];
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d_scale =
                f16::from_le_bytes([table_bytes[off], table_bytes[off + 1]]).to_f32();
            let scales_h =
                u16::from_le_bytes([table_bytes[off + 2], table_bytes[off + 3]]);
            let scales_l = &table_bytes[off + 4..off + 8];
            let qs = &table_bytes[off + 8..off + 8 + 128];
            let dst_block = &mut dst[b * QK_K..(b + 1) * QK_K];
            for ib in 0..8 {
                let lo_nibble = if ib % 2 == 0 {
                    scales_l[ib / 2] & 0x0F
                } else {
                    scales_l[ib / 2] >> 4
                };
                let hi_bits = ((scales_h >> (2 * ib)) & 0x03) as u8;
                let ls = (lo_nibble | (hi_bits << 4)) as i8 - 32;
                let sub_d = d_scale * (ls as f32);
                let q_off = ib * 16;
                let dst_off = ib * 32;
                for j in 0..16 {
                    let q = qs[q_off + j];
                    let lo = (q & 0x0F) as usize;
                    let hi = (q >> 4) as usize;
                    dst_block[dst_off + j] = sub_d * (KVALUES_IQ4XS[lo] as f32);
                    dst_block[dst_off + 16 + j] = sub_d * (KVALUES_IQ4XS[hi] as f32);
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,ssse3")]
unsafe fn embed_lookup_iq4_xs_avx2(table_bytes: &[u8], ids: &[i32], out: &mut [f32], d: usize) {
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 136;
    const QK_K: usize = 256;
    let blocks_per_row = d / QK_K;
    let row_bytes = blocks_per_row * BLOCK_BYTES;

    let codebook = _mm_setr_epi8(
        KVALUES_IQ4XS[0], KVALUES_IQ4XS[1], KVALUES_IQ4XS[2], KVALUES_IQ4XS[3],
        KVALUES_IQ4XS[4], KVALUES_IQ4XS[5], KVALUES_IQ4XS[6], KVALUES_IQ4XS[7],
        KVALUES_IQ4XS[8], KVALUES_IQ4XS[9], KVALUES_IQ4XS[10], KVALUES_IQ4XS[11],
        KVALUES_IQ4XS[12], KVALUES_IQ4XS[13], KVALUES_IQ4XS[14], KVALUES_IQ4XS[15],
    );
    let mask_low = _mm_set1_epi8(0x0F);

    for (i, &id) in ids.iter().enumerate() {
        let row_start = (id as usize) * row_bytes;
        let dst = &mut out[i * d..(i + 1) * d];
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d_bits = u16::from_le_bytes([table_bytes[off], table_bytes[off + 1]]);
            let d_scale = f16::from_bits(d_bits).to_f32();
            let scales_h = u16::from_le_bytes([table_bytes[off + 2], table_bytes[off + 3]]);
            let scales_l = [
                table_bytes[off + 4], table_bytes[off + 5],
                table_bytes[off + 6], table_bytes[off + 7],
            ];
            let dst_block = &mut dst[b * QK_K..(b + 1) * QK_K];
            for ib in 0..8 {
                let lo_nibble = if ib % 2 == 0 {
                    scales_l[ib / 2] & 0x0F
                } else {
                    scales_l[ib / 2] >> 4
                };
                let hi_bits = ((scales_h >> (2 * ib)) & 0x03) as u8;
                let ls = (lo_nibble | (hi_bits << 4)) as i8 - 32;
                let sub_d = _mm256_set1_ps(d_scale * (ls as f32));

                let q_ptr = table_bytes.as_ptr().add(off + 8 + ib * 16);
                let q = _mm_loadu_si128(q_ptr as *const __m128i);
                let lo_idx = _mm_and_si128(q, mask_low);
                let hi_idx = _mm_and_si128(_mm_srli_epi16(q, 4), mask_low);
                let lo_vals_i8 = _mm_shuffle_epi8(codebook, lo_idx);
                let hi_vals_i8 = _mm_shuffle_epi8(codebook, hi_idx);

                let lo_upper = _mm_srli_si128(lo_vals_i8, 8);
                let hi_upper = _mm_srli_si128(hi_vals_i8, 8);
                let lo_first = _mm256_mul_ps(
                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo_vals_i8)),
                    sub_d,
                );
                let lo_second = _mm256_mul_ps(
                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo_upper)),
                    sub_d,
                );
                let hi_first = _mm256_mul_ps(
                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi_vals_i8)),
                    sub_d,
                );
                let hi_second = _mm256_mul_ps(
                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi_upper)),
                    sub_d,
                );

                let dst_ptr = dst_block.as_mut_ptr().add(ib * 32);
                _mm256_storeu_ps(dst_ptr, lo_first);
                _mm256_storeu_ps(dst_ptr.add(8), lo_second);
                _mm256_storeu_ps(dst_ptr.add(16), hi_first);
                _mm256_storeu_ps(dst_ptr.add(24), hi_second);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f")]
unsafe fn embed_lookup_iq4_xs_avx512(
    table_bytes: &[u8],
    ids: &[i32],
    out: &mut [f32],
    d: usize,
) {
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 136;
    const QK_K: usize = 256;
    let blocks_per_row = d / QK_K;
    let row_bytes = blocks_per_row * BLOCK_BYTES;

    let codebook = _mm512_setr_epi32(
        KVALUES_IQ4XS[0] as i32, KVALUES_IQ4XS[1] as i32,
        KVALUES_IQ4XS[2] as i32, KVALUES_IQ4XS[3] as i32,
        KVALUES_IQ4XS[4] as i32, KVALUES_IQ4XS[5] as i32,
        KVALUES_IQ4XS[6] as i32, KVALUES_IQ4XS[7] as i32,
        KVALUES_IQ4XS[8] as i32, KVALUES_IQ4XS[9] as i32,
        KVALUES_IQ4XS[10] as i32, KVALUES_IQ4XS[11] as i32,
        KVALUES_IQ4XS[12] as i32, KVALUES_IQ4XS[13] as i32,
        KVALUES_IQ4XS[14] as i32, KVALUES_IQ4XS[15] as i32,
    );
    let nibble_mask = _mm512_set1_epi32(0x0F);

    for (i, &id) in ids.iter().enumerate() {
        let row_start = (id as usize) * row_bytes;
        let dst = &mut out[i * d..(i + 1) * d];
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d_bits = u16::from_le_bytes([table_bytes[off], table_bytes[off + 1]]);
            let d_scale = f16::from_bits(d_bits).to_f32();
            let scales_h = u16::from_le_bytes([table_bytes[off + 2], table_bytes[off + 3]]);
            let scales_l = [
                table_bytes[off + 4], table_bytes[off + 5],
                table_bytes[off + 6], table_bytes[off + 7],
            ];
            let dst_block = &mut dst[b * QK_K..(b + 1) * QK_K];
            for ib in 0..8 {
                let lo_nibble = if ib % 2 == 0 {
                    scales_l[ib / 2] & 0x0F
                } else {
                    scales_l[ib / 2] >> 4
                };
                let hi_bits = ((scales_h >> (2 * ib)) & 0x03) as u8;
                let ls = (lo_nibble | (hi_bits << 4)) as i8 - 32;
                let sub_d = _mm512_set1_ps(d_scale * (ls as f32));

                let q_ptr = table_bytes.as_ptr().add(off + 8 + ib * 16);
                let q_raw = _mm_loadu_si128(q_ptr as *const __m128i);
                let q_wide = _mm512_cvtepu8_epi32(q_raw);
                let lo_idx = _mm512_and_si512(q_wide, nibble_mask);
                let hi_idx = _mm512_and_si512(_mm512_srli_epi32(q_wide, 4), nibble_mask);

                let lo_vals = _mm512_mul_ps(
                    _mm512_cvtepi32_ps(_mm512_permutexvar_epi32(lo_idx, codebook)),
                    sub_d,
                );
                let hi_vals = _mm512_mul_ps(
                    _mm512_cvtepi32_ps(_mm512_permutexvar_epi32(hi_idx, codebook)),
                    sub_d,
                );

                let dst_ptr = dst_block.as_mut_ptr().add(ib * 32);
                _mm512_storeu_ps(dst_ptr, lo_vals);
                _mm512_storeu_ps(dst_ptr.add(16), hi_vals);
            }
        }
    }
}

/// Decode Q3_K rows by id and write to `out` as F32. Delegates to
/// [`rustllama_gguf::dequant::dequant_q3_k`] per row; the layout
/// unpacking is gnarly enough that a hand-rolled scalar duplicate
/// would just bloat the kernel module.
pub fn embed_lookup_q3_k(table_bytes: &[u8], ids: &[i32], out: &mut [f32], d: usize) {
    const BLOCK_BYTES: usize = 110;
    const QK_K: usize = 256;
    assert_eq!(d % QK_K, 0, "Q3_K embed lookup requires d % 256 == 0");
    let blocks_per_row = d / QK_K;
    let row_bytes = blocks_per_row * BLOCK_BYTES;
    for (i, &id) in ids.iter().enumerate() {
        let row_start = (id as usize) * row_bytes;
        let dst = &mut out[i * d..(i + 1) * d];
        let row = &table_bytes[row_start..row_start + row_bytes];
        rustllama_gguf::dequant::dequant_q3_k(row, dst);
    }
}

/// Decode a single Q4_K row by id and write it into `out` as F32. Layout
/// mirrors `dequant_q4_k`; we only walk the rows referenced by `ids`.
///
/// Dispatches to AVX-512 when available — that path unpacks 16 nibbles
/// per iteration and stores 16 f32s per call, vs the scalar loop's
/// per-nibble pattern.
pub fn embed_lookup_q4_k(table_bytes: &[u8], ids: &[i32], out: &mut [f32], d: usize) {
    const QK_K: usize = 256;
    assert_eq!(d % QK_K, 0, "Q4_K embed lookup requires d % 256 == 0");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: runtime feature detection.
            unsafe { embed_lookup_q4_k_avx512(table_bytes, ids, out, d) };
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime feature detection.
            unsafe { embed_lookup_q4_k_avx2(table_bytes, ids, out, d) };
            return;
        }
    }
    embed_lookup_q4_k_scalar(table_bytes, ids, out, d);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn embed_lookup_q4_k_avx2(table_bytes: &[u8], ids: &[i32], out: &mut [f32], d: usize) {
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 144;
    const QK_K: usize = 256;
    let blocks_per_row = d / QK_K;
    let row_bytes = blocks_per_row * BLOCK_BYTES;
    let mask_low_nibble = _mm256_set1_epi32(0x0F);

    // 8 nibbles per YMM lane (vs 16 for the AVX-512 path). Each group
    // produces 64 outputs as 8 YMM stores. Compare to AVX-512's 4
    // ZMM stores per group — same total bytes, twice the instructions.
    for (i, &id) in ids.iter().enumerate() {
        let row_start = (id as usize) * row_bytes;
        let dst = &mut out[i * d..(i + 1) * d];
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d_bits = u16::from_le_bytes([table_bytes[off], table_bytes[off + 1]]);
            let dmin_bits = u16::from_le_bytes([table_bytes[off + 2], table_bytes[off + 3]]);
            let d_scale = f16::from_bits(d_bits).to_f32();
            let dmin = f16::from_bits(dmin_bits).to_f32();
            let (sc, mn) = unpack_q4k_scales(&table_bytes[off + 4..off + 16]);
            let qs_ptr = table_bytes.as_ptr().add(off + 16);
            let dst_block = &mut dst[b * QK_K..(b + 1) * QK_K];

            for group in 0..4 {
                let group_qs = qs_ptr.add(group * 32);
                let d_lo_v = _mm256_set1_ps(d_scale * sc[group * 2] as f32);
                let m_lo_v = _mm256_set1_ps(dmin * mn[group * 2] as f32);
                let d_hi_v = _mm256_set1_ps(d_scale * sc[group * 2 + 1] as f32);
                let m_hi_v = _mm256_set1_ps(dmin * mn[group * 2 + 1] as f32);

                // 32 q-bytes → 4 × 8-lane nibble vectors (8 lanes ×
                // 4 = 32 outputs for low nibbles, similarly for high).
                let qs_lo32 = _mm256_cvtepu8_epi32(_mm_loadl_epi64(group_qs as *const __m128i));
                let qs_lo32_b = _mm256_cvtepu8_epi32(_mm_loadl_epi64(group_qs.add(8) as *const __m128i));
                let qs_lo32_c = _mm256_cvtepu8_epi32(_mm_loadl_epi64(group_qs.add(16) as *const __m128i));
                let qs_lo32_d = _mm256_cvtepu8_epi32(_mm_loadl_epi64(group_qs.add(24) as *const __m128i));

                let lo_a = _mm256_and_si256(qs_lo32, mask_low_nibble);
                let lo_b = _mm256_and_si256(qs_lo32_b, mask_low_nibble);
                let lo_c = _mm256_and_si256(qs_lo32_c, mask_low_nibble);
                let lo_d = _mm256_and_si256(qs_lo32_d, mask_low_nibble);
                let hi_a = _mm256_and_si256(_mm256_srli_epi32(qs_lo32, 4), mask_low_nibble);
                let hi_b = _mm256_and_si256(_mm256_srli_epi32(qs_lo32_b, 4), mask_low_nibble);
                let hi_c = _mm256_and_si256(_mm256_srli_epi32(qs_lo32_c, 4), mask_low_nibble);
                let hi_d = _mm256_and_si256(_mm256_srli_epi32(qs_lo32_d, 4), mask_low_nibble);

                // d * q - m for each nibble lane.
                let lo_a_f = _mm256_fmsub_ps(_mm256_cvtepi32_ps(lo_a), d_lo_v, m_lo_v);
                let lo_b_f = _mm256_fmsub_ps(_mm256_cvtepi32_ps(lo_b), d_lo_v, m_lo_v);
                let lo_c_f = _mm256_fmsub_ps(_mm256_cvtepi32_ps(lo_c), d_lo_v, m_lo_v);
                let lo_d_f = _mm256_fmsub_ps(_mm256_cvtepi32_ps(lo_d), d_lo_v, m_lo_v);
                let hi_a_f = _mm256_fmsub_ps(_mm256_cvtepi32_ps(hi_a), d_hi_v, m_hi_v);
                let hi_b_f = _mm256_fmsub_ps(_mm256_cvtepi32_ps(hi_b), d_hi_v, m_hi_v);
                let hi_c_f = _mm256_fmsub_ps(_mm256_cvtepi32_ps(hi_c), d_hi_v, m_hi_v);
                let hi_d_f = _mm256_fmsub_ps(_mm256_cvtepi32_ps(hi_d), d_hi_v, m_hi_v);

                let base = group * 64;
                let dst_ptr = dst_block.as_mut_ptr();
                _mm256_storeu_ps(dst_ptr.add(base), lo_a_f);
                _mm256_storeu_ps(dst_ptr.add(base + 8), lo_b_f);
                _mm256_storeu_ps(dst_ptr.add(base + 16), lo_c_f);
                _mm256_storeu_ps(dst_ptr.add(base + 24), lo_d_f);
                _mm256_storeu_ps(dst_ptr.add(base + 32), hi_a_f);
                _mm256_storeu_ps(dst_ptr.add(base + 40), hi_b_f);
                _mm256_storeu_ps(dst_ptr.add(base + 48), hi_c_f);
                _mm256_storeu_ps(dst_ptr.add(base + 56), hi_d_f);
            }
        }
    }
}

fn embed_lookup_q4_k_scalar(table_bytes: &[u8], ids: &[i32], out: &mut [f32], d: usize) {
    const BLOCK_BYTES: usize = 144;
    const QK_K: usize = 256;
    let blocks_per_row = d / QK_K;
    let row_bytes = blocks_per_row * BLOCK_BYTES;
    for (i, &id) in ids.iter().enumerate() {
        let row_start = (id as usize) * row_bytes;
        let dst = &mut out[i * d..(i + 1) * d];
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d_scale = f16::from_le_bytes([table_bytes[off], table_bytes[off + 1]]).to_f32();
            let dmin =
                f16::from_le_bytes([table_bytes[off + 2], table_bytes[off + 3]]).to_f32();
            let (sc, mn) = unpack_q4k_scales(&table_bytes[off + 4..off + 16]);
            let qs = &table_bytes[off + 16..off + 16 + 128];
            let dst_block = &mut dst[b * QK_K..(b + 1) * QK_K];
            for group in 0..4 {
                let q_chunk = &qs[group * 32..(group + 1) * 32];
                let d_lo = d_scale * sc[group * 2] as f32;
                let m_lo = dmin * mn[group * 2] as f32;
                let d_hi = d_scale * sc[group * 2 + 1] as f32;
                let m_hi = dmin * mn[group * 2 + 1] as f32;
                let base = group * 64;
                for l in 0..32 {
                    let q = q_chunk[l];
                    dst_block[base + l] = d_lo * (q & 0x0F) as f32 - m_lo;
                    dst_block[base + 32 + l] = d_hi * (q >> 4) as f32 - m_hi;
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f")]
unsafe fn embed_lookup_q4_k_avx512(table_bytes: &[u8], ids: &[i32], out: &mut [f32], d: usize) {
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 144;
    const QK_K: usize = 256;
    let blocks_per_row = d / QK_K;
    let row_bytes = blocks_per_row * BLOCK_BYTES;
    let mask_low_nibble = _mm512_set1_epi32(0x0F);

    // Unpack 16 q-bytes → (lo_nibbles, hi_nibbles) as ZMM i32x16 lanes.
    #[inline(always)]
    unsafe fn unpack_16_q_bytes_avx512(
        qs_ptr: *const u8,
        mask: __m512i,
    ) -> (__m512i, __m512i) {
        let raw = _mm_loadu_si128(qs_ptr as *const __m128i);
        let widened = _mm512_cvtepu8_epi32(raw);
        let lo = _mm512_and_si512(widened, mask);
        let hi = _mm512_and_si512(_mm512_srli_epi32(widened, 4), mask);
        (lo, hi)
    }

    for (i, &id) in ids.iter().enumerate() {
        let row_start = (id as usize) * row_bytes;
        let dst = &mut out[i * d..(i + 1) * d];
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d_bits = u16::from_le_bytes([table_bytes[off], table_bytes[off + 1]]);
            let dmin_bits = u16::from_le_bytes([table_bytes[off + 2], table_bytes[off + 3]]);
            let d_scale = f16::from_bits(d_bits).to_f32();
            let dmin = f16::from_bits(dmin_bits).to_f32();
            let (sc, mn) = unpack_q4k_scales(&table_bytes[off + 4..off + 16]);
            let qs_ptr = table_bytes.as_ptr().add(off + 16);
            let dst_block = &mut dst[b * QK_K..(b + 1) * QK_K];

            for group in 0..4 {
                let group_qs = qs_ptr.add(group * 32);
                let d_lo_v = _mm512_set1_ps(d_scale * sc[group * 2] as f32);
                let m_lo_v = _mm512_set1_ps(dmin * mn[group * 2] as f32);
                let d_hi_v = _mm512_set1_ps(d_scale * sc[group * 2 + 1] as f32);
                let m_hi_v = _mm512_set1_ps(dmin * mn[group * 2 + 1] as f32);

                // Two 16-byte halves of the 32-byte q chunk → 4 ZMM lanes
                // (2 low + 2 high) of 16 nibbles each.
                let (lo_a, hi_a) = unpack_16_q_bytes_avx512(group_qs, mask_low_nibble);
                let (lo_b, hi_b) =
                    unpack_16_q_bytes_avx512(group_qs.add(16), mask_low_nibble);

                // d * q - m for each nibble lane.
                let lo_a_f = _mm512_fmsub_ps(_mm512_cvtepi32_ps(lo_a), d_lo_v, m_lo_v);
                let lo_b_f = _mm512_fmsub_ps(_mm512_cvtepi32_ps(lo_b), d_lo_v, m_lo_v);
                let hi_a_f = _mm512_fmsub_ps(_mm512_cvtepi32_ps(hi_a), d_hi_v, m_hi_v);
                let hi_b_f = _mm512_fmsub_ps(_mm512_cvtepi32_ps(hi_b), d_hi_v, m_hi_v);

                // Layout in dst_block matches the scalar path:
                //   [base..base+32)   ← low nibbles (32 of them)
                //   [base+32..base+64) ← high nibbles
                let base = group * 64;
                let dst_ptr = dst_block.as_mut_ptr();
                _mm512_storeu_ps(dst_ptr.add(base), lo_a_f);
                _mm512_storeu_ps(dst_ptr.add(base + 16), lo_b_f);
                _mm512_storeu_ps(dst_ptr.add(base + 32), hi_a_f);
                _mm512_storeu_ps(dst_ptr.add(base + 48), hi_b_f);
            }
        }
    }
}

/// Decode a single Q5_K row by id and write it into `out` as F32.
pub fn embed_lookup_q5_k(table_bytes: &[u8], ids: &[i32], out: &mut [f32], d: usize) {
    const QK_K: usize = 256;
    assert_eq!(d % QK_K, 0, "Q5_K embed lookup requires d % 256 == 0");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: runtime feature detection.
            unsafe { embed_lookup_q5_k_avx512(table_bytes, ids, out, d) };
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime feature detection.
            unsafe { embed_lookup_q5_k_avx2(table_bytes, ids, out, d) };
            return;
        }
    }
    embed_lookup_q5_k_scalar(table_bytes, ids, out, d);
}

fn embed_lookup_q5_k_scalar(table_bytes: &[u8], ids: &[i32], out: &mut [f32], d: usize) {
    const BLOCK_BYTES: usize = 176;
    const QK_K: usize = 256;
    let blocks_per_row = d / QK_K;
    let row_bytes = blocks_per_row * BLOCK_BYTES;
    for (i, &id) in ids.iter().enumerate() {
        let row_start = (id as usize) * row_bytes;
        let dst = &mut out[i * d..(i + 1) * d];
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d_scale = f16::from_le_bytes([table_bytes[off], table_bytes[off + 1]]).to_f32();
            let dmin =
                f16::from_le_bytes([table_bytes[off + 2], table_bytes[off + 3]]).to_f32();
            let (sc, mn) = unpack_q4k_scales(&table_bytes[off + 4..off + 16]);
            let qh = &table_bytes[off + 16..off + 48];
            let qs = &table_bytes[off + 48..off + 48 + 128];
            let dst_block = &mut dst[b * QK_K..(b + 1) * QK_K];
            for group in 0..4 {
                let q_chunk = &qs[group * 32..(group + 1) * 32];
                let d_lo = d_scale * sc[group * 2] as f32;
                let m_lo = dmin * mn[group * 2] as f32;
                let d_hi = d_scale * sc[group * 2 + 1] as f32;
                let m_hi = dmin * mn[group * 2 + 1] as f32;
                let bit_lo = group * 2;
                let bit_hi = group * 2 + 1;
                let base = group * 64;
                for l in 0..32 {
                    let q_byte = q_chunk[l];
                    let qh_byte = qh[l];
                    let lo = q_byte & 0x0F;
                    let hi = q_byte >> 4;
                    let bit_lo_set = ((qh_byte >> bit_lo) & 1) != 0;
                    let bit_hi_set = ((qh_byte >> bit_hi) & 1) != 0;
                    let lo_full = lo | (if bit_lo_set { 16 } else { 0 });
                    let hi_full = hi | (if bit_hi_set { 16 } else { 0 });
                    dst_block[base + l] = d_lo * lo_full as f32 - m_lo;
                    dst_block[base + 32 + l] = d_hi * hi_full as f32 - m_hi;
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn embed_lookup_q5_k_avx2(table_bytes: &[u8], ids: &[i32], out: &mut [f32], d: usize) {
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 176;
    const QK_K: usize = 256;
    let blocks_per_row = d / QK_K;
    let row_bytes = blocks_per_row * BLOCK_BYTES;
    let mask_low_nibble = _mm256_set1_epi32(0x0F);
    let one = _mm256_set1_epi32(1);

    #[inline(always)]
    unsafe fn qh_high_bits(
        qh_ptr: *const u8,
        shift_count: __m128i,
        one: __m256i,
    ) -> __m256i {
        let qh_lane = _mm256_cvtepu8_epi32(_mm_loadl_epi64(qh_ptr as *const __m128i));
        let shifted = _mm256_srl_epi32(qh_lane, shift_count);
        let bit = _mm256_and_si256(shifted, one);
        _mm256_slli_epi32(bit, 4)
    }

    for (i, &id) in ids.iter().enumerate() {
        let row_start = (id as usize) * row_bytes;
        let dst = &mut out[i * d..(i + 1) * d];
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d_bits = u16::from_le_bytes([table_bytes[off], table_bytes[off + 1]]);
            let dmin_bits = u16::from_le_bytes([table_bytes[off + 2], table_bytes[off + 3]]);
            let d_scale = f16::from_bits(d_bits).to_f32();
            let dmin = f16::from_bits(dmin_bits).to_f32();
            let (sc, mn) = unpack_q4k_scales(&table_bytes[off + 4..off + 16]);
            let qh_ptr = table_bytes.as_ptr().add(off + 16);
            let qs_ptr = table_bytes.as_ptr().add(off + 48);
            let dst_block = &mut dst[b * QK_K..(b + 1) * QK_K];

            for group in 0..4 {
                let d_lo_v = _mm256_set1_ps(d_scale * sc[group * 2] as f32);
                let m_lo_v = _mm256_set1_ps(dmin * mn[group * 2] as f32);
                let d_hi_v = _mm256_set1_ps(d_scale * sc[group * 2 + 1] as f32);
                let m_hi_v = _mm256_set1_ps(dmin * mn[group * 2 + 1] as f32);
                let shift_lo = _mm_cvtsi32_si128((group * 2) as i32);
                let shift_hi = _mm_cvtsi32_si128((group * 2 + 1) as i32);

                let group_qs = qs_ptr.add(group * 32);
                // 4 × 8-lane nibble vectors per half (low + high).
                let qs_a = _mm256_cvtepu8_epi32(_mm_loadl_epi64(group_qs as *const __m128i));
                let qs_b = _mm256_cvtepu8_epi32(_mm_loadl_epi64(group_qs.add(8) as *const __m128i));
                let qs_c = _mm256_cvtepu8_epi32(_mm_loadl_epi64(group_qs.add(16) as *const __m128i));
                let qs_d = _mm256_cvtepu8_epi32(_mm_loadl_epi64(group_qs.add(24) as *const __m128i));

                let lo_a = _mm256_and_si256(qs_a, mask_low_nibble);
                let lo_b = _mm256_and_si256(qs_b, mask_low_nibble);
                let lo_c = _mm256_and_si256(qs_c, mask_low_nibble);
                let lo_d = _mm256_and_si256(qs_d, mask_low_nibble);
                let hi_a = _mm256_and_si256(_mm256_srli_epi32(qs_a, 4), mask_low_nibble);
                let hi_b = _mm256_and_si256(_mm256_srli_epi32(qs_b, 4), mask_low_nibble);
                let hi_c = _mm256_and_si256(_mm256_srli_epi32(qs_c, 4), mask_low_nibble);
                let hi_d = _mm256_and_si256(_mm256_srli_epi32(qs_d, 4), mask_low_nibble);

                // qh high-bit contributions, one per 8-lane chunk.
                let qhi_lo_a = qh_high_bits(qh_ptr, shift_lo, one);
                let qhi_lo_b = qh_high_bits(qh_ptr.add(8), shift_lo, one);
                let qhi_lo_c = qh_high_bits(qh_ptr.add(16), shift_lo, one);
                let qhi_lo_d = qh_high_bits(qh_ptr.add(24), shift_lo, one);
                let qhi_hi_a = qh_high_bits(qh_ptr, shift_hi, one);
                let qhi_hi_b = qh_high_bits(qh_ptr.add(8), shift_hi, one);
                let qhi_hi_c = qh_high_bits(qh_ptr.add(16), shift_hi, one);
                let qhi_hi_d = qh_high_bits(qh_ptr.add(24), shift_hi, one);

                let q_lo_a = _mm256_or_si256(lo_a, qhi_lo_a);
                let q_lo_b = _mm256_or_si256(lo_b, qhi_lo_b);
                let q_lo_c = _mm256_or_si256(lo_c, qhi_lo_c);
                let q_lo_d = _mm256_or_si256(lo_d, qhi_lo_d);
                let q_hi_a = _mm256_or_si256(hi_a, qhi_hi_a);
                let q_hi_b = _mm256_or_si256(hi_b, qhi_hi_b);
                let q_hi_c = _mm256_or_si256(hi_c, qhi_hi_c);
                let q_hi_d = _mm256_or_si256(hi_d, qhi_hi_d);

                let lo_a_f = _mm256_fmsub_ps(_mm256_cvtepi32_ps(q_lo_a), d_lo_v, m_lo_v);
                let lo_b_f = _mm256_fmsub_ps(_mm256_cvtepi32_ps(q_lo_b), d_lo_v, m_lo_v);
                let lo_c_f = _mm256_fmsub_ps(_mm256_cvtepi32_ps(q_lo_c), d_lo_v, m_lo_v);
                let lo_d_f = _mm256_fmsub_ps(_mm256_cvtepi32_ps(q_lo_d), d_lo_v, m_lo_v);
                let hi_a_f = _mm256_fmsub_ps(_mm256_cvtepi32_ps(q_hi_a), d_hi_v, m_hi_v);
                let hi_b_f = _mm256_fmsub_ps(_mm256_cvtepi32_ps(q_hi_b), d_hi_v, m_hi_v);
                let hi_c_f = _mm256_fmsub_ps(_mm256_cvtepi32_ps(q_hi_c), d_hi_v, m_hi_v);
                let hi_d_f = _mm256_fmsub_ps(_mm256_cvtepi32_ps(q_hi_d), d_hi_v, m_hi_v);

                let base = group * 64;
                let dst_ptr = dst_block.as_mut_ptr();
                _mm256_storeu_ps(dst_ptr.add(base), lo_a_f);
                _mm256_storeu_ps(dst_ptr.add(base + 8), lo_b_f);
                _mm256_storeu_ps(dst_ptr.add(base + 16), lo_c_f);
                _mm256_storeu_ps(dst_ptr.add(base + 24), lo_d_f);
                _mm256_storeu_ps(dst_ptr.add(base + 32), hi_a_f);
                _mm256_storeu_ps(dst_ptr.add(base + 40), hi_b_f);
                _mm256_storeu_ps(dst_ptr.add(base + 48), hi_c_f);
                _mm256_storeu_ps(dst_ptr.add(base + 56), hi_d_f);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f")]
unsafe fn embed_lookup_q5_k_avx512(table_bytes: &[u8], ids: &[i32], out: &mut [f32], d: usize) {
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 176;
    const QK_K: usize = 256;
    let blocks_per_row = d / QK_K;
    let row_bytes = blocks_per_row * BLOCK_BYTES;
    let mask_low_nibble = _mm512_set1_epi32(0x0F);
    let one = _mm512_set1_epi32(1);

    #[inline(always)]
    unsafe fn qh_high_bits_avx512(
        qh_ptr: *const u8,
        shift_count: __m128i,
        one: __m512i,
    ) -> __m512i {
        let raw = _mm_loadu_si128(qh_ptr as *const __m128i);
        let widened = _mm512_cvtepu8_epi32(raw);
        let shifted = _mm512_srl_epi32(widened, shift_count);
        let bit = _mm512_and_si512(shifted, one);
        _mm512_slli_epi32(bit, 4)
    }

    for (i, &id) in ids.iter().enumerate() {
        let row_start = (id as usize) * row_bytes;
        let dst = &mut out[i * d..(i + 1) * d];
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d_bits = u16::from_le_bytes([table_bytes[off], table_bytes[off + 1]]);
            let dmin_bits = u16::from_le_bytes([table_bytes[off + 2], table_bytes[off + 3]]);
            let d_scale = f16::from_bits(d_bits).to_f32();
            let dmin = f16::from_bits(dmin_bits).to_f32();
            let (sc, mn) = unpack_q4k_scales(&table_bytes[off + 4..off + 16]);
            let qh_ptr = table_bytes.as_ptr().add(off + 16);
            let qs_ptr = table_bytes.as_ptr().add(off + 48);
            let dst_block = &mut dst[b * QK_K..(b + 1) * QK_K];

            for group in 0..4 {
                let d_lo_v = _mm512_set1_ps(d_scale * sc[group * 2] as f32);
                let m_lo_v = _mm512_set1_ps(dmin * mn[group * 2] as f32);
                let d_hi_v = _mm512_set1_ps(d_scale * sc[group * 2 + 1] as f32);
                let m_hi_v = _mm512_set1_ps(dmin * mn[group * 2 + 1] as f32);
                let shift_lo = _mm_cvtsi32_si128((group * 2) as i32);
                let shift_hi = _mm_cvtsi32_si128((group * 2 + 1) as i32);

                let group_qs = qs_ptr.add(group * 32);
                // 16 q-bytes per ZMM lane group → two halves per group.
                let qs_a = _mm512_cvtepu8_epi32(_mm_loadu_si128(group_qs as *const __m128i));
                let qs_b =
                    _mm512_cvtepu8_epi32(_mm_loadu_si128(group_qs.add(16) as *const __m128i));

                let lo_a = _mm512_and_si512(qs_a, mask_low_nibble);
                let lo_b = _mm512_and_si512(qs_b, mask_low_nibble);
                let hi_a = _mm512_and_si512(_mm512_srli_epi32(qs_a, 4), mask_low_nibble);
                let hi_b = _mm512_and_si512(_mm512_srli_epi32(qs_b, 4), mask_low_nibble);

                let qhi_lo_a = qh_high_bits_avx512(qh_ptr, shift_lo, one);
                let qhi_lo_b = qh_high_bits_avx512(qh_ptr.add(16), shift_lo, one);
                let qhi_hi_a = qh_high_bits_avx512(qh_ptr, shift_hi, one);
                let qhi_hi_b = qh_high_bits_avx512(qh_ptr.add(16), shift_hi, one);

                let q_lo_a = _mm512_or_si512(lo_a, qhi_lo_a);
                let q_lo_b = _mm512_or_si512(lo_b, qhi_lo_b);
                let q_hi_a = _mm512_or_si512(hi_a, qhi_hi_a);
                let q_hi_b = _mm512_or_si512(hi_b, qhi_hi_b);

                let lo_a_f = _mm512_fmsub_ps(_mm512_cvtepi32_ps(q_lo_a), d_lo_v, m_lo_v);
                let lo_b_f = _mm512_fmsub_ps(_mm512_cvtepi32_ps(q_lo_b), d_lo_v, m_lo_v);
                let hi_a_f = _mm512_fmsub_ps(_mm512_cvtepi32_ps(q_hi_a), d_hi_v, m_hi_v);
                let hi_b_f = _mm512_fmsub_ps(_mm512_cvtepi32_ps(q_hi_b), d_hi_v, m_hi_v);

                let base = group * 64;
                let dst_ptr = dst_block.as_mut_ptr();
                _mm512_storeu_ps(dst_ptr.add(base), lo_a_f);
                _mm512_storeu_ps(dst_ptr.add(base + 16), lo_b_f);
                _mm512_storeu_ps(dst_ptr.add(base + 32), hi_a_f);
                _mm512_storeu_ps(dst_ptr.add(base + 48), hi_b_f);
            }
        }
    }
}

/// Decode a single Q6_K row by id and write it into `out` as F32. Mirrors
/// the per-block layout in `dequant_q6_k`; scales are signed int8.
pub fn embed_lookup_q6_k(table_bytes: &[u8], ids: &[i32], out: &mut [f32], d: usize) {
    const QK_K: usize = 256;
    assert_eq!(d % QK_K, 0, "Q6_K embed lookup requires d % 256 == 0");

    // Q6_K's strand layout (4 strands × 4 lanes × 8 outputs = 128
    // outputs per half × 2 halves) is gnarly to SIMD cleanly. The
    // scalar path here writes outputs at strided positions (32-byte
    // gaps), and the per-strand 2-bit qh field needs runtime-variable
    // shifts. For now we keep the scalar path; AVX2/AVX-512 follow-ups
    // would mirror the matvec AVX-512 path's strand structure. Q6_K
    // appears mainly in `token_embd` and `output` tensors for some
    // K-quant variants, so embed_lookup runs ~once per token — much
    // less hot than the matvec.
    embed_lookup_q6_k_scalar(table_bytes, ids, out, d);
}

fn embed_lookup_q6_k_scalar(table_bytes: &[u8], ids: &[i32], out: &mut [f32], d: usize) {
    const BLOCK_BYTES: usize = 210;
    const QK_K: usize = 256;
    let blocks_per_row = d / QK_K;
    let row_bytes = blocks_per_row * BLOCK_BYTES;
    for (i, &id) in ids.iter().enumerate() {
        let row_start = (id as usize) * row_bytes;
        let dst = &mut out[i * d..(i + 1) * d];
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let ql = &table_bytes[off..off + 128];
            let qh = &table_bytes[off + 128..off + 128 + 64];
            let scales_raw = &table_bytes[off + 128 + 64..off + 128 + 64 + 16];
            let d_scale =
                f16::from_le_bytes([table_bytes[off + 208], table_bytes[off + 209]]).to_f32();
            let dst_block = &mut dst[b * QK_K..(b + 1) * QK_K];
            for n in 0..2 {
                for l in 0..32 {
                    let is = l / 16 + n * 8;
                    let q1 = ((ql[64 * n + l] & 0x0F) as i32
                        | (((qh[32 * n + l] >> 0) & 0x03) << 4) as i32)
                        - 32;
                    let q2 = ((ql[64 * n + l + 32] & 0x0F) as i32
                        | (((qh[32 * n + l] >> 2) & 0x03) << 4) as i32)
                        - 32;
                    let q3 = ((ql[64 * n + l] >> 4) as i32
                        | (((qh[32 * n + l] >> 4) & 0x03) << 4) as i32)
                        - 32;
                    let q4 = ((ql[64 * n + l + 32] >> 4) as i32
                        | (((qh[32 * n + l] >> 6) & 0x03) << 4) as i32)
                        - 32;
                    let base = n * 128 + l;
                    let s0 = (scales_raw[is] as i8) as f32;
                    let s1 = (scales_raw[is + 2] as i8) as f32;
                    let s2 = (scales_raw[is + 4] as i8) as f32;
                    let s3 = (scales_raw[is + 6] as i8) as f32;
                    dst_block[base] = d_scale * s0 * q1 as f32;
                    dst_block[base + 32] = d_scale * s1 * q2 as f32;
                    dst_block[base + 64] = d_scale * s2 * q3 as f32;
                    dst_block[base + 96] = d_scale * s3 * q4 as f32;
                }
            }
        }
    }
}

/// Decode a single Q5_0 row by id and write it into `out` as F32.
pub fn embed_lookup_q5_0(table_bytes: &[u8], ids: &[i32], out: &mut [f32], d: usize) {
    const QK: usize = 32;
    assert_eq!(d % QK, 0, "Q5_0 embed lookup requires d % 32 == 0");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: runtime feature detection.
            unsafe { embed_lookup_q5_0_avx512(table_bytes, ids, out, d) };
            return;
        }
        if is_x86_feature_detected!("avx2") {
            // SAFETY: runtime feature detection.
            unsafe { embed_lookup_q5_0_avx2(table_bytes, ids, out, d) };
            return;
        }
    }
    embed_lookup_q5_0_scalar(table_bytes, ids, out, d);
}

fn embed_lookup_q5_0_scalar(table_bytes: &[u8], ids: &[i32], out: &mut [f32], d: usize) {
    const BLOCK_BYTES: usize = 22;
    const QK: usize = 32;
    let blocks_per_row = d / QK;
    let row_bytes = blocks_per_row * BLOCK_BYTES;
    for (i, &id) in ids.iter().enumerate() {
        let row_start = (id as usize) * row_bytes;
        let dst = &mut out[i * d..(i + 1) * d];
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let scale =
                f16::from_le_bytes([table_bytes[off], table_bytes[off + 1]]).to_f32();
            let qh = u32::from_le_bytes([
                table_bytes[off + 2],
                table_bytes[off + 3],
                table_bytes[off + 4],
                table_bytes[off + 5],
            ]);
            let qs = &table_bytes[off + 6..off + 22];
            let dst_chunk = &mut dst[b * QK..(b + 1) * QK];
            for j in 0..16 {
                let xh_0 = ((qh >> j) << 4) & 0x10;
                let xh_1 = (qh >> (j + 12)) & 0x10;
                let x0 = ((qs[j] & 0x0F) as i32 | xh_0 as i32) - 16;
                let x1 = ((qs[j] >> 4) as i32 | xh_1 as i32) - 16;
                dst_chunk[j] = scale * x0 as f32;
                dst_chunk[j + 16] = scale * x1 as f32;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2")]
unsafe fn embed_lookup_q5_0_avx2(table_bytes: &[u8], ids: &[i32], out: &mut [f32], d: usize) {
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 22;
    const QK: usize = 32;
    let blocks_per_row = d / QK;
    let row_bytes = blocks_per_row * BLOCK_BYTES;

    // Per-lane shifts so a broadcast of the 32-bit `qh` field shifted
    // by each lane's count gives us the 5th-bit table for outputs
    // 0..7 / 8..15 / 16..23 / 24..31.
    let shifts_0_7 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let shifts_8_15 = _mm256_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15);
    let shifts_16_23 = _mm256_setr_epi32(16, 17, 18, 19, 20, 21, 22, 23);
    let shifts_24_31 = _mm256_setr_epi32(24, 25, 26, 27, 28, 29, 30, 31);
    let one = _mm256_set1_epi32(1);
    let sixteen = _mm256_set1_epi32(16);
    let mask_low_nibble = _mm256_set1_epi32(0x0F);

    #[inline(always)]
    unsafe fn high_bits(qh_vec: __m256i, shifts: __m256i, one: __m256i) -> __m256i {
        let shifted = _mm256_srlv_epi32(qh_vec, shifts);
        let bit = _mm256_and_si256(shifted, one);
        _mm256_slli_epi32(bit, 4)
    }

    for (i, &id) in ids.iter().enumerate() {
        let row_start = (id as usize) * row_bytes;
        let dst = &mut out[i * d..(i + 1) * d];
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let scale_bits = u16::from_le_bytes([table_bytes[off], table_bytes[off + 1]]);
            let scale = _mm256_set1_ps(f16::from_bits(scale_bits).to_f32());
            let qh_u32 = u32::from_le_bytes([
                table_bytes[off + 2], table_bytes[off + 3],
                table_bytes[off + 4], table_bytes[off + 5],
            ]);
            let qh_vec = _mm256_set1_epi32(qh_u32 as i32);
            let qs_ptr = table_bytes.as_ptr().add(off + 6);

            let qs_lo32 = _mm256_cvtepu8_epi32(_mm_loadl_epi64(qs_ptr as *const __m128i));
            let qs_hi32 = _mm256_cvtepu8_epi32(_mm_loadl_epi64(qs_ptr.add(8) as *const __m128i));

            let lo_07 = _mm256_and_si256(qs_lo32, mask_low_nibble);
            let lo_815 = _mm256_and_si256(qs_hi32, mask_low_nibble);
            let hi_07 = _mm256_and_si256(_mm256_srli_epi32(qs_lo32, 4), mask_low_nibble);
            let hi_815 = _mm256_and_si256(_mm256_srli_epi32(qs_hi32, 4), mask_low_nibble);

            // 5-bit signed: (nibble | (qh_bit << 4)) - 16.
            let q0 = _mm256_sub_epi32(
                _mm256_or_si256(lo_07, high_bits(qh_vec, shifts_0_7, one)), sixteen);
            let q1 = _mm256_sub_epi32(
                _mm256_or_si256(lo_815, high_bits(qh_vec, shifts_8_15, one)), sixteen);
            let q2 = _mm256_sub_epi32(
                _mm256_or_si256(hi_07, high_bits(qh_vec, shifts_16_23, one)), sixteen);
            let q3 = _mm256_sub_epi32(
                _mm256_or_si256(hi_815, high_bits(qh_vec, shifts_24_31, one)), sixteen);

            let dst_ptr = dst.as_mut_ptr().add(b * QK);
            _mm256_storeu_ps(dst_ptr, _mm256_mul_ps(scale, _mm256_cvtepi32_ps(q0)));
            _mm256_storeu_ps(dst_ptr.add(8), _mm256_mul_ps(scale, _mm256_cvtepi32_ps(q1)));
            _mm256_storeu_ps(dst_ptr.add(16), _mm256_mul_ps(scale, _mm256_cvtepi32_ps(q2)));
            _mm256_storeu_ps(dst_ptr.add(24), _mm256_mul_ps(scale, _mm256_cvtepi32_ps(q3)));
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f")]
unsafe fn embed_lookup_q5_0_avx512(table_bytes: &[u8], ids: &[i32], out: &mut [f32], d: usize) {
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 22;
    const QK: usize = 32;
    let blocks_per_row = d / QK;
    let row_bytes = blocks_per_row * BLOCK_BYTES;

    let shifts_0_15 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    let shifts_16_31 = _mm512_setr_epi32(
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    );
    let one = _mm512_set1_epi32(1);
    let sixteen = _mm512_set1_epi32(16);
    let mask_low_nibble = _mm512_set1_epi32(0x0F);

    #[inline(always)]
    unsafe fn high_bits(qh_vec: __m512i, shifts: __m512i, one: __m512i) -> __m512i {
        let shifted = _mm512_srlv_epi32(qh_vec, shifts);
        let bit = _mm512_and_si512(shifted, one);
        _mm512_slli_epi32(bit, 4)
    }

    for (i, &id) in ids.iter().enumerate() {
        let row_start = (id as usize) * row_bytes;
        let dst = &mut out[i * d..(i + 1) * d];
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let scale_bits = u16::from_le_bytes([table_bytes[off], table_bytes[off + 1]]);
            let scale = _mm512_set1_ps(f16::from_bits(scale_bits).to_f32());
            let qh_u32 = u32::from_le_bytes([
                table_bytes[off + 2], table_bytes[off + 3],
                table_bytes[off + 4], table_bytes[off + 5],
            ]);
            let qh_vec = _mm512_set1_epi32(qh_u32 as i32);

            // 16 qs bytes → 16 low nibbles + 16 high nibbles.
            let qs_raw = _mm_loadu_si128(table_bytes.as_ptr().add(off + 6) as *const __m128i);
            let qs_lane = _mm512_cvtepu8_epi32(qs_raw);
            let lo_nibbles = _mm512_and_si512(qs_lane, mask_low_nibble);
            let hi_nibbles = _mm512_and_si512(_mm512_srli_epi32(qs_lane, 4), mask_low_nibble);

            let q_lo = _mm512_sub_epi32(
                _mm512_or_si512(lo_nibbles, high_bits(qh_vec, shifts_0_15, one)),
                sixteen,
            );
            let q_hi = _mm512_sub_epi32(
                _mm512_or_si512(hi_nibbles, high_bits(qh_vec, shifts_16_31, one)),
                sixteen,
            );

            let dst_ptr = dst.as_mut_ptr().add(b * QK);
            _mm512_storeu_ps(dst_ptr, _mm512_mul_ps(scale, _mm512_cvtepi32_ps(q_lo)));
            _mm512_storeu_ps(dst_ptr.add(16), _mm512_mul_ps(scale, _mm512_cvtepi32_ps(q_hi)));
        }
    }
}

/// Q4_0 weight matvec — 18 bytes per 32 weights (f16 scale + 16 q-bytes
/// packing two 4-bit nibbles each). Decode-and-FMA fused so weights
/// never touch an intermediate F16/F32 buffer.
///
/// Per-block dequant: `v_lo = d * (qs[j] & 0xF - 8)` for j in 0..16
/// (outputs 0..15); `v_hi = d * (qs[j] >> 4 - 8)` (outputs 16..31).
///
/// Dispatches AVX-512 → AVX2 → scalar.
///
/// Assumes `k % 32 == 0` (matches real-world Linear weights).
pub fn matvec_q4_0_w_f32_a(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    const BLOCK_BYTES: usize = 18;
    const QK: usize = 32;
    debug_assert_eq!(k % QK, 0);
    let blocks_per_row = k / QK;
    debug_assert_eq!(w_bytes.len(), m * blocks_per_row * BLOCK_BYTES);
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(out.len(), m);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: runtime feature detection above.
            unsafe { matvec_q4_0_w_f32_a_avx512(w_bytes, x, out, m, k) };
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime feature detection above.
            unsafe { matvec_q4_0_w_f32_a_avx2(w_bytes, x, out, m, k) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64.
        unsafe { matvec_q4_0_w_f32_a_neon(w_bytes, x, out, m, k) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    matvec_q4_0_w_f32_a_scalar(w_bytes, x, out, m, k);
}

/// AArch64 NEON Q4_0 matvec (weight Q4_0, activation f32). Mirrors the
/// AVX2/scalar math: per 32-weight block, 16 low nibbles (outputs 0..15,
/// activations x[0..16]) and 16 high nibbles (outputs 16..31, x[16..32]),
/// each recentered by -8 and scaled by the f16 block scale `d`. The 16
/// packed bytes widen to two sets of four `float32x4` lanes via the shared
/// `u8x16_to_f32x4x4` helper; 4 accumulators keep the FMA pipeline full;
/// one horizontal sum per row. NEON is baseline on aarch64 (no runtime
/// detect). Not bit-identical to the scalar reference (d folded per block,
/// vector summation) — same tolerance contract as the AVX2 path.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn matvec_q4_0_w_f32_a_neon(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use std::arch::aarch64::*;
    const BLOCK_BYTES: usize = 18;
    const QK: usize = 32;
    let blocks_per_row = k / QK;
    let mask_lo = vdupq_n_u8(0x0F);
    let eight = vdupq_n_f32(8.0);
    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = [vdupq_n_f32(0.0); 4];
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let dv = vdupq_n_f32(f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32());
            // 16 packed bytes → 16 low nibbles (qs[j]&0xF) + 16 high nibbles (qs[j]>>4).
            let qb16 = vld1q_u8(w_bytes.as_ptr().add(off + 2));
            let lo_f = u8x16_to_f32x4x4(vandq_u8(qb16, mask_lo));
            let hi_f = u8x16_to_f32x4x4(vshrq_n_u8::<4>(qb16));
            let xlo = x.as_ptr().add(b * QK);
            let xhi = xlo.add(16);
            for c in 0..4 {
                let xl = vld1q_f32(xlo.add(c * 4));
                let xh = vld1q_f32(xhi.add(c * 4));
                // coef = d * (nibble - 8)
                let coef_lo = vmulq_f32(dv, vsubq_f32(lo_f[c], eight));
                let coef_hi = vmulq_f32(dv, vsubq_f32(hi_f[c], eight));
                acc[c] = vfmaq_f32(acc[c], coef_lo, xl);
                acc[c] = vfmaq_f32(acc[c], coef_hi, xh);
            }
        }
        let s = vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3]));
        out[i] = vaddvq_f32(s);
    }
}

fn matvec_q4_0_w_f32_a_scalar(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    const BLOCK_BYTES: usize = 18;
    const QK: usize = 32;
    let blocks_per_row = k / QK;

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d = f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32();
            let qs = &w_bytes[off + 2..off + 2 + 16];
            let x_chunk = &x[b * QK..(b + 1) * QK];
            for j in 0..16 {
                let x0 = (qs[j] & 0x0F) as i32 - 8;
                let x1 = (qs[j] >> 4) as i32 - 8;
                acc += d * x0 as f32 * x_chunk[j];
                acc += d * x1 as f32 * x_chunk[j + 16];
            }
        }
        out[i] = acc;
    }
}

/// Q4_0 matvec — AVX-512 path. One block (32 weights) per iteration:
/// load 16 packed-nibble bytes, widen to 16 i32 lanes, mask + shift
/// for the low/high halves, subtract 8, multiply by `d`, FMA against
/// 32 elements of x split across two ZMM loads.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f")]
unsafe fn matvec_q4_0_w_f32_a_avx512(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 18;
    const QK: usize = 32;
    let blocks_per_row = k / QK;
    let mask_low_nibble = _mm512_set1_epi32(0x0F);
    let eight = _mm512_set1_epi32(8);

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();

        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d_bits = u16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]);
            let d_f32 = f16::from_bits(d_bits).to_f32();
            let d = _mm512_set1_ps(d_f32);

            let qs_ptr = w_bytes.as_ptr().add(off + 2);
            let qs_raw = _mm_loadu_si128(qs_ptr as *const __m128i);
            let qs_lane = _mm512_cvtepu8_epi32(qs_raw);
            let lo_nibbles = _mm512_and_si512(qs_lane, mask_low_nibble);
            let hi_nibbles =
                _mm512_and_si512(_mm512_srli_epi32(qs_lane, 4), mask_low_nibble);

            let q_lo = _mm512_sub_epi32(lo_nibbles, eight);
            let q_hi = _mm512_sub_epi32(hi_nibbles, eight);

            let dq_lo = _mm512_mul_ps(d, _mm512_cvtepi32_ps(q_lo));
            let dq_hi = _mm512_mul_ps(d, _mm512_cvtepi32_ps(q_hi));

            let xptr = x.as_ptr().add(b * QK);
            let xv0 = _mm512_loadu_ps(xptr);
            let xv1 = _mm512_loadu_ps(xptr.add(16));

            acc0 = _mm512_fmadd_ps(dq_lo, xv0, acc0);
            acc1 = _mm512_fmadd_ps(dq_hi, xv1, acc1);
        }
        out[i] = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
    }
}

/// Q4_0 matvec — AVX2 path. 16 weights per YMM iteration (half a block);
/// two YMM ops per block. Same recipe as the AVX-512 path scaled down.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn matvec_q4_0_w_f32_a_avx2(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 18;
    const QK: usize = 32;
    let blocks_per_row = k / QK;
    let mask_low_nibble = _mm256_set1_epi32(0x0F);
    let eight = _mm256_set1_epi32(8);

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d_bits = u16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]);
            let d_f32 = f16::from_bits(d_bits).to_f32();
            let d = _mm256_set1_ps(d_f32);

            // 16 qs bytes split into 2 × 8-byte halves; each half yields
            // 8 i32 lanes after `_mm256_cvtepu8_epi32`. For Q4_0 we then
            // extract low + high nibbles from each, giving 4 YMM lanes
            // (= 32 weights) per block.
            let qs_ptr = w_bytes.as_ptr().add(off + 2);
            let lo_half = _mm_loadl_epi64(qs_ptr as *const __m128i);
            let hi_half = _mm_loadl_epi64(qs_ptr.add(8) as *const __m128i);
            let lo_widened = _mm256_cvtepu8_epi32(lo_half);
            let hi_widened = _mm256_cvtepu8_epi32(hi_half);

            let lo_lo = _mm256_and_si256(lo_widened, mask_low_nibble);
            let lo_hi =
                _mm256_and_si256(_mm256_srli_epi32(lo_widened, 4), mask_low_nibble);
            let hi_lo = _mm256_and_si256(hi_widened, mask_low_nibble);
            let hi_hi =
                _mm256_and_si256(_mm256_srli_epi32(hi_widened, 4), mask_low_nibble);

            let q_lo_lo = _mm256_sub_epi32(lo_lo, eight);
            let q_lo_hi = _mm256_sub_epi32(lo_hi, eight);
            let q_hi_lo = _mm256_sub_epi32(hi_lo, eight);
            let q_hi_hi = _mm256_sub_epi32(hi_hi, eight);

            let dq0 = _mm256_mul_ps(d, _mm256_cvtepi32_ps(q_lo_lo));
            let dq1 = _mm256_mul_ps(d, _mm256_cvtepi32_ps(q_hi_lo));
            let dq2 = _mm256_mul_ps(d, _mm256_cvtepi32_ps(q_lo_hi));
            let dq3 = _mm256_mul_ps(d, _mm256_cvtepi32_ps(q_hi_hi));

            // x layout per block: [0..8 ↔ lo low nibbles (qs[0..8] & 0xF)],
            // [8..16 ↔ hi low nibbles (qs[8..16] & 0xF)],
            // [16..24 ↔ lo high nibbles (qs[0..8] >> 4)],
            // [24..32 ↔ hi high nibbles (qs[8..16] >> 4)].
            let xptr = x.as_ptr().add(b * QK);
            let xv0 = _mm256_loadu_ps(xptr);
            let xv1 = _mm256_loadu_ps(xptr.add(8));
            let xv2 = _mm256_loadu_ps(xptr.add(16));
            let xv3 = _mm256_loadu_ps(xptr.add(24));

            acc0 = _mm256_fmadd_ps(dq0, xv0, acc0);
            acc1 = _mm256_fmadd_ps(dq1, xv1, acc1);
            acc2 = _mm256_fmadd_ps(dq2, xv2, acc2);
            acc3 = _mm256_fmadd_ps(dq3, xv3, acc3);
        }
        let s01 = _mm256_add_ps(acc0, acc1);
        let s23 = _mm256_add_ps(acc2, acc3);
        let total = _mm256_add_ps(s01, s23);
        let lo = _mm256_castps256_ps128(total);
        let hi = _mm256_extractf128_ps(total, 1);
        let sum128 = _mm_add_ps(lo, hi);
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0b01));
        out[i] = _mm_cvtss_f32(sum32);
    }
}

/// Decode Q4_0 rows by id and write to `out` as F32. Trivial wrapper
/// over [`rustllama_gguf::dequant::dequant_q4_0`] applied row-by-row.
pub fn embed_lookup_q4_0(table_bytes: &[u8], ids: &[i32], out: &mut [f32], d: usize) {
    const BLOCK_BYTES: usize = 18;
    const QK: usize = 32;
    assert_eq!(d % QK, 0, "Q4_0 embed lookup requires d % 32 == 0");
    let blocks_per_row = d / QK;
    let row_bytes = blocks_per_row * BLOCK_BYTES;
    for (i, &id) in ids.iter().enumerate() {
        let row_start = (id as usize) * row_bytes;
        let dst = &mut out[i * d..(i + 1) * d];
        let row = &table_bytes[row_start..row_start + row_bytes];
        rustllama_gguf::dequant::dequant_q4_0(row, dst);
    }
}

/// Q5_0 weight layout (32 weights per 22-byte block: f16 scale + 32 5-bit
/// signed quants split as 4 low bits in `qs` and 1 high bit in `qh`).
/// Decode-and-FMA fused so weights never touch an intermediate F16/F32
/// buffer. Halves memory bandwidth vs F16 storage (≈65% reduction).
///
/// Assumes `k % 32 == 0` (matches real-world Linear weights).
pub fn matvec_q5_0_w_f32_a(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    const BLOCK_BYTES: usize = 22;
    const QK: usize = 32;
    debug_assert_eq!(k % QK, 0, "Q5_0 matvec requires k % 32 == 0");
    let blocks_per_row = k / QK;
    debug_assert_eq!(w_bytes.len(), m * blocks_per_row * BLOCK_BYTES);
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(out.len(), m);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: runtime feature detection.
            unsafe { matvec_q5_0_w_f32_a_avx512(w_bytes, x, out, m, k) };
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime feature detection above.
            unsafe { matvec_q5_0_w_f32_a_avx2(w_bytes, x, out, m, k) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64.
        unsafe { matvec_q5_0_w_f32_a_neon(w_bytes, x, out, m, k) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    matvec_q5_0_w_f32_a_scalar(w_bytes, x, out, m, k);
}

/// Expand the two low bytes of a packed bit field into 16 `u8` lanes,
/// each `val` where the corresponding bit is set and 0 otherwise — the
/// NEON building block for the Q5_0/Q5_1 "5th bit" (bit `j` of the 32-bit
/// `qh` field selects lane `j`). `lo_byte` drives lanes 0..8, `hi_byte`
/// lanes 8..16. `vtst` turns a bit-test into a 0xFF/0x00 lane mask, which
/// we AND with `val` to get `val`-or-0 (the AVX2/AVX-512 paths do the same
/// with a per-lane variable shift; NEON's bit-expand trick is cheaper than
/// a 16-lane `vshl`).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn q5_bit_expand_neon(lo_byte: u8, hi_byte: u8, val: u8) -> std::arch::aarch64::uint8x16_t {
    use std::arch::aarch64::*;
    const BITS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];
    let bitmask = vld1_u8(BITS.as_ptr());
    let valv = vdup_n_u8(val);
    let lo = vand_u8(vtst_u8(vdup_n_u8(lo_byte), bitmask), valv);
    let hi = vand_u8(vtst_u8(vdup_n_u8(hi_byte), bitmask), valv);
    vcombine_u8(lo, hi)
}

/// AArch64 NEON Q5_0 matvec. 5-bit signed codes: 4 low bits in `qs`, the
/// 5th bit in the per-block 32-bit `qh` field. For output `j` (low
/// nibble) the 5th bit is `qh` bit `j`; for output `j+16` (high nibble)
/// it is `qh` bit `j+16`. We OR the expanded 5th bit (0 or 0x10) into the
/// nibble, widen unsigned, recenter by -16, scale by `d`, FMA. NEON
/// baseline; same tolerance contract as the AVX2 path.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn matvec_q5_0_w_f32_a_neon(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use std::arch::aarch64::*;
    const BLOCK_BYTES: usize = 22;
    const QK: usize = 32;
    let blocks_per_row = k / QK;
    let mask_lo = vdupq_n_u8(0x0F);
    let sixteen = vdupq_n_f32(16.0);
    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = [vdupq_n_f32(0.0); 4];
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let dv = vdupq_n_f32(f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32());
            let qh = u32::from_le_bytes([
                w_bytes[off + 2],
                w_bytes[off + 3],
                w_bytes[off + 4],
                w_bytes[off + 5],
            ]);
            let qb16 = vld1q_u8(w_bytes.as_ptr().add(off + 6));
            let low_nib = vandq_u8(qb16, mask_lo);
            let high_nib = vshrq_n_u8::<4>(qb16);
            // 5th bit: outputs 0..15 ← qh bits 0..15; outputs 16..31 ← qh bits 16..31.
            let hb_lo = q5_bit_expand_neon((qh & 0xFF) as u8, ((qh >> 8) & 0xFF) as u8, 0x10);
            let hb_hi =
                q5_bit_expand_neon(((qh >> 16) & 0xFF) as u8, ((qh >> 24) & 0xFF) as u8, 0x10);
            let lo_f = u8x16_to_f32x4x4(vorrq_u8(low_nib, hb_lo)); // 0..31
            let hi_f = u8x16_to_f32x4x4(vorrq_u8(high_nib, hb_hi));
            let xlo = x.as_ptr().add(b * QK);
            let xhi = xlo.add(16);
            for c in 0..4 {
                let xl = vld1q_f32(xlo.add(c * 4));
                let xh = vld1q_f32(xhi.add(c * 4));
                let coef_lo = vmulq_f32(dv, vsubq_f32(lo_f[c], sixteen));
                let coef_hi = vmulq_f32(dv, vsubq_f32(hi_f[c], sixteen));
                acc[c] = vfmaq_f32(acc[c], coef_lo, xl);
                acc[c] = vfmaq_f32(acc[c], coef_hi, xh);
            }
        }
        let s = vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3]));
        out[i] = vaddvq_f32(s);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f")]
unsafe fn matvec_q5_0_w_f32_a_avx512(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 22;
    const QK: usize = 32;
    let blocks_per_row = k / QK;

    // Per-lane shifts so a broadcast of the 32-bit `qh` field, shifted
    // by each lane's count, lets us extract the 5th-bit table for
    // outputs 0..16 (low half) and 16..32 (high half) in one shot per
    // half.
    let shifts_0_15 = _mm512_setr_epi32(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    );
    let shifts_16_31 = _mm512_setr_epi32(
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    );
    let one = _mm512_set1_epi32(1);
    let sixteen = _mm512_set1_epi32(16);
    let mask_low_nibble = _mm512_set1_epi32(0x0F);

    #[inline(always)]
    unsafe fn high_bits(qh_vec: __m512i, shifts: __m512i, one: __m512i) -> __m512i {
        let shifted = _mm512_srlv_epi32(qh_vec, shifts);
        let bit = _mm512_and_si512(shifted, one);
        _mm512_slli_epi32(bit, 4)
    }

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();

        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d_bits = u16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]);
            let d_f32 = f16::from_bits(d_bits).to_f32();
            let d = _mm512_set1_ps(d_f32);

            let qh_u32 = u32::from_le_bytes([
                w_bytes[off + 2],
                w_bytes[off + 3],
                w_bytes[off + 4],
                w_bytes[off + 5],
            ]);
            let qh_vec = _mm512_set1_epi32(qh_u32 as i32);

            // 16 qs bytes split into 16 low nibbles + 16 high nibbles,
            // each becoming a ZMM i32x16 lane.
            let qs_ptr = w_bytes.as_ptr().add(off + 6);
            let qs_raw = _mm_loadu_si128(qs_ptr as *const __m128i);
            let qs_lane = _mm512_cvtepu8_epi32(qs_raw);
            let lo_nibbles = _mm512_and_si512(qs_lane, mask_low_nibble);
            let hi_nibbles = _mm512_and_si512(_mm512_srli_epi32(qs_lane, 4), mask_low_nibble);

            // 5-bit signed: (nibble | high_bit<<4) - 16.
            let q_lo = _mm512_sub_epi32(
                _mm512_or_si512(lo_nibbles, high_bits(qh_vec, shifts_0_15, one)),
                sixteen,
            );
            let q_hi = _mm512_sub_epi32(
                _mm512_or_si512(hi_nibbles, high_bits(qh_vec, shifts_16_31, one)),
                sixteen,
            );

            let dq_lo = _mm512_mul_ps(d, _mm512_cvtepi32_ps(q_lo));
            let dq_hi = _mm512_mul_ps(d, _mm512_cvtepi32_ps(q_hi));

            let xptr = x.as_ptr().add(b * QK);
            let xv0 = _mm512_loadu_ps(xptr);
            let xv1 = _mm512_loadu_ps(xptr.add(16));

            acc0 = _mm512_fmadd_ps(dq_lo, xv0, acc0);
            acc1 = _mm512_fmadd_ps(dq_hi, xv1, acc1);
        }

        out[i] = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
    }
}

fn matvec_q5_0_w_f32_a_scalar(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    const BLOCK_BYTES: usize = 22;
    const QK: usize = 32;
    let blocks_per_row = k / QK;
    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d = f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32();
            let qh = u32::from_le_bytes([
                w_bytes[off + 2],
                w_bytes[off + 3],
                w_bytes[off + 4],
                w_bytes[off + 5],
            ]);
            let qs = &w_bytes[off + 6..off + 22];
            let x_chunk = &x[b * QK..(b + 1) * QK];
            for j in 0..16 {
                let xh_0 = ((qh >> j) << 4) & 0x10;
                let xh_1 = (qh >> (j + 12)) & 0x10;
                let x0 = ((qs[j] & 0x0F) as i32 | xh_0 as i32) - 16;
                let x1 = ((qs[j] >> 4) as i32 | xh_1 as i32) - 16;
                acc += d * x0 as f32 * x_chunk[j];
                acc += d * x1 as f32 * x_chunk[j + 16];
            }
        }
        out[i] = acc;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn matvec_q5_0_w_f32_a_avx2(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 22;
    const QK: usize = 32;
    let blocks_per_row = k / QK;

    // Per-lane "shift amounts" so SIMD-extracting one bit per lane from
    // a broadcast u32 gives us the 5th-bit table for outputs 0..7,
    // outputs 8..15, outputs 16..23, outputs 24..31.
    let shifts_0_7 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let shifts_8_15 = _mm256_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15);
    let shifts_16_23 = _mm256_setr_epi32(16, 17, 18, 19, 20, 21, 22, 23);
    let shifts_24_31 = _mm256_setr_epi32(24, 25, 26, 27, 28, 29, 30, 31);
    let one = _mm256_set1_epi32(1);
    let sixteen = _mm256_set1_epi32(16);
    let mask_low_nibble = _mm256_set1_epi32(0x0F);

    // Helper: given qh broadcast and the per-lane shift vector, produce
    // a vector of {0, 16} per lane representing the 5th-bit value.
    #[inline(always)]
    unsafe fn high_bits(qh_vec: __m256i, shifts: __m256i, one: __m256i) -> __m256i {
        let shifted = _mm256_srlv_epi32(qh_vec, shifts);
        let bit = _mm256_and_si256(shifted, one);
        _mm256_slli_epi32(bit, 4)
    }

    // Unpack 16 bytes of qs into two 256-bit vectors of i32: low nibbles
    // (16 lanes split 8/8) and high nibbles (same).
    #[inline(always)]
    unsafe fn unpack_qs(
        qs_ptr: *const u8,
        mask: __m256i,
    ) -> (__m256i, __m256i, __m256i, __m256i) {
        // qs_lo: bytes 0..8, qs_hi: bytes 8..16
        let qs_lo8 = _mm_loadl_epi64(qs_ptr as *const __m128i);
        let qs_hi8 = _mm_loadl_epi64(qs_ptr.add(8) as *const __m128i);
        let qs_lo32 = _mm256_cvtepu8_epi32(qs_lo8); // bytes 0..7 as u32
        let qs_hi32 = _mm256_cvtepu8_epi32(qs_hi8); // bytes 8..15 as u32

        let low_n_07 = _mm256_and_si256(qs_lo32, mask); // low nibble bytes 0..7
        let low_n_815 = _mm256_and_si256(qs_hi32, mask); // low nibble bytes 8..15
        let high_n_07 = _mm256_and_si256(_mm256_srli_epi32(qs_lo32, 4), mask); // high nibble bytes 0..7
        let high_n_815 = _mm256_and_si256(_mm256_srli_epi32(qs_hi32, 4), mask); // high nibble bytes 8..15
        (low_n_07, low_n_815, high_n_07, high_n_815)
    }

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;

            let d_bits = u16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]);
            let d_f32 = f16::from_bits(d_bits).to_f32();
            let d = _mm256_set1_ps(d_f32);

            let qh_u32 = u32::from_le_bytes([
                w_bytes[off + 2],
                w_bytes[off + 3],
                w_bytes[off + 4],
                w_bytes[off + 5],
            ]);
            let qh_vec = _mm256_set1_epi32(qh_u32 as i32);

            let (lo_07, lo_815, hi_07, hi_815) =
                unpack_qs(w_bytes.as_ptr().add(off + 6), mask_low_nibble);

            // Reconstruct 5-bit signed values: nibble | (high_bit << 4) - 16.
            // For Q5_0 block layout: output[j] = (qs[j].low | bit_j(qh)<<4) - 16
            //                         output[j+16] = (qs[j].high | bit_{j+16}(qh)<<4) - 16
            let q0_07 = _mm256_sub_epi32(
                _mm256_or_si256(lo_07, high_bits(qh_vec, shifts_0_7, one)),
                sixteen,
            );
            let q0_815 = _mm256_sub_epi32(
                _mm256_or_si256(lo_815, high_bits(qh_vec, shifts_8_15, one)),
                sixteen,
            );
            let q16_07 = _mm256_sub_epi32(
                _mm256_or_si256(hi_07, high_bits(qh_vec, shifts_16_23, one)),
                sixteen,
            );
            let q16_815 = _mm256_sub_epi32(
                _mm256_or_si256(hi_815, high_bits(qh_vec, shifts_24_31, one)),
                sixteen,
            );

            // Convert to f32 and multiply by d
            let dq_lo_07 = _mm256_mul_ps(d, _mm256_cvtepi32_ps(q0_07));
            let dq_lo_815 = _mm256_mul_ps(d, _mm256_cvtepi32_ps(q0_815));
            let dq_hi_07 = _mm256_mul_ps(d, _mm256_cvtepi32_ps(q16_07));
            let dq_hi_815 = _mm256_mul_ps(d, _mm256_cvtepi32_ps(q16_815));

            // Load x[b*32..(b+1)*32]
            let xptr = x.as_ptr().add(b * QK);
            let xv0 = _mm256_loadu_ps(xptr); // x[0..8]   → output[0..8]
            let xv1 = _mm256_loadu_ps(xptr.add(8)); // x[8..16]  → output[8..16]
            let xv2 = _mm256_loadu_ps(xptr.add(16)); // x[16..24] → output[16..24]
            let xv3 = _mm256_loadu_ps(xptr.add(24)); // x[24..32] → output[24..32]

            acc0 = _mm256_fmadd_ps(dq_lo_07, xv0, acc0);
            acc1 = _mm256_fmadd_ps(dq_lo_815, xv1, acc1);
            acc2 = _mm256_fmadd_ps(dq_hi_07, xv2, acc2);
            acc3 = _mm256_fmadd_ps(dq_hi_815, xv3, acc3);
        }

        let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        let mut sum128 = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        out[i] = _mm_cvtss_f32(sum128);
    }
}

/// Q4_1 weight matvec with on-the-fly dequant. 32 weights per 20-byte
/// block: f16 scale `d`, f16 min/offset `m`, then 16 bytes of unsigned
/// 4-bit values packed two-per-byte. Per-weight value is `d * q + m`
/// where `q` is the unsigned nibble in `[0, 15]`.
///
/// Per-iteration recipe is one extra fmadd vs Q4_0: instead of
/// `dq = d * q` we compute `dq = d * q + m` (so the final dot product
/// is `sum (d*q + m) * x` without ever materializing the `m * sum x`
/// term separately). FLOP count matches Q4_0 — just one more
/// broadcast load per block.
///
/// Dispatches AVX-512 → AVX2 → scalar.
///
/// Assumes `k % 32 == 0` (matches real-world Linear weights).
pub fn matvec_q4_1_w_f32_a(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m_rows: usize,
    k: usize,
) {
    const BLOCK_BYTES: usize = 20;
    const QK: usize = 32;
    debug_assert_eq!(k % QK, 0);
    let blocks_per_row = k / QK;
    debug_assert_eq!(w_bytes.len(), m_rows * blocks_per_row * BLOCK_BYTES);
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(out.len(), m_rows);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: runtime feature detection above.
            unsafe { matvec_q4_1_w_f32_a_avx512(w_bytes, x, out, m_rows, k) };
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime feature detection above.
            unsafe { matvec_q4_1_w_f32_a_avx2(w_bytes, x, out, m_rows, k) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64.
        unsafe { matvec_q4_1_w_f32_a_neon(w_bytes, x, out, m_rows, k) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    matvec_q4_1_w_f32_a_scalar(w_bytes, x, out, m_rows, k);
}

/// AArch64 NEON Q4_1 matvec (weight Q4_1, activation f32). Like Q4_0 but
/// the codes are unsigned (no `-8` recenter) and each block carries a
/// second f16 field `m` (the min/offset): the per-weight value is
/// `d*q + m`, folded with one `vfmaq` (`m + q*d`) before the activation
/// FMA. Low nibbles map to outputs 0..15 (x[0..16]), high nibbles to
/// 16..31 (x[16..32]). NEON baseline; same tolerance contract as AVX2.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn matvec_q4_1_w_f32_a_neon(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m_rows: usize,
    k: usize,
) {
    use std::arch::aarch64::*;
    const BLOCK_BYTES: usize = 20;
    const QK: usize = 32;
    let blocks_per_row = k / QK;
    let mask_lo = vdupq_n_u8(0x0F);
    for i in 0..m_rows {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = [vdupq_n_f32(0.0); 4];
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let dv = vdupq_n_f32(f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32());
            let mv = vdupq_n_f32(f16::from_le_bytes([w_bytes[off + 2], w_bytes[off + 3]]).to_f32());
            let qb16 = vld1q_u8(w_bytes.as_ptr().add(off + 4));
            let lo_f = u8x16_to_f32x4x4(vandq_u8(qb16, mask_lo));
            let hi_f = u8x16_to_f32x4x4(vshrq_n_u8::<4>(qb16));
            let xlo = x.as_ptr().add(b * QK);
            let xhi = xlo.add(16);
            for c in 0..4 {
                let xl = vld1q_f32(xlo.add(c * 4));
                let xh = vld1q_f32(xhi.add(c * 4));
                // coef = d*q + m (fused: m + q*d)
                let coef_lo = vfmaq_f32(mv, lo_f[c], dv);
                let coef_hi = vfmaq_f32(mv, hi_f[c], dv);
                acc[c] = vfmaq_f32(acc[c], coef_lo, xl);
                acc[c] = vfmaq_f32(acc[c], coef_hi, xh);
            }
        }
        let s = vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3]));
        out[i] = vaddvq_f32(s);
    }
}

fn matvec_q4_1_w_f32_a_scalar(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m_rows: usize,
    k: usize,
) {
    const BLOCK_BYTES: usize = 20;
    const QK: usize = 32;
    let blocks_per_row = k / QK;
    for i in 0..m_rows {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d = f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32();
            let m = f16::from_le_bytes([w_bytes[off + 2], w_bytes[off + 3]]).to_f32();
            let qs = &w_bytes[off + 4..off + 4 + 16];
            let x_chunk = &x[b * QK..(b + 1) * QK];
            for j in 0..16 {
                let q0 = (qs[j] & 0x0F) as i32;
                let q1 = (qs[j] >> 4) as i32;
                acc += (d * q0 as f32 + m) * x_chunk[j];
                acc += (d * q1 as f32 + m) * x_chunk[j + 16];
            }
        }
        out[i] = acc;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f")]
unsafe fn matvec_q4_1_w_f32_a_avx512(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m_rows: usize,
    k: usize,
) {
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 20;
    const QK: usize = 32;
    let blocks_per_row = k / QK;
    let mask_low_nibble = _mm512_set1_epi32(0x0F);

    for i in 0..m_rows {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();

        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d_bits = u16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]);
            let m_bits = u16::from_le_bytes([w_bytes[off + 2], w_bytes[off + 3]]);
            let d = _mm512_set1_ps(f16::from_bits(d_bits).to_f32());
            let m_vec = _mm512_set1_ps(f16::from_bits(m_bits).to_f32());

            let qs_ptr = w_bytes.as_ptr().add(off + 4);
            let qs_raw = _mm_loadu_si128(qs_ptr as *const __m128i);
            let qs_lane = _mm512_cvtepu8_epi32(qs_raw);
            let lo_nibbles = _mm512_and_si512(qs_lane, mask_low_nibble);
            let hi_nibbles =
                _mm512_and_si512(_mm512_srli_epi32(qs_lane, 4), mask_low_nibble);

            // dq = d * q + m (one fmadd; q is unsigned, no subtract).
            let dq_lo = _mm512_fmadd_ps(d, _mm512_cvtepi32_ps(lo_nibbles), m_vec);
            let dq_hi = _mm512_fmadd_ps(d, _mm512_cvtepi32_ps(hi_nibbles), m_vec);

            let xptr = x.as_ptr().add(b * QK);
            let xv0 = _mm512_loadu_ps(xptr);
            let xv1 = _mm512_loadu_ps(xptr.add(16));

            acc0 = _mm512_fmadd_ps(dq_lo, xv0, acc0);
            acc1 = _mm512_fmadd_ps(dq_hi, xv1, acc1);
        }
        out[i] = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn matvec_q4_1_w_f32_a_avx2(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m_rows: usize,
    k: usize,
) {
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 20;
    const QK: usize = 32;
    let blocks_per_row = k / QK;
    let mask_low_nibble = _mm256_set1_epi32(0x0F);

    for i in 0..m_rows {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d_bits = u16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]);
            let m_bits = u16::from_le_bytes([w_bytes[off + 2], w_bytes[off + 3]]);
            let d = _mm256_set1_ps(f16::from_bits(d_bits).to_f32());
            let m_vec = _mm256_set1_ps(f16::from_bits(m_bits).to_f32());

            // Same byte-layout as Q4_0: 16 qs bytes split into two 8-byte
            // halves; each half widened to 8 i32 lanes; split each into
            // low/high nibbles → 4 YMM lanes (32 weights) per block.
            let qs_ptr = w_bytes.as_ptr().add(off + 4);
            let lo_half = _mm_loadl_epi64(qs_ptr as *const __m128i);
            let hi_half = _mm_loadl_epi64(qs_ptr.add(8) as *const __m128i);
            let lo_widened = _mm256_cvtepu8_epi32(lo_half);
            let hi_widened = _mm256_cvtepu8_epi32(hi_half);

            let lo_lo = _mm256_and_si256(lo_widened, mask_low_nibble);
            let lo_hi =
                _mm256_and_si256(_mm256_srli_epi32(lo_widened, 4), mask_low_nibble);
            let hi_lo = _mm256_and_si256(hi_widened, mask_low_nibble);
            let hi_hi =
                _mm256_and_si256(_mm256_srli_epi32(hi_widened, 4), mask_low_nibble);

            let dq0 = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(lo_lo), m_vec);
            let dq1 = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(hi_lo), m_vec);
            let dq2 = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(lo_hi), m_vec);
            let dq3 = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(hi_hi), m_vec);

            let xptr = x.as_ptr().add(b * QK);
            let xv0 = _mm256_loadu_ps(xptr);
            let xv1 = _mm256_loadu_ps(xptr.add(8));
            let xv2 = _mm256_loadu_ps(xptr.add(16));
            let xv3 = _mm256_loadu_ps(xptr.add(24));

            acc0 = _mm256_fmadd_ps(dq0, xv0, acc0);
            acc1 = _mm256_fmadd_ps(dq1, xv1, acc1);
            acc2 = _mm256_fmadd_ps(dq2, xv2, acc2);
            acc3 = _mm256_fmadd_ps(dq3, xv3, acc3);
        }
        let s01 = _mm256_add_ps(acc0, acc1);
        let s23 = _mm256_add_ps(acc2, acc3);
        let total = _mm256_add_ps(s01, s23);
        let lo = _mm256_castps256_ps128(total);
        let hi = _mm256_extractf128_ps(total, 1);
        let sum128 = _mm_add_ps(lo, hi);
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0b01));
        out[i] = _mm_cvtss_f32(sum32);
    }
}

/// Decode Q4_1 rows by id and write to `out` as F32. Thin wrapper over
/// [`rustllama_gguf::dequant::dequant_q4_1`] applied row-by-row.
pub fn embed_lookup_q4_1(table_bytes: &[u8], ids: &[i32], out: &mut [f32], d: usize) {
    const BLOCK_BYTES: usize = 20;
    const QK: usize = 32;
    assert_eq!(d % QK, 0, "Q4_1 embed lookup requires d % 32 == 0");
    let blocks_per_row = d / QK;
    let row_bytes = blocks_per_row * BLOCK_BYTES;
    for (i, &id) in ids.iter().enumerate() {
        let row_start = (id as usize) * row_bytes;
        let dst = &mut out[i * d..(i + 1) * d];
        let row = &table_bytes[row_start..row_start + row_bytes];
        rustllama_gguf::dequant::dequant_q4_1(row, dst);
    }
}

/// Q5_1 weight matvec with on-the-fly dequant. 32 weights per 24-byte
/// block: f16 scale `d`, f16 min/offset `m`, packed-u32 `qh` (5th bits),
/// then 16 bytes of low 4 bits packed two-per-byte. Per-weight value is
/// `d * q + m` where `q = (low4 | bit5 << 4)` is the unsigned 5-bit
/// value in `[0, 31]`.
///
/// Dispatches AVX-512 → AVX2 → scalar.
///
/// Assumes `k % 32 == 0`.
pub fn matvec_q5_1_w_f32_a(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m_rows: usize,
    k: usize,
) {
    const BLOCK_BYTES: usize = 24;
    const QK: usize = 32;
    debug_assert_eq!(k % QK, 0, "Q5_1 matvec requires k % 32 == 0");
    let blocks_per_row = k / QK;
    debug_assert_eq!(w_bytes.len(), m_rows * blocks_per_row * BLOCK_BYTES);
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(out.len(), m_rows);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: runtime feature detection above.
            unsafe { matvec_q5_1_w_f32_a_avx512(w_bytes, x, out, m_rows, k) };
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime feature detection above.
            unsafe { matvec_q5_1_w_f32_a_avx2(w_bytes, x, out, m_rows, k) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64.
        unsafe { matvec_q5_1_w_f32_a_neon(w_bytes, x, out, m_rows, k) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    matvec_q5_1_w_f32_a_scalar(w_bytes, x, out, m_rows, k);
}

/// AArch64 NEON Q5_1 matvec. Like Q5_0 but unsigned 5-bit codes (no -16
/// recenter) plus a per-block f16 min `m`: value = `d*q + m`. The 5th bit
/// comes from the 32-bit `qh` field exactly as in Q5_0 (output `j` low ←
/// `qh` bit `j`, output `j+16` high ← bit `j+16`). NEON baseline; same
/// tolerance contract as the AVX2 path.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn matvec_q5_1_w_f32_a_neon(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m_rows: usize,
    k: usize,
) {
    use std::arch::aarch64::*;
    const BLOCK_BYTES: usize = 24;
    const QK: usize = 32;
    let blocks_per_row = k / QK;
    let mask_lo = vdupq_n_u8(0x0F);
    for i in 0..m_rows {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = [vdupq_n_f32(0.0); 4];
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let dv = vdupq_n_f32(f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32());
            let mv = vdupq_n_f32(f16::from_le_bytes([w_bytes[off + 2], w_bytes[off + 3]]).to_f32());
            let qh = u32::from_le_bytes([
                w_bytes[off + 4],
                w_bytes[off + 5],
                w_bytes[off + 6],
                w_bytes[off + 7],
            ]);
            let qb16 = vld1q_u8(w_bytes.as_ptr().add(off + 8));
            let low_nib = vandq_u8(qb16, mask_lo);
            let high_nib = vshrq_n_u8::<4>(qb16);
            let hb_lo = q5_bit_expand_neon((qh & 0xFF) as u8, ((qh >> 8) & 0xFF) as u8, 0x10);
            let hb_hi =
                q5_bit_expand_neon(((qh >> 16) & 0xFF) as u8, ((qh >> 24) & 0xFF) as u8, 0x10);
            let lo_f = u8x16_to_f32x4x4(vorrq_u8(low_nib, hb_lo)); // 0..31
            let hi_f = u8x16_to_f32x4x4(vorrq_u8(high_nib, hb_hi));
            let xlo = x.as_ptr().add(b * QK);
            let xhi = xlo.add(16);
            for c in 0..4 {
                let xl = vld1q_f32(xlo.add(c * 4));
                let xh = vld1q_f32(xhi.add(c * 4));
                // coef = d*q + m (fused: m + q*d)
                let coef_lo = vfmaq_f32(mv, lo_f[c], dv);
                let coef_hi = vfmaq_f32(mv, hi_f[c], dv);
                acc[c] = vfmaq_f32(acc[c], coef_lo, xl);
                acc[c] = vfmaq_f32(acc[c], coef_hi, xh);
            }
        }
        let s = vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3]));
        out[i] = vaddvq_f32(s);
    }
}

fn matvec_q5_1_w_f32_a_scalar(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m_rows: usize,
    k: usize,
) {
    const BLOCK_BYTES: usize = 24;
    const QK: usize = 32;
    let blocks_per_row = k / QK;
    for i in 0..m_rows {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d = f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32();
            let m = f16::from_le_bytes([w_bytes[off + 2], w_bytes[off + 3]]).to_f32();
            let qh = u32::from_le_bytes([
                w_bytes[off + 4],
                w_bytes[off + 5],
                w_bytes[off + 6],
                w_bytes[off + 7],
            ]);
            let qs = &w_bytes[off + 8..off + 24];
            let x_chunk = &x[b * QK..(b + 1) * QK];
            for j in 0..16 {
                let xh_0 = ((qh >> j) << 4) & 0x10;
                let xh_1 = (qh >> (j + 12)) & 0x10;
                let q0 = ((qs[j] & 0x0F) as i32) | (xh_0 as i32);
                let q1 = ((qs[j] >> 4) as i32) | (xh_1 as i32);
                acc += (d * q0 as f32 + m) * x_chunk[j];
                acc += (d * q1 as f32 + m) * x_chunk[j + 16];
            }
        }
        out[i] = acc;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f")]
unsafe fn matvec_q5_1_w_f32_a_avx512(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m_rows: usize,
    k: usize,
) {
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 24;
    const QK: usize = 32;
    let blocks_per_row = k / QK;

    // Per-lane shift amounts so a broadcast of the 32-bit `qh` field,
    // shifted by each lane's count, gives us the 5th-bit table for
    // outputs 0..16 (low half) and 16..32 (high half) in one shot per
    // half. Same recipe as the Q5_0 AVX-512 path.
    let shifts_0_15 = _mm512_setr_epi32(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    );
    let shifts_16_31 = _mm512_setr_epi32(
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    );
    let one = _mm512_set1_epi32(1);
    let mask_low_nibble = _mm512_set1_epi32(0x0F);

    #[inline(always)]
    unsafe fn high_bits(qh_vec: __m512i, shifts: __m512i, one: __m512i) -> __m512i {
        use std::arch::x86_64::*;
        let shifted = _mm512_srlv_epi32(qh_vec, shifts);
        let bit = _mm512_and_si512(shifted, one);
        _mm512_slli_epi32(bit, 4)
    }

    for i in 0..m_rows {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();

        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d_bits = u16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]);
            let m_bits = u16::from_le_bytes([w_bytes[off + 2], w_bytes[off + 3]]);
            let d = _mm512_set1_ps(f16::from_bits(d_bits).to_f32());
            let m_vec = _mm512_set1_ps(f16::from_bits(m_bits).to_f32());

            let qh_u32 = u32::from_le_bytes([
                w_bytes[off + 4],
                w_bytes[off + 5],
                w_bytes[off + 6],
                w_bytes[off + 7],
            ]);
            let qh_vec = _mm512_set1_epi32(qh_u32 as i32);

            let qs_ptr = w_bytes.as_ptr().add(off + 8);
            let qs_raw = _mm_loadu_si128(qs_ptr as *const __m128i);
            let qs_lane = _mm512_cvtepu8_epi32(qs_raw);
            let lo_nibbles = _mm512_and_si512(qs_lane, mask_low_nibble);
            let hi_nibbles =
                _mm512_and_si512(_mm512_srli_epi32(qs_lane, 4), mask_low_nibble);

            // Unsigned 5-bit value: nibble | (high_bit << 4). No subtract.
            let q_lo = _mm512_or_si512(lo_nibbles, high_bits(qh_vec, shifts_0_15, one));
            let q_hi = _mm512_or_si512(hi_nibbles, high_bits(qh_vec, shifts_16_31, one));

            let dq_lo = _mm512_fmadd_ps(d, _mm512_cvtepi32_ps(q_lo), m_vec);
            let dq_hi = _mm512_fmadd_ps(d, _mm512_cvtepi32_ps(q_hi), m_vec);

            let xptr = x.as_ptr().add(b * QK);
            let xv0 = _mm512_loadu_ps(xptr);
            let xv1 = _mm512_loadu_ps(xptr.add(16));

            acc0 = _mm512_fmadd_ps(dq_lo, xv0, acc0);
            acc1 = _mm512_fmadd_ps(dq_hi, xv1, acc1);
        }
        out[i] = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn matvec_q5_1_w_f32_a_avx2(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m_rows: usize,
    k: usize,
) {
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 24;
    const QK: usize = 32;
    let blocks_per_row = k / QK;

    let shifts_0_7 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let shifts_8_15 = _mm256_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15);
    let shifts_16_23 = _mm256_setr_epi32(16, 17, 18, 19, 20, 21, 22, 23);
    let shifts_24_31 = _mm256_setr_epi32(24, 25, 26, 27, 28, 29, 30, 31);
    let one = _mm256_set1_epi32(1);
    let mask_low_nibble = _mm256_set1_epi32(0x0F);

    #[inline(always)]
    unsafe fn high_bits(qh_vec: __m256i, shifts: __m256i, one: __m256i) -> __m256i {
        use std::arch::x86_64::*;
        let shifted = _mm256_srlv_epi32(qh_vec, shifts);
        let bit = _mm256_and_si256(shifted, one);
        _mm256_slli_epi32(bit, 4)
    }

    #[inline(always)]
    unsafe fn unpack_qs(
        qs_ptr: *const u8,
        mask: __m256i,
    ) -> (__m256i, __m256i, __m256i, __m256i) {
        use std::arch::x86_64::*;
        let qs_lo8 = _mm_loadl_epi64(qs_ptr as *const __m128i);
        let qs_hi8 = _mm_loadl_epi64(qs_ptr.add(8) as *const __m128i);
        let qs_lo32 = _mm256_cvtepu8_epi32(qs_lo8);
        let qs_hi32 = _mm256_cvtepu8_epi32(qs_hi8);
        let low_n_07 = _mm256_and_si256(qs_lo32, mask);
        let low_n_815 = _mm256_and_si256(qs_hi32, mask);
        let high_n_07 = _mm256_and_si256(_mm256_srli_epi32(qs_lo32, 4), mask);
        let high_n_815 = _mm256_and_si256(_mm256_srli_epi32(qs_hi32, 4), mask);
        (low_n_07, low_n_815, high_n_07, high_n_815)
    }

    for i in 0..m_rows {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d_bits = u16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]);
            let m_bits = u16::from_le_bytes([w_bytes[off + 2], w_bytes[off + 3]]);
            let d = _mm256_set1_ps(f16::from_bits(d_bits).to_f32());
            let m_vec = _mm256_set1_ps(f16::from_bits(m_bits).to_f32());

            let qh_u32 = u32::from_le_bytes([
                w_bytes[off + 4],
                w_bytes[off + 5],
                w_bytes[off + 6],
                w_bytes[off + 7],
            ]);
            let qh_vec = _mm256_set1_epi32(qh_u32 as i32);

            let (lo_07, lo_815, hi_07, hi_815) =
                unpack_qs(w_bytes.as_ptr().add(off + 8), mask_low_nibble);

            // Unsigned 5-bit value: nibble | (high_bit << 4). Output index
            // layout matches Q5_0: lo nibble lanes → out[0..16], hi nibble
            // lanes → out[16..32].
            let q0_07 = _mm256_or_si256(lo_07, high_bits(qh_vec, shifts_0_7, one));
            let q0_815 = _mm256_or_si256(lo_815, high_bits(qh_vec, shifts_8_15, one));
            let q16_07 = _mm256_or_si256(hi_07, high_bits(qh_vec, shifts_16_23, one));
            let q16_815 = _mm256_or_si256(hi_815, high_bits(qh_vec, shifts_24_31, one));

            let dq_lo_07 = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(q0_07), m_vec);
            let dq_lo_815 = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(q0_815), m_vec);
            let dq_hi_07 = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(q16_07), m_vec);
            let dq_hi_815 = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(q16_815), m_vec);

            let xptr = x.as_ptr().add(b * QK);
            let xv0 = _mm256_loadu_ps(xptr);
            let xv1 = _mm256_loadu_ps(xptr.add(8));
            let xv2 = _mm256_loadu_ps(xptr.add(16));
            let xv3 = _mm256_loadu_ps(xptr.add(24));

            acc0 = _mm256_fmadd_ps(dq_lo_07, xv0, acc0);
            acc1 = _mm256_fmadd_ps(dq_lo_815, xv1, acc1);
            acc2 = _mm256_fmadd_ps(dq_hi_07, xv2, acc2);
            acc3 = _mm256_fmadd_ps(dq_hi_815, xv3, acc3);
        }

        let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        let mut sum128 =
            _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        out[i] = _mm_cvtss_f32(sum128);
    }
}

/// Decode Q5_1 rows by id and write to `out` as F32. Thin wrapper over
/// [`rustllama_gguf::dequant::dequant_q5_1`] applied row-by-row.
pub fn embed_lookup_q5_1(table_bytes: &[u8], ids: &[i32], out: &mut [f32], d: usize) {
    const BLOCK_BYTES: usize = 24;
    const QK: usize = 32;
    assert_eq!(d % QK, 0, "Q5_1 embed lookup requires d % 32 == 0");
    let blocks_per_row = d / QK;
    let row_bytes = blocks_per_row * BLOCK_BYTES;
    for (i, &id) in ids.iter().enumerate() {
        let row_start = (id as usize) * row_bytes;
        let dst = &mut out[i * d..(i + 1) * d];
        let row = &table_bytes[row_start..row_start + row_bytes];
        rustllama_gguf::dequant::dequant_q5_1(row, dst);
    }
}

/// Decode a single Q8_0 row by id and write it into `out` as F32.
pub fn embed_lookup_q8_0(table_bytes: &[u8], ids: &[i32], out: &mut [f32], d: usize) {
    const QK: usize = 32;
    assert_eq!(d % QK, 0, "Q8_0 embed lookup requires d % 32 == 0");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: runtime feature detection.
            unsafe { embed_lookup_q8_0_avx512(table_bytes, ids, out, d) };
            return;
        }
        if is_x86_feature_detected!("avx2") {
            // SAFETY: runtime feature detection.
            unsafe { embed_lookup_q8_0_avx2(table_bytes, ids, out, d) };
            return;
        }
    }
    embed_lookup_q8_0_scalar(table_bytes, ids, out, d);
}

fn embed_lookup_q8_0_scalar(table_bytes: &[u8], ids: &[i32], out: &mut [f32], d: usize) {
    const BLOCK_BYTES: usize = 34;
    const QK: usize = 32;
    let blocks_per_row = d / QK;
    let row_bytes = blocks_per_row * BLOCK_BYTES;
    for (i, &id) in ids.iter().enumerate() {
        let row_start = (id as usize) * row_bytes;
        let dst = &mut out[i * d..(i + 1) * d];
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let scale =
                f16::from_le_bytes([table_bytes[off], table_bytes[off + 1]]).to_f32();
            let qs = &table_bytes[off + 2..off + 2 + QK];
            let dst_chunk = &mut dst[b * QK..(b + 1) * QK];
            for (d_out, q) in dst_chunk.iter_mut().zip(qs.iter()) {
                *d_out = scale * (*q as i8 as f32);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2")]
unsafe fn embed_lookup_q8_0_avx2(table_bytes: &[u8], ids: &[i32], out: &mut [f32], d: usize) {
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 34;
    const QK: usize = 32;
    let blocks_per_row = d / QK;
    let row_bytes = blocks_per_row * BLOCK_BYTES;
    for (i, &id) in ids.iter().enumerate() {
        let row_start = (id as usize) * row_bytes;
        let dst = &mut out[i * d..(i + 1) * d];
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d_bits = u16::from_le_bytes([table_bytes[off], table_bytes[off + 1]]);
            let scale = _mm256_set1_ps(f16::from_bits(d_bits).to_f32());
            let qs_ptr = table_bytes.as_ptr().add(off + 2) as *const i8;
            // 32 i8 quants → 4 × 8-lane f32 stores.
            let q0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                qs_ptr as *const __m128i,
            )));
            let q1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                qs_ptr.add(8) as *const __m128i,
            )));
            let q2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                qs_ptr.add(16) as *const __m128i,
            )));
            let q3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                qs_ptr.add(24) as *const __m128i,
            )));
            let dst_ptr = dst.as_mut_ptr().add(b * QK);
            _mm256_storeu_ps(dst_ptr, _mm256_mul_ps(scale, q0));
            _mm256_storeu_ps(dst_ptr.add(8), _mm256_mul_ps(scale, q1));
            _mm256_storeu_ps(dst_ptr.add(16), _mm256_mul_ps(scale, q2));
            _mm256_storeu_ps(dst_ptr.add(24), _mm256_mul_ps(scale, q3));
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f")]
unsafe fn embed_lookup_q8_0_avx512(table_bytes: &[u8], ids: &[i32], out: &mut [f32], d: usize) {
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 34;
    const QK: usize = 32;
    let blocks_per_row = d / QK;
    let row_bytes = blocks_per_row * BLOCK_BYTES;
    for (i, &id) in ids.iter().enumerate() {
        let row_start = (id as usize) * row_bytes;
        let dst = &mut out[i * d..(i + 1) * d];
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d_bits = u16::from_le_bytes([table_bytes[off], table_bytes[off + 1]]);
            let scale = _mm512_set1_ps(f16::from_bits(d_bits).to_f32());
            let qs_ptr = table_bytes.as_ptr().add(off + 2) as *const i8;
            // 32 i8 quants → 2 × 16-lane f32 stores.
            let q0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(
                qs_ptr as *const __m128i,
            )));
            let q1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(
                qs_ptr.add(16) as *const __m128i,
            )));
            let dst_ptr = dst.as_mut_ptr().add(b * QK);
            _mm512_storeu_ps(dst_ptr, _mm512_mul_ps(scale, q0));
            _mm512_storeu_ps(dst_ptr.add(16), _mm512_mul_ps(scale, q1));
        }
    }
}

/// Q8_0 weight layout (32 weights per 34-byte block: f16 scale + 32 signed
/// int8 quants). Decode-and-FMA in one pass so the weights never touch the
/// F16 / F32 intermediate that the rest of the pipeline uses — halves
/// memory bandwidth vs F16 storage.
///
/// Assumes `k % 32 == 0`. Production LM heads and Linear weights satisfy
/// this in practice; if it doesn't, we'd need a tail handler.
pub fn matvec_q8_0_w_f32_a(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    const BLOCK_BYTES: usize = 34;
    const QK: usize = 32;
    debug_assert_eq!(k % QK, 0, "Q8_0 matvec requires k % 32 == 0");
    let blocks_per_row = k / QK;
    debug_assert_eq!(w_bytes.len(), m * blocks_per_row * BLOCK_BYTES);
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(out.len(), m);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: runtime feature detection.
            unsafe { matvec_q8_0_w_f32_a_avx512(w_bytes, x, out, m, k) };
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime feature detection above.
            unsafe { matvec_q8_0_w_f32_a_avx2(w_bytes, x, out, m, k) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64.
        unsafe { matvec_q8_0_w_f32_a_neon(w_bytes, x, out, m, k) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    matvec_q8_0_w_f32_a_scalar(w_bytes, x, out, m, k);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f")]
unsafe fn matvec_q8_0_w_f32_a_avx512(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 34;
    const QK: usize = 32;
    let blocks_per_row = k / QK;

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        // 32 lanes per block (2 × 16-wide ZMM) — Q8_0's block size is
        // 32 quants so a single block fills exactly two ZMM accumulators.
        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();

        for b in 0..blocks_per_row {
            let block_off = row_start + b * BLOCK_BYTES;
            let d_bits = u16::from_le_bytes([w_bytes[block_off], w_bytes[block_off + 1]]);
            let d_f32 = f16::from_bits(d_bits).to_f32();
            let d = _mm512_set1_ps(d_f32);

            // 32 i8 quants → two ZMM f32x16 lanes.
            // _mm_loadu_si128 loads 16 i8s; _mm512_cvtepi8_epi32 sign-
            // extends to i32x16; _mm512_cvtepi32_ps to f32x16.
            let qs_ptr = w_bytes.as_ptr().add(block_off + 2) as *const i8;
            let q0_raw = _mm_loadu_si128(qs_ptr as *const __m128i);
            let q1_raw = _mm_loadu_si128(qs_ptr.add(16) as *const __m128i);
            let q0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q0_raw));
            let q1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q1_raw));

            let xptr = x.as_ptr().add(b * QK);
            let xv0 = _mm512_loadu_ps(xptr);
            let xv1 = _mm512_loadu_ps(xptr.add(16));

            // Fold d into the quant value first, then FMA with x.
            let dq0 = _mm512_mul_ps(d, q0);
            let dq1 = _mm512_mul_ps(d, q1);
            acc0 = _mm512_fmadd_ps(dq0, xv0, acc0);
            acc1 = _mm512_fmadd_ps(dq1, xv1, acc1);
        }

        out[i] = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
    }
}

fn matvec_q8_0_w_f32_a_scalar(w_bytes: &[u8], x: &[f32], out: &mut [f32], m: usize, k: usize) {
    const BLOCK_BYTES: usize = 34;
    const QK: usize = 32;
    let blocks_per_row = k / QK;
    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            let block_off = row_start + b * BLOCK_BYTES;
            let d =
                f16::from_le_bytes([w_bytes[block_off], w_bytes[block_off + 1]]).to_f32();
            let qs = &w_bytes[block_off + 2..block_off + 2 + QK];
            let x_chunk = &x[b * QK..(b + 1) * QK];
            for (q, xv) in qs.iter().zip(x_chunk.iter()) {
                acc += d * (*q as i8 as f32) * *xv;
            }
        }
        out[i] = acc;
    }
}

/// AArch64 NEON Q8_0 matvec (weight Q8_0, activation f32). Mirrors the
/// AVX2/AVX512 approach: widen the block's 32 i8 quants to f32, fold the
/// f16 block scale `d`, and `fmla` against the activations. Four
/// `float32x4` accumulators break the FMA chain; one horizontal sum per
/// row. NEON is baseline on aarch64 (no runtime detection). Not
/// bit-identical to the scalar reference (d folded per block, vector
/// summation) — same tolerance contract as the AVX2 path.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn matvec_q8_0_w_f32_a_neon(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use std::arch::aarch64::*;
    const BLOCK_BYTES: usize = 34;
    const QK: usize = 32;
    let blocks_per_row = k / QK;
    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = [vdupq_n_f32(0.0); 4];
        for b in 0..blocks_per_row {
            let block_off = row_start + b * BLOCK_BYTES;
            let d_bits = u16::from_le_bytes([w_bytes[block_off], w_bytes[block_off + 1]]);
            let dv = vdupq_n_f32(f16::from_bits(d_bits).to_f32());
            let qs = w_bytes.as_ptr().add(block_off + 2) as *const i8;
            let xptr = x.as_ptr().add(b * QK);
            let mut ai = 0usize;
            let mut off = 0usize;
            while off < QK {
                // 8 i8 quants → i16x8 → two f32x4.
                let q16 = vmovl_s8(vld1_s8(qs.add(off)));
                let f_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16)));
                let f_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16)));
                let x_lo = vld1q_f32(xptr.add(off));
                let x_hi = vld1q_f32(xptr.add(off + 4));
                acc[ai & 3] = vfmaq_f32(acc[ai & 3], vmulq_f32(dv, f_lo), x_lo);
                ai += 1;
                acc[ai & 3] = vfmaq_f32(acc[ai & 3], vmulq_f32(dv, f_hi), x_hi);
                ai += 1;
                off += 8;
            }
        }
        let s = vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3]));
        out[i] = vaddvq_f32(s);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn matvec_q8_0_w_f32_a_avx2(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 34;
    const QK: usize = 32;
    let blocks_per_row = k / QK;

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        for b in 0..blocks_per_row {
            let block_off = row_start + b * BLOCK_BYTES;
            // Load f16 scale, convert to f32 via Rust scalar (single shift +
            // mask, faster than a SIMD round-trip through memory), then
            // broadcast across the 8 SIMD lanes.
            let d_bits = u16::from_le_bytes([w_bytes[block_off], w_bytes[block_off + 1]]);
            let d_f32 = f16::from_bits(d_bits).to_f32();
            let d = _mm256_set1_ps(d_f32);

            // Load 32 i8 quants in 4 chunks of 8.
            let qs_ptr = w_bytes.as_ptr().add(block_off + 2) as *const i8;
            // _mm_loadl_epi64 loads 8 bytes; _mm256_cvtepi8_epi32 sign-extends
            // 8 i8 → 8 i32; then convert to f32.
            let q0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(qs_ptr as *const __m128i)));
            let q1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(qs_ptr.add(8) as *const __m128i)));
            let q2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(qs_ptr.add(16) as *const __m128i)));
            let q3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(qs_ptr.add(24) as *const __m128i)));

            // Load 32 input activations.
            let xptr = x.as_ptr().add(b * QK);
            let xv0 = _mm256_loadu_ps(xptr);
            let xv1 = _mm256_loadu_ps(xptr.add(8));
            let xv2 = _mm256_loadu_ps(xptr.add(16));
            let xv3 = _mm256_loadu_ps(xptr.add(24));

            // FMA: acc += d * q * x
            // We can do (d * q) first as one mul, then FMA with x.
            let dq0 = _mm256_mul_ps(d, q0);
            let dq1 = _mm256_mul_ps(d, q1);
            let dq2 = _mm256_mul_ps(d, q2);
            let dq3 = _mm256_mul_ps(d, q3);
            acc0 = _mm256_fmadd_ps(dq0, xv0, acc0);
            acc1 = _mm256_fmadd_ps(dq1, xv1, acc1);
            acc2 = _mm256_fmadd_ps(dq2, xv2, acc2);
            acc3 = _mm256_fmadd_ps(dq3, xv3, acc3);
        }

        // Horizontal sum of 4×8-lane accumulators.
        let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        let mut sum128 = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        out[i] = _mm_cvtss_f32(sum128);
    }
}

/// Q8_K weight matvec with on-the-fly dequant. Block layout (292 B
/// per 256 weights): `{ d: f32, qs: [i8; 256], bsums: [i16; 16] }`.
/// `bsums` is unused in the Q8_K × F32 path — it's only useful for
/// accelerating Q4_K × Q8_K matmuls upstream in K-quant ensembles.
///
/// Same recipe as Q8_0 scaled to 256-weight super-blocks: load 256
/// i8 quants, sign-extend to i32, convert to f32, multiply by `d`
/// (which is f32 here, not f16), FMA against 256 elements of x.
///
/// Dispatches AVX-512 → AVX2 → scalar.
///
/// Requires `k % 256 == 0`.
pub fn matvec_q8_k_w_f32_a(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    const BLOCK_BYTES: usize = 292;
    const QK_K: usize = 256;
    debug_assert_eq!(k % QK_K, 0, "Q8_K matvec requires k % 256 == 0");
    let blocks_per_row = k / QK_K;
    debug_assert_eq!(w_bytes.len(), m * blocks_per_row * BLOCK_BYTES);
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(out.len(), m);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: runtime feature detection above.
            unsafe { matvec_q8_k_w_f32_a_avx512(w_bytes, x, out, m, k) };
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime feature detection above.
            unsafe { matvec_q8_k_w_f32_a_avx2(w_bytes, x, out, m, k) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64.
        unsafe { matvec_q8_k_w_f32_a_neon(w_bytes, x, out, m, k) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    matvec_q8_k_w_f32_a_scalar(w_bytes, x, out, m, k);
}

/// AArch64 NEON Q8_K matvec. Like Q8_0 scaled to 256-weight super-blocks,
/// but the block scale `d` is f32 (not f16) and `bsums` is unused in the
/// Q8_K × F32 path. Widen the 256 i8 quants 8 at a time to f32, fold `d`,
/// `fmla` against the activations; 4 accumulators, one horizontal sum per
/// row. NEON baseline; same tolerance contract as the AVX2 path.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn matvec_q8_k_w_f32_a_neon(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use std::arch::aarch64::*;
    const BLOCK_BYTES: usize = 292;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;
    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = [vdupq_n_f32(0.0); 4];
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let dv = vdupq_n_f32(f32::from_le_bytes([
                w_bytes[off],
                w_bytes[off + 1],
                w_bytes[off + 2],
                w_bytes[off + 3],
            ]));
            let qs = w_bytes.as_ptr().add(off + 4) as *const i8;
            let xptr = x.as_ptr().add(b * QK_K);
            let mut ai = 0usize;
            let mut o = 0usize;
            while o < QK_K {
                // 8 i8 quants → i16x8 → two f32x4.
                let q16 = vmovl_s8(vld1_s8(qs.add(o)));
                let f_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16)));
                let f_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16)));
                acc[ai & 3] = vfmaq_f32(acc[ai & 3], vmulq_f32(dv, f_lo), vld1q_f32(xptr.add(o)));
                ai += 1;
                acc[ai & 3] =
                    vfmaq_f32(acc[ai & 3], vmulq_f32(dv, f_hi), vld1q_f32(xptr.add(o + 4)));
                ai += 1;
                o += 8;
            }
        }
        let s = vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3]));
        out[i] = vaddvq_f32(s);
    }
}

fn matvec_q8_k_w_f32_a_scalar(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    const BLOCK_BYTES: usize = 292;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;
    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d = f32::from_le_bytes([
                w_bytes[off],
                w_bytes[off + 1],
                w_bytes[off + 2],
                w_bytes[off + 3],
            ]);
            let qs = &w_bytes[off + 4..off + 4 + 256];
            let x_block = &x[b * QK_K..(b + 1) * QK_K];
            for (q, xv) in qs.iter().zip(x_block.iter()) {
                acc += d * (*q as i8 as f32) * *xv;
            }
        }
        out[i] = acc;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f")]
unsafe fn matvec_q8_k_w_f32_a_avx512(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 292;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        // 16 ZMM accumulators round-robin over the 16 sub-iterations
        // per super-block (256 / 16 = 16 per block). This keeps the
        // FP pipelines busy through the d-broadcast cost amortized.
        let mut acc = [_mm512_setzero_ps(); 4];

        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d_f32 = f32::from_le_bytes([
                w_bytes[off],
                w_bytes[off + 1],
                w_bytes[off + 2],
                w_bytes[off + 3],
            ]);
            let d = _mm512_set1_ps(d_f32);
            let qs_ptr = w_bytes.as_ptr().add(off + 4) as *const i8;
            let xptr = x.as_ptr().add(b * QK_K);

            // 256 = 16 lanes × 16 ZMM sub-iters.
            for sub in 0..16 {
                let q_raw = _mm_loadu_si128(qs_ptr.add(sub * 16) as *const __m128i);
                let q_f = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q_raw));
                let dq = _mm512_mul_ps(d, q_f);
                let xv = _mm512_loadu_ps(xptr.add(sub * 16));
                let slot = sub & 3;
                acc[slot] = _mm512_fmadd_ps(dq, xv, acc[slot]);
            }
        }
        let s01 = _mm512_add_ps(acc[0], acc[1]);
        let s23 = _mm512_add_ps(acc[2], acc[3]);
        let total = _mm512_add_ps(s01, s23);
        out[i] = _mm512_reduce_add_ps(total);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn matvec_q8_k_w_f32_a_avx2(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use std::arch::x86_64::*;
    const BLOCK_BYTES: usize = 292;
    const QK_K: usize = 256;
    let blocks_per_row = k / QK_K;

    for i in 0..m {
        let row_start = i * blocks_per_row * BLOCK_BYTES;
        // 8 YMM accumulators round-robin over the 32 sub-iterations
        // per super-block (256 / 8 = 32).
        let mut acc = [_mm256_setzero_ps(); 8];

        for b in 0..blocks_per_row {
            let off = row_start + b * BLOCK_BYTES;
            let d_f32 = f32::from_le_bytes([
                w_bytes[off],
                w_bytes[off + 1],
                w_bytes[off + 2],
                w_bytes[off + 3],
            ]);
            let d = _mm256_set1_ps(d_f32);
            let qs_ptr = w_bytes.as_ptr().add(off + 4) as *const i8;
            let xptr = x.as_ptr().add(b * QK_K);

            for sub in 0..32 {
                let q_raw = _mm_loadl_epi64(qs_ptr.add(sub * 8) as *const __m128i);
                let q_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_raw));
                let dq = _mm256_mul_ps(d, q_f);
                let xv = _mm256_loadu_ps(xptr.add(sub * 8));
                let slot = sub & 7;
                acc[slot] = _mm256_fmadd_ps(dq, xv, acc[slot]);
            }
        }
        let s01 = _mm256_add_ps(acc[0], acc[1]);
        let s23 = _mm256_add_ps(acc[2], acc[3]);
        let s45 = _mm256_add_ps(acc[4], acc[5]);
        let s67 = _mm256_add_ps(acc[6], acc[7]);
        let total = _mm256_add_ps(_mm256_add_ps(s01, s23), _mm256_add_ps(s45, s67));
        let lo = _mm256_castps256_ps128(total);
        let hi = _mm256_extractf128_ps(total, 1);
        let sum128 = _mm_add_ps(lo, hi);
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0b01));
        out[i] = _mm_cvtss_f32(sum32);
    }
}

/// Decode PQ2_0 rows by id and write to `out` as F32. Thin wrapper
/// over [`rustllama_gguf::dequant::dequant_pq2_0`] applied
/// row-by-row. Load-bearing for Bonsai models: `token_embd` ships
/// ternary.
pub fn embed_lookup_pq2_0(table_bytes: &[u8], ids: &[i32], out: &mut [f32], d: usize) {
    const BLOCK_BYTES: usize = 34;
    const QK: usize = 128;
    assert_eq!(d % QK, 0, "PQ2_0 embed lookup requires d % 128 == 0");
    let row_bytes = d / QK * BLOCK_BYTES;
    for (i, &id) in ids.iter().enumerate() {
        let row_start = (id as usize) * row_bytes;
        let dst = &mut out[i * d..(i + 1) * d];
        let row = &table_bytes[row_start..row_start + row_bytes];
        rustllama_gguf::dequant::dequant_pq2_0(row, dst);
    }
}

/// Decode PTQ1_0 rows by id and write to `out` as F32. Thin wrapper
/// over [`rustllama_gguf::dequant::dequant_ptq1_0`] applied
/// row-by-row. Load-bearing for Bonsai models: `token_embd` ships
/// ternary.
pub fn embed_lookup_ptq1_0(table_bytes: &[u8], ids: &[i32], out: &mut [f32], d: usize) {
    const BLOCK_BYTES: usize = 28;
    const QK: usize = 128;
    assert_eq!(d % QK, 0, "PTQ1_0 embed lookup requires d % 128 == 0");
    let row_bytes = d / QK * BLOCK_BYTES;
    for (i, &id) in ids.iter().enumerate() {
        let row_start = (id as usize) * row_bytes;
        let dst = &mut out[i * d..(i + 1) * d];
        let row = &table_bytes[row_start..row_start + row_bytes];
        rustllama_gguf::dequant::dequant_ptq1_0(row, dst);
    }
}

/// Decode Q8_K rows by id and write to `out` as F32. Thin wrapper over
/// [`rustllama_gguf::dequant::dequant_q8_k`] applied row-by-row.
pub fn embed_lookup_q8_k(table_bytes: &[u8], ids: &[i32], out: &mut [f32], d: usize) {
    const BLOCK_BYTES: usize = 292;
    const QK_K: usize = 256;
    assert_eq!(d % QK_K, 0, "Q8_K embed lookup requires d % 256 == 0");
    let blocks_per_row = d / QK_K;
    let row_bytes = blocks_per_row * BLOCK_BYTES;
    for (i, &id) in ids.iter().enumerate() {
        let row_start = (id as usize) * row_bytes;
        let dst = &mut out[i * d..(i + 1) * d];
        let row = &table_bytes[row_start..row_start + row_bytes];
        rustllama_gguf::dequant::dequant_q8_k(row, dst);
    }
}

/// Convenience wrapper: matrix–vector multiply with F16 weights × F32 vector.
///
/// On x86_64 with AVX2 + F16C + FMA detected at runtime, uses the vectorized
/// 8-lane path with on-the-fly `vcvtph2ps` to convert F16 weights to F32 —
/// halves memory bandwidth compared to F32-weight storage at the cost of
/// one cheap conversion per chunk.
/// Install a custom rayon thread pool sized to `threads`. Called by
/// the engine at load time so the parallel CPU matmul (and any
/// future rayon-backed kernel) uses the user's `[inference].threads`
/// value instead of rayon's default `num_cpus::get()` heuristic.
///
/// `threads == 0` leaves the global pool alone (rayon picks). Any
/// non-zero value sizes the new pool. Idempotent: subsequent calls
/// after the first are no-ops because rayon refuses to replace its
/// already-built global pool — logged at `debug` level so re-loads
/// don't spam.
pub fn install_thread_pool(threads: usize) {
    use std::sync::OnceLock;
    static INITED: OnceLock<()> = OnceLock::new();
    if threads == 0 {
        return;
    }
    INITED.get_or_init(|| {
        match rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("rustllama-cpu-{i}"))
            .build_global()
        {
            Ok(()) => tracing::info!(threads, "CPU thread pool sized via rayon"),
            Err(e) => tracing::debug!(error = %e, "rayon pool already built; threads setting ignored"),
        }
    });
}

/// Shared one-shot guard: rayon's global pool builds once per process, so
/// whichever of [`install_thread_pool`] / [`install_thread_pool_pinned`]
/// runs first wins. (rayon's own `build_global` is also idempotent — this
/// just keeps the second call from logging noise.)
static POOL_INIT: std::sync::OnceLock<()> = std::sync::OnceLock::new();

/// Install a rayon global pool restricted to — and thread-affinity-PINNED
/// onto — a specific set of enabled logical processors. This is the CPU
/// analog of the GPU disable-list: `enabled_cpus` is the set of logical
/// processors left after removing `[inference].disabled_cpus` (P/E-core
/// aware), and each rayon worker is pinned to one of them so disabled
/// cores run no kernel work and the OS scheduler can't migrate a worker
/// onto a core the user excluded.
///
/// Pool size = `enabled_cpus.len()`, further capped to `threads` when
/// `threads > 0` (honoring `[inference].threads`). Worker `i` is pinned to
/// `enabled_cpus[i % enabled_cpus.len()]`.
///
/// Pinning is **best-effort**: a failed affinity call logs at `debug` and
/// the worker keeps running unpinned. If `enabled_cpus` is empty this
/// delegates to [`install_thread_pool`] (no pinning) so an empty/degenerate
/// set never yields a zero-thread pool.
///
/// Platform affinity:
/// - **Windows** — `SetThreadAffinityMask(GetCurrentThread(), 1 << cpu)`.
///   Processors `>= 64` (processor-group systems) are left unpinned
///   (best-effort; the dev hardware is a single group).
/// - **Linux** — `sched_setaffinity(0, ..., {cpu})` on the calling worker.
/// - **Other** — no pinning (pool is still sized to the enabled count).
pub fn install_thread_pool_pinned(enabled_cpus: Vec<u32>, threads: usize) {
    if enabled_cpus.is_empty() {
        // Degenerate: nothing to pin onto. Preserve historical behavior.
        install_thread_pool(threads);
        return;
    }
    let size = if threads > 0 {
        threads.min(enabled_cpus.len())
    } else {
        enabled_cpus.len()
    }
    .max(1);
    POOL_INIT.get_or_init(|| {
        let pins = enabled_cpus.clone();
        let result = rayon::ThreadPoolBuilder::new()
            .num_threads(size)
            .thread_name(|i| format!("rustllama-cpu-{i}"))
            .start_handler(move |worker_index| {
                let cpu = pins[worker_index % pins.len()];
                pin_current_thread_to_cpu(cpu);
            })
            .build_global();
        match result {
            Ok(()) => tracing::info!(
                threads = size,
                enabled = enabled_cpus.len(),
                "CPU thread pool sized + affinity-pinned to enabled logical processors"
            ),
            Err(e) => {
                tracing::debug!(error = %e, "rayon pool already built; pinned pool request ignored")
            }
        }
    });
}

/// Best-effort: pin the CALLING thread to a single logical processor.
/// Logs at `debug` on failure and returns (the worker keeps running).
#[cfg(windows)]
fn pin_current_thread_to_cpu(cpu: u32) {
    if cpu >= 64 {
        // Single-affinity-mask path only covers processor group 0. Leave
        // procs in higher groups unpinned rather than mis-pinning them.
        tracing::debug!(cpu, "affinity: proc >= 64 (processor group); left unpinned");
        return;
    }
    extern "system" {
        fn GetCurrentThread() -> isize;
        fn SetThreadAffinityMask(thread: isize, mask: usize) -> usize;
    }
    // SAFETY: pseudo-handle from GetCurrentThread is valid for the calling
    // thread; the mask selects exactly one processor. A 0 return = failure.
    let prev = unsafe { SetThreadAffinityMask(GetCurrentThread(), 1usize << cpu) };
    if prev == 0 {
        tracing::debug!(cpu, "SetThreadAffinityMask failed; worker left unpinned");
    }
}

/// Best-effort: pin the CALLING thread to a single logical processor.
#[cfg(target_os = "linux")]
fn pin_current_thread_to_cpu(cpu: u32) {
    extern "C" {
        // `pid == 0` targets the calling thread (a Linux task).
        fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const u64) -> i32;
    }
    // cpu_set_t is a 1024-bit bitmap (16 × u64). Set exactly one bit.
    let mut set = [0u64; 16];
    let word = (cpu / 64) as usize;
    if word >= set.len() {
        tracing::debug!(cpu, "affinity: proc out of cpu_set_t range; left unpinned");
        return;
    }
    set[word] = 1u64 << (cpu % 64);
    // SAFETY: `set` is a valid 128-byte buffer matching cpu_set_t's size;
    // sched_setaffinity only reads it. libc is linked by the Rust std.
    let rc = unsafe {
        sched_setaffinity(
            0,
            std::mem::size_of::<[u64; 16]>(),
            set.as_ptr(),
        )
    };
    if rc != 0 {
        tracing::debug!(cpu, "sched_setaffinity failed; worker left unpinned");
    }
}

#[cfg(not(any(windows, target_os = "linux")))]
fn pin_current_thread_to_cpu(_cpu: u32) {
    // No portable affinity API; pool is still sized to the enabled count.
}

/// Parallel F16 matvec: same contract as [`matvec_f16_w_f32_a`] but
/// distributes the M-axis outer loop across the rayon thread pool.
/// Win is real when `M * K` is large enough to amortize the work-
/// stealing overhead (~10-30 µs); for `M * K < ~64K` the serial path
/// is usually faster. The [`matvec_f16_w_f32_a`] auto-dispatcher
/// picks this at `m ≥ parallel_matvec_crossover()`.
///
/// Splits `out` into `m / chunk_rows`-sized horizontal stripes; each
/// stripe processes its rows serially using the same SIMD inner
/// loop as the serial path. `chunk_rows` is heuristic: 32 rows
/// per chunk keeps work-stealing latency low without producing
/// tiny tasks.
pub fn matvec_f16_w_f32_a_parallel(
    w: &[f16],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use rayon::iter::{IndexedParallelIterator, ParallelIterator};
    use rayon::slice::ParallelSliceMut;
    debug_assert_eq!(w.len(), m * k);
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(out.len(), m);
    if m == 0 {
        return;
    }
    // Chunk size = ceil(m / (4 × n_threads)). 4× the thread count
    // gives rayon's work-stealing some slack while keeping per-
    // chunk overhead a fraction of total compute. Empirically on
    // Iris Xe (8 logical cores), 32-row chunks were 10× too small
    // — overhead dominated until M=16384. With this gate the
    // crossover drops to M≈1024.
    let n_threads = rayon::current_num_threads().max(1);
    let chunk_rows = ((m + 4 * n_threads - 1) / (4 * n_threads)).max(64).min(m);
    out.par_chunks_mut(chunk_rows).enumerate().for_each(|(chunk_idx, out_chunk)| {
        let row_start = chunk_idx * chunk_rows;
        let row_end = (row_start + out_chunk.len()).min(m);
        let w_chunk = &w[row_start * k..row_end * k];
        let n_rows = out_chunk.len();
        // Each chunk uses the serial path. Calling the auto-
        // dispatcher would recurse into rayon and starve the work-
        // steal queue.
        matvec_f16_w_f32_a_serial(w_chunk, x, out_chunk, n_rows, k);
    });
}

pub fn matvec_f16_w_f32_a(w: &[f16], x: &[f32], out: &mut [f32], m: usize, k: usize) {
    debug_assert_eq!(w.len(), m * k);
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(out.len(), m);

    // Auto-dispatch by row count. The crossover lives at M ≈ 256
    // on Iris Xe (8 logical cores, see
    // `examples/matvec_parallel_crossover.rs`); below that the
    // rayon overhead dominates the matmul work. Above, parallel
    // wins 1.5-4×. Env-var override: `RUSTLLAMA_PARALLEL_MATVEC_M`
    // sets a custom threshold (0 disables, picking serial always).
    let crossover = parallel_matvec_crossover();
    if crossover > 0 && m >= crossover {
        matvec_f16_w_f32_a_parallel(w, x, out, m, k);
        return;
    }

    matvec_f16_w_f32_a_serial(w, x, out, m, k);
}

/// Read the parallel-matvec crossover once per process. Default
/// `256` is set by the Iris Xe microbench; users on different
/// hosts override via `RUSTLLAMA_PARALLEL_MATVEC_M`. `0` disables
/// the parallel path entirely (always serial).
fn parallel_matvec_crossover() -> usize {
    use std::sync::OnceLock;
    static GATE: OnceLock<usize> = OnceLock::new();
    *GATE.get_or_init(|| {
        std::env::var("RUSTLLAMA_PARALLEL_MATVEC_M")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(256)
    })
}

/// Widen four f16 bit-patterns (each zero-extended into a `u32` lane) to
/// `float32x4`, hand-rolled so it needs only baseline NEON — the
/// `vcvt_f32_f16` intrinsics are still behind the unstable
/// `stdarch_neon_f16` feature, so they can't be used on stable Rust. Uses
/// Fabian Giesen's branchless float-scale trick: place the 15 exponent +
/// mantissa bits, multiply by a magic `2^112` to re-bias the exponent
/// (which also reconstructs f16 subnormals for free, since the product is a
/// normal f32), lift Inf/NaN via a compare, then OR the sign back. Exact for
/// every finite f16 (all are representable in f32), so bit-identical to
/// `half::f16::to_f32` on real weight data.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn f16x4_to_f32x4(h: std::arch::aarch64::uint32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    let magic = vreinterpretq_f32_u32(vdupq_n_u32((254 - 15) << 23));
    let was_infnan = vdupq_n_f32(f32::from_bits((127 + 16) << 23));
    let expmant = vshlq_n_u32::<13>(vandq_u32(h, vdupq_n_u32(0x7fff)));
    let scaled = vmulq_f32(vreinterpretq_f32_u32(expmant), magic);
    // Lanes that overflow the f16 finite range (Inf/NaN) get the exponent
    // forced to all-ones; `vcgeq_f32` yields the per-lane select mask.
    let is_infnan = vcgeq_f32(scaled, was_infnan);
    let infnan_bits = vandq_u32(is_infnan, vdupq_n_u32(255 << 23));
    let sign = vshlq_n_u32::<16>(vandq_u32(h, vdupq_n_u32(0x8000)));
    let bits = vorrq_u32(vorrq_u32(vreinterpretq_u32_f32(scaled), infnan_bits), sign);
    vreinterpretq_f32_u32(bits)
}

/// AArch64 NEON serial F16 matvec. Widens f16 weights to f32 with the stable
/// hand-rolled [`f16x4_to_f32x4`] (no unstable fp16 intrinsics). 16-lane
/// main tile (4 accumulators), 4-lane mid loop, scalar tail — the f16 twin
/// of the BF16 NEON path. NEON baseline; same tolerance contract as the
/// AVX2 path.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn matvec_f16_w_f32_a_neon(w: &[f16], x: &[f32], out: &mut [f32], m: usize, k: usize) {
    use std::arch::aarch64::*;
    let k_main = k & !15;
    let k_main4 = k & !3;
    for i in 0..m {
        let wptr = w.as_ptr().add(i * k) as *const u16;
        let mut a = [vdupq_n_f32(0.0); 4];
        let mut p = 0;
        while p < k_main {
            let h0 = vld1q_u16(wptr.add(p));
            let h1 = vld1q_u16(wptr.add(p + 8));
            let w0 = f16x4_to_f32x4(vmovl_u16(vget_low_u16(h0)));
            let w1 = f16x4_to_f32x4(vmovl_u16(vget_high_u16(h0)));
            let w2 = f16x4_to_f32x4(vmovl_u16(vget_low_u16(h1)));
            let w3 = f16x4_to_f32x4(vmovl_u16(vget_high_u16(h1)));
            let xp = x.as_ptr().add(p);
            a[0] = vfmaq_f32(a[0], w0, vld1q_f32(xp));
            a[1] = vfmaq_f32(a[1], w1, vld1q_f32(xp.add(4)));
            a[2] = vfmaq_f32(a[2], w2, vld1q_f32(xp.add(8)));
            a[3] = vfmaq_f32(a[3], w3, vld1q_f32(xp.add(12)));
            p += 16;
        }
        while p < k_main4 {
            let h = f16x4_to_f32x4(vmovl_u16(vld1_u16(wptr.add(p))));
            a[0] = vfmaq_f32(a[0], h, vld1q_f32(x.as_ptr().add(p)));
            p += 4;
        }
        let mut sum = vaddvq_f32(vaddq_f32(vaddq_f32(a[0], a[1]), vaddq_f32(a[2], a[3])));
        while p < k {
            sum += w[i * k + p].to_f32() * x[p];
            p += 1;
        }
        out[i] = sum;
    }
}

/// Serial F16 matvec — the original SIMD-only path. The public
/// [`matvec_f16_w_f32_a`] auto-dispatcher picks this for small M;
/// callers that always want the serial path call this directly.
pub fn matvec_f16_w_f32_a_serial(
    w: &[f16],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    debug_assert_eq!(w.len(), m * k);
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(out.len(), m);

    #[cfg(target_arch = "x86_64")]
    {
        // AVX-512 + F16C is the rare combo that lets us widen f16→f32
        // in 16-lane chunks. Without AVX-512, fall through to AVX2.
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("f16c") {
            // SAFETY: runtime feature detection above.
            unsafe { matvec_f16_w_f32_a_avx512(w, x, out, m, k) };
            return;
        }
        if is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && is_x86_feature_detected!("f16c")
        {
            // SAFETY: runtime feature detection above.
            unsafe { matvec_f16_w_f32_a_avx2(w, x, out, m, k) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64.
        unsafe { matvec_f16_w_f32_a_neon(w, x, out, m, k) };
        return;
    }

    #[cfg(not(target_arch = "aarch64"))]
    gemm_f16_w_f32_a(w, x, out, m, 1, k);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma,f16c,avx512f")]
unsafe fn matvec_f16_w_f32_a_avx512(w: &[f16], x: &[f32], out: &mut [f32], m: usize, k: usize) {
    use std::arch::x86_64::*;

    // 32-lane outer tile (2 × 16-wide FMA accumulators), 16-lane mid
    // tile, scalar tail. The f16→f32 widening uses `vcvtph2ps` on a
    // 128-bit XMM source — there's no AVX-512 equivalent that takes a
    // ZMM-sized i16 source, so we do two 128-bit XMM loads per 16-lane
    // step.
    let k_main_32 = k & !31;
    let k_main_16 = k & !15;

    for i in 0..m {
        let w_row = &w[i * k..(i + 1) * k];
        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();

        let mut p = 0;
        while p < k_main_32 {
            let wptr = w_row.as_ptr().add(p) as *const __m128i;
            let xptr = x.as_ptr().add(p);
            // Two 16-wide widens: 16 f16 per ZMM accumulator. Each
            // widen is itself two XMM→YMM `vcvtph2ps` and a YMM→ZMM
            // concat; the compiler picks the cheapest path.
            let wf0 = _mm512_cvtph_ps(_mm256_loadu_si256(wptr as *const __m256i));
            let wf1 = _mm512_cvtph_ps(_mm256_loadu_si256(wptr.add(2) as *const __m256i));
            let xv0 = _mm512_loadu_ps(xptr);
            let xv1 = _mm512_loadu_ps(xptr.add(16));
            acc0 = _mm512_fmadd_ps(wf0, xv0, acc0);
            acc1 = _mm512_fmadd_ps(wf1, xv1, acc1);
            p += 32;
        }

        while p < k_main_16 {
            let w_lane = _mm256_loadu_si256(w_row.as_ptr().add(p) as *const __m256i);
            let wf = _mm512_cvtph_ps(w_lane);
            let xv = _mm512_loadu_ps(x.as_ptr().add(p));
            acc0 = _mm512_fmadd_ps(wf, xv, acc0);
            p += 16;
        }

        let acc = _mm512_add_ps(acc0, acc1);
        let mut sum = _mm512_reduce_add_ps(acc);

        while p < k {
            sum += w_row[p].to_f32() * x[p];
            p += 1;
        }

        out[i] = sum;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma,f16c")]
unsafe fn matvec_f16_w_f32_a_avx2(w: &[f16], x: &[f32], out: &mut [f32], m: usize, k: usize) {
    use std::arch::x86_64::*;

    let k_main_16 = k & !15; // 16-lane chunks (two 8-wide AVX vectors)
    let k_main_8 = k & !7;

    for i in 0..m {
        let w_row = &w[i * k..(i + 1) * k];
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();

        let mut p = 0;
        while p < k_main_16 {
            // 16 f16 weights = 32 bytes = two 128-bit SIMD loads, then
            // split via two F16C conversions.
            let wptr = w_row.as_ptr().add(p) as *const __m128i;
            let xptr = x.as_ptr().add(p);
            let wf0 = _mm256_cvtph_ps(_mm_loadu_si128(wptr));
            let wf1 = _mm256_cvtph_ps(_mm_loadu_si128(wptr.add(1)));
            let xv0 = _mm256_loadu_ps(xptr);
            let xv1 = _mm256_loadu_ps(xptr.add(8));
            acc0 = _mm256_fmadd_ps(wf0, xv0, acc0);
            acc1 = _mm256_fmadd_ps(wf1, xv1, acc1);
            p += 16;
        }

        while p < k_main_8 {
            let w_lane = _mm_loadu_si128(w_row.as_ptr().add(p) as *const __m128i);
            let wf = _mm256_cvtph_ps(w_lane);
            let xv = _mm256_loadu_ps(x.as_ptr().add(p));
            acc0 = _mm256_fmadd_ps(wf, xv, acc0);
            p += 8;
        }

        let acc = _mm256_add_ps(acc0, acc1);
        let mut sum128 = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        let mut sum = _mm_cvtss_f32(sum128);

        while p < k {
            sum += w_row[p].to_f32() * x[p];
            p += 1;
        }

        out[i] = sum;
    }
}

/// BF16 weights × F32 vector matrix-vector multiply. BF16 is the top
/// 16 bits of an IEEE-754 f32 — widening to f32 is a single zero-fill
/// left-shift by 16 bits per lane, which AVX-512 / AVX2 can do
/// natively on i32 registers. No `f16c` / `vcvtph2ps` required.
///
/// Layout convention: `w` is treated as `[m, k]` row-major BF16 bytes;
/// `as_bytes(tensor)` produces the right slice when the caller has a
/// `Dtype::Bf16Raw` tensor.
///
/// On x86_64 with AVX-512 → AVX2 + FMA → scalar fallback. The widening
/// step (`u16 → i32 → shl 16 → reinterpret as f32`) is two instructions
/// on AVX-512 (`vpmovzxwd` + `vpslld`) and the same on AVX2.
pub fn matvec_bf16_w_f32_a(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    debug_assert_eq!(w_bytes.len(), m * k * 2);
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(out.len(), m);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: runtime feature detection above.
            unsafe { matvec_bf16_w_f32_a_avx512(w_bytes, x, out, m, k) };
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime feature detection above.
            unsafe { matvec_bf16_w_f32_a_avx2(w_bytes, x, out, m, k) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64.
        unsafe { matvec_bf16_w_f32_a_neon(w_bytes, x, out, m, k) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    matvec_bf16_w_f32_a_scalar(w_bytes, x, out, m, k);
}

/// AArch64 NEON BF16 matvec. BF16 widens to f32 by a zero-fill left-shift
/// of 16 bits — `vshll_n_u16::<16>` lands the 16 BF16 bits in the top of a
/// u32, which reinterprets directly as f32 (no `fp16`/vcvt needed, so this
/// is stable-NEON). 16-lane main tile (4 accumulators), 4-lane mid loop,
/// scalar tail. NEON baseline; same tolerance contract as the AVX2 path.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn matvec_bf16_w_f32_a_neon(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use std::arch::aarch64::*;
    let k_main = k & !15;
    let k_main4 = k & !3;
    for i in 0..m {
        let row = i * k * 2;
        let wptr = w_bytes.as_ptr().add(row) as *const u16;
        let mut a = [vdupq_n_f32(0.0); 4];
        let mut p = 0;
        while p < k_main {
            let u0 = vld1q_u16(wptr.add(p));
            let u1 = vld1q_u16(wptr.add(p + 8));
            let w0 = vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(u0)));
            let w1 = vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_high_u16(u0)));
            let w2 = vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(u1)));
            let w3 = vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_high_u16(u1)));
            let xp = x.as_ptr().add(p);
            a[0] = vfmaq_f32(a[0], w0, vld1q_f32(xp));
            a[1] = vfmaq_f32(a[1], w1, vld1q_f32(xp.add(4)));
            a[2] = vfmaq_f32(a[2], w2, vld1q_f32(xp.add(8)));
            a[3] = vfmaq_f32(a[3], w3, vld1q_f32(xp.add(12)));
            p += 16;
        }
        while p < k_main4 {
            let w = vreinterpretq_f32_u32(vshll_n_u16::<16>(vld1_u16(wptr.add(p))));
            a[0] = vfmaq_f32(a[0], w, vld1q_f32(x.as_ptr().add(p)));
            p += 4;
        }
        let mut sum = vaddvq_f32(vaddq_f32(vaddq_f32(a[0], a[1]), vaddq_f32(a[2], a[3])));
        while p < k {
            let u = u16::from_le_bytes([w_bytes[row + p * 2], w_bytes[row + p * 2 + 1]]);
            sum += f32::from_bits((u as u32) << 16) * x[p];
            p += 1;
        }
        out[i] = sum;
    }
}

fn matvec_bf16_w_f32_a_scalar(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    for i in 0..m {
        let row_off = i * k * 2;
        let mut acc = 0.0f32;
        for p in 0..k {
            let lo = w_bytes[row_off + p * 2];
            let hi = w_bytes[row_off + p * 2 + 1];
            let u = u16::from_le_bytes([lo, hi]);
            let w_f32 = f32::from_bits((u as u32) << 16);
            acc += w_f32 * x[p];
        }
        out[i] = acc;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f")]
unsafe fn matvec_bf16_w_f32_a_avx512(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use std::arch::x86_64::*;
    // 32-lane main loop (2 × 16-wide FMAs); 16-lane mid loop; scalar tail.
    let k_main_32 = k & !31;
    let k_main_16 = k & !15;

    for i in 0..m {
        let row_off = i * k * 2;
        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();

        let mut p = 0;
        while p < k_main_32 {
            // 16 BF16 lanes per ZMM: load 32 bytes (16 u16), widen to
            // 16 i32, shift left 16, reinterpret as f32.
            let wptr = w_bytes.as_ptr().add(row_off + p * 2);
            let w0_u16 = _mm256_loadu_si256(wptr as *const __m256i);
            let w1_u16 = _mm256_loadu_si256(wptr.add(32) as *const __m256i);
            let w0_i32 = _mm512_cvtepu16_epi32(w0_u16);
            let w1_i32 = _mm512_cvtepu16_epi32(w1_u16);
            let w0_shifted = _mm512_slli_epi32::<16>(w0_i32);
            let w1_shifted = _mm512_slli_epi32::<16>(w1_i32);
            let w0_f = _mm512_castsi512_ps(w0_shifted);
            let w1_f = _mm512_castsi512_ps(w1_shifted);

            let xptr = x.as_ptr().add(p);
            let xv0 = _mm512_loadu_ps(xptr);
            let xv1 = _mm512_loadu_ps(xptr.add(16));
            acc0 = _mm512_fmadd_ps(w0_f, xv0, acc0);
            acc1 = _mm512_fmadd_ps(w1_f, xv1, acc1);
            p += 32;
        }

        while p < k_main_16 {
            let wptr = w_bytes.as_ptr().add(row_off + p * 2);
            let w_u16 = _mm256_loadu_si256(wptr as *const __m256i);
            let w_i32 = _mm512_cvtepu16_epi32(w_u16);
            let w_shifted = _mm512_slli_epi32::<16>(w_i32);
            let w_f = _mm512_castsi512_ps(w_shifted);
            let xv = _mm512_loadu_ps(x.as_ptr().add(p));
            acc0 = _mm512_fmadd_ps(w_f, xv, acc0);
            p += 16;
        }

        let acc = _mm512_add_ps(acc0, acc1);
        let mut sum = _mm512_reduce_add_ps(acc);

        // Scalar tail for any leftover < 16 elements.
        while p < k {
            let lo = w_bytes[row_off + p * 2];
            let hi = w_bytes[row_off + p * 2 + 1];
            let u = u16::from_le_bytes([lo, hi]);
            sum += f32::from_bits((u as u32) << 16) * x[p];
            p += 1;
        }
        out[i] = sum;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn matvec_bf16_w_f32_a_avx2(
    w_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
) {
    use std::arch::x86_64::*;
    let k_main_16 = k & !15;
    let k_main_8 = k & !7;

    for i in 0..m {
        let row_off = i * k * 2;
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();

        let mut p = 0;
        while p < k_main_16 {
            // 16 BF16 = 32 bytes; split into two 8-wide YMM widens.
            let wptr = w_bytes.as_ptr().add(row_off + p * 2);
            let w0_u16 = _mm_loadu_si128(wptr as *const __m128i);
            let w1_u16 = _mm_loadu_si128(wptr.add(16) as *const __m128i);
            let w0_i32 = _mm256_cvtepu16_epi32(w0_u16);
            let w1_i32 = _mm256_cvtepu16_epi32(w1_u16);
            let w0_f = _mm256_castsi256_ps(_mm256_slli_epi32::<16>(w0_i32));
            let w1_f = _mm256_castsi256_ps(_mm256_slli_epi32::<16>(w1_i32));

            let xptr = x.as_ptr().add(p);
            let xv0 = _mm256_loadu_ps(xptr);
            let xv1 = _mm256_loadu_ps(xptr.add(8));
            acc0 = _mm256_fmadd_ps(w0_f, xv0, acc0);
            acc1 = _mm256_fmadd_ps(w1_f, xv1, acc1);
            p += 16;
        }

        while p < k_main_8 {
            let wptr = w_bytes.as_ptr().add(row_off + p * 2);
            let w_u16 = _mm_loadu_si128(wptr as *const __m128i);
            let w_i32 = _mm256_cvtepu16_epi32(w_u16);
            let w_f = _mm256_castsi256_ps(_mm256_slli_epi32::<16>(w_i32));
            let xv = _mm256_loadu_ps(x.as_ptr().add(p));
            acc0 = _mm256_fmadd_ps(w_f, xv, acc0);
            p += 8;
        }

        let acc = _mm256_add_ps(acc0, acc1);
        let mut sum128 =
            _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        let mut sum = _mm_cvtss_f32(sum128);

        while p < k {
            let lo = w_bytes[row_off + p * 2];
            let hi = w_bytes[row_off + p * 2 + 1];
            let u = u16::from_le_bytes([lo, hi]);
            sum += f32::from_bits((u as u32) << 16) * x[p];
            p += 1;
        }
        out[i] = sum;
    }
}

/// Decode BF16 rows by id and write to `out` as F32. Thin wrapper over
/// [`rustllama_gguf::dequant::dequant_bf16`] applied row-by-row.
pub fn embed_lookup_bf16(table_bytes: &[u8], ids: &[i32], out: &mut [f32], d: usize) {
    let row_bytes = d * 2;
    for (i, &id) in ids.iter().enumerate() {
        let row_start = (id as usize) * row_bytes;
        let dst = &mut out[i * d..(i + 1) * d];
        let row = &table_bytes[row_start..row_start + row_bytes];
        rustllama_gguf::dequant::dequant_bf16(row, dst);
    }
}

/// F32 weights × F32 vector matrix-vector multiply. Used by the F32 weight
/// storage path; keeps full precision after dequantization.
///
/// Auto-dispatches by row count like [`matvec_f16_w_f32_a`]: at
/// `m ≥ parallel_matvec_crossover()` the M-axis splits across the
/// rayon pool (per-row accumulation stays serial inside the SIMD
/// kernel, so results are bit-identical to the serial path). F32
/// projections are the dominant per-token DRAM traffic on
/// F32-projection quant recipes and previously ran on ONE core.
pub fn matvec_f32(w: &[f32], x: &[f32], out: &mut [f32], m: usize, k: usize) {
    let crossover = parallel_matvec_crossover();
    if crossover > 0 && m >= crossover {
        use rayon::iter::{IndexedParallelIterator, ParallelIterator};
        use rayon::slice::ParallelSliceMut;
        let n_threads = rayon::current_num_threads().max(1);
        let chunk_rows = ((m + 4 * n_threads - 1) / (4 * n_threads)).max(64).min(m);
        out.par_chunks_mut(chunk_rows)
            .enumerate()
            .for_each(|(ci, oc)| {
                let r0 = ci * chunk_rows;
                let nr = oc.len();
                matvec_f32_serial(&w[r0 * k..(r0 + nr) * k], x, oc, nr, k);
            });
        return;
    }
    matvec_f32_serial(w, x, out, m, k);
}

/// Serial F32 matvec — the original SIMD-only path. On x86_64 with
/// AVX2 + FMA detected at runtime, dispatches to the vectorized
/// 8-lane path with 4 accumulators (≈ 1 FMA/cycle throughput).
/// Falls back to a naive scalar loop otherwise.
pub fn matvec_f32_serial(w: &[f32], x: &[f32], out: &mut [f32], m: usize, k: usize) {
    debug_assert_eq!(w.len(), m * k);
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(out.len(), m);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: runtime feature detection.
            unsafe {
                matvec_f32_avx512(w, x, out, m, k);
            }
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: we just verified the target features at runtime.
            unsafe {
                matvec_f32_avx2(w, x, out, m, k);
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64.
        unsafe { matvec_f32_neon(w, x, out, m, k) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    matvec_f32_scalar(w, x, out, m, k);
}

/// AArch64 NEON F32 matvec. 16-lane main tile (4 × `float32x4` FMA
/// accumulators), 4-lane mid loop, scalar tail — the ARM analogue of the
/// AVX2 4-accumulator structure. `k` need not be a multiple of the tile.
/// NEON baseline; vector summation differs from scalar only in reduction
/// order (same tolerance contract as the AVX2 path).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn matvec_f32_neon(w: &[f32], x: &[f32], out: &mut [f32], m: usize, k: usize) {
    use std::arch::aarch64::*;
    let k_main = k & !15;
    let k_main4 = k & !3;
    for i in 0..m {
        let w_row = w.as_ptr().add(i * k);
        let mut a = [vdupq_n_f32(0.0); 4];
        let mut p = 0;
        while p < k_main {
            let wp = w_row.add(p);
            let xp = x.as_ptr().add(p);
            a[0] = vfmaq_f32(a[0], vld1q_f32(wp), vld1q_f32(xp));
            a[1] = vfmaq_f32(a[1], vld1q_f32(wp.add(4)), vld1q_f32(xp.add(4)));
            a[2] = vfmaq_f32(a[2], vld1q_f32(wp.add(8)), vld1q_f32(xp.add(8)));
            a[3] = vfmaq_f32(a[3], vld1q_f32(wp.add(12)), vld1q_f32(xp.add(12)));
            p += 16;
        }
        while p < k_main4 {
            a[0] = vfmaq_f32(a[0], vld1q_f32(w_row.add(p)), vld1q_f32(x.as_ptr().add(p)));
            p += 4;
        }
        let mut sum = vaddvq_f32(vaddq_f32(vaddq_f32(a[0], a[1]), vaddq_f32(a[2], a[3])));
        while p < k {
            sum += *w_row.add(p) * x[p];
            p += 1;
        }
        out[i] = sum;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f")]
unsafe fn matvec_f32_avx512(w: &[f32], x: &[f32], out: &mut [f32], m: usize, k: usize) {
    use std::arch::x86_64::*;

    // 64 lanes per tile (4 × 16-wide ZMM) — same 4-accumulator structure
    // as the AVX2 path, just doubled lane width. Saturates the FMA unit
    // on Ice Lake / Sapphire Rapids / Zen 4.
    let k_main_64 = k & !63;
    let k_main_16 = k & !15;

    for i in 0..m {
        let w_row = &w[i * k..(i + 1) * k];
        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();
        let mut acc2 = _mm512_setzero_ps();
        let mut acc3 = _mm512_setzero_ps();

        let mut p = 0;
        while p < k_main_64 {
            let wptr = w_row.as_ptr().add(p);
            let xptr = x.as_ptr().add(p);
            let wv0 = _mm512_loadu_ps(wptr);
            let wv1 = _mm512_loadu_ps(wptr.add(16));
            let wv2 = _mm512_loadu_ps(wptr.add(32));
            let wv3 = _mm512_loadu_ps(wptr.add(48));
            let xv0 = _mm512_loadu_ps(xptr);
            let xv1 = _mm512_loadu_ps(xptr.add(16));
            let xv2 = _mm512_loadu_ps(xptr.add(32));
            let xv3 = _mm512_loadu_ps(xptr.add(48));
            acc0 = _mm512_fmadd_ps(wv0, xv0, acc0);
            acc1 = _mm512_fmadd_ps(wv1, xv1, acc1);
            acc2 = _mm512_fmadd_ps(wv2, xv2, acc2);
            acc3 = _mm512_fmadd_ps(wv3, xv3, acc3);
            p += 64;
        }

        while p < k_main_16 {
            let wv = _mm512_loadu_ps(w_row.as_ptr().add(p));
            let xv = _mm512_loadu_ps(x.as_ptr().add(p));
            acc0 = _mm512_fmadd_ps(wv, xv, acc0);
            p += 16;
        }

        let acc = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
        let mut sum = _mm512_reduce_add_ps(acc);

        while p < k {
            sum += w_row[p] * x[p];
            p += 1;
        }

        out[i] = sum;
    }
}

fn matvec_f32_scalar(w: &[f32], x: &[f32], out: &mut [f32], m: usize, k: usize) {
    for i in 0..m {
        let w_row = &w[i * k..(i + 1) * k];
        let mut acc = 0.0f32;
        for p in 0..k {
            acc += w_row[p] * x[p];
        }
        out[i] = acc;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn matvec_f32_avx2(w: &[f32], x: &[f32], out: &mut [f32], m: usize, k: usize) {
    use std::arch::x86_64::*;

    // Tile the inner loop into chunks of 32 lanes (4 × 8-wide SIMD) so we
    // keep four independent FMA accumulators in flight and saturate the
    // FMA unit (latency 4, throughput 0.5).
    let k_main_32 = k & !31;
    let k_main_8 = k & !7;

    for i in 0..m {
        let w_row = &w[i * k..(i + 1) * k];
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        let mut p = 0;
        while p < k_main_32 {
            let wptr = w_row.as_ptr().add(p);
            let xptr = x.as_ptr().add(p);
            let wv0 = _mm256_loadu_ps(wptr);
            let wv1 = _mm256_loadu_ps(wptr.add(8));
            let wv2 = _mm256_loadu_ps(wptr.add(16));
            let wv3 = _mm256_loadu_ps(wptr.add(24));
            let xv0 = _mm256_loadu_ps(xptr);
            let xv1 = _mm256_loadu_ps(xptr.add(8));
            let xv2 = _mm256_loadu_ps(xptr.add(16));
            let xv3 = _mm256_loadu_ps(xptr.add(24));
            acc0 = _mm256_fmadd_ps(wv0, xv0, acc0);
            acc1 = _mm256_fmadd_ps(wv1, xv1, acc1);
            acc2 = _mm256_fmadd_ps(wv2, xv2, acc2);
            acc3 = _mm256_fmadd_ps(wv3, xv3, acc3);
            p += 32;
        }

        while p < k_main_8 {
            let wv = _mm256_loadu_ps(w_row.as_ptr().add(p));
            let xv = _mm256_loadu_ps(x.as_ptr().add(p));
            acc0 = _mm256_fmadd_ps(wv, xv, acc0);
            p += 8;
        }

        let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        let mut sum128 = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        let mut sum = _mm_cvtss_f32(sum128);

        while p < k {
            sum += w_row[p] * x[p];
            p += 1;
        }

        out[i] = sum;
    }
}

/// Embedding lookup from F32 table → F32 activations.
pub fn embed_lookup_f32(table: &[f32], ids: &[i32], out: &mut [f32], d: usize) {
    debug_assert_eq!(out.len(), ids.len() * d);
    for (i, &id) in ids.iter().enumerate() {
        let row_start = (id as usize) * d;
        out[i * d..(i + 1) * d].copy_from_slice(&table[row_start..row_start + d]);
    }
}

/// F32 × F32 gemm (legacy).
pub fn gemm_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
}

/// Root-mean-square layer norm of a single row.
///   `y[i] = x[i] * weight[i] / sqrt(mean(x^2) + eps)`
#[inline]
pub fn rmsnorm_f32_row(x: &[f32], weight: &[f32], y: &mut [f32], eps: f32) {
    debug_assert_eq!(x.len(), weight.len());
    debug_assert_eq!(x.len(), y.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: runtime feature detection above.
            unsafe { rmsnorm_f32_row_avx512f(x, weight, y, eps) };
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime feature detection above.
            unsafe { rmsnorm_f32_row_avx2(x, weight, y, eps) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64.
        unsafe { rmsnorm_f32_row_neon(x, weight, y, eps) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    rmsnorm_f32_row_scalar(x, weight, y, eps);
}

/// AArch64 NEON `rmsnorm_f32_row`. Pass 1 accumulates the sum of squares
/// with `vfmaq` (4 lanes) + scalar tail; the `inv = 1/sqrt(mean+eps)`
/// reciprocal is computed in scalar f32 identically to the reference (so
/// the normalization factor matches bit-for-bit). Pass 2 writes
/// `y = x * inv * w`. NEON baseline.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn rmsnorm_f32_row_neon(x: &[f32], weight: &[f32], y: &mut [f32], eps: f32) {
    use std::arch::aarch64::*;
    let d = x.len();
    let n4 = d & !3;
    let xp = x.as_ptr();
    let wp = weight.as_ptr();
    let yp = y.as_mut_ptr();

    // Pass 1: sum of squares.
    let mut acc = vdupq_n_f32(0.0);
    let mut p = 0;
    while p < n4 {
        let v = vld1q_f32(xp.add(p));
        acc = vfmaq_f32(acc, v, v);
        p += 4;
    }
    let mut sum_sq = vaddvq_f32(acc);
    while p < d {
        let v = *xp.add(p);
        sum_sq += v * v;
        p += 1;
    }

    let inv = 1.0 / (sum_sq / d as f32 + eps).sqrt();
    let inv_b = vdupq_n_f32(inv);

    // Pass 2: y = x * inv * w.
    let mut p = 0;
    while p < n4 {
        let xv = vld1q_f32(xp.add(p));
        let wv = vld1q_f32(wp.add(p));
        vst1q_f32(yp.add(p), vmulq_f32(vmulq_f32(xv, inv_b), wv));
        p += 4;
    }
    while p < d {
        *yp.add(p) = *xp.add(p) * inv * *wp.add(p);
        p += 1;
    }
}

/// Scalar fallback / parity reference for `rmsnorm_f32_row`.
pub(crate) fn rmsnorm_f32_row_scalar(x: &[f32], weight: &[f32], y: &mut [f32], eps: f32) {
    let d = x.len();
    let mut sum_sq = 0.0f32;
    for v in x {
        sum_sq += v * v;
    }
    let inv = 1.0 / (sum_sq / d as f32 + eps).sqrt();
    for ((dst, src), w) in y.iter_mut().zip(x.iter()).zip(weight.iter()) {
        *dst = src * inv * w;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn rmsnorm_f32_row_avx2(x: &[f32], weight: &[f32], y: &mut [f32], eps: f32) {
    use std::arch::x86_64::*;
    let d = x.len();
    let n8 = d & !7;
    let xp = x.as_ptr();
    let wp = weight.as_ptr();
    let yp = y.as_mut_ptr();

    // Pass 1: sum of squares.
    let mut acc = _mm256_setzero_ps();
    let mut p = 0;
    while p < n8 {
        let v = _mm256_loadu_ps(xp.add(p));
        acc = _mm256_fmadd_ps(v, v, acc);
        p += 8;
    }
    // Horizontal reduce of 8 lanes.
    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps::<1>(acc);
    let s128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(s128);
    let sums = _mm_add_ps(s128, shuf);
    let shuf = _mm_movehl_ps(shuf, sums);
    let sums = _mm_add_ss(sums, shuf);
    let mut sum_sq = _mm_cvtss_f32(sums);
    while p < d {
        let v = *xp.add(p);
        sum_sq += v * v;
        p += 1;
    }

    let inv = 1.0 / (sum_sq / d as f32 + eps).sqrt();
    let inv_b = _mm256_set1_ps(inv);

    // Pass 2: y = x * inv * w.
    let mut p = 0;
    while p < n8 {
        let xv = _mm256_loadu_ps(xp.add(p));
        let wv = _mm256_loadu_ps(wp.add(p));
        let scaled = _mm256_mul_ps(xv, inv_b);
        _mm256_storeu_ps(yp.add(p), _mm256_mul_ps(scaled, wv));
        p += 8;
    }
    while p < d {
        *yp.add(p) = *xp.add(p) * inv * *wp.add(p);
        p += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f")]
unsafe fn rmsnorm_f32_row_avx512f(x: &[f32], weight: &[f32], y: &mut [f32], eps: f32) {
    use std::arch::x86_64::*;
    let d = x.len();
    let n16 = d & !15;
    let xp = x.as_ptr();
    let wp = weight.as_ptr();
    let yp = y.as_mut_ptr();

    // Pass 1: sum of squares.
    let mut acc = _mm512_setzero_ps();
    let mut p = 0;
    while p < n16 {
        let v = _mm512_loadu_ps(xp.add(p));
        acc = _mm512_fmadd_ps(v, v, acc);
        p += 16;
    }
    let mut sum_sq = _mm512_reduce_add_ps(acc);
    while p < d {
        let v = *xp.add(p);
        sum_sq += v * v;
        p += 1;
    }

    let inv = 1.0 / (sum_sq / d as f32 + eps).sqrt();
    let inv_b = _mm512_set1_ps(inv);

    // Pass 2: y = x * inv * w.
    let mut p = 0;
    while p < n16 {
        let xv = _mm512_loadu_ps(xp.add(p));
        let wv = _mm512_loadu_ps(wp.add(p));
        let scaled = _mm512_mul_ps(xv, inv_b);
        _mm512_storeu_ps(yp.add(p), _mm512_mul_ps(scaled, wv));
        p += 16;
    }
    while p < d {
        *yp.add(p) = *xp.add(p) * inv * *wp.add(p);
        p += 1;
    }
}

/// Multi-row RMSNorm operating on a contiguous `[n_rows, d]` buffer.
pub fn rmsnorm_f32(x: &[f32], weight: &[f32], y: &mut [f32], eps: f32) {
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len() % weight.len(), 0);
    let d = weight.len();
    for (row_x, row_y) in x.chunks_exact(d).zip(y.chunks_exact_mut(d)) {
        rmsnorm_f32_row(row_x, weight, row_y, eps);
    }
}

/// SiLU(x) * y, fused for SwiGLU's `down(silu(gate) * up)`.
///
/// `expf` is the bottleneck — the surrounding mul/add chain doesn't
/// auto-vectorize because the compiler can't lift `expf` out of the
/// loop. The AVX2 path uses [`expf_approx_avx2`] (~2e-6 relative
/// error) and hits ~6× scalar throughput on AVX2 hosts.
#[inline]
pub fn silu_mul_f32(x: &[f32], y: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), out.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime feature detection above.
            unsafe { silu_mul_f32_avx2(x, y, out) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64.
        unsafe { silu_mul_f32_neon(x, y, out) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    silu_mul_f32_scalar(x, y, out);
}

/// AArch64 NEON SiLU-gate: `out = (x / (1 + exp(-x))) * y`, 4 lanes at a
/// time with [`expf_approx_neon`], scalar (`libm` exp) tail. Mirrors the
/// AVX2 path; same tolerance contract vs the scalar reference.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn silu_mul_f32_neon(x: &[f32], y: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;
    let n = x.len();
    let n4 = n & !3;
    let xp = x.as_ptr();
    let yp = y.as_ptr();
    let op = out.as_mut_ptr();
    let one = vdupq_n_f32(1.0);
    let zero = vdupq_n_f32(0.0);
    let mut p = 0;
    while p < n4 {
        let xv = vld1q_f32(xp.add(p));
        let yv = vld1q_f32(yp.add(p));
        // silu(x) = x / (1 + exp(-x))
        let e = expf_approx_neon(vsubq_f32(zero, xv));
        let silu = vdivq_f32(xv, vaddq_f32(one, e));
        vst1q_f32(op.add(p), vmulq_f32(silu, yv));
        p += 4;
    }
    while p < n {
        let xv = *xp.add(p);
        *op.add(p) = (xv / (1.0 + (-xv).exp())) * *yp.add(p);
        p += 1;
    }
}

pub(crate) fn silu_mul_f32_scalar(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &xv), &yv) in out.iter_mut().zip(x.iter()).zip(y.iter()) {
        let silu = xv / (1.0 + (-xv).exp());
        *o = silu * yv;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn silu_mul_f32_avx2(x: &[f32], y: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = x.len();
    let n8 = n & !7;
    let xp = x.as_ptr();
    let yp = y.as_ptr();
    let op = out.as_mut_ptr();

    let one = _mm256_set1_ps(1.0);
    let mut p = 0;
    while p < n8 {
        let xv = _mm256_loadu_ps(xp.add(p));
        let yv = _mm256_loadu_ps(yp.add(p));
        // silu(x) = x / (1 + exp(-x))
        let neg_x = _mm256_sub_ps(_mm256_setzero_ps(), xv);
        let e = expf_approx_avx2(neg_x);
        let denom = _mm256_add_ps(one, e);
        let silu = _mm256_div_ps(xv, denom);
        _mm256_storeu_ps(op.add(p), _mm256_mul_ps(silu, yv));
        p += 8;
    }
    while p < n {
        let xv = *xp.add(p);
        let yv = *yp.add(p);
        let silu = xv / (1.0 + (-xv).exp());
        *op.add(p) = silu * yv;
        p += 1;
    }
}

/// Elementwise add: `a += b`. AVX2 fast path when available.
#[inline]
pub fn add_inplace_f32(a: &mut [f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        // SAFETY: runtime feature detection above.
        unsafe { add_inplace_f32_avx2(a, b) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64.
        unsafe { add_inplace_f32_neon(a, b) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    for (lhs, rhs) in a.iter_mut().zip(b.iter()) {
        *lhs += *rhs;
    }
}

/// AArch64 NEON `a += b` over f32 slices. 4-lane body + scalar tail. Add
/// is exact in f32, so this is bit-identical to the scalar reference.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn add_inplace_f32_neon(a: &mut [f32], b: &[f32]) {
    use std::arch::aarch64::*;
    let n = a.len();
    let n4 = n & !3;
    let mut p = 0;
    while p < n4 {
        let av = vld1q_f32(a.as_ptr().add(p));
        let bv = vld1q_f32(b.as_ptr().add(p));
        vst1q_f32(a.as_mut_ptr().add(p), vaddq_f32(av, bv));
        p += 4;
    }
    while p < n {
        a[p] += b[p];
        p += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2")]
unsafe fn add_inplace_f32_avx2(a: &mut [f32], b: &[f32]) {
    use std::arch::x86_64::*;
    let n = a.len();
    let n8 = n & !7;
    let mut p = 0;
    while p < n8 {
        let av = _mm256_loadu_ps(a.as_ptr().add(p));
        let bv = _mm256_loadu_ps(b.as_ptr().add(p));
        _mm256_storeu_ps(a.as_mut_ptr().add(p), _mm256_add_ps(av, bv));
        p += 8;
    }
    while p < n {
        a[p] += b[p];
        p += 1;
    }
}

/// Convert a slice of `f16` to `f32`.
pub fn f16_to_f32(src: &[f16], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len());
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = s.to_f32();
    }
}

/// Embedding lookup: gather rows `ids[..]` from `table: [vocab, d]` F16 into
/// `out: [n_ids, d]` F32.
pub fn embed_lookup_f16_to_f32(table: &[f16], ids: &[i32], out: &mut [f32], d: usize) {
    debug_assert_eq!(out.len(), ids.len() * d);
    for (i, &id) in ids.iter().enumerate() {
        let row_start = (id as usize) * d;
        let src = &table[row_start..row_start + d];
        let dst = &mut out[i * d..(i + 1) * d];
        f16_to_f32(src, dst);
    }
}

/// Rotary position embeddings (neox / half-split variant: pairs are
/// `(x[i], x[i + head_dim/2])`). Operates over `n_heads` heads laid out
/// contiguously in `x: [n_heads * head_dim]`.
///
/// `theta` defaults to 10000 in Llama-2/3/Qwen2; check the GGUF
/// `*.rope.freq_base` for the model-specific value.
#[inline]
pub fn rope_inplace_neox(x: &mut [f32], n_heads: usize, head_dim: usize, pos: u32, theta: f32) {
    debug_assert_eq!(x.len(), n_heads * head_dim);
    debug_assert_eq!(head_dim % 2, 0);
    let half = head_dim / 2;

    // Precompute (cos, sin) per i once. The trig is shared across
    // all heads at this token, so amortizing it over n_heads is a
    // 30-100× reduction in expf/cosf/sinf calls vs the inner-loop
    // form. Stack-buffer it when `half <= 128` (covers head_dim ≤
    // 256, the universe of v1-targeted models) to avoid touching
    // the allocator on the per-token path.
    const STACK_MAX_HALF: usize = 128;
    let mut stack_cs = [0.0f32; 2 * STACK_MAX_HALF];
    let heap_cs: Vec<f32>;
    let cs: &[f32] = if half <= STACK_MAX_HALF {
        for i in 0..half {
            let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
            let p = pos as f32 * freq;
            stack_cs[2 * i] = p.cos();
            stack_cs[2 * i + 1] = p.sin();
        }
        &stack_cs[..2 * half]
    } else {
        heap_cs = (0..half)
            .flat_map(|i| {
                let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
                let p = pos as f32 * freq;
                [p.cos(), p.sin()]
            })
            .collect();
        &heap_cs[..]
    };

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime feature detection above.
            unsafe { rope_inplace_neox_avx2(x, n_heads, head_dim, half, cs) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64.
        unsafe { rope_inplace_neox_neon(x, n_heads, head_dim, half, cs) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    rope_inplace_neox_scalar(x, n_heads, head_dim, half, cs);
}

pub(crate) fn rope_inplace_neox_scalar(
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    half: usize,
    cs: &[f32],
) {
    for h in 0..n_heads {
        let base = h * head_dim;
        for i in 0..half {
            let cos = cs[2 * i];
            let sin = cs[2 * i + 1];
            let x0 = x[base + i];
            let x1 = x[base + i + half];
            x[base + i] = x0 * cos - x1 * sin;
            x[base + i + half] = x0 * sin + x1 * cos;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn rope_inplace_neox_avx2(
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    half: usize,
    cs: &[f32],
) {
    use std::arch::x86_64::*;
    // Materialize separate cos / sin slabs so the inner loop does
    // contiguous 8-wide loads instead of interleaved gathers. This is
    // a one-time cost amortized across n_heads heads.
    let mut cos_buf = vec![0f32; half];
    let mut sin_buf = vec![0f32; half];
    for i in 0..half {
        cos_buf[i] = cs[2 * i];
        sin_buf[i] = cs[2 * i + 1];
    }
    let cosp = cos_buf.as_ptr();
    let sinp = sin_buf.as_ptr();

    let n8 = half & !7;
    let xp = x.as_mut_ptr();
    for h in 0..n_heads {
        let base = h * head_dim;
        let mut i = 0;
        while i < n8 {
            let cos_v = _mm256_loadu_ps(cosp.add(i));
            let sin_v = _mm256_loadu_ps(sinp.add(i));
            let x0 = _mm256_loadu_ps(xp.add(base + i));
            let x1 = _mm256_loadu_ps(xp.add(base + i + half));
            // y0 = x0*cos - x1*sin = fmsub(x0, cos, x1*sin)
            let x1sin = _mm256_mul_ps(x1, sin_v);
            let y0 = _mm256_fmsub_ps(x0, cos_v, x1sin);
            // y1 = x0*sin + x1*cos = fmadd(x1, cos, x0*sin)
            let x0sin = _mm256_mul_ps(x0, sin_v);
            let y1 = _mm256_fmadd_ps(x1, cos_v, x0sin);
            _mm256_storeu_ps(xp.add(base + i), y0);
            _mm256_storeu_ps(xp.add(base + i + half), y1);
            i += 8;
        }
        while i < half {
            let cos = *cosp.add(i);
            let sin = *sinp.add(i);
            let x0 = *xp.add(base + i);
            let x1 = *xp.add(base + i + half);
            *xp.add(base + i) = x0 * cos - x1 * sin;
            *xp.add(base + i + half) = x0 * sin + x1 * cos;
            i += 1;
        }
    }
}

/// AArch64 NEON NeoX RoPE: 4-lane twin of [`rope_inplace_neox_avx2`]. Same
/// pre-split cos/sin slabs for contiguous loads, then per head:
///   y0 = x0*cos - x1*sin  (vfmsq: a - b*c)
///   y1 = x0*sin + x1*cos  (vfmaq: a + b*c)
/// over the lower `half` lanes, scalar tail. Bit-exact to scalar up to FMA
/// rounding (the AVX2 path uses the same fused form).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn rope_inplace_neox_neon(
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    half: usize,
    cs: &[f32],
) {
    use std::arch::aarch64::*;
    // Materialize separate cos / sin slabs so the inner loop does contiguous
    // 4-wide loads (one-time cost amortized across n_heads heads).
    let mut cos_buf = vec![0f32; half];
    let mut sin_buf = vec![0f32; half];
    for i in 0..half {
        cos_buf[i] = cs[2 * i];
        sin_buf[i] = cs[2 * i + 1];
    }
    let cosp = cos_buf.as_ptr();
    let sinp = sin_buf.as_ptr();

    let n4 = half & !3;
    let xp = x.as_mut_ptr();
    for h in 0..n_heads {
        let base = h * head_dim;
        let mut i = 0;
        while i < n4 {
            let cos_v = vld1q_f32(cosp.add(i));
            let sin_v = vld1q_f32(sinp.add(i));
            let x0 = vld1q_f32(xp.add(base + i));
            let x1 = vld1q_f32(xp.add(base + i + half));
            // y0 = x0*cos - x1*sin = (x0*cos) - x1*sin
            let x0cos = vmulq_f32(x0, cos_v);
            let y0 = vfmsq_f32(x0cos, x1, sin_v);
            // y1 = x0*sin + x1*cos = (x0*sin) + x1*cos
            let x0sin = vmulq_f32(x0, sin_v);
            let y1 = vfmaq_f32(x0sin, x1, cos_v);
            vst1q_f32(xp.add(base + i), y0);
            vst1q_f32(xp.add(base + i + half), y1);
            i += 4;
        }
        while i < half {
            let cos = *cosp.add(i);
            let sin = *sinp.add(i);
            let x0 = *xp.add(base + i);
            let x1 = *xp.add(base + i + half);
            *xp.add(base + i) = x0 * cos - x1 * sin;
            *xp.add(base + i + half) = x0 * sin + x1 * cos;
            i += 1;
        }
    }
}

/// Rotary position embeddings (interleaved / "normal" variant: pairs are
/// `(x[2i], x[2i+1])`). Used by ggml's original Llama path
/// (`LLAMA_ROPE_TYPE_NORM`). For HF-converted GGUFs, the neox variant above
/// is the right one; this is here as an A/B option to verify which convention
/// matches a specific GGUF.
pub fn rope_inplace_interleaved(
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    pos: u32,
    theta: f32,
) {
    debug_assert_eq!(x.len(), n_heads * head_dim);
    debug_assert_eq!(head_dim % 2, 0);
    for h in 0..n_heads {
        let base = h * head_dim;
        for i in 0..head_dim / 2 {
            let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
            let p = pos as f32 * freq;
            let cos = p.cos();
            let sin = p.sin();
            let x0 = x[base + 2 * i];
            let x1 = x[base + 2 * i + 1];
            x[base + 2 * i] = x0 * cos - x1 * sin;
            x[base + 2 * i + 1] = x0 * sin + x1 * cos;
        }
    }
}

/// Numerically-stable in-place softmax over the slice.
///
/// Dispatches to AVX-512 / AVX2 SIMD paths when available. The
/// polynomial `expf` approximation used by the SIMD variants is
/// accurate to within ~2e-6 relative error over the softmax-relevant
/// domain (after the standard max-subtraction trick, inputs land in
/// `(-∞, 0]`). The downstream normalization absorbs the small
/// per-element error to ≤1e-6 abs in the output probabilities.
#[inline]
pub fn softmax_f32_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: runtime feature detection above.
            unsafe { softmax_f32_inplace_avx512f(x) };
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime feature detection above.
            unsafe { softmax_f32_inplace_avx2(x) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64.
        unsafe { softmax_f32_inplace_neon(x) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    softmax_f32_inplace_scalar(x);
}

/// AArch64 NEON in-place softmax: 3 passes (max / exp-and-sum / normalize),
/// 4 lanes per step with a scalar tail. The bulk uses [`expf_approx_neon`]
/// and the tail uses `libm` exp, exactly like the AVX2 path; the
/// post-normalization absorbs the ~2e-6 approximation error. NEON baseline.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn softmax_f32_inplace_neon(x: &mut [f32]) {
    use std::arch::aarch64::*;
    let n = x.len();
    let ptr = x.as_mut_ptr();
    let n4 = n & !3;

    // Pass 1: max.
    let mut max_v = vdupq_n_f32(f32::NEG_INFINITY);
    let mut p = 0;
    while p < n4 {
        max_v = vmaxq_f32(max_v, vld1q_f32(ptr.add(p)));
        p += 4;
    }
    let mut max_s = vmaxvq_f32(max_v);
    while p < n {
        let v = *ptr.add(p);
        if v > max_s {
            max_s = v;
        }
        p += 1;
    }

    // Pass 2: exp(v - max), accumulate sum.
    let max_b = vdupq_n_f32(max_s);
    let mut sum_v = vdupq_n_f32(0.0);
    let mut p = 0;
    while p < n4 {
        let e = expf_approx_neon(vsubq_f32(vld1q_f32(ptr.add(p)), max_b));
        vst1q_f32(ptr.add(p), e);
        sum_v = vaddq_f32(sum_v, e);
        p += 4;
    }
    let mut sum_s = vaddvq_f32(sum_v);
    while p < n {
        let e = (*ptr.add(p) - max_s).exp();
        *ptr.add(p) = e;
        sum_s += e;
        p += 1;
    }

    // Pass 3: normalize.
    let inv_b = vdupq_n_f32(1.0 / sum_s);
    let mut p = 0;
    while p < n4 {
        vst1q_f32(ptr.add(p), vmulq_f32(vld1q_f32(ptr.add(p)), inv_b));
        p += 4;
    }
    let inv = 1.0 / sum_s;
    while p < n {
        *ptr.add(p) *= inv;
        p += 1;
    }
}

/// Scalar-only softmax — fallback path + parity reference for the
/// SIMD variants. Kept `pub(crate)` so the parity test can drive it
/// directly without dispatching.
pub(crate) fn softmax_f32_inplace_scalar(x: &mut [f32]) {
    let mut max = f32::NEG_INFINITY;
    for v in x.iter() {
        if *v > max {
            max = *v;
        }
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// Polynomial `expf` approximation over `__m256`. Cephes-style range
/// reduction `x = n·ln2 + r`, `r ∈ [-ln2/2, ln2/2]`, then a 5-term
/// Horner polynomial of the Taylor series. Clamps inputs to
/// `[-88, 88]` to avoid `inf`/`NaN` in the bit-scale step; the
/// softmax + SiLU call sites never exceed those bounds in practice.
///
/// Max relative error ~2e-6 over the reduced range; sufficient for
/// softmax (post-normalization) and SiLU (whose output is `x·σ(x)`
/// and rounds to f32 anyway).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
#[inline]
unsafe fn expf_approx_avx2(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let max_in = _mm256_set1_ps(88.0);
    let min_in = _mm256_set1_ps(-88.0);
    let x = _mm256_min_ps(_mm256_max_ps(x, min_in), max_in);

    // n = round(x * log2(e))
    let log2e = _mm256_set1_ps(std::f32::consts::LOG2_E);
    let n = _mm256_round_ps(
        _mm256_mul_ps(x, log2e),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
    );
    // r = x - n*ln2 using a Cephes-style 2-piece decomposition for
    // extra precision (the big chunk and the remainder applied
    // separately to fmadd, not their sum).
    let c1 = _mm256_set1_ps(0.693_359_375);
    let c2 = _mm256_set1_ps(-2.121_944_4e-4);
    let r = _mm256_fnmadd_ps(n, c1, x);
    let r = _mm256_fnmadd_ps(n, c2, r);

    // exp(r) Horner polynomial (5 terms; max abs error ~2e-6 on the
    // reduced range).
    let p = _mm256_set1_ps(1.0 / 120.0);
    let p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0 / 24.0));
    let p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0 / 6.0));
    let p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5));
    let p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0));
    let p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0));

    // 2^n via bit cast of `(n + 127) << 23` into the f32 exponent
    // field. n is in range [-127, 127] after the clamp, so the
    // biased value stays in [0, 254] (denormal/inf saturation
    // handled by the input clamp above).
    let n_i = _mm256_cvtps_epi32(n);
    let n_biased = _mm256_add_epi32(n_i, _mm256_set1_epi32(127));
    let pow2n = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(n_biased));
    _mm256_mul_ps(p, pow2n)
}

/// Polynomial `expf` for AVX-512. Same algorithm as the AVX2 variant
/// but over 16-wide lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f")]
#[inline]
unsafe fn expf_approx_avx512f(x: std::arch::x86_64::__m512) -> std::arch::x86_64::__m512 {
    use std::arch::x86_64::*;
    let max_in = _mm512_set1_ps(88.0);
    let min_in = _mm512_set1_ps(-88.0);
    let x = _mm512_min_ps(_mm512_max_ps(x, min_in), max_in);

    let log2e = _mm512_set1_ps(std::f32::consts::LOG2_E);
    // _mm512_roundscale_ps with imm=0 rounds to nearest-even.
    let n = _mm512_roundscale_ps::<0>(_mm512_mul_ps(x, log2e));
    let c1 = _mm512_set1_ps(0.693_359_375);
    let c2 = _mm512_set1_ps(-2.121_944_4e-4);
    let r = _mm512_fnmadd_ps(n, c1, x);
    let r = _mm512_fnmadd_ps(n, c2, r);

    let p = _mm512_set1_ps(1.0 / 120.0);
    let p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0 / 24.0));
    let p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0 / 6.0));
    let p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(0.5));
    let p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0));
    let p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0));

    let n_i = _mm512_cvtps_epi32(n);
    let n_biased = _mm512_add_epi32(n_i, _mm512_set1_epi32(127));
    let pow2n = _mm512_castsi512_ps(_mm512_slli_epi32::<23>(n_biased));
    _mm512_mul_ps(p, pow2n)
}

/// AArch64 NEON `expf` approximation over `float32x4_t` — the exact same
/// Cephes range-reduction + 5-term Horner polynomial as [`expf_approx_avx2`]
/// (same constants, so the same ~2e-6 relative error), using `vrndnq_f32`
/// for the round-to-nearest, `vfmsq_f32` (`a - b*c`) for the range
/// reduction, `vfmaq_f32` for the polynomial, and a `<< 23` exponent bit
/// trick for `2^n`. Shared by the NEON SiLU / softmax kernels.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn expf_approx_neon(x: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    let x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-88.0)), vdupq_n_f32(88.0));
    // n = round(x * log2(e)); r = x - n*ln2 (2-piece for precision).
    let n = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(std::f32::consts::LOG2_E)));
    let r = vfmsq_f32(x, n, vdupq_n_f32(0.693_359_375));
    let r = vfmsq_f32(r, n, vdupq_n_f32(-2.121_944_4e-4));
    // exp(r) Horner polynomial (5 terms).
    let p = vdupq_n_f32(1.0 / 120.0);
    let p = vfmaq_f32(vdupq_n_f32(1.0 / 24.0), p, r);
    let p = vfmaq_f32(vdupq_n_f32(1.0 / 6.0), p, r);
    let p = vfmaq_f32(vdupq_n_f32(0.5), p, r);
    let p = vfmaq_f32(vdupq_n_f32(1.0), p, r);
    let p = vfmaq_f32(vdupq_n_f32(1.0), p, r);
    // 2^n via (n + 127) << 23 in the f32 exponent field.
    let n_biased = vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127));
    let pow2n = vreinterpretq_f32_s32(vshlq_n_s32::<23>(n_biased));
    vmulq_f32(p, pow2n)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn softmax_f32_inplace_avx2(x: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = x.len();
    let ptr = x.as_mut_ptr();

    // Pass 1: horizontal max. 8 lanes per iter; scalar tail.
    let n8 = n & !7;
    let mut max_v = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut p = 0;
    while p < n8 {
        max_v = _mm256_max_ps(max_v, _mm256_loadu_ps(ptr.add(p)));
        p += 8;
    }
    // Horizontal max over the 8 lanes.
    let mut max_arr = [f32::NEG_INFINITY; 8];
    _mm256_storeu_ps(max_arr.as_mut_ptr(), max_v);
    let mut max_s = max_arr[0];
    for &v in &max_arr[1..] {
        if v > max_s {
            max_s = v;
        }
    }
    while p < n {
        let v = *ptr.add(p);
        if v > max_s {
            max_s = v;
        }
        p += 1;
    }

    // Pass 2: write exp(v - max) and accumulate sum.
    let max_b = _mm256_set1_ps(max_s);
    let mut sum_v = _mm256_setzero_ps();
    let mut p = 0;
    while p < n8 {
        let v = _mm256_sub_ps(_mm256_loadu_ps(ptr.add(p)), max_b);
        let e = expf_approx_avx2(v);
        _mm256_storeu_ps(ptr.add(p), e);
        sum_v = _mm256_add_ps(sum_v, e);
        p += 8;
    }
    let mut sum_arr = [0.0f32; 8];
    _mm256_storeu_ps(sum_arr.as_mut_ptr(), sum_v);
    let mut sum_s = sum_arr.iter().sum::<f32>();
    while p < n {
        let e = (*ptr.add(p) - max_s).exp();
        *ptr.add(p) = e;
        sum_s += e;
        p += 1;
    }

    // Pass 3: normalize.
    let inv = 1.0 / sum_s;
    let inv_b = _mm256_set1_ps(inv);
    let mut p = 0;
    while p < n8 {
        let v = _mm256_loadu_ps(ptr.add(p));
        _mm256_storeu_ps(ptr.add(p), _mm256_mul_ps(v, inv_b));
        p += 8;
    }
    while p < n {
        *ptr.add(p) *= inv;
        p += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f")]
unsafe fn softmax_f32_inplace_avx512f(x: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = x.len();
    let ptr = x.as_mut_ptr();

    // Pass 1: horizontal max.
    let n16 = n & !15;
    let mut max_v = _mm512_set1_ps(f32::NEG_INFINITY);
    let mut p = 0;
    while p < n16 {
        max_v = _mm512_max_ps(max_v, _mm512_loadu_ps(ptr.add(p)));
        p += 16;
    }
    let mut max_s = _mm512_reduce_max_ps(max_v);
    while p < n {
        let v = *ptr.add(p);
        if v > max_s {
            max_s = v;
        }
        p += 1;
    }

    // Pass 2: exp + sum.
    let max_b = _mm512_set1_ps(max_s);
    let mut sum_v = _mm512_setzero_ps();
    let mut p = 0;
    while p < n16 {
        let v = _mm512_sub_ps(_mm512_loadu_ps(ptr.add(p)), max_b);
        let e = expf_approx_avx512f(v);
        _mm512_storeu_ps(ptr.add(p), e);
        sum_v = _mm512_add_ps(sum_v, e);
        p += 16;
    }
    let mut sum_s = _mm512_reduce_add_ps(sum_v);
    while p < n {
        let e = (*ptr.add(p) - max_s).exp();
        *ptr.add(p) = e;
        sum_s += e;
        p += 1;
    }

    // Pass 3: normalize.
    let inv = 1.0 / sum_s;
    let inv_b = _mm512_set1_ps(inv);
    let mut p = 0;
    while p < n16 {
        let v = _mm512_loadu_ps(ptr.add(p));
        _mm512_storeu_ps(ptr.add(p), _mm512_mul_ps(v, inv_b));
        p += 16;
    }
    while p < n {
        *ptr.add(p) *= inv;
        p += 1;
    }
}

/// F3: batched F32 → F16-bits packing. Used by the USM upload paths
/// (`try_silu_mul_usm_f32`, `try_rmsnorm_usm_f32`, `try_rope_usm_f32`)
/// which marshal F32 inputs into a F16-bit USM buffer once per call.
/// The scalar version (`half::f16::from_f32(v).to_bits()` in a loop)
/// is the per-element baseline; on x86_64 with F16C we vectorize
/// 8 lanes per pass via `_mm_cvtps_ph` → ~5× faster than scalar
/// on the F16-conversion side and saves ~µs per token across each
/// of the ~32 layers × 2 ops/layer that take this path.
#[inline]
pub fn f32_to_f16_bits(src: &[f32], dst: &mut [u16]) {
    let n = src.len().min(dst.len());
    #[cfg(target_arch = "x86_64")]
    {
        if n >= 8 && is_x86_feature_detected!("f16c") && is_x86_feature_detected!("avx") {
            // SAFETY: runtime feature detection above.
            unsafe { f32_to_f16_bits_f16c(&src[..n], &mut dst[..n]) };
            return;
        }
    }
    f32_to_f16_bits_scalar(&src[..n], &mut dst[..n]);
}

pub(crate) fn f32_to_f16_bits_scalar(src: &[f32], dst: &mut [u16]) {
    for (i, &v) in src.iter().enumerate() {
        dst[i] = half::f16::from_f32(v).to_bits();
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
unsafe fn f32_to_f16_bits_f16c(src: &[f32], dst: &mut [u16]) {
    use std::arch::x86_64::*;
    let n = src.len();
    let n8 = n & !7;
    let s_ptr = src.as_ptr();
    let d_ptr = dst.as_mut_ptr();
    let mut p = 0;
    while p < n8 {
        let v = _mm256_loadu_ps(s_ptr.add(p));
        // Rounding mode 0 = round-to-nearest-even (matches
        // `half::f16::from_f32`); MXCSR-driven mode would be 4, but
        // the immediate is constrained to 3 bits so we use the
        // explicit per-instruction control.
        let h = _mm256_cvtps_ph::<0>(v);
        _mm_storeu_si128(d_ptr.add(p) as *mut __m128i, h);
        p += 8;
    }
    // Scalar tail.
    while p < n {
        *d_ptr.add(p) = half::f16::from_f32(*s_ptr.add(p)).to_bits();
        p += 1;
    }
}

/// F1: fused temperature scale + softmax. Mathematically equivalent
/// to `for v in x { *v *= inv_temp } softmax(x)` but folds the
/// temperature multiply into pass 1's max scan — three vocab
/// passes instead of four. On a 128K-vocab Qwen2.5 sample at
/// temperature 0.7 this saves ~25% of softmax wall time per token.
///
/// Caller passes the **reciprocal** temperature so the hot loop
/// is one FMA instead of a divide.
#[inline]
pub fn fused_temp_softmax_inplace(x: &mut [f32], inv_temp: f32) {
    if x.is_empty() {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: runtime feature detection above.
            unsafe { fused_temp_softmax_inplace_avx512f(x, inv_temp) };
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime feature detection above.
            unsafe { fused_temp_softmax_inplace_avx2(x, inv_temp) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64.
        unsafe { fused_temp_softmax_inplace_neon(x, inv_temp) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    fused_temp_softmax_inplace_scalar(x, inv_temp);
}

/// AArch64 NEON fused temperature-scaled softmax: pass 1 writes
/// `x * inv_temp` back while tracking the max, then the standard
/// exp-and-sum / normalize passes (bulk via [`expf_approx_neon`], scalar
/// `libm` exp tail). Mirrors the AVX2 path; NEON baseline.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn fused_temp_softmax_inplace_neon(x: &mut [f32], inv_temp: f32) {
    use std::arch::aarch64::*;
    let n = x.len();
    let ptr = x.as_mut_ptr();
    let n4 = n & !3;
    let inv_t_b = vdupq_n_f32(inv_temp);

    // Pass 1: scale by inv_temp in place + track max.
    let mut max_v = vdupq_n_f32(f32::NEG_INFINITY);
    let mut p = 0;
    while p < n4 {
        let v = vmulq_f32(vld1q_f32(ptr.add(p)), inv_t_b);
        vst1q_f32(ptr.add(p), v);
        max_v = vmaxq_f32(max_v, v);
        p += 4;
    }
    let mut max_s = vmaxvq_f32(max_v);
    while p < n {
        let v = *ptr.add(p) * inv_temp;
        *ptr.add(p) = v;
        if v > max_s {
            max_s = v;
        }
        p += 1;
    }

    // Pass 2: exp(v - max), accumulate sum.
    let max_b = vdupq_n_f32(max_s);
    let mut sum_v = vdupq_n_f32(0.0);
    let mut p = 0;
    while p < n4 {
        let e = expf_approx_neon(vsubq_f32(vld1q_f32(ptr.add(p)), max_b));
        vst1q_f32(ptr.add(p), e);
        sum_v = vaddq_f32(sum_v, e);
        p += 4;
    }
    let mut sum_s = vaddvq_f32(sum_v);
    while p < n {
        let e = (*ptr.add(p) - max_s).exp();
        *ptr.add(p) = e;
        sum_s += e;
        p += 1;
    }

    // Pass 3: normalize.
    let inv = 1.0 / sum_s;
    let inv_b = vdupq_n_f32(inv);
    let mut p = 0;
    while p < n4 {
        vst1q_f32(ptr.add(p), vmulq_f32(vld1q_f32(ptr.add(p)), inv_b));
        p += 4;
    }
    while p < n {
        *ptr.add(p) *= inv;
        p += 1;
    }
}

pub(crate) fn fused_temp_softmax_inplace_scalar(x: &mut [f32], inv_temp: f32) {
    let mut max = f32::NEG_INFINITY;
    for v in x.iter_mut() {
        *v *= inv_temp;
        if *v > max {
            max = *v;
        }
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn fused_temp_softmax_inplace_avx2(x: &mut [f32], inv_temp: f32) {
    use std::arch::x86_64::*;
    let n = x.len();
    let ptr = x.as_mut_ptr();
    let inv_t_b = _mm256_set1_ps(inv_temp);

    // Pass 1: write `x * inv_temp` back AND track lane-wise max.
    let n8 = n & !7;
    let mut max_v = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut p = 0;
    while p < n8 {
        let v = _mm256_mul_ps(_mm256_loadu_ps(ptr.add(p)), inv_t_b);
        _mm256_storeu_ps(ptr.add(p), v);
        max_v = _mm256_max_ps(max_v, v);
        p += 8;
    }
    let mut max_arr = [f32::NEG_INFINITY; 8];
    _mm256_storeu_ps(max_arr.as_mut_ptr(), max_v);
    let mut max_s = max_arr[0];
    for &v in &max_arr[1..] {
        if v > max_s {
            max_s = v;
        }
    }
    while p < n {
        let v = *ptr.add(p) * inv_temp;
        *ptr.add(p) = v;
        if v > max_s {
            max_s = v;
        }
        p += 1;
    }

    // Pass 2: exp + sum.
    let max_b = _mm256_set1_ps(max_s);
    let mut sum_v = _mm256_setzero_ps();
    let mut p = 0;
    while p < n8 {
        let v = _mm256_sub_ps(_mm256_loadu_ps(ptr.add(p)), max_b);
        let e = expf_approx_avx2(v);
        _mm256_storeu_ps(ptr.add(p), e);
        sum_v = _mm256_add_ps(sum_v, e);
        p += 8;
    }
    let mut sum_arr = [0.0f32; 8];
    _mm256_storeu_ps(sum_arr.as_mut_ptr(), sum_v);
    let mut sum_s = sum_arr.iter().sum::<f32>();
    while p < n {
        let e = (*ptr.add(p) - max_s).exp();
        *ptr.add(p) = e;
        sum_s += e;
        p += 1;
    }

    // Pass 3: normalize.
    let inv = 1.0 / sum_s;
    let inv_b = _mm256_set1_ps(inv);
    let mut p = 0;
    while p < n8 {
        let v = _mm256_loadu_ps(ptr.add(p));
        _mm256_storeu_ps(ptr.add(p), _mm256_mul_ps(v, inv_b));
        p += 8;
    }
    while p < n {
        *ptr.add(p) *= inv;
        p += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f")]
unsafe fn fused_temp_softmax_inplace_avx512f(x: &mut [f32], inv_temp: f32) {
    use std::arch::x86_64::*;
    let n = x.len();
    let ptr = x.as_mut_ptr();
    let inv_t_b = _mm512_set1_ps(inv_temp);

    // Pass 1: write `x * inv_temp` back AND track max.
    let n16 = n & !15;
    let mut max_v = _mm512_set1_ps(f32::NEG_INFINITY);
    let mut p = 0;
    while p < n16 {
        let v = _mm512_mul_ps(_mm512_loadu_ps(ptr.add(p)), inv_t_b);
        _mm512_storeu_ps(ptr.add(p), v);
        max_v = _mm512_max_ps(max_v, v);
        p += 16;
    }
    let mut max_s = _mm512_reduce_max_ps(max_v);
    while p < n {
        let v = *ptr.add(p) * inv_temp;
        *ptr.add(p) = v;
        if v > max_s {
            max_s = v;
        }
        p += 1;
    }

    // Pass 2: exp + sum.
    let max_b = _mm512_set1_ps(max_s);
    let mut sum_v = _mm512_setzero_ps();
    let mut p = 0;
    while p < n16 {
        let v = _mm512_sub_ps(_mm512_loadu_ps(ptr.add(p)), max_b);
        let e = expf_approx_avx512f(v);
        _mm512_storeu_ps(ptr.add(p), e);
        sum_v = _mm512_add_ps(sum_v, e);
        p += 16;
    }
    let mut sum_s = _mm512_reduce_add_ps(sum_v);
    while p < n {
        let e = (*ptr.add(p) - max_s).exp();
        *ptr.add(p) = e;
        sum_s += e;
        p += 1;
    }

    // Pass 3: normalize.
    let inv = 1.0 / sum_s;
    let inv_b = _mm512_set1_ps(inv);
    let mut p = 0;
    while p < n16 {
        let v = _mm512_loadu_ps(ptr.add(p));
        _mm512_storeu_ps(ptr.add(p), _mm512_mul_ps(v, inv_b));
        p += 16;
    }
    while p < n {
        *ptr.add(p) *= inv;
        p += 1;
    }
}

/// Numerically-stable `log(sum(exp(x_i)))` over a slice. Returns
/// `f32::NEG_INFINITY` on an empty input. Used by the sampler's
/// `compute_logprobs` to derive per-token logprobs; called per
/// generated token on the OpenAI `logprobs`-enabled path, so the
/// SIMD variants pay back across long completions.
///
/// Math: `max(x) + log(sum(exp(x_i - max(x))))`. Two scalar passes
/// in the reference; SIMD path uses [`expf_approx_avx2`] /
/// [`expf_approx_avx512f`] and the same horizontal-reduction
/// primitives as `softmax_f32_inplace`.
#[inline]
pub fn log_sum_exp_f32(x: &[f32]) -> f32 {
    if x.is_empty() {
        return f32::NEG_INFINITY;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: runtime feature detection above.
            return unsafe { log_sum_exp_f32_avx512f(x) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime feature detection above.
            return unsafe { log_sum_exp_f32_avx2(x) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64.
        unsafe { log_sum_exp_f32_neon(x) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        log_sum_exp_f32_scalar(x)
    }
}

pub(crate) fn log_sum_exp_f32_scalar(x: &[f32]) -> f32 {
    let mut max = f32::NEG_INFINITY;
    for &v in x {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0f32;
    for &v in x {
        sum += (v - max).exp();
    }
    max + sum.ln()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn log_sum_exp_f32_avx2(x: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = x.len();
    let n8 = n & !7;
    let ptr = x.as_ptr();

    // Pass 1: horizontal max.
    let mut max_v = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut p = 0;
    while p < n8 {
        max_v = _mm256_max_ps(max_v, _mm256_loadu_ps(ptr.add(p)));
        p += 8;
    }
    let mut max_arr = [f32::NEG_INFINITY; 8];
    _mm256_storeu_ps(max_arr.as_mut_ptr(), max_v);
    let mut max_s = max_arr[0];
    for &v in &max_arr[1..] {
        if v > max_s {
            max_s = v;
        }
    }
    while p < n {
        let v = *ptr.add(p);
        if v > max_s {
            max_s = v;
        }
        p += 1;
    }

    // Pass 2: sum of exp(x - max).
    let max_b = _mm256_set1_ps(max_s);
    let mut sum_v = _mm256_setzero_ps();
    let mut p = 0;
    while p < n8 {
        let v = _mm256_sub_ps(_mm256_loadu_ps(ptr.add(p)), max_b);
        let e = expf_approx_avx2(v);
        sum_v = _mm256_add_ps(sum_v, e);
        p += 8;
    }
    let mut sum_arr = [0.0f32; 8];
    _mm256_storeu_ps(sum_arr.as_mut_ptr(), sum_v);
    let mut sum_s = sum_arr.iter().sum::<f32>();
    while p < n {
        sum_s += (*ptr.add(p) - max_s).exp();
        p += 1;
    }

    max_s + sum_s.ln()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f")]
unsafe fn log_sum_exp_f32_avx512f(x: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = x.len();
    let n16 = n & !15;
    let ptr = x.as_ptr();

    let mut max_v = _mm512_set1_ps(f32::NEG_INFINITY);
    let mut p = 0;
    while p < n16 {
        max_v = _mm512_max_ps(max_v, _mm512_loadu_ps(ptr.add(p)));
        p += 16;
    }
    let mut max_s = _mm512_reduce_max_ps(max_v);
    while p < n {
        let v = *ptr.add(p);
        if v > max_s {
            max_s = v;
        }
        p += 1;
    }

    let max_b = _mm512_set1_ps(max_s);
    let mut sum_v = _mm512_setzero_ps();
    let mut p = 0;
    while p < n16 {
        let v = _mm512_sub_ps(_mm512_loadu_ps(ptr.add(p)), max_b);
        let e = expf_approx_avx512f(v);
        sum_v = _mm512_add_ps(sum_v, e);
        p += 16;
    }
    let mut sum_s = _mm512_reduce_add_ps(sum_v);
    while p < n {
        sum_s += (*ptr.add(p) - max_s).exp();
        p += 1;
    }

    max_s + sum_s.ln()
}

/// AArch64 NEON `log_sum_exp`: 4-lane twin of the AVX2 path. Pass 1 folds a
/// horizontal max (`vmaxvq_f32`), pass 2 sums `exp(x - max)` via
/// [`expf_approx_neon`] + `vaddvq_f32`, with a scalar `libm` tail. Byte-layout
/// irrelevant (pure f32); close-not-bit-identical to scalar (reordered sum +
/// the ~2e-6 expf approximation), gated by the parity test like the x86 paths.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn log_sum_exp_f32_neon(x: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = x.len();
    let n4 = n & !3;
    let ptr = x.as_ptr();

    // Pass 1: horizontal max.
    let mut max_v = vdupq_n_f32(f32::NEG_INFINITY);
    let mut p = 0;
    while p < n4 {
        max_v = vmaxq_f32(max_v, vld1q_f32(ptr.add(p)));
        p += 4;
    }
    let mut max_s = vmaxvq_f32(max_v);
    while p < n {
        let v = *ptr.add(p);
        if v > max_s {
            max_s = v;
        }
        p += 1;
    }

    // Pass 2: sum of exp(x - max).
    let max_b = vdupq_n_f32(max_s);
    let mut sum_v = vdupq_n_f32(0.0);
    let mut p = 0;
    while p < n4 {
        let e = expf_approx_neon(vsubq_f32(vld1q_f32(ptr.add(p)), max_b));
        sum_v = vaddq_f32(sum_v, e);
        p += 4;
    }
    let mut sum_s = vaddvq_f32(sum_v);
    while p < n {
        sum_s += (*ptr.add(p) - max_s).exp();
        p += 1;
    }

    max_s + sum_s.ln()
}

/// Grouped-query attention over a single query token.
///
/// - `q`: `[n_heads, head_dim]` (rotated query for the current position)
/// - `k_cache`: `[n_kv_heads, max_ctx, head_dim]` keys up to `kv_len`
/// - `v_cache`: `[n_kv_heads, max_ctx, head_dim]` values up to `kv_len`
/// - `out`: `[n_heads, head_dim]`
/// - `kv_len`: number of valid timesteps in the cache (1..=max_ctx)
///
/// `n_heads` may be a multiple of `n_kv_heads` (GQA); each query head maps to
/// `kv_head = head_idx / n_gqa`. Uses AVX2/FMA when available.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_one_step(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len: usize,
) {
    debug_assert_eq!(q.len(), n_heads * head_dim);
    debug_assert_eq!(out.len(), n_heads * head_dim);
    debug_assert_eq!(k_cache.len(), n_kv_heads * max_ctx * head_dim);
    debug_assert_eq!(v_cache.len(), n_kv_heads * max_ctx * head_dim);
    debug_assert!(kv_len <= max_ctx);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: runtime feature detection.
            unsafe {
                gqa_attention_one_step_avx512(
                    q, k_cache, v_cache, out, n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
                );
            }
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime feature detection.
            unsafe {
                gqa_attention_one_step_avx2(
                    q, k_cache, v_cache, out, n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
                );
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64.
        unsafe {
            gqa_attention_one_step_neon(
                q, k_cache, v_cache, out, n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
            );
        }
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    gqa_attention_one_step_scalar(
        q, k_cache, v_cache, out, n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn gqa_attention_one_step_avx512(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len: usize,
) {
    use std::arch::x86_64::*;
    let n_gqa = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let head_dim_16 = head_dim & !15;
    let mut scores = vec![0.0f32; kv_len];

    for h in 0..n_heads {
        let kv_h = h / n_gqa;
        let q_h = &q[h * head_dim..(h + 1) * head_dim];

        for t in 0..kv_len {
            let k_off = (kv_h * max_ctx + t) * head_dim;
            let k_row = &k_cache[k_off..k_off + head_dim];
            let mut acc = _mm512_setzero_ps();
            let mut d = 0;
            while d < head_dim_16 {
                let qv = _mm512_loadu_ps(q_h.as_ptr().add(d));
                let kv = _mm512_loadu_ps(k_row.as_ptr().add(d));
                acc = _mm512_fmadd_ps(qv, kv, acc);
                d += 16;
            }
            let mut acc_scalar = _mm512_reduce_add_ps(acc);
            while d < head_dim {
                acc_scalar += q_h[d] * k_row[d];
                d += 1;
            }
            scores[t] = acc_scalar * scale;
        }
        softmax_f32_inplace(&mut scores);

        let out_h = &mut out[h * head_dim..(h + 1) * head_dim];
        let mut d = 0;
        while d < head_dim_16 {
            _mm512_storeu_ps(out_h.as_mut_ptr().add(d), _mm512_setzero_ps());
            d += 16;
        }
        while d < head_dim {
            out_h[d] = 0.0;
            d += 1;
        }

        for t in 0..kv_len {
            let v_off = (kv_h * max_ctx + t) * head_dim;
            let v_row = &v_cache[v_off..v_off + head_dim];
            let s = _mm512_set1_ps(scores[t]);
            let mut d = 0;
            while d < head_dim_16 {
                let vv = _mm512_loadu_ps(v_row.as_ptr().add(d));
                let o = _mm512_loadu_ps(out_h.as_ptr().add(d));
                let nv = _mm512_fmadd_ps(s, vv, o);
                _mm512_storeu_ps(out_h.as_mut_ptr().add(d), nv);
                d += 16;
            }
            let st = scores[t];
            while d < head_dim {
                out_h[d] += st * v_row[d];
                d += 1;
            }
        }
    }
}

#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub(crate) fn gqa_attention_one_step_scalar(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len: usize,
) {
    let n_gqa = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut scores = vec![0.0f32; kv_len];
    for h in 0..n_heads {
        let kv_h = h / n_gqa;
        let q_h = &q[h * head_dim..(h + 1) * head_dim];
        for t in 0..kv_len {
            let k_off = (kv_h * max_ctx + t) * head_dim;
            let k_row = &k_cache[k_off..k_off + head_dim];
            let mut acc = 0.0f32;
            for d in 0..head_dim {
                acc += q_h[d] * k_row[d];
            }
            scores[t] = acc * scale;
        }
        softmax_f32_inplace(&mut scores);
        let out_h = &mut out[h * head_dim..(h + 1) * head_dim];
        for v in out_h.iter_mut() {
            *v = 0.0;
        }
        for t in 0..kv_len {
            let v_off = (kv_h * max_ctx + t) * head_dim;
            let v_row = &v_cache[v_off..v_off + head_dim];
            let s = scores[t];
            for d in 0..head_dim {
                out_h[d] += s * v_row[d];
            }
        }
    }
}

/// AArch64 NEON twin of [`gqa_attention_one_step_scalar`] (3-pass GQA). The
/// Q·K dot (`vfmaq_f32` + `vaddvq_f32` horizontal) and the `out += s*v`
/// weighted-V accumulation are 4-lane; the softmax over `scores` reuses the
/// dispatched [`softmax_f32_inplace`] (NEON on aarch64). Close-not-bit-identical
/// to scalar (reordered reductions + expf approx in softmax).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn gqa_attention_one_step_neon(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len: usize,
) {
    use std::arch::aarch64::*;
    let n_gqa = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let hd4 = head_dim & !3;
    let mut scores = vec![0.0f32; kv_len];
    for h in 0..n_heads {
        let kv_h = h / n_gqa;
        let qp = q[h * head_dim..(h + 1) * head_dim].as_ptr();
        for t in 0..kv_len {
            let k_off = (kv_h * max_ctx + t) * head_dim;
            let kp = k_cache.as_ptr().add(k_off);
            let mut acc_v = vdupq_n_f32(0.0);
            let mut d = 0;
            while d < hd4 {
                acc_v = vfmaq_f32(acc_v, vld1q_f32(qp.add(d)), vld1q_f32(kp.add(d)));
                d += 4;
            }
            let mut acc = vaddvq_f32(acc_v);
            while d < head_dim {
                acc += *qp.add(d) * *kp.add(d);
                d += 1;
            }
            scores[t] = acc * scale;
        }
        softmax_f32_inplace(&mut scores);
        let out_h = &mut out[h * head_dim..(h + 1) * head_dim];
        for v in out_h.iter_mut() {
            *v = 0.0;
        }
        let op = out_h.as_mut_ptr();
        for t in 0..kv_len {
            let v_off = (kv_h * max_ctx + t) * head_dim;
            let vp = v_cache.as_ptr().add(v_off);
            let s_b = vdupq_n_f32(scores[t]);
            let s = scores[t];
            let mut d = 0;
            while d < hd4 {
                let o = vld1q_f32(op.add(d));
                vst1q_f32(op.add(d), vfmaq_f32(o, s_b, vld1q_f32(vp.add(d))));
                d += 4;
            }
            while d < head_dim {
                *op.add(d) += s * *vp.add(d);
                d += 1;
            }
        }
    }
}

/// FlashAttention-decode variant of [`gqa_attention_one_step`].
/// Fuses Q·Kᵀ, softmax, and ·V into a single pass over the kv_len
/// dimension using the online-softmax recurrence from FlashAttention
/// v1. Compared to the 3-pass kernel above:
///
///   - No `[kv_len]` scratch buffer for scores.
///   - Two passes over (kv_len × head_dim) instead of three.
///   - One `exp` per `t` instead of `kv_len` exps in the softmax pass.
///
/// Numerical guarantees: bit-for-bit identical to the standard
/// implementation up to floating-point reduction order. Greedy
/// sampling results match within ~1e-5 in our parity tests.
///
/// Plumbed via `[inference].flash_attention = true` in config (the
/// default) when the engine's F32 KV path runs. Quantized KV paths
/// (Q8_0 / TurboQuant / NVFP4) keep their fused dequant+attention
/// kernels — flash-decode there is a follow-up.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_flash_decode(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len: usize,
) {
    debug_assert_eq!(q.len(), n_heads * head_dim);
    debug_assert_eq!(out.len(), n_heads * head_dim);
    debug_assert_eq!(k_cache.len(), n_kv_heads * max_ctx * head_dim);
    debug_assert_eq!(v_cache.len(), n_kv_heads * max_ctx * head_dim);
    debug_assert!(kv_len <= max_ctx);
    if kv_len == 0 {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: runtime feature detection above.
            unsafe {
                gqa_attention_flash_decode_avx512(
                    q, k_cache, v_cache, out, n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
                );
            }
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime feature detection above.
            unsafe {
                gqa_attention_flash_decode_avx2(
                    q, k_cache, v_cache, out, n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
                );
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64.
        unsafe {
            gqa_attention_flash_decode_neon(
                q, k_cache, v_cache, out, n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
            );
        }
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    gqa_attention_flash_decode_scalar(
        q, k_cache, v_cache, out, n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
    );
}

/// Multi-query FlashAttention prefill kernel for F32 KV.
///
/// Processes `n_new` query positions in one call, each computing
/// attention over the K cache up to its causal-allowed kv_len:
/// query at *new-batch position* `i` attends to cache positions
/// `[0, kv_len_base + i + 1)`. This is the standard causal mask
/// for autoregressive prefill — query `i` sees its own K position
/// and everything before, but not future-batch positions.
///
/// Compared to calling [`gqa_attention_flash_decode`] N times in
/// a loop:
///   - Same total compute (online softmax per query × kv_len
///     positions); no cross-query parallelism in this kernel.
///   - Same per-call memory savings (no per-query scores scratch).
///   - One kernel-launch overhead instead of N; on CPU this is
///     small but with future SIMD/GPU work the cross-query batch
///     gets vectorized — that's the path this kernel's shape
///     unlocks.
///
/// The K/V cache layout is the same `[n_kv_heads, max_ctx, head_dim]`
/// row-major slab the decode kernel uses; the caller is expected
/// to have already written `K_new[0..n_new]` / `V_new[0..n_new]`
/// to cache positions `[kv_len_base, kv_len_base + n_new)` before
/// invoking this kernel.
///
/// Inputs:
///   - `q: [n_new, n_heads, head_dim]` — Q rows for the new positions
///   - `k_cache: [n_kv_heads, max_ctx, head_dim]` — full K cache
///   - `v_cache: [n_kv_heads, max_ctx, head_dim]` — full V cache
///   - `out: [n_new, n_heads, head_dim]` — written
///   - `kv_len_base` — kv positions filled BEFORE this batch
///   - `n_new` — number of new queries in the batch (each at
///     cache position `kv_len_base + i`)
///
/// Scalar reference for now; SIMD specializations follow the
/// same pattern as flash-decode (the per-query body is identical
/// to the decode kernel's inner loop).
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_flash_prefill(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len_base: usize,
    n_new: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: runtime feature detection above.
            unsafe {
                gqa_attention_flash_prefill_avx512(
                    q, k_cache, v_cache, out, n_heads, n_kv_heads,
                    head_dim, max_ctx, kv_len_base, n_new,
                );
            }
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime feature detection above.
            unsafe {
                gqa_attention_flash_prefill_avx2(
                    q, k_cache, v_cache, out, n_heads, n_kv_heads,
                    head_dim, max_ctx, kv_len_base, n_new,
                );
            }
            return;
        }
    }
    gqa_attention_flash_prefill_scalar(
        q, k_cache, v_cache, out, n_heads, n_kv_heads,
        head_dim, max_ctx, kv_len_base, n_new,
    );
}

#[allow(clippy::too_many_arguments)]
fn gqa_attention_flash_prefill_scalar(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len_base: usize,
    n_new: usize,
) {
    debug_assert_eq!(q.len(), n_new * n_heads * head_dim);
    debug_assert_eq!(out.len(), n_new * n_heads * head_dim);
    debug_assert_eq!(k_cache.len(), n_kv_heads * max_ctx * head_dim);
    debug_assert_eq!(v_cache.len(), n_kv_heads * max_ctx * head_dim);
    debug_assert!(kv_len_base + n_new <= max_ctx);
    debug_assert_eq!(n_heads % n_kv_heads, 0);
    if n_new == 0 {
        return;
    }
    // Zero out the output so the in-place accumulation below works
    // from a clean state. Per-iteration writes use `+=` rescale +
    // weighted V update, so leftover bytes would corrupt output.
    for x in out.iter_mut() {
        *x = 0.0;
    }
    let n_gqa = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let kv_len_total = kv_len_base + n_new;
    // Per-query online-softmax state, reused across the (kv_h,
    // q_in_group) loops by re-init.
    let mut m_state = vec![f32::NEG_INFINITY; n_new];
    let mut l_state = vec![0.0f32; n_new];

    for kv_h in 0..n_kv_heads {
        let cache_base = kv_h * max_ctx * head_dim;
        for q_in_group in 0..n_gqa {
            let h = kv_h * n_gqa + q_in_group;
            for slot in m_state.iter_mut() {
                *slot = f32::NEG_INFINITY;
            }
            for slot in l_state.iter_mut() {
                *slot = 0.0;
            }
            // Outer loop: walk cache positions in order. Inner loop:
            // update all queries whose causal mask admits this `t`.
            // K and V rows for this (kv_h, t) are loaded ONCE and
            // reused across `n_gqa × admitted_queries` writes —
            // that's the cache-locality win over running n_new
            // independent flash-decode calls.
            for t in 0..kv_len_total {
                let k_off = cache_base + t * head_dim;
                let k_row = &k_cache[k_off..k_off + head_dim];
                let v_row = &v_cache[k_off..k_off + head_dim];
                // First query position whose causal mask admits
                // this `t`: queries at new-batch positions
                // `[q_pos_min, n_new)` see this `t`. For
                // `t < kv_len_base` that's all queries; for
                // `t >= kv_len_base` it's `[t - kv_len_base, n_new)`.
                let q_pos_min = t.saturating_sub(kv_len_base);
                for q_pos in q_pos_min..n_new {
                    let q_off = (q_pos * n_heads + h) * head_dim;
                    let q_row = &q[q_off..q_off + head_dim];
                    let mut s = 0f32;
                    for d in 0..head_dim {
                        s += q_row[d] * k_row[d];
                    }
                    s *= scale;
                    let m_old = m_state[q_pos];
                    let m_new = m_old.max(s);
                    let rescale =
                        if m_old.is_finite() { (m_old - m_new).exp() } else { 0.0 };
                    let p = (s - m_new).exp();
                    l_state[q_pos] = l_state[q_pos] * rescale + p;
                    let out_off = (q_pos * n_heads + h) * head_dim;
                    for d in 0..head_dim {
                        out[out_off + d] = out[out_off + d] * rescale + p * v_row[d];
                    }
                    m_state[q_pos] = m_new;
                }
            }
            for q_pos in 0..n_new {
                let l = l_state[q_pos];
                let inv_l = if l > 0.0 { 1.0 / l } else { 0.0 };
                let out_off = (q_pos * n_heads + h) * head_dim;
                for d in 0..head_dim {
                    out[out_off + d] *= inv_l;
                }
            }
        }
    }
}

/// AVX-2 + FMA specialization of [`gqa_attention_flash_prefill`].
/// Preserves the cache-friendly K-outer ordering (each K/V row is
/// loaded once per kv_h and reused across admitted queries) while
/// vectorizing the inner d-loops at 8 floats per iteration:
/// the Q·K dot product and the V accumulation with rescale.
/// Cross-query and cross-`t` recurrences stay scalar — each
/// iteration's online-softmax state update depends on the prior
/// one, no parallelism there.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn gqa_attention_flash_prefill_avx2(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len_base: usize,
    n_new: usize,
) {
    use std::arch::x86_64::*;
    debug_assert_eq!(q.len(), n_new * n_heads * head_dim);
    debug_assert_eq!(out.len(), n_new * n_heads * head_dim);
    debug_assert_eq!(k_cache.len(), n_kv_heads * max_ctx * head_dim);
    debug_assert_eq!(v_cache.len(), n_kv_heads * max_ctx * head_dim);
    debug_assert!(kv_len_base + n_new <= max_ctx);
    debug_assert_eq!(n_heads % n_kv_heads, 0);
    if n_new == 0 {
        return;
    }
    for x in out.iter_mut() {
        *x = 0.0;
    }
    let n_gqa = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let kv_len_total = kv_len_base + n_new;
    let head_dim_8 = head_dim & !7;
    let mut m_state = vec![f32::NEG_INFINITY; n_new];
    let mut l_state = vec![0.0f32; n_new];

    for kv_h in 0..n_kv_heads {
        let cache_base = kv_h * max_ctx * head_dim;
        for q_in_group in 0..n_gqa {
            let h = kv_h * n_gqa + q_in_group;
            for slot in m_state.iter_mut() {
                *slot = f32::NEG_INFINITY;
            }
            for slot in l_state.iter_mut() {
                *slot = 0.0;
            }
            // K-outer loop: load each (k_row, v_row) once per kv_h
            // and process all admitted queries' inner d-loops with
            // AVX-2.
            for t in 0..kv_len_total {
                let k_off = cache_base + t * head_dim;
                let k_row = &k_cache[k_off..k_off + head_dim];
                let v_row = &v_cache[k_off..k_off + head_dim];
                let q_pos_min = t.saturating_sub(kv_len_base);
                for q_pos in q_pos_min..n_new {
                    let q_off = (q_pos * n_heads + h) * head_dim;
                    let q_row = &q[q_off..q_off + head_dim];
                    // Q·K dot, vectorized.
                    let mut acc = _mm256_setzero_ps();
                    let mut d = 0;
                    while d < head_dim_8 {
                        let qv = _mm256_loadu_ps(q_row.as_ptr().add(d));
                        let kv = _mm256_loadu_ps(k_row.as_ptr().add(d));
                        acc = _mm256_fmadd_ps(qv, kv, acc);
                        d += 8;
                    }
                    let mut sum128 = _mm_add_ps(
                        _mm256_castps256_ps128(acc),
                        _mm256_extractf128_ps(acc, 1),
                    );
                    sum128 = _mm_hadd_ps(sum128, sum128);
                    sum128 = _mm_hadd_ps(sum128, sum128);
                    let mut s = _mm_cvtss_f32(sum128);
                    while d < head_dim {
                        s += q_row[d] * k_row[d];
                        d += 1;
                    }
                    s *= scale;
                    // Scalar online-softmax state update.
                    let m_old = m_state[q_pos];
                    let m_new = m_old.max(s);
                    let rescale =
                        if m_old.is_finite() { (m_old - m_new).exp() } else { 0.0 };
                    let p = (s - m_new).exp();
                    l_state[q_pos] = l_state[q_pos] * rescale + p;
                    // V accumulation, vectorized: out[q_pos] = out[q_pos] * rescale + p * v_row.
                    let out_off = (q_pos * n_heads + h) * head_dim;
                    let rescale_v = _mm256_set1_ps(rescale);
                    let p_v = _mm256_set1_ps(p);
                    let mut d = 0;
                    while d < head_dim_8 {
                        let cur = _mm256_loadu_ps(out.as_ptr().add(out_off + d));
                        let scaled = _mm256_mul_ps(cur, rescale_v);
                        let vv = _mm256_loadu_ps(v_row.as_ptr().add(d));
                        let updated = _mm256_fmadd_ps(p_v, vv, scaled);
                        _mm256_storeu_ps(out.as_mut_ptr().add(out_off + d), updated);
                        d += 8;
                    }
                    while d < head_dim {
                        out[out_off + d] = out[out_off + d] * rescale + p * v_row[d];
                        d += 1;
                    }
                    m_state[q_pos] = m_new;
                }
            }
            // Final normalize per query, vectorized.
            for q_pos in 0..n_new {
                let l = l_state[q_pos];
                let inv_l = if l > 0.0 { 1.0 / l } else { 0.0 };
                let out_off = (q_pos * n_heads + h) * head_dim;
                let inv_l_v = _mm256_set1_ps(inv_l);
                let mut d = 0;
                while d < head_dim_8 {
                    let cur = _mm256_loadu_ps(out.as_ptr().add(out_off + d));
                    _mm256_storeu_ps(
                        out.as_mut_ptr().add(out_off + d),
                        _mm256_mul_ps(cur, inv_l_v),
                    );
                    d += 8;
                }
                while d < head_dim {
                    out[out_off + d] *= inv_l;
                    d += 1;
                }
            }
        }
    }
}

/// AVX-512 specialization of [`gqa_attention_flash_prefill`].
/// Same K-outer cache-friendly ordering as the AVX-2 variant; the
/// inner d-loops widen to 16 floats per iteration and the
/// horizontal sum uses `_mm512_reduce_add_ps`. Bigger win at
/// `head_dim >= 64` on hardware that exposes AVX-512.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn gqa_attention_flash_prefill_avx512(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len_base: usize,
    n_new: usize,
) {
    use std::arch::x86_64::*;
    debug_assert_eq!(q.len(), n_new * n_heads * head_dim);
    debug_assert_eq!(out.len(), n_new * n_heads * head_dim);
    debug_assert_eq!(k_cache.len(), n_kv_heads * max_ctx * head_dim);
    debug_assert_eq!(v_cache.len(), n_kv_heads * max_ctx * head_dim);
    debug_assert!(kv_len_base + n_new <= max_ctx);
    debug_assert_eq!(n_heads % n_kv_heads, 0);
    if n_new == 0 {
        return;
    }
    for x in out.iter_mut() {
        *x = 0.0;
    }
    let n_gqa = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let kv_len_total = kv_len_base + n_new;
    let head_dim_16 = head_dim & !15;
    let mut m_state = vec![f32::NEG_INFINITY; n_new];
    let mut l_state = vec![0.0f32; n_new];

    for kv_h in 0..n_kv_heads {
        let cache_base = kv_h * max_ctx * head_dim;
        for q_in_group in 0..n_gqa {
            let h = kv_h * n_gqa + q_in_group;
            for slot in m_state.iter_mut() {
                *slot = f32::NEG_INFINITY;
            }
            for slot in l_state.iter_mut() {
                *slot = 0.0;
            }
            for t in 0..kv_len_total {
                let k_off = cache_base + t * head_dim;
                let k_row = &k_cache[k_off..k_off + head_dim];
                let v_row = &v_cache[k_off..k_off + head_dim];
                let q_pos_min = t.saturating_sub(kv_len_base);
                for q_pos in q_pos_min..n_new {
                    let q_off = (q_pos * n_heads + h) * head_dim;
                    let q_row = &q[q_off..q_off + head_dim];
                    let mut acc = _mm512_setzero_ps();
                    let mut d = 0;
                    while d < head_dim_16 {
                        let qv = _mm512_loadu_ps(q_row.as_ptr().add(d));
                        let kv = _mm512_loadu_ps(k_row.as_ptr().add(d));
                        acc = _mm512_fmadd_ps(qv, kv, acc);
                        d += 16;
                    }
                    let mut s = _mm512_reduce_add_ps(acc);
                    while d < head_dim {
                        s += q_row[d] * k_row[d];
                        d += 1;
                    }
                    s *= scale;
                    let m_old = m_state[q_pos];
                    let m_new = m_old.max(s);
                    let rescale =
                        if m_old.is_finite() { (m_old - m_new).exp() } else { 0.0 };
                    let p = (s - m_new).exp();
                    l_state[q_pos] = l_state[q_pos] * rescale + p;
                    let out_off = (q_pos * n_heads + h) * head_dim;
                    let rescale_v = _mm512_set1_ps(rescale);
                    let p_v = _mm512_set1_ps(p);
                    let mut d = 0;
                    while d < head_dim_16 {
                        let cur = _mm512_loadu_ps(out.as_ptr().add(out_off + d));
                        let scaled = _mm512_mul_ps(cur, rescale_v);
                        let vv = _mm512_loadu_ps(v_row.as_ptr().add(d));
                        let updated = _mm512_fmadd_ps(p_v, vv, scaled);
                        _mm512_storeu_ps(out.as_mut_ptr().add(out_off + d), updated);
                        d += 16;
                    }
                    while d < head_dim {
                        out[out_off + d] = out[out_off + d] * rescale + p * v_row[d];
                        d += 1;
                    }
                    m_state[q_pos] = m_new;
                }
            }
            for q_pos in 0..n_new {
                let l = l_state[q_pos];
                let inv_l = if l > 0.0 { 1.0 / l } else { 0.0 };
                let out_off = (q_pos * n_heads + h) * head_dim;
                let inv_l_v = _mm512_set1_ps(inv_l);
                let mut d = 0;
                while d < head_dim_16 {
                    let cur = _mm512_loadu_ps(out.as_ptr().add(out_off + d));
                    _mm512_storeu_ps(
                        out.as_mut_ptr().add(out_off + d),
                        _mm512_mul_ps(cur, inv_l_v),
                    );
                    d += 16;
                }
                while d < head_dim {
                    out[out_off + d] *= inv_l;
                    d += 1;
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gqa_attention_flash_decode_scalar(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len: usize,
) {
    let n_gqa = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    for h in 0..n_heads {
        let kv_h = h / n_gqa;
        let q_h = &q[h * head_dim..(h + 1) * head_dim];
        let out_h = &mut out[h * head_dim..(h + 1) * head_dim];
        for v in out_h.iter_mut() {
            *v = 0.0;
        }
        // Online softmax state.
        //   m: running max over scaled scores seen so far
        //   l: running sum of `exp(score - m)` over scores seen so far
        //   out_h: running weighted sum, rescaled on each m update
        // Initialize m at -inf so the first iteration sets it.
        let mut m = f32::NEG_INFINITY;
        let mut l = 0.0f32;
        for t in 0..kv_len {
            let k_off = (kv_h * max_ctx + t) * head_dim;
            let k_row = &k_cache[k_off..k_off + head_dim];
            // score = scale * (q · k_row)
            let mut s = 0.0f32;
            for d in 0..head_dim {
                s += q_h[d] * k_row[d];
            }
            s *= scale;
            // Update running max + rescale prior accumulations.
            let m_new = m.max(s);
            let rescale = if m.is_finite() { (m - m_new).exp() } else { 0.0 };
            let p = (s - m_new).exp();
            l = l * rescale + p;
            let v_off = (kv_h * max_ctx + t) * head_dim;
            let v_row = &v_cache[v_off..v_off + head_dim];
            for d in 0..head_dim {
                out_h[d] = out_h[d] * rescale + p * v_row[d];
            }
            m = m_new;
        }
        // Final normalize.
        let inv_l = if l > 0.0 { 1.0 / l } else { 0.0 };
        for v in out_h.iter_mut() {
            *v *= inv_l;
        }
    }
}

/// AArch64 NEON twin of [`gqa_attention_flash_decode_scalar`] (online-softmax
/// flash decode). Per kv position, the Q·K dot (`vfmaq_f32` + `vaddvq_f32`) and
/// the `out = out*rescale + p*v` accumulation are 4-lane; the running
/// max/rescale/`exp`/`l` recurrence stays scalar (`libm` exp) exactly as the
/// AVX2 path. Close-not-bit-identical to scalar (reordered dot reduction).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn gqa_attention_flash_decode_neon(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len: usize,
) {
    use std::arch::aarch64::*;
    let n_gqa = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let hd4 = head_dim & !3;
    for h in 0..n_heads {
        let kv_h = h / n_gqa;
        let qp = q[h * head_dim..(h + 1) * head_dim].as_ptr();
        let out_h = &mut out[h * head_dim..(h + 1) * head_dim];
        for v in out_h.iter_mut() {
            *v = 0.0;
        }
        let op = out_h.as_mut_ptr();
        let mut m = f32::NEG_INFINITY;
        let mut l = 0.0f32;
        for t in 0..kv_len {
            let k_off = (kv_h * max_ctx + t) * head_dim;
            let kp = k_cache.as_ptr().add(k_off);
            // score = scale * (q · k_row)
            let mut acc_v = vdupq_n_f32(0.0);
            let mut d = 0;
            while d < hd4 {
                acc_v = vfmaq_f32(acc_v, vld1q_f32(qp.add(d)), vld1q_f32(kp.add(d)));
                d += 4;
            }
            let mut s = vaddvq_f32(acc_v);
            while d < head_dim {
                s += *qp.add(d) * *kp.add(d);
                d += 1;
            }
            s *= scale;
            // Online-softmax update + rescale of the running accumulator.
            let m_new = m.max(s);
            let rescale = if m.is_finite() { (m - m_new).exp() } else { 0.0 };
            let p = (s - m_new).exp();
            l = l * rescale + p;
            let v_off = (kv_h * max_ctx + t) * head_dim;
            let vp = v_cache.as_ptr().add(v_off);
            let rescale_b = vdupq_n_f32(rescale);
            let p_b = vdupq_n_f32(p);
            let mut d = 0;
            while d < hd4 {
                // out = out*rescale + p*v
                let o = vmulq_f32(vld1q_f32(op.add(d)), rescale_b);
                vst1q_f32(op.add(d), vfmaq_f32(o, p_b, vld1q_f32(vp.add(d))));
                d += 4;
            }
            while d < head_dim {
                *op.add(d) = *op.add(d) * rescale + p * *vp.add(d);
                d += 1;
            }
            m = m_new;
        }
        // Final normalize.
        let inv_l = if l > 0.0 { 1.0 / l } else { 0.0 };
        let inv_b = vdupq_n_f32(inv_l);
        let mut d = 0;
        while d < hd4 {
            vst1q_f32(op.add(d), vmulq_f32(vld1q_f32(op.add(d)), inv_b));
            d += 4;
        }
        while d < head_dim {
            *op.add(d) *= inv_l;
            d += 1;
        }
    }
}

/// AVX-512 specialization of [`gqa_attention_flash_decode`]. Mirrors
/// the AVX-2 variant but processes 16 floats per inner-loop iteration
/// instead of 8. Used by AVX-512-enabled Intel server / AMD Zen 4+
/// chips; consumer Alder/Raptor Lake disabled AVX-512 in microcode so
/// the runtime dispatcher falls through to AVX-2 there.
///
/// The horizontal sum after the dot product uses `_mm512_reduce_add_ps`,
/// which is a single intrinsic on AVX-512 instead of the two-stage
/// hadd_ps dance AVX-2 needs.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn gqa_attention_flash_decode_avx512(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len: usize,
) {
    use std::arch::x86_64::*;
    let n_gqa = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let head_dim_16 = head_dim & !15;

    for h in 0..n_heads {
        let kv_h = h / n_gqa;
        let q_h = &q[h * head_dim..(h + 1) * head_dim];
        let out_h = &mut out[h * head_dim..(h + 1) * head_dim];
        // Zero out the accumulator.
        let mut d = 0;
        while d < head_dim_16 {
            _mm512_storeu_ps(out_h.as_mut_ptr().add(d), _mm512_setzero_ps());
            d += 16;
        }
        while d < head_dim {
            out_h[d] = 0.0;
            d += 1;
        }
        let mut m = f32::NEG_INFINITY;
        let mut l = 0.0f32;
        for t in 0..kv_len {
            let k_off = (kv_h * max_ctx + t) * head_dim;
            let k_row = &k_cache[k_off..k_off + head_dim];
            // Q · K row, vectorized dot product at 16 floats / iter.
            let mut acc = _mm512_setzero_ps();
            let mut d = 0;
            while d < head_dim_16 {
                let qv = _mm512_loadu_ps(q_h.as_ptr().add(d));
                let kv = _mm512_loadu_ps(k_row.as_ptr().add(d));
                acc = _mm512_fmadd_ps(qv, kv, acc);
                d += 16;
            }
            // Single-instruction horizontal sum on AVX-512.
            let mut s = _mm512_reduce_add_ps(acc);
            while d < head_dim {
                s += q_h[d] * k_row[d];
                d += 1;
            }
            s *= scale;
            // Online softmax state update.
            let m_new = m.max(s);
            let rescale = if m.is_finite() { (m - m_new).exp() } else { 0.0 };
            let p = (s - m_new).exp();
            l = l * rescale + p;
            // V accumulation: out_h = out_h * rescale + p * v_row.
            let v_off = (kv_h * max_ctx + t) * head_dim;
            let v_row = &v_cache[v_off..v_off + head_dim];
            let rescale_v = _mm512_set1_ps(rescale);
            let p_v = _mm512_set1_ps(p);
            let mut d = 0;
            while d < head_dim_16 {
                let cur = _mm512_loadu_ps(out_h.as_ptr().add(d));
                let scaled = _mm512_mul_ps(cur, rescale_v);
                let vv = _mm512_loadu_ps(v_row.as_ptr().add(d));
                let updated = _mm512_fmadd_ps(p_v, vv, scaled);
                _mm512_storeu_ps(out_h.as_mut_ptr().add(d), updated);
                d += 16;
            }
            while d < head_dim {
                out_h[d] = out_h[d] * rescale + p * v_row[d];
                d += 1;
            }
            m = m_new;
        }
        let inv_l = if l > 0.0 { 1.0 / l } else { 0.0 };
        let inv_l_v = _mm512_set1_ps(inv_l);
        let mut d = 0;
        while d < head_dim_16 {
            let cur = _mm512_loadu_ps(out_h.as_ptr().add(d));
            _mm512_storeu_ps(out_h.as_mut_ptr().add(d), _mm512_mul_ps(cur, inv_l_v));
            d += 16;
        }
        while d < head_dim {
            out_h[d] *= inv_l;
            d += 1;
        }
    }
}

/// AVX-2 + FMA specialization of [`gqa_attention_flash_decode`].
/// Same online-softmax recurrence; the inner d-loops (Q·K dot,
/// V-weighted accumulation with rescale) are vectorized at 8 floats
/// per iteration. The cross-`t` recurrence stays scalar — each
/// iteration needs the running max from the previous, no parallelism
/// there.
///
/// With AVX-2 the flash-decode path beats the standard 3-pass
/// implementation across all `kv_len` because the inner loops have
/// equivalent SIMD utilization, but flash does 2 passes instead of
/// 3. The `kv_len >= 4096` threshold can drop to ~256 once this is
/// in place; see `RUSTLLAMA_FLASH_KV_LEN_MIN` env override.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn gqa_attention_flash_decode_avx2(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len: usize,
) {
    use std::arch::x86_64::*;
    let n_gqa = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let head_dim_8 = head_dim & !7;

    for h in 0..n_heads {
        let kv_h = h / n_gqa;
        let q_h = &q[h * head_dim..(h + 1) * head_dim];
        let out_h = &mut out[h * head_dim..(h + 1) * head_dim];
        // Zero out the accumulator. Vectorize the wide span.
        let mut d = 0;
        while d < head_dim_8 {
            _mm256_storeu_ps(out_h.as_mut_ptr().add(d), _mm256_setzero_ps());
            d += 8;
        }
        while d < head_dim {
            out_h[d] = 0.0;
            d += 1;
        }
        let mut m = f32::NEG_INFINITY;
        let mut l = 0.0f32;
        for t in 0..kv_len {
            let k_off = (kv_h * max_ctx + t) * head_dim;
            let k_row = &k_cache[k_off..k_off + head_dim];
            // Q · K row, vectorized dot product.
            let mut acc = _mm256_setzero_ps();
            let mut d = 0;
            while d < head_dim_8 {
                let qv = _mm256_loadu_ps(q_h.as_ptr().add(d));
                let kv = _mm256_loadu_ps(k_row.as_ptr().add(d));
                acc = _mm256_fmadd_ps(qv, kv, acc);
                d += 8;
            }
            // Horizontal sum of the AVX-2 vector. Two hadd_ps +
            // extractf128 + add_ss reduces 8 floats to scalar.
            let mut sum128 = _mm_add_ps(
                _mm256_castps256_ps128(acc),
                _mm256_extractf128_ps(acc, 1),
            );
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            let mut s = _mm_cvtss_f32(sum128);
            // Scalar tail for non-multiple-of-8 head_dim.
            while d < head_dim {
                s += q_h[d] * k_row[d];
                d += 1;
            }
            s *= scale;
            // Online softmax state update.
            let m_new = m.max(s);
            let rescale = if m.is_finite() { (m - m_new).exp() } else { 0.0 };
            let p = (s - m_new).exp();
            l = l * rescale + p;
            // V accumulation: out_h = out_h * rescale + p * v_row.
            // Vectorize the d loop.
            let v_off = (kv_h * max_ctx + t) * head_dim;
            let v_row = &v_cache[v_off..v_off + head_dim];
            let rescale_v = _mm256_set1_ps(rescale);
            let p_v = _mm256_set1_ps(p);
            let mut d = 0;
            while d < head_dim_8 {
                let cur = _mm256_loadu_ps(out_h.as_ptr().add(d));
                let scaled = _mm256_mul_ps(cur, rescale_v);
                let vv = _mm256_loadu_ps(v_row.as_ptr().add(d));
                let updated = _mm256_fmadd_ps(p_v, vv, scaled);
                _mm256_storeu_ps(out_h.as_mut_ptr().add(d), updated);
                d += 8;
            }
            while d < head_dim {
                out_h[d] = out_h[d] * rescale + p * v_row[d];
                d += 1;
            }
            m = m_new;
        }
        let inv_l = if l > 0.0 { 1.0 / l } else { 0.0 };
        let inv_l_v = _mm256_set1_ps(inv_l);
        let mut d = 0;
        while d < head_dim_8 {
            let cur = _mm256_loadu_ps(out_h.as_ptr().add(d));
            _mm256_storeu_ps(out_h.as_mut_ptr().add(d), _mm256_mul_ps(cur, inv_l_v));
            d += 8;
        }
        while d < head_dim {
            out_h[d] *= inv_l;
            d += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn gqa_attention_one_step_avx2(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len: usize,
) {
    use std::arch::x86_64::*;
    let n_gqa = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let head_dim_8 = head_dim & !7;
    let mut scores = vec![0.0f32; kv_len];

    for h in 0..n_heads {
        let kv_h = h / n_gqa;
        let q_h = &q[h * head_dim..(h + 1) * head_dim];

        // scores[t] = q · k[kv_h, t, :] * scale, vectorized
        for t in 0..kv_len {
            let k_off = (kv_h * max_ctx + t) * head_dim;
            let k_row = &k_cache[k_off..k_off + head_dim];
            let mut acc = _mm256_setzero_ps();
            let mut d = 0;
            while d < head_dim_8 {
                let qv = _mm256_loadu_ps(q_h.as_ptr().add(d));
                let kv = _mm256_loadu_ps(k_row.as_ptr().add(d));
                acc = _mm256_fmadd_ps(qv, kv, acc);
                d += 8;
            }
            let mut sum128 =
                _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            let mut acc_scalar = _mm_cvtss_f32(sum128);
            while d < head_dim {
                acc_scalar += q_h[d] * k_row[d];
                d += 1;
            }
            scores[t] = acc_scalar * scale;
        }
        softmax_f32_inplace(&mut scores);

        // out[h, :] = sum_t scores[t] * v[kv_h, t, :], vectorized.
        let out_h = &mut out[h * head_dim..(h + 1) * head_dim];
        // Zero the accumulator
        let mut d = 0;
        while d < head_dim_8 {
            _mm256_storeu_ps(out_h.as_mut_ptr().add(d), _mm256_setzero_ps());
            d += 8;
        }
        while d < head_dim {
            out_h[d] = 0.0;
            d += 1;
        }

        for t in 0..kv_len {
            let v_off = (kv_h * max_ctx + t) * head_dim;
            let v_row = &v_cache[v_off..v_off + head_dim];
            let s = _mm256_set1_ps(scores[t]);
            let mut d = 0;
            while d < head_dim_8 {
                let v = _mm256_loadu_ps(v_row.as_ptr().add(d));
                let o = _mm256_loadu_ps(out_h.as_ptr().add(d));
                let nv = _mm256_fmadd_ps(s, v, o);
                _mm256_storeu_ps(out_h.as_mut_ptr().add(d), nv);
                d += 8;
            }
            let st = scores[t];
            while d < head_dim {
                out_h[d] += st * v_row[d];
                d += 1;
            }
        }
    }
}

/// GQA attention with a Q8_0-quantized K/V cache. Layout of the cache:
///   `k_q[(h * max_ctx + t) * head_dim + d]` is the i8 quant of K[h,t,d].
///   `k_scales[h * max_ctx + t]` is the f32 scale for that row.
///
/// Dequant is inlined into the dot product and weighted sum; we never
/// materialize the whole cache as f32. Dispatches to AVX2/FMA at runtime
/// when available; falls back to scalar otherwise.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_one_step_q8_0(
    q: &[f32],
    k_q: &[i8],
    k_scales: &[f32],
    v_q: &[i8],
    v_scales: &[f32],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len: usize,
) {
    debug_assert_eq!(q.len(), n_heads * head_dim);
    debug_assert_eq!(out.len(), n_heads * head_dim);
    debug_assert_eq!(k_q.len(), n_kv_heads * max_ctx * head_dim);
    debug_assert_eq!(v_q.len(), n_kv_heads * max_ctx * head_dim);
    debug_assert_eq!(k_scales.len(), n_kv_heads * max_ctx);
    debug_assert_eq!(v_scales.len(), n_kv_heads * max_ctx);
    debug_assert!(kv_len <= max_ctx);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            // SAFETY: runtime feature detection.
            unsafe {
                gqa_attention_one_step_q8_0_avx512(
                    q, k_q, k_scales, v_q, v_scales, out, n_heads, n_kv_heads, head_dim,
                    max_ctx, kv_len,
                );
            }
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime feature detection.
            unsafe {
                gqa_attention_one_step_q8_0_avx2(
                    q, k_q, k_scales, v_q, v_scales, out, n_heads, n_kv_heads, head_dim,
                    max_ctx, kv_len,
                );
            }
            return;
        }
    }
    gqa_attention_one_step_q8_0_scalar(
        q, k_q, k_scales, v_q, v_scales, out, n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f,avx512bw")]
#[allow(clippy::too_many_arguments)]
unsafe fn gqa_attention_one_step_q8_0_avx512(
    q: &[f32],
    k_q: &[i8],
    k_scales: &[f32],
    v_q: &[i8],
    v_scales: &[f32],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len: usize,
) {
    use std::arch::x86_64::*;
    let n_gqa = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let head_dim_16 = head_dim & !15;
    let mut scores = vec![0.0f32; kv_len];

    // Load 16 i8s and widen to a ZMM f32x16. `_mm_loadu_si128` pulls a
    // 128-bit XMM register (16 bytes / 16 i8s); `_mm512_cvtepi8_epi32`
    // sign-extends to i32x16 in a ZMM; `_mm512_cvtepi32_ps` converts to
    // f32x16 in one shot.
    #[inline(always)]
    unsafe fn load_i8x16_as_f32x16(ptr: *const i8) -> __m512 {
        let bytes = _mm_loadu_si128(ptr as *const __m128i);
        let i32s = _mm512_cvtepi8_epi32(bytes);
        _mm512_cvtepi32_ps(i32s)
    }

    for h in 0..n_heads {
        let kv_h = h / n_gqa;
        let q_h = &q[h * head_dim..(h + 1) * head_dim];

        for t in 0..kv_len {
            let row_idx = kv_h * max_ctx + t;
            let k_off = row_idx * head_dim;
            let k_row = &k_q[k_off..k_off + head_dim];
            let row_scale = k_scales[row_idx];
            let mut acc = _mm512_setzero_ps();
            let mut d = 0;
            while d < head_dim_16 {
                let qv = _mm512_loadu_ps(q_h.as_ptr().add(d));
                let kv = load_i8x16_as_f32x16(k_row.as_ptr().add(d));
                acc = _mm512_fmadd_ps(qv, kv, acc);
                d += 16;
            }
            let mut acc_scalar = _mm512_reduce_add_ps(acc);
            while d < head_dim {
                acc_scalar += q_h[d] * (k_row[d] as f32);
                d += 1;
            }
            // Fold row_scale + scale at row-end (saved head_dim multiplies).
            scores[t] = acc_scalar * row_scale * scale;
        }
        softmax_f32_inplace(&mut scores);

        let out_h = &mut out[h * head_dim..(h + 1) * head_dim];
        let mut d = 0;
        while d < head_dim_16 {
            _mm512_storeu_ps(out_h.as_mut_ptr().add(d), _mm512_setzero_ps());
            d += 16;
        }
        while d < head_dim {
            out_h[d] = 0.0;
            d += 1;
        }

        for t in 0..kv_len {
            let row_idx = kv_h * max_ctx + t;
            let v_off = row_idx * head_dim;
            let v_row = &v_q[v_off..v_off + head_dim];
            let row_scale = v_scales[row_idx];
            let s_eff = scores[t] * row_scale;
            let s_vec = _mm512_set1_ps(s_eff);
            let mut d = 0;
            while d < head_dim_16 {
                let vv = load_i8x16_as_f32x16(v_row.as_ptr().add(d));
                let o = _mm512_loadu_ps(out_h.as_ptr().add(d));
                let nv = _mm512_fmadd_ps(s_vec, vv, o);
                _mm512_storeu_ps(out_h.as_mut_ptr().add(d), nv);
                d += 16;
            }
            while d < head_dim {
                out_h[d] += s_eff * (v_row[d] as f32);
                d += 1;
            }
        }
    }
}

#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn gqa_attention_one_step_q8_0_scalar(
    q: &[f32],
    k_q: &[i8],
    k_scales: &[f32],
    v_q: &[i8],
    v_scales: &[f32],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len: usize,
) {
    let n_gqa = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut scores = vec![0.0f32; kv_len];

    for h in 0..n_heads {
        let kv_h = h / n_gqa;
        let q_h = &q[h * head_dim..(h + 1) * head_dim];
        for t in 0..kv_len {
            let row_idx = kv_h * max_ctx + t;
            let k_off = row_idx * head_dim;
            let k_row = &k_q[k_off..k_off + head_dim];
            let row_scale = k_scales[row_idx];
            // Factor row_scale out of the d-loop. The pre-quant value is
            // `k_row[d] as f32 * row_scale`, so the inner sum is
            //   sum(q[d] * k_row[d] as f32) * row_scale.
            // This saves head_dim multiplies per row.
            let mut acc = 0.0f32;
            for d in 0..head_dim {
                acc += q_h[d] * (k_row[d] as f32);
            }
            scores[t] = acc * row_scale * scale;
        }
        softmax_f32_inplace(&mut scores);
        let out_h = &mut out[h * head_dim..(h + 1) * head_dim];
        for v in out_h.iter_mut() {
            *v = 0.0;
        }
        for t in 0..kv_len {
            let row_idx = kv_h * max_ctx + t;
            let v_off = row_idx * head_dim;
            let v_row = &v_q[v_off..v_off + head_dim];
            let row_scale = v_scales[row_idx];
            // Same factoring: dequantized value is v_row[d] as f32 *
            // row_scale; the per-d work is just `s_eff * v_row[d] as f32`.
            let s_eff = scores[t] * row_scale;
            for d in 0..head_dim {
                out_h[d] += s_eff * (v_row[d] as f32);
            }
        }
    }
}

/// FlashAttention-decode variant of [`gqa_attention_one_step_q8_0`].
/// Mirrors the F32 + TQ flash kernels — fuses Q·Kᵀ, softmax, and ·V
/// into a single pass over `kv_len` with the online-softmax
/// recurrence. Dequant is inlined into both the dot product (×
/// k_scale) and the V accumulation (× v_scale), matching the
/// non-flash kernel's factoring.
///
/// Plumbed via `[inference].flash_attention = true` (the default)
/// at `kv_len >= RUSTLLAMA_FLASH_KV_LEN_MIN` (default 4096). Greedy
/// parity vs the standard Q8_0 kernel is asserted by the
/// `flash_decode_q8_0_matches_standard_q8_0` test below.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_flash_decode_q8_0(
    q: &[f32],
    k_q: &[i8],
    k_scales: &[f32],
    v_q: &[i8],
    v_scales: &[f32],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len: usize,
) {
    debug_assert_eq!(q.len(), n_heads * head_dim);
    debug_assert_eq!(out.len(), n_heads * head_dim);
    debug_assert_eq!(k_q.len(), n_kv_heads * max_ctx * head_dim);
    debug_assert_eq!(v_q.len(), n_kv_heads * max_ctx * head_dim);
    debug_assert_eq!(k_scales.len(), n_kv_heads * max_ctx);
    debug_assert_eq!(v_scales.len(), n_kv_heads * max_ctx);
    debug_assert!(kv_len <= max_ctx);
    if kv_len == 0 {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            // SAFETY: runtime feature detection above.
            unsafe {
                gqa_attention_flash_decode_q8_0_avx512(
                    q, k_q, k_scales, v_q, v_scales, out, n_heads, n_kv_heads, head_dim,
                    max_ctx, kv_len,
                );
            }
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime feature detection above.
            unsafe {
                gqa_attention_flash_decode_q8_0_avx2(
                    q, k_q, k_scales, v_q, v_scales, out, n_heads, n_kv_heads, head_dim,
                    max_ctx, kv_len,
                );
            }
            return;
        }
    }
    gqa_attention_flash_decode_q8_0_scalar(
        q, k_q, k_scales, v_q, v_scales, out, n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
    );
}

#[allow(clippy::too_many_arguments)]
fn gqa_attention_flash_decode_q8_0_scalar(
    q: &[f32],
    k_q: &[i8],
    k_scales: &[f32],
    v_q: &[i8],
    v_scales: &[f32],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len: usize,
) {
    let n_gqa = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    for h in 0..n_heads {
        let kv_h = h / n_gqa;
        let q_h = &q[h * head_dim..(h + 1) * head_dim];
        let out_h = &mut out[h * head_dim..(h + 1) * head_dim];
        for v in out_h.iter_mut() {
            *v = 0.0;
        }
        let mut m = f32::NEG_INFINITY;
        let mut l = 0.0f32;
        for t in 0..kv_len {
            let row_idx = kv_h * max_ctx + t;
            let k_off = row_idx * head_dim;
            let k_row = &k_q[k_off..k_off + head_dim];
            let k_row_scale = k_scales[row_idx];
            // Q · K row in dequantized space: factor k_row_scale and
            // attention scale out of the inner loop, same as the
            // non-flash kernel.
            let mut s = 0.0f32;
            for d in 0..head_dim {
                s += q_h[d] * (k_row[d] as f32);
            }
            s *= k_row_scale * scale;
            // Online softmax state update.
            let m_new = m.max(s);
            let rescale = if m.is_finite() { (m - m_new).exp() } else { 0.0 };
            let p = (s - m_new).exp();
            l = l * rescale + p;
            let v_off = row_idx * head_dim;
            let v_row = &v_q[v_off..v_off + head_dim];
            let v_row_scale = v_scales[row_idx];
            // Factor v_row_scale * p out of the inner loop.
            let p_eff = p * v_row_scale;
            for d in 0..head_dim {
                out_h[d] = out_h[d] * rescale + p_eff * (v_row[d] as f32);
            }
            m = m_new;
        }
        let inv_l = if l > 0.0 { 1.0 / l } else { 0.0 };
        for v in out_h.iter_mut() {
            *v *= inv_l;
        }
    }
}

/// Multi-query FlashAttention prefill for Q8_0 KV. Same online-
/// softmax + inline-dequant pattern as
/// [`gqa_attention_flash_decode_q8_0`] but processes `n_new`
/// queries in one call with causal masking — query `i` attends to
/// `[0, kv_len_base + i + 1)`.
///
/// Q layout: `[n_new, n_heads, head_dim]`. Out matches. K and V
/// cache are the standard `[n_kv_heads, max_ctx, head_dim]` i8
/// slabs the engine writes after quantization. The kernel
/// inlines per-row dequant (× row_scale) for both the Q·K dot and
/// the V accumulation, factoring the scale out of the inner
/// d-loop — same factoring as the decode kernel.
///
/// Scalar today; the AVX-2 / AVX-512 specializations follow the
/// same shape as the decode SIMD variants once the prefill path
/// has a wall-clock budget that needs them.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_flash_prefill_q8_0(
    q: &[f32],
    k_q: &[i8],
    k_scales: &[f32],
    v_q: &[i8],
    v_scales: &[f32],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len_base: usize,
    n_new: usize,
) {
    debug_assert_eq!(q.len(), n_new * n_heads * head_dim);
    debug_assert_eq!(out.len(), n_new * n_heads * head_dim);
    debug_assert_eq!(k_q.len(), n_kv_heads * max_ctx * head_dim);
    debug_assert_eq!(v_q.len(), n_kv_heads * max_ctx * head_dim);
    debug_assert_eq!(k_scales.len(), n_kv_heads * max_ctx);
    debug_assert_eq!(v_scales.len(), n_kv_heads * max_ctx);
    debug_assert!(kv_len_base + n_new <= max_ctx);
    if n_new == 0 {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            // SAFETY: runtime feature detection above.
            unsafe {
                gqa_attention_flash_prefill_q8_0_avx512(
                    q, k_q, k_scales, v_q, v_scales, out, n_heads, n_kv_heads,
                    head_dim, max_ctx, kv_len_base, n_new,
                );
            }
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime feature detection above.
            unsafe {
                gqa_attention_flash_prefill_q8_0_avx2(
                    q, k_q, k_scales, v_q, v_scales, out, n_heads, n_kv_heads,
                    head_dim, max_ctx, kv_len_base, n_new,
                );
            }
            return;
        }
    }
    gqa_attention_flash_prefill_q8_0_scalar(
        q, k_q, k_scales, v_q, v_scales, out, n_heads, n_kv_heads,
        head_dim, max_ctx, kv_len_base, n_new,
    );
}

#[allow(clippy::too_many_arguments)]
fn gqa_attention_flash_prefill_q8_0_scalar(
    q: &[f32],
    k_q: &[i8],
    k_scales: &[f32],
    v_q: &[i8],
    v_scales: &[f32],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len_base: usize,
    n_new: usize,
) {
    let n_gqa = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    for i in 0..n_new {
        let kv_len_i = kv_len_base + i + 1;
        let q_base = i * n_heads * head_dim;
        let out_base = i * n_heads * head_dim;
        for h in 0..n_heads {
            let kv_h = h / n_gqa;
            let q_h = &q[q_base + h * head_dim..q_base + (h + 1) * head_dim];
            let out_h = &mut out[out_base + h * head_dim..out_base + (h + 1) * head_dim];
            for v in out_h.iter_mut() {
                *v = 0.0;
            }
            let mut m = f32::NEG_INFINITY;
            let mut l = 0.0f32;
            for t in 0..kv_len_i {
                let row_idx = kv_h * max_ctx + t;
                let k_off = row_idx * head_dim;
                let k_row = &k_q[k_off..k_off + head_dim];
                let k_row_scale = k_scales[row_idx];
                let mut s = 0.0f32;
                for d in 0..head_dim {
                    s += q_h[d] * (k_row[d] as f32);
                }
                s *= k_row_scale * scale;
                let m_new = m.max(s);
                let rescale = if m.is_finite() { (m - m_new).exp() } else { 0.0 };
                let p = (s - m_new).exp();
                l = l * rescale + p;
                let v_off = row_idx * head_dim;
                let v_row = &v_q[v_off..v_off + head_dim];
                let v_row_scale = v_scales[row_idx];
                let p_eff = p * v_row_scale;
                for d in 0..head_dim {
                    out_h[d] = out_h[d] * rescale + p_eff * (v_row[d] as f32);
                }
                m = m_new;
            }
            let inv_l = if l > 0.0 { 1.0 / l } else { 0.0 };
            for v in out_h.iter_mut() {
                *v *= inv_l;
            }
        }
    }
}

/// AVX-2 + FMA specialization of [`gqa_attention_flash_prefill_q8_0`].
/// Inner d-loops vectorized at 8 floats / iter; i8 K and V rows
/// widened to f32x8 via `_mm256_cvtepi8_epi32` + `_mm256_cvtepi32_ps`
/// like the matching decode kernel. The outer (query, head, t)
/// loops stay scalar because of the cross-`t` online-softmax
/// recurrence.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn gqa_attention_flash_prefill_q8_0_avx2(
    q: &[f32],
    k_q: &[i8],
    k_scales: &[f32],
    v_q: &[i8],
    v_scales: &[f32],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len_base: usize,
    n_new: usize,
) {
    use std::arch::x86_64::*;
    let n_gqa = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let head_dim_8 = head_dim & !7;

    #[inline(always)]
    unsafe fn load_i8x8_as_f32x8(ptr: *const i8) -> __m256 {
        let bytes = _mm_loadl_epi64(ptr as *const __m128i);
        let i32s = _mm256_cvtepi8_epi32(bytes);
        _mm256_cvtepi32_ps(i32s)
    }

    for i in 0..n_new {
        let kv_len_i = kv_len_base + i + 1;
        let q_base = i * n_heads * head_dim;
        let out_base = i * n_heads * head_dim;
        for h in 0..n_heads {
            let kv_h = h / n_gqa;
            let q_h = &q[q_base + h * head_dim..q_base + (h + 1) * head_dim];
            let out_h = &mut out[out_base + h * head_dim..out_base + (h + 1) * head_dim];
            // Zero accumulator.
            let mut d = 0;
            while d < head_dim_8 {
                _mm256_storeu_ps(out_h.as_mut_ptr().add(d), _mm256_setzero_ps());
                d += 8;
            }
            while d < head_dim {
                out_h[d] = 0.0;
                d += 1;
            }
            let mut m = f32::NEG_INFINITY;
            let mut l = 0.0f32;
            for t in 0..kv_len_i {
                let row_idx = kv_h * max_ctx + t;
                let k_off = row_idx * head_dim;
                let k_row = &k_q[k_off..k_off + head_dim];
                let k_row_scale = k_scales[row_idx];
                // Q · K with i8 widen.
                let mut acc = _mm256_setzero_ps();
                let mut d = 0;
                while d < head_dim_8 {
                    let qv = _mm256_loadu_ps(q_h.as_ptr().add(d));
                    let kv = load_i8x8_as_f32x8(k_row.as_ptr().add(d));
                    acc = _mm256_fmadd_ps(qv, kv, acc);
                    d += 8;
                }
                let mut sum128 = _mm_add_ps(
                    _mm256_castps256_ps128(acc),
                    _mm256_extractf128_ps(acc, 1),
                );
                sum128 = _mm_hadd_ps(sum128, sum128);
                sum128 = _mm_hadd_ps(sum128, sum128);
                let mut s = _mm_cvtss_f32(sum128);
                while d < head_dim {
                    s += q_h[d] * (k_row[d] as f32);
                    d += 1;
                }
                s *= k_row_scale * scale;
                let m_new = m.max(s);
                let rescale = if m.is_finite() { (m - m_new).exp() } else { 0.0 };
                let p = (s - m_new).exp();
                l = l * rescale + p;
                let v_off = row_idx * head_dim;
                let v_row = &v_q[v_off..v_off + head_dim];
                let v_row_scale = v_scales[row_idx];
                let p_eff = p * v_row_scale;
                let rescale_v = _mm256_set1_ps(rescale);
                let p_eff_v = _mm256_set1_ps(p_eff);
                let mut d = 0;
                while d < head_dim_8 {
                    let cur = _mm256_loadu_ps(out_h.as_ptr().add(d));
                    let scaled = _mm256_mul_ps(cur, rescale_v);
                    let vv = load_i8x8_as_f32x8(v_row.as_ptr().add(d));
                    let updated = _mm256_fmadd_ps(p_eff_v, vv, scaled);
                    _mm256_storeu_ps(out_h.as_mut_ptr().add(d), updated);
                    d += 8;
                }
                while d < head_dim {
                    out_h[d] = out_h[d] * rescale + p_eff * (v_row[d] as f32);
                    d += 1;
                }
                m = m_new;
            }
            let inv_l = if l > 0.0 { 1.0 / l } else { 0.0 };
            let inv_l_v = _mm256_set1_ps(inv_l);
            let mut d = 0;
            while d < head_dim_8 {
                let cur = _mm256_loadu_ps(out_h.as_ptr().add(d));
                _mm256_storeu_ps(out_h.as_mut_ptr().add(d), _mm256_mul_ps(cur, inv_l_v));
                d += 8;
            }
            while d < head_dim {
                out_h[d] *= inv_l;
                d += 1;
            }
        }
    }
}

/// AVX-512 specialization of [`gqa_attention_flash_prefill_q8_0`].
/// 16 floats / iter; same i8 widen pattern as the decode AVX-512
/// kernel; `_mm512_reduce_add_ps` for the dot horizontal sum.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f,avx512bw")]
#[allow(clippy::too_many_arguments)]
unsafe fn gqa_attention_flash_prefill_q8_0_avx512(
    q: &[f32],
    k_q: &[i8],
    k_scales: &[f32],
    v_q: &[i8],
    v_scales: &[f32],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len_base: usize,
    n_new: usize,
) {
    use std::arch::x86_64::*;
    let n_gqa = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let head_dim_16 = head_dim & !15;

    #[inline(always)]
    unsafe fn load_i8x16_as_f32x16(ptr: *const i8) -> __m512 {
        let bytes = _mm_loadu_si128(ptr as *const __m128i);
        let i32s = _mm512_cvtepi8_epi32(bytes);
        _mm512_cvtepi32_ps(i32s)
    }

    for i in 0..n_new {
        let kv_len_i = kv_len_base + i + 1;
        let q_base = i * n_heads * head_dim;
        let out_base = i * n_heads * head_dim;
        for h in 0..n_heads {
            let kv_h = h / n_gqa;
            let q_h = &q[q_base + h * head_dim..q_base + (h + 1) * head_dim];
            let out_h = &mut out[out_base + h * head_dim..out_base + (h + 1) * head_dim];
            let mut d = 0;
            while d < head_dim_16 {
                _mm512_storeu_ps(out_h.as_mut_ptr().add(d), _mm512_setzero_ps());
                d += 16;
            }
            while d < head_dim {
                out_h[d] = 0.0;
                d += 1;
            }
            let mut m = f32::NEG_INFINITY;
            let mut l = 0.0f32;
            for t in 0..kv_len_i {
                let row_idx = kv_h * max_ctx + t;
                let k_off = row_idx * head_dim;
                let k_row = &k_q[k_off..k_off + head_dim];
                let k_row_scale = k_scales[row_idx];
                let mut acc = _mm512_setzero_ps();
                let mut d = 0;
                while d < head_dim_16 {
                    let qv = _mm512_loadu_ps(q_h.as_ptr().add(d));
                    let kv = load_i8x16_as_f32x16(k_row.as_ptr().add(d));
                    acc = _mm512_fmadd_ps(qv, kv, acc);
                    d += 16;
                }
                let mut s = _mm512_reduce_add_ps(acc);
                while d < head_dim {
                    s += q_h[d] * (k_row[d] as f32);
                    d += 1;
                }
                s *= k_row_scale * scale;
                let m_new = m.max(s);
                let rescale = if m.is_finite() { (m - m_new).exp() } else { 0.0 };
                let p = (s - m_new).exp();
                l = l * rescale + p;
                let v_off = row_idx * head_dim;
                let v_row = &v_q[v_off..v_off + head_dim];
                let v_row_scale = v_scales[row_idx];
                let p_eff = p * v_row_scale;
                let rescale_v = _mm512_set1_ps(rescale);
                let p_eff_v = _mm512_set1_ps(p_eff);
                let mut d = 0;
                while d < head_dim_16 {
                    let cur = _mm512_loadu_ps(out_h.as_ptr().add(d));
                    let scaled = _mm512_mul_ps(cur, rescale_v);
                    let vv = load_i8x16_as_f32x16(v_row.as_ptr().add(d));
                    let updated = _mm512_fmadd_ps(p_eff_v, vv, scaled);
                    _mm512_storeu_ps(out_h.as_mut_ptr().add(d), updated);
                    d += 16;
                }
                while d < head_dim {
                    out_h[d] = out_h[d] * rescale + p_eff * (v_row[d] as f32);
                    d += 1;
                }
                m = m_new;
            }
            let inv_l = if l > 0.0 { 1.0 / l } else { 0.0 };
            let inv_l_v = _mm512_set1_ps(inv_l);
            let mut d = 0;
            while d < head_dim_16 {
                let cur = _mm512_loadu_ps(out_h.as_ptr().add(d));
                _mm512_storeu_ps(out_h.as_mut_ptr().add(d), _mm512_mul_ps(cur, inv_l_v));
                d += 16;
            }
            while d < head_dim {
                out_h[d] *= inv_l;
                d += 1;
            }
        }
    }
}

/// AVX-512 specialization of [`gqa_attention_flash_decode_q8_0`].
/// Widens 16 i8s → f32x16 per iteration via
/// `_mm512_cvtepi8_epi32` + `_mm512_cvtepi32_ps` (matching the
/// non-flash AVX-512 Q8_0 kernel) and uses `_mm512_reduce_add_ps`
/// for the horizontal sum.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f,avx512bw")]
#[allow(clippy::too_many_arguments)]
unsafe fn gqa_attention_flash_decode_q8_0_avx512(
    q: &[f32],
    k_q: &[i8],
    k_scales: &[f32],
    v_q: &[i8],
    v_scales: &[f32],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len: usize,
) {
    use std::arch::x86_64::*;
    let n_gqa = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let head_dim_16 = head_dim & !15;

    #[inline(always)]
    unsafe fn load_i8x16_as_f32x16(ptr: *const i8) -> __m512 {
        let bytes = _mm_loadu_si128(ptr as *const __m128i);
        let i32s = _mm512_cvtepi8_epi32(bytes);
        _mm512_cvtepi32_ps(i32s)
    }

    for h in 0..n_heads {
        let kv_h = h / n_gqa;
        let q_h = &q[h * head_dim..(h + 1) * head_dim];
        let out_h = &mut out[h * head_dim..(h + 1) * head_dim];
        let mut d = 0;
        while d < head_dim_16 {
            _mm512_storeu_ps(out_h.as_mut_ptr().add(d), _mm512_setzero_ps());
            d += 16;
        }
        while d < head_dim {
            out_h[d] = 0.0;
            d += 1;
        }
        let mut m = f32::NEG_INFINITY;
        let mut l = 0.0f32;
        for t in 0..kv_len {
            let row_idx = kv_h * max_ctx + t;
            let k_off = row_idx * head_dim;
            let k_row = &k_q[k_off..k_off + head_dim];
            let k_row_scale = k_scales[row_idx];
            let mut acc = _mm512_setzero_ps();
            let mut d = 0;
            while d < head_dim_16 {
                let qv = _mm512_loadu_ps(q_h.as_ptr().add(d));
                let kv = load_i8x16_as_f32x16(k_row.as_ptr().add(d));
                acc = _mm512_fmadd_ps(qv, kv, acc);
                d += 16;
            }
            let mut s = _mm512_reduce_add_ps(acc);
            while d < head_dim {
                s += q_h[d] * (k_row[d] as f32);
                d += 1;
            }
            s *= k_row_scale * scale;
            let m_new = m.max(s);
            let rescale = if m.is_finite() { (m - m_new).exp() } else { 0.0 };
            let p = (s - m_new).exp();
            l = l * rescale + p;
            let v_off = row_idx * head_dim;
            let v_row = &v_q[v_off..v_off + head_dim];
            let v_row_scale = v_scales[row_idx];
            let p_eff = p * v_row_scale;
            let rescale_v = _mm512_set1_ps(rescale);
            let p_eff_v = _mm512_set1_ps(p_eff);
            let mut d = 0;
            while d < head_dim_16 {
                let cur = _mm512_loadu_ps(out_h.as_ptr().add(d));
                let scaled = _mm512_mul_ps(cur, rescale_v);
                let vv = load_i8x16_as_f32x16(v_row.as_ptr().add(d));
                let updated = _mm512_fmadd_ps(p_eff_v, vv, scaled);
                _mm512_storeu_ps(out_h.as_mut_ptr().add(d), updated);
                d += 16;
            }
            while d < head_dim {
                out_h[d] = out_h[d] * rescale + p_eff * (v_row[d] as f32);
                d += 1;
            }
            m = m_new;
        }
        let inv_l = if l > 0.0 { 1.0 / l } else { 0.0 };
        let inv_l_v = _mm512_set1_ps(inv_l);
        let mut d = 0;
        while d < head_dim_16 {
            let cur = _mm512_loadu_ps(out_h.as_ptr().add(d));
            _mm512_storeu_ps(out_h.as_mut_ptr().add(d), _mm512_mul_ps(cur, inv_l_v));
            d += 16;
        }
        while d < head_dim {
            out_h[d] *= inv_l;
            d += 1;
        }
    }
}

/// AVX-2 + FMA specialization of [`gqa_attention_flash_decode_q8_0`].
/// Same online-softmax recurrence; the inner d-loops widen i8 → f32
/// via `_mm256_cvtepi8_epi32` + `_mm256_cvtepi32_ps` (matching the
/// non-flash AVX-2 Q8_0 kernel) and vectorize at 8 floats / iter.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn gqa_attention_flash_decode_q8_0_avx2(
    q: &[f32],
    k_q: &[i8],
    k_scales: &[f32],
    v_q: &[i8],
    v_scales: &[f32],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len: usize,
) {
    use std::arch::x86_64::*;
    let n_gqa = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let head_dim_8 = head_dim & !7;

    // Load 8 i8s starting at `ptr` and widen to a YMM f32x8.
    #[inline(always)]
    unsafe fn load_i8x8_as_f32x8(ptr: *const i8) -> __m256 {
        let bytes = _mm_loadl_epi64(ptr as *const __m128i);
        let i32s = _mm256_cvtepi8_epi32(bytes);
        _mm256_cvtepi32_ps(i32s)
    }

    for h in 0..n_heads {
        let kv_h = h / n_gqa;
        let q_h = &q[h * head_dim..(h + 1) * head_dim];
        let out_h = &mut out[h * head_dim..(h + 1) * head_dim];
        // Zero the accumulator.
        let mut d = 0;
        while d < head_dim_8 {
            _mm256_storeu_ps(out_h.as_mut_ptr().add(d), _mm256_setzero_ps());
            d += 8;
        }
        while d < head_dim {
            out_h[d] = 0.0;
            d += 1;
        }
        let mut m = f32::NEG_INFINITY;
        let mut l = 0.0f32;
        for t in 0..kv_len {
            let row_idx = kv_h * max_ctx + t;
            let k_off = row_idx * head_dim;
            let k_row = &k_q[k_off..k_off + head_dim];
            let k_row_scale = k_scales[row_idx];
            // Q · K row, vectorized i8→f32 dot product.
            let mut acc = _mm256_setzero_ps();
            let mut d = 0;
            while d < head_dim_8 {
                let qv = _mm256_loadu_ps(q_h.as_ptr().add(d));
                let kv = load_i8x8_as_f32x8(k_row.as_ptr().add(d));
                acc = _mm256_fmadd_ps(qv, kv, acc);
                d += 8;
            }
            let mut sum128 = _mm_add_ps(
                _mm256_castps256_ps128(acc),
                _mm256_extractf128_ps(acc, 1),
            );
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            let mut s = _mm_cvtss_f32(sum128);
            while d < head_dim {
                s += q_h[d] * (k_row[d] as f32);
                d += 1;
            }
            s *= k_row_scale * scale;
            // Online softmax state update.
            let m_new = m.max(s);
            let rescale = if m.is_finite() { (m - m_new).exp() } else { 0.0 };
            let p = (s - m_new).exp();
            l = l * rescale + p;
            // V accumulation, vectorized: out_h = out_h*rescale + p_eff*v_row.
            let v_off = row_idx * head_dim;
            let v_row = &v_q[v_off..v_off + head_dim];
            let v_row_scale = v_scales[row_idx];
            let p_eff = p * v_row_scale;
            let rescale_v = _mm256_set1_ps(rescale);
            let p_eff_v = _mm256_set1_ps(p_eff);
            let mut d = 0;
            while d < head_dim_8 {
                let cur = _mm256_loadu_ps(out_h.as_ptr().add(d));
                let scaled = _mm256_mul_ps(cur, rescale_v);
                let vv = load_i8x8_as_f32x8(v_row.as_ptr().add(d));
                let updated = _mm256_fmadd_ps(p_eff_v, vv, scaled);
                _mm256_storeu_ps(out_h.as_mut_ptr().add(d), updated);
                d += 8;
            }
            while d < head_dim {
                out_h[d] = out_h[d] * rescale + p_eff * (v_row[d] as f32);
                d += 1;
            }
            m = m_new;
        }
        let inv_l = if l > 0.0 { 1.0 / l } else { 0.0 };
        let inv_l_v = _mm256_set1_ps(inv_l);
        let mut d = 0;
        while d < head_dim_8 {
            let cur = _mm256_loadu_ps(out_h.as_ptr().add(d));
            _mm256_storeu_ps(out_h.as_mut_ptr().add(d), _mm256_mul_ps(cur, inv_l_v));
            d += 8;
        }
        while d < head_dim {
            out_h[d] *= inv_l;
            d += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn gqa_attention_one_step_q8_0_avx2(
    q: &[f32],
    k_q: &[i8],
    k_scales: &[f32],
    v_q: &[i8],
    v_scales: &[f32],
    out: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_len: usize,
) {
    use std::arch::x86_64::*;
    let n_gqa = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let head_dim_8 = head_dim & !7;
    let mut scores = vec![0.0f32; kv_len];

    // Helper: load 8 i8s starting at `ptr` and widen to a YMM f32 vector.
    // Uses `_mm_loadl_epi64` to pull 8 bytes, `_mm256_cvtepi8_epi32` to
    // sign-extend to 8× i32, then `_mm256_cvtepi32_ps` to f32.
    #[inline(always)]
    unsafe fn load_i8x8_as_f32x8(ptr: *const i8) -> __m256 {
        let bytes = _mm_loadl_epi64(ptr as *const __m128i);
        let i32s = _mm256_cvtepi8_epi32(bytes);
        _mm256_cvtepi32_ps(i32s)
    }

    for h in 0..n_heads {
        let kv_h = h / n_gqa;
        let q_h = &q[h * head_dim..(h + 1) * head_dim];

        // QKᵀ — accumulate q · k_row, multiply by row_scale * scale at the end.
        for t in 0..kv_len {
            let row_idx = kv_h * max_ctx + t;
            let k_off = row_idx * head_dim;
            let k_row = &k_q[k_off..k_off + head_dim];
            let row_scale = k_scales[row_idx];
            let mut acc = _mm256_setzero_ps();
            let mut d = 0;
            while d < head_dim_8 {
                let qv = _mm256_loadu_ps(q_h.as_ptr().add(d));
                let kv = load_i8x8_as_f32x8(k_row.as_ptr().add(d));
                acc = _mm256_fmadd_ps(qv, kv, acc);
                d += 8;
            }
            // Horizontal reduce.
            let mut sum128 =
                _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            let mut acc_scalar = _mm_cvtss_f32(sum128);
            // Tail.
            while d < head_dim {
                acc_scalar += q_h[d] * (k_row[d] as f32);
                d += 1;
            }
            scores[t] = acc_scalar * row_scale * scale;
        }
        softmax_f32_inplace(&mut scores);

        // Weighted sum over V — out += s_eff * dequant(v_row), with
        // s_eff = scores[t] * row_scale (row_scale factored out of d-loop).
        let out_h = &mut out[h * head_dim..(h + 1) * head_dim];
        // Zero accumulator.
        let mut d = 0;
        while d < head_dim_8 {
            _mm256_storeu_ps(out_h.as_mut_ptr().add(d), _mm256_setzero_ps());
            d += 8;
        }
        while d < head_dim {
            out_h[d] = 0.0;
            d += 1;
        }

        for t in 0..kv_len {
            let row_idx = kv_h * max_ctx + t;
            let v_off = row_idx * head_dim;
            let v_row = &v_q[v_off..v_off + head_dim];
            let row_scale = v_scales[row_idx];
            let s_eff = scores[t] * row_scale;
            let s_vec = _mm256_set1_ps(s_eff);
            let mut d = 0;
            while d < head_dim_8 {
                let vv = load_i8x8_as_f32x8(v_row.as_ptr().add(d));
                let o = _mm256_loadu_ps(out_h.as_ptr().add(d));
                let nv = _mm256_fmadd_ps(s_vec, vv, o);
                _mm256_storeu_ps(out_h.as_mut_ptr().add(d), nv);
                d += 8;
            }
            while d < head_dim {
                out_h[d] += s_eff * (v_row[d] as f32);
                d += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reverse of [`unpack_q4k_scales`] used by the
    /// `rustllama-kernels-sycl` Q4_K_M / Q5_K_M parity tests. The
    /// SYCL parity tests need to *generate* a valid 12-byte scales
    /// block from a fresh `(sc[8], mn[8])` pair to feed to the
    /// kernel, then assert that unpacking that same block on the
    /// CPU yields the original values. If this inverse is wrong,
    /// the parity tests still pass (because both sides decode the
    /// "wrong" packing identically) and the kernel bug hides.
    /// Round-tripping it here in unit tests — which run on every
    /// `cargo test`, no GPU required — keeps the packer honest.
    fn pack_q4k_scales(sc: &[u8; 8], mn: &[u8; 8]) -> [u8; 12] {
        let mut out = [0u8; 12];
        for j in 0..4 {
            out[j] = sc[j] & 0x3F;
            out[j + 4] = mn[j] & 0x3F;
        }
        for j in 4..8 {
            out[j + 4] = (sc[j] & 0x0F) | ((mn[j] & 0x0F) << 4);
            out[j - 4] |= ((sc[j] >> 4) & 0x03) << 6;
            out[j] |= ((mn[j] >> 4) & 0x03) << 6;
        }
        out
    }

    #[test]
    fn q4k_scales_pack_unpack_roundtrip_exhaustive_low4() {
        // For j in 0..4, sc[j] / mn[j] are stored as full 6-bit
        // values (range 0..63). Exhaustively round-trip the first
        // four slots with all 64×64 combinations, keeping the
        // upper four slots zero.
        for sc0 in 0..64u8 {
            for mn0 in 0..64u8 {
                let mut sc = [0u8; 8];
                let mut mn = [0u8; 8];
                sc[0] = sc0;
                mn[0] = mn0;
                let packed = pack_q4k_scales(&sc, &mn);
                let (sc_back, mn_back) = unpack_q4k_scales(&packed);
                assert_eq!(sc, sc_back, "sc round-trip @ sc0={sc0}, mn0={mn0}");
                assert_eq!(mn, mn_back, "mn round-trip @ sc0={sc0}, mn0={mn0}");
            }
        }
    }

    #[test]
    fn q4k_scales_pack_unpack_roundtrip_exhaustive_high4() {
        // For j in 4..8, sc[j] / mn[j] split across bytes: low 4
        // bits in scales[j+4], high 2 bits in scales[j-4]. Walk
        // the full 64-value range for one high slot while leaving
        // the rest zero, then mirror for mn.
        for j in 4..8 {
            for v in 0..64u8 {
                let mut sc = [0u8; 8];
                let mut mn = [0u8; 8];
                sc[j] = v;
                let packed = pack_q4k_scales(&sc, &mn);
                let (sc_back, mn_back) = unpack_q4k_scales(&packed);
                assert_eq!(sc, sc_back, "sc[{j}] round-trip @ v={v}");
                assert_eq!(mn, mn_back, "mn unaffected @ sc[{j}]={v}");

                sc = [0u8; 8];
                mn[j] = v;
                let packed = pack_q4k_scales(&sc, &mn);
                let (sc_back, mn_back) = unpack_q4k_scales(&packed);
                assert_eq!(mn, mn_back, "mn[{j}] round-trip @ v={v}");
                assert_eq!(sc, sc_back, "sc unaffected @ mn[{j}]={v}");
            }
        }
    }

    #[test]
    fn q4k_scales_pack_unpack_roundtrip_mixed() {
        // Mixed-slot patterns: the parity tests in
        // `rustllama-kernels-sycl` build their packed scales from
        // formulas like `((row*7 + sb*11 + j*3) % 63)`, so they're
        // never all-zero in the way the per-slot tests above
        // cover. Verify a handful of dense `(sc, mn)` populations
        // round-trip cleanly.
        let patterns: &[fn(usize) -> u8] = &[
            |j| ((j * 7 + 3) % 63) as u8,
            |j| ((j * 11 + 17) % 63) as u8,
            |j| (63 - (j * 5) % 63) as u8,
            |j| ((j * 13 + j * j) % 63) as u8,
        ];
        for (sc_pat, mn_pat) in patterns.iter().flat_map(|s| patterns.iter().map(move |m| (s, m)))
        {
            let mut sc = [0u8; 8];
            let mut mn = [0u8; 8];
            for j in 0..8 {
                sc[j] = sc_pat(j);
                mn[j] = mn_pat(j);
            }
            let packed = pack_q4k_scales(&sc, &mn);
            let (sc_back, mn_back) = unpack_q4k_scales(&packed);
            assert_eq!(sc, sc_back, "sc mixed-pattern round-trip");
            assert_eq!(mn, mn_back, "mn mixed-pattern round-trip");
        }
    }

    #[test]
    fn flash_decode_matches_standard_attention() {
        // FlashAttention-decode is bit-for-bit identical to the
        // standard 3-pass implementation up to FP reduction order.
        // For a small (n_heads=4, head_dim=16, kv_len=32) workload
        // the difference should be within a few ULPs.
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 16;
        let max_ctx = 64;
        let kv_len = 32;
        // Deterministic synthetic inputs.
        let q: Vec<f32> = (0..n_heads * head_dim)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.13)
            .collect();
        let k_cache: Vec<f32> = (0..n_kv_heads * max_ctx * head_dim)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.07)
            .collect();
        let v_cache: Vec<f32> = (0..n_kv_heads * max_ctx * head_dim)
            .map(|i| ((i % 31) as f32 - 15.0) * 0.09)
            .collect();
        let mut out_standard = vec![0f32; n_heads * head_dim];
        let mut out_flash = vec![0f32; n_heads * head_dim];
        gqa_attention_one_step(
            &q, &k_cache, &v_cache, &mut out_standard,
            n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
        );
        gqa_attention_flash_decode(
            &q, &k_cache, &v_cache, &mut out_flash,
            n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
        );
        let mut max_err = 0f32;
        for (a, b) in out_standard.iter().zip(out_flash.iter()) {
            let e = (a - b).abs();
            if e > max_err {
                max_err = e;
            }
        }
        assert!(
            max_err < 1e-5,
            "flash vs standard attention max abs error {max_err} > 1e-5",
        );
    }

    #[test]
    fn flash_decode_q8_0_matches_standard_q8_0() {
        // Mirror of `flash_decode_matches_standard_attention` but for
        // the Q8_0 KV path. Synthetic 4-head × head_dim 16 × kv_len 32
        // workload. Quantization is the existing per-row absmax path
        // used by the engine.
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 16;
        let max_ctx = 64;
        let kv_len = 32;
        // Quantize deterministic K/V rows. Each row's scale is
        // absmax / 127; values quantize to i8 = round(v / scale).
        let mut k_q = vec![0i8; n_kv_heads * max_ctx * head_dim];
        let mut v_q = vec![0i8; n_kv_heads * max_ctx * head_dim];
        let mut k_scales = vec![0f32; n_kv_heads * max_ctx];
        let mut v_scales = vec![0f32; n_kv_heads * max_ctx];
        for kv_h in 0..n_kv_heads {
            for t in 0..kv_len {
                let row_idx = kv_h * max_ctx + t;
                let off = row_idx * head_dim;
                // K row: deterministic synthetic values.
                let k_row: Vec<f32> = (0..head_dim)
                    .map(|d| ((kv_h * 7 + t * 3 + d) % 13) as f32 * 0.1 - 0.6)
                    .collect();
                let absmax = k_row.iter().fold(0f32, |a, b| a.max(b.abs()));
                let sc = if absmax > 0.0 { absmax / 127.0 } else { 1.0 };
                k_scales[row_idx] = sc;
                for d in 0..head_dim {
                    k_q[off + d] = (k_row[d] / sc).round().clamp(-128.0, 127.0) as i8;
                }
                let v_row: Vec<f32> = (0..head_dim)
                    .map(|d| ((kv_h * 5 + t * 2 + d) % 11) as f32 * 0.15 - 0.7)
                    .collect();
                let absmax = v_row.iter().fold(0f32, |a, b| a.max(b.abs()));
                let sc = if absmax > 0.0 { absmax / 127.0 } else { 1.0 };
                v_scales[row_idx] = sc;
                for d in 0..head_dim {
                    v_q[off + d] = (v_row[d] / sc).round().clamp(-128.0, 127.0) as i8;
                }
            }
        }
        let q: Vec<f32> = (0..n_heads * head_dim)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.12)
            .collect();
        let mut out_standard = vec![0f32; n_heads * head_dim];
        let mut out_flash = vec![0f32; n_heads * head_dim];
        gqa_attention_one_step_q8_0(
            &q, &k_q, &k_scales, &v_q, &v_scales, &mut out_standard,
            n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
        );
        gqa_attention_flash_decode_q8_0(
            &q, &k_q, &k_scales, &v_q, &v_scales, &mut out_flash,
            n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
        );
        let mut max_err = 0f32;
        for (a, b) in out_standard.iter().zip(out_flash.iter()) {
            let e = (a - b).abs();
            if e > max_err {
                max_err = e;
            }
        }
        assert!(
            max_err < 1e-5,
            "Q8_0 flash vs standard max abs error {max_err} > 1e-5",
        );
    }

    #[test]
    fn flash_prefill_q8_0_matches_decode_loop() {
        // Multi-query Q8_0 prefill kernel must match calling the
        // Q8_0 flash-decode kernel `n_new` times with the right
        // `kv_len` per call.
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 16;
        let max_ctx = 64;
        let kv_len_base = 8;
        let n_new = 6;
        // Quantize K/V for every position the kernel will touch.
        let mut k_q = vec![0i8; n_kv_heads * max_ctx * head_dim];
        let mut v_q = vec![0i8; n_kv_heads * max_ctx * head_dim];
        let mut k_scales = vec![0f32; n_kv_heads * max_ctx];
        let mut v_scales = vec![0f32; n_kv_heads * max_ctx];
        for kv_h in 0..n_kv_heads {
            for t in 0..(kv_len_base + n_new) {
                let row_idx = kv_h * max_ctx + t;
                let off = row_idx * head_dim;
                let k_row: Vec<f32> = (0..head_dim)
                    .map(|d| ((kv_h * 7 + t * 3 + d) % 13) as f32 * 0.1 - 0.6)
                    .collect();
                let absmax = k_row.iter().fold(0f32, |a, b| a.max(b.abs()));
                let sc = if absmax > 0.0 { absmax / 127.0 } else { 1.0 };
                k_scales[row_idx] = sc;
                for d in 0..head_dim {
                    k_q[off + d] = (k_row[d] / sc).round().clamp(-128.0, 127.0) as i8;
                }
                let v_row: Vec<f32> = (0..head_dim)
                    .map(|d| ((kv_h * 5 + t * 2 + d) % 11) as f32 * 0.15 - 0.7)
                    .collect();
                let absmax = v_row.iter().fold(0f32, |a, b| a.max(b.abs()));
                let sc = if absmax > 0.0 { absmax / 127.0 } else { 1.0 };
                v_scales[row_idx] = sc;
                for d in 0..head_dim {
                    v_q[off + d] = (v_row[d] / sc).round().clamp(-128.0, 127.0) as i8;
                }
            }
        }
        let q: Vec<f32> = (0..n_new * n_heads * head_dim)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.12)
            .collect();
        let mut out_prefill = vec![0f32; n_new * n_heads * head_dim];
        gqa_attention_flash_prefill_q8_0(
            &q, &k_q, &k_scales, &v_q, &v_scales, &mut out_prefill,
            n_heads, n_kv_heads, head_dim, max_ctx,
            kv_len_base, n_new,
        );
        let mut out_reference = vec![0f32; n_new * n_heads * head_dim];
        for i in 0..n_new {
            let kv_len_i = kv_len_base + i + 1;
            let q_slice = &q[i * n_heads * head_dim..(i + 1) * n_heads * head_dim];
            let out_slice =
                &mut out_reference[i * n_heads * head_dim..(i + 1) * n_heads * head_dim];
            gqa_attention_flash_decode_q8_0(
                q_slice, &k_q, &k_scales, &v_q, &v_scales, out_slice,
                n_heads, n_kv_heads, head_dim, max_ctx, kv_len_i,
            );
        }
        let mut max_err = 0f32;
        for (a, b) in out_prefill.iter().zip(out_reference.iter()) {
            let e = (a - b).abs();
            if e > max_err {
                max_err = e;
            }
        }
        assert!(
            max_err < 1e-5,
            "Q8_0 prefill vs decode-loop max abs error {max_err} > 1e-5",
        );
    }

    #[test]
    fn flash_prefill_matches_flash_decode_loop() {
        // Multi-query prefill flash kernel must match calling
        // flash-decode N times with the right per-query kv_len.
        // This is the parity gate: prefill is allowed to batch but
        // the per-query output stays bit-identical (up to FP order).
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 16;
        let max_ctx = 64;
        let kv_len_base = 8; // some history already cached
        let n_new = 12;     // a small prefill batch
        // Pre-fill K/V cache for positions [0, kv_len_base + n_new).
        let total_positions = kv_len_base + n_new;
        let k_cache: Vec<f32> = (0..n_kv_heads * max_ctx * head_dim)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.07)
            .collect();
        let v_cache: Vec<f32> = (0..n_kv_heads * max_ctx * head_dim)
            .map(|i| ((i % 31) as f32 - 15.0) * 0.09)
            .collect();
        // Synthetic Q for all n_new positions.
        let q: Vec<f32> = (0..n_new * n_heads * head_dim)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.13)
            .collect();
        // Run the batched prefill kernel.
        let mut out_prefill = vec![0f32; n_new * n_heads * head_dim];
        gqa_attention_flash_prefill(
            &q, &k_cache, &v_cache, &mut out_prefill,
            n_heads, n_kv_heads, head_dim, max_ctx,
            kv_len_base, n_new,
        );
        // Reference: call flash-decode once per new position with
        // kv_len = kv_len_base + i + 1 (causal).
        let mut out_reference = vec![0f32; n_new * n_heads * head_dim];
        for i in 0..n_new {
            let kv_len_i = kv_len_base + i + 1;
            let q_slice = &q[i * n_heads * head_dim..(i + 1) * n_heads * head_dim];
            let out_slice = &mut out_reference
                [i * n_heads * head_dim..(i + 1) * n_heads * head_dim];
            gqa_attention_flash_decode(
                q_slice, &k_cache, &v_cache, out_slice,
                n_heads, n_kv_heads, head_dim, max_ctx, kv_len_i,
            );
        }
        let _ = total_positions; // documented intent only.
        let mut max_err = 0f32;
        for (a, b) in out_prefill.iter().zip(out_reference.iter()) {
            let e = (a - b).abs();
            if e > max_err {
                max_err = e;
            }
        }
        assert!(
            max_err < 1e-5,
            "flash_prefill vs flash_decode-loop max abs error {max_err} > 1e-5",
        );
    }

    #[test]
    fn flash_prefill_empty_batch_is_noop() {
        // n_new = 0 should leave the (empty) output untouched
        // and not divide-by-zero anywhere.
        let mut out = vec![];
        gqa_attention_flash_prefill(
            &[], &[0f32; 4], &[0f32; 4], &mut out,
            2, 2, 1, 2, 0, 0,
        );
        assert!(out.is_empty());
    }

    #[test]
    fn flash_prefill_matches_n_flash_decodes() {
        // The batched prefill kernel must produce identical output to
        // calling flash-decode `n_new` times — once per query, each
        // with its own causal kv_len. Synthetic 4-head × head_dim 16
        // × kv_len_base 24 + n_new 8 = 32 total. Query i corresponds
        // to cache position 24+i and sees [0, 24+i+1).
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 16;
        let max_ctx = 64;
        let kv_len_base = 24;
        let n_new = 8;
        let q: Vec<f32> = (0..n_new * n_heads * head_dim)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.13)
            .collect();
        let k_cache: Vec<f32> = (0..n_kv_heads * max_ctx * head_dim)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.07)
            .collect();
        let v_cache: Vec<f32> = (0..n_kv_heads * max_ctx * head_dim)
            .map(|i| ((i % 31) as f32 - 15.0) * 0.09)
            .collect();
        // Reference: call flash-decode once per query, passing the
        // appropriate causal kv_len. Decode reads Q as `[n_heads,
        // head_dim]` (one query), so we extract per-query slices.
        let mut out_reference = vec![0f32; n_new * n_heads * head_dim];
        for q_pos in 0..n_new {
            let q_off = q_pos * n_heads * head_dim;
            let q_one = &q[q_off..q_off + n_heads * head_dim];
            let causal_kv_len = kv_len_base + q_pos + 1;
            let mut out_one = vec![0f32; n_heads * head_dim];
            gqa_attention_flash_decode(
                q_one, &k_cache, &v_cache, &mut out_one,
                n_heads, n_kv_heads, head_dim, max_ctx, causal_kv_len,
            );
            out_reference[q_off..q_off + n_heads * head_dim]
                .copy_from_slice(&out_one);
        }
        // Batched prefill.
        let mut out_batched = vec![0f32; n_new * n_heads * head_dim];
        gqa_attention_flash_prefill(
            &q, &k_cache, &v_cache, &mut out_batched,
            n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, n_new,
        );
        let mut max_err = 0f32;
        for (a, b) in out_reference.iter().zip(out_batched.iter()) {
            let e = (a - b).abs();
            if e > max_err {
                max_err = e;
            }
        }
        assert!(
            max_err < 1e-5,
            "flash_prefill vs n×flash_decode max abs error {max_err} > 1e-5",
        );
    }

    #[test]
    fn flash_prefill_zero_n_new_is_noop() {
        // n_new == 0 is reachable when the caller passes an empty
        // batch (e.g. all-cached prompt). Must not write to `out`
        // and not panic on the empty state allocations.
        let n_heads = 2;
        let n_kv_heads = 2;
        let head_dim = 4;
        let max_ctx = 8;
        let q: Vec<f32> = vec![];
        let k_cache = vec![0f32; n_kv_heads * max_ctx * head_dim];
        let v_cache = vec![0f32; n_kv_heads * max_ctx * head_dim];
        let mut out: Vec<f32> = vec![];
        gqa_attention_flash_prefill(
            &q, &k_cache, &v_cache, &mut out,
            n_heads, n_kv_heads, head_dim, max_ctx,
            /*kv_len_base*/ 0, /*n_new*/ 0,
        );
        // No assertion needed — the test passes if no panic.
    }

    #[test]
    fn flash_decode_empty_kv_zeros_output() {
        // kv_len=0 is reachable on the very first prefill chunk;
        // it must zero `out` rather than divide-by-zero.
        let mut out = vec![999.0f32; 8];
        gqa_attention_flash_decode(
            &[0f32; 8], &[], &[], &mut out, 2, 2, 4, 0, 0,
        );
        assert_eq!(out, vec![0f32; 8]);
    }

    #[test]
    fn gemm_identity() {
        let a = vec![1.0f32, 0.0, 0.0, 1.0];
        let b = vec![2.0f32, 3.0, 4.0, 5.0];
        let mut c = vec![0.0f32; 4];
        gemm_f32(&a, &b, &mut c, 2, 2, 2);
        assert_eq!(c, vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn gemm_f16_matches_f32() {
        // W = [[1, 2, 3], [4, 5, 6]], x = [1, 1, 1] -> y = [6, 15]
        let w_f32 = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let w_f16: Vec<f16> = w_f32.iter().map(|v| f16::from_f32(*v)).collect();
        let x = vec![1.0f32, 1.0, 1.0];
        let mut y = vec![0.0f32; 2];
        matvec_f16_w_f32_a(&w_f16, &x, &mut y, 2, 3);
        assert!((y[0] - 6.0).abs() < 1e-4);
        assert!((y[1] - 15.0).abs() < 1e-4);
    }

    #[test]
    fn rmsnorm_unit_input() {
        let x = vec![1.0f32; 4];
        let w = vec![1.0f32; 4];
        let mut y = vec![0.0f32; 4];
        rmsnorm_f32_row(&x, &w, &mut y, 1e-6);
        for v in &y {
            assert!((v - 1.0).abs() < 1e-5, "got {v}");
        }
    }

    #[test]
    fn rope_zero_position_is_identity() {
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let orig = x.clone();
        rope_inplace_neox(&mut x, 1, 8, 0, 10000.0);
        for (a, b) in x.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn rope_preserves_norm() {
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let n_in: f32 = x.iter().map(|v| v * v).sum();
        rope_inplace_neox(&mut x, 1, 8, 42, 10000.0);
        let n_out: f32 = x.iter().map(|v| v * v).sum();
        assert!((n_in - n_out).abs() < 1e-3, "in {n_in} out {n_out}");
    }

    #[test]
    fn softmax_uniform() {
        let mut x = vec![0.5f32; 5];
        softmax_f32_inplace(&mut x);
        for v in &x {
            assert!((v - 0.2).abs() < 1e-6);
        }
    }

    #[test]
    fn softmax_handles_large_logits() {
        let mut x = vec![1000.0f32, 1000.0, 1000.0];
        softmax_f32_inplace(&mut x);
        for v in &x {
            assert!((v - 1.0 / 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn embedding_lookup_basic() {
        // vocab = 3, d = 2 -> table = [[1,2],[3,4],[5,6]]
        let table: Vec<f16> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .map(|v| f16::from_f32(*v))
            .collect();
        let ids = vec![2i32, 0];
        let mut out = vec![0.0f32; 4];
        embed_lookup_f16_to_f32(&table, &ids, &mut out, 2);
        assert_eq!(out, vec![5.0, 6.0, 1.0, 2.0]);
    }

    /// Run with `cargo test -p rustllama-kernels-cpu --release matvec_perf -- --nocapture --ignored`
    #[test]
    #[ignore]
    fn matvec_perf_avx2_vs_scalar() {
        let m = 896;
        let k = 4864;
        let mut w = vec![0f32; m * k];
        let mut x = vec![0f32; k];
        let mut s: u32 = 0xC0FFEE;
        for v in w.iter_mut().chain(x.iter_mut()) {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = (s as i32 as f32) / (i32::MAX as f32) * 0.05;
        }
        let mut out = vec![0f32; m];
        let iters = 100;

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            matvec_f32_scalar(&w, &x, &mut out, m, k);
        }
        let scalar_t = t0.elapsed();
        let scalar_sum: f32 = out.iter().sum();

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            matvec_f32(&w, &x, &mut out, m, k);
        }
        let dispatch_t = t0.elapsed();
        let dispatch_sum: f32 = out.iter().sum();

        #[cfg(target_arch = "x86_64")]
        let m_avx = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        #[cfg(not(target_arch = "x86_64"))]
        let m_avx = false;
        eprintln!(
            "matvec_perf m={m} k={k} iters={iters} (avx2_detected={m_avx})\n  \
             scalar    : {scalar_t:?} ({:.2} ms/call)\n  \
             dispatch  : {dispatch_t:?} ({:.2} ms/call) speedup {:.2}×",
            scalar_t.as_secs_f64() * 1000.0 / iters as f64,
            dispatch_t.as_secs_f64() * 1000.0 / iters as f64,
            scalar_t.as_secs_f64() / dispatch_t.as_secs_f64()
        );
        // Sanity: both should sum to ~the same thing.
        assert!((scalar_sum - dispatch_sum).abs() / scalar_sum.abs().max(1e-6) < 1e-3);
    }

    /// Run with `cargo test -p rustllama-kernels-cpu --release matvec_q8_0_perf -- --nocapture --ignored`
    #[test]
    #[ignore]
    fn matvec_q8_0_perf_vs_f16() {
        const QK: usize = 32;
        const BLOCK_BYTES: usize = 34;
        // LM-head sized: vocab=151936, d_model=896. The bandwidth crossover
        // happens around m=32k for our DDR4 box — pick a size that exceeds
        // L3 to show the memory-bound regime where Q8_0 should win.
        let m: usize = std::env::var("RUSTLLAMA_BENCH_M")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(65536);
        let k = 896; // multiple of 32 ✓
        let blocks_per_row = k / QK;

        // Build deterministic Q8_0 bytes.
        let mut w_q8 = vec![0u8; m * blocks_per_row * BLOCK_BYTES];
        let mut s: u32 = 0xCAFE_BABE;
        for off in (0..w_q8.len()).step_by(BLOCK_BYTES) {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let d_f32 = ((s as i32 as f32) / (i32::MAX as f32)) * 0.05 + 0.001;
            let d_bytes = f16::from_f32(d_f32).to_le_bytes();
            w_q8[off] = d_bytes[0];
            w_q8[off + 1] = d_bytes[1];
            for j in 0..QK {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                w_q8[off + 2 + j] = (s >> 24) as u8;
            }
        }

        // Equivalent F16 view: dequant Q8_0 -> F32 -> reencode to F16.
        let mut w_f32 = vec![0f32; m * k];
        rustllama_gguf::dequant::dequant_q8_0(&w_q8, &mut w_f32);
        let w_f16: Vec<f16> = w_f32.iter().map(|v| f16::from_f32(*v)).collect();

        // Random x.
        let mut x = vec![0f32; k];
        for v in x.iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = (s as i32 as f32) / (i32::MAX as f32);
        }

        let mut out = vec![0f32; m];
        let iters = 200;

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            matvec_f16_w_f32_a(&w_f16, &x, &mut out, m, k);
        }
        let t_f16 = t0.elapsed();
        let s_f16: f32 = out.iter().sum();

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            matvec_q8_0_w_f32_a(&w_q8, &x, &mut out, m, k);
        }
        let t_q8 = t0.elapsed();
        let s_q8: f32 = out.iter().sum();

        eprintln!(
            "matvec_perf m={m} k={k} iters={iters}\n  \
             F16        : {t_f16:?} ({:.3} ms/call)\n  \
             Q8_0 direct: {t_q8:?} ({:.3} ms/call) speedup {:.2}×",
            t_f16.as_secs_f64() * 1000.0 / iters as f64,
            t_q8.as_secs_f64() * 1000.0 / iters as f64,
            t_f16.as_secs_f64() / t_q8.as_secs_f64()
        );
        // Both should sum to roughly the same thing.
        assert!((s_f16 - s_q8).abs() / s_f16.abs().max(1e-4) < 1e-2,
            "F16 sum {s_f16} vs Q8_0 sum {s_q8}");
    }

    #[test]
    fn matvec_q6_k_matches_dequant_then_matvec() {
        const QK_K: usize = 256;
        const BLOCK_BYTES: usize = 210;
        let m = 3;
        let k = 512; // 2 super-blocks per row
        let blocks_per_row = k / QK_K;
        let total_blocks = m * blocks_per_row;

        let mut w_bytes = vec![0u8; total_blocks * BLOCK_BYTES];
        let mut s: u32 = 0xDEAD_F00D;
        for b in 0..total_blocks {
            let off = b * BLOCK_BYTES;
            // ql[128] + qh[64] + scales[16] = first 208 random bytes
            for j in 0..208 {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                w_bytes[off + j] = (s >> 24) as u8;
            }
            // d at 208..210
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let d = ((s >> 24) as f32 / 256.0) * 0.05 + 0.001;
            w_bytes[off + 208..off + 210].copy_from_slice(&f16::from_f32(d).to_le_bytes());
        }

        let mut x = vec![0f32; k];
        for v in x.iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = (s as i32 as f32) / (i32::MAX as f32);
        }

        let mut out_fused = vec![0f32; m];
        matvec_q6_k_w_f32_a(&w_bytes, &x, &mut out_fused, m, k);

        // Reference: dequant_q6_k → matvec_f32_scalar
        let mut w_f32 = vec![0f32; m * k];
        rustllama_gguf::dequant::dequant_q6_k(&w_bytes, &mut w_f32);
        let mut out_ref = vec![0f32; m];
        matvec_f32_scalar(&w_f32, &x, &mut out_ref, m, k);

        for (i, (a, b)) in out_fused.iter().zip(out_ref.iter()).enumerate() {
            let abs_err = (a - b).abs();
            let rel_err = abs_err / b.abs().max(1e-6);
            assert!(
                abs_err < 1e-2 || rel_err < 1e-3,
                "row {i}: fused={a} ref={b} abs={abs_err} rel={rel_err}"
            );
        }
    }

    #[test]
    fn matvec_q5_k_matches_dequant_then_matvec() {
        const QK_K: usize = 256;
        const BLOCK_BYTES: usize = 176;
        let m = 3;
        let k = 512; // 2 super-blocks per row
        let blocks_per_row = k / QK_K;
        let total_blocks = m * blocks_per_row;

        let mut w_bytes = vec![0u8; total_blocks * BLOCK_BYTES];
        let mut s: u32 = 0xFACE_FEED;
        for b in 0..total_blocks {
            let off = b * BLOCK_BYTES;
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let d = ((s >> 24) as f32 / 256.0) * 0.05 + 0.001;
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let dmin = ((s >> 24) as f32 / 256.0) * 0.02 + 0.0005;
            w_bytes[off..off + 2].copy_from_slice(&f16::from_f32(d).to_le_bytes());
            w_bytes[off + 2..off + 4].copy_from_slice(&f16::from_f32(dmin).to_le_bytes());
            // scales[12] + qh[32] + qs[128]: random bytes (172 total)
            for j in 0..172 {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                w_bytes[off + 4 + j] = (s >> 24) as u8;
            }
        }

        let mut x = vec![0f32; k];
        for v in x.iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = (s as i32 as f32) / (i32::MAX as f32);
        }

        let mut out_fused = vec![0f32; m];
        matvec_q5_k_w_f32_a(&w_bytes, &x, &mut out_fused, m, k);

        // Reference: dequant_q5_k → matvec_f32_scalar
        let mut w_f32 = vec![0f32; m * k];
        rustllama_gguf::dequant::dequant_q5_k(&w_bytes, &mut w_f32);
        let mut out_ref = vec![0f32; m];
        matvec_f32_scalar(&w_f32, &x, &mut out_ref, m, k);

        for (i, (a, b)) in out_fused.iter().zip(out_ref.iter()).enumerate() {
            let abs_err = (a - b).abs();
            let rel_err = abs_err / b.abs().max(1e-6);
            assert!(
                abs_err < 1e-2 || rel_err < 1e-3,
                "row {i}: fused={a} ref={b} abs={abs_err} rel={rel_err}"
            );
        }
    }

    /// Build random IQ4_NL bytes (18 per block).
    fn random_iq4_nl_table(n_rows: usize, d: usize, seed: u32) -> Vec<u8> {
        const BLOCK_BYTES: usize = 18;
        const QK: usize = 32;
        let blocks = n_rows * (d / QK);
        let mut bytes = vec![0u8; blocks * BLOCK_BYTES];
        let mut s = seed;
        for b in 0..blocks {
            let off = b * BLOCK_BYTES;
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let scale = ((s >> 24) as f32 / 256.0) * 0.05 + 0.001;
            bytes[off..off + 2].copy_from_slice(&f16::from_f32(scale).to_le_bytes());
            for j in 0..16 {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                bytes[off + 2 + j] = (s >> 24) as u8;
            }
        }
        bytes
    }

    #[test]
    fn matvec_iq4_nl_matches_dequant_then_matvec() {
        let m = 3;
        let k = 128; // 4 blocks per row
        let w = random_iq4_nl_table(m, k, 0xCC0FFE);
        let mut sx = 0xBEAD_C0DEu32;
        let x: Vec<f32> = (0..k).map(|_| lcg(&mut sx)).collect();
        let mut out_fused = vec![0f32; m];
        matvec_iq4_nl_w_f32_a(&w, &x, &mut out_fused, m, k);
        let mut w_f32 = vec![0f32; m * k];
        rustllama_gguf::dequant::dequant_iq4_nl(&w, &mut w_f32);
        let mut out_ref = vec![0f32; m];
        matvec_f32_scalar(&w_f32, &x, &mut out_ref, m, k);
        for (i, (a, b)) in out_fused.iter().zip(out_ref.iter()).enumerate() {
            let abs = (a - b).abs();
            let rel = abs / b.abs().max(1e-6);
            assert!(
                abs < 1e-3 || rel < 1e-5,
                "row {i}: fused={a} ref={b} abs={abs} rel={rel}"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn matvec_iq4_nl_simd_matches_scalar() {
        let m = 4;
        let k = 128;
        let w = random_iq4_nl_table(m, k, 0xCC1FFE);
        let mut sx = 0xC0DE_BEEFu32;
        let x: Vec<f32> = (0..k).map(|_| lcg(&mut sx)).collect();

        let mut out_scalar = vec![0f32; m];
        matvec_iq4_nl_w_f32_a_scalar(&w, &x, &mut out_scalar, m, k);

        if is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && is_x86_feature_detected!("ssse3")
        {
            let mut out_avx2 = vec![0f32; m];
            unsafe { matvec_iq4_nl_w_f32_a_avx2(&w, &x, &mut out_avx2, m, k) };
            for (i, (a, b)) in out_scalar.iter().zip(out_avx2.iter()).enumerate() {
                let abs = (a - b).abs();
                let rel = abs / a.abs().max(1e-6);
                assert!(
                    abs < 1e-3 || rel < 1e-5,
                    "row {i}: scalar={a} avx2={b}"
                );
            }
        }
        if is_x86_feature_detected!("avx512f") {
            let mut out_avx512 = vec![0f32; m];
            unsafe { matvec_iq4_nl_w_f32_a_avx512(&w, &x, &mut out_avx512, m, k) };
            for (i, (a, b)) in out_scalar.iter().zip(out_avx512.iter()).enumerate() {
                let abs = (a - b).abs();
                let rel = abs / a.abs().max(1e-6);
                assert!(
                    abs < 1e-3 || rel < 1e-5,
                    "row {i}: scalar={a} avx512={b}"
                );
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn embed_lookup_iq4_nl_simd_matches_scalar() {
        let n_rows = 5;
        let d = 128;
        let table = random_iq4_nl_table(n_rows, d, 0xE110FFE);
        let ids = [0i32, 3, 4, 1];
        let mut out_scalar = vec![0f32; ids.len() * d];
        embed_lookup_iq4_nl_scalar(&table, &ids, &mut out_scalar, d);

        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("ssse3") {
            let mut out_avx2 = vec![0f32; ids.len() * d];
            unsafe { embed_lookup_iq4_nl_avx2(&table, &ids, &mut out_avx2, d) };
            for (i, (a, b)) in out_scalar.iter().zip(out_avx2.iter()).enumerate() {
                assert!((a - b).abs() < 1e-5, "avx2 idx {i}: scalar={a} avx2={b}");
            }
        }
        if is_x86_feature_detected!("avx512f") {
            let mut out_avx512 = vec![0f32; ids.len() * d];
            unsafe { embed_lookup_iq4_nl_avx512(&table, &ids, &mut out_avx512, d) };
            for (i, (a, b)) in out_scalar.iter().zip(out_avx512.iter()).enumerate() {
                assert!((a - b).abs() < 1e-5, "avx512 idx {i}: scalar={a} avx512={b}");
            }
        }
    }

    /// Build random IQ3_S bytes (110 per 256-weight super-block). All
    /// byte positions are filled with PRNG output — the grid is 512
    /// entries so any 9-bit index from `qs | qh-bit` is in-range, sign
    /// bits are arbitrary, and the scale nibbles map onto valid odd
    /// scales `1+2*n` for `n` in `0..15`.
    fn random_iq3_s_table(n_rows: usize, d: usize, seed: u32) -> Vec<u8> {
        const BLOCK_BYTES: usize = 110;
        const QK_K: usize = 256;
        let blocks = n_rows * (d / QK_K);
        let mut bytes = vec![0u8; blocks * BLOCK_BYTES];
        let mut s = seed;
        for b in 0..blocks {
            let off = b * BLOCK_BYTES;
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let scale = ((s >> 24) as f32 / 256.0) * 0.05 + 0.001;
            bytes[off..off + 2].copy_from_slice(&f16::from_f32(scale).to_le_bytes());
            for j in 2..BLOCK_BYTES {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                bytes[off + j] = (s >> 24) as u8;
            }
        }
        bytes
    }

    #[test]
    fn matvec_iq3_s_matches_dequant_then_matvec() {
        let m = 3;
        let k = 512; // 2 super-blocks per row
        let w = random_iq3_s_table(m, k, 0x1310FFE);
        let mut sx = 0xFACE_F00Du32;
        let x: Vec<f32> = (0..k).map(|_| lcg(&mut sx)).collect();
        let mut out_fused = vec![0f32; m];
        matvec_iq3_s_w_f32_a(&w, &x, &mut out_fused, m, k);
        let mut w_f32 = vec![0f32; m * k];
        rustllama_gguf::dequant::dequant_iq3_s(&w, &mut w_f32);
        let mut out_ref = vec![0f32; m];
        matvec_f32_scalar(&w_f32, &x, &mut out_ref, m, k);
        for (i, (a, b)) in out_fused.iter().zip(out_ref.iter()).enumerate() {
            let abs = (a - b).abs();
            let rel = abs / b.abs().max(1e-6);
            assert!(
                abs < 1e-3 || rel < 1e-5,
                "row {i}: fused={a} ref={b} abs={abs} rel={rel}"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn matvec_iq3_s_simd_paths_match_scalar() {
        let m = 4;
        let k = 512;
        let w = random_iq3_s_table(m, k, 0x13_5A_AC_E5u32);
        let mut sx = 0x13_AC_E51_7u32;
        let x: Vec<f32> = (0..k).map(|_| lcg(&mut sx)).collect();

        let mut out_scalar = vec![0f32; m];
        matvec_iq3_s_w_f32_a_scalar(&w, &x, &mut out_scalar, m, k);

        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let mut out_avx2 = vec![0f32; m];
            unsafe { matvec_iq3_s_w_f32_a_avx2(&w, &x, &mut out_avx2, m, k) };
            for (i, (a, b)) in out_scalar.iter().zip(out_avx2.iter()).enumerate() {
                let abs = (a - b).abs();
                let rel = abs / a.abs().max(1e-6);
                assert!(
                    abs < 1e-3 || rel < 1e-5,
                    "row {i}: scalar={a} avx2={b}"
                );
            }
        }
        if is_x86_feature_detected!("avx512f") {
            let mut out_avx512 = vec![0f32; m];
            unsafe { matvec_iq3_s_w_f32_a_avx512(&w, &x, &mut out_avx512, m, k) };
            for (i, (a, b)) in out_scalar.iter().zip(out_avx512.iter()).enumerate() {
                let abs = (a - b).abs();
                let rel = abs / a.abs().max(1e-6);
                assert!(
                    abs < 1e-3 || rel < 1e-5,
                    "row {i}: scalar={a} avx512={b}"
                );
            }
        }
    }

    #[test]
    fn embed_lookup_iq3_s_matches_dequant() {
        let n_rows = 4;
        let d = 256;
        let table = random_iq3_s_table(n_rows, d, 0xE131_0FFE);
        let ids = [2i32, 0, 3, 1];
        let mut out = vec![0f32; ids.len() * d];
        embed_lookup_iq3_s(&table, &ids, &mut out, d);
        const BLOCK_BYTES: usize = 110;
        const QK_K: usize = 256;
        let blocks_per_row = d / QK_K;
        for (slot, &id) in ids.iter().enumerate() {
            let row_off = id as usize * blocks_per_row * BLOCK_BYTES;
            let row = &table[row_off..row_off + blocks_per_row * BLOCK_BYTES];
            let mut ref_row = vec![0f32; d];
            rustllama_gguf::dequant::dequant_iq3_s(row, &mut ref_row);
            let got = &out[slot * d..(slot + 1) * d];
            for (j, (a, b)) in got.iter().zip(ref_row.iter()).enumerate() {
                assert!((a - b).abs() < 1e-6, "id {id} slot {slot} elem {j}: {a} vs {b}");
            }
        }
    }

    /// Build random IQ2_XXS bytes (66 per 256-weight super-block). The
    /// 256-entry grid covers any 8-bit `aux0` byte; the 7-bit sign
    /// indices in `aux1` are also fully covered by `KSIGNS_IQ2XS`.
    /// Top nibble of `aux1` is treated as a sub-block scale in 0..15.
    fn random_iq2_xxs_table(n_rows: usize, d: usize, seed: u32) -> Vec<u8> {
        const BLOCK_BYTES: usize = 66;
        const QK_K: usize = 256;
        let blocks = n_rows * (d / QK_K);
        let mut bytes = vec![0u8; blocks * BLOCK_BYTES];
        let mut s = seed;
        for b in 0..blocks {
            let off = b * BLOCK_BYTES;
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let scale = ((s >> 24) as f32 / 256.0) * 0.05 + 0.001;
            bytes[off..off + 2].copy_from_slice(&f16::from_f32(scale).to_le_bytes());
            for j in 2..BLOCK_BYTES {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                bytes[off + j] = (s >> 24) as u8;
            }
        }
        bytes
    }

    /// Build random IQ2_XS bytes (74 per 256-weight super-block).
    /// 9-bit grid indices and 7-bit sign indices come from the qs u16s
    /// and are fully covered by the 512-entry grid + 128-entry signs;
    /// each scales byte holds two 4-bit nibbles (range 0..15).
    fn random_iq2_xs_table(n_rows: usize, d: usize, seed: u32) -> Vec<u8> {
        const BLOCK_BYTES: usize = 74;
        const QK_K: usize = 256;
        let blocks = n_rows * (d / QK_K);
        let mut bytes = vec![0u8; blocks * BLOCK_BYTES];
        let mut s = seed;
        for b in 0..blocks {
            let off = b * BLOCK_BYTES;
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let scale = ((s >> 24) as f32 / 256.0) * 0.05 + 0.001;
            bytes[off..off + 2].copy_from_slice(&f16::from_f32(scale).to_le_bytes());
            for j in 2..BLOCK_BYTES {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                bytes[off + j] = (s >> 24) as u8;
            }
        }
        bytes
    }

    #[test]
    fn matvec_iq2_xxs_matches_dequant_then_matvec() {
        let m = 3;
        let k = 512;
        let w = random_iq2_xxs_table(m, k, 0x12_0FFE);
        let mut sx = 0xCAFE_BABEu32;
        let x: Vec<f32> = (0..k).map(|_| lcg(&mut sx)).collect();
        let mut out_fused = vec![0f32; m];
        matvec_iq2_xxs_w_f32_a(&w, &x, &mut out_fused, m, k);
        let mut w_f32 = vec![0f32; m * k];
        rustllama_gguf::dequant::dequant_iq2_xxs(&w, &mut w_f32);
        let mut out_ref = vec![0f32; m];
        matvec_f32_scalar(&w_f32, &x, &mut out_ref, m, k);
        for (i, (a, b)) in out_fused.iter().zip(out_ref.iter()).enumerate() {
            let abs = (a - b).abs();
            let rel = abs / b.abs().max(1e-6);
            assert!(
                abs < 1e-3 || rel < 1e-5,
                "row {i}: fused={a} ref={b} abs={abs} rel={rel}"
            );
        }
    }

    #[test]
    fn matvec_iq2_xs_matches_dequant_then_matvec() {
        let m = 3;
        let k = 512;
        let w = random_iq2_xs_table(m, k, 0x12_50FFE);
        let mut sx = 0xDEAD_F00Du32;
        let x: Vec<f32> = (0..k).map(|_| lcg(&mut sx)).collect();
        let mut out_fused = vec![0f32; m];
        matvec_iq2_xs_w_f32_a(&w, &x, &mut out_fused, m, k);
        let mut w_f32 = vec![0f32; m * k];
        rustllama_gguf::dequant::dequant_iq2_xs(&w, &mut w_f32);
        let mut out_ref = vec![0f32; m];
        matvec_f32_scalar(&w_f32, &x, &mut out_ref, m, k);
        for (i, (a, b)) in out_fused.iter().zip(out_ref.iter()).enumerate() {
            let abs = (a - b).abs();
            let rel = abs / b.abs().max(1e-6);
            assert!(
                abs < 1e-3 || rel < 1e-5,
                "row {i}: fused={a} ref={b} abs={abs} rel={rel}"
            );
        }
    }

    /// Build random IQ3_XXS bytes (98 per 256-weight super-block).
    /// 8-bit grid indices land in the full 0..=255 range covered by
    /// [`rustllama_gguf::dequant::IQ3XXS_GRID`]; the 7-bit sign indices
    /// + 4-bit scale come from random u32 words in the scales+signs
    /// region.
    fn random_iq3_xxs_table(n_rows: usize, d: usize, seed: u32) -> Vec<u8> {
        const BLOCK_BYTES: usize = 98;
        const QK_K: usize = 256;
        let blocks = n_rows * (d / QK_K);
        let mut bytes = vec![0u8; blocks * BLOCK_BYTES];
        let mut s = seed;
        for b in 0..blocks {
            let off = b * BLOCK_BYTES;
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let scale = ((s >> 24) as f32 / 256.0) * 0.05 + 0.001;
            bytes[off..off + 2].copy_from_slice(&f16::from_f32(scale).to_le_bytes());
            for j in 2..BLOCK_BYTES {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                bytes[off + j] = (s >> 24) as u8;
            }
        }
        bytes
    }

    #[test]
    fn matvec_iq3_xxs_matches_dequant_then_matvec() {
        // Parity: fused matvec must equal the dequant-then-f32-matmul
        // reference. Covers the whole IQ3_XXS codepath: grid lookup,
        // sign index, scale unpack, accumulate.
        let m = 3;
        let k = 512;
        let w = random_iq3_xxs_table(m, k, 0x33_DCAFE);
        let mut sx = 0x33CC_DEEDu32;
        let x: Vec<f32> = (0..k).map(|_| lcg(&mut sx)).collect();
        let mut out_fused = vec![0f32; m];
        matvec_iq3_xxs_w_f32_a(&w, &x, &mut out_fused, m, k);
        let mut w_f32 = vec![0f32; m * k];
        rustllama_gguf::dequant::dequant_iq3_xxs(&w, &mut w_f32);
        let mut out_ref = vec![0f32; m];
        matvec_f32_scalar(&w_f32, &x, &mut out_ref, m, k);
        for (i, (a, b)) in out_fused.iter().zip(out_ref.iter()).enumerate() {
            let abs = (a - b).abs();
            let rel = abs / b.abs().max(1e-6);
            assert!(
                abs < 1e-3 || rel < 1e-5,
                "row {i}: fused={a} ref={b} abs={abs} rel={rel}"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn matvec_iq3_xxs_simd_paths_match_scalar() {
        let m = 4;
        let k = 512;
        let w = random_iq3_xxs_table(m, k, 0x33_5181u32);
        let mut sx = 0x33_F00Du32;
        let x: Vec<f32> = (0..k).map(|_| lcg(&mut sx)).collect();
        let mut out_scalar = vec![0f32; m];
        matvec_iq3_xxs_w_f32_a_scalar(&w, &x, &mut out_scalar, m, k);

        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let mut out_avx2 = vec![0f32; m];
            unsafe { matvec_iq3_xxs_w_f32_a_avx2(&w, &x, &mut out_avx2, m, k) };
            for (i, (a, b)) in out_scalar.iter().zip(out_avx2.iter()).enumerate() {
                let abs = (a - b).abs();
                let rel = abs / a.abs().max(1e-6);
                assert!(
                    abs < 5e-3 || rel < 1e-4,
                    "row {i}: scalar={a} avx2={b} abs={abs} rel={rel}"
                );
            }
        }
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx2") {
            let mut out_avx512 = vec![0f32; m];
            unsafe { matvec_iq3_xxs_w_f32_a_avx512(&w, &x, &mut out_avx512, m, k) };
            for (i, (a, b)) in out_scalar.iter().zip(out_avx512.iter()).enumerate() {
                let abs = (a - b).abs();
                let rel = abs / a.abs().max(1e-6);
                assert!(
                    abs < 5e-3 || rel < 1e-4,
                    "row {i}: scalar={a} avx512={b} abs={abs} rel={rel}"
                );
            }
        }
    }

    #[test]
    fn embed_lookup_iq3_xxs_matches_dequant() {
        // Parity: embed_lookup_iq3_xxs(ids) must equal slicing the
        // ids' rows out of the table + dequant each. Exercises the
        // row-by-id stride math + correct dequant invocation.
        let n_rows = 4;
        let d = 256;
        let table = random_iq3_xxs_table(n_rows, d, 0x33_BEEF1);
        let ids = [2i32, 0, 3, 1];
        let mut out = vec![0f32; ids.len() * d];
        embed_lookup_iq3_xxs(&table, &ids, &mut out, d);
        const BLOCK_BYTES: usize = 98;
        const QK_K: usize = 256;
        let blocks_per_row = d / QK_K;
        for (slot, &id) in ids.iter().enumerate() {
            let row_off = id as usize * blocks_per_row * BLOCK_BYTES;
            let row = &table[row_off..row_off + blocks_per_row * BLOCK_BYTES];
            let mut ref_row = vec![0f32; d];
            rustllama_gguf::dequant::dequant_iq3_xxs(row, &mut ref_row);
            let got = &out[slot * d..(slot + 1) * d];
            for (j, (a, b)) in got.iter().zip(ref_row.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-6,
                    "id {id} slot {slot} elem {j}: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn embed_lookup_iq2_xxs_matches_dequant() {
        let n_rows = 4;
        let d = 256;
        let table = random_iq2_xxs_table(n_rows, d, 0xE120_0FFE);
        let ids = [2i32, 0, 3, 1];
        let mut out = vec![0f32; ids.len() * d];
        embed_lookup_iq2_xxs(&table, &ids, &mut out, d);
        const BLOCK_BYTES: usize = 66;
        const QK_K: usize = 256;
        let blocks_per_row = d / QK_K;
        for (slot, &id) in ids.iter().enumerate() {
            let row_off = id as usize * blocks_per_row * BLOCK_BYTES;
            let row = &table[row_off..row_off + blocks_per_row * BLOCK_BYTES];
            let mut ref_row = vec![0f32; d];
            rustllama_gguf::dequant::dequant_iq2_xxs(row, &mut ref_row);
            let got = &out[slot * d..(slot + 1) * d];
            for (j, (a, b)) in got.iter().zip(ref_row.iter()).enumerate() {
                assert!((a - b).abs() < 1e-6, "id {id} slot {slot} elem {j}: {a} vs {b}");
            }
        }
    }

    fn random_bf16_bytes(m: usize, k: usize, seed: u32) -> Vec<u8> {
        // Sample f32 values in [-1, 1) and truncate the bottom 16 bits
        // to land on a BF16 grid value. Truncation (rather than
        // round-to-nearest-even) matches the synth-builder convention.
        let mut bytes = vec![0u8; m * k * 2];
        let mut s = seed;
        for p in 0..m * k {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let u = (s as i32 as f32) / (i32::MAX as f32);
            let bf_bits = (u.to_bits() >> 16) as u16;
            bytes[p * 2..p * 2 + 2].copy_from_slice(&bf_bits.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn matvec_bf16_matches_dequant_then_matvec() {
        let m = 4;
        let k = 70; // odd-tail size so the scalar branch fires
        let w = random_bf16_bytes(m, k, 0xBF16_C0FE);
        let mut sx = 0xBEEF_BABEu32;
        let x: Vec<f32> = (0..k).map(|_| lcg(&mut sx)).collect();
        let mut out_fused = vec![0f32; m];
        matvec_bf16_w_f32_a(&w, &x, &mut out_fused, m, k);

        // Reference: dequant whole tensor to f32, then matvec_f32.
        let mut w_f32 = vec![0f32; m * k];
        rustllama_gguf::dequant::dequant_bf16(&w, &mut w_f32);
        let mut out_ref = vec![0f32; m];
        matvec_f32_scalar(&w_f32, &x, &mut out_ref, m, k);
        for (i, (a, b)) in out_fused.iter().zip(out_ref.iter()).enumerate() {
            let abs = (a - b).abs();
            let rel = abs / b.abs().max(1e-6);
            assert!(
                abs < 1e-3 || rel < 1e-5,
                "row {i}: fused={a} ref={b} abs={abs} rel={rel}"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn matvec_bf16_simd_paths_match_scalar() {
        // Use k=128 so both the 32-lane and 16-lane main paths fire in
        // AVX-512, and k=70 separately for tail coverage.
        for &k in &[128usize, 70] {
            let m = 3;
            let w = random_bf16_bytes(m, k, 0xBF16_F00D);
            let mut sx = 0xBEEF_DEEDu32;
            let x: Vec<f32> = (0..k).map(|_| lcg(&mut sx)).collect();

            let mut out_scalar = vec![0f32; m];
            matvec_bf16_w_f32_a_scalar(&w, &x, &mut out_scalar, m, k);

            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                let mut out_avx2 = vec![0f32; m];
                unsafe { matvec_bf16_w_f32_a_avx2(&w, &x, &mut out_avx2, m, k) };
                for (i, (a, b)) in out_scalar.iter().zip(out_avx2.iter()).enumerate() {
                    let abs = (a - b).abs();
                    let rel = abs / a.abs().max(1e-6);
                    assert!(
                        abs < 1e-3 || rel < 1e-5,
                        "k={k} row {i}: scalar={a} avx2={b}"
                    );
                }
            }
            if is_x86_feature_detected!("avx512f") {
                let mut out_avx512 = vec![0f32; m];
                unsafe { matvec_bf16_w_f32_a_avx512(&w, &x, &mut out_avx512, m, k) };
                for (i, (a, b)) in out_scalar.iter().zip(out_avx512.iter()).enumerate() {
                    let abs = (a - b).abs();
                    let rel = abs / a.abs().max(1e-6);
                    assert!(
                        abs < 1e-3 || rel < 1e-5,
                        "k={k} row {i}: scalar={a} avx512={b}"
                    );
                }
            }
        }
    }

    #[test]
    fn embed_lookup_bf16_matches_dequant() {
        let n_rows = 4;
        let d = 64;
        let table = random_bf16_bytes(n_rows, d, 0xE125_BF16);
        let ids = [2i32, 0, 3, 1];
        let mut out = vec![0f32; ids.len() * d];
        embed_lookup_bf16(&table, &ids, &mut out, d);
        let row_bytes = d * 2;
        for (slot, &id) in ids.iter().enumerate() {
            let row_off = id as usize * row_bytes;
            let row = &table[row_off..row_off + row_bytes];
            let mut ref_row = vec![0f32; d];
            rustllama_gguf::dequant::dequant_bf16(row, &mut ref_row);
            let got = &out[slot * d..(slot + 1) * d];
            for (j, (a, b)) in got.iter().zip(ref_row.iter()).enumerate() {
                assert!((a - b).abs() < 1e-6, "id {id} slot {slot} elem {j}: {a} vs {b}");
            }
        }
    }

    /// Build random IQ1_S bytes (50 per 256-weight super-block). All
    /// bytes are PRNG-filled; the f16 super-block `d` is sampled to be
    /// moderate so floating arithmetic stays in a reasonable band.
    fn random_iq1_s_table(n_rows: usize, d: usize, seed: u32) -> Vec<u8> {
        const BLOCK_BYTES: usize = 50;
        const QK_K: usize = 256;
        let blocks = n_rows * (d / QK_K);
        let mut bytes = vec![0u8; blocks * BLOCK_BYTES];
        let mut s = seed;
        for b in 0..blocks {
            let off = b * BLOCK_BYTES;
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let d_val = ((s >> 24) as f32 / 256.0) * 0.05 + 0.001;
            bytes[off..off + 2].copy_from_slice(&f16::from_f32(d_val).to_le_bytes());
            for j in 2..BLOCK_BYTES {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                bytes[off + j] = (s >> 24) as u8;
            }
        }
        bytes
    }

    #[test]
    fn matvec_iq1_s_matches_dequant_then_matvec() {
        let m = 3;
        let k = 512;
        let w = random_iq1_s_table(m, k, 0x1A_C0FFE);
        let mut sx = 0x1A_BABEu32;
        let x: Vec<f32> = (0..k).map(|_| lcg(&mut sx)).collect();
        let mut out_fused = vec![0f32; m];
        matvec_iq1_s_w_f32_a(&w, &x, &mut out_fused, m, k);
        let mut w_f32 = vec![0f32; m * k];
        rustllama_gguf::dequant::dequant_iq1_s(&w, &mut w_f32);
        let mut out_ref = vec![0f32; m];
        matvec_f32_scalar(&w_f32, &x, &mut out_ref, m, k);
        for (i, (a, b)) in out_fused.iter().zip(out_ref.iter()).enumerate() {
            let abs = (a - b).abs();
            let rel = abs / b.abs().max(1e-6);
            assert!(
                abs < 1e-3 || rel < 1e-5,
                "row {i}: fused={a} ref={b} abs={abs} rel={rel}"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn matvec_iq1_s_avx512_matches_scalar() {
        if !is_x86_feature_detected!("avx512f") {
            eprintln!("skipped: host lacks AVX-512F");
            return;
        }
        let m = 4;
        let k = 512;
        let w = random_iq1_s_table(m, k, 0x1A_F00Du32);
        let mut sx = 0x1A_BEEFu32;
        let x: Vec<f32> = (0..k).map(|_| lcg(&mut sx)).collect();
        let mut out_scalar = vec![0f32; m];
        matvec_iq1_s_w_f32_a_scalar(&w, &x, &mut out_scalar, m, k);
        let mut out_avx512 = vec![0f32; m];
        unsafe { matvec_iq1_s_w_f32_a_avx512(&w, &x, &mut out_avx512, m, k) };
        for (i, (a, b)) in out_scalar.iter().zip(out_avx512.iter()).enumerate() {
            let abs = (a - b).abs();
            let rel = abs / a.abs().max(1e-6);
            // Loosen the tolerance slightly relative to other quants:
            // IQ1 layers ~256 multiplications per block × 2 (scale +
            // delta) into the FMA chain, so accumulated rounding sits
            // a few ULPs higher than for Q8_0-style kernels.
            assert!(
                abs < 5e-3 || rel < 1e-4,
                "row {i}: scalar={a} avx512={b} abs={abs} rel={rel}"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn matvec_iq1_s_avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("skipped: host lacks AVX2/FMA");
            return;
        }
        let m = 4;
        let k = 512;
        let w = random_iq1_s_table(m, k, 0x1A_A2A2u32);
        let mut sx = 0x1A_AAAAu32;
        let x: Vec<f32> = (0..k).map(|_| lcg(&mut sx)).collect();
        let mut out_scalar = vec![0f32; m];
        matvec_iq1_s_w_f32_a_scalar(&w, &x, &mut out_scalar, m, k);
        let mut out_avx2 = vec![0f32; m];
        unsafe { matvec_iq1_s_w_f32_a_avx2(&w, &x, &mut out_avx2, m, k) };
        for (i, (a, b)) in out_scalar.iter().zip(out_avx2.iter()).enumerate() {
            let abs = (a - b).abs();
            let rel = abs / a.abs().max(1e-6);
            assert!(
                abs < 5e-3 || rel < 1e-4,
                "row {i}: scalar={a} avx2={b} abs={abs} rel={rel}"
            );
        }
    }

    #[test]
    fn embed_lookup_iq1_s_matches_dequant() {
        let n_rows = 4;
        let d = 256;
        let table = random_iq1_s_table(n_rows, d, 0xE120_1AEE);
        let ids = [2i32, 0, 3, 1];
        let mut out = vec![0f32; ids.len() * d];
        embed_lookup_iq1_s(&table, &ids, &mut out, d);
        const BLOCK_BYTES: usize = 50;
        const QK_K: usize = 256;
        let blocks_per_row = d / QK_K;
        for (slot, &id) in ids.iter().enumerate() {
            let row_off = id as usize * blocks_per_row * BLOCK_BYTES;
            let row = &table[row_off..row_off + blocks_per_row * BLOCK_BYTES];
            let mut ref_row = vec![0f32; d];
            rustllama_gguf::dequant::dequant_iq1_s(row, &mut ref_row);
            let got = &out[slot * d..(slot + 1) * d];
            for (j, (a, b)) in got.iter().zip(ref_row.iter()).enumerate() {
                assert!((a - b).abs() < 1e-6, "id {id} slot {slot} elem {j}: {a} vs {b}");
            }
        }
    }

    /// Build random IQ1_M bytes (56 per 256-weight super-block).
    fn random_iq1_m_table(n_rows: usize, d: usize, seed: u32) -> Vec<u8> {
        const BLOCK_BYTES: usize = 56;
        const QK_K: usize = 256;
        let blocks = n_rows * (d / QK_K);
        let mut bytes = vec![0u8; blocks * BLOCK_BYTES];
        let mut s = seed;
        for b in 0..blocks {
            let off = b * BLOCK_BYTES;
            for j in 0..BLOCK_BYTES {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                bytes[off + j] = (s >> 24) as u8;
            }
        }
        bytes
    }

    #[test]
    fn matvec_iq1_m_matches_dequant_then_matvec() {
        let m = 3;
        let k = 512;
        let w = random_iq1_m_table(m, k, 0x1C_C0FFE);
        let mut sx = 0x1C_BABEu32;
        let x: Vec<f32> = (0..k).map(|_| lcg(&mut sx)).collect();
        let mut out_fused = vec![0f32; m];
        matvec_iq1_m_w_f32_a(&w, &x, &mut out_fused, m, k);
        let mut w_f32 = vec![0f32; m * k];
        rustllama_gguf::dequant::dequant_iq1_m(&w, &mut w_f32);
        let mut out_ref = vec![0f32; m];
        matvec_f32_scalar(&w_f32, &x, &mut out_ref, m, k);
        for (i, (a, b)) in out_fused.iter().zip(out_ref.iter()).enumerate() {
            let abs = (a - b).abs();
            let rel = abs / b.abs().max(1e-6);
            assert!(
                abs < 1e-3 || rel < 1e-5,
                "row {i}: fused={a} ref={b} abs={abs} rel={rel}"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn matvec_iq1_m_avx512_matches_scalar() {
        if !is_x86_feature_detected!("avx512f") {
            eprintln!("skipped: host lacks AVX-512F");
            return;
        }
        let m = 4;
        let k = 512;
        let w = random_iq1_m_table(m, k, 0x1C_F00Du32);
        let mut sx = 0x1C_BEEFu32;
        let x: Vec<f32> = (0..k).map(|_| lcg(&mut sx)).collect();
        let mut out_scalar = vec![0f32; m];
        matvec_iq1_m_w_f32_a_scalar(&w, &x, &mut out_scalar, m, k);
        let mut out_avx512 = vec![0f32; m];
        unsafe { matvec_iq1_m_w_f32_a_avx512(&w, &x, &mut out_avx512, m, k) };
        for (i, (a, b)) in out_scalar.iter().zip(out_avx512.iter()).enumerate() {
            let abs = (a - b).abs();
            let rel = abs / a.abs().max(1e-6);
            assert!(
                abs < 5e-3 || rel < 1e-4,
                "row {i}: scalar={a} avx512={b} abs={abs} rel={rel}"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn matvec_iq1_m_avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("skipped: host lacks AVX2/FMA");
            return;
        }
        let m = 4;
        let k = 512;
        // Reuse the AVX-512 test's seeds — they're known-good (the
        // packed scale-word top nibbles produce a finite `d`, so
        // neither path returns NaN).
        let w = random_iq1_m_table(m, k, 0x1C_F00Du32);
        let mut sx = 0x1C_BEEFu32;
        let x: Vec<f32> = (0..k).map(|_| lcg(&mut sx)).collect();
        let mut out_scalar = vec![0f32; m];
        matvec_iq1_m_w_f32_a_scalar(&w, &x, &mut out_scalar, m, k);
        let mut out_avx2 = vec![0f32; m];
        unsafe { matvec_iq1_m_w_f32_a_avx2(&w, &x, &mut out_avx2, m, k) };
        for (i, (a, b)) in out_scalar.iter().zip(out_avx2.iter()).enumerate() {
            let abs = (a - b).abs();
            let rel = abs / a.abs().max(1e-6);
            assert!(
                abs < 5e-3 || rel < 1e-4,
                "row {i}: scalar={a} avx2={b} abs={abs} rel={rel}"
            );
        }
    }

    #[test]
    fn embed_lookup_iq1_m_matches_dequant() {
        let n_rows = 4;
        let d = 256;
        let table = random_iq1_m_table(n_rows, d, 0xE120_1CEE);
        let ids = [2i32, 0, 3, 1];
        let mut out = vec![0f32; ids.len() * d];
        embed_lookup_iq1_m(&table, &ids, &mut out, d);
        const BLOCK_BYTES: usize = 56;
        const QK_K: usize = 256;
        let blocks_per_row = d / QK_K;
        for (slot, &id) in ids.iter().enumerate() {
            let row_off = id as usize * blocks_per_row * BLOCK_BYTES;
            let row = &table[row_off..row_off + blocks_per_row * BLOCK_BYTES];
            let mut ref_row = vec![0f32; d];
            rustllama_gguf::dequant::dequant_iq1_m(row, &mut ref_row);
            let got = &out[slot * d..(slot + 1) * d];
            for (j, (a, b)) in got.iter().zip(ref_row.iter()).enumerate() {
                assert!((a - b).abs() < 1e-6, "id {id} slot {slot} elem {j}: {a} vs {b}");
            }
        }
    }

    /// Build random Q8_K bytes (292 per 256-weight super-block).
    /// The f32 super-block scale `d` is sampled to be moderate so the
    /// float arithmetic stays in a reasonable magnitude band; the 256
    /// i8 quants are PRNG-filled; the trailing 32 bytes (bsums) are
    /// left zero — the matvec ignores them.
    fn random_q8_k_table(n_rows: usize, d: usize, seed: u32) -> Vec<u8> {
        const BLOCK_BYTES: usize = 292;
        const QK_K: usize = 256;
        let blocks = n_rows * (d / QK_K);
        let mut bytes = vec![0u8; blocks * BLOCK_BYTES];
        let mut s = seed;
        for b in 0..blocks {
            let off = b * BLOCK_BYTES;
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let d_val = ((s >> 24) as f32 / 256.0) * 0.05 + 0.001;
            bytes[off..off + 4].copy_from_slice(&d_val.to_le_bytes());
            for j in 0..256 {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                bytes[off + 4 + j] = (s >> 24) as u8;
            }
            // bsums left zero — irrelevant to Q8_K × F32 matvec.
        }
        bytes
    }

    #[test]
    fn matvec_q8_k_matches_dequant_then_matvec() {
        let m = 3;
        let k = 512;
        let w = random_q8_k_table(m, k, 0x8C_C0FFE);
        let mut sx = 0x8C_BABEu32;
        let x: Vec<f32> = (0..k).map(|_| lcg(&mut sx)).collect();
        let mut out_fused = vec![0f32; m];
        matvec_q8_k_w_f32_a(&w, &x, &mut out_fused, m, k);
        let mut w_f32 = vec![0f32; m * k];
        rustllama_gguf::dequant::dequant_q8_k(&w, &mut w_f32);
        let mut out_ref = vec![0f32; m];
        matvec_f32_scalar(&w_f32, &x, &mut out_ref, m, k);
        for (i, (a, b)) in out_fused.iter().zip(out_ref.iter()).enumerate() {
            let abs = (a - b).abs();
            let rel = abs / b.abs().max(1e-6);
            assert!(
                abs < 1e-3 || rel < 1e-5,
                "row {i}: fused={a} ref={b} abs={abs} rel={rel}"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn matvec_q8_k_simd_paths_match_scalar() {
        let m = 4;
        let k = 512;
        let w = random_q8_k_table(m, k, 0x8C_F00Du32);
        let mut sx = 0x8C_BEEFu32;
        let x: Vec<f32> = (0..k).map(|_| lcg(&mut sx)).collect();
        let mut out_scalar = vec![0f32; m];
        matvec_q8_k_w_f32_a_scalar(&w, &x, &mut out_scalar, m, k);

        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let mut out_avx2 = vec![0f32; m];
            unsafe { matvec_q8_k_w_f32_a_avx2(&w, &x, &mut out_avx2, m, k) };
            for (i, (a, b)) in out_scalar.iter().zip(out_avx2.iter()).enumerate() {
                let abs = (a - b).abs();
                let rel = abs / a.abs().max(1e-6);
                assert!(
                    abs < 1e-3 || rel < 1e-5,
                    "row {i}: scalar={a} avx2={b} abs={abs} rel={rel}"
                );
            }
        }
        if is_x86_feature_detected!("avx512f") {
            let mut out_avx512 = vec![0f32; m];
            unsafe { matvec_q8_k_w_f32_a_avx512(&w, &x, &mut out_avx512, m, k) };
            for (i, (a, b)) in out_scalar.iter().zip(out_avx512.iter()).enumerate() {
                let abs = (a - b).abs();
                let rel = abs / a.abs().max(1e-6);
                assert!(
                    abs < 1e-3 || rel < 1e-5,
                    "row {i}: scalar={a} avx512={b} abs={abs} rel={rel}"
                );
            }
        }
    }

    #[test]
    fn embed_lookup_q8_k_matches_dequant() {
        let n_rows = 4;
        let d = 256;
        let table = random_q8_k_table(n_rows, d, 0xE120_8CEE);
        let ids = [2i32, 0, 3, 1];
        let mut out = vec![0f32; ids.len() * d];
        embed_lookup_q8_k(&table, &ids, &mut out, d);
        const BLOCK_BYTES: usize = 292;
        const QK_K: usize = 256;
        let blocks_per_row = d / QK_K;
        for (slot, &id) in ids.iter().enumerate() {
            let row_off = id as usize * blocks_per_row * BLOCK_BYTES;
            let row = &table[row_off..row_off + blocks_per_row * BLOCK_BYTES];
            let mut ref_row = vec![0f32; d];
            rustllama_gguf::dequant::dequant_q8_k(row, &mut ref_row);
            let got = &out[slot * d..(slot + 1) * d];
            for (j, (a, b)) in got.iter().zip(ref_row.iter()).enumerate() {
                assert!((a - b).abs() < 1e-6, "id {id} slot {slot} elem {j}: {a} vs {b}");
            }
        }
    }

    /// Build random Q2_K bytes (84 per 256-weight super-block). The
    /// scales array (16 bytes) and qs (64 bytes) are PRNG-filled; the
    /// trailing 4 bytes are `d`/`dmin` f16 values, sampled to be
    /// moderate so the float magnitudes stay in a reasonable band.
    fn random_q2_k_table(n_rows: usize, d: usize, seed: u32) -> Vec<u8> {
        const BLOCK_BYTES: usize = 84;
        const QK_K: usize = 256;
        let blocks = n_rows * (d / QK_K);
        let mut bytes = vec![0u8; blocks * BLOCK_BYTES];
        let mut s = seed;
        for b in 0..blocks {
            let off = b * BLOCK_BYTES;
            // scales[0..16] + qs[16..80] — arbitrary bytes.
            for j in 0..(BLOCK_BYTES - 4) {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                bytes[off + j] = (s >> 24) as u8;
            }
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let d_val = ((s >> 24) as f32 / 256.0) * 0.05 + 0.001;
            bytes[off + 80..off + 82]
                .copy_from_slice(&f16::from_f32(d_val).to_le_bytes());
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let dmin_val = ((s >> 24) as f32 / 256.0) * 0.05;
            bytes[off + 82..off + 84]
                .copy_from_slice(&f16::from_f32(dmin_val).to_le_bytes());
        }
        bytes
    }

    #[test]
    fn matvec_q2_k_matches_dequant_then_matvec() {
        let m = 3;
        let k = 512;
        let w = random_q2_k_table(m, k, 0x2C_C0FFEu32);
        let mut sx = 0xC2_BABEu32;
        let x: Vec<f32> = (0..k).map(|_| lcg(&mut sx)).collect();
        let mut out_fused = vec![0f32; m];
        matvec_q2_k_w_f32_a(&w, &x, &mut out_fused, m, k);
        let mut w_f32 = vec![0f32; m * k];
        rustllama_gguf::dequant::dequant_q2_k(&w, &mut w_f32);
        let mut out_ref = vec![0f32; m];
        matvec_f32_scalar(&w_f32, &x, &mut out_ref, m, k);
        for (i, (a, b)) in out_fused.iter().zip(out_ref.iter()).enumerate() {
            let abs = (a - b).abs();
            let rel = abs / b.abs().max(1e-6);
            assert!(
                abs < 1e-3 || rel < 1e-5,
                "row {i}: fused={a} ref={b} abs={abs} rel={rel}"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn matvec_q2_k_simd_paths_match_scalar() {
        let m = 4;
        let k = 512;
        let w = random_q2_k_table(m, k, 0x2C_F00Du32);
        let mut sx = 0xC2_BEEFu32;
        let x: Vec<f32> = (0..k).map(|_| lcg(&mut sx)).collect();
        let mut out_scalar = vec![0f32; m];
        matvec_q2_k_w_f32_a_scalar(&w, &x, &mut out_scalar, m, k);

        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let mut out_avx2 = vec![0f32; m];
            unsafe { matvec_q2_k_w_f32_a_avx2(&w, &x, &mut out_avx2, m, k) };
            for (i, (a, b)) in out_scalar.iter().zip(out_avx2.iter()).enumerate() {
                let abs = (a - b).abs();
                let rel = abs / a.abs().max(1e-6);
                assert!(
                    abs < 1e-3 || rel < 1e-5,
                    "row {i}: scalar={a} avx2={b} abs={abs} rel={rel}"
                );
            }
        }
        if is_x86_feature_detected!("avx512f") {
            let mut out_avx512 = vec![0f32; m];
            unsafe { matvec_q2_k_w_f32_a_avx512(&w, &x, &mut out_avx512, m, k) };
            for (i, (a, b)) in out_scalar.iter().zip(out_avx512.iter()).enumerate() {
                let abs = (a - b).abs();
                let rel = abs / a.abs().max(1e-6);
                assert!(
                    abs < 1e-3 || rel < 1e-5,
                    "row {i}: scalar={a} avx512={b} abs={abs} rel={rel}"
                );
            }
        }
    }

    #[test]
    fn embed_lookup_q2_k_matches_dequant() {
        let n_rows = 4;
        let d = 256;
        let table = random_q2_k_table(n_rows, d, 0xE120_C2EE);
        let ids = [2i32, 0, 3, 1];
        let mut out = vec![0f32; ids.len() * d];
        embed_lookup_q2_k(&table, &ids, &mut out, d);
        const BLOCK_BYTES: usize = 84;
        const QK_K: usize = 256;
        let blocks_per_row = d / QK_K;
        for (slot, &id) in ids.iter().enumerate() {
            let row_off = id as usize * blocks_per_row * BLOCK_BYTES;
            let row = &table[row_off..row_off + blocks_per_row * BLOCK_BYTES];
            let mut ref_row = vec![0f32; d];
            rustllama_gguf::dequant::dequant_q2_k(row, &mut ref_row);
            let got = &out[slot * d..(slot + 1) * d];
            for (j, (a, b)) in got.iter().zip(ref_row.iter()).enumerate() {
                assert!((a - b).abs() < 1e-6, "id {id} slot {slot} elem {j}: {a} vs {b}");
            }
        }
    }

    /// Build random Q3_K bytes (110 per 256-weight super-block). All
    /// byte positions are PRNG-filled; the 12 packed-scale bytes have
    /// no structural constraints (any 6-bit value yields a valid
    /// signed sub-scale in `[-32, 31]`), and `qs`/`hmask` are likewise
    /// arbitrary. The f16 super-block `d` is sampled to be moderate so
    /// the float arithmetic stays in a reasonable magnitude band.
    fn random_q3_k_table(n_rows: usize, d: usize, seed: u32) -> Vec<u8> {
        const BLOCK_BYTES: usize = 110;
        const QK_K: usize = 256;
        let blocks = n_rows * (d / QK_K);
        let mut bytes = vec![0u8; blocks * BLOCK_BYTES];
        let mut s = seed;
        for b in 0..blocks {
            let off = b * BLOCK_BYTES;
            for j in 0..(BLOCK_BYTES - 2) {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                bytes[off + j] = (s >> 24) as u8;
            }
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let scale = ((s >> 24) as f32 / 256.0) * 0.05 + 0.001;
            bytes[off + BLOCK_BYTES - 2..off + BLOCK_BYTES]
                .copy_from_slice(&f16::from_f32(scale).to_le_bytes());
        }
        bytes
    }

    #[test]
    fn matvec_q3_k_matches_dequant_then_matvec() {
        let m = 3;
        let k = 512;
        let w = random_q3_k_table(m, k, 0xC3_C0FFE);
        let mut sx = 0xDEAD_BABEu32;
        let x: Vec<f32> = (0..k).map(|_| lcg(&mut sx)).collect();
        let mut out_fused = vec![0f32; m];
        matvec_q3_k_w_f32_a(&w, &x, &mut out_fused, m, k);
        let mut w_f32 = vec![0f32; m * k];
        rustllama_gguf::dequant::dequant_q3_k(&w, &mut w_f32);
        let mut out_ref = vec![0f32; m];
        matvec_f32_scalar(&w_f32, &x, &mut out_ref, m, k);
        for (i, (a, b)) in out_fused.iter().zip(out_ref.iter()).enumerate() {
            let abs = (a - b).abs();
            let rel = abs / b.abs().max(1e-6);
            assert!(
                abs < 1e-3 || rel < 1e-5,
                "row {i}: fused={a} ref={b} abs={abs} rel={rel}"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn matvec_q3_k_simd_paths_match_scalar() {
        let m = 4;
        let k = 512;
        let w = random_q3_k_table(m, k, 0xC3_51_4D_u32);
        let mut sx = 0xC3_5117_u32;
        let x: Vec<f32> = (0..k).map(|_| lcg(&mut sx)).collect();

        let mut out_scalar = vec![0f32; m];
        matvec_q3_k_w_f32_a_scalar(&w, &x, &mut out_scalar, m, k);

        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let mut out_avx2 = vec![0f32; m];
            unsafe { matvec_q3_k_w_f32_a_avx2(&w, &x, &mut out_avx2, m, k) };
            for (i, (a, b)) in out_scalar.iter().zip(out_avx2.iter()).enumerate() {
                let abs = (a - b).abs();
                let rel = abs / a.abs().max(1e-6);
                assert!(
                    abs < 1e-3 || rel < 1e-5,
                    "row {i}: scalar={a} avx2={b}"
                );
            }
        }
        if is_x86_feature_detected!("avx512f") {
            let mut out_avx512 = vec![0f32; m];
            unsafe { matvec_q3_k_w_f32_a_avx512(&w, &x, &mut out_avx512, m, k) };
            for (i, (a, b)) in out_scalar.iter().zip(out_avx512.iter()).enumerate() {
                let abs = (a - b).abs();
                let rel = abs / a.abs().max(1e-6);
                assert!(
                    abs < 1e-3 || rel < 1e-5,
                    "row {i}: scalar={a} avx512={b}"
                );
            }
        }
    }

    #[test]
    fn embed_lookup_q3_k_matches_dequant() {
        let n_rows = 4;
        let d = 256;
        let table = random_q3_k_table(n_rows, d, 0xE13_C0FFE);
        let ids = [2i32, 0, 3, 1];
        let mut out = vec![0f32; ids.len() * d];
        embed_lookup_q3_k(&table, &ids, &mut out, d);
        const BLOCK_BYTES: usize = 110;
        const QK_K: usize = 256;
        let blocks_per_row = d / QK_K;
        for (slot, &id) in ids.iter().enumerate() {
            let row_off = id as usize * blocks_per_row * BLOCK_BYTES;
            let row = &table[row_off..row_off + blocks_per_row * BLOCK_BYTES];
            let mut ref_row = vec![0f32; d];
            rustllama_gguf::dequant::dequant_q3_k(row, &mut ref_row);
            let got = &out[slot * d..(slot + 1) * d];
            for (j, (a, b)) in got.iter().zip(ref_row.iter()).enumerate() {
                assert!((a - b).abs() < 1e-6, "id {id} slot {slot} elem {j}: {a} vs {b}");
            }
        }
    }

    /// Build random IQ2_S bytes (82 per 256-weight super-block).
    /// `qs` (64 bytes), `qh` (8), and `scales` (8) are all PRNG-filled;
    /// the 10-bit grid index space is fully covered by the 1024-entry
    /// grid for any (qs, qh) pair.
    fn random_iq2_s_table(n_rows: usize, d: usize, seed: u32) -> Vec<u8> {
        const BLOCK_BYTES: usize = 82;
        const QK_K: usize = 256;
        let blocks = n_rows * (d / QK_K);
        let mut bytes = vec![0u8; blocks * BLOCK_BYTES];
        let mut s = seed;
        for b in 0..blocks {
            let off = b * BLOCK_BYTES;
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let scale = ((s >> 24) as f32 / 256.0) * 0.05 + 0.001;
            bytes[off..off + 2].copy_from_slice(&f16::from_f32(scale).to_le_bytes());
            for j in 2..BLOCK_BYTES {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                bytes[off + j] = (s >> 24) as u8;
            }
        }
        bytes
    }

    #[test]
    fn matvec_iq2_s_matches_dequant_then_matvec() {
        let m = 3;
        let k = 512;
        let w = random_iq2_s_table(m, k, 0x12_5_0FFE);
        let mut sx = 0xBEEF_F00Du32;
        let x: Vec<f32> = (0..k).map(|_| lcg(&mut sx)).collect();
        let mut out_fused = vec![0f32; m];
        matvec_iq2_s_w_f32_a(&w, &x, &mut out_fused, m, k);
        let mut w_f32 = vec![0f32; m * k];
        rustllama_gguf::dequant::dequant_iq2_s(&w, &mut w_f32);
        let mut out_ref = vec![0f32; m];
        matvec_f32_scalar(&w_f32, &x, &mut out_ref, m, k);
        for (i, (a, b)) in out_fused.iter().zip(out_ref.iter()).enumerate() {
            let abs = (a - b).abs();
            let rel = abs / b.abs().max(1e-6);
            assert!(
                abs < 1e-3 || rel < 1e-5,
                "row {i}: fused={a} ref={b} abs={abs} rel={rel}"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn matvec_iq2_family_simd_paths_match_scalar() {
        let m = 4;
        let k = 512;
        let mut sx = 0x12_AC_E000u32;
        let x: Vec<f32> = (0..k).map(|_| lcg(&mut sx)).collect();
        let has_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        let has_avx512 = has_avx2 && is_x86_feature_detected!("avx512f");

        // IQ2_XXS
        {
            let w = random_iq2_xxs_table(m, k, 0x12_20_AC_E5);
            let mut out_scalar = vec![0f32; m];
            matvec_iq2_xxs_w_f32_a_scalar(&w, &x, &mut out_scalar, m, k);
            if has_avx2 {
                let mut out_avx2 = vec![0f32; m];
                unsafe { matvec_iq2_xxs_w_f32_a_avx2(&w, &x, &mut out_avx2, m, k) };
                for (i, (a, b)) in out_scalar.iter().zip(out_avx2.iter()).enumerate() {
                    let abs = (a - b).abs();
                    let rel = abs / a.abs().max(1e-6);
                    assert!(abs < 1e-3 || rel < 1e-5, "iq2_xxs avx2 row {i}: {a} vs {b}");
                }
            }
            if has_avx512 {
                let mut out_avx512 = vec![0f32; m];
                unsafe { matvec_iq2_xxs_w_f32_a_avx512(&w, &x, &mut out_avx512, m, k) };
                for (i, (a, b)) in out_scalar.iter().zip(out_avx512.iter()).enumerate() {
                    let abs = (a - b).abs();
                    let rel = abs / a.abs().max(1e-6);
                    assert!(abs < 1e-3 || rel < 1e-5, "iq2_xxs avx512 row {i}: {a} vs {b}");
                }
            }
        }
        // IQ2_XS
        {
            let w = random_iq2_xs_table(m, k, 0x12_25_AC_E5);
            let mut out_scalar = vec![0f32; m];
            matvec_iq2_xs_w_f32_a_scalar(&w, &x, &mut out_scalar, m, k);
            if has_avx2 {
                let mut out_avx2 = vec![0f32; m];
                unsafe { matvec_iq2_xs_w_f32_a_avx2(&w, &x, &mut out_avx2, m, k) };
                for (i, (a, b)) in out_scalar.iter().zip(out_avx2.iter()).enumerate() {
                    let abs = (a - b).abs();
                    let rel = abs / a.abs().max(1e-6);
                    assert!(abs < 1e-3 || rel < 1e-5, "iq2_xs avx2 row {i}: {a} vs {b}");
                }
            }
            if has_avx512 {
                let mut out_avx512 = vec![0f32; m];
                unsafe { matvec_iq2_xs_w_f32_a_avx512(&w, &x, &mut out_avx512, m, k) };
                for (i, (a, b)) in out_scalar.iter().zip(out_avx512.iter()).enumerate() {
                    let abs = (a - b).abs();
                    let rel = abs / a.abs().max(1e-6);
                    assert!(abs < 1e-3 || rel < 1e-5, "iq2_xs avx512 row {i}: {a} vs {b}");
                }
            }
        }
        // IQ2_S
        {
            let w = random_iq2_s_table(m, k, 0x12_5_AC_E5);
            let mut out_scalar = vec![0f32; m];
            matvec_iq2_s_w_f32_a_scalar(&w, &x, &mut out_scalar, m, k);
            if has_avx2 {
                let mut out_avx2 = vec![0f32; m];
                unsafe { matvec_iq2_s_w_f32_a_avx2(&w, &x, &mut out_avx2, m, k) };
                for (i, (a, b)) in out_scalar.iter().zip(out_avx2.iter()).enumerate() {
                    let abs = (a - b).abs();
                    let rel = abs / a.abs().max(1e-6);
                    assert!(abs < 1e-3 || rel < 1e-5, "iq2_s avx2 row {i}: {a} vs {b}");
                }
            }
            if has_avx512 {
                let mut out_avx512 = vec![0f32; m];
                unsafe { matvec_iq2_s_w_f32_a_avx512(&w, &x, &mut out_avx512, m, k) };
                for (i, (a, b)) in out_scalar.iter().zip(out_avx512.iter()).enumerate() {
                    let abs = (a - b).abs();
                    let rel = abs / a.abs().max(1e-6);
                    assert!(abs < 1e-3 || rel < 1e-5, "iq2_s avx512 row {i}: {a} vs {b}");
                }
            }
        }
    }

    #[test]
    fn embed_lookup_iq2_s_matches_dequant() {
        let n_rows = 4;
        let d = 256;
        let table = random_iq2_s_table(n_rows, d, 0xE12_5_0FFE);
        let ids = [0i32, 3, 1, 2];
        let mut out = vec![0f32; ids.len() * d];
        embed_lookup_iq2_s(&table, &ids, &mut out, d);
        const BLOCK_BYTES: usize = 82;
        const QK_K: usize = 256;
        let blocks_per_row = d / QK_K;
        for (slot, &id) in ids.iter().enumerate() {
            let row_off = id as usize * blocks_per_row * BLOCK_BYTES;
            let row = &table[row_off..row_off + blocks_per_row * BLOCK_BYTES];
            let mut ref_row = vec![0f32; d];
            rustllama_gguf::dequant::dequant_iq2_s(row, &mut ref_row);
            let got = &out[slot * d..(slot + 1) * d];
            for (j, (a, b)) in got.iter().zip(ref_row.iter()).enumerate() {
                assert!((a - b).abs() < 1e-6, "id {id} slot {slot} elem {j}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn embed_lookup_iq2_xs_matches_dequant() {
        let n_rows = 4;
        let d = 256;
        let table = random_iq2_xs_table(n_rows, d, 0xE125_0FFE);
        let ids = [3i32, 1, 0, 2];
        let mut out = vec![0f32; ids.len() * d];
        embed_lookup_iq2_xs(&table, &ids, &mut out, d);
        const BLOCK_BYTES: usize = 74;
        const QK_K: usize = 256;
        let blocks_per_row = d / QK_K;
        for (slot, &id) in ids.iter().enumerate() {
            let row_off = id as usize * blocks_per_row * BLOCK_BYTES;
            let row = &table[row_off..row_off + blocks_per_row * BLOCK_BYTES];
            let mut ref_row = vec![0f32; d];
            rustllama_gguf::dequant::dequant_iq2_xs(row, &mut ref_row);
            let got = &out[slot * d..(slot + 1) * d];
            for (j, (a, b)) in got.iter().zip(ref_row.iter()).enumerate() {
                assert!((a - b).abs() < 1e-6, "id {id} slot {slot} elem {j}: {a} vs {b}");
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn matvec_iq4_xs_avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2")
            || !is_x86_feature_detected!("fma")
            || !is_x86_feature_detected!("ssse3")
        {
            eprintln!("skipped: host lacks AVX2/FMA/SSSE3");
            return;
        }
        const QK_K: usize = 256;
        const BLOCK_BYTES: usize = 136;
        let m = 4;
        let k = 512;
        let blocks_per_row = k / QK_K;
        let total_blocks = m * blocks_per_row;

        let mut w_bytes = vec![0u8; total_blocks * BLOCK_BYTES];
        let mut s = 0xCAFEDEADu32;
        for b in 0..total_blocks {
            let off = b * BLOCK_BYTES;
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let d = ((s >> 24) as f32 / 256.0) * 0.05 + 0.001;
            w_bytes[off..off + 2].copy_from_slice(&f16::from_f32(d).to_le_bytes());
            for j in 0..(BLOCK_BYTES - 2) {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                w_bytes[off + 2 + j] = (s >> 24) as u8;
            }
        }
        let mut x = vec![0f32; k];
        for v in x.iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = (s as i32 as f32) / (i32::MAX as f32);
        }

        let mut out_scalar = vec![0f32; m];
        let mut out_avx2 = vec![0f32; m];
        matvec_iq4_xs_w_f32_a_scalar(&w_bytes, &x, &mut out_scalar, m, k);
        // SAFETY: feature-detected above.
        unsafe { matvec_iq4_xs_w_f32_a_avx2(&w_bytes, &x, &mut out_avx2, m, k) };

        for (i, (a, b)) in out_scalar.iter().zip(out_avx2.iter()).enumerate() {
            let abs_err = (a - b).abs();
            let rel_err = abs_err / a.abs().max(1e-6);
            assert!(
                abs_err < 1e-3 || rel_err < 1e-5,
                "row {i}: scalar={a} avx2={b} abs={abs_err} rel={rel_err}"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn matvec_iq4_xs_avx512_matches_scalar() {
        if !is_x86_feature_detected!("avx512f") {
            eprintln!("skipped: host lacks AVX-512F");
            return;
        }
        const QK_K: usize = 256;
        const BLOCK_BYTES: usize = 136;
        let m = 4;
        let k = 512; // 2 super-blocks per row — exercises the per-block loop.
        let blocks_per_row = k / QK_K;
        let total_blocks = m * blocks_per_row;

        // Construct random IQ4_XS rows. Mostly-random bytes are fine
        // because the codebook + per-sub-block scale folding masks
        // the structural meaning at this granularity.
        let mut w_bytes = vec![0u8; total_blocks * BLOCK_BYTES];
        let mut s = 0xBEEFC0DEu32;
        for b in 0..total_blocks {
            let off = b * BLOCK_BYTES;
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let d = ((s >> 24) as f32 / 256.0) * 0.05 + 0.001;
            w_bytes[off..off + 2].copy_from_slice(&f16::from_f32(d).to_le_bytes());
            for j in 0..(BLOCK_BYTES - 2) {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                w_bytes[off + 2 + j] = (s >> 24) as u8;
            }
        }
        let mut x = vec![0f32; k];
        for v in x.iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = (s as i32 as f32) / (i32::MAX as f32);
        }

        let mut out_scalar = vec![0f32; m];
        let mut out_avx512 = vec![0f32; m];
        matvec_iq4_xs_w_f32_a_scalar(&w_bytes, &x, &mut out_scalar, m, k);
        // SAFETY: feature-detected above.
        unsafe { matvec_iq4_xs_w_f32_a_avx512(&w_bytes, &x, &mut out_avx512, m, k) };

        // Floating-point FMA reorderings can perturb low bits; loosen
        // tolerance slightly relative to f32-only tests because the
        // codebook + sub-block scaling stack more multiplications.
        for (i, (a, b)) in out_scalar.iter().zip(out_avx512.iter()).enumerate() {
            let abs_err = (a - b).abs();
            let rel_err = abs_err / a.abs().max(1e-6);
            assert!(
                abs_err < 1e-3 || rel_err < 1e-5,
                "row {i}: scalar={a} avx512={b} abs={abs_err} rel={rel_err}"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn matvec_f16_avx512_matches_scalar() {
        if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("f16c") {
            eprintln!("skipped: host lacks AVX-512F or F16C");
            return;
        }
        // m=3, k=70 — exercises 32-wide outer tile, 16-wide mid tile, scalar tail.
        let m = 3usize;
        let k = 70usize;
        let mut s = 0xFEED_BEEFu32;
        let w_f32: Vec<f32> = (0..m * k).map(|_| lcg(&mut s)).collect();
        let w_f16: Vec<f16> = w_f32.iter().map(|v| f16::from_f32(*v)).collect();
        let x: Vec<f32> = (0..k).map(|_| lcg(&mut s)).collect();

        let mut out_avx2 = vec![0f32; m];
        let mut out_avx512 = vec![0f32; m];
        // SAFETY: feature-detected above.
        unsafe { matvec_f16_w_f32_a_avx2(&w_f16, &x, &mut out_avx2, m, k) };
        unsafe { matvec_f16_w_f32_a_avx512(&w_f16, &x, &mut out_avx512, m, k) };
        for (i, (a, b)) in out_avx2.iter().zip(out_avx512.iter()).enumerate() {
            let abs = (a - b).abs();
            let rel = abs / a.abs().max(1e-6);
            assert!(
                abs < 1e-3 || rel < 1e-5,
                "row {i}: avx2={a} avx512={b} abs={abs} rel={rel}"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn embed_lookup_iq4_xs_avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("ssse3") {
            eprintln!("skipped: host lacks AVX2/SSSE3");
            return;
        }
        const BLOCK_BYTES: usize = 136;
        let n_rows = 5;
        let d = 512;
        let table = random_qk_table(n_rows, d, BLOCK_BYTES, 0xA1C4);
        let ids = [0i32, 2, 4, 1];
        let mut out_scalar = vec![0f32; ids.len() * d];
        let mut out_avx2 = vec![0f32; ids.len() * d];
        embed_lookup_iq4_xs_scalar(&table, &ids, &mut out_scalar, d);
        unsafe { embed_lookup_iq4_xs_avx2(&table, &ids, &mut out_avx2, d) };
        for (i, (a, b)) in out_scalar.iter().zip(out_avx2.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "idx {i}: scalar={a} avx2={b}"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn embed_lookup_iq4_xs_avx512_matches_scalar() {
        if !is_x86_feature_detected!("avx512f") {
            eprintln!("skipped: host lacks AVX-512F");
            return;
        }
        const BLOCK_BYTES: usize = 136;
        let n_rows = 5;
        let d = 512;
        let table = random_qk_table(n_rows, d, BLOCK_BYTES, 0xA5C4);
        let ids = [3i32, 0, 4];
        let mut out_scalar = vec![0f32; ids.len() * d];
        let mut out_avx512 = vec![0f32; ids.len() * d];
        embed_lookup_iq4_xs_scalar(&table, &ids, &mut out_scalar, d);
        unsafe { embed_lookup_iq4_xs_avx512(&table, &ids, &mut out_avx512, d) };
        for (i, (a, b)) in out_scalar.iter().zip(out_avx512.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "idx {i}: scalar={a} avx512={b}"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn embed_lookup_q8_0_simd_matches_scalar() {
        const BLOCK_BYTES: usize = 34;
        const QK: usize = 32;
        let n_rows = 5;
        let d = 128;
        let blocks = n_rows * (d / QK);
        let mut table = vec![0u8; blocks * BLOCK_BYTES];
        let mut s = 0xA8C0u32;
        for b in 0..blocks {
            let off = b * BLOCK_BYTES;
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let scale = ((s >> 24) as f32 / 256.0) * 0.05 + 0.001;
            table[off..off + 2].copy_from_slice(&f16::from_f32(scale).to_le_bytes());
            for j in 0..QK {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                table[off + 2 + j] = (s >> 24) as u8;
            }
        }
        let ids = [0i32, 1, 3, 2];
        let mut out_scalar = vec![0f32; ids.len() * d];
        embed_lookup_q8_0_scalar(&table, &ids, &mut out_scalar, d);

        if is_x86_feature_detected!("avx2") {
            let mut out_avx2 = vec![0f32; ids.len() * d];
            unsafe { embed_lookup_q8_0_avx2(&table, &ids, &mut out_avx2, d) };
            for (i, (a, b)) in out_scalar.iter().zip(out_avx2.iter()).enumerate() {
                assert!((a - b).abs() < 1e-5, "avx2 idx {i}: scalar={a} avx2={b}");
            }
        }
        if is_x86_feature_detected!("avx512f") {
            let mut out_avx512 = vec![0f32; ids.len() * d];
            unsafe { embed_lookup_q8_0_avx512(&table, &ids, &mut out_avx512, d) };
            for (i, (a, b)) in out_scalar.iter().zip(out_avx512.iter()).enumerate() {
                assert!((a - b).abs() < 1e-5, "avx512 idx {i}: scalar={a} avx512={b}");
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn embed_lookup_q5_0_simd_matches_scalar() {
        const BLOCK_BYTES: usize = 22;
        const QK: usize = 32;
        let n_rows = 5;
        let d = 128;
        let blocks = n_rows * (d / QK);
        let mut table = vec![0u8; blocks * BLOCK_BYTES];
        let mut s = 0xA5C0u32;
        for b in 0..blocks {
            let off = b * BLOCK_BYTES;
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let scale = ((s >> 24) as f32 / 256.0) * 0.05 + 0.001;
            table[off..off + 2].copy_from_slice(&f16::from_f32(scale).to_le_bytes());
            for j in 0..(BLOCK_BYTES - 2) {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                table[off + 2 + j] = (s >> 24) as u8;
            }
        }
        let ids = [2i32, 0, 4, 1];
        let mut out_scalar = vec![0f32; ids.len() * d];
        embed_lookup_q5_0_scalar(&table, &ids, &mut out_scalar, d);

        if is_x86_feature_detected!("avx2") {
            let mut out_avx2 = vec![0f32; ids.len() * d];
            unsafe { embed_lookup_q5_0_avx2(&table, &ids, &mut out_avx2, d) };
            for (i, (a, b)) in out_scalar.iter().zip(out_avx2.iter()).enumerate() {
                assert!((a - b).abs() < 1e-5, "avx2 idx {i}: scalar={a} avx2={b}");
            }
        }
        if is_x86_feature_detected!("avx512f") {
            let mut out_avx512 = vec![0f32; ids.len() * d];
            unsafe { embed_lookup_q5_0_avx512(&table, &ids, &mut out_avx512, d) };
            for (i, (a, b)) in out_scalar.iter().zip(out_avx512.iter()).enumerate() {
                assert!((a - b).abs() < 1e-5, "avx512 idx {i}: scalar={a} avx512={b}");
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn embed_lookup_q5_k_simd_matches_scalar() {
        const BLOCK_BYTES: usize = 176;
        let n_rows = 4;
        let d = 512;
        let table = random_qk_table(n_rows, d, BLOCK_BYTES, 0xA50C);
        let ids = [0i32, 1, 3, 2];
        let mut out_scalar = vec![0f32; ids.len() * d];
        embed_lookup_q5_k_scalar(&table, &ids, &mut out_scalar, d);

        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let mut out_avx2 = vec![0f32; ids.len() * d];
            unsafe { embed_lookup_q5_k_avx2(&table, &ids, &mut out_avx2, d) };
            for (i, (a, b)) in out_scalar.iter().zip(out_avx2.iter()).enumerate() {
                assert!((a - b).abs() < 1e-5, "avx2 idx {i}: scalar={a} avx2={b}");
            }
        }
        if is_x86_feature_detected!("avx512f") {
            let mut out_avx512 = vec![0f32; ids.len() * d];
            unsafe { embed_lookup_q5_k_avx512(&table, &ids, &mut out_avx512, d) };
            for (i, (a, b)) in out_scalar.iter().zip(out_avx512.iter()).enumerate() {
                assert!((a - b).abs() < 1e-5, "avx512 idx {i}: scalar={a} avx512={b}");
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn embed_lookup_q4_k_avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("skipped: host lacks AVX2/FMA");
            return;
        }
        const BLOCK_BYTES: usize = 144;
        let n_rows = 5;
        let d = 512;
        let table = random_qk_table(n_rows, d, BLOCK_BYTES, 0xE5B0);
        let ids = [1i32, 2, 4, 0];
        let mut out_scalar = vec![0f32; ids.len() * d];
        let mut out_avx2 = vec![0f32; ids.len() * d];
        embed_lookup_q4_k_scalar(&table, &ids, &mut out_scalar, d);
        unsafe { embed_lookup_q4_k_avx2(&table, &ids, &mut out_avx2, d) };
        for (i, (a, b)) in out_scalar.iter().zip(out_avx2.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "idx {i}: scalar={a} avx2={b}"
            );
        }
    }

    #[test]
    fn matvec_iq4_xs_matches_dequant_then_matvec() {
        const QK_K: usize = 256;
        const BLOCK_BYTES: usize = 136;
        let m = 3;
        let k = 512; // 2 super-blocks per row
        let blocks_per_row = k / QK_K;
        let total_blocks = m * blocks_per_row;

        let mut w_bytes = vec![0u8; total_blocks * BLOCK_BYTES];
        let mut s: u32 = 0xDEADCAFE;
        for b in 0..total_blocks {
            let off = b * BLOCK_BYTES;
            // f16 d
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let d = ((s >> 24) as f32 / 256.0) * 0.05 + 0.001;
            w_bytes[off..off + 2].copy_from_slice(&f16::from_f32(d).to_le_bytes());
            // scales_h + scales_l + qs: random
            for j in 0..(BLOCK_BYTES - 2) {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                w_bytes[off + 2 + j] = (s >> 24) as u8;
            }
        }

        let mut x = vec![0f32; k];
        for v in x.iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = (s as i32 as f32) / (i32::MAX as f32);
        }

        // Fused path: matvec_iq4_xs_w_f32_a (on-the-fly dequant inside matvec)
        let mut out_fused = vec![0f32; m];
        matvec_iq4_xs_w_f32_a(&w_bytes, &x, &mut out_fused, m, k);

        // Reference: dequant the whole table, then matvec_f32_scalar.
        let mut w_f32 = vec![0f32; m * k];
        rustllama_gguf::dequant::dequant_iq4_xs(&w_bytes, &mut w_f32);
        let mut out_ref = vec![0f32; m];
        matvec_f32_scalar(&w_f32, &x, &mut out_ref, m, k);

        for (i, (a, b)) in out_fused.iter().zip(out_ref.iter()).enumerate() {
            let abs_err = (a - b).abs();
            let rel_err = abs_err / b.abs().max(1e-6);
            assert!(
                abs_err < 1e-3 || rel_err < 1e-5,
                "row {i}: fused={a} ref={b} abs={abs_err} rel={rel_err}"
            );
        }
    }

    #[test]
    fn matvec_q4_k_matches_dequant_then_matvec() {
        const QK_K: usize = 256;
        const BLOCK_BYTES: usize = 144;
        let m = 3;
        let k = 512; // 2 super-blocks per row
        let blocks_per_row = k / QK_K;
        let total_blocks = m * blocks_per_row;

        let mut w_bytes = vec![0u8; total_blocks * BLOCK_BYTES];
        let mut s: u32 = 0xBADD_C0DE;
        for b in 0..total_blocks {
            let off = b * BLOCK_BYTES;
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let d = ((s >> 24) as f32 / 256.0) * 0.05 + 0.001;
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let dmin = ((s >> 24) as f32 / 256.0) * 0.02 + 0.0005;
            w_bytes[off..off + 2].copy_from_slice(&f16::from_f32(d).to_le_bytes());
            w_bytes[off + 2..off + 4].copy_from_slice(&f16::from_f32(dmin).to_le_bytes());
            // scales[12] + qs[128]: random bytes
            for j in 0..140 {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                w_bytes[off + 4 + j] = (s >> 24) as u8;
            }
        }

        let mut x = vec![0f32; k];
        for v in x.iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = (s as i32 as f32) / (i32::MAX as f32);
        }

        // Fused path
        let mut out_fused = vec![0f32; m];
        matvec_q4_k_w_f32_a(&w_bytes, &x, &mut out_fused, m, k);

        // Reference: dequant then matvec
        let mut w_f32 = vec![0f32; m * k];
        rustllama_gguf::dequant::dequant_q4_k(&w_bytes, &mut w_f32);
        let mut out_ref = vec![0f32; m];
        matvec_f32_scalar(&w_f32, &x, &mut out_ref, m, k);

        for (i, (a, b)) in out_fused.iter().zip(out_ref.iter()).enumerate() {
            let abs_err = (a - b).abs();
            let rel_err = abs_err / b.abs().max(1e-6);
            assert!(
                abs_err < 1e-2 || rel_err < 1e-3,
                "row {i}: fused={a} ref={b} abs={abs_err} rel={rel_err}"
            );
        }
    }

    #[test]
    fn matvec_q4_0_matches_dequant_then_matvec() {
        const QK: usize = 32;
        const BLOCK_BYTES: usize = 18;
        let m = 5;
        let k = 96; // 3 blocks per row
        let blocks_per_row = k / QK;
        let total_blocks = m * blocks_per_row;

        let mut w_bytes = vec![0u8; total_blocks * BLOCK_BYTES];
        let mut s: u32 = 0x4040_BEEF;
        for b in 0..total_blocks {
            let off = b * BLOCK_BYTES;
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let d = ((s >> 24) as f32 / 256.0) * 0.05 + 0.001;
            let d_bytes = f16::from_f32(d).to_le_bytes();
            w_bytes[off] = d_bytes[0];
            w_bytes[off + 1] = d_bytes[1];
            for j in 0..16 {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                w_bytes[off + 2 + j] = (s >> 24) as u8;
            }
        }
        let mut x = vec![0f32; k];
        for v in x.iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = (s as i32 as f32) / (i32::MAX as f32);
        }
        let mut out_fused = vec![0f32; m];
        matvec_q4_0_w_f32_a(&w_bytes, &x, &mut out_fused, m, k);

        let mut w_f32 = vec![0f32; m * k];
        rustllama_gguf::dequant::dequant_q4_0(&w_bytes, &mut w_f32);
        let mut out_ref = vec![0f32; m];
        matvec_f32_scalar(&w_f32, &x, &mut out_ref, m, k);
        for (i, (a, b)) in out_fused.iter().zip(out_ref.iter()).enumerate() {
            let abs = (a - b).abs();
            let rel = abs / b.abs().max(1e-6);
            assert!(
                abs < 1e-3 || rel < 1e-5,
                "row {i}: fused={a} ref={b} abs={abs} rel={rel}"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn matvec_q4_0_simd_paths_match_scalar() {
        const QK: usize = 32;
        const BLOCK_BYTES: usize = 18;
        let m = 4;
        let k = 128;
        let blocks_per_row = k / QK;
        let total_blocks = m * blocks_per_row;

        let mut w_bytes = vec![0u8; total_blocks * BLOCK_BYTES];
        let mut s: u32 = 0x4040_F00D;
        for b in 0..total_blocks {
            let off = b * BLOCK_BYTES;
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let d = ((s >> 24) as f32 / 256.0) * 0.05 + 0.001;
            let d_bytes = f16::from_f32(d).to_le_bytes();
            w_bytes[off] = d_bytes[0];
            w_bytes[off + 1] = d_bytes[1];
            for j in 0..16 {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                w_bytes[off + 2 + j] = (s >> 24) as u8;
            }
        }
        let mut x = vec![0f32; k];
        for v in x.iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = (s as i32 as f32) / (i32::MAX as f32);
        }

        let mut out_scalar = vec![0f32; m];
        matvec_q4_0_w_f32_a_scalar(&w_bytes, &x, &mut out_scalar, m, k);

        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let mut out_avx2 = vec![0f32; m];
            unsafe { matvec_q4_0_w_f32_a_avx2(&w_bytes, &x, &mut out_avx2, m, k) };
            for (i, (a, b)) in out_scalar.iter().zip(out_avx2.iter()).enumerate() {
                let abs = (a - b).abs();
                let rel = abs / a.abs().max(1e-6);
                assert!(
                    abs < 1e-3 || rel < 1e-5,
                    "row {i}: scalar={a} avx2={b}"
                );
            }
        }
        if is_x86_feature_detected!("avx512f") {
            let mut out_avx512 = vec![0f32; m];
            unsafe { matvec_q4_0_w_f32_a_avx512(&w_bytes, &x, &mut out_avx512, m, k) };
            for (i, (a, b)) in out_scalar.iter().zip(out_avx512.iter()).enumerate() {
                let abs = (a - b).abs();
                let rel = abs / a.abs().max(1e-6);
                assert!(
                    abs < 1e-3 || rel < 1e-5,
                    "row {i}: scalar={a} avx512={b}"
                );
            }
        }
    }

    #[test]
    fn matvec_q4_1_matches_dequant_then_matvec() {
        // End-to-end equivalence check: fused on-the-fly dequant+matvec
        // must equal "dequant whole tensor to f32, then plain matvec_f32".
        const QK: usize = 32;
        const BLOCK_BYTES: usize = 20;
        let m = 5;
        let k = 96; // 3 blocks per row
        let blocks_per_row = k / QK;
        let total_blocks = m * blocks_per_row;

        let mut w_bytes = vec![0u8; total_blocks * BLOCK_BYTES];
        let mut s: u32 = 0x4141_BEEF;
        for b in 0..total_blocks {
            let off = b * BLOCK_BYTES;
            // d
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let d = ((s >> 24) as f32 / 256.0) * 0.05 + 0.001;
            w_bytes[off..off + 2].copy_from_slice(&f16::from_f32(d).to_le_bytes());
            // m (sweep negative side too — Q4_1 explicitly allows asymmetric range)
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let m_off = ((s >> 24) as f32 / 256.0 - 0.5) * 0.1;
            w_bytes[off + 2..off + 4].copy_from_slice(&f16::from_f32(m_off).to_le_bytes());
            for j in 0..16 {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                w_bytes[off + 4 + j] = (s >> 24) as u8;
            }
        }
        let mut x = vec![0f32; k];
        for v in x.iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = (s as i32 as f32) / (i32::MAX as f32);
        }
        let mut out_fused = vec![0f32; m];
        matvec_q4_1_w_f32_a(&w_bytes, &x, &mut out_fused, m, k);

        let mut w_f32 = vec![0f32; m * k];
        rustllama_gguf::dequant::dequant_q4_1(&w_bytes, &mut w_f32);
        let mut out_ref = vec![0f32; m];
        matvec_f32_scalar(&w_f32, &x, &mut out_ref, m, k);
        for (i, (a, b)) in out_fused.iter().zip(out_ref.iter()).enumerate() {
            let abs = (a - b).abs();
            let rel = abs / b.abs().max(1e-6);
            assert!(
                abs < 1e-3 || rel < 1e-5,
                "row {i}: fused={a} ref={b} abs={abs} rel={rel}"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn matvec_q4_1_simd_paths_match_scalar() {
        const QK: usize = 32;
        const BLOCK_BYTES: usize = 20;
        let m = 4;
        let k = 128;
        let blocks_per_row = k / QK;
        let total_blocks = m * blocks_per_row;

        let mut w_bytes = vec![0u8; total_blocks * BLOCK_BYTES];
        let mut s: u32 = 0x4141_F00D;
        for b in 0..total_blocks {
            let off = b * BLOCK_BYTES;
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let d = ((s >> 24) as f32 / 256.0) * 0.05 + 0.001;
            w_bytes[off..off + 2].copy_from_slice(&f16::from_f32(d).to_le_bytes());
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let m_off = ((s >> 24) as f32 / 256.0 - 0.5) * 0.1;
            w_bytes[off + 2..off + 4].copy_from_slice(&f16::from_f32(m_off).to_le_bytes());
            for j in 0..16 {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                w_bytes[off + 4 + j] = (s >> 24) as u8;
            }
        }
        let mut x = vec![0f32; k];
        for v in x.iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = (s as i32 as f32) / (i32::MAX as f32);
        }

        let mut out_scalar = vec![0f32; m];
        matvec_q4_1_w_f32_a_scalar(&w_bytes, &x, &mut out_scalar, m, k);

        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let mut out_avx2 = vec![0f32; m];
            unsafe { matvec_q4_1_w_f32_a_avx2(&w_bytes, &x, &mut out_avx2, m, k) };
            for (i, (a, b)) in out_scalar.iter().zip(out_avx2.iter()).enumerate() {
                let abs = (a - b).abs();
                let rel = abs / a.abs().max(1e-6);
                assert!(
                    abs < 1e-3 || rel < 1e-5,
                    "row {i}: scalar={a} avx2={b}"
                );
            }
        }
        if is_x86_feature_detected!("avx512f") {
            let mut out_avx512 = vec![0f32; m];
            unsafe { matvec_q4_1_w_f32_a_avx512(&w_bytes, &x, &mut out_avx512, m, k) };
            for (i, (a, b)) in out_scalar.iter().zip(out_avx512.iter()).enumerate() {
                let abs = (a - b).abs();
                let rel = abs / a.abs().max(1e-6);
                assert!(
                    abs < 1e-3 || rel < 1e-5,
                    "row {i}: scalar={a} avx512={b}"
                );
            }
        }
    }

    #[test]
    fn matvec_q5_0_matches_dequant_then_matvec() {
        const QK: usize = 32;
        const BLOCK_BYTES: usize = 22;
        let m = 5;
        let k = 96; // 3 blocks per row
        let blocks_per_row = k / QK;
        let total_blocks = m * blocks_per_row;

        let mut w_bytes = vec![0u8; total_blocks * BLOCK_BYTES];
        let mut s: u32 = 0xDEAD_BEEF;
        for b in 0..total_blocks {
            let off = b * BLOCK_BYTES;
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let d = ((s >> 24) as f32 / 256.0) * 0.05 + 0.001;
            let d_bytes = f16::from_f32(d).to_le_bytes();
            w_bytes[off] = d_bytes[0];
            w_bytes[off + 1] = d_bytes[1];
            // qh: 4 bytes
            for j in 0..4 {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                w_bytes[off + 2 + j] = (s >> 24) as u8;
            }
            // qs: 16 bytes
            for j in 0..16 {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                w_bytes[off + 6 + j] = (s >> 24) as u8;
            }
        }

        let mut x = vec![0f32; k];
        for v in x.iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = (s as i32 as f32) / (i32::MAX as f32);
        }

        let mut out_fused = vec![0f32; m];
        matvec_q5_0_w_f32_a(&w_bytes, &x, &mut out_fused, m, k);

        // Reference: dequant whole tensor to F32, then matvec_f32_scalar.
        let mut w_f32 = vec![0f32; m * k];
        rustllama_gguf::dequant::dequant_q5_0(&w_bytes, &mut w_f32);
        let mut out_ref = vec![0f32; m];
        matvec_f32_scalar(&w_f32, &x, &mut out_ref, m, k);

        for (i, (a, b)) in out_fused.iter().zip(out_ref.iter()).enumerate() {
            let abs_err = (a - b).abs();
            let rel_err = abs_err / b.abs().max(1e-6);
            assert!(
                abs_err < 1e-3 || rel_err < 1e-3,
                "row {i}: fused={a} ref={b} abs={abs_err} rel={rel_err}"
            );
        }
    }

    #[test]
    fn matvec_q5_1_matches_dequant_then_matvec() {
        // End-to-end equivalence check: fused on-the-fly dequant+matvec
        // for Q5_1 must equal "dequant whole tensor to f32, then plain
        // matvec_f32". Exercises both `d * q` and the `+ m` offset.
        const QK: usize = 32;
        const BLOCK_BYTES: usize = 24;
        let m = 5;
        let k = 96; // 3 blocks per row
        let blocks_per_row = k / QK;
        let total_blocks = m * blocks_per_row;

        let mut w_bytes = vec![0u8; total_blocks * BLOCK_BYTES];
        let mut s: u32 = 0x5151_BEEF;
        for b in 0..total_blocks {
            let off = b * BLOCK_BYTES;
            // d
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let d = ((s >> 24) as f32 / 256.0) * 0.05 + 0.001;
            w_bytes[off..off + 2].copy_from_slice(&f16::from_f32(d).to_le_bytes());
            // m
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let m_off = ((s >> 24) as f32 / 256.0 - 0.5) * 0.1;
            w_bytes[off + 2..off + 4].copy_from_slice(&f16::from_f32(m_off).to_le_bytes());
            // qh (4 bytes) + qs (16 bytes)
            for j in 0..(BLOCK_BYTES - 4) {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                w_bytes[off + 4 + j] = (s >> 24) as u8;
            }
        }
        let mut x = vec![0f32; k];
        for v in x.iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = (s as i32 as f32) / (i32::MAX as f32);
        }

        let mut out_fused = vec![0f32; m];
        matvec_q5_1_w_f32_a(&w_bytes, &x, &mut out_fused, m, k);

        let mut w_f32 = vec![0f32; m * k];
        rustllama_gguf::dequant::dequant_q5_1(&w_bytes, &mut w_f32);
        let mut out_ref = vec![0f32; m];
        matvec_f32_scalar(&w_f32, &x, &mut out_ref, m, k);

        for (i, (a, b)) in out_fused.iter().zip(out_ref.iter()).enumerate() {
            let abs_err = (a - b).abs();
            let rel_err = abs_err / b.abs().max(1e-6);
            assert!(
                abs_err < 1e-3 || rel_err < 1e-3,
                "row {i}: fused={a} ref={b} abs={abs_err} rel={rel_err}"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn matvec_q5_1_simd_paths_match_scalar() {
        const QK: usize = 32;
        const BLOCK_BYTES: usize = 24;
        let m = 4;
        let k = 128;
        let blocks_per_row = k / QK;
        let total_blocks = m * blocks_per_row;

        let mut w_bytes = vec![0u8; total_blocks * BLOCK_BYTES];
        let mut s: u32 = 0x5151_F00D;
        for b in 0..total_blocks {
            let off = b * BLOCK_BYTES;
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let d = ((s >> 24) as f32 / 256.0) * 0.05 + 0.001;
            w_bytes[off..off + 2].copy_from_slice(&f16::from_f32(d).to_le_bytes());
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let m_off = ((s >> 24) as f32 / 256.0 - 0.5) * 0.1;
            w_bytes[off + 2..off + 4].copy_from_slice(&f16::from_f32(m_off).to_le_bytes());
            for j in 0..(BLOCK_BYTES - 4) {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                w_bytes[off + 4 + j] = (s >> 24) as u8;
            }
        }
        let mut x = vec![0f32; k];
        for v in x.iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = (s as i32 as f32) / (i32::MAX as f32);
        }

        let mut out_scalar = vec![0f32; m];
        matvec_q5_1_w_f32_a_scalar(&w_bytes, &x, &mut out_scalar, m, k);

        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let mut out_avx2 = vec![0f32; m];
            unsafe { matvec_q5_1_w_f32_a_avx2(&w_bytes, &x, &mut out_avx2, m, k) };
            for (i, (a, b)) in out_scalar.iter().zip(out_avx2.iter()).enumerate() {
                let abs = (a - b).abs();
                let rel = abs / a.abs().max(1e-6);
                assert!(
                    abs < 1e-3 || rel < 1e-5,
                    "row {i}: scalar={a} avx2={b}"
                );
            }
        }
        if is_x86_feature_detected!("avx512f") {
            let mut out_avx512 = vec![0f32; m];
            unsafe { matvec_q5_1_w_f32_a_avx512(&w_bytes, &x, &mut out_avx512, m, k) };
            for (i, (a, b)) in out_scalar.iter().zip(out_avx512.iter()).enumerate() {
                let abs = (a - b).abs();
                let rel = abs / a.abs().max(1e-6);
                assert!(
                    abs < 1e-3 || rel < 1e-5,
                    "row {i}: scalar={a} avx512={b}"
                );
            }
        }
    }

    #[test]
    fn matvec_q8_0_matches_dequant_then_matvec() {
        // Build a Q8_0 weight block-by-block, dequant it to F32 via the
        // direct decoder, and check the fused matvec matches the F32 matvec.
        const QK: usize = 32;
        const BLOCK_BYTES: usize = 34;
        let m = 7;
        let k = 64; // 2 blocks per row
        let blocks_per_row = k / QK;
        let total_blocks = m * blocks_per_row;

        // Build random-ish bytes; bytes 0..1 are the scale, 2..33 are quants.
        let mut w_bytes = vec![0u8; total_blocks * BLOCK_BYTES];
        let mut s: u32 = 0xC0FFEE;
        for b in 0..total_blocks {
            let off = b * BLOCK_BYTES;
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            // Use a small fp16 scale (typical of Q8_0).
            let d = ((s >> 24) as f32 / 256.0) * 0.05 + 0.001;
            let d_bytes = f16::from_f32(d).to_le_bytes();
            w_bytes[off] = d_bytes[0];
            w_bytes[off + 1] = d_bytes[1];
            for j in 0..QK {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                w_bytes[off + 2 + j] = (s >> 24) as u8; // any byte value, interpreted as i8
            }
        }

        // Random x.
        let mut x = vec![0f32; k];
        for v in x.iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = (s as i32 as f32) / (i32::MAX as f32);
        }

        // Fused path
        let mut out_fused = vec![0f32; m];
        matvec_q8_0_w_f32_a(&w_bytes, &x, &mut out_fused, m, k);

        // Reference path: dequant whole tensor to F32, then matvec_f32_scalar.
        let mut w_f32 = vec![0f32; m * k];
        rustllama_gguf::dequant::dequant_q8_0(&w_bytes, &mut w_f32);
        let mut out_ref = vec![0f32; m];
        matvec_f32_scalar(&w_f32, &x, &mut out_ref, m, k);

        for (i, (a, b)) in out_fused.iter().zip(out_ref.iter()).enumerate() {
            let abs_err = (a - b).abs();
            let rel_err = abs_err / b.abs().max(1e-6);
            assert!(
                abs_err < 1e-4 || rel_err < 1e-4,
                "row {i}: fused={a} ref={b} abs={abs_err} rel={rel_err}"
            );
        }
    }

    #[test]
    fn matvec_f32_avx2_matches_scalar() {
        // Random fixture, asserts AVX2 dispatch produces same results as scalar.
        let m = 17;
        let k = 73;
        let mut w = vec![0f32; m * k];
        let mut x = vec![0f32; k];
        // Cheap deterministic noise via a simple LCG
        let mut s: u32 = 0xDEAD_BEEF;
        for v in w.iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = (s as i32 as f32) / (i32::MAX as f32) * 0.5;
        }
        for v in x.iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = (s as i32 as f32) / (i32::MAX as f32);
        }
        let mut out_a = vec![0f32; m];
        let mut out_b = vec![0f32; m];
        matvec_f32(&w, &x, &mut out_a, m, k);
        matvec_f32_scalar(&w, &x, &mut out_b, m, k);
        for (a, b) in out_a.iter().zip(out_b.iter()) {
            assert!((a - b).abs() < 1e-4, "AVX2 vs scalar mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn gqa_single_head_attention() {
        // n_heads=1, n_kv_heads=1, head_dim=2, kv_len=2.
        // Identity-ish setup: K = [[1,0],[0,1]], V = [[10,20],[30,40]], Q = [1,0]
        // scaled scores = [1/sqrt(2), 0]; softmax weights favor token 0
        // out should be a mix biased toward [10,20].
        let q = vec![1.0f32, 0.0];
        let k = vec![1.0f32, 0.0, 0.0, 1.0, 0.0, 0.0]; // max_ctx=3 but kv_len=2
        let v = vec![10.0f32, 20.0, 30.0, 40.0, 0.0, 0.0];
        let mut out = vec![0.0f32; 2];
        gqa_attention_one_step(&q, &k, &v, &mut out, 1, 1, 2, 3, 2);
        // softmax([1/sqrt(2), 0]) ~ [0.67, 0.33]; out ~ 0.67*[10,20] + 0.33*[30,40] ~ [16.6, 26.6]
        assert!(out[0] > 14.0 && out[0] < 18.0, "got {}", out[0]);
        assert!(out[1] > 24.0 && out[1] < 28.0, "got {}", out[1]);
    }

    /// Tiny PRNG: linear congruential. Same seed → same sequence. Used to
    /// stamp deterministic but non-trivial test inputs.
    fn lcg(state: &mut u32) -> f32 {
        *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        // Map to [-1, 1) so the i8 quant exercises both signs.
        ((*state >> 8) as f32) / (1u32 << 23) as f32 - 1.0
    }

    #[test]
    fn gqa_attention_q8_0_scalar_matches_f32_within_tolerance() {
        // n_heads=2 (GQA group), n_kv_heads=1, head_dim=16, kv_len=5.
        // Build a synthetic K/V then compare the F32 path's output to the
        // Q8_0 scalar path's output (with K/V quantized per-row first).
        let n_heads = 2usize;
        let n_kv_heads = 1usize;
        let head_dim = 16usize;
        let max_ctx = 8usize;
        let kv_len = 5usize;

        let mut s = 0xC0FFEE_u32;
        let q: Vec<f32> = (0..n_heads * head_dim).map(|_| lcg(&mut s)).collect();
        // Cache regions are [n_kv_heads * max_ctx * head_dim].
        let mut k_f32 = vec![0.0f32; n_kv_heads * max_ctx * head_dim];
        let mut v_f32 = vec![0.0f32; n_kv_heads * max_ctx * head_dim];
        for t in 0..kv_len {
            for d in 0..head_dim {
                k_f32[t * head_dim + d] = lcg(&mut s);
                v_f32[t * head_dim + d] = lcg(&mut s);
            }
        }

        // Quantize K and V per row.
        let mut k_q = vec![0i8; k_f32.len()];
        let mut v_q = vec![0i8; v_f32.len()];
        let mut k_scales = vec![0.0f32; n_kv_heads * max_ctx];
        let mut v_scales = vec![0.0f32; n_kv_heads * max_ctx];
        for t in 0..kv_len {
            // We need this helper from rustllama-models, but it lives in
            // a different crate. Reproduce the absmax→127 quant inline.
            let kr = &k_f32[t * head_dim..(t + 1) * head_dim];
            let vr = &v_f32[t * head_dim..(t + 1) * head_dim];
            let k_max = kr.iter().fold(0f32, |a, &b| a.max(b.abs()));
            let v_max = vr.iter().fold(0f32, |a, &b| a.max(b.abs()));
            let ks = if k_max == 0.0 { 1.0 } else { k_max / 127.0 };
            let vs = if v_max == 0.0 { 1.0 } else { v_max / 127.0 };
            k_scales[t] = ks;
            v_scales[t] = vs;
            for d in 0..head_dim {
                k_q[t * head_dim + d] =
                    (kr[d] / ks).round().clamp(-128.0, 127.0) as i8;
                v_q[t * head_dim + d] =
                    (vr[d] / vs).round().clamp(-128.0, 127.0) as i8;
            }
        }

        let mut out_f32 = vec![0.0f32; n_heads * head_dim];
        let mut out_q8 = vec![0.0f32; n_heads * head_dim];
        gqa_attention_one_step(
            &q, &k_f32, &v_f32, &mut out_f32, n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
        );
        gqa_attention_one_step_q8_0_scalar(
            &q, &k_q, &k_scales, &v_q, &v_scales, &mut out_q8, n_heads, n_kv_heads, head_dim,
            max_ctx, kv_len,
        );

        // With per-row absmax quant on 16-dim rows, per-element rounding
        // is bounded by scale/2 ≈ max/254. Cumulative output error across
        // 5 rows and a head_dim-long weighted sum stays under ~0.05 in
        // practice; loosen to 0.1 to absorb softmax sensitivity.
        for (i, (a, b)) in out_f32.iter().zip(out_q8.iter()).enumerate() {
            assert!(
                (a - b).abs() < 0.1,
                "i={i}: f32={a} q8={b} diff={}",
                (a - b).abs()
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn gqa_attention_q8_0_avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("skipped: host lacks AVX2/FMA");
            return;
        }
        // Bigger inputs than the parity test above so the SIMD 8-wide
        // loop actually runs (head_dim multiple of 8) and we see the
        // tail-cleanup path (head_dim that's also NOT a multiple of 16).
        let n_heads = 4usize;
        let n_kv_heads = 2usize;
        let head_dim = 24usize;
        let max_ctx = 16usize;
        let kv_len = 11usize;

        let mut s = 0xBADC0DE_u32;
        let q: Vec<f32> = (0..n_heads * head_dim).map(|_| lcg(&mut s)).collect();
        let total_rows = n_kv_heads * max_ctx;
        let mut k_q = vec![0i8; total_rows * head_dim];
        let mut v_q = vec![0i8; total_rows * head_dim];
        let mut k_scales = vec![0.0f32; total_rows];
        let mut v_scales = vec![0.0f32; total_rows];
        for h in 0..n_kv_heads {
            for t in 0..kv_len {
                let row_idx = h * max_ctx + t;
                let off = row_idx * head_dim;
                let mut k_max = 0f32;
                let mut v_max = 0f32;
                let mut tmp_k = vec![0f32; head_dim];
                let mut tmp_v = vec![0f32; head_dim];
                for d in 0..head_dim {
                    tmp_k[d] = lcg(&mut s);
                    tmp_v[d] = lcg(&mut s);
                    k_max = k_max.max(tmp_k[d].abs());
                    v_max = v_max.max(tmp_v[d].abs());
                }
                let ks = if k_max == 0.0 { 1.0 } else { k_max / 127.0 };
                let vs = if v_max == 0.0 { 1.0 } else { v_max / 127.0 };
                k_scales[row_idx] = ks;
                v_scales[row_idx] = vs;
                for d in 0..head_dim {
                    k_q[off + d] = (tmp_k[d] / ks).round().clamp(-128.0, 127.0) as i8;
                    v_q[off + d] = (tmp_v[d] / vs).round().clamp(-128.0, 127.0) as i8;
                }
            }
        }

        let mut out_scalar = vec![0.0f32; n_heads * head_dim];
        let mut out_avx2 = vec![0.0f32; n_heads * head_dim];
        gqa_attention_one_step_q8_0_scalar(
            &q,
            &k_q,
            &k_scales,
            &v_q,
            &v_scales,
            &mut out_scalar,
            n_heads,
            n_kv_heads,
            head_dim,
            max_ctx,
            kv_len,
        );
        // SAFETY: feature-detected above.
        unsafe {
            gqa_attention_one_step_q8_0_avx2(
                &q,
                &k_q,
                &k_scales,
                &v_q,
                &v_scales,
                &mut out_avx2,
                n_heads,
                n_kv_heads,
                head_dim,
                max_ctx,
                kv_len,
            );
        }
        // The two paths do the same math; only the order/grouping of
        // FMAs differs. Floating-point reorderings can perturb the LSBs
        // but with 11 terms in the dot product, the difference should
        // stay well below 1e-4.
        for (i, (a, b)) in out_scalar.iter().zip(out_avx2.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "i={i}: scalar={a} avx2={b} diff={}",
                (a - b).abs()
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn matvec_f32_avx512_matches_scalar() {
        if !is_x86_feature_detected!("avx512f") {
            eprintln!("skipped: host lacks AVX-512F");
            return;
        }
        // m=3 outputs × k=70 inputs — exercises the 64-wide outer tile,
        // the 16-wide mid tile, and the scalar tail (70 = 64 + 6).
        let m = 3usize;
        let k = 70usize;
        let mut s = 0xFEED_FACE_u32;
        let w: Vec<f32> = (0..m * k).map(|_| lcg(&mut s)).collect();
        let x: Vec<f32> = (0..k).map(|_| lcg(&mut s)).collect();

        let mut out_scalar = vec![0f32; m];
        let mut out_avx512 = vec![0f32; m];
        matvec_f32_scalar(&w, &x, &mut out_scalar, m, k);
        // SAFETY: feature-detected above.
        unsafe { matvec_f32_avx512(&w, &x, &mut out_avx512, m, k) };

        for (i, (a, b)) in out_scalar.iter().zip(out_avx512.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "i={i}: scalar={a} avx512={b} diff={}",
                (a - b).abs()
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn gqa_attention_avx512_matches_scalar() {
        if !is_x86_feature_detected!("avx512f") {
            eprintln!("skipped: host lacks AVX-512F");
            return;
        }
        // head_dim=24 forces the 16-wide tile + 8-element tail.
        let n_heads = 2usize;
        let n_kv_heads = 1usize;
        let head_dim = 24usize;
        let max_ctx = 12usize;
        let kv_len = 7usize;
        let mut s = 0xD00FBEEF_u32;
        let q: Vec<f32> = (0..n_heads * head_dim).map(|_| lcg(&mut s)).collect();
        let k_cache: Vec<f32> = (0..n_kv_heads * max_ctx * head_dim)
            .map(|_| lcg(&mut s))
            .collect();
        let v_cache: Vec<f32> = (0..n_kv_heads * max_ctx * head_dim)
            .map(|_| lcg(&mut s))
            .collect();
        let mut out_scalar = vec![0f32; n_heads * head_dim];
        let mut out_avx512 = vec![0f32; n_heads * head_dim];
        gqa_attention_one_step_scalar(
            &q,
            &k_cache,
            &v_cache,
            &mut out_scalar,
            n_heads,
            n_kv_heads,
            head_dim,
            max_ctx,
            kv_len,
        );
        // SAFETY: feature-detected above.
        unsafe {
            gqa_attention_one_step_avx512(
                &q,
                &k_cache,
                &v_cache,
                &mut out_avx512,
                n_heads,
                n_kv_heads,
                head_dim,
                max_ctx,
                kv_len,
            );
        }
        for (i, (a, b)) in out_scalar.iter().zip(out_avx512.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "i={i}: scalar={a} avx512={b} diff={}",
                (a - b).abs()
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn gqa_attention_q8_0_avx512_matches_scalar() {
        if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512bw") {
            eprintln!("skipped: host lacks AVX-512F/BW");
            return;
        }
        // head_dim=24 again to exercise tile + tail.
        let n_heads = 4usize;
        let n_kv_heads = 2usize;
        let head_dim = 24usize;
        let max_ctx = 16usize;
        let kv_len = 11usize;
        let mut s = 0xBEADBEEF_u32;
        let q: Vec<f32> = (0..n_heads * head_dim).map(|_| lcg(&mut s)).collect();
        let total_rows = n_kv_heads * max_ctx;
        let mut k_q = vec![0i8; total_rows * head_dim];
        let mut v_q = vec![0i8; total_rows * head_dim];
        let mut k_scales = vec![0f32; total_rows];
        let mut v_scales = vec![0f32; total_rows];
        for h in 0..n_kv_heads {
            for t in 0..kv_len {
                let row_idx = h * max_ctx + t;
                let off = row_idx * head_dim;
                let mut tmp_k = vec![0f32; head_dim];
                let mut tmp_v = vec![0f32; head_dim];
                let mut k_max = 0f32;
                let mut v_max = 0f32;
                for d in 0..head_dim {
                    tmp_k[d] = lcg(&mut s);
                    tmp_v[d] = lcg(&mut s);
                    k_max = k_max.max(tmp_k[d].abs());
                    v_max = v_max.max(tmp_v[d].abs());
                }
                let ks = if k_max == 0.0 { 1.0 } else { k_max / 127.0 };
                let vs = if v_max == 0.0 { 1.0 } else { v_max / 127.0 };
                k_scales[row_idx] = ks;
                v_scales[row_idx] = vs;
                for d in 0..head_dim {
                    k_q[off + d] = (tmp_k[d] / ks).round().clamp(-128.0, 127.0) as i8;
                    v_q[off + d] = (tmp_v[d] / vs).round().clamp(-128.0, 127.0) as i8;
                }
            }
        }

        let mut out_scalar = vec![0f32; n_heads * head_dim];
        let mut out_avx512 = vec![0f32; n_heads * head_dim];
        gqa_attention_one_step_q8_0_scalar(
            &q,
            &k_q,
            &k_scales,
            &v_q,
            &v_scales,
            &mut out_scalar,
            n_heads,
            n_kv_heads,
            head_dim,
            max_ctx,
            kv_len,
        );
        // SAFETY: feature-detected above.
        unsafe {
            gqa_attention_one_step_q8_0_avx512(
                &q,
                &k_q,
                &k_scales,
                &v_q,
                &v_scales,
                &mut out_avx512,
                n_heads,
                n_kv_heads,
                head_dim,
                max_ctx,
                kv_len,
            );
        }
        for (i, (a, b)) in out_scalar.iter().zip(out_avx512.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "i={i}: scalar={a} avx512={b} diff={}",
                (a - b).abs()
            );
        }
    }

    // ---- embed_lookup tests: row-pluck against full dequant -------------

    /// Build a random K-quant table of `n_rows` rows × `d` columns, where
    /// each row is `blocks_per_row` × `block_bytes` bytes. The seed wraps
    /// into d_min / d_scale fields so the test is reproducible.
    fn random_qk_table(n_rows: usize, d: usize, block_bytes: usize, seed: u32) -> Vec<u8> {
        const QK_K: usize = 256;
        let blocks_per_row = d / QK_K;
        let mut bytes = vec![0u8; n_rows * blocks_per_row * block_bytes];
        let mut s = seed;
        for b in 0..bytes.len() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            bytes[b] = (s >> 24) as u8;
        }
        // Stamp valid f16 d/dmin into the first 4 bytes of each Q4_K / Q5_K
        // block. Q6_K stores d at offset 208–209.
        for b in 0..n_rows * blocks_per_row {
            let off = b * block_bytes;
            let f = (((b as u32).wrapping_mul(0x9E3779B9) >> 16) as f32 / u16::MAX as f32) * 0.05 + 0.001;
            let fbytes = half::f16::from_f32(f).to_le_bytes();
            if block_bytes == 210 {
                bytes[off + 208] = fbytes[0];
                bytes[off + 209] = fbytes[1];
            } else {
                bytes[off] = fbytes[0];
                bytes[off + 1] = fbytes[1];
                let m = (((b as u32).wrapping_mul(0x85EBCA6B) >> 16) as f32 / u16::MAX as f32)
                    * 0.02 + 0.0005;
                let mbytes = half::f16::from_f32(m).to_le_bytes();
                bytes[off + 2] = mbytes[0];
                bytes[off + 3] = mbytes[1];
            }
        }
        bytes
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn matvec_q4_k_avx512_matches_scalar() {
        if !is_x86_feature_detected!("avx512f") {
            eprintln!("skipped: host lacks AVX-512F");
            return;
        }
        const BLOCK_BYTES: usize = 144;
        let m = 5usize;
        let k = 512usize; // 2 super-blocks per row
        let w = random_qk_table(m, k, BLOCK_BYTES, 0xC4F0);
        let mut s = 0xBEEF1234u32;
        let x: Vec<f32> = (0..k).map(|_| lcg(&mut s)).collect();
        let mut out_scalar = vec![0f32; m];
        let mut out_avx512 = vec![0f32; m];
        matvec_q4_k_w_f32_a_scalar(&w, &x, &mut out_scalar, m, k);
        // SAFETY: feature-detected above.
        unsafe { matvec_q4_k_w_f32_a_avx512(&w, &x, &mut out_avx512, m, k) };
        for (i, (a, b)) in out_scalar.iter().zip(out_avx512.iter()).enumerate() {
            let abs = (a - b).abs();
            let rel = abs / a.abs().max(1e-6);
            assert!(
                abs < 1e-3 || rel < 1e-5,
                "row {i}: scalar={a} avx512={b} abs={abs} rel={rel}"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn matvec_q5_k_avx512_matches_scalar() {
        if !is_x86_feature_detected!("avx512f") {
            eprintln!("skipped: host lacks AVX-512F");
            return;
        }
        const BLOCK_BYTES: usize = 176;
        let m = 5usize;
        let k = 512usize;
        let w = random_qk_table(m, k, BLOCK_BYTES, 0xC5F0);
        let mut s = 0xBEEF5678u32;
        let x: Vec<f32> = (0..k).map(|_| lcg(&mut s)).collect();
        let mut out_scalar = vec![0f32; m];
        let mut out_avx512 = vec![0f32; m];
        matvec_q5_k_w_f32_a_scalar(&w, &x, &mut out_scalar, m, k);
        unsafe { matvec_q5_k_w_f32_a_avx512(&w, &x, &mut out_avx512, m, k) };
        for (i, (a, b)) in out_scalar.iter().zip(out_avx512.iter()).enumerate() {
            let abs = (a - b).abs();
            let rel = abs / a.abs().max(1e-6);
            assert!(
                abs < 1e-3 || rel < 1e-5,
                "row {i}: scalar={a} avx512={b} abs={abs} rel={rel}"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn matvec_q6_k_avx512_matches_scalar() {
        if !is_x86_feature_detected!("avx512f") {
            eprintln!("skipped: host lacks AVX-512F");
            return;
        }
        const BLOCK_BYTES: usize = 210;
        let m = 5usize;
        let k = 512usize;
        let w = random_qk_table(m, k, BLOCK_BYTES, 0xC6F0);
        let mut s = 0xBEEFABCDu32;
        let x: Vec<f32> = (0..k).map(|_| lcg(&mut s)).collect();
        let mut out_scalar = vec![0f32; m];
        let mut out_avx512 = vec![0f32; m];
        matvec_q6_k_w_f32_a_scalar(&w, &x, &mut out_scalar, m, k);
        unsafe { matvec_q6_k_w_f32_a_avx512(&w, &x, &mut out_avx512, m, k) };
        for (i, (a, b)) in out_scalar.iter().zip(out_avx512.iter()).enumerate() {
            let abs = (a - b).abs();
            let rel = abs / a.abs().max(1e-6);
            assert!(
                abs < 1e-2 || rel < 1e-4,
                "row {i}: scalar={a} avx512={b} abs={abs} rel={rel}"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn matvec_q8_0_w_avx512_matches_scalar() {
        if !is_x86_feature_detected!("avx512f") {
            eprintln!("skipped: host lacks AVX-512F");
            return;
        }
        const BLOCK_BYTES: usize = 34;
        const QK: usize = 32;
        let m = 4usize;
        let k = 128usize; // 4 blocks per row
        // Build a random Q8_0 table with reasonable scales.
        let mut w = vec![0u8; m * (k / QK) * BLOCK_BYTES];
        let mut s = 0xDEADBEEFu32;
        for b in 0..(m * (k / QK)) {
            let off = b * BLOCK_BYTES;
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let scale = ((s >> 24) as f32 / 256.0) * 0.05 + 0.001;
            let sb = half::f16::from_f32(scale).to_le_bytes();
            w[off] = sb[0];
            w[off + 1] = sb[1];
            for j in 0..QK {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                w[off + 2 + j] = (s >> 24) as u8; // arbitrary i8 quants
            }
        }
        let mut sx = 0xC0FFEEu32;
        let x: Vec<f32> = (0..k).map(|_| lcg(&mut sx)).collect();
        let mut out_scalar = vec![0f32; m];
        let mut out_avx512 = vec![0f32; m];
        matvec_q8_0_w_f32_a_scalar(&w, &x, &mut out_scalar, m, k);
        unsafe { matvec_q8_0_w_f32_a_avx512(&w, &x, &mut out_avx512, m, k) };
        for (i, (a, b)) in out_scalar.iter().zip(out_avx512.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-3 || (a - b).abs() / a.abs().max(1e-6) < 1e-5,
                "row {i}: scalar={a} avx512={b}"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn matvec_q5_0_w_avx512_matches_scalar() {
        if !is_x86_feature_detected!("avx512f") {
            eprintln!("skipped: host lacks AVX-512F");
            return;
        }
        const BLOCK_BYTES: usize = 22;
        const QK: usize = 32;
        let m = 4usize;
        let k = 128usize;
        let mut w = vec![0u8; m * (k / QK) * BLOCK_BYTES];
        let mut s = 0xC5F0F00Du32;
        for b in 0..(m * (k / QK)) {
            let off = b * BLOCK_BYTES;
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let scale = ((s >> 24) as f32 / 256.0) * 0.05 + 0.001;
            let sb = half::f16::from_f32(scale).to_le_bytes();
            w[off] = sb[0];
            w[off + 1] = sb[1];
            // qh (4 bytes) + qs (16 bytes) — arbitrary bits.
            for j in 0..(BLOCK_BYTES - 2) {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                w[off + 2 + j] = (s >> 24) as u8;
            }
        }
        let mut sx = 0xBADBEEFu32;
        let x: Vec<f32> = (0..k).map(|_| lcg(&mut sx)).collect();
        let mut out_scalar = vec![0f32; m];
        let mut out_avx512 = vec![0f32; m];
        matvec_q5_0_w_f32_a_scalar(&w, &x, &mut out_scalar, m, k);
        unsafe { matvec_q5_0_w_f32_a_avx512(&w, &x, &mut out_avx512, m, k) };
        for (i, (a, b)) in out_scalar.iter().zip(out_avx512.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-3 || (a - b).abs() / a.abs().max(1e-6) < 1e-5,
                "row {i}: scalar={a} avx512={b}"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn embed_lookup_q4_k_avx512_matches_scalar() {
        if !is_x86_feature_detected!("avx512f") {
            eprintln!("skipped: host lacks AVX-512F");
            return;
        }
        const BLOCK_BYTES: usize = 144;
        let n_rows = 6;
        let d = 512; // 2 super-blocks per row
        let table = random_qk_table(n_rows, d, BLOCK_BYTES, 0xE4B0);
        let ids = [0i32, 1, 3, 5];
        let mut out_scalar = vec![0f32; ids.len() * d];
        let mut out_avx512 = vec![0f32; ids.len() * d];
        embed_lookup_q4_k_scalar(&table, &ids, &mut out_scalar, d);
        unsafe { embed_lookup_q4_k_avx512(&table, &ids, &mut out_avx512, d) };
        for (i, (a, b)) in out_scalar.iter().zip(out_avx512.iter()).enumerate() {
            // Per-element error tolerance: the only operations are
            // d*q - m where d, q, m are exact f32 — but reorderings in
            // SIMD FMA mean the result can differ from scalar by ~1
            // ULP. 1e-5 absolute is generous and catches real bugs.
            assert!(
                (a - b).abs() < 1e-5,
                "idx {i}: scalar={a} avx512={b}"
            );
        }
    }

    #[test]
    fn embed_lookup_q4_k_matches_dequant_then_lookup() {
        const BLOCK_BYTES: usize = 144;
        let n_rows = 7;
        let d = 512; // 2 super-blocks per row
        let table = random_qk_table(n_rows, d, BLOCK_BYTES, 0xC0FFEE);

        let ids = [0i32, 2, 5, 6];
        let mut out_fused = vec![0f32; ids.len() * d];
        embed_lookup_q4_k(&table, &ids, &mut out_fused, d);

        let mut full_f32 = vec![0f32; n_rows * d];
        rustllama_gguf::dequant::dequant_q4_k(&table, &mut full_f32);
        let mut out_ref = vec![0f32; ids.len() * d];
        embed_lookup_f32(&full_f32, &ids, &mut out_ref, d);

        for (i, (a, b)) in out_fused.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "idx {i}: fused={a} ref={b}"
            );
        }
    }

    #[test]
    fn embed_lookup_q5_k_matches_dequant_then_lookup() {
        const BLOCK_BYTES: usize = 176;
        let n_rows = 5;
        let d = 512;
        let table = random_qk_table(n_rows, d, BLOCK_BYTES, 0xBADBEEF);

        let ids = [3i32, 0, 4, 1];
        let mut out_fused = vec![0f32; ids.len() * d];
        embed_lookup_q5_k(&table, &ids, &mut out_fused, d);

        let mut full_f32 = vec![0f32; n_rows * d];
        rustllama_gguf::dequant::dequant_q5_k(&table, &mut full_f32);
        let mut out_ref = vec![0f32; ids.len() * d];
        embed_lookup_f32(&full_f32, &ids, &mut out_ref, d);

        for (i, (a, b)) in out_fused.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "idx {i}: fused={a} ref={b}"
            );
        }
    }

    #[test]
    fn embed_lookup_q6_k_matches_dequant_then_lookup() {
        const BLOCK_BYTES: usize = 210;
        let n_rows = 6;
        let d = 512;
        let table = random_qk_table(n_rows, d, BLOCK_BYTES, 0x5EED_5EED);

        let ids = [0i32, 1, 5];
        let mut out_fused = vec![0f32; ids.len() * d];
        embed_lookup_q6_k(&table, &ids, &mut out_fused, d);

        let mut full_f32 = vec![0f32; n_rows * d];
        rustllama_gguf::dequant::dequant_q6_k(&table, &mut full_f32);
        let mut out_ref = vec![0f32; ids.len() * d];
        embed_lookup_f32(&full_f32, &ids, &mut out_ref, d);

        for (i, (a, b)) in out_fused.iter().zip(out_ref.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "idx {i}: fused={a} ref={b}"
            );
        }
    }

    // ============================================================
    // Tier A SIMD parity tests
    // ============================================================
    //
    // These compare the runtime-dispatched SIMD path (used by the
    // public functions on hosts that report AVX2 / AVX-512) against
    // the pure-scalar reference implementations. Inputs cover a
    // mix of element counts that exercise both the SIMD body and
    // the scalar tail (sizes not multiples of 8 / 16).

    fn rand_vec(n: usize, seed: u32) -> Vec<f32> {
        // Tiny deterministic LCG — no rand crate dep, no allocation
        // beyond the Vec. Range [-4, 4] keeps softmax/expf well in
        // the supported [-88, 88] range and exercises the
        // numerically-tricky region near zero.
        let mut s: u32 = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let u = (s >> 8) as f32 / (1u32 << 24) as f32;
                u * 8.0 - 4.0
            })
            .collect()
    }

    #[test]
    fn softmax_simd_parity_with_scalar() {
        for &n in &[1usize, 3, 7, 8, 15, 16, 17, 31, 32, 33, 127, 128, 129, 1024, 4097] {
            let input = rand_vec(n, n as u32);
            let mut a = input.clone();
            softmax_f32_inplace(&mut a);
            let mut b = input.clone();
            softmax_f32_inplace_scalar(&mut b);
            for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
                assert!(
                    (av - bv).abs() < 1e-5,
                    "softmax SIMD parity at n={n} idx={i}: simd={av} scalar={bv}"
                );
            }
            // Outputs are a valid probability distribution.
            let sum: f32 = a.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "softmax SIMD output for n={n} did not sum to 1.0: {sum}"
            );
        }
    }

    /// F4: a Rust re-port of the SYCL IQ4_NL kernel arithmetic. This
    /// mirrors `matvec_iq4_nl_packed_f32_usm_impl` in
    /// `rsl_kernels.cpp` line-for-line — same byte layout reads,
    /// same FMA order, same KVALUES codebook lookup. The parity gate
    /// asserts it matches the existing scalar CPU reference
    /// (`matvec_iq4_nl_w_f32_a_scalar`), which pins the SYCL
    /// arithmetic against the same correctness contract even
    /// without GPU hardware to execute the device kernel.
    fn iq4_nl_sycl_port_scalar(
        w_bytes: &[u8],
        x: &[f32],
        out: &mut [f32],
        m: usize,
        k: usize,
    ) {
        const BLOCK_BYTES: usize = 18;
        const QK: usize = 32;
        let blocks_per_row = k / QK;
        let bytes_per_row = blocks_per_row * BLOCK_BYTES;
        for mi in 0..m {
            let row_start = mi * bytes_per_row;
            let mut acc = 0.0f32;
            for b in 0..blocks_per_row {
                let off = row_start + b * BLOCK_BYTES;
                let d_bits = (w_bytes[off] as u16) | ((w_bytes[off + 1] as u16) << 8);
                let d = half::f16::from_bits(d_bits).to_f32();
                let x_base = b * QK;
                for j in 0..16 {
                    let q = w_bytes[off + 2 + j];
                    let lo = (q & 0x0F) as usize;
                    let hi = (q >> 4) as usize;
                    acc += d * (KVALUES_IQ4XS[lo] as f32) * x[x_base + j];
                    acc += d * (KVALUES_IQ4XS[hi] as f32) * x[x_base + j + 16];
                }
            }
            out[mi] = acc;
        }
    }

    /// F4: Rust re-port of the SYCL IQ4_XS kernel arithmetic. Same
    /// rationale as `iq4_nl_sycl_port_scalar`.
    fn iq4_xs_sycl_port_scalar(
        w_bytes: &[u8],
        x: &[f32],
        out: &mut [f32],
        m: usize,
        k: usize,
    ) {
        const BLOCK_BYTES: usize = 136;
        const QK_K: usize = 256;
        let blocks_per_row = k / QK_K;
        let bytes_per_row = blocks_per_row * BLOCK_BYTES;
        for mi in 0..m {
            let row_start = mi * bytes_per_row;
            let mut acc = 0.0f32;
            for b in 0..blocks_per_row {
                let off = row_start + b * BLOCK_BYTES;
                let d_bits = (w_bytes[off] as u16) | ((w_bytes[off + 1] as u16) << 8);
                let d = half::f16::from_bits(d_bits).to_f32();
                let scales_h = (w_bytes[off + 2] as u16) | ((w_bytes[off + 3] as u16) << 8);
                let scales_l = &w_bytes[off + 4..off + 8];
                let qs = &w_bytes[off + 8..off + 8 + 128];
                let x_base = b * QK_K;
                for ib in 0..8 {
                    let lo_nibble = if ib % 2 == 0 {
                        scales_l[ib / 2] & 0x0F
                    } else {
                        scales_l[ib / 2] >> 4
                    };
                    let hi_bits = ((scales_h >> (2 * ib)) & 0x03) as u8;
                    let ls_i = (lo_nibble | (hi_bits << 4)) as i8 - 32;
                    let sub_d = d * (ls_i as f32);
                    let q_off = ib * 16;
                    let x_off = ib * 32;
                    for j in 0..16 {
                        let q = qs[q_off + j];
                        let lo = (q & 0x0F) as usize;
                        let hi = (q >> 4) as usize;
                        acc += sub_d * (KVALUES_IQ4XS[lo] as f32) * x[x_base + x_off + j];
                        acc += sub_d * (KVALUES_IQ4XS[hi] as f32) * x[x_base + x_off + 16 + j];
                    }
                }
            }
            out[mi] = acc;
        }
    }

    /// Synthesize a deterministic IQ4_NL byte buffer (`m * (k/32) *
    /// 18` bytes) — enough to drive the parity test without a GGUF
    /// dependency.
    fn synth_iq4_nl_bytes(m: usize, k: usize, seed: u32) -> Vec<u8> {
        let blocks = m * (k / 32);
        let mut bytes = vec![0u8; blocks * 18];
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
        for chunk in bytes.chunks_mut(18) {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            // f16 d in a sensible range, ~[-0.5, 0.5].
            let scale_f32 = ((s >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 1.0;
            let d_bits = half::f16::from_f32(scale_f32).to_bits();
            chunk[0] = (d_bits & 0xFF) as u8;
            chunk[1] = ((d_bits >> 8) & 0xFF) as u8;
            for b in chunk[2..18].iter_mut() {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                *b = (s >> 24) as u8;
            }
        }
        bytes
    }

    /// Synthesize a deterministic IQ4_XS byte buffer.
    fn synth_iq4_xs_bytes(m: usize, k: usize, seed: u32) -> Vec<u8> {
        let blocks = m * (k / 256);
        let mut bytes = vec![0u8; blocks * 136];
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
        for chunk in bytes.chunks_mut(136) {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let scale_f32 = ((s >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 1.0;
            let d_bits = half::f16::from_f32(scale_f32).to_bits();
            chunk[0] = (d_bits & 0xFF) as u8;
            chunk[1] = ((d_bits >> 8) & 0xFF) as u8;
            for b in chunk[2..136].iter_mut() {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                *b = (s >> 24) as u8;
            }
        }
        bytes
    }

    #[test]
    fn iq4_nl_sycl_arithmetic_matches_cpu_reference() {
        for &(m, k) in &[(1usize, 32), (3, 64), (8, 128), (5, 1024)] {
            let w = synth_iq4_nl_bytes(m, k, m as u32 * 31 + k as u32);
            let x = rand_vec(k, (m * k) as u32);
            let mut out_sycl = vec![0f32; m];
            let mut out_ref = vec![0f32; m];
            iq4_nl_sycl_port_scalar(&w, &x, &mut out_sycl, m, k);
            matvec_iq4_nl_w_f32_a_scalar(&w, &x, &mut out_ref, m, k);
            for (i, (&a, &b)) in out_sycl.iter().zip(out_ref.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-4,
                    "iq4_nl SYCL-port parity (m={m}, k={k}) idx={i}: sycl={a} ref={b}"
                );
            }
        }
    }

    #[test]
    fn iq4_xs_sycl_arithmetic_matches_cpu_reference() {
        for &(m, k) in &[(1usize, 256), (3, 512), (5, 1024)] {
            let w = synth_iq4_xs_bytes(m, k, m as u32 * 31 + k as u32);
            let x = rand_vec(k, (m * k) as u32);
            let mut out_sycl = vec![0f32; m];
            let mut out_ref = vec![0f32; m];
            iq4_xs_sycl_port_scalar(&w, &x, &mut out_sycl, m, k);
            matvec_iq4_xs_w_f32_a_scalar(&w, &x, &mut out_ref, m, k);
            for (i, (&a, &b)) in out_sycl.iter().zip(out_ref.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-4,
                    "iq4_xs SYCL-port parity (m={m}, k={k}) idx={i}: sycl={a} ref={b}"
                );
            }
        }
    }

    /// Rust port of the IQ2_XXS SYCL kernel arithmetic. Mirrors
    /// `matvec_iq2_xxs_packed_f32_usm_impl` in `rsl_kernels.cpp`
    /// byte-for-byte (same LE loads from `aux0`/`aux1`, same
    /// sub-block scale `db = d * (0.5 + (aux1 >> 28)) * 0.25`,
    /// same per-chunk grid lookup + sign-mask multiply). Pinned
    /// against the existing scalar reference
    /// `matvec_iq2_xxs_w_f32_a_scalar` so the SYCL arithmetic is
    /// validated without GPU hardware.
    fn iq2_xxs_sycl_port_scalar(
        w_bytes: &[u8],
        x: &[f32],
        out: &mut [f32],
        m: usize,
        k: usize,
    ) {
        use rustllama_gguf::dequant::{IQ2XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
        const BLOCK_BYTES: usize = 66;
        const QK_K: usize = 256;
        let blocks_per_row = k / QK_K;
        let bytes_per_row = blocks_per_row * BLOCK_BYTES;
        for mi in 0..m {
            let row_start = mi * bytes_per_row;
            let mut acc = 0.0f32;
            for b in 0..blocks_per_row {
                let off = row_start + b * BLOCK_BYTES;
                let d_bits = (w_bytes[off] as u16) | ((w_bytes[off + 1] as u16) << 8);
                let d = half::f16::from_bits(d_bits).to_f32();
                let x_base = b * QK_K;
                for ib32 in 0..8 {
                    let qb = off + 2 + 8 * ib32;
                    let aux0 = (w_bytes[qb] as u32)
                        | ((w_bytes[qb + 1] as u32) << 8)
                        | ((w_bytes[qb + 2] as u32) << 16)
                        | ((w_bytes[qb + 3] as u32) << 24);
                    let aux1 = (w_bytes[qb + 4] as u32)
                        | ((w_bytes[qb + 5] as u32) << 8)
                        | ((w_bytes[qb + 6] as u32) << 16)
                        | ((w_bytes[qb + 7] as u32) << 24);
                    let db = d * (0.5 + (aux1 >> 28) as f32) * 0.25;
                    for l in 0..4 {
                        let grid_idx = ((aux0 >> (8 * l)) & 0xFF) as usize;
                        let grid_bits = IQ2XXS_GRID[grid_idx];
                        let signs = KSIGNS_IQ2XS[((aux1 >> (7 * l)) & 127) as usize];
                        let x_off = x_base + ib32 * 32 + l * 8;
                        for j in 0..8 {
                            let gi = ((grid_bits >> (j * 8)) & 0xFF) as u8;
                            let s = if signs & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                            acc += db * (gi as f32) * s * x[x_off + j];
                        }
                    }
                }
            }
            out[mi] = acc;
        }
    }

    fn synth_iq2_xxs_bytes(m: usize, k: usize, seed: u32) -> Vec<u8> {
        let blocks = m * (k / 256);
        let mut bytes = vec![0u8; blocks * 66];
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
        for chunk in bytes.chunks_mut(66) {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let scale_f32 = ((s >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 1.0;
            let d_bits = half::f16::from_f32(scale_f32).to_bits();
            chunk[0] = (d_bits & 0xFF) as u8;
            chunk[1] = ((d_bits >> 8) & 0xFF) as u8;
            for b in chunk[2..66].iter_mut() {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                *b = (s >> 24) as u8;
            }
        }
        bytes
    }

    #[test]
    fn iq2_xxs_sycl_arithmetic_matches_cpu_reference() {
        for &(m, k) in &[(1usize, 256), (3, 512), (5, 1024)] {
            let w = synth_iq2_xxs_bytes(m, k, m as u32 * 37 + k as u32);
            let x = rand_vec(k, (m * k + 11) as u32);
            let mut out_sycl = vec![0f32; m];
            let mut out_ref = vec![0f32; m];
            iq2_xxs_sycl_port_scalar(&w, &x, &mut out_sycl, m, k);
            matvec_iq2_xxs_w_f32_a_scalar(&w, &x, &mut out_ref, m, k);
            for (i, (&a, &b)) in out_sycl.iter().zip(out_ref.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-4,
                    "iq2_xxs SYCL-port parity (m={m}, k={k}) idx={i}: sycl={a} ref={b}"
                );
            }
        }
    }

    /// Rust port of the IQ1_M SYCL kernel arithmetic. Mirrors
    /// `matvec_iq1_m_packed_f32_usm_impl` in `rsl_kernels.cpp`
    /// byte-for-byte (same scale-word load order, same f16 d
    /// reassembly from packed nibbles, same dl1/dl2 sub-block
    /// scales, same per-lane delta-sign flip, same 11-bit grid
    /// index reconstruction).
    fn iq1_m_sycl_port_scalar(
        w_bytes: &[u8],
        x: &[f32],
        out: &mut [f32],
        m: usize,
        k: usize,
    ) {
        use rustllama_gguf::dequant::IQ1S_DELTA;
        use rustllama_gguf::iq1_grid::IQ1S_GRID;
        const BLOCK_BYTES: usize = 56;
        const QK_K: usize = 256;
        let blocks_per_row = k / QK_K;
        let bytes_per_row = blocks_per_row * BLOCK_BYTES;
        for mi in 0..m {
            let row_start = mi * bytes_per_row;
            let mut acc = 0.0f32;
            for b in 0..blocks_per_row {
                let off = row_start + b * BLOCK_BYTES;
                let qs_off = off;
                let qh_off = off + 32;
                let sc_off = off + 48;
                let mut sc = [0u16; 4];
                for ii in 0..4 {
                    sc[ii] = (w_bytes[sc_off + ii * 2] as u16)
                        | ((w_bytes[sc_off + ii * 2 + 1] as u16) << 8);
                }
                let d_bits: u16 = (sc[0] >> 12)
                    | ((sc[1] >> 8) & 0x00F0)
                    | ((sc[2] >> 4) & 0x0F00)
                    | (sc[3] & 0xF000);
                let d = half::f16::from_bits(d_bits).to_f32();
                let x_base = b * QK_K;
                let mut x_off = 0usize;
                for ib in 0..8 {
                    let s_word = sc[ib / 2];
                    let shift0 = 6 * (ib % 2);
                    let shift1 = 6 * (ib % 2) + 3;
                    let dl1 = d * (2.0 * ((s_word >> shift0) & 0x7) as f32 + 1.0);
                    let dl2 = d * (2.0 * ((s_word >> shift1) & 0x7) as f32 + 1.0);
                    let qh0 = w_bytes[qh_off + ib * 2];
                    let qh1 = w_bytes[qh_off + ib * 2 + 1];
                    let delta_l = [
                        if qh0 & 0x08 != 0 { -1.0 - IQ1S_DELTA } else { -1.0 + IQ1S_DELTA },
                        if qh0 & 0x80 != 0 { -1.0 - IQ1S_DELTA } else { -1.0 + IQ1S_DELTA },
                        if qh1 & 0x08 != 0 { -1.0 - IQ1S_DELTA } else { -1.0 + IQ1S_DELTA },
                        if qh1 & 0x80 != 0 { -1.0 - IQ1S_DELTA } else { -1.0 + IQ1S_DELTA },
                    ];
                    let idx_l = [
                        w_bytes[qs_off + ib * 4] as usize | (((qh0 & 0x07) as usize) << 8),
                        w_bytes[qs_off + ib * 4 + 1] as usize | ((((qh0 >> 4) & 0x07) as usize) << 8),
                        w_bytes[qs_off + ib * 4 + 2] as usize | (((qh1 & 0x07) as usize) << 8),
                        w_bytes[qs_off + ib * 4 + 3] as usize | ((((qh1 >> 4) & 0x07) as usize) << 8),
                    ];
                    let dl_l = [dl1, dl1, dl2, dl2];
                    for l in 0..4 {
                        let grid_bits = IQ1S_GRID[idx_l[l]];
                        let dl = dl_l[l];
                        let delta_val = delta_l[l];
                        for j in 0..8 {
                            let gi = ((grid_bits >> (j * 8)) & 0xFF) as i8 as f32;
                            acc += dl * (gi + delta_val) * x[x_base + x_off + 8 * l + j];
                        }
                    }
                    x_off += 32;
                }
            }
            out[mi] = acc;
        }
    }

    fn synth_iq1_m_bytes(m: usize, k: usize, seed: u32) -> Vec<u8> {
        let blocks = m * (k / 256);
        let mut bytes = vec![0u8; blocks * 56];
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
        for chunk in bytes.chunks_mut(56) {
            // qs[0..32] + qh[32..48]: random bytes.
            for b in chunk[..48].iter_mut() {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                *b = (s >> 24) as u8;
            }
            // scales[48..56]: random low 12 bits per u16 (these
            // carry the dl1/dl2 sub-block scales — exercising both
            // SYCL and CPU paths). Top nibble of each u16 packs
            // one nibble of the f16 super-block scale `d`. We
            // pick a sensible d in [-0.5, 0.5] and back-encode its
            // 16 bits across the four scale words so the d_bits
            // reassembly produces a finite, non-NaN value.
            for b in chunk[48..56].iter_mut() {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                *b = (s >> 24) as u8;
            }
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let target_d = ((s >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 1.0;
            let d_bits = half::f16::from_f32(target_d).to_bits();
            for ii in 0..4 {
                let nib = ((d_bits >> (4 * ii)) & 0xF) as u8;
                // Clear the top nibble of sc[ii] (= chunk[48+2*ii+1]'s
                // top 4 bits) and rewrite with the target nibble.
                let hi_byte = &mut chunk[48 + 2 * ii + 1];
                *hi_byte = (*hi_byte & 0x0F) | (nib << 4);
            }
        }
        bytes
    }

    #[test]
    fn iq1_m_sycl_arithmetic_matches_cpu_reference() {
        for &(m, k) in &[(1usize, 256), (3, 512), (5, 1024)] {
            let w = synth_iq1_m_bytes(m, k, m as u32 * 41 + k as u32);
            let x = rand_vec(k, (m * k + 17) as u32);
            let mut out_sycl = vec![0f32; m];
            let mut out_ref = vec![0f32; m];
            iq1_m_sycl_port_scalar(&w, &x, &mut out_sycl, m, k);
            matvec_iq1_m_w_f32_a_scalar(&w, &x, &mut out_ref, m, k);
            for (i, (&a, &b)) in out_sycl.iter().zip(out_ref.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-4,
                    "iq1_m SYCL-port parity (m={m}, k={k}) idx={i}: sycl={a} ref={b}"
                );
            }
        }
    }

    /// Rust port of the IQ2_XS SYCL kernel arithmetic. Mirrors
    /// `matvec_iq2_xs_packed_f32_usm_impl` in `rsl_kernels.cpp`
    /// byte-for-byte (same per-u16 word load order, same 9-bit
    /// grid index extraction + 7-bit sign-table index, same
    /// db_lo / db_hi nibble unpack, same per-lane FMA).
    fn iq2_xs_sycl_port_scalar(
        w_bytes: &[u8],
        x: &[f32],
        out: &mut [f32],
        m: usize,
        k: usize,
    ) {
        use rustllama_gguf::dequant::{IQ2XS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
        const BLOCK_BYTES: usize = 74;
        const QK_K: usize = 256;
        let blocks_per_row = k / QK_K;
        let bytes_per_row = blocks_per_row * BLOCK_BYTES;
        for mi in 0..m {
            let row_start = mi * bytes_per_row;
            let mut acc = 0.0f32;
            for b in 0..blocks_per_row {
                let off = row_start + b * BLOCK_BYTES;
                let d_bits = (w_bytes[off] as u16) | ((w_bytes[off + 1] as u16) << 8);
                let d = half::f16::from_bits(d_bits).to_f32();
                let qs_off = off + 2;
                let scales_off = off + 2 + 64;
                let x_base = b * QK_K;
                for ib32 in 0..8 {
                    let scale_byte = w_bytes[scales_off + ib32];
                    let db_lo = d * (0.5 + (scale_byte & 0x0F) as f32) * 0.25;
                    let db_hi = d * (0.5 + (scale_byte >> 4) as f32) * 0.25;
                    let base = 8 * ib32;
                    for l in 0..4 {
                        let q = (w_bytes[qs_off + base + 2 * l] as u16)
                            | ((w_bytes[qs_off + base + 2 * l + 1] as u16) << 8);
                        let grid_idx = (q & 0x1FF) as usize;
                        let sign_idx = (q >> 9) as usize;
                        let grid_bits = IQ2XS_GRID[grid_idx];
                        let signs = KSIGNS_IQ2XS[sign_idx];
                        let db = if l < 2 { db_lo } else { db_hi };
                        let x_off = x_base + ib32 * 32 + l * 8;
                        for j in 0..8 {
                            let gi = ((grid_bits >> (j * 8)) & 0xFF) as u8;
                            let s = if signs & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                            acc += db * (gi as f32) * s * x[x_off + j];
                        }
                    }
                }
            }
            out[mi] = acc;
        }
    }

    fn synth_iq2_xs_bytes(m: usize, k: usize, seed: u32) -> Vec<u8> {
        let blocks = m * (k / 256);
        let mut bytes = vec![0u8; blocks * 74];
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
        for chunk in bytes.chunks_mut(74) {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let scale_f32 = ((s >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 1.0;
            let d_bits = half::f16::from_f32(scale_f32).to_bits();
            chunk[0] = (d_bits & 0xFF) as u8;
            chunk[1] = ((d_bits >> 8) & 0xFF) as u8;
            // qs[2..66] (32 × u16) — grid_idx (9 bits) into a
            // 512-entry codebook + 7-bit sign-table index. The
            // sign-table is full-range so any pattern is valid.
            // The grid index must be < 512; clamp by masking the
            // high bit of each u16's second byte's high bit.
            for chunk_pair in chunk[2..66].chunks_mut(2) {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let val = (s >> 16) as u16;
                // 9-bit grid index in low 9 bits; ensure < 512 by
                // construction (mask top 7 bits = sign-table idx,
                // which is already random in [0, 127] range).
                chunk_pair[0] = (val & 0xFF) as u8;
                chunk_pair[1] = ((val >> 8) & 0xFF) as u8;
            }
            // scales[66..74]: random — each byte is two 4-bit
            // sub-block scales (full range 0..16 is valid).
            for b in chunk[66..74].iter_mut() {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                *b = (s >> 24) as u8;
            }
        }
        bytes
    }

    #[test]
    fn iq2_xs_sycl_arithmetic_matches_cpu_reference() {
        for &(m, k) in &[(1usize, 256), (3, 512), (5, 1024)] {
            let w = synth_iq2_xs_bytes(m, k, m as u32 * 43 + k as u32);
            let x = rand_vec(k, (m * k + 19) as u32);
            let mut out_sycl = vec![0f32; m];
            let mut out_ref = vec![0f32; m];
            iq2_xs_sycl_port_scalar(&w, &x, &mut out_sycl, m, k);
            matvec_iq2_xs_w_f32_a_scalar(&w, &x, &mut out_ref, m, k);
            for (i, (&a, &b)) in out_sycl.iter().zip(out_ref.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-4,
                    "iq2_xs SYCL-port parity (m={m}, k={k}) idx={i}: sycl={a} ref={b}"
                );
            }
        }
    }

    /// Rust port of the IQ2_S SYCL kernel arithmetic. Mirrors
    /// `matvec_iq2_s_packed_f32_usm_impl` in `rsl_kernels.cpp`
    /// byte-for-byte (same qs_lo/signs/qh/scales split, same
    /// 10-bit grid index reconstruction from qs_lo + qh, same
    /// inline sign-mask, same db_lo/db_hi nibble unpack).
    fn iq2_s_sycl_port_scalar(
        w_bytes: &[u8],
        x: &[f32],
        out: &mut [f32],
        m: usize,
        k: usize,
    ) {
        use rustllama_gguf::dequant::{IQ2S_GRID, KMASK_IQ2XS};
        const BLOCK_BYTES: usize = 82;
        const QK_K: usize = 256;
        let blocks_per_row = k / QK_K;
        let bytes_per_row = blocks_per_row * BLOCK_BYTES;
        for mi in 0..m {
            let row_start = mi * bytes_per_row;
            let mut acc = 0.0f32;
            for b in 0..blocks_per_row {
                let off = row_start + b * BLOCK_BYTES;
                let d_bits = (w_bytes[off] as u16) | ((w_bytes[off + 1] as u16) << 8);
                let d = half::f16::from_bits(d_bits).to_f32();
                let qs_lo_off = off + 2;
                let signs_off = off + 2 + 32;
                let qh_off = off + 2 + 64;
                let scales_off = off + 2 + 64 + 8;
                let x_base = b * QK_K;
                for ib32 in 0..8 {
                    let scale_byte = w_bytes[scales_off + ib32];
                    let db_lo = d * (0.5 + (scale_byte & 0x0F) as f32) * 0.25;
                    let db_hi = d * (0.5 + (scale_byte >> 4) as f32) * 0.25;
                    let qs_off = ib32 * 4;
                    let qh_byte = w_bytes[qh_off + ib32];
                    for l in 0..4 {
                        let high_bits = ((qh_byte as usize) << (8 - 2 * l)) & 0x300;
                        let grid_idx = (w_bytes[qs_lo_off + qs_off + l] as usize) | high_bits;
                        let sign_byte = w_bytes[signs_off + qs_off + l];
                        let grid_bits = IQ2S_GRID[grid_idx];
                        let db = if l < 2 { db_lo } else { db_hi };
                        let x_off = x_base + ib32 * 32 + l * 8;
                        for j in 0..8 {
                            let gi = ((grid_bits >> (j * 8)) & 0xFF) as u8;
                            let s = if sign_byte & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                            acc += db * (gi as f32) * s * x[x_off + j];
                        }
                    }
                }
            }
            out[mi] = acc;
        }
    }

    fn synth_iq2_s_bytes(m: usize, k: usize, seed: u32) -> Vec<u8> {
        let blocks = m * (k / 256);
        let mut bytes = vec![0u8; blocks * 82];
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
        for chunk in bytes.chunks_mut(82) {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let scale_f32 = ((s >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 1.0;
            let d_bits = half::f16::from_f32(scale_f32).to_bits();
            chunk[0] = (d_bits & 0xFF) as u8;
            chunk[1] = ((d_bits >> 8) & 0xFF) as u8;
            // qs_lo + signs + qh + scales: all random bytes. The
            // 10-bit grid index is always < 1024 since the high
            // 2 bits come from qh, which is naturally u8 ∈ [0,255]
            // and we shift bit-pairs into positions 8-9, both
            // ≤ 3 → grid_idx ≤ 0x3FF = 1023. Always in-range.
            for b in chunk[2..82].iter_mut() {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                *b = (s >> 24) as u8;
            }
        }
        bytes
    }

    #[test]
    fn iq2_s_sycl_arithmetic_matches_cpu_reference() {
        for &(m, k) in &[(1usize, 256), (3, 512), (5, 1024)] {
            let w = synth_iq2_s_bytes(m, k, m as u32 * 47 + k as u32);
            let x = rand_vec(k, (m * k + 23) as u32);
            let mut out_sycl = vec![0f32; m];
            let mut out_ref = vec![0f32; m];
            iq2_s_sycl_port_scalar(&w, &x, &mut out_sycl, m, k);
            matvec_iq2_s_w_f32_a_scalar(&w, &x, &mut out_ref, m, k);
            for (i, (&a, &b)) in out_sycl.iter().zip(out_ref.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-4,
                    "iq2_s SYCL-port parity (m={m}, k={k}) idx={i}: sycl={a} ref={b}"
                );
            }
        }
    }

    /// Rust port of the IQ3_XXS SYCL kernel arithmetic. Mirrors
    /// `matvec_iq3_xxs_packed_f32_usm_impl` in `rsl_kernels.cpp`
    /// byte-for-byte (same qs_grid/qs_sas split, same sub-block
    /// scale extraction `db = d * (0.5 + (aux32 >> 28)) * 0.5`,
    /// same dual-grid lookup per 8-weight chunk, same KSIGNS
    /// nibble-split for low/high halves).
    fn iq3_xxs_sycl_port_scalar(
        w_bytes: &[u8],
        x: &[f32],
        out: &mut [f32],
        m: usize,
        k: usize,
    ) {
        use rustllama_gguf::dequant::{IQ3XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
        const BLOCK_BYTES: usize = 98;
        const QK_K: usize = 256;
        let blocks_per_row = k / QK_K;
        let bytes_per_row = blocks_per_row * BLOCK_BYTES;
        for mi in 0..m {
            let row_start = mi * bytes_per_row;
            let mut acc = 0.0f32;
            for b in 0..blocks_per_row {
                let off = row_start + b * BLOCK_BYTES;
                let d_bits = (w_bytes[off] as u16) | ((w_bytes[off + 1] as u16) << 8);
                let d = half::f16::from_bits(d_bits).to_f32();
                let qs_grid_off = off + 2;
                let qs_sas_off = off + 2 + 64;
                let x_base = b * QK_K;
                for ib32 in 0..8 {
                    let sas = qs_sas_off + 4 * ib32;
                    let aux32 = (w_bytes[sas] as u32)
                        | ((w_bytes[sas + 1] as u32) << 8)
                        | ((w_bytes[sas + 2] as u32) << 16)
                        | ((w_bytes[sas + 3] as u32) << 24);
                    let db = d * (0.5 + (aux32 >> 28) as f32) * 0.5;
                    let qs_off = 8 * ib32;
                    for l in 0..4 {
                        let g1_idx = w_bytes[qs_grid_off + qs_off + 2 * l] as usize;
                        let g2_idx = w_bytes[qs_grid_off + qs_off + 2 * l + 1] as usize;
                        let grid1_bits = IQ3XXS_GRID[g1_idx];
                        let grid2_bits = IQ3XXS_GRID[g2_idx];
                        let signs = KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize];
                        let x_off = x_base + ib32 * 32 + l * 8;
                        for j in 0..4 {
                            let g1 = ((grid1_bits >> (j * 8)) & 0xFF) as u8;
                            let g2 = ((grid2_bits >> (j * 8)) & 0xFF) as u8;
                            let s_lo = if signs & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                            let s_hi = if signs & KMASK_IQ2XS[j + 4] != 0 { -1.0 } else { 1.0 };
                            acc += db * (g1 as f32) * s_lo * x[x_off + j];
                            acc += db * (g2 as f32) * s_hi * x[x_off + j + 4];
                        }
                    }
                }
            }
            out[mi] = acc;
        }
    }

    fn synth_iq3_xxs_bytes(m: usize, k: usize, seed: u32) -> Vec<u8> {
        let blocks = m * (k / 256);
        let mut bytes = vec![0u8; blocks * 98];
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
        for chunk in bytes.chunks_mut(98) {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let scale_f32 = ((s >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 1.0;
            let d_bits = half::f16::from_f32(scale_f32).to_bits();
            chunk[0] = (d_bits & 0xFF) as u8;
            chunk[1] = ((d_bits >> 8) & 0xFF) as u8;
            // qs_grid + qs_sas: random bytes. grid indices are
            // u8 ∈ [0,255], all in-range for the 256-entry codebook.
            // KSIGNS indices are 7-bit so any u32 byte pattern
            // also lands in-range.
            for b in chunk[2..98].iter_mut() {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                *b = (s >> 24) as u8;
            }
        }
        bytes
    }

    #[test]
    fn iq3_xxs_sycl_arithmetic_matches_cpu_reference() {
        for &(m, k) in &[(1usize, 256), (3, 512), (5, 1024)] {
            let w = synth_iq3_xxs_bytes(m, k, m as u32 * 53 + k as u32);
            let x = rand_vec(k, (m * k + 29) as u32);
            let mut out_sycl = vec![0f32; m];
            let mut out_ref = vec![0f32; m];
            iq3_xxs_sycl_port_scalar(&w, &x, &mut out_sycl, m, k);
            matvec_iq3_xxs_w_f32_a_scalar(&w, &x, &mut out_ref, m, k);
            for (i, (&a, &b)) in out_sycl.iter().zip(out_ref.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-4,
                    "iq3_xxs SYCL-port parity (m={m}, k={k}) idx={i}: sycl={a} ref={b}"
                );
            }
        }
    }

    /// Rust port of the IQ3_S SYCL kernel arithmetic. Mirrors
    /// `matvec_iq3_s_packed_f32_usm_impl` in `rsl_kernels.cpp`
    /// byte-for-byte. Uses the flat per-ib32 loop form (pair =
    /// ib32 >> 1) rather than the CPU reference's nested pair-of-
    /// ib32 walker — they produce the same per-chunk operations
    /// in the same FMA order because the byte offsets line up.
    fn iq3_s_sycl_port_scalar(
        w_bytes: &[u8],
        x: &[f32],
        out: &mut [f32],
        m: usize,
        k: usize,
    ) {
        use rustllama_gguf::dequant::{IQ3S_GRID, KMASK_IQ2XS};
        const BLOCK_BYTES: usize = 110;
        const QK_K: usize = 256;
        let blocks_per_row = k / QK_K;
        let bytes_per_row = blocks_per_row * BLOCK_BYTES;
        for mi in 0..m {
            let row_start = mi * bytes_per_row;
            let mut acc = 0.0f32;
            for b in 0..blocks_per_row {
                let off = row_start + b * BLOCK_BYTES;
                let d_bits = (w_bytes[off] as u16) | ((w_bytes[off + 1] as u16) << 8);
                let d = half::f16::from_bits(d_bits).to_f32();
                let qs_off = off + 2;
                let qh_off = off + 2 + 64;
                let signs_off = off + 2 + 64 + 8;
                let scales_off = off + 2 + 64 + 8 + 32;
                let x_base = b * QK_K;
                for ib32 in 0..8 {
                    let pair = ib32 >> 1;
                    let scale_byte = w_bytes[scales_off + pair];
                    let db = if ib32 & 1 != 0 {
                        d * (1.0 + 2.0 * (scale_byte >> 4) as f32)
                    } else {
                        d * (1.0 + 2.0 * (scale_byte & 0x0F) as f32)
                    };
                    let qsoff_local = ib32 * 8;
                    let signs_local = ib32 * 4;
                    let qh_byte = w_bytes[qh_off + ib32];
                    let x_off_block = ib32 * 32;
                    for l in 0..4 {
                        let g1_idx = w_bytes[qs_off + qsoff_local + 2 * l] as usize
                            | (((qh_byte as usize) << (8 - 2 * l)) & 0x100);
                        let g2_idx = w_bytes[qs_off + qsoff_local + 2 * l + 1] as usize
                            | (((qh_byte as usize) << (7 - 2 * l)) & 0x100);
                        let grid1_bits = IQ3S_GRID[g1_idx];
                        let grid2_bits = IQ3S_GRID[g2_idx];
                        let sign_byte = w_bytes[signs_off + signs_local + l];
                        let x_off = x_base + x_off_block + l * 8;
                        for j in 0..4 {
                            let g1 = ((grid1_bits >> (j * 8)) & 0xFF) as u8;
                            let g2 = ((grid2_bits >> (j * 8)) & 0xFF) as u8;
                            let s_lo = if sign_byte & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                            let s_hi = if sign_byte & KMASK_IQ2XS[j + 4] != 0 { -1.0 } else { 1.0 };
                            acc += db * (g1 as f32) * s_lo * x[x_off + j];
                            acc += db * (g2 as f32) * s_hi * x[x_off + j + 4];
                        }
                    }
                }
            }
            out[mi] = acc;
        }
    }

    fn synth_iq3_s_bytes(m: usize, k: usize, seed: u32) -> Vec<u8> {
        let blocks = m * (k / 256);
        let mut bytes = vec![0u8; blocks * 110];
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
        for chunk in bytes.chunks_mut(110) {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let scale_f32 = ((s >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 1.0;
            let d_bits = half::f16::from_f32(scale_f32).to_bits();
            chunk[0] = (d_bits & 0xFF) as u8;
            chunk[1] = ((d_bits >> 8) & 0xFF) as u8;
            // qs + qh + signs + scales: random bytes. The 9-bit
            // grid index is always < 512: low 8 bits are u8;
            // high 1 bit comes from qh shifts that produce 0x100
            // at most → grid_idx ≤ 0x1FF = 511. Always in-range.
            for b in chunk[2..110].iter_mut() {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                *b = (s >> 24) as u8;
            }
        }
        bytes
    }

    #[test]
    fn iq3_s_sycl_arithmetic_matches_cpu_reference() {
        for &(m, k) in &[(1usize, 256), (3, 512), (5, 1024)] {
            let w = synth_iq3_s_bytes(m, k, m as u32 * 59 + k as u32);
            let x = rand_vec(k, (m * k + 31) as u32);
            let mut out_sycl = vec![0f32; m];
            let mut out_ref = vec![0f32; m];
            iq3_s_sycl_port_scalar(&w, &x, &mut out_sycl, m, k);
            matvec_iq3_s_w_f32_a_scalar(&w, &x, &mut out_ref, m, k);
            for (i, (&a, &b)) in out_sycl.iter().zip(out_ref.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-4,
                    "iq3_s SYCL-port parity (m={m}, k={k}) idx={i}: sycl={a} ref={b}"
                );
            }
        }
    }

    /// F4 follow-up: Rust port of the **batched** SYCL IQ4_NL kernel
    /// (`matvec_iq4_nl_packed_f32_batched_usm_impl`). The per-cell
    /// arithmetic is identical to the single-row port; the batched
    /// kernel adds an outer N dimension reading `x_usm + n*K` and
    /// writing `out_usm[n*M + m]`. This port is what the SYCL kernel
    /// will execute on a device when SPIR-V codegen lands.
    fn iq4_nl_sycl_port_scalar_batched(
        w_bytes: &[u8],
        x: &[f32],
        out: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) {
        for ni in 0..n {
            let x_row = &x[ni * k..(ni + 1) * k];
            let mut row_out = vec![0f32; m];
            iq4_nl_sycl_port_scalar(w_bytes, x_row, &mut row_out, m, k);
            for mi in 0..m {
                out[ni * m + mi] = row_out[mi];
            }
        }
    }

    /// F4 follow-up: Rust port of the **batched** SYCL IQ4_XS kernel
    /// (`matvec_iq4_xs_packed_f32_batched_usm_impl`). Same structure
    /// as `iq4_nl_sycl_port_scalar_batched`.
    fn iq4_xs_sycl_port_scalar_batched(
        w_bytes: &[u8],
        x: &[f32],
        out: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) {
        for ni in 0..n {
            let x_row = &x[ni * k..(ni + 1) * k];
            let mut row_out = vec![0f32; m];
            iq4_xs_sycl_port_scalar(w_bytes, x_row, &mut row_out, m, k);
            for mi in 0..m {
                out[ni * m + mi] = row_out[mi];
            }
        }
    }

    #[test]
    fn iq4_nl_batched_sycl_arithmetic_matches_cpu_reference() {
        // (M=output rows, K=K-dim, N=batch rows).
        for &(m, k, n) in &[(4usize, 32, 2), (8, 128, 4), (5, 1024, 3)] {
            let w = synth_iq4_nl_bytes(m, k, m as u32 * 17 + k as u32 + n as u32);
            let x = rand_vec(n * k, (m * k * n) as u32);
            let mut out_sycl = vec![0f32; n * m];
            let mut out_ref = vec![0f32; n * m];
            iq4_nl_sycl_port_scalar_batched(&w, &x, &mut out_sycl, m, k, n);
            for ni in 0..n {
                let x_row = &x[ni * k..(ni + 1) * k];
                let mut row_ref = vec![0f32; m];
                matvec_iq4_nl_w_f32_a_scalar(&w, x_row, &mut row_ref, m, k);
                for mi in 0..m {
                    out_ref[ni * m + mi] = row_ref[mi];
                }
            }
            for (i, (&a, &b)) in out_sycl.iter().zip(out_ref.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-4,
                    "iq4_nl batched SYCL-port parity (m={m}, k={k}, n={n}) idx={i}: sycl={a} ref={b}"
                );
            }
        }
    }

    #[test]
    fn iq4_xs_batched_sycl_arithmetic_matches_cpu_reference() {
        for &(m, k, n) in &[(3usize, 256, 2), (5, 512, 3), (4, 1024, 4)] {
            let w = synth_iq4_xs_bytes(m, k, m as u32 * 19 + k as u32 + n as u32);
            let x = rand_vec(n * k, (m * k * n) as u32);
            let mut out_sycl = vec![0f32; n * m];
            let mut out_ref = vec![0f32; n * m];
            iq4_xs_sycl_port_scalar_batched(&w, &x, &mut out_sycl, m, k, n);
            for ni in 0..n {
                let x_row = &x[ni * k..(ni + 1) * k];
                let mut row_ref = vec![0f32; m];
                matvec_iq4_xs_w_f32_a_scalar(&w, x_row, &mut row_ref, m, k);
                for mi in 0..m {
                    out_ref[ni * m + mi] = row_ref[mi];
                }
            }
            for (i, (&a, &b)) in out_sycl.iter().zip(out_ref.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-4,
                    "iq4_xs batched SYCL-port parity (m={m}, k={k}, n={n}) idx={i}: sycl={a} ref={b}"
                );
            }
        }
    }

    /// F3: SIMD F32 → F16-bits packing must produce byte-identical
    /// output to the scalar `half::f16::from_f32(v).to_bits()` loop.
    /// Round-to-nearest-even semantics match because the F16C
    /// instruction uses the same rounding mode (imm8 = 0) as the
    /// half crate's scalar conversion.
    #[test]
    fn f32_to_f16_bits_simd_matches_scalar() {
        for &n in &[0usize, 1, 7, 8, 15, 16, 17, 31, 32, 64, 4097] {
            let input = rand_vec(n, n as u32 * 11 + 3);
            let mut a = vec![0u16; n];
            f32_to_f16_bits(&input, &mut a);
            let mut b = vec![0u16; n];
            f32_to_f16_bits_scalar(&input, &mut b);
            for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
                assert_eq!(
                    av, bv,
                    "f32_to_f16_bits SIMD parity at n={n} idx={i}: simd={av:#06x} scalar={bv:#06x}"
                );
            }
        }
    }

    /// F1: `fused_temp_softmax_inplace(x, 1/T)` must equal
    /// `scale_by(x, 1/T)` then `softmax_f32_inplace_scalar(x)`. The
    /// fusion folds the temperature multiply into pass 1; the math
    /// is invariant.
    #[test]
    fn fused_temp_softmax_matches_scaled_then_scalar_softmax() {
        for &n in &[1usize, 7, 8, 17, 32, 128, 1024, 4097] {
            for &t in &[0.5_f32, 0.7, 1.0, 1.5, 2.0] {
                let input = rand_vec(n, n as u32 * 13 + t.to_bits());
                let inv_t = 1.0 / t;
                let mut a = input.clone();
                fused_temp_softmax_inplace(&mut a, inv_t);
                let mut b = input.clone();
                for v in b.iter_mut() {
                    *v *= inv_t;
                }
                softmax_f32_inplace_scalar(&mut b);
                for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
                    assert!(
                        (av - bv).abs() < 1e-5,
                        "fused_temp_softmax parity at n={n} t={t} idx={i}: fused={av} ref={bv}"
                    );
                }
                let sum: f32 = a.iter().sum();
                assert!(
                    (sum - 1.0).abs() < 1e-5,
                    "fused_temp_softmax output for n={n} t={t} did not sum to 1.0: {sum}"
                );
            }
        }
    }

    #[test]
    fn rmsnorm_simd_parity_with_scalar() {
        for &d in &[1usize, 7, 8, 15, 16, 17, 64, 128, 4096, 4097] {
            let x = rand_vec(d, d as u32);
            let w = rand_vec(d, (d * 7 + 1) as u32);
            let mut a = vec![0f32; d];
            rmsnorm_f32_row(&x, &w, &mut a, 1e-5);
            let mut b = vec![0f32; d];
            rmsnorm_f32_row_scalar(&x, &w, &mut b, 1e-5);
            for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
                assert!(
                    (av - bv).abs() < 1e-4,
                    "rmsnorm SIMD parity at d={d} idx={i}: simd={av} scalar={bv}"
                );
            }
        }
    }

    #[test]
    fn rope_simd_parity_with_scalar() {
        for &(n_heads, head_dim) in
            &[(1usize, 64usize), (4, 64), (8, 128), (32, 128), (1, 32), (2, 256)]
        {
            let x = rand_vec(n_heads * head_dim, (n_heads * head_dim) as u32);
            let mut a = x.clone();
            rope_inplace_neox(&mut a, n_heads, head_dim, 17, 10000.0);
            // Run the scalar variant via the public path by
            // temporarily disabling SIMD: simulate by directly
            // calling the scalar core with a hand-built cs slab.
            let half = head_dim / 2;
            let mut cs = vec![0f32; 2 * half];
            for i in 0..half {
                let freq = 1.0 / (10000f32).powf(2.0 * i as f32 / head_dim as f32);
                let p = 17.0 * freq;
                cs[2 * i] = p.cos();
                cs[2 * i + 1] = p.sin();
            }
            let mut b = x.clone();
            rope_inplace_neox_scalar(&mut b, n_heads, head_dim, half, &cs);
            for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
                assert!(
                    (av - bv).abs() < 1e-4,
                    "rope SIMD parity at ({n_heads},{head_dim}) idx={i}: \
                     simd={av} scalar={bv}"
                );
            }
        }
    }

    #[test]
    fn log_sum_exp_simd_parity_with_scalar() {
        // Same input sizes as softmax_simd_parity. The SIMD path
        // shares `expf_approx_*` with softmax, so its relative
        // error budget is the same; for log_sum_exp the absolute
        // error per call is dominated by the final `.ln()` which
        // happens scalar — we just verify the SIMD output stays
        // within 1e-4 abs of the scalar reference.
        for &n in &[1usize, 7, 8, 15, 16, 17, 31, 32, 33, 127, 128, 4097] {
            let input = rand_vec(n, n as u32 * 11 + 7);
            let got = log_sum_exp_f32(&input);
            let want = log_sum_exp_f32_scalar(&input);
            assert!(
                (got - want).abs() < 1e-4,
                "log_sum_exp SIMD parity at n={n}: simd={got} scalar={want}"
            );
        }
        // Empty input returns NEG_INFINITY (matches scalar).
        assert!(log_sum_exp_f32(&[]).is_infinite());
    }

    #[test]
    fn silu_mul_simd_parity_with_scalar() {
        for &n in &[1usize, 7, 8, 17, 64, 4096, 11008] {
            let x = rand_vec(n, n as u32);
            let y = rand_vec(n, (n * 13 + 1) as u32);
            let mut a = vec![0f32; n];
            silu_mul_f32(&x, &y, &mut a);
            let mut b = vec![0f32; n];
            silu_mul_f32_scalar(&x, &y, &mut b);
            for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
                // SiLU has higher tolerance than softmax because the
                // surrounding `x / (1 + exp(-x))` amplifies the
                // expf approximation's relative error. 1e-3 abs
                // covers the worst case across the test range.
                assert!(
                    (av - bv).abs() < 1e-3,
                    "silu_mul SIMD parity at n={n} idx={i}: simd={av} scalar={bv}"
                );
            }
        }
    }

    // ============================================================
    // AArch64 NEON parity tests
    // ============================================================
    //
    // These run only when the crate is BUILT for aarch64 (natively or
    // cross-compiled + executed under qemu-user). Each compares the NEON
    // kernel against the pure-scalar reference on identical random inputs,
    // the ARM twin of the x86 `*_simd_paths_match_scalar` tests. Tolerance
    // matches the x86 parity tests (the vector path folds scales per block
    // and sums in a different order, so it is close-not-bit-identical).
    #[cfg(target_arch = "aarch64")]
    mod neon_parity {
        use super::*;

        /// Small LCG → a reproducible byte/float stream (no `rand` dep).
        struct Lcg(u32);
        impl Lcg {
            fn next_u32(&mut self) -> u32 {
                self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
                self.0
            }
            fn byte(&mut self) -> u8 {
                (self.next_u32() >> 24) as u8
            }
            /// A "sane" f16 scale/min (small magnitude, finite) as LE bytes.
            fn f16_le(&mut self, spread: f32, offset: f32) -> [u8; 2] {
                let u = (self.next_u32() >> 16) as f32 / 65536.0;
                f16::from_f32(u * spread + offset).to_le_bytes()
            }
            /// A "sane" f32 scale (small magnitude, finite) as LE bytes.
            fn f32_le(&mut self, spread: f32, offset: f32) -> [u8; 4] {
                let u = (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32;
                (u * spread + offset).to_le_bytes()
            }
            fn activation(&mut self) -> f32 {
                (self.next_u32() as i32 as f32) / (i32::MAX as f32)
            }
        }

        fn assert_close(scalar: &[f32], neon: &[f32], tag: &str) {
            for (i, (a, b)) in scalar.iter().zip(neon.iter()).enumerate() {
                let abs = (a - b).abs();
                let rel = abs / a.abs().max(1e-6);
                assert!(
                    abs < 1e-3 || rel < 1e-5,
                    "{tag} row {i}: scalar={a} neon={b} abs={abs} rel={rel}"
                );
            }
        }

        // ---- Forward-pass NEON ports (log_sum_exp / rope / GQA attention) ----

        #[test]
        fn log_sum_exp_neon_matches_scalar() {
            let mut l = Lcg(0x1053_0001);
            for n in [1usize, 3, 4, 7, 16, 33, 128, 1000] {
                let x: Vec<f32> = (0..n).map(|_| l.activation() * 8.0).collect();
                let s = log_sum_exp_f32_scalar(&x);
                let v = unsafe { log_sum_exp_f32_neon(&x) };
                let abs = (s - v).abs();
                let rel = abs / s.abs().max(1e-6);
                assert!(
                    abs < 1e-3 || rel < 1e-5,
                    "log_sum_exp n={n}: scalar={s} neon={v} abs={abs} rel={rel}"
                );
            }
        }

        #[test]
        fn rope_neox_neon_matches_scalar() {
            let mut l = Lcg(0x8042_0001);
            let n_heads = 6usize;
            // head_dim 70 → half 35 (exercises the 4-wide body + scalar tail).
            let head_dim = 70usize;
            let half = head_dim / 2;
            let theta = 10000.0f32;
            let pos = 37u32;
            let cs: Vec<f32> = (0..half)
                .flat_map(|i| {
                    let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
                    let p = pos as f32 * freq;
                    [p.cos(), p.sin()]
                })
                .collect();
            let base: Vec<f32> = (0..n_heads * head_dim).map(|_| l.activation()).collect();
            let mut xs = base.clone();
            let mut xn = base.clone();
            rope_inplace_neox_scalar(&mut xs, n_heads, head_dim, half, &cs);
            unsafe { rope_inplace_neox_neon(&mut xn, n_heads, head_dim, half, &cs) };
            assert_close(&xs, &xn, "rope_neox");
        }

        #[test]
        fn gqa_one_step_neon_matches_scalar() {
            let mut l = Lcg(0x6911_0001);
            let (n_heads, n_kv_heads, head_dim, max_ctx, kv_len) = (8usize, 2, 48, 32, 19);
            let q: Vec<f32> = (0..n_heads * head_dim).map(|_| l.activation()).collect();
            let k: Vec<f32> = (0..n_kv_heads * max_ctx * head_dim)
                .map(|_| l.activation())
                .collect();
            let v: Vec<f32> = (0..n_kv_heads * max_ctx * head_dim)
                .map(|_| l.activation())
                .collect();
            let mut os = vec![0f32; n_heads * head_dim];
            let mut on = vec![0f32; n_heads * head_dim];
            gqa_attention_one_step_scalar(
                &q, &k, &v, &mut os, n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
            );
            unsafe {
                gqa_attention_one_step_neon(
                    &q, &k, &v, &mut on, n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
                )
            };
            assert_close(&os, &on, "gqa_one_step");
        }

        #[test]
        fn gqa_flash_decode_neon_matches_scalar() {
            let mut l = Lcg(0xF1A5_0001);
            let (n_heads, n_kv_heads, head_dim, max_ctx, kv_len) = (8usize, 4, 64, 40, 23);
            let q: Vec<f32> = (0..n_heads * head_dim).map(|_| l.activation()).collect();
            let k: Vec<f32> = (0..n_kv_heads * max_ctx * head_dim)
                .map(|_| l.activation())
                .collect();
            let v: Vec<f32> = (0..n_kv_heads * max_ctx * head_dim)
                .map(|_| l.activation())
                .collect();
            let mut os = vec![0f32; n_heads * head_dim];
            let mut on = vec![0f32; n_heads * head_dim];
            gqa_attention_flash_decode_scalar(
                &q, &k, &v, &mut os, n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
            );
            unsafe {
                gqa_attention_flash_decode_neon(
                    &q, &k, &v, &mut on, n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
                )
            };
            assert_close(&os, &on, "gqa_flash_decode");
        }

        fn run_block_quant<const BB: usize>(
            tag: &str,
            m: usize,
            k: usize,
            seed: u32,
            // fills one block's bytes given (lcg, &mut block_slice)
            fill: impl Fn(&mut Lcg, &mut [u8]),
            scalar: unsafe fn(&[u8], &[f32], &mut [f32], usize, usize),
            neon: unsafe fn(&[u8], &[f32], &mut [f32], usize, usize),
            qk: usize,
        ) {
            let blocks_per_row = k / qk;
            let total_blocks = m * blocks_per_row;
            let mut lcg = Lcg(seed);
            let mut w = vec![0u8; total_blocks * BB];
            for blk in w.chunks_exact_mut(BB) {
                fill(&mut lcg, blk);
            }
            let mut x = vec![0f32; k];
            for v in x.iter_mut() {
                *v = lcg.activation();
            }
            let mut out_scalar = vec![0f32; m];
            let mut out_neon = vec![0f32; m];
            // SAFETY: NEON baseline on aarch64; scalar is plain Rust.
            unsafe {
                scalar(&w, &x, &mut out_scalar, m, k);
                neon(&w, &x, &mut out_neon, m, k);
            }
            assert_close(&out_scalar, &out_neon, tag);
        }

        #[test]
        fn q4_0_neon_matches_scalar() {
            run_block_quant::<18>(
                "q4_0",
                5,
                128,
                0x4040_1111,
                |l, blk| {
                    blk[0..2].copy_from_slice(&l.f16_le(0.05, 0.001));
                    for q in &mut blk[2..18] {
                        *q = l.byte();
                    }
                },
                |w, x, o, m, k| matvec_q4_0_w_f32_a_scalar(w, x, o, m, k),
                |w, x, o, m, k| unsafe { matvec_q4_0_w_f32_a_neon(w, x, o, m, k) },
                32,
            );
        }

        #[test]
        fn q4_1_neon_matches_scalar() {
            run_block_quant::<20>(
                "q4_1",
                5,
                128,
                0x4141_2222,
                |l, blk| {
                    blk[0..2].copy_from_slice(&l.f16_le(0.05, 0.001));
                    blk[2..4].copy_from_slice(&l.f16_le(0.1, -0.05));
                    for q in &mut blk[4..20] {
                        *q = l.byte();
                    }
                },
                |w, x, o, m, k| matvec_q4_1_w_f32_a_scalar(w, x, o, m, k),
                |w, x, o, m, k| unsafe { matvec_q4_1_w_f32_a_neon(w, x, o, m, k) },
                32,
            );
        }

        #[test]
        fn q5_0_neon_matches_scalar() {
            run_block_quant::<22>(
                "q5_0",
                5,
                128,
                0x5050_3333,
                |l, blk| {
                    blk[0..2].copy_from_slice(&l.f16_le(0.05, 0.001));
                    for q in &mut blk[2..22] {
                        *q = l.byte();
                    }
                },
                |w, x, o, m, k| matvec_q5_0_w_f32_a_scalar(w, x, o, m, k),
                |w, x, o, m, k| unsafe { matvec_q5_0_w_f32_a_neon(w, x, o, m, k) },
                32,
            );
        }

        #[test]
        fn q5_1_neon_matches_scalar() {
            run_block_quant::<24>(
                "q5_1",
                5,
                128,
                0x5151_4444,
                |l, blk| {
                    blk[0..2].copy_from_slice(&l.f16_le(0.05, 0.001));
                    blk[2..4].copy_from_slice(&l.f16_le(0.1, -0.05));
                    for q in &mut blk[4..24] {
                        *q = l.byte();
                    }
                },
                |w, x, o, m, k| matvec_q5_1_w_f32_a_scalar(w, x, o, m, k),
                |w, x, o, m, k| unsafe { matvec_q5_1_w_f32_a_neon(w, x, o, m, k) },
                32,
            );
        }

        #[test]
        fn q6_k_neon_matches_scalar() {
            // Q6_K super-block: 128 ql + 64 qh + 16 i8 scales + f16 d.
            run_block_quant::<210>(
                "q6_k",
                4,
                512,
                0x6060_5555,
                |l, blk| {
                    for q in &mut blk[0..208] {
                        *q = l.byte();
                    }
                    blk[208..210].copy_from_slice(&l.f16_le(0.02, 0.001));
                },
                |w, x, o, m, k| matvec_q6_k_w_f32_a_scalar(w, x, o, m, k),
                |w, x, o, m, k| unsafe { matvec_q6_k_w_f32_a_neon(w, x, o, m, k) },
                256,
            );
        }

        #[test]
        fn q2_k_neon_matches_scalar() {
            // Q2_K super-block: scales[16] + qs[64] + d(f16) + dmin(f16).
            run_block_quant::<84>(
                "q2_k",
                4,
                512,
                0x2020_6666,
                |l, blk| {
                    for s in &mut blk[0..80] {
                        *s = l.byte();
                    }
                    blk[80..82].copy_from_slice(&l.f16_le(0.03, 0.001));
                    blk[82..84].copy_from_slice(&l.f16_le(0.03, 0.001));
                },
                |w, x, o, m, k| matvec_q2_k_w_f32_a_scalar(w, x, o, m, k),
                |w, x, o, m, k| unsafe { matvec_q2_k_w_f32_a_neon(w, x, o, m, k) },
                256,
            );
        }

        #[test]
        fn q3_k_neon_matches_scalar() {
            // Q3_K super-block: hmask[32] + qs[64] + sc[12] + d(f16).
            run_block_quant::<110>(
                "q3_k",
                4,
                512,
                0x3030_7777,
                |l, blk| {
                    for s in &mut blk[0..108] {
                        *s = l.byte();
                    }
                    blk[108..110].copy_from_slice(&l.f16_le(0.02, 0.001));
                },
                |w, x, o, m, k| matvec_q3_k_w_f32_a_scalar(w, x, o, m, k),
                |w, x, o, m, k| unsafe { matvec_q3_k_w_f32_a_neon(w, x, o, m, k) },
                256,
            );
        }

        #[test]
        fn q4_k_neon_matches_scalar() {
            // Q4_K super-block: d(f16) + dmin(f16) + sc[12] + qs[128].
            run_block_quant::<144>(
                "q4_k",
                4,
                512,
                0x4040_8888,
                |l, blk| {
                    blk[0..2].copy_from_slice(&l.f16_le(0.03, 0.001));
                    blk[2..4].copy_from_slice(&l.f16_le(0.03, 0.001));
                    for s in &mut blk[4..144] {
                        *s = l.byte();
                    }
                },
                |w, x, o, m, k| matvec_q4_k_w_f32_a_scalar(w, x, o, m, k),
                |w, x, o, m, k| unsafe { matvec_q4_k_w_f32_a_neon(w, x, o, m, k) },
                256,
            );
        }

        #[test]
        fn q8_0_neon_matches_scalar() {
            // Q8_0 block: f16 d + 32 int8 qs (34 bytes / 32 weights).
            run_block_quant::<34>(
                "q8_0",
                4,
                512,
                0x8080_9999,
                |l, blk| {
                    blk[0..2].copy_from_slice(&l.f16_le(0.02, 0.001));
                    for q in &mut blk[2..34] {
                        *q = l.byte();
                    }
                },
                |w, x, o, m, k| matvec_q8_0_w_f32_a_scalar(w, x, o, m, k),
                |w, x, o, m, k| unsafe { matvec_q8_0_w_f32_a_neon(w, x, o, m, k) },
                32,
            );
        }

        #[test]
        fn q5_k_neon_matches_scalar() {
            // Q5_K super-block: d(f16) + dmin(f16) + scales[12] + qh[32] + qs[128].
            run_block_quant::<176>(
                "q5_k",
                4,
                512,
                0x5050_8888,
                |l, blk| {
                    blk[0..2].copy_from_slice(&l.f16_le(0.03, 0.001));
                    blk[2..4].copy_from_slice(&l.f16_le(0.03, 0.001));
                    for s in &mut blk[4..176] {
                        *s = l.byte();
                    }
                },
                |w, x, o, m, k| matvec_q5_k_w_f32_a_scalar(w, x, o, m, k),
                |w, x, o, m, k| unsafe { matvec_q5_k_w_f32_a_neon(w, x, o, m, k) },
                256,
            );
        }

        #[test]
        fn q8_k_neon_matches_scalar() {
            // Q8_K super-block: d(f32) + qs[i8;256] + bsums[i16;16] (unused here).
            run_block_quant::<292>(
                "q8_k",
                4,
                512,
                0x8080_9999,
                |l, blk| {
                    blk[0..4].copy_from_slice(&l.f32_le(0.02, 0.001));
                    for s in &mut blk[4..292] {
                        *s = l.byte();
                    }
                },
                |w, x, o, m, k| matvec_q8_k_w_f32_a_scalar(w, x, o, m, k),
                |w, x, o, m, k| unsafe { matvec_q8_k_w_f32_a_neon(w, x, o, m, k) },
                256,
            );
        }

        // Dense matvecs + forward helpers use `k`/`n = 70` (not a multiple
        // of the 16-lane tile) so the scalar tail is exercised too.
        #[test]
        fn matvec_f32_neon_matches_scalar() {
            let mut l = Lcg(0x0F32_0001);
            let (m, k) = (5usize, 70usize);
            let w: Vec<f32> = (0..m * k).map(|_| l.activation()).collect();
            let x: Vec<f32> = (0..k).map(|_| l.activation()).collect();
            let mut os = vec![0f32; m];
            let mut on = vec![0f32; m];
            matvec_f32_scalar(&w, &x, &mut os, m, k);
            unsafe { matvec_f32_neon(&w, &x, &mut on, m, k) };
            assert_close(&os, &on, "matvec_f32");
        }

        #[test]
        fn matvec_bf16_neon_matches_scalar() {
            let mut l = Lcg(0x0BF1_0001);
            let (m, k) = (5usize, 70usize);
            let mut w = vec![0u8; m * k * 2];
            for pair in w.chunks_exact_mut(2) {
                // A sane bf16 = top 16 bits of a small f32.
                let bf = (l.activation().to_bits() >> 16) as u16;
                pair.copy_from_slice(&bf.to_le_bytes());
            }
            let x: Vec<f32> = (0..k).map(|_| l.activation()).collect();
            let mut os = vec![0f32; m];
            let mut on = vec![0f32; m];
            matvec_bf16_w_f32_a_scalar(&w, &x, &mut os, m, k);
            unsafe { matvec_bf16_w_f32_a_neon(&w, &x, &mut on, m, k) };
            assert_close(&os, &on, "matvec_bf16");
        }

        #[test]
        fn matvec_f16_neon_matches_scalar() {
            // k = 70 exercises the 16-lane tile, the 4-lane mid loop, and the
            // scalar tail. The hand-rolled widen is exact for finite f16, so
            // this should be bit-identical (abs == 0) to the naive reference.
            let mut l = Lcg(0x0F16_0001);
            let (m, k) = (5usize, 70usize);
            let w: Vec<f16> = (0..m * k).map(|_| f16::from_f32(l.activation())).collect();
            let x: Vec<f32> = (0..k).map(|_| l.activation()).collect();
            let mut os = vec![0f32; m];
            for (i, o) in os.iter_mut().enumerate() {
                let mut acc = 0f32;
                for p in 0..k {
                    acc += w[i * k + p].to_f32() * x[p];
                }
                *o = acc;
            }
            let mut on = vec![0f32; m];
            unsafe { matvec_f16_w_f32_a_neon(&w, &x, &mut on, m, k) };
            assert_close(&os, &on, "matvec_f16");
        }

        #[test]
        fn rmsnorm_neon_matches_scalar() {
            let mut l = Lcg(0x8311_0001);
            let d = 70usize;
            let x: Vec<f32> = (0..d).map(|_| l.activation()).collect();
            let w: Vec<f32> = (0..d).map(|_| l.activation()).collect();
            let mut ys = vec![0f32; d];
            let mut yn = vec![0f32; d];
            rmsnorm_f32_row_scalar(&x, &w, &mut ys, 1e-5);
            unsafe { rmsnorm_f32_row_neon(&x, &w, &mut yn, 1e-5) };
            assert_close(&ys, &yn, "rmsnorm");
        }

        #[test]
        fn add_inplace_neon_matches_scalar() {
            let mut l = Lcg(0x0ADD_0001);
            let n = 70usize;
            let base: Vec<f32> = (0..n).map(|_| l.activation()).collect();
            let b: Vec<f32> = (0..n).map(|_| l.activation()).collect();
            let mut a_s = base.clone();
            let mut a_n = base.clone();
            for (lhs, rhs) in a_s.iter_mut().zip(b.iter()) {
                *lhs += *rhs;
            }
            unsafe { add_inplace_f32_neon(&mut a_n, &b) };
            assert_close(&a_s, &a_n, "add_inplace"); // add is bit-exact
        }

        #[test]
        fn iq4_nl_neon_matches_scalar() {
            // IQ4_NL block: d(f16) + qs[16].
            run_block_quant::<18>(
                "iq4_nl",
                5,
                128,
                0x4949_1010,
                |l, blk| {
                    blk[0..2].copy_from_slice(&l.f16_le(0.05, 0.001));
                    for q in &mut blk[2..18] {
                        *q = l.byte();
                    }
                },
                |w, x, o, m, k| matvec_iq4_nl_w_f32_a_scalar(w, x, o, m, k),
                |w, x, o, m, k| unsafe { matvec_iq4_nl_w_f32_a_neon(w, x, o, m, k) },
                32,
            );
        }

        #[test]
        fn iq4_xs_neon_matches_scalar() {
            // IQ4_XS super-block: d(f16) + scales_h(u16) + scales_l[4] + qs[128].
            run_block_quant::<136>(
                "iq4_xs",
                4,
                512,
                0x4958_2020,
                |l, blk| {
                    blk[0..2].copy_from_slice(&l.f16_le(0.02, 0.001));
                    for s in &mut blk[2..136] {
                        *s = l.byte();
                    }
                },
                |w, x, o, m, k| matvec_iq4_xs_w_f32_a_scalar(w, x, o, m, k),
                |w, x, o, m, k| unsafe { matvec_iq4_xs_w_f32_a_neon(w, x, o, m, k) },
                256,
            );
        }

        // ---- Grid-codebook IQ quants (scalar-gather + NEON arithmetic) ----
        //
        // The block layouts carry the super-block scale `d` as a leading f16
        // (sane-small via `f16_le`) with the rest of the block fully random;
        // every grid index / sign index the random bytes can produce is in
        // range (the grids span the full index space), so the NEON and scalar
        // paths decode identical runs.

        #[test]
        fn iq2_xxs_neon_matches_scalar() {
            run_block_quant::<66>(
                "iq2_xxs",
                4,
                512,
                0x2222_1a1a,
                |l, blk| {
                    blk[0..2].copy_from_slice(&l.f16_le(0.05, 0.001));
                    for q in &mut blk[2..66] {
                        *q = l.byte();
                    }
                },
                |w, x, o, m, k| matvec_iq2_xxs_w_f32_a_scalar(w, x, o, m, k),
                |w, x, o, m, k| unsafe { matvec_iq2_xxs_w_f32_a_neon(w, x, o, m, k) },
                256,
            );
        }

        #[test]
        fn iq2_xs_neon_matches_scalar() {
            run_block_quant::<74>(
                "iq2_xs",
                4,
                512,
                0x2358_1b1b,
                |l, blk| {
                    blk[0..2].copy_from_slice(&l.f16_le(0.05, 0.001));
                    for q in &mut blk[2..74] {
                        *q = l.byte();
                    }
                },
                |w, x, o, m, k| matvec_iq2_xs_w_f32_a_scalar(w, x, o, m, k),
                |w, x, o, m, k| unsafe { matvec_iq2_xs_w_f32_a_neon(w, x, o, m, k) },
                256,
            );
        }

        #[test]
        fn iq2_s_neon_matches_scalar() {
            run_block_quant::<82>(
                "iq2_s",
                4,
                512,
                0x2525_1c1c,
                |l, blk| {
                    blk[0..2].copy_from_slice(&l.f16_le(0.05, 0.001));
                    for q in &mut blk[2..82] {
                        *q = l.byte();
                    }
                },
                |w, x, o, m, k| matvec_iq2_s_w_f32_a_scalar(w, x, o, m, k),
                |w, x, o, m, k| unsafe { matvec_iq2_s_w_f32_a_neon(w, x, o, m, k) },
                256,
            );
        }

        #[test]
        fn iq3_xxs_neon_matches_scalar() {
            run_block_quant::<98>(
                "iq3_xxs",
                4,
                512,
                0x3358_1d1d,
                |l, blk| {
                    blk[0..2].copy_from_slice(&l.f16_le(0.05, 0.001));
                    for q in &mut blk[2..98] {
                        *q = l.byte();
                    }
                },
                |w, x, o, m, k| matvec_iq3_xxs_w_f32_a_scalar(w, x, o, m, k),
                |w, x, o, m, k| unsafe { matvec_iq3_xxs_w_f32_a_neon(w, x, o, m, k) },
                256,
            );
        }

        #[test]
        fn iq3_s_neon_matches_scalar() {
            run_block_quant::<110>(
                "iq3_s",
                4,
                512,
                0x3535_1e1e,
                |l, blk| {
                    blk[0..2].copy_from_slice(&l.f16_le(0.05, 0.001));
                    for q in &mut blk[2..110] {
                        *q = l.byte();
                    }
                },
                |w, x, o, m, k| matvec_iq3_s_w_f32_a_scalar(w, x, o, m, k),
                |w, x, o, m, k| unsafe { matvec_iq3_s_w_f32_a_neon(w, x, o, m, k) },
                256,
            );
        }

        #[test]
        fn iq1_s_neon_matches_scalar() {
            run_block_quant::<50>(
                "iq1_s",
                4,
                512,
                0x1515_1f1f,
                |l, blk| {
                    blk[0..2].copy_from_slice(&l.f16_le(0.05, 0.001));
                    for q in &mut blk[2..50] {
                        *q = l.byte();
                    }
                },
                |w, x, o, m, k| matvec_iq1_s_w_f32_a_scalar(w, x, o, m, k),
                |w, x, o, m, k| unsafe { matvec_iq1_s_w_f32_a_neon(w, x, o, m, k) },
                256,
            );
        }

        #[test]
        fn iq1_m_neon_matches_scalar() {
            // IQ1_M (56 bytes): qs[32] + qh[16] + 4 packed scale words. The
            // super-block scale `d` is reassembled from the high nibbles of
            // the four scale words, so plant a known small f16 across those
            // nibbles and leave the rest (dl fields + qs + qh) random.
            run_block_quant::<56>(
                "iq1_m",
                4,
                512,
                0x1949_3030,
                |l, blk| {
                    for q in &mut blk[0..48] {
                        *q = l.byte();
                    }
                    let target = f16::from_f32(0.05).to_bits();
                    let r0 = (l.next_u32() as u16) & 0x0FFF;
                    let r1 = (l.next_u32() as u16) & 0xF0FF;
                    let r2 = (l.next_u32() as u16) & 0x0FFF;
                    let r3 = (l.next_u32() as u16) & 0x0FFF;
                    let sc0 = r0 | ((target & 0x000F) << 12);
                    let sc1 = r1 | (((target >> 4) & 0x000F) << 8);
                    let sc2 = r2 | (((target >> 8) & 0x000F) << 12);
                    let sc3 = r3 | (target & 0xF000);
                    blk[48..50].copy_from_slice(&sc0.to_le_bytes());
                    blk[50..52].copy_from_slice(&sc1.to_le_bytes());
                    blk[52..54].copy_from_slice(&sc2.to_le_bytes());
                    blk[54..56].copy_from_slice(&sc3.to_le_bytes());
                },
                |w, x, o, m, k| matvec_iq1_m_w_f32_a_scalar(w, x, o, m, k),
                |w, x, o, m, k| unsafe { matvec_iq1_m_w_f32_a_neon(w, x, o, m, k) },
                256,
            );
        }
    }
}
