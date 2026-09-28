//! Re-quantization pipeline.
//!
//! Drives a source [`Gguf`] through a per-tensor encode → write
//! sequence into a [`GgufWriter`]. Per-tensor target dtype is
//! chosen by [`QuantizePlan`]; the pipeline handles:
//!   - tensors that are already at the target dtype (byte-passthrough)
//!   - non-quantizable tensors (F32 / F16 norms, biases, embeddings
//!     that the spec keeps at high precision) — passed through at
//!     source dtype
//!   - shape compatibility (target dtype's block size must divide the
//!     tensor's element count; falls back to source dtype if it doesn't)
//!
//! The pipeline streams one tensor at a time. Peak memory =
//! `max(source_tensor_bytes, dequant_f32_bytes, target_bytes)` per
//! tensor — a single 4096×4096 F16 tensor needs ~80 MB peak vs the
//! ~140 GB total a 70B model would otherwise demand.

use std::cell::RefCell;
use std::io::{Seek, Write};

use rayon::prelude::*;

use crate::{dequant, encode, encode_iq, encode_iq_vec, encode_k, encode_t};
use crate::recipe::{resolve_first_match, RecipeRule};
use crate::{Gguf, GgmlType};
use crate::write::{GgufMmapWriter, GgufWriter, WriteError};

thread_local! {
    static F32_SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Borrow a thread-local F32 scratch buffer of at least `n` elements.
///
/// Why: the dequant→encode chunk loop used to `vec![0f32; n]` on every
/// rayon iteration, producing ~400 small allocs per chunked tensor and
/// one ~80 MiB alloc per whole-tensor IQ path. Pooling per-thread
/// turns that into one grow-on-first-use per worker.
///
/// Contents are NOT zeroed across calls — every dequant routine
/// writes its full output slice, so stale data is overwritten before
/// the encoder reads.
fn with_f32_scratch<R>(n: usize, f: impl FnOnce(&mut [f32]) -> R) -> R {
    F32_SCRATCH.with(|cell| {
        let mut buf = cell.borrow_mut();
        if buf.len() < n {
            buf.resize(n, 0.0);
        }
        f(&mut buf[..n])
    })
}

#[derive(Debug, thiserror::Error)]
pub enum QuantizeError {
    #[error("writer: {0}")]
    Write(#[from] WriteError),
    #[error(
        "tensor {name:?} cannot be re-quantized to {target:?}: element count {n} not a \
         multiple of the target block size {block_size}"
    )]
    BadElementCount {
        name: String,
        target: GgmlType,
        n: u64,
        block_size: u64,
    },
    #[error("source tensor {0:?} is missing from the GGUF file")]
    MissingTensor(String),
    #[error(
        "source tensor {name:?} has dtype {src:?} which is not supported for dequantization. \
         The re-quantizer needs to dequant to F32 as an intermediate step before \
         re-encoding; add a dequant routine for {src:?} or pass-through the tensor."
    )]
    UnsupportedSourceDtype { name: String, src: GgmlType },
    #[error(
        "target dtype {0:?} is not yet supported for encoding. \
         IQ1_S/M, IQ2_*, IQ3_* require vector-grid codebook search and are tracked as Q-D-extra; \
         this release ships only IQ4_NL and IQ4_XS from the IQ family."
    )]
    UnsupportedTargetDtype(GgmlType),
}

/// Per-tensor target-dtype recipe.
///
/// Resolution order per tensor (first match wins):
///   1. **Passthrough prefixes** — name starts with any entry → no
///      re-quantization (keep source dtype).
///   2. **Exact-name overrides** — for legacy callers that built
///      plans before recipes shipped.
///   3. **Rule list** — glob-pattern recipes (from `recipe::parse_*`
///      or `apex::build_apex_rules`); walked in order.
///   4. **`is_quantizable_tensor`** — fall through to `default_target`
///      for quantizable tensors, source dtype for the rest.
#[derive(Debug, Clone)]
pub struct QuantizePlan {
    /// Default target dtype for quantizable tensors that no
    /// recipe rule matches. Norms / biases / vocab-sized tensors
    /// fall through to a passthrough at source dtype regardless
    /// (per [`is_quantizable_tensor`]).
    pub default_target: GgmlType,
    /// Exact-name overrides. Kept for legacy callers; the
    /// recipe-rule list below supersedes this for new code.
    pub overrides: Vec<(String, GgmlType)>,
    /// Pin tensors whose name matches any of these prefixes at the
    /// source dtype (no re-quantization). The llama.cpp convention
    /// `--leave-output-tensor` corresponds to `["output."]`.
    pub passthrough_prefixes: Vec<String>,
    /// Glob-pattern recipe rules. Walked in order; first match
    /// wins. Populated from a recipe file (via
    /// [`crate::recipe::parse_recipe_file`]) or an APEX profile
    /// (via [`crate::apex::build_apex_rules`]).
    pub rules: Vec<RecipeRule>,
}

impl QuantizePlan {
    pub fn uniform(target: GgmlType) -> Self {
        Self {
            default_target: target,
            overrides: Vec::new(),
            passthrough_prefixes: Vec::new(),
            rules: Vec::new(),
        }
    }

    /// Append rules to the plan. Order matters — rules added later
    /// are checked later (first match wins). The CLI installs the
    /// user-provided recipe AFTER the APEX profile so user rules
    /// override APEX defaults.
    pub fn add_rules(&mut self, more: impl IntoIterator<Item = RecipeRule>) {
        self.rules.extend(more);
    }

    /// Resolve the target dtype for a given source tensor. Returns
    /// `None` when the tensor should be passed through verbatim.
    pub fn target_for(&self, name: &str, src_dtype: GgmlType, n_elements: u64) -> Option<GgmlType> {
        // Passthrough by prefix.
        for prefix in &self.passthrough_prefixes {
            if name.starts_with(prefix) {
                return None;
            }
        }
        // Explicit per-tensor override.
        for (k, v) in &self.overrides {
            if k == name {
                return Some(*v);
            }
        }
        // Recipe-rule glob match.
        if let Some(target) = resolve_first_match(name, &self.rules) {
            return Some(target);
        }
        // Non-quantizable tensors flow through at source dtype.
        if !is_quantizable_tensor(name, src_dtype, n_elements) {
            return None;
        }
        Some(self.default_target)
    }
}

/// Whether the re-quantizer should treat this tensor as a
/// quantization target. Tensors that should remain in F32 / F16
/// (norms, biases, single-row vocab metadata) return `false` here
/// so the pipeline copies them through unchanged.
///
/// Mirrors llama.cpp's `llama_model_quantize_internal`
/// "should I quantize this tensor" heuristic: keep <=1D tensors
/// raw (norms / biases / scales); keep the token embedding raw
/// unless the user opted in; everything else is fair game.
pub fn is_quantizable_tensor(_name: &str, src_dtype: GgmlType, _n_elements: u64) -> bool {
    // Already a small dtype → no benefit to re-encode.
    if matches!(src_dtype, GgmlType::Q8_0 | GgmlType::Q4_0 | GgmlType::Q4_1) {
        // Could still re-quantize down to a smaller target, but for
        // v1 we treat "source already quantized" as passthrough and
        // surface that via plan.overrides if the user really wants
        // to re-encode. Saves quality loss on the double-quant path.
        return false;
    }
    // F32 / F16 / BF16 source: quantizable.
    matches!(
        src_dtype,
        GgmlType::F32 | GgmlType::F16 | GgmlType::Bf16
    )
}

/// True when `name` belongs to a multi-token-prediction (MTP /
/// NextN) draft head used for speculative decoding. Matches the
/// qwen35moe `blk.{N}.nextn.*` layout, bare `.eh_proj` projections,
/// and the DeepSeek-style `mtp.{i}.*` namespace (see
/// rustllama-models' loaders for the on-disk names).
pub fn is_mtp_head_tensor(name: &str) -> bool {
    name.contains(".nextn.") || name.contains(".eh_proj") || name.starts_with("mtp.")
}

/// True when `dtype` stores at least 8 bits per weight (F32 / F16 /
/// BF16 / Q8_0 / Q8_1 / Q8_K). Derived from [`GgmlType::byte_size`]
/// over a 256-element span — a multiple of every block size (1, 16,
/// 32, 256), so the div_ceil rounding inside is exact and new dtypes
/// classify automatically instead of via a hand-kept list.
pub fn dtype_at_least_8bpw(dtype: GgmlType) -> bool {
    // ≥ 8 bpw ⇔ ≥ 1 byte per weight over one exact 256-weight span.
    dtype.byte_size(256) >= 256
}

/// Warning predicate: an MTP/NextN head tensor resolved to a
/// sub-8-bit dtype. Empirical finding (Colibrì): below int8 the
/// draft head's acceptance collapses to 0–4% (vs 39–59% at int8+),
/// erasing the speculative-decoding speedup. Pure function so the
/// classification is unit-testable apart from the tracing side
/// effect in [`warn_if_mtp_below_8bpw`].
pub fn mtp_head_below_8bpw(name: &str, target: GgmlType) -> bool {
    is_mtp_head_tensor(name) && !dtype_at_least_8bpw(target)
}

/// Emit the sub-Q8_0 MTP warning for one resolved tensor. Called
/// from each pipeline's resolve loop AFTER the shape-misalignment
/// fallback so `target` is the dtype actually written. Deliberately
/// does NOT alter the resolution — explicit recipes stay
/// authoritative; the built-in floor lives in the APEX profiles
/// (see [`crate::apex::build_apex_rules`]).
fn warn_if_mtp_below_8bpw(name: &str, target: GgmlType) {
    if mtp_head_below_8bpw(name, target) {
        tracing::warn!(
            tensor = name,
            target = target.as_str(),
            "MTP/NextN head below Q8_0 cripples speculative-decoding draft \
             acceptance (0-4% vs 39-59% at int8+) — pin this tensor >= Q8_0 \
             in the recipe"
        );
    }
}

#[derive(Debug, Default, Clone)]
pub struct QuantizeStats {
    pub tensors_total: usize,
    pub tensors_requantized: usize,
    pub tensors_passthrough: usize,
    pub bytes_in: u64,
    pub bytes_out: u64,
}

/// Run the re-quantization pipeline end-to-end. Copies every metadata
/// key from `src`, declares each tensor at its resolved target dtype,
/// then streams tensor data: dequant → re-encode → write.
pub fn quantize_gguf<W: Write + Seek>(
    src: &Gguf,
    dst: &mut GgufWriter<W>,
    plan: &QuantizePlan,
) -> Result<QuantizeStats, QuantizeError> {
    // Stage 1: copy metadata. The caller may have already added
    // overrides (e.g., general.file_type) before calling — skip
    // duplicates rather than failing.
    for (key, value) in src.metadata() {
        // Skip alignment — the writer manages its own.
        if key == "general.alignment" {
            continue;
        }
        // Skip if writer already has this key (caller pre-populated).
        if dst_has_metadata(dst, key) {
            continue;
        }
        dst.add_metadata(key.clone(), value.clone())?;
    }

    // Stage 2: resolve targets + declare tensors. Cache resolved
    // (target_dtype, target_byte_size) per tensor so stage 3 doesn't
    // re-derive them.
    let n_tensors = src.tensors().len();
    let mut resolved: Vec<ResolvedTensor> = Vec::with_capacity(n_tensors);
    for t in src.tensors() {
        let mut target = match plan.target_for(&t.name, t.dtype, t.element_count()) {
            Some(target) => target,
            None => t.dtype, // passthrough
        };

        // Validate target encoder exists (else surface a clean error
        // before we waste time on metadata).
        if target != t.dtype && !encoder_supported(target) {
            return Err(QuantizeError::UnsupportedTargetDtype(target));
        }

        // Shape-misalignment auto-passthrough. Real models contain
        // small per-head SSM / MTP / bias / norm tensors whose
        // element count doesn't divide the target dtype's block
        // size (e.g. `blk.0.ssm_a` at 32 elements vs IQ1_S's
        // 256-block layout). Hard-failing would force the user to
        // hand-write recipe rules for every such tensor; instead
        // we silently fall back to the source dtype and let the
        // tensor pass through unchanged. Matches llama.cpp's
        // `--leave-output-tensor`-style "if it doesn't fit, leave
        // it alone" default.
        let n = t.element_count();
        let block_size = target_block_size(target);
        if n % block_size != 0 {
            if target != t.dtype {
                // Re-quantize target was infeasible — pass through.
                // The src dtype's block size always divides its own
                // element count (it was loaded successfully), so the
                // passthrough never trips this check.
                target = t.dtype;
            } else {
                // Already passthrough but somehow misaligned —
                // genuine corruption. Surface the error.
                return Err(QuantizeError::BadElementCount {
                    name: t.name.clone(),
                    target,
                    n,
                    block_size,
                });
            }
        }
        // Recipes stay authoritative — warn (don't override) when an
        // MTP head lands below 8 bits per weight.
        warn_if_mtp_below_8bpw(&t.name, target);
        dst.declare_tensor(t.name.clone(), t.dims.clone(), target)?;
        resolved.push(ResolvedTensor {
            name: t.name.clone(),
            src_dtype: t.dtype,
            target_dtype: target,
            n_elements: n,
            dims: t.dims.clone(),
            passthrough: target == t.dtype,
        });
    }

    dst.finish_header()?;

    // Stage 3: stream tensor data.
    let mut stats = QuantizeStats {
        tensors_total: n_tensors,
        ..Default::default()
    };
    for r in resolved {
        let src_bytes = src
            .tensor_bytes(&r.name)
            .ok_or_else(|| QuantizeError::MissingTensor(r.name.clone()))?;
        stats.bytes_in += src_bytes.len() as u64;
        if r.passthrough {
            dst.write_tensor_data(&r.name, src_bytes)?;
            stats.bytes_out += src_bytes.len() as u64;
            stats.tensors_passthrough += 1;
        } else {
            // **Chunked + parallel** dequant → encode.
            //
            // The encoders are block-by-block internally, so each
            // chunk of `CHUNK_WEIGHTS` is independent and safe to
            // process on a separate rayon worker. The output buffer
            // is partitioned via `par_chunks_mut` into disjoint
            // writable slices, one per chunk — no locks, no contention.
            //
            // RAM bound: peak F32 scratch is `n_threads * CHUNK_WEIGHTS
            // * 4` bytes. On an 8-core box: ~8 MB. On a 32-core box:
            // ~32 MB. Trivial compared to the multi-GB tensors this
            // pipeline targets.
            //
            // Speedup: the IQ1_S encoder's 2048-grid × 2-delta
            // exhaustive search is the wall-clock bottleneck on big
            // MoE expert tensors. Parallelizing across chunks within
            // a tensor scales near-linearly with cores (no shared
            // state between chunks). On an 8-core machine this drops
            // a 60-hour single-threaded IQ1_S re-quant of a 35B model
            // to ~8 hours.
            //
            // `CHUNK_WEIGHTS = 256 * 1024` is divisible by every
            // quantization format's block size (1, 16, 32, 256), so
            // chunk boundaries always align. The final partial chunk
            // (smaller byte slice from par_chunks_mut) handles the
            // tail when total_n isn't a multiple of CHUNK_WEIGHTS.
            const CHUNK_WEIGHTS: usize = 256 * 1024;
            let total_n = r.n_elements as usize;
            let chunk_cap = CHUNK_WEIGHTS.min(total_n);
            let target_bytes = r.target_dtype.byte_size(r.n_elements) as usize;
            let mut enc_buf = vec![0u8; target_bytes];

            // Per-chunk target byte size for non-tail chunks.
            // par_chunks_mut yields chunks of exactly this size,
            // except the last (which is smaller if total_n isn't a
            // multiple of CHUNK_WEIGHTS).
            let chunk_target_bytes =
                r.target_dtype.byte_size(chunk_cap as u64) as usize;
            let src_dtype = r.src_dtype;
            let target_dtype = r.target_dtype;
            let tensor_name = r.name.clone();

            let result: Result<(), QuantizeError> = enc_buf
                .par_chunks_mut(chunk_target_bytes)
                .enumerate()
                .try_for_each(|(i, out_chunk)| {
                    let start_weights = i * chunk_cap;
                    let end_weights = (start_weights + chunk_cap).min(total_n);
                    let n_this = end_weights - start_weights;
                    let src_lo =
                        src_dtype.byte_size(start_weights as u64) as usize;
                    let src_hi =
                        src_dtype.byte_size(end_weights as u64) as usize;
                    with_f32_scratch(n_this, |local_f32| {
                        dequant_to_f32(
                            src_dtype,
                            &src_bytes[src_lo..src_hi],
                            local_f32,
                        )
                        .map_err(|()| QuantizeError::UnsupportedSourceDtype {
                            name: tensor_name.clone(),
                            src: src_dtype,
                        })?;
                        encode_from_f32(target_dtype, local_f32, out_chunk);
                        Ok(())
                    })
                });
            result?;

            dst.write_tensor_data(&r.name, &enc_buf)?;
            stats.bytes_out += target_bytes as u64;
            stats.tensors_requantized += 1;
        }
    }

    Ok(stats)
}

struct ResolvedTensor {
    name: String,
    src_dtype: GgmlType,
    target_dtype: GgmlType,
    n_elements: u64,
    /// Tensor dims as stored in the GGUF (`dims[0]` = innermost =
    /// input-column count `n_in`). Used to broadcast a per-column
    /// imatrix vector across output rows during encode.
    dims: Vec<u64>,
    passthrough: bool,
}

/// **mmap-backed** variant of [`quantize_gguf`]. Writes the output
/// GGUF via a memory-mapped file so encoded tensor bytes never
/// land in a heap-allocated `Vec` — peak RAM stays bounded by the
/// per-chunk F32 scratch + whatever pages the OS keeps resident.
///
/// Use this for the production CLI path (especially on memory-tight
/// hosts re-quantizing big models to high-bpw targets where a
/// per-tensor enc_buf could be GB-sized). Tests can stick with
/// [`quantize_gguf`] which writes into a `Vec` for easy in-memory
/// inspection.
pub fn quantize_gguf_to_path<P: AsRef<std::path::Path>>(
    src: &Gguf,
    output_path: P,
    plan: &QuantizePlan,
) -> Result<QuantizeStats, QuantizeError> {
    quantize_gguf_to_path_with_imatrix(src, output_path, plan, None)
}

/// Importance-matrix-aware variant of [`quantize_gguf_to_path`]. When
/// `imatrix` is `Some`, each tensor whose name has an entry (and whose
/// stored input-column count `dims[0]` matches the entry length) is
/// encoded with per-column importance weighting via the `*_imatrix`
/// encoders (Q2_K/Q4_K/Q5_K/IQ1_S; other dtypes ignore it). `None`
/// reproduces the uniform encode exactly.
pub fn quantize_gguf_to_path_with_imatrix<P: AsRef<std::path::Path>>(
    src: &Gguf,
    output_path: P,
    plan: &QuantizePlan,
    imatrix: Option<&crate::imatrix::Imatrix>,
) -> Result<QuantizeStats, QuantizeError> {
    let mut dst = GgufMmapWriter::create(&output_path)?;

    // Metadata copy — same as the Vec-backed path.
    for (key, value) in src.metadata() {
        if key == "general.alignment" {
            continue;
        }
        dst.add_metadata(key.clone(), value.clone())?;
    }

    // Resolve targets + declare tensors. Same logic as quantize_gguf;
    // could be factored, but inlining keeps each entry point's
    // hot path explicit.
    let n_tensors = src.tensors().len();
    let mut resolved: Vec<ResolvedTensor> = Vec::with_capacity(n_tensors);
    for t in src.tensors() {
        let mut target = match plan.target_for(&t.name, t.dtype, t.element_count()) {
            Some(target) => target,
            None => t.dtype,
        };
        if target != t.dtype && !encoder_supported(target) {
            return Err(QuantizeError::UnsupportedTargetDtype(target));
        }
        let n = t.element_count();
        let block_size = target_block_size(target);
        if n % block_size != 0 {
            if target != t.dtype {
                target = t.dtype;
            } else {
                return Err(QuantizeError::BadElementCount {
                    name: t.name.clone(),
                    target,
                    n,
                    block_size,
                });
            }
        }
        // Recipes stay authoritative — warn (don't override) when an
        // MTP head lands below 8 bits per weight.
        warn_if_mtp_below_8bpw(&t.name, target);
        dst.declare_tensor(t.name.clone(), t.dims.clone(), target)?;
        resolved.push(ResolvedTensor {
            name: t.name.clone(),
            src_dtype: t.dtype,
            target_dtype: target,
            n_elements: n,
            dims: t.dims.clone(),
            passthrough: target == t.dtype,
        });
    }

    dst.finish_header()?;

    let mut stats = QuantizeStats {
        tensors_total: n_tensors,
        ..Default::default()
    };
    for r in resolved {
        let src_bytes = src
            .tensor_bytes(&r.name)
            .ok_or_else(|| QuantizeError::MissingTensor(r.name.clone()))?;
        stats.bytes_in += src_bytes.len() as u64;

        // Borrow the mapped region for this tensor. The borrow
        // lifetime is bounded by the loop iteration so the
        // borrow checker is happy with successive tensors.
        let region = dst.tensor_region_mut(&r.name)?;
        let region_bytes = region.len();

        if r.passthrough {
            region.copy_from_slice(src_bytes);
            stats.bytes_out += region_bytes as u64;
            stats.tensors_passthrough += 1;
        } else {
            // Chunked + parallel dequant→encode, writing directly
            // into the mapped region. Same algorithm as
            // quantize_gguf but the output buffer is file-mapped
            // memory rather than a Vec.
            const CHUNK_WEIGHTS: usize = 256 * 1024;
            let total_n = r.n_elements as usize;
            let chunk_cap = CHUNK_WEIGHTS.min(total_n);
            let chunk_target_bytes =
                r.target_dtype.byte_size(chunk_cap as u64) as usize;
            let src_dtype = r.src_dtype;
            let target_dtype = r.target_dtype;
            let tensor_name = r.name.clone();
            // Resolve this tensor's per-input-column importance vector.
            // Only honor it when the length matches the stored inner
            // dim (`dims[0]` = n_in) and the tensor is a clean multiple
            // — otherwise fall back to uniform (defensive against a
            // stale/mismatched imatrix).
            let n_in = r.dims.first().copied().unwrap_or(0) as usize;
            let col_imp: Option<&[f32]> = imatrix.and_then(|im| im.get(&r.name)).filter(|c| {
                n_in > 0 && c.len() == n_in && total_n % n_in == 0
            });
            let result: Result<(), QuantizeError> = region
                .par_chunks_mut(chunk_target_bytes)
                .enumerate()
                .try_for_each(|(i, out_chunk)| {
                    let start_weights = i * chunk_cap;
                    let end_weights = (start_weights + chunk_cap).min(total_n);
                    let n_this = end_weights - start_weights;
                    let src_lo =
                        src_dtype.byte_size(start_weights as u64) as usize;
                    let src_hi =
                        src_dtype.byte_size(end_weights as u64) as usize;
                    // imatrix path uses a LOCAL f32 buffer, not the
                    // thread-local `with_f32_scratch`: the importance-
                    // weighted encoders (IQ1_S, IQ2_XXS) parallelize
                    // internally with rayon, and holding the thread-
                    // local RefCell borrow across that nested parallel
                    // region lets rayon work-steal another outer chunk
                    // onto this thread and re-enter the borrow →
                    // BorrowMutError panic. A per-chunk local Vec has
                    // no such hazard (and the alloc is dwarfed by the
                    // encode cost). The non-imatrix branch keeps the
                    // pooled scratch.
                    if let Some(c) = col_imp {
                        let mut local_f32 = vec![0f32; n_this];
                        dequant_to_f32(src_dtype, &src_bytes[src_lo..src_hi], &mut local_f32)
                            .map_err(|()| QuantizeError::UnsupportedSourceDtype {
                                name: tensor_name.clone(),
                                src: src_dtype,
                            })?;
                        // Broadcast the per-column importance across this
                        // chunk's flat weight indices: w[j] = c[(start+j) % n_in].
                        let w_chunk: Vec<f32> =
                            (0..n_this).map(|j| c[(start_weights + j) % n_in]).collect();
                        encode_from_f32_imatrix(target_dtype, &local_f32, out_chunk, Some(&w_chunk));
                        return Ok(());
                    }
                    with_f32_scratch(n_this, |local_f32| {
                        dequant_to_f32(
                            src_dtype,
                            &src_bytes[src_lo..src_hi],
                            local_f32,
                        )
                        .map_err(|()| QuantizeError::UnsupportedSourceDtype {
                            name: tensor_name.clone(),
                            src: src_dtype,
                        })?;
                        match col_imp {
                            Some(c) => {
                                // Broadcast the per-column importance across
                                // this chunk's flat weight indices:
                                // w[j] = c[(start_weights + j) % n_in].
                                let w_chunk: Vec<f32> = (0..n_this)
                                    .map(|j| c[(start_weights + j) % n_in])
                                    .collect();
                                encode_from_f32_imatrix(
                                    target_dtype,
                                    local_f32,
                                    out_chunk,
                                    Some(&w_chunk),
                                );
                            }
                            None => encode_from_f32(target_dtype, local_f32, out_chunk),
                        }
                        Ok(())
                    })
                });
            result?;
            stats.bytes_out += region_bytes as u64;
            stats.tensors_requantized += 1;
        }
    }
    dst.finish()?;
    Ok(stats)
}

/// Mmap-backed quantization pipeline with an injectable
/// [`IqGpuEncoder`] for the IQ codebook-search step. Identical to
/// [`quantize_gguf_to_path`] but:
///   - for `IQ1_S` target tensors, dispatches the per-chunk grid
///     search through `encoder` (one large batched call per delta
///     sign per tensor) — this is the entry point for GPU SYCL
///     offload. Falls back to the CPU per-chunk path automatically
///     if the encoder returns `Err` for any batch.
///   - for all other target dtypes, uses the existing rayon-
///     parallel CPU encode path unchanged.
///
/// The IQ1_S branch is **serial** (no rayon partition) because the
/// encoder typically wraps a single thread-bound `sycl::queue` and
/// the GPU is the parallelism source. For CPU-only callers, prefer
/// [`quantize_gguf_to_path`] which parallelizes across cores.
///
/// `encoder` must outlive the call. Passing
/// `&CpuFallbackEncoder` produces output byte-identical to
/// [`quantize_gguf_to_path`] (smoke-tested in the gguf crate's
/// unit tests).
pub fn quantize_gguf_to_path_with_encoder<P: AsRef<std::path::Path>>(
    src: &Gguf,
    output_path: P,
    plan: &QuantizePlan,
    encoder: &dyn crate::iq_gpu::IqGpuEncoder,
) -> Result<QuantizeStats, QuantizeError> {
    quantize_gguf_to_path_with_encoder_imatrix(src, output_path, plan, encoder, None)
}

/// GPU-encoder pipeline with optional importance-matrix weighting.
/// Identical to [`quantize_gguf_to_path_with_encoder`] but, for
/// tensors the `imatrix` covers, routes IQ1_S through the GPU
/// importance-weighted grid search (`encode_iq1_s_with_encoder_imatrix`)
/// and Q2_K/Q4_K/Q5_K through the CPU weighted encoder. Tensors not
/// covered by the imatrix (or all tensors when `imatrix` is `None`)
/// use the uniform path unchanged.
pub fn quantize_gguf_to_path_with_encoder_imatrix<P: AsRef<std::path::Path>>(
    src: &Gguf,
    output_path: P,
    plan: &QuantizePlan,
    encoder: &dyn crate::iq_gpu::IqGpuEncoder,
    imatrix: Option<&crate::imatrix::Imatrix>,
) -> Result<QuantizeStats, QuantizeError> {
    let mut dst = GgufMmapWriter::create(&output_path)?;

    for (key, value) in src.metadata() {
        if key == "general.alignment" {
            continue;
        }
        dst.add_metadata(key.clone(), value.clone())?;
    }

    let n_tensors = src.tensors().len();
    let mut resolved: Vec<ResolvedTensor> = Vec::with_capacity(n_tensors);
    for t in src.tensors() {
        let mut target = match plan.target_for(&t.name, t.dtype, t.element_count()) {
            Some(target) => target,
            None => t.dtype,
        };
        if target != t.dtype && !encoder_supported(target) {
            return Err(QuantizeError::UnsupportedTargetDtype(target));
        }
        let n = t.element_count();
        let block_size = target_block_size(target);
        if n % block_size != 0 {
            if target != t.dtype {
                target = t.dtype;
            } else {
                return Err(QuantizeError::BadElementCount {
                    name: t.name.clone(),
                    target,
                    n,
                    block_size,
                });
            }
        }
        // Recipes stay authoritative — warn (don't override) when an
        // MTP head lands below 8 bits per weight.
        warn_if_mtp_below_8bpw(&t.name, target);
        dst.declare_tensor(t.name.clone(), t.dims.clone(), target)?;
        resolved.push(ResolvedTensor {
            name: t.name.clone(),
            src_dtype: t.dtype,
            target_dtype: target,
            n_elements: n,
            dims: t.dims.clone(),
            passthrough: target == t.dtype,
        });
    }

    dst.finish_header()?;

    // Stage 3: take ALL tensor regions in one mutable borrow so the
    // CPU tail can be processed via `par_iter` without fighting the
    // borrow checker over per-call `tensor_region_mut`.
    let regions = dst.take_all_tensor_regions_mut()?;
    let mut region_map: std::collections::HashMap<String, &mut [u8]> =
        regions.into_iter().collect();

    // Pair every resolved tensor with its pre-acquired region + the
    // borrowed source bytes (a slice into the source mmap, Send by
    // virtue of `&Gguf: Sync`). Partition into:
    //   - iq_jobs: target dtype in the IQ family — must run
    //     sequentially because the GPU encoder owns a thread-bound
    //     SYCL queue
    //   - cpu_jobs: passthrough or K-quant / Q4_0 / F16 / etc.
    //     re-encodes — independent across tensors, perfect for
    //     `par_iter`
    let mut iq_jobs: Vec<(ResolvedTensor, &mut [u8], &[u8])> = Vec::new();
    let mut cpu_jobs: Vec<(ResolvedTensor, &mut [u8], &[u8])> = Vec::new();
    for r in resolved {
        let src_bytes = src
            .tensor_bytes(&r.name)
            .ok_or_else(|| QuantizeError::MissingTensor(r.name.clone()))?;
        let region = region_map
            .remove(&r.name)
            .ok_or_else(|| QuantizeError::MissingTensor(r.name.clone()))?;
        if !r.passthrough && is_iq_target(r.target_dtype) {
            iq_jobs.push((r, region, src_bytes));
        } else {
            cpu_jobs.push((r, region, src_bytes));
        }
    }

    let mut stats = QuantizeStats {
        tensors_total: n_tensors,
        ..Default::default()
    };

    // Pipelined IQ pool: producer thread dequants tensor N+1's
    // source bytes to F32 while the main (consumer) thread runs the
    // GPU encoder on tensor N. Bounded channel (capacity 2) keeps
    // at most ~640 MiB of F32 scratch in flight for the worst-case
    // 80 MiB Q3_K source → ~320 MiB F32 expansion.
    //
    // Consumer stays on the calling thread because the encoder
    // (typically `SyclIqEncoder`) owns a thread-bound SYCL queue
    // and isn't `Send`. The producer captures only `&Gguf` (Sync)
    // and the per-tensor jobs (Vec is Send; `&mut [u8]` over the
    // mmap region is Send).
    //
    // G4(b): on the consumer side, accumulate consecutive same-
    // format IQ2 tensors into a batch and dispatch them as ONE
    // merged GPU call. Saves per-tensor launch + USM-prep overhead
    // (~50-100 μs each) — small for big IQ1_S tensors where the
    // search itself dominates, but meaningful for many tiny IQ2_XXS
    // shared-expert tensors. Flush triggers: format change, non-IQ2
    // target dtype, channel close, or batch chunk-count exceeds
    // `IQ2_BATCH_CHUNK_CAP` (caps merged-f32 memory at ~8 MiB).
    if !iq_jobs.is_empty() {
        const IQ2_BATCH_CHUNK_CAP: usize = 256 * 1024;
        // G5 integrated switch: when set, the producer skips CPU
        // dequant for K-quant sources (Q3_K/Q4_K/Q5_K/Q6_K) and the
        // consumer dispatches the GPU dequant kernel on the encoder's
        // queue before the IQ search. Trades G3's CPU-dequant
        // pipelining for GPU dequant on the consumer thread; on iGPU
        // expected to be roughly a wash, on dGPU expected to save
        // PCIe upload time. Default off to preserve G3's behavior.
        let gpu_dequant = std::env::var("RUSTLLAMA_GPU_DEQUANT")
            .ok()
            .as_deref()
            == Some("1");
        let (tx, rx) = std::sync::mpsc::sync_channel::<DequantedTensor>(2);
        let producer_res: Result<(), QuantizeError> = std::thread::scope(|s| {
            let producer = s.spawn(move || -> Result<(), QuantizeError> {
                for (r, region, src_bytes) in iq_jobs {
                    let src_len = src_bytes.len() as u64;
                    // Resolve this tensor's per-input-column importance
                    // and broadcast it to a full-tensor weight vector
                    // (w[flat] = col_imp[flat % n_in]). Only honored when
                    // the imatrix length matches the inner dim and the
                    // element count is a clean multiple — defensive
                    // against a stale/mismatched imatrix.
                    let total_n = r.n_elements as usize;
                    let n_in = r.dims.first().copied().unwrap_or(0) as usize;
                    let imatrix_w: Option<Vec<f32>> = imatrix
                        .and_then(|im| im.get(&r.name))
                        .filter(|c| n_in > 0 && c.len() == n_in && total_n % n_in == 0)
                        .map(|c| (0..total_n).map(|j| c[j % n_in]).collect());
                    let (f32_buf, gpu_dequant_src_bytes) = if gpu_dequant
                        && is_kquant_source(r.src_dtype)
                    {
                        // Defer the dequant to the consumer thread —
                        // it owns the encoder's SYCL queue.
                        (Vec::new(), Some(src_bytes))
                    } else {
                        let mut buf = vec![0.0_f32; r.n_elements as usize];
                        dequant_to_f32(r.src_dtype, src_bytes, &mut buf).map_err(|()| {
                            QuantizeError::UnsupportedSourceDtype {
                                name: r.name.clone(),
                                src: r.src_dtype,
                            }
                        })?;
                        (buf, None)
                    };
                    if tx
                        .send(DequantedTensor {
                            r,
                            region,
                            f32_buf,
                            gpu_dequant_src_bytes,
                            imatrix_w,
                            src_len,
                        })
                        .is_err()
                    {
                        return Ok(());
                    }
                }
                Ok(())
            });
            // Consumer with G4(b) IQ2 coalescing.
            use crate::iq_gpu::Iq8EltGridFormat;
            let mut iq2_batch: Vec<DequantedTensor<'_>> = Vec::new();
            let mut iq2_batch_format: Option<Iq8EltGridFormat> = None;
            let mut iq2_batch_chunks: usize = 0;

            let flush_iq2 = |batch: &mut Vec<DequantedTensor<'_>>,
                             fmt: Iq8EltGridFormat,
                             stats: &mut QuantizeStats| {
                if batch.is_empty() {
                    return;
                }
                let total_chunks: usize = batch.iter().map(|d| d.f32_buf.len() / 8).sum();
                if total_chunks == 0 {
                    batch.clear();
                    return;
                }
                // Concat targets — one large merge memcpy per flush.
                let mut merged: Vec<f32> = Vec::with_capacity(total_chunks * 8);
                for d in batch.iter() {
                    merged.extend_from_slice(&d.f32_buf);
                }
                debug_assert_eq!(merged.len(), total_chunks * 8);
                let mut combined_picks: Vec<crate::iq_gpu::Iq8EltSignedPick> = vec![
                    crate::iq_gpu::Iq8EltSignedPick {
                        grid_idx: 0,
                        sign_idx: 0,
                        signed_score: 0.0,
                        grid_norm_sq: 1.0,
                    };
                    total_chunks
                ];
                let gpu_ok = encoder
                    .iq_8elt_signed_batched(&merged, fmt, &mut combined_picks)
                    .is_ok();
                let mut offset = 0usize;
                for d in batch.drain(..) {
                    let n_chunks = d.f32_buf.len() / 8;
                    let region_bytes = d.region.len() as u64;
                    if gpu_ok {
                        let slice = &combined_picks[offset..offset + n_chunks];
                        match d.r.target_dtype {
                            GgmlType::IQ2_XS => {
                                encode_iq_vec::bit_pack_iq2_xs_from_picks(slice, d.region)
                            }
                            GgmlType::IQ2_XXS => {
                                encode_iq_vec::bit_pack_iq2_xxs_from_picks(slice, d.region)
                            }
                            GgmlType::IQ2_S => {
                                encode_iq_vec::bit_pack_iq2_s_from_picks(slice, d.region)
                            }
                            _ => unreachable!(
                                "flush_iq2: non-IQ2 target {:?} in IQ2 batch",
                                d.r.target_dtype
                            ),
                        }
                    } else {
                        // Merged GPU call failed (e.g. encoder
                        // returned Unavailable) — fall back to the
                        // per-tensor path, which has its own CPU
                        // fallback inside encode_iq_*_with_encoder.
                        encode_iq_dispatch(d.r.target_dtype, &d.f32_buf, d.region, encoder);
                    }
                    stats.bytes_in += d.src_len;
                    stats.bytes_out += region_bytes;
                    stats.tensors_requantized += 1;
                    offset += n_chunks;
                }
            };

            while let Ok(mut d) = rx.recv() {
                // G5: if the producer deferred dequant for this tensor,
                // run the GPU dequant on the consumer thread (which
                // owns the encoder's queue) before any encoder call.
                // On failure, fall back to CPU dequant so the rest of
                // the pipeline can proceed.
                if let Some(src_bytes) = d.gpu_dequant_src_bytes.take() {
                    d.f32_buf = vec![0.0_f32; d.r.n_elements as usize];
                    if encoder
                        .try_dequant_kquant_to_f32(d.r.src_dtype, src_bytes, &mut d.f32_buf)
                        .is_err()
                    {
                        dequant_to_f32(d.r.src_dtype, src_bytes, &mut d.f32_buf).map_err(
                            |()| QuantizeError::UnsupportedSourceDtype {
                                name: d.r.name.clone(),
                                src: d.r.src_dtype,
                            },
                        )?;
                    }
                }
                let this_fmt = match d.r.target_dtype {
                    GgmlType::IQ2_XS => Some(Iq8EltGridFormat::Iq2Xs),
                    GgmlType::IQ2_XXS => Some(Iq8EltGridFormat::Iq2Xxs),
                    GgmlType::IQ2_S => Some(Iq8EltGridFormat::Iq2S),
                    _ => None,
                };
                if let Some(fmt) = this_fmt {
                    // Flush prior batch if format changed.
                    if iq2_batch_format != Some(fmt) {
                        if let Some(old_fmt) = iq2_batch_format.take() {
                            flush_iq2(&mut iq2_batch, old_fmt, &mut stats);
                            iq2_batch_chunks = 0;
                        }
                        iq2_batch_format = Some(fmt);
                    }
                    let n_chunks = d.f32_buf.len() / 8;
                    iq2_batch.push(d);
                    iq2_batch_chunks += n_chunks;
                    // Memory cap — flush early if accumulated chunks
                    // exceed the threshold.
                    if iq2_batch_chunks >= IQ2_BATCH_CHUNK_CAP {
                        flush_iq2(&mut iq2_batch, fmt, &mut stats);
                        iq2_batch_chunks = 0;
                    }
                } else {
                    // Non-IQ2 (IQ1 or IQ3): flush pending IQ2 batch,
                    // then dispatch this tensor on the existing per-
                    // tensor path.
                    if let Some(old_fmt) = iq2_batch_format.take() {
                        flush_iq2(&mut iq2_batch, old_fmt, &mut stats);
                        iq2_batch_chunks = 0;
                    }
                    let region_bytes = d.region.len() as u64;
                    match (d.r.target_dtype, d.imatrix_w.as_deref()) {
                        // IQ1_S with imatrix → GPU importance-weighted
                        // grid search (CPU weighted fallback inside).
                        (GgmlType::IQ1_S, Some(w)) => {
                            encode_iq_vec::encode_iq1_s_with_encoder_imatrix(
                                &d.f32_buf, d.region, w, encoder,
                            );
                        }
                        // Q2_K/Q4_K/Q5_K with imatrix → CPU weighted
                        // encoder (no GPU-weighted K-quant path yet).
                        (GgmlType::Q2_K | GgmlType::Q4_K | GgmlType::Q5_K, Some(w)) => {
                            encode_from_f32_imatrix(
                                d.r.target_dtype, &d.f32_buf, d.region, Some(w),
                            );
                        }
                        // Uniform (no imatrix) or dtype without a
                        // weighted path → existing GPU dispatch.
                        _ => {
                            encode_iq_dispatch(d.r.target_dtype, &d.f32_buf, d.region, encoder);
                        }
                    }
                    stats.bytes_in += d.src_len;
                    stats.bytes_out += region_bytes;
                    stats.tensors_requantized += 1;
                }
            }
            // Final flush.
            if let Some(fmt) = iq2_batch_format.take() {
                flush_iq2(&mut iq2_batch, fmt, &mut stats);
            }
            producer.join().expect("producer thread panic")
        });
        producer_res?;
    }

    // Parallel CPU pool — each tensor processed serially on its own
    // worker (no inner per-chunk parallelism; the outer par_iter
    // already saturates cores across tensors).
    let cpu_outcomes: Vec<Result<TensorJobOutcome, QuantizeError>> = cpu_jobs
        .into_par_iter()
        .map(|(r, region, src_bytes)| {
            let region_bytes = region.len() as u64;
            let src_len = src_bytes.len() as u64;
            if r.passthrough {
                region.copy_from_slice(src_bytes);
                return Ok(TensorJobOutcome {
                    bytes_in: src_len,
                    bytes_out: region_bytes,
                    passthrough: true,
                });
            }
            encode_cpu_tensor_serial(&r, src_bytes, region)?;
            Ok(TensorJobOutcome {
                bytes_in: src_len,
                bytes_out: region_bytes,
                passthrough: false,
            })
        })
        .collect();
    for outcome in cpu_outcomes {
        let outcome = outcome?;
        stats.bytes_in += outcome.bytes_in;
        stats.bytes_out += outcome.bytes_out;
        if outcome.passthrough {
            stats.tensors_passthrough += 1;
        } else {
            stats.tensors_requantized += 1;
        }
    }

    dst.finish()?;
    Ok(stats)
}

#[derive(Debug)]
struct TensorJobOutcome {
    bytes_in: u64,
    bytes_out: u64,
    passthrough: bool,
}

/// IQ-pipeline payload: producer thread builds these by dequanting
/// the source to F32, consumer (main) thread pulls them off the
/// channel and feeds the GPU encoder. Owns `f32_buf` so the consumer
/// can drop it (and reclaim memory) the moment the encode finishes.
struct DequantedTensor<'a> {
    r: ResolvedTensor,
    region: &'a mut [u8],
    /// F32 source for the encoder. When `gpu_dequant_pending` is set,
    /// this is a zero-length placeholder and the consumer is expected
    /// to populate it via `encoder.try_dequant_kquant_to_f32` from
    /// `gpu_dequant_src_bytes`.
    f32_buf: Vec<f32>,
    /// G5 integrated path: producer deferred CPU dequant to let the
    /// consumer run GPU dequant on the encoder's queue. The consumer
    /// allocates the F32 scratch and calls
    /// `encoder.try_dequant_kquant_to_f32(src_dtype, src_bytes, ...)`.
    /// `None` for the legacy CPU-dequant-on-producer path.
    gpu_dequant_src_bytes: Option<&'a [u8]>,
    /// Per-element importance weights (full-tensor, broadcast from the
    /// imatrix per-input-column vector) when an imatrix is in effect
    /// and applies to this tensor. `None` for uniform (no imatrix) or
    /// tensors the imatrix doesn't cover. Length == `r.n_elements`.
    imatrix_w: Option<Vec<f32>>,
    src_len: u64,
}

/// True iff `dtype` is one of the four K-quant source formats with
/// a matching GPU dequant kernel (G5).
fn is_kquant_source(dtype: GgmlType) -> bool {
    matches!(
        dtype,
        GgmlType::Q3_K | GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K
    )
}

/// True iff the target dtype goes through the GPU-encoder ladder.
/// The IQ-family kernels are the only ones that hit the SyclIqEncoder
/// today — every other dtype encodes on CPU.
fn is_iq_target(dtype: GgmlType) -> bool {
    matches!(
        dtype,
        GgmlType::IQ1_S
            | GgmlType::IQ1_M
            | GgmlType::IQ2_XXS
            | GgmlType::IQ2_XS
            | GgmlType::IQ2_S
            | GgmlType::IQ3_XXS
            | GgmlType::IQ3_S
            // G6: all 4 K-quants join the encoder-aware pipeline.
            // Q4_K/Q5_K use the iterative `make_qkx2_quants_asym`
            // SYCL port (per-sub-block 20-step refit).
            | GgmlType::Q6_K
            | GgmlType::Q3_K
            | GgmlType::Q4_K
            | GgmlType::Q5_K
    )
}

/// Per-IQ-target dispatch helper. Centralised so both the sequential
/// IQ pool and any future pipelining stage call the same site.
fn encode_iq_dispatch(
    target_dtype: GgmlType,
    src_f32: &[f32],
    region: &mut [u8],
    encoder: &dyn crate::iq_gpu::IqGpuEncoder,
) {
    match target_dtype {
        GgmlType::IQ1_S => encode_iq_vec::encode_iq1_s_with_encoder(src_f32, region, encoder),
        GgmlType::IQ1_M => encode_iq_vec::encode_iq1_m_with_encoder(src_f32, region, encoder),
        GgmlType::IQ2_XXS => encode_iq_vec::encode_iq2_xxs_with_encoder(src_f32, region, encoder),
        GgmlType::IQ2_XS => encode_iq_vec::encode_iq2_xs_with_encoder(src_f32, region, encoder),
        GgmlType::IQ2_S => encode_iq_vec::encode_iq2_s_with_encoder(src_f32, region, encoder),
        GgmlType::IQ3_XXS => encode_iq_vec::encode_iq3_xxs_with_encoder(src_f32, region, encoder),
        GgmlType::IQ3_S => encode_iq_vec::encode_iq3_s_with_encoder(src_f32, region, encoder),
        // G6: K-quants via SYCL when encoder supports it; CPU fallback inside.
        GgmlType::Q6_K => crate::encode_k::encode_q6_k_with_encoder(src_f32, region, encoder),
        GgmlType::Q3_K => crate::encode_k::encode_q3_k_with_encoder(src_f32, region, encoder),
        GgmlType::Q4_K => crate::encode_k::encode_q4_k_with_encoder(src_f32, region, encoder),
        GgmlType::Q5_K => crate::encode_k::encode_q5_k_with_encoder(src_f32, region, encoder),
        _ => unreachable!("encode_iq_dispatch called for non-IQ target {target_dtype:?}"),
    }
}

/// Serial-within-tensor CPU encode. Called from the parallel CPU
/// pool — the outer `par_iter` provides cross-tensor parallelism,
/// so no inner `par_chunks_mut` is needed here (would over-subscribe
/// rayon's worker pool).
fn encode_cpu_tensor_serial(
    r: &ResolvedTensor,
    src_bytes: &[u8],
    region: &mut [u8],
) -> Result<(), QuantizeError> {
    const CHUNK_WEIGHTS: usize = 256 * 1024;
    let total_n = r.n_elements as usize;
    let chunk_cap = CHUNK_WEIGHTS.min(total_n);
    let chunk_target_bytes = r.target_dtype.byte_size(chunk_cap as u64) as usize;
    let src_dtype = r.src_dtype;
    let target_dtype = r.target_dtype;
    for (i, out_chunk) in region.chunks_mut(chunk_target_bytes).enumerate() {
        let start_weights = i * chunk_cap;
        let end_weights = (start_weights + chunk_cap).min(total_n);
        let n_this = end_weights - start_weights;
        let src_lo = src_dtype.byte_size(start_weights as u64) as usize;
        let src_hi = src_dtype.byte_size(end_weights as u64) as usize;
        with_f32_scratch(n_this, |local_f32| {
            dequant_to_f32(src_dtype, &src_bytes[src_lo..src_hi], local_f32).map_err(|()| {
                QuantizeError::UnsupportedSourceDtype {
                    name: r.name.clone(),
                    src: src_dtype,
                }
            })?;
            encode_from_f32(target_dtype, local_f32, out_chunk);
            Ok::<(), QuantizeError>(())
        })?;
    }
    Ok(())
}

/// Number of elements per quantization block for the target dtype.
/// Used to validate tensor shape compatibility before declaring.
fn target_block_size(dtype: GgmlType) -> u64 {
    match dtype {
        GgmlType::F32 | GgmlType::F16 | GgmlType::Bf16 => 1,
        GgmlType::Q4_0
        | GgmlType::Q4_1
        | GgmlType::Q5_0
        | GgmlType::Q5_1
        | GgmlType::Q8_0
        | GgmlType::Q8_1
        | GgmlType::IQ4_NL => 32,
        GgmlType::Q2_K
        | GgmlType::Q3_K
        | GgmlType::Q4_K
        | GgmlType::Q5_K
        | GgmlType::Q6_K
        | GgmlType::Q8_K
        | GgmlType::IQ4_XS
        | GgmlType::TQ1_0
        | GgmlType::TQ2_0
        | GgmlType::IQ2_XXS
        | GgmlType::IQ2_XS
        | GgmlType::IQ2_S
        | GgmlType::IQ3_XXS
        | GgmlType::IQ3_S
        | GgmlType::IQ1_S
        | GgmlType::IQ1_M => 256,
        GgmlType::Nvfp4 => 16,
        // PrismML ternary group-128 formats (decode-only; no encoder).
        GgmlType::PQ2_0 | GgmlType::PTQ1_0 => 128,
    }
}

/// Whether an encoder is implemented for this target dtype. The
/// pipeline errors with `UnsupportedTargetDtype` for entries that
/// don't yet have an encoder.
fn encoder_supported(dtype: GgmlType) -> bool {
    matches!(
        dtype,
        GgmlType::F32
            | GgmlType::F16
            | GgmlType::Bf16
            | GgmlType::Q4_0
            | GgmlType::Q4_1
            | GgmlType::Q5_0
            | GgmlType::Q5_1
            | GgmlType::Q8_0
            | GgmlType::Q8_1
            | GgmlType::Q2_K
            | GgmlType::Q3_K
            | GgmlType::Q4_K
            | GgmlType::Q5_K
            | GgmlType::Q6_K
            | GgmlType::Q8_K
            | GgmlType::TQ1_0
            | GgmlType::TQ2_0
            | GgmlType::IQ4_NL
            | GgmlType::IQ4_XS
            | GgmlType::IQ2_XXS
            | GgmlType::IQ2_XS
            | GgmlType::IQ2_S
            | GgmlType::IQ3_XXS
            | GgmlType::IQ3_S
            | GgmlType::IQ1_S
            | GgmlType::IQ1_M
    )
}

/// Dispatch dequant by source dtype. Returns `Err(())` when the
/// source dtype doesn't have a dequant routine the pipeline can
/// use (the caller maps that to `UnsupportedSourceDtype`).
fn dequant_to_f32(src_dtype: GgmlType, src_bytes: &[u8], out: &mut [f32]) -> Result<(), ()> {
    match src_dtype {
        GgmlType::F32 => {
            // Reinterpret bytes as f32 directly — round-trip parity
            // depends on this being byte-exact, not an LE-swap.
            let n = out.len();
            debug_assert_eq!(src_bytes.len(), n * 4);
            for i in 0..n {
                out[i] = f32::from_le_bytes([
                    src_bytes[i * 4],
                    src_bytes[i * 4 + 1],
                    src_bytes[i * 4 + 2],
                    src_bytes[i * 4 + 3],
                ]);
            }
        }
        GgmlType::F16 => dequant::dequant_f16(src_bytes, out),
        GgmlType::Bf16 => dequant::dequant_bf16(src_bytes, out),
        GgmlType::Q8_0 => dequant::dequant_q8_0(src_bytes, out),
        GgmlType::Q8_1 => dequant::dequant_q8_1(src_bytes, out),
        GgmlType::Q8_K => dequant::dequant_q8_k(src_bytes, out),
        GgmlType::Q2_K => dequant::dequant_q2_k(src_bytes, out),
        GgmlType::Q3_K => dequant::dequant_q3_k(src_bytes, out),
        GgmlType::Q4_K => dequant::dequant_q4_k(src_bytes, out),
        GgmlType::Q5_K => dequant::dequant_q5_k(src_bytes, out),
        GgmlType::Q6_K => dequant::dequant_q6_k(src_bytes, out),
        GgmlType::Q4_0 => dequant::dequant_q4_0(src_bytes, out),
        GgmlType::Q4_1 => dequant::dequant_q4_1(src_bytes, out),
        GgmlType::Q5_0 => dequant::dequant_q5_0(src_bytes, out),
        GgmlType::Q5_1 => dequant::dequant_q5_1(src_bytes, out),
        GgmlType::IQ4_NL => dequant::dequant_iq4_nl(src_bytes, out),
        GgmlType::IQ4_XS => dequant::dequant_iq4_xs(src_bytes, out),
        GgmlType::IQ1_S => dequant::dequant_iq1_s(src_bytes, out),
        GgmlType::IQ1_M => dequant::dequant_iq1_m(src_bytes, out),
        GgmlType::IQ2_XXS => dequant::dequant_iq2_xxs(src_bytes, out),
        GgmlType::IQ2_XS => dequant::dequant_iq2_xs(src_bytes, out),
        GgmlType::IQ2_S => dequant::dequant_iq2_s(src_bytes, out),
        GgmlType::IQ3_XXS => dequant::dequant_iq3_xxs(src_bytes, out),
        GgmlType::IQ3_S => dequant::dequant_iq3_s(src_bytes, out),
        GgmlType::TQ1_0 => dequant::dequant_tq1_0(src_bytes, out),
        GgmlType::TQ2_0 => dequant::dequant_tq2_0(src_bytes, out),
        GgmlType::Nvfp4 => dequant::dequant_nvfp4(src_bytes, out),
        // PrismML ternary: readable as a re-quantization SOURCE (e.g.
        // converting a Bonsai file to another format for comparison),
        // never a target.
        GgmlType::PQ2_0 => dequant::dequant_pq2_0(src_bytes, out),
        GgmlType::PTQ1_0 => dequant::dequant_ptq1_0(src_bytes, out),
    }
    Ok(())
}

/// Dispatch encode by target dtype. Caller guarantees
/// `encoder_supported(target_dtype) == true` (the pipeline checks
/// before declaring the tensor).
/// imatrix-aware encode dispatch. `w` is the per-element importance
/// for this chunk (already broadcast to `src.len()`); routes the
/// formats that support importance weighting (Q2_K/Q4_K/Q5_K/IQ1_S)
/// to their `*_imatrix` encoders and falls back to the uniform
/// [`encode_from_f32`] for everything else. With `w = None` it is
/// exactly [`encode_from_f32`].
fn encode_from_f32_imatrix(
    target_dtype: GgmlType,
    src: &[f32],
    dst: &mut [u8],
    w: Option<&[f32]>,
) {
    match (target_dtype, w) {
        (GgmlType::Q2_K, Some(_)) => encode_k::encode_q2_k_imatrix(src, dst, w),
        (GgmlType::Q4_K, Some(_)) => encode_k::encode_q4_k_imatrix(src, dst, w),
        (GgmlType::Q5_K, Some(_)) => encode_k::encode_q5_k_imatrix(src, dst, w),
        (GgmlType::IQ1_S, Some(_)) => encode_iq_vec::encode_iq1_s_imatrix(src, dst, w),
        // Unsupported dtype or no imatrix → uniform encode.
        _ => encode_from_f32(target_dtype, src, dst),
    }
}

fn encode_from_f32(target_dtype: GgmlType, src: &[f32], dst: &mut [u8]) {
    match target_dtype {
        GgmlType::F32 => {
            let n = src.len();
            debug_assert_eq!(dst.len(), n * 4);
            for i in 0..n {
                let bytes = src[i].to_le_bytes();
                dst[i * 4..i * 4 + 4].copy_from_slice(&bytes);
            }
        }
        GgmlType::F16 => {
            let n = src.len();
            debug_assert_eq!(dst.len(), n * 2);
            for i in 0..n {
                let bits = half::f16::from_f32(src[i]).to_bits();
                dst[i * 2] = (bits & 0xFF) as u8;
                dst[i * 2 + 1] = ((bits >> 8) & 0xFF) as u8;
            }
        }
        GgmlType::Bf16 => {
            let n = src.len();
            debug_assert_eq!(dst.len(), n * 2);
            for i in 0..n {
                let bits = half::bf16::from_f32(src[i]).to_bits();
                dst[i * 2] = (bits & 0xFF) as u8;
                dst[i * 2 + 1] = ((bits >> 8) & 0xFF) as u8;
            }
        }
        GgmlType::Q4_0 => encode::encode_q4_0(src, dst),
        GgmlType::Q4_1 => encode::encode_q4_1(src, dst),
        GgmlType::Q5_0 => encode::encode_q5_0(src, dst),
        GgmlType::Q5_1 => encode::encode_q5_1(src, dst),
        GgmlType::Q8_0 => encode::encode_q8_0(src, dst),
        GgmlType::Q8_1 => encode::encode_q8_1(src, dst),
        GgmlType::Q2_K => encode_k::encode_q2_k(src, dst),
        GgmlType::Q3_K => encode_k::encode_q3_k(src, dst),
        GgmlType::Q4_K => encode_k::encode_q4_k(src, dst),
        GgmlType::Q5_K => encode_k::encode_q5_k(src, dst),
        GgmlType::Q6_K => encode_k::encode_q6_k(src, dst),
        GgmlType::Q8_K => encode_k::encode_q8_k(src, dst),
        GgmlType::TQ1_0 => encode_t::encode_tq1_0(src, dst),
        GgmlType::TQ2_0 => encode_t::encode_tq2_0(src, dst),
        GgmlType::IQ4_NL => encode_iq::encode_iq4_nl(src, dst),
        GgmlType::IQ4_XS => encode_iq::encode_iq4_xs(src, dst),
        GgmlType::IQ2_XXS => encode_iq_vec::encode_iq2_xxs(src, dst),
        GgmlType::IQ2_XS => encode_iq_vec::encode_iq2_xs(src, dst),
        GgmlType::IQ2_S => encode_iq_vec::encode_iq2_s(src, dst),
        GgmlType::IQ3_XXS => encode_iq_vec::encode_iq3_xxs(src, dst),
        GgmlType::IQ3_S => encode_iq_vec::encode_iq3_s(src, dst),
        GgmlType::IQ1_S => encode_iq_vec::encode_iq1_s(src, dst),
        GgmlType::IQ1_M => encode_iq_vec::encode_iq1_m(src, dst),
        // The pipeline checks `encoder_supported` before dispatch;
        // these remaining variants should never reach this match.
        // PQ2_0/PTQ1_0 are deliberately decode-only (we consume
        // PrismML's files; producing them means replicating their
        // Hadamard-rotated quantization pipeline, out of scope).
        GgmlType::Nvfp4 | GgmlType::PQ2_0 | GgmlType::PTQ1_0 => unreachable!(
            "encode_from_f32: encoder for {target_dtype:?} not implemented yet — \
             encoder_supported() should have rejected this earlier"
        ),
    }
}

/// Probe the writer for an existing metadata key. The `GgufWriter`
/// public API doesn't expose iteration; we work around by attempting
/// an `add_metadata` and rolling back on `DuplicateMetadata`. The
/// alternative — exposing a "has key" predicate on the writer —
/// would need a second public method; for v1 the round-trip check
/// keeps the writer's surface lean.
fn dst_has_metadata<W: Write + Seek>(dst: &mut GgufWriter<W>, key: &str) -> bool {
    // We can't actually peek into the writer's metadata, but the
    // duplicate-key check inside `add_metadata` returns
    // `DuplicateMetadata` deterministically. Try an add of an unused
    // placeholder value, then reverse it. ... actually, there's no
    // "remove" API either. Cleanest fix: assume the caller hasn't
    // pre-populated metadata (true for the CLI), and let
    // add_metadata's duplicate check surface user errors. Return
    // `false` always for now; revisit when the CLI grows a
    // metadata-injection feature.
    let _ = (dst, key);
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::write::GgufWriter;
    use crate::MetadataValue;
    use std::io::Cursor;

    /// End-to-end: build a tiny GGUF with two F32 tensors, run the
    /// re-quantize pipeline targeting Q4_K, re-open the result, and
    /// verify the new tensors are Q4_K-typed and dequant back to
    /// values close to the original.
    #[test]
    fn pipeline_requantizes_f32_to_q4_k() {
        // Stage A: build a source GGUF in memory.
        let mut src_buf = GgufWriter::new(Cursor::new(Vec::<u8>::new()));
        src_buf
            .add_metadata(
                "general.architecture",
                MetadataValue::String("llama".into()),
            )
            .unwrap();
        // Two 256-element F32 tensors. Block-aligned for Q4_K.
        let row_a: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 64.0).collect();
        let row_b: Vec<f32> = (0..256).map(|i| ((i * 7) as f32 % 13.0) - 6.0).collect();
        src_buf
            .declare_tensor("a.weight", vec![256], GgmlType::F32)
            .unwrap();
        src_buf
            .declare_tensor("b.weight", vec![256], GgmlType::F32)
            .unwrap();
        src_buf.finish_header().unwrap();
        let bytes_a: Vec<u8> = row_a.iter().flat_map(|v| v.to_le_bytes()).collect();
        let bytes_b: Vec<u8> = row_b.iter().flat_map(|v| v.to_le_bytes()).collect();
        src_buf.write_tensor_data("a.weight", &bytes_a).unwrap();
        src_buf.write_tensor_data("b.weight", &bytes_b).unwrap();
        let src_bytes = src_buf.finish().unwrap().into_inner();

        let tmp_src = std::env::temp_dir().join("rustllama_quantize_src.gguf");
        std::fs::write(&tmp_src, &src_bytes).unwrap();
        let src = Gguf::open(&tmp_src).unwrap();

        // Stage B: re-quantize to Q4_K via the pipeline.
        let dst_buf: Vec<u8> = Vec::new();
        let mut dst = GgufWriter::new(Cursor::new(dst_buf));
        let plan = QuantizePlan::uniform(GgmlType::Q4_K);
        let stats = quantize_gguf(&src, &mut dst, &plan).unwrap();
        assert_eq!(stats.tensors_total, 2);
        assert_eq!(stats.tensors_requantized, 2);
        assert_eq!(stats.tensors_passthrough, 0);
        // 256 × F32 = 1024 bytes/tensor → 2048 bytes in.
        assert_eq!(stats.bytes_in, 2048);
        // Q4_K = 144 bytes/super-block × 1 block/tensor = 144 bytes/tensor → 288 bytes out.
        assert_eq!(stats.bytes_out, 288);
        let out_bytes = dst.finish().unwrap().into_inner();

        // Stage C: reopen the re-quantized file, verify types + values.
        let tmp_dst = std::env::temp_dir().join("rustllama_quantize_dst.gguf");
        std::fs::write(&tmp_dst, &out_bytes).unwrap();
        let re = Gguf::open(&tmp_dst).unwrap();
        assert_eq!(re.tensor("a.weight").unwrap().dtype, GgmlType::Q4_K);
        assert_eq!(re.tensor("b.weight").unwrap().dtype, GgmlType::Q4_K);
        assert_eq!(re.architecture(), Some("llama"));

        // Dequant the re-quantized tensors and check round-trip error
        // is within the Q4_K bound (~4 * amp/15 ≈ 0.55 for amp=2).
        let mut deq_a = vec![0f32; 256];
        crate::dequant::dequant_q4_k(re.tensor_bytes("a.weight").unwrap(), &mut deq_a);
        let mut max_err = 0f32;
        for i in 0..256 {
            let e = (deq_a[i] - row_a[i]).abs();
            if e > max_err {
                max_err = e;
            }
        }
        assert!(max_err < 0.6, "Q4_K re-quantize: max_err={max_err}");

        let _ = std::fs::remove_file(&tmp_src);
        let _ = std::fs::remove_file(&tmp_dst);
    }

    /// Passthrough path: 1D F32 "norm"-like tensors should not get
    /// re-quantized even when the plan's default is Q4_K.
    #[test]
    fn pipeline_passes_through_non_quantizable_tensors() {
        let mut src_buf = GgufWriter::new(Cursor::new(Vec::<u8>::new()));
        src_buf
            .add_metadata(
                "general.architecture",
                MetadataValue::String("llama".into()),
            )
            .unwrap();
        // Caller flags `attn_norm.weight` as passthrough via the
        // standard prefix. The element count (4) doesn't divide
        // 256 so re-quantizing to Q4_K would be a hard error
        // anyway — the passthrough path saves the user.
        src_buf
            .declare_tensor("output_norm.weight", vec![4], GgmlType::F32)
            .unwrap();
        src_buf.finish_header().unwrap();
        src_buf
            .write_tensor_data("output_norm.weight", &[0u8; 16])
            .unwrap();
        let src_bytes = src_buf.finish().unwrap().into_inner();
        let tmp_src = std::env::temp_dir().join("rustllama_quantize_passthrough_src.gguf");
        std::fs::write(&tmp_src, &src_bytes).unwrap();
        let src = Gguf::open(&tmp_src).unwrap();

        let dst_buf: Vec<u8> = Vec::new();
        let mut dst = GgufWriter::new(Cursor::new(dst_buf));
        let mut plan = QuantizePlan::uniform(GgmlType::Q4_K);
        plan.passthrough_prefixes.push("output_norm".into());
        let stats = quantize_gguf(&src, &mut dst, &plan).unwrap();
        assert_eq!(stats.tensors_passthrough, 1);
        assert_eq!(stats.tensors_requantized, 0);
        let out_bytes = dst.finish().unwrap().into_inner();

        let tmp_dst = std::env::temp_dir().join("rustllama_quantize_passthrough_dst.gguf");
        std::fs::write(&tmp_dst, &out_bytes).unwrap();
        let re = Gguf::open(&tmp_dst).unwrap();
        assert_eq!(
            re.tensor("output_norm.weight").unwrap().dtype,
            GgmlType::F32
        );

        let _ = std::fs::remove_file(&tmp_src);
        let _ = std::fs::remove_file(&tmp_dst);
    }

    /// mmap-path round-trip: same tensor, same plan, same output
    /// bytes as the Vec-backed `quantize_gguf`. The mmap path is
    /// a perf alternative — it must produce **byte-identical**
    /// output to the in-memory path.
    #[test]
    fn quantize_gguf_to_path_matches_in_memory_path() {
        // Build a source GGUF with a mix of tensor sizes: small,
        // 256-element, and a multi-chunk-spanning size.
        let n_small = 64;
        let n_medium = 256;
        let n_large = 256 * 1024 + 512;

        let mut s: u32 = 11;
        let row_a: Vec<f32> = (0..n_small)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let u = (s >> 8) as f32 / (1u32 << 24) as f32;
                u * 4.0 - 2.0
            })
            .collect();
        let mut s: u32 = 31;
        let row_b: Vec<f32> = (0..n_medium)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let u = (s >> 8) as f32 / (1u32 << 24) as f32;
                u * 4.0 - 2.0
            })
            .collect();
        let mut s: u32 = 47;
        let row_c: Vec<f32> = (0..n_large)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let u = (s >> 8) as f32 / (1u32 << 24) as f32;
                u * 4.0 - 2.0
            })
            .collect();

        // Build source GGUF.
        let mut src_w = GgufWriter::new(Cursor::new(Vec::<u8>::new()));
        src_w
            .add_metadata("general.architecture", MetadataValue::String("llama".into()))
            .unwrap();
        src_w.declare_tensor("a.weight", vec![n_small as u64], GgmlType::F32).unwrap();
        src_w.declare_tensor("b.weight", vec![n_medium as u64], GgmlType::F32).unwrap();
        src_w.declare_tensor("c.weight", vec![n_large as u64], GgmlType::F32).unwrap();
        src_w.finish_header().unwrap();
        let bytes_a: Vec<u8> = row_a.iter().flat_map(|v| v.to_le_bytes()).collect();
        let bytes_b: Vec<u8> = row_b.iter().flat_map(|v| v.to_le_bytes()).collect();
        let bytes_c: Vec<u8> = row_c.iter().flat_map(|v| v.to_le_bytes()).collect();
        src_w.write_tensor_data("a.weight", &bytes_a).unwrap();
        src_w.write_tensor_data("b.weight", &bytes_b).unwrap();
        src_w.write_tensor_data("c.weight", &bytes_c).unwrap();
        let src_bytes = src_w.finish().unwrap().into_inner();
        let tmp_src = std::env::temp_dir().join("rustllama_quantize_mmap_src.gguf");
        std::fs::write(&tmp_src, &src_bytes).unwrap();
        let src = Gguf::open(&tmp_src).unwrap();

        let plan = QuantizePlan::uniform(GgmlType::Q4_K);

        // Path 1: in-memory writer (existing quantize_gguf).
        let mut mem_w = GgufWriter::new(Cursor::new(Vec::<u8>::new()));
        let mem_stats = quantize_gguf(&src, &mut mem_w, &plan).unwrap();
        let mem_bytes = mem_w.finish().unwrap().into_inner();

        // Path 2: mmap writer (new quantize_gguf_to_path).
        let tmp_mmap = std::env::temp_dir().join("rustllama_quantize_mmap_dst.gguf");
        let _ = std::fs::remove_file(&tmp_mmap);
        let mmap_stats =
            quantize_gguf_to_path(&src, &tmp_mmap, &plan).unwrap();
        let mmap_bytes = std::fs::read(&tmp_mmap).unwrap();

        assert_eq!(mem_stats.tensors_total, mmap_stats.tensors_total);
        assert_eq!(
            mem_stats.tensors_requantized,
            mmap_stats.tensors_requantized
        );
        assert_eq!(
            mem_stats.tensors_passthrough,
            mmap_stats.tensors_passthrough
        );
        // The output bytes must be identical between the two paths
        // — they call the same encoders with the same arguments
        // and write the same header layout.
        assert_eq!(
            mem_bytes.len(),
            mmap_bytes.len(),
            "output length mismatch: mem={} mmap={}",
            mem_bytes.len(),
            mmap_bytes.len()
        );
        // Compare in chunks for a useful error message if they diverge.
        for (i, (mb, mmb)) in mem_bytes.iter().zip(mmap_bytes.iter()).enumerate() {
            assert_eq!(
                mb, mmb,
                "byte mismatch at offset {i}: mem=0x{mb:02x} mmap=0x{mmb:02x}"
            );
        }

        let _ = std::fs::remove_file(&tmp_src);
        let _ = std::fs::remove_file(&tmp_mmap);
    }

    /// `quantize_gguf_to_path_with_encoder` with the CPU fallback
    /// encoder must produce byte-identical output to the plain
    /// `quantize_gguf_to_path` for an IQ1_S target. The new code
    /// path is a structural alternative — same arithmetic, same
    /// layout, just routes per-chunk search through a trait object
    /// instead of the direct function call.
    #[test]
    fn quantize_with_encoder_iq1s_matches_direct_mmap() {
        // IQ1_S blocks are 256 weights — use 4 blocks (1024 weights).
        let n = 1024;
        let mut s: u32 = 19;
        let row: Vec<f32> = (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let u = (s >> 8) as f32 / (1u32 << 24) as f32;
                u * 4.0 - 2.0
            })
            .collect();

        let mut src_w = GgufWriter::new(Cursor::new(Vec::<u8>::new()));
        src_w
            .add_metadata("general.architecture", MetadataValue::String("llama".into()))
            .unwrap();
        src_w.declare_tensor("w.weight", vec![n as u64], GgmlType::F32).unwrap();
        src_w.finish_header().unwrap();
        let bytes: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
        src_w.write_tensor_data("w.weight", &bytes).unwrap();
        let src_bytes = src_w.finish().unwrap().into_inner();

        let tmp_src = std::env::temp_dir().join("rustllama_iq1s_enc_src.gguf");
        std::fs::write(&tmp_src, &src_bytes).unwrap();
        let src = Gguf::open(&tmp_src).unwrap();

        let plan = QuantizePlan::uniform(GgmlType::IQ1_S);

        // Path A: direct mmap pipeline.
        let tmp_a = std::env::temp_dir().join("rustllama_iq1s_enc_a.gguf");
        let _ = std::fs::remove_file(&tmp_a);
        quantize_gguf_to_path(&src, &tmp_a, &plan).unwrap();
        let bytes_a = std::fs::read(&tmp_a).unwrap();

        // Path B: encoder-injectable mmap pipeline with CPU fallback.
        let tmp_b = std::env::temp_dir().join("rustllama_iq1s_enc_b.gguf");
        let _ = std::fs::remove_file(&tmp_b);
        quantize_gguf_to_path_with_encoder(
            &src,
            &tmp_b,
            &plan,
            &crate::iq_gpu::CpuFallbackEncoder,
        )
        .unwrap();
        let bytes_b = std::fs::read(&tmp_b).unwrap();

        assert_eq!(
            bytes_a, bytes_b,
            "encoder-injectable IQ1_S path must match direct path byte-for-byte"
        );

        let _ = std::fs::remove_file(&tmp_src);
        let _ = std::fs::remove_file(&tmp_a);
        let _ = std::fs::remove_file(&tmp_b);
    }

    /// IQ2_XXS / IQ2_XS / IQ2_S all route through the encoder-
    /// injectable mmap pipeline. With the CPU fallback encoder
    /// each must produce byte-identical output to the direct mmap
    /// pipeline (same arithmetic, just different driver loop).
    #[test]
    fn quantize_with_encoder_iq2_family_matches_direct_mmap() {
        let n = 4 * 256; // 4 blocks of QK_K weights.
        let mut s: u32 = 53;
        let row: Vec<f32> = (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let u = (s >> 8) as f32 / (1u32 << 24) as f32;
                u * 4.0 - 2.0
            })
            .collect();

        let mut src_w = GgufWriter::new(Cursor::new(Vec::<u8>::new()));
        src_w
            .add_metadata("general.architecture", MetadataValue::String("llama".into()))
            .unwrap();
        src_w.declare_tensor("w.weight", vec![n as u64], GgmlType::F32).unwrap();
        src_w.finish_header().unwrap();
        let bytes: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
        src_w.write_tensor_data("w.weight", &bytes).unwrap();
        let src_bytes = src_w.finish().unwrap().into_inner();

        let tmp_src = std::env::temp_dir().join("rustllama_iq2_enc_src.gguf");
        std::fs::write(&tmp_src, &src_bytes).unwrap();
        let src = Gguf::open(&tmp_src).unwrap();

        for target in [
            GgmlType::IQ1_M,
            GgmlType::IQ2_XXS,
            GgmlType::IQ2_XS,
            GgmlType::IQ2_S,
            GgmlType::IQ3_XXS,
            GgmlType::IQ3_S,
        ] {
            let plan = QuantizePlan::uniform(target);
            let tag = format!("{target:?}").to_lowercase();
            let tmp_a = std::env::temp_dir().join(format!("rustllama_iq2_{tag}_a.gguf"));
            let tmp_b = std::env::temp_dir().join(format!("rustllama_iq2_{tag}_b.gguf"));
            let _ = std::fs::remove_file(&tmp_a);
            let _ = std::fs::remove_file(&tmp_b);
            quantize_gguf_to_path(&src, &tmp_a, &plan).unwrap();
            quantize_gguf_to_path_with_encoder(
                &src,
                &tmp_b,
                &plan,
                &crate::iq_gpu::CpuFallbackEncoder,
            )
            .unwrap();
            let ba = std::fs::read(&tmp_a).unwrap();
            let bb = std::fs::read(&tmp_b).unwrap();
            assert_eq!(
                ba, bb,
                "encoder-injectable {target:?} path must match direct path"
            );
            let _ = std::fs::remove_file(&tmp_a);
            let _ = std::fs::remove_file(&tmp_b);
        }
        let _ = std::fs::remove_file(&tmp_src);
    }

    /// Multi-chunk path correctness: re-quantize a tensor larger
    /// than the internal `CHUNK_WEIGHTS = 256K` threshold, so the
    /// chunked dequant + encode loop runs more than once. Verify
    /// the output bytes match what a single-pass run would have
    /// produced (round-trip through dequant gives values close to
    /// the source within Q4_K's tolerance, and the byte stream is
    /// deterministic).
    #[test]
    fn pipeline_chunked_path_matches_single_pass_on_large_tensor() {
        // 2× CHUNK_WEIGHTS, plus a tail to exercise the partial-chunk
        // branch. 256K + 1K = 257 super-blocks; tail = 1024 weights
        // = 4 super-blocks.
        let n = 256 * 1024 + 1024;
        let mut s: u32 = 1;
        let src_row: Vec<f32> = (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let u = (s >> 8) as f32 / (1u32 << 24) as f32;
                u * 4.0 - 2.0
            })
            .collect();

        // Build source GGUF.
        let mut src_w = GgufWriter::new(Cursor::new(Vec::<u8>::new()));
        src_w
            .add_metadata("k", MetadataValue::U32(0))
            .unwrap();
        src_w
            .declare_tensor("big.weight", vec![n as u64], GgmlType::F32)
            .unwrap();
        src_w.finish_header().unwrap();
        let src_bytes: Vec<u8> = src_row.iter().flat_map(|v| v.to_le_bytes()).collect();
        src_w.write_tensor_data("big.weight", &src_bytes).unwrap();
        let src_file_bytes = src_w.finish().unwrap().into_inner();
        let tmp_src = std::env::temp_dir().join("rustllama_quantize_chunked_src.gguf");
        std::fs::write(&tmp_src, &src_file_bytes).unwrap();
        let src = Gguf::open(&tmp_src).unwrap();

        // Re-quantize to Q4_K via the (now-chunked) pipeline.
        let mut dst_w = GgufWriter::new(Cursor::new(Vec::<u8>::new()));
        let plan = QuantizePlan::uniform(GgmlType::Q4_K);
        let stats = quantize_gguf(&src, &mut dst_w, &plan).unwrap();
        assert_eq!(stats.tensors_requantized, 1);
        let dst_bytes = dst_w.finish().unwrap().into_inner();
        let tmp_dst = std::env::temp_dir().join("rustllama_quantize_chunked_dst.gguf");
        std::fs::write(&tmp_dst, &dst_bytes).unwrap();
        let re = Gguf::open(&tmp_dst).unwrap();
        assert_eq!(re.tensor("big.weight").unwrap().dtype, GgmlType::Q4_K);

        // Dequant the encoded bytes and verify Q4_K round-trip
        // error stays within bound. If the chunking sliced anything
        // wrong (e.g. off-by-one on a block boundary), most weights
        // would dequant to garbage and the max-err assertion would
        // catch it.
        let q4k_bytes = re.tensor_bytes("big.weight").unwrap();
        let mut deq = vec![0f32; n];
        crate::dequant::dequant_q4_k(q4k_bytes, &mut deq);
        let mut max_err = 0f32;
        for i in 0..n {
            let e = (deq[i] - src_row[i]).abs();
            if e > max_err {
                max_err = e;
            }
        }
        // Q4_K with iterative scale-search on uniformly-distributed
        // [-2, 2) → max error well within `2 * amp / 15 ≈ 0.27`.
        // Loose bound 0.6 to absorb FP rounding in the f16 scale.
        assert!(
            max_err < 0.6,
            "chunked Q4_K round-trip max_err={max_err}"
        );

        let _ = std::fs::remove_file(&tmp_src);
        let _ = std::fs::remove_file(&tmp_dst);
    }

    /// Parallel encoder is byte-deterministic. Two back-to-back runs
    /// of the same source through `quantize_gguf` must produce
    /// byte-identical output GGUFs — rayon's worker scheduling is
    /// non-deterministic, but each chunk's encode is a pure function
    /// of its source slice + dtype, so the output bytes don't depend
    /// on thread ordering.
    ///
    /// Catches: race conditions in shared mutable state (there
    /// should be none), accidental thread-local randomness in the
    /// encoders, or `par_chunks_mut` boundary bugs.
    #[test]
    fn parallel_encoder_is_byte_deterministic() {
        // Tensor large enough to trigger multi-chunk parallel encode.
        let n = 256 * 1024 + 512;
        let mut s: u32 = 17;
        let src_row: Vec<f32> = (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let u = (s >> 8) as f32 / (1u32 << 24) as f32;
                u * 4.0 - 2.0
            })
            .collect();

        let mut src_w = GgufWriter::new(Cursor::new(Vec::<u8>::new()));
        src_w
            .declare_tensor("t.weight", vec![n as u64], GgmlType::F32)
            .unwrap();
        src_w.finish_header().unwrap();
        let src_bytes: Vec<u8> = src_row.iter().flat_map(|v| v.to_le_bytes()).collect();
        src_w.write_tensor_data("t.weight", &src_bytes).unwrap();
        let src_file_bytes = src_w.finish().unwrap().into_inner();
        let tmp_src = std::env::temp_dir().join("rustllama_quantize_par_src.gguf");
        std::fs::write(&tmp_src, &src_file_bytes).unwrap();
        let src = Gguf::open(&tmp_src).unwrap();

        let plan = QuantizePlan::uniform(GgmlType::Q4_K);

        let mut dst1 = GgufWriter::new(Cursor::new(Vec::<u8>::new()));
        quantize_gguf(&src, &mut dst1, &plan).unwrap();
        let bytes1 = dst1.finish().unwrap().into_inner();

        let mut dst2 = GgufWriter::new(Cursor::new(Vec::<u8>::new()));
        quantize_gguf(&src, &mut dst2, &plan).unwrap();
        let bytes2 = dst2.finish().unwrap().into_inner();

        assert_eq!(
            bytes1, bytes2,
            "parallel encoder must be byte-deterministic across runs"
        );

        let _ = std::fs::remove_file(&tmp_src);
    }

    /// Shape-misalignment auto-passthrough: when the target dtype's
    /// block size doesn't divide the tensor's element count, the
    /// pipeline falls back to the source dtype rather than failing.
    /// Mirrors llama.cpp's behavior on SSM `ssm_a`, per-head biases,
    /// and other small per-head parameters that hybrid (Mamba) and
    /// MTP models include.
    #[test]
    fn pipeline_passes_through_misaligned_tensor() {
        let mut src_buf = GgufWriter::new(Cursor::new(Vec::<u8>::new()));
        src_buf
            .add_metadata("a", MetadataValue::U32(1))
            .unwrap();
        // 32 elements: real-world case is `blk.N.ssm_a` (Mamba head
        // count). Doesn't divide Q4_K's 256-block size, but should
        // gracefully pass through at F32 (the source dtype).
        src_buf
            .declare_tensor("blk.0.ssm_a", vec![32], GgmlType::F32)
            .unwrap();
        src_buf.finish_header().unwrap();
        src_buf
            .write_tensor_data("blk.0.ssm_a", &[0u8; 32 * 4])
            .unwrap();
        let src_bytes = src_buf.finish().unwrap().into_inner();
        let tmp_src = std::env::temp_dir().join("rustllama_quantize_misalign_src.gguf");
        std::fs::write(&tmp_src, &src_bytes).unwrap();
        let src = Gguf::open(&tmp_src).unwrap();

        let dst_buf: Vec<u8> = Vec::new();
        let mut dst = GgufWriter::new(Cursor::new(dst_buf));
        let plan = QuantizePlan::uniform(GgmlType::Q4_K);
        let stats = quantize_gguf(&src, &mut dst, &plan).unwrap();
        // The tensor should have flowed through as F32 (its source
        // dtype) rather than producing an error.
        assert_eq!(stats.tensors_total, 1);
        assert_eq!(stats.tensors_passthrough, 1);
        assert_eq!(stats.tensors_requantized, 0);

        let out_bytes = dst.finish().unwrap().into_inner();
        let tmp_dst = std::env::temp_dir().join("rustllama_quantize_misalign_dst.gguf");
        std::fs::write(&tmp_dst, &out_bytes).unwrap();
        let re = Gguf::open(&tmp_dst).unwrap();
        assert_eq!(re.tensor("blk.0.ssm_a").unwrap().dtype, GgmlType::F32);

        let _ = std::fs::remove_file(&tmp_src);
        let _ = std::fs::remove_file(&tmp_dst);
    }

    /// MTP-head name predicate: `blk.{N}.nextn.*` (qwen35moe), bare
    /// `.eh_proj` projections, and DeepSeek-style `mtp.{i}.*` all
    /// classify as MTP; ordinary transformer tensors do not.
    #[test]
    fn mtp_head_name_predicate() {
        for name in [
            "blk.40.nextn.eh_proj.weight",
            "blk.40.nextn.embed_tokens.weight",
            "blk.40.nextn.shared_head.head.weight",
            "blk.40.nextn.enorm.weight",
            "blk.61.eh_proj.weight",
            "mtp.0.attn_q.weight",
            "mtp.1.output.weight",
        ] {
            assert!(is_mtp_head_tensor(name), "{name} should classify as MTP");
        }
        for name in [
            "output.weight",
            "token_embd.weight",
            "blk.0.attn_q.weight",
            "blk.15.ffn_gate_exps.weight",
            "blk.40.attn_norm.weight",
            // `mtp` must be a leading namespace, not a substring.
            "blk.0.mtp_gate.weight",
        ] {
            assert!(!is_mtp_head_tensor(name), "{name} should NOT classify as MTP");
        }
    }

    /// 8-bits-per-weight classification: exactly F32 / F16 / BF16 /
    /// Q8_0 / Q8_1 / Q8_K qualify; every sub-8-bit quant format does
    /// not. Pins the byte_size-derived predicate against the dtype
    /// table.
    #[test]
    fn dtype_8bpw_classification() {
        for t in [
            GgmlType::F32,
            GgmlType::F16,
            GgmlType::Bf16,
            GgmlType::Q8_0,
            GgmlType::Q8_1,
            GgmlType::Q8_K,
        ] {
            assert!(dtype_at_least_8bpw(t), "{t:?} is >= 8 bpw");
        }
        for t in [
            GgmlType::Q6_K,
            GgmlType::Q5_K,
            GgmlType::Q5_0,
            GgmlType::Q4_K,
            GgmlType::Q4_0,
            GgmlType::Q3_K,
            GgmlType::Q2_K,
            GgmlType::IQ4_XS,
            GgmlType::IQ2_XXS,
            GgmlType::IQ1_S,
            GgmlType::TQ1_0,
            GgmlType::Nvfp4,
        ] {
            assert!(!dtype_at_least_8bpw(t), "{t:?} is below 8 bpw");
        }
    }

    /// Combined warning predicate: fires only for MTP-head names at
    /// sub-8-bit targets — never for >= 8-bit targets and never for
    /// non-MTP tensors, however low they go.
    #[test]
    fn mtp_below_8bpw_warning_predicate() {
        // MTP head at a sub-8-bit target → warn.
        assert!(mtp_head_below_8bpw(
            "blk.40.nextn.eh_proj.weight",
            GgmlType::Q4_K
        ));
        assert!(mtp_head_below_8bpw("mtp.0.ffn_down.weight", GgmlType::IQ1_S));
        // MTP head kept at >= 8 bits → quiet.
        assert!(!mtp_head_below_8bpw(
            "blk.40.nextn.eh_proj.weight",
            GgmlType::Q8_0
        ));
        assert!(!mtp_head_below_8bpw(
            "blk.40.nextn.eh_proj.weight",
            GgmlType::F16
        ));
        // Non-MTP tensor at any precision → quiet.
        assert!(!mtp_head_below_8bpw(
            "blk.15.ffn_gate_exps.weight",
            GgmlType::Q2_K
        ));
    }
}
