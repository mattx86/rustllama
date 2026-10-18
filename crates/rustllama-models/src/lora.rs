//! P11 — LoRA / multi-LoRA adapters.
//!
//! Clean-room implementation of the public ggml LoRA-adapter GGUF format
//! (NOT llama.cpp code). An adapter GGUF carries
//! `general.architecture = "<base-arch>"`, `adapter.type = "lora"`, a model-
//! wide `adapter.lora.alpha` (f32), and, per adapted base tensor, a low-rank
//! factor pair:
//!
//! - `<base>.weight.lora_a` with ggml `ne = [in, rank]`  → matrix `A [rank,in]`
//! - `<base>.weight.lora_b` with ggml `ne = [rank, out]` → matrix `B [out,rank]`
//!
//! The effective update to a base weight `W [out,in]` is
//! `W' = W + s · B·A` with scale `s = alpha / rank`. Two application modes:
//!
//! **Phase 1 — merge-at-load** ([`merge_tensor_in_place`] /
//! [`crate::llama_arch::LlamaWeights::merge_lora`]): for an F32/F16/BF16 base,
//! dequantize to F32, add `s·B·A` once, and keep the merged weight. Bit-exact
//! to a merged checkpoint; the quantized base is left untouched (its delta is
//! applied at dispatch instead).
//!
//! **Phase 2 — delta-apply at dispatch** ([`apply_delta`]): keep the packed
//! base resident and wrap the matvec as `y = W·x + s·B·(A·x)`. The base path
//! is byte-identical when no adapter is active — the hook is a single
//! `thread_local` borrow + `Option` check before it touches `out`.
//!
//! **Phase 3 — multi-LoRA** ([`LoraRegistry`] + [`set_active_lora`] /
//! [`RowAdapters`]): hold N adapters resident keyed by name. Two selection
//! modes feed the SAME [`apply_delta`] dispatch hook:
//!   - **single-active** ([`ACTIVE_LORA`] via [`scope_active_lora`]): one
//!     adapter for the whole forward — the contiguous per-request path, where
//!     one request = one thread = one adapter.
//!   - **per-row** ([`ROW_ADAPTERS`] via [`scope_row_adapters`]): a table
//!     mapping each concatenated row to its adapter — the batched / ragged
//!     path, where ONE fused forward serves several slots that each named a
//!     different (or no) adapter. The batched matvec dispatch drives
//!     [`set_current_lora_row`] per row so row `r`'s delta is
//!     `ROW_ADAPTERS[r]`.
//! Either mode `None` ⇒ the base dispatch is byte-identical.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::Arc;

use rustllama_gguf::{GgmlType, Gguf};
use rustllama_tensor::{Dtype, Tensor};

/// One adapted tensor's low-rank factors, dequantized to F32.
///
/// Row-major convention (matches the ggml adapter layout this repo reads):
/// `a[k*in + c]` = `A[k][c]` (`rank` rows of length `in`), and
/// `b[r*rank + k]` = `B[r][k]` (`out` rows of length `rank`).
#[derive(Debug, Clone)]
pub struct LoraFactors {
    pub rank: usize,
    pub in_dim: usize,
    pub out_dim: usize,
    /// Effective scale `alpha / rank` folded in at apply/merge time.
    pub scale: f32,
    /// `A` down-projection, `[rank * in]` row-major.
    pub a: Vec<f32>,
    /// `B` up-projection, `[out * rank]` row-major.
    pub b: Vec<f32>,
}

impl LoraFactors {
    /// Phase 2: add `scale · B·(A·x)` into `out` (an additive low-rank
    /// correction around the base matvec `y = W·x`). `x` has length `in`,
    /// `out` length `out`.
    #[inline]
    pub fn apply_delta(&self, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(x.len(), self.in_dim, "lora apply_delta: x width");
        debug_assert_eq!(out.len(), self.out_dim, "lora apply_delta: out width");
        // u = A·x  (rank-length intermediate).
        let mut u = vec![0.0f32; self.rank];
        for (k, uk) in u.iter_mut().enumerate() {
            let row = &self.a[k * self.in_dim..(k + 1) * self.in_dim];
            let mut acc = 0.0f32;
            for c in 0..self.in_dim {
                acc += row[c] * x[c];
            }
            *uk = acc;
        }
        // out += scale · B·u.
        for (r, o) in out.iter_mut().enumerate() {
            let row = &self.b[r * self.rank..(r + 1) * self.rank];
            let mut acc = 0.0f32;
            for k in 0..self.rank {
                acc += row[k] * u[k];
            }
            *o += self.scale * acc;
        }
    }

    /// Phase 1: add `scale · B·A` into the dequantized base matrix `base`
    /// (row-major `[out, in]`: `base[r*in + c]`). Summation order (k ascending)
    /// matches the hand-applied reference so the merge is bit-reproducible.
    pub fn add_delta_into(&self, base: &mut [f32]) {
        debug_assert_eq!(
            base.len(),
            self.out_dim * self.in_dim,
            "lora merge: base size"
        );
        for r in 0..self.out_dim {
            let brow = &self.b[r * self.rank..(r + 1) * self.rank];
            for c in 0..self.in_dim {
                let mut acc = 0.0f32;
                for k in 0..self.rank {
                    acc += brow[k] * self.a[k * self.in_dim + c];
                }
                base[r * self.in_dim + c] += self.scale * acc;
            }
        }
    }
}

/// A loaded LoRA adapter: per-base-tensor factors keyed by the base tensor's
/// GGUF name (e.g. `blk.0.attn_q.weight`).
#[derive(Debug, Clone)]
pub struct LoraAdapter {
    /// Selection name (the registry key / request `base:adapter` suffix).
    pub name: String,
    /// `general.architecture` the adapter targets (informational).
    pub arch: Option<String>,
    /// Model-wide `adapter.lora.alpha` (`None` ⇒ per-tensor `alpha = rank`,
    /// i.e. scale 1.0).
    pub alpha: Option<f32>,
    /// Factors keyed by base tensor name.
    pub factors: HashMap<String, LoraFactors>,
}

/// Errors from loading / applying a LoRA adapter GGUF.
#[derive(Debug, thiserror::Error)]
pub enum LoraError {
    #[error("not a LoRA adapter GGUF: `adapter.type` is {0:?} (want \"lora\")")]
    NotLoraAdapter(Option<String>),
    #[error("adapter tensor `{0}` missing from GGUF")]
    MissingTensor(String),
    #[error("adapter factor `{name}` has unsupported dtype {dtype} (want F32/F16/BF16)")]
    UnsupportedFactorDtype { name: String, dtype: String },
    #[error("adapter pair for `{base}` has mismatched rank (a={rank_a}, b={rank_b})")]
    RankMismatch {
        base: String,
        rank_a: usize,
        rank_b: usize,
    },
    #[error("adapter tensor `{0}` has {1} dims, expected 2")]
    BadDims(String, usize),
    #[error("base model load failed: {0}")]
    BaseLoad(String),
    #[error("gguf open failed: {0}")]
    Gguf(String),
}

impl LoraAdapter {
    /// Parse a LoRA adapter from an already-open GGUF. `name` is the selection
    /// key this adapter is registered under.
    pub fn from_gguf(g: &Gguf, name: &str) -> Result<Self, LoraError> {
        let atype = g
            .metadata_get("adapter.type")
            .and_then(|v| v.as_string())
            .map(|s| s.to_string());
        if atype.as_deref() != Some("lora") {
            return Err(LoraError::NotLoraAdapter(atype));
        }
        let arch = g
            .metadata_get("general.architecture")
            .and_then(|v| v.as_string())
            .map(|s| s.to_string());
        let alpha = g
            .metadata_get("adapter.lora.alpha")
            .and_then(|v| v.as_f32());

        let mut factors = HashMap::new();
        for ti in g.tensors() {
            let Some(base) = ti.name.strip_suffix(".lora_a") else {
                continue;
            };
            let a_name = ti.name.clone();
            let b_name = format!("{base}.lora_b");
            let b_info = g
                .tensor(&b_name)
                .ok_or_else(|| LoraError::MissingTensor(b_name.clone()))?;
            if ti.dims.len() != 2 {
                return Err(LoraError::BadDims(a_name, ti.dims.len()));
            }
            if b_info.dims.len() != 2 {
                return Err(LoraError::BadDims(b_name, b_info.dims.len()));
            }
            // a ne = [in, rank]; b ne = [rank, out].
            let in_dim = ti.dims[0] as usize;
            let rank_a = ti.dims[1] as usize;
            let rank_b = b_info.dims[0] as usize;
            let out_dim = b_info.dims[1] as usize;
            if rank_a != rank_b {
                return Err(LoraError::RankMismatch {
                    base: base.to_string(),
                    rank_a,
                    rank_b,
                });
            }
            let rank = rank_a;
            let a = gguf_tensor_to_f32(g, &a_name)?;
            let b = gguf_tensor_to_f32(g, &b_name)?;
            // Missing alpha ⇒ alpha = rank (scale 1.0), the ggml default.
            let scale = alpha.unwrap_or(rank as f32) / rank.max(1) as f32;
            factors.insert(
                base.to_string(),
                LoraFactors {
                    rank,
                    in_dim,
                    out_dim,
                    scale,
                    a,
                    b,
                },
            );
        }
        Ok(Self {
            name: name.to_string(),
            arch,
            alpha,
            factors,
        })
    }

    /// Open a LoRA adapter GGUF from a path.
    pub fn from_path(path: &std::path::Path, name: &str) -> Result<Self, LoraError> {
        let g = Gguf::open(path).map_err(|e| LoraError::Gguf(e.to_string()))?;
        Self::from_gguf(&g, name)
    }

    /// Number of adapted base tensors.
    pub fn len(&self) -> usize {
        self.factors.len()
    }
    pub fn is_empty(&self) -> bool {
        self.factors.is_empty()
    }
    /// `true` iff this adapter adapts the base tensor `name`.
    pub fn adapts(&self, name: &str) -> bool {
        self.factors.contains_key(name)
    }
}

/// Read a GGUF tensor's bytes and widen to F32. Supports F32 / F16 / BF16
/// factor dtypes (what real adapters ship); anything else errors.
fn gguf_tensor_to_f32(g: &Gguf, name: &str) -> Result<Vec<f32>, LoraError> {
    let info = g
        .tensor(name)
        .ok_or_else(|| LoraError::MissingTensor(name.to_string()))?;
    let bytes = g
        .tensor_bytes(name)
        .ok_or_else(|| LoraError::MissingTensor(name.to_string()))?;
    let n = info.element_count() as usize;
    let mut out = vec![0.0f32; n];
    match info.dtype {
        GgmlType::F32 => {
            for (o, chunk) in out.iter_mut().zip(bytes.chunks_exact(4)) {
                *o = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
        }
        GgmlType::F16 => rustllama_gguf::dequant::dequant_f16(bytes, &mut out),
        GgmlType::Bf16 => rustllama_gguf::dequant::dequant_bf16(bytes, &mut out),
        other => {
            return Err(LoraError::UnsupportedFactorDtype {
                name: name.to_string(),
                dtype: format!("{other:?}"),
            })
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Phase 1 — merge-at-load
// ---------------------------------------------------------------------------

/// Outcome of attempting to merge one base tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeOutcome {
    /// `W += s·B·A` applied; the tensor is now an owned F32 weight.
    Merged,
    /// The adapter targets this tensor but the base is quantized — left
    /// packed for the Phase-2 delta path.
    SkippedQuant,
    /// The adapter does not target this tensor.
    NotTargeted,
    /// The adapter targets this tensor but its (out, in) disagree — skipped.
    ShapeMismatch,
}

/// Tally of a whole-model merge (see
/// [`crate::llama_arch::LlamaWeights::merge_lora`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MergeStats {
    pub merged: usize,
    pub skipped_quant: usize,
    pub shape_mismatch: usize,
}

/// `true` for base dtypes the Phase-1 merge dequantizes losslessly (F32 /
/// F16 / BF16). Quantized bases are intentionally excluded — merging them
/// would bloat memory + lose the quant; they use the Phase-2 delta path.
pub fn is_mergeable_dtype(d: Dtype) -> bool {
    matches!(d, Dtype::F32 | Dtype::F16 | Dtype::Bf16 | Dtype::Bf16Raw)
}

/// Dequantize an F32/F16/BF16 weight tensor to an owned F32 vector in
/// physical `[out, in]` row-major order (`v[r*in + c]`).
fn dequant_weight_to_f32(t: &Tensor) -> Vec<f32> {
    let n = t.element_count() as usize;
    match t.dtype {
        Dtype::F32 => rustllama_tensor::as_slice_f32(t).to_vec(),
        Dtype::F16 => rustllama_tensor::as_slice_f16(t)
            .iter()
            .map(|h| h.to_f32())
            .collect(),
        Dtype::Bf16 | Dtype::Bf16Raw => {
            let bytes = rustllama_tensor::as_bytes(t);
            let mut out = vec![0.0f32; n];
            rustllama_gguf::dequant::dequant_bf16(bytes, &mut out);
            out
        }
        other => panic!("dequant_weight_to_f32: unmergeable dtype {other:?}"),
    }
}

/// Phase 1: merge the adapter's delta for `t` in place when the adapter
/// targets it AND the base dtype is mergeable. Replaces `t` with an owned
/// F32 `[out, in]` tensor carrying the same name. Returns the outcome.
pub fn merge_tensor_in_place(adapter: &LoraAdapter, t: &mut Tensor) -> MergeOutcome {
    let Some(fac) = adapter.factors.get(&t.name) else {
        return MergeOutcome::NotTargeted;
    };
    if !is_mergeable_dtype(t.dtype) {
        return MergeOutcome::SkippedQuant;
    }
    if (t.out_dim(), t.in_dim()) != (fac.out_dim, fac.in_dim) {
        return MergeOutcome::ShapeMismatch;
    }
    let mut base = dequant_weight_to_f32(t);
    fac.add_delta_into(&mut base);
    *t = Tensor::from_vec_f32(
        t.name.clone(),
        vec![fac.out_dim as u64, fac.in_dim as u64],
        base,
    );
    MergeOutcome::Merged
}

// ---------------------------------------------------------------------------
// Phase 2 / 3 — delta-apply + per-request / per-slot selection
// ---------------------------------------------------------------------------

thread_local! {
    /// The thread's active adapter (`None` = base). Set per request / per slot
    /// immediately before the forward and restored after. Used by the
    /// single-active (contiguous) path: ONE adapter for the whole forward.
    static ACTIVE_LORA: RefCell<Option<Arc<LoraAdapter>>> = const { RefCell::new(None) };

    /// P11 Phase 3 — the thread's active PER-ROW adapter table (`None` = no
    /// per-row selection). Set by the engine around a batched / ragged forward
    /// where different rows (slots) carry different adapters; read by the
    /// batched matvec dispatch, which drives `CURRENT_ROW` per row so the delta
    /// for row `r` is `ROW_ADAPTERS[r]`. Mutually exclusive with `ACTIVE_LORA`
    /// in practice (the contiguous path sets the latter, the batched path the
    /// former); when both are set the row table takes precedence. `None` ⇒ the
    /// batched dispatch is byte-identical to the pre-LoRA path.
    static ROW_ADAPTERS: RefCell<Option<Arc<RowAdapters>>> = const { RefCell::new(None) };

    /// The concatenated-row index the batched dispatch is currently applying.
    /// Only read when `ROW_ADAPTERS` is `Some`; the batched per-row loop sets
    /// it immediately before each single-row `matvec_tensor_dispatch`.
    static CURRENT_ROW: Cell<usize> = const { Cell::new(0) };
}

/// Set the thread's active LoRA adapter, returning the previous value so the
/// caller can restore it. `None` selects the un-adapted base.
pub fn set_active_lora(a: Option<Arc<LoraAdapter>>) -> Option<Arc<LoraAdapter>> {
    ACTIVE_LORA.with(|c| c.replace(a))
}

/// The thread's active adapter, if any.
pub fn active_lora() -> Option<Arc<LoraAdapter>> {
    ACTIVE_LORA.with(|c| c.borrow().clone())
}

/// `true` iff the thread's active adapter adapts the base tensor `name`. The
/// matvec dispatch uses this to route an adapted tensor through the single-row
/// CPU path (so the delta is applied exactly once) while leaving non-adapted
/// tensors on the batched / fused fast paths.
#[inline]
pub fn lora_adapts(name: &str) -> bool {
    ACTIVE_LORA.with(|c| {
        c.borrow()
            .as_ref()
            .is_some_and(|a| a.factors.contains_key(name))
    })
}

/// Phase 2 / 3 delta-apply: add `s·B·(A·x)` into `out` for the active adapter's
/// factors on `name`. No-op (and does not touch `out`) when no adapter is active
/// or `name` isn't adapted — keeps the base dispatch byte-identical.
///
/// Selection precedence: when a PER-ROW adapter table is active (`ROW_ADAPTERS`,
/// the batched / ragged path), the delta comes from `ROW_ADAPTERS[CURRENT_ROW]`
/// — the batched dispatch sets `CURRENT_ROW` before each row's single-row
/// matvec. Otherwise the single thread-active `ACTIVE_LORA` adapter is used (the
/// contiguous per-request path). The two are mutually exclusive by construction;
/// the row table wins if both are somehow set.
#[inline]
pub fn apply_delta(name: &str, x: &[f32], out: &mut [f32]) {
    // Per-row table (batched / ragged forward) takes precedence.
    let had_row_table = ROW_ADAPTERS.with(|c| {
        let g = c.borrow();
        let Some(table) = g.as_ref() else {
            return false;
        };
        let row = CURRENT_ROW.with(|r| r.get());
        if let Some(a) = table.get(row) {
            if let Some(f) = a.factors.get(name) {
                f.apply_delta(x, out);
            }
        }
        true
    });
    if had_row_table {
        return;
    }
    ACTIVE_LORA.with(|c| {
        let g = c.borrow();
        let Some(a) = g.as_ref() else {
            return;
        };
        if let Some(f) = a.factors.get(name) {
            f.apply_delta(x, out);
        }
    });
}

/// RAII guard that restores the previously active adapter on drop (so a
/// scoped selection survives an early return / panic in the forward).
pub struct LoraScope(Option<Arc<LoraAdapter>>);

impl Drop for LoraScope {
    fn drop(&mut self) {
        set_active_lora(self.0.take());
    }
}

/// Make `a` the thread-active adapter for the lifetime of the returned guard.
#[must_use = "dropping the guard immediately restores the previous adapter"]
pub fn scope_active_lora(a: Option<Arc<LoraAdapter>>) -> LoraScope {
    LoraScope(set_active_lora(a))
}

// ---------------------------------------------------------------------------
// Phase 3 — per-row (per-slot) adapter selection in the batched / ragged forward
// ---------------------------------------------------------------------------

/// A per-row adapter selection for ONE batched / ragged forward: `per_row[r]`
/// is the adapter to apply at concatenated row `r` (`None` = base). The engine
/// builds one of these per tick (expanding each slot's adapter across the rows
/// it contributes) and scopes it via [`scope_row_adapters`]; the batched matvec
/// dispatch then applies `per_row[r]`'s delta at row `r`.
#[derive(Debug, Default, Clone)]
pub struct RowAdapters {
    per_row: Vec<Option<Arc<LoraAdapter>>>,
}

impl RowAdapters {
    pub fn new(per_row: Vec<Option<Arc<LoraAdapter>>>) -> Self {
        Self { per_row }
    }
    /// The adapter for concatenated row `row`, if any (out-of-range ⇒ `None`).
    #[inline]
    pub fn get(&self, row: usize) -> Option<Arc<LoraAdapter>> {
        self.per_row.get(row).and_then(|a| a.clone())
    }
    /// `true` iff ANY row's adapter adapts the base tensor `name`. The batched
    /// dispatch uses this to decide whether to force the per-row single-row path
    /// (so the delta is applied row-by-row) for this tensor.
    #[inline]
    pub fn adapts_any(&self, name: &str) -> bool {
        self.per_row
            .iter()
            .any(|a| a.as_ref().is_some_and(|ad| ad.factors.contains_key(name)))
    }
    /// `true` iff no row carries an adapter (every entry `None`). Lets a caller
    /// skip scoping entirely when the batch has no LoRA (byte-identical path).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.per_row.iter().all(|a| a.is_none())
    }
    pub fn len(&self) -> usize {
        self.per_row.len()
    }
}

/// Set the thread's active per-row adapter table, returning the previous value.
pub fn set_row_adapters(t: Option<Arc<RowAdapters>>) -> Option<Arc<RowAdapters>> {
    ROW_ADAPTERS.with(|c| c.replace(t))
}

/// The thread's active per-row adapter table, if any.
pub fn active_row_adapters() -> Option<Arc<RowAdapters>> {
    ROW_ADAPTERS.with(|c| c.borrow().clone())
}

/// `true` iff a per-row table is active AND some row's adapter adapts `name`.
/// The batched matvec dispatch ORs this with [`lora_adapts`] to decide whether
/// to route `name` through the per-row single-row path.
#[inline]
pub fn row_lora_adapts(name: &str) -> bool {
    ROW_ADAPTERS.with(|c| c.borrow().as_ref().is_some_and(|t| t.adapts_any(name)))
}

/// Set the concatenated-row index the batched dispatch is about to apply. Read
/// by [`apply_delta`] only when a per-row table is active; a no-op cost
/// otherwise (one `Cell` store).
#[inline]
pub fn set_current_lora_row(row: usize) {
    CURRENT_ROW.with(|c| c.set(row));
}

/// RAII guard restoring the previously active per-row table on drop.
pub struct RowLoraScope(Option<Arc<RowAdapters>>);

impl Drop for RowLoraScope {
    fn drop(&mut self) {
        set_row_adapters(self.0.take());
    }
}

/// Make `t` the thread-active per-row adapter table for the lifetime of the
/// returned guard. `None` (or an all-`None` table) selects the un-adapted base
/// for every row — byte-identical to the pre-LoRA batched path.
#[must_use = "dropping the guard immediately restores the previous per-row table"]
pub fn scope_row_adapters(t: Option<Arc<RowAdapters>>) -> RowLoraScope {
    RowLoraScope(set_row_adapters(t))
}

/// Phase 3: a set of named adapters held resident for per-request selection.
/// Cheap to clone (each adapter is `Arc`-shared); share as `Arc<LoraRegistry>`
/// and select per request/slot via [`LoraRegistry::get`] + [`scope_active_lora`].
#[derive(Debug, Default, Clone)]
pub struct LoraRegistry {
    adapters: HashMap<String, Arc<LoraAdapter>>,
}

impl LoraRegistry {
    pub fn new() -> Self {
        Self::default()
    }
    /// Register `adapter` under its own `name`, returning the shared handle.
    pub fn insert(&mut self, adapter: LoraAdapter) -> Arc<LoraAdapter> {
        let handle = Arc::new(adapter);
        self.adapters.insert(handle.name.clone(), handle.clone());
        handle
    }
    /// Look up a resident adapter by name.
    pub fn get(&self, name: &str) -> Option<Arc<LoraAdapter>> {
        self.adapters.get(name).cloned()
    }
    pub fn contains(&self, name: &str) -> bool {
        self.adapters.contains_key(name)
    }
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.adapters.keys().map(|s| s.as_str())
    }
    pub fn is_empty(&self) -> bool {
        self.adapters.is_empty()
    }
    pub fn len(&self) -> usize {
        self.adapters.len()
    }
}
