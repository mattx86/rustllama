//! Assemble a `LlamaModel` from the A-2b
//! [`ConvertedTensor`](crate::ConvertedTensor) list + the A-2c-1
//! [`LlamaConfig`](rustllama_models::llama_config::LlamaConfig).
//!
//! This is A-2c-2 — the final piece of the safetensors → rustllama
//! conversion arc. After this lands, a caller can do:
//!
//! ```ignore
//! let cfg     = parse_hf_config(&fs::read_to_string("config.json")?)?;
//! let blob    = fs::read("model.safetensors")?;
//! let tensors = convert_safetensors_to_gguf_tensors(&blob)?;
//! let model   = build_llama_model_from_safetensors(&cfg, tensors)?;
//! ```
//!
//! Output is identical in shape, dtype, and layout to what
//! `LlamaWeights::from_gguf` produces from an equivalent GGUF, so the
//! existing `LlamaModel` forward path consumes it unchanged.

use std::collections::HashMap;
use std::sync::Arc;

use bytemuck::cast_slice;
use half::f16;
use rustllama_models::llama_arch::{
    LlamaBlockWeights, LlamaModel, LlamaMoeBlockWeights, LlamaWeights,
};
use rustllama_models::llama_config::{LlamaConfig, MoeConfig};
use rustllama_tensor::{contiguous_strides, Device, Dtype, Storage, Tensor};

use crate::convert::{ConvertedDtype, ConvertedTensor};

#[derive(Debug, thiserror::Error)]
pub enum BuildError {
    #[error("required tensor `{0}` not present in safetensors")]
    MissingTensor(String),
    #[error(
        "tensor `{name}` has shape {got:?} but loader expected \
         {expected:?} for {role} (config: n_layers={n_layers}, \
         d_model={d_model}, d_ff={d_ff}, n_heads={n_heads}, \
         n_kv_heads={n_kv_heads}, head_dim={head_dim})"
    )]
    ShapeMismatch {
        name: String,
        role: &'static str,
        got: Vec<u64>,
        expected: Vec<u64>,
        n_layers: usize,
        d_model: usize,
        d_ff: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    },
    #[error(
        "MoE safetensors load is not supported in v1 (config declared \
         `expert_count > 0`); convert via GGUF or wait for the MoE
         conversion path"
    )]
    MoeUnsupported,
    #[error(
        "MoE build: stacked expert tensor `{name}` has {got} bytes, not a \
         whole multiple of {n_experts} experts ({per_expert} B each)"
    )]
    MoeExpertStackMismatch {
        name: String,
        got: usize,
        n_experts: usize,
        per_expert: usize,
    },
}

/// Assemble a [`LlamaModel`] from the A-2b converted-tensor list
/// against the architecture spec from A-2c-1.
///
/// Walks the named tensors in the standard Llama-family layout:
/// `token_embd.weight`, `output_norm.weight`, optional
/// `output.weight` (absent → tied embeddings), and for each layer
/// `blk.<i>.{attn_norm, attn_q, attn_k, attn_v, attn_output,
/// ffn_norm, ffn_gate, ffn_up, ffn_down}.weight`, plus the
/// optional `attn_{q,k,v}.bias` (Qwen2 family). Every tensor is
/// validated against the shape `cfg` predicts; a mismatch aborts
/// with a clear error rather than letting the engine load with
/// silently misshaped weights.
///
/// v1 is dense-only (`cfg.moe.is_none()`); MoE safetensors loads
/// surface [`BuildError::MoeUnsupported`].
pub fn build_llama_model_from_safetensors(
    cfg: &LlamaConfig,
    tensors: Vec<ConvertedTensor>,
) -> Result<LlamaModel, BuildError> {
    // Build a name → ConvertedTensor map so we can extract each
    // tensor by name and surface a clear MissingTensor error when
    // a required slot isn't present.
    let mut by_name: HashMap<String, ConvertedTensor> =
        tensors.into_iter().map(|t| (t.gguf_name.clone(), t)).collect();

    // MoE checkpoints (Qwen2-MoE / Qwen3-MoE / OLMoE / Mixtral) take a
    // separate assembly path: the engine's MLX transcoder stacks each
    // layer's experts into `blk.N.ffn_{gate,up,down}_exps.weight` + a
    // `ffn_gate_inp` router (+ optional shared expert), exactly the
    // GGUF MoE tensor set — this builds `LlamaMoeBlockWeights` from it.
    if let Some(moe) = cfg.moe.as_ref() {
        return build_moe_model_from_safetensors(cfg, moe, &mut by_name);
    }

    let d_model = cfg.d_model;
    let d_ff = cfg.d_ff;
    let n_heads = cfg.n_heads;
    let n_kv_heads = cfg.n_kv_heads;
    let head_dim = cfg.head_dim;
    let d_q = n_heads * head_dim;
    let d_kv = n_kv_heads * head_dim;

    // ---- Top-level tensors ------------------------------------
    let token_embd = take_tensor(
        &mut by_name,
        cfg,
        "token_embd.weight",
        "token_embd",
        &[cfg.vocab_size as u64, d_model as u64],
    )?;
    let output_norm = take_f32_vec(
        &mut by_name,
        cfg,
        "output_norm.weight",
        "output_norm",
        &[d_model as u64],
    )?;
    // Tied embeddings: `output.weight` is absent in the file; the
    // engine reuses `token_embd` for the LM head. Honor `cfg
    // .tie_word_embeddings` rather than just probing presence so a
    // file that ships an unused `output` tensor doesn't surprise us.
    let output = if cfg.tie_word_embeddings {
        None
    } else {
        Some(take_tensor(
            &mut by_name,
            cfg,
            "output.weight",
            "output",
            &[cfg.vocab_size as u64, d_model as u64],
        )?)
    };

    // ---- Per-layer dense blocks ------------------------------
    let mut blocks = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        let prefix = format!("blk.{i}");
        let attn_norm = take_f32_vec(
            &mut by_name,
            cfg,
            &format!("{prefix}.attn_norm.weight"),
            "attn_norm",
            &[d_model as u64],
        )?;
        let w_q = take_tensor(
            &mut by_name,
            cfg,
            &format!("{prefix}.attn_q.weight"),
            "attn_q",
            &[d_q as u64, d_model as u64],
        )?;
        let w_k = take_tensor(
            &mut by_name,
            cfg,
            &format!("{prefix}.attn_k.weight"),
            "attn_k",
            &[d_kv as u64, d_model as u64],
        )?;
        let w_v = take_tensor(
            &mut by_name,
            cfg,
            &format!("{prefix}.attn_v.weight"),
            "attn_v",
            &[d_kv as u64, d_model as u64],
        )?;
        let w_o = take_tensor(
            &mut by_name,
            cfg,
            &format!("{prefix}.attn_output.weight"),
            "attn_output",
            &[d_model as u64, d_q as u64],
        )?;
        let b_q = take_optional_f32_vec(
            &mut by_name,
            cfg,
            &format!("{prefix}.attn_q.bias"),
            "attn_q.bias",
            &[d_q as u64],
        )?;
        let b_k = take_optional_f32_vec(
            &mut by_name,
            cfg,
            &format!("{prefix}.attn_k.bias"),
            "attn_k.bias",
            &[d_kv as u64],
        )?;
        let b_v = take_optional_f32_vec(
            &mut by_name,
            cfg,
            &format!("{prefix}.attn_v.bias"),
            "attn_v.bias",
            &[d_kv as u64],
        )?;
        let ffn_norm = take_f32_vec(
            &mut by_name,
            cfg,
            &format!("{prefix}.ffn_norm.weight"),
            "ffn_norm",
            &[d_model as u64],
        )?;
        let w_gate = take_tensor(
            &mut by_name,
            cfg,
            &format!("{prefix}.ffn_gate.weight"),
            "ffn_gate",
            &[d_ff as u64, d_model as u64],
        )?;
        let w_up = take_tensor(
            &mut by_name,
            cfg,
            &format!("{prefix}.ffn_up.weight"),
            "ffn_up",
            &[d_ff as u64, d_model as u64],
        )?;
        let w_down = take_tensor(
            &mut by_name,
            cfg,
            &format!("{prefix}.ffn_down.weight"),
            "ffn_down",
            &[d_model as u64, d_ff as u64],
        )?;
        blocks.push(LlamaBlockWeights {
            attn_norm,
            w_q,
            w_k,
            w_v,
            // H3: safetensors path doesn't pre-concat QKV; the user
            // can opt in via env var on the GGUF path. Mechanical
            // mirror for safetensors is a follow-up.
            w_qkv_fused: None,
            w_o,
            b_q,
            b_k,
            b_v,
            ffn_norm,
            w_gate,
            w_up,
            w_down,
        });
    }

    Ok(LlamaModel {
        cfg: cfg.clone(),
        weights: LlamaWeights {
            token_embd,
            blocks,
            moe_blocks: None,
            hybrid_layers: None,
            output_norm,
            output,
            mtp_heads: None,
            nextn_head: None,
            hadamard: None,
        },
    })
}

/// Assemble a MoE [`LlamaModel`] from the converted-tensor list.
///
/// The engine's MLX transcoder (`cpu.rs::mlx_model_to_converted`)
/// stacks each layer's per-expert MLX linears into the GGUF MoE tensor
/// set — `blk.N.ffn_{gate,up,down}_exps.weight` (one tensor holding all
/// experts' bytes back-to-back, the exact byte layout
/// [`rustllama_models::moe::expert_view`] slices), a `blk.N.ffn_gate_inp`
/// router, and (Qwen2-MoE) a `blk.N.ffn_{gate,up,down}_shexp` shared
/// expert plus a `blk.N.ffn_gate_inp_shexp` sigmoid gate. This mirrors
/// the GGUF MoE loader (`LlamaWeights::from_gguf`): same tensor names,
/// same per-expert views, so the engine's existing MoE forward runs it
/// unchanged. Attention / norm / embedding slots reuse the dense path's
/// `take_*` helpers (identical to a dense Llama block).
fn build_moe_model_from_safetensors(
    cfg: &LlamaConfig,
    moe: &MoeConfig,
    by_name: &mut HashMap<String, ConvertedTensor>,
) -> Result<LlamaModel, BuildError> {
    let d_model = cfg.d_model;
    let d_ff = cfg.d_ff; // per-expert FFN width (moe_intermediate_size)
    let n_heads = cfg.n_heads;
    let n_kv_heads = cfg.n_kv_heads;
    let head_dim = cfg.head_dim;
    let d_q = n_heads * head_dim;
    let d_kv = n_kv_heads * head_dim;
    let n_experts = moe.n_experts as usize;

    // ---- Top-level tensors (identical to the dense path) ----------
    let token_embd = take_tensor(
        by_name,
        cfg,
        "token_embd.weight",
        "token_embd",
        &[cfg.vocab_size as u64, d_model as u64],
    )?;
    let output_norm = take_f32_vec(
        by_name,
        cfg,
        "output_norm.weight",
        "output_norm",
        &[d_model as u64],
    )?;
    let output = if cfg.tie_word_embeddings {
        None
    } else {
        Some(take_tensor(
            by_name,
            cfg,
            "output.weight",
            "output",
            &[cfg.vocab_size as u64, d_model as u64],
        )?)
    };

    // ---- Per-layer MoE blocks -------------------------------------
    let mut mbs = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        let prefix = format!("blk.{i}");
        // Attention + norms: same shapes/roles as a dense block.
        let attn_norm = take_f32_vec(
            by_name,
            cfg,
            &format!("{prefix}.attn_norm.weight"),
            "attn_norm",
            &[d_model as u64],
        )?;
        let w_q = take_tensor(
            by_name,
            cfg,
            &format!("{prefix}.attn_q.weight"),
            "attn_q",
            &[d_q as u64, d_model as u64],
        )?;
        let w_k = take_tensor(
            by_name,
            cfg,
            &format!("{prefix}.attn_k.weight"),
            "attn_k",
            &[d_kv as u64, d_model as u64],
        )?;
        let w_v = take_tensor(
            by_name,
            cfg,
            &format!("{prefix}.attn_v.weight"),
            "attn_v",
            &[d_kv as u64, d_model as u64],
        )?;
        let w_o = take_tensor(
            by_name,
            cfg,
            &format!("{prefix}.attn_output.weight"),
            "attn_output",
            &[d_model as u64, d_q as u64],
        )?;
        let b_q = take_optional_f32_vec(
            by_name,
            cfg,
            &format!("{prefix}.attn_q.bias"),
            "attn_q.bias",
            &[d_q as u64],
        )?;
        let b_k = take_optional_f32_vec(
            by_name,
            cfg,
            &format!("{prefix}.attn_k.bias"),
            "attn_k.bias",
            &[d_kv as u64],
        )?;
        let b_v = take_optional_f32_vec(
            by_name,
            cfg,
            &format!("{prefix}.attn_v.bias"),
            "attn_v.bias",
            &[d_kv as u64],
        )?;
        let ffn_norm = take_f32_vec(
            by_name,
            cfg,
            &format!("{prefix}.ffn_norm.weight"),
            "ffn_norm",
            &[d_model as u64],
        )?;

        // Router `[n_experts, d_model]`. Block-quant shape is logical,
        // so no strict pre-check — route_topk's matvec uses explicit
        // m/k. Required (every MoE layer has a router).
        let router = take_moe_tensor(by_name, &format!("{prefix}.ffn_gate_inp.weight"))?;

        // Stacked experts. Each is one tensor of n_experts back-to-back
        // `[d_out, d_in]` matrices; `expert_view` slices by byte offset.
        let w_gate_exps = take_moe_stacked(
            by_name,
            &format!("{prefix}.ffn_gate_exps.weight"),
            n_experts,
            d_ff,
            d_model,
        )?;
        let w_up_exps = take_moe_stacked(
            by_name,
            &format!("{prefix}.ffn_up_exps.weight"),
            n_experts,
            d_ff,
            d_model,
        )?;
        let w_down_exps = take_moe_stacked(
            by_name,
            &format!("{prefix}.ffn_down_exps.weight"),
            n_experts,
            d_model,
            d_ff,
        )?;

        // Pre-compute per-expert views (share storage via Arc slice —
        // no byte copy). Same call shape as the GGUF MoE loader.
        let gate_per_expert: Vec<Tensor> = (0..n_experts)
            .map(|e| rustllama_models::moe::expert_view(&w_gate_exps, e, d_ff, d_model))
            .collect();
        let up_per_expert: Vec<Tensor> = (0..n_experts)
            .map(|e| rustllama_models::moe::expert_view(&w_up_exps, e, d_ff, d_model))
            .collect();
        let down_per_expert: Vec<Tensor> = (0..n_experts)
            .map(|e| rustllama_models::moe::expert_view(&w_down_exps, e, d_model, d_ff))
            .collect();

        // Shared expert (Qwen2-MoE) — optional. Its width differs from
        // the routed experts; the forward reads it from the tensor. The
        // sigmoid gate (`ffn_gate_inp_shexp`) is also optional.
        let w_gate_shared =
            take_optional_moe_tensor(by_name, &format!("{prefix}.ffn_gate_shexp.weight"));
        let w_up_shared =
            take_optional_moe_tensor(by_name, &format!("{prefix}.ffn_up_shexp.weight"));
        let w_down_shared =
            take_optional_moe_tensor(by_name, &format!("{prefix}.ffn_down_shexp.weight"));
        let shared_router =
            take_optional_moe_tensor(by_name, &format!("{prefix}.ffn_gate_inp_shexp.weight"));

        mbs.push(LlamaMoeBlockWeights {
            attn_norm,
            w_q,
            w_k,
            w_v,
            w_o,
            b_q,
            b_k,
            b_v,
            ffn_norm,
            router,
            w_gate_exps,
            w_up_exps,
            w_down_exps,
            w_gate_shared,
            w_up_shared,
            w_down_shared,
            shared_router,
            gate_per_expert,
            up_per_expert,
            down_per_expert,
        });
    }

    Ok(LlamaModel {
        cfg: cfg.clone(),
        weights: LlamaWeights {
            token_embd,
            blocks: Vec::new(),
            moe_blocks: Some(mbs),
            hybrid_layers: None,
            output_norm,
            output,
            mtp_heads: None,
            nextn_head: None,
            hadamard: None,
        },
    })
}

/// Take a required MoE tensor by GGUF name with no shape check — used
/// for the router and stacked-expert tensors whose logical shape isn't
/// a plain `[out, in]` linear (the matvec/`expert_view` consumers pass
/// explicit dims).
fn take_moe_tensor(
    by_name: &mut HashMap<String, ConvertedTensor>,
    name: &str,
) -> Result<Tensor, BuildError> {
    let ct = by_name
        .remove(name)
        .ok_or_else(|| BuildError::MissingTensor(name.into()))?;
    Ok(converted_to_tensor(ct))
}

/// Take an optional MoE tensor (shared-expert / shared-gate slots that
/// only Qwen2-MoE carries). Absent → `None`.
fn take_optional_moe_tensor(
    by_name: &mut HashMap<String, ConvertedTensor>,
    name: &str,
) -> Option<Tensor> {
    by_name.remove(name).map(converted_to_tensor)
}

/// Take a stacked per-expert tensor and verify its byte length is
/// exactly `n_experts × per-expert-byte-size` so every
/// [`rustllama_models::moe::expert_view`] offset lands in bounds.
fn take_moe_stacked(
    by_name: &mut HashMap<String, ConvertedTensor>,
    name: &str,
    n_experts: usize,
    d_out: usize,
    d_in: usize,
) -> Result<Tensor, BuildError> {
    let ct = by_name
        .remove(name)
        .ok_or_else(|| BuildError::MissingTensor(name.into()))?;
    let t = converted_to_tensor(ct);
    let per_expert = t.dtype.byte_size((d_out * d_in) as u64) as usize;
    let got = t.storage.len_bytes();
    if per_expert == 0 || got != per_expert * n_experts {
        return Err(BuildError::MoeExpertStackMismatch {
            name: name.into(),
            got,
            n_experts,
            per_expert,
        });
    }
    Ok(t)
}

fn take_tensor(
    by_name: &mut HashMap<String, ConvertedTensor>,
    cfg: &LlamaConfig,
    name: &str,
    role: &'static str,
    expected_shape: &[u64],
) -> Result<Tensor, BuildError> {
    let ct = by_name
        .remove(name)
        .ok_or_else(|| BuildError::MissingTensor(name.into()))?;
    check_shape(cfg, name, role, &ct.shape, expected_shape)?;
    Ok(converted_to_tensor(ct))
}

fn take_f32_vec(
    by_name: &mut HashMap<String, ConvertedTensor>,
    cfg: &LlamaConfig,
    name: &str,
    role: &'static str,
    expected_shape: &[u64],
) -> Result<Vec<f32>, BuildError> {
    let ct = by_name
        .remove(name)
        .ok_or_else(|| BuildError::MissingTensor(name.into()))?;
    check_shape(cfg, name, role, &ct.shape, expected_shape)?;
    Ok(converted_to_f32_vec(&ct))
}

fn take_optional_f32_vec(
    by_name: &mut HashMap<String, ConvertedTensor>,
    cfg: &LlamaConfig,
    name: &str,
    role: &'static str,
    expected_shape: &[u64],
) -> Result<Option<Vec<f32>>, BuildError> {
    match by_name.remove(name) {
        Some(ct) => {
            check_shape(cfg, name, role, &ct.shape, expected_shape)?;
            Ok(Some(converted_to_f32_vec(&ct)))
        }
        None => Ok(None),
    }
}

fn check_shape(
    cfg: &LlamaConfig,
    name: &str,
    role: &'static str,
    got: &[u64],
    expected: &[u64],
) -> Result<(), BuildError> {
    if got == expected {
        return Ok(());
    }
    Err(BuildError::ShapeMismatch {
        name: name.into(),
        role,
        got: got.to_vec(),
        expected: expected.to_vec(),
        n_layers: cfg.n_layers,
        d_model: cfg.d_model,
        d_ff: cfg.d_ff,
        n_heads: cfg.n_heads,
        n_kv_heads: cfg.n_kv_heads,
        head_dim: cfg.head_dim,
    })
}

fn converted_to_tensor(ct: ConvertedTensor) -> Tensor {
    let dtype = match ct.dtype {
        ConvertedDtype::F32 => Dtype::F32,
        ConvertedDtype::F16 => Dtype::F16,
        // MLX affine: the bytes are already the packed self-describing
        // blob; wrap them verbatim (the logical `[out, in]` shape is
        // carried in `ct.shape`, so `check_shape` and the matvec `m`/`k`
        // stay correct). Matvec / embedding dispatch on this dtype route
        // to the packed MLX kernels — no dequant here.
        ConvertedDtype::MlxAffineRaw => Dtype::MlxAffineRaw,
        // MLX load-time transcoder output: the bytes are already standard
        // GGUF Q4_1 / Q8_0 blocks — byte-identical to what the GGUF loader
        // produces for these dtypes (see `bert_arch::tensor_from_info`), so
        // they wrap verbatim and the SHARED packed-quant matvec / embedding
        // kernels (CPU + SYCL + CUDA + MLX-Metal) pick them up with no new
        // dispatch. The block layout runs along the innermost (`in`) dim,
        // which is a multiple of 32, so blocks never cross a row boundary
        // and the logical `[out, in]` shape in `ct.shape` keeps `check_shape`
        // and the matvec `m`/`k` correct.
        ConvertedDtype::Q4_1Raw => Dtype::Q4_1Raw,
        // Q4_K super-blocks (256 weights) wrap identically to Q4_1: the bytes
        // are byte-identical to the GGUF loader's Q4_K output, and the
        // transcoder only picks Q4_K when `in_features % 256 == 0`, so a
        // super-block never crosses the innermost (`in`) row boundary — the
        // same no-cross invariant that keeps `[out, in]` `check_shape` + the
        // matvec `m`/`k` correct.
        ConvertedDtype::Q4_KRaw => Dtype::Q4_KRaw,
        ConvertedDtype::Q8_0Raw => Dtype::Q8_0Raw,
    };
    let strides = contiguous_strides(&ct.shape);
    Tensor {
        device: Device::Cpu,
        dtype,
        shape: ct.shape,
        strides,
        storage: Storage::CpuOwned(Arc::<[u8]>::from(ct.bytes)),
        name: ct.gguf_name,
    }
}

/// Convert a Plain f16/f32 `ConvertedTensor` into a flat `Vec<f32>`
/// for norm / bias slots. The engine stores these as `Vec<f32>` to
/// keep the per-token RMSNorm + bias-add hot loops in native f32.
fn converted_to_f32_vec(ct: &ConvertedTensor) -> Vec<f32> {
    match ct.dtype {
        ConvertedDtype::F32 => cast_slice::<u8, f32>(&ct.bytes).to_vec(),
        ConvertedDtype::F16 => {
            let f16s: &[f16] = cast_slice(&ct.bytes);
            f16s.iter().map(|h| h.to_f32()).collect()
        }
        // Only norm / bias slots (always full-precision in an MLX
        // checkpoint — RMSNorm weights, linear biases) reach this f32-vec
        // path; quantized linears route through `converted_to_tensor`.
        // An MLX-affine blob here would mean a norm got mis-classified as
        // quantized, so fail loudly rather than feed a packed blob to a
        // flat-f32 consumer.
        ConvertedDtype::MlxAffineRaw => unreachable!(
            "converted_to_f32_vec: MLX affine blob `{}` routed to the \
             norm/bias f32-vec path — quantized weights must go through \
             converted_to_tensor",
            ct.gguf_name
        ),
        // Transcoded GGUF-quant linears never reach the norm/bias f32-vec
        // path (same invariant as MlxAffineRaw above — only full-precision
        // norms/biases do). Fail loudly if one ever does.
        ConvertedDtype::Q4_1Raw | ConvertedDtype::Q4_KRaw | ConvertedDtype::Q8_0Raw => {
            unreachable!(
                "converted_to_f32_vec: GGUF-quant blob `{}` routed to the \
                 norm/bias f32-vec path — quantized weights must go through \
                 converted_to_tensor",
                ct.gguf_name
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use safetensors::tensor::TensorView;
    use safetensors::Dtype as StDtype;

    use crate::convert::convert_safetensors_to_gguf_tensors;
    use crate::hf_config::parse_hf_config;

    fn pack8(lanes: [u8; 8]) -> i32 {
        let mut acc: u32 = 0;
        for (k, v) in lanes.iter().enumerate() {
            acc |= ((*v as u32) & 0xF) << (k * 4);
        }
        acc as i32
    }
    fn i32_bytes(v: &[i32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }
    fn f16_bytes(v: &[f16]) -> Vec<u8> {
        v.iter().flat_map(|h| h.to_le_bytes()).collect()
    }
    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    /// Insert qweight+scales+qzeros for one AWQ linear in `hf_name` shape.
    /// Returns the byte buffers as a vec; the safetensors `serialize`
    /// API holds borrows back into these buffers, so the caller keeps
    /// them live for the lifetime of the serialize call.
    fn awq_linear_bytes(
        in_features: usize,
        out_features: usize,
        group_size: usize,
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let out_packs = out_features / 8;
        let n_groups = in_features / group_size;
        // Constant weight pattern: int4 lane 1 → after (1 - 0) * 0.5 = 0.5
        let qw: Vec<i32> =
            (0..in_features * out_packs).map(|_| pack8([1; 8])).collect();
        let sc: Vec<f16> = vec![f16::from_f32(0.5); n_groups * out_features];
        let qz: Vec<i32> =
            (0..n_groups * out_packs).map(|_| pack8([0; 8])).collect();
        (i32_bytes(&qw), f16_bytes(&sc), i32_bytes(&qz))
    }

    /// Build a minimal AWQ-shape safetensors blob for one fully-shaped
    /// 1-layer model. Architecture: hidden_size=16, intermediate=32,
    /// n_heads=2, head_dim=8 → d_q = d_kv = 16. Vocab=4.
    fn build_minimal_awq_safetensors() -> (String, Vec<u8>) {
        let vocab = 4usize;
        let d_model = 16usize;
        let d_ff = 32usize;
        let n_heads = 2usize;
        let head_dim = 8usize;
        let group_size = 16; // 1 group
        // Linear shapes the loader expects (out, in):
        //   q/k/v/o: 16x16  gate/up: 32x16  down: 16x32
        // AWQ stores them transposed (in, out/8) on disk; the converter
        // transposes back to (out, in) which is what we just stated.
        let (qw_qproj, sc_qproj, qz_qproj) =
            awq_linear_bytes(d_model, d_model, group_size);
        let (qw_kproj, sc_kproj, qz_kproj) =
            awq_linear_bytes(d_model, d_model, group_size);
        let (qw_vproj, sc_vproj, qz_vproj) =
            awq_linear_bytes(d_model, d_model, group_size);
        let (qw_oproj, sc_oproj, qz_oproj) =
            awq_linear_bytes(d_model, d_model, group_size);
        let (qw_gate, sc_gate, qz_gate) =
            awq_linear_bytes(d_model, d_ff, group_size);
        let (qw_up, sc_up, qz_up) = awq_linear_bytes(d_model, d_ff, group_size);
        let (qw_down, sc_down, qz_down) =
            awq_linear_bytes(d_ff, d_model, group_size);
        let token_embd_data: Vec<f16> = vec![f16::from_f32(0.1); vocab * d_model];
        let token_embd_bytes = f16_bytes(&token_embd_data);
        let attn_norm: Vec<f16> = vec![f16::from_f32(1.0); d_model];
        let attn_norm_bytes = f16_bytes(&attn_norm);
        let ffn_norm: Vec<f16> = vec![f16::from_f32(1.0); d_model];
        let ffn_norm_bytes = f16_bytes(&ffn_norm);
        let out_norm: Vec<f16> = vec![f16::from_f32(1.0); d_model];
        let out_norm_bytes = f16_bytes(&out_norm);
        let lm_head_data: Vec<f16> = vec![f16::from_f32(0.1); vocab * d_model];
        let lm_head_bytes = f16_bytes(&lm_head_data);

        let mut map: BTreeMap<String, TensorView<'_>> = BTreeMap::new();
        let out_packs_model = d_model / 8;
        let out_packs_ff = d_ff / 8;
        // q/k/v/o projections
        for (proj, qw, sc, qz) in [
            ("q_proj", &qw_qproj, &sc_qproj, &qz_qproj),
            ("k_proj", &qw_kproj, &sc_kproj, &qz_kproj),
            ("v_proj", &qw_vproj, &sc_vproj, &qz_vproj),
            ("o_proj", &qw_oproj, &sc_oproj, &qz_oproj),
        ] {
            map.insert(
                format!("model.layers.0.self_attn.{proj}.qweight"),
                TensorView::new(StDtype::I32, vec![d_model, out_packs_model], qw)
                    .unwrap(),
            );
            map.insert(
                format!("model.layers.0.self_attn.{proj}.scales"),
                TensorView::new(StDtype::F16, vec![1, d_model], sc).unwrap(),
            );
            map.insert(
                format!("model.layers.0.self_attn.{proj}.qzeros"),
                TensorView::new(StDtype::I32, vec![1, out_packs_model], qz).unwrap(),
            );
        }
        // gate / up
        for (proj, qw, sc, qz) in [
            ("gate_proj", &qw_gate, &sc_gate, &qz_gate),
            ("up_proj", &qw_up, &sc_up, &qz_up),
        ] {
            map.insert(
                format!("model.layers.0.mlp.{proj}.qweight"),
                TensorView::new(StDtype::I32, vec![d_model, out_packs_ff], qw)
                    .unwrap(),
            );
            map.insert(
                format!("model.layers.0.mlp.{proj}.scales"),
                TensorView::new(StDtype::F16, vec![1, d_ff], sc).unwrap(),
            );
            map.insert(
                format!("model.layers.0.mlp.{proj}.qzeros"),
                TensorView::new(StDtype::I32, vec![1, out_packs_ff], qz).unwrap(),
            );
        }
        // down (in=d_ff, out=d_model)
        map.insert(
            "model.layers.0.mlp.down_proj.qweight".into(),
            TensorView::new(StDtype::I32, vec![d_ff, out_packs_model], &qw_down)
                .unwrap(),
        );
        map.insert(
            "model.layers.0.mlp.down_proj.scales".into(),
            TensorView::new(StDtype::F16, vec![d_ff / group_size, d_model], &sc_down)
                .unwrap(),
        );
        map.insert(
            "model.layers.0.mlp.down_proj.qzeros".into(),
            TensorView::new(
                StDtype::I32,
                vec![d_ff / group_size, out_packs_model],
                &qz_down,
            )
            .unwrap(),
        );
        // Norms + embeddings.
        map.insert(
            "model.embed_tokens.weight".into(),
            TensorView::new(StDtype::F16, vec![vocab, d_model], &token_embd_bytes)
                .unwrap(),
        );
        map.insert(
            "model.layers.0.input_layernorm.weight".into(),
            TensorView::new(StDtype::F16, vec![d_model], &attn_norm_bytes)
                .unwrap(),
        );
        map.insert(
            "model.layers.0.post_attention_layernorm.weight".into(),
            TensorView::new(StDtype::F16, vec![d_model], &ffn_norm_bytes).unwrap(),
        );
        map.insert(
            "model.norm.weight".into(),
            TensorView::new(StDtype::F16, vec![d_model], &out_norm_bytes).unwrap(),
        );
        map.insert(
            "lm_head.weight".into(),
            TensorView::new(StDtype::F16, vec![vocab, d_model], &lm_head_bytes)
                .unwrap(),
        );
        let blob = safetensors::serialize(&map, &None).expect("serialize");

        let cfg_json = format!(
            r#"{{
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": {d_model},
                "intermediate_size": {d_ff},
                "num_hidden_layers": 1,
                "num_attention_heads": {n_heads},
                "num_key_value_heads": {n_heads},
                "head_dim": {head_dim},
                "vocab_size": {vocab},
                "max_position_embeddings": 64,
                "tie_word_embeddings": false,
                "bos_token_id": 0,
                "eos_token_id": 0
            }}"#
        );
        (cfg_json, blob)
    }

    #[test]
    fn build_minimal_awq_model_succeeds_and_has_correct_shapes() {
        let (cfg_json, blob) = build_minimal_awq_safetensors();
        let cfg = parse_hf_config(&cfg_json).unwrap();
        let tensors = convert_safetensors_to_gguf_tensors(&blob).unwrap();
        let model =
            build_llama_model_from_safetensors(&cfg, tensors).expect("build");

        // Architecture sanity checks.
        assert_eq!(model.cfg.n_layers, 1);
        assert_eq!(model.cfg.d_model, 16);
        assert_eq!(model.cfg.d_ff, 32);
        // Single dense block, no MoE.
        assert_eq!(model.weights.blocks.len(), 1);
        assert!(model.weights.moe_blocks.is_none());

        // Per-tensor shape verification.
        let blk = &model.weights.blocks[0];
        assert_eq!(blk.w_q.shape, vec![16, 16]);
        assert_eq!(blk.w_k.shape, vec![16, 16]);
        assert_eq!(blk.w_v.shape, vec![16, 16]);
        assert_eq!(blk.w_o.shape, vec![16, 16]);
        assert_eq!(blk.w_gate.shape, vec![32, 16]);
        assert_eq!(blk.w_up.shape, vec![32, 16]);
        assert_eq!(blk.w_down.shape, vec![16, 32]);
        assert_eq!(blk.attn_norm.len(), 16);
        assert_eq!(blk.ffn_norm.len(), 16);

        // Embeddings + LM head.
        assert_eq!(model.weights.token_embd.shape, vec![4, 16]);
        assert_eq!(model.weights.output_norm.len(), 16);
        let lm_head = model.weights.output.as_ref().expect("lm head loaded");
        assert_eq!(lm_head.shape, vec![4, 16]);

        // Quant trio cells should dequant to (1 - 0) * 0.5 = 0.5 across
        // every f16 entry of w_q.
        let bytes = match &blk.w_q.storage {
            Storage::CpuOwned(b) => b.as_ref().to_vec(),
            _ => panic!("expected CpuOwned"),
        };
        let f16s: &[f16] = cast_slice(&bytes);
        for v in f16s {
            assert!(
                (v.to_f32() - 0.5).abs() < 1e-3,
                "dequant cell got {}",
                v.to_f32()
            );
        }
    }

    #[test]
    fn missing_required_tensor_surfaces_clear_error() {
        // Drop ffn_gate from the blob; the loader must call it out.
        let (cfg_json, blob) = build_minimal_awq_safetensors();
        let cfg = parse_hf_config(&cfg_json).unwrap();
        let mut tensors = convert_safetensors_to_gguf_tensors(&blob).unwrap();
        tensors.retain(|t| t.gguf_name != "blk.0.ffn_gate.weight");
        let err = build_llama_model_from_safetensors(&cfg, tensors)
            .err()
            .expect("build should fail without ffn_gate");
        match err {
            BuildError::MissingTensor(ref n) if n == "blk.0.ffn_gate.weight" => {}
            other => panic!("expected MissingTensor, got {other:?}"),
        }
    }

    #[test]
    fn shape_mismatch_aborts_with_diagnostic_error() {
        // Mutate one tensor's shape so the loader's check fires.
        let (cfg_json, blob) = build_minimal_awq_safetensors();
        let cfg = parse_hf_config(&cfg_json).unwrap();
        let mut tensors = convert_safetensors_to_gguf_tensors(&blob).unwrap();
        for t in tensors.iter_mut() {
            if t.gguf_name == "blk.0.attn_q.weight" {
                // The bytes are now wrong-sized for this shape, but
                // shape-check fires before any byte access. Bump the
                // first dim to surface ShapeMismatch.
                t.shape = vec![17, 16];
                break;
            }
        }
        let err = build_llama_model_from_safetensors(&cfg, tensors)
            .err()
            .expect("build should fail on shape mismatch");
        match err {
            BuildError::ShapeMismatch { ref name, role, .. }
                if name == "blk.0.attn_q.weight" && role == "attn_q" => {}
            other => panic!("expected ShapeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn moe_config_routes_to_moe_builder() {
        // A MoE-enabled cfg no longer bails with MoeUnsupported — it now
        // enters the MoE assembly path. With an empty tensor set that
        // path fails at the first missing slot (`token_embd.weight`),
        // which proves it took the MoE branch (the dense branch would
        // report the same, but the point is the build no longer hard-
        // rejects MoE). See `moe_builder_assembles_stacked_experts` for
        // the positive path.
        use rustllama_models::llama_config::MoeConfig;
        let (cfg_json, _blob) = build_minimal_awq_safetensors();
        let mut cfg = parse_hf_config(&cfg_json).unwrap();
        cfg.moe = Some(MoeConfig {
            n_experts: 8,
            n_experts_used: 2,
            n_experts_shared: 0,
        });
        let err = build_llama_model_from_safetensors(&cfg, Vec::new())
            .err()
            .expect("empty MoE tensor set should fail on a missing slot");
        match err {
            BuildError::MissingTensor(ref n) if n == "token_embd.weight" => {}
            other => panic!("expected MissingTensor(token_embd.weight), got {other:?}"),
        }
    }

    /// Positive MoE path: hand the builder a synthetic stacked-expert
    /// tensor set (F16, 2 experts, no shared expert) and confirm it
    /// assembles `moe_blocks` with correctly-sliced per-expert views.
    #[test]
    fn moe_builder_assembles_stacked_experts() {
        use rustllama_models::llama_config::MoeConfig;
        let d_model = 16usize;
        let d_ff = 8usize; // per-expert FFN width
        let n_experts = 2usize;
        let vocab = 4usize;
        let n_heads = 2usize;
        let head_dim = 8usize; // d_q = d_kv = 16 = d_model

        let f16_bytes = |v: &[f32]| -> Vec<u8> {
            v.iter().flat_map(|x| f16::from_f32(*x).to_le_bytes()).collect()
        };
        let mut ts: Vec<ConvertedTensor> = Vec::new();
        let push_f16 = |ts: &mut Vec<ConvertedTensor>, name: &str, shape: Vec<u64>, n: usize| {
            let vals: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.01).collect();
            ts.push(ConvertedTensor {
                gguf_name: name.into(),
                shape,
                dtype: ConvertedDtype::F16,
                bytes: f16_bytes(&vals),
            });
        };
        push_f16(&mut ts, "token_embd.weight", vec![vocab as u64, d_model as u64], vocab * d_model);
        push_f16(&mut ts, "output_norm.weight", vec![d_model as u64], d_model);
        push_f16(&mut ts, "output.weight", vec![vocab as u64, d_model as u64], vocab * d_model);
        for i in 0..1usize {
            let p = format!("blk.{i}");
            push_f16(&mut ts, &format!("{p}.attn_norm.weight"), vec![d_model as u64], d_model);
            push_f16(&mut ts, &format!("{p}.ffn_norm.weight"), vec![d_model as u64], d_model);
            for proj in ["attn_q", "attn_k", "attn_v", "attn_output"] {
                push_f16(&mut ts, &format!("{p}.{proj}.weight"), vec![d_model as u64, d_model as u64], d_model * d_model);
            }
            // Router [n_experts, d_model].
            push_f16(&mut ts, &format!("{p}.ffn_gate_inp.weight"), vec![n_experts as u64, d_model as u64], n_experts * d_model);
            // Stacked experts: gate/up [n, d_ff, d_model], down [n, d_model, d_ff].
            push_f16(&mut ts, &format!("{p}.ffn_gate_exps.weight"), vec![n_experts as u64, d_ff as u64, d_model as u64], n_experts * d_ff * d_model);
            push_f16(&mut ts, &format!("{p}.ffn_up_exps.weight"), vec![n_experts as u64, d_ff as u64, d_model as u64], n_experts * d_ff * d_model);
            push_f16(&mut ts, &format!("{p}.ffn_down_exps.weight"), vec![n_experts as u64, d_model as u64, d_ff as u64], n_experts * d_model * d_ff);
        }

        let cfg_json = format!(
            r#"{{"architectures":["Qwen2MoeForCausalLM"],"hidden_size":{d_model},
                "intermediate_size":64,"moe_intermediate_size":{d_ff},
                "num_hidden_layers":1,"num_attention_heads":{n_heads},
                "num_key_value_heads":{n_heads},"head_dim":{head_dim},
                "vocab_size":{vocab},"num_experts":{n_experts},
                "num_experts_per_tok":2,"tie_word_embeddings":false}}"#
        );
        let cfg = parse_hf_config(&cfg_json).unwrap();
        assert_eq!(cfg.d_ff, d_ff, "MoE d_ff must be the per-expert width");
        assert!(matches!(cfg.moe, Some(MoeConfig { n_experts: 2, n_experts_used: 2, .. })));

        let model = build_llama_model_from_safetensors(&cfg, ts).expect("MoE build");
        assert!(model.weights.blocks.is_empty());
        let mbs = model.weights.moe_blocks.as_ref().expect("moe_blocks populated");
        assert_eq!(mbs.len(), 1);
        let mb = &mbs[0];
        assert_eq!(mb.gate_per_expert.len(), n_experts);
        assert_eq!(mb.down_per_expert.len(), n_experts);
        // Per-expert views carry the sliced [d_out, d_in] shape.
        assert_eq!(mb.gate_per_expert[0].shape, vec![d_ff as u64, d_model as u64]);
        assert_eq!(mb.down_per_expert[1].shape, vec![d_model as u64, d_ff as u64]);
        assert!(mb.w_gate_shared.is_none(), "no shared expert in this fixture");
        assert!(mb.shared_router.is_none());
    }

    #[test]
    fn tied_embeddings_skips_lm_head_lookup() {
        // Build a tied-embeddings config and a blob without lm_head.
        let (cfg_json, blob) = build_minimal_awq_safetensors();
        let mut cfg = parse_hf_config(&cfg_json).unwrap();
        cfg.tie_word_embeddings = true;
        let mut tensors = convert_safetensors_to_gguf_tensors(&blob).unwrap();
        tensors.retain(|t| t.gguf_name != "output.weight");
        let model =
            build_llama_model_from_safetensors(&cfg, tensors).expect("build");
        assert!(model.weights.output.is_none());
        assert_eq!(model.cfg.tie_word_embeddings, true);
    }

    #[test]
    fn f32_bias_round_trip_to_vec_f32() {
        // Smoke-test the bias path with a hand-built f32 norm tensor.
        // (The minimal AWQ blob doesn't include biases — Llama
        // doesn't carry them.)
        let bytes: Vec<u8> = f32_bytes(&[1.0_f32, 2.0, 3.0, 4.0]);
        let ct = ConvertedTensor {
            gguf_name: "scratch".into(),
            shape: vec![4],
            dtype: ConvertedDtype::F32,
            bytes,
        };
        let v = converted_to_f32_vec(&ct);
        assert_eq!(v, vec![1.0_f32, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn f16_norm_round_trip_to_vec_f32() {
        let h = vec![
            f16::from_f32(0.5),
            f16::from_f32(1.0),
            f16::from_f32(-2.5),
        ];
        let bytes = f16_bytes(&h);
        let ct = ConvertedTensor {
            gguf_name: "scratch".into(),
            shape: vec![3],
            dtype: ConvertedDtype::F16,
            bytes,
        };
        let v = converted_to_f32_vec(&ct);
        assert_eq!(v, vec![0.5, 1.0, -2.5]);
    }
}
