"""
Pure-numpy reference for the Gated DeltaNet layer in qwen35moe.

Mirrors `crates/rustllama-models/src/llama_arch.rs::forward_one_hybrid`
SSM branch + `crates/rustllama-kernels-cpu/src/delta_net.rs` math.

Equations (per fla/ops/gated_delta_rule/naive.py + fla/layers/gated_deltanet.py):

  qkv = W_qkv @ hidden                                  # [2 * ssm_inner]
  gate = W_gate @ hidden                                # [ssm_inner]
  alpha = W_alpha @ hidden                              # [n_v_heads]
  beta = sigmoid(W_beta @ hidden)                       # [n_v_heads]

  qkv = conv1d_depthwise(qkv, conv1d_weight, state)     # [2 * ssm_inner]
  qkv = silu(qkv)                                       # in place

  q = qkv[0 : d_model]
  k = qkv[d_model : 2*d_model]
  v = qkv[2*d_model : 2*ssm_inner]

  For each V-head h:
    decay = -exp(A_log[h]) * softplus(alpha[h] + dt_bias[h])
    kh    = h // (n_v_heads // n_qk_heads)
    q_h   = q[kh*hd : (kh+1)*hd]      # head_qk_dim
    k_h   = k[kh*hd : (kh+1)*hd]
    v_h   = v[h*hd  : (h+1)*hd]       # head_v_dim
    # HF `recurrent_gated_delta_rule(..., use_qk_l2norm_in_kernel=True)`:
    # L2-normalize Q/K per head before the recurrence.
    q_h   = q_h / sqrt(sum(q_h^2) + 1e-6)
    k_h   = k_h / sqrt(sum(k_h^2) + 1e-6)
    S     = exp(decay) * S
    v_til = (v_h - S.T @ k_h) * beta[h]
    S     = S + outer(k_h, v_til)
    o_h   = q_h.T @ S
    o_h   = rmsnorm(o_h) * silu(gate[h*hd:(h+1)*hd])

  v_out_concat = concat(o_h for h in range(n_v_heads))
  out = W_ssm_out @ v_out_concat                        # [d_model]
  return out, new_conv_state, new_recurrent_state

The script generates a fixture: tiny but real-shape weights + input + the
expected output, written to `tests/fixtures/deltanet_layer_v1.json`. A Rust
integration test loads the fixture and runs the same forward through
`delta_net_layer_forward_f32`, asserting parity within FP rounding.

Run: `python scripts/qwen35moe_reference_deltanet.py`
"""

import json
import math
from pathlib import Path

import numpy as np

# Shapes — tiny but real-shape: 8 d_model, 16 ssm_inner, 2 QK heads, 4 V
# heads. Same layout as the existing Rust unit test (so we can cross-
# check inputs trivially).
D_MODEL = 8
SSM_INNER = 16
N_QK_HEADS = 2
N_V_HEADS = 4
HEAD_QK_DIM = D_MODEL // N_QK_HEADS  # 4
HEAD_V_DIM = SSM_INNER // N_V_HEADS  # 4
CONV_KERNEL = 4
RMS_EPS = 1e-5
SEED = 0xC0DE


def softplus(x):
    # Numerically stable softplus.
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def sigmoid(x):
    # Numerically stable sigmoid.
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def silu(x):
    return x * sigmoid(x)


def rmsnorm(x, weight, eps):
    # `x` shape [d], `weight` shape [d]. Returns shape [d].
    var = float(np.mean(x.astype(np.float64) ** 2))
    norm = 1.0 / math.sqrt(var + eps)
    return (x * norm * weight).astype(np.float32)


def conv1d_depthwise_step(x_in, weight, state):
    """Single-step depthwise 1D conv. Returns (out, new_state).

    `weight` shape [channels, kernel] (matches the GGUF `ssm_conv1d.weight`
    layout: ne=[kernel, channels] → numpy shape (channels, kernel) row-major).
    `state` shape [(kernel - 1), channels] — prior `kernel-1` input rows.
    `x_in` shape [channels].
    """
    channels, kernel = weight.shape
    assert state.shape == (kernel - 1, channels)
    assert x_in.shape == (channels,)
    out = np.zeros(channels, dtype=np.float32)
    # window = state rows 0..(k-1) then x_in as the newest row
    for c in range(channels):
        acc = 0.0
        for kk in range(kernel - 1):
            acc += float(weight[c, kk]) * float(state[kk, c])
        acc += float(weight[c, kernel - 1]) * float(x_in[c])
        out[c] = acc
    # Shift state: drop column 0, shift left, place x_in at the last
    # state column (= the new "most-recent prior").
    new_state = np.zeros_like(state)
    if kernel >= 2:
        new_state[: kernel - 2] = state[1:]
        new_state[kernel - 2] = x_in
    return out, new_state


def delta_rule_step(q, k, v, g, beta, S):
    """Single-token gated delta rule update for one V-head.

    Mirrors fla.ops.gated_delta_rule.naive line-for-line:
      h *= g.exp()                          # decay
      v -= (h * k[..., None]).sum(-2)       # v - S^T @ k
      v *= beta
      h += k[:, None] * v[None, :]          # rank-1 update
      o = q @ h                             # output

    `q`, `k` shape [head_qk_dim], `v` shape [head_v_dim],
    `S` shape [head_qk_dim, head_v_dim].
    Returns (o, new_S) where o shape [head_v_dim].
    """
    head_qk_dim = q.shape[0]
    head_v_dim = v.shape[0]
    assert k.shape == (head_qk_dim,)
    assert S.shape == (head_qk_dim, head_v_dim)
    # 1. Decay.
    S2 = (math.exp(float(g)) * S).astype(np.float32)
    # 2. v_tilde = (v - S2^T @ k) * beta.
    v_tilde = (v - S2.T @ k) * float(beta)
    # 3. Rank-1 state update.
    S2 = (S2 + np.outer(k, v_tilde)).astype(np.float32)
    # 4. Output.
    o = (q @ S2).astype(np.float32)
    return o, S2


def deltanet_layer_forward(hidden, weights, state):
    """Full DeltaNet layer forward for one token. Returns (out, new_state)."""
    d = D_MODEL
    ssm_inner = SSM_INNER
    qkv_dim = 2 * ssm_inner

    # Projections.
    qkv_pre = (weights["W_qkv"] @ hidden).astype(np.float32)
    gate = (weights["W_gate"] @ hidden).astype(np.float32)
    alpha = (weights["W_alpha"] @ hidden).astype(np.float32)
    beta_raw = (weights["W_beta"] @ hidden).astype(np.float32)
    beta = sigmoid(beta_raw).astype(np.float32)

    # Conv1d depthwise on qkv + SiLU.
    qkv_conv, new_conv_state = conv1d_depthwise_step(
        qkv_pre, weights["W_conv1d"], state["conv"]
    )
    qkv_conv = silu(qkv_conv).astype(np.float32)

    # HF Qwen3-Next `fix_query_key_value_ordering`: per-K-head
    # interleaved layout. Every K-head's Q/K/V channels sit
    # contiguously in one block of size
    # `2*HEAD_QK_DIM + v_per_qk*HEAD_V_DIM`. NOT a flat
    # `[all_q | all_k | all_v]` layout — the old flat split was
    # correct only for kh=0 by coincidence.
    v_per_qk = N_V_HEADS // N_QK_HEADS
    qkv_stride_per_kh = 2 * HEAD_QK_DIM + v_per_qk * HEAD_V_DIM
    assert qkv_stride_per_kh * N_QK_HEADS == 2 * ssm_inner, (
        f"qkv stride mismatch: {qkv_stride_per_kh}*{N_QK_HEADS} != {2*ssm_inner}"
    )

    # Per-head delta rule + gated RMSNorm.
    v_out_concat = np.zeros(ssm_inner, dtype=np.float32)
    new_S = []
    for h in range(N_V_HEADS):
        decay = -math.exp(float(weights["A_log"][h])) * softplus(
            float(alpha[h]) + float(weights["dt_bias"][h])
        )
        kh = h // v_per_qk
        v_within_kh = h % v_per_qk
        kh_base = kh * qkv_stride_per_kh
        q_off = kh_base
        k_off = kh_base + HEAD_QK_DIM
        v_off = kh_base + 2 * HEAD_QK_DIM + v_within_kh * HEAD_V_DIM
        q_h = qkv_conv[q_off : q_off + HEAD_QK_DIM]
        k_h = qkv_conv[k_off : k_off + HEAD_QK_DIM]
        # HF `recurrent_gated_delta_rule(..., use_qk_l2norm_in_kernel=True)`
        # L2-normalizes Q and K per head before the recurrence. Without
        # this the recurrent state magnitude grows unboundedly with
        # sequence length and the residual stream gets corrupted.
        q_h = q_h / math.sqrt(float(np.sum(q_h * q_h)) + 1e-6)
        k_h = k_h / math.sqrt(float(np.sum(k_h * k_h)) + 1e-6)
        q_h = q_h.astype(np.float32)
        k_h = k_h.astype(np.float32)
        v_h = qkv_conv[v_off : v_off + HEAD_V_DIM]
        o_h, S_new = delta_rule_step(
            q_h, k_h, v_h, decay, beta[h], state["S"][h]
        )
        # Gated RMSNorm: rmsnorm(o_h) * silu(gate_h).
        gate_h = gate[h * HEAD_V_DIM : (h + 1) * HEAD_V_DIM]
        o_h_normed = rmsnorm(o_h, weights["ssm_norm"], RMS_EPS)
        o_h_normed = (o_h_normed * silu(gate_h)).astype(np.float32)
        v_out_concat[h * HEAD_V_DIM : (h + 1) * HEAD_V_DIM] = o_h_normed
        new_S.append(S_new)

    # Output projection.
    out = (weights["W_out"] @ v_out_concat).astype(np.float32)
    return out, {"conv": new_conv_state, "S": new_S}


def main():
    rng = np.random.default_rng(SEED)

    # Scale 0.3 — large enough that intermediate values stay well
    # above FP32 precision floor through the full layer composition
    # (4 successive matvecs + delta-rule + rmsnorm), small enough
    # that nothing overflows. Picked empirically by bumping until
    # output magnitudes land in the 1e-3..1e-1 range so a 1e-4
    # tolerance is meaningful.
    def mk(shape):
        return (rng.standard_normal(shape) * 0.3).astype(np.float32)

    qkv_dim = 2 * SSM_INNER
    weights = {
        "W_qkv": mk((qkv_dim, D_MODEL)),
        "W_gate": mk((SSM_INNER, D_MODEL)),
        "W_alpha": mk((N_V_HEADS, D_MODEL)),
        "W_beta": mk((N_V_HEADS, D_MODEL)),
        "W_conv1d": mk((qkv_dim, CONV_KERNEL)),
        "A_log": np.full(N_V_HEADS, math.log(0.5), dtype=np.float32),
        "dt_bias": np.zeros(N_V_HEADS, dtype=np.float32),
        "ssm_norm": np.ones(HEAD_V_DIM, dtype=np.float32),
        "W_out": mk((D_MODEL, SSM_INNER)),
    }

    # Two input tokens — to exercise state carry across calls.
    hidden_a = (rng.standard_normal(D_MODEL) * 0.5).astype(np.float32)
    hidden_b = (rng.standard_normal(D_MODEL) * 0.5).astype(np.float32)

    # Initial state.
    state = {
        "conv": np.zeros((CONV_KERNEL - 1, qkv_dim), dtype=np.float32),
        "S": [np.zeros((HEAD_QK_DIM, HEAD_V_DIM), dtype=np.float32) for _ in range(N_V_HEADS)],
    }

    out_a, state = deltanet_layer_forward(hidden_a, weights, state)
    out_b, _state = deltanet_layer_forward(hidden_b, weights, state)

    fixture = {
        "shapes": {
            "d_model": D_MODEL,
            "ssm_inner": SSM_INNER,
            "n_qk_heads": N_QK_HEADS,
            "n_v_heads": N_V_HEADS,
            "head_qk_dim": HEAD_QK_DIM,
            "head_v_dim": HEAD_V_DIM,
            "conv_kernel": CONV_KERNEL,
            "rms_eps": RMS_EPS,
        },
        "weights": {k: v.flatten().tolist() for k, v in weights.items()},
        "hidden_a": hidden_a.tolist(),
        "hidden_b": hidden_b.tolist(),
        "expected_out_a": out_a.tolist(),
        "expected_out_b": out_b.tolist(),
    }

    out_path = Path(__file__).parent.parent / "crates" / "rustllama-models" / "tests" / "fixtures" / "deltanet_layer_v1.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(fixture, f, indent=2)
    print(f"Wrote fixture: {out_path}")
    print(f"  expected_out_a[0..4] = {out_a[:4]}")
    print(f"  expected_out_b[0..4] = {out_b[:4]}")


if __name__ == "__main__":
    main()
