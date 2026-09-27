"""
Pure-numpy reference for the MoE FFN used by qwen35moe (and by every
other transformer-MoE arch routed through `moe_ffn_one_into_parts`).

Mirrors `crates/rustllama-models/src/moe.rs` line-for-line:

  expert_logits = softmax(router @ hidden)        # over all n_experts
  picks = top_k indices, renormalized over the top-K → weights sum to 1.0

  For each picked expert e with weight w_e:
    gate_e = silu(W_gate[e] @ hidden)
    up_e   = W_up[e] @ hidden
    ff_e   = gate_e * up_e
    out   += w_e * (W_down[e] @ ff_e)

  Optional shared expert (qwen35moe convention):
    shared_w = sigmoid(W_shexp_router @ hidden)   # scalar (router is [1, d_model])
    out += shared_w * SwiGLU(W_gate_sh, W_up_sh, W_down_sh, hidden)

  (DeepSeek-V3 convention is shared_w = 1.0 always — selected here via
   `--shared-router`.)

Produces a fixture `tests/fixtures/moe_ffn_v1.json` with weights, the
hidden input, picked-expert indices + weights, and the expected output.

Run: `python scripts/qwen35moe_reference_moe.py`
"""

import json
import math
from pathlib import Path

import numpy as np

D_MODEL = 8
D_FF = 16
N_EXPERTS = 4
TOP_K = 2
SEED = 0xBEEF


def sigmoid(x):
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def silu(x):
    return x * sigmoid(x)


def softmax(x):
    m = float(np.max(x))
    e = np.exp(x - m)
    return e / float(np.sum(e))


def route_topk(hidden, router, n_experts, top_k):
    """Mirrors `route_topk_into` in moe.rs.

    1. logits = router @ hidden
    2. softmax over all experts
    3. pick top-K (idx, weight) pairs, sorted by descending weight
       with index as tiebreaker (matches the Rust sort_by `then_with(|| a.0.cmp(&b.0))`)
    4. renormalize weights over the top-K so they sum to 1.0
    """
    logits = router @ hidden
    probs = softmax(logits)
    # Match Rust's stable sort: descending by weight, ties broken by
    # ascending index.
    indexed = list(enumerate(probs.tolist()))
    indexed.sort(key=lambda p: (-p[1], p[0]))
    picks = indexed[:top_k]
    tot = sum(w for _, w in picks)
    if tot > 0:
        picks = [(i, w / tot) for i, w in picks]
    return picks


def expert_ffn(hidden, w_gate, w_up, w_down):
    """SwiGLU FFN. Shapes: w_gate, w_up [d_ff, d_model]; w_down [d_model, d_ff]."""
    gate = silu(w_gate @ hidden).astype(np.float32)
    up = (w_up @ hidden).astype(np.float32)
    ff = (gate * up).astype(np.float32)
    return (w_down @ ff).astype(np.float32)


def moe_ffn_forward(
    hidden,
    router,
    gate_per_expert,
    up_per_expert,
    down_per_expert,
    shared_router=None,
    w_gate_shared=None,
    w_up_shared=None,
    w_down_shared=None,
):
    n_experts = router.shape[0]
    picks = route_topk(hidden, router, n_experts, TOP_K)
    out = np.zeros(D_MODEL, dtype=np.float32)
    for idx, weight in picks:
        out = out + weight * expert_ffn(
            hidden, gate_per_expert[idx], up_per_expert[idx], down_per_expert[idx]
        )
    if w_gate_shared is not None:
        shared_w = 1.0
        if shared_router is not None:
            # shared_router shape [1, d_model], yields a scalar.
            raw = float((shared_router @ hidden).item())
            shared_w = float(sigmoid(np.array([raw])).item())
        out = out + shared_w * expert_ffn(
            hidden, w_gate_shared, w_up_shared, w_down_shared
        )
    return out.astype(np.float32), picks


def main():
    rng = np.random.default_rng(SEED)

    def mk(shape):
        return (rng.standard_normal(shape) * 0.3).astype(np.float32)

    router = mk((N_EXPERTS, D_MODEL))
    # 3D layout: [n_experts, d_ff, d_model] for gate/up; [n_experts, d_model, d_ff] for down.
    gate_per_expert = [mk((D_FF, D_MODEL)) for _ in range(N_EXPERTS)]
    up_per_expert = [mk((D_FF, D_MODEL)) for _ in range(N_EXPERTS)]
    down_per_expert = [mk((D_MODEL, D_FF)) for _ in range(N_EXPERTS)]

    # Shared expert + its router.
    w_gate_shared = mk((D_FF, D_MODEL))
    w_up_shared = mk((D_FF, D_MODEL))
    w_down_shared = mk((D_MODEL, D_FF))
    shared_router = mk((1, D_MODEL))

    hidden = (rng.standard_normal(D_MODEL) * 0.5).astype(np.float32)

    # Two fixture variants: with and without the shared-router gate,
    # so the Rust test exercises both code paths (qwen35moe's
    # router-gated convention AND DeepSeek-V3's always-on shared
    # expert convention).
    out_gated, picks_gated = moe_ffn_forward(
        hidden, router, gate_per_expert, up_per_expert, down_per_expert,
        shared_router=shared_router,
        w_gate_shared=w_gate_shared,
        w_up_shared=w_up_shared,
        w_down_shared=w_down_shared,
    )
    out_always_on, picks_always_on = moe_ffn_forward(
        hidden, router, gate_per_expert, up_per_expert, down_per_expert,
        shared_router=None,  # always-on (DeepSeek-V3)
        w_gate_shared=w_gate_shared,
        w_up_shared=w_up_shared,
        w_down_shared=w_down_shared,
    )

    fixture = {
        "shapes": {
            "d_model": D_MODEL,
            "d_ff": D_FF,
            "n_experts": N_EXPERTS,
            "top_k": TOP_K,
        },
        "weights": {
            "router": router.flatten().tolist(),
            # Per-expert tensors stored flat as the model layout:
            # `[n_experts, d_ff, d_model]` for gate/up, `[n_experts, d_model, d_ff]` for down.
            "gate_per_expert_flat": np.stack(gate_per_expert).flatten().tolist(),
            "up_per_expert_flat": np.stack(up_per_expert).flatten().tolist(),
            "down_per_expert_flat": np.stack(down_per_expert).flatten().tolist(),
            "w_gate_shared": w_gate_shared.flatten().tolist(),
            "w_up_shared": w_up_shared.flatten().tolist(),
            "w_down_shared": w_down_shared.flatten().tolist(),
            "shared_router": shared_router.flatten().tolist(),
        },
        "hidden": hidden.tolist(),
        "picks_gated": picks_gated,
        "expected_out_gated": out_gated.tolist(),
        "picks_always_on": picks_always_on,
        "expected_out_always_on": out_always_on.tolist(),
    }

    out_path = (
        Path(__file__).parent.parent
        / "crates" / "rustllama-models" / "tests" / "fixtures"
        / "moe_ffn_v1.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(fixture, f, indent=2)
    print(f"Wrote fixture: {out_path}")
    print(f"  picks_gated: {picks_gated}")
    print(f"  expected_out_gated[0..4] = {out_gated[:4]}")
    print(f"  expected_out_always_on[0..4] = {out_always_on[:4]}")


if __name__ == "__main__":
    main()
