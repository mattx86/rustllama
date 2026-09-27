"""
Pure-numpy reference for the NextN/MTP composition used by qwen35moe.

Mirrors `nextn_compose_logits_f32` in
`crates/rustllama-models/src/llama_arch.rs`:

  e_next = token_embd[next_token_id]                # [d_model]
  e_n    = rmsnorm(e_next, head.embed_norm, eps)    # [d_model]
  h_n    = rmsnorm(hidden, head.hidden_norm, eps)   # [d_model]
  concat = [e_n; h_n]                                # [2 * d_model]
  proj   = head.eh_proj @ concat                    # [d_model]
  shared = rmsnorm(proj, head.shared_head_norm, eps) # [d_model]
  logits = lm_head @ shared                         # [vocab]

Produces `tests/fixtures/nextn_v1.json` with weights, the post-block
hidden state, a next-token id, and the expected logits.

Run: `python scripts/qwen35moe_reference_nextn.py`
"""

import json
import math
from pathlib import Path

import numpy as np

D_MODEL = 8
VOCAB = 32
RMS_EPS = 1e-5
NEXT_TOKEN_ID = 7
SEED = 0xCAFE


def rmsnorm(x, weight, eps):
    var = float(np.mean(x.astype(np.float64) ** 2))
    norm = 1.0 / math.sqrt(var + eps)
    return (x * norm * weight).astype(np.float32)


def main():
    rng = np.random.default_rng(SEED)

    def mk(shape):
        return (rng.standard_normal(shape) * 0.3).astype(np.float32)

    token_embd = mk((VOCAB, D_MODEL))
    embed_norm = (rng.standard_normal(D_MODEL).astype(np.float32) * 0.1 + 1.0)
    hidden_norm = (rng.standard_normal(D_MODEL).astype(np.float32) * 0.1 + 1.0)
    shared_head_norm = (rng.standard_normal(D_MODEL).astype(np.float32) * 0.1 + 1.0)
    eh_proj = mk((D_MODEL, 2 * D_MODEL))
    lm_head = mk((VOCAB, D_MODEL))

    hidden = (rng.standard_normal(D_MODEL).astype(np.float32) * 0.4)

    # Composition.
    e_next = token_embd[NEXT_TOKEN_ID]
    e_n = rmsnorm(e_next, embed_norm, RMS_EPS)
    h_n = rmsnorm(hidden, hidden_norm, RMS_EPS)
    concat = np.concatenate([e_n, h_n])
    proj = (eh_proj @ concat).astype(np.float32)
    shared = rmsnorm(proj, shared_head_norm, RMS_EPS)
    logits = (lm_head @ shared).astype(np.float32)

    fixture = {
        "shapes": {
            "d_model": D_MODEL,
            "vocab": VOCAB,
            "rms_eps": RMS_EPS,
            "next_token_id": NEXT_TOKEN_ID,
        },
        "weights": {
            "token_embd": token_embd.flatten().tolist(),
            "embed_norm": embed_norm.flatten().tolist(),
            "hidden_norm": hidden_norm.flatten().tolist(),
            "shared_head_norm": shared_head_norm.flatten().tolist(),
            "eh_proj": eh_proj.flatten().tolist(),
            "lm_head": lm_head.flatten().tolist(),
        },
        "hidden": hidden.tolist(),
        "expected_logits": logits.tolist(),
    }

    out_path = (
        Path(__file__).parent.parent
        / "crates" / "rustllama-models" / "tests" / "fixtures"
        / "nextn_v1.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(fixture, f, indent=2)
    print(f"Wrote fixture: {out_path}")
    print(f"  expected_logits[0..4] = {logits[:4]}")


if __name__ == "__main__":
    main()
