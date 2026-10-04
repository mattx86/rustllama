# GGUF Quantization Formats

This doc details rustllama's four *core* GGUF tensor formats (F16, Q8_0,
Q4_K_M, Q5_K_M) in depth. The loader decodes ~26 formats in total via
reference decoders in `crates/rustllama-gguf/src/dequant.rs`; a genuinely
unsupported type is rejected at load with a clear error. The full set:

- **Float:** F16, BF16.
- **Legacy `_0`/`_1` blocks (32/block):** Q4_0, Q4_1, Q5_0, Q5_1, Q8_0.
- **K-quants (256/super-block):** Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K.
- **IQ-family:** IQ1_S/M, IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_NL/XS (grid-codebook +
  sign-table decoders).
- **Ternary / packed:** TQ1_0, TQ2_0 (ggml-compatible) plus our own packed
  PTQ1_0 (base-3 trits, 128/block) and PQ2_0 (2-bit, 128/block).
- **Microscaling (OCP) + FP8:** MXFP4, MXFP6, MXFP8, NVFP4 (E2M1/E3M2/E4M3
  elements with E8M0/E4M3 block scales), and FP8 (OCP E4M3 / E5M2 and the
  NVIDIA variant).

Each format has a byte-exact decoder; the SYCL, CUDA, and Metal GPU kernels
decode the **same** on-disk layouts (no host-side pre-dequant), verified against
the CPU reference by the `doctor --{sycl,cuda,metal,cpu}-parity` harnesses.

## F16 — `GGML_TYPE_F16` (1)

Plain IEEE 754 binary16. 2 bytes per value. No block structure.

Layout: `[half; N]` stored as `[u8; 2*N]` little-endian.

## Q8_0 — `GGML_TYPE_Q8_0` (8)

32-element blocks. Each block is `{ d: half, qs: [i8; 32] }` = 34 bytes per
32 values = ~8.5 bits per weight.

Dequant: `value[i] = d * qs[i]`.

## Q4_K_M — uses `GGML_TYPE_Q4_K` (12) plus token-embedding F16 mix

256-element super-blocks. Each super-block is split into 8 sub-blocks of 32.

Super-block: `{ d: half, dmin: half, scales: [u8; 12], qs: [u8; 128] }` = 144
bytes per 256 values = ~4.5 bits per weight.

`scales[12]` encodes 8 (scale, min) pairs each as 6-bit values. `qs[128]` packs
two 4-bit weights per byte.

Dequant for sub-block `j` and weight index `i` within it:
  `value = d * scale[j] * q4_value - dmin * min[j]`

## Q5_K_M — uses `GGML_TYPE_Q5_K` (13) plus token-embedding F16 mix

256-element super-blocks. 5 bits per weight (4 bits in `qs`, 1 bit per weight
in a separate `qh` field). 176 bytes per 256 values = ~5.5 bits per weight.

## "Mixed" variants

`Q4_K_M` and `Q5_K_M` in the wild are *mixes*: most layers in Q4_K (or Q5_K),
with token embedding and `output.weight` kept at higher precision (F16 or Q6_K).
Our v1 loader handles a per-tensor dtype dispatch, so a model where most
tensors are Q4_K and a few are F16 loads correctly.

## Reference

Authoritative format definitions live in the ggml repository
(`ggml-quants.{c,h}`). Our pure-Rust decoders match those layouts byte for
byte; parity is verified in `crates/rustllama-gguf/tests/` against golden
dumps produced by `gguf-py`.
