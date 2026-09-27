//! Opt-in SYCL accelerator: a thin wrapper over the kernels-sycl FFI
//! that takes the f32 buffers the engine works with and routes them
//! through the GPU kernels, converting to/from f16-bit-pattern u16
//! arrays on each side. CPU fallback is built in — every call site
//! can ask the accelerator to run the op, and on a host without SYCL
//! (or without a usable device) the accelerator returns
//! `SyclAccelError::Unavailable` and the caller stays on the CPU
//! path.
//!
//! Status: **scaffold + parity harness, not yet wired into the model
//! forward pass.** The forward pass in `rustllama-models::llama_arch`
//! still routes every op through `rustllama-kernels-cpu`. Once the
//! `parity` test below passes on the user's hardware for each kernel,
//! we'll flip individual dispatches inside `forward_one` /
//! `forward_prefill` to consult the accelerator and only fall back to
//! CPU on `Unavailable`. The intentional sequencing — parity *before*
//! integration — is what keeps the GPU path from silently producing
//! wrong logits.
//!
//! The SYCL backend is always compiled (real-only), so
//! `try_new(device_index)` either returns a real accelerator backed by a
//! `SyclStream`, or surfaces `Unavailable` / `NoSuchDevice` when oneAPI
//! can't see a GPU at this index — in which case engine code that wires
//! through the accelerator gets the CPU fallback for free.

use half::f16;
use rustllama_kernels_sycl::{self as sk, SyclError, SyclStream};

#[derive(Debug, thiserror::Error)]
pub enum SyclAccelError {
    /// Either oneAPI isn't installed at runtime, or the SYCL runtime
    /// reports zero GPU devices. Callers should fall back to the CPU
    /// kernels. This isn't a hard error — it's the expected outcome
    /// on any host that doesn't have an Intel GPU.
    #[error("SYCL accelerator unavailable: {0}")]
    Unavailable(String),
    /// A real device-index error from the SYCL runtime — the caller
    /// asked for `sycl:7` but only `sycl:0` exists, that sort of
    /// thing. Distinct from `Unavailable` so config-validator code
    /// can surface a more specific message.
    #[error("SYCL device index {0} not available")]
    NoSuchDevice(u32),
    /// Shape mismatch the FFI rejected (e.g. RoPE with odd
    /// `head_dim`). Wrapped here so callers don't depend on
    /// `rustllama-kernels-sycl::SyclError` directly.
    #[error("invalid kernel shape: {0}")]
    InvalidShape(String),
    /// A kernel ran but produced output the parity harness flagged as
    /// out-of-tolerance against the CPU reference. Currently only
    /// raised by [`SyclAccelerator::parity_check_rmsnorm`] et al.
    #[error("parity check failed: {0}")]
    ParityFailed(String),
}

pub type Result<T> = std::result::Result<T, SyclAccelError>;

/// Owns a SYCL queue / stream and exposes the engine-facing kernel
/// surface. `!Send + !Sync` because the underlying `SyclStream` is
/// tied to its creating thread (SYCL has thread-local state).
pub struct SyclAccelerator {
    stream: SyclStream,
    device_index: u32,
}

impl std::fmt::Debug for SyclAccelerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SyclAccelerator")
            .field("device_index", &self.device_index)
            .finish_non_exhaustive()
    }
}

impl SyclAccelerator {
    /// Create an accelerator bound to a specific SYCL GPU index.
    /// Returns `Unavailable` on mock-mode builds; `NoSuchDevice` when
    /// the runtime is real but the index doesn't exist.
    pub fn try_new(device_index: u32) -> Result<Self> {
        match sk::create_stream(device_index) {
            Ok(stream) => Ok(Self { stream, device_index }),
            Err(SyclError::Unavailable) => {
                Err(SyclAccelError::Unavailable("kernels-sycl in mock mode".into()))
            }
            Err(SyclError::NoSuchDevice(i)) => Err(SyclAccelError::NoSuchDevice(i)),
            Err(SyclError::InvalidShape(s)) => Err(SyclAccelError::InvalidShape(s)),
            Err(SyclError::Runtime(s)) => Err(SyclAccelError::InvalidShape(s)),
            // create_stream doesn't produce this variant — it comes
            // out of the L0 import API, which doesn't flow through
            // here. Cover for exhaustiveness.
            Err(SyclError::L0ImportUnsupported(code)) => {
                Err(SyclAccelError::InvalidShape(format!("unexpected L0 import error {code}")))
            }
        }
    }

    pub fn device_index(&self) -> u32 {
        self.device_index
    }

    /// Per-row RMSNorm in f32 in, f32 out. Internally converts each
    /// row to f16, calls the SYCL kernel, and converts back. The
    /// engine works in f32 today; once full f16 storage lands in
    /// `rustllama-tensor` we can drop the conversion overhead.
    pub fn rmsnorm_f32(
        &mut self,
        x: &[f32],
        w: &[f32],
        y: &mut [f32],
        n_rows: u32,
        d: u32,
        eps: f32,
    ) -> Result<()> {
        let total = (n_rows as usize) * (d as usize);
        if x.len() < total || y.len() < total || w.len() < d as usize {
            return Err(SyclAccelError::InvalidShape(format!(
                "rmsnorm: x={}, w={}, y={}, need {n_rows}x{d}",
                x.len(),
                w.len(),
                y.len()
            )));
        }
        let x_bits: Vec<u16> = x[..total].iter().map(|v| f16::from_f32(*v).to_bits()).collect();
        let w_bits: Vec<u16> =
            w[..d as usize].iter().map(|v| f16::from_f32(*v).to_bits()).collect();
        let mut y_bits = vec![0u16; total];
        sk::rmsnorm(&mut self.stream, &x_bits, &w_bits, &mut y_bits, n_rows, d, eps)
            .map_err(map_sycl_err)?;
        for (i, bits) in y_bits.iter().enumerate() {
            y[i] = f16::from_bits(*bits).to_f32();
        }
        Ok(())
    }

    /// Gather rows from `table[V, d]` indexed by `ids[N]` into
    /// `out[N, d]`. f32-facing wrapper around the SYCL kernel.
    pub fn embedding_lookup_f32(
        &mut self,
        table: &[f32],
        ids: &[i32],
        out: &mut [f32],
        vocab: usize,
        d: u32,
    ) -> Result<()> {
        let d_us = d as usize;
        if table.len() < vocab * d_us {
            return Err(SyclAccelError::InvalidShape(format!(
                "embed: table={}, need {vocab}*{d_us} = {}",
                table.len(),
                vocab * d_us
            )));
        }
        if out.len() < ids.len() * d_us {
            return Err(SyclAccelError::InvalidShape(format!(
                "embed: out={}, need {}*{d_us}",
                out.len(),
                ids.len()
            )));
        }
        let table_bits: Vec<u16> =
            table.iter().map(|v| f16::from_f32(*v).to_bits()).collect();
        let mut out_bits = vec![0u16; ids.len() * d_us];
        sk::embedding_lookup(&mut self.stream, &table_bits, ids, &mut out_bits, d)
            .map_err(map_sycl_err)?;
        for (i, bits) in out_bits.iter().enumerate() {
            out[i] = f16::from_bits(*bits).to_f32();
        }
        Ok(())
    }

    /// SwiGLU activation `out[i] = silu(x[i]) * y[i]`. f32-facing
    /// wrapper. Used in the FFN's gate × up product.
    pub fn silu_mul_f32(
        &mut self,
        x: &[f32],
        y: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        if x.len() != y.len() || out.len() < x.len() {
            return Err(SyclAccelError::InvalidShape(format!(
                "silu_mul: x={}, y={}, out={}",
                x.len(),
                y.len(),
                out.len()
            )));
        }
        let x_bits: Vec<u16> = x.iter().map(|v| f16::from_f32(*v).to_bits()).collect();
        let y_bits: Vec<u16> = y.iter().map(|v| f16::from_f32(*v).to_bits()).collect();
        let mut out_bits = vec![0u16; x.len()];
        sk::silu_mul(&mut self.stream, &x_bits, &y_bits, &mut out_bits)
            .map_err(map_sycl_err)?;
        for (i, bits) in out_bits.iter().enumerate() {
            out[i] = f16::from_bits(*bits).to_f32();
        }
        Ok(())
    }

    /// Half-split RoPE on `qk: [n_heads, head_dim]`, in-place. f32
    /// in/out; the kernel runs in f16. `head_dim` must be even.
    pub fn rope_f32(
        &mut self,
        qk: &mut [f32],
        n_heads: u32,
        head_dim: u32,
        pos: u32,
        inv_freq: &[f32],
    ) -> Result<()> {
        if head_dim % 2 != 0 {
            return Err(SyclAccelError::InvalidShape(format!(
                "rope: head_dim must be even, got {head_dim}"
            )));
        }
        let total = (n_heads as usize) * (head_dim as usize);
        if qk.len() < total {
            return Err(SyclAccelError::InvalidShape(format!(
                "rope: qk={}, need n_heads*head_dim={total}",
                qk.len()
            )));
        }
        let half = (head_dim / 2) as usize;
        if inv_freq.len() < half {
            return Err(SyclAccelError::InvalidShape(format!(
                "rope: inv_freq={}, need head_dim/2={half}",
                inv_freq.len()
            )));
        }
        let mut qk_bits: Vec<u16> = qk[..total]
            .iter()
            .map(|v| f16::from_f32(*v).to_bits())
            .collect();
        let freq_bits: Vec<u16> = inv_freq[..half]
            .iter()
            .map(|v| f16::from_f32(*v).to_bits())
            .collect();
        sk::rope(&mut self.stream, &mut qk_bits, n_heads, head_dim, pos, &freq_bits)
            .map_err(map_sycl_err)?;
        for (i, bits) in qk_bits.iter().enumerate() {
            qk[i] = f16::from_bits(*bits).to_f32();
        }
        Ok(())
    }

    /// In-place softmax over `kv_len` axis of
    /// `scores: [n_heads, seq, kv_len]`. Optional additive mask of
    /// shape `[kv_len]` (typically a causal `-inf` mask).
    pub fn softmax_attn_f32(
        &mut self,
        scores: &mut [f32],
        mask: Option<&[f32]>,
        n_heads: u32,
        seq: u32,
        kv_len: u32,
        scale: f32,
    ) -> Result<()> {
        let total =
            (n_heads as usize) * (seq as usize) * (kv_len as usize);
        if scores.len() < total {
            return Err(SyclAccelError::InvalidShape(format!(
                "softmax_attn: scores={}, need n_heads*seq*kv_len={total}",
                scores.len()
            )));
        }
        let mut scores_bits: Vec<u16> = scores[..total]
            .iter()
            .map(|v| f16::from_f32(*v).to_bits())
            .collect();
        let mask_bits: Option<Vec<u16>> = mask.map(|m| {
            m[..kv_len as usize]
                .iter()
                .map(|v| f16::from_f32(*v).to_bits())
                .collect()
        });
        sk::softmax_attn(
            &mut self.stream,
            &mut scores_bits,
            mask_bits.as_deref(),
            n_heads,
            seq,
            kv_len,
            scale,
        )
        .map_err(map_sycl_err)?;
        for (i, bits) in scores_bits.iter().enumerate() {
            scores[i] = f16::from_bits(*bits).to_f32();
        }
        Ok(())
    }

    /// F16 GEMM: `C[M,N] = A[M,K] @ B[K,N]`, row-major. f32 in/out;
    /// the kernel runs in f16 with f32 accumulation on the chip.
    pub fn gemm_f16_f32(
        &mut self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<()> {
        let m_us = m as usize;
        let n_us = n as usize;
        let k_us = k as usize;
        if a.len() < m_us * k_us || b.len() < k_us * n_us || c.len() < m_us * n_us {
            return Err(SyclAccelError::InvalidShape(format!(
                "gemm: a={}, b={}, c={}, need {m}*{k}, {k}*{n}, {m}*{n}",
                a.len(),
                b.len(),
                c.len(),
            )));
        }
        let a_bits: Vec<u16> = a[..m_us * k_us]
            .iter()
            .map(|v| f16::from_f32(*v).to_bits())
            .collect();
        let b_bits: Vec<u16> = b[..k_us * n_us]
            .iter()
            .map(|v| f16::from_f32(*v).to_bits())
            .collect();
        let mut c_bits = vec![0u16; m_us * n_us];
        sk::gemm_f16(&mut self.stream, &a_bits, &b_bits, &mut c_bits, m, n, k)
            .map_err(map_sycl_err)?;
        for (i, bits) in c_bits.iter().enumerate() {
            c[i] = f16::from_bits(*bits).to_f32();
        }
        Ok(())
    }

    /// Parity smoke check: runs the SYCL `rmsnorm` against the
    /// pure-Rust reference and reports the max absolute error.
    /// Returns `Ok(max_abs_err)` if within `tolerance`, else
    /// `ParityFailed`. Used by `xtask doctor --kernels`.
    pub fn parity_check_rmsnorm(
        &mut self,
        n_rows: u32,
        d: u32,
        tolerance: f32,
    ) -> Result<f32> {
        // Deterministic synthetic input so the check is reproducible.
        let total = (n_rows as usize) * (d as usize);
        let x: Vec<f32> = (0..total)
            .map(|i| ((i % 31) as f32 - 15.0) * 0.1)
            .collect();
        let w: Vec<f32> = (0..d as usize).map(|i| 1.0 + (i as f32 / d as f32) * 0.5).collect();
        let mut y_gpu = vec![0f32; total];
        self.rmsnorm_f32(&x, &w, &mut y_gpu, n_rows, d, 1e-5)?;
        // CPU reference: full f32 RMS, then f16 round-trip per element
        // to match the GPU kernel's f16 storage on input.
        let mut y_cpu = vec![0f32; total];
        for r in 0..n_rows as usize {
            let base = r * d as usize;
            let mut sum_sq = 0f32;
            for i in 0..d as usize {
                let v = f16::from_f32(x[base + i]).to_f32();
                sum_sq += v * v;
            }
            let scale = 1.0f32 / (sum_sq / (d as f32) + 1e-5f32).sqrt();
            for i in 0..d as usize {
                let v = f16::from_f32(x[base + i]).to_f32();
                let wv = f16::from_f32(w[i]).to_f32();
                let z = v * scale * wv;
                // GPU stores back as f16, so round-trip here too.
                y_cpu[base + i] = f16::from_f32(z).to_f32();
            }
        }
        let mut max_err = 0f32;
        for (g, c) in y_gpu.iter().zip(y_cpu.iter()) {
            let e = (g - c).abs();
            if e > max_err {
                max_err = e;
            }
        }
        if max_err > tolerance {
            return Err(SyclAccelError::ParityFailed(format!(
                "rmsnorm n_rows={n_rows} d={d}: max_abs_err={max_err} > tol={tolerance}"
            )));
        }
        Ok(max_err)
    }

    /// Parity smoke check for embedding_lookup. Generates a synthetic
    /// `[vocab, d]` f32 table and queries a handful of ids against it.
    pub fn parity_check_embedding(
        &mut self,
        vocab: usize,
        d: u32,
        tolerance: f32,
    ) -> Result<f32> {
        let d_us = d as usize;
        let table: Vec<f32> = (0..vocab * d_us)
            .map(|i| ((i as f32).sin() * 2.0).cos())
            .collect();
        let ids: Vec<i32> = (0..16)
            .map(|i| ((i * 31 + 7) % vocab as i32) as i32)
            .collect();
        let mut out_gpu = vec![0f32; ids.len() * d_us];
        self.embedding_lookup_f32(&table, &ids, &mut out_gpu, vocab, d)?;
        let mut out_cpu = vec![0f32; ids.len() * d_us];
        for (i, &id) in ids.iter().enumerate() {
            if id < 0 {
                continue;
            }
            let src = (id as usize) * d_us;
            for j in 0..d_us {
                // CPU reference also f16-round-trips to match the GPU
                // table's f16 storage.
                out_cpu[i * d_us + j] = f16::from_f32(table[src + j]).to_f32();
            }
        }
        let mut max_err = 0f32;
        for (g, c) in out_gpu.iter().zip(out_cpu.iter()) {
            let e = (g - c).abs();
            if e > max_err {
                max_err = e;
            }
        }
        if max_err > tolerance {
            return Err(SyclAccelError::ParityFailed(format!(
                "embedding vocab={vocab} d={d}: max_abs_err={max_err} > tol={tolerance}"
            )));
        }
        Ok(max_err)
    }

    /// Parity smoke for SwiGLU: deterministic input, compare GPU to a
    /// CPU reference that f16-round-trips inputs so the comparison
    /// reflects the kernel's compute, not the conversion.
    pub fn parity_check_silu_mul(&mut self, n: usize, tolerance: f32) -> Result<f32> {
        let x: Vec<f32> = (0..n).map(|i| ((i % 41) as f32 - 20.0) * 0.07).collect();
        let y: Vec<f32> = (0..n).map(|i| ((i % 37) as f32 - 18.0) * 0.05).collect();
        let mut out_gpu = vec![0f32; n];
        self.silu_mul_f32(&x, &y, &mut out_gpu)?;
        let mut out_cpu = vec![0f32; n];
        for i in 0..n {
            let xv = f16::from_f32(x[i]).to_f32();
            let yv = f16::from_f32(y[i]).to_f32();
            let silu = xv / (1.0 + (-xv).exp());
            out_cpu[i] = f16::from_f32(silu * yv).to_f32();
        }
        let mut max_err = 0f32;
        for (g, c) in out_gpu.iter().zip(out_cpu.iter()) {
            let e = (g - c).abs();
            if e > max_err {
                max_err = e;
            }
        }
        if max_err > tolerance {
            return Err(SyclAccelError::ParityFailed(format!(
                "silu_mul n={n}: max_abs_err={max_err} > tol={tolerance}"
            )));
        }
        Ok(max_err)
    }

    /// Parity smoke for half-split RoPE. Synthesises a small
    /// `(n_heads, head_dim)` Q/K-like buffer + an inv-freq table
    /// matching llama's RoPE base and runs CPU + GPU in lockstep.
    pub fn parity_check_rope(
        &mut self,
        n_heads: u32,
        head_dim: u32,
        pos: u32,
        tolerance: f32,
    ) -> Result<f32> {
        if head_dim % 2 != 0 {
            return Err(SyclAccelError::InvalidShape(format!(
                "rope: head_dim must be even, got {head_dim}"
            )));
        }
        let total = (n_heads as usize) * (head_dim as usize);
        let half = (head_dim / 2) as usize;
        let qk_initial: Vec<f32> = (0..total)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.13)
            .collect();
        // RoPE base 10000 — standard llama convention. Inv-freq at
        // dim index `j` is `1 / 10000^(2j/head_dim)`.
        let inv_freq: Vec<f32> = (0..half)
            .map(|j| {
                let exponent = (2 * j) as f32 / head_dim as f32;
                1.0f32 / 10000f32.powf(exponent)
            })
            .collect();
        // GPU path: round-trip via shim.
        let mut qk_gpu = qk_initial.clone();
        self.rope_f32(&mut qk_gpu, n_heads, head_dim, pos, &inv_freq)?;
        // CPU reference: do the rotation in f32 but f16-round-trip
        // the IO to match the GPU's f16 storage.
        let mut qk_cpu: Vec<f32> = qk_initial
            .iter()
            .map(|v| f16::from_f32(*v).to_f32())
            .collect();
        let inv_freq_f16: Vec<f32> = inv_freq
            .iter()
            .map(|v| f16::from_f32(*v).to_f32())
            .collect();
        for h in 0..n_heads as usize {
            let base = h * head_dim as usize;
            for j in 0..half {
                let freq = inv_freq_f16[j];
                let angle = (pos as f32) * freq;
                let c = angle.cos();
                let s = angle.sin();
                let x0 = qk_cpu[base + j];
                let x1 = qk_cpu[base + j + half];
                qk_cpu[base + j] = f16::from_f32(x0 * c - x1 * s).to_f32();
                qk_cpu[base + j + half] = f16::from_f32(x0 * s + x1 * c).to_f32();
            }
        }
        let mut max_err = 0f32;
        for (g, c) in qk_gpu.iter().zip(qk_cpu.iter()) {
            let e = (g - c).abs();
            if e > max_err {
                max_err = e;
            }
        }
        if max_err > tolerance {
            return Err(SyclAccelError::ParityFailed(format!(
                "rope n_heads={n_heads} head_dim={head_dim} pos={pos}: \
                 max_abs_err={max_err} > tol={tolerance}"
            )));
        }
        Ok(max_err)
    }

    /// Parity smoke for fused softmax-attention. Builds a synthetic
    /// `[n_heads, seq, kv_len]` score tensor + causal mask and asserts
    /// the GPU output matches a CPU softmax with the same scale.
    pub fn parity_check_softmax_attn(
        &mut self,
        n_heads: u32,
        seq: u32,
        kv_len: u32,
        tolerance: f32,
    ) -> Result<f32> {
        let total = (n_heads as usize) * (seq as usize) * (kv_len as usize);
        let scores_initial: Vec<f32> = (0..total)
            .map(|i| ((i % 53) as f32 - 25.0) * 0.04)
            .collect();
        // Mask the last position with -inf so the softmax effectively
        // zeros it; everything else is 0 (no mask effect).
        let mut mask = vec![0f32; kv_len as usize];
        if kv_len > 0 {
            mask[kv_len as usize - 1] = f32::NEG_INFINITY;
        }
        let scale = 1.0f32 / (kv_len as f32).sqrt();
        let mut scores_gpu = scores_initial.clone();
        self.softmax_attn_f32(
            &mut scores_gpu,
            Some(&mask),
            n_heads,
            seq,
            kv_len,
            scale,
        )?;
        // CPU reference: per (head, query) row, scale + add mask + softmax.
        let mut scores_cpu = scores_initial.clone();
        let n_h = n_heads as usize;
        let n_s = seq as usize;
        let n_k = kv_len as usize;
        for h in 0..n_h {
            for q in 0..n_s {
                let base = h * n_s * n_k + q * n_k;
                let mut m = f32::NEG_INFINITY;
                for k in 0..n_k {
                    // f16 round-trip input to match GPU storage.
                    let v = f16::from_f32(scores_cpu[base + k]).to_f32() * scale + mask[k];
                    if v > m {
                        m = v;
                    }
                }
                let mut sum = 0f32;
                let mut exp_vals = vec![0f32; n_k];
                for k in 0..n_k {
                    let v = f16::from_f32(scores_cpu[base + k]).to_f32() * scale + mask[k];
                    let e = if v.is_infinite() && v < 0.0 {
                        0.0
                    } else {
                        (v - m).exp()
                    };
                    exp_vals[k] = e;
                    sum += e;
                }
                let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                for k in 0..n_k {
                    scores_cpu[base + k] = f16::from_f32(exp_vals[k] * inv).to_f32();
                }
            }
        }
        let mut max_err = 0f32;
        for (g, c) in scores_gpu.iter().zip(scores_cpu.iter()) {
            let e = (g - c).abs();
            if e > max_err {
                max_err = e;
            }
        }
        if max_err > tolerance {
            return Err(SyclAccelError::ParityFailed(format!(
                "softmax_attn n_heads={n_heads} seq={seq} kv_len={kv_len}: \
                 max_abs_err={max_err} > tol={tolerance}"
            )));
        }
        Ok(max_err)
    }

    /// Parity smoke for F16 GEMM. Small `[M, N] = [M, K] @ [K, N]`
    /// with deterministic inputs; compares to a CPU GEMM reference.
    pub fn parity_check_gemm(
        &mut self,
        m: u32,
        n: u32,
        k: u32,
        tolerance: f32,
    ) -> Result<f32> {
        let m_us = m as usize;
        let n_us = n as usize;
        let k_us = k as usize;
        let a: Vec<f32> = (0..m_us * k_us)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
            .collect();
        let b: Vec<f32> = (0..k_us * n_us)
            .map(|i| ((i % 19) as f32 - 9.0) * 0.04)
            .collect();
        let mut c_gpu = vec![0f32; m_us * n_us];
        self.gemm_f16_f32(&a, &b, &mut c_gpu, m, n, k)?;
        // CPU reference: naive triple-loop GEMM with f16-round-trip
        // inputs + final-write f16-round-trip to mirror the kernel.
        let mut c_cpu = vec![0f32; m_us * n_us];
        for r in 0..m_us {
            for col in 0..n_us {
                let mut acc = 0f32;
                for p in 0..k_us {
                    let av = f16::from_f32(a[r * k_us + p]).to_f32();
                    let bv = f16::from_f32(b[p * n_us + col]).to_f32();
                    acc += av * bv;
                }
                c_cpu[r * n_us + col] = f16::from_f32(acc).to_f32();
            }
        }
        let mut max_err = 0f32;
        for (g, c) in c_gpu.iter().zip(c_cpu.iter()) {
            let e = (g - c).abs();
            if e > max_err {
                max_err = e;
            }
        }
        if max_err > tolerance {
            return Err(SyclAccelError::ParityFailed(format!(
                "gemm M={m} N={n} K={k}: max_abs_err={max_err} > tol={tolerance}"
            )));
        }
        Ok(max_err)
    }
}

fn map_sycl_err(e: SyclError) -> SyclAccelError {
    match e {
        SyclError::Unavailable => {
            SyclAccelError::Unavailable("kernel call in mock mode".into())
        }
        SyclError::NoSuchDevice(i) => SyclAccelError::NoSuchDevice(i),
        SyclError::InvalidShape(s) => SyclAccelError::InvalidShape(s),
        // FFI-caught C++ exception. Engine treats this as a kernel
        // failure (same shape as InvalidShape from the caller's POV)
        // so the existing CPU-fallback path triggers.
        SyclError::Runtime(s) => SyclAccelError::InvalidShape(s),
        // L0 import is a separate path from the kernel-dispatch
        // accel layer; this variant shouldn't reach here in practice,
        // but the exhaustiveness check requires a covering arm.
        // Surface as InvalidShape so any accidental flow falls back
        // to CPU rather than crashing.
        SyclError::L0ImportUnsupported(code) => {
            SyclAccelError::InvalidShape(format!("L0 import unsupported (code {code})"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_new_does_not_panic() {
        // The accelerator MUST return a Result (Unavailable / NoSuchDevice
        // on a host with no Intel GPU, Ok otherwise) rather than panic —
        // that's the contract every fallback path depends on.
        let r = SyclAccelerator::try_new(0);
        let _ = r;
    }

    // The parity tests below need a real Intel GPU, so they are `#[ignore]`d
    // (workspace `cargo test` skips them). Run with
    // `cargo test -- --ignored parity` to exercise them against the host's
    // Intel GPU.

    #[test]
    #[ignore = "requires SYCL GPU; run with -- --ignored"]
    fn parity_rmsnorm_against_cpu_reference() {
        let mut a = match SyclAccelerator::try_new(0) {
            Ok(a) => a,
            Err(e) => panic!("no SYCL device available: {e}"),
        };
        // f16 quantization error compounds with the d=4096 reduction;
        // 1e-2 is a comfortable bound for round-trip parity (the
        // engine integration tests allow ~3e-3 between f32 and f16
        // attention paths). Adjust down if a kernel improvement
        // tightens it.
        let max = a.parity_check_rmsnorm(2, 4096, 1e-2).expect("parity");
        eprintln!("rmsnorm parity: max_abs_err = {max:.6}");
    }

    #[test]
    #[ignore = "requires SYCL GPU; run with -- --ignored"]
    fn parity_embedding_lookup_against_cpu_reference() {
        let mut a = match SyclAccelerator::try_new(0) {
            Ok(a) => a,
            Err(e) => panic!("no SYCL device available: {e}"),
        };
        // f16 round-trip parity should be exact within a single bit
        // since the kernel only does a memcpy — no compute. 1e-4 is
        // a generous bound.
        let max = a.parity_check_embedding(1024, 256, 1e-4).expect("parity");
        eprintln!("embedding parity: max_abs_err = {max:.6}");
    }
}
