//! GPU-side sampler pipeline.
//!
//! Stitches the five F1 SYCL kernels — penalty, fused
//! temperature+softmax, top-k mask, multinomial draw, argmax —
//! into a single per-token call that keeps `logits` resident in
//! USM across the whole chain. Only one device→host read happens
//! per sample: the chosen token id.
//!
//! Construct one [`SamplerGpuContext`] per thread (it owns a
//! `SyclStream` which is `!Send + !Sync`). Capacity for the logits
//! / recent-tokens buffers grows lazily on first use; the RNG
//! state + output index slots are fixed-size.
//!
//! **Bypass conditions**: callers must short-circuit to the CPU
//! sampler when:
//!   - a grammar mask is present (per-token vocab masking — not yet
//!     fused into the GPU pipeline),
//!   - mirostat-v1 or mirostat-v2 is enabled (needs per-step state
//!     feedback the GPU chain doesn't carry),
//!   - `top_p > 0.0 && top_p < 1.0` (top-p kernel pending),
//!   - `typical_p > 0.0 && typical_p < 1.0` (typical-p kernel pending),
//!   - `top_k > MAX_TOP_K_GPU` (256; GPU top-k is capped),
//!   - logprobs > 0 (callers need pre-softmax logits which the GPU
//!     chain consumes/mutates in-place).
//!
//! These conditions are documented + exposed via
//! [`SamplerGpuParams::is_supported`] so the call site has a single
//! place to decide CPU-vs-GPU.

use rustllama_kernels_sycl as sk;

/// Inputs to one GPU sample. The caller decides between greedy
/// (argmax) and stochastic (temperature + softmax + top-k +
/// multinomial) via [`Self::greedy`].
#[derive(Debug, Clone, Copy)]
pub struct SamplerGpuParams {
    pub repeat_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub temperature: f32,
    /// `0` means "no top-k filter". Values up to
    /// `sk::MAX_TOP_K_GPU` are GPU-eligible; larger falls to CPU.
    pub top_k: u32,
    /// True ⇒ argmax over logits (after penalties), no softmax /
    /// multinomial. The temperature field is ignored when greedy.
    pub greedy: bool,
    /// H1: top-p (nucleus) sampling. `0.0` or `≥ 1.0` ⇒ disabled.
    /// `0.0 < top_p < 1.0` ⇒ keep the smallest set of tokens whose
    /// cumulative probability ≥ `top_p`, mask the rest, renormalize.
    /// Applied after top-k (if any) and before multinomial.
    pub top_p: f32,
}

impl SamplerGpuParams {
    /// Whether this configuration can be handled entirely on GPU.
    /// Pair with the upstream bypass conditions (grammar / mirostat
    /// / typical-p / logprobs) for the full gate.
    pub fn is_supported(&self) -> bool {
        if self.top_k > sk::MAX_TOP_K_GPU {
            return false;
        }
        if !self.greedy && self.temperature <= 0.0 {
            // Caller should set `greedy = true` for T=0; refuse rather
            // than silently divide-by-zero on the GPU path.
            return false;
        }
        // H1: top_p between (0, 1) is GPU-supported via the existing
        // `rsl_sampler_top_p_usm` kernel. Outside that range the
        // kernel is a no-op anyway (the gate is permissive here).
        true
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SamplerGpuError {
    #[error("sampler GPU backend not available")]
    Unavailable,
    #[error("SYCL kernel failed: {0}")]
    KernelFailed(String),
    #[error("input/parameter shape invalid: {0}")]
    InvalidShape(String),
}

/// GPU-resident sampler context. Owns a stream + USM staging
/// buffers; threads logits + RNG state through all five kernels
/// in one call.
pub struct SamplerGpuContext {
    stream: sk::SyclStream,
    /// USM staging for the working logits buffer. Resized lazily
    /// on first call (and on vocab change between calls).
    logits_usm: *mut f32,
    logits_capacity: usize,
    /// USM staging for the recent-token window. Resized lazily.
    recent_usm: *mut u32,
    recent_capacity: usize,
    /// USM-resident SplitMix64 state — persists across sample
    /// calls so the GPU stream advances in lockstep with the
    /// caller's CPU `Rng::state`.
    rng_state_usm: *mut u64,
    /// USM slot for the chosen token id (i32) written by the
    /// argmax / multinomial kernel.
    out_idx_usm: *mut i32,
    /// H1: i32 USM slot where the top-p kernel writes `1` if it
    /// can't satisfy the cumsum threshold within `MAX_TOP_P_GPU`
    /// scanned candidates (i.e., the long tail is too flat for
    /// the GPU's bounded scan). On `1` we fall back to CPU top-p.
    top_p_fallback_usm: *mut i32,
}

impl SamplerGpuContext {
    /// Create a sampler context on the given SYCL device. Allocates
    /// the fixed-size RNG and output USM slots up-front; the
    /// resizable buffers grow on first call.
    pub fn new(device_index: u32) -> Result<Self, SamplerGpuError> {
        let stream = sk::create_stream(device_index).map_err(|e| {
            SamplerGpuError::KernelFailed(format!("create_stream({device_index}) failed: {e}"))
        })?;
        let rng_state_usm =
            sk::usm_alloc_shared(&stream, std::mem::size_of::<u64>()) as *mut u64;
        if rng_state_usm.is_null() {
            return Err(SamplerGpuError::Unavailable);
        }
        let out_idx_usm = sk::usm_alloc_shared(&stream, std::mem::size_of::<i32>()) as *mut i32;
        if out_idx_usm.is_null() {
            // SAFETY: rng_state_usm came from `usm_alloc_shared` on the
            // same stream and is currently owned by us.
            unsafe { sk::usm_free(&stream, rng_state_usm as *mut std::ffi::c_void) };
            return Err(SamplerGpuError::Unavailable);
        }
        // H1: top-p fallback signal slot.
        let top_p_fallback_usm = sk::usm_alloc_shared(&stream, std::mem::size_of::<i32>()) as *mut i32;
        if top_p_fallback_usm.is_null() {
            unsafe {
                sk::usm_free(&stream, rng_state_usm as *mut std::ffi::c_void);
                sk::usm_free(&stream, out_idx_usm as *mut std::ffi::c_void);
            }
            return Err(SamplerGpuError::Unavailable);
        }
        // Seed all slots so a kernel reading them on the first call
        // doesn't see UB-tainted garbage.
        unsafe {
            *rng_state_usm = 0;
            *out_idx_usm = 0;
            *top_p_fallback_usm = 0;
        }
        Ok(Self {
            stream,
            logits_usm: std::ptr::null_mut(),
            logits_capacity: 0,
            recent_usm: std::ptr::null_mut(),
            recent_capacity: 0,
            rng_state_usm,
            out_idx_usm,
            top_p_fallback_usm,
        })
    }

    fn ensure_logits_capacity(&mut self, vocab: usize) -> Result<(), SamplerGpuError> {
        if self.logits_capacity >= vocab {
            return Ok(());
        }
        if !self.logits_usm.is_null() {
            unsafe { sk::usm_free(&self.stream, self.logits_usm as *mut std::ffi::c_void) };
            self.logits_usm = std::ptr::null_mut();
            self.logits_capacity = 0;
        }
        let bytes = vocab * std::mem::size_of::<f32>();
        let p = sk::usm_alloc_shared(&self.stream, bytes) as *mut f32;
        if p.is_null() {
            return Err(SamplerGpuError::Unavailable);
        }
        self.logits_usm = p;
        self.logits_capacity = vocab;
        Ok(())
    }

    fn ensure_recent_capacity(&mut self, n: usize) -> Result<(), SamplerGpuError> {
        if self.recent_capacity >= n {
            return Ok(());
        }
        if !self.recent_usm.is_null() {
            unsafe { sk::usm_free(&self.stream, self.recent_usm as *mut std::ffi::c_void) };
            self.recent_usm = std::ptr::null_mut();
            self.recent_capacity = 0;
        }
        let bytes = n * std::mem::size_of::<u32>();
        let p = sk::usm_alloc_shared(&self.stream, bytes) as *mut u32;
        if p.is_null() {
            return Err(SamplerGpuError::Unavailable);
        }
        self.recent_usm = p;
        self.recent_capacity = n;
        Ok(())
    }

    /// Run the full sampler chain on GPU. `rng_state` carries the
    /// caller's CPU `Rng::state()`; the returned `new_rng_state` is
    /// the post-draw value (no-op when `params.greedy == true`,
    /// since argmax doesn't consume the RNG stream).
    ///
    /// `logits` is read once at the start of the call (copied into
    /// USM); the GPU mutates the USM copy but leaves the caller's
    /// slice alone. This matches the CPU sampler's "logits owned
    /// by the engine, sampler doesn't keep a reference" contract.
    pub fn sample(
        &mut self,
        logits: &[f32],
        recent: &[u32],
        rng_state: u64,
        params: &SamplerGpuParams,
    ) -> Result<(u32, u64), SamplerGpuError> {
        if !params.is_supported() {
            return Err(SamplerGpuError::InvalidShape(
                "params.is_supported() == false; caller should fall back to CPU".to_string(),
            ));
        }
        let vocab = logits.len();
        if vocab == 0 {
            return Err(SamplerGpuError::InvalidShape("empty logits".to_string()));
        }
        self.ensure_logits_capacity(vocab)?;
        if !recent.is_empty() {
            self.ensure_recent_capacity(recent.len())?;
        }
        // Stage inputs to USM.
        unsafe {
            std::ptr::copy_nonoverlapping(logits.as_ptr(), self.logits_usm, vocab);
            if !recent.is_empty() {
                std::ptr::copy_nonoverlapping(recent.as_ptr(), self.recent_usm, recent.len());
            }
            *self.rng_state_usm = rng_state;
            *self.out_idx_usm = 0;
        }

        // Pass 1 (optional): penalty pass.
        if !recent.is_empty()
            && (params.repeat_penalty != 1.0
                || params.frequency_penalty != 0.0
                || params.presence_penalty != 0.0)
        {
            // SAFETY: all pointers came from USM allocs on `self.stream`.
            unsafe {
                sk::sampler_penalty_usm_raw(
                    &self.stream,
                    self.logits_usm,
                    vocab as u32,
                    self.recent_usm,
                    recent.len() as u32,
                    params.repeat_penalty,
                    params.frequency_penalty,
                    params.presence_penalty,
                )
                .map_err(|e| SamplerGpuError::KernelFailed(format!("penalty: {e}")))?;
            }
        }

        // Greedy short-circuit: argmax over (penalized) logits.
        if params.greedy {
            // SAFETY: logits_usm and out_idx_usm are USM-shared on
            // self.stream; sizes match the kernel's expectations.
            unsafe {
                sk::sampler_argmax_usm_raw(
                    &self.stream,
                    self.logits_usm,
                    vocab as u32,
                    self.out_idx_usm,
                )
                .map_err(|e| SamplerGpuError::KernelFailed(format!("argmax: {e}")))?;
            }
            let idx = unsafe { *self.out_idx_usm };
            if idx < 0 {
                return Err(SamplerGpuError::KernelFailed(format!(
                    "argmax returned {idx}"
                )));
            }
            // RNG state unchanged on greedy.
            return Ok((idx as u32, rng_state));
        }

        // Stochastic path: temp+softmax → optional top-k → multinomial.
        let inv_t = 1.0 / params.temperature;
        unsafe {
            sk::sampler_temp_softmax_usm_raw(
                &self.stream,
                self.logits_usm,
                vocab as u32,
                inv_t,
            )
            .map_err(|e| SamplerGpuError::KernelFailed(format!("temp_softmax: {e}")))?;
        }
        if params.top_k > 0 && (params.top_k as usize) < vocab {
            // top_k <= MAX_TOP_K_GPU is locked in by is_supported().
            unsafe {
                sk::sampler_top_k_usm_raw(
                    &self.stream,
                    self.logits_usm,
                    vocab as u32,
                    params.top_k,
                )
                .map_err(|e| SamplerGpuError::KernelFailed(format!("top_k: {e}")))?;
            }
        }
        // H1: top-p (nucleus) sampling. Skip when disabled (`p ≤ 0`
        // or `p ≥ 1`). On `needs_fallback == 1` the kernel hit its
        // MAX_TOP_P_GPU candidate cap before crossing `p`; signal
        // the caller to retry on CPU.
        if params.top_p > 0.0 && params.top_p < 1.0 {
            unsafe {
                *self.top_p_fallback_usm = 0;
                sk::sampler_top_p_usm_raw(
                    &self.stream,
                    self.logits_usm,
                    vocab as u32,
                    params.top_p,
                    self.top_p_fallback_usm,
                )
                .map_err(|e| SamplerGpuError::KernelFailed(format!("top_p: {e}")))?;
                if *self.top_p_fallback_usm != 0 {
                    return Err(SamplerGpuError::InvalidShape(
                        "top_p exceeded GPU kernel's scan budget — caller should retry on CPU".to_string(),
                    ));
                }
            }
        }
        unsafe {
            sk::sampler_multinomial_usm_raw(
                &self.stream,
                self.logits_usm,
                vocab as u32,
                self.rng_state_usm,
                self.out_idx_usm,
            )
            .map_err(|e| SamplerGpuError::KernelFailed(format!("multinomial: {e}")))?;
        }
        let idx = unsafe { *self.out_idx_usm };
        let new_rng_state = unsafe { *self.rng_state_usm };
        if idx < 0 {
            return Err(SamplerGpuError::KernelFailed(format!(
                "multinomial returned {idx}"
            )));
        }
        Ok((idx as u32, new_rng_state))
    }
}

impl Drop for SamplerGpuContext {
    fn drop(&mut self) {
        unsafe {
            if !self.logits_usm.is_null() {
                sk::usm_free(&self.stream, self.logits_usm as *mut std::ffi::c_void);
            }
            if !self.recent_usm.is_null() {
                sk::usm_free(&self.stream, self.recent_usm as *mut std::ffi::c_void);
            }
            if !self.rng_state_usm.is_null() {
                sk::usm_free(&self.stream, self.rng_state_usm as *mut std::ffi::c_void);
            }
            if !self.out_idx_usm.is_null() {
                sk::usm_free(&self.stream, self.out_idx_usm as *mut std::ffi::c_void);
            }
            if !self.top_p_fallback_usm.is_null() {
                sk::usm_free(&self.stream, self.top_p_fallback_usm as *mut std::ffi::c_void);
            }
        }
        // self.stream Drop runs after this returns.
    }
}
