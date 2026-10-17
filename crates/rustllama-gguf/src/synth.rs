//! Synthetic GGUF builder for tests.
//!
//! Gated behind feature `synth`. Used by integration tests in
//! `rustllama-models` and `rustllama-engine` to produce deterministic tiny
//! models without requiring large downloaded fixtures.

use std::io::Write;
use std::path::Path;

use half::f16;

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_VERSION: u32 = 3;
const GGML_TYPE_F32: u32 = 0;
const GGML_TYPE_F16: u32 = 1;
const GGML_TYPE_BF16: u32 = 30;

#[repr(u32)]
#[allow(dead_code)]
enum MetaType {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    I32 = 5,
    F32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    U64 = 10,
    I64 = 11,
    F64 = 12,
}

struct Writer {
    buf: Vec<u8>,
}
impl Writer {
    fn new() -> Self {
        Self { buf: Vec::new() }
    }
    fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn str_(&mut self, s: &str) {
        self.u64(s.len() as u64);
        self.buf.extend_from_slice(s.as_bytes());
    }
}

struct MetaBuilder {
    w: Writer,
    n: usize,
}
impl MetaBuilder {
    fn new() -> Self {
        Self {
            w: Writer::new(),
            n: 0,
        }
    }
    fn u32(&mut self, k: &str, v: u32) {
        self.w.str_(k);
        self.w.u32(MetaType::U32 as u32);
        self.w.u32(v);
        self.n += 1;
    }
    fn f32(&mut self, k: &str, v: f32) {
        self.w.str_(k);
        self.w.u32(MetaType::F32 as u32);
        self.w.f32(v);
        self.n += 1;
    }
    fn bool(&mut self, k: &str, v: bool) {
        self.w.str_(k);
        self.w.u32(MetaType::Bool as u32);
        self.w.buf.push(if v { 1 } else { 0 });
        self.n += 1;
    }
    fn array_i32(&mut self, k: &str, v: &[i32]) {
        self.w.str_(k);
        self.w.u32(MetaType::Array as u32);
        self.w.u32(MetaType::I32 as u32);
        self.w.u64(v.len() as u64);
        for x in v {
            self.w.buf.extend_from_slice(&x.to_le_bytes());
        }
        self.n += 1;
    }
    fn string(&mut self, k: &str, v: &str) {
        self.w.str_(k);
        self.w.u32(MetaType::String as u32);
        self.w.str_(v);
        self.n += 1;
    }
    fn array_strings(&mut self, k: &str, v: &[String]) {
        self.w.str_(k);
        self.w.u32(MetaType::Array as u32);
        self.w.u32(MetaType::String as u32);
        self.w.u64(v.len() as u64);
        for s in v {
            self.w.str_(s);
        }
        self.n += 1;
    }
    fn array_bool(&mut self, k: &str, v: &[bool]) {
        self.w.str_(k);
        self.w.u32(MetaType::Array as u32);
        self.w.u32(MetaType::Bool as u32);
        self.w.u64(v.len() as u64);
        for x in v {
            self.w.buf.push(if *x { 1 } else { 0 });
        }
        self.n += 1;
    }
    fn array_f32(&mut self, k: &str, v: &[f32]) {
        self.w.str_(k);
        self.w.u32(MetaType::Array as u32);
        self.w.u32(MetaType::F32 as u32);
        self.w.u64(v.len() as u64);
        for s in v {
            self.w.f32(*s);
        }
        self.n += 1;
    }
}

struct TensorSpec {
    name: String,
    dims: Vec<u64>,
    dtype: u32,
    bytes: Vec<u8>,
}

impl TensorSpec {
    fn f32(name: &str, dims: Vec<u64>, data: Vec<f32>) -> Self {
        let expected: u64 = dims.iter().copied().product();
        assert_eq!(
            data.len() as u64,
            expected,
            "shape vs data length for {name}"
        );
        let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
        Self {
            name: name.into(),
            dims,
            dtype: GGML_TYPE_F32,
            bytes,
        }
    }
    fn f16(name: &str, dims: Vec<u64>, data: Vec<f16>) -> Self {
        let expected: u64 = dims.iter().copied().product();
        assert_eq!(
            data.len() as u64,
            expected,
            "shape vs data length for {name}"
        );
        let mut bytes = Vec::with_capacity(data.len() * 2);
        for h in &data {
            bytes.extend_from_slice(&h.to_le_bytes());
        }
        Self {
            name: name.into(),
            dims,
            dtype: GGML_TYPE_F16,
            bytes,
        }
    }
    fn bf16(name: &str, dims: Vec<u64>, data: Vec<f32>) -> Self {
        let expected: u64 = dims.iter().copied().product();
        assert_eq!(
            data.len() as u64,
            expected,
            "shape vs data length for {name}"
        );
        let mut bytes = Vec::with_capacity(data.len() * 2);
        for v in &data {
            // BF16 = top 16 bits of f32 (round-toward-zero — no rounding
            // needed for test fixture).
            let u = (v.to_bits() >> 16) as u16;
            bytes.extend_from_slice(&u.to_le_bytes());
        }
        Self {
            name: name.into(),
            dims,
            dtype: GGML_TYPE_BF16,
            bytes,
        }
    }
}

/// Numeric dtype to use for the bulk weight tensors in the synthetic
/// GGUF. Norm tensors stay F32 either way (matches real-world GGUFs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SynthWeightDtype {
    F16,
    Bf16,
}

/// Tokenizer family to embed in the synthetic GGUF.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SynthTokenizer {
    /// GPT2-style byte-level BPE (covers Qwen / Llama-3 / DeepSeek).
    Gpt2,
    /// SentencePiece / Unigram (covers Llama-1/2, Mistral, TinyLlama, Phi-3).
    Llama,
}

/// Hyperparameters for the synthetic Llama-family model. Sensible defaults
/// produce a 2-layer / 64-d_model / 32-vocab toy useful for round-trip and
/// CPU forward-pass tests in <1 second.
#[derive(Debug, Clone)]
pub struct SynthLlama {
    pub n_layers: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub d_model: u32,
    pub d_ff: u32,
    pub vocab: u32,
    pub ctx: u32,
    pub seed: u64,
    pub tokenizer: SynthTokenizer,
    /// If `true`, replace the last three tokens of the synthetic vocab
    /// with Qwen2.5-Coder-style FIM marker strings
    /// (`<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>`). Lets
    /// tests exercise [`rustllama_tokenizer::Tokenizer::fim_tokens`]
    /// + the `/v1/completions` FIM path without a real coding model.
    pub include_fim_tokens: bool,
    /// Dtype for the bulk weight tensors (embeddings, attention,
    /// FFN). Defaults to `F16`, which is what real-world coding-model
    /// GGUFs ship today. Set to `Bf16` to exercise the BF16 → F16
    /// load conversion path in [`rustllama_tensor::load_tensor_bytes`].
    pub weight_dtype: SynthWeightDtype,
    /// Optional MoE metadata. When `Some`, the writer emits the
    /// `{arch}.expert_count` + `expert_used_count` + (optional)
    /// `expert_shared_count` keys so the loader's MoE detection
    /// path fires. Tensor names are NOT swapped — the test fixture
    /// is just enough for the config-detection contract; full
    /// MoE-shaped tensor binding lands in the phase-2 MoE turn.
    pub moe: Option<SynthMoe>,
    /// Optional hybrid attention+SSM metadata. When `Some`, the
    /// writer emits the `{arch}.full_attention_interval` +
    /// `{arch}.ssm.*` keys so the loader's hybrid-detection path
    /// fires (`HybridConfig` populated). Tensor names are NOT
    /// swapped — this is the config-detection contract only;
    /// full hybrid tensor binding against synth fixtures requires
    /// emitting 11 SSM tensors per layer + a NextN head, which is
    /// out of scope for Phase 2's synth coverage.
    pub hybrid: Option<SynthHybrid>,
}

#[derive(Debug, Clone, Copy)]
pub struct SynthMoe {
    pub n_experts: u32,
    pub n_experts_used: u32,
    pub n_experts_shared: u32,
}

/// Hyperparameters for a synthetic DeepSeek-V2/V3-style MLA model
/// (`deepseek2` arch). Produces the full Multi-head Latent Attention
/// tensor set — low-rank Q (optional), compressed-latent KV, decoupled
/// RoPE key, up-projection — plus a per-layer FFN (dense for the first
/// `n_dense_layers`, MoE after, mirroring DeepSeek's `first_k_dense_replace`).
/// Small enough to run the REAL `forward_one_mla` in a unit test.
#[derive(Debug, Clone, Copy)]
pub struct SynthMla {
    pub n_layers: u32,
    pub n_heads: u32,
    pub d_model: u32,
    /// Dense / shared-expert FFN width.
    pub d_ff: u32,
    pub vocab: u32,
    pub ctx: u32,
    pub seed: u64,
    /// `Some(r)` emits the low-rank Q path (`attn_q_a`→norm→`attn_q_b`);
    /// `None` emits a single direct `attn_q` (DeepSeek-V2-Lite shape).
    pub q_lora_rank: Option<u32>,
    pub kv_lora_rank: u32,
    pub qk_nope_head_dim: u32,
    /// Rotary width — must be even (it is RoPE-rotated).
    pub qk_rope_head_dim: u32,
    pub v_head_dim: u32,
    /// `Some` ⇒ MoE FFN on the layers at/after `n_dense_layers`; `None`
    /// ⇒ every layer is a dense SwiGLU FFN.
    pub moe: Option<SynthMoe>,
    /// Number of leading layers that use a DENSE FFN even when `moe` is
    /// `Some` (DeepSeek `first_k_dense_replace`). Ignored when `moe` is
    /// `None` (all dense).
    pub n_dense_layers: u32,
}

impl Default for SynthMla {
    fn default() -> Self {
        Self {
            n_layers: 2,
            n_heads: 2,
            d_model: 32,
            d_ff: 64,
            vocab: 32,
            ctx: 64,
            seed: 0xD339_5EE2,
            q_lora_rank: Some(16),
            kv_lora_rank: 16,
            qk_nope_head_dim: 8,
            qk_rope_head_dim: 8,
            v_head_dim: 8,
            moe: None,
            n_dense_layers: 0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SynthHybrid {
    pub full_attention_interval: u32,
    pub ssm_state_size: u32,
    pub ssm_conv_kernel: u32,
    pub ssm_group_count: u32,
    pub ssm_time_step_rank: u32,
    pub ssm_inner_size: u32,
    pub shared_expert_feed_forward_length: Option<u32>,
    pub nextn_predict_layers: u32,
}

impl SynthHybrid {
    /// F9 — a PURE-SSM hybrid (zero full-attention layers). Setting
    /// `full_attention_interval` one past the last layer makes `(i+1) %
    /// interval` never 0, so no layer is stamped full-attention and no layer
    /// emits `attn_q.weight`; the writer emits the SSM/DeltaNet tensor set on
    /// EVERY layer. The resulting GGUF loads + runs through
    /// [`crate`]'s hybrid forward (`forward_one_hybrid`) with an all-
    /// placeholder KV cache — the recurrent DeltaNet state is the only cache,
    /// i.e. a pure Mamba/SSM model expressed as the hybrid forward with zero
    /// attention. Pair with a `SynthLlama` whose `head_dim` divides both
    /// `ssm_inner_size` and `d_model` (e.g. the F8/F9 fixture shape d_model=32,
    /// head_dim=8, ssm_inner_size=32).
    pub fn pure_ssm(n_layers: u32, ssm_inner_size: u32) -> Self {
        Self {
            full_attention_interval: n_layers + 1,
            ssm_state_size: 8,
            ssm_conv_kernel: 4,
            ssm_group_count: 1,
            ssm_time_step_rank: 2,
            ssm_inner_size,
            shared_expert_feed_forward_length: None,
            nextn_predict_layers: 0,
        }
    }
}

impl Default for SynthLlama {
    fn default() -> Self {
        Self {
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 32,
            d_model: 64,
            d_ff: 128,
            vocab: 32,
            ctx: 32,
            seed: 0xC0FFEE,
            tokenizer: SynthTokenizer::Gpt2,
            include_fim_tokens: false,
            moe: None,
            hybrid: None,
            weight_dtype: SynthWeightDtype::F16,
        }
    }
}

/// Write a synthetic Llama GGUF to `path` using deterministic RNG-seeded
/// F16 weights. The model is small enough (~50 KB) to live in a test.
pub fn write_synthetic_llama_gguf(path: &Path, params: &SynthLlama) {
    let moe = params.moe;
    let hybrid = params.hybrid;
    let SynthLlama {
        n_layers,
        n_heads,
        n_kv_heads,
        head_dim,
        d_model,
        d_ff,
        vocab,
        ctx,
        seed,
        tokenizer: _,
        include_fim_tokens: _,
        weight_dtype,
        moe: _,
        hybrid: _,
    } = *params;

    // Build a weight tensor at the requested dtype (F16 vs BF16). The
    // f32 source is the same in both cases; only the encoding differs.
    let weight_tensor = |name: &str, dims: Vec<u64>, n: usize, tseed: u64| -> TensorSpec {
        let f32s = random_f32(n, tseed);
        match weight_dtype {
            SynthWeightDtype::F16 => {
                let f16s = f32s.into_iter().map(f16::from_f32).collect();
                TensorSpec::f16(name, dims, f16s)
            }
            SynthWeightDtype::Bf16 => TensorSpec::bf16(name, dims, f32s),
        }
    };

    // metadata
    let mut m = MetaBuilder::new();
    m.string("general.architecture", "llama");
    m.u32("llama.block_count", n_layers);
    m.u32("llama.embedding_length", d_model);
    m.u32("llama.feed_forward_length", d_ff);
    m.u32("llama.attention.head_count", n_heads);
    m.u32("llama.attention.head_count_kv", n_kv_heads);
    m.f32("llama.attention.layer_norm_rms_epsilon", 1e-5);
    m.u32("llama.attention.key_length", head_dim);
    m.u32("llama.rope.dimension_count", head_dim);
    m.f32("llama.rope.freq_base", 10000.0);
    m.u32("llama.context_length", ctx);
    // MoE keys. Only emitted when the caller opts in via SynthLlama::moe.
    // Real Qwen3-MoE / Mixtral / DeepSeek-V3 GGUFs always carry these.
    if let Some(m_cfg) = moe {
        m.u32("llama.expert_count", m_cfg.n_experts);
        m.u32("llama.expert_used_count", m_cfg.n_experts_used);
        if m_cfg.n_experts_shared > 0 {
            m.u32("llama.expert_shared_count", m_cfg.n_experts_shared);
        }
    }
    // Hybrid attention+SSM keys. Only emitted when the caller opts
    // in via SynthLlama::hybrid. Real `qwen35moe` GGUFs always
    // carry the `full_attention_interval` + `ssm.*` family.
    if let Some(h) = hybrid {
        m.u32("llama.full_attention_interval", h.full_attention_interval);
        m.u32("llama.ssm.state_size", h.ssm_state_size);
        m.u32("llama.ssm.conv_kernel", h.ssm_conv_kernel);
        m.u32("llama.ssm.group_count", h.ssm_group_count);
        m.u32("llama.ssm.time_step_rank", h.ssm_time_step_rank);
        m.u32("llama.ssm.inner_size", h.ssm_inner_size);
        if let Some(sh) = h.shared_expert_feed_forward_length {
            m.u32("llama.expert_shared_feed_forward_length", sh);
        }
        if h.nextn_predict_layers > 0 {
            m.u32("llama.nextn_predict_layers", h.nextn_predict_layers);
        }
    }
    let mut vocab_tokens: Vec<String> = (0..vocab).map(|i| format!("<tok_{i}>")).collect();
    if params.include_fim_tokens {
        let n = vocab as usize;
        assert!(n >= 3, "synth vocab must have room for 3 FIM markers");
        vocab_tokens[n - 3] = "<|fim_prefix|>".to_string();
        vocab_tokens[n - 2] = "<|fim_suffix|>".to_string();
        vocab_tokens[n - 1] = "<|fim_middle|>".to_string();
    }
    m.array_strings("tokenizer.ggml.tokens", &vocab_tokens);
    match params.tokenizer {
        SynthTokenizer::Gpt2 => {
            m.string("tokenizer.ggml.model", "gpt2");
            m.array_strings("tokenizer.ggml.merges", &[]);
        }
        SynthTokenizer::Llama => {
            m.string("tokenizer.ggml.model", "llama");
            // Deterministic synthetic scores: linearly decreasing log-probs
            // so the Unigram model has a valid distribution.
            let scores: Vec<f32> = (0..vocab).map(|i| -1.0 - i as f32 * 0.05).collect();
            m.array_f32("tokenizer.ggml.scores", &scores);
        }
    }
    m.string(
        "tokenizer.ggml.chat_template",
        "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n'}}{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}",
    );
    m.u32("tokenizer.ggml.bos_token_id", 1);
    m.u32("tokenizer.ggml.eos_token_id", 2);

    // tensors
    let mut tensors = Vec::new();

    tensors.push(weight_tensor(
        "token_embd.weight",
        vec![d_model as u64, vocab as u64],
        (vocab * d_model) as usize,
        seed.wrapping_add(1),
    ));

    // Hybrid attention+SSM models (qwen35moe / Bonsai-2 shape) need an
    // ENTIRELY different per-layer tensor set than dense/MoE llama, so the
    // standard loop below is skipped when `hybrid` is set and the hybrid
    // emission block (just before `output_norm`) runs instead.
    for i in (0..n_layers).filter(|_| hybrid.is_none()) {
        let prefix = format!("blk.{i}");
        tensors.push(TensorSpec::f32(
            &format!("{prefix}.attn_norm.weight"),
            vec![d_model as u64],
            random_f32(d_model as usize, seed.wrapping_add(100 + i as u64)),
        ));
        tensors.push(weight_tensor(
            &format!("{prefix}.attn_q.weight"),
            vec![d_model as u64, (n_heads * head_dim) as u64],
            (d_model * n_heads * head_dim) as usize,
            seed.wrapping_add(200 + i as u64),
        ));
        tensors.push(weight_tensor(
            &format!("{prefix}.attn_k.weight"),
            vec![d_model as u64, (n_kv_heads * head_dim) as u64],
            (d_model * n_kv_heads * head_dim) as usize,
            seed.wrapping_add(300 + i as u64),
        ));
        tensors.push(weight_tensor(
            &format!("{prefix}.attn_v.weight"),
            vec![d_model as u64, (n_kv_heads * head_dim) as u64],
            (d_model * n_kv_heads * head_dim) as usize,
            seed.wrapping_add(400 + i as u64),
        ));
        tensors.push(weight_tensor(
            &format!("{prefix}.attn_output.weight"),
            vec![(n_heads * head_dim) as u64, d_model as u64],
            (n_heads * head_dim * d_model) as usize,
            seed.wrapping_add(500 + i as u64),
        ));
        tensors.push(TensorSpec::f32(
            &format!("{prefix}.ffn_norm.weight"),
            vec![d_model as u64],
            random_f32(d_model as usize, seed.wrapping_add(600 + i as u64)),
        ));
        // MoE: emit router + per-expert tensors instead of dense FFN.
        // Tensor shapes mirror llama.cpp's convention:
        //   ffn_gate_inp:  [n_experts, d_model]   (router)
        //   ffn_*_exps:    [n_experts, d_ff, d_model] (gate/up)
        //   ffn_down_exps: [n_experts, d_model, d_ff]
        // GGUF stores dims column-major, so the rightmost dim varies
        // fastest in memory — `[n_experts, d_ff, d_model]` means
        // n_experts contiguous (d_ff × d_model) blocks.
        if let Some(m_cfg) = moe {
            let n_e = m_cfg.n_experts as u64;
            tensors.push(weight_tensor(
                &format!("{prefix}.ffn_gate_inp.weight"),
                vec![d_model as u64, n_e],
                (d_model * m_cfg.n_experts) as usize,
                seed.wrapping_add(700 + i as u64),
            ));
            tensors.push(weight_tensor(
                &format!("{prefix}.ffn_gate_exps.weight"),
                vec![d_model as u64, d_ff as u64, n_e],
                (d_model * d_ff * m_cfg.n_experts) as usize,
                seed.wrapping_add(710 + i as u64),
            ));
            tensors.push(weight_tensor(
                &format!("{prefix}.ffn_up_exps.weight"),
                vec![d_model as u64, d_ff as u64, n_e],
                (d_model * d_ff * m_cfg.n_experts) as usize,
                seed.wrapping_add(720 + i as u64),
            ));
            tensors.push(weight_tensor(
                &format!("{prefix}.ffn_down_exps.weight"),
                vec![d_ff as u64, d_model as u64, n_e],
                (d_ff * d_model * m_cfg.n_experts) as usize,
                seed.wrapping_add(730 + i as u64),
            ));
            // Shared-expert tensors (DeepSeek-V3 style). Only
            // emitted when the caller opts in via
            // `n_experts_shared > 0`. Shape mirrors dense FFN.
            if m_cfg.n_experts_shared > 0 {
                tensors.push(weight_tensor(
                    &format!("{prefix}.ffn_gate_shexp.weight"),
                    vec![d_model as u64, d_ff as u64],
                    (d_model * d_ff) as usize,
                    seed.wrapping_add(740 + i as u64),
                ));
                tensors.push(weight_tensor(
                    &format!("{prefix}.ffn_up_shexp.weight"),
                    vec![d_model as u64, d_ff as u64],
                    (d_model * d_ff) as usize,
                    seed.wrapping_add(750 + i as u64),
                ));
                tensors.push(weight_tensor(
                    &format!("{prefix}.ffn_down_shexp.weight"),
                    vec![d_ff as u64, d_model as u64],
                    (d_ff * d_model) as usize,
                    seed.wrapping_add(760 + i as u64),
                ));
            }
        } else {
            tensors.push(weight_tensor(
                &format!("{prefix}.ffn_gate.weight"),
                vec![d_model as u64, d_ff as u64],
                (d_model * d_ff) as usize,
                seed.wrapping_add(700 + i as u64),
            ));
            tensors.push(weight_tensor(
                &format!("{prefix}.ffn_up.weight"),
                vec![d_model as u64, d_ff as u64],
                (d_model * d_ff) as usize,
                seed.wrapping_add(800 + i as u64),
            ));
            tensors.push(weight_tensor(
                &format!("{prefix}.ffn_down.weight"),
                vec![d_ff as u64, d_model as u64],
                (d_ff * d_model) as usize,
                seed.wrapping_add(900 + i as u64),
            ));
        }
    }

    // Hybrid attention+SSM tensor set, emitted when `hybrid` is set (the
    // dense/MoE per-layer loop above was skipped). SSM layers carry the 11
    // DeltaNet tensors; full-attention layers carry a GATED q-proj (`2*d_q`
    // output, chunked into query+gate in the forward); BOTH carry
    // `post_attention_norm.weight` (not `ffn_norm.weight`) + a dense SwiGLU FFN
    // (no MoE — a dense hybrid, the Bonsai-2 shape). Geometry is derived so the
    // hybrid forward's shape asserts (`DeltaNetCache::new_for_hybrid` +
    // `forward_one_hybrid_to_hidden`) all hold:
    //   head_v_dim = head_dim, n_v_heads = ssm_inner_size / head_dim,
    //   head_qk_dim = head_v_dim,
    //   n_qk_heads  = group_count > 1 ? group_count : d_model / head_qk_dim,
    //   qkv_dim     = 2*n_qk_heads*head_qk_dim + ssm_inner_size.
    // Weight values are arbitrary RNG: the hybrid parity tests compare the
    // PAGED engine against the CONTIGUOUS engine on the SAME weights, so only
    // internal consistency (finite, deterministic output) is required, not a
    // match to any external (HF) reference.
    if let Some(h) = hybrid {
        let ssm_inner = h.ssm_inner_size;
        let conv_kernel = h.ssm_conv_kernel.max(1);
        let head_v_dim = head_dim;
        assert!(
            head_v_dim > 0 && ssm_inner % head_v_dim == 0,
            "synth hybrid: ssm_inner_size ({ssm_inner}) must be divisible by head_dim ({head_v_dim})"
        );
        let n_v_heads = ssm_inner / head_v_dim;
        let head_qk_dim = head_v_dim;
        let n_qk_heads = if h.ssm_group_count > 1 {
            h.ssm_group_count
        } else {
            assert!(
                d_model % head_qk_dim == 0,
                "synth hybrid: d_model ({d_model}) must be divisible by head_dim ({head_qk_dim})"
            );
            d_model / head_qk_dim
        };
        let qkv_dim = 2 * n_qk_heads * head_qk_dim + ssm_inner;
        let d_q = n_heads * head_dim;
        let d_kv = n_kv_heads * head_dim;
        let interval = h.full_attention_interval.max(1);
        for i in 0..n_layers {
            let prefix = format!("blk.{i}");
            let sd = seed.wrapping_add(10_000 + (i as u64) * 50);
            // Pre-attention + post-attention (pre-FFN) norms — both kinds.
            tensors.push(TensorSpec::f32(
                &format!("{prefix}.attn_norm.weight"),
                vec![d_model as u64],
                random_f32(d_model as usize, sd),
            ));
            tensors.push(TensorSpec::f32(
                &format!("{prefix}.post_attention_norm.weight"),
                vec![d_model as u64],
                random_f32(d_model as usize, sd.wrapping_add(1)),
            ));
            // Dense SwiGLU FFN (no MoE → `HybridFfn::Dense`).
            tensors.push(weight_tensor(
                &format!("{prefix}.ffn_gate.weight"),
                vec![d_model as u64, d_ff as u64],
                (d_model * d_ff) as usize,
                sd.wrapping_add(2),
            ));
            tensors.push(weight_tensor(
                &format!("{prefix}.ffn_up.weight"),
                vec![d_model as u64, d_ff as u64],
                (d_model * d_ff) as usize,
                sd.wrapping_add(3),
            ));
            tensors.push(weight_tensor(
                &format!("{prefix}.ffn_down.weight"),
                vec![d_ff as u64, d_model as u64],
                (d_ff * d_model) as usize,
                sd.wrapping_add(4),
            ));
            // Layer kind: the loader keys on `attn_q.weight` PRESENCE (SSM
            // layers never carry it), with `(i+1) % interval == 0` as the
            // modulo twin. Emit full-attn tensors on modulo layers, SSM
            // tensors elsewhere, so presence and modulo agree.
            let is_full_attn = (i + 1) % interval == 0;
            if is_full_attn {
                // GATED q-proj: output width is `2*d_q` (query || per-head gate).
                tensors.push(weight_tensor(
                    &format!("{prefix}.attn_q.weight"),
                    vec![d_model as u64, (2 * d_q) as u64],
                    (d_model * 2 * d_q) as usize,
                    sd.wrapping_add(10),
                ));
                tensors.push(weight_tensor(
                    &format!("{prefix}.attn_k.weight"),
                    vec![d_model as u64, d_kv as u64],
                    (d_model * d_kv) as usize,
                    sd.wrapping_add(11),
                ));
                tensors.push(weight_tensor(
                    &format!("{prefix}.attn_v.weight"),
                    vec![d_model as u64, d_kv as u64],
                    (d_model * d_kv) as usize,
                    sd.wrapping_add(12),
                ));
                tensors.push(weight_tensor(
                    &format!("{prefix}.attn_output.weight"),
                    vec![d_q as u64, d_model as u64],
                    (d_q * d_model) as usize,
                    sd.wrapping_add(13),
                ));
            } else {
                // SSM / DeltaNet layer — the 11-tensor set.
                tensors.push(weight_tensor(
                    &format!("{prefix}.attn_qkv.weight"),
                    vec![d_model as u64, qkv_dim as u64],
                    (d_model * qkv_dim) as usize,
                    sd.wrapping_add(10),
                ));
                tensors.push(weight_tensor(
                    &format!("{prefix}.attn_gate.weight"),
                    vec![d_model as u64, ssm_inner as u64],
                    (d_model * ssm_inner) as usize,
                    sd.wrapping_add(11),
                ));
                // conv1d: `[conv_kernel, qkv_dim]` — the loader reads the LAST
                // dim as the depthwise conv channel count (== qkv_dim).
                tensors.push(weight_tensor(
                    &format!("{prefix}.ssm_conv1d.weight"),
                    vec![conv_kernel as u64, qkv_dim as u64],
                    (conv_kernel * qkv_dim) as usize,
                    sd.wrapping_add(12),
                ));
                // `ssm_dt.bias` / `ssm_a` are norm-style f32 vectors of length
                // n_v_heads (ssm_a's length is how the loader derives n_v_heads).
                tensors.push(TensorSpec::f32(
                    &format!("{prefix}.ssm_dt.bias"),
                    vec![n_v_heads as u64],
                    random_f32(n_v_heads as usize, sd.wrapping_add(13)),
                ));
                tensors.push(TensorSpec::f32(
                    &format!("{prefix}.ssm_a"),
                    vec![n_v_heads as u64],
                    random_f32(n_v_heads as usize, sd.wrapping_add(14)),
                ));
                tensors.push(weight_tensor(
                    &format!("{prefix}.ssm_alpha.weight"),
                    vec![d_model as u64, n_v_heads as u64],
                    (d_model * n_v_heads) as usize,
                    sd.wrapping_add(15),
                ));
                tensors.push(weight_tensor(
                    &format!("{prefix}.ssm_beta.weight"),
                    vec![d_model as u64, n_v_heads as u64],
                    (d_model * n_v_heads) as usize,
                    sd.wrapping_add(16),
                ));
                // Gated-RMSNorm weight is per-(v-head) of width head_v_dim.
                tensors.push(TensorSpec::f32(
                    &format!("{prefix}.ssm_norm.weight"),
                    vec![head_v_dim as u64],
                    random_f32(head_v_dim as usize, sd.wrapping_add(17)),
                ));
                tensors.push(weight_tensor(
                    &format!("{prefix}.ssm_out.weight"),
                    vec![ssm_inner as u64, d_model as u64],
                    (ssm_inner * d_model) as usize,
                    sd.wrapping_add(18),
                ));
            }
        }
    }

    tensors.push(TensorSpec::f32(
        "output_norm.weight",
        vec![d_model as u64],
        random_f32(d_model as usize, seed.wrapping_add(99999)),
    ));
    // output.weight intentionally omitted -> tied to token_embd

    // Assemble file.
    let mut prefix = Writer::new();
    prefix.buf.extend_from_slice(GGUF_MAGIC);
    prefix.u32(GGUF_VERSION);
    prefix.u64(tensors.len() as u64);
    prefix.u64(m.n as u64);
    let mut header_kv = m.w.buf;
    prefix.buf.append(&mut header_kv);

    let alignment: u64 = 32;
    let mut info = Writer::new();
    let mut rolling: u64 = 0;
    for t in &tensors {
        info.str_(&t.name);
        info.u32(t.dims.len() as u32);
        for d in &t.dims {
            info.u64(*d);
        }
        info.u32(t.dtype);
        info.u64(rolling);
        let pad = (alignment - (t.bytes.len() as u64 % alignment)) % alignment;
        rolling += t.bytes.len() as u64 + pad;
    }

    let pre_data_len = (prefix.buf.len() + info.buf.len()) as u64;
    let pad_before = (alignment - (pre_data_len % alignment)) % alignment;

    let mut file = std::fs::File::create(path).expect("create");
    file.write_all(&prefix.buf).unwrap();
    file.write_all(&info.buf).unwrap();
    file.write_all(&vec![0u8; pad_before as usize]).unwrap();
    for t in &tensors {
        file.write_all(&t.bytes).unwrap();
        let pad = (alignment - (t.bytes.len() as u64 % alignment)) % alignment;
        if pad > 0 {
            file.write_all(&vec![0u8; pad as usize]).unwrap();
        }
    }
    file.flush().unwrap();
}

/// Write a synthetic DeepSeek-V2/V3-style MLA GGUF (`deepseek2` arch) to
/// `path`. Emits the full Multi-head Latent Attention tensor set so a
/// loadable fixture drives the REAL `forward_one_mla`:
///   - per layer: `attn_norm`, the Q path (`attn_q_a`+`attn_q_a_norm`+
///     `attn_q_b` when `q_lora_rank` is `Some`, else a single `attn_q`),
///     the KV path (`attn_kv_a_mqa`+`attn_kv_a_norm`+`attn_kv_b`),
///     `attn_output`, and a dense or MoE FFN.
/// GGUF dims are column-major `[in, out]` — the same convention the
/// Llama writer uses — so the loader's `out_dim`/`in_dim` resolve
/// correctly. Weight values are deterministic small RNG (coherence isn't
/// the point; LOAD + finite logits + naive≡absorbed equality are).
pub fn write_synthetic_deepseek_mla_gguf(path: &Path, params: &SynthMla) {
    let SynthMla {
        n_layers,
        n_heads,
        d_model,
        d_ff,
        vocab,
        ctx,
        seed,
        q_lora_rank,
        kv_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        moe,
        n_dense_layers,
    } = *params;

    assert!(qk_rope_head_dim % 2 == 0, "qk_rope_head_dim must be even (it is RoPE-rotated)");
    let qk_head_dim = qk_nope_head_dim + qk_rope_head_dim;
    let q_out = n_heads * qk_head_dim; // per-head [nope | rope]
    let kv_a_out = kv_lora_rank + qk_rope_head_dim; // latent | decoupled rope key
    let kv_b_out = n_heads * (qk_nope_head_dim + v_head_dim); // per-head [nope-K | V]
    let o_in = n_heads * v_head_dim;

    // All weights F16, norms F32 — matches real-world GGUFs.
    let wt = |name: &str, dims: Vec<u64>, n: usize, tseed: u64| -> TensorSpec {
        let f32s = random_f32(n, tseed);
        let f16s = f32s.into_iter().map(f16::from_f32).collect();
        TensorSpec::f16(name, dims, f16s)
    };
    let norm = |name: &str, n: usize, tseed: u64| -> TensorSpec {
        // Norm weights near 1.0 so RMSNorm behaves like a real model's.
        let v: Vec<f32> = random_f32(n, tseed).into_iter().map(|x| 1.0 + x).collect();
        TensorSpec::f32(name, vec![n as u64], v)
    };

    let mut m = MetaBuilder::new();
    m.string("general.architecture", "deepseek2");
    m.u32("deepseek2.block_count", n_layers);
    m.u32("deepseek2.embedding_length", d_model);
    m.u32("deepseek2.feed_forward_length", d_ff);
    m.u32("deepseek2.attention.head_count", n_heads);
    m.u32("deepseek2.attention.head_count_kv", n_heads);
    m.f32("deepseek2.attention.layer_norm_rms_epsilon", 1e-5);
    // MLA dims. `key_length` = qk_nope_head_dim (the loader also derives
    // it from the attn_kv_b tensor, so either reading is consistent).
    m.u32("deepseek2.attention.key_length", qk_nope_head_dim);
    m.u32("deepseek2.attention.value_length", v_head_dim);
    m.u32("deepseek2.rope.dimension_count", qk_rope_head_dim);
    m.f32("deepseek2.rope.freq_base", 10000.0);
    m.u32("deepseek2.context_length", ctx);
    m.u32("deepseek2.attention.kv_lora_rank", kv_lora_rank);
    if let Some(r) = q_lora_rank {
        m.u32("deepseek2.attention.q_lora_rank", r);
    }
    if let Some(mc) = moe {
        m.u32("deepseek2.expert_count", mc.n_experts);
        m.u32("deepseek2.expert_used_count", mc.n_experts_used);
        m.u32("deepseek2.expert_feed_forward_length", d_ff);
        if mc.n_experts_shared > 0 {
            m.u32("deepseek2.expert_shared_count", mc.n_experts_shared);
        }
    }
    let vocab_tokens: Vec<String> = (0..vocab).map(|i| format!("<tok_{i}>")).collect();
    m.array_strings("tokenizer.ggml.tokens", &vocab_tokens);
    m.string("tokenizer.ggml.model", "gpt2");
    m.array_strings("tokenizer.ggml.merges", &[]);
    m.string(
        "tokenizer.ggml.chat_template",
        "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n'}}{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}",
    );
    m.u32("tokenizer.ggml.bos_token_id", 1);
    m.u32("tokenizer.ggml.eos_token_id", 2);

    let mut tensors = Vec::new();
    tensors.push(wt(
        "token_embd.weight",
        vec![d_model as u64, vocab as u64],
        (vocab * d_model) as usize,
        seed.wrapping_add(1),
    ));

    for i in 0..n_layers {
        let prefix = format!("blk.{i}");
        let sd = seed.wrapping_add(10_000 + (i as u64) * 100);
        tensors.push(norm(&format!("{prefix}.attn_norm.weight"), d_model as usize, sd));
        // Q path.
        match q_lora_rank {
            Some(r) => {
                tensors.push(wt(
                    &format!("{prefix}.attn_q_a.weight"),
                    vec![d_model as u64, r as u64],
                    (d_model * r) as usize,
                    sd.wrapping_add(1),
                ));
                tensors.push(norm(&format!("{prefix}.attn_q_a_norm.weight"), r as usize, sd.wrapping_add(2)));
                tensors.push(wt(
                    &format!("{prefix}.attn_q_b.weight"),
                    vec![r as u64, q_out as u64],
                    (r * q_out) as usize,
                    sd.wrapping_add(3),
                ));
            }
            None => {
                tensors.push(wt(
                    &format!("{prefix}.attn_q.weight"),
                    vec![d_model as u64, q_out as u64],
                    (d_model * q_out) as usize,
                    sd.wrapping_add(1),
                ));
            }
        }
        // KV path.
        tensors.push(wt(
            &format!("{prefix}.attn_kv_a_mqa.weight"),
            vec![d_model as u64, kv_a_out as u64],
            (d_model * kv_a_out) as usize,
            sd.wrapping_add(4),
        ));
        tensors.push(norm(&format!("{prefix}.attn_kv_a_norm.weight"), kv_lora_rank as usize, sd.wrapping_add(5)));
        tensors.push(wt(
            &format!("{prefix}.attn_kv_b.weight"),
            vec![kv_lora_rank as u64, kv_b_out as u64],
            (kv_lora_rank * kv_b_out) as usize,
            sd.wrapping_add(6),
        ));
        tensors.push(wt(
            &format!("{prefix}.attn_output.weight"),
            vec![o_in as u64, d_model as u64],
            (o_in * d_model) as usize,
            sd.wrapping_add(7),
        ));
        // FFN. Dense for the first `n_dense_layers` (or all layers when
        // `moe` is None); MoE after.
        let is_dense = moe.is_none() || i < n_dense_layers;
        tensors.push(norm(&format!("{prefix}.ffn_norm.weight"), d_model as usize, sd.wrapping_add(8)));
        if is_dense {
            tensors.push(wt(
                &format!("{prefix}.ffn_gate.weight"),
                vec![d_model as u64, d_ff as u64],
                (d_model * d_ff) as usize,
                sd.wrapping_add(9),
            ));
            tensors.push(wt(
                &format!("{prefix}.ffn_up.weight"),
                vec![d_model as u64, d_ff as u64],
                (d_model * d_ff) as usize,
                sd.wrapping_add(10),
            ));
            tensors.push(wt(
                &format!("{prefix}.ffn_down.weight"),
                vec![d_ff as u64, d_model as u64],
                (d_ff * d_model) as usize,
                sd.wrapping_add(11),
            ));
        } else {
            let mc = moe.unwrap();
            let n_e = mc.n_experts as u64;
            tensors.push(wt(
                &format!("{prefix}.ffn_gate_inp.weight"),
                vec![d_model as u64, n_e],
                (d_model * mc.n_experts) as usize,
                sd.wrapping_add(9),
            ));
            tensors.push(wt(
                &format!("{prefix}.ffn_gate_exps.weight"),
                vec![d_model as u64, d_ff as u64, n_e],
                (d_model * d_ff * mc.n_experts) as usize,
                sd.wrapping_add(10),
            ));
            tensors.push(wt(
                &format!("{prefix}.ffn_up_exps.weight"),
                vec![d_model as u64, d_ff as u64, n_e],
                (d_model * d_ff * mc.n_experts) as usize,
                sd.wrapping_add(11),
            ));
            tensors.push(wt(
                &format!("{prefix}.ffn_down_exps.weight"),
                vec![d_ff as u64, d_model as u64, n_e],
                (d_ff * d_model * mc.n_experts) as usize,
                sd.wrapping_add(12),
            ));
            if mc.n_experts_shared > 0 {
                tensors.push(wt(
                    &format!("{prefix}.ffn_gate_shexp.weight"),
                    vec![d_model as u64, d_ff as u64],
                    (d_model * d_ff) as usize,
                    sd.wrapping_add(13),
                ));
                tensors.push(wt(
                    &format!("{prefix}.ffn_up_shexp.weight"),
                    vec![d_model as u64, d_ff as u64],
                    (d_model * d_ff) as usize,
                    sd.wrapping_add(14),
                ));
                tensors.push(wt(
                    &format!("{prefix}.ffn_down_shexp.weight"),
                    vec![d_ff as u64, d_model as u64],
                    (d_ff * d_model) as usize,
                    sd.wrapping_add(15),
                ));
            }
        }
    }

    tensors.push(norm("output_norm.weight", d_model as usize, seed.wrapping_add(99999)));
    // output.weight omitted → tied to token_embd.

    write_gguf_file(path, m, &tensors);
}

// ----------------------------------------------------------------
// LoRA adapter GGUF (P11) — merge / delta-apply / multi-LoRA fixture
// ----------------------------------------------------------------

/// Hyperparameters for a synthetic LoRA adapter GGUF matched to a base
/// [`SynthLlama`] model. Emits the public ggml adapter keys
/// (`general.type = "adapter"`, `adapter.type = "lora"`,
/// `adapter.lora.alpha`) + per-projection `<base>.weight.lora_a` /
/// `.lora_b` factor pairs sized from the base geometry, so the LoRA
/// merge / delta-apply / multi-LoRA tests run without a real downloaded
/// adapter. Factors are F32 (exact) with deterministic RNG values.
///
/// Factor layout (matches the real ggml adapter convention this repo
/// reads): `lora_a` has ggml `ne = [in, rank]` (→ `A [rank,in]`) and
/// `lora_b` has `ne = [rank, out]` (→ `B [out,rank]`); the effective
/// update is `W += (alpha/rank)·B·A`.
#[derive(Debug, Clone)]
pub struct SynthLoraAdapter {
    /// Base geometry the adapter targets (its projections are sized from
    /// `n_heads`/`head_dim`/`d_model`/`d_ff`/`n_layers`).
    pub base: SynthLlama,
    /// Low-rank dimension of every factor pair.
    pub rank: u32,
    /// `adapter.lora.alpha`; effective per-tensor scale is `alpha / rank`.
    pub alpha: f32,
    /// RNG seed for the factor values (distinct seeds ⇒ distinct adapters,
    /// which the multi-LoRA test relies on).
    pub seed: u64,
}

impl Default for SynthLoraAdapter {
    fn default() -> Self {
        Self {
            base: SynthLlama::default(),
            rank: 8,
            alpha: 16.0,
            seed: 0x10_4A_5EED,
        }
    }
}

/// Write a synthetic LoRA adapter GGUF to `path`, matched to `params.base`.
/// Adapts the dense per-layer attention + FFN projections (`attn_q`,
/// `attn_k`, `attn_v`, `attn_output`, `ffn_gate`, `ffn_up`, `ffn_down`) with
/// F32 low-rank factor pairs keyed by the base tensor names.
pub fn write_synthetic_lora_adapter_gguf(path: &Path, params: &SynthLoraAdapter) {
    let b = &params.base;
    let rank = params.rank;
    let d_model = b.d_model;
    let d_q = b.n_heads * b.head_dim;
    let d_kv = b.n_kv_heads * b.head_dim;
    let d_ff = b.d_ff;

    let mut m = MetaBuilder::new();
    // Base arch + adapter identity (public ggml adapter spec).
    m.string("general.architecture", "llama");
    m.string("general.type", "adapter");
    m.string("adapter.type", "lora");
    m.f32("adapter.lora.alpha", params.alpha);

    let f32t = |name: &str, dims: Vec<u64>, n: usize, tseed: u64| -> TensorSpec {
        TensorSpec::f32(name, dims, random_f32(n, tseed))
    };

    // (projection base-name, out, in) — base GGUF ne is `[in, out]`.
    let projections: [(&str, u32, u32); 7] = [
        ("attn_q", d_q, d_model),
        ("attn_k", d_kv, d_model),
        ("attn_v", d_kv, d_model),
        ("attn_output", d_model, d_q),
        ("ffn_gate", d_ff, d_model),
        ("ffn_up", d_ff, d_model),
        ("ffn_down", d_model, d_ff),
    ];

    let mut tensors: Vec<TensorSpec> = Vec::new();
    for i in 0..b.n_layers {
        for (pi, (proj, out, inp)) in projections.iter().enumerate() {
            let base_name = format!("blk.{i}.{proj}.weight");
            let sd = params
                .seed
                .wrapping_add((i as u64) * 1000 + (pi as u64) * 13 + 1);
            // lora_a: ggml ne [in, rank]  (physical rank rows × in cols).
            tensors.push(f32t(
                &format!("{base_name}.lora_a"),
                vec![*inp as u64, rank as u64],
                (*inp * rank) as usize,
                sd,
            ));
            // lora_b: ggml ne [rank, out] (physical out rows × rank cols).
            tensors.push(f32t(
                &format!("{base_name}.lora_b"),
                vec![rank as u64, *out as u64],
                (rank * *out) as usize,
                sd.wrapping_add(0x5_0000),
            ));
        }
    }

    write_gguf_file(path, m, &tensors);
}

/// Parameters for a synthetic P10b EAGLE draft-head companion GGUF, matched to
/// a [`SynthLlama`] base. Emits the public clean-room EAGLE head namespace
/// (`adapter.type = "eagle"` + `eagle.*` tensors) that
/// `rustllama_models::llama_arch::EagleHead::from_gguf` loads.
#[derive(Debug, Clone)]
pub struct SynthEagleHead {
    /// Base geometry the head drafts for (its tensors are sized from
    /// `d_model` / `n_heads` / `head_dim` / `d_ff` / `vocab`).
    pub base: SynthLlama,
    /// RNG seed for the head's weight values.
    pub seed: u64,
    /// When `true`, emit a dedicated `eagle.output.weight` LM head +
    /// `eagle.token_embd.weight`; otherwise the head ties to the base output /
    /// embedding (the common case, and what `EagleHead { lm_head: None,
    /// token_embd: None }` exercises).
    pub own_lm_head: bool,
}

impl Default for SynthEagleHead {
    fn default() -> Self {
        Self {
            base: SynthLlama::default(),
            seed: 0xEA61_E5EE,
            own_lm_head: false,
        }
    }
}

/// Write a synthetic EAGLE draft-head companion GGUF to `path`, matched to
/// `params.base`. Carries the fusion projection (`eagle.fc.weight` +
/// pre-fusion `enorm`/`hnorm` + `fc.bias`), one decoder block
/// (`eagle.{attn_*,ffn_*}`), and — when `own_lm_head` is set — a dedicated
/// output norm + LM head + embedding (else tied to the base).
pub fn write_synthetic_eagle_head_gguf(path: &Path, params: &SynthEagleHead) {
    let b = &params.base;
    let d_model = b.d_model;
    let d_q = b.n_heads * b.head_dim;
    let d_kv = b.n_kv_heads * b.head_dim;
    let d_ff = b.d_ff;
    let vocab = b.vocab;
    let seed = params.seed;

    let mut m = MetaBuilder::new();
    // Base arch + adapter identity (clean-room EAGLE head namespace).
    m.string("general.architecture", "llama");
    m.string("general.type", "adapter");
    m.string("adapter.type", "eagle");
    m.u32("eagle.embedding_length", d_model);

    let f32w = |name: &str, dims: Vec<u64>, n: usize, tseed: u64| -> TensorSpec {
        TensorSpec::f32(name, dims, random_f32(n, tseed))
    };

    let mut tensors: Vec<TensorSpec> = Vec::new();
    // Fusion: concat([embed, hidden]) [2·d] → [d]. ggml ne = [in, out].
    tensors.push(f32w(
        "eagle.fc.weight",
        vec![(2 * d_model) as u64, d_model as u64],
        (2 * d_model * d_model) as usize,
        seed.wrapping_add(1),
    ));
    tensors.push(f32w("eagle.fc.bias", vec![d_model as u64], d_model as usize, seed.wrapping_add(2)));
    tensors.push(f32w("eagle.enorm.weight", vec![d_model as u64], d_model as usize, seed.wrapping_add(3)));
    tensors.push(f32w("eagle.hnorm.weight", vec![d_model as u64], d_model as usize, seed.wrapping_add(4)));

    // Decoder block (same shapes as a base `blk.{i}.*` layer).
    tensors.push(f32w("eagle.attn_norm.weight", vec![d_model as u64], d_model as usize, seed.wrapping_add(10)));
    tensors.push(f32w(
        "eagle.attn_q.weight",
        vec![d_model as u64, d_q as u64],
        (d_model * d_q) as usize,
        seed.wrapping_add(11),
    ));
    tensors.push(f32w(
        "eagle.attn_k.weight",
        vec![d_model as u64, d_kv as u64],
        (d_model * d_kv) as usize,
        seed.wrapping_add(12),
    ));
    tensors.push(f32w(
        "eagle.attn_v.weight",
        vec![d_model as u64, d_kv as u64],
        (d_model * d_kv) as usize,
        seed.wrapping_add(13),
    ));
    tensors.push(f32w(
        "eagle.attn_output.weight",
        vec![d_q as u64, d_model as u64],
        (d_q * d_model) as usize,
        seed.wrapping_add(14),
    ));
    tensors.push(f32w("eagle.ffn_norm.weight", vec![d_model as u64], d_model as usize, seed.wrapping_add(15)));
    tensors.push(f32w(
        "eagle.ffn_gate.weight",
        vec![d_model as u64, d_ff as u64],
        (d_model * d_ff) as usize,
        seed.wrapping_add(16),
    ));
    tensors.push(f32w(
        "eagle.ffn_up.weight",
        vec![d_model as u64, d_ff as u64],
        (d_model * d_ff) as usize,
        seed.wrapping_add(17),
    ));
    tensors.push(f32w(
        "eagle.ffn_down.weight",
        vec![d_ff as u64, d_model as u64],
        (d_ff * d_model) as usize,
        seed.wrapping_add(18),
    ));

    if params.own_lm_head {
        tensors.push(f32w("eagle.output_norm.weight", vec![d_model as u64], d_model as usize, seed.wrapping_add(20)));
        tensors.push(f32w(
            "eagle.output.weight",
            vec![d_model as u64, vocab as u64],
            (d_model * vocab) as usize,
            seed.wrapping_add(21),
        ));
        tensors.push(f32w(
            "eagle.token_embd.weight",
            vec![d_model as u64, vocab as u64],
            (d_model * vocab) as usize,
            seed.wrapping_add(22),
        ));
    }

    write_gguf_file(path, m, &tensors);
}

/// Hyperparameters for the synthetic BERT-family embedding model.
/// Defaults produce a tiny 2-layer / 64-d_model BERT useful for
/// round-trip + load tests of `rustllama_models::bert_arch`.
#[derive(Debug, Clone)]
pub struct SynthBert {
    pub n_layers: u32,
    pub n_heads: u32,
    pub d_model: u32,
    pub d_ff: u32,
    pub vocab: u32,
    pub ctx: u32,
    pub n_token_types: u32,
    pub seed: u64,
    pub weight_dtype: SynthWeightDtype,
    /// When > 0, writes a `cls.weight` `[d_model, n_labels]` +
    /// `cls.bias` `[n_labels]` head — turns the synth GGUF into a
    /// reranker-style fixture that `BertModel::forward_classify`
    /// can run. 0 = pure embedding model (the default).
    pub classifier_n_labels: u32,
}

impl Default for SynthBert {
    fn default() -> Self {
        Self {
            n_layers: 2,
            n_heads: 2,
            d_model: 64,
            d_ff: 128,
            // Default vocab is sized for the synthetic BERT WordPiece
            // vocab: 5 special tokens ([PAD]/[UNK]/[CLS]/[SEP]/[MASK])
            // + 64 word tokens + 32 `##suffix` continuing tokens =
            // 101 tokens. Sized at 128 to round up.
            vocab: 128,
            ctx: 32,
            n_token_types: 2,
            seed: 0xBE57_BE57_BE57_BE57_u64,
            weight_dtype: SynthWeightDtype::F16,
            classifier_n_labels: 0,
        }
    }
}

impl SynthBert {
    /// Returns a copy of `self` with a `cls.weight` / `cls.bias`
    /// classifier head sized to `n_labels` outputs. Use to write
    /// reranker-style GGUFs (BGE-reranker convention is
    /// `n_labels = 1` → single relevance scalar).
    pub fn with_classifier_head(mut self, n_labels: u32) -> Self {
        self.classifier_n_labels = n_labels;
        self
    }
}

/// IDs of the BERT special tokens in the synthetic vocab. Stable
/// across calls so tests can hard-code expected ID values.
pub const SYNTH_BERT_PAD_ID: u32 = 0;
pub const SYNTH_BERT_UNK_ID: u32 = 1;
pub const SYNTH_BERT_CLS_ID: u32 = 2;
pub const SYNTH_BERT_SEP_ID: u32 = 3;
pub const SYNTH_BERT_MASK_ID: u32 = 4;
/// Number of BERT special tokens emitted (PAD/UNK/CLS/SEP/MASK).
pub const SYNTH_BERT_N_SPECIAL: u32 = 5;

/// Write a synthetic BERT GGUF to `path`. Mirrors
/// `write_synthetic_llama_gguf` but emits the BERT-family metadata
/// keys + tensor names that `rustllama_models::bert_arch::BertModel::load`
/// expects: token / token_types / position embeddings + LN bias
/// tensors + per-layer Q/K/V/O projections-with-bias + FFN-with-bias.
///
/// Used by the v1.1 embedding-loader tests. The forward pass is a
/// follow-up turn, so this synth only needs metadata + tensor
/// presence to be correct — the actual weight values can be
/// arbitrary RNG.
pub fn write_synthetic_bert_gguf(path: &Path, params: &SynthBert) {
    let classifier_n_labels = params.classifier_n_labels;
    let SynthBert {
        n_layers,
        n_heads,
        d_model,
        d_ff,
        vocab,
        ctx,
        n_token_types,
        seed,
        weight_dtype,
        classifier_n_labels: _,
    } = *params;
    let head_dim = d_model / n_heads.max(1);

    let weight_tensor = |name: &str, dims: Vec<u64>, n: usize, tseed: u64| -> TensorSpec {
        let f32s = random_f32(n, tseed);
        match weight_dtype {
            SynthWeightDtype::F16 => {
                let f16s = f32s.into_iter().map(f16::from_f32).collect();
                TensorSpec::f16(name, dims, f16s)
            }
            SynthWeightDtype::Bf16 => TensorSpec::bf16(name, dims, f32s),
        }
    };
    let f32_bias = |name: &str, n: usize, tseed: u64| -> TensorSpec {
        TensorSpec::f32(name, vec![n as u64], random_f32(n, tseed))
    };

    let mut m = MetaBuilder::new();
    m.string("general.architecture", "bert");
    m.u32("bert.block_count", n_layers);
    m.u32("bert.embedding_length", d_model);
    m.u32("bert.feed_forward_length", d_ff);
    m.u32("bert.attention.head_count", n_heads);
    m.u32("bert.attention.key_length", head_dim);
    m.f32("bert.attention.layer_norm_epsilon", 1e-12);
    m.u32("bert.context_length", ctx);
    m.u32("bert.token_types_count", n_token_types);
    m.string("bert.pooling_type", "mean");

    // BERT WordPiece-style vocab: 5 special tokens first (PAD, UNK,
    // CLS, SEP, MASK) so the IDs are stable, then enough real
    // ASCII-lowercase word + `##suffix` tokens to encode common test
    // strings. Remaining vocab slots are padded with `<unused_N>`
    // entries so embedding weight tensors can be sized at the caller's
    // chosen `vocab` count without us having to expand the word list.
    let mut vocab_tokens: Vec<String> = Vec::with_capacity(vocab as usize);
    vocab_tokens.push("[PAD]".into());
    vocab_tokens.push("[UNK]".into());
    vocab_tokens.push("[CLS]".into());
    vocab_tokens.push("[SEP]".into());
    vocab_tokens.push("[MASK]".into());
    // ASCII lowercase a-z + common test words. These are the tokens a
    // WordPiece tokenizer will produce for a normalized + whitespace-
    // split + char-level-fallback input on lowercase text.
    let words: &[&str] = &[
        "the", "a", "an", "and", "or", "is", "are", "was", "were", "be",
        "to", "of", "in", "on", "at", "for", "with", "by", "from", "as",
        "this", "that", "it", "i", "you", "he", "she", "we", "they",
        "hello", "world", "test", "input", "embed", "model", "bert",
        "fox", "dog", "cat", "jump", "over", "lazy", "quick", "brown",
    ];
    for w in words {
        vocab_tokens.push((*w).to_string());
    }
    // Single ASCII letters as a char-level fallback so unknown words
    // still WordPiece-split into letters rather than [UNK].
    for c in b'a'..=b'z' {
        let s = (c as char).to_string();
        if !vocab_tokens.iter().any(|t| t == &s) {
            vocab_tokens.push(s);
        }
    }
    // Continuing-subword variants (`##s`, `##ed`, ...) for common suffixes.
    let suffixes: &[&str] = &["##s", "##ed", "##ing", "##ly", "##er", "##est"];
    for s in suffixes {
        vocab_tokens.push((*s).to_string());
    }
    // Pad out to the configured vocab size with deterministic
    // `<unused_N>` placeholders.
    let mut next_pad = 0u32;
    while (vocab_tokens.len() as u32) < vocab {
        vocab_tokens.push(format!("<unused_{next_pad}>"));
        next_pad += 1;
    }
    // Truncate if the configured vocab is smaller than our seeded
    // word list (shouldn't happen with default vocab=128, but stays
    // robust for callers that lower it).
    vocab_tokens.truncate(vocab as usize);
    m.array_strings("tokenizer.ggml.tokens", &vocab_tokens);

    // Token-type array: 3 (control) for the special tokens, 1
    // (normal) for the word + char + suffix entries, 5 (unused) for
    // the padded slots.
    let token_types: Vec<i32> = vocab_tokens
        .iter()
        .map(|t| {
            if matches!(
                t.as_str(),
                "[PAD]" | "[UNK]" | "[CLS]" | "[SEP]" | "[MASK]"
            ) {
                3
            } else if t.starts_with("<unused_") {
                5
            } else {
                1
            }
        })
        .collect();
    m.array_i32("tokenizer.ggml.token_type", &token_types);

    m.string("tokenizer.ggml.model", "bert");
    m.u32("tokenizer.ggml.padding_token_id", SYNTH_BERT_PAD_ID);
    m.u32("tokenizer.ggml.unknown_token_id", SYNTH_BERT_UNK_ID);
    m.u32("tokenizer.ggml.cls_token_id", SYNTH_BERT_CLS_ID);
    m.u32("tokenizer.ggml.sep_token_id", SYNTH_BERT_SEP_ID);
    m.u32("tokenizer.ggml.mask_token_id", SYNTH_BERT_MASK_ID);

    let mut tensors = Vec::new();
    tensors.push(weight_tensor(
        "token_embd.weight",
        vec![d_model as u64, vocab as u64],
        (vocab * d_model) as usize,
        seed.wrapping_add(1),
    ));
    tensors.push(weight_tensor(
        "token_types.weight",
        vec![d_model as u64, n_token_types as u64],
        (n_token_types * d_model) as usize,
        seed.wrapping_add(2),
    ));
    tensors.push(weight_tensor(
        "position_embd.weight",
        vec![d_model as u64, ctx as u64],
        (ctx * d_model) as usize,
        seed.wrapping_add(3),
    ));
    tensors.push(f32_bias(
        "token_embd_norm.weight",
        d_model as usize,
        seed.wrapping_add(4),
    ));
    tensors.push(f32_bias(
        "token_embd_norm.bias",
        d_model as usize,
        seed.wrapping_add(5),
    ));

    for i in 0..n_layers {
        let prefix = format!("blk.{i}");
        let base = 1000 + (i as u64) * 100;
        // Q/K/V projections (no GQA — n_heads == n_kv_heads).
        for (name_off, name) in [(10u64, "attn_q"), (12, "attn_k"), (14, "attn_v")] {
            tensors.push(weight_tensor(
                &format!("{prefix}.{name}.weight"),
                vec![d_model as u64, d_model as u64],
                (d_model * d_model) as usize,
                seed.wrapping_add(base + name_off),
            ));
            tensors.push(f32_bias(
                &format!("{prefix}.{name}.bias"),
                d_model as usize,
                seed.wrapping_add(base + name_off + 1),
            ));
        }
        // Output projection + post-attn LN.
        tensors.push(weight_tensor(
            &format!("{prefix}.attn_output.weight"),
            vec![d_model as u64, d_model as u64],
            (d_model * d_model) as usize,
            seed.wrapping_add(base + 20),
        ));
        tensors.push(f32_bias(
            &format!("{prefix}.attn_output.bias"),
            d_model as usize,
            seed.wrapping_add(base + 21),
        ));
        tensors.push(f32_bias(
            &format!("{prefix}.attn_output_norm.weight"),
            d_model as usize,
            seed.wrapping_add(base + 22),
        ));
        tensors.push(f32_bias(
            &format!("{prefix}.attn_output_norm.bias"),
            d_model as usize,
            seed.wrapping_add(base + 23),
        ));
        // FFN: single up + down (BERT uses one hidden layer; no
        // gated up/down split like SwiGLU).
        tensors.push(weight_tensor(
            &format!("{prefix}.ffn_up.weight"),
            vec![d_model as u64, d_ff as u64],
            (d_model * d_ff) as usize,
            seed.wrapping_add(base + 30),
        ));
        tensors.push(f32_bias(
            &format!("{prefix}.ffn_up.bias"),
            d_ff as usize,
            seed.wrapping_add(base + 31),
        ));
        tensors.push(weight_tensor(
            &format!("{prefix}.ffn_down.weight"),
            vec![d_ff as u64, d_model as u64],
            (d_ff * d_model) as usize,
            seed.wrapping_add(base + 32),
        ));
        tensors.push(f32_bias(
            &format!("{prefix}.ffn_down.bias"),
            d_model as usize,
            seed.wrapping_add(base + 33),
        ));
        tensors.push(f32_bias(
            &format!("{prefix}.layer_output_norm.weight"),
            d_model as usize,
            seed.wrapping_add(base + 40),
        ));
        tensors.push(f32_bias(
            &format!("{prefix}.layer_output_norm.bias"),
            d_model as usize,
            seed.wrapping_add(base + 41),
        ));
    }

    // Classifier head — present when classifier_n_labels > 0.
    // BGE-reranker convention: `cls.weight` shape
    // `[d_model, n_labels]`, `cls.bias` shape `[n_labels]`.
    if classifier_n_labels > 0 {
        tensors.push(weight_tensor(
            "cls.weight",
            vec![d_model as u64, classifier_n_labels as u64],
            (d_model * classifier_n_labels) as usize,
            seed.wrapping_add(0xCC_CCCC),
        ));
        tensors.push(f32_bias(
            "cls.bias",
            classifier_n_labels as usize,
            seed.wrapping_add(0xCC_CCCD),
        ));
    }

    // Assemble file — same shape as the Llama writer below.
    let mut prefix = Writer::new();
    prefix.buf.extend_from_slice(GGUF_MAGIC);
    prefix.u32(GGUF_VERSION);
    prefix.u64(tensors.len() as u64);
    prefix.u64(m.n as u64);
    let mut header_kv = m.w.buf;
    prefix.buf.append(&mut header_kv);

    let alignment: u64 = 32;
    let mut info = Writer::new();
    let mut rolling: u64 = 0;
    for t in &tensors {
        info.str_(&t.name);
        info.u32(t.dims.len() as u32);
        for d in &t.dims {
            info.u64(*d);
        }
        info.u32(t.dtype);
        info.u64(rolling);
        let pad = (alignment - (t.bytes.len() as u64 % alignment)) % alignment;
        rolling += t.bytes.len() as u64 + pad;
    }
    let pre_data_len = (prefix.buf.len() + info.buf.len()) as u64;
    let pad_before = (alignment - (pre_data_len % alignment)) % alignment;

    use std::io::Write as _;
    let mut file = std::fs::File::create(path).expect("create");
    file.write_all(&prefix.buf).unwrap();
    file.write_all(&info.buf).unwrap();
    file.write_all(&vec![0u8; pad_before as usize]).unwrap();
    for t in &tensors {
        file.write_all(&t.bytes).unwrap();
        let pad = (alignment - (t.bytes.len() as u64 % alignment)) % alignment;
        if pad > 0 {
            file.write_all(&vec![0u8; pad as usize]).unwrap();
        }
    }
    file.flush().unwrap();
}

// ----------------------------------------------------------------
// CLIP mmproj GGUF (vision-tower + projector) — phase V-0 scaffold
// ----------------------------------------------------------------
//
// Minimal synth fixture for the `rustllama_models::vision_arch`
// module. Today this only needs to carry the `clip.vision.*`
// metadata keys + a single placeholder tensor (GGUFs with zero
// tensors are technically valid but parsers occasionally trip on
// them; one minimal tensor keeps round-trips clean). Phase V-1 of
// the VLM arc expands this with patch-embed / position-embed /
// per-layer attention+FFN tensors once the loader binds them.

/// Write a synthetic CLIP mmproj GGUF to `path`. Carries
/// `clip.vision.*` metadata + every tensor `VisionModel::load`
/// binds: patch_embd, position_embd (with class-token row),
/// class_embd, pre/post LN, per-block attn (Q/K/V/O + biases) +
/// pre-LN, per-block FFN (up/down + biases) + pre-LN, and the
/// 2-layer MLP projector (`mm.0` + `mm.2`).
pub fn write_synthetic_clip_mmproj_gguf(path: &Path) {
    // LLaVA-1.5 shape scaled to fit a unit-test fixture. 336/14 = 24
    // patches per side → 576 patches; plus the class token = 577 rows
    // in the position embedding. n_layers=2 + d_model=128 keep the
    // file small while still exercising the per-block binding loop.
    let image_size: u32 = 336;
    let patch_size: u32 = 14;
    let n_channels: u32 = 3;
    let n_layers: u32 = 2;
    let n_heads: u32 = 4;
    let d_model: u32 = 128;
    let d_ff: u32 = 256;
    // Projector intermediate dim — for an mmproj-only fixture the
    // text-decoder's d_model is what's on the OTHER side of the
    // projector; we don't have it here, so just pick a small value
    // that's distinct from `d_model` so a wrong projector binding
    // would surface as a shape mismatch.
    let d_proj_hidden: u32 = 64;
    let d_text: u32 = 96;
    let per_side = image_size / patch_size;
    let num_patches = per_side * per_side;
    let pos_rows = num_patches + 1; // with class token

    let mut m = MetaBuilder::new();
    m.string("general.architecture", "clip");
    m.bool("clip.has_vision_encoder", true);
    m.bool("clip.has_text_encoder", false);
    m.string("clip.projector_type", "mlp");
    m.u32("clip.vision.image_size", image_size);
    m.u32("clip.vision.patch_size", patch_size);
    m.u32("clip.vision.block_count", n_layers);
    m.u32("clip.vision.attention.head_count", n_heads);
    m.u32("clip.vision.embedding_length", d_model);
    m.u32("clip.vision.feed_forward_length", d_ff);
    m.f32("clip.vision.attention.layer_norm_epsilon", 1e-6);

    let seed: u64 = 0xC11C_C11Cu64;
    let f32_tensor = |name: &str, dims: Vec<u64>, n: usize, tseed: u64| -> TensorSpec {
        TensorSpec::f32(name, dims, random_f32(n, tseed))
    };
    let f32_bias = |name: &str, n: usize, tseed: u64| -> TensorSpec {
        TensorSpec::f32(name, vec![n as u64], random_f32(n, tseed))
    };

    let mut tensors: Vec<TensorSpec> = Vec::new();
    // Patch embedding: conv2d kernel `[d_model, n_channels, ps, ps]`.
    // The loader binds it as-is; phase V-1b will reshape to
    // `[d_model, n_channels*ps*ps]` for the matvec hot path.
    tensors.push(f32_tensor(
        "v.patch_embd.weight",
        vec![d_model as u64, n_channels as u64, patch_size as u64, patch_size as u64],
        (d_model * n_channels * patch_size * patch_size) as usize,
        seed.wrapping_add(1),
    ));
    tensors.push(f32_bias(
        "v.patch_embd.bias",
        d_model as usize,
        seed.wrapping_add(2),
    ));
    // Position embedding `[num_patches + 1, d_model]`.
    tensors.push(f32_tensor(
        "v.position_embd.weight",
        vec![pos_rows as u64, d_model as u64],
        (pos_rows * d_model) as usize,
        seed.wrapping_add(3),
    ));
    // Class token `[d_model]`.
    tensors.push(f32_bias(
        "v.class_embd",
        d_model as usize,
        seed.wrapping_add(4),
    ));
    // Pre-LN (optional in spec, present in real CLIP weights).
    tensors.push(f32_bias(
        "v.pre_ln.weight",
        d_model as usize,
        seed.wrapping_add(5),
    ));
    tensors.push(f32_bias(
        "v.pre_ln.bias",
        d_model as usize,
        seed.wrapping_add(6),
    ));
    // Post-LN.
    tensors.push(f32_bias(
        "v.post_ln.weight",
        d_model as usize,
        seed.wrapping_add(7),
    ));
    tensors.push(f32_bias(
        "v.post_ln.bias",
        d_model as usize,
        seed.wrapping_add(8),
    ));

    // Per-block: pre-attn LN, Q/K/V/O + biases, pre-FFN LN, FFN
    // up/down + biases.
    for i in 0..n_layers {
        let prefix = format!("v.blk.{i}");
        let base = 100 + (i as u64) * 100;
        // Pre-attn LN.
        tensors.push(f32_bias(
            &format!("{prefix}.attn_norm.weight"),
            d_model as usize,
            seed.wrapping_add(base),
        ));
        tensors.push(f32_bias(
            &format!("{prefix}.attn_norm.bias"),
            d_model as usize,
            seed.wrapping_add(base + 1),
        ));
        // Q / K / V projections + biases.
        for (off, name) in [(10u64, "attn_q"), (12, "attn_k"), (14, "attn_v")] {
            tensors.push(f32_tensor(
                &format!("{prefix}.{name}.weight"),
                vec![d_model as u64, d_model as u64],
                (d_model * d_model) as usize,
                seed.wrapping_add(base + off),
            ));
            tensors.push(f32_bias(
                &format!("{prefix}.{name}.bias"),
                d_model as usize,
                seed.wrapping_add(base + off + 1),
            ));
        }
        // Output projection + bias.
        tensors.push(f32_tensor(
            &format!("{prefix}.attn_output.weight"),
            vec![d_model as u64, d_model as u64],
            (d_model * d_model) as usize,
            seed.wrapping_add(base + 20),
        ));
        tensors.push(f32_bias(
            &format!("{prefix}.attn_output.bias"),
            d_model as usize,
            seed.wrapping_add(base + 21),
        ));
        // Pre-FFN LN.
        tensors.push(f32_bias(
            &format!("{prefix}.ffn_norm.weight"),
            d_model as usize,
            seed.wrapping_add(base + 22),
        ));
        tensors.push(f32_bias(
            &format!("{prefix}.ffn_norm.bias"),
            d_model as usize,
            seed.wrapping_add(base + 23),
        ));
        // FFN up `[d_ff, d_model]` + bias `[d_ff]`.
        tensors.push(f32_tensor(
            &format!("{prefix}.ffn_up.weight"),
            vec![d_ff as u64, d_model as u64],
            (d_ff * d_model) as usize,
            seed.wrapping_add(base + 30),
        ));
        tensors.push(f32_bias(
            &format!("{prefix}.ffn_up.bias"),
            d_ff as usize,
            seed.wrapping_add(base + 31),
        ));
        // FFN down `[d_model, d_ff]` + bias `[d_model]`.
        tensors.push(f32_tensor(
            &format!("{prefix}.ffn_down.weight"),
            vec![d_model as u64, d_ff as u64],
            (d_model * d_ff) as usize,
            seed.wrapping_add(base + 32),
        ));
        tensors.push(f32_bias(
            &format!("{prefix}.ffn_down.bias"),
            d_model as usize,
            seed.wrapping_add(base + 33),
        ));
    }

    // Projector — 2-layer MLP per `clip.projector_type = "mlp"`:
    // mm.0: [d_proj_hidden, d_model] + bias [d_proj_hidden]
    // mm.2: [d_text, d_proj_hidden] + bias [d_text]
    // (the `mm.1` index slot is the GELU activation in the PyTorch
    // module list; GGUF skips it.)
    tensors.push(f32_tensor(
        "mm.0.weight",
        vec![d_proj_hidden as u64, d_model as u64],
        (d_proj_hidden * d_model) as usize,
        seed.wrapping_add(900),
    ));
    tensors.push(f32_bias(
        "mm.0.bias",
        d_proj_hidden as usize,
        seed.wrapping_add(901),
    ));
    tensors.push(f32_tensor(
        "mm.2.weight",
        vec![d_text as u64, d_proj_hidden as u64],
        (d_text * d_proj_hidden) as usize,
        seed.wrapping_add(902),
    ));
    tensors.push(f32_bias(
        "mm.2.bias",
        d_text as usize,
        seed.wrapping_add(903),
    ));

    // Assemble file — same layout as the Llama / BERT synths.
    let mut prefix = Writer::new();
    prefix.buf.extend_from_slice(GGUF_MAGIC);
    prefix.u32(GGUF_VERSION);
    prefix.u64(tensors.len() as u64);
    prefix.u64(m.n as u64);
    let mut header_kv = m.w.buf;
    prefix.buf.append(&mut header_kv);

    let alignment: u64 = 32;
    let mut info = Writer::new();
    let mut rolling: u64 = 0;
    for t in &tensors {
        info.str_(&t.name);
        info.u32(t.dims.len() as u32);
        for d in &t.dims {
            info.u64(*d);
        }
        info.u32(t.dtype);
        info.u64(rolling);
        let pad = (alignment - (t.bytes.len() as u64 % alignment)) % alignment;
        rolling += t.bytes.len() as u64 + pad;
    }

    let pre_data_len = (prefix.buf.len() + info.buf.len()) as u64;
    let pad_before = (alignment - (pre_data_len % alignment)) % alignment;

    let mut file = std::fs::File::create(path).expect("create");
    file.write_all(&prefix.buf).unwrap();
    file.write_all(&info.buf).unwrap();
    file.write_all(&vec![0u8; pad_before as usize]).unwrap();
    for t in &tensors {
        file.write_all(&t.bytes).unwrap();
        let pad = (alignment - (t.bytes.len() as u64 % alignment)) % alignment;
        if pad > 0 {
            file.write_all(&vec![0u8; pad as usize]).unwrap();
        }
    }
    file.flush().unwrap();
}

/// Qwen3-VL-shaped mmproj fixture — the Bonsai 2 vision-tower layout
/// scaled down for unit tests. Differences from the classic CLIP
/// fixture, all matching the REAL 27B mmproj:
///   - fused per-block `attn_qkv.{weight,bias}`, `attn_out.*`,
///     `ln1`/`ln2` norm names (no attn_norm/ffn_norm)
///   - DUAL patch embeddings (`v.patch_embd.weight` + `.weight.1`)
///   - no class token, no pre_ln; `v.position_embd` rows == patches
///   - `clip.projector_type = "qwen3vl_merger"`,
///     `clip.vision.spatial_merge_size = 2`,
///     `clip.vision.is_deepstack_layers` (all-false, or all-true via
///     `deepstack_true` for the refusal test)
///   - 2-D tensors use the REAL GGUF dim order `[in, out]` (ne0
///     fastest) — the classic fixture writes `[out, in]`, which
///     masked a dim-order bug once already.
/// Geometry: image 32, patch 8 → 4×4 = 16 patches → 4 merged tokens;
/// d_model 64, 4 heads (d_head 16), ffn 128, 2 layers, d_text 96.
pub fn write_synthetic_qwen3vl_mmproj_gguf(path: &Path, deepstack_true: bool) {
    let image_size: u32 = 32;
    let patch_size: u32 = 8;
    let n_channels: u32 = 3;
    let n_layers: u32 = 2;
    let n_heads: u32 = 4;
    let d_model: u32 = 64;
    let d_ff: u32 = 128;
    let d_text: u32 = 96;
    let merge: u32 = 2;
    let d_merged = d_model * merge * merge; // 256
    let per_side = image_size / patch_size;
    let num_patches = per_side * per_side; // 16, no class token

    let mut m = MetaBuilder::new();
    m.string("general.architecture", "clip");
    m.bool("clip.has_vision_encoder", true);
    m.string("clip.projector_type", "qwen3vl_merger");
    m.bool("clip.use_gelu", true);
    m.u32("clip.vision.image_size", image_size);
    m.u32("clip.vision.patch_size", patch_size);
    m.u32("clip.vision.block_count", n_layers);
    m.u32("clip.vision.attention.head_count", n_heads);
    m.u32("clip.vision.embedding_length", d_model);
    m.u32("clip.vision.feed_forward_length", d_ff);
    m.u32("clip.vision.projection_dim", d_text);
    m.u32("clip.vision.spatial_merge_size", merge);
    m.f32("clip.vision.attention.layer_norm_epsilon", 1e-6);
    m.array_f32("clip.vision.image_mean", &[0.5, 0.5, 0.5]);
    m.array_f32("clip.vision.image_std", &[0.5, 0.5, 0.5]);
    m.array_bool(
        "clip.vision.is_deepstack_layers",
        &vec![deepstack_true; n_layers as usize],
    );

    let seed: u64 = 0x3B05_A120u64;
    let f32_tensor = |name: &str, dims: Vec<u64>, n: usize, tseed: u64| -> TensorSpec {
        TensorSpec::f32(name, dims, random_f32(n, tseed))
    };
    let f32_bias = |name: &str, n: usize, tseed: u64| -> TensorSpec {
        TensorSpec::f32(name, vec![n as u64], random_f32(n, tseed))
    };

    let mut tensors: Vec<TensorSpec> = Vec::new();
    // Dual patch conv kernels — real dims order [ps, ps, ch, d_model].
    let patch_elems = (patch_size * patch_size * n_channels * d_model) as usize;
    let patch_dims = vec![
        patch_size as u64,
        patch_size as u64,
        n_channels as u64,
        d_model as u64,
    ];
    tensors.push(f32_tensor("v.patch_embd.weight", patch_dims.clone(), patch_elems, seed + 1));
    tensors.push(f32_tensor("v.patch_embd.weight.1", patch_dims, patch_elems, seed + 2));
    tensors.push(f32_bias("v.patch_embd.bias", d_model as usize, seed + 3));
    // Position embedding — real dims order [d_model, rows], rows ==
    // patches (no class token).
    tensors.push(f32_tensor(
        "v.position_embd.weight",
        vec![d_model as u64, num_patches as u64],
        (d_model * num_patches) as usize,
        seed + 4,
    ));
    tensors.push(f32_bias("v.post_ln.weight", d_model as usize, seed + 5));
    tensors.push(f32_bias("v.post_ln.bias", d_model as usize, seed + 6));

    for i in 0..n_layers {
        let prefix = format!("v.blk.{i}");
        let base = seed + 100 + (i as u64) * 100;
        // Fused QKV — [in, 3*out] real order; bias [3*out].
        tensors.push(f32_tensor(
            &format!("{prefix}.attn_qkv.weight"),
            vec![d_model as u64, (3 * d_model) as u64],
            (3 * d_model * d_model) as usize,
            base + 1,
        ));
        tensors.push(f32_bias(
            &format!("{prefix}.attn_qkv.bias"),
            (3 * d_model) as usize,
            base + 2,
        ));
        tensors.push(f32_tensor(
            &format!("{prefix}.attn_out.weight"),
            vec![d_model as u64, d_model as u64],
            (d_model * d_model) as usize,
            base + 3,
        ));
        tensors.push(f32_bias(&format!("{prefix}.attn_out.bias"), d_model as usize, base + 4));
        tensors.push(f32_bias(&format!("{prefix}.ln1.weight"), d_model as usize, base + 5));
        tensors.push(f32_bias(&format!("{prefix}.ln1.bias"), d_model as usize, base + 6));
        tensors.push(f32_bias(&format!("{prefix}.ln2.weight"), d_model as usize, base + 7));
        tensors.push(f32_bias(&format!("{prefix}.ln2.bias"), d_model as usize, base + 8));
        tensors.push(f32_tensor(
            &format!("{prefix}.ffn_up.weight"),
            vec![d_model as u64, d_ff as u64],
            (d_ff * d_model) as usize,
            base + 9,
        ));
        tensors.push(f32_bias(&format!("{prefix}.ffn_up.bias"), d_ff as usize, base + 10));
        tensors.push(f32_tensor(
            &format!("{prefix}.ffn_down.weight"),
            vec![d_ff as u64, d_model as u64],
            (d_model * d_ff) as usize,
            base + 11,
        ));
        tensors.push(f32_bias(&format!("{prefix}.ffn_down.bias"), d_model as usize, base + 12));
    }

    // Merger projector — mm.0 [4d, 4d] square, mm.2 [4d, d_text],
    // real dims order [in, out].
    tensors.push(f32_tensor(
        "mm.0.weight",
        vec![d_merged as u64, d_merged as u64],
        (d_merged * d_merged) as usize,
        seed + 900,
    ));
    tensors.push(f32_bias("mm.0.bias", d_merged as usize, seed + 901));
    tensors.push(f32_tensor(
        "mm.2.weight",
        vec![d_merged as u64, d_text as u64],
        (d_merged * d_text) as usize,
        seed + 902,
    ));
    tensors.push(f32_bias("mm.2.bias", d_text as usize, seed + 903));

    // Assemble file — same layout as the other synth writers.
    let mut prefix = Writer::new();
    prefix.buf.extend_from_slice(GGUF_MAGIC);
    prefix.u32(GGUF_VERSION);
    prefix.u64(tensors.len() as u64);
    prefix.u64(m.n as u64);
    let mut header_kv = m.w.buf;
    prefix.buf.append(&mut header_kv);

    let alignment: u64 = 32;
    let mut info = Writer::new();
    let mut rolling: u64 = 0;
    for t in &tensors {
        info.str_(&t.name);
        info.u32(t.dims.len() as u32);
        for d in &t.dims {
            info.u64(*d);
        }
        info.u32(t.dtype);
        info.u64(rolling);
        let pad = (alignment - (t.bytes.len() as u64 % alignment)) % alignment;
        rolling += t.bytes.len() as u64 + pad;
    }

    let pre_data_len = (prefix.buf.len() + info.buf.len()) as u64;
    let pad_before = (alignment - (pre_data_len % alignment)) % alignment;

    let mut file = std::fs::File::create(path).expect("create");
    file.write_all(&prefix.buf).unwrap();
    file.write_all(&info.buf).unwrap();
    file.write_all(&vec![0u8; pad_before as usize]).unwrap();
    for t in &tensors {
        file.write_all(&t.bytes).unwrap();
        let pad = (alignment - (t.bytes.len() as u64 % alignment)) % alignment;
        if pad > 0 {
            file.write_all(&vec![0u8; pad as usize]).unwrap();
        }
    }
    file.flush().unwrap();
}

/// Assemble + write a GGUF file from built metadata + tensor specs —
/// the same header/alignment/data layout the Llama / BERT / mmproj
/// synths inline. Factored out for the MLA writer.
fn write_gguf_file(path: &Path, m: MetaBuilder, tensors: &[TensorSpec]) {
    let mut prefix = Writer::new();
    prefix.buf.extend_from_slice(GGUF_MAGIC);
    prefix.u32(GGUF_VERSION);
    prefix.u64(tensors.len() as u64);
    prefix.u64(m.n as u64);
    let mut header_kv = m.w.buf;
    prefix.buf.append(&mut header_kv);

    let alignment: u64 = 32;
    let mut info = Writer::new();
    let mut rolling: u64 = 0;
    for t in tensors {
        info.str_(&t.name);
        info.u32(t.dims.len() as u32);
        for d in &t.dims {
            info.u64(*d);
        }
        info.u32(t.dtype);
        info.u64(rolling);
        let pad = (alignment - (t.bytes.len() as u64 % alignment)) % alignment;
        rolling += t.bytes.len() as u64 + pad;
    }

    let pre_data_len = (prefix.buf.len() + info.buf.len()) as u64;
    let pad_before = (alignment - (pre_data_len % alignment)) % alignment;

    let mut file = std::fs::File::create(path).expect("create");
    file.write_all(&prefix.buf).unwrap();
    file.write_all(&info.buf).unwrap();
    file.write_all(&vec![0u8; pad_before as usize]).unwrap();
    for t in tensors {
        file.write_all(&t.bytes).unwrap();
        let pad = (alignment - (t.bytes.len() as u64 % alignment)) % alignment;
        if pad > 0 {
            file.write_all(&vec![0u8; pad as usize]).unwrap();
        }
    }
    file.flush().unwrap();
}

fn random_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((state >> 33) as u32) as f32 / u32::MAX as f32;
        out.push((u - 0.5) * 0.2);
    }
    out
}

