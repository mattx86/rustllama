use super::*;

#[test]
fn spill_gguf_q8_0_roundtrip_within_one_quant_step() {
    // 64 elems = 2 full Q8_0 blocks (block-32). The GGUF-native codec
    // encodes to real Q8_0 bytes and reconstructs through the shared
    // `rustllama_gguf::dequant` path (the matvec reference).
    let vals: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
    let enc = spill_encode_gguf(&vals, Dtype::Q8_0Raw).expect("q8_0 encode");
    assert_eq!(enc.len(), (vals.len() / 32) * 34, "Q8_0 = 34 B / 32-block");
    let dec = dtype_dequant_to_f32(Dtype::Q8_0Raw, &enc, vals.len()).expect("q8_0 dequant");
    assert_eq!(dec.len(), vals.len());
    // Per-block symmetric Q8_0 error is bounded by one step (amax / 127).
    for blk in 0..(vals.len() / 32) {
        let s = blk * 32;
        let e = s + 32;
        let amax = vals[s..e].iter().fold(0f32, |m, &v| m.max(v.abs()));
        let step = amax / 127.0;
        for i in s..e {
            assert!(
                (dec[i] - vals[i]).abs() <= step + 1e-4,
                "i={i}: got {} want {} (step {step})",
                dec[i],
                vals[i]
            );
        }
    }
}

#[test]
fn spill_store_dtype_rule_is_near_lossless() {
    // High-precision sources re-encode to Q8_0 (block-aligned); already
    // block-quantized sources spill native (None); odd geometry → native.
    assert_eq!(spill_store_dtype_for(Dtype::F16, 256), Some(Dtype::Q8_0Raw));
    assert_eq!(spill_store_dtype_for(Dtype::F32, 64), Some(Dtype::Q8_0Raw));
    assert_eq!(spill_store_dtype_for(Dtype::Q4_KRaw, 256), None);
    assert_eq!(spill_store_dtype_for(Dtype::Q8_0Raw, 256), None);
    assert_eq!(spill_store_dtype_for(Dtype::F16, 70), None); // not a 32-multiple
}

#[test]
fn spill_encode_gguf_rejects_non_block_geometry() {
    let vals: Vec<f32> = (0..70).map(|i| i as f32 * 0.01).collect();
    assert!(spill_encode_gguf(&vals, Dtype::Q8_0Raw).is_none());
    assert!(spill_encode_gguf(&[], Dtype::Q8_0Raw).is_none());
}

#[test]
fn spill_dequant_f32_is_identity() {
    let vals = [1.5f32, -2.25, 0.0, 100.0, -0.5];
    let mut bytes = Vec::new();
    for v in &vals {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let out = dtype_dequant_to_f32(Dtype::F32, &bytes, vals.len()).unwrap();
    assert_eq!(out, vals.to_vec());
}

#[test]
fn spill_dequant_skips_exotic_dtype() {
    // Exotic codebook formats must fall through to native spill (None).
    assert!(dtype_dequant_to_f32(Dtype::IQ1_SRaw, &[0u8; 50], 256).is_none());
    assert!(dtype_dequant_to_f32(Dtype::PTQ1_0Raw, &[0u8; 28], 128).is_none());
}

#[test]
fn try_rmsnorm_with_no_stream_returns_false() {
    // Without an installed stream the function MUST return false
    // so the caller's CPU fallback fires. Pinning this so a
    // future change that accidentally panics on "no stream"
    // (e.g. unwrap on the TLS slot) gets caught.
    let x = vec![1.0f32; 4096];
    let w = vec![1.0f32; 4096];
    let mut y = vec![0f32; 4096];
    assert!(!try_rmsnorm_f32(&x, &w, &mut y, 4096, 1e-5));
    // `y` should be untouched.
    assert_eq!(y, vec![0f32; 4096]);
}

#[test]
fn flash_attention_flag_round_trips_through_tls() {
    // Engine calls `set_flash_attention(cfg.inference.flash_attention)`
    // at the top of each generate, and `forward_one` reads via
    // `flash_attention_enabled()`. Pin the round-trip so a future
    // change that breaks the TLS doesn't silently route attention
    // through the wrong path.
    set_flash_attention(true);
    assert!(flash_attention_enabled());
    set_flash_attention(false);
    assert!(!flash_attention_enabled());
    // Restore default so subsequent tests see a clean slate.
    set_flash_attention(true);
}

#[test]
fn rope_inv_freq_table_matches_neox_convention() {
    // The SYCL rope kernel expects a pre-computed inv_freq table;
    // the CPU rope reads `rope_theta` directly. Pin that the
    // table builder produces values matching the formula the CPU
    // rope uses internally so the two paths are equivalent.
    let head_dim = 64;
    let rope_theta = 10000.0f32;
    let table = rope_inv_freq_table(head_dim, rope_theta);
    assert_eq!(table.len(), head_dim / 2);
    // First entry (j=0) is 1 / theta^0 = 1.0.
    assert!((table[0] - 1.0).abs() < 1e-6);
    // Last entry (j = half-1 = 31) is 1 / theta^(62/64) ≈ 1.65e-4.
    let expected_last = 1.0f32 / rope_theta.powf(62.0 / 64.0);
    assert!((table[31] - expected_last).abs() < 1e-6);
}

#[test]
fn rope_scaled_table_is_identity_without_extension() {
    use crate::llama_config::{RopeScaling, RopeScalingKind};
    let head_dim = 64;
    let theta = 10000.0f32;
    let base = rope_inv_freq_table(head_dim, theta);
    // `None` scaling returns the base table verbatim.
    assert_eq!(rope_inv_freq_table_scaled(head_dim, theta, None), base);
    // Every scheme at factor == 1.0 (no extension) is BIT-identical to the
    // base table — this is the guard that a rope-scaled model collapses to
    // the exact scalar-theta path when the factor is a no-op, so coherence
    // on non-extended models can't drift.
    for kind in [
        RopeScalingKind::None,
        RopeScalingKind::Linear,
        RopeScalingKind::Ntk,
        RopeScalingKind::Yarn,
        RopeScalingKind::LongRope,
        RopeScalingKind::Llama3,
    ] {
        let s = RopeScaling {
            kind,
            factor: 1.0,
            orig_ctx: 4096,
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
        };
        let scaled = rope_inv_freq_table_scaled(head_dim, theta, Some(&s));
        assert_eq!(scaled.len(), base.len(), "{kind:?} len");
        assert_eq!(scaled, base, "{kind:?} must equal base at factor 1.0");
        // mscale is a no-op at factor 1.0 for every scheme.
        assert_eq!(rope_yarn_mscale(Some(&s)), 1.0, "{kind:?} mscale@1");
    }
}

#[test]
fn rope_scaled_table_extends_at_factor_gt_one() {
    use crate::llama_config::{RopeScaling, RopeScalingKind};
    let head_dim = 64;
    let theta = 10000.0f32;
    let base = rope_inv_freq_table(head_dim, theta);
    let mk = |kind| RopeScaling {
        kind,
        factor: 4.0,
        orig_ctx: 4096,
        attn_factor: 1.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
        low_freq_factor: 1.0,
        high_freq_factor: 4.0,
    };
    // Linear: every frequency divided by the factor (slower rotation).
    let t = rope_inv_freq_table_scaled(head_dim, theta, Some(&mk(RopeScalingKind::Linear)));
    for (&a, &b) in t.iter().zip(base.iter()) {
        assert!((a - b / 4.0).abs() < 1e-9, "linear = base/factor");
    }
    // NTK rescales the base theta up, so every freq shrinks vs base (but by
    // less than the full /factor of linear, and strictly nonzero).
    let n = rope_inv_freq_table_scaled(head_dim, theta, Some(&mk(RopeScalingKind::Ntk)));
    assert_eq!(n.len(), base.len());
    assert!(
        (n[0] - base[0]).abs() < 1e-6,
        "ntk leaves the j=0 freq ≈ unchanged"
    );
    assert!(n.iter().all(|&v| v > 0.0));
    // YaRN: per-dim freqs bounded between full interpolation (base/factor)
    // and full extrapolation (base); mscale grows as (1 + 0.1·ln factor).
    let yarn = mk(RopeScalingKind::Yarn);
    let yt = rope_inv_freq_table_scaled(head_dim, theta, Some(&yarn));
    for (&v, &b) in yt.iter().zip(base.iter()) {
        assert!(
            v >= b / 4.0 - 1e-6 && v <= b + 1e-6,
            "yarn freq {v} in band"
        );
    }
    let m = rope_yarn_mscale(Some(&yarn));
    assert!(
        (m - (1.0 + 0.1 * 4.0f32.ln())).abs() < 1e-6 && m > 1.0,
        "yarn mscale"
    );
}

#[test]
fn try_rmsnorm_rejects_bad_shape() {
    let x = vec![1.0f32; 100];
    let w = vec![1.0f32; 32];
    let mut y = vec![0f32; 100];
    // d=32 doesn't divide x.len()=100 → false without touching y.
    assert!(!try_rmsnorm_f32(&x, &w, &mut y, 32, 1e-5));
    // d=0 → false.
    assert!(!try_rmsnorm_f32(&x, &w, &mut y, 0, 1e-5));
    // w shorter than d → false.
    let w_short = vec![1.0f32; 16];
    assert!(!try_rmsnorm_f32(&x, &w_short, &mut y, 32, 1e-5));
}

#[test]
fn packed_weights_preloaded_flag_round_trips() {
    // Per-thread "weights preloaded" gate. Engine sets this once
    // after the first preload; subsequent generations on the same
    // thread skip the (idempotent but log-noisy) re-preload.
    // This test runs on a fresh thread; the flag MUST start false.
    std::thread::spawn(|| {
        assert!(!packed_weights_preloaded(), "fresh thread starts unset");
        mark_packed_weights_preloaded();
        assert!(packed_weights_preloaded(), "set after mark");
    })
    .join()
    .unwrap();
}

#[test]
fn parse_q4k_shape_bucket_round_trips() {
    // The bucket string format is the contract between tuner +
    // engine; pin it here so a typo on either side gets caught.
    assert_eq!(parse_q4k_shape_bucket("M=1536,K=1536"), Some((1536, 1536)));
    assert_eq!(parse_q4k_shape_bucket("M=8960,K=1536"), Some((8960, 1536)));
    assert_eq!(parse_q4k_shape_bucket("M=0,K=0"), Some((0, 0)));
}

#[test]
fn parse_q4k_shape_bucket_rejects_garbage() {
    // Hand-edited cache shouldn't be able to silently load as
    // (0, 0) — every misparse must yield None so the cache loader
    // skips the entry with a warning.
    assert_eq!(parse_q4k_shape_bucket(""), None);
    assert_eq!(parse_q4k_shape_bucket("M=1536"), None); // no comma
    assert_eq!(parse_q4k_shape_bucket("M=abc,K=1536"), None); // non-int M
    assert_eq!(parse_q4k_shape_bucket("X=1536,Y=1536"), None); // wrong prefix
    assert_eq!(parse_q4k_shape_bucket("1536,1536"), None); // missing prefixes
}

#[test]
fn tuned_lws_returns_zero_with_empty_cache() {
    // The OnceLock for the packed-USM LWS cache initializes on
    // first call. In mock mode (no SYCL device fingerprint) the
    // init returns an empty map, so every shape lookup yields 0
    // (the kernel's hand-picked default).
    // Note: this test shares process-global state with other
    // tuned_lws_for callers; if a future test populates the
    // cache, this will start failing — split into a separate
    // test binary if that happens.
    assert_eq!(
        tuned_lws_for(rustllama_tuner::KERNEL_Q4K_PACKED_USM, 1536, 1536),
        0
    );
    assert_eq!(
        tuned_lws_for(rustllama_tuner::KERNEL_Q4K_PACKED_USM, 8960, 1536),
        0
    );
}

/// Hybrid-placement TLS round-trip on a fresh thread. The
/// setter / getter pair must read back the same value, and
/// `gpu_active_for_current_layer` must compute the comparison
/// against the current layer index correctly.
#[test]
fn n_gpu_layers_tls_round_trips_per_thread() {
    std::thread::spawn(|| {
        // Fresh thread starts with defaults — n_gpu_layers ==
        // u32::MAX, current_layer_idx == 0, so the GPU path
        // is "active" by default.
        assert_eq!(n_gpu_layers(), u32::MAX);
        assert_eq!(current_layer_idx(), 0);
        assert!(
            gpu_active_for_current_layer(),
            "default = all GPU; layer 0 must be active"
        );

        // Engine pushes the config value before each generate.
        set_n_gpu_layers(4);
        assert_eq!(n_gpu_layers(), 4);
        assert!(gpu_active_for_current_layer(), "layer 0 < cutoff 4");

        // Model advances per-layer.
        for i in 0..4u32 {
            set_current_layer_idx(i);
            assert!(
                gpu_active_for_current_layer(),
                "layer {i} < cutoff 4 must be GPU"
            );
        }
        for i in 4..8u32 {
            set_current_layer_idx(i);
            assert!(
                !gpu_active_for_current_layer(),
                "layer {i} >= cutoff 4 must be CPU"
            );
        }

        // n_gpu_layers = 0 means "all CPU."
        set_n_gpu_layers(0);
        set_current_layer_idx(0);
        assert!(
            !gpu_active_for_current_layer(),
            "n_gpu_layers=0 => layer 0 also CPU"
        );
    })
    .join()
    .unwrap();
}

/// Each thread has independent placement TLS. A setter on
/// thread A doesn't leak to thread B. Regression: a
/// stray `static` instead of `thread_local!` would surface
/// here as cross-thread bleed.
#[test]
fn n_gpu_layers_tls_is_per_thread_not_shared() {
    // Thread A: cutoff = 2.
    let a = std::thread::spawn(|| {
        set_n_gpu_layers(2);
        // Yield a few times so thread B has a chance to run
        // before we read back; if the TLS were shared, B's
        // write could clobber A's value.
        for _ in 0..10 {
            std::thread::yield_now();
        }
        n_gpu_layers()
    });
    // Thread B: cutoff = 999.
    let b = std::thread::spawn(|| {
        set_n_gpu_layers(999);
        for _ in 0..10 {
            std::thread::yield_now();
        }
        n_gpu_layers()
    });
    assert_eq!(
        a.join().unwrap(),
        2,
        "thread A's TLS must not see B's write"
    );
    assert_eq!(
        b.join().unwrap(),
        999,
        "thread B's TLS must not see A's write"
    );
}

/// `tensor_forced_to_cpu` is substring-based: `"ffn"` matches
/// `"blk.5.ffn_gate.weight"` and any other name containing
/// the substring. Multiple patterns are OR'd. Empty list
/// matches nothing. Empty-string patterns are skipped (they'd
/// otherwise match every tensor — almost certainly a config
/// typo).
#[test]
fn tensor_forced_to_cpu_substring_matching() {
    std::thread::spawn(|| {
        // No patterns set yet — nothing matches.
        set_cpu_force_patterns(vec![]);
        assert!(!tensor_forced_to_cpu("blk.5.ffn_gate.weight"));
        assert!(!tensor_forced_to_cpu("token_embd.weight"));

        // Single pattern.
        set_cpu_force_patterns(vec!["ffn".to_string()]);
        assert!(tensor_forced_to_cpu("blk.5.ffn_gate.weight"));
        assert!(tensor_forced_to_cpu("blk.0.ffn_up.weight"));
        assert!(tensor_forced_to_cpu("ffn"));
        assert!(!tensor_forced_to_cpu("blk.5.attn_q.weight"));
        assert!(!tensor_forced_to_cpu("token_embd.weight"));

        // Multiple patterns — any match triggers CPU.
        set_cpu_force_patterns(vec!["ffn".to_string(), "attn_v".to_string()]);
        assert!(tensor_forced_to_cpu("blk.5.ffn_gate.weight"));
        assert!(tensor_forced_to_cpu("blk.10.attn_v.weight"));
        assert!(!tensor_forced_to_cpu("blk.5.attn_q.weight"));

        // Empty-string pattern is skipped (would otherwise
        // match everything, which is almost always a bug).
        set_cpu_force_patterns(vec!["".to_string(), "attn_v".to_string()]);
        assert!(!tensor_forced_to_cpu("blk.5.ffn_gate.weight"));
        assert!(tensor_forced_to_cpu("blk.5.attn_v.weight"));

        // Reset for sibling tests.
        set_cpu_force_patterns(vec![]);
    })
    .join()
    .unwrap();
}

/// CPU-force pattern list is per-thread. A setter on one
/// thread doesn't leak to another — required so concurrent
/// requests with different configs don't interfere.
#[test]
fn cpu_force_patterns_are_per_thread() {
    let a = std::thread::spawn(|| {
        set_cpu_force_patterns(vec!["ffn".to_string()]);
        for _ in 0..10 {
            std::thread::yield_now();
        }
        tensor_forced_to_cpu("blk.0.ffn_gate.weight")
    });
    let b = std::thread::spawn(|| {
        set_cpu_force_patterns(vec!["attn_q".to_string()]);
        for _ in 0..10 {
            std::thread::yield_now();
        }
        // Thread B's TLS has `attn_q`, not `ffn` — the ffn
        // name should NOT match if isolation works.
        tensor_forced_to_cpu("blk.0.ffn_gate.weight")
    });
    assert!(a.join().unwrap(), "thread A pinned ffn → must match");
    assert!(
        !b.join().unwrap(),
        "thread B's pattern is attn_q — ffn must not leak from A"
    );
}

/// `try_*_usm_f32` helpers short-circuit when
/// `gpu_active_for_current_layer()` is false, even when
/// `usm_attn_enabled()` is true. Mock mode reaches the
/// gating check before any SYCL call, so this exercises the
/// gate independently of GPU presence.
#[test]
fn try_usm_helpers_short_circuit_when_layer_past_cutoff() {
    std::thread::spawn(|| {
        // Force "GPU off for this layer."
        set_n_gpu_layers(2);
        set_current_layer_idx(5);
        assert!(!gpu_active_for_current_layer());

        // try_silu_mul_usm_f32 takes the simplest input shape
        // — exercise it directly. In mock mode it would
        // normally fall through to the rest of the body and
        // return false at the SYCL boundary; with the
        // placement gate it returns false IMMEDIATELY,
        // skipping the body entirely. Either way the caller
        // sees `false`, which is the contract.
        let g = vec![0.5f32; 16];
        let u = vec![0.3f32; 16];
        let mut out = vec![0f32; 16];
        assert!(
            !try_silu_mul_usm_f32(&g, &u, &mut out),
            "placement-gated silu_mul must return false on a CPU layer"
        );

        // Restore so subsequent tests on this thread aren't
        // affected — TLS persists for the thread's lifetime.
        set_n_gpu_layers(u32::MAX);
        set_current_layer_idx(0);
    })
    .join()
    .unwrap();
}
