//! P3 cascade (shared-prefix) decode parity.
//!
//! The cascade kernel's `cascade == per-sequence` equivalence is already pinned
//! at the kernel layer (`rustllama_kernels_cpu::cascade`). This test pins the
//! MODEL-LAYER wiring: `forward_decode_paged_batched_cascade_f32` with a
//! `CascadeGroup` (shared-prefix attention computed once + merged per slot) is
//! numerically equivalent — within the online-softmax merge's ~1e-6 tolerance —
//! to the plain per-slot `forward_decode_paged_batched_f32`, for 2 slots that
//! SHARE a page-aligned prefix (pinned shared pages) but have different
//! suffixes.

use rustllama_gguf::synth::{write_synthetic_llama_gguf, SynthLlama};
use rustllama_gguf::Gguf;
use rustllama_models::llama_arch::{CascadeGroup, DecodeSlot, LlamaModel};
use rustllama_models::llama_config::LlamaConfig;
use rustllama_models::page_table::PageTable;
use rustllama_models::paged_kv_cache::PagedKvCache;
use rustllama_models::paged_kv_store::PagedKvStore;
use rustllama_models::shared_paged_kv::SharedPagedKv;

fn synth(tag: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("rustllama-cascade-{tag}.gguf"))
}

#[test]
fn cascade_decode_matches_per_slot_within_tolerance() {
    let path = synth("parity");
    write_synthetic_llama_gguf(&path, &SynthLlama::default());
    let gguf = Gguf::open(&path).expect("open synth");
    let cfg = LlamaConfig::from_gguf(&gguf).expect("config");
    let model = LlamaModel::load(&gguf).expect("load");

    let page_size = 4u32;
    let prefix: Vec<i32> = vec![5, 1, 4, 2]; // 4 tokens == one full page (page-aligned)
    let prefix_len = prefix.len();
    let suffix_a: Vec<i32> = vec![3, 6];
    let suffix_b: Vec<i32> = vec![7, 0];
    let kv_before = prefix_len + suffix_a.len(); // same suffix length for both
    let decode_pos = kv_before as u32; // decode writes here
    let total_capacity = (kv_before + 1) as u32;
    let pages_per = total_capacity.div_ceil(page_size);
    let total_pages = pages_per * 3; // donor + 2 slots

    let store = PagedKvStore::new(
        total_pages,
        cfg.n_layers as u32,
        cfg.n_kv_heads as u32,
        page_size,
        cfg.head_dim as u32,
    )
    .expect("store");
    let table = PageTable::new(total_pages, page_size);
    let shared = SharedPagedKv::new(store, table);
    let geom = PagedKvStore::new(
        total_pages,
        cfg.n_layers as u32,
        cfg.n_kv_heads as u32,
        page_size,
        cfg.head_dim as u32,
    )
    .expect("geom");

    // Donor: prefill the shared prefix once; its page(s) are shared by both slots.
    let mut donor = PagedKvCache::new_for(&geom);
    donor
        .ensure_capacity_shared(&shared, prefix_len as u32)
        .expect("donor cap");
    shared.with_store_mut(|st| {
        for (i, &t) in prefix.iter().enumerate() {
            let mut tmp = vec![0f32; cfg.vocab_size];
            model.forward_one_paged_f32(t, i as u32, &mut donor, st, &mut tmp);
        }
    });
    let shared_pages: Vec<_> = donor.pages()[..(prefix_len / page_size as usize)].to_vec();

    // Build two slots that SHARE the prefix pages, then prefill distinct suffixes.
    let build_slot = |suffix: &[i32]| -> PagedKvCache {
        let mut c = PagedKvCache::new_for(&geom);
        shared.with_table_mut(|t| t.add_ref(&shared_pages));
        c.pin_shared_prefix(shared_pages.clone(), prefix_len as u32);
        c.ensure_capacity_shared(&shared, total_capacity)
            .expect("cap");
        shared.with_store_mut(|st| {
            for (j, &t) in suffix.iter().enumerate() {
                let pos = prefix_len + j;
                let mut tmp = vec![0f32; cfg.vocab_size];
                model.forward_one_paged_f32(t, pos as u32, &mut c, st, &mut tmp);
            }
        });
        c
    };
    let mut cache_a = build_slot(&suffix_a);
    let mut cache_b = build_slot(&suffix_b);

    let last_a = *suffix_a.last().unwrap();
    let last_b = *suffix_b.last().unwrap();

    // Reference: plain per-slot batched decode (no cascade). Writes `decode_pos`.
    let mut ref_a = vec![0f32; cfg.vocab_size];
    let mut ref_b = vec![0f32; cfg.vocab_size];
    {
        let mut slots = vec![
            DecodeSlot {
                token_id: last_a,
                pos: decode_pos,
                cache: &mut cache_a,
                logits_out: &mut ref_a,
            },
            DecodeSlot {
                token_id: last_b,
                pos: decode_pos,
                cache: &mut cache_b,
                logits_out: &mut ref_b,
            },
        ];
        model.forward_decode_paged_batched_f32(&mut slots, &shared);
    }

    // Cascade: same slots, same pos (the re-write of `decode_pos` is idempotent —
    // identical K/V — so the only difference under test is the attention path).
    let group = vec![CascadeGroup {
        prefix_len,
        members: vec![0, 1],
    }];
    let mut cas_a = vec![0f32; cfg.vocab_size];
    let mut cas_b = vec![0f32; cfg.vocab_size];
    {
        let mut slots = vec![
            DecodeSlot {
                token_id: last_a,
                pos: decode_pos,
                cache: &mut cache_a,
                logits_out: &mut cas_a,
            },
            DecodeSlot {
                token_id: last_b,
                pos: decode_pos,
                cache: &mut cache_b,
                logits_out: &mut cas_b,
            },
        ];
        model.forward_decode_paged_batched_cascade_f32(&mut slots, &shared, Some(&group));
    }

    let mut max_err = 0f32;
    for (r, c) in ref_a.iter().zip(cas_a.iter()) {
        max_err = max_err.max((r - c).abs());
    }
    for (r, c) in ref_b.iter().zip(cas_b.iter()) {
        max_err = max_err.max((r - c).abs());
    }
    assert!(
        max_err < 1e-3,
        "cascade decode diverged from per-slot decode: max logit err {max_err} >= 1e-3"
    );
    // Sanity: the two slots produced DIFFERENT logits (so the test isn't
    // vacuously comparing identical rows — the distinct suffixes matter).
    let slot_diff = ref_a
        .iter()
        .zip(ref_b.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        slot_diff > 1e-6,
        "the two slots should differ (distinct suffixes); got identical logits"
    );
    let _ = std::fs::remove_file(&path);
}
