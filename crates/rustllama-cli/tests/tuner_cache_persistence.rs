//! Verifies that `tune --placement --measure` and `tune --batch-size`
//! winners actually land in the tuner cache file (and round-trip
//! back out via `load_cache`).
//!
//! Uses the lower-level `*_to(cache_dir, key, device, ...)` variants so the
//! tests can write into a tempdir keyed by a synthetic whole-system
//! fingerprint string, without depending on a real SYCL device — covers both
//! mock and SYCL builds. The high-level `persist_*` wrappers are a thin
//! convenience shim over these and don't need their own test.

use rustllama_cli::{
    batch_size_from_cache_or_default_to, persist_batch_size_winner_to, persist_placement_winner_to,
    placement_from_cache_or_default_to,
};
use rustllama_tuner::{load_cache, DeviceFingerprint};

/// The cache FILE key — a whole-system fingerprint string. Each test uses a
/// distinct temp dir, so a shared constant never collides across tests.
const KEY: &str = "synthetic-system-fp";

fn synth_device() -> DeviceFingerprint {
    DeviceFingerprint {
        pci_id: 0xDEAD_BEEF,
        driver_ver: "test.0.0".into(),
        name: "test-device".into(),
        vram_mb: 4096,
    }
}

#[test]
fn placement_winner_persists_to_cache() {
    let dir = std::env::temp_dir().join("rustllama-cli-tune-cache-placement");
    let _ = std::fs::remove_dir_all(&dir);
    let device = synth_device();

    let path = persist_placement_winner_to(&dir, KEY, &device, "qwen2.5-coder-7b-q4km", 16)
        .expect("persist must succeed");
    assert!(
        path.exists(),
        "cache file should exist at {}",
        path.display()
    );

    let loaded = load_cache(&dir, KEY)
        .expect("load_cache must succeed")
        .expect("cache should be populated after persist");
    let entry = loaded
        .placement
        .get("qwen2.5-coder-7b-q4km")
        .expect("placement entry should exist for the model key");
    assert_eq!(
        entry.n_gpu_layers, 16,
        "n_gpu_layers should round-trip exactly"
    );
    assert!(loaded.last_tuned.is_some(), "last_tuned should be set");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn placement_persist_merges_with_existing_entries() {
    let dir = std::env::temp_dir().join("rustllama-cli-tune-cache-placement-merge");
    let _ = std::fs::remove_dir_all(&dir);
    let device = synth_device();

    persist_placement_winner_to(&dir, KEY, &device, "model-a", 8)
        .expect("first persist must succeed");
    persist_placement_winner_to(&dir, KEY, &device, "model-b", 24)
        .expect("second persist must succeed");

    let loaded = load_cache(&dir, KEY).unwrap().unwrap();
    assert_eq!(
        loaded.placement.get("model-a").map(|p| p.n_gpu_layers),
        Some(8),
        "model-a should still be present after model-b's write"
    );
    assert_eq!(
        loaded.placement.get("model-b").map(|p| p.n_gpu_layers),
        Some(24),
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn batch_size_winner_persists_to_cache() {
    let dir = std::env::temp_dir().join("rustllama-cli-tune-cache-batch-size");
    let _ = std::fs::remove_dir_all(&dir);
    let device = synth_device();

    let path =
        persist_batch_size_winner_to(&dir, KEY, &device, 512).expect("persist must succeed");
    assert!(
        path.exists(),
        "cache file should exist at {}",
        path.display()
    );

    let loaded = load_cache(&dir, KEY).unwrap().unwrap();
    assert_eq!(loaded.batch_size, Some(512));
    assert!(loaded.last_tuned.is_some());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn placement_and_batch_size_coexist_in_same_cache_file() {
    let dir = std::env::temp_dir().join("rustllama-cli-tune-cache-coexist");
    let _ = std::fs::remove_dir_all(&dir);
    let device = synth_device();

    persist_placement_winner_to(&dir, KEY, &device, "shared-model", 12).expect("placement write");
    persist_batch_size_winner_to(&dir, KEY, &device, 256).expect("batch_size write");

    let loaded = load_cache(&dir, KEY).unwrap().unwrap();
    assert_eq!(
        loaded.placement.get("shared-model").map(|p| p.n_gpu_layers),
        Some(12),
        "placement entry survives the batch_size write"
    );
    assert_eq!(
        loaded.batch_size,
        Some(256),
        "batch_size entry survives the placement write"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// Persister → consumer round-trip for placement: write a winner,
/// read it back via the consumer helper. The whole point of the
/// loop closure.
#[test]
fn placement_consumer_reads_back_persisted_winner() {
    let dir = std::env::temp_dir().join("rustllama-cli-tune-consume-placement");
    let _ = std::fs::remove_dir_all(&dir);
    let device = synth_device();

    persist_placement_winner_to(&dir, KEY, &device, "qwen-coder-7b", 18).expect("persist");
    let applied = placement_from_cache_or_default_to(&dir, KEY, "qwen-coder-7b", 999);
    assert_eq!(applied, 18, "consumer should return the persisted winner");

    let _ = std::fs::remove_dir_all(&dir);
}

/// Cache miss for an unknown model key → consumer returns the
/// supplied config default. Pins the fresh-install path where the
/// user has never run `tune --placement` and the legacy behavior
/// should kick in unchanged.
#[test]
fn placement_consumer_falls_back_to_default_on_cache_miss() {
    let dir = std::env::temp_dir().join("rustllama-cli-tune-consume-miss");
    let _ = std::fs::remove_dir_all(&dir);
    let device = synth_device();

    persist_placement_winner_to(&dir, KEY, &device, "some-other-model", 12)
        .expect("persist a different model so the cache file exists");
    let applied = placement_from_cache_or_default_to(&dir, KEY, "unrelated-model", 999);
    assert_eq!(
        applied, 999,
        "unknown model key → config default wins, not someone else's entry"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// No cache file at all → consumer returns the supplied default
/// without erroring. Covers the very-first-launch path.
#[test]
fn placement_consumer_returns_default_when_cache_absent() {
    let dir = std::env::temp_dir().join("rustllama-cli-tune-consume-no-cache");
    let _ = std::fs::remove_dir_all(&dir);

    let applied = placement_from_cache_or_default_to(&dir, KEY, "any-model", 999);
    assert_eq!(applied, 999, "absent cache → default wins");
}

/// Batch-size consumer mirrors placement: writes round-trip and
/// reads back via the consumer.
#[test]
fn batch_size_consumer_reads_back_persisted_winner() {
    let dir = std::env::temp_dir().join("rustllama-cli-tune-consume-batch");
    let _ = std::fs::remove_dir_all(&dir);
    let device = synth_device();

    persist_batch_size_winner_to(&dir, KEY, &device, 1024).expect("persist");
    let applied = batch_size_from_cache_or_default_to(&dir, KEY, 512);
    assert_eq!(applied, 1024);

    let _ = std::fs::remove_dir_all(&dir);
}

/// Batch-size cache miss → returns default. Note: a cache file
/// containing only placement entries still counts as "miss" for
/// batch_size since `TuningResult.batch_size` is `None`.
#[test]
fn batch_size_consumer_falls_back_when_field_unset() {
    let dir = std::env::temp_dir().join("rustllama-cli-tune-consume-batch-miss");
    let _ = std::fs::remove_dir_all(&dir);
    let device = synth_device();

    // Write a placement entry only — batch_size stays None on disk.
    persist_placement_winner_to(&dir, KEY, &device, "some-model", 8).expect("persist");
    let applied = batch_size_from_cache_or_default_to(&dir, KEY, 512);
    assert_eq!(
        applied, 512,
        "placement-only cache leaves batch_size None → default wins"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// A subsequent persist for the same model_key overwrites the
/// previous entry — the user re-running `tune --placement` after
/// hardware / model changes expects the cache to reflect the
/// latest winner, not append history.
#[test]
fn placement_persist_overwrites_same_model_key() {
    let dir = std::env::temp_dir().join("rustllama-cli-tune-cache-overwrite");
    let _ = std::fs::remove_dir_all(&dir);
    let device = synth_device();

    persist_placement_winner_to(&dir, KEY, &device, "model-x", 4).expect("first persist");
    persist_placement_winner_to(&dir, KEY, &device, "model-x", 28).expect("second persist");

    let loaded = load_cache(&dir, KEY).unwrap().unwrap();
    assert_eq!(
        loaded.placement.get("model-x").map(|p| p.n_gpu_layers),
        Some(28),
        "the more recent write should win"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
