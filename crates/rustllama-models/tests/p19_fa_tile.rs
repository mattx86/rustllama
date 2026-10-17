//! P19: the adaptive flash-attn-v3 KV_TILE per-kv_len-bucket selection.
//!
//! Verifies the dispatch-side contract that keeps the DEFAULT byte-identical to
//! pre-P19 (no tuned winner ⇒ `fa_v3_tile_for` returns 0 ⇒ the kernel uses its
//! built-in env+kv_len heuristic) and that a banked per-bucket winner routes by
//! `kv_len`. Pure global-atomic logic — no device needed.

use rustllama_models::accel::{fa_v3_tile_for, set_flash_v3_kv_tiles};

#[test]
fn fa_v3_tile_default_is_zero_and_buckets_route_by_kv_len() {
    // Default / reset: every bucket 0 ⇒ the kernel keeps its built-in heuristic
    // (byte-identical to pre-P19).
    set_flash_v3_kv_tiles(0, 0, 0);
    assert_eq!(fa_v3_tile_for(100), 0);
    assert_eq!(fa_v3_tile_for(2000), 0);
    assert_eq!(fa_v3_tile_for(9000), 0);

    // A banked per-bucket winner routes by kv_len (crossover points <512 /
    // 512..4096 / >=4096, matching the kernel's own tiling heuristic).
    set_flash_v3_kv_tiles(16, 32, 64);
    assert_eq!(fa_v3_tile_for(0), 16);
    assert_eq!(fa_v3_tile_for(100), 16); // short
    assert_eq!(fa_v3_tile_for(511), 16);
    assert_eq!(fa_v3_tile_for(512), 32); // mid
    assert_eq!(fa_v3_tile_for(2000), 32);
    assert_eq!(fa_v3_tile_for(4095), 32);
    assert_eq!(fa_v3_tile_for(4096), 64); // long
    assert_eq!(fa_v3_tile_for(9000), 64);

    // Out-of-set values (not 16/32/64) are rejected → 0 (heuristic), so a bad
    // cache entry can never feed the kernel an unsupported tile.
    set_flash_v3_kv_tiles(17, 0, 100);
    assert_eq!(fa_v3_tile_for(100), 0);
    assert_eq!(fa_v3_tile_for(2000), 0);
    assert_eq!(fa_v3_tile_for(9000), 0);

    // Reset so we don't leak state into any other test in this binary.
    set_flash_v3_kv_tiles(0, 0, 0);
}
