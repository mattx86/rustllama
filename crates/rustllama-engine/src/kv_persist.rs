//! Warm restarts (roadmap Phase 5): persist the prefix-cache pool to
//! a `.rlkv` sidecar next to the model, Colibrì-style, so a restarted
//! engine reopens its conversations warm — the first request whose
//! prompt extends a persisted snapshot skips straight to decode
//! instead of re-prefilling the whole history.
//!
//! Scope (v1): opt-in via `RUSTLLAMA_KV_PERSIST_MB` (promoted from
//! `[inference].kv_persist_mb`); F32 KV only — the hybrid 35B target
//! forces F32, and dense models on quantized KV re-prefill fast
//! enough that serializing four quant layouts isn't worth it yet.
//! Saved on graceful engine drop, loaded at engine start. A `kill -9`
//! loses the session (periodic checkpointing is a follow-up).
//!
//! Like every sidecar here, the file is a perf hint, never a
//! correctness input: a fingerprint (model size + mtime + head hash)
//! plus per-record checksums gate loading, and anything that fails
//! validation is ignored wholesale.

use std::path::{Path, PathBuf};

use rustllama_models::llama_arch::{
    DeltaNetLayerState, DeltaNetSnapshot, KvBuf, KvDtype, KvLayer, KvSnapshot,
};

use crate::expert_cache::{fingerprint, Fingerprint};
use crate::prefix_cache::{fnv1a64, PrefixCachePool, PrefixSnapshot};

/// Sidecar magic + format version (bump on layout change).
const MAGIC: &[u8; 8] = b"RLKV\x01\0\0\0";

/// Where a model's KV-persistence sidecar lives: `<gguf path>.rlkv`.
pub fn rlkv_path(model_path: &Path) -> PathBuf {
    let mut os = model_path.as_os_str().to_os_string();
    os.push(".rlkv");
    PathBuf::from(os)
}

/// `RUSTLLAMA_KV_PERSIST_MB` — cap on the sidecar's serialized size;
/// `0`/unset disables persistence entirely. Read per call (load and
/// teardown only, not hot-path).
pub fn kv_persist_cap_bytes() -> u64 {
    // T3: typed `LoadOptions`-backed accel knob (env-initialized from
    // `RUSTLLAMA_KV_PERSIST_MB` for back-compat).
    rustllama_models::accel::kv_persist_mb().saturating_mul(1024 * 1024)
}

/// `RUSTLLAMA_KV_PERSIST_INTERVAL_SECS` — F12 periodic-checkpoint cadence.
/// `0`/unset disables periodic checkpointing (save-on-drop only, the historical
/// behavior). Read per request boundary (not hot-path).
pub fn kv_persist_interval() -> std::time::Duration {
    std::time::Duration::from_secs(rustllama_models::accel::kv_persist_interval_secs())
}

/// Elapsed-time gate for F12 periodic checkpointing. Pure + deterministic
/// (`Instant` injected) so the "should I checkpoint now?" decision is unit-
/// testable without a real clock. A zero `interval` means the feature is off:
/// [`Self::should_checkpoint`] always returns `false`, keeping save-on-drop as
/// the only write (byte-identical to pre-F12).
#[derive(Debug)]
pub struct CheckpointClock {
    interval: std::time::Duration,
    last: std::time::Instant,
}

impl CheckpointClock {
    /// Build a clock for `interval` (0 = feature off), anchored at `now` — so
    /// the first checkpoint can't fire until a full interval has elapsed since
    /// engine construction.
    pub fn new(interval: std::time::Duration, now: std::time::Instant) -> Self {
        Self { interval, last: now }
    }

    /// `true` iff periodic checkpointing is enabled AND at least `interval` has
    /// elapsed since the last fire; records `now` as the new baseline when it
    /// returns `true` (so the next window measures from this checkpoint).
    pub fn should_checkpoint(&mut self, now: std::time::Instant) -> bool {
        if self.interval.is_zero() {
            return false;
        }
        if now.duration_since(self.last) >= self.interval {
            self.last = now;
            true
        } else {
            false
        }
    }
}

fn push_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn push_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn push_f32s(buf: &mut Vec<u8>, xs: &[f32]) {
    buf.reserve(xs.len() * 4);
    for x in xs {
        buf.extend_from_slice(&x.to_le_bytes());
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    off: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let s = self.bytes.get(self.off..self.off.checked_add(n)?)?;
        self.off += n;
        Some(s)
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f32s(&mut self, n: usize) -> Option<Vec<f32>> {
        let raw = self.take(n.checked_mul(4)?)?;
        Some(
            raw.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        )
    }
}

/// FNV-1a over raw bytes (record checksums).
fn fnv1a64_bytes(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Serialize one snapshot into `buf`. Returns `false` (nothing
/// written) for entries the v1 format can't carry: non-F32 KV, or
/// VLM entries (image state isn't re-validated across restarts).
fn push_snapshot(buf: &mut Vec<u8>, snap: &PrefixSnapshot) -> bool {
    if snap.kv.dtype != KvDtype::F32 || snap.image_hash.is_some() {
        return false;
    }
    let record_start = buf.len();
    push_u32(buf, snap.ids.len() as u32);
    for id in &snap.ids {
        push_u32(buf, *id);
    }
    push_u64(buf, snap.kv.prefix_len as u64);
    push_u32(buf, snap.kv.layers.len() as u32);
    for layer in &snap.kv.layers {
        let KvLayer::F32 { k, v } = layer else {
            // dtype checked F32 above; mixed layers would be a bug.
            buf.truncate(record_start);
            return false;
        };
        push_u64(buf, (**k).len() as u64);
        push_f32s(buf, k);
        push_u64(buf, (**v).len() as u64);
        push_f32s(buf, v);
    }
    match &snap.dn {
        None => push_u32(buf, 0),
        Some(dn) => {
            push_u32(buf, dn.layers.len() as u32);
            for l in &dn.layers {
                push_u64(buf, l.conv_state.len() as u64);
                push_f32s(buf, &l.conv_state);
                push_u64(buf, l.recurrent_state.len() as u64);
                push_f32s(buf, &l.recurrent_state);
            }
        }
    }
    let checksum = fnv1a64_bytes(&buf[record_start..]);
    push_u64(buf, checksum);
    true
}

fn read_snapshot(
    r: &mut Reader<'_>,
    n_kv_heads: usize,
    head_dim: usize,
) -> Option<PrefixSnapshot> {
    let record_start = r.off;
    let ids_len = r.u32()? as usize;
    let mut ids = Vec::with_capacity(ids_len.min(1 << 20));
    for _ in 0..ids_len {
        ids.push(r.u32()?);
    }
    let prefix_len = r.u64()? as usize;
    let n_layers = r.u32()? as usize;
    let mut layers = Vec::with_capacity(n_layers.min(1 << 12));
    for _ in 0..n_layers {
        let k_len = r.u64()? as usize;
        let k = r.f32s(k_len)?;
        let v_len = r.u64()? as usize;
        let v = r.f32s(v_len)?;
        layers.push(KvLayer::F32 {
            k: KvBuf::from_vec(k),
            v: KvBuf::from_vec(v),
        });
    }
    let dn_layers = r.u32()? as usize;
    let dn = if dn_layers == 0 {
        None
    } else {
        let mut ls = Vec::with_capacity(dn_layers.min(1 << 12));
        for _ in 0..dn_layers {
            let conv_len = r.u64()? as usize;
            let conv_state = r.f32s(conv_len)?;
            let rec_len = r.u64()? as usize;
            let recurrent_state = r.f32s(rec_len)?;
            ls.push(DeltaNetLayerState {
                conv_state,
                recurrent_state,
            });
        }
        Some(DeltaNetSnapshot { layers: ls })
    };
    let record_end = r.off;
    let expect = fnv1a64_bytes(&r.bytes[record_start..record_end]);
    let checksum = r.u64()?;
    if checksum != expect {
        return None;
    }
    let hash = fnv1a64(&ids);
    Some(PrefixSnapshot {
        ids,
        hash,
        image_hash: None,
        kv_len: prefix_len,
        kv: KvSnapshot {
            layers,
            prefix_len,
            n_kv_heads,
            head_dim,
            dtype: KvDtype::F32,
        },
        dn,
        last_used: 0,
    })
}

/// Write the pool's snapshots to `<model>.rlkv`, most-recently-used
/// first, stopping once `cap_bytes` would be exceeded. Empty pools
/// (or pools whose every entry is unserializable) leave no file —
/// and remove a stale one, so a cleared session doesn't resurrect.
/// Returns the number of snapshots written.
pub fn save_pool(
    model_path: &Path,
    pool: &PrefixCachePool,
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    cap_bytes: u64,
) -> std::io::Result<usize> {
    let path = rlkv_path(model_path);
    let Some(fp) = fingerprint(model_path) else {
        return Ok(0);
    };
    let mut order: Vec<&PrefixSnapshot> = pool.entries().iter().collect();
    order.sort_by(|a, b| b.last_used.cmp(&a.last_used));

    let mut buf = Vec::new();
    buf.extend_from_slice(MAGIC);
    push_u64(&mut buf, fp.size);
    push_u64(&mut buf, fp.mtime_secs);
    push_u64(&mut buf, fp.head_hash);
    push_u32(&mut buf, n_layers as u32);
    push_u32(&mut buf, n_kv_heads as u32);
    push_u32(&mut buf, head_dim as u32);
    let count_off = buf.len();
    push_u32(&mut buf, 0); // patched below

    let mut written = 0u32;
    for snap in order {
        let before = buf.len();
        if !push_snapshot(&mut buf, snap) {
            continue;
        }
        if buf.len() as u64 > cap_bytes {
            buf.truncate(before);
            break;
        }
        written += 1;
    }
    if written == 0 {
        let _ = std::fs::remove_file(&path);
        return Ok(0);
    }
    buf[count_off..count_off + 4].copy_from_slice(&written.to_le_bytes());

    // Crash-safe-enough for a perf hint: temp sibling then swap
    // (Windows rename won't overwrite → remove first).
    let tmp = path.with_extension("rlkv.tmp");
    {
        use std::io::Write as _;
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(&buf)?;
        f.sync_all()?;
    }
    let _ = std::fs::remove_file(&path);
    std::fs::rename(&tmp, &path)?;
    Ok(written as usize)
}

/// Load `<model>.rlkv` and return its snapshots (most-recently-used
/// first, as saved). `None` when the file is absent, stale (model
/// fingerprint changed), or shaped for a different KV geometry.
/// Individual corrupt records stop the parse at that point; the
/// records before them are still returned.
pub fn load_pool(
    model_path: &Path,
    expect_layers: usize,
    expect_kv_heads: usize,
    expect_head_dim: usize,
) -> Option<Vec<PrefixSnapshot>> {
    let path = rlkv_path(model_path);
    let bytes = std::fs::read(&path).ok()?;
    let fp = fingerprint(model_path)?;
    let mut r = Reader {
        bytes: &bytes,
        off: 0,
    };
    if r.take(8)? != MAGIC {
        return None;
    }
    let file_fp = Fingerprint {
        size: r.u64()?,
        mtime_secs: r.u64()?,
        head_hash: r.u64()?,
    };
    if file_fp != fp {
        tracing::info!(
            path = %path.display(),
            "kv persist sidecar is for a different model file; ignoring"
        );
        return None;
    }
    let n_layers = r.u32()? as usize;
    let n_kv_heads = r.u32()? as usize;
    let head_dim = r.u32()? as usize;
    if n_layers != expect_layers || n_kv_heads != expect_kv_heads || head_dim != expect_head_dim {
        tracing::info!(
            path = %path.display(),
            "kv persist sidecar has a different KV geometry; ignoring"
        );
        return None;
    }
    let n_snapshots = r.u32()? as usize;
    let mut out = Vec::with_capacity(n_snapshots.min(64));
    for _ in 0..n_snapshots {
        match read_snapshot(&mut r, n_kv_heads, head_dim) {
            Some(s) if s.kv.layers.len() == n_layers => out.push(s),
            _ => break, // corrupt tail — keep what parsed clean
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustllama_models::llama_arch::{KvCache, KvDtype};
    use rustllama_models::llama_config::LlamaConfig;

    fn cfg() -> LlamaConfig {
        LlamaConfig {
            arch: "test".into(),
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 2,
            d_model: 8,
            d_ff: 16,
            head_dim: 4,
            rope_dim: 4,
            rope_theta: 10000.0,
            rms_eps: 1e-5,
            vocab_size: 32,
            ctx_train: 16,
            bos_token_id: None,
            eos_token_id: None,
            tie_word_embeddings: false,
            n_mtp_heads: 0,
            moe: None,
            hybrid: None,
            hadamard: None,
            ..Default::default()
        }
    }

    fn tmp_model(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("rustllama-kv-persist-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join(name);
        std::fs::write(&p, format!("model-bytes-{name}")).unwrap();
        p
    }

    fn fill_kv(kv: &mut KvCache, seq_len: usize, base: f32) {
        kv.seq_len = seq_len;
        for layer in kv.layers.iter_mut() {
            if let KvLayer::F32 { k, v } = layer {
                for (i, x) in k.iter_mut().enumerate() {
                    *x = base + i as f32 * 0.01;
                }
                for (i, x) in v.iter_mut().enumerate() {
                    *x = -(base + i as f32 * 0.01);
                }
            }
        }
    }

    #[test]
    fn pool_round_trips_including_dn_state() {
        let model = tmp_model("m1.gguf");
        let c = cfg();
        let mut live = KvCache::new_with_dtype(&c, 16, KvDtype::F32);
        let mut pool = PrefixCachePool::new(4);
        fill_kv(&mut live, 5, 1.0);
        pool.snapshot_and_insert(vec![1, 2, 3, 4, 5], &live);
        // Hand-build an anchor entry (dn present) — 2 layers, one
        // empty sentinel (attn) + one with state (ssm).
        let dn = DeltaNetSnapshot {
            layers: vec![
                DeltaNetLayerState {
                    conv_state: Vec::new(),
                    recurrent_state: Vec::new(),
                },
                DeltaNetLayerState {
                    conv_state: vec![1.5; 6],
                    recurrent_state: vec![-2.5; 8],
                },
            ],
        };
        let mut anchor = pool.entries()[0].clone();
        anchor.ids = vec![9, 9, 9, 9];
        anchor.dn = Some(dn);
        pool.insert_snapshot(anchor);
        assert_eq!(pool.len(), 2);

        let n = save_pool(&model, &pool, 2, 2, 4, 64 * 1024 * 1024).unwrap();
        assert_eq!(n, 2);
        let loaded = load_pool(&model, 2, 2, 4).expect("load back");
        assert_eq!(loaded.len(), 2);
        let anchor_back = loaded
            .iter()
            .find(|s| s.ids == vec![9, 9, 9, 9])
            .expect("anchor entry");
        let dn_back = anchor_back.dn.as_ref().expect("dn survives round trip");
        assert_eq!(dn_back.layers[1].conv_state, vec![1.5; 6]);
        assert_eq!(dn_back.layers[1].recurrent_state, vec![-2.5; 8]);
        let plain = loaded
            .iter()
            .find(|s| s.ids == vec![1, 2, 3, 4, 5])
            .expect("plain entry");
        assert!(plain.dn.is_none());
        assert_eq!(plain.kv.prefix_len, 5);
        let _ = std::fs::remove_file(rlkv_path(&model));
    }

    #[test]
    fn stale_fingerprint_is_rejected() {
        let model = tmp_model("m2.gguf");
        let c = cfg();
        let mut live = KvCache::new_with_dtype(&c, 16, KvDtype::F32);
        let mut pool = PrefixCachePool::new(4);
        fill_kv(&mut live, 3, 1.0);
        pool.snapshot_and_insert(vec![1, 2, 3], &live);
        save_pool(&model, &pool, 2, 2, 4, 64 * 1024 * 1024).unwrap();
        // Change the model file → fingerprint mismatch.
        std::fs::write(&model, b"different-model-bytes!").unwrap();
        assert!(load_pool(&model, 2, 2, 4).is_none());
        let _ = std::fs::remove_file(rlkv_path(&model));
    }

    #[test]
    fn geometry_mismatch_is_rejected() {
        let model = tmp_model("m3.gguf");
        let c = cfg();
        let mut live = KvCache::new_with_dtype(&c, 16, KvDtype::F32);
        let mut pool = PrefixCachePool::new(4);
        fill_kv(&mut live, 3, 1.0);
        pool.snapshot_and_insert(vec![1, 2, 3], &live);
        save_pool(&model, &pool, 2, 2, 4, 64 * 1024 * 1024).unwrap();
        assert!(load_pool(&model, 3, 2, 4).is_none(), "layer count differs");
        assert!(load_pool(&model, 2, 4, 4).is_none(), "kv heads differ");
        let _ = std::fs::remove_file(rlkv_path(&model));
    }

    #[test]
    fn cap_limits_snapshot_count_by_recency() {
        let model = tmp_model("m4.gguf");
        let c = cfg();
        let mut live = KvCache::new_with_dtype(&c, 16, KvDtype::F32);
        let mut pool = PrefixCachePool::new(4);
        fill_kv(&mut live, 8, 1.0);
        pool.snapshot_and_insert(vec![1; 8], &live);
        fill_kv(&mut live, 8, 2.0);
        pool.snapshot_and_insert(vec![2; 8], &live);
        // Cap between one and two records. Record size at this
        // geometry: ids(4+32) + prefix(8) + n_layers(4) + 2 layers ×
        // (k len 8 + 256B + v len 8 + 256B) + dn(4) + checksum(8)
        // ≈ 1116 B; header 48 B → one snapshot ≈ 1164 B total, two
        // ≈ 2280 B.
        let n = save_pool(&model, &pool, 2, 2, 4, 2000).unwrap();
        assert_eq!(n, 1, "cap must limit to one snapshot");
        let loaded = load_pool(&model, 2, 2, 4).expect("one survives");
        assert_eq!(loaded.len(), 1);
        // Most-recently-used first: entry [2;8] was inserted last.
        assert_eq!(loaded[0].ids, vec![2; 8]);
        let _ = std::fs::remove_file(rlkv_path(&model));
    }

    #[test]
    fn checkpoint_clock_off_never_fires() {
        // interval 0 ⇒ feature off ⇒ save-on-drop only (byte-identical to pre-F12).
        let t0 = std::time::Instant::now();
        let mut clk = CheckpointClock::new(std::time::Duration::ZERO, t0);
        assert!(!clk.should_checkpoint(t0));
        assert!(!clk.should_checkpoint(t0 + std::time::Duration::from_secs(3600)));
    }

    #[test]
    fn checkpoint_clock_fires_once_per_interval() {
        let t0 = std::time::Instant::now();
        let secs = std::time::Duration::from_secs;
        let mut clk = CheckpointClock::new(secs(10), t0);
        // Before a full interval: no.
        assert!(!clk.should_checkpoint(t0 + secs(5)));
        // At the interval boundary: fire, and reset the window.
        assert!(clk.should_checkpoint(t0 + secs(10)));
        // Immediately after firing: not yet.
        assert!(!clk.should_checkpoint(t0 + secs(15)));
        // A full interval past the LAST fire (t0+10): fire again.
        assert!(clk.should_checkpoint(t0 + secs(20)));
    }

    #[test]
    fn checkpoint_round_trips_mid_session() {
        // F12 anchor: a checkpoint written "mid-generation" reloads to the EXACT
        // KV state. A checkpoint is just a `save_pool` of the live pool at a
        // request boundary, so saving a partially-grown session + reloading must
        // reproduce the snapshot byte-for-byte.
        let model = tmp_model("m_ckpt.gguf");
        let c = cfg();
        let mut live = KvCache::new_with_dtype(&c, 16, KvDtype::F32);
        let mut pool = PrefixCachePool::new(4);
        fill_kv(&mut live, 7, 3.0);
        pool.snapshot_and_insert(vec![10, 11, 12, 13, 14, 15, 16], &live);
        // "Mid-session checkpoint": write now, before the engine would drop.
        let n = save_pool(&model, &pool, 2, 2, 4, 64 * 1024 * 1024).unwrap();
        assert_eq!(n, 1);
        let loaded = load_pool(&model, 2, 2, 4).expect("reload the checkpoint");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].ids, vec![10, 11, 12, 13, 14, 15, 16]);
        assert_eq!(loaded[0].kv.prefix_len, 7);
        // Exact KV tensors survive (layer 0 K[0] == base 3.0 from fill_kv).
        let KvLayer::F32 { k, .. } = &loaded[0].kv.layers[0] else {
            panic!("expected F32 KV layer");
        };
        assert_eq!(k[0], 3.0, "checkpoint must restore the exact KV state");
        let _ = std::fs::remove_file(rlkv_path(&model));
    }

    #[test]
    fn empty_pool_removes_stale_sidecar() {
        let model = tmp_model("m5.gguf");
        let c = cfg();
        let mut live = KvCache::new_with_dtype(&c, 16, KvDtype::F32);
        let mut pool = PrefixCachePool::new(4);
        fill_kv(&mut live, 3, 1.0);
        pool.snapshot_and_insert(vec![1, 2, 3], &live);
        save_pool(&model, &pool, 2, 2, 4, 64 * 1024 * 1024).unwrap();
        assert!(rlkv_path(&model).exists());
        pool.clear();
        let n = save_pool(&model, &pool, 2, 2, 4, 64 * 1024 * 1024).unwrap();
        assert_eq!(n, 0);
        assert!(
            !rlkv_path(&model).exists(),
            "cleared session must not resurrect on next load"
        );
    }
}
