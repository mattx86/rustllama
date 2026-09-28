//! Importance-matrix (imatrix) load/save.
//!
//! An imatrix maps each weight-tensor name to a per-**input-column**
//! importance vector — the mean squared activation entering that
//! tensor over a calibration corpus. The quantizer weights its
//! reconstruction error by it so high-importance columns are encoded
//! more faithfully (critical for ≤2-bpw formats like IQ1_S).
//!
//! File format `RLIM` v1 (little-endian):
//! ```text
//!   magic    : [u8; 4] = b"RLIM"
//!   version  : u32      = 1
//!   n_tensors: u32
//!   repeated n_tensors times:
//!     name_len : u32
//!     name     : [u8; name_len]   (UTF-8)
//!     n_cols   : u32
//!     values   : [f32; n_cols]
//! ```
//! The consumption side (quantize) broadcasts each `n_cols`-length
//! vector across the tensor's output rows; the column for a flat
//! weight index `f` is `f % n_cols`.

use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::path::Path;

const MAGIC: &[u8; 4] = b"RLIM";
const VERSION: u32 = 1;

/// Per-tensor input-column importance vectors.
#[derive(Debug, Default, Clone)]
pub struct Imatrix {
    map: HashMap<String, Vec<f32>>,
}

impl Imatrix {
    pub fn new() -> Self {
        Self { map: HashMap::new() }
    }

    /// Importance vector for `name`, or `None` if absent. Length is
    /// the tensor's input-column count.
    pub fn get(&self, name: &str) -> Option<&[f32]> {
        self.map.get(name).map(|v| v.as_slice())
    }

    pub fn insert(&mut self, name: impl Into<String>, values: Vec<f32>) {
        self.map.insert(name.into(), values);
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Load an `RLIM` imatrix file.
    pub fn load<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        let mut cur = bytes;
        let mut magic = [0u8; 4];
        cur.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("imatrix: bad magic {magic:?} (expected RLIM)"),
            ));
        }
        let version = read_u32(&mut cur)?;
        if version != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("imatrix: unsupported version {version} (expected {VERSION})"),
            ));
        }
        let n_tensors = read_u32(&mut cur)? as usize;
        // `n_tensors` and `n_cols` are untrusted length prefixes — cap
        // the up-front allocation hints (mirroring `parse.rs`'s
        // `.min(1 << 16)` guard) so a malformed header can't request a
        // multi-GB allocation before any bytes are read. The loops
        // still read the true count and fail cleanly at
        // `read_exact`/`read_f32` if the stream is short.
        let mut map = HashMap::with_capacity(n_tensors.min(1 << 16));
        for _ in 0..n_tensors {
            let name_len = read_u32(&mut cur)? as usize;
            let mut name_bytes = vec![0u8; name_len];
            cur.read_exact(&mut name_bytes)?;
            let name = String::from_utf8(name_bytes).map_err(|e| {
                io::Error::new(io::ErrorKind::InvalidData, format!("imatrix: bad utf8 name: {e}"))
            })?;
            let n_cols = read_u32(&mut cur)? as usize;
            let mut vals = Vec::with_capacity(n_cols.min(1 << 16));
            for _ in 0..n_cols {
                vals.push(read_f32(&mut cur)?);
            }
            map.insert(name, vals);
        }
        Ok(Self { map })
    }

    /// Write an `RLIM` imatrix file. Used by the generation path.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&(self.map.len() as u32).to_le_bytes());
        // Deterministic order (sorted by name) so the file is
        // reproducible across runs.
        let mut names: Vec<&String> = self.map.keys().collect();
        names.sort();
        for name in names {
            let vals = &self.map[name];
            buf.extend_from_slice(&(name.len() as u32).to_le_bytes());
            buf.extend_from_slice(name.as_bytes());
            buf.extend_from_slice(&(vals.len() as u32).to_le_bytes());
            for &v in vals {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        let mut f = std::fs::File::create(path)?;
        f.write_all(&buf)
    }
}

fn read_u32(cur: &mut &[u8]) -> io::Result<u32> {
    let mut b = [0u8; 4];
    cur.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_f32(cur: &mut &[u8]) -> io::Result<f32> {
    let mut b = [0u8; 4];
    cur.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_bytes() {
        let mut im = Imatrix::new();
        im.insert("blk.0.ffn_gate_exps.weight", vec![1.0, 2.5, 0.0, -3.0]);
        im.insert("blk.0.attn_q.weight", vec![0.1; 8]);
        let dir = std::env::temp_dir();
        let path = dir.join("rsl_imatrix_roundtrip_test.rlim");
        im.save(&path).unwrap();
        let loaded = Imatrix::load(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.get("blk.0.ffn_gate_exps.weight"), Some(&[1.0, 2.5, 0.0, -3.0][..]));
        assert_eq!(loaded.get("blk.0.attn_q.weight").unwrap().len(), 8);
        assert_eq!(loaded.get("missing"), None);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn rejects_bad_magic() {
        let bytes = b"XXXX\x01\x00\x00\x00\x00\x00\x00\x00";
        assert!(Imatrix::from_bytes(bytes).is_err());
    }
}
