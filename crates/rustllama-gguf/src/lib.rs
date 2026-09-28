//! GGUF v3 reader.
//!
//! Parses the header, metadata key-value section, and tensor info table of a
//! GGUF file, then mmap-backs the tensor data so callers can borrow slices
//! without copying. Quantization decoders live in [`dequant`].

#![forbid(unsafe_op_in_unsafe_fn)]

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use memmap2::{Mmap, MmapOptions};

// Encode-side modules sit behind the default-on `encoder` feature —
// the slim server build (`--no-default-features`) drops them. The
// decode side (dequant/parse/iq1_grid/imatrix) stays unconditional:
// loading models always needs it, and the shared IQ grids in
// `iq1_grid` are read by `dequant` too.
#[cfg(feature = "encoder")]
pub mod apex;
pub mod dequant;
#[cfg(feature = "encoder")]
pub mod encode;
#[cfg(feature = "encoder")]
pub mod encode_iq;
#[cfg(feature = "encoder")]
pub mod encode_iq_vec;
#[cfg(feature = "encoder")]
pub mod encode_k;
#[cfg(feature = "encoder")]
pub mod iq_gpu;
#[cfg(feature = "encoder")]
pub mod encode_t;
pub mod imatrix;
pub mod iq1_grid;
mod parse;
#[cfg(feature = "encoder")]
pub mod quantize;
#[cfg(feature = "encoder")]
pub mod recipe;
#[cfg(feature = "synth")]
pub mod synth;
#[cfg(feature = "encoder")]
pub mod write;

pub use parse::{GgmlType, MetadataValue};

const GGUF_MAGIC: [u8; 4] = *b"GGUF";
const GGUF_DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Debug, thiserror::Error)]
pub enum GgufError {
    #[error("io error reading {path:?}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("unexpected eof at offset {offset} (need {need} bytes)")]
    UnexpectedEof { offset: usize, need: usize },
    #[error("bad magic: expected b\"GGUF\", got {0:?}")]
    BadMagic([u8; 4]),
    #[error("unsupported gguf version {0}; rustllama supports v3")]
    UnsupportedVersion(u32),
    #[error("unknown metadata value type id {0}")]
    UnknownMetadataType(u32),
    #[error("unknown ggml type id {0}")]
    UnknownGgmlType(u32),
    #[error("invalid general.alignment {0}: must be a non-zero power of two")]
    InvalidAlignment(u64),
    #[error("string at offset {offset} is not valid utf-8: {source}")]
    BadUtf8 {
        offset: usize,
        #[source]
        source: std::str::Utf8Error,
    },
    #[error("tensor data for {name} would extend past end of file (offset {offset}, size {size}, file {file_size})")]
    TensorOutOfBounds {
        name: String,
        offset: u64,
        size: u64,
        file_size: u64,
    },
}

pub type Result<T> = std::result::Result<T, GgufError>;

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub dtype: GgmlType,
    /// Offset of this tensor's data, relative to the start of the tensor-data
    /// region (i.e. after metadata + tensor-info table + alignment padding).
    pub rel_offset: u64,
    /// Absolute byte offset in the mmap.
    pub abs_offset: u64,
    /// Size in bytes occupied by this tensor's data.
    pub byte_size: u64,
}

impl TensorInfo {
    /// Number of elements in this tensor.
    pub fn element_count(&self) -> u64 {
        self.dims.iter().copied().product()
    }
}

/// Parsed GGUF file. Holds an [`Mmap`] of the file and indexes into it.
pub struct Gguf {
    path: PathBuf,
    /// `Arc` so zero-copy tensors ([`crate::iq_gpu`]-adjacent loaders via
    /// `Tensor::from_gguf_borrowed`) can hold clones that keep the mapping
    /// alive after the `Gguf` itself is dropped.
    mmap: Arc<Mmap>,
    version: u32,
    alignment: u64,
    metadata: Vec<(String, MetadataValue)>,
    tensors: Vec<TensorInfo>,
    data_start: u64,
}

impl Gguf {
    /// Open and parse a GGUF file. Lazy: tensor *data* is not touched, only
    /// the header / metadata / tensor-info table.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path).map_err(|source| GgufError::Io {
            path: path.clone(),
            source,
        })?;
        // SAFETY: We hold the File open for the lifetime of the Mmap, and we
        // promise not to modify the file while it is mapped.
        let mmap = unsafe { MmapOptions::new().map(&file) }.map_err(|source| GgufError::Io {
            path: path.clone(),
            source,
        })?;

        let parsed = parse::parse(&mmap)?;
        let file_size = mmap.len() as u64;
        for t in &parsed.tensors {
            let end = t.abs_offset.checked_add(t.byte_size).ok_or_else(|| {
                GgufError::TensorOutOfBounds {
                    name: t.name.clone(),
                    offset: t.abs_offset,
                    size: t.byte_size,
                    file_size,
                }
            })?;
            if end > file_size {
                return Err(GgufError::TensorOutOfBounds {
                    name: t.name.clone(),
                    offset: t.abs_offset,
                    size: t.byte_size,
                    file_size,
                });
            }
        }

        Ok(Self {
            path,
            mmap: Arc::new(mmap),
            version: parsed.version,
            alignment: parsed.alignment,
            metadata: parsed.metadata,
            tensors: parsed.tensors,
            data_start: parsed.data_start,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    pub fn alignment(&self) -> u64 {
        self.alignment
    }

    pub fn metadata(&self) -> &[(String, MetadataValue)] {
        &self.metadata
    }

    pub fn metadata_get(&self, key: &str) -> Option<&MetadataValue> {
        self.metadata.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Borrow the raw bytes of a tensor's data region in the mmap.
    pub fn tensor_bytes(&self, name: &str) -> Option<&[u8]> {
        let info = self.tensor(name)?;
        let start = info.abs_offset as usize;
        let end = start + info.byte_size as usize;
        self.mmap.get(start..end)
    }

    /// Type-erased handle to the backing mmap, for zero-copy tensor
    /// loads ([`crate::Gguf`] + `Tensor::from_gguf_borrowed`). Cloning the
    /// returned `Arc` keeps the file mapping alive independently of this
    /// `Gguf`, so borrowed tensors stay valid after `drop(gguf)`.
    pub fn mmap_backing(&self) -> Arc<dyn AsRef<[u8]> + Send + Sync> {
        self.mmap.clone()
    }

    /// Architecture identifier from the `general.architecture` metadata key.
    pub fn architecture(&self) -> Option<&str> {
        match self.metadata_get("general.architecture")? {
            MetadataValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn data_start(&self) -> u64 {
        self.data_start
    }
}

impl std::fmt::Debug for Gguf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gguf")
            .field("path", &self.path)
            .field("version", &self.version)
            .field("alignment", &self.alignment)
            .field("n_metadata", &self.metadata.len())
            .field("n_tensors", &self.tensors.len())
            .field("data_start", &self.data_start)
            .finish()
    }
}
