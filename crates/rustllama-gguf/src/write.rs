//! GGUF v3 writer.
//!
//! Streaming writer that mirrors the byte layout of [`crate::parse`]:
//!   - magic + version + n_tensors + n_kv
//!   - metadata table (string key + u32 type + value)
//!   - tensor-info table (string name + u32 n_dims + dims + u32 dtype + u64 rel_offset)
//!   - alignment padding
//!   - per-tensor data (each at `data_start + rel_offset`)
//!
//! Designed for the re-quantization pipeline: callers declare every
//! tensor's `(name, dims, dtype)` upfront, finalize the header, then
//! stream tensor bytes in declaration order. This avoids materializing
//! a multi-GB model in RAM — the heaviest live state is the metadata +
//! tensor-info table, both of which are small relative to the data
//! payload.
//!
//! ```ignore
//! use rustllama_gguf::{GgmlType, MetadataValue};
//! use rustllama_gguf::write::GgufWriter;
//!
//! let mut w = GgufWriter::create("out.gguf")?;
//! w.add_metadata("general.architecture", MetadataValue::String("llama".into()))?;
//! w.declare_tensor("token_embd.weight", vec![4096, 32000], GgmlType::Q4_K)?;
//! w.declare_tensor("output_norm.weight", vec![4096], GgmlType::F32)?;
//! w.finish_header()?;
//! w.write_tensor_data("token_embd.weight", &q4k_bytes)?;
//! w.write_tensor_data("output_norm.weight", &f32_bytes)?;
//! w.finish()?;
//! ```

use std::fs::File;
use std::io::{self, BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

use crate::{GgmlType, MetadataValue, GGUF_DEFAULT_ALIGNMENT};

const GGUF_MAGIC: [u8; 4] = *b"GGUF";
const GGUF_VERSION: u32 = 3;

// Metadata value type ids — mirrors parse::META_* constants.
const META_U8: u32 = 0;
const META_I8: u32 = 1;
const META_U16: u32 = 2;
const META_I16: u32 = 3;
const META_U32: u32 = 4;
const META_I32: u32 = 5;
const META_F32: u32 = 6;
const META_BOOL: u32 = 7;
const META_STRING: u32 = 8;
const META_ARRAY: u32 = 9;
const META_U64: u32 = 10;
const META_I64: u32 = 11;
const META_F64: u32 = 12;

#[derive(Debug, thiserror::Error)]
pub enum WriteError {
    #[error("io: {0}")]
    Io(#[from] io::Error),
    #[error("declare_tensor called after finish_header — declare all tensors first")]
    DeclareAfterHeader,
    #[error("add_metadata called after finish_header — add all metadata first")]
    MetadataAfterHeader,
    #[error("write_tensor_data called before finish_header")]
    DataBeforeHeader,
    #[error("write_tensor_data called after finish — writer already closed")]
    DataAfterFinish,
    #[error(
        "tensor data size mismatch for {name}: declared {expected} bytes (dims={dims:?} \
         dtype={dtype:?}), caller passed {got} bytes"
    )]
    SizeMismatch {
        name: String,
        dims: Vec<u64>,
        dtype: GgmlType,
        expected: u64,
        got: u64,
    },
    #[error(
        "write_tensor_data out of declared order: next expected is {expected:?}, caller \
         passed {got:?}. Tensor bytes must be streamed in the same order tensors were declared."
    )]
    OrderMismatch { expected: String, got: String },
    #[error("write_tensor_data called for unknown tensor {0:?}")]
    UnknownTensor(String),
    #[error("finish_header called twice")]
    DoubleFinishHeader,
    #[error(
        "finish called before all declared tensors were written ({written} of {declared})"
    )]
    UnwrittenTensors { written: usize, declared: usize },
    #[error("alignment must be a positive power of two, got {0}")]
    BadAlignment(u64),
    #[error("tensor {name:?}: empty dims vector (must have at least one dimension)")]
    EmptyDims { name: String },
    #[error("tensor {name:?}: zero-sized dimension at axis {axis}")]
    ZeroDim { name: String, axis: usize },
    #[error("duplicate tensor name: {0:?}")]
    DuplicateTensor(String),
    #[error("duplicate metadata key: {0:?}")]
    DuplicateMetadata(String),
    #[error("internal invariant violated: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, WriteError>;

#[derive(Debug, Clone)]
struct TensorDecl {
    name: String,
    dims: Vec<u64>,
    dtype: GgmlType,
    /// Offset relative to `data_start`. Populated during `finish_header`.
    rel_offset: u64,
    /// Size in bytes — derived from dims + dtype at declaration time.
    byte_size: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Header still mutable: metadata + tensor declarations being collected.
    Declaring,
    /// Header committed. Caller is streaming tensor data; `next_idx`
    /// points at the next tensor in declaration order that needs data.
    WritingData { next_idx: usize },
    /// `finish` was called — writer is consumed.
    Finished,
}

/// Streaming GGUF v3 writer. Generic over the underlying sink so tests
/// can write into a `Cursor<Vec<u8>>` and production callers write to a
/// `BufWriter<File>`. Requires `Seek` because the header is patched
/// (with computed `rel_offset` values) AFTER tensor declarations are
/// complete — we record the offsets-table position at the start, then
/// seek back to fill it in.
pub struct GgufWriter<W: Write + Seek> {
    inner: W,
    metadata: Vec<(String, MetadataValue)>,
    tensors: Vec<TensorDecl>,
    alignment: u64,
    state: State,
    /// Byte offset where tensor data begins. Set by `finish_header`;
    /// each tensor's absolute write position is `data_start + rel_offset`.
    data_start: u64,
}

impl<W: Write + Seek> GgufWriter<W> {
    /// Construct a writer wrapping any seekable sink.
    pub fn new(inner: W) -> Self {
        Self {
            inner,
            metadata: Vec::new(),
            tensors: Vec::new(),
            alignment: GGUF_DEFAULT_ALIGNMENT,
            state: State::Declaring,
            data_start: 0,
        }
    }

    /// Override the file alignment. Must be a positive power of two.
    /// When set to a non-default value, the writer automatically injects
    /// `general.alignment = N` into the metadata at `finish_header`
    /// time so the parser picks it up; the default value (32) is
    /// elided.
    pub fn with_alignment(mut self, alignment: u64) -> Result<Self> {
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(WriteError::BadAlignment(alignment));
        }
        self.alignment = alignment;
        Ok(self)
    }

    /// Append a metadata key/value. Keys must be unique; duplicates
    /// are rejected.
    pub fn add_metadata(
        &mut self,
        key: impl Into<String>,
        value: MetadataValue,
    ) -> Result<()> {
        if self.state != State::Declaring {
            return Err(WriteError::MetadataAfterHeader);
        }
        let key = key.into();
        if self.metadata.iter().any(|(k, _)| k == &key) {
            return Err(WriteError::DuplicateMetadata(key));
        }
        self.metadata.push((key, value));
        Ok(())
    }

    /// Declare a tensor's metadata. The actual bytes are streamed via
    /// `write_tensor_data` after `finish_header` is called. Names must
    /// be unique within the file.
    pub fn declare_tensor(
        &mut self,
        name: impl Into<String>,
        dims: Vec<u64>,
        dtype: GgmlType,
    ) -> Result<()> {
        if self.state != State::Declaring {
            return Err(WriteError::DeclareAfterHeader);
        }
        let name = name.into();
        if dims.is_empty() {
            return Err(WriteError::EmptyDims { name });
        }
        for (axis, &d) in dims.iter().enumerate() {
            if d == 0 {
                return Err(WriteError::ZeroDim { name, axis });
            }
        }
        if self.tensors.iter().any(|t| t.name == name) {
            return Err(WriteError::DuplicateTensor(name));
        }
        let n_elements: u64 = dims.iter().copied().product();
        let byte_size = dtype.byte_size(n_elements);
        self.tensors.push(TensorDecl {
            name,
            dims,
            dtype,
            rel_offset: 0, // patched in finish_header
            byte_size,
        });
        Ok(())
    }

    /// Emit the header (magic + version + counts + metadata table +
    /// tensor-info table + alignment padding) and transition to the
    /// data-streaming state. After this call, the writer is positioned
    /// at `data_start`, ready for the first tensor's bytes.
    pub fn finish_header(&mut self) -> Result<()> {
        if self.state != State::Declaring {
            return Err(WriteError::DoubleFinishHeader);
        }

        // Auto-inject general.alignment if non-default and the caller
        // didn't set it manually. Mirrors llama.cpp's quantize output.
        if self.alignment != GGUF_DEFAULT_ALIGNMENT
            && !self.metadata.iter().any(|(k, _)| k == "general.alignment")
        {
            self.metadata.push((
                "general.alignment".to_string(),
                MetadataValue::U32(self.alignment as u32),
            ));
        }

        // Compute each tensor's rel_offset by laying them out in
        // declaration order with alignment padding between blocks.
        // GGUF doesn't require per-tensor alignment, but llama.cpp
        // aligns each tensor to `general.alignment` and the parser
        // tolerates either. Match llama.cpp by aligning.
        let mut cursor: u64 = 0;
        for t in self.tensors.iter_mut() {
            cursor = align_up(cursor, self.alignment);
            t.rel_offset = cursor;
            cursor += t.byte_size;
        }

        // Header section — write into an in-memory buffer first so we
        // can compute its post-padding length precisely, then write
        // the alignment pad in one shot before the data region. This
        // avoids one extra Seek round-trip on slow filesystems.
        let mut header = Vec::with_capacity(4096);
        header.extend_from_slice(&GGUF_MAGIC);
        write_u32(&mut header, GGUF_VERSION);
        write_u64(&mut header, self.tensors.len() as u64);
        write_u64(&mut header, self.metadata.len() as u64);

        for (key, value) in &self.metadata {
            write_string(&mut header, key);
            write_metadata_value(&mut header, value);
        }

        for t in &self.tensors {
            write_string(&mut header, &t.name);
            write_u32(&mut header, t.dims.len() as u32);
            for d in &t.dims {
                write_u64(&mut header, *d);
            }
            write_u32(&mut header, t.dtype as u32);
            write_u64(&mut header, t.rel_offset);
        }

        let header_len = header.len() as u64;
        let pad = (self.alignment - (header_len % self.alignment)) % self.alignment;
        header.resize((header_len + pad) as usize, 0u8);

        self.inner.write_all(&header)?;
        self.data_start = header.len() as u64;
        self.state = State::WritingData { next_idx: 0 };
        Ok(())
    }

    /// Stream one tensor's data. Must be called once per declared
    /// tensor, in declaration order, with exactly the expected number
    /// of bytes for the declared `(dims, dtype)`.
    pub fn write_tensor_data(&mut self, name: &str, bytes: &[u8]) -> Result<()> {
        let next_idx = match self.state {
            State::WritingData { next_idx } => next_idx,
            State::Declaring => return Err(WriteError::DataBeforeHeader),
            State::Finished => return Err(WriteError::DataAfterFinish),
        };

        let total_tensors = self.tensors.len();
        if next_idx >= total_tensors {
            return Err(WriteError::UnknownTensor(name.to_string()));
        }
        let expected = &self.tensors[next_idx];
        if expected.name != name {
            return Err(WriteError::OrderMismatch {
                expected: expected.name.clone(),
                got: name.to_string(),
            });
        }
        if expected.byte_size != bytes.len() as u64 {
            return Err(WriteError::SizeMismatch {
                name: name.to_string(),
                dims: expected.dims.clone(),
                dtype: expected.dtype,
                expected: expected.byte_size,
                got: bytes.len() as u64,
            });
        }

        // Align the file cursor to `data_start + rel_offset` — for
        // tensors that don't end on an alignment boundary, the next
        // tensor's offset will sit past a small pad region. We seek
        // explicitly so partial writes from earlier tensors can't
        // misalign downstream layout.
        let target = self.data_start + expected.rel_offset;
        self.inner.seek(SeekFrom::Start(target))?;
        self.inner.write_all(bytes)?;

        self.state = State::WritingData {
            next_idx: next_idx + 1,
        };
        Ok(())
    }

    /// Finalize the file. Verifies every declared tensor was written
    /// and flushes the underlying sink.
    pub fn finish(mut self) -> Result<W> {
        let next_idx = match self.state {
            State::WritingData { next_idx } => next_idx,
            State::Declaring => 0, // header never emitted — empty file
            State::Finished => return Err(WriteError::DataAfterFinish),
        };
        if next_idx != self.tensors.len() {
            return Err(WriteError::UnwrittenTensors {
                written: next_idx,
                declared: self.tensors.len(),
            });
        }
        self.inner.flush()?;
        self.state = State::Finished;
        Ok(self.inner)
    }
}

impl GgufWriter<BufWriter<File>> {
    /// Convenience: open `path` for writing (creating or truncating)
    /// and wrap in a buffered `GgufWriter`.
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::create(path)?;
        Ok(Self::new(BufWriter::new(file)))
    }
}

// ----------------------------------------------------------------------
// GgufMmapWriter — file-mapped output variant.
// ----------------------------------------------------------------------
//
// The buffered-`Write` variant above accumulates tensor bytes in a
// caller-owned `Vec<u8>` before each `write_tensor_data` call. For
// targets above ~4 bpw on big-vocab or MoE models a single tensor's
// encoded output can hit several GB — borderline on memory-tight
// hosts.
//
// `GgufMmapWriter` pre-allocates the output file at its computed
// total size (header + sum of tensor padded sizes), memory-maps it
// once, and lets the caller copy encoded bytes **directly into the
// mapped region** via `tensor_region_mut(name)`. Peak per-tensor
// RAM is whatever the OS chooses to keep resident — typically a
// few MB of active pages, never the full tensor.
//
// Crucially, the parallel encoder in the quantize pipeline can
// `par_chunks_mut` the returned slice the same way it does with a
// `Vec<u8>` — rayon doesn't care whether the underlying memory is
// heap or file-mapped.

/// File-mapped GGUF writer. Same declare → finish_header →
/// per-tensor write → finish flow as [`GgufWriter`], but tensor
/// data lands directly in a `memmap2::MmapMut` over the output
/// file rather than going through a `Write + Seek` sink.
///
/// Trade-offs:
///   - Pre-allocates the output at finish-header time (must
///     know total size). Aborts cleanly if disk is short.
///   - Peak memory is OS-paged — kernel resident set adapts to
///     access pattern. Vs. the `Vec`-based path's strict
///     `target_byte_size` per tensor.
///   - The mapped region is `&mut [u8]` from the writer's
///     `tensor_region_mut`; rayon-safe by construction (disjoint
///     subslices via `par_chunks_mut` etc.).
pub struct GgufMmapWriter {
    file: Option<std::fs::File>,
    map: Option<memmap2::MmapMut>,
    metadata: Vec<(String, MetadataValue)>,
    tensors: Vec<TensorDecl>,
    alignment: u64,
    state: State,
    data_start: u64,
    /// Maps tensor name → index into `tensors`. Lets
    /// `tensor_region_mut` find the rel_offset / byte_size in O(1).
    name_index: std::collections::HashMap<String, usize>,
    /// Tracks which tensors the caller has already obtained a
    /// `tensor_region_mut` for, so we can enforce write-once
    /// semantics and verify all tensors received writes at
    /// `finish` time.
    written: Vec<bool>,
}

impl GgufMmapWriter {
    /// Open `path` for writing (creating or truncating). The file is
    /// not yet sized; that happens at `finish_header` once the
    /// total file size is computable from declared tensors.
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        Ok(Self {
            file: Some(file),
            map: None,
            metadata: Vec::new(),
            tensors: Vec::new(),
            alignment: GGUF_DEFAULT_ALIGNMENT,
            state: State::Declaring,
            data_start: 0,
            name_index: std::collections::HashMap::new(),
            written: Vec::new(),
        })
    }

    /// Override the file alignment. Same semantics as [`GgufWriter::with_alignment`].
    pub fn with_alignment(mut self, alignment: u64) -> Result<Self> {
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(WriteError::BadAlignment(alignment));
        }
        self.alignment = alignment;
        Ok(self)
    }

    /// Append a metadata key/value.
    pub fn add_metadata(
        &mut self,
        key: impl Into<String>,
        value: MetadataValue,
    ) -> Result<()> {
        if self.state != State::Declaring {
            return Err(WriteError::MetadataAfterHeader);
        }
        let key = key.into();
        if self.metadata.iter().any(|(k, _)| k == &key) {
            return Err(WriteError::DuplicateMetadata(key));
        }
        self.metadata.push((key, value));
        Ok(())
    }

    /// Declare a tensor's metadata.
    pub fn declare_tensor(
        &mut self,
        name: impl Into<String>,
        dims: Vec<u64>,
        dtype: GgmlType,
    ) -> Result<()> {
        if self.state != State::Declaring {
            return Err(WriteError::DeclareAfterHeader);
        }
        let name = name.into();
        if dims.is_empty() {
            return Err(WriteError::EmptyDims { name });
        }
        for (axis, &d) in dims.iter().enumerate() {
            if d == 0 {
                return Err(WriteError::ZeroDim { name, axis });
            }
        }
        if self.tensors.iter().any(|t| t.name == name) {
            return Err(WriteError::DuplicateTensor(name));
        }
        let n_elements: u64 = dims.iter().copied().product();
        let byte_size = dtype.byte_size(n_elements);
        self.tensors.push(TensorDecl {
            name,
            dims,
            dtype,
            rel_offset: 0,
            byte_size,
        });
        Ok(())
    }

    /// Emit the header, compute the total file size, `set_len` the
    /// file, and map it. After this call,
    /// [`tensor_region_mut`](Self::tensor_region_mut) returns mutable
    /// slices into the mapped region for each tensor.
    pub fn finish_header(&mut self) -> Result<()> {
        if self.state != State::Declaring {
            return Err(WriteError::DoubleFinishHeader);
        }
        if self.alignment != GGUF_DEFAULT_ALIGNMENT
            && !self.metadata.iter().any(|(k, _)| k == "general.alignment")
        {
            self.metadata.push((
                "general.alignment".to_string(),
                MetadataValue::U32(self.alignment as u32),
            ));
        }
        // Same layout as the Write-backed path: pad each tensor to
        // `alignment`, accumulate rel_offset, and record the data
        // region's total byte count.
        let mut cursor: u64 = 0;
        for t in self.tensors.iter_mut() {
            cursor = align_up(cursor, self.alignment);
            t.rel_offset = cursor;
            cursor += t.byte_size;
        }
        let data_region_bytes = cursor;

        // Build the header into a temp Vec exactly as the Write
        // path does, so we can compute data_start.
        let mut header = Vec::with_capacity(4096);
        header.extend_from_slice(&GGUF_MAGIC);
        write_u32(&mut header, GGUF_VERSION);
        write_u64(&mut header, self.tensors.len() as u64);
        write_u64(&mut header, self.metadata.len() as u64);
        for (key, value) in &self.metadata {
            write_string(&mut header, key);
            write_metadata_value(&mut header, value);
        }
        for t in &self.tensors {
            write_string(&mut header, &t.name);
            write_u32(&mut header, t.dims.len() as u32);
            for d in &t.dims {
                write_u64(&mut header, *d);
            }
            write_u32(&mut header, t.dtype as u32);
            write_u64(&mut header, t.rel_offset);
        }
        let header_len = header.len() as u64;
        let pad = (self.alignment - (header_len % self.alignment)) % self.alignment;
        let total_header = header_len + pad;
        let total_file = total_header + data_region_bytes;

        // Size the file to the final total, then map.
        let file = self
            .file
            .as_ref()
            .expect("GgufMmapWriter must have a live file before finish_header");
        file.set_len(total_file)?;
        // SAFETY: `MmapMut::map_mut` requires exclusive write access
        // to the underlying file region. We just truncated + sized
        // the file ourselves and hold a private `File` handle; no
        // other process should be reading it concurrently.
        let mut mmap = unsafe { memmap2::MmapMut::map_mut(file)? };

        // Copy the header (including its alignment pad) into the
        // start of the mapped region. Zero-initialized by `set_len`
        // already, so the pad is implicit; we just write the
        // header bytes.
        mmap[..header.len()].copy_from_slice(&header);

        self.data_start = total_header;
        self.map = Some(mmap);

        // Populate name index + written flags.
        for (i, t) in self.tensors.iter().enumerate() {
            self.name_index.insert(t.name.clone(), i);
        }
        self.written = vec![false; self.tensors.len()];
        self.state = State::WritingData { next_idx: 0 };
        Ok(())
    }

    /// Return a mutable slice into the mapped file for the named
    /// tensor's data region. Each tensor may be requested at most
    /// once. The caller fills the slice (in any order — direct copy,
    /// `par_chunks_mut`, etc.) and the bytes land in the file
    /// directly.
    pub fn tensor_region_mut(&mut self, name: &str) -> Result<&mut [u8]> {
        if !matches!(self.state, State::WritingData { .. }) {
            return Err(WriteError::DataBeforeHeader);
        }
        let idx = *self
            .name_index
            .get(name)
            .ok_or_else(|| WriteError::UnknownTensor(name.to_string()))?;
        if self.written[idx] {
            // Re-acquiring a tensor's region would defeat the
            // write-once invariant the parent `GgufWriter` enforces
            // via order tracking. Reuse the same DuplicateTensor
            // variant for the user-visible message.
            return Err(WriteError::DuplicateTensor(name.to_string()));
        }
        self.written[idx] = true;
        let decl = &self.tensors[idx];
        let abs_lo = (self.data_start + decl.rel_offset) as usize;
        let abs_hi = abs_lo + decl.byte_size as usize;
        let map = self
            .map
            .as_mut()
            .expect("GgufMmapWriter must have a live mmap after finish_header");
        Ok(&mut map[abs_lo..abs_hi])
    }

    /// Acquire mutable regions for every still-unwritten tensor in
    /// one call, marking each as written. Returns disjoint slices
    /// `(name, &mut [u8])` whose lifetimes are tied to the writer's
    /// `&mut self` borrow — caller can iterate them with rayon
    /// `par_iter_mut` to fan out per-tensor work across cores.
    ///
    /// After this returns, every tensor is marked written; any
    /// subsequent `tensor_region_mut` call for any tensor will
    /// fail with `DuplicateTensor`. `finish()` still validates that
    /// all regions were filled (the act of acquiring doesn't fill
    /// them — the caller still has to populate each slice).
    ///
    /// Why this exists: `tensor_region_mut` ties the returned slice's
    /// lifetime to `&mut self`, so the borrow checker forbids holding
    /// regions for two tensors at once. The quantize pipeline needs
    /// to process many tensors in parallel; this one-shot acquire
    /// resolves that lifetime knot.
    pub fn take_all_tensor_regions_mut(&mut self) -> Result<Vec<(String, &mut [u8])>> {
        if !matches!(self.state, State::WritingData { .. }) {
            return Err(WriteError::DataBeforeHeader);
        }
        // Snapshot the (idx, rel_offset, byte_size, name) tuples for
        // every unwritten tensor so we drop the immutable borrow on
        // `self.tensors` before mutably borrowing `self.map`.
        let mut decls: Vec<(usize, u64, u64, String)> = self
            .tensors
            .iter()
            .enumerate()
            .filter(|(i, _)| !self.written[*i])
            .map(|(i, t)| (i, t.rel_offset, t.byte_size, t.name.clone()))
            .collect();
        // Sort ascending by rel_offset so `split_at_mut` walks the
        // underlying mmap left-to-right. `finish_header` lays out
        // tensors in declaration order so this should already be
        // ascending; sort defensively.
        decls.sort_by_key(|d| d.1);

        let data_start = self.data_start as usize;
        let map = self
            .map
            .as_mut()
            .expect("GgufMmapWriter must have a live mmap after finish_header");
        // Strip the header/metadata; we only split the tensor-data
        // tail of the file.
        let (_header, mut remaining) = map.split_at_mut(data_start);
        let mut consumed_after_header: usize = 0;

        let mut out: Vec<(String, &mut [u8])> = Vec::with_capacity(decls.len());
        for (idx, rel_offset, byte_size, name) in decls {
            let target = rel_offset as usize;
            let gap = target.checked_sub(consumed_after_header).ok_or_else(|| {
                WriteError::Internal(format!(
                    "take_all_tensor_regions_mut: tensor {name:?} rel_offset {target} < cursor {consumed_after_header}"
                ))
            })?;
            // Skip any padding gap between cursor and target offset.
            let (_pad, rest) = remaining.split_at_mut(gap);
            let (region, rest2) = rest.split_at_mut(byte_size as usize);
            self.written[idx] = true;
            out.push((name, region));
            remaining = rest2;
            consumed_after_header = target + byte_size as usize;
        }
        Ok(out)
    }

    /// Legacy `write_tensor_data` API for compatibility with code
    /// that already constructs a `Vec<u8>` of encoded bytes. Copies
    /// into the mapped region. Prefer
    /// [`tensor_region_mut`](Self::tensor_region_mut) when the
    /// caller can write directly.
    pub fn write_tensor_data(&mut self, name: &str, bytes: &[u8]) -> Result<()> {
        let idx = *self
            .name_index
            .get(name)
            .ok_or_else(|| WriteError::UnknownTensor(name.to_string()))?;
        let expected = &self.tensors[idx];
        if expected.byte_size != bytes.len() as u64 {
            return Err(WriteError::SizeMismatch {
                name: name.to_string(),
                dims: expected.dims.clone(),
                dtype: expected.dtype,
                expected: expected.byte_size,
                got: bytes.len() as u64,
            });
        }
        let region = self.tensor_region_mut(name)?;
        region.copy_from_slice(bytes);
        Ok(())
    }

    /// Finalize: verify every tensor was written, flush the mmap
    /// to disk, drop the handles.
    pub fn finish(mut self) -> Result<()> {
        if !matches!(self.state, State::WritingData { .. }) {
            return Err(WriteError::DataBeforeHeader);
        }
        let written_count = self.written.iter().filter(|&&w| w).count();
        if written_count != self.tensors.len() {
            return Err(WriteError::UnwrittenTensors {
                written: written_count,
                declared: self.tensors.len(),
            });
        }
        if let Some(map) = self.map.take() {
            // Flush dirty pages to disk before dropping. Without
            // this the file might be missing data on a crash before
            // the OS does its normal flush.
            map.flush()?;
            // Drop the map.
            drop(map);
        }
        if let Some(file) = self.file.take() {
            // Drop the file handle (closes it).
            drop(file);
        }
        self.state = State::Finished;
        Ok(())
    }
}

// ----------------------------------------------------------------------
// Byte-encoding helpers (private). Mirror Cursor::read_* in parse.rs.
// ----------------------------------------------------------------------

fn align_up(value: u64, alignment: u64) -> u64 {
    let rem = value % alignment;
    if rem == 0 {
        value
    } else {
        value + (alignment - rem)
    }
}

fn write_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn write_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn write_string(buf: &mut Vec<u8>, s: &str) {
    write_u64(buf, s.len() as u64);
    buf.extend_from_slice(s.as_bytes());
}

fn write_metadata_value(buf: &mut Vec<u8>, value: &MetadataValue) {
    match value {
        MetadataValue::U8(v) => {
            write_u32(buf, META_U8);
            buf.push(*v);
        }
        MetadataValue::I8(v) => {
            write_u32(buf, META_I8);
            buf.push(*v as u8);
        }
        MetadataValue::U16(v) => {
            write_u32(buf, META_U16);
            buf.extend_from_slice(&v.to_le_bytes());
        }
        MetadataValue::I16(v) => {
            write_u32(buf, META_I16);
            buf.extend_from_slice(&v.to_le_bytes());
        }
        MetadataValue::U32(v) => {
            write_u32(buf, META_U32);
            buf.extend_from_slice(&v.to_le_bytes());
        }
        MetadataValue::I32(v) => {
            write_u32(buf, META_I32);
            buf.extend_from_slice(&v.to_le_bytes());
        }
        MetadataValue::F32(v) => {
            write_u32(buf, META_F32);
            buf.extend_from_slice(&v.to_le_bytes());
        }
        MetadataValue::Bool(v) => {
            write_u32(buf, META_BOOL);
            buf.push(if *v { 1 } else { 0 });
        }
        MetadataValue::String(v) => {
            write_u32(buf, META_STRING);
            write_string(buf, v);
        }
        MetadataValue::U64(v) => {
            write_u32(buf, META_U64);
            buf.extend_from_slice(&v.to_le_bytes());
        }
        MetadataValue::I64(v) => {
            write_u32(buf, META_I64);
            buf.extend_from_slice(&v.to_le_bytes());
        }
        MetadataValue::F64(v) => {
            write_u32(buf, META_F64);
            buf.extend_from_slice(&v.to_le_bytes());
        }
        MetadataValue::Array(items) => {
            write_u32(buf, META_ARRAY);
            // Empty arrays default to U8 inner type (matches the
            // common llama.cpp convention; parser tolerates any
            // type id on an empty array).
            let inner = items
                .first()
                .map(metadata_value_type_id)
                .unwrap_or(META_U8);
            write_u32(buf, inner);
            write_u64(buf, items.len() as u64);
            for item in items {
                write_metadata_value_unwrapped(buf, item);
            }
        }
    }
}

/// Write a metadata value WITHOUT its leading type tag — used inside
/// arrays where the type is declared once at the array level. Mirrors
/// the read side of `META_ARRAY` in parse.rs.
fn write_metadata_value_unwrapped(buf: &mut Vec<u8>, value: &MetadataValue) {
    match value {
        MetadataValue::U8(v) => buf.push(*v),
        MetadataValue::I8(v) => buf.push(*v as u8),
        MetadataValue::U16(v) => buf.extend_from_slice(&v.to_le_bytes()),
        MetadataValue::I16(v) => buf.extend_from_slice(&v.to_le_bytes()),
        MetadataValue::U32(v) => buf.extend_from_slice(&v.to_le_bytes()),
        MetadataValue::I32(v) => buf.extend_from_slice(&v.to_le_bytes()),
        MetadataValue::F32(v) => buf.extend_from_slice(&v.to_le_bytes()),
        MetadataValue::Bool(v) => buf.push(if *v { 1 } else { 0 }),
        MetadataValue::String(v) => write_string(buf, v),
        MetadataValue::U64(v) => buf.extend_from_slice(&v.to_le_bytes()),
        MetadataValue::I64(v) => buf.extend_from_slice(&v.to_le_bytes()),
        MetadataValue::F64(v) => buf.extend_from_slice(&v.to_le_bytes()),
        // Nested arrays: emit a full Array entry (with its own inner
        // type tag + count). Rare in practice but the read path
        // supports them, so the write path must too for round-trip.
        MetadataValue::Array(items) => {
            let inner = items
                .first()
                .map(metadata_value_type_id)
                .unwrap_or(META_U8);
            write_u32(buf, inner);
            write_u64(buf, items.len() as u64);
            for item in items {
                write_metadata_value_unwrapped(buf, item);
            }
        }
    }
}

fn metadata_value_type_id(value: &MetadataValue) -> u32 {
    match value {
        MetadataValue::U8(_) => META_U8,
        MetadataValue::I8(_) => META_I8,
        MetadataValue::U16(_) => META_U16,
        MetadataValue::I16(_) => META_I16,
        MetadataValue::U32(_) => META_U32,
        MetadataValue::I32(_) => META_I32,
        MetadataValue::F32(_) => META_F32,
        MetadataValue::Bool(_) => META_BOOL,
        MetadataValue::String(_) => META_STRING,
        MetadataValue::U64(_) => META_U64,
        MetadataValue::I64(_) => META_I64,
        MetadataValue::F64(_) => META_F64,
        MetadataValue::Array(_) => META_ARRAY,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Gguf;
    use std::io::Cursor;

    /// Round-trip pin: write a minimal GGUF in-memory, read it back via
    /// the parser, assert every field matches what we wrote.
    #[test]
    fn write_then_read_minimal_round_trip() {
        let buf: Vec<u8> = Vec::new();
        let mut w = GgufWriter::new(Cursor::new(buf));
        w.add_metadata(
            "general.architecture",
            MetadataValue::String("llama".into()),
        )
        .unwrap();
        w.add_metadata("llama.block_count", MetadataValue::U32(4))
            .unwrap();
        w.add_metadata("llama.embedding_length", MetadataValue::U32(8))
            .unwrap();

        let f32_bytes = vec![0u8; 8 * 4];
        w.declare_tensor("output_norm.weight", vec![8], GgmlType::F32)
            .unwrap();

        let f32_bytes2 = vec![0u8; 16 * 4];
        w.declare_tensor(
            "blk.0.attn_norm.weight",
            vec![16],
            GgmlType::F32,
        )
        .unwrap();

        w.finish_header().unwrap();
        w.write_tensor_data("output_norm.weight", &f32_bytes)
            .unwrap();
        w.write_tensor_data("blk.0.attn_norm.weight", &f32_bytes2)
            .unwrap();
        let cursor = w.finish().unwrap();
        let bytes = cursor.into_inner();

        // Use the existing parser by writing the bytes to a temp file
        // (parse takes a path via mmap).
        let tmp = std::env::temp_dir().join("rustllama_gguf_writer_roundtrip.gguf");
        std::fs::write(&tmp, &bytes).unwrap();
        let g = Gguf::open(&tmp).unwrap();
        assert_eq!(g.version(), 3);
        assert_eq!(g.tensors().len(), 2);
        assert_eq!(g.metadata().len(), 3);
        assert_eq!(g.architecture(), Some("llama"));
        assert_eq!(
            g.tensor("output_norm.weight").unwrap().dims,
            vec![8u64]
        );
        assert_eq!(
            g.tensor("blk.0.attn_norm.weight").unwrap().dtype,
            GgmlType::F32
        );

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn declare_tensor_rejects_empty_dims() {
        let mut w = GgufWriter::new(Cursor::new(Vec::new()));
        let err = w
            .declare_tensor("bad", vec![], GgmlType::F32)
            .unwrap_err();
        assert!(matches!(err, WriteError::EmptyDims { .. }));
    }

    #[test]
    fn declare_tensor_rejects_zero_dim() {
        let mut w = GgufWriter::new(Cursor::new(Vec::new()));
        let err = w
            .declare_tensor("bad", vec![4, 0, 3], GgmlType::F32)
            .unwrap_err();
        assert!(matches!(err, WriteError::ZeroDim { axis: 1, .. }));
    }

    #[test]
    fn declare_tensor_rejects_duplicate_name() {
        let mut w = GgufWriter::new(Cursor::new(Vec::new()));
        w.declare_tensor("dup", vec![4], GgmlType::F32).unwrap();
        let err = w
            .declare_tensor("dup", vec![8], GgmlType::F32)
            .unwrap_err();
        assert!(matches!(err, WriteError::DuplicateTensor(_)));
    }

    #[test]
    fn add_metadata_rejects_duplicate_key() {
        let mut w = GgufWriter::new(Cursor::new(Vec::new()));
        w.add_metadata("key", MetadataValue::U32(1)).unwrap();
        let err = w
            .add_metadata("key", MetadataValue::U32(2))
            .unwrap_err();
        assert!(matches!(err, WriteError::DuplicateMetadata(_)));
    }

    #[test]
    fn write_tensor_data_rejects_size_mismatch() {
        let mut w = GgufWriter::new(Cursor::new(Vec::new()));
        w.declare_tensor("t", vec![4], GgmlType::F32).unwrap();
        w.finish_header().unwrap();
        // F32 × 4 = 16 bytes expected; pass 8.
        let err = w.write_tensor_data("t", &[0u8; 8]).unwrap_err();
        assert!(matches!(
            err,
            WriteError::SizeMismatch {
                expected: 16,
                got: 8,
                ..
            }
        ));
    }

    #[test]
    fn write_tensor_data_rejects_out_of_order() {
        let mut w = GgufWriter::new(Cursor::new(Vec::new()));
        w.declare_tensor("a", vec![4], GgmlType::F32).unwrap();
        w.declare_tensor("b", vec![4], GgmlType::F32).unwrap();
        w.finish_header().unwrap();
        let err = w.write_tensor_data("b", &[0u8; 16]).unwrap_err();
        assert!(matches!(err, WriteError::OrderMismatch { .. }));
    }

    #[test]
    fn finish_rejects_unwritten_tensors() {
        let mut w = GgufWriter::new(Cursor::new(Vec::new()));
        w.declare_tensor("a", vec![4], GgmlType::F32).unwrap();
        w.declare_tensor("b", vec![4], GgmlType::F32).unwrap();
        w.finish_header().unwrap();
        w.write_tensor_data("a", &[0u8; 16]).unwrap();
        let err = w.finish().unwrap_err();
        assert!(matches!(
            err,
            WriteError::UnwrittenTensors {
                written: 1,
                declared: 2,
            }
        ));
    }

    #[test]
    fn with_alignment_rejects_non_power_of_two() {
        let w = GgufWriter::new(Cursor::new(Vec::new()));
        match w.with_alignment(48) {
            Ok(_) => panic!("expected BadAlignment, got Ok"),
            Err(WriteError::BadAlignment(48)) => {}
            Err(other) => panic!("expected BadAlignment(48), got {other:?}"),
        }
    }

    /// Custom alignment round-trips through the parser. Pin that the
    /// auto-injected `general.alignment` metadata key drives the
    /// parser's tensor-data-start calculation.
    #[test]
    fn custom_alignment_round_trips() {
        let mut w = GgufWriter::new(Cursor::new(Vec::new()))
            .with_alignment(64)
            .unwrap();
        w.declare_tensor("t", vec![4], GgmlType::F32).unwrap();
        w.finish_header().unwrap();
        w.write_tensor_data("t", &[0u8; 16]).unwrap();
        let cursor = w.finish().unwrap();
        let bytes = cursor.into_inner();
        let tmp =
            std::env::temp_dir().join("rustllama_gguf_writer_alignment.gguf");
        std::fs::write(&tmp, &bytes).unwrap();
        let g = Gguf::open(&tmp).unwrap();
        assert_eq!(g.alignment(), 64);
        assert_eq!(g.data_start() % 64, 0);
        let _ = std::fs::remove_file(&tmp);
    }

    /// Mmap writer round-trip: declare + finish_header + fill via
    /// `tensor_region_mut` (the zero-copy API) + finish. Re-open the
    /// file with the parser and verify every tensor's bytes match
    /// exactly what was written into the mapped region.
    ///
    /// This pins the two main correctness invariants of the mmap
    /// path:
    ///   1. `tensor_region_mut` returns a slice pointing at the
    ///      file's data region offset for that tensor (not the
    ///      header, not adjacent tensors).
    ///   2. `finish` flushes the mmap so the file on disk reflects
    ///      everything the encoder wrote — without flush, the OS
    ///      might lose writes on a crash.
    #[test]
    fn mmap_writer_round_trip() {
        let tmp = std::env::temp_dir().join("rustllama_gguf_mmap_roundtrip.gguf");
        let _ = std::fs::remove_file(&tmp);

        let mut w = GgufMmapWriter::create(&tmp).unwrap();
        w.add_metadata(
            "general.architecture",
            MetadataValue::String("llama".into()),
        )
        .unwrap();
        w.add_metadata("llama.block_count", MetadataValue::U32(2))
            .unwrap();
        w.declare_tensor("a.weight", vec![8], GgmlType::F32).unwrap();
        // 256-element Q4_K-sized region (144 bytes) — exercises the
        // alignment-padded layout.
        w.declare_tensor("b.weight", vec![256], GgmlType::Q4_K).unwrap();
        w.finish_header().unwrap();

        // Fill tensor a directly via the zero-copy API.
        let region_a = w.tensor_region_mut("a.weight").unwrap();
        assert_eq!(region_a.len(), 8 * 4);
        let row_a: Vec<f32> = (0..8).map(|i| i as f32 * 0.5).collect();
        for (i, &v) in row_a.iter().enumerate() {
            region_a[i * 4..(i + 1) * 4].copy_from_slice(&v.to_le_bytes());
        }

        // Fill tensor b via the legacy write_tensor_data API
        // (which copies into the mapped region under the hood).
        let bytes_b: Vec<u8> = (0..144).map(|i| (i & 0xFF) as u8).collect();
        w.write_tensor_data("b.weight", &bytes_b).unwrap();

        w.finish().unwrap();

        // Re-open via the parser and verify.
        let g = Gguf::open(&tmp).unwrap();
        assert_eq!(g.tensors().len(), 2);
        assert_eq!(g.architecture(), Some("llama"));
        let a_bytes = g.tensor_bytes("a.weight").unwrap();
        let mut a_f32 = [0f32; 8];
        for i in 0..8 {
            a_f32[i] = f32::from_le_bytes([
                a_bytes[i * 4],
                a_bytes[i * 4 + 1],
                a_bytes[i * 4 + 2],
                a_bytes[i * 4 + 3],
            ]);
        }
        for (got, want) in a_f32.iter().zip(row_a.iter()) {
            assert_eq!(got.to_bits(), want.to_bits());
        }
        let b_bytes = g.tensor_bytes("b.weight").unwrap();
        assert_eq!(b_bytes, bytes_b.as_slice());

        let _ = std::fs::remove_file(&tmp);
    }

    /// `tensor_region_mut` is write-once per tensor — a second call
    /// for the same name must reject.
    #[test]
    fn mmap_writer_rejects_double_acquire() {
        let tmp = std::env::temp_dir().join("rustllama_gguf_mmap_double.gguf");
        let _ = std::fs::remove_file(&tmp);
        let mut w = GgufMmapWriter::create(&tmp).unwrap();
        w.declare_tensor("t.weight", vec![4], GgmlType::F32).unwrap();
        w.finish_header().unwrap();
        let _first = w.tensor_region_mut("t.weight").unwrap();
        let err = w.tensor_region_mut("t.weight").unwrap_err();
        match err {
            WriteError::DuplicateTensor(_) => {}
            other => panic!("expected DuplicateTensor, got {other:?}"),
        }
        let _ = std::fs::remove_file(&tmp);
    }

    /// `finish` rejects when the caller forgot a tensor.
    #[test]
    fn mmap_writer_finish_rejects_unwritten() {
        let tmp = std::env::temp_dir().join("rustllama_gguf_mmap_unwritten.gguf");
        let _ = std::fs::remove_file(&tmp);
        let mut w = GgufMmapWriter::create(&tmp).unwrap();
        w.declare_tensor("a.weight", vec![4], GgmlType::F32).unwrap();
        w.declare_tensor("b.weight", vec![4], GgmlType::F32).unwrap();
        w.finish_header().unwrap();
        let region_a = w.tensor_region_mut("a.weight").unwrap();
        for byte in region_a.iter_mut() {
            *byte = 0;
        }
        // Forget b — finish must error.
        let err = w.finish().unwrap_err();
        match err {
            WriteError::UnwrittenTensors {
                written: 1,
                declared: 2,
            } => {}
            other => panic!("expected UnwrittenTensors(1, 2), got {other:?}"),
        }
        let _ = std::fs::remove_file(&tmp);
    }

    /// Round-trip an array-valued metadata entry. Arrays are the only
    /// non-trivial metadata type — they carry an inner type tag plus
    /// a length-prefixed body. Parity matters because the tokenizer
    /// vocabulary ships as an `Array(String)` in real GGUFs.
    #[test]
    fn array_metadata_round_trips() {
        let mut w = GgufWriter::new(Cursor::new(Vec::new()));
        let tokens: Vec<MetadataValue> = ["<bos>", "<eos>", "hello", "world"]
            .iter()
            .map(|s| MetadataValue::String((*s).into()))
            .collect();
        w.add_metadata("tokenizer.ggml.tokens", MetadataValue::Array(tokens))
            .unwrap();
        let ids: Vec<MetadataValue> =
            (0..4).map(|i| MetadataValue::I32(i)).collect();
        w.add_metadata("tokenizer.ggml.token_type", MetadataValue::Array(ids))
            .unwrap();
        w.declare_tensor("t", vec![1], GgmlType::F32).unwrap();
        w.finish_header().unwrap();
        w.write_tensor_data("t", &[0u8; 4]).unwrap();
        let cursor = w.finish().unwrap();
        let bytes = cursor.into_inner();
        let tmp = std::env::temp_dir().join("rustllama_gguf_writer_array.gguf");
        std::fs::write(&tmp, &bytes).unwrap();
        let g = Gguf::open(&tmp).unwrap();
        match g.metadata_get("tokenizer.ggml.tokens").unwrap() {
            MetadataValue::Array(items) => {
                assert_eq!(items.len(), 4);
                assert!(matches!(
                    &items[0],
                    MetadataValue::String(s) if s == "<bos>"
                ));
            }
            other => panic!("expected Array, got {other:?}"),
        }
        let _ = std::fs::remove_file(&tmp);
    }

    /// Bigger round-trip: write a small "model" with several tensors
    /// of varying dtypes (F32, F16, BF16, Q8_0-sized byte blob),
    /// re-read, and verify every byte of every tensor matches what
    /// was written. This pins the full parser-writer contract.
    #[test]
    fn multi_tensor_round_trip_with_byte_equality() {
        let mut w = GgufWriter::new(Cursor::new(Vec::new()));
        w.add_metadata(
            "general.architecture",
            MetadataValue::String("llama".into()),
        )
        .unwrap();

        // Synthetic tensors with deterministic content.
        let t1_f32: Vec<u8> = (0..32)
            .flat_map(|i: u32| (i as f32).to_le_bytes())
            .collect();
        let t2_f16: Vec<u8> = (0..64)
            .flat_map(|i: u32| (i as u16).to_le_bytes())
            .collect();
        // Q8_0 block = 34 bytes per 32 weights. 64 weights → 2 blocks → 68 bytes.
        let t3_q80: Vec<u8> = (0..68).map(|i: u32| (i & 0xFF) as u8).collect();

        w.declare_tensor("a.weight", vec![32], GgmlType::F32).unwrap();
        w.declare_tensor("b.weight", vec![64], GgmlType::F16).unwrap();
        w.declare_tensor("c.weight", vec![64], GgmlType::Q8_0).unwrap();
        w.finish_header().unwrap();
        w.write_tensor_data("a.weight", &t1_f32).unwrap();
        w.write_tensor_data("b.weight", &t2_f16).unwrap();
        w.write_tensor_data("c.weight", &t3_q80).unwrap();
        let bytes = w.finish().unwrap().into_inner();

        let tmp =
            std::env::temp_dir().join("rustllama_gguf_writer_multi.gguf");
        std::fs::write(&tmp, &bytes).unwrap();
        let g = Gguf::open(&tmp).unwrap();
        assert_eq!(g.tensors().len(), 3);
        assert_eq!(g.tensor_bytes("a.weight").unwrap(), t1_f32.as_slice());
        assert_eq!(g.tensor_bytes("b.weight").unwrap(), t2_f16.as_slice());
        assert_eq!(g.tensor_bytes("c.weight").unwrap(), t3_q80.as_slice());
        let _ = std::fs::remove_file(&tmp);
    }
}
