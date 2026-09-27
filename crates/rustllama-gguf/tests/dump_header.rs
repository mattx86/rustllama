//! Diagnostic: dump a real GGUF's metadata + tensor table.
//!
//! Ignored by default — point `RUSTLLAMA_DUMP_GGUF` at a file and run
//!
//!   RUSTLLAMA_DUMP_GGUF=path/to/model.gguf \
//!     cargo test -p rustllama-gguf --test dump_header -- --ignored --nocapture
//!
//! Exists because the release CLI's `models inspect` lags behind
//! parser changes (a freshly added GgmlType can be inspected here
//! before any binary rebuild). A successful run also cross-checks
//! block-size math for every dtype in the file: the parser validates
//! each tensor's byte range against the actual data section, so a
//! wrong `byte_size` surfaces as `TensorOutOfBounds`, not silence.

use rustllama_gguf::Gguf;

#[test]
#[ignore = "diagnostic — needs RUSTLLAMA_DUMP_GGUF pointing at a real file"]
fn dump_gguf_header() {
    let path = std::env::var("RUSTLLAMA_DUMP_GGUF")
        .expect("set RUSTLLAMA_DUMP_GGUF to the .gguf to dump");
    let g = Gguf::open(&path).expect("parse gguf");

    println!("== metadata ({} keys) ==", g.metadata().len());
    for (k, v) in g.metadata() {
        // Elide giant arrays (vocab, merges, sign vectors) to a summary.
        let vs = format!("{v:?}");
        if vs.len() > 160 {
            println!("  {k} = <{} chars: {}...>", vs.len(), &vs[..120]);
        } else {
            println!("  {k} = {vs}");
        }
    }

    println!("== tensors ({}) ==", g.tensors().len());
    let mut by_dtype: std::collections::BTreeMap<&'static str, (usize, u64)> =
        Default::default();
    for t in g.tensors() {
        let e = by_dtype.entry(t.dtype.as_str()).or_default();
        e.0 += 1;
        e.1 += t.byte_size;
    }
    for (dt, (count, bytes)) in &by_dtype {
        println!("  {dt}: {count} tensors, {:.2} MiB", *bytes as f64 / (1024.0 * 1024.0));
    }
    println!("== first/last blocks + non-block tensors ==");
    for t in g.tensors() {
        let name = &t.name;
        let show = !name.starts_with("blk.")
            || name.starts_with("blk.0.")
            || name.starts_with("blk.1.")
            || name.starts_with("blk.3.")
            || name.starts_with("blk.63.");
        if show {
            println!("  {name}  {:?}  {}", t.dims, t.dtype.as_str());
        }
    }
}
