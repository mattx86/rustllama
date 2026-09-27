//! Integration tests for the BERT/WordPiece `from_gguf` path.
//! Uses the synthetic BERT GGUF writer to round-trip a vocab
//! through the tokenizer crate and verify CLS/SEP wrapping +
//! basic encode/decode behavior.

use rustllama_gguf::synth::{
    write_synthetic_bert_gguf, SynthBert, SYNTH_BERT_CLS_ID, SYNTH_BERT_MASK_ID, SYNTH_BERT_PAD_ID,
    SYNTH_BERT_SEP_ID, SYNTH_BERT_UNK_ID,
};
use rustllama_gguf::Gguf;
use rustllama_tokenizer::Tokenizer;

fn build_tokenizer(tag: &str) -> (Tokenizer, std::path::PathBuf) {
    let path = std::env::temp_dir().join(format!("rustllama-tokenizer-bert-{tag}.gguf"));
    write_synthetic_bert_gguf(&path, &SynthBert::default());
    let gguf = Gguf::open(&path).expect("open synth bert gguf");
    let tk = Tokenizer::from_gguf(&gguf).expect("from_gguf bert");
    (tk, path)
}

#[test]
fn from_gguf_bert_loads_and_reports_vocab() {
    let (tk, path) = build_tokenizer("vocab-size");
    assert!(
        tk.vocab_size() >= 5,
        "vocab includes at least the 5 special tokens"
    );
    // CLS and SEP IDs from the GGUF metadata are exposed via the BOS
    // /EOS aliases (BERT models reuse those slots in our reader).
    assert_eq!(tk.bos_token_id(), None, "BERT does not set a BOS id");
    assert_eq!(tk.eos_token_id(), None, "BERT does not set an EOS id");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn from_gguf_bert_wraps_input_with_cls_sep_when_add_special_true() {
    let (tk, path) = build_tokenizer("cls-sep");
    let ids = tk
        .encode("hello world", true)
        .expect("encode with specials");
    assert!(
        ids.first().copied() == Some(SYNTH_BERT_CLS_ID),
        "first token must be [CLS] (id {SYNTH_BERT_CLS_ID}), got {ids:?}"
    );
    assert!(
        ids.last().copied() == Some(SYNTH_BERT_SEP_ID),
        "last token must be [SEP] (id {SYNTH_BERT_SEP_ID}), got {ids:?}"
    );
    // The two words `hello` and `world` are in the synth vocab and
    // should each tokenize to a single id (no WordPiece split needed).
    assert_eq!(
        ids.len(),
        4,
        "[CLS] + hello + world + [SEP] = 4 tokens, got {ids:?}"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn from_gguf_bert_no_specials_does_not_wrap() {
    let (tk, path) = build_tokenizer("no-specials");
    let ids = tk
        .encode("hello world", false)
        .expect("encode without specials");
    assert!(
        !ids.contains(&SYNTH_BERT_CLS_ID),
        "no-specials encoding must not include [CLS]: {ids:?}"
    );
    assert!(
        !ids.contains(&SYNTH_BERT_SEP_ID),
        "no-specials encoding must not include [SEP]: {ids:?}"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn from_gguf_bert_lowercases_uppercase_input() {
    let (tk, path) = build_tokenizer("lowercase");
    // Synth vocab is uncased — `Hello World` must hit the same ids as
    // `hello world` after the BertNormalizer pass.
    let lower = tk.encode("hello world", true).unwrap();
    let upper = tk.encode("Hello World", true).unwrap();
    assert_eq!(
        lower, upper,
        "uncased BERT must produce identical ids for upper/lower input"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn from_gguf_bert_unknown_word_falls_back_to_chars_or_unk() {
    let (tk, path) = build_tokenizer("unk-fallback");
    // "xyz" is not in the synth word list but the individual letters
    // `x`, `y`, `z` are — WordPiece's char fallback should yield 3
    // ids (or one [UNK] if the char fallback misses). Either path is
    // valid; the contract we pin is "does not panic, produces something".
    let ids = tk.encode("xyz", false).expect("encode unknown");
    assert!(
        !ids.is_empty(),
        "unknown-word encoding must produce at least one id"
    );
    // All ids must be within vocab — defensive check that the
    // WordPiece model didn't emit a sentinel id outside the table.
    for id in &ids {
        assert!(
            (*id as usize) < tk.vocab_size(),
            "id {id} >= vocab_size {} — out of table",
            tk.vocab_size()
        );
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn from_gguf_bert_special_ids_are_distinct_and_complete() {
    // Defensive: the const IDs we publish for tests must not overlap
    // and must all fit in the configured vocab.
    let ids = [
        SYNTH_BERT_PAD_ID,
        SYNTH_BERT_UNK_ID,
        SYNTH_BERT_CLS_ID,
        SYNTH_BERT_SEP_ID,
        SYNTH_BERT_MASK_ID,
    ];
    let mut sorted = ids.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), 5, "synth bert special ids must be distinct");
    assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
}
