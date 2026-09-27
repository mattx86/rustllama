//! Integration: load tokenizer + CpuEngine from synthetic GGUF; exercise
//! chat-template rendering through the engine's stack.

use rustllama_engine::CpuEngine;
use rustllama_gguf::synth::{write_synthetic_llama_gguf, SynthLlama, SynthTokenizer};
use rustllama_gguf::Gguf;
use rustllama_tokenizer::{ChatMessage, Tokenizer};

fn synth_path(tag: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("rustllama-tok-{tag}.gguf"))
}

#[test]
fn tokenizer_loads_from_synthetic_gguf() {
    let path = synth_path("loads");
    write_synthetic_llama_gguf(&path, &SynthLlama::default());
    let gguf = Gguf::open(&path).expect("open");

    let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer load");
    assert_eq!(tok.bos_token_id(), Some(1));
    assert_eq!(tok.eos_token_id(), Some(2));
    assert!(tok.chat_template().is_some(), "chat template carried");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn chat_template_renders_through_tokenizer() {
    let path = synth_path("template");
    write_synthetic_llama_gguf(&path, &SynthLlama::default());
    let gguf = Gguf::open(&path).expect("open");
    let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer load");

    let rendered = tok
        .render_chat(
            &[
                ChatMessage {
                    role: "system",
                    content: "be terse",
                },
                ChatMessage {
                    role: "user",
                    content: "hello",
                },
            ],
            true,
        )
        .expect("render");
    assert!(rendered.contains("<|im_start|>system\nbe terse<|im_end|>"));
    assert!(rendered.contains("<|im_start|>user\nhello<|im_end|>"));
    assert!(rendered.ends_with("<|im_start|>assistant\n"));

    let _ = std::fs::remove_file(&path);
}

#[test]
fn cpu_engine_loads_with_tokenizer() {
    let path = synth_path("engine");
    write_synthetic_llama_gguf(&path, &SynthLlama::default());

    let engine = CpuEngine::load_with_tokenizer(&path, 16).expect("load engine");
    assert!(engine.tokenizer().is_some());
    assert_eq!(engine.vocab_size(), 32);
    assert_eq!(engine.model_id(), "rustllama-tok-engine");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn encode_streaming_concat_equals_single_encode() {
    // Streaming encode splits the prompt at chat-template `<|...|>`
    // boundaries and encodes each segment separately. Concatenating
    // all chunks must reproduce exactly what a single `encode` call
    // would have returned — otherwise the prefill loop would see a
    // different token sequence depending on which API the engine uses.
    let path = synth_path("stream-parity");
    write_synthetic_llama_gguf(&path, &SynthLlama::default());
    let gguf = Gguf::open(&path).expect("open");
    let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer load");

    let rendered = tok
        .render_chat(
            &[
                ChatMessage {
                    role: "system",
                    content: "be terse",
                },
                ChatMessage {
                    role: "user",
                    content: "what is 2+2?",
                },
                ChatMessage {
                    role: "assistant",
                    content: "4",
                },
                ChatMessage {
                    role: "user",
                    content: "and 3+3?",
                },
            ],
            true,
        )
        .expect("render");
    let baseline = tok.encode(&rendered, true).expect("single encode");
    let stream = tok
        .encode_streaming(&rendered, true)
        .expect("streaming encode start");
    let streamed = stream.collect_into_vec().expect("drain stream");
    assert_eq!(
        baseline, streamed,
        "streamed concat must equal single encode"
    );

    let _ = std::fs::remove_file(&path);
}

#[test]
fn encode_streaming_handles_plain_text_without_specials() {
    // A prompt with no `<|` markers falls back to one segment; the
    // stream still produces the same tokens as a single encode.
    let path = synth_path("stream-plain");
    write_synthetic_llama_gguf(&path, &SynthLlama::default());
    let gguf = Gguf::open(&path).expect("open");
    let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer load");

    let text = "no special tokens here at all";
    let baseline = tok.encode(text, true).expect("single encode");
    let streamed = tok
        .encode_streaming(text, true)
        .expect("streaming encode start")
        .collect_into_vec()
        .expect("drain stream");
    assert_eq!(baseline, streamed);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn sentencepiece_tokenizer_loads() {
    let path = synth_path("spm");
    let params = SynthLlama {
        tokenizer: SynthTokenizer::Llama,
        ..SynthLlama::default()
    };
    write_synthetic_llama_gguf(&path, &params);
    let gguf = Gguf::open(&path).expect("open");
    let tok = Tokenizer::from_gguf(&gguf).expect("spm tokenizer load");
    assert_eq!(tok.bos_token_id(), Some(1));
    assert_eq!(tok.eos_token_id(), Some(2));
    assert!(tok.chat_template().is_some());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn engine_chat_stream_interleaved_prefill_is_deterministic() {
    // Two back-to-back chat() calls with the same prompt and seed must
    // produce the same token stream. Exercises the full streaming
    // encode → interleaved prefill → decode path; the second call also
    // exercises the prefix-cache pool restoring chunk 0's tokens.
    use futures::StreamExt;
    use rustllama_engine::{ChatMessage as EngineChat, Engine, SamplingParams};

    let path = synth_path("stream-determ");
    write_synthetic_llama_gguf(&path, &SynthLlama::default());

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("rt");
    let load_path = path.clone();
    runtime.block_on(async move {
        let engine = CpuEngine::load_with_tokenizer(&load_path, 16).expect("load engine");
        let sampling = SamplingParams {
            temperature: 0.0,
            top_p: 0.0,
            top_k: 0,
            repeat_penalty: 1.0,
            max_tokens: 4,
            stop: vec![],
            seed: 7,
            ..SamplingParams::default()
        };
        let prompt = || {
            vec![EngineChat {
                role: "user".into(),
                content: "<tok_5>".into(),
                images: Vec::new(),
            }]
        };

        let mut first: Vec<u32> = Vec::new();
        let mut first_err = false;
        let mut s = engine.chat(&prompt(), &sampling).expect("construct 1");
        while let Some(item) = s.next().await {
            match item {
                Ok(t) => first.push(t.id),
                Err(_) => {
                    first_err = true;
                    break;
                }
            }
        }
        drop(s);

        let mut second: Vec<u32> = Vec::new();
        let mut second_err = false;
        let mut s = engine.chat(&prompt(), &sampling).expect("construct 2");
        while let Some(item) = s.next().await {
            match item {
                Ok(t) => second.push(t.id),
                Err(_) => {
                    second_err = true;
                    break;
                }
            }
        }
        drop(s);
        assert_eq!(
            first_err, second_err,
            "error/success disposition must match across runs"
        );
        assert_eq!(
            first, second,
            "streaming chat must be deterministic across identical inputs"
        );
    });

    let _ = std::fs::remove_file(&path);
}

#[test]
fn engine_chat_stream_does_not_panic() {
    use futures::StreamExt;
    use rustllama_engine::{ChatMessage as EngineChat, Engine, SamplingParams};

    let path = synth_path("stream");
    write_synthetic_llama_gguf(&path, &SynthLlama::default());

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("rt");

    let load_path = path.clone();
    runtime.block_on(async move {
        let engine = CpuEngine::load_with_tokenizer(&load_path, 16).expect("load engine");
        let sampling = SamplingParams {
            temperature: 0.0,
            top_p: 0.0,
            top_k: 0,
            repeat_penalty: 1.0,
            max_tokens: 4,
            stop: vec![],
            seed: 1,
            ..SamplingParams::default()
        };
        let mut stream = engine
            .chat(
                &[EngineChat {
                    role: "user".into(),
                    content: "<tok_5>".into(),
                    images: Vec::new(),
                }],
                &sampling,
            )
            .expect("construct stream");

        // The synthetic vocab has no byte-level coverage, so encoding may
        // produce an error mid-stream. Either way, consuming should not
        // panic.
        let mut got_anything = false;
        while let Some(item) = stream.next().await {
            got_anything = true;
            // Errors are acceptable on the synthetic; we only assert no panic.
            let _ = item;
        }
        // It's fine for `got_anything` to be false (e.g., empty encode), as
        // long as the consume loop completed.
        let _ = got_anything;
    });

    let _ = std::fs::remove_file(&path);
}
