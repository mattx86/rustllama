use rustllama_gguf::Gguf;
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let g = Gguf::open(path).expect("open");
    for (k, v) in g.metadata() {
        let preview = match v {
            rustllama_gguf::MetadataValue::String(s) => {
                let s = if s.len() > 60 { &s[..60] } else { s.as_str() };
                format!("str = {:?}", s)
            }
            rustllama_gguf::MetadataValue::Array(a) => {
                // Show first ~8 elements so we can see e.g. the
                // rope.dimension_sections values inline.
                let preview: Vec<String> = a.iter().take(8).map(|v| format!("{v:?}")).collect();
                let suffix = if a.len() > 8 { ", …" } else { "" };
                format!("array (len {}) = [{}{}]", a.len(), preview.join(", "), suffix)
            }
            other => format!("{:?}", other),
        };
        println!("{k} : {preview}");
    }
}
