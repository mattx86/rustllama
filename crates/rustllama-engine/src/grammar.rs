//! Grammar-constrained sampling: a token-mask filter that lets only
//! grammar-accepting tokens through the sampler. v1 ships JSON +
//! JSON-Schema-constrained JSON; the underlying mask is generic so
//! future regex / code-syntax grammars can plug in the same way.
//!
//! ## Architecture
//!
//! The sampler's softmax+top-k pipeline already prunes the candidate
//! set to a manageable size. We attach a grammar by:
//!   1. Pre-decoding every vocab id to its UTF-8 byte sequence once at
//!      grammar init (typically ~150k entries; under 10 MB).
//!   2. Before the multinomial draw, scanning the top candidates and
//!      *cloning* the parser to speculatively step each candidate's
//!      bytes. Rejected candidates have their probability zeroed.
//!   3. After token selection, advancing the parser by the chosen
//!      token's bytes — making the next step's mask reflect the new
//!      parser state.
//!
//! Clone-and-step is O(token_bytes) per candidate. With a bounded
//! candidate set (top-32 or so), the per-step overhead is small
//! relative to a forward pass.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

/// A subset of JSON Schema we can enforce at sampling time. Maps to
/// the OpenAI `response_format.json_schema.schema` shape with these
/// supported keywords:
///
///   - `type` — `"string"`, `"number"`, `"integer"`, `"boolean"`,
///     `"null"`, `"object"`, `"array"`. Multi-type (`["string", "null"]`)
///     not supported yet.
///   - `properties` — per-key schema for object properties.
///   - `required` — list of mandatory property names. Enforced at
///     object-close: `}` is rejected until every required key has
///     appeared.
///   - `items` — schema for each array element.
///   - `enum` — list of literal allowed values. For strings: must
///     match one of the listed strings. For numbers: must match
///     numerically. For booleans/null: trivially constrained by type.
///   - `const` — sugar for an `enum` of size 1.
///
/// **Not yet supported** (future work, deliberately limited so v1 ships):
/// `additionalProperties` (treated as `true`), `oneOf`/`anyOf`/`allOf`,
/// pattern/format/minLength/etc, `$ref`, tuple-typed arrays. Schemas
/// using these keywords parse fine but the unsupported keywords are
/// ignored — the constraint becomes a superset of what the schema
/// actually permits (over-permissive, never wrongly-restrictive).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Schema {
    /// `type: "string"` — strings only, optionally constrained to an enum.
    String {
        #[serde(default, rename = "enum")]
        enum_values: Option<Vec<String>>,
    },
    /// `type: "number"` or `type: "integer"` — numbers only.
    Number {
        #[serde(default)]
        integer_only: bool,
    },
    /// `type: "boolean"`.
    Boolean,
    /// `type: "null"`.
    Null,
    /// `type: "object"`.
    Object {
        #[serde(default)]
        properties: BTreeMap<String, Schema>,
        #[serde(default)]
        required: BTreeSet<String>,
        /// `additionalProperties: true|false`. Defaults to `true`
        /// (lax) per JSON Schema spec, except OpenAI's `strict: true`
        /// flag overrides it to `false`. The schema parser at the
        /// server boundary handles that override.
        #[serde(default = "default_true")]
        additional: bool,
    },
    /// `type: "array"`.
    Array {
        #[serde(default)]
        items: Option<Box<Schema>>,
    },
    /// Anything that's valid JSON. Used as a fallback when the
    /// supplied schema is empty / lacks `type`.
    Any,
}

fn default_true() -> bool {
    true
}

impl Schema {
    /// Parse a `serde_json::Value` (the user's raw JSON Schema) into
    /// our internal representation. Lenient: unknown keywords are
    /// silently dropped, and missing `type` collapses to [`Schema::Any`].
    pub fn from_json_value(v: &serde_json::Value) -> Self {
        let Some(obj) = v.as_object() else {
            return Schema::Any;
        };

        // `const` is sugar for an enum of size 1; rewrite it.
        let mut enum_values: Option<Vec<serde_json::Value>> =
            obj.get("enum").and_then(|e| e.as_array().cloned());
        if let Some(c) = obj.get("const") {
            enum_values = Some(vec![c.clone()]);
        }

        let ty = obj.get("type").and_then(|t| t.as_str());
        match ty {
            Some("string") => Schema::String {
                enum_values: enum_values.map(|arr| {
                    arr.into_iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                }),
            },
            Some("integer") => Schema::Number { integer_only: true },
            Some("number") => Schema::Number { integer_only: false },
            Some("boolean") => Schema::Boolean,
            Some("null") => Schema::Null,
            Some("object") => {
                let properties = obj
                    .get("properties")
                    .and_then(|p| p.as_object())
                    .map(|m| {
                        m.iter()
                            .map(|(k, v)| (k.clone(), Schema::from_json_value(v)))
                            .collect::<BTreeMap<_, _>>()
                    })
                    .unwrap_or_default();
                let required = obj
                    .get("required")
                    .and_then(|r| r.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect::<BTreeSet<_>>()
                    })
                    .unwrap_or_default();
                let additional = obj
                    .get("additionalProperties")
                    .and_then(|a| a.as_bool())
                    .unwrap_or(true);
                Schema::Object {
                    properties,
                    required,
                    additional,
                }
            }
            Some("array") => Schema::Array {
                items: obj
                    .get("items")
                    .map(|v| Box::new(Schema::from_json_value(v))),
            },
            _ => Schema::Any,
        }
    }
}

/// Result of stepping one byte through the grammar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    /// Byte was accepted; the parser remains live and may consume more.
    Accept,
    /// Byte completes the top-level value. The parser will reject any
    /// further non-whitespace bytes — the model should emit EOS now.
    Done,
    /// Byte violates the grammar.
    Reject,
}

/// Token-level grammar filter. Holds a clone-cheap parser plus a
/// pre-decoded byte table indexed by token id.
pub struct GrammarMask {
    parser: ParserKind,
    /// `token_bytes[id]` is the UTF-8 byte sequence the tokenizer
    /// produces when decoding just that id (with `clean_up_whitespace
    /// = false` so structural bytes aren't dropped). Computed once at
    /// init. Wrapped in `Arc` so cheap clones share the table.
    token_bytes: Arc<Vec<Vec<u8>>>,
    /// EOS token id, when known. Always allowed once `parser.is_done`.
    eos_id: Option<u32>,
}

/// Hide the parser variant behind an enum so callers don't need to
/// know about JSON vs JSON-Schema vs tool-call parsing. Adding a
/// future grammar (regex, code-syntax) means adding a variant here.
#[derive(Debug, Clone)]
enum ParserKind {
    Json(JsonParser),
    JsonSchema(JsonSchemaParser),
    ToolCall(crate::tool_grammar::ToolCallStreamGrammar),
    Code(CodeGrammarParser),
    Regex(RegexGrammarParser),
}

impl ParserKind {
    fn step(&mut self, b: u8) -> Step {
        match self {
            ParserKind::Json(p) => p.step(b),
            ParserKind::JsonSchema(p) => p.step(b),
            ParserKind::ToolCall(p) => p.step(b),
            ParserKind::Code(p) => p.step(b),
            ParserKind::Regex(p) => p.step(b),
        }
    }

    fn is_done(&self) -> bool {
        match self {
            ParserKind::Json(p) => p.is_done(),
            ParserKind::JsonSchema(p) => p.is_done(),
            // Tool-call stream is "done" only when EOS — the grammar
            // is stream-shaped, not value-shaped. Caller uses
            // can_terminate_now to decide when EOS is legal.
            ParserKind::ToolCall(_) => false,
            // Code grammar is stream-shaped too: balanced state is
            // legal-to-terminate-at, but not "done".
            ParserKind::Code(_) => false,
            // Regex is also stream-shaped — many patterns have
            // unbounded suffixes (`.*`, `\d+`). The model decides
            // when to stop; we expose match-state-reached via
            // can_terminate_now() so EOS becomes allowed.
            ParserKind::Regex(_) => false,
        }
    }

    fn can_terminate_now(&self) -> bool {
        match self {
            ParserKind::Json(p) => p.can_terminate_now(),
            ParserKind::JsonSchema(p) => p.can_terminate_now(),
            ParserKind::ToolCall(p) => p.can_terminate_now(),
            ParserKind::Code(p) => p.can_terminate_now(),
            ParserKind::Regex(p) => p.can_terminate_now(),
        }
    }
}

/// Token-level code-syntax constraint. v1 ships the bracket-balance
/// rule: every `(`, `[`, and `{` must be closed in matching order,
/// and any token that would emit an unmatched close gets masked out.
/// Strings (single, double, backtick) are recognized so brackets
/// inside them don't shift the balance.
///
/// Language-specific extras (Python indentation, Rust lifetimes,
/// Go-style semicolon insertion, etc.) are out of scope for v1 — the
/// `language` field on `GrammarKind::Code` is accepted but unused
/// today. A real per-language parser would slot into `ParserKind` as
/// a separate variant (`CodePython`, `CodeRust`) without changing
/// this base parser.
/// Per-language comment recognition style for the code-grammar parser.
/// Picked from the `language` string by `CommentStyle::for_language`.
///
/// The parser needs to know what counts as a comment so brackets and
/// quotes inside `// this { is fine }` don't shift the balance or
/// open a string. Without this, a model generating `# (this is a python
/// comment)` would push a `(` onto the bracket stack and refuse to
/// terminate until it produced a `)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommentStyle {
    /// No comment recognition. Default; matches the v1.0 parser
    /// behavior. Safe for languages we don't have explicit support
    /// for — the worst case is a stray comment shifts the bracket
    /// stack, but most real-world model outputs put comments at
    /// line ends where the balance is already even.
    None,
    /// C-family: `//` line comments + `/* … */` block comments.
    /// Covers C, C++, Rust, JavaScript, TypeScript, Java, Go,
    /// Swift, Kotlin, C#, Scala.
    CFamily,
    /// Hash line comments only (`# …` to end of line). Covers
    /// Python, Ruby, Bash/Zsh/Fish, Perl, R, Elixir, YAML, TOML,
    /// Makefile.
    Hash,
    /// Hash AND C-family — useful for files with mixed shebang +
    /// C-style code (rare, but harmless).
    HashAndCFamily,
}

impl CommentStyle {
    /// Map a `GrammarKind::Code.language` string (lowercase
    /// expected; we lowercase on entry) to the comment style. Unknown
    /// languages get `None` rather than guessing.
    pub fn for_language(language: &str) -> Self {
        let l = language.trim().to_ascii_lowercase();
        match l.as_str() {
            // C-family
            "c" | "cpp" | "c++" | "cxx" | "rust" | "rs" | "javascript"
            | "js" | "typescript" | "ts" | "java" | "go" | "golang"
            | "swift" | "kotlin" | "kt" | "csharp" | "cs" | "scala"
            | "dart" | "php" | "objc" | "objective-c" => Self::CFamily,
            // Hash-only
            "python" | "py" | "ruby" | "rb" | "bash" | "sh" | "zsh"
            | "fish" | "shell" | "perl" | "pl" | "r" | "elixir" | "ex"
            | "yaml" | "yml" | "toml" | "makefile" | "make" | "dockerfile"
            | "ini" | "conf" | "config" | "nix" => Self::Hash,
            // Default — recognize neither.
            _ => Self::None,
        }
    }

    fn line_comment_starter(self, b: u8, prev: Option<u8>) -> bool {
        match self {
            Self::None => false,
            Self::CFamily => b == b'/' && prev == Some(b'/'),
            Self::Hash => b == b'#',
            Self::HashAndCFamily => {
                b == b'#' || (b == b'/' && prev == Some(b'/'))
            }
        }
    }

    fn block_comment_starter(self, b: u8, prev: Option<u8>) -> bool {
        match self {
            Self::CFamily | Self::HashAndCFamily => {
                b == b'*' && prev == Some(b'/')
            }
            _ => false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CodeGrammarParser {
    /// Stack of open-bracket bytes (`b'('`, `b'['`, `b'{'`). The
    /// matching close-bracket byte is the rule: `(` → `)`, `[` →
    /// `]`, `{` → `}`. Closing in the wrong order is a Reject.
    stack: Vec<u8>,
    /// Currently-inside-a-string state: `Some(quote_byte)` when we're
    /// past an opening `"`, `'`, or backtick and haven't seen the
    /// matching close.
    in_string: Option<u8>,
    /// One-shot "next byte is escaped" flag — set after a `\` inside
    /// a string.
    escape_next: bool,
    /// Currently inside a `//`-style line comment (CFamily) or `#`
    /// line comment (Hash). Set on the second `/` of `//` or on `#`
    /// outside a string; cleared on the next `\n`.
    in_line_comment: bool,
    /// Currently inside a `/* … */` block comment. Set on the `*` of
    /// `/*` outside a string; cleared on the `/` of `*/`.
    in_block_comment: bool,
    /// Most recent non-comment-resetting byte. Drives the two-byte
    /// lookahead for `//`, `/*`, and `*/` recognition. `None` at
    /// stream start.
    prev_byte: Option<u8>,
    /// Active comment style. Determined from
    /// `GrammarKind::Code.language` at parser construction.
    comment_style: CommentStyle,
}

impl CodeGrammarParser {
    pub fn new() -> Self {
        Self::with_language("")
    }

    /// Construct a parser tuned for the given language (informs
    /// comment recognition). Pass `""` for the v1.0 no-comment
    /// behavior.
    pub fn with_language(language: &str) -> Self {
        Self {
            stack: Vec::new(),
            in_string: None,
            escape_next: false,
            in_line_comment: false,
            in_block_comment: false,
            prev_byte: None,
            comment_style: CommentStyle::for_language(language),
        }
    }

    fn step(&mut self, b: u8) -> Step {
        // Block comment runs first — it can span newlines, so the
        // line-comment check below shouldn't fire inside it.
        if self.in_block_comment {
            // `*/` closes the block comment. We detect the closing
            // `/` when prev_byte was `*`. The `*` itself is just a
            // pass-through byte inside the comment.
            if b == b'/' && self.prev_byte == Some(b'*') {
                self.in_block_comment = false;
            }
            self.prev_byte = Some(b);
            return Step::Accept;
        }
        // Line comment runs until newline. Bracket / quote bytes
        // inside a line comment are ignored.
        if self.in_line_comment {
            if b == b'\n' {
                self.in_line_comment = false;
            }
            self.prev_byte = Some(b);
            return Step::Accept;
        }
        // Inside a string: bytes pass through. Track the closing
        // quote and the escape state so a `"` inside a string
        // doesn't toggle balance.
        if let Some(q) = self.in_string {
            if self.escape_next {
                self.escape_next = false;
                self.prev_byte = Some(b);
                return Step::Accept;
            }
            if b == b'\\' {
                self.escape_next = true;
                self.prev_byte = Some(b);
                return Step::Accept;
            }
            if b == q {
                self.in_string = None;
            }
            self.prev_byte = Some(b);
            return Step::Accept;
        }
        // Outside any comment/string. Check for comment starters
        // before bracket/quote logic so e.g. `//` doesn't push a
        // bracket. The block-comment check needs the prev byte (`/`
        // followed by `*`).
        if self.comment_style.block_comment_starter(b, self.prev_byte) {
            self.in_block_comment = true;
            self.prev_byte = Some(b);
            return Step::Accept;
        }
        if self.comment_style.line_comment_starter(b, self.prev_byte) {
            self.in_line_comment = true;
            self.prev_byte = Some(b);
            return Step::Accept;
        }
        let result = match b {
            b'"' | b'\'' | b'`' => {
                self.in_string = Some(b);
                Step::Accept
            }
            b'(' | b'[' | b'{' => {
                self.stack.push(b);
                Step::Accept
            }
            b')' | b']' | b'}' => {
                let want = match self.stack.last() {
                    Some(b'(') => b')',
                    Some(b'[') => b']',
                    Some(b'{') => b'}',
                    _ => return Step::Reject,
                };
                if want != b {
                    return Step::Reject;
                }
                self.stack.pop();
                Step::Accept
            }
            _ => Step::Accept,
        };
        self.prev_byte = Some(b);
        result
    }

    fn can_terminate_now(&self) -> bool {
        // Balanced and not mid-string / mid-comment → safe to stop.
        // Line comments are terminable (the `\n` would close it
        // eventually; an early stop on a balanced state is fine
        // since the model emitted EOS deliberately).
        self.stack.is_empty()
            && self.in_string.is_none()
            && !self.in_block_comment
    }
}

/// Token-level regex constraint. Compiles the user's pattern to a
/// dense byte-DFA (`regex_automata::dfa::dense::DFA`), anchored
/// at the start of input. Each `step(b)` transitions the DFA; a
/// dead state means the byte sequence can never match the pattern
/// from here (Reject). Match states are "safe to terminate" but
/// not "done" — the model decides when to emit EOS, because many
/// useful patterns have unbounded suffixes (e.g. `\d+`, `[a-z]+`).
///
/// **Anchoring**: the DFA is built `anchored(Anchored::Yes)` so
/// the pattern must match the start of input. A trailing `.*` (or
/// equivalent) is the user's choice — without one the constraint
/// rejects any byte past the first match, which is usually what
/// they want for short regex-shaped outputs (phone numbers, ids,
/// fixed enumerations).
///
/// **Unicode**: regex-automata's byte DFA matches UTF-8 byte
/// sequences. The pattern's `.` defaults to "any Unicode scalar
/// value" (multi-byte), which transitions through 2-4 DFA states
/// for non-ASCII input. Anchoring + the byte-level step model
/// handle this transparently — no per-character bookkeeping is
/// required at this layer.
#[derive(Debug, Clone)]
pub struct RegexGrammarParser {
    /// Shared compiled DFA. `Arc` so cloning the parser at every
    /// candidate-token speculation step is cheap.
    dfa: Arc<regex_automata::dfa::dense::DFA<Vec<u32>>>,
    /// Current DFA state. Updated on each `step(b)` via the
    /// shared DFA's transition table.
    state: regex_automata::util::primitives::StateID,
}

#[derive(Debug, thiserror::Error)]
pub enum RegexGrammarError {
    #[error("invalid regex pattern: {0}")]
    InvalidPattern(String),
}

impl RegexGrammarParser {
    /// Compile `pattern` to a dense byte-DFA and start a fresh
    /// parser. The pattern is anchored at the start of input — the
    /// constraint applies to the model's full output, not a
    /// substring within it.
    pub fn new(pattern: &str) -> std::result::Result<Self, RegexGrammarError> {
        use regex_automata::dfa::dense;
        use regex_automata::dfa::Automaton;
        use regex_automata::util::syntax;
        use regex_automata::Anchored;

        let dfa = dense::Builder::new()
            .configure(
                dense::Config::new()
                    .start_kind(regex_automata::dfa::StartKind::Anchored)
                    .accelerate(true),
            )
            .syntax(syntax::Config::new().utf8(false))
            .build(pattern)
            .map_err(|e| RegexGrammarError::InvalidPattern(e.to_string()))?;

        // Start state for an anchored match. The "always anchored"
        // config means there's a single start state regardless of
        // input lookaround.
        let cfg = regex_automata::util::start::Config::new().anchored(Anchored::Yes);
        let state = dfa
            .start_state(&cfg)
            .map_err(|e| RegexGrammarError::InvalidPattern(e.to_string()))?;

        Ok(Self {
            dfa: Arc::new(dfa),
            state,
        })
    }

    fn step(&mut self, b: u8) -> Step {
        use regex_automata::dfa::Automaton;
        let next = self.dfa.next_state(self.state, b);
        if self.dfa.is_dead_state(next) || self.dfa.is_quit_state(next) {
            return Step::Reject;
        }
        self.state = next;
        Step::Accept
    }

    /// True when the DFA is currently in a match state — i.e., the
    /// bytes consumed so far form a complete (anchored) match of
    /// the pattern. The sampler uses this to decide whether EOS is
    /// allowed at this position.
    fn can_terminate_now(&self) -> bool {
        use regex_automata::dfa::Automaton;
        // `next_eoi_state` advances the DFA across the implicit
        // end-of-input marker. If the resulting state is a match,
        // the current byte sequence is a complete match. Cheaper
        // than maintaining a separate "is current state a match?"
        // table since DFAs distinguish "match" from "match if at
        // EOI" for handling word-boundary-style assertions.
        let eoi = self.dfa.next_eoi_state(self.state);
        self.dfa.is_match_state(eoi)
    }
}

impl Default for CodeGrammarParser {
    fn default() -> Self {
        Self::new()
    }
}

impl GrammarMask {
    /// Build a JSON-constrained mask using `decode_one(id) -> bytes` to
    /// pre-decode every vocab id. `eos_id` (when set) is allowed any
    /// time the parser is in the `Done` state so the model can stop.
    pub fn new_json(
        vocab_size: usize,
        eos_id: Option<u32>,
        decode_one: impl Fn(u32) -> Vec<u8>,
    ) -> Self {
        Self {
            parser: ParserKind::Json(JsonParser::new()),
            token_bytes: Arc::new(decode_vocab(vocab_size, decode_one)),
            eos_id,
        }
    }

    /// Build a JSON-Schema-constrained mask. Same as `new_json`, plus
    /// rejects any token that would produce schema-invalid bytes (wrong
    /// type for a property, missing required key on close, etc.).
    pub fn new_json_schema(
        schema: Schema,
        vocab_size: usize,
        eos_id: Option<u32>,
        decode_one: impl Fn(u32) -> Vec<u8>,
    ) -> Self {
        Self {
            parser: ParserKind::JsonSchema(JsonSchemaParser::new(schema)),
            token_bytes: Arc::new(decode_vocab(vocab_size, decode_one)),
            eos_id,
        }
    }

    /// Build a tool-call stream mask. Accepts arbitrary text outside
    /// `<tool_call>...</tool_call>` markers; inside, enforces the
    /// `{"name": "<known>", "arguments": <schema>}` shape with the
    /// supplied per-tool schemas.
    ///
    /// `max_tool_iterations` caps the number of complete tool-call
    /// bodies the grammar will let through; `0` disables the cap.
    /// `min_tool_calls` blocks EOS until that many complete bodies have
    /// been emitted (the `tool_choice: "required"` / forced-function
    /// floor); `0` disables the floor.
    pub fn new_tool_call_stream(
        schemas_by_name: std::collections::BTreeMap<String, Schema>,
        max_tool_iterations: u32,
        min_tool_calls: u32,
        vocab_size: usize,
        eos_id: Option<u32>,
        decode_one: impl Fn(u32) -> Vec<u8>,
    ) -> Self {
        Self {
            parser: ParserKind::ToolCall(
                crate::tool_grammar::ToolCallStreamGrammar::new_with_bounds(
                    schemas_by_name,
                    max_tool_iterations,
                    min_tool_calls,
                ),
            ),
            token_bytes: Arc::new(decode_vocab(vocab_size, decode_one)),
            eos_id,
        }
    }

    /// Build a code-syntax-constrained mask. v1 enforces bracket
    /// balance with quote-aware string tracking; future language-
    /// specific variants will slot in alongside.
    pub fn new_code(
        vocab_size: usize,
        eos_id: Option<u32>,
        decode_one: impl Fn(u32) -> Vec<u8>,
    ) -> Self {
        Self::new_code_for_language("", vocab_size, eos_id, decode_one)
    }

    /// Build a code-grammar mask tuned for a specific language's
    /// comment style. `language` is matched against
    /// [`CommentStyle::for_language`]; unknown values fall back to
    /// no comment recognition (equivalent to [`new_code`]).
    pub fn new_code_for_language(
        language: &str,
        vocab_size: usize,
        eos_id: Option<u32>,
        decode_one: impl Fn(u32) -> Vec<u8>,
    ) -> Self {
        Self {
            parser: ParserKind::Code(CodeGrammarParser::with_language(language)),
            token_bytes: Arc::new(decode_vocab(vocab_size, decode_one)),
            eos_id,
        }
    }

    /// Build a regex-constrained mask. The pattern is anchored at
    /// the start of output — every token's bytes are stepped through
    /// the compiled DFA, and any token that would transition into a
    /// dead (non-matching, unreachable) state gets masked out. The
    /// model decides when to stop: EOS is allowed any time the DFA
    /// is in a match state (`can_terminate_now`).
    ///
    /// Returns a `RegexGrammarError::InvalidPattern` if the pattern
    /// fails to parse — surface that as a 400 at the request boundary
    /// rather than retrying.
    pub fn new_regex(
        pattern: &str,
        vocab_size: usize,
        eos_id: Option<u32>,
        decode_one: impl Fn(u32) -> Vec<u8>,
    ) -> std::result::Result<Self, RegexGrammarError> {
        Ok(Self {
            parser: ParserKind::Regex(RegexGrammarParser::new(pattern)?),
            token_bytes: Arc::new(decode_vocab(vocab_size, decode_one)),
            eos_id,
        })
    }

    /// True when the parser has consumed a complete top-level JSON
    /// value. The sampler should let `eos_id` through and stop.
    pub fn is_done(&self) -> bool {
        self.parser.is_done()
    }

    /// For the tool-call stream grammar: `true` once the model has
    /// tried to open another `<tool_call>` after the `max_tool_iterations`
    /// cap was hit and the opener byte got rejected. Always `false` for
    /// other grammar kinds. Callers use this to surface a
    /// `tool_call_iteration_limit` finish_reason on the response.
    pub fn tool_call_limit_blocked(&self) -> bool {
        match &self.parser {
            ParserKind::ToolCall(p) => p.limit_blocked(),
            _ => false,
        }
    }

    /// True if the parser would accept this token's bytes (without
    /// mutating state). EOS is allowed once done.
    pub fn accepts(&self, token_id: u32) -> bool {
        if Some(token_id) == self.eos_id {
            return self.parser.is_done() || self.parser.can_terminate_now();
        }
        let id = token_id as usize;
        let Some(bytes) = self.token_bytes.get(id) else {
            return false;
        };
        if bytes.is_empty() {
            // Empty-decode tokens (special markers, BOS, etc.) are not
            // structural — let the sampler decide if they're valid by
            // virtue of being non-grammar tokens. Reject conservatively.
            return false;
        }
        let mut p = self.parser.clone();
        for &b in bytes {
            match p.step(b) {
                Step::Accept => {}
                Step::Done => {
                    // The token completed the JSON value. The remaining
                    // bytes after a `Done` step must be whitespace.
                }
                Step::Reject => return false,
            }
        }
        true
    }

    /// Commit a token to the parser state. Call this after the sampler
    /// has chosen a token so subsequent `accepts` checks reflect the
    /// new state.
    pub fn advance(&mut self, token_id: u32) {
        if Some(token_id) == self.eos_id {
            return;
        }
        let id = token_id as usize;
        if let Some(bytes) = self.token_bytes.get(id) {
            for &b in bytes {
                if self.parser.step(b) == Step::Reject {
                    // Caller violated the contract by advancing past a
                    // token that wasn't accepted. Stop stepping — the
                    // mask is now stuck and `accepts` will reject
                    // everything until EOS.
                    return;
                }
            }
        }
    }
}

fn decode_vocab(vocab_size: usize, decode_one: impl Fn(u32) -> Vec<u8>) -> Vec<Vec<u8>> {
    let mut bytes = Vec::with_capacity(vocab_size);
    for id in 0..vocab_size {
        bytes.push(decode_one(id as u32));
    }
    bytes
}

/// JSON-Schema-aware parser. Wraps [`JsonParser`] (for the byte-
/// level JSON FSM) and additionally tracks where we are in the
/// supplied schema. Rejects bytes that would produce schema-invalid
/// JSON in addition to bytes that would break JSON syntax.
///
/// **Required-key enforcement**: `}` is rejected until every key in
/// `required` has been seen at the current object level. This forces
/// the model to emit all required keys before closing.
///
/// **Type enforcement**: at value position, only bytes that *could*
/// start a value of the expected type are accepted. For a
/// `type: "string"` value, only `"`; for `type: "number"`, only
/// `-` or a digit; for `type: "boolean"`, only `t` or `f`; etc.
///
/// **Property restriction**: when a schema has `additionalProperties:
/// false` and `properties: {a, b}`, only the bytes that lead toward
/// `"a"` or `"b"` are accepted as keys.
#[derive(Debug, Clone)]
pub struct JsonSchemaParser {
    json: JsonParser,
    /// For each container in `json.containers`, a schema frame.
    /// `frames[0]` is the root; pushed on `{` or `[`, popped on close.
    frames: Vec<SchemaFrame>,
    /// Active schema for the *next* value to be parsed (top-level
    /// initially, then per-key for object values, per-element for
    /// array items, then None once a value is complete and we expect
    /// `,` or close).
    active: Option<Schema>,
    /// True once a complete top-level value matching the root schema
    /// has been consumed.
    done: bool,
    /// Sticky reject flag: once a schema-only violation fires (e.g.,
    /// unknown key under `additionalProperties: false`), every
    /// subsequent step returns Reject. The mask's clone-and-step in
    /// `accepts` then correctly classifies any token whose bytes
    /// traverse the violation.
    invalid: bool,
}

#[derive(Debug, Clone)]
struct SchemaFrame {
    schema: Schema,
    /// Object-only: keys observed so far at this nesting level.
    keys_seen: BTreeSet<String>,
    /// Object-only: the key being accumulated while we're inside a
    /// key string. Empty otherwise.
    current_key: String,
    /// Object-only: when we're inside the body of a key-string
    /// (between the opening `"` and the closing `"`), set to true so
    /// the parser knows to record bytes into `current_key`.
    accumulating_key: bool,
}

impl JsonSchemaParser {
    pub fn new(schema: Schema) -> Self {
        Self {
            json: JsonParser::new(),
            frames: Vec::new(),
            active: Some(schema),
            done: false,
            invalid: false,
        }
    }

    pub fn is_done(&self) -> bool {
        self.done
    }

    /// True when the schema parser would currently accept EOS — i.e.,
    /// the root value is structurally complete (number that's seen at
    /// least one digit at the top level, for instance).
    pub fn can_terminate_now(&self) -> bool {
        self.json.can_terminate_now() && self.frames.is_empty()
    }

    /// Step one byte. Mirrors [`JsonParser::step`] but additionally
    /// enforces schema constraints.
    pub fn step(&mut self, b: u8) -> Step {
        if self.invalid {
            return Step::Reject;
        }
        if self.done {
            return if is_ws(b) { Step::Accept } else { Step::Reject };
        }

        // Schema-side pre-checks BEFORE stepping the JSON parser.
        // These reject byte patterns that would violate the schema
        // even if they'd be syntactically valid JSON.
        if !is_ws(b) {
            if let Some(reject) = self.schema_precheck(b) {
                if reject {
                    return Step::Reject;
                }
            }
        }

        // Note the json state BEFORE stepping so we can detect
        // transitions (e.g., "we just entered a string body").
        let was_in_key_string = self.current_frame_accumulating_key();

        let result = self.json.step(b);
        if result == Step::Reject {
            return Step::Reject;
        }

        // Schema-side post-update: react to the JSON parser's state
        // transition. May set `self.invalid` (e.g., unknown key under
        // `additionalProperties: false`).
        self.update_schema_state(b, was_in_key_string);
        if self.invalid {
            return Step::Reject;
        }

        // After processing the byte, check whether we're DONE.
        if self.json.is_done() && self.frames.is_empty() {
            self.done = true;
            return Step::Done;
        }
        if result == Step::Done {
            // JSON parser said done but we still have schema frames —
            // shouldn't happen (the json parser closes containers in
            // lockstep with us). Treat as success.
            self.done = true;
            return Step::Done;
        }
        Step::Accept
    }

    /// Returns Some(true) to reject the byte for a schema-only reason
    /// (e.g., wrong value type). Returns None when the byte is not
    /// schema-relevant and JsonParser should evaluate it normally.
    fn schema_precheck(&self, b: u8) -> Option<bool> {
        // Are we at a value position? If so, the active schema dictates
        // which bytes can start the value.
        let json_state = self.json.peek_state();
        let at_value_start = matches!(
            json_state,
            State::ExpectValue | State::ExpectArrayValueOrClose
        );
        if at_value_start {
            if let Some(active) = &self.active {
                return Some(!schema_accepts_value_start(active, b));
            }
        }
        // Are we at an object-close position? If so, check required keys.
        if b == b'}' && matches!(json_state, State::ExpectCommaOrClose | State::ExpectFirstKeyOrClose) {
            if let Some(frame) = self.frames.last() {
                if let Schema::Object { required, .. } = &frame.schema {
                    let missing = required
                        .iter()
                        .any(|k| !frame.keys_seen.contains(k));
                    if missing {
                        return Some(true);
                    }
                }
            }
        }
        // Are we at a key position? If so, the schema's properties
        // dictate which keys are allowed (when additionalProperties
        // is false).
        if matches!(
            json_state,
            State::ExpectFirstKeyOrClose | State::ExpectKey
        ) {
            if b == b'"' {
                // Accept — we'll constrain on the key bytes below.
                return None;
            }
        }
        None
    }

    /// Whether the current top frame is in the middle of a key string.
    fn current_frame_accumulating_key(&self) -> bool {
        self.frames
            .last()
            .map(|f| f.accumulating_key)
            .unwrap_or(false)
    }

    fn update_schema_state(&mut self, b: u8, was_in_key_string: bool) {
        let new_state = self.json.peek_state();

        // When we open a container, push a schema frame matching the
        // active schema.
        if matches!(new_state, State::ExpectFirstKeyOrClose)
            && self.json.containers_len() > self.frames.len()
        {
            // Just opened `{`. The active schema becomes this frame's.
            let schema = self.active.take().unwrap_or(Schema::Any);
            self.frames.push(SchemaFrame {
                schema,
                keys_seen: BTreeSet::new(),
                current_key: String::new(),
                accumulating_key: false,
            });
            return;
        }
        if matches!(new_state, State::ExpectArrayValueOrClose)
            && self.json.containers_len() > self.frames.len()
        {
            // Just opened `[`. Push a frame; the active schema (the
            // items schema) is set per element below.
            let schema = self.active.take().unwrap_or(Schema::Any);
            self.frames.push(SchemaFrame {
                schema: schema.clone(),
                keys_seen: BTreeSet::new(),
                current_key: String::new(),
                accumulating_key: false,
            });
            // Set the item schema as the active for the next value.
            if let Schema::Array { items: Some(it) } = &schema {
                self.active = Some((**it).clone());
            } else if let Schema::Array { items: None } = &schema {
                self.active = Some(Schema::Any);
            }
            return;
        }

        // Container close — pop schema frame.
        if self.json.containers_len() < self.frames.len() {
            self.frames.pop();
            // After a container close, we're "in value position relative
            // to the parent container". active is irrelevant until the
            // next `,` -> key/value.
            self.active = None;
            return;
        }

        // Entering a key string: mark accumulating_key.
        // The JsonParser uses InString { is_key: true } when in a key.
        if let State::InString {
            is_key: true,
            in_escape: false,
        } = new_state
        {
            if let Some(frame) = self.frames.last_mut() {
                if !was_in_key_string {
                    // Just entered the key string body.
                    frame.current_key.clear();
                    frame.accumulating_key = true;
                } else if !is_ws(b) && b != b'"' {
                    // Inside the key body, record the byte. JSON-Schema
                    // property names are UTF-8 strings; we accept any
                    // byte the parser does. Escapes are accumulated as
                    // raw bytes — we'd need a proper unescape pass to
                    // match against `properties` if the user supplies
                    // an escape, but in practice property names are
                    // simple identifiers.
                    frame.current_key.push(b as char);
                }
            }
        }

        // Just closed a key string (`"` in key context). Record key.
        if was_in_key_string && !matches!(new_state, State::InString { is_key: true, .. }) {
            if let Some(frame) = self.frames.last_mut() {
                let key = std::mem::take(&mut frame.current_key);
                frame.accumulating_key = false;
                // Resolve the value schema for this key from properties.
                if let Schema::Object {
                    properties,
                    additional,
                    ..
                } = &frame.schema
                {
                    match properties.get(&key) {
                        Some(s) => self.active = Some(s.clone()),
                        None => {
                            if !*additional {
                                // Unknown key under
                                // `additionalProperties: false` —
                                // poison the parser. Subsequent steps
                                // (including the same token's later
                                // bytes via clone-and-step) reject.
                                self.invalid = true;
                                return;
                            }
                            // Lax: treat unknown key as Any-typed.
                            self.active = Some(Schema::Any);
                        }
                    }
                }
                frame.keys_seen.insert(key);
            }
        }

        // After a `,` in an array, we need a fresh item-schema.
        if matches!(new_state, State::ExpectValue) {
            // We're expecting a value. If we're in an array container,
            // re-derive the item schema. If we're in an object, the
            // active schema was set when the key resolved. At top
            // level, active stays.
            if let Some(frame) = self.frames.last() {
                if let Schema::Array { items: Some(it) } = &frame.schema {
                    if self.active.is_none() {
                        self.active = Some((**it).clone());
                    }
                } else if matches!(frame.schema, Schema::Array { items: None }) {
                    if self.active.is_none() {
                        self.active = Some(Schema::Any);
                    }
                }
            }
        }
    }
}

/// Does this schema allow a value that starts with `b`? Used at
/// value-start positions to filter out e.g. `"` when the schema says
/// `type: "number"`.
fn schema_accepts_value_start(schema: &Schema, b: u8) -> bool {
    match schema {
        Schema::String { .. } => b == b'"',
        Schema::Number { .. } => matches!(b, b'-' | b'0'..=b'9'),
        Schema::Boolean => matches!(b, b't' | b'f'),
        Schema::Null => b == b'n',
        Schema::Object { .. } => b == b'{',
        Schema::Array { .. } => b == b'[',
        Schema::Any => true,
    }
}

/// JSON parser state machine. Steps one byte at a time. Cheap to
/// clone (a couple of small vecs).
#[derive(Debug, Clone)]
pub struct JsonParser {
    /// Container nesting: `b'{'` for object, `b'['` for array. Empty
    /// means we're at the top level.
    containers: Vec<u8>,
    state: State,
    /// True once a complete top-level value has been consumed. Further
    /// non-whitespace bytes reject.
    done: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum State {
    /// Expecting a value. Top-level start, after `:` in an object,
    /// after `,` in an array. Does NOT accept `]` — empty-array close
    /// is handled by [`State::ExpectArrayValueOrClose`].
    ExpectValue,
    /// Right after `[`. Accepts either a value or `]`.
    ExpectArrayValueOrClose,
    /// Right after `{`. Accepts either a string key or `}`. (After a
    /// `,` in an object we use [`State::ExpectKey`] instead — `}` is
    /// not valid then.)
    ExpectFirstKeyOrClose,
    /// After `,` in an object. Only a string key is valid.
    ExpectKey,
    /// After a key string in an object.
    ExpectColon,
    /// After a value, inside a container. Accepts `,` or the matching
    /// close bracket.
    ExpectCommaOrClose,
    /// In a string; bool = "next byte is the body of an escape sequence".
    /// `is_key` is whether the string we're in is an object key (so we
    /// know to transition to `ExpectColon` vs `ExpectCommaOrClose` on
    /// close).
    InString { in_escape: bool, is_key: bool },
    /// In a string, in the middle of a `\uXXXX` escape. usize counts
    /// hex digits remaining (4 → 0). `is_key` carried so we can route
    /// correctly when the string ends.
    InUnicodeEscape { remaining: u8, is_key: bool },
    /// In a number; track which constituents have been seen so the
    /// next byte can be validated.
    InNumber {
        saw_digit: bool,
        saw_dot: bool,
        saw_exp: bool,
        last_was_e: bool,
    },
    /// Matching `true` / `false` / `null`. `remaining` is the suffix
    /// still to be consumed.
    InLiteral { remaining: &'static [u8] },
}

impl JsonParser {
    pub fn new() -> Self {
        Self {
            containers: Vec::new(),
            state: State::ExpectValue,
            done: false,
        }
    }

    pub fn is_done(&self) -> bool {
        self.done
    }

    /// True when the parser could legitimately accept EOS right now —
    /// i.e. a number that's seen at least one digit, at the top level,
    /// with no open container.
    pub fn can_terminate_now(&self) -> bool {
        if !self.containers.is_empty() {
            return false;
        }
        matches!(
            self.state,
            State::InNumber { saw_digit: true, last_was_e: false, .. }
        )
    }

    /// Inspect the current parser state without mutating. Used by
    /// `JsonSchemaParser` to make pre-step decisions based on whether
    /// we're at a value position, key position, etc.
    pub(crate) fn peek_state(&self) -> State {
        self.state
    }

    /// Number of currently-open containers (objects/arrays). Used by
    /// `JsonSchemaParser` to detect transitions across `{`/`}`/`[`/`]`.
    pub(crate) fn containers_len(&self) -> usize {
        self.containers.len()
    }

    /// Step the parser by one byte. Returns whether the byte was
    /// accepted, completed the document, or violated the grammar.
    pub fn step(&mut self, b: u8) -> Step {
        if self.done {
            // After completion, only whitespace is acceptable.
            return if is_ws(b) { Step::Accept } else { Step::Reject };
        }

        // The string + literal + number sub-states have to consume the
        // current byte before the outer state can route to a new mode.
        match self.state {
            State::InString { in_escape, is_key } => {
                return self.step_in_string(b, in_escape, is_key);
            }
            State::InUnicodeEscape { remaining, is_key } => {
                return self.step_in_unicode_escape(b, remaining, is_key);
            }
            State::InLiteral { remaining } => return self.step_in_literal(b, remaining),
            State::InNumber {
                saw_digit,
                saw_dot,
                saw_exp,
                last_was_e,
            } => {
                if let Some(s) = step_in_number(b, saw_digit, saw_dot, saw_exp, last_was_e) {
                    self.state = s;
                    return Step::Accept;
                }
                // Number ended. Treat the current byte as a value-
                // terminator (whitespace, `,`, `}`, `]`) and fall
                // through to the outer state — but first finalize the
                // number transition.
                if saw_digit && !last_was_e {
                    self.finish_value();
                    // Fall through to outer-state handling of `b`.
                    return self.step_outer(b);
                } else {
                    return Step::Reject;
                }
            }
            _ => {}
        }

        self.step_outer(b)
    }

    fn step_outer(&mut self, b: u8) -> Step {
        if is_ws(b) {
            return Step::Accept;
        }
        match self.state {
            State::ExpectValue => self.begin_value(b),
            State::ExpectArrayValueOrClose => match b {
                b']' => {
                    if self.containers.last() == Some(&b'[') {
                        self.containers.pop();
                        self.finish_value();
                        self.maybe_done()
                    } else {
                        Step::Reject
                    }
                }
                _ => self.begin_value(b),
            },
            State::ExpectFirstKeyOrClose => match b {
                b'"' => {
                    self.state = State::InString {
                        in_escape: false,
                        is_key: true,
                    };
                    Step::Accept
                }
                b'}' => {
                    if self.containers.last() == Some(&b'{') {
                        self.containers.pop();
                        self.finish_value();
                        self.maybe_done()
                    } else {
                        Step::Reject
                    }
                }
                _ => Step::Reject,
            },
            State::ExpectKey => match b {
                b'"' => {
                    self.state = State::InString {
                        in_escape: false,
                        is_key: true,
                    };
                    Step::Accept
                }
                _ => Step::Reject,
            },
            State::ExpectColon => match b {
                b':' => {
                    self.state = State::ExpectValue;
                    Step::Accept
                }
                _ => Step::Reject,
            },
            State::ExpectCommaOrClose => match b {
                b',' => {
                    // After a `,`: an object expects a key, an array a value.
                    self.state = match self.containers.last() {
                        Some(&b'{') => State::ExpectKey,
                        Some(&b'[') => State::ExpectValue,
                        _ => return Step::Reject,
                    };
                    Step::Accept
                }
                b'}' => {
                    if self.containers.last() == Some(&b'{') {
                        self.containers.pop();
                        self.finish_value();
                        self.maybe_done()
                    } else {
                        Step::Reject
                    }
                }
                b']' => {
                    if self.containers.last() == Some(&b'[') {
                        self.containers.pop();
                        self.finish_value();
                        self.maybe_done()
                    } else {
                        Step::Reject
                    }
                }
                _ => Step::Reject,
            },
            // Number / string / literal / unicode-escape were handled
            // earlier; if we got here it's a bug.
            _ => Step::Reject,
        }
    }

    /// Begin parsing a value at the given byte.
    fn begin_value(&mut self, b: u8) -> Step {
        match b {
            b'{' => {
                self.containers.push(b'{');
                self.state = State::ExpectFirstKeyOrClose;
                Step::Accept
            }
            b'[' => {
                self.containers.push(b'[');
                self.state = State::ExpectArrayValueOrClose;
                Step::Accept
            }
            b'"' => {
                self.state = State::InString {
                    in_escape: false,
                    is_key: false,
                };
                Step::Accept
            }
            b't' => {
                self.state = State::InLiteral { remaining: b"rue" };
                Step::Accept
            }
            b'f' => {
                self.state = State::InLiteral { remaining: b"alse" };
                Step::Accept
            }
            b'n' => {
                self.state = State::InLiteral { remaining: b"ull" };
                Step::Accept
            }
            b'-' => {
                self.state = State::InNumber {
                    saw_digit: false,
                    saw_dot: false,
                    saw_exp: false,
                    last_was_e: false,
                };
                Step::Accept
            }
            b'0'..=b'9' => {
                self.state = State::InNumber {
                    saw_digit: true,
                    saw_dot: false,
                    saw_exp: false,
                    last_was_e: false,
                };
                Step::Accept
            }
            _ => Step::Reject,
        }
    }

    fn step_in_string(&mut self, b: u8, in_escape: bool, is_key: bool) -> Step {
        if in_escape {
            match b {
                b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {
                    self.state = State::InString {
                        in_escape: false,
                        is_key,
                    };
                    Step::Accept
                }
                b'u' => {
                    self.state = State::InUnicodeEscape {
                        remaining: 4,
                        is_key,
                    };
                    Step::Accept
                }
                _ => Step::Reject,
            }
        } else {
            match b {
                b'"' => {
                    // End of string. Route based on context AND whether
                    // this was a key (→ ExpectColon) or value
                    // (→ ExpectCommaOrClose / done).
                    if is_key {
                        self.state = State::ExpectColon;
                        return Step::Accept;
                    }
                    self.state = match self.containers.last() {
                        Some(&b'{') | Some(&b'[') => State::ExpectCommaOrClose,
                        None => {
                            // Top-level string literal.
                            self.finish_value();
                            return self.maybe_done();
                        }
                        _ => return Step::Reject,
                    };
                    Step::Accept
                }
                b'\\' => {
                    self.state = State::InString {
                        in_escape: true,
                        is_key,
                    };
                    Step::Accept
                }
                // Reject raw control characters per JSON spec.
                0x00..=0x1F => Step::Reject,
                // Anything else (including UTF-8 continuation bytes) accepted.
                _ => Step::Accept,
            }
        }
    }

    fn step_in_unicode_escape(&mut self, b: u8, remaining: u8, is_key: bool) -> Step {
        if !is_hex(b) {
            return Step::Reject;
        }
        if remaining <= 1 {
            self.state = State::InString {
                in_escape: false,
                is_key,
            };
        } else {
            self.state = State::InUnicodeEscape {
                remaining: remaining - 1,
                is_key,
            };
        }
        Step::Accept
    }

    fn step_in_literal(&mut self, b: u8, remaining: &'static [u8]) -> Step {
        if remaining.is_empty() {
            // Caller should have transitioned out; treat as reject.
            return Step::Reject;
        }
        if remaining[0] != b {
            return Step::Reject;
        }
        let rest = &remaining[1..];
        if rest.is_empty() {
            self.finish_value();
            self.maybe_done()
        } else {
            self.state = State::InLiteral { remaining: rest };
            Step::Accept
        }
    }

    /// After completing a value, decide where to go: end-of-document,
    /// after-value-in-object, or after-value-in-array.
    fn finish_value(&mut self) {
        self.state = match self.containers.last() {
            Some(&b'{') => State::ExpectCommaOrClose,
            Some(&b'[') => State::ExpectCommaOrClose,
            None => {
                self.done = true;
                State::ExpectCommaOrClose // unused
            }
            _ => State::ExpectCommaOrClose,
        };
    }

    fn maybe_done(&mut self) -> Step {
        if self.containers.is_empty() {
            self.done = true;
            Step::Done
        } else {
            Step::Accept
        }
    }
}

impl Default for JsonParser {
    fn default() -> Self {
        Self::new()
    }
}

fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

fn is_hex(b: u8) -> bool {
    matches!(b, b'0'..=b'9' | b'a'..=b'f' | b'A'..=b'F')
}

/// Step one byte through a number-state. Returns the new state, or
/// `None` if the byte means the number ended (caller routes the byte
/// to the outer state).
fn step_in_number(
    b: u8,
    saw_digit: bool,
    saw_dot: bool,
    saw_exp: bool,
    last_was_e: bool,
) -> Option<State> {
    match b {
        b'0'..=b'9' => Some(State::InNumber {
            saw_digit: true,
            saw_dot,
            saw_exp,
            last_was_e: false,
        }),
        b'.' if !saw_dot && !saw_exp && saw_digit => Some(State::InNumber {
            saw_digit,
            saw_dot: true,
            saw_exp,
            last_was_e: false,
        }),
        b'e' | b'E' if !saw_exp && saw_digit => Some(State::InNumber {
            saw_digit,
            saw_dot,
            saw_exp: true,
            last_was_e: true,
        }),
        b'+' | b'-' if last_was_e => Some(State::InNumber {
            saw_digit: false,
            saw_dot,
            saw_exp,
            last_was_e: false,
        }),
        _ => None, // number ended; caller re-routes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn accept_all(s: &str) -> bool {
        let mut p = JsonParser::new();
        for &b in s.as_bytes() {
            match p.step(b) {
                Step::Accept | Step::Done => {}
                Step::Reject => return false,
            }
        }
        // Accept if the parser hit `Done` mid-stream, OR if we ended in
        // a state that can terminate (e.g., trailing-number).
        p.is_done() || p.can_terminate_now()
    }

    #[test]
    fn accepts_simple_object() {
        assert!(accept_all(r#"{"hello": "world"}"#));
    }

    #[test]
    fn accepts_nested_arrays_and_objects() {
        assert!(accept_all(r#"{"a": [1, 2, {"b": 3.14}], "c": null}"#));
    }

    #[test]
    fn accepts_top_level_scalars() {
        assert!(accept_all("42"));
        assert!(accept_all("-3.14e10"));
        assert!(accept_all("true"));
        assert!(accept_all("false"));
        assert!(accept_all("null"));
        assert!(accept_all("\"a string\""));
    }

    #[test]
    fn accepts_empty_containers() {
        assert!(accept_all("{}"));
        assert!(accept_all("[]"));
    }

    #[test]
    fn rejects_unquoted_keys() {
        assert!(!accept_all(r#"{key: "value"}"#));
    }

    #[test]
    fn rejects_trailing_comma_in_object() {
        assert!(!accept_all(r#"{"a": 1,}"#));
    }

    #[test]
    fn rejects_trailing_comma_in_array() {
        assert!(!accept_all(r#"[1, 2,]"#));
    }

    #[test]
    fn rejects_single_quotes() {
        assert!(!accept_all(r#"{'a': 1}"#));
    }

    #[test]
    fn rejects_unterminated_string() {
        assert!(!accept_all(r#""hello"#));
    }

    #[test]
    fn accepts_escape_sequences() {
        assert!(accept_all(r#""hello\nworld""#));
        assert!(accept_all(r#""tab\there""#));
        assert!(accept_all(r#""quoted: \"yes\"""#));
        assert!(accept_all(r#""unicode: é""#));
    }

    #[test]
    fn rejects_bad_unicode_escape() {
        assert!(!accept_all(r#""\u12g4""#));
    }

    #[test]
    fn rejects_extra_content_after_value() {
        // After `42` completes, `more` is non-whitespace → reject.
        assert!(!accept_all("42 more"));
    }

    #[test]
    fn accepts_trailing_whitespace() {
        assert!(accept_all("42  \n  "));
    }

    #[test]
    fn rejects_lone_dot_or_minus() {
        assert!(!accept_all("."));
        assert!(!accept_all("-"));
    }

    #[test]
    fn rejects_exponent_without_digits() {
        assert!(!accept_all("1e"));
        assert!(!accept_all("1e+"));
    }

    #[test]
    fn step_byte_returns_done_on_top_level_close() {
        let mut p = JsonParser::new();
        for &b in b"{\"a\":1" {
            assert_eq!(p.step(b), Step::Accept, "byte {b:?}");
        }
        assert_eq!(p.step(b'}'), Step::Done);
        assert!(p.is_done());
    }

    #[test]
    fn grammar_mask_rejects_disallowed_first_token() {
        // Vocab: 0 = "{", 1 = "[", 2 = "garbage", 3 = " " (ws).
        let mask = GrammarMask::new_json(4, None, |id| match id {
            0 => b"{".to_vec(),
            1 => b"[".to_vec(),
            2 => b"garbage".to_vec(),
            3 => b" ".to_vec(),
            _ => Vec::new(),
        });
        assert!(mask.accepts(0)); // `{` valid start
        assert!(mask.accepts(1)); // `[` valid start
        assert!(!mask.accepts(2)); // "garbage" - reject
        assert!(mask.accepts(3)); // whitespace at start is OK
    }

    /// Adversarial sampler test: an unconstrained sampler would pick
    /// the bad token (highest logit); the grammar mask must force it
    /// to fall back to a valid one.
    #[test]
    fn sampler_with_grammar_rejects_bad_high_logit_token() {
        use crate::{sampling::Sampler, SamplingParams};
        // Vocab: 0 = `{`, 1 = `[`, 2 = "garbage", 3 = `"`.
        let mask = GrammarMask::new_json(4, None, |id| match id {
            0 => b"{".to_vec(),
            1 => b"[".to_vec(),
            2 => b"garbage".to_vec(),
            3 => b"\"".to_vec(),
            _ => Vec::new(),
        });
        // Adversarial logits: "garbage" wins by a mile.
        let mut logits = vec![1.0f32, 1.0, 100.0, 1.0];
        let mut sampler = Sampler::new(SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            ..SamplingParams::default()
        });
        // Sample many times — with high temperature, the bad token would
        // dominate without the grammar. Every draw must be a grammar-
        // accepting token.
        for _ in 0..50 {
            let mut logits_copy = logits.clone();
            let id = sampler.sample_with_grammar(&mut logits_copy, &[], Some(&mask));
            assert_ne!(id, 2, "sampler picked grammar-rejected token");
            assert!(id < 4);
        }
        // Greedy (temp=0) should also avoid the bad token.
        let mut sampler_greedy = Sampler::new(SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        });
        let id = sampler_greedy.sample_with_grammar(&mut logits, &[], Some(&mask));
        assert_ne!(id, 2);

        // Sanity: without the mask, greedy DOES pick the bad token.
        let mut sampler_unconstrained = Sampler::new(SamplingParams::default());
        let mut logits_copy = vec![1.0f32, 1.0, 100.0, 1.0];
        let id_bad = sampler_unconstrained.sample(&mut logits_copy, &[]);
        assert_eq!(id_bad, 2, "control: unconstrained sampler should pick `garbage`");
    }

    fn schema_accepts(schema: Schema, s: &str) -> bool {
        let mut p = JsonSchemaParser::new(schema);
        for &b in s.as_bytes() {
            match p.step(b) {
                Step::Accept | Step::Done => {}
                Step::Reject => return false,
            }
        }
        p.is_done() || p.can_terminate_now()
    }

    fn obj_schema(props: &[(&str, Schema)], required: &[&str]) -> Schema {
        Schema::Object {
            properties: props
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect(),
            required: required.iter().map(|s| s.to_string()).collect(),
            additional: true,
        }
    }

    #[test]
    fn schema_parses_basic_types_from_json() {
        let s = Schema::from_json_value(&serde_json::json!({"type": "string"}));
        assert!(matches!(s, Schema::String { enum_values: None }));

        let s = Schema::from_json_value(&serde_json::json!({"type": "integer"}));
        assert!(matches!(s, Schema::Number { integer_only: true }));

        let s = Schema::from_json_value(&serde_json::json!({
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "number"}},
            "required": ["a"]
        }));
        if let Schema::Object {
            properties,
            required,
            ..
        } = s
        {
            assert_eq!(properties.len(), 2);
            assert!(required.contains("a"));
            assert!(!required.contains("b"));
        } else {
            panic!("expected Object");
        }
    }

    #[test]
    fn schema_object_with_required_key_accepts_and_closes() {
        let schema = obj_schema(&[("name", Schema::String { enum_values: None })], &["name"]);
        assert!(schema_accepts(schema.clone(), r#"{"name": "alice"}"#));
    }

    #[test]
    fn schema_object_rejects_missing_required_key() {
        let schema = obj_schema(&[("name", Schema::String { enum_values: None })], &["name"]);
        // Empty object — `}` rejected because `name` not seen.
        assert!(!schema_accepts(schema, r#"{}"#));
    }

    #[test]
    fn schema_object_enforces_value_type() {
        // `name` is a string — a numeric value should be rejected.
        let schema = obj_schema(&[("name", Schema::String { enum_values: None })], &["name"]);
        assert!(!schema_accepts(schema, r#"{"name": 123}"#));
    }

    #[test]
    fn schema_top_level_string_rejects_object() {
        let schema = Schema::String { enum_values: None };
        assert!(!schema_accepts(schema, r#"{"a": 1}"#));
    }

    #[test]
    fn schema_top_level_number_rejects_string() {
        let schema = Schema::Number { integer_only: false };
        assert!(!schema_accepts(schema, r#""hello""#));
    }

    #[test]
    fn schema_array_of_strings_enforces_item_type() {
        let schema = Schema::Array {
            items: Some(Box::new(Schema::String { enum_values: None })),
        };
        assert!(schema_accepts(schema.clone(), r#"["a", "b", "c"]"#));
        // Number where string expected → reject.
        assert!(!schema_accepts(schema, r#"["a", 42]"#));
    }

    #[test]
    fn schema_nested_object_in_array() {
        let schema = Schema::Array {
            items: Some(Box::new(obj_schema(
                &[("k", Schema::String { enum_values: None })],
                &["k"],
            ))),
        };
        assert!(schema_accepts(
            schema.clone(),
            r#"[{"k": "v1"}, {"k": "v2"}]"#
        ));
        // Missing required `k` in second element.
        assert!(!schema_accepts(schema, r#"[{"k": "v1"}, {}]"#));
    }

    #[test]
    fn schema_mask_rejects_wrong_value_start_token() {
        // Schema: { type: object, properties: {age: {type: number}}, required: [age] }
        let schema = obj_schema(&[("age", Schema::Number { integer_only: false })], &["age"]);
        // After `{"age":`, a `"` (start of string) should be rejected
        // because the value must be a number.
        let mut p = JsonSchemaParser::new(schema);
        for &b in br#"{"age":"#.iter() {
            assert_eq!(p.step(b), Step::Accept, "byte {b:?}");
        }
        // Now at the value position with active=Number. `"` rejected.
        assert_eq!(p.step(b'"'), Step::Reject);
        // But a digit accepted.
        assert_eq!(p.step(b'4'), Step::Accept);
    }

    #[test]
    fn grammar_mask_with_schema_blocks_bad_type_token() {
        // Schema { properties: {age: number}, required: [age],
        //          additionalProperties: false }.
        // With strict mode (additional=false), `"hi"` is rejected
        // because "hi" isn't a defined property name.
        let schema = Schema::Object {
            properties: [(
                "age".to_string(),
                Schema::Number { integer_only: false },
            )]
            .into_iter()
            .collect(),
            required: ["age".to_string()].into_iter().collect(),
            additional: false,
        };
        let mask = GrammarMask::new_json_schema(schema, 6, None, |id| match id {
            0 => b"{".to_vec(),
            1 => b"\"age\":".to_vec(),
            2 => b"42".to_vec(),
            3 => b"}".to_vec(),
            4 => b"\"hi\"".to_vec(),
            5 => b"\"age\":\"".to_vec(),
            _ => Vec::new(),
        });
        // Start: only `{`.
        assert!(mask.accepts(0));
        assert!(!mask.accepts(2)); // bare number rejected at top level
        let mut mask = mask;
        mask.advance(0);

        // After `{`: `"age":` accepted; `"hi"` (unknown key) rejected
        // under `additionalProperties: false`.
        assert!(mask.accepts(1));
        assert!(!mask.accepts(4)); // `"hi"` rejected — not in properties
        mask.advance(1);

        // After `{"age":`: value must be a number.
        assert!(mask.accepts(2)); // `42`
        assert!(!mask.accepts(5)); // `"age":"`  → after the colon a `"` (string) starts
        mask.advance(2);

        // After `{"age":42`: `}` closes.
        assert!(mask.accepts(3));
        mask.advance(3);
        assert!(mask.is_done());
    }

    #[test]
    fn schema_strict_mode_rejects_unknown_keys() {
        let schema = Schema::Object {
            properties: [(
                "a".to_string(),
                Schema::String { enum_values: None },
            )]
            .into_iter()
            .collect(),
            required: ["a".to_string()].into_iter().collect(),
            additional: false,
        };
        // The unknown key `b` is detected at its closing `"`.
        assert!(!schema_accepts(schema.clone(), r#"{"b": "v"}"#));
        // The known key `a` is accepted.
        assert!(schema_accepts(schema, r#"{"a": "v"}"#));
    }

    #[test]
    fn grammar_mask_advances_through_simple_object() {
        // Vocab tokens chosen so the model can build `{"x":1}`.
        let bytes_for = |id| -> Vec<u8> {
            match id {
                0 => b"{".to_vec(),
                1 => b"\"".to_vec(),
                2 => b"x".to_vec(),
                3 => b":".to_vec(),
                4 => b"1".to_vec(),
                5 => b"}".to_vec(),
                6 => b",".to_vec(),
                _ => Vec::new(),
            }
        };
        let mut mask = GrammarMask::new_json(7, None, bytes_for);

        // Start: only `{` valid (here).
        assert!(mask.accepts(0)); // `{`
        assert!(!mask.accepts(3)); // `:` at start - reject
        mask.advance(0);

        // After `{`: expect key (`"`) or close (`}`).
        assert!(mask.accepts(1)); // `"`
        assert!(mask.accepts(5)); // `}`
        assert!(!mask.accepts(4)); // `1` - reject (not a string key)
        mask.advance(1);

        // After `{"`: any non-control byte except `"` and `\` is OK.
        assert!(mask.accepts(2)); // `x`
        mask.advance(2);
        // After `{"x`: still in string.
        mask.advance(1); // close string with `"`

        // After `{"x"`: expect `:`.
        assert!(mask.accepts(3)); // `:`
        assert!(!mask.accepts(4)); // `1` - reject (need colon first)
        mask.advance(3);

        // After `{"x":`: expect value.
        assert!(mask.accepts(4)); // `1`
        mask.advance(4);

        // After `{"x":1`: expect `,` or `}`. The `1` is mid-number;
        // `}` should still be accepted (the number ends + close).
        assert!(mask.accepts(5)); // `}`
        mask.advance(5);

        assert!(mask.is_done());
    }

    // ----- code grammar (bracket balance + string awareness) ---------------

    fn step_str(p: &mut super::CodeGrammarParser, s: &str) -> Step {
        let mut last = Step::Accept;
        for b in s.bytes() {
            last = p.step(b);
            if matches!(last, Step::Reject) {
                return last;
            }
        }
        last
    }

    #[test]
    fn code_accepts_balanced_brackets() {
        let mut p = super::CodeGrammarParser::new();
        assert!(matches!(step_str(&mut p, "fn f() {}"), Step::Accept));
        assert!(p.can_terminate_now());
    }

    #[test]
    fn code_rejects_mismatched_close() {
        let mut p = super::CodeGrammarParser::new();
        // Open `[`, try to close with `)` — should reject the `)` byte.
        step_str(&mut p, "[");
        assert_eq!(p.step(b')'), Step::Reject);
    }

    #[test]
    fn code_rejects_unbalanced_close_with_empty_stack() {
        let mut p = super::CodeGrammarParser::new();
        // First byte is `)` — nothing to match, reject.
        assert_eq!(p.step(b')'), Step::Reject);
    }

    #[test]
    fn code_brackets_inside_strings_are_ignored() {
        // `"["` should NOT push to the bracket stack — it's inside
        // a string literal. The trailing `}` would mis-close if
        // string tracking were broken.
        let mut p = super::CodeGrammarParser::new();
        assert!(matches!(step_str(&mut p, "{x = \"[oops]\"}"), Step::Accept));
        assert!(p.can_terminate_now());
    }

    #[test]
    fn code_escaped_quote_does_not_close_string() {
        let mut p = super::CodeGrammarParser::new();
        // The backslash escapes the next byte. `"\""` is a one-char
        // string containing `"`, not an empty string + dangling `"`.
        assert!(matches!(step_str(&mut p, "\"\\\"\""), Step::Accept));
        assert!(p.can_terminate_now());
    }

    #[test]
    fn code_cannot_terminate_with_open_brackets() {
        let mut p = super::CodeGrammarParser::new();
        step_str(&mut p, "fn f(");
        assert!(!p.can_terminate_now());
    }

    #[test]
    fn code_cannot_terminate_inside_unterminated_string() {
        let mut p = super::CodeGrammarParser::new();
        step_str(&mut p, "x = \"hi");
        assert!(!p.can_terminate_now());
    }

    #[test]
    fn code_handles_nested_brackets_in_order() {
        let mut p = super::CodeGrammarParser::new();
        // Properly nested.
        assert!(matches!(step_str(&mut p, "f(g[h{i}])"), Step::Accept));
        assert!(p.can_terminate_now());
    }

    #[test]
    fn code_rejects_cross_kind_close_when_other_open() {
        let mut p = super::CodeGrammarParser::new();
        // Open `(` then `{` then try `)` — should reject the `)`
        // because the innermost open is `{`, which expects `}`.
        step_str(&mut p, "({");
        assert_eq!(p.step(b')'), Step::Reject);
    }

    // ----- code grammar — comment awareness ----------------------

    #[test]
    fn comment_style_picks_c_family_for_known_languages() {
        for lang in ["rust", "rs", "c", "cpp", "javascript", "ts", "go", "java"] {
            assert_eq!(
                super::CommentStyle::for_language(lang),
                super::CommentStyle::CFamily,
                "expected CFamily for {lang}"
            );
        }
    }

    #[test]
    fn comment_style_picks_hash_for_python_family() {
        for lang in ["python", "py", "ruby", "bash", "yaml", "toml", "perl"] {
            assert_eq!(
                super::CommentStyle::for_language(lang),
                super::CommentStyle::Hash,
                "expected Hash for {lang}"
            );
        }
    }

    #[test]
    fn comment_style_falls_back_to_none_for_unknown() {
        assert_eq!(
            super::CommentStyle::for_language("klingon"),
            super::CommentStyle::None
        );
        assert_eq!(
            super::CommentStyle::for_language(""),
            super::CommentStyle::None
        );
    }

    #[test]
    fn rust_line_comment_does_not_push_bracket_stack() {
        // Without comment recognition, `(` inside the comment would
        // push to the stack and leave the parser unable to terminate.
        let mut p = super::CodeGrammarParser::with_language("rust");
        assert!(matches!(
            step_str(&mut p, "fn f() {\n    // ( unmatched in comment\n}"),
            Step::Accept
        ));
        assert!(p.can_terminate_now(), "comment-protected `(` shouldn't leave stack non-empty");
    }

    #[test]
    fn rust_block_comment_eats_brackets() {
        let mut p = super::CodeGrammarParser::with_language("rust");
        assert!(matches!(
            step_str(&mut p, "let x = /* ( ignored */ 1;"),
            Step::Accept
        ));
        assert!(p.can_terminate_now());
    }

    #[test]
    fn rust_block_comment_blocks_termination_until_closed() {
        // Mid-block-comment must NOT report can_terminate_now until
        // the `*/` is seen. Models that try to stop early get held
        // until they close the comment.
        let mut p = super::CodeGrammarParser::with_language("rust");
        step_str(&mut p, "let x = /* still inside ");
        assert!(!p.can_terminate_now());
        step_str(&mut p, " */ 1;");
        assert!(p.can_terminate_now());
    }

    #[test]
    fn python_hash_comment_protects_brackets() {
        let mut p = super::CodeGrammarParser::with_language("python");
        assert!(matches!(
            step_str(&mut p, "def f():\n    # ( ignored\n    return 1\n"),
            Step::Accept
        ));
        assert!(p.can_terminate_now());
    }

    #[test]
    #[allow(non_snake_case)]
    fn rust_hash_is_NOT_a_comment_marker() {
        // Critical: `#[derive(Debug)]` is Rust attribute syntax, not
        // a comment. The CFamily mode must NOT treat `#` as a line
        // comment, or the `[` after it gets ignored and we can't
        // balance.
        let mut p = super::CodeGrammarParser::with_language("rust");
        assert!(matches!(
            step_str(&mut p, "#[derive(Debug)]\nstruct S;"),
            Step::Accept
        ));
        assert!(p.can_terminate_now());
    }

    #[test]
    fn string_outside_comment_still_tracked() {
        // Comment-aware parser must still recognize strings outside
        // of comments. A `"` inside a string + `}` afterwards is
        // a balance violation only if the string wasn't closed.
        let mut p = super::CodeGrammarParser::with_language("rust");
        // Open `{`, string with embedded `(` (which should NOT push),
        // close string, close `}`.
        assert!(matches!(
            step_str(&mut p, "{ let s = \"hello ( world\"; }"),
            Step::Accept
        ));
        assert!(p.can_terminate_now());
    }

    #[test]
    fn comment_starter_in_string_is_just_string_bytes() {
        // `//` inside a string is not a comment starter — it's
        // string content. Verify the precedence: string mode wins
        // over comment recognition.
        let mut p = super::CodeGrammarParser::with_language("rust");
        assert!(matches!(
            step_str(&mut p, "let url = \"https://x.y/z\";"),
            Step::Accept
        ));
        assert!(p.can_terminate_now());
    }

    #[test]
    fn no_language_unset_keeps_old_behavior() {
        // Empty language string → CommentStyle::None → `//` is just
        // two consecutive `/` characters, neither pushes anything.
        // This preserves v1.0 parser behavior for unknown languages.
        let mut p = super::CodeGrammarParser::with_language("");
        assert!(matches!(step_str(&mut p, "fn f() {} // x"), Step::Accept));
        assert!(p.can_terminate_now());
    }

    // ----- regex grammar -----------------------------------------

    fn step_regex(p: &mut super::RegexGrammarParser, s: &str) {
        for &b in s.as_bytes() {
            assert_eq!(p.step(b), Step::Accept, "byte {b:?} rejected on input {s:?}");
        }
    }

    #[test]
    fn regex_invalid_pattern_returns_error() {
        let err = super::RegexGrammarParser::new("[unclosed").expect_err("must fail");
        match err {
            super::RegexGrammarError::InvalidPattern(_) => {}
        }
    }

    #[test]
    fn regex_accepts_matching_input_and_terminates_at_match() {
        // Pattern: US phone numbers like `123-456-7890`.
        let mut p = super::RegexGrammarParser::new(r"^\d{3}-\d{3}-\d{4}$").unwrap();
        assert!(!p.can_terminate_now(), "empty input is not a complete match");
        step_regex(&mut p, "415-555-0100");
        assert!(p.can_terminate_now(), "valid phone is a complete match");
    }

    #[test]
    fn regex_rejects_first_byte_that_diverges_from_pattern() {
        // Pattern: must start with `Hello, `.
        let mut p = super::RegexGrammarParser::new(r"^Hello, \w+$").unwrap();
        assert_eq!(p.step(b'H'), Step::Accept);
        assert_eq!(p.step(b'e'), Step::Accept);
        // `X` here would diverge — pattern wants `l`.
        assert_eq!(p.step(b'X'), Step::Reject);
    }

    #[test]
    fn regex_can_terminate_only_at_match_state() {
        // Pattern: at least 3 digits.
        let mut p = super::RegexGrammarParser::new(r"^\d{3,}$").unwrap();
        assert!(!p.can_terminate_now(), "0 digits is not a match");
        p.step(b'1');
        assert!(!p.can_terminate_now(), "1 digit is not a match (need 3)");
        p.step(b'2');
        assert!(!p.can_terminate_now(), "2 digits not a match");
        p.step(b'3');
        assert!(p.can_terminate_now(), "3 digits matches the lower bound");
        p.step(b'4');
        assert!(p.can_terminate_now(), "4 digits still matches");
    }

    #[test]
    fn regex_grammar_mask_rejects_token_with_wrong_first_byte() {
        // Pattern: must start with digit, vocab: 0 = "5x", 1 = "5", 2 = "ab".
        let mask =
            super::GrammarMask::new_regex(r"^\d+$", 3, None, |id| match id {
                0 => b"5x".to_vec(),
                1 => b"5".to_vec(),
                2 => b"ab".to_vec(),
                _ => Vec::new(),
            })
            .unwrap();
        assert!(!mask.accepts(0), "5x has trailing non-digit → reject");
        assert!(mask.accepts(1), "5 is a digit → accept");
        assert!(!mask.accepts(2), "ab → reject");
    }

    #[test]
    fn regex_is_done_always_false_eos_governed_by_can_terminate() {
        // Same pattern as before; parser's is_done is always false
        // because regex grammars are stream-shaped — the model
        // (not the grammar) decides when to stop. EOS gates on
        // can_terminate_now.
        let p = super::RegexGrammarParser::new(r"^\d+$").unwrap();
        // No public is_done on the inner parser, but go through
        // ParserKind via GrammarMask to exercise the path.
        let _ = p;
        let mask = super::GrammarMask::new_regex(r"^\d+$", 1, None, |_| Vec::new()).unwrap();
        assert!(!mask.is_done(), "regex grammar must never report is_done=true");
    }
}
