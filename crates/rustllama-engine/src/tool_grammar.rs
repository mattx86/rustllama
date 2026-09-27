//! Tool-call streaming grammar: a stateful constraint that wraps the
//! plain-text model output, recognizes `<tool_call>...</tool_call>`
//! markers at the byte level, and engages [`JsonSchemaParser`] for the
//! `arguments` field of the inner JSON body.
//!
//! ## Body shape
//!
//! The grammar assumes the chat template renders tool calls in the
//! shape:
//!
//! ```json
//! <tool_call>{"name": "<one-of-known-names>", "arguments": <schema-valid-value>}</tool_call>
//! ```
//!
//! Strict key order (`name` first, `arguments` second), no
//! whitespace-tolerant tricks beyond ASCII spaces / tabs / newlines
//! between tokens. This matches the body emitted by the Qwen2.5-Coder
//! and DeepSeek-Coder chat templates we drive. Models that emit a
//! different body shape will see token-mask rejections (their tool
//! calls just won't surface) but the rest of the output is
//! unconstrained.
//!
//! ## Token-mask integration
//!
//! Same clone-and-step pattern as [`crate::grammar::GrammarMask`]:
//! the candidate token's bytes are stepped through a *clone* of the
//! grammar; the token is accepted only if every byte stays in the
//! grammar. After the sampler picks, `advance` commits the chosen
//! token's bytes to the live state.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::grammar::{JsonSchemaParser, Schema, Step};

const OPEN: &str = "<tool_call>";
const CLOSE: &str = "</tool_call>";

/// Top-level stream grammar that:
///   - accepts arbitrary bytes outside `<tool_call>` markers,
///   - switches to a strict body parser inside,
///   - delegates the `arguments` value bytes to a [`JsonSchemaParser`]
///     keyed by the resolved tool name.
///
/// Optionally caps the number of complete tool-call bodies the model
/// is allowed to emit per response — see [`Self::new_with_max`]. Once
/// the limit is reached the grammar rejects any byte that would open
/// a new `<tool_call>` marker, forcing the sampler to pick text or
/// EOS tokens instead.
#[derive(Debug, Clone)]
pub struct ToolCallStreamGrammar {
    schemas_by_name: BTreeMap<String, Schema>,
    state: State,
    /// Buffer for in-progress marker matching at OutsideToolCall and
    /// AfterBody states. We track at most `OPEN.len()` bytes (the
    /// longest marker we care about).
    marker_buf: Vec<u8>,
    /// Active body parser when `state == InBody`.
    body: Option<BodyParser>,
    /// Number of complete `<tool_call>...</tool_call>` bodies the
    /// grammar has accepted so far in this stream. Incremented when
    /// the body parser reports `BodyStep::Done`.
    completed: u32,
    /// Maximum number of complete tool-call bodies allowed. `0` means
    /// unlimited. Default for the no-arg constructor is unlimited; the
    /// engine sets a finite value via [`Self::new_with_max`].
    max_completed: u32,
    /// Minimum number of complete tool-call bodies required before the
    /// grammar will report `can_terminate_now()` — i.e. before EOS is
    /// legal. `0` (the default) imposes no lower bound. The engine sets
    /// this to `1` for OpenAI `tool_choice: "required"` / a forced
    /// function so the model can't stop until it has emitted at least
    /// one call. See [`Self::new_with_bounds`].
    min_completed: u32,
    /// Sticky flag set the moment the grammar rejects a byte because
    /// the limit was reached and the model tried to open another
    /// `<tool_call>`. Distinct from `completed == max_completed` —
    /// the model might naturally emit exactly the cap and finish
    /// without ever trying for more. This flag tells the caller "the
    /// limit actually constrained the output."
    limit_blocked: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum State {
    OutsideToolCall,
    InBody,
    AfterBody,
    /// Sticky rejection — once set, all further bytes reject. Used when
    /// the body parser fires a non-recoverable schema violation.
    Invalid,
}

impl ToolCallStreamGrammar {
    pub fn new(schemas_by_name: BTreeMap<String, Schema>) -> Self {
        Self::new_with_max(schemas_by_name, 0)
    }

    /// Construct a grammar that allows at most `max_completed` complete
    /// tool-call bodies before refusing to open another `<tool_call>`.
    /// Pass `0` for unlimited (same as [`Self::new`]).
    pub fn new_with_max(schemas_by_name: BTreeMap<String, Schema>, max_completed: u32) -> Self {
        Self::new_with_bounds(schemas_by_name, max_completed, 0)
    }

    /// Construct a grammar with both an upper (`max_completed`) and a
    /// lower (`min_completed`) bound on the number of complete tool-call
    /// bodies. `max_completed = 0` disables the cap; `min_completed = 0`
    /// disables the floor. When `min_completed > 0`, EOS is blocked
    /// (via [`Self::can_terminate_now`]) until that many calls have
    /// completed — this is how the server forces
    /// `tool_choice: "required"` / a specific function.
    pub fn new_with_bounds(
        schemas_by_name: BTreeMap<String, Schema>,
        max_completed: u32,
        min_completed: u32,
    ) -> Self {
        Self {
            schemas_by_name,
            state: State::OutsideToolCall,
            marker_buf: Vec::new(),
            body: None,
            completed: 0,
            max_completed,
            min_completed,
            limit_blocked: false,
        }
    }

    /// Number of complete tool-call bodies emitted so far.
    pub fn completed_count(&self) -> u32 {
        self.completed
    }

    /// `true` once the model has attempted to open another `<tool_call>`
    /// after the `max_completed` cap was reached and the grammar
    /// blocked the opener byte.
    pub fn limit_blocked(&self) -> bool {
        self.limit_blocked
    }

    /// EOS is only legal outside a tool_call (in-progress bodies must
    /// finish + the close marker must arrive) AND once at least
    /// `min_completed` complete tool-call bodies have been emitted. The
    /// `min_completed` floor is what forces `tool_choice: "required"` /
    /// a specific function — until it's met the sampler's EOS token is
    /// masked out, so the model keeps generating until it produces a
    /// call.
    pub fn can_terminate_now(&self) -> bool {
        matches!(self.state, State::OutsideToolCall) && self.completed >= self.min_completed
    }

    pub fn step(&mut self, b: u8) -> Step {
        match self.state {
            State::Invalid => Step::Reject,
            State::OutsideToolCall => self.step_outside(b),
            State::InBody => self.step_in_body(b),
            State::AfterBody => self.step_after_body(b),
        }
    }

    fn step_outside(&mut self, b: u8) -> Step {
        // Accept any byte; track the longest suffix of marker_buf that
        // is a prefix of OPEN so we can detect the marker on completion.
        self.marker_buf.push(b);
        // Bound the buffer at OPEN.len() — older bytes can't contribute.
        if self.marker_buf.len() > OPEN.len() {
            let cut = self.marker_buf.len() - OPEN.len();
            self.marker_buf.drain(..cut);
        }
        if self.marker_buf.ends_with(OPEN.as_bytes()) {
            // Recursion guard: reject the byte that would have started
            // a new tool-call body if we've already emitted the limit.
            // The sampler's token mask then forces the model to pick
            // either regular text or EOS.
            if self.max_completed > 0 && self.completed >= self.max_completed {
                self.marker_buf.pop();
                self.limit_blocked = true;
                return Step::Reject;
            }
            self.marker_buf.clear();
            self.state = State::InBody;
            self.body = Some(BodyParser::new(self.schemas_by_name.clone()));
        }
        Step::Accept
    }

    fn step_in_body(&mut self, b: u8) -> Step {
        let body = self.body.as_mut().expect("body parser missing");
        match body.step(b) {
            BodyStep::Accept => Step::Accept,
            BodyStep::Done => {
                // Body's closing `}` was just consumed. Now we need
                // exactly `</tool_call>` to follow (whitespace not
                // allowed between body and the close marker — the
                // standard templates don't emit any).
                self.state = State::AfterBody;
                self.marker_buf.clear();
                self.completed = self.completed.saturating_add(1);
                Step::Accept
            }
            BodyStep::Reject => {
                self.state = State::Invalid;
                Step::Reject
            }
        }
    }

    fn step_after_body(&mut self, b: u8) -> Step {
        // Strict prefix match against CLOSE: only the next-expected byte
        // is acceptable. (Standard chat templates emit the close marker
        // immediately after the body's `}`; we don't allow whitespace
        // between them, matching what the templates produce.)
        let pos = self.marker_buf.len();
        if pos >= CLOSE.len() {
            self.state = State::Invalid;
            return Step::Reject;
        }
        if CLOSE.as_bytes()[pos] != b {
            self.state = State::Invalid;
            return Step::Reject;
        }
        self.marker_buf.push(b);
        if self.marker_buf.len() == CLOSE.len() {
            self.marker_buf.clear();
            self.state = State::OutsideToolCall;
        }
        Step::Accept
    }
}

/// Hand-rolled FSM for the inner body. Strict shape:
/// `{"name": "<known>", "arguments": <schema-valid value>}`. Whitespace
/// (`' '`, `'\t'`, `'\n'`, `'\r'`) is permitted between tokens but not
/// inside the key strings themselves.
#[derive(Debug, Clone)]
struct BodyParser {
    schemas_by_name: BTreeMap<String, Schema>,
    state: BodyState,
    /// Accumulated `name` value once parsed; used to resolve the
    /// argument schema when the parser enters the args value.
    resolved_name: String,
    /// Active args-value sub-parser (set once the `arguments` key
    /// resolves to a schema).
    args_parser: Option<JsonSchemaParser>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum BodyState {
    /// Awaiting `{` (with leading whitespace allowed).
    ExpectOpenBrace,
    /// Inside the body, awaiting the `"name"` key literal.
    ExpectNameKey { progress: u8 },
    /// Awaiting `:` after the name key.
    ExpectColonAfterName,
    /// Awaiting `"` that opens the name's string value.
    ExpectNameValueOpenQuote,
    /// Accumulating the name's bytes.
    InNameValue,
    /// Awaiting `,` after the name value's close quote.
    ExpectCommaAfterName,
    /// Awaiting the `"arguments"` key literal.
    ExpectArgumentsKey { progress: u8 },
    /// Awaiting `:` after the arguments key.
    ExpectColonAfterArgs,
    /// Inside the arguments value (delegated to `args_parser`).
    InArgumentsValue,
    /// Awaiting `}` (with leading whitespace allowed).
    ExpectCloseBrace,
}

const NAME_KEY: &[u8] = b"\"name\"";
const ARGS_KEY: &[u8] = b"\"arguments\"";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BodyStep {
    Accept,
    Done,
    Reject,
}

impl BodyParser {
    fn new(schemas_by_name: BTreeMap<String, Schema>) -> Self {
        Self {
            schemas_by_name,
            state: BodyState::ExpectOpenBrace,
            resolved_name: String::new(),
            args_parser: None,
        }
    }

    fn step(&mut self, b: u8) -> BodyStep {
        // Whitespace handling: most states tolerate leading whitespace
        // between tokens. The exceptions (inside strings, inside the
        // args value's nested JSON) handle their own.
        if is_ws(b)
            && !matches!(
                self.state,
                BodyState::InNameValue | BodyState::InArgumentsValue
            )
            && !matches!(self.state, BodyState::ExpectNameKey { progress: 1.. })
            && !matches!(self.state, BodyState::ExpectArgumentsKey { progress: 1.. })
        {
            return BodyStep::Accept;
        }

        match self.state {
            BodyState::ExpectOpenBrace => {
                if b == b'{' {
                    self.state = BodyState::ExpectNameKey { progress: 0 };
                    BodyStep::Accept
                } else {
                    BodyStep::Reject
                }
            }
            BodyState::ExpectNameKey { progress } => {
                let i = progress as usize;
                if i < NAME_KEY.len() && NAME_KEY[i] == b {
                    if i + 1 == NAME_KEY.len() {
                        self.state = BodyState::ExpectColonAfterName;
                    } else {
                        self.state = BodyState::ExpectNameKey {
                            progress: progress + 1,
                        };
                    }
                    BodyStep::Accept
                } else {
                    BodyStep::Reject
                }
            }
            BodyState::ExpectColonAfterName => {
                if b == b':' {
                    self.state = BodyState::ExpectNameValueOpenQuote;
                    BodyStep::Accept
                } else {
                    BodyStep::Reject
                }
            }
            BodyState::ExpectNameValueOpenQuote => {
                if b == b'"' {
                    self.state = BodyState::InNameValue;
                    self.resolved_name.clear();
                    BodyStep::Accept
                } else {
                    BodyStep::Reject
                }
            }
            BodyState::InNameValue => {
                if b == b'"' {
                    // Close of name value. Validate the name is known —
                    // models may hallucinate.
                    if self.schemas_by_name.contains_key(&self.resolved_name) {
                        self.state = BodyState::ExpectCommaAfterName;
                        BodyStep::Accept
                    } else {
                        BodyStep::Reject
                    }
                } else if b == b'\\' {
                    // We don't support escaped names — chat templates
                    // don't emit them. Reject to keep the FSM simple.
                    BodyStep::Reject
                } else {
                    // Reject raw control characters (per JSON).
                    if b < 0x20 {
                        return BodyStep::Reject;
                    }
                    // Constrain the accumulating name to the *prefix*
                    // of at least one known tool name. This means the
                    // model can't waste tokens on a name that's already
                    // diverged from every known tool.
                    let mut candidate = self.resolved_name.clone();
                    candidate.push(b as char);
                    let any_matches = self
                        .schemas_by_name
                        .keys()
                        .any(|n| n.as_bytes().starts_with(candidate.as_bytes()));
                    if !any_matches {
                        return BodyStep::Reject;
                    }
                    self.resolved_name.push(b as char);
                    BodyStep::Accept
                }
            }
            BodyState::ExpectCommaAfterName => {
                if b == b',' {
                    self.state = BodyState::ExpectArgumentsKey { progress: 0 };
                    BodyStep::Accept
                } else {
                    BodyStep::Reject
                }
            }
            BodyState::ExpectArgumentsKey { progress } => {
                let i = progress as usize;
                if i < ARGS_KEY.len() && ARGS_KEY[i] == b {
                    if i + 1 == ARGS_KEY.len() {
                        self.state = BodyState::ExpectColonAfterArgs;
                    } else {
                        self.state = BodyState::ExpectArgumentsKey {
                            progress: progress + 1,
                        };
                    }
                    BodyStep::Accept
                } else {
                    BodyStep::Reject
                }
            }
            BodyState::ExpectColonAfterArgs => {
                if b == b':' {
                    // Engage the schema parser for the args value.
                    let schema = self
                        .schemas_by_name
                        .get(&self.resolved_name)
                        .cloned()
                        .unwrap_or(Schema::Any);
                    self.args_parser = Some(JsonSchemaParser::new(schema));
                    self.state = BodyState::InArgumentsValue;
                    BodyStep::Accept
                } else {
                    BodyStep::Reject
                }
            }
            BodyState::InArgumentsValue => {
                let p = self.args_parser.as_mut().expect("args parser missing");
                match p.step(b) {
                    Step::Accept => BodyStep::Accept,
                    Step::Done => {
                        self.state = BodyState::ExpectCloseBrace;
                        BodyStep::Accept
                    }
                    Step::Reject => BodyStep::Reject,
                }
            }
            BodyState::ExpectCloseBrace => {
                if b == b'}' {
                    BodyStep::Done
                } else {
                    BodyStep::Reject
                }
            }
        }
    }
}

fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

/// Wire-shape configuration of a tool — name + parameter schema.
/// This is what the server hands to the engine when tools are present
/// on a request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolGrammarSpec {
    pub name: String,
    /// JSON Schema for the tool's argument object. The chat handler
    /// resolves OpenAI's `function.parameters` field through
    /// `Schema::from_json_value`.
    pub schema: Schema,
}

impl ToolGrammarSpec {
    /// Build a `schemas_by_name` map from a list of specs. Skips
    /// duplicate names (last one wins).
    pub fn into_map(specs: Vec<ToolGrammarSpec>) -> BTreeMap<String, Schema> {
        let mut out = BTreeMap::new();
        for s in specs {
            out.insert(s.name, s.schema);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drive(grammar: &mut ToolCallStreamGrammar, s: &str) -> bool {
        for &b in s.as_bytes() {
            if grammar.step(b) == Step::Reject {
                return false;
            }
        }
        true
    }

    fn one_tool() -> BTreeMap<String, Schema> {
        let mut m = BTreeMap::new();
        m.insert(
            "get_weather".into(),
            Schema::Object {
                properties: [
                    ("city".to_string(), Schema::String { enum_values: None }),
                    ("unit".to_string(), Schema::String { enum_values: None }),
                ]
                .into_iter()
                .collect(),
                required: ["city".to_string()].into_iter().collect(),
                additional: true,
            },
        );
        m
    }

    #[test]
    fn accepts_well_formed_tool_call() {
        let mut g = ToolCallStreamGrammar::new(one_tool());
        let ok = drive(
            &mut g,
            "Some prose first.\n<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}</tool_call>",
        );
        assert!(ok);
        assert!(matches!(g.state, State::OutsideToolCall));
    }

    #[test]
    fn rejects_unknown_tool_name() {
        let mut g = ToolCallStreamGrammar::new(one_tool());
        // After the second `"`, "fa" is no longer a prefix of any known
        // tool name (`get_weather`), so the parser rejects the `a`.
        let ok = drive(
            &mut g,
            "<tool_call>{\"name\": \"fake_tool\", \"arguments\": {}}</tool_call>",
        );
        assert!(!ok);
    }

    #[test]
    fn rejects_missing_required_arg() {
        let mut g = ToolCallStreamGrammar::new(one_tool());
        // `city` is required; the body parser rejects on `}` (the
        // arguments-value close) because keys_seen is empty.
        let ok = drive(
            &mut g,
            "<tool_call>{\"name\": \"get_weather\", \"arguments\": {}}</tool_call>",
        );
        assert!(!ok);
    }

    #[test]
    fn rejects_wrong_type_for_arg() {
        let mut g = ToolCallStreamGrammar::new(one_tool());
        // `city` must be a string; a number value is rejected.
        let ok = drive(
            &mut g,
            "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": 42}}</tool_call>",
        );
        assert!(!ok);
    }

    #[test]
    fn accepts_multiple_calls_with_text_between() {
        let mut g = ToolCallStreamGrammar::new(one_tool());
        let ok = drive(
            &mut g,
            "First call:\n\
             <tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}</tool_call>\n\
             Second call:\n\
             <tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Tokyo\", \"unit\": \"C\"}}</tool_call>\n\
             Done.",
        );
        assert!(ok);
        assert!(matches!(g.state, State::OutsideToolCall));
    }

    #[test]
    fn rejects_extra_chars_after_close_marker() {
        // After the body's `}`, only the exact bytes of `</tool_call>`
        // are accepted. A space between the body and the marker breaks
        // the strict template shape and rejects.
        let mut g = ToolCallStreamGrammar::new(one_tool());
        let ok = drive(
            &mut g,
            "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"P\"}} </tool_call>",
        );
        assert!(!ok);
    }

    #[test]
    fn outside_tool_call_accepts_any_text() {
        let mut g = ToolCallStreamGrammar::new(one_tool());
        let ok = drive(
            &mut g,
            "This is free text with random punctuation, !@#$%^&*()_+{}|:<>?\nand newlines and emojis 😀.\n",
        );
        assert!(ok);
    }

    #[test]
    fn can_terminate_only_outside_tool_call() {
        let mut g = ToolCallStreamGrammar::new(one_tool());
        assert!(g.can_terminate_now());
        drive(&mut g, "<tool_call>{");
        assert!(!g.can_terminate_now());
    }

    #[test]
    fn max_completed_zero_means_unlimited() {
        // No cap: 5 sequential tool calls are all accepted.
        let mut g = ToolCallStreamGrammar::new_with_max(one_tool(), 0);
        let one = "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"X\"}}</tool_call>";
        for _ in 0..5 {
            assert!(drive(&mut g, one), "all 5 accepted with limit=0");
            assert!(drive(&mut g, " "), "whitespace between calls accepted");
        }
        assert_eq!(g.completed_count(), 5);
    }

    #[test]
    fn max_completed_limit_rejects_extra_open_marker() {
        // limit=2: first two complete, third's opener must reject.
        let mut g = ToolCallStreamGrammar::new_with_max(one_tool(), 2);
        let one = "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"X\"}}</tool_call>";
        assert!(drive(&mut g, one));
        assert!(drive(&mut g, one));
        assert_eq!(g.completed_count(), 2);
        // Third attempt's open marker: the final `>` of `<tool_call>`
        // is the byte the grammar rejects.
        let third_open = "<tool_call";
        assert!(drive(&mut g, third_open), "prefix bytes still accepted");
        // The next byte that completes the OPEN marker is `>`.
        assert_eq!(g.step(b'>'), Step::Reject);
    }

    #[test]
    fn limit_blocked_flag_starts_false() {
        let mut g = ToolCallStreamGrammar::new_with_max(one_tool(), 2);
        let one = "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"X\"}}</tool_call>";
        assert!(!g.limit_blocked());
        // Emitting up to the cap doesn't flip the flag — the cap only
        // affects the NEXT opener attempt.
        assert!(drive(&mut g, one));
        assert!(drive(&mut g, one));
        assert_eq!(g.completed_count(), 2);
        assert!(!g.limit_blocked(), "cap reached but never blocked yet");
    }

    #[test]
    fn limit_blocked_flag_sets_when_extra_opener_rejected() {
        let mut g = ToolCallStreamGrammar::new_with_max(one_tool(), 1);
        let one = "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"X\"}}</tool_call>";
        assert!(drive(&mut g, one));
        assert!(!g.limit_blocked());
        // Drive prefix of `<tool_call>` — all bytes up to but not
        // including the final `>` are still accepted.
        assert!(drive(&mut g, "<tool_call"));
        assert_eq!(g.step(b'>'), Step::Reject);
        assert!(g.limit_blocked(), "limit_blocked must flip after the reject");
    }

    #[test]
    fn limit_still_allows_plain_text_after_cap_hit() {
        // After the cap is reached, plain text + EOS-shaped content
        // remains accepted; only the OPEN marker is gated.
        let mut g = ToolCallStreamGrammar::new_with_max(one_tool(), 1);
        let one = "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"X\"}}</tool_call>";
        assert!(drive(&mut g, one));
        assert!(g.can_terminate_now());
        // Plain prose after the cap is fine.
        assert!(drive(
            &mut g,
            "\nThe weather is 75 degrees and sunny in X. No more tool calls needed.\n",
        ));
        assert!(g.can_terminate_now());
    }

    #[test]
    fn min_completed_blocks_terminate_until_one_call() {
        // min_completed=1 (as set for tool_choice: "required" / a forced
        // function): EOS is illegal until exactly one complete tool-call
        // body has been emitted, even though the grammar starts (and, in
        // between calls, sits) in State::OutsideToolCall.
        let mut g = ToolCallStreamGrammar::new_with_bounds(one_tool(), 0, 1);
        // Fresh grammar: outside a tool_call but zero completed → blocked.
        assert!(!g.can_terminate_now(), "min floor not met at start");
        // Some leading prose is fine and must NOT unblock termination.
        assert!(drive(&mut g, "Let me look that up. "));
        assert!(!g.can_terminate_now(), "prose alone must not satisfy the floor");
        // One complete call satisfies the floor.
        let one =
            "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"X\"}}</tool_call>";
        assert!(drive(&mut g, one));
        assert_eq!(g.completed_count(), 1);
        assert!(g.can_terminate_now(), "floor met after one call → EOS legal");
    }
}
