//! GBNF (GGML BNF) grammar → token-constraint compiler.
//!
//! GBNF is the public grammar text format documented by the ggml/llama.cpp
//! project (`grammars/README.md`). This module is a **clean-room** Rust
//! reimplementation of the format from that public specification — no
//! llama.cpp source is used. It parses GBNF TEXT into an internal rule set
//! and drives the SAME byte-stepping [`crate::grammar::Step`] interface the
//! other [`crate::GrammarKind`]s use, so a GBNF grammar plugs into the
//! existing [`crate::grammar::GrammarMask`] token filter unchanged.
//!
//! ## What GBNF is
//!
//! A grammar is a set of named rules; `root` is the entry point. Each rule is
//! `name ::= alternates`, where alternates are `|`-separated sequences of
//! terms. Terms are:
//!   - string literals `"abc"` (with `\n \r \t \\ \" \xNN \uNNNN \UNNNNNNNN`),
//!   - character classes `[a-z0-9_]`, negated `[^ \t\n]`,
//!   - rule references (bare identifiers),
//!   - parenthesized groups `( ... )`,
//!   - any of the above followed by a repetition operator `* + ?` or a
//!     bounded range `{m}` / `{m,}` / `{m,n}`.
//! `#` begins a line comment.
//!
//! ## Why a stack machine (and not an NFA)
//!
//! GBNF rules may be **recursive** (`root ::= "(" root ")" | ""`), so the
//! language is context-free, not regular — a finite NFA cannot represent it.
//! Following the documented algorithm, we simulate a nondeterministic
//! pushdown automaton: the parser holds a SET of *stacks*, each stack a list
//! of positions `(rule, element-index)` whose top is the next terminal to
//! match. Matching a character advances every stack whose top accepts it;
//! rule references are expanded (epsilon-closure) by `advance_stack`. A stack
//! that becomes empty is an accepting (fully-matched) parse.
//!
//! Characters are Unicode scalar values; tokens arrive as UTF-8 **bytes**
//! (byte-level BPE tokens can even split a codepoint), so the parser buffers
//! bytes until a full codepoint is available, then matches it. Mid-codepoint,
//! `step` returns `Accept` provisionally and `can_terminate_now` is false.

use std::collections::BTreeSet;
use std::sync::Arc;

use crate::grammar::Step;

/// One compiled grammar element. Mirrors the documented GBNF element kinds
/// (clean-room). A rule is a flat `Vec<Element>` with [`Element::Alt`]
/// separating alternates and [`Element::End`] terminating the rule.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Element {
    /// End of the current alternate (and, as the last element, of the rule).
    End,
    /// Separator between alternates of a rule.
    Alt,
    /// Non-terminal: match the referenced rule here.
    RuleRef(u32),
    /// Start of a (possibly multi-range) character set: a literal / range
    /// lower bound. Followed optionally by [`Element::CharRngUpper`] (making a
    /// range) and any number of [`Element::CharAlt`] (additional members).
    Char(u32),
    /// Start of a NEGATED character set (`[^...]`): matches any codepoint NOT
    /// in the set.
    CharNot(u32),
    /// Upper bound of the immediately-preceding `Char` / `CharAlt` range.
    CharRngUpper(u32),
    /// An additional member (or range lower bound) of the current char set.
    CharAlt(u32),
}

/// A compiled GBNF grammar: `rules[i]` is the flat element list for rule `i`,
/// and `root` is the entry rule id. Shared behind an `Arc` so cloning a
/// [`GbnfParser`] (which happens per candidate token inside the mask) only
/// clones the small active-stack set, never the grammar.
#[derive(Debug, PartialEq, Eq)]
pub struct CompiledGbnf {
    rules: Vec<Vec<Element>>,
    root: u32,
}

/// Error compiling GBNF text. Surfaced at the request boundary as a clean 400.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum GbnfError {
    #[error("invalid GBNF grammar: {0}")]
    Parse(String),
}

impl CompiledGbnf {
    /// Parse + compile GBNF grammar text. Returns [`GbnfError::Parse`] with a
    /// human-readable reason for any malformed input (unknown rule reference,
    /// unterminated string/class, missing `root`, bad escape, …).
    pub fn compile(text: &str) -> Result<Arc<Self>, GbnfError> {
        let mut p = Parser::new(text);
        p.parse_grammar()?;
        let root = *p
            .rule_ids
            .get("root")
            .ok_or_else(|| GbnfError::Parse("grammar has no `root` rule".to_string()))?;
        if !p.defined[root as usize] {
            return Err(GbnfError::Parse(
                "`root` rule is referenced but never defined".to_string(),
            ));
        }
        // Verify every rule reference resolves to a DEFINED rule — an undefined
        // non-terminal (a forward reference that was never given a body) would
        // otherwise silently match the empty string, so fail-fast at compile
        // time rather than masking the typo at sample time.
        for rule in p.rules.iter() {
            for el in rule {
                if let Element::RuleRef(r) = el {
                    let idx = *r as usize;
                    if idx >= p.rules.len() || !p.defined[idx] {
                        // Recover the name for the message when it's a named rule.
                        let name = p
                            .rule_ids
                            .iter()
                            .find(|(_, id)| **id == *r)
                            .map(|(n, _)| n.as_str())
                            .unwrap_or("<anonymous>");
                        return Err(GbnfError::Parse(format!(
                            "reference to undefined rule `{name}`"
                        )));
                    }
                }
            }
        }
        Ok(Arc::new(Self {
            rules: p.rules,
            root,
        }))
    }

    /// Whether codepoint `cp` is matched by the char set that begins at
    /// `rule`'s element `idx` (a `Char`/`CharNot`), and the index of the first
    /// element AFTER that set.
    fn char_set_matches(&self, rule: u32, idx: usize, cp: u32) -> (bool, usize) {
        let elems = &self.rules[rule as usize];
        let negate = matches!(elems[idx], Element::CharNot(_));
        let mut i = idx;
        let mut found = false;
        // First member.
        let lo = match elems[i] {
            Element::Char(c) | Element::CharNot(c) => c,
            _ => unreachable!("char_set_matches called off a non-char element"),
        };
        i += 1;
        // Optional range upper bound for the first member.
        if i < elems.len() {
            if let Element::CharRngUpper(hi) = elems[i] {
                if cp >= lo && cp <= hi {
                    found = true;
                }
                i += 1;
            } else if cp == lo {
                found = true;
            }
        } else if cp == lo {
            found = true;
        }
        // Additional members (`CharAlt`), each optionally a range.
        while i < elems.len() {
            if let Element::CharAlt(c) = elems[i] {
                i += 1;
                if i < elems.len() {
                    if let Element::CharRngUpper(hi) = elems[i] {
                        if cp >= c && cp <= hi {
                            found = true;
                        }
                        i += 1;
                        continue;
                    }
                }
                if cp == c {
                    found = true;
                }
            } else {
                break;
            }
        }
        (found ^ negate, i)
    }
}

/// Validate GBNF grammar text, returning `Ok(())` if it compiles. Used at the
/// HTTP request boundary to reject a malformed grammar with a clean 400 before
/// any generation starts (the engine dispatch sites also warn-and-drop).
pub fn validate_gbnf(text: &str) -> Result<(), GbnfError> {
    CompiledGbnf::compile(text).map(|_| ())
}

/// A live GBNF parse: the compiled grammar plus the current nondeterministic
/// stack set and a UTF-8 reassembly buffer. Cloning is cheap (the grammar is
/// shared via `Arc`; only the stack set + tiny byte buffer clone).
#[derive(Debug, Clone)]
pub struct GbnfParser {
    grammar: Arc<CompiledGbnf>,
    /// Set of active stacks. Each stack is a list of `(rule, element-index)`
    /// positions; the LAST entry is the current element to match. An EMPTY
    /// stack is an accepting (fully-matched) parse.
    stacks: Vec<Vec<(u32, usize)>>,
    /// Bytes of a partially-received multi-byte UTF-8 codepoint.
    utf8_buf: Vec<u8>,
    /// Total bytes expected for the in-progress codepoint (0 when idle).
    utf8_need: usize,
    /// Set once the stack set goes empty — no token can recover, so every
    /// further non-terminating byte is a reject.
    dead: bool,
}

/// Cap on stacks / recursion expansion. A pathological or left-recursive
/// grammar could otherwise expand without bound; hitting the cap fails the
/// expansion closed (the offending stacks are dropped) rather than hanging.
const MAX_STACKS: usize = 4096;

impl GbnfParser {
    /// Start a fresh parse at the grammar's `root` rule.
    pub fn new(grammar: Arc<CompiledGbnf>) -> Self {
        let root = grammar.root;
        let mut out: Vec<Vec<(u32, usize)>> = Vec::new();
        // Seed from each alternate of the root rule, then epsilon-close.
        Self::expand_rule_into(&grammar, root, &[], &mut out);
        Self {
            grammar,
            stacks: dedup_stacks(out),
            utf8_buf: Vec::new(),
            utf8_need: 0,
            dead: false,
        }
    }

    /// Expand every alternate of `rule` onto `base` (the return stack below
    /// the rule reference), epsilon-closing each result into `out`.
    fn expand_rule_into(
        grammar: &CompiledGbnf,
        rule: u32,
        base: &[(u32, usize)],
        out: &mut Vec<Vec<(u32, usize)>>,
    ) {
        let elems = &grammar.rules[rule as usize];
        let mut alt_start = 0usize;
        let mut i = 0usize;
        // Walk the flat element list; each alternate runs until `Alt`/`End`.
        loop {
            if i >= elems.len() || elems[i] == Element::Alt || elems[i] == Element::End {
                // Alternate [alt_start, i): if it has a first element, push a
                // stack positioned at it and advance-close; empty alternate →
                // the base stack itself is a (possibly accepting) result.
                if alt_start < i {
                    let mut stack = base.to_vec();
                    stack.push((rule, alt_start));
                    advance_stack(grammar, stack, out, 0);
                } else {
                    // Empty alternate (e.g. `x?` desugar): nothing to match
                    // from this rule — the continuation already on `base` is
                    // the result.
                    advance_stack(grammar, base.to_vec(), out, 0);
                }
                if i >= elems.len() || elems[i] == Element::End {
                    break;
                }
                // Skip the `Alt` and begin the next alternate.
                i += 1;
                alt_start = i;
                continue;
            }
            i += 1;
        }
    }

    /// Step one byte. Buffers multi-byte UTF-8 until a full codepoint, then
    /// matches it against every active stack.
    pub fn step(&mut self, b: u8) -> Step {
        if self.dead {
            return Step::Reject;
        }
        if self.utf8_need == 0 {
            // Leading byte: determine the codepoint length.
            self.utf8_need = match b {
                0x00..=0x7F => 1,
                0xC0..=0xDF => 2,
                0xE0..=0xEF => 3,
                0xF0..=0xF7 => 4,
                // Continuation / invalid leading byte with no codepoint in
                // progress — not valid UTF-8 at a boundary.
                _ => {
                    self.dead = true;
                    return Step::Reject;
                }
            };
            self.utf8_buf.clear();
        }
        self.utf8_buf.push(b);
        if self.utf8_buf.len() < self.utf8_need {
            // Still mid-codepoint: provisionally accept; cannot reject or
            // terminate until the scalar is complete.
            return Step::Accept;
        }
        // Full codepoint assembled.
        let cp = match std::str::from_utf8(&self.utf8_buf) {
            Ok(s) => s.chars().next().map(|c| c as u32),
            Err(_) => None,
        };
        self.utf8_buf.clear();
        self.utf8_need = 0;
        let Some(cp) = cp else {
            self.dead = true;
            return Step::Reject;
        };
        self.accept_char(cp)
    }

    /// Advance every active stack by codepoint `cp`.
    fn accept_char(&mut self, cp: u32) -> Step {
        let mut next: Vec<Vec<(u32, usize)>> = Vec::new();
        for stack in &self.stacks {
            let Some(&(rule, idx)) = stack.last() else {
                // Empty (accepting) stack cannot consume another char.
                continue;
            };
            let el = &self.grammar.rules[rule as usize][idx];
            if !matches!(el, Element::Char(_) | Element::CharNot(_)) {
                // Not a terminal position — advance_stack should have resolved
                // it; skip defensively.
                continue;
            }
            let (matched, after) = self.grammar.char_set_matches(rule, idx, cp);
            if !matched {
                continue;
            }
            let mut base = stack.clone();
            base.pop(); // consume the char-set position
                        // Push the continuation only if more of THIS alternate remains.
                        // `End` ends the rule; `Alt` ends this alternate — either way the
                        // frame is done and we return to the caller's continuation below.
            if !matches!(
                self.grammar.rules[rule as usize][after],
                Element::End | Element::Alt
            ) {
                base.push((rule, after));
            }
            advance_stack(&self.grammar, base, &mut next, 0);
        }
        self.stacks = dedup_stacks(next);
        if self.stacks.is_empty() {
            self.dead = true;
            Step::Reject
        } else {
            Step::Accept
        }
    }

    /// True when the bytes consumed so far form a COMPLETE match of the
    /// grammar — i.e. some active stack is empty AND no partial codepoint is
    /// buffered. The mask lets EOS through here.
    pub fn can_terminate_now(&self) -> bool {
        self.utf8_need == 0 && self.stacks.iter().any(|s| s.is_empty())
    }
}

/// Epsilon-closure of one stack: resolve `RuleRef` tops by expanding the
/// referenced rule's alternates (pushing the continuation as the return
/// frame), until the top is a terminal (`Char`/`CharNot`) or the stack is
/// empty. Results (terminal-topped or empty stacks) are pushed into `out`.
fn advance_stack(
    grammar: &CompiledGbnf,
    stack: Vec<(u32, usize)>,
    out: &mut Vec<Vec<(u32, usize)>>,
    depth: usize,
) {
    if out.len() >= MAX_STACKS || depth > MAX_STACKS {
        // Fail-closed on runaway expansion (e.g. left recursion).
        return;
    }
    let Some(&(rule, idx)) = stack.last() else {
        // Empty stack = accepting parse. Keep it.
        out.push(stack);
        return;
    };
    match grammar.rules[rule as usize][idx] {
        Element::Char(_) | Element::CharNot(_) => {
            // Terminal top — ready to match a char.
            out.push(stack);
        }
        Element::RuleRef(r) => {
            // Return frame = the element after the reference. Skip it when the
            // reference ends the alternate (`End` or `Alt`) — then the return
            // continuation is whatever is already below on the stack.
            let mut base = stack.clone();
            base.pop();
            if !matches!(
                grammar.rules[rule as usize][idx + 1],
                Element::End | Element::Alt
            ) {
                base.push((rule, idx + 1));
            }
            // Expand each alternate of the referenced rule onto `base`.
            let elems = &grammar.rules[r as usize];
            let mut alt_start = 0usize;
            let mut i = 0usize;
            loop {
                if i >= elems.len() || elems[i] == Element::Alt || elems[i] == Element::End {
                    if alt_start < i {
                        let mut s = base.clone();
                        s.push((r, alt_start));
                        advance_stack(grammar, s, out, depth + 1);
                    } else {
                        // Empty alternate: continuation only.
                        advance_stack(grammar, base.clone(), out, depth + 1);
                    }
                    if i >= elems.len() || elems[i] == Element::End {
                        break;
                    }
                    i += 1;
                    alt_start = i;
                    continue;
                }
                i += 1;
            }
        }
        // `Alt`/`End`/`CharRngUpper`/`CharAlt` are never a stack top.
        _ => {}
    }
}

/// Deduplicate a stack set (nondeterministic expansion produces duplicates),
/// preserving determinism via a sorted set.
fn dedup_stacks(stacks: Vec<Vec<(u32, usize)>>) -> Vec<Vec<(u32, usize)>> {
    let set: BTreeSet<Vec<(u32, usize)>> = stacks.into_iter().collect();
    set.into_iter().collect()
}

// ============================================================
// Parser: GBNF text → Vec<Vec<Element>>
// ============================================================

struct Parser<'a> {
    src: &'a [u8],
    pos: usize,
    rules: Vec<Vec<Element>>,
    rule_ids: std::collections::HashMap<String, u32>,
    /// `defined[id]` is `true` once rule `id` has a real body. Anonymous rules
    /// (groups / repetition desugar) are born defined; a NAMED rule interned by
    /// a forward reference stays `false` until its `name ::= ...` is parsed, so
    /// an undefined reference is caught at compile time rather than silently
    /// matching the empty string.
    defined: Vec<bool>,
}

impl<'a> Parser<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            src: text.as_bytes(),
            pos: 0,
            rules: Vec::new(),
            rule_ids: std::collections::HashMap::new(),
            defined: Vec::new(),
        }
    }

    /// Reserve (or fetch) the id for a rule name. References can appear before
    /// a rule is defined, so names are interned lazily with an empty body
    /// (marked not-yet-defined).
    fn rule_id(&mut self, name: &str) -> u32 {
        if let Some(id) = self.rule_ids.get(name) {
            return *id;
        }
        let id = self.rules.len() as u32;
        self.rules.push(Vec::new());
        self.defined.push(false);
        self.rule_ids.insert(name.to_string(), id);
        id
    }

    /// Allocate a fresh anonymous rule (for groups / repetition desugaring).
    /// Anonymous rules always receive a body from their caller, so they are
    /// born defined.
    fn new_anon_rule(&mut self) -> u32 {
        let id = self.rules.len() as u32;
        self.rules.push(Vec::new());
        self.defined.push(true);
        id
    }

    fn parse_grammar(&mut self) -> Result<(), GbnfError> {
        loop {
            self.skip_space_and_comments();
            if self.pos >= self.src.len() {
                break;
            }
            self.parse_rule()?;
        }
        if self.rules.is_empty() {
            return Err(GbnfError::Parse("empty grammar".to_string()));
        }
        Ok(())
    }

    fn parse_rule(&mut self) -> Result<(), GbnfError> {
        let name = self.parse_name()?;
        self.skip_inline_space();
        if !self.consume_str("::=") {
            return Err(GbnfError::Parse(format!("rule `{name}` missing `::=`")));
        }
        let id = self.rule_id(&name);
        let body = self.parse_alternates()?;
        // Overwrite any previously-interned empty body + mark it defined.
        self.rules[id as usize] = body;
        self.defined[id as usize] = true;
        Ok(())
    }

    /// Parse `seq ( "|" seq )*` into a flat element list with `Alt` separators
    /// and a trailing `End`.
    fn parse_alternates(&mut self) -> Result<Vec<Element>, GbnfError> {
        let mut out: Vec<Element> = Vec::new();
        loop {
            let seq = self.parse_sequence()?;
            out.extend(seq);
            self.skip_inline_space();
            if self.peek() == Some(b'|') {
                self.pos += 1;
                out.push(Element::Alt);
                self.skip_space_and_comments();
                continue;
            }
            break;
        }
        out.push(Element::End);
        Ok(out)
    }

    /// Parse a space-separated run of terms (with repetition suffixes) up to a
    /// `|`, `)`, newline rule boundary, or EOF.
    fn parse_sequence(&mut self) -> Result<Vec<Element>, GbnfError> {
        let mut out: Vec<Element> = Vec::new();
        loop {
            self.skip_inline_space_and_comments();
            match self.peek() {
                None => break,
                Some(b'|') | Some(b')') => break,
                Some(b'\n') | Some(b'\r') => {
                    // A newline ends the rule unless the next non-space line
                    // continues an alternate via `|` — GBNF rules are
                    // single-logical-line, so stop here.
                    break;
                }
                _ => {}
            }
            // A bare identifier that is actually the START of the next rule
            // (`name ::=`) must terminate this sequence. Look ahead.
            if self.at_rule_header() {
                break;
            }
            let term = self.parse_term()?;
            if term.is_empty() {
                break;
            }
            if is_postfix(self.peek()) {
                // A repetition operator binds to the LAST atom of the term only
                // (GBNF/llama.cpp semantics): `"abc"*` == `"ab" "c"*`. For a
                // single-atom term (char class, group, rule ref, 1-char string)
                // the whole term is the atom.
                let (head, last_atom) = split_last_atom(term);
                out.extend(head);
                out.extend(self.apply_repetition(last_atom)?);
            } else {
                out.extend(term);
            }
        }
        Ok(out)
    }

    /// One term = string | char-class | group | rule-ref (WITHOUT its
    /// repetition suffix, which `apply_repetition` handles). Returns the
    /// element list that matches the term once.
    fn parse_term(&mut self) -> Result<Vec<Element>, GbnfError> {
        match self.peek() {
            Some(b'"') => self.parse_string(),
            Some(b'[') => self.parse_char_class(),
            Some(b'(') => {
                self.pos += 1; // '('
                let inner = self.parse_alternates()?;
                self.skip_inline_space_and_comments();
                if self.peek() != Some(b')') {
                    return Err(GbnfError::Parse("unterminated `(` group".to_string()));
                }
                self.pos += 1; // ')'
                let gid = self.new_anon_rule();
                self.rules[gid as usize] = inner;
                Ok(vec![Element::RuleRef(gid)])
            }
            Some(c) if is_name_start(c) => {
                let name = self.parse_name()?;
                let id = self.rule_id(&name);
                Ok(vec![Element::RuleRef(id)])
            }
            _ => Ok(Vec::new()),
        }
    }

    /// Apply a trailing `* + ? {m} {m,} {m,n}` (if any) to `term` by desugaring
    /// into synthetic rules. `term` is the single-match element list.
    fn apply_repetition(&mut self, term: Vec<Element>) -> Result<Vec<Element>, GbnfError> {
        match self.peek() {
            Some(b'*') => {
                self.pos += 1;
                Ok(vec![Element::RuleRef(self.make_star_rule(term))])
            }
            Some(b'+') => {
                self.pos += 1;
                // X+ == X X*
                let star = self.make_star_rule(term.clone());
                let rid = self.new_anon_rule();
                let mut body = term;
                body.push(Element::RuleRef(star));
                body.push(Element::End);
                self.rules[rid as usize] = body;
                Ok(vec![Element::RuleRef(rid)])
            }
            Some(b'?') => {
                self.pos += 1;
                // X? == ( X | )
                let rid = self.new_anon_rule();
                let mut body = term;
                body.push(Element::Alt);
                body.push(Element::End);
                self.rules[rid as usize] = body;
                Ok(vec![Element::RuleRef(rid)])
            }
            Some(b'{') => {
                let (min, max) = self.parse_repeat_range()?;
                Ok(vec![Element::RuleRef(
                    self.make_bounded_rule(term, min, max),
                )])
            }
            _ => Ok(term),
        }
    }

    /// Build a rule `S ::= term S |` matching zero-or-more of `term`.
    fn make_star_rule(&mut self, term: Vec<Element>) -> u32 {
        let rid = self.new_anon_rule();
        let mut body = term;
        body.push(Element::RuleRef(rid)); // self-reference (right recursion)
        body.push(Element::Alt); // empty alternate
        body.push(Element::End);
        self.rules[rid as usize] = body;
        rid
    }

    /// Desugar `term{min,max}` (max `None` = unbounded) into synthetic rules.
    fn make_bounded_rule(&mut self, term: Vec<Element>, min: u32, max: Option<u32>) -> u32 {
        let rid = self.new_anon_rule();
        let mut body: Vec<Element> = Vec::new();
        // `min` mandatory copies.
        for _ in 0..min {
            body.extend(term.iter().cloned());
        }
        match max {
            None => {
                // ... followed by term*
                let star = self.make_star_rule(term);
                body.push(Element::RuleRef(star));
            }
            Some(max) => {
                // ... then (max-min) optional copies, nested so later ones
                // require the earlier: build right-to-left as optionals.
                let extra = max.saturating_sub(min);
                if extra > 0 {
                    let opt = self.make_opt_chain_rule(term, extra);
                    body.push(Element::RuleRef(opt));
                }
            }
        }
        body.push(Element::End);
        self.rules[rid as usize] = body;
        rid
    }

    /// Build a rule matching 0..=`count` copies of `term` where each copy is
    /// optional but ordered: `R_k ::= term R_{k-1} |`, `R_0 ::= `.
    fn make_opt_chain_rule(&mut self, term: Vec<Element>, count: u32) -> u32 {
        // R_1 ::= term |
        let mut prev = {
            let rid = self.new_anon_rule();
            let mut body = term.clone();
            body.push(Element::Alt);
            body.push(Element::End);
            self.rules[rid as usize] = body;
            rid
        };
        for _ in 1..count {
            let rid = self.new_anon_rule();
            let mut body = term.clone();
            body.push(Element::RuleRef(prev));
            body.push(Element::Alt);
            body.push(Element::End);
            self.rules[rid as usize] = body;
            prev = rid;
        }
        prev
    }

    /// Parse `{m}` / `{m,}` / `{m,n}` after a term. `pos` is at `{`.
    fn parse_repeat_range(&mut self) -> Result<(u32, Option<u32>), GbnfError> {
        self.pos += 1; // '{'
        let min = self.parse_uint()?;
        self.skip_inline_space();
        match self.peek() {
            Some(b'}') => {
                self.pos += 1;
                Ok((min, Some(min)))
            }
            Some(b',') => {
                self.pos += 1;
                self.skip_inline_space();
                if self.peek() == Some(b'}') {
                    self.pos += 1;
                    Ok((min, None))
                } else {
                    let max = self.parse_uint()?;
                    self.skip_inline_space();
                    if self.peek() != Some(b'}') {
                        return Err(GbnfError::Parse("unterminated `{m,n}`".to_string()));
                    }
                    self.pos += 1;
                    if max < min {
                        return Err(GbnfError::Parse(format!(
                            "repeat range {{{min},{max}}} has max < min"
                        )));
                    }
                    Ok((min, Some(max)))
                }
            }
            _ => Err(GbnfError::Parse("malformed `{...}` repeat".to_string())),
        }
    }

    /// Parse a `"..."` string literal into a flat run of single-char `Char`
    /// elements (one singleton set per codepoint). A following `Char` neither
    /// extends the preceding set (not `CharRngUpper`/`CharAlt`) nor is a
    /// continuation, so each `Char` is matched as its own 1-char set.
    fn parse_string(&mut self) -> Result<Vec<Element>, GbnfError> {
        self.pos += 1; // opening quote
        let mut out: Vec<Element> = Vec::new();
        loop {
            match self.peek() {
                None => return Err(GbnfError::Parse("unterminated string literal".to_string())),
                Some(b'"') => {
                    self.pos += 1;
                    break;
                }
                Some(b'\\') => {
                    let cp = self.parse_escape()?;
                    out.push(Element::Char(cp));
                }
                Some(_) => {
                    let cp = self.next_codepoint()?;
                    out.push(Element::Char(cp));
                }
            }
        }
        Ok(out)
    }

    /// Parse a `[...]` character class into `Char`/`CharNot` + `CharRngUpper`/
    /// `CharAlt` elements.
    fn parse_char_class(&mut self) -> Result<Vec<Element>, GbnfError> {
        self.pos += 1; // '['
        let negate = self.peek() == Some(b'^');
        if negate {
            self.pos += 1;
        }
        let mut out: Vec<Element> = Vec::new();
        let mut first = true;
        loop {
            match self.peek() {
                None => return Err(GbnfError::Parse("unterminated `[` char class".to_string())),
                Some(b']') => {
                    self.pos += 1;
                    break;
                }
                _ => {
                    let lo = if self.peek() == Some(b'\\') {
                        self.parse_escape()?
                    } else {
                        self.next_codepoint()?
                    };
                    // Range?  `a-z` (but a trailing `-]` is a literal `-`).
                    let is_range = self.peek() == Some(b'-') && self.peek_at(1) != Some(b']');
                    if first {
                        out.push(if negate {
                            Element::CharNot(lo)
                        } else {
                            Element::Char(lo)
                        });
                        first = false;
                    } else {
                        out.push(Element::CharAlt(lo));
                    }
                    if is_range {
                        self.pos += 1; // '-'
                        let hi = if self.peek() == Some(b'\\') {
                            self.parse_escape()?
                        } else {
                            self.next_codepoint()?
                        };
                        out.push(Element::CharRngUpper(hi));
                    }
                }
            }
        }
        if out.is_empty() {
            return Err(GbnfError::Parse("empty `[]` char class".to_string()));
        }
        Ok(out)
    }

    /// Parse a `\`-escape (after the backslash) → codepoint.
    fn parse_escape(&mut self) -> Result<u32, GbnfError> {
        self.pos += 1; // backslash
        let c = self
            .peek()
            .ok_or_else(|| GbnfError::Parse("dangling `\\` escape".to_string()))?;
        self.pos += 1;
        Ok(match c {
            b'n' => b'\n' as u32,
            b'r' => b'\r' as u32,
            b't' => b'\t' as u32,
            b'\\' => b'\\' as u32,
            b'"' => b'"' as u32,
            b'\'' => b'\'' as u32,
            b'[' => b'[' as u32,
            b']' => b']' as u32,
            b'-' => b'-' as u32,
            b'x' => self.parse_hex(2)?,
            b'u' => self.parse_hex(4)?,
            b'U' => self.parse_hex(8)?,
            other => {
                return Err(GbnfError::Parse(format!(
                    "unknown escape `\\{}`",
                    other as char
                )))
            }
        })
    }

    fn parse_hex(&mut self, n: usize) -> Result<u32, GbnfError> {
        let mut v: u32 = 0;
        for _ in 0..n {
            let c = self
                .peek()
                .ok_or_else(|| GbnfError::Parse("truncated hex escape".to_string()))?;
            let d = (c as char)
                .to_digit(16)
                .ok_or_else(|| GbnfError::Parse(format!("bad hex digit `{}`", c as char)))?;
            v = v * 16 + d;
            self.pos += 1;
        }
        Ok(v)
    }

    /// Decode the next UTF-8 codepoint from the source.
    fn next_codepoint(&mut self) -> Result<u32, GbnfError> {
        let rest = &self.src[self.pos..];
        let s = std::str::from_utf8(rest)
            .map_err(|_| GbnfError::Parse("invalid UTF-8 in grammar".to_string()))?;
        let c = s
            .chars()
            .next()
            .ok_or_else(|| GbnfError::Parse("unexpected end of grammar".to_string()))?;
        self.pos += c.len_utf8();
        Ok(c as u32)
    }

    fn parse_uint(&mut self) -> Result<u32, GbnfError> {
        let start = self.pos;
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
            self.pos += 1;
        }
        if self.pos == start {
            return Err(GbnfError::Parse("expected a number".to_string()));
        }
        std::str::from_utf8(&self.src[start..self.pos])
            .unwrap()
            .parse::<u32>()
            .map_err(|_| GbnfError::Parse("number out of range".to_string()))
    }

    fn parse_name(&mut self) -> Result<String, GbnfError> {
        self.skip_inline_space();
        let start = self.pos;
        if !matches!(self.peek(), Some(c) if is_name_start(c)) {
            return Err(GbnfError::Parse("expected a rule name".to_string()));
        }
        while matches!(self.peek(), Some(c) if is_name_char(c)) {
            self.pos += 1;
        }
        Ok(String::from_utf8_lossy(&self.src[start..self.pos]).into_owned())
    }

    /// Lookahead: are we positioned at the start of a NEW rule header
    /// (`name ::=`)?  Used to end a sequence when the grammar flows onto the
    /// next rule without a blank line.
    fn at_rule_header(&self) -> bool {
        let mut i = self.pos;
        if !matches!(self.src.get(i), Some(&c) if is_name_start(c)) {
            return false;
        }
        while matches!(self.src.get(i), Some(&c) if is_name_char(c)) {
            i += 1;
        }
        while matches!(self.src.get(i), Some(&c) if c == b' ' || c == b'\t') {
            i += 1;
        }
        self.src[i..].starts_with(b"::=")
    }

    // --- low-level cursor helpers ---

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn peek_at(&self, off: usize) -> Option<u8> {
        self.src.get(self.pos + off).copied()
    }

    fn consume_str(&mut self, s: &str) -> bool {
        if self.src[self.pos..].starts_with(s.as_bytes()) {
            self.pos += s.len();
            true
        } else {
            false
        }
    }

    /// Skip spaces, tabs, newlines, and `#` comments.
    fn skip_space_and_comments(&mut self) {
        loop {
            match self.peek() {
                Some(b' ') | Some(b'\t') | Some(b'\r') | Some(b'\n') => self.pos += 1,
                Some(b'#') => self.skip_line(),
                _ => break,
            }
        }
    }

    /// Skip spaces/tabs only (not newlines).
    fn skip_inline_space(&mut self) {
        while matches!(self.peek(), Some(b' ') | Some(b'\t')) {
            self.pos += 1;
        }
    }

    /// Skip spaces/tabs and `#` comments, but NOT newlines (which bound rules).
    fn skip_inline_space_and_comments(&mut self) {
        loop {
            match self.peek() {
                Some(b' ') | Some(b'\t') => self.pos += 1,
                Some(b'#') => {
                    // A comment runs to end of line; consuming it leaves the
                    // newline for the sequence loop to see as a boundary.
                    self.skip_line_no_newline();
                }
                _ => break,
            }
        }
    }

    fn skip_line(&mut self) {
        while let Some(c) = self.peek() {
            self.pos += 1;
            if c == b'\n' {
                break;
            }
        }
    }

    fn skip_line_no_newline(&mut self) {
        while let Some(c) = self.peek() {
            if c == b'\n' {
                break;
            }
            self.pos += 1;
        }
    }
}

fn is_name_start(c: u8) -> bool {
    c.is_ascii_alphabetic() || c == b'_'
}

fn is_name_char(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_' || c == b'-'
}

/// Is the next byte a repetition-operator start?
fn is_postfix(c: Option<u8>) -> bool {
    matches!(c, Some(b'*') | Some(b'+') | Some(b'?') | Some(b'{'))
}

/// Split a term's element list into (leading elements, last atom). An ATOM is
/// one match unit: a rule reference (1 element) or a character set (a
/// `Char`/`CharNot` plus any trailing `CharRngUpper`/`CharAlt`). The last atom
/// starts at the last `Char`/`CharNot`/`RuleRef` index; everything before it is
/// the head. A repetition operator binds to that last atom only.
fn split_last_atom(term: Vec<Element>) -> (Vec<Element>, Vec<Element>) {
    let start = term
        .iter()
        .rposition(|e| {
            matches!(
                e,
                Element::Char(_) | Element::CharNot(_) | Element::RuleRef(_)
            )
        })
        .unwrap_or(0);
    let head = term[..start].to_vec();
    let last = term[start..].to_vec();
    (head, last)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Step a byte string through a parser; return the final Step (Reject if
    /// any byte rejects).
    fn feed(p: &mut GbnfParser, s: &str) -> Step {
        let mut last = Step::Accept;
        for &b in s.as_bytes() {
            last = p.step(b);
            if last == Step::Reject {
                return Step::Reject;
            }
        }
        last
    }

    #[test]
    fn yes_no_alternation() {
        let g = CompiledGbnf::compile(r#"root ::= "yes" | "no""#).unwrap();
        // "yes" is accepted and terminable.
        let mut p = GbnfParser::new(g.clone());
        assert_ne!(feed(&mut p, "yes"), Step::Reject);
        assert!(p.can_terminate_now());
        // "no" is accepted and terminable.
        let mut p = GbnfParser::new(g.clone());
        assert_ne!(feed(&mut p, "no"), Step::Reject);
        assert!(p.can_terminate_now());
        // "maybe" is rejected at the first non-matching byte.
        let mut p = GbnfParser::new(g.clone());
        assert_eq!(feed(&mut p, "maybe"), Step::Reject);
        // A prefix "ye" is accepted but NOT yet terminable.
        let mut p = GbnfParser::new(g);
        assert_ne!(feed(&mut p, "ye"), Step::Reject);
        assert!(!p.can_terminate_now());
    }

    #[test]
    fn char_class_and_plus() {
        let g = CompiledGbnf::compile(r#"root ::= [0-9]+"#).unwrap();
        let mut p = GbnfParser::new(g.clone());
        assert_ne!(feed(&mut p, "1234"), Step::Reject);
        assert!(p.can_terminate_now());
        // Empty is not terminable (+ requires >= 1).
        let p = GbnfParser::new(g.clone());
        assert!(!p.can_terminate_now());
        // A letter is rejected.
        let mut p = GbnfParser::new(g);
        assert_eq!(feed(&mut p, "a"), Step::Reject);
    }

    #[test]
    fn negated_class() {
        let g = CompiledGbnf::compile(r#"root ::= [^ \n]+"#).unwrap();
        let mut p = GbnfParser::new(g.clone());
        assert_ne!(feed(&mut p, "abc"), Step::Reject);
        let mut p = GbnfParser::new(g);
        assert_eq!(feed(&mut p, " "), Step::Reject);
    }

    #[test]
    fn recursive_balanced_parens() {
        // Context-free: only a stack machine can accept this.
        let g = CompiledGbnf::compile(r#"root ::= "(" root ")" | """#).unwrap();
        let mut p = GbnfParser::new(g.clone());
        assert_ne!(feed(&mut p, "(())"), Step::Reject);
        assert!(p.can_terminate_now());
        let mut p = GbnfParser::new(g);
        // Unbalanced: ")" with no opener rejects.
        assert_eq!(feed(&mut p, ")"), Step::Reject);
    }

    #[test]
    fn optional_and_star() {
        let g = CompiledGbnf::compile(r#"root ::= "a" "b"? "c"*"#).unwrap();
        for s in ["a", "ab", "abc", "abccc", "ac"] {
            let mut p = GbnfParser::new(g.clone());
            assert_ne!(feed(&mut p, s), Step::Reject, "{s} should match");
            assert!(p.can_terminate_now(), "{s} should terminate");
        }
        let mut p = GbnfParser::new(g);
        assert_eq!(feed(&mut p, "abd"), Step::Reject);
    }

    #[test]
    fn bounded_repeat() {
        let g = CompiledGbnf::compile(r#"root ::= [a]{2,3}"#).unwrap();
        for (s, ok) in [("a", false), ("aa", true), ("aaa", true)] {
            let mut p = GbnfParser::new(g.clone());
            let r = feed(&mut p, s);
            assert_eq!(r != Step::Reject && p.can_terminate_now(), ok, "{s}");
        }
        // 4 'a's: the 4th is rejected.
        let mut p = GbnfParser::new(g);
        assert_eq!(feed(&mut p, "aaaa"), Step::Reject);
    }

    #[test]
    fn comments_and_whitespace() {
        let g =
            CompiledGbnf::compile("# a greeting grammar\nroot ::= \"hi\"  # trailing\n").unwrap();
        let mut p = GbnfParser::new(g);
        assert_ne!(feed(&mut p, "hi"), Step::Reject);
        assert!(p.can_terminate_now());
    }

    #[test]
    fn multi_rule_reference() {
        let g = CompiledGbnf::compile(
            "root ::= greeting \" \" name\ngreeting ::= \"hello\" | \"hi\"\nname ::= [A-Z] [a-z]*",
        )
        .unwrap();
        let mut p = GbnfParser::new(g.clone());
        assert_ne!(feed(&mut p, "hello World"), Step::Reject);
        assert!(p.can_terminate_now());
        let mut p = GbnfParser::new(g);
        assert_ne!(feed(&mut p, "hi A"), Step::Reject);
        assert!(p.can_terminate_now());
    }

    #[test]
    fn invalid_grammars_error() {
        // No root rule.
        assert!(CompiledGbnf::compile(r#"foo ::= "x""#).is_err());
        // Missing ::=.
        assert!(CompiledGbnf::compile(r#"root "x""#).is_err());
        // Unterminated string.
        assert!(CompiledGbnf::compile(r#"root ::= "x"#).is_err());
        // Undefined rule reference.
        assert!(CompiledGbnf::compile(r#"root ::= missing"#).is_err());
        // Unterminated char class.
        assert!(CompiledGbnf::compile(r#"root ::= [a-z"#).is_err());
        // Empty grammar.
        assert!(CompiledGbnf::compile("   \n # just a comment\n").is_err());
    }

    #[test]
    fn utf8_multibyte_literal() {
        // A multi-byte codepoint in a literal, matched byte-by-byte.
        let g = CompiledGbnf::compile("root ::= \"café\"").unwrap();
        let mut p = GbnfParser::new(g.clone());
        assert_ne!(feed(&mut p, "café"), Step::Reject);
        assert!(p.can_terminate_now());
        // Mid-codepoint is not terminable.
        let mut p = GbnfParser::new(g);
        let bytes = "café".as_bytes();
        // Feed all but the last byte of 'é' (which is 2 bytes).
        for &b in &bytes[..bytes.len() - 1] {
            assert_ne!(p.step(b), Step::Reject);
        }
        assert!(!p.can_terminate_now());
    }
}
