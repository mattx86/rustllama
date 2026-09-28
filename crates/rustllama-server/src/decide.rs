//! Typed-decision endpoints.
//!
//! Instead of generating prose, these return a **typed value + calibrated
//! probabilities** by scoring a small set of candidate options against the
//! loaded model and taking the softmax over their conditional
//! log-likelihoods. The primitives:
//!
//!   - `POST /v1/decide/choice`  — pick one from a known set of `options`
//!     (optional split-conformal `prediction_set` when `coverage` is set).
//!   - `POST /v1/decide/score`   — rate on an ordered scale of `levels`
//!     (returns the distribution + an expected value, and — when the levels
//!     parse as numbers — a `stddev` and a `p10`/`p90` interval).
//!   - `POST /v1/decide/boolean` — a yes/no `question` → P(yes).
//!   - `POST /v1/decide/rank`    — the FULL ranked order over `options`.
//!   - `POST /v1/decide/labels`  — independent yes/no per label in `labels`.
//!   - `POST /v1/decide/best_of` — score `candidates` by likelihood and pick
//!     the best (length-normalized).
//!   - `POST /v1/decide/tool`    — score which of `tools` best fits.
//!   - `POST /v1/score`          — sequence log-likelihood / perplexity of an
//!     `input` string or `messages`.
//!
//! The decision primitives reduce to [`CpuEngine::score_continuations`], which
//! teacher-forces each option as a continuation of the context and
//! snapshot/restores the KV + DeltaNet state so the live serving model is
//! never mutated. `/v1/score` uses the sibling [`CpuEngine::sequence_logprob`].
//!
//! Every response carries uncertainty signals — `entropy` (nats), `margin`
//! (top1 − top2 probability) and `effective_options` (`exp(entropy)`, the
//! perplexity of the decision distribution) — and honors an optional
//! `abstain_below`: when the top *calibrated* probability falls below it the
//! response is flagged `declined: true` (the full distribution is still
//! returned) rather than forcing a pick.
//!
//! Calibration (a stage of the auto-tune sweep) fits a per-model temperature
//! (divides the option logits before softmax) plus split-conformal quantiles;
//! both are read from the tuner cache when
//! `[tuning].auto_apply_decision_calibration` is on.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::AppState;

#[derive(Deserialize, Clone)]
pub struct DecideMsg {
    pub role: String,
    pub content: String,
}

#[derive(Deserialize, Clone)]
pub struct DecideRequest {
    #[serde(default)]
    pub model: Option<String>,
    /// Plain context string (used as-is). Mutually usable with `messages`.
    #[serde(default)]
    pub context: Option<String>,
    /// Chat messages, rendered via the model's chat template (with a
    /// generation prompt appended) when present.
    #[serde(default)]
    pub messages: Option<Vec<DecideMsg>>,
    /// choice / rank: the candidate set.
    #[serde(default)]
    pub options: Option<Vec<String>>,
    /// score: the ordered scale labels.
    #[serde(default)]
    pub levels: Option<Vec<String>>,
    /// boolean: the yes/no question.
    #[serde(default)]
    pub question: Option<String>,
    /// labels: the set of labels to score independently as yes/no.
    #[serde(default)]
    pub labels: Option<Vec<String>>,
    /// best_of: the candidate strings whose likelihood we compare.
    #[serde(default)]
    pub candidates: Option<Vec<String>>,
    /// tool: an OpenAI-shaped `tools` array (`[{"function":{"name":..}}]`).
    #[serde(default)]
    pub tools: Option<Value>,
    /// Length normalization for multi-token options: "mean" (default) or "sum".
    #[serde(default)]
    pub normalize: Option<String>,
    /// Abstention threshold on the top calibrated probability. When the
    /// winner's probability is below this, the response is flagged
    /// `declined` instead of forcing a confident pick.
    #[serde(default)]
    pub abstain_below: Option<f32>,
    /// choice: target coverage for a split-conformal `prediction_set`
    /// (e.g. 0.8 / 0.9 / 0.95). `None` keeps plain argmax behavior.
    #[serde(default)]
    pub coverage: Option<f32>,
}

// ----- uncertainty signals (shared) -----------------------------------------

/// Shannon `entropy` (nats), `margin` (top1 − top2 probability), and
/// `effective_options` (`exp(entropy)`, the perplexity of the distribution).
fn uncertainty_signals(probs: &[f32]) -> (f32, f32, f32) {
    let entropy: f32 = probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum();
    let mut sorted: Vec<f32> = probs.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let margin = match (sorted.first(), sorted.get(1)) {
        (Some(a), Some(b)) => a - b,
        (Some(a), None) => *a,
        _ => 0.0,
    };
    (entropy, margin, entropy.exp())
}

// ----- response shapes -------------------------------------------------------

#[derive(Serialize)]
struct ChoiceResponse {
    value: String,
    index: usize,
    probabilities: Vec<f32>,
    confidence: f32,
    logprobs: Vec<f32>,
    calibrated: bool,
    entropy: f32,
    margin: f32,
    effective_options: f32,
    declined: bool,
    /// Split-conformal prediction set (present only when `coverage` was
    /// requested AND a fitted quantile exists for it): every option whose
    /// calibrated probability meets the `1 − q` threshold.
    #[serde(skip_serializing_if = "Option::is_none")]
    prediction_set: Option<Vec<String>>,
}

#[derive(Serialize)]
struct ScoreResponse {
    value: String,
    index: usize,
    score: f32,
    probabilities: Vec<f32>,
    confidence: f32,
    calibrated: bool,
    entropy: f32,
    margin: f32,
    effective_options: f32,
    declined: bool,
    /// Present only when every level parses as a number: the prob-weighted
    /// standard deviation around the expected value, and a `p10`/`p90`
    /// interval from the cumulative distribution over sorted numeric levels.
    #[serde(skip_serializing_if = "Option::is_none")]
    stddev: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    p10: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    p90: Option<f32>,
}

#[derive(Serialize)]
struct BooleanResponse {
    value: bool,
    probability: f32,
    confidence: f32,
    calibrated: bool,
    entropy: f32,
    margin: f32,
    effective_options: f32,
    declined: bool,
}

#[derive(Serialize)]
struct RankItem {
    value: String,
    index: usize,
    probability: f32,
    logprob: f32,
}

#[derive(Serialize)]
struct RankResponse {
    ranking: Vec<RankItem>,
    calibrated: bool,
    entropy: f32,
    margin: f32,
    effective_options: f32,
    declined: bool,
}

#[derive(Serialize)]
struct LabelResult {
    label: String,
    value: bool,
    probability: f32,
    confidence: f32,
    entropy: f32,
    margin: f32,
    effective_options: f32,
    declined: bool,
    calibrated: bool,
}

#[derive(Serialize)]
struct LabelsResponse {
    labels: Vec<LabelResult>,
}

#[derive(Serialize)]
struct BestOfResponse {
    index: usize,
    value: String,
    /// Length-normalized log-likelihood per candidate (higher = better).
    scores: Vec<f32>,
    probabilities: Vec<f32>,
    confidence: f32,
    calibrated: bool,
    normalize: String,
    entropy: f32,
    margin: f32,
    effective_options: f32,
    declined: bool,
}

#[derive(Serialize)]
struct ToolResponse {
    tool: String,
    index: usize,
    confidence: f32,
    probabilities: Vec<f32>,
    /// Tool names in the same order as `probabilities`.
    tools: Vec<String>,
    calibrated: bool,
    entropy: f32,
    margin: f32,
    effective_options: f32,
    declined: bool,
}

/// Softmax + argmax over per-option scores → (probabilities, best_index).
fn softmax_argmax(scores: &[f32]) -> (Vec<f32>, usize) {
    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|e| if sum > 0.0 { e / sum } else { 0.0 }).collect();
    let best = probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    (probs, best)
}

/// Per-model decision calibration (temperature + conformal quantiles) from the
/// tuner cache, when `[tuning].auto_apply_decision_calibration` is on and an
/// entry exists.
fn load_decision_calibration(
    state: &AppState,
    model_key: &str,
) -> Option<rustllama_tuner::DecisionCalibration> {
    let cfg = state
        .config_path
        .as_ref()
        .and_then(|p| rustllama_config::load(p.as_ref()).ok())?;
    if !cfg.tuning.auto_apply_decision_calibration {
        return None;
    }
    let dir = rustllama_tuner::default_cache_dir()?;
    let key = rustllama_tuner::system_fingerprint();
    let t = rustllama_tuner::load_cache(&dir, &key).ok().flatten()?;
    t.decision_calibration.get(model_key).cloned()
}

/// The outcome of one decision-scoring pass.
struct DecisionOutcome {
    /// Summed conditional log-likelihood per option.
    raw_scores: Vec<f32>,
    /// Length-normalized (pre-calibration) per-option logit.
    normalized: Vec<f32>,
    /// Calibrated softmax probabilities over the options.
    probs: Vec<f32>,
    /// Argmax over `probs`.
    best: usize,
    /// Whether a per-model temperature was applied.
    calibrated: bool,
    /// The full calibration entry (carries conformal quantiles), if any.
    calibration: Option<rustllama_tuner::DecisionCalibration>,
}

/// Shared core: resolve model → admit → gate → build context + option tokens
/// → `score_continuations` → normalize → (temperature-scaled) softmax.
async fn run_decision(
    state: &AppState,
    req: &DecideRequest,
    option_strings: &[String],
) -> Result<DecisionOutcome, Response> {
    if option_strings.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "no options to decide among").into_response());
    }
    let Some(serving) = state.resolve(req.model.as_deref()).await else {
        return Err((StatusCode::NOT_FOUND, "model not loaded").into_response());
    };
    let model_key = serving.model_id.clone();
    let pregate = match serving.try_admit() {
        Ok(p) => p,
        Err(be) => return Err(be.into_response()),
    };
    let shared_cpu = serving.cpu_engine.as_ref().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "decision endpoints require a real (non-mock) engine",
        )
            .into_response()
    })?;
    let tokenizer = shared_cpu.tokenizer().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "decision endpoints require a model with a tokenizer",
        )
            .into_response()
    })?;

    // Build the context token ids (chat-templated when `messages` present,
    // else the plain `context` string).
    let context_text: String = if let Some(msgs) = req.messages.as_ref() {
        let tok_msgs: Vec<rustllama_tokenizer::ChatMessage<'_>> = msgs
            .iter()
            .map(|m| rustllama_tokenizer::ChatMessage {
                role: &m.role,
                content: &m.content,
            })
            .collect();
        match tokenizer.render_chat(&tok_msgs, true) {
            Ok(s) => s,
            Err(e) => {
                return Err(
                    (StatusCode::BAD_REQUEST, format!("chat render failed: {e}")).into_response(),
                )
            }
        }
    } else if let Some(c) = req.context.as_ref() {
        c.clone()
    } else {
        return Err((StatusCode::BAD_REQUEST, "provide `context` or `messages`").into_response());
    };

    let context_ids: Vec<i32> = match tokenizer.encode(&context_text, tokenizer.add_bos_token()) {
        Ok(ids) => ids.into_iter().map(|t| t as i32).collect(),
        Err(e) => return Err((StatusCode::BAD_REQUEST, format!("encode: {e}")).into_response()),
    };
    if context_ids.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "empty context after tokenization").into_response());
    }

    // Tokenize each option (no BOS / specials). Empty options are dropped.
    let mut option_tokens: Vec<Vec<u32>> = Vec::with_capacity(option_strings.len());
    for opt in option_strings {
        match tokenizer.encode(opt, false) {
            Ok(ids) if !ids.is_empty() => option_tokens.push(ids),
            Ok(_) => {
                return Err((
                    StatusCode::BAD_REQUEST,
                    format!("option {opt:?} tokenized to nothing"),
                )
                    .into_response())
            }
            Err(e) => {
                return Err(
                    (StatusCode::BAD_REQUEST, format!("encode option: {e}")).into_response(),
                )
            }
        }
    }

    // Wait for the gate, then use the per-request fork off the blocking pool.
    let permit = match pregate.acquire_gate().await {
        Ok(h) => h,
        Err(be) => return Err(be.into_response()),
    };
    let cpu = permit.cpu_engine.as_ref().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "decision endpoints require a real (non-mock) engine",
        )
            .into_response()
    })?;
    let cpu = cpu.clone();
    let ctx = context_ids.clone();
    let opts = option_tokens.clone();
    let scores = tokio::task::spawn_blocking(move || cpu.score_continuations(&ctx, &opts))
        .await
        .map_err(|e| {
            (StatusCode::INTERNAL_SERVER_ERROR, format!("decision task panicked: {e}"))
                .into_response()
        })?
        .map_err(|e| {
            (StatusCode::INTERNAL_SERVER_ERROR, format!("scoring failed: {e}")).into_response()
        })?;
    drop(permit);

    // Length-normalize (default "mean") so multi-token options aren't
    // penalized for their token count, then softmax across options.
    let norm = req.normalize.as_deref().unwrap_or("mean");
    let normalized: Vec<f32> = scores
        .iter()
        .zip(option_tokens.iter())
        .map(|(s, toks)| {
            if norm == "sum" || toks.is_empty() {
                *s
            } else {
                *s / toks.len() as f32
            }
        })
        .collect();
    // Decision-calibration: divide the per-option logits by the fitted per-model
    // temperature before softmax to calibrate the confidences. `None` (no
    // cache entry / flag off) leaves them at the raw softmax (T = 1).
    let calibration = load_decision_calibration(state, &model_key);
    let temperature = calibration
        .as_ref()
        .map(|c| c.temperature)
        .filter(|&t| t > 0.0);
    let calibrated = temperature.is_some();
    let t = temperature.unwrap_or(1.0);
    let scaled: Vec<f32> = normalized.iter().map(|v| v / t).collect();
    let (probs, best) = softmax_argmax(&scaled);
    Ok(DecisionOutcome {
        raw_scores: scores,
        normalized,
        probs,
        best,
        calibrated,
        calibration,
    })
}

/// Render `messages` (optionally with `tools` exposed) or return the plain
/// `context` string. Used by the `labels` and `tool` endpoints, which fold
/// per-item text into the context before scoring.
async fn resolve_context_text(
    state: &AppState,
    req: &DecideRequest,
    tools: Option<&Value>,
) -> Result<String, Response> {
    if let Some(msgs) = req.messages.as_ref() {
        let Some(serving) = state.resolve(req.model.as_deref()).await else {
            return Err((StatusCode::NOT_FOUND, "model not loaded").into_response());
        };
        let shared_cpu = serving.cpu_engine.as_ref().ok_or_else(|| {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "decision endpoints require a real (non-mock) engine",
            )
                .into_response()
        })?;
        let tokenizer = shared_cpu.tokenizer().ok_or_else(|| {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "decision endpoints require a model with a tokenizer",
            )
                .into_response()
        })?;
        match tools {
            Some(t) => {
                let json_msgs: Vec<Value> = msgs
                    .iter()
                    .map(|m| json!({"role": m.role, "content": m.content}))
                    .collect();
                tokenizer
                    .render_chat_messages_json(&json_msgs, true, Some(t))
                    .map_err(|e| {
                        (StatusCode::BAD_REQUEST, format!("chat render failed: {e}")).into_response()
                    })
            }
            None => {
                let tok_msgs: Vec<rustllama_tokenizer::ChatMessage<'_>> = msgs
                    .iter()
                    .map(|m| rustllama_tokenizer::ChatMessage {
                        role: &m.role,
                        content: &m.content,
                    })
                    .collect();
                tokenizer.render_chat(&tok_msgs, true).map_err(|e| {
                    (StatusCode::BAD_REQUEST, format!("chat render failed: {e}")).into_response()
                })
            }
        }
    } else if let Some(c) = req.context.as_ref() {
        Ok(c.clone())
    } else {
        Err((StatusCode::BAD_REQUEST, "provide `context` or `messages`").into_response())
    }
}

// ----- /v1/decide/choice -----------------------------------------------------

pub async fn choice(State(state): State<AppState>, Json(req): Json<DecideRequest>) -> Response {
    let options = match req.options.clone() {
        Some(o) if !o.is_empty() => o,
        _ => {
            return (StatusCode::BAD_REQUEST, "`options` is required for /decide/choice")
                .into_response()
        }
    };
    match run_decision(&state, &req, &options).await {
        Ok(o) => {
            let (entropy, margin, effective_options) = uncertainty_signals(&o.probs);
            let best_prob = o.probs.get(o.best).copied().unwrap_or(0.0);
            let declined = req.abstain_below.map(|thr| best_prob < thr).unwrap_or(false);
            // Split-conformal prediction set: only when `coverage` requested and
            // a fitted quantile exists for it. Smallest set whose members meet
            // the `1 − q` calibrated-probability bar.
            let prediction_set = req.coverage.and_then(|cov| {
                let q = o
                    .calibration
                    .as_ref()?
                    .conformal_q
                    .get(&format!("{cov:.2}"))
                    .copied()?;
                let thr = 1.0 - q;
                Some(
                    options
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| o.probs.get(*i).copied().unwrap_or(0.0) >= thr)
                        .map(|(_, v)| v.clone())
                        .collect::<Vec<String>>(),
                )
            });
            let value = options[o.best].clone();
            Json(ChoiceResponse {
                value,
                index: o.best,
                confidence: best_prob,
                probabilities: o.probs,
                logprobs: o.raw_scores,
                calibrated: o.calibrated,
                entropy,
                margin,
                effective_options,
                declined,
                prediction_set,
            })
            .into_response()
        }
        Err(resp) => resp,
    }
}

// ----- /v1/decide/score ------------------------------------------------------

pub async fn score(State(state): State<AppState>, Json(req): Json<DecideRequest>) -> Response {
    let levels = match req.levels.clone() {
        Some(l) if !l.is_empty() => l,
        _ => {
            return (StatusCode::BAD_REQUEST, "`levels` is required for /decide/score")
                .into_response()
        }
    };
    match run_decision(&state, &req, &levels).await {
        Ok(o) => {
            let probs = &o.probs;
            // Expected value: if every level parses as a number, weight by
            // that; otherwise weight by the level's index on the scale.
            let numeric: Option<Vec<f32>> =
                levels.iter().map(|l| l.trim().parse::<f32>().ok()).collect();
            let expected: f32 = match &numeric {
                Some(nums) => probs.iter().zip(nums).map(|(p, n)| p * n).sum(),
                None => probs.iter().enumerate().map(|(i, p)| p * i as f32).sum(),
            };
            // Regression extras when the levels are numeric: prob-weighted
            // stddev around the expected value, and a p10/p90 interval from
            // the cumulative distribution over sorted numeric levels.
            let (stddev, p10, p90) = match &numeric {
                Some(nums) => {
                    let variance: f32 = probs
                        .iter()
                        .zip(nums)
                        .map(|(p, n)| *p * (*n - expected).powi(2))
                        .sum();
                    let stddev = variance.max(0.0).sqrt();
                    let mut pairs: Vec<(f32, f32)> =
                        nums.iter().zip(probs.iter()).map(|(&n, &p)| (n, p)).collect();
                    pairs.sort_by(|a, b| {
                        a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let quantile = |q: f32| -> f32 {
                        let mut cum = 0.0f32;
                        let mut last = pairs.first().map(|p| p.0).unwrap_or(0.0);
                        for (n, p) in &pairs {
                            cum += *p;
                            last = *n;
                            if cum >= q {
                                return *n;
                            }
                        }
                        last
                    };
                    (Some(stddev), Some(quantile(0.10)), Some(quantile(0.90)))
                }
                None => (None, None, None),
            };
            let (entropy, margin, effective_options) = uncertainty_signals(probs);
            let best_prob = probs.get(o.best).copied().unwrap_or(0.0);
            let declined = req.abstain_below.map(|thr| best_prob < thr).unwrap_or(false);
            Json(ScoreResponse {
                value: levels[o.best].clone(),
                index: o.best,
                score: expected,
                confidence: best_prob,
                probabilities: o.probs.clone(),
                calibrated: o.calibrated,
                entropy,
                margin,
                effective_options,
                declined,
                stddev,
                p10,
                p90,
            })
            .into_response()
        }
        Err(resp) => resp,
    }
}

// ----- /v1/decide/boolean ----------------------------------------------------

pub async fn boolean(State(state): State<AppState>, Json(mut req): Json<DecideRequest>) -> Response {
    // Fold the yes/no question into the context so the options are scored as
    // its answer, then score a shared single-token-ish affirmative/negative
    // pair (leading space suits byte-level BPE continuations).
    if let Some(q) = req.question.clone() {
        req.context = Some(match req.context.take() {
            Some(c) if !c.is_empty() => format!("{c}\n{q}"),
            _ => q,
        });
        // A question via `context` means we don't chat-template it.
        req.messages = None;
    }
    let options = vec![" yes".to_string(), " no".to_string()];
    match run_decision(&state, &req, &options).await {
        Ok(o) => {
            let p_yes = o.probs.first().copied().unwrap_or(0.0);
            let confidence = p_yes.max(1.0 - p_yes);
            let (entropy, margin, effective_options) = uncertainty_signals(&o.probs);
            let declined = req.abstain_below.map(|thr| confidence < thr).unwrap_or(false);
            Json(BooleanResponse {
                value: p_yes >= 0.5,
                probability: p_yes,
                confidence,
                calibrated: o.calibrated,
                entropy,
                margin,
                effective_options,
                declined,
            })
            .into_response()
        }
        Err(resp) => resp,
    }
}

// ----- /v1/decide/rank -------------------------------------------------------

pub async fn rank(State(state): State<AppState>, Json(req): Json<DecideRequest>) -> Response {
    let options = match req.options.clone() {
        Some(o) if !o.is_empty() => o,
        _ => {
            return (StatusCode::BAD_REQUEST, "`options` is required for /decide/rank")
                .into_response()
        }
    };
    match run_decision(&state, &req, &options).await {
        Ok(o) => {
            let (entropy, margin, effective_options) = uncertainty_signals(&o.probs);
            let best_prob = o.probs.get(o.best).copied().unwrap_or(0.0);
            let declined = req.abstain_below.map(|thr| best_prob < thr).unwrap_or(false);
            let mut ranking: Vec<RankItem> = options
                .iter()
                .enumerate()
                .map(|(i, v)| RankItem {
                    value: v.clone(),
                    index: i,
                    probability: o.probs.get(i).copied().unwrap_or(0.0),
                    logprob: o.raw_scores.get(i).copied().unwrap_or(f32::NEG_INFINITY),
                })
                .collect();
            ranking.sort_by(|a, b| {
                b.probability
                    .partial_cmp(&a.probability)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            Json(RankResponse {
                ranking,
                calibrated: o.calibrated,
                entropy,
                margin,
                effective_options,
                declined,
            })
            .into_response()
        }
        Err(resp) => resp,
    }
}

// ----- /v1/decide/labels -----------------------------------------------------

pub async fn labels(State(state): State<AppState>, Json(req): Json<DecideRequest>) -> Response {
    let labels = match req.labels.clone() {
        Some(l) if !l.is_empty() => l,
        _ => {
            return (StatusCode::BAD_REQUEST, "`labels` is required for /decide/labels")
                .into_response()
        }
    };
    // Render the base context once (preserving `messages`), then fold each
    // label into it and score " yes"/" no" independently — like `boolean`.
    let base = match resolve_context_text(&state, &req, None).await {
        Ok(s) => s,
        Err(resp) => return resp,
    };
    let yesno = vec![" yes".to_string(), " no".to_string()];
    let mut results = Vec::with_capacity(labels.len());
    for label in &labels {
        let mut sub = req.clone();
        sub.context = Some(if base.is_empty() {
            label.clone()
        } else {
            format!("{base}\n{label}")
        });
        sub.messages = None;
        match run_decision(&state, &sub, &yesno).await {
            Ok(o) => {
                let p_yes = o.probs.first().copied().unwrap_or(0.0);
                let confidence = p_yes.max(1.0 - p_yes);
                let (entropy, margin, effective_options) = uncertainty_signals(&o.probs);
                let declined = req.abstain_below.map(|thr| confidence < thr).unwrap_or(false);
                results.push(LabelResult {
                    label: label.clone(),
                    value: p_yes >= 0.5,
                    probability: p_yes,
                    confidence,
                    entropy,
                    margin,
                    effective_options,
                    declined,
                    calibrated: o.calibrated,
                });
            }
            Err(resp) => return resp,
        }
    }
    Json(LabelsResponse { labels: results }).into_response()
}

// ----- /v1/decide/best_of ----------------------------------------------------

pub async fn best_of(State(state): State<AppState>, Json(req): Json<DecideRequest>) -> Response {
    let candidates = match req.candidates.clone() {
        Some(c) if !c.is_empty() => c,
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                "`candidates` is required for /decide/best_of",
            )
                .into_response()
        }
    };
    match run_decision(&state, &req, &candidates).await {
        Ok(o) => {
            let (entropy, margin, effective_options) = uncertainty_signals(&o.probs);
            let conf = o.probs.get(o.best).copied().unwrap_or(0.0);
            let declined = req.abstain_below.map(|thr| conf < thr).unwrap_or(false);
            let normalize = req.normalize.clone().unwrap_or_else(|| "mean".to_string());
            Json(BestOfResponse {
                index: o.best,
                value: candidates[o.best].clone(),
                scores: o.normalized,
                probabilities: o.probs,
                confidence: conf,
                calibrated: o.calibrated,
                normalize,
                entropy,
                margin,
                effective_options,
                declined,
            })
            .into_response()
        }
        Err(resp) => resp,
    }
}

// ----- /v1/decide/tool -------------------------------------------------------

/// Pick the tool-call opener to fold into the scoring context, matching the
/// model's chat-template framing. Every supported format wraps the call in a
/// JSON object that begins `{"name": "`, so the family marker (when the
/// template reveals one) just puts that JSON on-distribution:
///   - Qwen / DeepSeek → `<tool_call>\n{"name": "`
///   - Llama-3.1       → `<|python_tag|>{"name": "`
///   - Mistral         → `[TOOL_CALLS][{"name": "`
///   - unknown / none  → bare `{"name": "` (format-agnostic; fits them all)
fn tool_call_opener(template: Option<&str>) -> String {
    const JSON: &str = "{\"name\": \"";
    match template {
        Some(t) if t.contains("<tool_call>") => format!("<tool_call>\n{JSON}"),
        Some(t) if t.contains("python_tag") => format!("<|python_tag|>{JSON}"),
        Some(t) if t.contains("[TOOL_CALLS]") => format!("[TOOL_CALLS][{JSON}"),
        _ => JSON.to_string(),
    }
}

pub async fn tool(State(state): State<AppState>, Json(req): Json<DecideRequest>) -> Response {
    let tools = match req.tools.as_ref() {
        Some(t) => t.clone(),
        None => {
            return (StatusCode::BAD_REQUEST, "`tools` is required for /decide/tool")
                .into_response()
        }
    };
    // Extract the named functions.
    let names: Vec<String> = match tools.as_array() {
        Some(arr) => arr
            .iter()
            .filter_map(|t| {
                t.get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|n| n.as_str())
                    .map(|s| s.to_string())
            })
            .collect(),
        None => Vec::new(),
    };
    if names.is_empty() {
        return (StatusCode::BAD_REQUEST, "no named functions in `tools`").into_response();
    }
    // Derive the tool-call opener from the model's chat template so scoring
    // primes the model in its NATIVE framing (Qwen `<tool_call>`, Llama-3.1
    // `<|python_tag|>`, Mistral `[TOOL_CALLS]`) instead of hardcoding the Qwen
    // form — mirrors the multi-format handling the chat tool path does. Falls
    // back to a neutral JSON opener that fits every format when the template
    // is unrecognized / absent.
    let template = state
        .resolve(req.model.as_deref())
        .await
        .and_then(|s| {
            s.cpu_engine
                .as_ref()
                .and_then(|c| c.tokenizer())
                .and_then(|t| t.chat_template())
                .map(|t| t.to_string())
        });
    let opener = tool_call_opener(template.as_deref());
    // Render the prompt with the tools exposed, then fold the derived opener
    // so each option scores just the tool NAME (the discriminating suffix),
    // with the shared opener living in the context.
    let base = match resolve_context_text(&state, &req, Some(&tools)).await {
        Ok(s) => s,
        Err(resp) => return resp,
    };
    let folded = format!("{base}{opener}");
    let options: Vec<String> = names.iter().map(|n| format!("{n}\"")).collect();

    let mut sub = req.clone();
    sub.context = Some(folded);
    sub.messages = None;
    sub.tools = None;
    match run_decision(&state, &sub, &options).await {
        Ok(o) => {
            let (entropy, margin, effective_options) = uncertainty_signals(&o.probs);
            let conf = o.probs.get(o.best).copied().unwrap_or(0.0);
            let declined = req.abstain_below.map(|thr| conf < thr).unwrap_or(false);
            Json(ToolResponse {
                tool: names[o.best].clone(),
                index: o.best,
                confidence: conf,
                probabilities: o.probs,
                tools: names,
                calibrated: o.calibrated,
                entropy,
                margin,
                effective_options,
                declined,
            })
            .into_response()
        }
        Err(resp) => resp,
    }
}

// ----- /v1/score (sequence likelihood / perplexity) --------------------------

#[derive(Deserialize)]
pub struct SequenceScoreRequest {
    #[serde(default)]
    pub model: Option<String>,
    /// A plain input string to score.
    #[serde(default)]
    pub input: Option<String>,
    /// Chat messages, chat-templated then scored, when `input` is absent.
    #[serde(default)]
    pub messages: Option<Vec<DecideMsg>>,
    /// When true, include per-token `{token, logprob}` entries.
    #[serde(default)]
    pub logprobs: Option<bool>,
}

#[derive(Serialize)]
struct PerTokenLogprob {
    token: String,
    logprob: f32,
}

#[derive(Serialize)]
struct SequenceScoreResponse {
    /// Total log-likelihood of the scored tokens (all but the first).
    logprob: f32,
    /// `exp(−mean token logprob)` over the scored tokens.
    perplexity: f32,
    /// The number of scored tokens (sequence length minus one).
    token_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    per_token: Option<Vec<PerTokenLogprob>>,
}

pub async fn sequence_score(
    State(state): State<AppState>,
    Json(req): Json<SequenceScoreRequest>,
) -> Response {
    let Some(serving) = state.resolve(req.model.as_deref()).await else {
        return (StatusCode::NOT_FOUND, "model not loaded").into_response();
    };
    let pregate = match serving.try_admit() {
        Ok(p) => p,
        Err(be) => return be.into_response(),
    };
    let shared_cpu = match serving.cpu_engine.as_ref() {
        Some(e) => e,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                "/v1/score requires a real (non-mock) engine",
            )
                .into_response()
        }
    };
    let tokenizer = match shared_cpu.tokenizer() {
        Some(t) => t,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                "/v1/score requires a model with a tokenizer",
            )
                .into_response()
        }
    };

    let text: String = if let Some(msgs) = req.messages.as_ref() {
        let tok_msgs: Vec<rustllama_tokenizer::ChatMessage<'_>> = msgs
            .iter()
            .map(|m| rustllama_tokenizer::ChatMessage {
                role: &m.role,
                content: &m.content,
            })
            .collect();
        // Score the conversation as written — NOT with `add_generation_prompt`.
        // Appending the assistant generation prompt would fold the template's
        // trailing role tokens into the sequence, inflating `sequence_logprob`
        // with tokens the caller never supplied.
        match tokenizer.render_chat(&tok_msgs, false) {
            Ok(s) => s,
            Err(e) => {
                return (StatusCode::BAD_REQUEST, format!("chat render failed: {e}"))
                    .into_response()
            }
        }
    } else if let Some(i) = req.input.as_ref() {
        i.clone()
    } else {
        return (StatusCode::BAD_REQUEST, "provide `input` or `messages`").into_response();
    };

    let ids: Vec<i32> = match tokenizer.encode(&text, tokenizer.add_bos_token()) {
        Ok(v) => v.into_iter().map(|t| t as i32).collect(),
        Err(e) => return (StatusCode::BAD_REQUEST, format!("encode: {e}")).into_response(),
    };
    if ids.len() < 2 {
        return (
            StatusCode::BAD_REQUEST,
            "sequence must tokenize to at least 2 tokens to score",
        )
            .into_response();
    }

    // Decode the scored tokens (ids[1..]) up front while the tokenizer is in
    // hand, so the per-token output can name them without a second pass.
    let want_logprobs = req.logprobs.unwrap_or(false);
    let token_strs: Option<Vec<String>> = if want_logprobs {
        Some(
            ids[1..]
                .iter()
                .map(|&id| tokenizer.decode_single(id as u32, true).unwrap_or_default())
                .collect(),
        )
    } else {
        None
    };

    let permit = match pregate.acquire_gate().await {
        Ok(h) => h,
        Err(be) => return be.into_response(),
    };
    let cpu = match permit.cpu_engine.as_ref() {
        Some(e) => e.clone(),
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                "/v1/score requires a real (non-mock) engine",
            )
                .into_response()
        }
    };
    let ids_clone = ids.clone();
    let res = tokio::task::spawn_blocking(move || cpu.sequence_logprob(&ids_clone)).await;
    drop(permit);
    let (total, per_token) = match res {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            return (StatusCode::INTERNAL_SERVER_ERROR, format!("scoring failed: {e}"))
                .into_response()
        }
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("scoring task panicked: {e}"),
            )
                .into_response()
        }
    };

    let token_count = per_token.len();
    let perplexity = if token_count > 0 {
        (-(total / token_count as f32)).exp()
    } else {
        1.0
    };
    let per_token_out = token_strs.map(|toks| {
        toks.into_iter()
            .zip(per_token.iter())
            .map(|(token, &logprob)| PerTokenLogprob { token, logprob })
            .collect::<Vec<_>>()
    });
    Json(SequenceScoreResponse {
        logprob: total,
        perplexity,
        token_count,
        per_token: per_token_out,
    })
    .into_response()
}
