//! Conversation history persistence (sqlite-backed).
//!
//! Gated behind feature `history`. The GUI's Chat page sidebar lists
//! prior conversations and lets users return to one; CLI / server-only
//! deployments leave the feature off so the bundled `rusqlite` C compile
//! doesn't slow the build.
//!
//! Schema (current version, v1):
//!
//! ```sql
//! CREATE TABLE conversations (
//!   id          INTEGER PRIMARY KEY AUTOINCREMENT,
//!   title       TEXT NOT NULL,
//!   created_at  INTEGER NOT NULL,
//!   updated_at  INTEGER NOT NULL
//! );
//! CREATE TABLE messages (
//!   id              INTEGER PRIMARY KEY AUTOINCREMENT,
//!   conversation_id INTEGER NOT NULL,
//!   role            TEXT NOT NULL,
//!   content         TEXT NOT NULL,
//!   created_at      INTEGER NOT NULL,
//!   FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
//! );
//! CREATE INDEX idx_messages_conv ON messages(conversation_id);
//! ```
//!
//! Full-text search is wired via the sqlite `fts5` extension (bundled
//! with rusqlite's `bundled` feature):
//!
//! ```sql
//! CREATE VIRTUAL TABLE messages_fts USING fts5(
//!   content, content='messages', content_rowid='id', tokenize='porter unicode61'
//! );
//! ```
//!
//! INSERT/UPDATE/DELETE on `messages` keep `messages_fts` in sync via
//! AFTER triggers. `search_messages(q)` runs `MATCH` against the FTS
//! table and returns the matching messages with their conversation
//! context + an HTML-marked excerpt snippet.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};

use crate::AppState;

/// Wrapper around a `Mutex<Connection>`. sqlite's threading model is
/// "serialized" by default but we want to be explicit about contention
/// — the GUI is the only writer, and reads from the listing endpoint
/// are tiny. A Mutex is fine and avoids the connection-pool ceremony.
pub struct HistoryStore {
    conn: Mutex<Connection>,
}

#[derive(Debug, thiserror::Error)]
pub enum HistoryError {
    #[error("sqlite: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, HistoryError>;

impl HistoryStore {
    /// Open (or create) the history database at `path`. Runs the
    /// schema migrations idempotently.
    pub fn open(path: PathBuf) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(&path)?;
        // Enable foreign keys so DELETE FROM conversations cascades.
        conn.execute_batch("PRAGMA foreign_keys = ON;")?;
        conn.execute_batch(SCHEMA_SQL)?;
        backfill_fts_if_needed(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// In-memory store for tests. Drops everything when the
    /// `HistoryStore` goes out of scope.
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch("PRAGMA foreign_keys = ON;")?;
        conn.execute_batch(SCHEMA_SQL)?;
        backfill_fts_if_needed(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Full-text search over message contents. `query` accepts the
    /// sqlite FTS5 MATCH syntax — bare words behave as AND
    /// (`"deploy migration"` matches messages containing both),
    /// quoted phrases require the exact sequence, `*` is a prefix
    /// wildcard. Results are ranked by FTS5 `bm25()` (lower = more
    /// relevant) and joined back to the conversation + message
    /// metadata + an HTML-marked excerpt snippet.
    pub fn search_messages(&self, query: &str, limit: u32) -> Result<Vec<SearchHit>> {
        let conn = self.conn.lock().expect("history conn lock");
        let limit = limit.max(1).min(200) as i64;
        let mut stmt = conn.prepare(
            "SELECT m.id, m.conversation_id, c.title, m.role, m.created_at, \
                    snippet(messages_fts, 0, '<mark>', '</mark>', '…', 16), \
                    bm25(messages_fts) AS rank \
             FROM messages_fts \
             JOIN messages m ON m.id = messages_fts.rowid \
             JOIN conversations c ON c.id = m.conversation_id \
             WHERE messages_fts MATCH ?1 \
             ORDER BY rank ASC \
             LIMIT ?2",
        )?;
        let rows = stmt
            .query_map(params![query, limit], |r| {
                Ok(SearchHit {
                    message_id: r.get(0)?,
                    conversation_id: r.get(1)?,
                    conversation_title: r.get(2)?,
                    role: r.get(3)?,
                    created_at: r.get(4)?,
                    snippet: r.get(5)?,
                    score: r.get::<_, f64>(6)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// List all conversations, newest-first by `updated_at`. Returns
    /// the lightweight `ConversationSummary` shape (no message bodies)
    /// so the sidebar render is cheap.
    pub fn list_conversations(&self) -> Result<Vec<ConversationSummary>> {
        let conn = self.conn.lock().expect("history conn lock");
        let mut stmt = conn.prepare(
            "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC",
        )?;
        let rows = stmt
            .query_map([], |r| {
                Ok(ConversationSummary {
                    id: r.get(0)?,
                    title: r.get(1)?,
                    created_at: r.get(2)?,
                    updated_at: r.get(3)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Fetch a conversation by id, including all its messages in
    /// chronological order.
    pub fn get_conversation(&self, id: i64) -> Result<Option<Conversation>> {
        let conn = self.conn.lock().expect("history conn lock");
        let mut head = conn.prepare(
            "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?1",
        )?;
        let summary: Option<ConversationSummary> = head
            .query_row(params![id], |r| {
                Ok(ConversationSummary {
                    id: r.get(0)?,
                    title: r.get(1)?,
                    created_at: r.get(2)?,
                    updated_at: r.get(3)?,
                })
            })
            .ok();
        let Some(summary) = summary else {
            return Ok(None);
        };
        let mut q = conn.prepare(
            "SELECT id, role, content, created_at FROM messages \
             WHERE conversation_id = ?1 ORDER BY id ASC",
        )?;
        let messages = q
            .query_map(params![id], |r| {
                Ok(StoredMessage {
                    id: r.get(0)?,
                    role: r.get(1)?,
                    content: r.get(2)?,
                    created_at: r.get(3)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(Some(Conversation { summary, messages }))
    }

    /// Create a new conversation with the given title. Returns the
    /// new id.
    pub fn create_conversation(&self, title: &str) -> Result<i64> {
        let now = unix_secs();
        let conn = self.conn.lock().expect("history conn lock");
        conn.execute(
            "INSERT INTO conversations (title, created_at, updated_at) VALUES (?1, ?2, ?2)",
            params![title, now as i64],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// Append a message to a conversation and bump its `updated_at`.
    /// The bump matters for the sidebar's "newest first" ordering — a
    /// long-running thread that gets a fresh reply hops to the top.
    pub fn append_message(
        &self,
        conversation_id: i64,
        role: &str,
        content: &str,
    ) -> Result<i64> {
        let now = unix_secs();
        let conn = self.conn.lock().expect("history conn lock");
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) \
             VALUES (?1, ?2, ?3, ?4)",
            params![conversation_id, role, content, now as i64],
        )?;
        conn.execute(
            "UPDATE conversations SET updated_at = ?1 WHERE id = ?2",
            params![now as i64, conversation_id],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// Delete a conversation and all its messages.
    pub fn delete_conversation(&self, id: i64) -> Result<bool> {
        let conn = self.conn.lock().expect("history conn lock");
        let affected = conn.execute(
            "DELETE FROM conversations WHERE id = ?1",
            params![id],
        )?;
        Ok(affected > 0)
    }

    /// Update a conversation's title. Useful for an "auto-rename
    /// after first response" flow that the GUI can wire up.
    pub fn rename(&self, id: i64, title: &str) -> Result<bool> {
        let conn = self.conn.lock().expect("history conn lock");
        let affected = conn.execute(
            "UPDATE conversations SET title = ?1, updated_at = ?2 WHERE id = ?3",
            params![title, unix_secs() as i64, id],
        )?;
        Ok(affected > 0)
    }
}

/// If the messages table has rows but the FTS index is empty (i.e.,
/// the DB was created before the FTS5 schema landed), rebuild the
/// index in one shot. Cheap when the index is already populated —
/// the COUNT comparison is the only work in that case.
fn backfill_fts_if_needed(conn: &Connection) -> Result<()> {
    let messages_count: i64 = conn.query_row("SELECT COUNT(*) FROM messages", [], |r| r.get(0))?;
    if messages_count == 0 {
        return Ok(());
    }
    let fts_count: i64 =
        conn.query_row("SELECT COUNT(*) FROM messages_fts", [], |r| r.get(0))?;
    if fts_count < messages_count {
        // The FTS5 'rebuild' command repopulates from the external
        // content table. Safe to run repeatedly.
        conn.execute_batch(
            "INSERT INTO messages_fts(messages_fts) VALUES ('rebuild');",
        )?;
        tracing::info!(
            messages = messages_count,
            previously_indexed = fts_count,
            "rebuilt conversation-history FTS index for existing rows"
        );
    }
    Ok(())
}

const SCHEMA_SQL: &str = "
CREATE TABLE IF NOT EXISTS conversations (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  title       TEXT NOT NULL,
  created_at  INTEGER NOT NULL,
  updated_at  INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  conversation_id INTEGER NOT NULL,
  role            TEXT NOT NULL,
  content         TEXT NOT NULL,
  created_at      INTEGER NOT NULL,
  FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);

-- FTS5 external-content virtual table over messages.content.
-- `content='messages'` means the FTS index stores only the
-- inverted lookup; the actual text comes from messages.content
-- so we don't duplicate storage. `content_rowid='id'` ties the
-- FTS rowid to messages.id.
--
-- Tokenizer: unicode61 only (case-fold + diacritic-strip).
-- Porter stemming was tried but rejected — it collides with
-- prefix-wildcard intent (`deploy*` after stemming searches
-- different stems than the user typed) and changes the
-- semantics of code identifiers in unpredictable ways. Case-
-- folding alone gives good ergonomics for both prose and code.
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
  content,
  content='messages',
  content_rowid='id',
  tokenize='unicode61'
);

-- Sync triggers. The external-content FTS5 idiom: each write to
-- messages must mirror itself into messages_fts via the
-- `messages_fts(rowid, content)` insert form (for inserts) or
-- the `(messages_fts, rowid, content)` 'delete' command form
-- (for deletes — required because external-content tables
-- can't self-derive what bytes were removed).
CREATE TRIGGER IF NOT EXISTS messages_ai
AFTER INSERT ON messages BEGIN
  INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS messages_ad
AFTER DELETE ON messages BEGIN
  INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.id, old.content);
END;

CREATE TRIGGER IF NOT EXISTS messages_au
AFTER UPDATE ON messages BEGIN
  INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.id, old.content);
  INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;
";

fn unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ---------------- wire types ----------------

#[derive(Debug, Serialize, Clone)]
pub struct ConversationSummary {
    pub id: i64,
    pub title: String,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Debug, Serialize, Clone)]
pub struct StoredMessage {
    pub id: i64,
    pub role: String,
    pub content: String,
    pub created_at: i64,
}

#[derive(Debug, Serialize, Clone)]
pub struct Conversation {
    #[serde(flatten)]
    pub summary: ConversationSummary,
    pub messages: Vec<StoredMessage>,
}

#[derive(Debug, Deserialize)]
pub struct CreateConversationRequest {
    #[serde(default = "default_title")]
    pub title: String,
}

fn default_title() -> String {
    "New chat".to_string()
}

#[derive(Debug, Deserialize)]
pub struct AppendMessageRequest {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct RenameRequest {
    pub title: String,
}

/// A single FTS hit: the message that matched, its conversation
/// context, the FTS5 `snippet()` excerpt (HTML `<mark>` tags around
/// matched terms, `…` ellipsis at truncation points), and the
/// `bm25()` relevance score (lower = more relevant).
#[derive(Debug, Serialize, Clone)]
pub struct SearchHit {
    pub message_id: i64,
    pub conversation_id: i64,
    pub conversation_title: String,
    pub role: String,
    pub created_at: i64,
    pub snippet: String,
    pub score: f64,
}

#[derive(Debug, Deserialize)]
pub struct SearchQuery {
    pub q: String,
    #[serde(default = "default_search_limit")]
    pub limit: u32,
}

fn default_search_limit() -> u32 {
    25
}

// ---------------- HTTP handlers ----------------

/// Re-export type the router uses to attach the optional store.
pub type SharedHistory = Arc<HistoryStore>;

fn store(state: &AppState) -> std::result::Result<SharedHistory, axum::response::Response> {
    state
        .history
        .clone()
        .ok_or_else(|| {
            (
                StatusCode::NOT_IMPLEMENTED,
                Json(serde_json::json!({
                    "error": "history feature not enabled",
                    "detail": "rebuild the server with `--features history` to enable \
                               conversation persistence (sqlite-backed).",
                })),
            )
                .into_response()
        })
}

pub async fn list_handler(State(state): State<AppState>) -> axum::response::Response {
    let s = match store(&state) {
        Ok(s) => s,
        Err(resp) => return resp,
    };
    match s.list_conversations() {
        Ok(rows) => Json(serde_json::json!({ "conversations": rows })).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

pub async fn get_handler(
    State(state): State<AppState>,
    Path(id): Path<i64>,
) -> axum::response::Response {
    let s = match store(&state) {
        Ok(s) => s,
        Err(resp) => return resp,
    };
    match s.get_conversation(id) {
        Ok(Some(c)) => Json(c).into_response(),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "conversation not found", "id": id })),
        )
            .into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

pub async fn create_handler(
    State(state): State<AppState>,
    Json(req): Json<CreateConversationRequest>,
) -> axum::response::Response {
    let s = match store(&state) {
        Ok(s) => s,
        Err(resp) => return resp,
    };
    match s.create_conversation(&req.title) {
        Ok(id) => Json(serde_json::json!({ "id": id, "title": req.title })).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

pub async fn append_handler(
    State(state): State<AppState>,
    Path(id): Path<i64>,
    Json(req): Json<AppendMessageRequest>,
) -> axum::response::Response {
    let s = match store(&state) {
        Ok(s) => s,
        Err(resp) => return resp,
    };
    if !matches!(req.role.as_str(), "user" | "assistant" | "system" | "tool") {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "invalid role",
                "expected": ["user", "assistant", "system", "tool"],
                "got": req.role,
            })),
        )
            .into_response();
    }
    match s.append_message(id, &req.role, &req.content) {
        Ok(msg_id) => Json(serde_json::json!({ "message_id": msg_id })).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

/// `GET /api/conversations/search?q=…&limit=25` — FTS5 search over
/// all stored message contents. Returns ranked hits with an HTML-
/// marked snippet excerpt + conversation context.
pub async fn search_handler(
    State(state): State<AppState>,
    axum::extract::Query(q): axum::extract::Query<SearchQuery>,
) -> axum::response::Response {
    let s = match store(&state) {
        Ok(s) => s,
        Err(resp) => return resp,
    };
    if q.q.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "q must be a non-empty query string",
                "detail": "FTS5 MATCH syntax: bare words AND-join, quoted phrases require \
                           the exact sequence, `*` is a prefix wildcard. Example: q=deploy*",
            })),
        )
            .into_response();
    }
    match s.search_messages(&q.q, q.limit) {
        Ok(hits) => Json(serde_json::json!({
            "results": hits,
            "query": q.q,
        }))
        .into_response(),
        Err(HistoryError::Sqlite(e)) => {
            // The only SqliteFailure that can fire during
            // `search_messages` in production is bad MATCH syntax
            // (the JOINs hit known schema with foreign-key-enforced
            // rows). Any `SqliteFailure` here is therefore a client
            // query error → 400 with the underlying SQLite message
            // surfaced so the user can fix the query. Non-SQLite
            // errors (locked, busy, I/O) keep the 500.
            match e {
                rusqlite::Error::SqliteFailure(_, _) => (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({
                        "error": "invalid FTS5 query",
                        "detail": e.to_string(),
                    })),
                )
                    .into_response(),
                other => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": "search failed",
                        "detail": other.to_string(),
                    })),
                )
                    .into_response(),
            }
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

pub async fn delete_handler(
    State(state): State<AppState>,
    Path(id): Path<i64>,
) -> axum::response::Response {
    let s = match store(&state) {
        Ok(s) => s,
        Err(resp) => return resp,
    };
    match s.delete_conversation(id) {
        Ok(true) => Json(serde_json::json!({ "deleted": id })).into_response(),
        Ok(false) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "conversation not found", "id": id })),
        )
            .into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

pub async fn rename_handler(
    State(state): State<AppState>,
    Path(id): Path<i64>,
    Json(req): Json<RenameRequest>,
) -> axum::response::Response {
    let s = match store(&state) {
        Ok(s) => s,
        Err(resp) => return resp,
    };
    match s.rename(id, &req.title) {
        Ok(true) => Json(serde_json::json!({ "renamed": id, "title": req.title }))
            .into_response(),
        Ok(false) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "conversation not found", "id": id })),
        )
            .into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

/// Query string for `/api/conversations/:id/export`. `format`
/// defaults to `"md"`; `"json"` is also accepted. Anything else
/// returns 400.
#[derive(Debug, Deserialize)]
pub struct ExportQuery {
    #[serde(default = "default_export_format")]
    pub format: String,
}

fn default_export_format() -> String {
    "md".to_string()
}

/// Render a stored conversation as portable markdown. Each message
/// becomes a `## <Role>` block followed by the raw content (with no
/// further escaping — the chat content is already free text). The
/// title goes at the top as `# <Title>`, with the `created_at` and
/// `updated_at` ISO timestamps in an HTML comment so they don't
/// render visibly but are still extractable.
fn render_markdown(c: &Conversation) -> String {
    let mut out = String::new();
    out.push_str(&format!("# {}\n\n", c.summary.title));
    out.push_str(&format!(
        "<!-- created_at={}, updated_at={} -->\n\n",
        c.summary.created_at, c.summary.updated_at
    ));
    for m in &c.messages {
        let title = match m.role.as_str() {
            "user" => "User",
            "assistant" => "Assistant",
            "system" => "System",
            "tool" => "Tool",
            other => other,
        };
        out.push_str(&format!("## {title}\n\n{}\n\n", m.content));
    }
    out
}

/// Export a conversation as `text/markdown` (default) or
/// `application/json`. `Content-Disposition: attachment` so editors
/// and browsers download a sensibly-named file rather than rendering
/// inline. The file stem is the conversation title with characters
/// outside `[A-Za-z0-9_-]` collapsed to `_`.
pub async fn export_handler(
    State(state): State<AppState>,
    Path(id): Path<i64>,
    axum::extract::Query(q): axum::extract::Query<ExportQuery>,
) -> axum::response::Response {
    let s = match store(&state) {
        Ok(s) => s,
        Err(resp) => return resp,
    };
    let conv = match s.get_conversation(id) {
        Ok(Some(c)) => c,
        Ok(None) => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({ "error": "conversation not found", "id": id })),
            )
                .into_response();
        }
        Err(e) => {
            return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response();
        }
    };
    let stem: String = conv
        .summary
        .title
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect();
    let stem = if stem.is_empty() {
        format!("conversation-{id}")
    } else {
        stem
    };
    match q.format.as_str() {
        "md" | "markdown" => {
            let body = render_markdown(&conv);
            let disposition = format!("attachment; filename=\"{stem}.md\"");
            (
                StatusCode::OK,
                [
                    ("content-type", "text/markdown; charset=utf-8"),
                    ("content-disposition", disposition.as_str()),
                ],
                body,
            )
                .into_response()
        }
        "json" => {
            let body = match serde_json::to_string_pretty(&conv) {
                Ok(s) => s,
                Err(e) => {
                    return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response();
                }
            };
            let disposition = format!("attachment; filename=\"{stem}.json\"");
            (
                StatusCode::OK,
                [
                    ("content-type", "application/json"),
                    ("content-disposition", disposition.as_str()),
                ],
                body,
            )
                .into_response()
        }
        other => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "unknown export format",
                "expected": ["md", "json"],
                "got": other,
            })),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_markdown_includes_title_and_roles() {
        // The rendered markdown should embed the title as an H1,
        // each message as an H2 with role-titled heading, and a
        // metadata comment for the timestamps. Pins the export
        // shape so downstream tools can grep for the patterns.
        let s = HistoryStore::open_in_memory().unwrap();
        let id = s.create_conversation("Hello World").unwrap();
        s.append_message(id, "user", "ping").unwrap();
        s.append_message(id, "assistant", "pong").unwrap();
        let conv = s.get_conversation(id).unwrap().unwrap();
        let md = render_markdown(&conv);
        assert!(md.starts_with("# Hello World\n\n"));
        assert!(md.contains("<!-- created_at="));
        assert!(md.contains("## User\n\nping\n\n"));
        assert!(md.contains("## Assistant\n\npong\n\n"));
    }

    #[test]
    fn create_list_get_round_trip() {
        let s = HistoryStore::open_in_memory().expect("open");
        let id = s.create_conversation("hello world").expect("create");
        assert!(id > 0);

        s.append_message(id, "user", "hi").expect("append user");
        s.append_message(id, "assistant", "hi back").expect("append asst");

        let list = s.list_conversations().expect("list");
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].id, id);
        assert_eq!(list[0].title, "hello world");

        let conv = s.get_conversation(id).expect("get").expect("present");
        assert_eq!(conv.messages.len(), 2);
        assert_eq!(conv.messages[0].role, "user");
        assert_eq!(conv.messages[0].content, "hi");
        assert_eq!(conv.messages[1].role, "assistant");
    }

    #[test]
    fn delete_cascades_to_messages() {
        // Foreign key constraint with ON DELETE CASCADE means deleting
        // a conversation must drop its messages too. Pin that so a
        // future "let me reuse this id" refactor can't silently leave
        // orphaned rows behind.
        let s = HistoryStore::open_in_memory().expect("open");
        let id = s.create_conversation("doomed").expect("create");
        s.append_message(id, "user", "you'll be back").unwrap();
        s.append_message(id, "assistant", "no I won't").unwrap();

        assert!(s.delete_conversation(id).expect("delete"));
        let conv = s.get_conversation(id).expect("get");
        assert!(conv.is_none());

        // Verify cascade: the messages table has no rows for that id.
        let conn = s.conn.lock().unwrap();
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM messages WHERE conversation_id = ?1",
                params![id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(count, 0, "messages must cascade-delete");
    }

    #[test]
    fn update_bumps_updated_at_so_newest_first_works() {
        let s = HistoryStore::open_in_memory().expect("open");
        let a = s.create_conversation("first").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1100));
        let b = s.create_conversation("second").unwrap();

        // List order: b should be newer, so first in the listing.
        let list = s.list_conversations().unwrap();
        assert_eq!(list[0].id, b);
        assert_eq!(list[1].id, a);

        // Now append to A — it should hop to the front.
        std::thread::sleep(std::time::Duration::from_millis(1100));
        s.append_message(a, "user", "wait").unwrap();
        let list = s.list_conversations().unwrap();
        assert_eq!(
            list[0].id, a,
            "appending to A should bump it to the front"
        );
    }

    #[test]
    fn rename_changes_title_and_bumps_updated_at() {
        let s = HistoryStore::open_in_memory().expect("open");
        let id = s.create_conversation("original").unwrap();
        let renamed = s.rename(id, "renamed!").unwrap();
        assert!(renamed);
        let conv = s.get_conversation(id).unwrap().unwrap();
        assert_eq!(conv.summary.title, "renamed!");
    }

    #[test]
    fn delete_unknown_returns_false() {
        let s = HistoryStore::open_in_memory().expect("open");
        let r = s.delete_conversation(99999).unwrap();
        assert!(!r);
    }

    #[test]
    fn rename_unknown_returns_false() {
        let s = HistoryStore::open_in_memory().expect("open");
        let r = s.rename(99999, "anything").unwrap();
        assert!(!r);
    }
}
