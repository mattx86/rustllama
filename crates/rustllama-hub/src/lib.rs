//! HuggingFace model puller + model-card metadata.

use std::path::{Path, PathBuf};

use indicatif::ProgressBar;

pub mod model_card;
pub use model_card::ModelCard;

#[derive(Debug, thiserror::Error)]
pub enum HubError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid hub reference: {0:?}; expected `owner/repo:filename`")]
    InvalidRef(String),
    #[error("hf-hub: {0}")]
    HfHub(String),
    #[error("http: {0}")]
    Http(String),
}

/// One HuggingFace model repo from a search (`GET /api/models`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HfModel {
    pub id: String,
    #[serde(default)]
    pub downloads: u64,
    #[serde(default)]
    pub likes: u64,
}

/// One downloadable file inside a repo (`.../tree/main`).
#[derive(Debug, Clone, serde::Serialize)]
pub struct HfFile {
    /// File path within the repo (e.g. `model-q4_k_m.gguf`).
    pub rfilename: String,
    /// Size in bytes when the API reports it, else 0.
    pub size: u64,
}

/// Search HuggingFace for GGUF-carrying model repos, most-downloaded first.
/// Returns up to `limit` repos. Network call to the public HF API (no auth
/// needed for public repos).
pub async fn hf_search(query: &str, limit: u32) -> Result<Vec<HfModel>> {
    let q = query.trim();
    if q.is_empty() {
        return Ok(Vec::new());
    }
    let url = format!(
        "https://huggingface.co/api/models?search={}&filter=gguf&sort=downloads&direction=-1&limit={}",
        urlencode(q),
        limit
    );
    let body = http_get_text(&url).await?;
    let models: Vec<HfModel> =
        serde_json::from_str(&body).map_err(|e| HubError::Http(format!("parse search: {e}")))?;
    Ok(models)
}

/// List the `.gguf` files (with sizes) in a repo's `main` revision via the
/// tree API. Used to turn a searched repo into concrete pullable files.
pub async fn hf_gguf_files(repo_id: &str) -> Result<Vec<HfFile>> {
    let repo = repo_id.trim();
    if repo.is_empty() {
        return Ok(Vec::new());
    }
    let url = format!("https://huggingface.co/api/models/{repo}/tree/main?recursive=true");
    let body = http_get_text(&url).await?;
    #[derive(serde::Deserialize)]
    struct TreeEntry {
        #[serde(rename = "type")]
        kind: String,
        path: String,
        #[serde(default)]
        size: u64,
    }
    let entries: Vec<TreeEntry> =
        serde_json::from_str(&body).map_err(|e| HubError::Http(format!("parse tree: {e}")))?;
    let mut out: Vec<HfFile> = entries
        .into_iter()
        .filter(|e| e.kind == "file" && e.path.to_ascii_lowercase().ends_with(".gguf"))
        .map(|e| HfFile {
            rfilename: e.path,
            size: e.size,
        })
        .collect();
    out.sort_by(|a, b| a.rfilename.cmp(&b.rfilename));
    Ok(out)
}

async fn http_get_text(url: &str) -> Result<String> {
    let client = reqwest::Client::builder()
        .user_agent("rustllama/0.0")
        .build()
        .map_err(|e| HubError::Http(e.to_string()))?;
    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| HubError::Http(e.to_string()))?;
    if !resp.status().is_success() {
        return Err(HubError::Http(format!("HF API {} for {url}", resp.status())));
    }
    resp.text().await.map_err(|e| HubError::Http(e.to_string()))
}

/// Minimal percent-encoding for a query string value (space + the handful
/// of chars HF's search cares about). Avoids pulling in a urlencoding dep.
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

pub type Result<T> = std::result::Result<T, HubError>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct HubRef {
    pub owner: String,
    pub repo: String,
    pub filename: String,
}

impl HubRef {
    pub fn parse(s: &str) -> Result<Self> {
        let (repo_part, filename) = s
            .rsplit_once(':')
            .ok_or_else(|| HubError::InvalidRef(s.to_string()))?;
        let (owner, repo) = repo_part
            .split_once('/')
            .ok_or_else(|| HubError::InvalidRef(s.to_string()))?;
        if owner.is_empty() || repo.is_empty() || filename.is_empty() {
            return Err(HubError::InvalidRef(s.to_string()));
        }
        Ok(Self {
            owner: owner.into(),
            repo: repo.into(),
            filename: filename.into(),
        })
    }

    /// `owner/repo` (no filename).
    pub fn repo_id(&self) -> String {
        format!("{}/{}", self.owner, self.repo)
    }

    /// Cache subdirectory: `<owner>__<repo>/<filename>`, sanitized so no
    /// component can escape `cache_dir`.
    ///
    /// The fields come from user / config input, and `filename` may also come
    /// from the HF tree API (which reports nested paths like `sub/dir/x.gguf`
    /// when `recursive=true`). Left unsanitized, a `..` segment — in the repo
    /// id or the filename — would let the join walk out of the cache and
    /// clobber arbitrary files. Separators are honored as directory boundaries,
    /// but any `.` / `..` / empty / NUL-bearing segment is dropped, so the
    /// result always stays under `cache_dir`.
    pub fn local_path(&self, cache_dir: &Path) -> PathBuf {
        let repo_dir = format!("{}__{}", self.owner, self.repo);
        let mut out = cache_dir.to_path_buf();
        // Repo id first (its own separators become nested dirs), then the
        // filename (which may itself be a nested path).
        for raw in [repo_dir.as_str(), self.filename.as_str()] {
            for seg in raw.split(['/', '\\']) {
                // Strip NUL bytes, then reject the traversal vectors.
                let seg = seg.replace('\0', "");
                if seg.is_empty() || seg == "." || seg == ".." {
                    continue;
                }
                out.push(seg);
            }
        }
        out
    }
}

pub fn default_cache_dir() -> Option<PathBuf> {
    Some(rustllama_runtime::paths().cache_dir.clone())
}

pub fn list_cached(cache_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    if !cache_dir.exists() {
        return Ok(out);
    }
    for entry in walkdir::WalkDir::new(cache_dir)
        .min_depth(2)
        .max_depth(2)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if entry.file_type().is_file() {
            out.push(entry.into_path());
        }
    }
    Ok(out)
}

/// Asynchronously download `<owner>/<repo>:<filename>` into
/// `cache_dir/<owner>__<repo>/<filename>` and return the local path.
/// If the file is already present at the destination, returns immediately.
///
/// `progress` controls indicatif-style download UI:
///   - `Some(ProgressBar)` → drive that bar (caller owns lifecycle)
///   - `None`              → silent
pub async fn download(
    hub_ref: &HubRef,
    cache_dir: &Path,
    progress: Option<&ProgressBar>,
) -> Result<PathBuf> {
    let dst = hub_ref.local_path(cache_dir);
    if dst.exists() {
        return Ok(dst);
    }
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let api = hf_hub::api::tokio::ApiBuilder::new()
        .with_progress(progress.is_some())
        .build()
        .map_err(|e| HubError::HfHub(e.to_string()))?;
    let repo = api.model(hub_ref.repo_id());
    let cached = repo
        .get(&hub_ref.filename)
        .await
        .map_err(|e| HubError::HfHub(e.to_string()))?;

    // hf-hub stores under its own cache dir; copy into our cache layout so
    // tools and the GUI can list models predictably. Copy to a `.part` sibling
    // then atomically rename, mirroring `download_with_progress` — an
    // interrupted copy must never leave a partial file that the `dst.exists()`
    // short-circuit above would later treat as a complete download.
    let mut tmp_os = dst.clone().into_os_string();
    tmp_os.push(".part");
    let tmp = PathBuf::from(tmp_os);
    std::fs::copy(&cached, &tmp)?;
    std::fs::rename(&tmp, &dst)?;

    if let Some(p) = progress {
        p.finish_with_message(format!("downloaded {}", hub_ref.filename));
    }

    Ok(dst)
}

/// Streaming download with a byte-progress callback. Fetches the GGUF from
/// the HF `resolve/main` URL, writing chunks to `cache_dir/<owner>__<repo>/
/// <filename>` and invoking `on_progress(downloaded, total)` as bytes
/// arrive (`total` = Content-Length, or 0 when the server doesn't send it).
/// Downloads to a `.part` sibling and atomically renames on success so a
/// partial file is never mistaken for complete. Skips if already present.
/// Public repos need no auth.
pub async fn download_with_progress(
    hub_ref: &HubRef,
    cache_dir: &Path,
    // `+ Send` so a caller can hold the callback across `.await` inside a
    // `tokio::spawn`ed task (the /api/pull handler does exactly that).
    on_progress: &mut (dyn FnMut(u64, u64) + Send),
) -> Result<PathBuf> {
    use std::io::Write;
    let dst = hub_ref.local_path(cache_dir);
    if dst.exists() {
        let sz = std::fs::metadata(&dst).map(|m| m.len()).unwrap_or(0);
        on_progress(sz, sz);
        return Ok(dst);
    }
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let url = format!(
        "https://huggingface.co/{}/resolve/main/{}",
        hub_ref.repo_id(),
        hub_ref.filename
    );
    let client = reqwest::Client::builder()
        .user_agent("rustllama/0.0")
        .build()
        .map_err(|e| HubError::Http(e.to_string()))?;
    let mut resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| HubError::Http(e.to_string()))?;
    if !resp.status().is_success() {
        return Err(HubError::Http(format!(
            "download {} for {url}",
            resp.status()
        )));
    }
    let total = resp.content_length().unwrap_or(0);
    let mut tmp_os = dst.clone().into_os_string();
    tmp_os.push(".part");
    let tmp = PathBuf::from(tmp_os);
    let mut file = std::fs::File::create(&tmp)?;
    let mut downloaded: u64 = 0;
    on_progress(0, total);
    while let Some(chunk) = resp
        .chunk()
        .await
        .map_err(|e| HubError::Http(e.to_string()))?
    {
        file.write_all(&chunk)?;
        downloaded += chunk.len() as u64;
        on_progress(downloaded, total);
    }
    file.flush()?;
    drop(file);
    std::fs::rename(&tmp, &dst)?;
    Ok(dst)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_valid_ref() {
        let r = HubRef::parse(
            "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF:qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        )
        .unwrap();
        assert_eq!(r.owner, "Qwen");
        assert_eq!(r.repo, "Qwen2.5-Coder-7B-Instruct-GGUF");
        assert!(r.filename.ends_with(".gguf"));
        assert_eq!(r.repo_id(), "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF");
    }

    #[test]
    fn parse_rejects_missing_filename() {
        assert!(HubRef::parse("a/b").is_err());
        assert!(HubRef::parse("a/b:").is_err());
        assert!(HubRef::parse(":x.gguf").is_err());
    }

    #[test]
    fn local_path_uses_owner_double_underscore_repo() {
        let r = HubRef::parse("a/b:c.gguf").unwrap();
        let p = r.local_path(Path::new("/tmp/cache"));
        assert!(p.ends_with("a__b/c.gguf"));
    }

    #[test]
    fn local_path_rejects_parent_dir_traversal() {
        let cache = Path::new("/tmp/cache");
        // `..` in the filename must not escape the cache dir.
        let r = HubRef {
            owner: "a".into(),
            repo: "b".into(),
            filename: "../../etc/passwd".into(),
        };
        let p = r.local_path(cache);
        assert!(p.starts_with(cache), "sanitized path {p:?} escaped {cache:?}");
        assert!(
            !p.components().any(|c| c.as_os_str() == ".."),
            "path {p:?} must contain no `..` component"
        );
        // Traversal via the repo id (its `/` split into segments) too.
        let r2 = HubRef {
            owner: "x".into(),
            repo: "../..".into(),
            filename: "m.gguf".into(),
        };
        let p2 = r2.local_path(cache);
        assert!(p2.starts_with(cache));
        assert!(!p2.components().any(|c| c.as_os_str() == ".."));
    }
}
