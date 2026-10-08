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

/// List ALL files in a repo's `main` revision (path + size) via the tree API.
/// The superset [`hf_gguf_files`] filters down to GGUFs; `pull` uses this to
/// discover the `.kvbias.gguf` / mmproj COMPANION files that live next to a
/// chosen GGUF so a pulled model lands complete on disk (see
/// [`companion_sidecars`]).
pub async fn hf_repo_files(repo_id: &str) -> Result<Vec<HfFile>> {
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
        .filter(|e| e.kind == "file")
        .map(|e| HfFile {
            rfilename: e.path,
            size: e.size,
        })
        .collect();
    out.sort_by(|a, b| a.rfilename.cmp(&b.rfilename));
    Ok(out)
}

/// List the `.gguf` files (with sizes) in a repo's `main` revision via the
/// tree API. Used to turn a searched repo into concrete pullable files.
pub async fn hf_gguf_files(repo_id: &str) -> Result<Vec<HfFile>> {
    let mut out: Vec<HfFile> = hf_repo_files(repo_id)
        .await?
        .into_iter()
        .filter(|f| f.rfilename.to_ascii_lowercase().ends_with(".gguf"))
        .collect();
    out.sort_by(|a, b| a.rfilename.cmp(&b.rfilename));
    Ok(out)
}

/// Given the main GGUF filename just pulled and the repo's full file list,
/// return the repo-relative paths of COMPANION files worth fetching alongside
/// it so the model is actually complete on disk — the single GGUF alone
/// silently loses these:
///   - `<stem>.kvbias.gguf` — KV-bias calibration; its presence NEXT TO the
///     GGUF is exactly what lets `coherence_safe_kv_dtype` trust a quantized
///     KV cache for this model (without it, quant KV is coerced to f32).
///   - `*mmproj*.gguf`      — the vision projector for a multimodal model
///     (without it a pulled vision model can't see).
/// Case-insensitive; the main file itself is never returned. A plain
/// single-GGUF repo yields an empty list. (imatrix / tokenizer / config are
/// deliberately excluded — imatrix is a quantize-time input, and a GGUF already
/// embeds its tokenizer + config.)
pub fn companion_sidecars(main_filename: &str, repo_files: &[HfFile]) -> Vec<String> {
    let main_lc = main_filename.to_ascii_lowercase();
    // `<stem>.kvbias.gguf` sibling — keep the full repo path so a nested layout
    // (`sub/model.gguf` → `sub/model.kvbias.gguf`) still matches, mirroring the
    // engine's `model_path.with_extension("kvbias.gguf")` lookup.
    let stem = main_lc.strip_suffix(".gguf").unwrap_or(main_lc.as_str());
    let kvbias = format!("{stem}.kvbias.gguf");
    let mut out = Vec::new();
    for f in repo_files {
        let p = f.rfilename.to_ascii_lowercase();
        if p == main_lc {
            continue;
        }
        let is_kvbias = p == kvbias;
        let is_mmproj = p.ends_with(".gguf") && p.contains("mmproj");
        if is_kvbias || is_mmproj {
            out.push(f.rfilename.clone());
        }
    }
    out
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

/// Resolve a user-supplied model spec to a concrete on-disk path — a `.gguf`
/// file or an MLX model directory — trying every form the CLI, the HTTP
/// server, and the GUI accept so all three behave identically. Pure lookup;
/// never downloads.
///
///   1. an existing path (absolute OR relative to the CWD) → used verbatim;
///   2. a hub ref `owner/repo:file.gguf` → `<cache>/owner__repo/file.gguf`,
///      when that file is actually present;
///   3. a bare cached id / file stem (e.g.
///      `qwen2.5-coder-7b-instruct-q4_k_m`) → a root-level `<id>.gguf`, else
///      any hub-layout GGUF whose file stem equals `spec`, else an MLX model
///      directory whose name equals `spec`.
///
/// Returns `None` when nothing matches. This is the single resolver the
/// `serve` / `chat` / `model` CLI paths, the `/v1/models/load` handler, and
/// the GUI all funnel through, so "the exact ref you pulled", "the id the
/// model list shows", and "a path on disk" are interchangeable everywhere.
pub fn resolve_model_spec(spec: &str, cache_dir: &Path) -> Option<PathBuf> {
    // 1. An existing path wins outright (absolute or relative to the CWD).
    let as_path = PathBuf::from(spec);
    if as_path.exists() {
        return Some(as_path);
    }
    // 2. A hub ref maps to its deterministic cache location — but only count
    //    it when the file is actually there (a not-yet-pulled ref falls
    //    through so the caller can apply its own "would-be path" logic).
    if let Ok(hub_ref) = HubRef::parse(spec) {
        let path = hub_ref.local_path(cache_dir);
        if path.exists() {
            return Some(path);
        }
    }
    // 3a. A locally-dropped GGUF at the cache root (`<cache>/<id>.gguf`) —
    //     `list_cached` only walks the depth-2 hub layout, so check this first.
    let direct = cache_dir.join(format!("{spec}.gguf"));
    if direct.is_file() {
        return Some(direct);
    }
    // 3b. Any hub-layout GGUF whose file stem matches the spec (the id the
    //     model list / `/api/tags` shows).
    if let Ok(paths) = list_cached(cache_dir) {
        if let Some(hit) = paths.into_iter().find(|p| {
            p.file_stem()
                .and_then(|s| s.to_str())
                .map(|s| s == spec)
                .unwrap_or(false)
        }) {
            return Some(hit);
        }
    }
    // 3c. An MLX model *directory* whose name matches (mlx-lm / mlx-community
    //     checkpoints load by directory, not by a single file).
    if let Ok(models) = list_cached_models(cache_dir) {
        if let Some(m) = models.into_iter().find(|m| m.name == spec) {
            return Some(m.path);
        }
    }
    None
}

/// One resolvable model on disk: either a GGUF **file** or an MLX model
/// **directory** (mlx-lm / mlx-community layout). Both carry a `name` the
/// caller can match a short load request against — a GGUF's file stem, or an
/// MLX dir's file name (a directory has no extension to strip). This is the
/// shape the on-demand load path + `/api/tags` listing resolve against so a
/// user can `Load` an MLX model by name the same way they load a cached GGUF.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachedModel {
    /// Absolute path — a `.gguf` file, or an MLX model directory (the thing
    /// `CpuEngine::load_auto` is handed).
    pub path: PathBuf,
    /// Resolvable identity: the GGUF's file stem, or the MLX dir's file name.
    pub name: String,
    /// `true` when `path` is an MLX model directory (load via `load_auto`);
    /// `false` for a plain `.gguf` file (load via the GGUF-tuned path).
    pub is_dir: bool,
}

/// Enumerate cached models: GGUF **files** (as [`list_cached`]) *plus* MLX
/// model **directories**. An MLX directory is an mlx-lm / mlx-community
/// checkpoint — a folder holding a `config.json` with a `quantization` block
/// and at least one `*.safetensors` shard. Such a dir can sit either directly
/// under the cache (`<cache>/<repo>/…`, a hub-mirrored flat download) or one
/// level deeper (`<cache>/<owner>__<repo>/<subdir>/…`), so both depth 1 and
/// depth 2 are probed.
///
/// GGUF entries keep [`list_cached`]'s file-stem identity (byte-identical
/// discovery); only `.gguf` files surface as standalone models, so an MLX
/// shard / config sitting at a listable depth never masquerades as its own
/// GGUF row. MLX entries take the directory's `file_name` as their name.
pub fn list_cached_models(cache_dir: &Path) -> Result<Vec<CachedModel>> {
    let mut out = Vec::new();
    if !cache_dir.exists() {
        return Ok(out);
    }

    // 1) MLX model directories (depth 1, else its immediate children at
    //    depth 2). A depth-1 match is reported as-is and NOT descended into,
    //    so a model can't be double-counted.
    let mut mlx_dirs: Vec<PathBuf> = Vec::new();
    for d1 in read_subdirs(cache_dir) {
        if is_mlx_dir_shallow(&d1) {
            mlx_dirs.push(d1);
        } else {
            for d2 in read_subdirs(&d1) {
                if is_mlx_dir_shallow(&d2) {
                    mlx_dirs.push(d2);
                }
            }
        }
    }
    for dir in &mlx_dirs {
        if let Some(name) = dir.file_name().and_then(|s| s.to_str()) {
            out.push(CachedModel {
                path: dir.clone(),
                name: name.to_string(),
                is_dir: true,
            });
        }
    }

    // 2) GGUF files (the hub `owner__repo/file.gguf` layout). Filter to
    //    `.gguf` and skip any file that lives inside a detected MLX directory
    //    so an MLX shard / config.json never shows up as a standalone model.
    for p in list_cached(cache_dir)? {
        let is_gguf = p
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false);
        if !is_gguf {
            continue;
        }
        if mlx_dirs.iter().any(|d| p.starts_with(d)) {
            continue;
        }
        if let Some(name) = p.file_stem().and_then(|s| s.to_str()) {
            out.push(CachedModel {
                path: p.clone(),
                name: name.to_string(),
                is_dir: false,
            });
        }
    }
    Ok(out)
}

/// Immediate subdirectories of `dir` (best-effort; an unreadable `dir`
/// yields an empty list). Non-recursive.
fn read_subdirs(dir: &Path) -> Vec<PathBuf> {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.is_dir())
                .collect()
        })
        .unwrap_or_default()
}

/// Cheap MLX-directory probe used by discovery (no dependency on
/// rustllama-safetensors, so the hub crate stays toolchain-free): does `dir`
/// hold a `config.json` carrying a `quantization` block AND at least one
/// `*.safetensors` shard? That's the mlx-lm / mlx-community marker. The
/// engine's `load_auto` re-checks with the stricter `.scales`/`.biases`
/// discriminator before committing to the MLX load path, so this only needs
/// to be a fast, permissive filter for the listing. Any IO / parse error is
/// swallowed as `false`.
fn is_mlx_dir_shallow(dir: &Path) -> bool {
    if !dir.is_dir() {
        return false;
    }
    let Ok(text) = std::fs::read_to_string(dir.join("config.json")) else {
        return false;
    };
    let has_quant = serde_json::from_str::<serde_json::Value>(&text)
        .ok()
        .and_then(|v| v.get("quantization").map(|q| q.is_object()))
        .unwrap_or(false);
    if !has_quant {
        return false;
    }
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok()).any(|e| {
                e.path()
                    .extension()
                    .and_then(|x| x.to_str())
                    .map(|x| x.eq_ignore_ascii_case("safetensors"))
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false)
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
    fn companion_sidecars_picks_kvbias_and_mmproj_only() {
        let f = |p: &str| HfFile {
            rfilename: p.to_string(),
            size: 1,
        };
        let files = vec![
            f("Model-Q4_K_M.gguf"),
            f("Model-Q4_K_M.kvbias.gguf"), // KV calibration for THIS variant
            f("mmproj-Model-Q8_0.gguf"),   // vision projector
            f("Model-Q8_0.gguf"),          // a DIFFERENT quant variant
            f("README.md"),
            f("model.imatrix"), // quantize-time input, not an inference companion
        ];
        let comps = companion_sidecars("Model-Q4_K_M.gguf", &files);
        assert!(comps.contains(&"Model-Q4_K_M.kvbias.gguf".to_string()));
        assert!(comps.contains(&"mmproj-Model-Q8_0.gguf".to_string()));
        // Must NOT pull a sibling quant variant, the main file, the card, or the
        // imatrix.
        assert!(!comps.contains(&"Model-Q8_0.gguf".to_string()));
        assert!(!comps.contains(&"Model-Q4_K_M.gguf".to_string()));
        assert!(!comps.contains(&"README.md".to_string()));
        assert!(!comps.contains(&"model.imatrix".to_string()));
        // A plain single-GGUF repo yields no companions.
        assert!(companion_sidecars("solo.gguf", &[f("solo.gguf")]).is_empty());
        // Nested layout: the kvbias sibling shares the GGUF's directory.
        let nested = vec![f("sub/m.gguf"), f("sub/m.kvbias.gguf")];
        assert_eq!(
            companion_sidecars("sub/m.gguf", &nested),
            vec!["sub/m.kvbias.gguf".to_string()]
        );
    }

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
    fn list_cached_models_surfaces_gguf_files_and_mlx_dirs() {
        // Build a throwaway cache tree:
        //   <root>/owner__repo/model-q4_k_m.gguf        → GGUF file
        //   <root>/Qwen2.5-0.5B-Instruct-4bit/          → MLX dir (depth 1)
        //       config.json (quantization block) + model.safetensors
        //   <root>/nested/DeepSomething-3bit/           → MLX dir (depth 2)
        //       config.json + weights.safetensors
        let root = std::env::temp_dir().join(format!(
            "rustllama_hub_list_cached_models_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("owner__repo")).unwrap();
        std::fs::write(root.join("owner__repo/model-q4_k_m.gguf"), b"GGUF\0\0\0\0").unwrap();

        let mlx1 = root.join("Qwen2.5-0.5B-Instruct-4bit");
        std::fs::create_dir_all(&mlx1).unwrap();
        std::fs::write(
            mlx1.join("config.json"),
            br#"{"quantization":{"group_size":64,"bits":4}}"#,
        )
        .unwrap();
        std::fs::write(mlx1.join("model.safetensors"), b"\x00").unwrap();
        std::fs::write(mlx1.join("tokenizer.json"), b"{}").unwrap();

        let mlx2 = root.join("nested").join("DeepSomething-3bit");
        std::fs::create_dir_all(&mlx2).unwrap();
        std::fs::write(
            mlx2.join("config.json"),
            br#"{"quantization":{"group_size":32,"bits":3}}"#,
        )
        .unwrap();
        std::fs::write(mlx2.join("weights.safetensors"), b"\x00").unwrap();

        // A plain fp16 HF dir (no `quantization` block) must NOT surface.
        let plain = root.join("plain-hf-dir");
        std::fs::create_dir_all(&plain).unwrap();
        std::fs::write(plain.join("config.json"), br#"{"model_type":"llama"}"#).unwrap();
        std::fs::write(plain.join("model.safetensors"), b"\x00").unwrap();

        let models = list_cached_models(&root).unwrap();

        // GGUF file surfaces by file stem, not a dir.
        let gguf = models
            .iter()
            .find(|m| m.name == "model-q4_k_m")
            .expect("gguf model-q4_k_m should surface");
        assert!(!gguf.is_dir);
        assert!(gguf.path.ends_with("owner__repo/model-q4_k_m.gguf"));

        // MLX dir at depth 1 surfaces by dir name.
        let d1 = models
            .iter()
            .find(|m| m.name == "Qwen2.5-0.5B-Instruct-4bit")
            .expect("depth-1 MLX dir should surface");
        assert!(d1.is_dir);
        assert_eq!(d1.path, mlx1);

        // MLX dir at depth 2 surfaces too.
        let d2 = models
            .iter()
            .find(|m| m.name == "DeepSomething-3bit")
            .expect("depth-2 MLX dir should surface");
        assert!(d2.is_dir);
        assert_eq!(d2.path, mlx2);

        // The non-quantized HF dir must be absent, and its `model.safetensors`
        // must not masquerade as a GGUF row.
        assert!(
            !models.iter().any(|m| m.name == "plain-hf-dir"),
            "a dir without a quantization block must not surface as MLX"
        );
        assert!(
            !models.iter().any(|m| m.name == "model" || m.name == "config"),
            "safetensors / config.json must never surface as standalone models"
        );

        let _ = std::fs::remove_dir_all(&root);
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
