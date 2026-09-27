//! HTTP-side `image_url.url` -> raw bytes decoder.
//!
//! The OpenAI multimodal wire shape carries an image inside a content
//! block of the form `{"type": "image_url", "image_url": {"url": ...}}`.
//! The url can be either:
//!
//!   1. A `data:image/<png|jpeg>;base64,<payload>` URI (the shape
//!      everything emits in practice: ChatGPT web client, Cursor,
//!      Continue, the OpenAI SDK image upload helper, the Anthropic
//!      "base64-encoded source" shape after the adapter rewrites it).
//!   2. A remote URL (`https://...`).
//!   3. A local file URL (`file://...`).
//!
//! v1 only decodes case 1. Cases 2 and 3 are rejected with a typed
//! error so the chat handler can return a 400 with a clear message
//! instead of silently dropping the image:
//!
//!   * `https://` fetch from the server process is a clear SSRF
//!     vector (an attacker could ask the server to fetch internal
//!     metadata endpoints like `http://169.254.169.254/...`); we
//!     don't enable it until there's a configurable allow-list +
//!     egress proxy story, which is itself a roadmap item.
//!   * `file://` lets an attacker exfiltrate arbitrary files
//!     readable by the server process via the image preprocessor
//!     error messages (path-traversal style); same gate.
//!
//! The decoder only handles the wire-protocol parse + base64 decode;
//! the resulting bytes are handed to `rustllama_models::vision_arch
//! ::preprocess_image`, which is what enforces image-format validity,
//! decoded dimensions, and channel count.

use base64::Engine as _;

/// All the ways an `image_url.url` can be malformed or forbidden in
/// v1. Each variant maps cleanly to an HTTP 400 with a human-readable
/// reason; the variant itself is the discriminator for tests and the
/// error_param field in the OpenAI-style error envelope.
#[derive(Debug, thiserror::Error)]
pub enum ImageUrlError {
    /// The url is empty (clients sometimes send `{"image_url": {"url": ""}}`).
    #[error("image_url.url is empty")]
    Empty,
    /// A `data:` URI was given but the format is wrong (missing the
    /// `;base64,` segment, wrong media type, etc.).
    #[error("malformed data: URI: {0}")]
    MalformedDataUri(&'static str),
    /// The media type inside the data URI is not one of the supported
    /// image formats. v1 supports `image/png` and `image/jpeg`; the
    /// preprocessor (`image` crate) decodes both. JPEG variants like
    /// `image/jpg` (no `e`) are also accepted because clients are
    /// inconsistent.
    #[error("unsupported media type: {0}; v1 supports image/png and image/jpeg")]
    UnsupportedMediaType(String),
    /// Base64 payload failed to decode.
    #[error("base64 decode failed: {0}")]
    Base64(#[from] base64::DecodeError),
    /// A remote URL was provided. v1 doesn't fetch (SSRF / egress
    /// policy concerns); clients must inline the image as a data URI.
    #[error("remote image URLs ({0}://...) are not supported in v1; inline the image as a data: URI")]
    RemoteUrlNotSupported(String),
    /// A `file://` URL was provided.
    #[error("file:// URLs are not supported; inline the image as a data: URI")]
    FileUrlNotSupported,
}

/// Decode an `image_url.url` field to raw image bytes ready for the
/// image preprocessor. Returns the decoded payload on success; the
/// caller is expected to hand those bytes to `preprocess_image` in
/// `rustllama-models`, which is what actually validates the image
/// format and shape.
///
/// v1 only accepts `data:image/<png|jpeg|jpg>;base64,<payload>`.
pub fn decode_image_url(url: &str) -> Result<Vec<u8>, ImageUrlError> {
    if url.is_empty() {
        return Err(ImageUrlError::Empty);
    }
    if let Some(rest) = url.strip_prefix("data:") {
        return decode_data_uri(rest);
    }
    if let Some(scheme_end) = url.find("://") {
        let scheme = &url[..scheme_end];
        return match scheme {
            "http" | "https" => Err(ImageUrlError::RemoteUrlNotSupported(scheme.to_string())),
            "file" => Err(ImageUrlError::FileUrlNotSupported),
            other => Err(ImageUrlError::RemoteUrlNotSupported(other.to_string())),
        };
    }
    // Neither a data: URI nor a recognized scheme. Treat as malformed
    // rather than guessing the client meant a relative HTTPS URL.
    Err(ImageUrlError::MalformedDataUri(
        "expected data:image/<png|jpeg>;base64,<payload>",
    ))
}

/// Parse the body of a `data:` URI (everything after `data:`). Splits
/// on the first `,` to separate the metadata segment from the payload,
/// validates that the metadata declares `image/<png|jpeg>` followed by
/// `;base64`, then base64-decodes the payload.
fn decode_data_uri(rest: &str) -> Result<Vec<u8>, ImageUrlError> {
    let comma = rest
        .find(',')
        .ok_or(ImageUrlError::MalformedDataUri("missing comma separator"))?;
    let metadata = &rest[..comma];
    let payload = &rest[comma + 1..];

    // Metadata shape: <media-type>[;param=value;...];base64
    // The terminal token MUST be `;base64` — v1 doesn't accept
    // un-encoded (URL-encoded) data: payloads because every real
    // client uses base64 for image bytes.
    let (media_type, rest_params) = match metadata.find(';') {
        Some(idx) => (&metadata[..idx], &metadata[idx..]),
        None => {
            return Err(ImageUrlError::MalformedDataUri(
                "missing ;base64 terminator",
            ))
        }
    };
    if !rest_params.split(';').any(|p| p.eq_ignore_ascii_case("base64")) {
        return Err(ImageUrlError::MalformedDataUri(
            "missing ;base64 terminator",
        ));
    }

    let media_type_lower = media_type.to_ascii_lowercase();
    match media_type_lower.as_str() {
        "image/png" | "image/jpeg" | "image/jpg" => {}
        other => return Err(ImageUrlError::UnsupportedMediaType(other.to_string())),
    }

    // Strip ASCII whitespace from the payload before decoding.
    // Some clients pretty-print base64 with newlines every 76 chars
    // (the RFC 2045 default); the standard alphabet engine does not
    // tolerate whitespace by default.
    let cleaned: Vec<u8> = payload
        .bytes()
        .filter(|b| !b.is_ascii_whitespace())
        .collect();
    let bytes = base64::engine::general_purpose::STANDARD.decode(cleaned)?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `data:image/png;base64,<...>` URI from raw bytes.
    fn make_png_data_uri(bytes: &[u8]) -> String {
        let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
        format!("data:image/png;base64,{encoded}")
    }

    #[test]
    fn decodes_png_data_uri_roundtrip() {
        let raw = b"\x89PNG\r\n\x1a\n_fake_png_header_only";
        let url = make_png_data_uri(raw);
        let got = decode_image_url(&url).expect("decode should succeed");
        assert_eq!(got, raw);
    }

    #[test]
    fn decodes_jpeg_data_uri_roundtrip() {
        let raw = b"\xff\xd8\xff_fake_jpeg_marker_only";
        let encoded = base64::engine::general_purpose::STANDARD.encode(raw);
        let url = format!("data:image/jpeg;base64,{encoded}");
        let got = decode_image_url(&url).expect("decode should succeed");
        assert_eq!(got, raw);
    }

    #[test]
    fn accepts_image_jpg_alias() {
        // Some clients emit `image/jpg` (no `e`). The PNG/JPEG
        // decoder downstream handles whichever bytes we hand it,
        // so accept the alias rather than rejecting the request.
        let raw = b"\xff\xd8\xff_jpg_alias";
        let encoded = base64::engine::general_purpose::STANDARD.encode(raw);
        let url = format!("data:image/jpg;base64,{encoded}");
        let got = decode_image_url(&url).expect("decode should succeed");
        assert_eq!(got, raw);
    }

    #[test]
    fn tolerates_pretty_printed_base64_whitespace() {
        // RFC 2045 base64 inserts a newline every 76 chars. The
        // `STANDARD` engine alone rejects whitespace; we pre-strip it.
        let raw: Vec<u8> = (0..200u8).collect();
        let mut encoded = base64::engine::general_purpose::STANDARD.encode(&raw);
        // Insert newlines every 64 chars to simulate a chunked encoder.
        let mut chunked = String::new();
        while encoded.len() > 64 {
            chunked.push_str(&encoded[..64]);
            chunked.push('\n');
            encoded = encoded[64..].to_string();
        }
        chunked.push_str(&encoded);
        let url = format!("data:image/png;base64,{chunked}");
        let got = decode_image_url(&url).expect("whitespace should be tolerated");
        assert_eq!(got, raw);
    }

    #[test]
    fn rejects_empty_url() {
        assert!(matches!(decode_image_url(""), Err(ImageUrlError::Empty)));
    }

    #[test]
    fn rejects_https_url_with_ssrf_message() {
        let err = decode_image_url("https://example.com/x.png").unwrap_err();
        let msg = err.to_string();
        assert!(matches!(err, ImageUrlError::RemoteUrlNotSupported(ref s) if s == "https"));
        assert!(msg.contains("data: URI"), "msg = {msg}");
    }

    #[test]
    fn rejects_http_url() {
        let err = decode_image_url("http://example.com/x.png").unwrap_err();
        assert!(matches!(err, ImageUrlError::RemoteUrlNotSupported(ref s) if s == "http"));
    }

    #[test]
    fn rejects_file_url() {
        let err = decode_image_url("file:///etc/passwd").unwrap_err();
        assert!(matches!(err, ImageUrlError::FileUrlNotSupported));
    }

    #[test]
    fn rejects_unknown_scheme() {
        // Any other scheme (e.g. ftp://) is bucketed into the same
        // "remote URLs not supported" message; clients should not
        // expect us to invent fetch behavior.
        let err = decode_image_url("ftp://example.com/x.png").unwrap_err();
        assert!(matches!(err, ImageUrlError::RemoteUrlNotSupported(ref s) if s == "ftp"));
    }

    #[test]
    fn rejects_data_uri_without_comma() {
        let err = decode_image_url("data:image/png;base64").unwrap_err();
        assert!(matches!(err, ImageUrlError::MalformedDataUri(s) if s.contains("comma")));
    }

    #[test]
    fn rejects_data_uri_without_base64_marker() {
        // `data:image/png,<raw>` is technically RFC 2397 but no client
        // emits images this way. Treat as malformed rather than
        // implementing a URL-decode path that would never run.
        let err = decode_image_url("data:image/png,abc").unwrap_err();
        assert!(matches!(err, ImageUrlError::MalformedDataUri(s) if s.contains("base64")));
    }

    #[test]
    fn rejects_unsupported_media_type() {
        // image/gif and image/webp are not in the v1 codec list; the
        // workspace `image` dep is built with only `png` + `jpeg`.
        let encoded = base64::engine::general_purpose::STANDARD.encode(b"data");
        let url = format!("data:image/gif;base64,{encoded}");
        let err = decode_image_url(&url).unwrap_err();
        assert!(matches!(err, ImageUrlError::UnsupportedMediaType(ref s) if s == "image/gif"));
    }

    #[test]
    fn rejects_non_image_media_type() {
        let encoded = base64::engine::general_purpose::STANDARD.encode(b"data");
        let url = format!("data:application/pdf;base64,{encoded}");
        let err = decode_image_url(&url).unwrap_err();
        assert!(matches!(err, ImageUrlError::UnsupportedMediaType(ref s) if s == "application/pdf"));
    }

    #[test]
    fn rejects_malformed_base64_payload() {
        // `!@#$` is not in the base64 alphabet; the decoder must
        // surface the underlying base64 error.
        let url = "data:image/png;base64,!@#$";
        let err = decode_image_url(url).unwrap_err();
        assert!(matches!(err, ImageUrlError::Base64(_)));
    }

    #[test]
    fn rejects_relative_path() {
        // No scheme + no `data:` prefix. Treat as malformed rather
        // than guessing the client meant a relative HTTPS URL — we
        // don't fetch even with the scheme; we definitely don't
        // guess at one.
        let err = decode_image_url("/path/to/image.png").unwrap_err();
        assert!(matches!(err, ImageUrlError::MalformedDataUri(_)));
    }

    #[test]
    fn media_type_match_is_case_insensitive() {
        // RFC 2397 declares media types case-insensitive (per RFC 2045);
        // some clients send `Image/PNG`. Normalize before matching.
        let raw = b"\x89PNG\r\n\x1a\n_caseinsensitive";
        let encoded = base64::engine::general_purpose::STANDARD.encode(raw);
        let url = format!("data:Image/PNG;base64,{encoded}");
        let got = decode_image_url(&url).expect("case-insensitive match should succeed");
        assert_eq!(got, raw);
    }
}
