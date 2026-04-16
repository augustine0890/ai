#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import json
import os
import queue
import re
import sys
import threading
import time

from logger import log_event
from hashlib import sha256
from importlib.util import find_spec
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import ParseResult, unquote, urljoin, urlparse

try:
    from playwright.sync_api import BrowserContext, sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

HAS_BROTLI = find_spec("brotli") is not None or find_spec("brotlicffi") is not None

# ---------------------------------------------------------------------------
# Config / constants
# ---------------------------------------------------------------------------

# Module-level shorthand for structured log events.
_MOD = "dds.web_downloader"


def _log(event: str, **fields) -> None:
    """Emit one structured log event to trace.jsonl via logger.log_event."""
    log_event(_MOD, event, **fields)


# Extensions we treat as “static assets” worth downloading and rewriting.
# Used in multiple places: HTML attribute rewriting, CSS url(...) rewriting,
# JS string rewriting, and crawl-time asset detection.
ASSET_EXTENSIONS = (
    ".css",
    ".js",
    ".mjs",
    ".map",
    ".json",
    ".wasm",
    ".webmanifest",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".avif",
    ".svg",
    ".ico",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".mp4",
    ".webm",
    ".mp3",
)

# Conservative JS string rewriting:
# - JS_URL_RE: matches root-relative strings like "/assets/app.js"
# - JS_ABS_URL_RE: matches absolute or protocol-relative strings like
#   "https://cdn.example.com/app.js" or "//cdn.example.com/app.js"
#
# This is intentionally limited to common static file extensions to avoid
# rewriting API endpoints or dynamic URLs that could break functionality.
JS_URL_RE = re.compile(
    r"""["'](/[^"']+\.(?:png|jpg|jpeg|gif|svg|webp|avif|ico|css|js|mjs|map|woff|woff2|ttf|eot|json|wasm|webmanifest)(?:\?[^"']*)?)["']""",
    re.IGNORECASE,
)

JS_ABS_URL_RE = re.compile(
    r"""["']((?:https?:)?//[^"']+\.(?:png|jpg|jpeg|gif|svg|webp|avif|ico|css|js|mjs|map|woff|woff2|ttf|eot|json|wasm|webmanifest)(?:\?[^"']*)?)["']""",
    re.IGNORECASE,
)

# Default headers can help with sites that block "non-browser" clients.
_ACCEPT_ENCODING = "gzip, deflate, br" if HAS_BROTLI else "gzip, deflate"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": _ACCEPT_ENCODING,
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

# Network timeouts + streaming chunk size for binary downloads.
TIMEOUT = 15  # seconds
CHUNK_SIZE = 8192  # bytes

# Conservative margins under common OS limits (~255–260 bytes).
# These protect you from “File name too long” and odd Windows path rules.
MAX_PATH_LEN = 240
MAX_SEG_LEN = 120

# Collapse 3+ dots ("....") down to a single dot to avoid weird filenames.
_MULTI_DOTS_RE = re.compile(r"\.{3,}")

# CSS url(...) extractor. Note: this is simple (not a full CSS parser),
# but good enough for most sites.
CSS_URL_RE = re.compile(r"url\(([^)]+)\)")

# CSS @import extractor. Also simple-but-effective.
CSS_IMPORT_RE = re.compile(
    r"""@import\s+(?:url\()?['"]?([^'"\);]+)['"]?\)?\s*;""",
    re.IGNORECASE,
)

# Characters that commonly cause filesystem issues, especially on Windows.
_BAD_SEG_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1F]')

# Windows reserved filenames; writing these can fail or behave badly.
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}

RESOURCE_LINK_RELS = {
    "stylesheet",
    "icon",
    "shortcut",
    "apple-touch-icon",
    "preload",
    "modulepreload",
    "manifest",
}

# ---------------------------------------------------------------------------
# HTTP session (retry, timeouts, custom UA)
# ---------------------------------------------------------------------------

# Shared session improves performance and keeps connection pooling.
SESSION = requests.Session()

# Retry strategy for transient issues (rate limits, 5xx). Helps stability.
RETRY_STRAT = Retry(
    total=5,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "HEAD"],
)

SESSION.mount("http://", HTTPAdapter(max_retries=RETRY_STRAT))
SESSION.mount("https://", HTTPAdapter(max_retries=RETRY_STRAT))
SESSION.headers.update(DEFAULT_HEADERS)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def create_dir(path: Path) -> None:
    """Create path (and parents) if it does not already exist."""
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


# Schemes that are valid URLs in HTML but are not HTTP fetch targets.
# If we try to request these, requests will throw InvalidSchema.
NON_FETCHABLE_SCHEMES = {
    "mailto",
    "tel",
    "sms",
    "javascript",
    "data",
    "geo",
    "blob",
    "about",
}


def is_httpish(u: str) -> bool:
    """
    True iff the URL is http(s) or relative (no scheme).

    Why:
    - We only fetch http(s) resources.
    - Relative URLs should still be handled because we can join them to base URLs.
    """
    p = urlparse(u)
    return (p.scheme in ("http", "https")) or (p.scheme == "")


def is_non_fetchable(u: str) -> bool:
    """
    True iff the URL clearly shouldn't be fetched (mailto:, tel:, data:, ...).
    """
    p = urlparse(u)
    return p.scheme in NON_FETCHABLE_SCHEMES


def is_internal(link: str, root_netloc: str) -> bool:
    """
    Decide whether `link` belongs to the same site as `root_netloc`.

    Notes:
    - Relative URLs are internal.
    - We normalize "www." so example.com and www.example.com count as same.
    """
    parsed = urlparse(link)
    netloc = _canonical_netloc(parsed)

    if not netloc:
        return True

    if netloc == root_netloc:
        return True

    # normalize www
    if netloc.startswith("www."):
        netloc = netloc[4:]
    root = root_netloc[4:] if root_netloc.startswith("www.") else root_netloc

    return netloc == root


def _sanitize_segment(segment: str) -> str:
    """
    Sanitize a single path segment for safe writing to disk.

    - URL decode (turn %20 into space, etc.)
    - Strip whitespace / trailing dot-space combos (Windows issues)
    - Collapse accidental multi-dots
    - Replace illegal filesystem chars with '_'
    - Neutralize '.' and '..' to prevent traversal-like paths
    - Avoid Windows reserved names (CON, PRN, COM1, ...)
    """
    segment = unquote(segment).strip()
    segment = segment.strip(" .")
    segment = _MULTI_DOTS_RE.sub(".", segment)
    segment = _BAD_SEG_CHARS_RE.sub("_", segment)

    if segment in ("", ".", ".."):
        segment = "_"

    if segment.upper() in _WINDOWS_RESERVED_NAMES:
        segment = f"_{segment}_"

    return segment


def _shorten_segment(segment: str, limit: int = MAX_SEG_LEN) -> str:
    """
    Shorten a path segment if it exceeds a length limit.

    Strategy:
    - Keep the original extension
    - Truncate the stem
    - Append a short hash so different long names don't collide
    """
    if len(segment) <= limit:
        return segment
    p = Path(segment)
    stem, suffix = p.stem, p.suffix
    h = sha256(segment.encode("utf-8")).hexdigest()[:12]
    keep = max(0, limit - len(suffix) - 13)  # '-' + hash is 13 chars total
    return f"{stem[:keep]}-{h}{suffix}"


def _rel_url(target: Path, base_dir: Path) -> str:
    """
    Compute a URL-style relative path (forward slashes),
    not an OS-specific path.
    """
    try:
        rel = os.path.relpath(target, base_dir)
    except ValueError:
        # Happens if paths are on different drives on Windows.
        return target.as_posix()
    return Path(rel).as_posix()


def to_local_path(parsed: ParseResult, site_root: Path) -> Path:
    """
    Map an internal *page* URL to a local HTML file under site_root.

    Rules:
    - "/" -> index.html
    - "/foo/" -> /foo/index.html
    - "/foo" (no extension) -> /foo.html
    - query strings get a short hash to prevent collisions:
      /page?id=1 and /page?id=2 should not overwrite each other
    - filesystem hardening: sanitize segments, limit segment length and overall path
    """
    rel = parsed.path.lstrip("/")
    if not rel:
        rel = "index.html"
    elif rel.endswith("/"):
        rel += "index.html"
    elif not Path(rel).suffix:
        rel += ".html"

    if parsed.query:
        qh = sha256(parsed.query.encode("utf-8")).hexdigest()[:10]
        p = Path(rel)
        rel = str(p.with_name(f"{p.stem}-q{qh}{p.suffix}"))

    parts = Path(rel).parts
    parts = tuple(_sanitize_segment(seg) for seg in parts)
    parts = tuple(_shorten_segment(seg, MAX_SEG_LEN) for seg in parts)
    local_path = site_root / Path(*parts)

    if len(str(local_path)) > MAX_PATH_LEN:
        p = local_path
        h = sha256(parsed.geturl().encode("utf-8")).hexdigest()[:16]
        leaf = _shorten_segment(f"{p.stem}-{h}{p.suffix}", MAX_SEG_LEN)
        local_path = p.with_name(leaf)

    return local_path


def to_local_asset_path(parsed: ParseResult, site_root: Path) -> Path:
    """
    Map an internal *asset* URL to a local file path under site_root.

    Difference vs to_local_path():
    - We do NOT force .html for extensionless paths.
      (Some sites serve extensionless assets, though less common.)
    """
    rel = parsed.path.lstrip("/")
    if not rel:
        rel = "index"
    elif rel.endswith("/"):
        rel += "index"

    if parsed.query:
        qh = sha256(parsed.query.encode("utf-8")).hexdigest()[:10]
        p = Path(rel)
        name = f"{p.stem}-q{qh}{p.suffix}" if p.suffix else f"{p.name}-q{qh}"
        rel = str(p.with_name(name))

    parts = Path(rel).parts
    parts = tuple(_sanitize_segment(seg) for seg in parts)
    parts = tuple(_shorten_segment(seg, MAX_SEG_LEN) for seg in parts)
    local_path = site_root / Path(*parts)

    if len(str(local_path)) > MAX_PATH_LEN:
        p = local_path
        h = sha256(parsed.geturl().encode("utf-8")).hexdigest()[:16]
        leaf = _shorten_segment(f"{p.stem}-{h}{p.suffix}", MAX_SEG_LEN)
        local_path = p.with_name(leaf)

    return local_path


def cdn_local_path(parsed: ParseResult, site_root: Path) -> Path:
    """
    Map an external (CDN) URL to a local path under:
        site_root/cdn/<netloc>/...

    Why:
    - Keeps external host assets separated from internal assets.
    - Avoids collisions where internal and external paths look similar.
    """
    rel = parsed.path.lstrip("/")
    if not rel:
        rel = "index"
    elif rel.endswith("/"):
        rel += "index"

    if parsed.query:
        qh = sha256(parsed.query.encode("utf-8")).hexdigest()[:10]
        p = Path(rel)
        name = f"{p.stem}-q{qh}{p.suffix}" if p.suffix else f"{p.name}-q{qh}"
        rel = str(p.with_name(name))

    parts = Path(rel).parts
    parts = tuple(_sanitize_segment(seg) for seg in parts)
    parts = tuple(_shorten_segment(seg, MAX_SEG_LEN) for seg in parts)

    netloc = _canonical_netloc(parsed)
    local_path = site_root / "cdn" / _sanitize_segment(netloc) / Path(*parts)

    if len(str(local_path)) > MAX_PATH_LEN:
        p = local_path
        h = sha256(parsed.geturl().encode("utf-8")).hexdigest()[:16]
        leaf = _shorten_segment(f"{p.stem}-{h}{p.suffix}", MAX_SEG_LEN)
        local_path = p.with_name(leaf)

    return local_path


def safe_write_text(path: Path, text: str, encoding: str = "utf-8") -> Path:
    """
    Write text to path safely.

    If the OS rejects the filename/path (often: path too long), we:
    - hash the leaf name
    - write to a fallback name
    - return the final path used
    """
    try:
        path.write_text(text, encoding=encoding)
        return path
    except OSError as exc:
        _log("page_write_fail_fallback", path=path, error=exc)
        p = path
        h = sha256(str(p).encode("utf-8")).hexdigest()[:16]
        fallback = p.with_name(_shorten_segment(f"{p.stem}-{h}{p.suffix}", MAX_SEG_LEN))
        create_dir(fallback.parent)
        fallback.write_text(text, encoding=encoding)
        return fallback


def normalize_url(url: str) -> str:
    """
    Normalize URLs to avoid duplicates caused by fragments.

    Example:
    - https://site/page#section1 and https://site/page#section2
      are the same document for our crawler.
    """
    parsed = urlparse(url)
    clean = parsed._replace(fragment="")
    return clean.geturl()


def _protocol_fix(url: str, base_url: str) -> str:
    """
    Normalize protocol-relative URLs (//host/path) to absolute ones.

    Browsers interpret //example.com/a.css as "use the current page scheme".
    We do the same using base_url's scheme.
    """
    if url.startswith("//"):
        base = urlparse(base_url)
        scheme = base.scheme or "https"
        return f"{scheme}:{url}"
    return url


def parse_cookie_header(cookie_header: str) -> list[tuple[str, str]]:
    """
    Parse a Cookie header value into (name, value) pairs.

    Input example:
      "token=abc; cf_clearance=xyz; csrf-token=123"

    Notes:
    - Splits on ';' and first '=' in each segment.
    - Preserves cookie value as-is (except surrounding whitespace).
    - Skips malformed segments.
    """
    pairs: list[tuple[str, str]] = []
    for raw in cookie_header.split(";"):
        raw = raw.strip()
        if not raw or "=" not in raw:
            continue
        k, v = raw.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue
        pairs.append((k, v))
    return pairs


def _parse_json_list(raw: str) -> Optional[list[str]]:
    """
    Parse a JSON array string into a cleaned list of strings.

    Returns None when the input is not valid JSON list syntax so callers can
    fall back to simpler separators.
    """
    text = raw.strip()
    if not text.startswith("["):
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    return [str(item).strip() for item in parsed if str(item).strip()]


def parse_env_token_list(raw: str) -> list[str]:
    """
    Parse .env list values that may be written as JSON, comma-separated,
    newline-separated, or whitespace-separated tokens.

    Intended for URL/domain lists such as START_URLS, SEED_URLS, and
    EXTERNAL_DOMAINS.
    """
    text = raw.strip()
    if not text:
        return []

    json_items = _parse_json_list(text)
    if json_items is not None:
        return json_items

    items: list[str] = []
    for line in text.replace("\r", "\n").splitlines():
        chunk = line.strip()
        if not chunk:
            continue
        if "," in chunk:
            items.extend(part.strip() for part in chunk.split(",") if part.strip())
        else:
            items.extend(part.strip() for part in chunk.split() if part.strip())
    return items


def parse_env_csv_list(raw: str) -> list[str]:
    """
    Parse .env list values that should only split on commas, with optional
    JSON array support.

    Intended for selectors and similar values that may contain spaces.
    """
    text = raw.strip()
    if not text:
        return []

    json_items = _parse_json_list(text)
    if json_items is not None:
        return json_items

    return [item.strip() for item in text.split(",") if item.strip()]


def _sample(values: list[str], limit: int = 8) -> list[str]:
    """Return a short stable sample for debug logs."""
    out = sorted(set(v for v in values if v))
    return out[:limit]


def _domain_matches(candidate: str, hosts: set[str]) -> bool:
    """True when a cookie domain belongs to one of the first-party hosts."""
    candidate = candidate.lstrip(".").lower()
    if not candidate:
        return False
    return any(
        candidate == host or candidate.endswith("." + host) or host.endswith("." + candidate)
        for host in hosts
    )


def load_storage_state_into_session(
    storage_state_path: Path,
    first_party_hosts: set[str],
) -> None:
    """
    Import first-party Playwright storage_state cookies into requests.Session.

    This lets asset downloads reuse the same authenticated browser cookies when
    the user relies on PLAYWRIGHT_STORAGE_STATE instead of manual COOKIE=.
    """
    try:
        state = json.loads(storage_state_path.read_text(encoding="utf-8"))
    except OSError as exc:
        _log(
            "storage_state_session_cookie_load_failed",
            path=str(storage_state_path),
            error=str(exc),
        )
        return
    except json.JSONDecodeError as exc:
        _log(
            "storage_state_session_cookie_parse_failed",
            path=str(storage_state_path),
            error=str(exc),
        )
        return

    raw_cookies = state.get("cookies")
    if not isinstance(raw_cookies, list):
        _log(
            "storage_state_session_cookie_missing",
            path=str(storage_state_path),
            hint="storage_state JSON does not contain a cookies list",
        )
        return

    imported_names: list[str] = []
    for cookie in raw_cookies:
        if not isinstance(cookie, dict):
            continue
        name = str(cookie.get("name", "")).strip()
        value = str(cookie.get("value", "")).strip()
        domain = str(cookie.get("domain", "")).strip()
        path = str(cookie.get("path", "/")).strip() or "/"
        if not name or not value:
            continue
        if domain and first_party_hosts and not _domain_matches(domain, first_party_hosts):
            continue

        kwargs = {"path": path}
        if domain:
            kwargs["domain"] = domain
        SESSION.cookies.set(name, value, **kwargs)
        imported_names.append(name)

    if imported_names:
        _log(
            "storage_state_session_cookies_injected",
            path=str(storage_state_path),
            count=len(imported_names),
            names=sorted(set(imported_names)),
        )
    else:
        _log(
            "storage_state_session_cookies_empty",
            path=str(storage_state_path),
            hosts=sorted(first_party_hosts),
            hint="No first-party cookies were imported into the requests session",
        )


def _cookie_meta_for_debug(
    cookies: list[dict],
    expected_cookie_names: list[str],
) -> list[dict]:
    """
    Return non-sensitive cookie metadata for expected cookie names only.
    """
    expected = {n.lower() for n in expected_cookie_names}
    out: list[dict] = []
    for c in cookies:
        name = str(c.get("name", "")).strip()
        if not name or name.lower() not in expected:
            continue
        out.append(
            {
                "name": name,
                "domain": c.get("domain"),
                "path": c.get("path"),
                "secure": c.get("secure"),
                "http_only": c.get("httpOnly"),
                "same_site": c.get("sameSite"),
                "expires": c.get("expires"),
            }
        )
    return out


def inject_runtime_hide_selectors(soup: BeautifulSoup, selectors: list[str]) -> bool:
    """
    Inject CSS that hides selectors even if a client-side app re-adds them later.

    This is useful for SPA overlays that appear only after hydration. Static
    removal alone is not enough in that case because the elements do not exist
    in the initial HTML response.
    """
    cleaned = [s.strip() for s in selectors if s and s.strip()]
    if not cleaned:
        return False

    style = soup.new_tag("style")
    style["data-dds-runtime-hide"] = "1"
    style.string = (
        "/* Hide dynamic overlays added after offline hydration. */\n"
        f"{', '.join(cleaned)}"
        "{display:none !important;visibility:hidden !important;"
        "opacity:0 !important;pointer-events:none !important;}"
    )

    head = soup.head
    if head is not None:
        head.append(style)
        return True

    html = soup.html
    if html is not None:
        head = soup.new_tag("head")
        head.append(style)
        html.insert(0, head)
        return True

    soup.insert(0, style)
    return True


def inject_runtime_asset_fixups(
    soup: BeautifulSoup,
    site_root: Path,
    page_dir: Path,
) -> bool:
    """
    Inject a small runtime shim that fixes asset URLs generated by hydrated
    Next.js code when pages are opened directly from the filesystem.

    This covers cases like:
    - /_next/image?url=%2Fimages%2F...
    - /media/... that should point to _next/static/media/...
    - wrong relative media paths emitted by rewritten JS bundles
    """
    root_rel = _rel_url(site_root, page_dir)
    if root_rel in ("", "."):
        root_rel = "./"
    else:
        root_rel = root_rel.rstrip("/") + "/"

    script = soup.new_tag("script")
    script["data-dds-runtime-assets"] = "1"
    script.string = (
        "(() => {\n"
        f"  const SITE_ROOT = {json.dumps(root_rel)};\n"
        "  const DOT_SEG_RE = /^(?:\\.\\/)?(?:\\.\\.\\/)+/;\n"
        "  const ABS_SCHEME_RE = /^[a-zA-Z][a-zA-Z\\d+.-]*:/;\n"
        "  function toLocal(value) {\n"
        "    if (value == null) return value;\n"
        "    let raw = String(value);\n"
        "    if (!raw || raw.startsWith('#')) return raw;\n"
        "    if (/^(?:data|blob|javascript|mailto|tel):/i.test(raw)) return raw;\n"
        "    if (raw.startsWith('file:')) return raw;\n"
        "    if (ABS_SCHEME_RE.test(raw)) return raw;\n"
        "    if (raw.startsWith('/_next/image?')) {\n"
        "      try {\n"
        "        const parsed = new URL(raw, 'https://offline.local');\n"
        "        raw = parsed.searchParams.get('url') || raw;\n"
        "      } catch (_) {}\n"
        "    }\n"
        "    raw = raw.replace(/\\\\\\//g, '/');\n"
        "    if (raw.startsWith('/media/')) return SITE_ROOT + '_next/static' + raw;\n"
        "    if (raw.startsWith('/')) return SITE_ROOT + raw.slice(1);\n"
        "    const stripped = raw.replace(DOT_SEG_RE, '');\n"
        "    if (stripped.startsWith('media/')) return SITE_ROOT + '_next/static/' + stripped;\n"
        "    return raw;\n"
        "  }\n"
        "  function rewriteSrcset(value) {\n"
        "    if (value == null) return value;\n"
        "    return String(value).split(',').map((entry) => {\n"
        "      const trimmed = entry.trim();\n"
        "      if (!trimmed) return trimmed;\n"
        "      const parts = trimmed.split(/\\s+/);\n"
        "      parts[0] = toLocal(parts[0]);\n"
        "      return parts.join(' ');\n"
        "    }).join(', ');\n"
        "  }\n"
        "  function forceEagerImage(el) {\n"
        "    if (!el || el.tagName !== 'IMG') return false;\n"
        "    const attrLoading = (el.getAttribute('loading') || '').toLowerCase();\n"
        "    const propLoading = String(el.loading || '').toLowerCase();\n"
        "    const hadLazy = attrLoading === 'lazy' || propLoading === 'lazy';\n"
        "    if (hadLazy) {\n"
        "      el.loading = 'eager';\n"
        "      el.setAttribute('loading', 'eager');\n"
        "    }\n"
        "    return hadLazy;\n"
        "  }\n"
        "  function patchProto(proto) {\n"
        "    if (!proto) return;\n"
        "    for (const [prop, mapper] of [['src', toLocal], ['srcset', rewriteSrcset]]) {\n"
        "      const desc = Object.getOwnPropertyDescriptor(proto, prop);\n"
        "      if (!desc || !desc.set || !desc.get) continue;\n"
        "      Object.defineProperty(proto, prop, {\n"
        "        configurable: true,\n"
        "        enumerable: desc.enumerable,\n"
        "        get() { return desc.get.call(this); },\n"
        "        set(v) { return desc.set.call(this, mapper(v)); },\n"
        "      });\n"
        "    }\n"
        "  }\n"
        "  function processOne(el) {\n"
        "    if (!el || el.nodeType !== 1) return;\n"
        "    const retriggerLoad = forceEagerImage(el);\n"
        "    if (el.hasAttribute && el.hasAttribute('src')) {\n"
        "      const v = el.getAttribute('src');\n"
        "      const mapped = toLocal(v);\n"
        "      if (mapped !== v) el.setAttribute('src', mapped);\n"
        "    }\n"
        "    if (el.hasAttribute && el.hasAttribute('srcset')) {\n"
        "      const v = el.getAttribute('srcset');\n"
        "      const mapped = rewriteSrcset(v);\n"
        "      if (mapped !== v) el.setAttribute('srcset', mapped);\n"
        "    }\n"
        "    if (retriggerLoad && !el.hasAttribute('data-dds-eager-load')) {\n"
        "      el.setAttribute('data-dds-eager-load', '1');\n"
        "      if (el.hasAttribute('src')) el.setAttribute('src', el.getAttribute('src'));\n"
        "      if (el.hasAttribute('srcset')) el.setAttribute('srcset', el.getAttribute('srcset'));\n"
        "    }\n"
        "  }\n"
        "  function scan(root) {\n"
        "    if (!root || root.nodeType !== 1 && root.nodeType !== 9) return;\n"
        "    processOne(root);\n"
        "    if (root.querySelectorAll) root.querySelectorAll('[src], [srcset]').forEach(processOne);\n"
        "  }\n"
        "  patchProto(window.HTMLImageElement && HTMLImageElement.prototype);\n"
        "  patchProto(window.HTMLSourceElement && HTMLSourceElement.prototype);\n"
        "  const origSetAttribute = Element.prototype.setAttribute;\n"
        "  Element.prototype.setAttribute = function(name, value) {\n"
        "    if (name === 'src') value = toLocal(value);\n"
        "    else if (name === 'srcset') value = rewriteSrcset(value);\n"
        "    else if (name === 'loading' && this.tagName === 'IMG' && String(value).toLowerCase() === 'lazy') value = 'eager';\n"
        "    return origSetAttribute.call(this, name, value);\n"
        "  };\n"
        "  scan(document);\n"
        "  document.addEventListener('DOMContentLoaded', () => scan(document), { once: true });\n"
        "  window.addEventListener('load', () => scan(document), { once: true });\n"
        "  new MutationObserver((mutations) => {\n"
        "    for (const mutation of mutations) {\n"
        "      if (mutation.type === 'attributes') processOne(mutation.target);\n"
        "      for (const node of mutation.addedNodes || []) scan(node);\n"
        "    }\n"
        "  }).observe(document.documentElement, {\n"
        "    subtree: true,\n"
        "    childList: true,\n"
        "    attributes: true,\n"
        "    attributeFilter: ['src', 'srcset'],\n"
        "  });\n"
        "})();"
    )

    head = soup.head
    if head is not None:
        head.insert(0, script)
        return True

    html = soup.html
    if html is not None:
        head = soup.new_tag('head')
        head.append(script)
        html.insert(0, head)
        return True

    soup.insert(0, script)
    return True


def _parse_jwt_claims_unverified(token: str) -> Optional[dict]:
    """
    Parse JWT payload claims without verifying signature.

    This is only for diagnostics (exp/iat visibility), never for trust.
    """
    parts = token.split(".")
    if len(parts) < 2:
        return None

    payload = parts[1]
    payload += "=" * ((4 - len(payload) % 4) % 4)

    try:
        decoded = base64.urlsafe_b64decode(payload.encode("ascii"))
        parsed = json.loads(decoded.decode("utf-8"))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def log_token_lifetime(token_cookie: str) -> None:
    """
    Emit diagnostic logs for JWT token lifetime (iat/exp).

    Helps identify auth failures caused by expired tokens copied from browser.
    """
    claims = _parse_jwt_claims_unverified(token_cookie)
    if not claims:
        _log("token_claims_unreadable")
        return

    now_epoch = int(time.time())

    exp_raw = claims.get("exp")
    iat_raw = claims.get("iat")

    exp_epoch = int(exp_raw) if isinstance(exp_raw, (int, float, str)) and str(exp_raw).isdigit() else None
    iat_epoch = int(iat_raw) if isinstance(iat_raw, (int, float, str)) and str(iat_raw).isdigit() else None

    exp_utc = (
        datetime.fromtimestamp(exp_epoch, tz=timezone.utc).isoformat()
        if exp_epoch is not None
        else None
    )
    iat_utc = (
        datetime.fromtimestamp(iat_epoch, tz=timezone.utc).isoformat()
        if iat_epoch is not None
        else None
    )

    _log(
        "token_lifetime",
        now_epoch=now_epoch,
        now_utc=datetime.fromtimestamp(now_epoch, tz=timezone.utc).isoformat(),
        iat_epoch=iat_epoch,
        iat_utc=iat_utc,
        exp_epoch=exp_epoch,
        exp_utc=exp_utc,
        seconds_to_exp=(exp_epoch - now_epoch) if exp_epoch is not None else None,
    )

    if exp_epoch is not None and now_epoch >= exp_epoch:
        _log(
            "token_expired",
            now_epoch=now_epoch,
            exp_epoch=exp_epoch,
            expired_by_s=now_epoch - exp_epoch,
        )


def rewrite_css_text(
    css_text: str,
    base_url: str,
    *,
    site_root: Path,
    root_netloc: str,
    base_dir: Path,
    download_external_assets: bool,
    external_domains: Optional[set[str]] = None,
    download_q: Optional[queue.Queue[tuple[str, Path]]] = None,
) -> str:
    """
    Rewrite CSS url(...) and @import references to local relative paths.

    base_url:
      - the remote URL of the CSS *context*
      - external stylesheet URL for downloaded .css
      - page URL for inline <style> blocks or style="..."

    base_dir:
      - local directory where this CSS lives (controls the relative path output)

    Also:
    - If download_q is provided, enqueue newly discovered assets referenced by CSS.
    """

    def map_one(url_part: str) -> Optional[str]:
        url_part = url_part.strip()

        # Skip empties / anchors / non-fetchable schemes.
        if not url_part:
            return None
        if url_part.startswith("#"):
            return None
        if url_part.startswith(("data:", "javascript:", "about:")):
            return None

        url_part2 = _protocol_fix(url_part, base_url)
        if is_non_fetchable(url_part2) or not is_httpish(url_part2):
            return None

        # Canonicalize to a stable absolute URL
        abs_url = canonicalize_url(url_part2, base_url)
        parsed = urlparse(abs_url)
        if not parsed.path:
            return None

        # Only rewrite things that look like static assets.
        # (Avoid rewriting API URLs accidentally.)
        if not parsed.path.lower().endswith(ASSET_EXTENSIONS):
            return None

        is_ext = not is_internal(abs_url, root_netloc)
        if is_ext and not is_allowed_external(abs_url, external_domains):
            return None

        if is_ext and not download_external_assets:
            return None

        # Decide where to store it locally
        local_path = (
            cdn_local_path(parsed, site_root)
            if is_ext
            else to_local_asset_path(parsed, site_root)
        )

        # Queue it for downloading if not already present
        if download_q is not None and not local_path.exists():
            download_q.put((abs_url, local_path))

        # Output a relative URL for the rewritten CSS
        rel = _rel_url(local_path, base_dir)
        if parsed.fragment:
            rel = f"{rel}#{parsed.fragment}"
        return rel

    # Replace url(...) references
    def repl_url(m: re.Match) -> str:
        raw = m.group(1).strip()
        quote = ""
        url_part = raw

        # Preserve quoting style if present
        if len(raw) >= 2 and raw[0] in ("'", '"') and raw[-1] == raw[0]:
            quote = raw[0]
            url_part = raw[1:-1].strip()

        mapped = map_one(url_part)
        if mapped is None:
            return m.group(0)

        if quote:
            return f"url({quote}{mapped}{quote})"
        return f"url({mapped})"

    # Replace @import references
    def repl_import(m: re.Match) -> str:
        url_part = m.group(1).strip().strip("'\"")
        mapped = map_one(url_part)
        if mapped is None:
            return m.group(0)
        return f'@import "{mapped}";'

    css_text = CSS_URL_RE.sub(repl_url, css_text)
    css_text = CSS_IMPORT_RE.sub(repl_import, css_text)
    return css_text


def _map_static_asset_reference(
    url_part: str,
    base_url: str,
    *,
    site_root: Path,
    root_netloc: str,
    base_dir: Path,
    download_external_assets: bool,
    external_domains: Optional[set[str]] = None,
    download_q: Optional[queue.Queue[tuple[str, Path]]] = None,
) -> Optional[str]:
    """
    Map a static asset URL to the local offline path.

    Returns a relative URL suitable for writing back into HTML/CSS/JS or
    ``None`` if the value should be left unchanged.
    """
    url_part = url_part.strip()

    if not url_part:
        return None
    if url_part.startswith("#"):
        return None
    if url_part.startswith(("data:", "javascript:", "about:")):
        return None

    url_part2 = _protocol_fix(url_part, base_url)
    if is_non_fetchable(url_part2) or not is_httpish(url_part2):
        return None

    abs_url = canonicalize_url(url_part2, base_url)
    parsed = urlparse(abs_url)

    if not parsed.path.lower().endswith(ASSET_EXTENSIONS):
        return None

    is_ext = not is_internal(abs_url, root_netloc)
    if is_ext and not is_allowed_external(abs_url, external_domains):
        return None

    if is_ext and not download_external_assets:
        return None

    local_path = (
        cdn_local_path(parsed, site_root)
        if is_ext
        else to_local_asset_path(parsed, site_root)
    )

    if download_q is not None and not local_path.exists():
        download_q.put((abs_url, local_path))

    rel = _rel_url(local_path, base_dir)
    if parsed.fragment:
        rel = f"{rel}#{parsed.fragment}"
    return rel


def rewrite_js_text(
    js_text: str,
    base_url: str,
    *,
    site_root: Path,
    root_netloc: str,
    base_dir: Path,
    download_external_assets: bool,
    external_domains: Optional[set[str]] = None,
    download_q: Optional[queue.Queue[tuple[str, Path]]] = None,
) -> str:
    """
    Rewrite obvious static asset URL strings inside JS.

    Important:
    - This does NOT parse JS AST; it does simple regex matching on string literals.
    - It ONLY rewrites strings that look like static assets by extension.
    - This prevents accidentally rewriting API endpoints or app routes.
    """

    def repl_root_rel(m: re.Match) -> str:
        url_part = m.group(1)
        mapped = _map_static_asset_reference(
            url_part,
            base_url,
            site_root=site_root,
            root_netloc=root_netloc,
            base_dir=base_dir,
            download_external_assets=download_external_assets,
            external_domains=external_domains,
            download_q=download_q,
        )
        if mapped is None:
            return m.group(0)
        quote = m.group(0)[0]
        return f"{quote}{mapped}{quote}"

    def repl_abs(m: re.Match) -> str:
        url_part = m.group(1)
        mapped = _map_static_asset_reference(
            url_part,
            base_url,
            site_root=site_root,
            root_netloc=root_netloc,
            base_dir=base_dir,
            download_external_assets=download_external_assets,
            external_domains=external_domains,
            download_q=download_q,
        )
        if mapped is None:
            return m.group(0)
        quote = m.group(0)[0]
        return f"{quote}{mapped}{quote}"

    js_text = JS_URL_RE.sub(repl_root_rel, js_text)
    js_text = JS_ABS_URL_RE.sub(repl_abs, js_text)
    return js_text


def rewrite_next_data_json_text(
    raw_json: str,
    base_url: str,
    *,
    site_root: Path,
    root_netloc: str,
    base_dir: Path,
    download_external_assets: bool,
    external_domains: Optional[set[str]] = None,
    download_q: Optional[queue.Queue[tuple[str, Path]]] = None,
) -> tuple[str, int]:
    """
    Rewrite asset references inside a Next.js ``__NEXT_DATA__`` JSON payload.

    This is more reliable than regexing the raw JSON script because important
    chapter assets are often hidden inside string leaves like ``pageProps.code``.
    """
    try:
        data = json.loads(raw_json)
    except Exception:  # noqa: BLE001
        return raw_json, 0

    rewrites = 0

    def walk(node: object) -> object:
        nonlocal rewrites

        if isinstance(node, str):
            rewritten_node = node

            mapped = _map_static_asset_reference(
                rewritten_node,
                base_url,
                site_root=site_root,
                root_netloc=root_netloc,
                base_dir=base_dir,
                download_external_assets=download_external_assets,
                external_domains=external_domains,
                download_q=download_q,
            )
            if mapped is not None and mapped != rewritten_node:
                rewritten_node = mapped
                rewrites += 1

            if "/" in rewritten_node and ("'" in rewritten_node or '"' in rewritten_node):
                js_rewritten = rewrite_js_text(
                    rewritten_node,
                    base_url,
                    site_root=site_root,
                    root_netloc=root_netloc,
                    base_dir=base_dir,
                    download_external_assets=download_external_assets,
                    external_domains=external_domains,
                    download_q=download_q,
                )
                if js_rewritten != rewritten_node:
                    rewritten_node = js_rewritten
                    rewrites += 1

            return rewritten_node

        if isinstance(node, list):
            return [walk(item) for item in node]

        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items()}

        return node

    rewritten_data = walk(data)
    if rewrites == 0:
        return raw_json, 0

    serialized = json.dumps(
        rewritten_data,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    serialized = (
        serialized.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    return serialized, rewrites


def _canonical_netloc(parsed: ParseResult) -> str:
    """
    Lowercase hostname and drop default ports so we don't create different
    local folders for the same host.

    Example:
      https://EXAMPLE.com:443/a.css -> example.com
    """
    host = (parsed.hostname or "").lower()
    port = parsed.port
    if not host:
        return parsed.netloc.lower()

    if (parsed.scheme == "https" and port == 443) or (
        parsed.scheme == "http" and port == 80
    ):
        port = None

    return f"{host}:{port}" if port else host


def canonicalize_url(url: str, base_url: str = "") -> str:
    """
    Produce a stable absolute URL key for de-duping + mapping.

    Steps:
    - Fix protocol-relative URLs
    - Join relative URLs against base_url
    - Drop fragments (#...)
    - Normalize host casing + default ports
    """
    if base_url:
        url = urljoin(base_url, _protocol_fix(url, base_url))
    else:
        url = _protocol_fix(url, url)

    p = urlparse(url)

    # If still relative, join using base_url (when available).
    if not p.scheme and not p.netloc:
        p = urlparse(urljoin(base_url, url)) if base_url else p

    netloc = _canonical_netloc(p) if p.netloc else ""
    p = p._replace(fragment="", netloc=netloc)
    return p.geturl()


def is_allowed_external(url: str, allowed_domains: Optional[set[str]]) -> bool:
    if allowed_domains is None:
        return True

    host = (urlparse(url).hostname or "").lower()

    return any(host == d or host.endswith("." + d) for d in allowed_domains)


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------


def fetch_html(url: str) -> Optional[BeautifulSoup]:
    """
    Download an HTML page and return a BeautifulSoup tree.

    We return None on error so the crawler can continue on failures.
    """
    try:
        resp = SESSION.get(url, timeout=TIMEOUT)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")
    except Exception as exc:  # noqa: BLE001
        _log("http_error", url=url, error=exc)
        return None


def fetch_html_rendered(
    url: str,
    context: "BrowserContext",
    wait_for: str = "body",
    render_settle_ms: int = 4000,
    debug_auth: bool = False,
    expected_cookie_names: Optional[list[str]] = None,
) -> Optional[BeautifulSoup]:
    """
    Render a page with a headless Chromium browser and return the post-JS DOM.

    Use this instead of fetch_html() for SPAs (Next.js, React, etc.) where
    the server sends an empty shell and content is injected by JavaScript.

    context: a Playwright BrowserContext that already has auth cookies injected.
    wait_for: CSS selector to wait for before extracting the page HTML.
              Default "body" works for any page; use something specific like
              "article" or "[class*='lesson']" to ensure content has rendered.
    """
    if not HAS_PLAYWRIGHT:
        _log("playwright_missing", error="Playwright not installed. Run: uv add playwright && uv run playwright install chromium")
        return None
    page = context.new_page()

    expected_cookie_names = [n for n in (expected_cookie_names or []) if n]
    pre_cookie_names: list[str] = []
    pre_cookie_count = 0
    pre_cookie_meta: list[dict] = []
    nav_status: Optional[int] = None
    nav_set_cookie_hint: dict[str, dict[str, bool]] = {}
    first_party_host = (urlparse(url).hostname or "").lower()

    net = {
        "xhr_fetch_total": 0,
        "xhr_fetch_with_auth": 0,
        "failed_requests": 0,
    }
    xhr_status = Counter()
    xhr_fail_urls: list[str] = []
    xhr_sample: list[str] = []
    request_fail_urls: list[str] = []
    request_auth_probe: list[dict] = []
    console_errors: list[str] = []

    def _is_first_party(req_url: str) -> bool:
        host = (urlparse(req_url).hostname or "").lower()
        if not host or not first_party_host:
            return False
        return host == first_party_host or host.endswith("." + first_party_host)

    def on_response(resp) -> None:  # noqa: ANN001
        try:
            req = resp.request
            if req.resource_type in ("xhr", "fetch"):
                net["xhr_fetch_total"] += 1
                status = int(resp.status)
                xhr_status[status] += 1
                if status >= 400 and len(xhr_fail_urls) < 10:
                    xhr_fail_urls.append(f"{status} {resp.url}")
                headers = req.headers or {}
                has_auth = any(k.lower() == "authorization" and str(v).strip() for k, v in headers.items())
                if has_auth:
                    net["xhr_fetch_with_auth"] += 1
                if _is_first_party(resp.url) and len(xhr_sample) < 12:
                    p = urlparse(resp.url)
                    xhr_sample.append(f"{status} {p.path or '/'} auth={has_auth}")
        except Exception:
            # Keep crawler resilient even if Playwright internals change.
            pass

    def on_request(req) -> None:  # noqa: ANN001
        try:
            if req.resource_type not in ("document", "xhr", "fetch"):
                return
            if not _is_first_party(req.url):
                return
            if len(request_auth_probe) >= 20:
                return
            headers = req.headers or {}
            cookie_header = ""
            for k, v in headers.items():
                if k.lower() == "cookie":
                    cookie_header = str(v)
                    break
            auth_present = any(k.lower() == "authorization" and str(v).strip() for k, v in headers.items())
            p = urlparse(req.url)
            request_auth_probe.append(
                {
                    "type": req.resource_type,
                    "path": p.path or "/",
                    "has_authorization": auth_present,
                    "has_cookie_header": bool(cookie_header),
                    "cookie_has_token": ("token=" in cookie_header),
                    "cookie_has_csrf_token": ("csrf-token=" in cookie_header),
                }
            )
        except Exception:
            pass

    def on_request_failed(req) -> None:  # noqa: ANN001
        try:
            net["failed_requests"] += 1
            if len(request_fail_urls) < 10:
                err = req.failure or ""
                request_fail_urls.append(f"{req.resource_type} {req.url} {err}")
        except Exception:
            pass

    def on_console(msg) -> None:  # noqa: ANN001
        try:
            if msg.type in ("error", "warning") and len(console_errors) < 12:
                console_errors.append(f"{msg.type}: {msg.text}")
        except Exception:
            pass

    page.on("request", on_request)
    page.on("response", on_response)
    page.on("requestfailed", on_request_failed)
    page.on("console", on_console)

    try:
        if debug_auth:
            try:
                pre_cookies = context.cookies([url])
                pre_cookie_names = _sample(
                    [str(c.get("name", "")).strip() for c in pre_cookies], limit=30
                )
                pre_cookie_count = len(pre_cookies)
                pre_cookie_meta = _cookie_meta_for_debug(pre_cookies, expected_cookie_names)
                pre_name_set = {n.lower() for n in pre_cookie_names}
                pre_missing = [
                    name for name in expected_cookie_names
                    if name.lower() not in pre_name_set
                ]
                _log(
                    "playwright_auth_cookie_probe",
                    url=url,
                    stage="before_goto",
                    cookie_count=pre_cookie_count,
                    cookie_names=pre_cookie_names,
                    cookie_meta=pre_cookie_meta,
                    expected_cookie_names=expected_cookie_names,
                    missing_expected_cookies=pre_missing,
                )
            except Exception as exc:  # noqa: BLE001
                _log("playwright_auth_cookie_probe_fail", url=url, stage="before_goto", error=exc)

        nav_response = page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        if debug_auth and nav_response is not None:
            try:
                nav_status = int(nav_response.status)
            except Exception:
                nav_status = None
            try:
                nav_headers = nav_response.headers or {}
                set_cookie_raw = str(nav_headers.get("set-cookie") or "")
                low = set_cookie_raw.lower()
                for name in expected_cookie_names:
                    key = name.lower()
                    nav_set_cookie_hint[name] = {
                        "mentioned": (f"{key}=" in low),
                        "cleared_like": (f"{key}=;" in low) or (f"{key}=deleted" in low),
                    }
            except Exception:
                nav_set_cookie_hint = {}
        try:
            page.wait_for_selector(wait_for, timeout=15_000)
        except Exception:
            _log("selector_missing_warn", selector=wait_for, url=url)
        # Give SPA auth/bootstrap calls time to update DOM state.
        try:
            page.wait_for_load_state("networkidle", timeout=max(1_000, render_settle_ms))
        except Exception:
            _log("networkidle_timeout_warn", url=url, waited_ms=max(1_000, render_settle_ms))
        if render_settle_ms > 0:
            page.wait_for_timeout(render_settle_ms)

        if debug_auth:
            state = {}
            try:
                state = page.evaluate(
                    """() => ({
                        finalUrl: window.location.href,
                        title: document.title || "",
                        readyState: document.readyState || "",
                        hasUnlock: (document.body?.innerText || "").includes("Unlock Full Access"),
                        hasLogin: (document.body?.innerText || "").includes("Login"),
                        localStorageKeys: Object.keys(window.localStorage || {}),
                        sessionStorageKeys: Object.keys(window.sessionStorage || {}),
                    })"""
                ) or {}
            except Exception as exc:  # noqa: BLE001
                _log("playwright_auth_debug_eval_fail", url=url, error=exc)

            cookies = []
            try:
                cookies = context.cookies([url])
            except Exception:
                pass

            cookie_names = _sample([str(c.get("name", "")).strip() for c in cookies], limit=20)
            post_cookie_meta = _cookie_meta_for_debug(cookies, expected_cookie_names)
            post_name_set = {n.lower() for n in cookie_names}
            missing_expected = [
                name for name in expected_cookie_names
                if name.lower() not in post_name_set
            ]
            status_map = {str(k): v for k, v in sorted(xhr_status.items())}
            _log(
                "playwright_auth_debug",
                url=url,
                final_url=page.url,
                eval_final_url=state.get("finalUrl"),
                title=state.get("title"),
                ready_state=state.get("readyState"),
                has_unlock=bool(state.get("hasUnlock")),
                has_login=bool(state.get("hasLogin")),
                nav_status=nav_status,
                nav_set_cookie_hint=nav_set_cookie_hint,
                pre_cookie_count=pre_cookie_count,
                pre_cookie_names=pre_cookie_names,
                pre_cookie_meta=pre_cookie_meta,
                cookie_count=len(cookies),
                cookie_names=cookie_names,
                post_cookie_meta=post_cookie_meta,
                expected_cookie_names=expected_cookie_names,
                missing_expected_cookies=missing_expected,
                local_storage_keys=_sample(list(state.get("localStorageKeys") or []), limit=30),
                session_storage_keys=_sample(list(state.get("sessionStorageKeys") or []), limit=30),
                xhr_fetch_total=net["xhr_fetch_total"],
                xhr_fetch_with_auth=net["xhr_fetch_with_auth"],
                xhr_status=status_map,
                xhr_fail_urls=xhr_fail_urls,
                xhr_sample=xhr_sample,
                request_auth_probe=request_auth_probe,
                request_failures=net["failed_requests"],
                request_fail_urls=request_fail_urls,
                console_errors=console_errors,
            )

        final_path = urlparse(page.url).path.rstrip("/")
        requested_path = urlparse(url).path.rstrip("/")
        if final_path == "/login" and requested_path != "/login":
            _log("playwright_login_redirect", url=url, final_url=page.url)
            return None

        html = page.content()
        return BeautifulSoup(html, "html.parser")
    except Exception as exc:  # noqa: BLE001
        _log("playwright_page_fail", url=url, error=exc)
        return None
    finally:
        page.close()


def fetch_binary(
    url: str,
    dest: Path,
    download_q: Optional[queue.Queue[tuple[str, Path]]] = None,
    *,
    site_root: Optional[Path] = None,
    root_netloc: str = "",
    download_external_assets: bool = False,
    external_domains: Optional[set[str]] = None,
) -> None:
    """
    Stream a binary/static resource to disk.

    Notes:
    - If already exists, skip.
    - Writes using streaming so we don't keep big files in memory.
    - If the file is CSS or JS, rewrite embedded asset URLs and enqueue them.
    """
    is_ext = not is_internal(url, root_netloc)

    if is_ext:
        if not download_external_assets:
            return

        if not is_allowed_external(url, external_domains):
            return

    if dest.exists():
        return

    try:
        request_headers = None
        if is_ext and any(k.lower() == "authorization" for k in SESSION.headers.keys()):
            # Never forward first-party bearer tokens to third-party CDNs.
            request_headers = {"Authorization": None}

        resp = SESSION.get(url, timeout=TIMEOUT, stream=True, headers=request_headers)
        resp.raise_for_status()
        content_type = str(resp.headers.get("content-type", "")).lower()

        # Assets that resolve to HTML are usually route fallbacks, auth redirects,
        # or metadata links (for example canonical URLs), not real static files.
        if "text/html" in content_type or "application/xhtml+xml" in content_type:
            _log(
                "asset_skip_html_response",
                url=url,
                dest=dest,
                content_type=content_type,
            )
            resp.close()
            return

        create_dir(dest.parent)

        # Try normal write
        try:
            with dest.open("wb") as fh:
                for chunk in resp.iter_content(CHUNK_SIZE):
                    if chunk:
                        fh.write(chunk)

        # If filesystem rejects it (path too long, invalid name), fallback
        except OSError as exc:
            _log("asset_write_fail_fallback", dest=dest, error=exc)

            h = sha256(str(dest).encode("utf-8")).hexdigest()[:16]
            fallback = dest.with_name(
                _shorten_segment(f"{dest.stem}-{h}{dest.suffix}", MAX_SEG_LEN)
            )
            create_dir(fallback.parent)

            with fallback.open("wb") as fh:
                for chunk in resp.iter_content(CHUNK_SIZE):
                    if chunk:
                        fh.write(chunk)

            dest = fallback

        # If we downloaded CSS, rewrite its url(...) and @import references,
        # and enqueue referenced assets (images/fonts/etc).
        if (
            dest.suffix.lower() == ".css"
            and download_q is not None
            and site_root is not None
            and root_netloc
        ):
            try:
                css_text = dest.read_text(encoding="utf-8", errors="ignore")
                rewritten = rewrite_css_text(
                    css_text,
                    url,
                    site_root=site_root,
                    root_netloc=root_netloc,
                    base_dir=dest.parent,
                    download_external_assets=download_external_assets,
                    external_domains=external_domains,
                    download_q=download_q,
                )
                if rewritten != css_text:
                    dest.write_text(rewritten, encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                _log("css_rewrite_fail", dest=dest, error=exc)

        # If we downloaded JS, rewrite obvious static URL strings,
        # and enqueue referenced assets (only those matching ASSET_EXTENSIONS).
        if (
            dest.suffix.lower() in {".js", ".mjs"}
            and download_q is not None
            and site_root is not None
            and root_netloc
        ):
            try:
                js_text = dest.read_text(encoding="utf-8", errors="ignore")
                rewritten = rewrite_js_text(
                    js_text,
                    url,
                    site_root=site_root,
                    root_netloc=root_netloc,
                    base_dir=dest.parent,
                    download_external_assets=download_external_assets,
                    external_domains=external_domains,
                    download_q=download_q,
                )
                if rewritten != js_text:
                    dest.write_text(rewritten, encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                _log("js_rewrite_fail", dest=dest, error=exc)

    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else 0
        if status in (404, 410):
            # Asset simply doesn't exist on the server — common for stale HTML
            # references. Log as WARN (not ERROR) to avoid false alarms.
            _log("asset_not_found", url=url, status=status)
        else:
            _log("asset_download_fail", url=url, error=exc)
    except Exception as exc:  # noqa: BLE001
        _log("asset_download_fail", url=url, error=exc)


# ---------------------------------------------------------------------------
# Link rewriting
# ---------------------------------------------------------------------------


def rewrite_links(
    soup: BeautifulSoup,
    page_url: str,
    site_root: Path,
    page_dir: Path,
    download_external_assets: bool = False,
    external_domains: Optional[set[str]] = None,
    download_q: Optional[queue.Queue[tuple[str, Path]]] = None,
) -> None:
    """
    Rewrite HTML so it can be opened offline.

    Rules:
    - Internal page links (<a href>) become local HTML file paths.
    - Internal asset links (img/src, script/src, link/href, etc) become local asset paths.
    - External asset links are rewritten to local cdn/... paths when
        external downloading is enabled and the URL is allowed.
    - External page links (for example <a href="https://...">) are kept unchanged.
    - Remove <base href="..."> because it changes browser URL resolution offline.
    """
    root_netloc = _canonical_netloc(urlparse(page_url))

    # <base href> breaks relative paths when opening offline.
    base_tag = soup.find("base")
    if base_tag is not None and base_tag.has_attr("href"):
        base_tag.decompose()

    # Common attributes that contain URL-like values.
    url_attrs = {"src", "href", "data-src", "poster", "xlink:href"}

    def strip_sri_and_cors(tag) -> None:
        for attr in ("integrity", "crossorigin"):
            if tag.has_attr(attr):
                del tag[attr]

    for tag in soup.find_all(True):
        if tag.name == "img" and str(tag.get("loading", "")).lower() == "lazy":
            tag["loading"] = "eager"

        # For <link>, only rewrite rel-types that are actually fetched by browsers.
        # This avoids rewriting <link rel="canonical"> or <link rel="alternate"> etc.
        if tag.name == "link":
            rel = tag.get("rel", [])
            if isinstance(rel, str):
                rel = [rel]
            rel = [r.lower() for r in rel]

            rel_set = set(rel)
            if not rel_set & RESOURCE_LINK_RELS:
                continue

        # ------------------------------------------------------------------
        # META IMAGE REWRITE (make og/twitter images local)
        # ------------------------------------------------------------------
        if tag.name == "meta":
            content = str(tag.get("content", "")).strip()
            prop = (tag.get("property") or tag.get("name") or "").lower()

            if content and ("og:image" in prop or "twitter:image" in prop):

                url_part = _protocol_fix(content, page_url)

                if (
                    not url_part
                    or url_part.startswith("#")
                    or url_part.startswith(("data:", "javascript:", "about:"))
                    or is_non_fetchable(url_part)
                    or not is_httpish(url_part)
                ):
                    continue

                abs_url = canonicalize_url(url_part, page_url)
                parsed = urlparse(abs_url)

                is_ext = not is_internal(abs_url, root_netloc)

                if is_ext:
                    if not download_external_assets:
                        continue
                    if not is_allowed_external(abs_url, external_domains):
                        continue

                # map to local path
                local_path = (
                    cdn_local_path(parsed, site_root)
                    if is_ext
                    else to_local_asset_path(parsed, site_root)
                )

                # rewrite to relative path
                rel = _rel_url(local_path, page_dir)
                tag["content"] = rel

        # Rewrite each URL attribute we care about
        for attr in url_attrs:
            if not tag.has_attr(attr):
                continue

            original_raw = str(tag.get(attr, "")).strip()
            if not original_raw:
                continue

            original = _protocol_fix(original_raw, page_url)

            # Skip anchors, non-fetchable schemes, and things that are not http(s)/relative.
            if (
                original.startswith("#")
                or is_non_fetchable(original)
                or not is_httpish(original)
            ):
                continue

            abs_url = canonicalize_url(original, page_url)
            parsed = urlparse(abs_url)

            is_ext = not is_internal(abs_url, root_netloc)
            if is_ext:
                if not download_external_assets:
                    continue
                if not is_allowed_external(abs_url, external_domains):
                    continue

            # Treat <a href> as a "page". Everything else is treated as an asset.
            treat_as_page = tag.name == "a" and attr == "href"

            rewritten_external_asset = False

            if is_ext and treat_as_page:
                continue

            if is_ext:
                if not download_external_assets:
                    continue
                if not is_allowed_external(abs_url, external_domains):
                    continue
                local_path = cdn_local_path(parsed, site_root)
                rewritten_external_asset = True
            else:
                local_path = (
                    to_local_path(parsed, site_root)
                    if treat_as_page
                    else to_local_asset_path(parsed, site_root)
                )

            rel = _rel_url(local_path, page_dir)
            if parsed.fragment:
                rel = f"{rel}#{parsed.fragment}"
            tag[attr] = rel

            if rewritten_external_asset and tag.name in {"script", "link"}:
                strip_sri_and_cors(tag)

        # srcset="url1 1x, url2 2x" needs special parsing
        if tag.has_attr("srcset"):
            new_entries = []
            for entry in str(tag["srcset"]).split(","):
                entry = entry.strip()
                if not entry:
                    continue

                parts = entry.split()
                url_part = _protocol_fix(parts[0], page_url)

                if (
                    url_part.startswith("#")
                    or is_non_fetchable(url_part)
                    or not is_httpish(url_part)
                ):
                    new_entries.append(entry)
                    continue

                abs_url = normalize_url(canonicalize_url(url_part, page_url))
                parsed = urlparse(abs_url)

                is_ext = not is_internal(abs_url, root_netloc)
                if is_ext:
                    if not download_external_assets:
                        new_entries.append(entry)
                        continue

                    if not is_allowed_external(abs_url, external_domains):
                        new_entries.append(entry)
                        continue

                    local_path = cdn_local_path(parsed, site_root)
                else:
                    local_path = to_local_asset_path(parsed, site_root)

                rel = _rel_url(local_path, page_dir)
                if parsed.fragment:
                    rel = f"{rel}#{parsed.fragment}"

                parts[0] = rel
                new_entries.append(" ".join(parts))

            tag["srcset"] = ", ".join(new_entries)

        # Inline style="background:url(...)" rewriting
        if tag.has_attr("style"):
            style = str(tag["style"])

            def repl_style(m: re.Match) -> str:
                raw = m.group(1).strip()
                quote = ""
                url_part = raw

                if len(raw) >= 2 and raw[0] in ("'", '"') and raw[-1] == raw[0]:
                    quote = raw[0]
                    url_part = raw[1:-1].strip()

                if (
                    not url_part
                    or url_part.startswith("#")
                    or url_part.startswith(("data:", "javascript:", "about:"))
                ):
                    return m.group(0)

                url_part2 = _protocol_fix(url_part, page_url)
                if is_non_fetchable(url_part2) or not is_httpish(url_part2):
                    return m.group(0)

                abs_url = canonicalize_url(url_part2, page_url)
                parsed = urlparse(abs_url)

                # Only rewrite things that look like assets.
                if not parsed.path.lower().endswith(ASSET_EXTENSIONS):
                    return m.group(0)

                is_ext = not is_internal(abs_url, root_netloc)
                if is_ext:
                    if not download_external_assets:
                        return m.group(0)

                    if not is_allowed_external(abs_url, external_domains):
                        return m.group(0)

                    local_path = cdn_local_path(parsed, site_root)
                else:
                    local_path = to_local_asset_path(parsed, site_root)

                rel = _rel_url(local_path, page_dir)
                if parsed.fragment:
                    rel = f"{rel}#{parsed.fragment}"

                if quote:
                    return f"url({quote}{rel}{quote})"
                return f"url({rel})"

            style = CSS_URL_RE.sub(repl_style, style)
            tag["style"] = style

    # Rewrite Next.js __NEXT_DATA__ first. It is JSON, not executable JS, so a
    # structured walk is more reliable than regexing the raw script text.
    for script_tag in soup.find_all("script", {"id": "__NEXT_DATA__"}):
        if script_tag.has_attr("src"):
            continue
        try:
            raw_json = script_tag.string or script_tag.get_text()
            if not raw_json:
                continue
            rewritten_json, rewrite_count = rewrite_next_data_json_text(
                raw_json,
                page_url,
                site_root=site_root,
                root_netloc=root_netloc,
                base_dir=page_dir,
                download_external_assets=download_external_assets,
                external_domains=external_domains,
                download_q=download_q,
            )
            if rewrite_count:
                script_tag.string = rewritten_json
                _log(
                    "next_data_assets_rewritten",
                    url=page_url,
                    count=rewrite_count,
                )
        except Exception as exc:  # noqa: BLE001
            _log("next_data_rewrite_fail", url=page_url, error=exc)

    # Rewrite inline <script> blocks too. This is especially important for
    # app scripts that contain asset paths outside __NEXT_DATA__.
    for script_tag in soup.find_all("script"):
        if script_tag.has_attr("src"):
            continue
        if script_tag.get("id") == "__NEXT_DATA__":
            continue
        try:
            js_text = script_tag.string or script_tag.get_text()
            if not js_text:
                continue
            rewritten = rewrite_js_text(
                js_text,
                page_url,
                site_root=site_root,
                root_netloc=root_netloc,
                base_dir=page_dir,
                download_external_assets=download_external_assets,
                external_domains=external_domains,
                download_q=download_q,
            )
            if rewritten != js_text:
                script_tag.string = rewritten
        except Exception as exc:  # noqa: BLE001
            _log("script_rewrite_fail", url=page_url, error=exc)

    # Rewrite <style> blocks too (internal assets only; CDN kept unchanged here)
    for style_tag in soup.find_all("style"):
        try:
            css_text = style_tag.string or style_tag.get_text()
            if not css_text:
                continue
            rewritten = rewrite_css_text(
                css_text,
                page_url,
                site_root=site_root,
                root_netloc=root_netloc,
                base_dir=page_dir,
                download_external_assets=download_external_assets,
                external_domains=external_domains,
                download_q=None,
            )
            if rewritten != css_text:
                style_tag.string = rewritten
        except Exception as exc:  # noqa: BLE001
            _log("style_rewrite_fail", url=page_url, error=exc)


# ---------------------------------------------------------------------------
# Crawl coordinator
# ---------------------------------------------------------------------------


def extract_css_assets(css_text: str) -> list[str]:
    """
    Extract asset URLs from CSS url(...) and @import patterns.

    This is used when scanning <style> blocks during HTML parse time
    (before the CSS is written to disk).
    """
    results: list[str] = []

    for match in CSS_URL_RE.findall(css_text):
        url = match.strip().strip("'\"")
        if not url or url.startswith(("data:", "javascript:", "about:", "#")):
            continue
        results.append(url)

    for match in CSS_IMPORT_RE.findall(css_text):
        url = match.strip().strip("'\"")
        if not url or url.startswith(("data:", "javascript:", "about:", "#")):
            continue
        results.append(url)

    return results


def _slug_parts(value: object) -> list[str]:
    """
    Normalize a slug value into clean path segments.

    Supports:
    - "two-pointers"
    - "two-pointers/next-lexicographical-sequence"
    - ["two-pointers", "next-lexicographical-sequence"]
    """
    parts: list[str] = []

    def _append_one(raw: object) -> None:
        if not isinstance(raw, str):
            return
        text = raw.strip().strip("/")
        if not text:
            return
        for piece in text.split("/"):
            piece = piece.strip()
            if not piece or "[" in piece or "]" in piece:
                continue
            parts.append(piece)

    if isinstance(value, str):
        _append_one(value)
    elif isinstance(value, list):
        for item in value:
            _append_one(item)

    return parts


def _extract_root_relative_paths_from_json(
    node: object,
    *,
    course_path: Optional[str] = None,
) -> list[str]:
    """
    Extract candidate internal page paths from a JSON tree.

    This handles both:
    - direct root-relative strings like "/courses/x/y"
    - structured Next.js route objects like:
      {"course": "coding-patterns", "slug": ["two-pointers", "next-lexicographical-sequence"]}

    The structured form matters for deeper hierarchies because some APIs expose
    path segments as arrays instead of already-joined URLs.
    """
    results: list[str] = []
    seen: set[str] = set()

    def _add_path(raw: object) -> None:
        if not isinstance(raw, str):
            return
        path = raw.strip()
        if not path.startswith("/"):
            return
        if "[" in path or "]" in path:
            return
        if "\\" in path:
            return
        if len(path) > 500:
            return
        if path.startswith(("/_next", "/__nextjs", "/_error", "/api/")):
            return
        if Path(path.split("?")[0]).suffix.lower() in ASSET_EXTENSIONS:
            return
        if path not in seen:
            seen.add(path)
            results.append(path)

    def _recurse(current: object) -> None:
        if isinstance(current, str):
            _add_path(current)
            return

        if isinstance(current, list):
            for item in current:
                _recurse(item)
            return

        if not isinstance(current, dict):
            return

        root_path = current.get("rootPath") or current.get("root_path")
        course = current.get("course")
        slug_parts = _slug_parts(current.get("slug"))
        query = current.get("query")
        default_chapter = current.get("defaultChapter") or current.get("default_chapter")

        if isinstance(default_chapter, str):
            _add_path(default_chapter)

        if isinstance(root_path, str) and root_path.startswith("/") and slug_parts:
            _add_path(root_path.rstrip("/") + "/" + "/".join(slug_parts))

        if isinstance(course, str) and slug_parts:
            _add_path(f"/courses/{course.strip('/')}/" + "/".join(slug_parts))

        if course_path and slug_parts:
            _add_path(course_path.rstrip("/") + "/" + "/".join(slug_parts))

        if isinstance(query, dict):
            query_course = query.get("course")
            query_slug_parts = _slug_parts(query.get("slug"))
            if isinstance(query_course, str) and query_slug_parts:
                _add_path(f"/courses/{query_course.strip('/')}/" + "/".join(query_slug_parts))
            elif course_path and query_slug_parts:
                _add_path(course_path.rstrip("/") + "/" + "/".join(query_slug_parts))

        for value in current.values():
            _recurse(value)

    _recurse(node)
    return results


def _extract_next_data_urls(
    soup: BeautifulSoup,
    base_url: str,
    prefix: Optional[str],
    root_netloc: str,
) -> list[str]:
    """
    Extract internal page URLs from the Next.js ``__NEXT_DATA__`` JSON block.

    Next.js always embeds a ``<script id="__NEXT_DATA__" type="application/json">``
    tag containing the full server-side props for the page, including navigation
    data such as the complete chapter list.  This data is present even when the
    corresponding sidebar items are inside a *collapsed* accordion — i.e. they
    produce no ``<a>`` tags visible to a normal DOM walk.

    Strategy:
    - Parse the JSON and re-serialise it as a flat string.
    - Find every string value that looks like an internal relative path
      (starts with ``/``, no scheme, same netloc when resolved).
    - Apply the same ``prefix`` filter used for ``<a>`` link enqueuing.

    Returns a de-duplicated list of canonical absolute URLs.
    """
    import json as _json

    tag = soup.find("script", {"id": "__NEXT_DATA__"})
    if tag is None:
        return []

    raw_json = tag.string or ""
    if not raw_json.strip():
        return []

    try:
        data = _json.loads(raw_json)
    except Exception:  # noqa: BLE001
        return []

    parsed_base = urlparse(base_url)
    base_scheme_host = f"{parsed_base.scheme}://{parsed_base.netloc}"
    course_path = parsed_base.path.rstrip("/") or None
    collected = _extract_root_relative_paths_from_json(data, course_path=course_path)

    seen: set[str] = set()
    results: list[str] = []

    for raw in collected:
        abs_url = normalize_url(canonicalize_url(f"{base_scheme_host}{raw}", base_url))
        p = urlparse(abs_url)

        # Must be same domain
        if _canonical_netloc(p) != root_netloc:
            continue

        # Apply prefix filter
        if prefix and not p.path.startswith(prefix):
            continue

        if abs_url not in seen:
            seen.add(abs_url)
            results.append(abs_url)

    if results:
        _log("next_data_urls_found", count=len(results), page=base_url)

    return results


def _discover_course_chapters(
    course_url: str,
    pw_context: "BrowserContext",
    url_prefix: Optional[str],
    root_netloc: str,
    wait_for: str = "body",
    render_settle_ms: int = 4000,
) -> list[str]:
    """
    Auto-discover chapter URLs from a course index page using Playwright.

    Loads the course root URL in a Playwright page, intercepts JSON responses,
    and merges chapter candidates from multiple sources:
    1. JSON API responses
    2. DOM <a href> links
    3. DOM data-menu-id attributes used by JS-only sidebars
    4. __NEXT_DATA__ mining
    """
    parsed_course = urlparse(course_url)
    course_path = parsed_course.path.rstrip("/")
    base_origin = f"{parsed_course.scheme}://{parsed_course.netloc}"
    chapter_prefix = course_path + "/"

    _prefix: Optional[str] = None
    if url_prefix and url_prefix.strip():
        _prefix = "/" + url_prefix.strip().strip("/")

    def _matches_filters(abs_url: str) -> bool:
        p = urlparse(abs_url)
        if _canonical_netloc(p) != root_netloc:
            return False
        if _prefix and not p.path.startswith(_prefix):
            return False
        return True

    def _to_abs(raw_path: str) -> str:
        return normalize_url(canonicalize_url(f"{base_origin}{raw_path}", course_url))

    def _add_url(raw_or_abs: str, out: list[str]) -> bool:
        raw = str(raw_or_abs).strip()
        if not raw:
            return False
        if raw.startswith("http"):
            abs_url = normalize_url(canonicalize_url(raw, course_url))
        elif raw.startswith("/"):
            abs_url = _to_abs(raw)
        else:
            return False
        if not urlparse(abs_url).path.startswith(chapter_prefix):
            return False
        if not _matches_filters(abs_url):
            return False
        if abs_url in out:
            return False
        out.append(abs_url)
        return True

    _log("chapter_discovery_start", url=course_url)

    api_found_paths: list[str] = []
    api_seen: set[str] = set()

    page = pw_context.new_page()
    try:
        def on_response(resp) -> None:  # noqa: ANN001
            try:
                ct = resp.headers.get("content-type", "")
                if "json" not in ct:
                    return
                data = resp.json()
                before = len(api_found_paths)
                for candidate in _extract_root_relative_paths_from_json(
                    data,
                    course_path=course_path,
                ):
                    if (
                        candidate.startswith(chapter_prefix)
                        and len(candidate) > len(chapter_prefix)
                        and candidate not in api_seen
                    ):
                        api_seen.add(candidate)
                        api_found_paths.append(candidate)
                after = len(api_found_paths)
                if after > before:
                    _log(
                        "chapter_discovery_api_hit",
                        api_url=resp.url,
                        new_paths=after - before,
                    )
            except Exception:
                pass

        page.on("response", on_response)

        try:
            page.goto(course_url, wait_until="networkidle", timeout=30_000)
        except Exception as exc:
            _log("chapter_discovery_nav_error", error=str(exc))

        try:
            page.wait_for_selector(wait_for, timeout=15_000)
        except Exception:
            pass

        if render_settle_ms > 0:
            page.wait_for_timeout(render_settle_ms)

        chapter_urls: list[str] = []
        source_counts = {"api": 0, "dom": 0, "menu": 0, "next_data": 0}

        for raw in api_found_paths:
            if _add_url(raw, chapter_urls):
                source_counts["api"] += 1
        if source_counts["api"]:
            _log("chapter_discovery_api_result", count=source_counts["api"])

        html = page.content()
        soup = BeautifulSoup(html, "html.parser")

        for a_tag in soup.find_all("a", href=True):
            if _add_url(str(a_tag["href"]).strip(), chapter_urls):
                source_counts["dom"] += 1
        if source_counts["dom"]:
            _log("chapter_discovery_dom_result", count=source_counts["dom"])

        for tag in soup.find_all(attrs={"data-menu-id": True}):
            raw_menu_id = str(tag.get("data-menu-id", "")).strip()
            if not raw_menu_id:
                continue
            marker = raw_menu_id.rfind("/courses/")
            if marker == -1:
                continue
            if _add_url(raw_menu_id[marker:], chapter_urls):
                source_counts["menu"] += 1
        if source_counts["menu"]:
            _log("chapter_discovery_menu_result", count=source_counts["menu"])

        for u in _extract_next_data_urls(soup, course_url, _prefix, root_netloc):
            if _add_url(u, chapter_urls):
                source_counts["next_data"] += 1
        if source_counts["next_data"]:
            _log("chapter_discovery_next_data_result", count=source_counts["next_data"])

        if chapter_urls:
            _log(
                "chapter_discovery_complete",
                count=len(chapter_urls),
                sources=source_counts,
                sample=chapter_urls[:5],
            )
        else:
            _log(
                "chapter_discovery_failed",
                url=course_url,
                hint="No chapter URLs found automatically. Set SEED_URLS manually.",
            )

        return chapter_urls

    finally:
        page.close()

def crawl_site(
    start_url: str,
    root: Path,
    max_pages: int,
    threads: int,
    download_external_assets: bool = False,
    external_domains: Optional[set[str]] = None,
    fetch_fn: Optional[Callable[[str], Optional[BeautifulSoup]]] = None,
    url_prefix: Optional[str] = None,
    seed_urls: Optional[list[str]] = None,
    remove_js: bool = False,
    auth_fail_text: Optional[str] = None,
    follow_links: bool = True,
    strip_selectors: Optional[list[str]] = None,
) -> None:
    """
    Breadth-first crawl limited to max_pages.

    - q_pages: pages to crawl (HTML only, internal-only)
    - download_q: assets to download (internal, and optionally external)
    - worker threads: process download_q and write to disk
    - url_prefix: if set, only crawl pages whose path starts with this prefix.
      Example: "/courses/" restricts crawl to course pages only.
    - seed_urls: extra URLs to pre-populate the queue before crawling.
      Use when a site does not expose chapter links as <a> tags (e.g.
      ByteByteGo sidebar), so all chapter URLs must be listed explicitly.
      Set via SEED_URLS (newline or space-separated) in .env.
    - remove_js: if true, strip all <script> and JS preload tags from the
      downloaded HTML. Prevents JS hydration from breaking the page.
    - follow_links: if false, do not discover/enqueue new HTML pages from <a>.
      Useful for a simple "download exactly the URLs I provide" workflow.
    """
    q_pages: queue.Queue[str] = queue.Queue()
    q_pages.put(start_url)

    # Pre-populate queue with any explicitly provided seed URLs.
    _extra: list[str] = []
    for s in (seed_urls or []):
        u = normalize_url(canonicalize_url(s))
        if u and u not in {start_url}:
            _extra.append(u)
    for u in _extra:
        q_pages.put(u)

    seen_pages: set[str] = set()
    queued_pages: set[str] = {start_url} | set(_extra)

    # queued_assets ensures we don't enqueue the same asset URL many times.
    queued_assets: set[str] = set()

    # download_q holds (abs_url, destination_path) pairs.
    download_q: queue.Queue[tuple[str, Path]] = queue.Queue()

    root_netloc = _canonical_netloc(urlparse(start_url))

    # Normalise url_prefix: ensure it starts with '/' and has no trailing slash
    # so we can do a simple startswith() check on parsed.path.
    _prefix: Optional[str] = None
    if url_prefix and url_prefix.strip():
        _prefix = "/" + url_prefix.strip().strip("/")

    def worker() -> None:
        """Download worker thread: pulls tasks from download_q and writes them."""
        while True:
            url, dest = download_q.get()
            try:
                if is_non_fetchable(url) or not is_httpish(url):
                    continue
                fetch_binary(
                    url,
                    dest,
                    download_q,
                    site_root=root,
                    root_netloc=root_netloc,
                    download_external_assets=download_external_assets,
                    external_domains=external_domains,
                )
            finally:
                download_q.task_done()

    # Spawn the asset download workers.
    for i in range(max(1, threads)):
        t = threading.Thread(target=worker, name=f"DL-{i + 1}", daemon=True)
        t.start()

    _fetch = fetch_fn if fetch_fn is not None else fetch_html
    _log(
        "crawl_start",
        url=start_url,
        max_pages=max_pages if max_pages < sys.maxsize else "unlimited",
        threads=threads,
        follow_links=follow_links,
        seed_count=len(_extra),
    )

    start_time = time.time()
    PAGE_SUFFIXES = {"", ".html", ".htm"}

    while not q_pages.empty() and len(seen_pages) < max_pages:
        page_url = canonicalize_url(q_pages.get())
        if page_url in seen_pages:
            continue

        seen_pages.add(page_url)
        _log("page_crawl", index=len(seen_pages), total=max_pages if max_pages < sys.maxsize else "unlimited", url=page_url)

        soup = _fetch(page_url)
        if soup is None:
            continue

        # ── Strip paywall overlays / cookie banners ──────────────────────
        # Some SPAs (e.g. ByteByteGo) SSR the full content but render a
        # CSS overlay + "Unlock" buttons client-side when Firebase auth
        # state is missing from indexedDB.  The content is already in the
        # DOM — just remove the overlay elements so the page is clean.
        if strip_selectors:
            _stripped = 0
            for sel in strip_selectors:
                for el in soup.select(sel):
                    el.decompose()
                    _stripped += 1
            if _stripped:
                _log("strip_selectors_applied", url=page_url, removed=_stripped,
                     selectors=strip_selectors)
            if inject_runtime_hide_selectors(soup, strip_selectors):
                _log("runtime_hide_selectors_injected", url=page_url, selectors=strip_selectors)

        if auth_fail_text and auth_fail_text in soup.get_text():
            # Dump the HTML Playwright received so the user can diagnose the issue
            debug_path = root / "_debug_auth_fail.html"
            create_dir(root)
            debug_path.write_text(str(soup), encoding="utf-8")
            page_text = soup.get_text(" ", strip=True)
            lock_icon_count = len(
                soup.find_all(
                    "img",
                    src=lambda v: isinstance(v, str) and "lock" in v.lower(),
                )
            )
            _log(
                "auth_failed_fatal",
                url=page_url,
                reason=f"Found '{auth_fail_text}'",
                debug_html=str(debug_path),
                login_marker=(" login " in f" {page_text.lower()} "),
                unlock_marker=(auth_fail_text.lower() in page_text.lower()),
                lock_icons=lock_icon_count,
            )
            sys.exit(1)

        # ── Next.js __NEXT_DATA__ discovery ───────────────────────────────────
        # Many Next.js sites (incl. ByteByteGo) hide chapter links inside
        # collapsed accordion sections.  Those links never appear as <a> tags
        # in the rendered DOM.  The __NEXT_DATA__ JSON block always contains
        # the full navigation structure, so we mine it here for additional URLs.
        if follow_links:
            for next_url in _extract_next_data_urls(soup, page_url, _prefix, root_netloc):
                if next_url not in seen_pages and next_url not in queued_pages:
                    q_pages.put(next_url)
                    queued_pages.add(next_url)

        # Walk the DOM once and:
        # 1) enqueue internal pages from <a href=...>
        # 2) enqueue assets referenced via src/href/data-src/poster/srcset/style/<style>
        for tag in soup.find_all(True):

            # Common URL-bearing attributes
            for attr in ("src", "href", "data-src", "poster"):
                if not tag.has_attr(attr):
                    continue

                if tag.name == "link" and attr == "href":
                    rel = tag.get("rel", [])
                    if isinstance(rel, str):
                        rel = [rel]
                    rel_set = {str(r).lower() for r in rel if str(r).strip()}
                    if not rel_set & RESOURCE_LINK_RELS:
                        continue

                link_raw = str(tag.get(attr, "")).strip()
                if not link_raw:
                    continue

                link = _protocol_fix(link_raw, page_url)
                if (
                    link.startswith("#")
                    or is_non_fetchable(link)
                    or not is_httpish(link)
                ):
                    continue

                abs_url = normalize_url(canonicalize_url(link, page_url))
                parsed = urlparse(abs_url)
                is_ext = not is_internal(abs_url, root_netloc)

                # Only crawl internal HTML pages from <a href=...>
                suffix = Path(parsed.path).suffix.lower()
                is_page = (
                    tag.name == "a"
                    and not is_ext
                    and (parsed.path.endswith("/") or suffix in PAGE_SUFFIXES)
                )

                if is_page:
                    if not follow_links:
                        continue
                    # URL_PREFIX filter: skip pages outside the allowed path subtree.
                    if _prefix and not parsed.path.startswith(_prefix):
                        continue
                    if abs_url not in seen_pages and abs_url not in queued_pages:
                        q_pages.put(abs_url)
                        queued_pages.add(abs_url)
                    continue

                # Otherwise treat it as an asset candidate.
                if is_ext:
                    if not download_external_assets:
                        continue

                    if not is_allowed_external(abs_url, external_domains):
                        continue

                    # External assets without extensions are only allowed for <script> and <link>
                    # because CDNs sometimes serve JS/CSS without filename extensions.
                    if tag.name not in (
                        "script",
                        "link",
                    ) and not parsed.path.lower().endswith(ASSET_EXTENSIONS):
                        continue

                    dest_path = cdn_local_path(parsed, root)
                else:
                    dest_path = to_local_asset_path(parsed, root)

                if abs_url not in queued_assets:
                    queued_assets.add(abs_url)
                    create_dir(dest_path.parent)
                    download_q.put((abs_url, dest_path))

            # ------------------------------------------------------------------
            # META IMAGE SUPPORT (og:image, twitter:image)
            # ------------------------------------------------------------------
            if tag.name == "meta":
                content = str(tag.get("content", "")).strip()
                prop = (tag.get("property") or tag.get("name") or "").lower()

                if content and ("og:image" in prop or "twitter:image" in prop):
                    url_part = _protocol_fix(content, page_url)

                    if (
                        not url_part
                        or url_part.startswith("#")
                        or url_part.startswith(("data:", "javascript:", "about:"))
                        or is_non_fetchable(url_part)
                        or not is_httpish(url_part)
                    ):
                        continue
                    else:
                        abs_url = normalize_url(canonicalize_url(url_part, page_url))
                        parsed = urlparse(abs_url)

                        if parsed.path.lower().endswith(ASSET_EXTENSIONS):
                            is_ext = not is_internal(abs_url, root_netloc)

                            if is_ext:
                                if not download_external_assets:
                                    continue
                                elif not is_allowed_external(abs_url, external_domains):
                                    continue
                                else:
                                    dest_path = cdn_local_path(parsed, root)

                                    if abs_url not in queued_assets:
                                        queued_assets.add(abs_url)
                                        create_dir(dest_path.parent)
                                        download_q.put((abs_url, dest_path))
                            else:
                                dest_path = to_local_asset_path(parsed, root)

                                if abs_url not in queued_assets:
                                    queued_assets.add(abs_url)
                                    create_dir(dest_path.parent)
                                    download_q.put((abs_url, dest_path))

            # srcset handling (images at multiple resolutions)
            if tag.has_attr("srcset"):
                for entry in str(tag["srcset"]).split(","):
                    entry = entry.strip()
                    if not entry:
                        continue

                    url_part = _protocol_fix(entry.split()[0], page_url)
                    if (
                        url_part.startswith("#")
                        or is_non_fetchable(url_part)
                        or not is_httpish(url_part)
                    ):
                        continue

                    abs_url = normalize_url(canonicalize_url(url_part, page_url))
                    parsed = urlparse(abs_url)
                    is_ext = not is_internal(abs_url, root_netloc)

                    if is_ext:
                        if not download_external_assets:
                            continue

                        if not is_allowed_external(abs_url, external_domains):
                            continue

                        if not parsed.path.lower().endswith(ASSET_EXTENSIONS):
                            continue

                        dest_path = cdn_local_path(parsed, root)
                    else:
                        dest_path = to_local_asset_path(parsed, root)

                    if abs_url not in queued_assets:
                        queued_assets.add(abs_url)
                        create_dir(dest_path.parent)
                        download_q.put((abs_url, dest_path))

            # inline style="...url(...)..." assets
            if tag.has_attr("style"):
                style = str(tag["style"])
                for match in CSS_URL_RE.findall(style):
                    url_part = _protocol_fix(match.strip().strip("'\""), page_url)
                    if (
                        not url_part
                        or url_part.startswith("#")
                        or url_part.startswith(("data:", "javascript:", "about:"))
                        or is_non_fetchable(url_part)
                        or not is_httpish(url_part)
                    ):
                        continue

                    abs_url = normalize_url(canonicalize_url(url_part, page_url))
                    parsed = urlparse(abs_url)

                    if not parsed.path.lower().endswith(ASSET_EXTENSIONS):
                        continue

                    is_ext = not is_internal(abs_url, root_netloc)

                    if is_ext:
                        if not download_external_assets:
                            continue

                        if not is_allowed_external(abs_url, external_domains):
                            continue

                    dest_path = (
                        cdn_local_path(parsed, root)
                        if is_ext
                        else to_local_asset_path(parsed, root)
                    )
                    if abs_url not in queued_assets:
                        queued_assets.add(abs_url)
                        create_dir(dest_path.parent)
                        download_q.put((abs_url, dest_path))

            # <style> blocks: extract CSS asset references and enqueue them
            if tag.name == "style":
                css_text = tag.string or tag.get_text()
                if not css_text:
                    continue

                for asset in extract_css_assets(css_text):
                    asset = _protocol_fix(asset, page_url)
                    if (
                        not asset
                        or asset.startswith("#")
                        or asset.startswith(("data:", "javascript:", "about:"))
                        or is_non_fetchable(asset)
                        or not is_httpish(asset)
                    ):
                        continue

                    abs_url = canonicalize_url(asset, page_url)
                    parsed = urlparse(abs_url)

                    if not parsed.path.lower().endswith(ASSET_EXTENSIONS):
                        continue

                    is_ext = not is_internal(abs_url, root_netloc)

                    if is_ext:
                        if not download_external_assets:
                            continue

                        if not is_allowed_external(abs_url, external_domains):
                            continue

                    dest_path = (
                        cdn_local_path(parsed, root)
                        if is_ext
                        else to_local_asset_path(parsed, root)
                    )
                    if abs_url not in queued_assets:
                        queued_assets.add(abs_url)
                        create_dir(dest_path.parent)
                        download_q.put((abs_url, dest_path))

        # Save current page:
        # - determine local filename
        # - rewrite links inside the HTML
        # - write out the HTML
        if remove_js:
            for s_tag in soup.find_all("script"):
                s_tag.decompose()
            for l_tag in soup.find_all("link", attrs={"rel": "preload"}):
                if l_tag.get("as") == "script" or (l_tag.get("href") and l_tag["href"].endswith(".js")):
                    l_tag.decompose()
            for l_tag in soup.find_all("link", attrs={"rel": "modulepreload"}):
                l_tag.decompose()

        local_path = to_local_path(urlparse(page_url), root)
        create_dir(local_path.parent)
        if not remove_js and inject_runtime_asset_fixups(soup, root, local_path.parent):
            _log("runtime_asset_fixups_injected", url=page_url)
        rewrite_links(
            soup,
            page_url,
            root,
            local_path.parent,
            download_external_assets,
            external_domains,
            download_q,
        )
        safe_write_text(local_path, str(soup), encoding="utf-8")

    # Wait for all queued asset downloads to finish
    download_q.join()

    elapsed = time.time() - start_time
    if seen_pages:
        _log(
            "crawl_complete",
            pages=len(seen_pages),
            elapsed_s=round(elapsed, 2),
            avg_s=round(elapsed / len(seen_pages), 2),
        )
    else:
        _log("crawl_warn_empty", hint="check URL, credentials, or connectivity")


# ---------------------------------------------------------------------------
# Helper function for output folder
# ---------------------------------------------------------------------------


def make_root(url: str, custom: Optional[str]) -> Path:
    """
    Derive output folder from URL if custom not supplied.

    Example:
      https://example.com -> example_com
    """
    if custom:
        expanded = os.path.expandvars(os.path.expanduser(custom))
        return Path(expanded)
    return Path(urlparse(url).netloc.replace(".", "_"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments, with .env file as the default source for every value.

    Priority (highest → lowest):
      1. CLI flags (e.g. --url https://...)
      2. .env file variables (loaded via python-dotenv)
      3. Hard-coded defaults below

    All config can live entirely in .env so the script runs with just:
      uv run python web_downloader.py

    .env variables:
      URL                       Starting URL to crawl
      START_URLS                Extra starting URL(s), JSON/comma/newline/space separated
      COOKIE                    Full Cookie header string from browser DevTools
      HEADER_<NAME>             One header per var, e.g. HEADER_AUTHORIZATION=Bearer eyJ...
      PLAYWRIGHT                true/1/yes to enable headless Chromium rendering
      PLAYWRIGHT_STORAGE_STATE  Path to Playwright storage_state JSON (cookies + localStorage)
      WAIT_FOR                  CSS selector to wait for before extracting HTML
      RENDER_SETTLE_MS          Extra wait after selector (helps SPA auth state settle)
      AUTO_AUTH_HEADER_FROM_COOKIE true/false (derive Bearer from token cookie)
      MAX_PAGES                 Max pages (unset = unlimited)
      THREADS                   Concurrent workers (default 1)
      DESTINATION               Output folder
      DOWNLOAD_EXTERNAL_ASSETS  true/1/yes to download CDN assets
      EXTERNAL_DOMAINS          JSON/comma/newline/space-separated CDN domains
      USER_AGENT                Browser UA string (recommended with cf_clearance)
      FOLLOW_LINKS              true/false to discover extra pages from links
      AUTH_DEBUG                true/false for detailed auth/network diagnostics
    """
    from dotenv import load_dotenv
    load_dotenv()

    # ── Read env vars ────────────────────────────────────────────────────────
    env_url = os.getenv("URL", "")
    _start_raw = os.getenv("START_URLS", "")
    env_start_urls = parse_env_token_list(_start_raw)
    env_cookie = os.getenv("COOKIE")
    env_playwright = os.getenv("PLAYWRIGHT", "").lower() in ("1", "true", "yes")
    _page_fetch_raw = os.getenv("PLAYWRIGHT_PAGE_FETCH", "")
    if _page_fetch_raw.strip():
        env_playwright_page_fetch = _page_fetch_raw.lower() in ("1", "true", "yes")
    else:
        env_playwright_page_fetch = env_playwright
    env_playwright_storage_state = os.getenv("PLAYWRIGHT_STORAGE_STATE") or None
    env_wait_for = os.getenv("WAIT_FOR", "body")
    _settle = os.getenv("RENDER_SETTLE_MS", "4000")
    env_render_settle_ms = int(_settle) if _settle.strip().isdigit() else 4000
    env_auto_auth_header = os.getenv("AUTO_AUTH_HEADER_FROM_COOKIE", "true").lower() in ("1", "true", "yes")
    env_user_agent = os.getenv("USER_AGENT") or None
    env_follow_links = os.getenv("FOLLOW_LINKS", "true").lower() in ("1", "true", "yes")
    env_auth_debug = os.getenv("AUTH_DEBUG", "false").lower() in ("1", "true", "yes")
    _strip_raw = os.getenv("STRIP_SELECTORS", "")
    env_strip_selectors = parse_env_csv_list(_strip_raw)
    env_destination = os.getenv("DESTINATION") or None
    env_discover_chapters = os.getenv("DISCOVER_CHAPTERS", "false").lower() in ("1", "true", "yes")

    _max = os.getenv("MAX_PAGES", "")
    env_max_pages = int(_max) if _max.strip().isdigit() and int(_max) > 0 else sys.maxsize

    _threads = os.getenv("THREADS", "1")
    env_threads = int(_threads) if _threads.strip().isdigit() else 1

    env_download_external = os.getenv("DOWNLOAD_EXTERNAL_ASSETS", "").lower() in ("1", "true", "yes")

    _ext = parse_env_token_list(os.getenv("EXTERNAL_DOMAINS", ""))
    env_external_domains = _ext if _ext else None

    env_url_prefix = os.getenv("URL_PREFIX") or None

    # SEED_URLS: JSON, comma, newline, or space-separated list of extra URLs.
    _seed_raw = os.getenv("SEED_URLS", "")
    env_seed_urls = parse_env_token_list(_seed_raw)

    env_remove_js = str(os.getenv("REMOVE_JS", "")).strip().lower() == "true"
    env_auth_fail_text = os.getenv("AUTH_FAIL_TEXT") or None

    # HEADER_AUTHORIZATION=Bearer xyz  →  "Authorization: Bearer xyz"
    # HEADER_X_API_KEY=abc             →  "X-Api-Key: abc"
    env_headers: list[str] = []
    for k, v in os.environ.items():
        if k.startswith("HEADER_") and v:
            header_name = k[7:].replace("_", "-").title()
            env_headers.append(f"{header_name}: {v}")

    # ── Build parser ─────────────────────────────────────────────────────────
    p = argparse.ArgumentParser(
        description=(
            "Recursively mirror a website for offline use. "
            "All options can be set in a .env file — see .env.example."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--url",
        default=env_url,
        required=not bool(env_url or env_start_urls),
        help="Starting URL to crawl. Can be set via URL= in .env.",
    )
    p.add_argument(
        "--start-url",
        action="append",
        dest="start_urls",
        default=env_start_urls,
        metavar="URL",
        help=(
            "Additional starting URL(s) to process (repeatable). "
            "Useful for chapter-only downloads. Can be set via START_URLS= in .env "
            "(JSON array, comma, newline, or space-separated)."
        ),
    )
    p.add_argument(
        "--destination",
        default=env_destination,
        help="Output folder (default: derived from URL hostname).",
    )
    p.add_argument(
        "--max-pages",
        type=int,
        default=env_max_pages,
        help="Maximum HTML pages to crawl (default: unlimited).",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=env_threads,
        help="Concurrent download workers (default: 1 to avoid rate limits).",
    )
    p.add_argument(
        "--download-external-assets",
        action="store_true",
        default=env_download_external,
        help="Download external CDN/static assets and rewrite links for offline use.",
    )
    p.add_argument(
        "--external-domains",
        nargs="+",
        default=env_external_domains,
        help="Whitelist of external domains to download from (implies external download).",
    )
    p.add_argument(
        "--cookie",
        default=env_cookie,
        metavar="COOKIE_STRING",
        help='Cookie header value from browser DevTools. Can be set via COOKIE= in .env.',
    )
    p.add_argument(
        "--header",
        action="append",
        default=env_headers,
        metavar="KEY:VALUE",
        help='Extra request header (repeatable). Set via HEADER_<NAME>= in .env.',
    )
    p.add_argument(
        "--playwright",
        action=argparse.BooleanOptionalAction,
        default=env_playwright,
        help=(
            "Use headless Chromium to render JS before saving HTML (needed for SPAs). "
            "Can be set via PLAYWRIGHT=true in .env. "
            "Install: uv add playwright && uv run playwright install chromium"
        ),
    )
    p.add_argument(
        "--playwright-page-fetch",
        action=argparse.BooleanOptionalAction,
        default=env_playwright_page_fetch,
        help=(
            "Use Playwright for the final page HTML fetch. Disable this to use "
            "plain requests for page downloads while still using Playwright for "
            "chapter discovery. Can be set via PLAYWRIGHT_PAGE_FETCH= in .env."
        ),
    )
    p.add_argument(
        "--playwright-storage-state",
        default=env_playwright_storage_state,
        metavar="JSON_PATH",
        help=(
            "Path to Playwright storage_state JSON exported from a logged-in browser session. "
            "Includes cookies + localStorage and is often more reliable than manual COOKIE= alone. "
            "Can be set via PLAYWRIGHT_STORAGE_STATE= in .env."
        ),
    )
    p.add_argument(
        "--wait-for",
        default=env_wait_for,
        metavar="CSS_SELECTOR",
        help="CSS selector Playwright waits for before extracting HTML. Set via WAIT_FOR= in .env.",
    )
    p.add_argument(
        "--render-settle-ms",
        type=int,
        default=env_render_settle_ms,
        metavar="MS",
        help=(
            "Extra time (ms) to wait after WAIT_FOR before extracting HTML. "
            "Helps SPAs finish auth/state hydration. Set via RENDER_SETTLE_MS= in .env."
        ),
    )
    p.add_argument(
        "--url-prefix",
        default=env_url_prefix,
        metavar="PATH_PREFIX",
        help=(
            "Only crawl pages whose path starts with this prefix (e.g. /courses/). "
            "Prevents following navbar links to unrelated pages. "
            "Can be set via URL_PREFIX= in .env."
        ),
    )
    p.add_argument(
        "--auth-fail-text",
        default=env_auth_fail_text,
        metavar="TEXT",
        help="Abort download if this exact text is found on any page. Used to detect paywalls. Can be set via AUTH_FAIL_TEXT= in .env."
    )
    p.add_argument(
        "--remove-js",
        action=argparse.BooleanOptionalAction,
        default=env_remove_js,
        help=(
            "Remove all <script> and JS preload tags from the saved HTML. "
            "Highly recommended for SPAs like Next.js so React hydration doesn't break the offline page. "
            "Can be set via REMOVE_JS=true in .env."
        ),
    )
    p.add_argument(
        "--seed-url",
        action="append",
        dest="seed_urls",
        default=env_seed_urls,
        metavar="URL",
        help=(
            "Extra URL(s) to add to the crawl queue before starting (repeatable). "
            "Use when the site does not expose all chapter links as <a> tags. "
            "Can be set via SEED_URLS= (JSON array, comma, newline, or "
            "space-separated) in .env."
        ),
    )
    p.add_argument(
        "--auto-auth-header-from-cookie",
        action=argparse.BooleanOptionalAction,
        default=env_auto_auth_header,
        help=(
            "Auto-add Authorization: Bearer <token-cookie> when COOKIE contains token= "
            "and no explicit Authorization header is provided. "
            "Can be set via AUTO_AUTH_HEADER_FROM_COOKIE= in .env."
        ),
    )
    p.add_argument(
        "--user-agent",
        default=env_user_agent,
        metavar="UA",
        help=(
            "Override the User-Agent for both requests and Playwright. "
            "IMPORTANT: if you include cf_clearance in --cookie, the UA MUST match "
            "the browser that generated that cookie, otherwise Cloudflare rejects it. "
            "Can be set via USER_AGENT= in .env."
        ),
    )
    p.add_argument(
        "--follow-links",
        action=argparse.BooleanOptionalAction,
        default=env_follow_links,
        help=(
            "Discover and crawl additional HTML pages found in <a href> and __NEXT_DATA__. "
            "Disable for a simple explicit-URL download flow. "
            "Can be set via FOLLOW_LINKS= in .env."
        ),
    )
    p.add_argument(
        "--auth-debug",
        action=argparse.BooleanOptionalAction,
        default=env_auth_debug,
        help=(
            "Emit detailed Playwright auth/network diagnostics (cookies, storage keys, xhr status). "
            "Can be set via AUTH_DEBUG= in .env."
        ),
    )
    p.add_argument(
        "--strip-selector",
        action="append",
        dest="strip_selectors",
        default=env_strip_selectors,
        metavar="CSS",
        help=(
            "CSS selector(s) to remove from HTML after Playwright render (repeatable). "
            "Use to strip paywall overlays, cookie banners, etc. that hide real content. "
            "The content underneath is preserved. "
            "Can be set via STRIP_SELECTORS= (comma-separated) in .env."
        ),
    )
    p.add_argument(
        "--discover-chapters",
        action=argparse.BooleanOptionalAction,
        default=env_discover_chapters,
        dest="discover_chapters",
        help=(
            "Auto-discover chapter URLs from the course root page before crawling. "
            "Loads the root URL with Playwright, captures JSON API responses, and "
            "extracts chapter paths so you don't have to list them in SEED_URLS manually. "
            "Requires PLAYWRIGHT=true. Discovered URLs are added to the crawl queue. "
            "Can be set via DISCOVER_CHAPTERS=true in .env."
        ),
    )
    return p.parse_args()


def log_site_mode_hints(
    start_url: str,
    url_prefix: Optional[str],
    follow_links: bool,
    discover_chapters: bool,
) -> None:
    """
    Emit targeted config hints for known site structures.

    ByteByteGo guides are flat URLs under ``/guides/``:
      - category: /guides/ai-machine-learning/
      - article:  /guides/what-is-an-ai-agent/

    Articles are *not* nested below the category path, so the crawler must stay
    on the broader ``/guides/`` prefix and keep ``FOLLOW_LINKS=true``.
    """
    parsed = urlparse(start_url)
    host = (parsed.hostname or "").lower()
    path = parsed.path.rstrip("/") or "/"
    normalized_prefix = (
        "/" + url_prefix.strip().strip("/")
        if url_prefix and url_prefix.strip()
        else None
    )

    if path == "/guides" or path.startswith("/guides/"):
        _log(
            "guides_mode_detected",
            url=start_url,
            url_prefix=normalized_prefix or "",
            follow_links=follow_links,
            discover_chapters=discover_chapters,
        )
        if discover_chapters:
            _log(
                "config_warn",
                setting="DISCOVER_CHAPTERS",
                url=start_url,
                hint="Guides pages already expose normal links. Set DISCOVER_CHAPTERS=false.",
            )
        if not follow_links:
            _log(
                "config_warn",
                setting="FOLLOW_LINKS",
                url=start_url,
                hint="Guides crawling needs FOLLOW_LINKS=true so category pages can enqueue article pages.",
            )
        if normalized_prefix and not normalized_prefix.startswith("/guides"):
            _log(
                "config_warn",
                setting="URL_PREFIX",
                value=normalized_prefix,
                url=start_url,
                hint="Guides article URLs are flat under /guides/. Use URL_PREFIX=/guides/ or leave it unset.",
            )
    elif host == "learn.wqu.edu":
        _log(
            "site_mode_detected",
            site="learn.wqu.edu",
            url=start_url,
            follow_links=follow_links,
            discover_chapters=discover_chapters,
        )
        if follow_links:
            _log(
                "config_hint",
                setting="FOLLOW_LINKS",
                url=start_url,
                hint="WQU lesson downloads are usually more predictable with FOLLOW_LINKS=false and explicit START_URLS.",
            )
        _log(
            "config_hint",
            setting="PLAYWRIGHT_STORAGE_STATE",
            url=start_url,
            hint=(
                "For WQU, prefer PLAYWRIGHT_STORAGE_STATE from a real logged-in browser "
                "session instead of manually reconstructing cookies from the Application tab."
            ),
        )


if __name__ == "__main__":
    # Basic argument validation
    args = parse_args()
    if args.max_pages < 1:
        _log("arg_error", arg="--max-pages", error="must be >= 1")
        sys.exit(2)
    if args.threads < 1:
        _log("arg_error", arg="--threads", error="must be >= 1")
        sys.exit(2)

    # --- User-Agent: must be consistent across requests + Playwright ----------
    # Cloudflare's cf_clearance cookie is bound to the User-Agent that solved
    # the challenge. A mismatch causes Cloudflare to reject the request.
    effective_ua = args.user_agent or DEFAULT_HEADERS["User-Agent"]
    SESSION.headers["User-Agent"] = effective_ua
    if args.user_agent:
        _log("user_agent_override", ua=effective_ua)

    # --- Auth cookies: normalize + guard cf_clearance/UA mismatch ------------
    cookie_pairs = parse_cookie_header(args.cookie) if args.cookie else []
    has_cf_clearance = any(k.lower() == "cf_clearance" for k, _ in cookie_pairs)
    if has_cf_clearance and not args.user_agent:
        # Cloudflare binds cf_clearance to the exact UA that solved challenge.
        # If USER_AGENT is not explicitly provided, this cookie commonly causes
        # false auth failures, so drop it by default.
        cookie_pairs = [(k, v) for k, v in cookie_pairs if k.lower() != "cf_clearance"]
        _log(
            "cf_clearance_ignored_no_user_agent",
            hint="Set USER_AGENT from browser if you need cf_clearance",
        )
    elif has_cf_clearance and args.user_agent:
        _log("cf_clearance_with_user_agent", ua_bound=True)

    token_cookie = next((v for k, v in cookie_pairs if k.lower() == "token"), None)
    if token_cookie:
        log_token_lifetime(token_cookie)

    # If token cookie is present, many SPAs expect Authorization Bearer header
    # on API calls. Auto-derive it unless user already provided one.
    has_auth_header = any(
        ":" in raw and raw.split(":", 1)[0].strip().lower() == "authorization"
        for raw in args.header
    )
    if (
        args.auto_auth_header_from_cookie
        and not has_auth_header
        and not args.playwright_storage_state
    ):
        if token_cookie:
            args.header.append(f"Authorization: Bearer {token_cookie}")
            _log("auth_header_from_token_cookie")
    elif args.auto_auth_header_from_cookie and args.playwright_storage_state:
        _log("auth_header_from_cookie_skipped_storage_state")

    # --- Auth: inject cookies and headers into the shared requests SESSION ---
    if cookie_pairs:
        for k, v in cookie_pairs:
            SESSION.cookies.set(k, v)
        _log("session_cookies_injected", count=len(cookie_pairs), names=[k for k, _ in cookie_pairs])

    headers_injected = []
    for raw_header in args.header:
        if ":" in raw_header:
            k, v = raw_header.split(":", 1)
            SESSION.headers[k.strip()] = v.strip()
            headers_injected.append(k.strip())
    if headers_injected:
        _log("session_headers_injected", headers=headers_injected)

    # Build start URL list. Supports simple explicit URL sets via --start-url/START_URLS.
    raw_starts: list[str] = []
    if args.start_urls and not args.follow_links:
        # Explicit URL mode: crawl exactly the chapter links provided.
        raw_starts.extend(args.start_urls or [])
        _log("start_urls_explicit_mode", count=len(args.start_urls or []))
    else:
        if args.url:
            raw_starts.append(args.url)
        raw_starts.extend(args.start_urls or [])

    start_urls: list[str] = []
    seen_start: set[str] = set()
    for raw in raw_starts:
        if not raw.strip():
            continue
        u = canonicalize_url(raw.strip())
        if u not in seen_start:
            seen_start.add(u)
            start_urls.append(u)

    if not start_urls:
        _log("arg_error", arg="--url/--start-url", error="at least one start URL is required")
        sys.exit(2)

    host = start_urls[0]
    log_site_mode_hints(
        start_url=host,
        url_prefix=args.url_prefix,
        follow_links=bool(args.follow_links),
        discover_chapters=bool(args.discover_chapters),
    )
    extra_start_urls = start_urls[1:]
    merged_seed_urls = extra_start_urls + (args.seed_urls or [])
    root = make_root(host, args.destination)
    first_party_hosts = {
        h.lower()
        for h in ((urlparse(u).hostname or "") for u in start_urls)
        if h
    }

    primary_host = (urlparse(host).hostname or "").lower()
    if primary_host == "learn.wqu.edu" and not args.remove_js:
        args.remove_js = True
        _log(
            "site_mode_adjustment",
            site="learn.wqu.edu",
            setting="REMOVE_JS",
            value=True,
            url=host,
            hint=(
                "WQU lesson pages already contain rendered HTML. Keeping site JS in "
                "offline mode causes the client app to blank the content on file://."
            ),
        )

    if args.playwright_storage_state:
        load_storage_state_into_session(Path(args.playwright_storage_state), first_party_hosts)

    external_domains = (
        {
            urlparse(d).hostname.lower() if "://" in d else d.lower()
            for d in args.external_domains
        }
        if args.external_domains
        else None
    )

    download_external_assets = (
        args.download_external_assets or args.external_domains is not None
    )

    # --- Playwright mode: launch browser, inject cookies, build fetch_fn ---
    _pw_stack = None
    fetch_fn: Optional[Callable[[str], Optional[BeautifulSoup]]] = None

    if args.playwright:
        if not HAS_PLAYWRIGHT:
            _log("playwright_missing", error="Playwright not installed. Run: uv add playwright && uv run playwright install chromium")
            sys.exit(1)

        _pw_stack = sync_playwright().__enter__()
        browser = _pw_stack.chromium.launch(headless=True)

        # Build the Playwright cookie list from --cookie string.
        # If storage_state is used, avoid overriding it with possibly stale .env cookies.
        pw_cookies = []
        if cookie_pairs and not args.playwright_storage_state:
            parsed_start = urlparse(host)
            origin = f"{parsed_start.scheme or 'https'}://{parsed_start.netloc}"
            for k, v in cookie_pairs:
                pw_cookies.append({
                    "name": k,
                    "value": v,
                    "url": origin,
                })
        elif cookie_pairs and args.playwright_storage_state:
            _log("playwright_cookie_overlay_skipped_storage_state")

        # Build extra HTTP headers for Playwright from --header args
        pw_headers = {}
        pw_auth_value: Optional[str] = None
        for raw_header in args.header:
            if ":" in raw_header:
                k, v = raw_header.split(":", 1)
                key = k.strip()
                val = v.strip()
                if key.lower() == "authorization":
                    pw_auth_value = val
                else:
                    pw_headers[key] = val

        context_kwargs = {
            "user_agent": effective_ua,
        }
        if pw_headers:
            context_kwargs["extra_http_headers"] = pw_headers

        if args.playwright_storage_state:
            state_path = Path(args.playwright_storage_state)
            if not state_path.exists():
                _log("arg_error", arg="--playwright-storage-state", error=f"file not found: {state_path}")
                sys.exit(2)
            context_kwargs["storage_state"] = str(state_path)
            _log("playwright_storage_state_loaded", path=str(state_path))

        pw_context = browser.new_context(**context_kwargs)

        # Scope Authorization to first-party hosts only. Sending bearer tokens
        # to third-party scripts/styles causes CORS failures and token leakage.
        scoped_hosts = sorted(first_party_hosts)
        if pw_auth_value and scoped_hosts:

            def route_with_scoped_auth(route, request) -> None:  # noqa: ANN001
                req_host = (urlparse(request.url).hostname or "").lower()
                same_party = any(
                    req_host == host or req_host.endswith("." + host)
                    for host in scoped_hosts
                )
                if same_party:
                    headers = dict(request.headers)
                    headers["Authorization"] = pw_auth_value
                    route.continue_(headers=headers)
                else:
                    route.continue_()

            pw_context.route("**/*", route_with_scoped_auth)
            _log("playwright_auth_header_scoped", hosts=scoped_hosts)

        if pw_cookies:
            pw_context.add_cookies(pw_cookies)
            _log("playwright_cookies_injected", count=len(pw_cookies), names=[c["name"] for c in pw_cookies])

        wait_for = args.wait_for
        render_settle_ms = max(0, args.render_settle_ms)
        debug_auth = bool(args.auth_debug)
        expected_cookie_names = [k for k, _ in cookie_pairs]
        if args.playwright_page_fetch:
            fetch_fn = (
                lambda url: fetch_html_rendered(
                    url,
                    pw_context,
                    wait_for,
                    render_settle_ms,
                    debug_auth,
                    expected_cookie_names,
                )
            )  # noqa: E731
            _log(
                "playwright_mode_start",
                wait_for=wait_for,
                render_settle_ms=render_settle_ms,
                auth_debug=debug_auth,
                page_fetch=True,
            )
        else:
            _log(
                "playwright_mode_start",
                wait_for=wait_for,
                render_settle_ms=render_settle_ms,
                auth_debug=debug_auth,
                page_fetch=False,
            )

        # ── Chapter auto-discovery ────────────────────────────────────────────
        # When DISCOVER_CHAPTERS=true and no SEED_URLS are already provided,
        # load the course root URL and extract chapter links from API responses
        # or the rendered DOM so the user doesn't need to list them manually.
        if args.discover_chapters:
            discovered = _discover_course_chapters(
                course_url=host,
                pw_context=pw_context,
                url_prefix=args.url_prefix,
                root_netloc=_canonical_netloc(urlparse(host)),
                wait_for=wait_for,
                render_settle_ms=render_settle_ms,
            )
            if discovered:
                # Merge: discovered URLs not already in seed list get added.
                existing = set(merged_seed_urls)
                new_chapters = [u for u in discovered if u not in existing]
                merged_seed_urls = merged_seed_urls + new_chapters
                _log(
                    "chapter_discovery_merged",
                    added=len(new_chapters),
                    total_seeds=len(merged_seed_urls),
                )
            elif not merged_seed_urls:
                _log(
                    "chapter_discovery_no_seeds",
                    hint="Discovery found nothing and SEED_URLS is empty. "
                         "Only the root URL will be crawled.",
                )

    elif args.discover_chapters and not args.playwright:
        _log(
            "chapter_discovery_skipped",
            reason="DISCOVER_CHAPTERS=true requires PLAYWRIGHT=true",
        )

    try:
        # Kick off crawl
        crawl_site(
            host,
            root,
            args.max_pages,
            args.threads,
            download_external_assets,
            external_domains,
            fetch_fn=fetch_fn,
            url_prefix=args.url_prefix,
            seed_urls=merged_seed_urls or None,
            remove_js=args.remove_js,
            auth_fail_text=args.auth_fail_text,
            follow_links=bool(args.follow_links),
            strip_selectors=args.strip_selectors or None,
        )
    finally:
        if _pw_stack is not None:
            _pw_stack.stop()
