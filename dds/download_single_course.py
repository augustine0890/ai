import json
import shutil
import string
import subprocess
import urllib.request
from pathlib import Path
from typing import Any, List, cast
from urllib.parse import urlparse

import requests
import yt_dlp
import imageio_ffmpeg
from course_model import CourseModel, VideoItem
from video_model import VideoModel


# Global state to share updated token across functions mid-run
_CURRENT_AUTH_TOKEN = None

def get_auth_token(fallback: str) -> str:
    global _CURRENT_AUTH_TOKEN
    return _CURRENT_AUTH_TOKEN if _CURRENT_AUTH_TOKEN else fallback

def update_auth_token(new_token: str) -> None:
    global _CURRENT_AUTH_TOKEN
    _CURRENT_AUTH_TOKEN = new_token
    input_file = Path(__file__).parent / "input.json"
    if input_file.exists():
        try:
            data = json.loads(input_file.read_text(encoding="utf-8"))
            data["authorization_token"] = new_token
            input_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass


def prompt_new_token(skip_label: str = "skip") -> str | None:
    """Ask the user to update input.json instead of pasting in the terminal.

    Returns the new token string, or None if the user chose to skip/abort.
    Reads the token from input.json after the user confirms, avoiding
    terminal chunking issues with long Bearer tokens.
    """
    input_file = Path(__file__).parent / "input.json"
    print(f"\n  → Open  {input_file}")
    print(    "  → Replace the value of \"authorization_token\" with your new token")
    print(   f"  → Press Enter when done, or type 's' to {skip_label}: ", end="", flush=True)
    choice = input().strip().lower()
    if choice == "s":
        return None
    try:
        data = json.loads(input_file.read_text(encoding="utf-8"))
        token = data.get("authorization_token", "").strip()
        if token:
            update_auth_token(token)
            return token
    except Exception:
        pass
    print("  ⚠️  Could not read token from input.json")
    return None


def log_event(event: str, **fields: Any) -> None:
    from logger import log_event as _log
    _log("dds.worker", event, **fields)


def normalize_name(name: str) -> str:
    return name.translate(str.maketrans("", "", string.punctuation))


def save_html_asset(filepath: Path, title: str, html_content: str) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg:      #f0f2f5;
      --card:    #ffffff;
      --text:    #1a1d2e;
      --muted:   #6b7280;
      --accent:  #4361ee;
      --accent2: #ebefff;
      --border:  #e5e7eb;
      --code-bg: #f3f4f6;
      --shadow:  0 8px 32px rgba(67,97,238,.10);
      --radius:  14px;
    }}
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.75;
      padding: 2.5rem 1rem;
      font-size: 16px;
    }}
    article {{
      max-width: 800px;
      margin: 0 auto;
      background: var(--card);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 3.5rem 4rem;
    }}
    /* ── Title ── */
    .lesson-title {{
      font-size: 2rem;
      font-weight: 700;
      color: var(--text);
      line-height: 1.25;
      margin-bottom: 2rem;
      padding-bottom: 1.25rem;
      border-bottom: 3px solid var(--accent);
    }}
    /* ── Headings ── */
    h2 {{ font-size: 1.55rem; font-weight: 700; margin: 2.2rem 0 .75rem; line-height: 1.3; }}
    h3 {{ font-size: 1.25rem; font-weight: 600; margin: 1.8rem 0 .6rem;  line-height: 1.35; }}
    h4, h5, h6 {{ font-size: 1.05rem; font-weight: 600; margin: 1.4rem 0 .5rem; }}
    /* ── Body text ── */
    p {{
      margin: .9rem 0;
      color: var(--text);
    }}
    /* ── Lists ── */
    ul, ol {{
      margin: .75rem 0 .75rem 1.6rem;
    }}
    li {{ margin: .4rem 0; }}
    li::marker {{ color: var(--accent); font-weight: 600; }}
    /* ── Blockquote ── */
    blockquote {{
      border-left: 4px solid var(--accent);
      background: var(--accent2);
      margin: 1.75rem 0;
      padding: 1rem 1.5rem;
      border-radius: 0 8px 8px 0;
      font-style: italic;
    }}
    blockquote cite {{
      display: block;
      margin-top: .5rem;
      font-style: normal;
      font-weight: 600;
      color: var(--muted);
      font-size: .875rem;
    }}
    /* ── Callout ── */
    .callout {{
      background: var(--accent2);
      border: 1px solid var(--accent);
      border-radius: 8px;
      padding: 1rem 1.25rem;
      margin: 1.25rem 0;
      display: flex;
      gap: .75rem;
      align-items: flex-start;
    }}
    .callout-icon {{ font-size: 1.2rem; flex-shrink: 0; margin-top: .05rem; }}
    /* ── Table ── */
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 1.5rem 0;
      font-size: .95rem;
    }}
    th, td {{
      padding: .65rem 1rem;
      text-align: left;
      border: 1px solid var(--border);
    }}
    thead th {{
      background: var(--bg);
      font-weight: 600;
      color: var(--text);
    }}
    tbody tr:nth-child(even) {{ background: #fafafa; }}
    /* ── Divider ── */
    hr {{
      border: none;
      border-top: 2px solid var(--border);
      margin: 2rem 0;
    }}
    /* ── Images ── */
    figure {{
      margin: 1.75rem 0;
      text-align: center;
    }}
    figure img {{
      max-width: 100%;
      border-radius: 10px;
      box-shadow: var(--shadow);
    }}
    figcaption {{
      color: var(--muted);
      font-size: .85rem;
      margin-top: .5rem;
    }}
    /* ── Inline ── */
    strong, b {{ font-weight: 600; }}
    em, i     {{ font-style: italic; }}
    code {{
      background: var(--code-bg);
      padding: .15em .4em;
      border-radius: 4px;
      font-size: .875em;
      font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
    }}
    a {{ color: var(--accent); text-decoration: underline; }}
    /* ── Checklist ── */
    .checklist {{ list-style: none; margin-left: 0; }}
    .checklist li {{
      display: flex;
      align-items: flex-start;
      gap: .5rem;
    }}
    .checklist li::before {{
      content: '☐';
      color: var(--accent);
      font-size: 1.1rem;
      flex-shrink: 0;
    }}
    .checklist li.checked::before {{ content: '☑'; }}
    /* ── Learning Objectives ── */
    .learning-objectives {{
      background: var(--accent2);
      border: 1px solid var(--accent);
      border-radius: 10px;
      padding: 1.25rem 1.5rem;
      margin: 1.5rem 0;
    }}
    .learning-objectives h3 {{
      font-size: 1rem;
      font-weight: 700;
      color: var(--accent);
      text-transform: uppercase;
      letter-spacing: .08em;
      margin-bottom: .75rem;
    }}
    .learning-objectives ul {{
      margin-left: 1.25rem;
    }}
    .learning-objectives li {{
      margin: .45rem 0;
      font-size: .97rem;
    }}
    /* ── Responsive ── */
    @media (max-width: 640px) {{
      article {{ padding: 1.75rem 1.25rem; }}
      .lesson-title {{ font-size: 1.5rem; }}
    }}
  </style>
</head>
<body>
  <article>
    <h1 class="lesson-title">{title}</h1>
    {html_content}
  </article>
</body>
</html>
"""
    filepath.write_text(document, encoding="utf-8")


def _extract_html_from_payload(payload: Any) -> str | None:
    if isinstance(payload, str):
        stripped = payload.strip()
        if "<" in stripped and ">" in stripped:
            return payload
        return None

    if isinstance(payload, dict):
        for key in ("text", "html", "content", "body", "description"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        for value in payload.values():
            nested = _extract_html_from_payload(value)
            if nested:
                return nested

    if isinstance(payload, list):
        for item in payload:
            nested = _extract_html_from_payload(item)
            if nested:
                return nested

    return None


def html_to_plain_text(html_content: str) -> str:
    """Strip HTML tags and return clean, readable plain text."""
    from bs4 import BeautifulSoup  # pyright: ignore[reportMissingImports]
    soup = BeautifulSoup(html_content, "html.parser")
    # Remove script / style noise
    for tag in soup(["script", "style", "noscript", "iframe"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def _collect_all_strings(payload: Any, seen: set[int] | None = None) -> list[str]:
    """Recursively walk any JSON value and collect every non-trivial string.

    Intentionally broad: prefers priority keys first, then scans everything,
    so content is captured even under unexpected field names.
    """
    if seen is None:
        seen = set()
    obj_id = id(payload)
    if obj_id in seen:
        return []
    seen.add(obj_id)

    results: list[str] = []
    if isinstance(payload, str):
        stripped = payload.strip()
        # Include strings that look like real text: have a space (multi-word)
        # or are long enough to be a meaningful sentence/paragraph.
        # Skip URLs and data URIs which are never lesson text.
        if (
            stripped
            and not stripped.startswith(("http://", "https://", "data:", "//"))
            and (" " in stripped or len(stripped) > 30)
        ):
            results.append(stripped)
    elif isinstance(payload, dict):
        # Priority fields first — most likely to hold the actual lesson content
        priority_keys = (
            "text", "html", "content", "body", "description",
            "summary", "transcript", "lesson", "notes",
            "htmlContent", "textContent", "richText", "markdown",
        )
        for key in priority_keys:
            if key in payload:
                results.extend(_collect_all_strings(payload[key], seen))
        for key, value in payload.items():
            if key not in priority_keys:
                results.extend(_collect_all_strings(value, seen))
    elif isinstance(payload, list):
        for item in payload:
            results.extend(_collect_all_strings(item, seen))
    return results


# The real API endpoint for text lesson content is:
#   GET /course/text/{asset_id}  (singular "course", not "courses")
# It returns Editor.js structured JSON with blocks (header, paragraph, list, etc.)
# Discovered by intercepting browser network calls on the logged-in lesson page.


def editorjs_to_html_and_text(editorjs_data: Any) -> tuple[str, str] | None:
    """Convert an Editor.js JSON payload into (html_fragment, plain_text).

    Real structure from the 365 platform API:
    - The payload is a flat JSON *array* of block objects.
    - Each block: {"id": "...", "type": "...", "data": {...}, "tunes": {...}}
    - "id", "tunes" → metadata only, ignored in output.
    - "data.words", "data.number", "data.level" → metadata, ignored.
    - listUnordered / listOrdered → each block is ONE list item (data.content).
      Consecutive same-type blocks are grouped into a single <ul> / <ol>.
    - templateLearningObjectives → data.items is an array of {content, words}.
    """
    blocks: list[Any] = []
    if isinstance(editorjs_data, dict):
        blocks = editorjs_data.get("blocks", [])
        if not blocks and isinstance(editorjs_data.get("data"), dict):
            blocks = editorjs_data["data"].get("blocks", [])
        if not blocks and isinstance(editorjs_data.get("data"), list):
            blocks = editorjs_data["data"]
    elif isinstance(editorjs_data, list):
        blocks = editorjs_data

    if not blocks:
        return None

    html_parts: list[str] = []
    text_parts: list[str] = []

    # ── List-grouping state ───────────────────────────────────────────────────
    # Because each listUnordered/listOrdered block = one item, we accumulate
    # consecutive items and flush them as a single <ul>/<ol>.
    pending_list_tag: str = ""          # "ul" or "ol"
    pending_list_html: list[str] = []   # <li> strings
    pending_list_text: list[str] = []   # plain-text lines

    def flush_list() -> None:
        nonlocal pending_list_tag, pending_list_html, pending_list_text
        if pending_list_html:
            items_html = "\n".join(pending_list_html)
            html_parts.append(f"<{pending_list_tag}>\n{items_html}\n</{pending_list_tag}>")
            text_parts.append("\n".join(pending_list_text))
        pending_list_tag = ""
        pending_list_html = []
        pending_list_text = []

    LIST_BLOCK_TYPES = {"listUnordered", "listOrdered", "list"}

    for raw_block in blocks:
        # Handle stringified JSON blocks (API sometimes returns blocks as strings)
        block = raw_block
        if isinstance(raw_block, str):
            try:
                block = json.loads(raw_block)
            except (json.JSONDecodeError, ValueError):
                if raw_block.strip():
                    flush_list()
                    html_parts.append(f"<p>{raw_block}</p>")
                    text_parts.append(raw_block)
                continue

        if not isinstance(block, dict):
            continue

        btype: str = block.get("type", "")
        data: Any = block.get("data", {})
        if not isinstance(data, dict):
            data = {}

        # If this block is NOT a list item, flush any pending list first
        if btype not in LIST_BLOCK_TYPES:
            flush_list()

        # ── Header ─────────────────────────────────────────────────────────
        if btype == "header":
            raw = data.get("text", "")
            level = min(max(int(data.get("level", 2)), 1), 6)
            if raw:
                plain = html_to_plain_text(raw)
                html_parts.append(f"<h{level}>{raw}</h{level}>")
                underline = "═" * len(plain) if level <= 2 else "─" * len(plain)
                text_parts.append(f"\n{plain}\n{underline}")

        # ── Paragraph ──────────────────────────────────────────────────────
        elif btype == "paragraph":
            raw = data.get("text", "")
            if raw:
                html_parts.append(f"<p>{raw}</p>")
                text_parts.append(html_to_plain_text(raw))

        # ── Per-item list blocks ────────────────────────────────────────────
        # Each block = exactly ONE list item (data.content holds the item text).
        elif btype in ("listUnordered", "listOrdered"):
            content = data.get("content", "")
            if not content:
                continue
            tag = "ol" if btype == "listOrdered" else "ul"
            # Start a new list group if the tag changed
            if pending_list_tag and pending_list_tag != tag:
                flush_list()
            pending_list_tag = tag
            plain_item = html_to_plain_text(content)
            pending_list_html.append(f"  <li>{content}</li>")
            prefix = f"{len(pending_list_html)}." if tag == "ol" else "•"
            pending_list_text.append(f"  {prefix} {plain_item}")

        # ── Standard Editor.js list block (data.items array) ───────────────
        elif btype == "list":
            items = data.get("items", [])
            style = data.get("style", "unordered")
            tag = "ol" if style == "ordered" else "ul"
            if items:
                # Each item may be a string or {content: "..."} dict
                def _extract(i: Any) -> str:
                    if isinstance(i, str):
                        return i
                    if isinstance(i, dict):
                        return i.get("content", "") or i.get("text", "")
                    return str(i)
                li_html = "\n".join(f"  <li>{_extract(it)}</li>" for it in items)
                html_parts.append(f"<{tag}>\n{li_html}\n</{tag}>")
                lines = []
                for idx, it in enumerate(items, 1):
                    t = html_to_plain_text(_extract(it))
                    pfx = f"{idx}." if style == "ordered" else "•"
                    lines.append(f"  {pfx} {t}")
                text_parts.append("\n".join(lines))

        # ── Learning objectives template block ──────────────────────────────
        elif btype == "templateLearningObjectives":
            items = data.get("items", [])
            if items:
                li_html = "\n".join(
                    f'  <li>{it.get("content", "") if isinstance(it, dict) else it}</li>'
                    for it in items
                )
                html_parts.append(
                    '<div class="learning-objectives">'
                    "<h3>Learning Objectives</h3>"
                    f"<ul>\n{li_html}\n</ul>"
                    "</div>"
                )
                obj_lines = ["Learning Objectives"]
                obj_lines.append("─" * 22)
                for it in items:
                    content = it.get("content", "") if isinstance(it, dict) else str(it)
                    obj_lines.append(f"  • {html_to_plain_text(content)}")
                text_parts.append("\n".join(obj_lines))

        # ── Quote ──────────────────────────────────────────────────────────
        elif btype == "quote":
            raw = data.get("text", "")
            caption = data.get("caption", "")
            if raw:
                cite_html = f"<cite>{caption}</cite>" if caption else ""
                html_parts.append(f"<blockquote><p>{raw}</p>{cite_html}</blockquote>")
                plain = html_to_plain_text(raw)
                text_parts.append(f'"{plain}"' + (f"\n    — {caption}" if caption else ""))

        # ── Delimiter / Divider ────────────────────────────────────────────
        elif btype in ("delimiter", "divider"):
            html_parts.append("<hr>")
            text_parts.append("─" * 60)

        # ── Image / Images ────────────────────────────────────────────────
        elif btype in ("image", "images"):
            file_info = data.get("file") or {}
            url = file_info.get("url", "") or data.get("url", "")
            caption = data.get("caption", "")
            if url:
                html_parts.append(
                    f'<figure><img src="{url}" alt="{caption}" loading="lazy">'
                    + (f"<figcaption>{caption}</figcaption>" if caption else "")
                    + "</figure>"
                )
                text_parts.append(f"[Image{': ' + caption if caption else ''}]")

        # ── File attachment ───────────────────────────────────────────────
        elif btype == "attaches":
            file_info = data.get("file") or {}
            file_url = file_info.get("url", "")
            file_title = file_info.get("title", "") or file_info.get("name", "attachment")
            file_ext = file_info.get("extension", "")
            file_size = file_info.get("size", "")
            if file_url:
                size_str = ""
                if file_size:
                    try:
                        size_kb = int(file_size) / 1024
                        size_str = f" ({size_kb:.0f} KB)" if size_kb < 1024 else f" ({size_kb/1024:.1f} MB)"
                    except (ValueError, TypeError):
                        pass
                html_parts.append(
                    f'<div class="attachment">'
                    f'<a href="{file_url}" download>📎 {file_title}{size_str}</a>'
                    f'</div>'
                )
                text_parts.append(f"📎 {file_title}{size_str}: {file_url}")

        # ── Template custom block (recursive) ─────────────────────────────
        elif btype == "templateCustomBlock":
            inner_content = data.get("content")
            if isinstance(inner_content, dict):
                inner_blocks = inner_content.get("blocks", [])
                if inner_blocks:
                    inner_result = editorjs_to_html_and_text(inner_blocks)
                    if inner_result:
                        inner_html, inner_txt = inner_result
                        html_parts.append(
                            f'<div class="callout">{inner_html}</div>'
                        )
                        text_parts.append(inner_txt)

        # ── Table ──────────────────────────────────────────────────────────
        elif btype == "table":
            rows = data.get("content", [])
            with_headings = data.get("withHeadings", False)
            if isinstance(rows, list) and rows:
                thead, tbody_rows = "", rows
                if with_headings and rows:
                    hcells = "".join(f"<th>{c}</th>" for c in rows[0])
                    thead = f"<thead><tr>{hcells}</tr></thead>"
                    tbody_rows = rows[1:]
                body_html = "".join(
                    "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
                    for row in tbody_rows
                )
                html_parts.append(f"<table>{thead}<tbody>{body_html}</tbody></table>")
                for row in rows:
                    text_parts.append("  " + " │ ".join(str(c) for c in row))

        # ── Callout / Warning / Info ────────────────────────────────────────
        elif btype in ("callOut", "warning", "alert"):
            text = data.get("message", "") or data.get("text", "")
            icon = data.get("icon", "") or ("⚠️" if btype == "warning" else "💡")
            if text:
                html_parts.append(
                    f'<div class="callout">'
                    f'<span class="callout-icon">{icon}</span>'
                    f"<div>{text}</div></div>"
                )
                text_parts.append(f"{icon}  {html_to_plain_text(text)}")

        # ── Checklist ──────────────────────────────────────────────────────
        elif btype == "checklist":
            items = data.get("items", [])
            if items:
                li_html = "\n".join(
                    f'  <li class="{"checked" if (i.get("checked") if isinstance(i, dict) else False) else ""}">'
                    f'{i.get("content", i) if isinstance(i, dict) else i}</li>'
                    for i in items
                )
                html_parts.append(f'<ul class="checklist">\n{li_html}\n</ul>')
                for item in items:
                    checked = item.get("checked", False) if isinstance(item, dict) else False
                    content = item.get("content", item) if isinstance(item, dict) else item
                    mark = "[x]" if checked else "[ ]"
                    text_parts.append(f"  {mark} {html_to_plain_text(str(content))}")

        # ── Generic fallback ────────────────────────────────────────────────
        else:
            raw = data.get("text", "") or data.get("message", "")
            # Only use data.content if it's a string (avoid stringifying dicts)
            if not raw:
                c = data.get("content", "")
                if isinstance(c, str):
                    raw = c
            if raw and isinstance(raw, str):
                html_parts.append(f"<p>{raw}</p>")
                text_parts.append(html_to_plain_text(raw))

    # ── Assemble ─────────────────────────────────────────────────────────────
    combined_html = "\n".join(html_parts)
    # Join text parts: single blank line between paragraphs/items
    combined_text = "\n\n".join(t.strip() for t in text_parts if t.strip())
    if combined_text.strip():
        return combined_html, combined_text
    return None


def ensure_parsed_html(raw_content: str) -> str:
    """If raw_content looks like JSON (Editor.js blocks), parse it into HTML.

    Returns proper HTML in all cases — either parsed from JSON or the original
    string if it's already HTML or plain text.
    """
    stripped = raw_content.strip()
    # Quick check: does it look like JSON?
    if stripped.startswith(("[", "{")):
        try:
            payload = json.loads(stripped)
            result = editorjs_to_html_and_text(payload)
            if result:
                return result[0]
        except (json.JSONDecodeError, ValueError):
            pass
    return raw_content


def fetch_text_lesson_content(
    asset_id: int,
    authorization_token: str,
    api_base_url: str,
) -> tuple[str, str] | None:
    """Fetch text lesson content from /course/text/{asset_id}.

    This is the real endpoint the 365 platform SPA uses for reading/text type
    lessons. It returns Editor.js structured JSON which we convert to HTML+text.
    """
    # Note: singular 'course' (not 'courses') — this is specific to how
    # the 365 platform names this endpoint.
    url = f"{api_base_url.rstrip('/')}/course/text/{asset_id}"
    
    while True:
        token = get_auth_token(authorization_token)
        headers = {
            "accept": "application/json, text/plain, */*",
            "content-type": "application/json;charset=UTF-8",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Authorization": f"Bearer {token}",
            "referer": "https://learn.365financialanalyst.com/",
        }
        log_event("text_lesson_api_request", url=url, asset_id=asset_id)
        try:
            response = requests.get(url, headers=headers, timeout=20)
        except requests.RequestException as exc:
            log_event("text_lesson_api_error", url=url, error=str(exc))
            return None

        if response.status_code >= 400:
            try:
                err = response.json()
                reason = err.get("message") or err.get("error") or str(err)[:120]
            except ValueError:
                reason = response.text[:120]
                
            if response.status_code in (401, 403) or "invalid or expired" in reason.lower():
                print(f"\n⚠️  [Text API] Token expired or invalid: {reason}")
                new_token = prompt_new_token(skip_label="skip asset")
                if not new_token:
                    log_event("text_lesson_api_skip", url=url, status_code=response.status_code, reason=reason)
                    return None
                continue

            log_event("text_lesson_api_skip", url=url, status_code=response.status_code, reason=reason)
            return None
            
        # Success, break loop
        break

    try:
        payload = response.json()
    except ValueError:
        raw = response.text.strip()
        if raw:
            return f"<pre>{raw}</pre>", raw
        return None

    log_event("text_lesson_api_ok", url=url, status_code=response.status_code)

    # Try Editor.js block format first (this is what the 365 platform returns)
    result = editorjs_to_html_and_text(payload)
    if result:
        return result

    # Fallback: if the payload is plain HTML or a different JSON structure
    html_frag = _extract_html_from_payload(payload)
    if html_frag:
        return html_frag, html_to_plain_text(html_frag)

    chunks = _collect_all_strings(payload)
    if chunks:
        plain_parts: list[str] = []
        html_parts: list[str] = []
        for chunk in chunks:
            if "<" in chunk and ">" in chunk:
                plain_parts.append(html_to_plain_text(chunk))
                html_parts.append(chunk)
            else:
                plain_parts.append(chunk)
                html_parts.append(f"<p>{chunk}</p>")
        combined_plain = "\n\n".join(p for p in plain_parts if p.strip())
        combined_html = "\n".join(h for h in html_parts if h.strip())
        if combined_plain.strip():
            return combined_html, combined_plain

    return None



def request_lecture_html(
    course_slug: str,
    lecture_id: int,
    authorization_token: str,
    api_base_url: str,
) -> str | None:
    headers_for_365 = {
        "accept": "application/json, text/plain, */*",
        "content-type": "application/json;charset=UTF-8",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Authorization": f"Bearer {authorization_token}",
    }

    candidate_urls = [
        f"{api_base_url}/courses/{course_slug}/lectures/{lecture_id}",
        f"{api_base_url}/courses/lectures/{lecture_id}",
        f"{api_base_url}/lectures/{lecture_id}",
    ]

    for url in candidate_urls:
        log_event("lecture_api_request", url=url, lecture_id=lecture_id)
        response = requests.get(url, headers=headers_for_365, timeout=20)
        if response.status_code >= 400:
            log_event("lecture_api_skip", url=url, status_code=response.status_code)
            continue

        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type and response.text.strip():
            log_event("lecture_api_ok", url=url, source="html")
            return response.text

        try:
            payload = response.json()
        except ValueError:
            if response.text.strip():
                log_event("lecture_api_ok", url=url, source="raw_text")
                return response.text
            continue

        extracted = _extract_html_from_payload(payload)
        if extracted:
            log_event("lecture_api_ok", url=url, source="json_field")
            return extracted

    return None


# fetch_lesson_page_html removed: the site is a SPA so requests.get only
# returns an empty JS shell. Content is fetched via fetch_lecture_text_content.


def download_video_from_stream_url(
    video_stream_url: str, filepath: str, quality: str
) -> None:
    """Download a video from stream url
    :param video_stream_url: stream url
    :param filepath: file path where to download
    :param quality: quality to select
    """
    quality_limit = "".join(ch for ch in quality if ch.isdigit()) or "1080"
    ffmpeg_exe = shutil.which("ffmpeg")
    ffmpeg_source = "system"
    if not ffmpeg_exe:
        try:
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            ffmpeg_source = "python-fallback"
        except Exception:
            ffmpeg_exe = None
            ffmpeg_source = "missing"

    ffmpeg_ready = False
    if ffmpeg_exe:
        try:
            subprocess.run(
                [ffmpeg_exe, "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            ffmpeg_ready = True
        except Exception:
            ffmpeg_ready = False

    ydl_opts: dict[str, Any] = {
        "concurrent_fragment_downloads": 15,
        "outtmpl": f"{filepath}.%(ext)s",
        "writesubtitles": True,
    }

    if ffmpeg_ready and ffmpeg_exe:
        ydl_opts["format"] = (
            f"bestvideo[height<={quality_limit}]+bestaudio/"
            f"best[height<={quality_limit}]/best"
        )
        ydl_opts["postprocessors"] = [{"key": "FFmpegFixupM3u8"}]
        ydl_opts["ffmpeg_location"] = ffmpeg_exe
    else:
        ydl_opts["format"] = f"best[height<={quality_limit}]/best"

    log_event(
        "video_download_start",
        filepath=filepath,
        quality=quality,
        quality_limit=quality_limit,
        ffmpeg_ready=ffmpeg_ready,
        ffmpeg_source=ffmpeg_source,
    )
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # pyright: ignore[reportArgumentType]
        ydl.download([video_stream_url])
    log_event("video_download_done", filepath=filepath)


def get_api_base_url(course_url: str) -> str:
    parsed = urlparse(course_url)
    host = parsed.hostname or ""

    if host.endswith("365financialanalyst.com"):
        return "https://api.365financialanalyst.com"

    return "https://api.365datascience.com"


def get_learn_base_url(course_url: str) -> str:
    parsed = urlparse(course_url)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"

    return "https://learn.365datascience.com"


def request_course_api(
    course_slug: str, authorization_token: str, api_base_url: str
) -> tuple[CourseModel, dict]:
    """Fetch the course player data.

    Returns both the typed CourseModel AND the raw JSON dict so callers can
    inspect fields that the Pydantic model doesn't map (e.g. 'content', 'html').
    """
    course_api_url = f"{api_base_url}/courses/{course_slug}/player"
    log_event("course_api_request", url=course_api_url)
    
    while True:
        token = get_auth_token(authorization_token)
        headers_for_365datascience = {
            "authority": "api.365datascience.com",
            "accept": "application/json, text/plain, */*",
            "accept-language": "en-US,en;q=0.9",
            "content-type": "application/json;charset=UTF-8",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36 Edg/107.0.1418.26",
            "Authorization": f"Bearer {token}",
        }
        response = requests.get(course_api_url, headers=headers_for_365datascience)
        
        if response.status_code in (400, 401, 403):
            reason = response.text[:200]
            try:
                err = response.json()
                reason = err.get("message") or err.get("error") or str(err)
            except ValueError:
                pass
                
            if response.status_code in (401, 403) or "invalid or expired" in reason.lower():
                print(f"\n⚠️  [Course API] Token expired or invalid: {reason}")
                new_token = prompt_new_token(skip_label="abort")
                if not new_token:
                    response.raise_for_status()
                continue
                
        response.raise_for_status()
        break
    log_event(
        "course_api_ok", status_code=response.status_code, course_slug=course_slug
    )
    raw = response.json()
    model = CourseModel.parse_raw(response.text)
    return model, raw


def build_raw_asset_lookup(raw_course: dict) -> dict[int, dict]:
    """Build a mapping of asset_id -> raw JSON dict from the player API response.

    This gives access to ALL fields the API returns for each asset, including
    ones that the Pydantic Asset model doesn't map (content, html, richText…).
    """
    lookup: dict[int, dict] = {}
    for section in raw_course.get("sections", []):
        for asset in section.get("assets", []):
            asset_id = asset.get("id")
            if isinstance(asset_id, int):
                lookup[asset_id] = asset
    return lookup


def request_course_resource_api(
    course_slug: str, course_id: int, authorization_token: str, api_base_url: str
) -> List[str]:
    # TODO: Extract common code from here and request_365datascience_course_api function
    course_resource_api_url = f"{api_base_url}/courses/file"
    log_event("resource_api_request", url=course_resource_api_url, course_id=course_id)
    headers_for_365datascience = {
        "authority": "api.365datascience.com",
        "accept": "application/json, text/plain, */*",
        "accept-language": "en-US,en;q=0.9",
        "content-type": "application/json;charset=UTF-8",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36 Edg/107.0.1418.26",
        "Authorization": f"Bearer {authorization_token}",
    }

    json_data = {
        "courseId": course_id,
        "name": f"{course_slug}.zip",
        "courseZip": True,
    }

    response = requests.post(
        course_resource_api_url, headers=headers_for_365datascience, json=json_data
    )
    if response.status_code in (400, 404):
        log_event(
            "resource_api_empty",
            status_code=response.status_code,
            course_slug=course_slug,
        )
        return []
    if 500 <= response.status_code < 600:
        log_event(
            "resource_api_server_error_skip",
            status_code=response.status_code,
            course_slug=course_slug,
            reason="treat_as_no_resources",
        )
        return []
    response.raise_for_status()
    urls = cast(List[str], json.loads(response.text))
    log_event(
        "resource_api_ok", status_code=response.status_code, resource_count=len(urls)
    )
    return urls


def request_brightcove_api(
    video_id: str, policy_key: str, learn_base_url: str
) -> VideoModel:
    header_for_brightcove = {
        "authority": "edge.api.brightcove.com",
        "accept": f"application/json;pk={policy_key}",
        "accept-language": "en-US,en;q=0.9",
        "origin": learn_base_url,
        "referer": f"{learn_base_url}/",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36 Edg/107.0.1418.26",
    }

    video_api_url = f"https://edge.api.brightcove.com/playback/v1/accounts/6258000438001/videos/{video_id}"
    log_event("brightcove_api_request", video_id=video_id)
    response = requests.get(video_api_url, headers=header_for_brightcove)
    log_event("brightcove_api_ok", status_code=response.status_code, video_id=video_id)
    return VideoModel.parse_raw(response.text)


def download_course(
    course_url: str, authorization_token: str, policy_key: str, quality: str
) -> None:
    course_slug = course_url.strip("/").split("/").pop()
    api_base_url = get_api_base_url(course_url)
    learn_base_url = get_learn_base_url(course_url)
    log_event(
        "download_course_start",
        course_url=course_url,
        course_slug=course_slug,
        api_base_url=api_base_url,
    )
    course_data, raw_course = request_course_api(course_slug, authorization_token, api_base_url)
    raw_assets = build_raw_asset_lookup(raw_course)
    log_event(
        "download_course_sections",
        course_name=course_data.info.name,
        section_count=len(course_data.sections),
    )

    for i, section in enumerate(course_data.sections, start=1):
        log_event(
            "section_start",
            section_index=i,
            section_name=section.name,
            asset_count=len(section.assets),
        )
        for j, asset in enumerate(section.assets, start=1):
            # ----------------------------------------------------------------
            # 1. Video download (lesson assets that carry a video)
            # ----------------------------------------------------------------
            has_video = bool(asset.video) and not isinstance(asset.video, bool)
            if asset.type == "lesson" and has_video:
                file_path = (
                    Path(Path.home() / "Downloads")
                    / "365DataScience"
                    / normalize_name(course_data.info.name)
                    / f"{i} - {normalize_name(section.name)}"
                    / f"{j} - {normalize_name(asset.name)}"
                )
                video_item = cast(VideoItem, asset.video)
                log_event(
                    "lesson_video_start",
                    section_index=i,
                    asset_index=j,
                    asset_name=asset.name,
                    ext_id=video_item.ext_id,
                )
                video_data = request_brightcove_api(
                    video_item.ext_id, policy_key, learn_base_url
                )
                source = video_data.sources.pop(0)
                master_m3u8_url = source.src
                download_video_from_stream_url(
                    master_m3u8_url, str(file_path), quality
                )

            # ----------------------------------------------------------------
            # 2. API text / lecture HTML (all asset types)
            # ----------------------------------------------------------------
            html_text = asset.text
            if not html_text and asset.lecture_id:
                html_text = request_lecture_html(
                    course_slug=course_slug,
                    lecture_id=asset.lecture_id,
                    authorization_token=authorization_token,
                    api_base_url=api_base_url,
                )

            if html_text:
                html_file_path = (
                    Path(Path.home() / "Downloads")
                    / "365DataScience"
                    / normalize_name(course_data.info.name)
                    / f"{i} - {normalize_name(section.name)}"
                    / f"{j} - {normalize_name(asset.name)}.html"
                )
                log_event(
                    "lesson_html_start",
                    section_index=i,
                    asset_index=j,
                    asset_name=asset.name,
                    filepath=str(html_file_path),
                )
                save_html_asset(html_file_path, asset.name, ensure_parsed_html(html_text))
                log_event(
                    "lesson_html_done",
                    section_index=i,
                    asset_index=j,
                    filepath=str(html_file_path),
                )
            elif asset.type in {"lesson", "lecture", "text"}:
                log_event(
                    "lesson_html_missing",
                    section_index=i,
                    asset_index=j,
                    asset_name=asset.name,
                    lecture_id=asset.lecture_id,
                )

            # ----------------------------------------------------------------
            # 3. Non-video section content: /course/text/{asset_id}
            #    This is the real endpoint the 365 platform SPA calls for
            #    reading/text lessons. Returns Editor.js JSON which we parse
            #    into clean HTML and plain text.
            # ----------------------------------------------------------------
            if not has_video:
                result = fetch_text_lesson_content(
                    asset_id=asset.id,
                    authorization_token=authorization_token,
                    api_base_url=api_base_url,
                )
                if result:
                    content_html3, content_txt3 = result
                    base_path = (
                        Path(Path.home() / "Downloads")
                        / "365DataScience"
                        / normalize_name(course_data.info.name)
                        / f"{i} - {normalize_name(section.name)}"
                        / f"{j} - {normalize_name(asset.name)}"
                    )
                    # --- HTML version ---
                    html_out = base_path.with_suffix(".html")
                    log_event(
                        "lesson_content_html_save",
                        section_index=i, asset_index=j,
                        asset_name=asset.name, filepath=str(html_out),
                    )
                    save_html_asset(html_out, asset.name, content_html3)
                    log_event("lesson_content_html_done", filepath=str(html_out))
                    # --- TXT version ---
                    txt_out = base_path.with_suffix(".txt")
                    txt_out.parent.mkdir(parents=True, exist_ok=True)
                    log_event(
                        "lesson_content_txt_save",
                        section_index=i, asset_index=j,
                        asset_name=asset.name, filepath=str(txt_out),
                    )
                    txt_out.write_text(content_txt3, encoding="utf-8")
                    log_event("lesson_content_txt_done", filepath=str(txt_out))
                else:
                    log_event(
                        "lesson_content_missing",
                        section_index=i, asset_index=j,
                        asset_name=asset.name, asset_id=asset.id,
                    )

    log_event("download_course_done", course_url=course_url)


def download_course_resource(course_url: str, authorization_token: str) -> None:
    course_slug = course_url.strip("/").split("/").pop()
    api_base_url = get_api_base_url(course_url)
    log_event(
        "download_resource_start",
        course_url=course_url,
        course_slug=course_slug,
        api_base_url=api_base_url,
    )
    course_data, _ = request_course_api(course_slug, authorization_token, api_base_url)
    course_resource_urls = request_course_resource_api(
        course_slug, course_data.id, authorization_token, api_base_url
    )

    log_event(
        "download_resource_items",
        course_slug=course_slug,
        count=len(course_resource_urls),
    )

    for i, course_resource_url in enumerate(course_resource_urls):
        file_path = (
            Path(Path.home() / "Downloads")
            / "365DataScience"
            / normalize_name(course_data.info.name)
            / f"{course_slug}_{i}.zip"
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)
        log_event("resource_download_start", index=i, filepath=str(file_path))
        urllib.request.urlretrieve(course_resource_url, file_path)
        log_event("resource_download_done", index=i, filepath=str(file_path))


if __name__ == "__main__":
    input_file = Path(__file__).parent / "input.json"
    input_data = json.loads(input_file.read_text())
    course_url = input_data.get("course_url")
    authorization_token = input_data.get("authorization_token")
    policy_key = input_data.get("policy_key")
    quality = input_data.get("quality")
    download_course_resource(
        course_url=course_url, authorization_token=authorization_token
    )
    download_course(
        course_url=course_url,
        authorization_token=authorization_token,
        policy_key=policy_key,
        quality=quality,
    )
