"""
clean_downloaded_files.py

Surgically repairs HTML and TXT files that contain raw Editor.js payloads
(either JSON arrays or Python dict reprs) embedded in otherwise readable text.

Instead of replacing the whole file, it finds and replaces only the bad
elements, preserving surrounding good content.
"""

import ast
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup  # pyright: ignore[reportMissingImports]

from download_single_course import editorjs_to_html_and_text, save_html_asset
from logger import log_event

_MODULE = "dds.clean"


# ── Block extraction ─────────────────────────────────────────────────────────

def _try_blocks_from_dict(text: str) -> list | None:
    """Parse {'blocks': [...]} in either JSON or Python single-quote repr."""
    # Fast pre-check before expensive parsing
    if "blocks" not in text:
        return None
    text = text.strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            data = parser(text)
            if isinstance(data, dict):
                blocks = data.get("blocks")
                if isinstance(blocks, list) and blocks and isinstance(blocks[0], dict):
                    return blocks
        except Exception:
            continue
    return None


def _try_blocks_from_array(text: str) -> list | None:
    """Parse a bare JSON array of Editor.js blocks."""
    text = text.strip()
    if not text.startswith("["):
        return None
    try:
        data = json.loads(text)
        if isinstance(data, list) and data and isinstance(data[0], dict):
            if "type" in data[0] or "id" in data[0]:
                return data
    except Exception:
        pass
    return None


def _extract_blocks(text: str) -> list | None:
    """Try all known Editor.js payload formats; return block list or None."""
    return _try_blocks_from_dict(text) or _try_blocks_from_array(text)


# ── HTML surgical repair ─────────────────────────────────────────────────────

def _clean_html_file(fp: Path, base_dir: Path) -> bool:
    """Replace raw Editor.js payloads inside a HTML file's elements in-place.

    The JSON often contains embedded HTML tags (<font>, <b>, <span>) inside
    text fields, so BeautifulSoup fragments the JSON. We must work on the raw
    string with regex instead.
    """
    content = fp.read_text(encoding="utf-8", errors="ignore")
    modified = False

    # Strategy 1: raw JSON between the lesson title <h1> and </article>
    # This is the most common case from save_html_asset() called with raw JSON.
    pattern = re.compile(
        r"(</h1>\s*)"       # end of the lesson title
        r"(\[.*?\]|{.*?})"  # the raw JSON payload (greedy across lines)
        r"(\s*</article>)",  # before closing article
        re.DOTALL,
    )
    m = pattern.search(content)
    if m:
        raw_json = m.group(2)
        blocks = _extract_blocks(raw_json)
        if blocks:
            result = editorjs_to_html_and_text(blocks)
            if result:
                html_out, _ = result
                content = content[:m.start(2)] + html_out + content[m.end(2):]
                modified = True

    # Strategy 2: raw JSON or Python dicts inside elements (for mixed-content files).
    # Only run BeautifulSoup if strategy 1 didn't fire, to avoid double-processing.
    if not modified:
        soup = BeautifulSoup(content, "html.parser")
        soup_modified = False
        for elem in list(soup.find_all(["p", "div", "li", "span", "pre", "code"])):
            raw_text = elem.get_text()
            blocks = _extract_blocks(raw_text)
            if not blocks:
                continue
            result = editorjs_to_html_and_text(blocks)
            if not result:
                continue
            html_out, _ = result
            replacement = BeautifulSoup(html_out, "html.parser")
            elem.replace_with(replacement)
            soup_modified = True
        if soup_modified:
            content = str(soup)
            modified = True

    if modified:
        fp.write_text(content, encoding="utf-8")
        print(f"✨ Cleaned HTML: {fp.relative_to(base_dir)}")
        log_event(_MODULE, "clean_html_done", file=str(fp.relative_to(base_dir)))

    return modified


# ── TXT surgical repair ──────────────────────────────────────────────────────

def _clean_txt_file(fp: Path, base_dir: Path) -> bool:
    """Replace raw Editor.js payloads inside a plain-text file in-place."""
    content = fp.read_text(encoding="utf-8", errors="ignore")

    # Split into lines, rebuild replacing any line that is a raw payload
    lines = content.splitlines()
    new_lines: list[str] = []
    modified = False

    for line in lines:
        blocks = _extract_blocks(line.strip())
        if not blocks:
            new_lines.append(line)
            continue

        result = editorjs_to_html_and_text(blocks)
        if not result:
            new_lines.append(line)
            continue

        _, txt_out = result
        new_lines.append(txt_out)
        modified = True

    if modified:
        fp.write_text("\n".join(new_lines), encoding="utf-8")
        print(f"✨ Cleaned TXT: {fp.relative_to(base_dir)}")
        log_event(_MODULE, "clean_txt_done", file=str(fp.relative_to(base_dir)))

    return modified


# ── Directory walker ─────────────────────────────────────────────────────────

def clean_directory(base_dir: Path) -> int:
    count = 0
    for fp in base_dir.rglob("*"):
        if not fp.is_file():
            continue
        try:
            if fp.suffix == ".html" and _clean_html_file(fp, base_dir):
                count += 1
            elif fp.suffix == ".txt" and _clean_txt_file(fp, base_dir):
                count += 1
        except Exception as exc:
            print(f"⚠️  Skipped {fp.name}: {exc}")
            log_event(_MODULE, "clean_file_error", file=fp.name, error=exc)
    return count


if __name__ == "__main__":
    downloads_dir = Path.home() / "Downloads" / "365DataScience"

    if downloads_dir.exists():
        print(f"Scanning {downloads_dir} for files to repair...")
        log_event(_MODULE, "clean_batch_start", base_dir=str(downloads_dir))
        cleaned = clean_directory(downloads_dir)
        print(f"\nDone! Repaired {cleaned} files total.")
        log_event(_MODULE, "clean_batch_done", repaired=cleaned)
    else:
        print(f"Could not find {downloads_dir}")
        log_event(_MODULE, "clean_batch_error", error=f"Base directory not found: {downloads_dir}")
