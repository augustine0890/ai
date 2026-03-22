"""
redownload_html.py

Re-downloads only the HTML and TXT text-lesson content for courses listed in
input.json, overwriting existing files. Videos are never touched.

Usage:
    uv run python redownload_html.py
"""

import json
from pathlib import Path
from typing import cast

from clean_downloaded_files import clean_directory
from download_single_course import (
    ensure_parsed_html,
    fetch_text_lesson_content,
    get_api_base_url,
    normalize_name,
    request_course_api,
    request_lecture_html,
    save_html_asset,
    update_auth_token,
    prompt_new_token,
    get_auth_token,
)


def log(msg: str) -> None:
    from logger import log_event
    log_event("dds.html", msg)


def _download_html_for_course(course_url: str, authorization_token: str) -> None:
    course_slug = course_url.strip("/").split("/").pop()
    api_base_url = get_api_base_url(course_url)

    print(f"\n{'─' * 60}")
    print(f"  Course : {course_slug}")
    print(f"  API    : {api_base_url}")
    print(f"{'─' * 60}")

    course_data, _ = request_course_api(course_slug, authorization_token, api_base_url)
    base_dir = (
        Path.home() / "Downloads" / "365DataScience"
        / normalize_name(course_data.info.name)
    )

    total_saved = 0
    total_skipped = 0

    for i, section in enumerate(course_data.sections, start=1):
        print(f"\n  [{i}/{len(course_data.sections)}] {section.name}")

        for j, asset in enumerate(section.assets, start=1):
            asset_base = (
                base_dir
                / f"{i} - {normalize_name(section.name)}"
                / f"{j} - {normalize_name(asset.name)}"
            )
            token = get_auth_token(authorization_token)

            # ── Step A: inline text from player payload ──────────────────────
            html_text = asset.text
            if not html_text and asset.lecture_id:
                html_text = request_lecture_html(
                    course_slug=course_slug,
                    lecture_id=asset.lecture_id,
                    authorization_token=token,
                    api_base_url=api_base_url,
                )

            if html_text:
                html_path = asset_base.with_suffix(".html")
                save_html_asset(html_path, asset.name, ensure_parsed_html(html_text))
                total_saved += 1

            # ── Step B: /course/text/{asset_id} (non-video assets only) ─────
            has_video = bool(asset.video) and not isinstance(asset.video, bool)
            if not has_video:
                result = fetch_text_lesson_content(
                    asset_id=asset.id,
                    authorization_token=token,
                    api_base_url=api_base_url,
                )
                if result:
                    content_html, content_txt = result
                    html_out = asset_base.with_suffix(".html")
                    txt_out = asset_base.with_suffix(".txt")
                    save_html_asset(html_out, asset.name, content_html)
                    txt_out.parent.mkdir(parents=True, exist_ok=True)
                    txt_out.write_text(content_txt, encoding="utf-8")
                    total_saved += 1
                    print(f"    ✓ [{j}] {asset.name}")
                else:
                    total_skipped += 1
                    print(f"    · [{j}] {asset.name}  (no text content)")

    print(f"\n  ✅ {course_data.info.name}: {total_saved} saved, {total_skipped} skipped")


def main() -> None:
    input_file = Path(__file__).parent / "input.json"
    data = json.loads(input_file.read_text(encoding="utf-8"))

    authorization_token = data.get("authorization_token", "")
    if not authorization_token:
        raise ValueError("authorization_token missing in input.json")
    update_auth_token(authorization_token)

    # Collect course URLs — same priority as main.py
    course_urls: list[str] = data.get("course_urls", [])
    single = data.get("course_url", "")
    if not course_urls and single:
        course_urls = [single]
    if not course_urls:
        raise ValueError("No course_url or course_urls found in input.json")

    print(f"Re-downloading HTML/TXT for {len(course_urls)} course(s)...")

    for idx, course_url in enumerate(course_urls, start=1):
        print(f"\n[{idx}/{len(course_urls)}] {course_url}")
        try:
            _download_html_for_course(course_url, authorization_token)
        except Exception as exc:
            print(f"  ⚠️  Failed: {exc}")
            # Give user a chance to refresh token and retry once
            new_token = prompt_new_token(skip_label="skip this course")
            if new_token:
                try:
                    _download_html_for_course(course_url, new_token)
                except Exception as exc2:
                    print(f"  ⚠️  Still failed after token refresh: {exc2}")

    # Post-pass: clean up any raw JSON still embedded in files
    downloads_dir = Path.home() / "Downloads" / "365DataScience"
    if downloads_dir.exists():
        print("\nRunning cleanup pass...")
        cleaned = clean_directory(downloads_dir)
        print(f"Cleaned {cleaned} files with embedded raw JSON.")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
