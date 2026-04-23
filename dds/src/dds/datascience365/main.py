import json
from pathlib import Path
from typing import Any, cast
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup  # pyright: ignore[reportMissingImports]
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from dds.datascience365.download_single_course import (
    download_course,
    download_course_resource,
)
from dds.datascience365.clean_downloaded_files import clean_directory


REQUEST_TIMEOUT_SECONDS = 20


def log_event(event: str, **fields: Any) -> None:
    from dds.common.logger import log_event as _log
    _log("dds.main", event, **fields)


def load_input_data(input_file: Path) -> dict[str, Any]:
    # Centralized config validation keeps runtime failures predictable.
    input_data = json.loads(input_file.read_text(encoding="utf-8"))

    required_fields = ("authorization_token", "policy_key", "quality")
    missing_fields = [field for field in required_fields if not input_data.get(field)]
    if missing_fields:
        missing_fields_text = ", ".join(missing_fields)
        raise ValueError(f"Missing required config/input.json fields: {missing_fields_text}")

    return input_data


def extract_course_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    # This function intentionally supports both old and new page structures.
    # If course-card markup changes, generic <a href> scanning still works.
    links: list[str] = []
    seen_links: set[str] = set()

    def add_link(href: str) -> None:
        clean_href = href.strip()
        if not clean_href or clean_href.startswith("#"):
            return
        if "/courses/" not in clean_href:
            return

        full_course_url = urljoin(base_url, clean_href)
        if full_course_url in seen_links:
            return

        seen_links.add(full_course_url)
        links.append(full_course_url)

    for course in soup.find_all("div", class_="course-card-body"):
        anchor = course.find("a", href=True)
        if anchor:
            add_link(cast(str, anchor["href"]))

    for anchor in soup.find_all("a", href=True):
        add_link(cast(str, anchor["href"]))

    return [url for url in links if is_downloadable_course_url(url)]


def is_downloadable_course_url(course_url: str) -> bool:
    # Guardrail: avoid non-course URLs like /courses/ that produce invalid slugs.
    parsed = urlparse(course_url)
    path_parts = [part for part in parsed.path.split("/") if part]

    if not path_parts or path_parts[0] != "courses":
        return False

    if len(path_parts) < 2:
        return False

    if path_parts[1] == "preview":
        return len(path_parts) >= 3

    return True


def get_course_links_from_input(input_data: dict[str, Any]) -> list[str]:
    # Fallback path when scraping is blocked by login or layout changes.
    single_course_url = input_data.get("course_url")
    course_urls = input_data.get("course_urls", [])

    if (
        isinstance(single_course_url, str)
        and single_course_url
        and is_downloadable_course_url(single_course_url)
    ):
        return [single_course_url]

    if isinstance(course_urls, list):
        seen: set[str] = set()
        result: list[str] = []
        for url in course_urls:
            if isinstance(url, str) and url and is_downloadable_course_url(url) and url not in seen:
                seen.add(url)
                result.append(url)
        return result

    return []


def main() -> None:
    # Main flow: validate config -> discover URLs -> process each course independently.
    input_file = Path(__file__).resolve().parents[3] / "config" / "input.json"
    input_data = load_input_data(input_file)
    log_event("config_loaded", input_file=str(input_file))

    base_url = input_data.get("base_url", "https://learn.365datascience.com/")
    courses_collector_path = input_data.get("courses_collector_path", "courses")
    courses_collector_url = urljoin(base_url, courses_collector_path)
    log_event("course_discovery_start", collector_url=courses_collector_url)

    page = requests.get(courses_collector_url, timeout=REQUEST_TIMEOUT_SECONDS)
    page.raise_for_status()
    log_event(
        "course_discovery_http_ok",
        status_code=page.status_code,
        final_url=page.url,
    )
    soup = BeautifulSoup(page.content, "html.parser")
    all_course_link = extract_course_links(soup, base_url)
    log_event("course_links_extracted", count=len(all_course_link), source="page")

    if not all_course_link:
        all_course_link = get_course_links_from_input(input_data)
        log_event("course_links_extracted", count=len(all_course_link), source="input")

    if not all_course_link:
        raise RuntimeError(
            "No course links found from courses page and no fallback found in config/input.json. "
            "Provide course_url or course_urls in config/input.json."
        )

    authorization_token = cast(str, input_data["authorization_token"])
    policy_key = cast(str, input_data["policy_key"])
    quality = cast(str, input_data["quality"])
    log_event("download_batch_start", courses=len(all_course_link), quality=quality)

    console = Console()
    console.print(Panel.fit(
        f"[bold cyan]dds[/bold cyan]  Â·  365 Course Downloader"
        f"  Â·  [bold]{len(all_course_link)}[/bold] course(s)  Â·  [bold]{quality}[/bold]",
        border_style="cyan",
    ))

    with Progress(
        TextColumn("{task.description}"),
        BarColumn(bar_width=26, complete_style="cyan", finished_style="green"),
        TextColumn("[dim]{task.fields[note]}[/dim]"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
        redirect_stdout=False,
        redirect_stderr=False,
    ) as progress:
        total = len(all_course_link)
        course_task = progress.add_task(
            f"[bold cyan]â— Courses  [0/{total}][/bold cyan]",
            total=total,
            note="",
        )

        for index, course_url in enumerate(all_course_link, start=1):
            course_slug = course_url.strip("/").split("/").pop()
            progress.update(
                course_task,
                description=f"[bold cyan]â— Courses  [{index}/{total}][/bold cyan]",
                note=course_slug,
            )
            try:
                log_event(
                    "course_start",
                    index=index,
                    total=len(all_course_link),
                    course_url=course_url,
                )
                # Resources and videos are separated because some courses expose only one of them.
                download_course_resource(
                    course_url=course_url, authorization_token=authorization_token
                )
                download_course(
                    course_url=course_url,
                    authorization_token=authorization_token,
                    policy_key=policy_key,
                    quality=quality,
                    progress=progress,
                    course_task_id=course_task,
                )
                log_event("course_done", index=index, course_url=course_url)
            except Exception as exc:
                # Continue batch processing on per-course failures.
                log_event(
                    "course_error", index=index, course_url=course_url, error=str(exc)
                )
                progress.console.print(f"[red]âš ï¸  Failed:[/red] {course_url}\n   {exc}")
            finally:
                progress.advance(course_task)

    # Final pass: clean up any files that were downloaded with raw stringified EditorJS JSON
    downloads_dir = Path.home() / "Downloads" / "365DataScience"
    if downloads_dir.exists():
        log_event("cleanup_start", directory=str(downloads_dir))
        try:
            cleaned = clean_directory(downloads_dir)
            log_event("cleanup_done", cleaned_files=cleaned)
        except Exception as exc:
            log_event("cleanup_error", error=str(exc))


if __name__ == "__main__":
    main()


