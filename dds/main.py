import json
from pathlib import Path
from typing import Any, cast
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup  # pyright: ignore[reportMissingImports]

from download_single_course import download_course, download_course_resource


REQUEST_TIMEOUT_SECONDS = 20


def load_input_data(input_file: Path) -> dict[str, Any]:
    input_data = json.loads(input_file.read_text(encoding="utf-8"))

    required_fields = ("authorization_token", "policy_key", "quality")
    missing_fields = [field for field in required_fields if not input_data.get(field)]
    if missing_fields:
        missing_fields_text = ", ".join(missing_fields)
        raise ValueError(f"Missing required input.json fields: {missing_fields_text}")

    return input_data


def extract_course_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    links: list[str] = []
    seen_links: set[str] = set()

    for course in soup.find_all("div", class_="course-card-body"):
        anchor = course.find("a", href=True)
        if not anchor:
            continue

        href = anchor["href"].strip()
        if not href or href.startswith("#"):
            continue

        full_course_url = urljoin(base_url, href)
        if full_course_url in seen_links:
            continue

        seen_links.add(full_course_url)
        links.append(full_course_url)

    return links


if __name__ == "__main__":
    input_file = Path(__file__).parent / "input.json"
    input_data = load_input_data(input_file)

    base_url = input_data.get("base_url", "https://learn.365datascience.com/")
    courses_collector_path = input_data.get("courses_collector_path", "courses")
    courses_collector_url = urljoin(base_url, courses_collector_path)

    page = requests.get(courses_collector_url, timeout=REQUEST_TIMEOUT_SECONDS)
    page.raise_for_status()
    soup = BeautifulSoup(page.content, "html.parser")
    all_course_link = extract_course_links(soup, base_url)

    if not all_course_link:
        raise RuntimeError(f"No course links found at {courses_collector_url}")

    authorization_token = cast(str, input_data["authorization_token"])
    policy_key = cast(str, input_data["policy_key"])
    quality = cast(str, input_data["quality"])

    for course_url in all_course_link:
        try:
            download_course_resource(
                course_url=course_url, authorization_token=authorization_token
            )
            download_course(
                course_url=course_url,
                authorization_token=authorization_token,
                policy_key=policy_key,
                quality=quality,
            )
        except Exception as exc:
            print(f"Failed to download course {course_url}: {exc}")
