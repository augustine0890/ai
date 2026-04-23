"""
list_courses.py

Fetches the full catalogue of 365 Data Science courses via the API and
writes the list of course URLs to a JSON file under config/ (default: courses_list.json).

Usage:
    uv run python -m dds.datascience365.list_courses
    uv run python -m dds.datascience365.list_courses --financial   # 365financialanalyst.com instead
    uv run python -m dds.datascience365.list_courses --free        # only free courses
    uv run python -m dds.datascience365.list_courses --out my_list.json

Output (courses_list.json):
    {
      "total": 129,
      "source": "https://learn.365datascience.com/",
      "course_urls": [
        "https://learn.365datascience.com/courses/python/",
        ...
      ]
    }
"""

import argparse
import json
from pathlib import Path

import requests

from dds.common.logger import log_event

_MODULE = "dds.list"


BASE_URLS = {
    "datascience":     ("https://api.365datascience.com",     "https://learn.365datascience.com"),
    "financialanalyst":("https://api.365financialanalyst.com","https://learn.365financialanalyst.com"),
}
REQUEST_TIMEOUT = 20


def load_token(input_file: Path) -> str:
    data = json.loads(input_file.read_text(encoding="utf-8"))
    token = data.get("authorization_token", "")
    if not token:
        raise ValueError("authorization_token missing in config/input.json")
    return token


def fetch_all_courses(api_base: str, token: str) -> list[dict]:
    url = f"{api_base}/courses/all"
    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list):
        raise ValueError(f"Unexpected API response shape: {type(data)}")
    return data


def build_url(slug: str, learn_base: str) -> str:
    return f"{learn_base}/courses/{slug}/"


def main() -> None:
    parser = argparse.ArgumentParser(description="List all 365 courses as JSON")
    parser.add_argument(
        "--financial", action="store_true",
        help="Use 365financialanalyst.com instead of 365datascience.com",
    )
    parser.add_argument(
        "--free", action="store_true",
        help="Only include free courses",
    )
    parser.add_argument(
        "--out", default="courses_list.json",
        help="Output filename (default: courses_list.json)",
    )
    args = parser.parse_args()

    platform = "financialanalyst" if args.financial else "datascience"
    api_base, learn_base = BASE_URLS[platform]

    input_file = Path(__file__).resolve().parents[3] / "config" / "input.json"
    token = load_token(input_file)

    print(f"Fetching course catalogue from {api_base}...")
    log_event(_MODULE, "list_fetch_start", api_base=api_base, free_only=args.free)
    try:
        courses = fetch_all_courses(api_base, token)
    except Exception as exc:
        log_event(_MODULE, "list_fetch_error", api_base=api_base, error=exc)
        raise

    if args.free:
        courses = [c for c in courses if c.get("free") or c.get("simpleFree")]
        print(f"Filtered to {len(courses)} free course(s).")

    urls = [build_url(c["slug"], learn_base) for c in courses if c.get("slug")]
    log_event(_MODULE, "list_fetch_done", total=len(urls), free_only=args.free)

    result = {
        "total": len(urls),
        "source": f"{learn_base}/courses/",
        "course_urls": urls,
    }

    out_path = Path(__file__).resolve().parents[3] / "config" / args.out
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log_event(_MODULE, "list_saved", path=str(out_path), total=len(urls))

    print(f"Found {len(urls)} course(s). Saved to {out_path.name}")
    print("\nFirst 5 URLs:")
    for url in urls[:5]:
        print(f"  {url}")
    if len(urls) > 5:
        print(f"  ... and {len(urls) - 5} more")


if __name__ == "__main__":
    main()

