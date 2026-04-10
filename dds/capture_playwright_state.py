#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Open a real browser, let you log in manually, then save "
            "Playwright storage_state (cookies + localStorage) to JSON."
        )
    )
    p.add_argument(
        "--url",
        default="https://bytebytego.com/courses/tech-resume/",
        help="Page to open before manual login.",
    )
    p.add_argument(
        "--output",
        default="playwright_state.json",
        help="Where to write storage_state JSON.",
    )
    p.add_argument(
        "--user-agent",
        default=None,
        metavar="UA",
        help="Optional custom browser User-Agent.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False)
        ctx_kwargs = {"user_agent": args.user_agent} if args.user_agent else {}
        context = browser.new_context(**ctx_kwargs)
        page = context.new_page()
        page.goto(args.url, wait_until="domcontentloaded")

        print("")
        print("Browser opened.")
        print("1) Log in normally in the opened browser window.")
        print("2) Navigate to the target course page and confirm paid content is visible.")
        print("3) Return here and press Enter to save storage state.")
        input("Press Enter to save storage state... ")

        context.storage_state(path=str(out))
        print(f"Saved storage state to: {out.resolve()}")
        context.close()
        browser.close()


if __name__ == "__main__":
    main()

