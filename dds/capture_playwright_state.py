#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from logger import log_event
from playwright.sync_api import Error, sync_playwright

_MOD = "dds.capture_playwright_state"


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
    log_event(
        _MOD,
        "capture_state_start",
        url=args.url,
        output=str(out),
        user_agent_set=bool(args.user_agent),
    )

    try:
        with sync_playwright() as pw:
            try:
                browser = pw.chromium.launch(headless=False)
                log_event(_MOD, "browser_launch_success", browser="chromium", headless=False)
            except Error as exc:
                hint = None
                if "Executable doesn't exist" in str(exc):
                    hint = "Run: uv run playwright install chromium"
                log_event(
                    _MOD,
                    "browser_launch_error",
                    browser="chromium",
                    hint=hint,
                    error=exc,
                )
                raise

            ctx_kwargs = {"user_agent": args.user_agent} if args.user_agent else {}
            context = browser.new_context(**ctx_kwargs)
            log_event(
                _MOD,
                "browser_context_created",
                user_agent_set=bool(args.user_agent),
            )
            try:
                page = context.new_page()
                page.goto(args.url, wait_until="domcontentloaded")
                log_event(_MOD, "page_opened", url=args.url)

                print("")
                print("Browser opened.")
                print("1) Log in normally in the opened browser window.")
                print("2) Navigate to the target course page and confirm paid content is visible.")
                print("3) Return here and press Enter to save storage state.")
                log_event(_MOD, "waiting_for_manual_login", url=args.url)
                input("Press Enter to save storage state... ")

                context.storage_state(path=str(out))
                log_event(_MOD, "storage_state_saved", output=str(out.resolve()))
                print(f"Saved storage state to: {out.resolve()}")
            finally:
                try:
                    context.close()
                    log_event(_MOD, "browser_context_closed")
                finally:
                    browser.close()
                    log_event(_MOD, "browser_closed")
    except KeyboardInterrupt:
        log_event(_MOD, "capture_state_interrupted", output=str(out))
        raise
    except Exception as exc:
        log_event(
            _MOD,
            "capture_state_error",
            url=args.url,
            output=str(out),
            error=exc,
        )
        raise
    else:
        log_event(_MOD, "capture_state_complete", output=str(out.resolve()))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        sys.exit(1)
