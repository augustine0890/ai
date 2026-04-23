# Site Profiles for `web_downloader.py`

Each `.env.*` file in this directory is a **complete, ready-to-use** configuration for one target site/mode. The main script always reads `dds/.env` - these files are templates you copy from.

## Available profiles

| Profile | Target site | Auth method |
|---|---|---|
| [.env.bytebytego-courses](.env.bytebytego-courses) | `bytebytego.com/courses/<slug>/` (paid content) | Cookie string from DevTools |
| [.env.bytebytego-guides](.env.bytebytego-guides) | `bytebytego.com/guides/` (public articles) | None |
| [.env.wqu](.env.wqu) | `learn.wqu.edu/my-courses/.../tasks/...` (paid lessons) | `PLAYWRIGHT_STORAGE_STATE` (captured via `capture_playwright_state.py`) |

## How to use

1. **Pick a profile** that matches the site you want to download.
2. **Copy it to `dds/.env`** (overwriting any previous config):
   ```bash
   cd dds
   cp profiles/.env.bytebytego-courses .env
   ```
3. **Edit `.env`** and fill in the credentials / URLs indicated at the top of the file.
4. **Run the downloader:**
   ```bash
   uv run python web_downloader.py
   ```

## Adding a new profile

If you support a new site, create `profiles/.env.<site-slug>` using an existing profile as a starting point. Keep the header-comment conventions (target URL, required credentials, expected output location).

## Why profiles exist

The script is unified - one crawler, one config loader. But the three supported sites differ in exactly three config axes: auth method, Playwright role (`PLAYWRIGHT_PAGE_FETCH`), and JS handling (`REMOVE_JS`). Bundling both profiles into one `.env` made the file cluttered and error-prone; splitting them makes the correct setup for each site unambiguous.
