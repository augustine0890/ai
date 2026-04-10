# web_downloader.py — Offline Site Mirror

Recursively crawls a website, downloads every HTML page and its static assets
(CSS, JS, images, fonts), rewrites all links to relative local paths, and
produces a fully browsable offline copy you can open in any browser.

---

## Quick Start

1. **Install dependencies** (if you haven't already):
   ```bash
   cd dds
   uv sync
   ```

2. **Copy the template:**
   ```bash
   cp .env.example .env
   ```

3. **Edit `.env` with your details:**
   - `URL` — the site you want to download
   - `URL_PREFIX` — restrict crawling to a path subtree (e.g. `/courses/`)
   - `COOKIE` — from DevTools Network tab (see [Getting cookies](#getting-cookie))
   - `PLAYWRIGHT=true` — if the site is a JavaScript SPA (Next.js, React, etc.)
   - `WAIT_FOR` — for SPAs, a CSS selector for the lesson/content area
   - `SEED_URLS` — for sites where chapters have no `<a>` links (see [Seed URLs](#seed-urls--sites-with-no-a-href-chapter-links))

4. **Run:**
   ```bash
   uv run python web_downloader.py
   ```

That's it. Everything is configured via `.env` — no CLI args needed. All values can still be overridden on the command line if you want.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Configuration via .env file](#configuration-via-env-file)
3. [Architecture](#architecture)
4. [How to get cookies and headers](#how-to-get-cookies-and-headers)
   - [Getting cookies](#getting-cookie)
   - [Getting a Bearer token (Authorization header)](#getting-header_authorization-bearer-token)
5. [Crawl scoping: URL_PREFIX](#crawl-scoping-url_prefix)
6. [Seed URLs — sites with no `<a href>` chapter links](#seed-urls--sites-with-no-a-href-chapter-links)
7. [Example .env files](#example-env-files)
8. [ByteByteGo / Next.js SPA — Playwright mode](#bytebytego--nextjs-spa--playwright-mode)
9. [All CLI flags](#all-cli-flags)
10. [Output structure](#output-structure)
11. [Limitations](#limitations)

---

## Configuration via .env file

All configuration lives in a `.env` file in the `dds/` directory. The script automatically loads it at startup with `python-dotenv`.

**Setup:**

```bash
cp .env.example .env
nano .env          # or your editor of choice
uv run python web_downloader.py
```

**Key variables:**

| Variable | Required | Example |
|---|---|---|
| `URL` | Yes | `https://bytebytego.com/courses/tech-resume/` |
| `URL_PREFIX` | Recommended | `/courses/` |
| `SEED_URLS` | For JS-nav sites | See [Seed URLs](#seed-urls--sites-with-no-a-href-chapter-links) |
| `COOKIE` | If behind login | `session=eyJhb...; id=123` |
| `HEADER_AUTHORIZATION` | If using Bearer token | `Bearer eyJhbGci...` |
| `HEADER_*` | Optional | Any header: `HEADER_X_CUSTOM=value` |
| `PLAYWRIGHT` | If SPA (Next.js/React) | `true` |
| `PLAYWRIGHT_STORAGE_STATE` | Recommended fallback for hard auth | `./playwright_state.json` |
| `WAIT_FOR` | For SPA content | `[class*="lesson"], article` |
| `RENDER_SETTLE_MS` | For SPA auth/state hydration | `4000` |
| `AUTO_AUTH_HEADER_FROM_COOKIE` | For cookie-only auth setups | `true` |
| `MAX_PAGES` | Optional | Unset = unlimited download |
| `THREADS` | Optional | Default `1` (safe for rate-limiting) |
| `DESTINATION` | Optional | Defaults to hostname-based folder |
| `DOWNLOAD_EXTERNAL_ASSETS` | Optional | `true` to localize CDN assets |
| `EXTERNAL_DOMAINS` | Optional | `cdn.site.com fonts.googleapis.com` |
| `USER_AGENT` | If using `cf_clearance` | Must match your browser's UA (see [cf_clearance and User-Agent](#cf_clearance-and-user-agent)) |

**Defaults:**
- `THREADS=1` (conservative, avoids rate limiting)
- `MAX_PAGES=unlimited` (crawl until no new links found)
- Everything else is empty unless set

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│  CLI (parse_args + __main__)                                             │
│  • reads --cookie / --header → injects into SESSION + Playwright context │
│  • --playwright → wraps fetch_html_rendered() as the page fetcher        │
│  • --url-prefix → restricts crawl to a URL path subtree                 │
│  • --seed-url   → pre-populates the queue with explicit chapter URLs     │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  crawl_site(start_url, root, max_pages, threads, ..., fetch_fn,         │
│             url_prefix, seed_urls)                                       │
│                                                                          │
│  q_pages (BFS queue of HTML page URLs)                                   │
│  • pre-populated with seed_urls (if any)                                 │
│  • filtered by url_prefix before enqueuing new <a href> links            │
│                                                                          │
│  download_q (asset download queue, consumed by worker threads)           │
│                                                                          │
│  Main loop per page:                                                     │
│    1. fetch_fn(page_url) → BeautifulSoup                                 │
│       ├─ fetch_html()           requests.get (fast, no JS)               │
│       └─ fetch_html_rendered()  Playwright headless Chromium (runs JS)   │
│    2. _extract_next_data_urls() → mine __NEXT_DATA__ JSON for URLs       │
│    3. Walk DOM → enqueue <a href> pages (filtered by url_prefix)         │
│       + enqueue all assets (src/href/srcset/style/<style>)              │
│    4. rewrite_links() → rewrite src/href/srcset/style to local paths     │
│    5. Write final HTML to disk                                           │
│                                                                          │
│  Worker threads (THREADS, default 1 to avoid rate limiting):             │
│    fetch_binary(url, dest)  stream binary to disk                        │
│    └─ 404/410 → WARN (asset_not_found); other errors → ERROR            │
│    └─ if .css/.js → rewrite_css_text / rewrite_js_text (local paths)    │
└──────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
               output/<host>/           (local HTML pages)
               output/<host>/cdn/       (external CDN assets)
               web_scraper.log          (DEBUG log file)
```

### Key modules

| Function | Role |
|---|---|
| `crawl_site()` | BFS page loop + asset queuing coordinator |
| `fetch_html()` | Simple requests.get → BeautifulSoup |
| `fetch_html_rendered()` | Playwright headless Chromium → BeautifulSoup (JS executed) |
| `_extract_next_data_urls()` | Mines Next.js `__NEXT_DATA__` JSON block for additional page URLs |
| `rewrite_links()` | Rewrites all URL attrs in an HTML tree to relative local paths |
| `rewrite_css_text()` | Rewrites `url(...)` and `@import` inside CSS |
| `rewrite_js_text()` | Rewrites obvious static asset strings inside JS (regex, not AST) |
| `to_local_path()` | Maps a page URL to a local `.html` file path |
| `to_local_asset_path()` | Maps an asset URL to a local file path |
| `cdn_local_path()` | Maps an external CDN URL to `cdn/<netloc>/...` |

---

## How to get cookies and headers

### Getting `COOKIE`

Cookies carry your login session. Without them, every page behind a login wall
returns a redirect to the sign-in page — you'd download the login page instead
of the actual course content.

#### Option A — Network tab (recommended, one-copy, no reconstruction)

1. **Log in** to the site in your browser.
2. **Open DevTools:**
   - Windows / Linux: `F12` or `Ctrl+Shift+I`
   - macOS: `Cmd+Option+I`
3. Click the **Network** tab.
4. **Reload the page** (`F5` / `Ctrl+R`) so requests appear in the list.
5. In the filter bar type your site domain (e.g. `bytebytego`) or select **Doc**
   to reduce noise.
6. Click **any request** to that domain in the left panel.
7. In the right panel → **Headers** sub-tab → scroll to **Request Headers**.
8. Find the `Cookie:` row — it looks like:
   ```
   Cookie: _ga=GA1.1.123; token=eyJhb...; cf_clearance=abc123; csrf-token=xyz
   ```
9. **Right-click the value → Copy value**
   *(or click in the value area → `Ctrl+A` → `Ctrl+C`)*
10. Paste it into `.env` — everything **after** `Cookie: `:
    ```env
    COOKIE=_ga=GA1.1.123; token=eyJhb...; cf_clearance=abc123; csrf-token=xyz
    ```

> ⚠ **Do NOT include the `Cookie:` prefix.** Paste only the `name=value; ...` part.  
> ⚠ **Do NOT wrap the value in quotes.**

#### Option B — From Application tab / Storage (pick individual values)

Use this if Option A doesn't show the Cookie header clearly (e.g. you've
refreshed DevTools, or the Network tab is empty).

1. **Log in** to the site in your browser.
2. DevTools → **Application** tab (Chrome / Edge) or **Storage** tab (Firefox).
3. Left sidebar → expand **Cookies** → click your site domain
   (e.g., `https://bytebytego.com`).
4. You will see a table of cookie names on the right side. It looks like:
   ```
   Name                    Value
   ─────────────────────   ──────────────────
   _ga                     GA1.1.1234567890
   _ga_JPXSGYZ0D5          GS1.1.abc...
   _tt_enable_cookie       1
   _ttp                    abc123...
   cf_clearance            VdiUXW3Zm...        ← COPY THIS VALUE
   cookieyes-consent       ...
   csrf-token              177578...           ← COPY THIS VALUE
   token                   eyJhbGci...         ← COPY THIS VALUE
   ttcsid                  ...
   ```
5. Click the row named **`token`**. Copy the long `eyJ...` string from the
   **Value** column (double-click the value cell, or look at the detail panel
   at the bottom of DevTools).
6. Click the row named **`cf_clearance`**. Copy its **Value**.
7. Click the row named **`csrf-token`**. Copy its **Value**.
8. Open your `.env` file and paste all three values in this format:
   ```env
   COOKIE=token=PASTE_TOKEN_HERE; cf_clearance=PASTE_CF_HERE; csrf-token=PASTE_CSRF_HERE
   ```
   Replace each `PASTE_..._HERE` with the value you copied. Keep the
   `token=`, `cf_clearance=`, and `csrf-token=` prefixes — only replace
   the part after the `=`.

> ⚠ **CRITICAL:** Do NOT wrap the value in quotes. Do NOT include the
> `Cookie:` prefix. Do NOT accidentally delete the `token=`, `cf_clearance=`
> or `csrf-token=` prefixes when pasting.

#### Cookie reference for bytebytego.com

Open `F12 → Application → Cookies → https://bytebytego.com` to see all of these:

| Cookie name | Required? | Expires | Purpose |
|---|---|---|---|
| `token` | ✅ Yes | ~1 hour (JWT `exp`) | Firebase login / session token |
| `cf_clearance` | ⚠️ Maybe | ~30 minutes | Cloudflare bot-challenge clearance (see note below) |
| `csrf-token` | ✅ Yes | Session | CSRF protection |
| `_ga`, `_ga_JPXSGYZ0D5` | Optional | — | Google Analytics |
| `_tt_enable_cookie`, `_ttp`, `ttcsid` | Optional | — | TikTok Pixel analytics |
| `cookieyes-consent` | Optional | — | Cookie consent banner |

> **Recommended:** Use Option A and copy the full cookie string — it includes
> everything automatically and guarantees the correct format.
>
> **⚠ Cookies expire fast!** The `token` JWT expires in ~1 hour and
> `cf_clearance` in ~30 minutes. If your download takes longer than that,
> or if you get `auth_failed_fatal`, re-copy fresh cookies and re-run.
> Already-downloaded files are skipped automatically.

#### `cf_clearance` and User-Agent

Cloudflare binds `cf_clearance` to the **exact User-Agent string** of the browser
that solved the challenge. If the downloader sends a different User-Agent,
Cloudflare rejects the request — even if the cookie itself is still valid.

**Try without `cf_clearance` first.** Many sites work with just `token` + `csrf-token`.
If Cloudflare blocks you, then include `cf_clearance` **and** set `USER_AGENT` in `.env`:

```bash
# In your browser: F12 → Console → type: navigator.userAgent → Enter → copy it
USER_AGENT=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36
```

This ensures the downloader's User-Agent matches what Cloudflare expects.

If `COOKIE` includes `cf_clearance` but `USER_AGENT` is not set, the downloader
now auto-ignores `cf_clearance` and logs `cf_clearance_ignored_no_user_agent`
to avoid common false auth failures from UA mismatch.

#### Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `auth_failed_fatal` on first page | Cookies expired or UA mismatch | Re-copy fresh cookies; if using `cf_clearance`, set `USER_AGENT` |
| `cf_clearance_ignored_no_user_agent` in log | `cf_clearance` present but no `USER_AGENT` set | Add `USER_AGENT` from browser DevTools if you intentionally need `cf_clearance` |
| Downloading the login/redirect page | Missing or expired cookie | Re-copy fresh cookies and re-run |
| `403 Forbidden` on every page | Cookie format wrong | Check: no `Cookie:` prefix, no quotes |
| Works briefly then fails | `cf_clearance` or session expired | Re-copy from DevTools and re-run (existing files are skipped) |
| Some pages redirect, others work | Only some cookies copied | Use Option A to copy the full string |
| `_debug_auth_fail.html` created | Auth check failed | Open the debug file in a browser to see what Playwright actually received |
| Cookie/header setup still shows guest page | Site also depends on browser localStorage auth state | Use `PLAYWRIGHT_STORAGE_STATE` from a real logged-in browser session |

---

### Playwright storage_state fallback (recommended for hard auth)

Some SPAs authenticate using data in browser storage (not only request cookies/headers).
When this happens, use a real Playwright `storage_state` file.

1. Generate state from a logged-in browser session:
   ```bash
   cd dds
   uv run python capture_playwright_state.py --output playwright_state.json
   ```
2. Log in in the opened browser window and confirm paid course content is visible.
3. Press Enter in terminal to save state.
4. Set in `.env`:
   ```env
   PLAYWRIGHT_STORAGE_STATE=./playwright_state.json
   ```
   With storage_state enabled, downloader avoids overlaying Playwright with
   `COOKIE`-derived auth headers/cookies to prevent stale-token conflicts.
5. Run downloader again.

### Getting `HEADER_AUTHORIZATION` (Bearer token)

Some sites — especially Next.js / React SPAs — make authenticated API calls
using a **JWT Bearer token** in the `Authorization` header, in addition to (or
instead of) cookies.

> **Note:** Not all sites use Bearer tokens. If you don't see an `Authorization`
> header in the steps below, your site uses cookies only — leave
> `HEADER_AUTHORIZATION` commented out in `.env` and skip this section.

#### Steps

1. **Log in** to the site. Open DevTools → **Network** tab (`F12`).
2. In the filter bar, click **XHR** or **Fetch** to show only API calls.
3. **Navigate to a course lesson** in the browser (or reload) so the page fires
   API requests.
4. Look for requests to API endpoints — often `/api/...`, `/graphql`, or a
   subdomain like `api.bytebytego.com`.
5. Click one of those requests in the left panel.
6. Right panel → **Headers** sub-tab → scroll to **Request Headers**.
7. Look for the `Authorization` row:
   ```
   Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOi...
   ```
8. **Copy the entire value** — including the word `Bearer` and the space after it:
   ```
   Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOi...
   ```
9. Paste it into `.env`:
   ```env
   HEADER_AUTHORIZATION=Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOi...
   ```

#### How the `HEADER_` prefix works

Any `.env` variable prefixed with `HEADER_` is automatically injected as an
HTTP request header. Underscores become hyphens:

| `.env` variable | Resulting HTTP header |
|---|---|
| `HEADER_AUTHORIZATION` | `Authorization: Bearer ...` |
| `HEADER_X_API_KEY` | `X-Api-Key: your-key` |
| `HEADER_X_CUSTOM_HEADER` | `X-Custom-Header: value` |

---

## Crawl scoping: `URL_PREFIX`

### The problem

Without scoping, the crawler follows **every** internal `<a href>` link it
finds — including site-wide navigation links (e.g. `/pricing`, `/our-team`,
`/privacy-policy`, `/guides`). Starting from a course URL like
`https://bytebytego.com/courses/tech-resume/`, the very first page's navbar
would queue the entire marketing site, wasting time and disk space.

### The fix

Set `URL_PREFIX` in `.env` to restrict the crawl to a specific path subtree:

```env
URL_PREFIX=/courses/
```

With this setting:

- ✅ `/courses/tech-resume/p1-c2-the-hiring-pipeline` → **crawled** (path starts with `/courses/`)
- ❌ `/pricing` → **skipped**
- ❌ `/our-team` → **skipped**
- ❌ `/guides/api-web-development/` → **skipped**

### How it works

When the crawler finds an `<a href>` to another internal page, it checks
whether the target URL's path starts with `_prefix` before enqueuing it:

```python
if _prefix and not parsed.path.startswith(_prefix):
    continue   # skip pages outside the allowed subtree
```

The prefix is normalised: leading `/` is always added, trailing `/` is
stripped, so `URL_PREFIX=courses`, `/courses`, `/courses/` all work the same.

### Examples

| Site type | `URL_PREFIX` value |
|---|---|
| Full site (no restriction) | *(leave unset)* |
| Course platform | `/courses/` |
| Documentation | `/docs/` |
| Blog only | `/blog/` |
| Specific course | `/courses/tech-resume/` |

### CLI equivalent

```bash
uv run python web_downloader.py --url-prefix /courses/
```

---

## Seed URLs — sites with no `<a href>` chapter links

### The problem

Some sites — notably ByteByteGo — use **JavaScript `onClick` handlers** for
their sidebar navigation instead of standard `<a href>` anchor tags. Even with
Playwright (full JS rendering), the chapter links are never rendered as `<a>`
elements in the DOM.

This means the crawler can only ever find the handful of pages explicitly
linked from the start URL, regardless of how many chapters exist.

**Confirmed for ByteByteGo:** Playwright + `networkidle` only found **2 links**
on the course index page. 17 of the 19 chapters were simply not discoverable
by any automated crawl technique.

### Why `__NEXT_DATA__` also doesn't help

The script automatically mines the `__NEXT_DATA__` JSON block that Next.js
embeds in every page. However, ByteByteGo only stores the current page's
route in `__NEXT_DATA__` — the full chapter list is fetched via a separate
API call after Playwright has already extracted the DOM. So `__NEXT_DATA__`
only ever yields the same 2 links the DOM walk finds.

> `__NEXT_DATA__` discovery *is* still useful for other Next.js sites where
> the page data does include navigation — the filtering rejects route template
> placeholders like `/courses/[course]/[...slug]` and backslash-escaped
> strings that arise from re-serialisation.

### The fix: `SEED_URLS`

Explicitly list every chapter URL in `.env`:

```env
SEED_URLS="https://bytebytego.com/courses/tech-resume/p0-acknowledgements
https://bytebytego.com/courses/tech-resume/p0-c2-introduction
https://bytebytego.com/courses/tech-resume/p1-c1-why-resumes-and-cvs-are-important
https://bytebytego.com/courses/tech-resume/p1-c2-the-hiring-pipeline
https://bytebytego.com/courses/tech-resume/p2-c3-tech-resume-basics
..."
```

These URLs are put into the crawl queue **before** the crawl starts, so every
chapter is downloaded regardless of whether the site exposes `<a>` links.
Duplicate URLs (already reachable via normal crawling) are automatically
de-duplicated.

### How to find chapter URLs for bytebytego.com

1. Log in to your browser and open the course.
2. Click each **chapter title** in the left sidebar.
3. Copy the URL from your browser's address bar after the page loads.
   It follows the pattern:
   ```
   https://bytebytego.com/courses/<course-slug>/p<N>-c<M>-<chapter-slug>
   ```
4. Paste each URL into `SEED_URLS` in `.env` (space or newline separated,
   wrapped in double-quotes for multi-line).

### dotenv multi-line syntax

`python-dotenv` supports quoted multi-line values with `\n` as the line
separator when the value is enclosed in double-quotes:

```env
# ✅ Correct — dotenv parses this as a newline-separated list
SEED_URLS="https://example.com/page-1
https://example.com/page-2
https://example.com/page-3"

# ❌ Wrong — indented continuation lines are NOT supported by dotenv
SEED_URLS=
  https://example.com/page-1
  https://example.com/page-2
```

### CLI equivalent

Use `--seed-url` (repeatable):

```bash
uv run python web_downloader.py \
  --seed-url https://bytebytego.com/courses/tech-resume/p0-acknowledgements \
  --seed-url https://bytebytego.com/courses/tech-resume/p1-c1-why-resumes-and-cvs-are-important \
  --seed-url https://bytebytego.com/courses/tech-resume/p2-c3-tech-resume-basics
```

---

## Example .env files

All examples below are `.env` files. After creating one, just run:
```bash
uv run python web_downloader.py
```

### Basic public site (no auth needed)

**.env:**
```env
URL=https://example.com/
```

### Authenticated server-rendered site

**.env:**
```env
URL=https://somesite.com/lessons/
COOKIE=session=abc123; user_id=42; auth_token=xyz
MAX_PAGES=200
THREADS=2
```

### SPA with JavaScript rendering (Next.js / React)

**.env:**
```env
URL=https://bytebytego.com/courses/tech-resume/
URL_PREFIX=/courses/
COOKIE=token=eyJ...; cf_clearance=abc...; csrf-token=xyz...
PLAYWRIGHT=true
WAIT_FOR=[class*="lesson"], article, main
REMOVE_JS=true
THREADS=1
```

### SPA with no `<a>` chapter links (seed URLs required)

**.env:**
```env
URL=https://bytebytego.com/courses/tech-resume/
URL_PREFIX=/courses/
COOKIE=token=eyJ...; cf_clearance=abc...; csrf-token=xyz...
PLAYWRIGHT=true
WAIT_FOR=[class*="lesson"], article, main
SEED_URLS="https://bytebytego.com/courses/tech-resume/p0-acknowledgements
https://bytebytego.com/courses/tech-resume/p0-c2-introduction
https://bytebytego.com/courses/tech-resume/p1-c1-why-resumes-and-cvs-are-important
https://bytebytego.com/courses/tech-resume/p1-c2-the-hiring-pipeline
https://bytebytego.com/courses/tech-resume/p2-c3-tech-resume-basics"
THREADS=1
```

### Include CDN assets (fonts, images from external hosts)

**.env:**
```env
URL=https://somesite.com/
DOWNLOAD_EXTERNAL_ASSETS=true
EXTERNAL_DOMAINS=cdn.somesite.com fonts.gstatic.com cdn.jsdelivr.net
THREADS=1
```

### Custom output folder

**.env:**
```env
URL=https://somesite.com/docs/
DESTINATION=./offline-documentation
THREADS=1
```

---

## ByteByteGo / Next.js SPA — Playwright mode

ByteByteGo is a **Next.js** / React SPA. The server sends an HTML shell with
an empty `<div id="__next">` — the actual lesson content is injected by
JavaScript after the page loads. `requests.get()` receives the shell, not the
rendered content.

**Solution:** use `PLAYWRIGHT=true`. This launches a headless Chromium browser
(via Playwright), navigates to each page, waits for the content to render,
and returns the fully populated DOM.

### Install Playwright (one-time)

```bash
uv add playwright
uv run playwright install chromium
```

### Setup `.env` for ByteByteGo (complete example)

```env
# ── Step 1: URLs ──────────────────────────────────────────────────────────
URL=https://bytebytego.com/courses/tech-resume/
URL_PREFIX=/courses/

# ── Step 2: Auth cookies (copy from F12 → Network → Cookie header) ────────
COOKIE=token=eyJ...; cf_clearance=abc...; csrf-token=xyz...

# ── Step 3: Chapter URLs (sidebar uses onClick, not <a href>) ─────────────
SEED_URLS="https://bytebytego.com/courses/tech-resume/p0-acknowledgements
https://bytebytego.com/courses/tech-resume/p0-c2-introduction
https://bytebytego.com/courses/tech-resume/p1-c1-why-resumes-and-cvs-are-important
https://bytebytego.com/courses/tech-resume/p1-c2-the-hiring-pipeline
https://bytebytego.com/courses/tech-resume/p2-c3-tech-resume-basics
https://bytebytego.com/courses/tech-resume/p2-c4-resume-structure
https://bytebytego.com/courses/tech-resume/p2-c5-standing-out
https://bytebytego.com/courses/tech-resume/p2-c6-common-mistakes
https://bytebytego.com/courses/tech-resume/p2-c7-different-experience-levels-different-career-paths
https://bytebytego.com/courses/tech-resume/p2-c8-exercises-to-polish-your-resume
https://bytebytego.com/courses/tech-resume/p2-c9-beyond-the-resume
https://bytebytego.com/courses/tech-resume/p3-c10-good-resume-template-principles
https://bytebytego.com/courses/tech-resume/p3-c11-resume-templates
https://bytebytego.com/courses/tech-resume/p3-c12-resume-improvement-examples
https://bytebytego.com/courses/tech-resume/p3-c13-advice-for-hiring-managers-on-running-a-good-screening-process
https://bytebytego.com/courses/tech-resume/p3-conclusion"

# ── Step 4: Rendering options ─────────────────────────────────────────────
PLAYWRIGHT=true
WAIT_FOR=[class*="lesson"], article, main
REMOVE_JS=true
THREADS=1
```

Run:

```bash
uv run python web_downloader.py
```

### What `WAIT_FOR` does

After navigating to each page, Playwright waits until the CSS selector in
`WAIT_FOR` appears in the DOM before extracting HTML. This ensures the
lesson body has finished rendering. Good selectors for ByteByteGo:

- `[class*="lesson"]` — any element whose class contains "lesson"
- `article` — the article container
- `main` — the main content area

Default is `body` (always present but may be too early for SPAs).

`RENDER_SETTLE_MS` adds a fixed delay after `WAIT_FOR` before HTML extraction.
For auth-heavy SPAs, this helps avoid capturing a transient guest-state DOM.

---

## All CLI flags

All of these can be set in `.env` file (recommended) or via CLI (for quick overrides).

| CLI Flag | `.env` Variable | Default | Description |
|---|---|---|---|
| `--url` | `URL` | *(required)* | Starting URL to crawl |
| `--url-prefix` | `URL_PREFIX` | *(none)* | Only crawl pages whose path starts with this prefix |
| `--seed-url` | `SEED_URLS` | *(none)* | Extra URL(s) to pre-queue (repeatable via CLI) |
| `--destination` | `DESTINATION` | Derived from URL | Output folder |
| `--max-pages` | `MAX_PAGES` | Unlimited | Max HTML pages to crawl |
| `--threads` | `THREADS` | `1` | Concurrent download worker threads |
| `--cookie` | `COOKIE` | — | Cookie string from browser DevTools |
| `--header` | `HEADER_*` | — | Extra request header (repeatable) |
| `--playwright` | `PLAYWRIGHT` | off | Render JS before saving HTML |
| `--playwright-storage-state` | `PLAYWRIGHT_STORAGE_STATE` | — | Load Playwright storage_state JSON (cookies + localStorage) |
| `--wait-for` | `WAIT_FOR` | `body` | CSS selector Playwright waits for |
| `--render-settle-ms` | `RENDER_SETTLE_MS` | `4000` | Extra wait after `WAIT_FOR` before snapshot |
| `--auto-auth-header-from-cookie` | `AUTO_AUTH_HEADER_FROM_COOKIE` | on | Auto-add `Authorization: Bearer <token-cookie>` when missing |
| `--download-external-assets` | `DOWNLOAD_EXTERNAL_ASSETS` | off | Download and localize CDN/external assets |
| `--external-domains` | `EXTERNAL_DOMAINS` | — | Space-separated whitelist of CDN domains |

---

## Output structure

```
<destination>/
├── index.html                   ← home / start page
├── courses/
│   └── tech-resume/
│       ├── index.html           ← course TOC
│       ├── p0-acknowledgements.html
│       ├── p1-c2-the-hiring-pipeline.html
│       └── ...
├── _next/
│   └── static/                  ← Next.js bundles (JS/CSS)
└── cdn/
    ├── fonts.googleapis.com/    ← external fonts
    └── cdn.somesite.com/        ← other external assets
```

All links inside HTML/CSS/JS are rewritten to relative local paths so
you can open `index.html` directly in a browser with no server.

---

## Limitations

| Limitation | Workaround |
|---|---|
| `requests` mode cannot execute JavaScript | Use `PLAYWRIGHT=true` |
| Sites where chapter links use `onClick` (not `<a href>`) | Use `SEED_URLS` to list all chapter URLs explicitly |
| `__NEXT_DATA__` only contains the current page's route on some sites | Use `SEED_URLS` — `__NEXT_DATA__` discovery is a best-effort supplement |
| Crawl may follow unrelated nav links (pricing, about, etc.) | Set `URL_PREFIX=/courses/` to restrict to the relevant path |
| `--max-pages` caps the crawl | Increase it (`MAX_PAGES=1000`) |
| ByteByteGo may rate-limit or block headless browsers | Keep `THREADS=1`; re-run after a delay (existing files are skipped) |
| Session cookies expire | Re-copy fresh cookies from DevTools and re-run (`cf_clearance` expires in ~30 min) |
| 404 assets (broken site references) | Logged as `WARN` (`asset_not_found`) — safe to ignore |
