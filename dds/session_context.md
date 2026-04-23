# Session Context: ByteByteGo + WQU Downloader Working State

## 1. Goal

Build a downloader that can:

1. Download one protected ByteByteGo chapter and its assets so it opens offline.
2. Later download a full course from the course root URL by auto-discovering chapters.
3. Download exact `learn.wqu.edu` lesson URLs so they open offline.

This session reached a working solution for both the downloader strategy and the offline rendering issues.

---

## 2. Files That Matter

- `dds/web_downloader.py`
- `dds/.env`
- `dds/.env.example`
- `dds/web_downloader.md`
- `dds/trace.jsonl`

---

## 2b. ByteByteGo Guides Mode

The downloader also supports the public ByteByteGo guides site under
`https://bytebytego.com/guides/`, but the URL structure is easy to misread.

Important behavior:

- Guide categories and guide articles are both first-level pages under `/guides/`.
- Example category:
  - `/guides/ai-machine-learning/`
- Example article inside that category:
  - `/guides/what-is-an-ai-agent/`
- The article URL is **not** nested like:
  - `/guides/ai-machine-learning/what-is-an-ai-agent/`

What this means for config:

- Use `URL_PREFIX=/guides/`
- Use `FOLLOW_LINKS=true`
- Use `DISCOVER_CHAPTERS=false`

Reason:

- Unlike protected course sidebars, guides pages already expose normal
  `<a href>` links.
- The crawler can reach article pages by following links from the guides
  landing page and category pages.
- The old course-specific discovery logic is unnecessary for guides.

Guardrail added in code:

- `web_downloader.py` now logs targeted config warnings when the start URL is
  under `/guides/` but the config still looks like course mode
  (`DISCOVER_CHAPTERS=true`, `FOLLOW_LINKS=false`, or `URL_PREFIX=/courses/`).

---

## 2c. WQU Exact-URL Mode

The downloader now also supports `https://learn.wqu.edu/...` lesson pages, but
the working offline strategy is different from ByteByteGo.

Important behavior:

- WQU lesson pages can be captured successfully with Playwright using a real
  `PLAYWRIGHT_STORAGE_STATE` file from a logged-in browser session.
- The saved HTML source already contains the lesson body.
- But if site JavaScript is kept, the offline `file://` page reboots the app,
  fails to load JS/CSS chunks correctly, and the visible content collapses to a
  cookie-banner/minimal shell state.

What this means for config:

- Use exact URL mode:
  - `FOLLOW_LINKS=false`
  - `START_URLS=[...]`
- Use browser state auth:
  - `PLAYWRIGHT_STORAGE_STATE=./playwright_state.json`
- Use Playwright for final page capture:
  - `PLAYWRIGHT=true`
  - `PLAYWRIGHT_PAGE_FETCH=true`
- Remove JavaScript in the saved offline page:
  - `REMOVE_JS=true`

Reason:

- For WQU, the rendered HTML is already sufficient for offline reading.
- Unlike ByteByteGo, WQU does not need client hydration offline.
- Keeping app JS is harmful on `file://`.

Extra implementation detail:

- `web_downloader.py` now auto-enables `REMOVE_JS=true` for `learn.wqu.edu`
  when the user forgets to set it.
- The downloader also rewrites `xlink:href` so SVG sprite references become
  local relative paths.
- The downloader now skips non-resource `<link href>` values during asset
  enqueueing (for example canonical/alternate metadata links) and skips any
  asset response that returns HTML content-type. This prevents extra same-name
  extensionless files that looked blank when opened offline.

---

## 3. Final Working Strategy

### What did not work reliably

Using Playwright for both:

1. chapter discovery, and
2. final page HTML capture

was unstable on ByteByteGo.

Reason:

- Playwright could load the page and even receive SSR content.
- But during client hydration, the browser path often downgraded back to guest state.
- Protected pages then showed the paywall overlay (`Unlock Full Access`) or login-like behavior.

This made Playwright page capture unreliable for saved protected chapters.

### What works reliably

Use a mixed strategy:

1. **Playwright only for discovery**
   - Needed because ByteByteGo is a Next.js SPA and chapter links are not reliably exposed as normal `<a>` links.
2. **`requests` session for final HTML fetch**
   - This path stays stable for protected pages when valid cookies are present.
3. **Keep JavaScript in saved pages**
   - `REMOVE_JS=false`
   - Needed so Next.js can hydrate offline from embedded `__NEXT_DATA__`.
4. **Hide paywall overlay/buttons in offline HTML**
   - `STRIP_SELECTORS=[class*="unlockAllBtn"]`
5. **Rewrite runtime asset paths for offline mode**
   - Needed because hydrated Next.js image paths do not work correctly on `file://`.

This is the final working design.

---

## 4. Auth Lessons Learned

### Cookie-only auth in Playwright browser capture was misleading

Observed behavior from logs:

- `token` and `csrf-token` were injected successfully.
- Navigation often returned `200`.
- But after page load, the page could still fall back to guest/paywall state in the browser path.

### Stable path

The authenticated `requests` path was the stable one for protected chapter HTML.

### Current practical auth guidance

1. Refresh `COOKIE` regularly because the token expires quickly.
2. If `cf_clearance` is included, `USER_AGENT` must match the browser.
3. Keep `AUTO_AUTH_HEADER_FROM_COOKIE=true`.
4. Use `PLAYWRIGHT_STORAGE_STATE` only as a fallback if needed later.

### WQU auth guidance

1. Do not try to reconstruct WQU login from Application-tab tracking cookies.
2. Use `capture_playwright_state.py` and log in manually in the opened browser.
3. `playwright_state.json` is the primary WQU auth mechanism.
4. The saved Playwright storage state may contain mostly analytics cookies in
   the plain cookie list, but the Playwright browser context can still open the
   authenticated lesson correctly.

---

## 5. Discovery Lessons Learned

### Problem

ByteByteGo chapter discovery is not a simple DOM crawl problem.

- The course root can redirect to a free intro page.
- `__NEXT_DATA__` on that page may expose only a partial set.
- Sidebar chapter entries are stored in JS-driven structures, not just normal links.

### Final discovery behavior

`DISCOVER_CHAPTERS` now works by **merging multiple sources**:

1. JSON API responses captured during page load
2. DOM links
3. sidebar `data-menu-id` values
4. `__NEXT_DATA__`

Important:

- Discovery must merge all sources.
- It must not trust the first non-empty source.
- Discovery must support **multi-level lesson hierarchies**, not just
  `course/chapter`.

### Hierarchy update

ByteByteGo now also exposes deeper lesson URLs such as:

- `/courses/coding-patterns/two-pointers/`
- `/courses/coding-patterns/two-pointers/next-lexicographical-sequence`

The original extractor mostly relied on already-built root-relative strings.
That was too narrow for newer payloads where routes are represented as
structured objects like:

```json
{"course":"coding-patterns","slug":["two-pointers","next-lexicographical-sequence"]}
```

The downloader was updated to reconstruct URLs from `course + slug[]`,
`rootPath + slug[]`, and `query.course + query.slug[]` shapes inside JSON.

This makes chapter discovery and `__NEXT_DATA__` mining robust for arbitrary
depth under a course path.

This is now implemented and working.

---

## 6. Offline Rendering Lessons Learned

After the auth/capture strategy was fixed, the remaining issue was offline images.

### Root cause

Saved chapter HTML still contained asset references that broke after offline hydration:

1. `/_next/image?...` optimizer URLs do not work on `file://`
2. some `/media/...` paths needed remapping to `_next/static/media/...`
3. chapter screenshot image paths were still embedded inside `__NEXT_DATA__`
4. some valid local images still failed because they remained `loading="lazy"` offline

### Final image solution

Implemented in `web_downloader.py`:

1. **Runtime asset fixup script**
   - rewrites `/_next/image?...`
   - rewrites `/media/...`
   - rewrites root-relative asset paths for offline use
2. **Structured `__NEXT_DATA__` asset rewriting**
   - rewrites asset references inside JSON payloads
   - queues those assets for download
3. **Force eager image loading offline**
   - convert lazy images to eager
   - retrigger `src` / `srcset` assignment

This solved the last offline image issue.

### WQU offline rendering issue

Observed behavior:

- The saved WQU HTML file looked "blank" when opened offline.
- But inspection showed the lesson content was physically present in the HTML.
- A headless browser check showed:
  - `main.innerText` became empty after load when JS was kept
  - visible text collapsed to the cookie banner / shell
  - console errors referenced offline chunk loads such as `file:///assets/...`

Root cause:

- WQU client-side scripts rehydrated on `file://`
- then attempted to load app chunks/root-relative assets in a way that broke
  offline rendering
- the app shell replaced the static lesson body in the visible DOM

Final WQU fix:

1. Save WQU pages with `REMOVE_JS=true`
2. Do not inject the downloader runtime asset-fixup script when `REMOVE_JS=true`
3. Rewrite `xlink:href` statically in HTML

Result:

- offline WQU pages keep the rendered lesson body visible
- the main content remains readable on `file://`
- some SVG sprite icons may still warn under `file://`, but the lesson content
  now renders correctly

---

## 7. Key Code Changes

### In `web_downloader.py`

- Added better auth diagnostics:
  - token lifetime logs
  - cookie probes
  - request/xhr debug
- Scoped Playwright `Authorization` to first-party hosts only
- Prevented leaking bearer auth to external asset downloads
- Added explicit URL mode:
  - `START_URLS`
  - `FOLLOW_LINKS=false`
- Added `DISCOVER_CHAPTERS`
- Added `PLAYWRIGHT_PAGE_FETCH`
- Added structured JSON route extraction for deeper hierarchy URLs
- Added runtime hide CSS injection for paywall buttons
- Added runtime offline asset fixup injection
- Added structured `__NEXT_DATA__` asset rewriting
- Added eager image handling for offline pages
- Added flexible `.env` list parsing for JSON/comma URL lists
- Added `PLAYWRIGHT_STORAGE_STATE` cookie import into the requests session
- Added `learn.wqu.edu` site-mode hints
- Added automatic `REMOVE_JS=true` adjustment for `learn.wqu.edu`
- Added `xlink:href` rewriting for offline SVG sprite paths
- Skipped runtime asset-fixup injection when `REMOVE_JS=true`
- Skipped non-resource `<link href>` metadata URLs during asset enqueueing
- Skipped asset writes when response content-type is HTML

### In `.env.example`

Documented:

- `START_URLS`
- `PLAYWRIGHT_PAGE_FETCH`
- `REMOVE_JS=false`
- `STRIP_SELECTORS`
- `FOLLOW_LINKS`
- `AUTH_DEBUG`
- WQU exact-URL example
- `DESTINATION=~/Downloads/...`
- `REMOVE_JS=true` for WQU

### In `web_downloader.md`

Documentation updated to match the working strategy and recommended settings.

---

## 8. Final Recommended Config for ByteByteGo

Core settings:

```env
URL=https://bytebytego.com/courses/tech-resume/
URL_PREFIX=/courses/
DISCOVER_CHAPTERS=true
PLAYWRIGHT=true
PLAYWRIGHT_PAGE_FETCH=false
REMOVE_JS=false
STRIP_SELECTORS=[class*="unlockAllBtn"]
AUTO_AUTH_HEADER_FROM_COOKIE=true
FOLLOW_LINKS=false
AUTH_DEBUG=true
```

---

## 9. Final Recommended Config for WQU

Core settings:

```env
START_URLS=["https://learn.wqu.edu/my-courses/courses/financial-markets/modules/m-1-credit-risk-and-financing/tasks/lesson-1-saving-borrowing-lesson-notes"]
FOLLOW_LINKS=false
PLAYWRIGHT=true
PLAYWRIGHT_PAGE_FETCH=true
PLAYWRIGHT_STORAGE_STATE=./playwright_state.json
REMOVE_JS=true
WAIT_FOR=main, article
THREADS=1
DESTINATION=~/Downloads/wqu
```

Capture browser state first:

```bash
cd dds
uv run python capture_playwright_state.py \
  --url https://learn.wqu.edu/my-courses/ \
  --output playwright_state.json
```

Notes:

- `URL` is optional when `START_URLS` is set and `FOLLOW_LINKS=false`.
- `FOLLOW_LINKS=false` is good for predictable queue behavior when using discovered/seeded URLs.
- `COOKIE` must be refreshed when expired.
- `USER_AGENT` is needed only if `cf_clearance` is used.

---

## 9. Verified Result

Final verification was done with a fresh run after the image fix.

### Verified behavior

- chapter discovery completed successfully
- protected pages downloaded successfully
- saved chapter HTML opened offline
- chapter screenshots and other images rendered correctly offline

### Important concrete verification

Protected chapter:

- `courses/tech-resume/p2-c5-different-experience-levels-different-career-paths.html`

Offline verification result:

- **30 / 30 images loaded successfully**

This confirmed the end-to-end downloader workflow is working.

---

## 10. Remaining Minor Note

There is still a harmless log entry in some runs:

- `asset_not_found url='https://bytebytego.com/script.js' status=404`

This did **not** block successful offline rendering of the saved course pages.

Treat it as non-blocking unless future work proves otherwise.

---

## 11. Recommended Next Session Starting Point

If another LLM continues from here, it should assume:

1. The current strategy is correct.
2. Do **not** revert back to Playwright full page capture for protected chapters unless there is a very strong reason.
3. Prefer debugging from this working baseline.

Recommended next-step areas:

1. cleanup/refactor `web_downloader.py`
2. improve docs/examples further
3. optional safer handling of stale `cf_clearance`
4. broader testing on more nested course hierarchies and other sites

---

## 11b. SingleFile Output Mode

The downloader now supports an alternative save format inspired by
[SingleFile](https://www.getsinglefile.com/): every page becomes one
self-contained `.html` file with all CSS, JS, images, and fonts inlined as
`data:` URIs or inline `<style>` / `<script>` blocks.

Enable with:

```env
SINGLE_FILE=true
```

Or pass `--single-file` on the CLI.

Behavior:

- Asset download queue / sidecar folders are bypassed entirely; assets are
  fetched by the shared `requests` session and inlined at save time.
- CSS `url(...)` and `@import` are resolved recursively (depth limited to 3).
- `<link rel="preload">` and `modulepreload` are dropped so no network fetch
  happens when the file is opened.
- Inter-page `<a href>` links are **not** rewritten — each SingleFile page is
  a standalone document.
- Works with every existing auth mechanism (`COOKIE`, `HEADER_*`,
  `PLAYWRIGHT_STORAGE_STATE`). Same SESSION fetches both the HTML and the
  inlined assets.
- First-party `Authorization` headers are stripped when inlining third-party
  CDN assets (mirrors `fetch_binary` safety).
- `REMOVE_JS=true` still works: scripts are dropped instead of inlined.
- Unresolvable assets (404, HTML redirects) are left in place — the page
  degrades gracefully. Per-page telemetry goes to `trace.jsonl` as
  `single_file_inlined inlined=N failed=N unique_assets=N`.

Key code added to `web_downloader.py`:

- `MIME_BY_EXT`, `_guess_mime_type`, `_to_data_uri`
- `_fetch_asset_bytes` (shared fetch with HTML-response guard + first-party
  Authorization scoping)
- `_inline_css_text` (recursive `url()` + `@import` inlining)
- `_inline_srcset`, `_inline_one_url`
- `inline_all_assets` (top-level entry point)
- New `single_file` parameter on `crawl_site`
- New CLI flag `--single-file` / env var `SINGLE_FILE`
- DOM-walk short-circuits that skip asset-queue enqueue in single-file mode

Trade-offs vs. the default sidecar mode:

- Larger per-file size (~1.3× due to base64).
- No shared assets across pages.
- But: each file is portable, shareable, and has no relative-path concerns.

---

## 12. Short Summary for Future LLMs

ByteByteGo now works with this model:

- Playwright for chapter discovery
- `requests` for final protected page HTML
- keep JS in offline HTML
- hide paywall overlay selectors
- rebuild nested course URLs from structured JSON route objects
- rewrite runtime asset paths
- rewrite `__NEXT_DATA__` image paths
- force eager offline image loading

That combination is the reason the downloader now works end to end.

For single-archive workflows, `SINGLE_FILE=true` produces one self-contained
`.html` per page with all CSS/JS/images/fonts inlined as `data:` URIs. It
runs on the same fetch/auth pipeline and is an alternative output mode, not a
replacement for sidecar mode.

---

## 13. Site Profile Files (implemented via plan.md Option A)

Introduced `dds/profiles/` with one complete `.env` template per supported site:

- `profiles/.env.bytebytego-courses` - paid courses, cookie auth, discovery on
- `profiles/.env.bytebytego-guides` - public guides, no auth, normal link-crawl
- `profiles/.env.wqu` - WQU lessons, storage_state auth, REMOVE_JS=true

`.env.example` is now a minimal generic template. `web_downloader.py` unchanged -
profile-switching is purely a copy-paste operation:

    cp profiles/.env.wqu .env

This resolves the prior concern that `.env` had to document all three sites at
once, which made the correct setup for each site unclear.

See `dds/profiles/README.md` for the full workflow.

---

## 14. Site Profile Templates and Plan Storage

This section absorbs the durable background that used to live at the top of
`dds/plan.md`. The intent is to keep future `plan.md` files short and
execution-focused.

### Why profiles exist

`web_downloader.py` is one unified downloader. The site differences are mostly
configuration, not separate code paths:

- ByteByteGo courses: cookie auth, `DISCOVER_CHAPTERS=true`,
  `PLAYWRIGHT_PAGE_FETCH=false`, keep JS
- ByteByteGo guides: mostly public, `FOLLOW_LINKS=true`, normal link crawl
- WQU lessons: `PLAYWRIGHT_STORAGE_STATE`, `PLAYWRIGHT_PAGE_FETCH=true`,
  `REMOVE_JS=true`

Putting all of that into one giant `.env.example` made the correct setup hard to
see. Splitting it into committed profile templates makes the workflow explicit.

### Runtime behavior remains unchanged

No loader behavior changed:

- `web_downloader.py` still calls `load_dotenv()` and reads `dds/.env`
- there is still no `--env-file`
- profile files under `dds/profiles/` are templates, not runtime inputs

The user flow is:

```bash
cd dds
cp profiles/.env.wqu .env
uv run python web_downloader.py
```

That means `PLAYWRIGHT_STORAGE_STATE=./playwright_state.json` still works
normally. It is read from the copied `.env`, then used by the existing
Playwright/session setup code.

### Profile design

The profile set now follows these rules:

- `dds/.env.example` stays minimal and generic
- each `dds/profiles/.env.*` file is complete for one supported site or mode
- each profile starts with enough header comments to explain auth, required
  edits, and expected output
- real secrets stay only in `dds/.env` and `dds/playwright_state.json`, both of
  which remain ignored

### Directory shape

```text
dds/
|- .env
|- .env.example
|- profiles/
|  |- README.md
|  |- .env.bytebytego-courses
|  |- .env.bytebytego-guides
|  `- .env.wqu
`- web_downloader.py
```

### Where planning content should live

Use the files like this going forward:

- `session_context.md`: stable background, goals, design rationale, decisions,
  and lessons learned
- `plan.md`: the current actionable checklist, verification steps, rollback,
  and status only

That split keeps long-lived context in one place without turning every new plan
into a large design document.
