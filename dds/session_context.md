# Session Context: ByteByteGo Downloader Final Working State

## 1. Goal

Build a downloader that can:

1. Download one protected ByteByteGo chapter and its assets so it opens offline.
2. Later download a full course from the course root URL by auto-discovering chapters.

This session reached a working solution for both the downloader strategy and the offline rendering issues.

---

## 2. Files That Matter

- `dds/web_downloader.py`
- `dds/.env`
- `dds/.env.example`
- `dds/web_downloader.md`
- `dds/trace.jsonl`

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
- Added runtime hide CSS injection for paywall buttons
- Added runtime offline asset fixup injection
- Added structured `__NEXT_DATA__` asset rewriting
- Added eager image handling for offline pages

### In `.env.example`

Documented:

- `START_URLS`
- `PLAYWRIGHT_PAGE_FETCH`
- `REMOVE_JS=false`
- `STRIP_SELECTORS`
- `FOLLOW_LINKS`
- `AUTH_DEBUG`

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

Notes:

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
4. optional broader testing on other courses/sites

---

## 12. Short Summary for Future LLMs

ByteByteGo now works with this model:

- Playwright for chapter discovery
- `requests` for final protected page HTML
- keep JS in offline HTML
- hide paywall overlay selectors
- rewrite runtime asset paths
- rewrite `__NEXT_DATA__` image paths
- force eager offline image loading

That combination is the reason the downloader now works end to end.
