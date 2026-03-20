# dds

Simple Python downloader utilities for 365 Data Science course content.

This project includes:

- `main.py`: crawls the course listing page and tries to download all discovered courses.
- `download_single_course.py`: downloads videos, text lessons (HTML/TXT), and resources for a course.
- `clean_downloaded_files.py`: post-processes downloaded HTML/TXT files to clean embedded Editor.js JSON payloads.
- `course_model.py` and `video_model.py`: Pydantic models for API responses.

## Technical architecture

### Main-file component diagram (`main.py`)

```text
input.json
   |
   v
load_input_data()
   |
   v
fetch courses page -----------------------------+
   |                                            |
   v                                            |
extract_course_links()                          |
   |                                            |
   +-- empty? --> get_course_links_from_input() |
                  (course_url / course_urls)    |
                                                |
validated all_course_link <---------------------+
   |
   v
for each course_url
   |-
   |  download_course_resource()
   |    -> request_course_api()
   |    -> request_course_resource_api()
   |
   |  download_course()
   |    -> request_course_api()
   |    -> request_brightcove_api()
   |    -> download_video_from_stream_url()
   |    -> fetch_text_lesson_content()   # /course/text/{asset_id}
   |    -> save_html_asset() + .txt export
   |
   +-> per-course try/except (continue on failure)
   |
   +-> clean_directory() post-pass for malformed HTML/TXT
```

### Why this design works well

- `main.py` is orchestration-only: it validates input, discovers course URLs, and coordinates per-course execution.
- API details and download logic stay in `download_single_course.py`, keeping responsibilities separated.
- Link fallback (`course_url` / `course_urls`) makes the script robust when the listing page is login-gated.
- Per-course exception isolation allows batch runs without stopping on one bad course.

### High-level flow

1. `main.py` loads `input.json` and validates required auth/config fields.
2. It fetches the course listing page and extracts candidate course links.
3. It filters links to valid course URL shapes and falls back to `course_url` / `course_urls` from `input.json` when scraping returns nothing.
4. For each course URL, it calls:
   - `download_course_resource(...)` to fetch downloadable zip resources (if available), then
   - `download_course(...)` to fetch and download lesson videos and text lessons.
5. Text lessons are fetched from the player payload and `/course/text/{asset_id}` fallback, then saved as `.html` and `.txt`.
6. A cleanup pass normalizes any files that still contain raw stringified Editor.js payloads.
7. Per-course failures are isolated so one broken course does not stop the whole run.

### API integration points

- Course player API (domain-aware):
  - `GET https://api.365datascience.com/courses/{course_slug}/player`
  - `GET https://api.365financialanalyst.com/courses/{course_slug}/player`
  - Parsed into `CourseModel` (`course_model.py`)
- Course resource API (domain-aware):
  - `POST https://api.365datascience.com/courses/file`
  - `POST https://api.365financialanalyst.com/courses/file`
  - Returns zip URLs for some courses
  - `400/404` and `5xx` are treated as "no resources available" (skip resource download, continue course)
- Brightcove playback API:
  - `GET https://edge.api.brightcove.com/playback/v1/accounts/6258000438001/videos/{video_id}`
  - Parsed into `VideoModel` (`video_model.py`)
- Text lesson API (primary for non-video lessons):
  - `GET {api_base_url}/course/text/{asset_id}`
  - Returns Editor.js block JSON, converted to clean HTML and plain text
- Lecture content fallback API (best effort):
  - Multiple lecture endpoints are probed when both player payload and text API miss
  - Used to recover additional HTML/text-only lessons

### Download engine behavior

- Video downloads use `yt-dlp`.
- If FFmpeg is available, downloader requests split formats (`bestvideo+bestaudio`) and merges/fixes streams.
- If FFmpeg is not available, downloader falls back to single-stream format (`best`) to avoid merge failure.
- Text/non-video lessons are exported as both `.html` and `.txt`.
- Text extraction order is: player payload -> `/course/text/{asset_id}` -> lecture endpoint fallback.
- `main.py` runs a cleanup pass (`clean_downloaded_files.clean_directory`) after downloads.
- Downloaded files are written to `~/Downloads/365DataScience/...`.

### Logging format (LLM-friendly)

- Logs are structured as key/value events, e.g.:
  - `[2026-03-20T09:10:00+00:00] [dds.main] event=course_start index=1 total=3 course_url='...'`
  - `[2026-03-20T09:10:02+00:00] [dds.worker] event=lesson_content_txt_done filepath='...'`
- Prefixes:
  - `[dds.main]` orchestration and batch-level flow
  - `[dds.worker]` per-course API/download internals
- Every event is also appended to `trace.jsonl` for machine/LLM parsing.

## Technical Deep Dive: The Text Extraction & Formatting Journey

Extracting the non-video (text/HTML) lessons proved to be the most challenging part of this project. Here is the technical breakdown of what we discovered and how we solved it:

### 1. The Editor.js payload and the "Ugly JSON" problem
**What happened:** Initially, downloading text lessons resulted in HTML files that were basically unreadable on the screen. The content was displayed as raw, stringified JSON such as `[{"id":"8DSOgugSaE", "type":"header", ...}]`.
**Why it happened:** The 365 platform does not serve pre-rendered HTML for its reading lessons. Instead, it serves structured blocks created by the **Editor.js** framework. To make things worse, sometimes the platform's API returns this array of blocks cleanly, and other times it returns them as a *heavily stringified* JSON string nested inside fallback objects. When our initial universal parsing tool (`_collect_all_strings`) tried to scrape the text, it encountered the stringified JSON and naively output it directly inside a `<p>` tag, creating the ugly output.
**How we fixed it:** 
We built a robust, custom `editorjs_to_html_and_text` parser from scratch. This parser:
- Detects stringified payloads and defensively un-strings them (`json.loads()`).
- Iterates over blocks and maps specific Editor.js block types (`header`, `paragraph`, `listUnordered`, `listOrdered`, `quote`, `table`, `image`, `checklist`) into carefully structured HTML fragments.
- Accumulates consecutive per-item list blocks into correctly grouped `<ul>` or `<ol>` elements.

### 2. Beautiful HTML styling
**What happened:** The initial parser output was mathematically correct but visually drab (Times New Roman, no margins, visually exhausting to read).
**How we fixed it:** We upgraded `save_html_asset()` to wrap the generated fragments in a modern, responsive HTML5 boilerplate. It features:
- **Inter** typeface (via Google Fonts).
- A centered "card" layout with rounded corners and a soft drop-shadow.
- Custom CSS for complex blocks like zebra-striped tables, quote blocks matching the UI, interactive checklists (`☐` / `☑`), and highlighted "Learning Objectives" containers with accent colors.
- We also added a parallel Plain Text (`.txt`) generator that mimics the structure utilizing standard ASCII dividers (`═════` and `─────`) and indentation so local command-line users can read it naturally.

### 3. The `Token Expired` block wall
**What happened:** Mid-download, the text-extraction API (`/course/text/{asset_id}`) would randomly start skipping all lessons, throwing `400 Token provided is either invalid or expired`.
**Why it happened:** The main video CDN (Brightcove) relies on a long-lived `policy_key`, but the text/metadata API uses a strict JWT `authorization_token` bound to the user's login session. These JWTs expire roughly **1 hour** after being issued. Because a video course download can easily take longer than an hour, the token expires mid-loop.
**How we fixed it:** We introduced a global mutable state manager with a terminal prompt. If the API returns a 401/403 or "invalid" message mid-download, the script pauses execution, alerts you in the terminal (`⚠️ Token expired or invalid`), and waits for you to paste a new token from your browser. It then permanently updates your local `input.json` and resumes the loop exactly where it left off!

### 4. The auto-cleaning `clean_downloaded_files` pass
**What happened:** Even with the fixes in place, previously run downloads still contained the ugly stringified JSON files on disk. Expecting the user to find and delete these manually was a poor developer experience.
**How we fixed it:** We created a cleanup module that uses Regex (`\[\s*\{.*\}\s*\]`) to hunt down and intercept any lingering stringified JSON blocks inside previously written `.html` or `.txt` files. We integrated this directly into `main.py` so that when a batch download completes, it does a final, silent sweep of the `Downloads` directory, automatically converting any legacy ugly files into gorgeous, newly-styled assets without making any additional network requests.

## Requirements

- Python 3.14+ (current project setting)
- `uv` package manager
- Valid 365 Data Science credentials/tokens

## Setup

From the `dds` folder:

```bash
uv sync
```

This creates `.venv` and installs all dependencies from `pyproject.toml` / `uv.lock`.

## Configure input

Create local config from the example, then edit:

```bash
cp input.example.json input.json
```

Set required values in `input.json`:

```json
{
  "course_url": "https://learn.365datascience.com/courses/preview/web-scraping-and-api-fundamentals-in-python/",
  "authorization_token": "<YOUR_BEARER_TOKEN>",
  "policy_key": "<YOUR_BRIGHTCOVE_POLICY_KEY>",
  "quality": "1080p"
}
```

Optional keys for `main.py`:

- `base_url` (default: `https://learn.365datascience.com/`)
- `courses_collector_path` (default: `courses`)
- `course_urls` (optional list for batch input fallback)

Supported course domains include:

- `https://learn.365datascience.com/...`
- `https://learn.365financialanalyst.com/...`

Security note:

- `dds/input.json` is gitignored and should contain real secrets only on your local machine.
- Keep only placeholders in `dds/input.example.json`.

## Run

Download a single course:

```bash
uv run python download_single_course.py
```

Download all discovered courses from the listing page:

```bash
uv run python main.py
```

## FFmpeg installation guide

FFmpeg is recommended for best video quality and stream merging.

### Windows (recommended)

Install with winget:

```powershell
winget install -e --id Gyan.FFmpeg --accept-package-agreements --accept-source-agreements
```

Verify:

```powershell
ffmpeg -version
```

If `ffmpeg` is not recognized, close and reopen terminal and run verify again.

### Python fallback (already supported in this project)

This project also uses `imageio-ffmpeg` as fallback. If system FFmpeg is not found in PATH, it will try bundled FFmpeg automatically.

## Troubleshooting and common install/runtime bugs

### 1) `ffmpeg is not installed` while downloading

Cause:
- FFmpeg not in PATH, or shell not restarted after install.

Fix:
- Reopen terminal, run `ffmpeg -version`.
- Re-run `uv sync` to ensure `imageio-ffmpeg` is installed.
- Run via uv: `uv run python main.py`.

### 2) `No course links found at .../courses`

Cause:
- Listing page redirected to login or changed HTML layout.

Fix:
- Set `course_url` (or `course_urls`) directly in `input.json`.
- Ensure `authorization_token` is valid.

### 3) `404 ... /courses/{slug}/player`

Cause:
- Invalid/non-course URL used as input.

Fix:
- Use full course URL format like:
  - `https://learn.365datascience.com/courses/preview/<course-slug>/`
  - or `https://learn.365datascience.com/courses/<course-slug>/`

### 4) `400 ... /courses/file`

Cause:
- That course has no downloadable resource zip, or endpoint rejects resource request.

Fix:
- This is handled gracefully; videos should still download.

### 5) `500 ... /courses/file`

Cause:
- Upstream resource endpoint failure for that course.

Fix:
- This is handled as "no resources" and the script continues with videos/HTML.

### 6) Pydantic validation errors for course fields

Cause:
- Upstream API response shape changed.

Fix:
- Pull latest repo changes.
- If error persists, capture the exact validation block and patch models in `course_model.py`.

### 7) HTML lesson pages are missing

Cause:
- Some lessons do not expose inline `asset.text` in course payload.

Fix:
- The script fetches text via `/course/text/{asset_id}` and then tries lecture fallback URLs.
- Check logs for `lesson_content_missing` and `lesson_html_missing` to identify assets where upstream APIs returned no textual payload.

### 8) Authorization token expires during long runs

Cause:
- API responds `401/403` or "invalid or expired" mid-download.

Fix:
- Script prompts for a fresh token in terminal.
- New token is persisted to local `input.json` and reused for remaining API calls.

## Output

Downloads are saved under:

- `~/Downloads/365DataScience/...`

Typical file outputs per lesson:

- Video lesson: `.../<index> - <lesson>.mp4` (or similar extension)
- Text lesson: `.../<index> - <lesson>.html` and `.../<index> - <lesson>.txt`

## Notes

- Use only with content you are authorized to access.
- Network/API changes from upstream may break scraping or download behavior.
