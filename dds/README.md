# dds

Simple Python downloader utilities for 365 Data Science course content.

This project includes:

- `main.py`: crawls the course listing page and tries to download all discovered courses.
- `download_single_course.py`: downloads one course and its resources.
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
   |    -> request_365datascience_course_api()
   |    -> request_365datascience_course_resource_api()
   |
   |  download_course()
   |    -> request_365datascience_course_api()
   |    -> request_brightcove_api()
   |    -> download_video_from_stream_url()
   |
   +-> per-course try/except (continue on failure)
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
   - `download_course(...)` to fetch and download lesson videos.
5. Per-course failures are isolated so one broken course does not stop the whole run.

### API integration points

- 365 Data Science player API:
  - `GET https://api.365datascience.com/courses/{course_slug}/player`
  - Parsed into `CourseModel` (`course_model.py`)
- 365 Data Science resource API:
  - `POST https://api.365datascience.com/courses/file`
  - Returns zip URLs for some courses
  - `400/404` are treated as "no resources available"
- Brightcove playback API:
  - `GET https://edge.api.brightcove.com/playback/v1/accounts/6258000438001/videos/{video_id}`
  - Parsed into `VideoModel` (`video_model.py`)

### Download engine behavior

- Video downloads use `yt-dlp`.
- If FFmpeg is available, downloader requests split formats (`bestvideo+bestaudio`) and merges/fixes streams.
- If FFmpeg is not available, downloader falls back to single-stream format (`best`) to avoid merge failure.
- Downloaded files are written to `~/Downloads/365DataScience/...`.

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

Edit `input.json` and set the required values:

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

### 5) Pydantic validation errors for course fields

Cause:
- Upstream API response shape changed.

Fix:
- Pull latest repo changes.
- If error persists, capture the exact validation block and patch models in `course_model.py`.

## Output

Downloads are saved under:

- `~/Downloads/365DataScience/...`

## Notes

- Use only with content you are authorized to access.
- Network/API changes from upstream may break scraping or download behavior.
