# dds — 365 Data Science Course Downloader

A Python CLI tool that downloads videos, text lessons (HTML + TXT), and resource ZIPs from [365datascience.com](https://learn.365datascience.com) and [365financialanalyst.com](https://learn.365financialanalyst.com).

---

## Table of Contents
- `main.py`: crawls the course listing page and tries to download all discovered courses.
- `download_single_course.py`: downloads videos, text lessons (HTML/TXT), and resources for a course.
- `clean_downloaded_files.py`: post-processes downloaded HTML/TXT files to clean embedded Editor.js JSON payloads.
- `compress_courses.py`: bundles each downloaded course into a `.zip` with smart compression; optionally transcodes videos to H.265 first.
- `course_model.py` and `video_model.py`: Pydantic models for API responses.

1. [Requirements](#requirements)
2. [Setup](#setup)
3. [Configuring input.json](#configuring-inputjson)
   - [All fields reference](#all-fields-reference)
   - [How to get authorization\_token](#how-to-get-authorization_token)
   - [How to get policy\_key](#how-to-get-policy_key)
   - [How to get course\_url](#how-to-get-course_url)
4. [Running the downloader](#running-the-downloader)
5. [Token expiry mid-run](#token-expiry-mid-run)
6. [Output structure](#output-structure)
7. [Course compression](#course-compression)
8. [FFmpeg installation](#ffmpeg-installation)
9. [Logs and debugging](#logs-and-debugging)
10. [Troubleshooting](#troubleshooting)
11. [Technical architecture](#technical-architecture)
12. [Case study: fixing raw JSON in HTML files](#case-study-fixing-raw-json-in-html-files)

---

## Requirements

- Python 3.14+
- `uv` package manager
- A valid 365 Data Science (or 365 Financial Analyst) account with access to the courses you want to download
- FFmpeg (recommended, but there is an automatic Python fallback)

---

## Setup

### 1. Install uv

| Platform | Command |
|---|---|
| macOS (Homebrew) | `brew install uv` |
| macOS / Linux | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Windows | `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 \| iex"` |

### 2. Install dependencies

From the `dds/` folder:

```bash
uv sync
```

This creates `.venv` and installs all packages from `pyproject.toml`.

### 3. Create your config file

```bash
cp input.example.json input.json
```

Then fill in `input.json` — see the next section for every field.

---

## Configuring input.json

`input.json` is the single config file the downloader reads at startup. It is gitignored so your tokens never leave your machine.

### Minimal example (single course)

```json
{
  "authorization_token": "eyJhbGci...",
  "policy_key": "BCpkADawqM...",
  "quality": "1080p",
  "course_url": "https://learn.365datascience.com/courses/preview/python-programmer-bootcamp/"
}
```

### Full example (batch — all courses)

```json
{
  "authorization_token": "eyJhbGci...",
  "policy_key": "BCpkADawqM...",
  "quality": "1080p",
  "base_url": "https://learn.365datascience.com/",
  "courses_collector_path": "courses",
  "course_urls": [
    "https://learn.365datascience.com/courses/preview/sql/",
    "https://learn.365datascience.com/courses/python/"
  ]
}
```

---

### All fields reference

| Field | Required | Default | Description |
|---|---|---|---|
| `authorization_token` | **Yes** | — | JWT Bearer token for the 365 API. Expires ~1 hour after login. |
| `policy_key` | **Yes** | — | Brightcove playback policy key. Long-lived — rarely changes. |
| `quality` | **Yes** | — | Preferred video resolution: `"1080p"`, `"720p"`, `"480p"`, `"360p"`. |
| `course_url` | No | — | Single course URL. Used when `main.py` cannot scrape the listing page. |
| `course_urls` | No | `[]` | List of course URLs. Same fallback purpose as `course_url`. |
| `base_url` | No | `https://learn.365datascience.com/` | Base domain for the listing page scrape. |
| `courses_collector_path` | No | `courses` | Path appended to `base_url` to find the all-courses listing page. |

> `course_url` and `course_urls` are **fallbacks**. `main.py` first tries to scrape all course links from the listing page. If that page is login-gated or returns no links, it falls back to these fields.

---

### How to get `authorization_token`

This is the JWT your browser sends to the 365 API. It expires roughly **1 hour** after you log in.

**Steps:**

1. Log in to [learn.365datascience.com](https://learn.365datascience.com) in your browser.
2. Open **DevTools** (`F12` or `Cmd+Option+I` on macOS).
3. Go to the **Network** tab and make sure recording is on.
4. Open any course or lesson page so the browser makes API calls.
5. In the filter box type `api.365datascience.com` (or `api.365financialanalyst.com`).
6. Click any request to that domain.
7. Go to **Headers** → **Request Headers**.
8. Find the `Authorization` header. Its value looks like:
   ```
   Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
   ```
9. Copy everything **after** the word `Bearer ` (the long `eyJ...` string).
10. Paste it as the `authorization_token` value in `input.json`.

> **Tip — faster alternative:** In DevTools, go to the **Application** tab → **Local Storage** → `https://learn.365datascience.com`. Look for a key named `token`, `access_token`, or `auth`. The value is your JWT.

---

### How to get `policy_key`

This is a Brightcove playback key used to fetch video stream URLs. It is long-lived and rarely changes.

**Steps:**

1. Log in and open any **video lesson** (the key only appears when a video is loaded).
2. Open **DevTools** → **Network** tab.
3. In the filter box type `brightcove`.
4. Play the video (or wait for it to load) — a request to `edge.api.brightcove.com` will appear.
5. Click that request → **Headers** → **Request Headers**.
6. Find the `Accept` header. It looks like:
   ```
   application/json;pk=BCpkADawqM0T8lW3nMChuAbrcunBBLn...
   ```
7. Copy everything **after** `pk=` — the full `BCpkAD...` string.
8. Paste it as the `policy_key` value in `input.json`.

**Alternative — search the page source:**

1. DevTools → **Sources** tab (Chrome) or **Debugger** (Firefox).
2. Press `Ctrl+Shift+F` to search all files.
3. Search for `BCpk` — Brightcove policy keys always start with this prefix.
4. Copy the full string.

---

### How to get `course_url`

1. Navigate to the course page on the 365 platform.
2. Copy the URL from the browser address bar.

Supported URL formats:

```
https://learn.365datascience.com/courses/preview/<course-slug>/
https://learn.365datascience.com/courses/<course-slug>/
https://learn.365financialanalyst.com/courses/preview/<course-slug>/
```

> The downloader is domain-aware: it automatically routes API calls to `api.365datascience.com` or `api.365financialanalyst.com` based on the URL you provide.

---

## Running the downloader

### Download all courses (auto-discovered from listing page)

```bash
uv run python main.py
```

`main.py` scrapes the `/courses` listing page, extracts all course links, and downloads each one in sequence. If the listing page is login-gated, it falls back to `course_url` / `course_urls` in `input.json`.

## Course Compression

Once courses are downloaded, you can bundle each folder into a single `.zip` archive for easy storage or sharing.

Run the compression utility from the `dds` folder:

```bash
uv run python compress_courses.py
```

### Compression features

- **Smart per-format compression** — three strategies applied automatically:
  - **LZMA** for text, HTML, VTT/SRT subtitles, JSON, XML, CSV (60–90% size reduction)
  - **STORE** (no compression) for already-compressed media: MP4, MKV, MP3, JPG, PNG, PDF, ZIP … (avoids bloating and wastes no CPU)
  - **DEFLATE level 9** for everything else
- **Partial name filter** — pass a partial name to compress only matching courses
- **`--list` mode** — show file counts and uncompressed sizes before committing

### Optional H.265 video transcoding (`--transcode`)

Before zipping, `--transcode` re-encodes every eligible video file (`.mp4`, `.mkv`, `.webm`, `.mov`, `.avi`) to H.265 using FFmpeg. This typically reduces lecture/screencast video size by **40–50%** before the zip is even created.

**Hardware encoder auto-detection** — the tool probes available encoders in priority order and selects the fastest one that actually works on your machine:

| Priority | Encoder | Platform |
|---|---|---|
| 1 | `hevc_nvenc` | NVIDIA GPU (Windows / Linux) |
| 2 | `hevc_qsv` | Intel Quick Sync (Intel iGPU) |
| 3 | `hevc_amf` | AMD GPU (Windows / Linux) |
| 4 | `hevc_videotoolbox` | Apple Silicon / macOS |
| 5 | `libx265` | Software fallback (always available, slowest) |

Each candidate is verified with a short 1-second test encode before being committed. Videos already encoded in H.265/HEVC are skipped automatically.

After all courses are processed, a transcoding summary table is printed showing per-course and total byte savings.

### Commands

| Command | Action |
|---|---|
| `uv run python compress_courses.py` | Compress all courses |
| `uv run python compress_courses.py "SQL"` | Compress only courses matching "SQL" |
| `uv run python compress_courses.py --list` | List courses with file counts and sizes |
| `uv run python compress_courses.py --transcode` | Transcode videos to H.265, then compress all courses |
| `uv run python compress_courses.py --transcode "SQL"` | Transcode + compress matching course only |
| `uv run python compress_courses.py --transcode --crf 28` | Transcode with custom CRF value (default: 26) |

> **CRF guidance:** lower values = higher quality / larger file (18–24 for archival), higher values = smaller file / lower quality (28–32 for space-saving). Default 26 is a good balance for lecture content.

## FFmpeg installation guide

### Download a single course

Set `course_url` in `input.json`, then:

```bash
uv run python download_single_course.py
```

### Re-clean already downloaded files

If you have previously downloaded files that contain raw Editor.js JSON:

```bash
uv run python clean_downloaded_files.py
```

This scans `~/Downloads/365DataScience/` and rewrites any malformed `.html` / `.txt` files in-place — no network requests needed.

---

## Token expiry mid-run

The `authorization_token` (JWT) expires roughly **1 hour** after you log in. During a long batch run, the token may expire before all courses finish downloading. When this happens the downloader pauses automatically and prints:

```
⚠️  [Course API] Token expired or invalid: ...

  → Open  /path/to/dds/input.json
  → Replace the value of "authorization_token" with your new token
  → Press Enter when done, or type 's' to abort:
```

**What to do:**

1. Go back to your browser and get a fresh token using the steps in [How to get authorization_token](#how-to-get-authorization_token).
2. Open `input.json` in your editor and replace `authorization_token` with the new value. Save the file.
3. Press **Enter** in the terminal.

The downloader reads the new token from `input.json`, updates its in-memory state, and **resumes from where it left off** — no restart, no re-downloading.

> Typing `s` (then Enter) skips the current asset or aborts the current course API call, depending on where the expiry was detected.

---

## Output structure

All downloads go to `~/Downloads/365DataScience/`:

```
~/Downloads/365DataScience/
└── <Course Name>/
    ├── <course-slug>_0.zip          ← resource ZIP (if available)
    ├── 1 - <Section Name>/
    │   ├── 1 - <Lesson Name>.mp4    ← video lesson
    │   ├── 2 - <Lesson Name>.html   ← text lesson (styled HTML)
    │   └── 2 - <Lesson Name>.txt    ← text lesson (plain text)
    └── 2 - <Section Name>/
        └── ...
```

Each text lesson is exported in two formats:
- **`.html`** — styled with Inter font, card layout, syntax-highlighted code blocks, tables, checklists, quotes
- **`.txt`** — clean plain text with ASCII section dividers, suitable for reading in a terminal or feeding to an LLM

---

## FFmpeg installation

FFmpeg is recommended for best video quality. Without it the downloader falls back to single-stream format (slightly lower quality) via the bundled `imageio-ffmpeg` package.

### macOS

```bash
brew install ffmpeg
```

### Windows

```powershell
winget install -e --id Gyan.FFmpeg --accept-package-agreements --accept-source-agreements
```

After install, close and reopen your terminal, then verify:

```bash
ffmpeg -version
```

### Python fallback (automatic)

If FFmpeg is not found in PATH, the downloader automatically tries the `imageio-ffmpeg` bundled binary. No extra steps required.

---

## Logs and debugging

Every run writes structured logs to `trace.jsonl` (in the `dds/` folder) and prints them to the terminal.

### Log entry format

```json
{
  "ts":      "2026-03-21T10:30:00Z",
  "seq":     42,
  "session": "a1b2c3d4",
  "level":   "ERROR",
  "module":  "dds.worker",
  "event":   "course_error",
  "data":    { "index": 3, "course_url": "https://..." },
  "error":   "401 Unauthorized"
}
```

| Field | Description |
|---|---|
| `ts` | UTC timestamp |
| `seq` | Monotonic counter — use to reconstruct exact event order |
| `session` | 8-character hex ID that groups all events from one run |
| `level` | `INFO`, `WARN`, or `ERROR` — filter without reading event names |
| `module` | `dds.main`, `dds.worker`, `dds.compress`, `dds.clean`, or `dds.list` — identifies which script emitted the event |
| `event` | Snake-case event name describing what happened |
| `data` | All context fields (course URL, filepath, asset index, etc.) |
| `error` | Present only on errors — the error message string |

The file is automatically trimmed to the **100 most recent entries** so it stays readable without growing unbounded.

### Sharing logs with an LLM for debugging

To debug a failed run, paste the relevant portion of `trace.jsonl` into your LLM conversation. The most useful approach:

1. Open `trace.jsonl`.
2. Find the `session` value from the run you want to debug (it appears in every line).
3. Filter lines by that session:
   ```bash
   grep '"session": "a1b2c3d4"' trace.jsonl
   ```
4. Paste those lines with a question like: *"This is my run log. Course 3 failed — what went wrong and how do I fix it?"*

Key events to look for when debugging:

**Downloader events (`dds.main` / `dds.worker`)**

| Event | Meaning |
|---|---|
| `course_error` | A whole course failed — check `error` field |
| `lesson_content_missing` | Text lesson had no content in any API |
| `lesson_html_missing` | HTML lesson content not found |
| `text_lesson_api_skip` | `/course/text/{id}` returned non-200, asset skipped |
| `resource_api_empty` | Course has no downloadable ZIP (normal for many courses) |
| `video_download_start` / `done` | Video download lifecycle |

**Compression / transcoding events (`dds.compress`)**

| Event | Meaning |
|---|---|
| `compress_batch_start` / `compress_batch_done` | Overall zip run started / finished |
| `compress_course_done` | One course zip completed — includes `original_bytes` / `compressed_bytes` |
| `compress_course_error` | A course zip failed — check `message` field |
| `transcode_batch_start` | Transcoding run started — includes encoder label and CRF |
| `transcode_video_done` | One video transcoded — includes `original_bytes`, `new_bytes`, `saved_bytes` |
| `transcode_course_done` | All videos in a course transcoded — totals `before_bytes` / `after_bytes` |
| `transcode_ffmpeg_error` | FFmpeg exited non-zero — `stderr_tail` contains the last 400 chars of ffmpeg output |
| `transcode_timeout` | FFmpeg exceeded the 1-hour per-file ceiling |
| `transcode_zero_output` | FFmpeg exited 0 but produced an empty output file |
| `transcode_rename_error` | Original deleted but rename of `.tmp.mp4` failed — file may be lost |
| `transcode_file_missing_before` | Source file disappeared before transcoding started |
| `transcode_file_missing_after` | Source file missing after a failed transcode attempt |

**HTML/TXT repair events (`dds.clean`)**

| Event | Meaning |
|---|---|
| `clean_batch_start` | Scan started — includes `base_dir` |
| `clean_batch_done` | Scan finished — includes `repaired` count |
| `clean_batch_error` | Base directory not found |
| `clean_html_done` | One HTML file repaired — includes `file` path |
| `clean_txt_done` | One TXT file repaired — includes `file` path |
| `clean_file_error` | A file was skipped due to an exception |

**Course listing events (`dds.list`)**

| Event | Meaning |
|---|---|
| `list_fetch_start` | API call started — includes `api_base` and `free_only` |
| `list_fetch_done` | API call succeeded — includes `total` course count |
| `list_fetch_error` | API call failed — check `error` field |
| `list_saved` | Output JSON written — includes `path` and `total` |

---

## Troubleshooting

### `No course links found`

The listing page redirected to login or changed its HTML structure.

**Fix:** Set `course_url` or `course_urls` directly in `input.json`.

### `404 on /courses/{slug}/player`

Invalid or non-existent course slug.

**Fix:** Use a full course URL from the browser address bar:
```
https://learn.365datascience.com/courses/preview/<course-slug>/
```

### `401` / `403` / `Token expired`

Your `authorization_token` is invalid or has expired.

**Fix:** [Get a fresh token](#how-to-get-authorization_token) and update `input.json`. If this happens mid-run, the downloader will pause and prompt you — see [Token expiry mid-run](#token-expiry-mid-run).

### `400` or `404` on `/courses/file`

That course has no downloadable resource ZIP. This is expected for most courses.

**Fix:** Nothing to do — the downloader treats this as "no resources" and continues with videos and text lessons.

### `500` on `/courses/file`

Upstream resource endpoint failure.

**Fix:** Treated automatically as "no resources" — videos and text lessons still download.

### HTML lessons are missing or empty

The lesson has no inline `text` in the course payload.

**Fix:** The downloader automatically tries `/course/text/{asset_id}` and then lecture fallback endpoints. If all return nothing, the asset is logged as `lesson_content_missing`. Check your `trace.jsonl` for those events.

### HTML files contain raw JSON

Older downloads may have been saved before the Editor.js parser was in place.

**Fix:**
```bash
uv run python clean_downloaded_files.py
```

### Pydantic validation errors

The upstream API changed its response shape.

**Fix:** Pull the latest repo version. If the error persists, capture the raw API response and patch `course_model.py`.

---

## Technical architecture

### System overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              dds — local machine                            │
│                                                                             │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────────────────────┐   │
│  │  input.json │───▶│   main.py    │───▶│   download_single_course.py  │   │
│  │  (config)   │    │ orchestrator │    │      download engine         │   │
│  └─────────────┘    └──────┬───────┘    └──────────────┬───────────────┘   │
│                            │                           │                   │
│  ┌─────────────┐           │            ┌──────────────┴───────────────┐   │
│  │  logger.py  │◀──────────┴────────────│   course_model.py            │   │
│  │ trace.jsonl │                        │   video_model.py             │   │
│  └─────────────┘                        │   (Pydantic data models)     │   │
│                                         └──────────────────────────────┘   │
│  ┌─────────────────────────┐                                                │
│  │  clean_downloaded_files │  (post-run pass, no network calls)            │
│  └─────────────────────────┘                                                │
│                                                                             │
│              ~/Downloads/365DataScience/  (all output files)               │
└───────────────────────────────────────┬─────────────────────────────────────┘
                                        │ HTTPS
          ┌─────────────────────────────┼──────────────────────────┐
          ▼                             ▼                          ▼
┌─────────────────────┐   ┌────────────────────────┐   ┌──────────────────────┐
│  365 Platform API   │   │  Brightcove Playback   │   │  Course Listing Page │
│ api.365datascience  │   │  edge.api.brightcove   │   │  learn.365datascience│
│    .com             │   │       .com             │   │       .com/courses   │
│                     │   │                        │   │  (HTML scraping)     │
│ /courses/{slug}/    │   │ /playback/v1/accounts/ │   └──────────────────────┘
│   player            │   │  6258000438001/videos/ │
│ /courses/file       │   │  {video_id}            │
│ /course/text/{id}   │   │                        │
│ /courses/{slug}/    │   │  Returns: HLS m3u8     │
│   lectures/{id}     │   │  stream URL            │
└─────────────────────┘   └────────────────────────┘
```

---

### Module responsibilities

| File | Lines | Responsibility |
|---|---|---|
| `main.py` | ~195 | Orchestration: config validation, course discovery via scraping, batch loop, cleanup trigger |
| `download_single_course.py` | ~1 100 | All API calls, video download with yt-dlp/ffmpeg, text parsing, file writing, token refresh |
| `logger.py` | ~100 | Structured JSONL logger, session IDs, level inference, rolling 500-line window |
| `course_model.py` | ~126 | Pydantic v1 models for the `/courses/{slug}/player` API response |
| `video_model.py` | ~82 | Pydantic v1 models for the Brightcove `/playback/v1/…` API response |
| `clean_downloaded_files.py` | ~72 | Post-run offline cleanup: regex-detects raw Editor.js JSON in `.html`/`.txt` and rewrites it |

---

### Phase 1 — Course discovery (`main.py`)

```
  ┌──────────────────────────────────────────────────────────────────┐
  │  STARTUP                                                         │
  │                                                                  │
  │  input.json ──▶ load_input_data()                               │
  │                   validates: authorization_token,                │
  │                              policy_key, quality                 │
  └──────────────────────────────┬───────────────────────────────────┘
                                 │
                                 ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  COURSE DISCOVERY                                                │
  │                                                                  │
  │  GET {base_url}/{courses_collector_path}                         │
  │       (default: learn.365datascience.com/courses)               │
  │              │                                                   │
  │              ▼                                                   │
  │  BeautifulSoup HTML parse                                        │
  │     1. scan <div class="course-card-body"> for anchors          │
  │     2. scan all <a href> as fallback                             │
  │              │                                                   │
  │              ▼                                                   │
  │  extract_course_links()  ──▶  is_downloadable_course_url()      │
  │     filters: must contain "/courses/" in path                   │
  │              must have a slug after "/courses/"                  │
  │              "/courses/preview/" needs slug after "preview"     │
  │              │                                                   │
  │     empty? ──┤                                                   │
  │              ▼                                                   │
  │  get_course_links_from_input()   ◀── course_url / course_urls   │
  │              │                        in input.json (fallback)  │
  │              ▼                                                   │
  │  all_course_link  [ url1, url2, url3, ... ]                     │
  └──────────────────────────────┬───────────────────────────────────┘
                                 │
                                 ▼
                    (Phase 2 — per-course download)
```

> **Why scraping first, fallback second:** The listing page gives all courses automatically without the user needing to collect URLs manually. When the page is behind a login wall or changes layout, the fallback ensures the script still works.

---

### Phase 2 — Per-course download (`download_single_course.py`)

Each course URL goes through this pipeline. All three tracks run sequentially per course; a failure in one does **not** stop the others.

```
  course_url  (e.g. learn.365datascience.com/courses/preview/python/)
       │
       ├─▶ get_api_base_url()      resolves to api.365datascience.com
       │                           or api.365financialanalyst.com
       └─▶ get_learn_base_url()    resolves to learn.365datascience.com


  ┌─────────────────────────────────────────────────────────────────────────┐
  │  TRACK A — Resources (ZIP files)                                        │
  │                                                                         │
  │  request_course_api()   GET /courses/{slug}/player   ──▶  CourseModel  │
  │       │                                                                 │
  │       └─▶  request_course_resource_api()                               │
  │                POST /courses/file                                       │
  │                body: { courseId, name, courseZip: true }               │
  │                │                                                        │
  │                ├── 400 / 404  ──▶  no resources, skip silently         │
  │                ├── 5xx        ──▶  treat as no resources, continue     │
  │                └── 200        ──▶  list of presigned S3 URLs           │
  │                                        │                               │
  │                                        ▼                               │
  │                              urllib.request.urlretrieve()              │
  │                              ~/Downloads/365DataScience/               │
  │                                {CourseName}/{slug}_{i}.zip             │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  TRACK B — Videos                          (asset.type == "lesson"     │
  │                                             and asset.video != False)  │
  │                                                                         │
  │  for section in CourseModel.sections:                                  │
  │    for asset in section.assets:                                         │
  │      if asset has video (VideoItem with ext_id):                       │
  │                                                                         │
  │        request_brightcove_api(video_item.ext_id, policy_key)           │
  │          GET edge.api.brightcove.com/playback/v1/accounts/             │
  │              6258000438001/videos/{ext_id}                             │
  │          Accept: application/json;pk={policy_key}                      │
  │                 │                                                       │
  │                 ▼                                                       │
  │          VideoModel  ──▶  sources[0].src  (master HLS .m3u8 URL)      │
  │                 │                                                       │
  │                 ▼                                                       │
  │          download_video_from_stream_url()                              │
  │            ┌── ffmpeg in PATH?  ──▶  bestvideo+bestaudio merged        │
  │            └── no ffmpeg?       ──▶  imageio-ffmpeg fallback binary    │
  │                                      best single-stream format         │
  │            yt_dlp.YoutubeDL(concurrent_fragment_downloads=15)         │
  │                 │                                                       │
  │                 ▼                                                       │
  │          ~/Downloads/365DataScience/{CourseName}/                      │
  │            {i} - {SectionName}/{j} - {LessonName}.mp4                 │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  TRACK C — Text lessons  (all asset types, every asset in the course)  │
  │                                                                         │
  │  (see Phase 3 — Text extraction pipeline below)                        │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  POST-RUN CLEANUP                                                       │
  │                                                                         │
  │  clean_directory(~/Downloads/365DataScience/)                          │
  │    rglob("*.html") + rglob("*.txt")                                    │
  │    regex scan for \[\s*\{  …  \}\s*\]  (stringified JSON array)       │
  │    if found ──▶ editorjs_to_html_and_text() ──▶ overwrite in-place    │
  │    (no network calls — pure local file rewriting)                      │
  └─────────────────────────────────────────────────────────────────────────┘
```

---

### Phase 3 — Text extraction pipeline

Text content for every asset is attempted through three levels, each a fallback for the previous. The first level that returns content wins; the rest are skipped.

```
  for every asset in the course
       │
       ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  LEVEL 1 — Inline payload (zero extra network calls)                   │
  │                                                                         │
  │  asset.text  (field from /courses/{slug}/player response)              │
  │  asset.lecture_id ──▶ request_lecture_html()                           │
  │    tries 3 URL patterns:                                               │
  │      /courses/{slug}/lectures/{id}                                     │
  │      /courses/lectures/{id}                                            │
  │      /lectures/{id}                                                    │
  │    accepts text/html response OR extracts HTML from JSON payload       │
  │                                                                         │
  │  result? ──▶ save_html_asset()  (styled HTML wrapper)                 │
  └────────────────────────────┬────────────────────────────────────────────┘
                               │ no result
                               ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  LEVEL 2 — Text lesson API  (primary for reading/non-video lessons)    │
  │                                                                         │
  │  fetch_text_lesson_content(asset.id)                                   │
  │    GET {api_base}/course/text/{asset_id}                               │
  │    Authorization: Bearer {token}                                       │
  │                                                                         │
  │    on 401/403 ──▶ token refresh prompt (see Token Refresh section)    │
  │                                                                         │
  │    response is parsed through 3 strategies (first match wins):         │
  │      a) editorjs_to_html_and_text()   ◀── most common                 │
  │      b) _extract_html_from_payload()  ◀── if payload is plain HTML    │
  │      c) _collect_all_strings()        ◀── last-resort text scrape     │
  │                                                                         │
  │  result? ──▶  save_html_asset()  →  {lesson}.html                     │
  │               write_text()       →  {lesson}.txt                      │
  └────────────────────────────┬────────────────────────────────────────────┘
                               │ no result
                               ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  LEVEL 3 — Missing (logged, no crash)                                  │
  │                                                                         │
  │  log_event("lesson_content_missing", asset_id=..., asset_name=...)    │
  │  continue to next asset                                                │
  └─────────────────────────────────────────────────────────────────────────┘
```

---

### Editor.js block parser (`editorjs_to_html_and_text`)

The 365 platform stores reading lesson content as **Editor.js JSON** — a flat array of typed blocks. This parser converts that into styled HTML and plain text.

```
  Raw API response: flat JSON array of block objects
  ┌──────────────────────────────────────────────────────────┐
  │  [                                                       │
  │    { "id": "abc", "type": "header",                     │
  │      "data": { "text": "What is Python?", "level": 2 }} │
  │    { "id": "def", "type": "paragraph",                  │
  │      "data": { "text": "Python is a..." }}              │
  │    { "id": "ghi", "type": "listUnordered",              │
  │      "data": { "content": "Easy to read" }}             │
  │    ...                                                   │
  │  ]                                                       │
  └──────────────────────────────────────────────────────────┘
                           │
                           ▼
            editorjs_to_html_and_text()
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    html_parts[ ]    text_parts[ ]    list-grouping state
                                      pending_list_tag
                                      pending_list_html
                                      pending_list_text

  Block type mapping:
  ┌──────────────────────────┬──────────────────────┬────────────────────────┐
  │ Editor.js block type     │ HTML output          │ Plain text output      │
  ├──────────────────────────┼──────────────────────┼────────────────────────┤
  │ header                   │ <h1>–<h6>            │ text + ═══ / ─── rule  │
  │ paragraph                │ <p>                  │ plain text             │
  │ listUnordered            │ <ul><li>…            │ • item (grouped)       │
  │ listOrdered              │ <ol><li>…            │ 1. item (grouped)      │
  │ list (standard)          │ <ul>/<ol> (all items)│ bullet/numbered list   │
  │ templateLearningObjectiv.│ <div class="learn…"> │ Learning Objectives    │
  │ quote                    │ <blockquote>         │ "text" — caption       │
  │ image                    │ <figure><img>        │ [Image: caption]       │
  │ table                    │ <table>              │ col1 │ col2 │ col3     │
  │ callOut / warning /alert │ <div class="callout">│ 💡 / ⚠️  message       │
  │ checklist                │ <ul class="checklis">│ [x] / [ ] item         │
  │ delimiter                │ <hr>                 │ ─────────────────────  │
  │ (unknown type)           │ <p> (fallback)       │ plain text fallback    │
  └──────────────────────────┴──────────────────────┴────────────────────────┘

  List-grouping behaviour:
    listUnordered / listOrdered blocks each = ONE item.
    Consecutive same-type blocks are accumulated and flushed as a single
    <ul> or <ol> when a non-list block is encountered.
    Mixed ul/ol sequences flush the pending list before starting the new one.

  Output:
  ┌──────────────────┐    ┌──────────────────────────────────────────────┐
  │  {lesson}.html   │    │  <article> styled with Inter font, card      │
  │                  │    │  layout, accent colours, responsive CSS       │
  │                  │    │  generated by save_html_asset()               │
  └──────────────────┘    └──────────────────────────────────────────────┘
  ┌──────────────────┐    ┌──────────────────────────────────────────────┐
  │  {lesson}.txt    │    │  clean readable text, headings underlined     │
  │                  │    │  with ═══/───, lists bulleted, no HTML tags   │
  └──────────────────┘    └──────────────────────────────────────────────┘
```

---

### Token refresh mechanism

```
  API call (any endpoint)
        │
        ▼
  response.status_code ∈ {401, 403}
  OR "invalid or expired" in response body?
        │ yes
        ▼
  ┌──────────────────────────────────────────────────────────┐
  │  prompt_new_token()                                      │
  │                                                          │
  │  prints: "→ Open  /path/to/input.json"                  │
  │           "→ Replace authorization_token"               │
  │           "→ Press Enter when done, or 's' to skip"     │
  │                                                          │
  │  user saves new token to input.json in their editor     │
  │  user presses Enter                                      │
  │                                                          │
  │  reads input.json ──▶ update_auth_token(new_token)      │
  │    sets module-level _CURRENT_AUTH_TOKEN                 │
  │    re-writes input.json with new value                   │
  └──────────────────────────────────────────────────────────┘
        │
        ▼
  `continue` — the while True loop retries the same request
  with the new token via get_auth_token()

  All subsequent API calls in the same run use the new token
  automatically through get_auth_token(fallback).
```

> Every API function that uses the token has a `while True` retry loop. On a successful response it `break`s out. On a 401/403 it prompts, updates the token, then loops back to retry — so no request is permanently lost due to expiry.

---

### Data models

#### `CourseModel` (from `/courses/{slug}/player`)

```
  CourseModel
  ├── id: int
  ├── slug: str
  ├── info: Info
  │     ├── name: str               ← used as the folder name on disk
  │     ├── free: bool
  │     └── paidAccess: bool
  ├── examId: int
  ├── stats: Stats
  │     ├── lectureDownloadables: int
  │     ├── lectureDownloadablesFree: int
  │     └── lectures: int
  ├── nextLecture: NextLecture
  └── sections: List[Section]
        ├── order: int              ← folder prefix (1, 2, 3 …)
        ├── name: str               ← folder name segment
        ├── duration: int
        ├── progress: Progress
        └── assets: List[Asset]
              ├── id: int           ← used for /course/text/{id}
              ├── type: str         ← "lesson", "quiz", "exam" …
              ├── name: str         ← file name on disk
              ├── lectureId: int    ← used for lecture fallback API
              ├── text: str | None  ← inline HTML if present
              ├── video: VideoItem | bool | None
              │     ├── extId: str  ← Brightcove video ID
              │     └── provider: str
              └── downloadables: List[Downloadable]
```

#### `VideoModel` (from Brightcove `/playback/v1/…`)

```
  VideoModel
  ├── id: str
  ├── name: str
  ├── sources: List[Source]
  │     ├── src: str      ← HLS master playlist URL (.m3u8)  ← used
  │     ├── type: str     ← "application/x-mpegURL"
  │     └── height: int   ← resolution
  └── text_tracks: List[TextTrack]   ← subtitle/caption tracks
```

---

### API endpoints reference

| # | API | Method | Endpoint | Auth |
|---|---|---|---|---|
| 1 | Course listing page | `GET` | `{base_url}/courses` | none (public) |
| 2 | Course player | `GET` | `{api_base}/courses/{slug}/player` | Bearer JWT |
| 3 | Course resources | `POST` | `{api_base}/courses/file` | Bearer JWT |
| 4 | Text lesson | `GET` | `{api_base}/course/text/{asset_id}` | Bearer JWT |
| 5 | Lecture content | `GET` | `{api_base}/courses/{slug}/lectures/{id}` | Bearer JWT |
| 6 | Brightcove video | `GET` | `https://edge.api.brightcove.com/playback/v1/accounts/6258000438001/videos/{video_id}` | `Accept: application/json;pk={policy_key}` |

`{api_base}` is resolved per course URL:

| Course domain | API base |
|---|---|
| `learn.365datascience.com` | `https://api.365datascience.com` |
| `learn.365financialanalyst.com` | `https://api.365financialanalyst.com` |

---

### Logging system (`logger.py`)

```
  log_event(module, event, **fields)
       │
       ├─▶ infer level:
       │     "error" in event name OR error= kwarg ──▶ ERROR
       │     "skip"/"missing"/"expired"/"warn"/"fail" ──▶ WARN
       │     everything else ──▶ INFO
       │
       ├─▶ build entry dict:
       │     { ts, seq, session, level, module, event, data, error? }
       │
       ├─▶ print to stdout (human-readable, one line)
       │
       └─▶ append to trace.jsonl
               │
               └─▶ _trim_if_needed()
                     if lines > 550:
                       keep last 500 lines
                       rewrite file

  session = uuid4().hex[:8]   generated once at import time
  seq     = monotonic int     increments with every call in the session
```

Both `main.py` and `download_single_course.py` delegate to this module via thin wrappers that fix the `module` tag (`dds.main` / `dds.worker`). All entries in a single run share the same `session` value, making it trivial to filter one run's logs out of the rolling file.

---

## Case study: fixing raw JSON in HTML files

This section documents a real bug that appeared during development and how it was resolved step by step. Reading it will help you debug similar pipeline problems in the future.

---

### The symptom

After running `redownload_html.py`, opening an `.html` file showed raw JSON instead of readable text:

```
[{"id":"abc","type":"paragraph","data":{"text":"Python is a programming..."}},
 {"id":"def","type":"header","data":{"text":"Variables","level":2}},
 {"id":"ghi","type":"attaches","data":{"file":{"url":"https://...","name":"slides.pdf"}}}]
```

The page was completely unreadable. You expected nicely formatted headings, paragraphs, and download links — you got a JSON dump instead.

---

### Step 1 — Where is the content written to disk?

The first question to answer in any pipeline bug: **at which point does the bad data enter the pipeline?**

The write path for text lessons is:

```
API response
    └─▶ asset.text  (or request_lecture_html())
           └─▶ save_html_asset(html_path, asset.name, content)
                      └─▶ fp.write_text(...)   ← disk
```

By adding a `print(repr(content[:200]))` before `save_html_asset()` you can immediately see whether the bad data comes *in* from the API or is introduced *inside* the save function.

**What was found:** `asset.text` and `request_lecture_html()` were returning raw JSON strings directly from the API. The save function was innocent — it was just faithfully writing what it received.

**Lesson:** Always find the exact point where good data turns bad before trying to fix anything. "What does the data look like just before the write?" is the fastest way to isolate a pipeline bug.

---

### Step 2 — Why was the API returning raw JSON?

The 365 platform uses [Editor.js](https://editorjs.io/) to store lesson content. Editor.js saves content as a JSON array of typed "blocks":

```json
[
  { "type": "header", "data": { "text": "My Title", "level": 2 } },
  { "type": "paragraph", "data": { "text": "Some content..." } }
]
```

The API was returning this JSON **as-is**. The code was supposed to call `editorjs_to_html_and_text()` to convert it to readable HTML, but this conversion step was missing in two places:
- When saving `asset.text`
- When saving the result of `request_lecture_html()`

**What was fixed:** Added `ensure_parsed_html()` — a small guard function that sits between the API response and `save_html_asset()`. It checks if the content looks like JSON (starts with `[` or `{`), tries to parse and render it, and only falls through to write raw text if parsing fails:

```python
def ensure_parsed_html(raw_content: str) -> str:
    stripped = raw_content.strip()
    if stripped.startswith(("[", "{")):
        try:
            payload = json.loads(stripped)
            result = editorjs_to_html_and_text(payload)
            if result:
                return result[0]   # the HTML
        except (json.JSONDecodeError, ValueError):
            pass
    return raw_content   # already HTML, pass through
```

**Lesson:** When your pipeline handles two different content formats (JSON and HTML), add an explicit *format detection* step at each entry point. Don't assume the upstream always sends what you expect. A small guard at the boundary is safer than assumptions throughout the code.

---

### Step 3 — Why did the parser fail silently on some blocks?

Even after adding the guard, some lessons still rendered incorrectly. The `editorjs_to_html_and_text()` parser was missing handlers for block types the platform actually used:

| Block type | What it is | Was it handled? |
|---|---|---|
| `images` | Gallery of images | No |
| `attaches` | File download attachment | No |
| `divider` | Horizontal divider line | No |
| `templateCustomBlock` | Nested Editor.js payload | No |

The most problematic was `templateCustomBlock`. Its `data.content` field is itself an Editor.js object `{"blocks": [...]}`. The generic fallback code stringified this dict as `{'blocks': [...]}` (a Python repr, single-quotes), writing that raw string into the HTML — which then broke the cleaner too.

**What was fixed:** Added explicit handlers for each missing type:
- `images` — same rendering logic as `image`
- `attaches` — link with filename and human-readable file size
- `divider` — `<hr>` tag (same as existing `delimiter`)
- `templateCustomBlock` — recursive: extract `data.content.blocks` and call `editorjs_to_html_and_text()` again

**Lesson:** When you write a parser, always add a **logging or print statement in the unknown-type fallback path**. Something like:

```python
else:
    print(f"[UNKNOWN BLOCK TYPE]: {block_type}")
```

This immediately tells you which types you are missing, rather than silently producing wrong output. For data-driven parsers (JSON, API responses), unknown cases should be visible, not silent.

---

### Step 4 — Why did the cleaner fail to repair existing broken files?

Even with the fixes above, files downloaded before the fix were already broken on disk. The cleaner `clean_downloaded_files.py` was supposed to repair them — but it was reporting 0 files cleaned.

The cleaner's first approach used **BeautifulSoup** to find elements containing raw JSON and replace them. This failed because:

The raw JSON strings contained **embedded HTML tags inside string values**:

```json
{"type":"paragraph","data":{"text":"Learn <b>Python</b> with <font color='red'>examples</font>"}}
```

When BeautifulSoup parsed the HTML file, it treated those `<b>` and `<font>` tags inside the JSON as real HTML elements, **fragmenting the JSON string** into dozens of NavigableStrings. The cleaner could no longer see the JSON as a coherent string — it was already shredded.

**What was fixed:** Switched to regex-based extraction for the main case. The raw JSON always appears in a predictable location in the HTML structure — between `</h1>` and `</article>`. By extracting that region as a raw string *before* BeautifulSoup parses it, the JSON is intact:

```python
pattern = re.compile(
    r"(</h1>\s*)"        # end of the lesson title
    r"(\[.*?\]|{.*?})"  # the raw JSON payload
    r"(\s*</article>)",  # before closing article
    re.DOTALL,
)
```

Only if the regex finds nothing does the code fall back to BeautifulSoup (for edge cases where JSON is inside a `<p>` or `<div>` rather than in the article body).

**Lesson:** BeautifulSoup is excellent for navigating valid HTML but dangerous when the HTML contains embedded non-HTML text (JSON, code blocks, template syntax). For those cases, **do the raw string extraction first with regex**, then hand the clean JSON string off to a proper JSON parser. The rule: use the right tool for each format — regex for structural position, JSON parser for content.

---

### Step 5 — The Python dict repr format

One more format appeared that no one anticipated. Some older files had content that looked like this:

```
{'blocks': [{'id': 'abc', 'type': 'paragraph', 'data': {'text': '...'}}]}
```

Note the **single quotes** — this is Python's `repr()` format for dicts, not valid JSON. `json.loads()` rejects it. This happened because somewhere in the old code a dict was printed or written using Python string conversion instead of `json.dumps()`.

**What was fixed:** Added `ast.literal_eval()` as a second parser alongside `json.loads()`:

```python
for parser in (json.loads, ast.literal_eval):
    try:
        data = parser(text)
        ...
    except Exception:
        continue
```

`ast.literal_eval` safely evaluates Python literals (strings, numbers, lists, dicts) without executing arbitrary code — it is the standard library's safe alternative to `eval()`.

**Lesson:** When you receive data that might have been serialized in multiple ways (JSON, Python repr, YAML, CSV), try multiple parsers in order of strictness. Always try the strictest/most-standard format first (`json.loads`), then fallback to the more permissive one (`ast.literal_eval`). This way you don't over-rely on the permissive parser, but you also don't silently fail on legacy data.

---

### Summary: the debugging method used

Every step above followed the same pattern:

```
1. Observe the symptom (broken file on disk)
        │
        ▼
2. Find the exact point in the pipeline where good becomes bad
   (add print/log just before each write to disk)
        │
        ▼
3. Ask: is the data wrong coming IN or wrong going OUT?
   → wrong IN  : fix the upstream caller or add a guard at the entry point
   → wrong OUT : fix the transformation function
        │
        ▼
4. Check what cases the transformation doesn't handle
   (add logging to the "unknown type" fallback)
        │
        ▼
5. Check if existing broken data can be repaired
   (consider tool limitations — BeautifulSoup vs regex)
        │
        ▼
6. Test with grep to confirm 0 broken files remain
   grep -rl '"type":"paragraph"' ~/Downloads/365DataScience/ --include="*.html"
```

The key insight: **pipeline bugs are easier to fix when you know which stage introduced the bad data.** Binary-search the pipeline with strategic print statements rather than reading all the code trying to reason about it.

---

## Notes

- Use only with content you are authorized to access.
- `input.json` is gitignored — never commit real tokens.
- Network or API changes upstream may break scraping or download behavior; check `trace.jsonl` for details.
