# Plan: Clean-up and Reorganize `dds/`

This plan reorganizes the `dds/` folder into a clean, package-style layout
without changing runtime behavior. Execution-plan only — no implementation
steps are performed here.

Durable rationale, design notes, and target structure live in
`dds/docs/session_context.md` under section 15, "Folder Cleanup and Module Layout".

---

## 1. Scope

Reorganize the `dds/` directory so that:

- source code lives in a proper Python package (`dds/src/...`)
- the two tool families (365DataScience downloader, generic web_downloader)
  are clearly separated
- config, data, docs, and output artifacts are each in their own directory
- each script still runs exactly the same way as before (via `uv run`)

In scope:

- move files into sub-folders
- add `__init__.py` files to make packages importable
- fix imports that break from the moves (mechanical rename only)
- split oversized docs (`README.md`, `WEB_DOWNLOADER.md`) only if trivially
  separable
- update `dds/session_context.md` and `dds/pyproject.toml` to reflect the
  new structure
- add `.gitignore` rules for moved generated content

Out of scope:

- refactoring `web_downloader.py` internals (still one file; splitting it is
  a separate effort)
- rewriting logic, renaming functions, changing CLI behavior
- adding tests (a follow-up plan)
- changing the `.env` loader or profile workflow
- touching the actual downloaded content under `bytebytego_com/`

---

## 2. Current Issues (what the new layout fixes)

| # | Issue | Evidence |
|---|---|---|
| I1 | Two unrelated toolchains share one flat folder | `main.py` (365DS) next to `web_downloader.py` (generic) |
| I2 | No package structure, imports are bare (`from logger import ...`) | See top of `main.py`, `download_single_course.py` |
| I3 | Downloaded output `bytebytego_com/` sits beside source code | Root of `dds/` |
| I4 | Runtime artifacts (`trace.jsonl`) committed alongside source | `dds/trace.jsonl` |
| I5 | Config/data at root with no home (`input.json`, `courses_list.json`) | Root of `dds/` |
| I6 | `web_downloader.py` is 148 KB / ~3400 lines in one file | Single file |
| I7 | `README.md` (61 KB) mixes overview, 365DS docs, web_downloader docs | One doc, many topics |
| I8 | No tests directory | `ls dds/` shows none |
| I9 | Missing `__init__.py`, so `dds/` isn't importable as a package | `ls dds/*.py` |

---

## 3. Target Layout

```text
dds/
├── pyproject.toml
├── uv.lock
├── README.md                       # short overview + pointers
├── .env.example
├── .env                            # local, ignored
│
├── src/
│   └── dds/
│       ├── __init__.py
│       ├── common/                 # shared utilities
│       │   ├── __init__.py
│       │   └── logger.py
│       ├── datascience365/         # 365DataScience tool
│       │   ├── __init__.py
│       │   ├── main.py
│       │   ├── list_courses.py
│       │   ├── download_single_course.py
│       │   ├── redownload_html.py
│       │   ├── clean_downloaded_files.py
│       │   ├── compress_courses.py
│       │   ├── course_model.py
│       │   └── video_model.py
│       └── web_downloader/         # generic web_downloader tool
│           ├── __init__.py
│           ├── __main__.py         # thin entrypoint wrapping web_downloader.py
│           ├── web_downloader.py
│           └── capture_playwright_state.py
│
├── config/
│   ├── input.example.json
│   ├── input.json                  # local, ignored
│   └── courses_list.json
│
├── profiles/                       # unchanged
│   ├── README.md
│   ├── .env.bytebytego-courses
│   ├── .env.bytebytego-guides
│   └── .env.wqu
│
├── docs/
│   ├── web_downloader.md           # was WEB_DOWNLOADER.md
│   ├── datascience365.md           # extracted from README.md
│   └── session_context.md          # moved from root
│
├── output/                         # all runtime output, ignored
│   ├── bytebytego_com/             # moved
│   ├── trace.jsonl                 # moved
│   └── playwright_state.json       # ignored
│
└── tests/                          # placeholder for future tests
    └── .gitkeep
```

Console scripts exposed via `pyproject.toml`:

- `dds-web = "dds.web_downloader.__main__:main"`
- `dds-365 = "dds.datascience365.main:main"`

This preserves `uv run python -m dds.web_downloader` and
`uv run python -m dds.datascience365.main` as the new invocation pattern,
with shim scripts in `dds/` root (optional) mapping to the package form.

---

## 4. Execution Steps

Each step is a discrete, verifiable change. Do them in order.

| # | Step | Status |
|---|---|---|
| 4.1 | Create target directories: `src/dds/{common,datascience365,web_downloader}`, `config/`, `docs/`, `output/`, `tests/` | `Done` |
| 4.2 | Add empty `__init__.py` in each new package directory | `Done` |
| 4.3 | Move shared module: `logger.py` → `src/dds/common/logger.py` | `Done` |
| 4.4 | Move 365DS modules to `src/dds/datascience365/` (8 files: `main.py`, `list_courses.py`, `download_single_course.py`, `redownload_html.py`, `clean_downloaded_files.py`, `compress_courses.py`, `course_model.py`, `video_model.py`) | `Done` |
| 4.5 | Move web_downloader modules to `src/dds/web_downloader/` (`web_downloader.py`, `capture_playwright_state.py`) | `Done` |
| 4.6 | Add `src/dds/web_downloader/__main__.py` that calls the main function in `web_downloader.py` | `Done` |
| 4.7 | Fix imports: change bare `from logger import ...` → `from dds.common.logger import ...`; same for `course_model`, `video_model`, `clean_downloaded_files`, `download_single_course` references across the 365DS modules | `Done` |
| 4.8 | Move config files: `input.json`, `input.example.json`, `courses_list.json` → `config/`. Update all code paths that read them (search for string `input.json`, `courses_list.json`) | `Done` |
| 4.9 | Move output artifacts: `bytebytego_com/` → `output/bytebytego_com/`, `trace.jsonl` → `output/trace.jsonl`. Update default `DESTINATION` in profiles and default trace path in `web_downloader.py` | `Done` |
| 4.10 | Move docs: `WEB_DOWNLOADER.md` → `docs/web_downloader.md`; `session_context.md` → `docs/session_context.md` | `Done` |
| 4.11 | Shrink `README.md` to a short overview + links to `docs/web_downloader.md`, `docs/datascience365.md`, `profiles/README.md` | `Done` |
| 4.12 | Extract 365DS-specific content out of `README.md` into `docs/datascience365.md` | `Done` |
| 4.13 | Update `pyproject.toml`: add `[tool.setuptools.packages.find] where=["src"]`, add `[project.scripts]` entries for `dds-web` and `dds-365` | `Done` |
| 4.14 | Update `.gitignore` to cover `output/`, `__pycache__/`, `.venv/`, `playwright_state.json`, `config/input.json`, `.env` | `Done` |
| 4.15 | Update `docs/session_context.md` with section 15 documenting the new layout and file-move map | `Done` |
| 4.16 | Update `dds/profiles/*.env.*` `DESTINATION=` paths to point at `./output/<site>/` | `Done` |
| 4.17 | Update every doc reference to moved files (hyperlinks, code blocks) in `README.md`, `docs/web_downloader.md`, `docs/session_context.md`, `profiles/README.md` | `Done` |
| 4.18 | Remove stale `__pycache__/` from the repo tree | `Done` |
| 4.19 | Final sweep: `grep -R "from logger"`, `grep -R "input.json"`, `grep -R "courses_list.json"`, `grep -R "trace.jsonl"`, `grep -R "WEB_DOWNLOADER.md"` — confirm no bare references remain | `Done` |

---

## 5. Verification

Run after each phase or at the end.

Package resolves:

```bash
cd dds
uv run python -c "import dds, dds.common.logger, dds.web_downloader, dds.datascience365"
```

CLI entry points still work:

```bash
cd dds
uv run python -m dds.web_downloader --help
uv run python -m dds.datascience365.main --help
```

Profile workflow unchanged:

```bash
cd dds
cp profiles/.env.bytebytego-guides .env
uv run python -m dds.web_downloader --help
```

Import hygiene:

```bash
cd dds
grep -R "^from logger" src/ && echo "FAIL: bare logger import remains"
grep -R "^import logger" src/ && echo "FAIL: bare logger import remains"
grep -R "from course_model" src/ && echo "FAIL: bare course_model import remains"
grep -R "from video_model" src/ && echo "FAIL: bare video_model import remains"
```

Output paths:

```bash
cd dds
test -d output && echo "OK output/ exists"
test -d src/dds && echo "OK src/dds/ exists"
test ! -f logger.py && echo "OK logger.py moved"
test ! -f web_downloader.py && echo "OK web_downloader.py moved"
```

Ignore rules:

```bash
cd dds
git check-ignore -v output/trace.jsonl
git check-ignore -v output/bytebytego_com/anything
git check-ignore -v config/input.json
```

---

## 6. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Hidden hard-coded paths break (`input.json`, `trace.jsonl`, `bytebytego_com/`) | Step 4.19 grep sweep before declaring done |
| Relative imports break because current scripts use bare imports | Step 4.7 fixes imports explicitly, and step 4.13 adds `src/` layout to `pyproject.toml` |
| `web_downloader.py` default `TRACE_FILE` written to old path | Step 4.9 updates default; fallback: pass absolute path via env |
| User has uncommitted work in `bytebytego_com/` | Check `git status` before step 4.9; confirm with user if dirty |
| Splitting `README.md` loses anchor links | Keep section anchors identical where possible; audit in step 4.17 |

---

## 7. Rollback

All moves are tracked in git. If the layout causes trouble:

```bash
cd dds
git checkout -- .
git clean -fd src/ config/ docs/ output/ tests/
```

For a partial rollback after commits, revert the reorganize commit(s) with
`git revert <sha>`.

---

## 8. Notes

- This plan is structural only. `web_downloader.py` stays a single file in
  this pass; a follow-up plan can split it into modules.
- Output artifacts (`bytebytego_com/`, `trace.jsonl`) are moved to `output/`
  and git-ignored. If any are already committed, `git rm --cached` them in
  step 4.14.
- The `profiles/` directory is already well-organized and stays put.
- The new `src/`-layout is the standard modern Python packaging convention
  and prevents accidental imports from the project root.

