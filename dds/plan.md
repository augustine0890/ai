# Plan: Site-Specific Profile Templates

This file is the execution plan and status tracker only.

Durable context, rationale, and design notes live in `dds/session_context.md`
under section 14, "Site Profile Templates and Plan Storage".

---

## 1. Scope

Implement site-specific `.env` profile templates without changing
`web_downloader.py` runtime behavior.

In scope:

- add committed templates under `dds/profiles/`
- keep `.env.example` generic and minimal
- document the copy-to-`.env` workflow
- record the design and storage guidance in `session_context.md`

Out of scope:

- adding `--env-file`
- changing `load_dotenv()` behavior
- changing `capture_playwright_state.py`

---

## 2. Execution Status

| # | Component | Status |
|---|---|---|
| 2.1 | Create `dds/profiles/` directory | `Done` |
| 2.2 | Add `dds/profiles/README.md` | `Done` |
| 2.3 | Add `dds/profiles/.env.bytebytego-courses` | `Done` |
| 2.4 | Add `dds/profiles/.env.bytebytego-guides` | `Done` |
| 2.5 | Add `dds/profiles/.env.wqu` | `Done` |
| 2.6 | Simplify `dds/.env.example` | `Done` |
| 2.7 | Verify ignore rules for `.env` and storage state | `Done` |
| 2.8 | Add profiles section to `dds/WEB_DOWNLOADER.md` | `Done` |
| 2.9 | Record the change in `dds/session_context.md` | `Done` |
| 2.10 | Leave `dds/.env` untouched | `Done` |
| 2.11 | Leave `dds/web_downloader.py` untouched | `Done` |
| 2.12 | Keep `capture_playwright_state.py` unchanged | `Done` |

---

## 3. Verification

Filesystem:

```bash
cd dds
ls profiles/
```

Profile parsing:

```bash
cd dds
for f in profiles/.env.bytebytego-courses profiles/.env.bytebytego-guides profiles/.env.wqu; do
  uv run python -c "from dotenv import dotenv_values; vals = dotenv_values('$f'); assert 'URL' in vals or 'START_URLS' in vals"
done
```

Runtime behavior:

```bash
cd dds
uv run python web_downloader.py --help
```

Round-trip profile switch:

```bash
cd dds
cp .env .env.backup
cp profiles/.env.bytebytego-guides .env
uv run python web_downloader.py --help
mv .env.backup .env
```

Ignore rules:

```bash
cd dds
git check-ignore -v .env
git check-ignore -v playwright_state.json
git check-ignore -v profiles/.env.wqu
```

Expected:

- `.env` is ignored
- `playwright_state.json` is ignored
- `profiles/.env.wqu` is not ignored

---

## 4. Notes

- `web_downloader.py` still reads only `dds/.env`.
- Profile files are templates for humans; they are copied into `.env`.
- `PLAYWRIGHT_STORAGE_STATE=./playwright_state.json` still works exactly the
  same after the profile split, because the copied `.env` is what the script
  reads.
- `.env.example` is intentionally generic; site-specific guidance belongs in
  `dds/profiles/`.

---

## 5. Rollback

```bash
cd dds
rm -rf profiles/
git checkout -- .env.example WEB_DOWNLOADER.md session_context.md
```
