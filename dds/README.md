# dds

This directory now uses a package-style layout under `src/dds/`.

Main entrypoints:

- generic site mirror: `uv run python -m dds.web_downloader`
- 365 Data Science downloader: `uv run python -m dds.datascience365.main`

Docs:

- web downloader: [docs/web_downloader.md](docs/web_downloader.md)
- 365 Data Science downloader: [docs/datascience365.md](docs/datascience365.md)
- session context and design history: [docs/session_context.md](docs/session_context.md)
- profile templates: [profiles/README.md](profiles/README.md)

Config and output:

- local env: `dds/.env`
- JSON config: `dds/config/input.json`
- runtime output: `dds/output/`
