# Data Extraction Script (`data_extract.py`)

## 1. Purpose & Overview

`data_extract.py` is a production-ready data enrichment pipeline that ingests a company dataset and enriches each row with verified, structured information about the company’s AI/technology focus. It automates search, content scraping, relevance verification, and structured extraction while supporting caching, checkpointing, and robust logging.

Key capabilities:
- **Website discovery** via Serper search API with prioritized AI/tech-focused queries
- **Content scraping** with Trafilatura for clean, readable text
- **Relevance verification** using Gemini AI with confidence scoring and category assignment
- **Structured extraction** using Gemini AI into a validated schema
- **Caching** (in-memory LRU + optional persistent disk cache)
- **Checkpointing** to resume safely after interruptions
- **Incremental saving** to prevent data loss
- **Detailed logging** for monitoring and troubleshooting

## 2. Architecture Breakdown

### 2.1 Modules & Dependencies
The script relies on:
- **polars**: DataFrame I/O and Excel/CSV handling
- **requests**: HTTP client for Serper API
- **trafilatura**: Webpage downloading and content extraction
- **google-genai**: Gemini AI client
- **pydantic**: Structured data models and validation
- **python-dotenv**: Environment variable loading
- **tqdm**: Progress bar
- **standard library**: logging, json, os, datetime, pathlib, hashlib, collections

### 2.2 Core Classes

#### `CacheStats`
Tracks hit/miss statistics for each cache layer:
- Search, Scrape, Verification, Extraction
- Computes hit rates and total API calls saved

#### `LRUCache`
OrderedDict-backed LRU cache with size cap to prevent memory growth.

#### `DataCache`
Manages four LRU caches with optional disk persistence.
- Uses MD5 hashing of parameters for consistent keys
- Converts Pydantic models to dicts for JSON serialization

#### `CompanyInfo`
Pydantic model for structured extraction output:
- `location`
- `contact_email`
- `application_service`
- `homepage_url`

#### `WebsiteRelevance`
Pydantic model for relevance verification output:
- `is_relevant`
- `relevance_category`
- `confidence_score`
- `reason`

### 2.3 Core Functions

#### Logging & Progress
- `setup_logging()`
- `save_checkpoint()`
- `load_checkpoint()`
- `clear_checkpoint()`

#### Caching
- `init_cache()`
- `get_cache()`
- `clear_cache()`

#### Pipeline Steps
- `build_search_queries()`
- `search_google_serper()`
- `scrape_website_content()`
- `verify_website_relevance()`
- `analyze_with_gemini()`

#### Data I/O
- `load_data()`
- `save_results()`

#### Processing
- `process_company()`
- `main()`

### 2.4 Interaction Summary
1. `main()` initializes logging and cache
2. `load_data()` reads input (or continues from output)
3. Loop per row:
   - `build_search_queries()` → `search_google_serper()`
   - `scrape_website_content()`
   - `verify_website_relevance()`
   - `analyze_with_gemini()`
4. Results saved periodically via `save_results()`
5. Checkpoints saved via `save_checkpoint()`
6. On completion: final save, cache stats logged, checkpoint cleared

## 3. End-to-End Workflow

1. **Input Load**
   - Reads `data/companies.xlsx` (or CSV)
   - Drops duplicate columns (Company/PIC/ROO/Position)
   - Ensures `Area` column exists

2. **Row Processing**
   - Skip if `Homepage URL` already filled (configurable)
   - Generate prioritized search queries
   - Search for candidate URLs
   - Scrape content and verify relevance
   - Extract structured fields from relevant site

3. **Output & Persistence**
   - Save results every `save_interval` rows
   - Save checkpoint with last processed index
   - Optionally persist cache to disk

4. **Completion**
   - Final save
   - Print cache stats
   - Clear checkpoint

## 4. Dependencies & Requirements

### Python Packages
Install via:
```bash
uv pip install polars requests trafilatura tqdm google-genai pydantic python-dotenv
```

### Environment Variables
Create `.env`:
```env
SERPER_API_KEY=your_serper_api_key
GEMINI_API_KEY=your_google_gemini_api_key
```

### Input File Requirements
- Excel (`.xlsx`) or CSV
- **Required column:** `Company`
- **Optional columns:** `PIC`, `ROO`

## 5. Configuration Options

### Constants
- `INPUT_FILE`: `data/companies.xlsx`
- `OUTPUT_FILE`: `data/companies_enriched.xlsx`
- `PROGRESS_FILE`: `data/progress_checkpoint.json`
- `CACHE_FILE`: `data/cache.json`
- `LOG_DIR`: `logs/`

### Main Function Parameters
- `start_row`, `end_row`: Subset processing
- `skip_filled`: Skip rows with existing `Homepage URL`
- `save_interval`: Save every N rows
- `rate_limit`: Delay between API calls
- `resume_from_checkpoint`: Resume from last checkpoint
- `continue_previous`: Load output file if exists
- `search_location`, `search_language`: Geo/language targeting
- `max_queries`, `max_urls_per_query`, `min_confidence`: Search + verification controls
- `enable_cache`, `enable_disk_cache`, `cache_size`: Caching settings

## 6. Data Flow Documentation

### Input → Intermediate → Output

**Input Columns**
- `Company` (required)
- `PIC` (optional)
- `ROO` (optional)

**Intermediate Steps**
1. Search queries built
2. URLs retrieved
3. Content scraped
4. Relevance verified
5. Structured extraction

**Output Columns**
- `Homepage URL` (website)
- `Area` (location)
- `Application` (service/product)
- `Email` (contact)

## 7. Error Handling & Retry Logic

- **Search errors**: logged; empty results cached
- **Scrape errors**: logged; empty content cached
- **Gemini errors**: logged; returns `Error` values, cached
- **KeyboardInterrupt**: triggers save + checkpoint
- **Fatal errors**: emergency save + checkpoint

There is no explicit retry loop; the design favors caching and fallback queries to reduce repeat failures.

## 8. Caching & Checkpoint System

### Caching
- In-memory LRU caches for search/scrape/verification/extraction
- Optional disk persistence via `data/cache.json`
- Significantly reduces API calls and runtime on repeated runs

### Checkpoints
- Saved after every `save_interval`
- Stored in `data/progress_checkpoint.json`
- Used for resumption with `resume_from_checkpoint=True`

## 9. API Integrations

### Serper API
- Endpoint: `https://google.serper.dev/search`
- Method: POST
- Payload: query + num results + optional geo/language
- Rate limiting: controlled by `rate_limit`

### Trafilatura
- `fetch_url()` downloads page
- `extract()` pulls readable content

### Google Gemini
- Model: `gemini-3-flash-preview`
- Verification: `WebsiteRelevance` schema
- Extraction: `CompanyInfo` schema

## 10. Usage Examples

### Default Run
```bash
uv run data_extract.py
```

### Process Subset
```python
main(start_row=0, end_row=100)
```

### Resume After Crash
```python
main(resume_from_checkpoint=True)
```

### Enable Disk Cache
```python
main(enable_disk_cache=True, cache_size=2000)
```

## 11. Performance & Optimization

- **FAST Mode**: fewer queries/URLs, lower confidence threshold
- **BALANCED Mode**: default recommended
- **THOROUGH Mode**: more queries + higher confidence

Optimization strategies:
- Lower `max_queries` and `max_urls_per_query`
- Enable cache (especially disk cache for multi-day runs)
- Increase `save_interval` to reduce I/O overhead

## 12. Troubleshooting Guide

| Issue | Cause | Resolution |
|------|-------|------------|
| Too many “Not Found” | High confidence threshold | Lower `min_confidence`, increase `max_queries` |
| Slow runtime | Too many API calls | Use FAST mode, enable caching |
| API rate limits | Excessive request frequency | Increase `rate_limit` |
| Missing outputs | Output file overwritten or not saved | Check `continue_previous`, verify output path |
| Checkpoint not resuming | Missing checkpoint | Ensure `resume_from_checkpoint=True` and file exists |

## 13. Command-Line Arguments

The script is executed as a Python module; parameters are passed by editing the `main()` call under `if __name__ == "__main__":`. There is no CLI parser implemented (e.g., argparse). To simulate CLI usage, modify the `main()` call with desired arguments.

## 14. File I/O Operations

- **Input**: Excel/CSV via Polars
- **Output**: Excel (`companies_enriched.xlsx`) via Polars
- **Checkpoint**: JSON file
- **Cache**: JSON file (optional)
- **Logs**: Text logs in `logs/`

## 15. Logging & Monitoring

- Console + file logging
- Timestamped log files
- Per-row processing messages
- Cache stats summary
- Fatal error logging + stack traces

## 16. Current Limitations

- No built-in CLI argument parser
- No explicit retry backoff beyond query fallbacks
- Single-threaded (no concurrency)

## 17. Suggested Enhancements (Future)

- Add argparse CLI
- Add configurable retry backoff
- Optional concurrency with rate-limited workers
- Export JSON results alongside Excel
