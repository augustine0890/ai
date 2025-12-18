# Data Extraction Script (`data_extract.py`)

## Overview
This script is designed to enrich a list of companies with detailed information by automating web searching, content scraping, and AI-powered data extraction.

It performs the following steps for each company in the input file:
1.  **Search**: Uses the Serper Dev API to find the official homepage URL based on the Company name and Person In Charge (PIC).
2.  **Scrape**: Downloads the website content using `trafilatura`.
3.  **Extract**: Uses Google Gemini (Flash model) to analyze the website text and extract:
    *   Location (Address)
    *   Contact Email
    *   Application/Service Name
4.  **Save**: Updates the dataset and saves progress incrementally.

## Prerequisites

### Environment Variables
You must create a `.env` file in the root directory with the following keys:

```env
SERPER_API_KEY=your_serper_api_key
GEMINI_API_KEY=your_google_gemini_api_key
```

### Python Dependencies
Ensure you have the required packages installed. Based on the script imports, you need:

*   `polars` (for data manipulation)
*   `requests` (for API calls)
*   `trafilatura` (for web scraping)
*   `tqdm` (for progress bars)
*   `google-genai` (for Gemini API)
*   `pydantic` (for data validation)
*   `python-dotenv` (for loading .env)
*   `fastexcel` (recommended for reading Excel files with Polars)

You can install them via pip or uv:

```bash
uv pip install polars requests trafilatura tqdm google-genai pydantic python-dotenv fastexcel
```

## Input & Output

*   **Input**: `data/companies.xlsx`
    *   Expected columns: `Company`, `PIC` (Person In Charge).
    *   Existing columns like `Homepage URL`, `Email`, `Application` will be preserved or updated.
*   **Output**: `data/companies_enriched.xlsx`
    *   The script saves to this file every 10 rows to prevent data loss.

## How to Run

1.  Place your input file at `data/companies.xlsx`.
2.  Run the script from the project root:

```bash
uv run data_extract.py
```

## Logic Details

*   **Resuming**: The script checks if `Homepage URL` is already filled. If the script crashes, you can restart it, and it will skip already processed rows.
*   **Rate Limiting**: There is a 0.5-second sleep between requests to be polite to servers and stay within API rate limits.
*   **Error Handling**:
    *   If a website cannot be scraped (e.g., JavaScript-only sites), it marks the location as "Scrape Failed".
    *   If the search finds no results, it marks the URL as "Not Found".