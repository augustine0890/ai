# Data Extraction Script (`data_extract.py`)

## Overview

This script is designed to enrich a list of companies with detailed information by automating web searching, content scraping, and AI-powered data extraction. It works globally with companies in any language and location.

**Key Features:**
- 🌐 **Global Search**: Works with websites in any language (English, Korean, Japanese, etc.)
- 🤖 **Smart Relevance Verification**: Uses Gemini AI to verify websites match company and AI/tech industry
- 🚀 **Optimized Performance**: Configurable performance profiles for speed/quality tradeoff
- 📊 **Multi-Day Batch Processing**: Accumulate results across multiple runs in one file
- 💾 **Progress Tracking**: Automatic checkpoints and resume capability
- 📝 **Comprehensive Logging**: Detailed logs saved to `logs/` folder
- 🎯 **Flexible Row Selection**: Process any range of rows (e.g., rows 100-200)

## How It Works

For each company in the input file, the script performs these steps:

1. **Search**: Uses Serper API to find official homepage URL based on:
   - Company name
   - Person In Charge (PIC) - optional
   - ROO (Region/Office) - optional
   - Multiple search strategies with fallbacks

2. **Verify**: Validates website relevance using Gemini AI:
   - Checks if website belongs to the company
   - Verifies company is in AI/tech industry (detailed categories below)
   - Filters out unrelated search results

3. **Scrape**: Downloads website content using `trafilatura`

4. **Extract**: Uses Google Gemini Flash to analyze content and extract:
   - Location (Physical address or location)
   - Contact Email
   - Application/Service Name

5. **Save**: Updates dataset and saves progress incrementally

### Search Strategy Priorities

The script employs a 10-level priority system for search queries to maximize the chance of finding the correct AI/tech-related website:

1. **Official & AI Context**: `"{Company} AI technology official website"`, `"{Company} artificial intelligence company"`
2. **Industry Specific**: `"{Company} robotics automation AI"`, `"{Company} machine learning AI platform"`, etc.
3. **PIC Context** (if available): `"{Company} {PIC} AI technology"`
4. **Manufacturing AI**: `"{Company} smart factory AI manufacturing"`
5. **Tech Domains**: `"{Company} vision AI image recognition"`, `"{Company} autonomous vehicle self-driving"`
6. **ROO Context** (if available): `"{Company} {ROO} AI technology"`
7. **General Tech**: `"{Company} AI software platform"`, `"{Company} deep learning"`
8. **Service/Healthcare**: `"{Company} healthcare AI medical"`, `"{Company} logistics AI"`
9. **Standard Fallback**: `"{Company} official website"`
10. **Name Only**: `"{Company}"`

## Prerequisites

### Environment Variables
Create a `.env` file in the root directory:

```env
SERPER_API_KEY=your_serper_api_key
GEMINI_API_KEY=your_google_gemini_api_key
```

### Python Dependencies

```bash
uv pip install polars requests trafilatura tqdm google-genai pydantic python-dotenv
```

## Input & Output

### Input: `data/companies.xlsx`
- **Required columns**: `Company`
- **Optional columns**: `PIC` (Person In Charge), `ROO` (Region/Office)
- **Handles**: Duplicate column names, mixed data types, any language
- **Preserved**: Existing columns like `Homepage URL`, `Email`, `Application`, `Area`

### Output: `data/companies_enriched.xlsx`
- Incrementally saved (default: every 10 rows)
- Accumulates results across multiple runs
- All columns from input + extracted data
- **Mapped Columns**:
  - `Homepage URL` ← Found Website URL
  - `Area` ← Extracted Location (Physical address)
  - `Application` ← Extracted Application/Service Name
  - `Email` ← Extracted Contact Email

### Logs: `logs/data_extract_YYYYMMDD_HHMMSS.log`
- Detailed processing information
- Success/failure per company
- Performance metrics
- Error tracking

## Supported AI/Tech Categories

The script identifies and verifies these industry categories:

1. **Robotics and Automation AI**
   - Physical Movement, Control, Cognition
   - Industrial/Cooperative robots, Robot control software
   - Transfer and Palletizing systems

2. **Vision AI**
   - Image and image recognition
   - Camera/Sensor-based recognition
   - Defective inspection, Object/Face/Behavior Recognition
   - Medical and manufacturing vision

3. **AI Software and Platform**
   - AI engine and tools (no physical hardware)
   - AI Development Platforms
   - Data Analysis, MLOps, AI API

4. **Smart Factory and Manufacturing AI**
   - Manufacturing Process Optimization
   - Process automation, Production optimization, Quality control

5. **Logistics and Mobility AI**
   - Movement, route, transportation optimization
   - Self-driving, Autonomous vehicles
   - Drone, AGV/AMR, Logistics optimization

6. **Service, Education, and Healthcare AI**
   - Educational AI, Medical and Health Care
   - Chatbot, Emotion and Behavior Analysis

7. **AI Semiconductor and Hardware**
   - AI computation technology, AI chips
   - Edge devices, Sensors

## Usage Examples

### Basic Usage (Default Settings)
Process all rows with default performance settings:
```python
python data_extract.py
# Or with uv:
uv run data_extract.py
```

### Quick Start: Process Specific Row Range
```python
# Process rows 0-100
main(start_row=0, end_row=100)

# Process rows 100-200
main(start_row=100, end_row=200)

# Process from row 500 to end
main(start_row=500)
```

### Multi-Day Batch Processing (Recommended for 1800+ rows)

**Day 1: Process rows 0-300 (~30-45 minutes)**
```python
main(start_row=0, end_row=300, max_queries=2, max_urls_per_query=1)
# Output: companies_enriched.xlsx has rows 0-299 filled
```

**Day 2: Process rows 300-600 (~30-45 minutes)**
```python
main(start_row=300, end_row=600, max_queries=2, max_urls_per_query=1)
# Previous results automatically preserved!
# Output: companies_enriched.xlsx now has rows 0-599 filled
```

Continue this pattern for remaining days. Results automatically accumulate in one file!

### Resume After Interruption
```python
# Resume from last checkpoint
main(resume_from_checkpoint=True)
```

### Performance Profiles

**FAST MODE** (3-6 sec/row) - For large datasets
```python
main(
    max_queries=2,
    max_urls_per_query=1,
    min_confidence=0.65,
    rate_limit=0.3
)
```

**BALANCED MODE** (6-8 sec/row) - Default, good quality/speed
```python
main(
    max_queries=3,
    max_urls_per_query=2,
    min_confidence=0.7,
    rate_limit=0.5
)
```

**THOROUGH MODE** (10-14 sec/row) - Best quality
```python
main(
    max_queries=5,
    max_urls_per_query=3,
    min_confidence=0.75,
    rate_limit=0.5
)
```

### Location/Language Filters (Optional)
```python
# Search for Korean companies
main(search_location='kr', search_language='ko')

# Search for US companies
main(search_location='us', search_language='en')

# Default: Global search, any language
main()
```

### Combined Example
```python
# Process Korean companies, rows 100-200, with thorough verification
main(
    start_row=100,
    end_row=200,
    search_location='kr',
    search_language='ko',
    max_queries=5,
    max_urls_per_query=3,
    min_confidence=0.75
)
```

## Configuration Parameters

### Row Selection
- `start_row` (int): Starting row index (0-based), default: None (from beginning)
- `end_row` (int): Ending row index (0-based), default: None (to end)

### Processing Control
- `skip_filled` (bool): Skip rows already processed, default: True
- `continue_previous` (bool): Continue from output file if exists, default: True
- `resume_from_checkpoint` (bool): Resume from last checkpoint, default: False

### Performance Tuning
- `max_queries` (int): Max search queries per company, default: 3 (range: 1-10)
- `max_urls_per_query` (int): Max URLs to verify per query, default: 2 (range: 1-3)
- `min_confidence` (float): Min confidence score to accept, default: 0.7 (range: 0.5-1.0)
- `rate_limit` (float): Sleep time between API calls (sec), default: 0.5

### Storage
- `save_interval` (int): Save progress every N rows, default: 10
- `input_file` (str): Input file path, default: 'data/companies.xlsx'
- `output_file` (str): Output file path, default: 'data/companies_enriched.xlsx'

### Search Filters
- `search_location` (str): Country code (e.g., 'us', 'kr', 'jp'), default: None (global)
- `search_language` (str): Language code (e.g., 'en', 'ko', 'ja'), default: None (auto)

## Time Estimates (for 1800 rows)

| Mode | Time/Row | Total Time | Quality | Use Case |
|------|----------|-----------|---------|----------|
| FAST | 3-6 sec | 2-3 hours | Good | Large datasets, quick results |
| BALANCED | 6-8 sec | 3-4 hours | Very Good | Default, recommended |
| THOROUGH | 10-14 sec | 5-7 hours | Excellent | Small datasets, high accuracy |

## Features

### Smart Duplicate Handling
- Automatically detects and removes duplicate columns
- Keeps first occurrence of: Company, PIC, ROO, Position
- Works with malformed Excel files

### Automatic Progress Tracking
- Saves checkpoint after every batch
- Checkpoint file: `data/progress_checkpoint.json`
- Automatically cleared on successful completion

### Flexible Result Continuation
- **`continue_previous=True`** (default): Loads output file if exists, preserves previous results
- **`continue_previous=False`**: Always start fresh from input file
- Perfect for multi-day processing!

### Comprehensive Logging
- Console output + file logging
- Timestamps on all entries
- Per-row processing status
- Summary statistics at completion

### Error Handling
- Graceful handling of network errors
- Failed searches or irrelevant websites marked as "Not Found"
- Processing errors marked as "Error"
- Keyboard interrupt (Ctrl+C) safe - saves progress before exit
- Emergency saves on fatal errors
- Automatic checkpointing prevents data loss

### Technical Implementation Details

- **Token Optimization**: 
  - Website content is cleaned (whitespace removal) and truncated before sending to Gemini.
  - Verification limit: 8,000 characters.
  - Extraction limit: 10,000 characters.
- **Model Version**: Uses `gemini-3-flash-preview` for high speed and cost-efficiency.
- **Excel Loading**: Attempts to read the first sheet of the input Excel file. If the file contains duplicate columns (e.g., multiple 'Company' columns), the script automatically drops the duplicates to prevent data corruption.
- **Relevance Threshold**: Content must be >50 characters to be considered for verification.

## Running the Script

### Standard Run
```bash
uv run data_extract.py
```

### Uncomment desired configuration in `if __name__ == "__main__":` section

### Example configurations provided:
1. Process all rows (default)
2. Process specific range
3. Resume from checkpoint
4. Multi-day batch processing
5. Location/language filters

## File Structure

```
D:\source\ai\
├── data_extract.py              # Main script
├── .env                         # API keys (create this)
├── data/
│   ├── companies.xlsx           # Input file
│   ├── companies_enriched.xlsx   # Output file (accumulated)
│   └── progress_checkpoint.json   # Progress tracking
└── logs/
    └── data_extract_YYYYMMDD_HHMMSS.log
```

## Best Practices

1. **Test First**: Always test with 5-10 rows before large batches
2. **Use Batch Processing**: Split large datasets into manageable chunks
3. **Monitor Logs**: Check logs for quality and error rates
4. **Skip Filled Rows**: Use `skip_filled=True` to avoid reprocessing
5. **Adjust Confidence**: Lower `min_confidence` if finding too many "Not Found"
6. **Rate Limiting**: Increase `rate_limit` if getting API rate limit errors

## Troubleshooting

### Many "Not Found" results
- Lower `min_confidence` to 0.65
- Increase `max_queries` to 5
- Check if companies are actually in AI/tech industry

### Script runs too slowly
- Use FAST mode: `max_queries=2, max_urls_per_query=1`
- Lower `rate_limit` to 0.3
- Process in smaller batches

### Missing data after run
- Check `continue_previous=True` is set
- Verify `output_file` path is correct
- Check logs for errors

### API Rate Limits
- Increase `rate_limit` (default 0.5)
- Use FAST mode (fewer API calls)
- Process in smaller batches with delays between runs

## API Costs

**Approximate costs for 1800 rows:**
- **FAST mode**: ~$18-20 (Serper $18 + Gemini $1-2)
- **BALANCED mode**: ~$27-30 (Serper $27 + Gemini $1.50)
- **THOROUGH mode**: ~$45-50 (Serper $45 + Gemini $3-5)

## Support & Improvements

The script is designed to be:
- **Flexible**: Configurable for different needs
- **Resilient**: Handles errors gracefully
- **Observable**: Comprehensive logging
- **Resumable**: Can pause and continue anytime
- **Efficient**: Optimized for large datasets
