import polars as pl
import requests
import trafilatura
import time
import json
import os
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ================= CONFIGURATION =================
INPUT_FILE = 'data/companies.xlsx'  # Input file from data folder
OUTPUT_FILE = 'data/companies_enriched.xlsx'  # Output file to data folder
PROGRESS_FILE = 'data/progress_checkpoint.json'  # Progress tracking file
LOG_DIR = 'logs'  # Log directory
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Validate API keys
if not SERPER_API_KEY or not GEMINI_API_KEY:
    raise ValueError("API keys not found. Please create a .env file with SERPER_API_KEY and GEMINI_API_KEY")

# Initialize Gemini Client
client = genai.Client(api_key=GEMINI_API_KEY)


# ================= LOGGING & PROGRESS TRACKING =================

def setup_logging():
    """
    Set up logging to both console and file in logs directory.
    Returns the logger instance.
    """
    # Create logs directory if it doesn't exist
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = Path(LOG_DIR) / f'data_extract_{timestamp}.log'

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # Also print to console
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def save_checkpoint(processed_rows, total_rows, last_row_index):
    """
    Save progress checkpoint to resume later if needed.
    """
    checkpoint = {
        'timestamp': datetime.now().isoformat(),
        'processed_rows': processed_rows,
        'total_rows': total_rows,
        'last_row_index': last_row_index
    }

    # Ensure data directory exists
    Path('data').mkdir(parents=True, exist_ok=True)

    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, indent=2)


def load_checkpoint():
    """
    Load the last checkpoint if it exists.
    Returns the checkpoint dict or None.
    """
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
                return checkpoint
        except Exception as e:
            logging.warning(f"Could not load checkpoint: {e}")
    return None


def clear_checkpoint():
    """
    Clear the checkpoint file after successful completion.
    """
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        logging.info("Checkpoint cleared")


# Define the data structure we want Gemini to extract
class CompanyInfo(BaseModel):
    location: str = Field(..., description="Physical address or location of the company")
    contact_email: str = Field(..., description="Contact email address found on the page")
    application_service: str = Field(..., description="Name of main application, service, or product")
    homepage_url: str = Field(..., description="The official website URL")


class WebsiteRelevance(BaseModel):
    is_relevant: bool = Field(..., description="Whether the website is relevant to the company and AI/tech industry")
    relevance_category: str = Field(..., description="Primary category: 'Robotics and Automation AI', 'Vision AI', 'AI Software and Platform', 'Smart Factory and Manufacturing AI', 'Logistics and Mobility AI', 'Service/Education/Healthcare AI', 'AI Semiconductor and Hardware', or 'Not Relevant'")
    confidence_score: float = Field(..., description="Confidence score from 0.0 to 1.0")
    reason: str = Field(..., description="Brief reason for the relevance decision, mentioning specific keywords found")


# ================= HELPER FUNCTIONS =================

def search_google_serper(query, num_results=3, location=None, language=None):
    """
    Uses Serper.dev to find the official website URL.
    Returns a list of organic links (up to num_results).

    Args:
        query: Search query string
        num_results: Number of results to return
        location: Country code (e.g., 'us', 'kr', 'jp') or None for global
        language: Language code (e.g., 'en', 'ko', 'ja') or None for auto
    """
    url = "https://google.serper.dev/search"
    payload = {
        "q": query,
        "num": num_results
    }

    # Add optional location and language filters
    if location:
        payload["gl"] = location
    if language:
        payload["hl"] = language

    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }

    try:
        response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
        results = response.json()

        # Return multiple organic links for verification
        if 'organic' in results and len(results['organic']) > 0:
            return [result['link'] for result in results['organic'][:num_results]]
    except Exception as e:
        logging.error(f"Search Error: {e}")
    return []


def build_search_queries(company_name, pic_name=None, roo_name=None):
    """
    Build multiple search queries with AI/tech-focused keywords.
    Prioritizes queries that target relevant industry websites.
    Works globally with any language.
    Optionally includes PIC (Person In Charge) and ROO for more specific searches.

    Args:
        company_name: Company name (required)
        pic_name: Person In Charge name (optional)
        roo_name: ROO name (optional)

    Returns a list of search queries ordered by priority.
    """
    queries = []

    # Priority 1: Official website with AI/tech context
    queries.append(f"{company_name} AI technology official website")
    queries.append(f"{company_name} artificial intelligence company")

    # Priority 2: Specific AI/tech industry keywords
    queries.append(f"{company_name} robotics automation AI")
    queries.append(f"{company_name} machine learning AI platform")
    queries.append(f"{company_name} computer vision AI")

    # Priority 3: Add PIC with AI context if available
    if pic_name:
        queries.append(f"{company_name} {pic_name} AI technology")
        queries.append(f"{company_name} {pic_name} robotics")

    # Priority 4: Manufacturing and industrial AI keywords
    queries.append(f"{company_name} smart factory AI manufacturing")
    queries.append(f"{company_name} industrial automation robotics")

    # Priority 5: Specific technology domains
    queries.append(f"{company_name} vision AI image recognition")
    queries.append(f"{company_name} autonomous vehicle self-driving")
    queries.append(f"{company_name} AGV AMR robotics")

    # Priority 6: Add ROO with tech context if available
    if roo_name:
        queries.append(f"{company_name} {roo_name} AI technology")

    # Priority 7: General tech and software keywords
    queries.append(f"{company_name} AI software platform")
    queries.append(f"{company_name} deep learning neural network")
    queries.append(f"{company_name} AI chip semiconductor")

    # Priority 8: Healthcare, service, logistics AI
    queries.append(f"{company_name} healthcare AI medical")
    queries.append(f"{company_name} logistics AI optimization")
    queries.append(f"{company_name} chatbot AI service")

    # Priority 9: Standard searches (broader fallback)
    queries.append(f"{company_name} official website")
    queries.append(f"{company_name} company technology")

    # Priority 10: Simple company name (last resort)
    queries.append(f"{company_name}")

    return queries


def scrape_website_content(url):
    """
    Downloads the website and extracts the main text using Trafilatura.
    """
    if not url:
        return ""

    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            # extract_metadata can sometimes get description/title if body is empty
            text = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
            return text if text else ""
    except Exception as e:
        logging.error(f"Scrape Error for {url}: {e}")
    return ""


def verify_website_relevance(text_content, company_name, url):
    """
    Verify if the website is relevant to the company and related to AI/tech industries.
    Returns WebsiteRelevance object.
    """
    if not text_content or len(text_content) < 50:
        return WebsiteRelevance(
            is_relevant=False,
            relevance_category="Not Relevant",
            confidence_score=0.0,
            reason="Insufficient content to verify"
        )

    # Clean and truncate text for token optimization
    clean_text = " ".join(text_content.split())[:8000]

    prompt = f"""
    Verify if this website belongs to the company '{company_name}' and is relevant to AI/technology industries.

    The company should be related to one or more of these categories and keywords:

    1. Robotics and Automation AI:
       - Physical Movement, Control, Cognition
       - Industrial robot, Cooperative robot
       - Robot control software
       - Transfer and Palletizing

    2. Vision AI:
       - Image and image recognition
       - Camera/Sensor based recognition
       - Image Analysis, defective inspection
       - Object, Face, Behavior Recognition
       - Medical and manufacturing vision

    3. AI Software and Platform:
       - Provides AI engine and tools without physical hardware
       - AI Development Platform
       - Data Analysis, MLOps, AI API

    4. Smart Factory and Manufacturing AI:
       - Manufacturing Process Optimization
       - Process automation, preservation of foresight
       - Production optimization, quality control

    5. Logistics and Mobility AI:
       - Movement, route, and transportation optimization
       - Self-driving, Autonomous vehicles
       - Drone, AGV/AMR
       - Logistics optimization

    6. Service, Education, and Healthcare AI:
       - People-to-people service
       - Educational AI
       - Medical and Health Care
       - Chatbot
       - Analysis of Emotions and Behaviors

    7. AI Semiconductor and Hardware:
       - Foundation technology for AI computation
       - AI chips, Edge devices, Sensors

    Website URL: {url}

    Website content:
    {clean_text}

    Analyze if:
    1. The website actually belongs to the company (not a different company or unrelated site)
    2. The company operates in any of the AI/tech categories and keywords listed above
    3. This is the official company website (not news, blog, social media about the company)

    Return is_relevant=true only if BOTH conditions are met:
    - The website clearly belongs to the company
    - The company is in AI/tech industry (matches one or more categories above)
    """

    try:
        response = client.models.generate_content(
            model='gemini-3-flash-preview',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type='application/json',
                response_schema=WebsiteRelevance,
            ),
        )
        return response.parsed
    except Exception as e:
        logging.error(f"Verification Error: {e}")
        return WebsiteRelevance(
            is_relevant=False,
            relevance_category="Error",
            confidence_score=0.0,
            reason=f"Error during verification: {str(e)}"
        )


def analyze_with_gemini(text_content, company_name, url):
    """
    Sends the website text to Gemini Flash to extract specific fields.
    """
    if not text_content:
        return CompanyInfo(location="Not Found", contact_email="Not Found", application_service="Not Found",
                           homepage_url=url or "Not Found")

    # Optimize tokens: Remove excessive whitespace/newlines and truncate
    # 10k chars is usually sufficient (~3-5k tokens)
    clean_text = " ".join(text_content.split())[:10000]

    prompt = f"""
    Analyze the website text for company '{company_name}'.
    Extract:
    1. location: Physical address or location (city, country, full address if available).
    2. contact_email: Public contact email address.
    3. application_service: Main product/service name.

    Text:
    {clean_text}
    """

    try:
        response = client.models.generate_content(
            model='gemini-3-flash-preview',  # Use Flash for speed/cost
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type='application/json',
                response_schema=CompanyInfo,
            ),
        )
        return response.parsed
    except Exception as e:
        logging.error(f"Gemini Error: {e}")
        return CompanyInfo(location="Error", contact_email="Error", application_service="Error", homepage_url=url)


# ================= MAIN EXECUTION =================

def load_data(file_path, logger, output_file=None, continue_previous=True):
    """
    Load data from Excel or CSV file (first sheet only).
    Handles duplicate column names by keeping only the first occurrence.
    Can continue from previous output file to preserve results.

    Args:
        file_path: Path to input file
        logger: Logger instance
        output_file: Path to output file (to check for previous results)
        continue_previous: If True and output_file exists, load from output_file instead

    Returns a Polars DataFrame.
    """
    # Check if we should continue from previous output
    if continue_previous and output_file and os.path.exists(output_file):
        logger.info(f"Found existing output file: {output_file}")
        logger.info(f"Loading from output file to continue previous work...")
        file_to_load = output_file
    else:
        logger.info(f"Loading fresh data from input file: {file_path}")
        file_to_load = file_path

    logger.info(f"Loading data from {file_to_load}...")

    # Handle potential Excel file or CSV
    if file_to_load.endswith('.xlsx'):
        try:
            # Read only the first sheet (sheet_id is 1-indexed in Polars)
            result = pl.read_excel(file_to_load, sheet_id=1)

            # If result is a dict (multiple sheets), get the first one
            if isinstance(result, dict):
                df = list(result.values())[0]
            else:
                df = result
        except Exception as e:
            logger.warning(f"Failed to read Excel with sheet_id, trying without: {e}")
            try:
                # Try reading without sheet_id (reads first sheet by default)
                result = pl.read_excel(file_to_load)
                if isinstance(result, dict):
                    df = list(result.values())[0]
                else:
                    df = result
            except Exception:
                logger.warning("Failed to read as Excel, trying as CSV")
                df = pl.read_csv(file_to_load)
    else:
        df = pl.read_csv(file_to_load)

    # Handle duplicate columns: Keep only first occurrence of Company, PIC, ROO, Position
    # Polars automatically renames duplicates as "column_name", "column_name_1", etc.
    columns_to_drop = []
    for col in df.columns:
        # Drop duplicate columns that have "_1", "_2" suffix
        if col.endswith(('_1', '_2', '_3')):
            base_name = col.rsplit('_', 1)[0]
            # Only drop if it's one of the columns we know are duplicated
            if base_name in ['Company', 'PIC', 'ROO', 'Position']:
                columns_to_drop.append(col)

    if columns_to_drop:
        df = df.drop(columns_to_drop)
        logger.info(f"Dropped duplicate columns: {columns_to_drop}")

    # Ensure Area column exists (we use Area instead of Location)
    if 'Area' not in df.columns:
        df = df.with_columns(pl.lit(None).alias('Area'))

    # Count how many rows already have data (if loading from output file)
    if file_to_load == output_file:
        filled_count = sum(1 for row in df.to_dicts()
                          if row.get('Homepage URL') and
                          row['Homepage URL'] not in [None, "Not Found", "", "None", "Error"])
        logger.info(f"📊 Previous results: {filled_count} rows already processed")

    logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    return df


def process_company(row, index, logger, search_location=None, search_language=None,
                    max_queries=3, max_urls_per_query=2, min_confidence=0.7):
    """
    Process a single company row with relevance verification.

    Args:
        max_queries: Maximum number of search queries to try (default: 3)
        max_urls_per_query: Maximum URLs to check per query (default: 2)
        min_confidence: Minimum confidence score to accept (default: 0.7)

    Returns True if processed successfully, False if failed.
    """
    company_name = row.get('Company') or ''

    if not company_name:
        row['Homepage URL'] = "Not Found"
        logger.warning(f"Row {index}: Empty company name")
        return True

    # Get PIC and ROO if available (optional fields)
    pic_name = row.get('PIC') or None
    roo_name = row.get('ROO') or None

    logger.info(f"Row {index}: Processing '{company_name}'" +
                (f" (PIC: {pic_name})" if pic_name else "") +
                (f" (ROO: {roo_name})" if roo_name else ""))

    # Step A: Build and try multiple search queries (includes PIC/ROO if available)
    all_queries = build_search_queries(company_name, pic_name, roo_name)
    search_queries = all_queries[:max_queries]  # Limit number of queries

    found_url = None
    verified_relevance = None
    website_text = None

    try:
        # Try each search query until we find a relevant website
        for query_idx, search_query in enumerate(search_queries):
            # Get top results for each query
            urls = search_google_serper(search_query, num_results=max_urls_per_query,
                                       location=search_location, language=search_language)

            if not urls:
                continue

            # Step B: Try each URL and verify relevance
            for url in urls[:max_urls_per_query]:  # Limit URLs checked
                # Scrape the website
                text = scrape_website_content(url)

                if not text or len(text) < 50:
                    continue

                # Step C: Verify if website is relevant to company and AI/tech industry
                relevance = verify_website_relevance(text, company_name, url)

                # If we found a relevant website with good confidence, use it
                if relevance.is_relevant and relevance.confidence_score >= min_confidence:
                    found_url = url
                    verified_relevance = relevance
                    website_text = text
                    logger.info(f"Row {index}: ✓ Found relevant site for '{company_name}': {url}")
                    logger.info(f"Row {index}:   Category: {relevance.relevance_category}, Confidence: {relevance.confidence_score:.2f}")
                    break

            # If we found a relevant site, stop trying other queries
            if found_url:
                break

            # Small delay between query attempts (only if trying more queries)
            if query_idx < len(search_queries) - 1:
                time.sleep(0.2)

        # Step D: Extract detailed information if we found a relevant website
        if found_url and website_text:
            extracted_data = analyze_with_gemini(website_text, company_name, found_url)

            # Update existing columns from Excel
            row['Homepage URL'] = found_url
            row['Area'] = extracted_data.location
            row['Application'] = extracted_data.application_service
            row['Email'] = extracted_data.contact_email
            logger.info(f"Row {index}: ✓ Successfully extracted data for '{company_name}'")
        else:
            # No relevant website found
            row['Homepage URL'] = "Not Found"
            reason = verified_relevance.reason if verified_relevance else "No results found"
            logger.warning(f"Row {index}: ✗ No relevant site for '{company_name}': {reason}")

        return True

    except Exception as e:
        logger.error(f"Row {index}: Error processing '{company_name}': {str(e)}")
        row['Homepage URL'] = "Error"
        return False


def save_results(data, output_file):
    """
    Save the processed data to Excel file.
    """
    pl.DataFrame(data).write_excel(output_file)


def should_skip_row(row, skip_filled=True):
    """
    Determine if a row should be skipped based on existing data.
    """
    if not skip_filled:
        return False

    homepage = row.get('Homepage URL')
    return homepage is not None and homepage not in [None, "Not Found", "", "None"]


def main(
    input_file=INPUT_FILE,
    output_file=OUTPUT_FILE,
    start_row=None,
    end_row=None,
    skip_filled=True,
    save_interval=10,
    rate_limit=0.5,
    resume_from_checkpoint=False,
    continue_previous=True,
    search_location=None,
    search_language=None,
    max_queries=3,
    max_urls_per_query=2,
    min_confidence=0.7
):
    """
    Main execution function with configurable parameters.

    Args:
        input_file: Path to input Excel/CSV file
        output_file: Path to output Excel file
        start_row: Starting row index (0-based, None = from beginning)
        end_row: Ending row index (0-based, None = to end)
        skip_filled: Skip rows that already have Homepage URL filled
        save_interval: Save progress every N rows
        rate_limit: Sleep time in seconds between API calls
        resume_from_checkpoint: Resume from last saved checkpoint
        continue_previous: Continue from output file if exists (preserves previous results)
        search_location: Country code for search (e.g., 'us', 'kr', 'jp') or None for global
        search_language: Language code for search (e.g., 'en', 'ko', 'ja') or None for auto
        max_queries: Maximum search queries to try per company (default: 3, range: 1-10)
        max_urls_per_query: Maximum URLs to verify per query (default: 2, range: 1-3)
        min_confidence: Minimum confidence score to accept result (default: 0.7, range: 0.5-1.0)
    """
    # Initialize logging
    logger = setup_logging()
    logger.info("="*60)
    logger.info("Starting data extraction process")
    logger.info("="*60)

    try:
        # 1. Load Data (continues from output file if exists)
        df = load_data(input_file, logger, output_file, continue_previous)

        # Convert to list of dicts for mutable iteration
        data = df.to_dicts()
        total_data_rows = len(data)

        # 2. Determine row range to process
        # Check for checkpoint first if resume is requested
        if resume_from_checkpoint:
            checkpoint = load_checkpoint()
            if checkpoint:
                logger.info(f"Found checkpoint from {checkpoint['timestamp']}")
                logger.info(f"Last processed row: {checkpoint['last_row_index']}")
                start_row = checkpoint['last_row_index'] + 1
                logger.info(f"Resuming from row {start_row}")
            else:
                logger.info("No checkpoint found, starting from beginning")

        # Set default range if not specified
        actual_start = start_row if start_row is not None else 0
        actual_end = end_row if end_row is not None else total_data_rows

        # Validate range
        actual_start = max(0, min(actual_start, total_data_rows))
        actual_end = max(actual_start, min(actual_end, total_data_rows))

        rows_to_process = data[actual_start:actual_end]
        total_rows_to_process = len(rows_to_process)

        logger.info(f"Processing rows {actual_start} to {actual_end-1} ({total_rows_to_process} rows)")
        logger.info(f"Total rows in file: {total_data_rows}")
        logger.info(f"Continue mode: {'Enabled - preserving previous results' if continue_previous else 'Disabled - fresh start'}")
        logger.info(f"Performance settings:")
        logger.info(f"  - Max queries per company: {max_queries}")
        logger.info(f"  - Max URLs per query: {max_urls_per_query}")
        logger.info(f"  - Min confidence: {min_confidence}")
        logger.info(f"  - Rate limit: {rate_limit}s")
        logger.info(f"  - Save interval: {save_interval} rows")
        logger.info(f"  - Skip filled: {skip_filled}")

        # 3. Process Row by Row
        processed_count = 0
        success_count = 0
        error_count = 0
        skipped_count = 0

        for local_index, row in tqdm(enumerate(rows_to_process), total=total_rows_to_process, desc="Processing"):
            actual_index = actual_start + local_index

            # Skip if already filled (useful if script crashes and you restart)
            if should_skip_row(row, skip_filled):
                skipped_count += 1
                continue

            # Process the company
            success = process_company(row, actual_index, logger, search_location, search_language,
                                     max_queries, max_urls_per_query, min_confidence)
            processed_count += 1

            if success:
                success_count += 1
            else:
                error_count += 1

            # Rate Limiting (Politeness + API limits)
            time.sleep(rate_limit)

            # Save every N rows to prevent data loss
            if processed_count % save_interval == 0:
                save_results(data, output_file)
                save_checkpoint(processed_count, total_rows_to_process, actual_index)
                logger.info(f"Progress saved at row {actual_index} ({processed_count}/{total_rows_to_process} processed)")

        # 4. Final Save
        save_results(data, output_file)
        logger.info("="*60)
        logger.info("Processing completed successfully!")
        logger.info(f"Total processed: {processed_count} companies")
        logger.info(f"Success: {success_count}, Errors: {error_count}, Skipped: {skipped_count}")
        logger.info(f"Results saved to: {output_file}")
        logger.info("="*60)

        # Clear checkpoint on successful completion
        clear_checkpoint()

    except KeyboardInterrupt:
        logger.warning("\n" + "="*60)
        logger.warning("Process interrupted by user!")
        logger.warning(f"Progress saved to: {output_file}")
        logger.warning(f"Checkpoint saved. Use resume_from_checkpoint=True to continue")
        logger.warning("="*60)
        # Save current progress if possible
        try:
            save_results(data, output_file)
            if 'processed_count' in locals() and 'total_rows_to_process' in locals() and 'actual_index' in locals():
                save_checkpoint(processed_count, total_rows_to_process, actual_index)
        except Exception as save_err:
            logger.error(f"Could not save progress on interrupt: {save_err}")

    except Exception as e:
        logger.error("="*60)
        logger.error(f"Fatal error occurred: {str(e)}")
        logger.error("="*60)
        logger.exception(e)
        # Try to save what we have
        try:
            if 'data' in locals():
                save_results(data, output_file)
            if 'processed_count' in locals() and 'total_rows_to_process' in locals() and 'actual_index' in locals():
                save_checkpoint(processed_count, total_rows_to_process, actual_index)
            logger.info("Emergency save completed")
        except Exception as save_err:
            logger.error(f"Could not save emergency backup: {save_err}")
        raise


if __name__ == "__main__":
    # ================= EXAMPLE CONFIGURATIONS =================
    # Time estimates for 1800 rows based on configuration:
    # - FAST mode: ~2-3 hours (3-6 sec/row)
    # - BALANCED mode: ~3-4 hours (6-8 sec/row)
    # - THOROUGH mode: ~5-7 hours (10-14 sec/row)

    # ========== SPEED PROFILES ==========

    # FAST MODE (Recommended for large datasets like 1800 rows)
    # Estimated: ~2-3 hours for 1800 rows
    # main(
    #     max_queries=2,           # Try only 2 search queries
    #     max_urls_per_query=1,    # Check only top result per query
    #     min_confidence=0.65,     # Accept slightly lower confidence
    #     rate_limit=0.3,          # Faster rate limit
    #     save_interval=20         # Save less frequently
    # )

    # BALANCED MODE (Default - good quality/speed tradeoff)
    # Estimated: ~3-4 hours for 1800 rows
    # main(
    #     max_queries=3,
    #     max_urls_per_query=2,
    #     min_confidence=0.7,
    #     rate_limit=0.5,
    #     save_interval=10
    # )

    # THOROUGH MODE (Best quality, slower)
    # Estimated: ~5-7 hours for 1800 rows
    # main(
    #     max_queries=5,
    #     max_urls_per_query=3,
    #     min_confidence=0.75,
    #     rate_limit=0.5,
    #     save_interval=10
    # )

    # ========== COMMON USE CASES ==========

    # Test first 10 rows with FAST mode
    main(start_row=705, end_row=1763, max_queries=2, max_urls_per_query=1, rate_limit=0.2)

    # Process specific range (e.g., rows 100-200)
    # main(start_row=100, end_row=200)

    # Resume after interruption
    # main(resume_from_checkpoint=True)

    # Process all with Korean location filter
    # main(search_location='kr', search_language='ko')

    # ========== MULTI-DAY BATCH PROCESSING (1800 rows) ==========
    # Process in batches over multiple days - results accumulate automatically!
    # The script will load previous results and continue from where you left off.

    # DAY 1: Process rows 0-300 (30-45 minutes)
    # main(start_row=0, end_row=300, max_queries=2, max_urls_per_query=1)

    # DAY 2: Process rows 300-600 (30-45 minutes)
    # Previous results (0-299) are preserved automatically!
    # main(start_row=300, end_row=600, max_queries=2, max_urls_per_query=1)

    # DAY 3: Process rows 600-900 (30-45 minutes)
    # main(start_row=600, end_row=900, max_queries=2, max_urls_per_query=1)

    # DAY 4: Process rows 900-1200 (30-45 minutes)
    # main(start_row=900, end_row=1200, max_queries=2, max_urls_per_query=1)

    # DAY 5: Process rows 1200-1500 (30-45 minutes)
    # main(start_row=1200, end_row=1500, max_queries=2, max_urls_per_query=1)

    # DAY 6: Process rows 1500-1800 (30-45 minutes)
    # main(start_row=1500, end_row=1800, max_queries=2, max_urls_per_query=1)

    # Result: All 1800 rows in ONE file after 6 days! 🎉

    # ========== DISABLE CONTINUE MODE (Fresh Start) ==========
    # If you want to start fresh and ignore previous results:
    # main(continue_previous=False)