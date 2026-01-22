"""
Company Information Enrichment and Data Extraction System

This automated data enrichment pipeline processes company datasets to extract and verify
comprehensive information about AI/technology companies. The system combines web search,
content scraping, AI-powered verification, and structured data extraction.

Core Workflow:
1. **Search Phase**: Uses Serper.dev API to find official company websites through
   intelligent, AI-focused search queries with fallback strategies
2. **Scraping Phase**: Extracts clean text content from websites using Trafilatura,
   handling various HTML structures and content types
3. **Verification Phase**: Employs Google Gemini AI to verify website relevance to
   the company and AI/tech industry, with confidence scoring
4. **Extraction Phase**: Uses Gemini AI to extract structured information (location,
   contact email, products/services) from verified websites

Key Features:
- **Intelligent Caching**: LRU-based memory cache with optional disk persistence to
  minimize API calls and improve performance across runs
- **Progress Tracking**: Automatic checkpointing allows resuming after interruptions
  without losing progress
- **Batch Processing**: Configurable speed profiles (FAST/BALANCED/THOROUGH) for
  different dataset sizes and quality requirements
- **Multi-day Support**: Continue mode preserves previous results, enabling processing
  large datasets over multiple sessions
- **Comprehensive Logging**: Detailed logs with timestamps for debugging and monitoring
- **Error Handling**: Robust error recovery with emergency save capabilities
- **Performance Monitoring**: Cache statistics and hit rates for optimization insights

Target Use Cases:
- Enriching company databases with verified website information
- Building AI/tech company directories with structured data
- Validating and categorizing companies by AI technology focus
- Batch processing large datasets (100s-1000s of companies)

Performance:
- FAST mode: ~3-6 seconds per company (recommended for 1000+ companies)
- BALANCED mode: ~6-8 seconds per company (default, good quality/speed)
- THOROUGH mode: ~10-14 seconds per company (maximum accuracy)

Requirements:
- SERPER_API_KEY: For Google search via Serper.dev
- GEMINI_API_KEY: For AI-powered verification and extraction
- Input: Excel/CSV file with company names (optional: PIC, ROO fields)
- Output: Enriched Excel file with Homepage URL, Area, Application, Email

Author: Data Enrichment Team
Version: 2.0
Last Updated: 2026-01-22
"""

import polars as pl
import requests
import trafilatura
import time
import json
import os
import logging
import hashlib
import re
from urllib.parse import urlparse, urlunparse
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from tqdm import tqdm
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from functools import lru_cache
from collections import OrderedDict

# Load environment variables from .env file
load_dotenv()

# ================= CONFIGURATION =================
INPUT_FILE = "data/companies.xlsx"  # Input file from data folder
OUTPUT_FILE = "data/companies_enriched.xlsx"  # Output file to data folder
PROGRESS_FILE = "data/progress_checkpoint.json"  # Progress tracking file
CACHE_FILE = "data/cache.json"  # Persistent cache file
LOG_DIR = "logs"  # Log directory
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Validate API keys
if not SERPER_API_KEY or not GEMINI_API_KEY:
    raise ValueError(
        "API keys not found. Please create a .env file with SERPER_API_KEY and GEMINI_API_KEY"
    )

# Initialize Gemini Client
client = genai.Client(api_key=GEMINI_API_KEY)


# ================= LOGGING & PROGRESS TRACKING =================


def setup_logging():
    """
    Initialize comprehensive logging system with file and console output.
    
    Creates a timestamped log file in the logs directory and configures
    logging to write to both the file and console simultaneously. This
    ensures all processing steps, errors, and statistics are captured
    for debugging and monitoring purposes.
    
    The log format includes timestamp, log level, and message for easy
    parsing and analysis. All logs use UTF-8 encoding to support
    international characters in company names and addresses.
    
    Returns:
        logging.Logger: Configured logger instance for the module
        
    Side Effects:
        - Creates 'logs' directory if it doesn't exist
        - Creates new log file with format: data_extract_YYYYMMDD_HHMMSS.log
        - Configures global logging settings
    """
    # Create logs directory if it doesn't exist
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(LOG_DIR) / f"data_extract_{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),  # Also print to console
        ],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def save_checkpoint(processed_rows, total_rows, last_row_index):
    """
    Persist processing progress to enable resumption after interruptions.
    
    Creates a JSON checkpoint file containing the current processing state,
    allowing the script to resume from the last processed row if interrupted
    by errors, user cancellation, or system issues. This prevents data loss
    and avoids reprocessing already completed rows.
    
    Args:
        processed_rows (int): Number of rows successfully processed so far
        total_rows (int): Total number of rows in the current batch
        last_row_index (int): Zero-based index of the last processed row
        
    Side Effects:
        - Creates 'data' directory if it doesn't exist
        - Writes/overwrites progress_checkpoint.json file
        - Includes ISO format timestamp for tracking
    """
    checkpoint = {
        "timestamp": datetime.now().isoformat(),
        "processed_rows": processed_rows,
        "total_rows": total_rows,
        "last_row_index": last_row_index,
    }

    # Ensure data directory exists
    Path("data").mkdir(parents=True, exist_ok=True)

    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)


def load_checkpoint():
    """
    Retrieve saved processing progress from previous run.
    
    Attempts to load the checkpoint file created by save_checkpoint().
    If the file exists and is valid JSON, returns the checkpoint data
    to allow resuming from the last processed row. If the file doesn't
    exist or is corrupted, returns None to start from the beginning.
    
    Returns:
        dict or None: Checkpoint data containing:
            - timestamp (str): ISO format timestamp of last save
            - processed_rows (int): Number of rows processed
            - total_rows (int): Total rows in batch
            - last_row_index (int): Last processed row index
        Returns None if no valid checkpoint exists
        
    Note:
        Logs a warning if checkpoint file exists but cannot be loaded
    """
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
                return checkpoint
        except Exception as e:
            logging.warning(f"Could not load checkpoint: {e}")
    return None


def clear_checkpoint():
    """
    Remove checkpoint file after successful processing completion.
    
    Deletes the progress checkpoint file once all rows have been
    successfully processed. This prevents accidentally resuming
    from an old checkpoint on the next run.
    
    Side Effects:
        - Deletes progress_checkpoint.json if it exists
        - Logs info message when checkpoint is cleared
        - Silently succeeds if file doesn't exist
    """
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        logging.info("Checkpoint cleared")


# ================= CACHING SYSTEM =================


class CacheStats:
    """
    Performance monitoring for caching system with detailed hit/miss tracking.
    
    Tracks cache effectiveness across all four cache types (search, scrape,
    verification, extraction) to provide insights into API call savings and
    cache efficiency. Used for performance optimization and cost analysis.
    
    Attributes:
        search_hits (int): Number of search results served from cache
        search_misses (int): Number of search API calls made
        scrape_hits (int): Number of website contents served from cache
        scrape_misses (int): Number of websites scraped
        verification_hits (int): Number of verifications served from cache
        verification_misses (int): Number of Gemini verification API calls
        extraction_hits (int): Number of extractions served from cache
        extraction_misses (int): Number of Gemini extraction API calls
        
    Methods:
        get_*_hit_rate(): Calculate hit rate percentage for each cache type
        get_total_api_calls_saved(): Calculate total API calls avoided
        print_stats(): Output formatted statistics to logger
    """

    def __init__(self):
        self.search_hits = 0
        self.search_misses = 0
        self.scrape_hits = 0
        self.scrape_misses = 0
        self.verification_hits = 0
        self.verification_misses = 0
        self.extraction_hits = 0
        self.extraction_misses = 0

    def record_search_hit(self):
        self.search_hits += 1

    def record_search_miss(self):
        self.search_misses += 1

    def record_scrape_hit(self):
        self.scrape_hits += 1

    def record_scrape_miss(self):
        self.scrape_misses += 1

    def record_verification_hit(self):
        self.verification_hits += 1

    def record_verification_miss(self):
        self.verification_misses += 1

    def record_extraction_hit(self):
        self.extraction_hits += 1

    def record_extraction_miss(self):
        self.extraction_misses += 1

    def get_search_hit_rate(self) -> float:
        total = self.search_hits + self.search_misses
        return (self.search_hits / total * 100) if total > 0 else 0.0

    def get_scrape_hit_rate(self) -> float:
        total = self.scrape_hits + self.scrape_misses
        return (self.scrape_hits / total * 100) if total > 0 else 0.0

    def get_verification_hit_rate(self) -> float:
        total = self.verification_hits + self.verification_misses
        return (self.verification_hits / total * 100) if total > 0 else 0.0

    def get_extraction_hit_rate(self) -> float:
        total = self.extraction_hits + self.extraction_misses
        return (self.extraction_hits / total * 100) if total > 0 else 0.0

    def get_total_api_calls_saved(self) -> int:
        """Calculate how many API calls were saved by caching."""
        return self.search_hits + self.verification_hits + self.extraction_hits

    def print_stats(self, logger):
        """Print cache statistics to logger."""
        logger.info("=" * 60)
        logger.info("CACHE PERFORMANCE STATISTICS")
        logger.info("=" * 60)
        logger.info(
            f"Search Cache:        {self.search_hits:4d} hits, {self.search_misses:4d} misses ({self.get_search_hit_rate():.1f}% hit rate)"
        )
        logger.info(
            f"Scrape Cache:        {self.scrape_hits:4d} hits, {self.scrape_misses:4d} misses ({self.get_scrape_hit_rate():.1f}% hit rate)"
        )
        logger.info(
            f"Verification Cache:  {self.verification_hits:4d} hits, {self.verification_misses:4d} misses ({self.get_verification_hit_rate():.1f}% hit rate)"
        )
        logger.info(
            f"Extraction Cache:    {self.extraction_hits:4d} hits, {self.extraction_misses:4d} misses ({self.get_extraction_hit_rate():.1f}% hit rate)"
        )
        logger.info(f"Total API calls saved: {self.get_total_api_calls_saved()}")
        logger.info("=" * 60)


class LRUCache:
    """
    Least Recently Used (LRU) cache implementation with automatic eviction.
    
    Implements an LRU caching strategy using OrderedDict to maintain access
    order. When the cache reaches its maximum size, the least recently used
    entry is automatically evicted to make room for new entries. This prevents
    unbounded memory growth while maintaining frequently accessed data.
    
    The LRU strategy is ideal for this use case because:
    - Companies may appear multiple times in the dataset
    - Recent searches are more likely to be reused
    - Memory usage remains bounded even for large datasets
    
    Args:
        max_size (int): Maximum number of entries before eviction starts.
            Default is 1000 entries. Adjust based on available memory.
            
    Attributes:
        cache (OrderedDict): Ordered dictionary maintaining insertion/access order
        max_size (int): Maximum cache capacity
        
    Methods:
        get(key): Retrieve value and mark as recently used
        set(key, value): Store value and evict oldest if needed
        clear(): Remove all entries
        size(): Get current number of entries
    """

    def __init__(self, max_size: int = 1000):
        self.cache: OrderedDict = OrderedDict()
        self.max_size = max_size

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache, mark as recently used."""
        if key not in self.cache:
            return None
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]

    def set(self, key: str, value: Any):
        """Set item in cache, evict oldest if needed."""
        if key in self.cache:
            # Update existing key, move to end
            self.cache.move_to_end(key)
        self.cache[key] = value

        # Evict oldest if over size limit
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)  # Remove oldest (first) item

    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()

    def size(self) -> int:
        """Return current cache size."""
        return len(self.cache)


class DataCache:
    """
    Comprehensive multi-level caching system for API results and web content.
    
    Provides intelligent caching across all four stages of the enrichment pipeline:
    1. Search results from Serper API
    2. Scraped website content from Trafilatura
    3. Website relevance verification from Gemini AI
    4. Company information extraction from Gemini AI
    
    Features:
    - **Separate LRU Caches**: Each data type has its own cache with appropriate
      sizing (search/scrape get full size, verification/extraction get half)
    - **Hash-based Keys**: Uses MD5 hashing of parameters for consistent lookups
    - **Pydantic Conversion**: Automatically converts Pydantic models to dicts
      for JSON serialization
    - **Disk Persistence**: Optional save/load from JSON file for cross-run caching
    - **Statistics Tracking**: Monitors hit/miss rates for performance analysis
    
    Cache Sizing Strategy:
    - Search cache: Full size (most likely to be reused)
    - Scrape cache: Full size (expensive to re-fetch)
    - Verification cache: Half size (less likely to be reused)
    - Extraction cache: Half size (less likely to be reused)
    
    Args:
        enable_disk_cache (bool): Whether to persist cache to disk between runs.
            Recommended for multi-day batch processing. Default: False
        cache_file (str): Path to JSON file for persistent storage.
            Default: 'data/cache.json'
        max_memory_size (int): Maximum entries per full-size cache.
            Default: 1000. Adjust based on available RAM.
            
    Attributes:
        stats (CacheStats): Performance statistics tracker
        search_cache (LRUCache): Cache for Serper search results
        scrape_cache (LRUCache): Cache for scraped website content
        verification_cache (LRUCache): Cache for Gemini verification results
        extraction_cache (LRUCache): Cache for Gemini extraction results
    """

    def __init__(
        self,
        enable_disk_cache: bool = False,
        cache_file: str = CACHE_FILE,
        max_memory_size: int = 1000,
    ):
        """
        Initialize the cache system.

        Args:
            enable_disk_cache: Whether to persist cache to disk
            cache_file: Path to cache file for persistent storage
            max_memory_size: Maximum number of entries in memory cache (LRU eviction)
        """
        self.enable_disk_cache = enable_disk_cache
        self.cache_file = cache_file
        self.stats = CacheStats()

        # Separate LRU caches for different data types
        self.search_cache = LRUCache(max_size=max_memory_size)
        self.scrape_cache = LRUCache(max_size=max_memory_size)
        self.verification_cache = LRUCache(
            max_size=max_memory_size // 2
        )  # Smaller, less likely to reuse
        self.extraction_cache = LRUCache(max_size=max_memory_size // 2)

        # Load persistent cache if enabled
        if self.enable_disk_cache:
            self._load_from_disk()

    @staticmethod
    def _hash_key(*args) -> str:
        """Create a hash key from arguments for cache lookup."""
        key_string = "|".join(str(arg) for arg in args)
        return hashlib.md5(key_string.encode()).hexdigest()

    def get_search_results(
        self,
        query: str,
        num_results: int,
        location: Optional[str],
        language: Optional[str],
    ) -> Optional[List[str]]:
        """Get cached search results."""
        key = self._hash_key(query, num_results, location or "", language or "")
        result = self.search_cache.get(key)

        if result is not None:
            self.stats.record_search_hit()
        else:
            self.stats.record_search_miss()

        return result

    def set_search_results(
        self,
        query: str,
        num_results: int,
        location: Optional[str],
        language: Optional[str],
        results: List[str],
    ):
        """Cache search results."""
        key = self._hash_key(query, num_results, location or "", language or "")
        self.search_cache.set(key, results)

    def get_scraped_content(self, url: str) -> Optional[str]:
        """Get cached website content."""
        key = self._hash_key(url)
        result = self.scrape_cache.get(key)

        if result is not None:
            self.stats.record_scrape_hit()
        else:
            self.stats.record_scrape_miss()

        return result

    def set_scraped_content(self, url: str, content: str):
        """Cache website content."""
        key = self._hash_key(url)
        self.scrape_cache.set(key, content)

    def get_verification_result(self, url: str, company_name: str) -> Optional[Dict]:
        """Get cached verification result."""
        key = self._hash_key(url, company_name)
        result = self.verification_cache.get(key)

        if result is not None:
            self.stats.record_verification_hit()
        else:
            self.stats.record_verification_miss()

        return result

    def set_verification_result(self, url: str, company_name: str, result: Any):
        """Cache verification result (convert Pydantic to dict)."""
        key = self._hash_key(url, company_name)
        # Convert Pydantic model to dict for JSON serialization
        if hasattr(result, "model_dump"):
            result_dict = result.model_dump()
        elif hasattr(result, "dict"):
            result_dict = result.dict()
        else:
            result_dict = result
        self.verification_cache.set(key, result_dict)

    def get_extraction_result(self, url: str, company_name: str) -> Optional[Dict]:
        """Get cached extraction result."""
        key = self._hash_key(url, company_name)
        result = self.extraction_cache.get(key)

        if result is not None:
            self.stats.record_extraction_hit()
        else:
            self.stats.record_extraction_miss()

        return result

    def set_extraction_result(self, url: str, company_name: str, result: Any):
        """Cache extraction result (convert Pydantic to dict)."""
        key = self._hash_key(url, company_name)
        # Convert Pydantic model to dict for JSON serialization
        if hasattr(result, "model_dump"):
            result_dict = result.model_dump()
        elif hasattr(result, "dict"):
            result_dict = result.dict()
        else:
            result_dict = result
        self.extraction_cache.set(key, result_dict)

    def _load_from_disk(self):
        """Load cache from disk if file exists."""
        if not os.path.exists(self.cache_file):
            logging.info("No persistent cache file found, starting fresh")
            return

        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Load each cache type
            if "search" in data:
                for key, value in data["search"].items():
                    self.search_cache.set(key, value)

            if "scrape" in data:
                for key, value in data["scrape"].items():
                    self.scrape_cache.set(key, value)

            if "verification" in data:
                for key, value in data["verification"].items():
                    self.verification_cache.set(key, value)

            if "extraction" in data:
                for key, value in data["extraction"].items():
                    self.extraction_cache.set(key, value)

            logging.info(f"Loaded persistent cache from {self.cache_file}")
            logging.info(f"  Search entries: {self.search_cache.size()}")
            logging.info(f"  Scrape entries: {self.scrape_cache.size()}")
            logging.info(f"  Verification entries: {self.verification_cache.size()}")
            logging.info(f"  Extraction entries: {self.extraction_cache.size()}")

        except Exception as e:
            logging.warning(f"Could not load cache from disk: {e}")

    def save_to_disk(self):
        """Save cache to disk for persistence across runs."""
        if not self.enable_disk_cache:
            return

        try:
            # Ensure data directory exists
            Path("data").mkdir(parents=True, exist_ok=True)

            data = {
                "search": dict(self.search_cache.cache),
                "scrape": dict(self.scrape_cache.cache),
                "verification": dict(self.verification_cache.cache),
                "extraction": dict(self.extraction_cache.cache),
                "saved_at": datetime.now().isoformat(),
            }

            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logging.info(f"Saved cache to disk: {self.cache_file}")

        except Exception as e:
            logging.error(f"Could not save cache to disk: {e}")

    def clear_all(self):
        """Clear all caches."""
        self.search_cache.clear()
        self.scrape_cache.clear()
        self.verification_cache.clear()
        self.extraction_cache.clear()
        logging.info("All caches cleared")

    def get_stats(self) -> CacheStats:
        """Return cache statistics."""
        return self.stats


# Global cache instance (initialized in main)
_cache: Optional[DataCache] = None


def get_cache() -> Optional[DataCache]:
    """Get the global cache instance."""
    return _cache


def init_cache(
    enable_disk_cache: bool = False, max_memory_size: int = 1000
) -> DataCache:
    """
    Initialize and configure the global caching system.
    
    Creates a singleton DataCache instance that will be used throughout
    the script execution. This function should be called once at the
    start of main() before any processing begins.
    
    The cache significantly improves performance by:
    - Avoiding duplicate API calls for the same searches
    - Preventing re-scraping of already visited websites
    - Reusing AI verification and extraction results
    - Reducing costs by minimizing API usage
    
    Args:
        enable_disk_cache (bool): If True, cache persists to disk and loads
            on next run. Highly recommended for multi-day batch processing
            or when processing datasets with duplicate companies. Default: False
        max_memory_size (int): Maximum number of entries in each full-size
            cache before LRU eviction begins. Larger values use more memory
            but improve hit rates. Default: 1000
            
    Returns:
        DataCache: Configured cache instance ready for use
        
    Side Effects:
        - Sets global _cache variable
        - Loads existing cache from disk if enable_disk_cache=True
        - Logs cache initialization status
        
    Example:
        >>> cache = init_cache(enable_disk_cache=True, max_memory_size=2000)
        >>> # Cache is now available via get_cache() throughout the script
    """
    global _cache
    _cache = DataCache(
        enable_disk_cache=enable_disk_cache, max_memory_size=max_memory_size
    )
    return _cache


def clear_cache():
    """
    Clear all in-memory caches and delete persistent cache file.
    
    Removes all cached data from memory and deletes the disk cache file
    if it exists. Use this function to force fresh API calls for all
    operations, typically for testing or when cache data is suspected
    to be stale or corrupted.
    
    Side Effects:
        - Clears all four LRU caches (search, scrape, verification, extraction)
        - Deletes cache.json file if it exists
        - Logs success/failure messages
        - Resets cache statistics
        
    Warning:
        This operation cannot be undone. All cached data will be lost
        and subsequent operations will require fresh API calls.
    """
    cache = get_cache()
    if cache:
        cache.clear_all()
        logging.info("Memory cache cleared")

        # Also delete disk cache file if it exists
        if os.path.exists(CACHE_FILE):
            try:
                os.remove(CACHE_FILE)
                logging.info(f"Disk cache file deleted: {CACHE_FILE}")
            except Exception as e:
                logging.error(f"Could not delete cache file: {e}")
    else:
        logging.warning("No cache instance to clear")


# ================= DATA MODELS =================

class CompanyInfo(BaseModel):
    """
    Structured data model for extracted company information.
    
    Defines the schema for company data extracted by Gemini AI from
    website content. Used with Pydantic for automatic validation and
    JSON serialization. Gemini AI is instructed to extract these specific
    fields from the website text.
    
    Attributes:
        location (str): Physical address or location of the company.
            Can be city, country, or full address. Examples:
            - "Seoul, South Korea"
            - "123 Tech Street, San Francisco, CA 94105"
            - "Tokyo, Japan"
        contact_email (str): Public contact email address found on website.
            Typically info@, contact@, or sales@ addresses. Examples:
            - "info@company.com"
            - "contact@example.co.kr"
        application_service (str): Name of main product, service, or application.
            The primary offering of the company. Examples:
            - "Industrial Robot Control System"
            - "AI-powered Vision Inspection Platform"
            - "Autonomous Mobile Robot (AMR)"
        homepage_url (str): The official website URL that was analyzed.
            Full URL including protocol. Example: "https://company.com"
    """
    location: str = Field(..., description="Physical address or location of the company")
    contact_email: str = Field(..., description="Contact email address found on the page")
    application_service: str = Field(..., description="Name of main application, service, or product")
    homepage_url: str = Field(..., description="The official website URL")


class WebsiteRelevance(BaseModel):
    """
    Structured verification result for website relevance assessment.
    
    Defines the schema for Gemini AI's analysis of whether a website
    belongs to the target company and operates in the AI/tech industry.
    Includes confidence scoring and categorization for quality control.
    
    Attributes:
        is_relevant (bool): True if website belongs to the company AND
            the company operates in AI/tech industry. Both conditions
            must be met. False otherwise.
        relevance_category (str): Primary AI/tech category. One of:
            - "Robotics and Automation AI": Physical robots, control systems
            - "Vision AI": Image recognition, computer vision
            - "AI Software and Platform": AI tools, APIs, development platforms
            - "Smart Factory and Manufacturing AI": Production optimization
            - "Logistics and Mobility AI": Autonomous vehicles, route optimization
            - "Service/Education/Healthcare AI": Chatbots, medical AI
            - "AI Semiconductor and Hardware": AI chips, edge devices
            - "Not Relevant": Not AI/tech or wrong company
        confidence_score (float): Confidence level from 0.0 to 1.0.
            Higher scores indicate stronger evidence. Typical thresholds:
            - 0.9-1.0: Very high confidence
            - 0.7-0.9: Good confidence (default minimum)
            - 0.5-0.7: Moderate confidence
            - 0.0-0.5: Low confidence
        reason (str): Brief explanation for the decision, citing specific
            keywords or evidence found on the website. Examples:
            - "Website mentions 'industrial robots' and 'automation systems'"
            - "Different company - this is a news site about the company"
            - "No AI/tech focus - appears to be a consulting firm"
    """
    is_relevant: bool = Field(..., description="Whether the website is relevant to the company and AI/tech industry")
    relevance_category: str = Field(..., description="Primary category: 'Robotics and Automation AI', 'Vision AI', 'AI Software and Platform', 'Smart Factory and Manufacturing AI', 'Logistics and Mobility AI', 'Service/Education/Healthcare AI', 'AI Semiconductor and Hardware', or 'Not Relevant'")
    confidence_score: float = Field(..., description="Confidence score from 0.0 to 1.0")
    reason: str = Field(
        ...,
        description="Brief reason for the relevance decision, mentioning specific keywords found",
    )


# ================= HELPER FUNCTIONS =================

LEGAL_SUFFIXES = {
    "inc",
    "inc.",
    "incorporated",
    "corp",
    "corp.",
    "corporation",
    "co",
    "co.",
    "company",
    "ltd",
    "ltd.",
    "limited",
    "llc",
    "plc",
    "gmbh",
    "s.a.",
    "s.a",
    "ag",
    "bv",
    "oy",
    "oyj",
    "sas",
    "sa",
    "kg",
    "합자회사",
    "주식회사",
    "유한회사",
}

OFFICIAL_TERMS = {
    "official",
    "homepage",
    "website",
    "about",
    "company",
    "corporate",
    "contact",
    "investor",
    "ir",
    "기업",
    "회사",
    "공식",
    "홈페이지",
    "회사",
    "공식",
    "홈페이지",
}

AI_TECH_TERMS = {
    "ai",
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "robotics",
    "robot",
    "automation",
    "vision",
    "computer vision",
    "manufacturing",
    "factory",
    "smart factory",
    "logistics",
    "mobility",
    "autonomous",
    "semiconductor",
    "chip",
    "edge",
    "healthcare",
    "medical",
    "chatbot",
}

LOW_SIGNAL_DOMAINS = {
    "linkedin.com",
    "facebook.com",
    "instagram.com",
    "twitter.com",
    "x.com",
    "youtube.com",
    "tiktok.com",
    "crunchbase.com",
    "wikipedia.org",
    "bloomberg.com",
    "reuters.com",
    "medium.com",
    "angel.co",
    "pitchbook.com",
    "glassdoor.com",
}


def normalize_company_name(company_name: str) -> str:
    """
    Normalize a company name for matching and query generation.

    Strips legal suffixes, punctuation, and redundant whitespace to produce
    a stable identifier for scoring search results.

    Args:
        company_name: Raw company name from the dataset

    Returns:
        Normalized company name in lowercase without legal suffixes
    """
    if not company_name:
        return ""

    cleaned = re.sub(r"[\.,;:()\[\]{}<>/\\\-]+", " ", company_name.lower())
    tokens = [token for token in cleaned.split() if token and token not in LEGAL_SUFFIXES]
    return " ".join(tokens).strip()


def extract_company_tokens(company_name: str) -> List[str]:
    """
    Extract meaningful company tokens for relevance scoring.

    Args:
        company_name: Raw company name

    Returns:
        List of lowercase tokens with legal suffixes removed
    """
    normalized = normalize_company_name(company_name)
    return [token for token in normalized.split() if len(token) > 1]


def dedupe_preserve_order(items: List[str]) -> List[str]:
    """
    Remove duplicates while preserving original order.

    Args:
        items: List of strings

    Returns:
        Deduplicated list with original ordering
    """
    seen = set()
    deduped = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def normalize_url(url: str) -> str:
    """
    Normalize a URL for consistent comparison and de-duplication.

    Args:
        url: Raw URL string

    Returns:
        Normalized URL without query/fragment and trailing slash
    """
    if not url:
        return ""
    parsed = urlparse(url.strip())
    normalized = parsed._replace(query="", fragment="")
    cleaned = urlunparse(normalized).rstrip("/")
    return cleaned.lower()


def score_search_result(
    result: Dict[str, Any],
    company_tokens: List[str],
    query_terms: List[str]
) -> float:
    """
    Score a Serper search result for relevance to the target company.

    The scoring model prioritizes official company sites by looking for
    company tokens in the domain/title/snippet, boosting official keywords
    and AI/tech terms, and penalizing low-signal domains.

    Args:
        result: Serper result object with link/title/snippet
        company_tokens: Normalized company name tokens
        query_terms: Lowercase query tokens for context

    Returns:
        Numeric relevance score (higher is better)
    """
    link = result.get("link", "")
    title = (result.get("title") or "").lower()
    snippet = (result.get("snippet") or "").lower()

    parsed = urlparse(link)
    domain = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()

    score = 0.0

    # Company token matches
    for token in company_tokens:
        if token in domain:
            score += 6.0
        if token in path:
            score += 2.0
        if token in title:
            score += 3.0
        if token in snippet:
            score += 1.5

    # Official/brand signals
    for term in OFFICIAL_TERMS:
        if term in title or term in snippet:
            score += 1.5
        if term in path:
            score += 0.5

    # AI/tech signals
    for term in AI_TECH_TERMS:
        if term in title or term in snippet:
            score += 0.75

    # Penalize low-signal domains (social/news/aggregators)
    if any(domain.endswith(bad) for bad in LOW_SIGNAL_DOMAINS):
        score -= 5.0

    # Minor preference for HTTPS
    if parsed.scheme == "https":
        score += 0.5

    # Favor exact query terms in title/snippet
    for term in query_terms:
        if term in title:
            score += 0.5
        if term in snippet:
            score += 0.25

    return score


def search_google_serper(
    query: str,
    num_results: int = 3,
    location: Optional[str] = None,
    language: Optional[str] = None,
    timeout: int = 10,
    company_name: Optional[str] = None
) -> List[str]:
    """
    Query Serper.dev and rank results for company relevance.

    Enhancements over raw Serper ordering:
    - Scores results using company token matches, official keywords, and AI/tech terms
    - Penalizes low-signal domains (social media, news aggregators)
    - De-duplicates URLs while preserving the highest-scoring results
    - Caches ranked results to reduce repeated API calls

    Args:
        query: Search query string
        num_results: Number of results to return (default: 3)
        location: Country code (e.g., 'us', 'kr', 'jp') or None for global
        language: Language code (e.g., 'en', 'ko', 'ja') or None for auto
        timeout: Request timeout in seconds (default: 10)
        company_name: Company name for result scoring (optional but recommended)

    Returns:
        List of URLs from ranked organic search results
    """
    # Check cache first
    cache = get_cache()
    if cache:
        cached_results = cache.get_search_results(
            query, num_results, location, language
        )
        if cached_results is not None:
            return cached_results

    url = "https://google.serper.dev/search"
    payload = {"q": query, "num": num_results}

    # Add optional location and language filters
    if location:
        payload["gl"] = location
    if language:
        payload["hl"] = language

    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        results = response.json()

        organic_results = results.get("organic") or []
        if not organic_results:
            return []

        query_terms = [term for term in query.lower().split() if len(term) > 1]
        company_tokens = extract_company_tokens(company_name or query)

        scored_results = []
        for idx, result in enumerate(organic_results):
            score = score_search_result(result, company_tokens, query_terms)
            scored_results.append((score, idx, result))

        scored_results.sort(key=lambda item: (-item[0], item[1]))

        ordered_links = []
        seen = set()
        for score, _, result in scored_results:
            link = result.get("link", "")
            normalized_link = normalize_url(link)
            if not normalized_link or normalized_link in seen:
                continue
            seen.add(normalized_link)
            ordered_links.append(link)
            if len(ordered_links) >= num_results:
                break

        if cache is not None:
            cache.set_search_results(query, num_results, location, language, ordered_links)

        return ordered_links
    except requests.exceptions.Timeout:
        logging.error(f"Search timeout for query: {query}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Search request error for '{query}': {e}")
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse search results: {e}")
    except Exception as e:
        logging.error(f"Search Error: {e}")
    return []


def build_search_queries(
    company_name: str,
    pic_name: Optional[str] = None,
    roo_name: Optional[str] = None
) -> List[str]:
    """
    Build prioritized search queries with AI/tech-focused keywords.

    The queries are ordered by priority, starting with the most specific
    AI/tech-focused searches and falling back to broader searches.

    Args:
        company_name: Company name (required)
        pic_name: Person In Charge name (optional)
        roo_name: ROO name (optional)

    Returns:
        List of search queries ordered by priority
    """
    normalized_name = normalize_company_name(company_name)
    normalized_company = normalized_name or company_name
    quoted_name = f"\"{company_name}\""

    queries = []

    # Priority 0: Exact company name matching
    queries.append(f"{quoted_name} official website")
    queries.append(f"{quoted_name} company")
    queries.append(f"{quoted_name} AI")

    # Priority 1: Official website with AI/tech context
    queries.append(f"{company_name} AI technology official website")
    queries.append(f"{normalized_company} artificial intelligence company")

    # Priority 2: Specific AI/tech industry keywords
    queries.append(f"{normalized_company} robotics automation AI")
    queries.append(f"{normalized_company} machine learning AI platform")
    queries.append(f"{normalized_company} computer vision AI")

    # Priority 3: Add PIC with AI context if available
    if pic_name:
        queries.append(f"{company_name} {pic_name} AI technology")
        queries.append(f"{company_name} {pic_name} robotics")

    # Priority 4: Manufacturing and industrial AI keywords
    queries.append(f"{normalized_company} smart factory AI manufacturing")
    queries.append(f"{normalized_company} industrial automation robotics")

    # Priority 5: Specific technology domains
    queries.append(f"{normalized_company} vision AI image recognition")
    queries.append(f"{normalized_company} autonomous vehicle self-driving")
    queries.append(f"{normalized_company} AGV AMR robotics")

    # Priority 6: Add ROO with tech context if available
    if roo_name:
        queries.append(f"{company_name} {roo_name} AI technology")

    # Priority 7: General tech and software keywords
    queries.append(f"{normalized_company} AI software platform")
    queries.append(f"{normalized_company} deep learning neural network")
    queries.append(f"{normalized_company} AI chip semiconductor")

    # Priority 8: Healthcare, service, logistics AI
    queries.append(f"{normalized_company} healthcare AI medical")
    queries.append(f"{normalized_company} logistics AI optimization")
    queries.append(f"{normalized_company} chatbot AI service")

    # Priority 9: Standard searches (broader fallback)
    queries.append(f"{company_name} official website")
    queries.append(f"{normalized_company} company technology")
    queries.append(f"{normalized_company} corporate site")
    queries.append(f"{normalized_company} about us")
    queries.append(f"{normalized_company} 홈페이지")
    # Korean-only and English-only targeting (no Chinese/Japanese terms)

    # Priority 10: Simple company name (last resort)
    queries.append(f"{company_name}")
    queries.append(f"{normalized_company}")

    return dedupe_preserve_order(queries)


def scrape_website_content(url: str, timeout: int = 30) -> str:
    """
    Downloads the website and extracts the main text using Trafilatura.
    """
    if not url:
        return ""

    # Check cache first
    cache = get_cache()
    if cache:
        cached_content = cache.get_scraped_content(url)
        if cached_content is not None:
            return cached_content

    try:
        downloaded = trafilatura.fetch_url(url, timeout=timeout)
        if downloaded:
            # extract_metadata can sometimes get description/title if body is empty
            text = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
            return text if text else ""
    except Exception as e:
        logging.error(f"Scrape error for {url}: {e}")
    return ""


def verify_website_relevance(
    text_content: str,
    company_name: str,
    url: str
) -> WebsiteRelevance:
    """
    Verify if the website is relevant to the company and related to AI/tech industries.
    Returns WebsiteRelevance object.
    """
    if not text_content or len(text_content) < 50:
        return WebsiteRelevance(
            is_relevant=False,
            relevance_category="Not Relevant",
            confidence_score=0.0,
            reason="Insufficient content to verify",
        )

    # Check cache first
    cache = get_cache()
    if cache:
        cached_result = cache.get_verification_result(url, company_name)
        if cached_result is not None:
            # Convert dict back to WebsiteRelevance object
            return WebsiteRelevance(**cached_result)

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
            model="gemini-3-flash-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=WebsiteRelevance,
            ),
        )
        result = response.parsed

        # Cache the result
        if cache:
            cache.set_verification_result(url, company_name, result)

        return result
    except Exception as e:
        logging.error(f"Verification Error: {e}")
        error_result = WebsiteRelevance(
            is_relevant=False,
            relevance_category="Error",
            confidence_score=0.0,
            reason=f"Error during verification: {str(e)}",
        )

        # Cache error results too (to avoid retrying)
        if cache:
            cache.set_verification_result(url, company_name, error_result)

        return error_result


def analyze_with_gemini(
    text_content: str,
    company_name: str,
    url: str
) -> CompanyInfo:
    """
    Sends the website text to Gemini Flash to extract specific fields.
    """
    if not text_content:
        return CompanyInfo(
            location="Not Found",
            contact_email="Not Found",
            application_service="Not Found",
            homepage_url=url or "Not Found",
        )

    # Check cache first
    cache = get_cache()
    if cache:
        cached_result = cache.get_extraction_result(url, company_name)
        if cached_result is not None:
            # Convert dict back to CompanyInfo object
            return CompanyInfo(**cached_result)

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
            model="gemini-3-flash-preview",  # Use Flash for speed/cost
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CompanyInfo,
            ),
        )
        result = response.parsed

        # Cache the result
        if cache:
            cache.set_extraction_result(url, company_name, result)

        return result
    except Exception as e:
        logging.error(f"Gemini Error: {e}")
        error_result = CompanyInfo(
            location="Error",
            contact_email="Error",
            application_service="Error",
            homepage_url=url,
        )

        # Cache error results too
        if cache:
            cache.set_extraction_result(url, company_name, error_result)

        return error_result


# ================= MAIN EXECUTION =================


def load_data(
    file_path: str,
    logger: logging.Logger,
    output_file: Optional[str] = None,
    continue_previous: bool = True
) -> pl.DataFrame:
    """
    Load data from Excel or CSV file with duplicate column handling.

    Automatically detects and handles:
    - Excel files (.xlsx) - reads first sheet only
    - CSV files
    - Duplicate column names (keeps first occurrence)
    - Previous output files for continuation

    Args:
        file_path: Path to input file
        logger: Logger instance
        output_file: Path to output file (to check for previous results)
        continue_previous: If True and output_file exists, load from output_file

    Returns:
        Polars DataFrame with loaded data
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
    if file_to_load.endswith(".xlsx"):
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
        if col.endswith(("_1", "_2", "_3")):
            base_name = col.rsplit("_", 1)[0]
            # Only drop if it's one of the columns we know are duplicated
            if base_name in ["Company", "PIC", "ROO", "Position"]:
                columns_to_drop.append(col)

    if columns_to_drop:
        df = df.drop(columns_to_drop)
        logger.info(f"Dropped duplicate columns: {columns_to_drop}")

    # Ensure Area column exists (we use Area instead of Location)
    if "Area" not in df.columns:
        df = df.with_columns(pl.lit(None).alias("Area"))

    # Count how many rows already have data (if loading from output file)
    if file_to_load == output_file:
        filled_count = sum(
            1
            for row in df.to_dicts()
            if row.get("Homepage URL")
            and row["Homepage URL"] not in [None, "Not Found", "", "None", "Error"]
        )
        logger.info(f"📊 Previous results: {filled_count} rows already processed")

    logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    return df


def process_company(row, index, logger, search_location=None, search_language=None,
                    max_queries=3, max_urls_per_query=2, min_confidence=0.7):
    """
    Process a single company row with multi-stage verification.

    Processing stages:
    1. Build prioritized search queries
    2. Search for candidate websites
    3. Scrape and verify website relevance
    4. Extract detailed company information

    Args:
        row: Company data row (dict)
        index: Row index for logging
        logger: Logger instance
        search_location: Country code for search localization
        search_language: Language code for search
        max_queries: Maximum search queries to try (default: 3)
        max_urls_per_query: Maximum URLs to verify per query (default: 2)
        min_confidence: Minimum confidence score to accept (default: 0.7)

    Returns:
        True if processed successfully, False if failed
    """
    company_name = row.get("Company") or ""

    if not company_name:
        row["Homepage URL"] = "Not Found"
        logger.warning(f"Row {index}: Empty company name")
        return True

    # Get PIC and ROO if available (optional fields)
    pic_name = row.get("PIC") or None
    roo_name = row.get("ROO") or None

    logger.info(
        f"Row {index}: Processing '{company_name}'"
        + (f" (PIC: {pic_name})" if pic_name else "")
        + (f" (ROO: {roo_name})" if roo_name else "")
    )

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
            urls = search_google_serper(
                search_query,
                num_results=max_urls_per_query,
                location=search_location,
                language=search_language,
                company_name=company_name,
            )

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
                if (
                    relevance.is_relevant
                    and relevance.confidence_score >= min_confidence
                ):
                    found_url = url
                    verified_relevance = relevance
                    website_text = text
                    logger.info(
                        f"Row {index}: ✓ Found relevant site for '{company_name}': {url}"
                    )
                    logger.info(
                        f"Row {index}:   Category: {relevance.relevance_category}, Confidence: {relevance.confidence_score:.2f}"
                    )
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
            row["Homepage URL"] = found_url
            row["Area"] = extracted_data.location
            row["Application"] = extracted_data.application_service
            row["Email"] = extracted_data.contact_email
            logger.info(
                f"Row {index}: ✓ Successfully extracted data for '{company_name}'"
            )
        else:
            # No relevant website found
            row["Homepage URL"] = "Not Found"
            reason = (
                verified_relevance.reason if verified_relevance else "No results found"
            )
            logger.warning(
                f"Row {index}: ✗ No relevant site for '{company_name}': {reason}"
            )

        return True

    except Exception as e:
        logger.error(f"Row {index}: Error processing '{company_name}': {str(e)}")
        row["Homepage URL"] = "Error"
        return False


def save_results(data: List[Dict[str, Any]], output_file: str) -> None:
    """
    Save processed data to Excel file.

    Args:
        data: List of company data dictionaries
        output_file: Path to output Excel file
    """
    try:
        pl.DataFrame(data).write_excel(output_file)
    except Exception as e:
        logging.error(f"Failed to save results to {output_file}: {e}")
        raise


def should_skip_row(row: Dict[str, Any], skip_filled: bool = True) -> bool:
    """
    Determine if a row should be skipped based on existing data.

    Args:
        row: Company data row
        skip_filled: Whether to skip rows with existing Homepage URL

    Returns:
        True if row should be skipped, False otherwise
    """
    if not skip_filled:
        return False

    homepage = row.get("Homepage URL")
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
        enable_cache: Enable in-memory caching (default: True)
        enable_disk_cache: Persist cache to disk across runs (default: False)
        cache_size: Maximum entries in memory cache (default: 1000)
    """
    # Initialize logging
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("Starting data extraction process")
    logger.info("=" * 60)

    # Initialize cache system
    if enable_cache:
        cache = init_cache(
            enable_disk_cache=enable_disk_cache, max_memory_size=cache_size
        )
        logger.info(
            f"Cache system initialized (disk_cache={'enabled' if enable_disk_cache else 'disabled'}, max_size={cache_size})"
        )
    else:
        logger.info("Cache system disabled")

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
                start_row = checkpoint["last_row_index"] + 1
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

        logger.info(
            f"Processing rows {actual_start} to {actual_end - 1} ({total_rows_to_process} rows)"
        )
        logger.info(f"Total rows in file: {total_data_rows}")
        logger.info(
            f"Continue mode: {'Enabled - preserving previous results' if continue_previous else 'Disabled - fresh start'}"
        )
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

        for local_index, row in tqdm(
            enumerate(rows_to_process), total=total_rows_to_process, desc="Processing"
        ):
            actual_index = actual_start + local_index

            # Skip if already filled (useful if script crashes and you restart)
            if should_skip_row(row, skip_filled):
                skipped_count += 1
                continue

            # Process the company
            success = process_company(
                row,
                actual_index,
                logger,
                search_location,
                search_language,
                max_queries,
                max_urls_per_query,
                min_confidence,
            )
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

                # Also save cache if disk caching is enabled
                if enable_cache and enable_disk_cache and cache:
                    cache.save_to_disk()

                logger.info(
                    f"Progress saved at row {actual_index} ({processed_count}/{total_rows_to_process} processed)"
                )

        # 4. Final Save
        save_results(data, output_file)
        logger.info("=" * 60)
        logger.info("Processing completed successfully!")
        logger.info(f"Total processed: {processed_count} companies")
        logger.info(
            f"Success: {success_count}, Errors: {error_count}, Skipped: {skipped_count}"
        )
        logger.info(f"Results saved to: {output_file}")
        logger.info("=" * 60)

        # Print cache statistics
        if enable_cache and cache:
            cache.get_stats().print_stats(logger)

            # Save cache to disk if enabled
            if enable_disk_cache:
                cache.save_to_disk()

        # Clear checkpoint on successful completion
        clear_checkpoint()

    except KeyboardInterrupt:
        logger.warning("\n" + "=" * 60)
        logger.warning("Process interrupted by user!")
        logger.warning(f"Progress saved to: {output_file}")
        logger.warning(f"Checkpoint saved. Use resume_from_checkpoint=True to continue")
        logger.warning("=" * 60)
        # Save current progress if possible
        try:
            save_results(data, output_file)
            if (
                "processed_count" in locals()
                and "total_rows_to_process" in locals()
                and "actual_index" in locals()
            ):
                save_checkpoint(processed_count, total_rows_to_process, actual_index)

            # Save cache if enabled
            if enable_cache and enable_disk_cache and "cache" in locals():
                cache.save_to_disk()
                logger.info("Cache saved to disk")
        except Exception as save_err:
            logger.error(f"Could not save progress on interrupt: {save_err}")

    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"Fatal error occurred: {str(e)}")
        logger.error("=" * 60)
        logger.exception(e)
        # Try to save what we have
        try:
            if "data" in locals():
                save_results(data, output_file)
            if (
                "processed_count" in locals()
                and "total_rows_to_process" in locals()
                and "actual_index" in locals()
            ):
                save_checkpoint(processed_count, total_rows_to_process, actual_index)

            # Save cache if enabled
            if enable_cache and enable_disk_cache and "cache" in locals():
                cache.save_to_disk()
                logger.info("Cache saved to disk")

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
    main(
        max_queries=3,
        max_urls_per_query=2,
        min_confidence=0.7,
        rate_limit=0.5,
        save_interval=10
    )

    # THOROUGH MODE (Best quality, slower)
    # Estimated: ~5-7 hours for 1800 rows
    # main(
    #     max_queries=5,
    #     max_urls_per_query=3,
    #     min_confidence=0.75,
    #     rate_limit=0.5,
    #     save_interval=10
    # )

    # ========== CACHING EXAMPLES ==========

    # Enable persistent disk cache (cache persists across runs - saves API calls!)
    # Recommended for multi-day batch processing
    # main(
    #     enable_cache=True,           # Enable caching (default: True)
    #     enable_disk_cache=True,      # Persist cache to disk (default: False)
    #     cache_size=2000              # Max entries in memory (default: 1000)
    # )

    # Disable caching (not recommended, but available)
    # main(enable_cache=False)

    # ========== COMMON USE CASES ==========

    # Test first 10 rows with FAST mode
    main(start_row=705, end_row=1763, max_queries=2, max_urls_per_query=1, rate_limit=0.2)

    # Process specific range (e.g., rows 100-200)
    # main(start_row=100, end_row=200)

    # Resume after interruption
    # main(resume_from_checkpoint=True)

    # Process all with Korean location filter and persistent cache
    # main(
    #     search_location='kr',
    #     search_language='ko',
    #     enable_disk_cache=True  # Cache persists across runs
    # )

    # ========== MULTI-DAY BATCH PROCESSING (1800 rows) ==========
    # Process in batches over multiple days - results accumulate automatically!
    # The script will load previous results and continue from where you left off.
    # IMPORTANT: Use enable_disk_cache=True to cache search/scrape results across days!

    # DAY 1: Process rows 0-300 (30-45 minutes)
    # main(start_row=0, end_row=300, max_queries=2, max_urls_per_query=1, enable_disk_cache=True)

    # DAY 2: Process rows 300-600 (30-45 minutes)
    # Previous results (0-299) are preserved automatically!
    # Cache from Day 1 is reused - duplicate companies processed instantly!
    # main(start_row=300, end_row=600, max_queries=2, max_urls_per_query=1, enable_disk_cache=True)

    # DAY 3: Process rows 600-900 (30-45 minutes)
    # main(start_row=600, end_row=900, max_queries=2, max_urls_per_query=1, enable_disk_cache=True)

    # DAY 4: Process rows 900-1200 (30-45 minutes)
    # main(start_row=900, end_row=1200, max_queries=2, max_urls_per_query=1, enable_disk_cache=True)

    # DAY 5: Process rows 1200-1500 (30-45 minutes)
    # main(start_row=1200, end_row=1500, max_queries=2, max_urls_per_query=1, enable_disk_cache=True)

    # DAY 6: Process rows 1500-1800 (30-45 minutes)
    # main(start_row=1500, end_row=1800, max_queries=2, max_urls_per_query=1, enable_disk_cache=True)

    # Result: All 1800 rows in ONE file after 6 days! 🎉
    # With disk cache enabled, duplicate companies are processed 10x faster!

    # ========== DISABLE CONTINUE MODE (Fresh Start) ==========
    # If you want to start fresh and ignore previous results:
    # main(continue_previous=False)
