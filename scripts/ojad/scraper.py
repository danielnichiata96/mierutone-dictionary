"""On-demand scraper for OJAD Suzuki-kun (Prosody Tutor).

Suzuki-kun provides more detailed prosody analysis than the basic OJAD search.
URL: http://www.gavo.t.u-tokyo.ac.jp/ojad/phrasing/index

This scraper supports both on-demand lookups and bulk scraping (50k+ words).
Features:
- SQLite cache for efficient storage and resume capability
- Retry with exponential backoff for resilience
- Checkpoint/resume for long-running jobs
- Structured logging for debugging
- Progress metrics and ETA tracking
"""

import json
import logging
import re
import sqlite3
import threading
import time
import urllib.parse
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .yomichan_parser import OJADEntry


# ============================================================================
# Configuration
# ============================================================================

SUZUKI_BASE = "http://www.gavo.t.u-tokyo.ac.jp/ojad"
SUZUKI_PHRASING_URL = f"{SUZUKI_BASE}/phrasing/index"
SUZUKI_SEARCH_URL = f"{SUZUKI_BASE}/search/index"

# Rate limiting
MIN_REQUEST_INTERVAL = 2.0  # seconds between requests
_last_request_time = 0.0
_rate_limit_lock = threading.Lock()

# Retry configuration
MAX_RETRIES = 3
BASE_RETRY_DELAY = 5.0  # seconds

# Paths
DEFAULT_CACHE_PATH = Path(__file__).parent.parent.parent / "data" / "ojad_cache.db"
DEFAULT_LOG_PATH = Path(__file__).parent.parent.parent / "data" / "ojad_scraper.log"

# Legacy JSON cache for backward compatibility
LEGACY_JSON_CACHE = Path(__file__).parent.parent.parent / "data" / "ojad_cache.json"


# ============================================================================
# Logging Setup
# ============================================================================

_logger: Optional[logging.Logger] = None


def get_logger() -> logging.Logger:
    """Get or create the scraper logger."""
    global _logger
    if _logger is None:
        _logger = logging.getLogger("ojad_scraper")
        _logger.setLevel(logging.INFO)

        # Only add handlers if none exist
        if not _logger.handlers:
            # Console handler (INFO level)
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            console.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)s %(message)s",
                datefmt="%H:%M:%S"
            ))
            _logger.addHandler(console)

    return _logger


def setup_file_logging(log_path: Path = DEFAULT_LOG_PATH) -> None:
    """Add file handler for persistent logging."""
    logger = get_logger()

    # Check if file handler already exists
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s [%(funcName)s] %(message)s"
    ))
    logger.addHandler(file_handler)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ScrapedEntry:
    """Entry scraped from OJAD."""
    surface: str
    reading: str
    patterns: List[int]
    pos: str
    source_url: str
    fetched_at: str


@dataclass
class ScraperStats:
    """Statistics for scraping progress."""
    total: int = 0
    success: int = 0
    not_found: int = 0
    errors: int = 0
    cache_hits: int = 0
    start_time: float = field(default_factory=time.time)

    def rate(self) -> float:
        """Words per second."""
        elapsed = time.time() - self.start_time
        return self.total / elapsed if elapsed > 0 else 0

    def eta_hours(self, remaining: int) -> float:
        """Estimated hours to complete remaining words."""
        rate = self.rate()
        return remaining / rate / 3600 if rate > 0 else float('inf')

    def report(self, remaining: int = 0) -> str:
        """Generate progress report."""
        elapsed = time.time() - self.start_time
        pct_success = 100 * self.success / self.total if self.total > 0 else 0

        lines = [
            f"Progress: {self.total:,} words processed",
            f"  Success: {self.success:,} ({pct_success:.1f}%)",
            f"  Not found: {self.not_found:,}",
            f"  Errors: {self.errors:,}",
            f"  Cache hits: {self.cache_hits:,}",
            f"  Rate: {self.rate():.2f} words/sec",
            f"  Elapsed: {elapsed/3600:.1f} hours",
        ]

        if remaining > 0:
            lines.append(f"  ETA: {self.eta_hours(remaining):.1f} hours ({remaining:,} remaining)")

        return "\n".join(lines)


# ============================================================================
# SQLite Cache Database
# ============================================================================

_db_lock = threading.Lock()


def init_cache_db(db_path: Path = DEFAULT_CACHE_PATH) -> sqlite3.Connection:
    """Initialize SQLite cache database.

    Args:
        db_path: Path to database file.

    Returns:
        Database connection.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row

    conn.executescript("""
        -- Cache table for scraped entries
        CREATE TABLE IF NOT EXISTS cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            surface TEXT NOT NULL,
            reading TEXT NOT NULL,
            patterns TEXT NOT NULL,  -- JSON array
            pos TEXT DEFAULT '',
            source_url TEXT DEFAULT '',
            fetched_at TEXT NOT NULL,
            UNIQUE(surface, reading)
        );

        CREATE INDEX IF NOT EXISTS idx_cache_surface ON cache(surface);
        CREATE INDEX IF NOT EXISTS idx_cache_reading ON cache(reading);

        -- Progress table for checkpoint/resume
        CREATE TABLE IF NOT EXISTS progress (
            word TEXT PRIMARY KEY,
            status TEXT NOT NULL,  -- 'pending', 'success', 'not_found', 'error'
            attempts INTEGER DEFAULT 0,
            last_error TEXT,
            updated_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_progress_status ON progress(status);
    """)

    conn.commit()
    return conn


def migrate_json_to_sqlite(
    json_path: Path = LEGACY_JSON_CACHE,
    db_path: Path = DEFAULT_CACHE_PATH
) -> int:
    """Migrate legacy JSON cache to SQLite.

    Args:
        json_path: Path to legacy JSON cache.
        db_path: Path to SQLite database.

    Returns:
        Number of entries migrated.
    """
    if not json_path.exists():
        return 0

    logger = get_logger()
    logger.info(f"Migrating JSON cache from {json_path}")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to read JSON cache: {e}")
        return 0

    conn = init_cache_db(db_path)
    migrated = 0

    for key, data in cache.items():
        try:
            conn.execute("""
                INSERT OR IGNORE INTO cache
                (surface, reading, patterns, pos, source_url, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                data.get("surface", ""),
                data.get("reading", ""),
                json.dumps(data.get("patterns", [])),
                data.get("pos", ""),
                data.get("source_url", ""),
                data.get("fetched_at", datetime.now().isoformat()),
            ))
            migrated += 1
        except Exception as e:
            logger.warning(f"Failed to migrate entry {key}: {e}")

    conn.commit()
    conn.close()

    # Rename old file to backup
    backup_path = json_path.with_suffix(".json.bak")
    json_path.rename(backup_path)
    logger.info(f"Migrated {migrated} entries, backup at {backup_path}")

    return migrated


# ============================================================================
# Retry Decorator
# ============================================================================

def retry_with_backoff(
    max_retries: int = MAX_RETRIES,
    base_delay: float = BASE_RETRY_DELAY,
    exceptions: Tuple = (urllib.error.URLError, TimeoutError),
) -> Callable:
    """Decorator for retry with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay in seconds (doubles each retry).
        exceptions: Tuple of exceptions to catch and retry.

    Returns:
        Decorated function.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = get_logger()
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {max_retries + 1} attempts failed for {func.__name__}"
                        )

            raise last_exception
        return wrapper
    return decorator


# ============================================================================
# Rate Limiting
# ============================================================================

def _rate_limit() -> None:
    """Enforce rate limiting between requests (thread-safe)."""
    global _last_request_time
    with _rate_limit_lock:
        elapsed = time.time() - _last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        _last_request_time = time.time()


# ============================================================================
# HTTP Request
# ============================================================================

@retry_with_backoff(max_retries=MAX_RETRIES, base_delay=BASE_RETRY_DELAY)
def _make_request(url: str) -> str:
    """Make HTTP request with retry and rate limiting.

    Args:
        url: URL to fetch.

    Returns:
        Response body as string.

    Raises:
        urllib.error.URLError: On network failure after retries.
        ValueError: On empty response.
    """
    _rate_limit()

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; mierutone-dictionary/1.0; research)",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "ja,en;q=0.9",
    }

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as response:
        html = response.read().decode("utf-8", errors="replace")
        if not html:
            raise ValueError("Empty response")
        return html


def _make_request_safe(url: str) -> Optional[str]:
    """Make HTTP request, returning None on failure.

    Args:
        url: URL to fetch.

    Returns:
        Response body as string, or None on failure.
    """
    try:
        return _make_request(url)
    except Exception as e:
        get_logger().error(f"Request failed after retries: {url} - {e}")
        return None


# ============================================================================
# HTML Parsing Helpers
# ============================================================================

def _is_hiragana(char: str) -> bool:
    """Check if character is hiragana."""
    return '\u3040' <= char <= '\u309f'


def _is_katakana(char: str) -> bool:
    """Check if character is katakana."""
    return '\u30a0' <= char <= '\u30ff'


def _is_kana(text: str) -> bool:
    """Check if text is all hiragana/katakana."""
    return all(_is_hiragana(c) or _is_katakana(c) or c in 'ー・' for c in text)


def _normalize_for_match(text: str) -> str:
    """Normalize text for matching (remove conjugation suffixes like ・ます)."""
    return text.split('・')[0].strip()


def _extract_word_rows(html: str) -> List[str]:
    """Extract word rows from OJAD HTML, handling nested tables.

    Args:
        html: Full HTML response.

    Returns:
        List of row HTML strings.
    """
    rows = []

    for match in re.finditer(r'<tr id="word_\d+">', html):
        start = match.start()
        pos = start
        depth = 0
        end_pos = None

        while pos < len(html):
            if html[pos:pos+3] == '<tr':
                depth += 1
            elif html[pos:pos+5] == '</tr>':
                depth -= 1
                if depth == 0:
                    end_pos = pos + 5
                    break
            pos += 1

        if end_pos:
            rows.append(html[start:end_pos])

    return rows


def _extract_accented_spans(html: str) -> List[str]:
    """Extract accented_word span contents using tag depth counting.

    Args:
        html: HTML to search in.

    Returns:
        List of inner HTML contents of accented_word spans.
    """
    spans = []

    for match in re.finditer(r'<span class="accented_word">', html):
        start = match.end()
        pos = match.start()
        depth = 0
        end_pos = None

        while pos < len(html):
            if html[pos:pos+5] == '<span':
                depth += 1
            elif html[pos:pos+7] == '</span>':
                depth -= 1
                if depth == 0:
                    end_pos = pos
                    break
            pos += 1

        if end_pos:
            spans.append(html[start:end_pos])

    return spans


# ============================================================================
# Main Scraping Functions
# ============================================================================

def scrape_word(word: str, quiet: bool = False) -> Optional[ScrapedEntry]:
    """Scrape pitch accent for a single word from OJAD.

    Args:
        word: Japanese word to look up (kanji or hiragana).
        quiet: If True, suppress status messages.

    Returns:
        ScrapedEntry if found, None otherwise.
    """
    logger = get_logger()
    encoded_word = urllib.parse.quote(word)
    search_url = f"{SUZUKI_SEARCH_URL}/word:{encoded_word}"

    if not quiet:
        logger.debug(f"Scraping: {word}")

    html = _make_request_safe(search_url)
    if not html:
        return None

    return _parse_search_result(html, word, search_url)


def _parse_search_result(html: str, word: str, url: str) -> Optional[ScrapedEntry]:
    """Parse OJAD search result HTML.

    Args:
        html: Raw HTML response.
        word: The word we searched for.
        url: Source URL for reference.

    Returns:
        ScrapedEntry if found and parsed, None otherwise.
    """
    rows = _extract_word_rows(html)
    if not rows:
        return None

    # Two-pass matching: first check midashi, then check readings
    matching_row = None
    matched_midashi = None

    # First pass: check midashi matches (higher priority)
    for row_html in rows:
        midashi_match = re.search(r'<p class="midashi_word">([^<]+)</p>', row_html)
        if not midashi_match:
            continue

        midashi = midashi_match.group(1).strip()
        midashi_base = _normalize_for_match(midashi)

        if midashi_base == word or midashi == word:
            matching_row = row_html
            matched_midashi = midashi_base
            break

        if _is_kana(word) and word in midashi:
            matching_row = row_html
            matched_midashi = midashi_base
            break

    # Second pass: for hiragana searches, check reading matches
    if not matching_row and _is_kana(word):
        for row_html in rows:
            midashi_match = re.search(r'<p class="midashi_word">([^<]+)</p>', row_html)
            midashi = midashi_match.group(1).strip() if midashi_match else ""
            midashi_base = _normalize_for_match(midashi)

            accented_spans = _extract_accented_spans(row_html)
            for span in accented_spans:
                chars = re.findall(r'<span class="char">(.)</span>', span)
                reading = ''.join(chars)
                if reading == word:
                    matching_row = row_html
                    matched_midashi = midashi_base
                    break
            if matching_row:
                break

    if not matching_row:
        return None

    # Extract accent patterns
    accented_matches = _extract_accented_spans(matching_row)
    if not accented_matches:
        return None

    patterns_found = set()
    base_reading = None

    for accented_html in accented_matches:
        chars = re.findall(r'<span class="char">(.)</span>', accented_html)
        reading = ''.join(chars)

        if not reading or not _is_kana(reading):
            continue

        if base_reading is None:
            base_reading = reading
        elif len(reading) != len(base_reading):
            continue

        mora_positions = re.findall(r'mola_(-\d+)', accented_html)
        mora_count = len(set(mora_positions))

        if mora_count == 0:
            continue

        accent_top_match = re.search(
            r'class="[^"]*accent_top[^"]*mola_(-\d+)"',
            accented_html
        )

        if accent_top_match:
            neg_pos = int(accent_top_match.group(1))
            accent = mora_count + neg_pos + 1
        else:
            accent = 0

        patterns_found.add(accent)

    if not patterns_found or not base_reading:
        return None

    pos_pattern = r'<td[^>]*class="[^"]*hinshi[^"]*"[^>]*>([^<]+)</td>'
    pos_match = re.search(pos_pattern, matching_row)
    pos = pos_match.group(1).strip() if pos_match else ""

    return ScrapedEntry(
        surface=word,
        reading=base_reading,
        patterns=sorted(patterns_found),
        pos=pos,
        source_url=url,
        fetched_at=datetime.now().isoformat(),
    )


# ============================================================================
# Cache Operations (SQLite)
# ============================================================================

def lookup_cached_db(
    conn: sqlite3.Connection,
    word: str,
    reading: Optional[str] = None,
) -> Optional[ScrapedEntry]:
    """Look up word in SQLite cache.

    Args:
        conn: Database connection.
        word: Surface form to look up.
        reading: Optional reading (if known).

    Returns:
        Cached ScrapedEntry if found, None otherwise.
    """
    with _db_lock:
        if reading:
            row = conn.execute(
                "SELECT * FROM cache WHERE surface = ? AND reading = ?",
                (word, reading)
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM cache WHERE surface = ?",
                (word,)
            ).fetchone()

    if row:
        return ScrapedEntry(
            surface=row["surface"],
            reading=row["reading"],
            patterns=json.loads(row["patterns"]),
            pos=row["pos"] or "",
            source_url=row["source_url"] or "",
            fetched_at=row["fetched_at"],
        )

    return None


def cache_entry_db(conn: sqlite3.Connection, entry: ScrapedEntry) -> None:
    """Add entry to SQLite cache.

    Args:
        conn: Database connection.
        entry: Entry to cache.
    """
    with _db_lock:
        conn.execute("""
            INSERT OR REPLACE INTO cache
            (surface, reading, patterns, pos, source_url, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            entry.surface,
            entry.reading,
            json.dumps(entry.patterns),
            entry.pos,
            entry.source_url,
            entry.fetched_at,
        ))
        conn.commit()


# ============================================================================
# Progress Tracking (Checkpoint/Resume)
# ============================================================================

def init_progress(conn: sqlite3.Connection, words: List[str]) -> int:
    """Initialize progress tracking for a word list.

    Args:
        conn: Database connection.
        words: List of words to process.

    Returns:
        Number of new words added (not already tracked).
    """
    now = datetime.now().isoformat()
    added = 0

    with _db_lock:
        for word in words:
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO progress (word, status, updated_at)
                    VALUES (?, 'pending', ?)
                """, (word, now))
                if conn.total_changes > 0:
                    added += 1
            except Exception:
                pass
        conn.commit()

    return added


def get_pending_words(
    conn: sqlite3.Connection,
    limit: int = 1000,
    max_attempts: int = 3,
) -> List[str]:
    """Get words that still need processing.

    Args:
        conn: Database connection.
        limit: Maximum number of words to return.
        max_attempts: Retry failed words up to this many attempts.

    Returns:
        List of words to process.
    """
    with _db_lock:
        rows = conn.execute("""
            SELECT word FROM progress
            WHERE status = 'pending'
               OR (status = 'error' AND attempts < ?)
            ORDER BY attempts ASC, word ASC
            LIMIT ?
        """, (max_attempts, limit)).fetchall()

    return [row[0] for row in rows]


def update_progress(
    conn: sqlite3.Connection,
    word: str,
    status: str,
    error: Optional[str] = None,
) -> None:
    """Update progress for a word.

    Args:
        conn: Database connection.
        word: Word to update.
        status: New status ('success', 'not_found', 'error').
        error: Error message if status is 'error'.
    """
    now = datetime.now().isoformat()

    with _db_lock:
        if status == 'error':
            conn.execute("""
                UPDATE progress
                SET status = ?, attempts = attempts + 1, last_error = ?, updated_at = ?
                WHERE word = ?
            """, (status, error, now, word))
        else:
            conn.execute("""
                UPDATE progress
                SET status = ?, updated_at = ?
                WHERE word = ?
            """, (status, now, word))
        conn.commit()


def get_progress_stats(conn: sqlite3.Connection) -> Dict[str, int]:
    """Get progress statistics.

    Args:
        conn: Database connection.

    Returns:
        Dict with counts by status.
    """
    with _db_lock:
        rows = conn.execute("""
            SELECT status, COUNT(*) as count FROM progress GROUP BY status
        """).fetchall()

    return {row[0]: row[1] for row in rows}


# ============================================================================
# High-Level Scraping Functions
# ============================================================================

def scrape_with_cache(
    word: str,
    conn: Optional[sqlite3.Connection] = None,
    force: bool = False,
    quiet: bool = False,
) -> Optional[ScrapedEntry]:
    """Scrape word with caching.

    Args:
        word: Word to look up.
        conn: Database connection (creates one if None).
        force: If True, ignore cache and fetch fresh.
        quiet: If True, suppress messages.

    Returns:
        ScrapedEntry if found, None otherwise.
    """
    logger = get_logger()
    own_conn = conn is None

    if own_conn:
        conn = init_cache_db()

    try:
        # Check cache first
        if not force:
            cached = lookup_cached_db(conn, word)
            if cached:
                if not quiet:
                    logger.debug(f"Cache hit: {word}")
                return cached

        # Scrape from OJAD
        entry = scrape_word(word, quiet=quiet)
        if entry:
            cache_entry_db(conn, entry)

        return entry

    finally:
        if own_conn:
            conn.close()


def scrape_batch(
    words: List[str],
    db_path: Path = DEFAULT_CACHE_PATH,
    checkpoint_interval: int = 100,
    quiet: bool = False,
) -> ScraperStats:
    """Scrape a batch of words with checkpoint/resume.

    Args:
        words: List of words to scrape.
        db_path: Path to database file.
        checkpoint_interval: Commit every N words.
        quiet: If True, suppress progress output.

    Returns:
        ScraperStats with results.
    """
    logger = get_logger()
    setup_file_logging()

    conn = init_cache_db(db_path)
    stats = ScraperStats()

    # Initialize progress tracking
    init_progress(conn, words)
    total_pending = len(get_pending_words(conn, limit=999999))

    logger.info(f"Starting batch scrape: {total_pending} words pending")

    try:
        while True:
            pending = get_pending_words(conn, limit=checkpoint_interval)
            if not pending:
                break

            for word in pending:
                try:
                    # Check cache first
                    cached = lookup_cached_db(conn, word)
                    if cached:
                        update_progress(conn, word, 'success')
                        stats.cache_hits += 1
                        stats.success += 1
                    else:
                        # Scrape from OJAD
                        entry = scrape_word(word, quiet=True)

                        if entry:
                            cache_entry_db(conn, entry)
                            update_progress(conn, word, 'success')
                            stats.success += 1
                        else:
                            update_progress(conn, word, 'not_found')
                            stats.not_found += 1

                except Exception as e:
                    update_progress(conn, word, 'error', str(e))
                    stats.errors += 1
                    logger.error(f"Error scraping {word}: {e}")

                stats.total += 1

            # Checkpoint: commit and report
            conn.commit()
            remaining = total_pending - stats.total

            if not quiet:
                logger.info(stats.report(remaining))

    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Progress saved.")
        conn.commit()

    finally:
        conn.close()

    logger.info(f"Batch complete.\n{stats.report()}")
    return stats


# ============================================================================
# Legacy Compatibility (JSON Cache)
# ============================================================================

# Keep these for backward compatibility with existing code

def load_cache(cache_path: Path = LEGACY_JSON_CACHE) -> Dict:
    """Load scrape cache from JSON file (legacy, use SQLite instead)."""
    if not cache_path.exists():
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_cache(cache: Dict, cache_path: Path = LEGACY_JSON_CACHE) -> None:
    """Save scrape cache to JSON file (legacy, use SQLite instead)."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def get_cache_key(surface: str, reading: str) -> str:
    """Generate cache key for a word."""
    return f"{surface}_{reading}"


def lookup_cached(
    word: str,
    reading: Optional[str] = None,
    cache_path: Path = LEGACY_JSON_CACHE,
) -> Optional[ScrapedEntry]:
    """Look up word in JSON cache (legacy)."""
    cache = load_cache(cache_path)

    if reading:
        key = get_cache_key(word, reading)
        if key in cache:
            data = cache[key]
            return ScrapedEntry(
                surface=data["surface"],
                reading=data["reading"],
                patterns=data["patterns"],
                pos=data.get("pos", ""),
                source_url=data.get("source_url", ""),
                fetched_at=data.get("fetched_at", ""),
            )

    for key, data in cache.items():
        if data.get("surface") == word:
            return ScrapedEntry(
                surface=data["surface"],
                reading=data["reading"],
                patterns=data["patterns"],
                pos=data.get("pos", ""),
                source_url=data.get("source_url", ""),
                fetched_at=data.get("fetched_at", ""),
            )

    return None


def cache_entry(entry: ScrapedEntry, cache_path: Path = LEGACY_JSON_CACHE) -> None:
    """Add entry to JSON cache (legacy)."""
    cache = load_cache(cache_path)
    key = get_cache_key(entry.surface, entry.reading)
    cache[key] = {
        "surface": entry.surface,
        "reading": entry.reading,
        "patterns": entry.patterns,
        "pos": entry.pos,
        "source_url": entry.source_url,
        "fetched_at": entry.fetched_at,
    }
    save_cache(cache, cache_path)


# ============================================================================
# Utility Functions
# ============================================================================

def convert_to_ojad_entry(scraped: ScrapedEntry) -> OJADEntry:
    """Convert ScrapedEntry to OJADEntry for comparison."""
    return OJADEntry(
        surface=scraped.surface,
        reading=scraped.reading,
        patterns=scraped.patterns,
        pos=scraped.pos,
    )


def scrape_phrase(phrase: str) -> List[ScrapedEntry]:
    """Scrape pitch accent for a phrase using Suzuki-kun.

    Note: This function may not work as Suzuki-kun uses JavaScript rendering.

    Args:
        phrase: Japanese phrase or sentence.

    Returns:
        List of ScrapedEntry for each word in the phrase.
    """
    logger = get_logger()
    encoded_phrase = urllib.parse.quote(phrase)
    phrasing_url = f"{SUZUKI_PHRASING_URL}?inputtext={encoded_phrase}"

    logger.info(f"Analyzing phrase: {phrase}")

    html = _make_request_safe(phrasing_url)
    if not html:
        return []

    return _parse_phrasing_result(html, phrasing_url)


def _parse_phrasing_result(html: str, url: str) -> List[ScrapedEntry]:
    """Parse Suzuki-kun phrasing result."""
    entries = []

    word_pattern = r'<div[^>]*class="[^"]*phrasing_word[^"]*"[^>]*>.*?</div>'
    words = re.findall(word_pattern, html, re.DOTALL)

    for word_html in words:
        surface_match = re.search(
            r'<span[^>]*class="[^"]*surface[^"]*"[^>]*>([^<]+)</span>',
            word_html
        )
        reading_match = re.search(
            r'<span[^>]*class="[^"]*reading[^"]*"[^>]*>([^<]+)</span>',
            word_html
        )

        if not surface_match:
            continue

        surface = surface_match.group(1).strip()
        reading = reading_match.group(1).strip() if reading_match else surface

        accent_match = re.search(r'data-accent="(\d+)"', word_html)
        if not accent_match:
            continue

        accent = int(accent_match.group(1))

        entries.append(ScrapedEntry(
            surface=surface,
            reading=reading,
            patterns=[accent],
            pos="",
            source_url=url,
            fetched_at=datetime.now().isoformat(),
        ))

    return entries
