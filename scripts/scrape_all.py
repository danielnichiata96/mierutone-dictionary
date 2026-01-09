#!/usr/bin/env python3
"""Production script to scrape all words from Kanjium database against OJAD.

Usage:
    # Run locally (will take ~28 hours for 50k words)
    python scrape_all.py

    # Run in background (recommended)
    nohup python -u scrape_all.py >> scrape.log 2>&1 &

    # Check progress
    python scrape_all.py --status

    # Resume after interruption (automatic - just run again)
    python scrape_all.py

    # Run with screen (can disconnect and reconnect)
    screen -S scraper -dm python scrape_all.py
    screen -r scraper  # to reconnect
"""

import argparse
import sqlite3
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ojad.scraper import (
    init_cache_db,
    scrape_batch,
    get_progress_stats,
    migrate_json_to_sqlite,
    setup_file_logging,
    get_logger,
    DEFAULT_CACHE_PATH,
)


# Paths
SCRIPTS_DIR = Path(__file__).parent
DATA_DIR = SCRIPTS_DIR.parent / "data"
KANJIUM_DB = DATA_DIR / "pitch.db"
OJAD_CACHE_DB = DATA_DIR / "ojad_cache.db"


def get_all_words_from_kanjium(max_frequency_rank: int = 0) -> list[str]:
    """Extract unique words from Kanjium database.

    Args:
        max_frequency_rank: If > 0, only return words with frequency_rank <= this value.
                           0 means all words.
    """
    if not KANJIUM_DB.exists():
        raise FileNotFoundError(f"Kanjium database not found: {KANJIUM_DB}")

    conn = sqlite3.connect(str(KANJIUM_DB))

    if max_frequency_rank > 0:
        # Get words filtered by frequency rank (most common first)
        rows = conn.execute("""
            SELECT DISTINCT surface FROM pitch_accents
            WHERE surface IS NOT NULL AND surface != ''
              AND frequency_rank IS NOT NULL
              AND frequency_rank > 0
              AND frequency_rank <= ?
            ORDER BY frequency_rank ASC
        """, (max_frequency_rank,)).fetchall()
    else:
        # Get all unique surface forms
        rows = conn.execute("""
            SELECT DISTINCT surface FROM pitch_accents
            WHERE surface IS NOT NULL AND surface != ''
            ORDER BY surface
        """).fetchall()

    conn.close()

    return [row[0] for row in rows]


def show_status():
    """Show current scraping progress."""
    logger = get_logger()

    if not OJAD_CACHE_DB.exists():
        print("No scraping started yet. Run without --status to begin.")
        return

    conn = init_cache_db(OJAD_CACHE_DB)
    stats = get_progress_stats(conn)

    # Get total words
    total_words = len(get_all_words_from_kanjium())

    # Calculate stats
    done = stats.get('success', 0) + stats.get('not_found', 0)
    pending = stats.get('pending', 0)
    errors = stats.get('error', 0)

    print(f"""
OJAD Scraping Status
{'=' * 40}
Total words in Kanjium:  {total_words:,}
Progress tracked:        {sum(stats.values()):,}

Status breakdown:
  Success:    {stats.get('success', 0):,}
  Not found:  {stats.get('not_found', 0):,}
  Pending:    {pending:,}
  Errors:     {errors:,}

Completion: {100 * done / total_words:.1f}% ({done:,} / {total_words:,})

Estimated time remaining: {pending * 2 / 3600:.1f} hours
(at 2 seconds per word)
""")

    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Scrape all Kanjium words against OJAD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="Show current progress without scraping"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=0,
        help="Limit number of words to scrape (0 = all)"
    )
    parser.add_argument(
        "--frequency", "-f",
        type=int,
        default=0,
        help="Only scrape words with frequency_rank <= this value (0 = all)"
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=int,
        default=100,
        help="Report progress every N words (default: 100)"
    )

    args = parser.parse_args()

    # Setup logging
    setup_file_logging()
    logger = get_logger()

    if args.status:
        show_status()
        return

    # Migrate legacy JSON cache if exists
    migrate_json_to_sqlite()

    # Get words (optionally filtered by frequency)
    logger.info("Loading words from Kanjium database...")
    words = get_all_words_from_kanjium(max_frequency_rank=args.frequency)

    if args.frequency > 0:
        logger.info(f"Found {len(words):,} words with frequency_rank <= {args.frequency}")
    else:
        logger.info(f"Found {len(words):,} unique words")

    if args.limit > 0:
        words = words[:args.limit]
        logger.info(f"Limited to {len(words):,} words")

    # Start scraping
    logger.info("Starting scrape (Ctrl+C to pause, run again to resume)...")

    stats = scrape_batch(
        words=words,
        db_path=OJAD_CACHE_DB,
        checkpoint_interval=args.checkpoint,
        quiet=False,
    )

    print(f"\nFinal Results:\n{stats.report()}")


if __name__ == "__main__":
    main()
