"""Import word frequency data from Kanjium into pitch.db.

Downloads wikipedia_freq.txt from Kanjium and adds frequency_rank column
to pitch_accents table. Designed to be idempotent - safe to run multiple times.

Usage:
    python scripts/import_frequency.py
"""

import sqlite3
import urllib.request
import urllib.error
import time
from pathlib import Path
from typing import Dict, Optional

# Kanjium frequency data (same commit as accents.txt for consistency)
KANJIUM_COMMIT = "685d4d723d6d20bf9beb169103aeac188eb067ad"
FREQ_URL = f"https://raw.githubusercontent.com/mifunetoshiro/kanjium/{KANJIUM_COMMIT}/data/source_files/raw/wikipedia_freq.txt"
DB_PATH = Path(__file__).parent.parent / "data" / "pitch.db"

REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 2


def download_frequency() -> Optional[str]:
    """Download wikipedia_freq.txt from Kanjium.

    Returns None on failure (graceful degradation).
    """
    print(f"Downloading frequency data from Kanjium...")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(FREQ_URL, timeout=REQUEST_TIMEOUT) as response:
                content = response.read().decode("utf-8")
            print(f"Downloaded {len(content):,} bytes")
            return content
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < MAX_RETRIES:
                print(f"Attempt {attempt} failed: {e}. Retrying...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"Warning: Failed to download frequency data after {MAX_RETRIES} attempts: {e}")
                print("Database will remain usable without frequency ranking.")
                return None


def parse_frequency(content: str) -> Dict[str, int]:
    """Parse frequency file into word -> rank mapping.

    Returns dict where rank 1 = most common word.
    """
    freq_map = {}
    rank = 1

    for line in content.strip().split("\n"):
        if not line or line.startswith("#"):
            continue

        parts = line.split("\t")
        if len(parts) >= 2:
            word = parts[0]
            # Store rank (1 = most common), not raw frequency
            if word not in freq_map:
                freq_map[word] = rank
                rank += 1

    return freq_map


def add_frequency_column(db_path: Path) -> bool:
    """Add frequency_rank column if not exists.

    Returns True if column was added or already exists.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check if column exists
        cursor.execute("PRAGMA table_info(pitch_accents)")
        columns = [row[1] for row in cursor.fetchall()]

        if "frequency_rank" not in columns:
            print("Adding frequency_rank column...")
            cursor.execute("ALTER TABLE pitch_accents ADD COLUMN frequency_rank INTEGER")
            conn.commit()
        else:
            print("frequency_rank column already exists")
        return True
    except sqlite3.Error as e:
        print(f"Error adding column: {e}")
        return False
    finally:
        conn.close()


def update_frequency(db_path: Path, freq_map: Dict[str, int]) -> int:
    """Update frequency_rank for all entries.

    Uses a transaction for atomicity - all or nothing.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Get all unique surfaces
        cursor.execute("SELECT DISTINCT surface FROM pitch_accents")
        surfaces = [row[0] for row in cursor.fetchall()]

        updated = 0
        for surface in surfaces:
            if surface in freq_map:
                cursor.execute(
                    "UPDATE pitch_accents SET frequency_rank = ? WHERE surface = ?",
                    (freq_map[surface], surface)
                )
                updated += cursor.rowcount

        conn.commit()
        return updated
    except sqlite3.Error as e:
        conn.rollback()
        print(f"Error updating frequency: {e}")
        return 0
    finally:
        conn.close()


def create_frequency_indexes(db_path: Path) -> None:
    """Create indexes for frequency-based lookups."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Simple index on frequency_rank
        cursor.execute("DROP INDEX IF EXISTS idx_frequency")
        cursor.execute("CREATE INDEX idx_frequency ON pitch_accents(frequency_rank)")

        # Composite index for homophone lookup optimization
        # This speeds up: WHERE reading = ? ORDER BY frequency_rank
        cursor.execute("DROP INDEX IF EXISTS idx_reading_frequency")
        cursor.execute(
            "CREATE INDEX idx_reading_frequency ON pitch_accents(reading, frequency_rank)"
        )

        conn.commit()
        print("Created frequency indexes (idx_frequency, idx_reading_frequency)")
    except sqlite3.Error as e:
        print(f"Warning: Could not create indexes: {e}")
    finally:
        conn.close()


def main():
    print("=== Kanjium Frequency Importer ===\n")

    if not DB_PATH.exists():
        print(f"Warning: pitch.db not found at {DB_PATH}")
        print("Run 'python scripts/import_kanjium.py' first.")
        return

    # Add column first (safe even without frequency data)
    if not add_frequency_column(DB_PATH):
        return

    # Download and parse frequency data
    content = download_frequency()
    if content is None:
        print("\nFrequency import skipped. Database remains usable.")
        return

    freq_map = parse_frequency(content)
    print(f"Parsed {len(freq_map):,} unique words with frequency data\n")

    # Update database
    updated = update_frequency(DB_PATH, freq_map)
    if updated > 0:
        create_frequency_indexes(DB_PATH)

    print(f"\nUpdated {updated:,} entries with frequency rank")
    print(f"Database: {DB_PATH}")


if __name__ == "__main__":
    main()
