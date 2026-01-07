"""Import NHK pitch accent data from Ajatt-Tools.

Downloads kanjium_data.tsv from Ajatt-Tools/PitchAccent and merges
with existing pitch.db, adding entries not already present.

Uses 'data_source' column to track origin (NOT goshu, which is word origin).

Usage:
    python scripts/import_nhk.py
"""

import sqlite3
import urllib.request
import urllib.error
import time
from pathlib import Path
from typing import Optional, List, Set, Tuple

# Ajatt-Tools data (archived repo, stable as of 2023-08-17)
AJATT_BASE = "https://raw.githubusercontent.com/Ajatt-Tools/PitchAccent/main/database/accent_dict"
KANJIUM_TSV_URL = f"{AJATT_BASE}/kanjium_data.tsv"

DB_PATH = Path(__file__).parent.parent / "data" / "pitch.db"

REQUEST_TIMEOUT = 60
MAX_RETRIES = 3
RETRY_DELAY = 2


def verify_url_accessible(url: str) -> bool:
    """Check if URL is accessible before attempting download."""
    try:
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status == 200
    except Exception:
        return False


def download_file(url: str, name: str) -> Optional[str]:
    """Download file with retry logic. Returns None on failure."""
    print(f"Downloading {name}...")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT) as response:
                content = response.read().decode("utf-8")
            print(f"Downloaded {len(content):,} bytes")
            return content
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < MAX_RETRIES:
                print(f"Attempt {attempt} failed: {e}. Retrying...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"Failed after {MAX_RETRIES} attempts: {e}")
                return None


def ensure_data_source_column(db_path: Path) -> bool:
    """Add data_source column if not exists."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute("PRAGMA table_info(pitch_accents)")
        columns = [row[1] for row in cursor.fetchall()]

        if "data_source" not in columns:
            print("Adding data_source column...")
            cursor.execute(
                "ALTER TABLE pitch_accents ADD COLUMN data_source TEXT DEFAULT 'kanjium'"
            )
            # Mark existing entries as from original kanjium
            cursor.execute(
                "UPDATE pitch_accents SET data_source = 'kanjium' WHERE data_source IS NULL"
            )
            conn.commit()
        return True
    except sqlite3.Error as e:
        print(f"Error adding column: {e}")
        return False
    finally:
        conn.close()


def get_existing_entries(db_path: Path) -> Set[Tuple[str, str]]:
    """Get set of (surface, reading) pairs already in database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT surface, reading FROM pitch_accents")
    existing = {(row[0], row[1]) for row in cursor.fetchall()}
    conn.close()
    return existing


def parse_kanjium_tsv(content: str) -> List[Tuple[str, str, str]]:
    """Parse Ajatt-Tools kanjium_data.tsv format.

    Format: surface\treading\taccent_pattern(s)
    Returns: List of (surface, reading, accent_pattern)
    """
    entries = []
    for line in content.strip().split("\n"):
        if not line or line.startswith("#"):
            continue

        parts = line.split("\t")
        if len(parts) >= 3:
            surface, reading, pattern = parts[0], parts[1], parts[2]
            entries.append((surface, reading, pattern))

    return entries


def import_new_entries(
    db_path: Path,
    entries: List[Tuple[str, str, str]],
    existing: Set[Tuple[str, str]],
    data_source: str
) -> int:
    """Import entries not already in database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    new_count = 0
    try:
        for surface, reading, pattern in entries:
            if (surface, reading) not in existing:
                cursor.execute(
                    """INSERT INTO pitch_accents (surface, reading, accent_pattern, data_source)
                       VALUES (?, ?, ?, ?)""",
                    (surface, reading, pattern, data_source)
                )
                new_count += 1
                existing.add((surface, reading))  # Prevent duplicates within import

        conn.commit()
    except sqlite3.Error as e:
        conn.rollback()
        print(f"Error importing entries: {e}")
        return 0
    finally:
        conn.close()

    return new_count


def main():
    print("=== NHK/Ajatt-Tools Pitch Accent Importer ===\n")

    if not DB_PATH.exists():
        print(f"Warning: pitch.db not found at {DB_PATH}")
        print("Run 'python scripts/import_kanjium.py' first.")
        return

    # Verify URL is accessible
    print("Verifying Ajatt-Tools repository accessibility...")
    if not verify_url_accessible(KANJIUM_TSV_URL):
        print("Warning: Ajatt-Tools URL not accessible.")
        print("The repository may have moved or been archived.")
        print("Skipping NHK import. Database remains usable.")
        return

    # Ensure data_source column exists
    if not ensure_data_source_column(DB_PATH):
        return

    # Get existing entries to avoid duplicates
    existing = get_existing_entries(DB_PATH)
    print(f"Existing entries in database: {len(existing):,}\n")

    # Import Kanjium TSV from Ajatt-Tools
    kanjium_content = download_file(KANJIUM_TSV_URL, "kanjium_data.tsv")
    if kanjium_content is None:
        print("Skipping NHK import due to download failure.")
        return

    kanjium_entries = parse_kanjium_tsv(kanjium_content)
    print(f"Parsed {len(kanjium_entries):,} entries from kanjium_data.tsv")

    new_kanjium = import_new_entries(DB_PATH, kanjium_entries, existing, "ajatt")
    print(f"Added {new_kanjium:,} new entries from Ajatt-Tools\n")

    # Final count
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM pitch_accents")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT data_source, COUNT(*) FROM pitch_accents GROUP BY data_source")
    print("Entries by source:")
    for row in cursor.fetchall():
        print(f"  {row[0] or 'unknown'}: {row[1]:,}")

    conn.close()

    print(f"\nTotal entries in database: {total:,}")
    print(f"Database: {DB_PATH}")


if __name__ == "__main__":
    main()
