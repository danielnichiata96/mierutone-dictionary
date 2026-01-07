"""Import Kanjium accents.txt into SQLite database.

Downloads and imports pitch accent data from the Kanjium project.
License: CC BY-SA 4.0 - https://github.com/mifunetoshiro/kanjium

Usage:
    python scripts/import_kanjium.py
"""

import sqlite3
import urllib.request
import urllib.error
import time
from pathlib import Path

# Pinned to specific commit for reproducible builds (2024-06-16)
KANJIUM_COMMIT = "685d4d723d6d20bf9beb169103aeac188eb067ad"
KANJIUM_URL = f"https://raw.githubusercontent.com/mifunetoshiro/kanjium/{KANJIUM_COMMIT}/data/source_files/raw/accents.txt"
DB_PATH = Path(__file__).parent.parent / "data" / "pitch.db"

# Network settings
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


def download_kanjium() -> str:
    """Download accents.txt from Kanjium repository with retry logic."""
    print(f"Downloading from Kanjium (commit {KANJIUM_COMMIT[:8]})...")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(KANJIUM_URL, timeout=REQUEST_TIMEOUT) as response:
                content = response.read().decode("utf-8")
            print(f"Downloaded {len(content):,} bytes")
            return content
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < MAX_RETRIES:
                print(f"Attempt {attempt} failed: {e}. Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                raise RuntimeError(f"Failed to download after {MAX_RETRIES} attempts: {e}")


def create_database(db_path: Path) -> sqlite3.Connection:
    """Create SQLite database with pitch_accents table."""
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS pitch_accents")
    cursor.execute("""
        CREATE TABLE pitch_accents (
            id INTEGER PRIMARY KEY,
            surface TEXT NOT NULL,
            reading TEXT NOT NULL,
            accent_pattern TEXT NOT NULL,
            goshu TEXT,
            goshu_jp TEXT
        )
    """)
    cursor.execute("CREATE INDEX idx_surface ON pitch_accents(surface)")
    cursor.execute("CREATE INDEX idx_reading ON pitch_accents(reading)")
    cursor.execute("CREATE INDEX idx_surface_reading ON pitch_accents(surface, reading)")

    conn.commit()
    return conn


def import_accents(conn: sqlite3.Connection, content: str) -> int:
    """Import accent data into database."""
    cursor = conn.cursor()
    count = 0

    for line in content.strip().split("\n"):
        if not line or line.startswith("#"):
            continue

        parts = line.split("\t")
        if len(parts) >= 3:
            surface, reading, pattern = parts[0], parts[1], parts[2]
            cursor.execute(
                "INSERT INTO pitch_accents (surface, reading, accent_pattern) VALUES (?, ?, ?)",
                (surface, reading, pattern)
            )
            count += 1

    conn.commit()
    return count


def main():
    print("=== Kanjium Pitch Accent Importer ===\n")

    content = download_kanjium()
    conn = create_database(DB_PATH)
    count = import_accents(conn, content)

    conn.close()

    print(f"\nImported {count:,} entries to {DB_PATH}")
    print("\nAttribution (CC BY-SA 4.0):")
    print("  Pitch accent data from Kanjium by mifunetoshiro")
    print("  https://github.com/mifunetoshiro/kanjium")


if __name__ == "__main__":
    main()
