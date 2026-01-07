"""Enrich pitch_accents table with goshu from UniDic.

Uses fugashi/UniDic to classify each word's etymology:
- wago: Native Japanese
- kango: Sino-Japanese (Chinese origin)
- gairaigo: Foreign loanwords
- proper: Proper nouns
- mixed: Mixed origin
- symbol: Symbols
- unknown: Unknown

Usage:
    python scripts/import_goshu.py
"""

from __future__ import annotations
import sys
import sqlite3
from typing import Optional, Tuple

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

from fugashi import Tagger
import unidic
from tqdm import tqdm

DB_PATH = Path(__file__).parent.parent / "data" / "pitch.db"

# Map UniDic goshu codes to readable labels
GOSHU_MAP = {
    "和": ("wago", "和語"),
    "漢": ("kango", "漢語"),
    "外": ("gairaigo", "外来語"),
    "固": ("proper", "固有名詞"),
    "混": ("mixed", "混種語"),
    "記号": ("symbol", "記号"),
    "不明": ("unknown", "不明"),
}


def add_goshu_columns(conn: sqlite3.Connection):
    """Add goshu columns to pitch_accents table if they don't exist."""
    cursor = conn.cursor()

    # Check if columns exist
    cursor.execute("PRAGMA table_info(pitch_accents)")
    columns = {row[1] for row in cursor.fetchall()}

    if "goshu" not in columns:
        print("Adding 'goshu' column...")
        cursor.execute("ALTER TABLE pitch_accents ADD COLUMN goshu TEXT")

    if "goshu_jp" not in columns:
        print("Adding 'goshu_jp' column...")
        cursor.execute("ALTER TABLE pitch_accents ADD COLUMN goshu_jp TEXT")

    conn.commit()


def get_goshu(tagger: Tagger, surface: str) -> Tuple[Optional[str], Optional[str]]:
    """Get goshu for a word using UniDic.

    Returns:
        Tuple of (goshu_en, goshu_jp) or (None, None) if not found.
    """
    try:
        tokens = list(tagger(surface))
        if not tokens:
            return None, None

        # Use first token's goshu (most words are single tokens)
        goshu_code = tokens[0].feature.goshu
        if goshu_code and goshu_code in GOSHU_MAP:
            return GOSHU_MAP[goshu_code]
        return None, None
    except Exception:
        return None, None


def enrich_with_goshu(conn: sqlite3.Connection, tagger: Tagger):
    """Add goshu data to all entries in pitch_accents."""
    cursor = conn.cursor()

    # Get all entries without goshu
    cursor.execute("""
        SELECT id, surface FROM pitch_accents
        WHERE goshu IS NULL
    """)
    entries = cursor.fetchall()

    if not entries:
        print("All entries already have goshu data.")
        return 0

    print(f"Enriching {len(entries):,} entries with goshu...")

    updated = 0
    batch = []
    batch_size = 1000

    for entry_id, surface in tqdm(entries, desc="Processing"):
        goshu_en, goshu_jp = get_goshu(tagger, surface)

        if goshu_en:
            batch.append((goshu_en, goshu_jp, entry_id))

        if len(batch) >= batch_size:
            cursor.executemany(
                "UPDATE pitch_accents SET goshu = ?, goshu_jp = ? WHERE id = ?",
                batch
            )
            conn.commit()
            updated += len(batch)
            batch = []

    # Final batch
    if batch:
        cursor.executemany(
            "UPDATE pitch_accents SET goshu = ?, goshu_jp = ? WHERE id = ?",
            batch
        )
        conn.commit()
        updated += len(batch)

    return updated


def print_stats(conn: sqlite3.Connection):
    """Print goshu distribution statistics."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT goshu, goshu_jp, COUNT(*) as count
        FROM pitch_accents
        WHERE goshu IS NOT NULL
        GROUP BY goshu
        ORDER BY count DESC
    """)

    print("\n=== Goshu Distribution ===")
    total = 0
    for goshu, goshu_jp, count in cursor.fetchall():
        print(f"  {goshu_jp or goshu}: {count:,}")
        total += count

    cursor.execute("SELECT COUNT(*) FROM pitch_accents WHERE goshu IS NULL")
    null_count = cursor.fetchone()[0]

    print(f"\n  Total with goshu: {total:,}")
    print(f"  Without goshu: {null_count:,}")


def main():
    print("=== Goshu (語種) Enrichment ===\n")

    if not DB_PATH.exists():
        print(f"Error: Database not found at {DB_PATH}")
        print("Run 'python scripts/import_kanjium.py' first.")
        return

    # Check if UniDic is downloaded
    if not Path(unidic.DICDIR).exists() or not any(Path(unidic.DICDIR).iterdir()):
        print("Error: UniDic dictionary not found.")
        print("Please download it first with:")
        print("  python -m unidic download")
        return

    # Initialize UniDic tagger
    print("Loading UniDic dictionary...")
    tagger = Tagger(f'-d "{unidic.DICDIR}"')

    # Connect to database
    conn = sqlite3.connect(DB_PATH)

    # Add columns if needed
    add_goshu_columns(conn)

    # Enrich entries
    updated = enrich_with_goshu(conn, tagger)
    print(f"\nUpdated {updated:,} entries with goshu data.")

    # Show statistics
    print_stats(conn)

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
