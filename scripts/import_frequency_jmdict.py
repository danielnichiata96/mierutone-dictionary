"""Import frequency data from JMdict into pitch.db.

Complements Wikipedia frequency with JMdict nfxx rankings.
Only matches entries where both surface AND reading match (precise).

Usage:
    python scripts/import_frequency_jmdict.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

PITCH_DB = Path(__file__).parent.parent / "data" / "pitch.db"
JMDICT_DB = Path(__file__).parent.parent / "data" / "jmdict.db"


def import_jmdict_frequency() -> int:
    """Import frequency from jmdict.db into pitch.db entries missing frequency."""
    if not PITCH_DB.exists():
        print(f"Error: pitch.db not found at {PITCH_DB}")
        return 0

    if not JMDICT_DB.exists():
        print(f"Error: jmdict.db not found at {JMDICT_DB}")
        return 0

    conn = sqlite3.connect(PITCH_DB)
    cursor = conn.cursor()

    # Attach jmdict.db
    cursor.execute(f"ATTACH DATABASE '{JMDICT_DB}' AS jm")

    # Count entries that will be updated
    cursor.execute("""
        SELECT COUNT(*) FROM pitch_accents p
        WHERE p.frequency_rank IS NULL
        AND EXISTS (
            SELECT 1 FROM jm.jmdict_entries j
            WHERE j.frequency_rank IS NOT NULL
            AND j.word = p.surface
            AND j.reading = p.reading
        )
    """)
    count = cursor.fetchone()[0]
    print(f"Found {count:,} entries to update from JMdict...")

    if count == 0:
        conn.close()
        return 0

    # Update frequency_rank from JMdict where both surface and reading match
    # JMdict frequency is 1-48 scale, Wikipedia is 1-50000+
    # Normalize JMdict to similar scale: multiply by 1000 and add offset
    # This puts JMdict frequencies after Wikipedia (lower = more common)
    cursor.execute("""
        UPDATE pitch_accents
        SET frequency_rank = (
            SELECT j.frequency_rank * 1000 + 50000
            FROM jm.jmdict_entries j
            WHERE j.word = pitch_accents.surface
            AND j.reading = pitch_accents.reading
            AND j.frequency_rank IS NOT NULL
            LIMIT 1
        )
        WHERE frequency_rank IS NULL
        AND EXISTS (
            SELECT 1 FROM jm.jmdict_entries j
            WHERE j.frequency_rank IS NOT NULL
            AND j.word = pitch_accents.surface
            AND j.reading = pitch_accents.reading
        )
    """)

    updated = cursor.rowcount
    conn.commit()

    # Detach and close
    cursor.execute("DETACH DATABASE jm")
    conn.close()

    return updated


def print_stats():
    """Print frequency coverage statistics."""
    conn = sqlite3.connect(PITCH_DB)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM pitch_accents")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM pitch_accents WHERE frequency_rank IS NOT NULL")
    with_freq = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM pitch_accents WHERE frequency_rank IS NOT NULL AND frequency_rank < 50000")
    wikipedia_freq = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM pitch_accents WHERE frequency_rank IS NOT NULL AND frequency_rank >= 50000")
    jmdict_freq = cursor.fetchone()[0]

    conn.close()

    pct = (with_freq / total) * 100 if total else 0
    print(f"\n=== Frequency Coverage ===")
    print(f"  Total entries:     {total:,}")
    print(f"  With frequency:    {with_freq:,} ({pct:.1f}%)")
    print(f"    - Wikipedia:     {wikipedia_freq:,}")
    print(f"    - JMdict:        {jmdict_freq:,}")
    print(f"  Without frequency: {total - with_freq:,}")


def main():
    print("=== JMdict Frequency Import ===\n")

    print("Before:")
    print_stats()

    updated = import_jmdict_frequency()
    print(f"\nUpdated {updated:,} entries with JMdict frequency.")

    print("\nAfter:")
    print_stats()

    print("\nDone!")


if __name__ == "__main__":
    main()
