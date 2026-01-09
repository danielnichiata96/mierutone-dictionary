#!/usr/bin/env python3
"""Add confidence tracking columns to pitch.db.

This enables the "voting" system for pitch accent confidence:
- confidence: 0-100 score based on source agreement
- verified_sources: comma-separated list of sources that verified this entry
- variation_note: optional note about regional/generational variation

Usage:
    python scripts/add_confidence_columns.py
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "pitch.db"


def add_columns():
    """Add confidence tracking columns if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Check existing columns
    cursor.execute("PRAGMA table_info(pitch_accents)")
    existing = {row[1] for row in cursor.fetchall()}

    columns_to_add = [
        ("confidence", "INTEGER DEFAULT 50"),
        ("verified_sources", "TEXT DEFAULT ''"),
        ("variation_note", "TEXT DEFAULT ''"),
    ]

    for col_name, col_def in columns_to_add:
        if col_name not in existing:
            print(f"Adding column: {col_name}")
            cursor.execute(f"ALTER TABLE pitch_accents ADD COLUMN {col_name} {col_def}")
        else:
            print(f"Column already exists: {col_name}")

    conn.commit()
    conn.close()
    print("\nDone!")


def show_confidence_distribution():
    """Show current confidence distribution."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            CASE
                WHEN confidence >= 80 THEN 'Gold (80-100)'
                WHEN confidence >= 50 THEN 'Blue (50-79)'
                ELSE 'Yellow (0-49)'
            END as level,
            COUNT(*) as count
        FROM pitch_accents
        GROUP BY level
        ORDER BY level
    """)

    print("\nConfidence Distribution:")
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]:,}")

    conn.close()


if __name__ == "__main__":
    print("=== Adding Confidence Columns ===\n")
    add_columns()
    show_confidence_distribution()
