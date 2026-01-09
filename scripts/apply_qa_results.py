#!/usr/bin/env python3
"""Apply QA validation results to pitch.db.

Reads qa_verified_*.csv and qa_conflicts_*.csv files and updates:
- confidence scores (+20 for verified, flag for conflicts)
- verified_sources (adds 'ojad' when verified)
- variation_note (adds note for words with multiple valid patterns)

Usage:
    # Apply most recent QA results
    python scripts/apply_qa_results.py

    # Apply specific QA session
    python scripts/apply_qa_results.py --session 20260108_141512

    # Dry-run (show what would change)
    python scripts/apply_qa_results.py --dry-run
"""

import argparse
import csv
import sqlite3
from pathlib import Path
from typing import List, Dict

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: Install tqdm for progress bars: pip install tqdm")

DB_PATH = Path(__file__).parent.parent / "data" / "pitch.db"
DATA_DIR = Path(__file__).parent.parent / "data"

# Batch commit size for database operations
BATCH_SIZE = 1000


def find_latest_qa_files() -> Dict[str, Path]:
    """Find most recent QA result files."""
    files = {}

    for pattern in ["qa_verified_*.csv", "qa_conflicts_*.csv", "qa_new_*.csv"]:
        matches = sorted(DATA_DIR.glob(pattern), reverse=True)
        if matches:
            key = pattern.split("_")[1].replace("*.csv", "")
            files[key] = matches[0]

    return files


def find_qa_files_by_session(session: str) -> Dict[str, Path]:
    """Find QA files for specific session timestamp."""
    files = {}

    for ftype in ["verified", "conflicts", "new"]:
        path = DATA_DIR / f"qa_{ftype}_{session}.csv"
        if path.exists():
            files[ftype] = path

    return files


def load_csv(path: Path) -> List[Dict]:
    """Load CSV file as list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def apply_verified(conn: sqlite3.Connection, entries: List[Dict], dry_run: bool, show_progress: bool = True) -> int:
    """Apply verified entries - increase confidence, add to verified_sources.

    Uses batch commits for performance on large datasets.
    """
    cursor = conn.cursor()
    updated = 0
    batch_count = 0

    # Create iterator with optional progress bar
    iterator = entries
    if show_progress and HAS_TQDM and not dry_run:
        iterator = tqdm(entries, desc="Verifying", unit="entry", ncols=70)

    for entry in iterator:
        surface = entry["surface"]
        reading = entry["reading"]

        if dry_run:
            print(f"  Would verify: {surface} ({reading})")
            updated += 1
            continue

        # Update confidence and verified_sources
        cursor.execute("""
            UPDATE pitch_accents
            SET confidence = MIN(100, confidence + 20),
                verified_sources = CASE
                    WHEN verified_sources = '' THEN 'ojad'
                    WHEN verified_sources NOT LIKE '%ojad%' THEN verified_sources || ',ojad'
                    ELSE verified_sources
                END
            WHERE surface = ? AND reading = ?
        """, (surface, reading))

        if cursor.rowcount > 0:
            updated += 1
            batch_count += 1

            # Batch commit for performance
            if batch_count >= BATCH_SIZE:
                conn.commit()
                batch_count = 0

    # Final commit for remaining entries
    if batch_count > 0 and not dry_run:
        conn.commit()

    return updated


def apply_conflicts(conn: sqlite3.Connection, entries: List[Dict], dry_run: bool, show_progress: bool = True) -> int:
    """Apply conflict entries - add OJAD patterns, set variation note.

    Uses batch commits for performance on large datasets.
    """
    cursor = conn.cursor()
    updated = 0
    batch_count = 0

    # Create iterator with optional progress bar
    iterator = entries
    if show_progress and HAS_TQDM and not dry_run:
        iterator = tqdm(entries, desc="Resolving conflicts", unit="entry", ncols=70)

    for entry in iterator:
        surface = entry["surface"]
        reading = entry["reading"]
        ojad_pattern = entry["ojad_pattern"]
        kanjium_pattern = entry["kanjium_pattern"]
        only_ojad = entry.get("only_ojad", "")

        if dry_run:
            print(f"  Would update: {surface} ({reading}): {kanjium_pattern} -> add {only_ojad}")
            updated += 1
            continue

        # Merge patterns (add OJAD patterns to existing)
        all_patterns = set(kanjium_pattern.split(","))
        all_patterns.update(ojad_pattern.split(","))
        merged = ",".join(sorted(all_patterns, key=int))

        # Create variation note
        if only_ojad:
            note = f"OJAD also lists: {only_ojad}"
        else:
            note = "Multiple patterns verified"

        cursor.execute("""
            UPDATE pitch_accents
            SET accent_pattern = ?,
                variation_note = ?,
                verified_sources = CASE
                    WHEN verified_sources = '' THEN 'ojad'
                    WHEN verified_sources NOT LIKE '%ojad%' THEN verified_sources || ',ojad'
                    ELSE verified_sources
                END
            WHERE surface = ? AND reading = ?
        """, (merged, note, surface, reading))

        if cursor.rowcount > 0:
            updated += 1
            batch_count += 1

            # Batch commit for performance
            if batch_count >= BATCH_SIZE:
                conn.commit()
                batch_count = 0

    # Final commit for remaining entries
    if batch_count > 0 and not dry_run:
        conn.commit()

    return updated


def apply_new_entries(conn: sqlite3.Connection, entries: List[Dict], dry_run: bool, show_progress: bool = True) -> int:
    """Add new entries from OJAD that don't exist in Kanjium.

    Uses batch commits for performance on large datasets.
    """
    cursor = conn.cursor()
    added = 0
    batch_count = 0

    # Create iterator with optional progress bar
    iterator = entries
    if show_progress and HAS_TQDM and not dry_run:
        iterator = tqdm(entries, desc="Adding new", unit="entry", ncols=70)

    for entry in iterator:
        surface = entry["surface"]
        reading = entry["reading"]
        pattern = entry["ojad_pattern"]

        if dry_run:
            print(f"  Would add: {surface} ({reading}) = {pattern}")
            added += 1
            continue

        # Check if exists (might have been added since QA ran)
        cursor.execute(
            "SELECT 1 FROM pitch_accents WHERE surface = ? AND reading = ?",
            (surface, reading)
        )
        if cursor.fetchone():
            continue

        cursor.execute("""
            INSERT INTO pitch_accents (surface, reading, accent_pattern, confidence, verified_sources, data_source)
            VALUES (?, ?, ?, 50, 'ojad', 'ojad')
        """, (surface, reading, pattern))

        if cursor.rowcount > 0:
            added += 1
            batch_count += 1

            # Batch commit for performance
            if batch_count >= BATCH_SIZE:
                conn.commit()
                batch_count = 0

    # Final commit for remaining entries
    if batch_count > 0 and not dry_run:
        conn.commit()

    return added


def show_confidence_distribution(conn: sqlite3.Connection):
    """Show current confidence distribution."""
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
        ORDER BY level DESC
    """)

    print("\nConfidence Distribution:")
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]:,}")


def main():
    parser = argparse.ArgumentParser(description="Apply QA results to pitch.db")
    parser.add_argument("--session", help="QA session timestamp (e.g., 20260108_141512)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without applying")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch commit size (default: 1000)")
    args = parser.parse_args()

    # Update batch size if specified
    global BATCH_SIZE
    BATCH_SIZE = args.batch_size
    show_progress = not args.no_progress

    print("=== Apply QA Results ===\n")

    # Find QA files
    if args.session:
        files = find_qa_files_by_session(args.session)
    else:
        files = find_latest_qa_files()

    if not files:
        print("No QA result files found.")
        print("Run 'python scripts/import_ojad.py --batch wordlists/common.txt' first.")
        return

    print("Found QA files:")
    for ftype, path in files.items():
        print(f"  {ftype}: {path.name}")

    if args.dry_run:
        print("\n[DRY-RUN MODE - No changes will be made]\n")

    # Connect to database
    conn = sqlite3.connect(DB_PATH)

    try:
        # Apply verified entries
        if "verified" in files:
            entries = load_csv(files["verified"])
            print(f"\nApplying {len(entries):,} verified entries...")
            updated = apply_verified(conn, entries, args.dry_run, show_progress)
            print(f"  Updated: {updated:,}")

        # Apply conflicts (merge patterns)
        if "conflicts" in files:
            entries = load_csv(files["conflicts"])
            print(f"\nApplying {len(entries):,} conflict resolutions...")
            updated = apply_conflicts(conn, entries, args.dry_run, show_progress)
            print(f"  Updated: {updated:,}")

        # Add new entries
        if "new" in files:
            entries = load_csv(files["new"])
            print(f"\nAdding {len(entries):,} new entries...")
            added = apply_new_entries(conn, entries, args.dry_run, show_progress)
            print(f"  Added: {added:,}")

        if not args.dry_run:
            print("\nAll changes committed!")

        show_confidence_distribution(conn)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
