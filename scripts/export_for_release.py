#!/usr/bin/env python3
"""Export pitch.db for public release.

This script:
1. Copies the database from mierutone-app
2. Applies community corrections from data/corrections.csv
3. Exports to CSV for non-SQLite users
4. Validates the data

Run from mierutone-dictionary root:
    python scripts/export_for_release.py --source ../pitch/backend/data/pitch.db
"""

import argparse
import csv
import shutil
import sqlite3
from pathlib import Path


def apply_corrections(conn: sqlite3.Connection, corrections_path: Path) -> int:
    """Apply community corrections from CSV."""
    if not corrections_path.exists():
        print(f"No corrections file at {corrections_path}")
        return 0

    applied = 0
    with open(corrections_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cursor = conn.execute(
                """
                UPDATE pitch_accents
                SET accent_pattern = ?, goshu = ?
                WHERE surface = ? AND reading = ?
                """,
                (
                    row["accent_pattern"],
                    row.get("goshu") or None,
                    row["surface"],
                    row["reading"],
                ),
            )
            if cursor.rowcount > 0:
                applied += 1
                print(f"  Applied: {row['surface']} ({row['reading']}) → type {row['accent_pattern']}")

    conn.commit()
    return applied


def export_to_csv(conn: sqlite3.Connection, output_path: Path) -> int:
    """Export database to CSV."""
    cursor = conn.execute(
        "SELECT surface, reading, accent_pattern, goshu, goshu_jp FROM pitch_accents ORDER BY surface"
    )

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["surface", "reading", "accent_pattern", "goshu", "goshu_jp"])

        count = 0
        for row in cursor:
            writer.writerow(row)
            count += 1

    return count


def validate_database(conn: sqlite3.Connection) -> list[str]:
    """Run validation checks."""
    issues = []

    # Check for NULL accent patterns
    cursor = conn.execute("SELECT COUNT(*) FROM pitch_accents WHERE accent_pattern IS NULL")
    null_count = cursor.fetchone()[0]
    if null_count > 0:
        issues.append(f"Warning: {null_count} entries with NULL accent_pattern")

    # Check for invalid accent patterns
    cursor = conn.execute(
        "SELECT DISTINCT accent_pattern FROM pitch_accents WHERE accent_pattern NOT GLOB '[0-9]*'"
    )
    invalid = [row[0] for row in cursor if row[0]]
    if invalid:
        issues.append(f"Warning: Non-numeric accent patterns: {invalid[:5]}")

    # Check entry count
    cursor = conn.execute("SELECT COUNT(*) FROM pitch_accents")
    total = cursor.fetchone()[0]
    if total < 100000:
        issues.append(f"Warning: Only {total} entries (expected 100k+)")

    return issues


def main():
    parser = argparse.ArgumentParser(description="Export pitch.db for public release")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("../pitch/backend/data/pitch.db"),
        help="Source database path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pitch.db"),
        help="Output database path",
    )
    args = parser.parse_args()

    if not args.source.exists():
        print(f"Error: Source database not found at {args.source}")
        return 1

    # Copy database
    print(f"Copying {args.source} → {args.output}")
    shutil.copy(args.source, args.output)

    # Connect and apply corrections
    conn = sqlite3.connect(args.output)

    corrections_path = Path("data/corrections.csv")
    print(f"\nApplying corrections from {corrections_path}...")
    applied = apply_corrections(conn, corrections_path)
    print(f"Applied {applied} corrections")

    # Export to CSV
    csv_path = Path("pitch_accents.csv")
    print(f"\nExporting to {csv_path}...")
    count = export_to_csv(conn, csv_path)
    print(f"Exported {count} entries")

    # Validate
    print("\nValidating...")
    issues = validate_database(conn)
    for issue in issues:
        print(f"  {issue}")

    if not issues:
        print("  All checks passed!")

    conn.close()

    print(f"\nDone! Files ready for release:")
    print(f"  - {args.output}")
    print(f"  - {csv_path}")

    return 0


if __name__ == "__main__":
    exit(main())
