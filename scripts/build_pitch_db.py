"""Build complete pitch.db with all data sources.

Combines:
1. Kanjium accents.txt (primary source)
2. Goshu (word origin) data
3. Wikipedia frequency ranking
4. Ajatt-Tools supplementary data

Usage:
    python scripts/build_pitch_db.py [--skip-download]
"""

import argparse
import sqlite3
from pathlib import Path
import subprocess
import sys


SCRIPTS_DIR = Path(__file__).parent
DB_PATH = SCRIPTS_DIR.parent / "data" / "pitch.db"

# Scripts to run in order (with dependencies)
BUILD_STEPS = [
    ("import_kanjium.py", "Import base Kanjium pitch accent data", True),
    ("import_goshu.py", "Add word origin (goshu) data", True),
    ("import_frequency.py", "Add Wikipedia frequency ranking", False),  # Optional
    ("import_nhk.py", "Import supplementary Ajatt-Tools data", False),  # Optional
]


def run_script(script_name: str, description: str, required: bool) -> bool:
    """Run a script in the scripts directory.

    Returns True if successful, False if failed.
    """
    script_path = SCRIPTS_DIR / script_name

    if not script_path.exists():
        if required:
            print(f"ERROR: Required script not found: {script_path}")
            return False
        else:
            print(f"SKIP: Optional script not found: {script_path}")
            return True

    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"Script: {script_name}")
    print('='*60)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=SCRIPTS_DIR.parent,
    )

    if result.returncode != 0:
        if required:
            print(f"ERROR: {script_name} failed with code {result.returncode}")
            return False
        else:
            print(f"WARNING: Optional step failed, continuing...")
            return True

    return True


def show_stats(db_path: Path) -> None:
    """Show database statistics."""
    if not db_path.exists():
        print("Database not found!")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print(f"\n{'='*60}")
    print("Database Statistics")
    print('='*60)

    cursor.execute("SELECT COUNT(*) FROM pitch_accents")
    total = cursor.fetchone()[0]
    print(f"Total entries: {total:,}")

    # By goshu (word origin)
    cursor.execute("""
        SELECT goshu, COUNT(*) as cnt
        FROM pitch_accents
        GROUP BY goshu
        ORDER BY cnt DESC
    """)
    print("\nBy word origin (goshu):")
    for row in cursor.fetchall():
        goshu = row[0] or "NULL"
        print(f"  {goshu}: {row[1]:,}")

    # By data source (if column exists)
    cursor.execute("PRAGMA table_info(pitch_accents)")
    columns = [row[1] for row in cursor.fetchall()]
    if "data_source" in columns:
        cursor.execute("""
            SELECT data_source, COUNT(*) as cnt
            FROM pitch_accents
            GROUP BY data_source
            ORDER BY cnt DESC
        """)
        print("\nBy data source:")
        for row in cursor.fetchall():
            source = row[0] or "unknown"
            print(f"  {source}: {row[1]:,}")

    # Frequency coverage
    if "frequency_rank" in columns:
        cursor.execute("""
            SELECT COUNT(*) FROM pitch_accents
            WHERE frequency_rank IS NOT NULL
        """)
        with_freq = cursor.fetchone()[0]
        print(f"\nWith frequency data: {with_freq:,} ({100*with_freq/total:.1f}%)")

    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Build complete pitch.db")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip base import if pitch.db exists")
    args = parser.parse_args()

    print("=" * 60)
    print("Mierutone Pitch Database Builder")
    print("=" * 60)

    # Check if we should skip base import
    if args.skip_download and DB_PATH.exists():
        print(f"\nUsing existing database: {DB_PATH}")
        # Skip first step (import_kanjium.py)
        steps_to_run = BUILD_STEPS[1:]
    else:
        steps_to_run = BUILD_STEPS

    # Run each step
    for script_name, description, required in steps_to_run:
        if not run_script(script_name, description, required):
            print("\nBuild failed!")
            sys.exit(1)

    # Show final stats
    show_stats(DB_PATH)

    print(f"\n{'='*60}")
    print("Build complete!")
    print('='*60)
    if DB_PATH.exists():
        print(f"\nDatabase: {DB_PATH}")
        print(f"Size: {DB_PATH.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
