#!/usr/bin/env python3
"""Import and validate OJAD pitch accent data against Kanjium.

This script:
1. Batch validates words from a list file against OJAD
2. Compares against existing pitch.db (Kanjium)
3. Generates validation reports (CSV)
4. Optionally applies changes to database

Usage:
    # Batch validate from word list (QA workflow)
    python scripts/import_ojad.py --batch wordlists/common.txt

    # Batch validate with fresh scrapes (ignore cache)
    python scripts/import_ojad.py --batch wordlists/jlpt_n5.txt --force-scrape

    # Scrape specific word from OJAD
    python scripts/import_ojad.py --scrape 食べる

    # Force fresh scrape (ignore cache)
    python scripts/import_ojad.py --scrape 東京 --force-scrape

Sources:
    - OJAD Suzuki-kun: http://www.gavo.t.u-tokyo.ac.jp/ojad/search/index
"""

import argparse
import asyncio
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: Install tqdm for progress bars: pip install tqdm")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from ojad.yomichan_parser import parse_yomichan_ojad, load_or_download_ojad, OJADEntry
from ojad.comparator import (
    load_kanjium_entries,
    compare_entries,
    apply_changes,
    ComparisonResult,
    ComparisonStatus,
)
from ojad.reporter import (
    write_validation_report,
    write_conflicts,
    write_ojad_only,
    print_summary,
    generate_markdown_report,
)
from ojad.scraper import (
    scrape_with_cache,
    convert_to_ojad_entry,
    DEFAULT_CACHE_PATH,
)


# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
DB_PATH = DATA_DIR / "pitch.db"
CACHE_DIR = DATA_DIR / "ojad_cache"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate OJAD pitch accent data against Kanjium",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Batch validate from word list (QA workflow)
    python scripts/import_ojad.py --batch wordlists/common.txt

    # Batch validate with fresh scrapes
    python scripts/import_ojad.py --batch wordlists/jlpt_n5.txt --force-scrape

    # Scrape a specific word
    python scripts/import_ojad.py --scrape 東京

    # Force fresh scrape (ignore cache)
    python scripts/import_ojad.py --scrape 食べる --force-scrape

    # Apply verified matches to database
    python scripts/import_ojad.py --batch wordlists/common.txt --apply
        """,
    )

    parser.add_argument(
        "--batch",
        type=Path,
        default=None,
        metavar="FILE",
        help="Word list file for batch validation (one word per line)",
    )

    parser.add_argument(
        "--scrape",
        type=str,
        default=None,
        metavar="WORD",
        help="Scrape a specific word from OJAD",
    )

    parser.add_argument(
        "--force-scrape",
        action="store_true",
        help="Ignore cache and fetch fresh from OJAD",
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes to pitch.db (default: dry-run, report only)",
    )

    parser.add_argument(
        "--min-confidence",
        type=int,
        default=None,
        metavar="N",
        help="Only apply changes with confidence_change >= N (use with --apply)",
    )

    parser.add_argument(
        "--db",
        type=Path,
        default=DB_PATH,
        help=f"Path to pitch.db (default: {DB_PATH})",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR,
        help=f"Output directory for reports (default: {DATA_DIR})",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Limit batch processing to first N words (for testing)",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        metavar="N",
        help="Number of concurrent workers for scraping (default: 3, max recommended: 5)",
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar (useful for logging to files)",
    )

    return parser.parse_args()


def run_scrape_mode(args: argparse.Namespace) -> int:
    """Run single-word scrape mode.

    Args:
        args: Parsed command line arguments.

    Returns:
        Exit code (0 = success).
    """
    word = args.scrape
    print(f"\n=== OJAD On-Demand Scrape ===\n")
    print(f"Word: {word}")
    print(f"Force: {args.force_scrape}")
    print()

    # Scrape the word
    entry = scrape_with_cache(
        word,
        force=args.force_scrape,
        cache_path=DEFAULT_CACHE_PATH,
    )

    if not entry:
        print(f"Could not find pitch accent for: {word}")
        return 1

    print(f"\nResult:")
    print(f"  Surface: {entry.surface}")
    print(f"  Reading: {entry.reading}")
    print(f"  Pattern: {','.join(map(str, entry.patterns))}")
    print(f"  POS: {entry.pos or '(unknown)'}")
    print(f"  Source: {entry.source_url}")

    # Compare with Kanjium
    if args.db.exists():
        print(f"\nComparing with Kanjium...")
        kanjium_data = load_kanjium_entries(args.db)
        key = (entry.surface, entry.reading)

        if key in kanjium_data:
            kanjium_patterns = kanjium_data[key]
            kanjium_str = ",".join(map(str, kanjium_patterns))
            ojad_str = ",".join(map(str, entry.patterns))

            if set(kanjium_patterns) == set(entry.patterns):
                print(f"  ✓ VERIFIED: Kanjium ({kanjium_str}) matches OJAD ({ojad_str})")
            else:
                print(f"  ⚠ CONFLICT: Kanjium ({kanjium_str}) vs OJAD ({ojad_str})")
        else:
            print(f"  + NEW: Not in Kanjium database")

    return 0


def run_batch_mode(args: argparse.Namespace) -> int:
    """Run batch validation mode from word list.

    Scrapes words from OJAD, compares against Kanjium, and generates
    CSV reports for manual review.

    Args:
        args: Parsed command line arguments.

    Returns:
        Exit code (0 = success).
    """
    word_list_path = args.batch

    print("\n" + "=" * 60)
    print("OJAD Batch Validation (QA Workflow)")
    print("=" * 60)

    # Check files exist
    if not word_list_path.exists():
        print(f"\nError: Word list not found: {word_list_path}")
        return 1

    if not args.db.exists():
        print(f"\nError: Database not found: {args.db}")
        print("Run 'python scripts/build_pitch_db.py' first.")
        return 1

    # Load word list
    print(f"\nLoading word list: {word_list_path}")
    words = []
    with open(word_list_path, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip()
            # Skip empty lines and comments
            if word and not word.startswith("#"):
                words.append(word)

    if args.limit:
        words = words[:args.limit]
        print(f"Limited to first {args.limit} words")

    print(f"Loaded: {len(words):,} words")

    # Load Kanjium data
    print(f"\nLoading Kanjium database: {args.db}")
    kanjium_data = load_kanjium_entries(args.db)
    print(f"Loaded: {len(kanjium_data):,} entries")

    # Process each word with parallel scraping
    print(f"\nValidating words against OJAD...")
    print(f"Workers: {args.workers} | Rate limit: 2s between requests per worker")
    print("-" * 60)

    results = {
        "verified": [],
        "conflicts": [],
        "new": [],
        "not_found": [],
        "errors": [],
    }

    start_time = time.time()

    # Determine if we should suppress scraper output (when using progress bar)
    use_progress = HAS_TQDM and not args.no_progress
    quiet_scraper = use_progress  # Suppress scraper messages when using progress bar

    def process_word(word: str) -> Dict[str, Any]:
        """Process a single word - scrape and compare."""
        try:
            # Scrape from OJAD
            entry = scrape_with_cache(
                word,
                force=args.force_scrape,
                cache_path=DEFAULT_CACHE_PATH,
                quiet=quiet_scraper,
            )

            if not entry:
                return {
                    "type": "not_found",
                    "data": {"word": word, "reason": "Not found in OJAD"},
                    "display": f"{word}: NOT FOUND",
                }

            # Look up in Kanjium
            key = (entry.surface, entry.reading)
            kanjium_patterns = kanjium_data.get(key)
            ojad_str = ",".join(map(str, entry.patterns))

            if kanjium_patterns is None:
                return {
                    "type": "new",
                    "data": {
                        "surface": entry.surface,
                        "reading": entry.reading,
                        "ojad_pattern": ojad_str,
                        "kanjium_pattern": "",
                        "status": "new",
                    },
                    "display": f"{word} ({entry.reading}): + NEW [{ojad_str}]",
                }

            kanjium_str = ",".join(map(str, kanjium_patterns))
            kanjium_set = set(kanjium_patterns)
            ojad_set = set(entry.patterns)

            if ojad_set == kanjium_set or ojad_set.issubset(kanjium_set):
                return {
                    "type": "verified",
                    "data": {
                        "surface": entry.surface,
                        "reading": entry.reading,
                        "ojad_pattern": ojad_str,
                        "kanjium_pattern": kanjium_str,
                        "status": "verified",
                    },
                    "display": f"{word} ({entry.reading}): ✓ VERIFIED [{kanjium_str}]",
                }

            # Conflict
            common = kanjium_set & ojad_set
            only_kanjium = kanjium_set - ojad_set
            only_ojad = ojad_set - kanjium_set

            delta_parts = []
            if only_ojad:
                delta_parts.append(f"+ojad:{','.join(map(str, sorted(only_ojad)))}")
            if only_kanjium:
                delta_parts.append(f"-kanjium:{','.join(map(str, sorted(only_kanjium)))}")

            return {
                "type": "conflicts",
                "data": {
                    "surface": entry.surface,
                    "reading": entry.reading,
                    "ojad_pattern": ojad_str,
                    "kanjium_pattern": kanjium_str,
                    "delta": " ".join(delta_parts),
                    "common": ",".join(map(str, sorted(common))) if common else "",
                    "only_ojad": ",".join(map(str, sorted(only_ojad))) if only_ojad else "",
                    "only_kanjium": ",".join(map(str, sorted(only_kanjium))) if only_kanjium else "",
                    "status": "conflict",
                },
                "display": f"{word} ({entry.reading}): ⚠ CONFLICT [K:{kanjium_str} vs O:{ojad_str}]",
            }

        except Exception as e:
            return {
                "type": "errors",
                "data": {"word": word, "error": str(e)},
                "display": f"{word}: ERROR - {e}",
            }

    # Use ThreadPoolExecutor for parallel scraping
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        future_to_word = {executor.submit(process_word, word): word for word in words}

        # Create progress bar or simple counter
        if use_progress:
            pbar = tqdm(
                as_completed(future_to_word),
                total=len(words),
                desc="Validating",
                unit="word",
                ncols=80,
            )
        else:
            pbar = as_completed(future_to_word)

        completed = 0
        for future in pbar:
            completed += 1
            word = future_to_word[future]
            try:
                result = future.result()
                results[result["type"]].append(result["data"])

                # Update progress bar description or print status
                if use_progress:
                    # Show current word status in progress bar
                    status_char = {"verified": "✓", "conflicts": "⚠", "new": "+", "not_found": "?", "errors": "✗"}
                    pbar.set_postfix_str(f"{status_char.get(result['type'], '?')} {word[:10]}")
                else:
                    print(f"[{completed}/{len(words)}] {result['display']}")

            except Exception as e:
                results["errors"].append({"word": word, "error": str(e)})
                if not use_progress:
                    print(f"[{completed}/{len(words)}] {word}: ERROR - {e}")

        if use_progress:
            pbar.close()
            # Ensure progress bar output is complete before continuing
            sys.stdout.flush()
            time.sleep(0.1)

    elapsed = time.time() - start_time

    # Generate reports
    print("\n" + "-" * 60)
    print("Generating reports...")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Write conflicts CSV (main QA file)
    conflicts_path = output_dir / f"qa_conflicts_{timestamp}.csv"
    if results["conflicts"]:
        with open(conflicts_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "surface", "reading", "kanjium_pattern", "ojad_pattern",
                "delta", "common", "only_ojad", "only_kanjium", "status",
            ])
            writer.writeheader()
            writer.writerows(results["conflicts"])
        print(f"  Written: {conflicts_path} ({len(results['conflicts']):,} conflicts)")

    # Write verified CSV
    verified_path = output_dir / f"qa_verified_{timestamp}.csv"
    if results["verified"]:
        with open(verified_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "surface", "reading", "kanjium_pattern", "ojad_pattern", "status",
            ])
            writer.writeheader()
            writer.writerows(results["verified"])
        print(f"  Written: {verified_path} ({len(results['verified']):,} verified)")

    # Write new entries CSV
    new_path = output_dir / f"qa_new_{timestamp}.csv"
    if results["new"]:
        with open(new_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "surface", "reading", "ojad_pattern", "kanjium_pattern", "status",
            ])
            writer.writeheader()
            writer.writerows(results["new"])
        print(f"  Written: {new_path} ({len(results['new']):,} new entries)")

    # Write not found CSV
    if results["not_found"]:
        not_found_path = output_dir / f"qa_not_found_{timestamp}.csv"
        with open(not_found_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["word", "reason"])
            writer.writeheader()
            writer.writerows(results["not_found"])
        print(f"  Written: {not_found_path} ({len(results['not_found']):,} not found)")

    # Write summary markdown
    summary_path = output_dir / f"qa_summary_{timestamp}.md"
    _write_summary_markdown(summary_path, results, word_list_path, elapsed, len(kanjium_data))
    print(f"  Written: {summary_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("QA Validation Summary")
    print("=" * 60)
    print(f"\nTotal words processed: {len(words):,}")
    print(f"Time elapsed: {elapsed:.1f}s ({elapsed/len(words):.2f}s per word)")
    print()
    print("Results:")
    print(f"  ✓ Verified:    {len(results['verified']):>6,}")
    print(f"  ⚠ Conflicts:   {len(results['conflicts']):>6,}")
    print(f"  + New:         {len(results['new']):>6,}")
    print(f"  ? Not found:   {len(results['not_found']):>6,}")
    print(f"  ✗ Errors:      {len(results['errors']):>6,}")

    if results["conflicts"]:
        print(f"\n→ Review conflicts in: {conflicts_path}")

    return 0


def _write_summary_markdown(
    path: Path,
    results: dict,
    word_list_path: Path,
    elapsed: float,
    kanjium_total: int,
) -> None:
    """Write QA summary markdown report."""
    total = (
        len(results["verified"]) +
        len(results["conflicts"]) +
        len(results["new"]) +
        len(results["not_found"]) +
        len(results["errors"])
    )

    verified_pct = len(results["verified"]) / total * 100 if total else 0
    conflicts_pct = len(results["conflicts"]) / total * 100 if total else 0

    content = f"""# OJAD QA Validation Report

Generated: {datetime.now().isoformat()}

## Configuration

| Setting | Value |
|---------|-------|
| Word list | `{word_list_path}` |
| Kanjium entries | {kanjium_total:,} |
| Words validated | {total:,} |
| Time elapsed | {elapsed:.1f}s |

## Results Summary

| Status | Count | Percentage |
|--------|-------|------------|
| ✓ Verified | {len(results['verified']):,} | {verified_pct:.1f}% |
| ⚠ Conflicts | {len(results['conflicts']):,} | {conflicts_pct:.1f}% |
| + New (OJAD only) | {len(results['new']):,} | - |
| ? Not in OJAD | {len(results['not_found']):,} | - |
| ✗ Errors | {len(results['errors']):,} | - |

## Conflict Analysis

Conflicts occur when OJAD patterns differ from Kanjium:
- **+ojad** = patterns in OJAD but not Kanjium (potential additions)
- **-kanjium** = patterns in Kanjium but not OJAD (potential removals)

## Files Generated

- `qa_conflicts_*.csv` - **Review this file** - entries with pattern differences
- `qa_verified_*.csv` - Entries where patterns match
- `qa_new_*.csv` - Entries only in OJAD (not in Kanjium)
- `qa_not_found_*.csv` - Words not found in OJAD

## Next Steps

1. Review `qa_conflicts_*.csv` manually
2. For each conflict, decide:
   - **Add OJAD patterns** - if OJAD has valid additional patterns
   - **Keep Kanjium** - if Kanjium is correct
   - **Flag for research** - if unclear
3. Run with `--apply` to update database (after review)

## Top Conflicts Preview

"""

    # Add top 10 conflicts preview
    if results["conflicts"]:
        content += "| Word | Kanjium | OJAD | Delta |\n"
        content += "|------|---------|------|-------|\n"
        for entry in results["conflicts"][:10]:
            content += f"| {entry['surface']} ({entry['reading']}) | {entry['kanjium_pattern']} | {entry['ojad_pattern']} | {entry['delta']} |\n"
        if len(results["conflicts"]) > 10:
            content += f"\n*... and {len(results['conflicts']) - 10} more conflicts*\n"
    else:
        content += "*No conflicts found!*\n"

    path.write_text(content, encoding="utf-8")


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Mode selection
    if args.scrape:
        return run_scrape_mode(args)
    elif args.batch:
        return run_batch_mode(args)
    else:
        # Show help if no mode specified
        print("Error: Please specify either --scrape WORD or --batch FILE")
        print("\nExamples:")
        print("  python scripts/import_ojad.py --scrape 東京")
        print("  python scripts/import_ojad.py --batch wordlists/common.txt")
        print("\nRun with --help for more options.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
