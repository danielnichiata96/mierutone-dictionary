"""Generate CSV reports from comparison results."""

import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List

from .comparator import ComparisonResult, ComparisonStatus, DiscrepancyType


def write_validation_report(results: List[ComparisonResult], output_path: Path) -> int:
    """Write full validation report to CSV.

    Args:
        results: List of comparison results.
        output_path: Path to output CSV file.

    Returns:
        Number of rows written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "surface",
            "reading",
            "kanjium_pattern",
            "ojad_pattern",
            "status",
            "delta",
            "confidence_change",
            "pos",
        ])

        for result in results:
            kanjium_str = (
                ",".join(map(str, result.kanjium_patterns))
                if result.kanjium_patterns else ""
            )
            ojad_str = ",".join(map(str, result.ojad_patterns))

            writer.writerow([
                result.surface,
                result.reading,
                kanjium_str,
                ojad_str,
                result.status.value,
                result.delta,
                result.confidence_change,
                result.pos,
            ])

    return len(results)


def write_conflicts(results: List[ComparisonResult], output_path: Path) -> int:
    """Write conflicts to CSV for manual review.

    Args:
        results: List of comparison results.
        output_path: Path to output CSV file.

    Returns:
        Number of conflicts written.
    """
    conflicts = [r for r in results if r.status == ComparisonStatus.CONFLICT]

    if not conflicts:
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "surface",
            "reading",
            "kanjium",
            "ojad",
            "discrepancy_type",
            "delta",
            "pos",
            "is_resolved",
        ])

        for result in conflicts:
            kanjium_str = (
                ",".join(map(str, result.kanjium_patterns))
                if result.kanjium_patterns else ""
            )
            ojad_str = ",".join(map(str, result.ojad_patterns))

            writer.writerow([
                result.surface,
                result.reading,
                kanjium_str,
                ojad_str,
                result.discrepancy_type.value,
                result.delta,
                result.pos,
                "FALSE",
            ])

    return len(conflicts)


def write_ojad_only(results: List[ComparisonResult], output_path: Path) -> int:
    """Write OJAD-only entries to CSV (entries not in Kanjium).

    Args:
        results: List of comparison results.
        output_path: Path to output CSV file.

    Returns:
        Number of entries written.
    """
    ojad_only = [r for r in results if r.status == ComparisonStatus.OJAD_ONLY]

    if not ojad_only:
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "surface",
            "reading",
            "ojad_pattern",
            "pos",
        ])

        for result in ojad_only:
            ojad_str = ",".join(map(str, result.ojad_patterns))
            writer.writerow([
                result.surface,
                result.reading,
                ojad_str,
                result.pos,
            ])

    return len(ojad_only)


def print_summary(results: List[ComparisonResult]) -> Dict:
    """Print summary statistics to console.

    Args:
        results: List of comparison results.

    Returns:
        Dict with summary statistics.
    """
    total = len(results)
    if total == 0:
        print("No results to summarize.")
        return {}

    # Count by status
    status_counts = Counter(r.status for r in results)

    verified = status_counts.get(ComparisonStatus.VERIFIED, 0)
    conflicts = status_counts.get(ComparisonStatus.CONFLICT, 0)
    ojad_only = status_counts.get(ComparisonStatus.OJAD_ONLY, 0)

    # Count by discrepancy type (for conflicts)
    discrepancy_counts = Counter(
        r.discrepancy_type for r in results if r.status == ComparisonStatus.CONFLICT
    )

    # Print summary
    print("\n" + "=" * 50)
    print("OJAD Validation Summary")
    print("=" * 50)

    print(f"\nTotal OJAD entries compared: {total:,}")
    print()

    print("Results:")
    print(f"  ✓ Verified (match):     {verified:>6,} ({verified/total*100:>5.1f}%)")
    print(f"  ⚠ Conflicts:            {conflicts:>6,} ({conflicts/total*100:>5.1f}%)")
    print(f"  + OJAD only:            {ojad_only:>6,} ({ojad_only/total*100:>5.1f}%)")

    if conflicts > 0:
        print("\nConflict breakdown:")
        for dtype, count in sorted(discrepancy_counts.items(), key=lambda x: -x[1]):
            print(f"    {dtype.value}: {count:,}")

    print()

    return {
        "total": total,
        "verified": verified,
        "conflicts": conflicts,
        "ojad_only": ojad_only,
        "discrepancy_counts": dict(discrepancy_counts),
    }


def generate_markdown_report(
    results: List[ComparisonResult],
    output_path: Path,
    kanjium_total: int,
) -> None:
    """Generate Markdown summary report.

    Args:
        results: List of comparison results.
        output_path: Path to output Markdown file.
        kanjium_total: Total entries in Kanjium database.
    """
    total = len(results)
    status_counts = Counter(r.status for r in results)

    verified = status_counts.get(ComparisonStatus.VERIFIED, 0)
    conflicts = status_counts.get(ComparisonStatus.CONFLICT, 0)
    ojad_only = status_counts.get(ComparisonStatus.OJAD_ONLY, 0)

    coverage = (verified + conflicts) / kanjium_total * 100 if kanjium_total > 0 else 0

    content = f"""# OJAD Validation Report

Generated: {__import__('datetime').datetime.now().isoformat()}

## Summary

| Metric | Value |
|--------|-------|
| Kanjium entries | {kanjium_total:,} |
| OJAD entries compared | {total:,} |
| Coverage | {coverage:.1f}% |

## Results

| Status | Count | Percentage |
|--------|-------|------------|
| ✓ Verified | {verified:,} | {verified/total*100:.1f}% |
| ⚠ Conflicts | {conflicts:,} | {conflicts/total*100:.1f}% |
| + OJAD only | {ojad_only:,} | {ojad_only/total*100:.1f}% |

## Confidence Impact

- **Verified entries**: +20 confidence (max 100)
- **Conflicts**: -10 confidence (marked for review)
- **OJAD only**: New entries with 50 confidence

## Files Generated

- `validation_report.csv` - Full comparison results
- `conflicts.csv` - Entries needing manual review
- `ojad_only.csv` - Entries only in OJAD

## Next Steps

1. Review `conflicts.csv` manually
2. Mark reviewed entries with `is_resolved=TRUE`
3. Run with `--apply` to update database
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
