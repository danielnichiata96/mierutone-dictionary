"""Compare OJAD entries against Kanjium pitch database.

Comparison logic:
1. Match by (surface, reading) - strict matching to avoid homonym confusion
2. Compare accent patterns
3. Classify result: verified, conflict, ojad_only
4. Calculate confidence change
"""

import sqlite3
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from .yomichan_parser import OJADEntry


class ComparisonStatus(Enum):
    """Result of comparing OJAD vs Kanjium."""
    VERIFIED = "verified"        # Patterns match
    CONFLICT = "conflict"        # Patterns differ
    OJAD_ONLY = "ojad_only"      # Only in OJAD, not in Kanjium
    KANJIUM_ONLY = "kanjium_only"  # Only in Kanjium (for completeness)


class DiscrepancyType(Enum):
    """Type of discrepancy for conflicts."""
    NONE = "none"               # No discrepancy (verified)
    MISMATCH = "mismatch"       # Different patterns
    MORA_ERROR = "mora_error"   # Mora count doesn't match pattern
    READING_DIFF = "reading_diff"  # Reading normalization issue
    NEW_ENTRY = "new_entry"     # Not in Kanjium


@dataclass
class ComparisonResult:
    """Result of comparing a single entry."""
    surface: str
    reading: str
    kanjium_patterns: Optional[List[int]]  # None if not in Kanjium
    ojad_patterns: List[int]
    status: ComparisonStatus
    discrepancy_type: DiscrepancyType
    delta: str  # Human-readable delta description
    confidence_change: int  # +20, 0, -10, etc.
    pos: str  # Part of speech from OJAD
    is_resolved: bool = False  # For tracking manual review


# Confidence change values
CONFIDENCE_VERIFIED = +20       # OJAD confirms Kanjium
CONFIDENCE_CONFLICT = -10       # OJAD differs (needs review)
CONFIDENCE_NEW = 0              # New entry from OJAD


def load_kanjium_entries(db_path: Path) -> Dict[Tuple[str, str], List[int]]:
    """Load all Kanjium entries indexed by (surface, reading).

    Args:
        db_path: Path to pitch.db.

    Returns:
        Dict mapping (surface, reading) to list of accent patterns.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        "SELECT surface, reading, accent_pattern FROM pitch_accents"
    )

    entries = {}
    for surface, reading, pattern in cursor:
        key = (surface, reading)

        # Parse pattern (can be "0", "1,2", etc.)
        if pattern:
            patterns = [int(p.strip()) for p in str(pattern).split(",") if p.strip().isdigit()]
        else:
            patterns = []

        # Merge with existing patterns for same key
        if key in entries:
            existing = set(entries[key])
            existing.update(patterns)
            entries[key] = sorted(existing)
        else:
            entries[key] = patterns

    conn.close()
    return entries


def compare_patterns(
    kanjium_patterns: Optional[List[int]],
    ojad_patterns: List[int]
) -> Tuple[ComparisonStatus, DiscrepancyType, str, int]:
    """Compare Kanjium and OJAD patterns.

    Args:
        kanjium_patterns: Patterns from Kanjium (None if not found).
        ojad_patterns: Patterns from OJAD.

    Returns:
        Tuple of (status, discrepancy_type, delta_description, confidence_change).
    """
    if kanjium_patterns is None:
        return (
            ComparisonStatus.OJAD_ONLY,
            DiscrepancyType.NEW_ENTRY,
            f"new:{','.join(map(str, ojad_patterns))}",
            CONFIDENCE_NEW,
        )

    kanjium_set = set(kanjium_patterns)
    ojad_set = set(ojad_patterns)

    # Check for exact match or subset match
    if ojad_set == kanjium_set:
        return (
            ComparisonStatus.VERIFIED,
            DiscrepancyType.NONE,
            "match",
            CONFIDENCE_VERIFIED,
        )

    # OJAD is subset of Kanjium (Kanjium has more patterns, OJAD confirms some)
    if ojad_set.issubset(kanjium_set):
        return (
            ComparisonStatus.VERIFIED,
            DiscrepancyType.NONE,
            f"subset:{','.join(map(str, ojad_patterns))}⊂{','.join(map(str, kanjium_patterns))}",
            CONFIDENCE_VERIFIED,
        )

    # Kanjium is subset of OJAD (OJAD has additional patterns)
    if kanjium_set.issubset(ojad_set):
        new_patterns = ojad_set - kanjium_set
        return (
            ComparisonStatus.CONFLICT,
            DiscrepancyType.MISMATCH,
            f"add:{','.join(map(str, sorted(new_patterns)))}",
            CONFIDENCE_CONFLICT,
        )

    # Complete mismatch or partial overlap
    common = kanjium_set & ojad_set
    only_kanjium = kanjium_set - ojad_set
    only_ojad = ojad_set - kanjium_set

    delta_parts = []
    if common:
        delta_parts.append(f"common:{','.join(map(str, sorted(common)))}")
    if only_kanjium:
        delta_parts.append(f"kanjium:{','.join(map(str, sorted(only_kanjium)))}")
    if only_ojad:
        delta_parts.append(f"ojad:{','.join(map(str, sorted(only_ojad)))}")

    return (
        ComparisonStatus.CONFLICT,
        DiscrepancyType.MISMATCH,
        " | ".join(delta_parts),
        CONFIDENCE_CONFLICT,
    )


def compare_entries(
    ojad_entries: Iterator[OJADEntry],
    kanjium_data: Dict[Tuple[str, str], List[int]],
) -> Iterator[ComparisonResult]:
    """Compare OJAD entries against Kanjium database.

    Args:
        ojad_entries: Iterator of parsed OJAD entries.
        kanjium_data: Dict from load_kanjium_entries().

    Yields:
        ComparisonResult for each OJAD entry.
    """
    for entry in ojad_entries:
        key = (entry.surface, entry.reading)
        kanjium_patterns = kanjium_data.get(key)

        status, discrepancy, delta, confidence = compare_patterns(
            kanjium_patterns,
            entry.patterns,
        )

        yield ComparisonResult(
            surface=entry.surface,
            reading=entry.reading,
            kanjium_patterns=kanjium_patterns,
            ojad_patterns=entry.patterns,
            status=status,
            discrepancy_type=discrepancy,
            delta=delta,
            confidence_change=confidence,
            pos=entry.pos,
        )


def apply_changes(
    results: List[ComparisonResult],
    db_path: Path,
    min_confidence: Optional[int] = None,
) -> Tuple[int, int, int]:
    """Apply comparison results to database.

    Args:
        results: List of comparison results.
        db_path: Path to pitch.db.
        min_confidence: Minimum confidence change to apply (None = apply all).

    Returns:
        Tuple of (verified_count, conflicts_updated, new_entries_added).
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Ensure columns exist
    _ensure_columns(cursor)

    verified = 0
    conflicts_updated = 0
    new_added = 0

    for result in results:
        # Skip if below minimum confidence threshold
        if min_confidence is not None and result.confidence_change < min_confidence:
            continue

        if result.status == ComparisonStatus.VERIFIED:
            # Update confidence for verified entries
            cursor.execute(
                """
                UPDATE pitch_accents
                SET confidence = MIN(100, COALESCE(confidence, 50) + ?)
                WHERE surface = ? AND reading = ?
                """,
                (result.confidence_change, result.surface, result.reading),
            )
            if cursor.rowcount > 0:
                verified += 1

        elif result.status == ComparisonStatus.CONFLICT:
            # Add new patterns (multi-pattern support)
            cursor.execute(
                "SELECT accent_pattern FROM pitch_accents WHERE surface = ? AND reading = ?",
                (result.surface, result.reading),
            )
            row = cursor.fetchone()
            if row:
                existing_patterns = set(
                    int(p.strip()) for p in str(row[0]).split(",") if p.strip().isdigit()
                )
                all_patterns = existing_patterns | set(result.ojad_patterns)
                new_pattern_str = ",".join(map(str, sorted(all_patterns)))

                cursor.execute(
                    """
                    UPDATE pitch_accents
                    SET accent_pattern = ?,
                        confidence = MAX(20, COALESCE(confidence, 50) + ?),
                        origin_source = COALESCE(origin_source, 'kanjium') || '+ojad'
                    WHERE surface = ? AND reading = ?
                    """,
                    (new_pattern_str, result.confidence_change, result.surface, result.reading),
                )
                if cursor.rowcount > 0:
                    conflicts_updated += 1

        elif result.status == ComparisonStatus.OJAD_ONLY:
            # Add new entry from OJAD
            pattern_str = ",".join(map(str, result.ojad_patterns))
            cursor.execute(
                """
                INSERT INTO pitch_accents (surface, reading, accent_pattern, confidence, origin_source, data_source)
                VALUES (?, ?, ?, 50, 'ojad', 'ojad')
                """,
                (result.surface, result.reading, pattern_str),
            )
            if cursor.rowcount > 0:
                new_added += 1

    conn.commit()
    conn.close()

    return verified, conflicts_updated, new_added


def _ensure_columns(cursor: sqlite3.Cursor) -> None:
    """Ensure confidence and origin_source columns exist."""
    cursor.execute("PRAGMA table_info(pitch_accents)")
    columns = {row[1] for row in cursor.fetchall()}

    if "confidence" not in columns:
        cursor.execute(
            "ALTER TABLE pitch_accents ADD COLUMN confidence INTEGER DEFAULT 50"
        )

    if "origin_source" not in columns:
        cursor.execute(
            "ALTER TABLE pitch_accents ADD COLUMN origin_source TEXT DEFAULT 'kanjium'"
        )
