#!/usr/bin/env python3
"""Import pitch accent data from Wadoku XML dump.

Wadoku is a Japanese-German dictionary with pitch accent information.
Download: https://www.wadoku.de/wiki/display/WAD/Downloads+und+Links

Usage:
    python import_wadoku.py                    # Parse and show stats
    python import_wadoku.py --apply            # Apply to Kanjium database
    python import_wadoku.py --export wadoku.csv  # Export to CSV
"""

import argparse
import re
import sqlite3
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Set
from collections import defaultdict


# Paths
SCRIPTS_DIR = Path(__file__).parent
DATA_DIR = SCRIPTS_DIR.parent / "data"
WADOKU_XML = DATA_DIR / "wadoku-xml-20260104" / "wadoku.xml"
KANJIUM_DB = DATA_DIR / "pitch.db"


@dataclass
class WadokuEntry:
    """Entry parsed from Wadoku XML."""
    entry_id: str
    surface: str  # Kanji/word form
    reading: str  # Hiragana reading
    accents: List[int]  # Pitch accent positions
    pos: str  # Part of speech (if available)


def parse_wadoku_xml(xml_path: Path) -> Iterator[WadokuEntry]:
    """Parse Wadoku XML and yield entries with pitch accent.

    Uses iterparse for memory efficiency with large XML files.

    Args:
        xml_path: Path to wadoku.xml file.

    Yields:
        WadokuEntry for each entry with pitch accent data.
    """
    # Wadoku uses default namespace (no prefix) in each entry
    NS = "{http://www.wadoku.de/xml/entry}"

    # Track progress
    entries_seen = 0
    entries_with_accent = 0

    # Use iterparse for memory efficiency
    context = ET.iterparse(str(xml_path), events=("end",))

    for event, elem in context:
        # Only process entry elements (with or without namespace)
        if not (elem.tag.endswith("entry") or elem.tag == f"{NS}entry"):
            continue

        entries_seen += 1

        # Get entry ID
        entry_id = elem.get("id", "")

        # Find form element (try with and without namespace)
        form = elem.find(f"{NS}form")
        if form is None:
            form = elem.find("form")
        if form is None:
            elem.clear()
            continue

        # Get orthography (surface form)
        # Prefer midashigo="true" entries, otherwise take first
        orths = form.findall(f"{NS}orth")
        if not orths:
            orths = form.findall("orth")
        surface = None
        for orth in orths:
            if orth.get("midashigo") == "true":
                surface = orth.text
                break
        if surface is None and orths:
            surface = orths[0].text

        if not surface:
            elem.clear()
            continue

        # Clean surface (remove △, ×, parentheses markers)
        surface = clean_surface(surface)
        if not surface:
            elem.clear()
            continue

        # Get reading element
        reading_elem = form.find(f"{NS}reading")
        if reading_elem is None:
            reading_elem = form.find("reading")
        if reading_elem is None:
            elem.clear()
            continue

        # Get hiragana reading
        hira = reading_elem.find(f"{NS}hira")
        if hira is None:
            hira = reading_elem.find("hira")
        if hira is None or not hira.text:
            elem.clear()
            continue

        reading = hira.text.strip()

        # Get accent values
        accent_elems = reading_elem.findall(f"{NS}accent")
        if not accent_elems:
            accent_elems = reading_elem.findall("accent")
        if not accent_elems:
            elem.clear()
            continue

        accents = []
        for acc in accent_elems:
            if acc.text and acc.text.strip().isdigit():
                accents.append(int(acc.text.strip()))

        if not accents:
            elem.clear()
            continue

        entries_with_accent += 1

        # Get part of speech (optional)
        pos = ""
        gram = elem.find(f"{NS}gramGrp")
        if gram is None:
            gram = elem.find("gramGrp")
        if gram is not None:
            # Check for common POS tags
            if gram.find(f"{NS}meishi") is not None or gram.find("meishi") is not None:
                pos = "名詞"
            elif gram.find(f"{NS}doushi") is not None or gram.find("doushi") is not None:
                pos = "動詞"
            elif gram.find(f"{NS}keiyoushi") is not None or gram.find("keiyoushi") is not None:
                pos = "形容詞"
            elif gram.find(f"{NS}fukushi") is not None or gram.find("fukushi") is not None:
                pos = "副詞"

        yield WadokuEntry(
            entry_id=entry_id,
            surface=surface,
            reading=reading,
            accents=sorted(set(accents)),
            pos=pos,
        )

        # Clear element to free memory
        elem.clear()

        # Progress update
        if entries_seen % 50000 == 0:
            print(f"  Processed {entries_seen:,} entries, {entries_with_accent:,} with accent...")

    print(f"Total: {entries_seen:,} entries, {entries_with_accent:,} with pitch accent")


def clean_surface(text: str) -> str:
    """Clean surface form by removing markers.

    Wadoku uses various markers:
    - △ = rare kanji
    - × = non-standard kanji
    - (い) = optional okurigana
    - ･ = word boundary in compounds

    Args:
        text: Raw surface text.

    Returns:
        Cleaned surface text.
    """
    if not text:
        return ""

    # Remove markers
    text = text.replace("△", "")
    text = text.replace("×", "")
    text = text.replace("･", "")

    # Remove parenthesized optional parts like (い), (き)
    text = re.sub(r"\([ぁ-んァ-ン]\)", "", text)

    # Remove other parentheses content
    text = re.sub(r"\([^)]*\)", "", text)

    return text.strip()


def normalize_reading(reading: str) -> str:
    """Normalize reading for comparison.

    Args:
        reading: Hiragana reading.

    Returns:
        Normalized reading.
    """
    # Remove special characters
    reading = reading.replace("…", "")
    reading = reading.replace("･", "")

    # Convert small kana variations
    # (Wadoku might use different conventions)

    return reading.strip()


def load_wadoku_entries(xml_path: Path = WADOKU_XML) -> List[WadokuEntry]:
    """Load all Wadoku entries with pitch accent.

    Args:
        xml_path: Path to wadoku.xml.

    Returns:
        List of WadokuEntry objects.
    """
    if not xml_path.exists():
        raise FileNotFoundError(f"Wadoku XML not found: {xml_path}")

    print(f"Parsing Wadoku XML: {xml_path}")
    entries = list(parse_wadoku_xml(xml_path))
    print(f"Loaded {len(entries):,} entries with pitch accent")

    return entries


def get_kanjium_entries() -> dict:
    """Load Kanjium entries for comparison.

    Returns:
        Dict mapping (surface, reading) to entry data.
    """
    if not KANJIUM_DB.exists():
        raise FileNotFoundError(f"Kanjium database not found: {KANJIUM_DB}")

    conn = sqlite3.connect(str(KANJIUM_DB))
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT id, surface, reading, accent_pattern, confidence, verified_sources
        FROM pitch_accents
    """).fetchall()

    entries = {}
    for row in rows:
        key = (row["surface"], row["reading"])
        entries[key] = dict(row)

    conn.close()
    return entries


def analyze_coverage(wadoku_entries: List[WadokuEntry], kanjium_entries: dict) -> dict:
    """Analyze overlap between Wadoku and Kanjium.

    Args:
        wadoku_entries: List of Wadoku entries.
        kanjium_entries: Dict of Kanjium entries.

    Returns:
        Analysis statistics.
    """
    stats = {
        "wadoku_total": len(wadoku_entries),
        "kanjium_total": len(kanjium_entries),
        "exact_match": 0,
        "surface_match": 0,
        "pattern_match": 0,
        "pattern_conflict": 0,
        "new_entries": 0,
        "new_patterns": 0,
    }

    # Group Kanjium by surface for partial matching
    kanjium_by_surface = defaultdict(list)
    for (surface, reading), data in kanjium_entries.items():
        kanjium_by_surface[surface].append((reading, data))

    for entry in wadoku_entries:
        key = (entry.surface, entry.reading)

        if key in kanjium_entries:
            # Exact match (surface + reading)
            stats["exact_match"] += 1

            # Compare patterns
            kanjium_data = kanjium_entries[key]
            kanjium_pattern = kanjium_data.get("accent_pattern", "")
            try:
                kanjium_accents = set(int(p) for p in kanjium_pattern.split(",") if p.strip())
            except:
                kanjium_accents = set()

            wadoku_accents = set(entry.accents)

            if wadoku_accents == kanjium_accents:
                stats["pattern_match"] += 1
            elif wadoku_accents & kanjium_accents:
                stats["pattern_conflict"] += 1
                stats["new_patterns"] += len(wadoku_accents - kanjium_accents)
            else:
                stats["pattern_conflict"] += 1
                stats["new_patterns"] += len(wadoku_accents)

        elif entry.surface in kanjium_by_surface:
            # Surface matches but reading differs
            stats["surface_match"] += 1
        else:
            # New entry not in Kanjium
            stats["new_entries"] += 1

    return stats


def apply_to_kanjium(wadoku_entries: List[WadokuEntry], dry_run: bool = True) -> dict:
    """Apply Wadoku data to Kanjium database.

    Args:
        wadoku_entries: List of Wadoku entries.
        dry_run: If True, don't actually modify database.

    Returns:
        Statistics about changes made.
    """
    conn = sqlite3.connect(str(KANJIUM_DB))
    conn.row_factory = sqlite3.Row

    stats = {
        "verified": 0,
        "patterns_added": 0,
        "confidence_updated": 0,
        "skipped": 0,
    }

    for entry in wadoku_entries:
        # Find matching Kanjium entry
        row = conn.execute("""
            SELECT id, surface, reading, accent_pattern, confidence, verified_sources
            FROM pitch_accents
            WHERE surface = ? AND reading = ?
        """, (entry.surface, entry.reading)).fetchone()

        if not row:
            # Try matching by surface only
            row = conn.execute("""
                SELECT id, surface, reading, accent_pattern, confidence, verified_sources
                FROM pitch_accents
                WHERE surface = ?
                LIMIT 1
            """, (entry.surface,)).fetchone()

        if not row:
            stats["skipped"] += 1
            continue

        # Parse current patterns
        current_pattern = row["accent_pattern"] or ""
        try:
            current_accents = set(int(p) for p in current_pattern.split(",") if p.strip())
        except:
            current_accents = set()

        wadoku_accents = set(entry.accents)

        # Merge patterns
        merged_accents = current_accents | wadoku_accents
        new_pattern = ",".join(str(a) for a in sorted(merged_accents))

        # Update confidence
        current_confidence = row["confidence"] or 50
        if wadoku_accents == current_accents:
            # Perfect match - higher confidence boost
            new_confidence = min(100, current_confidence + 15)
            stats["verified"] += 1
        elif wadoku_accents & current_accents:
            # Partial overlap - moderate boost
            new_confidence = min(100, current_confidence + 10)
            stats["patterns_added"] += len(wadoku_accents - current_accents)
        else:
            # No overlap - smaller boost (alternative patterns)
            new_confidence = min(100, current_confidence + 5)
            stats["patterns_added"] += len(wadoku_accents)

        # Update verified sources
        verified_sources = row["verified_sources"] or ""
        if "Wadoku" not in verified_sources:
            new_sources = f"{verified_sources},Wadoku" if verified_sources else "Wadoku"
        else:
            new_sources = verified_sources

        if not dry_run:
            conn.execute("""
                UPDATE pitch_accents
                SET accent_pattern = ?, confidence = ?, verified_sources = ?
                WHERE id = ?
            """, (new_pattern, new_confidence, new_sources, row["id"]))

        stats["confidence_updated"] += 1

    if not dry_run:
        conn.commit()

    conn.close()
    return stats


def insert_new_entries(wadoku_entries: List[WadokuEntry], dry_run: bool = True) -> dict:
    """Insert new Wadoku entries not in Kanjium database.

    Args:
        wadoku_entries: List of Wadoku entries.
        dry_run: If True, don't actually modify database.

    Returns:
        Statistics about insertions.
    """
    conn = sqlite3.connect(str(KANJIUM_DB))
    conn.row_factory = sqlite3.Row

    # Get existing entries for deduplication
    existing = set()
    rows = conn.execute("SELECT surface, reading FROM pitch_accents").fetchall()
    for row in rows:
        existing.add((row["surface"], row["reading"]))

    print(f"  Existing entries in database: {len(existing):,}")

    stats = {
        "inserted": 0,
        "skipped_duplicate": 0,
        "skipped_empty": 0,
    }

    batch = []
    batch_size = 1000

    for entry in wadoku_entries:
        key = (entry.surface, entry.reading)

        # Skip if already exists
        if key in existing:
            stats["skipped_duplicate"] += 1
            continue

        # Skip entries with empty/invalid data
        if not entry.surface or not entry.reading or not entry.accents:
            stats["skipped_empty"] += 1
            continue

        # Format accent pattern
        accent_pattern = ",".join(str(a) for a in entry.accents)

        batch.append((
            entry.surface,
            entry.reading,
            accent_pattern,
            None,  # goshu
            None,  # goshu_jp
            None,  # frequency_rank
            "wadoku",  # data_source
            70,  # confidence (Wadoku is reliable)
            "Wadoku",  # verified_sources
            None,  # variation_note
        ))

        # Mark as existing to prevent duplicates in same batch
        existing.add(key)
        stats["inserted"] += 1

        # Insert in batches
        if len(batch) >= batch_size:
            if not dry_run:
                conn.executemany("""
                    INSERT INTO pitch_accents
                    (surface, reading, accent_pattern, goshu, goshu_jp,
                     frequency_rank, data_source, confidence, verified_sources, variation_note)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, batch)
                conn.commit()
            print(f"    Inserted {stats['inserted']:,} entries...")
            batch = []

    # Insert remaining batch
    if batch and not dry_run:
        conn.executemany("""
            INSERT INTO pitch_accents
            (surface, reading, accent_pattern, goshu, goshu_jp,
             frequency_rank, data_source, confidence, verified_sources, variation_note)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, batch)
        conn.commit()

    conn.close()
    return stats


def export_to_csv(wadoku_entries: List[WadokuEntry], output_path: Path) -> None:
    """Export Wadoku entries to CSV.

    Args:
        wadoku_entries: List of entries.
        output_path: Output CSV path.
    """
    import csv

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["surface", "reading", "accents", "pos", "entry_id"])

        for entry in wadoku_entries:
            writer.writerow([
                entry.surface,
                entry.reading,
                ",".join(str(a) for a in entry.accents),
                entry.pos,
                entry.entry_id,
            ])

    print(f"Exported {len(wadoku_entries):,} entries to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Import pitch accent data from Wadoku",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--apply", "-a",
        action="store_true",
        help="Apply Wadoku data to Kanjium database (update existing entries)"
    )
    parser.add_argument(
        "--add-new", "-n",
        action="store_true",
        help="Add new Wadoku entries not in Kanjium database"
    )
    parser.add_argument(
        "--export", "-e",
        type=str,
        help="Export Wadoku entries to CSV file"
    )
    parser.add_argument(
        "--xml", "-x",
        type=str,
        default=str(WADOKU_XML),
        help="Path to wadoku.xml file"
    )

    args = parser.parse_args()

    # Load Wadoku data
    wadoku_entries = load_wadoku_entries(Path(args.xml))

    # Show basic stats
    print(f"\n{'=' * 50}")
    print("Wadoku Statistics:")
    print(f"  Total entries with pitch accent: {len(wadoku_entries):,}")

    # Count unique surfaces
    surfaces = set(e.surface for e in wadoku_entries)
    print(f"  Unique surface forms: {len(surfaces):,}")

    # Count by POS
    pos_counts = defaultdict(int)
    for e in wadoku_entries:
        pos_counts[e.pos or "unknown"] += 1
    print(f"  By POS: {dict(pos_counts)}")

    # Analyze coverage against Kanjium
    if KANJIUM_DB.exists():
        print(f"\n{'=' * 50}")
        print("Coverage Analysis (vs Kanjium):")

        kanjium_entries = get_kanjium_entries()
        stats = analyze_coverage(wadoku_entries, kanjium_entries)

        print(f"  Kanjium entries: {stats['kanjium_total']:,}")
        print(f"  Exact matches (surface+reading): {stats['exact_match']:,}")
        print(f"  Surface-only matches: {stats['surface_match']:,}")
        print(f"  Pattern matches: {stats['pattern_match']:,}")
        print(f"  Pattern conflicts: {stats['pattern_conflict']:,}")
        print(f"  New patterns to add: {stats['new_patterns']:,}")
        print(f"  New entries (not in Kanjium): {stats['new_entries']:,}")

    # Export if requested
    if args.export:
        export_to_csv(wadoku_entries, Path(args.export))

    # Apply if requested
    if args.apply:
        print(f"\n{'=' * 50}")
        print("Applying Wadoku data to Kanjium...")

        apply_stats = apply_to_kanjium(wadoku_entries, dry_run=False)

        print(f"  Verified (patterns match): {apply_stats['verified']:,}")
        print(f"  New patterns added: {apply_stats['patterns_added']:,}")
        print(f"  Entries updated: {apply_stats['confidence_updated']:,}")
        print(f"  Skipped (not in Kanjium): {apply_stats['skipped']:,}")
        print("Done!")

    # Add new entries if requested
    if args.add_new:
        print(f"\n{'=' * 50}")
        print("Adding new Wadoku entries to Kanjium...")

        insert_stats = insert_new_entries(wadoku_entries, dry_run=False)

        print(f"  New entries inserted: {insert_stats['inserted']:,}")
        print(f"  Skipped (already exists): {insert_stats['skipped_duplicate']:,}")
        print(f"  Skipped (empty/invalid): {insert_stats['skipped_empty']:,}")
        print("Done!")


if __name__ == "__main__":
    main()
