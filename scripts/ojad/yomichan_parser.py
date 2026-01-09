"""Parse Yomichan OJAD dictionary format.

Yomichan dictionary format (term_bank_*.json):
[
    [surface, reading, pos_tags, rules, score, [definitions], seq, ""],
    ...
]

For pitch accent dictionaries, the definitions contain pitch patterns like "[0]", "[2]".
"""

import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Union
import urllib.request
import urllib.error


@dataclass
class OJADEntry:
    """Parsed OJAD entry."""
    surface: str
    reading: str
    patterns: List[int]  # e.g., [0] or [0, 2] for multiple patterns
    pos: str  # Part of speech tag


# Yomichan OJAD dictionary sources
YOMICHAN_OJAD_URLS = [
    # Primary: FooSoft's Yomichan dictionaries (archived)
    "https://github.com/FooSoft/yomichan/releases/download/dictionaries/kanjium_pitch_accents.zip",
    # Fallback: Community mirrors
    "https://github.com/MarvNC/yomichan-dictionaries/releases/latest/download/OJAD.zip",
]


def download_yomichan_dict(output_path: Path) -> bool:
    """Download Yomichan OJAD dictionary.

    Args:
        output_path: Where to save the ZIP file.

    Returns:
        True if download succeeded.
    """
    for url in YOMICHAN_OJAD_URLS:
        print(f"Trying: {url}")
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "mierutone-dictionary/1.0"}
            )
            with urllib.request.urlopen(req, timeout=60) as response:
                content = response.read()
                output_path.write_bytes(content)
                print(f"Downloaded: {len(content):,} bytes")
                return True
        except urllib.error.URLError as e:
            print(f"Failed: {e}")
            continue

    return False


def parse_pitch_patterns(definitions: list) -> List[int]:
    """Extract pitch pattern numbers from Yomichan definitions.

    Examples:
        ["[0]"] → [0]
        ["[0]", "[2]"] → [0, 2]
        [{"pitch": [{"position": 0}]}] → [0]  # Structured format

    Args:
        definitions: List of definition entries from Yomichan.

    Returns:
        List of pitch pattern integers.
    """
    patterns = []

    for item in definitions:
        # String format: "[0]", "[2]"
        if isinstance(item, str):
            match = re.search(r'\[(\d+)\]', item)
            if match:
                patterns.append(int(match.group(1)))

        # Structured format (newer Yomichan)
        elif isinstance(item, dict):
            if "pitch" in item:
                for pitch_info in item.get("pitch", []):
                    if "position" in pitch_info:
                        patterns.append(int(pitch_info["position"]))
            elif "pitches" in item:
                for pitch_info in item.get("pitches", []):
                    if "position" in pitch_info:
                        patterns.append(int(pitch_info["position"]))

    return sorted(set(patterns))  # Deduplicate and sort


def parse_yomichan_ojad(source: Union[Path, str]) -> Iterator[OJADEntry]:
    """Parse Yomichan OJAD dictionary.

    Supports:
    - ZIP file (standard Yomichan dict format)
    - Directory with extracted JSON files
    - Single JSON file

    Args:
        source: Path to ZIP, directory, or JSON file.

    Yields:
        OJADEntry for each valid entry.
    """
    source = Path(source)

    if source.suffix == ".zip":
        yield from _parse_from_zip(source)
    elif source.is_dir():
        yield from _parse_from_directory(source)
    elif source.suffix == ".json":
        yield from _parse_from_json(source)
    else:
        raise ValueError(f"Unsupported source format: {source}")


def _parse_from_zip(zip_path: Path) -> Iterator[OJADEntry]:
    """Parse from Yomichan ZIP file."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.startswith("term_bank_") and name.endswith(".json"):
                with zf.open(name) as f:
                    data = json.load(f)
                    yield from _parse_entries(data)


def _parse_from_directory(dir_path: Path) -> Iterator[OJADEntry]:
    """Parse from extracted directory."""
    for json_file in sorted(dir_path.glob("term_bank_*.json")):
        yield from _parse_from_json(json_file)


def _parse_from_json(json_path: Path) -> Iterator[OJADEntry]:
    """Parse from single JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        yield from _parse_entries(data)


def _parse_entries(data: list) -> Iterator[OJADEntry]:
    """Parse Yomichan term entries.

    Format: [surface, reading, pos, rules, score, definitions, seq, ""]
    """
    for entry in data:
        if not isinstance(entry, list) or len(entry) < 6:
            continue

        surface = entry[0]
        reading = entry[1]
        pos = entry[2] if len(entry) > 2 else ""
        definitions = entry[5] if len(entry) > 5 else []

        # Skip entries without reading
        if not surface or not reading:
            continue

        # Parse pitch patterns from definitions
        patterns = parse_pitch_patterns(definitions)

        # Skip entries without valid pitch patterns
        if not patterns:
            continue

        yield OJADEntry(
            surface=surface,
            reading=reading,
            patterns=patterns,
            pos=pos,
        )


def load_or_download_ojad(cache_dir: Path) -> Path:
    """Load OJAD dictionary, downloading if necessary.

    Args:
        cache_dir: Directory to store downloaded files.

    Returns:
        Path to the dictionary (ZIP or directory).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "ojad_yomichan.zip"

    if zip_path.exists():
        print(f"Using cached: {zip_path}")
        return zip_path

    print("Downloading OJAD Yomichan dictionary...")
    if download_yomichan_dict(zip_path):
        return zip_path

    raise RuntimeError("Failed to download OJAD dictionary from any source")
