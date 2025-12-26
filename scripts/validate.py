#!/usr/bin/env python3
"""Validate corrections.csv before merging.

Run from mierutone-dictionary root:
    python scripts/validate.py
"""

import csv
import sys
from pathlib import Path


def validate_corrections():
    corrections_path = Path("data/corrections.csv")

    if not corrections_path.exists():
        print("No corrections.csv found - nothing to validate")
        return True

    errors = []
    warnings = []

    with open(corrections_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required_fields = {"surface", "reading", "accent_pattern"}
        if not required_fields.issubset(set(reader.fieldnames or [])):
            errors.append(f"Missing required columns: {required_fields - set(reader.fieldnames or [])}")
            return False

        for i, row in enumerate(reader, start=2):  # Start at 2 (header is 1)
            # Check required fields
            if not row.get("surface"):
                errors.append(f"Line {i}: Missing surface")
            if not row.get("reading"):
                errors.append(f"Line {i}: Missing reading")
            if not row.get("accent_pattern"):
                errors.append(f"Line {i}: Missing accent_pattern")

            # Validate accent_pattern is numeric
            pattern = row.get("accent_pattern", "")
            if pattern and not pattern.isdigit():
                errors.append(f"Line {i}: accent_pattern '{pattern}' is not a number")

            # Validate goshu if present
            valid_goshu = {"wago", "kango", "gairaigo", "proper", "mixed", ""}
            goshu = row.get("goshu", "")
            if goshu and goshu not in valid_goshu:
                errors.append(f"Line {i}: Invalid goshu '{goshu}'")

            # Check for source (warning only)
            if not row.get("source"):
                warnings.append(f"Line {i}: No source provided for {row.get('surface')}")

    # Print results
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"  {w}")

    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  {e}")
        return False

    print("Validation passed!")
    return True


if __name__ == "__main__":
    success = validate_corrections()
    sys.exit(0 if success else 1)
