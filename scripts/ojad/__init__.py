"""OJAD import and validation module."""

from .yomichan_parser import parse_yomichan_ojad, OJADEntry
from .comparator import compare_entries, ComparisonResult
from .reporter import write_validation_report, write_conflicts, print_summary

__all__ = [
    "parse_yomichan_ojad",
    "OJADEntry",
    "compare_entries",
    "ComparisonResult",
    "write_validation_report",
    "write_conflicts",
    "print_summary",
]
