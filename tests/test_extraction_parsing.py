"""
Tests for core/extraction_parsing.py - shared parsing logic for the
document_type:/visible_dates:/personal_names:/place_names:/
subject_keywords: extraction prompt contract, used by all 11
extraction-role loaders. No pytest in this environment - plain
assert-based, directly runnable, matching this project's existing
tests/test_run_context.py convention.

Usage:
    python tests/test_extraction_parsing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.extraction_parsing import parse_kv_block, REQUIRED_KEYS

_PASS = 0
_FAIL = 0


def check(condition: bool, description: str) -> None:
    global _PASS, _FAIL
    if condition:
        _PASS += 1
        print(f"  PASS: {description}")
    else:
        _FAIL += 1
        print(f"  FAIL: {description}")


def test_underscore_field_names_unaffected():
    print("\ntest_underscore_field_names_unaffected")
    raw = (
        "document_type: Census Record\n"
        "personal_names: John Smith|confirmed\n"
        "place_names: Toronto|partial\n"
    )
    fields = parse_kv_block(raw)
    check(fields.get("document_type") == "Census Record", "document_type parses unchanged")
    check(fields.get("personal_names") == "John Smith|confirmed", "personal_names parses unchanged")
    check(not (REQUIRED_KEYS - fields.keys()), "all required keys present")


def test_stray_space_after_underscore():
    """Pre-existing confirmed case (SmolVLM2): 'personal_ names:'."""
    print("\ntest_stray_space_after_underscore")
    raw = "document_type: X\npersonal_ names: Jane Doe|confirmed\nplace_names: Y|confirmed\n"
    fields = parse_kv_block(raw)
    check(fields.get("personal_names") == "Jane Doe|confirmed",
          "'personal_ names:' (stray space after underscore) still normalizes correctly")


def test_space_instead_of_underscore():
    """2026-08-15 confirmed case: a printed_document extraction on the
    WSL vLLM path emitted 'Personal names:'/'Place names:'/'Visible
    dates:'/'Subject keywords:' - space where the prompt contract calls
    for an underscore, no underscore at all. Before the fix, this
    silently discarded all four fields as 'unrecognized' (replace(" ",
    "") produced 'personalnames', matching nothing) even though the
    values were correctly pipe-delimited and otherwise well-formed -
    the ACTUAL real-world raw output that triggered this fix."""
    print("\ntest_space_instead_of_underscore")
    raw = (
        "Passenger Declaration: 1920\n"
        "Visible dates: 7|unclear|25|unclear\n"
        "Personal names: Florence Campbell|partial\n"
        "Place names: Victoria BC|partial\n"
        "Subject keywords: Canada, immigration, passenger declaration"
    )
    fields = parse_kv_block(raw)
    check(fields.get("visible_dates") == "7|unclear|25|unclear", "'Visible dates:' normalizes to visible_dates")
    check(fields.get("personal_names") == "Florence Campbell|partial",
          "'Personal names:' normalizes to personal_names")
    check(fields.get("place_names") == "Victoria BC|partial", "'Place names:' normalizes to place_names")
    check(fields.get("subject_keywords") == "Canada, immigration, passenger declaration",
          "'Subject keywords:' normalizes to subject_keywords")
    missing = REQUIRED_KEYS - fields.keys()
    check(missing == {"document_type"},
          "ONLY document_type missing (the model genuinely never labeled it - "
          "not fabricated, not silently dropped alongside the other three)")


def test_duplicate_block_guard_still_works():
    """The degeneration guard (repeated field block -> ValueError) must
    survive the normalization change unchanged."""
    print("\ntest_duplicate_block_guard_still_works")
    raw = (
        "document_type: X\n"
        "personal_names: A|confirmed\n"
        "document_type: X\n"
        "personal_names: A|confirmed\n"
    )
    try:
        parse_kv_block(raw)
        check(False, "duplicate block raises ValueError")
    except ValueError:
        check(True, "duplicate block raises ValueError")


def main():
    test_underscore_field_names_unaffected()
    test_stray_space_after_underscore()
    test_space_instead_of_underscore()
    test_duplicate_block_guard_still_works()

    print(f"\n{'='*60}")
    print(f"RESULTS: {_PASS} passed, {_FAIL} failed")
    print(f"{'='*60}")
    if _FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
