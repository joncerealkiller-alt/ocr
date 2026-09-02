"""
Tests for core/row_extraction.py's stage1<->stage2 agreement comparator
(2026-08-15, hint-free two-stage design - docs/TWO_STAGE_HINT_FREE_
PROPOSAL.md). Plain assert-based, matching this project's existing
tests/test_resource_guard.py convention (no pytest in this env).

Every case here is a REAL shape observed in live runs, not invented:
the containment shapes come from actual smolvlm2 stage-1 output, the
"27"/"7" case is the measured census_pairing_1921 comparator bug, and
the "m2" case is the 1921 Sex-column spill.

Usage:
    python tests/test_field_agreement.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.row_extraction import fields_agree, _normalize_for_agreement, _build_field_loader, _RemoteVllmFieldLoader
from core.loaders.base_loader import load_model_config

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


def test_comparator():
    print("test_comparator")
    cases = [
        ('3. Manitoba', 'Manitoba', True, 'containment - real stage1 noise shape'),
        ('The answer is 18.', '18', True, 'numeric whole-token in noisy string'),
        ('27', '7', False, 'the measured 1921 substring bug stays fixed'),
        ('m2', 'm', False, 'column-spill (1921 Sex) routes to review'),
        ('The answer is F', 'F', True, 'short-code token in noisy string'),
        ('Huggie Bertha 1.', 'Hugie Bertha', False, 'near-miss spellings disagree'),
        ('Huzie Bertha', 'Huzie Bertha|confirmed', True, 'confidence pipe-suffix stripped'),
        ('?', 'Manitoba', False, 'stage1 abstention -> review'),
        ('Manitoba', '?', False, 'stage2 abstention -> review'),
        ('', '18', False, 'empty stage1 -> review'),
        ('5,1', '14', False, 'the hinted-leg age-anchoring case disagrees'),
        ('0 1 2 3 4 5 6 7 8 9 10', '44', False, 'garbled digit-run vs real age'),
        ('The Roberts Buston G.', 'Robert Bustin G.', False, 'near-miss names disagree'),
        ('MANITOBA', 'manitoba.', True, 'case + trailing punctuation normalized'),
    ]
    for s1, s2, expected, why in cases:
        got = fields_agree(s1, s2)
        check(got == expected, f'{why} (fields_agree({s1!r}, {s2!r}) == {expected})')


def test_normalize():
    print("test_normalize")
    check(_normalize_for_agreement('Head|confirmed') == 'head', 'strips |confirmed')
    check(_normalize_for_agreement('  Mc  Roberts. ') == 'mc roberts', 'collapses whitespace, strips trailing period')
    check(_normalize_for_agreement(None) == '', 'None -> empty')


def test_loader_dispatch():
    print("test_loader_dispatch")
    vllm_cfg = load_model_config('gemma_12b_w4a16')
    loader = _build_field_loader('gemma_12b_w4a16', vllm_cfg)
    check(isinstance(loader, _RemoteVllmFieldLoader),
          'runtime=vllm profile dispatches to the remote wrapper (no GPU touched)')
    check(loader.config is vllm_cfg, 'wrapper carries the real config (content_hash etc. work)')
    try:
        loader.apply_checkpoint('anything')
        check(False, 'apply_checkpoint on a vllm wrapper raises')
    except RuntimeError:
        check(True, 'apply_checkpoint on a vllm wrapper raises clearly')

    tf_cfg = load_model_config('smolvlm2_2b')
    tf_loader = _build_field_loader('smolvlm2_2b', tf_cfg)
    check(type(tf_loader).__name__ == 'SmolVLM2Loader',
          'transformers profile dispatches through LOADER_REGISTRY exactly as before')


def test_comparator_hardening():
    """2026-09-02 detector-hardening rules - every case is a measured
    false auto-accept from the 549-cell study (experiments/
    gguf_two_stage_20260902), not a hypothetical. See fields_agree()'s
    docstring for the per-rule evidence."""
    print("test_comparator_hardening")
    check(not fields_agree("5 1 4", "4"), "multi-digit-token stage1 is ambiguous -> review (measured FP: GT 14)")
    check(not fields_agree("5 10", "5"), "multi-digit-token stage1 ambiguous, second shape (measured FP: GT 10)")
    check(not fields_agree("<think>", "12"), "CoT contamination is not a reading (measured FP)")
    check(not fields_agree("Manitoba", "Umanitoba"), "substring mangle rejected at token boundary (measured FP)")
    check(fields_agree("3. Manitoba", "Manitoba"), "token containment still accepts noisy-but-correct wrap")
    check(fields_agree("The answer is 18.", "18"), "single digit token in prose still accepts")


def test_column_schema():
    """Schema gate (Jon's design, 2026-09-02): per-column expected
    content; failures route to review even on agreement."""
    print("test_column_schema")
    from core.row_extraction import column_schema_valid
    check(column_schema_valid("Age", "14"), "age integer")
    check(column_schema_valid("Age", "3 mo"), "age infant months (enumerator abbreviation)")
    check(column_schema_valid("Age", "7/12"), "age months-as-twelfths fraction")
    check(not column_schema_valid("Age", "5 1 4"), "age garbage rejected")
    check(column_schema_valid("Sex", "M."), "sex with trailing period (normalized away)")
    check(not column_schema_valid("Sex", "m2"), "sex column spill rejected (the 1921 case)")
    check(not column_schema_valid("Sex", "M.M."), "sex stutter artifact rejected")
    check(not column_schema_valid("Birthplace", "Eng &"), "birthplace symbol garbage rejected")
    check(column_schema_valid("Birthplace", "Nova Scotia"), "birthplace multiword accepted")
    check(not column_schema_valid("Relationship to Head", "brother 3"), "relationship digit spill rejected")
    check(column_schema_valid("Relationship to Head", "Son-in-law"), "hyphenated relationship accepted")
    check(column_schema_valid("Name", '" Gordon'), "ditto-mark name accepted")
    check(column_schema_valid("UnknownColumn", "x9!!"), "unlisted column permissive by design")
    check(column_schema_valid("Age", "?"), "abstention passes schema (routing is fields_agree's job)")


def test_page_context_veto():
    """Detector layer 3 (2026-09-02): page-local convention veto.
    Synthetic pages mirror the MEASURED shapes from the 549-cell study
    (1931_174's 15/16 province-level profile with the Hamilton FP; the
    1921/31228 pages' 1-trusted-cell abstain condition)."""
    print("test_page_context_veto")
    from core.row_extraction import RowExtractionResult, RowFieldValue, apply_page_context_vetoes

    def mk(ri, bp_value, agree=True):
        return RowExtractionResult(
            row_index=ri, bbox=[0, 0, 1, 1],
            fields={"Birthplace": RowFieldValue(value=bp_value, confidence="confirmed")},
            raw_output="", model="t", runtime_seconds=0.1,
            schema_pass=True, schema_error=None,
            field_agreement={"Birthplace": agree})

    page = [mk(i, v) for i, v in enumerate(
        ["Manitoba"] * 5 + ["Scotland"] * 4 + ["Ontario"] * 3 + ["England"] * 3 + ["Hamilton"])]
    n = apply_page_context_vetoes(page)
    check(n == 1, "strong province-level page vetoes the city-level singleton (the Hamilton FP)")
    check(page[-1].field_agreement["Birthplace"] is False, "vetoed cell routed to review")
    check("page-context" in page[-1].context_vetoes.get("Birthplace", ""), "veto reason recorded")
    check(page[-1].fields["Birthplace"].value == "Hamilton", "value NEVER substituted - review only")
    check(page[0].field_agreement["Birthplace"] is True, "in-vocabulary cells untouched")

    low = [mk(0, "Manitoba"), mk(1, "Hamilton")]
    check(apply_page_context_vetoes(low) == 0 and low[1].field_agreement["Birthplace"],
          "below min_trusted the page abstains (the 1921/31228 condition)")

    mixed = [mk(i, v) for i, v in enumerate(
        ["Manitoba"] * 5 + ["Winnipeg", "Brandon", "Selkirk", "Hamilton", "Dauphin"])]
    check(apply_page_context_vetoes(mixed) == 0, "mixed-granularity page abstains, no false rejections")

    rare = [mk(i, v) for i, v in enumerate(["Manitoba"] * 8 + ["Saskatchewan"])]
    check(apply_page_context_vetoes(rare) == 0 and rare[-1].field_agreement["Birthplace"],
          "rare-but-legitimate in-vocabulary singleton survives (why vocabulary beats frequency)")


if __name__ == "__main__":
    test_comparator()
    test_normalize()
    test_loader_dispatch()
    test_comparator_hardening()
    test_column_schema()
    test_page_context_veto()
    print(f"\n{_PASS} passed, {_FAIL} failed")
    sys.exit(1 if _FAIL else 0)
