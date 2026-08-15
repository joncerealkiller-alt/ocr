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


if __name__ == "__main__":
    test_comparator()
    test_normalize()
    test_loader_dispatch()
    print(f"\n{_PASS} passed, {_FAIL} failed")
    sys.exit(1 if _FAIL else 0)
