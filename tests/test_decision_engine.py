"""
Tests for core/routing_decision_engine.py, built 2026-08-07 alongside
the module itself. No pytest in this environment (checked before
writing this - `python -c "import pytest"` fails, no requirements.txt/
pyproject.toml at project root) - plain assert-based, directly runnable,
matching this project's existing diagnostics/test_*.py convention
rather than introducing a new test framework dependency.

Covers the 8 scenarios Jon specified:
  1. strong Gemma + strong vision agreement
  2. strong Gemma / weak vision disagreement
  3. weak Gemma / strong vision disagreement
  4. multiple trained vision sensors agreeing against Gemma
  5. all evidence weak
  6. correlated towers agreeing but independent CV evidence contradicting
  7. class-specific authority override
  8. policy switched from gemma_primary to vision_primary WITHOUT code
     changes (config-only) - the most important test per Jon's explicit
     "I want proof that changing the primary authority is
     configuration-driven" requirement.

Usage:
    python tests/test_decision_engine.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.routing_decision_engine import (
    DecisionConfig, EvidenceRecord, TrustState, decide,
)

_PASS = 0
_FAIL = 0


def check(condition: bool, label: str):
    global _PASS, _FAIL
    if condition:
        _PASS += 1
        print(f"  PASS: {label}")
    else:
        _FAIL += 1
        print(f"  FAIL: {label}")


def gemma(pred: str, margin: float) -> EvidenceRecord:
    return EvidenceRecord(
        sensor_name="gemma", sensor_family="gemma_semantic", predicted_class=pred,
        raw_score=margin, metadata={"score_kind": "logit_margin"},
    )


def tower(name: str, family: str, pred: str, softmax: float) -> EvidenceRecord:
    return EvidenceRecord(
        sensor_name=name, sensor_family=family, predicted_class=pred,
        normalized_score=softmax, metadata={"score_kind": "softmax_probability"},
    )


def cv_signal(name: str, pred: str, confidence: float) -> EvidenceRecord:
    return EvidenceRecord(
        sensor_name=name, sensor_family="classical_cv", predicted_class=pred,
        normalized_score=confidence, metadata={"score_kind": "ordinal_confidence"},
    )


def default_config() -> DecisionConfig:
    cfg = DecisionConfig()
    cfg.sensor_families = {
        "gemma_semantic": ["gemma"],
        "vit_isotropic_strong_cluster": ["dinov2", "beit", "siglip", "vit21k"],
        "cnn_modern": ["convnext"],
        "cnn_lightweight_independent": ["mobilenetv2"],
        "hierarchical_transformer": ["swin"],
        "classical_cv": ["table_confidence", "row_regularity", "column_grid"],
    }
    return cfg


def test_1_strong_gemma_strong_vision_agreement():
    print("\n[Test 1] Strong Gemma + strong vision agreement")
    cfg = default_config()
    evidence = [
        gemma("dense_tabular_rows", margin=24.2),
        tower("siglip", "vit_isotropic_strong_cluster", "dense_tabular_rows", 0.96),
        cv_signal("table_confidence", "dense_tabular_rows", 1.0),
        cv_signal("row_regularity", "dense_tabular_rows", 0.9),
    ]
    result = decide(evidence, cfg)
    check(result.selected_class == "dense_tabular_rows", "selected_class == dense_tabular_rows")
    check(result.trust_state == TrustState.AUTO_ACCEPT, f"trust_state == AUTO_ACCEPT (got {result.trust_state})")
    check(len(result.audit_trail) > 0, "audit_trail is non-empty")


def test_2_strong_gemma_weak_vision_disagreement():
    print("\n[Test 2] Strong Gemma / weak vision disagreement (matches Jon's worked example 1)")
    cfg = default_config()
    evidence = [
        gemma("dense_tabular_rows", margin=24.2),
        tower("siglip", "vit_isotropic_strong_cluster", "dense_tabular_rows", 0.96),
        cv_signal("row_regularity", "dense_tabular_rows", 0.85),
        cv_signal("table_confidence", "dense_tabular_rows", 0.9),
        tower("convnext", "cnn_modern", "printed_document", 0.55),  # weak disagreement
    ]
    result = decide(evidence, cfg)
    check(result.selected_class == "dense_tabular_rows", "selected_class == dense_tabular_rows")
    check(result.trust_state == TrustState.AUTO_ACCEPT,
          f"trust_state == AUTO_ACCEPT despite one weak contradiction (got {result.trust_state})")


def test_3_weak_gemma_strong_vision_disagreement():
    print("\n[Test 3] Weak Gemma / strong vision disagreement (matches Jon's worked example 2)")
    cfg = default_config()
    evidence = [
        gemma("dense_tabular_rows", margin=3.8),
        tower("siglip", "vit_isotropic_strong_cluster", "printed_document", 0.95),
        cv_signal("row_regularity", "dense_tabular_rows", 0.3),
    ]
    result = decide(evidence, cfg)
    check(result.trust_state == TrustState.QUARANTINE, f"trust_state == QUARANTINE (got {result.trust_state})")
    check(result.quarantine_reason is not None, "quarantine_reason is set")


def test_4_multiple_vision_sensors_against_gemma():
    print("\n[Test 4] Multiple trained vision sensors (different families) agreeing against Gemma")
    cfg = default_config()
    evidence = [
        gemma("website_screenshot", margin=22.0),  # confident but wrong per the towers
        tower("siglip", "vit_isotropic_strong_cluster", "printed_document", 0.93),
        tower("convnext", "cnn_modern", "printed_document", 0.91),
        tower("mobilenetv2", "cnn_lightweight_independent", "printed_document", 0.88),
    ]
    result = decide(evidence, cfg)
    check(result.trust_state == TrustState.QUARANTINE,
          f"trust_state == QUARANTINE (2+ independent families confidently disagree, got {result.trust_state})")
    check(result.selected_class == "website_screenshot",
          "selected_class stays gemma's pick even under QUARANTINE (arbitration doesn't silently flip)")


def test_5_all_evidence_weak():
    print("\n[Test 5] All evidence weak")
    cfg = default_config()
    evidence = [
        gemma("printed_document", margin=2.1),
        tower("siglip", "vit_isotropic_strong_cluster", "mixed_text_image", 0.40),
        cv_signal("table_confidence", "printed_document", 0.2),
    ]
    result = decide(evidence, cfg)
    check(result.trust_state == TrustState.QUARANTINE, f"trust_state == QUARANTINE (got {result.trust_state})")


def test_6_correlated_towers_agree_independent_cv_contradicts():
    print("\n[Test 6] Correlated towers agree, independent CV evidence contradicts")
    cfg = default_config()
    # 4 members of the SAME family agreeing should not be treated as 4x
    # the evidence weight of the 1 independent CV signal - use
    # weighted_fusion here since that's the policy whose whole point is
    # family-weighted (not raw-count) voting.
    evidence = [
        tower("dinov2", "vit_isotropic_strong_cluster", "dense_tabular_rows", 0.9),
        tower("beit", "vit_isotropic_strong_cluster", "dense_tabular_rows", 0.9),
        tower("siglip", "vit_isotropic_strong_cluster", "dense_tabular_rows", 0.9),
        tower("vit21k", "vit_isotropic_strong_cluster", "dense_tabular_rows", 0.9),
        cv_signal("row_regularity", "printed_document", 0.85),
    ]
    result = decide(evidence, cfg, policy="weighted_fusion")
    # family-weighted: vit_isotropic_strong_cluster = 1 vote (not 4),
    # classical_cv = 1 vote -> a genuine 1-vs-1 tie/near-tie, not a 4-1 blowout
    check(result.trust_state in (TrustState.QUARANTINE, TrustState.ACCEPT_WITH_CAUTION),
          f"trust_state reflects genuine near-tie once family-weighted, not false confidence "
          f"(got {result.trust_state})")


def test_7_class_specific_authority_override():
    print("\n[Test 7] Class-specific authority override")
    cfg = default_config()
    cfg.class_specific_authority = {
        "map_land_record": {"primary": "dinov2", "supporting": ["gemma"]},
    }
    evidence = [
        gemma("map_land_record", margin=15.0),
        tower("dinov2", "vit_isotropic_strong_cluster", "map_land_record", 0.97),
    ]
    result = decide(evidence, cfg, policy="class_specific_authority")
    check(result.selected_class == "map_land_record", "selected_class == map_land_record")
    check(result.policy_used.startswith("class_specific_authority"),
          f"policy_used reflects class_specific_authority (got {result.policy_used})")
    check(any("dinov2" in line for line in result.audit_trail),
          "audit_trail shows dinov2 was consulted as the configured primary")


def test_8_policy_switch_via_config_only():
    print("\n[Test 8] *** Policy switch from gemma_primary to vision_primary WITHOUT code changes ***")
    cfg_gemma = default_config()
    cfg_gemma.default_policy = "gemma_primary"

    cfg_vision = default_config()
    cfg_vision.default_policy = "vision_primary"

    # a case where Gemma and vision disagree, both confident enough to
    # win under their own policy
    evidence = [
        gemma("printed_document", margin=22.0),
        tower("siglip", "vit_isotropic_strong_cluster", "map_land_record", 0.94),
        tower("convnext", "cnn_modern", "map_land_record", 0.90),
    ]

    result_gemma_policy = decide(evidence, cfg_gemma)  # policy=None -> reads cfg.default_policy
    result_vision_policy = decide(evidence, cfg_vision)  # SAME evidence, SAME decide() call signature

    check(result_gemma_policy.policy_used == "gemma_primary",
          f"gemma_primary config used gemma_primary policy (got {result_gemma_policy.policy_used})")
    check(result_vision_policy.policy_used == "vision_primary",
          f"vision_primary config used vision_primary policy (got {result_vision_policy.policy_used})")
    check(result_gemma_policy.selected_class == "printed_document",
          f"gemma_primary selected Gemma's class (got {result_gemma_policy.selected_class})")
    check(result_vision_policy.selected_class == "map_land_record",
          f"vision_primary selected the vision consensus class (got {result_vision_policy.selected_class})")
    check(result_gemma_policy.selected_class != result_vision_policy.selected_class,
          "the TWO policies genuinely produced DIFFERENT routing decisions on IDENTICAL evidence, "
          "driven entirely by a config field - no core/routing_decision_engine.py code changed "
          "between the two decide() calls above")


def test_no_evidence_at_all():
    print("\n[Extra] No evidence at all -> NO_DECISION, not a crash")
    cfg = default_config()
    result = decide([], cfg)
    check(result.trust_state == TrustState.NO_DECISION, f"trust_state == NO_DECISION (got {result.trust_state})")
    check(result.selected_class is None, "selected_class is None")


def main():
    test_1_strong_gemma_strong_vision_agreement()
    test_2_strong_gemma_weak_vision_disagreement()
    test_3_weak_gemma_strong_vision_disagreement()
    test_4_multiple_vision_sensors_against_gemma()
    test_5_all_evidence_weak()
    test_6_correlated_towers_agree_independent_cv_contradicts()
    test_7_class_specific_authority_override()
    test_8_policy_switch_via_config_only()
    test_no_evidence_at_all()

    print(f"\n{'='*60}")
    print(f"RESULTS: {_PASS} passed, {_FAIL} failed")
    print(f"{'='*60}")
    if _FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
