"""
Phase 6 of the Stage-4 error-analysis pipeline (2026-08-08): decision
framework INFRASTRUCTURE - thresholds and rules mapping a failure-cause
distribution (Phase 2's manual review, once complete) to recommended
engineering actions. Every threshold here is a PROPOSED STARTING POINT,
explicitly marked as such, not a validated cutoff - matching docs/
GEMMA_HIDDEN_STATE_ERROR_ANALYSIS_STAGE4_RESEARCH.md section 8's own
caveat that these are directional defaults given how few failure images
exist to classify (~26-32 per location).

This script runs the rules against WHATEVER review progress exists in
failure_annotations.json right now (via failure_taxonomy.py) - if Phase 2's
manual review isn't done yet (the expected state immediately after this
handoff), it reports that plainly rather than fabricating a verdict from
0 reviewed items.

Usage:
    python diagnostics/error_analysis/phase6_decision_framework.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics.error_analysis.common import OUT_DIR
from diagnostics.error_analysis.failure_taxonomy import (
    load_annotations, cause_distribution_report, PENDING_REVIEW,
)

DECISION_RULES_PATH = OUT_DIR / "phase6_decision_rules.json"
DECISION_REPORT_PATH = OUT_DIR / "phase6_decision_report.json"

STATUS = "PROPOSED - NOT YET VALIDATED"

# Each rule: which primary causes it aggregates over, the proposed
# threshold (fraction of REVIEWED failures), and the recommended action
# if that threshold is met. Thresholds/wording reproduced from docs/
# GEMMA_HIDDEN_STATE_ERROR_ANALYSIS_STAGE4_RESEARCH.md section 8 as data,
# not re-derived here.
DECISION_RULES = [
    {
        "rule_id": "vision_limitation_dominant",
        "causes": ["missing_visual_info"],
        "threshold_fraction": 0.60,
        "comparison": ">=",
        "action": "Vision-tower fine-tuning is justified (per docs/GEMMA_VISION_TOWER_FINETUNE_RESEARCH.md), "
                  "though temper expectations given that doc's finding that Gemma's frozen tower already "
                  "starts near ceiling relative to the zero-shot baselines that produced big fine-tune gains "
                  "for other architectures in this project.",
        "status": STATUS,
    },
    {
        "rule_id": "reasoning_limitation_dominant",
        "causes": ["requires_reasoning"],
        "threshold_fraction": 0.50,
        "comparison": ">=",
        "action": "Favor full-generation fallback for this failure subset via Decision Engine fusion "
                  "(route low-confidence hidden-state predictions to full Gemma generation), NOT vision-tower "
                  "fine-tuning - fine-tuning the vision tower cannot add prompt-conditioned reasoning.",
        "status": STATUS,
    },
    {
        "rule_id": "preprocessing_dominant",
        "causes": ["image_quality", "skew_crop", "low_contrast", "preprocessing_artifact"],
        "threshold_fraction": 0.30,
        "comparison": ">=",
        "action": "Preprocessing improvements (Stage A) are the higher-leverage fix - a pipeline-input "
                  "problem that would degrade full generation and every hidden-state location equally, "
                  "not specific to any modeling choice.",
        "status": STATUS,
    },
    {
        "rule_id": "taxonomy_refinement_needed",
        "causes": ["ambiguous_taxonomy"],
        "threshold_fraction": 0.20,
        "comparison": ">=",
        "action": "Taxonomy refinement (classifier_guidance / category-definition edits) - no amount of "
                  "additional training fixes a genuinely undefined category boundary.",
        "status": STATUS,
    },
    {
        "rule_id": "label_review_needed",
        "causes": ["label_error"],
        "threshold_fraction": 0.10,
        "comparison": ">=",
        "action": "Manual label review for the flagged subset - not a modeling change at all.",
        "status": STATUS,
    },
    {
        "rule_id": "no_dominant_cause",
        "causes": "ANY_SINGLE_CAUSE_BELOW_ALL_ABOVE_THRESHOLDS",
        "threshold_fraction": None,
        "comparison": "fallback",
        "action": "Decision Engine fusion (multiple independent sensors covering each other's blind spots) "
                  "is the most robust response when no single cause dominates - matches this project's "
                  "standing philosophy that no single stage needs to be perfect if fusion covers the gaps.",
        "status": STATUS,
    },
]


def evaluate_rules(distribution: dict) -> list[dict]:
    """Applies DECISION_RULES to a cause_distribution_report() payload.
    Returns one verdict per rule, each explicitly marked with the rule's
    own PROPOSED status - never silently upgraded to a "confirmed"
    recommendation regardless of the numbers, since the thresholds
    themselves are unvalidated per this phase's own scope."""
    n_reviewed = distribution["n_reviewed"]
    pct = distribution["primary_cause_percentages"]

    verdicts = []
    any_threshold_met = False
    for rule in DECISION_RULES:
        if rule["comparison"] == "fallback":
            continue
        causes = rule["causes"]
        fraction = sum(pct.get(c, 0) for c in causes) / 100.0 if n_reviewed > 0 else None
        met = (fraction is not None and fraction >= rule["threshold_fraction"])
        if met:
            any_threshold_met = True
        verdicts.append({
            "rule_id": rule["rule_id"],
            "causes_aggregated": causes,
            "threshold_fraction": rule["threshold_fraction"],
            "observed_fraction": round(fraction, 4) if fraction is not None else None,
            "threshold_met": met,
            "n_reviewed_at_evaluation_time": n_reviewed,
            "recommended_action": rule["action"],
            "status": rule["status"],
        })

    fallback_rule = next(r for r in DECISION_RULES if r["rule_id"] == "no_dominant_cause")
    verdicts.append({
        "rule_id": fallback_rule["rule_id"],
        "causes_aggregated": None,
        "threshold_fraction": None,
        "observed_fraction": None,
        "threshold_met": (n_reviewed > 0 and not any_threshold_met),
        "n_reviewed_at_evaluation_time": n_reviewed,
        "recommended_action": fallback_rule["action"],
        "status": fallback_rule["status"],
    })
    return verdicts


def main():
    DECISION_RULES_PATH.write_text(json.dumps(DECISION_RULES, indent=2), encoding="utf-8")
    print(f"Decision rules (all marked {STATUS}) written to {DECISION_RULES_PATH}")

    annotations = load_annotations()
    if not annotations:
        print("\nNo failure_annotations.json found - run failure_taxonomy.py (Phase 2) first.")
        report = {"status": "NOT_RUN", "reason": "failure_annotations.json missing"}
        DECISION_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return

    distribution = cause_distribution_report(annotations)
    verdicts = evaluate_rules(distribution)

    print(f"\nReview progress: {distribution['n_reviewed']}/{distribution['n_total']} annotations reviewed.")
    if distribution["n_reviewed"] == 0:
        print("\n*** No manual review has been completed yet (Phase 2 built the infrastructure and template "
              "annotations, but every entry is still pending_review). No decision-framework verdict is "
              "meaningful until a human completes the review described in docs/GEMMA_HIDDEN_STATE_ERROR_"
              "ANALYSIS_STAGE4_RESEARCH.md section 2. This is the expected state right after this handoff, "
              "not an error. ***\n")
    else:
        print("\nCause distribution (reviewed items only):")
        for cause, count in distribution["primary_cause_counts"].items():
            pct = distribution["primary_cause_percentages"][cause]
            print(f"  {cause:<24s} {count:3d}  ({pct}%)")
        print("\nRule verdicts (ALL marked PROPOSED - NOT YET VALIDATED, regardless of outcome):")
        for v in verdicts:
            marker = "MET" if v["threshold_met"] else "not met"
            print(f"  [{marker:>7s}] {v['rule_id']}: observed={v['observed_fraction']} "
                  f"vs threshold={v['threshold_fraction']}")

    report = {
        "status": "EVALUATED" if distribution["n_reviewed"] > 0 else "PENDING_MANUAL_REVIEW",
        "review_progress": distribution,
        "rule_verdicts": verdicts,
        "important_caveat": (
            "Every verdict above is derived from PROPOSED, UNVALIDATED thresholds "
            "(docs/GEMMA_HIDDEN_STATE_ERROR_ANALYSIS_STAGE4_RESEARCH.md section 8's own caveat: "
            "percentages will be noisy given how few failure images exist to classify - "
            "treat as directional, not a hard statistical verdict)."
        ),
    }
    DECISION_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nFull decision report written to {DECISION_REPORT_PATH}")


if __name__ == "__main__":
    main()
