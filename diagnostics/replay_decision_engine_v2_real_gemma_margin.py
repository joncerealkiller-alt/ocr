"""
Second replay pass, per Jon's explicit follow-up after the Decision
Engine's first milestone: "Once [the live Gemma logit-margin adapter]
exists, you can finally evaluate gemma_primary fairly against the
vision-primary and weighted policies using real uncertainty from both
sides." diagnostics/replay_decision_engine.py's first pass had to
record Gemma's evidence UNSCORED (no raw logit margin was available
from the production call path at that time) - this replay uses
diagnostics/run_gemma_logit_adapter_on_benchmark_holdout.py's real,
live-extracted decision-token logit margins instead, so gemma_primary
can finally reach AUTO_ACCEPT on its genuinely confident cases, not
just ACCEPT_WITH_CAUTION on all of them (the first replay's honest,
stated limitation).

Same 332-image held-out split, same 7 vision towers, same config
(config/decision_engine.yaml) as the first replay - ONLY the Gemma
evidence source changed (gemma_flat8_logit_margin/ instead of
gemma_flat8/), so any difference in trust-state distribution or
policy-comparison numbers is attributable to that one change, not a
confound from anything else moving at the same time.

Usage:
    python diagnostics/replay_decision_engine_v2_real_gemma_margin.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.routing_decision_engine import (
    DecisionConfig, EvidenceRecord, TrustState, decide, load_decision_config,
)
from diagnostics.vit_family_benchmark_common import BENCHMARK_ROOT, get_split

REPORT_PATH = PROJECT_ROOT / "data" / "logs" / "reviewed" / "decision_engine_replay_v2_real_gemma_margin_report.txt"

TOWER_NAMES = ["convnext", "mobilenetv2", "dinov2", "beit", "swin", "siglip", "vit21k"]

TOWER_FAMILY = {
    "convnext": "cnn_modern", "mobilenetv2": "cnn_lightweight_independent",
    "dinov2": "vit_isotropic_strong_cluster", "beit": "vit_isotropic_strong_cluster",
    "siglip": "vit_isotropic_strong_cluster", "vit21k": "vit_isotropic_strong_cluster",
    "swin": "hierarchical_transformer",
}


def load_evidence_sources() -> tuple[dict, dict]:
    gemma_path = BENCHMARK_ROOT / "gemma_flat8_logit_margin" / "gemma_flat8_logit_margin_test_predictions.json"
    gemma_data = json.loads(gemma_path.read_text(encoding="utf-8"))["predictions"]

    tower_data = {}
    for name in TOWER_NAMES:
        path = BENCHMARK_ROOT / name / f"{name}_test_logits.json"
        tower_data[name] = json.loads(path.read_text(encoding="utf-8"))["predictions"]
    return gemma_data, tower_data


def build_evidence_for_image(path: str, gemma_data: dict, tower_data: dict) -> list[EvidenceRecord]:
    evidence = []
    g = gemma_data.get(path)
    if g and g.get("pred"):
        margin = g.get("raw_logit_margin")
        evidence.append(EvidenceRecord(
            sensor_name="gemma", sensor_family="gemma_semantic",
            predicted_class=g["pred"],
            raw_score=margin, normalized_score=None,
            calibrated=False,
            metadata={
                "score_kind": "logit_margin" if margin is not None else None,
                "self_reported_confidence": g.get("confidence"),
                "decision_token_top1_text": g.get("decision_token_top1_text"),
                "decision_token_top2_text": g.get("decision_token_top2_text"),
                "note": "real live-extracted decision-token logit margin (v2 replay)",
            },
        ))
    for name in TOWER_NAMES:
        t = tower_data[name].get(path)
        if t is None:
            continue
        evidence.append(EvidenceRecord(
            sensor_name=name, sensor_family=TOWER_FAMILY[name], predicted_class=t["pred"],
            raw_score=t["top1_top2_margin"], normalized_score=t["top1_softmax"],
            uncertainty=t["entropy"], calibrated=False,
            metadata={"score_kind": "softmax_probability"},
        ))
    return evidence


def run_policy_pass(policy_name, config, test_items, gemma_data, tower_data):
    results = []
    for path, gt in test_items:
        evidence = build_evidence_for_image(path, gemma_data, tower_data)
        result = decide(evidence, config, policy=policy_name, image_id=path)
        results.append((path, gt, result))
    return results


def main():
    lines = []
    def log(s=""):
        lines.append(s)
        print(s)

    log("Loading real recorded evidence, including the NEW live Gemma logit-margin run...")
    gemma_data, tower_data = load_evidence_sources()
    by_cat, train_items, val_items, test_items = get_split()
    log(f"Held-out test set: {len(test_items)} images")
    margin_available = sum(1 for g in gemma_data.values() if g.get("raw_logit_margin") is not None)
    log(f"Gemma real logit margin available for: {margin_available}/{len(gemma_data)} images\n")

    config = load_decision_config()
    log(f"Loaded config/decision_engine.yaml - default_policy={config.default_policy!r}\n")

    log("=" * 90)
    log("PART 1: gemma_primary WITH real logit margin (vs. the first replay's unscored version)")
    log("=" * 90)
    results = run_policy_pass("gemma_primary", config, test_items, gemma_data, tower_data)
    by_state = defaultdict(list)
    for path, gt, result in results:
        by_state[result.trust_state].append((path, gt, result))

    log(f"\nTrust-state distribution across {len(results)} images (gemma_primary, real margin):")
    for state in TrustState:
        n = len(by_state.get(state, []))
        log(f"  {state.value:<20s} {n} ({100*n/len(results):.1f}%)")
    log("\nAccuracy conditional on trust state:")
    for state in TrustState:
        items = by_state.get(state, [])
        if not items:
            continue
        correct = sum(1 for _, gt, r in items if r.selected_class == gt)
        log(f"  {state.value:<20s} {correct}/{len(items)} ({100*correct/len(items):.1f}%)")
    overall_correct = sum(1 for _, gt, r in results if r.selected_class == gt)
    log(f"\nOverall: {overall_correct}/{len(results)} ({100*overall_correct/len(results):.1f}%)")

    auto_accepts = by_state.get(TrustState.AUTO_ACCEPT, [])
    log(f"\n*** KEY COMPARISON: gemma_primary reached AUTO_ACCEPT on {len(auto_accepts)} images "
        f"this time (0 in the first replay, since Gemma's evidence was unscored there) ***")
    if auto_accepts:
        path, gt, result = auto_accepts[0]
        log(f"\nExample AUTO_ACCEPT (real margin):")
        log(f"Image: {path}\nGround truth: {gt}")
        for line in result.audit_trail:
            log(f"  {line}")

    log("\n\n" + "=" * 90)
    log("PART 2: fair 3-way policy comparison, real uncertainty on both sides now")
    log("=" * 90)
    for policy_name in ["gemma_primary", "vision_primary", "weighted_fusion"]:
        alt_results = run_policy_pass(policy_name, config, test_items, gemma_data, tower_data)
        correct = sum(1 for _, gt, r in alt_results if r.selected_class == gt)
        state_counts = Counter(r.trust_state.value for _, _, r in alt_results)
        auto_accept_items = [(p, gt, r) for p, gt, r in alt_results if r.trust_state == TrustState.AUTO_ACCEPT]
        auto_accept_correct = sum(1 for _, gt, r in auto_accept_items if r.selected_class == gt)
        auto_accept_n = len(auto_accept_items)
        log(f"\n  policy={policy_name:<25s} overall_acc={correct}/{len(alt_results)} "
            f"({100*correct/len(alt_results):.1f}%)  trust_states={dict(state_counts)}")
        if auto_accept_n:
            log(f"    AUTO_ACCEPT-only accuracy: {auto_accept_correct}/{auto_accept_n} "
                f"({100*auto_accept_correct/auto_accept_n:.1f}%)")

    log("\n\n" + "=" * 90)
    log("PART 3: margin-vs-correctness sanity check on the real Gemma data (does the live "
        "adapter's margin actually track correctness, same as the earlier standalone experiments?)")
    log("=" * 90)
    import statistics
    correct_margins = [g["raw_logit_margin"] for g in gemma_data.values()
                        if g.get("correct") and g.get("raw_logit_margin") is not None]
    wrong_margins = [g["raw_logit_margin"] for g in gemma_data.values()
                      if not g.get("correct") and g.get("raw_logit_margin") is not None]
    if correct_margins:
        log(f"Margin when Gemma correct (n={len(correct_margins)}): "
            f"mean={statistics.mean(correct_margins):.2f}  median={statistics.median(correct_margins):.2f}")
    if wrong_margins:
        log(f"Margin when Gemma WRONG (n={len(wrong_margins)}): "
            f"mean={statistics.mean(wrong_margins):.2f}  median={statistics.median(wrong_margins):.2f}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(f"\n\nReport: {REPORT_PATH}")


if __name__ == "__main__":
    main()
