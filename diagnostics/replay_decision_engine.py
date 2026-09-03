"""
First real milestone for core/routing_decision_engine.py, per Jon's
explicit instruction: "Start with a small implementation capable of
replaying already-recorded sensor results through the engine... A good
first milestone is: recorded evidence JSON -> Decision Engine ->
route/quarantine/audit record."

Feeds REAL, already-recorded evidence through the engine - no new
inference, no live sensor calls:
  - Gemma: `data/outputs/vit_family_benchmark/gemma_flat8/
    gemma_flat8_test_predictions.json` (the flat-8-bucket production
    prompt run, the fairer of the two Gemma runs recorded - see
    docs/MULTI_SOURCE_VOTING_CLASSIFIER_PROPOSAL.md's "Flat 8-bucket
    Gemma re-run" section)
  - 7 fine-tuned vision towers: `data/outputs/vit_family_benchmark/
    <name>/<name>_test_logits.json` (full per-image logit vectors,
    already extracted from the saved checkpoints)
All on the SAME 332-image held-out split (diagnostics/vit_family_
benchmark_common.py's get_split(), seed 42).

HONEST GAP, stated up front rather than papered over: Gemma's
production `classify()` path does not expose a raw decision-token logit
margin - only the self-reported confidence field, which this project
has repeatedly measured to be a weak signal (see the proposal doc's
several confirmations). Using that field as a stand-in for
"score_kind=logit_margin" would misrepresent the engine's own threshold
semantics (which were calibrated against REAL logit-margin numbers from
a different experiment, on a different sample). This replay therefore
treats Gemma's evidence as UNSCORED (raw_score=None, normalized_score=
None) rather than quietly substituting a known-unreliable number - the
engine already has a defined, tested behavior for unscored evidence
(treated conservatively, see core/routing_decision_engine.py's
_is_confident()/_is_weak() returning None). A real raw-logit-margin
Gemma adapter is a genuine next step (see the module's own
"INTEGRATION STATUS" note), not built here.

Usage:
    python diagnostics/replay_decision_engine.py
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

REPORT_PATH = PROJECT_ROOT / "data" / "logs" / "reviewed" / "decision_engine_replay_report.txt"

TOWER_NAMES = ["convnext", "mobilenetv2", "dinov2", "beit", "swin", "siglip", "vit21k"]


def load_evidence_sources() -> tuple[dict, dict]:
    gemma_path = BENCHMARK_ROOT / "gemma_flat8" / "gemma_flat8_test_predictions.json"
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
        evidence.append(EvidenceRecord(
            sensor_name="gemma", sensor_family="gemma_semantic",
            predicted_class=g["pred"],
            raw_score=None, normalized_score=None,  # see module docstring - honest gap
            calibrated=False,
            metadata={"score_kind": None, "self_reported_confidence": g.get("confidence"),
                      "note": "raw logit margin not captured for this production run"},
        ))
    for name in TOWER_NAMES:
        t = tower_data[name].get(path)
        if t is None:
            continue
        family = {
            "convnext": "cnn_modern", "mobilenetv2": "cnn_lightweight_independent",
            "dinov2": "vit_isotropic_strong_cluster", "beit": "vit_isotropic_strong_cluster",
            "siglip": "vit_isotropic_strong_cluster", "vit21k": "vit_isotropic_strong_cluster",
            "swin": "hierarchical_transformer",
        }[name]
        evidence.append(EvidenceRecord(
            sensor_name=name, sensor_family=family, predicted_class=t["pred"],
            raw_score=t["top1_top2_margin"], normalized_score=t["top1_softmax"],
            uncertainty=t["entropy"], calibrated=False,
            metadata={"score_kind": "softmax_probability"},
        ))
    return evidence


def main():
    lines = []
    def log(s=""):
        lines.append(s)
        print(s)

    log("Loading real recorded evidence (no new inference)...")
    gemma_data, tower_data = load_evidence_sources()
    by_cat, train_items, val_items, test_items = get_split()
    log(f"Held-out test set: {len(test_items)} images\n")

    config = load_decision_config()
    log(f"Loaded config/decision_engine.yaml - default_policy={config.default_policy!r}\n")

    results = []
    for path, gt in test_items:
        evidence = build_evidence_for_image(path, gemma_data, tower_data)
        result = decide(evidence, config, image_id=path)
        results.append((path, gt, result))

    # -- overall trust-state distribution + accuracy conditional on trust state --
    log("=" * 90)
    log(f"REPLAY RESULTS - policy={config.default_policy!r} (from config, unmodified)")
    log("=" * 90)
    by_state = defaultdict(list)
    for path, gt, result in results:
        by_state[result.trust_state].append((path, gt, result))

    log(f"\nTrust-state distribution across {len(results)} images:")
    for state in TrustState:
        n = len(by_state.get(state, []))
        log(f"  {state.value:<20s} {n} ({100*n/len(results):.1f}%)")

    log("\nAccuracy conditional on trust state (selected_class == ground truth):")
    for state in TrustState:
        items = by_state.get(state, [])
        if not items:
            continue
        correct = sum(1 for _, gt, r in items if r.selected_class == gt)
        log(f"  {state.value:<20s} {correct}/{len(items)} ({100*correct/len(items):.1f}%)")

    overall_correct = sum(1 for _, gt, r in results if r.selected_class == gt)
    log(f"\nOverall (all trust states, selected_class == gt): {overall_correct}/{len(results)} "
        f"({100*overall_correct/len(results):.1f}%)")

    # -- sample audit trails: one AUTO_ACCEPT, one QUARANTINE, one where selected != gt --
    log("\n\n" + "=" * 90)
    log("SAMPLE AUDIT TRAILS")
    log("=" * 90)

    def show_sample(label, items):
        if not items:
            log(f"\n-- {label}: none found --")
            return
        path, gt, result = items[0]
        log(f"\n-- {label} --")
        log(f"Image: {path}")
        log(f"Ground truth: {gt}")
        for line in result.audit_trail:
            log(f"  {line}")

    auto_accepts = by_state.get(TrustState.AUTO_ACCEPT, [])
    quarantines = by_state.get(TrustState.QUARANTINE, [])
    wrong_selections = [(p, gt, r) for p, gt, r in results if r.selected_class and r.selected_class != gt]

    show_sample("Example AUTO_ACCEPT", auto_accepts)
    show_sample("Example QUARANTINE", quarantines)
    show_sample("Example where selected_class != ground truth (any trust state)", wrong_selections)

    # -- policy comparison on the SAME evidence: prove config-driven switching on real data --
    log("\n\n" + "=" * 90)
    log("POLICY COMPARISON ON REAL DATA (same evidence, policy overridden per call - "
        "no config file edited, no code changed)")
    log("=" * 90)
    for alt_policy in ["gemma_primary", "vision_primary", "weighted_fusion"]:
        alt_results = []
        for path, gt in test_items:
            evidence = build_evidence_for_image(path, gemma_data, tower_data)
            result = decide(evidence, config, policy=alt_policy, image_id=path)
            alt_results.append((path, gt, result))
        correct = sum(1 for _, gt, r in alt_results if r.selected_class == gt)
        state_counts = Counter(r.trust_state.value for _, _, r in alt_results)
        auto_accept_correct = sum(
            1 for _, gt, r in alt_results
            if r.trust_state == TrustState.AUTO_ACCEPT and r.selected_class == gt)
        auto_accept_n = state_counts.get("auto_accept", 0)
        log(f"\n  policy={alt_policy:<25s} overall_acc={correct}/{len(alt_results)} "
            f"({100*correct/len(alt_results):.1f}%)  trust_states={dict(state_counts)}")
        if auto_accept_n:
            log(f"    AUTO_ACCEPT-only accuracy: {auto_accept_correct}/{auto_accept_n} "
                f"({100*auto_accept_correct/auto_accept_n:.1f}%)")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(f"\n\nReport: {REPORT_PATH}")


if __name__ == "__main__":
    main()
