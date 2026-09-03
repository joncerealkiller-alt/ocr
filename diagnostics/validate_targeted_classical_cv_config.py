"""
Re-validates the TARGETED config (2026-08-09): table_confidence/
layout_detector added to config/decision_engine.yaml's dense_tabular_rows
class_specific_authority rule ONLY - never included in the evidence list
passed to gemma_primary/weighted_fusion at all, per Jon's explicit
"supporting evidence for dense_tabular_rows only, keep out of the global
policies" instruction.

This is NOT the same test as diagnostics/validate_classical_cv_sensors_
in_decision_engine.py, which fed classical_cv evidence into ALL THREE
policies uniformly to measure "what if every policy could see it" (that
test is what surfaced the real auto-accept cost under gemma_primary/
weighted_fusion). This script builds two DIFFERENT evidence lists per
image - one without classical_cv (used for gemma_primary/weighted_
fusion) and one with it (used only for class_specific_authority) -
matching how a real production caller would build evidence per policy,
not a single universal list.

Expected result, if the targeted config genuinely isolates the new
sensors: gemma_primary and weighted_fusion should be BYTE-IDENTICAL to
their original "WITHOUT classical_cv" numbers (94.0% / 92.8%, 111 / 305
auto_accepts) - not just similar, identical, since they never receive
this evidence at all now. class_specific_authority should match the
earlier "WITH classical_cv" result (310/332, 95/97 on dense_tabular_rows).

Classical-CV evidence (analyze_image() + DocLayout-YOLO) is cached to
disk on first run - a re-run (e.g. after a future config tweak) reuses
the cache instead of re-running CV/YOLO inference.

Usage:
    python diagnostics/validate_targeted_classical_cv_config.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from core.image_analysis import analyze_image
from core.layout_detector import build_layout_model, detect_layout
from core.sensor_adapters import table_confidence_evidence, layout_detector_evidence
from core.routing_decision_engine import EvidenceRecord, decide, load_decision_config
from diagnostics.vit_family_benchmark_common import BENCHMARK_ROOT, get_split
from diagnostics.error_analysis.common import OLD_WORKING_PREFIX, LEGACY_WORKING_IMAGES_DIR

REPORT_PATH = PROJECT_ROOT / "data" / "logs" / "reviewed" / "targeted_classical_cv_config_validation_report.txt"
CACHE_PATH = PROJECT_ROOT / "data" / "outputs" / "classical_cv_evidence_cache.json"

TOWER_NAMES = ["convnext", "mobilenetv2", "dinov2", "beit", "swin", "siglip", "vit21k"]
TOWER_FAMILY = {
    "convnext": "cnn_modern", "mobilenetv2": "cnn_lightweight_independent",
    "dinov2": "vit_isotropic_strong_cluster", "beit": "vit_isotropic_strong_cluster",
    "siglip": "vit_isotropic_strong_cluster", "vit21k": "vit_isotropic_strong_cluster",
    "swin": "hierarchical_transformer",
}


def resolve_path(path_str: str) -> str:
    if Path(path_str).exists():
        return path_str
    if path_str.startswith(OLD_WORKING_PREFIX):
        filename = path_str[len(OLD_WORKING_PREFIX):].lstrip("\\/")
        remapped = str(LEGACY_WORKING_IMAGES_DIR / filename)
        if Path(remapped).exists():
            return remapped
    return path_str


def load_recorded_evidence_sources() -> tuple[dict, dict]:
    gemma_path = BENCHMARK_ROOT / "gemma_flat8_logit_margin" / "gemma_flat8_logit_margin_test_predictions.json"
    gemma_data = json.loads(gemma_path.read_text(encoding="utf-8"))["predictions"]
    tower_data = {}
    for name in TOWER_NAMES:
        path = BENCHMARK_ROOT / name / f"{name}_test_logits.json"
        tower_data[name] = json.loads(path.read_text(encoding="utf-8"))["predictions"]
    return gemma_data, tower_data


def build_base_evidence(path: str, gemma_data: dict, tower_data: dict) -> list[EvidenceRecord]:
    evidence = []
    g = gemma_data.get(path)
    if g and g.get("pred"):
        margin = g.get("raw_logit_margin")
        evidence.append(EvidenceRecord(
            sensor_name="gemma", sensor_family="gemma_semantic", predicted_class=g["pred"],
            raw_score=margin, calibrated=False,
            metadata={"score_kind": "logit_margin" if margin is not None else None},
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


def compute_classical_cv_cache(test_items, log) -> dict:
    if CACHE_PATH.exists():
        log(f"Loading cached classical-CV evidence from {CACHE_PATH}...")
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))

    log("No cache found - computing classical-CV evidence live (analyze_image() + DocLayout-YOLO)...")
    layout_model = build_layout_model()
    cache = {}
    for i, (path, gt) in enumerate(test_items, 1):
        resolved = resolve_path(path)
        try:
            img = Image.open(resolved).convert("RGB")
        except Exception as e:
            log(f"  [{i}/{len(test_items)}] FAILED to open {resolved}: {e}")
            cache[path] = {"table_confidence": None, "layout_detector": None}
            continue

        analysis = analyze_image(img)
        tc = table_confidence_evidence(analysis)
        detections = detect_layout(layout_model, img)
        ld = layout_detector_evidence(detections)

        cache[path] = {
            "table_confidence": tc.to_dict() if tc else None,
            "layout_detector": ld.to_dict() if ld else None,
        }
        if i % 50 == 0:
            log(f"  [{i}/{len(test_items)}] processed")

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    log(f"Cached to {CACHE_PATH}")
    return cache


def evidence_record_from_dict(d: dict) -> EvidenceRecord:
    return EvidenceRecord(
        sensor_name=d["sensor_name"], sensor_family=d["sensor_family"],
        predicted_class=d["predicted_class"], raw_score=d["raw_score"],
        normalized_score=d["normalized_score"], uncertainty=d["uncertainty"],
        calibrated=d["calibrated"], reliability_by_class=d["reliability_by_class"],
        metadata=d["metadata"],
    )


def main():
    lines = []
    def log(s=""):
        lines.append(s)
        print(s, flush=True)

    log("Targeted-config re-validation: table_confidence/layout_detector added to "
        "dense_tabular_rows's class_specific_authority rule ONLY - gemma_primary and "
        "weighted_fusion never receive this evidence at all in this test.\n")

    gemma_data, tower_data = load_recorded_evidence_sources()
    by_cat, train_items, val_items, test_items = get_split()
    log(f"Held-out test set: {len(test_items)} images\n")

    cv_cache = compute_classical_cv_cache(test_items, log)
    config = load_decision_config()
    gt_by_path = dict(test_items)

    # Two evidence lists per image: base-only (for gemma_primary/
    # weighted_fusion) and base+classical_cv (for class_specific_authority
    # ONLY - matches how a real caller would build per-policy evidence,
    # not a single universal list every policy shares).
    base_evidence_by_path = {}
    full_evidence_by_path = {}
    for path, gt in test_items:
        base = build_base_evidence(path, gemma_data, tower_data)
        base_evidence_by_path[path] = base
        cv_entry = cv_cache.get(path, {})
        classical_cv = []
        if cv_entry.get("table_confidence"):
            classical_cv.append(evidence_record_from_dict(cv_entry["table_confidence"]))
        if cv_entry.get("layout_detector"):
            classical_cv.append(evidence_record_from_dict(cv_entry["layout_detector"]))
        full_evidence_by_path[path] = base + classical_cv

    log("=" * 90)
    log("gemma_primary and weighted_fusion - MUST match the original 'WITHOUT classical_cv' "
        "numbers exactly (94.0% / 111 auto_accept, 92.8% / 305 auto_accept), since this "
        "evidence is never included for these policies now.")
    log("=" * 90)
    for policy_name in ["gemma_primary", "weighted_fusion"]:
        results = [decide(base_evidence_by_path[p], config, policy=policy_name, image_id=p) for p, _ in test_items]
        correct = sum(1 for (p, gt), r in zip(test_items, results) if r.selected_class == gt)
        state_counts = Counter(r.trust_state.value for r in results)
        log(f"\n  policy={policy_name:<20s} acc={correct}/{len(results)} ({100*correct/len(results):.1f}%)  "
            f"trust_states={dict(state_counts)}")

    log("\n\n" + "=" * 90)
    log("class_specific_authority - should reproduce the earlier 'WITH classical_cv' "
        "result (310/332, dense_tabular_rows 95/97), since the new supporting sensors are "
        "now real config for this category's rule.")
    log("=" * 90)
    results = [decide(full_evidence_by_path[p], config, policy="class_specific_authority", image_id=p)
               for p, _ in test_items]
    correct = sum(1 for (p, gt), r in zip(test_items, results) if r.selected_class == gt)
    state_counts = Counter(r.trust_state.value for r in results)
    log(f"\n  policy=class_specific_authority acc={correct}/{len(results)} ({100*correct/len(results):.1f}%)  "
        f"trust_states={dict(state_counts)}")

    dtr_paths = [p for p, gt in test_items if gt == "dense_tabular_rows"]
    dtr_correct = sum(
        1 for p in dtr_paths
        if decide(full_evidence_by_path[p], config, policy="class_specific_authority", image_id=p).selected_class
        == "dense_tabular_rows"
    )
    log(f"  dense_tabular_rows-specific: {dtr_correct}/{len(dtr_paths)} correctly routed")

    # Confirm OTHER categories' class_specific_authority rules are
    # completely unaffected (they don't reference table_confidence/
    # layout_detector, but worth verifying directly rather than assuming).
    log("\n  Per-category accuracy under class_specific_authority (confirms other "
        "categories' rules are unaffected by this change):")
    by_cat_totals = Counter(gt for _, gt in test_items)
    by_cat_correct = Counter()
    for (p, gt), r in zip(test_items, results):
        if r.selected_class == gt:
            by_cat_correct[gt] += 1
    for cat in sorted(by_cat_totals):
        log(f"    {cat:<20s} {by_cat_correct[cat]}/{by_cat_totals[cat]}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(f"\n\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
