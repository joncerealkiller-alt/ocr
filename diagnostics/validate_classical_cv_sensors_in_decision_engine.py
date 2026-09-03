"""
Live validation of the new core/sensor_adapters.py adapters
(table_confidence, layout_detector) against the Decision Engine - the
"wire it up and check it actually helps" step, per this project's own
Hypothesis -> Experiment -> Benchmark -> Evidence -> Production
discipline (no rule gets promoted on a guess).

Runs BOTH classical-CV sensors LIVE (real analyze_image() call, real
DocLayout-YOLO forward pass) on the same 332-image flat-8 held-out test
split diagnostics/replay_decision_engine_v2_real_gemma_margin.py already
has recorded Gemma-logit-margin + 7-tower evidence for - so the ONLY
thing that changes between the "before" and "after" pass is whether the
two new classical_cv EvidenceRecords are included, nothing else.

Reports: how often each new sensor actually fires (produces usable
evidence, not None), trust-state distribution before/after, accuracy
before/after per policy, and specifically whether dense_tabular_rows
outcomes improve - that's the category both new adapters target and the
category this session's own probe-vs-generation work found weakest.

Usage:
    python diagnostics/validate_classical_cv_sensors_in_decision_engine.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from core.image_analysis import analyze_image
from core.layout_detector import build_layout_model, detect_layout
from core.sensor_adapters import table_confidence_evidence, layout_detector_evidence
from core.routing_decision_engine import (
    EvidenceRecord, TrustState, decide, load_decision_config,
)
from diagnostics.vit_family_benchmark_common import BENCHMARK_ROOT, get_split
from diagnostics.error_analysis.common import OLD_WORKING_PREFIX, LEGACY_WORKING_IMAGES_DIR

REPORT_PATH = PROJECT_ROOT / "data" / "logs" / "reviewed" / "classical_cv_sensor_validation_report.txt"

TOWER_NAMES = ["convnext", "mobilenetv2", "dinov2", "beit", "swin", "siglip", "vit21k"]
TOWER_FAMILY = {
    "convnext": "cnn_modern", "mobilenetv2": "cnn_lightweight_independent",
    "dinov2": "vit_isotropic_strong_cluster", "beit": "vit_isotropic_strong_cluster",
    "siglip": "vit_isotropic_strong_cluster", "vit21k": "vit_isotropic_strong_cluster",
    "swin": "hierarchical_transformer",
}


def resolve_path(path_str: str) -> str:
    """Same legacy-corpus remap as diagnostics/gemma_hidden_state_latency_
    comparison.py - the underlying split CSV still stores pre-migration
    data\\working\\... paths."""
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


def build_recorded_evidence(path: str, gemma_data: dict, tower_data: dict) -> list[EvidenceRecord]:
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


def main():
    lines = []
    def log(s=""):
        lines.append(s)
        print(s, flush=True)

    log("Live validation: core/sensor_adapters.py's table_confidence + layout_detector "
        "adapters, fed into the Decision Engine alongside already-recorded Gemma + 7-tower evidence.\n")

    gemma_data, tower_data = load_recorded_evidence_sources()
    by_cat, train_items, val_items, test_items = get_split()
    log(f"Held-out test set: {len(test_items)} images\n")

    log("Loading DocLayout-YOLO layout model (real inference, once)...")
    layout_model = build_layout_model()
    log("Loaded.\n")

    config = load_decision_config()

    n_table_conf_fired, n_layout_fired = 0, 0
    per_image_evidence = {}  # path -> (base_evidence, classical_cv_evidence)

    log("Running classical-CV sensors on every test image (real analyze_image() + real YOLO forward pass)...")
    for i, (path, gt) in enumerate(test_items, 1):
        resolved = resolve_path(path)
        try:
            img = Image.open(resolved).convert("RGB")
        except Exception as e:
            log(f"  [{i}/{len(test_items)}] FAILED to open {resolved}: {e}")
            continue

        analysis = analyze_image(img)
        tc_evidence = table_confidence_evidence(analysis)
        if tc_evidence is not None:
            n_table_conf_fired += 1

        detections = detect_layout(layout_model, img)
        ld_evidence = layout_detector_evidence(detections)
        if ld_evidence is not None:
            n_layout_fired += 1

        classical_cv = [e for e in (tc_evidence, ld_evidence) if e is not None]
        base = build_recorded_evidence(path, gemma_data, tower_data)
        per_image_evidence[path] = (base, classical_cv)

        if i % 50 == 0:
            log(f"  [{i}/{len(test_items)}] processed "
                f"(table_confidence fired {n_table_conf_fired}x, layout_detector fired {n_layout_fired}x so far)")

    n_processed = len(per_image_evidence)
    log(f"\nProcessed {n_processed}/{len(test_items)} images.")
    log(f"table_confidence produced usable evidence on {n_table_conf_fired}/{n_processed} "
        f"({100*n_table_conf_fired/n_processed:.1f}%)")
    log(f"layout_detector produced usable evidence on {n_layout_fired}/{n_processed} "
        f"({100*n_layout_fired/n_processed:.1f}%)\n")

    gt_by_path = dict(test_items)

    log("=" * 90)
    log("BEFORE vs. AFTER comparison, per policy")
    log("=" * 90)
    for policy_name in ["gemma_primary", "weighted_fusion", "class_specific_authority"]:
        for variant, include_cv in [("WITHOUT classical_cv", False), ("WITH classical_cv", True)]:
            results = []
            for path, (base, classical_cv) in per_image_evidence.items():
                evidence = base + (classical_cv if include_cv else [])
                result = decide(evidence, config, policy=policy_name, image_id=path)
                results.append((path, gt_by_path[path], result))

            correct = sum(1 for _, gt, r in results if r.selected_class == gt)
            state_counts = Counter(r.trust_state.value for _, _, r in results)
            log(f"\n  policy={policy_name:<25s} {variant:<22s} "
                f"acc={correct}/{len(results)} ({100*correct/len(results):.1f}%)  "
                f"trust_states={dict(state_counts)}")

    # --- dense_tabular_rows-specific breakdown - the category both new sensors target ---
    log("\n\n" + "=" * 90)
    log("dense_tabular_rows-SPECIFIC: did the new sensors change anything for this category?")
    log("=" * 90)
    dtr_paths = [p for p, (b, c) in per_image_evidence.items() if gt_by_path[p] == "dense_tabular_rows"]
    log(f"{len(dtr_paths)} dense_tabular_rows images in the test set.\n")

    for policy_name in ["gemma_primary", "weighted_fusion", "class_specific_authority"]:
        for variant, include_cv in [("WITHOUT", False), ("WITH", True)]:
            correct, changed_trust = 0, 0
            for path in dtr_paths:
                base, classical_cv = per_image_evidence[path]
                evidence = base + (classical_cv if include_cv else [])
                result = decide(evidence, config, policy=policy_name, image_id=path)
                if result.selected_class == "dense_tabular_rows":
                    correct += 1
            log(f"  policy={policy_name:<25s} {variant:<8s} classical_cv: "
                f"{correct}/{len(dtr_paths)} correctly routed to dense_tabular_rows")

    # A few real examples where classical_cv evidence actually fired and
    # disagreed/agreed with the primary - concrete audit-trail illustration.
    log("\n\nExample audit trails where classical_cv evidence fired:")
    shown = 0
    for path, (base, classical_cv) in per_image_evidence.items():
        if not classical_cv or shown >= 3:
            continue
        evidence = base + classical_cv
        result = decide(evidence, config, policy="gemma_primary", image_id=path)
        log(f"\nImage: {path}")
        log(f"Ground truth: {gt_by_path[path]}")
        for line in result.audit_trail:
            log(f"  {line}")
        shown += 1

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(f"\n\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
