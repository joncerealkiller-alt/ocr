"""
Runs the live Gemma decision-token logit-margin adapter
(core/gemma_logit_margin_adapter.py) against the SAME 332-image held-out
split as every other model in this benchmark, using the SAME flat-8
production prompt as diagnostics/run_gemma_flat8_on_benchmark_holdout.py
(the fairer of the two recorded Gemma runs - see docs/MULTI_SOURCE_
VOTING_CLASSIFIER_PROPOSAL.md's "Flat 8-bucket Gemma re-run" section).

Unlike the earlier flat-8 run, this one captures a REAL raw decision-
token logit margin per image (via output_scores=True), not just the
self-reported confidence field - built specifically so
core/routing_decision_engine.py's gemma_primary policy can finally be
evaluated fairly against vision_primary/weighted_fusion using real
uncertainty evidence on BOTH sides (see diagnostics/replay_decision_
engine.py's honest gap: without this, gemma_primary could never reach
AUTO_ACCEPT).

Usage:
    python diagnostics/run_gemma_logit_adapter_on_benchmark_holdout.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from PIL import Image

from core.classifier import build_classifier_loader
from core.gemma_logit_margin_adapter import classify_with_logit_margin
from diagnostics.run_gemma_flat8_on_benchmark_holdout import build_flat8_prompt
from diagnostics.vit_family_benchmark_common import get_split

OUT_DIR = PROJECT_ROOT / "data" / "outputs" / "vit_family_benchmark" / "gemma_flat8_logit_margin"
REPORT_PATH = PROJECT_ROOT / "data" / "logs" / "reviewed" / "gemma_flat8_logit_margin_report.txt"
PREDICTIONS_PATH = OUT_DIR / "gemma_flat8_logit_margin_test_predictions.json"


def main():
    lines = []
    def log(s=""):
        lines.append(s)
        print(s, flush=True)

    log("Loading the SAME held-out split used by every other model in this benchmark...")
    by_cat, train_items, val_items, test_items = get_split()
    classes = sorted(by_cat.keys())
    log(f"Held-out test set: {len(test_items)} images\n")

    pipeline_cfg = yaml.safe_load((PROJECT_ROOT / "config" / "pipeline.yaml").read_text(encoding="utf-8"))
    log("Loading Gemma via the real production build_classifier_loader() path, "
        "flat-8 prompt, output_scores enabled...")
    loader = build_classifier_loader(pipeline_cfg, debug=False)
    loader.config.prompt_text = build_flat8_prompt()
    log("Loaded.\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    correct_n = 0
    margin_found_n = 0
    start = time.perf_counter()

    for i, (path, gt) in enumerate(test_items, 1):
        try:
            img = Image.open(path).convert("RGB")
            classification, evidence = classify_with_logit_margin(loader, path, img)
            pred = classification.category.value
            correct = pred == gt
            if correct:
                correct_n += 1
            if evidence.raw_score is not None:
                margin_found_n += 1
            results[path] = {
                "gt": gt, "pred": pred, "correct": correct,
                "confidence": classification.confidence,
                "raw_logit_margin": evidence.raw_score,
                "score_kind": evidence.metadata.get("score_kind"),
                "decision_token_index": evidence.metadata.get("decision_token_index"),
                "decision_token_top1_text": evidence.metadata.get("decision_token_top1_text"),
                "decision_token_top2_text": evidence.metadata.get("decision_token_top2_text"),
                "decision_token_top1_prob": evidence.metadata.get("decision_token_top1_prob"),
            }
            mark = "OK  " if correct else "WRONG"
            margin_str = f"{evidence.raw_score:.2f}" if evidence.raw_score is not None else "N/A"
            log(f"[{i}/{len(test_items)}] {mark} gt={gt:<20s} pred={pred:<20s} margin={margin_str}")
        except Exception as e:
            log(f"[{i}/{len(test_items)}] FAILED {path}: {type(e).__name__}: {e}")
            results[path] = {"gt": gt, "pred": None, "correct": False, "error": str(e)}

        if i % 25 == 0:
            PREDICTIONS_PATH.write_text(json.dumps({"classes": classes, "predictions": results}, indent=2),
                                          encoding="utf-8")
            REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
            REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    elapsed = time.perf_counter() - start
    n = len(test_items)
    log(f"\n\n=== GEMMA FLAT-8 + LOGIT MARGIN HELD-OUT ACCURACY: {correct_n}/{n} "
        f"({100*correct_n/n:.1f}%) ===")
    log(f"Decision token located (real margin captured): {margin_found_n}/{n} "
        f"({100*margin_found_n/n:.1f}%)")
    log(f"Total inference time: {elapsed:.1f}s ({elapsed/n:.2f}s/image)")

    correct_margins = [r["raw_logit_margin"] for r in results.values()
                        if r.get("correct") and r.get("raw_logit_margin") is not None]
    wrong_margins = [r["raw_logit_margin"] for r in results.values()
                      if not r.get("correct") and r.get("raw_logit_margin") is not None]
    import statistics
    if correct_margins:
        log(f"\nMargin when correct (n={len(correct_margins)}): "
            f"mean={statistics.mean(correct_margins):.2f}  median={statistics.median(correct_margins):.2f}")
    if wrong_margins:
        log(f"Margin when WRONG (n={len(wrong_margins)}): "
            f"mean={statistics.mean(wrong_margins):.2f}  median={statistics.median(wrong_margins):.2f}")

    PREDICTIONS_PATH.write_text(json.dumps({"classes": classes, "predictions": results}, indent=2),
                                  encoding="utf-8")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(f"\nPredictions: {PREDICTIONS_PATH}")
    log(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
