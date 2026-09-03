"""
The highest-priority follow-up experiment identified in the Decision-
Engine Sensor Complementarity Analysis (docs/MULTI_SOURCE_VOTING_
CLASSIFIER_PROPOSAL.md's "Recommendations" section, 2026-08-07): every
existing Gemma-vs-tower comparison in this project used a DIFFERENT
sample/task than the 7-model vision-tower benchmark (the 48-case
disagreement set, or a binary dense_tabular_rows gate on old-taxonomy
ground truth) - never a same-image, same-split, apples-to-apples run.
This runs Gemma on the EXACT SAME 332-image held-out test set the 7
fine-tuned vision towers were benchmarked and logit-extracted against
(diagnostics/vit_family_benchmark_common.py's get_split(), seed 42,
copied split code - identical partition).

Uses the REAL PRODUCTION Stage 5 classifier path, not a custom research
prompt: core/classifier.py's build_classifier_loader() (loads whichever
model + prompt config/pipeline.yaml's classifier section currently
points at - config/prompts/classifier_classify_v1.txt, THIS project's
version with the dynamic {{CLASSIFIER_CATEGORY_CHOICES}} taxonomy
placeholder and the website_screenshot pre-check, confirmed via diff
against the sibling "genealogy_pipeline - Main" checkout's older
hardcoded version - Jon's question "would using the original pipeline
prompt help" is answered by using exactly this, not a hand-built
research variant) and the SAME classify(file_path, raw_image) call
Stage 5 makes in a real pipeline run - not a reimplementation, so this
result is directly representative of what production Gemma actually
does today, not a best-case research variant of it.

Per-image output preserves: predicted category, self-reported
confidence, reason string, raw model output, correctness - saved to
JSON so nothing needs to be rerun for the complementarity analysis to
consume it.

Usage:
    python diagnostics/run_gemma_on_benchmark_holdout.py
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
from diagnostics.vit_family_benchmark_common import get_split

OUT_DIR = PROJECT_ROOT / "data" / "outputs" / "vit_family_benchmark" / "gemma"
REPORT_PATH = PROJECT_ROOT / "data" / "logs" / "reviewed" / "gemma_on_benchmark_holdout_report.txt"
PREDICTIONS_PATH = OUT_DIR / "gemma_test_predictions.json"


def main():
    lines = []
    def log(s=""):
        lines.append(s)
        print(s, flush=True)

    log("Loading the SAME held-out split the 7-model vision-tower benchmark used "
        "(seed=42, identical split code)...")
    by_cat, train_items, val_items, test_items = get_split()
    classes = sorted(by_cat.keys())
    log(f"Classes: {classes}")
    log(f"Held-out test set: {len(test_items)} images\n")

    pipeline_cfg_path = PROJECT_ROOT / "config" / "pipeline.yaml"
    pipeline_cfg = yaml.safe_load(pipeline_cfg_path.read_text(encoding="utf-8"))
    log(f"Production classifier config: model={pipeline_cfg['classifier']['model']}  "
        f"prompt_file={pipeline_cfg['classifier']['prompt_file']}")

    log("\nLoading Gemma via the real production build_classifier_loader() path "
        "(same loader, same prompt Stage 5 actually uses)...")
    loader = build_classifier_loader(pipeline_cfg, debug=False)
    log("Loaded.\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    correct_n = 0
    start = time.perf_counter()

    for i, (path, gt) in enumerate(test_items, 1):
        try:
            img = Image.open(path).convert("RGB")
            result = loader.classify(path, img)
            pred = result.category.value
            correct = pred == gt
            if correct:
                correct_n += 1
            results[path] = {
                "gt": gt, "pred": pred, "correct": correct,
                "confidence": result.confidence, "reason": result.reason,
                "model": result.model,
            }
            mark = "OK  " if correct else "WRONG"
            log(f"[{i}/{len(test_items)}] {mark} gt={gt:<20s} pred={pred:<20s} "
                f"conf={result.confidence:.3f}")
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
    log(f"\n\n=== GEMMA HELD-OUT TEST ACCURACY (production prompt, same split as the "
        f"7-tower benchmark): {correct_n}/{n} ({100*correct_n/n:.1f}%) ===")
    log(f"Total inference time: {elapsed:.1f}s ({elapsed/n:.2f}s/image)")

    from collections import Counter, defaultdict
    per_cat_correct, per_cat_total = Counter(), Counter()
    conf = defaultdict(Counter)
    for path, r in results.items():
        per_cat_total[r["gt"]] += 1
        if r["correct"]:
            per_cat_correct[r["gt"]] += 1
        if r.get("pred"):
            conf[r["gt"]][r["pred"]] += 1

    log("\nPer-category accuracy:")
    for c in classes:
        tot = per_cat_total.get(c, 0)
        corr = per_cat_correct.get(c, 0)
        if tot:
            log(f"  {c:<20s} {corr}/{tot} ({100*corr/tot:.1f}%)")

    log("\nConfusion matrix (rows=gt, cols=pred):")
    header = "gt\\pred".ljust(22) + "".join(c[:10].ljust(12) for c in classes)
    log(header)
    for gt_c in classes:
        row = gt_c.ljust(22) + "".join(str(conf[gt_c].get(c, 0)).ljust(12) for c in classes)
        log(row)

    PREDICTIONS_PATH.write_text(json.dumps({"classes": classes, "predictions": results}, indent=2),
                                  encoding="utf-8")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(f"\nPredictions: {PREDICTIONS_PATH}")
    log(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
