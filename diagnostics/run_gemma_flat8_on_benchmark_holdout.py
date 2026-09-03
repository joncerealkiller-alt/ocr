"""
Re-run of diagnostics/run_gemma_on_benchmark_holdout.py per Jon's
follow-up (2026-08-07): the first Gemma same-split run used the real
PRODUCTION prompt as-is, which offers 11 categories (the original 8 plus
`photo_collage`/`casual_photo`/`cemetery_photo`, added 2026-08-04 - see
docs/MULTI_SOURCE_VOTING_CLASSIFIER_PROPOSAL.md's "Gemma error-dissent
analysis" section, which found 7 of 21 "errors" were Gemma using
`casual_photo`, a category the ground truth/towers can't express at
all). This reruns with a prompt restricted to EXACTLY the 8 categories
`manifest_final.csv`'s ground truth was built against - `dense_tabular_
rows, genealogy_chart, handwritten_ledger, map_land_record, mixed_text_
image, portrait_photo, printed_document, website_screenshot` - plus
`uncertain_review` (the real classifier's standing escape valve, kept
so this isn't ALSO changing "must force an answer" behavior at the same
time as the category-set change).

Built by reusing the REAL production prompt template
(config/prompts/classifier_classify_v1.txt) and the REAL, current,
tuned classifier_guidance text for each of the 8+1 categories from
core/taxonomy.py - NOT hand-written new descriptions, NOT the older
"Main" checkout's hardcoded prompt (confirmed in the prior pass to be
missing website_screenshot entirely, and out of date relative to this
project's current taxonomy machinery) - just the same rendering
mechanism (`taxonomy.render_classifier_category_block()`-equivalent),
filtered to a smaller category set before rendering.

Usage:
    python diagnostics/run_gemma_flat8_on_benchmark_holdout.py
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
from core.taxonomy import load_taxonomy
from diagnostics.vit_family_benchmark_common import get_split

OUT_DIR = PROJECT_ROOT / "data" / "outputs" / "vit_family_benchmark" / "gemma_flat8"
REPORT_PATH = PROJECT_ROOT / "data" / "logs" / "reviewed" / "gemma_flat8_on_benchmark_holdout_report.txt"
PREDICTIONS_PATH = OUT_DIR / "gemma_flat8_test_predictions.json"
PROMPT_TEMPLATE_PATH = PROJECT_ROOT / "config" / "prompts" / "classifier_classify_v1.txt"

# The 8 categories manifest_final.csv's ground truth actually uses,
# plus uncertain_review (the real classifier's standing escape valve -
# kept so this run isolates ONLY the category-set change, not also
# removing the "allowed to punt" option at the same time).
ALLOWED_CATEGORY_IDS = {
    "dense_tabular_rows", "genealogy_chart", "handwritten_ledger",
    "map_land_record", "mixed_text_image", "portrait_photo",
    "printed_document", "website_screenshot", "uncertain_review",
}


def build_flat8_prompt() -> str:
    taxonomy = load_taxonomy()
    categories = [c for c in taxonomy.categories_for_classifier_prompt() if c.id in ALLOWED_CATEGORY_IDS]
    found_ids = {c.id for c in categories}
    missing = ALLOWED_CATEGORY_IDS - found_ids
    if missing:
        raise ValueError(f"Expected categories missing from taxonomy (no classifier_guidance?): {missing}")
    category_block = "\n".join(f"- {c.id}: {c.classifier_guidance}" for c in categories)

    template = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    if "{{CLASSIFIER_CATEGORY_CHOICES}}" not in template:
        raise ValueError(f"{PROMPT_TEMPLATE_PATH} has no {{{{CLASSIFIER_CATEGORY_CHOICES}}}} placeholder")
    return template.replace("{{CLASSIFIER_CATEGORY_CHOICES}}", category_block)


def main():
    lines = []
    def log(s=""):
        lines.append(s)
        print(s, flush=True)

    log("Loading the SAME held-out split used by the 7-tower benchmark and the "
        "first (11-category) Gemma run...")
    by_cat, train_items, val_items, test_items = get_split()
    classes = sorted(by_cat.keys())
    log(f"Ground-truth classes: {classes}")
    log(f"Held-out test set: {len(test_items)} images\n")

    flat8_prompt = build_flat8_prompt()
    log("Flat 8-bucket prompt built (real production template + real taxonomy "
        f"guidance text, restricted to: {sorted(ALLOWED_CATEGORY_IDS)})\n")
    log("=" * 70)
    log(flat8_prompt)
    log("=" * 70 + "\n")

    pipeline_cfg_path = PROJECT_ROOT / "config" / "pipeline.yaml"
    pipeline_cfg = yaml.safe_load(pipeline_cfg_path.read_text(encoding="utf-8"))
    log(f"Model config: {pipeline_cfg['classifier']['model']} "
        f"(prompt overridden below the loader-config level, model/loader unchanged)\n")

    log("Loading Gemma via the real production build_classifier_loader() path, "
        "then overriding its prompt_text with the flat-8 version...")
    loader = build_classifier_loader(pipeline_cfg, debug=False)
    loader.config.prompt_text = flat8_prompt
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
    log(f"\n\n=== GEMMA FLAT-8 HELD-OUT TEST ACCURACY: {correct_n}/{n} "
        f"({100*correct_n/n:.1f}%) ===")
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
