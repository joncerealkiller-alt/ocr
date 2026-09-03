"""
Analysis pass for the multi-architecture sensor-qualification benchmark
(diagnostics/run_vit_family_benchmark.py), per Jon's explicit framing:
the goal is NOT highest accuracy, it's identifying COMPLEMENTARY
archive-trained sensors for the evidence-fusion engine. Two models with
similar accuracy but different confusion matrices are more valuable
together than two models that make identical mistakes.

Builds:
  1. The comparison table Jon asked for (zero-shot acc, fine-tuned acc,
     absolute improvement, dense_tabular_rows improvement,
     printed_document improvement, other notable class deltas) - vit21k's
     already-known numbers are folded in alongside the 6 newly benchmarked
     architectures, not re-run.
  2. Pairwise prediction AGREEMENT matrix on the held-out test set -
     for every pair of fine-tuned models, what fraction of test images
     did they predict the SAME label on (right or wrong, agreement is
     about correlation, not correctness).
  3. Pairwise ERROR OVERLAP - of the images where model A was wrong, what
     fraction did model B ALSO get wrong (and specifically, wrong the
     SAME way - same incorrect predicted label)? High error overlap =
     redundant sensors; low overlap = complementary, valuable to fusion.
  4. A per-model "unique correct" count - how many held-out images did
     THIS model get right that every other model got wrong? A model with
     modest overall accuracy but many unique-correct cases is more
     valuable to a fusion ensemble than raw accuracy suggests.

Requires diagnostics/run_vit_family_benchmark.py to have completed and
written data/outputs/vit_family_benchmark/<arch>/<arch>_test_predictions.json
for each architecture, plus the existing vit21k held-out predictions
(reconstructed from the earlier holdout run if not already saved as
per-file JSON - see VIT21K_* constants below).

Usage:
    python diagnostics/analyze_vit_family_benchmark.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BENCHMARK_ROOT = PROJECT_ROOT / "data" / "outputs" / "vit_family_benchmark"
REPORT_PATH = PROJECT_ROOT / "data" / "logs" / "reviewed" / "vit_family_benchmark_analysis.txt"

# vit21k's already-established numbers (from the earlier held-out run +
# ablation), folded into the comparison table without re-running -
# zero-shot number is from the EARLIER standalone zero-shot test
# (different split/manifest than this benchmark's train/test partition,
# noted explicitly in the report rather than silently treated as
# equivalent).
VIT21K_SUMMARY = {
    "name": "vit21k", "timm_tag": "vit_base_patch16_224.orig_in21k", "family": "vit21k",
    "zeroshot_acc": 0.544,  # from test_vit_base21k_classification.py - DIFFERENT split, noted
    "zeroshot_per_class": {
        "dense_tabular_rows": [11, 30], "handwritten_ledger": [11, 13], "map_land_record": [11, 30],
        "mixed_text_image": [4, 6], "portrait_photo": [4, 30], "printed_document": [21, 30],
        "website_screenshot": [30, 30], "genealogy_chart": [0, 0],
    },
    "finetuned_test_acc": 0.898,  # from train_vit21k_document_classifier_holdout.py, SAME split as this benchmark
    "finetuned_per_class": {
        "dense_tabular_rows": [90, 97], "genealogy_chart": [1, 1], "handwritten_ledger": [3, 4],
        "map_land_record": [14, 14], "mixed_text_image": [2, 3], "portrait_photo": [9, 12],
        "printed_document": [153, 175], "website_screenshot": [26, 26],
    },
}


def load_benchmark_results() -> list[dict]:
    summary_path = BENCHMARK_ROOT / "all_results_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"{summary_path} not found - run diagnostics/run_vit_family_benchmark.py first")
    results = json.loads(summary_path.read_text(encoding="utf-8"))
    results.append(VIT21K_SUMMARY)
    return results


def load_predictions(name: str) -> dict[str, dict] | None:
    if name == "vit21k":
        # vit21k's original holdout run didn't save per-file JSON predictions -
        # not reconstructable without re-running it, so it's excluded from the
        # pairwise agreement/overlap analysis (accuracy-only comparison still
        # includes it via VIT21K_SUMMARY above).
        return None
    path = BENCHMARK_ROOT / name / f"{name}_test_predictions.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_comparison_table(results: list[dict], lines: list[str]):
    def log(s=""):
        lines.append(s)
        print(s)

    log("=" * 100)
    log("COMPARISON TABLE - zero-shot vs. fine-tuned, per architecture")
    log("=" * 100)
    log(f"{'Model':<14s} {'ZeroShot':>9s} {'FineTuned':>10s} {'AbsImprove':>11s} "
        f"{'DTR_imp':>9s} {'PrintDoc_imp':>13s} {'Params(M)':>10s} {'Speed(img/s)':>13s} {'TrainTime(s)':>13s}")
    for r in results:
        zs = r["zeroshot_acc"]
        ft = r["finetuned_test_acc"]
        imp = ft - zs
        zs_pc = r["zeroshot_per_class"]
        ft_pc = r["finetuned_per_class"]

        def cat_rate(pc, cat):
            v = pc.get(cat)
            if not v or v[1] == 0:
                return None
            return v[0] / v[1]

        dtr_zs, dtr_ft = cat_rate(zs_pc, "dense_tabular_rows"), cat_rate(ft_pc, "dense_tabular_rows")
        dtr_imp = (dtr_ft - dtr_zs) if (dtr_zs is not None and dtr_ft is not None) else None
        pd_zs, pd_ft = cat_rate(zs_pc, "printed_document"), cat_rate(ft_pc, "printed_document")
        pd_imp = (pd_ft - pd_zs) if (pd_zs is not None and pd_ft is not None) else None

        params = r.get("params_total_M", float("nan"))
        speed = r.get("inference_images_per_sec", float("nan"))
        ttime = r.get("training_time_sec", float("nan"))

        log(f"{r['name']:<14s} {zs:>9.3f} {ft:>10.3f} {imp:>+11.3f} "
            f"{('%+.3f' % dtr_imp) if dtr_imp is not None else 'n/a':>9s} "
            f"{('%+.3f' % pd_imp) if pd_imp is not None else 'n/a':>13s} "
            f"{params:>10.1f} {speed:>13.1f} {ttime:>13.1f}")

    log("\nNote: vit21k's zero-shot number (54.4%) is from an earlier standalone "
        "test with a DIFFERENT train/test partition than this benchmark's shared "
        "split - not perfectly apples-to-apples with the other 6 models' zero-shot "
        "numbers (which all share this benchmark's exact partition), though the "
        "magnitude/direction is consistent with the other ImageNet-style zero-shot "
        "results. vit21k's fine-tuned number (89.8%) DOES use the same partition.")

    log("\n\nOther notable per-class deltas (categories with n>=10 in test, "
        "excluding dense_tabular_rows/printed_document already shown above):")
    for r in results:
        zs_pc, ft_pc = r["zeroshot_per_class"], r["finetuned_per_class"]
        notes = []
        for cat in ("map_land_record", "portrait_photo", "website_screenshot"):
            zsv, ftv = zs_pc.get(cat), ft_pc.get(cat)
            if zsv and ftv and zsv[1] >= 10 and ftv[1] >= 10:
                d = (ftv[0] / ftv[1]) - (zsv[0] / zsv[1])
                notes.append(f"{cat}={d:+.3f}")
        log(f"  {r['name']:<14s} " + "  ".join(notes))


def build_agreement_matrices(results: list[dict], lines: list[str]):
    def log(s=""):
        lines.append(s)
        print(s)

    names = [r["name"] for r in results]
    preds = {name: load_predictions(name) for name in names}
    available = [n for n in names if preds[n] is not None]
    if len(available) < 2:
        log("\nNot enough per-file prediction sets available for pairwise analysis.")
        return

    # common file set across all available models
    common_paths = set(preds[available[0]].keys())
    for n in available[1:]:
        common_paths &= set(preds[n].keys())
    common_paths = sorted(common_paths)
    log(f"\n\n{'='*100}\nPAIRWISE AGREEMENT / ERROR OVERLAP  (n={len(common_paths)} common held-out images, "
        f"models: {available})\n{'='*100}")

    log("\n-- Prediction AGREEMENT matrix (fraction of images both models predicted the SAME label, "
        "right or wrong) --")
    header = "".ljust(14) + "".join(n[:12].ljust(14) for n in available)
    log(header)
    for a in available:
        row = a.ljust(14)
        for b in available:
            if a == b:
                row += "-".ljust(14)
                continue
            agree = sum(1 for p in common_paths if preds[a][p]["pred"] == preds[b][p]["pred"])
            row += f"{agree/len(common_paths):.3f}".ljust(14)
        log(row)

    log("\n-- ERROR OVERLAP matrix: of images where model A (row) was WRONG, fraction where "
        "model B (col) was ALSO wrong (any error, not necessarily the same wrong label) --")
    log(header)
    for a in available:
        a_wrong = [p for p in common_paths if preds[a][p]["pred"] != preds[a][p]["gt"]]
        row = a.ljust(14)
        for b in available:
            if a == b or not a_wrong:
                row += ("-" if a == b else "n/a").ljust(14)
                continue
            both_wrong = sum(1 for p in a_wrong if preds[b][p]["pred"] != preds[b][p]["gt"])
            row += f"{both_wrong/len(a_wrong):.3f}".ljust(14)
        log(row)
    log("(Low values = when this model is wrong, others are usually still right - "
        "COMPLEMENTARY. High values = models fail together - REDUNDANT.)")

    log("\n-- SAME-WRONG-LABEL overlap: of images where model A (row) was wrong, fraction where "
        "model B (col) predicted the EXACT SAME (also wrong) label --")
    log(header)
    for a in available:
        a_wrong = [p for p in common_paths if preds[a][p]["pred"] != preds[a][p]["gt"]]
        row = a.ljust(14)
        for b in available:
            if a == b or not a_wrong:
                row += ("-" if a == b else "n/a").ljust(14)
                continue
            same_wrong = sum(1 for p in a_wrong
                              if preds[b][p]["pred"] != preds[b][p]["gt"]
                              and preds[b][p]["pred"] == preds[a][p]["pred"])
            row += f"{same_wrong/len(a_wrong):.3f}".ljust(14)
        log(row)

    log("\n-- Per-model UNIQUE-CORRECT count: held-out images this model got right that EVERY "
        "other available model got wrong (a modest-accuracy model with many unique-correct "
        "cases is more valuable to a fusion ensemble than raw accuracy alone suggests) --")
    for a in available:
        unique_correct = 0
        for p in common_paths:
            if preds[a][p]["pred"] != preds[a][p]["gt"]:
                continue
            if all(preds[b][p]["pred"] != preds[b][p]["gt"] for b in available if b != a):
                unique_correct += 1
        log(f"  {a:<14s} {unique_correct}")

    log("\n-- Full agreement/error-overlap disagreement set size per pair "
        "(images where predictions differ) --")
    for i, a in enumerate(available):
        for b in available[i + 1:]:
            disagree = sum(1 for p in common_paths if preds[a][p]["pred"] != preds[b][p]["pred"])
            log(f"  {a} vs {b}: {disagree}/{len(common_paths)} disagree "
                f"({100*disagree/len(common_paths):.1f}%)")


def main():
    lines = []
    def log(s=""):
        lines.append(s)
        print(s)

    results = load_benchmark_results()
    log(f"Loaded {len(results)} model results: {[r['name'] for r in results]}\n")

    build_comparison_table(results, lines)
    build_agreement_matrices(results, lines)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(f"\n\nReport written: {REPORT_PATH}")


if __name__ == "__main__":
    main()
