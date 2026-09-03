"""
Phase 3 of the Stage-4 error-analysis pipeline (2026-08-08): statistics
from CACHED predictions only (Phase 1's review_table.csv +
probe_softmax_probs.pt) - no inference, no new model calls, pure
aggregation/arithmetic over data already on disk.

Per location, per split (val/test - train is in-sample and reported
separately, not mixed into the same accuracy/calibration numbers):
    - confusion matrix
    - per-class precision/recall/F1
    - confidence histogram (binned)
    - calibration summary (accuracy within each confidence bin)
    - confidence-vs-correctness summary stats
    - class imbalance stats (support per class)

Output: one JSON per location under data/outputs/error_analysis/
phase3_prediction_stats/<location>.json, plus a combined summary CSV.

Usage:
    python diagnostics/error_analysis/phase3_prediction_stats.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics.error_analysis.common import LOCATIONS, OUT_DIR

REVIEW_TABLE_PATH = OUT_DIR / "review_table.csv"
STATS_DIR = OUT_DIR / "phase3_prediction_stats"
SUMMARY_CSV_PATH = OUT_DIR / "phase3_summary.csv"

CONFIDENCE_BINS = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 0.99), (0.99, 1.001)]


def confusion_matrix(rows: list[dict], classes: list[str], loc: str) -> dict:
    cm = {gt: {pred: 0 for pred in classes} for gt in classes}
    for r in rows:
        cm[r["gt"]][r[f"pred_{loc}"]] += 1
    return cm


def precision_recall_f1(cm: dict, classes: list[str]) -> dict:
    result = {}
    for c in classes:
        tp = cm[c][c]
        fp = sum(cm[gt][c] for gt in classes if gt != c)
        fn = sum(cm[c][pred] for pred in classes if pred != c)
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = (2 * precision * recall / (precision + recall)
              if precision is not None and recall is not None and (precision + recall) > 0 else None)
        support = sum(cm[c].values())
        result[c] = {
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
            "support": support,
        }
    return result


def confidence_histogram(confs: list[float]) -> dict:
    hist = {f"{lo:.2f}-{hi:.2f}": 0 for lo, hi in CONFIDENCE_BINS}
    for c in confs:
        for lo, hi in CONFIDENCE_BINS:
            if lo <= c < hi:
                hist[f"{lo:.2f}-{hi:.2f}"] += 1
                break
    return hist


def calibration_summary(rows: list[dict], loc: str) -> dict:
    """Accuracy within each confidence bin - a real calibration measurement
    (bucketed reliability), not just the mean-confidence-correct-vs-wrong
    number already reported in the Stage 1/2 scripts. A well-calibrated
    probe should show ~90% bin accuracy in the 0.90-0.95 bin, etc.; a
    consistently-overconfident probe shows bin accuracy well below the
    bin's own range across the board."""
    bins = {f"{lo:.2f}-{hi:.2f}": {"n": 0, "n_correct": 0} for lo, hi in CONFIDENCE_BINS}
    for r in rows:
        conf = float(r[f"conf_{loc}"])
        correct = r[f"correct_{loc}"] == "True"
        for lo, hi in CONFIDENCE_BINS:
            if lo <= conf < hi:
                key = f"{lo:.2f}-{hi:.2f}"
                bins[key]["n"] += 1
                if correct:
                    bins[key]["n_correct"] += 1
                break
    out = {}
    for key, v in bins.items():
        out[key] = {
            "n": v["n"],
            "accuracy": round(v["n_correct"] / v["n"], 4) if v["n"] > 0 else None,
        }
    return out


def confidence_vs_correctness(rows: list[dict], loc: str) -> dict:
    correct_confs = [float(r[f"conf_{loc}"]) for r in rows if r[f"correct_{loc}"] == "True"]
    wrong_confs = [float(r[f"conf_{loc}"]) for r in rows if r[f"correct_{loc}"] == "False"]

    def _stats(vals):
        if not vals:
            return {"n": 0, "mean": None, "min": None, "max": None}
        return {"n": len(vals), "mean": round(sum(vals) / len(vals), 4), "min": round(min(vals), 4), "max": round(max(vals), 4)}

    return {"correct": _stats(correct_confs), "incorrect": _stats(wrong_confs)}


def class_imbalance_stats(rows: list[dict], classes: list[str]) -> dict:
    counts = Counter(r["gt"] for r in rows)
    total = len(rows)
    return {
        c: {"n": counts.get(c, 0), "pct": round(100 * counts.get(c, 0) / total, 2) if total else None}
        for c in classes
    }


def main():
    if not REVIEW_TABLE_PATH.exists():
        raise FileNotFoundError(f"{REVIEW_TABLE_PATH} not found - run phase1_build_review_dataset.py first.")

    with open(REVIEW_TABLE_PATH, encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    classes = sorted(set(r["gt"] for r in all_rows))
    STATS_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for split in ("val", "test", "train"):
        split_rows = [r for r in all_rows if r["split"] == split]
        for loc in LOCATIONS:
            cm = confusion_matrix(split_rows, classes, loc)
            prf = precision_recall_f1(cm, classes)
            confs = [float(r[f"conf_{loc}"]) for r in split_rows]
            hist = confidence_histogram(confs)
            calib = calibration_summary(split_rows, loc)
            conf_vs_correct = confidence_vs_correctness(split_rows, loc)
            imbalance = class_imbalance_stats(split_rows, classes)

            n_correct = sum(1 for r in split_rows if r[f"correct_{loc}"] == "True")
            accuracy = n_correct / len(split_rows) if split_rows else None

            payload = {
                "split": split, "location": loc, "n": len(split_rows),
                "accuracy": round(accuracy, 4) if accuracy is not None else None,
                "confusion_matrix": cm,
                "precision_recall_f1": prf,
                "confidence_histogram": hist,
                "calibration_summary": calib,
                "confidence_vs_correctness": conf_vs_correct,
                "class_imbalance": imbalance,
                "note": ("train-split stats are IN-SAMPLE (the probe was trained on this data) - "
                         "do not compare train accuracy/calibration to val/test as if held-out.") if split == "train" else "",
            }
            out_path = STATS_DIR / f"{split}_{loc}.json"
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            macro_f1_vals = [v["f1"] for v in prf.values() if v["f1"] is not None]
            macro_f1 = sum(macro_f1_vals) / len(macro_f1_vals) if macro_f1_vals else None
            summary_rows.append({
                "split": split, "location": loc, "n": len(split_rows),
                "accuracy": round(accuracy, 4) if accuracy is not None else None,
                "macro_f1": round(macro_f1, 4) if macro_f1 is not None else None,
                "mean_conf_correct": conf_vs_correct["correct"]["mean"],
                "mean_conf_incorrect": conf_vs_correct["incorrect"]["mean"],
            })
            print(f"{split:>5s} {loc:<18s} n={len(split_rows):4d} acc={payload['accuracy']} macro_f1={round(macro_f1,4) if macro_f1 else None}")

    with open(SUMMARY_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nPer-(split,location) detail JSONs written to {STATS_DIR}/")
    print(f"Summary CSV written to {SUMMARY_CSV_PATH}")


if __name__ == "__main__":
    main()
