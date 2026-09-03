"""
Quick quality pass over the 2026-08-05 full-corpus layout-detection
backfill (data/baseline_embeddings.json's "layout_detections" key,
1750/1750 images, core/layout_detector.py / DocLayout-YOLO). Checks how
common the "spurious full-page box" failure mode is (flagged from a
single-image smoke test in docs/CODE_MAP.md, not yet measured at scale)
- a detection whose bbox covers most of the image is very unlikely to be
a real title/table/figure region and more likely a degenerate detection
on this corpus's atypical (dense, aged, non-academic-document) content.

Read-only: joins layout_detections against data/outputs/image_analysis/
analysis_report.csv's width/height columns (already captured, avoids
re-opening 1750 images just to read dimensions).

Usage:
    python -m benchmark.layout_detection_quality_check
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = PROJECT_ROOT / "data" / "baseline_embeddings.json"
ANALYSIS_REPORT = PROJECT_ROOT / "data" / "outputs" / "image_analysis" / "analysis_report.csv"

SPURIOUS_AREA_FRAC_THRESHOLD = 0.70  # bbox covers >=70% of the image area


def main() -> None:
    with open(ANALYSIS_REPORT, newline="", encoding="utf-8") as f:
        dims = {}
        for row in csv.DictReader(f):
            try:
                dims[row["file_path"]] = (float(row["width"]), float(row["height"]))
            except (ValueError, KeyError):
                continue
    print(f"Loaded dimensions for {len(dims)} images from analysis_report.csv")

    records = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(records)} baseline records")

    n_with_layout = 0
    n_missing_dims = 0
    total_detections = 0
    spurious_detections = 0
    images_with_any_spurious = 0
    images_with_zero_detections = 0
    class_counts = Counter()
    spurious_class_counts = Counter()
    area_frac_samples = []

    for rec in records:
        ld = rec.get("layout_detections", {}).get("doclayout_yolo")
        if not ld:
            continue
        n_with_layout += 1
        image_path = rec["image"]
        wh = dims.get(image_path)
        if wh is None:
            n_missing_dims += 1
            continue
        width, height = wh
        img_area = width * height
        detections = ld.get("detections", [])
        if not detections:
            images_with_zero_detections += 1
            continue

        has_spurious = False
        for det in detections:
            total_detections += 1
            class_counts[det["class_name"]] += 1
            x0, y0, x1, y1 = det["bbox_xyxy"]
            box_area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
            frac = box_area / img_area if img_area > 0 else 0.0
            area_frac_samples.append(frac)
            if frac >= SPURIOUS_AREA_FRAC_THRESHOLD:
                spurious_detections += 1
                spurious_class_counts[det["class_name"]] += 1
                has_spurious = True
        if has_spurious:
            images_with_any_spurious += 1

    print(f"\nImages with layout_detections: {n_with_layout}")
    print(f"Images missing dimension lookup: {n_missing_dims}")
    print(f"Images with zero detections: {images_with_zero_detections}")
    print(f"Total detections: {total_detections}")

    print(f"\n=== Spurious full-page box check (bbox area >= {SPURIOUS_AREA_FRAC_THRESHOLD:.0%} of image) ===")
    print(f"Spurious detections: {spurious_detections}/{total_detections} "
          f"({100*spurious_detections/total_detections:.1f}% of all detections)" if total_detections else "n/a")
    print(f"Images with >=1 spurious detection: {images_with_any_spurious}/{n_with_layout} "
          f"({100*images_with_any_spurious/n_with_layout:.1f}%)")

    print("\nClass distribution, ALL detections:")
    for cls, n in class_counts.most_common():
        print(f"  {cls:<20} {n}")

    print("\nClass distribution, SPURIOUS (>=70% area) detections only:")
    for cls, n in spurious_class_counts.most_common():
        print(f"  {cls:<20} {n}")

    if area_frac_samples:
        area_frac_samples.sort()
        n = len(area_frac_samples)
        def pct(p): return area_frac_samples[min(n - 1, int(n * p))]
        print(f"\nArea-fraction percentiles across ALL detections: "
              f"p50={pct(0.50):.3f} p75={pct(0.75):.3f} p90={pct(0.90):.3f} "
              f"p95={pct(0.95):.3f} p99={pct(0.99):.3f} max={area_frac_samples[-1]:.3f}")


if __name__ == "__main__":
    main()
