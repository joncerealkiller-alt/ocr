"""
Follow-up to scripts/triage_flagged_deskew.py and
calibrate_deskew_anomaly_threshold.py: the vision-tower similarity
signal turned out to measure semantic content, not geometric quality
(0/126 flagged images looked anomalous - DINOv2 was specifically
qualified for being robust to deskew, so it doesn't react to it by
design). This checks whether a purely deterministic, much cheaper
signal - aspect ratio versus each bucket's normal range - can flag
rotation candidates instead, per the "crop/measure before trust"
principle already established in core/image_analysis.py.

Logic: a 90-degree rotation swaps width and height, inverting the
aspect ratio. If an image's aspect ratio sits way outside its bucket's
normal range, but its RECIPROCAL aspect ratio fits comfortably inside
that range, that's a cheap, deterministic, content-independent signal
the image is very likely rotated 90 degrees (not just skewed) -
computed from PIL image size alone, no CV pass, no embeddings.

Usage:
    python -m scripts.check_deterministic_rotation_signal
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUCKET_DIR = PROJECT_ROOT / "data" / "buckets"
TRIAGE_CSV = PROJECT_ROOT / "data" / "outputs" / "flagged_deskew_triage.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "outputs" / "flagged_deskew_aspect_ratio_check.csv"

BASELINE_PER_BUCKET = 30  # cheap (just PIL .size), can afford more than the embedding-based calibration


def load_triage_rows() -> list[dict]:
    with open(TRIAGE_CSV, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["status"] == "OK"]


def build_aspect_ratio_baseline(bucket: str, exclude: set[str]) -> list[float]:
    csv_path = BUCKET_DIR / f"{bucket}.csv"
    if not csv_path.exists():
        return []
    ratios = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            p = Path(row["file_path"])
            if str(p) in exclude or not p.exists():
                continue
            try:
                w, h = Image.open(p).size
                ratios.append(w / h)
            except Exception:
                continue
            if len(ratios) >= BASELINE_PER_BUCKET:
                break
    return ratios


def main():
    triage_rows = load_triage_rows()
    flagged_paths = {r["file_path"] for r in triage_rows}

    buckets = sorted({r["bucket"] for r in triage_rows})
    baseline: dict[str, list[float]] = {}
    for bucket in buckets:
        ratios = build_aspect_ratio_baseline(bucket, flagged_paths)
        baseline[bucket] = ratios
        if ratios:
            print(f"{bucket:<20} n={len(ratios):>2}  "
                  f"median_ar={np.median(ratios):.3f}  p10={np.percentile(ratios,10):.3f}  "
                  f"p90={np.percentile(ratios,90):.3f}  range=[{min(ratios):.2f}, {max(ratios):.2f}]")
        else:
            print(f"{bucket:<20} n=0")

    results = []
    flip_flag_count = 0
    for row in triage_rows:
        bucket = row["bucket"]
        base = baseline.get(bucket, [])
        ar = float(row["det_aspect_ratio"]) if row["det_aspect_ratio"] else None
        if ar is None or not base:
            results.append({**row, "in_normal_ar_range": "", "reciprocal_fits_better": ""})
            continue

        p10, p90 = np.percentile(base, 10), np.percentile(base, 90)
        in_range = p10 <= ar <= p90
        reciprocal = 1.0 / ar
        reciprocal_in_range = p10 <= reciprocal <= p90

        flip_flag = (not in_range) and reciprocal_in_range
        if flip_flag:
            flip_flag_count += 1

        results.append({
            **row,
            "bucket_ar_p10": round(p10, 3), "bucket_ar_p90": round(p90, 3),
            "in_normal_ar_range": in_range,
            "reciprocal_ar": round(reciprocal, 3),
            "reciprocal_fits_better": flip_flag,
        })
        if flip_flag:
            print(f"  FLAGGED (aspect-ratio flip signal): {Path(row['file_path']).name}  "
                  f"ar={ar:.3f} (bucket range [{p10:.2f},{p90:.2f}])  reciprocal={reciprocal:.3f}")

    print(f"\n{flip_flag_count} of {len(results)} images show an aspect ratio outside the "
          f"bucket's normal range whose RECIPROCAL fits comfortably inside it - "
          f"a deterministic, content-independent 90-degree-rotation candidate signal.")

    fieldnames = list(results[0].keys()) if results else []
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"Full results written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
