"""
Triage tool for data/flagged_bad_deskew.csv (Jon's manual review-UI
flags, 2026-08-01). Some of these are small-angle deskew failures the
existing pipeline should just do better on; some are 90/180/270-degree
rotations, which core/row_segmentation.py's estimate_deskew_angle()
cannot detect at all (it searches a small angle range, not full
rotations) and need a categorically different fix.

Approach: reuse the Vision Qualification Battery's validated DINOv2
candidate (Round 1's qualified "semantic retrieval" role) as a cheap
orientation detector. For each flagged image, embed it at 0/90/180/270
degrees and compare each to a reference set of NON-flagged images from
the same bucket. Whichever rotation is most similar to how that
bucket's images normally look is the recommended orientation - if that's
0 degrees, the image most likely just needs a better small-angle
deskew; if it's 90/180/270, it needs an explicit rotation BEFORE
deskew even runs.

Reference-set caveat, stated plainly: "non-flagged" means "not flagged
by Jon's own review pass" (deskew, misclassification, or new-bucket
flags) - not an independently verified ground truth. Reasonable given
what's actually available, not a formal label set.

Usage:
    python -m scripts.triage_flagged_deskew
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from benchmark.vision_encoder_qualification import build_model_and_transform, embed_pooled, cosine_sim
from core.image_analysis import analyze_image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FLAGGED_CSV = PROJECT_ROOT / "data" / "flagged_bad_deskew.csv"
BUCKET_DIR = PROJECT_ROOT / "data" / "buckets"
OUTPUT_CSV = PROJECT_ROOT / "data" / "outputs" / "flagged_deskew_triage.csv"

CANDIDATE = "vit_small_patch14_dinov2.lvd142m"
REFERENCE_PER_BUCKET = 15
ROTATIONS = [0, 90, 180, 270]
CLEAR_MARGIN = 0.03  # similarity gap needed to call a rotation "clearly better," not noise


def load_flagged() -> list[dict]:
    rows = []
    with open(FLAGGED_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def build_reference_embeddings(model, transform, bucket: str, exclude: set[str]) -> list[np.ndarray]:
    csv_path = BUCKET_DIR / f"{bucket}.csv"
    if not csv_path.exists():
        return []
    embeddings = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            p = Path(row["file_path"])
            if str(p) in exclude or not p.exists():
                continue
            try:
                img = Image.open(p).convert("RGB")
                embeddings.append(embed_pooled(model, transform, img))
            except Exception:
                continue
            if len(embeddings) >= REFERENCE_PER_BUCKET:
                break
    return embeddings


def main():
    flagged = load_flagged()
    flagged_paths = {row["file_path"] for row in flagged}
    print(f"Total flagged rows: {len(flagged)}")

    model, transform = build_model_and_transform(CANDIDATE)

    reference_cache: dict[str, list[np.ndarray]] = {}
    results = []
    missing_count = 0

    for row in flagged:
        path = Path(row["file_path"])
        bucket = row["bucket"]
        if not path.exists():
            missing_count += 1
            results.append({**row, "status": "FILE_MISSING"})
            continue

        if bucket not in reference_cache:
            reference_cache[bucket] = build_reference_embeddings(model, transform, bucket, flagged_paths)
        refs = reference_cache[bucket]
        if not refs:
            results.append({**row, "status": "NO_REFERENCE_SET_AVAILABLE"})
            continue

        try:
            analysis = analyze_image(str(path))
            det_aspect_ratio = analysis.aspect_ratio
            det_deskew_angle = analysis.deskew_angle_deg
        except Exception as e:
            det_aspect_ratio, det_deskew_angle = None, None
            print(f"  image_analysis failed on {path.name}: {e}")

        orig_img = Image.open(path).convert("RGB")
        sims = {}
        for rot in ROTATIONS:
            rotated = orig_img.rotate(-rot, expand=True) if rot else orig_img
            emb = embed_pooled(model, transform, rotated)
            sims[rot] = float(np.mean([cosine_sim(emb, r) for r in refs]))

        best_rot = max(sims, key=sims.get)
        sorted_sims = sorted(sims.values(), reverse=True)
        margin = sorted_sims[0] - sorted_sims[1]

        if margin < CLEAR_MARGIN:
            recommendation = "ambiguous - needs manual review"
        elif best_rot == 0:
            recommendation = "orientation OK - likely a small-angle deskew fix, not a rotation problem"
        else:
            recommendation = f"likely needs {best_rot} degree rotation before deskew"

        results.append({
            **row,
            "status": "OK",
            "det_aspect_ratio": det_aspect_ratio,
            "det_deskew_angle_deg": det_deskew_angle,
            "sim_0": round(sims[0], 4), "sim_90": round(sims[90], 4),
            "sim_180": round(sims[180], 4), "sim_270": round(sims[270], 4),
            "best_rotation_deg": best_rot, "margin": round(margin, 4),
            "recommendation": recommendation,
        })
        print(f"{path.name:<50} best_rot={best_rot:>3}  margin={margin:.3f}  -> {recommendation}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["bucket", "file_path", "category", "reason", "status", "det_aspect_ratio",
                  "det_deskew_angle_deg", "sim_0", "sim_90", "sim_180", "sim_270",
                  "best_rotation_deg", "margin", "recommendation"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\nMissing files: {missing_count}")
    print(f"Full triage results written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
