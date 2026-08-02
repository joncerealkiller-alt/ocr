"""
Follow-up to scripts/triage_flagged_deskew.py: Jon's actual goal is a
vision-tower signal that flags images to bypass auto-deskew directly -
not necessarily knowing which rotation fixes them. The rotation-margin
column answers "which orientation wins," which is a different question
from "is this image anomalous enough to route away from the normal
pipeline at all."

This reuses the already-computed sim_0/sim_90/sim_180/sim_270 columns
in data/outputs/flagged_deskew_triage.csv (no re-embedding of flagged
images) and adds what was missing: a baseline of what NORMAL
(non-flagged, non-reference) images in the same buckets score against
the same bucket reference set, so "low similarity" has something to be
low relative to.

Usage:
    python -m scripts.calibrate_deskew_anomaly_threshold
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.vision_encoder_qualification import build_model_and_transform, embed_pooled, cosine_sim
from scripts.triage_flagged_deskew import CANDIDATE, REFERENCE_PER_BUCKET, build_reference_embeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUCKET_DIR = PROJECT_ROOT / "data" / "buckets"
TRIAGE_CSV = PROJECT_ROOT / "data" / "outputs" / "flagged_deskew_triage.csv"
FLAGGED_CSV = PROJECT_ROOT / "data" / "flagged_bad_deskew.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "outputs" / "flagged_deskew_anomaly_flags.csv"

CALIBRATION_PER_BUCKET = 15


def load_triage_rows() -> list[dict]:
    with open(TRIAGE_CSV, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["status"] == "OK"]


def build_calibration_sample(model, transform, bucket: str, exclude: set[str]) -> list[float]:
    """Non-flagged, non-reference-set images' sim_0 against the SAME
    reference set the flagged batch was scored against - the baseline
    "what does normal look like" distribution."""
    refs = build_reference_embeddings(model, transform, bucket, exclude)
    if not refs:
        return []
    csv_path = BUCKET_DIR / f"{bucket}.csv"
    sims = []
    seen_as_ref = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            p = Path(row["file_path"])
            if str(p) in exclude or not p.exists():
                continue
            # skip the first REFERENCE_PER_BUCKET non-excluded files - those
            # ARE the reference set itself, scoring them against themselves
            # would be circular
            if seen_as_ref < REFERENCE_PER_BUCKET:
                seen_as_ref += 1
                continue
            try:
                img = Image.open(p).convert("RGB")
                emb = embed_pooled(model, transform, img)
            except Exception:
                continue
            sims.append(float(np.mean([cosine_sim(emb, r) for r in refs])))
            if len(sims) >= CALIBRATION_PER_BUCKET:
                break
    return sims


def main():
    triage_rows = load_triage_rows()
    flagged_paths = set()
    with open(FLAGGED_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            flagged_paths.add(row["file_path"])

    model, transform = build_model_and_transform(CANDIDATE)

    buckets = sorted({r["bucket"] for r in triage_rows})
    baseline: dict[str, list[float]] = {}
    for bucket in buckets:
        sims = build_calibration_sample(model, transform, bucket, flagged_paths)
        baseline[bucket] = sims
        if sims:
            print(f"{bucket:<20} baseline n={len(sims):>2}  "
                  f"mean={np.mean(sims):.3f}  p10={np.percentile(sims,10):.3f}  "
                  f"p25={np.percentile(sims,25):.3f}  min={min(sims):.3f}")
        else:
            print(f"{bucket:<20} baseline n=0 - no calibration sample available")

    results = []
    for row in triage_rows:
        bucket = row["bucket"]
        base = baseline.get(bucket, [])
        max_sim = max(float(row["sim_0"]), float(row["sim_90"]), float(row["sim_180"]), float(row["sim_270"]))
        sim_0 = float(row["sim_0"])
        if base:
            p10 = np.percentile(base, 10)
            below_baseline_sim0 = sim_0 < p10
            below_baseline_maxsim = max_sim < p10
        else:
            p10, below_baseline_sim0, below_baseline_maxsim = None, None, None
        results.append({
            **row,
            "baseline_p10": round(p10, 4) if p10 is not None else "",
            "max_sim_across_rotations": round(max_sim, 4),
            "flag_low_even_at_best_rotation": below_baseline_maxsim,
            "flag_low_at_current_orientation": below_baseline_sim0,
        })

    flag_count = sum(1 for r in results if r["flag_low_even_at_best_rotation"] is True)
    print(f"\n{flag_count} of {len(results)} images score below the bucket's p10 baseline "
          f"even at their BEST-scoring rotation - anomalous relative to normal bucket "
          f"members regardless of orientation, independent of the rotation-margin question.")

    fieldnames = list(results[0].keys()) if results else []
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"Full results written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
