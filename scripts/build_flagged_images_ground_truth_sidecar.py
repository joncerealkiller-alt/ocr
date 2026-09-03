"""
Builds data/misclassifications.csv - the ground-truth labeling queue
debug_tools/review_uncertain.py's --source misclassifications mode reads
- populated with the 77 "flagged" images from benchmark/gained_lost_
agreement_characterization.py (57 gained-agreement + 20 lost-agreement).
Jon is manually classifying these to get real ground truth for the
question this whole investigation has deliberately left open: which
snapshot's tower reading (or Gemma's) is actually correct.

This mode was chosen deliberately over --source uncertain: these images
are a curated research sample, not live uncertain_review.csv queue
entries - --source misclassifications labels correct_category IN PLACE
and never touches a real bucket CSV or the DB (see that script's own
module docstring, "DUAL-SOURCE MODE"), exactly the non-destructive
behavior wanted here.

file_path points at the PRODUCTION working image (data/working/...,
i.e. the post-Stage-3 pixels) - the real pipeline artifact whose bucket
assignment ground truth actually matters, not a research-only raw copy.
bucket/category/confidence/reason/model/prompt_version are pulled
directly from that image's REAL row in data/buckets/<bucket>.csv (the
authoritative source - not recomputed or guessed) so the review UI's
"Gemma predicted: ..." display matches production exactly.

Read-only against data/buckets/*.csv. Writes ONLY data/misclassifications.
csv, which did not exist before this ran (confirmed via git status -
tracked historically, deleted during this session's earlier data reset).

Usage:
    python -m scripts.build_flagged_images_ground_truth_sidecar
"""
from __future__ import annotations

import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHAR_DIR = PROJECT_ROOT / "data" / "outputs" / "gained_lost_agreement_characterization"
BUCKET_DIR = PROJECT_ROOT / "data" / "buckets"
OUTPUT_PATH = PROJECT_ROOT / "data" / "misclassifications.csv"

MISCLASSIFICATION_FIELDS = [
    "bucket", "file_path", "category", "confidence", "reason",
    "model", "prompt_version", "correct_category",
]


def _load_flagged() -> list[dict]:
    rows = []
    for name in ("gained_agreement_detail.csv", "lost_agreement_detail.csv"):
        with open(CHAR_DIR / name, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r["_source_group"] = name.replace("_agreement_detail.csv", "")
                rows.append(r)
    return rows


def _bucket_csv_rows_by_path(bucket: str) -> dict[str, dict]:
    path = BUCKET_DIR / f"{bucket}.csv"
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {r["file_path"]: r for r in csv.DictReader(f)}


def main() -> None:
    flagged = _load_flagged()
    print(f"{len(flagged)} flagged images (gained + lost agreement).")

    bucket_cache: dict[str, dict[str, dict]] = {}
    out_rows = []
    missing = []

    for row in flagged:
        image_path = row["image"]
        bucket = row["gemma_bucket"]
        if bucket not in bucket_cache:
            bucket_cache[bucket] = _bucket_csv_rows_by_path(bucket)
        bucket_row = bucket_cache[bucket].get(image_path)

        if bucket_row is None:
            missing.append((image_path, bucket))
            # Fall back to what's already known (from the earlier flip
            # report) rather than dropping the image from the ground-
            # truth set entirely - reason/prompt_version are the only
            # fields not already available from that source.
            out_rows.append({
                "bucket": bucket,
                "file_path": image_path,
                "category": bucket,
                "confidence": row.get("gemma_confidence", ""),
                "reason": "",
                "model": row.get("gemma_model", ""),
                "prompt_version": "",
                "correct_category": "",
            })
            continue

        out_rows.append({
            "bucket": bucket,
            "file_path": image_path,
            "category": bucket_row.get("category", bucket),
            "confidence": bucket_row.get("confidence", ""),
            "reason": bucket_row.get("reason", ""),
            "model": bucket_row.get("model", ""),
            "prompt_version": bucket_row.get("prompt_version", ""),
            "correct_category": "",
        })

    if missing:
        print(f"\n{len(missing)} image(s) not found in their expected bucket CSV "
              f"(fell back to flip-report data, no 'reason' available):")
        for path, bucket in missing:
            print(f"  {path}  (expected in {bucket}.csv)")

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MISCLASSIFICATION_FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\n{len(out_rows)} row(s) written to {OUTPUT_PATH}")
    print("Load with: python debug_tools/review_uncertain.py --source misclassifications")


if __name__ == "__main__":
    main()
