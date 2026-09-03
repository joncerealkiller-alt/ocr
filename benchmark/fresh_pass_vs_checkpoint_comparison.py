"""
Compares the 2026-08-04 fresh Stage 0-5 pass (this session's changes:
GPU-sharded k=4 semantic capture, 3 new taxonomy categories, taxonomy-
driven prompt template) against the reference_pipeline_v3 checkpoint -
the exact same 1750-image corpus, same source files, run through the
pre-change pipeline. Since the input is held constant, any difference
in classification output is attributable to the pipeline changes, not
a different corpus.

Read-only against both bucket-CSV sets. Matches by basename (working_path
filenames are stable across both runs, since the same source list and
working_dir were used).

Usage:
    python -m benchmark.fresh_pass_vs_checkpoint_comparison
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRESH_BUCKET_DIR = PROJECT_ROOT / "data" / "buckets"
CHECKPOINT_BUCKET_DIR = PROJECT_ROOT / "data" / "outputs" / "reference_pipeline_v3" / "buckets"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "fresh_pass_vs_checkpoint_comparison"

ALL_BUCKETS = [
    "dense_tabular_rows", "genealogy_chart", "handwritten_ledger", "map_land_record",
    "printed_document", "mixed_text_image", "portrait_photo", "website_screenshot",
    "photo_collage", "casual_photo", "cemetery_photo", "uncertain_review",
]


def _load_bucket_dir(bucket_dir: Path) -> dict[str, dict]:
    """Returns {basename: {"bucket": ..., "confidence": ..., "reason": ...}}"""
    by_basename = {}
    for bucket in ALL_BUCKETS:
        path = bucket_dir / f"{bucket}.csv"
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fp = row.get("file_path", "")
                if not fp:
                    continue
                basename = Path(fp).name
                by_basename[basename] = {
                    "bucket": bucket,
                    "confidence": row.get("confidence", ""),
                    "reason": row.get("reason", ""),
                }
    return by_basename


def main() -> None:
    fresh = _load_bucket_dir(FRESH_BUCKET_DIR)
    checkpoint = _load_bucket_dir(CHECKPOINT_BUCKET_DIR)

    print(f"Fresh run: {len(fresh)} classified images")
    print(f"Checkpoint: {len(checkpoint)} classified images")

    common = sorted(set(fresh.keys()) & set(checkpoint.keys()))
    only_fresh = sorted(set(fresh.keys()) - set(checkpoint.keys()))
    only_checkpoint = sorted(set(checkpoint.keys()) - set(fresh.keys()))
    print(f"Common (both runs classified): {len(common)}")
    print(f"Only in fresh: {len(only_fresh)}")
    print(f"Only in checkpoint: {len(only_checkpoint)}")

    print("\n=== Bucket distribution: checkpoint vs fresh ===")
    checkpoint_counts = Counter(v["bucket"] for v in checkpoint.values())
    fresh_counts = Counter(v["bucket"] for v in fresh.values())
    for bucket in ALL_BUCKETS:
        c, f = checkpoint_counts.get(bucket, 0), fresh_counts.get(bucket, 0)
        delta = f - c
        print(f"  {bucket:<20} checkpoint={c:<6} fresh={f:<6} delta={delta:+d}")

    changed = []
    unchanged = 0
    for basename in common:
        c_bucket, f_bucket = checkpoint[basename]["bucket"], fresh[basename]["bucket"]
        if c_bucket != f_bucket:
            changed.append({
                "image": basename,
                "checkpoint_bucket": c_bucket,
                "fresh_bucket": f_bucket,
                "checkpoint_confidence": checkpoint[basename]["confidence"],
                "fresh_confidence": fresh[basename]["confidence"],
                "checkpoint_reason": checkpoint[basename]["reason"],
                "fresh_reason": fresh[basename]["reason"],
            })
        else:
            unchanged += 1

    print(f"\n=== Per-image classification changes ===")
    print(f"Unchanged: {unchanged}/{len(common)} ({100*unchanged/len(common):.1f}%)")
    print(f"Changed:   {len(changed)}/{len(common)} ({100*len(changed)/len(common):.1f}%)")

    print("\nTransition tally (checkpoint -> fresh):")
    transitions = Counter(f"{c['checkpoint_bucket']} -> {c['fresh_bucket']}" for c in changed)
    for key, n in transitions.most_common(20):
        print(f"  {key:<50} {n}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    changed_path = OUTPUT_DIR / "changed_classifications.csv"
    with open(changed_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(changed[0].keys()) if changed else [])
        writer.writeheader()
        writer.writerows(changed)
    print(f"\nChanged-classification detail written to {changed_path}")


if __name__ == "__main__":
    main()
