"""
Compares the current data/buckets/ classification state (2026-08-04
fresh full-corpus run, confirmed current as of ~2026-08-05 01:19) against
the reference_pipeline_v4 checkpoint - taken immediately before that run,
same 1750-image corpus. Same methodology as
benchmark/fresh_pass_vs_checkpoint_comparison.py (v3 vs. fresh), applied
one checkpoint later.

Unlike the v3 comparison, v4 and "current" were classified under
IDENTICAL classifier code+config (no taxonomy/prompt changes happened
between the v4 checkpoint and this run) - so this comparison is a
determinism/drift check under the SAME conditions the checkpoint used,
not a before/after-a-change comparison. (The image_seq_length token-
budget fix and layout-detector addition both landed in the repo AFTER
this "current" corpus run completed, per direct confirmation, so neither
is exercised by either side of this diff.)

Read-only against both bucket-CSV sets. Matches by basename.

Usage:
    python -m benchmark.fresh_pass_vs_v4_checkpoint_comparison
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CURRENT_BUCKET_DIR = PROJECT_ROOT / "data" / "buckets"
CHECKPOINT_BUCKET_DIR = PROJECT_ROOT / "data" / "outputs" / "reference_pipeline_v4" / "buckets"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "fresh_pass_vs_v4_checkpoint_comparison"

ALL_BUCKETS = [
    "dense_tabular_rows", "genealogy_chart", "handwritten_ledger", "map_land_record",
    "printed_document", "mixed_text_image", "portrait_photo", "website_screenshot",
    "photo_collage", "casual_photo", "cemetery_photo", "uncertain_review",
]


def _load_bucket_dir(bucket_dir: Path) -> dict[str, dict]:
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
    current = _load_bucket_dir(CURRENT_BUCKET_DIR)
    checkpoint = _load_bucket_dir(CHECKPOINT_BUCKET_DIR)

    print(f"Current data/buckets: {len(current)} classified images")
    print(f"reference_pipeline_v4 checkpoint: {len(checkpoint)} classified images")

    common = sorted(set(current.keys()) & set(checkpoint.keys()))
    only_current = sorted(set(current.keys()) - set(checkpoint.keys()))
    only_checkpoint = sorted(set(checkpoint.keys()) - set(current.keys()))
    print(f"Common (both have a classification): {len(common)}")
    print(f"Only in current: {len(only_current)}")
    print(f"Only in checkpoint: {len(only_checkpoint)}")
    if only_current:
        print(f"  sample only-in-current: {only_current[:5]}")
    if only_checkpoint:
        print(f"  sample only-in-checkpoint: {only_checkpoint[:5]}")

    print("\n=== Bucket distribution: v4 checkpoint vs current ===")
    checkpoint_counts = Counter(v["bucket"] for v in checkpoint.values())
    current_counts = Counter(v["bucket"] for v in current.values())
    for bucket in ALL_BUCKETS:
        c, f = checkpoint_counts.get(bucket, 0), current_counts.get(bucket, 0)
        delta = f - c
        print(f"  {bucket:<20} v4={c:<6} current={f:<6} delta={delta:+d}")

    changed = []
    unchanged = 0
    for basename in common:
        c_bucket, f_bucket = checkpoint[basename]["bucket"], current[basename]["bucket"]
        if c_bucket != f_bucket:
            changed.append({
                "image": basename,
                "v4_bucket": c_bucket,
                "current_bucket": f_bucket,
                "v4_confidence": checkpoint[basename]["confidence"],
                "current_confidence": current[basename]["confidence"],
                "v4_reason": checkpoint[basename]["reason"],
                "current_reason": current[basename]["reason"],
            })
        else:
            unchanged += 1

    print(f"\n=== Per-image classification changes ===")
    print(f"Unchanged: {unchanged}/{len(common)} ({100*unchanged/len(common):.1f}%)")
    print(f"Changed:   {len(changed)}/{len(common)} ({100*len(changed)/len(common):.1f}%)")

    print("\nTransition tally (v4 -> current):")
    transitions = Counter(f"{c['v4_bucket']} -> {c['current_bucket']}" for c in changed)
    for key, n in transitions.most_common(20):
        print(f"  {key:<50} {n}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    changed_path = OUTPUT_DIR / "changed_classifications.csv"
    with open(changed_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(changed[0].keys()) if changed else
                                 ["image", "v4_bucket", "current_bucket", "v4_confidence",
                                  "current_confidence", "v4_reason", "current_reason"])
        writer.writeheader()
        writer.writerows(changed)
    print(f"\nChanged-classification detail written to {changed_path}")


if __name__ == "__main__":
    main()
