"""
Three-way comparison, 2026-08-04: for every image whose tower-consensus
prediction changed between Stage 1 (pre-Stage-3) and Stage 4 (post-
Stage-3) - the 1217-image flip set found by benchmark/stage4_before_
after_comparison.py - adds Gemma's REAL classification decision
(pipeline.db's images.bucket/classifier_confidence, written by
core/classifier.py) so each flip can be read as "did this change move
the tower TOWARD or AWAY FROM what Gemma actually decided."

Extends the cross-check design from docs/BENCHMARK2_METADATA_LAYER_
QUALIFICATION.md's Sixth extension (the n=11 real pilot's "A=Gemma"/
"B=Gemma" columns) to the full corpus's flipped subset, now that both
snapshots are known-good (see docs/STAGE1_STAGE4_BEFORE_AFTER_
COMPARISON.md for the physical-data bug found and fixed first).

Measurement only - reports agreement/disagreement patterns, does not
decide whether pre or post is "more correct" (no ground truth exists to
judge that against, per this project's own established discipline).
Read-only against all inputs; reuses benchmark/stage4_before_after_
comparison.py's semantic_comparison() rather than recomputing the flip
logic a second time.

Usage:
    python -m benchmark.stage1_gemma_stage4_flip_report
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmark.stage4_before_after_comparison import semantic_comparison
from core.pipeline_db import PipelineDatabase, DEFAULT_DB_PATH

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "stage1_gemma_stage4_flip_report"


def _agreement_category(pre_agrees: bool | None, post_agrees: bool | None) -> str:
    if pre_agrees is None or post_agrees is None:
        return "no_gemma_bucket"
    if pre_agrees and post_agrees:
        return "agrees_both"
    if pre_agrees and not post_agrees:
        return "lost_agreement"
    if not pre_agrees and post_agrees:
        return "gained_agreement"
    return "disagrees_both"


def main() -> None:
    print("Computing Stage 1 vs Stage 4 semantic comparison (reused, not recomputed)...")
    semantic = semantic_comparison()
    flip_rows = semantic["flip_rows"]
    print(f"\n{len(flip_rows)} images with a tower-prediction change (consensus, top bucket, "
          f"or at least one encoder's vote).")

    db = PipelineDatabase(DEFAULT_DB_PATH)
    report_rows = []
    counts = {"agrees_both": 0, "lost_agreement": 0, "gained_agreement": 0,
              "disagrees_both": 0, "no_gemma_bucket": 0}

    for row in flip_rows:
        image_path = row["image"]
        image = db.get_image_by_path(image_path)
        gemma_bucket = image["bucket"] if image else None
        gemma_confidence = image["classifier_confidence"] if image else None
        gemma_model = image["classifier_model"] if image else None

        pre_agrees = (row["top_bucket_pre"] == gemma_bucket) if gemma_bucket else None
        post_agrees = (row["top_bucket_post"] == gemma_bucket) if gemma_bucket else None
        category = _agreement_category(pre_agrees, post_agrees)
        counts[category] += 1

        report_rows.append({
            "image": image_path,
            "tower_top_bucket_pre": row["top_bucket_pre"],
            "tower_top_bucket_post": row["top_bucket_post"],
            "tower_consensus_pre": row["consensus_pre"],
            "tower_consensus_post": row["consensus_post"],
            "gemma_bucket": gemma_bucket,
            "gemma_confidence": gemma_confidence,
            "gemma_model": gemma_model,
            "pre_agrees_gemma": pre_agrees,
            "post_agrees_gemma": post_agrees,
            "agreement_category": category,
            "n_encoders_flipped": len(row["encoders_flipped"]),
            "encoders_flipped": ",".join(row["encoders_flipped"]),
        })

    print("\n=== Agreement-with-Gemma pattern, across all flipped images ===")
    total = len(flip_rows)
    for category, label in [
        ("gained_agreement", "Pre disagreed with Gemma, post now agrees"),
        ("lost_agreement", "Pre agreed with Gemma, post no longer does"),
        ("agrees_both", "Agreed with Gemma both before and after"),
        ("disagrees_both", "Disagreed with Gemma both before and after"),
        ("no_gemma_bucket", "No Gemma classification on record"),
    ]:
        n = counts[category]
        pct = 100 * n / total if total else 0
        print(f"  {label:<50} {n:>5} ({pct:.1f}%)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "flip_report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()) if report_rows else [])
        writer.writeheader()
        writer.writerows(report_rows)
    print(f"\nPer-image report written to {csv_path}")

    json_path = OUTPUT_DIR / "flip_report_summary.json"
    json_path.write_text(json.dumps({"counts": counts, "total": total}, indent=2), encoding="utf-8")
    print(f"Summary written to {json_path}")


if __name__ == "__main__":
    main()
