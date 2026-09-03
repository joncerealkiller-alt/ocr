"""
Stage 1 (pre-Stage-3) vs Stage 4 (post-Stage-3) comparison, 2026-08-04 -
the "future comparison/analysis stage" explicitly deferred when Stage 4
was built. Both snapshots now exist for the full 1750-image corpus:

  semantic: data/baseline_embeddings.json (pre) vs
            data/postprocessing_embeddings.json (post)
  physical: data/outputs/image_analysis/analysis_report.csv (pre) vs
            data/outputs/image_analysis/analysis_report_postprocessing.csv (post)

Read-only against all four inputs. Reuses existing, already-validated
functions rather than reimplementing: core/vision_embeddings.py's
cosine_sim(), core/decision_engine.py's build_reference_embeddings() /
compute_tower_consensus_for_image() (the exact nearest-cluster logic the
Multi-Tower Routing Audit validated) for the bucket-flip check.

Deliberately measurement only - reports deltas and flip counts, does not
decide which snapshot is "more correct" (no ground truth exists to judge
that against, per this project's own established discipline).

Usage:
    python -m benchmark.stage4_before_after_comparison
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from core.baseline_embeddings import (
    DEFAULT_BASELINE_PATH, DEFAULT_POSTPROCESSING_PATH, resolve_baseline_image_path,
)
from core.decision_engine import build_reference_embeddings, compute_tower_consensus_for_image, BUCKET_DIR
from core.vision_embeddings import QUALIFIED_ENCODERS, cosine_sim

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHYSICAL_PRE_PATH = PROJECT_ROOT / "data" / "outputs" / "image_analysis" / "analysis_report.csv"
PHYSICAL_POST_PATH = PROJECT_ROOT / "data" / "outputs" / "image_analysis" / "analysis_report_postprocessing.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "stage4_before_after_comparison"

DRIFT_OUTLIER_THRESHOLD = 0.9999  # same threshold used throughout this session's equivalence work

PHYSICAL_NUMERIC_FIELDS = [
    "deskew_angle_deg", "blur_laplacian_var", "noise_residual_std",
    "page_contrast_p5_p95_spread", "page_luminance_mean", "page_text_height_px",
    "page_stroke_width_px", "page_ink_fraction",
    "table_contrast_p5_p95_spread", "table_luminance_mean", "table_text_height_px",
    "table_stroke_width_px", "table_ink_fraction",
]


def _load_embeddings(path: Path) -> dict[str, dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    return {resolve_baseline_image_path(r["image"]): r for r in records}


def semantic_comparison() -> dict:
    print("=== Semantic comparison (baseline vs postprocessing embeddings) ===")
    pre = _load_embeddings(DEFAULT_BASELINE_PATH)
    post = _load_embeddings(DEFAULT_POSTPROCESSING_PATH)
    common = sorted(set(pre.keys()) & set(post.keys()))
    print(f"pre: {len(pre)} records, post: {len(post)} records, common: {len(common)}")

    per_encoder_sims: dict[str, list[float]] = {name: [] for name, _ in QUALIFIED_ENCODERS}
    per_image_mean_sim: dict[str, float] = {}
    outlier_rows = []

    for path in common:
        pre_rec, post_rec = pre[path], post[path]
        sims_this_image = []
        for encoder_name, _ in QUALIFIED_ENCODERS:
            pre_entry = pre_rec["embeddings"].get(encoder_name)
            post_entry = post_rec["embeddings"].get(encoder_name)
            if pre_entry is None or post_entry is None:
                continue
            v_pre = np.array(pre_entry["vector"], dtype=np.float64)
            v_post = np.array(post_entry["vector"], dtype=np.float64)
            sim = cosine_sim(v_pre, v_post)
            per_encoder_sims[encoder_name].append(sim)
            sims_this_image.append(sim)
            if sim < DRIFT_OUTLIER_THRESHOLD:
                outlier_rows.append({
                    "image": path, "encoder": encoder_name, "cosine_sim": round(float(sim), 6),
                })
        if sims_this_image:
            per_image_mean_sim[path] = float(np.mean(sims_this_image))

    print("\nPer-encoder cosine similarity (pre vs post), full corpus:")
    encoder_stats = {}
    for encoder_name, sims in per_encoder_sims.items():
        arr = np.array(sims)
        encoder_stats[encoder_name] = {
            "n": len(arr), "min": float(arr.min()), "mean": float(arr.mean()),
            "max": float(arr.max()), "p50": float(np.median(arr)),
            "p01": float(np.percentile(arr, 1)),
        }
        print(f"  {encoder_name:<15} n={len(arr):<5} min={arr.min():.6f} "
              f"p01={np.percentile(arr, 1):.6f} p50={np.median(arr):.6f} "
              f"mean={arr.mean():.6f} max={arr.max():.6f}")

    all_sims = np.array([s for sims in per_encoder_sims.values() for s in sims])
    print(f"\nOverall ({len(all_sims)} pairs): min={all_sims.min():.6f} mean={all_sims.mean():.6f} "
          f"max={all_sims.max():.6f}")
    print(f"Pairs below {DRIFT_OUTLIER_THRESHOLD} threshold: {len(outlier_rows)}/{len(all_sims)}")

    # Bucket-prediction flip check: self-consistent per snapshot (each
    # image's pre-embedding scored against a reference set built from
    # OTHER images' pre-embeddings; same for post) - matches the n=11
    # pilot's Checkpoint A/B design (docs/BENCHMARK2_METADATA_LAYER_
    # QUALIFICATION.md's Sixth extension).
    print("\nBuilding reference embeddings (pre and post, self-consistent)...")
    ref_pre = build_reference_embeddings(bucket_dir=BUCKET_DIR, baseline_path=DEFAULT_BASELINE_PATH)
    ref_post = build_reference_embeddings(bucket_dir=BUCKET_DIR, baseline_path=DEFAULT_POSTPROCESSING_PATH)

    n_scored = 0
    n_consensus_flips = 0
    n_per_encoder_votes = 0
    n_per_encoder_flips = 0
    flip_rows = []

    for path in common:
        pre_rec, post_rec = pre[path], post[path]
        cat_pre, top_pre, enc_pre = compute_tower_consensus_for_image(pre_rec, ref_pre, exclude_key=path)
        cat_post, top_post, enc_post = compute_tower_consensus_for_image(post_rec, ref_post, exclude_key=path)
        if cat_pre is None and cat_post is None:
            continue
        n_scored += 1

        per_encoder_flip_here = []
        for encoder_name in enc_pre:
            if encoder_name not in enc_post:
                continue
            n_per_encoder_votes += 1
            if enc_pre[encoder_name]["winner"] != enc_post[encoder_name]["winner"]:
                n_per_encoder_flips += 1
                per_encoder_flip_here.append(encoder_name)

        if cat_pre != cat_post or top_pre != top_post or per_encoder_flip_here:
            n_consensus_flips += 1 if (cat_pre != cat_post or top_pre != top_post) else 0
            flip_rows.append({
                "image": path,
                "consensus_pre": cat_pre, "consensus_post": cat_post,
                "top_bucket_pre": top_pre, "top_bucket_post": top_post,
                "encoders_flipped": per_encoder_flip_here,
            })

    print(f"\nBucket-prediction flip check ({n_scored} images with evidence in both snapshots):")
    print(f"  consensus_category or top_bucket changed: {n_consensus_flips}/{n_scored}")
    print(f"  per-encoder winner flips: {n_per_encoder_flips}/{n_per_encoder_votes}")
    print(f"  images with at least one encoder flip: {len(flip_rows)}/{n_scored}")

    return {
        "n_common_images": len(common),
        "encoder_stats": encoder_stats,
        "overall": {"n_pairs": len(all_sims), "min": float(all_sims.min()),
                    "mean": float(all_sims.mean()), "max": float(all_sims.max())},
        "n_outlier_pairs": len(outlier_rows),
        "outlier_rows": outlier_rows,
        "bucket_flip_check": {
            "n_scored": n_scored,
            "n_consensus_or_top_bucket_flips": n_consensus_flips,
            "n_per_encoder_votes": n_per_encoder_votes,
            "n_per_encoder_flips": n_per_encoder_flips,
            "n_images_with_any_encoder_flip": len(flip_rows),
        },
        "flip_rows": flip_rows,
        "per_image_mean_sim": per_image_mean_sim,
    }


def physical_comparison() -> dict:
    print("\n=== Physical comparison (analysis_report pre vs post) ===")
    with open(PHYSICAL_PRE_PATH, newline="", encoding="utf-8") as f:
        pre_rows = {r["file_path"]: r for r in csv.DictReader(f)}
    with open(PHYSICAL_POST_PATH, newline="", encoding="utf-8") as f:
        post_rows = {r["file_path"]: r for r in csv.DictReader(f)}
    common = sorted(set(pre_rows.keys()) & set(post_rows.keys()))
    print(f"pre: {len(pre_rows)} rows, post: {len(post_rows)} rows, common: {len(common)}")

    field_deltas: dict[str, list[float]] = {f: [] for f in PHYSICAL_NUMERIC_FIELDS}
    for path in common:
        pre_r, post_r = pre_rows[path], post_rows[path]
        for field in PHYSICAL_NUMERIC_FIELDS:
            pre_v, post_v = pre_r.get(field, ""), post_r.get(field, "")
            if pre_v in ("", None) or post_v in ("", None):
                continue
            try:
                field_deltas[field].append(float(post_v) - float(pre_v))
            except ValueError:
                continue

    field_stats = {}
    print("\nPer-field delta (post - pre), full corpus:")
    for field, deltas in field_deltas.items():
        if not deltas:
            print(f"  {field:<30} n=0 (no comparable values)")
            continue
        arr = np.array(deltas)
        field_stats[field] = {
            "n": len(arr), "min": float(arr.min()), "mean": float(arr.mean()),
            "max": float(arr.max()), "abs_mean": float(np.abs(arr).mean()),
            "n_nonzero": int((arr != 0).sum()),
        }
        print(f"  {field:<30} n={len(arr):<5} mean_delta={arr.mean():+.4f} "
              f"abs_mean_delta={np.abs(arr).mean():.4f} min={arr.min():+.4f} max={arr.max():+.4f} "
              f"n_nonzero={int((arr != 0).sum())}")

    # page/table detection method + boundary agreement
    method_changed = sum(
        1 for p in common if pre_rows[p].get("page_method") != post_rows[p].get("page_method")
    )
    table_changed = sum(
        1 for p in common
        if (pre_rows[p].get("table_boundary") != post_rows[p].get("table_boundary"))
    )
    print(f"\npage_method changed: {method_changed}/{len(common)}")
    print(f"table_boundary presence changed: {table_changed}/{len(common)}")

    return {
        "n_common_rows": len(common),
        "field_stats": field_stats,
        "page_method_changed": method_changed,
        "table_boundary_changed": table_changed,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    semantic = semantic_comparison()
    physical = physical_comparison()

    report = {"semantic": semantic, "physical": physical}
    out_path = OUTPUT_DIR / "report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nFull report written to {out_path}")


if __name__ == "__main__":
    main()
