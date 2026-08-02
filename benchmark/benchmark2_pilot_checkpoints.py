"""
Benchmark 2 pilot: real data, not synthetic transforms. Collects the
two genuine embedding states (true pre-preprocessing baseline, and
current post-preprocessing) plus a real Gemma classification decision,
for the 11 archival-scan images in the Round 1-9 continuity sample that
have a traceable raw source.

Read-only on the raw source tree (J:\\Screenshots\\Knott_Ancestry) -
Image.open() only, never copied/written/modified there, per Jon's
explicit instruction. Does NOT write to data/buckets/*.csv - this is a
pilot, not a production classification run; results go to a separate
output file only.

Usage:
    python -m benchmark.benchmark2_pilot_checkpoints
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.vision_encoder_qualification import (
    build_model_and_transform, embed_pooled, cosine_sim, sample_real_images,
)
from core.classifier import build_classifier_loader, load_pipeline_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_SOURCE_ROOT = Path(r"J:\Screenshots\Knott_Ancestry\Archive Microfilms")
OUTPUT_JSON = PROJECT_ROOT / "data" / "outputs" / "benchmark2_pilot_checkpoints.json"

DINOV2_CANDIDATE = "vit_small_patch14_dinov2.lvd142m"
TEST_FILENAMES = [
    "oocihm.lac_reel_c10264.767.jpg", "oocihm.lac_reel_c10414.116.jpg",
    "oocihm.lac_reel_c10414.126.jpg", "oocihm.lac_reel_c10610.354.jpg",
    "oocihm.lac_reel_c10264.658.jpg", "oocihm.lac_reel_c10264.730.jpg",
    "oocihm.lac_reel_c10264.734.jpg", "oocihm.lac_reel_c10264.735.jpg",
    "oocihm.lac_reel_c10414.129.jpg", "oocihm.lac_reel_c10301.601.jpg",
    "oocihm.lac_reel_t2185.798.jpg",
]
REFERENCE_PER_BUCKET = 15


def find_working_copy(filename: str) -> Path | None:
    for bucket_csv in (PROJECT_ROOT / "data" / "buckets").glob("*.csv"):
        with open(bucket_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if Path(row["file_path"]).name == filename:
                    p = Path(row["file_path"])
                    if p.exists():
                        return p
    return None


def build_bucket_reference(model, transform, exclude_names: set[str]) -> dict[str, list[np.ndarray]]:
    clusters = sample_real_images(images_per_cluster=REFERENCE_PER_BUCKET)
    refs: dict[str, list[np.ndarray]] = {}
    for bucket, paths in clusters.items():
        embs = []
        for p in paths:
            if p.name in exclude_names:
                continue
            try:
                embs.append(embed_pooled(model, transform, Image.open(p).convert("RGB")))
            except Exception:
                continue
        refs[bucket] = embs
    return refs


def predict_bucket(emb: np.ndarray, refs: dict[str, list[np.ndarray]]) -> tuple[str, float]:
    scores = {}
    for bucket, vecs in refs.items():
        if vecs:
            scores[bucket] = float(np.mean([cosine_sim(emb, v) for v in vecs]))
    best = max(scores, key=scores.get)
    return best, scores[best]


def main():
    print(f"Pilot: {len(TEST_FILENAMES)} images with traceable raw source\n")

    # --- locate working copies ---
    working_paths = {}
    for name in TEST_FILENAMES:
        p = find_working_copy(name)
        if p:
            working_paths[name] = p
        else:
            print(f"  WARNING: no working-copy bucket entry found for {name}, skipping")
    names = list(working_paths.keys())

    # --- DINOv2 embeddings: raw (read-only) vs working (post-preprocessing) ---
    model, transform = build_model_and_transform(DINOV2_CANDIDATE)
    raw_emb, working_emb, drift = {}, {}, {}
    for name in names:
        raw_path = RAW_SOURCE_ROOT / name
        raw_img = Image.open(raw_path).convert("RGB")  # read-only, never written
        working_img = Image.open(working_paths[name]).convert("RGB")
        raw_emb[name] = embed_pooled(model, transform, raw_img)
        working_emb[name] = embed_pooled(model, transform, working_img)
        drift[name] = 1.0 - cosine_sim(raw_emb[name], working_emb[name])
        print(f"{name:<40} real preprocessing drift = {drift[name]:.4f}")

    # --- bucket reference set (excludes these 11 test images) ---
    refs = build_bucket_reference(model, transform, exclude_names=set(names))

    tower_predictions = {}
    for name in names:
        raw_bucket, raw_score = predict_bucket(raw_emb[name], refs)
        working_bucket, working_score = predict_bucket(working_emb[name], refs)
        tower_predictions[name] = {
            "raw_predicted_bucket": raw_bucket, "raw_score": raw_score,
            "working_predicted_bucket": working_bucket, "working_score": working_score,
        }

    # --- real Gemma classification, on the working (post-preprocessing) copies ---
    # This is what the real pipeline does - Gemma never sees the raw pre-preprocessing
    # version in production. Loaded once, released after, per this project's
    # load-once discipline. Does NOT write to data/buckets/*.csv.
    print("\nLoading Gemma for real classification (this is new territory this session)...")
    pipeline_cfg = load_pipeline_config()
    loader = build_classifier_loader(pipeline_cfg)
    gemma_results = {}
    try:
        for name in names:
            path = working_paths[name]
            with Image.open(path) as raw_image:
                result = loader.classify(str(path), raw_image)
            gemma_results[name] = {
                "category": result.category.value, "confidence": result.confidence,
                "reason": result.reason,
            }
            print(f"  {name:<40} Gemma -> {result.category.value} (conf={result.confidence:.2f})")
    finally:
        loader.release()

    # --- compare ---
    print(f"\n{'name':<40} {'drift':>7} {'raw_pred':>18} {'work_pred':>18} {'gemma':>18} {'raw==gemma':>11} {'work==gemma':>12}")
    agree_raw, agree_work = 0, 0
    full_results = []
    for name in names:
        tp = tower_predictions[name]
        gr = gemma_results[name]
        raw_agree = tp["raw_predicted_bucket"] == gr["category"]
        work_agree = tp["working_predicted_bucket"] == gr["category"]
        agree_raw += raw_agree
        agree_work += work_agree
        print(f"{name:<40} {drift[name]:>7.4f} {tp['raw_predicted_bucket']:>18} "
              f"{tp['working_predicted_bucket']:>18} {gr['category']:>18} "
              f"{str(raw_agree):>11} {str(work_agree):>12}")
        full_results.append({
            "name": name, "real_preprocessing_drift": drift[name],
            "tower_raw_prediction": tp, "gemma_result": gr,
            "raw_agrees_with_gemma": raw_agree, "working_agrees_with_gemma": work_agree,
        })

    print(f"\nTower (raw embedding) agreement with Gemma: {agree_raw}/{len(names)}")
    print(f"Tower (post-preprocessing embedding) agreement with Gemma: {agree_work}/{len(names)}")
    print(f"Mean real preprocessing drift: {np.mean(list(drift.values())):.4f}")

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment": "benchmark2_pilot_checkpoints",
        "n_images": len(names),
        "mean_real_preprocessing_drift": float(np.mean(list(drift.values()))),
        "tower_raw_agreement_with_gemma": agree_raw,
        "tower_working_agreement_with_gemma": agree_work,
        "results": full_results,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"\nFull results written to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
