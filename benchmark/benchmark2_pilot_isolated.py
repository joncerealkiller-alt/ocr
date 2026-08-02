"""
Benchmark 2 pilot - real data, entirely observational, entirely isolated
from production state. Per-image checkpoint folders (Jon's explicit
structure) so a single image's whole lifecycle is inspectable without
joining multiple CSVs/JSON files.

Pattern:

    Original archive (read-only, J:\\Screenshots\\Knott_Ancestry)
        -> copy into this image's own folder (raw_copy.*)
        -> Checkpoint A: metadata_raw.json (pre-preprocessing embedding)
        -> preprocessing (a separate copy in the same folder, preprocessed.*
           - never the raw_copy, never the original)
        -> Checkpoint B: metadata_preprocessed.json (post-preprocessing
           embedding + real drift vs checkpoint A)
        -> Gemma classification (real inference on preprocessed.*, result
           captured to gemma_reference.json only - never written to
           data/buckets/*.csv or any production manifest)
        -> comparison.json (per-image: does the tower's prediction at
           each checkpoint agree with Gemma's real decision)

data/outputs/benchmark2/summary.json rolls up all images. Nothing
outside data/outputs/benchmark2/ is ever written; the archive is only
ever opened for reading (shutil.copy2, matching core/manifest_pipeline.py's
own copy_to_working_dir() contract).

Usage:
    python -m benchmark.benchmark2_pilot_isolated
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.vision_encoder_qualification import (
    build_model_and_transform, embed_pooled, cosine_sim, sample_real_images,
)
from core.classifier import build_classifier_loader, load_pipeline_config
from core.manifest_pipeline import preprocess_for_manifest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_SOURCE_ROOT = Path(r"J:\Screenshots\Knott_Ancestry\Archive Microfilms")
BENCHMARK2_DIR = PROJECT_ROOT / "data" / "outputs" / "benchmark2"

DINOV2_CANDIDATE = "vit_small_patch14_dinov2.lvd142m"
TEST_FILENAMES = sorted([
    "oocihm.lac_reel_c10264.767.jpg", "oocihm.lac_reel_c10414.116.jpg",
    "oocihm.lac_reel_c10414.126.jpg", "oocihm.lac_reel_c10610.354.jpg",
    "oocihm.lac_reel_c10264.658.jpg", "oocihm.lac_reel_c10264.730.jpg",
    "oocihm.lac_reel_c10264.734.jpg", "oocihm.lac_reel_c10264.735.jpg",
    "oocihm.lac_reel_c10414.129.jpg", "oocihm.lac_reel_c10301.601.jpg",
    "oocihm.lac_reel_t2185.798.jpg",
])
REFERENCE_PER_BUCKET = 15


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
    scores = {b: float(np.mean([cosine_sim(emb, v) for v in vecs])) for b, vecs in refs.items() if vecs}
    best = max(scores, key=scores.get)
    return best, scores[best]


def main():
    BENCHMARK2_DIR.mkdir(parents=True, exist_ok=True)

    source_paths = {name: RAW_SOURCE_ROOT / name for name in TEST_FILENAMES}
    missing = [p for p in source_paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing from read-only archive: {missing}")

    image_dirs = {}
    for i, name in enumerate(TEST_FILENAMES, 1):
        d = BENCHMARK2_DIR / f"image_{i:04d}"
        d.mkdir(exist_ok=True)
        image_dirs[name] = d

    model, transform = build_model_and_transform(DINOV2_CANDIDATE)

    baseline_embeddings, postprocess_embeddings, drift = {}, {}, {}
    preprocessing_log = {}

    for name in TEST_FILENAMES:
        d = image_dirs[name]
        ext = source_paths[name].suffix
        raw_copy = d / f"raw_copy{ext}"
        preprocessed_copy = d / f"preprocessed{ext}"

        # Archive opened for reading only (shutil.copy2), twice - one
        # copy stays pristine forever (raw_copy), one gets preprocessed
        # in place (preprocessed) - never the same file, never the original.
        shutil.copy2(source_paths[name], raw_copy)
        shutil.copy2(source_paths[name], preprocessed_copy)

        # --- Checkpoint A ---
        emb_a = embed_pooled(model, transform, Image.open(raw_copy).convert("RGB"))
        baseline_embeddings[name] = emb_a
        with open(d / "metadata_raw.json", "w", encoding="utf-8") as f:
            json.dump({
                "checkpoint": "A_baseline_pre_preprocessing",
                "original_filename": name,
                "source_archive_path": str(source_paths[name]),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "encoder": DINOV2_CANDIDATE,
                "embedding_dim": len(emb_a),
                "embedding": emb_a.tolist(),
            }, f, indent=2)

        # --- Real preprocessing, the "preprocessed" copy only ---
        angle, profile = preprocess_for_manifest(preprocessed_copy)
        preprocessing_log[name] = {"deskew_angle_deg": angle, "profile": profile}

        # --- Checkpoint B ---
        emb_b = embed_pooled(model, transform, Image.open(preprocessed_copy).convert("RGB"))
        postprocess_embeddings[name] = emb_b
        d_val = 1.0 - cosine_sim(emb_a, emb_b)
        drift[name] = d_val
        with open(d / "metadata_preprocessed.json", "w", encoding="utf-8") as f:
            json.dump({
                "checkpoint": "B_post_preprocessing",
                "original_filename": name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "encoder": DINOV2_CANDIDATE,
                "embedding_dim": len(emb_b),
                "embedding": emb_b.tolist(),
                "deskew_angle_deg": angle, "preprocessing_profile": profile,
                "real_preprocessing_drift_vs_checkpoint_A": d_val,
            }, f, indent=2)
        print(f"{name:<40} deskew={angle:+.3f}  drift={d_val:.4f}")

    # --- Tower nearest-neighbor bucket prediction, both checkpoints ---
    refs = build_bucket_reference(model, transform, exclude_names=set(TEST_FILENAMES))
    tower_predictions = {}
    for name in TEST_FILENAMES:
        a_bucket, a_score = predict_bucket(baseline_embeddings[name], refs)
        b_bucket, b_score = predict_bucket(postprocess_embeddings[name], refs)
        tower_predictions[name] = {
            "checkpoint_A_predicted_bucket": a_bucket, "checkpoint_A_score": a_score,
            "checkpoint_B_predicted_bucket": b_bucket, "checkpoint_B_score": b_score,
        }

    # --- Checkpoint C: real Gemma classification on preprocessed.* only -
    # never written to data/buckets/*.csv or any production manifest ---
    #
    # Provenance note (found the hard way, mid-session): ClassificationResult's
    # own prompt_version field is NOT reliably bumped when the prompt file's
    # actual content changes - confirmed directly, prompt_version stayed "v1"
    # across a real edit that added a whole new bucket category
    # (website_screenshot) to classifier_classify_v1.txt. A version LABEL can
    # go stale; a hash of the actual file content can't - same principle
    # already established for the vision-tower schema, now confirmed to apply
    # to Gemma's prompt too, not just encoder checkpoints.
    print("\nLoading Gemma for real classification (observational only)...")
    pipeline_cfg = load_pipeline_config()
    prompt_path = PROJECT_ROOT / pipeline_cfg["classifier"]["prompt_file"]
    prompt_file_hash = hashlib.sha256(prompt_path.read_bytes()).hexdigest()[:16]
    loader = build_classifier_loader(pipeline_cfg)
    gemma_results = {}
    try:
        for name in TEST_FILENAMES:
            d = image_dirs[name]
            preprocessed_copy = d / f"preprocessed{source_paths[name].suffix}"
            with Image.open(preprocessed_copy) as raw_image:
                result = loader.classify(str(preprocessed_copy), raw_image)
            gemma_results[name] = {
                "category": result.category.value, "confidence": result.confidence, "reason": result.reason,
                "model": result.model, "prompt_version": result.prompt_version,
                "prompt_file": str(prompt_path.relative_to(PROJECT_ROOT)),
                "prompt_file_hash": prompt_file_hash,
            }
            with open(d / "gemma_reference.json", "w", encoding="utf-8") as f:
                json.dump({
                    "checkpoint": "C_gemma_reference_classification",
                    "original_filename": name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    **gemma_results[name],
                }, f, indent=2)
            print(f"  {name:<40} Gemma -> {result.category.value} (conf={result.confidence:.2f})")
    finally:
        loader.release()

    # --- Per-image comparison.json ---
    agree_a, agree_b = 0, 0
    for name in TEST_FILENAMES:
        tp = tower_predictions[name]
        gr = gemma_results[name]
        a_agree = tp["checkpoint_A_predicted_bucket"] == gr["category"]
        b_agree = tp["checkpoint_B_predicted_bucket"] == gr["category"]
        agree_a += a_agree
        agree_b += b_agree
        with open(image_dirs[name] / "comparison.json", "w", encoding="utf-8") as f:
            json.dump({
                "original_filename": name,
                "real_preprocessing_drift": drift[name],
                "tower_prediction": tp,
                "gemma_reference": gr,
                "checkpoint_A_agrees_with_gemma": a_agree,
                "checkpoint_B_agrees_with_gemma": b_agree,
            }, f, indent=2)

    # --- Corpus-level summary.json ---
    n = len(TEST_FILENAMES)
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_images": n,
        "image_folders": [f"image_{i:04d}" for i in range(1, n + 1)],
        "mean_real_preprocessing_drift": float(np.mean(list(drift.values()))),
        "tower_checkpoint_A_agreement_with_gemma": f"{agree_a}/{n}",
        "tower_checkpoint_B_agreement_with_gemma": f"{agree_b}/{n}",
        "note": "Observational only - nothing written to production manifests, "
                "bucket CSVs, or operational metadata. J:\\Screenshots\\Knott_Ancestry "
                "was opened for reading only throughout.",
    }
    with open(BENCHMARK2_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTower (checkpoint A) agreement with Gemma: {agree_a}/{n}")
    print(f"Tower (checkpoint B) agreement with Gemma: {agree_b}/{n}")
    print(f"Mean real preprocessing drift: {np.mean(list(drift.values())):.4f}")
    print(f"\nPer-image folders + summary.json written to {BENCHMARK2_DIR}/")


if __name__ == "__main__":
    main()
