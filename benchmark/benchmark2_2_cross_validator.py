"""
Benchmark 2.2 - Cross-Validator Qualification.

Research question: can an independent vision encoder act as a
statistically useful second opinion on the routing classifier (Gemma)?
Not "which encoder is best" (Benchmark 1) and not "what should the
metadata layer look like" (Benchmark 2) - this is specifically about
whether tower/Gemma disagreement is a usable gate for uncertain_review.

Stratified sample (not random - buckets aren't equally difficult, and
random sampling would be dominated by printed_document): 20 each from
printed_document, dense_tabular_rows, map_land_record, genealogy_chart,
portrait_photo, mixed_text_image, website_screenshot; all 4 from
handwritten_ledger (real corpus size, too small for a separate
reference pool - handled via leave-one-out); both of uncertain_review's
2 images included in the test set for completeness, but uncertain_review
is EXCLUDED from the tower's candidate/reference categories - it's not a
coherent visual category (a human-review catch-all, not a document
type), so there's no reference cluster for the tower to match against.

Entirely observational: reads data/working images directly (same as
every Benchmark 1 qualification round), writes nothing back to
data/buckets/*.csv or any production manifest. Every disagreement gets
a permanent record under data/outputs/benchmark2_2/disagreements/ -
image, both predictions, both confidences - for later human-verdict
review and as a standing regression set for future model/prompt changes.

Usage:
    python -m benchmark.benchmark2_2_cross_validator
"""
from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.vision_encoder_qualification import build_model_and_transform, embed_pooled, cosine_sim
from core.classifier import build_classifier_loader, load_pipeline_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUCKET_DIR = PROJECT_ROOT / "data" / "buckets"
OUT_DIR = PROJECT_ROOT / "data" / "outputs" / "benchmark2_2"
DISAGREEMENTS_DIR = OUT_DIR / "disagreements"

DINOV2_CANDIDATE = "vit_small_patch14_dinov2.lvd142m"

# (bucket, test_count, reference_count) - reference_count=0 for
# handwritten_ledger signals leave-one-out (reference == test set itself)
BUCKET_PLAN = [
    ("printed_document", 20, 15),
    ("dense_tabular_rows", 20, 15),
    ("map_land_record", 20, 15),
    ("genealogy_chart", 20, 15),
    ("portrait_photo", 20, 15),
    ("mixed_text_image", 20, 8),
    ("website_screenshot", 20, 15),
    ("handwritten_ledger", 4, 0),
]
UNCERTAIN_REVIEW_TEST_COUNT = 2  # test-only, never a tower candidate bucket


def load_existing_paths(bucket: str) -> list[Path]:
    paths = []
    with open(BUCKET_DIR / f"{bucket}.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            p = Path(row["file_path"])
            if p.exists():
                paths.append(p)
    return paths


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DISAGREEMENTS_DIR.mkdir(parents=True, exist_ok=True)

    model, transform = build_model_and_transform(DINOV2_CANDIDATE)

    test_items = []  # (name, path, true_bucket_from_sample)
    reference_embeddings: dict[str, list[tuple[str, np.ndarray]]] = {}

    for bucket, test_n, ref_n in BUCKET_PLAN:
        all_paths = load_existing_paths(bucket)
        test_paths = all_paths[:test_n]
        if ref_n == 0:
            # leave-one-out: reference pool IS the test set, excluded per-image at prediction time
            ref_paths = test_paths
        else:
            ref_paths = all_paths[test_n:test_n + ref_n]

        for p in test_paths:
            test_items.append((p.name, p, bucket))

        embs = []
        for p in ref_paths:
            try:
                embs.append((p.name, embed_pooled(model, transform, Image.open(p).convert("RGB"))))
            except Exception as e:
                print(f"  reference embed failed for {p.name}: {e}")
        reference_embeddings[bucket] = embs
        print(f"{bucket:<20} test={len(test_paths):>3}  reference={len(embs):>3}"
              + ("  (leave-one-out)" if ref_n == 0 else ""))

    # uncertain_review: test-only, no reference category
    ur_paths = load_existing_paths("uncertain_review")[:UNCERTAIN_REVIEW_TEST_COUNT]
    for p in ur_paths:
        test_items.append((p.name, p, "uncertain_review"))
    print(f"{'uncertain_review':<20} test={len(ur_paths):>3}  reference=  0  (excluded - not a visual category)")

    print(f"\nTotal test images: {len(test_items)}\n")

    def predict_bucket(emb: np.ndarray, exclude_name: str | None) -> tuple[str, float]:
        scores = {}
        for bucket, vecs in reference_embeddings.items():
            filtered = [v for (n, v) in vecs if n != exclude_name]
            if filtered:
                scores[bucket] = float(np.mean([cosine_sim(emb, v) for v in filtered]))
        best = max(scores, key=scores.get)
        return best, scores[best]

    # --- tower predictions ---
    tower_results = {}
    for name, path, sample_bucket in test_items:
        emb = embed_pooled(model, transform, Image.open(path).convert("RGB"))
        bucket, score = predict_bucket(emb, exclude_name=name)
        tower_results[name] = {"tower_bucket": bucket, "tower_score": score, "sample_bucket": sample_bucket}

    # --- real Gemma classification, observational only ---
    print("Loading Gemma for real classification (observational only, not written to production buckets)...")
    pipeline_cfg = load_pipeline_config()
    prompt_path = PROJECT_ROOT / pipeline_cfg["classifier"]["prompt_file"]
    import hashlib
    prompt_hash = hashlib.sha256(prompt_path.read_bytes()).hexdigest()[:16]
    loader = build_classifier_loader(pipeline_cfg)
    gemma_results = {}
    try:
        for i, (name, path, sample_bucket) in enumerate(test_items, 1):
            with Image.open(path) as raw_image:
                result = loader.classify(str(path), raw_image)
            gemma_results[name] = {
                "gemma_bucket": result.category.value, "gemma_confidence": result.confidence,
                "gemma_reason": result.reason,
            }
            print(f"[{i}/{len(test_items)}] {name:<45} tower={tower_results[name]['tower_bucket']:<18} "
                  f"gemma={result.category.value}")
    finally:
        loader.release()

    # --- compare, per-bucket + overall ---
    per_bucket_agree = {}
    disagreements = []
    for name, path, sample_bucket in test_items:
        tr = tower_results[name]
        gr = gemma_results[name]
        agree = tr["tower_bucket"] == gr["gemma_bucket"]
        per_bucket_agree.setdefault(sample_bucket, []).append(agree)
        if not agree:
            disagreements.append((name, path, sample_bucket, tr, gr))

    print(f"\n{'sample_bucket':<20}{'n':>5}{'agree_rate':>12}")
    for bucket, results in per_bucket_agree.items():
        rate = sum(results) / len(results)
        print(f"{bucket:<20}{len(results):>5}{rate:>11.1%}")

    overall_n = len(test_items)
    overall_agree = sum(1 for v in per_bucket_agree.values() for x in v if x)
    print(f"\nOverall: {overall_agree}/{overall_n} = {overall_agree/overall_n:.1%}")
    print(f"Disagreements: {len(disagreements)}")

    # --- save every disagreement permanently ---
    for i, (name, path, sample_bucket, tr, gr) in enumerate(disagreements, 1):
        d = DISAGREEMENTS_DIR / f"disagreement_{i:03d}"
        d.mkdir(exist_ok=True)
        shutil.copy2(path, d / f"image{path.suffix}")
        record = {
            "original_filename": name, "sample_bucket": sample_bucket,
            "source_path": str(path), "timestamp": datetime.now(timezone.utc).isoformat(),
            "tower_bucket": tr["tower_bucket"], "tower_score": tr["tower_score"],
            "gemma_bucket": gr["gemma_bucket"], "gemma_confidence": gr["gemma_confidence"],
            "gemma_reason": gr["gemma_reason"],
            "human_verdict": None,  # to be filled in during manual review: TOWER / GEMMA / BOTH / NEITHER
            "human_verdict_notes": None,
        }
        with open(d / "record.json", "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark": "2.2_cross_validator_qualification",
        "n_total": overall_n,
        "overall_agreement": f"{overall_agree}/{overall_n}",
        "per_bucket_agreement": {b: f"{sum(v)}/{len(v)}" for b, v in per_bucket_agree.items()},
        "n_disagreements": len(disagreements),
        "prompt_file_hash": prompt_hash,
        "note": "Observational only - nothing written to production manifests or bucket CSVs.",
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDisagreements saved to {DISAGREEMENTS_DIR}/ ({len(disagreements)} folders)")
    print(f"Summary written to {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
