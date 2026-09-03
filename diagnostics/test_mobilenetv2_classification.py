"""
Tests google/mobilenet_v2_1.0_224 (timm tag: mobilenetv2_100.ra_in1k -
matching architecture/width/resolution, ImageNet-1k pretrained) as a
candidate NEW tower for the evidence-fusion classifier work, per Jon's
direction. NOT one of the 8 QUALIFIED_ENCODERS already validated in the
Multi-Tower Routing Audit (docs/BENCHMARK2_3_MULTI_TOWER_ROUTING_AUDIT.md)
- MobileNetV2 is much smaller/cheaper (a real candidate specifically for
"can a lightweight model still contribute useful signal"), trained on
ImageNet-1k's 1000 OBJECT categories (cat/dog/envelope/book jacket/etc),
not document types - the question is whether its embeddings still cluster
usefully for THIS project's document taxonomy despite that domain gap.

Reuses core/vision_embeddings.py's EXISTING, already-validated functions
directly (build_model_and_transform, embed_pooled, predict_nearest_
bucket, score_all_buckets) - same nearest-cluster-via-mean-cosine-
similarity methodology as the real Multi-Tower Routing Audit, not a
new/different evaluation method invented for this one model.

Ground truth: `data/outputs/reference_pipeline_v2/manifest_final.csv`
(the real, pre-automation whole-corpus human-reviewed set found
2026-08-06 - see docs/MULTI_SOURCE_VOTING_CLASSIFIER_PROPOSAL.md's
"Resource" section). Reference set: up to 15 images per category (or
all available if fewer). Test set: up to 30 DIFFERENT images per
category (disjoint from the reference set), held out for accuracy
measurement - same reference/held-out split discipline as the existing
audit work, not testing against images the reference set already saw.

Usage:
    python diagnostics/test_mobilenetv2_classification.py
"""

from __future__ import annotations

import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from core.vision_embeddings import (
    build_model_and_transform, embed_pooled, predict_nearest_bucket, score_all_buckets,
)

MANIFEST_PATH = PROJECT_ROOT / "data" / "outputs" / "reference_pipeline_v2" / "manifest_final.csv"
TIMM_TAG = "mobilenetv2_100.ra_in1k"  # matches google/mobilenet_v2_1.0_224's architecture/width/resolution
N_REFERENCE_PER_BUCKET = 15
N_TEST_PER_BUCKET = 30
RANDOM_SEED = 42
REPORT_PATH = PROJECT_ROOT / "data" / "logs" / "reviewed" / "mobilenetv2_classification_report.txt"


def load_by_category() -> dict[str, list[str]]:
    with open(MANIFEST_PATH, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    by_cat: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        if r["category"] == "uncertain_review":
            continue
        if Path(r["file_path"]).exists():
            by_cat[r["category"]].append(r["file_path"])
    return by_cat


def main():
    print(f"Loading {TIMM_TAG} (matches google/mobilenet_v2_1.0_224)...")
    model, transform = build_model_and_transform(TIMM_TAG)
    print("Loaded.\n")

    by_cat = load_by_category()
    rng = random.Random(RANDOM_SEED)

    reference_paths: dict[str, list[str]] = {}
    test_paths: dict[str, list[str]] = {}
    for cat, paths in by_cat.items():
        shuffled = paths[:]
        rng.shuffle(shuffled)
        ref_n = min(N_REFERENCE_PER_BUCKET, len(shuffled))
        reference_paths[cat] = shuffled[:ref_n]
        remaining = shuffled[ref_n:]
        test_paths[cat] = remaining[:N_TEST_PER_BUCKET]

    lines = []
    def log(s=""):
        lines.append(s)
        print(s)

    log(f"Categories: {sorted(by_cat.keys())}")
    log(f"Reference set sizes: {{cat: len(paths) for cat, paths in reference_paths.items()}}")
    for cat in sorted(reference_paths):
        log(f"  {cat}: {len(reference_paths[cat])} reference, {len(test_paths[cat])} test")
    log()

    # -- embed reference set --
    log("Embedding reference set...")
    reference_embeddings: dict[str, list[tuple[str, "object"]]] = {}
    for cat, paths in reference_paths.items():
        vecs = []
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
                vec = embed_pooled(model, transform, img)
                vecs.append((p, vec))
            except Exception as e:
                log(f"  SKIP reference {p}: {type(e).__name__}: {e}")
        reference_embeddings[cat] = vecs
        log(f"  {cat}: {len(vecs)} embedded")
    log()

    # -- embed + classify test set --
    log("Classifying test set...")
    results = []
    for cat, paths in test_paths.items():
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
                vec = embed_pooled(model, transform, img)
            except Exception as e:
                log(f"  SKIP test {p}: {type(e).__name__}: {e}")
                continue
            predicted, score = predict_nearest_bucket(vec, reference_embeddings)
            correct = predicted == cat
            results.append({"gt": cat, "predicted": predicted, "score": score, "correct": correct})

    log(f"\nClassified {len(results)} test images.\n")

    # -- report --
    n = len(results)
    correct_n = sum(1 for r in results if r["correct"])
    log(f"=== OVERALL ACCURACY: {correct_n}/{n} ({100*correct_n/n:.1f}%) ===\n")

    log("=== Per-category accuracy ===")
    for cat in sorted(test_paths.keys()):
        cat_results = [r for r in results if r["gt"] == cat]
        if not cat_results:
            continue
        cat_correct = sum(1 for r in cat_results if r["correct"])
        log(f"  {cat:<20s} {cat_correct}/{len(cat_results)} "
            f"({100*cat_correct/len(cat_results):.1f}%)")

    log("\n=== Confusion matrix (rows=ground truth, cols=predicted) ===")
    categories = sorted(test_paths.keys())
    header = "gt\\pred".ljust(22) + "".join(c[:10].ljust(12) for c in categories)
    log(header)
    for gt_cat in categories:
        row_results = [r for r in results if r["gt"] == gt_cat]
        counts = Counter(r["predicted"] for r in row_results)
        row = gt_cat.ljust(22) + "".join(str(counts.get(c, 0)).ljust(12) for c in categories)
        log(row)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(f"\nReport written: {REPORT_PATH}")


if __name__ == "__main__":
    main()
