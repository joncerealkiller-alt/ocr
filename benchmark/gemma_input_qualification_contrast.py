"""
Benchmark 2.3 - Gemma Input Qualification, Variable 3: contrast normalization.

One variable only: contrast treatment. Resolution fixed at 896px
(confirmed safe region, Variable 1), resize algorithm fixed at LANCZOS
(confirmed zero effect, Variable 2). Three real conditions, reusing
core/image_preprocessing.py's actual functions rather than
reimplementing anything:
  - none: no contrast treatment at all (PREPROCESSING_PROFILES["none"])
  - autocontrast: autocontrast(cutoff=1.0) - this IS production's real
    DEFAULT_PREPROCESSING_PROFILE, confirmed via PREPROCESSING_PROFILES -
    "existing pipeline normalization" and "Autocontrast" are the same
    condition, not two separate ones, so tested once here.
  - clahe: clahe(clip_limit=2.0, tile_size=8) - a real, already-existing
    function, distinct algorithm (local adaptive contrast vs.
    autocontrast's global histogram stretch).

Usage:
    python -m benchmark.gemma_input_qualification_contrast
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from core.classifier import build_classifier_loader, load_pipeline_config
from core.image_preprocessing import apply_profile, clahe as clahe_fn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "outputs" / "benchmark2_gemma_input_qualification" / "contrast"

TEST_IMAGES = {
    "c10264.767_known_misclassified": PROJECT_ROOT / "data/working/oocihm.lac_reel_c10264.767.jpg",
    "c10301.601_known_ambiguous": PROJECT_ROOT / "data/working/oocihm.lac_reel_c10301.601.jpg",
    "c10264.658_known_correct_control": PROJECT_ROOT / "data/working/oocihm.lac_reel_c10264.658.jpg",
}
FIXED_LONGEST_EDGE = 896


def resize_longest_edge(img: Image.Image, longest_edge: int) -> Image.Image:
    w, h = img.size
    scale = longest_edge / max(w, h)
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)


CONDITIONS = {
    "none": lambda img: apply_profile(img, "none"),
    "autocontrast_production_default": lambda img: apply_profile(img, "autocontrast"),
    "clahe": lambda img: clahe_fn(img, clip_limit=2.0, tile_size=8),
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pipeline_cfg = load_pipeline_config()
    loader = build_classifier_loader(pipeline_cfg)
    results = []
    try:
        for label, path in TEST_IMAGES.items():
            orig = Image.open(path).convert("RGB")
            resized = resize_longest_edge(orig, FIXED_LONGEST_EDGE)  # fixed resolution+algorithm baseline
            img_dir = OUT_DIR / label
            img_dir.mkdir(exist_ok=True)
            orig.save(img_dir / "original.png")
            print(f"\n=== {label} ===")

            for cond_name, fn in CONDITIONS.items():
                processed = fn(resized)
                processed.save(img_dir / f"processed_{cond_name}.png")

                result = loader.classify(str(path), processed)
                print(f"  condition={cond_name:<32} -> {result.category.value:<20} conf={result.confidence:.2f}")
                results.append({
                    "image": label, "condition": cond_name,
                    "category": result.category.value, "confidence": result.confidence, "reason": result.reason,
                })
    finally:
        loader.release()

    with open(OUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== Verdict per image ===")
    for label in TEST_IMAGES:
        rows = [r for r in results if r["image"] == label]
        categories = {r["category"] for r in rows}
        confidences = [r["confidence"] for r in rows]
        changed = len(categories) > 1
        print(f"{label:<35} categories_seen={categories}  changed={changed}  "
              f"confidence_range={max(confidences)-min(confidences):.3f}")

    print(f"\nResults + visual checkpoints written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
