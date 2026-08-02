"""
Benchmark 2.3 - Gemma Input Qualification, Variable 6 (final): denoising.

One variable only: denoising treatment. Resolution/algorithm/contrast/
color/sharpening all held at their qualified-safe settings (896px,
LANCZOS, no contrast treatment, RGB, no sharpening - all shown to have
no effect). Two conditions:
  - none: untouched
  - denoise: core.image_preprocessing.denoise() - MedianFilter(size=3),
    deliberately gentle per its own docstring (aggressive denoising
    risks blurring fine handwriting detail)

Usage:
    python -m benchmark.gemma_input_qualification_denoise
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from core.classifier import build_classifier_loader, load_pipeline_config
from core.image_preprocessing import denoise as denoise_fn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "outputs" / "benchmark2_gemma_input_qualification" / "denoise"

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
    "none": lambda img: img,
    "denoise_median3": lambda img: denoise_fn(img, radius=1.0),
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pipeline_cfg = load_pipeline_config()
    loader = build_classifier_loader(pipeline_cfg)
    results = []
    try:
        for label, path in TEST_IMAGES.items():
            orig = Image.open(path).convert("RGB")
            resized = resize_longest_edge(orig, FIXED_LONGEST_EDGE)
            img_dir = OUT_DIR / label
            img_dir.mkdir(exist_ok=True)
            orig.save(img_dir / "original.png")
            print(f"\n=== {label} ===")

            for cond_name, fn in CONDITIONS.items():
                processed = fn(resized)
                processed.save(img_dir / f"processed_{cond_name}.png")

                result = loader.classify(str(path), processed)
                print(f"  condition={cond_name:<16} -> {result.category.value:<20} conf={result.confidence:.2f}")
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
