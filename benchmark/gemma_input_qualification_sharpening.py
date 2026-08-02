"""
Benchmark 2.3 - Gemma Input Qualification, Variable 5: sharpening.

One variable only: sharpening treatment. Resolution/algorithm/contrast/
color held at their qualified-safe settings (896px, LANCZOS, no
contrast treatment, RGB - all shown to have no effect so far). Three
conditions, two genuinely distinct mechanisms already in
core/image_preprocessing.py:
  - none: untouched
  - sharpen: ImageEnhance.Sharpness global scalar boost (factor=2.0)
  - unsharp_mask: classic unsharp-mask edge filter (radius=1.5, amount=1.2)
    - a different mechanism (blurred-copy subtraction), documented to
      look less artificial on scanned handwriting than a flat global
      boost

Usage:
    python -m benchmark.gemma_input_qualification_sharpening
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from core.classifier import build_classifier_loader, load_pipeline_config
from core.image_preprocessing import sharpen as sharpen_fn, unsharp_mask as unsharp_mask_fn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "outputs" / "benchmark2_gemma_input_qualification" / "sharpening"

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
    "sharpen_global": lambda img: sharpen_fn(img, factor=2.0),
    "unsharp_mask": lambda img: unsharp_mask_fn(img, radius=1.5, amount=1.2),
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
