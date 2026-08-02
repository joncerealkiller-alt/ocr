"""
Benchmark 2.3 - Gemma Input Qualification, Variable 4: grayscale vs RGB.

One variable only: color channels. Resolution fixed at 896px (Variable 1
safe region), resize algorithm fixed at LANCZOS (Variable 2: no effect),
contrast fixed at none (Variable 3: no effect - simplest baseline).
Two conditions:
  - rgb: untouched (production default - documents are never converted
    to grayscale today)
  - grayscale: core.image_preprocessing.grayscale() - real existing
    function, desaturates then converts back to RGB so tensor shape is
    unchanged (loader sees 3 channels either way; only the color content
    differs)

Usage:
    python -m benchmark.gemma_input_qualification_grayscale
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from core.classifier import build_classifier_loader, load_pipeline_config
from core.image_preprocessing import grayscale as grayscale_fn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "outputs" / "benchmark2_gemma_input_qualification" / "grayscale"

TEST_IMAGES = {
    "c10264.767_known_misclassified": PROJECT_ROOT / "data/working/oocihm.lac_reel_c10264.767.jpg",
    "c10301.601_known_ambiguous": PROJECT_ROOT / "data/working/oocihm.lac_reel_c10301.601.jpg",
    "c10264.658_known_correct_control": PROJECT_ROOT / "data/working/oocihm.lac_reel_c10264.658.jpg",
    # microfilm scans above are monochrome-in-RGB (R=G=B, zero channel
    # spread) - grayscale() is a no-op on them. This 4th image is a real
    # color screenshot (channel_spread=28.3, measured) so the variable
    # actually gets exercised at least once.
    "genealogy_chart_screenshot_color": PROJECT_ROOT / "data/working/Screenshot 2026-05-03 182722.png",
}
FIXED_LONGEST_EDGE = 896


def resize_longest_edge(img: Image.Image, longest_edge: int) -> Image.Image:
    w, h = img.size
    scale = longest_edge / max(w, h)
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)


CONDITIONS = {
    "rgb": lambda img: img,
    "grayscale": lambda img: grayscale_fn(img),
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
                print(f"  condition={cond_name:<12} -> {result.category.value:<20} conf={result.confidence:.2f}")
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
