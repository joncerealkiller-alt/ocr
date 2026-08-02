"""
Benchmark 2.3 - Gemma Input Qualification, Variable 2: resize algorithm.

One variable only: which PIL resample filter is used in the pre-resize
step (LANCZOS / BICUBIC / BILINEAR). Resolution is held FIXED at 896px
longest edge - already confirmed in Variable 1's qualification to sit
in the safe region (>=448px), so any effect seen here is attributable
to algorithm choice, not confounded with the resolution threshold
effect already found.

Same rationale as Variable 1: the processor's own internal resize is
fixed (bicubic, confirmed via direct inspection), so this tests whether
MY pre-resize algorithm choice leaves a residual difference after being
resampled again by the processor - not whether Gemma "sees" the
algorithm directly.

Usage:
    python -m benchmark.gemma_input_qualification_resize_algorithm
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from core.classifier import build_classifier_loader, load_pipeline_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "outputs" / "benchmark2_gemma_input_qualification" / "resize_algorithm"

TEST_IMAGES = {
    "c10264.767_known_misclassified": PROJECT_ROOT / "data/working/oocihm.lac_reel_c10264.767.jpg",
    "c10301.601_known_ambiguous": PROJECT_ROOT / "data/working/oocihm.lac_reel_c10301.601.jpg",
    "c10264.658_known_correct_control": PROJECT_ROOT / "data/working/oocihm.lac_reel_c10264.658.jpg",
}
FIXED_LONGEST_EDGE = 896  # already confirmed safe/stable in Variable 1
ALGORITHMS = {"LANCZOS": Image.LANCZOS, "BICUBIC": Image.BICUBIC, "BILINEAR": Image.BILINEAR}


def resize_longest_edge(img: Image.Image, longest_edge: int, algorithm) -> Image.Image:
    w, h = img.size
    scale = longest_edge / max(w, h)
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), algorithm)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pipeline_cfg = load_pipeline_config()
    loader = build_classifier_loader(pipeline_cfg)
    results = []
    try:
        for label, path in TEST_IMAGES.items():
            orig = Image.open(path).convert("RGB")
            img_dir = OUT_DIR / label
            img_dir.mkdir(exist_ok=True)
            orig.save(img_dir / "original.png")
            print(f"\n=== {label} ===")

            for algo_name, algo in ALGORITHMS.items():
                processed = resize_longest_edge(orig, FIXED_LONGEST_EDGE, algo)
                processed.save(img_dir / f"processed_{algo_name}.png")

                result = loader.classify(str(path), processed)
                print(f"  algorithm={algo_name:<9} -> {result.category.value:<20} conf={result.confidence:.2f}")
                results.append({
                    "image": label, "algorithm": algo_name, "longest_edge": FIXED_LONGEST_EDGE,
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
