"""
Benchmark 2.3 - Gemma Input Qualification, Variable 1: input resolution.

Mechanistic finding confirmed before this ran (read-only processor
inspection, no inference needed): Gemma4ImageProcessor always resizes
to a fixed patch grid derived from image_seq_length (pixel_values shape
is IDENTICAL - (1, 2520, 768) - whether the input is pre-resized to
224x224, 1400x1400, or left at native resolution). So this experiment
is NOT testing "does Gemma see more pixels at higher resolution" (it
always sees the same shape) - it's testing whether compounding two
resize passes (mine, then the processor's own fixed bicubic resize)
changes the classification outcome versus a single-pass resize.

One variable only: resolution (longest edge). Resize algorithm fixed
at LANCZOS throughout (this experiment isn't testing algorithm choice -
that's a separate, later variable). Same 3 known images as the
token-budget test, for direct comparability: two known disagreements,
one known-correct control.

Saves original + the exact processed image handed to Gemma, for every
resolution tested, per Jon's visual-checkpoint requirement.

Usage:
    python -m benchmark.gemma_input_qualification_resolution
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from core.classifier import build_classifier_loader, load_pipeline_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "outputs" / "benchmark2_gemma_input_qualification" / "resolution"

TEST_IMAGES = {
    "c10264.767_known_misclassified": PROJECT_ROOT / "data/working/oocihm.lac_reel_c10264.767.jpg",
    "c10301.601_known_ambiguous": PROJECT_ROOT / "data/working/oocihm.lac_reel_c10301.601.jpg",
    "c10264.658_known_correct_control": PROJECT_ROOT / "data/working/oocihm.lac_reel_c10264.658.jpg",
}
# longest-edge targets - fixed LANCZOS resample throughout, only this varies
LONGEST_EDGES = [224, 448, 896, 1400, None]  # None = native resolution, no pre-resize


def resize_longest_edge(img: Image.Image, longest_edge: int | None) -> Image.Image:
    if longest_edge is None:
        return img
    w, h = img.size
    scale = longest_edge / max(w, h)
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)


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
            print(f"\n=== {label} (native size {orig.size}) ===")

            for edge in LONGEST_EDGES:
                processed = resize_longest_edge(orig, edge)
                edge_label = f"native" if edge is None else f"{edge}px"
                processed.save(img_dir / f"processed_{edge_label}.png")

                result = loader.classify(str(path), processed)
                print(f"  longest_edge={edge_label:<8} size={processed.size}  "
                      f"-> {result.category.value:<20} conf={result.confidence:.2f}")
                results.append({
                    "image": label, "longest_edge": edge_label, "processed_size": list(processed.size),
                    "category": result.category.value, "confidence": result.confidence, "reason": result.reason,
                })
    finally:
        loader.release()

    with open(OUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # --- qualified verdict per image: any change across the resolution sweep? ---
    print("\n=== Verdict per image ===")
    for label in TEST_IMAGES:
        rows = [r for r in results if r["image"] == label]
        categories = {r["category"] for r in rows}
        confidences = [r["confidence"] for r in rows]
        changed = len(categories) > 1
        conf_range = max(confidences) - min(confidences)
        print(f"{label:<35} categories_seen={categories}  changed={changed}  "
              f"confidence_range={conf_range:.3f}")

    print(f"\nResults + visual checkpoints written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
