"""
Quick, targeted test: does Gemma's classification answer change with a
higher image_token_budget than gemma.yaml's current default (140)?

Does NOT modify config/models/gemma.yaml on disk - overrides
config.extra["image_token_budget"] in memory only, per test call.

Test images:
  - c10264.767: KNOWN, directly visually confirmed handwritten cursive,
    but classified "printed_document" (conf 0.95) at budget=140 in
    Benchmark 2.2/pilot runs. If budget is the real bottleneck, a higher
    budget should have a real shot at correcting this.
  - c10301.601: the other known disagreement (mixed photo+caption
    content, tower gave a low-confidence "printed_document" guess,
    Gemma said "portrait_photo" citing "people and horses" - a reasoning
    detail already found to be imprecise). Check if budget affects this too.
  - c10264.658: a CONTROL - Gemma correctly agreed with the tower here at
    budget=140 (printed_document). If a higher budget flips this
    unexpectedly, that's a stability concern, not just a budget-adequacy one.

Usage:
    python -m benchmark.gemma_token_budget_test
"""
from __future__ import annotations

from pathlib import Path
from PIL import Image

from core.classifier import build_classifier_loader, load_pipeline_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_IMAGES = {
    "c10264.767 (known handwritten, misclassified@140)": PROJECT_ROOT / "data/working/oocihm.lac_reel_c10264.767.jpg",
    "c10301.601 (ambiguous mixed content)": PROJECT_ROOT / "data/working/oocihm.lac_reel_c10301.601.jpg",
    "c10264.658 (control, correct@140)": PROJECT_ROOT / "data/working/oocihm.lac_reel_c10264.658.jpg",
}
BUDGETS = [140, 280, 560]


def main():
    pipeline_cfg = load_pipeline_config()
    loader = build_classifier_loader(pipeline_cfg)
    try:
        for label, path in TEST_IMAGES.items():
            print(f"\n=== {label} ===")
            for budget in BUDGETS:
                loader.config.extra["image_token_budget"] = budget
                with Image.open(path) as raw_image:
                    result = loader.classify(str(path), raw_image)
                print(f"  budget={budget:>4}  -> {result.category.value:<20} "
                      f"conf={result.confidence:.2f}  reason: {result.reason}")
    finally:
        loader.release()


if __name__ == "__main__":
    main()
