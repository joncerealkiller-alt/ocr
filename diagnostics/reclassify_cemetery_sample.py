"""One-off recheck of the 3 grave-marker-plaque images flagged in the
Kemper_Ancestry review, after moving cemetery_photo's sort_order from
110 to 15 (2026-08-09) - see config/taxonomy.yaml's comment on that
category for why. Same standalone, read-only pattern as
diagnostics/reclassify_flagged_sample.py.

Usage:
    python diagnostics/reclassify_cemetery_sample.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from core.classifier import build_classifier_loader, load_pipeline_config

IMAGES = [
    ("J:\\Screenshots\\Kemper_Ancestry\\Screenshot 2026-05-01 023355.png", "printed_document"),
    ("J:\\Screenshots\\Kemper_Ancestry\\Screenshot 2026-05-01 023938.png", "printed_document"),
    ("J:\\Screenshots\\Kemper_Ancestry\\Screenshot 2026-05-04 190703.png", "printed_document"),
]


def main():
    pipeline_cfg = load_pipeline_config()
    print("Loading Gemma (real production path, taxonomy read fresh - "
          "picks up the sort_order change automatically)...")
    loader = build_classifier_loader(pipeline_cfg, debug=False)
    print("Loaded.\n")

    n_fixed = 0
    for path, old_category in IMAGES:
        img = Image.open(path).convert("RGB")
        result = loader.classify(path, img)
        fixed = result.category.value == "cemetery_photo"
        if fixed:
            n_fixed += 1
        marker = "FIXED  " if fixed else "still wrong"
        print(f"[{marker}] {Path(path).name:<40s} old={old_category:<18s} "
              f"new={result.category.value:<18s} conf={result.confidence:.2f}")
        print(f"           reason: {result.reason}")

    print(f"\n{n_fixed}/{len(IMAGES)} now correctly classified as cemetery_photo.")


if __name__ == "__main__":
    main()
