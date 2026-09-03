"""
Re-runs the real production classifier (unmodified prompt, reads
config/taxonomy.yaml fresh at call time - so this automatically picks up
the 2026-08-09 guidance edits) against a specific set of previously-
flagged images, to check whether the taxonomy edits actually changed the
outcome. Compares old (flagged-as-wrong) category against the new
prediction for each image.

Standalone, read-only against the external folder - same constraint as
diagnostics/classify_external_folder.py (never writes to data/buckets/,
manifest, or the pipeline DB).

Usage:
    python diagnostics/reclassify_flagged_sample.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from core.classifier import build_classifier_loader, load_pipeline_config

MISCLASSIFICATIONS_LOG = PROJECT_ROOT / "data" / "misclassifications.csv"


def main():
    with open(MISCLASSIFICATIONS_LOG, encoding="utf-8") as f:
        flagged = [r for r in csv.DictReader(f) if "Kemper_Ancestry" in r["file_path"]]

    print(f"{len(flagged)} previously-flagged images to re-check.\n")

    pipeline_cfg = load_pipeline_config()
    print("Loading Gemma via the real production build_classifier_loader() path "
          "(reads config/taxonomy.yaml fresh, picks up today's guidance edits automatically)...")
    loader = build_classifier_loader(pipeline_cfg, debug=False)
    print("Loaded.\n")

    n_changed, n_same = 0, 0
    for r in flagged:
        path = r["file_path"]
        old_category = r["category"]
        img = Image.open(path).convert("RGB")
        result = loader.classify(path, img)
        new_category = result.category.value
        changed = new_category != old_category
        if changed:
            n_changed += 1
        else:
            n_same += 1
        marker = "CHANGED" if changed else "same   "
        print(f"[{marker}] {Path(path).name:<45s} old={old_category:<20s} new={new_category:<20s} "
              f"conf={result.confidence:.2f}")
        print(f"          reason: {result.reason}")

    print(f"\n{n_changed}/{len(flagged)} predictions changed, {n_same}/{len(flagged)} stayed the same.")


if __name__ == "__main__":
    main()
