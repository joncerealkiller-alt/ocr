"""
Builds a real, fresh, held-out ground-truth set from Jon's manual review
of the 487-image Kemper_Ancestry classification pass (2026-08-08/09).

Ground truth = the original full-generation prediction UNLESS the image
was flagged in data/misclassifications.csv AND a corrected category was
determined during the 2026-08-09 review-pass discussion (visual
inspection of all 20 flagged images, confirmed/adjusted with Jon in
chat). One image (130926 - repeating-row handwritten register, ambiguous
dense_tabular_rows vs. handwritten_ledger) was left undetermined and is
excluded rather than guessed.

Filtered to the 8 flat categories the hidden-state linear probe was
trained on (dense_tabular_rows, genealogy_chart, handwritten_ledger,
map_land_record, mixed_text_image, portrait_photo, printed_document,
website_screenshot) - the probe has no way to predict cemetery_photo/
photo_collage/casual_photo/uncertain_review, so those get excluded here,
not silently scored as wrong.

This is the FIRST genuinely fresh, human-labeled holdout this project
has had for the flat-8 hidden-state probe work - every prior split
(Stage 1/2, multi-seed, MLP) reused the same original 2233-image
train/val/test partition. Nothing in this set has ever been seen by any
probe, threshold sweep, or the original vit_finetune_dataset.csv.

Usage:
    python diagnostics/build_kemper_holdout_ground_truth.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

KEMPER_REVIEW_CSV = PROJECT_ROOT / "data" / "outputs" / "kemper_ancestry_review.csv"
OUT_CSV = PROJECT_ROOT / "data" / "outputs" / "kemper_holdout_ground_truth.csv"

FLAT8 = {
    "dense_tabular_rows", "genealogy_chart", "handwritten_ledger", "map_land_record",
    "mixed_text_image", "portrait_photo", "printed_document", "website_screenshot",
}

# filename fragment -> corrected category, from the 2026-08-09 manual
# review-pass discussion. 201333 is listed even though unchanged, purely
# as an explicit confirmation record. 130926 deliberately absent - left
# undetermined by Jon, excluded below rather than guessed.
CORRECTIONS = {
    "023355": "cemetery_photo",
    "190703": "cemetery_photo",
    "023938": "cemetery_photo",
    "203602": "website_screenshot",
    "204419": "website_screenshot",
    "204608": "website_screenshot",
    "205033": "website_screenshot",
    "205053": "website_screenshot",
    "205121": "website_screenshot",
    "013324": "website_screenshot",
    "030205": "website_screenshot",
    "010411": "website_screenshot",
    "204258": "website_screenshot",
    "230428": "website_screenshot",
    "231440": "website_screenshot",
    "231806": "website_screenshot",
    "150346": "mixed_text_image",
    "001046": "photo_collage",
    "201333": "printed_document",  # confirmed unchanged
}
UNDETERMINED_FRAGMENTS = {"130926"}


def main():
    with open(KEMPER_REVIEW_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    out_rows = []
    n_corrected, n_excluded_undetermined, n_excluded_not_flat8 = 0, 0, 0
    for r in rows:
        path = r["file_path"]
        stem = Path(path).stem

        if any(frag in stem for frag in UNDETERMINED_FRAGMENTS):
            n_excluded_undetermined += 1
            continue

        gt_category = r["category"]
        for frag, corrected in CORRECTIONS.items():
            if frag in stem:
                gt_category = corrected
                if corrected != r["category"]:
                    n_corrected += 1
                break

        if gt_category not in FLAT8:
            n_excluded_not_flat8 += 1
            continue

        out_rows.append({
            "file_path": path,
            "gt": gt_category,
            "gemma_gen_original_pred": r["category"],
            "gemma_gen_confidence": r["confidence"],
            "was_corrected": gt_category != r["category"],
        })

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Source: {len(rows)} images in {KEMPER_REVIEW_CSV.name}")
    print(f"Excluded (undetermined, e.g. 130926): {n_excluded_undetermined}")
    print(f"Excluded (ground truth outside flat-8, e.g. cemetery_photo/photo_collage): {n_excluded_not_flat8}")
    print(f"Corrected from original prediction: {n_corrected}")
    print(f"Final flat-8 holdout set: {len(out_rows)} images")
    print(f"Written to {OUT_CSV}")

    from collections import Counter
    counts = Counter(r["gt"] for r in out_rows)
    print("\nClass distribution:")
    for cat, n in counts.most_common():
        print(f"  {cat:<20s} {n}")


if __name__ == "__main__":
    main()
