"""
Validation script for core.row_segmentation.detect_column_number_centers()
+ nearest_number_candidate() - the column-axis counterpart to the
row-number-anchor work (detect_row_number_centers()).

Runs against a real 1911 reference page with independently-confirmed
column positions (data/outputs/reference_pipeline_prerefactor/
row_segmentation/e078_e001946617_dewarped_sidecar.json's own
mask_keep_ranges - human-confirmed via ui/row_segmentation_ui.py, not
derived from anything this script is testing) and reports, for each
known column, how close the nearest detected number-blob center lands
to that column's REAL measured center.

This is RESEARCH validation only - detect_column_number_centers() is
NOT wired into core/auto_sidecar.py's production locate_columns(), per
direct instruction ("Do not modify the production pipeline until the
approach has been evaluated"). See docs/COLUMN_NUMBER_ANCHOR_RESEARCH.md
for the full research history and findings this script reproduces.

Usage:
    python -m diagnostics.test_column_number_anchor
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from core.row_segmentation import detect_column_number_centers, nearest_number_candidate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_IMAGE = (
    PROJECT_ROOT / "data" / "outputs" / "reference_pipeline_prerefactor"
    / "dewarped" / "e078_e001946617_dewarped.jpg"
)

# From the reference sidecar's own table_bbox / header_bbox.
TABLE_LEFT, TABLE_RIGHT = 52, 3494
TABLE_TOP = 625

# Number-row band, calibrated by direct visual check against THIS page
# (2026-08-06) - deliberately excludes a dashed sub-header guide-line
# found at y=583-590 in one section of the header; the real digits sit
# at y=592-604. Per-template calibration, not a generic constant - see
# this file's own docstring / detect_column_number_centers()'s.
NUMBER_ROW_Y0, NUMBER_ROW_Y1 = 592, 604

# From the reference sidecar's own mask_keep_ranges (human-confirmed
# ground truth, per-column real pixel ranges) and canada_census_1911.
# yaml's column_regions_approx (the template's own approximate x_frac,
# what a real caller would have available BEFORE this measurement).
KNOWN_COLUMNS = {
    # name: (real_x0, real_x1, template_x_frac0, template_x_frac1)
    "Name": (218, 513, 0.0497, 0.1351),
    "Sex": (677, 722, 0.1828, 0.1958),
    "Relationship to Head": (722, 851, 0.1960, 0.2335),
    "Age": (1074, 1117, 0.2977, 0.3104),
    "Birthplace": (1120, 1249, 0.3103, 0.3479),
}

SEARCH_WINDOW_PX = 60  # tolerance around the template's own approx position


def main() -> None:
    image = Image.open(REFERENCE_IMAGE)
    print(f"Loaded {REFERENCE_IMAGE.name} ({image.size})")

    blobs = detect_column_number_centers(
        image, x0=TABLE_LEFT, x1=TABLE_RIGHT, y0=NUMBER_ROW_Y0, y1=NUMBER_ROW_Y1,
    )
    centers = [c for c, _w in blobs]
    print(f"{len(blobs)} raw number-blobs detected in the header band\n")

    table_width = TABLE_RIGHT - TABLE_LEFT
    deltas = []
    print(f"{'Column':<24} {'real_center':>12} {'template_approx':>16} {'chosen_blob':>12} {'delta':>8}")
    for name, (real_x0, real_x1, xf0, xf1) in KNOWN_COLUMNS.items():
        real_center = (real_x0 + real_x1) / 2
        template_approx = TABLE_LEFT + (xf0 + xf1) / 2 * table_width
        chosen = nearest_number_candidate(centers, template_approx, SEARCH_WINDOW_PX)
        if chosen is None:
            print(f"{name:<24} {real_center:>12.1f} {template_approx:>16.1f} {'NONE FOUND':>12} {'-':>8}")
            continue
        delta = abs(chosen - real_center)
        deltas.append(delta)
        print(f"{name:<24} {real_center:>12.1f} {template_approx:>16.1f} {chosen:>12.1f} {delta:>8.1f}")

    if deltas:
        print(f"\n{len(deltas)}/{len(KNOWN_COLUMNS)} columns matched. "
              f"delta stats: min={min(deltas):.1f} max={max(deltas):.1f} "
              f"mean={sum(deltas)/len(deltas):.1f}")


if __name__ == "__main__":
    main()
