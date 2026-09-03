"""
Round 16: validate the crease-based dewarp (Round 14's fixed pipeline -
detect_page_crease_x + dewarp_page_at_crease, n_anchors=3,
INTER_NEAREST) against the other 5 confirmed-crease 1901 pages
(z000077122, z000077130, z000077132, z000077134, z000077145) - none of
which have full-column ground truth like z000077117, so this can't
report a match-rate percentage. Instead uses two ground-truth-free
signals already established as legitimate confidence indicators
throughout this investigation: find_number_row_band()'s own
plausibility SCORE (_score_number_row_band(), which rewards a plausible
blob count and digit-like widths - see Round 2's own docstring) and raw
blob count, before vs. after dewarp - plus a saved before/after crop
pair for direct visual confirmation, since Round 14 already showed a
metric moving the right way is not sufficient proof on its own.

Read-only except for the small before/after preview crops it saves.

Usage:
    python -m diagnostics.test_page_dewarp_multi_1901
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from core.page_dewarp import detect_page_crease_x, dewarp_page_at_crease
from core.row_segmentation import (
    _score_number_row_band,
    detect_column_number_centers_piecewise,
    find_number_row_band,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data/outputs/lac_pull_1901_batch1/raw_jpgs"
OUT_DIR = PROJECT_ROOT / "data/outputs/lac_pull_1901_batch1/_dewarp_validation"

SAMPLES = [
    {
        "label": "z000077122", "table_left": 190, "table_right": 3274,
        "band_search_y0": 400, "band_search_y1": 750, "crease_search_x": (1650, 1780),
    },
    {
        "label": "z000077130", "table_left": 190, "table_right": 3446,
        "band_search_y0": 400, "band_search_y1": 850, "crease_search_x": (1700, 1900),
    },
    {
        "label": "z000077132", "table_left": 190, "table_right": 3282,
        "band_search_y0": 400, "band_search_y1": 750, "crease_search_x": (1650, 1780),
    },
    {
        "label": "z000077134", "table_left": 190, "table_right": 3274,
        "band_search_y0": 380, "band_search_y1": 750, "crease_search_x": (1650, 1780),
    },
    {
        "label": "z000077145", "table_left": 190, "table_right": 3282,
        "band_search_y0": 400, "band_search_y1": 750, "crease_search_x": (1650, 1780),
    },
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for sample in SAMPLES:
        label = sample["label"]
        print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
        image_path = RAW_DIR / f"{label}.jpg"
        if not image_path.exists():
            print("  SKIPPED - image not found")
            continue
        image = Image.open(image_path)

        band_before = find_number_row_band(
            image, x0=sample["table_left"], x1=sample["table_right"],
            search_y0=sample["band_search_y0"], search_y1=sample["band_search_y1"],
        )
        if band_before is None:
            print("  SKIPPED - no usable band found before dewarp")
            continue
        by0, by1, blobs_before, score_before = band_before
        print(f"BEFORE: band=({by0},{by1}) score={score_before:.2f} n_blobs={len(blobs_before)}")

        crease = detect_page_crease_x(image, *sample["crease_search_x"])
        if crease is None:
            print("  NO CREASE DETECTED in expected window - skipping dewarp")
            continue
        crease_x, coverage, ratio = crease
        print(f"Crease detected at x={crease_x} (coverage={coverage:.2f}, ratio={ratio:.2f})")

        dewarped, debug = dewarp_page_at_crease(
            image, crease_x, table_left=sample["table_left"], table_right=sample["table_right"],
            search_y0=sample["band_search_y0"], search_y1=sample["band_search_y1"],
        )
        print(f"Dewarp anchors={[(round(x,1), round(y,1)) for x,y in debug['anchors']]}, "
              f"max_shift={debug['max_shift_up']:.1f}px")

        band_after = find_number_row_band(
            dewarped, x0=sample["table_left"], x1=sample["table_right"],
            search_y0=sample["band_search_y0"], search_y1=sample["band_search_y1"],
        )
        if band_after is None:
            print("AFTER: no usable band found")
            continue
        ay0, ay1, blobs_after, score_after = band_after
        print(f"AFTER:  band=({ay0},{ay1}) score={score_after:.2f} n_blobs={len(blobs_after)}")

        centers_pw = [
            c for c, _w in detect_column_number_centers_piecewise(
                dewarped, x0=sample["table_left"], x1=sample["table_right"],
                search_y0=sample["band_search_y0"], search_y1=sample["band_search_y1"], n_zones=3,
            )
        ]
        pw_score = _score_number_row_band([(c, 10) for c in centers_pw], sample["table_right"] - sample["table_left"])
        print(f"AFTER + piecewise n_zones=3: {len(centers_pw)} blobs (informational only, "
              f"score not directly comparable - different band per zone)")

        before_path = OUT_DIR / f"{label}_before.jpg"
        after_path = OUT_DIR / f"{label}_after.jpg"
        image.save(before_path, quality=85)
        dewarped.save(after_path)
        print(f"Saved {before_path.name} / {after_path.name}")


if __name__ == "__main__":
    main()
