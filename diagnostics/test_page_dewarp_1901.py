"""
Round 14: does dewarping at the confirmed physical crease let the
ORIGINAL, simple global-band approach work on 1901 - without any of
the calibration workarounds (piecewise grids, scorer tuning, trend
interpolation) tried and only partially successful in Rounds 9-13?

Since dewarp_page_at_crease() only shifts pixels VERTICALLY per column,
every real column's x0/x1 ground truth from the sidecar stays valid
unchanged - no remapping needed for the match-rate check itself.

Read-only except for the corrected image it produces in-memory (not
written back over the source file).

Usage:
    python -m diagnostics.test_page_dewarp_1901
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from core.page_dewarp import detect_page_crease_x, dewarp_page_at_crease
from core.row_segmentation import find_number_row_band, nearest_number_candidate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MATCH_WINDOW_PX = 60
MATCH_QUALITY_MAX_DELTA_PX = 15

SAMPLES = [
    {
        "label": "z000077117",
        "image_path": PROJECT_ROOT / "data/outputs/lac_pull_1901_batch1/raw_jpgs/z000077117.jpg",
        "sidecar_path": PROJECT_ROOT / "data/outputs/row_segmentation/z000077117_sidecar.json",
        "table_left": 192, "table_right": 3283,
        "band_search_y0": 400, "band_search_y1": 722,
        "crease_search_x": (1650, 1780),
    },
]


def load_real_columns(sidecar_path: Path) -> list[tuple[str, float, float]]:
    data = json.loads(sidecar_path.read_text(encoding="utf-8"))
    out = []
    for name, col in data["columns"].items():
        ranges = col.get("mask_keep_ranges") or []
        if not ranges:
            continue
        x0 = min(r[0] for r in ranges)
        x1 = max(r[1] for r in ranges)
        out.append((name, x0, x1))
    return sorted(out, key=lambda t: t[1])


def match_rate(real_cols, centers) -> tuple[int, list[str]]:
    n_matched = 0
    misses = []
    for name, x0, x1 in real_cols:
        real_center = (x0 + x1) / 2
        c = nearest_number_candidate(centers, real_center, MATCH_WINDOW_PX)
        if c is not None and abs(c - real_center) <= MATCH_QUALITY_MAX_DELTA_PX:
            n_matched += 1
        else:
            misses.append(name)
    return n_matched, misses


def main() -> None:
    for sample in SAMPLES:
        print(f"\n{'=' * 70}\n{sample['label']}\n{'=' * 70}")
        image = Image.open(sample["image_path"])
        real_cols = load_real_columns(sample["sidecar_path"])
        n = len(real_cols)
        print(f"{n} real ground-truth columns")

        crease = detect_page_crease_x(image, *sample["crease_search_x"])
        if crease is None:
            print("  NO CREASE DETECTED - skipping")
            continue
        crease_x, coverage, ratio = crease
        print(f"Detected crease at x={crease_x} (coverage={coverage:.2f}, ratio={ratio:.2f})")

        # Baseline (before dewarp) - Round 4/9's known 39% result, reproduced here for a fair side-by-side.
        band_before = find_number_row_band(
            image, x0=sample["table_left"], x1=sample["table_right"],
            search_y0=sample["band_search_y0"], search_y1=sample["band_search_y1"],
        )
        _by0, _by1, blobs_before, _score = band_before
        centers_before = [c for c, _w in blobs_before]
        matched_before, misses_before = match_rate(real_cols, centers_before)
        print(f"\nBEFORE dewarp: {matched_before}/{n} matched ({100*matched_before/n:.0f}%)")

        dewarped, debug = dewarp_page_at_crease(
            image, crease_x,
            table_left=sample["table_left"], table_right=sample["table_right"],
            search_y0=sample["band_search_y0"], search_y1=sample["band_search_y1"],
        )
        print(f"Dewarp debug: target_y={debug['target_y']:.1f}, "
              f"max_shift={debug['max_shift_up']:.1f}px")
        print(f"  anchors={[(round(x,1), round(y,1)) for x,y in debug['anchors']]}")

        band_after = find_number_row_band(
            dewarped, x0=sample["table_left"], x1=sample["table_right"],
            search_y0=sample["band_search_y0"], search_y1=sample["band_search_y1"],
        )
        if band_after is None:
            print("AFTER dewarp: no usable band found")
            continue
        _ay0, _ay1, blobs_after, _ascore = band_after
        centers_after = [c for c, _w in blobs_after]
        matched_after, misses_after = match_rate(real_cols, centers_after)
        print(f"AFTER dewarp:  {matched_after}/{n} matched ({100*matched_after/n:.0f}%)")
        print(f"  still missing: {misses_after}")

        out_path = sample["image_path"].parent / f"{sample['label']}_dewarped_preview.png"
        dewarped.save(out_path)
        print(f"\nSaved dewarped preview to {out_path}")


if __name__ == "__main__":
    main()
