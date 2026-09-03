"""
Round 10: does per-zone (piecewise) y-band calibration recover 1901's
column-number match rate? Direct test of the fix proposed at the end of
Round 9 (docs/COLUMN_NUMBER_ANCHOR_RESEARCH.md) after finding the
header row isn't level on this page (~80px/~1.5deg drift left to
right, visually confirmed).

Compares the OLD global-band match rate against
detect_column_number_centers_piecewise() (core/row_segmentation.py) at
a few n_zones values, using the same matching/quality-gate logic as the
projection harness (diagnostics/test_column_boundary_projection_
multiyear.py) - so this result is directly comparable to Round 4's
13/33 (39%) baseline.

Read-only research script.

Usage:
    python -m diagnostics.test_piecewise_band_1901
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from core.row_segmentation import (
    detect_column_number_centers_piecewise,
    find_number_row_band,
    nearest_number_candidate,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH = PROJECT_ROOT / "data/outputs/lac_pull_1901_batch1/raw_jpgs/z000077117.jpg"
SIDECAR_PATH = PROJECT_ROOT / "data/outputs/row_segmentation/z000077117_sidecar.json"
TABLE_LEFT, TABLE_RIGHT = 192, 3283
BAND_SEARCH_Y0, BAND_SEARCH_Y1 = 400, 722
MATCH_WINDOW_PX = 60
MATCH_QUALITY_MAX_DELTA_PX = 15


def load_real_columns() -> list[tuple[str, float, float]]:
    data = json.loads(SIDECAR_PATH.read_text(encoding="utf-8"))
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
    image = Image.open(IMAGE_PATH)
    real_cols = load_real_columns()
    n = len(real_cols)
    print(f"{n} real ground-truth columns loaded\n")

    # Baseline: old global-band approach (Round 4's 13/33).
    band = find_number_row_band(image, x0=TABLE_LEFT, x1=TABLE_RIGHT, search_y0=BAND_SEARCH_Y0, search_y1=BAND_SEARCH_Y1)
    _by0, _by1, global_blobs, _score = band
    global_centers = [c for c, _w in global_blobs]
    n_matched, misses = match_rate(real_cols, global_centers)
    print(f"Global band (old):  {n_matched}/{n} matched ({100*n_matched/n:.0f}%), {len(global_blobs)} blobs")

    best_n, best_matched, best_misses = None, -1, []
    for n_zones in range(2, 17):
        centers = [
            c for c, _w in detect_column_number_centers_piecewise(
                image, x0=TABLE_LEFT, x1=TABLE_RIGHT, search_y0=BAND_SEARCH_Y0, search_y1=BAND_SEARCH_Y1,
                n_zones=n_zones,
            )
        ]
        n_matched, misses = match_rate(real_cols, centers)
        print(f"Piecewise n_zones={n_zones:>2}: {n_matched}/{n} matched ({100*n_matched/n:.0f}%), {len(centers)} blobs")
        if n_matched > best_matched:
            best_n, best_matched, best_misses = n_zones, n_matched, misses

    print(f"\nBest fixed-zone: n_zones={best_n}, {best_matched}/{n} matched ({100*best_matched/n:.0f}%)")
    print(f"Still missing at best n_zones: {best_misses}")

    # Fixed disjoint zones create a hard boundary - a column sitting
    # near a zone edge can land in whichever zone's compromise band is
    # worse, which is exactly the non-monotonic noise seen above (36%
    # to 73% depending on n_zones, no clean trend). Test a smoother
    # alternative: search a LOCAL window centered on each column's own
    # expected position, no shared grid, no boundary effect.
    print(f"\n{'=' * 70}\nLocal per-column window search (no fixed grid)\n{'=' * 70}")
    for window_px in (200, 300, 400, 600, 800):
        n_matched = 0
        for name, x0, x1 in real_cols:
            real_center = (x0 + x1) / 2
            lo = max(TABLE_LEFT, int(real_center - window_px))
            hi = min(TABLE_RIGHT, int(real_center + window_px))
            band = find_number_row_band(image, x0=lo, x1=hi, search_y0=BAND_SEARCH_Y0, search_y1=BAND_SEARCH_Y1)
            if band is None:
                continue
            _by0, _by1, blobs, _score = band
            centers = [c for c, _w in blobs]
            c = nearest_number_candidate(centers, real_center, MATCH_WINDOW_PX)
            if c is not None and abs(c - real_center) <= MATCH_QUALITY_MAX_DELTA_PX:
                n_matched += 1
        print(f"window=+/-{window_px}px: {n_matched}/{n} matched ({100*n_matched/n:.0f}%)")


if __name__ == "__main__":
    main()
