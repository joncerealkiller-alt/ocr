"""
Root-cause WHY detect_column_number_centers() + nearest_number_candidate()
only confidently matches 13/33 (39%) of 1901's real columns (z000077117),
vs. 1926/1931's ~50-98%. Direct response to Jon's instruction: don't
assume header OCR is the answer - the blob-and-filter sensor already
works well elsewhere, so first find out WHERE and WHY it specifically
fails on this page, checking (in order): local y-drift across the
header row, irregular blob size/shape, touching digits, nearby header-
text contamination, edge warp, filtering thresholds, and the expected-
position matching itself.

Read-only diagnostic. Does not modify core/row_segmentation.py.

Usage:
    python -m diagnostics.diagnose_1901_column_match_failures
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from core.row_segmentation import (
    detect_column_number_centers,
    find_number_row_band,
    nearest_number_candidate,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH = PROJECT_ROOT / "data/outputs/lac_pull_1901_batch1/raw_jpgs/z000077117.jpg"
SIDECAR_PATH = PROJECT_ROOT / "data/outputs/row_segmentation/z000077117_sidecar.json"
TABLE_LEFT, TABLE_RIGHT = 192, 3283
GLOBAL_BAND_SEARCH_Y0, GLOBAL_BAND_SEARCH_Y1 = 400, 722

WIDE_WINDOW_PX = 100  # generous - distinguishes "no candidate nearby at all" from "gated"
MATCH_QUALITY_MAX_DELTA_PX = 15  # same gate used in the projection harness


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


def check_y_drift(image: Image.Image) -> None:
    """Does the optimal number-row band shift across the table's width?
    A single global band (found on the FULL width) could be a compromise
    that's locally wrong in some zones if the header isn't perfectly
    level/flat - print skew, scan curvature, or camera-angle warp."""
    print(f"\n{'=' * 70}\nY-DRIFT CHECK (independent band search per horizontal third)\n{'=' * 70}")
    third = (TABLE_RIGHT - TABLE_LEFT) // 3
    zones = [
        ("left third", TABLE_LEFT, TABLE_LEFT + third),
        ("middle third", TABLE_LEFT + third, TABLE_LEFT + 2 * third),
        ("right third", TABLE_LEFT + 2 * third, TABLE_RIGHT),
    ]
    for label, x0, x1 in zones:
        band = find_number_row_band(
            image, x0=x0, x1=x1, search_y0=GLOBAL_BAND_SEARCH_Y0, search_y1=GLOBAL_BAND_SEARCH_Y1,
        )
        if band is None:
            print(f"  {label:<14} (x={x0}-{x1}): no usable band found")
            continue
        by0, by1, blobs, score = band
        print(f"  {label:<14} (x={x0}-{x1}): band=({by0},{by1}) score={score:.2f} n_blobs={len(blobs)}")


def classify_and_report(real_cols: list[tuple[str, float, float]], blobs: list[tuple[float, int]]) -> None:
    print(f"\n{'=' * 70}\nPER-COLUMN DIAGNOSIS (global band, wide {WIDE_WINDOW_PX}px search)\n{'=' * 70}")
    all_centers = [c for c, _w in blobs]

    matched_widths, miss_nearby_widths = [], []
    zone_stats = {"left edge": [0, 0], "middle": [0, 0], "right edge": [0, 0]}  # [matched, total]
    n = len(real_cols)

    for idx, (name, x0, x1) in enumerate(real_cols):
        real_center = (x0 + x1) / 2
        # widen search so we can tell "nothing nearby" from "something
        # nearby but too far/wrong" - production uses 60px + a 15px gate
        best = None
        best_delta = None
        for c, w in blobs:
            d = abs(c - real_center)
            if best_delta is None or d < best_delta:
                best_delta, best, best_w = d, c, w

        if best is None:
            status = "NO_BLOBS_ON_PAGE"
        elif best_delta <= MATCH_QUALITY_MAX_DELTA_PX:
            status = "MATCHED"
            matched_widths.append(best_w)
        elif best_delta <= WIDE_WINDOW_PX:
            status = "NEAR_MISS (blob exists but off / wrong)"
            miss_nearby_widths.append(best_w)
        else:
            status = "FAR_OR_ABSENT (nothing within window)"

        zone = "left edge" if idx < n * 0.15 else ("right edge" if idx >= n * 0.85 else "middle")
        zone_stats[zone][1] += 1
        if status == "MATCHED":
            zone_stats[zone][0] += 1

        width_str = f"w={best_w}px" if best is not None else "w=-"
        print(f"  [{idx:>2}] {name:<40} real={real_center:>7.1f} "
              f"nearest={best if best is not None else -1:>7.1f} delta={best_delta if best is not None else -1:>6.1f} "
              f"{width_str:<8} {status}")

    print(f"\nMatched blob widths: {matched_widths}")
    print(f"Near-miss (wrong/off) blob widths: {miss_nearby_widths}")
    if matched_widths:
        print(f"  matched mean width={sum(matched_widths)/len(matched_widths):.1f}px")
    if miss_nearby_widths:
        print(f"  near-miss mean width={sum(miss_nearby_widths)/len(miss_nearby_widths):.1f}px")

    print("\nZone breakdown (matched/total):")
    for zone, (m, t) in zone_stats.items():
        print(f"  {zone:<12}: {m}/{t} ({100*m/t:.0f}%)" if t else f"  {zone}: n/a")


def check_filter_sensitivity(image: Image.Image, band_y0: int, band_y1: int) -> None:
    """Are the default min_blob_width_px/line_run_threshold_frac eating
    real digit blobs on this page? Compare blob counts with looser
    settings against the production defaults."""
    print(f"\n{'=' * 70}\nFILTER THRESHOLD SENSITIVITY\n{'=' * 70}")
    configs = [
        ("production defaults", dict()),
        ("looser min_blob_width_px=1", dict(min_blob_width_px=1)),
        ("looser line_run_threshold_frac=0.95", dict(line_run_threshold_frac=0.95)),
        ("both loosened", dict(min_blob_width_px=1, line_run_threshold_frac=0.95)),
    ]
    for label, kwargs in configs:
        blobs = detect_column_number_centers(image, x0=TABLE_LEFT, x1=TABLE_RIGHT, y0=band_y0, y1=band_y1, **kwargs)
        print(f"  {label:<40}: {len(blobs)} blobs")


def main() -> None:
    image = Image.open(IMAGE_PATH)
    real_cols = load_real_columns()
    print(f"{len(real_cols)} real ground-truth columns loaded")

    check_y_drift(image)

    band = find_number_row_band(
        image, x0=TABLE_LEFT, x1=TABLE_RIGHT, search_y0=GLOBAL_BAND_SEARCH_Y0, search_y1=GLOBAL_BAND_SEARCH_Y1,
    )
    band_y0, band_y1, blobs, score = band
    print(f"\nGlobal auto-calibrated band: ({band_y0},{band_y1}), score={score:.2f}, {len(blobs)} blobs")

    classify_and_report(real_cols, blobs)
    check_filter_sensitivity(image, band_y0, band_y1)


if __name__ == "__main__":
    main()
