"""
Round 13: does the trend-fit-and-interpolate calibration strategy
(detect_column_number_centers_trend(), core/row_segmentation.py) match
or beat Round 11's best piecewise results (73% Sample A / 100% Sample
B) WITHOUT re-invoking the unstable narrow-window scorer that caused
Round 12's regression?

Same two real 1901-family samples as Round 11 (see that round's own
caveat: Sample B is a sparse 5-column spot-check, not a full layout -
there is no second full-layout 1901 ground truth in this corpus).

Sweeps n_anchors (how many WIDE, reliable measurements the trend is
fit from) x n_slices (how many finer x-slices the trend is evaluated
at) - independent knobs, unlike the piecewise approach's single
n_zones parameter.

Read-only research script.

Usage:
    python -m diagnostics.test_trend_band_1901_two_sample
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from core.row_segmentation import (
    detect_column_number_centers_trend,
    find_number_row_band,
    nearest_number_candidate,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MATCH_WINDOW_PX = 60
MATCH_QUALITY_MAX_DELTA_PX = 15
N_ANCHORS_SWEEP = (2, 3, 4, 5)
N_SLICES_SWEEP = (5, 10, 15, 20)


@dataclass
class Sample:
    label: str
    image_path: Path
    sidecar_path: Path
    table_left: int
    table_right: int
    band_search_y0: int
    band_search_y1: int


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


SAMPLE_A = Sample(
    label="Sample A: z000077117 (full 33-column layout)",
    image_path=PROJECT_ROOT / "data/outputs/lac_pull_1901_batch1/raw_jpgs/z000077117.jpg",
    sidecar_path=PROJECT_ROOT / "data/outputs/row_segmentation/z000077117_sidecar.json",
    table_left=192, table_right=3283,
    band_search_y0=400, band_search_y1=722,
)

SAMPLE_B = Sample(
    label="Sample B: z000017634 (SPARSE - only 5 spot-checked columns)",
    image_path=PROJECT_ROOT / "data/outputs/reference_pipeline_prerefactor/dewarped/z000017634_dewarped.jpg",
    sidecar_path=PROJECT_ROOT / "data/outputs/reference_pipeline_prerefactor/row_segmentation/z000017634_dewarped_sidecar.json",
    table_left=0, table_right=3105,
    band_search_y0=400, band_search_y1=646,
)


def match_rate(real_cols, centers) -> int:
    n_matched = 0
    for name, x0, x1 in real_cols:
        real_center = (x0 + x1) / 2
        c = nearest_number_candidate(centers, real_center, MATCH_WINDOW_PX)
        if c is not None and abs(c - real_center) <= MATCH_QUALITY_MAX_DELTA_PX:
            n_matched += 1
    return n_matched


def run_sample(sample: Sample) -> None:
    print(f"\n{'=' * 70}\n{sample.label}\n{'=' * 70}")
    real_cols = load_real_columns(sample.sidecar_path)
    n = len(real_cols)
    print(f"{n} real ground-truth columns")

    image = Image.open(sample.image_path)

    band = find_number_row_band(
        image, x0=sample.table_left, x1=sample.table_right,
        search_y0=sample.band_search_y0, search_y1=sample.band_search_y1,
    )
    _by0, _by1, global_blobs, _score = band
    global_centers = [c for c, _w in global_blobs]
    global_matched = match_rate(real_cols, global_centers)
    print(f"Global band: {global_matched}/{n} matched ({100*global_matched/n:.0f}%)\n")

    best = (None, None, -1)
    print(f"{'n_anchors':>10} {'n_slices':>10} {'matched':>10} {'pct':>6}")
    for n_anchors in N_ANCHORS_SWEEP:
        for n_slices in N_SLICES_SWEEP:
            centers = [
                c for c, _w in detect_column_number_centers_trend(
                    image, x0=sample.table_left, x1=sample.table_right,
                    search_y0=sample.band_search_y0, search_y1=sample.band_search_y1,
                    n_anchors=n_anchors, n_slices=n_slices,
                )
            ]
            n_matched = match_rate(real_cols, centers)
            pct = 100 * n_matched / n
            print(f"{n_anchors:>10} {n_slices:>10} {n_matched:>10} {pct:>5.0f}%")
            if n_matched > best[2]:
                best = (n_anchors, n_slices, n_matched)

    print(f"\nBest: n_anchors={best[0]}, n_slices={best[1]} -> "
          f"{best[2]}/{n} matched ({100*best[2]/n:.0f}%), "
          f"+{100*best[2]/n - 100*global_matched/n:.0f}pt over global")


def main() -> None:
    run_sample(SAMPLE_A)
    run_sample(SAMPLE_B)


if __name__ == "__main__":
    main()
