"""
Round 11: does piecewise/local y-band calibration (Round 10) help
consistently across DIFFERENT 1901 pages, or was n_zones=10's 73% on
z000077117 a one-page fluke? Direct test of Jon's framing: "the most
important metric isn't which N wins, it's does local calibration
CONSISTENTLY improve recall across different 1901 pages."

Sample A: z000077117 - the full 33-column, human-confirmed reference
used throughout this research (Rounds 2-10).

Sample B: z000017634 - the ONLY other 1901-family page in this corpus
with any real, human-confirmed column ground truth. Important caveat,
reported honestly rather than glossed over: it only has 5 SPARSE
spot-checked columns (Name/Sex/Relationship to Head/Age/Birthplace),
not a full column_order like sample A - there is no second full-layout
1901 ground truth sample in this corpus. This reduces statistical power
for sample B (5 data points, not 33) but is the best real (not
invented) comparison available without new manual annotation work.

For each sample: global-band baseline, then the n_zones=2..16 sweep,
reporting best n_zones, MEDIAN gain over global across the whole sweep
(not just the best point - a fairer "does this generally help" signal
than cherry-picking the best score), stability (std dev across the
sweep), and which real columns are still missed at the best setting.

Read-only research script.

Usage:
    python -m diagnostics.test_piecewise_band_1901_two_sample
"""
from __future__ import annotations

import json
import statistics as stats
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from core.row_segmentation import (
    detect_column_number_centers_piecewise,
    find_number_row_band,
    nearest_number_candidate,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MATCH_WINDOW_PX = 60
MATCH_QUALITY_MAX_DELTA_PX = 15
ZONE_RANGE = range(2, 17)


@dataclass
class Sample:
    label: str
    image_path: Path
    sidecar_path: Path | None
    real_columns: list[tuple[str, float, float]] | None  # set directly if no full sidecar
    table_left: int
    table_right: int
    band_search_y0: int
    band_search_y1: int


def load_real_columns_from_sidecar(sidecar_path: Path) -> list[tuple[str, float, float]]:
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
    real_columns=None,
    table_left=192, table_right=3283,
    band_search_y0=400, band_search_y1=722,
)

SAMPLE_B = Sample(
    label="Sample B: z000017634 (SPARSE - only 5 spot-checked columns, not full layout)",
    image_path=PROJECT_ROOT / "data/outputs/reference_pipeline_prerefactor/dewarped/z000017634_dewarped.jpg",
    sidecar_path=PROJECT_ROOT / "data/outputs/reference_pipeline_prerefactor/row_segmentation/z000017634_dewarped_sidecar.json",
    real_columns=None,
    table_left=0, table_right=3105,
    band_search_y0=400, band_search_y1=646,
)


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


def run_sample(sample: Sample) -> dict:
    print(f"\n{'=' * 70}\n{sample.label}\n{'=' * 70}")
    if not sample.image_path.exists():
        print("  SKIPPED - image not found")
        return {}

    real_cols = sample.real_columns or load_real_columns_from_sidecar(sample.sidecar_path)
    n = len(real_cols)
    print(f"{n} real ground-truth columns")

    image = Image.open(sample.image_path)

    band = find_number_row_band(
        image, x0=sample.table_left, x1=sample.table_right,
        search_y0=sample.band_search_y0, search_y1=sample.band_search_y1,
    )
    if band is None:
        print("  SKIPPED - no usable global band found")
        return {}
    _by0, _by1, global_blobs, _score = band
    global_centers = [c for c, _w in global_blobs]
    global_matched, global_misses = match_rate(real_cols, global_centers)
    global_pct = 100 * global_matched / n
    print(f"Global band: {global_matched}/{n} matched ({global_pct:.0f}%)")

    sweep_pct = {}
    sweep_misses = {}
    for n_zones in ZONE_RANGE:
        centers = [
            c for c, _w in detect_column_number_centers_piecewise(
                image, x0=sample.table_left, x1=sample.table_right,
                search_y0=sample.band_search_y0, search_y1=sample.band_search_y1,
                n_zones=n_zones,
            )
        ]
        n_matched, misses = match_rate(real_cols, centers)
        pct = 100 * n_matched / n
        sweep_pct[n_zones] = pct
        sweep_misses[n_zones] = misses
        print(f"  n_zones={n_zones:>2}: {n_matched}/{n} matched ({pct:.0f}%)")

    gains = [p - global_pct for p in sweep_pct.values()]
    best_n = max(sweep_pct, key=lambda k: sweep_pct[k])
    median_gain = stats.median(gains)
    frac_positive = sum(1 for g in gains if g > 0) / len(gains)
    stdev_pct = stats.pstdev(sweep_pct.values())

    print(f"\nBest: n_zones={best_n} ({sweep_pct[best_n]:.0f}%, "
          f"+{sweep_pct[best_n]-global_pct:.0f}pt over global)")
    print(f"Median gain over global across n_zones=2..16: {median_gain:+.1f}pt")
    print(f"Fraction of n_zones settings that beat global: {100*frac_positive:.0f}%")
    print(f"Stdev of match% across the sweep (stability): {stdev_pct:.1f}pt")
    print(f"Misses at best n_zones: {sweep_misses[best_n]}")
    print(f"Misses at global band:  {global_misses}")

    return {
        "label": sample.label, "n": n, "global_pct": global_pct,
        "best_n": best_n, "best_pct": sweep_pct[best_n],
        "median_gain": median_gain, "frac_positive": frac_positive,
        "stdev_pct": stdev_pct, "sweep_pct": sweep_pct,
    }


def main() -> None:
    result_a = run_sample(SAMPLE_A)
    result_b = run_sample(SAMPLE_B)

    print(f"\n{'=' * 70}\nCOMPARISON\n{'=' * 70}")
    for r in (result_a, result_b):
        if not r:
            continue
        print(f"{r['label']}")
        print(f"  global={r['global_pct']:.0f}%  best(n={r['best_n']})={r['best_pct']:.0f}%  "
              f"median_gain={r['median_gain']:+.1f}pt  "
              f"beat_global={100*r['frac_positive']:.0f}% of settings  "
              f"stdev={r['stdev_pct']:.1f}pt")

    if result_a and result_b:
        both_positive = result_a["median_gain"] > 0 and result_b["median_gain"] > 0
        print(f"\nDoes local calibration consistently help across both samples? "
              f"{'YES - median gain positive on both' if both_positive else 'NOT CONSISTENT'}")


if __name__ == "__main__":
    main()
