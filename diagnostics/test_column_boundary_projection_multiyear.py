"""
Multi-year generalization test for the column-boundary PROJECTION
prototype (see diagnostics/test_column_boundary_projection.py for the
original single-sample 1931 validation and docs/COLUMN_NUMBER_ANCHOR_
RESEARCH.md's "Round 3" for the full write-up of that result).

Runs the SAME pipeline - auto-calibrated y-band -> detect_column_number_
centers() -> quality-gated match against real ground truth -> Variant A
(pure projection) / Variant B (detector-corrected) -> corridor capture
rate - across multiple real, independently-confirmed census years, to
see whether the approach generalizes or was tuned to the one 1931 sample
tested so far.

Two tiers of real reference data, both genuine (human-confirmed via
ui/row_segmentation_ui.py, not derived from anything tested here) but of
different completeness:

  FULL-LAYOUT samples (every real column in the table is known, so the
  recursive projection chain can be tested end-to-end): 1901, 1911,
  1926, 1931.

  SPARSE SPOT-CHECK samples (only ~5 non-contiguous columns are known,
  e.g. Name/Age/Sex/Relationship to Head/Birthplace) - NOT enough to
  chain a projection (the formula needs every column between two known
  points), so these only get the simpler nearest-candidate accuracy
  check from Round 2, run here for one additional year (1921) that has
  no full-layout sample available yet. Reported separately, not mixed
  into the projection metrics.

Read-only research script. Does NOT modify core/auto_sidecar.py's
production locate_columns().

Usage:
    python -m diagnostics.test_column_boundary_projection_multiyear
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from diagnostics.error_analysis.common import LEGACY_WORKING_IMAGES_DIR
from core.row_segmentation import (
    detect_column_number_centers,
    find_number_row_band,
    nearest_number_candidate,
)
from diagnostics.test_line_detection_enhancement import find_vertical_line_via_gradient

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MATCH_WINDOW_PX = 60
MATCH_QUALITY_MAX_DELTA_PX = 15
CORRIDOR_WIDTHS = [5, 10, 20, 30, 50]


@dataclass
class FullLayoutSample:
    year: str
    label: str
    image_path: Path
    sidecar_path: Path
    table_left: int
    table_right: int
    table_top: int
    table_bottom: int
    band_search_y0: int
    band_search_y1: int


@dataclass
class SpotCheckSample:
    year: str
    label: str
    image_path: Path
    known_columns: dict  # name -> (x0, x1)
    table_left: int
    table_right: int
    band_search_y0: int
    band_search_y1: int


FULL_LAYOUT_SAMPLES = [
    FullLayoutSample(
        year="1901", label="z000077117",
        image_path=PROJECT_ROOT / "data/outputs/lac_pull_1901_batch1/raw_jpgs/z000077117.jpg",
        sidecar_path=PROJECT_ROOT / "data/outputs/row_segmentation/z000077117_sidecar.json",
        table_left=192, table_right=3283, table_top=722, table_bottom=2564,
        band_search_y0=400, band_search_y1=722,
    ),
    FullLayoutSample(
        year="1926", label="e001926997",
        image_path=LEGACY_WORKING_IMAGES_DIR / "e001926997.png",
        sidecar_path=PROJECT_ROOT / "data/outputs/reference_pipeline_prerefactor/row_segmentation/e001926997_sidecar.json",
        table_left=413, table_right=7312, table_top=1408, table_bottom=4308,
        band_search_y0=1100, band_search_y1=1408,
    ),
    FullLayoutSample(
        year="1931", label="e011717826",
        image_path=PROJECT_ROOT / "data/outputs/lac_pull_1931_batch1/raw_jpgs/e011717826.jpg",
        sidecar_path=PROJECT_ROOT / "data/outputs/row_segmentation/e011717826_sidecar.json",
        table_left=173, table_right=4028, table_top=690, table_bottom=2400,
        band_search_y0=550, band_search_y1=690,
    ),
]

SPOT_CHECK_SAMPLES = [
    SpotCheckSample(
        year="1921", label="1921_022-E002880409",
        image_path=PROJECT_ROOT / "data/outputs/reference_pipeline_prerefactor/dewarped/1921_022-E002880409_dewarped.jpg",
        known_columns={
            "Relationship to Head": (1033, 1154),
            "Sex": (1155, 1198),
            "Age": (1241, 1287),
            "Birthplace": (1287, 1438),
        },
        table_left=21, table_right=3235,
        band_search_y0=475, band_search_y1=638,
    ),
    SpotCheckSample(
        year="1931b", label="1931_174-e011707164",
        image_path=PROJECT_ROOT / "data/outputs/reference_pipeline_prerefactor/dewarped/1931_174-e011707164_dewarped.jpg",
        known_columns={
            "Relationship to Head": (1124, 1246),
            "Sex": (1247, 1286),
            "Age": (1324, 1375),
            "Birthplace": (1377, 1567),
        },
        table_left=73, table_right=3930,
        band_search_y0=562 - 163, band_search_y1=562,
    ),
    # Only 5 columns are stored in this sidecar (spot-checked, not the
    # full column_order) - moved here from FULL_LAYOUT_SAMPLES after
    # the first multiyear run silently bridged the real, unlisted
    # columns between "Relationship to Head" and "Birthplace" as if
    # they were adjacent, producing a bogus 226px "failure". Real bug
    # in the harness, not in the projection technique - see
    # docs/COLUMN_NUMBER_ANCHOR_RESEARCH.md's Round 4 for the writeup.
    SpotCheckSample(
        year="1911", label="e078_e001946617",
        image_path=PROJECT_ROOT / "data/outputs/reference_pipeline_prerefactor/dewarped/e078_e001946617_dewarped.jpg",
        known_columns={
            "Name": (218, 513),
            "Sex": (677, 722),
            "Relationship to Head": (722, 851),
            "Age": (1074, 1117),
            "Birthplace": (1120, 1249),
        },
        table_left=52, table_right=3494,
        band_search_y0=500, band_search_y1=625,
    ),
]


def load_real_boundaries(sidecar_path: Path) -> list[tuple[str, float, float]]:
    """[(column_name, x0, x1), ...] sorted left to right, "done"/populated
    columns only - a column with no mask_keep_ranges (e.g. an in_progress
    placeholder) contributes nothing real to check against, so it is
    dropped rather than treated as a (fake) zero-width column."""
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


def find_clean_runs(matched_centers: list[float | None]) -> list[tuple[int, int]]:
    runs = []
    start = None
    for i, c in enumerate(matched_centers):
        if c is not None:
            if start is None:
                start = i
        else:
            if start is not None:
                runs.append((start, i - 1))
                start = None
    if start is not None:
        runs.append((start, len(matched_centers) - 1))
    return runs


def run_full_layout_sample(sample: FullLayoutSample) -> dict:
    print(f"\n{'=' * 70}\n{sample.year} ({sample.label}) - full layout\n{'=' * 70}")

    if not sample.image_path.exists():
        print(f"  SKIPPED - image not found: {sample.image_path}")
        return {"year": sample.year, "skipped": True}

    real_cols = load_real_boundaries(sample.sidecar_path)
    print(f"{len(real_cols)} real (human-confirmed) columns loaded")

    image = Image.open(sample.image_path)
    gray = np.array(image.convert("L"))

    band = find_number_row_band(
        image, x0=sample.table_left, x1=sample.table_right,
        search_y0=sample.band_search_y0, search_y1=sample.band_search_y1,
    )
    if band is None:
        print("  SKIPPED - find_number_row_band() found no usable band")
        return {"year": sample.year, "skipped": True}
    band_y0, band_y1, blobs, score = band
    print(f"auto-calibrated band: y=({band_y0},{band_y1}), score={score:.2f}, {len(blobs)} raw blobs")

    all_centers = [c for c, _w in blobs]

    matched = []
    for name, x0, x1 in real_cols:
        real_center = (x0 + x1) / 2
        c = nearest_number_candidate(all_centers, real_center, MATCH_WINDOW_PX)
        if c is not None and abs(c - real_center) > MATCH_QUALITY_MAX_DELTA_PX:
            print(f"  [quality gate] {name!r}: delta {abs(c - real_center):.1f}px - gapped out")
            c = None
        matched.append(c)

    n_matched = sum(1 for c in matched if c is not None)
    print(f"{n_matched}/{len(real_cols)} columns confidently matched")

    runs = find_clean_runs(matched)
    print(f"{len(runs)} clean run(s): "
          f"{[(real_cols[a][0], real_cols[b][0]) for a, b in runs]}")

    errA_all, errB_all = [], []
    corridor_hits = {w: 0 for w in CORRIDOR_WIDTHS}
    corridor_total = 0
    closures = []
    b_found_count = 0
    b_total_count = 0

    for run_start, run_end in runs:
        if run_end - run_start < 1:
            continue
        L = real_cols[run_start][1]
        b_prev_A = L
        b_prev_B = L
        for i in range(run_start, run_end + 1):
            name, x0, x1 = real_cols[i]
            c_i = matched[i]
            real_B = x1

            proj_A = 2 * c_i - b_prev_A
            errA_all.append(abs(proj_A - real_B))

            proj_B = 2 * c_i - b_prev_B
            # Round 6: gradient-based detector (Scharr dx=1), swapped in
            # for the old dark-ridge/Otsu _find_vertical_ruling_line() -
            # proved to find 55-100% of real lines vs. 0-43% for the old
            # detector on these same years, see docs/COLUMN_NUMBER_
            # ANCHOR_RESEARCH.md's "Round 6".
            measured_B, found, _ratio = find_vertical_line_via_gradient(
                gray, sample.table_top, sample.table_bottom, int(round(proj_B)), search_radius=30,
            )
            errB_all.append(abs(measured_B - real_B))
            b_total_count += 1
            if found:
                b_found_count += 1

            for w in CORRIDOR_WIDTHS:
                if abs(real_B - proj_B) <= w:
                    corridor_hits[w] += 1
            corridor_total += 1

            b_prev_A = proj_A
            b_prev_B = measured_B if found else proj_B

        closures.append(abs(b_prev_A - real_cols[run_end][2]))

    if errA_all:
        print(f"Variant A: mean={sum(errA_all)/len(errA_all):.1f} "
              f"median={sorted(errA_all)[len(errA_all)//2]:.1f} max={max(errA_all):.1f}")
        print(f"Variant B: mean={sum(errB_all)/len(errB_all):.1f} "
              f"median={sorted(errB_all)[len(errB_all)//2]:.1f} max={max(errB_all):.1f} "
              f"(detector found a line {b_found_count}/{b_total_count} times)")
        print(f"Closure errors per run: {[f'{c:.1f}px' for c in closures]}")
        for w in CORRIDOR_WIDTHS:
            pct = 100 * corridor_hits[w] / corridor_total
            print(f"  +/-{w}px corridor: {corridor_hits[w]}/{corridor_total} ({pct:.1f}%)")
    else:
        print("No run with >=2 columns - nothing to project.")

    return {
        "year": sample.year, "skipped": False,
        "n_real": len(real_cols), "n_matched": n_matched, "n_runs": len(runs),
        "errA": errA_all, "errB": errB_all,
        "corridor_hits": corridor_hits, "corridor_total": corridor_total,
        "closures": closures,
    }


def run_spot_check_sample(sample: SpotCheckSample) -> None:
    print(f"\n{'=' * 70}\n{sample.year} ({sample.label}) - sparse spot-check only\n{'=' * 70}")

    if not sample.image_path.exists():
        print(f"  SKIPPED - image not found: {sample.image_path}")
        return

    image = Image.open(sample.image_path)
    band = find_number_row_band(
        image, x0=sample.table_left, x1=sample.table_right,
        search_y0=sample.band_search_y0, search_y1=sample.band_search_y1,
    )
    if band is None:
        print("  SKIPPED - find_number_row_band() found no usable band")
        return
    band_y0, band_y1, blobs, score = band
    print(f"auto-calibrated band: y=({band_y0},{band_y1}), score={score:.2f}, {len(blobs)} raw blobs")

    all_centers = [c for c, _w in blobs]
    deltas = []
    for name, (x0, x1) in sample.known_columns.items():
        real_center = (x0 + x1) / 2
        c = nearest_number_candidate(all_centers, real_center, MATCH_WINDOW_PX)
        if c is None:
            print(f"  {name:<24} NO CANDIDATE FOUND")
            continue
        delta = abs(c - real_center)
        deltas.append(delta)
        print(f"  {name:<24} real={real_center:>8.1f} detected={c:>8.1f} delta={delta:>6.1f}px")

    if deltas:
        print(f"{len(deltas)}/{len(sample.known_columns)} matched. "
              f"mean={sum(deltas)/len(deltas):.1f} max={max(deltas):.1f}")


def main() -> None:
    results = [run_full_layout_sample(s) for s in FULL_LAYOUT_SAMPLES]
    for s in SPOT_CHECK_SAMPLES:
        run_spot_check_sample(s)

    print(f"\n{'=' * 70}\nCOMBINED SUMMARY (full-layout years only)\n{'=' * 70}")
    all_errA, all_errB = [], []
    all_hits = {w: 0 for w in CORRIDOR_WIDTHS}
    all_total = 0
    for r in results:
        if r.get("skipped"):
            print(f"{r['year']}: SKIPPED")
            continue
        errA, errB = r["errA"], r["errB"]
        if not errA:
            print(f"{r['year']}: {r['n_matched']}/{r['n_real']} matched, no runnable run")
            continue
        all_errA += errA
        all_errB += errB
        for w in CORRIDOR_WIDTHS:
            all_hits[w] += r["corridor_hits"][w]
        all_total += r["corridor_total"]
        print(f"{r['year']}: {r['n_matched']}/{r['n_real']} matched, {r['n_runs']} run(s), "
              f"A mean/median/max={sum(errA)/len(errA):.1f}/{sorted(errA)[len(errA)//2]:.1f}/{max(errA):.1f}px  "
              f"B mean/median/max={sum(errB)/len(errB):.1f}/{sorted(errB)[len(errB)//2]:.1f}/{max(errB):.1f}px  "
              f"closures={[f'{c:.1f}' for c in r['closures']]}")

    if all_errA:
        print(f"\nAcross all years combined (n={len(all_errA)} boundaries):")
        print(f"  Variant A: mean={sum(all_errA)/len(all_errA):.1f} "
              f"median={sorted(all_errA)[len(all_errA)//2]:.1f} max={max(all_errA):.1f}")
        print(f"  Variant B: mean={sum(all_errB)/len(all_errB):.1f} "
              f"median={sorted(all_errB)[len(all_errB)//2]:.1f} max={max(all_errB):.1f}")
        for w in CORRIDOR_WIDTHS:
            pct = 100 * all_hits[w] / all_total
            print(f"  +/-{w}px corridor: {all_hits[w]}/{all_total} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
