"""
Research/prototype for column-boundary PROJECTION from header-number
centre anchors (continuation of the column-anchor work - see
docs/COLUMN_NUMBER_ANCHOR_RESEARCH.md for the full history).

Geometry (per direct spec): each printed header number is centred
inside its own column, so given a KNOWN left edge B[0] and detected
centre C[1], column 1's right edge is B[1] = 2*C[1] - B[0] (symmetric
reflection of the known edge through the centre) - and B[1] becomes
column 2's left edge, recursively: B[i] = 2*C[i] - B[i-1].

Tests two variants:
  A. Pure projection - B[i] chained from projected values only, never
     corrected. Measures how much drift accumulates.
  B. Detector-corrected - after each projection, the EXISTING ruling-
     line detector (core/auto_sidecar.py's _find_vertical_ruling_line(),
     reused as-is, NOT reimplemented) searches a narrow corridor around
     the projection; if it finds a real line, that MEASURED position
     (not the raw projection) becomes B[i-1] for the next step.

Validated against real, independently-confirmed column boundaries -
NOT self-consistency. Only "clean consecutive runs" (every column in
the run has a confidently-matched header-number centre) are projected;
gaps (missing/merged/ambiguous centres, e.g. the already-known 19-21
touching-ink cluster) are flagged and skipped, never interpolated or
guessed.

Read-only research script. Does NOT modify core/auto_sidecar.py's
production locate_columns().

Usage:
    python -m diagnostics.test_column_boundary_projection
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from core.auto_sidecar import _find_vertical_ruling_line
from core.row_segmentation import detect_column_number_centers, nearest_number_candidate

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- 1931 real reference (e011717826) ---------------------------------
SIDECAR_1931 = PROJECT_ROOT / "data" / "outputs" / "row_segmentation" / "e011717826_sidecar.json"
IMAGE_1931 = PROJECT_ROOT / "data" / "outputs" / "lac_pull_1931_batch1" / "raw_jpgs" / "e011717826.jpg"
NUMBER_ROW_Y0_1931, NUMBER_ROW_Y1_1931 = 654, 666
TABLE_TOP_1931, TABLE_BOTTOM_1931 = 690, 2400
MATCH_WINDOW_PX = 60  # nearest_number_candidate tolerance, matches prior validated usage


def load_real_boundaries(sidecar_path: Path) -> list[tuple[str, float, float]]:
    """Returns [(column_name, x0, x1), ...] sorted left to right, real
    human-confirmed mask ranges (multi-range columns collapsed to their
    overall span - a remask artifact, not a real split, matching the
    reasoning already documented in core/auto_sidecar.py's locate_columns())."""
    data = json.loads(sidecar_path.read_text(encoding="utf-8"))
    out = []
    for name, col in data["columns"].items():
        ranges = col["mask_keep_ranges"]
        x0 = min(r[0] for r in ranges)
        x1 = max(r[1] for r in ranges)
        out.append((name, x0, x1))
    return sorted(out, key=lambda t: t[1])


def find_clean_runs(matched_centers: list[float | None]) -> list[tuple[int, int]]:
    """Returns [(start_idx, end_idx), ...] inclusive ranges of consecutive
    non-None centres - never bridges a gap."""
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


def main() -> None:
    real_cols = load_real_boundaries(SIDECAR_1931)
    print(f"{len(real_cols)} real columns loaded (1931 reference)")

    image = Image.open(IMAGE_1931)
    blobs = detect_column_number_centers(
        image, x0=173, x1=4028, y0=NUMBER_ROW_Y0_1931, y1=NUMBER_ROW_Y1_1931,
    )
    all_centers = [c for c, _w in blobs]
    print(f"{len(blobs)} raw header-number blobs detected\n")

    # Match each real column to its nearest detected centre - reuses the
    # already-validated filtering step, does not invent anything. A match
    # whose delta from the "real" center is implausibly large (beyond
    # MATCH_QUALITY_MAX_DELTA_PX) is treated as UNRELIABLE and dropped -
    # confirmed necessary: "Yea of Naturalization" has a known remask-
    # artifact ground-truth range (two disjoint mask_keep_ranges, the
    # same "remask noise, not a real split" class of bad reference data
    # locate_columns()'s own docstring already flags for a different
    # form/column), producing a real_center that doesn't correspond to
    # where the actual printed number sits - a 34.5px delta vs. 1-8px
    # for every genuinely good match nearby. Rather than silently trust
    # a match that's likely measuring against bad ground truth, gate it
    # out and treat it as a gap, same "don't invent, flag explicitly"
    # principle applied to unreliable REFERENCE data, not just missing
    # sensor data.
    MATCH_QUALITY_MAX_DELTA_PX = 15
    matched = []
    for name, x0, x1 in real_cols:
        real_center = (x0 + x1) / 2
        c = nearest_number_candidate(all_centers, real_center, MATCH_WINDOW_PX)
        if c is not None and abs(c - real_center) > MATCH_QUALITY_MAX_DELTA_PX:
            print(f"  [quality gate] {name!r}: delta {abs(c-real_center):.1f}px exceeds "
                  f"{MATCH_QUALITY_MAX_DELTA_PX}px - treating as unreliable, gapped out")
            c = None
        matched.append(c)

    n_matched = sum(1 for c in matched if c is not None)
    print(f"{n_matched}/{len(real_cols)} columns have a confidently-matched header-number centre\n")

    runs = find_clean_runs(matched)
    print(f"{len(runs)} clean consecutive run(s): "
          f"{[(real_cols[a][0], real_cols[b][0]) for a, b in runs]}\n")

    corridor_widths = [5, 10, 20, 30, 50]
    corridor_hits = {w: 0 for w in corridor_widths}
    corridor_total = 0

    for run_start, run_end in runs:
        if run_end - run_start < 1:
            continue  # need at least 2 columns to test a projection
        print(f"=== Run: {real_cols[run_start][0]!r} .. {real_cols[run_end][0]!r} "
              f"({run_end - run_start + 1} columns) ===")
        L = real_cols[run_start][1]  # true left edge of the first column in this run
        b_prev_A = L  # Variant A: pure projection chain
        b_prev_B = L  # Variant B: detector-corrected chain

        print(f"{'Column':<40} {'C[i]':>8} {'real_B':>8} "
              f"{'proj_A':>8} {'errA':>7} {'proj_B':>8} {'measB':>8} {'errB':>7}")
        for i in range(run_start, run_end + 1):
            name, x0, x1 = real_cols[i]
            c_i = matched[i]
            real_B = x1  # this column's real right edge

            proj_A = 2 * c_i - b_prev_A
            err_A = abs(proj_A - real_B)

            proj_B = 2 * c_i - b_prev_B
            measured_B, found = _find_vertical_ruling_line(
                image, TABLE_TOP_1931, TABLE_BOTTOM_1931, int(round(proj_B)), search_radius=30,
            )
            err_B = abs(measured_B - real_B)

            for w in corridor_widths:
                corridor_total_flag = abs(real_B - proj_B) <= w
                if corridor_total_flag:
                    corridor_hits[w] += 1
            corridor_total += 1

            print(f"{name:<40} {c_i:>8.1f} {real_B:>8.1f} "
                  f"{proj_A:>8.1f} {err_A:>7.1f} {proj_B:>8.1f} {measured_B:>8.1f} {err_B:>7.1f}")

            b_prev_A = proj_A
            b_prev_B = measured_B if found else proj_B

        closure_err = abs(b_prev_A - real_cols[run_end][2])
        print(f"  Variant A closure error at end of run: {closure_err:.1f}px")
        print()

    print("=== Corridor capture-rate summary (all runs combined) ===")
    for w in corridor_widths:
        pct = 100 * corridor_hits[w] / corridor_total if corridor_total else 0.0
        print(f"  +/-{w}px: {corridor_hits[w]}/{corridor_total} ({pct:.1f}%) real boundaries within corridor")


if __name__ == "__main__":
    main()
