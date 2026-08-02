"""
Automated sidecar generation - the automated-sidecar-generation branch's
experiment. Replaces ONLY the manual "define rows" step of the existing
workflow (deskew/bounds/row-boundary confirmation, done by hand in
ui/row_segmentation_ui.py) with a rule-based pipeline:

    image -> classify doc type -> load template -> locate table
    boundary -> locate header region -> detect data rows -> sidecar

Deliberately narrow scope, per the branch's own constraints:
  - Does NOT touch ui/row_segmentation_ui.py, core/row_extraction.py,
    or any prompt/OCR code - this module only ever CALLS INTO
    core/row_segmentation.py's existing functions (deskew estimation,
    ruling-line detection, build_sidecar, init_column_state), never
    modifies them.
  - Produces the SAME sidecar schema build_sidecar() already writes -
    Stage 1/2 OCR (core/row_extraction.py) and the manual per-column
    masking UI consume it completely unmodified. A human still opens
    ui/row_segmentation_ui.py afterward to mask columns exactly as
    before; this module only fills in what used to be the manual
    row-definition pass.
  - No model inference anywhere in this file (classification is rule-
    based, boundary/row detection is the same Otsu+projection-profile
    CV core/row_segmentation.py already uses) - CLAUDE.md's GPU-
    contention check doesn't apply to anything here.

Every located boundary is a REFINEMENT of a template's approximate
region against the real image (preferring an actually-detected printed
ruling line over the template's raw fraction), never the template
fraction taken blind - see locate_table_boundary()/locate_header_
region()'s docstrings for the search-window mechanics, which reuse
core/row_segmentation.py's own refine_boundary_position() (a per-
boundary local search) rather than inventing a parallel mechanism.

Row generation, 2026-07-27 (changed from the original design - see
detect_data_rows()'s docstring for the full incident): fixed-layout
templates (the three census years) do NOT search for each row's own
boundary. Only row 1's bottom edge is located; every row after that is
pure arithmetic (row_height = row1_bottom - row1_top, tiled down the
page) via core/row_segmentation.py's segment_rows_uniform_tile() - the
same mode the manual segmentation tool already uses on these forms.
Per-row searching was tried first and produced a real, confirmed
failure (two independent boundary searches on the same page each
snapped toward a bad local density minimum, squeezing one row to 17px
and inflating its neighbor to 48px) - deterministic tiling from a
single confirmed row cannot produce that failure mode, by construction.
Variable-row-count templates (the passenger manifest) still use real
per-row detection (row_strategy: "detect") - they have no fixed spacing
to exploit, so there's nothing to make deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw

from core.document_classification import ClassificationResult, classify_document
from core.document_templates import DocumentTemplate, load_template
from core.row_segmentation import (
    RowDetectionResult,
    _longest_run_per_row,
    _otsu_threshold,
    _row_density_profile,
    apply_deskew_angle,
    build_sidecar,
    detect_row_bands,
    estimate_deskew_angle,
    init_column_state,
    merge_wrapped_bands,
    refine_boundary_position,
    sanity_check_bands,
    segment_rows_uniform_tile,
)

# Search windows for refining a template's approximate boundary against
# the real image - expressed as a fraction of page height/width so they
# scale with scan resolution, same rationale as the templates'
# regions_approx fractions themselves.
_HORIZONTAL_SEARCH_RADIUS_FRAC = 0.03
_VERTICAL_SEARCH_RADIUS_FRAC = 0.03
_RULING_LINE_RUN_RATIO = 0.5

# Height-ratio anomaly thresholds for _quarantine_anomalous_rows(),
# split by mode - see that function's docstring for why periodic and
# detect modes need different tolerances. 0.65 (not a rounder-looking
# 0.6) is deliberate: the real Dauphin-174 row 19 case that motivated
# this measured 25px vs a 42px page median (ratio 0.595) - 0.6 would
# have caught it by a margin of 0.2px, too close for comfort against a
# different page's noise. 0.65 gives real headroom.
_PERIODIC_MIN_HEIGHT_RATIO = 0.65
_PERIODIC_MAX_HEIGHT_RATIO = 1.5
_DETECT_MIN_HEIGHT_RATIO = 0.3   # matches sanity_check_bands()'s own
_DETECT_MAX_HEIGHT_RATIO = 3.0   # general-purpose thresholds

# Page-level accept/reject threshold, per Jon's 2026-07-27 quality-
# policy direction: this pipeline targets finding-aid-quality row
# segmentation for semantic search, NOT archival OCR - recovering
# 90-95% of usable rows is preferable to rejecting a whole page over a
# few uncertain ones. A quarantined ROW is not a failure (the source
# image is always preserved, nothing is lost), so whole-PAGE
# quarantine should be the final safety net for pages that have
# clearly failed, not the default response to "some rows are
# uncertain." Configurable rather than hard-coded, per that same
# direction - see _assess_band_detection_quality()'s use of it.
PAGE_QUARANTINE_THRESHOLD = 0.50

# Preset retry ladder for "detect" strategy row detection, per Jon's
# 2026-07-27 direction - see detect_data_rows()'s "detect" branch and
# _assess_band_detection_quality()'s docstrings for the full design.
# Built after a real, direct measurement across the whole handwritten_
# manifest sample cluster showed NO single (smoothing_window,
# density_threshold_ratio) pair works for every page: "default" already
# handles most pages correctly (CANIMM1913PLIST: 31 bands, IMCANQC1865:
# 20 bands - both already good), while a handful of lower-quality scans
# collapse to 1-5 bands under it (e003558130, e003566165, e003667909)
# because density_threshold_ratio is defined relative to each image's
# OWN peak density, and on those specific scans the peak is itself an
# outlier (one stray dark mark), not representative of real row
# content. "lenient" and "lenient_alt" were the two settings that
# recovered plausible band counts (19, 15, 37 respectively) on those
# broken pages during that same measurement, without regressing
# CANIMM1913PLIST (31 bands either way). Ordered cheapest/most-trusted
# first - "default" already handles the common case, so most real pages
# never pay for a second attempt.
_DETECT_PRESETS = [
    {"name": "default", "smoothing_window": 5, "density_threshold_ratio": 0.05},
    {"name": "lenient", "smoothing_window": 6, "density_threshold_ratio": 0.15},
    {"name": "lenient_alt", "smoothing_window": 5, "density_threshold_ratio": 0.10},
]


def _assess_band_detection_quality(bands: list[tuple[int, int]], crop_height: int) -> dict:
    """
    Stage 1 of Jon's 2026-07-27 "detect obvious failures cheaply before
    doing anything expensive" design - refined same-day per his
    quality-policy direction (finding-aid-quality segmentation for
    semantic search, not archival OCR: recovering 90-95% of usable rows
    beats rejecting a whole page over a few uncertain ones). Two
    independent gates, both must pass:

    1. HARD structural checks - these mean "this isn't real row data
       at all," independent of any ratio, so they're checked
       regardless of the page-quality threshold: zero/near-zero bands
       over a real crop (the classic "collapsed into one blob"
       failure), a single band consuming most of the page, or an
       implausibly large average band height. Deliberately NOT scored
       on row count otherwise - two real pages can legitimately have
       different numbers of real entries, so count alone can't tell a
       correct detection from a broken one.

    2. PAGE_QUARANTINE_THRESHOLD gate - computes the REAL quarantine
       decision (via _find_quarantine_positions(), the exact same
       function _quarantine_anomalous_rows() uses on a built sidecar -
       not an approximation of it, the same computation) and compares
       the resulting rate against PAGE_QUARANTINE_THRESHOLD (default
       0.50). Below threshold: ACCEPT this preset even though some
       rows will still be individually quarantined later - per Jon's
       policy, "some rows quarantined, the rest still useful" is a
       normal, desired outcome (Accept-with-Review), not a detection
       failure. At/above threshold: this preset's detection is
       unreliable enough that the majority of what it found wouldn't
       survive real quarantine anyway - try the next preset.

    Returns {"passes": bool, "reason": str, "n_bands": int, ...} - the
    reason is always populated (even on a pass, "ok") so every attempt
    can be logged, not just failures.
    """
    n = len(bands)
    if n == 0:
        return {"passes": False, "reason": "zero bands detected", "n_bands": 0}
    if n <= 2 and crop_height > 200:
        return {
            "passes": False, "n_bands": n,
            "reason": f"only {n} band(s) detected over a {crop_height}px crop - "
                      f"the classic 'collapsed into one blob' failure",
        }

    heights = [b[1] - b[0] for b in bands]
    max_height = max(heights)
    mean_height = sum(heights) / n

    if max_height > crop_height * 0.30:
        return {
            "passes": False, "n_bands": n,
            "reason": f"a single band spans {max_height}px ({max_height / crop_height:.0%} of the "
                      f"{crop_height}px crop) - looks like a merged blob, not one row",
        }
    if mean_height > crop_height * 0.15:
        return {
            "passes": False, "n_bands": n,
            "reason": f"average band height ({mean_height:.0f}px) is implausibly large "
                      f"relative to the {crop_height}px crop",
        }

    quarantine_positions = _find_quarantine_positions(heights, _DETECT_MIN_HEIGHT_RATIO, _DETECT_MAX_HEIGHT_RATIO)
    quarantine_rate = len(quarantine_positions) / n
    if quarantine_rate >= PAGE_QUARANTINE_THRESHOLD:
        return {
            "passes": False, "n_bands": n, "quarantine_rate": quarantine_rate,
            "reason": f"{len(quarantine_positions)}/{n} bands ({quarantine_rate:.0%}) would be quarantined - "
                      f"at/above the {PAGE_QUARANTINE_THRESHOLD:.0%} page-quality threshold",
        }

    return {"passes": True, "n_bands": n, "quarantine_rate": quarantine_rate, "reason": "ok"}


@dataclass
class AutoSidecarResult:
    sidecar: dict | None            # None if classification was "unknown"
    debug_overlay: Image.Image | None
    classification: ClassificationResult
    diagnostics: dict
    warnings: list[str] = field(default_factory=list)

    # True when NO doc_type_override was supplied and the CV classifier
    # (core/document_classification.py) picked the template on its own.
    # Measured 2026-07-30, that classifier is not trustworthy for this
    # decision: it disagreed with Gemma on 6/16 real pages, scored 2/5 on
    # fresh LAC 1921/1931 scans (calling a 1921 page handwritten_manifest
    # at confidence 1.00), and its primary discriminator spans 13-43
    # vertical lines ACROSS PAGES OF A SINGLE YEAR - covering the 1921,
    # 1911 and 1931 ranges at once, so within-class spread exceeds
    # between-class separation and no re-calibration fixes it.
    # A True here means "this template choice was a guess" - the caller
    # should route the page for manual classification rather than trust
    # the sidecar it produced (see scripts/run_batch_auto_sidecar.py's
    # needs_manual_classification.csv).
    used_cv_fallback: bool = False


def _find_vertical_ruling_line(
    image: Image.Image, y0: int, y1: int, expected_x: int, search_radius: int,
    run_ratio_threshold: float = _RULING_LINE_RUN_RATIO,
) -> tuple[int, bool]:
    """
    Vertical-axis counterpart to refine_boundary_position(): searches a
    window around an EXPECTED x position for a real printed column-
    divider line, preferring it over the raw expected position. Reuses
    _longest_run_per_row on the TRANSPOSED crop (same trick as
    core/document_classification.py's _count_vertical_ruling_lines) so
    "longest unbroken run along a column" comes from the same helper
    detect_row_bands() already uses for the horizontal case, rather
    than a second hand-rolled implementation.

    Returns (x_position, found) - found=False means no candidate line
    cleared run_ratio_threshold anywhere in the window, so the caller
    should fall back to expected_x with a warning (same contract as
    locate_table_boundary()'s horizontal refinement).
    """
    w = image.width
    lo = max(0, expected_x - search_radius)
    hi = min(w, expected_x + search_radius + 1)
    if hi <= lo:
        return expected_x, False

    gray = image.convert("L")
    arr = np.array(gray)[y0:y1, lo:hi]
    if arr.size == 0:
        return expected_x, False
    thresh = _otsu_threshold(arr)
    binary = (arr <= thresh).astype(np.uint8)
    region_h = y1 - y0
    if region_h <= 0:
        return expected_x, False
    longest = _longest_run_per_row(binary.T, close_gap_px=4)
    ratios = longest / region_h

    candidates = np.where(ratios > run_ratio_threshold)[0]
    if len(candidates) == 0:
        return expected_x, False
    # Closest candidate to the expected position, same "prefer the
    # nearest real structural line" rule refine_boundary_position uses.
    local_expected = expected_x - lo
    best = candidates[np.argmin(np.abs(candidates - local_expected))]
    return lo + int(best), True


def _locate_rule_bottom_edge(
    density: np.ndarray, expected_y: int, search_radius: int,
    density_ratio_threshold: float = 0.2,
) -> tuple[int, dict]:
    """
    Finds the BOTTOM edge of a thick printed rule near an expected
    position - built 2026-07-27 per Jon's direction, after diagnosing a
    real miss: table_top (the boundary between the printed header block
    and row 1 - see build_sidecar()'s own docstring for why this exact
    value is also "where row 1 begins") kept landing several px too
    early. Root cause, confirmed by inspecting real per-row density
    numbers on 1921_022-E002880409 (see the corresponding conversation
    for the walkthrough, not just asserted here): the census forms'
    header/table divider is a THICK printed rule - measured there as
    ~5px of sharply elevated density (837/2082/2676/2320/1191 against a
    ~450 header-text baseline and a near-zero blank-gap baseline) - and
    refine_boundary_position() (used everywhere else in this module)
    has no notion of "a feature's trailing edge": its ruling-line path
    picks whichever ruling-line ROW is nearest the expected position
    (which can be the rule's leading edge, or any row within a thick
    rule, depending on where "nearest" happens to fall), and its
    density-MINIMUM fallback actively prefers the blank gap just above
    the rule over the rule itself, since blank paper is even lower
    density than printed ink. On that sample this landed table_top at
    y=635 (density 48, the blank-gap minimum) when the rule's own
    trailing edge - and true row-1 origin - was really ~y=642.

    Scans the search window for the density profile's own peak (the
    rule, whatever it measures - not an absolute threshold, since ink
    density varies by scan), takes every row within
    density_ratio_threshold of that peak as "elevated," groups elevated
    rows into CONTIGUOUS runs, and returns one past the last row of
    whichever run is CLOSEST to expected_y (same "prefer the nearest
    real structural feature" rule refine_boundary_position() and
    _find_vertical_ruling_line() both already use).

    The contiguous-run grouping is load-bearing, not decorative - a
    first version just took the LAST elevated row anywhere in the
    window, full stop, which is wrong whenever the window contains more
    than one elevated feature. Confirmed as a real bug on two of the
    three real census samples (1911, 1931): unlike 1921's comparatively
    clean single rule, their header/table transition is a busy multi-
    tier block (bilingual sub-labels, a numbered-column heading row -
    see core/row_segmentation.py's own periodic-mode docstring, which
    already documented this exact multi-tier structure) with several
    separate elevated-density bands in the same search window. Taking
    the LAST such row walked the boundary 65-71px past the TRUE rule
    (verified against each page's own manually-confirmed ground-truth
    table_top in data/outputs/row_segmentation/) into unrelated
    subsequent header content. Taking the run nearest expected_y (the
    template's own approx position, itself usually already close - both
    real misses were only 4-5px off before this function existed at
    all) fixes it: the true rule is essentially always the nearest
    elevated feature to the template's approximation, even when later,
    unrelated header content is locally MORE elevated (taller peak) or
    a wider run.

    0.2 (not the module's usual 0.5 ruling_line_run_ratio) is
    deliberate: this is a density-magnitude bar, not a run-length bar,
    and the 1921 sample's own header-text baseline (~450) sits close
    enough beneath a 0.5x-peak line (~1338 there) that a looser bar was
    needed to cleanly separate "part of the thick rule" from "normal
    header text density."

    Falls back to expected_y if the window has no clear peak (a flat,
    empty window - nothing resembling a rule found at all).
    """
    lo = max(0, expected_y - search_radius)
    hi = min(len(density), expected_y + search_radius + 1)
    window = density[lo:hi]
    if len(window) == 0:
        return expected_y, {"found": False}

    peak = window.max()
    if peak <= 0:
        return expected_y, {"found": False}

    is_elevated = window > (peak * density_ratio_threshold)
    if not is_elevated.any():
        return expected_y, {"found": False, "window_peak_density": float(peak)}

    # Group elevated rows into contiguous runs.
    runs = []
    run_start = None
    for i, v in enumerate(is_elevated):
        if v and run_start is None:
            run_start = i
        elif not v and run_start is not None:
            runs.append((run_start, i))  # [start, end) local indices
            run_start = None
    if run_start is not None:
        runs.append((run_start, len(is_elevated)))

    local_expected = expected_y - lo
    best_run = min(runs, key=lambda r: min(abs(r[0] - local_expected), abs(r[1] - 1 - local_expected)))
    rule_bottom_local = best_run[1]  # one past the run's last elevated row
    return lo + rule_bottom_local, {
        "found": True, "window_peak_density": float(peak),
        "rule_span": [lo + best_run[0], lo + rule_bottom_local],
        "all_runs_found": [[lo + r[0], lo + r[1]] for r in runs],
    }


# Lower bound tuned to 0.80 (not the rounder-looking 0.85), deliberate:
# real in-pipeline measurement (not the hand-picked numbers used while
# first designing this check) put 1921's own ratio at 0.786 and
# z000017634's at 0.842 - only 5.6 points apart, tighter than initial
# testing suggested (page-height-relative scaling of expected_row_height
# isn't perfectly uniform across real scans - see canada_census_1911
# .yaml's own comment). 0.80 clears 1921 with a small margin while still
# catching z000017634. Biased toward catching MORE suspicious cases
# rather than protecting against occasional over-flagging: a false
# positive here just costs an unnecessary manual review (cheap, per
# Jon's stated tolerance), a false negative silently ships a wrong
# table_top.
_TABLE_TOP_ROW_SPACING_RATIO_RANGE = (0.80, 1.20)


def _check_table_top_plausibility(rule_diag: dict, expected_row_height: float) -> tuple[bool, dict]:
    """
    Foundation-bug fix (2026-07-27, the real z000017634 incident - see
    _locate_rule_bottom_edge()'s "nearest to expected" selection, which
    is generally correct but assumes the template's calibrated approx
    fraction is close to this page's real layout; it isn't always).

    On a fixed-layout form, MULTIPLE candidate rules commonly appear in
    the search window: the true header/table divider, AND several
    row-separator lines further into the table (real, confirmed on
    every census sample tested, not hypothetical - see
    core/document_templates.py's own field-comment history). When the
    template's approx position is itself off for a given page (a
    genuine per-page layout variation, not a bug in the search), the
    "nearest to expected" pick can land on one of those row-separator
    lines instead of the true divider.

    The two look structurally different in one specific way: a row-
    separator line sits almost EXACTLY one row-height below the
    PREVIOUS candidate in the window (since periodic rows are
    near-uniform by construction), while the true header/table divider
    does not - there is no earlier row directly above it, only header
    content at a different visual rhythm. Checking the gap between the
    CHOSEN run and its immediate predecessor (if any) against
    expected_row_height, calibrated per-template from real samples, is
    a cheap, independent cross-check verified against all 5 real
    samples this project has: it cleanly separates the 4 correct picks
    (ratio 0.18-0.79) from the one confirmed-wrong pick (ratio 0.96)
    with real margin, not a coincidence of one sample.

    This does NOT attempt to auto-correct the boundary (tried a "walk
    back through the chain" version - too fragile close to the 1921
    sample's own 0.79 ratio, uncomfortably near the z000017634 case's
    0.96). It only flags ambiguity for the caller to route to whole-
    page quarantine - per Jon's explicit direction, an outlier reaching
    manual review is an acceptable outcome, preferable to a fragile
    auto-correction that risks trading one wrong answer for another.

    Returns (is_ambiguous, diagnostics).
    """
    all_runs = rule_diag.get("all_runs_found")
    rule_span = rule_diag.get("rule_span")
    if not all_runs or not rule_span or expected_row_height <= 0:
        return False, {"checked": False, "reason": "no runs/rule_span to compare"}

    try:
        chosen_index = all_runs.index(list(rule_span))
    except ValueError:
        return False, {"checked": False, "reason": "chosen run not found in all_runs_found"}

    if chosen_index == 0:
        return False, {"checked": True, "reason": "chosen run is the first candidate - no predecessor to compare"}

    predecessor = all_runs[chosen_index - 1]
    gap = rule_span[0] - predecessor[1]
    ratio = gap / expected_row_height
    lo, hi = _TABLE_TOP_ROW_SPACING_RATIO_RANGE
    is_ambiguous = lo <= ratio <= hi

    return is_ambiguous, {
        "checked": True, "gap_to_predecessor": gap, "expected_row_height": expected_row_height,
        "ratio": round(ratio, 3), "suspicious_range": [lo, hi],
        "reason": (
            f"gap to predecessor rule ({gap}px) is {ratio:.0%} of the expected row height "
            f"({expected_row_height:.1f}px) - looks like a row-separator line, not the true "
            f"header/table divider" if is_ambiguous else "gap does not match row spacing"
        ),
    }


def locate_table_boundary(
    image: Image.Image, template: DocumentTemplate,
) -> tuple[tuple[int, int, int, int], dict]:
    """
    Refines the template's approximate table region against real
    printed borders.

    table_top (2026-07-27, changed - see _locate_rule_bottom_edge()'s
    docstring for the full incident) uses _locate_rule_bottom_edge()
    instead of the generic refine_boundary_position() ONLY for
    "fixed_periodic" templates (the census forms): that boundary is
    defined as "where row 1's real content begins," which on those
    forms means the BOTTOM edge of a thick, sharply-elevated printed
    divider rule - confirmed by inspecting real density numbers (a
    single dominant spike, several times the surrounding baseline).

    "detect" templates (the passenger manifest) keep the ORIGINAL
    refine_boundary_position() logic for table_top too - tried
    _locate_rule_bottom_edge() there first and it was wrong, per Jon's
    own confirmation that typed manifests have no grid lines at all,
    just text with whitespace around it. Checked the real density
    profile to confirm rather than assume: no dominant spike anywhere
    near the expected position, just several small, similarly-sized
    bumps (column-header text, printed labels) - there's no "rule" for
    a rule-edge finder to correctly locate, and the generic ruling-line/
    density-minimum search (built for exactly this whitespace-delimited
    case) is the right tool here, not the census-specific one.

    table_bottom always uses the original refine_boundary_position()
    regardless of strategy - Jon's diagnosis was specifically about the
    header/table transition, and the bottom of the table (page margin,
    not a divider rule) doesn't share that failure mode on either doc
    family. Left/right use _find_vertical_ruling_line() above, the same
    idea on the other axis.

    Falls back to the template's raw approx fraction (converted to
    pixels) for any edge where no real line clears the confidence
    threshold - never leaves a boundary undefined.
    """
    w, h = image.size
    approx = template.regions_approx["table"]
    approx_y0, approx_y1 = approx["y_frac"]
    approx_x0, approx_x1 = approx["x_frac"]
    expected_top = int(approx_y0 * h)
    expected_bottom = int(approx_y1 * h)
    expected_left = int(approx_x0 * w)
    expected_right = int(approx_x1 * w)

    x_scope0, x_scope1 = expected_left, expected_right
    density, _ = _row_density_profile(image, x0=x_scope0, x1=x_scope1)
    _, _, ruling_line_rows = detect_row_bands(image, x0=x_scope0, x1=x_scope1)

    h_radius = max(2, int(h * _HORIZONTAL_SEARCH_RADIUS_FRAC))
    table_top_ambiguous = False
    if template.row_strategy == "fixed_periodic":
        table_top, rule_diag = _locate_rule_bottom_edge(density, expected_top, h_radius)
        if template.expected_row_height_frac is not None:
            table_top_ambiguous, ambiguity_diag = _check_table_top_plausibility(
                rule_diag, template.expected_row_height_frac * h)
            rule_diag["plausibility_check"] = ambiguity_diag
    else:
        table_top = refine_boundary_position(density, ruling_line_rows, expected_top, h_radius)
        rule_diag = {"skipped": "row_strategy != 'fixed_periodic', no divider rule to find"}
    table_bottom = refine_boundary_position(density, ruling_line_rows, expected_bottom, h_radius)

    v_radius = max(2, int(w * _VERTICAL_SEARCH_RADIUS_FRAC))
    table_left, left_found = _find_vertical_ruling_line(
        image, table_top, table_bottom, expected_left, v_radius)
    table_right, right_found = _find_vertical_ruling_line(
        image, table_top, table_bottom, expected_right, v_radius)

    if table_bottom <= table_top:
        table_bottom = table_top + 1  # never emit an inverted/zero-height table
    if table_right <= table_left:
        table_right = table_left + 1

    diagnostics = {
        "expected": [expected_left, expected_top, expected_right, expected_bottom],
        "refined": [table_left, table_top, table_right, table_bottom],
        "top_rule": rule_diag,
        "table_top_ambiguous": table_top_ambiguous,
        "left_from_ruling_line": left_found,
        "right_from_ruling_line": right_found,
    }
    return (table_left, table_top, table_right, table_bottom), diagnostics


def locate_header_region(
    image: Image.Image, template: DocumentTemplate, table_bbox: tuple[int, int, int, int],
) -> tuple[tuple[int, int, int, int] | None, int, dict]:
    """
    The header region is, by this project's own established sidecar
    schema (see core/row_segmentation.py's build_sidecar() docstring),
    exactly the gap between the metadata block and the table body:
    metadata_bottom -> table_top. table_top here is already the
    REFINED value from locate_table_boundary(), so this only needs to
    locate the other edge (metadata_bottom) via the same refine_
    boundary_position() search-window approach.
    """
    w, h = image.size
    table_left, table_top, table_right, table_bottom = table_bbox
    approx_metadata_bottom_frac = template.regions_approx["metadata"]["y_frac"][1]
    expected_metadata_bottom = int(approx_metadata_bottom_frac * h)

    density, _ = _row_density_profile(image, x0=table_left, x1=table_right)
    _, _, ruling_line_rows = detect_row_bands(image, x0=table_left, x1=table_right)
    h_radius = max(2, int(h * _HORIZONTAL_SEARCH_RADIUS_FRAC))
    metadata_bottom = refine_boundary_position(
        density, ruling_line_rows, expected_metadata_bottom, h_radius)

    warnings = []
    if metadata_bottom >= table_top:
        warnings.append(
            f"Refined metadata_bottom ({metadata_bottom}) landed at/past table_top "
            f"({table_top}) - falling back to the template's raw approx position "
            f"({expected_metadata_bottom}) instead of emitting a zero/negative-height header."
        )
        metadata_bottom = min(expected_metadata_bottom, table_top - 1) if table_top > 0 else 0

    header_bbox = (table_left, metadata_bottom, table_right, table_top) if metadata_bottom < table_top else None
    diagnostics = {"expected_metadata_bottom": expected_metadata_bottom,
                    "refined_metadata_bottom": metadata_bottom, "warnings": warnings}
    return header_bbox, metadata_bottom, diagnostics


# Floor on the column-edge search radius, as a fraction of TABLE width
# (changed 2026-07-29 from page width, to match locate_columns()'s own
# fix - see that function's comment) - combined with a per-column-
# width-relative component in locate_columns() below. Needed because
# some real columns are narrow (1911's "Sex" is only ~1.3% of table
# width), and a search window scaled purely to a narrow column's own
# width can be too tight to reach a real ruling line on a page that
# varies from the calibration sample. The original 3-sample 1911
# spread this constant was tuned against (two samples agreeing, a
# third disagreeing on Sex/Relationship to Head by ~0.03-0.05) turned
# out to be MOSTLY the page-vs-table-relative bug itself, not real
# per-page variation - kept as a real, smaller floor rather than
# removed, since ordinary scan-to-scan noise in ruling-line position
# is still a real (if now smaller) effect.
_COLUMN_SEARCH_RADIUS_MIN_FRAC = 0.015

# Plausible found-width-vs-expected-width ratio range for a located
# column - see locate_columns()'s use of this for the real incident
# (1931 census "Sex": found 83px vs expected 39px, ratio 2.13) that
# motivated it. Every correctly-located column across all 5 real
# reference samples stayed within ~0.85-1.15x; 1.5 gives real headroom
# above that while still catching a >2x miss with margin.
_COLUMN_WIDTH_PLAUSIBLE_RATIO_RANGE = (0.5, 1.5)

# How far a detected number-blob is allowed to shift a column's search
# anchor, as a fraction of page width - a safety bound so a wrongly-
# matched blob (the nearest one happens to belong to a DIFFERENT
# printed column than the one being searched for) can't drag the
# search window somewhere unrelated. Deliberately generous enough to
# cover the real motivating case (canada_census_1911.yaml's
# z000017634 outlier, off by up to ~0.056 of page width on Sex) while
# still bounded, per Jon's "triangulate, don't rely on it as the sole
# anchor" direction.
_NUMBER_BLOB_MAX_SHIFT_FRAC = 0.08

# Minimum spread (as a fraction of table width) the CONFIDENT edge
# matches must span before a registration affine is trusted. Two points
# close together can fit a line exactly but with a wildly wrong slope -
# this rejects that degenerate case rather than fitting one anyway.
_MIN_AFFINE_FIT_SPAN_FRAC = 0.05
_MIN_AFFINE_FIT_POINTS = 2


def _fit_registration_affine(pairs: list[tuple[float, float]], min_span: float) -> tuple[float, float] | None:
    """
    Fits refined_x = a * expected_x + b over (expected, refined) position
    pairs from CONFIDENTLY matched column edges (a real printed ruling
    line was found, not a template fallback) - see locate_columns()'s
    own docstring for why this is the fix, not more per-edge tuning.

    Uses cv2.fitLine with a robust (Huber) distance metric rather than
    an ordinary least-squares fit: a "confident" match is only the
    CLOSEST candidate line within the search window to the expected
    position (_find_vertical_ruling_line()'s own contract), which is not
    a guarantee of correctness - an occasional match can still land on
    the wrong physical line. A robust fit tolerates a handful of such
    outliers pulling the line without letting them dominate the fitted
    scale/shift the way an ordinary least-squares fit would.

    Returns (a, b) or None if there are too few confident points, or
    they don't span enough of the table width to fit a slope reliably
    (see _MIN_AFFINE_FIT_SPAN_FRAC) - the caller falls back to the
    original per-edge behavior in either case.
    """
    if len(pairs) < _MIN_AFFINE_FIT_POINTS:
        return None
    xs = [p[0] for p in pairs]
    if max(xs) - min(xs) < min_span:
        return None

    points = np.array(pairs, dtype=np.float32).reshape(-1, 1, 2)
    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
    if abs(vx) < 1e-9:
        return None  # degenerate (vertical) fit - expected values didn't vary
    a = float(vy / vx)
    b = float(y0 - a * x0)
    return a, b


def _interpolate_from_neighbors(
    target_expected: float, confident_sorted: list[tuple[float, float]],
) -> float | None:
    """
    Bracketed linear interpolation between the two CONFIDENT edge matches
    (real ruling lines, not template guesses) nearest target_expected on
    either side - Jon's design, 2026-07-30: "if we can find some columns
    and they snap correctly, can we approximate where the others should
    be" - combined with the global registration affine above into a
    single fallback cascade (see locate_columns()'s "RESOLUTION CASCADE"
    docstring section) rather than either/or.

    Requires a confident anchor on BOTH sides - true interpolation, not
    extrapolation from one point. A single nearby confident match could
    itself be an outlier (the affine's robust fit tolerates a few of
    those; a bare local extrapolation from just one would not), so this
    deliberately refuses to guess past the last known-good anchor in
    either direction - the caller falls through to the template position
    instead, same "don't fabricate past real evidence" discipline as
    every other boundary fallback in this module.

    `confident_sorted` is every confidently-matched edge across the WHOLE
    table (not just this column) sorted by expected_x - a target
    column's own bracket is often formed by its neighboring columns'
    matched edges, or even its OTHER edge if that one was itself
    confidently found.

    Returns None if no such bracket exists (target is beyond every
    confident match on one side, or there are fewer than 2 confident
    matches at all).
    """
    lo = hi = None
    for ex, found in confident_sorted:
        if ex <= target_expected:
            lo = (ex, found)
        elif hi is None:
            hi = (ex, found)
            break
    if lo is None or hi is None:
        return None
    (lo_x, lo_y), (hi_x, hi_y) = lo, hi
    if hi_x == lo_x:
        return lo_y
    frac = (target_expected - lo_x) / (hi_x - lo_x)
    return lo_y + frac * (hi_y - lo_y)


# How far a directly-measured edge is allowed to disagree with the page's
# own registration affine before the measurement is treated as a bad
# ruling-line snap rather than trusted evidence - Jon's refinement,
# 2026-07-30: "the registration becomes the sanity check for the
# measurement instead of competing with it." Adaptive PER PAGE (a
# multiple of the affine's own fit residual on the confident points it
# was built from) rather than one fixed magic number: a clean page's tight
# fit should distrust even a small disagreement, while a noisier page's
# own residual sets a naturally looser bar. Floored in px so a
# near-perfect fit doesn't produce an unreasonably tight tolerance from
# ordinary floating-point/pixel-rounding noise.
_MEASUREMENT_RESIDUAL_TOLERANCE_MULTIPLIER = 3.0
_MEASUREMENT_RESIDUAL_TOLERANCE_FLOOR_PX = 3.0


def _affine_residual_tolerance(
    affine: tuple[float, float], confident_pairs: list[tuple[float, float]],
) -> float:
    """
    Per-page disagreement tolerance for _resolve_edge()'s measurement-
    vs-affine sanity check - see that constant pair's own comment for
    the reasoning. Uses the MEDIAN residual (not mean/max) so a single
    genuinely bad snap already inside the affine's training set can't
    inflate the very tolerance meant to catch bad snaps.
    """
    a, b = affine
    residuals = sorted(abs((a * ex + b) - found) for ex, found in confident_pairs)
    median_residual = residuals[len(residuals) // 2] if residuals else 0.0
    return max(_MEASUREMENT_RESIDUAL_TOLERANCE_FLOOR_PX,
                median_residual * _MEASUREMENT_RESIDUAL_TOLERANCE_MULTIPLIER)


def _resolve_edge(
    expected: float, measured: float, found: bool,
    affine: tuple[float, float] | None, tolerance: float,
    confident_sorted: list[tuple[float, float]],
) -> tuple[float, str]:
    """
    Picks one edge's final position - see locate_columns()'s RESOLUTION
    CASCADE for the full tier ordering and reasoning. `tolerance` is
    meaningless when affine is None and is ignored in that case.
    """
    predicted = affine[0] * expected + affine[1] if affine is not None else None

    if found:
        if predicted is None or abs(measured - predicted) <= tolerance:
            return measured, "measured"
        return predicted, "affine_override"

    if predicted is not None:
        return predicted, "affine_predicted"

    interpolated = _interpolate_from_neighbors(expected, confident_sorted)
    if interpolated is not None:
        return interpolated, "locally_interpolated"

    return expected, "template"


def _detect_header_number_blobs(
    image: Image.Image, x0: int, x1: int, y0: int, y1: int,
    density_ratio_threshold: float = 0.15,
) -> list[tuple[int, int]]:
    """
    Detects individual printed-number blobs (the column numbers
    printed in a thin strip near the bottom of the header - "1 2 3 4a
    4b 4c 5 6 7..." - see each census template's own number_row_approx
    comment for how that strip's y-range was measured) within a given
    region. Returns [(center_x, width), ...] sorted left to right.

    Same column-density + contiguous-run grouping technique
    _locate_rule_bottom_edge() uses for finding a printed RULE's
    extent, applied here along the same axis (column-wise density) but
    for a row of separated glyphs instead of one continuous feature -
    each number is its own short run of elevated density, with real
    whitespace gaps between adjacent numbers (unlike a rule, which is
    one long contiguous run).
    """
    gray = image.convert("L")
    arr = np.array(gray)[y0:y1, x0:x1]
    if arr.size == 0:
        return []
    thresh = _otsu_threshold(arr)
    binary = (arr <= thresh).astype(np.uint8)
    col_density = binary.sum(axis=0).astype(float)
    peak = col_density.max()
    if peak <= 0:
        return []

    is_ink = col_density > (peak * density_ratio_threshold)
    blobs = []
    start = None
    for i, v in enumerate(is_ink):
        if v and start is None:
            start = i
        elif not v and start is not None:
            blobs.append((start, i))
            start = None
    if start is not None:
        blobs.append((start, len(is_ink)))

    return [(x0 + (b0 + b1) // 2, b1 - b0) for b0, b1 in blobs]


def locate_columns(
    image: Image.Image, template: DocumentTemplate, table_bbox: tuple[int, int, int, int],
) -> tuple[dict[str, list[tuple[int, int]]], dict]:
    """
    Column automation (2026-07-27, Jon's direction: "the last stage
    before handing to sidecar generation"). For every column in
    template.column_regions_approx, refines its approximate x0/x1
    against real printed column-divider lines within the table - same
    "template approx position + core-refines-against-real-structure"
    pattern as every other boundary in this module, just applied to
    the vertical dividers BETWEEN columns instead of the table/header
    edges. Reuses _find_vertical_ruling_line() directly (the same
    function locate_table_boundary() uses for the table's own left/
    right edges) called once per column edge, rather than a new
    mechanism.

    Columns NOT present in template.column_regions_approx are silently
    skipped (no entry in the returned dict) - there's no calibration
    data for them yet, so they stay exactly as init_column_state()
    left them (empty mask_keep_ranges, "pending" status) for manual
    masking, same as before this function existed. This is a real,
    current limitation for some columns (e.g. every census template's
    "Occupation" - the real reference sidecars never masked it either,
    see each template's own column_regions_approx comment), not
    something this function tries to paper over with a guess.

    Returns ({column_name: [(x0, x1)]}, diagnostics) - each column
    gets exactly one keep-range (a simple, single vertical strip;
    real columns with a genuinely split layout, e.g. printed_manifest's
    "Arrival" which had two near-duplicate manual ranges - almost
    certainly remask noise, not a real split - aren't modeled here).

    LOCKED IN 2026-07-27 as pure ruling-line refinement, per Jon's
    direction, after real testing: 0-15px accuracy across every column
    on every real reference sample except one outlier (z000017634,
    off by 50-175px on Sex/Relationship to Head - see
    canada_census_1911.yaml's own comment). A number-row-triangulation
    version was tried (_detect_header_number_blobs() shifting the
    search anchor toward the nearest detected printed number) and
    REMOVED again - it didn't fix the outlier it was built for, and
    introduced small new errors on pages that were already accurate
    (a 1D column-density profile can't cleanly separate individual
    digits at this print resolution - the same lesson later confirmed
    for the row-margin-number idea too: real signal, not reliably
    separable from noise with this technique). _detect_header_number_
    blobs() is kept, unused, as a documented starting point - a real
    fix would need 2D connected-component blob detection, not a 1D
    projection. The z000017634-class outlier is instead caught
    upstream by locate_table_boundary()'s row-height plausibility
    check (see that function's docstring) and whole-page-quarantined -
    per Jon's direction, an occasional outlier reaching manual review
    is an acceptable outcome, preferable to a fragile "fix" that
    trades one wrong answer for a different one.

    REGISTRATION AFFINE ADDED 2026-07-30 (Jon: "if the table edge can be
    detected the columns should land near where they actually are" -
    confirmed correct by measurement). Diagnosed against the 9 real
    manual sidecars with column masks in data/outputs/row_segmentation/:
    per-edge ruling-line refinement (the LOCKED IN mechanism above)
    barely helps accuracy at all (mean column-edge error 2.90% of table
    width with it OFF vs 2.85% with it ON - about a 0.05pp gain, and it
    makes 3/21 cases worse). Meanwhile fitting a single page-level
    scale+shift between two pages' column positions collapsed the
    SAME apparent error to 0.03pp on 1911 and 0.91pp on 1921. So the
    error was never really about individual column edges - template
    column fractions are already accurate; what varies page to page is
    the ANCHOR (table_left/table_right), which locate_table_boundary()
    sometimes snaps to the outer page border and sometimes to the first
    inner rule (measured table width 0.940-0.997 of page width across
    otherwise-similar samples).

    RESOLUTION CASCADE, EVIDENCE-DRIVEN not fixed-precedence (Jon's
    refinement, 2026-07-30, replacing an initial version of this fix
    that always preferred the affine outright): a directly-measured edge
    and the page-level affine aren't treated as competing sources where
    one always wins - the affine is used as a SANITY CHECK on the
    measurement. A measured edge that AGREES with the affine's own
    prediction (within an adaptive tolerance - see
    _affine_residual_tolerance()) is trusted directly; a measured edge
    that DISAGREES beyond tolerance is more likely a bad ruling-line
    snap than the affine being wrong, and gets overridden by the
    affine's predicted position instead. Per edge, in order:

      1. "measured"          - a real ruling line was found AND it
                                agrees with the page's own fitted
                                geometry (or no affine exists to check
                                it against, i.e. this is the only
                                available evidence).
      2. "affine_override"   - a real ruling line was found but
                                disagrees with the affine beyond
                                tolerance - the registration overrules
                                the snap.
      2. "affine_predicted"  - no line was found at all, but a
                                page-level affine exists - predict from
                                it directly. (Same tier as the override
                                case: both mean "use the affine's own
                                prediction for this edge".)
      3. "locally_interpolated" - no affine exists (too few/too
                                clustered confident points to trust a
                                page-level fit), but this edge's
                                expected position falls BETWEEN two
                                other confidently-matched edges
                                elsewhere in the table (possibly
                                including this same column's OTHER edge,
                                if that one was itself confidently
                                found) - bracketed linear interpolation
                                between them (_interpolate_from_
                                neighbors()). Requires anchors on BOTH
                                sides; never extrapolates past the last
                                known-good match.
      4. "template"           - none of the above; the raw template
                                fraction, same safety net every other
                                boundary in this module already has.

    This does NOT replace the per-edge search - it still runs first, on
    every column, exactly as before, and remains the ONLY source of real
    per-page structural evidence (both for direct measurement and as the
    calibration points the affine/interpolation are built from). When
    too few confident matches exist anywhere in the table for any
    correction, behavior is UNCHANGED from before this date - every
    column falls back to its own raw per-edge result.
    """
    table_left, table_top, table_right, table_bottom = table_bbox
    table_width = table_right - table_left

    # Pass 1: per-edge local search, unchanged from the original
    # implementation - still the only source of real per-page structural
    # evidence. Collected into `raw` (one entry per column) and into
    # `confident_pairs` (one entry per edge that matched a real ruling
    # line), which is what the registration fit below is built from.
    raw: dict[str, dict] = {}
    confident_pairs: list[tuple[float, float]] = []
    for col_name, approx in template.column_regions_approx.items():
        # x_frac is relative to the TABLE's own left-right span, not
        # the full page (fixed 2026-07-29, per Jon: confirmed against 3
        # real 1911 references that a page-width-relative fraction
        # doesn't transfer between scans with different left-margin/
        # dewarp crop amounts, even though the SAME physical column
        # sits at a near-identical position relative to the table
        # itself - e001926997's table starts at 5.4% of page width vs
        # ~1.5% for the two original calibration samples, which alone
        # explained a ~4pp Name-position "disagreement" that vanished
        # once measured relative to the table instead of the page).
        approx_x0_frac, approx_x1_frac = approx["x_frac"]
        expected_x0 = table_left + int(approx_x0_frac * table_width)
        expected_x1 = table_left + int(approx_x1_frac * table_width)
        col_width_approx = max(1, expected_x1 - expected_x0)

        radius = max(
            int(table_width * _COLUMN_SEARCH_RADIUS_MIN_FRAC),
            int(col_width_approx * 0.3),
        )

        x0, left_found = _find_vertical_ruling_line(
            image, table_top, table_bottom, expected_x0, radius)
        x1, right_found = _find_vertical_ruling_line(
            image, table_top, table_bottom, expected_x1, radius)

        raw[col_name] = {
            "expected_x0": expected_x0, "expected_x1": expected_x1,
            "col_width_approx": col_width_approx,
            "x0": x0, "x1": x1,
            "left_found": left_found, "right_found": right_found,
        }
        if left_found:
            confident_pairs.append((expected_x0, x0))
        if right_found:
            confident_pairs.append((expected_x1, x1))

    affine = _fit_registration_affine(confident_pairs, table_width * _MIN_AFFINE_FIT_SPAN_FRAC)
    tolerance = _affine_residual_tolerance(affine, confident_pairs) if affine is not None else 0.0
    confident_sorted = sorted(confident_pairs)

    # Pass 2: resolve each edge independently through the RESOLUTION
    # CASCADE above - a column's two edges can land in different tiers
    # (e.g. left edge directly measured and confirmed, right edge locally
    # interpolated from neighbors) since they're physically different
    # ruling lines with independent evidence.
    results: dict[str, list[tuple[int, int]]] = {}
    diagnostics: dict[str, dict] = {}

    for col_name, r in raw.items():
        expected_x0, expected_x1 = r["expected_x0"], r["expected_x1"]
        col_width_approx = r["col_width_approx"]

        x0, x0_source = _resolve_edge(
            expected_x0, r["x0"], r["left_found"], affine, tolerance, confident_sorted)
        x1, x1_source = _resolve_edge(
            expected_x1, r["x1"], r["right_found"], affine, tolerance, confident_sorted)

        if x1 <= x0:
            # Collapsed/inverted range (e.g. both edges resolved to the
            # same nearby line) - fall back to the raw approx position
            # rather than emit a zero/negative-width mask, same safety
            # net every other boundary in this module already has.
            x0, x1 = expected_x0, expected_x1
            x0_source = x1_source = "template_fallback"

        x0, x1 = int(round(x0)), int(round(x1))

        # Width-plausibility flag (2026-07-27, per Jon's direction:
        # "flag any that don't report correctly for quarantine" - the
        # real 1931 census "Sex" miss that motivated this: found width
        # 83px vs an expected 39px, a 2.1x ratio, while every correct
        # column across all 5 real reference samples stayed within
        # ~0.9-1.1x of its own expected width.
        found_width = x1 - x0
        width_ratio = found_width / col_width_approx
        width_implausible = not (_COLUMN_WIDTH_PLAUSIBLE_RATIO_RANGE[0]
                                  <= width_ratio <= _COLUMN_WIDTH_PLAUSIBLE_RATIO_RANGE[1])

        results[col_name] = [(x0, x1)]
        diagnostics[col_name] = {
            "expected": [expected_x0, expected_x1],
            "refined": [x0, x1],
            "raw_search": [r["x0"], r["x1"]],
            "source": [x0_source, x1_source],
            "width_ratio": round(width_ratio, 2),
            "width_implausible": width_implausible,
            "left_from_ruling_line": r["left_found"], "right_from_ruling_line": r["right_found"],
        }

    diag_affine = {"used": affine is not None}
    if affine is not None:
        diag_affine.update({"scale": round(affine[0], 4), "shift": round(affine[1], 2),
                             "confident_points": len(confident_pairs),
                             "residual_tolerance_px": round(tolerance, 2)})
    diagnostics["_registration_affine"] = diag_affine

    return results, diagnostics


def detect_data_rows(
    image: Image.Image, template: DocumentTemplate, table_bbox: tuple[int, int, int, int],
    metadata_bottom: int, deskew_angle: float, original_image: Image.Image,
) -> tuple[RowDetectionResult, dict]:
    """
    Dispatches on template.row_strategy:

    - "fixed_periodic": known, fixed row_count on a standardized printed
      form (the census templates) - deliberately NOT per-row boundary
      refinement (that was the original implementation, reusing
      segment_rows_periodic(); changed 2026-07-27 per Jon's direction
      after it produced a real, confirmed failure - see this function's
      own comment below for the incident). Locates ONLY row 1's bottom
      edge (one refinement search, same refine_boundary_position()
      mechanism used elsewhere in this module), then generates every
      other row by pure arithmetic via segment_rows_uniform_tile() -
      exactly the mode the MANUAL segmentation tool already falls back
      to on these forms (see ui/row_segmentation_ui.py's "uniform tile"
      mode; the real 1921_022-E002880409 sidecar in data/outputs/
      row_segmentation/ was hand-confirmed with mode="uniform_tile",
      not "periodic" - Jon had already independently arrived at the
      same conclusion by hand before asking for this change).

    - "detect": unknown/variable row_count (the manifest template) -
      no existing core/row_segmentation.py entry point restricts the
      GENERAL detector to an arbitrary table sub-region (segment_rows()
      operates on the whole page), so this crops to table_bbox, runs
      the same detect_row_bands -> merge_wrapped_bands ->
      sanity_check_bands sequence segment_rows() itself uses, and
      offsets the resulting bands back into full-page coordinates.
      Unchanged by the 2026-07-27 fixed_periodic switch - manifests
      have no fixed row count, so per-row detection is still the
      correct tool here, not a compromise.

    original_image (pre-deskew) is required for the "fixed_periodic"
    path - segment_rows_uniform_tile() does its own apply_deskew_angle()
    call internally and expects the RAW image, not an already-deskewed
    copy (deskewing twice would double-rotate). The "detect" path
    instead operates directly on the already-deskewed `image`, since
    table_bbox was itself computed in that same deskewed coordinate
    space.
    """
    table_left, table_top, table_right, table_bottom = table_bbox
    warnings: list[str] = []

    if template.row_strategy == "fixed_periodic":
        if not template.expected_row_count:
            raise ValueError(
                f"Template {template.doc_type!r} declares row_strategy='fixed_periodic' "
                f"but has no expected_row_count set."
            )
        # The ONLY per-row search this strategy performs: row 1's
        # bottom edge (== top of row 2). table_top is already row 1's
        # top (refined by locate_table_boundary()); everything else is
        # multiplication from here, per Jon's 2026-07-27 direction -
        # "Once row 1 has been correctly established, the remaining row
        # geometry is deterministic." This is precisely what would have
        # prevented the real Dauphin-174 (1931_174-e011707164) failure:
        # rows 19 and 20 came out 17px and 48px (vs a ~35px median)
        # because boundaries 18 AND 19 EACH independently searched for
        # a ruling line, found none nearby, and both fell back to a
        # local density minimum that happened to snap toward each other
        # - two independent per-row searches, two independent chances to
        # be wrong, compounding into a squeeze. Pure arithmetic
        # subdivision from a single confirmed row cannot produce that
        # failure mode, by construction.
        expected_row_height = (table_bottom - table_top) / template.expected_row_count
        density, _ = _row_density_profile(image, x0=table_left, x1=table_right)
        _, _, ruling_line_rows = detect_row_bands(image, x0=table_left, x1=table_right)
        expected_row1_bottom = int(table_top + expected_row_height)
        h_radius = max(2, int(expected_row_height * 0.3))
        row1_bottom = refine_boundary_position(
            density, ruling_line_rows, expected_row1_bottom, h_radius)
        if row1_bottom <= table_top:
            row1_bottom = table_top + max(1.0, expected_row_height)
            warnings.append(
                f"Row 1 bottom refinement collapsed onto/above table_top - fell back "
                f"to the raw expected position ({expected_row1_bottom})."
            )

        # Sanity-clamp against the theoretical average row height (table
        # span / row_count) - added after a real failure on
        # 1921_022-E002880409: row 1's refinement landed at 35px (vs a
        # 28.6px theoretical average), and tiling that 22%-too-tall
        # measurement 50 times ran the last several rows 316px past the
        # actual image bottom (this strategy trades away the OLD
        # adjacent-boundary-pull failure, but is correspondingly more
        # sensitive to row 1's own measurement being right, since
        # there's no second row-scoped search left to catch a bad one -
        # see this branch's opening comment). A single-row measurement
        # more than 20% off the page's own structural prior is more
        # likely a bad snap than genuine form variation, so fall back to
        # the theoretical value rather than let a 50x multiplication of
        # one bad number run off the end of the page.
        measured_row_height = row1_bottom - table_top
        if abs(measured_row_height - expected_row_height) > 0.20 * expected_row_height:
            warnings.append(
                f"Row 1 bottom refinement measured a {measured_row_height}px row "
                f"(vs a {expected_row_height:.1f}px theoretical average from the table's "
                f"own span / row_count) - more than 20% off, so falling back to the "
                f"theoretical value rather than tiling a likely-bad measurement "
                f"{template.expected_row_count} times."
            )
            # Float, NOT rounded to int - segment_rows_uniform_tile()'s own
            # docstring is explicit about why: rounding row_height before
            # tiling compounds a small per-row error into a large one by
            # the last row (exactly the 16px-overrun regression this fix
            # replaced - int(round(...)) here previously reintroduced the
            # very rounding-compounding bug that function was built to
            # avoid). table_top + expected_row_height stays exact.
            row1_bottom = table_top + expected_row_height

        result, _row_crops, _header_crop, _overlay = segment_rows_uniform_tile(
            original_image,
            row_count=template.expected_row_count,
            first_row_top=float(table_top), first_row_bottom=float(row1_bottom),
            table_left=table_left, table_right=table_right,
            header_row_count=1,
            deskew_angle=deskew_angle,
            header_box_top=metadata_bottom, header_box_bottom=table_top,
        )
        result.warnings = warnings + result.warnings
        diagnostics = {
            "strategy": "fixed_periodic", "row_count": len(result.bands),
            "row1_top": table_top, "expected_row1_bottom": expected_row1_bottom,
            "refined_row1_bottom": row1_bottom, "row_height": row1_bottom - table_top,
        }
        return result, diagnostics

    if template.row_strategy == "detect":
        crop = image.crop((table_left, table_top, table_right, table_bottom))
        crop_height = table_bottom - table_top

        # Preset retry ladder (Jon's 2026-07-27 design - see
        # _DETECT_PRESETS' and _assess_band_detection_quality()'s own
        # docstrings for the full reasoning): try the cheap/default
        # preset first, only pay for a retry when there's actual
        # evidence it failed, stop at the first preset that passes
        # structural sanity. This mirrors the same pattern the rest of
        # this pipeline already uses - classify_document() stops at
        # "unknown" rather than guessing; this stops row-detection
        # escalation the moment a preset looks trustworthy, rather than
        # always running every preset "just in case."
        attempt_log = []
        chosen_preset_name = None
        kept_bands, dropped_bands, sanity_warnings = [], [], []
        ruling_line_count = 0
        # Tracks the best-scoring attempt seen so far (lowest quarantine
        # rate; falls back to most bands found when neither attempt has
        # a computed rate, e.g. both hit a hard structural failure) -
        # used as the Stage 3 fallback content below if NO preset
        # passes, rather than just whichever preset happened to run
        # last. A human reviewing a quarantined page deserves the most
        # plausible starting point among what was actually tried.
        best_attempt = None  # (quality, kb, db, sw, rlc)

        for preset in _DETECT_PRESETS:
            raw_bands, rlc, ruling_line_rows = detect_row_bands(
                crop, smoothing_window=preset["smoothing_window"],
                density_threshold_ratio=preset["density_threshold_ratio"],
            )
            merged_bands = merge_wrapped_bands(raw_bands, ruling_line_rows=ruling_line_rows)
            kb, db, sw = sanity_check_bands(merged_bands)
            quality = _assess_band_detection_quality(kb, crop_height)
            attempt_log.append(f"{preset['name']} ({quality.get('n_bands', 0)} bands): "
                                f"{'PASS' if quality['passes'] else 'FAIL - ' + quality['reason']}")

            attempt_rank = (quality.get("quarantine_rate", 1.0), -quality.get("n_bands", 0))
            if best_attempt is None or attempt_rank < best_attempt[0]:
                best_attempt = (attempt_rank, kb, db, sw, rlc)

            if quality["passes"]:
                chosen_preset_name = preset["name"]
                kept_bands, dropped_bands, sanity_warnings, ruling_line_count = kb, db, sw, rlc
                break

        warnings.append("Row detection preset attempts: " + " | ".join(attempt_log))

        # Stage 3: every preset failed structural sanity - the whole
        # PAGE gets quarantined (flagged in diagnostics here;
        # generate_auto_sidecar() is what actually moves every row into
        # rows_needs_review, since that requires the built sidecar's
        # row indices, not available yet at this point in the
        # pipeline). Uses the BEST-scoring attempt among all presets
        # tried as the starting content, so a human reviewing has the
        # most plausible material to correct rather than an empty or
        # arbitrarily-chosen sidecar.
        page_detection_failed = chosen_preset_name is None
        if page_detection_failed:
            _, kept_bands, dropped_bands, sanity_warnings, ruling_line_count = best_attempt
        if page_detection_failed:
            warnings.append(
                f"All {len(_DETECT_PRESETS)} row-detection preset(s) failed structural sanity checks - "
                f"this page will be quarantined in full rather than trusting any of them."
            )
        else:
            warnings.append(f"Row detection succeeded using preset {chosen_preset_name!r}.")

        warnings.extend(sanity_warnings)
        warnings.append(
            f"Detected {ruling_line_count} ruling-line row(s) within the table region."
        )

        # Offset from crop-local to full deskewed-image coordinates.
        offset_bands = [(y0 + table_top, y1 + table_top) for (y0, y1) in kept_bands]
        offset_dropped = [(y0 + table_top, y1 + table_top) for (y0, y1) in dropped_bands]

        result = RowDetectionResult(
            bands=offset_bands,
            header_band=(metadata_bottom, table_top) if metadata_bottom < table_top else None,
            deskew_angle=deskew_angle,
            deskewed_image_size=image.size,
            warnings=warnings,
            dropped_bands=offset_dropped,
        )
        diagnostics = {
            "strategy": "detect", "row_count": len(offset_bands),
            "ruling_line_count": ruling_line_count,
            "preset_attempts": attempt_log, "chosen_preset": chosen_preset_name,
            "page_detection_failed": page_detection_failed,
        }
        return result, diagnostics

    raise ValueError(f"Unknown row_strategy {template.row_strategy!r} on template {template.doc_type!r}")


def _quarantine_whole_page(sidecar: dict, reason: str) -> list[dict]:
    """
    Stage 3 of Jon's 2026-07-27 preset-retry design: moves EVERY row
    out of sidecar["rows"] and into sidecar["rows_needs_review"] with
    the same shared reason, for a page whose row detection failed
    structural sanity on every preset tried (see detect_data_rows()'s
    "detect" branch and _assess_band_detection_quality()). Same
    mechanism _quarantine_anomalous_rows() uses for individual rows
    (pull from "rows", preserve each row's own "index", never
    renumber) - a page-level failure is just the case where every row
    happens to get flagged, so it reuses the same sidecar fields rather
    than inventing a parallel "page_status" concept.
    """
    rows = sidecar.get("rows", [])
    if not rows:
        return []

    quarantined = []
    for r in rows:
        flagged = dict(r)
        flagged["reason"] = reason
        quarantined.append(flagged)

    sidecar["rows"] = []
    sidecar["rows_needs_review"] = sidecar.get("rows_needs_review", []) + quarantined
    sidecar.setdefault("warnings", []).append(
        f"Quarantined ALL {len(quarantined)} row(s) to rows_needs_review - {reason}"
    )
    return quarantined


def _find_quarantine_positions(
    heights: list[float], min_ratio: float, max_ratio: float,
) -> dict[int, list[str]]:
    """
    Pure, reusable core of the row-quarantine decision: given a list of
    row heights in reading order (positionally indexed, no coupling to
    sidecar row dicts), returns {position: [reasons]} for every row
    that should be quarantined - implausibly thin/tall vs. the page's
    own median height, THEN propagated to each flagged row's immediate
    neighbors (a bad boundary corrupts the two rows sharing it, but the
    inflated side doesn't always clear max_ratio on its own - see the
    real Dauphin-174 row 19/20 incident this logic was built to catch,
    documented in full in _quarantine_anomalous_rows()'s docstring).

    Extracted 2026-07-27 (was inlined in _quarantine_anomalous_rows())
    so the SAME real quarantine decision - not an approximation of it -
    can be evaluated during preset selection in detect_data_rows()
    (see _assess_band_detection_quality()) before a sidecar even
    exists yet. Two call sites, one source of truth for what counts as
    quarantine-worthy.
    """
    n = len(heights)
    if n < 4:
        return {}  # not enough rows for a meaningful median

    median_h = sorted(heights)[n // 2]
    if median_h <= 0:
        return {}

    reasons: dict[int, list[str]] = {}
    for i, h in enumerate(heights):
        if h < median_h * min_ratio:
            reasons.setdefault(i, []).append(
                f"implausibly thin ({h}px vs page median {median_h}px)")
        elif h > median_h * max_ratio:
            reasons.setdefault(i, []).append(
                f"implausibly tall ({h}px vs page median {median_h}px)")

    primary_flagged = set(reasons.keys())
    for i in primary_flagged:
        for neighbor in (i - 1, i + 1):
            if 0 <= neighbor < n and neighbor not in primary_flagged:
                reasons.setdefault(neighbor, []).append(
                    f"shares a boundary with flagged row at position {i}")

    return reasons


def _quarantine_anomalous_rows(sidecar: dict) -> list[dict]:
    """
    Post-processes an already-built sidecar (build_sidecar() already
    ran - this never touches that function) to pull implausible rows
    OUT of sidecar["rows"] and into a new sidecar["rows_needs_review"]
    list, so a future automated OCR pass (which will only ever iterate
    sidecar["rows"]) naturally skips them without any changes to
    core/row_extraction.py.

    Built 2026-07-27 after a real miss on data/outputs/row_segmentation/
    1931_174-e011707164 (Dauphin, Manitoba, page 6): rows 19/20 came out
    squished/overlapping because BOTH of their shared boundaries fell
    back to a noisy density-minimum guess (no real ruling line detected
    nearby - the exact failure mode core/row_segmentation.py's own
    2026-07-12 docstring already warned about for periodic mode on real
    1931 census scans) and those two guesses happened to land close
    together instead of at the true row edges. Jon caught it by eye in
    the debug overlay; this makes that kind of miss self-flagging
    instead of requiring a human to notice it first.

    Deliberately operates on the FINAL, already-padded bboxes in
    sidecar["rows"] rather than on raw pre-padding bands - that's what
    actually ends up in the sidecar (and what a human visually
    reviewing the overlay actually sees), so detection matches the
    real symptom directly rather than a proxy for it.

    Two independent checks, either one is enough to flag a row:
      - height ratio vs. the page's own median row height. Thresholds
        differ by sidecar["mode"]: periodic rows are supposed to be
        near-uniform (segment_rows_periodic's whole premise is a fixed
        row_count over a known span), so a tight +/-40%-ish band is the
        right anomaly signal; "detect" mode (variable-row-count doc
        types like the passenger manifest) has genuinely more natural
        height variance, so it reuses core/row_segmentation.py's own
        sanity_check_bands() thresholds (0.3x/3x) instead of a tighter
        one that would over-flag legitimate rows.
      - vertical overlap with the next row down, in reading order -
        the direct visual symptom Jon spotted, independent of whether
        either row's OWN height looks anomalous in isolation (rows 19
        and 20 didn't individually look absurd - it's their shared
        boundary that was wrong).

    Rows are kept in sidecar["rows"] with their ORIGINAL "index" value
    (the real printed line number on the form) - never renumbered, so a
    gap (e.g. rows 1-18, 20-50 present, 19 missing) is exactly what it
    should be: line 19 is genuinely pending review, not silently
    absorbed into a shifted sequence.

    Returns the list of quarantined row dicts (each with an added
    "reason" key) for the caller to log/print.
    """
    rows = sidecar.get("rows", [])
    if len(rows) < 4:
        return []  # not enough rows for a meaningful median

    if sidecar.get("mode") == "auto_fixed_periodic":
        min_ratio, max_ratio = _PERIODIC_MIN_HEIGHT_RATIO, _PERIODIC_MAX_HEIGHT_RATIO
    else:
        min_ratio, max_ratio = _DETECT_MIN_HEIGHT_RATIO, _DETECT_MAX_HEIGHT_RATIO

    heights = [r["bbox"][3] - r["bbox"][1] for r in rows]
    reasons_by_position = _find_quarantine_positions(heights, min_ratio, max_ratio)
    if not reasons_by_position:
        return []

    reasons = {rows[i]["index"]: reasons_list for i, reasons_list in reasons_by_position.items()}

    kept, quarantined = [], []
    for r in rows:
        if r["index"] in reasons:
            flagged = dict(r)
            flagged["reason"] = "; ".join(reasons[r["index"]])
            quarantined.append(flagged)
        else:
            kept.append(r)

    sidecar["rows"] = kept
    sidecar["rows_needs_review"] = sidecar.get("rows_needs_review", []) + quarantined
    sidecar.setdefault("warnings", []).append(
        f"Quarantined {len(quarantined)} row(s) to rows_needs_review (indices "
        f"{sorted(r['index'] for r in quarantined)}) - implausible geometry, "
        f"not auto-processed. See each entry's 'reason'."
    )
    return quarantined


def render_auto_debug_overlay(
    image: Image.Image, table_bbox: tuple[int, int, int, int],
    header_bbox: tuple[int, int, int, int] | None, rows: list[dict],
    rows_needs_review: list[dict] | None = None,
) -> Image.Image:
    """
    Purpose-built overlay for this pipeline's own diagnostic needs - a
    DIFFERENT color convention from core/row_segmentation.py's own
    render_debug_overlay() (which never draws a table rectangle at all,
    since the manual workflow confirms table bounds interactively
    rather than needing them visualized after the fact). Per spec:
    green=table boundary, blue=header exclusion zone, red=each kept
    row. Orange (2026-07-27, after the Dauphin-174 row 19/20 miss) marks
    rows _quarantine_anomalous_rows() pulled into rows_needs_review.

    rows/rows_needs_review take the sidecar's own row dicts ({"index":,
    "bbox":}), not bare bboxes - labels use each row's REAL stored
    index, not its position in the list, since quarantining can leave
    gaps (e.g. 1-18, 20-50) that a plain enumerate() would mislabel.
    """
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)

    draw.rectangle(list(table_bbox), outline="green", width=4)
    draw.text((table_bbox[0] + 6, table_bbox[1] + 4), "TABLE", fill="green")

    if header_bbox is not None:
        draw.rectangle(list(header_bbox), outline="blue", width=3)
        draw.text((header_bbox[0] + 6, header_bbox[1] + 4), "HEADER (excluded)", fill="blue")

    for r in rows:
        bbox = r["bbox"]
        draw.rectangle(list(bbox), outline="red", width=1)
        draw.text((bbox[0] + 4, bbox[1] + 1), str(r["index"]), fill="red")

    for r in (rows_needs_review or []):
        bbox = r["bbox"]
        draw.rectangle(list(bbox), outline="orange", width=2)
        draw.text((bbox[0] + 4, bbox[1] + 1), f"{r['index']} REVIEW", fill="orange")

    return overlay


def generate_auto_sidecar(
    image_path, debug: bool = False,
    doc_type_override: str | None = None, external_confidence: float | None = None,
) -> AutoSidecarResult:
    """
    Full pipeline entry point: image -> classify -> template -> table
    boundary -> header region -> data rows -> sidecar dict, matching
    core/row_segmentation.py's build_sidecar() schema exactly (Stage 1/2
    OCR and the manual per-column masking UI consume it unmodified).

    No human confirms the deskew angle in this fully-automated path
    (unlike the interactive UI, where estimate_deskew_angle()'s result
    is always a starting suggestion a person confirms/overrides) -
    auto-estimate is used directly and logged as such in the sidecar's
    own warnings, the same fallback behavior segment_rows_periodic()
    already has for non-interactive/batch callers.

    doc_type_override (2026-07-27, added for the Gemma-subtype ->
    batch-CV pipeline, scripts/run_batch_auto_sidecar.py): when given,
    SKIPS core/document_classification.py's CV classifier entirely and
    uses this doc_type directly to pick the template - this is how an
    upstream semantic classification (Gemma's subtype stage, or any
    other source) drives which template the deterministic CV geometry
    work runs against, without the CV classifier's own aspect-ratio/
    ruling-line heuristics getting a vote. The CV classifier itself is
    UNCHANGED and still fully usable on its own (classify_document() is
    only skipped when this override is actually passed) - per Jon's
    Phase 2 design, it's retained as an independent benchmark/fallback,
    not modified or removed.

    Raises the same FileNotFoundError load_template() raises if
    doc_type_override names a doc_type with no template file - this is
    NOT silently caught into an "unknown" result, unlike the CV
    classifier's own path (which returns "unknown" for a low-confidence
    score, a normal outcome for that flow). An override naming a
    nonexistent template is a caller bug (e.g. a typo, or Gemma
    returning a string outside the agreed taxonomy) and should surface,
    not be swallowed the same way a low-confidence CV score is.
    """
    image_path = str(image_path)
    original = Image.open(image_path).convert("RGB")
    angle = estimate_deskew_angle(original)
    deskewed = apply_deskew_angle(original, angle)

    if doc_type_override is not None:
        classification = ClassificationResult(
            doc_type=doc_type_override,
            confidence=external_confidence if external_confidence is not None else 1.0,
            template_name=doc_type_override,
            features={}, scores={"source": "external_override"},
        )
    else:
        classification = classify_document(deskewed)
    used_cv_fallback = doc_type_override is None
    fallback_warning = (
        [
            "Template chosen by the CV classifier fallback (no doc_type_override "
            "given) - this is a GUESS, not a trusted classification. Route this "
            "page for manual classification rather than consuming the sidecar."
        ]
        if used_cv_fallback else []
    )
    diagnostics = {
        "used_cv_fallback": used_cv_fallback,
        "classification_features": classification.features,
        "classification_scores": classification.scores,
        "deskew_angle": angle,
    }

    if classification.doc_type == "unknown":
        return AutoSidecarResult(
            used_cv_fallback=used_cv_fallback,
            sidecar=None, debug_overlay=None, classification=classification,
            diagnostics=diagnostics,
            warnings=fallback_warning + [
                f"Document classified as 'unknown' (best confidence "
                f"{classification.confidence:.2f}, below threshold) - no template "
                f"to generate a sidecar against. See diagnostics for measured features."],
        )

    template = load_template(classification.doc_type)
    table_bbox, table_diag = locate_table_boundary(deskewed, template)
    header_bbox, metadata_bottom, header_diag = locate_header_region(deskewed, template, table_bbox)
    result, row_diag = detect_data_rows(
        deskewed, template, table_bbox, metadata_bottom, angle, original)

    diagnostics.update({
        "template": template.doc_type,
        "table_boundary": table_diag,
        "header_region": header_diag,
        "row_detection": row_diag,
    })

    sidecar = build_sidecar(
        result,
        source_image_path=image_path,
        mode=f"auto_{template.row_strategy}",
        parameters={
            "doc_type": classification.doc_type,
            "classification_confidence": classification.confidence,
            "template": template.doc_type,
            "expected_row_count": template.expected_row_count,
        },
        table_top=table_bbox[1], table_bottom=table_bbox[3],
        x0=table_bbox[0], x1=table_bbox[2],
        metadata_bottom=metadata_bottom,
    )
    init_column_state(sidecar, template.expected_columns)

    column_masks, column_diag = locate_columns(deskewed, template, table_bbox)
    flagged_columns = []
    for col_name, keep_ranges in column_masks.items():
        # Status stays "pending" (NOT "done") - column automation fills
        # in a real, calibrated starting mask, but nothing here has
        # been human-confirmed yet. A person opening this sidecar in
        # ui/row_segmentation_ui.py (untouched, still the same tool)
        # sees the mask already drawn and can accept or adjust it,
        # rather than starting from a blank column - the UI itself
        # needed no changes for this, since mask_keep_ranges is already
        # a real, independently-used field regardless of status.
        #
        # EXCEPT (2026-07-27, per Jon's "flag any that don't report
        # correctly for quarantine" direction): a column whose located
        # width looks structurally implausible (see locate_columns()'s
        # _COLUMN_WIDTH_PLAUSIBLE_RATIO_RANGE comment for the real 1931
        # "Sex" incident this targets) gets "needs_review" instead -
        # the schema's own existing status vocabulary already covers
        # exactly this case, nothing new to add. The mask is still
        # written (a wrong-but-present starting point beats an empty
        # one for a human to correct), just not silently trusted as
        # "pending" (implying "an ordinary column awaiting its normal
        # turn"), which would bury a real miss among routine columns.
        sidecar["columns"][col_name]["mask_keep_ranges"] = [list(r) for r in keep_ranges]
        if column_diag.get(col_name, {}).get("width_implausible"):
            sidecar["columns"][col_name]["status"] = "needs_review"
            flagged_columns.append(col_name)
    diagnostics["column_locating"] = column_diag
    diagnostics["flagged_columns"] = flagged_columns
    if flagged_columns:
        sidecar.setdefault("warnings", []).append(
            f"Column(s) flagged needs_review (implausible located width, see "
            f"diagnostics.column_locating): {flagged_columns}"
        )

    if row_diag.get("page_detection_failed"):
        # Stage 3 of Jon's preset-retry design: every "detect" preset
        # failed structural sanity on this page, so quarantine it in
        # full rather than run the normal PER-ROW anomaly check on
        # detection none of the presets trusted in the first place -
        # see detect_data_rows()'s "detect" branch for where this flag
        # gets set.
        quarantined = _quarantine_whole_page(
            sidecar,
            reason="row detection failed structural sanity checks on every preset tried "
                   "(see warnings for per-preset attempts) - whole page needs manual review",
        )
    elif table_diag.get("table_top_ambiguous"):
        # Foundation-bug fix (2026-07-27, the z000017634 incident) -
        # see _check_table_top_plausibility()'s own docstring for the
        # full mechanism. table_top itself looked structurally
        # suspicious (evenly spaced from an earlier candidate rule like
        # a row-separator, not the true header/table divider) - EVERY
        # row on this page was tiled from that possibly-wrong origin,
        # so none of them can be trusted individually. Whole-page
        # quarantine, not a per-row check.
        quarantined = _quarantine_whole_page(
            sidecar,
            reason="table_top looked structurally ambiguous (see diagnostics.table_boundary."
                   "top_rule.plausibility_check) - every row was tiled from a possibly-wrong "
                   "origin, so the whole page needs manual boundary confirmation",
        )
    else:
        quarantined = _quarantine_anomalous_rows(sidecar)
    diagnostics["quarantined_rows"] = [
        {"index": r["index"], "reason": r["reason"]} for r in quarantined
    ]

    debug_overlay = None
    if debug:
        debug_overlay = render_auto_debug_overlay(
            deskewed, table_bbox, header_bbox, sidecar["rows"], sidecar.get("rows_needs_review"))

    return AutoSidecarResult(
        used_cv_fallback=used_cv_fallback,
        sidecar=sidecar, debug_overlay=debug_overlay, classification=classification,
        diagnostics=diagnostics,
        warnings=fallback_warning + list(sidecar.get("warnings", [])),
    )
