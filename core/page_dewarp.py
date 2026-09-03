"""
Page-level crease/gutter dewarping for the 1901 census "South Lanark"
reel (and any other bound-volume source with the same problem).

Built 2026-08-06 after a long research investigation
(docs/COLUMN_NUMBER_ANCHOR_RESEARCH.md, Rounds 9-13) into why
1901's column-number match rate lagged 1926/1931's so badly. Every
attempt at fixing CALIBRATION on top of the existing pixels (piecewise
y-band search, scorer tuning, trend interpolation) hit real, confirmed
limits - see that doc's Round 12/13 for the two regressions found along
the way. The actual root cause turned out to be physical, not
algorithmic: Jon spotted a visible paper crease in several raw scans
and confirmed it by eye across 6 independently-checked pages
(z000077117, z000077122, z000077130, z000077132, z000077134,
z000077145) - all from District 81 South Lanark, different townships/
enumerators/sub-districts, all showing the IDENTICAL crease at the same
position relative to the printed form. That consistency across
unrelated pages means it's the bound volume's own gutter/binding fold,
not a one-off handling crease - present on every page in this reel,
not just the one sample this session's research happened to use.

This module detects that crease and corrects the resulting vertical
misalignment via a per-column shift (not a full 2D projective/
cylindrical dewarp - the observed distortion is table ROWS drifting
vertically, not horizontal stretching, so a column-wise vertical
realignment is the right level of correction, and avoids the far
larger complexity of a true optical unwarp). Deliberately reuses
already-validated pieces from core/row_segmentation.py's WIDE-anchor
measurement (`_find_number_row_band_anchors`, `_interpolate_band_center`)
rather than re-deriving them - those are exactly the "reliable at wide
scale" primitives Round 13 already proved out, just applied on each
side of the crease independently instead of across the whole page.

Dependency: cv2 (already used elsewhere in this project - core/
auto_sidecar.py, core/image_analysis.py - not a new addition).
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from core.row_segmentation import _find_number_row_band_anchors, _interpolate_band_center, find_number_row_band


def detect_page_crease_x(
    image: Image.Image,
    search_x_lo: int,
    search_x_hi: int,
    min_coverage: float = 0.5,
    min_peak_ratio: float = 1.5,
) -> tuple[int, float, float] | None:
    """
    Locates a physical paper crease/fold as a vertical structural
    feature running down the FULL PAGE HEIGHT - deliberately not scoped
    to one table-row band like detect_column_number_centers() is.

    Uses the same Scharr-gradient + coverage principle validated twice
    already this session, but for the OPPOSITE reason each time: Round
    6 used plain gradient magnitude and it worked great for hairline
    printed ruling lines. Round 8 added a coverage/continuity gate to
    fix wrong-line lock-on and it REGRESSED, because 1901's printed
    ruling lines turned out to be genuinely broken/discontinuous down
    their length (Round 8's finding). A physical paper CREASE is a
    different kind of feature - a structural fold in the paper itself,
    not printed ink - and should be continuous top-to-bottom in a way a
    printed line on a degraded page isn't. So the coverage gate that
    hurt ruling-line detection is expected to help here; this is a
    genuinely different measurement, not a re-application of a
    technique already shown to be unsound for this page family.

    Returns (crease_x, coverage_at_peak, peak_ratio), or None if no
    column in [search_x_lo, search_x_hi) both clears min_coverage and
    beats min_peak_ratio - i.e. this page may not have a detectable
    crease in that search window, and callers should not invent one.
    """
    gray = np.array(image.convert("L")).astype(np.float32)
    h, w = gray.shape
    lo = max(0, search_x_lo)
    hi = min(w, search_x_hi)
    if hi <= lo:
        return None
    crop = gray[:, lo:hi]
    grad = np.abs(cv2.Scharr(crop, cv2.CV_32F, dx=1, dy=0))
    profile = np.mean(grad, axis=0)
    threshold = float(np.median(grad)) * 2.0
    coverage = np.mean(grad > threshold, axis=0)

    qualifying = coverage >= min_coverage
    if not np.any(qualifying):
        return None
    masked_profile = np.where(qualifying, profile, -np.inf)
    peak_idx = int(np.argmax(masked_profile))
    peak_val = profile[peak_idx]
    background = float(np.median(profile))
    ratio = peak_val / (background + 1e-6)
    if ratio < min_peak_ratio:
        return None
    return lo + peak_idx, float(coverage[peak_idx]), ratio


def find_crease_x_near_prior(
    image: Image.Image, prior_x: int, search_radius: int = 60,
) -> tuple[int, float, float]:
    """
    Round 16: `detect_page_crease_x()`'s blind coverage/ratio thresholds
    found the crease on only 1 of 5 additional real pages tested - not
    because the crease wasn't there (confirmed present on all 5 by eye,
    Round 14), but because its gradient signal is measurably weaker on
    some pages than the 0.5 coverage / 1.5 ratio gate demanded (e.g.
    z000077134's best in-window ratio was only 1.33 - real, but below
    threshold). Jon supplied a strong structural prior instead: the
    crease sits exactly on the printed column 15/16 boundary
    (Nationality/Religion on z000077117 - x=1735, a 0.4992 fraction of
    the table width, essentially identical to the ~1731-1737 this
    session already measured there independently three different ways).

    Given that prior, independent statistical confirmation isn't doing
    useful work anymore - the template's own known geometry already
    supplies the confidence a blind search has to earn from the pixels.
    This function reflects that: NO min_coverage/min_peak_ratio gate,
    just a plain local argmax of Scharr gradient magnitude within a
    tight window around a caller-supplied expected position (compute
    prior_x as table_left + 0.4992 * table_width, or from any other
    known template geometry). Same "prior narrows the search corridor,
    detector still does the actual measurement" pattern used throughout
    this whole investigation for column boundaries - just without a
    confidence gate that was calibrated for the BLIND-search case, where
    something has to prove it's plausible before being trusted.
    """
    gray = np.array(image.convert("L")).astype(np.float32)
    h, w = gray.shape
    lo = max(0, prior_x - search_radius)
    hi = min(w, prior_x + search_radius + 1)
    crop = gray[:, lo:hi]
    grad = np.abs(cv2.Scharr(crop, cv2.CV_32F, dx=1, dy=0))
    profile = np.mean(grad, axis=0)
    threshold = float(np.median(grad)) * 2.0
    coverage = np.mean(grad > threshold, axis=0)
    peak_idx = int(np.argmax(profile))
    background = float(np.median(profile))
    ratio = profile[peak_idx] / (background + 1e-6)
    return lo + peak_idx, float(coverage[peak_idx]), ratio


def _find_overlapping_band_anchors(
    image: Image.Image,
    x0: int,
    x1: int,
    search_y0: int,
    search_y1: int,
    window_width: int,
    window_step: int,
    band_height: int = 12,
    step: int = 3,
) -> list[tuple[float, float]]:
    """
    Finer-resolution alternative to `_find_number_row_band_anchors()`'s
    disjoint n_anchors zones. Round 9-13's whole investigation found
    `find_number_row_band()`'s scorer only stays reliable at wide scale
    (~1000px+) - which caps disjoint zones at ~3 for a ~3100px table
    (any more and each zone shrinks below the reliable threshold, the
    exact mechanism behind Round 10/12's instability). This gets MORE
    sample points WITHOUT shrinking any individual measurement's width,
    by sliding a still-wide window across the table with overlap
    instead of partitioning it into non-overlapping pieces - each
    individual measurement stays exactly as reliable as the disjoint-
    zone case, there are just more of them.

    Windows are clipped at the table boundary rather than requesting
    an under-width edge measurement, and points closer to page_dewarp
    are still expected to be somewhat noisier near the very edges (less
    real form width to work with there regardless of technique) -
    unresolved, not a design claim.
    """
    anchors = []
    half = window_width / 2.0
    x = float(x0)
    while x <= x1:
        wx0 = max(x0, int(round(x - half)))
        wx1 = min(x1, int(round(x + half)))
        if wx1 - wx0 >= window_width * 0.6:  # skip badly-truncated edge windows
            band = find_number_row_band(image, wx0, wx1, search_y0, search_y1, band_height, step)
            if band is not None:
                by0, by1, _blobs, _score = band
                anchors.append((x, (by0 + by1) / 2.0))
        x += window_step
    return anchors


def _side_trend(
    image: Image.Image,
    x0: int,
    x1: int,
    search_y0: int,
    search_y1: int,
    n_anchors: int,
    band_height: int,
    step: int,
) -> list[tuple[float, float]]:
    """Thin wrapper - re-exports row_segmentation's already-validated
    wide-anchor trend measurement for use on one side of a crease."""
    return _find_number_row_band_anchors(image, x0, x1, search_y0, search_y1, n_anchors, band_height, step)


def dewarp_page_at_crease(
    image: Image.Image,
    crease_x: int,
    table_left: int,
    table_right: int,
    search_y0: int,
    search_y1: int,
    n_anchors: int = 3,
    band_height: int = 12,
    step: int = 3,
) -> tuple[Image.Image, dict]:
    """
    Corrects vertical drift via ONE set of WIDE, reliable anchors
    measured across the WHOLE table width - NOT two independently
    re-measured left/right anchor sets split at the crease. An earlier
    version of this function did split the measurement at the crease
    and picked the target y by EXTRAPOLATING the left side's slope out
    to the crease position; that was a real bug, not a design choice -
    the extrapolated value landed outside the range of every actual
    measurement (see docs/COLUMN_NUMBER_ANCHOR_RESEARCH.md's "Round 14"
    for the concrete before/after numbers), so every column got shifted
    the same direction by a large, systematically-biased amount, which
    dragged the raw scan's black photographic border into the visible
    page area near the edges - confirmed by inspecting the corners of
    the first attempt's output directly, not assumed.

    Why one connected trend is the right fix, not just a patch: the
    crease's confirmed position (~x=1700-1731) sits almost exactly on
    top of the middle anchor of the ALREADY-VALIDATED whole-table
    n_anchors=3 measurement from Round 9 (x=1737) - and a direct visual
    check across x=1400-2000 (straddling the crease) earlier this
    session found NO visible discontinuity in the printed content right
    at the fold. Both facts together mean the crease doesn't need to be
    treated as a hard "reset point" for THIS measurement - a normal
    piecewise-linear interpolation through wide, reliable anchors
    already passes through (or very near) the crease naturally, without
    needing to extrapolate anything to build a target value.

    target_y is the anchor CLOSEST to table_left - a real, measured
    value, never an extrapolation - so every other column's shift is
    "how far this column's real measurement differs from that anchor's
    real measurement," not "how far from a projected guess."

    Returns (dewarped_image, debug_info) - the measured anchors, target,
    and max shift, for validation/plotting.
    """
    anchors = _side_trend(image, table_left, table_right, search_y0, search_y1, n_anchors, band_height, step)
    if len(anchors) < 2:
        raise ValueError(
            f"Only {len(anchors)} anchor(s) measured across the table - need at least 2 "
            f"to build a trend. Refusing to guess a correction from an incomplete measurement."
        )

    target_y = anchors[0][1]  # real measured value, never extrapolated

    gray = np.array(image.convert("L"))
    h, w = gray.shape[:2]

    xs = np.arange(w, dtype=np.float32)
    shift_up = np.array([_interpolate_band_center(anchors, float(x)) - target_y for x in xs], dtype=np.float32)

    map_x, map_y = np.meshgrid(xs, np.arange(h, dtype=np.float32))
    map_y = map_y + shift_up[np.newaxis, :]
    # INTER_NEAREST, not INTER_LINEAR: linear interpolation blurs the
    # printed digit edges vertically, which measurably hurt downstream
    # detect_column_number_centers() (its Otsu threshold needs sharp
    # ink boundaries) - confirmed by direct A/B test, not assumed. See
    # docs/COLUMN_NUMBER_ANCHOR_RESEARCH.md's "Round 14" for the numbers.
    dewarped_arr = cv2.remap(
        gray, map_x, map_y, interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=255,
    )
    dewarped = Image.fromarray(dewarped_arr, mode="L")

    debug_info = {
        "crease_x": crease_x,
        "target_y": target_y,
        "anchors": anchors,
        "max_shift_up": float(np.max(np.abs(shift_up))),
    }
    return dewarped, debug_info
