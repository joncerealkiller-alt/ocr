"""
Perspective-warp DETECTION (not correction) - a pass/fail signal for
whether a dense_tabular_rows page (census/manifest) needs the full manual
4-point dewarp tool (core/dewarp.py, ui/dewarp_preprocessor_ui.py), or is
flat enough to proceed automatically. Scoped to dense_tabular_rows only
(Jon's direction, 2026-07-28) - every other bucket is a single whole-image
model call, which tolerates mild warp/skew fine; only the tight per-field
crop pipeline (core/row_extraction.py) actually depends on precise page
geometry.

Deliberately NOT trying to solve full auto-dewarp (locating the 4 real
corners automatically) - that's a substantially harder problem this
project has already decided not to take on (see ui/dewarp_preprocessor_ui
.py's own docstring). Detecting THAT a page is warped is a more tractable
sub-problem: the failure mode of a wrong "needs review" flag is a human
looking at one extra page, not a silently-wrong extraction downstream -
same asymmetric-cost reasoning behind every other quarantine gate in this
project (core/auto_sidecar.py's PAGE_QUARANTINE_THRESHOLD, etc.).

METHOD, v2 (2026-07-28) - REPLACES the original approach after real
calibration evidence disproved it (see diagnostics/
test_warp_detection_calibration.py's git history / this module's own
earlier version if you need the postmortem): v1 compared row-BAND
Y-positions between the left and right halves of the page, using
core/row_segmentation.py's detect_row_bands() (built for general text-
density projection). Tested against 23 real raw/dewarped pairs from this
project, it was inconclusive on 18/23 (band counts disagreed too much
between the narrow half-window slices to trust), and on several of the
remaining 5 it scored the ALREADY-DEWARPED image higher than its raw
counterpart - backwards. Text-density row bands are too noisy a signal
at this scale, especially on handwriting.

v2 uses printed RULING LINES instead - the same column-axis longest-run
technique already proven in core/document_classification.py's
_count_vertical_ruling_lines() (reused here via row_segmentation.py's
_otsu_threshold/_longest_run_per_row, the same private-helper reuse
pattern core/auto_sidecar.py already relies on throughout). Real printed
vertical column-separator lines are long, high-contrast, and reliably
detected on dense_tabular_rows forms specifically (unlike general text
density) - exactly the document type this module is now scoped to.

Real vertical lines on flat paper are physically straight. Measured near
the TOP of the page vs. near the BOTTOM of the page, a given line's
X-position should closely agree if the page is flat; under perspective
distortion (or a photographed/curled page), the same physical line's
X-position DRIFTS between top and bottom - the classic keystone
signature. This is measured directly rather than inferred from row
density, which is why it's expected to be more robust for this
document category specifically.

CALIBRATION (2026-07-28, real evidence, not guessed): tested against the
same 23 real raw/dewarped pairs used for v1 - see
diagnostics/test_warp_detection_calibration.py. _WARP_SCORE_THRESHOLD
below is set from those real numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from core.row_segmentation import _otsu_threshold, _longest_run_per_row

# Bands scanned near the top and bottom of the page for ruling-line
# X-positions - kept away from the very top/bottom margin (often noisy:
# header text, stamps, torn edges) the same way _SCAN_Y_FRAC does in
# document_classification.py, and kept narrow (a 0.20-height band) so
# each band's own detected X-positions are a genuinely LOCAL measurement,
# not blurred across a large chunk of the page's own possible curvature.
_TOP_BAND_FRAC = (0.08, 0.28)
_BOTTOM_BAND_FRAC = (0.72, 0.92)

# Column-axis longest-run-per-column threshold, as a fraction of the
# scanned band's own height - same value and meaning as document_
# classification.py's _count_vertical_ruling_lines() default, not
# re-derived blind.
_RUN_RATIO_THRESHOLD = 0.35

# A matched line's X-offset between the top and bottom band, as a
# fraction of the page's own median inter-line spacing - calibrated
# against real measured pairs (see module docstring). Below this: page
# reads as flat. At/above: flagged as needing manual dewarp review.
_WARP_SCORE_THRESHOLD = 0.5

# Matched lines only get compared if the top and bottom bands found
# within this many of the same count - a big mismatch means the
# comparison itself isn't trustworthy (e.g. one band's own content
# obscured a real line), not that the page is necessarily warped.
_MAX_LINE_COUNT_MISMATCH = 3


@dataclass
class WarpDetectionResult:
    needs_dewarp: bool
    warp_score: float  # median |top_x - bottom_x| / median_line_spacing, or -1.0 if inconclusive
    low_confidence: bool
    top_line_count: int
    bottom_line_count: int
    matched_line_count: int
    offsets_px: list[float] = field(default_factory=list)


def _extract_line_positions(is_line: np.ndarray) -> list[float]:
    """Groups contiguous True runs into single line-center X positions -
    a ruled line is several px wide, not a single column, same grouping
    idea as _count_vertical_ruling_lines()'s transition-counting, but
    keeping the actual center position instead of just incrementing a
    counter."""
    positions = []
    start = None
    for i, v in enumerate(is_line):
        if v and start is None:
            start = i
        elif not v and start is not None:
            positions.append((start + i - 1) / 2.0)
            start = None
    if start is not None:
        positions.append((start + len(is_line) - 1) / 2.0)
    return positions


def _find_ruling_line_positions(
    image: Image.Image, y0: int, y1: int, run_ratio_threshold: float = _RUN_RATIO_THRESHOLD,
) -> list[float]:
    gray = image.convert("L")
    arr = np.array(gray)[y0:y1, :]
    if arr.size == 0:
        return []
    thresh = _otsu_threshold(arr)
    binary = (arr <= thresh).astype(np.uint8)
    region_h = y1 - y0
    longest = _longest_run_per_row(binary.T, close_gap_px=4)
    is_line = (longest / region_h) > run_ratio_threshold if region_h > 0 else np.zeros(0, dtype=bool)
    return _extract_line_positions(is_line)


def detect_warp(image: Image.Image) -> WarpDetectionResult:
    """
    Runs the top-band vs. bottom-band ruling-line comparison described
    in this module's docstring. Caller is responsible for deskewing
    `image` first (core/row_segmentation.py's estimate_deskew_angle/
    apply_deskew_angle) - simple global rotation is a separate, already-
    solved problem; running this on a still-rotated image would conflate
    the two and produce a meaningless signal.
    """
    w, h = image.size
    top_y0, top_y1 = int(h * _TOP_BAND_FRAC[0]), int(h * _TOP_BAND_FRAC[1])
    bot_y0, bot_y1 = int(h * _BOTTOM_BAND_FRAC[0]), int(h * _BOTTOM_BAND_FRAC[1])

    top_lines = sorted(_find_ruling_line_positions(image, top_y0, top_y1))
    bottom_lines = sorted(_find_ruling_line_positions(image, bot_y0, bot_y1))

    if not top_lines or not bottom_lines:
        return WarpDetectionResult(
            needs_dewarp=False, warp_score=-1.0, low_confidence=True,
            top_line_count=len(top_lines), bottom_line_count=len(bottom_lines),
            matched_line_count=0,
        )

    if abs(len(top_lines) - len(bottom_lines)) > _MAX_LINE_COUNT_MISMATCH:
        return WarpDetectionResult(
            needs_dewarp=False, warp_score=-1.0, low_confidence=True,
            top_line_count=len(top_lines), bottom_line_count=len(bottom_lines),
            matched_line_count=0,
        )

    n = min(len(top_lines), len(bottom_lines))
    offsets = [abs(top_lines[i] - bottom_lines[i]) for i in range(n)]

    all_positions = sorted(top_lines + bottom_lines)
    spacings = [b - a for a, b in zip(all_positions, all_positions[1:]) if b > a]
    median_spacing = sorted(spacings)[len(spacings) // 2] if spacings else 1.0
    median_spacing = max(median_spacing, 1.0)

    offsets_sorted = sorted(offsets)
    median_offset = offsets_sorted[len(offsets_sorted) // 2]
    score = median_offset / median_spacing

    return WarpDetectionResult(
        needs_dewarp=score >= _WARP_SCORE_THRESHOLD,
        warp_score=score,
        low_confidence=False,
        top_line_count=len(top_lines),
        bottom_line_count=len(bottom_lines),
        matched_line_count=n,
        offsets_px=offsets,
    )
