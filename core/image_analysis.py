"""
Stage 1 (Raw Sensor Capture, physical half) per docs/PIPELINE_STAGE_
TERMINOLOGY.md's canonical Stage 0-6 naming (2026-08-02) - previously
called "Stage A" below and elsewhere; that name still appears in
older docs/comments not yet updated (see that doc's rollout status).
PURE MEASUREMENT of a page image. Measures, records, and returns
numbers. Never modifies an image, never chooses a preprocessing
profile, never decides anything.

Built 2026-07-29 per Jon's spec: split the old "rule table picks a
preprocessing function" idea into Stage A/now-Stage-1 (this module -
analysis) and Stage B/now-Stage-2 (Decision Engine - a separate module
that maps these numbers onto a named
profile in core/image_preprocessing.py's PREPROCESSING_PROFILES). His
reasoning, which is the whole design constraint here: "the analyser
becomes reusable - if in six months you replace your preprocessing
policy, you don't have to rewrite the measurements."

TWO ROI DETECTORS, NO DOCUMENT-SPECIFIC ASSUMPTIONS (Jon's revision,
2026-07-29, after the microfilm corpus proved the corpus is no longer
homogeneous - census schedules, passenger manifests, receipts,
typescript on backing board, and microfilm captures with huge borders
now all coexist). Different document types have different regions of
interest: for a census schedule or manifest the ROI is the TABLE, not
the paper; for a letter or receipt there is no table and the ROI is the
PAPER. Rather than this module picking one, it measures BOTH and
reports each with a confidence:

    page_boundary  / page_confidence  / page_method
    table_boundary / table_confidence

Stage B then selects (`if document_class in (census, manifest): ROI =
table_boundary else: ROI = page_boundary`), which keeps every
document-specific assumption out of this module.

PER-REGION MEASUREMENT BLOCKS - WHY THE SCHEMA IS NESTED

Some measurements are geometry-invariant and belong to the whole image:
aspect ratio, deskew angle, ruling-line counts and angles. Others are
meaningless until you say WHICH REGION you mean: Otsu threshold, ink
fraction, polarity, stroke width, illumination statistics (Jon's
distinction, 2026-07-29 - correctly softening this module's earlier and
too-strong claim that "cropping is a prerequisite for the analyser":
it is a prerequisite for SOME measurements, not all).

Since Stage B owns the ROI choice, Stage A cannot know which region to
measure tone inside. Having Stage B do that measurement itself would
mean Stage B opening images - which destroys the entire point of the
split, because retuning policy would once again mean reprocessing every
image. So the ROI-dependent measurements are computed once PER
CANDIDATE REGION and stored in separate `RegionStats` blocks under
`.regions`:

    regions["frame"] - the whole image, always present. Diagnostic
                       baseline; on a microfilm capture this is the
                       block whose tone numbers are meaningless.
    regions["page"]  - present when a page boundary was detected.
    regions["table"] - present when a table boundary was detected.

Stage B reads the block matching its chosen ROI and never touches a
pixel.

WHY THIS MODULE USES OpenCV WHEN THE REST OF THE PROJECT USES PIL+NUMPY

cv2 5.0.0 is installed on the main interpreter (C:\\Python314 - the one
core/ runs on; the .venv_* directories are per-model subprocess loader
environments only). Earlier CV work in this project -
core/image_preprocessing.py's hand-written clahe()/adaptive_threshold(),
core/warp_detection.py's two failed attempts, and the "full auto-dewarp
is a substantially harder problem we're not taking on" decision recorded
in ui/dewarp_preprocessor_ui.py - was all done with PIL + numpy only.
Several measurements below are single well-tested cv2 calls and hundreds
of lines of fragile numpy otherwise. Nothing already working is being
rewritten to use cv2; this is a new module using a library that turned
out to already be available. Note cv2 5.x's HoughLinesP returns (N, 4),
not 4.x's (N, 1, 4).

NO THRESHOLDS, AND WHY THAT DIFFERS SLIGHTLY FROM THE SKETCHED SPEC

Jon's sketch showed Stage A emitting categorical values ("contrast =
low", "blur = moderate"). This module emits the underlying NUMBERS
instead. The cutoff that makes a number count as "low" is a policy
decision, and if it lives here then retuning policy means re-running the
measurement pass over every image - exactly what the split was meant to
avoid. Binning belongs in Stage B, next to the thresholds it uses.

None means NOT MEASURABLE on this image - not zero, and not OK. Stage B
must handle it explicitly. Same abstention discipline as
core/document_classification.py returning doc_type="unknown" rather
than a forced guess.

MEASUREMENTS ARE UNCALIBRATED. Confidences especially: they are honest
ORDINAL signals (more supporting evidence -> higher number), NOT
probabilities, and no cutoff on them has been validated against
anything. Run the CLI over a real manifest and read the CSV before
writing policy against any field here. core/document_classification.py's
run_ratio_threshold story - a plausible-looking 0.5 was wrong, and only
measuring through the real pipeline path revealed it - is the precedent
for not guessing.

MEASURED FINDINGS THAT SHAPED THIS MODULE (2026-07-29, 218 real
microfilm scans from J:\\Screenshots\\Knott_Ancestry\\Archive Microfilms):
  - The unexposed black film surround dominated 215/218 frames (median
    88% of all frame ink in one connected component). On such a frame a
    single Otsu threshold splits PAPER from SURROUND, not INK from
    PAPER - so frame-level otsu_threshold/ink_fraction/polarity are
    wrong there. 43 pages were flagged inverted-polarity at frame level;
    the two most extreme were inspected and both were ordinary
    dark-on-light documents. This is the direct reason polarity is now
    reported per region rather than per image.
  - Sampling the stroke-width distance transform over ALL ink gave a
    median stroke of 165px against a median text height of 11px -
    impossible. Fixed by sampling only text-sized components.
  - Real skew exists in that corpus (median dominant vertical ruling
    angle 1.04 deg, max 12.57) even though this project's 16 labelled
    dewarp pages showed almost none. Don't generalize either way.

SHARED WITH AUTO-DEWARP: `page_boundary` (4 corners, source px, TL/TR/
BR/BL - core/dewarp.py's dewarp_quad() order) and the ruling-line
geometry are the front half of the quad-detection cascade, measured
once here rather than re-derived there with independently drifting
parameters.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict, field, fields
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from core.row_segmentation import estimate_deskew_angle

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORT_PATH = PROJECT_ROOT / "data" / "outputs" / "image_analysis" / "analysis_report.csv"

# Longest side every measurement is taken at. Real scans here run
# 2000-6600px on the long side; measuring at full resolution makes
# scale-sensitive numbers (stroke width, text height, noise) depend on
# each file's own scan DPI rather than on the document, and costs real
# time per page for no gain. Scale-sensitive outputs are in ANALYSIS-SPACE
# px at this size; BOUNDARY coordinates are converted back to source px.
_ANALYSIS_LONG_SIDE = 1600

# Tile grid for illumination unevenness. Applied per region, so on a small
# region this adapts down automatically (see _illumination_unevenness).
_ILLUM_TILE_GRID = 8

# Connected components outside this fraction-of-region-height band are not
# plausibly single text characters and are excluded from text-height and
# stroke-width estimates: below is speckle/JPEG artefacts, above is ruling
# lines, stamps, page edges, merged blobs. Wide on purpose - a sanity
# filter, not a tuned text detector.
_COMPONENT_H_FRAC_RANGE = (0.004, 0.06)

# Morphological line-extraction kernel length, as a fraction of the page's
# own width/height, for something to count as a printed ruling line.
# Deliberately NOT copied from core/document_classification.py's
# run_ratio_threshold=0.35: an opening tolerates the interpolation breakage
# that forced that value down, so the two numbers are not interchangeable
# and their line counts are not comparable.
_RULING_KERNEL_FRAC = 0.25

# Minimum fraction of total image area for a detected 4-gon contour to be a
# plausible page boundary rather than an artefact (a table border, a photo
# within the page, a shadow edge).
_MIN_PAGE_QUAD_AREA_FRAC = 0.30

# Minimum fraction of frame area for the bright-region page fallback. Lower
# than the quad detector's because this path exists for microfilm captures,
# where the paper genuinely occupies a minority of the frame - measured
# median page_quad_area_frac of 0.56, and a small receipt on a large dark
# backing board went well below that.
_MIN_PAGE_REGION_AREA_FRAC = 0.05

# Ruling lines needed on each axis before a table boundary is claimed at
# all. Two per axis is the weakest structure that bounds a region; the
# confidence score (not this gate) is what expresses "barely" vs "clearly".
_MIN_TABLE_LINES_PER_AXIS = 2

# Ruling lines whose position falls within this fraction of a frame edge are
# discarded before a table boundary is computed. Measured 2026-07-29: a
# microfilm frame's own black border is picked up as 2 vertical + 2
# horizontal "ruling lines", which trivially satisfies
# _MIN_TABLE_LINES_PER_AXIS and produced a "table" spanning 99.94% of the
# frame. A real table's outermost rule sits inside the paper, which itself
# sits inside the capture, so a line hugging the frame edge is a capture
# artefact rather than table structure.
_TABLE_EDGE_EXCLUSION_FRAC = 0.02

# Deskew search half-range handed to core/row_segmentation.py's
# estimate_deskew_angle. Its own default is 5.0, which is too narrow for
# this corpus: the microfilm scans measured dominant ruling-line angles up
# to 12.57 deg, and a page whose true skew exceeds the range silently
# returns the range itself (see deskew_angle_clamped).
_DESKEW_ANGLE_RANGE = 15.0

# Ruling lines per axis at which table_confidence saturates at 1.0. A real
# census schedule shows tens (measured 11-41 across the three years in
# document_classification.py's calibration), so this is the point where
# more lines stop adding evidence rather than a target.
_TABLE_CONFIDENCE_SATURATION = 8


@dataclass
class RegionStats:
    """
    The ROI-DEPENDENT measurements, for one named region. See module
    docstring for why these are per-region rather than per-image: every
    field here is meaningless without saying which region it describes,
    as the microfilm polarity finding demonstrated.

    bbox is [x0, y0, x1, y1] in SOURCE-image px, so it can be applied
    directly to the full-resolution original (matching core/dewarp.py's
    sidecars and core/row_segmentation.py's crop_region_from_source,
    which also work in source px). Measurements themselves are taken in
    analysis space - see _ANALYSIS_LONG_SIDE.
    """
    name: str
    bbox: list[float]
    area_frac: float               # of the whole frame

    luminance_mean: float
    luminance_std: float
    contrast_p5_p95_spread: float  # p95 - p5 luminance, 0..255
    otsu_threshold: float
    otsu_separation: float         # between-class variance / total variance, 0..1
    ink_fraction: float            # fraction of px on the dark side of Otsu
    ink_is_dark_on_light: bool     # trustworthy only when largest_ink_blob_frac is low
    illumination_unevenness: float # spread of tile means / 255, 0..1

    text_height_px: float | None   # analysis-space px
    stroke_width_px: float | None  # analysis-space px
    component_count: int

    # Largest single ink component as a fraction of this region's ink. A
    # CONTAMINATION WARNING for the tone fields above, not a document
    # property - see the module docstring's microfilm finding. High value
    # means the region is mostly one solid dark mass, so read its Otsu-
    # derived numbers (otsu_threshold, ink_fraction, ink_is_dark_on_light)
    # as describing that mass rather than the document's ink.
    largest_ink_blob_frac: float


@dataclass
class ImageAnalysis:
    """
    One page's measurements: geometry-invariant fields at the top level,
    ROI-dependent fields inside `.regions` (always "frame"; plus "page"
    and/or "table" when detected).

    Field names correspond to the categories in Jon's Stage A sketch:
      contrast    -> regions[r].contrast_p5_p95_spread
      blur        -> blur_laplacian_var          (whole-frame)
      noise       -> noise_residual_std          (whole-frame)
      grid        -> ruling_lines_vertical / _horizontal
      page_edge   -> page_boundary / page_confidence
      text_height -> regions[r].text_height_px
    """
    file_path: str

    # -- geometry-invariant: valid without any ROI --
    width: int
    height: int
    aspect_ratio: float
    analysis_scale: float          # source px * this = analysis px
    deskew_angle_deg: float        # core/row_segmentation.py's estimator
    # True when the estimate hit +/-_DESKEW_ANGLE_RANGE, i.e. the real angle
    # is AT LEAST this and the number is a floor, not a measurement. Found
    # 2026-07-29: a microfilm page reported exactly -5.00, which is
    # estimate_deskew_angle's own default angle_range, not a coincidence.
    deskew_angle_clamped: bool

    # Blur and noise are whole-frame on purpose: both are properties of the
    # capture (lens focus, sensor/film grain, JPEG quantisation), not of a
    # region, and both are computed from local operators so a large uniform
    # surround dilutes them predictably rather than corrupting them the way
    # a global histogram statistic gets corrupted.
    blur_laplacian_var: float      # LOWER = blurrier
    noise_residual_std: float      # HIGHER = noisier

    ruling_lines_vertical: int
    ruling_lines_horizontal: int
    dominant_vertical_angle_deg: float | None    # 0.0 = perfectly upright
    dominant_horizontal_angle_deg: float | None  # 0.0 = perfectly level

    # -- ROI candidates: both measured, Stage B chooses (see docstring) --
    page_boundary: list[list[float]] | None  # 4 [x, y] source px, TL/TR/BR/BL
    page_confidence: float                   # ordinal, NOT a probability
    page_method: str | None                  # "quad" | "bright_region" | None
    table_boundary: list[float] | None       # [x0, y0, x1, y1] source px
    table_confidence: float                  # ordinal, NOT a probability

    regions: dict[str, RegionStats] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# -- shared helpers ---------------------------------------------------------


def _to_analysis_gray(image: Image.Image) -> tuple[np.ndarray, float]:
    """Grayscale array downscaled so the longest side is
    _ANALYSIS_LONG_SIDE, plus the scale factor applied. Never upscales -
    that would invent detail and inflate every scale-sensitive
    measurement."""
    gray = image.convert("L")
    w, h = gray.size
    long_side = max(w, h)
    scale = min(1.0, _ANALYSIS_LONG_SIDE / long_side) if long_side else 1.0
    if scale < 1.0:
        gray = gray.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    return np.asarray(gray, dtype=np.uint8), scale


def _otsu(arr: np.ndarray) -> tuple[float, float]:
    """(threshold, separation). Separation is the between-class variance at
    the chosen threshold over total variance - the standard Otsu objective,
    normalized. Near 1.0 means a cleanly bimodal ink/paper region; near 0.0
    means one muddy blob, i.e. thresholding it is likely to destroy it."""
    thresh, _ = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hist = np.bincount(arr.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return float(thresh), 0.0
    p = hist / total
    levels = np.arange(256, dtype=np.float64)
    total_var = float(((levels - (p * levels).sum()) ** 2 * p).sum())
    k = int(round(thresh))
    w0 = float(p[: k + 1].sum())
    w1 = 1.0 - w0
    if w0 <= 0 or w1 <= 0 or total_var <= 0:
        return float(thresh), 0.0
    mu0 = float((levels[: k + 1] * p[: k + 1]).sum() / w0)
    mu1 = float((levels[k + 1:] * p[k + 1:]).sum() / w1)
    return float(thresh), float(min(1.0, w0 * w1 * (mu0 - mu1) ** 2 / total_var))


def _illumination_unevenness(arr: np.ndarray) -> float:
    """Percentile spread (p95-p5) of per-tile mean luminance, 0..1.
    Measures LARGE-SCALE lighting variation - what CLAHE fixes and a global
    autocontrast cannot. Percentile rather than max-min so one dark corner
    doesn't define the whole score."""
    h, w = arr.shape
    ty, tx = max(1, h // _ILLUM_TILE_GRID), max(1, w // _ILLUM_TILE_GRID)
    means = [
        float(arr[y: y + ty, x: x + tx].mean())
        for y in range(0, h - ty + 1, ty)
        for x in range(0, w - tx + 1, tx)
    ]
    if len(means) < 2:
        return 0.0
    return float((np.percentile(means, 95) - np.percentile(means, 5)) / 255.0)


def _text_and_stroke(ink: np.ndarray) -> tuple[float | None, float | None, int, float]:
    """
    (median text height, median stroke width, accepted component count,
    largest-ink-blob fraction), analysis-space px.

    Stroke width from the distance transform: for a stroke of width W,
    interior pixels sit at most W/2 from the nearest background, so 2 * (a
    high percentile of the DT) estimates W. 80th percentile, not the max,
    because the max is whatever the single thickest blob is (a stamp, an
    ink blot, a ruling-line junction).

    The DT is computed over the whole ink mask (so a stroke's true
    half-width is measured even where it touches non-text ink) but SAMPLED
    only at accepted text-sized components. Measured 2026-07-29: sampling
    all ink instead gave a median stroke of 165px against a median text
    height of 11px - impossible on its face - because one microfilm-surround
    component held 94.8% of the frame's ink, and the 80th percentile inside
    a blob that size measures the blob, not a stroke.
    """
    n, labels, stats, _centroids = cv2.connectedComponentsWithStats(ink, connectivity=8)
    region_h = ink.shape[0]
    lo, hi = _COMPONENT_H_FRAC_RANGE[0] * region_h, _COMPONENT_H_FRAC_RANGE[1] * region_h

    total_ink = float(ink.sum())
    areas = stats[1:, cv2.CC_STAT_AREA] if n > 1 else np.zeros(0)
    largest_blob_frac = float(areas.max() / total_ink) if areas.size and total_ink else 0.0

    accepted = [i for i in range(1, n) if lo <= stats[i, cv2.CC_STAT_HEIGHT] <= hi]
    if not accepted:
        return None, None, 0, largest_blob_frac

    heights = [float(stats[i, cv2.CC_STAT_HEIGHT]) for i in accepted]
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    text_distances = dist[np.isin(labels, accepted)]
    stroke = float(2.0 * np.percentile(text_distances, 80)) if text_distances.size else None
    return float(np.median(heights)), stroke, len(accepted), largest_blob_frac


def measure_region(
    arr: np.ndarray, name: str, bbox_analysis: tuple[int, int, int, int], scale: float,
) -> RegionStats:
    """
    Every ROI-dependent measurement, for one region of the analysis-space
    array. `bbox_analysis` is (x0, y0, x1, y1) in ANALYSIS px; the stored
    bbox is converted back to SOURCE px so callers can crop the original.

    Public (not underscore-prefixed) because it is the natural entry point
    if a future stage detects a better ROI than this module can and wants
    the same measurement block computed for it - the alternative would be
    re-implementing these seven measurements at that call site.
    """
    x0, y0, x1, y1 = bbox_analysis
    sub = arr[y0:y1, x0:x1]
    if sub.size == 0:
        sub = arr  # a degenerate bbox should not produce a crash-shaped hole
        x0, y0, x1, y1 = 0, 0, arr.shape[1], arr.shape[0]

    thresh, separation = _otsu(sub)
    ink_fraction = float((sub <= thresh).mean())
    # A region whose dark class is the MAJORITY is more likely inverted than
    # genuinely over half ink. 0.5 is a structural fact about which class is
    # bigger, not a tuned cutoff - and see this field's own docstring note
    # for when it cannot be trusted.
    ink_is_dark = ink_fraction <= 0.5
    ink = (sub <= thresh).astype(np.uint8) if ink_is_dark else (sub > thresh).astype(np.uint8)
    text_h, stroke_w, component_count, largest_blob_frac = _text_and_stroke(ink)

    inv = 1.0 / scale if scale else 1.0
    frame_area = float(arr.shape[0] * arr.shape[1])
    return RegionStats(
        name=name,
        bbox=[round(x0 * inv, 1), round(y0 * inv, 1), round(x1 * inv, 1), round(y1 * inv, 1)],
        area_frac=round(float(sub.size) / frame_area, 4) if frame_area else 0.0,
        luminance_mean=round(float(sub.mean()), 2),
        luminance_std=round(float(sub.std()), 2),
        contrast_p5_p95_spread=round(float(np.percentile(sub, 95) - np.percentile(sub, 5)), 2),
        otsu_threshold=round(thresh, 1),
        otsu_separation=round(separation, 4),
        ink_fraction=round(ink_fraction, 4),
        ink_is_dark_on_light=ink_is_dark,
        illumination_unevenness=round(_illumination_unevenness(sub), 4),
        text_height_px=round(text_h, 2) if text_h is not None else None,
        stroke_width_px=round(stroke_w, 2) if stroke_w is not None else None,
        component_count=component_count,
        largest_ink_blob_frac=round(largest_blob_frac, 4),
    )


# -- structure: ruling lines, and the two ROI detectors --------------------


def _ruling_lines(
    ink: np.ndarray,
) -> tuple[int, int, float | None, float | None, np.ndarray, np.ndarray]:
    """
    (vertical count, horizontal count, dominant vertical angle, dominant
    horizontal angle, vertical mask, horizontal mask).

    Counts come from morphological opening with a long 1-D kernel - the
    standard OpenCV table-line extraction - which tolerates the small
    breaks that interpolation introduces. NOT comparable to
    document_classification.py's unbroken-run counts; never feed these to a
    template's column_count_range.

    Angles come from HoughLinesP over each extracted mask, as signed degrees
    from perfectly upright / perfectly level. Tier 2 of the auto-dewarp
    cascade fits vanishing points to these; a flat square page reads ~0.0.

    The masks are returned because _table_boundary() needs the same
    extraction, and running it twice with two sets of parameters is how the
    two would silently drift apart.
    """
    h, w = ink.shape
    scaled = ink * 255

    def extract(kernel_shape, count_axis):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_shape)
        mask = cv2.morphologyEx(scaled, cv2.MORPH_OPEN, kernel, iterations=1)
        # Collapse each detected line to one count regardless of px width -
        # same "a ruled line is several px wide, not one" reasoning as
        # document_classification.py's _count_vertical_ruling_lines().
        profile = mask.max(axis=count_axis) > 0
        count = int(np.count_nonzero(profile[1:] & ~profile[:-1])) + int(bool(profile[0]))
        return mask, count

    v_mask, v_count = extract((1, max(3, int(h * _RULING_KERNEL_FRAC))), 0)
    h_mask, h_count = extract((max(3, int(w * _RULING_KERNEL_FRAC)), 1), 1)

    def dominant_angle(mask, vertical: bool) -> float | None:
        min_len = max(20, int((h if vertical else w) * _RULING_KERNEL_FRAC))
        segments = cv2.HoughLinesP(
            mask, rho=1, theta=np.pi / 720, threshold=80,
            minLineLength=min_len, maxLineGap=10,
        )
        if segments is None or len(segments) == 0:
            return None
        # cv2 5.x returns (N, 4); 4.x returned (N, 1, 4). Reshape covers both
        # rather than losing a batch run to a silent unpack failure.
        angles = []
        for x1, y1, x2, y2 in np.asarray(segments).reshape(-1, 4):
            dx, dy = float(x2 - x1), float(y2 - y1)
            if vertical and abs(dy) > 1e-6:
                angles.append(np.degrees(np.arctan2(dx, dy)))
            elif not vertical and abs(dx) > 1e-6:
                angles.append(np.degrees(np.arctan2(dy, dx)))
        # Only near-axis segments: a diagonal survivor of the opening is a
        # mis-detection, not a skewed ruling line, and would drag a mean off.
        # Median over the rest, same robustness reason.
        angles = [a for a in angles if abs(a) < 30.0]
        return float(np.median(angles)) if angles else None

    return (
        v_count, h_count,
        dominant_angle(v_mask, True), dominant_angle(h_mask, False),
        v_mask, h_mask,
    )


def _order_quad(pts: np.ndarray) -> list[list[float]]:
    """Orders 4 points TL, TR, BR, BL - the order core/dewarp.py's
    compute_homography()/dewarp_quad() expects, so a detected quad hands
    straight to it with no reordering at the call site."""
    pts = pts.astype(np.float64).reshape(4, 2)
    by_sum, by_diff = pts.sum(axis=1), pts[:, 0] - pts[:, 1]
    corners = (
        pts[int(np.argmin(by_sum))], pts[int(np.argmax(by_diff))],
        pts[int(np.argmax(by_sum))], pts[int(np.argmin(by_diff))],
    )
    return [[float(p[0]), float(p[1])] for p in corners]


def _page_boundary(
    arr: np.ndarray,
) -> tuple[list[list[float]] | None, float, str | None, tuple[int, int, int, int] | None]:
    """
    (quad in ANALYSIS px, confidence, method, bbox in ANALYSIS px).

    Two methods, tried in order, because they fail on opposite document
    types:

    "quad" - Canny + approxPolyDP for a convex 4-gon covering at least
      _MIN_PAGE_QUAD_AREA_FRAC. Gives true corners, so it is the only path
      that can feed a perspective correction. Fails when the page fills the
      frame with no visible margin. Measured 2026-07-29 against this
      project's 16 human-labelled corner sets: 14/16 detected, median
      corner error 1.95% of page width, worst 4.69% - which is LARGER than
      the ~1% keystone those humans were correcting, so this is not yet
      accurate enough to drive a perspective correction unsupervised.
      Errors clustered bimodally (~1.2% and ~4.5%), i.e. a systematic
      wrong-boundary pick rather than noise.

    "bright_region" - largest bright connected region above
      _MIN_PAGE_REGION_AREA_FRAC. Built for microfilm captures, where the
      paper is a bright island in a dark surround and the quad detector
      found nothing on 96/218 frames. Returns an axis-aligned bbox, so its
      "quad" is a rectangle and carries NO perspective information.

      KNOWN WRONG on a real, reproducible case - do not trust this method's
      boundary without checking it. On
      oocihm.lac_reel_c10639.733.jpg it returned the white "PUBLIC
      ARCHIVES / ARCHIVES PUBLIQUES CANADA" label board at the bottom of
      the frame (bbox y 3117-3727 of 3840) rather than the document, which
      occupies most of the frame above it. Cause: the assumption "the paper
      is the brightest large thing" fails on aged microfilm, where the
      document paper is mid-grey (frame luminance mean 103) and the modern
      printed archive label is genuinely brighter. Its confidence is capped
      at 0.55 for this reason - strictly below any "quad" result - but a cap
      is not a fix. Replacing this with a detector benchmarked against real
      labelled boundaries is the auto-crop work (build step 2), not
      something to guess at again here.

    Confidence is ordinal, not a probability: the quad path scores higher
    because true corners are strictly more informative than a bbox, and
    within each path a larger, more page-like region scores higher.
    """
    frame_area = float(arr.shape[0] * arr.shape[1])
    blurred = cv2.GaussianBlur(arr, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)
    # Close small gaps so an edge broken by shadow or low contrast still
    # forms one closed contour instead of several open arcs.
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    best, best_area = None, 0.0
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        area = abs(float(cv2.contourArea(approx)))
        if area / frame_area >= _MIN_PAGE_QUAD_AREA_FRAC and area > best_area:
            best, best_area = approx, area

    if best is not None:
        quad = _order_quad(best)
        xs = [p[0] for p in quad]
        ys = [p[1] for p in quad]
        bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
        area_frac = best_area / frame_area
        return quad, round(min(1.0, 0.6 + 0.4 * area_frac), 3), "quad", bbox

    thresh, _ = _otsu(blurred)
    bright = (blurred > thresh).astype(np.uint8)
    # Close gaps so printed text inside the paper doesn't fragment the paper
    # region into hundreds of pieces around the writing.
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, _labels, stats, _c = cv2.connectedComponentsWithStats(bright, connectivity=8)
    if n <= 1:
        return None, 0.0, None, None

    i = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    area_frac = float(stats[i, cv2.CC_STAT_AREA]) / frame_area
    if area_frac < _MIN_PAGE_REGION_AREA_FRAC:
        return None, 0.0, None, None

    x0, y0 = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
    x1 = x0 + int(stats[i, cv2.CC_STAT_WIDTH])
    y1 = y0 + int(stats[i, cv2.CC_STAT_HEIGHT])
    quad = [[float(x0), float(y0)], [float(x1), float(y0)],
            [float(x1), float(y1)], [float(x0), float(y1)]]
    return quad, round(min(0.55, 0.2 + 0.5 * area_frac), 3), "bright_region", (x0, y0, x1, y1)


def _table_boundary(
    v_mask: np.ndarray, h_mask: np.ndarray, v_count: int, h_count: int,
) -> tuple[tuple[int, int, int, int] | None, float]:
    """
    (bbox in ANALYSIS px, confidence) for the ruled table region - the ROI
    for a census schedule or passenger manifest, where the table and not
    the paper is what the extraction pipeline consumes (Jon, 2026-07-29).

    Bounded by the EXTENT OF THE DETECTED RULING LINES themselves: the
    outermost long vertical lines give x, the outermost long horizontal
    lines give y. Requires at least _MIN_TABLE_LINES_PER_AXIS on each axis -
    a single line on an axis bounds nothing, and a page with no ruling at
    all (a letter, a receipt) correctly yields None rather than a
    fabricated box.

    Confidence is ordinal: it scales with how many lines were found on the
    weaker axis, saturating at _TABLE_CONFIDENCE_SATURATION. Deliberately
    keyed to the WEAKER axis - a page showing 40 horizontal rules and 2
    verticals is not clearly a bounded table.
    """
    if v_count < _MIN_TABLE_LINES_PER_AXIS or h_count < _MIN_TABLE_LINES_PER_AXIS:
        return None, 0.0

    frame_h, frame_w = v_mask.shape
    v_cols = np.flatnonzero(v_mask.max(axis=0) > 0)
    h_rows = np.flatnonzero(h_mask.max(axis=1) > 0)

    # Drop frame-hugging lines (the microfilm border) before measuring
    # extent - see _TABLE_EDGE_EXCLUSION_FRAC.
    v_margin = _TABLE_EDGE_EXCLUSION_FRAC * frame_w
    h_margin = _TABLE_EDGE_EXCLUSION_FRAC * frame_h
    v_cols = v_cols[(v_cols >= v_margin) & (v_cols <= frame_w - v_margin)]
    h_rows = h_rows[(h_rows >= h_margin) & (h_rows <= frame_h - h_margin)]
    if v_cols.size == 0 or h_rows.size == 0:
        return None, 0.0

    # Recount lines from the surviving positions: a run of adjacent columns
    # is one line, same collapse rule as _ruling_lines()'s own counting. The
    # incoming v_count/h_count include the discarded border lines, so using
    # them for confidence would credit evidence that was just rejected.
    def runs(positions: np.ndarray) -> int:
        return 1 + int(np.count_nonzero(np.diff(positions) > 1)) if positions.size else 0

    v_kept, h_kept = runs(v_cols), runs(h_rows)
    if v_kept < _MIN_TABLE_LINES_PER_AXIS or h_kept < _MIN_TABLE_LINES_PER_AXIS:
        return None, 0.0

    bbox = (int(v_cols[0]), int(h_rows[0]), int(v_cols[-1]) + 1, int(h_rows[-1]) + 1)
    weaker = min(v_kept, h_kept)
    confidence = min(1.0, (weaker - 1) / float(_TABLE_CONFIDENCE_SATURATION))
    return bbox, round(confidence, 3)


# -- top level -------------------------------------------------------------


def analyze_image(image: Image.Image | str | Path) -> ImageAnalysis:
    """
    Measures one page. Accepts a PIL Image (matching
    core/document_classification.py's classify_document) or a path
    (convenient for batch use). Does not modify the image and writes
    nothing - see save_analysis()/analyze_manifest() for persistence.
    """
    if isinstance(image, (str, Path)):
        file_path = str(image)
        with Image.open(image) as opened:
            pil = opened.convert("RGB")
    else:
        file_path = str(getattr(image, "filename", "") or "")
        pil = image.convert("RGB")

    source_w, source_h = pil.size
    arr, scale = _to_analysis_gray(pil)
    frame_h, frame_w = arr.shape

    # Frame-level ink mask, used only for ruling-line extraction. Its
    # polarity comes from the frame's own Otsu, which the microfilm finding
    # showed can be wrong - acceptable here specifically because a ruling
    # line is a long thin run either way, and both masks are searched.
    frame_thresh, _ = _otsu(arr)
    frame_ink = (arr <= frame_thresh).astype(np.uint8)
    v_count, h_count, v_angle, h_angle, v_mask, h_mask = _ruling_lines(frame_ink)

    deskew_angle = float(estimate_deskew_angle(pil, angle_range=_DESKEW_ANGLE_RANGE))
    quad, page_conf, page_method, page_bbox = _page_boundary(arr)
    table_bbox, table_conf = _table_boundary(v_mask, h_mask, v_count, h_count)

    regions = {"frame": measure_region(arr, "frame", (0, 0, frame_w, frame_h), scale)}
    if page_bbox is not None:
        regions["page"] = measure_region(arr, "page", page_bbox, scale)
    if table_bbox is not None:
        regions["table"] = measure_region(arr, "table", table_bbox, scale)

    inv = 1.0 / scale if scale else 1.0
    return ImageAnalysis(
        file_path=file_path,
        width=source_w,
        height=source_h,
        aspect_ratio=round(source_w / source_h, 4) if source_h else 0.0,
        analysis_scale=round(scale, 4),
        deskew_angle_deg=round(deskew_angle, 3),
        deskew_angle_clamped=abs(deskew_angle) >= _DESKEW_ANGLE_RANGE - 1e-6,
        blur_laplacian_var=round(float(cv2.Laplacian(arr, cv2.CV_64F).var()), 2),
        noise_residual_std=round(
            float(np.abs(arr.astype(np.int16) - cv2.medianBlur(arr, 3).astype(np.int16)).std()), 3
        ),
        ruling_lines_vertical=v_count,
        ruling_lines_horizontal=h_count,
        dominant_vertical_angle_deg=round(v_angle, 3) if v_angle is not None else None,
        dominant_horizontal_angle_deg=round(h_angle, 3) if h_angle is not None else None,
        page_boundary=[[round(x * inv, 1), round(y * inv, 1)] for x, y in quad] if quad else None,
        page_confidence=page_conf,
        page_method=page_method,
        table_boundary=[round(v * inv, 1) for v in table_bbox] if table_bbox else None,
        table_confidence=table_conf,
        regions=regions,
    )


def analysis_sidecar_path(image_path: str | Path) -> Path:
    """Sibling `<name>_analysis.json` - same sidecar-beside-its-subject
    convention as core/dewarp.py's dewarp_sidecar_path()."""
    path = Path(image_path)
    return path.with_name(f"{path.stem}_analysis.json")


def save_analysis(analysis: ImageAnalysis, sidecar_path: str | Path | None = None) -> Path:
    """Persists one page's measurements. Full overwrite, matching
    save_dewarp_sidecar()'s convention - a re-measurement genuinely
    supersedes the previous one."""
    if sidecar_path is None:
        sidecar_path = analysis_sidecar_path(analysis.file_path)
    sidecar_path = Path(sidecar_path)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(json.dumps(analysis.to_dict(), indent=2), encoding="utf-8")
    return sidecar_path


def load_analysis(sidecar_path: str | Path) -> ImageAnalysis:
    """Reads a sidecar back. This is the call that lets Stage B re-run
    without re-measuring anything - the entire point of the split.

    Rejects a schema mismatch loudly rather than silently filling defaults:
    a policy tuned against fields that aren't really there is a worse
    outcome than an error telling you to re-run the analyser.
    """
    record = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
    top = {f.name for f in fields(ImageAnalysis)}
    region_keys = {f.name for f in fields(RegionStats)}

    for label, known, present in (
        ("ImageAnalysis", top, set(record)),
        *[
            (f"regions[{name!r}]", region_keys, set(block))
            for name, block in (record.get("regions") or {}).items()
        ],
    ):
        if present - known or known - present:
            raise ValueError(
                f"{sidecar_path}: {label} schema mismatch - unexpected "
                f"{sorted(present - known)}, missing {sorted(known - present)}. "
                f"Written by a different version of core/image_analysis.py; "
                f"re-run the analyser rather than mixing schema versions."
            )

    regions = {name: RegionStats(**block) for name, block in (record.get("regions") or {}).items()}
    return ImageAnalysis(**{**record, "regions": regions})


# Region blocks flattened into the CSV as "<region>_<field>" columns, for
# these regions in this order. A nested dict is not a CSV cell, and one
# wide flat table is what makes the corpus sortable in a spreadsheet -
# which is how Stage B's thresholds are supposed to get derived.
_CSV_REGIONS = ("frame", "page", "table")
_CSV_JSON_FIELDS = ("page_boundary", "table_boundary")


def _csv_columns() -> list[str]:
    top = [f.name for f in fields(ImageAnalysis) if f.name != "regions"]
    region_fields = [f.name for f in fields(RegionStats) if f.name != "name"]
    return top + [
        f"{region}_{name}" for region in _CSV_REGIONS for name in region_fields
    ] + ["error"]


def _csv_row(analysis: ImageAnalysis) -> dict:
    record = analysis.to_dict()
    regions = record.pop("regions", {}) or {}
    for key in _CSV_JSON_FIELDS:
        record[key] = json.dumps(record[key]) if record.get(key) is not None else ""
    for region in _CSV_REGIONS:
        block = regions.get(region) or {}
        for f in fields(RegionStats):
            if f.name == "name":
                continue
            value = block.get(f.name)
            if isinstance(value, list):
                value = json.dumps(value)
            # "" not 0 for an absent region - a page with no table must not
            # read as a table of zero size once it reaches a spreadsheet.
            record[f"{region}_{f.name}"] = "" if value is None else value
    record["error"] = ""
    return record


def analyze_manifest(
    manifest_path: str | Path,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    write_sidecars: bool = True,
) -> Path:
    """
    Measures every image in a manifest CSV's "file_path" column: one
    sidecar per image (unless write_sidecars=False) plus a rolled-up CSV
    at report_path, one row per page with region blocks flattened into
    prefixed columns. Reading that CSV is the intended way to derive Stage
    B thresholds from real numbers instead of guessing them.

    A page that fails to measure gets a row with an `error` column and does
    not abort the run - one corrupt file partway through a corpus should
    not cost every measurement taken so far (same tolerance as
    manifest_pipeline.build_working_manifest()'s per-file try/except).
    """
    from core.bucket_worklist import load_bucket_filepaths

    file_paths = load_bucket_filepaths(manifest_path)
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    written = failed = 0

    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_csv_columns())
        writer.writeheader()

        for i, file_path in enumerate(file_paths, 1):
            print(f"[{i}/{len(file_paths)}] {Path(file_path).name}", end=" ")
            try:
                analysis = analyze_image(file_path)
            except Exception as e:
                failed += 1
                print(f"FAILED ({type(e).__name__}: {e})")
                writer.writerow({"file_path": file_path, "error": f"{type(e).__name__}: {e}"})
                continue

            writer.writerow(_csv_row(analysis))
            written += 1
            if write_sidecars:
                save_analysis(analysis)

            roi = analysis.regions.get("table") or analysis.regions.get("page")
            print(
                f"page={analysis.page_method or 'no'}({analysis.page_confidence}) "
                f"table={'yes' if analysis.table_boundary else 'no'}"
                f"({analysis.table_confidence}) "
                f"skew={analysis.deskew_angle_deg:+.2f} "
                f"blur={analysis.blur_laplacian_var} "
                f"roi_contrast={roi.contrast_p5_p95_spread if roi else '-'} "
                f"roi_text_h={roi.text_height_px if roi else '-'}"
            )

    print(f"\nAnalysis report written to {report_path} ({written} measured, {failed} failed).")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage A image analysis - measures pages, modifies nothing.",
    )
    parser.add_argument(
        "target", help="An image file, or a manifest/bucket CSV with a 'file_path' column.",
    )
    parser.add_argument(
        "--report", default=str(DEFAULT_REPORT_PATH),
        help=f"CSV report path for CSV input (default: {DEFAULT_REPORT_PATH})",
    )
    parser.add_argument(
        "--no-sidecars", action="store_true",
        help="Write only the rolled-up CSV, no per-image *_analysis.json sidecars.",
    )
    args = parser.parse_args()

    target = Path(args.target)
    if target.suffix.lower() == ".csv":
        analyze_manifest(target, args.report, write_sidecars=not args.no_sidecars)
        return

    analysis = analyze_image(target)
    print(json.dumps(analysis.to_dict(), indent=2))
    if not args.no_sidecars:
        print(f"\nSidecar: {save_analysis(analysis)}")


if __name__ == "__main__":
    main()
