"""
Rule-based document classification for the automated sidecar generation
pipeline (core/auto_sidecar.py) - deliberately NOT a model call. This
runs before anything in core/loaders/ is touched, so it carries none of
CLAUDE.md's GPU-contention concerns.

Built 2026-07-27. Classifies against three structural signals, all
cheap (PIL + numpy, no OCR, reusing core/row_segmentation.py's Otsu
binarization and ruling-line run-length helpers rather than
reimplementing them):

  - aspect_ratio: page width/height. Cleanly separates the (landscape)
    census templates from the (portrait) passenger manifest - measured
    2026-07-27 on one real sample per doc type: census 1.57-1.64,
    manifest 0.88. The single strongest, cheapest signal this module
    has.

  - vertical_ruling_lines: count of tall, mostly-unbroken vertical dark
    runs (printed column dividers) within the page's likely table
    y-range. This is the PRIMARY signal for telling the three census
    years apart, since they all share the same 50-line-per-page layout
    and near-identical aspect ratio - row_count and aspect_ratio alone
    can't distinguish them.

    run_ratio_threshold=0.35 (not the more obvious-looking 0.5) is
    deliberate, found the hard way (2026-07-27): the FIRST calibration
    pass measured 1921/1911/1931 at 8/20/25 directly against the raw,
    un-deskewed dewarped JPEGs - but classify_document() always runs on
    the ALREADY-DESKEWED image (even a small ~0.5deg auto-correction),
    and that rotation's interpolation is enough to break a meaningful
    fraction of ruling-line pixels out of a strict "how much of this
    column is unbroken" measurement, collapsing 1911's real pipeline
    count from 20 to 9 - i.e. indistinguishable from 1921 at threshold
    0.5. Re-measured through the ACTUAL pipeline path (deskew ->
    count) at threshold 0.35: 1921 ~11, 1911 ~21, 1931 ~41, manifest 0.
    Every template's classification.column_count_range is calibrated
    against THESE numbers, not the original raw-image ones - if this
    threshold ever changes, the ranges in config/document_templates/
    *.yaml need re-measuring against real deskewed pipeline output
    again, not adjusted by feel.

  - row_count_estimate: kept-band count from a whole-page (unscoped)
    detect_row_bands -> merge_wrapped_bands -> sanity_check_bands pass.
    Deliberately the LOWEST-weighted signal: run unscoped (no known
    table x/y range yet - that's what classification feeds INTO, not
    what it can assume), it's noisy on real census scans (dense ruling
    causes heavy merging - measured 3-7 "rows" on real census pages
    that actually have 50, since header/metadata get folded in too).
    Kept anyway because it still separates "some tabular repetition
    detected" from "none" - useful mainly as a tiebreaker, not a
    primary discriminator.

Honest limitation, not hidden (matching this project's established
practice of documenting real gaps rather than glossing over them - see
core/row_segmentation.py's docstrings for the same pattern): distinguishing
1911 from 1921 by structure alone is inherently harder than either
vs. 1931 (which prints visibly more columns) or vs. the manifest
(different aspect ratio entirely). Expect closer confidence scores
between 1911/1921 than the other splits - that reflects genuine
structural similarity, not a bug in the scorer. True disambiguation
would need OCR against the printed header text, out of scope for a
rule-based, pre-OCR classification stage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from core.document_templates import DocumentTemplate, load_all_templates
from core.row_segmentation import (
    _otsu_threshold,
    _longest_run_per_row,
    detect_row_bands,
    merge_wrapped_bands,
    sanity_check_bands,
)

UNKNOWN_CONFIDENCE_THRESHOLD = 0.5

# Generic scan window for vertical-ruling-line counting - NOT any one
# template's approx table region (classification runs BEFORE a
# template is chosen, so it can't assume one). Covers the y-range where
# a table body plausibly sits across every known template (narrowest
# real table_top measured was 1931's 0.224; widest table_bottom was
# manifest's ~0.999) while staying clear of page-edge margin noise.
_SCAN_Y_FRAC = (0.25, 0.95)


@dataclass
class ClassificationResult:
    doc_type: str                    # e.g. "canada_census_1921", or "unknown"
    confidence: float                # 0..1
    template_name: str | None        # same as doc_type if matched, else None
    features: dict                   # measured signals, for --debug diagnostics
    scores: dict                     # per-template confidence, for --debug diagnostics


def _count_vertical_ruling_lines(
    image: Image.Image, y_frac: tuple[float, float], run_ratio_threshold: float = 0.35,
) -> int:
    """
    Column-axis counterpart to detect_row_bands()'s horizontal ruling-
    line detection: binarize (Otsu), then reuse _longest_run_per_row on
    the TRANSPOSED binary array so "longest run along a column" falls
    out of the same row-wise helper without duplicating it. A column
    whose longest unbroken dark run exceeds run_ratio_threshold of the
    scanned region's height is a printed divider; adjacent flagged
    columns are merged into one line (a ruled line is several px wide,
    not one).
    """
    w, h = image.size
    y0, y1 = int(h * y_frac[0]), int(h * y_frac[1])
    gray = image.convert("L")
    arr = np.array(gray)[y0:y1, :]
    if arr.size == 0:
        return 0
    thresh = _otsu_threshold(arr)
    binary = (arr <= thresh).astype(np.uint8)
    region_h = y1 - y0
    longest = _longest_run_per_row(binary.T, close_gap_px=4)
    is_line = (longest / region_h) > run_ratio_threshold if region_h > 0 else np.zeros(0, dtype=bool)

    count = 0
    prev = False
    for v in is_line:
        if v and not prev:
            count += 1
        prev = bool(v)
    return count


def _estimate_row_count(image: Image.Image) -> int:
    """Whole-page, unscoped kept-band count - see module docstring for
    why this is the lowest-weighted, tiebreaker-only signal."""
    bands, _, ruling_rows = detect_row_bands(image)
    merged = merge_wrapped_bands(bands, ruling_line_rows=ruling_rows)
    kept, _, _ = sanity_check_bands(merged)
    return len(kept)


def _range_score(value: float, lo: float, hi: float) -> float:
    """1.0 inside [lo, hi], decaying linearly to 0 at 1x the range's own
    width beyond either edge - a value just outside a range is a near
    miss, not an instant disqualification."""
    if lo <= value <= hi:
        return 1.0
    span = max(hi - lo, 1e-6)
    if value < lo:
        return max(0.0, 1.0 - (lo - value) / span)
    return max(0.0, 1.0 - (value - hi) / span)


def classify_document(
    image: Image.Image, templates: dict[str, DocumentTemplate] | None = None,
) -> ClassificationResult:
    """
    Scores every known template against measured structural features
    and returns the best match, or doc_type="unknown" if the best
    score falls below UNKNOWN_CONFIDENCE_THRESHOLD - an honest "I don't
    know" is the correct output for a page this pipeline has no
    template for, not a forced best-effort guess (a wrong guess here
    silently mis-scopes every downstream boundary locator).
    """
    templates = templates if templates is not None else load_all_templates()

    aspect_ratio = image.width / image.height
    column_count = _count_vertical_ruling_lines(image, _SCAN_Y_FRAC)
    row_count = _estimate_row_count(image)
    features = {
        "aspect_ratio": round(aspect_ratio, 3),
        "vertical_ruling_lines": column_count,
        "row_count_estimate": row_count,
    }

    scores: dict[str, float] = {}
    for doc_type, tmpl in templates.items():
        c = tmpl.classification
        aspect_score = _range_score(aspect_ratio, *c["aspect_ratio_range"])
        column_score = _range_score(column_count, *c["column_count_range"])
        row_score = _range_score(row_count, *c["row_count_range"])
        # aspect_ratio and column_count are the reliable discriminators
        # (see module docstring); row_count is a noisy tiebreaker only.
        scores[doc_type] = 0.4 * aspect_score + 0.4 * column_score + 0.2 * row_score

    if not scores:
        return ClassificationResult("unknown", 0.0, None, features, scores)

    best_doc_type = max(scores, key=scores.get)
    best_score = scores[best_doc_type]

    if best_score < UNKNOWN_CONFIDENCE_THRESHOLD:
        return ClassificationResult("unknown", best_score, None, features, scores)

    return ClassificationResult(best_doc_type, best_score, best_doc_type, features, scores)
