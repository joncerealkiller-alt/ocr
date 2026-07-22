"""
Row segmentation for census/table bucket preprocessing.

Built 2026-07-12 after olmOCR fabricated an entire household's worth of
census rows (silently replacing real Knott family entries with invented
ones, repeated twice) - see project history. The diagnosed mechanism
across every degeneration/fabrication failure this session was long,
dense, repetitive-structure content processed in one pass. This module
is the detection layer for isolating single rows before OCR, the same
"bound the risky part's scope" principle that fixed Qwen's list-field
runaway via a two-call split, applied here at the image level instead.

Build order (agreed 2026-07-12, Jon + Rook): deskew -> projection-profile
candidate bands -> sanity checks -> crop rows -> attach header/context.
OCR integration and reassembly deliberately NOT included here - this
module only produces reviewable row crops, proven correct on real census
pages before any model-call complexity gets added on top.

Dependency-free: PIL + numpy only (both already used in this project's
core/image_preprocessing.py) - no OpenCV. Binarization uses a manually
implemented Otsu threshold rather than a cv2.threshold() call.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw


def _otsu_threshold(gray_arr: np.ndarray) -> float:
    """
    Standard Otsu's method: finds the grayscale threshold that maximizes
    between-class variance (background vs. foreground/text). Implemented
    manually rather than via cv2.threshold(..., THRESH_OTSU) to avoid
    adding OpenCV as a dependency for one function.
    """
    hist, _ = np.histogram(gray_arr, bins=256, range=(0, 256))
    hist = hist.astype(float)
    total = gray_arr.size
    sum_total = np.dot(np.arange(256), hist)

    sum_bg, weight_bg = 0.0, 0.0
    max_var, threshold = 0.0, 0
    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t
    return float(threshold)


def estimate_deskew_angle(
    image: Image.Image,
    angle_range: float = 5.0,
    angle_step: float = 0.25,
    search_scale: float = 0.3,
) -> float:
    """
    Suggests a deskew angle via projection-profile variance maximization -
    a STARTING SUGGESTION, not authoritative (2026-07-13, per Jon's
    direction: "Auto-deskew can still exist as a starting suggestion, but
    it should not be authoritative"). The correct angle produces the
    sharpest horizontal projection profile (aligned text rows create
    high-contrast peaks/valleys; misalignment smears it flat).

    Deliberately separated from apply_deskew_angle() below - this
    function does the (relatively cheap, since it already runs on a
    downscaled copy) SEARCH only, without applying any rotation to the
    full-resolution image. Callers building an interactive UI should call
    this ONCE for the initial suggestion, let the user confirm/override
    via nudge buttons or direct entry, and use apply_deskew_angle() for
    every subsequent preview redraw - re-running this search on every
    nudge would be wasted work searching for an angle the user has
    already decided to override.
    """
    gray = image.convert("L")
    small = gray.resize(
        (max(1, int(gray.width * search_scale)), max(1, int(gray.height * search_scale)))
    )
    arr = np.array(small)
    thresh = _otsu_threshold(arr)
    # <= not < : see apply_deskew_angle's sibling note in deskew()'s
    # original docstring history - same Otsu boundary-inclusion fix.
    binary = (arr <= thresh).astype(np.uint8) * 255

    best_angle = 0.0
    best_score = -1.0
    angle = -angle_range
    binary_img = Image.fromarray(binary)
    while angle <= angle_range:
        rotated = binary_img.rotate(angle, expand=False, fillcolor=0, resample=Image.BILINEAR)
        row_sums = np.array(rotated).sum(axis=1)
        score = float(np.var(row_sums))
        if score > best_score:
            best_score = score
            best_angle = angle
        angle += angle_step
    return best_angle


def apply_deskew_angle(image: Image.Image, angle: float) -> Image.Image:
    """
    Applies a rotation correction at a GIVEN angle - no search, just the
    rotation itself. Cheap enough to call on every nudge-button click or
    manual angle-field edit for instant preview redraws, unlike re-running
    estimate_deskew_angle()'s search each time. angle=0.0 is a no-op
    (returns the image as-is, not a needless rotate() call).
    """
    if angle == 0.0:
        return image
    fill = (255, 255, 255) if image.mode == "RGB" else 255
    return image.rotate(angle, expand=True, fillcolor=fill, resample=Image.BICUBIC)


def deskew(
    image: Image.Image,
    angle_range: float = 5.0,
    angle_step: float = 0.25,
    search_scale: float = 0.3,
) -> tuple[Image.Image, float]:
    """
    Convenience wrapper combining estimate_deskew_angle() +
    apply_deskew_angle() - kept for existing callers (segment_rows,
    segment_rows_periodic) that don't need the two steps separated.
    New interactive UI code should call the two functions independently
    instead, per estimate_deskew_angle()'s docstring.
    """
    angle = estimate_deskew_angle(image, angle_range, angle_step, search_scale)
    corrected = apply_deskew_angle(image, angle)
    return corrected, angle


@dataclass
class RowDetectionResult:
    bands: list[tuple[int, int]]          # (y_start, y_end) per detected row
    header_band: tuple[int, int] | None   # merged header region, if requested
    deskew_angle: float = 0.0             # degrees applied - critical correctness
                                           # value for downstream cropping, kept as
                                           # a real field, not just text inside
                                           # warnings, so consumers don't need to
                                           # parse a human-readable message to get it
    deskewed_image_size: tuple[int, int] = (0, 0)  # (width, height) of the image
                                           # bboxes are actually relative to. NOT
                                           # the pre-deskew source size - rotation
                                           # with expand=True changes canvas
                                           # dimensions, and a caller building a
                                           # sidecar needs THIS size, not the
                                           # original's, or bbox x1/width will be
                                           # wrong (confirmed as a real bug via a
                                           # pixel-level round-trip test,
                                           # 2026-07-13 - not a hypothetical).
    warnings: list[str] = field(default_factory=list)
    dropped_bands: list[tuple[int, int]] = field(default_factory=list)


def _close_1d(row_bool: np.ndarray, radius: int) -> np.ndarray:
    """
    1D morphological closing (dilate then erode) along a single row -
    bridges small gaps up to ~2*radius px before they break up an
    otherwise-continuous run. Confirmed necessary (2026-07-12): a real
    scanned ruling line with just 40 small random breaks (2-6px each,
    simulating scan noise/faded ink/JPEG artifacts) dropped the longest-
    run ratio from 1.0 to 0.12 on a 2000px-wide test - even light,
    realistic imperfection completely defeated a naive longest-run
    measurement. Closing bridges those small breaks while leaving the
    much larger natural gaps between handwritten words/letters intact.
    """
    dilated = row_bool.copy()
    for shift in range(1, radius + 1):
        dilated[shift:] |= row_bool[:-shift]
        dilated[:-shift] |= row_bool[shift:]

    eroded = dilated.copy()
    for shift in range(1, radius + 1):
        shifted_right = np.ones_like(dilated)
        shifted_right[shift:] = dilated[:-shift]
        shifted_left = np.ones_like(dilated)
        shifted_left[:-shift] = dilated[shift:]
        eroded &= shifted_right & shifted_left
    return eroded


def _longest_run_per_row(binary: np.ndarray, close_gap_px: int = 4) -> np.ndarray:
    """
    For each row, computes the length of the longest contiguous run of
    dark (foreground) pixels, AFTER closing small gaps up to
    close_gap_px (see _close_1d) - without this, realistic scan noise
    breaking up an otherwise-continuous printed line defeats the
    measurement entirely (confirmed directly, not assumed - see
    _close_1d's docstring). Used to distinguish printed ruling lines
    (long unbroken-after-closing runs, often >60-80% of page width)
    from handwritten text content (many genuinely short runs, broken by
    natural gaps between letters/words/columns that closing does NOT
    bridge, since those gaps are much larger than scan-noise breaks).

    close_gap_px=4 chosen empirically (2026-07-12), not guessed: radius=6
    bridged enough of a realistic simulated text row's internal gaps
    that it became indistinguishable from a real ruling line (both hit
    ratio 1.000) - a real regression, caught by testing. radius=4 keeps
    clear separation: still fully bridges 2-6px noise breaks confirmed
    in a broken-ruling-line reproduction (ratio 1.000), while a
    realistic text row stays at ratio ~0.19, well clear of the 0.6
    classification threshold.
    """
    n_rows, n_cols = binary.shape
    longest = np.zeros(n_rows, dtype=np.int64)
    for y in range(n_rows):
        row = binary[y].astype(bool)
        if not row.any():
            continue
        closed = _close_1d(row, close_gap_px)
        padded = np.concatenate(([0], closed.astype(np.int8), [0]))
        diffs = np.diff(padded)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        if len(starts) == 0:
            continue
        longest[y] = (ends - starts).max()
    return longest


def detect_row_bands(
    image: Image.Image,
    smoothing_window: int = 5,
    density_threshold_ratio: float = 0.05,
    ruling_line_run_ratio: float = 0.6,
    x0: int | None = None,
    x1: int | None = None,
) -> tuple[list[tuple[int, int]], int, np.ndarray]:
    """
    Projection-profile row detection: sums dark-pixel density per
    horizontal row, smooths to reduce noise, and finds contiguous bands
    where density exceeds a fraction of the page's peak density. Returns
    raw candidate bands - NOT yet sanity-checked or merged, see
    sanity_check_bands() and merge_wrapped_bands() below.

    x0/x1 (optional): restrict detection to this column range - added
    2026-07-13 per Jon's direction that left/right bounds should be
    user-confirmed, not implicitly "always full image width." This
    actually restricts what pixels DENSITY AND RUN-LENGTH ARE COMPUTED
    FROM, not just what gets cropped out afterward - margin content
    (binding shadows, torn edges, adjacent page bleed) can otherwise
    pollute detection even though it was never going to end up in any
    row crop anyway. Defaults to None (full width) for backward
    compatibility with existing callers.

    Rows whose longest continuous dark run exceeds ruling_line_run_ratio
    of the (x0:x1) range's width are treated as separators (like a blank
    gap), regardless of their raw density - this is what lets a tightly
    ruled table (printed line between every row, near-zero true
    whitespace) still get split into individual rows. See
    _longest_run_per_row's docstring for why density alone can't do this.

    Returns (bands, ruling_line_count, ruling_line_rows) - the count is
    surfaced so callers/reports can show it as a diagnostic (e.g.
    "detected 49 ruling lines" is a strong confirmation signal that this
    mechanism actually engaged, vs. silently doing nothing on an unruled
    page).
    """
    gray = image.convert("L")
    arr = np.array(gray)
    if x0 is not None or x1 is not None:
        arr = arr[:, (x0 or 0):(x1 if x1 is not None else arr.shape[1])]
    thresh = _otsu_threshold(arr)
    binary = (arr <= thresh).astype(np.uint8)

    row_density = binary.sum(axis=1).astype(float)
    width = binary.shape[1]
    longest_run = _longest_run_per_row(binary)
    is_ruling_line = (longest_run / width) > ruling_line_run_ratio
    ruling_line_count = int(is_ruling_line.sum())

    # Dilation radius matches the smoothing kernel's reach, not an
    # arbitrary fixed 1px - confirmed necessary (2026-07-12) after a
    # ruling-line-adjacent gap row was still misclassified as "text"
    # because the density smoothing below bleeds a ruling line's very
    # high density up to smoothing_window//2 rows into its neighbors.
    # If the exclusion zone is narrower than the smoothing's actual
    # bleed radius, gap rows next to a ruling line can stay above
    # threshold from bleed alone, and adjacent text rows never
    # separate. +1 for a small safety margin beyond the exact half-width.
    dilation_radius = smoothing_window // 2 + 1
    is_ruling_line_dilated = is_ruling_line.copy()
    for shift in range(1, dilation_radius + 1):
        is_ruling_line_dilated[shift:] |= is_ruling_line[:-shift]
        is_ruling_line_dilated[:-shift] |= is_ruling_line[shift:]

    kernel = np.ones(smoothing_window) / smoothing_window
    smoothed = np.convolve(row_density, kernel, mode="same")
    density_threshold = smoothed.max() * density_threshold_ratio
    is_text_row = (smoothed > density_threshold) & (~is_ruling_line_dilated)

    bands: list[tuple[int, int]] = []
    start = None
    for y, val in enumerate(is_text_row):
        if val and start is None:
            start = y
        elif not val and start is not None:
            bands.append((start, y))
            start = None
    if start is not None:
        bands.append((start, len(is_text_row)))
    ruling_line_rows = np.where(is_ruling_line)[0]
    return bands, ruling_line_count, ruling_line_rows


def merge_wrapped_bands(
    bands: list[tuple[int, int]],
    max_gap_ratio: float = 0.4,
    ruling_line_rows=None,
) -> list[tuple[int, int]]:
    """
    Census rows can wrap onto a second line. Merges adjacent bands when
    the gap between them is small relative to the median band height -
    a small gap is more likely a wrapped entry than a genuine row
    boundary. Threshold is relative (per-document median), not an
    absolute pixel count, since text size varies across scans.

    ruling_line_rows (optional): row indices classified as printed
    ruling lines by detect_row_bands. If given, a gap containing ANY
    ruling-line row is NEVER merged, regardless of how small it is -
    confirmed necessary (2026-07-12) on a tightly ruled census table,
    where the natural gap between every legitimate row (created by the
    ruling line + exclusion dilation) was just as small as a genuine
    wrapped-entry gap, and gap-size alone couldn't tell them apart. A
    detected ruling line in the gap is much stronger evidence of a real
    boundary than gap size is evidence of a wrap.
    """
    if not bands:
        return bands
    heights = sorted(b[1] - b[0] for b in bands)
    median_height = heights[len(heights) // 2]
    ruling_set = set(int(y) for y in ruling_line_rows) if ruling_line_rows is not None else set()

    merged = [bands[0]]
    for b in bands[1:]:
        prev_start, prev_end = merged[-1]
        gap = b[0] - prev_end
        ruling_line_in_gap = any(y in ruling_set for y in range(prev_end, b[0]))
        if ruling_line_in_gap:
            merged.append(b)
            continue
        if median_height > 0 and gap < median_height * max_gap_ratio:
            merged[-1] = (prev_start, b[1])
        else:
            merged.append(b)
    return merged


def sanity_check_bands(
    bands: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[str]]:
    """
    Filters/flags detected bands per the agreed checklist: implausibly
    thin/tall bands, heavy overlap, large unexplained gaps. Thin bands
    are DROPPED (very likely noise, not a real row); tall bands are
    FLAGGED but kept (could be a genuine multi-line entry that
    merge_wrapped_bands should have caught, or could be a real wide row -
    ambiguous enough to warrant human review rather than auto-dropping).

    Returns (kept_bands, dropped_bands, warnings) - dropped_bands is
    returned (not discarded) so the debug overlay can show what was
    removed and why, rather than silently vanishing.
    """
    warnings: list[str] = []
    if not bands:
        warnings.append("No row bands detected at all - check deskew "
                         "angle and density_threshold_ratio.")
        return [], [], warnings

    heights = sorted(b[1] - b[0] for b in bands)
    median_h = heights[len(heights) // 2]

    kept, dropped = [], []
    for b in bands:
        h = b[1] - b[0]
        if median_h > 0 and h < median_h * 0.3:
            warnings.append(
                f"Band {b} implausibly thin ({h}px vs median {median_h}px) "
                f"- dropped, likely noise not a real row."
            )
            dropped.append(b)
            continue
        if median_h > 0 and h > median_h * 3:
            warnings.append(
                f"Band {b} implausibly tall ({h}px vs median {median_h}px) "
                f"- kept but flagged, may be an unmerged wrapped entry or "
                f"a genuine wide row. Worth visual review."
            )
        kept.append(b)

    for i in range(len(kept) - 1):
        if kept[i][1] > kept[i + 1][0]:
            warnings.append(f"Bands {kept[i]} and {kept[i + 1]} overlap.")

    if len(kept) > 1:
        gaps = [kept[i + 1][0] - kept[i][1] for i in range(len(kept) - 1)]
        med_gap = sorted(gaps)[len(gaps) // 2]
        for i, g in enumerate(gaps):
            if med_gap > 0 and g > med_gap * 4:
                warnings.append(
                    f"Large unexplained gap ({g}px, median is {med_gap}px) "
                    f"between band {i} and {i + 1} - possible missed row "
                    f"or genuine blank space, worth checking."
                )

    return kept, dropped, warnings


def _compute_padding(
    row_height: int,
    padding: int = 4,
    padding_pct: float | None = None,
    padding_top: int | None = None,
    padding_bottom: int | None = None,
) -> tuple[int, int]:
    """
    Shared padding logic for crop_rows() and build_sidecar() - keeps
    both in sync rather than letting them drift (the sidecar's stored
    bbox must match what actually gets cropped downstream, or
    coordinates silently disagree with reality - see crop_region_from_
    source's docstring for why that class of mismatch matters).

    Added 2026-07-13 per Jon/GPT's suggestion: "oversample the box by a
    configurable pixel or percentage" - the underlying idea (crop
    slightly beyond the detected line so descenders/ascenders/clerk
    overwrites aren't clipped) was ALREADY the default behavior
    (padding=4 baked into every stored bbox since this module's
    sidecar work), just hardcoded rather than exposed as a real
    tunable. This makes it configurable three ways:

    - padding: fixed pixels, same top/bottom (original behavior,
      still the default when nothing else is specified)
    - padding_pct: percentage of THIS row's height, recomputed per row
      rather than a single fixed value - takes precedence over
      `padding` when given, since a percentage is more likely to be an
      intentional choice than the pixel default
    - padding_top/padding_bottom: asymmetric override - if either is
      given, it replaces whatever padding/padding_pct would have
      produced for THAT side specifically (independently, so you can
      set just one side)

    Returns (top_padding, bottom_padding) as actual pixel values.

    Proactive warning (per GPT's own caution, worth enforcing not just
    documenting): padding beyond ~20% of row height risks reading into
    the adjacent row and merging two people's information - printed as
    a warning, not silently allowed through without comment.
    """
    base = round(row_height * padding_pct) if padding_pct is not None else padding
    top = padding_top if padding_top is not None else base
    bottom = padding_bottom if padding_bottom is not None else base

    for label, value in [("top", top), ("bottom", bottom)]:
        if row_height > 0 and value > row_height * 0.20:
            print(f"WARNING: {label} padding ({value}px) exceeds 20% of row "
                  f"height ({row_height}px) - risk of reading into the "
                  f"adjacent row and merging two people's information. "
                  f"Consider a smaller value.")

    return top, bottom


def crop_rows(
    image: Image.Image,
    bands: list[tuple[int, int]],
    header_band: tuple[int, int] | None = None,
    padding: int = 4,
    padding_pct: float | None = None,
    padding_top: int | None = None,
    padding_bottom: int | None = None,
    x0: int | None = None,
    x1: int | None = None,
) -> tuple[list[Image.Image], Image.Image | None]:
    """
    Crops each detected row band into its own image. If header_band is
    given, prepends that header strip to EVERY row crop (redundant, but
    per the agreed first-prototype approach: avoids header-mapping
    errors while testing whether row isolation itself improves
    extraction - optimize away the redundancy later once the core
    approach is proven).

    x0/x1 (optional): crop to this column range instead of always full
    image width - added 2026-07-13 so left/right bounds are actually
    respected in the output, not just in detection.

    padding/padding_pct/padding_top/padding_bottom: see
    _compute_padding()'s docstring for the full explanation - fixed
    pixels (default, backward compatible), percentage of row height,
    or asymmetric top/bottom overrides.
    """
    x0 = x0 if x0 is not None else 0
    x1 = x1 if x1 is not None else image.width

    header_crop = None
    if header_band is not None:
        header_height = header_band[1] - header_band[0]
        h_top, h_bottom = _compute_padding(
            header_height, padding, padding_pct, padding_top, padding_bottom)
        y0 = max(0, header_band[0] - h_top)
        y1 = min(image.height, header_band[1] + h_bottom)
        header_crop = image.crop((x0, y0, x1, y1))

    crops = []
    for (y0, y1) in bands:
        row_height = y1 - y0
        top_pad, bottom_pad = _compute_padding(
            row_height, padding, padding_pct, padding_top, padding_bottom)
        # Both bounds clamped against BOTH 0 and image.height - confirmed
        # bug (2026-07-13): previously y0p was only clamped against 0
        # (never checked against image.height), while y1p was only
        # clamped against image.height (never checked against 0 or
        # against y0p). A row computed beyond the image's actual bottom
        # edge (e.g. uniform_tile mode with a row_height/row_count
        # combination that overruns the confirmed table span) produced
        # y0p > y1p - an inverted box - crashing image.crop() with
        # PIL's "Coordinate 'lower' is less than 'upper'" deep inside a
        # cryptic call stack instead of surfacing as a clear diagnostic.
        y0p = max(0, min(image.height, y0 - top_pad))
        y1p = max(0, min(image.height, y1 + bottom_pad))
        y1p = max(y0p, y1p)  # final safety net - never pass an inverted box
        row_crop = image.crop((x0, y0p, x1, y1p))
        if header_crop is not None:
            combined = Image.new(
                "RGB", (x1 - x0, header_crop.height + row_crop.height), "white"
            )
            combined.paste(header_crop.convert("RGB"), (0, 0))
            combined.paste(row_crop.convert("RGB"), (0, header_crop.height))
            crops.append(combined)
        else:
            crops.append(row_crop)
    return crops, header_crop


def render_debug_overlay(
    image: Image.Image,
    bands: list[tuple[int, int]],
    dropped_bands: list[tuple[int, int]] | None = None,
    header_band: tuple[int, int] | None = None,
    x0: int | None = None,
    x1: int | None = None,
) -> Image.Image:
    """
    Draws detected bands directly on a copy of the image for visual
    inspection - the whole point of doing Option B (projection profile)
    first is that failures are understandable, and that only holds if
    you can actually SEE what was detected. Kept bands in alternating
    red/green (makes adjacent-row boundaries easy to distinguish),
    dropped bands in gray with a strike-through style X, header in blue.

    x0/x1 (optional): if given, row/header rectangles are drawn at these
    column bounds (not always full width), and a purple outline marks
    the overall table x-range across its full height - added 2026-07-13
    so left/right bounds are visually confirmable the same way top/
    bottom already were, not just trusted as numbers in a field.
    """
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    x0 = x0 if x0 is not None else 0
    x1 = x1 if x1 is not None else image.width

    if x0 != 0 or x1 != image.width:
        draw.rectangle([x0, 0, x1, image.height], outline="purple", width=2)

    if header_band is not None:
        draw.rectangle([x0, header_band[0], x1, header_band[1]],
                        outline="blue", width=3)
        draw.text((x0 + 5, header_band[0] + 2), "HEADER", fill="blue")

    for i, (y0, y1) in enumerate(bands):
        color = "red" if i % 2 == 0 else "green"
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        draw.text((x0 + 5, y0 + 2), str(i + 1), fill=color)

    if dropped_bands:
        for (y0, y1) in dropped_bands:
            draw.rectangle([x0, y0, x1, y1], outline="gray", width=2)
            draw.line([x0, y0, x1, y1], fill="gray", width=1)
            draw.line([x0, y1, x1, y0], fill="gray", width=1)

    return overlay


def _row_density_profile(
    image: Image.Image, x0: int | None = None, x1: int | None = None,
) -> tuple[np.ndarray, float]:
    """Shared helper: binarized per-row dark-pixel density and the Otsu
    threshold used to compute it, reused by both the general detector
    and the periodic-anchor refinement below. x0/x1 restrict the columns
    density is computed from, same rationale as detect_row_bands()."""
    gray = image.convert("L")
    arr = np.array(gray)
    if x0 is not None or x1 is not None:
        arr = arr[:, (x0 or 0):(x1 if x1 is not None else arr.shape[1])]
    thresh = _otsu_threshold(arr)
    binary = (arr <= thresh).astype(np.uint8)
    return binary.sum(axis=1).astype(float), thresh


def estimate_table_extent(
    image: Image.Image, edge_density_ratio: float = 0.02,
    x0: int | None = None, x1: int | None = None,
) -> tuple[int, int]:
    """
    Rough estimate of the table body's vertical extent (top of row 1 to
    bottom of the last row), as the topmost and bottommost rows whose
    density exceeds a small fraction of peak density - i.e. "where does
    real content start/stop, excluding blank margin." This is a coarse
    heuristic, not a real header/footer detector - callers should treat
    it as a starting point and override with known coordinates
    (table_top/table_bottom) when available, since getting this exactly
    right matters less here than in the general detector: periodic
    refinement only needs a roughly-correct extent to generate
    reasonable starting boundary guesses, which then get locally
    corrected against the real image anyway.

    x0/x1: same column-range restriction as detect_row_bands() /
    _row_density_profile() - if the table only occupies part of the
    image width, restricting the extent estimate to those columns avoids
    unrelated margin content (page number stamps, binding shadows)
    skewing where "real content" appears to start/stop.
    """
    density, _ = _row_density_profile(image, x0=x0, x1=x1)
    threshold = density.max() * edge_density_ratio
    nonzero = np.where(density > threshold)[0]
    if len(nonzero) == 0:
        return 0, image.height
    return int(nonzero[0]), int(nonzero[-1])


def refine_boundary_position(
    density: np.ndarray,
    ruling_line_rows,
    expected_y: int,
    search_radius: int,
) -> int:
    """
    Locally searches a small window around an EXPECTED boundary position
    (from periodic spacing, not blind whole-page search) for the best
    actual cut point: prefers a detected ruling line within the window
    if one exists (strong structural signal), otherwise falls back to
    the local density minimum (best available gap). This is the core of
    "refine 50 expected boundaries" rather than "find 50 unknown rows" -
    confined to a narrow window around a periodic estimate, so it's far
    less sensitive to whatever caused global ruling-line detection to
    fail to engage at all on a real scan (2026-07-12) than the general
    detector was.
    """
    lo = max(0, expected_y - search_radius)
    hi = min(len(density), expected_y + search_radius + 1)

    ruling_set = set(int(y) for y in ruling_line_rows) if ruling_line_rows is not None else set()
    ruling_in_window = [y for y in range(lo, hi) if y in ruling_set]
    if ruling_in_window:
        # If multiple, take the one closest to the expected position.
        return min(ruling_in_window, key=lambda y: abs(y - expected_y))

    window = density[lo:hi]
    if len(window) == 0:
        return expected_y
    local_min_offset = int(np.argmin(window))
    return lo + local_min_offset


def segment_rows_uniform_tile(
    image: Image.Image,
    row_count: int,
    first_row_top: float,
    first_row_bottom: float,
    table_left: int | None = None,
    table_right: int | None = None,
    header_row_count: int = 0,
    deskew_angle: float | None = None,
    padding: int = 4,
    padding_pct: float | None = None,
    padding_top: int | None = None,
    padding_bottom: int | None = None,
    metadata_bottom: int | None = None,
    header_box_top: int | None = None,
    header_box_bottom: int | None = None,
) -> tuple[RowDetectionResult, list[Image.Image], Image.Image, Image.Image]:
    """
    Simplest possible segmentation strategy: no Otsu, no ruling-line
    detection, no local search/refinement at all. The user confirms ONE
    row's exact bbox visually (via row_segmentation_ui.py's cheap
    preview - refresh, adjust, repeat, same workflow already used for
    deskew/bounds), then that row's height is tiled uniformly down the
    page row_count times.

    Built 2026-07-13 after periodic anchoring showed compounding drift
    on a real page (1931 census, e011707164) with an irregular line
    (line 18 - a marginal sub-district boundary annotation, not a
    person row) partway down: periodic mode computes every boundary
    from ONE global average row height, so a single irregular gap
    anywhere on the page throws off every boundary below it, with no
    local correction. It also turned out only ~16 of 50 boundaries on
    that page had a real ruling line to anchor to - the rest relied on
    the noisier density-minimum fallback, which risks snapping to
    handwriting density peaks that aren't real row edges.

    This sidesteps both problems by removing the mechanism that could
    be wrong: no detection to mis-anchor, no average that an irregular
    line can throw off. The bet is that real row spacing on a page like
    this is otherwise consistent enough that pure uniform tiling from
    one well-confirmed reference row is MORE robust than "smart" but
    occasionally-wrong local search - worth testing directly rather
    than assumed, same as everything else this session.

    Real limitation, not hidden: if the page's true row spacing genuinely
    drifts gradually (not just one irregular line, but truly uneven
    handwriting/ruling throughout), uniform tiling has no mechanism to
    correct for that either - it would need re-confirming in sections.
    This is the right tool for "mostly uniform with an occasional
    irregular line," not for "genuinely non-uniform throughout."

    first_row_top/first_row_bottom accept FLOAT precision - critical at
    row_count=50: a naive integer-only row height forces rounding (e.g.
    1730px / 50 rows = 34.6px, forced to 34 or 35), and that rounding
    error is NOT a one-time cost - it compounds every time the height
    gets added again for the next row, reaching a FULL row-height of
    accumulated drift by the last row if computed by repeatedly adding
    an already-rounded integer step. Fixed by keeping row_height as a
    float throughout and rounding each band's boundary INDEPENDENTLY
    from the original float arithmetic (first_row_top + i*row_height,
    rounded once per row directly from the float formula) rather than
    accumulating a rounded step 49 times - see the list comprehension
    below.
    """
    if deskew_angle is not None:
        deskewed = apply_deskew_angle(image, deskew_angle)
        angle = deskew_angle
    else:
        deskewed, angle = deskew(image)

    row_height = first_row_bottom - first_row_top  # stays float
    all_bands = [
        (round(first_row_top + i * row_height),
         round(first_row_top + (i + 1) * row_height))
        for i in range(row_count)
    ]

    # FIXED 2026-07-13 (same real bug as segment_rows_periodic - see
    # that function's comment for full explanation): header_row_count
    # used to sacrifice real person-row bands as fake "header." Fixed
    # to use the real, already-confirmed region above first_row_top
    # directly (first_row_top is uniform_tile's equivalent of
    # periodic's table_top - the confirmed boundary where real person
    # data starts).
    #
    # header_box_top/header_box_bottom (2026-07-13): explicit, user-
    # confirmed header region takes priority over the automatic
    # metadata_bottom-to-first_row_top span - see segment_rows_
    # periodic's matching comment for the full explanation (a real,
    # multi-tier bilingual header block makes the whole automatic gap
    # too imprecise to blindly prepend to every row).
    header_band = None
    data_bands = all_bands
    if header_box_top is not None and header_box_bottom is not None:
        header_band = (header_box_top, header_box_bottom)
    elif header_row_count > 0:
        header_band = (metadata_bottom if metadata_bottom is not None else 0,
                        round(first_row_top))

    x0, x1 = table_left, table_right
    warnings = [
        f"Deskew angle applied: {angle:.2f} degrees."
        + (" (user-confirmed, not auto-estimated)" if deskew_angle is not None else " (auto-estimated)"),
        f"Uniform-tile mode: first row ({first_row_top}, {first_row_bottom}), "
        f"row_height={row_height:.2f}px, tiled {row_count} times "
        f"(no detection, no ruling-line search - pure arithmetic).",
    ]

    # Proactive check (2026-07-13, real bug this caught): the tiled span
    # can overrun the actual image height if row_height x row_count
    # doesn't match the page's real content - this used to only surface
    # as a cryptic PIL crop crash deep in crop_rows(), now caught here
    # with a clear, actionable diagnostic instead. Rows that end up
    # out of bounds still get produced (crop_rows() now clamps safely -
    # see its docstring) but will be visibly empty/truncated, not a
    # silent correctness problem.
    tiled_span = row_count * row_height
    last_row_bottom = all_bands[-1][1]
    if last_row_bottom > deskewed.height:
        overrun = last_row_bottom - deskewed.height
        warnings.append(
            f"WARNING: tiled span ({row_count} x {row_height}px = "
            f"{tiled_span}px) runs {overrun}px past the image's actual "
            f"bottom edge ({deskewed.height}px). The last several rows "
            f"will be empty/truncated crops, not real data. Likely "
            f"cause: Row 1's confirmed height doesn't match this page's "
            f"real average row height, or row_count is wrong for this "
            f"page - re-check Row 1 top/bottom against the actual page, "
            f"or reduce row_count if this page genuinely has fewer rows."
        )
        print(warnings[-1])

    row_crops, header_crop = crop_rows(
        deskewed, data_bands, header_band=header_band, x0=x0, x1=x1,
        padding=padding, padding_pct=padding_pct,
        padding_top=padding_top, padding_bottom=padding_bottom,
    )
    debug_overlay = render_debug_overlay(
        deskewed, data_bands, header_band=header_band, x0=x0, x1=x1
    )

    result = RowDetectionResult(
        bands=data_bands, header_band=header_band, deskew_angle=angle,
        deskewed_image_size=deskewed.size, warnings=warnings,
    )
    return result, row_crops, header_crop, debug_overlay


def segment_rows_periodic(
    image: Image.Image,
    row_count: int,
    table_top: int | None = None,
    table_bottom: int | None = None,
    table_left: int | None = None,
    table_right: int | None = None,
    search_radius_ratio: float = 0.3,
    header_row_count: int = 1,
    deskew_angle: float | None = None,
    padding: int = 4,
    padding_pct: float | None = None,
    padding_top: int | None = None,
    padding_bottom: int | None = None,
    metadata_bottom: int | None = None,
    header_box_top: int | None = None,
    header_box_bottom: int | None = None,
) -> tuple[RowDetectionResult, list[Image.Image], Image.Image, Image.Image]:
    """
    Alternate segmentation strategy for document types with a KNOWN,
    fixed row count and near-uniform spacing (confirmed use case:
    standard Canadian census forms, which print a fixed number of
    numbered lines per page - row_count is directly readable off the
    form itself, e.g. 50). NOT a general-purpose replacement for
    segment_rows() - this is deliberately opt-in for document types
    where the row count is actually known in advance, not guessed.

    Built 2026-07-12 after global ruling-line detection failed to
    engage at all ("Detected 0 ruling-line row(s)") on a real census
    scan across two separate fix attempts, despite both fixes being
    individually confirmed correct against synthetic reproductions of
    the specific failure modes found - suggesting something about the
    real scan's structure that synthetic testing alone couldn't surface.
    Rather than continue tuning a blind whole-page search, this exploits
    the document type's known structure directly: estimate near-uniform
    row spacing from the table's overall extent, generate row_count+1
    expected boundary positions, then locally refine each one (see
    refine_boundary_position) rather than searching the whole page for
    unknown structure.

    deskew_angle (2026-07-13, per Jon's direction that auto-deskew should
    be a starting SUGGESTION, never authoritative): if given, this exact
    angle is applied via apply_deskew_angle() and estimate_deskew_angle()
    is never called - the caller (an interactive UI) is expected to have
    already shown the user an auto-suggested angle, let them confirm or
    override it via nudge buttons/direct entry, and pass the CONFIRMED
    value here. If None, falls back to full auto-estimate + apply (the
    original 2026-07-12 behavior) for backward-compatible non-interactive
    use (e.g. batch CLI runs where no human is confirming each page).

    table_top/table_bottom/table_left/table_right: pixel coordinates of
    the table body's extent. Top/bottom estimated automatically if not
    given (see estimate_table_extent - a coarse heuristic, meant to be
    visually confirmed/overridden, not trusted blind). left/right
    (2026-07-13) default to the full image width if not given - unlike
    top/bottom there's no automatic estimate offered for these yet, since
    unlike vertical extent (which has an obvious "where does real content
    start" heuristic) horizontal table bounds are harder to guess
    generically; supply these explicitly once confirmed visually.

    search_radius_ratio: each local refinement window is
    +/- (expected_row_height * search_radius_ratio) around its
    periodic estimate. 0.3 means a boundary can be found anywhere
    within 30% of a row-height of where uniform spacing predicts it -
    wide enough to absorb real form irregularity, narrow enough to
    stay anchored to the correct row rather than drifting into a
    neighbor.
    """
    if deskew_angle is not None:
        deskewed = apply_deskew_angle(image, deskew_angle)
        angle = deskew_angle
    else:
        deskewed, angle = deskew(image)

    x0, x1 = table_left, table_right
    density, _ = _row_density_profile(deskewed, x0=x0, x1=x1)

    if table_top is None or table_bottom is None:
        auto_top, auto_bottom = estimate_table_extent(deskewed, x0=x0, x1=x1)
        table_top = table_top if table_top is not None else auto_top
        table_bottom = table_bottom if table_bottom is not None else auto_bottom

    total_span = table_bottom - table_top
    expected_row_height = total_span / row_count if row_count > 0 else total_span
    search_radius = max(2, int(expected_row_height * search_radius_ratio))

    _, _, ruling_line_rows = detect_row_bands(deskewed, x0=x0, x1=x1)

    expected_boundaries = [
        table_top + round(i * expected_row_height) for i in range(row_count + 1)
    ]
    refined_boundaries = [
        refine_boundary_position(density, ruling_line_rows, y, search_radius)
        for y in expected_boundaries
    ]
    # Guarantee monotonic ordering - local refinement could in principle
    # push two adjacent boundaries out of order on a badly degraded
    # scan; enforce a minimum 1px separation rather than let that
    # silently produce a zero/negative-height band downstream.
    for i in range(1, len(refined_boundaries)):
        if refined_boundaries[i] <= refined_boundaries[i - 1]:
            refined_boundaries[i] = refined_boundaries[i - 1] + 1

    all_bands = [
        (refined_boundaries[i], refined_boundaries[i + 1])
        for i in range(len(refined_boundaries) - 1)
    ]

    # FIXED 2026-07-13 (real bug, found by Jon, not caught by any test
    # this session because header_row_count=0 was used everywhere,
    # avoiding it entirely): this used to take header_row_count bands
    # from the START of all_bands (the periodically-tiled DATA bands
    # beginning at table_top) as if that were the header - but once
    # table_top is correctly calibrated to the real first person row
    # (which every real test tonight confirmed it should be), that
    # meant header_row_count > 0 always sacrificed a genuine person row
    # as fake "header," never actually captured the real printed column
    # labels at all (those live ABOVE table_top, in the metadata_bottom-
    # to-table_top gap). Fixed to use that REAL, already-confirmed
    # region directly - no data bands consumed, any header_row_count > 0
    # now means "yes, include the real header," not "sacrifice this many
    # rows."
    #
    # header_box_top/header_box_bottom (2026-07-13, second real gap
    # found by Jon the same day): the whole metadata_bottom-to-table_top
    # gap can be a large, multi-tier, sometimes bilingual header block
    # (confirmed on a real 1931 census form - category labels, sub-
    # labels, French translation, THEN the actual numbered-column row
    # closest to the data). Prepending the ENTIRE gap to every row crop
    # is wasteful and imprecise when only a small strip (e.g. just the
    # numbered-column row) is actually useful context. If BOTH are
    # given, they define an EXACT, user-confirmed header region
    # (same visual workflow as Row 1's yellow box) that takes priority
    # over the automatic metadata_bottom-to-table_top span entirely -
    # letting the user select precisely which slice of a complex header
    # block gets attached, not just "off" vs. "the whole gap."
    header_band = None
    data_bands = all_bands
    if header_box_top is not None and header_box_bottom is not None:
        header_band = (header_box_top, header_box_bottom)
    elif header_row_count > 0 and table_top is not None:
        header_band = (metadata_bottom if metadata_bottom is not None else 0, table_top)

    warnings = [
        f"Deskew angle applied: {angle:.2f} degrees."
        + (" (user-confirmed, not auto-estimated)" if deskew_angle is not None else " (auto-estimated)"),
        f"Periodic mode: table extent top/bottom ({table_top}, {table_bottom}), "
        f"left/right ({x0 if x0 is not None else 0}, "
        f"{x1 if x1 is not None else deskewed.width}), "
        f"row_count={row_count}, expected_row_height={expected_row_height:.1f}px, "
        f"search_radius={search_radius}px.",
        f"{len(ruling_line_rows)} ruling-line row(s) available as a refinement "
        f"signal within local search windows "
        f"({'used where found' if len(ruling_line_rows) > 0 else 'none found - every boundary fell back to local density minimum'}).",
    ]

    row_crops, header_crop = crop_rows(
        deskewed, data_bands, header_band=header_band, x0=x0, x1=x1,
        padding=padding, padding_pct=padding_pct,
        padding_top=padding_top, padding_bottom=padding_bottom,
    )
    debug_overlay = render_debug_overlay(
        deskewed, data_bands, header_band=header_band, x0=x0, x1=x1
    )

    result = RowDetectionResult(
        bands=data_bands, header_band=header_band, deskew_angle=angle,
        deskewed_image_size=deskewed.size, warnings=warnings,
    )
    return result, row_crops, header_crop, debug_overlay


def build_sidecar(
    result: RowDetectionResult,
    source_image_path: str,
    mode: str,
    parameters: dict,
    table_top: int | None = None,
    table_bottom: int | None = None,
    x0: int | None = None,
    x1: int | None = None,
    metadata_bottom: int | None = None,
    padding: int = 4,
    padding_pct: float | None = None,
    padding_top: int | None = None,
    padding_bottom: int | None = None,
    mask_keep_ranges: list[tuple[int, int]] | None = None,
    mask_apply_header: bool = False,
    mask_apply_rows: bool = True,
) -> dict:
    """
    Builds the JSON-serializable sidecar dict - the normal output of this
    module going forward (2026-07-13), replacing per-row PNG files as the
    default. Individual row crops are expensive to store at scale (one
    file per row, times every row, times every page) and freeze a
    specific padding/header-handling decision at segmentation time; a
    coordinates-only sidecar keeps the ORIGINAL source image as the
    single source of truth and lets a downstream OCR stage decide how to
    crop (padding, header handling, resolution) without needing to
    regenerate anything upstream.

    metadata_bottom (2026-07-13, per Jon's direction): splits the region
    ABOVE table_top into two distinct sub-regions instead of treating it
    as one block:
        0 -> metadata_bottom           = page metadata (province,
                                          district, enumerator, page #)
        metadata_bottom -> table_top   = column headings/instructions
                                          (dense, often bilingual text -
                                          confirmed as a real source of
                                          confusion when the metadata
                                          extractor was fed this whole
                                          combined block: it's visually
                                          and informationally distinct
                                          from the compact metadata
                                          fields above it, not more of
                                          the same content)
    Stored as its own top-level sidecar field (not folded into
    table_bbox, since it's a genuinely separate boundary with its own
    meaning). Optional - a sidecar without it still works exactly as
    before (extract_page_header/run_row_extraction fall back to using
    the full 0-to-table_top block as the metadata region).
    specific padding/header-handling decision at segmentation time; a
    coordinates-only sidecar keeps the ORIGINAL source image as the
    single source of truth and lets a downstream OCR stage decide how to
    crop (padding, header handling, resolution) without needing to
    regenerate anything upstream.

    Schema matches Jon's specification exactly (2026-07-13):
        {
          "deskew_angle": -1.10,
          "coordinate_space": "deskewed_image",
          "table_bbox": [x0, table_top, x1, table_bottom],
          "header_bbox": [...] or null,
          "rows": [{"index": N, "bbox": [...]}, ...]
        }
    "coordinate_space": "deskewed_image" makes the sidecar self-
    documenting - a future consumer doesn't need to read this module's
    source/docstrings to know bboxes are relative to the POST-deskew
    image, not the raw original (see crop_region_from_source()'s
    docstring for why that distinction is correctness-critical).

    x0/x1 (left/right table bounds): if given, bboxes use these instead
    of always spanning full image width - added 2026-07-13 per Jon's
    direction that left/right should be user-confirmed like top/bottom
    already were, not implicitly "always full width."

    padding/padding_pct/padding_top/padding_bottom (2026-07-13, per Jon/
    GPT's suggestion to make row-crop oversampling configurable): see
    _compute_padding()'s docstring for full details - fixed pixels
    (padding, default 4, matches crop_rows()'s own default), percentage
    of each row's own height (padding_pct, takes precedence when given),
    or asymmetric top/bottom overrides. Uses the SAME shared helper as
    crop_rows() specifically so the sidecar's stored bboxes never
    silently diverge from what actually gets cropped downstream -
    confirmed necessary (2026-07-13, caught by a pixel-level round-trip
    test) after an earlier version let these two functions' padding
    logic drift apart.

    Width/height for full-page bboxes come from result.deskewed_image_
    size, NOT a separately-passed source image size - confirmed as a
    real bug (2026-07-13, caught by the same round-trip test) when this
    function used to take a caller-supplied size: rotation with
    expand=True changes canvas dimensions, so the PRE-deskew source size
    is the wrong value for bboxes relative to the POST-deskew image.
    """
    width, height = result.deskewed_image_size
    left = x0 if x0 is not None else 0
    right = x1 if x1 is not None else width

    def _to_bbox(band: tuple[int, int]) -> list[int]:
        row_height = band[1] - band[0]
        top_pad, bottom_pad = _compute_padding(
            row_height, padding, padding_pct, padding_top, padding_bottom)
        y0 = max(0, band[0] - top_pad)
        y1 = min(height, band[1] + bottom_pad)
        return [left, y0, right, y1]

    table_bbox = None
    if table_top is not None and table_bottom is not None:
        table_bbox = [left, table_top, right, table_bottom]

    metadata_bbox = None
    if metadata_bottom is not None:
        metadata_bbox = [0, 0, width, metadata_bottom]

    return {
        "source_image_path": source_image_path,
        "deskewed_image_size": [width, height],
        "deskew_angle": result.deskew_angle,
        "coordinate_space": "deskewed_image",
        "mode": mode,
        "parameters": parameters,
        "table_bbox": table_bbox,
        "metadata_bbox": metadata_bbox,
        "header_bbox": _to_bbox(result.header_band) if result.header_band else None,
        "mask_keep_ranges": [[k0, k1] for k0, k1 in mask_keep_ranges] if mask_keep_ranges else [],
        "mask_apply_header": mask_apply_header,
        "mask_apply_rows": mask_apply_rows,
        "rows": [
            {"index": i + 1, "bbox": _to_bbox(band)}
            for i, band in enumerate(result.bands)
        ],
        "dropped_bands": [_to_bbox(b) for b in result.dropped_bands],
        "warnings": result.warnings,
    }


def save_sidecar(sidecar: dict, path) -> None:
    """
    Full overwrite - still used for the page-level geometry build
    (build_sidecar's output) since that's the one case where replacing
    the whole file IS correct (a fresh 'Refine rows' + Save really does
    supersede prior geometry). Per-column state should go through
    update_sidecar() instead, which merges rather than replaces.

    Atomic (temp file + os.replace) so a crash mid-write never leaves a
    truncated/corrupt sidecar behind - required for the resume-after-
    interruption guarantee the per-column workflow depends on.
    """
    import json
    import os
    import tempfile
    path = str(path)
    dir_ = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_, prefix=".sidecar_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(sidecar, f, indent=2)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def load_sidecar(path) -> dict:
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def init_column_state(sidecar: dict, column_order: list[str]) -> dict:
    """
    Ensures a sidecar has the per-column progression scaffold
    ("columns", "column_order", "active_column", "progress"), adding it
    if absent. Idempotent - safe to call on every load.

    Migration: an older sidecar (pre this feature) may have a flat,
    global mask_keep_ranges/mask_apply_header/mask_apply_rows from the
    single-mask-per-file era. Rather than discarding that work, it's
    carried over into the FIRST column in column_order as its starting
    mask - a reasonable default since that's usually the column that
    was actually being isolated when the file was last saved under the
    old scheme. Every other column starts with an empty mask, same as
    if it had never been touched.
    """
    if "columns" in sidecar and "column_order" in sidecar:
        # Already migrated - but column_order may have grown since (a
        # new column added to the form after this sidecar was created).
        # Add any new names as pending, empty-mask entries, without
        # touching existing ones.
        for name in column_order:
            sidecar["columns"].setdefault(name, _empty_column_state())
        sidecar["column_order"] = column_order
        if sidecar.get("active_column") not in sidecar["columns"]:
            sidecar["active_column"] = _first_incomplete(sidecar)
        sidecar["progress"] = _compute_progress(sidecar)
        return sidecar

    legacy_ranges = [tuple(k) for k in sidecar.get("mask_keep_ranges", [])]
    legacy_apply_header = sidecar.get("mask_apply_header", False)
    legacy_apply_rows = sidecar.get("mask_apply_rows", True)

    columns = {}
    for i, name in enumerate(column_order):
        state = _empty_column_state()
        if i == 0 and legacy_ranges:
            state["mask_keep_ranges"] = [list(r) for r in legacy_ranges]
            state["mask_apply_header"] = legacy_apply_header
            state["mask_apply_rows"] = legacy_apply_rows
        columns[name] = state

    sidecar["columns"] = columns
    sidecar["column_order"] = list(column_order)
    sidecar["active_column"] = column_order[0] if column_order else None
    sidecar["progress"] = _compute_progress(sidecar)
    return sidecar


def _empty_column_state() -> dict:
    return {
        "status": "pending",  # pending | in_progress | done | needs_review
        "mask_keep_ranges": [],
        "mask_apply_header": False,
        "mask_apply_rows": True,
        "results": {},
        "extraction_meta": {},
    }


def _compute_progress(sidecar: dict) -> dict:
    columns = sidecar.get("columns", {})
    total = len(columns)
    completed = sum(1 for c in columns.values() if c.get("status") == "done")
    return {"completed": completed, "total": total}


def _first_incomplete(sidecar: dict) -> str | None:
    for name in sidecar.get("column_order", []):
        if sidecar["columns"].get(name, {}).get("status") != "done":
            return name
    return None


def _deep_merge(base: dict, patch: dict) -> dict:
    """Recursive dict merge - patch values win, nested dicts merge
    rather than replace wholesale (so e.g. patching just
    extraction_meta doesn't blow away an existing results dict that
    happens to live alongside it at the same level)."""
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def update_sidecar(path, column_name: str, patch: dict, column_order: list[str] | None = None) -> dict:
    """
    Merges `patch` into columns[column_name] of the sidecar at `path`
    and writes it back atomically, WITHOUT touching any other column's
    state or the page-level geometry fields (rows/table_bbox/etc.) -
    the core operation the persistent-sidecar workflow needs so that
    masking or extracting one column never destroys another's saved
    work.

    If the sidecar doesn't yet have the per-column scaffold, it's
    added first via init_column_state() (column_order is required in
    that case - there's no way to invent the full column list from a
    single patch). If it's already present, column_order is optional
    and only used to pick up newly-added columns.

    Also recomputes "progress" and, if the patched column's status
    became "done", advances "active_column" to the next incomplete
    column in column_order - so a caller doesn't need a separate call
    just to move the pointer forward in the common case. Callers that
    want the OLD active column to stay active despite marking it done
    (e.g. going back to fix something) should not set status: "done"
    in the same patch as an unrelated correction; call again after.

    Returns the full merged sidecar dict (already written to disk).
    """
    sidecar = load_sidecar(path)
    if "columns" not in sidecar:
        if not column_order:
            raise ValueError(
                f"Sidecar at {path} has no per-column state yet and no "
                "column_order was given to initialize it.")
        init_column_state(sidecar, column_order)
    elif column_order:
        init_column_state(sidecar, column_order)  # idempotent - picks up new names only

    sidecar["columns"].setdefault(column_name, _empty_column_state())
    _deep_merge(sidecar["columns"][column_name], patch)

    sidecar["progress"] = _compute_progress(sidecar)
    if sidecar["columns"][column_name].get("status") == "done" and \
       sidecar.get("active_column") == column_name:
        sidecar["active_column"] = _first_incomplete(sidecar)

    save_sidecar(sidecar, path)
    return sidecar


def advance_column(sidecar: dict, mark_current_done: bool = True) -> dict:
    """
    Pure state transition (does not save) - marks the current
    active_column "done" (unless mark_current_done=False, e.g. the
    operator just wants to jump ahead without finishing the current
    one) and moves active_column to the next pending/in_progress entry
    in column_order. Caller is responsible for persisting via
    update_sidecar() with the resulting active_column, and for
    rebuilding the mask overlay from that column's stored
    mask_keep_ranges afterward.
    """
    current = sidecar.get("active_column")
    if current is not None and mark_current_done and current in sidecar.get("columns", {}):
        sidecar["columns"][current]["status"] = "done"
    sidecar["progress"] = _compute_progress(sidecar)
    sidecar["active_column"] = _first_incomplete(sidecar)
    return sidecar


def compute_exclude_ranges(
    keep_ranges: list[tuple[int, int]], full_width: int
) -> list[tuple[int, int]]:
    """
    Computes the actual paint-white ranges (for apply_column_mask) as
    the COMPLEMENT of what the user selected to keep - built 2026-07-15
    after Jon corrected the original design: click the column(s) to
    KEEP, not each column to exclude (isolating one narrow target
    column like Age shouldn't require manually excluding everything
    else around it).

    If keep_ranges is empty, returns [] (nothing excluded - matches
    "no selection made yet" meaning "show everything", not "exclude
    everything"). Overlapping/unsorted keep_ranges are merged correctly
    before computing the gaps between them.
    """
    if not keep_ranges:
        return []
    merged = sorted(keep_ranges)
    combined: list[list[int]] = []
    for x0, x1 in merged:
        if combined and x0 <= combined[-1][1]:
            combined[-1][1] = max(combined[-1][1], x1)
        else:
            combined.append([x0, x1])

    exclude: list[tuple[int, int]] = []
    cursor = 0
    for x0, x1 in combined:
        if x0 > cursor:
            exclude.append((cursor, x0))
        cursor = max(cursor, x1)
    if cursor < full_width:
        exclude.append((cursor, full_width))
    return exclude


def tight_crop_to_ranges(
    image: Image.Image,
    keep_ranges: list[tuple[int, int]],
    crop_x0: int = 0,
    padding_px: int = 20,
    padding_pct: float | None = None,
) -> Image.Image:
    """
    Crops DOWN to a padded region around the union of keep_ranges -
    added 2026-07-22 after discovering that apply_column_mask() (above)
    only ever PAINTS outside the kept range white, it never narrows the
    image's actual pixel dimensions. A masked single-column crop was
    still the FULL row width (often 3000-4000px for a wide census
    table) with the real content occupying as little as ~9% of that
    width - every extraction call and every LoRA training image was
    sending a mostly-blank image to the model, with the real
    handwriting compressed to a sliver by whatever downsampling the
    model's own image preprocessor does before its vision encoder.

    keep_ranges are in FULL-IMAGE coordinates (same convention as every
    other range/bbox in this module) - crop_x0 (the already-cropped
    region's own x0 origin, e.g. the row bbox's left edge) translates
    them into `image`'s local coordinate space, same convention as
    apply_column_mask().

    padding_pct, if given, overrides padding_px with a padding equal to
    that fraction of the kept-range width (e.g. 0.1 = 10% of the
    range's own width added to each side) - useful when column widths
    vary a lot across a page and a single fixed pixel padding would be
    proportionally huge on a narrow column and negligible on a wide
    one. padding_px is the simpler default: a fixed margin regardless
    of column width.

    Clamps the final box to `image`'s own bounds - a keep_range that
    falls partially or fully outside this particular crop (e.g. a
    stale mask from a different row width) never produces a crop
    request outside the source image, which PIL would raise on.

    Returns `image` UNCHANGED if keep_ranges is empty (nothing to crop
    to) or the computed box is degenerate (zero or negative width),
    since a caller with no real mask has nothing to tighten around -
    matches apply_column_mask()'s existing no-op-on-empty convention.
    """
    if not keep_ranges:
        return image
    width, height = image.size
    local_ranges = [(x0 - crop_x0, x1 - crop_x0) for x0, x1 in keep_ranges]
    range_left = min(r[0] for r in local_ranges)
    range_right = max(r[1] for r in local_ranges)

    if padding_pct is not None:
        pad = (range_right - range_left) * padding_pct
    else:
        pad = padding_px

    left = max(0, round(range_left - pad))
    right = min(width, round(range_right + pad))
    if right <= left:
        return image
    return image.crop((left, 0, right, height))


def apply_column_mask(
    image: Image.Image,
    mask_ranges: list[tuple[int, int]],
    crop_x0: int = 0,
) -> Image.Image:
    """
    Paints WHITE over the given x-ranges, spanning the full height of
    `image` - built 2026-07-15 after real testing confirmed Age
    contamination varies row-to-row (dwelling numbers on one row,
    section/township/range on another), pointing to a spatial-counting
    problem rather than a labeling problem: asking a model to count to
    "column 14" of ~30 densely-packed columns is hard regardless of
    whether the label is correct. Masking makes the unwanted columns'
    PIXELS literally blank rather than just asking the model to ignore
    them - there's nothing there to misread.

    mask_ranges are stored in FULL ORIGINAL (deskewed) image
    coordinates (matching every other bbox in this module), but this
    function is applied to an ALREADY-CROPPED region - crop_x0 (the
    crop's own x0 origin) translates each mask range into the crop's
    local coordinate space before painting. Ranges that fall entirely
    outside the crop are silently skipped (nothing to paint); ranges
    that partially overlap are clamped to the crop's own width.

    NOTE this only paints - it does NOT narrow image dimensions. See
    tight_crop_to_ranges() above for that (a deliberately separate
    function, not merged into this one, since some callers - the
    legacy whole-row multi-column extraction path - want the masking
    behavior WITHOUT tightening, e.g. when isolating several kept
    columns at once within one wide row image).

    Returns a NEW image - never mutates the input, same convention as
    every other transform in this module.
    """
    if not mask_ranges:
        return image
    result = image.copy()
    if result.mode != "RGB":
        result = result.convert("RGB")
    draw = ImageDraw.Draw(result)
    width, height = result.size
    for x0, x1 in mask_ranges:
        local_x0 = max(0, x0 - crop_x0)
        local_x1 = min(width, x1 - crop_x0)
        if local_x1 > local_x0:
            draw.rectangle([local_x0, 0, local_x1, height], fill="white")
    return result


def crop_region_from_source(
    source_image_path: str,
    bbox: list[int],
    deskew_angle: float = 0.0,
    mask_ranges: list[tuple[int, int]] | None = None,
    tight_crop_keep_ranges: list[tuple[int, int]] | None = None,
    tight_crop_padding_px: int = 20,
    tight_crop_padding_pct: float | None = None,
) -> Image.Image:
    """
    Loads the ORIGINAL source image fresh and crops a region using
    coordinates from a sidecar JSON - the primitive a future OCR stage
    needs ("load the original source image and use those coordinates to
    crop each region in memory").

    CORRECTNESS-CRITICAL: bbox coordinates were computed on the DESKEWED
    image (see segment_rows/segment_rows_periodic - deskew happens
    first, then detection), NOT the raw original. If deskew_angle is
    nonzero and this function skipped re-applying it, crops would be
    subtly or badly misaligned on any page that needed rotation
    correction (which was every real page tested this session, e.g.
    -1.25 degrees). This re-applies the EXACT SAME rotation
    (expand=True, matching deskew()'s own behavior) to a freshly-loaded
    copy of the original before cropping, so the stored coordinates are
    valid against the image actually being cropped - "load original"
    means "reproduce the same pipeline state the coordinates were
    computed against," not "skip the correction step that made those
    coordinates valid in the first place."

    mask_ranges (2026-07-15): optional column-mask ranges in the SAME
    full-image coordinate space as bbox - applied AFTER cropping, via
    apply_column_mask() above, with crop_x0=bbox[0] so ranges land in
    the right place regardless of where this particular crop starts.

    tight_crop_keep_ranges (2026-07-22, opt-in, backward compatible -
    default None means IDENTICAL behavior to before this parameter
    existed): the ORIGINAL kept ranges (not the excluded/painted ones
    mask_ranges holds), in the same full-image coordinate space. If
    given, the returned image is additionally narrowed to a padded box
    around their union via tight_crop_to_ranges() - fixes single-
    column extraction/training crops being ~3800px wide and ~90% blank
    (see tight_crop_to_ranges()'s docstring). Deliberately NOT applied
    automatically whenever mask_ranges is set, because the legacy
    whole-row multi-column extraction path masks OUT unwanted columns
    while keeping several wanted ones spread across the row - tightening
    that case would cut off the very columns it's supposed to keep.
    Callers that isolate exactly ONE column (run_single_column_extraction,
    export_lora_dataset.py) should pass this explicitly.

    IMPORTANT: this function's job is unchanged for every EXISTING
    caller - none of them pass tight_crop_keep_ranges, so none of them
    see any behavior change from this parameter's addition. The row
    bbox geometry stored in the sidecar itself (sidecar["rows"]) is
    never touched by tightening - only the in-memory image this
    function RETURNS is narrower; the sidecar's own coordinate records
    stay full-row, so nothing downstream that relies on those
    coordinates (e.g. re-deriving bbox math, or a future caller that
    wants the untightened crop) is affected.
    """
    image = Image.open(source_image_path)
    if deskew_angle != 0.0:
        fill = (255, 255, 255) if image.mode == "RGB" else 255
        image = image.rotate(
            deskew_angle, expand=True, fillcolor=fill, resample=Image.BICUBIC
        )
    x0, y0, x1, y1 = bbox
    cropped = image.crop((x0, y0, x1, y1))
    if mask_ranges:
        cropped = apply_column_mask(cropped, mask_ranges, crop_x0=x0)
    if tight_crop_keep_ranges:
        cropped = tight_crop_to_ranges(
            cropped, tight_crop_keep_ranges, crop_x0=x0,
            padding_px=tight_crop_padding_px, padding_pct=tight_crop_padding_pct,
        )
    return cropped


def segment_rows(
    image: Image.Image,
    header_row_count: int = 1,
    padding: int = 4,
    padding_pct: float | None = None,
    padding_top: int | None = None,
    padding_bottom: int | None = None,
) -> tuple[RowDetectionResult, list[Image.Image], Image.Image, Image.Image]:
    """
    Full pipeline through the agreed build order: deskew -> detect ->
    merge wrapped -> sanity check -> crop -> debug overlay. Does NOT
    call any OCR model - returns row crops for visual review only, per
    the agreed scope ("prove the segmentation layer before adding
    model-call complexity").

    header_row_count: how many of the FIRST detected bands (after
    merging/sanity-checking) to treat as the header block, merged into
    one header region prepended to every data row crop. First-prototype
    approach per the agreed plan - automatic header-region detection
    (distinguishing it from data rows structurally) is a harder, later
    problem, not in scope here.

    Returns (result, row_crops, header_crop, debug_overlay_image) -
    the deskewed image is available via result if needed, but callers
    mainly want row_crops (to inspect) and debug_overlay_image (to see
    what was detected and why).
    """
    deskewed, angle = deskew(image)
    raw_bands, ruling_line_count, ruling_line_rows = detect_row_bands(deskewed)
    merged_bands = merge_wrapped_bands(raw_bands, ruling_line_rows=ruling_line_rows)
    kept_bands, dropped_bands, warnings = sanity_check_bands(merged_bands)

    warnings.insert(0, f"Deskew angle applied: {angle:.2f} degrees.")
    warnings.insert(
        1,
        f"Detected {ruling_line_count} ruling-line row(s), excluded from "
        f"content bands. {'This confirms the ruling-line exclusion engaged - ' if ruling_line_count > 0 else 'Zero detected - check ruling_line_run_ratio if this page has a ruled table and rows still look merged. '}"
    )

    header_band = None
    data_bands = kept_bands
    if header_row_count > 0 and len(kept_bands) > header_row_count:
        header_bands = kept_bands[:header_row_count]
        header_band = (header_bands[0][0], header_bands[-1][1])
        data_bands = kept_bands[header_row_count:]

    row_crops, header_crop = crop_rows(
        deskewed, data_bands, header_band=header_band,
        padding=padding, padding_pct=padding_pct,
        padding_top=padding_top, padding_bottom=padding_bottom,
    )
    debug_overlay = render_debug_overlay(
        deskewed, kept_bands, dropped_bands=dropped_bands, header_band=header_band
    )

    result = RowDetectionResult(
        bands=data_bands, header_band=header_band, deskew_angle=angle,
        deskewed_image_size=deskewed.size, warnings=warnings, dropped_bands=dropped_bands,
    )
    return result, row_crops, header_crop, debug_overlay
