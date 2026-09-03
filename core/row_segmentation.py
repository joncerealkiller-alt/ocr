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
    Crops each detected row band into its own image. header_band (if
    given) is cropped and returned separately as header_crop, but is NO
    LONGER prepended to every row crop (2026-07-26, per Jon's direction -
    this is exactly the "optimize away the redundancy later" this
    docstring already flagged as a known first-prototype-only
    compromise). Prepending the header image to every row made sense
    while extraction still needed the header's column labels visible
    inside a whole-row image to identify which field was which; it's
    genuinely redundant now that stage1/stage2 both operate per-FIELD,
    with the field's identity given explicitly via the prompt (Field:
    {name}), not inferred visually from a stacked header band - real
    extraction never used this combined image anyway (row["bbox"] in the
    sidecar/RowDetectionResult is always the row-only band, see
    build_sidecar() below), so this only ever affected the human-facing
    debug row crops (row_segmentation_ui.py's "Also save individual row
    PNGs" checkbox) - needless extra image height for a reviewer to
    scroll past, no benefit.

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
    left_number_candidates=None,
    right_number_candidates=None,
    agreement_tolerance: int = 5,
) -> tuple[int, str]:
    """
    Locally searches a small window around an EXPECTED boundary position
    (from periodic spacing, not blind whole-page search) for the best
    actual cut point. Returns (position, source) - source is one of
    "both_number_columns", "number_columns_disagree", "left_number_column",
    "right_number_column", "ruling_line", "density_minimum" - kept as real
    telemetry, not silently discarded (2026-08-05, per direct instruction:
    "agreement itself is useful telemetry... disagreement becomes
    measurable instead of silently hidden").

    Priority order, per boundary independently:
      1. Left and right printed row-number columns (2026-08-05) - kept as
         SEPARATE candidate pools, not concatenated-and-pick-nearest (a
         locally noisy candidate from one side could otherwise outrank a
         genuinely correct one from the other side just by sitting
         fractionally closer to the periodic estimate - confirmed as a
         real regression on a real page before this fix). Each side's
         nearest in-window candidate is found independently:
           - both found, agree within agreement_tolerance -> their
             average, source="both_number_columns" (highest confidence -
             two independent signals corroborate).
           - both found, disagree -> whichever is closer to the periodic
             expected position, source="number_columns_disagree" (a real,
             flagged uncertainty - e.g. one side has a printed defect -
             not silently resolved).
           - only one side found -> use it, source="left_number_column"/
             "right_number_column" (the other side rescues a printed
             defect - confirmed real cause: an ink blot obscuring one
             row's number on the opposite margin).
      2. A detected ruling line within the window (strong structural
         signal), source="ruling_line".
      3. Local density minimum (best available gap) - final fallback,
         source="density_minimum".

    Confined to a narrow window around a periodic estimate, so it's far
    less sensitive to whatever caused global ruling-line detection to
    fail to engage at all on a real scan (2026-07-12) than the general
    detector was.
    """
    lo = max(0, expected_y - search_radius)
    hi = min(len(density), expected_y + search_radius + 1)

    def nearest_in_window(candidates):
        if not candidates:
            return None
        in_window = [c for c in candidates if lo <= c <= hi]
        if not in_window:
            return None
        return min(in_window, key=lambda y: abs(y - expected_y))

    left = nearest_in_window(left_number_candidates)
    right = nearest_in_window(right_number_candidates)

    if left is not None and right is not None:
        if abs(left - right) <= agreement_tolerance:
            return round((left + right) / 2), "both_number_columns"
        if abs(left - expected_y) <= abs(right - expected_y):
            return int(round(left)), "number_columns_disagree"
        return int(round(right)), "number_columns_disagree"
    if left is not None:
        return int(round(left)), "left_number_column"
    if right is not None:
        return int(round(right)), "right_number_column"

    ruling_set = set(int(y) for y in ruling_line_rows) if ruling_line_rows is not None else set()
    ruling_in_window = [y for y in range(lo, hi) if y in ruling_set]
    if ruling_in_window:
        # If multiple, take the one closest to the expected position.
        return min(ruling_in_window, key=lambda y: abs(y - expected_y)), "ruling_line"

    window = density[lo:hi]
    if len(window) == 0:
        return expected_y, "density_minimum"
    local_min_offset = int(np.argmin(window))
    return lo + local_min_offset, "density_minimum"


def detect_row_number_centers(
    image: Image.Image,
    number_column_left: int,
    number_column_right: int,
    y0: int,
    y1: int,
    close_gap_px: int = 3,
    max_digit_run_px: int | None = None,
) -> list[float]:
    """
    Detects the vertical center of each printed row-number in a narrow
    column strip (the "1, 2, 3..." row-number column many census forms
    print at the left edge of the table) - a far stronger row anchor
    than ruling-line detection on forms with thin/faint/interrupted
    printed lines, since printed numerals are bold, isolated, and high-
    contrast by comparison. Confirmed 2026-08-05 on a real 1901 census
    page where ruling-line detection couldn't clear even a heavily
    gap-bridged 0.3 run-ratio threshold, but row-number centers matched
    the true (non-uniform) row spacing cleanly - 50/50 rows found,
    heights 34.8-41.3px vs segment_rows_uniform_tile's 36.84px single-
    row measurement on the same page.

    The column typically also contains one continuous vertical RULE LINE
    (the printed border to the right of the numbers), excluded before
    computing row density - via VERTICAL RUN LENGTH (the same longest-
    continuous-run technique detect_row_bands() already uses for
    horizontal ruling lines, transposed to the vertical axis), NOT total
    per-column ink coverage. Coverage alone was tried first and confirmed
    insufficient on a real page: page warp made the line drift gradually
    across ~5 columns (49-53 in one real strip) rather than sit in a
    single fixed column, so its PER-COLUMN coverage (0.22-0.70) never
    cleared even a generous 80% threshold, while digit columns (which
    repeat 50x down the page) independently reached up to 0.23 coverage
    too - the two overlap in raw coverage and can't be told apart that
    way. Run LENGTH separates them cleanly instead: the tallest
    continuous vertical run any real digit produced was 47px; the
    drifting line's least-covered column still ran 216px continuous -
    a 4.6x gap. max_digit_run_px defaults to None, which falls back to a
    fixed 60px (safely above any single glyph's height, safely below a
    real printed line's continuous run in any column it touches).

    A two-digit number's two digits (e.g. "1" and "0" of "10") don't need
    explicit clustering - summing ink density ACROSS the column's width
    naturally merges same-row content regardless of how many digit-glyphs
    produced it, so this needs no connected-component analysis, staying
    within this module's PIL+numpy-only, no-OpenCV/scipy design (see this
    file's own top-of-module docstring).

    Returns row-number vertical centers in the SAME y-coordinate space as
    y0/y1 (i.e. already offset by y0 - NOT relative-to-crop). Empty list
    if the column contains no detectable digits (e.g. wrong x-range, or
    this form doesn't print row numbers at all - callers should fall back
    to another detection strategy in that case, not treat [] as an error).
    """
    gray = image.convert("L")
    arr = np.array(gray)[y0:y1, number_column_left:number_column_right]
    if arr.size == 0:
        return []
    thresh = _otsu_threshold(arr)
    binary = (arr <= thresh).astype(np.uint8)

    threshold_px = max_digit_run_px if max_digit_run_px is not None else 60
    vertical_run = _longest_run_per_row(binary.T, close_gap_px=4)
    digit_only = binary[:, vertical_run <= threshold_px]
    if digit_only.shape[1] == 0:
        return []

    is_ink = digit_only.sum(axis=1) > 0
    closed = _close_1d(is_ink, close_gap_px)

    runs = []
    start = None
    for y, val in enumerate(closed):
        if val and start is None:
            start = y
        elif not val and start is not None:
            runs.append((start, y - 1))
            start = None
    if start is not None:
        runs.append((start, len(closed) - 1))

    # NOTE (2026-08-05): earlier revisions tried to force this list down
    # to exactly row_count runs here (merging any anomalously-small gap,
    # e.g. a printed period after a row number registering as its own
    # tiny run) - abandoned after confirming on a real page that widening
    # the merge threshold enough to catch that also merged two genuinely
    # separate ADJACENT rows elsewhere. Getting the COUNT exactly right
    # here isn't actually necessary: callers use these as boundary
    # CANDIDATES for a per-expected-position nearest-match search (see
    # refine_boundary_position()'s number_boundary_candidates parameter),
    # not as a directly-trusted final list - a spurious extra candidate
    # (like the period) simply never wins the "closest to this expected
    # row" comparison for any real row, and a missing one just means that
    # one boundary falls through to the ruling-line/density fallback
    # instead. Precision here would be nice but isn't load-bearing.
    centers = [(a + b) / 2 + y0 for a, b in runs]
    return centers


def row_boundaries_from_number_centers(
    centers: list[float],
    table_top: float,
    table_bottom: float,
    alignment: str = "center",
) -> list[float]:
    """
    Converts detected row-number centers (detect_row_number_centers) into
    row BOUNDARIES (row_count+1 values, index i separating row i from row
    i+1). alignment describes where the printed number sits relative to
    its row's true content span - CONFIRMED to vary by census year/form
    (per direct instruction, 2026-08-05: some forms center the row number
    within its row, others print it nearer the row's bottom edge) - a
    per-template calibration choice, never a hardcoded assumption.

    "center" (default): boundary[i] = midpoint(center[i], center[i+1]) -
    correct when the number sits at the visual center of its row. Visually
    confirmed correct for the 1901 form this was built against (digit
    centers landed equidistant from the ruling lines above/below).
    "bottom": boundary[i] = center[i] directly - correct when the number
    sits at (or very near) the row's bottom edge, so its own position
    already approximates the boundary rather than the row's midpoint.

    First/last boundaries always come from table_top/table_bottom (the
    already visually-confirmed table extent) rather than extrapolated
    from spacing or trusted from the first/last detected center - those
    two values are the one part of this measurement with independent,
    directly-confirmed ground truth, so they should never be overridden
    by a detection result.
    """
    if not centers:
        return [table_top, table_bottom]
    if alignment == "bottom":
        boundaries = [table_top] + list(centers)
        boundaries[-1] = table_bottom
        return boundaries
    boundaries = [table_top]
    for i in range(len(centers) - 1):
        boundaries.append((centers[i] + centers[i + 1]) / 2)
    boundaries.append(table_bottom)
    return boundaries


def detect_column_number_centers(
    image: Image.Image,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    close_gap_px: int = 1,
    min_blob_width_px: int = 3,
    line_run_threshold_frac: float = 0.85,
) -> list[tuple[float, int]]:
    """
    Detects individual printed-number blobs along a HORIZONTAL band -
    the printed column-number row many census forms print near the top
    of the table (e.g. "1 2 3 4 5 6..." each centered over its own
    column). Column-axis counterpart to detect_row_number_centers(),
    same underlying technique transposed: exclude tall/continuous
    structure (here, VERTICAL divider lines crossing the band) via
    longest-run-length rather than raw coverage, then group remaining
    ink into blobs.

    PURE MEASUREMENT - same "Stage A: emit numbers, decide nothing"
    principle as core/image_analysis.py's physical sensors. This
    function does NOT know which column any blob belongs to, does NOT
    filter by a template's expected position, and does NOT feed into
    any boundary-refinement decision - it just reports what it found.
    Matching a blob to a specific expected column (using a template's
    approximate x_frac as a prior + a search-window tolerance) is the
    caller's job, same separation as row-number centers vs. row-
    boundary derivation. See nearest_number_candidate() below for that
    step, and docs/COLUMN_NUMBER_ANCHOR_RESEARCH.md for the full
    research history this function is built from.

    Research findings this implementation reflects (2026-08-06, tested
    against a real 1911 reference page with independently-confirmed
    column positions - e078_e001946617):
      - The underlying geometric principle holds: printed numbers land
        within 2-5px of their column's true center.
      - Raw whole-row blob detection is genuinely noisy - NOT primarily
        from gap-bridging (falsified directly: close_gap_px=0 barely
        changed the worst merges) but from real touching/near-touching
        ink in the source print that no threshold tuning resolves.
      - A dashed sub-header guide-line sometimes sits just above the
        true number row and must be excluded from y0/y1, not just the
        vertical divider lines - confirmed on the same reference page
        (a section-specific "Citizenship, Nationality and Religion"
        category underline at y=583-590, well above the actual digits
        at y=592-604). Callers should calibrate y0/y1 per template by
        direct visual check, the same discipline as everywhere else in
        this pipeline - never assumed generic.
      - Reliability comes from FILTERING (nearest_number_candidate()
        below, scoped to each column's own known approximate position),
        not from perfecting raw detection - this mirrors the "prior
        narrows the search window, existing detector stays authoritative"
        principle the whole column-anchor idea is built on.

    Prior art, explicitly NOT repeated here: an earlier, cruder column-
    number attempt (_detect_header_number_blobs() in core/auto_sidecar.py,
    2026-07-27) used simple density-relative-to-peak with no line
    exclusion and no filtering step, was found to worsen already-accurate
    pages, and was removed. This function exists specifically because
    the underlying idea wasn't re-tested with the techniques the row-
    number work later proved necessary - see this function's own research
    history above for what's different.

    Returns [(center_x, width_px), ...] sorted left to right, in FULL-
    IMAGE x-coordinates - NOT filtered, NOT deduplicated against a
    template, exactly what was found in the given band.
    """
    gray = image.convert("L")
    arr = np.array(gray)[y0:y1, x0:x1]
    if arr.size == 0:
        return []
    thresh = _otsu_threshold(arr)
    binary = (arr <= thresh).astype(np.uint8)
    band_h = binary.shape[0]

    # Exclude vertical divider lines crossing the band - per-column
    # longest VERTICAL run within this (short) band, not raw coverage
    # (confirmed necessary: coverage alone can't separate a line from
    # digit columns that also accumulate real ink across many numbers -
    # same lesson as the row-axis case, see detect_row_number_centers()).
    vertical_run = _longest_run_per_row(binary.T, close_gap_px=1)
    threshold_px = int(band_h * line_run_threshold_frac)
    keep_mask = vertical_run <= threshold_px
    digit_only = binary[:, keep_mask]
    if digit_only.shape[1] == 0:
        return []
    orig_x = np.where(keep_mask)[0]

    col_ink = digit_only.sum(axis=0) > 0
    closed = _close_1d(col_ink, close_gap_px)

    blobs = []
    start = None
    for i, v in enumerate(closed):
        if v and start is None:
            start = i
        elif not v and start is not None:
            blobs.append((orig_x[start], orig_x[i - 1]))
            start = None
    if start is not None:
        blobs.append((orig_x[start], orig_x[-1]))

    blobs = [(b0, b1) for b0, b1 in blobs if (b1 - b0) >= min_blob_width_px]
    return sorted((x0 + (b0 + b1) / 2, b1 - b0) for b0, b1 in blobs)


def nearest_number_candidate(
    candidates: list[float], expected_position: float, max_distance: float,
) -> float | None:
    """
    Generic "is there a plausible anchor near where I expect one" lookup
    - the filtering step that made column-number detection reliable
    despite noisy raw blobs elsewhere on the page (confirmed 2026-08-06:
    scoping to each column's own known approximate x_frac position found
    the correct candidate for every one of 5 known columns on a real
    reference page, while raw whole-row parsing alone was too noisy to
    trust blindly - the SAME distinction refine_boundary_position()
    already draws between a candidate list and blindly trusting every
    detection). Shared by both row-number and column-number anchoring -
    not duplicated per axis.

    Returns the candidate closest to expected_position if one exists
    within max_distance, else None (never guesses when nothing plausible
    is nearby - same "no signal" contract as detect_row_number_centers()
    returning an empty list).
    """
    if not candidates:
        return None
    in_window = [c for c in candidates if abs(c - expected_position) <= max_distance]
    if not in_window:
        return None
    return min(in_window, key=lambda c: abs(c - expected_position))


def _score_number_row_band(blobs: list[tuple[float, int]], table_width: float) -> float:
    """
    Plausibility score for a candidate number-row band - HOW clean does
    detect_column_number_centers()'s output look, not WHICH column
    anything is. Same "score, don't just accept" principle
    core/auto_sidecar.py's row-detection preset ladder already uses
    (_assess_band_detection_quality()), applied here to calibration
    instead of detection strategy.

    Rewards: a plausible blob count (real forms run ~25-50 printed
    numbers; too few means most of the row was missed, implausibly many
    means noise is dominating) and blobs mostly falling in a digit-like
    width range (3-30px, calibrated against real detections on two
    different census years). Penalizes: one blob spanning more than 5%
    of the table width - a strong sign the band caught a continuous
    structural feature (a ruling line, a dashed guide) rather than
    isolated digits, confirmed as the exact failure mode found by hand
    on a real 1911 page before this scoring function existed.

    TRIED AND REVERTED (docs/COLUMN_NUMBER_ANCHOR_RESEARCH.md's
    "Round 12"): scaling count_score's cap by search width instead of
    this fixed 70, to fix Round 10's noisy piecewise/narrow-zone sweep.
    Two calibrations of the scaling constant were tried; the first was
    calibrated against the wrong quantity (real column density instead
    of raw blob density) and clearly regressed the piecewise task; the
    corrected version improved the piecewise task somewhat but NEVER
    beat this fixed-70 baseline's own best case, AND silently broke the
    already-validated global-band results for 1911 and 1926 (shifted by
    91px and 210px respectively, landing on handwriting noise instead
    of the real number row) - a real regression, not an improvement,
    confirmed by directly re-running Round 2's validated samples. Kept
    as fixed 70 pending a genuinely better-founded fix; see Round 12 for
    the full data and the "plausible_width_frac quantization at low n"
    hypothesis that wasn't yet tried.
    """
    if not blobs:
        return -1.0
    widths = [w for _c, w in blobs]
    n = len(blobs)
    max_w = max(widths)
    merge_penalty = 3.0 if max_w > table_width * 0.05 else 0.0
    count_score = min(n, 70) / 70.0
    plausible_width_frac = sum(1 for w in widths if 3 <= w <= 30) / n
    return count_score + plausible_width_frac - merge_penalty


def find_number_row_band(
    image: Image.Image,
    x0: int,
    x1: int,
    search_y0: int,
    search_y1: int,
    band_height: int = 12,
    step: int = 3,
) -> tuple[int, int, list[tuple[float, int]], float] | None:
    """
    Automatically locates the printed column-number row's y-band within
    a broader search region, instead of requiring a human to calibrate
    it per template/per page by direct visual inspection (the manual
    process detect_column_number_centers()'s own research history was
    built on - see docs/COLUMN_NUMBER_ANCHOR_RESEARCH.md). Sweeps a
    sliding window of band_height across [search_y0, search_y1], runs
    detect_column_number_centers() on each candidate, and keeps
    whichever scores best via _score_number_row_band().

    Confirmed 2026-08-06 against three real census pages (1911, 1931,
    and a genuinely different-year form, 1926): on the two pages a human
    had already carefully calibrated by hand, the auto-found band landed
    within 1-12px of the manual one and detected the same or a very
    similar blob count. On the 1926 page - where an initial rough manual
    guess at the band performed badly (2/12 numbers visible in one
    section) - the auto-search found a substantially better band on its
    own, without any human recalibration. Its SCORE also correctly
    stayed low/negative for that page, honestly signaling reduced
    confidence rather than reporting false certainty - this function
    reports what it found, it does not decide whether the result is
    "good enough" for any downstream use.

    search_y0/search_y1 should still be a reasonable region to search
    within (e.g. the gap between a template's metadata_bottom and
    table_top, or a similarly-sized window above a measured table_top) -
    this narrows WHERE to sweep, not WHICH exact pixels the number row
    occupies within that region. Not validated on forms with no printed
    column-number row at all - callers should expect a low score in
    that case (nothing plausible to find), not silently trust a result.

    Returns (band_y0, band_y1, blobs, score) for the best-scoring
    candidate, or None if search_y1 - search_y0 < band_height.
    """
    table_width = x1 - x0
    best = None
    best_score = float("-inf")
    y = search_y0
    while y + band_height <= search_y1:
        blobs = detect_column_number_centers(image, x0, x1, y, y + band_height)
        score = _score_number_row_band(blobs, table_width)
        if score > best_score:
            best_score = score
            best = (y, y + band_height, blobs, score)
        y += step
    return best


def find_number_row_band_piecewise(
    image: Image.Image,
    x0: int,
    x1: int,
    search_y0: int,
    search_y1: int,
    n_zones: int = 3,
    band_height: int = 12,
    step: int = 3,
) -> list[tuple[int, int, int, int, list[tuple[float, int]], float]]:
    """
    Per-zone counterpart to find_number_row_band(), built after a real
    1901 page (docs/COLUMN_NUMBER_ANCHOR_RESEARCH.md's "Round 9") showed
    the printed number row isn't always level: independently calibrating
    the SAME page's left/middle/right thirds found three genuinely
    different optimal bands (~80px apart, ~1.5deg effective tilt) - a
    single global band ends up being a compromise that's not really
    correct anywhere on a page like that. Root-caused as real page
    skew/curvature, not a detection bug (visually confirmed by cropping
    each zone) - see the research doc for the full writeup, including
    what was ruled out first (filter thresholds, touching digits).

    Splits [x0, x1] into n_zones equal-width slices and runs
    find_number_row_band() independently on each. On a page whose
    header IS level, expect each zone to land on nearly the same band
    anyway - this isn't a form-specific special case, just a strictly
    more general calibration that degrades gracefully to the global
    case when there's no real drift to find.

    Returns [(zone_x0, zone_x1, band_y0, band_y1, blobs, score), ...],
    one entry per zone (a zone is skipped if find_number_row_band()
    returns None for it, e.g. search_y1-search_y0 < band_height).
    """
    zone_width = (x1 - x0) / n_zones
    zones = []
    for i in range(n_zones):
        zx0 = int(round(x0 + i * zone_width))
        zx1 = x1 if i == n_zones - 1 else int(round(x0 + (i + 1) * zone_width))
        band = find_number_row_band(image, zx0, zx1, search_y0, search_y1, band_height, step)
        if band is None:
            continue
        by0, by1, blobs, score = band
        zones.append((zx0, zx1, by0, by1, blobs, score))
    return zones


def detect_column_number_centers_piecewise(
    image: Image.Image,
    x0: int,
    x1: int,
    search_y0: int,
    search_y1: int,
    n_zones: int = 3,
    band_height: int = 12,
    step: int = 3,
) -> list[tuple[float, int]]:
    """
    Drop-in replacement for "calibrate one global band, then detect
    column-number centres once over the full table width" - calibrates
    a separate band PER ZONE via find_number_row_band_piecewise() first,
    then merges each zone's own blobs (each zone's blobs are already
    x-scoped to that zone by construction, so no de-duplication needed).
    See find_number_row_band_piecewise()'s docstring for why this
    exists (a real, visually-confirmed non-level header row on a 1901
    page) and docs/COLUMN_NUMBER_ANCHOR_RESEARCH.md's "Round 9"/"Round
    10" for validated before/after match-rate numbers.
    """
    zones = find_number_row_band_piecewise(image, x0, x1, search_y0, search_y1, n_zones, band_height, step)
    merged: list[tuple[float, int]] = []
    for _zx0, _zx1, _by0, _by1, blobs, _score in zones:
        merged.extend(blobs)
    return sorted(merged, key=lambda t: t[0])


def _find_number_row_band_anchors(
    image: Image.Image,
    x0: int,
    x1: int,
    search_y0: int,
    search_y1: int,
    n_anchors: int,
    band_height: int,
    step: int,
) -> list[tuple[float, float]]:
    """
    Runs find_number_row_band() at n_anchors WIDE, roughly-equal slices
    of [x0, x1] - deliberately coarse (Round 9's left/middle/right-
    thirds check confirmed zones at this scale score cleanly, 1.31-1.40,
    with no sign of the narrow-window scorer instability Round 10/12
    hit trying to search many NARROW zones directly). Returns
    [(anchor_x_center, anchor_band_y_center), ...], sorted by x, for
    anchors where a band was actually found.
    """
    zone_width = (x1 - x0) / n_anchors
    anchors = []
    for i in range(n_anchors):
        zx0 = int(round(x0 + i * zone_width))
        zx1 = x1 if i == n_anchors - 1 else int(round(x0 + (i + 1) * zone_width))
        band = find_number_row_band(image, zx0, zx1, search_y0, search_y1, band_height, step)
        if band is None:
            continue
        by0, by1, _blobs, _score = band
        anchors.append(((zx0 + zx1) / 2.0, (by0 + by1) / 2.0))
    return sorted(anchors, key=lambda a: a[0])


def _interpolate_band_center(anchors: list[tuple[float, float]], x: float) -> float:
    """
    Linear interpolation (or nearest-segment-slope extrapolation past
    the outermost anchors) of the expected band CENTER y at position x,
    from a small set of (x_center, y_center) anchor measurements. A
    single anchor degenerates to a flat line (same as the old global-
    band behaviour). No scoring happens here - this is pure arithmetic
    over already-trusted wide-scale measurements, which is the whole
    point: it never re-invokes the unstable narrow-window scorer.
    """
    if len(anchors) == 1:
        return anchors[0][1]
    xs = [a[0] for a in anchors]
    ys = [a[1] for a in anchors]
    if x <= xs[0]:
        i0, i1 = 0, 1
    elif x >= xs[-1]:
        i0, i1 = len(xs) - 2, len(xs) - 1
    else:
        i0, i1 = 0, 1
        for i in range(len(xs) - 1):
            if xs[i] <= x <= xs[i + 1]:
                i0, i1 = i, i + 1
                break
    dx = xs[i1] - xs[i0]
    slope = (ys[i1] - ys[i0]) / dx if dx else 0.0
    return ys[i0] + slope * (x - xs[i0])


def detect_column_number_centers_trend(
    image: Image.Image,
    x0: int,
    x1: int,
    search_y0: int,
    search_y1: int,
    n_anchors: int = 3,
    n_slices: int = 10,
    band_height: int = 12,
    step: int = 3,
) -> list[tuple[float, int]]:
    """
    Third calibration strategy for a header row that isn't level (see
    find_number_row_band_piecewise()'s docstring for the original 1901
    finding). Round 10/12's piecewise approach re-ran the full search-
    and-SCORE sweep independently in each of up to 16 zones, which hit
    real instability in _score_number_row_band() at narrow scale
    (confirmed and documented, not just suspected - see docs/COLUMN_
    NUMBER_ANCHOR_RESEARCH.md's "Round 12" for the regression that
    resulted from trying to fix the scorer instead of avoiding it).

    This strategy sidesteps that entirely: take only `n_anchors` WIDE
    measurements (reliable at that scale, per Round 9), fit a simple
    linear trend through them, then INTERPOLATE (pure arithmetic, no
    search, no scoring) the expected band for `n_slices` finer x-slices
    from that trend. `n_slices` controls detection resolution; `n_anchors`
    controls how many real (reliable) measurements the trend is built
    from - these are independent knobs, unlike the piecewise approach
    where more zones meant more narrow (unreliable) searches.
    """
    anchors = _find_number_row_band_anchors(image, x0, x1, search_y0, search_y1, n_anchors, band_height, step)
    if not anchors:
        return []

    slice_width = (x1 - x0) / n_slices
    merged: list[tuple[float, int]] = []
    for i in range(n_slices):
        sx0 = int(round(x0 + i * slice_width))
        sx1 = x1 if i == n_slices - 1 else int(round(x0 + (i + 1) * slice_width))
        yc = _interpolate_band_center(anchors, (sx0 + sx1) / 2.0)
        sy0 = int(round(yc - band_height / 2))
        sy1 = sy0 + band_height
        blobs = detect_column_number_centers(image, sx0, sx1, sy0, sy1)
        merged.extend(blobs)
    return sorted(merged, key=lambda t: t[0])


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
    number_column_left: int | None = None,
    number_column_right: int | None = None,
    number_column2_left: int | None = None,
    number_column2_right: int | None = None,
    row_number_alignment: str = "center",
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

    number_column_left/number_column_right (and the optional
    number_column2_left/number_column2_right for a SECOND printed
    row-number column - many census forms print numbers down both the
    left AND right margins, e.g. confirmed on a real 1901 page)/
    row_number_alignment (2026-08-05): when a column's bounds are given,
    detect_row_number_centers() runs against it and the resulting
    boundary candidates feed into refine_boundary_position() as its
    highest-priority signal PER BOUNDARY (see that function's own
    docstring - not a whole-page trust-or-reject switch). Candidates from
    both columns, when both are configured, are pooled together - the
    nearest one within a boundary's search window wins regardless of
    which side it came from, so a printed defect on one side (an ink
    blot obscuring a digit, confirmed as a real cause on a real page) is
    covered by the other side rather than losing that boundary's anchor
    entirely. A boundary with no number-column candidate nearby (or none
    configured at all) falls through to ruling-line detection, then
    local density minimum, same as before this signal existed.
    """
    if deskew_angle is not None:
        deskewed = apply_deskew_angle(image, deskew_angle)
        angle = deskew_angle
    else:
        deskewed, angle = deskew(image)

    x0, x1 = table_left, table_right

    if table_top is None or table_bottom is None:
        auto_top, auto_bottom = estimate_table_extent(deskewed, x0=x0, x1=x1)
        table_top = table_top if table_top is not None else auto_top
        table_bottom = table_bottom if table_bottom is not None else auto_bottom

    # Scope density/ruling-line detection to the actual DATA-ROW region -
    # header_box_bottom if a header box is confirmed, else table_top -
    # down to table_bottom, NOT the whole page (2026-08-05, per Jon's
    # direction, confirmed on a real 1901 census page: table_top/
    # table_bottom exactly on the true first/last row borders still
    # produced misaligned intermediate rows). A header block above the
    # table has its own print density/ruling-line characteristics
    # (titles, multi-tier column labels) that pollute the Otsu threshold
    # and ruling-line classification computed over the whole page,
    # throwing off refinement even when the table's own top/bottom are
    # exactly right. Detection runs entirely in ROI-LOCAL coordinates
    # (0 = data_top); only the final refined boundary is shifted back to
    # full-image coordinates - density/ruling_line_rows/expected_boundaries
    # must never mix local and full-image coordinate spaces.
    data_top = header_box_bottom if header_box_bottom is not None else table_top

    total_span = table_bottom - table_top
    expected_row_height = total_span / row_count if row_count > 0 else total_span
    search_radius = max(2, int(expected_row_height * search_radius_ratio))

    # Row-number-column signal (2026-08-05): left and right columns are
    # kept as SEPARATE candidate pools, deliberately not concatenated -
    # refine_boundary_position() finds each side's nearest in-window
    # candidate independently and only THEN compares them (agree/
    # disagree/only-one-found), so a locally noisy candidate on one side
    # can never silently outrank a genuinely correct one on the other
    # side just by sitting fractionally closer to the periodic estimate
    # (confirmed as a real regression on a real page when both sides were
    # blindly pooled together first). A missing or spurious candidate on
    # either side only affects that ONE boundary's search, falling
    # through the priority cascade rather than discarding the signal for
    # the whole page. table_top/table_bottom themselves are excluded from
    # both candidate pools - those come from row_boundaries_from_number_
    # centers() as already-trusted endpoints, not something to re-search
    # for.
    def _local_candidates(col_left, col_right) -> list[float]:
        if col_left is None or col_right is None:
            return []
        number_centers = detect_row_number_centers(
            deskewed, col_left, col_right, data_top, table_bottom,
        )
        if not number_centers:
            return []
        raw_candidates = row_boundaries_from_number_centers(
            number_centers, table_top, table_bottom, alignment=row_number_alignment,
        )
        return [c - data_top for c in raw_candidates[1:-1]]

    left_candidates_local = _local_candidates(number_column_left, number_column_right)
    right_candidates_local = _local_candidates(number_column2_left, number_column2_right)
    number_centers_found = len(left_candidates_local) + len(right_candidates_local)

    roi = deskewed.crop((
        x0 or 0, data_top,
        x1 if x1 is not None else deskewed.width, table_bottom,
    ))
    density, _ = _row_density_profile(roi)
    _, _, ruling_line_rows = detect_row_bands(roi)

    expected_boundaries_local = [
        (table_top - data_top) + round(i * expected_row_height) for i in range(row_count + 1)
    ]
    boundary_results = [
        refine_boundary_position(
            density, ruling_line_rows, y, search_radius,
            left_number_candidates=left_candidates_local,
            right_number_candidates=right_candidates_local,
        )
        for y in expected_boundaries_local
    ]
    positions_local = [pos for pos, _source in boundary_results]
    sources = [source for _pos, source in boundary_results]
    reconstructed_runs = 0  # telemetry: how many multi-row anomalous runs needed interpolation

    # Row-trust anomaly recovery (2026-08-05, per direct instruction - the
    # validation unit is the ROW, not the boundary): each row i (spanning
    # positions_local[i] to positions_local[i+1]) is trusted if its height
    # is plausible relative to the page's own median, untrusted otherwise.
    # An untrusted row's OWN measurement is discarded entirely, NOT
    # repaired with a weaker signal (ruling-line/density) - the row
    # immediately above and below have ALREADY established their own
    # trusted top/bottom, and those boundaries already define this row's
    # geometry. Concretely: boundary[i] already equals row(i-1)'s bottom,
    # and boundary[i+1] already equals row(i+1)'s top - if BOTH neighbors
    # are trusted, there is nothing to recompute (both endpoints are
    # already validated via the trusted neighbor that produced them), so
    # an isolated single anomalous row is left exactly as-is - its odd
    # height is accepted as real page variation, not an error to chase.
    #
    # Only a RUN of two or more CONSECUTIVE untrusted rows lacks this
    # direct inheritance on at least one interior boundary (confirmed as
    # the real failure shape on an actual page: one bad printed digit
    # corrupts the boundary on both sides of it via the center-alignment
    # midpoint, producing exactly two consecutive untrusted rows - one
    # too short, the next too long). For that case, the two OUTER
    # boundaries flanking the whole run (each already validated by a
    # trusted row on its far side, or table_top/table_bottom at the very
    # ends) are trustworthy anchors; the interior boundaries within the
    # run are reconstructed by simple even interpolation across that
    # confirmed span - a "secondary recovery method," per direct
    # instruction, not another attempt to re-detect the same row's own
    # (already-shown-unreliable) signal.
    if row_count > 1:
        heights = [positions_local[i + 1] - positions_local[i] for i in range(row_count)]
        sorted_heights = sorted(heights)
        median_height = sorted_heights[len(sorted_heights) // 2]
        anomaly_ratio = 0.20  # deviation beyond this fraction of the median is untrusted
        row_trusted = [
            median_height <= 0 or abs(h - median_height) <= anomaly_ratio * median_height
            for h in heights
        ]

        i = 0
        while i < row_count:
            if row_trusted[i]:
                i += 1
                continue
            run_start = i  # first untrusted row in this run
            while i < row_count and not row_trusted[i]:
                i += 1
            run_end = i - 1  # last untrusted row in this run (inclusive)
            run_length = run_end - run_start + 1

            if run_length == 1:
                # Isolated untrusted row with trusted (or page-edge)
                # neighbors on both sides - its boundaries are already
                # inherited from them. Nothing to change.
                continue

            # Multi-row run: reconstruct the (run_length - 1) INTERIOR
            # boundaries by even interpolation between the two outer,
            # already-trusted anchor boundaries.
            reconstructed_runs += 1
            left_anchor = positions_local[run_start]       # = trusted row above's bottom (or table_top)
            right_anchor = positions_local[run_end + 1]     # = trusted row below's top (or table_bottom)
            span = right_anchor - left_anchor
            for k in range(1, run_length):
                idx = run_start + k
                positions_local[idx] = left_anchor + round(span * k / run_length)
                sources[idx] = "interpolated_multi_row_run"

    refined_boundaries = [pos + data_top for pos in positions_local]
    boundary_sources = sources
    # Guarantee monotonic ordering - local refinement (ruling-line/density
    # path or number-column candidates alike) could in principle push two
    # adjacent boundaries out of order; enforce a minimum 1px separation
    # rather than let that silently produce a zero/negative-height band.
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
    ]
    if left_candidates_local or right_candidates_local:
        source_counts = {}
        for s in boundary_sources:
            source_counts[s] = source_counts.get(s, 0) + 1
        source_summary = ", ".join(f"{v} {k}" for k, v in source_counts.items())
        warnings.append(
            f"Row-number-column signal: {number_centers_found} digit-group(s) found "
            f"({len(left_candidates_local)} left, {len(right_candidates_local)} right), "
            f"alignment={row_number_alignment!r}. Per-boundary source breakdown: "
            f"{source_summary}."
        )
        n_interpolated = source_counts.get("interpolated_multi_row_run", 0)
        if n_interpolated:
            warnings.append(
                f"{n_interpolated} interior boundary(-ies) across {reconstructed_runs} "
                f"run(s) of 2+ consecutive rows with implausible heights (>20% off the "
                f"page's own median) were reconstructed by even interpolation between "
                f"the nearest trusted boundaries flanking each run, discarding those "
                f"rows' own unreliable measurements entirely rather than re-deriving "
                f"them from a weaker signal. Isolated single-row anomalies (trusted "
                f"neighbors on both sides) are left as-is - their boundaries are "
                f"already inherited from those neighbors, so an odd height there is "
                f"treated as real page variation, not an error."
            )
        n_disagree = source_counts.get("number_columns_disagree", 0)
        if n_disagree:
            warnings.append(
                f"{n_disagree} boundary(-ies) had left/right number-column candidates "
                f"that DISAGREED (beyond the agreement tolerance) - resolved by picking "
                f"whichever was closer to the periodic estimate, not silently averaged. "
                f"Worth a visual spot-check on this page."
            )
    warnings.append(
        f"{len(ruling_line_rows)} ruling-line row(s) available as a refinement "
        f"signal within local search windows "
        f"({'used where found' if len(ruling_line_rows) > 0 else 'none found - every boundary fell back to local density minimum'})."
    )

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
    elif sidecar.get("active_column") is None:
        # Self-heal (2026-07-22, found via a real sidecar): active_column
        # can go stale-None even with incomplete columns remaining - e.g.
        # every column reaches "done" (active_column correctly becomes
        # None), then one is reopened and re-saved via Save (not Next/
        # Extract), setting its status back to "in_progress" without
        # ever going through the "mark done" branch above that would
        # normally recompute active_column. Nothing else in this
        # function's contract depends on active_column staying None once
        # real incomplete work exists, so recomputing it here is always
        # safe - None while incomplete is never a meaningful state.
        sidecar["active_column"] = _first_incomplete(sidecar)

    save_sidecar(sidecar, path)
    return sidecar


def update_sidecar_preprocessing(path, preprocessing: dict) -> dict:
    """
    Writes `preprocessing` (see core.image_preprocessing.apply_pipeline's
    config shape) into the sidecar's top-level "preprocessing" key and
    saves atomically - a small page-level sibling to update_sidecar()
    above, which is column-scoped and doesn't fit this (preprocessing
    is a whole-page display/legibility choice, not tied to any one
    column's mask or extraction state).

    WHOLESALE replace, not a deep merge like update_sidecar() uses for
    column patches - ui/row_segmentation_ui.py always sends its full
    current checkbox/parameter state here, never a partial patch, so a
    merge would only risk leaving stale keys around from a filter that
    was since removed from the UI's own config shape.

    Never touches columns/column_order/active_column/progress or any
    geometry field (rows/table_bbox/etc.) - safe to call independently
    of whether "Refine rows" has ever been run for this image, so
    choosing/adjusting preprocessing doesn't require re-running
    detection first.
    """
    sidecar = load_sidecar(path)
    sidecar["preprocessing"] = preprocessing
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


def upscale_to_target_height(
    image: Image.Image,
    target_height: int = 160,
    max_width: int = 4096,
) -> Image.Image:
    """
    Upscales (aspect-preserving, LANCZOS) so the image reaches
    target_height - added 2026-07-23 after real evidence showed every
    model comparison run this session had a confound sitting underneath
    it that nobody had checked: after tight_crop_to_ranges() correctly
    solved the "mostly blank, full-row-width" problem, the resulting
    crops were still tiny in absolute pixel terms - real column crops
    from actual sidecars measured as small as 79x36 pixels (a "Sex"
    column). That's not enough resolution for most vision encoders to
    have real detail to work with regardless of how good the model is,
    meaning some of this session's "model X is worse than model Y"
    conclusions may partly reflect resolution starvation rather than a
    genuine capability difference between them.

    NEVER SHRINKS - only upscales when the crop is genuinely smaller
    than target_height; a crop that's already tall enough (or genuinely
    huge - an unmasked full row) is returned via ordinary shrink-to-fit
    against target_height instead, same convention as thumbnail().

    max_width caps runaway aspect ratios (a very wide, very short mask -
    upscaling a 3155x36 crop to reach height 160 would produce an
    ~14,000px-wide image) - most model processors would just resize
    that back down internally anyway, so generating it is wasted work
    and wasted disk/network if the crop gets saved. When the natural
    aspect-preserving upscale would exceed max_width, width is capped
    and height falls proportionally short of target_height rather than
    distorting the aspect ratio to hit both targets exactly.

    Deliberately does NOT do grayscale conversion, contrast adjustment,
    denoising, or thresholding, even though those were also proposed
    (see this session's 2026-07-23 discussion) - upscaling alone is a
    non-lossy, unambiguously safe transform; the others involve real
    tradeoffs (denoising/thresholding can erase faint pen strokes or
    merge letters together) that need actual comparative model results
    to justify, not applied speculatively as a new default. A caller
    that wants to test those as controlled variants should do so
    explicitly and record which variant was used (see extraction_meta's
    "preprocessing" field in run_single_column_extraction), not have
    them silently bundled into this function.
    """
    width, height = image.size
    if height <= 0 or width <= 0:
        return image
    scale = target_height / height
    if scale <= 1.0:
        # Already tall enough (or exactly at target) - no upscale
        # needed. Still respect max_width in case the source is huge.
        if width <= max_width:
            return image
        shrink = max_width / width
        return image.resize(
            (max_width, max(1, round(height * shrink))), Image.LANCZOS
        )

    new_width = round(width * scale)
    if new_width > max_width:
        scale = max_width / width
        new_width = max_width
    new_height = max(1, round(height * scale))
    return image.resize((new_width, new_height), Image.LANCZOS)


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
    upscale_target_height: int | None = None,
    upscale_max_width: int = 4096,
    debug_stage_callback=None,
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

    upscale_target_height (2026-07-23): applied LAST, after any
    tightening - see upscale_to_target_height()'s docstring for why
    (real column crops measured as small as 79x36 pixels even after
    tightening solved the mostly-blank problem; tightening a mostly-
    blank image and having enough actual resolution left to read are
    two separate problems). None (default) means no upscaling - exact
    prior behavior. This is a METHODOLOGY parameter, not a bugfix
    default like tight_crop_keep_ranges was: unlike tightening (which
    only ever removes blank margin, a strict improvement with no
    tradeoff), upscaling doesn't add real information and its effect on
    any given model's own internal preprocessing is untested per-model.
    Callers should set it deliberately when running a "fair bench"
    comparison and record having done so (see extraction_meta's
    "preprocessing" field in run_single_column_extraction) rather than
    silently changing what past results mean.

    IMPORTANT: this function's job is unchanged for every EXISTING
    caller that doesn't pass tight_crop_keep_ranges/upscale_target_height -
    both default to None/no-op, so no existing behavior changes from
    either parameter's addition. The row bbox geometry stored in the
    sidecar itself (sidecar["rows"]) is never touched by either - only
    the in-memory image this function RETURNS is affected; the
    sidecar's own coordinate records stay full-row/original-resolution,
    so nothing downstream that relies on those coordinates is affected.

    debug_stage_callback (2026-07-24, --debug-model-inputs support):
    optional callable(stage_name: str, image: Image.Image) -> None,
    invoked after each preprocessing stage actually runs. This is the
    single instrumented point every extraction path already funnels
    through, so debug capture doesn't need duplicating per loader/
    caller - see core/debug_dump.py (DebugItemRecorder.stage_callback())
    for the real implementation. Two stages matter for that feature:
    "bbox_crop" (right after the initial bbox crop, before any masking/
    tightening/upscaling - the "original crop before preprocessing")
    and "final" (the exact return value of this function, i.e. the
    exact image object handed to loader._run_generate() - the "model
    input" save point). None (default) means no callback - zero
    overhead/behavior change for every existing caller.
    """
    image = Image.open(source_image_path)
    if deskew_angle != 0.0:
        fill = (255, 255, 255) if image.mode == "RGB" else 255
        image = image.rotate(
            deskew_angle, expand=True, fillcolor=fill, resample=Image.BICUBIC
        )
    x0, y0, x1, y1 = bbox
    cropped = image.crop((x0, y0, x1, y1))
    if debug_stage_callback:
        debug_stage_callback("bbox_crop", cropped)
    if mask_ranges:
        cropped = apply_column_mask(cropped, mask_ranges, crop_x0=x0)
    if tight_crop_keep_ranges:
        cropped = tight_crop_to_ranges(
            cropped, tight_crop_keep_ranges, crop_x0=x0,
            padding_px=tight_crop_padding_px, padding_pct=tight_crop_padding_pct,
        )
    if upscale_target_height is not None:
        cropped = upscale_to_target_height(
            cropped, target_height=upscale_target_height, max_width=upscale_max_width,
        )
    if debug_stage_callback:
        debug_stage_callback("final", cropped)
    return cropped


def segment_rows(
    image: Image.Image,
    header_row_count: int = 1,
    padding: int = 4,
    padding_pct: float | None = None,
    padding_top: int | None = None,
    padding_bottom: int | None = None,
    table_top: int | None = None,
    table_bottom: int | None = None,
    table_left: int | None = None,
    table_right: int | None = None,
    metadata_bottom: int | None = None,
    header_box_top: int | None = None,
    header_box_bottom: int | None = None,
    deskew_angle: float | None = None,
) -> tuple[RowDetectionResult, list[Image.Image], Image.Image, Image.Image]:
    """
    Full pipeline through the agreed build order: deskew -> detect ->
    merge wrapped -> sanity check -> crop -> debug overlay. Does NOT
    call any OCR model - returns row crops for visual review only, per
    the agreed scope ("prove the segmentation layer before adding
    model-call complexity").

    table_top/table_bottom/table_left/table_right/metadata_bottom/
    header_box_top/header_box_bottom/deskew_angle (2026-08-05, same
    parameters and reasoning as segment_rows_periodic(); previously this
    function accepted NONE of these and ran detect_row_bands() blind
    against the WHOLE page - confirmed on a real 1901 census page that
    a header block's own print density/ruling lines pollute detection
    even when the table's own extent is exactly right). Detection is
    now scoped to the actual DATA-ROW region - header_box_bottom (or
    table_top if no header box is confirmed) down to table_bottom - the
    same ROI-local-then-shift-back approach segment_rows_periodic() uses.
    table_top/table_bottom fall back to estimate_table_extent() if not
    given, matching that function's own fallback.

    header_row_count: legacy meaning ("treat the first N detected bands
    as the header") no longer applies once detection is scoped to
    exclude the header entirely - header_box_top/header_box_bottom (or
    metadata_bottom+table_top) now define the header region explicitly,
    same mechanism segment_rows_periodic() already uses. If NONE of
    those are given, header_band stays None rather than guessing (safer
    than the old first-N-bands heuristic, which could silently consume
    a real data row as fake "header" - see segment_rows_periodic()'s own
    2026-07-13 comment about exactly that failure mode).

    Returns (result, row_crops, header_crop, debug_overlay_image) -
    the deskewed image is available via result if needed, but callers
    mainly want row_crops (to inspect) and debug_overlay_image (to see
    what was detected and why).
    """
    if deskew_angle is not None:
        deskewed = apply_deskew_angle(image, deskew_angle)
        angle = deskew_angle
    else:
        deskewed, angle = deskew(image)

    x0, x1 = table_left, table_right
    if table_top is None or table_bottom is None:
        auto_top, auto_bottom = estimate_table_extent(deskewed, x0=x0, x1=x1)
        table_top = table_top if table_top is not None else auto_top
        table_bottom = table_bottom if table_bottom is not None else auto_bottom

    data_top = header_box_bottom if header_box_bottom is not None else table_top
    roi = deskewed.crop((
        x0 or 0, data_top,
        x1 if x1 is not None else deskewed.width, table_bottom,
    ))

    raw_bands_local, ruling_line_count, ruling_line_rows_local = detect_row_bands(roi)
    merged_bands_local = merge_wrapped_bands(raw_bands_local, ruling_line_rows=ruling_line_rows_local)
    kept_bands_local, dropped_bands_local, warnings = sanity_check_bands(merged_bands_local)

    # Shift ROI-local bands back to full-image y-coordinates - the only
    # point in this function where the two coordinate spaces meet.
    kept_bands = [(a + data_top, b + data_top) for a, b in kept_bands_local]
    dropped_bands = [(a + data_top, b + data_top) for a, b in dropped_bands_local]

    warnings.insert(
        0,
        f"Deskew angle applied: {angle:.2f} degrees."
        + (" (user-confirmed, not auto-estimated)" if deskew_angle is not None else " (auto-estimated)"),
    )
    warnings.insert(
        1,
        f"Detected {ruling_line_count} ruling-line row(s) within the data region "
        f"({data_top}-{table_bottom}), excluded from content bands. "
        f"{'This confirms the ruling-line exclusion engaged - ' if ruling_line_count > 0 else 'Zero detected - check ruling_line_run_ratio if this page has a ruled table and rows still look merged. '}"
    )

    header_band = None
    data_bands = kept_bands
    if header_box_top is not None and header_box_bottom is not None:
        header_band = (header_box_top, header_box_bottom)
    elif header_row_count > 0 and metadata_bottom is not None:
        header_band = (metadata_bottom, table_top)

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
