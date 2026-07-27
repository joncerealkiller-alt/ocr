"""
Image preprocessing for the assessment tool - test-only, applied to a
copy of the source image, never the original file on disk.

Built 2026-07-11 after Florence-2's <OCR_WITH_REGION> test showed the
worst-performing content all session was dense small print (the
registrar stamp read as "REGISTER CEMTAL"/"REGREATIAR CEMPAL" instead
of "Registrar-General") - worth testing whether that's an image-quality
ceiling separate from model capability or prompting, before concluding
anything more from further model/prompt iteration on the same
unmodified source image.

Every function here takes a PIL Image and returns a NEW PIL Image -
none of them mutate the input in place. Steps are independent and
toggleable via a named PREPROCESSING_PROFILES dict rather than one
fixed pipeline, since a blind "clean everything" transform could easily
help one kind of content (small stamped text) while hurting another
(handwriting) on the same image - consistent with this project's
practice of testing changes independently rather than bundling them and
losing the ability to tell what actually helped.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def enhance_contrast(image: Image.Image, factor: float = 1.5) -> Image.Image:
    """factor > 1.0 increases contrast. Returns a new image."""
    return ImageEnhance.Contrast(image).enhance(factor)


def sharpen(image: Image.Image, factor: float = 2.0) -> Image.Image:
    """factor > 1.0 increases sharpness. Returns a new image."""
    return ImageEnhance.Sharpness(image).enhance(factor)


def denoise(image: Image.Image, radius: float = 1.0) -> Image.Image:
    """
    Mild smoothing to reduce scan noise/grain before other steps -
    kept gentle (small default radius) since aggressive denoising can
    blur exactly the fine handwriting detail we're trying to preserve.
    """
    return image.filter(ImageFilter.MedianFilter(size=3))


def grayscale(image: Image.Image) -> Image.Image:
    """
    Converts to grayscale, then back to RGB (3-channel) so the output
    stays compatible with every loader's RGB expectation - the visual
    content is grayscale, but the tensor shape models expect is
    unchanged. Do not skip the RGB conversion step.
    """
    return ImageOps.grayscale(image).convert("RGB")


def upscale(image: Image.Image, factor: float = 2.0) -> Image.Image:
    """
    Upscales using LANCZOS resampling (best quality for upscaling,
    slower than nearest/bilinear but this is a one-off preprocessing
    step, not a hot loop). Motivated by SmolVLM2's confirmed ~1536px
    effective ceiling and Florence-2's smaller native tile size -
    upscaling a small source image before it reaches the model's own
    internal downsampling may preserve more fine detail than letting
    each model's own (varying, sometimes low) native resolution handle
    it cold.
    """
    new_size = (int(image.width * factor), int(image.height * factor))
    return image.resize(new_size, Image.LANCZOS)


def autocontrast(image: Image.Image, cutoff: float = 1.0) -> Image.Image:
    """
    Stretches the histogram so the darkest/lightest pixels hit true
    black/white, clipping `cutoff` percent from each end to avoid
    outlier pixels (dust specks, scan artifacts) dominating the
    stretch. Different mechanism from enhance_contrast (which scales
    around the existing midpoint) - autocontrast can help faded scans
    where the whole image sits in a narrow gray band.
    """
    return ImageOps.autocontrast(image, cutoff=cutoff)


# Named, independently-testable combinations. "none" is an explicit
# identity profile (not just the absence of a selection) so it shows up
# in the same dropdown/report trail as every other profile - the
# assessment tool's evidence-tracking convention throughout this
# project is to make "nothing was applied" a visible, logged choice,
# not an implicit default that's easy to lose track of.
def invert(image: Image.Image) -> Image.Image:
    """
    Inverts light/dark - added 2026-07-13 per Jon's direct observation
    from real forms: helps with faded text on some census pages. Worth
    noting the mechanism, since it's not simply "more contrast": plain
    inversion swaps which side is dark vs. light but doesn't by itself
    widen the tonal GAP between ink and background - if the real
    problem is low contrast (ink and background close in tone), plain
    inversion alone won't fix that, only relocate it. Where this is
    most likely to genuinely help: scans that are effectively
    photographic negatives (plausible for microfilm-sourced images -
    light ink on a dark/aged background rather than the far more common
    dark-on-light pattern most OCR/VLM training data assumes), or
    combined with contrast adjustment afterward (see invert_contrast
    profile below) rather than relied on alone.
    """
    return ImageOps.invert(image.convert("RGB"))


def median(image: Image.Image, size: int = 3) -> Image.Image:
    """
    Same operation as denoise() above, exposed under its own name for
    ui/row_segmentation_ui.py's checkbox-based preprocessing pipeline
    (2026-07-25) - "Median" is its own independent toggle there, not
    folded into one fixed "denoise" step. size must be odd (PIL's own
    MedianFilter requirement); even values are bumped up by 1 rather
    than raising, since a live-preview checkbox shouldn't throw over a
    parameter a user could reasonably type as an even number.
    """
    if size % 2 == 0:
        size += 1
    return image.filter(ImageFilter.MedianFilter(size=size))


def unsharp_mask(image: Image.Image, radius: float = 1.5, amount: float = 1.2) -> Image.Image:
    """
    PIL's own UnsharpMask filter - a different mechanism from sharpen()
    above (ImageEnhance.Sharpness, a global sharpness scalar): this
    sharpens edges specifically by subtracting a blurred copy of the
    image from itself (the classic "unsharp mask" darkroom technique),
    which tends to look less artificial on scanned handwriting than a
    flat global sharpness boost. amount maps to PIL's percent parameter
    (100 = PIL's own default strength; threshold left at PIL's default
    of 3 - only differences above that are sharpened, so flat
    background regions don't pick up sharpening noise).
    """
    percent = max(1, round(amount * 100))
    return image.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=3))


def clahe(image: Image.Image, clip_limit: float = 2.0, tile_size: int = 8) -> Image.Image:
    """
    Contrast Limited Adaptive Histogram Equalization, implemented in
    plain numpy - this project is deliberately dependency-free (PIL +
    numpy only, no OpenCV; see core/row_segmentation.py's manually
    implemented Otsu threshold for the same discipline applied
    elsewhere in this codebase), and CLAHE is conventionally an OpenCV
    (`cv2.createCLAHE`) call, so it's reimplemented here rather than
    adding OpenCV as a new dependency for one function.

    Operates on LUMINANCE only (converts to grayscale first, same
    RGB-compatibility handling as grayscale() above) - not per-channel
    color CLAHE. Unlike autocontrast()/enhance_contrast() above (both
    GLOBAL - one stretch/scale for the whole image), this equalizes
    contrast LOCALLY per tile, which is what actually helps a scan
    where different regions of the same page sit at different
    background tones (e.g. a shadow or aging gradient across the
    page) - a global stretch can't fix that, since it's driven by the
    same single histogram everywhere.

    tile_size is a GRID COUNT (e.g. 8 = an 8x8 grid of tiles spanning
    the whole image), matching OpenCV's `tileGridSize` convention most
    users encountering CLAHE elsewhere will already expect - NOT a
    pixel dimension.

    Standard algorithm, not a shortcut: each tile's own 256-bin
    histogram is clipped at `clip_limit * tile_pixel_count / 256`
    (excess redistributed uniformly across all bins, preventing the
    "hard clip and discard" look) then converted to a per-tile
    cumulative-distribution mapping; each pixel's final value is a
    BILINEAR interpolation between its 4 nearest tiles' mappings, not
    just its own tile's mapping alone - interpolation is what prevents
    visible blocking artifacts at tile boundaries (a per-tile-only
    mapping is windowed histogram equalization, not real CLAHE).
    Vectorized with numpy fancy indexing rather than a per-pixel Python
    loop, since these are full page scans (thousands of pixels wide).
    """
    gray_img = ImageOps.grayscale(image.convert("RGB"))
    gray = np.array(gray_img)
    h, w = gray.shape
    ty = tx = max(1, int(tile_size))

    y_edges = np.linspace(0, h, ty + 1).astype(int)
    x_edges = np.linspace(0, w, tx + 1).astype(int)

    luts = np.zeros((ty, tx, 256))
    for j in range(ty):
        for i in range(tx):
            tile = gray[y_edges[j]:y_edges[j + 1], x_edges[i]:x_edges[i + 1]]
            hist, _ = np.histogram(tile, bins=256, range=(0, 256))
            hist = hist.astype(np.float64)
            if clip_limit > 0 and tile.size > 0:
                clip = max(1.0, clip_limit * tile.size / 256.0)
                excess = np.clip(hist - clip, 0, None).sum()
                hist = np.minimum(hist, clip)
                hist = hist + excess / 256.0
            cdf = np.cumsum(hist)
            if cdf[-1] > 0:
                cdf = cdf / cdf[-1] * 255.0
            luts[j, i] = cdf

    centers_y = (y_edges[:-1] + y_edges[1:]) / 2.0
    centers_x = (x_edges[:-1] + x_edges[1:]) / 2.0

    j_pos = np.interp(np.arange(h), centers_y, np.arange(ty))
    i_pos = np.interp(np.arange(w), centers_x, np.arange(tx))
    j0 = np.clip(np.floor(j_pos).astype(int), 0, ty - 1)
    j1 = np.clip(j0 + 1, 0, ty - 1)
    i0 = np.clip(np.floor(i_pos).astype(int), 0, tx - 1)
    i1 = np.clip(i0 + 1, 0, tx - 1)
    wj = np.clip(j_pos - j0, 0, 1)
    wi = np.clip(i_pos - i0, 0, 1)

    J0, I0 = np.meshgrid(j0, i0, indexing="ij")
    J1, I1 = np.meshgrid(j1, i1, indexing="ij")
    WJ, WI = np.meshgrid(wj, wi, indexing="ij")

    top = luts[J0, I0, gray] * (1 - WI) + luts[J0, I1, gray] * WI
    bottom = luts[J1, I0, gray] * (1 - WI) + luts[J1, I1, gray] * WI
    result = np.clip(top * (1 - WJ) + bottom * WJ, 0, 255).astype(np.uint8)

    return Image.fromarray(result).convert("RGB")


def adaptive_threshold(image: Image.Image, block_size: int = 11, C: float = 2.0) -> Image.Image:
    """
    Local (not global) binarization - each pixel is compared against
    the mean of its OWN neighborhood rather than one fixed threshold
    for the whole page. Different use case from core/row_segmentation.
    py's global Otsu threshold (built for row-band detection): this is
    a display/legibility transform for uneven lighting across a single
    scan (a shadow or vignette means one fixed global threshold can't
    be right everywhere at once - one region ends up too dark, another
    washed out, at the same threshold value).

    Local mean computed via PIL's own BoxBlur (fast, C-implemented) -
    keeps this dependency-free (PIL + numpy only, no OpenCV, no manual
    sliding-window Python loop over every pixel). block_size must be
    odd (matches OpenCV's adaptiveThreshold convention most users will
    already expect); even values are bumped up by 1 rather than
    raising, same reasoning as median() above.

    Formula matches OpenCV's ADAPTIVE_THRESH_MEAN_C exactly: output is
    255 where the source pixel exceeds (local mean - C), else 0.
    """
    if block_size % 2 == 0:
        block_size += 1
    gray_img = ImageOps.grayscale(image.convert("RGB"))
    local_mean_img = gray_img.filter(ImageFilter.BoxBlur(block_size // 2))
    gray = np.array(gray_img, dtype=np.float64)
    local_mean = np.array(local_mean_img, dtype=np.float64)
    binary = np.where(gray > (local_mean - C), 255, 0).astype(np.uint8)
    return Image.fromarray(binary).convert("RGB")


# Fixed internal execution order for apply_pipeline() below - checkboxes
# in ui/row_segmentation_ui.py's Preprocessing section only ever
# enable/disable a step, never reorder it. Deliberate ordering (Jon's
# direction, 2026-07-25): grayscale/invert establish the tonal baseline
# before anything contrast- or edge-based runs; CLAHE (local contrast)
# before noise reduction/sharpening so those don't operate on the
# unequalized original; sharpening before upscale so edge enhancement
# isn't itself upscaled and softened again; upscale before threshold so
# binarization happens at final resolution, not on a source that's
# about to be resized out from under it. Deskew is NOT a step here - it
# already happens upstream, once, via apply_deskew_angle() before this
# pipeline ever runs, since it's a geometric correction the rest of
# this UI already owns, not a display/legibility filter.
PIPELINE_ORDER = [
    "grayscale", "invert", "clahe", "median", "unsharp", "upscale", "adaptive_threshold",
]

_PIPELINE_FUNCTIONS = {
    "grayscale": lambda img, cfg: grayscale(img),
    "invert": lambda img, cfg: invert(img),
    "clahe": lambda img, cfg: clahe(
        img, clip_limit=cfg.get("clip_limit", 2.0), tile_size=cfg.get("tile_size", 8)),
    "median": lambda img, cfg: median(img, size=cfg.get("size", 3)),
    "unsharp": lambda img, cfg: unsharp_mask(
        img, radius=cfg.get("radius", 1.5), amount=cfg.get("amount", 1.2)),
    "upscale": lambda img, cfg: upscale(img, factor=cfg.get("factor", 2.0)),
    "adaptive_threshold": lambda img, cfg: adaptive_threshold(
        img, block_size=cfg.get("block_size", 11), C=cfg.get("C", 2.0)),
}


def apply_pipeline(image: Image.Image, config: dict) -> Image.Image:
    """
    Applies whichever steps in PIPELINE_ORDER are enabled, in that
    FIXED order, regardless of what order they appear in `config` -
    the checkbox UI only controls enabled/disabled per step, never
    sequencing, per PIPELINE_ORDER's own docstring above (accidentally
    running e.g. threshold before CLAHE would binarize the image before
    contrast equalization ever saw real grayscale data to work with).

    config shape: {"grayscale": {"enabled": bool}, "clahe": {"enabled":
    bool, "clip_limit": float, "tile_size": int}, ...} - one entry per
    PIPELINE_ORDER name, each with at least "enabled"; missing entries
    or missing param keys fall back to the defaults baked into
    _PIPELINE_FUNCTIONS above, so a caller only needs to send the
    fields that differ from default. Same never-mutates-input,
    always-returns-a-new-image contract as apply_profile() above.
    """
    result = image.copy()
    if result.mode != "RGB":
        result = result.convert("RGB")
    for step_name in PIPELINE_ORDER:
        step_cfg = config.get(step_name) or {}
        if not step_cfg.get("enabled"):
            continue
        result = _PIPELINE_FUNCTIONS[step_name](result, step_cfg)
    return result


PREPROCESSING_PROFILES = {
    "none": [],
    "contrast_boost": [("enhance_contrast", {"factor": 1.5})],
    "autocontrast": [("autocontrast", {"cutoff": 1.0})],
    "sharpen_only": [("sharpen", {"factor": 2.0})],
    "grayscale_sharp": [("grayscale", {}), ("sharpen", {"factor": 2.0})],
    "upscale_2x": [("upscale", {"factor": 2.0})],
    "upscale_contrast_sharpen": [
        ("upscale", {"factor": 2.0}),
        ("autocontrast", {"cutoff": 1.0}),
        ("sharpen", {"factor": 1.5}),
    ],
    "denoise_contrast": [
        ("denoise", {}),
        ("autocontrast", {"cutoff": 1.0}),
    ],
    "invert_only": [("invert", {})],
    "invert_contrast": [("invert", {}), ("autocontrast", {"cutoff": 1.0})],
    "invert_contrast_sharpen": [
        ("invert", {}),
        ("autocontrast", {"cutoff": 1.0}),
        ("sharpen", {"factor": 1.5}),
    ],
}

_STEP_FUNCTIONS = {
    "enhance_contrast": enhance_contrast,
    "sharpen": sharpen,
    "denoise": denoise,
    "grayscale": grayscale,
    "upscale": upscale,
    "autocontrast": autocontrast,
    "invert": invert,
}


def apply_profile(image: Image.Image, profile_name: str) -> Image.Image:
    """
    Applies a named profile's steps in sequence, each step's output
    feeding the next. Returns a NEW image - the input is never mutated,
    and callers should treat the input as still-original afterward.
    Unknown profile name raises rather than silently falling back to
    "none", since a typo'd profile name should be visible, not silently
    a no-op.
    """
    if profile_name not in PREPROCESSING_PROFILES:
        raise ValueError(
            f"Unknown preprocessing profile {profile_name!r}. "
            f"Available: {list(PREPROCESSING_PROFILES.keys())}"
        )
    result = image.copy()  # defensive - ensure we never touch the caller's image
    if result.mode != "RGB":
        # Source PNGs are frequently RGBA (alpha channel) or palette
        # mode - some steps (confirmed: ImageOps.autocontrast on this
        # Pillow version) don't support RGBA and raise "not supported
        # for mode RGBA". Every loader already converts to RGB
        # defensively right before generation, but that happens AFTER
        # preprocessing in the current pipeline - converting here too
        # means preprocessing never depends on what mode the source
        # file happened to be saved in.
        result = result.convert("RGB")
    for step_name, kwargs in PREPROCESSING_PROFILES[profile_name]:
        result = _STEP_FUNCTIONS[step_name](result, **kwargs)
    return result
