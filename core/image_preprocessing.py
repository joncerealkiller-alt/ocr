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
