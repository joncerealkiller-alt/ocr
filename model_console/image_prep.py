"""
Prepares an arbitrary picked/dropped image file for a chat turn.

Deliberately NOT core/image_analysis.py's job - that module does
page-boundary/deskew analysis for scanned documents going through the
extraction pipeline, a different problem. This is a basic "don't hand
a 6000px scan straight to a VLM's processor" safety net for ad-hoc
chat images: normalize color mode, downscale only if oversized,
nothing else.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image

from model_console.session import ChatImage

DEFAULT_MAX_DIMENSION = 1568


def prep_for_chat(path: Path, max_dimension: int = DEFAULT_MAX_DIMENSION) -> ChatImage:
    """
    Opens `path`, converts to RGB (source files may be palette/CMYK/
    RGBA - a VLM processor expects RGB), downscales preserving aspect
    ratio only if either dimension exceeds max_dimension. Returns a
    ChatImage with sha256 already computed for log correlation and
    "same image already attached" detection in the UI.

    `max_dimension` should be picked by the caller from the target
    model's own config/models/<name>.yaml token-budget fields where
    present (e.g. max_pixels/max_image_tokens), falling back to
    DEFAULT_MAX_DIMENSION otherwise - so an unusually token-constrained
    model doesn't silently get truncated by the loader after this
    function already spent time preparing a full-resolution image.
    """
    path = Path(path)
    with Image.open(path) as opened:
        image = opened.convert("RGB")

    if max(image.width, image.height) > max_dimension:
        scale = max_dimension / max(image.width, image.height)
        new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(new_size, Image.LANCZOS)

    return ChatImage.from_pil(image, display_name=path.name, source_path=path)


def max_dimension_for_model(config, fallback: int = DEFAULT_MAX_DIMENSION) -> int:
    """
    Reads a token/pixel budget hint off a GenerationConfig if the
    model's YAML declares one, else returns `fallback`. Checked in this
    order because different loaders use different fields for this
    (max_pixels is the most common; max_image_tokens is set instead by
    a couple of loaders per config/models/*.yaml).
    """
    max_pixels = getattr(config, "max_pixels", None)
    if max_pixels:
        # max_pixels bounds total pixel count, not a single dimension -
        # take its square root as a rough per-side ceiling rather than
        # leaving max_dimension unbounded.
        return max(1, int(max_pixels ** 0.5))
    max_image_tokens = getattr(config, "max_image_tokens", None)
    if max_image_tokens:
        return fallback
    return fallback
