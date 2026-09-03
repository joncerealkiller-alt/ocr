"""
Line-detection-ONLY test: does denoise + CLAHE + vertical-morphology
preprocessing (Jon's suggested cv2 pipeline) help the EXISTING ruling-
line detector (core/auto_sidecar.py's _find_vertical_ruling_line(),
reused as-is) actually find real column-divider lines, independent of
whether the enhanced image would still be usable for OCR/extraction.

Direct follow-up to Round 4's open finding in docs/COLUMN_NUMBER_ANCHOR_
RESEARCH.md: on the 1901 sample (z000077117), Variant B's detector
correction never fired ONCE - every single measured position equalled
the raw projection, meaning _find_vertical_ruling_line() found no real
line in ANY of the ~10 search corridors tried on that page. Two
explanations were open: (a) the projected search corridors were already
off-target, or (b) this form's lines are genuinely too faint for the
detector's existing Otsu+run-length threshold to pick up. This script
isolates (b) directly: it searches CENTERED ON THE REAL, GROUND-TRUTH
boundary position (not a drifted projection), with a generous radius,
so a "not found" result here can only mean the line itself isn't
detectable at that location - not that the search window missed it.

Explicitly NOT judging output image quality for extraction (per direct
instruction: "this is not a profile for extraction quality") - the
enhanced/morphology-closed image is expected to look worse for a human
or an OCR model. The only thing measured is whether the vertical-run-
length signal the existing detector relies on gets stronger.

Read-only research script. Does NOT modify core/auto_sidecar.py.

Usage:
    python -m diagnostics.test_line_detection_enhancement
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from core.auto_sidecar import _find_vertical_ruling_line
from diagnostics.error_analysis.common import LEGACY_WORKING_IMAGES_DIR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEARCH_RADIUS = 15  # tight - centered on real ground truth, not hunting for it


@dataclass
class Sample:
    year: str
    label: str
    image_path: Path
    sidecar_path: Path
    table_top: int
    table_bottom: int


SAMPLES = [
    Sample(
        year="1901", label="z000077117",
        image_path=PROJECT_ROOT / "data/outputs/lac_pull_1901_batch1/raw_jpgs/z000077117.jpg",
        sidecar_path=PROJECT_ROOT / "data/outputs/row_segmentation/z000077117_sidecar.json",
        table_top=722, table_bottom=2564,
    ),
    Sample(
        year="1926", label="e001926997",
        image_path=LEGACY_WORKING_IMAGES_DIR / "e001926997.png",
        sidecar_path=PROJECT_ROOT / "data/outputs/reference_pipeline_prerefactor/row_segmentation/e001926997_sidecar.json",
        table_top=1408, table_bottom=4308,
    ),
    Sample(
        year="1931", label="e011717826",
        image_path=PROJECT_ROOT / "data/outputs/lac_pull_1931_batch1/raw_jpgs/e011717826.jpg",
        sidecar_path=PROJECT_ROOT / "data/outputs/row_segmentation/e011717826_sidecar.json",
        table_top=690, table_bottom=2400,
    ),
]


def load_real_boundaries(sidecar_path: Path) -> list[tuple[str, float]]:
    """Every real (non-empty) column's real RIGHT edge - the same
    boundary quantity Round 3/4's projection harness measured against."""
    data = json.loads(sidecar_path.read_text(encoding="utf-8"))
    out = []
    for name, col in data["columns"].items():
        ranges = col.get("mask_keep_ranges") or []
        if not ranges:
            continue
        x1 = max(r[1] for r in ranges)
        out.append((name, x1))
    return sorted(out, key=lambda t: t[1])


def find_vertical_line_via_gradient(
    gray_arr: np.ndarray, y0: int, y1: int, expected_x: int, search_radius: int,
    method: str = "scharr", min_peak_ratio: float = 1.5,
) -> tuple[int, bool, float]:
    """
    Jon's hypothesis: the existing detector looks for a DARK RIDGE
    (Otsu-threshold + longest-run-length of "this pixel is dark"),
    which assumes the line has enough absolute contrast to cross the
    threshold. A hairline-thin printed rule can fail that even where a
    real, sharp EDGE is present on both sides of it - polarity/contrast
    dependent vs. the gradient, which just needs a consistent brightness
    CHANGE and doesn't care how dark either side is.

    dx=1 Sobel/Scharr response summed down each column in the y-band,
    per column. found=True if the strongest column's average response
    beats the local background by min_peak_ratio - not just "is there
    any edge" (there's always some noise), but "is one column a clear
    outlier vs. its neighbors."
    """
    h, w = gray_arr.shape
    lo = max(0, expected_x - search_radius)
    hi = min(w, expected_x + search_radius + 1)
    if hi <= lo:
        return expected_x, False, 0.0
    crop = gray_arr[y0:y1, lo:hi].astype(np.float32)
    if crop.size == 0:
        return expected_x, False, 0.0

    if method == "sobel":
        grad = cv2.Sobel(crop, cv2.CV_32F, dx=1, dy=0, ksize=3)
    else:
        grad = cv2.Scharr(crop, cv2.CV_32F, dx=1, dy=0)
    profile = np.mean(np.abs(grad), axis=0)  # one value per x column in [lo, hi)

    peak_idx = int(np.argmax(profile))
    peak_val = profile[peak_idx]
    background = float(np.median(profile))
    ratio = peak_val / (background + 1e-6)
    found = ratio >= min_peak_ratio
    return lo + peak_idx, found, ratio


def find_vertical_line_via_projection_gradient_fusion(
    gray_arr: np.ndarray, y0: int, y1: int, expected_x: int, search_radius: int,
    scales: tuple[float, ...] = (0.0, 1.2),
    min_peak_ratio: float = 1.5, min_coverage: float = 0.5,
) -> tuple[int, bool, float, float]:
    """
    Round 8, direct response to Round 7's finding that the plain
    gradient detector (find_vertical_line_via_gradient()) fixed 1926/
    1931 but made 1901 WORSE - higher recall meant more chances to lock
    onto a real but WRONG line when the search corridor was already
    poorly centred (1901's low upstream column-number match rate).

    Jon's diagnosis: a plain per-column MEAN gradient magnitude can be
    fooled by a short, locally-strong response (a handwriting stroke, a
    digit, a stray mark) that only covers a few rows - it doesn't
    distinguish that from a real ruling line's response, which should
    be roughly uniform down the ENTIRE column height. Fix: require a
    candidate column to be a strong projection-profile peak (as before)
    AND have high COVERAGE - elevated response across most of the
    y-band, not just on average - a "is this continuous like a real
    line, or just tall in a few spots" check. Also runs Scharr at 2
    scales (raw + a mildly Gaussian-blurred copy) and takes the
    per-pixel max, since 1901's lines are thinner/lower-contrast and a
    slightly larger effective kernel can pick them up more cleanly
    without amplifying noise as much as a single fixed 3x3 kernel.
    Preceded by a light CLAHE pass (mild clip=1.5, not Round 5's
    stronger clip=2.5 - Round 5 already showed heavy contrast
    enhancement alone doesn't fix 1901, so this stays intentionally
    mild, just to help the gradient step, not to replace it).

    Returns (x_position, found, peak_ratio, coverage_at_peak).
    """
    h, w = gray_arr.shape
    lo = max(0, expected_x - search_radius)
    hi = min(w, expected_x + search_radius + 1)
    if hi <= lo:
        return expected_x, False, 0.0, 0.0
    crop_u8 = gray_arr[y0:y1, lo:hi]
    if crop_u8.size == 0:
        return expected_x, False, 0.0, 0.0

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    crop = clahe.apply(crop_u8).astype(np.float32)

    fused_mag = None
    for sigma in scales:
        band = crop if sigma <= 0 else cv2.GaussianBlur(crop, (0, 0), sigmaX=sigma)
        grad = np.abs(cv2.Scharr(band, cv2.CV_32F, dx=1, dy=0))
        fused_mag = grad if fused_mag is None else np.maximum(fused_mag, grad)

    profile = np.mean(fused_mag, axis=0)  # projection profile - one value per column
    threshold = float(np.median(fused_mag)) * 2.0
    coverage = np.mean(fused_mag > threshold, axis=0)  # fraction of rows "elevated" per column

    qualifying = coverage >= min_coverage
    if not np.any(qualifying):
        return expected_x, False, 0.0, float(np.max(coverage))

    masked_profile = np.where(qualifying, profile, -np.inf)
    peak_idx = int(np.argmax(masked_profile))
    peak_val = profile[peak_idx]
    background = float(np.median(profile))
    ratio = peak_val / (background + 1e-6)
    found = ratio >= min_peak_ratio
    return lo + peak_idx, found, ratio, float(coverage[peak_idx])


def enhance_for_line_detection(gray_arr: np.ndarray) -> np.ndarray:
    """Jon's suggested pipeline: mild denoise -> CLAHE -> vertical-line
    morphological closing. Deliberately kept close to the pasted script
    rather than re-tuned, so this is a fair test of the actual proposal.
    Vertical-only (not the horizontal+combine step) - this test only
    cares about VERTICAL column-divider lines."""
    denoised = cv2.fastNlMeansDenoising(gray_arr, None, h=8, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(16, 16))
    enhanced = clahe.apply(denoised)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    vertical = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, vertical_kernel, iterations=1)
    return vertical


def run_sample(sample: Sample) -> None:
    print(f"\n{'=' * 70}\n{sample.year} ({sample.label})\n{'=' * 70}")
    if not sample.image_path.exists():
        print(f"  SKIPPED - image not found: {sample.image_path}")
        return

    real_cols = load_real_boundaries(sample.sidecar_path)
    print(f"{len(real_cols)} real ground-truth boundaries to check")

    pil_image = Image.open(sample.image_path)
    gray = np.array(pil_image.convert("L"))

    # Scope the cv2 pipeline to the table's own height band - full-page
    # denoising on a 7000px-wide 1926 scan is both unnecessary (we only
    # ever search within the table rows) and slow.
    y0, y1 = sample.table_top, sample.table_bottom
    enhanced_band = enhance_for_line_detection(gray[y0:y1, :])
    # _find_vertical_ruling_line() expects a full-height image it can
    # itself slice by y0:y1 - rebuild a full-height array with the
    # enhanced band spliced in at the right offset so the same call
    # signature works unmodified.
    enhanced_full = gray.copy()
    enhanced_full[y0:y1, :] = enhanced_band
    enhanced_image = Image.fromarray(enhanced_full, mode="L")

    raw_found, raw_deltas = 0, []
    enh_found, enh_deltas = 0, []
    grad_found, grad_deltas, grad_ratios = 0, [], []
    fused_found, fused_deltas = 0, []
    for name, real_x in real_cols:
        rx, rfound = _find_vertical_ruling_line(pil_image, y0, y1, int(round(real_x)), SEARCH_RADIUS)
        ex, efound = _find_vertical_ruling_line(enhanced_image, y0, y1, int(round(real_x)), SEARCH_RADIUS)
        gx, gfound, gratio = find_vertical_line_via_gradient(
            gray, y0, y1, int(round(real_x)), SEARCH_RADIUS,
        )
        fx, ffound, fratio, fcov = find_vertical_line_via_projection_gradient_fusion(
            gray, y0, y1, int(round(real_x)), SEARCH_RADIUS,
        )
        if rfound:
            raw_found += 1
            raw_deltas.append(abs(rx - real_x))
        if efound:
            enh_found += 1
            enh_deltas.append(abs(ex - real_x))
        if gfound:
            grad_found += 1
            grad_deltas.append(abs(gx - real_x))
        grad_ratios.append(gratio)
        if ffound:
            fused_found += 1
            fused_deltas.append(abs(fx - real_x))
        marker = ""
        if gfound and not rfound:
            marker = "  <-- gradient found what raw/enh missed"
        print(f"  {name:<40} raw={'FOUND' if rfound else 'miss':<5} "
              f"enh={'FOUND' if efound else 'miss':<5} "
              f"grad={'FOUND' if gfound else 'miss':<5} (ratio={gratio:.2f}) "
              f"fused={'FOUND' if ffound else 'miss':<5} (ratio={fratio:.2f} cov={fcov:.2f}){marker}")

    n = len(real_cols)
    print(f"\nRaw:      {raw_found}/{n} found "
          f"({100*raw_found/n:.0f}%)" + (f", mean delta {sum(raw_deltas)/len(raw_deltas):.1f}px" if raw_deltas else ""))
    print(f"Enhanced: {enh_found}/{n} found "
          f"({100*enh_found/n:.0f}%)" + (f", mean delta {sum(enh_deltas)/len(enh_deltas):.1f}px" if enh_deltas else ""))
    print(f"Gradient: {grad_found}/{n} found "
          f"({100*grad_found/n:.0f}%)" + (f", mean delta {sum(grad_deltas)/len(grad_deltas):.1f}px" if grad_deltas else "")
          + f", mean ratio {sum(grad_ratios)/len(grad_ratios):.2f}")
    print(f"Fused:    {fused_found}/{n} found "
          f"({100*fused_found/n:.0f}%)" + (f", mean delta {sum(fused_deltas)/len(fused_deltas):.1f}px" if fused_deltas else ""))


def main() -> None:
    for s in SAMPLES:
        run_sample(s)


if __name__ == "__main__":
    main()
