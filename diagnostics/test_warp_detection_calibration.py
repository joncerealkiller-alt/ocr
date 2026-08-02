"""
Calibration script for core/warp_detection.py - measures its warp_score
against every real raw/dewarped image pair available in this project
(J:\\Screenshots\\Knott_Ancestry\\Knott\\*.jpg vs their data/outputs/
dewarped/*_dewarped.jpg counterparts), so the threshold in warp_detection
.py is set from real numbers, not a guess.

Both sides get deskewed first via core/row_segmentation.py's own
estimate_deskew_angle/apply_deskew_angle - detect_warp() assumes the
caller already did this (see its module docstring).

Usage:
    python diagnostics/test_warp_detection_calibration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from core.row_segmentation import estimate_deskew_angle, apply_deskew_angle
from core.warp_detection import detect_warp

RAW_DIR = Path(r"J:\Screenshots\Knott_Ancestry\Knott")
DEWARPED_DIR = PROJECT_ROOT / "data" / "outputs" / "dewarped"


def score_image(path: Path) -> str:
    image = Image.open(path).convert("RGB")
    angle = estimate_deskew_angle(image)
    deskewed = apply_deskew_angle(image, angle)
    result = detect_warp(deskewed)
    if result.low_confidence:
        return f"INCONCLUSIVE (top_lines={result.top_line_count}, bottom_lines={result.bottom_line_count})"
    return (f"score={result.warp_score:.3f}  needs_dewarp={result.needs_dewarp}  "
            f"(matched={result.matched_line_count} lines, deskew_angle={angle:+.2f})")


def main():
    pairs = []
    for dewarped_path in sorted(DEWARPED_DIR.glob("*_dewarped.jpg")):
        stem = dewarped_path.name.removesuffix("_dewarped.jpg")
        raw_path = RAW_DIR / f"{stem}.jpg"
        if raw_path.exists():
            pairs.append((stem, raw_path, dewarped_path))

    print(f"Found {len(pairs)} real raw/dewarped pairs.\n")
    for stem, raw_path, dewarped_path in pairs:
        print(f"=== {stem} ===")
        try:
            print(f"  RAW:      {score_image(raw_path)}")
        except Exception as e:
            print(f"  RAW:      ERROR {type(e).__name__}: {e}")
        try:
            print(f"  DEWARPED: {score_image(dewarped_path)}")
        except Exception as e:
            print(f"  DEWARPED: ERROR {type(e).__name__}: {e}")
        print()


if __name__ == "__main__":
    main()
