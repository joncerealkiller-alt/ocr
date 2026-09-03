"""
Visual side-by-side comparison of every named preprocessing profile in
core/image_preprocessing.py's PREPROCESSING_PROFILES, against one or
more real images - built to help pick a better default than the
currently-hardcoded DEFAULT_PREPROCESSING_PROFILE ("autocontrast")
before scaling preprocessing across new census years.

Read-only against the source image. Writes:
  - one full-resolution output per profile (so detail can be zoomed into)
  - one labeled contact-sheet grid per source image, downscaled for
    quick side-by-side comparison

Output layout: data/outputs/preprocessing_profile_comparison/<stem>/
  full/<profile_name>.png   (full resolution)
  grid.png                  (all profiles, labeled, one glance)

Usage:
    python -m training.preprocessing_profile_comparison "data/outputs/lac_pull_1901_batch1/raw_jpgs/z000077117.jpg"
    python -m training.preprocessing_profile_comparison path1.jpg path2.jpg ...
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from core.image_preprocessing import PREPROCESSING_PROFILES, apply_profile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "outputs" / "preprocessing_profile_comparison"

GRID_PANEL_WIDTH = 420  # per-profile thumbnail width in the contact sheet
LABEL_HEIGHT = 32


def _label_font():
    try:
        return ImageFont.truetype("arial.ttf", 20)
    except Exception:
        return ImageFont.load_default()


def compare_one(image_path: Path) -> Path:
    stem = image_path.stem
    out_dir = OUT_DIR / stem
    full_dir = out_dir / "full"
    full_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as original:
        source = original.convert("RGB")

    font = _label_font()
    profile_names = list(PREPROCESSING_PROFILES.keys())
    panels = []

    for name in profile_names:
        processed = apply_profile(source, name)
        processed.save(full_dir / f"{name}.png")

        scale = GRID_PANEL_WIDTH / processed.width
        thumb = processed.resize((GRID_PANEL_WIDTH, int(processed.height * scale)))
        panel = Image.new("RGB", (GRID_PANEL_WIDTH, thumb.height + LABEL_HEIGHT), "white")
        draw = ImageDraw.Draw(panel)
        draw.rectangle([0, 0, GRID_PANEL_WIDTH, LABEL_HEIGHT], fill="#264653")
        draw.text((8, 6), name, fill="white", font=font)
        panel.paste(thumb, (0, LABEL_HEIGHT))
        panels.append(panel)

    n_cols = 4
    n_rows = -(-len(panels) // n_cols)  # ceil
    panel_h = max(p.height for p in panels)
    grid = Image.new("RGB", (GRID_PANEL_WIDTH * n_cols, panel_h * n_rows), "black")
    for i, panel in enumerate(panels):
        row, col = divmod(i, n_cols)
        grid.paste(panel, (col * GRID_PANEL_WIDTH, row * panel_h))

    grid_path = out_dir / "grid.png"
    grid.save(grid_path)
    return grid_path


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    for arg in sys.argv[1:]:
        image_path = Path(arg)
        if not image_path.exists():
            print(f"SKIP (not found): {image_path}")
            continue
        grid_path = compare_one(image_path)
        print(f"{image_path.name}: grid -> {grid_path}")


if __name__ == "__main__":
    main()
