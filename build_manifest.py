"""
Minimal manifest builder for smoke-testing the classifier stage.

Usage:
    python build_manifest.py

Opens a native folder picker (tkinter, no extra deps). Walks the
selected folder for image files and writes data/manifest.csv with
one file_path per row - the exact input shape core/classifier.py
expects.

This is a standalone convenience script, not part of the UI proper.
Once the pipeline is validated, this logic should move into ui/app.py
as an "Add Files" button rather than staying a separate script.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from tkinter import Tk, filedialog

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}

PROJECT_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.csv"


def pick_folder() -> Path | None:
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(title="Select folder of images to classify")
    root.destroy()
    return Path(folder) if folder else None


def collect_image_paths(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def write_manifest(paths: list[Path]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_path"])
        for p in paths:
            writer.writerow([str(p)])


def main() -> None:
    folder = pick_folder()
    if folder is None:
        print("No folder selected. Exiting.")
        sys.exit(0)

    if not folder.exists():
        print(f"Folder does not exist: {folder}")
        sys.exit(1)

    paths = collect_image_paths(folder)
    if not paths:
        print(f"No image files found in {folder} (looked for {sorted(IMAGE_EXTENSIONS)})")
        sys.exit(1)

    write_manifest(paths)
    print(f"Found {len(paths)} image(s) in {folder}")
    print(f"Manifest written to {MANIFEST_PATH}")
    print("\nNext step:")
    print(f"  python -m core.classifier {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
