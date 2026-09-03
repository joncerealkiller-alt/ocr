"""
Runs the REAL production classifier (build_classifier_loader(), unmodified
prompt - full 11-category taxonomy + uncertain_review, not the flat-8
research prompt other diagnostics/ scripts use) over every image in an
external folder, writing a CSV in the exact schema core/classifier.py's
real bucket CSVs use (CSV_FIELDS) so the output is directly loadable by
ui/classifier_validation_ui.py.

Deliberately standalone from core/manifest_pipeline.py / core/pipeline_db.py -
never registers anything in the pipeline DB, never touches data/buckets/,
data/manifest.csv, or genealogy_workspace/ in any way. The source images
are read in place and never copied/moved. Built for reviewing personal/
external material that must NOT enter the production corpus (2026-08-08,
Jon's explicit constraint) while still reusing the real classifier and a
familiar review UI.

Usage:
    python diagnostics/classify_external_folder.py \
        "J:\\Screenshots\\Kemper_Ancestry" \
        --out data/outputs/kemper_ancestry_review.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from PIL import Image

from core.classifier import build_classifier_loader, load_pipeline_config, CSV_FIELDS

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="External folder to classify (read-only, never modified).")
    parser.add_argument("--out", required=True, help="Output CSV path.")
    parser.add_argument("--recursive", action="store_true", default=True,
                         help="Search subfolders too (default: on).")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only classify the first N images (for a quick smoke test).")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        raise FileNotFoundError(f"{folder} does not exist.")

    images = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if args.limit:
        images = images[:args.limit]
    print(f"Found {len(images)} images under {folder}\n")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pipeline_cfg = load_pipeline_config()
    print(f"Loading Gemma via the REAL production build_classifier_loader() path, "
          f"unmodified prompt (full taxonomy, not flat-8)...")
    loader = build_classifier_loader(pipeline_cfg, debug=False)
    print("Loaded.\n")

    rows = []
    start = time.perf_counter()
    for i, img_path in enumerate(images, 1):
        try:
            img = Image.open(img_path).convert("RGB")
            result = loader.classify(str(img_path), img)
            row = {
                "file_path": str(img_path),
                "category": result.category.value,
                "confidence": result.confidence,
                "text_density": result.text_density,
                "handwriting": result.handwriting,
                "table_layout": result.table_layout,
                "faces": result.faces,
                "map_like": result.map_like,
                "reason": result.reason,
                "model": result.model,
                "prompt_version": result.prompt_version,
            }
            print(f"[{i}/{len(images)}] {result.category.value:<20s} conf={result.confidence:.2f}  {img_path.name}")
        except Exception as e:
            row = {"file_path": str(img_path), "category": "", "confidence": "",
                   "text_density": "", "handwriting": "", "table_layout": "", "faces": "",
                   "map_like": "", "reason": f"CLASSIFY FAILED: {type(e).__name__}: {e}",
                   "model": "", "prompt_version": ""}
            print(f"[{i}/{len(images)}] FAILED {img_path.name}: {type(e).__name__}: {e}")
        rows.append(row)

        if i % 25 == 0 or i == len(images):
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows(rows)

    elapsed = time.perf_counter() - start
    print(f"\nDone. {len(rows)} images classified in {elapsed:.1f}s ({elapsed/len(rows):.2f}s/image).")
    print(f"Output written to {out_path}")


if __name__ == "__main__":
    main()
