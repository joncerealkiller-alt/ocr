"""
Generates fresh auto_sidecar overlays for every real image in
lac_pull_batch1 (the 1911 census reference batch - 92 images once split
out of Schedule_1_3a9fae1f, see column_calibration_workspace's own
folder for the working copies these were calibrated against) against the
CURRENT canada_census_1911.yaml template.

Built 2026-08-10 so Jon can eyeball table_top placement across the full
batch himself, after two real, confirmed misses (e001946628,
e001946629 - table_top landing on the column-number heading row instead
of the true rule beneath it) and a THIRD, unrelated pre-existing miss
(e001946707 - table_top skipping past the rule onto row 1's own
separator line) turned up during spot-checks. A code fix was attempted
and reverted the same day (see core/auto_sidecar.py's
_locate_rule_bottom_edge() docstring for the regression this caused on
e001946712/e001946719/e001946642) - this script exists so a human can
scan every real overlay directly rather than more spot-checking single
images one at a time.

Mirrors scripts/run_lac_pull_1906_autosidecar.py's structure and
step2_summary.json shape - see that script's own docstring for why it
calls core.auto_sidecar.generate_auto_sidecar() directly instead of
going through scripts/run_batch_auto_sidecar.py (doc_type is already
known from the folder, no Gemma classification/pipeline_db needed).

Usage:
    python scripts/run_lac_pull_1911_overlays.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from core.auto_sidecar import generate_auto_sidecar
from core.row_segmentation import apply_deskew_angle, estimate_deskew_angle, save_sidecar

# Source images: the working folder these 92 images (and the 6-sample
# template calibration) actually live in - lac_pull_batch1/images/ is a
# separate, older copy that predates the columns_file/expected_columns
# fix (see canada_census_1911.yaml's own history), not guaranteed to be
# the same 92 files post-fix.
RAW_DIR = Path(r"J:\Genealogy\genealogy_workspace\research\calibration\column_calibration_workspace\Schedule_1_3a9fae1f\images")
BATCH_DIR = Path(r"J:\Genealogy\genealogy_workspace\datasets\reference\lac_pull_batch1")
DESKEWED_DIR = BATCH_DIR / "deskewed_images"
SIDECAR_DIR = BATCH_DIR / "sidecars"
OVERLAY_DIR = BATCH_DIR / "overlays"
DOC_TYPE = "canada_census_1911"


def main():
    DESKEWED_DIR.mkdir(parents=True, exist_ok=True)
    SIDECAR_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    raw_images = sorted(RAW_DIR.glob("*.png"))
    print(f"Found {len(raw_images)} raw image(s) in {RAW_DIR}")

    summary = []
    for i, img_path in enumerate(raw_images, 1):
        stem = img_path.stem
        print(f"[{i}/{len(raw_images)}] {stem}")

        try:
            result = generate_auto_sidecar(
                str(img_path), debug=True,
                doc_type_override=DOC_TYPE, external_confidence=1.0,
            )
        except Exception as e:
            print(f"  -> FAILED: {e}")
            summary.append({
                "stem": stem, "ok": False, "status": "error",
                "source_used": "cv_only", "n_rows": 0, "n_quarantined_rows": 0,
                "table_top": None, "image_size": None, "sidecar_path": "",
                "deskewed_path": "", "overlay_path": "", "error": str(e)[:300],
            })
            continue

        if result.sidecar is None:
            print(f"  -> no sidecar produced: {'; '.join(result.warnings)[:200]}")
            summary.append({
                "stem": stem, "ok": False, "status": "no_sidecar",
                "source_used": "cv_only", "n_rows": 0, "n_quarantined_rows": 0,
                "table_top": None, "image_size": None, "sidecar_path": "",
                "deskewed_path": "", "overlay_path": "",
                "error": "; ".join(result.warnings)[:300],
            })
            continue

        original = Image.open(img_path).convert("RGB")
        angle = estimate_deskew_angle(original)
        deskewed = apply_deskew_angle(original, angle)
        deskewed_path = DESKEWED_DIR / f"{stem}.png"
        deskewed.save(deskewed_path)

        sidecar = result.sidecar
        sidecar["source_image_path"] = str(deskewed_path)
        sidecar_path = SIDECAR_DIR / f"{stem}_sidecar.json"
        save_sidecar(sidecar, sidecar_path)

        overlay_path = ""
        if result.debug_overlay is not None:
            overlay_path = str(OVERLAY_DIR / f"{stem}_debug_overlay.png")
            result.debug_overlay.save(overlay_path)

        n_rows = len(sidecar["rows"])
        n_quarantined = len(sidecar.get("rows_needs_review", []))
        table_top = sidecar.get("table_bbox", [None, None])[1]
        print(f"  -> OK: {n_rows} row(s), {n_quarantined} quarantined, table_top={table_top}")

        summary.append({
            "stem": stem, "ok": True, "status": "OK",
            "source_used": "cv_only",
            "n_rows": n_rows, "n_quarantined_rows": n_quarantined,
            "table_top": table_top,
            "image_size": list(deskewed.size),
            "sidecar_path": str(sidecar_path),
            "deskewed_path": str(deskewed_path),
            "overlay_path": overlay_path,
        })

    summary_path = BATCH_DIR / "step2_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    ok_count = sum(1 for s in summary if s["ok"])
    total_rows = sum(s["n_rows"] for s in summary if s["ok"])
    total_quarantined = sum(s["n_quarantined_rows"] for s in summary if s["ok"])
    print(f"\nDone. {ok_count}/{len(summary)} OK, {total_rows} row(s) total, "
          f"{total_quarantined} quarantined.")
    print(f"Overlays: {OVERLAY_DIR}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
