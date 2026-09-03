"""
Fills the one real gap in the LAC-pull row/column training-set test:
lac_pull_1906_batch1/raw_jpgs/ (24 raw, undewarped scans) has no
sidecars/deskewed_images/step2_summary.json yet, unlike lac_pull_batch1
(1911-era) and lac_pull_1921_batch1, which already have 100 complete
auto_sidecar sidecars each with 50/50 rows and calibrated column masks
(2026-08-09, per Jon's direction: test whether row+column geometry from
core/auto_sidecar.py is good enough to train YOLO on, before expanding
to all census years).

Deliberately does NOT go through scripts/run_batch_auto_sidecar.py -
that script requires a Gemma subtype-classification CSV and only
operates on images already registered as dewarped in pipeline_db
(neither exists for this reference dataset, which lives in
genealogy_workspace/datasets/reference/ precisely because it's
curated/reused input, not per-run pipeline output - see docs/
RUN_ARCHITECTURE.md's ownership model). Unlike that script's DB-driven
find_dewarped(), this calls core.auto_sidecar.generate_auto_sidecar()
directly, which does its own deskew internally (estimate_deskew_angle +
apply_deskew_angle) - no pre-dewarped image or DB registration needed.
doc_type is already known from the folder name (canada_census_1906),
so no Gemma classification step is needed either - passed straight in
via doc_type_override, exactly like scripts/run_batch_auto_sidecar.py
does per-row once Gemma has classified something.

Output layout and step2_summary.json schema deliberately MIRROR
lac_pull_1921_batch1/lac_pull_batch1 exactly (same per-image summary
dict shape) so whatever downstream tooling turns those two batches'
sidecars into YOLO training data works unmodified against 1906 too.

Usage:
    python scripts/run_lac_pull_1906_autosidecar.py
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

BATCH_DIR = Path(r"J:\Genealogy\genealogy_workspace\datasets\reference\lac_pull_1906_batch1")
# 2026-08-09, per Jon: the raw_jpgs/ scans each contain TWO census tables
# per image (not one) - unusable for row/column geometry as-is. Correct
# source is the already-split L/R halves from the column-calibration
# session's own working folder (one table per image, no blanks) - same
# folder those column_regions_approx bands in config/document_templates/
# canada_census_1906.yaml were calibrated against.
RAW_DIR = Path(r"J:\Genealogy\genealogy_workspace\research\calibration\column_calibration_workspace\raw_jpgs_b58a321e\images")
DESKEWED_DIR = BATCH_DIR / "deskewed_images"
SIDECAR_DIR = BATCH_DIR / "sidecars"
OVERLAY_DIR = BATCH_DIR / "overlays"
DOC_TYPE = "canada_census_1906"


def main():
    DESKEWED_DIR.mkdir(parents=True, exist_ok=True)
    SIDECAR_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    raw_images = sorted(RAW_DIR.glob("*.jpg"))
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
                "image_size": None, "sidecar_path": "", "deskewed_path": "",
                "error": str(e)[:300],
            })
            continue

        if result.sidecar is None:
            print(f"  -> no sidecar produced: {'; '.join(result.warnings)[:200]}")
            summary.append({
                "stem": stem, "ok": False, "status": "no_sidecar",
                "source_used": "cv_only", "n_rows": 0, "n_quarantined_rows": 0,
                "image_size": None, "sidecar_path": "", "deskewed_path": "",
                "error": "; ".join(result.warnings)[:300],
            })
            continue

        # Save the plain deskewed image alongside the sidecar - same
        # "deskewed_images/<stem>.png" convention lac_pull_batch1/
        # lac_pull_1921_batch1 already use, and what the sidecar's own
        # source_image_path below points at.
        original = Image.open(img_path).convert("RGB")
        angle = estimate_deskew_angle(original)
        deskewed = apply_deskew_angle(original, angle)
        deskewed_path = DESKEWED_DIR / f"{stem}.png"
        deskewed.save(deskewed_path)

        sidecar = result.sidecar
        sidecar["source_image_path"] = str(deskewed_path)
        sidecar_path = SIDECAR_DIR / f"{stem}_sidecar.json"
        save_sidecar(sidecar, sidecar_path)

        if result.debug_overlay is not None:
            result.debug_overlay.save(OVERLAY_DIR / f"{stem}_debug_overlay.png")

        n_rows = len(sidecar["rows"])
        n_quarantined = len(sidecar.get("rows_needs_review", []))
        print(f"  -> OK: {n_rows} row(s), {n_quarantined} quarantined")

        summary.append({
            "stem": stem, "ok": True, "status": "OK",
            "source_used": "cv_only",
            "n_rows": n_rows, "n_quarantined_rows": n_quarantined,
            "image_size": list(deskewed.size),
            "sidecar_path": str(sidecar_path),
            "deskewed_path": str(deskewed_path),
        })

    summary_path = BATCH_DIR / "step2_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    ok_count = sum(1 for s in summary if s["ok"])
    total_rows = sum(s["n_rows"] for s in summary if s["ok"])
    total_quarantined = sum(s["n_quarantined_rows"] for s in summary if s["ok"])
    print(f"\nDone. {ok_count}/{len(summary)} OK, {total_rows} row(s) total, "
          f"{total_quarantined} quarantined.")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
