"""
Layout-detector bootstrap training, step 1: run the 16 census images
through core/auto_sidecar.py's generate_auto_sidecar() (the "mostly
dialed in" auto_row/auto_column pipeline) to get real table/header/row
bounding boxes, then cross-validate row counts against
data/automatedgenealogy_pull.csv's real per-image transcribed row counts
before trusting these boxes as training labels.

Read-only against the corpus; writes only to
data/outputs/layout_bootstrap_train/sidecars/ (one JSON per image) and
prints the validation table.

Usage:
    python -m training.layout_bootstrap_step1_generate_sidecars
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKING_DIR = PROJECT_ROOT / "data" / "working"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "layout_bootstrap_train"

CENSUS_16 = [
    "e001926997", "e001928017", "e001943201", "e001946014", "e001946614",
    "e001946615", "e001946616", "e001946617", "e001946618", "e001946619",
    "e001946620", "e001946621", "e001946622", "e001946623", "e001961124",
    "e002101688",
]


def _load_expected_row_counts() -> dict[str, int]:
    with open(PROJECT_ROOT / "data" / "automatedgenealogy_pull.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    counts: dict[str, int] = {}
    for r in rows:
        for stem in CENSUS_16:
            if stem in r["source_pdf"]:
                counts[stem] = counts.get(stem, 0) + 1
    return counts


def main() -> None:
    from core.auto_sidecar import generate_auto_sidecar

    expected_counts = _load_expected_row_counts()

    sidecars_dir = OUTPUT_DIR / "sidecars"
    deskewed_dir = OUTPUT_DIR / "deskewed_images"
    sidecars_dir.mkdir(parents=True, exist_ok=True)
    deskewed_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'image':<16} {'expected_rows':<14} {'detected_rows':<14} {'table_bbox':<30} {'status'}")
    results = []
    for stem in CENSUS_16:
        image_path = WORKING_DIR / f"{stem}.png"
        if not image_path.exists():
            print(f"{stem:<16} MISSING FILE at {image_path}")
            continue
        result = generate_auto_sidecar(image_path, doc_type_override="canada_census_1911")
        expected = expected_counts.get(stem, "?")

        if result.sidecar is None:
            print(f"{stem:<16} {str(expected):<14} {'N/A':<14} {'N/A':<30} FAILED: {result.warnings}")
            results.append({"stem": stem, "ok": False, "warnings": result.warnings})
            continue

        detected = len(result.sidecar["rows"])
        table_bbox = result.sidecar["table_bbox"]
        status = "OK" if isinstance(expected, int) and abs(detected - expected) <= 3 else "CHECK"
        print(f"{stem:<16} {str(expected):<14} {detected:<14} {str(table_bbox):<30} {status}")

        # Save sidecar
        sidecar_path = sidecars_dir / f"{stem}_sidecar.json"
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(result.sidecar, f, indent=2)

        # Save the DESKEWED image (bboxes are in deskewed coordinate
        # space - training must use this, not the raw working copy)
        from core.image_analysis import DESKEW_ANGLE_RANGE
        from core.auto_sidecar import estimate_deskew_angle, apply_deskew_angle
        original = Image.open(str(image_path)).convert("RGB")
        angle = result.diagnostics["deskew_angle"]
        deskewed = apply_deskew_angle(original, angle)
        deskewed_path = deskewed_dir / f"{stem}.png"
        deskewed.save(deskewed_path)

        results.append({
            "stem": stem, "ok": True, "expected_rows": expected, "detected_rows": detected,
            "status": status, "sidecar_path": str(sidecar_path), "deskewed_path": str(deskewed_path),
        })

    summary_path = OUTPUT_DIR / "step1_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary written to {summary_path}")
    n_ok = sum(1 for r in results if r.get("ok") and r.get("status") == "OK")
    print(f"{n_ok}/{len(CENSUS_16)} images: row count within tolerance of transcription count")


if __name__ == "__main__":
    main()
