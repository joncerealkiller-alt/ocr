"""
LAC pull batch 2, conditional processing: converts PDFs to PNG, then for
each page runs baseline CV-only generate_auto_sidecar() FIRST. Only if
that result is quarantined (whole-page OR a high partial-quarantine
rate) does it rerun with the FROZEN v2 checkpoint's YOLO-assistance
(training/yolo_assisted_auto_sidecar.py) and use that result instead,
if it succeeds - operationalizing the [C] conditional-rescue design
from data/outputs/yolo_assisted_auto_sidecar/report.md as the actual
labeling strategy for dataset expansion, not just a future idea.

Pages already fine under baseline CV never touch YOLO at all - keeps
the "assist only when needed" design real, not just descriptive, and
matches the report's own finding that YOLO-assist has no measured
benefit (and a small cost) on already-working pages.

The v2 checkpoint (data/outputs/layout_bootstrap_train/runs/
census_bootstrap_v2/weights/best.pt) is used READ-ONLY here, purely as
a labeling aid - it is NOT touched, retrained, or continued. A
SEPARATE, later v3 training pass (fresh YOLOv26-small base, same as
v1/v2) is what actually learns from this batch's resulting labels.

Writes only to data/outputs/lac_pull_batch2/ - never touches
data/working/ or any production corpus file.

Usage:
    python -m training.lac_batch2_convert_and_sidecar_yolo_rescue
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BATCH_DIR = PROJECT_ROOT / "data" / "outputs" / "lac_pull_batch2"
PDF_DIR = BATCH_DIR / "raw_pdfs"
IMAGE_DIR = BATCH_DIR / "images"
SIDECAR_DIR = BATCH_DIR / "sidecars"
DESKEWED_DIR = BATCH_DIR / "deskewed_images"

# Same threshold reasoning as training/lac_batch1_convert_and_sidecar.py's
# HIGH_QUARANTINE_RATE check - a page with more than this fraction of
# its rows individually quarantined is not a usable pseudo-label source
# even though it technically has a sidecar.
_HIGH_PARTIAL_QUARANTINE_RATE = 0.3


def _is_quarantined(sidecar: dict | None, diagnostics: dict) -> tuple[bool, str]:
    if sidecar is None:
        return True, "classification/sidecar failed"
    whole_page = (
        diagnostics.get("row_detection", {}).get("page_detection_failed", False)
        or diagnostics.get("table_boundary", {}).get("table_top_ambiguous", False)
    )
    if whole_page:
        return True, "whole-page quarantined"
    n_rows = len(sidecar["rows"])
    n_quarantined = len(diagnostics.get("quarantined_rows", []))
    if n_rows > 0 and (n_quarantined / n_rows) > _HIGH_PARTIAL_QUARANTINE_RATE:
        return True, f"high partial-quarantine rate ({n_quarantined}/{n_rows})"
    return False, "ok"


def main() -> None:
    from core.auto_sidecar import generate_auto_sidecar, apply_deskew_angle
    from core.pdf_conversion import convert_pdf, DEFAULT_DPI
    from training.yolo_assisted_auto_sidecar import (
        build_yolo_table_model, generate_auto_sidecar_yolo_assisted,
    )

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    SIDECAR_DIR.mkdir(parents=True, exist_ok=True)
    DESKEWED_DIR.mkdir(parents=True, exist_ok=True)

    pdf_paths = sorted(PDF_DIR.glob("*.pdf"))
    print(f"Processing {len(pdf_paths)} PDF(s): baseline CV first, YOLO-rescue only if quarantined...")

    print("Loading v2 checkpoint (read-only, for rescue assistance only)...")
    yolo_model = build_yolo_table_model()

    results = []
    n_cv_only, n_rescued, n_still_bad = 0, 0, 0
    for i, pdf_path in enumerate(pdf_paths, 1):
        stem = pdf_path.stem
        written = convert_pdf(pdf_path, IMAGE_DIR, dpi=DEFAULT_DPI, ext="png")
        if not written:
            results.append({"stem": stem, "ok": False, "reason": "conversion produced no output"})
            continue
        image_path = written[0]
        with Image.open(image_path) as img:
            w, h = img.size

        baseline = generate_auto_sidecar(image_path, doc_type_override="canada_census_1911")
        bad, reason = _is_quarantined(baseline.sidecar, baseline.diagnostics)

        source_used = "cv_only"
        final_result = baseline
        if bad:
            assisted, exp_diag = generate_auto_sidecar_yolo_assisted(
                image_path, yolo_model, doc_type_override="canada_census_1911")
            still_bad, assisted_reason = _is_quarantined(assisted.sidecar, assisted.diagnostics)
            if not still_bad:
                final_result = assisted
                source_used = "yolo_rescued"
                n_rescued += 1
            else:
                n_still_bad += 1
                print(f"[{i}/{len(pdf_paths)}] {stem}: EXCLUDED - CV quarantined ({reason}), "
                      f"YOLO-rescue also failed ({assisted_reason})")
                results.append({"stem": stem, "ok": False, "reason": f"cv_bad({reason})+yolo_bad({assisted_reason})"})
                continue
        else:
            n_cv_only += 1

        sidecar = final_result.sidecar
        n_rows = len(sidecar["rows"])
        n_quarantined_rows = len(final_result.diagnostics.get("quarantined_rows", []))
        print(f"[{i}/{len(pdf_paths)}] {stem}: {source_used}, {n_rows} row(s), "
              f"{n_quarantined_rows} quarantined")

        sidecar_path = SIDECAR_DIR / f"{stem}_sidecar.json"
        sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

        angle = final_result.diagnostics["deskew_angle"]
        original = Image.open(str(image_path)).convert("RGB")
        deskewed = apply_deskew_angle(original, angle)
        deskewed_path = DESKEWED_DIR / f"{stem}.png"
        deskewed.save(deskewed_path)

        results.append({
            "stem": stem, "ok": True, "status": "OK", "source_used": source_used,
            "n_rows": n_rows, "n_quarantined_rows": n_quarantined_rows,
            "image_size": [w, h], "sidecar_path": str(sidecar_path), "deskewed_path": str(deskewed_path),
        })

    summary_path = BATCH_DIR / "step2_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    n_ok = sum(1 for r in results if r.get("ok"))
    print(f"\n{n_ok}/{len(pdf_paths)} usable ({n_cv_only} CV-only clean, {n_rescued} YOLO-rescued, "
          f"{n_still_bad} excluded even after rescue attempt)")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
