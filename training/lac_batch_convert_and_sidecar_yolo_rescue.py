"""
Generalized version of training/lac_batch2_convert_and_sidecar_yolo_rescue.py -
parameterized by batch directory, doc_type template, and rescue
checkpoint, so the same conditional CV-first-then-YOLO-rescue-if-
needed labeling pipeline can be reused across different census years
(1911, 1921, 1901, ...) without duplicating near-identical scripts.

For each page: baseline CV-only generate_auto_sidecar() runs FIRST.
Only if that result is quarantined (whole-page OR a high partial-
quarantine rate) does it rerun with YOLO-assistance (using whichever
checkpoint --checkpoint points at - the FROZEN, latest-validated
checkpoint, used read-only, never retrained here) and use that result
instead, if it succeeds.

Writes only to --batch-dir - never touches data/working/ or any
production corpus file.

Usage:
    python -m training.lac_batch_convert_and_sidecar_yolo_rescue \\
        --batch-dir data/outputs/lac_pull_batch3 --doc-type canada_census_1911
    python -m training.lac_batch_convert_and_sidecar_yolo_rescue \\
        --batch-dir data/outputs/lac_pull_1921_batch1 --doc-type canada_census_1921
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "data" / "outputs" / "layout_bootstrap_train" / "runs"
    / "census_bootstrap_v3" / "weights" / "best.pt"
)

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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-dir", type=str, required=True)
    parser.add_argument("--doc-type", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    args = parser.parse_args()

    batch_dir = Path(args.batch_dir)
    pdf_dir = batch_dir / "raw_pdfs"
    image_dir = batch_dir / "images"
    sidecar_dir = batch_dir / "sidecars"
    deskewed_dir = batch_dir / "deskewed_images"

    from core.auto_sidecar import generate_auto_sidecar, apply_deskew_angle
    from core.pdf_conversion import convert_pdf, DEFAULT_DPI
    from training.yolo_assisted_auto_sidecar import (
        build_yolo_table_model, generate_auto_sidecar_yolo_assisted,
    )

    image_dir.mkdir(parents=True, exist_ok=True)
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    deskewed_dir.mkdir(parents=True, exist_ok=True)

    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    print(f"Processing {len(pdf_paths)} PDF(s) for doc_type={args.doc_type}: "
          f"baseline CV first, YOLO-rescue only if quarantined...")

    checkpoint_path = Path(args.checkpoint)
    print(f"Loading checkpoint (read-only, for rescue assistance only): {checkpoint_path}")
    yolo_model = build_yolo_table_model(checkpoint_path)

    results = []
    n_cv_only, n_rescued, n_still_bad = 0, 0, 0
    for i, pdf_path in enumerate(pdf_paths, 1):
        stem = pdf_path.stem
        written = convert_pdf(pdf_path, image_dir, dpi=DEFAULT_DPI, ext="png")
        if not written:
            results.append({"stem": stem, "ok": False, "reason": "conversion produced no output"})
            continue
        image_path = written[0]
        with Image.open(image_path) as img:
            w, h = img.size

        baseline = generate_auto_sidecar(image_path, doc_type_override=args.doc_type)
        bad, reason = _is_quarantined(baseline.sidecar, baseline.diagnostics)

        source_used = "cv_only"
        final_result = baseline
        if bad:
            assisted, exp_diag = generate_auto_sidecar_yolo_assisted(
                image_path, yolo_model, doc_type_override=args.doc_type)
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

        sidecar_path = sidecar_dir / f"{stem}_sidecar.json"
        sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

        angle = final_result.diagnostics["deskew_angle"]
        original = Image.open(str(image_path)).convert("RGB")
        deskewed = apply_deskew_angle(original, angle)
        deskewed_path = deskewed_dir / f"{stem}.png"
        deskewed.save(deskewed_path)

        results.append({
            "stem": stem, "ok": True, "status": "OK", "source_used": source_used,
            "n_rows": n_rows, "n_quarantined_rows": n_quarantined_rows,
            "image_size": [w, h], "sidecar_path": str(sidecar_path), "deskewed_path": str(deskewed_path),
        })

    summary_path = batch_dir / "step2_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    n_ok = sum(1 for r in results if r.get("ok"))
    print(f"\n{n_ok}/{len(pdf_paths)} usable ({n_cv_only} CV-only clean, {n_rescued} YOLO-rescued, "
          f"{n_still_bad} excluded even after rescue attempt)")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
