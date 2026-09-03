"""
LAC pull batch 1, step 2: converts the 50 fetched PDFs
(data/outputs/lac_pull_batch1/raw_pdfs/) to PNG at 300 DPI (core/
pdf_conversion.py's convert_pdf(), same DPI this project's existing
corpus was captured at), then runs each through core/auto_sidecar.py's
generate_auto_sidecar() (doc_type_override="canada_census_1911") for
row/table/header pseudo-labels - same pipeline as training/layout_
bootstrap_step1_generate_sidecars.py, applied to this new batch.

NO automatedgenealogy_pull.csv-style external transcription ground
truth exists for these 50 NEW pages (that CSV only covers the original
16) - row-count validation here relies entirely on auto_sidecar's own
internal quality signals (page_detection_failed / table_top_ambiguous
whole-page quarantine flags, per-row anomaly quarantine) rather than an
external count to compare against. This is a real, weaker validation
than step 1 had - flagged explicitly in the output, not glossed over.

Writes only to data/outputs/lac_pull_batch1/ - never touches
data/working/ or any production corpus file.

Usage:
    python -m training.lac_batch1_convert_and_sidecar
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BATCH_DIR = PROJECT_ROOT / "data" / "outputs" / "lac_pull_batch1"
PDF_DIR = BATCH_DIR / "raw_pdfs"
IMAGE_DIR = BATCH_DIR / "images"
SIDECAR_DIR = BATCH_DIR / "sidecars"
DESKEWED_DIR = BATCH_DIR / "deskewed_images"


def main() -> None:
    from core.pdf_conversion import convert_pdf, DEFAULT_DPI
    from core.auto_sidecar import generate_auto_sidecar, apply_deskew_angle

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    SIDECAR_DIR.mkdir(parents=True, exist_ok=True)
    DESKEWED_DIR.mkdir(parents=True, exist_ok=True)

    pdf_paths = sorted(PDF_DIR.glob("*.pdf"))
    print(f"Converting {len(pdf_paths)} PDF(s) at {DEFAULT_DPI} DPI...")

    results = []
    for i, pdf_path in enumerate(pdf_paths, 1):
        stem = pdf_path.stem
        written = convert_pdf(pdf_path, IMAGE_DIR, dpi=DEFAULT_DPI, ext="png")
        if not written:
            print(f"[{i}/{len(pdf_paths)}] {stem}: conversion produced no output, skipping")
            results.append({"stem": stem, "ok": False, "reason": "no image produced"})
            continue
        image_path = written[0]

        with Image.open(image_path) as img:
            w, h = img.size
        # Real content pages are ~7000x4700px at 300 DPI; the notably
        # smaller downloads (~180-230KB PDFs) flagged after the pull
        # are worth a direct size check rather than assuming.
        is_small_page = (w * h) < 5_000_000

        try:
            result = generate_auto_sidecar(image_path, doc_type_override="canada_census_1911")
        except Exception as e:
            print(f"[{i}/{len(pdf_paths)}] {stem}: generate_auto_sidecar() raised {type(e).__name__}: {e}")
            results.append({"stem": stem, "ok": False, "reason": f"{type(e).__name__}: {e}",
                             "image_size": [w, h], "is_small_page": is_small_page})
            continue

        if result.sidecar is None:
            print(f"[{i}/{len(pdf_paths)}] {stem}: classification/sidecar FAILED - {result.warnings}")
            results.append({"stem": stem, "ok": False, "reason": "sidecar is None",
                             "warnings": result.warnings, "image_size": [w, h], "is_small_page": is_small_page})
            continue

        n_rows = len(result.sidecar["rows"])
        n_quarantined_rows = len(result.diagnostics.get("quarantined_rows", []))
        whole_page_quarantined = result.diagnostics.get("row_detection", {}).get("page_detection_failed", False) or \
            result.diagnostics.get("table_boundary", {}).get("table_top_ambiguous", False)

        status = "OK"
        if whole_page_quarantined:
            status = "WHOLE_PAGE_QUARANTINED"
        elif n_quarantined_rows > n_rows * 0.3:
            status = "HIGH_QUARANTINE_RATE"
        elif is_small_page:
            status = "SMALL_PAGE_CHECK"

        print(f"[{i}/{len(pdf_paths)}] {stem}: {n_rows} row(s), "
              f"{n_quarantined_rows} quarantined, size={w}x{h}, status={status}")

        sidecar_path = SIDECAR_DIR / f"{stem}_sidecar.json"
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(result.sidecar, f, indent=2)

        angle = result.diagnostics["deskew_angle"]
        original = Image.open(str(image_path)).convert("RGB")
        deskewed = apply_deskew_angle(original, angle)
        deskewed_path = DESKEWED_DIR / f"{stem}.png"
        deskewed.save(deskewed_path)

        results.append({
            "stem": stem, "ok": True, "status": status, "n_rows": n_rows,
            "n_quarantined_rows": n_quarantined_rows, "image_size": [w, h], "is_small_page": is_small_page,
            "sidecar_path": str(sidecar_path), "deskewed_path": str(deskewed_path),
        })

    summary_path = BATCH_DIR / "step2_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    n_ok = sum(1 for r in results if r.get("ok") and r.get("status") == "OK")
    n_flagged = sum(1 for r in results if r.get("ok") and r.get("status") != "OK")
    n_failed = sum(1 for r in results if not r.get("ok"))
    print(f"\n{n_ok} OK, {n_flagged} flagged for review, {n_failed} failed outright, "
          f"out of {len(results)} total")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
