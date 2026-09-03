"""
Controlled comparison: baseline (core/auto_sidecar.py's unmodified
generate_auto_sidecar(), CV-only) vs. experimental (training/
yolo_assisted_auto_sidecar.py's generate_auto_sidecar_yolo_assisted(),
YOLO v2-checkpoint table hint -> existing CV refinement -> existing
header/column/row/quarantine logic), over a fixed, representative
evaluation set.

Evaluation set (53 images, chosen to cover easy/normal/difficult/
quarantined cases per spec, NOT tuned or hand-picked to favor either
mode):
  - all 16 original transcribed census pages (data/working/) - includes
    3 pages already known difficult from the v1 bootstrap (row-count
    mismatch vs. data/automatedgenealogy_pull.csv)
  - all 10 whole-page-quarantined pages from the LAC pull batch
  - all 6 partial-quarantine ("difficult") pages from the LAC pull batch
  - 21 clean/OK LAC pull batch pages (every 4th, deterministic sample)

Writes to data/outputs/yolo_assisted_auto_sidecar/:
    baseline/sidecars/*.json, assisted/sidecars/*.json
    comparison.csv, summary.json

Usage:
    python -m training.yolo_assisted_auto_sidecar_compare
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKING_DIR = PROJECT_ROOT / "data" / "working"
LAC_IMAGES_DIR = PROJECT_ROOT / "data" / "outputs" / "lac_pull_batch1" / "images"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "yolo_assisted_auto_sidecar"

ORIGINAL_16 = [
    "e001926997", "e001928017", "e001943201", "e001946014", "e001946614",
    "e001946615", "e001946616", "e001946617", "e001946618", "e001946619",
    "e001946620", "e001946621", "e001946622", "e001946623", "e001961124",
    "e002101688",
]
LAC_QUARANTINED = [
    "e001946629", "e001946632", "e001946633", "e001946634", "e001946638",
    "e001946652", "e001946699", "e001946702", "e001946708", "e001946720",
]
LAC_DIFFICULT = [
    "e001946631", "e001946644", "e001946690", "e001946697", "e001946716", "e001946717",
]
LAC_CLEAN_SAMPLE = [
    "e001946624", "e001946628", "e001946637", "e001946642", "e001946647",
    "e001946651", "e001946656", "e001946660", "e001946664", "e001946668",
    "e001946672", "e001946676", "e001946680", "e001946684", "e001946688",
    "e001946693", "e001946698", "e001946704", "e001946709", "e001946713",
    "e001946719",
]


def _resolve_path(stem: str) -> Path:
    original_path = WORKING_DIR / f"{stem}.png"
    if original_path.exists():
        return original_path
    return LAC_IMAGES_DIR / f"{stem}.png"


def _load_transcription_row_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    with open(PROJECT_ROOT / "data" / "automatedgenealogy_pull.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for stem in ORIGINAL_16:
                if stem in row["source_pdf"]:
                    counts[stem] = counts.get(stem, 0) + 1
    return counts


def _rows_extending_outside(rows: list[dict], img_w: int, img_h: int) -> int:
    n = 0
    for r in rows:
        x0, y0, x1, y1 = r["bbox"]
        if x0 < 0 or y0 < 0 or x1 > img_w or y1 > img_h:
            n += 1
    return n


def _sidecar_metrics(sidecar: dict | None, diagnostics: dict, warnings: list[str]) -> dict:
    if sidecar is None:
        return {
            "status": "CLASSIFICATION_FAILED", "table_bbox": None, "table_area_frac": None,
            "n_rows": 0, "n_quarantined_rows": 0, "n_rows_outside_bounds": 0,
            "n_columns": 0, "warnings": "; ".join(warnings),
        }
    img_w, img_h = sidecar["deskewed_image_size"]
    table_bbox = sidecar["table_bbox"]
    table_area_frac = None
    if table_bbox:
        tx0, ty0, tx1, ty1 = table_bbox
        table_area_frac = ((tx1 - tx0) * (ty1 - ty0)) / (img_w * img_h) if img_w and img_h else None

    n_quarantined = len(diagnostics.get("quarantined_rows", []))
    whole_page_quarantined = (
        diagnostics.get("row_detection", {}).get("page_detection_failed", False)
        or diagnostics.get("table_boundary", {}).get("table_top_ambiguous", False)
    )
    status = "WHOLE_PAGE_QUARANTINED" if whole_page_quarantined else "OK"

    return {
        "status": status,
        "table_bbox": table_bbox,
        "table_area_frac": round(table_area_frac, 4) if table_area_frac is not None else None,
        "n_rows": len(sidecar["rows"]),
        "n_quarantined_rows": n_quarantined,
        "n_rows_outside_bounds": _rows_extending_outside(sidecar["rows"], img_w, img_h),
        "n_columns": len(sidecar.get("columns", {})),
        "warnings": "; ".join(warnings),
    }


def main() -> None:
    from core.auto_sidecar import generate_auto_sidecar
    from training.yolo_assisted_auto_sidecar import build_yolo_table_model, generate_auto_sidecar_yolo_assisted

    eval_stems = ORIGINAL_16 + LAC_QUARANTINED + LAC_DIFFICULT + LAC_CLEAN_SAMPLE
    print(f"Evaluation set: {len(eval_stems)} images "
          f"({len(ORIGINAL_16)} original + {len(LAC_QUARANTINED)} quarantined + "
          f"{len(LAC_DIFFICULT)} difficult + {len(LAC_CLEAN_SAMPLE)} clean sample)")

    transcription_counts = _load_transcription_row_counts()

    baseline_dir = OUTPUT_DIR / "baseline" / "sidecars"
    assisted_dir = OUTPUT_DIR / "assisted" / "sidecars"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    assisted_dir.mkdir(parents=True, exist_ok=True)

    print("Loading YOLO v2 checkpoint...")
    yolo_model = build_yolo_table_model()

    rows_out = []
    for i, stem in enumerate(eval_stems, 1):
        image_path = _resolve_path(stem)
        if not image_path.exists():
            print(f"[{i}/{len(eval_stems)}] {stem}: MISSING FILE, skipping")
            continue

        t0 = time.time()
        baseline_result = generate_auto_sidecar(image_path, doc_type_override="canada_census_1911")
        baseline_runtime = time.time() - t0

        t0 = time.time()
        assisted_result, exp_diag = generate_auto_sidecar_yolo_assisted(
            image_path, yolo_model, doc_type_override="canada_census_1911")
        assisted_runtime = time.time() - t0

        if baseline_result.sidecar is not None:
            (baseline_dir / f"{stem}_sidecar.json").write_text(
                json.dumps(baseline_result.sidecar, indent=2), encoding="utf-8")
        if assisted_result.sidecar is not None:
            (assisted_dir / f"{stem}_sidecar.json").write_text(
                json.dumps(assisted_result.sidecar, indent=2), encoding="utf-8")

        b = _sidecar_metrics(baseline_result.sidecar, baseline_result.diagnostics, baseline_result.warnings)
        a = _sidecar_metrics(assisted_result.sidecar, assisted_result.diagnostics, assisted_result.warnings)

        table_boundary_diff = None
        if b["table_bbox"] and a["table_bbox"]:
            table_boundary_diff = sum(abs(bb - aa) for bb, aa in zip(b["table_bbox"], a["table_bbox"]))

        transcription_count = transcription_counts.get(stem)
        row = {
            "stem": stem, "source": "original_16" if stem in ORIGINAL_16 else
                ("lac_quarantined" if stem in LAC_QUARANTINED else
                 ("lac_difficult" if stem in LAC_DIFFICULT else "lac_clean_sample")),
            "transcription_row_count": transcription_count,
            "baseline_status": b["status"], "assisted_status": a["status"],
            "baseline_table_bbox": b["table_bbox"], "assisted_table_bbox": a["table_bbox"],
            "baseline_table_area_frac": b["table_area_frac"], "assisted_table_area_frac": a["table_area_frac"],
            "table_boundary_diff_px_sum": table_boundary_diff,
            "baseline_n_rows": b["n_rows"], "assisted_n_rows": a["n_rows"],
            "baseline_n_quarantined_rows": b["n_quarantined_rows"], "assisted_n_quarantined_rows": a["n_quarantined_rows"],
            "baseline_n_rows_outside_bounds": b["n_rows_outside_bounds"], "assisted_n_rows_outside_bounds": a["n_rows_outside_bounds"],
            "baseline_n_columns": b["n_columns"], "assisted_n_columns": a["n_columns"],
            "baseline_row_count_error": abs(b["n_rows"] - transcription_count) if transcription_count else None,
            "assisted_row_count_error": abs(a["n_rows"] - transcription_count) if transcription_count else None,
            "yolo_hint_used": exp_diag.get("hint_used"),
            "yolo_hint_reason": exp_diag.get("hint_reason", exp_diag.get("hint_validation_reason", "")),
            "yolo_candidate_confidence": exp_diag.get("yolo_selected_candidate", {}).get("confidence")
                if exp_diag.get("yolo_selected_candidate") else None,
            "baseline_warnings": b["warnings"], "assisted_warnings": a["warnings"],
            "baseline_runtime_s": round(baseline_runtime, 2), "assisted_runtime_s": round(assisted_runtime, 2),
        }
        rows_out.append(row)

        rescued = b["status"] == "WHOLE_PAGE_QUARANTINED" and a["status"] == "OK"
        newly_quarantined = b["status"] == "OK" and a["status"] == "WHOLE_PAGE_QUARANTINED"
        flag = " <-- RESCUED" if rescued else (" <-- NEWLY QUARANTINED" if newly_quarantined else "")
        print(f"[{i}/{len(eval_stems)}] {stem}: baseline={b['status']}/{b['n_rows']}r "
              f"assisted={a['status']}/{a['n_rows']}r hint_used={exp_diag.get('hint_used')}{flag}")

    fieldnames = list(rows_out[0].keys()) if rows_out else []
    comparison_csv = OUTPUT_DIR / "comparison.csv"
    with open(comparison_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    # -- summary --------------------------------------------------------
    n_total = len(rows_out)
    n_rescued = sum(1 for r in rows_out if r["baseline_status"] == "WHOLE_PAGE_QUARANTINED" and r["assisted_status"] == "OK")
    n_newly_quarantined = sum(1 for r in rows_out if r["baseline_status"] == "OK" and r["assisted_status"] == "WHOLE_PAGE_QUARANTINED")
    n_hint_used = sum(1 for r in rows_out if r["yolo_hint_used"])
    n_hint_rejected = n_total - n_hint_used

    b_outside = sum(r["baseline_n_rows_outside_bounds"] for r in rows_out)
    a_outside = sum(r["assisted_n_rows_outside_bounds"] for r in rows_out)

    orig16_rows = [r for r in rows_out if r["source"] == "original_16" and r["transcription_row_count"]]
    b_row_err = [r["baseline_row_count_error"] for r in orig16_rows if r["baseline_row_count_error"] is not None]
    a_row_err = [r["assisted_row_count_error"] for r in orig16_rows if r["assisted_row_count_error"] is not None]

    improved = sum(1 for r in rows_out if r["assisted_row_count_error"] is not None and r["baseline_row_count_error"] is not None
                   and r["assisted_row_count_error"] < r["baseline_row_count_error"])
    degraded = sum(1 for r in rows_out if r["assisted_row_count_error"] is not None and r["baseline_row_count_error"] is not None
                   and r["assisted_row_count_error"] > r["baseline_row_count_error"])
    unchanged_rowcount = sum(1 for r in rows_out if r["assisted_row_count_error"] is not None and r["baseline_row_count_error"] is not None
                             and r["assisted_row_count_error"] == r["baseline_row_count_error"])

    summary = {
        "n_total_images": n_total,
        "n_rescued_from_quarantine": n_rescued,
        "n_newly_quarantined": n_newly_quarantined,
        "n_yolo_hint_used": n_hint_used,
        "n_yolo_hint_rejected_or_fallback": n_hint_rejected,
        "total_rows_outside_bounds": {"baseline": b_outside, "assisted": a_outside},
        "original_16_row_count_error": {
            "baseline_mean": round(sum(b_row_err) / len(b_row_err), 2) if b_row_err else None,
            "assisted_mean": round(sum(a_row_err) / len(a_row_err), 2) if a_row_err else None,
            "baseline_total": sum(b_row_err) if b_row_err else None,
            "assisted_total": sum(a_row_err) if a_row_err else None,
        },
        "row_count_accuracy_vs_transcription": {
            "images_improved": improved, "images_degraded": degraded, "images_unchanged": unchanged_rowcount,
        },
    }
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nComparison CSV: {comparison_csv}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
