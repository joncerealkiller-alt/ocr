"""
Batch-runs the CV sidecar pipeline (core/auto_sidecar.py) across every
row in a Gemma subtype-classification output CSV (data/buckets/
dense_tabular_rows_subtype.csv, from scripts/run_semantic_stages.py) -
Jon's "phase 2" (2026-07-27): the manifest Gemma already classified
drives which CV template gets used, instead of core/document_
classification.py's own CV classifier guessing again. That CV
classifier is NOT removed or modified - generate_auto_sidecar()'s new
doc_type_override parameter just skips calling it when a doc_type is
already known, per Jon's Phase 2 design (retained as an independent
benchmark/fallback, not the pipeline driver going forward).

Only operates on ALREADY-DEWARPED images (data/outputs/dewarped/
<stem>_dewarped.*) - dewarping stays a separate, human-confirmed step
(ui/dewarp_preprocessor_ui.py / core/bucket_worklist.py), matching the
existing manual workflow's own division of labor; this script does not
attempt to dewarp anything itself. Rows whose source image has no
dewarped version yet are skipped with a clear reason, not silently
dropped or force-processed.

Rows with document_type in ("", "unknown") or a parse error from the
subtype stage are skipped - no template exists for "unknown", and a
row with a parse error has no reliable doc_type to act on at all.

No model inference happens in this script (pure CV, like core/
auto_sidecar.py itself) - CLAUDE.md's GPU-contention check doesn't
apply here.

Usage:
    python scripts/run_batch_auto_sidecar.py \\
        [--input data/buckets/dense_tabular_rows_subtype.csv] \\
        [--out-dir data/outputs/auto_row_segmentation] \\
        [--debug] [--force]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.auto_sidecar import generate_auto_sidecar
from core.row_segmentation import save_sidecar
from core.manifest_pipeline import IMAGE_EXTENSIONS

DEWARPED_DIR = PROJECT_ROOT / "data" / "outputs" / "dewarped"
DEFAULT_INPUT = PROJECT_ROOT / "data" / "buckets" / "dense_tabular_rows_subtype.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "outputs" / "auto_row_segmentation"

SUMMARY_FIELDS = [
    "file_path", "stem", "gemma_doc_type", "gemma_confidence",
    "status", "sidecar_path", "rows_generated", "rows_quarantined",
    "warnings_count", "error",
]

SKIP_DOC_TYPES = {"", "unknown"}

# Queue of pages a human still has to assign a doc_type to, written
# alongside the run summary. Kept as its OWN csv rather than a status
# filter over batch_run_summary.csv so it can be handed straight to a
# manual-classification tool as that tool's input worklist (Jon,
# 2026-07-30) - same "one CSV per worklist" convention
# core/bucket_worklist.py already uses for the bucket/dewarp worklists.
#
# manual_doc_type is written INTENTIONALLY EMPTY: it is the column the
# human (or the manual-classifier UI) fills in, so the same file works as
# both the queue and the completed record, and re-reading it tells you
# what is still outstanding. Nothing in this pipeline writes to it.
MANUAL_CLASSIFICATION_FIELDS = [
    "file_path",        # the original source image
    "image_path",       # what a UI should actually display (dewarped if it exists)
    "stem",
    "reason",           # why this page needs a human
    "gemma_doc_type", "gemma_confidence",
    "cv_guess", "cv_confidence",   # UNTRUSTED - see below
    "manual_doc_type",  # <- for the human/UI to fill in
]

# Valid values for manual_doc_type, emitted into the CSV's own header
# comment row is NOT possible in plain csv, so they live here for
# whatever UI reads this file.
KNOWN_DOC_TYPES = [
    "canada_census_1911", "canada_census_1921", "canada_census_1931",
    "printed_manifest", "handwritten_manifest", "unknown",
]


def find_dewarped(stem: str) -> Path | None:
    """
    Finds this file's dewarped output image. ui/dewarp_preprocessor_ui.py
    writes BOTH an image (<stem>_dewarped.<ext>) and a sidecar-style
    metadata JSON (<stem>_dewarped.json) per file - a bare `*` glob
    matches both, and picks whichever sorts first alphabetically
    ("json" < most image extensions), which meant this was silently
    trying to open the metadata JSON as an image (confirmed against a
    real batch run, 2026-07-29 - every real image was skipped this way).
    Filtered to known image extensions only, so the JSON is never a
    candidate.
    """
    matches = [
        p for p in DEWARPED_DIR.glob(f"{stem}_dewarped.*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return matches[0] if matches else None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT),
                         help="Gemma subtype-classification CSV to drive this batch from.")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR),
                         help="Where sidecars (and --debug overlays) are written.")
    parser.add_argument("--debug", action="store_true",
                         help="Also write a debug overlay PNG per file.")
    parser.add_argument("--force", action="store_true",
                         help="Overwrite an existing sidecar output (refused by default, "
                              "same collision-avoidance rule scripts/auto_generate_sidecar.py uses).")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    if not input_path.exists():
        print(f"ERROR: input not found: {input_path}")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "batch_run_summary.csv"
    manual_path = out_dir / "needs_manual_classification.csv"
    summary_rows = []
    manual_rows = []

    def queue_for_manual(summary_row: dict, reason: str, dewarped: Path | None = None,
                         cv_guess: str = "", cv_confidence: str = "") -> None:
        """Adds one page to the manual-classification worklist. `reason`
        is written verbatim so a reviewer sees WHY the page reached them,
        not just that it did."""
        manual_rows.append({
            "file_path": summary_row["file_path"],
            "image_path": str(dewarped) if dewarped else summary_row["file_path"],
            "stem": summary_row["stem"],
            "reason": reason,
            "gemma_doc_type": summary_row["gemma_doc_type"],
            "gemma_confidence": summary_row["gemma_confidence"],
            "cv_guess": cv_guess,
            "cv_confidence": cv_confidence,
            "manual_doc_type": "",
        })

    processed = skipped_unknown = skipped_no_dewarp = failed = 0

    for i, row in enumerate(rows, 1):
        file_path = row["file_path"]
        doc_type = (row.get("document_type") or "").strip()
        confidence_raw = row.get("confidence") or ""
        stem = Path(file_path).stem
        print(f"[{i}/{len(rows)}] {stem}  (gemma_doc_type={doc_type!r})")

        summary = {
            "file_path": file_path, "stem": stem,
            "gemma_doc_type": doc_type, "gemma_confidence": confidence_raw,
            "status": "", "sidecar_path": "", "rows_generated": "",
            "rows_quarantined": "", "warnings_count": "", "error": "",
        }

        if doc_type in SKIP_DOC_TYPES:
            summary["status"] = "skipped_unknown_doc_type"
            print("  -> SKIP: doc_type is unknown/empty -> manual classification queue")
            skipped_unknown += 1
            queue_for_manual(
                summary,
                reason=f"upstream classifier returned {doc_type!r} - no template to run against",
                dewarped=find_dewarped(stem),
            )
            summary_rows.append(summary)
            continue

        dewarped_path = find_dewarped(stem)
        if dewarped_path is None:
            summary["status"] = "skipped_not_dewarped"
            print("  -> SKIP: no dewarped image found yet")
            skipped_no_dewarp += 1
            summary_rows.append(summary)
            continue

        try:
            confidence = float(confidence_raw) if confidence_raw else None
        except ValueError:
            confidence = None

        sidecar_out_path = out_dir / f"{stem}_dewarped_sidecar.json"
        if sidecar_out_path.exists() and not args.force:
            summary["status"] = "skipped_already_exists"
            summary["sidecar_path"] = str(sidecar_out_path)
            print(f"  -> SKIP: sidecar already exists ({sidecar_out_path.name}), use --force to overwrite")
            summary_rows.append(summary)
            continue

        try:
            result = generate_auto_sidecar(
                str(dewarped_path), debug=args.debug,
                doc_type_override=doc_type, external_confidence=confidence,
            )
            if result.sidecar is None:
                summary["status"] = "unknown_after_all"
                summary["error"] = "; ".join(result.warnings)[:300]
                print(f"  -> no sidecar produced: {summary['error']}")
                failed += 1
                queue_for_manual(
                    summary,
                    reason="no sidecar produced: " + "; ".join(result.warnings)[:200],
                    dewarped=dewarped_path,
                    cv_guess=result.classification.doc_type,
                    cv_confidence=f"{result.classification.confidence:.2f}",
                )
                summary_rows.append(summary)
                continue

            if result.used_cv_fallback:
                # Template was chosen by the CV classifier, not by an
                # upstream semantic classification - a guess, per that
                # flag's own docstring in core/auto_sidecar.py. The
                # sidecar is still written (it may be perfectly good),
                # but the page is queued so a human confirms the doc_type
                # before anything downstream trusts it.
                queue_for_manual(
                    summary,
                    reason="template chosen by CV fallback (untrusted) - confirm doc_type",
                    dewarped=dewarped_path,
                    cv_guess=result.classification.doc_type,
                    cv_confidence=f"{result.classification.confidence:.2f}",
                )

            save_sidecar(result.sidecar, sidecar_out_path)
            if args.debug and result.debug_overlay is not None:
                overlay_path = out_dir / f"{stem}_dewarped_auto_debug_overlay.png"
                result.debug_overlay.save(overlay_path)

            quarantined = len(result.sidecar.get("rows_needs_review", []))
            summary.update({
                "status": "ok",
                "sidecar_path": str(sidecar_out_path),
                "rows_generated": len(result.sidecar["rows"]),
                "rows_quarantined": quarantined,
                "warnings_count": len(result.warnings),
            })
            print(f"  -> OK: {len(result.sidecar['rows'])} row(s), {quarantined} quarantined")
            processed += 1
        except Exception as e:
            summary["status"] = "error"
            summary["error"] = str(e)[:300]
            print(f"  -> FAILED: {e}")
            failed += 1
            queue_for_manual(summary, reason=f"sidecar generation failed: {str(e)[:200]}",
                             dewarped=dewarped_path)

        summary_rows.append(summary)

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nDone. {processed} sidecar(s) generated, "
          f"{skipped_unknown} skipped (unknown doc_type), "
          f"{skipped_no_dewarp} skipped (not dewarped yet), "
          f"{failed} failed.")
    with open(manual_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANUAL_CLASSIFICATION_FIELDS)
        writer.writeheader()
        writer.writerows(manual_rows)

    print(f"Summary: {summary_path}")
    if manual_rows:
        print(f"MANUAL CLASSIFICATION NEEDED for {len(manual_rows)} page(s): {manual_path}")
        print(f"  fill in the empty 'manual_doc_type' column "
              f"(one of: {', '.join(KNOWN_DOC_TYPES)})")
    else:
        print(f"No pages need manual classification ({manual_path} written, header only).")


if __name__ == "__main__":
    main()
