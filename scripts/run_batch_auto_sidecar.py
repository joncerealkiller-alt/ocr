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
        [--run-id <hashed_run_id>] [--debug] [--force]

--input/--out-dir default to the legacy_pre_run_system run's paths once
docs/RUN_ARCHITECTURE.md's workspace migration has run, else the
pre-migration data/ layout - see the module-level DEFAULT_INPUT/
DEFAULT_OUT_DIR resolution below. --run-id resumes a real hashed run
explicitly (overriding --out-dir to ctx.outputs/'auto_row_segmentation'
and scoping DB lookups by that run); omitted, it's auto-detected from
--input/--out-dir when either already points inside a run directory.
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
from core.pipeline_db import PipelineDatabase, DEFAULT_DB_PATH
from core.workspace_context import WorkspaceContext
from core.run_context import RunContext

# Legacy fallback - see core/manifest_pipeline.py's identical comment
# and core/classifier.py's BUCKET_DIR for the same pattern. Resolves
# into the legacy_pre_run_system run once the workspace migration has
# run (docs/RUN_ARCHITECTURE.md); falls back to the pre-migration
# data/ layout until then. Real hashed runs should pass --run-id (or
# have it auto-detected, see _ctx_from_paths() below) instead of
# relying on these module-level constants - they exist ONLY so this
# script still does something sensible when invoked without a run.
_legacy_root = WorkspaceContext.resolve().runs_root / "legacy_pre_run_system"
if _legacy_root.exists():
    DEFAULT_INPUT = _legacy_root / "outputs" / "buckets" / "dense_tabular_rows_subtype.csv"
    DEFAULT_OUT_DIR = _legacy_root / "outputs" / "auto_row_segmentation"
else:
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


def _resolve_image(db: PipelineDatabase, file_path: str, run_id: str | None = None) -> dict | None:
    """
    Resolves the DB image row for a subtype-CSV row's file_path, whether
    or not that image has since been dewarped. A plain get_image_by_path()
    only matches while working_path still equals file_path (true before
    dewarp); once dewarp happens, working_path moves to the new output
    path and file_path (the pre-dewarp path) is only recoverable via the
    "stage2a_dewarp" stage_outputs row's lookup_key - see find_dewarped()
    below, which needs the identical resolution. Returns None if this
    file was never registered in the DB at all (an older workflow, or a
    path outside Stage 0's acquisition).

    run_id (docs/RUN_ARCHITECTURE.md): passed through to
    get_image_by_path() so two different runs' same-named files never
    collide in the lookup - None (the pre-run-system/legacy default)
    behaves exactly as before.
    """
    image = db.get_image_by_path(file_path, run_id=run_id)
    if image is not None:
        return image
    stage_output = db.find_stage_output("stage2a_dewarp", file_path)
    if stage_output is None:
        return None
    return db.get_image(stage_output["image_id"])


def find_dewarped(db: PipelineDatabase, file_path: str) -> Path | None:
    """
    Looks up this file's dewarped output via pipeline_db instead of
    globbing data/outputs/dewarped/ by filename convention (2026-08-03
    consolidation pass: pipeline_db already tracks dewarp completion -
    globbing the filesystem was reconstructing state that already has
    an authoritative owner, and the previous filename-convention version
    had already broken once for real, see the incident this replaced -
    a bare `*_dewarped.*` glob matched ui/dewarp_preprocessor_ui.py's
    OWN sidecar JSON alongside the image and picked whichever sorted
    first alphabetically, silently skipping every real image on a
    2026-07-29 batch run).

    Deliberately does NOT use _resolve_image() above - that helper also
    matches a NOT-YET-dewarped image (get_image_by_path() still finds it
    via its unchanged working_path), which would be wrong here: this
    function specifically means "has this been dewarped," not "is this
    image known to the DB at all."

    Returns None if this file hasn't been dewarped yet (or isn't known
    to the DB at all) - same "nothing to find yet" contract as before.
    """
    stage_output = db.find_stage_output("stage2a_dewarp", file_path)
    if stage_output is None:
        return None
    image = db.get_image(stage_output["image_id"])
    return Path(image["working_path"]) if image is not None else None


def _record_auto_sidecar_outcome(
    db: PipelineDatabase, file_path: str, status: str, note: str = "", run_id: str | None = None,
) -> None:
    """
    Records this run's per-image outcome as a stage_outputs event
    (stage="stage4_auto_sidecar") - 2026-08-03 consolidation pass. This
    module had ZERO pipeline_db wiring before this: its review-worthy
    outcomes (a CV-fallback-guessed template, quarantined rows, an
    unknown doc_type) were visible only inside this run's own CSV
    outputs (batch_run_summary.csv, needs_manual_classification.csv),
    never queryable pipeline-wide the way Stage 5's uncertain_review
    state already is. No new decision is made here - status values
    directly mirror outcomes this script already computes (see
    SUMMARY_FIELDS's "status" column) plus one new value, "needs_review",
    for an otherwise-successful sidecar (status "ok" in the CSV) that
    still has quarantined rows or an unconfirmed CV-guessed template -
    without this, that page reads as fully "ok" to anything querying the
    DB alone, hiding exactly the cases this module exists to flag.

    Appends one row per call (stage_outputs is append-only by design,
    per core/pipeline_db.py's own docstring) - re-running this script
    with --force naturally produces a second, later event rather than
    overwriting the first, preserving the full history.

    A file_path not resolvable to a DB image is skipped with a printed
    note rather than raised - same tolerance as every other stage's DB
    wiring in this project (a row from an older workflow Stage 0 never
    registered should not abort the whole batch).
    """
    image = _resolve_image(db, file_path, run_id=run_id)
    if image is None:
        print(f"  (DB: {file_path!r} not registered - stage_outputs event skipped)")
        return
    db.record_stage_output(
        image["id"], stage="stage4_auto_sidecar",
        status=status, note=(note[:300] if note else None),
    )


def _ctx_from_path(path: Path) -> RunContext | None:
    """Best-effort: if path sits under <runs_root>/<run_id>/..., resume
    that run and return its RunContext - same auto-detection trick as
    core/classifier.py's _ctx_from_manifest_path(), adapted since this
    script's own paths (--input/--out-dir) aren't shaped like a
    manifest.csv. Returns None (not an error) for any path outside a
    real run directory - e.g. the legacy/flat defaults above - so this
    script still works unchanged when no run is in play."""
    try:
        workspace = WorkspaceContext.resolve()
        resolved = path.resolve()
        runs_root = workspace.runs_root.resolve()
        if runs_root not in resolved.parents:
            return None
        run_id = resolved.relative_to(runs_root).parts[0]
        return RunContext.resume(workspace, run_id)
    except (IndexError, FileNotFoundError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT),
                         help="Gemma subtype-classification CSV to drive this batch from.")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR),
                         help="Where sidecars (and --debug overlays) are written.")
    parser.add_argument("--run-id", type=str, default=None,
                         help="Resume a real hashed run (docs/RUN_ARCHITECTURE.md) explicitly - "
                              "overrides --out-dir to ctx.outputs/'auto_row_segmentation' and "
                              "scopes DB lookups by this run. Auto-detected from --input/--out-dir "
                              "when omitted, if either already points inside a run directory.")
    parser.add_argument("--debug", action="store_true",
                         help="Also write a debug overlay PNG per file.")
    parser.add_argument("--force", action="store_true",
                         help="Overwrite an existing sidecar output (refused by default, "
                              "same collision-avoidance rule scripts/auto_generate_sidecar.py uses).")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH),
                         help=f"core/pipeline_db.py database path (default: {DEFAULT_DB_PATH})")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)

    ctx = None
    if args.run_id:
        ctx = RunContext.resume(WorkspaceContext.resolve(), args.run_id)
    else:
        ctx = _ctx_from_path(out_dir) or _ctx_from_path(input_path)
    if ctx is not None:
        print(f"Continuing run {ctx.run_id}")
        if args.out_dir == str(DEFAULT_OUT_DIR):
            out_dir = ctx.outputs / "auto_row_segmentation"

    if not input_path.exists():
        print(f"ERROR: input not found: {input_path}")
        sys.exit(1)

    db = PipelineDatabase(Path(args.db_path))
    run_id = ctx.run_id if ctx is not None else None

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
            reason = f"upstream classifier returned {doc_type!r} - no template to run against"
            queue_for_manual(summary, reason=reason, dewarped=find_dewarped(db, file_path))
            _record_auto_sidecar_outcome(db, file_path, "needs_review", note=reason, run_id=run_id)
            summary_rows.append(summary)
            continue

        dewarped_path = find_dewarped(db, file_path)
        if dewarped_path is None:
            summary["status"] = "skipped_not_dewarped"
            print("  -> SKIP: no dewarped image found yet")
            skipped_no_dewarp += 1
            _record_auto_sidecar_outcome(db, file_path, "skipped_not_dewarped", run_id=run_id)
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
                reason = "no sidecar produced: " + "; ".join(result.warnings)[:200]
                queue_for_manual(
                    summary, reason=reason, dewarped=dewarped_path,
                    cv_guess=result.classification.doc_type,
                    cv_confidence=f"{result.classification.confidence:.2f}",
                )
                _record_auto_sidecar_outcome(db, file_path, "needs_review", note=reason, run_id=run_id)
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

            # DB visibility for the two review-worthy conditions above -
            # a page can be summary["status"]=="ok" (a sidecar WAS
            # produced) while still needing a human look, and that
            # nuance was previously only visible by opening this run's
            # own CSVs. "done" only when NEITHER condition applies.
            if result.used_cv_fallback or quarantined:
                reasons = []
                if result.used_cv_fallback:
                    reasons.append("template chosen by CV fallback (untrusted)")
                if quarantined:
                    reasons.append(f"{quarantined}/{len(result.sidecar['rows']) + quarantined} row(s) quarantined")
                _record_auto_sidecar_outcome(db, file_path, "needs_review", note="; ".join(reasons), run_id=run_id)
            else:
                _record_auto_sidecar_outcome(db, file_path, "done", run_id=run_id)
        except Exception as e:
            summary["status"] = "error"
            summary["error"] = str(e)[:300]
            print(f"  -> FAILED: {e}")
            failed += 1
            queue_for_manual(summary, reason=f"sidecar generation failed: {str(e)[:200]}",
                             dewarped=dewarped_path)
            _record_auto_sidecar_outcome(db, file_path, "error", note=str(e)[:200], run_id=run_id)

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
