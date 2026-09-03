"""
Automated sidecar generation CLI - the automated-sidecar-generation
branch's experiment. Runs core/auto_sidecar.py's full rule-based
pipeline (classify -> template -> table boundary -> header region ->
data rows) against a dewarped page image and writes a sidecar JSON in
the SAME format core/row_segmentation.py's build_sidecar() already
produces - Stage 1/2 OCR and ui/row_segmentation_ui.py's per-column
masking step consume it completely unmodified.

Input is a DEWARPED image (e.g. data/outputs/dewarped/<name>_dewarped.jpg,
from ui/dewarp_preprocessor_ui.py) - this tool only replaces the manual
row-definition step, not perspective dewarping, same division of labor
the existing manual workflow already has.

Usage:
    python scripts/auto_generate_sidecar.py <dewarped_image.jpg> \\
        [--out PATH] [--run-id <hashed_run_id>] [--debug] [--force]

Default output: <legacy_or_flat_root>/outputs/auto_row_segmentation/
<stem>_sidecar.json - a SEPARATE directory from .../row_segmentation/
(where manual sidecars live), specifically so this experimental tool
can never collide with or silently overwrite real manually-segmented
work. Resolves into the legacy_pre_run_system run once
docs/RUN_ARCHITECTURE.md's workspace migration has run, else the
pre-migration data/outputs/ layout - see DEFAULT_OUT_DIR below. --out
overrides the path directly; --run-id resumes a real hashed run
explicitly, writing under ctx.outputs/'auto_row_segmentation' instead
(ignored if --out is also given); --force allows overwriting an
existing file at the resolved path (refused by default either way).

--debug additionally writes <stem>_auto_debug_overlay.png (green=table
boundary, blue=header exclusion zone, red=each detected row) and prints
classification + boundary + row-count diagnostics to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.auto_sidecar import generate_auto_sidecar
from core.row_segmentation import save_sidecar
from core.workspace_context import WorkspaceContext
from core.run_context import RunContext

# Legacy fallback - see scripts/run_batch_auto_sidecar.py's identical
# pattern/comment. Real hashed runs should pass --run-id instead of
# relying on this module-level constant.
_legacy_root = WorkspaceContext.resolve().runs_root / "legacy_pre_run_system"
if _legacy_root.exists():
    DEFAULT_OUT_DIR = _legacy_root / "outputs" / "auto_row_segmentation"
else:
    DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "outputs" / "auto_row_segmentation"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image_path", type=str, help="Path to a dewarped page image")
    parser.add_argument("--out", type=str, default=None,
                         help=f"Output sidecar JSON path. Default: "
                              f"{DEFAULT_OUT_DIR}/<stem>_sidecar.json")
    parser.add_argument("--run-id", type=str, default=None,
                         help="Resume a real hashed run (docs/RUN_ARCHITECTURE.md) explicitly - "
                              "writes under ctx.outputs/'auto_row_segmentation' instead of the "
                              "legacy/flat default above. Ignored if --out is also given.")
    parser.add_argument("--debug", action="store_true",
                         help="Write a debug overlay PNG and print classification/"
                              "boundary/row diagnostics.")
    parser.add_argument("--force", action="store_true",
                         help="Overwrite the output sidecar if it already exists "
                              "(refused by default).")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"ERROR: image not found: {image_path}")
        sys.exit(1)

    stem = image_path.stem
    if args.out:
        out_path = Path(args.out)
    elif args.run_id:
        ctx = RunContext.resume(WorkspaceContext.resolve(), args.run_id)
        print(f"Continuing run {ctx.run_id}")
        out_path = ctx.outputs / "auto_row_segmentation" / f"{stem}_sidecar.json"
    else:
        out_path = DEFAULT_OUT_DIR / f"{stem}_sidecar.json"

    if out_path.exists() and not args.force:
        print(f"ERROR: output already exists: {out_path}\n"
              f"Refusing to overwrite without --force (this tool never touches "
              f"data/outputs/row_segmentation/, where manually-confirmed sidecars "
              f"live, but it also won't silently clobber its own prior output).")
        sys.exit(1)

    print(f"Image: {image_path}")
    result = generate_auto_sidecar(str(image_path), debug=args.debug)

    c = result.classification
    print(f"\nClassification: doc_type={c.doc_type!r} confidence={c.confidence:.2f}")
    if args.debug:
        print(f"  Features: {result.diagnostics.get('classification_features')}")
        print(f"  Scores:   {result.diagnostics.get('classification_scores')}")

    if result.sidecar is None:
        print("\nNo sidecar generated - document classified as 'unknown'. "
              "See the classification scores above; either this page isn't one of "
              "the known templates, or the structural signals were too ambiguous.")
        for w in result.warnings:
            print(f"  {w}")
        sys.exit(2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_sidecar(result.sidecar, out_path)
    print(f"\nSidecar written: {out_path}")
    print(f"  Table bbox:  {result.sidecar['table_bbox']}")
    print(f"  Header bbox: {result.sidecar['header_bbox']}")
    print(f"  Rows:        {len(result.sidecar['rows'])}")
    print(f"  Columns:     {result.sidecar['column_order']}")

    needs_review = result.sidecar.get("rows_needs_review", [])
    if needs_review:
        print(f"\n  {len(needs_review)} row(s) quarantined to rows_needs_review "
              f"(NOT in 'rows', will be skipped by automated OCR):")
        for r in needs_review:
            print(f"    row {r['index']}: {r['reason']}")

    if result.warnings:
        print("\nWarnings:")
        for w in result.warnings:
            print(f"  {w}")

    if args.debug and result.debug_overlay is not None:
        overlay_path = out_path.with_name(out_path.stem.replace("_sidecar", "") + "_auto_debug_overlay.png")
        result.debug_overlay.save(overlay_path)
        print(f"\nDebug overlay written: {overlay_path}")
        print("\nDiagnostics:")
        print(json.dumps(result.diagnostics, indent=2, default=str))


if __name__ == "__main__":
    main()
