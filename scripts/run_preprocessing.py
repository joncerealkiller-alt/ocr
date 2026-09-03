"""
[Spans new Stage 0/1/3, see docs/PIPELINE_STAGE_TERMINOLOGY.md - not yet split]
Stage 0 CLI, CSV-input adapter over core/manifest_pipeline.py's real
engine (build_working_manifest_from_paths()) - reads paths from a
"file_path" CSV column (core/bucket_worklist.py's load_bucket_filepaths(),
the same reader used throughout this project) rather than walking a
folder, so it can process an arbitrary, possibly multi-source file
selection - e.g. ui/build_manifest_ui.py's queued files, not just one
directory tree.

This is the CSV adapter core/manifest_pipeline.py's own docstring said
"can be added the same way whenever something needs it" - ui/build_
manifest_ui.py is that something (2026-07-30). scripts/build_working_
manifest.py (the folder-walking CLI) is untouched; this is a second,
independent CLI over the same engine, not a replacement for it.

scripts/build_manifest.py is LEGACY (Jon, 2026-07-30: "build_manifest
was the initial starting module. its legacy now and wont be used going
forward") - this script does NOT import it or reuse its CSV shape by
coincidence; core/bucket_worklist.py's "file_path" column convention is
the real, project-wide standard every other stage already reads/writes.

Usage:
    python scripts/run_preprocessing.py <input_csv> \\
        [--working-dir DIR] [--output-manifest PATH] [--preprocessing-profile NAME]

input_csv MAY have a second column, "skip_preprocess_if_sidecar_exists"
("true"/"false" per row, 2026-08-09) - per-row bypass tagging, as
opposed to the --skip-preprocess-if-sidecar-exists CLI flag below which
applies to every row uniformly. ui/build_manifest_ui.py writes this
column when its "Trust matching sidecars / bypass preprocessing"
checkbox tagged individual files at add-time - see core.manifest_
pipeline.stage3_preprocess_manifest()'s skip_if_sidecar_exists_names
docstring for the full mechanism. A plain "file_path"-only CSV (no
second column) still works exactly as before - core.bucket_worklist.
load_bucket_filepaths() only ever reads that one column, so this
script reads the CSV a second, independent way to pick up the optional
extra column without changing that shared reader's own contract.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.manifest_pipeline import (
    build_working_manifest_from_paths, DEFAULT_WORKING_DIR, DEFAULT_MANIFEST_PATH,
    DEFAULT_PREPROCESSING_PROFILE,
)
from core.bucket_worklist import load_bucket_filepaths
from core.pipeline_db import DEFAULT_DB_PATH


def _read_per_row_skip_names(input_csv: str) -> set[str]:
    """
    Reads input_csv's OPTIONAL "skip_preprocess_if_sidecar_exists"
    column directly (core.bucket_worklist.load_bucket_filepaths() only
    ever surfaces "file_path", by design - see that function's own
    docstring - so this is a second, independent read of the same file,
    not a change to that shared reader). Returns the set of basenames
    (Path(file_path).name) whose row was "true" - empty set if the
    column is absent, or the CSV can't be read a second time for any
    reason (never fatal; just means no per-row bypass requested).
    """
    try:
        with open(input_csv, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "skip_preprocess_if_sidecar_exists" not in reader.fieldnames:
                return set()
            return {
                Path(row["file_path"]).name
                for row in reader
                if row.get("file_path") and row.get("skip_preprocess_if_sidecar_exists", "").strip().lower() == "true"
            }
    except OSError:
        return set()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_csv", help='CSV with a "file_path" column - the files to preprocess.')
    parser.add_argument("--working-dir", default=str(DEFAULT_WORKING_DIR),
                         help=f"Working directory for copies (default: {DEFAULT_WORKING_DIR})")
    parser.add_argument("--output-manifest", default=str(DEFAULT_MANIFEST_PATH),
                         help=f"Output manifest CSV path (default: {DEFAULT_MANIFEST_PATH})")
    parser.add_argument("--preprocessing-profile", default=DEFAULT_PREPROCESSING_PROFILE,
                         help=f"core/image_preprocessing.py PREPROCESSING_PROFILES key "
                              f"(default: {DEFAULT_PREPROCESSING_PROFILE!r})")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH),
                         help=f"core/pipeline_db.py database path (default: {DEFAULT_DB_PATH})")
    parser.add_argument(
        "--skip-preprocess-if-sidecar-exists", action="store_true",
        help="Don't deskew/preprocess an image if a matching JSON sidecar already exists next "
             "to it - see core.manifest_pipeline._find_matching_sidecar_json(). For pointing "
             "this pipeline at an already-corrected folder (e.g. a column-calibration "
             "workspace's images/) without redoing work already done.",
    )
    args = parser.parse_args()

    paths = [Path(p) for p in load_bucket_filepaths(args.input_csv)]
    if not paths:
        print(f"No file_path rows found in {args.input_csv}")
        sys.exit(1)
    skip_names = _read_per_row_skip_names(args.input_csv)
    if skip_names:
        print(f"{len(skip_names)} file(s) tagged to bypass preprocessing if a matching sidecar exists.")

    try:
        build_working_manifest_from_paths(
            paths,
            working_dir=Path(args.working_dir),
            manifest_path=Path(args.output_manifest),
            preprocessing_profile=args.preprocessing_profile,
            db_path=Path(args.db_path),
            skip_preprocess_if_sidecar_exists=args.skip_preprocess_if_sidecar_exists,
            skip_preprocess_if_sidecar_exists_names=skip_names,
        )
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
