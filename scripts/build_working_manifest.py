"""
[Spans new Stage 0/1/3, see docs/PIPELINE_STAGE_TERMINOLOGY.md - not yet split]
Stage 0 CLI: copy a folder of raw scans into a working directory, deskew +
preprocess every copy, and write data/manifest.csv pointing at those
working copies - see core/manifest_pipeline.py's module docstring for the
full pipeline-stage story. Originals are only ever read, never modified.

Supersedes scripts/build_manifest.py's role for the real pipeline.
scripts/build_manifest.py is now RETIRED and ARCHIVED to
scripts/archive/build_manifest.py (Jon, 2026-07-30: "build_manifest was
the initial starting module. its legacy now and wont be used going
forward") - nothing in this project should reference it going forward;
this script and scripts/run_preprocessing.py (the CSV-input adapter over
the same engine, for a multi-source file selection rather than one
folder) are the real Stage 0 entry points. Once classification+manual
dewarp are done (see core/manifest_pipeline.py), run
scripts/finalize_manifest.py.

Usage:
    python scripts/build_working_manifest.py                  # folder picker
    python scripts/build_working_manifest.py --source-folder "J:\\path\\to\\scans"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from tkinter import Tk, filedialog

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.manifest_pipeline import (
    build_working_manifest, DEFAULT_WORKING_DIR, DEFAULT_MANIFEST_PATH,
    DEFAULT_PREPROCESSING_PROFILE,
)
from core.workspace_context import WorkspaceContext
from core.run_context import RUN_TYPES, RunContext


def pick_folder() -> Path | None:
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(title="Select folder of raw scans")
    root.destroy()
    return Path(folder) if folder else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-folder", type=str, default=None,
                         help="Folder of raw scans to process. Omit to use a folder picker dialog.")
    parser.add_argument("--working-dir", type=str, default=str(DEFAULT_WORKING_DIR),
                         help=f"Working directory for copies (default: {DEFAULT_WORKING_DIR})")
    parser.add_argument("--manifest-path", type=str, default=str(DEFAULT_MANIFEST_PATH),
                         help=f"Output manifest CSV path (default: {DEFAULT_MANIFEST_PATH})")
    parser.add_argument("--preprocessing-profile", type=str, default=DEFAULT_PREPROCESSING_PROFILE,
                         help=f"core/image_preprocessing.py PREPROCESSING_PROFILES key "
                              f"(default: {DEFAULT_PREPROCESSING_PROFILE!r})")
    parser.add_argument("--new-run", action="store_true",
                         help="Create a fresh hashed run under genealogy_workspace/runs/ "
                              "(see docs/RUN_ARCHITECTURE.md) instead of writing into the "
                              "legacy default working_dir/manifest_path above - the "
                              "preferred mode going forward. --working-dir/--manifest-path "
                              "are ignored when this is set.")
    parser.add_argument("--run-name", default=None, help="Optional human-readable run name (--new-run only).")
    parser.add_argument("--run-type", default="production", choices=sorted(RUN_TYPES),
                         help="Run type (--new-run only, default: production).")
    args = parser.parse_args()

    if args.source_folder:
        folder = Path(args.source_folder)
    else:
        folder = pick_folder()
        if folder is None:
            print("No folder selected. Exiting.")
            sys.exit(0)

    if not folder.exists():
        print(f"Folder does not exist: {folder}")
        sys.exit(1)

    ctx = None
    if args.new_run:
        workspace = WorkspaceContext.resolve()
        ctx = RunContext.create(
            workspace, run_type=args.run_type, source_input=str(folder), run_name=args.run_name,
        )
        print(f"Run: {ctx.run_id}" + (f" ({args.run_name})" if args.run_name else ""))
        print(f"Run root: {ctx.run_root}")

    try:
        build_working_manifest(
            source_folder=folder,
            working_dir=Path(args.working_dir),
            manifest_path=Path(args.manifest_path),
            preprocessing_profile=args.preprocessing_profile,
            ctx=ctx,
        )
    except FileNotFoundError as e:
        if ctx is not None:
            ctx.mark_failed(str(e))
        print(f"ERROR: {e}")
        sys.exit(1)
    else:
        if ctx is not None:
            ctx.mark_completed()


if __name__ == "__main__":
    main()
