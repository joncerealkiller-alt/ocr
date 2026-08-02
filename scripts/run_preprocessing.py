"""
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
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.manifest_pipeline import (
    build_working_manifest_from_paths, DEFAULT_WORKING_DIR, DEFAULT_MANIFEST_PATH,
    DEFAULT_PREPROCESSING_PROFILE,
)
from core.bucket_worklist import load_bucket_filepaths


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
    args = parser.parse_args()

    paths = [Path(p) for p in load_bucket_filepaths(args.input_csv)]
    if not paths:
        print(f"No file_path rows found in {args.input_csv}")
        sys.exit(1)

    try:
        build_working_manifest_from_paths(
            paths,
            working_dir=Path(args.working_dir),
            manifest_path=Path(args.output_manifest),
            preprocessing_profile=args.preprocessing_profile,
        )
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
