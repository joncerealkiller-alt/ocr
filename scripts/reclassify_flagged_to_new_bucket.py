"""
Reclassifies files flagged as belonging in a new bucket (default:
data/flagged_needs_new_bucket.csv, columns: bucket, file_path, category,
reason - same schema as flagged_for_pruning.csv, per Jon's manual
review in ui/classifier_validation_ui.py) into that new bucket's CSV.

Unlike scripts/move_flagged_for_pruning.py, this does NOT move any
image files - these are still legitimate corpus entries, just filed
under the wrong category. Only the bucket CSV membership changes: each
flagged file's full row is moved OUT of its current (wrong) bucket CSV
and INTO the target bucket's CSV (created if it doesn't exist yet),
preserving every original column (confidence, text_density, handwriting,
table_layout, faces, map_like, model, prompt_version) except category,
which is overwritten to the target bucket. manifest.csv is untouched
(Stage 0 output, has no category column - these files are still
validly in the corpus, just reclassified within Stage 1's output).

RunContext-aware (2026-08-08 repo/workspace restructuring, see
docs/RUN_ARCHITECTURE.md): by default operates against the
legacy_pre_run_system run - the one that currently holds the real
bucket data (--run-id targets a different, real run instead).
RunContext.resume() rejects the legacy run outright, since it's marked
completed/immutable by design (core/run_context.py) - same situation
core/classifier.py's BUCKET_DIR/_ctx_from_manifest_path already handle,
mirrored here via _resolve_bucket_dir() rather than reinventing a
different fallback. Falls back further to the pre-migration data/
layout on a machine that hasn't run the migration at all yet.

GATED behind a single warning + interactive y/n confirmation before
anything is written, same pattern as move_flagged_for_pruning.py. Pass
--yes to skip the prompt; --dry-run shows the plan and exits without
touching anything.

Safe to re-run: a row already present in the target bucket CSV (by
file_path) is skipped, so a repeat run only picks up newly-flagged
files.

Usage:
    python scripts/reclassify_flagged_to_new_bucket.py website_screenshot --dry-run
    python scripts/reclassify_flagged_to_new_bucket.py website_screenshot
    python scripts/reclassify_flagged_to_new_bucket.py website_screenshot --run-id 20260808T193050927883_ec3556b6
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.workspace_context import WorkspaceContext
from core.run_context import RunContext

DEFAULT_CSV = PROJECT_ROOT / "data" / "flagged_needs_new_bucket.csv"
LEGACY_RUN_ID = "legacy_pre_run_system"


def _resolve_bucket_dir(run_id: str | None) -> Path:
    """See scripts/move_flagged_for_pruning.py's _resolve_run_paths() -
    identical reasoning, just the one path this script needs."""
    workspace = WorkspaceContext.resolve()
    target_run_id = run_id or LEGACY_RUN_ID
    try:
        ctx = RunContext.resume(workspace, target_run_id)
        return ctx.buckets
    except (FileNotFoundError, ValueError):
        run_root = workspace.runs_root / target_run_id
        if run_root.exists():
            return run_root / "outputs" / "buckets"
        return PROJECT_ROOT / "data" / "buckets"


def _read_csv(path: Path) -> tuple[list[str] | None, list[dict]]:
    if not path.exists():
        return None, []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def _plan(flagged_csv: Path, target_bucket: str,
          bucket_dir: Path) -> tuple[list[dict], list[tuple[str, str]], dict[str, list[str]]]:
    """
    Returns (rows to reclassify [full original-bucket row + resolved
    old_bucket], skipped[(path, reason)], old_bucket -> [file_paths still
    present there]) - pure planning, no writes.
    """
    _, flagged_rows = _read_csv(flagged_csv)

    target_path = bucket_dir / f"{target_bucket}.csv"
    _, target_rows = _read_csv(target_path)
    already_there = {r["file_path"] for r in target_rows}

    # Cache each old bucket CSV's rows once, keyed by file_path, so we
    # only read each one from disk a single time regardless of how many
    # flagged rows reference it.
    old_bucket_cache: dict[str, dict[str, dict]] = {}
    to_reclassify: list[dict] = []
    skipped: list[tuple[str, str]] = []

    for row in flagged_rows:
        path = row["file_path"]
        old_bucket = row["bucket"]
        if path in already_there:
            skipped.append((path, f"already in {target_bucket}.csv"))
            continue
        if old_bucket == target_bucket:
            skipped.append((path, "already flagged as the target bucket"))
            continue

        if old_bucket not in old_bucket_cache:
            _, rows = _read_csv(bucket_dir / f"{old_bucket}.csv")
            old_bucket_cache[old_bucket] = {r["file_path"]: r for r in rows}

        old_row = old_bucket_cache[old_bucket].get(path)
        if old_row is None:
            skipped.append((path, f"not found in {old_bucket}.csv (already reclassified elsewhere?)"))
            continue

        new_row = dict(old_row)
        new_row["category"] = target_bucket
        new_row["_old_bucket"] = old_bucket
        to_reclassify.append(new_row)

    by_old_bucket: dict[str, list[str]] = {}
    for row in to_reclassify:
        by_old_bucket.setdefault(row["_old_bucket"], []).append(row["file_path"])

    return to_reclassify, skipped, by_old_bucket


def reclassify(flagged_csv: Path, target_bucket: str, bucket_dir: Path,
                dry_run: bool = False, assume_yes: bool = False) -> None:
    to_reclassify, skipped, by_old_bucket = _plan(flagged_csv, target_bucket, bucket_dir)

    print(f"Plan: {len(to_reclassify)} file(s) to reclassify into {target_bucket}, "
          f"{len(skipped)} skipped.")
    print("From:")
    for old_bucket, paths in by_old_bucket.items():
        print(f"  {old_bucket}.csv: {len(paths)} row(s) removed")

    if dry_run:
        print("\n[dry-run] Nothing written.")
        for path, reason in skipped[:20]:
            print(f"  skip {path}: {reason}")
        return

    if not to_reclassify:
        print("\nNothing to do.")
        return

    if not assume_yes:
        print(f"\nWARNING: this will PERMANENTLY remove {len(to_reclassify)} row(s) from "
              f"{len(by_old_bucket)} bucket CSV(s) under {bucket_dir} and add them to "
              f"{bucket_dir / (target_bucket + '.csv')} (no file-level undo beyond, for the "
              f"pre-migration data/ layout only, `git diff`/`git checkout`). No image files are "
              f"moved or deleted.")
        answer = input("Proceed? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Aborted - nothing changed.")
            return

    # Remove reclassified rows from each old bucket CSV.
    moved_paths_by_old_bucket = {
        old_bucket: {p for p in paths} for old_bucket, paths in by_old_bucket.items()
    }
    for old_bucket, paths in moved_paths_by_old_bucket.items():
        old_path = bucket_dir / f"{old_bucket}.csv"
        fieldnames, rows = _read_csv(old_path)
        kept = [r for r in rows if r["file_path"] not in paths]
        tmp_path = old_path.with_suffix(".csv.tmp")
        with open(tmp_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept)
        tmp_path.replace(old_path)

    # Append to (or create) the target bucket CSV.
    target_path = bucket_dir / f"{target_bucket}.csv"
    target_fieldnames, target_rows = _read_csv(target_path)
    if target_fieldnames is None:
        # New bucket CSV - use the same column order as whichever source
        # bucket CSV the rows came from (all bucket CSVs share the same
        # schema - file_path,category,confidence,text_density,
        # handwriting,table_layout,faces,map_like,reason,model,prompt_version).
        target_fieldnames = [k for k in to_reclassify[0].keys() if k != "_old_bucket"]

    clean_rows = [{k: v for k, v in row.items() if k != "_old_bucket"} for row in to_reclassify]
    write_header = not target_path.exists()
    bucket_dir.mkdir(parents=True, exist_ok=True)
    with open(target_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=target_fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(clean_rows)

    print(f"\nReclassified: {len(to_reclassify)}")
    print(f"Target: {target_path}")
    for old_bucket, paths in by_old_bucket.items():
        print(f"  removed from {old_bucket}.csv: {len(paths)} row(s)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target_bucket", type=str,
                         help="The new/correct bucket name these flagged files should move to "
                              "(e.g. website_screenshot) - must be a valid DocumentCategory value.")
    parser.add_argument("--csv", type=str, default=str(DEFAULT_CSV))
    parser.add_argument("--run-id", type=str, default=None,
                         help=f"Operate against this run's bucket CSVs instead of the default "
                              f"({LEGACY_RUN_ID!r} - where the real data currently lives). Must "
                              f"be a real, resumable (not-yet-completed) run for anything other "
                              f"than the default.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Preview only - no CSV rows moved.")
    parser.add_argument("--yes", "-y", action="store_true",
                         help="Skip the interactive confirmation prompt.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        sys.exit(1)

    from core.schema import DocumentCategory
    valid_categories = {c.value for c in DocumentCategory}
    if args.target_bucket not in valid_categories:
        print(f"ERROR: {args.target_bucket!r} is not a known DocumentCategory value. "
              f"Known: {sorted(valid_categories)}")
        sys.exit(1)

    bucket_dir = _resolve_bucket_dir(args.run_id)
    print(f"Bucket dir: {bucket_dir}")

    reclassify(csv_path, args.target_bucket, bucket_dir, dry_run=args.dry_run, assume_yes=args.yes)


if __name__ == "__main__":
    main()
