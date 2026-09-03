"""
Moves files flagged for pruning (default: data/flagged_for_pruning.csv,
columns: bucket, file_path, category, reason) into a review folder
instead of deleting them - a safety buffer so a human can eyeball the
candidates before anything is permanently lost. Never deletes anything
itself, and never touches the flagged CSV.

Also removes the matching row from every reference CSV that still
points at a moved file's old working/ path (manifest.csv + every
bucket CSV) - otherwise those files would be left pointing at a path
that no longer exists there. All of those are git-tracked (for the
pre-migration data/ layout) or otherwise recoverable from the run's own
history, so a row removal that needs undoing is not the same risk as
the actual image files (not git-tracked, hence the review-folder-not-
delete design for those).

RunContext-aware (2026-08-08 repo/workspace restructuring, see
docs/RUN_ARCHITECTURE.md): by default operates against the
legacy_pre_run_system run - the one that currently holds the real
bucket/manifest data (--run-id targets a different, real run instead).
RunContext.resume() rejects the legacy run outright, since it's marked
completed/immutable by design (core/run_context.py) - same situation
core/classifier.py's BUCKET_DIR/_ctx_from_manifest_path already handle,
mirrored here via _resolve_run_paths() rather than reinventing a
different fallback. Falls back further to the pre-migration data/
layout on a machine that hasn't run the migration at all yet.

GATED behind a single warning + interactive y/n confirmation before
anything is moved or removed (per Jon's direction, 2026-07-30) - shows
the full plan (file count, destination, exactly which reference CSVs
lose how many rows) first. Pass --yes to skip the prompt (e.g. for
scripted use); --dry-run shows the same plan and exits without
touching anything, prompt or not.

Safe to re-run: every move is appended to an audit manifest
(<review-dir>/prune_review_manifest.csv - original path, bucket,
category, reason, new path, timestamp) and already-moved files (found
in that manifest) are skipped on a later run, so a repeat run after a
newer corpus sweep only moves the NEW candidates.

Usage:
    python scripts/move_flagged_for_pruning.py --dry-run   # preview first
    python scripts/move_flagged_for_pruning.py              # prompts, then moves
    python scripts/move_flagged_for_pruning.py --yes        # no prompt
    python scripts/move_flagged_for_pruning.py --run-id 20260808T193050927883_ec3556b6

To restore a file: read prune_review_manifest.csv's new_path/
original_path columns and move it back by hand (or write a small
restore script pointed at that manifest - not built here since nothing
has needed restoring yet). Reference-CSV rows would need re-adding by
hand too.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.workspace_context import WorkspaceContext
from core.run_context import RunContext

DEFAULT_CSV = PROJECT_ROOT / "data" / "flagged_for_pruning.csv"
MANIFEST_FIELDS = ["moved_at", "original_path", "bucket", "category", "reason", "new_path"]

LEGACY_RUN_ID = "legacy_pre_run_system"


def _resolve_run_paths(run_id: str | None) -> tuple[Path, Path, Path, Path]:
    """Returns (bucket_dir, manifest_csv, quarantine_dir, working_images_dir)
    for run_id (or the legacy run if None) - see module docstring for
    why this doesn't just call RunContext.resume() unconditionally."""
    workspace = WorkspaceContext.resolve()
    target_run_id = run_id or LEGACY_RUN_ID
    try:
        ctx = RunContext.resume(workspace, target_run_id)
        return ctx.buckets, ctx.manifest_csv, ctx.quarantine, ctx.working_images
    except (FileNotFoundError, ValueError):
        run_root = workspace.runs_root / target_run_id
        if run_root.exists():
            return (run_root / "outputs" / "buckets", run_root / "manifest" / "manifest.csv",
                    run_root / "quarantine", run_root / "working" / "images")
        return (PROJECT_ROOT / "data" / "buckets", PROJECT_ROOT / "data" / "manifest.csv",
                PROJECT_ROOT / "data", PROJECT_ROOT / "data" / "working")


def _resolve_source(file_path_str: str, working_images_dir: Path) -> Path | None:
    """Resolves a flagged CSV row's file_path to a real, existing file on
    disk. Tries the literal path first (covers both a pre-migration
    machine and any freshly-written flagged CSV that already has an
    up-to-date path). Falls back to rebasing a stale pre-2026-08-08
    `.../data/working/<name>` path onto the resolved run's own
    working_images dir (the exact rename the restructuring actually
    performed - see docs/RUN_ARCHITECTURE.md's Phase 1) - flagged CSVs
    written before that migration have this exact staleness baked in,
    confirmed directly against real data (data/flagged_bad_deskew.csv).
    Returns None if neither resolves to a real file."""
    literal = Path(file_path_str)
    if literal.exists():
        return literal
    legacy_working = PROJECT_ROOT / "data" / "working"
    try:
        rel = literal.resolve().relative_to(legacy_working.resolve())
    except ValueError:
        return None
    rebased = working_images_dir / rel
    return rebased if rebased.exists() else None


def _plan(csv_path: Path, review_dir: Path, working_images_dir: Path) -> tuple[list[dict], list[tuple[str, str]]]:
    """Pure planning pass, no filesystem writes - returns (rows to move, skipped)."""
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    manifest_path = review_dir / "prune_review_manifest.csv"
    already_moved: set[str] = set()
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8", newline="") as f:
            already_moved = {r["original_path"] for r in csv.DictReader(f)}

    to_move, skipped = [], []
    for row in rows:
        source_str = row["file_path"]
        if source_str in already_moved:
            skipped.append((source_str, "already moved (in manifest)"))
            continue
        if _resolve_source(source_str, working_images_dir) is None:
            skipped.append((source_str, "source not found (checked literal path and the "
                                          "post-migration working/images/ rebase)"))
            continue
        to_move.append(row)
    return to_move, skipped


def _count_reference_rows(paths: set[str], reference_csvs: list[Path]) -> dict[Path, int]:
    counts = {}
    for csv_path in reference_csvs:
        if not csv_path.exists():
            continue
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        counts[csv_path] = sum(1 for r in rows if r.get("file_path") in paths)
    return counts


def _remove_reference_rows(paths: set[str], reference_csvs: list[Path]) -> dict[Path, int]:
    removed_counts = {}
    for csv_path in reference_csvs:
        if not csv_path.exists():
            continue
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)
        kept = [r for r in rows if r.get("file_path") not in paths]
        n_removed = len(rows) - len(kept)
        removed_counts[csv_path] = n_removed
        if n_removed == 0:
            continue
        tmp_path = csv_path.with_suffix(".csv.tmp")
        with open(tmp_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept)
        tmp_path.replace(csv_path)
    return removed_counts


def move_flagged(csv_path: Path, review_dir: Path, reference_csvs: list[Path], working_images_dir: Path,
                  dry_run: bool = False, assume_yes: bool = False) -> None:
    to_move, skipped = _plan(csv_path, review_dir, working_images_dir)
    flagged_paths = {row["file_path"] for row in to_move}
    ref_counts = _count_reference_rows(flagged_paths, reference_csvs)
    total_ref_rows = sum(ref_counts.values())

    print(f"Plan: {len(to_move)} file(s) to move, {len(skipped)} skipped.")
    print(f"Destination: {review_dir}")
    print("Reference CSVs that will lose matching rows:")
    for csv_p, n in ref_counts.items():
        if n:
            print(f"  {csv_p}: {n} row(s)")
    if not total_ref_rows:
        print("  (none)")

    if dry_run:
        print("\n[dry-run] Nothing moved, nothing removed.")
        for path, reason in skipped[:20]:
            print(f"  skip {path}: {reason}")
        return

    if not to_move:
        print("\nNothing to do.")
        return

    if not assume_yes:
        print(f"\nWARNING: this will MOVE {len(to_move)} file(s) out of their current working "
              f"location into {review_dir} (not deleted - stays on disk, reversible by hand) AND "
              f"PERMANENTLY REMOVE {total_ref_rows} row(s) from the "
              f"{sum(1 for n in ref_counts.values() if n)} reference CSV(s) listed above (no "
              f"file-level undo for that removal beyond the audit manifest below and, for the "
              f"pre-migration data/ layout only, `git diff`/`git checkout`).")
        answer = input("Proceed? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Aborted - nothing changed.")
            return

    manifest_path = review_dir / "prune_review_manifest.csv"
    manifest_exists = manifest_path.exists()
    moved_records = []
    actually_moved_paths: set[str] = set()

    for row in to_move:
        source = _resolve_source(row["file_path"], working_images_dir)
        if source is None:
            # _plan() already filtered to only resolvable rows - a race
            # (file moved/deleted between planning and here) is the only
            # way this triggers. Skip rather than crash shutil.move on a
            # None path.
            print(f"  SKIPPED (source vanished since planning): {row['file_path']}")
            continue
        bucket_dir = review_dir / row["bucket"]
        dest = bucket_dir / source.name
        if dest.exists():
            stem, suffix, i = dest.stem, dest.suffix, 2
            while dest.exists():
                dest = bucket_dir / f"{stem}__{i}{suffix}"
                i += 1
        bucket_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(dest))
        actually_moved_paths.add(row["file_path"])
        moved_records.append({
            "moved_at": datetime.now(timezone.utc).isoformat(),
            "original_path": row["file_path"],
            "bucket": row["bucket"],
            "category": row["category"],
            "reason": row["reason"],
            "new_path": str(dest),
        })

    review_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if not manifest_exists:
            writer.writeheader()
        writer.writerows(moved_records)

    removed_counts = _remove_reference_rows(actually_moved_paths, reference_csvs)

    print(f"\nMoved: {len(moved_records)}")
    print(f"Audit manifest: {manifest_path}")
    print("Reference rows removed:")
    for csv_p, n in removed_counts.items():
        if n:
            print(f"  {csv_p}: {n} row(s)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=str, default=str(DEFAULT_CSV))
    parser.add_argument("--run-id", type=str, default=None,
                         help=f"Operate against this run's bucket/manifest CSVs instead of the "
                              f"default ({LEGACY_RUN_ID!r} - where the real data currently lives). "
                              f"Must be a real, resumable (not-yet-completed) run for anything "
                              f"other than the default.")
    parser.add_argument("--review-dir", type=str, default=None,
                         help="Default: <resolved run>/quarantine/pending_prune_review.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Preview only - no files moved, no CSV rows removed.")
    parser.add_argument("--yes", "-y", action="store_true",
                         help="Skip the interactive confirmation prompt.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        sys.exit(1)

    bucket_dir, manifest_csv, quarantine_dir, working_images_dir = _resolve_run_paths(args.run_id)
    print(f"Bucket dir:     {bucket_dir}")
    print(f"Manifest:       {manifest_csv}")
    print(f"Working images: {working_images_dir}")
    reference_csvs = [manifest_csv, *sorted(bucket_dir.glob("*.csv"))] if bucket_dir.exists() else [manifest_csv]
    review_dir = Path(args.review_dir) if args.review_dir else quarantine_dir / "pending_prune_review"

    move_flagged(csv_path, review_dir, reference_csvs, working_images_dir,
                 dry_run=args.dry_run, assume_yes=args.yes)


if __name__ == "__main__":
    main()
