"""
Moves files flagged for pruning (default: data/flagged_for_pruning.csv,
columns: bucket, file_path, category, reason) into a review folder
instead of deleting them - a safety buffer so a human can eyeball the
candidates before anything is permanently lost. Never deletes anything
itself, and never touches the flagged CSV.

Also removes the matching row from every reference CSV that still
points at a moved file's old data/working/ path (data/manifest.csv +
every data/buckets/*.csv) - otherwise those files would be left
pointing at a path that no longer exists there. All of those are
git-tracked, so `git diff`/`git checkout` is the recovery path for a
row removal that needs undoing - unlike the actual image files (not
git-tracked, hence the review-folder-not-delete design for those).

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

To restore a file: read prune_review_manifest.csv's new_path/
original_path columns and move it back by hand (or write a small
restore script pointed at that manifest - not built here since nothing
has needed restoring yet). Reference-CSV rows would need re-adding by
hand too, or via `git checkout -- <path>` if no other change has
touched that file since.
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

DEFAULT_CSV = PROJECT_ROOT / "data" / "flagged_for_pruning.csv"
DEFAULT_REVIEW_DIR = PROJECT_ROOT / "data" / "pending_prune_review"
MANIFEST_FIELDS = ["moved_at", "original_path", "bucket", "category", "reason", "new_path"]

REFERENCE_CSVS = [
    PROJECT_ROOT / "data" / "manifest.csv",
    *sorted((PROJECT_ROOT / "data" / "buckets").glob("*.csv")),
]


def _plan(csv_path: Path, review_dir: Path) -> tuple[list[dict], list[tuple[str, str]]]:
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
        if not Path(source_str).exists():
            skipped.append((source_str, "source not found"))
            continue
        to_move.append(row)
    return to_move, skipped


def _count_reference_rows(paths: set[str]) -> dict[Path, int]:
    counts = {}
    for csv_path in REFERENCE_CSVS:
        if not csv_path.exists():
            continue
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        counts[csv_path] = sum(1 for r in rows if r.get("file_path") in paths)
    return counts


def _remove_reference_rows(paths: set[str]) -> dict[Path, int]:
    removed_counts = {}
    for csv_path in REFERENCE_CSVS:
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


def move_flagged(csv_path: Path, review_dir: Path, dry_run: bool = False, assume_yes: bool = False) -> None:
    to_move, skipped = _plan(csv_path, review_dir)
    flagged_paths = {row["file_path"] for row in to_move}
    ref_counts = _count_reference_rows(flagged_paths)
    total_ref_rows = sum(ref_counts.values())

    print(f"Plan: {len(to_move)} file(s) to move, {len(skipped)} skipped.")
    print(f"Destination: {review_dir}")
    print("Reference CSVs that will lose matching rows:")
    for csv_p, n in ref_counts.items():
        if n:
            print(f"  {csv_p.relative_to(PROJECT_ROOT)}: {n} row(s)")
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
        print(f"\nWARNING: this will MOVE {len(to_move)} file(s) out of data/working/ into "
              f"{review_dir} (not deleted - stays on disk, reversible by hand) AND PERMANENTLY "
              f"REMOVE {total_ref_rows} row(s) from the {sum(1 for n in ref_counts.values() if n)} "
              f"reference CSV(s) listed above (git-tracked, `git diff`/`git checkout` recovers "
              f"them if needed).")
        answer = input("Proceed? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Aborted - nothing changed.")
            return

    manifest_path = review_dir / "prune_review_manifest.csv"
    manifest_exists = manifest_path.exists()
    moved_records = []
    actually_moved_paths: set[str] = set()

    for row in to_move:
        source = Path(row["file_path"])
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

    removed_counts = _remove_reference_rows(actually_moved_paths)

    print(f"\nMoved: {len(moved_records)}")
    print(f"Audit manifest: {manifest_path}")
    print("Reference rows removed:")
    for csv_p, n in removed_counts.items():
        if n:
            print(f"  {csv_p.relative_to(PROJECT_ROOT)}: {n} row(s)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=str, default=str(DEFAULT_CSV))
    parser.add_argument("--review-dir", type=str, default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--dry-run", action="store_true",
                         help="Preview only - no files moved, no CSV rows removed.")
    parser.add_argument("--yes", "-y", action="store_true",
                         help="Skip the interactive confirmation prompt.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        sys.exit(1)

    move_flagged(csv_path, Path(args.review_dir), dry_run=args.dry_run, assume_yes=args.yes)


if __name__ == "__main__":
    main()
