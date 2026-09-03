"""
One-time path-rewrite pass (2026-08-08) following the genealogy_workspace
migration (docs/RUN_ARCHITECTURE.md): Phase 1 moved data/working/ into
genealogy_workspace/runs/legacy_pre_run_system/working/images/ with NO
junction left behind (unlike Phase 2's data/outputs/ treatment), so every
cached artifact from this session's hidden-state-probe/error-analysis
work that stored an absolute "...\\data\\working\\<filename>" path is now
stale. Verified before writing this: 1749/2232 review_table.csv paths
(78.4%) failed Path.exists(), ALL of them under data/working - the other
483 (under data/outputs/lac_pull_*) were unaffected since Phase 1 never
touched that directory.

This rewrites the path STRINGS only - never touches the embedding
tensors themselves (x/x_by_loc), which are content-derived and were
never invalid. Per the design docs' own "DO NOT delete-and-regenerate"
guidance, this is the recommended fix instead of re-running Gemma
inference.

LEGACY_WORKING_IMAGES_DIR (defined once here, also mirrored into
common.py) is the single source of truth for where this specific legacy
corpus's images live post-migration - any FUTURE code that needs to
resolve one of these legacy paths should import it from common.py
rather than re-deriving or hardcoding data/working again.

Idempotent: rewriting a file that's already been rewritten (path already
under the new prefix) is a no-op for that file - safe to re-run.

Usage:
    python diagnostics/error_analysis/rewrite_legacy_paths.py --dry-run
    python diagnostics/error_analysis/rewrite_legacy_paths.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from diagnostics.error_analysis.common import OLD_WORKING_PREFIX, LEGACY_WORKING_IMAGES_DIR

SHARD_DIRS = [
    PROJECT_ROOT / "data" / "outputs" / "gemma_hidden_state_probe" / "shards",
    PROJECT_ROOT / "data" / "outputs" / "gemma_hidden_state_probe" / "shards_stage2",
]
REVIEW_TABLE_CSV = PROJECT_ROOT / "data" / "outputs" / "error_analysis" / "review_table.csv"
FAILURE_ANNOTATIONS_JSON = PROJECT_ROOT / "data" / "outputs" / "error_analysis" / "failure_annotations.json"
PHASE4_DIR = PROJECT_ROOT / "data" / "outputs" / "error_analysis" / "phase4_embedding_stats"
PHASE5_DIR = PROJECT_ROOT / "data" / "outputs" / "error_analysis" / "phase5_cross_layer_agreement"
GEMMA_GEN_PREDICTIONS_JSON = (
    PROJECT_ROOT / "data" / "outputs" / "vit_family_benchmark" / "gemma_flat8" / "gemma_flat8_test_predictions.json"
)

NEW_PREFIX_STR = str(LEGACY_WORKING_IMAGES_DIR)


def remap(path_str: str) -> str:
    """Rewrites a stale 'data\\working\\<filename>' path to the new
    location, filename unchanged. Any path NOT under OLD_WORKING_PREFIX
    (e.g. the 483 data/outputs/lac_pull_* rows, unaffected by the
    migration) is returned unchanged."""
    if not path_str.startswith(OLD_WORKING_PREFIX):
        return path_str
    filename = path_str[len(OLD_WORKING_PREFIX):].lstrip("\\/")
    return str(LEGACY_WORKING_IMAGES_DIR / filename)


def rewrite_csv(path: Path, dry_run: bool) -> tuple[int, int]:
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    n_changed = 0
    for row in rows:
        if "path" in row:
            new_path = remap(row["path"])
            if new_path != row["path"]:
                row["path"] = new_path
                n_changed += 1

    if not dry_run and n_changed > 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return len(rows), n_changed


def rewrite_shard_pt(path: Path, dry_run: bool) -> tuple[int, int]:
    data = torch.load(path)
    if "paths" not in data:
        return 0, 0
    n_changed = 0
    new_paths = []
    for p in data["paths"]:
        new_p = remap(p)
        if new_p != p:
            n_changed += 1
        new_paths.append(new_p)
    if not dry_run and n_changed > 0:
        data["paths"] = new_paths
        torch.save(data, path)
    return len(data["paths"]), n_changed


def rewrite_failure_annotations(path: Path, dry_run: bool) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    entries = json.loads(path.read_text(encoding="utf-8"))
    n_changed = 0
    for e in entries:
        if "path" in e:
            new_p = remap(e["path"])
            if new_p != e["path"]:
                e["path"] = new_p
                n_changed += 1
    if not dry_run and n_changed > 0:
        path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return len(entries), n_changed


def rewrite_gemma_gen_predictions(path: Path, dry_run: bool) -> tuple[int, int]:
    """Keyed BY path (dict keys), not a path-valued field - rebuild the
    dict with remapped keys rather than mutating in place."""
    if not path.exists():
        return 0, 0
    data = json.loads(path.read_text(encoding="utf-8"))
    preds = data["predictions"]
    n_changed = 0
    new_preds = {}
    for old_key, v in preds.items():
        new_key = remap(old_key)
        if new_key != old_key:
            n_changed += 1
        new_preds[new_key] = v
    if not dry_run and n_changed > 0:
        data["predictions"] = new_preds
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return len(preds), n_changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Report what would change without writing.")
    args = parser.parse_args()

    print(f"Old prefix: {OLD_WORKING_PREFIX}")
    print(f"New prefix: {NEW_PREFIX_STR}")
    print(f"{'DRY RUN - no files will be modified' if args.dry_run else 'LIVE - files will be rewritten'}\n")

    total_files, total_changed_files = 0, 0

    # -- shard .pt files (both Stage 1 and Stage 2 caches) --
    for shard_dir in SHARD_DIRS:
        if not shard_dir.exists():
            print(f"  (skip, not found) {shard_dir}")
            continue
        for shard_path in sorted(shard_dir.glob("*.pt")):
            n_rows, n_changed = rewrite_shard_pt(shard_path, args.dry_run)
            total_files += 1
            if n_changed > 0:
                total_changed_files += 1
                print(f"  {shard_path.relative_to(PROJECT_ROOT)}: {n_changed}/{n_rows} paths remapped")

    # -- review_table.csv --
    if REVIEW_TABLE_CSV.exists():
        n_rows, n_changed = rewrite_csv(REVIEW_TABLE_CSV, args.dry_run)
        total_files += 1
        if n_changed > 0:
            total_changed_files += 1
        print(f"\n{REVIEW_TABLE_CSV.relative_to(PROJECT_ROOT)}: {n_changed}/{n_rows} paths remapped")

    # -- phase4 PCA 2D projection CSVs --
    if PHASE4_DIR.exists():
        for csv_path in sorted(PHASE4_DIR.glob("*_pca2d.csv")):
            n_rows, n_changed = rewrite_csv(csv_path, args.dry_run)
            total_files += 1
            if n_changed > 0:
                total_changed_files += 1
                print(f"  {csv_path.relative_to(PROJECT_ROOT)}: {n_changed}/{n_rows} paths remapped")

    # -- phase5 per-image behavior + disagreement sets --
    if PHASE5_DIR.exists():
        per_image_path = PHASE5_DIR / "per_image_behavior.csv"
        if per_image_path.exists():
            n_rows, n_changed = rewrite_csv(per_image_path, args.dry_run)
            total_files += 1
            if n_changed > 0:
                total_changed_files += 1
                print(f"  {per_image_path.relative_to(PROJECT_ROOT)}: {n_changed}/{n_rows} paths remapped")
        disagreement_dir = PHASE5_DIR / "disagreement_sets"
        if disagreement_dir.exists():
            for csv_path in sorted(disagreement_dir.glob("*.csv")):
                n_rows, n_changed = rewrite_csv(csv_path, args.dry_run)
                total_files += 1
                if n_changed > 0:
                    total_changed_files += 1

    # -- failure_annotations.json --
    n_rows, n_changed = rewrite_failure_annotations(FAILURE_ANNOTATIONS_JSON, args.dry_run)
    total_files += 1
    if n_changed > 0:
        total_changed_files += 1
        print(f"\n{FAILURE_ANNOTATIONS_JSON.relative_to(PROJECT_ROOT)}: {n_changed}/{n_rows} paths remapped")

    # -- gemma_flat8_test_predictions.json (dict-keyed by path) --
    n_rows, n_changed = rewrite_gemma_gen_predictions(GEMMA_GEN_PREDICTIONS_JSON, args.dry_run)
    total_files += 1
    if n_changed > 0:
        total_changed_files += 1
        print(f"{GEMMA_GEN_PREDICTIONS_JSON.relative_to(PROJECT_ROOT)}: {n_changed}/{n_rows} keys remapped")

    print(f"\n{'Would modify' if args.dry_run else 'Modified'} {total_changed_files}/{total_files} files.")


if __name__ == "__main__":
    main()
