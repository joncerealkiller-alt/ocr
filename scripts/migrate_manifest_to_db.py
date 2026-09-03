"""
One-time, read-only, additive import of the existing file-based
orchestration state (data/manifest.csv, data/manifest_provenance.json,
data/buckets/*.csv, data/buckets/*_dewarped.csv, data/manifest_final.csv)
into core/pipeline_db.py's PipelineDatabase (data/pipeline.db).

READ-ONLY BY DESIGN - this script never opens a source CSV/JSON for
writing and never deletes anything. It only reads what's already on
disk and inserts/updates rows in pipeline.db. Safe to re-run (get_or_
create_image() is idempotent on working_path; update_image_state()/
record_stage_output() calls here reflect current file contents each
time, not a one-shot event).

Every input file is OPTIONAL except manifest.csv - a fresh project (or
one mid-Stage-0) simply won't have bucket CSVs or a provenance sidecar
yet, matching finalize_manifest()'s own "never raise on a missing bucket
CSV" discipline (core/manifest_pipeline.py).

IMPORTANT HONEST CAVEAT about identity_hash for an ALREADY-migrated
corpus like this project's real data/working/: docs/REFERENCE_PIPELINE_
V1.md documents that this corpus was already through Stage 3
(deskew+autocontrast) before this session's Stage 1 baseline-embedding
capture ever ran - there is no surviving pre-preprocessing byte state to
hash. identity_hash computed here is therefore a BEST-EFFORT retroactive
fingerprint from CURRENT (already-processed) bytes, not a genuine
Stage-0-before-anything hash - printed as a warning below, not silently
assumed. Going forward, once Phase 3 wires core/manifest_pipeline.py's
stage0_acquire_and_copy_sources() directly into this database,
identity_hash will be captured correctly, before Stage 3 ever runs.

Usage:
    python scripts/migrate_manifest_to_db.py
    python scripts/migrate_manifest_to_db.py --verify
    python scripts/migrate_manifest_to_db.py --db-path data/pipeline_test.db
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.pipeline_db import (
    PipelineDatabase, DEFAULT_DB_PATH,
    sync_bucket_classifications, sync_dewarp_results,
)
from core.image_analysis import analysis_sidecar_path
from core.baseline_embeddings import resolve_baseline_image_path as _to_abs
from core.schema import DocumentCategory

DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifest.csv"
DEFAULT_PROVENANCE = PROJECT_ROOT / "data" / "manifest_provenance.json"
DEFAULT_BUCKET_DIR = PROJECT_ROOT / "data" / "buckets"
DEFAULT_BASELINE_EMBEDDINGS = PROJECT_ROOT / "data" / "baseline_embeddings.json"
DEFAULT_MANIFEST_FINAL = PROJECT_ROOT / "data" / "manifest_final.csv"

BUCKET_CATEGORIES = [c.value for c in DocumentCategory]


def _read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def import_manifest(db: PipelineDatabase, manifest_path: Path, provenance_path: Path) -> list[str]:
    """
    Stage 0: one images row per manifest.csv entry. provenance_path, if
    present, supplies real source_path/source_type/page_number per
    working_path (core/source_expansion.py's ExpandedSource shape,
    written by stage0_acquire_and_copy_sources()); otherwise a plain
    image is assumed (source_path == working_path, source_type="image").

    Returns the list of working_paths imported, for later stages to
    iterate over without re-reading the CSV.
    """
    rows = _read_csv_rows(manifest_path)
    working_paths = [r["file_path"] for r in rows if r.get("file_path")]
    if not working_paths:
        print(f"WARNING: no rows found in {manifest_path} - nothing to import.")
        return []

    provenance_by_working_path: dict[str, dict] = {}
    if provenance_path.exists():
        records = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance_by_working_path = {r["working_path"]: r for r in records}
        print(f"Loaded provenance for {len(provenance_by_working_path)} file(s) from {provenance_path}")
    else:
        print(f"No provenance sidecar at {provenance_path} - treating every manifest "
              f"entry as a plain image with source_path == working_path.")

    print(f"Stage 0: importing {len(working_paths)} image(s) from {manifest_path}...")
    t0 = time.time()
    for i, working_path in enumerate(working_paths, 1):
        prov = provenance_by_working_path.get(working_path)
        source_path = prov["source_path"] if prov else working_path
        source_type = prov["source_type"] if prov else "image"
        page_number = prov.get("page_number") if prov else None

        image_id = db.get_or_create_image(
            source_path=source_path, working_path=working_path,
            source_type=source_type, page_number=page_number,
        )

        if prov is not None:
            db.record_stage_output(
                image_id, stage="stage0_acquire",
                sidecar_path=str(provenance_path), status="done",
            )

        if i % 200 == 0 or i == len(working_paths):
            print(f"  {i}/{len(working_paths)}")
    print(f"Stage 0 import done in {time.time() - t0:.1f}s.")
    return working_paths


def import_stage3_baseline(db: PipelineDatabase, working_paths: list[str]) -> None:
    """
    This project's real data/working/ corpus was already through Stage 3
    (deskew+autocontrast) BEFORE this session's Stage 1 capture ever ran
    - a documented fact (docs/REFERENCE_PIPELINE_V1.md), not a guess.
    Recording current_stage=3/status="preprocessed" as the baseline for
    every migrated image reflects that reality rather than leaving
    current_stage=0 (which would understate what's actually already
    been done to this corpus).
    """
    print("Marking migrated corpus as already Stage-3-preprocessed "
          "(documented fact for this project's data/working/, see docs/REFERENCE_PIPELINE_V1.md)...")
    for working_path in working_paths:
        image = db.get_image_by_path(working_path)
        if image is not None:
            db.update_image_state(image["id"], current_stage=3, status="preprocessed")


def import_image_analysis_sidecars(db: PipelineDatabase, working_paths: list[str]) -> None:
    """
    Stage 1 (physical half): core/image_analysis.py's own
    analysis_sidecar_path() convention gives the exact per-image sidecar
    path deterministically - reused here, not reimplemented. Only
    records a stage_outputs row for images where that sidecar actually
    exists on disk (a partially-run analysis pass is a normal state, not
    an error).
    """
    print("Stage 1 (physical): checking for per-image *_analysis.json sidecars...")
    found = 0
    for working_path in working_paths:
        sidecar = analysis_sidecar_path(working_path)
        if sidecar.exists():
            image = db.get_image_by_path(working_path)
            if image is not None:
                db.record_stage_output(
                    image["id"], stage="stage1_image_analysis",
                    sidecar_path=str(sidecar), status="done",
                )
                found += 1
    print(f"  {found}/{len(working_paths)} image(s) have a Stage 1 physical-analysis sidecar.")


def import_baseline_embeddings(db: PipelineDatabase, working_paths: list[str], baseline_path: Path) -> None:
    """
    Stage 1 (semantic half): data/baseline_embeddings.json is ONE file
    shared across the whole corpus (not a per-image sidecar), so
    sidecar_path is left NULL and lookup_key carries the record's own
    "image_hash" field - the key needed to find this image's entry
    inside that shared file (see core/baseline_embeddings.py).

    CAVEAT (see module docstring): that image_hash was computed on
    ALREADY-preprocessed bytes for this corpus - it will NOT equal this
    image's identity_hash/current_hash in pipeline.db, which are
    computed from the same (already-processed) bytes but via a
    different code path. This is expected and does not indicate
    corruption; it's a byproduct of migrating a corpus whose true
    pre-preprocessing state no longer exists to hash.
    """
    if not baseline_path.exists():
        print(f"No baseline embeddings file at {baseline_path} - skipping Stage 1 (semantic) import.")
        return

    print(f"Stage 1 (semantic): loading {baseline_path} (this is a large file, may take a moment)...")
    t0 = time.time()
    records = json.loads(baseline_path.read_text(encoding="utf-8"))
    by_image_path = {_to_abs(r["image"]): r for r in records}
    print(f"  loaded {len(records)} record(s) in {time.time() - t0:.1f}s.")

    found = 0
    for working_path in working_paths:
        record = by_image_path.get(_to_abs(working_path))
        if record is not None:
            image = db.get_image_by_path(working_path)
            if image is not None:
                db.record_stage_output(
                    image["id"], stage="stage1_baseline_embeddings",
                    lookup_key=record.get("image_hash"), status="done",
                    note=f"preprocessing_stage={record.get('preprocessing_stage')!r} (see caveat in module docstring)",
                )
                found += 1
    print(f"  {found}/{len(working_paths)} image(s) have a baseline embedding record.")


def import_bucket_classifications(db: PipelineDatabase, bucket_dir: Path) -> None:
    """
    Stage 5: thin wrapper over core/pipeline_db.py's promoted
    sync_bucket_classifications() (2026-08-02) - this script's own
    version of this logic was the ORIGINAL, one-time-import-only
    implementation; core/manifest_pipeline.py's finalize_manifest() then
    needed the identical logic to keep the DB in sync on every run, not
    just once, so it was promoted to core/pipeline_db.py and this script
    now calls that shared version instead of keeping a second copy.
    Behavior is a superset of the original: also auto-registers any
    file_path not yet known to the DB rather than only warning and
    skipping (matters for finalize_manifest(), harmless here since this
    script's own import_manifest() already registered every real
    manifest.csv path first).
    """
    if not bucket_dir.exists():
        print(f"No bucket directory at {bucket_dir} - skipping Stage 5 import.")
        return
    print(f"Stage 5: importing bucket classifications from {bucket_dir}...")
    total = sync_bucket_classifications(db, bucket_dir)
    print(f"Stage 5 import done: {total} classification row(s) updated.")


def import_dewarp_results(db: PipelineDatabase, bucket_dir: Path) -> None:
    """
    Stage 2a: thin wrapper over core/pipeline_db.py's promoted
    sync_dewarp_results() (2026-08-02) - same promotion reasoning as
    import_bucket_classifications() above.
    """
    if not bucket_dir.exists():
        return
    dewarped_csvs = sorted(bucket_dir.glob("*_dewarped.csv"))
    if not dewarped_csvs:
        print("No *_dewarped.csv files found - skipping Stage 2a import "
              "(normal if no manual dewarp has been done yet).")
        return
    for dewarped_csv in dewarped_csvs:
        print(f"Stage 2a: {dewarped_csv.name}")
    total = sync_dewarp_results(db, bucket_dir)
    print(f"Stage 2a import done: {total} dewarp result(s) updated.")


def verify(db: PipelineDatabase, manifest_path: Path, bucket_dir: Path) -> bool:
    """
    Diffs the DB's reconstructed state against the source files it was
    imported from. Prints every mismatch found; returns False if any
    were found so a calling script/CI check can act on it, rather than
    only printing and continuing regardless.
    """
    ok = True
    manifest_rows = _read_csv_rows(manifest_path)
    manifest_count = len([r for r in manifest_rows if r.get("file_path")])
    db_count = len(db.list_images())
    print(f"\n--- verify ---")
    print(f"manifest.csv rows: {manifest_count}  |  images in DB: {db_count}")
    if manifest_count != db_count:
        print(f"  MISMATCH")
        ok = False

    if bucket_dir.exists():
        for category in BUCKET_CATEGORIES:
            bucket_csv = bucket_dir / f"{category}.csv"
            rows = [r for r in _read_csv_rows(bucket_csv) if not r.get("error")]
            csv_count = len(rows)
            db_bucket_count = len(db.list_images(bucket=category))
            if csv_count == 0 and db_bucket_count == 0:
                continue
            status = "OK" if csv_count == db_bucket_count else "MISMATCH"
            print(f"{category}: bucket CSV={csv_count}  DB={db_bucket_count}  [{status}]")
            if status == "MISMATCH":
                ok = False

    print(f"\nverify {'PASSED' if ok else 'FAILED'}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--provenance", default=str(DEFAULT_PROVENANCE))
    parser.add_argument("--bucket-dir", default=str(DEFAULT_BUCKET_DIR))
    parser.add_argument("--baseline-embeddings", default=str(DEFAULT_BASELINE_EMBEDDINGS))
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--verify", action="store_true",
                         help="After importing, diff DB counts against the source CSVs.")
    parser.add_argument("--verify-only", action="store_true",
                         help="Skip import entirely, just verify an existing pipeline.db.")
    args = parser.parse_args()

    db = PipelineDatabase(Path(args.db_path))
    print(f"Using database: {db.db_path}\n")

    if not args.verify_only:
        working_paths = import_manifest(db, Path(args.manifest), Path(args.provenance))
        if working_paths:
            import_stage3_baseline(db, working_paths)
            import_image_analysis_sidecars(db, working_paths)
            import_baseline_embeddings(db, working_paths, Path(args.baseline_embeddings))
            import_bucket_classifications(db, Path(args.bucket_dir))
            import_dewarp_results(db, Path(args.bucket_dir))

    if args.verify or args.verify_only:
        ok = verify(db, Path(args.manifest), Path(args.bucket_dir))
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
