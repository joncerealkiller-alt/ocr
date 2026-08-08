"""
One-time migration: moves the pre-run-system data/ layout (data/working,
data/raw_from_pdf, data/manifest.csv, data/buckets, data/pipeline.db)
into a real synthetic run - workspace_root/runs/legacy_pre_run_system/ -
and rewrites pipeline.db so every row (old and new alike) obeys the
same run_id + run-relative-path invariant. See docs/RUN_ARCHITECTURE.md
and the approved plan (Repo/Workspace Split + Hashed Run System) for the
full design rationale.

SAFETY: operates entirely against the --data-root/--db-path/
--workspace-root arguments you pass it, and never deletes the source
data/ directory itself (files are copied, never moved, until you
explicitly pass --delete-source-after-verify). Run it once against
scratch copies (see the "Validating against copies" section in this
docstring) before ever pointing it at the real data/ directory.

Validating against copies (do this first):
    python scripts/migrate_pipeline_db_to_run_schema.py \\
        --data-root /path/to/a/COPY/of/data \\
        --workspace-root /path/to/a/scratch/genealogy_workspace \\
        --db-path /path/to/a/COPY/of/data/pipeline.db

Real cutover (only after the copy-based validation above passes and
tasks 5/8/tests/checksums are all done - see the approved plan):
    python scripts/migrate_pipeline_db_to_run_schema.py \\
        --data-root data \\
        --workspace-root ../genealogy_workspace \\
        --db-path data/pipeline.db

What moves where (see docs/RUN_ARCHITECTURE.md's mapping table):
    data/working/<name>.<ext>            -> runs/legacy_pre_run_system/working/images/<name>.<ext>
    data/working/<name>_analysis.json    -> runs/legacy_pre_run_system/working/analysis/<name>_analysis.json
    data/working/<name>_preprocess.json  -> runs/legacy_pre_run_system/working/analysis/<name>_preprocess.json
    data/raw_from_pdf/*                  -> runs/legacy_pre_run_system/raw_from_pdf/*
    data/manifest.csv, manifest_provenance.json, manifest_final.csv
                                          -> runs/legacy_pre_run_system/manifest/*
    data/buckets/*.csv                   -> runs/legacy_pre_run_system/outputs/buckets/*.csv
    data/pipeline.db                     -> workspace_root/pipeline.db (schema-migrated copy)

DB rewrite: every images row gets run_id="legacy_pre_run_system" and its
working_path/source_path rewritten to the run-relative form above (an
external source_path, e.g. a path outside data/ entirely, is left
unchanged - it isn't a run-owned artifact). Every stage_outputs row's
sidecar_path/lookup_key gets the same treatment when it looks like a
path under data/ (manifest_provenance.json, *_analysis.json,
*_preprocess.json, a bucket CSV, or a working-directory image path used
as a lookup_key); a lookup_key that isn't a path (e.g. an identity_hash
string, used by stage1_baseline_embeddings/stage1_layout_detections) is
left untouched.
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
from pathlib import Path

from core.pipeline_db import PipelineDatabase, _SCHEMA, _COLUMN_MIGRATIONS  # noqa: E402
from core.run_context import RunContext
from core.workspace_context import WorkspaceContext

LEGACY_RUN_ID = "legacy_pre_run_system"

ANALYSIS_SUFFIXES = ("_analysis.json", "_preprocess.json")


def _classify_and_relativize(path_str: str, data_root: Path) -> tuple[str, bool]:
    """Returns (new_value, is_run_owned). is_run_owned=False means the
    value is left completely unchanged (external path, or not a path at
    all - e.g. an identity_hash lookup_key)."""
    if not path_str:
        return path_str, False

    p = Path(path_str)
    if not p.is_absolute():
        return path_str, False  # already relative / not a filesystem path (e.g. a hash)

    try:
        rel_to_data = p.relative_to(data_root)
    except ValueError:
        return path_str, False  # outside data/ - an external original, leave as-is

    parts = rel_to_data.parts
    if not parts:
        return path_str, False

    top = parts[0]
    if top == "working" and len(parts) == 2:
        name = parts[1]
        if name.endswith(ANALYSIS_SUFFIXES):
            return str(Path("working") / "analysis" / name), True
        return str(Path("working") / "images" / name), True
    if top == "raw_from_pdf":
        return str(Path("raw_from_pdf", *parts[1:])), True
    if top == "buckets":
        return str(Path("outputs") / "buckets" / Path(*parts[1:])), True
    if top in ("manifest.csv", "manifest_provenance.json", "manifest_final.csv"):
        return str(Path("manifest") / top), True

    return path_str, False


def _copy_working_dir(data_root: Path, run_root: Path) -> None:
    working_src = data_root / "working"
    images_dst = run_root / "working" / "images"
    analysis_dst = run_root / "working" / "analysis"
    images_dst.mkdir(parents=True, exist_ok=True)
    analysis_dst.mkdir(parents=True, exist_ok=True)
    if not working_src.exists():
        print(f"  (no {working_src}, skipping)")
        return
    n_images, n_sidecars = 0, 0
    for f in working_src.iterdir():
        if not f.is_file():
            continue
        if f.name.endswith(ANALYSIS_SUFFIXES):
            shutil.copy2(f, analysis_dst / f.name)
            n_sidecars += 1
        else:
            shutil.copy2(f, images_dst / f.name)
            n_images += 1
    print(f"  working/: {n_images} image(s), {n_sidecars} sidecar(s)")


def _copy_tree_flat(src: Path, dst: Path, label: str) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        print(f"  (no {src}, skipping)")
        return
    n = 0
    for f in src.rglob("*"):
        if f.is_file():
            rel = f.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
            n += 1
    print(f"  {label}: {n} file(s)")


def _copy_manifest_and_buckets(data_root: Path, run_root: Path) -> None:
    manifest_dst = run_root / "manifest"
    manifest_dst.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.csv", "manifest_provenance.json", "manifest_final.csv"):
        src = data_root / name
        if src.exists():
            shutil.copy2(src, manifest_dst / name)
            print(f"  manifest/{name}")

    buckets_src = data_root / "buckets"
    buckets_dst = run_root / "outputs" / "buckets"
    buckets_dst.mkdir(parents=True, exist_ok=True)
    if buckets_src.exists():
        n = 0
        for f in buckets_src.glob("*.csv"):
            shutil.copy2(f, buckets_dst / f.name)
            n += 1
        print(f"  outputs/buckets/: {n} file(s)")


def migrate_data_files(data_root: Path, workspace: WorkspaceContext, source_input: str) -> RunContext:
    ctx = RunContext.create_legacy_migration_run(workspace, source_input=source_input)
    print(f"Legacy run root: {ctx.run_root}")
    print("Copying working/ ...")
    _copy_working_dir(data_root, ctx.run_root)
    print("Copying raw_from_pdf/ ...")
    _copy_tree_flat(data_root / "raw_from_pdf", ctx.raw_from_pdf, "raw_from_pdf")
    print("Copying manifest + buckets ...")
    _copy_manifest_and_buckets(data_root, ctx.run_root)
    return ctx


def migrate_database(src_db_path: Path, dst_db_path: Path, source_data_root: Path) -> dict:
    """Copies src_db_path to dst_db_path, then rewrites the copy in
    place: adds run_id (via PipelineDatabase's own column-migration
    machinery - dst_db_path is opened as a real PipelineDatabase first
    so _COLUMN_MIGRATIONS runs), backfills run_id=LEGACY_RUN_ID for
    every row, and relativizes every path-shaped column. Returns a
    before/after report dict for the caller to verify."""
    dst_db_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_db_path, dst_db_path)

    # Opening as PipelineDatabase runs _init_schema(), which applies
    # _COLUMN_MIGRATIONS (adds run_id if missing) - safe/idempotent.
    PipelineDatabase(dst_db_path)

    conn = sqlite3.connect(dst_db_path)
    conn.row_factory = sqlite3.Row
    try:
        images_before = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        stage_outputs_before = conn.execute("SELECT COUNT(*) FROM stage_outputs").fetchone()[0]

        images = conn.execute("SELECT id, source_path, working_path FROM images").fetchall()
        for row in images:
            new_working, _ = _classify_and_relativize(row["working_path"], source_data_root)
            new_source, _ = _classify_and_relativize(row["source_path"], source_data_root)
            conn.execute(
                "UPDATE images SET run_id = ?, working_path = ?, source_path = ? WHERE id = ?",
                (LEGACY_RUN_ID, new_working, new_source, row["id"]),
            )

        stage_outputs = conn.execute(
            "SELECT id, sidecar_path, lookup_key FROM stage_outputs"
        ).fetchall()
        for row in stage_outputs:
            new_sidecar, _ = _classify_and_relativize(row["sidecar_path"], source_data_root) if row["sidecar_path"] else (row["sidecar_path"], False)
            new_lookup, _ = _classify_and_relativize(row["lookup_key"], source_data_root) if row["lookup_key"] else (row["lookup_key"], False)
            conn.execute(
                "UPDATE stage_outputs SET sidecar_path = ?, lookup_key = ? WHERE id = ?",
                (new_sidecar, new_lookup, row["id"]),
            )

        conn.commit()

        images_after = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        stage_outputs_after = conn.execute("SELECT COUNT(*) FROM stage_outputs").fetchone()[0]
        run_tagged = conn.execute(
            "SELECT COUNT(*) FROM images WHERE run_id = ?", (LEGACY_RUN_ID,)
        ).fetchone()[0]
        still_absolute = conn.execute(
            "SELECT COUNT(*) FROM images WHERE working_path LIKE '/%' OR working_path LIKE '_:%'"
        ).fetchone()[0]
    finally:
        conn.close()

    return {
        "images_before": images_before,
        "images_after": images_after,
        "stage_outputs_before": stage_outputs_before,
        "stage_outputs_after": stage_outputs_after,
        "run_tagged": run_tagged,
        "working_path_still_absolute": still_absolute,
    }


def verify(dst_db_path: Path, ctx: RunContext, sample_size: int = 25) -> None:
    """Spot-checks that resolve_image_path() on a sample of migrated
    rows points at files that actually exist post-copy."""
    db = PipelineDatabase(dst_db_path)
    images = db.list_images()
    sample = images[:sample_size]
    missing = []
    for image in sample:
        resolved = db.resolve_image_path(image)
        if not resolved.exists():
            missing.append((image["id"], image["working_path"], resolved))
    print(f"Verified {len(sample)} sample row(s) resolve to real files "
          f"({len(missing)} missing).")
    for image_id, working_path, resolved in missing[:10]:
        print(f"  MISSING id={image_id} working_path={working_path!r} -> {resolved}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True,
                         help="The data/ directory to physically copy files FROM.")
    parser.add_argument("--source-data-root", type=Path, default=None,
                         help="The data/ directory PATH PREFIX as it appears inside the DB's "
                              "stored source_path/working_path/sidecar_path values - used only "
                              "for string matching, not for reading files. Defaults to "
                              "--data-root. Pass this separately when --data-root is a scratch "
                              "COPY (validation runs) so paths still match against the DB's "
                              "original absolute-path prefix, e.g. J:\\...\\genealogy_pipeline\\data.")
    parser.add_argument("--workspace-root", type=Path, required=True, help="Target genealogy_workspace/ directory.")
    parser.add_argument("--db-path", type=Path, required=True, help="Source pipeline.db to migrate.")
    args = parser.parse_args(argv)

    data_root = args.data_root.resolve()
    source_data_root = (args.source_data_root or args.data_root).resolve()
    workspace_root = args.workspace_root.resolve()
    db_path = args.db_path.resolve()

    import dataclasses
    workspace = dataclasses.replace(WorkspaceContext.resolve(), workspace_root=workspace_root)

    print(f"Source data root (files copied from): {data_root}")
    print(f"Source data root (DB path-prefix match): {source_data_root}")
    print(f"Target workspace root: {workspace_root}")
    print(f"Source DB: {db_path}")
    print()

    ctx = migrate_data_files(data_root, workspace, source_input=str(data_root))

    print()
    print("Migrating database...")
    report = migrate_database(db_path, workspace.pipeline_db_path, source_data_root)
    for k, v in report.items():
        print(f"  {k}: {v}")

    assert report["images_before"] == report["images_after"], "row count changed during migration!"
    assert report["stage_outputs_before"] == report["stage_outputs_after"], "stage_outputs row count changed!"
    assert report["run_tagged"] == report["images_after"], "not every image row got tagged with run_id!"

    print()
    verify(workspace.pipeline_db_path, ctx)

    ctx.mark_completed()
    print()
    print("Migration complete (source data/ directory untouched - copies only).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
