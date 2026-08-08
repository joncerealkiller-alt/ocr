"""
Integration test: a real Stage 0-3 pipeline run threaded through
RunContext, against an isolated temp workspace (never the real
genealogy_workspace/ or data/ directories). Plain assert-based, no
pytest - matches tests/test_decision_engine.py's convention.

This captures, as a runnable regression test, the same manual
verification done throughout the hashed-run-migration session (see
docs/RUN_ARCHITECTURE.md): a fresh run's DB rows are run-relative and
run_id-scoped, two runs never collide, and a second run doesn't touch
the first run's files.

Test [1] keeps capture_baseline/capture_layout OFF (they load real
torch/timm vision encoders - too slow for a unit test) but test [4]
verifies the OUTPUT PATH ROUTING for those stages via a mock, without
loading real models: baseline_embeddings.json/postprocessing_
embeddings.json are run-owned (written under ctx.embeddings) when ctx
is given, never the fixed project-wide file - see
docs/RUN_ARCHITECTURE.md.

Usage:
    python tests/test_manifest_pipeline_run_context.py
"""

from __future__ import annotations

import dataclasses
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

import csv

from core.workspace_context import WorkspaceContext
from core.run_context import RunContext
from core.pipeline_db import PipelineDatabase, sync_bucket_classifications
import core.manifest_pipeline as mp
from core.manifest_pipeline import build_working_manifest_from_paths, build_working_manifest

_PASS = 0
_FAIL = 0


def check(condition: bool, description: str) -> None:
    global _PASS, _FAIL
    if condition:
        _PASS += 1
        print(f"  PASS: {description}")
    else:
        _FAIL += 1
        print(f"  FAIL: {description}")


def _make_source_images(dir_: Path, n: int) -> list[Path]:
    dir_.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        p = dir_ / f"page{i}.jpg"
        Image.new("RGB", (60, 60), color=(180, 180, 180)).save(p)
        paths.append(p)
    return paths


def test_stage0to3_run_from_paths():
    print("\n[1] build_working_manifest_from_paths() with ctx - full Stage 0-3")
    tmp = Path(tempfile.mkdtemp())
    try:
        src_dir = tmp / "sources"
        sources = _make_source_images(src_dir, 2)

        ws = dataclasses.replace(WorkspaceContext.resolve(), workspace_root=tmp / "workspace")
        ctx = RunContext.create(ws, run_type="diagnostic", source_input=str(src_dir))

        manifest_path = build_working_manifest_from_paths(
            sources,
            capture_baseline=False, capture_physical_sensors=True,
            capture_layout=False, run_decision_engine=True,
            ctx=ctx,
        )
        check(manifest_path == ctx.manifest_csv, "manifest written to ctx.manifest_csv")

        db = PipelineDatabase(ctx.workspace.pipeline_db_path)
        images = db.list_images()
        check(len(images) == 2, f"2 images registered in DB (got {len(images)})")

        for image in images:
            check(image["run_id"] == ctx.run_id, "row tagged with this run's run_id")
            check(not Path(image["working_path"]).is_absolute(), "working_path stored run-relative")
            resolved = db.resolve_image_path(image)
            check(resolved.exists(), f"resolved working_path exists: {resolved}")
            check(image["current_stage"] == 3, "current_stage advanced to 3 (preprocessed)")
            check(image["status"] == "preprocessed", "status == preprocessed")

            outs = db.get_stage_outputs(image["id"])
            stages = {o["stage"] for o in outs}
            check("stage0_acquire" in stages, "stage0_acquire recorded")
            check("stage1_image_analysis" in stages, "stage1_image_analysis recorded")
            check("stage2_decide_profile" in stages, "stage2_decide_profile recorded")
            check("stage3_preprocess" in stages, "stage3_preprocess recorded")
            for o in outs:
                if o["sidecar_path"]:
                    check(not Path(o["sidecar_path"]).is_absolute(),
                          f"stage_outputs.sidecar_path is run-relative ({o['stage']})")
                    check(db.resolve_path(ctx.run_id, o["sidecar_path"]).exists(),
                          f"stage_outputs.sidecar_path resolves to a real file ({o['stage']})")

        check(ctx.run_root.exists(), "run directory exists on disk")
        check((ctx.config_snapshot / "pipeline.yaml").exists(), "config snapshot captured pipeline.yaml")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_second_run_does_not_touch_first():
    print("\n[2] A second run never overwrites or reads the first run's files")
    tmp = Path(tempfile.mkdtemp())
    try:
        ws = dataclasses.replace(WorkspaceContext.resolve(), workspace_root=tmp / "workspace")

        src1 = tmp / "src1"
        sources1 = _make_source_images(src1, 1)
        ctx1 = RunContext.create(ws, run_type="diagnostic", source_input=str(src1))
        build_working_manifest_from_paths(
            sources1, capture_baseline=False, capture_physical_sensors=False,
            capture_layout=False, run_decision_engine=False, ctx=ctx1,
        )
        run1_image = next(ctx1.working_images.glob("*.jpg"))
        run1_bytes_before = run1_image.read_bytes()

        src2 = tmp / "src2"
        sources2 = _make_source_images(src2, 1)
        # Force the SAME source filename as run 1, to stress the collision path.
        renamed = sources2[0].with_name(run1_image.name)
        sources2[0].rename(renamed)
        ctx2 = RunContext.create(ws, run_type="diagnostic", source_input=str(src2))
        build_working_manifest_from_paths(
            [renamed], capture_baseline=False, capture_physical_sensors=False,
            capture_layout=False, run_decision_engine=False, ctx=ctx2,
        )

        check(ctx1.run_id != ctx2.run_id, "two runs got distinct run_ids")
        check(run1_image.exists() and run1_image.read_bytes() == run1_bytes_before,
              "run 1's working image is byte-identical after run 2 executes")
        run2_image = next(ctx2.working_images.glob("*.jpg"))
        check(run2_image != run1_image, "run 2's working image is a different file on disk")

        db = PipelineDatabase(ws.pipeline_db_path)
        images = db.list_images()
        check(len(images) == 2, f"both runs' images present in the DB (got {len(images)})")
        run_ids = {i["run_id"] for i in images}
        check(run_ids == {ctx1.run_id, ctx2.run_id}, "each row tagged with its own run's run_id")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_folder_adapter_forwards_ctx():
    print("\n[3] build_working_manifest() (folder adapter) forwards ctx correctly")
    # NOT exercised end-to-end here: build_working_manifest() has no
    # capture_baseline/capture_layout override and always defaults both
    # to True (a PRE-EXISTING property of this function, not introduced
    # by the hashed-run migration) - actually calling it would attempt
    # write_baseline_embeddings() against the real, 112MB data/
    # baseline_embeddings.json (an "immutable reference data" file per
    # project convention - see docs/RUN_ARCHITECTURE.md's "Known gaps"),
    # requires real torch/timm encoders, and is far too slow/risky for a
    # unit test. Verified instead by source inspection: it must forward
    # its ctx= argument straight through to build_working_manifest_from_
    # paths() rather than silently dropping it.
    import inspect
    source = inspect.getsource(build_working_manifest)
    check("ctx=ctx" in source, "build_working_manifest() forwards ctx=ctx to build_working_manifest_from_paths()")


def test_embeddings_are_run_owned():
    print("\n[4] baseline/layout/postprocessing embeddings route under ctx.embeddings, "
          "never the fixed shared file (no real models loaded - mocked)")
    tmp = Path(tempfile.mkdtemp())
    try:
        src_dir = tmp / "sources"
        sources = _make_source_images(src_dir, 1)

        ws = dataclasses.replace(WorkspaceContext.resolve(), workspace_root=tmp / "workspace")
        ctx = RunContext.create(ws, run_type="diagnostic", source_input=str(src_dir))

        manifest_path = mp.stage0_acquire_and_copy_sources(
            sources, db_path=ctx.workspace.pipeline_db_path, ctx=ctx,
        )

        calls = []

        def fake_write_baseline_embeddings(paths, output_path=None):
            calls.append(("write_baseline_embeddings", output_path))
            return {}

        def fake_write_layout_detections(paths, output_path=None):
            calls.append(("write_layout_detections", output_path))
            return {}

        with patch("core.baseline_embeddings.write_baseline_embeddings", side_effect=fake_write_baseline_embeddings), \
             patch("core.baseline_embeddings.write_layout_detections", side_effect=fake_write_layout_detections):
            mp.stage1_capture_baseline_embeddings(manifest_path, ctx=ctx)
            mp.stage1_capture_layout_detections(manifest_path, ctx=ctx)
            mp.stage4_capture_postprocessing_embeddings(manifest_path, ctx=ctx)

        for name, path in calls:
            check(str(ctx.run_root) in str(path), f"{name}'s output_path is under ctx.run_root ({path})")

        check(calls[0] == ("write_baseline_embeddings", ctx.embeddings / "baseline_embeddings.json"),
              "stage1_capture_baseline_embeddings writes ctx.embeddings/baseline_embeddings.json")
        check(calls[1] == ("write_layout_detections", ctx.embeddings / "baseline_embeddings.json"),
              "stage1_capture_layout_detections writes the SAME per-run file (merge convention)")
        check(calls[2] == ("write_baseline_embeddings", ctx.embeddings / "postprocessing_embeddings.json"),
              "stage4_capture_postprocessing_embeddings writes ctx.embeddings/postprocessing_embeddings.json")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_sync_bucket_classifications_matches_ctx_scoped_rows():
    print("\n[5] sync_bucket_classifications() correctly matches a ctx-scoped row "
          "from an absolute bucket-CSV file_path (fixed after being a documented gap)")
    tmp = Path(tempfile.mkdtemp())
    try:
        ws = dataclasses.replace(WorkspaceContext.resolve(), workspace_root=tmp / "workspace")
        ctx = RunContext.create(ws, run_type="diagnostic", source_input="sync_test")

        img_path = ctx.working_images / "sync_test.jpg"
        Image.new("RGB", (10, 10)).save(img_path)

        db = PipelineDatabase(ctx.workspace.pipeline_db_path)
        image_id = db.get_or_create_image(
            source_path=str(img_path), working_path=ctx.to_relative(img_path),
            run_id=ctx.run_id, identity_hash="abc123",
        )

        # A bucket CSV as core/classifier.py would write it: file_path
        # stays ABSOLUTE (manifest/bucket CSVs are meant to be directly
        # openable without a RunContext), unlike the DB's own working_path.
        bucket_csv = ctx.buckets / "printed_document.csv"
        bucket_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(bucket_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["file_path", "category", "confidence", "model"])
            w.writeheader()
            w.writerow({
                "file_path": str(img_path), "category": "printed_document",
                "confidence": "0.95", "model": "gemma",
            })

        updated = sync_bucket_classifications(db, ctx.buckets, ctx=ctx)
        check(updated == 1, f"sync found and applied 1 change (got {updated})")

        row = db.get_image(image_id)
        check(row["bucket"] == "printed_document", "bucket updated")
        check(row["classifier_confidence"] == 0.95, "confidence updated")
        check(row["classifier_model"] == "gemma", "model updated")

        outs = db.get_stage_outputs(image_id, stage="stage5_classify")
        check(len(outs) == 1, "one stage5_classify row recorded")
        check(not Path(outs[0]["lookup_key"]).is_absolute(), "lookup_key stored run-relative")
        check(not Path(outs[0]["sidecar_path"]).is_absolute(), "sidecar_path stored run-relative")

        updated2 = sync_bucket_classifications(db, ctx.buckets, ctx=ctx)
        check(updated2 == 0, f"re-sync is idempotent, 0 changes (got {updated2})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    test_stage0to3_run_from_paths()
    test_second_run_does_not_touch_first()
    test_folder_adapter_forwards_ctx()
    test_embeddings_are_run_owned()
    test_sync_bucket_classifications_matches_ctx_scoped_rows()

    print(f"\n{'='*60}")
    print(f"RESULTS: {_PASS} passed, {_FAIL} failed")
    print(f"{'='*60}")
    if _FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
