"""
Tests for core/workspace_context.py + core/run_context.py + the
hashed-run migration's DB identity contract in core/pipeline_db.py.
No pytest in this environment - plain assert-based, directly runnable,
matching this project's existing tests/test_decision_engine.py
convention.

Every test operates against an isolated temp workspace_root (via
tempfile.mkdtemp(), cleaned up after) - NEVER against the real
genealogy_workspace/ or data/ directories. This mirrors the same
manual verification done throughout the hashed-run-migration session
(see docs/RUN_ARCHITECTURE.md), just captured as a runnable regression
test.

Usage:
    python tests/test_run_context.py
"""

from __future__ import annotations

import dataclasses
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from core.workspace_context import WorkspaceContext
from core.run_context import RunContext, RUN_TYPES
from core.pipeline_db import PipelineDatabase

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


class _TempWorkspace:
    """Context manager: yields a WorkspaceContext rooted in a fresh
    temp directory, deleted on exit regardless of outcome."""

    def __enter__(self) -> WorkspaceContext:
        self._tmp = Path(tempfile.mkdtemp())
        real = WorkspaceContext.resolve()
        self.ws = dataclasses.replace(real, workspace_root=self._tmp / "workspace")
        return self.ws

    def __exit__(self, *exc):
        shutil.rmtree(self._tmp, ignore_errors=True)


def test_workspace_context_sibling_default():
    print("\n[1] WorkspaceContext.resolve() sibling-dir default")
    ws = WorkspaceContext.resolve()
    check(ws.workspace_root == ws.pipeline_root.parent / "genealogy_workspace",
          "workspace_root defaults to a sibling of pipeline_root")
    check(ws.runs_root == ws.workspace_root / "runs", "runs_root derived correctly")
    check(ws.pipeline_db_path == ws.workspace_root / "pipeline.db", "pipeline_db_path derived correctly")


def test_run_id_uniqueness_and_hash_stability():
    print("\n[2] RunContext.create() uniqueness + hash stability")
    with _TempWorkspace() as ws:
        ctx1 = RunContext.create(ws, run_type="diagnostic", source_input="same_source")
        ctx2 = RunContext.create(ws, run_type="diagnostic", source_input="same_source")
        check(ctx1.run_id != ctx2.run_id, "two runs with identical config get different run_ids (timestamp)")
        hash1 = ctx1.run_id.split("_", 1)[1]
        hash2 = ctx2.run_id.split("_", 1)[1]
        check(hash1 == hash2, "same config+source -> same hash segment (content fingerprint)")

        ctx3 = RunContext.create(ws, run_type="diagnostic", source_input="different_source")
        hash3 = ctx3.run_id.split("_", 1)[1]
        check(hash3 != hash1, "different source_input -> different hash segment")


def test_reserved_legacy_run_type_rejected():
    print("\n[3] run_type='legacy' rejected from public create()")
    with _TempWorkspace() as ws:
        try:
            RunContext.create(ws, run_type="legacy", source_input="x")
            check(False, "should have raised ValueError")
        except ValueError:
            check(True, "ValueError raised for reserved run_type")

    check("legacy" not in RUN_TYPES, "'legacy' excluded from the public RUN_TYPES enum")


def test_resume_blocks_completed_run():
    print("\n[4] resume() on a completed run raises; in-progress run reopens")
    with _TempWorkspace() as ws:
        ctx = RunContext.create(ws, run_type="diagnostic", source_input="x")
        ctx.mark_completed()
        try:
            RunContext.resume(ws, ctx.run_id)
            check(False, "should have raised ValueError")
        except ValueError:
            check(True, "resume() on completed run raised ValueError")

        ctx2 = RunContext.create(ws, run_type="diagnostic", source_input="y")
        resumed = RunContext.resume(ws, ctx2.run_id)
        check(resumed.run_root == ctx2.run_root, "resume() on in-progress run reopens the same directory")


def test_buckets_under_outputs_not_sibling():
    print("\n[5] buckets/ is locked under outputs/, not a run-root sibling")
    with _TempWorkspace() as ws:
        ctx = RunContext.create(ws, run_type="diagnostic", source_input="x")
        check(ctx.buckets == ctx.outputs / "buckets", "ctx.buckets == ctx.outputs / 'buckets'")
        check(ctx.buckets.parent == ctx.outputs, "buckets' parent is outputs, not run_root")


def test_run_name_stored_and_optional():
    print("\n[6] run_name stored in metadata.json; None is a valid default")
    with _TempWorkspace() as ws:
        named = RunContext.create(ws, run_type="diagnostic", source_input="x", run_name="my_test_run")
        meta = named._read_metadata()
        check(meta["run_name"] == "my_test_run", "run_name persisted in metadata.json")

        unnamed = RunContext.create(ws, run_type="diagnostic", source_input="x")
        meta2 = unnamed._read_metadata()
        check(meta2["run_name"] is None, "run_name defaults to None when omitted")


def test_two_runs_same_filename_no_collision():
    print("\n[7] Two runs, identical filenames, no collision on disk or in the DB")
    with _TempWorkspace() as ws:
        ctx1 = RunContext.create(ws, run_type="diagnostic", source_input="a")
        ctx2 = RunContext.create(ws, run_type="diagnostic", source_input="b")

        img1 = ctx1.working_images / "page.jpg"
        img2 = ctx2.working_images / "page.jpg"
        Image.new("RGB", (10, 10)).save(img1)
        Image.new("RGB", (10, 10)).save(img2)
        check(img1 != img2 and img1.exists() and img2.exists(),
              "two same-named files in different runs coexist on disk")

        db = PipelineDatabase(ws.pipeline_db_path)
        id1 = db.get_or_create_image(
            source_path=str(img1), working_path=ctx1.to_relative(img1), run_id=ctx1.run_id,
            identity_hash="hash1",
        )
        id2 = db.get_or_create_image(
            source_path=str(img2), working_path=ctx2.to_relative(img2), run_id=ctx2.run_id,
            identity_hash="hash2",
        )
        check(id1 != id2, "same-named files in different runs get distinct DB rows")

        found1 = db.get_image_by_path(ctx1.to_relative(img1), run_id=ctx1.run_id)
        found2 = db.get_image_by_path(ctx2.to_relative(img2), run_id=ctx2.run_id)
        check(found1["id"] == id1 and found2["id"] == id2,
              "run-scoped lookup resolves each run's own row, not the other run's")


def test_db_rows_are_run_relative_and_resolve():
    print("\n[8] DB working_path is run-relative and resolve_image_path() works")
    with _TempWorkspace() as ws:
        ctx = RunContext.create(ws, run_type="diagnostic", source_input="x")
        img = ctx.working_images / "foo.jpg"
        Image.new("RGB", (10, 10)).save(img)

        db = PipelineDatabase(ws.pipeline_db_path)
        image_id = db.get_or_create_image(
            source_path=str(img), working_path=ctx.to_relative(img), run_id=ctx.run_id,
            identity_hash="somehash",
        )
        row = db.get_image(image_id)
        check(not Path(row["working_path"]).is_absolute(), "working_path stored run-relative")
        check(row["run_id"] == ctx.run_id, "run_id tagged on the row")

        resolved = db.resolve_image_path(row)
        check(resolved == img, "resolve_image_path() reconstructs the exact original path")
        check(resolved.exists(), "resolved path points at a real file")


def test_resolve_path_independent_of_ambient_workspace():
    print("\n[9] resolve_path() derives runs_root from the DB's own location, not global state")
    with _TempWorkspace() as ws:
        ctx = RunContext.create(ws, run_type="diagnostic", source_input="x")
        img = ctx.working_images / "foo.jpg"
        Image.new("RGB", (10, 10)).save(img)

        db = PipelineDatabase(ws.pipeline_db_path)  # opened against the ISOLATED temp workspace
        resolved = db.resolve_path(ctx.run_id, ctx.to_relative(img))
        check(resolved == img,
              "resolve_path() ignores the ambient/real WorkspaceContext.resolve() default "
              "and derives runs_root from self.db_path.parent instead")


def test_to_relative_rejects_external_path():
    print("\n[10] RunContext.to_relative() rejects a path outside the run's root")
    with _TempWorkspace() as ws:
        ctx = RunContext.create(ws, run_type="diagnostic", source_input="x")
        outside = Path(tempfile.gettempdir()) / "definitely_not_in_this_run.jpg"
        try:
            ctx.to_relative(outside)
            check(False, "should have raised ValueError")
        except ValueError:
            check(True, "ValueError raised for a path outside ctx.run_root")


def main():
    test_workspace_context_sibling_default()
    test_run_id_uniqueness_and_hash_stability()
    test_reserved_legacy_run_type_rejected()
    test_resume_blocks_completed_run()
    test_buckets_under_outputs_not_sibling()
    test_run_name_stored_and_optional()
    test_two_runs_same_filename_no_collision()
    test_db_rows_are_run_relative_and_resolve()
    test_resolve_path_independent_of_ambient_workspace()
    test_to_relative_rejects_external_path()

    print(f"\n{'='*60}")
    print(f"RESULTS: {_PASS} passed, {_FAIL} failed")
    print(f"{'='*60}")
    if _FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
