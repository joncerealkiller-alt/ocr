"""RunContext: the single owner of "where does this run's data live."

No pipeline module should independently construct a path like
Path("data/working") or DATA_DIR / "buckets". Instead it receives a
RunContext (explicit dependency injection, no global) and reads paths
off it. This makes a future path-layout change a one-file edit here
instead of a repository-wide grep-and-replace.

See docs/RUN_ARCHITECTURE.md for the full design rationale and the
run_id hashing scheme.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from core.workspace_context import WorkspaceContext

# Public run types a caller may request. "legacy" is intentionally not
# listed here - it is reserved for the one-time DB migration script's
# synthetic run (see scripts/migrate_pipeline_db_to_run_schema.py) and
# is rejected by RunContext.create().
RUN_TYPES = frozenset(
    {
        "production",
        "dataset_build",
        "benchmark",
        "research",
        "training",
        "calibration",
        "validation",
        "diagnostic",
    }
)
_RESERVED_RUN_TYPE = "legacy"

_CONFIG_SNAPSHOT_FILES = ("pipeline.yaml", "taxonomy.yaml", "decision_engine.yaml")


def _canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_commit_and_dirty(pipeline_root: Path) -> tuple[Optional[str], Optional[bool]]:
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=pipeline_root,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        dirty = (
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=pipeline_root,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
            != ""
        )
        return sha, dirty
    except Exception:
        return None, None


def _config_hashes(pipeline_root: Path) -> dict:
    hashes = {}
    for name in _CONFIG_SNAPSHOT_FILES:
        path = pipeline_root / "config" / name
        key = name.rsplit(".", 1)[0] + "_hash"
        if path.exists():
            hashes[key] = _hash_text(path.read_text(encoding="utf-8"))
        else:
            hashes[key] = None
    return hashes


def _compute_run_id_hash(pipeline_root: Path, source_input: str) -> str:
    """Fingerprint of the inputs that determine run *behavior*: resolved
    config file contents (not mtimes) + the source input identifier.
    Two runs with identical config+source get the same hash segment even
    though their timestamps (and therefore full run_id) differ - the hash
    is a "these are functionally equivalent" signal, not the uniqueness
    guarantee (the timestamp is)."""
    payload = {"source_input": source_input, **_config_hashes(pipeline_root)}
    return _hash_text(_canonical_json(payload))[:8]


@dataclass
class RunContext:
    run_id: str
    run_root: Path
    workspace: WorkspaceContext

    manifest_dir: Path = field(init=False)
    manifest_csv: Path = field(init=False)
    manifest_provenance: Path = field(init=False)

    raw_from_pdf: Path = field(init=False)

    working: Path = field(init=False)
    working_images: Path = field(init=False)
    working_analysis: Path = field(init=False)

    outputs: Path = field(init=False)
    buckets: Path = field(init=False)
    extraction: Path = field(init=False)
    routing: Path = field(init=False)
    embeddings: Path = field(init=False)

    quarantine: Path = field(init=False)
    diagnostics: Path = field(init=False)
    config_snapshot: Path = field(init=False)
    reports: Path = field(init=False)

    metadata_path: Path = field(init=False)

    def __post_init__(self):
        self.manifest_dir = self.run_root / "manifest"
        self.manifest_csv = self.manifest_dir / "manifest.csv"
        self.manifest_provenance = self.manifest_dir / "manifest_provenance.json"

        self.raw_from_pdf = self.run_root / "raw_from_pdf"

        self.working = self.run_root / "working"
        self.working_images = self.working / "images"
        self.working_analysis = self.working / "analysis"

        self.outputs = self.run_root / "outputs"
        self.buckets = self.outputs / "buckets"
        self.extraction = self.outputs / "extraction"
        self.routing = self.outputs / "routing"
        self.embeddings = self.outputs / "embeddings"

        self.quarantine = self.run_root / "quarantine"
        self.diagnostics = self.run_root / "diagnostics"
        self.config_snapshot = self.run_root / "config_snapshot"
        self.reports = self.run_root / "reports"

        self.metadata_path = self.run_root / "metadata.json"

    # -- directory tree -----------------------------------------------

    def _all_dirs(self):
        return (
            self.manifest_dir,
            self.raw_from_pdf,
            self.working_images,
            self.working_analysis,
            self.buckets,
            self.extraction,
            self.routing,
            self.embeddings,
            self.quarantine,
            self.diagnostics,
            self.config_snapshot,
            self.reports,
        )

    def _create_dirs(self) -> None:
        for d in self._all_dirs():
            d.mkdir(parents=True, exist_ok=True)

    # -- path normalization (the DB/context boundary) -------------------

    def to_relative(self, path: Path | str) -> str:
        """THE canonical normalization point between a real filesystem
        Path used by runtime code and the run-relative string form
        core/pipeline_db.py persists (working_path/sidecar_path/
        lookup_key for any artifact this run owns - see
        docs/RUN_ARCHITECTURE.md). Every call site that's about to write
        a run-owned artifact's path into the DB must go through this
        first, not construct or store an absolute path directly.
        Raises ValueError if path isn't actually under this run's root
        (a bug at the call site, not a case to silently tolerate -
        source_path for an external original is NOT run-owned and
        should never be passed here)."""
        p = Path(path).resolve()
        try:
            return str(p.relative_to(self.run_root))
        except ValueError:
            raise ValueError(
                f"{p} is not under this run's root ({self.run_root}) - "
                f"only run-owned artifacts (working/, outputs/, ...) go "
                f"through to_relative(); an external source path should "
                f"be stored as-is, not normalized against this run."
            )

    def to_absolute(self, relative_path: str) -> Path:
        """Inverse of to_relative() - reconstructs a real Path from a
        run-relative string, without needing a PipelineDatabase/
        WorkspaceContext round-trip when the RunContext is already in
        hand. Equivalent to db.resolve_path(self.run_id, relative_path)."""
        return self.run_root / relative_path

    def _snapshot_config(self) -> None:
        for name in _CONFIG_SNAPSHOT_FILES:
            src = self.workspace.pipeline_root / "config" / name
            if src.exists():
                shutil.copy2(src, self.config_snapshot / name)

    # -- lifecycle ------------------------------------------------------

    @classmethod
    def create(
        cls,
        workspace: WorkspaceContext,
        *,
        run_type: str,
        source_input: str,
        run_name: Optional[str] = None,
        parent_run_id: Optional[str] = None,
    ) -> "RunContext":
        if run_type == _RESERVED_RUN_TYPE:
            raise ValueError(
                f'run_type "{_RESERVED_RUN_TYPE}" is reserved for the migration script'
            )
        if run_type not in RUN_TYPES:
            raise ValueError(
                f"run_type must be one of {sorted(RUN_TYPES)}, got {run_type!r}"
            )

        workspace.ensure_dirs()

        now = datetime.now(timezone.utc)
        timestamp = now.strftime("%Y%m%dT%H%M%S%f")
        run_hash = _compute_run_id_hash(workspace.pipeline_root, source_input)
        run_id = f"{timestamp}_{run_hash}"

        # Belt-and-suspenders: guard against a directory collision (clock
        # rollback, mocked time in tests) rather than silently reusing or
        # erroring outright.
        base_run_id = run_id
        suffix = 2
        while (workspace.runs_root / run_id).exists():
            run_id = f"{base_run_id}_{suffix}"
            suffix += 1

        run_root = workspace.runs_root / run_id
        ctx = cls(run_id=run_id, run_root=run_root, workspace=workspace)
        ctx._create_dirs()
        ctx._snapshot_config()

        git_commit, git_dirty = _git_commit_and_dirty(workspace.pipeline_root)
        config_hashes = _config_hashes(workspace.pipeline_root)

        metadata = {
            "run_id": run_id,
            "run_name": run_name,
            "run_type": run_type,
            "created_at": now.isoformat(),
            "completed_at": None,
            "status": "in_progress",
            "source_input": source_input,
            "pipeline_version": git_commit,
            "git_dirty": git_dirty,
            "taxonomy_hash": config_hashes["taxonomy_hash"],
            "pipeline_config_hash": config_hashes["pipeline_hash"],
            "decision_engine_config_hash": config_hashes["decision_engine_hash"],
            "model_config": None,
            "prompt_versions": None,
            "preprocessing_config": None,
            "parent_run_id": parent_run_id,
            "artifact_summary": None,
        }
        ctx._write_metadata(metadata)
        return ctx

    @classmethod
    def create_legacy_migration_run(
        cls, workspace: WorkspaceContext, *, source_input: str
    ) -> "RunContext":
        """Used exclusively by scripts/migrate_pipeline_db_to_run_schema.py
        to create the synthetic `legacy_pre_run_system` run that holds
        pre-run-system data (see docs/RUN_ARCHITECTURE.md). Not for any
        other caller - bypasses the public run_type validation in
        create() specifically to allow the reserved "legacy" type, and
        writes status="completed" immediately since this data already
        exists on disk (nothing is "in progress")."""
        workspace.ensure_dirs()
        run_id = "legacy_pre_run_system"
        run_root = workspace.runs_root / run_id
        ctx = cls(run_id=run_id, run_root=run_root, workspace=workspace)
        ctx._create_dirs()

        now = datetime.now(timezone.utc)
        metadata = {
            "run_id": run_id,
            "run_name": "Pre-run-system snapshot (migrated)",
            "run_type": _RESERVED_RUN_TYPE,
            "created_at": None,
            "completed_at": now.isoformat(),
            "status": "completed",
            "source_input": source_input,
            "pipeline_version": None,
            "git_dirty": None,
            "taxonomy_hash": None,
            "pipeline_config_hash": None,
            "decision_engine_config_hash": None,
            "model_config": None,
            "prompt_versions": None,
            "preprocessing_config": None,
            "parent_run_id": None,
            "artifact_summary": None,
        }
        ctx._write_metadata(metadata)
        return ctx

    @classmethod
    def resume(cls, workspace: WorkspaceContext, run_id: str) -> "RunContext":
        run_root = workspace.runs_root / run_id
        if not run_root.exists():
            raise FileNotFoundError(f"No run directory for run_id {run_id!r} at {run_root}")

        ctx = cls(run_id=run_id, run_root=run_root, workspace=workspace)
        metadata = ctx._read_metadata()
        if metadata.get("status") == "completed":
            raise ValueError(
                f"Run {run_id!r} is completed and immutable; "
                f"use RunContext.create(parent_run_id={run_id!r}) instead"
            )
        ctx._create_dirs()
        return ctx

    def mark_completed(self) -> None:
        metadata = self._read_metadata()
        metadata["status"] = "completed"
        metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
        self._write_metadata(metadata)

    def mark_failed(self, error: str) -> None:
        metadata = self._read_metadata()
        metadata["status"] = "failed"
        metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
        metadata["error"] = error
        self._write_metadata(metadata)

    # -- metadata io ------------------------------------------------------

    def _read_metadata(self) -> dict:
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_metadata(self, metadata: dict) -> None:
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
