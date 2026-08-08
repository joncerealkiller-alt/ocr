"""Resolves the two filesystem roots the pipeline reads/writes:

- pipeline_root: this repo (code, config, docs, UI, tests) - never
  written to at runtime.
- workspace_root: generated/persistent data (runs, datasets, models,
  checkpoints, logs) - a sibling directory by default, never git-tracked.

No other module should hardcode `Path("data/...")` or similar. Get a
WorkspaceContext (via `WorkspaceContext.resolve()`) and derive paths
from it, or from a `RunContext` built on top of it (see
core/run_context.py).

See docs/RUN_ARCHITECTURE.md for the full ownership model this exists
to enforce, and config/workspace.yaml for the resolution order.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKSPACE_CONFIG_PATHS = (
    _REPO_ROOT / "config" / "workspace.local.yaml",
    _REPO_ROOT / "config" / "workspace.yaml",
)


def _load_workspace_config() -> dict:
    """First existing file wins: workspace.local.yaml (gitignored,
    machine-specific) takes precedence over the checked-in workspace.yaml."""
    for path in _WORKSPACE_CONFIG_PATHS:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return data
    return {}


@dataclass(frozen=True)
class WorkspaceContext:
    pipeline_root: Path
    workspace_root: Path

    @property
    def runs_root(self) -> Path:
        return self.workspace_root / "runs"

    @property
    def pipeline_db_path(self) -> Path:
        return self.workspace_root / "pipeline.db"

    @property
    def unmigrated_root(self) -> Path:
        """Everything not yet migrated into the run/dataset/model/research
        tree (see docs/RUN_ARCHITECTURE.md's phase-2 mapping). Not a run:
        has no metadata.json and is never resolved through
        PipelineDatabase.resolve_path()."""
        return self.workspace_root / "unmigrated"

    @classmethod
    def resolve(cls) -> "WorkspaceContext":
        config = _load_workspace_config()

        pipeline_root_raw = os.environ.get("GENEALOGY_PIPELINE_ROOT") or config.get(
            "pipeline_root"
        )
        pipeline_root = (
            Path(pipeline_root_raw).expanduser().resolve()
            if pipeline_root_raw
            else _REPO_ROOT
        )

        workspace_root_raw = os.environ.get("GENEALOGY_WORKSPACE_ROOT") or config.get(
            "workspace_root"
        )
        workspace_root = (
            Path(workspace_root_raw).expanduser().resolve()
            if workspace_root_raw
            else pipeline_root.parent / "genealogy_workspace"
        )

        return cls(pipeline_root=pipeline_root, workspace_root=workspace_root)

    def ensure_dirs(self) -> None:
        """Creates workspace_root/runs and workspace_root/unmigrated if
        they don't exist yet. Safe to call repeatedly."""
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.unmigrated_root.mkdir(parents=True, exist_ok=True)
