"""
On-disk persistence for Benchmark tab runs - one JSON file per run
under WorkspaceContext's research/ convention, same pattern as
model_console/session_log.py (see that module's docstring for why the
path is resolved off WorkspaceContext rather than hardcoded, and why
this is a new subsystem with no data/outputs/ legacy path to stay
compatible with).

Format: research/model_console/benchmark_runs/<run_id>/run.json, one
file, containing the full BenchmarkRunResult (every case's raw output,
telemetry, and score included - task #6: raw outputs are never
summarized away on disk). A lightweight index.json in the same root
lists {run_id, model_name, suite_qualified_id, started_at, counts} for
every run, rewritten after each save, so the Benchmark tab's history
list doesn't need to open and parse every run.json just to populate a
picker.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from core.workspace_context import WorkspaceContext
from benchmark.run_result import BenchmarkRunResult, CaseResult

BENCHMARK_RUNS_ROOT = (
    WorkspaceContext.resolve().workspace_root / "research" / "model_console" / "benchmark_runs"
)
INDEX_PATH = BENCHMARK_RUNS_ROOT / "index.json"


def run_dir(run_id: str) -> Path:
    return BENCHMARK_RUNS_ROOT / run_id


def save_run(run: BenchmarkRunResult) -> None:
    """
    Writes run_dir(run.run_id)/run.json and refreshes index.json.
    Mirrors session_log.py's save-failure discipline: catches its own
    exceptions, prints a WARNING, and returns rather than raising - a
    persistence failure must not make the caller think the benchmark
    run itself failed (the run already completed by the time this is
    called; losing the on-disk record is bad, but should not be
    reported to the user as "benchmark failed").
    """
    try:
        _save_run_unsafe(run)
    except Exception as e:
        print(f"[model_console.benchmark_store] WARNING: failed to save benchmark "
              f"run {run.run_id!r}: {type(e).__name__}: {e}")


def _save_run_unsafe(run: BenchmarkRunResult) -> None:
    out_dir = run_dir(run.run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "run.json", "w", encoding="utf-8") as f:
        json.dump(asdict(run), f, indent=2, default=str)
    _refresh_index()


def _refresh_index() -> None:
    """Rebuilds index.json from every run.json on disk - simple and
    correct (no incremental-update bugs to worry about) at the scale a
    single-user benchmark history actually reaches; a full rebuild here
    is a handful of small JSON reads, not a real cost."""
    if not BENCHMARK_RUNS_ROOT.exists():
        return
    summaries = []
    for entry_dir in sorted(BENCHMARK_RUNS_ROOT.iterdir()):
        run_json = entry_dir / "run.json"
        if not run_json.exists():
            continue
        try:
            with open(run_json, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        summaries.append(_summarize(data))
    summaries.sort(key=lambda s: s.get("started_at") or "", reverse=True)
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)


def _summarize(run_data: dict[str, Any]) -> dict[str, Any]:
    counts = {"pass": 0, "fail": 0, "error": 0, "unscored": 0, "cancelled": 0}
    for case in run_data.get("case_results") or []:
        counts[case.get("status", "error")] = counts.get(case.get("status", "error"), 0) + 1
    return {
        "run_id": run_data.get("run_id"),
        "model_name": run_data.get("model_name"),
        "suite_id": run_data.get("suite_id"),
        "suite_version": run_data.get("suite_version"),
        "suite_qualified_id": run_data.get("suite_qualified_id"),
        "backend": run_data.get("backend"),
        "started_at": run_data.get("started_at"),
        "status": run_data.get("status"),
        "total_runtime_seconds": run_data.get("total_runtime_seconds"),
        "counts": counts,
    }


def list_runs() -> list[dict[str, Any]]:
    """Newest first. Rebuilds the index on the fly if it's missing
    (e.g. first read after a manual copy of run.json files) rather than
    returning an empty list and hiding real runs."""
    if not INDEX_PATH.exists():
        _refresh_index()
    if not INDEX_PATH.exists():
        return []
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_run(run_id: str) -> Optional[BenchmarkRunResult]:
    run_json = run_dir(run_id) / "run.json"
    if not run_json.exists():
        return None
    with open(run_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    case_results = [CaseResult(**c) for c in data.get("case_results") or []]
    data = dict(data)
    data["case_results"] = case_results
    return BenchmarkRunResult(**data)
