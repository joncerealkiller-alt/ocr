"""
Result dataclasses for one Benchmark-tab run. Separate from
benchmark/suite_schema.py (which defines what to run) and
benchmark/console_runner.py (which executes it) so model_console/
benchmark_store.py can (de)serialize these without importing the
runner itself (which pulls in model_console.adapter / torch).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class CaseResult:
    case_id: str
    category: str
    prompt: str
    image_path: Optional[str]        # relative path, as declared in the suite YAML
    expected_display: str
    raw_output: Optional[str]        # None only if the inference call itself errored
    status: str                      # "pass" | "fail" | "error" | "unscored" | "cancelled"
    score: Optional[float]
    explanation: str
    runtime_seconds: Optional[float]
    telemetry: Optional[dict[str, Any]]   # adapter.send_turn()'s meta["telemetry"], verbatim
    error: Optional[str] = None      # inference-call exception text, if any


@dataclass
class BenchmarkRunResult:
    run_id: str
    model_name: str
    suite_id: str
    suite_version: str
    suite_qualified_id: str          # f"{suite_id}_v{suite_version}" - the comparison-safety key
    backend: str                     # "local" | "remote" (ChatBackendAdapter.backend)
    generation_settings: dict[str, Any]     # the suite's requested settings
    generation_config_snapshot: Optional[dict[str, Any]]  # actual GenerationConfig used (asdict), post-load
    was_resident_before_run: Optional[bool]  # None when unknown (e.g. remote backend)
    load_telemetry: Optional[dict[str, Any]]
    started_at: str                  # ISO 8601
    finished_at: Optional[str]
    total_runtime_seconds: Optional[float]
    status: str                      # "completed" | "cancelled" | "aborted"
    abort_reason: Optional[str]
    case_results: list[CaseResult] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        counts = {"pass": 0, "fail": 0, "error": 0, "unscored": 0, "cancelled": 0}
        for case in self.case_results:
            counts[case.status] = counts.get(case.status, 0) + 1
        return counts

    @property
    def scored_pass_rate(self) -> Optional[float]:
        """Fraction of PROGRAMMATICALLY SCORED cases (pass+fail, i.e.
        excluding unscored/error/cancelled) that passed - None if there
        were zero scored cases, rather than a misleading 0.0."""
        scored = [c for c in self.case_results if c.status in ("pass", "fail")]
        if not scored:
            return None
        return sum(1 for c in scored if c.status == "pass") / len(scored)

    @property
    def mean_tokens_per_sec(self) -> Optional[float]:
        values = [
            c.telemetry["generate"]["tokens_per_sec"]
            for c in self.case_results
            if c.telemetry and c.telemetry.get("generate", {}).get("tokens_per_sec") is not None
        ]
        return (sum(values) / len(values)) if values else None

    @property
    def peak_vram_mb(self) -> Optional[float]:
        values = [
            c.telemetry["generate"]["peak_vram_mb"]
            for c in self.case_results
            if c.telemetry and c.telemetry.get("generate", {}).get("peak_vram_mb") is not None
        ]
        return max(values) if values else None
