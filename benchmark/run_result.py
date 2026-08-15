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
    backend: str                     # "local" | "remote" (ChatBackendAdapter transport)
    generation_settings: dict[str, Any]     # the suite's requested settings
    generation_config_snapshot: Optional[dict[str, Any]]  # actual GenerationConfig used (asdict), post-load
    was_resident_before_run: Optional[bool]  # None when unknown
    load_telemetry: Optional[dict[str, Any]]
    started_at: str                  # ISO 8601
    finished_at: Optional[str]
    total_runtime_seconds: Optional[float]
    status: str                      # "completed" | "cancelled" | "aborted"
    abort_reason: Optional[str]
    case_results: list[CaseResult] = field(default_factory=list)

    # --- Backend-comparison fields (2026-08-16) ---------------------
    # `backend` above is TRANSPORT (local Windows in-process vs remote
    # HTTP-to-WSL) - orthogonal to `inference_engine`, which is the
    # dimension task's real subject: "transformers" (BaseLoader/HF
    # generate()) vs "vllm" (core/vllm_runtime.py's subprocess server).
    # Sourced directly from the model's config.runtime field (already
    # existing registry metadata - see core/loaders/base_loader.py's
    # GenerationConfig.runtime docstring) - not a new independent
    # concept invented for the benchmark. Defaults to "transformers"
    # (every run recorded before this field existed WAS a transformers
    # run) so old run.json files still deserialize correctly.
    inference_engine: str = "transformers"
    # The model's repo_id at run time - lets the comparison UI detect
    # "these two runs don't even share weights" (e.g. qwen25_vl_7b vs
    # qwen25_vl_7b_awq are the same base architecture but NOT the same
    # checkpoint - AWQ quantization changes the weights) instead of
    # silently treating same-suite-different-model as apples-to-apples.
    repo_id: Optional[str] = None
    # Human-readable notes about settings a backend could NOT honor
    # identically to what was requested (task #6) - e.g. vLLM has no
    # no_repeat_ngram_size equivalent at all. Empty list for the
    # transformers engine (nothing is translated - the loader receives
    # the GenerationConfig fields directly), populated for vllm (see
    # api/agent_main.py's _make_vllm_send_turn_fn()). Persisted per-run
    # (not just documented in code) so a reader comparing two OLD runs
    # doesn't have to go re-read source to know what differed.
    settings_translation_notes: list[str] = field(default_factory=list)

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
        """
        TRANSFORMERS ONLY - torch.cuda.max_memory_allocated()-derived,
        captured per-call by GemmaLoader (see base_loader.py's
        last_inference_telemetry). vLLM has no equivalent in-process
        counter available from outside its subprocess (task #8/the
        Equivalence caveats section: this is a genuine engine
        observability gap, not an oversight) - always None for
        inference_engine="vllm" runs. Use resident_vram_mb below for a
        metric that IS available (if less precise) under both engines.
        """
        values = [
            c.telemetry["generate"]["peak_vram_mb"]
            for c in self.case_results
            if c.telemetry and c.telemetry.get("generate", {}).get("peak_vram_mb") is not None
        ]
        return max(values) if values else None

    @property
    def resident_vram_mb(self) -> Optional[float]:
        """
        Best-available "how much GPU memory is this engine holding"
        figure that exists under BOTH engines - transformers reports
        final_vram_mb (torch's own allocator counter, in-process,
        exact); vLLM reports vram_used_mb (an `nvidia-smi` snapshot
        taken right after each response, driver-level, includes the
        WHOLE GPU's usage at that instant, not just this process' -
        less precise on a shared GPU, but the only figure the subprocess
        boundary makes available). NOT the same measurement methodology
        - see this run's settings_translation_notes / the completion
        report's Equivalence caveats section before treating the two
        numbers as strictly comparable.
        """
        values = [
            c.telemetry["generate"].get("final_vram_mb") or c.telemetry["generate"].get("vram_used_mb")
            for c in self.case_results
            if c.telemetry and c.telemetry.get("generate")
            and (c.telemetry["generate"].get("final_vram_mb") is not None
                 or c.telemetry["generate"].get("vram_used_mb") is not None)
        ]
        return values[-1] if values else None
