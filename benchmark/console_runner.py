"""
Executes one BenchmarkSuite against one model through the SAME
ChatBackendAdapter/model_residency path model_console/chat_tab.py's
plain-chat send already uses - no second inference/telemetry system.
Every capability check (text_only_supported / image_input_supported),
OOM-retry, and telemetry field this module reports comes straight out
of adapter.send_turn()'s existing meta dict (model_console/adapter.py)
and core/model_residency.py's last_load_telemetry - this module adds
NO new low-level instrumentation, only orchestrates suite cases through
the existing one and records the result (task requirement #5).

Determinism / controlled conditions (task #4, #12): every case in a
suite is sent one-shot (history=None, exactly chat_tab.py's default
one-shot path) with the SAME suite-wide generation_settings applied via
build_config()'s overrides, and an empty system prompt (a suite must
not depend on model_console's currently-typed system-prompt box - that
would make two runs of "the same suite" not actually comparable). The
resulting GenerationConfig (asdict) is recorded in the run so a run
can be audited against what was ACTUALLY used, not just requested -
if a backend silently ignored a setting, the snapshot still reflects
what was assigned, which is the best available intended-vs-actual
signal without per-backend introspection this project doesn't have.

Cancellation (task #10): checked BETWEEN cases only - an in-flight
model call cannot be safely interrupted mid-generation without a
per-loader cancellation hook this project doesn't have, so a Cancel
request finishes the current case and then stops. This is recorded
explicitly (status="cancelled" on the run and on every case that never
ran) rather than silently truncating the results list.
"""

from __future__ import annotations

import time
import traceback
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from PIL import Image

from benchmark.run_result import BenchmarkRunResult, CaseResult
from benchmark.scorers import score_case
from benchmark.suite_schema import BenchmarkSuite
from core.loader_registry import validate_model_assignment
from core.model_residency import residency
from model_console.adapter import ChatBackendAdapter, build_config


class BenchmarkCancelled(Exception):
    """Raised internally to unwind cleanly on a cancel request between cases."""


def eligible_for_suite(model_name: str, suite: BenchmarkSuite) -> tuple[bool, str]:
    """
    Returns (eligible, reason). Capability-only check - deliberately
    does NOT consult vision_validation_status (task #1/#9: an
    "untested" VLM must remain selectable for a vision suite, since
    running that suite is exactly how it gets evidence toward
    "validated"). Raises nothing - a missing/disabled model is simply
    "not eligible", since the caller (benchmark_tab.py) only calls this
    after already sourcing model_name from the shared registry listing.
    """
    try:
        cfg = validate_model_assignment(model_name)
    except ValueError as e:
        return False, str(e)
    if suite.required_capability == "vision" and not cfg.image_input_supported:
        return False, f"{model_name!r} has image_input_supported: false (text-only model)."
    if suite.required_capability == "text" and not cfg.text_only_supported:
        return False, f"{model_name!r} has text_only_supported: false (not confirmed text-capable)."
    return True, ""


def run_suite(
    adapter: ChatBackendAdapter,
    model_name: str,
    suite: BenchmarkSuite,
    *,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> BenchmarkRunResult:
    """
    Runs every case in `suite` against `model_name` via `adapter`, in
    order, sequentially. `on_progress(index, total, case_id)` is called
    before each case starts (index is 0-based) - the caller (a
    background thread in benchmark_tab.py) uses this to post a UI
    update. `cancel_check()`, if given, is polled before each case;
    returning True stops the run after the current case (see module
    docstring on cancellation granularity).

    A per-case inference or scoring exception is caught and recorded as
    that case's own "error" result - it does NOT abort the run (task
    #11: "handle errors per test where practical"). The run itself is
    only aborted early (status="aborted") if ensure_loaded() itself
    fails (a model that won't load at all leaves nothing further to
    run), or on cancellation (status="cancelled").
    """
    eligible, reason = eligible_for_suite(model_name, suite)
    if not eligible:
        raise ValueError(f"Model/suite mismatch: {reason}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    started_at = datetime.now(timezone.utc).isoformat()

    was_resident_before = (
        (residency.resident_model_name == model_name) if adapter.backend == "local" else None
    )

    config = build_config(model_name, overrides=dict(suite.generation_settings))

    case_results: list[CaseResult] = []
    run_status = "completed"
    abort_reason: Optional[str] = None
    load_telemetry: Optional[dict[str, Any]] = None
    generation_config_snapshot: Optional[dict[str, Any]] = None

    run_start = time.time()
    try:
        try:
            adapter.ensure_loaded(model_name, config)
        except Exception as e:
            raise _AbortRun(f"model load failed: {type(e).__name__}: {e}") from e

        if adapter.backend == "local":
            load_telemetry = residency.last_load_telemetry

        total = len(suite.cases)
        for index, case in enumerate(suite.cases):
            if cancel_check is not None and cancel_check():
                run_status = "cancelled"
                abort_reason = f"cancelled before case {case.case_id!r} ({index}/{total} completed)"
                break

            if on_progress is not None:
                on_progress(index, total, case.case_id)

            case_results.append(_run_one_case(adapter, case))

        else:
            # for/else: loop completed without break -> not cancelled.
            pass

    except _AbortRun as e:
        run_status = "aborted"
        abort_reason = str(e)
    finally:
        total_runtime = time.time() - run_start

    # generation_config_snapshot: reuse whatever the last successful
    # case reported (byte-identical config every case, so any one of
    # them is representative) - falls back to the pre-load config
    # (asdict) if every case errored before producing a snapshot, so
    # "what was requested" is still recorded even on a fully failed run.
    for case in case_results:
        if case.telemetry and "generation_config_snapshot" in case.telemetry:
            popped = case.telemetry.pop("generation_config_snapshot")
            if popped is not None:
                generation_config_snapshot = popped
    if generation_config_snapshot is None:
        from dataclasses import asdict
        generation_config_snapshot = asdict(config)

    return BenchmarkRunResult(
        run_id=run_id,
        model_name=model_name,
        suite_id=suite.suite_id,
        suite_version=suite.version,
        suite_qualified_id=suite.qualified_id,
        backend=adapter.backend,
        generation_settings=dict(suite.generation_settings),
        generation_config_snapshot=generation_config_snapshot,
        was_resident_before_run=was_resident_before,
        load_telemetry=load_telemetry,
        started_at=started_at,
        finished_at=datetime.now(timezone.utc).isoformat(),
        total_runtime_seconds=round(total_runtime, 3),
        status=run_status,
        abort_reason=abort_reason,
        case_results=case_results,
    )


class _AbortRun(Exception):
    pass


def _run_one_case(adapter: ChatBackendAdapter, case) -> CaseResult:
    image = None
    resolved_image_path = case.resolve_image_path()
    if resolved_image_path is not None:
        try:
            image = Image.open(resolved_image_path).convert("RGB")
        except Exception as e:
            return CaseResult(
                case_id=case.case_id, category=case.category, prompt=case.prompt,
                image_path=case.image_path, expected_display=case.expected_display,
                raw_output=None, status="error", score=None,
                explanation=f"could not load image asset: {type(e).__name__}: {e}",
                runtime_seconds=None, telemetry=None, error=str(e),
            )

    start = time.time()
    try:
        raw_output, meta = adapter.send_turn(case.prompt, image, system_prompt="", history=None)
    except Exception as e:
        return CaseResult(
            case_id=case.case_id, category=case.category, prompt=case.prompt,
            image_path=case.image_path, expected_display=case.expected_display,
            raw_output=None, status="error", score=None,
            explanation=f"inference call failed: {type(e).__name__}: {e}",
            runtime_seconds=round(time.time() - start, 3), telemetry=None,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        )

    score_result = score_case(case.scorer, case.scorer_args, raw_output)
    telemetry = dict(meta.get("telemetry") or {})
    # Stashed here transiently so run_suite() can pull ONE representative
    # snapshot for the whole run without threading it through as a
    # separate return value - popped back out before this CaseResult's
    # telemetry is persisted (see run_suite()'s loop above).
    telemetry["generation_config_snapshot"] = meta.get("generation_config_snapshot")

    return CaseResult(
        case_id=case.case_id, category=case.category, prompt=case.prompt,
        image_path=case.image_path, expected_display=case.expected_display,
        raw_output=raw_output, status=score_result.status, score=score_result.score,
        explanation=score_result.explanation,
        runtime_seconds=meta.get("runtime_seconds"), telemetry=telemetry,
    )
