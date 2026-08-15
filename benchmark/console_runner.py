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
from core.loader_registry import is_vllm_model, validate_model_assignment
from core.model_residency import residency
from model_console.adapter import (
    ChatBackendAdapter, build_config, model_requires_remote_backend,
)


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

    Deliberately independent of inference_engine (task #5: backend
    compatibility and model capability are separate dimensions) - a
    vLLM-only VLM is exactly as eligible for the vision suite as a
    transformers-only one; which ENGINE a model can run under is
    checked separately by required_backend_transport()/build_loader().
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


def inference_engine_for_model(model_name: str) -> str:
    """
    "transformers" | "vllm" - directly from config.runtime (existing
    registry metadata, see GenerationConfig.runtime's docstring). This
    is a per-config, FIXED property today: no single registered model
    entry supports being run under both engines - runtime is baked into
    which subprocess/loader machinery serves it. "Same model, both
    engines" (task #3) therefore means two DIFFERENT model_name entries
    that happen to share weights - see find_cross_engine_pairs() below,
    not a per-run toggle on one entry.
    """
    return validate_model_assignment(model_name).runtime


def required_backend_transport(model_name: str) -> str:
    """"local" | "remote" - which ChatBackendAdapter transport this
    model can actually run under, reusing model_requires_remote_backend()
    (existing config-driven check: runtime != "transformers", or an
    explicit extra.requires_backend: remote) rather than a second
    hard-coded compatibility list (task #4)."""
    return "remote" if model_requires_remote_backend(model_name) else "local"


def find_cross_engine_pairs() -> list[tuple[str, str]]:
    """
    Returns [(transformers_model_name, vllm_model_name), ...] for every
    pair of ENABLED registered models that share the exact same repo_id
    but run under different engines - the only configuration in this
    registry where "same model weights, different backend" (task #3) is
    actually true today. Computed by grouping config/models/*.yaml by
    repo_id, not a hand-maintained pairing list - a future config that
    happens to share a repo_id with an existing one is automatically
    picked up.

    As of 2026-08-16 there is exactly one such pair in this project's
    registry: gemma_12b_unified.yaml (transformers, Gemma4UnifiedLoader)
    and gemma_12b_w4a16.yaml (vllm) both declare repo_id
    "google/gemma-4-12B-it-qat-w4a16-ct" - literally the same checkpoint
    on disk, served by two different engines. Every other vLLM config
    (qwen25_vl_7b_awq, gemma_e4b_w4a16, internvl3_5_8b_awq,
    minicpm_v_gptq) uses a differently-quantized or differently-packaged
    repo_id than any transformers-path config, so those are NOT
    weight-identical pairs even when they're the same base model family
    - see the benchmark completion report's Equivalence caveats section
    for why qwen25_vl_7b vs qwen25_vl_7b_awq was used as a same-FAMILY
    (not same-weights) comparison instead.
    """
    from pathlib import Path
    import yaml as _yaml
    from core.loaders.base_loader import CONFIG_DIR

    by_repo: dict[str, list[tuple[str, str]]] = {}
    for path in sorted(Path(CONFIG_DIR).glob("*.yaml")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = _yaml.safe_load(f) or {}
        except Exception:
            continue
        repo_id = data.get("repo_id")
        if not repo_id:
            continue
        try:
            cfg = validate_model_assignment(path.stem)
        except ValueError:
            continue  # disabled/unregistered - not offerable either side
        engine = "vllm" if is_vllm_model(cfg) else "transformers"
        by_repo.setdefault(repo_id, []).append((path.stem, engine))

    pairs = []
    for repo_id, entries in by_repo.items():
        transformers_names = [n for n, e in entries if e == "transformers"]
        vllm_names = [n for n, e in entries if e == "vllm"]
        for t_name in transformers_names:
            for v_name in vllm_names:
                pairs.append((t_name, v_name))
    return pairs


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

    required_transport = required_backend_transport(model_name)
    if adapter.backend != required_transport:
        raise ValueError(
            f"Backend/model mismatch: {model_name!r} requires backend="
            f"{required_transport!r} (see model_console.adapter."
            f"model_requires_remote_backend()), but the given adapter is "
            f"backend={adapter.backend!r}."
        )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    started_at = datetime.now(timezone.utc).isoformat()

    # Real server-side residency truth, not this adapter instance's own
    # bookkeeping (task #9 - cold vs warm needs this to be correct for
    # BOTH engines, including a fresh adapter checking a server that
    # already has something resident from an earlier call).
    was_resident_before = adapter.server_resident_model_name() == model_name

    config = build_config(model_name, overrides=dict(suite.generation_settings))
    inference_engine = config.runtime
    repo_id = config.repo_id

    case_results: list[CaseResult] = []
    run_status = "completed"
    abort_reason: Optional[str] = None
    load_telemetry: Optional[dict[str, Any]] = None
    settings_translation_notes: list[str] = []
    generation_config_snapshot: Optional[dict[str, Any]] = None

    run_start = time.time()
    try:
        try:
            adapter.ensure_loaded(model_name, config)
        except Exception as e:
            raise _AbortRun(f"model load failed: {type(e).__name__}: {e}") from e

        total = len(suite.cases)
        for index, case in enumerate(suite.cases):
            if cancel_check is not None and cancel_check():
                run_status = "cancelled"
                abort_reason = f"cancelled before case {case.case_id!r} ({index}/{total} completed)"
                break

            if on_progress is not None:
                on_progress(index, total, case.case_id)

            case_results.append(_run_one_case(adapter, case, suite.system_prompt))

        else:
            # for/else: loop completed without break -> not cancelled.
            pass

    except _AbortRun as e:
        run_status = "aborted"
        abort_reason = str(e)
    finally:
        total_runtime = time.time() - run_start

    # generation_config_snapshot / load_telemetry / settings_translation_
    # notes: reused from whatever the FIRST case that actually captured
    # them reported - a fresh load only happens once per run, so the
    # first case to run it is the representative one (byte-identical
    # requested config every case regardless). Falls back to the
    # pre-load config (asdict) / residency's own last_load_telemetry
    # for the local-transformers path specifically, where load telemetry
    # is available even before this function's own case loop runs (a
    # single-case suite with a load failure could otherwise report None).
    for case in case_results:
        if not case.telemetry:
            continue
        if "generation_config_snapshot" in case.telemetry:
            popped = case.telemetry.pop("generation_config_snapshot")
            if popped is not None and generation_config_snapshot is None:
                generation_config_snapshot = popped
        if "settings_translation_notes" in case.telemetry:
            popped_notes = case.telemetry.pop("settings_translation_notes")
            if popped_notes and not settings_translation_notes:
                settings_translation_notes = popped_notes
        if load_telemetry is None and case.telemetry.get("load"):
            load_telemetry = case.telemetry["load"]
    if generation_config_snapshot is None:
        from dataclasses import asdict
        generation_config_snapshot = asdict(config)
    if load_telemetry is None and adapter.backend == "local":
        load_telemetry = residency.last_load_telemetry

    return BenchmarkRunResult(
        run_id=run_id,
        model_name=model_name,
        suite_id=suite.suite_id,
        suite_version=suite.version,
        suite_qualified_id=suite.qualified_id,
        backend=adapter.backend,
        inference_engine=inference_engine,
        repo_id=repo_id,
        generation_settings=dict(suite.generation_settings),
        generation_config_snapshot=generation_config_snapshot,
        settings_translation_notes=settings_translation_notes,
        was_resident_before_run=was_resident_before,
        load_telemetry=load_telemetry,
        started_at=started_at,
        finished_at=datetime.now(timezone.utc).isoformat(),
        total_runtime_seconds=round(total_runtime, 3),
        status=run_status,
        abort_reason=abort_reason,
        case_results=case_results,
    )


def run_suite_auto(
    model_name: str,
    suite: BenchmarkSuite,
    *,
    remote_base_url: Optional[str] = None,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> BenchmarkRunResult:
    """
    Convenience entry point (task #4/#11: "the UI should only offer
    valid model/backend combinations") - builds the correct
    ChatBackendAdapter transport for `model_name` automatically via
    required_backend_transport(), instead of the caller having to know
    whether a given model needs backend="local" or "remote". This is
    what benchmark_tab.py calls; run_suite() itself stays adapter-
    agnostic for tests/scripts that want to construct/reuse a specific
    adapter instance (e.g. to keep one adapter warm across two
    sequential suite runs).
    """
    transport = required_backend_transport(model_name)
    adapter = ChatBackendAdapter(backend=transport, base_url=remote_base_url)
    return run_suite(adapter, model_name, suite, on_progress=on_progress, cancel_check=cancel_check)


class _AbortRun(Exception):
    pass


def _run_one_case(adapter: ChatBackendAdapter, case, system_prompt: str = "") -> CaseResult:
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
        raw_output, meta = adapter.send_turn(case.prompt, image, system_prompt=system_prompt, history=None)
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
    # value for the whole run without threading extra return values
    # through _run_one_case()'s signature - both popped back out before
    # this CaseResult's telemetry is persisted (see run_suite()'s loop).
    telemetry["generation_config_snapshot"] = meta.get("generation_config_snapshot")
    if meta.get("settings_translation_notes"):
        telemetry["settings_translation_notes"] = meta["settings_translation_notes"]

    return CaseResult(
        case_id=case.case_id, category=case.category, prompt=case.prompt,
        image_path=case.image_path, expected_display=case.expected_display,
        raw_output=raw_output, status=score_result.status, score=score_result.score,
        explanation=score_result.explanation,
        runtime_seconds=meta.get("runtime_seconds"), telemetry=telemetry,
    )
