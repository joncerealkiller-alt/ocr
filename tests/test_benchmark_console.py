"""
Tests for the Model Console Benchmark tab (2026-08-15) - suite loading,
capability-based eligibility, scorers, versioning, and results
persistence. No pytest in this environment - plain assert-based,
matching this project's existing tests/test_resource_guard.py
convention.

Does NOT run a real model - eligible_for_suite()/scoring/persistence
are all pure-config/pure-function or filesystem-only, so this suite
runs without a GPU or any loader import. A real end-to-end model run
(text_baseline + vision_baseline against a real loaded model) was
verified separately, live, in the Model Console UI - see the
completion summary.

Usage:
    python tests/test_benchmark_console.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.run_result import BenchmarkRunResult, CaseResult
from benchmark.scorers import score_case
from benchmark.suite_schema import list_suites, load_suite
from benchmark.console_runner import (
    find_cross_engine_pairs, inference_engine_for_model, required_backend_transport,
)
from core.loader_registry import is_vllm_model, validate_model_assignment
from core.loaders.base_loader import load_model_config

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


def test_suite_loading():
    print("test_suite_loading")
    suites = list_suites()
    ids = {s.suite_id for s in suites}
    check("text_baseline" in ids, "text_baseline_v1.yaml loads")
    check("vision_baseline" in ids, "vision_baseline_v1.yaml loads")
    check(sum(1 for s in suites if s.suite_id == "vision_baseline") == 4,
          "vision_baseline v1-v4 all load as distinct suites")

    text_suite = load_suite("text_baseline_v1")
    check(text_suite.required_capability == "text", "text suite declares required_capability=text")
    check(len(text_suite.cases) >= 5, "text suite has at least 5 cases")
    check(text_suite.qualified_id == "text_baseline_v1", "qualified_id combines suite_id + version")

    vision_suite = load_suite("vision_baseline_v1")
    check(vision_suite.required_capability == "vision", "vision suite declares required_capability=vision")
    for case in vision_suite.cases:
        resolved = case.resolve_image_path()
        check(resolved is not None and resolved.exists(), f"vision case {case.case_id!r} image asset exists on disk")

    vision_suite_v2 = load_suite("vision_baseline_v2")
    check(vision_suite_v2.qualified_id != vision_suite.qualified_id,
          "vision_baseline v1 and v2 have distinct qualified_ids (never silently comparable)")
    check(any(c.case_id == "table_form_interpretation" for c in vision_suite_v2.cases),
          "v2 adds the table_form_interpretation case")
    for case in vision_suite_v2.cases:
        resolved = case.resolve_image_path()
        check(resolved is not None and resolved.exists(), f"v2 vision case {case.case_id!r} image asset exists on disk")

    vision_suite_v3 = load_suite("vision_baseline_v3")
    check(vision_suite_v3.qualified_id not in (vision_suite.qualified_id, vision_suite_v2.qualified_id),
          "vision_baseline v3 has its own distinct qualified_id")
    basic_desc_v2 = next(c for c in vision_suite_v2.cases if c.case_id == "basic_description")
    basic_desc_v3 = next(c for c in vision_suite_v3.cases if c.case_id == "basic_description")
    check(basic_desc_v2.scorer == "contains_any", "v2's basic_description scorer is unchanged (frozen, past runs stay valid)")
    check(basic_desc_v3.scorer == "manual", "v3 fixes basic_description to manual/unscored (subjective task, not fake-precision keyword match)")

    vision_suite_v4 = load_suite("vision_baseline_v4")
    check(vision_suite_v4.qualified_id not in
          (vision_suite.qualified_id, vision_suite_v2.qualified_id, vision_suite_v3.qualified_id),
          "vision_baseline v4 has its own distinct qualified_id")
    basic_desc_v4 = next(c for c in vision_suite_v4.cases if c.case_id == "basic_description")
    check(basic_desc_v4.scorer == "contains_any", "v4 restores basic_description to an objective scorer")
    check(basic_desc_v4.image_path == "data/_test_fixtures/fake_test_doc_unlabeled.png",
          "v4's basic_description uses the redacted (no-giveaway-text) asset")
    resolved_v4_asset = basic_desc_v4.resolve_image_path()
    check(resolved_v4_asset is not None and resolved_v4_asset.exists(), "v4's redacted asset exists on disk")
    check(vision_suite.cases[0].image_path == "data/_test_fixtures/fake_test_doc.png"
          and (PROJECT_ROOT / "data/_test_fixtures/fake_test_doc.png").exists(),
          "original fake_test_doc.png is untouched and still exists (shared with ui/review_uncertain.py)")

    try:
        load_suite("does_not_exist_xyz")
        check(False, "loading a missing suite raises")
    except FileNotFoundError:
        check(True, "loading a missing suite raises FileNotFoundError")


def test_eligibility_capability_gating():
    print("test_eligibility_capability_gating")
    from benchmark.console_runner import eligible_for_suite

    vision_suite = load_suite("vision_baseline_v1")
    text_suite = load_suite("text_baseline_v1")

    # qwen_research_text: text_only_supported True BY CONSTRUCTION
    # (TextLLMLoader, no vision path - see config/models/qwen_research_text.yaml),
    # image_input_supported False.
    ok, reason = eligible_for_suite("qwen_research_text", vision_suite)
    check(not ok, "text-only model (qwen_research_text) is NOT eligible for the vision suite")
    check("image_input_supported" in reason, "ineligibility reason names the missing capability")

    ok, _ = eligible_for_suite("qwen_research_text", text_suite)
    check(ok, "text-only model (qwen_research_text) IS eligible for the text suite")

    # gemma_extract: image_input_supported True (default for any VLM
    # config) - eligible for vision suite regardless of
    # vision_validation_status (task #1/#9: untested VLMs stay eligible).
    ok, _ = eligible_for_suite("gemma_extract", vision_suite)
    check(ok, "a vision-capable model (gemma_extract) IS eligible for the vision suite")

    ok, reason = eligible_for_suite("nonexistent_model_xyz", text_suite)
    check(not ok, "an unknown model is not eligible for any suite")


def test_scorers():
    print("test_scorers")
    r = score_case("exact_match_normalized", {"expected": "PONG"}, "  pong.\n")
    check(r.status == "pass" and r.score == 1.0, "exact_match_normalized ignores case/whitespace/trailing period")

    r = score_case("exact_match_normalized", {"expected": "PONG"}, "PONGY")
    check(r.status == "fail", "exact_match_normalized fails a near-miss")

    r = score_case("exact_match_any", {"expected_any": ["5:15 PM", "17:15"]}, "17:15")
    check(r.status == "pass", "exact_match_any accepts any listed acceptable form")

    r = score_case("contains_all", {"keywords": ["Ancestor A", "Ancestor B"]}, "I see Ancestor A and Ancestor B here.")
    check(r.status == "pass", "contains_all passes when every keyword is present")
    r = score_case("contains_all", {"keywords": ["Ancestor A", "Ancestor B"]}, "I see Ancestor A only.")
    check(r.status == "fail" and 0 < r.score < 1, "contains_all gives partial credit when only some keywords are present")

    r = score_case("json_field_match",
                    {"expected": {"bridegroom": "John Test Sample", "bride": "Jane Placeholder Doe"}},
                    '{"bridegroom": "John Test Sample", "bride": "Jane Placeholder Doe"}')
    check(r.status == "pass" and r.score == 1.0, "json_field_match passes on an exact structured match")

    r = score_case("json_field_match",
                    {"expected": {"bridegroom": "John Test Sample", "bride": "Jane Placeholder Doe"}},
                    'The bridegroom is John Test Sample. (not JSON)')
    check(r.status == "fail" and r.score == 0.0, "json_field_match fails cleanly when no JSON object is present")

    r = score_case("regex_line_format", {"expected_lines": 3, "line_pattern": "^[A-Za-z]+$"}, "Red\nGreen\nBlue")
    check(r.status == "pass", "regex_line_format passes 3 clean single-word lines")
    r = score_case("regex_line_format", {"expected_lines": 3, "line_pattern": "^[A-Za-z]+$"}, "1. Red\n2. Green")
    check(r.status == "fail", "regex_line_format fails wrong line count / numbering")

    r = score_case("abstention_expected", {"refusal_keywords": ["cannot", "don't know"]},
                    "I cannot determine that from the image provided.")
    check(r.status == "pass", "abstention_expected passes on a real hedge")
    r = score_case("abstention_expected", {"refusal_keywords": ["cannot", "don't know"]},
                    "The person's name is Robert Smith.")
    check(r.status == "fail", "abstention_expected fails a confident fabricated answer")

    r = score_case("manual", {}, "some free-text description")
    check(r.status == "unscored" and r.score is None, "manual scorer marks unscored, not pass/fail")

    r = score_case("nonexistent_scorer_xyz", {}, "anything")
    check(r.status == "error", "an unknown scorer name produces an error result, not a crash")

    r = score_case("exact_match_normalized", {"expected": "PONG"}, None)
    check(r.status == "error", "no output to score (inference failure) produces an error result")


def test_persistence_roundtrip():
    print("test_persistence_roundtrip")
    from model_console import benchmark_store

    fake_case = CaseResult(
        case_id="c1", category="cat", prompt="p", image_path=None, expected_display="x",
        raw_output="PONG", status="pass", score=1.0, explanation="matched",
        runtime_seconds=0.5, telemetry={"generate": {"tokens_per_sec": 12.3, "peak_vram_mb": 1000.0}},
    )
    fake_run = BenchmarkRunResult(
        run_id="test_run_benchmark_console_unit", model_name="fake_model",
        suite_id="text_baseline", suite_version="1", suite_qualified_id="text_baseline_v1",
        backend="local", generation_settings={"do_sample": False},
        generation_config_snapshot={"model_name": "fake_model"},
        was_resident_before_run=False, load_telemetry={"load_time_s": 1.2},
        started_at="2026-08-15T00:00:00Z", finished_at="2026-08-15T00:00:01Z",
        total_runtime_seconds=1.0, status="completed", abort_reason=None,
        case_results=[fake_case],
    )
    try:
        benchmark_store.save_run(fake_run)
        loaded = benchmark_store.load_run("test_run_benchmark_console_unit")
        check(loaded is not None, "saved run can be reloaded")
        check(loaded.model_name == "fake_model", "reloaded run preserves model_name")
        check(loaded.case_results[0].raw_output == "PONG", "reloaded run preserves raw output")
        check(loaded.counts["pass"] == 1, "run.counts reflects case statuses")

        runs = benchmark_store.list_runs()
        check(any(r["run_id"] == "test_run_benchmark_console_unit" for r in runs),
              "saved run appears in list_runs() index")
    finally:
        shutil.rmtree(benchmark_store.run_dir("test_run_benchmark_console_unit"), ignore_errors=True)
        benchmark_store._refresh_index()


def test_suite_version_isolation():
    print("test_suite_version_isolation")
    suite = load_suite("text_baseline_v1")
    check(suite.qualified_id != suite.suite_id, "qualified_id is not just the bare suite_id (carries version)")
    check(suite.qualified_id.endswith(f"_v{suite.version}"), "qualified_id encodes the version explicitly")


def test_vllm_registry_metadata():
    print("test_vllm_registry_metadata (backend comparison, 2026-08-16)")
    vllm_cfg = load_model_config("qwen25_vl_7b_awq")
    check(is_vllm_model(vllm_cfg), "qwen25_vl_7b_awq is correctly identified as a vLLM-runtime config")
    transformers_cfg = load_model_config("gemma_extract")
    check(not is_vllm_model(transformers_cfg), "gemma_extract is correctly identified as NOT a vLLM-runtime config")

    # This was a REAL bug found this session: loader_class="vllm" is a
    # documented sentinel never looked up in LOADER_REGISTRY, so every
    # vLLM config failed validate_model_assignment() before is_vllm_model()
    # was added to special-case it.
    cfg = validate_model_assignment("qwen25_vl_7b_awq")
    check(cfg is not None, "validate_model_assignment() no longer rejects a valid vllm-runtime config")

    check(inference_engine_for_model("qwen25_vl_7b_awq") == "vllm", "inference_engine_for_model reads config.runtime for a vllm config")
    check(inference_engine_for_model("gemma_extract") == "transformers", "inference_engine_for_model defaults to transformers")
    check(required_backend_transport("qwen25_vl_7b_awq") == "remote", "a vllm-runtime model requires backend=remote")
    check(required_backend_transport("gemma_extract") == "local", "a plain transformers model can run backend=local")


def test_cross_engine_pairing():
    print("test_cross_engine_pairing")
    pairs = find_cross_engine_pairs()
    check(("gemma_12b_unified", "gemma_12b_w4a16") in pairs,
          "gemma_12b_unified/gemma_12b_w4a16 (same repo_id, different runtime) is detected as a cross-engine pair")
    pair_repo_ids = {load_model_config(a).repo_id for a, b in pairs} | {load_model_config(b).repo_id for a, b in pairs}
    for a, b in pairs:
        check(load_model_config(a).repo_id == load_model_config(b).repo_id,
              f"paired models {a!r}/{b!r} genuinely share repo_id (weight-identical, not just same family)")
    check(("qwen25_vl_7b", "qwen25_vl_7b_awq") not in pairs,
          "qwen25_vl_7b/qwen25_vl_7b_awq is NOT treated as a pair - AWQ quantization means different repo_id/weights")


def test_backend_comparison_result_fields():
    print("test_backend_comparison_result_fields")
    from model_console import benchmark_store

    vllm_case = CaseResult(
        case_id="c1", category="cat", prompt="p", image_path=None, expected_display="x",
        raw_output="PONG", status="pass", score=1.0, explanation="matched",
        runtime_seconds=0.5, telemetry={"generate": {"tokens_per_sec": 40.0, "vram_used_mb": 7000.0}},
    )
    vllm_run = BenchmarkRunResult(
        run_id="test_run_vllm_fields_unit", model_name="fake_vllm_model",
        suite_id="text_baseline", suite_version="1", suite_qualified_id="text_baseline_v1",
        backend="remote", inference_engine="vllm", repo_id="fake/repo-AWQ",
        generation_settings={"do_sample": False},
        generation_config_snapshot={"do_sample": False, "temperature": 0.1, "runtime": "vllm"},
        settings_translation_notes=["no_repeat_ngram_size has NO vLLM sampler equivalent - NOT applied."],
        was_resident_before_run=False, load_telemetry={"load_time_s": 180.0},
        started_at="2026-08-16T00:00:00Z", finished_at="2026-08-16T00:00:01Z",
        total_runtime_seconds=1.0, status="completed", abort_reason=None,
        case_results=[vllm_case],
    )
    try:
        check(vllm_run.inference_engine == "vllm", "inference_engine field is distinct from backend (transport)")
        check(vllm_run.peak_vram_mb is None, "peak_vram_mb is honestly None for a vllm run (no true peak counter available)")
        check(vllm_run.resident_vram_mb == 7000.0, "resident_vram_mb falls back to vram_used_mb for a vllm run")
        check(len(vllm_run.settings_translation_notes) == 1, "settings_translation_notes is persisted on the run")

        benchmark_store.save_run(vllm_run)
        loaded = benchmark_store.load_run("test_run_vllm_fields_unit")
        check(loaded.inference_engine == "vllm", "inference_engine round-trips through persistence")
        check(loaded.repo_id == "fake/repo-AWQ", "repo_id round-trips through persistence")
        check(loaded.settings_translation_notes == vllm_run.settings_translation_notes,
              "settings_translation_notes round-trips through persistence")

        # Backward compatibility: a run saved BEFORE these fields existed
        # (no inference_engine/repo_id/settings_translation_notes keys in
        # its JSON at all) must still load, defaulting sensibly.
        import json
        old_style_path = benchmark_store.run_dir("test_run_vllm_fields_unit") / "run.json"
        with open(old_style_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key in ("inference_engine", "repo_id", "settings_translation_notes"):
            data.pop(key, None)
        with open(old_style_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        old_loaded = benchmark_store.load_run("test_run_vllm_fields_unit")
        check(old_loaded.inference_engine == "transformers",
              "a pre-2026-08-16 run missing inference_engine loads and defaults to 'transformers'")
        check(old_loaded.repo_id is None, "a pre-2026-08-16 run missing repo_id loads with repo_id=None")
    finally:
        shutil.rmtree(benchmark_store.run_dir("test_run_vllm_fields_unit"), ignore_errors=True)
        benchmark_store._refresh_index()


def test_comparison_mismatch_detection():
    print("test_comparison_mismatch_detection")
    from model_console.benchmark_tab import _diff_material_settings

    same = {"do_sample": False, "temperature": 0.1, "runtime": "transformers"}
    check(_diff_material_settings(same, dict(same)) == [], "identical settings produce no diffs")

    a = {"do_sample": False, "temperature": 0.1, "top_k": 20, "runtime": "transformers"}
    b = {"do_sample": False, "temperature": 0.1, "top_k": None, "runtime": "vllm"}
    diffs = _diff_material_settings(a, b)
    check(any("runtime" in d for d in diffs), "a runtime difference is flagged")
    check(any("top_k" in d for d in diffs), "a top_k difference (e.g. vllm not receiving it) is flagged")
    check(_diff_material_settings(None, b) == [], "a missing snapshot produces no diffs (nothing to compare), not a false positive")


if __name__ == "__main__":
    test_suite_loading()
    test_eligibility_capability_gating()
    test_scorers()
    test_persistence_roundtrip()
    test_suite_version_isolation()
    test_vllm_registry_metadata()
    test_cross_engine_pairing()
    test_backend_comparison_result_fields()
    test_comparison_mismatch_detection()
    print(f"\n{_PASS} passed, {_FAIL} failed")
    sys.exit(1 if _FAIL else 0)
