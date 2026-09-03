"""
Text-only E2B vs E4B research-role comparison.

Modeled directly on benchmark/benchmark2_3_e2b_vs_e4b.py (build_loader/
_release_model pattern reused as-is), but scoped to the TEXT-ONLY
research path instead of vision/classification:

  - Calls loader._run_generate(None, prompt) - no image - instead of
    loader.classify(path, img).
  - Uses each call's Phase 3 per-call telemetry (loader.
    last_inference_telemetry, core/loaders/base_loader.py) instead of a
    single whole-run peak_vram_mb the way Benchmark 2.3 measured it.
  - Compares "gemma" (E2B) against "gemma_e4b_research" (config/models/
    gemma_e4b_research.yaml), NOT "gemma_e4b" (that config is pinned to
    Benchmark 2.3's controlled classification comparison - see its own
    header comment - and isn't text-only-confirmed).
  - Uses only small STRUCTURAL smoke prompts (the same "echo PONG" /
    factual-question style already used to confirm text_only_supported
    for E2B, 2026-08-10) - deliberately NOT genealogy-specific
    evaluation questions. Jon is supplying those separately; this
    script's job is "does it load and run correctly, what does it cost,"
    not "is it good at the actual task."

Prerequisite this script assumes but does NOT check for you: gemma_e4b_
research.yaml's text_only_supported flag must already be True (i.e. a
real _run_generate(None, prompt) smoke test against that exact config
has already been run and produced sensible output - see that YAML's own
header comment). This script will still run structurally if it's False
(text_only_supported gates model_console's UI/adapter, not GemmaLoader
._run_generate() itself, which has no such check), but treat a run
against an unconfirmed config as PART OF gathering that confirmation,
not as evidence the confirmation already happened.

Entirely observational, same discipline as benchmark2_3_e2b_vs_e4b.py:
nothing written to production manifests, config/pipeline.yaml, or
pipeline.db.

Usage:
    python -m benchmark.benchmark_research_e2b_vs_e4b
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import torch

from core.loaders.base_loader import load_model_config
from core.loader_registry import LOADER_REGISTRY
from core.workspace_context import WorkspaceContext

OUT_DIR = WorkspaceContext.resolve().workspace_root / "research" / "benchmarks" / "research_e2b_vs_e4b"

MODEL_CONFIGS = ["gemma", "gemma_e4b_research"]

# Structural smoke prompts only - not a quality/capability benchmark.
# "echo" and "factual" mirror the exact style already used (2026-08-10,
# base_loader.py's text_only_supported docstring) to confirm gemma.yaml's
# text-only path: an instruction-following check and a knowledge check,
# both with an unambiguous right answer so a wrong/garbled response is
# obviously a real failure, not a judgment call.
SMOKE_PROMPTS = {
    "echo": "Reply with exactly one word: PONG.",
    "factual": "What is the capital of France? Answer in one word.",
    "longer_instruction": (
        "In exactly three sentences, explain what a census enumerator's "
        "job was in the early 20th century."
    ),
}


def build_loader(model_name: str):
    model_cfg = load_model_config(model_name)
    loader_cls = LOADER_REGISTRY[model_cfg.loader_class]
    loader = loader_cls(model_cfg)
    loader.initialize_model_and_tokenizer()
    return loader


def _release_model(loader) -> None:
    """Same VRAM-release discipline as benchmark2_3_e2b_vs_e4b.py's own
    _release_model - loader.release() alone does NOT free VRAM."""
    try:
        loader.release()
    except Exception:
        pass
    try:
        loader.model = None
        loader.processor = None
        loader.tokenizer = None
    except Exception:
        pass
    del loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_model(model_name: str) -> dict:
    baseline_vram_mb = None
    if torch.cuda.is_available():
        baseline_vram_mb = round(torch.cuda.memory_allocated() / 1e6, 1)
    load_start = time.perf_counter()
    loader = build_loader(model_name)
    load_time_s = round(time.perf_counter() - load_start, 3)
    after_load_vram_mb = None
    after_load_reserved_mb = None
    if torch.cuda.is_available():
        after_load_vram_mb = round(torch.cuda.memory_allocated() / 1e6, 1)
        after_load_reserved_mb = round(torch.cuda.memory_reserved() / 1e6, 1)

    per_prompt = {}
    try:
        for label, prompt in SMOKE_PROMPTS.items():
            t0 = time.perf_counter()
            response = loader._run_generate(None, prompt)
            elapsed = time.perf_counter() - t0
            print(f"  [{model_name}] [{label}] ({elapsed:.2f}s) -> {response!r}")
            per_prompt[label] = {
                "prompt": prompt,
                "response": response,
                "elapsed_seconds": round(elapsed, 3),
                "telemetry": loader.last_inference_telemetry,
            }
    finally:
        _release_model(loader)

    return {
        "model_name": model_name,
        "load": {
            "baseline_vram_mb": baseline_vram_mb,
            "after_load_vram_mb": after_load_vram_mb,
            "after_load_reserved_mb": after_load_reserved_mb,
            "load_time_s": load_time_s,
        },
        "prompts": per_prompt,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for model_name in MODEL_CONFIGS:
        print(f"=== {model_name} ===")
        results[model_name] = run_model(model_name)
        print()

    print("=== Load-time VRAM/timing comparison ===")
    for model_name in MODEL_CONFIGS:
        load = results[model_name]["load"]
        print(f"  [{model_name}] after_load={load['after_load_vram_mb']}MB "
              f"reserved={load['after_load_reserved_mb']}MB load_time={load['load_time_s']}s")
    print()

    print("=== Per-prompt generation VRAM/timing comparison ===")
    for label in SMOKE_PROMPTS:
        print(f"  -- {label} --")
        for model_name in MODEL_CONFIGS:
            t = results[model_name]["prompts"][label]["telemetry"] or {}
            print(f"    [{model_name}] prefill={t.get('vram_after_prefill_mb')}MB/"
                  f"{t.get('prefill_time_s')}s peak={t.get('peak_vram_mb')}MB "
                  f"gen={t.get('generated_tokens')}tok/{t.get('generation_time_s')}s "
                  f"({t.get('tokens_per_sec')}tok/s) stop={t.get('stop_reason')}")
    print()

    with open(OUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Full results: {OUT_DIR}/")


if __name__ == "__main__":
    main()
