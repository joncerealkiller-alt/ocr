"""
Cross-model visual-taxonomy semantic audit (2026-08-11, per Jon's exact
spec). Automated, one-shot, text-only battery run against every locally
available (in this repo) confirmed text_only_supported VLM loader, to see
whether the semantic-label mismatches already found on Gemma (see
docs/GEMMA_NATIVE_SEMANTICS_TAXONOMY_INVESTIGATION.md) reproduce across
model FAMILIES, not just Gemma checkpoints - continuing the cross-model
triangulation Jon has been running externally (Gemma 3 on-device, Qwen2.5
on-device) by adding our own local, instrumented, one-shot run.

MEASUREMENT ONLY. Does not touch config/taxonomy.yaml, any model's on-disk
yaml, or any production prompt file. Every override below is applied to an
in-memory GenerationConfig copy only (see model_console/adapter.py's
build_config(), which already guarantees this).

Reuses model_console.adapter.ChatBackendAdapter rather than re-deriving
per-loader generation code: that adapter already provides the exact
"ensure_loaded (releases whatever was previously resident first) / one
free-form text-only turn, no history retained / release" cycle this audit
needs, and is already the ONLY sanctioned non-core module allowed to call
loader._run_generate() directly outside of core/ itself. Each call to
send_turn() builds a brand-new single-turn message list inside the loader
(see e.g. gemma_loader.py's message construction) - nothing here or in the
adapter accumulates conversation history, so "one-shot, no carried state"
is satisfied by construction, not by extra bookkeeping in this script.

MODELS TESTED (all have an empirically-set text_only_supported: true in
their config/models/*.yaml - never assumed from architecture alone, per
Jon's explicit instruction):
  - gemma          (google/gemma-4-E2B-it)   - production model
  - qwen3b         (Qwen/Qwen2.5-VL-3B-Instruct) - same checkpoint family
    Jon independently tested on-device, direct comparison point
  - qwen3vl4b      (Qwen/Qwen3-VL-4B-Instruct) - different Qwen generation
  - internvl3_8b   (OpenGVLab/InternVL3-8B-hf) - different model family
    entirely (not Gemma, not Qwen)

Gemma 3 (phone/Google on-device) and Qwen2.5 (phone on-device) are NOT
re-tested here - those are already recorded as external/manual evidence in
the doc above; fabricating a local stand-in for a model this repo cannot
actually load would corrupt the evidence trail. Do not add a "Gemma 3"
label to any record produced by this script.

GENERATION SETTINGS: do_sample/temperature/top_p/top_k/repetition_penalty/
no_repeat_ngram_size are left at each model's own tested production values
(all already do_sample=false / greedy in every config tested) - not
touched, per Jon's "preserve each model's appropriate tested generation
configuration" instruction. Two settings ARE overridden in memory, for
every model, for the same reason already proven on Gemma (see Stage 2 in
the taxonomy-investigation doc): with no system prompt and free-form prose,
a charset mask collapses generation into repeated fallback tokens, so:
  - system_prompt: "" (no persona/framing - want raw pretrained priors)
  - restrict_output_charset: False (avoid the proven collapse mode)
max_new_tokens is raised per test-set (cold-definitions need much more
budget than the short unanchored-elicitation answers) - see
MAX_NEW_TOKENS_BY_SET below. This is the one difference from a single
global override, and is applied identically across all four models so it
cannot bias any one model's results.

REPETITION-LOOP HANDLING: no special early-stop logic is implemented here
(would require per-token streaming hooks this project's loaders don't
expose). If a model loops, the full max_new_tokens budget for that test's
set is spent and the raw (looped) output is recorded verbatim - the loop
itself is evidence, not an error to hide. A human reviewing the JSONL can
identify a loop by inspecting the raw_response field directly (as was done
manually for Gemma's handwritten_ledger loop in the doc's Stage 1).

OUTPUT: JSONL, one line per generation, flushed immediately - same
crash-safety discipline as every other script in this investigation.
"""

from __future__ import annotations

import gc
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_console.adapter import ChatBackendAdapter, build_config, model_supports_text_only
from core.workspace_context import WorkspaceContext
from core.run_context import RunContext

try:
    import torch
except ImportError:
    torch = None


MODELS = ["gemma", "qwen3b", "qwen3vl4b", "internvl3_8b"]

DEFINE_PROMPT_TEMPLATE = (
    'What does the visual document classification category "{CATEGORY}" '
    "mean to you? Define the category and list the visual characteristics "
    "or trigger features you would associate with it."
)

# ---- Test Set A: existing production labels ----
SET_A_CATEGORIES = [
    "cemetery_photo",
    "casual_photo",
    "photo_collage",
    "handwritten_ledger",
]

# ---- Test Set B: candidate replacement labels ----
SET_B_CATEGORIES = [
    "structured_handwritten_record",
    "handwritten_tabular_record",
]

# ---- Test Set C: unanchored morphology elicitation ----
SET_C_TESTS = [
    (
        "census",
        "You are designing a visual document classification taxonomy. You "
        "are shown an image of a historical handwritten Canadian census "
        "page. The page contains repeated person entries arranged in rows, "
        "with demographic attributes recorded under fixed column headings "
        "and visible table structure.\n\nBased only on its visual "
        "morphology, what classification category would you assign this "
        "image to?\n\nGive:\n1. A short machine-readable category name.\n"
        "2. A plain-language definition of that category.\n3. The visual "
        "features that caused you to choose it.\n\nClassify by visual "
        "structure rather than the historical subject matter.",
    ),
    (
        "marriage_register",
        "You are designing a visual document classification taxonomy. You "
        "are shown an image of a historical handwritten marriage register "
        "containing repeated entries organized into predefined fields, "
        "rows, and columns.\n\nBased only on its visual morphology, what "
        "classification category would you assign this image to?\n\nGive:"
        "\n1. A short machine-readable category name.\n2. A plain-language "
        "definition of that category.\n3. The visual features that caused "
        "you to choose it.\n\nClassify by visual structure rather than the "
        "historical subject matter.",
    ),
    (
        "passenger_manifest",
        "You are designing a visual document classification taxonomy. You "
        "are shown an image of a handwritten passenger manifest containing "
        "repeated passenger entries arranged in rows with attributes "
        "recorded under fixed column headings.\n\nBased only on its visual "
        "morphology, what classification category would you assign this "
        "image to?\n\nGive:\n1. A short machine-readable category name.\n"
        "2. A plain-language definition of that category.\n3. The visual "
        "features that caused you to choose it.\n\nClassify by visual "
        "structure rather than the historical subject matter.",
    ),
    (
        "personal_letter",
        "You are designing a visual document classification taxonomy. You "
        "are shown an image of a handwritten personal letter consisting "
        "primarily of continuous cursive prose written across ordinary "
        "lined paper.\n\nBased only on its visual morphology, what "
        "classification category would you assign this image to?\n\nGive:"
        "\n1. A short machine-readable category name.\n2. A plain-language "
        "definition of that category.\n3. The visual features that caused "
        "you to choose it.\n\nClassify by visual structure rather than the "
        "subject matter.",
    ),
    (
        "printed_table",
        "You are designing a visual document classification taxonomy. You "
        "are shown an image of a printed form containing repeated records "
        "arranged in rows and columns under fixed headings. The document "
        "contains no handwriting.\n\nBased only on its visual morphology, "
        "what classification category would you assign this image to?\n\n"
        "Give:\n1. A short machine-readable category name.\n2. A "
        "plain-language definition of that category.\n3. The visual "
        "features that caused you to choose it.\n\nClassify by visual "
        "structure rather than the subject matter.",
    ),
]

# ---- Test Set D: unanchored superclass discovery ----
SET_D_PROMPT = (
    "These three visual document types share a common visual morphology:\n"
    "\n- a handwritten census page containing repeated records in rows "
    "under fixed column headings\n- a handwritten marriage register "
    "containing repeated entries in predefined fields and columns\n- a "
    "handwritten passenger manifest containing repeated passenger records "
    "in rows and columns\n\nA handwritten personal letter containing "
    "continuous prose should NOT belong to the same visual category.\n\nA "
    "printed table containing no handwriting should NOT belong to the same "
    "visual category.\n\nPropose a single short machine-readable visual "
    "classification category name that covers the first three but "
    "excludes the final two.\n\nBase the category name on shared visual "
    "morphology rather than subject matter."
)

MAX_NEW_TOKENS_BY_SET = {
    "A": 2048,
    "B": 2048,
    "C": 768,
    "D": 512,
}


def build_test_battery() -> list[dict]:
    tests = []
    for cat in SET_A_CATEGORIES:
        tests.append({
            "set": "A",
            "test_id": f"A_{cat}",
            "category": cat,
            "system_prompt": "",
            "user_prompt": DEFINE_PROMPT_TEMPLATE.format(CATEGORY=cat),
        })
    for cat in SET_B_CATEGORIES:
        tests.append({
            "set": "B",
            "test_id": f"B_{cat}",
            "category": cat,
            "system_prompt": "",
            "user_prompt": DEFINE_PROMPT_TEMPLATE.format(CATEGORY=cat),
        })
    for name, prompt in SET_C_TESTS:
        tests.append({
            "set": "C",
            "test_id": f"C_{name}",
            "category": None,
            "system_prompt": "",
            "user_prompt": prompt,
        })
    tests.append({
        "set": "D",
        "test_id": "D_superclass_discovery",
        "category": None,
        "system_prompt": "",
        "user_prompt": SET_D_PROMPT,
    })
    return tests


def main():
    battery = build_test_battery()
    print(f"Test battery: {len(battery)} prompts x {len(MODELS)} models = "
          f"{len(battery) * len(MODELS)} total generations")

    for model_name in MODELS:
        if not model_supports_text_only(model_name):
            raise RuntimeError(
                f"{model_name!r} does not declare text_only_supported: true "
                f"in its config - refusing to silently skip or fake a "
                f"result. Fix MODELS or the config."
            )

    workspace = WorkspaceContext.resolve()
    ctx = RunContext.create(
        workspace, run_type="research",
        source_input="cross_model_taxonomy_semantic_audit",
        run_name="cross_model_taxonomy_semantic_audit",
    )
    results_path = ctx.reports / "results.jsonl"
    meta_path = ctx.reports / "run_metadata.json"
    print(f"Run: {ctx.run_id}")
    print(f"Results: {results_path}")

    meta = {
        "experiment": "cross_model_taxonomy_semantic_audit",
        "run_id": ctx.run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "models": MODELS,
        "test_ids": [t["test_id"] for t in battery],
        "max_new_tokens_by_set": MAX_NEW_TOKENS_BY_SET,
        "shared_overrides": {
            "system_prompt": "",
            "restrict_output_charset": False,
        },
        "note": "do_sample/temperature/top_p/top_k/repetition_penalty/"
                "no_repeat_ngram_size left at each model's own tested "
                "production values - not overridden.",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    adapter = ChatBackendAdapter()
    total_done = 0
    total = len(battery) * len(MODELS)

    try:
        with open(results_path, "a", encoding="utf-8") as f:
            for model_name in MODELS:
                print(f"\n=== Loading {model_name} ===")
                base_config = build_config(model_name, overrides={
                    "restrict_output_charset": False,
                })
                load_start = time.time()
                adapter.ensure_loaded(model_name, base_config)
                print(f"Loaded {model_name} in {time.time() - load_start:.1f}s")

                for test in battery:
                    max_new_tokens = MAX_NEW_TOKENS_BY_SET[test["set"]]
                    config = build_config(model_name, overrides={
                        "restrict_output_charset": False,
                        "max_new_tokens": max_new_tokens,
                    })
                    adapter.ensure_loaded(model_name, config)  # same model -> in-place config swap, no reload

                    error = None
                    raw_response = ""
                    meta_out = {}
                    start = time.time()
                    try:
                        raw_response, meta_out = adapter.send_turn(
                            test["user_prompt"], None, test["system_prompt"]
                        )
                    except Exception as e:
                        error = f"{type(e).__name__}: {e}"
                        traceback.print_exc()
                    runtime = time.time() - start

                    record = {
                        "model_name": model_name,
                        "repo_id": config.repo_id,
                        "loader_class": config.loader_class,
                        "test_id": test["test_id"],
                        "set": test["set"],
                        "category": test["category"],
                        "system_prompt": test["system_prompt"],
                        "user_prompt": test["user_prompt"],
                        "raw_response": raw_response,
                        "generation_settings": {
                            "do_sample": config.do_sample,
                            "temperature": config.temperature,
                            "top_p": config.top_p,
                            "top_k": config.top_k,
                            "repetition_penalty": config.repetition_penalty,
                            "no_repeat_ngram_size": config.no_repeat_ngram_size,
                            "max_new_tokens": config.max_new_tokens,
                            "restrict_output_charset": config.restrict_output_charset,
                        },
                        "runtime_seconds": round(runtime, 3),
                        "response_chars": len(raw_response),
                        "error": error,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()

                    total_done += 1
                    status = "ERROR" if error else "OK"
                    short = raw_response[:80].replace("\n", " ")
                    print(f"[{total_done}/{total}] {model_name} {test['test_id']}: "
                          f"{status} ({runtime:.1f}s, {len(raw_response)} chars) -> {short!r}")

                print(f"=== Releasing {model_name} ===")
                adapter.release()
    finally:
        adapter.release()
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    meta["completed_at"] = datetime.now(timezone.utc).isoformat()
    meta["total_generations"] = total_done
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    ctx.mark_completed()

    print(f"\nDone: {total_done}/{total} generations written to {results_path}")
    print(f"Run metadata: {meta_path}")


if __name__ == "__main__":
    main()
