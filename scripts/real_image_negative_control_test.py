"""
Real-image negative-control test for the structured_handwritten_record /
handwritten_tabular_record candidate labels (2026-08-11).

Closes the gap explicitly flagged in docs/GEMMA_NATIVE_SEMANTICS_TAXONOMY_
INVESTIGATION.md's Stage 5 "what this does NOT establish": Stage 5's
negative controls (personal letter, printed table) were TEXT-DESCRIBED
hypotheticals, not real images. This script runs the same unanchored
elicitation method against 3 REAL images:

  - POSITIVE (handwritten + tabular structure): a real cursive German
    parish-register entry, ruled columns
    data/misclassifications-adjacent corpus image, currently classified
    dense_tabular_rows in production.
  - NEGATIVE (handwritten, continuous free-form prose, NO tabular
    structure): a real 1707 Kirk Session minute page (National Records of
    Scotland), sourced manually by Jon and cropped locally - the exact
    case this investigation was missing (no genuine handwritten personal-
    letter/prose image existed anywhere in the corpus; the 8 automated
    "kirk session notes" reference-pull images were all confirmed to be
    book spines/title pages, a real bug in scripts/pull_reference_images.py
    unrelated to this test).
  - NEGATIVE (printed/typeset, tabular, NO handwriting): a real Birmingham
    electoral register page, currently classified dense_tabular_rows in
    production.

METHOD: uses the SAME unanchored elicitation prompt shape as Stage 5's Set
C (no candidate label name supplied, ask the model to independently name/
define/justify a category from visual morphology) - Stage 4/5 both found
that ANCHORED "would X belong to category Y?" membership questions induce
false-positive rationalization (Qwen2.5 accepted a hypothetical letter
into handwritten_tabular_record by inventing fake tabular structure), so
this test deliberately avoids that failure mode rather than reproducing
it. The desired boundary: the positive image should get a name/definition
combining "handwritten" + "structured/tabular/rows-columns"; both negative
images should get a name/definition that does NOT combine both of those -
the letter should read as prose/correspondence (not tabular), the register
should read as tabular/printed (not handwritten).

Reuses model_console.adapter.ChatBackendAdapter, same one-shot/no-history
discipline as scripts/cross_model_taxonomy_semantic_audit.py (Stage 5).
Models: gemma, qwen3b, qwen3vl4b, internvl3_8b - the same 4 confirmed
text_only_supported (irrelevant here since these ARE image calls, but kept
identical to Stage 5 for direct comparability) loaders used throughout
this investigation. do_sample/temperature/etc. left at each model's own
tested production values; only system_prompt (cleared) and
restrict_output_charset (disabled) overridden in memory, same
proven-necessary fix as every other stage.

MEASUREMENT ONLY. No taxonomy/config/prompt changes.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_console.adapter import ChatBackendAdapter, build_config, model_supports_text_only
from core.workspace_context import WorkspaceContext
from core.run_context import RunContext

try:
    import torch
except ImportError:
    torch = None


MODELS = ["gemma", "qwen3b", "qwen3vl4b", "internvl3_8b"]

UNANCHORED_PROMPT = (
    "You are designing a visual document classification taxonomy. "
    "Based only on its visual morphology, what classification category "
    "would you assign this image to?\n\nGive:\n1. A short machine-readable "
    "category name.\n2. A plain-language definition of that category.\n"
    "3. The visual features that caused you to choose it.\n\nClassify by "
    "visual structure rather than the subject matter or historical "
    "content."
)

IMAGES = [
    {
        "case_id": "positive_handwritten_tabular",
        "path": r"J:\Screenshots\Kemper_Ancestry\Screenshot 2026-04-30 003700.png",
        "expected_shape": "handwritten + structured/tabular (should combine both)",
    },
    {
        "case_id": "negative_handwritten_prose",
        "path": r"J:\Genealogy\genealogy_pipeline\data\outputs\manual_test_images\kirk_session_minutes_1707_nrscotland.png",
        "expected_shape": "handwritten, but NOT tabular (continuous prose)",
    },
    {
        "case_id": "negative_printed_tabular",
        "path": r"J:\Screenshots\Kemper_Ancestry\Screenshot 2026-05-02 152719.png",
        "expected_shape": "tabular, but NOT handwritten (printed/typeset)",
    },
]

MAX_NEW_TOKENS = 768


def main():
    for img in IMAGES:
        if not Path(img["path"]).exists():
            raise FileNotFoundError(f"Missing test image: {img['path']}")

    for model_name in MODELS:
        # Not required for image calls, but kept as a sanity check that
        # every model in this comparison is the same set Stage 5 used.
        model_supports_text_only(model_name)

    workspace = WorkspaceContext.resolve()
    ctx = RunContext.create(
        workspace, run_type="research",
        source_input="real_image_negative_control_test",
        run_name="real_image_negative_control_test",
    )
    results_path = ctx.reports / "results.jsonl"
    meta_path = ctx.reports / "run_metadata.json"
    print(f"Run: {ctx.run_id}")
    print(f"Results: {results_path}")

    meta = {
        "experiment": "real_image_negative_control_test",
        "run_id": ctx.run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "models": MODELS,
        "images": IMAGES,
        "prompt": UNANCHORED_PROMPT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "shared_overrides": {"system_prompt": "", "restrict_output_charset": False},
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    adapter = ChatBackendAdapter()
    total = len(MODELS) * len(IMAGES)
    done = 0

    try:
        with open(results_path, "a", encoding="utf-8") as f:
            for model_name in MODELS:
                print(f"\n=== Loading {model_name} ===")
                config = build_config(model_name, overrides={
                    "restrict_output_charset": False,
                    "max_new_tokens": MAX_NEW_TOKENS,
                })
                load_start = time.time()
                adapter.ensure_loaded(model_name, config)
                print(f"Loaded {model_name} in {time.time() - load_start:.1f}s")

                for img in IMAGES:
                    error = None
                    raw_response = ""
                    start = time.time()
                    try:
                        with Image.open(img["path"]) as im:
                            if im.mode != "RGB":
                                im = im.convert("RGB")
                            raw_response, meta_out = adapter.send_turn(
                                UNANCHORED_PROMPT, im, ""
                            )
                    except Exception as e:
                        error = f"{type(e).__name__}: {e}"
                    runtime = time.time() - start

                    record = {
                        "model_name": model_name,
                        "repo_id": config.repo_id,
                        "loader_class": config.loader_class,
                        "case_id": img["case_id"],
                        "image_path": img["path"],
                        "expected_shape": img["expected_shape"],
                        "prompt": UNANCHORED_PROMPT,
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

                    done += 1
                    status = "ERROR" if error else "OK"
                    short = raw_response[:80].replace("\n", " ")
                    print(f"[{done}/{total}] {model_name} {img['case_id']}: "
                          f"{status} ({runtime:.1f}s, {len(raw_response)} chars) -> {short!r}")

                print(f"=== Releasing {model_name} ===")
                adapter.release()
    finally:
        adapter.release()
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    meta["completed_at"] = datetime.now(timezone.utc).isoformat()
    meta["total_generations"] = done
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    ctx.mark_completed()

    print(f"\nDone: {done}/{total} generations written to {results_path}")
    print(f"Run metadata: {meta_path}")


if __name__ == "__main__":
    main()
