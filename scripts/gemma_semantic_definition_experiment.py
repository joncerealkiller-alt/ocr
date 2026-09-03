"""
Semantic-definition experiment against Gemma (2026-08-10, per Jon's exact
spec) - measures Gemma's OWN unprompted interpretation of each taxonomy
label, independent of this project's classifier_guidance wording, by
asking it directly and capturing every raw response verbatim.

TWO STAGES, run as two separate invocations (per Jon, 2026-08-10):
  Stage 1 (default, no flag): production archival-router system prompt
    left IN PLACE - measures Gemma's semantic reasoning WITH the
    classifier persona active, matching real production conditions.
  Stage 2 (--no-system-prompt): the SAME everything else, but
    config.extra["system_prompt"] cleared in memory for this run only -
    baseline semantics with no persona framing. Run this AFTER stage 1
    is done and recorded, never concurrently (one GPU, one model
    resident at a time - see CLAUDE.md).
Each stage writes to its OWN RunContext (run_type="research", distinct
run_id) - stage 2 never overwrites stage 1's results.jsonl/metadata, and
every per-run record carries its own "stage" field so downstream
analysis can compare the two without needing to track which file was
which by filename alone.

MEASUREMENT ONLY. Does not modify config/prompts/classifier_classify_v1.txt,
core/loaders/gemma_loader.py, config/models/gemma.yaml, or any other
production file - this is a standalone script that reuses the real
production model-loading path (same GemmaLoader, same config/models/
gemma.yaml settings: do_sample, temperature, top_p, top_k,
repetition_penalty, restrict_output_charset, reasoning_enabled all as
configured in production) but does NOT go through GemmaLoader.classify()/
_run_generate()/_build_prompt(), because those hard-require an image
(classify()'s whole interface assumes one) and a task="classify" prompt -
neither applies here. This script's _run_text_only_generate() mirrors
_run_generate()'s generation-kwargs construction line-for-line instead of
diverging from it, specifically so "normal production model/settings" is
actually true rather than approximated.

ONE explicit, flagged deviation from "identical settings" that could not
be avoided without violating "do not change code": max_new_tokens.
Production classification caps at 256 (config/models/gemma.yaml) - far
too short for a multi-part semantic-definition essay (characteristics +
accept-evidence + reject-evidence + confusable categories). Overridden
IN MEMORY ONLY (never written back to the yaml) to 2048, chosen as a
generous ceiling for a genuinely open-ended answer, not an attempt at an
actually-unbounded budget (no such thing exists for a fixed-context
model) - recorded explicitly in every saved run's metadata so this
deviation is auditable, not silently hidden.

FLAGGED, NOT OVERRIDDEN: config/models/gemma.yaml has
restrict_output_charset: true in production (core/loaders/
constrained_decoding.py's AllowedCharsLogitsProcessor - masks generation
to Latin/accented-Latin letters, digits, and a punctuation set tuned for
this project's OWN structured "key: value" output format). That
punctuation set does NOT include semicolons, parentheses, or exclamation
marks - characters a natural free-form essay response plausibly wants to
use. Left ON here because the task explicitly says "normal production
model/settings," but this is a real, live risk to the measurement's own
validity (an essay could come out subtly mangled/truncated at exactly
the punctuation marks natural prose would use) - flagged loudly in the
run metadata and in this script's own printed output, not silently
absorbed as a footnote. If any run's raw_response looks truncated or
oddly punctuated, re-run with restrict_output_charset overridden False
in a follow-up pass before concluding anything from THIS pass's content.

Labels: every core/taxonomy.py Category with enabled=True, excluding the
routing sentinel "uncertain_review" (enabled=False already, double-
excluded here defensively since it is not a real visual document
category Gemma should be asked to define) - 11 labels as of this run.

Output: a real RunContext (run_type="research" - this is a study, not a
production/dataset_build/etc. run, per docs/RUN_ARCHITECTURE.md's own
ownership table), results.jsonl under ctx.reports, append-only, one line
per run, written and flushed immediately after each generation (never
batched/held in memory) so a crash partway through never loses completed
runs. run_metadata.json alongside it records the shared experiment-wide
settings once (model, config, prompt template) rather than repeating
them in all 55 result lines.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

# Lives in scripts/ - one directory deeper than repo root, so repo root
# must be put back on sys.path before the core.* imports below resolve
# (same pattern every other scripts/*.py in this project already uses,
# e.g. scripts/run_row_extraction.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.taxonomy import load_taxonomy
from core.loaders.base_loader import load_model_config
from core.loader_registry import LOADER_REGISTRY
from core.workspace_context import WorkspaceContext
from core.run_context import RunContext

RUNS_PER_LABEL = 5
MAX_NEW_TOKENS_OVERRIDE = 2048  # see module docstring - not "unbounded", a generous explicit ceiling

PROMPT_TEMPLATE = (
    "What does the visual document classification category {LABEL} mean to "
    "you? Describe the visual and semantic characteristics you associate "
    "with this category, what evidence would make you classify an image "
    "into it, what evidence would make you reject it, and the categories "
    "or image types you would most easily confuse with it. Do not attempt "
    "to infer our intended definition. Explain your own interpretation of "
    "the label based on your existing learned representation."
)


def _get_labels() -> list[str]:
    taxonomy = load_taxonomy()
    return [
        c.id for c in taxonomy.categories
        if c.enabled and c.id != "uncertain_review"
    ]


def _run_text_only_generate(loader, prompt: str) -> str:
    """
    Mirrors GemmaLoader._run_generate()'s generation-kwargs construction
    exactly (core/loaders/gemma_loader.py) MINUS the image content block/
    images= kwarg - GemmaLoader itself has no text-only path (classify()/
    _run_generate() hard-require a real PIL image, _build_prompt() only
    supports task="classify"), and this experiment is explicitly barred
    from editing that file, so this function reimplements only what's
    needed rather than monkeypatching or duplicating the whole loader.
    Any divergence from the real _run_generate() here is a bug in THIS
    function, not an intentional deviation - kept deliberately close to
    that source for exactly this reason.
    """
    config = loader.config

    system_content = config.extra.get("system_prompt", "")
    if config.reasoning_enabled:
        system_content = "<|think|>" + system_content

    messages = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})

    text = loader.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=config.reasoning_enabled,
    )
    inputs = loader.processor(text=text, return_tensors="pt").to(loader.model.device)
    input_len = inputs["input_ids"].shape[-1]

    gen_kwargs = dict(
        max_new_tokens=config.max_new_tokens,
        do_sample=config.do_sample,
    )
    if config.do_sample:
        gen_kwargs.update(
            temperature=config.temperature, top_p=config.top_p, top_k=config.top_k,
        )
    if config.repetition_penalty and config.repetition_penalty != 1.0:
        gen_kwargs["repetition_penalty"] = config.repetition_penalty
    if config.no_repeat_ngram_size:
        gen_kwargs["no_repeat_ngram_size"] = config.no_repeat_ngram_size
    loader._maybe_add_charset_logits_processor(gen_kwargs)

    with torch.inference_mode():
        outputs = loader.model.generate(**inputs, **gen_kwargs)

    response = loader.processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    parsed = loader.processor.parse_response(response)
    final_text = parsed.get("content", parsed) if isinstance(parsed, dict) else parsed
    return str(final_text).strip()


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-system-prompt", action="store_true",
        help="Stage 2 (2026-08-10, per Jon: run stage 1 first with the production "
             "archival-router system prompt in place, then rerun with it temporarily "
             "removed for baseline semantics). Clears config.extra['system_prompt'] "
             "IN MEMORY ONLY for this run - config/models/gemma.yaml is never touched. "
             "Everything else (restrict_output_charset, sampling settings, prompt "
             "template, labels, runs-per-label) stays identical to stage 1, so the "
             "system prompt's presence/absence is the only isolated variable between "
             "the two stages' results.")
    parser.add_argument(
        "--no-charset-mask", action="store_true",
        help="Stage 2b (2026-08-10, per Jon, after stage 2 came back badly "
             "corrupted): clears config.restrict_output_charset IN MEMORY ONLY "
             "for this run. Real finding, not the mild risk originally flagged - "
             "without the system prompt's short/plain framing, Gemma's natural "
             "instinct for this open-ended question is heavily markdown-formatted "
             "prose (bold headers, numbered sections), and this project's charset "
             "mask (core/loaders/constrained_decoding.py) doesn't allow the "
             "punctuation that needs - every masked attempt collapsed generation "
             "into either repeated literal '?' tokens (6/11 labels, 8-19%% of the "
             "whole response) or an empty loop of '---' separators with zero real "
             "content (5/11 labels). Confirmed by direct inspection of all 11 "
             "stage-2 responses, not assumed. This is a separate mechanism from "
             "the system-prompt question stage 2 was actually testing, and "
             "leaving it on doesn't preserve anything meaningful when it prevents "
             "the model from answering at all - so it's turned off here rather "
             "than treated as an immovable 'production settings' constraint.")
    parser.add_argument(
        "--runs-per-label", type=int, default=RUNS_PER_LABEL,
        help=f"Default {RUNS_PER_LABEL}, matching stage 1. Stage 1's own 5-per-label "
             "results came back byte-identical across all 5 runs for every one of the "
             "11 labels (verified directly, not assumed) - production gemma.yaml has "
             "do_sample=false (pure greedy decoding), so there is no stochastic source "
             "for run-to-run variation to exist under; 5 runs measured nothing beyond "
             "what 1 run would. Per Jon's direction (2026-08-10), stage 2 defaults to "
             "--runs-per-label 1 at the call site (see the shell command that invokes "
             "this) rather than repeating that redundant compute for a stage that will "
             "produce the same duplication for the same reason.")
    args = parser.parse_args()
    stage = "baseline_no_system_prompt" if args.no_system_prompt else "with_system_prompt"
    if args.no_charset_mask:
        stage += "_no_charset_mask"
    runs_per_label = args.runs_per_label

    labels = _get_labels()
    print(f"Stage: {stage}")
    print(f"Labels ({len(labels)}): {', '.join(labels)}")
    print(f"Runs per label: {runs_per_label}  (total generations: {len(labels) * runs_per_label})")

    config = load_model_config("gemma")
    original_max_new_tokens = config.max_new_tokens
    config.max_new_tokens = MAX_NEW_TOKENS_OVERRIDE
    print(f"max_new_tokens: production={original_max_new_tokens} -> "
          f"experiment override={MAX_NEW_TOKENS_OVERRIDE} (in-memory only, "
          f"config/models/gemma.yaml on disk is untouched)")

    original_restrict_charset = config.restrict_output_charset
    if args.no_charset_mask:
        config.restrict_output_charset = False
        print(f"restrict_output_charset OVERRIDDEN for this run (in-memory only): "
              f"was {original_restrict_charset}, now False - see stage 2's real "
              f"'?'/empty-'---' degeneration finding for why.")
    elif config.restrict_output_charset:
        print("WARNING: restrict_output_charset=True (production setting, left ON per "
              "'normal production model/settings') - this can distort free-form prose "
              "at punctuation this project's charset mask doesn't allow (no semicolons/"
              "parentheses/exclamation marks). See this script's own module docstring.")

    original_system_prompt = config.extra.get("system_prompt", "")
    if args.no_system_prompt:
        config.extra["system_prompt"] = ""
        print(f"System prompt REMOVED for this run (in-memory only): was "
              f"{original_system_prompt!r}, now ''.")

    workspace = WorkspaceContext.resolve()
    ctx = RunContext.create(
        workspace, run_type="research",
        source_input=f"gemma_semantic_definition_experiment_{stage}",
        run_name=f"gemma_semantic_definition_experiment_{stage}",
    )
    results_path = ctx.reports / "results.jsonl"
    meta_path = ctx.reports / "run_metadata.json"
    print(f"Run: {ctx.run_id}")
    print(f"Results: {results_path}")

    shared_meta = {
        "experiment": "gemma_semantic_definition_experiment",
        "stage": stage,
        "run_id": ctx.run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model_name": config.model_name,
        "repo_id": config.repo_id,
        "prompt_template": PROMPT_TEMPLATE,
        "labels": labels,
        "runs_per_label": runs_per_label,
        "generation_settings": {
            "do_sample": config.do_sample,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "repetition_penalty": config.repetition_penalty,
            "no_repeat_ngram_size": config.no_repeat_ngram_size,
            "max_new_tokens": config.max_new_tokens,
            "max_new_tokens_production_value": original_max_new_tokens,
            "restrict_output_charset": config.restrict_output_charset,
            "reasoning_enabled": config.reasoning_enabled,
            "system_prompt": config.extra.get("system_prompt", ""),
        },
        "generation_config_hash": config.content_hash(),
        "no_image_provided": True,
        "no_examples_or_ground_truth_provided": True,
    }
    meta_path.write_text(json.dumps(shared_meta, indent=2), encoding="utf-8")

    loader_cls = LOADER_REGISTRY[config.loader_class]
    loader = loader_cls(config)
    loader.initialize_model_and_tokenizer()

    total = len(labels) * runs_per_label
    done = 0
    try:
        with open(results_path, "a", encoding="utf-8") as f:
            for label in labels:
                prompt = PROMPT_TEMPLATE.format(LABEL=label)
                for run_number in range(1, runs_per_label + 1):
                    start = time.time()
                    try:
                        raw_response = _run_text_only_generate(loader, prompt)
                        error = None
                    except Exception as e:
                        raw_response = ""
                        error = f"{type(e).__name__}: {e}"
                    runtime = time.time() - start

                    record = {
                        "stage": stage,
                        "label": label,
                        "run_number": run_number,
                        "prompt": prompt,
                        "model_name": config.model_name,
                        "generation_config_hash": config.content_hash(),
                        "raw_response": raw_response,
                        "error": error,
                        "runtime_seconds": round(runtime, 3),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()

                    done += 1
                    status = "ERROR" if error else "OK"
                    print(f"[{done}/{total}] {label} run {run_number}: {status} "
                          f"({runtime:.1f}s, {len(raw_response)} chars)")
    finally:
        loader.release()
        loader.model = None
        loader.processor = None
        loader.tokenizer = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    shared_meta["completed_at"] = datetime.now(timezone.utc).isoformat()
    shared_meta["total_runs"] = done
    meta_path.write_text(json.dumps(shared_meta, indent=2), encoding="utf-8")
    ctx.mark_completed()

    print(f"\nDone: {done}/{total} runs written to {results_path}")
    print(f"Run metadata: {meta_path}")


if __name__ == "__main__":
    main()
