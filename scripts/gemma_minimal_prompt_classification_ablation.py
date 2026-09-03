"""
Image classification ablation against Gemma (2026-08-11, per Jon's exact
direction) - "essentially nothing except: Classify this image." Tests
whether Gemma's OWN natural-language classification of a REAL image
(not a self-reported definition - see scripts/gemma_semantic_definition_
experiment.py, the text-only companion to this script) clusters toward
a more accurate taxonomy than the current production categories, using
known confident-wrong cases (data/misclassifications.csv,
data/outputs/kemper_ancestry_review.csv) as the highest-value test set -
the real Kemper memorial-plaque images that got called printed_document/
casual_photo under the current schema-heavy prompt are in this set.

MEASUREMENT ONLY - same discipline as the semantic-definition experiment.
Does not modify config/prompts/classifier_classify_v1.txt, core/loaders/
gemma_loader.py, or config/models/gemma.yaml. Calls GemmaLoader._run_
generate(raw_image, prompt) directly with prompt="Classify this image."
instead of going through classify()/_build_prompt() (which hard-require
config/prompts/classifier_classify_v1.txt's full taxonomy prompt) or
_parse_classification() (which requires the "category:/confidence:/
reason:" schema this experiment deliberately does NOT ask for - the
whole point is seeing what Gemma says with NO schema imposed).

Settings, per the SAME two real findings from the semantic-definition
experiment (not re-litigated here, just applied directly):
  - No system prompt (config.extra["system_prompt"] cleared in memory) -
    "essentially nothing except" the one instruction, matching Jon's
    exact ask, and per that experiment's own finding that the system
    prompt didn't change the underlying semantic content anyway (checked
    directly on handwritten_ledger/cemetery_photo, stage 1 vs 2b were
    identical) - only formatting/tone, which isn't what's being measured
    here.
  - restrict_output_charset overridden False in memory - the semantic-
    definition experiment's stage 2 proved this collapses free-form,
    no-system-prompt generation into repeated '?' tokens or empty output
    loops. A short classification response is less likely to hit this
    than a 2000-token essay was, but there's no reason to re-risk it
    when the fix is free and already proven.
  - max_new_tokens: 512 (in memory only) - generous for a short natural
    classification response (not the 2048 the essay experiment needed),
    kept well above production's 256 in case Gemma wants to explain its
    reasoning, not just emit a bare label.

Output: RunContext (run_type="research"), results.jsonl under
ctx.reports, one line per image, flushed immediately per image - same
crash-safety discipline as the semantic-definition experiment. Each
record carries the image's EXISTING production classification (category/
confidence/reason, pulled from the source CSV) alongside the new minimal-
prompt raw response, so comparison doesn't require a second join pass
later.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.loaders.base_loader import load_model_config
from core.loader_registry import LOADER_REGISTRY
from core.workspace_context import WorkspaceContext
from core.run_context import RunContext

MINIMAL_PROMPT = "Classify this image."
MAX_NEW_TOKENS_OVERRIDE = 512

# Real flagged confident-wrong cases (data/misclassifications.csv) -
# includes the actual Kemper memorial-plaque images Jon referenced,
# called printed_document/casual_photo under the current schema-heavy
# production prompt. Highest diagnostic value: smallest set, known
# failures, direct test of "does the minimal prompt do better here."
MISCLASSIFICATIONS_CSV = Path("data/misclassifications.csv")

# Full real production classification review set (487 images, current
# expanded taxonomy) - for the broader "do natural outputs cluster into
# a coherent alternate taxonomy across the corpus" question. Optional
# via --full-corpus (see main()) - the misclassifications set alone
# answers the specific plaque hypothesis quickly; this answers the
# clustering hypothesis but costs much more GPU time.
FULL_CORPUS_CSV = Path("data/outputs/kemper_ancestry_review.csv")


def _load_source_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-corpus", action="store_true",
        help=f"Use {FULL_CORPUS_CSV} (487 images) instead of the default "
             f"{MISCLASSIFICATIONS_CSV} (20 known confident-wrong images, "
             "includes the real Kemper plaque cases) - much larger GPU-time "
             "cost, run this only after the small set confirms the approach "
             "works and Jon wants the broader clustering signal.")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N rows of the chosen source CSV - for "
             "a quick smoke test before committing to the full set.")
    args = parser.parse_args()

    source_csv = FULL_CORPUS_CSV if args.full_corpus else MISCLASSIFICATIONS_CSV
    rows = _load_source_rows(source_csv)
    if args.limit:
        rows = rows[:args.limit]
    print(f"Source: {source_csv} ({len(rows)} image(s))")

    config = load_model_config("gemma")
    original_max_new_tokens = config.max_new_tokens
    config.max_new_tokens = MAX_NEW_TOKENS_OVERRIDE
    print(f"max_new_tokens: production={original_max_new_tokens} -> "
          f"experiment override={MAX_NEW_TOKENS_OVERRIDE} (in-memory only)")

    original_restrict_charset = config.restrict_output_charset
    config.restrict_output_charset = False
    print(f"restrict_output_charset OVERRIDDEN (in-memory only): "
          f"was {original_restrict_charset}, now False - see the "
          f"semantic-definition experiment's proven '?'-collapse finding.")

    original_system_prompt = config.extra.get("system_prompt", "")
    config.extra["system_prompt"] = ""
    print(f"System prompt REMOVED (in-memory only): was "
          f"{original_system_prompt!r}, now '' - \"essentially nothing "
          f"except\" the one instruction, per Jon's exact direction.")

    workspace = WorkspaceContext.resolve()
    ctx = RunContext.create(
        workspace, run_type="research",
        source_input=f"gemma_minimal_prompt_classification_ablation_{source_csv.stem}",
        run_name=f"gemma_minimal_prompt_classification_ablation_{source_csv.stem}",
    )
    results_path = ctx.reports / "results.jsonl"
    meta_path = ctx.reports / "run_metadata.json"
    print(f"Run: {ctx.run_id}")
    print(f"Results: {results_path}")

    shared_meta = {
        "experiment": "gemma_minimal_prompt_classification_ablation",
        "run_id": ctx.run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(source_csv),
        "image_count": len(rows),
        "model_name": config.model_name,
        "repo_id": config.repo_id,
        "prompt": MINIMAL_PROMPT,
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
        "no_taxonomy_or_schema_provided": True,
    }
    meta_path.write_text(json.dumps(shared_meta, indent=2), encoding="utf-8")

    loader_cls = LOADER_REGISTRY[config.loader_class]
    loader = loader_cls(config)
    loader.initialize_model_and_tokenizer()

    total = len(rows)
    done = 0
    try:
        with open(results_path, "a", encoding="utf-8") as f:
            for row in rows:
                file_path = row["file_path"]
                start = time.time()
                error = None
                raw_response = ""
                try:
                    with Image.open(file_path) as img:
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        raw_response = loader._run_generate(img, MINIMAL_PROMPT)
                except Exception as e:
                    error = f"{type(e).__name__}: {e}"
                runtime = time.time() - start

                record = {
                    "file_path": file_path,
                    "production_category": row.get("category"),
                    "production_confidence": row.get("confidence"),
                    "production_reason": row.get("reason"),
                    "minimal_prompt": MINIMAL_PROMPT,
                    "raw_response": raw_response,
                    "error": error,
                    "runtime_seconds": round(runtime, 3),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

                done += 1
                status = "ERROR" if error else "OK"
                short_resp = raw_response[:80].replace("\n", " ")
                print(f"[{done}/{total}] {Path(file_path).name}: {status} "
                      f"({runtime:.1f}s) prod={row.get('category')} -> {short_resp!r}")
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
