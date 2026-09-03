"""
Overnight batch (2026-08-06): tests the winning v7 gated-binary routing
tree (docs reference: diagnostics/test_gemma_prompt_tiering_variants.py,
census-gate-first, original wording) against EVERY other VLM loader this
project has a config for - the one test that was never done, since Gemma
already worked well enough at the time this project's routing role was
assigned and no side-by-side comparison against the other candidates was
ever run for THIS SPECIFIC ROLE (hierarchical family/layout/record-type/
census-year routing), only for whole-page classification/extraction in
earlier benchmark work.

Same 10-image ground-truth set, same 4 prompts (Tier0 family, Tier1
layout, Tier2 census/manifest gates in v7's census-first order, Tier3
census year) as the Gemma work - see diagnostics/test_gemma_prompt_
tiering.py and test_gemma_prompt_tiering_variants.py for the full history
of how these prompts were arrived at. This script does NOT re-tune
prompts per model - it deliberately reuses the exact wording that won
for Gemma, to answer "how does a different model do on the SAME task,
same prompts" rather than "what's the best each model could do with its
own tuning" (a different, much bigger question, out of scope for one
overnight run).

Design for unattended overnight execution - THIS IS THE PART THAT
MATTERS MOST, more than the routing logic itself, since a crash 2 models
in without this would waste the whole night:
  - Every model is wrapped in its own try/except at the OUTER level (load
    failure, missing deps, gated/license-required repos, OOM at load
    time) - a failure is logged and the script moves to the next model,
    never aborts the whole run.
  - Every per-image call is ALSO wrapped individually - one image
    erroring on one model doesn't lose that model's other 9 results.
  - Results are written to disk incrementally (flushed after every
    image, not buffered until the end) to a plain-text log AND a
    machine-readable JSON summary, so a script crash or process kill at
    ANY point still leaves everything done so far on disk, readable via
    the Read tool - not just whatever fits in a piped stdout tail.
  - Model released (loader.release() + model/processor/tokenizer=None +
    gc.collect() + torch.cuda.empty_cache()) after EVERY model,
    success or failure, before the next model loads - same discipline
    core/row_extraction.py's _release_model() uses, so no two models are
    ever resident in VRAM at once and a failed load can't leak into the
    next model's attempt.

Usage:
    python diagnostics/test_all_vlms_prompt_tiering.py
"""

from __future__ import annotations

import gc
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from core.loaders.base_loader import load_model_config
from core.loader_registry import LOADER_REGISTRY
from diagnostics.test_gemma_prompt_tiering import (
    TEST_CASES, WORKING_DIR, TIER0_PROMPT, TIER1_PROMPT, TIER3_PROMPT,
    parse_fields, parse_confidence, MIN_CONFIDENCE,
)
from diagnostics.test_gemma_prompt_tiering_variants import (
    GATE_MANIFEST_PROMPT, GATE_CENSUS_PROMPT,
)

OUT_DIR = PROJECT_ROOT / "data" / "outputs" / "overnight_vlm_routing_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT_DIR / "log.txt"
SUMMARY_PATH = OUT_DIR / "summary.json"

# Excludes "gemma" (already fully tested - this IS its v7 baseline, 7/10)
# and "gemma_extract" (identical weights to "gemma", only a different
# extraction-role system prompt - testing it here would mostly measure
# persona-prompt effects on the same checkpoint, not a different model,
# and isn't worth the overnight time budget). "qwen3b" excluded as a
# duplicate of "qwen2b" - both configs point at the same repo_id
# (Qwen/Qwen2.5-VL-3B-Instruct), confirmed via config/models/*.yaml.
# "chameleon" excluded from this run - already have a clean, uncontaminated
# result from before a 2026-08-06 incident where a stray orphaned process
# from an earlier failed manual nohup/disown launch attempt survived and
# ran CONCURRENTLY with the properly-backgrounded instance, both fighting
# over the same GPU (confirmed via nvidia-smi showing two python.exe
# compute processes, 100% GPU util, one process alone using 20GB system
# RAM from offload thrashing). Both killed; this run resumes from chandra
# onward with a single verified process. Chameleon's clean pre-collision
# result: 1/10, mode-collapsed to family=mixed_uncertain/confidence=0.5 on
# every single image - consistent with the prior known finding
# ([[project_chameleon_modecollapse]] in memory - predicted same category
# on all 68 test images in earlier testing too).
#
# OCR-specialized models excluded (2026-08-06, per Jon's direction) -
# chandra/glm_ocr/got_ocr2/nanonets_ocr2_3b/olmocr_2_7b/hunyuan_ocr are
# all tuned for raw text extraction, not free-form yes/no instruction-
# following - chandra's own run (before being killed for maxing the GPU
# alone, no collision this time) was already producing "<unparsed>"
# fields across the board, consistent with not answering this task's
# actual question shape at all. Not worth the overnight time budget on
# a task these models weren't built for.
#
# "gemma_12b_unified" excluded (2026-08-06, per Jon's direction) - the
# w4a16 12B checkpoint doesn't fit comfortably in this card's 16GB VRAM
# for this workload, forcing heavy CPU offload (15.7GB system RAM,
# 90-135s per prompt vs. gemma E2B's 2-5s). Killed partway through (6/10
# images done, 2 correct) - that partial result is NOT counted in the
# final comparison table, just noted here for the record. "deepseek_vl2_
# tiny", "florence_large", and "gemma_e4b" already completed successfully
# before this and are NOT re-run.
#
# "granite_vision_2b"/"granite_vision_4_1_4b" excluded entirely (2026-08-06,
# per Jon's direction) - NOT a mislabeled/wrong-size model (repo_id
# correctly resolves to the genuinely-2B ibm-granite/granite-vision-3.2-2b),
# but config/models/granite_vision_2b.yaml already documents a known
# repetition-loop degeneration bug from 2026-07-13 row-extraction testing
# (repetition_penalty/no_repeat_ngram_size were added specifically to
# mitigate it). Tonight's run showed the same signature on our tiered
# prompts - 15-21s/call (vs Gemma's 2-5s) and 100% unparseable output
# ("<unparsed>", confidence=None, malformed field values) - the existing
# mitigation isn't fully suppressing it here. Real, pre-existing model
# instability, not usable for this task as-is; both granite configs
# skipped rather than burning the rest of the overnight budget on it.
#
# "internvl3_2b" through "qwen3vl2b" already completed successfully
# before this and are NOT re-run (qwen3vl2b's last few answers degraded
# into malformed pipe-joined output - noted in the final report as a
# low-confidence result, not silently trusted).
#
# "qwen3vl2b_thinking" excluded (2026-08-06, per Jon's direction) - the
# reasoning/chain-of-thought variant produced wildly erratic per-image
# latency (66s, 675s, 1721s, 2114.9s/35min for a SINGLE image) before
# being killed partway through image 1/10 - its verbose reasoning traces
# make it impractical for this task's time budget regardless of whatever
# accuracy gain the extra reasoning might otherwise provide. Not
# re-attempted.
MODELS_TO_TEST = ["qwen3vl4b", "smolvlm2_2b"]


def log(f, msg: str) -> None:
    print(msg)
    print(msg, file=f)
    f.flush()


def release_loader(loader) -> None:
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
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def route_v7(loader, image) -> dict:
    """v7's winning tree: Tier0 family -> Tier1 layout -> census gate
    FIRST -> manifest gate second -> Tier3 census year. Every step is its
    own try/except so a single model's single weird response (e.g. a
    model that doesn't follow the output format at all) degrades to
    "<unparsed>"/quarantine for that step rather than crashing the whole
    image."""
    path = []

    def ask(prompt, label):
        try:
            raw = loader._run_generate(image, prompt)
        except Exception as e:
            return None, None, f"<call error: {type(e).__name__}: {e}>"
        fields = parse_fields(raw)
        conf = parse_confidence(fields)
        return fields, conf, raw

    f0, conf0, raw0 = ask(TIER0_PROMPT, "tier0")
    family = (f0 or {}).get("family", "<unparsed>")
    path.append(("tier0_family", family, conf0, raw0))
    if conf0 is None or conf0 < MIN_CONFIDENCE:
        return {"leaf": "quarantine_low_confidence", "path": path}
    if family != "document_content":
        return {"leaf": family, "path": path}

    f1, conf1, raw1 = ask(TIER1_PROMPT, "tier1")
    layout = (f1 or {}).get("layout", "<unparsed>")
    path.append(("tier1_layout", layout, conf1, raw1))
    if conf1 is None or conf1 < MIN_CONFIDENCE:
        return {"leaf": "quarantine_low_confidence", "path": path}
    if layout in ("narrative", "uncertain", "<unparsed>"):
        return {"leaf": f"layout_{layout}", "path": path}

    fgc, confgc, rawgc = ask(GATE_CENSUS_PROMPT, "gate_census")
    is_census = (fgc or {}).get("census", (fgc or {}).get("is_census", "<unparsed>"))
    path.append(("gate_census", is_census, confgc, rawgc))
    if confgc is None or confgc < MIN_CONFIDENCE:
        return {"leaf": "quarantine_low_confidence", "path": path}
    if is_census == "yes":
        f3, conf3, raw3 = ask(TIER3_PROMPT, "tier3")
        year = (f3 or {}).get("census_year", "<unparsed>")
        path.append(("tier3_census_year", year, conf3, raw3))
        if conf3 is None or conf3 < MIN_CONFIDENCE:
            return {"leaf": "quarantine_low_confidence", "path": path}
        return {"leaf": f"census_{year}", "path": path}

    fgm, confgm, rawgm = ask(GATE_MANIFEST_PROMPT, "gate_manifest")
    is_manifest = (fgm or {}).get("manifest", (fgm or {}).get("is_manifest", "<unparsed>"))
    path.append(("gate_manifest", is_manifest, confgm, rawgm))
    if confgm is None or confgm < MIN_CONFIDENCE:
        return {"leaf": "quarantine_low_confidence", "path": path}
    if is_manifest == "yes":
        return {"leaf": "passenger_manifest", "path": path}
    return {"leaf": "unknown", "path": path}


def is_correct(leaf: str, ground_truth: str) -> bool:
    return leaf == ground_truth or (
        ground_truth == "unknown" and leaf == "quarantine_low_confidence"
    )


def test_one_model(f, model_name: str) -> dict:
    log(f, f"\n{'=' * 70}")
    log(f, f"=== {model_name} ===")
    log(f, f"{'=' * 70}")

    t_load_start = time.time()
    loader = None
    try:
        model_cfg = load_model_config(model_name)
        model_cfg.prompt_text = ""
        loader_cls = LOADER_REGISTRY.get(model_cfg.loader_class)
        if loader_cls is None:
            raise ValueError(f"No loader registered for loader_class={model_cfg.loader_class!r}")
        loader = loader_cls(model_cfg)
        loader.initialize_model_and_tokenizer()
        load_time = time.time() - t_load_start
        log(f, f"Loaded in {load_time:.1f}s (loader_class={model_cfg.loader_class})")
    except Exception as e:
        load_time = time.time() - t_load_start
        err = f"{type(e).__name__}: {e}"
        log(f, f"LOAD FAILED after {load_time:.1f}s: {err}")
        log(f, traceback.format_exc())
        if loader is not None:
            release_loader(loader)
        return {"model": model_name, "status": "load_failed", "error": err,
                "load_time_s": load_time, "correct": 0, "total": 0, "rows": []}

    correct = 0
    total = 0
    rows = []
    try:
        for stem, ground_truth in TEST_CASES:
            img_path = WORKING_DIR / f"{stem}.jpg"
            if not img_path.exists():
                continue
            t0 = time.time()
            try:
                image = Image.open(img_path).convert("RGB")
                result = route_v7(loader, image)
                leaf = result["leaf"]
                ok = is_correct(leaf, ground_truth)
                elapsed = time.time() - t0
                total += 1
                correct += int(ok)
                mark = "OK  " if ok else "MISS"
                log(f, f"  [{mark}] {stem}  ({elapsed:.1f}s)  ground_truth={ground_truth!r}  leaf={leaf!r}")
                for tier_name, value, conf, raw in result["path"]:
                    log(f, f"        {tier_name}: {value!r} confidence={conf}")
                rows.append({"stem": stem, "ground_truth": ground_truth, "leaf": leaf,
                             "correct": ok, "elapsed_s": elapsed,
                             "path": [{"tier": t, "value": v, "confidence": c, "raw": r}
                                      for t, v, c, r in result["path"]]})
            except Exception as e:
                elapsed = time.time() - t0
                err = f"{type(e).__name__}: {e}"
                log(f, f"  [ERROR] {stem} after {elapsed:.1f}s: {err}")
                total += 1
                rows.append({"stem": stem, "ground_truth": ground_truth, "leaf": "error",
                             "correct": False, "elapsed_s": elapsed, "error": err, "path": []})
    finally:
        release_loader(loader)

    log(f, f"-> {correct}/{total}")
    return {"model": model_name, "status": "ok", "load_time_s": load_time,
            "correct": correct, "total": total, "rows": rows}


def main():
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        log(f, f"\n\n########## OVERNIGHT VLM ROUTING BATCH - started {datetime.now().isoformat()} ##########")
        log(f, f"Testing {len(MODELS_TO_TEST)} models against v7's gated-binary routing tree, "
               f"same {len(TEST_CASES)}-image ground-truth set used throughout the Gemma work.")
        log(f, "Reference baseline: gemma (v7_gated_census_first) = 7/10, already established.\n")

        all_results = []
        for i, model_name in enumerate(MODELS_TO_TEST, 1):
            log(f, f"\n[{i}/{len(MODELS_TO_TEST)}] Starting {model_name}...")
            result = test_one_model(f, model_name)
            all_results.append(result)

            # Write the summary JSON after EVERY model, not just at the end -
            # a crash on model 15/23 should still leave 1-14's full results
            # readable, not lose everything back to the start.
            with open(SUMMARY_PATH, "w", encoding="utf-8") as sf:
                json.dump(all_results, sf, indent=2, default=str)

        log(f, f"\n\n{'=' * 70}")
        log(f, "=== FINAL COMPARISON TABLE ===")
        log(f, f"{'=' * 70}")
        log(f, f"{'model':<24}{'status':<14}{'score':>8}{'load_s':>10}")
        log(f, f"{'gemma (reference)':<24}{'ok':<14}{'7/10':>8}{'~20':>10}")
        for r in all_results:
            score = f"{r['correct']}/{r['total']}" if r["total"] else "n/a"
            load_s = f"{r['load_time_s']:.0f}" if r.get("load_time_s") is not None else "?"
            log(f, f"{r['model']:<24}{r['status']:<14}{score:>8}{load_s:>10}")

        log(f, f"\nDone {datetime.now().isoformat()}. Full log: {LOG_PATH}")
        log(f, f"Machine-readable summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
