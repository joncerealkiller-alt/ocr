"""
Reference Classifier Qualification: Gemma 4 E2B (current reference
classifier, MoE + separate vision encoder) vs Gemma 4 12B Unified
QAT (Dense, encoder-free direct patch projection - compressed-tensors
w4a16, google/gemma-4-12B-it-qat-w4a16-ct).

Objective is NOT "which model is bigger/better in general" - it's
whether the architectural difference materially improves ROUTING
behavior specifically, on:
  1. The 4 known Gemma-input-qualification images (docs/BENCHMARK2_3_
     GEMMA_INPUT_QUALIFICATION.md). Ground truth is tracked with an
     explicit ground_truth_status, NOT silently redefined by whatever
     either model happens to answer:
       - c10264.767: status=UNDER_REVIEW, not "corrected". The original
         note (benchmark/gemma_token_budget_test.py: "true=
         handwritten_ledger") only checked the handwriting axis. Direct
         re-inspection (2026-08-01) shows the image is genuinely
         dual-natured: handwritten cursive content, organized in a
         clear repeating 3-column table (Immigrant/Farmer/P.Office,
         ~28 rows) - matching the ORIGINAL production bucket assignment
         (dense_tabular_rows). This benchmark does NOT resolve which of
         handwritten_ledger / dense_tabular_rows is "the" right answer
         - that a model (12B) happened to answer dense_tabular_rows is
         not evidence that dense_tabular_rows is correct; it's what
         prompted the review in the first place, and the question stays
         open pending separate adjudication. The one part that IS still
         status=CONFIRMED: printed_document is wrong - zero printed/
         typed text is visible, and Gemma 2B calls this printed_document
         at every input condition tested this session (Variables 1-6).
       - c10264.658: status=CONFIRMED, correct=printed_document (typed
         application form, stable correct control throughout)
       - c10301.601: status=AMBIGUOUS - genuinely mixed-content image
         (two photos, people+horses), no forced single answer, recorded
         but not scored right/wrong
       - genealogy_chart_screenshot_color: status=CONFIRMED,
         correct=genealogy_chart (added for Variable 4's color confound
         check, not a hard case)
  2. Benchmark 2.2's 64 real Gemma-vs-tower disagreement images
     (data/outputs/benchmark2_2/disagreements/) - human_verdict is
     still null for all of these (that review is paused), so this
     script does NOT claim ground-truth correctness on this set. What
     it measures instead (tagged INFERENCE, not MEASURED-correct) is a
     directional signal: does 12B's call match the ORIGINAL 2B call,
     match the independent TOWER call instead, or land on a third
     answer entirely.

Both models run through the identical, already-qualified Gemma input
pipeline (native resolution, no pre-resize - confirmed production-safe
in Variable 1 of the input qualification), same prompt file, same
sampling config, same image token budget, same system prompt. Only the
model/repo_id/loader differ - every other variable held constant per
the qualification's explicit design.

Run sequentially (2B fully, release, then 12B fully) rather than
concurrently: 12B(w4a16, ~10.3GB) + 2B resident together would leave
thin headroom on a 16GB card, and there's no need to risk it when
sequential loading costs only two model-load waits, not two full
corpus passes.

Usage:
    python -m benchmark.reference_classifier_qualification
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml
from PIL import Image

from core.loaders.base_loader import load_model_config
from core.loader_registry import LOADER_REGISTRY

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "outputs" / "reference_classifier_qualification"
DISAGREEMENTS_DIR = PROJECT_ROOT / "data" / "outputs" / "benchmark2_2" / "disagreements"

MODEL_CONFIGS = ["gemma", "chameleon"]

# (label, path, ground_truth dict)
# ground_truth_status is one of:
#   "confirmed"     - a single correct category, independently verified
#   "under_review"  - an earlier ground-truth assertion has been called
#                      into question (by direct re-inspection, NOT by
#                      either model's answer) and is not yet resolved;
#                      candidates lists the categories under
#                      consideration for the record, it is NOT an
#                      "any of these counts as correct" set
#   "ambiguous"      - genuinely no single correct answer expected
# confirmed_wrong (optional) - categories independently ruled out
# regardless of ground_truth_status, e.g. "not printed text" is
# confirmed even while the right bucket name is still under review.
QUALIFICATION_IMAGES = [
    ("c10264.767_known_misclassified", PROJECT_ROOT / "data/working/oocihm.lac_reel_c10264.767.jpg",
     {"ground_truth_status": "under_review",
      "candidates": {"dense_tabular_rows", "handwritten_ledger"},
      "confirmed_wrong": {"printed_document"}}),
    ("c10301.601_known_ambiguous", PROJECT_ROOT / "data/working/oocihm.lac_reel_c10301.601.jpg",
     {"ground_truth_status": "ambiguous", "candidates": set(), "confirmed_wrong": set()}),
    ("c10264.658_known_correct_control", PROJECT_ROOT / "data/working/oocihm.lac_reel_c10264.658.jpg",
     {"ground_truth_status": "confirmed", "candidates": {"printed_document"}, "confirmed_wrong": set()}),
    ("genealogy_chart_screenshot_color", PROJECT_ROOT / "data/working/Screenshot 2026-05-03 182722.png",
     {"ground_truth_status": "confirmed", "candidates": {"genealogy_chart"}, "confirmed_wrong": set()}),
]


def load_pipeline_prompt() -> str:
    with open(PROJECT_ROOT / "config" / "pipeline.yaml", "r", encoding="utf-8") as f:
        pipeline_cfg = yaml.safe_load(f)
    prompt_path = PROJECT_ROOT / pipeline_cfg["classifier"]["prompt_file"]
    return prompt_path.read_text(encoding="utf-8")


def build_loader(model_name: str, prompt_text: str):
    model_cfg = load_model_config(model_name)
    model_cfg.prompt_text = prompt_text
    loader_cls = LOADER_REGISTRY[model_cfg.loader_class]
    loader = loader_cls(model_cfg)
    loader.initialize_model_and_tokenizer()
    return loader


def _release_model(loader) -> None:
    """
    Same VRAM-release discipline as core/row_extraction.py's own
    _release_model (module-private by convention, duplicated rather
    than imported - see that function's docstring). loader.release()
    alone is a no-op hook for both Gemma loaders here - it does NOT
    null out model/processor/tokenizer or call torch.cuda.empty_cache(),
    so calling only that (a real bug caught the hard way: the first run
    of this script crashed loading the 12B because the 2B's VRAM was
    never actually freed, leaving less real headroom than device_map=
    "auto" assumed, which forced partial CPU offload and then a crash
    in compressed-tensors' decompression hook on the resulting meta
    tensors) leaves the previous model fully resident.
    """
    import gc
    import torch
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


def load_disagreement_corpus() -> list[dict]:
    corpus = []
    for folder in sorted(DISAGREEMENTS_DIR.iterdir()):
        record_path = folder / "record.json"
        if not record_path.exists():
            continue
        record = json.loads(record_path.read_text(encoding="utf-8"))
        corpus.append({
            "disagreement_id": folder.name,
            "source_path": record["source_path"],
            "gemma_bucket_original": record["gemma_bucket"],
            "gemma_confidence_original": record["gemma_confidence"],
            "tower_bucket": record["tower_bucket"],
            "tower_score": record["tower_score"],
        })
    return corpus


def run_model_over_corpus(model_name: str, loader, qualification_images, disagreement_corpus) -> list[dict]:
    results = []
    for label, path, ground_truth in qualification_images:
        img = Image.open(path).convert("RGB")
        result = loader.classify(str(path), img)
        results.append({
            "set": "qualification", "image": label, "path": str(path),
            "model": model_name, "category": result.category.value,
            "confidence": result.confidence, "reason": result.reason,
            "ground_truth_status": ground_truth["ground_truth_status"],
            "ground_truth_candidates": sorted(ground_truth["candidates"]),
            "ground_truth_confirmed_wrong": sorted(ground_truth["confirmed_wrong"]),
        })
        print(f"  [{model_name}] {label:<38} -> {result.category.value:<20} conf={result.confidence:.2f}")

    for entry in disagreement_corpus:
        img = Image.open(entry["source_path"]).convert("RGB")
        result = loader.classify(entry["source_path"], img)
        results.append({
            "set": "disagreement", "image": entry["disagreement_id"], "path": entry["source_path"],
            "model": model_name, "category": result.category.value,
            "confidence": result.confidence, "reason": result.reason,
            "gemma_bucket_original": entry["gemma_bucket_original"],
            "tower_bucket": entry["tower_bucket"],
        })
        print(f"  [{model_name}] {entry['disagreement_id']:<20} -> {result.category.value:<20} "
              f"conf={result.confidence:.2f}  (2B_orig={entry['gemma_bucket_original']}, tower={entry['tower_bucket']})")
    return results


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prompt_text = load_pipeline_prompt()
    disagreement_corpus = load_disagreement_corpus()
    print(f"Loaded {len(disagreement_corpus)} disagreement-corpus images, "
          f"{len(QUALIFICATION_IMAGES)} qualification images.\n")

    all_results = []
    for model_name in MODEL_CONFIGS:
        print(f"=== Loading {model_name} ===")
        loader = build_loader(model_name, prompt_text)
        try:
            results = run_model_over_corpus(model_name, loader, QUALIFICATION_IMAGES, disagreement_corpus)
            all_results.extend(results)
        finally:
            _release_model(loader)
        print()

    with open(OUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # -- Summary --------------------------------------------------------
    by_model = {m: [r for r in all_results if r["model"] == m] for m in MODEL_CONFIGS}
    qual_by_model = {m: [r for r in by_model[m] if r["set"] == "qualification"] for m in MODEL_CONFIGS}
    dis_by_model = {m: [r for r in by_model[m] if r["set"] == "disagreement"] for m in MODEL_CONFIGS}

    print("=== Qualification images: CONFIRMED ground truth only (pass/fail) ===")
    for m in MODEL_CONFIGS:
        correct, scored = 0, 0
        for r in qual_by_model[m]:
            if r["ground_truth_status"] != "confirmed":
                continue
            scored += 1
            is_correct = r["category"] in r["ground_truth_candidates"]
            if is_correct:
                correct += 1
            print(f"  [{m}] {r['image']:<38} predicted={r['category']:<20} "
                  f"truth={sorted(r['ground_truth_candidates'])}  {'CORRECT' if is_correct else 'WRONG'}")
        print(f"  [{m}] {correct}/{scored} correct on CONFIRMED-ground-truth images\n")

    print("=== Qualification images: UNDER_REVIEW / AMBIGUOUS - observation only, NOT scored ===")
    print("    (candidates are what's under consideration, not an 'any of these is correct' set -")
    print("     a model matching a candidate is NOT treated as evidence the candidate is right)")
    for m in MODEL_CONFIGS:
        for r in qual_by_model[m]:
            if r["ground_truth_status"] == "confirmed":
                continue
            note = "matches a candidate under review" if r["category"] in r["ground_truth_candidates"] else \
                   ("matches a previously CONFIRMED-wrong category" if r["category"] in r["ground_truth_confirmed_wrong"] else
                    "matches neither - a new data point for the open question, nothing more")
            print(f"  [{m}] {r['image']:<38} predicted={r['category']:<20} conf={r['confidence']:.2f}  "
                  f"status={r['ground_truth_status']:<12} ({note})")
    print()

    print("=== Disagreement corpus (n=%d): directional shift, NOT correctness (no human_verdict yet) ===" % len(disagreement_corpus))
    other_model = [m for m in MODEL_CONFIGS if m != "gemma"][0]
    dis_2b = {r["image"]: r for r in dis_by_model["gemma"]}
    dis_other = {r["image"]: r for r in dis_by_model[other_model]}
    matches_2b_original, matches_tower_instead, third_answer, same_as_2b_rerun = 0, 0, 0, 0
    for image_id, rother in dis_other.items():
        r2b = dis_2b[image_id]
        if rother["category"] == r2b["category"]:
            same_as_2b_rerun += 1
        if rother["category"] == rother["gemma_bucket_original"]:
            matches_2b_original += 1
        elif rother["category"] == rother["tower_bucket"]:
            matches_tower_instead += 1
        else:
            third_answer += 1
    n = len(dis_other)
    print(f"  {other_model} reproduces 2B's ORIGINAL bucket call: {matches_2b_original}/{n}")
    print(f"  {other_model} matches TOWER's call instead (where 2B and tower disagreed): {matches_tower_instead}/{n}")
    print(f"  {other_model} lands on a third answer (neither 2B-original nor tower): {third_answer}/{n}")
    print(f"  {other_model} == 2B (this run, re-classified fresh): {same_as_2b_rerun}/{n}")

    print(f"\nFull results written to {OUT_DIR}/results.json")


if __name__ == "__main__":
    main()
