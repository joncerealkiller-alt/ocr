"""
Benchmark 2.3 - E2B vs E4B Reference Model Comparison.

Single independent variable: Gemma 4 E2B -> Gemma 4 E4B. Everything
else is held constant by construction, not by re-derivation:
  - Same 146-image stratified corpus, same source_path per image, and
    the same 8 tower votes per image - all reused VERBATIM from the
    frozen benchmark2_3_multi_tower_routing_audit.py output
    (data/outputs/benchmark2_3_multi_tower/all_results.json), never
    recomputed. Towers are Gemma-independent, so recomputing them
    would only risk introducing drift, not add information.
  - Same prompt file, read from config/pipeline.yaml's
    classifier.prompt_file - config/pipeline.yaml's classifier.model
    field is NEVER read or touched by this script; production config
    stays exactly as it is.
  - Same sampling config, image token budget, system prompt,
    reasoning-toggle setting, restrict_output_charset - config/models/
    gemma_e4b.yaml is a byte-for-byte mirror of gemma.yaml except
    repo_id and extra.load_in_8bit (Jon's explicit, deliberate choice:
    E4B's raw bf16 weights, 16.02GB, exceed this card's 15.93GB total
    VRAM on their own - E2B is NOT quantized, this asymmetry is
    intentional and documented, not hidden).
  - Same GemmaLoader class for both - verified (not assumed) that E2B
    and E4B share the identical architecture class
    (Gemma4ForConditionalGeneration), identical processor_config.json,
    and identical chat_template.jinja (matching sha256) before this
    script was written.

Both E2B and E4B are re-classified FRESH in this script (not just E4B)
specifically so wall-clock/VRAM timing is measured under identical
current-session conditions for both - the original frozen run never
captured per-image timing at all, and mixing "old E2B categorical
answers" with "new E2B timing" from two different runs would be sloppy.
E2B's freshly-reproduced categories are verified against the frozen
original (greedy decoding, so they should match exactly) as a
determinism check before trusting anything else in this script.

This script does NOT modify benchmark2_3_multi_tower_routing_audit.py
or re-run the tower passes. Entirely observational: nothing written to
production manifests, bucket CSVs, or config/pipeline.yaml.

Usage:
    python -m benchmark.benchmark2_3_e2b_vs_e4b
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import torch
import yaml
from PIL import Image

from core.loaders.base_loader import load_model_config
from core.loader_registry import LOADER_REGISTRY

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FROZEN_RESULTS = PROJECT_ROOT / "data" / "outputs" / "benchmark2_3_multi_tower" / "all_results.json"
OUT_DIR = PROJECT_ROOT / "data" / "outputs" / "benchmark2_3_e2b_vs_e4b"

MODEL_CONFIGS = ["gemma", "gemma_e4b"]


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
    """Same VRAM-release discipline as core/row_extraction.py's own
    _release_model - loader.release() alone is a no-op hook here and
    does NOT free VRAM (the real bug that broke the first Gemma-12B
    comparison run in this same benchmark lineage)."""
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


def run_model_with_timing(model_name: str, loader, corpus: list[dict]) -> tuple[list[dict], dict]:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    results = []
    wall_start = time.perf_counter()
    for i, entry in enumerate(corpus, 1):
        img = Image.open(entry["source_path"]).convert("RGB")
        t0 = time.perf_counter()
        result = loader.classify(entry["source_path"], img)
        elapsed = time.perf_counter() - t0
        results.append({
            "image": entry["image"], "category": result.category.value,
            "confidence": result.confidence, "reason": result.reason,
            "elapsed_seconds": round(elapsed, 3),
        })
        print(f"  [{model_name}] [{i}/{len(corpus)}] {entry['image']:<45} "
              f"-> {result.category.value:<20} conf={result.confidence:.2f}  ({elapsed:.2f}s)")
    total_wall = time.perf_counter() - wall_start

    peak_vram_mb = None
    if torch.cuda.is_available():
        peak_vram_mb = round(torch.cuda.max_memory_allocated() / 1e6, 1)

    timing = {
        "total_wall_seconds": round(total_wall, 2),
        "mean_seconds_per_image": round(total_wall / len(corpus), 3),
        "peak_vram_mb": peak_vram_mb,
    }
    return results, timing


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frozen = json.loads(FROZEN_RESULTS.read_text(encoding="utf-8"))
    corpus = [{"image": r["image"], "source_path": r["source_path"], "sample_bucket": r["sample_bucket"]}
              for r in frozen]
    frozen_by_image = {r["image"]: r for r in frozen}
    print(f"Loaded {len(corpus)} images from the frozen Benchmark 2.3 multi-tower run "
          f"(reusing exact source_path + tower votes, not recomputing).\n")

    prompt_text = load_pipeline_prompt()

    all_timing = {}
    fresh_results = {}
    for model_name in MODEL_CONFIGS:
        print(f"=== Loading {model_name} ===")
        loader = build_loader(model_name, prompt_text)
        try:
            results, timing = run_model_with_timing(model_name, loader, corpus)
            fresh_results[model_name] = {r["image"]: r for r in results}
            all_timing[model_name] = timing
        finally:
            _release_model(loader)
        print()

    # --- determinism check: E2B's fresh reproduction vs the frozen original ---
    print("=== Determinism check: fresh E2B reproduction vs frozen original ===")
    mismatches = []
    for image, r in fresh_results["gemma"].items():
        orig = frozen_by_image[image]["gemma"]
        if r["category"] != orig["bucket"] or abs(r["confidence"] - orig["confidence"]) > 1e-6:
            mismatches.append((image, orig, r))
    if mismatches:
        print(f"  WARNING: {len(mismatches)}/{len(corpus)} images do NOT match the frozen original:")
        for image, orig, r in mismatches:
            print(f"    {image}: frozen={orig['bucket']}/{orig['confidence']} "
                  f"fresh={r['category']}/{r['confidence']}")
    else:
        print(f"  Clean: all {len(corpus)} images reproduce the frozen original exactly "
              f"(greedy decoding, same model/prompt/image - as expected).")
    print()

    # --- E2B vs E4B comparison ---
    e2b = fresh_results["gemma"]
    e4b = fresh_results["gemma_e4b"]
    agree = sum(1 for img in e2b if e2b[img]["category"] == e4b[img]["category"])
    n = len(corpus)
    print(f"=== E2B vs E4B agreement: {agree}/{n} ({agree/n:.1%}) ===\n")

    disagreements = [
        {"image": img, "sample_bucket": frozen_by_image[img]["sample_bucket"],
         "e2b_category": e2b[img]["category"], "e2b_confidence": e2b[img]["confidence"],
         "e4b_category": e4b[img]["category"], "e4b_confidence": e4b[img]["confidence"],
         "tower_consensus_bucket": frozen_by_image[img]["tower_consensus_bucket"],
         "tower_consensus_strength": frozen_by_image[img]["tower_consensus_strength"]}
        for img in e2b if e2b[img]["category"] != e4b[img]["category"]
    ]

    # of those disagreements, does E4B move TOWARD or AWAY FROM tower consensus?
    toward_consensus = sum(1 for d in disagreements if d["e4b_category"] == d["tower_consensus_bucket"])
    away_from_consensus = sum(1 for d in disagreements
                               if d["e2b_category"] == d["tower_consensus_bucket"]
                               and d["e4b_category"] != d["tower_consensus_bucket"])
    print(f"Of {len(disagreements)} E2B/E4B disagreements:")
    print(f"  E4B matches tower consensus (E2B didn't):   {toward_consensus}")
    print(f"  E4B leaves tower consensus (E2B matched it): {away_from_consensus}")
    print(f"  Neither matches tower consensus:              "
          f"{len(disagreements) - toward_consensus - away_from_consensus}\n")

    # --- performance specifically on the existing 56 consensus-disagreement cases ---
    consensus_disagreement_images = {r["image"] for r in frozen if not r["gemma_agrees_with_tower_consensus"]}
    print(f"=== Performance on the existing {len(consensus_disagreement_images)} "
          f"consensus-disagreement cases (E2B already disagreed with tower consensus there) ===")
    e4b_now_agrees = sum(1 for img in consensus_disagreement_images
                         if e4b[img]["category"] == frozen_by_image[img]["tower_consensus_bucket"])
    print(f"  E4B agrees with tower consensus on: {e4b_now_agrees}/{len(consensus_disagreement_images)} "
          f"(E2B agreed with tower consensus on: 0/{len(consensus_disagreement_images)}, by definition)\n")

    # --- confidence: more confident vs more correct ---
    mean_conf_e2b = sum(r["confidence"] for r in e2b.values()) / n
    mean_conf_e4b = sum(r["confidence"] for r in e4b.values()) / n
    print(f"=== Confidence comparison ===")
    print(f"  Mean confidence E2B: {mean_conf_e2b:.4f}")
    print(f"  Mean confidence E4B: {mean_conf_e4b:.4f}")
    if disagreements:
        mean_conf_e4b_on_disagreements = sum(d["e4b_confidence"] for d in disagreements) / len(disagreements)
        mean_conf_e2b_on_disagreements = sum(d["e2b_confidence"] for d in disagreements) / len(disagreements)
        print(f"  On the {len(disagreements)} disagreement images specifically:")
        print(f"    E2B mean confidence: {mean_conf_e2b_on_disagreements:.4f}")
        print(f"    E4B mean confidence: {mean_conf_e4b_on_disagreements:.4f}")
    print()

    # --- runtime / VRAM / throughput cost ---
    print("=== Runtime / VRAM / throughput cost (E2B -> E4B) ===")
    for m in MODEL_CONFIGS:
        t = all_timing[m]
        print(f"  [{m}] total={t['total_wall_seconds']}s  "
              f"mean/image={t['mean_seconds_per_image']}s  peak_vram_mb={t['peak_vram_mb']}")
    print()

    # --- save everything ---
    with open(OUT_DIR / "fresh_results.json", "w", encoding="utf-8") as f:
        json.dump(fresh_results, f, indent=2)
    with open(OUT_DIR / "disagreements.json", "w", encoding="utf-8") as f:
        json.dump(disagreements, f, indent=2)
    summary = {
        "n_total": n,
        "agreement": f"{agree}/{n}",
        "determinism_check_mismatches": len(mismatches),
        "n_disagreements": len(disagreements),
        "e4b_toward_tower_consensus": toward_consensus,
        "e4b_away_from_tower_consensus": away_from_consensus,
        "e4b_agreement_on_existing_consensus_disagreements": f"{e4b_now_agrees}/{len(consensus_disagreement_images)}",
        "mean_confidence": {"gemma_e2b": round(mean_conf_e2b, 4), "gemma_e4b": round(mean_conf_e4b, 4)},
        "timing": all_timing,
        "note": "Observational only - config/pipeline.yaml (production classifier.model) untouched.",
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Full results: {OUT_DIR}/")


if __name__ == "__main__":
    main()
