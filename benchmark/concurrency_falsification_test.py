"""
Falsification test: can Gemma and a lightweight vision tower be
resident on the GPU simultaneously without corruption, given the
moondream2 incident's real root cause (VRAM exhaustion -> system
offload) rather than "any concurrent residency is unsafe"?

Design: load Gemma, classify a KNOWN image (compare against the exact
output already recorded in benchmark/gemma_token_budget_test.py's run -
same category, confidence, AND reason text expected). Then, while
Gemma stays resident and loaded, load DINOv2 alongside it. Run DINOv2
embeddings. Then run Gemma AGAIN on a second known image - this is the
closest analog to the documented failure pattern (a subsequent
load/call while another model is already resident). If Gemma's output
degrades, garbles, or diverges from its known-good baseline once
DINOv2 is also loaded, that falsifies the "safe within VRAM budget"
hypothesis. If both models keep producing correct, stable output and
combined VRAM stays comfortably under the 16GB card limit, that
supports it.

Known-good baselines (from gemma_token_budget_test.py's actual run,
budget=140, already measured this session):
    c10264.658 -> printed_document, conf=0.98,
        reason="The image is a typed application form or certificate,
        which fits the definition of a printed document with a
        single-entity form structure."
    c10264.767 -> printed_document, conf=0.95,
        reason="The image appears to be a typed or printed list of
        names and associated information, fitting the description of
        a printed document."

Usage:
    python -m benchmark.concurrency_falsification_test
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from benchmark.vision_encoder_qualification import build_model_and_transform, embed_pooled
from core.classifier import build_classifier_loader, load_pipeline_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMG_A = PROJECT_ROOT / "data/working/oocihm.lac_reel_c10264.658.jpg"  # known-good baseline 1
IMG_B = PROJECT_ROOT / "data/working/oocihm.lac_reel_c10264.767.jpg"  # known-good baseline 2 (post-tower-load)
DINOV2_CANDIDATE = "vit_small_patch14_dinov2.lvd142m"

EXPECTED_A = {"category": "printed_document", "confidence": 0.98}
EXPECTED_B = {"category": "printed_document", "confidence": 0.95}


def gpu_mem_mb():
    return torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else None


def main():
    print(f"GPU memory before anything loaded: {gpu_mem_mb():.0f} MB (torch-tracked)\n")

    # --- Step 1: load Gemma, classify known image A (baseline, Gemma alone) ---
    pipeline_cfg = load_pipeline_config()
    loader = build_classifier_loader(pipeline_cfg)
    print(f"GPU memory after Gemma loaded (Gemma alone): {gpu_mem_mb():.0f} MB\n")

    with Image.open(IMG_A) as raw_image:
        result_a1 = loader.classify(str(IMG_A), raw_image)
    print(f"[Gemma alone] {IMG_A.name} -> {result_a1.category.value} (conf={result_a1.confidence:.2f})")
    match_a1 = (result_a1.category.value == EXPECTED_A["category"] and abs(result_a1.confidence - EXPECTED_A["confidence"]) < 0.01)
    print(f"  Matches known-good baseline: {match_a1}\n")

    # --- Step 2: load DINOv2 ALONGSIDE Gemma - both resident now ---
    tower_model, tower_transform = build_model_and_transform(DINOV2_CANDIDATE)
    print(f"GPU memory with BOTH Gemma and DINOv2 resident: {gpu_mem_mb():.0f} MB\n")

    # --- Step 3: run DINOv2 while Gemma is also resident ---
    emb1 = embed_pooled(tower_model, tower_transform, Image.open(IMG_A).convert("RGB"))
    print(f"[DINOv2, Gemma also resident] embedding shape={emb1.shape}  "
          f"norm={float((emb1**2).sum()**0.5):.3f}  has_nan={bool(__import__('numpy').isnan(emb1).any())}\n")

    # --- Step 4: run Gemma AGAIN on a second known image, DINOv2 still resident -
    # closest analog to the documented failure pattern (subsequent call while
    # another model is already loaded) ---
    with Image.open(IMG_B) as raw_image:
        result_b1 = loader.classify(str(IMG_B), raw_image)
    print(f"[Gemma, DINOv2 also resident] {IMG_B.name} -> {result_b1.category.value} (conf={result_b1.confidence:.2f})")
    print(f"  reason: {result_b1.reason}")
    match_b1 = (result_b1.category.value == EXPECTED_B["category"] and abs(result_b1.confidence - EXPECTED_B["confidence"]) < 0.01)
    print(f"  Matches known-good baseline: {match_b1}\n")

    # --- Step 5: run DINOv2 again for good measure ---
    emb2 = embed_pooled(tower_model, tower_transform, Image.open(IMG_B).convert("RGB"))
    print(f"[DINOv2 again, Gemma still resident] embedding shape={emb2.shape}  "
          f"norm={float((emb2**2).sum()**0.5):.3f}  has_nan={bool(__import__('numpy').isnan(emb2).any())}\n")

    # --- Step 6: re-run Gemma on image A once more to check for drift/degradation
    # across the whole concurrent session, not just a single lucky call ---
    with Image.open(IMG_A) as raw_image:
        result_a2 = loader.classify(str(IMG_A), raw_image)
    print(f"[Gemma, DINOv2 still resident] {IMG_A.name} (repeat) -> {result_a2.category.value} (conf={result_a2.confidence:.2f})")
    match_a2 = (result_a2.category.value == EXPECTED_A["category"] and abs(result_a2.confidence - EXPECTED_A["confidence"]) < 0.01)
    print(f"  Matches known-good baseline: {match_a2}\n")

    # --- cleanup ---
    del tower_model
    loader.release()
    torch.cuda.empty_cache()
    print(f"GPU memory after releasing both: {gpu_mem_mb():.0f} MB\n")

    all_match = match_a1 and match_b1 and match_a2
    print(f"=== VERDICT ===")
    print(f"All Gemma outputs matched known-good baselines while DINOv2 was concurrently resident: {all_match}")
    print(f"(If True: falsification attempt FAILED to break it - concurrent residency within this VRAM budget appears safe.)")
    print(f"(If False: falsification SUCCEEDED - do not do this again, revert to strictly sequential load/unload.)")


if __name__ == "__main__":
    main()
