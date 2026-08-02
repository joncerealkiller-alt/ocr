"""
Round 4: NaFlex native-resolution vs. fixed-resolution SigLIP,
aspect-ratio robustness (Architecture D research, see
docs/VISION_IR_RESEARCH.md). Deliberately NOT part of the frozen
Vision Qualification Battery v1.0 (benchmark/vision_encoder_qualification.py)
- that battery's resolution transforms are isotropic only and cannot
test this question, per the gap flagged after Round 3.

Real finding that shaped this script's design (checked directly before
writing this, not assumed): timm's default create_transform() for
naflexvit_base_patch16_siglip.v2_webli still does Resize+CenterCrop to
a square, exactly like a fixed-resolution model - it does NOT exercise
NaFlex's actual variable-resolution capability. That requires a
different call path entirely: ResizeKeepRatioToSequence (aspect-
preserving resize to a patch/sequence budget) -> patchify_image()
(-> patches, patch_coord, patch_valid) -> model.forward(patches,
patch_coord=..., patch_valid=...). Confirmed working end-to-end against
a real corpus image before being wired in here.

Two comparisons, not one:
  1. Same-model ablation (NaFlex only) - does native vs. squared
     preprocessing of the SAME image diverge more as aspect ratio gets
     more extreme? This isolates "does native resolution matter" from
     "is this just a different, better checkpoint."
  2. Cross-model comparison (NaFlex-native vs. SigLIP-base-squared) -
     the practically relevant question, using each real image's own
     DocumentCategory bucket as a (weak, Gemma-predicted, not
     manually-verified) proxy for "should retrieve similarly" - same
     caveat as Experiment 2's cluster-stability design, acceptable
     here since nothing is being trained against these labels, only
     used as an internal same-vs-different anchor.

Usage:
    python -m benchmark.vision_encoder_aspect_ratio_experiment
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import timm
from torchvision import transforms as T
from PIL import Image
from timm.data.naflex_transforms import ResizeKeepRatioToSequence, patchify_image

from benchmark.vision_encoder_qualification import (
    sample_real_images, cosine_sim, build_model_and_transform, embed_pooled,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_LOG = PROJECT_ROOT / "data" / "outputs" / "vision_encoder_aspect_ratio_log.jsonl"

NAFLEX_CANDIDATE = "naflexvit_base_patch16_siglip.v2_webli"
FIXED_CANDIDATE = "vit_base_patch16_siglip_224.v2_webli"
PATCH_SIZE = 16
MAX_SEQ_LEN = 576   # matches the 384x384 input used for NaFlex in Rounds 1-3
TOP_K = 5

_to_tensor = T.ToTensor()
_normalize = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
_resize_keep_ratio = ResizeKeepRatioToSequence(
    patch_size=PATCH_SIZE, max_sequence_len=MAX_SEQ_LEN, interpolation="bicubic"
)


@torch.no_grad()
def embed_naflex_native(model, pil_image: Image.Image) -> np.ndarray:
    """NaFlex's real native-resolution path - aspect-preserving resize
    to a patch/sequence budget, patchify, forward with patch_coord/
    patch_valid. NOT the same as embed_pooled() used in Rounds 1-3,
    which forces a square crop for this same model."""
    img = pil_image.convert("RGB")
    resized = _resize_keep_ratio(img)
    x = _normalize(_to_tensor(resized))
    patches, patch_coord, patch_valid = patchify_image(
        x, patch_size=(PATCH_SIZE, PATCH_SIZE), pad=True, include_info=True
    )
    out = model(
        patches.unsqueeze(0),
        patch_coord=patch_coord.unsqueeze(0),
        patch_valid=patch_valid.unsqueeze(0),
    )
    return out.squeeze(0).numpy()


def bucket_precision_at_k(query_name, query_bucket, embeddings, name_to_bucket, k):
    sims = [(n, cosine_sim(embeddings[query_name], v)) for n, v in embeddings.items() if n != query_name]
    sims.sort(key=lambda t: t[1], reverse=True)
    top = [n for n, _ in sims[:k]]
    same_bucket = sum(1 for n in top if name_to_bucket[n] == query_bucket)
    return same_bucket / k


SCALED_UP_IMAGES_PER_CLUSTER = 15  # up from Round 4's 4 - handwritten_ledger caps at 12 (all real files), others get 15


def main():
    clusters = sample_real_images(images_per_cluster=SCALED_UP_IMAGES_PER_CLUSTER)
    all_images: dict[str, tuple[str, Path]] = {}
    for bucket, paths in clusters.items():
        for p in paths:
            all_images[str(p)] = (bucket, p)
    names = list(all_images.keys())
    name_to_bucket = {n: b for n, (b, _) in all_images.items()}
    print(f"Total images: {len(names)}\n")

    naflex_model, naflex_transform = build_model_and_transform(NAFLEX_CANDIDATE)
    fixed_model, fixed_transform = build_model_and_transform(FIXED_CANDIDATE)

    aspect_ratios: dict[str, float] = {}
    naflex_native: dict[str, np.ndarray] = {}
    naflex_squared: dict[str, np.ndarray] = {}
    siglip_squared: dict[str, np.ndarray] = {}

    for name, (bucket, path) in all_images.items():
        img = Image.open(path).convert("RGB")
        w, h = img.size
        aspect_ratios[name] = w / h
        naflex_native[name] = embed_naflex_native(naflex_model, img)
        naflex_squared[name] = embed_pooled(naflex_model, naflex_transform, img)
        siglip_squared[name] = embed_pooled(fixed_model, fixed_transform, img)

    # --- 1. same-model ablation: native vs squared divergence vs aspect-ratio extremity ---
    ablation = []
    for name in names:
        ar = aspect_ratios[name]
        extremity = abs(np.log(ar))  # log so 2:1 and 1:2 are symmetric around 0
        sim = cosine_sim(naflex_native[name], naflex_squared[name])
        ablation.append((name, ar, extremity, sim))

    extremities = np.array([e for _, _, e, _ in ablation])
    sims = np.array([s for _, _, _, s in ablation])
    ablation_corr = float(np.corrcoef(extremities, sims)[0, 1])

    # --- 2. cross-model: same-bucket retrieval precision vs aspect-ratio extremity ---
    per_image_results = []
    for name in names:
        bucket = name_to_bucket[name]
        ar = aspect_ratios[name]
        extremity = abs(np.log(ar))
        prec_naflex_native = bucket_precision_at_k(name, bucket, naflex_native, name_to_bucket, TOP_K)
        prec_siglip_squared = bucket_precision_at_k(name, bucket, siglip_squared, name_to_bucket, TOP_K)
        per_image_results.append({
            "name": name, "bucket": bucket, "aspect_ratio": ar, "log_extremity": extremity,
            "naflex_native_precision_at_5": prec_naflex_native,
            "siglip_squared_precision_at_5": prec_siglip_squared,
        })

    extremities2 = np.array([r["log_extremity"] for r in per_image_results])
    naflex_precisions = np.array([r["naflex_native_precision_at_5"] for r in per_image_results])
    siglip_precisions = np.array([r["siglip_squared_precision_at_5"] for r in per_image_results])

    naflex_extremity_corr = float(np.corrcoef(extremities2, naflex_precisions)[0, 1])
    siglip_extremity_corr = float(np.corrcoef(extremities2, siglip_precisions)[0, 1])

    print("--- Same-model ablation (NaFlex native vs. squared) ---")
    print(f"correlation(aspect-ratio extremity, native-vs-squared similarity) = {ablation_corr:.3f}")
    print("(negative = more extreme aspect ratio -> native and squared diverge MORE, i.e. native mode is doing something different specifically where it should matter)\n")

    print("--- Cross-model: same-bucket precision@5 vs. aspect-ratio extremity ---")
    print(f"mean precision@5  NaFlex-native:   {naflex_precisions.mean():.3f}")
    print(f"mean precision@5  SigLIP-squared:  {siglip_precisions.mean():.3f}")
    print(f"correlation(extremity, precision)  NaFlex-native:  {naflex_extremity_corr:.3f}")
    print(f"correlation(extremity, precision)  SigLIP-squared: {siglip_extremity_corr:.3f}")
    print("(more negative = precision degrades more as aspect ratio gets more extreme)\n")

    print(f"{'name':<70} {'ar':>6} {'extremity':>10} {'naflex_p@5':>11} {'siglip_p@5':>11}")
    for r in sorted(per_image_results, key=lambda r: r["log_extremity"], reverse=True):
        short_name = Path(r["name"]).name
        print(f"{short_name:<70} {r['aspect_ratio']:>6.2f} {r['log_extremity']:>10.3f} "
              f"{r['naflex_native_precision_at_5']:>11.2f} {r['siglip_squared_precision_at_5']:>11.2f}")

    RESULTS_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment": "round_4_naflex_native_vs_fixed_siglip_aspect_ratio",
        "num_images": len(names),
        "ablation_extremity_vs_native_squared_similarity_corr": ablation_corr,
        "naflex_native_mean_precision_at_5": float(naflex_precisions.mean()),
        "siglip_squared_mean_precision_at_5": float(siglip_precisions.mean()),
        "naflex_native_extremity_precision_corr": naflex_extremity_corr,
        "siglip_squared_extremity_precision_corr": siglip_extremity_corr,
        "per_image": per_image_results,
    }
    with open(RESULTS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"\nAppended full record to {RESULTS_LOG}")


if __name__ == "__main__":
    main()
