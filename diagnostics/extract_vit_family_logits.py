"""
Extracts FULL class-logit vectors (not just the winning label) for every
fine-tuned model in the multi-architecture benchmark, on the shared
held-out test set, from the ALREADY-SAVED checkpoints - no retraining.
Per Jon's decision-engine sensor-complementarity request (2026-08-07):
"preserve the full class-logit vector, not merely the winning
probability" and "make sure the benchmark outputs retain enough
per-image information that we won't have to rerun every model later."

Covers all 7 fine-tuned models from the benchmark, including vit21k
(confirmed via diff to share the EXACT same split - same seed, same
copied split code, same dataset CSV - see conversation) even though its
checkpoint was produced by a separate earlier script.

For each model, for each held-out test image, saves:
  - full logit vector (raw, pre-softmax) in class order
  - softmax probability vector
  - top-1 / top-2 logits and their margin
  - entropy of the softmax distribution
  - predicted class, ground truth, correctness

Output: one JSON per model at
data/outputs/vit_family_benchmark/<name>/<name>_test_logits.json
(vit21k's goes to data/outputs/vit_family_benchmark/vit21k/ - a NEW
folder, since its checkpoint lives elsewhere and was never given a
benchmark subfolder).

Usage:
    python diagnostics/extract_vit_family_logits.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import timm
import torch
from PIL import Image

from diagnostics.vit_family_benchmark_common import (
    BENCHMARK_ROOT, DEVICE, get_split,
)

CHECKPOINTS = [
    {"name": "convnext", "path": BENCHMARK_ROOT / "convnext" / "convnext_best_holdout_eval.pt"},
    {"name": "mobilenetv2", "path": BENCHMARK_ROOT / "mobilenetv2" / "mobilenetv2_best_holdout_eval.pt"},
    {"name": "dinov2", "path": BENCHMARK_ROOT / "dinov2" / "dinov2_best_holdout_eval.pt"},
    {"name": "beit", "path": BENCHMARK_ROOT / "beit" / "beit_best_holdout_eval.pt"},
    {"name": "swin", "path": BENCHMARK_ROOT / "swin" / "swin_best_holdout_eval.pt"},
    {"name": "siglip", "path": BENCHMARK_ROOT / "siglip" / "siglip_best_holdout_eval.pt"},
    {"name": "vit21k", "path": PROJECT_ROOT / "data" / "outputs" / "vit21k_doc_classifier_checkpoints" / "best_holdout_eval.pt"},
]


def softmax_entropy(probs: np.ndarray) -> float:
    p = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def main():
    print(f"Device: {DEVICE}")
    by_cat, train_items, val_items, test_items = get_split()
    classes = sorted(by_cat.keys())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"Classes: {classes}")
    print(f"Held-out test set: {len(test_items)} images\n")

    for cfg in CHECKPOINTS:
        name, ckpt_path = cfg["name"], cfg["path"]
        if not ckpt_path.exists():
            print(f"[{name}] SKIP - checkpoint not found: {ckpt_path}")
            continue
        print(f"[{name}] Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt_classes = ckpt["classes"]
        if ckpt_classes != classes:
            print(f"[{name}] WARNING: checkpoint class order {ckpt_classes} != {classes}, skipping")
            continue
        timm_tag = ckpt.get("timm_tag")
        if timm_tag is None:
            # vit21k's checkpoint didn't record timm_tag explicitly
            timm_tag = "vit_base_patch16_224.orig_in21k"

        model = timm.create_model(timm_tag, pretrained=False, num_classes=len(classes))
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(DEVICE)
        model.eval()
        cfg_data = timm.data.resolve_data_config({}, model=model)
        transform = timm.data.create_transform(**cfg_data, is_training=False)

        per_image = {}
        with torch.no_grad():
            for path, cat in test_items:
                img = Image.open(path).convert("RGB")
                x = transform(img).unsqueeze(0).to(DEVICE)
                logits = model(x).squeeze(0).cpu().numpy()
                probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
                order = np.argsort(-logits)
                top1_idx, top2_idx = int(order[0]), int(order[1])
                pred_class = classes[top1_idx]
                per_image[path] = {
                    "gt": cat,
                    "pred": pred_class,
                    "correct": pred_class == cat,
                    "logits": [float(v) for v in logits],
                    "softmax": [float(v) for v in probs],
                    "top1_logit": float(logits[top1_idx]),
                    "top2_logit": float(logits[top2_idx]),
                    "top1_top2_margin": float(logits[top1_idx] - logits[top2_idx]),
                    "top1_softmax": float(probs[top1_idx]),
                    "entropy": softmax_entropy(probs),
                }
        out_dir = BENCHMARK_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{name}_test_logits.json"
        out_path.write_text(json.dumps({"classes": classes, "predictions": per_image}, indent=2), encoding="utf-8")
        n_correct = sum(1 for v in per_image.values() if v["correct"])
        print(f"[{name}] {n_correct}/{len(per_image)} correct. Written: {out_path}\n")

        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
