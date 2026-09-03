"""
Runs the controlled multi-architecture sensor-qualification benchmark
Jon requested (2026-08-07): one representative model per architecture
family, fine-tuned with an identical protocol (see
vit_family_benchmark_common.py's module docstring for exactly what's
held fixed vs. what legitimately varies per family), each producing
identical reporting - zero-shot baseline, fine-tuned held-out accuracy,
per-class accuracy, confusion matrix, params, inference speed, GPU
memory, training time - plus per-file held-out predictions saved for the
cross-architecture agreement analysis (diagnostics/analyze_vit_family_
benchmark.py, run separately after this).

Representative models (real timm tags, matching the project's own
already-qualified QUALIFIED_ENCODERS in core/vision_embeddings.py where
possible, so results are directly comparable to that prior work):
  - convnext     convnext_tiny.fb_in22k               (modern CNN)
  - mobilenetv2  mobilenetv2_100.ra_in1k               (lightweight CNN)
  - dinov2       vit_small_patch14_dinov2.lvd142m      (self-supervised transformer)
  - beit         beit_base_patch16_224.in22k_ft_in22k  (self-supervised transformer, masked-image-modeling family)
  - swin         swin_base_patch4_window7_224.ms_in22k (hierarchical transformer)
  - siglip       vit_base_patch16_siglip_224.v2_webli  (vision-language encoder)

vit21k (vit_base_patch16_224.orig_in21k) is NOT re-run here - its full
result (89.8% held-out, ablation-confirmed) already exists from the
immediately preceding work and is folded into the final comparison table
by the analysis script instead of wastefully re-training it.

Checkpoints/reports go to per-architecture subfolders under
data/outputs/vit_family_benchmark/<arch>/ - never touches
data/outputs/vit21k_doc_classifier_checkpoints/best.pt or
best_holdout_eval.pt.

Usage:
    python diagnostics/run_vit_family_benchmark.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from diagnostics.vit_family_benchmark_common import (
    BATCH_SIZE, BENCHMARK_ROOT, DEVICE, LR_BACKBONE, LR_HEAD, NUM_EPOCHS,
    REPORTS_DIR, DocImageDataset, build_model, count_params, evaluate,
    evaluate_with_paths, get_split, measure_inference_speed_and_memory,
    zeroshot_baseline_eval,
)
import numpy as np

MODEL_CONFIGS = [
    {"name": "convnext", "timm_tag": "convnext_tiny.fb_in22k", "family": "convnext"},
    {"name": "mobilenetv2", "timm_tag": "mobilenetv2_100.ra_in1k", "family": "mobilenetv2"},
    {"name": "dinov2", "timm_tag": "vit_small_patch14_dinov2.lvd142m", "family": "dinov2"},
    {"name": "beit", "timm_tag": "beit_base_patch16_224.in22k_ft_in22k", "family": "beit"},
    {"name": "swin", "timm_tag": "swin_base_patch4_window7_224.ms_in22k", "family": "swin"},
    {"name": "siglip", "timm_tag": "vit_base_patch16_siglip_224.v2_webli", "family": "siglip"},
]


def run_one(config: dict, by_cat, train_items, val_items, test_items, classes, class_to_idx):
    name = config["name"]
    timm_tag = config["timm_tag"]
    family = config["family"]

    out_dir = BENCHMARK_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"vit_family_benchmark_{name}_report.txt"

    lines = []
    def log(s=""):
        lines.append(s)
        print(s, flush=True)

    log(f"\n{'='*70}\n{name}  ({timm_tag}, family={family})\n{'='*70}")

    # -- zero-shot baseline (same train/test partition as fine-tuning) --
    log("\n-- Zero-shot frozen-embedding baseline --")
    zs_acc, zs_per_class, zs_preds = zeroshot_baseline_eval(timm_tag, train_items, test_items, classes)
    log(f"Zero-shot held-out accuracy: {zs_acc:.3f}")
    for c in classes:
        corr, tot = zs_per_class.get(c, (0, 0))
        if tot:
            log(f"  {c:<20s} {corr}/{tot} ({100*corr/tot:.1f}%)")

    # -- fine-tune --
    log(f"\n-- Fine-tuning {timm_tag} --")
    model, transform, head_params, backbone_params = build_model(timm_tag, family, len(classes))
    model.to(DEVICE)
    n_params_total = count_params(model)
    n_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Total params: {n_params_total/1e6:.1f}M  Trainable (unfrozen): {n_params_trainable/1e6:.1f}M")

    train_ds = DocImageDataset(train_items, class_to_idx, transform, train=True)
    val_ds = DocImageDataset(val_items, class_to_idx, transform, train=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    counts = np.array([len(by_cat[c]) for c in classes], dtype=np.float32)
    weights = (1.0 / counts)
    weights = weights / weights.sum() * len(classes)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": LR_HEAD},
        {"params": backbone_params, "lr": LR_BACKBONE},
    ])

    best_val_acc = -1.0
    best_epoch = -1
    best_state = None
    training_curve = []

    train_start = time.perf_counter()
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss, n_seen, n_correct = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            n_seen += x.size(0)
            n_correct += (logits.argmax(dim=-1) == y).sum().item()
        train_loss = total_loss / n_seen
        train_acc = n_correct / n_seen

        val_acc, val_per_class, _ = evaluate(model, val_loader, classes)
        log(f"Epoch {epoch}/{NUM_EPOCHS}  train_loss={train_loss:.4f}  "
            f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")
        training_curve.append({"epoch": epoch, "train_loss": train_loss,
                                "train_acc": train_acc, "val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train_elapsed = time.perf_counter() - train_start

    log(f"\nBest val_acc={best_val_acc:.3f} at epoch {best_epoch}. Training time: {train_elapsed:.1f}s")
    model.load_state_dict(best_state)

    # -- held-out test (once, on the winning checkpoint) --
    test_acc, test_per_class, pairs = evaluate(model, DataLoader(
        DocImageDataset(test_items, class_to_idx, transform, train=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0), classes)
    log(f"\n=== HELD-OUT TEST ACCURACY: {test_acc:.3f} ===")
    for c in classes:
        corr, tot = test_per_class[c]
        if tot:
            log(f"  {c:<20s} {corr}/{tot} ({100*corr/tot:.1f}%)")

    log("\nConfusion matrix (rows=gt, cols=pred):")
    from collections import Counter, defaultdict as dd
    conf = dd(Counter)
    for t, p in pairs:
        conf[classes[t]][classes[p]] += 1
    header = "gt\\pred".ljust(22) + "".join(c[:10].ljust(12) for c in classes)
    log(header)
    for gt_c in classes:
        row = gt_c.ljust(22) + "".join(str(conf[gt_c].get(c, 0)).ljust(12) for c in classes)
        log(row)

    # per-file predictions, for cross-architecture agreement analysis
    preds_by_path = evaluate_with_paths(model, test_items, class_to_idx, transform, classes)

    # -- inference speed + memory --
    log("\n-- Inference speed / memory profiling (batch=1, real project images) --")
    sample_paths = [p for p, _ in test_items[:30]]
    images_per_sec, peak_mem_mb = measure_inference_speed_and_memory(model, transform, sample_paths)
    log(f"Inference speed: {images_per_sec:.1f} images/sec (batch=1, {DEVICE})")
    log(f"Peak GPU memory during inference: {peak_mem_mb:.0f} MB")

    # -- save checkpoint (architecture-specific folder/name, never touches vit21k's) --
    ckpt_path = out_dir / f"{name}_best_holdout_eval.pt"
    torch.save({
        "model_state_dict": best_state, "classes": classes, "timm_tag": timm_tag,
        "family": family, "epoch": best_epoch, "val_acc": best_val_acc, "test_acc": test_acc,
    }, ckpt_path)
    log(f"\nCheckpoint: {ckpt_path}")

    result = {
        "name": name, "timm_tag": timm_tag, "family": family,
        "zeroshot_acc": zs_acc, "zeroshot_per_class": {c: list(v) for c, v in zs_per_class.items()},
        "finetuned_val_acc": best_val_acc, "finetuned_test_acc": test_acc,
        "finetuned_per_class": {c: list(v) for c, v in test_per_class.items()},
        "training_curve": training_curve, "best_epoch": best_epoch,
        "params_total_M": n_params_total / 1e6, "params_trainable_M": n_params_trainable / 1e6,
        "training_time_sec": train_elapsed, "inference_images_per_sec": images_per_sec,
        "peak_inference_mem_MB": peak_mem_mb, "checkpoint_path": str(ckpt_path),
    }

    (out_dir / f"{name}_test_predictions.json").write_text(
        json.dumps(preds_by_path, indent=2), encoding="utf-8")
    (out_dir / f"{name}_result_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"Report: {report_path}")

    if DEVICE == "cuda":
        del model
        torch.cuda.empty_cache()

    return result


def main():
    print(f"Device: {DEVICE}")
    by_cat, train_items, val_items, test_items = get_split()
    classes = sorted(by_cat.keys())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"Classes: {classes}")
    print(f"Train: {len(train_items)}  Val: {len(val_items)}  Test: {len(test_items)}")

    BENCHMARK_ROOT.mkdir(parents=True, exist_ok=True)
    all_results = []
    for config in MODEL_CONFIGS:
        result = run_one(config, by_cat, train_items, val_items, test_items, classes, class_to_idx)
        all_results.append(result)
        # checkpoint the accumulated results after each model in case a later one fails
        (BENCHMARK_ROOT / "all_results_summary.json").write_text(
            json.dumps(all_results, indent=2), encoding="utf-8")

    print("\n\n=== BENCHMARK COMPLETE ===")
    for r in all_results:
        print(f"  {r['name']:<14s} zeroshot={r['zeroshot_acc']:.3f}  "
              f"finetuned={r['finetuned_test_acc']:.3f}  "
              f"params={r['params_total_M']:.1f}M  "
              f"speed={r['inference_images_per_sec']:.1f}img/s")
    print(f"\nAll results: {BENCHMARK_ROOT / 'all_results_summary.json'}")


if __name__ == "__main__":
    main()
