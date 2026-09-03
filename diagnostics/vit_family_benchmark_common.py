"""
Shared machinery for the multi-architecture sensor-qualification benchmark
requested by Jon (2026-08-07), following up on the ViT-21k fine-tune
result (docs/MULTI_SOURCE_VOTING_CLASSIFIER_PROPOSAL.md's "Fine-tuned
ViT-21k..." section + ablation): does domain-adaptation fine-tuning help
OTHER architecture families as much as it helped ViT-21k, and do
different families retain DIFFERENT error patterns after fine-tuning
(the thing that would make them complementary fusion voters, not
redundant ones)?

Everything here is held IDENTICAL across every architecture tested, per
Jon's explicit "controlled benchmark, not ad hoc" requirement:
  - dataset: data/outputs/vit_finetune_dataset.csv (same file used for
    the successful ViT-21k run - manifest_final.csv + 484 hand-reviewed
    LAC census pages added to dense_tabular_rows)
  - 3-way stratified split, same seed (42), same fractions/floors as
    diagnostics/train_vit21k_document_classifier_holdout.py
  - class-weighted cross-entropy (inverse frequency)
  - same augmentation (occasional horizontal flip only)
  - same epoch count, batch size, optimizer family (AdamW, head LR 1e-3 /
    backbone LR 1e-5)
  - same held-out test set, evaluated exactly once per model on the
    checkpoint selected by val accuracy (never touched during training)

The one thing that legitimately differs per architecture is WHICH layers
get unfrozen, because the families have structurally different top-level
modules (verified via a real inspection pass before writing this, not
guessed):
  - isotropic ViT-style (dinov2, beit, siglip, the original vit21k run):
    `.blocks` is a flat list - unfreeze the last N blocks + `.norm` (+
    `.fc_norm` if present) + head.
  - Swin: `.layers` is a list of hierarchical STAGES (each containing its
    own internal blocks) - unfreezing "the last 2 blocks" doesn't apply
    the same way; unfreeze the last stage (`.layers[-1]`) as the
    structurally equivalent "final adaptable chunk" + `.norm` + head.
  - ConvNeXt: `.stages` is a list of CNN stages - unfreeze the last stage
    + `.norm_pre` + head.
  - MobileNetV2: `.blocks` is a list of CNN stages (different semantics
    from the ViT `.blocks` name despite the same attribute name) -
    unfreeze the last stage + `.conv_head`/`.bn2` + classifier.
"""

from __future__ import annotations

import csv
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_CSV = PROJECT_ROOT / "data" / "outputs" / "vit_finetune_dataset.csv"
BENCHMARK_ROOT = PROJECT_ROOT / "data" / "outputs" / "vit_family_benchmark"
REPORTS_DIR = PROJECT_ROOT / "data" / "logs" / "reviewed"

RANDOM_SEED = 42
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
VAL_MIN_PER_CLASS = 1
TEST_MIN_PER_CLASS = 1
BATCH_SIZE = 16
NUM_EPOCHS = 8
LR_HEAD = 1e-3
LR_BACKBONE = 1e-5
UNFREEZE_LAST_N_BLOCKS = 2  # isotropic ViT-style families only
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# reference-set size for the frozen zero-shot baseline (built from TRAIN
# split only, same partition every model uses for fine-tuning, so the
# zero-shot baseline is apples-to-apples with the fine-tuned result, not
# a different data slice like the earlier standalone MobileNetV2/ViT
# zero-shot tests used)
ZEROSHOT_REFERENCE_PER_CLASS = 15


def load_dataset() -> dict[str, list[str]]:
    by_cat: dict[str, list[str]] = defaultdict(list)
    with open(DATASET_CSV, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            by_cat[r["category"]].append(r["file_path"])
    return by_cat


def stratified_3way_split(by_cat: dict[str, list[str]], rng: random.Random):
    """IDENTICAL logic to train_vit21k_document_classifier_holdout.py's
    stratified_3way_split - copied, not re-derived, so every architecture
    gets the exact same partition given the same seed."""
    train, val, test = [], [], []
    for cat, paths in by_cat.items():
        shuffled = paths[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_val = max(VAL_MIN_PER_CLASS, int(n * VAL_FRACTION))
        n_test = max(TEST_MIN_PER_CLASS, int(n * TEST_FRACTION))
        while n_val + n_test >= n and (n_val > VAL_MIN_PER_CLASS or n_test > TEST_MIN_PER_CLASS):
            if n_val > VAL_MIN_PER_CLASS:
                n_val -= 1
            elif n_test > TEST_MIN_PER_CLASS:
                n_test -= 1
        n_val = min(n_val, max(0, n - 1))
        n_test = min(n_test, max(0, n - n_val - 1)) if n - n_val > 1 else 0
        test.extend((p, cat) for p in shuffled[:n_test])
        val.extend((p, cat) for p in shuffled[n_test:n_test + n_val])
        train.extend((p, cat) for p in shuffled[n_test + n_val:])
    return train, val, test


def get_split(seed: int = RANDOM_SEED):
    """The one true split every model in this benchmark must use."""
    by_cat = load_dataset()
    rng = random.Random(seed)
    return by_cat, *stratified_3way_split(by_cat, rng)


class DocImageDataset(Dataset):
    def __init__(self, items, class_to_idx, transform, train: bool):
        self.items = items
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, cat = self.items[idx]
        img = Image.open(path).convert("RGB")
        if self.train and random.random() < 0.1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        x = self.transform(img)
        y = self.class_to_idx[cat]
        return x, y


def configure_finetune_layers(model, family: str):
    """Unfreezes the structurally-equivalent 'final adaptable chunk' for
    each architecture family. Returns (head_params, backbone_params) for
    the two-LR optimizer setup."""
    for p in model.parameters():
        p.requires_grad = False

    head_params = list(model.get_classifier().parameters())
    for p in head_params:
        p.requires_grad = True

    backbone_params = []

    if family in ("dinov2", "beit", "siglip", "vit21k"):
        for blk in model.blocks[-UNFREEZE_LAST_N_BLOCKS:]:
            for p in blk.parameters():
                p.requires_grad = True
                backbone_params.append(p)
        for p in model.norm.parameters():
            p.requires_grad = True
            backbone_params.append(p)
        if hasattr(model, "fc_norm") and not isinstance(model.fc_norm, nn.Identity):
            for p in model.fc_norm.parameters():
                p.requires_grad = True
                backbone_params.append(p)

    elif family == "swin":
        for p in model.layers[-1].parameters():
            p.requires_grad = True
            backbone_params.append(p)
        for p in model.norm.parameters():
            p.requires_grad = True
            backbone_params.append(p)

    elif family == "convnext":
        for p in model.stages[-1].parameters():
            p.requires_grad = True
            backbone_params.append(p)
        for p in model.norm_pre.parameters():
            p.requires_grad = True
            backbone_params.append(p)

    elif family == "mobilenetv2":
        for p in model.blocks[-1].parameters():
            p.requires_grad = True
            backbone_params.append(p)
        for p in model.conv_head.parameters():
            p.requires_grad = True
            backbone_params.append(p)
        for p in model.bn2.parameters():
            p.requires_grad = True
            backbone_params.append(p)

    else:
        raise ValueError(f"Unknown family: {family}")

    return head_params, backbone_params


def build_model(timm_tag: str, family: str, num_classes: int):
    model = timm.create_model(timm_tag, pretrained=True, num_classes=num_classes)
    cfg = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**cfg, is_training=False)
    head_params, backbone_params = configure_finetune_layers(model, family)
    return model, transform, head_params, backbone_params


def evaluate(model, loader, classes):
    model.eval()
    correct, seen = 0, 0
    per_class_correct, per_class_total = Counter(), Counter()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(dim=-1)
            correct += (preds == y).sum().item()
            seen += x.size(0)
            for p, t in zip(preds.cpu().tolist(), y.cpu().tolist()):
                per_class_total[t] += 1
                if p == t:
                    per_class_correct[t] += 1
                all_preds.append(p)
                all_targets.append(t)
    acc = correct / seen if seen else 0.0
    per_class = {classes[i]: (per_class_correct.get(i, 0), per_class_total.get(i, 0)) for i in range(len(classes))}
    return acc, per_class, list(zip(all_targets, all_preds))


def evaluate_with_paths(model, items, class_to_idx, transform, classes, batch_size=16):
    """Like evaluate() but also returns {file_path: {"gt":cat,"pred":cat}}
    for cross-architecture agreement analysis - needs file_path identity,
    not just index position, since different models' DataLoaders may not
    preserve identical iteration order under all conditions."""
    ds = DocImageDataset(items, class_to_idx, transform, train=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    preds_by_path = {}
    idx = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            preds = model(x).argmax(dim=-1).cpu().tolist()
            for p in preds:
                path, cat = items[idx]
                preds_by_path[path] = {"gt": cat, "pred": classes[p]}
                idx += 1
    return preds_by_path


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def measure_inference_speed_and_memory(model, transform, sample_paths: list[str], n_warmup=3, n_measure=20):
    """Approximate single-image inference latency (images/sec) and peak
    GPU memory during inference. Uses real project images (not synthetic
    tensors) at batch=1, matching how a fusion-engine sensor would
    actually be called per-page."""
    model.eval()
    paths = sample_paths[: n_warmup + n_measure]
    if len(paths) < n_warmup + 1:
        paths = (paths * ((n_warmup + n_measure) // max(1, len(paths)) + 1))[: n_warmup + n_measure]

    tensors = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        tensors.append(transform(img).unsqueeze(0))

    if DEVICE == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for t in tensors[:n_warmup]:
            model(t.to(DEVICE))
        if DEVICE == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for t in tensors[n_warmup:n_warmup + n_measure]:
            model(t.to(DEVICE))
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    images_per_sec = n_measure / elapsed if elapsed > 0 else float("inf")
    peak_mem_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if DEVICE == "cuda" else 0.0
    return images_per_sec, peak_mem_mb


def zeroshot_baseline_eval(timm_tag: str, train_items, test_items, classes, rng_seed=RANDOM_SEED):
    """Frozen embedding, nearest-cluster-via-cosine baseline, using the
    SAME train/test partition every fine-tuned model in this benchmark
    uses (reference embeddings drawn from train only, evaluated on the
    same held-out test set) - apples-to-apples with the fine-tuned
    result, unlike the earlier standalone zero-shot tests which used a
    different manifest/split."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from core.vision_embeddings import build_model_and_transform, embed_pooled, predict_nearest_bucket

    model, transform = build_model_and_transform(timm_tag)

    by_cat_train = defaultdict(list)
    for p, c in train_items:
        by_cat_train[c].append(p)
    rng = random.Random(rng_seed)
    reference_embeddings = {}
    for c in classes:
        paths = by_cat_train.get(c, [])
        shuffled = paths[:]
        rng.shuffle(shuffled)
        ref_paths = shuffled[:ZEROSHOT_REFERENCE_PER_CLASS]
        vecs = []
        for p in ref_paths:
            try:
                img = Image.open(p).convert("RGB")
                vec = embed_pooled(model, transform, img)
                vecs.append((p, vec))
            except Exception:
                continue
        reference_embeddings[c] = vecs

    results = []
    preds_by_path = {}
    for p, c in test_items:
        try:
            img = Image.open(p).convert("RGB")
            vec = embed_pooled(model, transform, img)
        except Exception:
            continue
        predicted, score = predict_nearest_bucket(vec, reference_embeddings)
        results.append({"gt": c, "predicted": predicted, "correct": predicted == c})
        preds_by_path[p] = {"gt": c, "pred": predicted}

    n = len(results)
    correct_n = sum(1 for r in results if r["correct"])
    overall_acc = correct_n / n if n else 0.0
    per_class = defaultdict(lambda: [0, 0])
    for r in results:
        per_class[r["gt"]][1] += 1
        if r["correct"]:
            per_class[r["gt"]][0] += 1
    per_class = {c: tuple(v) for c, v in per_class.items()}

    return overall_acc, per_class, preds_by_path
