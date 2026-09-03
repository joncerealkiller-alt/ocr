"""
Fine-tunes google/vit-base-patch16-224 (timm tag vit_base_patch16_224.
orig_in21k) as a full 8-way document-type classifier, per Jon's direction
(2026-08-07): earlier zero-shot nearest-cluster tests of this same
checkpoint (diagnostics/test_vit_base21k_classification.py) and
MobileNetV2 both plateaued at ~54% - this tests whether actually fine-
tuning the head (+ optionally unfreezing top blocks) on real project data
does meaningfully better than frozen-embedding cosine-similarity voting.

Dataset: data/outputs/vit_finetune_dataset.csv (built by
diagnostics/build_vit_finetune_dataset.py) - manifest_final.csv's
existing 8-category ground truth, with dense_tabular_rows supplemented
by 484 real LAC census page images Jon hand-reviewed to exclude blanks/
title cards. IAM line crops explicitly excluded (crop-scale mismatch -
see conversation).

Stratified train/val split per category (small categories like
genealogy_chart, n=6, get a minimum floor rather than a percentage, so
val isn't empty for them).

Usage:
    python diagnostics/train_vit21k_document_classifier.py
"""

from __future__ import annotations

import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader

DATASET_CSV = PROJECT_ROOT / "data" / "outputs" / "vit_finetune_dataset.csv"
TIMM_TAG = "vit_base_patch16_224.orig_in21k"
CHECKPOINT_DIR = PROJECT_ROOT / "data" / "outputs" / "vit21k_doc_classifier_checkpoints"
REPORT_PATH = PROJECT_ROOT / "data" / "logs" / "reviewed" / "vit21k_finetune_report.txt"

RANDOM_SEED = 42
VAL_FRACTION = 0.15
VAL_MIN_PER_CLASS = 2  # floor so tiny classes (genealogy_chart, n=6) still get a val split
BATCH_SIZE = 16
NUM_EPOCHS = 8
LR_HEAD = 1e-3
LR_BACKBONE = 1e-5  # unfreeze top blocks at a much lower LR than the fresh head
UNFREEZE_LAST_N_BLOCKS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_dataset() -> dict[str, list[str]]:
    by_cat: dict[str, list[str]] = defaultdict(list)
    with open(DATASET_CSV, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            by_cat[r["category"]].append(r["file_path"])
    return by_cat


def stratified_split(by_cat: dict[str, list[str]], rng: random.Random):
    train, val = [], []
    for cat, paths in by_cat.items():
        shuffled = paths[:]
        rng.shuffle(shuffled)
        n_val = max(VAL_MIN_PER_CLASS, int(len(shuffled) * VAL_FRACTION))
        n_val = min(n_val, len(shuffled) - 1) if len(shuffled) > 1 else 0
        val.extend((p, cat) for p in shuffled[:n_val])
        train.extend((p, cat) for p in shuffled[n_val:])
    return train, val


class DocImageDataset(Dataset):
    def __init__(self, items: list[tuple[str, str]], class_to_idx: dict[str, int], transform, train: bool):
        self.items = items
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, cat = self.items[idx]
        img = Image.open(path).convert("RGB")
        if self.train:
            # light augmentation - these are scanned/screenshot documents, not natural
            # photos, so keep it conservative (no heavy color jitter / rotation)
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT) if random.random() < 0.1 else img
        x = self.transform(img)
        y = self.class_to_idx[cat]
        return x, y


def build_model(num_classes: int):
    model = timm.create_model(TIMM_TAG, pretrained=True, num_classes=num_classes)
    cfg = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**cfg, is_training=False)

    # freeze everything, then unfreeze the head + last N transformer blocks
    for p in model.parameters():
        p.requires_grad = False
    for p in model.get_classifier().parameters():
        p.requires_grad = True
    blocks = model.blocks
    for blk in blocks[-UNFREEZE_LAST_N_BLOCKS:]:
        for p in blk.parameters():
            p.requires_grad = True
    for p in model.norm.parameters():
        p.requires_grad = True

    return model, transform


def main():
    lines = []
    def log(s=""):
        lines.append(s)
        print(s)

    log(f"Device: {DEVICE}")
    log(f"Loading dataset manifest: {DATASET_CSV}")
    by_cat = load_dataset()
    classes = sorted(by_cat.keys())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    log(f"Classes ({len(classes)}): {classes}")
    for c in classes:
        log(f"  {c:<20s} {len(by_cat[c])}")

    rng = random.Random(RANDOM_SEED)
    train_items, val_items = stratified_split(by_cat, rng)
    log(f"\nTrain: {len(train_items)}  Val: {len(val_items)}")
    val_counts = Counter(c for _, c in val_items)
    for c in classes:
        log(f"  val/{c:<20s} {val_counts.get(c, 0)}")

    log(f"\nBuilding {TIMM_TAG} with {len(classes)}-way head, "
        f"unfreezing last {UNFREEZE_LAST_N_BLOCKS} blocks + norm + head...")
    model, transform = build_model(len(classes))
    model.to(DEVICE)

    train_ds = DocImageDataset(train_items, class_to_idx, transform, train=True)
    val_ds = DocImageDataset(val_items, class_to_idx, transform, train=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # inverse-frequency class weights - real imbalance here (genealogy_chart n=6
    # vs printed_document n~1169), plain CE would just ignore the tiny classes
    counts = np.array([len(by_cat[c]) for c in classes], dtype=np.float32)
    weights = (1.0 / counts)
    weights = weights / weights.sum() * len(classes)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
    log(f"\nClass weights (inverse-frequency): "
        f"{dict(zip(classes, [round(w,2) for w in weights]))}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    head_params = list(model.get_classifier().parameters())
    backbone_params = [p for n, p in model.named_parameters()
                        if p.requires_grad and not any(p is hp for hp in head_params)]
    optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": LR_HEAD},
        {"params": backbone_params, "lr": LR_BACKBONE},
    ])

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_acc = -1.0
    best_epoch = -1

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

        model.eval()
        val_correct, val_seen = 0, 0
        val_per_class_correct = Counter()
        val_per_class_total = Counter()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == y).sum().item()
                val_seen += x.size(0)
                for p, t in zip(preds.cpu().tolist(), y.cpu().tolist()):
                    val_per_class_total[t] += 1
                    if p == t:
                        val_per_class_correct[t] += 1
        val_acc = val_correct / val_seen if val_seen else 0.0

        log(f"\nEpoch {epoch}/{NUM_EPOCHS}  train_loss={train_loss:.4f}  "
            f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f} ({val_correct}/{val_seen})")
        for i, c in enumerate(classes):
            tot = val_per_class_total.get(i, 0)
            corr = val_per_class_correct.get(i, 0)
            if tot:
                log(f"    val/{c:<20s} {corr}/{tot} ({100*corr/tot:.1f}%)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            ckpt_path = CHECKPOINT_DIR / "best.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "epoch": epoch,
                "val_acc": val_acc,
            }, ckpt_path)
            log(f"    -> new best, saved {ckpt_path}")

    log(f"\n=== DONE. Best val_acc={best_val_acc:.3f} at epoch {best_epoch} ===")
    log(f"Checkpoint: {CHECKPOINT_DIR / 'best.pt'}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(f"Report written: {REPORT_PATH}")


if __name__ == "__main__":
    main()
