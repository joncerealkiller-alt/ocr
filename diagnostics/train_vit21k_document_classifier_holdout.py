"""
Same fine-tune as diagnostics/train_vit21k_document_classifier.py, but with
a genuine 3-way split (train/val/test) instead of train/val only, per
Jon's request (2026-08-07) for a cleaner number: the first run's 90.4%
used val for BOTH checkpoint selection and the reported accuracy, which
is optimistic (the model's best epoch was picked BECAUSE it did well on
that exact set). Here, val is used only to pick the best checkpoint
during training; the held-out test set is never touched until one final
evaluation pass at the end, on the winning checkpoint only.

Dataset, class weighting, unfreezing strategy, and hyperparameters are
otherwise IDENTICAL to the first run for direct comparability - only the
split changes.

Small-class floor: genealogy_chart (n=6) and mixed_text_image (n=21) etc.
get a minimum of 1 image in val AND 1 in test (not the 15% fraction,
which would round to 0) so every category has at least a nominal held-out
test signal, though results on n=1-2 test categories remain noise, not a
real measurement - flagged explicitly in the report.

Usage:
    python diagnostics/train_vit21k_document_classifier_holdout.py
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
REPORT_PATH = PROJECT_ROOT / "data" / "logs" / "reviewed" / "vit21k_finetune_holdout_report.txt"

RANDOM_SEED = 42
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
VAL_MIN_PER_CLASS = 1
TEST_MIN_PER_CLASS = 1
BATCH_SIZE = 16
NUM_EPOCHS = 8
LR_HEAD = 1e-3
LR_BACKBONE = 1e-5
UNFREEZE_LAST_N_BLOCKS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_dataset() -> dict[str, list[str]]:
    by_cat: dict[str, list[str]] = defaultdict(list)
    with open(DATASET_CSV, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            by_cat[r["category"]].append(r["file_path"])
    return by_cat


def stratified_3way_split(by_cat: dict[str, list[str]], rng: random.Random):
    train, val, test = [], [], []
    for cat, paths in by_cat.items():
        shuffled = paths[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_val = max(VAL_MIN_PER_CLASS, int(n * VAL_FRACTION))
        n_test = max(TEST_MIN_PER_CLASS, int(n * TEST_FRACTION))
        # cap so tiny classes don't get emptied out of train entirely
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
        if self.train and random.random() < 0.1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        x = self.transform(img)
        y = self.class_to_idx[cat]
        return x, y


def build_model(num_classes: int):
    model = timm.create_model(TIMM_TAG, pretrained=True, num_classes=num_classes)
    cfg = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**cfg, is_training=False)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.get_classifier().parameters():
        p.requires_grad = True
    for blk in model.blocks[-UNFREEZE_LAST_N_BLOCKS:]:
        for p in blk.parameters():
            p.requires_grad = True
    for p in model.norm.parameters():
        p.requires_grad = True
    return model, transform


def evaluate(model, loader, classes) -> tuple[float, dict, list]:
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


def main():
    lines = []
    def log(s=""):
        lines.append(s)
        print(s)

    log(f"Device: {DEVICE}")
    by_cat = load_dataset()
    classes = sorted(by_cat.keys())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    log(f"Classes ({len(classes)}): {classes}")

    rng = random.Random(RANDOM_SEED)
    train_items, val_items, test_items = stratified_3way_split(by_cat, rng)
    log(f"\nTrain: {len(train_items)}  Val: {len(val_items)}  Test (held out): {len(test_items)}")
    for c in classes:
        tr = sum(1 for _, cc in train_items if cc == c)
        va = sum(1 for _, cc in val_items if cc == c)
        te = sum(1 for _, cc in test_items if cc == c)
        log(f"  {c:<20s} train={tr:<5d} val={va:<4d} test={te:<4d}")

    log(f"\nBuilding {TIMM_TAG}, unfreezing last {UNFREEZE_LAST_N_BLOCKS} blocks + norm + head...")
    model, transform = build_model(len(classes))
    model.to(DEVICE)

    train_ds = DocImageDataset(train_items, class_to_idx, transform, train=True)
    val_ds = DocImageDataset(val_items, class_to_idx, transform, train=False)
    test_ds = DocImageDataset(test_items, class_to_idx, transform, train=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    counts = np.array([len(by_cat[c]) for c in classes], dtype=np.float32)
    weights = (1.0 / counts)
    weights = weights / weights.sum() * len(classes)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

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
    best_state = None

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
        log(f"\nEpoch {epoch}/{NUM_EPOCHS}  train_loss={train_loss:.4f}  "
            f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")
        for c in classes:
            corr, tot = val_per_class[c]
            if tot:
                log(f"    val/{c:<20s} {corr}/{tot} ({100*corr/tot:.1f}%)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            log(f"    -> new best (by val), epoch {epoch}")

    log(f"\n=== Training done. Best val_acc={best_val_acc:.3f} at epoch {best_epoch} ===")
    log("Loading best-by-val checkpoint and running the ONE-TIME held-out test evaluation...")
    model.load_state_dict(best_state)

    test_acc, test_per_class, pairs = evaluate(model, test_loader, classes)
    log(f"\n=== HELD-OUT TEST ACCURACY (never touched during training/selection): "
        f"{test_acc:.3f} ({sum(c for c,t in test_per_class.values())}/{sum(t for c,t in test_per_class.values())}) ===\n")
    log("Per-category held-out test accuracy:")
    for c in classes:
        corr, tot = test_per_class[c]
        note = "  (n<=2, noise not a real measurement)" if tot <= 2 else ""
        if tot:
            log(f"  {c:<20s} {corr}/{tot} ({100*corr/tot:.1f}%){note}")
        else:
            log(f"  {c:<20s} no test examples")

    log("\nConfusion matrix (rows=ground truth, cols=predicted) on held-out test set:")
    header = "gt\\pred".ljust(22) + "".join(c[:10].ljust(12) for c in classes)
    log(header)
    conf = defaultdict(Counter)
    for t, p in pairs:
        conf[classes[t]][classes[p]] += 1
    for gt_c in classes:
        row = gt_c.ljust(22) + "".join(str(conf[gt_c].get(c, 0)).ljust(12) for c in classes)
        log(row)

    ckpt_path = CHECKPOINT_DIR / "best_holdout_eval.pt"
    torch.save({
        "model_state_dict": best_state,
        "classes": classes,
        "epoch": best_epoch,
        "val_acc": best_val_acc,
        "test_acc": test_acc,
    }, ckpt_path)
    log(f"\nCheckpoint (selected by val, evaluated on held-out test): {ckpt_path}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(f"Report written: {REPORT_PATH}")


if __name__ == "__main__":
    main()
