"""
Layout-detector bootstrap training, v2: combines the original 13-image
bootstrap set (training/layout_bootstrap_step1_generate_sidecars.py)
with the 90-image LAC sequential pull
(training/lac_batch1_convert_and_sidecar.py) into one 103-image YOLO
dataset - same table/row/header class scheme as v1, built specifically
to test whether more data fixes v1's near-total "row" detection
failure (mAP50 0.0001 on 13 images/650 row instances) before investing
further effort pulling even more.

Deterministic 85/18 train/val split (roughly matching v1's ~77/23
proportion) - last ~18 images by sorted stem across the COMBINED pool
held out, not random, for reproducibility.

Usage:
    python -m training.layout_bootstrap_v2_combined_dataset
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V1_DIR = PROJECT_ROOT / "data" / "outputs" / "layout_bootstrap_train"
LAC_DIR = PROJECT_ROOT / "data" / "outputs" / "lac_pull_batch1"
YOLO_DIR = V1_DIR / "yolo_dataset_v2_combined"

CLASS_NAMES = ["table", "row", "header"]
N_VAL = 18


def _to_yolo_bbox(bbox: list[float], img_w: int, img_h: int) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2 / img_w
    cy = (y0 + y1) / 2 / img_h
    w = (x1 - x0) / img_w
    h = (y1 - y0) / img_h
    return cx, cy, w, h


def _write_example(stem: str, sidecar_path: Path, deskewed_path: Path, split: str) -> int:
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    img_w, img_h = sidecar["deskewed_image_size"]

    lines = []
    if sidecar.get("table_bbox"):
        cx, cy, w, h = _to_yolo_bbox(sidecar["table_bbox"], img_w, img_h)
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    if sidecar.get("header_bbox"):
        cx, cy, w, h = _to_yolo_bbox(sidecar["header_bbox"], img_w, img_h)
        lines.append(f"2 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    for row in sidecar.get("rows", []):
        cx, cy, w, h = _to_yolo_bbox(row["bbox"], img_w, img_h)
        lines.append(f"1 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    img_out = YOLO_DIR / "images" / split / f"{stem}.png"
    shutil.copy2(deskewed_path, img_out)
    label_out = YOLO_DIR / "labels" / split / f"{stem}.txt"
    label_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def main() -> None:
    v1_summary = json.loads((V1_DIR / "step1_summary.json").read_text(encoding="utf-8"))
    lac_summary = json.loads((LAC_DIR / "step2_summary.json").read_text(encoding="utf-8"))

    v1_good = [r for r in v1_summary if r.get("ok") and r.get("status") == "OK"]
    lac_good = [r for r in lac_summary if r.get("ok") and r.get("status") == "OK"]
    print(f"v1 bootstrap good images: {len(v1_good)}")
    print(f"LAC pull batch 1 good images: {len(lac_good)}")

    combined = sorted(v1_good + lac_good, key=lambda r: r["stem"])
    print(f"Combined pool: {len(combined)}")

    val_set = set(r["stem"] for r in combined[-N_VAL:])

    for split in ("train", "val"):
        (YOLO_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_train, n_val, total_boxes = 0, 0, 0
    for r in combined:
        stem = r["stem"]
        split = "val" if stem in val_set else "train"
        if split == "train":
            n_train += 1
        else:
            n_val += 1
        n_boxes = _write_example(stem, Path(r["sidecar_path"]), Path(r["deskewed_path"]), split)
        total_boxes += n_boxes

    data_yaml = YOLO_DIR / "data.yaml"
    data_yaml.write_text(
        f"path: {YOLO_DIR.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        + "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASS_NAMES)),
        encoding="utf-8",
    )

    print(f"\ntrain images: {n_train}, val images: {n_val}")
    print(f"total boxes: {total_boxes}")
    print(f"data.yaml written to {data_yaml}")


if __name__ == "__main__":
    main()
