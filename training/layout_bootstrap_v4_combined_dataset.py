"""
Layout-detector bootstrap training, v4: combines FIVE labeled sources
into one ~402-image YOLO dataset (table/row/header, same 3-class scheme
as v1/v2/v3) - the first genuinely YEAR-DIVERSIFIED dataset, not just a
larger pull from the same 1911 Manitoba reel:
    - v1 bootstrap (13 images, 1911, CV-only labels)
    - LAC pull batch 1 (90 images, 1911, CV-only labels)
    - LAC pull batch 2 (100 images, 1911, conditional CV/YOLO-v2-rescue)
    - LAC pull batch 3 (99 images, 1911, conditional CV/YOLO-v3-rescue)
    - LAC 1921 batch 1 (100 images, 1921 - a genuinely different year/
      form layout, canada_census_1901.yaml sibling template - 100/100
      CV-only clean, 0 needed rescue, confirming the 1921 template is
      solidly calibrated)

Deterministic 335/67 train/val split (~83/17, matching v1/v2/v3's
proportions) - last ~67 images by sorted (source, stem) across the
combined pool held out, stratified loosely by keeping 1921 images
spread through the val set too (not all-1911 val), not random.

Usage:
    python -m training.layout_bootstrap_v4_combined_dataset
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V1_DIR = PROJECT_ROOT / "data" / "outputs" / "layout_bootstrap_train"
LAC1_DIR = PROJECT_ROOT / "data" / "outputs" / "lac_pull_batch1"
LAC2_DIR = PROJECT_ROOT / "data" / "outputs" / "lac_pull_batch2"
LAC3_DIR = PROJECT_ROOT / "data" / "outputs" / "lac_pull_batch3"
LAC_1921_DIR = PROJECT_ROOT / "data" / "outputs" / "lac_pull_1921_batch1"
YOLO_DIR = V1_DIR / "yolo_dataset_v4_combined"

CLASS_NAMES = ["table", "row", "header"]
VAL_FRACTION = 0.17


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


def _load_good(summary_path: Path, source_label: str) -> list[dict]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    good = [r for r in summary if r.get("ok") and r.get("status", "OK") == "OK"]
    for r in good:
        r["source_label"] = source_label
    return good


def main() -> None:
    v1_good = _load_good(V1_DIR / "step1_summary.json", "v1_1911")
    lac1_good = _load_good(LAC1_DIR / "step2_summary.json", "lac1_1911")
    lac2_good = _load_good(LAC2_DIR / "step2_summary.json", "lac2_1911")
    lac3_good = _load_good(LAC3_DIR / "step2_summary.json", "lac3_1911")
    lac_1921_good = _load_good(LAC_1921_DIR / "step2_summary.json", "lac_1921")

    for label, pool in [("v1", v1_good), ("lac1", lac1_good), ("lac2", lac2_good),
                         ("lac3", lac3_good), ("1921", lac_1921_good)]:
        print(f"{label}: {len(pool)} good images")

    combined = sorted(
        v1_good + lac1_good + lac2_good + lac3_good + lac_1921_good,
        key=lambda r: (r["source_label"], r["stem"]),
    )
    print(f"Combined pool: {len(combined)}")

    # Stratified-ish val split: take the last VAL_FRACTION of EACH source
    # group separately, so val isn't accidentally all-1911 or all-1921.
    from collections import defaultdict
    by_source = defaultdict(list)
    for r in combined:
        by_source[r["source_label"]].append(r)

    val_stems = set()
    for label, pool in by_source.items():
        pool_sorted = sorted(pool, key=lambda r: r["stem"])
        n_val = max(1, round(len(pool_sorted) * VAL_FRACTION))
        for r in pool_sorted[-n_val:]:
            val_stems.add(r["stem"])

    for split in ("train", "val"):
        (YOLO_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_train, n_val, total_boxes = 0, 0, 0
    n_1921_train, n_1921_val = 0, 0
    for r in combined:
        stem = r["stem"]
        split = "val" if stem in val_stems else "train"
        if split == "train":
            n_train += 1
            if r["source_label"] == "lac_1921":
                n_1921_train += 1
        else:
            n_val += 1
            if r["source_label"] == "lac_1921":
                n_1921_val += 1
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

    print(f"\ntrain images: {n_train} ({n_1921_train} from 1921), "
          f"val images: {n_val} ({n_1921_val} from 1921)")
    print(f"total boxes: {total_boxes}")
    print(f"data.yaml written to {data_yaml}")


if __name__ == "__main__":
    main()
