"""
Layout-detector bootstrap training, step 2: converts step 1's sidecars
(real table_bbox/header_bbox/rows[].bbox, in DESKEWED image coordinate
space) into a YOLO-format training dataset.

Class scheme (deliberately small and specific to what auto_sidecar.py
reliably produces, NOT DocLayNet's 11 classes - this is a from-scratch
class scheme for this bootstrap, not a fine-tune of v26's existing
head against its own label space):
    0 = table   (one box per page: sidecar["table_bbox"])
    1 = row     (one box per detected data row: sidecar["rows"][i]["bbox"])
    2 = header  (one box per page, if present: sidecar["header_bbox"])

Writes to data/outputs/layout_bootstrap_train/yolo_dataset/:
    images/train/*.png, images/val/*.png
    labels/train/*.txt, labels/val/*.txt   (YOLO format: class cx cy w h, normalized)
    data.yaml

Only step 1's "OK" images are used (3 of the 16 - e001946615,
e001946619, e001946623 - came back with the row detector tiling a fixed
50-row count regardless of actual content, confirmed wrong against
data/automatedgenealogy_pull.csv's real transcribed row counts and
flagged by the row detector's own "runs past the image's actual bottom
edge" warning; feeding those bad pseudo-labels into training would
teach the model wrong boxes on those specific pages). 13 good images
total - split 10 train / 3 val. Not enough data for a meaningful held-
out metric, but ultralytics' trainer expects a val set to run at all;
treat any val-set number from this bootstrap as indicative only, not a
real generalization estimate.

Usage:
    python -m training.layout_bootstrap_step2_build_yolo_dataset
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BOOTSTRAP_DIR = PROJECT_ROOT / "data" / "outputs" / "layout_bootstrap_train"
YOLO_DIR = BOOTSTRAP_DIR / "yolo_dataset"

CLASS_NAMES = ["table", "row", "header"]

# Deterministic split - last 3 of the 13 GOOD (status == "OK") images
# held out as "val", not random (13 examples is too few for a random
# split to be meaningfully different from a fixed one, and fixed is
# reproducible across reruns).
VAL_STEMS = {"e001961124", "e002101688", "e001946622"}


def _to_yolo_bbox(bbox: list[float], img_w: int, img_h: int) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2 / img_w
    cy = (y0 + y1) / 2 / img_h
    w = (x1 - x0) / img_w
    h = (y1 - y0) / img_h
    return cx, cy, w, h


def main() -> None:
    summary = json.loads((BOOTSTRAP_DIR / "step1_summary.json").read_text(encoding="utf-8"))
    ok_entries = [r for r in summary if r.get("ok") and r.get("status") == "OK"]
    excluded = [r["stem"] for r in summary if r.get("ok") and r.get("status") != "OK"]
    print(f"{len(ok_entries)} sidecars available from step 1 (status == OK)")
    if excluded:
        print(f"Excluded (row-count mismatch, see step 1's CHECK flags): {excluded}")

    for split in ("train", "val"):
        (YOLO_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_train, n_val = 0, 0
    total_boxes = {"table": 0, "row": 0, "header": 0}

    for entry in ok_entries:
        stem = entry["stem"]
        sidecar = json.loads(Path(entry["sidecar_path"]).read_text(encoding="utf-8"))
        deskewed_path = Path(entry["deskewed_path"])
        img_w, img_h = sidecar["deskewed_image_size"]

        split = "val" if stem in VAL_STEMS else "train"
        if split == "train":
            n_train += 1
        else:
            n_val += 1

        lines = []
        if sidecar.get("table_bbox"):
            cx, cy, w, h = _to_yolo_bbox(sidecar["table_bbox"], img_w, img_h)
            lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            total_boxes["table"] += 1
        if sidecar.get("header_bbox"):
            cx, cy, w, h = _to_yolo_bbox(sidecar["header_bbox"], img_w, img_h)
            lines.append(f"2 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            total_boxes["header"] += 1
        for row in sidecar.get("rows", []):
            cx, cy, w, h = _to_yolo_bbox(row["bbox"], img_w, img_h)
            lines.append(f"1 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            total_boxes["row"] += 1

        img_out = YOLO_DIR / "images" / split / f"{stem}.png"
        shutil.copy2(deskewed_path, img_out)
        label_out = YOLO_DIR / "labels" / split / f"{stem}.txt"
        label_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    data_yaml = YOLO_DIR / "data.yaml"
    data_yaml.write_text(
        f"path: {YOLO_DIR.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        + "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASS_NAMES)),
        encoding="utf-8",
    )

    print(f"train images: {n_train}, val images: {n_val}")
    print(f"total boxes written: {total_boxes}")
    print(f"data.yaml written to {data_yaml}")


if __name__ == "__main__":
    main()
