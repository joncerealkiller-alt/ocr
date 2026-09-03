"""
Layout-detector bootstrap training, v2: fine-tunes YOLOv26-small from
the SAME fresh pretrained checkpoint used for v1 (not continuing from
v1's own fine-tuned weights) - a clean, comparable experiment testing
whether 103 images / 5330 boxes (vs. v1's 13 images / 676 boxes) fixes
the near-total "row" class failure. Same hyperparameters as v1
(epochs=80, imgsz=1280, batch=2, patience=30) so any metric difference
is attributable to dataset size, not a training-config change.

NOT wired into the pipeline. Writes only to
data/outputs/layout_bootstrap_train/runs/census_bootstrap_v2/.

Usage:
    python -m training.layout_bootstrap_v2_train
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BOOTSTRAP_DIR = PROJECT_ROOT / "data" / "outputs" / "layout_bootstrap_train"
DATA_YAML = BOOTSTRAP_DIR / "yolo_dataset_v2_combined" / "data.yaml"
RUNS_DIR = BOOTSTRAP_DIR / "runs"


def main() -> None:
    from core.layout_detector_v26 import YOLO26_REPO_ID, YOLO26_SMALL_FILENAME
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO

    weights_path = hf_hub_download(repo_id=YOLO26_REPO_ID, filename=YOLO26_SMALL_FILENAME)
    print(f"Starting from pretrained checkpoint: {weights_path}")

    model = YOLO(weights_path)

    results = model.train(
        data=str(DATA_YAML),
        epochs=80,
        imgsz=1280,
        batch=2,
        project=str(RUNS_DIR),
        name="census_bootstrap_v2",
        exist_ok=True,
        patience=30,
        verbose=True,
        plots=True,
    )
    print("\nTraining complete.")
    print(f"Best weights: {RUNS_DIR / 'census_bootstrap_v2' / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
