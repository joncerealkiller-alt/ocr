"""
Layout-detector bootstrap training, step 3: fine-tunes YOLOv26-small
(core/layout_detector_v26.py's checkpoint - Armaggheddon/yolo26-
document-layout, yolo26s_doc_layout.pt) on the 13-image, 3-class
(table/row/header) dataset step 2 built from real auto_sidecar.py
output, cross-validated against automatedgenealogy_pull.csv transcribed
row counts.

This REPLACES the pretrained detection head (13 images can't support
DocLayNet's original 11-class head at all) while keeping the pretrained
backbone/neck as initialization - standard ultralytics transfer-
learning behavior when data.yaml's class count differs from the
checkpoint's. Explicitly a BOOTSTRAP ("does learning help at all" - not
a production model): 13 images / 676 boxes is far below normal object-
detection training scale, so overfitting is expected and the result
should be read as a directional signal (does the model start producing
tighter, more consistent table/row/header boxes on THIS corpus's
material than the untouched pretrained weights), not a generalizable
model, until a much larger labeled set exists.

NOT wired into the pipeline. Writes only to
data/outputs/layout_bootstrap_train/runs/ - never touches
core/layout_detector.py, core/layout_detector_v26.py, or
data/baseline_embeddings.json.

Usage:
    python -m training.layout_bootstrap_step3_train
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BOOTSTRAP_DIR = PROJECT_ROOT / "data" / "outputs" / "layout_bootstrap_train"
DATA_YAML = BOOTSTRAP_DIR / "yolo_dataset" / "data.yaml"
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
        name="census_bootstrap",
        exist_ok=True,
        patience=30,  # early stop if val loss stalls - guards against wasted epochs on 13 images
        verbose=True,
        plots=True,
    )
    print("\nTraining complete.")
    print(f"Best weights: {RUNS_DIR / 'census_bootstrap' / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
