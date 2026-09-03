"""
Layout-detector bootstrap training, step 4: visual before/after
comparison - vanilla pretrained YOLOv26-small vs. the census-bootstrap
fine-tuned weights (data/outputs/layout_bootstrap_train/runs/
census_bootstrap/weights/best.pt, from epoch 7) - on the same 3 held-out
val images plus 2 train images, side by side.

NOT wired into the pipeline. Writes only to
data/outputs/layout_bootstrap_train/comparison/.

Usage:
    python -m training.layout_bootstrap_step4_compare
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BOOTSTRAP_DIR = PROJECT_ROOT / "data" / "outputs" / "layout_bootstrap_train"
FINE_TUNED_WEIGHTS = BOOTSTRAP_DIR / "runs" / "census_bootstrap" / "weights" / "best.pt"
COMPARISON_DIR = BOOTSTRAP_DIR / "comparison"

BOOTSTRAP_CLASS_COLORS = {"table": "#2a9d8f", "row": "#e63946", "header": "#f4a261"}
V26_CLASS_COLORS = {
    "Title": "#e63946", "Text": "#1d3557", "Picture": "#f4a261",
    "Caption": "#e9c46a", "Table": "#2a9d8f", "Section-header": "#8ab17d",
    "Footnote": "#606c38", "Formula": "#9d4edd",
    "List-item": "#c77dff", "Page-footer": "#888888", "Page-header": "#457b9d",
}

SAMPLE_STEMS = ["e001961124", "e002101688", "e001946622", "e001926997", "e001946014"]


def _draw(image_path: Path, detections: list[dict], colors: dict, out_path: Path) -> None:
    with Image.open(image_path) as img:
        img = img.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except Exception:
        font = ImageFont.load_default()
    for det in detections:
        x0, y0, x1, y1 = det["bbox_xyxy"]
        color = colors.get(det["class_name"], "#ffffff")
        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        draw.rectangle([x0, max(0, y0 - 44), x0 + 20 + len(label) * 20, y0], fill=color)
        draw.text((x0 + 6, max(0, y0 - 42)), label, fill="black", font=font)
    img.save(out_path)


def main() -> None:
    from ultralytics import YOLO
    from core.layout_detector_v26 import build_layout_model_v26, detect_layout_v26, DEFAULT_IMGSZ, DEFAULT_CONF

    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading vanilla pretrained v26-small...")
    vanilla_model = build_layout_model_v26()

    print(f"Loading fine-tuned bootstrap weights from {FINE_TUNED_WEIGHTS}...")
    finetuned_model = YOLO(str(FINE_TUNED_WEIGHTS))

    deskewed_dir = BOOTSTRAP_DIR / "deskewed_images"

    for stem in SAMPLE_STEMS:
        image_path = deskewed_dir / f"{stem}.png"
        if not image_path.exists():
            print(f"SKIP (no deskewed copy): {stem}")
            continue

        with Image.open(image_path) as img:
            img_rgb = img.convert("RGB")
            vanilla_dets = detect_layout_v26(vanilla_model, img_rgb, imgsz=DEFAULT_IMGSZ, conf=DEFAULT_CONF)

        results = finetuned_model.predict(str(image_path), imgsz=1280, conf=0.2, verbose=False)
        result = results[0]
        names = result.names
        finetuned_dets = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                finetuned_dets.append({
                    "class_id": class_id,
                    "class_name": names.get(class_id, str(class_id)),
                    "confidence": round(float(box.conf.item()), 4),
                    "bbox_xyxy": [round(float(v), 1) for v in box.xyxy[0].tolist()],
                })

        _draw(image_path, vanilla_dets, V26_CLASS_COLORS, COMPARISON_DIR / f"{stem}_1_vanilla_v26small.png")
        _draw(image_path, finetuned_dets, BOOTSTRAP_CLASS_COLORS, COMPARISON_DIR / f"{stem}_2_finetuned_bootstrap.png")

        print(f"{stem}: vanilla={len(vanilla_dets)} detection(s), finetuned={len(finetuned_dets)} detection(s)")

    print(f"\nComparison images written to {COMPARISON_DIR}")


if __name__ == "__main__":
    main()
