"""
Draws detected layout boxes on top of real images, for visual review of
the 2026-08-05 full-corpus layout-detection backfill - picks a mix of
images with a spurious (>=70% area) box and images with only normal-
sized boxes, so both failure and success cases are visible side by side.

Read-only against data/baseline_embeddings.json and the real working
images; writes annotated copies to a scratch output dir only.

Usage:
    python -m benchmark.layout_detection_visual_check
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = PROJECT_ROOT / "data" / "baseline_embeddings.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "layout_detection_visual_check"

SPURIOUS_AREA_FRAC_THRESHOLD = 0.70

CLASS_COLORS = {
    "title": "#e63946",
    "plain text": "#1d3557",
    "figure": "#f4a261",
    "figure_caption": "#e9c46a",
    "table": "#2a9d8f",
    "table_caption": "#8ab17d",
    "table_footnote": "#606c38",
    "isolate_formula": "#9d4edd",
    "formula_caption": "#c77dff",
    "abandon": "#888888",
}


def _draw(image_path: Path, detections: list[dict], out_path: Path) -> None:
    with Image.open(image_path) as img:
        img = img.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        x0, y0, x1, y1 = det["bbox_xyxy"]
        color = CLASS_COLORS.get(det["class_name"], "#ffffff")
        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        draw.rectangle([x0, max(0, y0 - 44), x0 + 20 + len(label) * 20, y0], fill=color)
        draw.text((x0 + 6, max(0, y0 - 42)), label, fill="black", font=font)

    img.save(out_path)


def main() -> None:
    records = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    spurious_examples = []
    clean_examples = []
    for rec in records:
        ld = rec.get("layout_detections", {}).get("doclayout_yolo")
        if not ld or not ld.get("detections"):
            continue
        image_path = Path(rec["image"])
        if not image_path.exists():
            continue
        with Image.open(image_path) as img:
            w, h = img.size
        img_area = w * h
        detections = ld["detections"]
        max_frac = max(
            (max(0.0, d["bbox_xyxy"][2] - d["bbox_xyxy"][0]) * max(0.0, d["bbox_xyxy"][3] - d["bbox_xyxy"][1])) / img_area
            for d in detections
        )
        if max_frac >= SPURIOUS_AREA_FRAC_THRESHOLD and len(spurious_examples) < 4:
            spurious_examples.append((image_path, detections))
        elif max_frac < 0.5 and len(detections) >= 2 and len(clean_examples) < 4:
            clean_examples.append((image_path, detections))
        if len(spurious_examples) >= 4 and len(clean_examples) >= 4:
            break

    print(f"Found {len(spurious_examples)} spurious examples, {len(clean_examples)} clean examples")

    written = []
    for i, (image_path, detections) in enumerate(spurious_examples, 1):
        out_path = OUTPUT_DIR / f"spurious_{i}_{image_path.stem}.png"
        _draw(image_path, detections, out_path)
        written.append(out_path)
        print(f"  spurious: {out_path}")

    for i, (image_path, detections) in enumerate(clean_examples, 1):
        out_path = OUTPUT_DIR / f"clean_{i}_{image_path.stem}.png"
        _draw(image_path, detections, out_path)
        written.append(out_path)
        print(f"  clean: {out_path}")

    print(f"\n{len(written)} annotated images written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
