"""
Test-compare run for core/layout_detector_v26.py (YOLOv26-small,
DocLayNet-trained) against the SAME sample images already visually
reviewed against core/layout_detector.py's DocLayout-YOLO/DocStructBench
output this session - so the two sensors' output is directly comparable
image-for-image, not just via aggregate stats.

NOT wired into the pipeline. Writes ONLY to
data/outputs/layout_detection_yolo26_test/ (a detections.json keyed by
image path, plus annotated overlay PNGs) - never touches
data/baseline_embeddings.json or pipeline.db, per direct instruction.

Sample set: all 23 images from the Census source folder (the set just
verified image-by-image against the current sensor) plus the 6 extra
spread images (handwritten_ledger x1, map_land_record x2, portrait_photo
x1, genealogy_chart x1, dense_tabular_rows x1 already covered by the
census set) pulled from the earlier bucket-rate investigation.

Usage:
    python -m benchmark.layout_detector_v26_test
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "layout_detection_yolo26_test"
WORKING_DIR = PROJECT_ROOT / "data" / "working"

CLASS_COLORS = {
    "Title": "#e63946", "Text": "#1d3557", "Picture": "#f4a261",
    "Caption": "#e9c46a", "Table": "#2a9d8f", "Section-header": "#8ab17d",
    "Footnote": "#606c38", "Formula": "#9d4edd",
    "List-item": "#c77dff", "Page-footer": "#888888", "Page-header": "#457b9d",
}


def _sample_image_names() -> list[str]:
    prov = json.loads((PROJECT_ROOT / "data" / "manifest_provenance.json").read_text(encoding="utf-8"))
    census_names = [
        Path(r["working_path"]).name for r in prov
        if str(Path(r["source_path"]).parent).endswith("Census")
    ]
    extra_names = [
        "oocihm.lac_reel_t2185.798.jpg",       # handwritten_ledger
        "oocihm.lac_reel_c10414.129.jpg",       # map_land_record
        "Screenshot 2026-05-05 212335.png",     # map_land_record
        "Screenshot 2026-06-10 205357.png",     # portrait_photo
        "Screenshot 2026-05-03 182722.png",     # genealogy_chart
    ]
    names = census_names + extra_names
    # de-dupe, preserve order
    seen = set()
    out = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


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
    from core.layout_detector_v26 import build_layout_model_v26, detect_layout_v26, DEFAULT_IMGSZ, DEFAULT_CONF

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    overlays_dir = OUTPUT_DIR / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    names = _sample_image_names()
    print(f"Testing YOLOv26-small against {len(names)} sample image(s)...")

    model = build_layout_model_v26()
    print(f"Model loaded: {type(model).__name__}, class names: {getattr(model, 'names', None)}")

    results = []
    for i, name in enumerate(names, 1):
        path = WORKING_DIR / name
        if not path.exists():
            print(f"  [{i}/{len(names)}] SKIP (missing file): {name}")
            continue
        with Image.open(path) as img:
            detections = detect_layout_v26(model, img.convert("RGB"), imgsz=DEFAULT_IMGSZ, conf=DEFAULT_CONF)
        results.append({
            "image": str(path),
            "checkpoint": "Armaggheddon/yolo26-document-layout/yolo26s_doc_layout.pt",
            "imgsz": DEFAULT_IMGSZ,
            "conf_threshold": DEFAULT_CONF,
            "detections": detections,
        })
        out_overlay = overlays_dir / f"{path.stem}.png"
        _draw(path, detections, out_overlay)
        print(f"  [{i}/{len(names)}] {name}: {len(detections)} detection(s)")

    detections_path = OUTPUT_DIR / "detections.json"
    with open(detections_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n{len(results)} image(s) processed.")
    print(f"Detections written to {detections_path}")
    print(f"Overlays written to {overlays_dir}")


if __name__ == "__main__":
    main()
