"""
Standalone visual/quantitative comparison: vanilla pretrained
YOLOv26-small (core/layout_detector_v26.py, DocLayNet's 11-class scheme)
vs. the census-bootstrap fine-tuned checkpoint (data/outputs/
layout_bootstrap_train's census_bootstrap_v4/weights/best.pt, project-
specific 3-class table/row/header scheme, per docs/LAYOUT_DETECTOR_
BOOTSTRAP_TRAINING.md's v4 section).

NOT wired into core/routing_decision_engine.py or anything else - per
Jon's explicit "use it outside the engine for now" (2026-08-09). Pure
detection comparison: what does each model actually draw on real
images, on BOTH the census/dense_tabular_rows material the fine-tuned
checkpoint was trained on AND handwritten_ledger images it has never
seen, to check whether its learned "table"/"row" concepts transfer to a
different, unrelated document type - the real open question this run
is meant to answer, not assumed either way.

Writes side-by-side annotated images (same _draw() pattern as
training/layout_bootstrap_step4_compare.py) plus a plain-text detection
summary (counts/classes/confidence per image per model) to
data/outputs/layout_v26_vs_census_checkpoint_comparison/.

Usage:
    python diagnostics/compare_v26_baseline_vs_census_checkpoint.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image, ImageDraw, ImageFont

from core.layout_detector_v26 import build_layout_model_v26, detect_layout_v26, DEFAULT_IMGSZ, DEFAULT_CONF
from diagnostics.vit_family_benchmark_common import get_split
from diagnostics.error_analysis.common import OLD_WORKING_PREFIX, LEGACY_WORKING_IMAGES_DIR

CENSUS_CHECKPOINT = (
    PROJECT_ROOT.parent / "genealogy_workspace" / "datasets" / "bootstrap" / "layout_bootstrap_train"
    / "runs" / "census_bootstrap_v4" / "weights" / "best.pt"
)
# 2026-08-13 (RunContext/output-location audit): this script was
# written AFTER the data/outputs -> genealogy_workspace migration
# (2026-08-08) but still wrote into the legacy data/outputs location -
# the only post-migration stray writer found. Its existing output
# (20 files, 226M) was moved to the workspace research area (same-
# volume rename, counts verified) and this constant repointed; per the
# migration's own ownership table, a diagnostic comparison study is
# research/experiments territory.
OUT_DIR = (
    PROJECT_ROOT.parent / "genealogy_workspace" / "research" / "experiments"
    / "layout_v26_vs_census_checkpoint_comparison"
)

V26_CLASS_COLORS = {
    "Title": "#e63946", "Text": "#1d3557", "Picture": "#f4a261",
    "Caption": "#e9c46a", "Table": "#2a9d8f", "Section-header": "#8ab17d",
    "Footnote": "#606c38", "Formula": "#9d4edd",
    "List-item": "#c77dff", "Page-footer": "#888888", "Page-header": "#457b9d",
}
CENSUS_CLASS_COLORS = {"table": "#2a9d8f", "row": "#e63946", "header": "#f4a261"}

N_PER_CATEGORY = 5
CATEGORIES_TO_SAMPLE = ["handwritten_ledger", "dense_tabular_rows"]


def resolve_path(path_str: str) -> str:
    if Path(path_str).exists():
        return path_str
    if path_str.startswith(OLD_WORKING_PREFIX):
        filename = path_str[len(OLD_WORKING_PREFIX):].lstrip("\\/")
        remapped = str(LEGACY_WORKING_IMAGES_DIR / filename)
        if Path(remapped).exists():
            return remapped
    return path_str


def draw_detections(image_path: str, detections: list[dict], colors: dict, out_path: Path) -> None:
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


def run_census_checkpoint(model, image_path: str) -> list[dict]:
    results = model.predict(image_path, imgsz=1280, conf=0.2, verbose=False)
    result = results[0]
    names = result.names
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            detections.append({
                "class_id": class_id,
                "class_name": names.get(class_id, str(class_id)),
                "confidence": round(float(box.conf.item()), 4),
                "bbox_xyxy": [round(float(v), 1) for v in box.xyxy[0].tolist()],
            })
    return detections


def main():
    lines = []
    def log(s=""):
        lines.append(s)
        print(s, flush=True)

    if not CENSUS_CHECKPOINT.exists():
        raise FileNotFoundError(f"{CENSUS_CHECKPOINT} not found.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log("Loading vanilla pretrained YOLOv26-small (DocLayNet, 11-class)...")
    vanilla_model = build_layout_model_v26()
    log(f"Loading census-bootstrap fine-tuned checkpoint (table/row/header) from {CENSUS_CHECKPOINT}...")
    from ultralytics import YOLO
    census_model = YOLO(str(CENSUS_CHECKPOINT))
    log("Both models loaded.\n")

    by_cat, train_items, val_items, test_items = get_split()

    for category in CATEGORIES_TO_SAMPLE:
        paths = by_cat.get(category, [])[:N_PER_CATEGORY]
        log(f"\n{'='*80}\n{category} ({len(paths)} sample images)\n{'='*80}")

        for path in paths:
            resolved = resolve_path(path)
            if not Path(resolved).exists():
                log(f"  SKIP (not found): {path}")
                continue
            stem = Path(resolved).stem

            with Image.open(resolved) as img:
                img_rgb = img.convert("RGB")
                vanilla_dets = detect_layout_v26(vanilla_model, img_rgb, imgsz=DEFAULT_IMGSZ, conf=DEFAULT_CONF)

            census_dets = run_census_checkpoint(census_model, resolved)

            draw_detections(resolved, vanilla_dets, V26_CLASS_COLORS,
                             OUT_DIR / f"{category}_{stem}_1_vanilla_v26.png")
            draw_detections(resolved, census_dets, CENSUS_CLASS_COLORS,
                             OUT_DIR / f"{category}_{stem}_2_census_checkpoint.png")

            vanilla_summary = ", ".join(f"{d['class_name']}({d['confidence']:.2f})" for d in vanilla_dets) or "none"
            census_summary = ", ".join(f"{d['class_name']}({d['confidence']:.2f})" for d in census_dets) or "none"
            log(f"\n  {stem}")
            log(f"    vanilla v26  ({len(vanilla_dets)}): {vanilla_summary}")
            log(f"    census ckpt  ({len(census_dets)}): {census_summary}")

    report_path = PROJECT_ROOT / "data" / "logs" / "reviewed" / "layout_v26_vs_census_checkpoint_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"\n\nAnnotated images written to {OUT_DIR}/")
    log(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
