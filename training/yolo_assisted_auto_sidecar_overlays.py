"""
Visual overlays for the YOLO-assisted auto_sidecar comparison - 4 panels
per priority case (original image, baseline CV table boundary + rows,
raw YOLO table proposal, assisted refined boundary + rows), per spec.

Reads the already-saved comparison output (data/outputs/
yolo_assisted_auto_sidecar/{baseline,assisted}/sidecars/) and
regenerates the YOLO raw-candidate box directly (cheap, not saved to
disk by the comparison driver) for the specific priority stems.

Usage:
    python -m training.yolo_assisted_auto_sidecar_overlays
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "yolo_assisted_auto_sidecar"
OVERLAYS_DIR = OUTPUT_DIR / "overlays"

WORKING_DIR = PROJECT_ROOT / "data" / "working"
LAC_IMAGES_DIR = PROJECT_ROOT / "data" / "outputs" / "lac_pull_batch1" / "images"

PRIORITY_CASES = {
    "rescued_from_quarantine": "e001946629",
    "rescued_partial_to_full": "e001946631",
    "clear_improvement_exact_fix": "e001946619",
    "regression_broke_perfect_page": "e001946614",
    "regression_broke_perfect_page_2": "e002101688",
    "hint_rejected_fallback_worked": "e001946664",
    "large_row_drop_clean_page": "e001946688",
}


def _resolve_path(stem: str) -> Path:
    p = WORKING_DIR / f"{stem}.png"
    return p if p.exists() else LAC_IMAGES_DIR / f"{stem}.png"


def _draw_overlay(image: Image.Image, table_bbox, rows, quarantined_rows, label: str,
                   extra_box=None, extra_label=None, extra_color="#9d4edd") -> Image.Image:
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 48)
    except Exception:
        font = ImageFont.load_default()

    if table_bbox:
        draw.rectangle(list(table_bbox), outline="#2a9d8f", width=8)
        draw.text((table_bbox[0] + 8, table_bbox[1] + 8), "TABLE", fill="#2a9d8f", font=font)
    for r in (rows or []):
        bbox = r["bbox"]
        draw.rectangle(list(bbox), outline="red", width=2)
    for r in (quarantined_rows or []):
        bbox = r.get("bbox")
        if bbox and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
            draw.rectangle(list(bbox), outline="orange", width=3)
    if extra_box:
        draw.rectangle(list(extra_box), outline=extra_color, width=8)
        if extra_label:
            draw.text((extra_box[0] + 8, max(0, extra_box[1] - 60)), extra_label, fill=extra_color, font=font)

    draw.rectangle([0, 0, 900, 70], fill="black")
    draw.text((10, 10), label, fill="white", font=font)
    return img


def main() -> None:
    from training.yolo_assisted_auto_sidecar import (
        build_yolo_table_model, detect_table_candidates, select_table_candidate,
        validate_and_pad_table_hint,
    )
    from core.auto_sidecar import estimate_deskew_angle, apply_deskew_angle
    from core.image_analysis import DESKEW_ANGLE_RANGE

    OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)
    yolo_model = build_yolo_table_model()

    for case_name, stem in PRIORITY_CASES.items():
        image_path = _resolve_path(stem)
        baseline_sidecar_path = OUTPUT_DIR / "baseline" / "sidecars" / f"{stem}_sidecar.json"
        assisted_sidecar_path = OUTPUT_DIR / "assisted" / "sidecars" / f"{stem}_sidecar.json"

        if not baseline_sidecar_path.exists() and not assisted_sidecar_path.exists():
            print(f"SKIP {case_name} ({stem}): no saved sidecars (likely whole-page quarantined in baseline)")

        original = Image.open(str(image_path)).convert("RGB")
        angle = estimate_deskew_angle(original, angle_range=DESKEW_ANGLE_RANGE)
        deskewed = apply_deskew_angle(original, angle)

        candidates = detect_table_candidates(yolo_model, deskewed)
        selected = select_table_candidate(candidates, deskewed.size)
        yolo_raw_box = selected["bbox_xyxy"] if selected else None
        yolo_conf = selected["confidence"] if selected else None

        panels = []

        panels.append(_draw_overlay(deskewed, None, None, None, f"{case_name}\n{stem} - ORIGINAL (deskewed)"))

        if baseline_sidecar_path.exists():
            b = json.loads(baseline_sidecar_path.read_text(encoding="utf-8"))
            panels.append(_draw_overlay(
                deskewed, b["table_bbox"], b["rows"], b.get("rows_needs_review"),
                f"BASELINE (CV-only): {len(b['rows'])} rows"))
        else:
            panels.append(_draw_overlay(deskewed, None, None, None, "BASELINE: WHOLE-PAGE QUARANTINED (no table_bbox)"))

        yolo_label = f"YOLO proposal (conf={yolo_conf:.2f})" if yolo_raw_box else "YOLO: no credible detection"
        panels.append(_draw_overlay(
            deskewed, None, None, None, yolo_label,
            extra_box=yolo_raw_box, extra_label="YOLO raw box"))

        if assisted_sidecar_path.exists():
            a = json.loads(assisted_sidecar_path.read_text(encoding="utf-8"))
            panels.append(_draw_overlay(
                deskewed, a["table_bbox"], a["rows"], a.get("rows_needs_review"),
                f"ASSISTED (YOLO-hinted CV): {len(a['rows'])} rows"))
        else:
            panels.append(_draw_overlay(deskewed, None, None, None, "ASSISTED: WHOLE-PAGE QUARANTINED"))

        # Stack panels vertically, downscaled for a manageable file size
        scale = 900 / deskewed.width
        thumb_size = (int(deskewed.width * scale), int(deskewed.height * scale))
        thumbs = [p.resize(thumb_size) for p in panels]
        combined = Image.new("RGB", (thumb_size[0], thumb_size[1] * len(thumbs)), "white")
        for i, t in enumerate(thumbs):
            combined.paste(t, (0, i * thumb_size[1]))

        out_path = OVERLAYS_DIR / f"{case_name}_{stem}.png"
        combined.save(out_path)
        print(f"{case_name} ({stem}): yolo_box={yolo_raw_box} conf={yolo_conf} -> {out_path}")


if __name__ == "__main__":
    main()
