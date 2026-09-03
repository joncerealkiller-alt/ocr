"""
Baseline stability check for the v4 checkpoint against the 4 newly-
pulled census years (1901, 1906, 1926, 1931) - all genuinely unseen by
any of v1/v2/v3/v4 training. Run BEFORE any templating/labeling/
training happens on this new pool, so there's a real "before" snapshot
to diff against once a v5 checkpoint is trained on them.

Runs table detection ONLY (same as the v3/v4 generalization probes -
table detection doesn't depend on a per-doc-type template, unlike row/
header, which need calibrated regions_approx). Raw pulled images, no
Stage 0/3 preprocessing - PIL opens them directly, same as
layout_bootstrap_v3_generalization_probe.py.

Read-only. Writes only to
data/outputs/layout_bootstrap_train/v4_new_years_baseline/
(summary.csv + per-image overlays under overlays/<year>/).

Usage:
    python -m training.layout_bootstrap_v4_new_years_baseline
"""
from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V4_CHECKPOINT = (
    PROJECT_ROOT / "data" / "outputs" / "layout_bootstrap_train" / "runs"
    / "census_bootstrap_v4" / "weights" / "best.pt"
)
OUT_DIR = PROJECT_ROOT / "data" / "outputs" / "layout_bootstrap_train" / "v4_new_years_baseline"

YEAR_DIRS = {
    "1901": PROJECT_ROOT / "data" / "outputs" / "lac_pull_1901_batch1" / "raw_jpgs",
    "1906": PROJECT_ROOT / "data" / "outputs" / "lac_pull_1906_batch1" / "raw_jpgs",
    "1926": PROJECT_ROOT / "data" / "outputs" / "lac_pull_1926_batch1" / "raw_jpgs",
    "1931": PROJECT_ROOT / "data" / "outputs" / "lac_pull_1931_batch1" / "raw_jpgs",
}


def main() -> None:
    from ultralytics import YOLO

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(V4_CHECKPOINT))
    print(f"Loaded v4 checkpoint: {V4_CHECKPOINT}")

    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except Exception:
        font = ImageFont.load_default()

    rows = []
    for year, img_dir in YEAR_DIRS.items():
        if not img_dir.exists():
            print(f"SKIP year {year}: {img_dir} not found")
            continue
        overlay_dir = OUT_DIR / "overlays" / year
        overlay_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(img_dir.glob("*.jpg"))
        print(f"\n=== {year}: {len(image_paths)} images ===")

        for path in image_paths:
            with Image.open(path) as img:
                img_rgb = img.convert("RGB")
                w, h = img_rgb.size
                orientation = "landscape" if w > h else ("portrait" if h > w else "square")

                preds = model.predict(img_rgb, imgsz=1280, conf=0.1, verbose=False)
                result = preds[0]
                names = result.names
                table_dets = []
                if result.boxes is not None:
                    for box in result.boxes:
                        cid = int(box.cls.item())
                        if names.get(cid) == "table":
                            table_dets.append({
                                "confidence": round(float(box.conf.item()), 4),
                                "bbox": [round(float(v), 1) for v in box.xyxy[0].tolist()],
                            })

                overlay = img_rgb.copy()
                draw = ImageDraw.Draw(overlay)
                for det in table_dets:
                    x0, y0, x1, y1 = det["bbox"]
                    draw.rectangle([x0, y0, x1, y1], outline="#2a9d8f", width=8)
                    label = f"table {det['confidence']:.2f}"
                    draw.rectangle([x0, max(0, y0 - 50), x0 + 24 + len(label) * 22, y0], fill="#2a9d8f")
                    draw.text((x0 + 8, max(0, y0 - 46)), label, fill="black", font=font)
                draw.rectangle([0, 0, 1100, 60], fill="black")
                draw.text((10, 8), f"{path.stem} ({orientation}, {w}x{h})", fill="white", font=font)

                scale = min(1.0, 1400 / w)
                overlay = overlay.resize((int(w * scale), int(h * scale)))
                overlay.save(overlay_dir / f"{path.stem}.png")

            best_conf = max((d["confidence"] for d in table_dets), default=None)
            n_dets = len(table_dets)
            flag = ""
            if n_dets == 0:
                flag = "NO_DETECTION"
            elif n_dets > 1:
                flag = "MULTI_DETECTION"
            elif best_conf is not None and best_conf < 0.3:
                flag = "LOW_CONF"

            print(f"  {path.stem:<20} {orientation:<10} {w}x{h:<8} dets={n_dets} best_conf={best_conf} {flag}")
            rows.append({
                "year": year, "stem": path.stem, "orientation": orientation,
                "width": w, "height": h, "n_table_detections": n_dets,
                "best_confidence": best_conf, "flag": flag,
            })

    csv_path = OUT_DIR / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["year", "stem", "orientation", "width", "height",
                                                 "n_table_detections", "best_confidence", "flag"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'='*60}")
    print("Summary by year:")
    for year in YEAR_DIRS:
        year_rows = [r for r in rows if r["year"] == year]
        if not year_rows:
            continue
        n_any = sum(1 for r in year_rows if r["n_table_detections"] > 0)
        n_flagged = sum(1 for r in year_rows if r["flag"])
        confs = [r["best_confidence"] for r in year_rows if r["best_confidence"] is not None]
        avg_conf = sum(confs) / len(confs) if confs else None
        print(f"  {year}: {n_any}/{len(year_rows)} got >=1 detection, "
              f"{n_flagged} flagged (no/multi/low-conf), avg_best_conf={avg_conf:.3f}" if avg_conf else
              f"  {year}: {n_any}/{len(year_rows)} got >=1 detection, {n_flagged} flagged")

    print(f"\nTotal: {len(rows)} images")
    print(f"Summary CSV: {csv_path}")
    print(f"Overlays: {OUT_DIR / 'overlays'}")


if __name__ == "__main__":
    main()
