"""
Quick cross-year/cross-form/cross-orientation generalization sanity
check for the v3 checkpoint, per direct instruction, BEFORE scaling
training to ~400 pages: tests whether v3's table-detection improvement
represents a genuinely reusable visual feature or just memorization of
the Manitoba-1911-reel material it was trained on.

Runs v3's "table" detection ONLY (not row/header - the instruction was
specifically "how well the table detection alone transfers") against
every image in the Census source folder NOT used in any of v1/v2/v3
training (21 of 23 - the other 2 are duplicate copies of an already-
trained image, e001946617). Covers 1921 Canada, 1931 Canada, England
1891/1901/1911/1939 records (genuinely different country and form),
and two genuinely portrait-oriented pages - real diversity, not
hand-picked to flatter the result.

Read-only, writes overlays only to
data/outputs/layout_bootstrap_train/v3_generalization_probe/.

Usage:
    python -m training.layout_bootstrap_v3_generalization_probe
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKING_DIR = PROJECT_ROOT / "data" / "working"
V3_CHECKPOINT = (
    PROJECT_ROOT / "data" / "outputs" / "layout_bootstrap_train" / "runs"
    / "census_bootstrap_v3" / "weights" / "best.pt"
)
OUT_DIR = PROJECT_ROOT / "data" / "outputs" / "layout_bootstrap_train" / "v3_generalization_probe"

# Every Census-folder image NOT used in v1/v2/v3 training (excludes the
# 2 duplicate copies of e001946617, which WAS trained on).
PROBE_STEMS = [
    "14765_0303", "14765_0304", "14765_0535",
    "1921_013-E002869156", "1921_022-E002880409", "1921_158-E003219273",
    "1931_174-e011707164",
    "30953_148096-00296",
    "31228_4363955-00089", "31228_4363956-00124",
    "DEVRG12_1767_1769-0573", "DEVRG13_2122_2125-0400", "DEVRG13_2139_2141-0371",
    "Screenshot 2026-05-08 081916",
    "WARRG13_2924_2926-0161", "WARRG13_2948_2949-0078",
    "rg14_04575_0097_03", "rg14_18900_0013_03", "rg14_18900_0087_03",
    "tna_r39_5711_5711d_015",
    "z000017634",
]


def _find_path(stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png"):
        p = WORKING_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def main() -> None:
    from ultralytics import YOLO

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(V3_CHECKPOINT))
    print(f"Loaded v3 checkpoint: {V3_CHECKPOINT}")

    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except Exception:
        font = ImageFont.load_default()

    results_summary = []
    for stem in PROBE_STEMS:
        path = _find_path(stem)
        if path is None:
            print(f"SKIP (not found): {stem}")
            continue

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
            draw.text((10, 8), f"{stem} ({orientation}, {w}x{h})", fill="white", font=font)

            scale = min(1.0, 1400 / w)
            overlay = overlay.resize((int(w * scale), int(h * scale)))
            out_path = OUT_DIR / f"{stem}.png"
            overlay.save(out_path)

        best_conf = max((d["confidence"] for d in table_dets), default=None)
        print(f"{stem:<35} {orientation:<10} {w}x{h:<8} table_dets={len(table_dets)} best_conf={best_conf}")
        results_summary.append({
            "stem": stem, "orientation": orientation, "size": [w, h],
            "n_table_detections": len(table_dets), "best_confidence": best_conf,
        })

    n_any_detection = sum(1 for r in results_summary if r["n_table_detections"] > 0)
    print(f"\n{n_any_detection}/{len(results_summary)} images got at least one table detection")
    print(f"Overlays written to {OUT_DIR}")


if __name__ == "__main__":
    main()
