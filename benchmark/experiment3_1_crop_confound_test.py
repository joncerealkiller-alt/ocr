"""
Experiment 3.1: does removing the pipeline's own detected page
boundary shift ConvNeXt's coarse activation away from border
artifacts (reel leader / archive label) and toward document content?

Redesigned after Experiment 3's pilot run identified a real confound
(uncropped frames' activation tracked the bright reel-leader tag, not
document structure - see docs/VISION_IR_RESEARCH.md). This is NOT a
test of whether ConvNeXt is "good" - it isolates whether border
artifacts specifically are the causal factor, per Jon's H0/H1 framing:

  H0: after removing non-document border artifacts, activation remains
      unrelated to document structure.
  H1: after removing non-document border artifacts, activation shifts
      toward meaningful document regions.

Crop source: core/image_analysis.py's own page_boundary detector - a
real, already-built pipeline component, not a manual crop (per Jon's
requirement #1). No dedicated dewarp sidecar existed for these test
images, so this is the actual pipeline output available for them.

Test set (requirements #5/#6):
  - two intentionally difficult images with reel leader / archive
    label / heavy microfilm surround (oocihm.lac_reel_c10264.767,
    oocihm.lac_reel_c10295.1199 - the same two from the Experiment 3
    pilot, for direct before/after comparability)
  - one clean page where page_boundary already covers ~99.8% of the
    frame (oocihm.lac_reel_c10301.529 - a near-no-op crop, included as
    an internal consistency check: if before/after differ wildly here
    despite almost nothing changing, that's a flag on the method
    itself, not a real finding)

Usage:
    python -m benchmark.experiment3_1_crop_confound_test
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import timm
from PIL import Image

from core.image_analysis import analyze_image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_LOG = PROJECT_ROOT / "data" / "outputs" / "experiment3_1_crop_confound_log.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "experiment3_1_outputs"
CANDIDATE = "convnext_tiny.fb_in22k"

TEST_IMAGES = {
    "difficult_A_c10264.767": ("data/working/oocihm.lac_reel_c10264.767.jpg", "reel_leader+archive_label+microfilm_surround"),
    "difficult_B_c10295.1199": ("data/working/oocihm.lac_reel_c10295.1199.jpg", "reel_leader+archive_label+microfilm_surround"),
    "clean_C_c10301.529": ("data/working/oocihm.lac_reel_c10301.529.jpg", "none_expected (page_confidence=0.993, area_frac=0.998)"),
}


def get_page_bbox(image_path: str):
    """The pipeline's own crop output - core/image_analysis.py's page_boundary quad, reduced to an axis-aligned bbox."""
    a = analyze_image(image_path)
    pb = a.page_boundary
    if not pb:
        return None, a.page_confidence
    xs = [pt[0] for pt in pb]
    ys = [pt[1] for pt in pb]
    return (min(xs), min(ys), max(xs), max(ys)), a.page_confidence


def compute_activation(model, transform, pil_image: Image.Image):
    """Same call, same params, every time - crop state is the only variable across before/after."""
    x = transform(pil_image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        feats = model(x)  # features_only=True -> list of 4 stages
    coarse = feats[-1][0]  # (C, h, w), coarsest stage
    heat = coarse.mean(dim=0).numpy()
    return heat  # raw, not yet normalized - keep raw for area/centroid math


def hotspot_metrics(heat: np.ndarray, doc_center_norm: tuple[float, float], doc_bbox_norm=None):
    """
    doc_center_norm: (x, y) in [0,1] normalized coords of the TRUE
    document center, in the coordinate system of whatever image
    produced `heat` (full frame for "before", crop for "after" - for
    "after" this is trivially (0.5, 0.5) since the crop IS the
    document by construction).
    doc_bbox_norm: (x0,y0,x1,y1) normalized document bbox in the same
    coordinate system, or None if the whole frame is the document
    (the "after" condition, or a page_confidence-driven fallback).
    """
    h, w = heat.shape
    max_val = heat.max()
    argmax_idx = np.unravel_index(np.argmax(heat), heat.shape)
    centroid_norm = ((argmax_idx[1] + 0.5) / w, (argmax_idx[0] + 0.5) / h)

    threshold = max_val * 0.8
    hotspot_area_frac = float((heat >= threshold).sum() / heat.size)

    dist_from_doc_center = float(np.hypot(
        centroid_norm[0] - doc_center_norm[0], centroid_norm[1] - doc_center_norm[1]
    ))

    inside_doc = True
    if doc_bbox_norm is not None:
        x0, y0, x1, y1 = doc_bbox_norm
        inside_doc = bool((x0 <= centroid_norm[0] <= x1) and (y0 <= centroid_norm[1] <= y1))

    return {
        "hotspot_centroid_norm": (float(centroid_norm[0]), float(centroid_norm[1])),
        "hotspot_area_frac": hotspot_area_frac,
        "distance_from_document_center": dist_from_doc_center,
        "centroid_inside_document_bbox": inside_doc,
    }


def make_panel(orig: Image.Image, heat: np.ndarray, display_w=700) -> Image.Image:
    heat_norm = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    heat_img = Image.fromarray((heat_norm * 255).astype(np.uint8)).resize(orig.size, Image.NEAREST)
    scale = display_w / orig.width
    display_size = (display_w, int(orig.height * scale))
    orig_small = orig.resize(display_size, Image.LANCZOS)
    heat_small = heat_img.convert("L").resize(display_size, Image.NEAREST)
    canvas = Image.new("RGB", (display_size[0] * 2 + 20, display_size[1]), "white")
    canvas.paste(orig_small, (0, 0))
    canvas.paste(heat_small.convert("RGB"), (display_size[0] + 20, 0))
    return canvas


def classify(metrics: dict) -> str:
    """Objective first pass - qualitative visual judgment layered on top separately, not replaced by this."""
    if metrics["centroid_inside_document_bbox"]:
        return "document-dominated"
    return "border-dominated"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    model = timm.create_model(CANDIDATE, pretrained=True, features_only=True)
    model.eval()
    cfg = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**cfg)

    results = []
    for label, (rel_path, border_desc) in TEST_IMAGES.items():
        path = PROJECT_ROOT / rel_path
        orig = Image.open(path).convert("RGB")
        w, h = orig.size
        bbox, page_conf = get_page_bbox(str(path))

        print(f"\n=== {label} ({path.name}) ===")
        print(f"border artifacts present: {border_desc}")
        print(f"page_boundary bbox={bbox}  page_confidence={page_conf}")

        # --- BEFORE: full uncropped frame ---
        heat_before = compute_activation(model, transform, orig)
        if bbox:
            doc_center_before = ((bbox[0] + bbox[2]) / 2 / w, (bbox[1] + bbox[3]) / 2 / h)
            doc_bbox_before = (bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h)
        else:
            doc_center_before, doc_bbox_before = (0.5, 0.5), None
        metrics_before = hotspot_metrics(heat_before, doc_center_before, doc_bbox_before)
        classification_before = classify(metrics_before)
        panel_before = make_panel(orig, heat_before)
        panel_before_path = OUTPUT_DIR / f"{label}_BEFORE.png"
        panel_before.save(panel_before_path)

        # --- AFTER: cropped to the pipeline's own page_boundary ---
        if bbox:
            cropped = orig.crop((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
        else:
            cropped = orig
        heat_after = compute_activation(model, transform, cropped)
        doc_center_after, doc_bbox_after = (0.5, 0.5), None  # crop IS the document, by construction
        metrics_after = hotspot_metrics(heat_after, doc_center_after, doc_bbox_after)
        classification_after = "document-dominated"  # trivially true post-crop unless flagged otherwise on visual review
        panel_after = make_panel(cropped, heat_after)
        panel_after_path = OUTPUT_DIR / f"{label}_AFTER.png"
        panel_after.save(panel_after_path)

        migration = metrics_before["distance_from_document_center"] - metrics_after["distance_from_document_center"]

        print(f"BEFORE: {metrics_before}  -> {classification_before}")
        print(f"AFTER:  {metrics_after}  -> {classification_after} (trivial-by-construction, see qualitative note)")
        print(f"distance migration (before - after): {migration:+.3f}  (positive = moved toward document center)")

        results.append({
            "label": label, "image": str(path), "border_artifacts": border_desc,
            "page_boundary_bbox": bbox, "page_confidence": page_conf,
            "before": {**metrics_before, "classification": classification_before, "panel": str(panel_before_path)},
            "after": {**metrics_after, "classification": classification_after, "panel": str(panel_after_path)},
            "distance_migration": migration,
        })

    RESULTS_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), "experiment": "3.1_crop_confound_test", "results": results}
    with open(RESULTS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"\nAppended full record to {RESULTS_LOG}")
    print(f"Panels saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
