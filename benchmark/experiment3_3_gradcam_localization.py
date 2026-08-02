"""
Experiment 3.3: Grad-CAM (task-conditioned attribution), the
replacement method recommended when Experiment 3 was closed - unlike
Experiments 3/3.1/3.2's unconditional channel-mean pooling, this
backpropagates a real classification target through the network, so
the resulting map reflects what the model considers relevant to ITS
prediction, not just raw activation magnitude.

Same image set, same page_boundary crop source (core/image_analysis.py,
not a manual crop), and the SAME characterize() metrics as Experiment
3.2 (weighted centroid, spread, entropy, top-k mass, argmax) - imported
directly, not reimplemented, so results are directly comparable to the
channel-mean baseline rather than measured on a different scale.

Caveat stated up front, not after the fact: ConvNeXt's classification
head is ImageNet-22k, which has never seen an archival document - the
predicted "class" backpropagated from will not be semantically
meaningful for this corpus. That's fine for this experiment's actual
purpose (does task-conditioned attribution produce a qualitatively
different, less uniform spatial pattern than unconditional pooling
did), which does not require the class label itself to be correct.

Usage:
    python -m benchmark.experiment3_3_gradcam_localization
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import timm
from PIL import Image, ImageDraw

from core.image_analysis import analyze_image
from benchmark.experiment3_2_robust_localization import (
    characterize, activation_distribution, get_page_bbox, TEST_IMAGES,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_LOG = PROJECT_ROOT / "data" / "outputs" / "experiment3_3_gradcam_log.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "experiment3_3_outputs"
CANDIDATE = "convnext_tiny.fb_in22k"


def compute_gradcam(model, transform, pil_image: Image.Image):
    x = transform(pil_image.convert("RGB")).unsqueeze(0)
    features = model.forward_features(x)
    features.retain_grad()
    logits = model.forward_head(features)
    top_class = logits.argmax(dim=1)
    top_confidence = float(torch.softmax(logits, dim=1)[0, top_class])
    loss = logits[0, top_class]
    model.zero_grad()
    loss.backward()
    grad = features.grad
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * features).sum(dim=1))
    heat = cam[0].detach().numpy()
    return heat, int(top_class.item()), top_confidence


def make_annotated_panel(orig: Image.Image, heat: np.ndarray, metrics: dict, display_w=700) -> Image.Image:
    heat_norm = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    heat_img = Image.fromarray((heat_norm * 255).astype(np.uint8)).resize(orig.size, Image.NEAREST).convert("RGB")

    p = activation_distribution(heat)
    h, w = heat.shape
    flat_sorted_idx = np.argsort(p.flatten())[::-1]
    cutoff_count = max(1, round(0.50 * p.size))
    top50_mask = np.zeros_like(p, dtype=bool)
    top50_mask.flat[flat_sorted_idx[:cutoff_count]] = True

    cell_w, cell_h = orig.width / w, orig.height / h
    draw = ImageDraw.Draw(heat_img)
    for r in range(h):
        for c in range(w):
            if top50_mask[r, c]:
                x0, y0 = c * cell_w, r * cell_h
                draw.rectangle([x0, y0, x0 + cell_w, y0 + cell_h], outline=(0, 200, 0), width=3)

    ax, ay = metrics["argmax_norm"][0] * orig.width, metrics["argmax_norm"][1] * orig.height
    cx, cy = metrics["weighted_centroid_norm"][0] * orig.width, metrics["weighted_centroid_norm"][1] * orig.height
    s = orig.width * 0.02
    draw.line([ax - s, ay - s, ax + s, ay + s], fill=(255, 0, 0), width=4)
    draw.line([ax - s, ay + s, ax + s, ay - s], fill=(255, 0, 0), width=4)
    draw.ellipse([cx - s, cy - s, cx + s, cy + s], outline=(0, 100, 255), width=4)

    scale = display_w / orig.width
    display_size = (display_w, int(orig.height * scale))
    orig_small = orig.resize(display_size, Image.LANCZOS)
    heat_small = heat_img.resize(display_size, Image.NEAREST)
    canvas = Image.new("RGB", (display_size[0] * 2 + 20, display_size[1]), "white")
    canvas.paste(orig_small, (0, 0))
    canvas.paste(heat_small, (display_size[0] + 20, 0))
    return canvas


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    model = timm.create_model(CANDIDATE, pretrained=True)  # real classification head, not num_classes=0
    model.eval()
    cfg = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**cfg)

    all_results = []
    for label, rel_path in TEST_IMAGES.items():
        path = PROJECT_ROOT / rel_path
        orig = Image.open(path).convert("RGB")
        w, h = orig.size
        bbox, page_conf = get_page_bbox(str(path))
        print(f"\n=== {label} ===  page_confidence={page_conf}  bbox={bbox}")

        conditions = {}
        for cond_name, img_for_model in [
            ("before", orig),
            ("after", orig.crop((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))) if bbox else orig),
        ]:
            heat, top_class, top_conf = compute_gradcam(model, transform, img_for_model)
            metrics = characterize(heat)

            if cond_name == "before" and bbox:
                doc_bbox_norm = (bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h)
                doc_center_norm = ((doc_bbox_norm[0] + doc_bbox_norm[2]) / 2, (doc_bbox_norm[1] + doc_bbox_norm[3]) / 2)
                p = activation_distribution(heat)
                hh, ww = heat.shape
                rows, cols = np.mgrid[0:hh, 0:ww]
                row_norm, col_norm = (rows + 0.5) / hh, (cols + 0.5) / ww
                inside = (col_norm >= doc_bbox_norm[0]) & (col_norm <= doc_bbox_norm[2]) & \
                          (row_norm >= doc_bbox_norm[1]) & (row_norm <= doc_bbox_norm[3])
                mass_inside_document = float(p[inside].sum())
            else:
                doc_center_norm = (0.5, 0.5)
                mass_inside_document = 1.0

            centroid_doc_dist = float(np.hypot(
                metrics["weighted_centroid_norm"][0] - doc_center_norm[0],
                metrics["weighted_centroid_norm"][1] - doc_center_norm[1],
            ))

            panel = make_annotated_panel(img_for_model, heat, metrics)
            panel_path = OUTPUT_DIR / f"{label}_{cond_name}.png"
            panel.save(panel_path)

            conditions[cond_name] = {
                **metrics,
                "top_predicted_class_idx": top_class,
                "top_class_confidence": top_conf,
                "distance_centroid_to_document_center": centroid_doc_dist,
                "activation_mass_inside_document_bbox": mass_inside_document,
                "panel": str(panel_path),
            }

            print(f"  [{cond_name}] top_class={top_class} (conf={top_conf:.3f})  "
                  f"centroid={metrics['weighted_centroid_norm']}  argmax={metrics['argmax_norm']}  "
                  f"centroid-argmax dist={metrics['centroid_argmax_distance']:.3f}  "
                  f"spread={metrics['spread_trace']:.4f}  entropy={metrics['entropy_normalized']:.3f}  "
                  f"top10/20/50%={metrics['top10pct_mass_fraction']:.2f}/"
                  f"{metrics['top20pct_mass_fraction']:.2f}/{metrics['top50pct_mass_fraction']:.2f}  "
                  f"dist-to-doc-center={centroid_doc_dist:.3f}  mass-inside-doc={mass_inside_document:.3f}")

        all_results.append({"label": label, "image": str(path), "page_boundary_bbox": bbox,
                             "page_confidence": page_conf, "conditions": conditions})

    RESULTS_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), "experiment": "3.3_gradcam_localization",
              "results": all_results}
    with open(RESULTS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"\nAppended full record to {RESULTS_LOG}")
    print(f"Panels saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
