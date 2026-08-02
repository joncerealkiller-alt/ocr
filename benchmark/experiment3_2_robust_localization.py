"""
Experiment 3.2: measurement validation, not a ConvNeXt verdict.

Experiment 3.1 found that single-cell argmax is fragile (a near-null
crop flipped it entirely) and cannot distinguish "border artifacts
cause mislocalization" from "ConvNeXt's coarse stage has a
content-independent center bias." This experiment replaces the single
statistic with a full characterization of the activation distribution,
per Jon's explicit requirement list, and treats validating the METRIC
as the primary objective - conclusions about ConvNeXt itself are
secondary and contingent on the metric surviving this validation.

H0: the apparent center preference is primarily an argmax-reduction
    artifact (centroid/spread stay stable while argmax jumps around).
H1: the center preference persists under robust statistics too
    (centroid ALSO consistently centers despite real image changes) -
    a genuine representation property, not a metric artifact.

Same image set and crop conditions as Experiment 3.1 (requirement #1):
core/image_analysis.py's own page_boundary detector, not a manual crop.

Usage:
    python -m benchmark.experiment3_2_robust_localization
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import timm
from PIL import Image, ImageDraw

from core.image_analysis import analyze_image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_LOG = PROJECT_ROOT / "data" / "outputs" / "experiment3_2_robust_localization_log.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "experiment3_2_outputs"
CANDIDATE = "convnext_tiny.fb_in22k"

TEST_IMAGES = {
    "difficult_A_c10264.767": "data/working/oocihm.lac_reel_c10264.767.jpg",
    "difficult_B_c10295.1199": "data/working/oocihm.lac_reel_c10295.1199.jpg",
    "clean_C_c10301.529": "data/working/oocihm.lac_reel_c10301.529.jpg",
}


def get_page_bbox(image_path: str):
    a = analyze_image(image_path)
    pb = a.page_boundary
    if not pb:
        return None, a.page_confidence
    xs = [pt[0] for pt in pb]
    ys = [pt[1] for pt in pb]
    return (min(xs), min(ys), max(xs), max(ys)), a.page_confidence


def activation_distribution(heat: np.ndarray) -> np.ndarray:
    """Turn a raw activation map into a proper probability distribution
    over cells - shift to non-negative, normalize to sum to 1. Linear
    (not softmax) normalization, deliberately: softmax's temperature
    would be an extra arbitrary parameter, and the point here is to
    characterize the distribution as measured, not to sharpen it."""
    shifted = heat - heat.min()
    total = shifted.sum()
    return shifted / total if total > 0 else np.full_like(heat, 1.0 / heat.size)


def characterize(heat: np.ndarray) -> dict:
    h, w = heat.shape
    p = activation_distribution(heat)

    rows, cols = np.mgrid[0:h, 0:w]
    row_norm = (rows + 0.5) / h
    col_norm = (cols + 0.5) / w

    centroid_row = float((p * row_norm).sum())
    centroid_col = float((p * col_norm).sum())
    centroid_norm = (centroid_col, centroid_row)  # (x, y)

    var_row = float((p * (row_norm - centroid_row) ** 2).sum())
    var_col = float((p * (col_norm - centroid_col) ** 2).sum())
    cov_rc = float((p * (row_norm - centroid_row) * (col_norm - centroid_col)).sum())
    spread_trace = var_row + var_col  # total variance, a single "spread" scalar

    nonzero = p[p > 0]
    entropy = float(-(nonzero * np.log(nonzero)).sum())
    max_entropy = float(np.log(heat.size))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    sorted_p = np.sort(p.flatten())[::-1]
    n = len(sorted_p)
    top10_frac = float(sorted_p[: max(1, round(0.10 * n))].sum())
    top20_frac = float(sorted_p[: max(1, round(0.20 * n))].sum())
    top50_frac = float(sorted_p[: max(1, round(0.50 * n))].sum())

    argmax_idx = np.unravel_index(np.argmax(heat), heat.shape)
    argmax_norm = ((argmax_idx[1] + 0.5) / w, (argmax_idx[0] + 0.5) / h)
    centroid_argmax_dist = float(np.hypot(centroid_norm[0] - argmax_norm[0], centroid_norm[1] - argmax_norm[1]))

    return {
        "weighted_centroid_norm": centroid_norm,
        "argmax_norm": argmax_norm,
        "centroid_argmax_distance": centroid_argmax_dist,
        "spread_var_row": var_row, "spread_var_col": var_col, "spread_cov": cov_rc,
        "spread_trace": spread_trace,
        "entropy_normalized": normalized_entropy,
        "top10pct_mass_fraction": top10_frac,
        "top20pct_mass_fraction": top20_frac,
        "top50pct_mass_fraction": top50_frac,
    }


def make_annotated_panel(orig: Image.Image, heat: np.ndarray, metrics: dict, display_w=700) -> Image.Image:
    heat_norm = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    heat_img = Image.fromarray((heat_norm * 255).astype(np.uint8)).resize(orig.size, Image.NEAREST).convert("RGB")

    # Top-50%-mass contour proxy: outline the cells contributing to the
    # top 50% activation mass directly on the full-resolution heat image,
    # before downscaling for display (kept simple, per "optional if
    # straightforward" - a thresholded cell outline, not a smooth contour).
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

    # argmax marker (red X) and weighted centroid marker (blue circle)
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
    model = timm.create_model(CANDIDATE, pretrained=True, features_only=True)
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
            x = transform(img_for_model.convert("RGB")).unsqueeze(0)
            with torch.no_grad():
                feats = model(x)  # 4 stages

            stage_metrics = {}
            for stage_idx in range(len(feats)):
                heat = feats[stage_idx][0].mean(dim=0).numpy()
                stage_metrics[f"stage_{stage_idx}"] = characterize(heat)

            coarse_heat = feats[-1][0].mean(dim=0).numpy()
            coarse_metrics = stage_metrics["stage_3"]

            # document-center reference, and (for "before" only) fraction
            # of activation mass actually inside the true document bbox
            if cond_name == "before" and bbox:
                doc_bbox_norm = (bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h)
                doc_center_norm = ((doc_bbox_norm[0] + doc_bbox_norm[2]) / 2, (doc_bbox_norm[1] + doc_bbox_norm[3]) / 2)
                p = activation_distribution(coarse_heat)
                hh, ww = coarse_heat.shape
                rows, cols = np.mgrid[0:hh, 0:ww]
                row_norm, col_norm = (rows + 0.5) / hh, (cols + 0.5) / ww
                inside = (col_norm >= doc_bbox_norm[0]) & (col_norm <= doc_bbox_norm[2]) & \
                          (row_norm >= doc_bbox_norm[1]) & (row_norm <= doc_bbox_norm[3])
                mass_inside_document = float(p[inside].sum())
            else:
                doc_center_norm = (0.5, 0.5)  # crop IS the document, by construction
                mass_inside_document = 1.0

            centroid_doc_dist = float(np.hypot(
                coarse_metrics["weighted_centroid_norm"][0] - doc_center_norm[0],
                coarse_metrics["weighted_centroid_norm"][1] - doc_center_norm[1],
            ))

            panel = make_annotated_panel(img_for_model, coarse_heat, coarse_metrics)
            panel_path = OUTPUT_DIR / f"{label}_{cond_name}.png"
            panel.save(panel_path)

            conditions[cond_name] = {
                "stage_metrics": stage_metrics,
                "coarse_stage_summary": coarse_metrics,
                "distance_centroid_to_document_center": centroid_doc_dist,
                "activation_mass_inside_document_bbox": mass_inside_document,
                "panel": str(panel_path),
            }

            print(f"  [{cond_name}] centroid={coarse_metrics['weighted_centroid_norm']}  "
                  f"argmax={coarse_metrics['argmax_norm']}  "
                  f"centroid-argmax dist={coarse_metrics['centroid_argmax_distance']:.3f}  "
                  f"spread={coarse_metrics['spread_trace']:.4f}  "
                  f"entropy={coarse_metrics['entropy_normalized']:.3f}  "
                  f"top10/20/50%={coarse_metrics['top10pct_mass_fraction']:.2f}/"
                  f"{coarse_metrics['top20pct_mass_fraction']:.2f}/{coarse_metrics['top50pct_mass_fraction']:.2f}  "
                  f"dist-to-doc-center={centroid_doc_dist:.3f}  "
                  f"mass-inside-doc={mass_inside_document:.3f}")

        # cross-stage center-bias check (requirement #7: where does it come from)
        print("  cross-stage argmax/centroid (before condition):")
        for stage_idx in range(4):
            sm = conditions["before"]["stage_metrics"][f"stage_{stage_idx}"]
            print(f"    stage {stage_idx}: argmax={sm['argmax_norm']}  centroid={sm['weighted_centroid_norm']}  "
                  f"spread={sm['spread_trace']:.4f}")

        all_results.append({"label": label, "image": str(path), "page_boundary_bbox": bbox,
                             "page_confidence": page_conf, "conditions": conditions})

    RESULTS_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), "experiment": "3.2_robust_localization",
              "results": all_results}
    with open(RESULTS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"\nAppended full record to {RESULTS_LOG}")
    print(f"Panels saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
