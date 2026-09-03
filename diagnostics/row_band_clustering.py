"""
Reusable clustering logic for the census-bootstrap checkpoint's "row"
class detections (census_bootstrap_v4/weights/best.pt, per docs/
LAYOUT_DETECTOR_BOOTSTRAP_TRAINING.md) - built 2026-08-09 after visual
inspection of handwritten_ledger images showed the raw "row" detections
were NOT random noise: on both images that produced 90+ overlapping
boxes, the boxes were tightly concentrated on the actual bands of
repeated handwritten record structure and correctly avoided archival
labels/margins/header text - a real localization-quality problem
(heavy box duplication), not a signal-quality problem.

This module turns that raw, over-generated box soup into a small number
of coherent "row bands" (contiguous vertical regions of dense row-like
detections) plus a few summary features - a candidate SIGNAL for
"repeated record structure is present in this region," distinct from
attempting precise per-line row localization (which the training doc
already found needs far more data to work well).

DELIBERATELY NOT WIRED into core/sensor_adapters.py or the Decision
Engine yet - per Jon's explicit "build it as a test item that could be
integrated if it proves it's a valid signal" (2026-08-09). See
diagnostics/test_row_band_signal_validity.py for the actual validation
experiment this module exists to support.
"""

from __future__ import annotations

import statistics


def cluster_row_detections_into_bands(
    detections: list[dict],
    gap_ratio: float = 1.5,
) -> list[dict]:
    """
    Groups raw "row"-class detections into vertically-contiguous bands.
    Two detections join the same band if the vertical gap between them
    is <= gap_ratio * the median row-box height seen in this image -
    i.e. "close enough to be part of the same run of repeated lines,"
    not an exact line-matching attempt. Detections of any OTHER class
    (table/header) are ignored here - this function is specifically
    about the row class's over-generation pattern.

    Returns bands sorted top-to-bottom, each:
    {"n_boxes": int, "y_top": float, "y_bottom": float, "height": float,
     "mean_confidence": float} - in the same pixel coordinate space
    detect_layout()/detect_layout_v26() already return (source image
    pixels, not resized/normalized).
    """
    row_dets = [d for d in detections if d["class_name"] == "row"]
    if not row_dets:
        return []

    row_dets = sorted(row_dets, key=lambda d: d["bbox_xyxy"][1])
    heights = [d["bbox_xyxy"][3] - d["bbox_xyxy"][1] for d in row_dets]
    median_height = statistics.median(heights) if heights else 10.0
    gap_threshold = median_height * gap_ratio

    raw_bands: list[list[dict]] = [[row_dets[0]]]
    current_max_y1 = row_dets[0]["bbox_xyxy"][3]
    for d in row_dets[1:]:
        y0 = d["bbox_xyxy"][1]
        if (y0 - current_max_y1) <= gap_threshold:
            raw_bands[-1].append(d)
            current_max_y1 = max(current_max_y1, d["bbox_xyxy"][3])
        else:
            raw_bands.append([d])
            current_max_y1 = d["bbox_xyxy"][3]

    bands = []
    for band in raw_bands:
        y0s = [d["bbox_xyxy"][1] for d in band]
        y1s = [d["bbox_xyxy"][3] for d in band]
        bands.append({
            "n_boxes": len(band),
            "y_top": min(y0s),
            "y_bottom": max(y1s),
            "height": max(y1s) - min(y0s),
            "mean_confidence": statistics.mean(d["confidence"] for d in band),
        })
    return bands


def row_band_signal(detections: list[dict], image_height: float, min_boxes_for_substantial: int = 5) -> dict:
    """
    Image-level summary features from the clustered bands - candidate
    inputs for a future EvidenceRecord, NOT an EvidenceRecord itself
    (this module has zero import of core/routing_decision_engine.py on
    purpose, matching that module's own "never runs/produces evidence
    directly" boundary discipline until a signal is actually validated).

    "substantial" bands (>= min_boxes_for_substantial boxes) filter out
    stray single/double detections that don't represent real repeated
    structure - a real result on both the census and handwritten_ledger
    inspection images had dozens of boxes per genuine band, so a low bar
    like 5 already discards noise while keeping real signal.
    """
    bands = cluster_row_detections_into_bands(detections)
    substantial = [b for b in bands if b["n_boxes"] >= min_boxes_for_substantial]
    coverage = sum(b["height"] for b in bands) / image_height if image_height else 0.0
    substantial_coverage = sum(b["height"] for b in substantial) / image_height if image_height else 0.0
    return {
        "n_bands": len(bands),
        "n_substantial_bands": len(substantial),
        "total_row_boxes": sum(b["n_boxes"] for b in bands),
        "max_band_box_count": max((b["n_boxes"] for b in bands), default=0),
        "coverage_fraction": round(coverage, 4),
        "substantial_coverage_fraction": round(substantial_coverage, 4),
    }
