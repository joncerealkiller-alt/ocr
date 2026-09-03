"""
EXPERIMENTAL ONLY - not wired into the pipeline, not a replacement for
core/auto_sidecar.py's CV-only generate_auto_sidecar() (still the
production path, completely untouched by this module - not one line
of core/auto_sidecar.py is modified).

Tests whether seeding core/auto_sidecar.py's EXISTING
locate_table_boundary() search with a YOLO-detected table region (the
data/outputs/layout_bootstrap_train/runs/census_bootstrap_v2/weights/
best.pt checkpoint - the v2 bootstrap that showed real signal, see
docs/LAYOUT_DETECTOR_BOOTSTRAP_TRAINING.md) produces better table
boundaries than the template's fixed fractional prior alone.

THE KEY MECHANISM, worth being explicit about since it's the entire
trick this module relies on: locate_table_boundary() (core/
auto_sidecar.py) never takes a bbox argument directly - it derives its
search center (expected_top/bottom/left/right) purely from
`template.regions_approx["table"]`'s x_frac/y_frac fractions, then runs
a FIXED-RADIUS CV search (ruling-line/density-minimum detection) around
that point, snapping to whatever real printed boundary it finds nearby.
So a YOLO table hint can be injected with ZERO changes to
locate_table_boundary() itself: build a CLONED template
(dataclasses.replace(), never mutates the original) whose
regions_approx["table"] fractions reflect YOLO's proposal instead of
the template file's hardcoded default, then call the real, unmodified
locate_table_boundary() with that clone. Every other CV stage (header
region, columns, rows, quarantine) runs completely unchanged afterward,
consuming whatever boundary locate_table_boundary() actually settled
on - YOLO never touches those directly, exactly per the "coarse prior,
CV owns precise geometry" architecture this experiment was scoped to.

Coordinate-space discipline: core/auto_sidecar.py's generate_auto_sidecar()
deskews BEFORE calling locate_table_boundary() (bboxes are in deskewed-
image space). generate_auto_sidecar_yolo_assisted() below deskews first,
identically, THEN runs YOLO on that same deskewed image - no separate
coordinate transform is needed because YOLO and the CV boundary search
observe the exact same pixels in the exact same space.

Does NOT import ultralytics/YOLO at module level - only inside
build_yolo_table_model() and the assisted-generation path, so nothing
here forces a YOLO import cost onto a caller that only wants the
selection/validation helpers, and definitely never onto anything in
core/.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from PIL import Image

from core.document_templates import DocumentTemplate, load_template
from core.auto_sidecar import (
    ClassificationResult, classify_document, estimate_deskew_angle, apply_deskew_angle,
    locate_table_boundary, locate_header_region, detect_data_rows, locate_columns,
    init_column_state, _quarantine_whole_page, _quarantine_anomalous_rows,
    AutoSidecarResult, render_auto_debug_overlay,
)
from core.row_segmentation import build_sidecar

V2_CHECKPOINT_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "outputs" / "layout_bootstrap_train"
    / "runs" / "census_bootstrap_v2" / "weights" / "best.pt"
)

# -- tunable, documented selection/validation constants ---------------------
# Plausible table area-fraction range for this document family (a census
# table occupies the large majority of the page, per every real example
# checked in the bootstrap runs - direct observation, not a guess):
# rejects both noise-sized boxes and a degenerate near-full-image box.
_PLAUSIBLE_AREA_FRAC_RANGE = (0.05, 0.98)
# Census tables are landscape-oriented (wide columns spread across page
# width); reject boxes far outside a plausible width/height ratio.
_PLAUSIBLE_ASPECT_RATIO_RANGE = (0.3, 6.0)
# Centrality weighting: how much a candidate's distance from image
# center penalizes its selection score, relative to its confidence.
_CENTRALITY_WEIGHT = 0.25
# Padding applied to the selected/validated hint before use, as a
# fraction of the hint's own width/height (spec: "initially around 1-3%").
_HINT_PADDING_FRAC = 0.02


def build_yolo_table_model(checkpoint_path: Path = V2_CHECKPOINT_PATH):
    """Loads the v2 bootstrap checkpoint via the standard ultralytics
    YOLO class - same loading mechanism core/layout_detector_v26.py
    uses for the vanilla pretrained checkpoint, just pointed at our
    fine-tuned weights instead. NEVER called from core/ or any
    production path - experimental-script callers only."""
    from ultralytics import YOLO

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"v2 bootstrap checkpoint not found at {checkpoint_path} - run "
            f"training/layout_bootstrap_v2_train.py first, or pass an explicit checkpoint_path."
        )
    return YOLO(str(checkpoint_path))


def detect_table_candidates(model, deskewed_image: Image.Image, conf: float = 0.1) -> list[dict]:
    """Runs the v2 checkpoint against the DESKEWED image and returns
    only "table"-class detections (class name from the v2 dataset's own
    3-class scheme: table/row/header - see training/layout_bootstrap_
    v2_combined_dataset.py). conf=0.1, lower than core/layout_detector_
    v26.py's DEFAULT_CONF=0.2, chosen deliberately: the v2 checkpoint's
    "table" class already showed high precision even at low confidence
    in the bootstrap comparison (docs/LAYOUT_DETECTOR_BOOTSTRAP_TRAINING.md),
    and this experiment's own validate_and_pad_table_hint() below is the
    real quality gate, not the confidence cutoff."""
    results = model.predict(deskewed_image, imgsz=1280, conf=conf, verbose=False)
    result = results[0]
    names = result.names
    candidates = []
    boxes = result.boxes
    if boxes is None:
        return candidates
    for box in boxes:
        class_id = int(box.cls.item())
        class_name = names.get(class_id, str(class_id))
        if class_name != "table":
            continue
        candidates.append({
            "confidence": round(float(box.conf.item()), 4),
            "bbox_xyxy": [round(float(v), 1) for v in box.xyxy[0].tolist()],
        })
    return candidates


def select_table_candidate(candidates: list[dict], image_size: tuple[int, int]) -> dict | None:
    """
    Documented selection rule for choosing among multiple YOLO "table"
    detections (spec requirement: "choose using a documented rule based
    on confidence, plausible page coverage, containment of the expected
    census structure, and central/page alignment"):

    1. Filter to candidates within _PLAUSIBLE_AREA_FRAC_RANGE (page-
       coverage plausibility - this document family's real table region
       is large; both noise-sized and degenerate near-full-page boxes
       are excluded here).
    2. Score each survivor: confidence - _CENTRALITY_WEIGHT * (normalized
       distance of the box's center from the image's center) - a census
       table sits in the main page body, not hugging an edge/corner, so
       this directly operationalizes "central/page alignment." No
       separate "containment of expected census structure" signal
       exists beyond this (no independent structure detector to check
       containment against) - the area-fraction filter and centrality
       score together are this rule's full operationalization of that
       requirement, not a fourth independent term.
    3. Return the highest-scoring survivor, or None if none survive step 1.
    """
    img_w, img_h = image_size
    img_area = img_w * img_h
    cx_img, cy_img = img_w / 2, img_h / 2
    max_dist = (cx_img ** 2 + cy_img ** 2) ** 0.5

    survivors = []
    for c in candidates:
        x0, y0, x1, y1 = c["bbox_xyxy"]
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        frac = area / img_area if img_area > 0 else 0.0
        if _PLAUSIBLE_AREA_FRAC_RANGE[0] <= frac <= _PLAUSIBLE_AREA_FRAC_RANGE[1]:
            survivors.append((c, frac))

    if not survivors:
        return None

    best, best_score = None, float("-inf")
    for c, frac in survivors:
        x0, y0, x1, y1 = c["bbox_xyxy"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        dist = ((cx - cx_img) ** 2 + (cy - cy_img) ** 2) ** 0.5
        normalized_dist = dist / max_dist if max_dist > 0 else 0.0
        score = c["confidence"] - _CENTRALITY_WEIGHT * normalized_dist
        if score > best_score:
            best, best_score = c, score
    return best


def validate_and_pad_table_hint(
    bbox_xyxy: list[float], image_size: tuple[int, int],
) -> tuple[tuple[int, int, int, int] | None, str]:
    """
    Structural sanity checks (spec: "implausibly small area, implausibly
    narrow or short aspect, mostly outside the image, or no credible
    table detection") + padding + clamping. Returns (bbox_or_None, reason).

    A None first element means REJECT - caller must fall back to the
    unmodified template (baseline behavior), per spec.
    """
    img_w, img_h = image_size
    x0, y0, x1, y1 = bbox_xyxy

    # Clamp to image bounds first, so "mostly outside the image" is
    # measured against what actually survives clamping.
    clamped_x0, clamped_y0 = max(0.0, x0), max(0.0, y0)
    clamped_x1, clamped_y1 = min(float(img_w), x1), min(float(img_h), y1)
    if clamped_x1 <= clamped_x0 or clamped_y1 <= clamped_y0:
        return None, "clamped box has zero/negative area - entirely outside image bounds"

    orig_area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    clamped_area = (clamped_x1 - clamped_x0) * (clamped_y1 - clamped_y0)
    if orig_area > 0 and (clamped_area / orig_area) < 0.5:
        return None, f"mostly outside image bounds (only {clamped_area/orig_area:.0%} retained after clamping)"

    img_area = img_w * img_h
    frac = clamped_area / img_area if img_area > 0 else 0.0
    if frac < _PLAUSIBLE_AREA_FRAC_RANGE[0]:
        return None, f"implausibly small area ({frac:.1%} of image, minimum {_PLAUSIBLE_AREA_FRAC_RANGE[0]:.0%})"
    if frac > _PLAUSIBLE_AREA_FRAC_RANGE[1]:
        return None, f"implausibly large area ({frac:.1%} of image, maximum {_PLAUSIBLE_AREA_FRAC_RANGE[1]:.0%})"

    width, height = clamped_x1 - clamped_x0, clamped_y1 - clamped_y0
    aspect = width / height if height > 0 else float("inf")
    if not (_PLAUSIBLE_ASPECT_RATIO_RANGE[0] <= aspect <= _PLAUSIBLE_ASPECT_RATIO_RANGE[1]):
        return None, f"implausible aspect ratio ({aspect:.2f}, expected {_PLAUSIBLE_ASPECT_RATIO_RANGE})"

    pad_x, pad_y = width * _HINT_PADDING_FRAC, height * _HINT_PADDING_FRAC
    padded_x0 = max(0, int(clamped_x0 - pad_x))
    padded_y0 = max(0, int(clamped_y0 - pad_y))
    padded_x1 = min(img_w, int(clamped_x1 + pad_x))
    padded_y1 = min(img_h, int(clamped_y1 + pad_y))

    return (padded_x0, padded_y0, padded_x1, padded_y1), "accepted"


def build_hinted_template(template: DocumentTemplate, table_bbox: tuple[int, int, int, int],
                           image_size: tuple[int, int]) -> DocumentTemplate:
    """Clones `template` (dataclasses.replace - the original object is
    never mutated) with regions_approx["table"] overridden to reflect
    table_bbox as x_frac/y_frac fractions of image_size - the ONLY
    change from the real template. Every other region (metadata, header,
    columns) and every other field (row_strategy, expected_row_count,
    etc.) is untouched, so locate_header_region()/detect_data_rows()/
    locate_columns() downstream all still use their normal, real
    calibrated approximations - only the table search's own starting
    point changes."""
    img_w, img_h = image_size
    x0, y0, x1, y1 = table_bbox
    new_regions_approx = dict(template.regions_approx)
    new_regions_approx["table"] = {
        "x_frac": [x0 / img_w, x1 / img_w],
        "y_frac": [y0 / img_h, y1 / img_h],
    }
    return dataclasses.replace(template, regions_approx=new_regions_approx)


def generate_auto_sidecar_yolo_assisted(
    image_path, yolo_model, doc_type_override: str | None = None,
    external_confidence: float | None = None, debug: bool = False,
) -> tuple[AutoSidecarResult, dict[str, Any]]:
    """
    Mirrors core/auto_sidecar.py's generate_auto_sidecar() EXACTLY,
    stage for stage, EXCEPT the table-boundary step uses a YOLO-hinted
    template (built via build_hinted_template() above) instead of the
    template's raw file default, when a hint survives selection/
    validation. Falls back to the real, unmodified template (identical
    to generate_auto_sidecar()'s own behavior) when YOLO finds nothing
    plausible - never a silent failure, always recorded in the returned
    experiment_diagnostics dict.

    Returns (AutoSidecarResult, experiment_diagnostics) - the second
    element is THIS experiment's own bookkeeping (hint candidates, which
    was selected, validation outcome, whether the hint was actually
    used) and is not part of core/auto_sidecar.py's real schema.
    """
    image_path = str(image_path)
    original = Image.open(image_path).convert("RGB")
    from core.image_analysis import DESKEW_ANGLE_RANGE
    angle = estimate_deskew_angle(original, angle_range=DESKEW_ANGLE_RANGE)
    deskewed = apply_deskew_angle(original, angle)

    if doc_type_override is not None:
        classification = ClassificationResult(
            doc_type=doc_type_override,
            confidence=external_confidence if external_confidence is not None else 1.0,
            template_name=doc_type_override,
            features={}, scores={"source": "external_override"},
        )
    else:
        classification = classify_document(deskewed)
    used_cv_fallback = doc_type_override is None

    experiment_diag: dict[str, Any] = {"deskew_angle": angle}

    if classification.doc_type == "unknown":
        experiment_diag["hint_used"] = False
        experiment_diag["hint_reason"] = "classification unknown - no template to hint against"
        return AutoSidecarResult(
            used_cv_fallback=used_cv_fallback, sidecar=None, debug_overlay=None,
            classification=classification, diagnostics={"used_cv_fallback": used_cv_fallback},
            warnings=["classification unknown"],
        ), experiment_diag

    template = load_template(classification.doc_type)
    image_size = deskewed.size

    raw_candidates = detect_table_candidates(yolo_model, deskewed)
    experiment_diag["yolo_candidates"] = raw_candidates
    selected = select_table_candidate(raw_candidates, image_size)

    active_template = template
    if selected is None:
        experiment_diag["hint_used"] = False
        experiment_diag["hint_reason"] = (
            "no credible table detection" if not raw_candidates
            else "no candidate survived plausible-area filter"
        )
    else:
        validated_bbox, reason = validate_and_pad_table_hint(selected["bbox_xyxy"], image_size)
        experiment_diag["yolo_selected_candidate"] = selected
        experiment_diag["hint_validation_reason"] = reason
        if validated_bbox is None:
            experiment_diag["hint_used"] = False
            experiment_diag["hint_reason"] = reason
        else:
            experiment_diag["hint_used"] = True
            experiment_diag["hint_bbox"] = validated_bbox
            active_template = build_hinted_template(template, validated_bbox, image_size)

    table_bbox, table_diag = locate_table_boundary(deskewed, active_template)
    header_bbox, metadata_bottom, header_diag = locate_header_region(deskewed, active_template, table_bbox)
    result, row_diag = detect_data_rows(deskewed, active_template, table_bbox, metadata_bottom, angle, original)

    diagnostics = {
        "used_cv_fallback": used_cv_fallback,
        "classification_features": classification.features,
        "classification_scores": classification.scores,
        "deskew_angle": angle,
        "template": template.doc_type,
        "table_boundary": table_diag,
        "header_region": header_diag,
        "row_detection": row_diag,
    }

    sidecar = build_sidecar(
        result, source_image_path=image_path, mode=f"yolo_assisted_auto_{template.row_strategy}",
        parameters={
            "doc_type": classification.doc_type, "classification_confidence": classification.confidence,
            "template": template.doc_type, "expected_row_count": template.expected_row_count,
            "yolo_hint_used": experiment_diag["hint_used"],
        },
        table_top=table_bbox[1], table_bottom=table_bbox[3],
        x0=table_bbox[0], x1=table_bbox[2], metadata_bottom=metadata_bottom,
    )
    init_column_state(sidecar, template.expected_columns)

    column_masks, column_diag = locate_columns(deskewed, active_template, table_bbox)
    flagged_columns = []
    for col_name, keep_ranges in column_masks.items():
        sidecar["columns"][col_name]["mask_keep_ranges"] = [list(r) for r in keep_ranges]
        if column_diag.get(col_name, {}).get("width_implausible"):
            sidecar["columns"][col_name]["status"] = "needs_review"
            flagged_columns.append(col_name)
    diagnostics["column_locating"] = column_diag
    diagnostics["flagged_columns"] = flagged_columns

    fallback_warning = (
        ["Template chosen by the CV classifier fallback - GUESS, not a trusted classification."]
        if used_cv_fallback else []
    )

    if row_diag.get("page_detection_failed"):
        quarantined = _quarantine_whole_page(
            sidecar, reason="row detection failed structural sanity checks on every preset tried")
    elif table_diag.get("table_top_ambiguous"):
        quarantined = _quarantine_whole_page(
            sidecar, reason="table_top looked structurally ambiguous")
    else:
        quarantined = _quarantine_anomalous_rows(sidecar)
    diagnostics["quarantined_rows"] = [{"index": r["index"], "reason": r["reason"]} for r in quarantined]

    debug_overlay = None
    if debug:
        debug_overlay = render_auto_debug_overlay(
            deskewed, table_bbox, header_bbox, sidecar["rows"], sidecar.get("rows_needs_review"))

    return AutoSidecarResult(
        used_cv_fallback=used_cv_fallback, sidecar=sidecar, debug_overlay=debug_overlay,
        classification=classification, diagnostics=diagnostics, warnings=fallback_warning,
    ), experiment_diag
