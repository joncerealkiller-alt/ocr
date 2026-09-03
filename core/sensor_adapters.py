"""
EvidenceRecord adapters for the classical-CV sensors - the piece
core/routing_decision_engine.py's own "INTEGRATION STATUS" note flagged
as NOT YET BUILT ("layout-detector EvidenceRecord adapter... both are
captured but unevaluated against this project's own taxonomy at real
scale"). `config/decision_engine.yaml`'s `sensor_families.classical_cv`
already lists `table_confidence`/`layout_detector` by name - this module
is what actually produces EvidenceRecords under those exact sensor
names, so the config's existing plumbing becomes real instead of
aspirational.

Per core/routing_decision_engine.py's own "CORE PRINCIPLE" (that module
never runs a sensor) - the reverse discipline applies here: this module
DOES call real sensor code (core/image_analysis.py, core/layout_detector.py)
but never touches routing_decision_engine.py's arbitration logic. Clean
one-way dependency: sensor_adapters -> {image_analysis, layout_detector,
routing_decision_engine (only for the EvidenceRecord dataclass)}.

DELIBERATELY CONSERVATIVE, not a full layout-based classifier: both
adapters below return None (no usable evidence) far more often than they
return a real EvidenceRecord - per the "captured but unevaluated at real
scale" caveat, this only votes when the structural read is close to
unambiguous, rather than pretending classical CV/YOLO region detection
alone can discriminate all 11 taxonomy categories. Building a genuine
multi-class layout classifier is a separate, larger, not-yet-validated
project - see docs/MULTI_SOURCE_VOTING_CLASSIFIER_PROPOSAL.md for why
this project promotes rules only after measured evidence, never before.
"""

from __future__ import annotations

from core.image_analysis import ImageAnalysis
from core.routing_decision_engine import EvidenceRecord

# core/image_analysis.py's own real, measured table_confidence floor (16
# labelled census pages scored min 0.375; 218 microfilm non-tables scored
# median 0.125) - see that module's docstring and config/decision_
# engine.yaml's threshold comment for the calibration this number comes
# from. Below this floor, table_confidence has never been shown to mean
# anything - treated as "no usable evidence," not a vote against a table.
TABLE_CONFIDENCE_FLOOR = 0.375

# DocLayout-YOLO's 10 class names (core/layout_detector.py), grouped by
# what they structurally imply. class_name strings come from the loaded
# model at call time in detect_layout() - these sets are matched against
# whatever detect_layout() actually returned, not re-declared elsewhere.
_TABLE_CLASSES = {"table", "table_caption", "table_footnote"}
_TEXT_CLASSES = {"title", "plain text"}
_FIGURE_CLASSES = {"figure", "figure_caption"}


def table_confidence_evidence(analysis: ImageAnalysis) -> EvidenceRecord | None:
    """
    core/image_analysis.py's classical ruling-line/table-boundary
    detector, as routing evidence. table_confidence measures "is a table
    region present" - the only class it can honestly vote for is
    dense_tabular_rows (this project's one category defined by repeating
    tabular structure); it has no basis to vote FOR or AGAINST any other
    category, so every other outcome is "no usable evidence" (None),
    never a vote for "not dense_tabular_rows".
    """
    conf = analysis.table_confidence
    if conf is None or conf < TABLE_CONFIDENCE_FLOOR:
        return None
    return EvidenceRecord(
        sensor_name="table_confidence",
        sensor_family="classical_cv",
        predicted_class="dense_tabular_rows",
        raw_score=conf,
        normalized_score=conf,
        calibrated=False,
        metadata={
            "score_kind": "ordinal_confidence",
            "table_boundary": analysis.table_boundary,
            "note": "classical ruling-line/table-boundary detector, core/image_analysis.py",
        },
    )


def layout_detector_evidence(detections: list[dict]) -> EvidenceRecord | None:
    """
    core/layout_detector.py's DocLayout-YOLO region detections, as
    routing evidence. Only votes in two unambiguous cases:

    1. A real table/table_caption/table_footnote region was detected ->
       dense_tabular_rows, scored by that detection's own confidence.
       Independent measurement mechanism from table_confidence's
       classical ruling-line approach (same sensor_family="classical_cv"
       so the Decision Engine's family-vote mechanism treats these two
       as ONE combined vote, not two separately-counted ones, exactly
       the correlated-sensor discounting weighted_fusion already applies
       to the vision-tower ensemble).
    2. Both a figure/figure_caption region AND a title/plain-text region
       were detected, with NO table region - this matches mixed_text_
       image's own classifier_guidance directly ("substantial text AND
       a photograph/illustration... neither dominates").

    Every other detection pattern (text-only, figure-only, empty) is
    None - deliberately not attempting to distinguish printed_document
    vs. website_screenshot vs. portrait_photo vs. anything else from
    layout regions alone; that would need real validated evidence this
    sensor doesn't have yet.
    """
    if not detections:
        return None

    table_dets = [d for d in detections if d["class_name"] in _TABLE_CLASSES]
    text_dets = [d for d in detections if d["class_name"] in _TEXT_CLASSES]
    figure_dets = [d for d in detections if d["class_name"] in _FIGURE_CLASSES]

    if table_dets:
        best = max(table_dets, key=lambda d: d["confidence"])
        return EvidenceRecord(
            sensor_name="layout_detector",
            sensor_family="classical_cv",
            predicted_class="dense_tabular_rows",
            raw_score=best["confidence"],
            normalized_score=best["confidence"],
            calibrated=False,
            metadata={
                "score_kind": "ordinal_confidence",
                "n_table_regions": len(table_dets),
                "note": "DocLayout-YOLO table-class region detected",
            },
        )

    if figure_dets and text_dets:
        scores = [d["confidence"] for d in figure_dets + text_dets]
        return EvidenceRecord(
            sensor_name="layout_detector",
            sensor_family="classical_cv",
            predicted_class="mixed_text_image",
            raw_score=min(scores),
            normalized_score=None,
            calibrated=False,
            metadata={
                "score_kind": "ordinal_confidence",
                "n_figure_regions": len(figure_dets),
                "n_text_regions": len(text_dets),
                "note": "figure + text regions detected, no table region",
            },
        )

    return None
