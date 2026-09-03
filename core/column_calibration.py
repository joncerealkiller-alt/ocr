"""
Column-boundary calibration records - a second, PARALLEL artifact type
alongside the sidecars core/row_segmentation.py owns, built 2026-08-07
per Jon's "Column Calibration / Sidecar Correction UI" spec.

Purpose: core/auto_sidecar.py's locate_columns() (ruling-line search +
affine registration, see that function's own docstring) and the
research-only header-number-anchor detectors in row_segmentation.py are
both sensors/priors, not ground truth. The only way to measure their
real error pattern - systematic offset, appropriate search-window width,
year/form-specific geometry - is human-verified column geometry. This
module owns the correction record that captures AUTO -> HUMAN -> delta
for that purpose, and the merge step that folds a reviewed page's
corrections back into a normal sidecar without ever touching the
original sidecar file.

Narrow module, same pattern core/auto_sidecar.py itself already follows:
calls into core/row_segmentation.py (load_sidecar/save_sidecar) and
core/auto_sidecar.py (_fit_registration_affine, a precedented cross-
module private import - core/page_dewarp.py and core/warp_detection.py
already import underscore-prefixed helpers from row_segmentation.py the
same way) without modifying either.

BOUNDARY MODEL (per Jon's explicit decision, not the sidecar's own
per-column independent-range model): columns are ordered left-to-right
and share DIVIDING LINES - for N columns there are N-1 interior
dividers (the shared edge between column i and column i+1). The two
OUTER edges (left of column 0, right of the last column) are NOT
dividers here - they come from the sidecar's existing table_bbox and
are out of scope for this tool (table bounds are a different UI's job).
This is a genuine translation layer against the sidecar's own schema,
where each column independently stores mask_keep_ranges: [[x0,x1]] (two
neighbors CAN slightly disagree on their shared edge - real
locate_columns() behavior, not a bug) - build_correction_record()
reconciles that on load, merge_column_corrections() converts back on
save.

IMPORTANT DISTINCTION (Jon, 2026-08-07): existing sidecars' per-column
masks are EXTRACTION-oriented and may deliberately omit a column
entirely (e.g. every census template's "Occupation" - see locate_
columns()'s own docstring: columns absent from column_regions_approx are
silently skipped, left with empty mask_keep_ranges, by design, not a
detection failure). Calibration dividers are GEOMETRY-oriented and must
represent EVERY physical interior boundary regardless of whether either
neighboring column happens to have an extraction mask defined -
build_correction_record() always creates one divider entry per adjacent
column_order pair, never skipping a pair just because one or both
neighbors have no mask_keep_ranges yet; a divider with no sidecar data on
either side simply starts with auto_x=None ("nothing detected/masked
yet, needs a human to place it"), not as a missing/absent divider.

Never writes to the original sidecar file. Correction records live at
data/outputs/column_calibration/{stem}_correction.json - a separate
directory from data/outputs/row_segmentation/ so a `glob("*_sidecar.
json")` anywhere in the codebase (the pattern find_quarantined_
sidecars() in ui/quarantine_review_ui.py already uses) never picks one
up by accident.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.workspace_context import WorkspaceContext

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CALIBRATION_ROOT = WorkspaceContext.resolve().workspace_root / "research" / "calibration"

# Resolved via WorkspaceContext (2026-08-09, per Jon - see core.
# calibration_workspace.WORKSPACE_ROOT's own comment for the full "why":
# used to hardcode PROJECT_ROOT/data/outputs/column_calibration, which
# only worked via an NTFS junction to genealogy_workspace/research/
# calibration/column_calibration - same real data, same folder name,
# just no longer riding on the junction indirection).
CORRECTION_DIR = _CALIBRATION_ROOT / "column_calibration"

# RESERVED, not produced by this module or by ui/column_calibration_ui.py.
# A future offline script could populate this from row_segmentation.py's
# detect_column_number_centers()/find_number_row_band_piecewise() (real
# CV, deliberately never run live by this module or its UI - see
# try_load_anchor_cache()'s own docstring). Confirmed empty today: no
# such cache exists anywhere in the pipeline as of this writing -
# locate_columns()'s own diagnostics dict is computed in-memory inside
# core.auto_sidecar.generate_auto_sidecar() and discarded, never
# persisted. No legacy junction for this one (never populated) - placed
# directly under the same real calibration root as CORRECTION_DIR.
ANCHOR_CACHE_DIR = _CALIBRATION_ROOT / "column_number_anchors"

SCHEMA_VERSION = 7  # bumped 2026-08-07: added row_number_anchor (left/right
                     # margin row-number-column centers, see build_
                     # correction_record()'s docstring) - load_correction_
                     # record() migrates any older record missing this key.
                     # Bumped again 2026-08-08: added table_edges (left/
                     # right outer table boundary provenance, see build_
                     # correction_record()'s docstring) - same migration
                     # pattern, load_correction_record() adds it with
                     # defaults if missing.
                     # Bumped again 2026-08-09: added table_edges["top"]
                     # (table_bbox[1]/y0, a Y-axis boundary - table_top
                     # detection confirmed wrong on ~half a real batch in
                     # a separate standalone generate_auto_sidecar() run)
                     # - load_correction_record() migrates any record
                     # missing just this one sub-key too.
                     # Bumped again 2026-08-09 (same day): added table_
                     # edges["bottom"] (table_bbox[3]/y1) - per Jon, a
                     # human-marked real table bottom lets row pitch be
                     # computed exactly per-page ((bottom-top)/(row_count-1))
                     # instead of relying on one averaged template constant
                     # across pages with genuine registration variance (see
                     # the row-pitch sanity check that motivated this). Not
                     # yet WIRED IN to auto_sidecar's tiling - this is the
                     # UI/data-capture half only, per Jon's own sequencing
                     # ("do the ui first... then wiring it in").
                     # Bumped again 2026-08-09 (same day): added
                     # measured_row_pitch - a real, independently human-
                     # measured row pitch (2 clicks: row 1 top, row N top)
                     # from ui/column_calibration_ui.py's "Measure row
                     # pitch..." tool. Motivated by a real, confirmed
                     # circularity bug: once a template's expected_row_
                     # height_frac is set, core/auto_sidecar.py's fixed_
                     # periodic tiling SKIPS its own per-page row-1 search
                     # entirely and just multiplies the template's own
                     # value by page height for every row - meaning a
                     # regenerated sidecar's row bboxes can no longer
                     # supply an INDEPENDENT row-pitch measurement (they
                     # just echo the template back at zero spread,
                     # regardless of whether the template value is right).
                     # This field is the one remaining source of a real,
                     # template-independent per-page pitch measurement -
                     # core/template_refinement.py's refine_template_from_
                     # calibration() now reads THIS field for its ROW
                     # PITCH statistic instead of sidecar rows.
                     # Bumped again 2026-08-09 (same day): added
                     # manual_deskew_delta_deg - previously ui/column_
                     # calibration_ui.py's _preview_rotation_delta (manual
                     # nudge buttons / rotation-refinement test) only ever
                     # lived in memory and reset to 0.0 on every page
                     # reload. Per Jon: reopening an already-masked page
                     # re-derived deskew from scratch, and dividers placed
                     # against a PREVIOUS session's rotated preview could
                     # appear visibly shifted once that correction was
                     # gone. Persisting it here and restoring it on load
                     # fixes that.

# Divider provenance vocabulary. Deliberately does NOT reuse locate_
# columns()'s own "measured"/"affine_override"/"affine_predicted"/
# "locally_interpolated"/"template" tier vocabulary for auto_x, because
# that per-edge diagnostic is never persisted anywhere in the pipeline
# today (confirmed - see ANCHOR_CACHE_DIR comment above) and is out of
# scope to start persisting here; auto_x is always just "whatever the
# sidecar's mask_keep_ranges already said."
PROVENANCE_UNREVIEWED = "auto_accepted_unreviewed"
PROVENANCE_CONFIRMED = "human_confirmed"
PROVENANCE_PLACED = "human_placed"


def correction_path_for(image_stem: str) -> Path:
    return CORRECTION_DIR / f"{image_stem}_correction.json"


def _expected_divider_x(
    template: Any, column_order: list[str], index: int, table_bbox: list[int],
) -> float | None:
    """
    Pure arithmetic, zero image access - same formula locate_columns()
    itself uses (table_left + frac * table_width), applied to whichever
    of the two columns sharing this divider have a column_regions_approx
    entry. Averages both sides when both exist (they're nominally the
    same physical position; template fractions calibrated per-column can
    disagree slightly). Returns None if NEITHER side has approx data -
    matches locate_columns()'s own "no calibration data yet, don't guess"
    behavior for columns absent from column_regions_approx.
    """
    table_left, _, table_right, _ = table_bbox
    table_width = table_right - table_left
    left_name, right_name = column_order[index], column_order[index + 1]
    left_approx = template.column_regions_approx.get(left_name)
    right_approx = template.column_regions_approx.get(right_name)

    candidates = []
    if left_approx:
        candidates.append(table_left + left_approx["x_frac"][1] * table_width)
    if right_approx:
        candidates.append(table_left + right_approx["x_frac"][0] * table_width)
    if not candidates:
        return None
    return sum(candidates) / len(candidates)


def build_correction_record(
    sidecar: dict, sidecar_path, doc_type: str, column_order: list[str],
    columns_file: str | None = None,
) -> dict:
    """
    Seeds a fresh correction record from a loaded sidecar. For each
    adjacent column pair in column_order, reconciles the two
    independently-stored mask_keep_ranges into one shared divider:
    auto_x = the mean of left_column's own right edge and right_column's
    own left edge when BOTH exist (real disagreement between them is
    preserved as auto_disagreement_px, not silently discarded), or
    whichever single side exists, or None if neither column has been
    auto-detected yet. human_x starts equal to auto_x and provenance
    starts "auto_accepted_unreviewed" - nothing counts as reviewed until
    a caller explicitly touches it.

    Also seeds an empty row_number_anchor block (left_x/right_x, both
    None) - the human-confirmed x-centers of the printed VERTICAL row-
    number margin(s) (the "1, 2, 3..." numbering down the page edge many
    census forms print on the left, and sometimes also the right - see
    DocumentTemplate.row_number_column_x_frac/row_number_column2_x_frac
    in core/document_templates.py). This is a DIFFERENT axis/purpose from
    header_number_anchor above (that's the HORIZONTAL row of printed
    COLUMN numbers near the header; this is the VERTICAL margin of
    printed ROW numbers) - confirmed real-world motivation (Jon,
    2026-08-07): the ~54px gap between a real table_bbox's left edge and
    its first column's actual mask start (see merge_column_corrections()'s
    "EDGE-COLUMN FIX" docstring) is exactly this left row-number margin,
    reserved for detect_row_number_centers()'s search window - dialing in
    its exact center improves that detector's blob-finding accuracy.

    Creates one divider per adjacent column_order pair UNCONDITIONALLY -
    a column the sidecar never masked (extraction-oriented omission, not
    a detection failure - see this module's docstring, "IMPORTANT
    DISTINCTION") still gets a real divider entry on each of its sides,
    just with auto_x=None, rather than being silently dropped.
    """
    columns = sidecar.get("columns", {})
    dividers = []
    for i in range(len(column_order) - 1):
        left_name, right_name = column_order[i], column_order[i + 1]
        left_ranges = (columns.get(left_name) or {}).get("mask_keep_ranges") or []
        right_ranges = (columns.get(right_name) or {}).get("mask_keep_ranges") or []
        left_x1 = left_ranges[0][1] if left_ranges else None
        right_x0 = right_ranges[0][0] if right_ranges else None

        if left_x1 is not None and right_x0 is not None:
            auto_x = round((left_x1 + right_x0) / 2)
            auto_disagreement_px = abs(left_x1 - right_x0)
        elif left_x1 is not None:
            auto_x, auto_disagreement_px = left_x1, 0
        elif right_x0 is not None:
            auto_x, auto_disagreement_px = right_x0, 0
        else:
            auto_x, auto_disagreement_px = None, None

        dividers.append({
            "index": i,
            "left_column": left_name,
            "right_column": right_name,
            "auto_x": auto_x,
            "auto_disagreement_px": auto_disagreement_px,
            "human_x": auto_x,
            "projected_x": None,
            "delta_px": 0 if auto_x is not None else None,
            "provenance": PROVENANCE_UNREVIEWED,
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "source_sidecar_path": str(sidecar_path),
        "source_image_path": sidecar.get("source_image_path"),
        "deskewed_image_size": sidecar.get("deskewed_image_size"),
        "coordinate_space": sidecar.get("coordinate_space", "deskewed_image"),
        "doc_type": doc_type,
        "columns_file": columns_file,
        "column_order": list(column_order),
        "review_status": "in_progress",
        "table_bbox": sidecar.get("table_bbox"),
        "dividers": dividers,
        "header_number_anchor": {
            "band_y": None,
            "band_y_source": "manual",
            "detected_centers": None,
        },
        "row_number_anchor": {
            "left_x": None,
            "right_x": None,
        },
        # Added 2026-08-08, per Jon: the table's OUTER left/right edges
        # (unlike every interior divider above) have never been
        # click-adjustable in this UI - "Out of scope, always: table
        # bounds" per this tool's own original design. That's exactly
        # why the FIRST and LAST columns' own outer boundary has never
        # had real calibration data (see canada_census_1906.yaml's own
        # "No of family in order of visitation has NO entry" note) - no
        # divider can exist there, only the raw auto-detected table_bbox
        # edge, which often includes blank margin rather than the true
        # content boundary. table_edges tracks provenance for these two
        # specific values the SAME way an interior divider does
        # (auto_accepted_unreviewed / human_confirmed / human_placed),
        # but the position itself lives in table_bbox[0]/table_bbox[2]
        # directly (not a separate x field here) - moving an edge here
        # IS moving table_bbox, which merge_column_corrections() already
        # reads as its edge-column fallback, so no separate merge-time
        # handling was needed for this to take effect.
        #
        # "top" added 2026-08-09, per Jon: running generate_auto_sidecar()
        # standalone (a separate session, real production-scale testing)
        # found locate_table_boundary()'s own table_top detection wrong
        # on roughly half of a real batch - a Y-axis boundary, unlike
        # left/right which are X. table_bbox[1] is the position (index 1
        # = y0), same "moving the edge here IS moving table_bbox
        # directly" mechanism as left/right.
        #
        # "bottom" added 2026-08-09 (same day), per Jon: a human-marked
        # real table_bottom (table_bbox[3]/y1) lets row pitch be computed
        # exactly per-page instead of relying on one averaged template
        # constant - see the SCHEMA_VERSION=5 comment above for the full
        # motivation. Same mechanism as top/left/right: the position
        # lives in table_bbox[3] directly, not a separate field here.
        "table_edges": {
            "left": {"provenance": PROVENANCE_UNREVIEWED},
            "right": {"provenance": PROVENANCE_UNREVIEWED},
            "top": {"provenance": PROVENANCE_UNREVIEWED},
            "bottom": {"provenance": PROVENANCE_UNREVIEWED},
        },
        # Added 2026-08-09 (same day), per the SCHEMA_VERSION=6 comment
        # above - a real, human-measured row pitch, immune to the fixed_
        # periodic tiling circularity bug. None until "Measure row
        # pitch..." is used on this page. Stores the FULL measurement
        # provenance, not just the final fraction (per Jon: "that buys
        # you a lot later") - pixels/page_height let a future consumer
        # re-derive fraction without trusting a stale computation,
        # row_start/row_end/sample_count document exactly which rows and
        # how many gaps were averaged, and "provenance" leaves room for
        # a future non-human source (e.g. a trusted per-page detector)
        # without a schema change. core/template_refinement.py reads
        # "fraction".
        "measured_row_pitch": {
            "fraction": None,
            "pixels": None,
            "page_height": None,
            "row_start": None,
            "row_end": None,
            "y_start": None,
            "y_end": None,
            "sample_count": None,
            "provenance": None,
            "measured_at": None,
        },
        # Added 2026-08-09 (same day), per the SCHEMA_VERSION=7 comment
        # above - persists the manual deskew nudge (nudge buttons or
        # accepted rotation-refinement test) across page reloads.
        "manual_deskew_delta_deg": 0.0,
        "saved_at": None,
        "reviewer_notes": "",
    }


def load_or_build_correction_record(
    sidecar: dict, sidecar_path, correction_path, doc_type: str,
    column_order: list[str], columns_file: str | None = None,
) -> dict:
    """
    Resumes a partially-reviewed page if a correction record already
    exists on disk (preserving prior human edits), else builds fresh.
    If column_order has changed since the record was last saved (a
    column added to the form after this page's first review pass - the
    same real scenario init_column_state() already handles for
    sidecars), rebuilds the divider list fresh but carries forward
    human_x/provenance/delta_px for every divider whose (left_column,
    right_column) pair still exists. header_number_anchor and
    row_number_anchor are independent of column_order (they're page-level,
    not per-column) and are always carried forward verbatim across a
    rebuild, never reset.
    """
    correction_path = Path(correction_path)
    if not correction_path.exists():
        return build_correction_record(sidecar, sidecar_path, doc_type, column_order, columns_file)

    record = load_correction_record(correction_path)
    if record.get("column_order") == list(column_order):
        return record

    fresh = build_correction_record(sidecar, sidecar_path, doc_type, column_order, columns_file)
    fresh_by_pair = {(d["left_column"], d["right_column"]): d for d in fresh["dividers"]}
    for old in record.get("dividers", []):
        pair = (old.get("left_column"), old.get("right_column"))
        if pair in fresh_by_pair:
            fresh_by_pair[pair]["human_x"] = old.get("human_x")
            fresh_by_pair[pair]["delta_px"] = old.get("delta_px")
            fresh_by_pair[pair]["provenance"] = old.get("provenance", PROVENANCE_UNREVIEWED)
    fresh["header_number_anchor"] = record.get("header_number_anchor", fresh["header_number_anchor"])
    fresh["row_number_anchor"] = record.get("row_number_anchor", fresh["row_number_anchor"])
    # table_edges/table_bbox are page-level, not per-column, like the two
    # anchors above - a column_order change (this function's own trigger
    # for rebuilding dividers) has no bearing on whether the table's
    # outer edges were already human-placed, so both carry forward
    # verbatim, never reset.
    fresh["table_edges"] = record.get("table_edges", fresh["table_edges"])
    fresh["table_bbox"] = record.get("table_bbox", fresh["table_bbox"])
    # measured_row_pitch/manual_deskew_delta_deg (added 2026-08-09) are
    # ALSO page-level, not per-column, same reasoning as the two anchors
    # and table_edges/table_bbox above - a column_order change has no
    # bearing on either. REAL BUG this fixes (caught by Jon, 2026-08-09):
    # this rebuild branch was missing both when they were added, so any
    # time it fired (column_order changing between loads) it silently
    # reset a saved manual deskew nudge back to 0.0 in memory even though
    # the correct value was still sitting on disk - "nudge > save > next
    # image > nudge > save > previous image > preview reverts back to
    # unskewed image."
    fresh["measured_row_pitch"] = record.get("measured_row_pitch", fresh["measured_row_pitch"])
    fresh["manual_deskew_delta_deg"] = record.get("manual_deskew_delta_deg", fresh["manual_deskew_delta_deg"])
    return fresh


def save_correction_record(record: dict, path) -> None:
    """
    Atomic (temp file + os.replace), same idiom as row_segmentation.
    save_sidecar - written inline rather than imported since that
    function is framed as sidecar-specific I/O and this is a distinct
    artifact type. Stamps saved_at (UTC, seconds precision) on every
    save as a plain audit trail; not used by any merge/comparison logic.
    """
    record = dict(record)
    record["saved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    path = str(path)
    dir_ = os.path.dirname(path) or "."
    os.makedirs(dir_, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_, prefix=".correction_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def load_correction_record(path) -> dict:
    """
    Migrates a schema_version 1 record (predates row_number_anchor,
    2026-08-07) in place on load by adding the field with defaults -
    idempotent, same "pick up new structure without discarding existing
    work" discipline row_segmentation.init_column_state() already uses
    for sidecars. No real schema_version 1 records existed in production
    at the time this was added (only build-time test artifacts), but the
    migration costs nothing to keep for genuine forward compatibility.
    """
    with open(path, "r", encoding="utf-8") as f:
        record = json.load(f)
    record.setdefault("row_number_anchor", {"left_x": None, "right_x": None})
    record.setdefault("table_edges", {})
    # "top" added 2026-08-09 - a record saved between the left/right
    # table_edges feature (2026-08-08) and this one already has a real
    # table_edges dict, so the whole-key setdefault() above is a no-op
    # for it; this second setdefault reaches inside to add just the new
    # sub-key, same "add what's missing, keep what's there" idempotent
    # migration discipline as everything else in this function.
    record["table_edges"].setdefault("left", {"provenance": PROVENANCE_UNREVIEWED})
    record["table_edges"].setdefault("right", {"provenance": PROVENANCE_UNREVIEWED})
    record["table_edges"].setdefault("top", {"provenance": PROVENANCE_UNREVIEWED})
    # "bottom" added 2026-08-09 (same day) - same idempotent pattern.
    record["table_edges"].setdefault("bottom", {"provenance": PROVENANCE_UNREVIEWED})
    # measured_row_pitch added 2026-08-09 (same day) - same idempotent
    # pattern, see SCHEMA_VERSION=6 comment above.
    record.setdefault("measured_row_pitch", {
        "fraction": None, "pixels": None, "page_height": None,
        "row_start": None, "row_end": None, "y_start": None, "y_end": None,
        "sample_count": None, "provenance": None, "measured_at": None,
    })
    # manual_deskew_delta_deg added 2026-08-09 (same day) - same
    # idempotent pattern, see SCHEMA_VERSION=7 comment above.
    record.setdefault("manual_deskew_delta_deg", 0.0)
    return record


def apply_divider_move(record: dict, index: int, new_x: int) -> None:
    """
    In-memory only - caller decides when to persist via
    save_correction_record(). Sets human_x, recomputes delta_px against
    the ORIGINAL auto_x (never against a prior human_x, so delta_px
    always answers "how far is the human's final answer from what the
    detector proposed", not "how far did the last drag move it").
    provenance becomes human_confirmed if an auto value existed for this
    divider, human_placed if it didn't (no auto proposal to compare
    against - a genuinely new boundary the reviewer is placing from
    scratch).
    """
    d = record["dividers"][index]
    d["human_x"] = int(round(new_x))
    d["delta_px"] = (d["human_x"] - d["auto_x"]) if d["auto_x"] is not None else None
    d["provenance"] = PROVENANCE_CONFIRMED if d["auto_x"] is not None else PROVENANCE_PLACED


def confirm_divider_as_is(record: dict, index: int) -> None:
    """
    Marks a divider reviewed WITHOUT moving it - needed so a correct
    auto proposal can be explicitly recorded as confirmed rather than
    silently left "auto_accepted_unreviewed". That distinction matters
    for the residual-error analysis this tool exists to support: an
    untouched divider contributes no signal about the detector's
    accuracy, not a zero-offset data point, and the two must stay
    distinguishable in the saved record.
    """
    d = record["dividers"][index]
    if d["auto_x"] is None:
        raise ValueError(
            f"Divider {index} has no auto value to confirm - "
            "use apply_divider_move() to place it instead."
        )
    d["human_x"] = d["auto_x"]
    d["delta_px"] = 0
    d["provenance"] = PROVENANCE_CONFIRMED


def clear_divider(record: dict, index: int) -> None:
    """Reverts a divider to its auto value (or to fully undefined, if it
    never had one) and provenance back to auto_accepted_unreviewed."""
    d = record["dividers"][index]
    d["human_x"] = d["auto_x"]
    d["delta_px"] = 0 if d["auto_x"] is not None else None
    d["projected_x"] = None
    d["provenance"] = PROVENANCE_UNREVIEWED


_TABLE_EDGE_INDEX = {"left": 0, "top": 1, "right": 2, "bottom": 3}  # table_bbox = [x0, y0, x1, y1]


def apply_table_edge_move(record: dict, side: str, new_value: int) -> None:
    """
    Moves one of the table's own outer edges (side is "left"/"right" - X
    axis, table_bbox[0]/[2] - or "top"/"bottom" - Y axis, table_bbox[1]/
    [3], "top" added 2026-08-09 per Jon: a separate session running
    generate_auto_sidecar() standalone found table_top wrong on roughly
    half a real batch; "bottom" added the same day so real per-page row
    pitch can eventually be computed exactly instead of relying on one
    averaged template constant) -
    unlike apply_divider_move(), there's no separate auto_x to compare
    against here (table_bbox's initial value IS the "auto" detection,
    but this module never stored a second, protected copy of it - the
    original auto-sidecar's table_bbox is only ever a page away, in the
    sidecar file itself), so provenance simply becomes human_placed on
    any real move - this module has no way to distinguish "confirmed the
    auto position exactly" from "moved it and it happened to land back
    on the same pixel," and doesn't need to: the record's own reviewer_
    notes/saved_at already show a human touched this page.
    """
    idx = _TABLE_EDGE_INDEX[side]
    record["table_bbox"][idx] = int(round(new_value))
    record["table_edges"][side]["provenance"] = PROVENANCE_PLACED


def confirm_table_edge_as_is(record: dict, side: str) -> None:
    """Marks a table edge reviewed WITHOUT moving it - same "explicitly
    confirmed, not just silently defaulted" distinction confirm_divider_
    as_is() makes for interior dividers."""
    record["table_edges"][side]["provenance"] = PROVENANCE_CONFIRMED


def clear_table_edge(record: dict, side: str) -> None:
    """
    Reverts a table edge's provenance back to auto_accepted_unreviewed
    (2026-08-09, per Jon - a real damaged/shifted-scan case: "one sample
    i had to guess at edge as the page was damaged... the table is
    shifted to the right from edge of actual image" - that guess showed
    up as the single biggest outlier in core/template_refinement.py's
    LEFT-edge spread across an otherwise-tight 8-sample 1906 batch).
    Mirrors clear_divider()'s exact intent for interior dividers, at
    table-edge granularity: leaves table_bbox's actual position
    UNTOUCHED (a damaged page's own extraction still needs a real left
    edge, even a guessed one) - this only stops that specific edge from
    counting as human-reviewed calibration data, so refine_template_
    from_calibration()'s "only human-touched" filter naturally excludes
    it from the template average without deleting the guess itself.
    """
    record["table_edges"][side]["provenance"] = PROVENANCE_UNREVIEWED


def apply_left_crop(record: dict, crop_px: int) -> None:
    """
    Shifts EVERY stored X-coordinate in the record left by crop_px, to
    match a physical pixel crop of the same amount applied to the
    working image's left edge (2026-08-09, per Jon's "crop_left_of_
    marker" design - see ui/column_calibration_ui.py's "Crop left
    edge..." tool for the interactive side of this). Unlike a rotation
    bake (expand=False keeps the same coordinate frame, no shift
    needed - see save_page()'s deskew-bake comment), a CROP genuinely
    shrinks the image and moves the origin, so every X position placed
    against the OLD, wider image would silently be wrong on the new,
    narrower one unless corrected here.

    Y-coordinates are untouched - this is a LEFT-edge-only horizontal
    crop, never a vertical one. Covers every X-bearing field this
    module's schema has: divider human_x/auto_x/projected_x, table_bbox
    x0/x1 (index 0/2), row_number_anchor left_x/right_x, and header_
    number_anchor's detected_centers (each a (center_x, width_px) pair
    - only the center shifts, width is unaffected by a translation).
    deskewed_image_size[0] (width) is reduced by crop_px to match.

    Deliberately does NOT touch self.sidecar (the original auto-
    detected sidecar this correction record was built from) - that
    object is expected to become stale and get replaced wholesale by a
    fresh "Regenerate sidecars" pass against the now-cropped working
    image, not patched in place; only the correction record (the
    thing that survives regeneration) needs to stay valid.

    In-memory only, same "caller decides when to persist" contract as
    apply_divider_move() - caller is expected to call this in the same
    action that performs the physical image crop, so the two never
    drift out of sync.
    """
    for d in record.get("dividers", []):
        for key in ("human_x", "auto_x", "projected_x"):
            if d.get(key) is not None:
                d[key] = d[key] - crop_px

    table_bbox = record.get("table_bbox")
    if table_bbox:
        table_bbox[0] -= crop_px
        table_bbox[2] -= crop_px

    row_number = record.get("row_number_anchor", {})
    if row_number.get("left_x") is not None:
        row_number["left_x"] -= crop_px
    if row_number.get("right_x") is not None:
        row_number["right_x"] -= crop_px

    header_anchor = record.get("header_number_anchor", {})
    centers = header_anchor.get("detected_centers")
    if centers:
        header_anchor["detected_centers"] = [[cx - crop_px, cw] for cx, cw in centers]

    size = record.get("deskewed_image_size")
    if size:
        record["deskewed_image_size"] = [size[0] - crop_px, size[1]]


def apply_top_crop(record: dict, crop_px: int) -> None:
    """
    Shifts EVERY stored Y-coordinate in the record up by crop_px, to
    match a physical pixel crop of the same amount applied to the
    working image's top edge (2026-08-09, per Jon - a real, near-
    universal damaged/torn top border found across the 1906 batch,
    with one genuine outlier (e001211831_L/_R) where the damage extends
    far deeper than the ~100-110px norm - see ui/column_calibration_
    ui.py's "Crop top edge..." tool for the interactive side, and the
    real measured gap data (table_top - blob_y, 661-701px across 14
    confirmed samples) that validated a safe offset). Same "crop from
    the already-calibrated table edge, not from the artifact itself"
    principle as apply_left_crop() - per Jon: "it keeps every image
    uniform then... the layout should snap right over where to search."

    X-coordinates are untouched - this is a TOP-edge-only vertical
    crop, never a horizontal one. Covers every Y-bearing field this
    module's schema has: table_bbox y0/y1 (index 1/3), header_number_
    anchor.band_y, and measured_row_pitch.y_start/y_end (the 2 raw
    click positions "Measure row pitch..." recorded - a real gap this
    function closes: a measurement taken BEFORE a top crop would
    silently read wrong against the AFTER-crop image otherwise, since
    core.template_refinement.refine_template_from_calibration() trusts
    this field directly, never re-deriving it from pixels).
    deskewed_image_size[1] (height) is reduced by crop_px to match.
    Divider human_x/auto_x/projected_x and row_number_anchor left_x/
    right_x are X-only, unaffected by a vertical crop.

    measured_row_pitch.fraction/page_height are ALSO recomputed here,
    not just y_start/y_end (real bug found 2026-08-09, via Jon: "the
    rows are being squished and not spanning the whole table" - a
    directly visible symptom, confirmed against real data: every
    measured sample's stored "fraction" was still computed against the
    PRE-crop page height, understating the true post-crop fraction by
    ~7-9% across the board, since "pixels" (the raw row1-to-rowN pitch)
    is translation-invariant under a uniform Y-shift and correctly
    stayed the same, but "fraction" = pixels/page_height still needs
    the NEW, shorter page_height as its denominator - a stale
    denominator alone was enough to silently corrupt every downstream
    consumer of this field, including core.template_refinement.py's
    own aggregate template value). "pixels" itself is left untouched
    (still correct, never needed recomputing).

    Same "in-memory only, sidecar left stale for a future Regenerate
    pass" contract as apply_left_crop() - see that function's own
    docstring for the full reasoning.
    """
    table_bbox = record.get("table_bbox")
    if table_bbox:
        table_bbox[1] -= crop_px
        table_bbox[3] -= crop_px

    header_anchor = record.get("header_number_anchor", {})
    if header_anchor.get("band_y") is not None:
        header_anchor["band_y"] -= crop_px

    size = record.get("deskewed_image_size")
    new_height = (size[1] - crop_px) if size else None

    pitch = record.get("measured_row_pitch", {})
    if pitch.get("y_start") is not None:
        pitch["y_start"] -= crop_px
    if pitch.get("y_end") is not None:
        pitch["y_end"] -= crop_px
    if pitch.get("pixels") is not None and new_height:
        pitch["page_height"] = new_height
        pitch["fraction"] = pitch["pixels"] / new_height

    if size:
        record["deskewed_image_size"] = [size[0], new_height]


def merge_column_corrections(sidecar: dict, correction_record: dict) -> dict:
    """
    Returns a NEW dict (json round-trip deep copy - sidecars are plain
    JSON-safe data already, same assumption build_sidecar()/save_sidecar()
    make) with mask_keep_ranges overwritten ONLY for columns where at
    least one bounding divider has been reviewed (provenance != auto_
    accepted_unreviewed). A column whose both bounding dividers remain
    untouched is left byte-identical to the input sidecar. Never touches
    header_bbox, metadata_bbox, rows, dropped_bands, warnings,
    preprocessing, progress, active_column, or any other column's
    status/results/extraction_meta. table_bbox is the ONE exception,
    added 2026-08-08 (see TABLE_BBOX SYNC below) - x0/x1 update only
    when the corresponding table edge was actually human-placed via
    ui/column_calibration_ui.py's "Mark table left/right edges"; y0/y1
    (vertical bounds) are never touched, same as everything else here.

    Does NOT call save_sidecar() - returns the merged dict only. Per the
    hard requirement this tool was built around, the original sidecar
    file is never overwritten; a caller that wants a persisted merged
    copy must explicitly choose a destination path.

    Deliberately NOT built as an extension of row_segmentation.
    update_sidecar(): that function loads/writes a single column from/to
    a FILE PATH, recomputes progress/active_column (extraction-workflow
    concepts this tool has no business touching), and does a general
    recursive dict merge - none of which fits a targeted, in-memory,
    mask_keep_ranges-only overwrite across potentially many columns at
    once.

    COLUMN_ORDER SYNC (2026-08-07, found via real-data testing against a
    real 1906 census page routed to the wrong template by the CV
    classifier - ui/column_calibration_ui.py's "Load columns file..."
    let a human load the CORRECT column list, but this function was
    still leaving the sidecar's OWN top-level column_order untouched,
    still naming the 5 wrong columns from the original bad template
    guess, while `columns` gained real mask data under 18 different
    (correct) names - a downstream consumer iterating column_order would
    see only the wrong 5 and miss every real calibrated column entirely.
    column_order didn't used to need syncing because there was no way
    for it to change during a merge before "Load columns file..."
    existed - this is a new interaction, not a pre-existing edge case
    that was missed. Now: whenever correction_record's column_order
    differs from the sidecar's own, the merged sidecar's column_order,
    active_column, and progress are all replaced to match the
    correction record (active_column reset to the new list's first
    entry, progress recomputed from the new column_order's own
    "status"=="done" counts - the OLD active_column/progress values are
    guaranteed stale/meaningless once the column set itself changed, not
    just approximately outdated), and every `columns` entry whose name
    is NOT in the new column_order is dropped (stale leftovers from the
    wrong template - keeping them around as dead, unreferenced dict
    entries would just be confusing, not merely harmless, since a human
    re-opening the merged file later has no way to tell they're inert).

    EDGE-COLUMN FIX (2026-08-07, found via real-data testing - a
    synthetic test with table_bbox coinciding with the first/last
    column's actual mask edge didn't exercise this): the first column
    has no LEFT divider (index -1 doesn't exist - the outer table edge
    is out of scope, per requirement #1), and the last column has no
    RIGHT divider, by design. The first version of this function fell
    back to table_bbox's raw edge for that missing side whenever the
    column's OTHER (interior) divider was touched - wrong whenever the
    column's real mask doesn't start exactly at table_bbox (confirmed on
    a real sidecar: "Dwelling House" is masked [246, 321] while
    table_bbox starts at 192 - a real ~54px margin, almost certainly a
    printed row-number column never allocated to any named column).
    Touching only the interior divider must never silently redefine the
    untouched outer edge. Fixed to fall back to the column's OWN
    existing mask_keep_ranges edge first, and only to table_bbox as a
    last resort for a column that had no mask at all yet.
    """
    merged = json.loads(json.dumps(sidecar))
    column_order = correction_record["column_order"]
    table_bbox = correction_record["table_bbox"]
    dividers = correction_record["dividers"]
    table_edges = correction_record.get("table_edges", {})
    left_edge_touched = table_edges.get("left", {}).get("provenance") != PROVENANCE_UNREVIEWED
    right_edge_touched = table_edges.get("right", {}).get("provenance") != PROVENANCE_UNREVIEWED
    top_edge_touched = table_edges.get("top", {}).get("provenance") != PROVENANCE_UNREVIEWED
    bottom_edge_touched = table_edges.get("bottom", {}).get("provenance") != PROVENANCE_UNREVIEWED

    def divider_at(idx: int) -> dict | None:
        return dividers[idx] if 0 <= idx < len(dividers) else None

    def original_edge(name: str, side: int) -> int | None:
        """side: 0 for the column's own existing left edge, 1 for right."""
        ranges = (merged.get("columns", {}).get(name) or {}).get("mask_keep_ranges") or []
        return ranges[0][side] if ranges else None

    for i, name in enumerate(column_order):
        left_divider = divider_at(i - 1)
        right_divider = divider_at(i)
        is_first_column = (i == 0)
        is_last_column = (i == len(column_order) - 1)
        # TABLE-EDGE AWARENESS (2026-08-08, added alongside ui/column_
        # calibration_ui.py's "Mark table left/right edges" - see
        # core/column_calibration.py's table_edges docstring): the FIRST
        # column has no left divider and the LAST has no right divider
        # by design (the outer table edge, out of scope for a divider to
        # exist at) - previously the only way to touch either boundary
        # was to already have SOME existing mask (original_edge()), so a
        # column whose ONLY change was a human-placed table edge was
        # silently skipped by the `touched` check below. Folding the two
        # edge flags into that check (only for the specific column each
        # applies to) fixes that, and a human-placed edge takes priority
        # over a stale original_edge() mask reading below - same "the
        # real correction wins" principle every divider already follows.
        touched = any(
            d is not None and d["provenance"] != PROVENANCE_UNREVIEWED
            for d in (left_divider, right_divider)
        ) or (is_first_column and left_edge_touched) or (is_last_column and right_edge_touched)
        if not touched:
            continue

        if left_divider is not None:
            new_x0 = left_divider["human_x"]
        elif is_first_column and left_edge_touched:
            new_x0 = table_bbox[0]
        else:
            new_x0 = original_edge(name, 0)
            if new_x0 is None:
                new_x0 = table_bbox[0]

        if right_divider is not None:
            new_x1 = right_divider["human_x"]
        elif is_last_column and right_edge_touched:
            new_x1 = table_bbox[2]
        else:
            new_x1 = original_edge(name, 1)
            if new_x1 is None:
                new_x1 = table_bbox[2]

        if new_x0 is None or new_x1 is None:
            # A bounding divider was touched (e.g. cleared/deleted) but
            # never actually placed - nothing usable to merge for this
            # column yet, leave it as the sidecar already had it.
            continue

        merged.setdefault("columns", {}).setdefault(name, {})
        merged["columns"][name]["mask_keep_ranges"] = [[new_x0, new_x1]]

    # TABLE_BBOX SYNC (2026-08-08, extended 2026-08-09 for "top", then
    # "bottom" same day) - the merged sidecar's own top-level table_bbox
    # IS now updated, but ONLY for whichever edge was actually human-
    # touched (x0 if left_edge_touched, x1 if right_edge_touched, y0 if
    # top_edge_touched, y1 if bottom_edge_touched). Before table_edges
    # existed, this function correctly never touched table_bbox at all
    # (nothing here could produce a real correction to it) - now that a
    # human CAN place a real edge, leaving the merged sidecar's own
    # table_bbox stale would silently discard that correction for any
    # downstream consumer reading table_bbox directly instead of
    # re-deriving it from column masks.
    if left_edge_touched or right_edge_touched or top_edge_touched or bottom_edge_touched:
        merged_table_bbox = list(merged.get("table_bbox") or table_bbox)
        if left_edge_touched:
            merged_table_bbox[0] = table_bbox[0]
        if right_edge_touched:
            merged_table_bbox[2] = table_bbox[2]
        if top_edge_touched:
            merged_table_bbox[1] = table_bbox[1]
        if bottom_edge_touched:
            merged_table_bbox[3] = table_bbox[3]
        merged["table_bbox"] = merged_table_bbox

    # COLUMN_ORDER SYNC - see this function's own docstring above. Only
    # touches column_order/active_column/progress/columns-pruning at all
    # when it actually changed, so the common case (column_order was
    # never overridden, same list as when the sidecar was generated)
    # produces a byte-identical column_order to before, matching this
    # function's existing "leave everything else untouched" contract.
    if merged.get("column_order") != list(column_order):
        merged["column_order"] = list(column_order)
        merged["active_column"] = column_order[0] if column_order else None
        merged["columns"] = {
            name: col for name, col in merged.get("columns", {}).items()
            if name in column_order
        }
        completed = sum(
            1 for name in column_order
            if (merged["columns"].get(name) or {}).get("status") == "done"
        )
        merged["progress"] = {"completed": completed, "total": len(column_order)}

    return merged


def project_remaining_dividers(record: dict, template: Any) -> str:
    """
    "Project from confirmed anchors" - reuses ONLY the pure-arithmetic
    "predict from a page-level affine" step of locate_columns()'s
    resolution cascade (core.auto_sidecar._resolve_edge's
    affine_predicted branch's math), never its CV-based tiers (measured/
    affine_override, which require a fresh ruling-line search this tool
    has no business running). Needs zero image access: builds (expected,
    human) position pairs from every divider already reviewed this
    session (human_confirmed or human_placed), fits _fit_registration_
    affine (imported directly from core.auto_sidecar - the same function
    locate_columns() itself uses), and if that succeeds, writes a
    projected_x value onto every NOT-yet-reviewed divider.

    Mutates `record` in place but only ever touches projected_x, never
    human_x/provenance/delta_px - a projection is an advisory overlay,
    not an automatic acceptance. Promoting one to a real correction
    still requires an explicit apply_divider_move() call, exactly like
    any other proposed value.

    Returns a short human-readable status string for the UI to display.
    Never raises on insufficient data - _fit_registration_affine's own
    "None on too few/too clustered points" contract is passed straight
    through as an honest status message, not a guess.
    """
    from core.auto_sidecar import _fit_registration_affine, _MIN_AFFINE_FIT_SPAN_FRAC

    column_order = record["column_order"]
    table_bbox = record["table_bbox"]
    table_width = table_bbox[2] - table_bbox[0]
    dividers = record["dividers"]

    confident_pairs = []
    for d in dividers:
        if d["provenance"] in (PROVENANCE_CONFIRMED, PROVENANCE_PLACED) and d["human_x"] is not None:
            expected = _expected_divider_x(template, column_order, d["index"], table_bbox)
            if expected is not None:
                confident_pairs.append((expected, float(d["human_x"])))

    affine = _fit_registration_affine(confident_pairs, table_width * _MIN_AFFINE_FIT_SPAN_FRAC)
    if affine is None:
        return "Not enough confirmed anchors yet to project remaining boundaries."

    a, b = affine
    projected_count = 0
    for d in dividers:
        if d["provenance"] in (PROVENANCE_CONFIRMED, PROVENANCE_PLACED):
            continue  # already reviewed - nothing to project
        expected = _expected_divider_x(template, column_order, d["index"], table_bbox)
        if expected is None:
            d["projected_x"] = None
            continue
        d["projected_x"] = int(round(a * expected + b))
        projected_count += 1

    return f"Projected {projected_count} divider(s) from {len(confident_pairs)} confirmed anchor(s)."


def try_load_anchor_cache(image_stem: str) -> list[tuple[float, int]] | None:
    """
    Pure Path.exists() + json.load() - NEVER runs detection itself. This
    module and its UI must not call detect_column_number_centers() or
    any variant unconditionally; if data/outputs/column_number_anchors/
    {stem}_anchors.json exists (produced by some future, separate
    offline script - out of scope here), returns its detected_centers as
    [(center_x, width_px), ...]; otherwise returns None and the UI falls
    back to manual anchor-line placement.
    """
    path = ANCHOR_CACHE_DIR / f"{image_stem}_anchors.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    centers = data.get("detected_centers")
    if not centers:
        return None
    return [(float(c), int(w)) for c, w in centers]
