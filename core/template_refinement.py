"""
Computes suggested config/document_templates/<doc_type>.yaml numeric
values from real, human-reviewed column_calibration correction records
- built 2026-08-09, per Jon's explicit spec: "I'd build it as a shared
refine_template_from_calibration() function that the UI calls."

For a given doc_type, this scans every correction record
(core/column_calibration.py's CORRECTION_DIR), keeps only ones with
review_status == "done" for that doc_type, and averages:

- each table edge (left/right/top/bottom) as a PAGE-relative fraction,
  but ONLY from samples where that specific edge was actually human-
  touched (provenance != auto_accepted_unreviewed) - an untouched edge
  is still just the raw CV guess, not real calibration data;
- each interior divider as a TABLE-relative x_frac, same human-touched-
  only filter;
- expected_row_height_frac from each sample's OWN measured_row_pitch
  field (ui/column_calibration_ui.py's "Measure row pitch..." 2-click
  tool - real row-TOP-to-row-TOP pitch, never padded row bbox height).

ROW PITCH SOURCE, real history worth knowing (2026-08-09): this field
used to be computed from each sample's regenerated sidecar rows
directly. That broke twice in one session - first from stale pre-fix
sidecars (canada_census_1921.yaml's original 0.01472 value was built
from sidecars generated hours before a same-day pitch fix landed), then
from a deeper CIRCULARITY bug even after "fixing" staleness by
regenerating: once a template's expected_row_height_frac is set, core/
auto_sidecar.py's fixed_periodic tiling (detect_data_rows()) skips its
own per-page row-1 search entirely and just multiplies the template's
own value by page height for every row - so a regenerated sidecar's row
bboxes can no longer supply a measurement INDEPENDENT of the template
being validated; they just echo it back at zero spread regardless of
whether the value is right. measured_row_pitch is immune to both
problems (it comes from 2 direct human clicks on the real image, never
from tiling), so it's now the only source this function trusts for row
pitch - specifically its "fraction" sub-field (the record also stores
pixels/page_height/row_start/row_end/y_start/y_end/sample_count/
provenance for full traceability, per Jon: "that buys you a lot
later" - not read here, but available to any future consumer that
needs to re-derive or audit a measurement without trusting this
function's own arithmetic). A sample with no measured_row_pitch simply
contributes nothing to this field - it still contributes normally to
the divider/edge stats above, which come from the correction record's
own placed geometry and were never affected by either bug.

PREVIEW ONLY as of this writing. Deliberately does not write any YAML -
that's a later "Apply to template" step, once this preview has been
used and trusted across more than one document year (per Jon's own
staged plan: "Regenerate sidecars -> mark geometry -> mark Done ->
Preview refinement -> inspect spreads -> manually/apply update").
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from core.column_calibration import CORRECTION_DIR, PROVENANCE_UNREVIEWED, load_correction_record

# Below this many reviewed samples, a field is flagged REVIEW even if
# its spread looks tight - a tight spread across 2 samples isn't
# meaningfully different from luck yet (same reasoning already applied
# by hand throughout this session's real template rebuilds).
MIN_SAMPLES_GOOD = 4

# Column-divider/table-edge fractions are relative to table/page span
# (0..1). A real, already-documented spread exists in production data
# (1906's own divider spread was ~0.05 across 6 pages, genuine page-to-
# page registration noise, not a bug) - this threshold just flags an
# outlier worth a human glance. Advisory only; nothing here blocks
# anything (see this module's PREVIEW ONLY docstring note above).
FRAC_SPREAD_REVIEW_THRESHOLD = 0.01

# Row pitch is a much smaller absolute number than a position fraction,
# so judged as RELATIVE spread ((max-min)/mean) instead of absolute -
# 5%, chosen against the real per-page pitch variance already measured
# across the 1921 6-sample batch (0.01180-0.01374, i.e. real page-to-
# page registration noise on this exact field). Advisory only.
ROW_PITCH_RELATIVE_SPREAD_REVIEW_THRESHOLD = 0.05


@dataclass
class FieldStat:
    name: str
    mean: float | None
    spread: float | None  # max - min
    n: int
    min: float | None = None
    max: float | None = None
    status: str = "INSUFFICIENT"  # GOOD / REVIEW / INSUFFICIENT
    note: str = ""


@dataclass
class RefinementResult:
    doc_type: str
    samples_accepted: list[str]
    samples_rejected: list[tuple[str, str]]  # (stem, reason)
    table_edges: dict[str, FieldStat]  # "left"/"right"/"top"/"bottom"
    columns: dict[str, FieldStat]  # "<index>: <left col> | <right col>" -> stat
    row_pitch: FieldStat
    warnings: list[str] = field(default_factory=list)


def _stat_from_values(name: str, values: list[float], relative: bool = False) -> FieldStat:
    n = len(values)
    if n == 0:
        return FieldStat(name=name, mean=None, spread=None, n=0, status="INSUFFICIENT",
                          note="no reviewed samples")
    lo, hi = min(values), max(values)
    mean = sum(values) / n
    spread = hi - lo
    threshold = (ROW_PITCH_RELATIVE_SPREAD_REVIEW_THRESHOLD * mean) if relative else FRAC_SPREAD_REVIEW_THRESHOLD
    if n < MIN_SAMPLES_GOOD:
        status, note = "REVIEW", f"only {n} sample(s), below the {MIN_SAMPLES_GOOD}-sample floor"
    elif spread > threshold:
        status, note = "REVIEW", f"spread {spread:.5f} exceeds the review threshold"
    else:
        status, note = "GOOD", ""
    return FieldStat(name=name, mean=mean, spread=spread, n=n, min=lo, max=hi, status=status, note=note)


def refine_template_from_calibration(doc_type: str, correction_dir: Path = CORRECTION_DIR) -> RefinementResult:
    """
    Scans correction_dir for every *_correction.json matching doc_type
    with review_status == "done" and returns the averaged geometry -
    see this module's own docstring for the full field-by-field
    methodology.
    """
    samples_accepted: list[str] = []
    samples_rejected: list[tuple[str, str]] = []
    warnings: list[str] = []

    divider_values: dict[str, list[float]] = {}
    edge_values: dict[str, list[float]] = {"left": [], "right": [], "top": [], "bottom": []}
    pitch_values: list[float] = []

    for path in sorted(correction_dir.glob("*_correction.json")):
        stem = path.stem[: -len("_correction")]
        try:
            # load_correction_record(), not a raw json.loads() - it
            # applies the same schema migration (setdefault on missing
            # table_edges sub-keys etc.) the UI itself always sees. A
            # raw json.loads() on a record saved before "top"/"bottom"
            # existed would read a MISSING table_edges.bottom key as
            # {}.get("provenance") -> None, which is != PROVENANCE_
            # UNREVIEWED - i.e. a field nobody ever touched would look
            # "touched" (real bug caught 2026-08-09 via this exact
            # function's own first real test run against production
            # data: TABLE BOTTOM showed a fake GOOD/n=6 despite no
            # sample ever having "Mark table bottom" clicked).
            record = load_correction_record(path)
        except (OSError, json.JSONDecodeError) as e:
            samples_rejected.append((stem, f"unreadable correction record: {e}"))
            continue

        if record.get("doc_type") != doc_type:
            continue  # not part of this doc_type's population - not an error
        if record.get("review_status") != "done":
            samples_rejected.append((stem, f"review_status={record.get('review_status')!r}, not 'done'"))
            continue

        table_bbox = record.get("table_bbox")
        page_size = record.get("deskewed_image_size")
        if not table_bbox or not page_size:
            samples_rejected.append((stem, "missing table_bbox or deskewed_image_size"))
            continue
        x0, y0, x1, y1 = table_bbox
        table_w = x1 - x0
        page_w, page_h = page_size

        # DIVIDERS - table-relative x_frac, only human-reviewed ones.
        column_order = record.get("column_order", [])
        for d in record.get("dividers", []):
            if d.get("provenance") == PROVENANCE_UNREVIEWED or d.get("human_x") is None or not table_w:
                continue
            idx = d["index"]
            label = (f"{idx}: {column_order[idx]} | {column_order[idx + 1]}"
                     if 0 <= idx < len(column_order) - 1 else f"divider[{idx}]")
            divider_values.setdefault(label, []).append((d["human_x"] - x0) / table_w)

        # TABLE EDGES - page-relative fraction, only human-touched ones.
        table_edges = record.get("table_edges", {})
        edge_positions = {"left": (x0, page_w), "right": (x1, page_w), "top": (y0, page_h), "bottom": (y1, page_h)}
        for side, (pos, span) in edge_positions.items():
            touched = table_edges.get(side, {}).get("provenance") != PROVENANCE_UNREVIEWED
            if touched and span:
                edge_values[side].append(pos / span)

        # ROW PITCH - real, human-measured pitch from ui/column_
        # calibration_ui.py's "Measure row pitch..." 2-click tool (see
        # module docstring's ROW PITCH SOURCE note for why sidecar rows
        # are no longer used here at all). Rejection here doesn't
        # disqualify the sample from the divider/edge stats above - only
        # row pitch is affected.
        measured = record.get("measured_row_pitch") or {}
        pitch_frac = measured.get("fraction")
        if pitch_frac is None:
            warnings.append(f"{stem}: no measured_row_pitch on record - excluded from row pitch. "
                             f"Use \"Measure row pitch...\" on this page.")
        else:
            pitch_values.append(pitch_frac)

        samples_accepted.append(stem)

    columns = {label: _stat_from_values(label, values) for label, values in divider_values.items()}
    table_edges_stats = {side: _stat_from_values(side, values) for side, values in edge_values.items()}
    row_pitch_stat = _stat_from_values("row_pitch", pitch_values, relative=True)

    return RefinementResult(
        doc_type=doc_type,
        samples_accepted=samples_accepted,
        samples_rejected=samples_rejected,
        table_edges=table_edges_stats,
        columns=columns,
        row_pitch=row_pitch_stat,
        warnings=warnings,
    )


def format_refinement_report(result: RefinementResult) -> str:
    """Plain-text report matching Jon's own sketched preview format -
    used by ui/column_calibration_ui.py's preview dialog, kept here
    (not UI-only) so a future CLI script can reuse it verbatim."""
    lines = [result.doc_type, f"Samples accepted: {len(result.samples_accepted)}"]
    if result.samples_rejected:
        lines.append(f"Samples rejected: {len(result.samples_rejected)}")
        for stem, reason in result.samples_rejected:
            lines.append(f"  - {stem}: {reason}")
    lines.append("")

    def _block(title: str, stat: FieldStat) -> list[str]:
        block = [title]
        if stat.mean is None:
            block.append(f"status: {stat.status}  ({stat.note})")
        else:
            block.append(f"mean:   {stat.mean:.5f}")
            block.append(f"spread: {stat.spread:.5f}")
            block.append(f"n:      {stat.n}")
            block.append(f"status: {stat.status}" + (f"  ({stat.note})" if stat.note else ""))
        block.append("")
        return block

    for side in ("left", "right", "top", "bottom"):
        lines += _block(f"TABLE {side.upper()}", result.table_edges[side])

    for label, stat in result.columns.items():
        lines += _block(f"COLUMN {label}", stat)

    lines += _block("ROW PITCH", result.row_pitch)

    if result.warnings:
        lines.append("WARNINGS:")
        for w in result.warnings:
            lines.append(f"  - {w}")

    return "\n".join(lines)
