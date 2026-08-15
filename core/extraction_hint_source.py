"""
Extraction hint source - feeds Phase 1 of ui/hint_validation_ui.py from
a two-stage extraction output JSON (scripts/run_two_stage_extraction.py
--out), the same way core/csv_hint_source.py feeds it from a reference
transcription CSV. Built 2026-08-15 to wire the hint-free two-stage
design's field_agreement signal (docs/TWO_STAGE_HINT_FREE_PROPOSAL.md,
core/row_extraction.py) into the existing review UI rather than
building a new tool - the UI's hint interface was already pluggable.

What field_agreement means here (and why the queue cares):
  True  = the two INDEPENDENT model reads agreed under
          core.row_extraction.fields_agree()'s rules. Measured live
          (2026-08-15, 12B+MiniCPM pair, hint-free template): zero
          false auto-accepts across every validation run - so these are
          fast Yes-confirmations for a human, or skippable entirely in
          a spot-check-only workflow.
  False = the reads disagreed (or either abstained) - these are the
          fields the review queue EXISTS for, shown first, with stage
          1's independent reading displayed alongside stage 2's value
          because the evidence says stage 1 is sometimes the correct
          one (e.g. MiniCPM's stage-1 "Huzie Bertha" vs 12B's stage-2
          "Henzie Bertha" - review recovers these).

Row matching: an extraction JSON is produced FROM a specific sidecar,
so the pairing is explicit (the reviewer loads both) - row_index is
already the same index space. match_check() offers a bbox sanity check
so a mispaired file fails loudly instead of silently reviewing the
wrong page's values.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def load_extraction_results(json_path: str | Path) -> dict[int, dict[str, Any]]:
    """row_index -> row record, from a run_two_stage_extraction --out JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data["rows"] if isinstance(data, dict) and "rows" in data else data
    return {r["row_index"]: r for r in rows}


def match_check(extraction_rows: dict[int, dict], sidecar: dict) -> Optional[str]:
    """Returns a human-readable mismatch warning if the extraction rows'
    bboxes don't line up with the loaded sidecar's rows (a mispaired
    file), or None if everything checks out."""
    sidecar_bboxes = {r["index"]: list(r["bbox"]) for r in sidecar.get("rows", [])}
    for idx, rec in extraction_rows.items():
        sc = sidecar_bboxes.get(idx)
        if sc is None:
            return f"extraction row {idx} does not exist in the loaded sidecar"
        if list(rec.get("bbox", [])) != sc:
            return (f"extraction row {idx} bbox {rec.get('bbox')} != sidecar row bbox {sc} - "
                    "is this extraction JSON from a different sidecar/page?")
    return None


def _field_value(row_record: dict, column_name: str) -> Optional[str]:
    field = (row_record.get("fields") or {}).get(column_name)
    if field is None:
        return None
    value = field.get("value") if isinstance(field, dict) else getattr(field, "value", None)
    return value


def get_hint(row_record: Optional[dict], column_name: str) -> Optional[str]:
    """Stage 2's extracted value for this field, or None when there is
    nothing worth confirming: no record, missing field, empty value, or
    an explicit '?' abstention (an abstention is a routing signal, not
    a candidate answer - the field goes to manual entry instead)."""
    if row_record is None:
        return None
    value = _field_value(row_record, column_name)
    if value is None:
        return None
    value = str(value).strip()
    if not value or value == "?":
        return None
    return value


def agreement(row_record: Optional[dict], column_name: str) -> Optional[bool]:
    """field_agreement for this field: True/False, or None when the
    record predates the field (or the field wasn't extracted). Handles
    both real booleans and the "True"/"False" strings the pipeline's
    JSON serialization produces."""
    if row_record is None:
        return None
    fa = row_record.get("field_agreement") or {}
    raw = fa.get(column_name)
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() == "true"


def stage1_reading(row_record: Optional[dict], column_name: str) -> Optional[str]:
    """Stage 1's independent raw reading for this field, parsed out of
    the persisted stage1_raw_output ("Column: reading" lines)."""
    if row_record is None:
        return None
    for line in (row_record.get("stage1_raw_output") or "").split("\n"):
        if ":" in line:
            k, v = line.split(":", 1)
            if k.strip() == column_name:
                return v.strip()
    return None
