"""
Loads per-doc-type templates for the automated sidecar generation
pipeline (core/auto_sidecar.py) from config/document_templates/*.yaml.

Built 2026-07-27 as part of the automated-sidecar-generation
experiment - see core/auto_sidecar.py's module docstring for the full
pipeline this feeds into (classify -> template -> table boundary ->
header region -> rows -> sidecar). Templates are GUIDES, not
authority: every fraction here is a search-window seed for
core/auto_sidecar.py's boundary locators, which prefer an actually-
detected printed border over these numbers - see individual template
files' own comments for why (they're calibrated against a single real
sample per doc type, a thin base).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "config" / "document_templates"


@dataclass
class DocumentTemplate:
    doc_type: str
    display_name: str
    row_strategy: str          # "periodic" (known row_count) or "detect" (general)
    expected_row_count: int | None
    expected_columns: list[str]
    regions_approx: dict       # {"metadata": {"y_frac": [..]}, "header": {...}, "table": {"y_frac": [..], "x_frac": [..]}}
    classification: dict       # {"aspect_ratio_range": [..], "row_count_range": [..], "column_count_range": [..]}
    # {"<column name>": {"x_frac": [..]}, ...} - column automation
    # (core.auto_sidecar.locate_columns()), 2026-07-27. Optional per
    # column - a name absent here just means no calibration data exists
    # yet for that column, and it's left for manual masking as before.
    column_regions_approx: dict = field(default_factory=dict)
    # {"top_offset_frac": .., "bottom_offset_frac": ..} - the printed
    # column-NUMBER row, expressed as an offset above table_top (not an
    # absolute y_frac, since table_top is already reliably located by
    # the time this is used). Optional - templates without it (e.g.
    # printed_manifest, no ruled grid at all) just skip the number-row
    # anchor and rely on column_regions_approx alone, same as before
    # this field existed. See core/auto_sidecar.py's
    # _detect_header_number_blobs() (built, tested, NOT currently wired
    # in - see locate_columns()'s own docstring for why).
    number_row_approx: dict | None = None
    # Calibrated real row height, as a fraction of PAGE HEIGHT - used
    # ONLY by locate_table_boundary()'s table_top plausibility check
    # (2026-07-27, the z000017634 foundation-bug fix) to detect when
    # the chosen boundary is suspiciously evenly-spaced from an earlier
    # candidate rule (a sign the algorithm picked a row-separator line
    # instead of the true header/table divider). None for templates
    # without a fixed, known row spacing (e.g. "detect"-strategy
    # templates, which have no such notion).
    expected_row_height_frac: float | None = None
    # config/columns/*.txt filename (e.g. "census_core_fields.txt") -
    # the curated, ordered set of fields ui/quarantine_review_ui.py
    # offers for manual column masking (a subset of expected_columns
    # above, in OUTPUT order - see that UI's own DEFAULT_COLUMNS_FILE_
    # BY_DOC_TYPE-turned-this-field comment for why it's a distinct
    # list). Optional - templates without it just fall back to
    # expected_columns there, with a visible warning, same as an
    # unmapped doc_type did before this field existed.
    columns_file: str | None = None


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_template(doc_type: str) -> DocumentTemplate:
    path = TEMPLATES_DIR / f"{doc_type}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"No document template for doc_type={doc_type!r} - expected {path}. "
            f"Known templates: {sorted(t.doc_type for t in load_all_templates().values())}"
        )
    return _template_from_dict(_load_yaml(path))


def load_all_templates() -> dict[str, DocumentTemplate]:
    """Scans config/document_templates/*.yaml - every file there is a
    known template, no separate registry to keep in sync."""
    templates = {}
    for path in sorted(TEMPLATES_DIR.glob("*.yaml")):
        tmpl = _template_from_dict(_load_yaml(path))
        templates[tmpl.doc_type] = tmpl
    return templates


def _template_from_dict(data: dict) -> DocumentTemplate:
    return DocumentTemplate(
        doc_type=data["doc_type"],
        display_name=data.get("display_name", data["doc_type"]),
        row_strategy=data["row_strategy"],
        expected_row_count=data.get("expected_row_count"),
        expected_columns=list(data.get("expected_columns", [])),
        regions_approx=data.get("regions_approx", {}),
        classification=data.get("classification", {}),
        column_regions_approx=data.get("column_regions_approx", {}),
        number_row_approx=data.get("number_row_approx"),
        expected_row_height_frac=data.get("expected_row_height_frac"),
        columns_file=data.get("columns_file"),
    )
