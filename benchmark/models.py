"""
Data model for the OCR Benchmark Review tool.

Normalizes data/debug_model_inputs/<run_id>/ into a BenchmarkRun of
RowGroups of BenchmarkItems - the GUI only ever talks to these
dataclasses, it never inspects a folder name or a metadata.json key
directly (loader.py is the only place that does), so a future
debug_dump.py schema change only requires updating loader.py.

SIMPLIFIED 2026-07-26: the original design (2026-07-25) assumed Stage 1
was field-level (one dir per column) but Stage 2 was still row-level on
disk (one output.txt with up to 5 "ColumnName: value|confidence"
lines), needing a BenchmarkItem.is_virtual/Evidence split so several
"virtual" per-column items could be parsed out of one shared row-level
Evidence object. That assumption is now stale: the same-day stage2
field-level redesign (core/row_extraction.py) made stage2 folders
identical in shape to stage1's - one real column_NN_stage2/ folder per
field, with its own single-line output.txt - so there is no longer a
virtual-splitting case to handle. Every debug_model_inputs/ run using
the old shape was already cleared out (Jon: "arent usable to test
current pipeline outputs when comparing them"), so there's no legacy
data on disk to support either. Per this project's "don't design for
hypothetical requirements" rule, that complexity is dropped rather than
kept dormant - every BenchmarkItem now maps 1:1 to one real folder.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class Stage(str, Enum):
    STAGE1 = "stage1"
    STAGE2 = "stage2"


class OCRScore(str, Enum):
    PERFECT = "Perfect"
    MINOR_ERROR = "Minor Error"
    MAJOR_ERROR = "Major Error"
    HALLUCINATED = "Hallucinated"
    FAILED = "Failed"


# (csv_field_name, human label) - order here is the display order for
# the checkbox panel, and the name is reused verbatim as the
# ScoreRecord/CSV field name.
ERROR_TYPE_FIELDS: list[tuple[str, str]] = [
    ("wrong_character", "Wrong character"),
    ("missing_character", "Missing character"),
    ("inserted_character", "Inserted character"),
    ("merged_words", "Merged words"),
    ("split_words", "Split words"),
    ("hallucinated", "Hallucinated text"),
    ("repeated_tokens", "Repeated tokens"),
    ("formatting_issue", "Formatting issue"),
    ("prompt_ignored", "Prompt ignored"),
    ("preprocessing_artifact", "Preprocessing artifact"),
    ("wrong_field", "Wrong field"),
    ("truncated_output", "Truncated output"),
]


def _short_hash(text: str) -> str:
    """Same style as core/row_extraction.py's generation_config_hash -
    a short, stable identifier for "was this the same prompt wording",
    not a security hash. No prompt_hash field exists in metadata.json
    on disk (only generation_config_hash does), so this is computed
    fresh at load time from the prompt text actually used."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class BenchmarkItem:
    """One scoreable unit - one on-disk column_NN_stageN/ folder, 1:1.
    See this module's docstring for why there's no virtual/Evidence
    split anymore.

    item_id is stable across reloads of the same immutable debug run
    (it's literally the folder's own item_id from metadata.json /
    run_metadata.json) - the resume/green-check/CSV join key.
    """

    item_id: str                       # e.g. "row_0001/column_03_stage2"
    run_id: str
    folder: Path

    row_index: int
    column_index: int
    column_name: str
    stage: Stage
    stage_label: str                   # raw metadata "stage" value, e.g. "stage2_structure_field"

    model: Optional[str]
    runtime_seconds: Optional[float]
    generation_config_hash: Optional[str]
    reasoning_enabled: Optional[bool]
    source_image_path: Optional[str]

    prompt_text: str
    prompt_hash: str
    output_text: str
    value: str
    confidence: Optional[str]          # stage1 has no confidence concept - None there
    stage1_raw_reading: Optional[str]  # only ever present on stage2 items

    row_bbox: Optional[list]
    field_bbox: Optional[list]
    original_crop_path: Optional[Path]
    model_input_path: Optional[Path]

    empty_output: bool
    exception: Optional[str]
    metadata: dict[str, Any]           # full raw metadata.json, for anything not lifted above


@dataclass
class RowGroup:
    row_index: int
    items: list[BenchmarkItem] = field(default_factory=list)

    def item_for(self, column_name: str, stage: Stage) -> Optional[BenchmarkItem]:
        for item in self.items:
            if item.column_name == column_name and item.stage == stage:
                return item
        return None

    def column_names(self) -> list[str]:
        """Column names in this row, in on-disk column_index order, de-duplicated
        (stage1 and stage2 both contribute an entry per column)."""
        seen: dict[str, int] = {}
        for item in self.items:
            seen.setdefault(item.column_name, item.column_index)
        return sorted(seen, key=lambda name: seen[name])


@dataclass
class BenchmarkRun:
    run_id: str
    run_dir: Path
    source_image_path: Optional[str]
    rows: list[RowGroup] = field(default_factory=list)

    def all_items(self) -> list[BenchmarkItem]:
        return [item for row in self.rows for item in row.items]

    def item_by_id(self, item_id: str) -> Optional[BenchmarkItem]:
        for item in self.all_items():
            if item.item_id == item_id:
                return item
        return None


@dataclass
class RunSummary:
    """Lightweight entry for the run-picker, built without loading and
    parsing every item in the run (see loader.discover_runs)."""

    run_id: str
    run_dir: Path
    source_image_path: Optional[str]
    num_rows: int
    num_items: int


@dataclass
class ScoreRecord:
    """One row of benchmark_results/benchmark_results.csv. Field order
    here IS the CSV column order - keep in sync with
    benchmark_db.CSV_FIELDNAMES."""

    timestamp: str
    benchmark_session: str
    item_id: str
    run_id: str
    stage: str
    row: str
    column: str
    column_name: str
    model: str
    runtime: str
    generation_config_hash: str
    prompt_hash: str
    ocr_score: str
    confidence: str
    wrong_character: bool
    missing_character: bool
    inserted_character: bool
    merged_words: bool
    split_words: bool
    hallucinated: bool
    repeated_tokens: bool
    formatting_issue: bool
    prompt_ignored: bool
    preprocessing_artifact: bool
    wrong_field: bool
    truncated_output: bool
    notes: str
