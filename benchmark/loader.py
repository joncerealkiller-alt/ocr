"""
Loads data/debug_model_inputs/<run_id>/ runs into the models.py
dataclasses - the ONLY place in this tool that touches a folder name or
a metadata.json key directly (see models.py's docstring for why: "GUI
never knows about folder layouts").

Current on-disk shape (2026-07-26, both stage1 and stage2 field-level):

    data/debug_model_inputs/<run_id>/
        run_metadata.json                      - {run_id, started_at, items: [{item_id, dir}, ...]}
        row_0001/
            column_01_stage1/
                metadata.json, prompt.txt, output.txt,
                original_crop.png, model_input.png
            column_01_stage2/
                (same file set)
            column_02_stage1/  ...

run_metadata.json's own "items" list is the authoritative manifest of
what this run actually captured (in write order) - loader.py reads it
first rather than walking the filesystem, so a run directory containing
stray/partial files never gets misread as more complete than it is.
Any item_id listed but missing its folder on disk (an interrupted run)
is skipped with a warning, not a hard failure - a partially-copied or
still-in-progress run should still be reviewable for what it does have.

No support for the pre-2026-07-25 "stage2 is row-level, virtual items"
shape - see models.py's docstring for why that's a deliberate, not an
oversight (no such data exists on disk anymore).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional

from benchmark.models import BenchmarkItem, BenchmarkRun, RowGroup, RunSummary, Stage, _short_hash

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEBUG_ROOT = PROJECT_ROOT / "data" / "debug_model_inputs"

# Folders under data/debug_model_inputs/ that are NOT runs (no
# run_metadata.json of their own) - skipped by discover_runs() rather
# than reported as a broken/empty run.
_NON_RUN_NAMES = {"Ground_truth", "ground_truth"}


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _existing_or_none(path: Path) -> Optional[Path]:
    return path if path.exists() else None


def _parse_stage2_output(output_text: str) -> tuple[str, Optional[str]]:
    """
    Best-effort split of a stage2 output.txt line into (value,
    confidence) - deliberately permissive (never raises, never drops
    anything) since a malformed line is exactly the kind of thing a
    human reviewer needs to SEE, not have silently discarded. The
    field's OWN raw output_text is always preserved in full on the
    BenchmarkItem regardless of how this parses, so nothing is lost
    even when this returns something unexpected.

    Expected shape: "<label>: value|confidence" - the label is ignored
    here (the item already carries its real column_name from
    metadata.json; a mismatched label is a real finding, not something
    to silently correct - the raw_output the reviewer sees will still
    show whatever label the model actually wrote).
    """
    text = output_text.strip()
    if not text:
        return "", None
    _, sep, rest = text.partition(":")
    if not sep:
        # No colon at all - the model didn't even attempt the
        # "label: value|confidence" shape. Treat the whole line as the
        # value so it's still visible, rather than discarding it.
        rest = text
    rest = rest.strip()
    if "|" in rest:
        value, _, confidence = rest.rpartition("|")
        return value.strip(), confidence.strip() or None
    return rest, None


def _load_item(run_id: str, folder: Path) -> Optional[BenchmarkItem]:
    metadata_path = folder / "metadata.json"
    if not metadata_path.exists():
        warnings.warn(f"Skipping {folder} - no metadata.json (incomplete/interrupted run item).")
        return None
    metadata = _read_json(metadata_path)

    stage_label = metadata.get("stage", "")
    stage = Stage.STAGE2 if "stage2" in stage_label else Stage.STAGE1

    prompt_text = _read_text(folder / "prompt.txt") or metadata.get("prompt", "")
    output_text = _read_text(folder / "output.txt")

    if stage is Stage.STAGE2:
        value, confidence = _parse_stage2_output(output_text)
    else:
        value, confidence = output_text.strip(), None

    return BenchmarkItem(
        item_id=metadata.get("item_id", f"{folder.parent.name}/{folder.name}"),
        run_id=run_id,
        folder=folder,
        row_index=metadata.get("row_index", 0),
        column_index=metadata.get("column_index", 0),
        column_name=metadata.get("column_name", "?"),
        stage=stage,
        stage_label=stage_label,
        model=metadata.get("model"),
        runtime_seconds=metadata.get("runtime_seconds"),
        generation_config_hash=metadata.get("generation_config_hash"),
        reasoning_enabled=metadata.get("reasoning_enabled"),
        source_image_path=metadata.get("source_image_path"),
        prompt_text=prompt_text,
        prompt_hash=_short_hash(prompt_text),
        output_text=output_text,
        value=value,
        confidence=confidence,
        stage1_raw_reading=metadata.get("stage1_raw_reading"),
        row_bbox=metadata.get("row_bbox"),
        field_bbox=metadata.get("field_bbox"),
        original_crop_path=_existing_or_none(folder / "original_crop.png"),
        model_input_path=_existing_or_none(folder / "model_input.png"),
        empty_output=metadata.get("empty_output", not output_text.strip()),
        exception=metadata.get("exception"),
        metadata=metadata,
    )


def load_run(run_dir: Path) -> BenchmarkRun:
    """Full parse of one run - reads every item's metadata.json/prompt.txt/
    output.txt. Rows are sorted by row_index; items within a row are
    sorted by (column_index, stage) so a field's stage1/stage2 pair sit
    adjacent - useful for reviewing "what stage1 read vs what stage2
    confirmed/corrected it to" for the same field side by side."""
    run_dir = Path(run_dir)
    run_metadata = _read_json(run_dir / "run_metadata.json")
    run_id = run_metadata.get("run_id", run_dir.name)

    rows: dict[int, RowGroup] = {}
    source_image_path: Optional[str] = None

    for entry in run_metadata.get("items", []):
        folder = run_dir / entry["dir"]
        if not folder.is_dir():
            warnings.warn(f"Skipping {entry.get('item_id', entry)} - folder missing on disk "
                           f"(interrupted or partially-cleared run).")
            continue
        item = _load_item(run_id, folder)
        if item is None:
            continue
        if source_image_path is None:
            source_image_path = item.source_image_path
        rows.setdefault(item.row_index, RowGroup(row_index=item.row_index)).items.append(item)

    row_list = [rows[i] for i in sorted(rows)]
    for row in row_list:
        row.items.sort(key=lambda it: (it.column_index, it.stage.value))

    return BenchmarkRun(run_id=run_id, run_dir=run_dir, source_image_path=source_image_path,
                         rows=row_list)


def discover_runs(debug_root: Path | None = None) -> list[RunSummary]:
    """
    Lightweight run listing for the run-picker - reads each run's own
    run_metadata.json (for num_rows/num_items, cheap) plus exactly ONE
    item's metadata.json (for source_image_path, which run_metadata.json
    itself doesn't carry) rather than the full load_run() parse of
    every item. Runs are returned newest-first (run_id is a sortable
    UTC timestamp string, e.g. "20260726T071432303827Z").
    """
    debug_root = Path(debug_root) if debug_root is not None else DEFAULT_DEBUG_ROOT
    if not debug_root.exists():
        return []

    summaries: list[RunSummary] = []
    for run_dir in sorted(debug_root.iterdir(), reverse=True):
        if not run_dir.is_dir() or run_dir.name in _NON_RUN_NAMES:
            continue
        run_metadata_path = run_dir / "run_metadata.json"
        if not run_metadata_path.exists():
            continue
        try:
            run_metadata = _read_json(run_metadata_path)
        except (OSError, json.JSONDecodeError) as e:
            warnings.warn(f"Skipping {run_dir.name} - unreadable run_metadata.json: {e}")
            continue

        items = run_metadata.get("items", [])
        row_indices = set()
        source_image_path = None
        for entry in items:
            item_id = entry.get("item_id", "")
            if "/" in item_id:
                row_part = item_id.split("/", 1)[0]
                row_indices.add(row_part)
            if source_image_path is None:
                folder = run_dir / entry["dir"]
                meta_path = folder / "metadata.json"
                if meta_path.exists():
                    try:
                        source_image_path = _read_json(meta_path).get("source_image_path")
                    except (OSError, json.JSONDecodeError):
                        pass

        summaries.append(RunSummary(
            run_id=run_metadata.get("run_id", run_dir.name),
            run_dir=run_dir,
            source_image_path=source_image_path,
            num_rows=len(row_indices),
            num_items=len(items),
        ))
    return summaries
