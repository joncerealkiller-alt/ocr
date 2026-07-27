"""
Append-only CSV storage for scored BenchmarkItems - benchmark_results/
benchmark_results.csv, a PROJECT-LEVEL folder (Jon's direction,
2026-07-25: kept separate from data/debug_model_inputs/, since a scored
run's evaluation history should survive independently of whatever
happens to the debug capture it was scored against - e.g. an old debug
run being cleared out later shouldn't take its scoring history with
it).

Same append-only discipline as core/bucket_worklist.py and
ground_truth_log.jsonl/reviewed_uncertain.csv elsewhere in this project:
flush + fsync before returning, header written once on first creation,
existing rows NEVER rewritten or reordered. Re-scoring an item (the
reviewer changes their mind, or a later pass wants to correct a
mistake) APPENDS a new row rather than overwriting the old one - the
full history of every scoring pass is preserved; "the current score"
for an item is simply whichever row for that item_id was written last.
"""

from __future__ import annotations

import csv
import os
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path

from benchmark.models import ERROR_TYPE_FIELDS, ScoreRecord

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "benchmark_results"
DEFAULT_RESULTS_CSV = DEFAULT_RESULTS_DIR / "benchmark_results.csv"

# Field order here IS the CSV column order - kept in sync with
# ScoreRecord's own field order (dataclasses.fields() reads it directly
# off ScoreRecord rather than hand-maintaining a parallel list that
# could silently drift out of sync with it).
CSV_FIELDNAMES: list[str] = [f.name for f in fields(ScoreRecord)]


def new_benchmark_session_id() -> str:
    """Same timestamp format as core/debug_dump.py's run_id (
    %Y%m%dT%H%M%S%fZ) - one session id groups every score entered in
    one sitting of the GUI, for later run-comparison (Gemini spec's
    "compare two runs" suggestion - a session groups "the scores I gave
    while reviewing run A" separately from "the scores I gave reviewing
    run B", even if both happen to touch overlapping item_ids)."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def append_score(record: ScoreRecord, csv_path: Path | str = DEFAULT_RESULTS_CSV) -> None:
    """Appends exactly one row - never rewrites or reorders prior rows.
    Writes the header once, on first creation. Flushed + fsynced before
    returning so a crash immediately after doesn't silently lose the
    record."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    row = asdict(record)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def load_all_scores(csv_path: Path | str = DEFAULT_RESULTS_CSV) -> list[ScoreRecord]:
    """Full history, in the order written - every scoring pass for
    every item, not just the latest. Returns [] if the file doesn't
    exist yet ("nothing scored so far" is a normal starting state)."""
    path = Path(csv_path)
    if not path.exists():
        return []
    field_names = {f.name for f in fields(ScoreRecord)}
    # dataclasses.fields()' own .type is an unresolved STRING under
    # `from __future__ import annotations` (models.py uses it), so
    # `f.type is bool` would silently never match - the boolean field
    # names are instead the canonical ones ERROR_TYPE_FIELDS already
    # defines (the same list the GUI's checkbox panel is built from),
    # reused here rather than re-deriving them a second, driftable way.
    bool_fields = {name for name, _ in ERROR_TYPE_FIELDS}
    records = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            kwargs = {k: v for k, v in row.items() if k in field_names}
            for bf in bool_fields:
                kwargs[bf] = str(kwargs.get(bf, "")).strip().lower() in ("true", "1", "yes")
            records.append(ScoreRecord(**kwargs))
    return records


def load_latest_scores(csv_path: Path | str = DEFAULT_RESULTS_CSV) -> dict[str, ScoreRecord]:
    """The CURRENT score per item_id - the last row written for each
    item_id, since the CSV is strictly append-only in chronological
    order. This is what the GUI's resume/green-check logic and the
    navigator's "already scored" indicator should read, NOT
    load_all_scores() (which would need the caller to redo this same
    "keep the last one" reduction themselves)."""
    latest: dict[str, ScoreRecord] = {}
    for record in load_all_scores(csv_path):
        latest[record.item_id] = record
    return latest
