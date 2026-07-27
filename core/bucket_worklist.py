"""
Shared read/append helpers for standalone preprocessing tools that
consume a core/classifier.py bucket CSV as their input worklist and
record their own output as a sibling CSV.

Architecture context (Jon's direction, 2026-07-25): every module in
this project is currently a standalone tool by design - isolates
development/testing/debugging while the pipeline is still evolving.
End-to-end automation is planned for later, once individual modules
have stabilized, not now. This module exists so that when that
orchestration is eventually built, each stage's input/output is
already a well-defined file-based interface (read one bucket CSV,
write one sibling result CSV) rather than something that needs
retrofitting - but nothing here runs unattended today. A tool built on
these helpers (ui/dewarp_preprocessor_ui.py today, others later) still
processes ONE image at a time with a human confirming/adjusting before
each Save - this module only provides the shared read-worklist/
append-result primitives, not a batch runner.

Input side: data/buckets/<category>.csv, written by core/classifier.py
- CSV_FIELDS there includes "file_path" as the first column; this
module only ever reads that one column, so it works unchanged even if
core/classifier.py's other columns change later.

Output side: a "preprocessed bucket" - data/buckets/<category>_<stage>
.csv, e.g. dense_tabular_rows.csv + stage="dewarped" ->
dense_tabular_rows_dewarped.csv. Same flat data/buckets/ directory as
the classification bucket it came from (Jon's direction, 2026-07-25) -
sits right next to it, not in a separate subfolder. One row per
processed source file: source_file_path, output_file_path (the new
file this stage produced, or the SAME path if the human chose to
bypass/pass through unmodified), status ("dewarped" or whatever
per-stage status a caller passes), and a UTC timestamp. Append-only,
same discipline as ground_truth_log.jsonl and reviewed_uncertain.csv
elsewhere in this project - never rewrites a prior row.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from pathlib import Path

PREPROCESSED_FIELDS = ["source_file_path", "output_file_path", "status", "timestamp"]


def load_bucket_filepaths(bucket_csv_path: str | Path) -> list[str]:
    """
    Reads the "file_path" column from an existing core/classifier.py
    bucket CSV, in file order (the order the classifier originally
    wrote them). Raises the normal csv/OSError if the path doesn't
    exist or isn't a real CSV - a caller picking the wrong file should
    see that error directly, not a silently empty worklist.
    """
    path = Path(bucket_csv_path)
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [row["file_path"] for row in reader if row.get("file_path")]


def preprocessed_bucket_path(bucket_csv_path: str | Path, stage: str) -> Path:
    """
    Sibling output CSV path for one pipeline stage's results - see
    module docstring for the naming convention. `stage` should be a
    short, filesystem-safe word (e.g. "dewarped") identifying which
    preprocessing step produced this file, since more than one stage
    may eventually read the same input bucket and each needs its own
    output CSV rather than overwriting a shared one.
    """
    path = Path(bucket_csv_path)
    return path.with_name(f"{path.stem}_{stage}.csv")


def load_processed_records(preprocessed_csv_path: str | Path) -> list[dict]:
    """
    Full rows (source_file_path, output_file_path, status, timestamp)
    from a preprocessed-bucket CSV - richer than load_processed_sources()
    below for callers that need to VERIFY what was recorded (e.g.
    ui/dewarp_preprocessor_ui.py's self-heal check: does output_
    file_path still actually exist on disk, and if not, can it be
    regenerated from a corner sidecar), not just check whether an
    entry exists at all. Returns [] if the file doesn't exist yet.
    """
    path = Path(preprocessed_csv_path)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_processed_sources(preprocessed_csv_path: str | Path) -> set[str]:
    """
    Source file_paths already recorded in a preprocessed-bucket CSV -
    used for resume support (skip anything already done on reopen,
    same "don't silently redo completed work" discipline as
    ui/row_segmentation_ui.py's per-column resume). Returns an empty
    set if the file doesn't exist yet - "nothing processed so far" is
    a normal starting state, not an error.
    """
    return {
        row["source_file_path"] for row in load_processed_records(preprocessed_csv_path)
        if row.get("source_file_path")
    }


def append_preprocessed_record(
    preprocessed_csv_path: str | Path,
    source_file_path: str,
    output_file_path: str,
    status: str,
) -> None:
    """
    Appends exactly one row - never rewrites or reorders prior rows.
    Writes the header once, on first creation, same convention as
    core/classifier.py's own open_bucket_writers(). Flushed + fsynced
    before returning so a crash immediately after doesn't silently
    lose the record - matches this project's established append-only
    file discipline elsewhere (ground_truth_log.jsonl,
    reviewed_uncertain.csv).
    """
    path = Path(preprocessed_csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PREPROCESSED_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({
            "source_file_path": source_file_path,
            "output_file_path": output_file_path,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        f.flush()
        os.fsync(f.fileno())
