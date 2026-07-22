"""
Scores a run_two_stage_extraction.py (or workflow_gui.py's Two-Stage
Extraction panel, which just shells out to the same script) JSON output
against your own human-verified ground_truth_log.jsonl - built because
there is NO existing evidence in this project for which stage1/stage2
model pairing performs best (config/pipeline.yaml explicitly marks
every assignment a placeholder; data/outputs/model_assessments/ is
empty; the two GROUND_TRUTH_*.md files are about whole-document
extraction on different document types, not this row-level pipeline).
Rather than guess, this turns each candidate pairing you try through
the GUI into a real accuracy number against labels you already trust.

Usage:
    python score_two_stage_against_ground_truth.py \\
        --results-json data/outputs/row_segmentation/<n>_twostage_extraction.json \\
        --ground-truth-log data/outputs/ground_truth_log.jsonl \\
        --sidecar-path data/outputs/row_segmentation/<n>_sidecar.json

--sidecar-path filters the (potentially multi-page) ground-truth log
down to just the page this results JSON was extracted from - required
since ground_truth_log.jsonl is a single append-only file spanning
every page you've ever labeled, not just one.

Target normalization mirrors export_lora_dataset.py's STATUS_TO_TARGET
mapping exactly (illegible/blank become those literal strings; readable/
partially_readable use the typed value) - the SAME notion of "correct"
used to build the LoRA training set, so this scorer and that dataset
never silently disagree about what counts as the right answer for a
given status.

Comparison is exact-match after stripping whitespace and casefolding -
deliberately strict (a stage2 model outputting "john" for ground truth
"John" counts as correct; "Jon" for "John" does not) since the whole
point of this project's confidence-tagging discipline is not to fuzzy-
match away real transcription differences. If you want to eyeball NEAR
misses too, --show-mismatches prints every non-exact-match pair for
manual review rather than just being reported.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


# Mirrors export_lora_dataset.py's STATUS_TO_TARGET exactly - kept as a
# literal duplicate rather than importing that module (not meant as a
# library, it's a standalone CLI tool, same convention as every other
# script in this project).
def _record_to_expected(record: dict) -> str | None:
    status = record.get("status")
    if status == "illegible":
        return "illegible"
    if status == "blank":
        return "blank"
    if status in ("readable", "partially_readable"):
        value = (record.get("value") or "").strip()
        return value if value else None
    return None


def _normalize(s: str) -> str:
    return s.strip().casefold()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-json", type=str, required=True,
                         help="Path to a _twostage_extraction.json produced by "
                              "run_two_stage_extraction.py (or the GUI panel).")
    parser.add_argument("--ground-truth-log", type=str, required=True)
    parser.add_argument("--sidecar-path", type=str, required=True,
                         help="Filters ground_truth_log.jsonl to records from this "
                              "page only - must match the sidecar_path field exactly "
                              "as stored in the log (check a line of the log if "
                              "unsure of the exact string).")
    parser.add_argument("--show-mismatches", action="store_true",
                         help="Print every non-exact-match (predicted, expected) pair "
                              "for manual review, not just the summary counts.")
    args = parser.parse_args()

    results_path = Path(args.results_json)
    if not results_path.exists():
        print(f"ERROR: {results_path} not found.")
        return
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    # results is a list of RowExtractionResult.model_dump() dicts - build
    # a (row_index, column) -> predicted value lookup, one entry per
    # field actually present (a field can be legitimately absent if
    # stage 2 dropped/never produced it - see "missing" counts below,
    # do not treat an absent field as an empty-string match).
    predicted_by_row_col: dict[tuple[int, str], str] = {}
    schema_pass_by_row: dict[int, bool] = {}
    for row_result in results:
        row_index = row_result["row_index"]
        schema_pass_by_row[row_index] = row_result.get("schema_pass", False)
        for column, field in (row_result.get("fields") or {}).items():
            predicted_by_row_col[(row_index, column)] = field.get("value", "")

    ground_truth_path = Path(args.ground_truth_log)
    if not ground_truth_path.exists():
        print(f"ERROR: {ground_truth_path} not found.")
        return

    records = []
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    page_records = [r for r in records if r.get("sidecar_path") == args.sidecar_path]

    if not page_records:
        print(f"ERROR: no ground-truth records found for sidecar_path "
              f"{args.sidecar_path!r}. Available sidecar_paths in this log:")
        for p in sorted({r.get("sidecar_path") for r in records}):
            print(f"  {p}")
        return

    per_column = defaultdict(lambda: {"correct": 0, "total": 0, "missing": 0})
    per_status = defaultdict(lambda: {"correct": 0, "total": 0, "missing": 0})
    mismatches = []

    for record in page_records:
        expected = _record_to_expected(record)
        if expected is None:
            continue  # same skip condition export_lora_dataset.py uses
        column = record.get("column")
        row_index = record.get("row_index")
        status = record.get("status")

        key = (row_index, column)
        if key not in predicted_by_row_col:
            per_column[column]["total"] += 1
            per_column[column]["missing"] += 1
            per_status[status]["total"] += 1
            per_status[status]["missing"] += 1
            continue

        predicted = predicted_by_row_col[key]
        is_correct = _normalize(predicted) == _normalize(expected)

        per_column[column]["total"] += 1
        per_status[status]["total"] += 1
        if is_correct:
            per_column[column]["correct"] += 1
            per_status[status]["correct"] += 1
        else:
            mismatches.append({
                "row": row_index, "column": column, "status": status,
                "predicted": predicted, "expected": expected,
            })

    total_rows_in_results = len(schema_pass_by_row)
    schema_failed = sum(1 for v in schema_pass_by_row.values() if not v)

    print(f"Results file:      {results_path}")
    print(f"Ground-truth page: {args.sidecar_path}")
    print(f"Rows in results:   {total_rows_in_results} ({schema_failed} failed schema validation)")
    print()

    print("Accuracy by column:")
    overall_correct = overall_total = 0
    for column, counts in sorted(per_column.items()):
        acc = counts["correct"] / counts["total"] if counts["total"] else 0.0
        missing_note = f", {counts['missing']} missing" if counts["missing"] else ""
        print(f"  {column}: {counts['correct']}/{counts['total']} ({acc:.0%}){missing_note}")
        overall_correct += counts["correct"]
        overall_total += counts["total"]

    print()
    print("Accuracy by ground-truth status (shows whether abstention specifically "
          "is being handled, not just clean-text reading):")
    for status, counts in sorted(per_status.items()):
        acc = counts["correct"] / counts["total"] if counts["total"] else 0.0
        missing_note = f", {counts['missing']} missing" if counts["missing"] else ""
        print(f"  {status}: {counts['correct']}/{counts['total']} ({acc:.0%}){missing_note}")

    overall_acc = overall_correct / overall_total if overall_total else 0.0
    print()
    print(f"OVERALL: {overall_correct}/{overall_total} ({overall_acc:.0%})")

    if args.show_mismatches and mismatches:
        print(f"\n{len(mismatches)} mismatch(es):")
        for m in mismatches:
            print(f"  row {m['row']} [{m['column']}] ({m['status']}): "
                  f"predicted {m['predicted']!r} vs expected {m['expected']!r}")


if __name__ == "__main__":
    main()
