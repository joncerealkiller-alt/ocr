"""
Scores a run_two_stage_extraction.py (or run_single_column_extraction's
CSV/JSON, or the GUI panels that shell out to either) result file against
your own human-verified ground_truth_log.jsonl - built because there is
NO existing evidence in this project for which stage1/stage2 model
pairing performs best (config/pipeline.yaml explicitly marks every
assignment a placeholder; data/outputs/model_assessments/ is empty; the
two GROUND_TRUTH_*.md files are about whole-document extraction on
different document types, not this row-level pipeline). Rather than
guess, this turns each candidate pairing you try into a real accuracy
number against labels you already trust.

CLI usage:
    python score_two_stage_against_ground_truth.py \
        --results-json data/outputs/row_segmentation/<n>_twostage_extraction.json \
        --ground-truth-log data/outputs/ground_truth_log.jsonl \
        --sidecar-path data/outputs/row_segmentation/<n>_sidecar.json

Library usage (2026-07-22, added for workflow_gui.py's Scoring tab - see
that tab's docstring): import score_results() directly for a structured
ScoringReport instead of parsed console text. main() below is a THIN
wrapper around the same function - the CLI and the GUI tab produce
byte-for-byte identical numbers because they call the same code, not two
separately-maintained evaluators.

--sidecar-path filters the (potentially multi-page) ground-truth log
down to just the page this results JSON was extracted from - required
since ground_truth_log.jsonl is a single append-only file spanning
every page you've ever labeled, not just one.

Target normalization mirrors export_lora_dataset.py's STATUS_TO_TARGET
mapping (illegible/blank become those literal strings; readable/
partially_readable use the typed value) for LOOKING UP the ground-truth
answer, but comparison against a live extraction's PREDICTED value uses
the live pipeline's own abstention convention (empty string for blank,
literal "?" for illegible - see _is_correct's docstring for the real
scoring bug this distinction fixed).

Comparison is exact-match after stripping whitespace and casefolding -
deliberately strict (a stage2 model outputting "john" for ground truth
"John" counts as correct; "Jon" for "John" does not) since the whole
point of this project's confidence-tagging discipline is not to fuzzy-
match away real transcription differences. If you want to eyeball NEAR
misses too, --show-mismatches (CLI) prints every non-exact-match pair.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ScoringReport:
    """Structured result of score_results() - the single shape both the
    CLI's main() and workflow_gui.py's Scoring tab consume, so a table
    column in the GUI and a printed line in the CLI are always reading
    the exact same numbers."""
    results_path: str
    ground_truth_path: str
    sidecar_path: str
    model_name: str | None
    total_rows_in_results: int
    schema_failed: int
    per_column: dict[str, dict] = field(default_factory=dict)
    per_status: dict[str, dict] = field(default_factory=dict)
    overall_correct: int = 0
    overall_total: int = 0
    mismatches: list[dict] = field(default_factory=list)
    false_confidence: list[dict] = field(default_factory=list)
    error: str | None = None
    available_sidecar_paths: list[str] = field(default_factory=list)

    @property
    def overall_accuracy(self) -> float:
        return self.overall_correct / self.overall_total if self.overall_total else 0.0

    @property
    def false_confidence_rate(self) -> float:
        return len(self.false_confidence) / len(self.mismatches) if self.mismatches else 0.0


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


def _is_correct(predicted: str, expected: str) -> bool:
    """
    Exact match after normalization, EXCEPT for a real convention
    mismatch found 2026-07-22: _record_to_expected() (above) returns
    the literal words "blank"/"illegible" - correct for comparing
    against export_lora_dataset.py's LoRA training targets, matching
    that tool's own STATUS_TO_TARGET convention exactly (see this
    script's module docstring). But the LIVE extraction prompts
    (build_row_prompt/build_structuring_prompt) use a DIFFERENT,
    equally legitimate convention for the same two concepts: an EMPTY
    value string for "genuinely blank" (prompt rule: "write the column
    name and colon followed immediately by the pipe... with nothing in
    between"), and a literal "?" character for "cannot read" - never
    the words "blank"/"illegible" themselves. Comparing a correctly-
    behaving live-extraction result against the LoRA-target words
    directly was scoring genuinely correct abstention as wrong -
    caught when a real epistemically-strict prompt test came back a
    reported 0% despite every value being an honest "" or "?" against
    ground-truth blank/illegible rows.
    """
    if expected == "blank":
        return predicted.strip() == "" or _normalize(predicted) == "blank"
    if expected == "illegible":
        return predicted.strip() == "?" or _normalize(predicted) == "illegible"
    return _normalize(predicted) == _normalize(expected)


def score_results(
    results_path,
    ground_truth_path,
    sidecar_path: str,
    model_name: str | None = None,
) -> ScoringReport:
    """
    The single evaluator - both main() (CLI) and workflow_gui.py's
    Scoring tab call this directly and render the SAME numbers, one as
    printed text, one as a GUI table. Never duplicate this logic
    elsewhere; if the GUI needs a number this doesn't compute, add it
    here so both front ends get it.

    Returns a ScoringReport with .error set (and everything else at
    defaults) on a recoverable problem (missing file, no matching
    ground-truth records) rather than raising - both front ends check
    .error and display it appropriately instead of crashing on a bad
    path, which is the common case when a human is picking files by
    hand in a GUI.
    """
    results_path = Path(results_path)
    ground_truth_path = Path(ground_truth_path)

    if not results_path.exists():
        return ScoringReport(
            results_path=str(results_path), ground_truth_path=str(ground_truth_path),
            sidecar_path=sidecar_path, model_name=model_name,
            total_rows_in_results=0, schema_failed=0,
            error=f"Results file not found: {results_path}")
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    predicted_by_row_col = {}
    schema_pass_by_row = {}
    for row_result in results:
        row_index = row_result["row_index"]
        schema_pass_by_row[row_index] = row_result.get("schema_pass", False)
        for column, field_data in (row_result.get("fields") or {}).items():
            predicted_by_row_col[(row_index, column)] = (
                field_data.get("value", ""), field_data.get("confidence", ""))

    if not ground_truth_path.exists():
        return ScoringReport(
            results_path=str(results_path), ground_truth_path=str(ground_truth_path),
            sidecar_path=sidecar_path, model_name=model_name,
            total_rows_in_results=len(schema_pass_by_row),
            schema_failed=sum(1 for v in schema_pass_by_row.values() if not v),
            error=f"Ground-truth log not found: {ground_truth_path}")

    records = []
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    page_records = [r for r in records if r.get("sidecar_path") == sidecar_path]

    if not page_records:
        return ScoringReport(
            results_path=str(results_path), ground_truth_path=str(ground_truth_path),
            sidecar_path=sidecar_path, model_name=model_name,
            total_rows_in_results=len(schema_pass_by_row),
            schema_failed=sum(1 for v in schema_pass_by_row.values() if not v),
            error=f"No ground-truth records found for sidecar_path {sidecar_path!r}.",
            available_sidecar_paths=sorted({r.get("sidecar_path") for r in records}))

    per_column = defaultdict(lambda: {"correct": 0, "total": 0, "missing": 0})
    per_status = defaultdict(lambda: {"correct": 0, "total": 0, "missing": 0})
    mismatches = []
    false_confidence = []

    for record in page_records:
        expected = _record_to_expected(record)
        if expected is None:
            continue
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

        predicted, predicted_confidence = predicted_by_row_col[key]
        is_correct = _is_correct(predicted, expected)

        per_column[column]["total"] += 1
        per_status[status]["total"] += 1
        if is_correct:
            per_column[column]["correct"] += 1
            per_status[status]["correct"] += 1
        else:
            entry = {
                "row": row_index, "column": column, "status": status,
                "predicted": predicted, "expected": expected,
                "predicted_confidence": predicted_confidence,
            }
            mismatches.append(entry)
            if predicted_confidence == "confirmed":
                false_confidence.append(entry)

    overall_correct = sum(c["correct"] for c in per_column.values())
    overall_total = sum(c["total"] for c in per_column.values())

    return ScoringReport(
        results_path=str(results_path), ground_truth_path=str(ground_truth_path),
        sidecar_path=sidecar_path, model_name=model_name,
        total_rows_in_results=len(schema_pass_by_row),
        schema_failed=sum(1 for v in schema_pass_by_row.values() if not v),
        per_column=dict(per_column), per_status=dict(per_status),
        overall_correct=overall_correct, overall_total=overall_total,
        mismatches=mismatches, false_confidence=false_confidence,
    )


def print_report(report: ScoringReport, show_mismatches: bool = False) -> None:
    """The CLI's rendering of a ScoringReport - kept separate from
    score_results() so the GUI can render the SAME data its own way
    (a table) without inheriting any print()-specific formatting."""
    if report.error:
        print(f"ERROR: {report.error}")
        if report.available_sidecar_paths:
            print("Available sidecar_paths in this log:")
            for p in report.available_sidecar_paths:
                print(f"  {p}")
        return

    print(f"Results file:      {report.results_path}")
    print(f"Ground-truth page: {report.sidecar_path}")
    print(f"Rows in results:   {report.total_rows_in_results} "
          f"({report.schema_failed} failed schema validation)")
    print()

    print("Accuracy by column:")
    for column, counts in sorted(report.per_column.items()):
        acc = counts["correct"] / counts["total"] if counts["total"] else 0.0
        missing_note = f", {counts['missing']} missing" if counts["missing"] else ""
        print(f"  {column}: {counts['correct']}/{counts['total']} ({acc:.0%}){missing_note}")

    print()
    print("Accuracy by ground-truth status (shows whether abstention specifically "
          "is being handled, not just clean-text reading):")
    for status, counts in sorted(report.per_status.items()):
        acc = counts["correct"] / counts["total"] if counts["total"] else 0.0
        missing_note = f", {counts['missing']} missing" if counts["missing"] else ""
        print(f"  {status}: {counts['correct']}/{counts['total']} ({acc:.0%}){missing_note}")

    print()
    print(f"OVERALL: {report.overall_correct}/{report.overall_total} ({report.overall_accuracy:.0%})")

    print()
    if report.false_confidence:
        print(f"FALSE CONFIDENCE: {len(report.false_confidence)}/{len(report.mismatches)} "
              f"wrong answers ({report.false_confidence_rate:.0%}) were tagged 'confirmed' - "
              f"wrong AND falsely certain, not just wrong. This is a materially worse "
              f"failure mode than an ordinary mismatch (see this script's module docstring).")
    else:
        print("FALSE CONFIDENCE: none - every wrong answer was tagged with a lower "
              "confidence than 'confirmed' (or there were no wrong answers).")

    if show_mismatches and report.mismatches:
        print(f"\n{len(report.mismatches)} mismatch(es):")
        for m in report.mismatches:
            flag = " [FALSE CONFIDENCE]" if m["predicted_confidence"] == "confirmed" else ""
            print(f"  row {m['row']} [{m['column']}] ({m['status']}, predicted "
                  f"confidence={m['predicted_confidence']!r}){flag}: "
                  f"predicted {m['predicted']!r} vs expected {m['expected']!r}")


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

    report = score_results(args.results_json, args.ground_truth_log, args.sidecar_path)
    print_report(report, show_mismatches=args.show_mismatches)


if __name__ == "__main__":
    main()
