"""
Renders benchmark/prompt_sweep.py's append-only JSONL history log
(default data/outputs/prompt_sweep_log.jsonl) as a single comparison
table across EVERY sweep ever run, not just the latest - same role
training/report_lora_evals.py plays for lora_eval_log.jsonl, same
pattern deliberately reused here (_load_records/_rate/widths+print_row)
so the two tools read the same way.

Reads every line in the log every time - a new sweep run just appends,
so re-running this script picks it up with no extra bookkeeping. This
is the piece that answers "was v11 actually better than v15" without
relying on memory or re-reading old terminal scrollback.

Usage:
    python benchmark/prompt_sweep_report.py
    python benchmark/prompt_sweep_report.py --sort correct_rate
    python benchmark/prompt_sweep_report.py --run-id 20260729T120000Z
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SORT_CHOICES = ["timestamp", "correct_rate", "wrong_rate", "abstained_rate"]


def _load_records(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"WARNING: skipping malformed line {i} in {log_path}: {e}")
    return records


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-file", type=str, default="data/outputs/prompt_sweep_log.jsonl")
    parser.add_argument("--sort", type=str, choices=SORT_CHOICES, default="timestamp",
                         help="Default: timestamp (run order). 'wrong_rate' surfaces the "
                              "safest combos (lowest hallucination rate) first.")
    parser.add_argument("--run-id", type=str, default=None,
                         help="Only show combos from this one sweep invocation.")
    args = parser.parse_args()

    log_path = Path(args.log_file)
    records = _load_records(log_path)
    if args.run_id:
        records = [r for r in records if r.get("run_id") == args.run_id]
    if not records:
        print(f"No sweep runs logged yet in {log_path} - run benchmark/prompt_sweep.py first.")
        return

    rows = []
    for r in records:
        fm_total = r.get("failure_modes_total", {})
        total = r.get("overall_total", 0)
        wrong = fm_total.get("confident_wrong", 0)
        abstained = fm_total.get("honest_hedge", 0) + fm_total.get("incorrectly_silent", 0)
        rows.append({
            "timestamp": r.get("timestamp", "?"),
            "run_id": r.get("run_id", "?"),
            "ocr_model": r.get("ocr_model", "?"),
            "ocr_prompt": r.get("ocr_prompt", "?"),
            "structuring_model": r.get("structuring_model", "?"),
            "structuring_prompt": r.get("structuring_prompt", "?"),
            "n": total,
            "correct_rate": _rate(r.get("overall_correct", 0), total),
            "wrong_rate": _rate(wrong, total),
            "wrong_n": wrong,
            "abstained_rate": _rate(abstained, total),
            "avg_total_seconds": r.get("avg_total_seconds", 0.0),
        })

    reverse = args.sort in ("correct_rate",)  # higher correct_rate first; wrong/abstained sort ascending (safest first)
    if args.sort != "timestamp":
        rows.sort(key=lambda row: row[args.sort], reverse=reverse)

    headers = ["OCR Model", "OCR Prompt", "Struct Model", "Struct Prompt", "N",
               "Correct", "Abstained", "Wrong", "Avg Time", "Timestamp"]
    col_rows = []
    for row in rows:
        col_rows.append([
            row["ocr_model"], row["ocr_prompt"], row["structuring_model"], row["structuring_prompt"],
            str(row["n"]), f"{row['correct_rate']:.0%}", f"{row['abstained_rate']:.0%}",
            f"{row['wrong_rate']:.0%} ({row['wrong_n']})", f"{row['avg_total_seconds']:.1f}s",
            row["timestamp"],
        ])

    widths = [max(len(headers[i]), max((len(r[i]) for r in col_rows), default=0)) for i in range(len(headers))]

    def print_row(cells: list[str]):
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)))

    run_note = f", run_id={args.run_id}" if args.run_id else ""
    print(f"Prompt sweep comparison ({len(rows)} combo(s) from {log_path}{run_note}, sorted by {args.sort}):\n")
    print_row(headers)
    print_row(["-" * w for w in widths])
    for cells in col_rows:
        print_row(cells)

    print("\nWrong = confident, specific, false content (no self-flagged '?' doubt) - the real "
          "hallucination-risk rate, lower is safer. Abstained = honest '?' hedge or safe silence "
          "on a real field - not counted against the prompt. Correct = exact match "
          "(case/whitespace-normalized), including honest correct abstention on illegible/blank fields.")


if __name__ == "__main__":
    main()
