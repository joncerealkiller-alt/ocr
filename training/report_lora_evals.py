"""
Renders training/test_lora_checkpoint.py's append-only JSONL eval log
(default data/outputs/lora_eval_log.jsonl) as a single comparison table -
built 2026-07-28 per Jon's direction, so comparing checkpoints/models
doesn't mean re-reading scrollback from several separate terminal runs.

Reads every line in the log (never just the latest) - each line is one
past eval run, so re-running this script after a new eval appends
naturally shows up without re-running anything else.

Usage:
    python training/report_lora_evals.py
    python training/report_lora_evals.py --log-file data/outputs/lora_eval_log.jsonl
    python training/report_lora_evals.py --sort confident_wrong_rate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SORT_CHOICES = ["timestamp", "overall_rate", "confident_wrong_rate", "model"]


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
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-file", type=str, default="data/outputs/lora_eval_log.jsonl")
    parser.add_argument("--sort", type=str, choices=SORT_CHOICES, default="timestamp",
                         help="Default: timestamp (run order). 'confident_wrong_rate' surfaces "
                              "the safest checkpoints (lowest hallucination rate) first.")
    args = parser.parse_args()

    log_path = Path(args.log_file)
    records = _load_records(log_path)
    if not records:
        print(f"No eval runs logged yet in {log_path} - run training/test_lora_checkpoint.py first.")
        return

    rows = []
    for r in records:
        fm_total = r.get("failure_modes_total", {})
        total = r.get("overall_total", 0)
        rows.append({
            "timestamp": r.get("timestamp", "?"),
            "model": r.get("model", "?"),
            "checkpoint": r.get("checkpoint", "?"),
            "n": total,
            "overall_rate": _rate(r.get("overall_correct", 0), total),
            "confident_wrong_rate": _rate(fm_total.get("confident_wrong", 0), total),
            "honest_hedge_rate": _rate(fm_total.get("honest_hedge", 0), total),
            "incorrectly_silent_rate": _rate(fm_total.get("incorrectly_silent", 0), total),
            "confident_wrong_n": fm_total.get("confident_wrong", 0),
        })

    reverse = args.sort in ("overall_rate",)  # higher overall_rate first; confident_wrong_rate sorts ascending (safest first)
    if args.sort != "timestamp":
        rows.sort(key=lambda row: row[args.sort], reverse=reverse)

    # Checkpoint path shortened to just the trailing epoch_N (or "(base)")
    # for column width - the full path is redundant with the model column
    # for anything produced by this project's own out-dir convention.
    def short_checkpoint(cp: str) -> str:
        if cp.startswith("("):
            return cp
        return Path(cp).name

    headers = ["Model", "Checkpoint", "N", "Overall", "Confident-Wrong", "Honest-Hedge", "Incorrectly-Silent", "Timestamp"]
    col_rows = []
    for row in rows:
        col_rows.append([
            row["model"],
            short_checkpoint(row["checkpoint"]),
            str(row["n"]),
            f"{row['overall_rate']:.0%}",
            f"{row['confident_wrong_rate']:.0%} ({row['confident_wrong_n']})",
            f"{row['honest_hedge_rate']:.0%}",
            f"{row['incorrectly_silent_rate']:.0%}",
            row["timestamp"],
        ])

    widths = [max(len(headers[i]), max((len(r[i]) for r in col_rows), default=0)) for i in range(len(headers))]

    def print_row(cells: list[str]):
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)))

    print(f"LoRA eval comparison ({len(rows)} run(s) from {log_path}, sorted by {args.sort}):\n")
    print_row(headers)
    print_row(["-" * w for w in widths])
    for cells in col_rows:
        print_row(cells)

    print("\nConfident-Wrong = asserted specific, plausible-looking content that was actually "
          "wrong (no self-flagged '?' doubt) - the real hallucination-risk rate, lower is safer. "
          "Honest-Hedge = mismatched but self-flagged uncertain (contains '?', or silent on an "
          "illegible/blank field) - acceptable, not counted against the checkpoint. "
          "Overall = raw exact-match accuracy (case/whitespace-normalized).")


if __name__ == "__main__":
    main()
