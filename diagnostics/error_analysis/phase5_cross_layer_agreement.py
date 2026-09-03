"""
Phase 5 of the Stage-4 error-analysis pipeline (2026-08-08): comparison
FRAMEWORK between vision_encoder / vision_projected / decoder_early /
decoder_mid / decoder_final / gemma_gen (full autoregressive generation,
where available). Produces reusable tables only - per the handoff
prompt's explicit instruction, this phase draws NO conclusions about
whether any location or the decoder specifically is "adding information."
That question is deferred to whoever consumes these tables next (see
docs/GEMMA_HIDDEN_STATE_ERROR_ANALYSIS_STAGE4_RESEARCH.md section 6/7 for
the reasoning these tables are meant to support, not resolve here).

Uses only Phase 1's review_table.csv (already-cached predictions) - no
new inference.

Output (all under data/outputs/error_analysis/phase5_cross_layer_agreement/):
    pairwise_agreement_summary.csv   - agreement rate for every location pair, per split
    per_image_behavior.csv           - one row per image: pred per location + gemma_gen,
                                        n_locations_correct, all_locations_agree, unanimous_correct
    disagreement_sets/<a>_vs_<b>.csv - the actual disagreeing rows for each location pair (test split)

Usage:
    python diagnostics/error_analysis/phase5_cross_layer_agreement.py
"""

from __future__ import annotations

import csv
import itertools
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics.error_analysis.common import LOCATIONS, OUT_DIR

REVIEW_TABLE_PATH = OUT_DIR / "review_table.csv"
PHASE5_DIR = OUT_DIR / "phase5_cross_layer_agreement"
DISAGREEMENT_DIR = PHASE5_DIR / "disagreement_sets"
AGREEMENT_SUMMARY_PATH = PHASE5_DIR / "pairwise_agreement_summary.csv"
PER_IMAGE_BEHAVIOR_PATH = PHASE5_DIR / "per_image_behavior.csv"

# "classifier output" per the handoff prompt's phase 5 framing = the
# hidden-state linear probe's own prediction, already covered by
# LOCATIONS (all 5 ARE classifier outputs, one per hook location) -
# gemma_gen is the separate, non-hidden-state comparison point (full
# autoregressive generation), included wherever available (test split only).
ALL_SOURCES = LOCATIONS + ["gemma_gen"]


def get_pred(row: dict, source: str) -> str | None:
    if source == "gemma_gen":
        return row["gemma_gen_pred"] if row["in_gemma_gen_set"] == "True" else None
    return row[f"pred_{source}"]


def main():
    if not REVIEW_TABLE_PATH.exists():
        raise FileNotFoundError(f"{REVIEW_TABLE_PATH} not found - run phase1_build_review_dataset.py first.")

    PHASE5_DIR.mkdir(parents=True, exist_ok=True)
    DISAGREEMENT_DIR.mkdir(parents=True, exist_ok=True)

    with open(REVIEW_TABLE_PATH, encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    # --- Pairwise agreement summary, per split ---
    agreement_rows = []
    for split in ("train", "val", "test"):
        split_rows = [r for r in all_rows if r["split"] == split]
        for a, b in itertools.combinations(ALL_SOURCES, 2):
            comparable = [r for r in split_rows if get_pred(r, a) is not None and get_pred(r, b) is not None]
            if not comparable:
                continue
            n_agree = sum(1 for r in comparable if get_pred(r, a) == get_pred(r, b))
            agreement_rows.append({
                "split": split, "source_a": a, "source_b": b,
                "n_comparable": len(comparable), "n_agree": n_agree,
                "agreement_rate": round(n_agree / len(comparable), 4),
            })

    with open(AGREEMENT_SUMMARY_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(agreement_rows[0].keys()))
        writer.writeheader()
        writer.writerows(agreement_rows)
    print(f"Wrote {len(agreement_rows)} pairwise agreement rows to {AGREEMENT_SUMMARY_PATH}")

    # --- Per-image behavior table (all splits, all sources where available) ---
    behavior_rows = []
    for r in all_rows:
        preds = {src: get_pred(r, src) for src in ALL_SOURCES}
        available_preds = {src: p for src, p in preds.items() if p is not None}
        n_correct = sum(1 for src, p in available_preds.items() if p == r["gt"])
        n_available = len(available_preds)
        distinct_preds = set(available_preds.values())
        row = {
            "review_id": r["review_id"], "path": r["path"], "split": r["split"], "gt": r["gt"],
            **{f"pred_{src}": preds[src] if preds[src] is not None else "" for src in ALL_SOURCES},
            "n_sources_available": n_available,
            "n_sources_correct": n_correct,
            "all_sources_agree": len(distinct_preds) <= 1,
            "unanimous_correct": (n_correct == n_available and n_available > 0),
            "unanimous_incorrect": (n_correct == 0 and n_available > 0),
        }
        behavior_rows.append(row)

    with open(PER_IMAGE_BEHAVIOR_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(behavior_rows[0].keys()))
        writer.writeheader()
        writer.writerows(behavior_rows)
    print(f"Wrote {len(behavior_rows)} per-image behavior rows to {PER_IMAGE_BEHAVIOR_PATH}")

    # --- Disagreement sets, test split only (where gemma_gen is available
    # and where the taxonomy failure-review work in Phase 2 is scoped) ---
    test_rows = [r for r in all_rows if r["split"] == "test"]
    n_disagreement_files = 0
    for a, b in itertools.combinations(ALL_SOURCES, 2):
        disagreeing = [
            {
                "review_id": r["review_id"], "path": r["path"], "gt": r["gt"],
                f"pred_{a}": get_pred(r, a), f"pred_{b}": get_pred(r, b),
            }
            for r in test_rows
            if get_pred(r, a) is not None and get_pred(r, b) is not None and get_pred(r, a) != get_pred(r, b)
        ]
        if not disagreeing:
            continue
        out_path = DISAGREEMENT_DIR / f"{a}_vs_{b}.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(disagreeing[0].keys()))
            writer.writeheader()
            writer.writerows(disagreeing)
        n_disagreement_files += 1

    print(f"Wrote {n_disagreement_files} disagreement-set files to {DISAGREEMENT_DIR}/")

    # Summary counts only - no interpretation, per this phase's explicit scope.
    n_unanimous_correct = sum(1 for r in behavior_rows if r["split"] == "test" and r["unanimous_correct"])
    n_unanimous_incorrect = sum(1 for r in behavior_rows if r["split"] == "test" and r["unanimous_incorrect"])
    n_test = sum(1 for r in behavior_rows if r["split"] == "test")
    print(f"\nTest split counts (no interpretation - see docs/GEMMA_HIDDEN_STATE_ERROR_ANALYSIS_STAGE4_"
          f"RESEARCH.md section 6 for how these feed the vision-vs-reasoning tests):")
    print(f"  unanimous_correct (all available sources agree AND correct): {n_unanimous_correct}/{n_test}")
    print(f"  unanimous_incorrect (all available sources agree AND wrong): {n_unanimous_incorrect}/{n_test}")


if __name__ == "__main__":
    main()
