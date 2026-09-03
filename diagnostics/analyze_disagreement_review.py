"""
Joins the 50 human-reviewed tower-consensus-disagreement verdicts
(data/logs/reviewed/manual_classification_log.csv, bucket ==
"tower_consensus_disagreements") against each case's full archived
evidence (record.json - Gemma's answer, all 8 tower predictions,
tower_consensus_bucket/strength) to answer the questions
docs/MULTI_SOURCE_VOTING_CLASSIFIER_PROPOSAL.md's experiment design
laid out:

  - When Gemma disagrees with tower consensus, who's actually right?
  - Per-tower accuracy - which individual encoders are reliable?
  - Family-collapsed consensus (CNN-family: convnext/beit/swin;
    SigLIP-family: naflex_siglip/siglip_fixed; independent: dinov2,
    eva02; outlier: mae) vs. the original flat 8-way tower_consensus -
    does collapsing correlated families to one vote each change the
    answer, per the "towers are correlated families, not independent
    votes" correction already recorded in the proposal doc?
  - Does MAE ever uniquely catch the correct answer when the correlated
    majority is wrong, or is its disagreement just noise?

Read-only analysis - does not modify manual_classification_log.csv,
disagreement_review_queue.csv, or any record.json. Writes a report to
data/logs/reviewed/disagreement_analysis_report.txt (human-readable)
and disagreement_analysis.json (machine-readable, one record per case)
so this can be re-run/extended without re-deriving the join.

Usage:
    python diagnostics/analyze_disagreement_review.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LOG_PATH = PROJECT_ROOT / "data" / "logs" / "reviewed" / "manual_classification_log.csv"
QUEUE_PATH = PROJECT_ROOT / "data" / "logs" / "reviewed" / "disagreement_review_queue.csv"
REPORT_TXT = PROJECT_ROOT / "data" / "logs" / "reviewed" / "disagreement_analysis_report.txt"
REPORT_JSON = PROJECT_ROOT / "data" / "logs" / "reviewed" / "disagreement_analysis.json"

# Family grouping per the correction in docs/MULTI_SOURCE_VOTING_
# CLASSIFIER_PROPOSAL.md - "towers are correlated families, not 8
# independent votes," based on the Multi-Tower Routing Audit's own
# pairwise-agreement matrix (docs/BENCHMARK2_3_MULTI_TOWER_ROUTING_
# AUDIT.md): ConvNeXt/BEiT/Swin cluster 0.84-0.89, the two SigLIPs
# cluster 0.82, MAE is a consistent outlier, DINOv2/EVA02 semi-
# independent of both clusters and each other.
TOWER_FAMILIES = {
    "cnn_family": ["convnext", "beit", "swin"],
    "siglip_family": ["naflex_siglip", "siglip_fixed"],
    "dinov2": ["dinov2"],
    "eva02": ["eva02"],
    "mae": ["mae"],
}


def load_human_verdicts() -> dict[str, dict]:
    """file_path -> verdict row, disagreement-queue cases only."""
    with open(LOG_PATH, "r", encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["bucket"] == "tower_consensus_disagreements"]
    return {r["file_path"]: r for r in rows}


def load_queue_index() -> dict[str, dict]:
    """file_path -> {disagreement_id, record_json_path}."""
    with open(QUEUE_PATH, "r", encoding="utf-8", newline="") as f:
        return {r["file_path"]: r for r in csv.DictReader(f)}


def family_vote(record: dict) -> tuple[str, dict]:
    """Collapses the 8 raw tower predictions into up to 5 family votes
    (one per TOWER_FAMILIES group, majority within a multi-member
    family), then takes the plurality bucket among those family votes -
    NOT the same number as tower_consensus_bucket in record.json, which
    was computed as a flat 8-way vote. Returns (winning_bucket,
    per-family winning bucket dict) so both the final answer and the
    intermediate family votes are available for the report."""
    family_buckets: dict[str, str] = {}
    for family_name, members in TOWER_FAMILIES.items():
        member_buckets = [record["towers"][m]["bucket"] for m in members if m in record["towers"]]
        if not member_buckets:
            continue
        # majority within the family; ties broken by Counter's stable
        # first-seen order (members listed in a fixed order above)
        family_buckets[family_name] = Counter(member_buckets).most_common(1)[0][0]

    if not family_buckets:
        return "unknown", family_buckets
    winner = Counter(family_buckets.values()).most_common(1)[0][0]
    return winner, family_buckets


def main() -> None:
    human = load_human_verdicts()
    queue = load_queue_index()

    cases = []
    skipped_needs_bucket = 0
    skipped_no_record = 0

    for file_path, verdict_row in human.items():
        if verdict_row["verdict"] != "classified":
            skipped_needs_bucket += 1
            continue
        q = queue.get(file_path)
        if q is None:
            skipped_no_record += 1
            continue
        record_path = Path(q["record_json_path"])
        if not record_path.exists():
            skipped_no_record += 1
            continue
        record = json.loads(record_path.read_text(encoding="utf-8"))

        human_label = verdict_row["category_id"]
        gemma_bucket = record["gemma"]["bucket"]
        gemma_correct = gemma_bucket == human_label

        tower_consensus_bucket = record["tower_consensus_bucket"]
        tower_consensus_correct = tower_consensus_bucket == human_label

        fam_winner, fam_votes = family_vote(record)
        family_consensus_correct = fam_winner == human_label

        per_tower_correct = {
            name: (t["bucket"] == human_label)
            for name, t in record["towers"].items()
        }

        cases.append({
            "disagreement_id": q["disagreement_id"],
            "file_path": file_path,
            "human_label": human_label,
            "human_subtype": verdict_row.get("subtype_id", ""),
            "gemma_bucket": gemma_bucket,
            "gemma_confidence": record["gemma"].get("confidence"),
            "gemma_correct": gemma_correct,
            "tower_consensus_bucket": tower_consensus_bucket,
            "tower_consensus_strength": record.get("tower_consensus_strength"),
            "tower_consensus_correct": tower_consensus_correct,
            "family_consensus_bucket": fam_winner,
            "family_votes": fam_votes,
            "family_consensus_correct": family_consensus_correct,
            "per_tower_correct": per_tower_correct,
            "per_tower_bucket": {name: t["bucket"] for name, t in record["towers"].items()},
            "consensus_category": record.get("consensus_category"),
        })

    # -- aggregate stats -----------------------------------------------
    n = len(cases)
    gemma_right = sum(c["gemma_correct"] for c in cases)
    tower_right = sum(c["tower_consensus_correct"] for c in cases)
    family_right = sum(c["family_consensus_correct"] for c in cases)
    both_right = sum(c["gemma_correct"] and c["tower_consensus_correct"] for c in cases)
    both_wrong = sum(not c["gemma_correct"] and not c["tower_consensus_correct"] for c in cases)
    only_gemma_right = sum(c["gemma_correct"] and not c["tower_consensus_correct"] for c in cases)
    only_tower_right = sum(c["tower_consensus_correct"] and not c["gemma_correct"] for c in cases)

    per_tower_accuracy = {}
    for name in TOWER_FAMILIES["cnn_family"] + TOWER_FAMILIES["siglip_family"] + ["dinov2", "eva02", "mae"]:
        vals = [c["per_tower_correct"].get(name) for c in cases if name in c["per_tower_correct"]]
        vals = [v for v in vals if v is not None]
        per_tower_accuracy[name] = (sum(vals), len(vals))

    # MAE-uniquely-right: cases where MAE was correct but the winning
    # family-collapsed consensus (excluding MAE) was wrong.
    mae_uniquely_right = 0
    mae_cases_wrong_when_family_wrong = 0
    for c in cases:
        family_wrong = not c["family_consensus_correct"]
        mae_correct = c["per_tower_correct"].get("mae")
        if family_wrong and mae_correct is not None:
            mae_cases_wrong_when_family_wrong += 1
            if mae_correct:
                mae_uniquely_right += 1

    consensus_category_breakdown = Counter(c["consensus_category"] for c in cases)

    # -- write reports ----------------------------------------------------
    REPORT_JSON.write_text(json.dumps(cases, indent=2), encoding="utf-8")

    lines = []
    def log(s=""):
        lines.append(s)
        print(s)

    log("=" * 70)
    log("DISAGREEMENT REVIEW ANALYSIS")
    log("=" * 70)
    log(f"Cases analyzed: {n} (classified verdicts only)")
    log(f"Excluded: {skipped_needs_bucket} needs_bucket verdict(s), "
        f"{skipped_no_record} missing record.json")
    log()
    log("-- Accuracy against human ground truth --")
    log(f"Gemma correct:            {gemma_right}/{n} ({100*gemma_right/n:.1f}%)")
    log(f"Tower consensus (flat 8): {tower_right}/{n} ({100*tower_right/n:.1f}%)")
    log(f"Family consensus (5-way): {family_right}/{n} ({100*family_right/n:.1f}%)")
    log()
    log("-- Where they diverge --")
    log(f"Both right:         {both_right}")
    log(f"Both wrong:         {both_wrong}")
    log(f"Only Gemma right:   {only_gemma_right}")
    log(f"Only towers right:  {only_tower_right}")
    log()
    log("-- Per-tower individual accuracy --")
    for name, (correct, total) in per_tower_accuracy.items():
        pct = 100 * correct / total if total else 0
        log(f"  {name:<15} {correct}/{total} ({pct:.1f}%)")
    log()
    log("-- MAE specifically --")
    log(f"Cases where family-collapsed consensus (excl. MAE) was WRONG: "
        f"{mae_cases_wrong_when_family_wrong}")
    log(f"Of those, MAE alone was CORRECT: {mae_uniquely_right}")
    if mae_cases_wrong_when_family_wrong:
        log(f"  -> MAE uniquely rescued {mae_uniquely_right}/"
            f"{mae_cases_wrong_when_family_wrong} "
            f"({100*mae_uniquely_right/mae_cases_wrong_when_family_wrong:.1f}%) "
            f"of the cases where the correlated majority missed.")
    log()
    log("-- consensus_category distribution (from original audit) --")
    for cat, cnt in consensus_category_breakdown.most_common():
        log(f"  {cat}: {cnt}")
    log()
    log("-- Per-case detail --")
    for c in cases:
        g = "OK" if c["gemma_correct"] else "MISS"
        t = "OK" if c["tower_consensus_correct"] else "MISS"
        f = "OK" if c["family_consensus_correct"] else "MISS"
        log(f"  {c['disagreement_id']}: human={c['human_label']!r} | "
            f"gemma={c['gemma_bucket']!r}[{g}] | "
            f"tower_flat={c['tower_consensus_bucket']!r}[{t}] | "
            f"family={c['family_consensus_bucket']!r}[{f}]")

    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReports written:\n  {REPORT_TXT}\n  {REPORT_JSON}")


if __name__ == "__main__":
    main()
