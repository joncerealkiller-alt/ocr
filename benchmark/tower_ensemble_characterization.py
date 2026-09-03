"""
Tower Ensemble Characterization (2026-08-03).

Pure characterization of the 8-encoder tower-consensus ensemble's
BEHAVIOR, not accuracy. Jon's explicit framing: "the purpose of this
pass is characterization, not optimization... low margin is not the
same thing as low usefulness... determine whether each encoder is
genuinely adding useful information, neutral, or actively harming the
ensemble - this must be answered from evidence, not inferred from
margin alone."

WHY NO ACCURACY NUMBERS APPEAR HERE: there is currently no human-
reviewed bucket ground truth anywhere in this project for the live
corpus (data/misclassifications.csv and data/outputs/reviewed_
uncertain.csv don't exist for it; the archived data/reference_
prerefactor/misclassifications.csv has 116 flagged rows but ZERO with
correct_category filled in; data/outputs/ground_truth_log.jsonl is
row-level OCR field values, unrelated to bucket classification).
Every measurement below is an OBJECTIVE PROPERTY OF THE ENSEMBLE,
computable from already-recorded evidence with zero model reruns and
zero ground truth - agreement, uniqueness, stability, margins,
confusability. None of it is a proxy for correctness.

REPLAY, NOT IMPLEMENTATION: the leave-one-out and scenario-removal
numbers here describe what WOULD happen to consensus voting under each
scenario - they are not applied anywhere. "weighted by historical
accuracy" (one of the scenarios originally requested) is NOT computed -
it requires ground truth, which doesn't exist yet. Once real labels
exist, this same per-image vote/margin data (already sitting in
*_tower_consensus.json sidecars) can be replayed against them without
re-running anything here.

Usage:
    python -m benchmark.tower_ensemble_characterization
"""
from __future__ import annotations

import itertools
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from core.vision_embeddings import QUALIFIED_ENCODERS, classify_consensus

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKING_DIR = PROJECT_ROOT / "data" / "working"
ENCODERS = [name for name, _ in QUALIFIED_ENCODERS]

MARGIN_BINS = [(0, 0.01), (0.01, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 1.01)]
MARGIN_BIN_LABELS = ["<0.01", "0.01-0.05", "0.05-0.1", "0.1-0.2", "0.2-0.4", ">0.4"]


def load_all_sidecars(working_dir: Path = WORKING_DIR) -> list[dict]:
    """Every *_tower_consensus.json under working_dir - already-recorded
    evidence, zero model reruns to read this."""
    records = []
    for sidecar in working_dir.glob("*_tower_consensus.json"):
        try:
            records.append(__import__("json").loads(sidecar.read_text(encoding="utf-8")))
        except Exception:
            continue
    return records


def pairwise_agreement(per_image_votes: list[dict]) -> dict[tuple[str, str], float]:
    """[A] For every pair of encoders, the fraction of images (where
    BOTH cast a vote) on which they picked the same bucket. High
    agreement = redundant pair; low agreement = diverse pair."""
    agree = defaultdict(int)
    total = defaultdict(int)
    for votes in per_image_votes:
        for e1, e2 in itertools.combinations(ENCODERS, 2):
            if e1 in votes and e2 in votes:
                total[(e1, e2)] += 1
                if votes[e1] == votes[e2]:
                    agree[(e1, e2)] += 1
    return {pair: (agree[pair] / total[pair] if total[pair] else 0.0) for pair in total}


def vote_uniqueness(per_image_votes: list[dict]) -> dict[str, tuple[int, int]]:
    """[A] For each encoder, how often its vote matches NO other
    encoder's vote on that image (an "outlier" vote, agreeing with
    nobody) - (outlier_count, total_votes) per encoder."""
    outlier = Counter()
    total = Counter()
    for votes in per_image_votes:
        vote_counts = Counter(votes.values())
        for enc, v in votes.items():
            total[enc] += 1
            if vote_counts[v] == 1:
                outlier[enc] += 1
    return {enc: (outlier[enc], total[enc]) for enc in ENCODERS}


def leave_one_out_stability(per_image_votes: list[dict]) -> dict[str, dict]:
    """[A] For each encoder, replay consensus with that encoder's vote
    removed - how often does the WINNING BUCKET change, and how often
    does the CONSENSUS CATEGORY (unanimous/majority/split/complete_
    disagreement) change, versus the real 8-vote outcome. This is the
    "effect of removing each encoder" - a stability/contribution
    measurement, not an accuracy one (no ground truth involved: we're
    comparing 7-vote outcomes to the 8-vote outcome, not to any
    correct answer)."""
    flips_bucket = Counter()
    flips_category = Counter()
    valid = Counter()
    for votes in per_image_votes:
        if len(votes) < 3:
            continue
        vote_list = list(votes.values())
        original_top = Counter(vote_list).most_common(1)[0][0]
        original_cat = classify_consensus(vote_list)
        for enc in votes:
            remaining = [v for e, v in votes.items() if e != enc]
            if not remaining:
                continue
            valid[enc] += 1
            new_top = Counter(remaining).most_common(1)[0][0]
            new_cat = classify_consensus(remaining)
            if new_top != original_top:
                flips_bucket[enc] += 1
            if new_cat != original_cat:
                flips_category[enc] += 1
    return {
        enc: {
            "flips_bucket": flips_bucket[enc], "flips_category": flips_category[enc],
            "n": valid[enc],
        }
        for enc in ENCODERS
    }


def margin_distribution(per_image_margins: list[dict]) -> dict[str, list[float]]:
    """[A] Raw margin list per encoder - avg/median/histogram computed
    from this in main()."""
    margins_by_encoder = defaultdict(list)
    for margins in per_image_margins:
        for enc, m in margins.items():
            margins_by_encoder[enc].append(m)
    return margins_by_encoder


def boundary_ambiguity(records: list[dict]) -> Counter:
    """[A] Corpus-wide, encoder-agnostic (winner, runner-up) bucket-pair
    frequency - which pairs of buckets are most often adjacent/confused,
    system-wide, regardless of which encoder cast the vote."""
    pair_counts = Counter()
    for data in records:
        for t in data["towers"].values():
            if t.get("runner_up"):
                pair = tuple(sorted([t["winner"], t["runner_up"]]))
                pair_counts[pair] += 1
    return pair_counts


def replay_scenarios(per_image_votes: list[dict], per_image_margins: list[dict]) -> dict:
    """[A] Consensus category distribution under each named encoder
    subset/weighting scenario. NOTE: "weighted by historical accuracy"
    is deliberately NOT included - it requires ground truth, which
    doesn't exist yet."""
    subset_scenarios = {
        "baseline (all 8)": ENCODERS,
        "remove MAE": [e for e in ENCODERS if e != "mae"],
        "remove SigLIP (both)": [e for e in ENCODERS if e not in ("siglip_fixed", "naflex_siglip")],
        "remove MAE + SigLIP": [e for e in ENCODERS if e not in ("mae", "siglip_fixed", "naflex_siglip")],
        "reliable-5 only": ["eva02", "dinov2", "swin", "beit", "convnext"],
    }

    results = {}
    for name, subset in subset_scenarios.items():
        cat_counts = Counter()
        for votes in per_image_votes:
            subset_votes = [votes[e] for e in subset if e in votes]
            if len(subset_votes) < 2:
                continue
            cat_counts[classify_consensus(subset_votes)] += 1
        results[name] = cat_counts

    # margin-weighted vote vs simple-majority vote - how often do they
    # disagree on the WINNING bucket? Objective, no ground truth needed.
    diff_from_majority = 0
    n_compared = 0
    for votes, margins in zip(per_image_votes, per_image_margins):
        if len(votes) < 2:
            continue
        majority_winner = Counter(votes.values()).most_common(1)[0][0]
        weighted = defaultdict(float)
        for enc, bucket in votes.items():
            weighted[bucket] += margins.get(enc, 0.0)
        if not weighted:
            continue
        weighted_winner = max(weighted, key=weighted.get)
        n_compared += 1
        if weighted_winner != majority_winner:
            diff_from_majority += 1
    results["_margin_weighted_vs_majority"] = (diff_from_majority, n_compared)

    return results


def main() -> None:
    records = load_all_sidecars()
    n = len(records)
    print(f"Loaded {n} tower-consensus sidecars (data/working/*_tower_consensus.json).\n")

    per_image_votes = [
        {enc: t["winner"] for enc, t in data["towers"].items()} for data in records
    ]
    per_image_margins = [
        {enc: t["margin"] for enc, t in data["towers"].items() if t["margin"] is not None}
        for data in records
    ]

    print("=" * 78)
    print("[A] EVIDENCE - measured properties, no ground truth involved")
    print("=" * 78)

    print("\n--- Pairwise agreement (fraction of images both encoders agree on) ---")
    agreement = pairwise_agreement(per_image_votes)
    header = "".join(f"{e:>15}" for e in ENCODERS)
    print(f"{'':>15}{header}")
    for e1 in ENCODERS:
        row = []
        for e2 in ENCODERS:
            if e1 == e2:
                row.append(f"{'-':>15}")
            else:
                pair = (e1, e2) if (e1, e2) in agreement else (e2, e1)
                row.append(f"{agreement.get(pair, 0):>15.3f}")
        print(f"{e1:>15}" + "".join(row))

    print("\n--- Vote uniqueness (encoder agrees with NO other encoder on that image) ---")
    uniqueness = vote_uniqueness(per_image_votes)
    for enc in ENCODERS:
        outlier, total = uniqueness[enc]
        rate = 100 * outlier / total if total else 0
        print(f"{enc:<15} {outlier:>5}/{total:<6} outlier votes ({rate:.1f}%)")

    print("\n--- Leave-one-out: effect of removing each encoder on consensus outcome ---")
    loo = leave_one_out_stability(per_image_votes)
    print(f"{'encoder':<15}{'flips winning bucket':<26}{'flips consensus category'}")
    for enc in ENCODERS:
        d = loo[enc]
        tb = 100 * d["flips_bucket"] / d["n"] if d["n"] else 0
        cc = 100 * d["flips_category"] / d["n"] if d["n"] else 0
        print(f"{enc:<15}{d['flips_bucket']:>4}/{d['n']:<6}({tb:>5.1f}%)     "
              f"{d['flips_category']:>4}/{d['n']:<6}({cc:>5.1f}%)")

    print("\n--- Margin distribution per encoder ---")
    margins_by_encoder = margin_distribution(per_image_margins)
    bin_header = "".join(f"{b:>12}" for b in MARGIN_BIN_LABELS)
    print(f"{'encoder':<15}{'n':<7}{'avg':<8}{'median':<8}{bin_header}")
    for enc in ENCODERS:
        ms = margins_by_encoder[enc]
        if not ms:
            continue
        avg, med = sum(ms) / len(ms), statistics.median(ms)
        counts = []
        for lo, hi in MARGIN_BINS:
            c = sum(1 for m in ms if lo <= m < hi)
            counts.append(f"{100 * c / len(ms):>11.1f}%")
        print(f"{enc:<15}{len(ms):<7}{avg:<8.4f}{med:<8.4f}" + "".join(counts))

    print("\n--- Boundary ambiguity: most frequent (winner, runner-up) pairs, all encoders ---")
    pair_counts = boundary_ambiguity(records)
    for pair, count in pair_counts.most_common(15):
        print(f"  {pair[0]:<20} <-> {pair[1]:<20} {count}")

    print("\n--- Replay: consensus category distribution per scenario ---")
    scenarios = replay_scenarios(per_image_votes, per_image_margins)
    for name, cat_counts in scenarios.items():
        if name.startswith("_"):
            continue
        total = sum(cat_counts.values())
        parts = ", ".join(f"{k}={v}({100 * v / total:.1f}%)" for k, v in cat_counts.most_common())
        print(f"  {name:<25} n={total:<6} {parts}")

    diff, n_cmp = scenarios["_margin_weighted_vs_majority"]
    print(f"\n  margin-weighted vote vs simple-majority vote: differs on "
          f"{diff}/{n_cmp} images ({100 * diff / n_cmp:.1f}%)")
    print("  (NOTE: 'weighted by historical accuracy' scenario NOT computed - "
          "requires ground truth, which doesn't exist yet.)")

    print(f"\nTotal images characterized: {n}")


if __name__ == "__main__":
    main()
