"""
Benchmark 2.4 - Selective E4B Escalation Analysis.

Purely observational - reads only already-frozen data from Benchmark
2.3's multi-tower audit and E2B-vs-E4B comparison. No new model calls,
no corpus changes, no tower recomputation, no consensus recomputation,
nothing written to production manifests. This script does not call
loader.classify() or load any model at all.

Question: does invoking E4B only on a targeted subset (where the
towers already disagree among themselves AND E2B disagrees with their
consensus) capture most of E4B's benefit while avoiding most of its
runtime cost, versus running E4B on every image?

Escalation criterion (both conditions, exactly as specified):
  1. Multi-tower consensus is not unanimous (consensus_category != "unanimous")
  2. E2B disagrees with the tower-majority prediction
     (gemma_agrees_with_tower_consensus == False, i.e. E2B's bucket !=
     tower_consensus_bucket)

Usage:
    python -m benchmark.benchmark2_4_selective_e4b_escalation
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MULTI_TOWER = PROJECT_ROOT / "data" / "outputs" / "benchmark2_3_multi_tower" / "all_results.json"
E2B_VS_E4B_RESULTS = PROJECT_ROOT / "data" / "outputs" / "benchmark2_3_e2b_vs_e4b" / "fresh_results.json"
E2B_VS_E4B_SUMMARY = PROJECT_ROOT / "data" / "outputs" / "benchmark2_3_e2b_vs_e4b" / "summary.json"
OUT_DIR = PROJECT_ROOT / "data" / "outputs" / "benchmark2_4_selective_escalation"

TOWER_NAMES = ["dinov2", "convnext", "naflex_siglip", "siglip_fixed", "eva02", "beit", "swin", "mae"]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    multi = json.loads(MULTI_TOWER.read_text(encoding="utf-8"))
    multi_by_image = {r["image"]: r for r in multi}
    comparison = json.loads(E2B_VS_E4B_RESULTS.read_text(encoding="utf-8"))
    e2b_fresh = comparison["gemma"]
    e4b_fresh = comparison["gemma_e4b"]
    timing = json.loads(E2B_VS_E4B_SUMMARY.read_text(encoding="utf-8"))["timing"]

    n_total = len(multi)
    assert n_total == len(e2b_fresh) == len(e4b_fresh), "corpus size mismatch between frozen files"

    # --- escalation selection: both conditions, verified separately not assumed equivalent ---
    escalated = []
    cond1_only_count = 0  # non-unanimous but E2B DID agree with consensus (not escalated)
    for image, rec in multi_by_image.items():
        cond1 = rec["consensus_category"] != "unanimous"
        cond2 = not rec["gemma_agrees_with_tower_consensus"]
        if cond1 and not cond2:
            cond1_only_count += 1
        if cond1 and cond2:
            escalated.append(image)

    # Check whether condition 2 alone would have produced the same set
    # (a real, checked property, not assumed): cond2-only set.
    cond2_only_set = {img for img, rec in multi_by_image.items() if not rec["gemma_agrees_with_tower_consensus"]}
    sets_identical = set(escalated) == cond2_only_set

    print(f"Total corpus: {n_total}")
    print(f"Condition 1 (non-unanimous) images where E2B still agreed with consensus (not escalated): {cond1_only_count}")
    print(f"Escalated (both conditions): {len(escalated)}")
    print(f"Condition 2 alone (E2B disagrees with tower consensus) would select: {len(cond2_only_set)}")
    print(f"Are the two selection rules identical in practice? {sets_identical}\n")

    # --- per-image escalation records ---
    escalation_records = []
    toward_consensus, still_disagrees = 0, 0
    for image in escalated:
        rec = multi_by_image[image]
        tower_votes = [rec["towers"][t]["bucket"] for t in TOWER_NAMES]
        vote_distribution = dict(Counter(tower_votes))
        e2b = e2b_fresh[image]
        e4b = e4b_fresh[image]
        tower_consensus_bucket = rec["tower_consensus_bucket"]
        e4b_matches_consensus = e4b["category"] == tower_consensus_bucket
        e4b_changed_from_e2b = e4b["category"] != e2b["category"]
        if e4b_matches_consensus:
            toward_consensus += 1
            movement = "toward_consensus"
        else:
            still_disagrees += 1
            movement = "still_disagrees_different_answer" if e4b_changed_from_e2b else "still_disagrees_unchanged"

        escalation_records.append({
            "image": image,
            "sample_bucket": rec["sample_bucket"],
            "tower_vote_distribution": vote_distribution,
            "consensus_category": rec["consensus_category"],
            "tower_consensus_bucket": tower_consensus_bucket,
            "tower_consensus_strength": rec["tower_consensus_strength"],
            "e2b_prediction": e2b["category"], "e2b_confidence": e2b["confidence"],
            "e4b_prediction": e4b["category"], "e4b_confidence": e4b["confidence"],
            "movement": movement,
        })

    with open(OUT_DIR / "escalation_records.json", "w", encoding="utf-8") as f:
        json.dump(escalation_records, f, indent=2)

    # --- summary stats ---
    n_escalated = len(escalated)
    pct_escalated = n_escalated / n_total

    # "regression" relative to the tower-consensus proxy: E2B matched
    # consensus but E4B (if it were run on this image) would not. By
    # construction, every escalated image already has E2B disagreeing
    # with consensus, so this is structurally impossible WITHIN the
    # escalated set - verified directly, not assumed, by checking no
    # escalated record has e2b matching tower_consensus_bucket.
    e2b_matches_consensus_within_escalated = sum(
        1 for r in escalation_records if r["e2b_prediction"] == r["tower_consensus_bucket"]
    )

    e2b_mean = timing["gemma"]["mean_seconds_per_image"]
    e4b_mean = timing["gemma_e4b"]["mean_seconds_per_image"]

    always_e2b_total = n_total * e2b_mean
    always_e4b_total = n_total * e4b_mean
    # Second-stage-reviewer model: E2B always runs, E4B is an ADDITIONAL
    # pass only on the escalated subset (not a replacement) - matches
    # "targeted second-stage reviewer" framing, not "instead of".
    selective_total = n_total * e2b_mean + n_escalated * e4b_mean

    reduction_vs_always_e4b = 1 - (selective_total / always_e4b_total)

    print(f"=== Escalation set: {n_escalated}/{n_total} ({pct_escalated:.1%}) ===")
    print(f"E4B moves an escalated image to match tower consensus (correction captured): {toward_consensus}")
    print(f"E4B still disagrees with tower consensus after escalation: {still_disagrees}")
    print(f"'Regressions' (E2B matched consensus, would be escalated anyway) - structurally "
          f"impossible by this rule's own gate: {e2b_matches_consensus_within_escalated} "
          f"(should be 0, confirming the gate's own safety property relative to the tower-consensus proxy)\n")

    print("=== Runtime projection (E2B/E4B classification cost only, not tower cost) ===")
    print(f"  Always E2B:            {always_e2b_total:8.1f}s ({always_e2b_total/60:.1f} min)")
    print(f"  Always E4B:             {always_e4b_total:8.1f}s ({always_e4b_total/60:.1f} min)")
    print(f"  Selective escalation:   {selective_total:8.1f}s ({selective_total/60:.1f} min)")
    print(f"  Reduction vs always-E4B: {reduction_vs_always_e4b:.1%}")
    print(f"  Overhead vs always-E2B: {(selective_total/always_e2b_total - 1):.1%}\n")

    # --- optional: alternative trigger candidates (report only, do not change the benchmark) ---
    print("=== Optional: alternative trigger candidates (observation only) ===")
    # E2B confidence threshold: how many images would a low-confidence-only trigger select?
    for threshold in [0.9, 0.95, 0.97]:
        low_conf_images = [img for img, r in e2b_fresh.items() if r["confidence"] < threshold]
        overlap = len(set(low_conf_images) & set(escalated))
        print(f"  E2B confidence < {threshold}: {len(low_conf_images)} images "
              f"({overlap}/{len(low_conf_images)} overlap with the {n_escalated}-image escalation set)")

    # vote margin: strength of tower consensus (e.g. "3/8" vs "8/8")
    strength_dist = Counter(r["tower_consensus_strength"] for r in escalation_records)
    print(f"  Tower consensus strength distribution within escalated set: {dict(sorted(strength_dist.items()))}")

    # known taxonomy-boundary canary case
    canary = "oocihm.lac_reel_c10264.767.jpg"
    print(f"  Known taxonomy-boundary canary ({canary}) in escalation set: {canary in escalated}")
    print()

    summary = {
        "n_total": n_total,
        "n_escalated": n_escalated,
        "pct_escalated": round(pct_escalated, 4),
        "condition1_and_not_condition2_count": cond1_only_count,
        "condition2_alone_set_size": len(cond2_only_set),
        "escalation_rules_identical_in_practice": sets_identical,
        "e4b_corrections_captured": toward_consensus,
        "e4b_still_disagrees_after_escalation": still_disagrees,
        "regressions_within_escalated_set": e2b_matches_consensus_within_escalated,
        "timing": {
            "e2b_mean_seconds_per_image": e2b_mean,
            "e4b_mean_seconds_per_image": e4b_mean,
            "always_e2b_total_seconds": round(always_e2b_total, 1),
            "always_e4b_total_seconds": round(always_e4b_total, 1),
            "selective_escalation_total_seconds": round(selective_total, 1),
            "reduction_vs_always_e4b": round(reduction_vs_always_e4b, 4),
        },
        "note": "Observational only - no models run, no corpus/tower/consensus/manifest changes.",
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Full results: {OUT_DIR}/")


if __name__ == "__main__":
    main()
