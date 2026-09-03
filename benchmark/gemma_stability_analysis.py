"""
Analysis pass for benchmark/gemma_stability_sandbox.py's two-pass
reclassification output. Computes [A] stability, [B] confidence deltas,
[D] tower correlation, [E] human ground truth comparison, [F] logit/
uncertainty stats. [C] (reason-stability 4-way tiering) needs direct
reading of the unstable subset, done separately after this script
identifies which images those are (kept small by construction - most
of the 683 should be stable).

Usage:
    python -m benchmark.gemma_stability_analysis
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SANDBOX_DIR = PROJECT_ROOT / "data" / "outputs" / "gemma_stability_sandbox"
DISAGREEMENTS_CSV = PROJECT_ROOT / "data" / "outputs" / "tower_consensus_audit" / "20260804T132128Z" / "disagreements.csv"
MISCLASSIFICATIONS_CSV = PROJECT_ROOT / "data" / "outputs" / "reference_pipeline_v3" / "misclassifications.csv"


def main() -> None:
    pass1 = {r["file_path"]: r for r in json.loads((SANDBOX_DIR / "pass1.json").read_text(encoding="utf-8"))}
    pass2 = {r["file_path"]: r for r in json.loads((SANDBOX_DIR / "pass2.json").read_text(encoding="utf-8"))}
    common = sorted(set(pass1) & set(pass2))
    print(f"Analyzed: {len(common)} images with both passes")

    with open(DISAGREEMENTS_CSV, newline="", encoding="utf-8") as f:
        tower_disagree = {row["working_path"] for row in csv.DictReader(f)}

    # tower consensus_category per image (for [D])
    tower_category = {}
    with open(DISAGREEMENTS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tower_category[row["working_path"]] = row["consensus_category"]

    with open(MISCLASSIFICATIONS_CSV, newline="", encoding="utf-8") as f:
        human_gt = {row["file_path"]: row.get("correct_category", "").strip() for row in csv.DictReader(f)}

    # -- [A] Stability --------------------------------------------------
    stable, unstable = [], []
    for fp in common:
        p1, p2 = pass1[fp], pass2[fp]
        if p1["category"] == p2["category"]:
            stable.append(fp)
        else:
            unstable.append(fp)

    print("\n=== [A] Stability Summary ===")
    print(f"Identical category (stable): {len(stable)}/{len(common)} ({100*len(stable)/len(common):.1f}%)")
    print(f"Changed category (unstable): {len(unstable)}/{len(common)} ({100*len(unstable)/len(common):.1f}%)")

    # -- [B] Confidence analysis -----------------------------------------
    deltas = []
    for fp in common:
        p1, p2 = pass1[fp], pass2[fp]
        if p1["confidence"] is not None and p2["confidence"] is not None:
            deltas.append(p2["confidence"] - p1["confidence"])
    stable_deltas = [pass2[fp]["confidence"] - pass1[fp]["confidence"] for fp in stable
                     if pass1[fp]["confidence"] is not None and pass2[fp]["confidence"] is not None]
    unstable_deltas = [pass2[fp]["confidence"] - pass1[fp]["confidence"] for fp in unstable
                       if pass1[fp]["confidence"] is not None and pass2[fp]["confidence"] is not None]

    print("\n=== [B] Confidence Analysis ===")
    print(f"All pairs (n={len(deltas)}): mean_delta={statistics.mean(deltas):+.4f} "
          f"median_delta={statistics.median(deltas):+.4f} "
          f"variance={statistics.variance(deltas) if len(deltas)>1 else 0:.6f}")
    if stable_deltas:
        print(f"Stable-category pairs (n={len(stable_deltas)}): mean_delta={statistics.mean(stable_deltas):+.4f}")
    if unstable_deltas:
        print(f"Unstable-category pairs (n={len(unstable_deltas)}): mean_delta={statistics.mean(unstable_deltas):+.4f}")

    # -- [D] Tower correlation --------------------------------------------
    print("\n=== [D] Tower Correlation ===")
    unstable_with_tower = [fp for fp in unstable if fp in tower_category]
    stable_with_tower = [fp for fp in stable if fp in tower_category]
    print(f"Unstable images with tower evidence: {len(unstable_with_tower)}/{len(unstable)}")
    from collections import Counter
    unstable_tower_dist = Counter(tower_category[fp] for fp in unstable_with_tower)
    stable_tower_dist = Counter(tower_category[fp] for fp in stable_with_tower)
    print("Tower consensus_category distribution, UNSTABLE Gemma decisions:")
    for cat, n in unstable_tower_dist.most_common():
        print(f"  {cat:<25} {n} ({100*n/len(unstable_with_tower):.1f}%)")
    print("Tower consensus_category distribution, STABLE Gemma decisions (for comparison):")
    for cat, n in stable_tower_dist.most_common():
        print(f"  {cat:<25} {n} ({100*n/len(stable_with_tower):.1f}%)")

    all_disagree_rate = sum(1 for fp in common if fp in tower_disagree) / len(common)
    unstable_disagree_rate = sum(1 for fp in unstable if fp in tower_disagree) / len(unstable) if unstable else 0
    stable_disagree_rate = sum(1 for fp in stable if fp in tower_disagree) / len(stable) if stable else 0
    print(f"\nTower disagreement rate: all candidates={100*all_disagree_rate:.1f}%  "
          f"unstable={100*unstable_disagree_rate:.1f}%  stable={100*stable_disagree_rate:.1f}%")

    # -- [E] Human ground truth --------------------------------------------
    print("\n=== [E] Human Ground Truth ===")
    labeled = [fp for fp in common if human_gt.get(fp) and human_gt[fp] not in ("ignored", "needs_new_bucket", "bad_deskew")]
    print(f"Candidates with a real human label: {len(labeled)}/{len(common)}")
    for fp in labeled:
        p1, p2 = pass1[fp], pass2[fp]
        gt = human_gt[fp]
        p1_match = "MATCH" if p1["category"] == gt else "miss"
        p2_match = "MATCH" if p2["category"] == gt else "miss"
        stability = "STABLE" if p1["category"] == p2["category"] else "UNSTABLE"
        print(f"  {Path(fp).name:<45} gt={gt:<20} pass1={p1['category']:<20}({p1_match})  "
              f"pass2={p2['category']:<20}({p2_match})  [{stability}]")

    # -- [F] Logit-based pre-reasoning confidence -------------------------
    print("\n=== [F] Pre-Reasoning Confidence (logit-based) ===")
    stable_top1 = [pass1[fp]["logit_stats"]["top1_prob"] for fp in stable
                   if pass1[fp].get("logit_stats") and pass2[fp].get("logit_stats")]
    unstable_top1_p1 = [pass1[fp]["logit_stats"]["top1_prob"] for fp in unstable if pass1[fp].get("logit_stats")]
    unstable_top1_p2 = [pass2[fp]["logit_stats"]["top1_prob"] for fp in unstable if pass2[fp].get("logit_stats")]
    stable_margin = [pass1[fp]["logit_stats"]["margin"] for fp in stable if pass1[fp].get("logit_stats")]
    unstable_margin_p1 = [pass1[fp]["logit_stats"]["margin"] for fp in unstable if pass1[fp].get("logit_stats")]
    stable_entropy = [pass1[fp]["logit_stats"]["entropy"] for fp in stable if pass1[fp].get("logit_stats")]
    unstable_entropy_p1 = [pass1[fp]["logit_stats"]["entropy"] for fp in unstable if pass1[fp].get("logit_stats")]

    n_with_logits = sum(1 for fp in common if pass1[fp].get("logit_stats"))
    print(f"Images with logit_stats captured: {n_with_logits}/{len(common)}")
    if stable_top1:
        print(f"STABLE   top1_prob: mean={statistics.mean(stable_top1):.4f} min={min(stable_top1):.4f}")
    if unstable_top1_p1:
        print(f"UNSTABLE top1_prob (pass1): mean={statistics.mean(unstable_top1_p1):.4f} min={min(unstable_top1_p1):.4f}")
    if unstable_top1_p2:
        print(f"UNSTABLE top1_prob (pass2): mean={statistics.mean(unstable_top1_p2):.4f} min={min(unstable_top1_p2):.4f}")
    if stable_margin:
        print(f"STABLE   margin: mean={statistics.mean(stable_margin):.4f}")
    if unstable_margin_p1:
        print(f"UNSTABLE margin (pass1): mean={statistics.mean(unstable_margin_p1):.4f}")
    if stable_entropy:
        print(f"STABLE   entropy: mean={statistics.mean(stable_entropy):.6f}")
    if unstable_entropy_p1:
        print(f"UNSTABLE entropy (pass1): mean={statistics.mean(unstable_entropy_p1):.6f}")

    # Write unstable set for the [C] reason-stability manual read
    out_path = SANDBOX_DIR / "unstable_pairs.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["file_path", "pass1_category", "pass2_category", "pass1_confidence", "pass2_confidence",
                      "pass1_reason", "pass2_reason", "pass1_top1_prob", "pass2_top1_prob",
                      "tower_category", "tower_disagrees", "human_ground_truth"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for fp in unstable:
            p1, p2 = pass1[fp], pass2[fp]
            writer.writerow({
                "file_path": fp,
                "pass1_category": p1["category"], "pass2_category": p2["category"],
                "pass1_confidence": p1["confidence"], "pass2_confidence": p2["confidence"],
                "pass1_reason": p1["reason"], "pass2_reason": p2["reason"],
                "pass1_top1_prob": p1["logit_stats"]["top1_prob"] if p1.get("logit_stats") else "",
                "pass2_top1_prob": p2["logit_stats"]["top1_prob"] if p2.get("logit_stats") else "",
                "tower_category": tower_category.get(fp, ""),
                "tower_disagrees": fp in tower_disagree,
                "human_ground_truth": human_gt.get(fp, ""),
            })
    print(f"\nUnstable pairs (for [C] manual reason-stability read) written to {out_path}")


if __name__ == "__main__":
    main()
