"""
Generic pairwise conditional-disagreement + confidence/margin analysis
across every fine-tuned vision-tower sensor in the family benchmark, per
Jon's "Decision-Engine Sensor Complementarity Analysis" request
(2026-08-07). Explicitly NOT a model-selection leaderboard - the goal is
characterizing complementarity/disagreement/confidence-usefulness for a
future multi-source decision engine.

Requires diagnostics/extract_vit_family_logits.py to have already run
(reads data/outputs/vit_family_benchmark/<name>/<name>_test_logits.json
for each of the 7 models - full per-image logit vectors, softmax,
margin, entropy, already computed once from the saved checkpoints, no
re-inference here).

For every pair of sensors (21 pairs across 7 models), computes:
  - both correct / A-correct-B-wrong / A-wrong-B-correct / both wrong
  - both wrong SAME predicted label vs. both wrong DIFFERENT labels
  - disagreement rate (any differing prediction, right or wrong)
  - accuracy of A conditional on disagreement, and of B conditional on
    disagreement (the two "who do you trust when they differ" numbers)
  - per-category versions of all of the above, computed for every
    category with n>=5 in that pair's disagreement set; smaller counts
    are still reported but explicitly marked UNRELIABLE, never hidden

Confidence/margin analysis, per model, grouped by: correct / incorrect /
agreement-with-each-other-model / disagreement-with-each-other-model /
"this-model-right-other-wrong" / "this-model-wrong-other-right":
  - top1 logit, top1/top2 margin, top1 softmax probability, entropy

Specific test requested: does MobileNet's margin/confidence carry signal
about the rare cases where it's right and SigLIP is wrong (i.e., could a
margin threshold flag "trust MobileNet here, override SigLIP")? Compared
directly against the margin distribution when MobileNet is wrong (SigLIP
right) - if the "should override" cases don't show a distinguishably
different margin than ordinary wrong predictions, there's no cheap
confidence-based override rule available, not just a hypothesis to state
without checking.

Also writes the literal list of MobileNet-correct/SigLIP-wrong and
SigLIP-correct/MobileNet-wrong file paths (not just counts) so the
actual images can be inspected.

Usage:
    python diagnostics/analyze_sensor_pairwise_conditional.py
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BENCHMARK_ROOT = PROJECT_ROOT / "data" / "outputs" / "vit_family_benchmark"
REPORT_PATH = PROJECT_ROOT / "data" / "logs" / "reviewed" / "sensor_pairwise_conditional_analysis.txt"
DETAIL_JSON_PATH = PROJECT_ROOT / "data" / "logs" / "reviewed" / "sensor_pairwise_conditional_analysis.json"
SIGLIP_MOBILENET_CASES_PATH = PROJECT_ROOT / "data" / "logs" / "reviewed" / "siglip_vs_mobilenet_disagreement_cases.txt"

MODEL_NAMES = ["convnext", "mobilenetv2", "dinov2", "beit", "swin", "siglip", "vit21k"]
MIN_RELIABLE_N = 5


def load_all_logits() -> dict[str, dict]:
    data = {}
    for name in MODEL_NAMES:
        path = BENCHMARK_ROOT / name / f"{name}_test_logits.json"
        if not path.exists():
            print(f"WARNING: {path} missing, skipping {name}")
            continue
        data[name] = json.loads(path.read_text(encoding="utf-8"))
    return data


def common_paths(data: dict[str, dict]) -> list[str]:
    sets = [set(data[n]["predictions"].keys()) for n in data]
    common = set.intersection(*sets) if sets else set()
    return sorted(common)


def pair_conditional_stats(data, a, b, paths, category_filter=None) -> dict:
    pa, pb = data[a]["predictions"], data[b]["predictions"]
    if category_filter:
        paths = [p for p in paths if pa[p]["gt"] == category_filter]
    n = len(paths)
    if n == 0:
        return {"n": 0}

    both_correct = a_correct_b_wrong = a_wrong_b_correct = both_wrong = 0
    both_wrong_same = both_wrong_diff = 0
    disagree = 0
    a_correct_on_disagree = b_correct_on_disagree = 0

    for p in paths:
        ca, cb = pa[p]["correct"], pb[p]["correct"]
        preda, predb = pa[p]["pred"], pb[p]["pred"]
        if ca and cb:
            both_correct += 1
        elif ca and not cb:
            a_correct_b_wrong += 1
        elif not ca and cb:
            a_wrong_b_correct += 1
        else:
            both_wrong += 1
            if preda == predb:
                both_wrong_same += 1
            else:
                both_wrong_diff += 1
        if preda != predb:
            disagree += 1
            if ca:
                a_correct_on_disagree += 1
            if cb:
                b_correct_on_disagree += 1

    return {
        "n": n,
        "reliable": n >= MIN_RELIABLE_N,
        "both_correct": both_correct,
        "a_correct_b_wrong": a_correct_b_wrong,
        "a_wrong_b_correct": a_wrong_b_correct,
        "both_wrong": both_wrong,
        "both_wrong_same_label": both_wrong_same,
        "both_wrong_diff_label": both_wrong_diff,
        "disagreement_rate": disagree / n,
        "disagreement_n": disagree,
        "a_acc_given_disagree": (a_correct_on_disagree / disagree) if disagree else None,
        "b_acc_given_disagree": (b_correct_on_disagree / disagree) if disagree else None,
    }


def write_generic_pairwise(data, paths, classes, lines):
    def log(s=""):
        lines.append(s)
        print(s)

    log("=" * 100)
    log("GENERIC PAIRWISE CONDITIONAL-DISAGREEMENT ANALYSIS "
        f"(n={len(paths)} common held-out images across all {len(data)} sensors)")
    log("=" * 100)

    names = list(data.keys())
    all_pair_stats = {}
    for a, b in combinations(names, 2):
        stats = pair_conditional_stats(data, a, b, paths)
        all_pair_stats[f"{a}|{b}"] = stats
        log(f"\n-- {a} vs {b} (n={stats['n']}) --")
        log(f"  both_correct={stats['both_correct']}  "
            f"{a}_correct/{b}_wrong={stats['a_correct_b_wrong']}  "
            f"{a}_wrong/{b}_correct={stats['a_wrong_b_correct']}  "
            f"both_wrong={stats['both_wrong']} "
            f"(same_label={stats['both_wrong_same_label']}, diff_label={stats['both_wrong_diff_label']})")
        log(f"  disagreement_rate={stats['disagreement_rate']:.3f} (n={stats['disagreement_n']})  "
            f"{a}_acc_given_disagree={stats['a_acc_given_disagree']:.3f}  "
            f"{b}_acc_given_disagree={stats['b_acc_given_disagree']:.3f}"
            if stats["disagreement_n"] else "  no disagreement cases")

        # per-category, every category, tiny-N explicitly marked not hidden
        cat_stats = {}
        for cat in classes:
            cs = pair_conditional_stats(data, a, b, paths, category_filter=cat)
            if cs["n"] == 0:
                continue
            cat_stats[cat] = cs
            reliability = "" if cs["reliable"] else "  [UNRELIABLE - n<5]"
            dr = cs["disagreement_rate"]
            log(f"    {cat:<20s} n={cs['n']:<4d} disagree_rate={dr:.3f}{reliability}")
        all_pair_stats[f"{a}|{b}"]["per_category"] = cat_stats

    return all_pair_stats


def confidence_margin_analysis(data, paths, lines):
    def log(s=""):
        lines.append(s)
        print(s)

    log("\n\n" + "=" * 100)
    log("CONFIDENCE / MARGIN / ENTROPY ANALYSIS (per model)")
    log("=" * 100)
    log("Do NOT assume softmax probability is calibrated confidence - raw "
        "top1/top2 logit margin and entropy are reported alongside it, "
        "not as a replacement metric but as an independent cross-check.\n")

    for name, d in data.items():
        preds = d["predictions"]
        correct_margins, incorrect_margins = [], []
        correct_softmax, incorrect_softmax = [], []
        correct_entropy, incorrect_entropy = [], []
        for p in paths:
            rec = preds[p]
            if rec["correct"]:
                correct_margins.append(rec["top1_top2_margin"])
                correct_softmax.append(rec["top1_softmax"])
                correct_entropy.append(rec["entropy"])
            else:
                incorrect_margins.append(rec["top1_top2_margin"])
                incorrect_softmax.append(rec["top1_softmax"])
                incorrect_entropy.append(rec["entropy"])

        def stat_line(label, vals):
            if not vals:
                return f"    {label}: n=0"
            return (f"    {label}: n={len(vals)}  mean={statistics.mean(vals):.3f}  "
                    f"median={statistics.median(vals):.3f}  "
                    f"stdev={(statistics.stdev(vals) if len(vals) > 1 else 0):.3f}")

        log(f"[{name}]")
        log(f"  Margin (top1-top2 logit):")
        log(stat_line("correct", correct_margins))
        log(stat_line("incorrect", incorrect_margins))
        log(f"  Top-1 softmax probability:")
        log(stat_line("correct", correct_softmax))
        log(stat_line("incorrect", incorrect_softmax))
        log(f"  Entropy:")
        log(stat_line("correct", correct_entropy))
        log(stat_line("incorrect", incorrect_entropy))
        log("")


def siglip_vs_mobilenet_deep_dive(data, paths, lines):
    def log(s=""):
        lines.append(s)
        print(s)

    if "siglip" not in data or "mobilenetv2" not in data:
        log("siglip or mobilenetv2 logits missing, skipping deep dive.")
        return

    sig, mob = data["siglip"]["predictions"], data["mobilenetv2"]["predictions"]

    mob_right_sig_wrong = [p for p in paths if mob[p]["correct"] and not sig[p]["correct"]]
    sig_right_mob_wrong = [p for p in paths if sig[p]["correct"] and not mob[p]["correct"]]

    log("\n\n" + "=" * 100)
    log("DETAILED SIGLIP <-> MOBILENETV2 ANALYSIS")
    log("=" * 100)
    log(f"\nMobileNet correct / SigLIP wrong: {len(mob_right_sig_wrong)} cases")
    log(f"SigLIP correct / MobileNet wrong: {len(sig_right_mob_wrong)} cases")

    case_lines = []
    case_lines.append("=== MobileNet CORRECT, SigLIP WRONG ===")
    for p in mob_right_sig_wrong:
        case_lines.append(
            f"  {p}\n"
            f"    gt={mob[p]['gt']}\n"
            f"    mobilenetv2: pred={mob[p]['pred']} (correct)  "
            f"margin={mob[p]['top1_top2_margin']:.3f}  softmax={mob[p]['top1_softmax']:.3f}  "
            f"entropy={mob[p]['entropy']:.3f}\n"
            f"    siglip:      pred={sig[p]['pred']} (WRONG)  "
            f"margin={sig[p]['top1_top2_margin']:.3f}  softmax={sig[p]['top1_softmax']:.3f}  "
            f"entropy={sig[p]['entropy']:.3f}"
        )
    case_lines.append("\n=== SigLIP CORRECT, MobileNet WRONG ===")
    for p in sig_right_mob_wrong:
        case_lines.append(
            f"  {p}\n"
            f"    gt={sig[p]['gt']}\n"
            f"    siglip:      pred={sig[p]['pred']} (correct)  "
            f"margin={sig[p]['top1_top2_margin']:.3f}  softmax={sig[p]['top1_softmax']:.3f}  "
            f"entropy={sig[p]['entropy']:.3f}\n"
            f"    mobilenetv2: pred={mob[p]['pred']} (WRONG)  "
            f"margin={mob[p]['top1_top2_margin']:.3f}  softmax={mob[p]['top1_softmax']:.3f}  "
            f"entropy={mob[p]['entropy']:.3f}"
        )
    SIGLIP_MOBILENET_CASES_PATH.write_text("\n".join(case_lines), encoding="utf-8")
    log(f"\nFull per-case listing written to: {SIGLIP_MOBILENET_CASES_PATH}")

    # what ground-truth categories do these disagreements cluster in?
    mob_right_cats = Counter(mob[p]["gt"] for p in mob_right_sig_wrong)
    sig_right_cats = Counter(sig[p]["gt"] for p in sig_right_mob_wrong)
    log(f"\nMobileNet-right/SigLIP-wrong, by ground-truth category: {dict(mob_right_cats)}")
    log(f"SigLIP-right/MobileNet-wrong, by ground-truth category: {dict(sig_right_cats)}")

    # what did siglip predict instead, on the cases where it was wrong but mobilenet was right?
    sig_wrong_preds = Counter(sig[p]["pred"] for p in mob_right_sig_wrong)
    log(f"\nWhen SigLIP is wrong but MobileNet is right, SigLIP's mistaken predictions: "
        f"{dict(sig_wrong_preds)}")
    mob_wrong_preds = Counter(mob[p]["pred"] for p in sig_right_mob_wrong)
    log(f"When MobileNet is wrong but SigLIP is right, MobileNet's mistaken predictions: "
        f"{dict(mob_wrong_preds)}")

    # THE key question: does mobilenet's margin distinguish "trust me, override siglip"
    # cases from its ordinary wrong-prediction cases?
    log("\n-- Does MobileNet's own margin/confidence distinguish its RARE-CORRECT-OVERRIDE "
        "cases from its ORDINARY WRONG cases? --")
    mob_wrong_overall = [p for p in paths if not mob[p]["correct"]]
    override_margins = [mob[p]["top1_top2_margin"] for p in mob_right_sig_wrong]
    ordinary_wrong_margins = [mob[p]["top1_top2_margin"] for p in mob_wrong_overall]
    override_softmax = [mob[p]["top1_softmax"] for p in mob_right_sig_wrong]
    ordinary_wrong_softmax = [mob[p]["top1_softmax"] for p in mob_wrong_overall]

    if override_margins:
        log(f"  MobileNet margin when it's the correct override (n={len(override_margins)}): "
            f"mean={statistics.mean(override_margins):.3f}  median={statistics.median(override_margins):.3f}")
    if ordinary_wrong_margins:
        log(f"  MobileNet margin on its OTHER wrong predictions (n={len(ordinary_wrong_margins)}): "
            f"mean={statistics.mean(ordinary_wrong_margins):.3f}  median={statistics.median(ordinary_wrong_margins):.3f}")
    # also compare against mobilenet's margin when it's simply correct in general (baseline)
    mob_correct_overall = [p for p in paths if mob[p]["correct"]]
    general_correct_margins = [mob[p]["top1_top2_margin"] for p in mob_correct_overall]
    if general_correct_margins:
        log(f"  MobileNet margin on ALL its correct predictions (n={len(general_correct_margins)}, "
            f"includes override cases): mean={statistics.mean(general_correct_margins):.3f}  "
            f"median={statistics.median(general_correct_margins):.3f}")

    if override_margins and ordinary_wrong_margins:
        # can a simple threshold separate them? report how many of each group fall
        # above the midpoint between the two medians, as a cheap separability check
        mid = (statistics.median(override_margins) + statistics.median(ordinary_wrong_margins)) / 2
        override_above = sum(1 for m in override_margins if m >= mid)
        wrong_above = sum(1 for m in ordinary_wrong_margins if m >= mid)
        log(f"\n  Simple threshold check at margin={mid:.3f} (midpoint of the two medians):")
        log(f"    override cases above threshold: {override_above}/{len(override_margins)}")
        log(f"    ordinary-wrong cases above threshold: {wrong_above}/{len(ordinary_wrong_margins)}")
        if override_above / len(override_margins) > 0.6 and wrong_above / len(ordinary_wrong_margins) < 0.4:
            log("    -> [B] some separability visible at this small n - worth a larger-scale check")
        else:
            log("    -> [B] no clean separability visible - margin alone likely NOT sufficient "
                "to flag override-worthy cases at this sample size")
    else:
        log("  [B] Sample too small (n="
            f"{len(override_margins)} override cases) to draw any conclusion about "
            "margin-based override detection - this needs a larger disagreement set, "
            "not answerable from this benchmark's 332-image held-out split alone.")


def main():
    lines = []
    def log(s=""):
        lines.append(s)
        print(s)

    log("Loading per-model logit extractions...")
    data = load_all_logits()
    log(f"Loaded models: {list(data.keys())}")
    if not data:
        log("No logit files found - run diagnostics/extract_vit_family_logits.py first.")
        return

    classes = data[next(iter(data))]["classes"]
    paths = common_paths(data)
    log(f"Common held-out test images across all loaded models: {len(paths)}\n")

    pair_stats = write_generic_pairwise(data, paths, classes, lines)
    confidence_margin_analysis(data, paths, lines)
    siglip_vs_mobilenet_deep_dive(data, paths, lines)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    DETAIL_JSON_PATH.write_text(json.dumps(pair_stats, indent=2), encoding="utf-8")
    print(f"\n\nReport: {REPORT_PATH}")
    print(f"Pairwise JSON detail: {DETAIL_JSON_PATH}")


if __name__ == "__main__":
    main()
