"""
Answers the [C] open questions from the Decision-Engine Sensor
Complementarity Analysis (docs/MULTI_SOURCE_VOTING_CLASSIFIER_PROPOSAL.md):
now that Gemma has been run on the EXACT SAME 332-image held-out split as
the 7 fine-tuned vision towers (diagnostics/run_gemma_on_benchmark_
holdout.py, real production prompt/loader), this computes the same
conditional-disagreement metrics as diagnostics/analyze_sensor_pairwise_
conditional.py, Gemma vs. each tower, plus Gemma's self-reported
confidence vs. correctness (Gemma doesn't expose a full logit vector via
the production classify() path the way the towers' checkpoints do, so
this uses its confidence field, not a raw margin).

Usage:
    python diagnostics/analyze_gemma_vs_towers.py
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BENCHMARK_ROOT = PROJECT_ROOT / "data" / "outputs" / "vit_family_benchmark"
REPORT_PATH = PROJECT_ROOT / "data" / "logs" / "reviewed" / "gemma_flat8_vs_towers_analysis.txt"

TOWER_NAMES = ["convnext", "mobilenetv2", "dinov2", "beit", "swin", "siglip", "vit21k"]


def load_predictions(name: str) -> dict | None:
    if name == "gemma":
        path = BENCHMARK_ROOT / "gemma_flat8" / "gemma_flat8_test_predictions.json"
    else:
        path = BENCHMARK_ROOT / name / f"{name}_test_logits.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def pair_stats(gemma_preds, tower_preds, paths):
    both_correct = g_correct_t_wrong = g_wrong_t_correct = both_wrong = 0
    both_wrong_same = both_wrong_diff = 0
    disagree = 0
    g_correct_on_disagree = t_correct_on_disagree = 0
    for p in paths:
        g, t = gemma_preds[p], tower_preds[p]
        gc, tc = g["correct"], t["correct"]
        if gc and tc:
            both_correct += 1
        elif gc and not tc:
            g_correct_t_wrong += 1
        elif not gc and tc:
            g_wrong_t_correct += 1
        else:
            both_wrong += 1
            if g["pred"] == t["pred"]:
                both_wrong_same += 1
            else:
                both_wrong_diff += 1
        if g["pred"] != t["pred"]:
            disagree += 1
            if gc:
                g_correct_on_disagree += 1
            if tc:
                t_correct_on_disagree += 1
    n = len(paths)
    return {
        "n": n, "both_correct": both_correct, "gemma_correct_tower_wrong": g_correct_t_wrong,
        "gemma_wrong_tower_correct": g_wrong_t_correct, "both_wrong": both_wrong,
        "both_wrong_same_label": both_wrong_same, "both_wrong_diff_label": both_wrong_diff,
        "disagreement_rate": disagree / n if n else 0.0, "disagreement_n": disagree,
        "gemma_acc_given_disagree": (g_correct_on_disagree / disagree) if disagree else None,
        "tower_acc_given_disagree": (t_correct_on_disagree / disagree) if disagree else None,
    }


def main():
    lines = []
    def log(s=""):
        lines.append(s)
        print(s)

    gemma_data = load_predictions("gemma")
    if gemma_data is None:
        log("Gemma predictions not found - run diagnostics/run_gemma_on_benchmark_holdout.py first.")
        return
    gemma_preds = gemma_data["predictions"]
    classes = gemma_data["classes"]

    gemma_correct_n = sum(1 for v in gemma_preds.values() if v["correct"])
    log(f"Gemma held-out accuracy (production prompt, this split): "
        f"{gemma_correct_n}/{len(gemma_preds)} ({100*gemma_correct_n/len(gemma_preds):.1f}%)\n")

    all_pair_stats = {}
    log("=" * 100)
    log("GEMMA vs. EACH FINE-TUNED VISION TOWER - same held-out split")
    log("=" * 100)

    for name in TOWER_NAMES:
        tower_data = load_predictions(name)
        if tower_data is None:
            log(f"\n[{name}] predictions missing, skipping")
            continue
        tower_preds = tower_data["predictions"]
        common = sorted(set(gemma_preds.keys()) & set(tower_preds.keys()))
        stats = pair_stats(gemma_preds, tower_preds, common)
        all_pair_stats[name] = stats

        log(f"\n-- gemma vs {name} (n={stats['n']}) --")
        log(f"  both_correct={stats['both_correct']}  "
            f"gemma_correct/{name}_wrong={stats['gemma_correct_tower_wrong']}  "
            f"gemma_wrong/{name}_correct={stats['gemma_wrong_tower_correct']}  "
            f"both_wrong={stats['both_wrong']} "
            f"(same_label={stats['both_wrong_same_label']}, diff_label={stats['both_wrong_diff_label']})")
        if stats["disagreement_n"]:
            log(f"  disagreement_rate={stats['disagreement_rate']:.3f} (n={stats['disagreement_n']})  "
                f"gemma_acc_given_disagree={stats['gemma_acc_given_disagree']:.3f}  "
                f"{name}_acc_given_disagree={stats['tower_acc_given_disagree']:.3f}")
        else:
            log("  no disagreement cases")

        # per-category disagreement, same discipline as the tower-only analysis
        cat_counts = Counter(gemma_preds[p]["gt"] for p in common if gemma_preds[p]["pred"] != tower_preds[p]["pred"])
        cat_totals = Counter(gemma_preds[p]["gt"] for p in common)
        for cat in classes:
            tot = cat_totals.get(cat, 0)
            if tot == 0:
                continue
            dis = cat_counts.get(cat, 0)
            reliability = "" if tot >= 5 else "  [UNRELIABLE - n<5]"
            log(f"    {cat:<20s} n={tot:<4d} disagree_rate={dis/tot:.3f}{reliability}")

    # -- unique-correct: images gemma alone got right vs. images each tower alone got right --
    log("\n\n" + "=" * 100)
    log("UNIQUE-CORRECT: does Gemma catch cases NO tower catches, and vice versa?")
    log("=" * 100)
    tower_data_all = {n: load_predictions(n) for n in TOWER_NAMES}
    tower_data_all = {n: d["predictions"] for n, d in tower_data_all.items() if d is not None}
    common_all = sorted(set(gemma_preds.keys()).intersection(*[set(d.keys()) for d in tower_data_all.values()]))
    log(f"(n={len(common_all)} images common to Gemma + all 7 towers)")

    gemma_unique_correct = [p for p in common_all if gemma_preds[p]["correct"]
                             and all(not tower_data_all[n][p]["correct"] for n in tower_data_all)]
    log(f"\nGemma correct, EVERY tower wrong: {len(gemma_unique_correct)} cases")
    for p in gemma_unique_correct:
        log(f"  {p}  gt={gemma_preds[p]['gt']}  gemma_pred={gemma_preds[p]['pred']}  "
            f"conf={gemma_preds[p].get('confidence')}")

    for n in tower_data_all:
        tower_unique = [p for p in common_all if tower_data_all[n][p]["correct"] and not gemma_preds[p]["correct"]
                         and all(not tower_data_all[m][p]["correct"] for m in tower_data_all if m != n)]
        log(f"\n{n} correct, Gemma AND every other tower wrong: {len(tower_unique)} cases")
        for p in tower_unique:
            log(f"  {p}  gt={tower_data_all[n][p]['gt']}  {n}_pred={tower_data_all[n][p]['pred']}  "
                f"gemma_pred={gemma_preds[p]['pred']}")

    # -- all-agree-wrong: hardest cases, everything fails --
    all_wrong = [p for p in common_all if not gemma_preds[p]["correct"]
                 and all(not tower_data_all[n][p]["correct"] for n in tower_data_all)]
    log(f"\n\nImages where GEMMA AND EVERY TOWER are wrong (hardest cases in this "
        f"benchmark): {len(all_wrong)}")
    for p in all_wrong:
        log(f"  {p}  gt={gemma_preds[p]['gt']}  gemma_pred={gemma_preds[p]['pred']}")

    # -- Gemma confidence vs correctness --
    log("\n\n" + "=" * 100)
    log("GEMMA SELF-REPORTED CONFIDENCE vs. CORRECTNESS (not a raw logit margin - "
        "the production classify() path only exposes this field)")
    log("=" * 100)
    correct_conf = [v["confidence"] for v in gemma_preds.values() if v["correct"] and v.get("confidence") is not None]
    wrong_conf = [v["confidence"] for v in gemma_preds.values() if not v["correct"] and v.get("confidence") is not None]
    if correct_conf:
        log(f"Correct (n={len(correct_conf)}): mean={statistics.mean(correct_conf):.3f}  "
            f"median={statistics.median(correct_conf):.3f}")
    if wrong_conf:
        log(f"Incorrect (n={len(wrong_conf)}): mean={statistics.mean(wrong_conf):.3f}  "
            f"median={statistics.median(wrong_conf):.3f}")
    log("(Consistent with this project's earlier finding that Gemma's self-reported "
        "confidence field is a much weaker signal than its raw decision-token logit "
        "gap - see the 'Gemma comparator' section for that prior result. This "
        "production-path confidence field is what Stage 5 actually has access to "
        "today, though - the logit-gap mechanism would need custom instrumentation "
        "to run inside the production classify() call, not just this benchmark.)")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(f"\n\nReport: {REPORT_PATH}")


if __name__ == "__main__":
    main()
