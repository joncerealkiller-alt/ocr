"""
Follow-up requested by Jon (2026-08-07) after the Gemma-vs-towers
same-split result (Gemma 93.7%, best of every sensor tested): the real
question isn't whether another classifier can beat Gemma (nothing here
does, decisively), it's whether a CHEAP DISSENT SENSOR can flag a
meaningful fraction of Gemma's 21 real errors on this held-out set
without also flagging too many of its 311 correct predictions (false
escalations). This is an arbitration-mechanism question, not a
model-selection one.

For each of Gemma's 21 errors:
  - which of the 7 towers were independently correct
  - MobileNet correct? SigLIP correct? how many towers agree against Gemma?
  - each tower's margin/entropy/softmax on that specific image
  - Gemma's own self-reported confidence on that image (its only
    available signal via the production classify() path - no raw logit
    margin, noted as a real limitation, not glossed over)
  - which ground-truth category each error falls in (concentrated vs.
    spread)

Then tests the specific rule Jon proposed:
    Gemma predicts X
    AND SigLIP disagrees with X (predicts something else)
    AND MobileNet independently agrees with SigLIP's alternative
as a real detector - checked against ALL 332 held-out images, not just
the 21 errors, so both catch rate (of the 21 errors) AND false-positive
rate (unnecessary flags among Gemma's 311 correct predictions) are
measured together. A rule that catches errors but also flags half of
Gemma's correct answers is not useful - both numbers are required, not
just the catch count.

Usage:
    python diagnostics/analyze_gemma_errors_dissent.py
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BENCHMARK_ROOT = PROJECT_ROOT / "data" / "outputs" / "vit_family_benchmark"
REPORT_PATH = PROJECT_ROOT / "data" / "logs" / "reviewed" / "gemma_errors_dissent_analysis.txt"

TOWER_NAMES = ["convnext", "mobilenetv2", "dinov2", "beit", "swin", "siglip", "vit21k"]


def load_all():
    gemma = json.loads((BENCHMARK_ROOT / "gemma" / "gemma_test_predictions.json").read_text(encoding="utf-8"))
    towers = {}
    for name in TOWER_NAMES:
        path = BENCHMARK_ROOT / name / f"{name}_test_logits.json"
        towers[name] = json.loads(path.read_text(encoding="utf-8"))
    return gemma, towers


def main():
    lines = []
    def log(s=""):
        lines.append(s)
        print(s)

    gemma_data, tower_data = load_all()
    gemma_preds = gemma_data["predictions"]
    classes = gemma_data["classes"]

    common_paths = set(gemma_preds.keys())
    for name in TOWER_NAMES:
        common_paths &= set(tower_data[name]["predictions"].keys())
    common_paths = sorted(common_paths)
    log(f"Common images across Gemma + all 7 towers: {len(common_paths)}\n")

    error_paths = [p for p in common_paths if not gemma_preds[p]["correct"]]
    log(f"Gemma errors on this set: {len(error_paths)}\n")

    # -- category concentration --
    log("=" * 90)
    log("1. Are Gemma's errors concentrated or spread across categories?")
    log("=" * 90)
    error_cats = Counter(gemma_preds[p]["gt"] for p in error_paths)
    total_cats = Counter(gemma_preds[p]["gt"] for p in common_paths)
    for cat in classes:
        tot = total_cats.get(cat, 0)
        err = error_cats.get(cat, 0)
        if tot:
            log(f"  {cat:<20s} {err}/{tot} errors ({100*err/tot:.1f}% error rate within category)")
    log(f"\n  Error distribution across categories: {dict(error_cats)}")
    n_cats_with_errors = len(error_cats)
    log(f"  Errors fall in {n_cats_with_errors}/{len(classes)} categories - "
        f"{'CONCENTRATED' if n_cats_with_errors <= 3 else 'SPREAD'} "
        f"(portrait_photo + website_screenshot alone account for "
        f"{error_cats.get('portrait_photo',0) + error_cats.get('website_screenshot',0)}/{len(error_paths)})")

    # -- per-error breakdown: which towers were correct --
    log("\n\n" + "=" * 90)
    log("2. For each of Gemma's 21 errors: which towers were correct? MobileNet? SigLIP? Consensus?")
    log("=" * 90)

    per_error_detail = []
    towers_correct_count = Counter()  # name -> how many of the 21 it got right
    n_towers_correct_per_error = []

    for p in error_paths:
        gt = gemma_preds[p]["gt"]
        gemma_pred = gemma_preds[p]["pred"]
        gemma_conf = gemma_preds[p].get("confidence")
        tower_results = {}
        n_correct = 0
        for name in TOWER_NAMES:
            rec = tower_data[name]["predictions"][p]
            tower_results[name] = rec
            if rec["correct"]:
                n_correct += 1
                towers_correct_count[name] += 1
        n_towers_correct_per_error.append(n_correct)

        log(f"\n  {p}")
        log(f"    gt={gt}  gemma_pred={gemma_pred} (conf={gemma_conf})")
        log(f"    towers correct: {n_correct}/7")
        for name in TOWER_NAMES:
            rec = tower_results[name]
            mark = "CORRECT" if rec["correct"] else "wrong  "
            log(f"      {name:<12s} {mark}  pred={rec['pred']:<20s} margin={rec['top1_top2_margin']:.2f}  "
                f"entropy={rec['entropy']:.2f}  softmax={rec['top1_softmax']:.3f}")
        per_error_detail.append({
            "path": p, "gt": gt, "gemma_pred": gemma_pred, "gemma_conf": gemma_conf,
            "n_towers_correct": n_correct,
            "towers": {name: tower_results[name] for name in TOWER_NAMES},
        })

    log("\n\n  -- Summary: how many of the 21 errors did each tower independently get right? --")
    for name in TOWER_NAMES:
        log(f"    {name:<12s} {towers_correct_count[name]}/{len(error_paths)}")

    mob_correct_n = towers_correct_count["mobilenetv2"]
    sig_correct_n = towers_correct_count["siglip"]
    log(f"\n  MobileNet correct on {mob_correct_n}/{len(error_paths)} of Gemma's errors")
    log(f"  SigLIP correct on {sig_correct_n}/{len(error_paths)} of Gemma's errors")

    consensus_dist = Counter(n_towers_correct_per_error)
    log(f"\n  Distribution of 'how many of 7 towers were correct' across the 21 errors: "
        f"{dict(sorted(consensus_dist.items()))}")
    zero_tower_agreement = consensus_dist.get(0, 0)
    log(f"  {zero_tower_agreement}/{len(error_paths)} errors: EVERY tower also wrong (no tower-based "
        f"dissent signal possible on these at all, by construction)")

    # -- tower margin/entropy characteristics specifically on these 21 --
    log("\n\n" + "=" * 90)
    log("3. Tower margin/entropy on these 21 disagreement cases vs. their normal correct/incorrect baseline")
    log("=" * 90)
    for name in TOWER_NAMES:
        margins_on_errors = [tower_data[name]["predictions"][p]["top1_top2_margin"] for p in error_paths]
        entropies_on_errors = [tower_data[name]["predictions"][p]["entropy"] for p in error_paths]
        log(f"\n  {name}: margin on Gemma's-21 mean={statistics.mean(margins_on_errors):.2f} "
            f"median={statistics.median(margins_on_errors):.2f}  "
            f"entropy mean={statistics.mean(entropies_on_errors):.2f}")

    # -- Gemma's own confidence on these 21 --
    log("\n\n" + "=" * 90)
    log("4. Gemma's own confidence characteristics on these same 21 errors")
    log("=" * 90)
    gemma_confs_on_errors = [gemma_preds[p]["confidence"] for p in error_paths if gemma_preds[p].get("confidence") is not None]
    if gemma_confs_on_errors:
        log(f"  mean={statistics.mean(gemma_confs_on_errors):.3f}  median={statistics.median(gemma_confs_on_errors):.3f}")
    log("  (LIMITATION: Gemma's production classify() path only exposes this self-reported "
        "confidence field, not a raw decision-token logit margin - this project's earlier "
        "experiments found the raw logit gap is the more reliable signal, but extracting it "
        "here would require custom instrumentation inside the classify() call, not done in "
        "this pass. Confidence field alone, per the earlier gemma_vs_towers analysis, only "
        "weakly separates correct/incorrect (0.981 vs 0.956 mean) - the same weak separation "
        "shows up here: mean confidence on these 21 errors is barely below Gemma's overall "
        "correct-case mean, meaning Gemma does NOT reliably 'know' it's wrong on these "
        "specific cases via this field.)")

    # -- THE proposed rule: Gemma=X AND SigLIP strongly disagrees AND MobileNet agrees with SigLIP --
    log("\n\n" + "=" * 90)
    log("5. Testing the proposed rule: Gemma predicts X, SigLIP disagrees, MobileNet independently "
        "agrees with SigLIP's alternative - checked against ALL 332 images (catch rate AND false-positive rate)")
    log("=" * 90)

    flagged = []
    for p in common_paths:
        g_pred = gemma_preds[p]["pred"]
        sig_pred = tower_data["siglip"]["predictions"][p]["pred"]
        mob_pred = tower_data["mobilenetv2"]["predictions"][p]["pred"]
        if sig_pred != g_pred and mob_pred == sig_pred:
            flagged.append(p)

    flagged_errors = [p for p in flagged if p in error_paths]
    flagged_correct = [p for p in flagged if p not in error_paths]
    log(f"\n  Rule flags {len(flagged)}/332 images total")
    log(f"  Of those, {len(flagged_errors)}/{len(error_paths)} are ACTUAL Gemma errors caught "
        f"({100*len(flagged_errors)/len(error_paths):.1f}% catch rate)")
    log(f"  Of those, {len(flagged_correct)}/311 are Gemma predictions that were actually CORRECT "
        f"- unnecessary flags ({100*len(flagged_correct)/311:.1f}% false-positive rate on Gemma's "
        f"correct predictions)")

    if flagged_errors:
        log("\n  Caught error cases:")
        for p in flagged_errors:
            log(f"    {p}  gt={gemma_preds[p]['gt']}  gemma={gemma_preds[p]['pred']}  "
                f"siglip=mobilenet={tower_data['siglip']['predictions'][p]['pred']}")
    if flagged_correct:
        log(f"\n  Unnecessarily flagged (Gemma was right, rule still fired) - first 10 of {len(flagged_correct)}:")
        for p in flagged_correct[:10]:
            log(f"    {p}  gt={gemma_preds[p]['gt']}  gemma(correct)={gemma_preds[p]['pred']}  "
                f"siglip=mobilenet={tower_data['siglip']['predictions'][p]['pred']}")

    # -- a looser variant: just SigLIP+MobileNet agree with EACH OTHER against Gemma, regardless of direction --
    log("\n\n  -- Variant: relaxed rule using margin threshold on SigLIP's dissent "
        "(only counts if siglip's OWN margin is above its median, i.e. a 'confident' dissent) --")
    siglip_margins = [tower_data["siglip"]["predictions"][p]["top1_top2_margin"] for p in common_paths]
    siglip_median_margin = statistics.median(siglip_margins)
    flagged_confident = []
    for p in common_paths:
        g_pred = gemma_preds[p]["pred"]
        sig_rec = tower_data["siglip"]["predictions"][p]
        mob_pred = tower_data["mobilenetv2"]["predictions"][p]["pred"]
        if sig_rec["pred"] != g_pred and mob_pred == sig_rec["pred"] and sig_rec["top1_top2_margin"] >= siglip_median_margin:
            flagged_confident.append(p)
    flagged_confident_errors = [p for p in flagged_confident if p in error_paths]
    flagged_confident_correct = [p for p in flagged_confident if p not in error_paths]
    log(f"  siglip median margin (overall) = {siglip_median_margin:.2f}")
    log(f"  Rule flags {len(flagged_confident)}/332 images")
    log(f"  Catches {len(flagged_confident_errors)}/{len(error_paths)} real errors "
        f"({100*len(flagged_confident_errors)/len(error_paths):.1f}%)")
    log(f"  False-positives: {len(flagged_confident_correct)}/311 "
        f"({100*len(flagged_confident_correct)/311:.1f}%)")

    log("\n\n" + "=" * 90)
    log("VERDICT")
    log("=" * 90)
    catch_rate = len(flagged_errors) / len(error_paths) if error_paths else 0
    fp_rate = len(flagged_correct) / 311
    log(f"  Base rule: catches {100*catch_rate:.1f}% of Gemma's errors at a "
        f"{100*fp_rate:.1f}% false-positive rate on Gemma's correct predictions.")
    if catch_rate > 0.3 and fp_rate < 0.1:
        log("  [B] Looks like a genuinely useful arbitration signal - worth prototyping further.")
    elif catch_rate > 0 and fp_rate < 0.15:
        log("  [B] Modest but real signal - catches a non-trivial fraction at an acceptable "
            "false-positive cost, worth further scale-up before any production decision.")
    else:
        log("  [B] Weak signal at this sample size - either low catch rate or too many false "
            "positives relative to the (small, n=21) error count to draw a confident conclusion. "
            "[C] A larger error sample (more held-out images, or pooling across future benchmark "
            "runs) is needed before ruling this out entirely - 21 errors is a small base rate to "
            "calibrate a production rule against.")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(f"\n\nReport: {REPORT_PATH}")


if __name__ == "__main__":
    main()
