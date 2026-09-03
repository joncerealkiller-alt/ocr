"""
Scaled-up follow-up to test_gemma_logit_confidence_known_wrong.py - same
mechanism (GATE_CENSUS_PROMPT + decision-token logit extraction), but
against all 48 classified disagreement-review images instead of 5,
per Jon's direction ("we have ground truth for a lot of images, use
them"). Every one of these 48 is confirmed NOT dense_tabular_rows by
real human review (data/logs/reviewed/manual_classification_log.csv),
so ground truth is_census=False for all of them - asking gate_census
against this set is a real, not synthetic, stress test: several of
these images are exactly the confusable class (Gemma's ORIGINAL wide
prompt call was wrong on some of them) this whole session's census/
manifest disambiguation work has been fighting, so wrong "yes" answers
are expected to occur naturally, not need to be manufactured.

Tracks both softmax probability AND raw logit gap (top-1 vs runner-up)
per the first known-wrong run's finding: softmax saturates near 1.0 and
hides resolution that the raw logit gap retains.

Usage:
    python diagnostics/test_gemma_logit_confidence_scaled.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image

from diagnostics.test_gemma_logit_confidence import (
    load_gemma, build_inputs, find_decision_token_index,
)
from diagnostics.test_gemma_prompt_tiering_variants import GATE_CENSUS_PROMPT

LOG_PATH = PROJECT_ROOT / "data" / "logs" / "reviewed" / "manual_classification_log.csv"
MAX_NEW_TOKENS = 40
REPORT_PATH = PROJECT_ROOT / "data" / "logs" / "reviewed" / "logit_confidence_scaled_report.txt"


def load_disagreement_cases() -> list[dict]:
    with open(LOG_PATH, "r", encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r["bucket"] == "tower_consensus_disagreements" and r["verdict"] == "classified"]
    return rows


def main():
    loader = load_gemma()
    system_content = "You are a strict, non-interpretive archival routing classifier."
    gen_kwargs = dict(do_sample=False, use_cache=True,
                       return_dict_in_generate=True, output_scores=True,
                       max_new_tokens=MAX_NEW_TOKENS)
    loader._maybe_add_charset_logits_processor(gen_kwargs)

    cases = load_disagreement_cases()
    results = []
    lines = []

    def log(s=""):
        lines.append(s)
        print(s)

    log(f"Running gate_census against {len(cases)} images, all ground_truth is_census=False\n")

    for i, row in enumerate(cases, 1):
        file_path = row["file_path"]
        p = Path(file_path)
        if not p.exists():
            log(f"[{i}/{len(cases)}] SKIP {file_path}: not found")
            continue
        image = Image.open(p).convert("RGB")
        inputs = build_inputs(loader, image, GATE_CENSUS_PROMPT, system_content)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            out = loader.model.generate(**inputs, **gen_kwargs)
        generated = out.sequences[0][input_len:]
        decision_idx = find_decision_token_index(loader, generated)

        if decision_idx is None:
            log(f"[{i}/{len(cases)}] {row['category_display']}: decision token NOT FOUND")
            results.append({"category": row["category_display"], "found": False})
            del image
            continue

        decision_token_id = generated[decision_idx].item()
        decision_text = loader.processor.decode([decision_token_id], skip_special_tokens=True).strip().lower()
        logits = out.scores[decision_idx][0]
        probs = torch.softmax(logits, dim=-1)
        own_prob = probs[decision_token_id].item()
        topk = torch.topk(logits, k=2)
        logit_gap = (topk.values[0] - topk.values[1]).item()

        correct = decision_text == "no"  # ground truth is always "no" here
        mark = "OK  " if correct else "WRONG"
        log(f"[{i}/{len(cases)}] {mark} {row['category_display']:<28s} "
            f"gate={decision_text:>3s}  P={own_prob:.6f}  gap={logit_gap:7.3f}")

        results.append({
            "category": row["category_display"], "found": True,
            "decision_text": decision_text, "own_prob": own_prob,
            "logit_gap": logit_gap, "correct": correct,
        })
        del image
        torch.cuda.empty_cache()

    found = [r for r in results if r["found"]]
    correct_cases = [r for r in found if r["correct"]]
    wrong_cases = [r for r in found if not r["correct"]]

    log(f"\n\n=== SUMMARY ({len(found)}/{len(cases)} decision tokens found) ===")
    log(f"gate_census correct (said 'no'): {len(correct_cases)}/{len(found)}")
    log(f"gate_census WRONG (said 'yes'):  {len(wrong_cases)}/{len(found)}")

    if correct_cases:
        mean_p_correct = sum(r["own_prob"] for r in correct_cases) / len(correct_cases)
        mean_gap_correct = sum(r["logit_gap"] for r in correct_cases) / len(correct_cases)
        log(f"\nCorrect cases: mean P={mean_p_correct:.6f}  mean logit_gap={mean_gap_correct:.3f}")
    if wrong_cases:
        mean_p_wrong = sum(r["own_prob"] for r in wrong_cases) / len(wrong_cases)
        mean_gap_wrong = sum(r["logit_gap"] for r in wrong_cases) / len(wrong_cases)
        log(f"WRONG cases:   mean P={mean_p_wrong:.6f}  mean logit_gap={mean_gap_wrong:.3f}")
        log(f"\nWrong-case detail:")
        for r in wrong_cases:
            log(f"  {r['category']}: P={r['own_prob']:.6f} gap={r['logit_gap']:.3f}")

    if correct_cases and wrong_cases:
        log(f"\nLogit gap ratio (correct/wrong): {mean_gap_correct/mean_gap_wrong:.2f}x")
        # simple separability check: does every wrong case's gap fall
        # below every correct case's gap, or is there overlap?
        min_correct_gap = min(r["logit_gap"] for r in correct_cases)
        max_wrong_gap = max(r["logit_gap"] for r in wrong_cases)
        log(f"Min correct-case gap: {min_correct_gap:.3f}   Max wrong-case gap: {max_wrong_gap:.3f}")
        log(f"Clean separation (no overlap): {min_correct_gap > max_wrong_gap}")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(f"\nReport written: {REPORT_PATH}")


if __name__ == "__main__":
    main()
