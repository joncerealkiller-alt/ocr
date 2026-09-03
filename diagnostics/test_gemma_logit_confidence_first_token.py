"""
Direct follow-up to the earlier-tokens trajectory probe (diagnostics/
test_gemma_logit_confidence_trajectory.py's finding: tokens before the
decision token are 100% deterministic format-echo, zero decision
content - the decision genuinely happens exactly at the token we were
already reading). This script tests the deeper version of Jon's
question: does making the ANSWER ITSELF the very first generated token
(no "is_census: " field-label echo to produce first) change anything
about the logit-gap signal's reliability, compared to the current
format where the real decision sits at index 4?

Reuses the EXACT SAME 368-image sample (168 real dense_tabular_rows
positives + 200 negatives, same RANDOM_SEED=42) as diagnostics/
test_gemma_logit_confidence_old_taxonomy.py, so this is a direct,
apples-to-apples comparison against that run's result, not a fresh
uncontrolled sample.

Usage:
    python diagnostics/test_gemma_logit_confidence_first_token.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image

from diagnostics.test_gemma_logit_confidence import (
    load_gemma, build_inputs, find_decision_token_index,
)
from diagnostics.test_gemma_logit_confidence_old_taxonomy import load_sample
from core.taxonomy import load_taxonomy

MAX_NEW_TOKENS = 8  # only need the first token or two
REPORT_PATH = PROJECT_ROOT / "data" / "logs" / "reviewed" / "logit_confidence_first_token_report.txt"


def build_first_token_prompt() -> str:
    taxonomy = load_taxonomy()
    guidance = taxonomy.get_category("dense_tabular_rows").classifier_guidance
    return f"""Is this image a "dense tabular rows" document? Definition: {guidance}

Respond with ONLY the single word yes or no - no punctuation, no label, no other text, nothing else. Your response must start with that word."""


def main():
    prompt = build_first_token_prompt()
    loader = load_gemma()
    system_content = "You are a strict, non-interpretive archival routing classifier."
    gen_kwargs = dict(do_sample=False, use_cache=True,
                       return_dict_in_generate=True, output_scores=True,
                       max_new_tokens=MAX_NEW_TOKENS)
    loader._maybe_add_charset_logits_processor(gen_kwargs)

    sample = load_sample()  # same seeded sample as the old-taxonomy run
    n_pos = sum(1 for r in sample if r["ground_truth_is_dtr"])
    n_neg = len(sample) - n_pos

    lines = []
    def log(s=""):
        lines.append(s)
        print(s)

    log(f"Gate prompt:\n{prompt}\n")
    log(f"Sample: {len(sample)} images ({n_pos} positives, {n_neg} negatives) - "
        f"SAME sample as test_gemma_logit_confidence_old_taxonomy.py\n")

    results = []
    idx_not_zero = 0
    for i, row in enumerate(sample, 1):
        file_path = row["file_path"]
        p = Path(file_path)
        try:
            image = Image.open(p).convert("RGB")
        except Exception as e:
            log(f"[{i}/{len(sample)}] SKIP {file_path}: {type(e).__name__}: {e}")
            continue
        inputs = build_inputs(loader, image, prompt, system_content)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            out = loader.model.generate(**inputs, **gen_kwargs)
        generated = out.sequences[0][input_len:]
        decision_idx = find_decision_token_index(loader, generated)

        gt = row["ground_truth_is_dtr"]
        gt_str = "yes" if gt else "no"

        if decision_idx is None:
            log(f"[{i}/{len(sample)}] {row['category']:<20s} gt={gt_str:>3s}  NOT FOUND - "
                f"raw: {loader.processor.decode(generated, skip_special_tokens=True)!r}")
            results.append({"category": row["category"], "gt": gt, "found": False})
            del image
            continue

        if decision_idx != 0:
            idx_not_zero += 1

        decision_token_id = generated[decision_idx].item()
        decision_text = loader.processor.decode([decision_token_id], skip_special_tokens=True).strip().lower()
        logits = out.scores[decision_idx][0]
        probs = torch.softmax(logits, dim=-1)
        own_prob = probs[decision_token_id].item()
        topk = torch.topk(logits, k=2)
        logit_gap = (topk.values[0] - topk.values[1]).item()

        predicted = decision_text == "yes"
        correct = predicted == gt
        mark = "OK  " if correct else "WRONG"
        idx_note = "" if decision_idx == 0 else f"  [decision at idx={decision_idx}, not 0!]"
        log(f"[{i}/{len(sample)}] {mark} {row['category']:<20s} gt={gt_str:>3s} "
            f"pred={decision_text:>3s}  P={own_prob:.6f}  gap={logit_gap:7.3f}{idx_note}")

        results.append({
            "category": row["category"], "gt": gt, "found": True,
            "decision_text": decision_text, "own_prob": own_prob,
            "logit_gap": logit_gap, "correct": correct, "decision_idx": decision_idx,
        })
        del image
        torch.cuda.empty_cache()

        if i % 25 == 0:
            REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    found = [r for r in results if r["found"]]
    correct_cases = [r for r in found if r["correct"]]
    wrong_cases = [r for r in found if not r["correct"]]

    log(f"\n\n=== SUMMARY ({len(found)}/{len(sample)} decision tokens found) ===")
    log(f"Decision token at index 0 for {len(found) - idx_not_zero}/{len(found)} images "
        f"(non-zero for {idx_not_zero})")
    log(f"Overall accuracy: {len(correct_cases)}/{len(found)} ({100*len(correct_cases)/len(found):.1f}%)")

    tp = sum(1 for r in found if r["gt"] and r["decision_text"] == "yes")
    fn = sum(1 for r in found if r["gt"] and r["decision_text"] == "no")
    tn = sum(1 for r in found if not r["gt"] and r["decision_text"] == "no")
    fp = sum(1 for r in found if not r["gt"] and r["decision_text"] == "yes")
    log(f"True positives: {tp}  False negatives: {fn}  True negatives: {tn}  False positives: {fp}")

    if correct_cases:
        mean_p_correct = sum(r["own_prob"] for r in correct_cases) / len(correct_cases)
        mean_gap_correct = sum(r["logit_gap"] for r in correct_cases) / len(correct_cases)
        log(f"\nCorrect cases (n={len(correct_cases)}): mean P={mean_p_correct:.6f}  "
            f"mean logit_gap={mean_gap_correct:.3f}")
    if wrong_cases:
        mean_p_wrong = sum(r["own_prob"] for r in wrong_cases) / len(wrong_cases)
        mean_gap_wrong = sum(r["logit_gap"] for r in wrong_cases) / len(wrong_cases)
        log(f"WRONG cases (n={len(wrong_cases)}): mean P={mean_p_wrong:.6f}  "
            f"mean logit_gap={mean_gap_wrong:.3f}")

    if correct_cases and wrong_cases:
        import statistics
        min_correct_gap = min(r["logit_gap"] for r in correct_cases)
        max_wrong_gap = max(r["logit_gap"] for r in wrong_cases)
        log(f"\nMin correct-case gap: {min_correct_gap:.3f}   Max wrong-case gap: {max_wrong_gap:.3f}")
        log(f"Clean separation (no overlap): {min_correct_gap > max_wrong_gap}")
        log(f"Correct-case gap median: {statistics.median(r['logit_gap'] for r in correct_cases):.3f}")
        log(f"Wrong-case gap median: {statistics.median(r['logit_gap'] for r in wrong_cases):.3f}")

    log(f"\n--- Comparison point: the earlier (index-4, label-echo format) run got ---")
    log(f"Overall accuracy: 326/368 (88.6%), TP=126 FN=42 TN=200 FP=0")
    log(f"Correct median gap=30.406, Wrong median gap=13.562, overlap: min_correct=0.125 max_wrong=30.625")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(f"\nReport written: {REPORT_PATH}")


if __name__ == "__main__":
    main()
