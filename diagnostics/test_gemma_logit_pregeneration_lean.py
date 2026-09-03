"""
Deeper follow-up to the trajectory probe (diagnostics/test_gemma_logit_
trajectory.py) and the first-token experiment (diagnostics/test_gemma_
logit_confidence_first_token.py). Both established:
  - Tokens 0-3 of the LABEL-ECHO format ("is_census: ") are perfectly
    deterministic (p=1.000000) - the top-1 winner at every earlier
    position is just format-following, not a content decision.
  - Forcing the answer to BE token 0 (stripping the label-echo) made
    both accuracy and the confidence signal WORSE, not better - the
    "boring" echo tokens are functionally useful extra computation
    steps, not decision-irrelevant filler.

Neither of those checked what THIS script checks: at the LAST PROMPT
POSITION (index N-1, i.e. `generate(max_new_tokens=1)`'s scores[0] -
BEFORE any token, including "is", has been generated), what is the RAW
PROBABILITY MASS specifically on yes/no-like tokens, buried under
whatever the actual top-1 winner ("is") is? If the model's underlying
"lean" toward the eventual answer is already detectable in that tail
probability - even though it's nowhere near winning - that would be a
genuinely different, and potentially available with LESS work (a single
forward pass, no multi-step generation needed) signal than the current
decision-token approach.

Uses the SAME label-echo GATE_DENSE_TABULAR_PROMPT and the SAME
368-image sample (seed=42) as test_gemma_logit_confidence_old_
taxonomy.py, for direct comparability.

Usage:
    python diagnostics/test_gemma_logit_pregeneration_lean.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image

from diagnostics.test_gemma_logit_confidence import load_gemma, build_inputs
from diagnostics.test_gemma_logit_confidence_old_taxonomy import load_sample, build_gate_prompt

MAX_NEW_TOKENS = 1  # only need position N-1's logits - generate() does exactly
                     # one forward pass to produce this, nothing more needed
REPORT_PATH = PROJECT_ROOT / "data" / "logs" / "reviewed" / "logit_pregeneration_lean_report.txt"

# Candidate surface forms for "yes"/"no" at this position - checked via
# real tokenizer encoding, not guessed, since leading-space/case variants
# tokenize differently depending on what precedes them.
YES_SURFACE_FORMS = ["yes", " yes", "Yes", " Yes", "YES", " YES"]
NO_SURFACE_FORMS = ["no", " no", "No", " No", "NO", " NO"]


def resolve_token_ids(tokenizer, surface_forms: list[str]) -> set[int]:
    ids = set()
    for s in surface_forms:
        encoded = tokenizer.encode(s, add_special_tokens=False)
        if len(encoded) == 1:
            ids.add(encoded[0])
    return ids


def main():
    prompt = build_gate_prompt()
    loader = load_gemma()
    system_content = "You are a strict, non-interpretive archival routing classifier."
    gen_kwargs = dict(do_sample=False, use_cache=True,
                       return_dict_in_generate=True, output_scores=True,
                       max_new_tokens=MAX_NEW_TOKENS)
    loader._maybe_add_charset_logits_processor(gen_kwargs)

    yes_ids = resolve_token_ids(loader.processor.tokenizer, YES_SURFACE_FORMS)
    no_ids = resolve_token_ids(loader.processor.tokenizer, NO_SURFACE_FORMS)

    sample = load_sample()  # same seeded sample
    lines = []
    def log(s=""):
        lines.append(s)
        print(s)

    log(f"yes-cluster token ids: {yes_ids}")
    log(f"no-cluster token ids: {no_ids}\n")
    log(f"Sample: {len(sample)} images - checking position N-1 (pre-generation) "
        f"yes/no probability mass\n")

    results = []
    for i, row in enumerate(sample, 1):
        file_path = row["file_path"]
        p = Path(file_path)
        try:
            image = Image.open(p).convert("RGB")
        except Exception as e:
            log(f"[{i}/{len(sample)}] SKIP {file_path}: {type(e).__name__}: {e}")
            continue
        inputs = build_inputs(loader, image, prompt, system_content)

        with torch.inference_mode():
            out = loader.model.generate(**inputs, **gen_kwargs)
        # scores[0] = logits at position N-1, computed BEFORE any token
        # (including the format-echo "is") was generated - exactly the
        # position the user's formula describes.
        logits = out.scores[0][0]
        probs = torch.softmax(logits, dim=-1)

        top1_id = torch.argmax(probs).item()
        top1_text = loader.processor.decode([top1_id], skip_special_tokens=True)
        top1_prob = probs[top1_id].item()

        p_yes = sum(probs[tid].item() for tid in yes_ids)
        p_no = sum(probs[tid].item() for tid in no_ids)
        lean = "yes" if p_yes > p_no else ("no" if p_no > p_yes else "tie")
        lean_ratio = (max(p_yes, p_no) / min(p_yes, p_no)) if min(p_yes, p_no) > 0 else float("inf")

        gt = row["ground_truth_is_dtr"]
        gt_str = "yes" if gt else "no"
        lean_correct = lean == gt_str

        mark = "OK  " if lean_correct else "WRONG"
        log(f"[{i}/{len(sample)}] {mark} {row['category']:<20s} gt={gt_str:>3s}  "
            f"top1={top1_text!r}(p={top1_prob:.4f})  "
            f"P(yes-tail)={p_yes:.3e}  P(no-tail)={p_no:.3e}  lean={lean:>3s}  ratio={lean_ratio:.2f}x")

        results.append({
            "category": row["category"], "gt": gt, "top1_text": top1_text,
            "p_yes": p_yes, "p_no": p_no, "lean": lean, "lean_correct": lean_correct,
            "lean_ratio": lean_ratio,
        })
        del image
        torch.cuda.empty_cache()

        if i % 50 == 0:
            REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    n = len(results)
    lean_correct_n = sum(1 for r in results if r["lean_correct"])
    log(f"\n\n=== SUMMARY (n={n}) ===")
    log(f"Pre-generation yes/no LEAN matches ground truth: {lean_correct_n}/{n} "
        f"({100*lean_correct_n/n:.1f}%)")
    log(f"(Random-chance baseline would be ~50%, and the actual gate's real "
        f"decision - after generating the full label-echo + answer - scored "
        f"88.6% overall accuracy in the earlier run, for direct comparison.)")

    correct_ratios = [r["lean_ratio"] for r in results if r["lean_correct"] and r["lean_ratio"] != float("inf")]
    wrong_ratios = [r["lean_ratio"] for r in results if not r["lean_correct"] and r["lean_ratio"] != float("inf")]
    if correct_ratios:
        log(f"\nMean lean ratio when lean matches ground truth: "
            f"{sum(correct_ratios)/len(correct_ratios):.2f}x (n={len(correct_ratios)})")
    if wrong_ratios:
        log(f"Mean lean ratio when lean does NOT match ground truth: "
            f"{sum(wrong_ratios)/len(wrong_ratios):.2f}x (n={len(wrong_ratios)})")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(f"\nReport written: {REPORT_PATH}")


if __name__ == "__main__":
    main()
