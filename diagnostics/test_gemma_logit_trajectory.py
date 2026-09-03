"""
Answers Jon's question directly with data instead of theory: with the
current prompt format ("is_census: <yes|no>\\nconfidence: ..."), the
actual decision token ("yes"/"no") sits at a fixed index (4, confirmed
in test_gemma_logit_confidence.py) - but tokens 0-3 are the model
echoing its own field label ("is_census", ":", " ") before ever
reaching the real answer. Do those earlier tokens ALREADY carry
decision-relevant signal (e.g. does the model's distribution at step 0
already lean toward answer-consistent tokens), or are they purely
format-following and boring (~1.0 probability on the one syntactically
required token, regardless of image content)?

Dumps the full top-8 candidates at EVERY generated step from 0 through
a few steps past the decision token, for a handful of images, so this
is answered by looking at real numbers, not assumption.

Usage:
    python diagnostics/test_gemma_logit_trajectory.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image

from diagnostics.test_gemma_logit_confidence import (
    load_gemma, build_inputs, find_decision_token_index, TEST_CASES, WORKING_DIR, gt_is_census,
)
from diagnostics.test_gemma_prompt_tiering_variants import GATE_CENSUS_PROMPT

MAX_NEW_TOKENS = 12  # only need a few steps past the decision token
STEPS_PAST_DECISION = 2


def main():
    loader = load_gemma()
    system_content = "You are a strict, non-interpretive archival routing classifier."
    gen_kwargs = dict(do_sample=False, use_cache=True,
                       return_dict_in_generate=True, output_scores=True,
                       max_new_tokens=MAX_NEW_TOKENS)
    loader._maybe_add_charset_logits_processor(gen_kwargs)

    # One clearly-census and one clearly-not-census image, for contrast.
    sample = [TEST_CASES[0], TEST_CASES[3]]  # (stem, ground_truth) tuples

    for stem, ground_truth in sample:
        img_path = WORKING_DIR / f"{stem}.jpg"
        if not img_path.exists():
            continue
        image = Image.open(img_path).convert("RGB")
        inputs = build_inputs(loader, image, GATE_CENSUS_PROMPT, system_content)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            out = loader.model.generate(**inputs, **gen_kwargs)
        generated = out.sequences[0][input_len:]
        decision_idx = find_decision_token_index(loader, generated)

        print(f"\n{'='*70}")
        print(f"=== {stem} (ground_truth is_census={gt_is_census(ground_truth)}) ===")
        print(f"decision token at index {decision_idx}")
        print(f"{'='*70}")

        end_step = min(len(out.scores), (decision_idx or 0) + STEPS_PAST_DECISION + 1)
        for step in range(end_step):
            actual_token_id = generated[step].item()
            actual_text = loader.processor.decode([actual_token_id], skip_special_tokens=True)
            logits = out.scores[step][0]
            probs = torch.softmax(logits, dim=-1)
            topk = torch.topk(probs, k=8)
            marker = " <-- DECISION TOKEN" if step == decision_idx else ""
            print(f"\n  step {step} (generated: {actual_text!r}){marker}")
            for p, idx in zip(topk.values.tolist(), topk.indices.tolist()):
                piece = loader.processor.decode([idx], skip_special_tokens=True)
                is_yn = piece.strip().lower() in ("yes", "no")
                flag = "  ** yes/no candidate **" if is_yn else ""
                print(f"    {piece!r:<20s} p={p:.6f}{flag}")

        del image
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
