"""
Follow-up to diagnostics/test_gemma_logit_confidence.py - that run showed
the decision-token mechanism works cleanly (stable index, truncation-safe)
but every probability came back ~1.0 on 10 images the gate answered
CORRECTLY, which doesn't actually test whether logit-confidence drops on
errors - the real question for fixing the broken confidence-gating
problem.

This run targets 5 KNOWN-WRONG cases: real disagreement-review images
where Gemma's ORIGINAL wide classify prompt confidently (wrongly) called
dense_tabular_rows, but human ground truth (data/logs/reviewed/
disagreement_analysis.json) confirms none of them are tabular/census
content at all (printed_document x2, handwritten_ledger, website_
screenshot x2). Ground truth is_census = False for all 5. Runs the SAME
narrow GATE_CENSUS_PROMPT fresh against these images - a genuine stress
test of whether logit confidence is lower on images the model is prone
to getting tabularity-confused about, even asked a narrower question.

Usage:
    python diagnostics/test_gemma_logit_confidence_known_wrong.py
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
from diagnostics.test_gemma_prompt_tiering_variants import GATE_CENSUS_PROMPT

# (disagreement_id, file_path, human_label) - all confirmed NOT tabular/
# census, all cases where Gemma's original wide-prompt call wrongly said
# dense_tabular_rows (data/logs/reviewed/disagreement_analysis.json).
KNOWN_WRONG_CASES = [
    ("disagreement_004", r"J:\Genealogy\genealogy_pipeline\data\working\oocihm.lac_reel_c7319.121.jpg", "printed_document"),
    ("disagreement_005", r"J:\Genealogy\genealogy_pipeline\data\working\oocihm.lac_reel_c7319.122.jpg", "printed_document"),
    ("disagreement_007", r"J:\Genealogy\genealogy_pipeline\data\working\oocihm.lac_reel_t2185.797.jpg", "handwritten_ledger"),
    ("disagreement_008", r"J:\Genealogy\genealogy_pipeline\data\working\Screenshot 2026-05-05 233029.png", "website_screenshot"),
    ("disagreement_009", r"J:\Genealogy\genealogy_pipeline\data\working\Screenshot 2026-05-05 233419.png", "website_screenshot"),
]

MAX_NEW_TOKENS = 40


def main():
    loader = load_gemma()
    system_content = "You are a strict, non-interpretive archival routing classifier."
    gen_kwargs = dict(do_sample=False, use_cache=True,
                       return_dict_in_generate=True, output_scores=True,
                       max_new_tokens=MAX_NEW_TOKENS)
    loader._maybe_add_charset_logits_processor(gen_kwargs)

    results = []
    for disagreement_id, file_path, human_label in KNOWN_WRONG_CASES:
        p = Path(file_path)
        if not p.exists():
            print(f"SKIP {disagreement_id}: {file_path} not found")
            continue
        image = Image.open(p).convert("RGB")
        inputs = build_inputs(loader, image, GATE_CENSUS_PROMPT, system_content)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            out = loader.model.generate(**inputs, **gen_kwargs)
        generated = out.sequences[0][input_len:]
        decision_idx = find_decision_token_index(loader, generated)

        print(f"\n=== {disagreement_id} (human={human_label!r}, ground_truth is_census=False, "
              f"Gemma's ORIGINAL wide-prompt call was WRONG: dense_tabular_rows) ===")
        if decision_idx is None:
            print(f"  Could NOT locate a 'yes'/'no' token - raw: "
                  f"{loader.processor.decode(generated, skip_special_tokens=True)!r}")
            results.append({"id": disagreement_id, "found": False})
            continue

        decision_token_id = generated[decision_idx].item()
        decision_text = loader.processor.decode([decision_token_id], skip_special_tokens=True).strip().lower()
        logits = out.scores[decision_idx][0]
        probs = torch.softmax(logits, dim=-1)
        own_prob = probs[decision_token_id].item()
        # raw logit gap to the runner-up - more resolution than softmax
        # probability near the top of the scale (Part of what the first
        # run's IMCANQC1865 0.9999-vs-1.0000 result hinted at).
        topk = torch.topk(logits, k=2)
        logit_gap = (topk.values[0] - topk.values[1]).item()

        topk_probs = torch.topk(probs, k=5)
        alt_desc = []
        for pr, idx in zip(topk_probs.values.tolist(), topk_probs.indices.tolist()):
            piece = loader.processor.decode([idx], skip_special_tokens=True).strip().lower()
            alt_desc.append(f"{piece!r}={pr:.6f}")

        predicted_is_census = decision_text == "yes"
        correct = predicted_is_census == False  # ground truth is always False here
        print(f"  decision token index: {decision_idx}  (decoded: {decision_text!r})")
        print(f"  P(chosen={decision_text!r}) = {own_prob:.6f}   raw logit gap to runner-up: {logit_gap:.3f}")
        print(f"  top-5: {alt_desc}")
        print(f"  predicted_is_census={predicted_is_census}  correct(should be False)={correct}")

        results.append({
            "id": disagreement_id, "found": True, "decision_text": decision_text,
            "own_prob": own_prob, "logit_gap": logit_gap, "correct": correct,
        })
        del image
        torch.cuda.empty_cache()

    print("\n\n=== SUMMARY: known-wrong-prone cases (all ground_truth is_census=False) ===")
    for r in results:
        if not r["found"]:
            print(f"{r['id']}: NOT FOUND")
            continue
        print(f"{r['id']}: gate_answer={r['decision_text']:>3s}  P={r['own_prob']:.6f}  "
              f"logit_gap={r['logit_gap']:6.3f}  gate_correct={r['correct']}")

    found = [r for r in results if r["found"]]
    if found:
        print(f"\nMean P(chosen) on known-wrong-prone cases: "
              f"{sum(r['own_prob'] for r in found)/len(found):.6f}")
        print(f"Mean logit gap on known-wrong-prone cases: "
              f"{sum(r['logit_gap'] for r in found)/len(found):.3f}")
        print("(Compare against the clean-case run: mean P(chosen) when correct = 1.0000, "
              "n=10, all with much larger logit gaps presumably - this run's numbers show "
              "whether tabularity-prone-confusion images produce a measurably different "
              "signal even when the gate still answers correctly.)")


if __name__ == "__main__":
    main()
