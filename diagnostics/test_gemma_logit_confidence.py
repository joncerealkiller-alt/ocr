"""
Logit-based confidence test (2026-08-06) - replaces "ask Gemma to
self-report a confidence number" (confirmed broken twice this session:
malformed non-numeric output like "-99.9"/"Can 1.0", and even when
well-formed, near-uniformly high regardless of correctness) with reading
the model's OWN next-token probability distribution at the actual
decision token. Mechanistically grounded (what the model's logits
actually say), not a self-narrated afterthought.

Uses `model.generate(output_scores=True, return_dict_in_generate=True)`
- a first-class HF feature, not the manual decode-loop machinery from
the earlier KV-cache-reuse investigation (docs/GEMMA_HIERARCHICAL_
ROUTING_INVESTIGATION.md) - `.scores` gives the raw per-step logits
(pre-sampling, post any registered LogitsProcessor e.g. the charset
mask) for every generated token.

Two questions, per Jon's direction:
  1. Can the actual decision token (the "yes"/"no" in GATE_CENSUS_
     PROMPT's "is_census: <yes|no>" answer) be reliably located in the
     generated sequence?
  2. Does that token's OUTPUT INDEX stay stable as max_new_tokens grows
     (i.e. does letting the model keep generating - confidence line,
     reason text - change WHERE the decision token landed)? Greedy
     decoding (do_sample=false, this project's default everywhere)
     guarantees this in theory (step k only depends on steps 0..k-1,
     never on the eventual max_new_tokens budget) - tested empirically
     here rather than assumed, since prompt-formatting behavior could
     in principle differ from the theory.

Uses the same v7 gated-binary GATE_CENSUS_PROMPT already validated
this session (diagnostics/test_gemma_prompt_tiering_variants.py) -
single yes/no answer, ideal for this technique per the design
discussion (multi-token free-form category answers are a much worse
target for logit extraction).

Usage:
    python diagnostics/test_gemma_logit_confidence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image

from diagnostics.test_gemma_prompt_tiering import TEST_CASES, WORKING_DIR, load_model_config
from diagnostics.test_gemma_prompt_tiering_variants import GATE_CENSUS_PROMPT
from core.loader_registry import LOADER_REGISTRY

MAX_NEW_TOKENS_FULL = 40
MAX_NEW_TOKENS_TRUNCATED = [4, 8, 16]  # sanity-check budgets, must be a prefix of the full run


def load_gemma():
    print("Loading gemma (config/models/gemma.yaml)...")
    model_cfg = load_model_config("gemma")
    model_cfg.prompt_text = ""
    loader_cls = LOADER_REGISTRY.get(model_cfg.loader_class)
    loader = loader_cls(model_cfg)
    loader.initialize_model_and_tokenizer()
    print(f"Loaded (do_sample={loader.config.do_sample})\n")
    return loader


def build_inputs(loader, image, prompt_text, system_content):
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ]},
    ]
    text = loader.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = loader.processor(
        text=text, images=image,
        images_kwargs={"max_soft_tokens": loader.config.extra.get("image_token_budget", 280)},
        return_tensors="pt",
    ).to(loader.model.device)
    return inputs


def find_decision_token_index(loader, generated_ids: torch.Tensor) -> int | None:
    """Decodes each generated token individually and returns the index of
    the first one whose stripped/lowercased text is exactly "yes" or "no" -
    the actual decision token, wherever the model's own field-label
    preamble ("is_census: ") happens to put it."""
    for i, tok_id in enumerate(generated_ids):
        piece = loader.processor.decode([tok_id.item()], skip_special_tokens=True).strip().lower()
        if piece in ("yes", "no"):
            return i
    return None


def gt_is_census(ground_truth: str) -> bool:
    return ground_truth.startswith("census_")


def main():
    loader = load_gemma()
    system_content = "You are a strict, non-interpretive archival routing classifier."
    gen_kwargs_base = dict(do_sample=False, use_cache=True,
                            return_dict_in_generate=True, output_scores=True)
    loader._maybe_add_charset_logits_processor(gen_kwargs_base)

    results = []

    for stem, ground_truth in TEST_CASES:
        img_path = WORKING_DIR / f"{stem}.jpg"
        if not img_path.exists():
            continue
        image = Image.open(img_path).convert("RGB")
        inputs = build_inputs(loader, image, GATE_CENSUS_PROMPT, system_content)
        input_len = inputs["input_ids"].shape[-1]

        # -- full run: locate the decision token + read its logits --
        with torch.inference_mode():
            out_full = loader.model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS_FULL, **gen_kwargs_base,
            )
        full_generated = out_full.sequences[0][input_len:]
        decision_idx = find_decision_token_index(loader, full_generated)

        print(f"\n=== {stem} (ground_truth is_census={gt_is_census(ground_truth)}) ===")
        if decision_idx is None:
            print("  Could NOT locate a 'yes'/'no' token in the generated output - "
                  f"raw: {loader.processor.decode(full_generated, skip_special_tokens=True)!r}")
            results.append({"stem": stem, "found": False})
            continue

        decision_token_id = full_generated[decision_idx].item()
        decision_text = loader.processor.decode([decision_token_id], skip_special_tokens=True).strip().lower()
        logits = out_full.scores[decision_idx][0]  # (vocab,) - pre-sampling, post logits-processor
        probs = torch.softmax(logits, dim=-1)
        own_prob = probs[decision_token_id].item()

        # top-5 alternatives at this step, to see if the opposite answer
        # (yes vs no) shows up as a real competing candidate
        topk = torch.topk(probs, k=5)
        alt_desc = []
        opposite_prob = None
        opposite_word = "no" if decision_text == "yes" else "yes"
        for p, idx in zip(topk.values.tolist(), topk.indices.tolist()):
            piece = loader.processor.decode([idx], skip_special_tokens=True).strip().lower()
            alt_desc.append(f"{piece!r}={p:.4f}")
            if piece == opposite_word and opposite_prob is None:
                opposite_prob = p

        print(f"  decision token index: {decision_idx}  (decoded: {decision_text!r})")
        print(f"  P(chosen={decision_text!r}) = {own_prob:.4f}")
        print(f"  P({opposite_word!r}) = {opposite_prob if opposite_prob is not None else '<not in top-5>'}")
        print(f"  top-5 at decision step: {alt_desc}")

        predicted_is_census = decision_text == "yes"
        correct = predicted_is_census == gt_is_census(ground_truth)
        print(f"  predicted_is_census={predicted_is_census}  correct={correct}")

        # -- truncation sanity check: same tokens up to each smaller budget? --
        truncation_ok = True
        for budget in MAX_NEW_TOKENS_TRUNCATED:
            with torch.inference_mode():
                out_trunc = loader.model.generate(
                    **inputs, max_new_tokens=budget, **gen_kwargs_base,
                )
            trunc_generated = out_trunc.sequences[0][input_len:]
            compare_len = min(len(trunc_generated), decision_idx + 1, budget)
            full_prefix = full_generated[:compare_len]
            trunc_prefix = trunc_generated[:compare_len]
            matches = torch.equal(full_prefix, trunc_prefix)
            if decision_idx < budget and not matches:
                truncation_ok = False
            print(f"  budget={budget:3d}: prefix matches full run = {matches}"
                  f"{'  (decision token within this budget)' if decision_idx < budget else '  (decision token beyond this budget)'}")

        results.append({
            "stem": stem, "found": True, "decision_idx": decision_idx,
            "decision_text": decision_text, "own_prob": own_prob,
            "opposite_prob": opposite_prob, "correct": correct,
            "truncation_stable": truncation_ok,
        })

        del image
        torch.cuda.empty_cache()

    print("\n\n=== SUMMARY ===")
    for r in results:
        if not r["found"]:
            print(f"{r['stem']}: decision token NOT FOUND")
            continue
        print(f"{r['stem']}: idx={r['decision_idx']:2d}  {r['decision_text']:>3s}  "
              f"P={r['own_prob']:.4f}  correct={r['correct']}  "
              f"truncation_stable={r['truncation_stable']}")

    found = [r for r in results if r["found"]]
    if found:
        idxs = set(r["decision_idx"] for r in found)
        print(f"\nDecision token index varies across images: {sorted(idxs)} "
              f"({'STABLE - always the same index' if len(idxs) == 1 else 'VARIES by image'})")
        all_stable = all(r["truncation_stable"] for r in found)
        print(f"Truncation stability (position doesn't shift with more budget): "
              f"{'CONFIRMED for all images' if all_stable else 'FAILED for at least one image'}")
        correct_probs = [r["own_prob"] for r in found if r["correct"]]
        wrong_probs = [r["own_prob"] for r in found if not r["correct"]]
        if correct_probs:
            print(f"Mean P(chosen) when CORRECT: {sum(correct_probs)/len(correct_probs):.4f} (n={len(correct_probs)})")
        if wrong_probs:
            print(f"Mean P(chosen) when WRONG:   {sum(wrong_probs)/len(wrong_probs):.4f} (n={len(wrong_probs)})")


if __name__ == "__main__":
    main()
