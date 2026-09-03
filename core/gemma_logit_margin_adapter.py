"""
Live Gemma decision-token logit-margin adapter, built 2026-08-07 per
Jon's direction: give the Decision Engine (core/routing_decision_
engine.py) real Gemma uncertainty evidence instead of the unscored
placeholder the first replay had to use (see docs/MULTI_SOURCE_VOTING_
CLASSIFIER_PROPOSAL.md's "Decision Engine implementation begins"
section - GemmaLoader.classify() doesn't expose logits, only the
known-unreliable self-reported confidence field).

DOES NOT MODIFY core/loaders/gemma_loader.py. Same established
convention as core/classifier.py's `_enable_raw_output_debug()`
("wrapping at this call site instead means the flag never needs to
touch it, live run or not") - this module wraps an already-initialized
GemmaLoader instance, replicating `_run_generate()`'s exact input-
construction logic (chat template, image_token_budget, gen_kwargs) so
behavior stays IDENTICAL to production, but adds `output_scores=True,
return_dict_in_generate=True` so the raw per-step logits are available,
which `_run_generate()` deliberately doesn't request (it only needs
the decoded text, not the logits).

WHY THE PRODUCTION CLASSIFY() OUTPUT NEEDS A DIFFERENT DECISION-TOKEN
MECHANISM than the earlier binary-gate experiments
(diagnostics/test_gemma_logit_confidence.py): that mechanism located a
single-token "yes"/"no" answer by exact string match. The production
classify() prompt's answer is a multi-token free-form category name
(e.g. "dense_tabular_rows") after a "category: " field label, not a
single token - `find_category_decision_token_index()` below locates the
FIRST token of the category VALUE (not the whole multi-token span) by
finding where "category:" appears in the decoded text and mapping that
character offset back to a token index. [B] Using only the first value
token as "the decision" is a real simplification, not proven optimal:
it's defensible because every category id in this taxonomy starts with
a distinct first word (dense/genealogy/handwritten/map/mixed/portrait/
printed/uncertain/website), so the first generated token very likely
already commits to one category over the others in the common case -
but this is NOT independently verified against every real tokenization
collision the way the yes/no case's exact-match approach was. Flagged
as [C] - not re-derived from a fresh tokenizer-collision check here.

Usage (see diagnostics/run_gemma_logit_adapter_on_benchmark_holdout.py
for a full example against real held-out data):

    from core.classifier import build_classifier_loader
    from core.gemma_logit_margin_adapter import classify_with_logit_margin
    loader = build_classifier_loader(pipeline_cfg)
    result, evidence = classify_with_logit_margin(loader, file_path, image)
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch
from PIL import Image

from core.routing_decision_engine import EvidenceRecord

if TYPE_CHECKING:
    from core.loaders.gemma_loader import GemmaLoader
    from core.schema import ClassificationResult


def _build_generate_kwargs(loader: "GemmaLoader") -> dict:
    """Mirrors GemmaLoader._run_generate()'s gen_kwargs construction
    exactly (same config fields, same conditionals) - duplicated rather
    than imported because _run_generate() doesn't expose this as a
    separate method, and this file deliberately does not modify
    core/loaders/gemma_loader.py. If that method's kwargs logic changes,
    this needs a matching update - flagged here as the one real
    maintenance cost of not touching the loader file."""
    gen_kwargs = dict(
        max_new_tokens=loader.config.max_new_tokens,
        do_sample=loader.config.do_sample,
    )
    if loader.config.do_sample:
        gen_kwargs.update(
            temperature=loader.config.temperature,
            top_p=loader.config.top_p,
            top_k=loader.config.top_k,
        )
    if loader.config.repetition_penalty and loader.config.repetition_penalty != 1.0:
        gen_kwargs["repetition_penalty"] = loader.config.repetition_penalty
    if loader.config.no_repeat_ngram_size:
        gen_kwargs["no_repeat_ngram_size"] = loader.config.no_repeat_ngram_size
    return gen_kwargs


def _build_inputs(loader: "GemmaLoader", raw_image: Image.Image, prompt: str):
    """Mirrors GemmaLoader._run_generate()'s message/template/processor
    construction exactly - same duplication rationale as above."""
    if raw_image.mode != "RGB":
        raw_image = raw_image.convert("RGB")

    system_content = loader.config.extra.get("system_prompt", "")
    if loader.config.reasoning_enabled:
        system_content = "<|think|>" + system_content

    messages = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": raw_image},
            {"type": "text", "text": prompt},
        ],
    })

    image_token_budget = loader.config.extra.get("image_token_budget", 280)
    text = loader.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=loader.config.reasoning_enabled,
    )
    inputs = loader.processor(
        text=text, images=raw_image,
        images_kwargs={"max_soft_tokens": image_token_budget},
        return_tensors="pt",
    ).to(loader.model.device)
    return inputs


def find_category_decision_token_index(loader: "GemmaLoader", generated_ids: torch.Tensor) -> int | None:
    """Locates the FIRST token of the category value following a
    "category:" field label in the generated sequence, by mapping a
    character-offset regex match back to a token index (unlike the
    binary yes/no case's exact single-token match - see module
    docstring for why a different mechanism is needed here)."""
    full_text = ""
    token_spans: list[tuple[int, int]] = []
    for tok_id in generated_ids:
        piece = loader.processor.decode([tok_id.item()], skip_special_tokens=True)
        start = len(full_text)
        full_text += piece
        token_spans.append((start, len(full_text)))

    match = re.search(r"category\s*:\s*", full_text, re.IGNORECASE)
    if match is None:
        return None
    value_start = match.end()
    while value_start < len(full_text) and full_text[value_start].isspace():
        value_start += 1
    if value_start >= len(full_text):
        return None

    for i, (start, end) in enumerate(token_spans):
        if start <= value_start < end:
            return i
    for i, (start, _end) in enumerate(token_spans):
        if start == value_start:
            return i
    return None


def classify_with_logit_margin(
    loader: "GemmaLoader", file_path: str, raw_image: Image.Image,
) -> tuple["ClassificationResult", EvidenceRecord]:
    """
    Runs ONE classification call with output_scores enabled, returning
    BOTH the real production-shaped ClassificationResult (via loader's
    own, unmodified `_parse_classification()` - not a reimplemented
    parser) AND an EvidenceRecord carrying the real raw decision-token
    logit margin, ready to feed core/routing_decision_engine.py.

    Behaviorally identical generation to loader.classify() (same
    prompt, same gen_kwargs, same greedy/sampling config) - the ONLY
    difference is requesting output_scores/return_dict_in_generate so
    the logits are available afterward; this does not change what text
    gets generated (verified: same config, same inputs, same
    deterministic do_sample=False decoding).
    """
    prompt = loader._build_prompt(task="classify")
    inputs = _build_inputs(loader, raw_image, prompt)
    input_len = inputs["input_ids"].shape[-1]

    gen_kwargs = _build_generate_kwargs(loader)
    gen_kwargs["return_dict_in_generate"] = True
    gen_kwargs["output_scores"] = True
    loader._maybe_add_charset_logits_processor(gen_kwargs)

    with torch.inference_mode():
        out = loader.model.generate(**inputs, **gen_kwargs)

    generated_ids = out.sequences[0][input_len:]
    response = loader.processor.decode(generated_ids, skip_special_tokens=False)
    parsed = loader.processor.parse_response(response)
    final_text = parsed.get("content", parsed) if isinstance(parsed, dict) else parsed
    final_text = str(final_text).strip()

    classification_result = loader._parse_classification(file_path, final_text)

    decision_idx = find_category_decision_token_index(loader, generated_ids)
    margin = None
    top1_text = top2_text = None
    top1_prob = None
    if decision_idx is not None and decision_idx < len(out.scores):
        logits = out.scores[decision_idx][0]
        probs = torch.softmax(logits, dim=-1)
        topk = torch.topk(logits, k=2)
        top1_id, top2_id = topk.indices.tolist()
        top1_logit, top2_logit = topk.values.tolist()
        margin = top1_logit - top2_logit
        top1_text = loader.processor.decode([top1_id], skip_special_tokens=True).strip()
        top2_text = loader.processor.decode([top2_id], skip_special_tokens=True).strip()
        top1_prob = probs[top1_id].item()

    evidence = EvidenceRecord(
        sensor_name="gemma", sensor_family="gemma_semantic",
        predicted_class=classification_result.category.value,
        raw_score=margin,
        normalized_score=None,
        calibrated=False,
        metadata={
            "score_kind": "logit_margin" if margin is not None else None,
            "decision_token_index": decision_idx,
            "decision_token_top1_text": top1_text,
            "decision_token_top2_text": top2_text,
            "decision_token_top1_prob": top1_prob,
            "self_reported_confidence": classification_result.confidence,
            "note": ("real raw decision-token logit margin, live-extracted" if margin is not None
                     else "could not locate 'category:' decision token in this output"),
        },
    )
    return classification_result, evidence
