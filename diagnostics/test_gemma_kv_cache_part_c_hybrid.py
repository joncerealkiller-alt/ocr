"""
Part C: embedding/cache reuse via standard model.generate() (not the manual
Python decode loop Part B used), per direct request after Part B validated
the mechanism works but loses to baseline on latency (naive Python loop
overhead) and the token-count sweep suggested the crossover is real but
narrow. Part C asks: does going through generate() itself - and getting
its C++/compiled decode-step optimizations - close that gap while still
only encoding the image once?

STEP 0 - literal draft check: the originally proposed call shape
    model.generate(input_ids=prompt_input_ids, inputs_embeds=multimodal_inputs_embeds, ...)
passes BOTH input_ids and inputs_embeds together. Gemma4Model.forward()
(modeling_gemma4.py:2251-2252, confirmed by direct read) enforces
"exactly one of input_ids or inputs_embeds" via an XOR check - this WILL
raise ValueError as literally drafted. Checked first, cheaply, before
building the corrected version, so this is a confirmed fact rather than
an assumption.

CORRECTED call shape, matching HF's supported "resume generation with an
externally-held cache" pattern: pass ONLY the new suffix (prompt N's real
input_ids, not the whole conversation) + an explicit attention_mask
spanning the FULL true length (past cache + new suffix) + the prior
call's past_key_values. Providing the full previous conversation's
input_ids again (rather than just the new suffix) was tried and rejected
before running: GenerationMixin.prepare_inputs_for_generation()'s base
implementation (generation/utils.py) only slices input_ids down to
`next_sequence_length` when NOT the first iteration of a call - on a
FRESH top-level generate() call (always is_first_iteration=True per the
earlier finding), next_sequence_length is None, so no slicing happens
and the full history would be re-embedded/re-processed on top of an
already-populated cache of the same content - duplicate, wrong. The new-
suffix-only + full-length-attention_mask shape mirrors what Part B's
manual loop already proved correct (attention_mask1's construction there
is the direct analog), now handed to generate() instead of a hand-rolled
loop.

Reports, per image: does call 2 run at all (Q1), vision-tower fire count
across both calls (Q2), answer1 vs a cold independent call's answer1
(Q3), and (separately, in the paired run at the bottom) latency/VRAM
against BOTH the cold baseline and Part B's custom loop (Q4) - three-way
comparison table, not a two-way one, since Part B already exists and
should stay in the picture, not be presumed superseded.

Usage:
    python diagnostics/test_gemma_kv_cache_part_c_hybrid.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image

from diagnostics.test_gemma_kv_cache_reuse import (
    TEST_IMAGES, PROMPT_0, PROMPT_1, load_gemma, build_inputs, decode_new_tokens,
)
from diagnostics.test_gemma_kv_cache_latency_vram import run_baseline, sync, peak_vram_mb
from diagnostics.test_gemma_kv_cache_crossover_sweep import _timed_reuse as run_part_b_timed
from core.loaders.constrained_decoding import AllowedCharsLogitsProcessor

MAX_NEW_TOKENS = 8  # near the crossover region found in the sweep, not the
                     # 32-token point where Part B lost badly


def step0_literal_draft_check(loader, image, system_content):
    """Confirms the XOR ValueError - one call, cheap, once."""
    print("  -- Step 0: literal draft (input_ids= AND inputs_embeds= together) --")
    inputs0 = build_inputs(loader, image, PROMPT_0, system_content)
    model = loader.model
    embed_tokens = model.model.language_model.embed_tokens
    dummy_ids = inputs0["input_ids"][:, :1]
    dummy_embeds = embed_tokens(inputs0["input_ids"])
    try:
        with torch.inference_mode():
            model.generate(input_ids=dummy_ids, inputs_embeds=dummy_embeds, max_new_tokens=1)
        print("     UNEXPECTED: no error raised")
    except ValueError as e:
        print(f"     confirmed ValueError (as predicted from source read): {e}")
    except Exception as e:
        print(f"     different exception than predicted: {type(e).__name__}: {e}")


def part_c_hybrid(loader, image, system_content, charset_processor, n_tokens):
    """Call 1: standard cold generate() over image+Prompt0 (real production
    call shape, real pixel_values, no manual splicing needed for this call).
    Call 2: generate() again, reusing call 1's past_key_values, given ONLY
    Prompt1's new tokens + a full-length attention_mask - see module
    docstring for why NOT the full conversation's input_ids."""
    model = loader.model
    vision_tower = model.model.vision_tower
    fire_count = {"n": 0}

    def _count_hook(module, args, output):
        fire_count["n"] += 1

    handle = vision_tower.register_forward_hook(_count_hook)
    try:
        inputs0 = build_inputs(loader, image, PROMPT_0, system_content)
        gen_kwargs0 = dict(max_new_tokens=n_tokens, do_sample=False, use_cache=True,
                            return_dict_in_generate=True)
        if charset_processor is not None:
            gen_kwargs0["logits_processor"] = [charset_processor]
        with torch.inference_mode():
            out0 = model.generate(**inputs0, **gen_kwargs0)
        vision_fires_after_call1 = fire_count["n"]

        input_len0 = inputs0["input_ids"].shape[-1]
        answer0 = decode_new_tokens(loader, out0.sequences, input_len0)
        total_len_after_call1 = out0.sequences.shape[-1]
        pkv = out0.past_key_values

        prompt1_text = loader.processor.apply_chat_template(
            [{"role": "user", "content": PROMPT_1}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        prompt1_ids = loader.processor.tokenizer(
            prompt1_text, return_tensors="pt", add_special_tokens=False,
        ).input_ids.to(model.device)
        full_attention_mask = torch.ones(
            (1, total_len_after_call1 + prompt1_ids.shape[-1]),
            dtype=torch.long, device=model.device,
        )

        gen_kwargs1 = dict(max_new_tokens=n_tokens, do_sample=False, use_cache=True,
                            return_dict_in_generate=True)
        if charset_processor is not None:
            gen_kwargs1["logits_processor"] = [charset_processor]

        error = None
        answer1 = None
        try:
            with torch.inference_mode():
                out1 = model.generate(
                    input_ids=prompt1_ids,
                    attention_mask=full_attention_mask,
                    past_key_values=pkv,
                    **gen_kwargs1,
                )
            answer1 = decode_new_tokens(loader, out1.sequences, prompt1_ids.shape[-1])
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
        vision_fires_after_call2 = fire_count["n"]
    finally:
        handle.remove()

    return {
        "answer0": answer0, "answer1": answer1, "error": error,
        "vision_fires_call1": vision_fires_after_call1,
        "vision_fires_total": vision_fires_after_call2,
    }


def cold_answer1(loader, image, system_content, charset_processor, n_tokens):
    inputs1 = build_inputs(loader, image, PROMPT_1, system_content)
    gen_kwargs = dict(max_new_tokens=n_tokens, do_sample=False, use_cache=True)
    if charset_processor is not None:
        gen_kwargs["logits_processor"] = [charset_processor]
    with torch.inference_mode():
        out = loader.model.generate(**inputs1, **gen_kwargs)
    return decode_new_tokens(loader, out, inputs1["input_ids"].shape[-1])


def timed_part_c(loader, image, system_content, charset_processor, n_tokens):
    torch.cuda.reset_peak_memory_stats()
    sync()
    t0 = time.perf_counter()
    result = part_c_hybrid(loader, image, system_content, charset_processor, n_tokens)
    sync()
    elapsed = time.perf_counter() - t0
    return elapsed, peak_vram_mb(), result


def main():
    loader = load_gemma()
    system_content = "You are a strict, non-interpretive archival routing classifier."
    charset_processor = None
    if loader.config.restrict_output_charset:
        charset_processor = AllowedCharsLogitsProcessor(loader.tokenizer)

    images = []
    for path in TEST_IMAGES:
        if Path(path).exists():
            images.append((Path(path).name, Image.open(path).convert("RGB")))
        else:
            print(f"SKIP {path}: not found")

    step0_literal_draft_check(loader, images[0][1], system_content)
    print()

    print(f"-- Correctness check (max_new_tokens={MAX_NEW_TOKENS}) --")
    mismatches = 0
    for name, image in images:
        r = part_c_hybrid(loader, image, system_content, charset_processor, MAX_NEW_TOKENS)
        cold1 = cold_answer1(loader, image, system_content, charset_processor, MAX_NEW_TOKENS)
        if r["error"]:
            print(f"{name}: call2 FAILED: {r['error']}")
            mismatches += 1
            continue
        match = r["answer1"].strip().lower() == cold1.strip().lower()
        reencoded = r["vision_fires_total"] > r["vision_fires_call1"]
        if not match:
            mismatches += 1
        print(f"{name}: answer0={r['answer0']!r}  answer1={r['answer1']!r}  cold1={cold1!r}  "
              f"match={match}  vision_fires={r['vision_fires_total']} "
              f"({'RE-ENCODED (BAD)' if reencoded else 'no re-encode (good)'})")
    print(f"\n{len(images) - mismatches}/{len(images)} images: call2 succeeded AND matched cold baseline\n")

    print(f"-- Three-way latency/VRAM comparison (max_new_tokens={MAX_NEW_TOKENS}) --")
    gen_kwargs = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True)
    loader._maybe_add_charset_logits_processor(gen_kwargs)

    baseline_t, partb_t, partc_t = [], [], []
    baseline_v, partb_v, partc_v = [], [], []

    for name, image in images:
        # untimed warmup, each path
        run_baseline(loader, image, system_content, gen_kwargs)
        run_part_b_timed(loader, image, system_content, charset_processor, MAX_NEW_TOKENS)
        timed_part_c(loader, image, system_content, charset_processor, MAX_NEW_TOKENS)

        bt, bv = run_baseline(loader, image, system_content, gen_kwargs)
        pt, pv = run_part_b_timed(loader, image, system_content, charset_processor, MAX_NEW_TOKENS)
        ct, cv, _ = timed_part_c(loader, image, system_content, charset_processor, MAX_NEW_TOKENS)

        baseline_t.append(bt); baseline_v.append(bv)
        partb_t.append(pt); partb_v.append(pv)
        partc_t.append(ct); partc_v.append(cv)

        print(f"{name}:  baseline {bt*1000:7.1f} ms / {bv:7.1f} MB   "
              f"partB(loop) {pt*1000:7.1f} ms / {pv:7.1f} MB   "
              f"partC(hybrid) {ct*1000:7.1f} ms / {cv:7.1f} MB")

        torch.cuda.empty_cache()

    print("\n=== MEDIANS ===")
    print(f"{'path':<20}{'latency_ms':>12}{'vram_mb':>10}{'speedup_vs_baseline':>22}")
    b_med_t, b_med_v = statistics.median(baseline_t) * 1000, statistics.median(baseline_v)
    p_med_t, p_med_v = statistics.median(partb_t) * 1000, statistics.median(partb_v)
    c_med_t, c_med_v = statistics.median(partc_t) * 1000, statistics.median(partc_v)
    print(f"{'baseline (cold x2)':<20}{b_med_t:>12.1f}{b_med_v:>10.1f}{'1.00x':>22}")
    print(f"{'partB (custom loop)':<20}{p_med_t:>12.1f}{p_med_v:>10.1f}{b_med_t/p_med_t:>21.2f}x")
    print(f"{'partC (hybrid gen)':<20}{c_med_t:>12.1f}{c_med_v:>10.1f}{b_med_t/c_med_t:>21.2f}x")


if __name__ == "__main__":
    main()
