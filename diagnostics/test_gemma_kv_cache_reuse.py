"""
Phase-1-gating experiment for the hierarchical-routing proposal (2026-08-05).

The whole "one image encoding, N narrow prompts" architecture depends on
one unverified claim: that a Gemma-4 `generate()` call can be seeded with
a PREVIOUS call's KV cache (containing the encoded image) and answer a
NEW, different question without re-running the vision tower and without
producing incorrect output.

Static finding BEFORE running anything (read directly from the installed
transformers 5.12.1 source, not assumed):

  - `Gemma4ForConditionalGeneration.prepare_inputs_for_generation()`
    (modeling_gemma4.py) only omits `pixel_values` from `model_inputs`
    when `is_first_iteration=False`.
  - `GenerationMixin._sample()` (generation/utils.py:2788) hard-codes
    `is_first_iteration=not generation_config.is_assistant` for the
    PREFILL step of every top-level `generate()` call. `is_assistant`
    is the speculative-decoding "draft model" flag - unrelated to
    whether a `past_key_values` cache was pre-supplied. There is no
    public kwarg that lets a caller override this for a normal
    (non-speculative-decoding) `generate()` call.
  - Net effect: calling `model.generate()` a SECOND time - even if you
    hand it `past_key_values=<cache from call 1>` - will still treat its
    own first internal step as "first iteration" and therefore still
    re-attach `pixel_values` to that step's `model_inputs`, i.e. the
    vision tower runs again. The public `generate()` API gives no
    supported path to skip re-encoding on a second top-level call.

This script empirically checks that finding (Part A) and then tests the
ONLY route left per Phase 1's design note - bypassing `generate()`
entirely for prompt 2+ via a manual forward-pass loop seeded with
`inputs_embeds` + the real image features computed once via
`get_image_features()` (Part B). Both parts detach/release tensors
immediately after use, single image at a time, no state held across
images - same lifetime discipline as the existing --debug vision hooks
in core/loaders/gemma_loader.py.

Does NOT modify core/loaders/gemma_loader.py, core/classifier.py, or
any prompt/config file. Standalone, read-only against the model.

Usage:
    python diagnostics/test_gemma_kv_cache_reuse.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image

from core.loaders.base_loader import load_model_config
from core.loader_registry import LOADER_REGISTRY
from core.loaders.constrained_decoding import AllowedCharsLogitsProcessor

# A handful of real corpus images - just need coverage, not ground truth,
# since this experiment tests MECHANISM (does reuse work at all), not
# classification accuracy.
TEST_IMAGES = [
    r"J:\Genealogy\genealogy_pipeline\data\working\e001926997.png",
    r"J:\Genealogy\genealogy_pipeline\data\working\e001928017.png",
    r"J:\Genealogy\genealogy_pipeline\data\working\e001943201.png",
    r"J:\Genealogy\genealogy_pipeline\data\working\e001946014.png",
    r"J:\Genealogy\genealogy_pipeline\data\working\e001946614.png",
]

PROMPT_0 = (
    "Answer with exactly one line: family: <visual_content|document_content|mixed_uncertain>\n"
    "Which processing family best describes this page? Answer only with the line above."
)
PROMPT_1 = (
    "Answer with exactly one line: layout: <printed|handwritten|tabular|narrative>\n"
    "Describe this page's layout. Answer only with the line above."
)


def load_gemma():
    print("Loading gemma (config/models/gemma.yaml)...")
    model_cfg = load_model_config("gemma")
    model_cfg.prompt_text = "unused"  # not routed through _build_prompt() here
    loader_cls = LOADER_REGISTRY.get(model_cfg.loader_class)
    loader = loader_cls(model_cfg)
    t0 = time.time()
    loader.initialize_model_and_tokenizer()
    print(f"Loaded in {time.time() - t0:.1f}s\n")
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


def decode_new_tokens(loader, output_ids, input_len):
    text = loader.processor.decode(output_ids[0][input_len:], skip_special_tokens=False)
    parsed = loader.processor.parse_response(text)
    return str(parsed.get("content", parsed) if isinstance(parsed, dict) else parsed).strip()


# ---------------------------------------------------------------------------
# Part A: does a SECOND top-level generate() call, seeded with the first
# call's past_key_values, actually skip vision re-encoding? (Expectation
# from static analysis above: NO - pixel_values gets re-attached because
# is_first_iteration is call-scoped, not cache-scoped.)
# ---------------------------------------------------------------------------

def part_a_naive_generate_reuse(loader, image, system_content):
    print("  -- Part A: naive generate()->generate() cache handoff --")
    vision_tower = loader.model.model.vision_tower
    fire_count = {"n": 0}

    def _count_hook(module, args, output):
        fire_count["n"] += 1

    handle = vision_tower.register_forward_hook(_count_hook)
    try:
        inputs0 = build_inputs(loader, image, PROMPT_0, system_content)
        with torch.inference_mode():
            out0 = loader.model.generate(
                **inputs0, max_new_tokens=32, do_sample=False,
                return_dict_in_generate=True, use_cache=True,
            )
        vision_fires_after_call1 = fire_count["n"]
        answer0 = decode_new_tokens(loader, out0.sequences, inputs0["input_ids"].shape[-1])
        cache = out0.past_key_values

        # Build ONLY the new prompt's tokens (no image tag) as a naive
        # second-call attempt would - most likely wrong on its own, but
        # this is exactly the naive approach the proposal implies is
        # possible, so it's the one to test first.
        prompt1_text = loader.processor.apply_chat_template(
            [{"role": "user", "content": PROMPT_1}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        prompt1_ids = loader.processor.tokenizer(
            prompt1_text, return_tensors="pt", add_special_tokens=False,
        ).input_ids.to(loader.model.device)

        try:
            with torch.inference_mode():
                out1 = loader.model.generate(
                    input_ids=prompt1_ids,
                    past_key_values=cache,
                    max_new_tokens=32,
                    do_sample=False,
                    use_cache=True,
                )
            vision_fires_after_call2 = fire_count["n"]
            answer1 = decode_new_tokens(loader, out1, prompt1_ids.shape[-1])
            error = None
        except Exception as e:
            vision_fires_after_call2 = fire_count["n"]
            answer1 = None
            error = f"{type(e).__name__}: {e}"
    finally:
        handle.remove()

    return {
        "answer0": answer0,
        "vision_fires_call1": vision_fires_after_call1,
        "answer1": answer1,
        "vision_fires_total_after_call2": vision_fires_after_call2,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Part B: manual reuse - compute image features ONCE via get_image_features(),
# build inputs_embeds by hand for prompt 0, run a manual forward() to prefill
# + decode, then build inputs_embeds for prompt 1 (text-only, no image) and
# continue from the SAME cache via a second manual forward()/decode loop.
# This is the actual mechanism the architecture would need if Part A fails.
# ---------------------------------------------------------------------------

def manual_greedy_decode(loader, initial_input_ids, initial_inputs_embeds, past_key_values,
                          initial_attention_mask, max_new_tokens=32, charset_processor=None):
    """Minimal greedy decode loop driving model.forward() directly, bypassing
    generate() (and therefore bypassing its is_first_iteration=True-always
    behavior) so pixel_values is never re-attached on this call.

    Gemma4 needs `per_layer_inputs` (its Per-Layer-Embedding token-identity
    component) computed from real input_ids at every step - passing
    inputs_embeds alone forces the model to reverse-lookup ids by comparing
    against the full embedding table (embed_tokens.weight), an O(seq_len *
    vocab_size * hidden) operation that OOM'd at ~127GB on the first attempt.
    Real input_ids are available at every step here (the merged image+text
    ids for the prefill step, the greedily-sampled id for every step after),
    so per_layer_inputs is computed directly instead - see
    Gemma4TextModel.get_per_layer_inputs()'s own docstring on the reverse-
    embedding fallback this avoids.

    charset_processor: an AllowedCharsLogitsProcessor (core/loaders/
    constrained_decoding.py), or None. Applied identically to how
    BaseLoader._maybe_add_charset_logits_processor() wires it into
    generate()'s own logits_processor= list - masks disallowed-script
    token logits to -inf BEFORE the argmax, every step, not just the
    first. generate() re-applies every registered LogitsProcessor at
    every decode step (see generation/utils.py's per-step _get_logits_
    processor() usage) - since this loop bypasses generate() entirely,
    that per-step application has to be reproduced by hand here rather
    than assumed to come along for free. input_ids passed to the
    processor's __call__ is the running full generated-so-far sequence
    (processor's own mask lookup ignores it, but this keeps the call
    signature faithful to real generate() usage rather than passing a
    placeholder)."""
    model = loader.model
    text_model = model.model.language_model
    embed_tokens = text_model.embed_tokens
    generated_ids = []
    inputs_embeds = initial_inputs_embeds
    input_ids = initial_input_ids
    attention_mask = initial_attention_mask
    pkv = past_key_values
    full_ids = initial_input_ids

    eos_ids = set(model.generation_config.eos_token_id or [])
    if isinstance(model.generation_config.eos_token_id, int):
        eos_ids = {model.generation_config.eos_token_id}

    for _ in range(max_new_tokens):
        per_layer_inputs = text_model.get_per_layer_inputs(input_ids, None)
        with torch.inference_mode():
            out = model(
                inputs_embeds=inputs_embeds,
                per_layer_inputs=per_layer_inputs,
                attention_mask=attention_mask,
                past_key_values=pkv,
                use_cache=True,
            )
        next_token_logits = out.logits[:, -1, :]
        if charset_processor is not None:
            next_token_logits = charset_processor(full_ids, next_token_logits)
        next_id = torch.argmax(next_token_logits, dim=-1)
        generated_ids.append(next_id.item())
        pkv = out.past_key_values
        if next_id.item() in eos_ids:
            break
        input_ids = next_id.unsqueeze(0)
        full_ids = torch.cat([full_ids, input_ids], dim=-1)
        inputs_embeds = embed_tokens(input_ids)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)],
            dim=-1,
        )
    return generated_ids, pkv


def part_b_manual_embedding_reuse(loader, image, system_content):
    print("  -- Part B: manual get_image_features() + inputs_embeds reuse --")
    model = loader.model
    inputs0 = build_inputs(loader, image, PROMPT_0, system_content)

    # Same opt-in this loader's own _run_generate() honors via BaseLoader.
    # _maybe_add_charset_logits_processor() (gemma.yaml sets it True) -
    # applied by hand every decode step below since this path bypasses
    # generate() (and therefore its logits_processor= kwarg) entirely.
    charset_processor = None
    if loader.config.restrict_output_charset:
        charset_processor = AllowedCharsLogitsProcessor(loader.tokenizer)

    vision_tower = model.model.vision_tower
    fire_count = {"n": 0}

    def _count_hook(module, args, output):
        fire_count["n"] += 1

    handle = vision_tower.register_forward_hook(_count_hook)
    try:
        embed_tokens = model.model.language_model.embed_tokens
        input_ids0 = inputs0["input_ids"]
        text_embeds0 = embed_tokens(input_ids0)

        pixel_values = inputs0.get("pixel_values")
        image_position_ids = inputs0.get("image_position_ids")
        if pixel_values is None:
            handle.remove()
            return {"error": "processor produced no pixel_values for this image - cannot test reuse."}

        with torch.inference_mode():
            image_features = model.get_image_features(pixel_values, image_position_ids)
        # get_image_features() returns the vision_tower's raw output object;
        # .last_hidden_state is the UNPROJECTED 768-dim vision-tower output,
        # .pooler_output is what Gemma4Model.get_image_features() actually
        # sets to embed_vision(last_hidden_state) - the real LM-space (1536-
        # dim) projected embedding that belongs in the text sequence. Confirmed
        # by reading modeling_gemma4.py's Gemma4Model.get_image_features()
        # source directly (assigns vision_outputs.pooler_output = self.embed_
        # vision(...) then returns vision_outputs) after the first attempt
        # (.last_hidden_state) produced a 768-vs-1536 shape mismatch.
        if hasattr(image_features, "pooler_output"):
            image_features = image_features.pooler_output
        vision_fires_after_encode = fire_count["n"]

        # Splice image features into the text embedding sequence at the
        # image-token positions - mirrors what Gemma4Model.forward() does
        # internally via its mm_token_type_ids-driven scatter, done by hand
        # here since we're bypassing that forward() path for the embed step.
        image_token_id = model.config.image_token_id if hasattr(model.config, "image_token_id") else None
        if image_token_id is None:
            # fall back: locate via processor special token
            image_token_id = loader.processor.tokenizer.convert_tokens_to_ids(
                getattr(loader.processor, "image_token", "<image_soft_token>")
            )
        image_mask = (input_ids0 == image_token_id)
        n_image_tokens = int(image_mask.sum().item())
        n_features = image_features.shape[0] * image_features.shape[1] if image_features.dim() == 3 else image_features.shape[0]
        if n_image_tokens == 0 or n_image_tokens != image_features.reshape(-1, image_features.shape[-1]).shape[0]:
            handle.remove()
            return {
                "error": (
                    f"image-token/feature count mismatch: {n_image_tokens} image "
                    f"placeholder tokens vs {image_features.reshape(-1, image_features.shape[-1]).shape[0]} "
                    "feature vectors - manual splice needs adjustment, cannot proceed safely."
                ),
                "vision_fires_after_encode": vision_fires_after_encode,
            }

        merged_embeds0 = text_embeds0.clone()
        merged_embeds0[image_mask] = image_features.reshape(-1, image_features.shape[-1]).to(merged_embeds0.dtype)

        attention_mask0 = inputs0["attention_mask"]

        gen_ids0, pkv = manual_greedy_decode(
            loader, input_ids0, merged_embeds0, None, attention_mask0, max_new_tokens=32,
            charset_processor=charset_processor,
        )
        vision_fires_after_call1 = fire_count["n"]
        answer0 = loader.processor.decode(gen_ids0, skip_special_tokens=True).strip()

        # Prompt 1: text-only continuation, NO image content, reusing pkv.
        prompt1_text = loader.processor.apply_chat_template(
            [{"role": "user", "content": PROMPT_1}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        prompt1_ids = loader.processor.tokenizer(
            prompt1_text, return_tensors="pt", add_special_tokens=False,
        ).input_ids.to(model.device)
        prompt1_embeds = embed_tokens(prompt1_ids)
        # attention_mask must cover: original prompt+answer0 length + new prompt1 tokens
        full_len_so_far = attention_mask0.shape[-1] + len(gen_ids0)
        attention_mask1 = torch.ones((1, full_len_so_far + prompt1_ids.shape[-1]),
                                      dtype=attention_mask0.dtype, device=attention_mask0.device)

        gen_ids1, _ = manual_greedy_decode(
            loader, prompt1_ids, prompt1_embeds, pkv, attention_mask1, max_new_tokens=32,
            charset_processor=charset_processor,
        )
        vision_fires_after_call2 = fire_count["n"]
        answer1 = loader.processor.decode(gen_ids1, skip_special_tokens=True).strip()
        error = None
    except Exception as e:
        import traceback
        traceback.print_exc()
        vision_fires_after_call1 = fire_count["n"]
        vision_fires_after_call2 = fire_count["n"]
        answer0 = None
        answer1 = None
        error = f"{type(e).__name__}: {e}"
    finally:
        handle.remove()

    return {
        "answer0": answer0,
        "vision_fires_after_encode": vision_fires_after_encode if 'vision_fires_after_encode' in dir() else None,
        "vision_fires_after_call1": vision_fires_after_call1,
        "answer1": answer1,
        "vision_fires_total_after_call2": vision_fires_after_call2,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Part C (baseline, cold): what today's production path gives for Prompt 1
# asked independently, for comparison against Part A/B's reused-cache answer.
# ---------------------------------------------------------------------------

def part_c_cold_baseline(loader, image, system_content):
    inputs1 = build_inputs(loader, image, PROMPT_1, system_content)
    vision_tower = loader.model.model.vision_tower
    fire_count = {"n": 0}

    def _count_hook(module, args, output):
        fire_count["n"] += 1

    # Same charset gate Part B now applies by hand - going through the
    # real _maybe_add_charset_logits_processor() here (not a re-implemented
    # copy) so this baseline is the actual production call shape, not an
    # approximation of it.
    gen_kwargs = dict(max_new_tokens=32, do_sample=False, use_cache=True)
    loader._maybe_add_charset_logits_processor(gen_kwargs)

    handle = vision_tower.register_forward_hook(_count_hook)
    try:
        with torch.inference_mode():
            out = loader.model.generate(**inputs1, **gen_kwargs)
        answer1_cold = decode_new_tokens(loader, out, inputs1["input_ids"].shape[-1])
    finally:
        handle.remove()
    return {"answer1_cold": answer1_cold, "vision_fires": fire_count["n"]}


def main():
    loader = load_gemma()
    system_content = "You are a strict, non-interpretive archival routing classifier."

    results = []
    for path in TEST_IMAGES:
        if not Path(path).exists():
            print(f"SKIP {path}: not found")
            continue
        print(f"\n=== {Path(path).name} ===")
        image = Image.open(path).convert("RGB")

        t0 = time.time()
        a = part_a_naive_generate_reuse(loader, image, system_content)
        t_a = time.time() - t0
        print(f"     [A] answer0={a['answer0']!r}")
        print(f"     [A] vision_tower fires after call 1: {a['vision_fires_call1']} (expect 1)")
        if a["error"]:
            print(f"     [A] call 2 FAILED: {a['error']}")
        else:
            print(f"     [A] answer1={a['answer1']!r}")
        print(f"     [A] vision_tower fires TOTAL after call 2: {a['vision_fires_total_after_call2']} "
              f"({'re-encoded (BAD)' if a['vision_fires_total_after_call2'] > a['vision_fires_call1'] else 'no re-encode'})")
        print(f"     [A] elapsed {t_a:.1f}s")

        t0 = time.time()
        b = part_b_manual_embedding_reuse(loader, image, system_content)
        t_b = time.time() - t0
        if b.get("error"):
            print(f"     [B] FAILED: {b['error']}")
        else:
            print(f"     [B] answer0={b['answer0']!r}")
            print(f"     [B] vision_tower fires after encode: {b['vision_fires_after_encode']} (expect 1)")
            print(f"     [B] vision_tower fires after call 1 decode: {b['vision_fires_after_call1']} (expect same as above)")
            print(f"     [B] answer1={b['answer1']!r}")
            print(f"     [B] vision_tower fires TOTAL after call 2: {b['vision_fires_total_after_call2']} "
                  f"({'re-encoded (BAD)' if b['vision_fires_total_after_call2'] > b['vision_fires_after_call1'] else 'no re-encode (GOOD)'})")
        print(f"     [B] elapsed {t_b:.1f}s")

        t0 = time.time()
        c = part_c_cold_baseline(loader, image, system_content)
        t_c = time.time() - t0
        print(f"     [C cold baseline] answer1_cold={c['answer1_cold']!r}  "
              f"vision_fires={c['vision_fires']} (expect 1)  elapsed {t_c:.1f}s")

        results.append({
            "file": Path(path).name,
            "part_a": a, "part_b": b, "part_c": c,
            "elapsed": {"a": t_a, "b": t_b, "c": t_c},
        })

        # explicit cleanup between images - no tensor should carry over
        del image
        torch.cuda.empty_cache()

    print("\n\n=== SUMMARY ===")
    for r in results:
        a, b, c = r["part_a"], r["part_b"], r["part_c"]
        a_reencoded = (not a["error"]) and a["vision_fires_total_after_call2"] > a["vision_fires_call1"]
        b_ok = not b.get("error")
        b_reencoded = b_ok and b["vision_fires_total_after_call2"] > b["vision_fires_after_call1"]
        b_matches_cold = b_ok and b["answer1"] is not None and b["answer1"].strip().lower() == c["answer1_cold"].strip().lower()
        print(f"{r['file']}: "
              f"A={'ERROR' if a['error'] else ('re-encoded' if a_reencoded else 'no-reencode')} | "
              f"B={'ERROR' if not b_ok else ('re-encoded' if b_reencoded else 'no-reencode')} | "
              f"B_answer==cold_answer: {b_matches_cold if b_ok else 'n/a'}")


if __name__ == "__main__":
    main()
