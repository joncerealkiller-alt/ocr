"""
Latency/VRAM comparison for the two mechanisms validated in
diagnostics/test_gemma_kv_cache_reuse.py:

  BASELINE ("current" shape) - two fully independent classify()-style
  calls, each running its own vision encode + decode via the real
  loader._run_generate()-equivalent path (model.generate(), with the
  same charset logits processor production uses).

  REUSE ("proposed" shape) - one vision encode via get_image_features(),
  then two manual forward()-loop decode passes sharing the same
  past_key_values, per the mechanism validated in test_gemma_kv_cache_
  reuse.py's Part B.

Both paths use max_new_tokens=32 (not gemma.yaml's production 256) -
this measures the MECHANISM's cost, not a specific prompt's real output
length; a narrower hierarchical-routing question is exactly the point of
the proposal, so 32 is a deliberately generous stand-in for "a short
answer," not an attempt to make either path look better.

Methodology: one untimed warmup call per image (first CUDA call on a
freshly loaded model pays extra allocator/kernel-selection cost
unrelated to the mechanism itself), then TRIALS_PER_IMAGE timed
repetitions, reporting the median. torch.cuda.synchronize() brackets
every timed region - without it, Python-side wall time would include
only kernel LAUNCH time, not actual GPU completion, understating
whichever path issues more/smaller kernels (the manual per-token loop).

VRAM: torch.cuda.reset_peak_memory_stats() before each timed region,
torch.cuda.max_memory_allocated() after - peak allocated bytes during
that region, not steady-state (steady-state is dominated by resident
model weights either way, not by what this comparison is about).

Does NOT modify core/loaders/gemma_loader.py, core/classifier.py, or
any prompt/config file.

Usage:
    python diagnostics/test_gemma_kv_cache_latency_vram.py
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
    TEST_IMAGES, PROMPT_0, PROMPT_1, load_gemma, build_inputs, manual_greedy_decode,
)
from core.loaders.constrained_decoding import AllowedCharsLogitsProcessor

MAX_NEW_TOKENS = 32
TRIALS_PER_IMAGE = 3  # first is a warmup, not counted


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def peak_vram_mb() -> float:
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def run_baseline(loader, image, system_content, gen_kwargs):
    """Two independent cold generate() calls - today's shape, applied twice."""
    inputs0 = build_inputs(loader, image, PROMPT_0, system_content)
    inputs1 = build_inputs(loader, image, PROMPT_1, system_content)

    torch.cuda.reset_peak_memory_stats()
    sync()
    t0 = time.perf_counter()
    with torch.inference_mode():
        loader.model.generate(**inputs0, **gen_kwargs)
        loader.model.generate(**inputs1, **gen_kwargs)
    sync()
    elapsed = time.perf_counter() - t0
    return elapsed, peak_vram_mb()


def run_reuse(loader, image, system_content, charset_processor):
    """One encode + two manual cache-sharing decode passes - the validated
    Part B mechanism from test_gemma_kv_cache_reuse.py, inlined here rather
    than imported wholesale since that file's part_b_* also does forward-hook
    bookkeeping/error handling irrelevant to a timing run."""
    model = loader.model
    embed_tokens = model.model.language_model.embed_tokens
    inputs0 = build_inputs(loader, image, PROMPT_0, system_content)

    torch.cuda.reset_peak_memory_stats()
    sync()
    t0 = time.perf_counter()

    input_ids0 = inputs0["input_ids"]
    text_embeds0 = embed_tokens(input_ids0)
    pixel_values = inputs0["pixel_values"]
    image_position_ids = inputs0.get("image_position_ids")

    with torch.inference_mode():
        image_features = model.get_image_features(pixel_values, image_position_ids)
    image_features = image_features.pooler_output

    image_token_id = model.config.image_token_id
    image_mask = (input_ids0 == image_token_id)
    merged_embeds0 = text_embeds0.clone()
    merged_embeds0[image_mask] = image_features.reshape(-1, image_features.shape[-1]).to(merged_embeds0.dtype)

    attention_mask0 = inputs0["attention_mask"]
    gen_ids0, pkv = manual_greedy_decode(
        loader, input_ids0, merged_embeds0, None, attention_mask0,
        max_new_tokens=MAX_NEW_TOKENS, charset_processor=charset_processor,
    )

    prompt1_text = loader.processor.apply_chat_template(
        [{"role": "user", "content": PROMPT_1}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    prompt1_ids = loader.processor.tokenizer(
        prompt1_text, return_tensors="pt", add_special_tokens=False,
    ).input_ids.to(model.device)
    prompt1_embeds = embed_tokens(prompt1_ids)
    full_len_so_far = attention_mask0.shape[-1] + len(gen_ids0)
    attention_mask1 = torch.ones((1, full_len_so_far + prompt1_ids.shape[-1]),
                                  dtype=attention_mask0.dtype, device=attention_mask0.device)
    manual_greedy_decode(
        loader, prompt1_ids, prompt1_embeds, pkv, attention_mask1,
        max_new_tokens=MAX_NEW_TOKENS, charset_processor=charset_processor,
    )

    sync()
    elapsed = time.perf_counter() - t0
    return elapsed, peak_vram_mb()


def main():
    loader = load_gemma()
    system_content = "You are a strict, non-interpretive archival routing classifier."

    gen_kwargs = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True)
    loader._maybe_add_charset_logits_processor(gen_kwargs)

    charset_processor = None
    if loader.config.restrict_output_charset:
        charset_processor = AllowedCharsLogitsProcessor(loader.tokenizer)

    all_baseline_t, all_reuse_t = [], []
    all_baseline_vram, all_reuse_vram = [], []

    for path in TEST_IMAGES:
        if not Path(path).exists():
            print(f"SKIP {path}: not found")
            continue
        name = Path(path).name
        image = Image.open(path).convert("RGB")

        # Untimed warmup (both paths) - absorbs first-call allocator/kernel
        # selection cost so it doesn't bias whichever path happens to run first.
        run_baseline(loader, image, system_content, gen_kwargs)
        run_reuse(loader, image, system_content, charset_processor)

        baseline_times, baseline_vrams = [], []
        reuse_times, reuse_vrams = [], []
        for _ in range(TRIALS_PER_IMAGE):
            t, v = run_baseline(loader, image, system_content, gen_kwargs)
            baseline_times.append(t)
            baseline_vrams.append(v)
            t, v = run_reuse(loader, image, system_content, charset_processor)
            reuse_times.append(t)
            reuse_vrams.append(v)

        b_med_t, r_med_t = statistics.median(baseline_times), statistics.median(reuse_times)
        b_med_v, r_med_v = statistics.median(baseline_vrams), statistics.median(reuse_vrams)
        speedup = b_med_t / r_med_t if r_med_t else float("nan")

        print(f"{name}:")
        print(f"  baseline  median {b_med_t*1000:7.1f} ms  (all: {[f'{x*1000:.0f}' for x in baseline_times]})"
              f"   peak VRAM {b_med_v:7.1f} MB")
        print(f"  reuse     median {r_med_t*1000:7.1f} ms  (all: {[f'{x*1000:.0f}' for x in reuse_times]})"
              f"   peak VRAM {r_med_v:7.1f} MB")
        print(f"  speedup: {speedup:.2f}x   vram delta (reuse - baseline): {r_med_v - b_med_v:+.1f} MB")
        print()

        all_baseline_t.append(b_med_t)
        all_reuse_t.append(r_med_t)
        all_baseline_vram.append(b_med_v)
        all_reuse_vram.append(r_med_v)

        del image
        torch.cuda.empty_cache()

    print("=== OVERALL (median of per-image medians) ===")
    print(f"baseline: {statistics.median(all_baseline_t)*1000:.1f} ms, "
          f"{statistics.median(all_baseline_vram):.1f} MB peak")
    print(f"reuse:    {statistics.median(all_reuse_t)*1000:.1f} ms, "
          f"{statistics.median(all_reuse_vram):.1f} MB peak")
    print(f"overall speedup: {statistics.median(all_baseline_t)/statistics.median(all_reuse_t):.2f}x")


if __name__ == "__main__":
    main()
