"""
Crossover sweep for the baseline-vs-reuse latency question raised against
test_gemma_kv_cache_latency_vram.py's single-operating-point result (32
output tokens per prompt, reuse 10% SLOWER there).

That result is the wrong end of the curve for the actual proposal: a
hierarchical Decision Engine's real prompts look like

    Q1 -> "YES"
    Q2 -> "TABULAR"
    Q3 -> "CENSUS"
    Q4 -> "PORTRAIT"
    Q5 -> "1911"

i.e. a handful of decode tokens per question, not 32. As max_new_tokens
shrinks, decode-side cost (where the reuse path's naive Python loop is
currently LOSING to generate()'s optimized internals - see the previous
script's findings) shrinks with it, while vision-encode cost (which the
reuse path pays ONCE regardless of how many downstream questions follow,
vs. the baseline paying it once PER question) stays fixed. Sweeping
max_new_tokens across [2, 4, 8, 16, 32] traces out whether/where the two
costs cross.

Same two paths as test_gemma_kv_cache_latency_vram.py (imports its
run_baseline/run_reuse machinery directly, just parameterized over
max_new_tokens now instead of a fixed constant). Reduced to 3 images x 2
timed trials (1 warmup) per sweep point to keep total runtime reasonable
across 5 sweep points x 2 paths.

Usage:
    python diagnostics/test_gemma_kv_cache_crossover_sweep.py
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image

from diagnostics.test_gemma_kv_cache_reuse import TEST_IMAGES, load_gemma
from diagnostics.test_gemma_kv_cache_latency_vram import run_baseline
from core.loaders.constrained_decoding import AllowedCharsLogitsProcessor

SWEEP_TOKEN_COUNTS = [2, 4, 8, 16, 32]
SWEEP_IMAGES = TEST_IMAGES[:3]
TRIALS_PER_POINT = 2  # + 1 untimed warmup


def main():
    loader = load_gemma()
    system_content = "You are a strict, non-interpretive archival routing classifier."

    charset_processor = None
    if loader.config.restrict_output_charset:
        charset_processor = AllowedCharsLogitsProcessor(loader.tokenizer)

    images = []
    for path in SWEEP_IMAGES:
        if Path(path).exists():
            images.append((Path(path).name, Image.open(path).convert("RGB")))
        else:
            print(f"SKIP {path}: not found")

    print(f"Sweeping max_new_tokens={SWEEP_TOKEN_COUNTS} over {len(images)} images, "
          f"{TRIALS_PER_POINT} timed trials/point (+1 warmup)\n")

    results = []  # (n_tokens, baseline_ms, reuse_ms, speedup)

    for n_tokens in SWEEP_TOKEN_COUNTS:
        gen_kwargs = dict(max_new_tokens=n_tokens, do_sample=False, use_cache=True)
        loader._maybe_add_charset_logits_processor(gen_kwargs)

        baseline_times, reuse_times = [], []
        baseline_vrams, reuse_vrams = [], []

        for name, image in images:
            # untimed warmup
            run_baseline(loader, image, system_content, gen_kwargs)
            _warm_reuse(loader, image, system_content, charset_processor, n_tokens)

            for _ in range(TRIALS_PER_POINT):
                t, v = run_baseline(loader, image, system_content, gen_kwargs)
                baseline_times.append(t)
                baseline_vrams.append(v)
                t, v = _timed_reuse(loader, image, system_content, charset_processor, n_tokens)
                reuse_times.append(t)
                reuse_vrams.append(v)

        b_med = statistics.median(baseline_times) * 1000
        r_med = statistics.median(reuse_times) * 1000
        b_vram = statistics.median(baseline_vrams)
        r_vram = statistics.median(reuse_vrams)
        speedup = b_med / r_med if r_med else float("nan")
        results.append((n_tokens, b_med, r_med, speedup, b_vram, r_vram))
        marker = "  <-- reuse WINS" if speedup > 1.0 else ""
        print(f"max_new_tokens={n_tokens:3d}:  baseline {b_med:7.1f} ms   reuse {r_med:7.1f} ms   "
              f"speedup {speedup:.2f}x   vram(base/reuse) {b_vram:.0f}/{r_vram:.0f} MB{marker}")

        torch.cuda.empty_cache()

    print("\n=== SUMMARY TABLE ===")
    print(f"{'tokens':>7} {'baseline_ms':>12} {'reuse_ms':>10} {'speedup':>8}")
    for n_tokens, b_med, r_med, speedup, _, _ in results:
        print(f"{n_tokens:>7} {b_med:>12.1f} {r_med:>10.1f} {speedup:>8.2f}")

    crossings = [r for r in results if r[3] > 1.0]
    if crossings:
        first = crossings[0]
        print(f"\nCrossover found: reuse becomes faster at max_new_tokens<={first[0]} "
              f"(speedup {first[3]:.2f}x)")
    else:
        print("\nNo crossover in the swept range [2,32] - reuse path did not beat "
              "baseline at any tested output length. Either the crossover point is "
              "below 2 tokens (implausible - see write-up) or the naive Python decode "
              "loop's per-step overhead dominates at every length tested, meaning the "
              "architecture's win would require a genuinely optimized decode "
              "implementation, not just fewer output tokens.")


# --- helpers: reuse path with a variable max_new_tokens, since
# test_gemma_kv_cache_latency_vram.run_reuse() hardcodes its module-level
# MAX_NEW_TOKENS constant rather than taking a parameter. Duplicating its
# body here with n_tokens threaded through, rather than editing that
# module's public signature and risking the already-validated latency/
# VRAM script's own behavior. ---

def _timed_reuse(loader, image, system_content, charset_processor, n_tokens):
    import time
    from diagnostics.test_gemma_kv_cache_reuse import build_inputs, manual_greedy_decode, PROMPT_1

    model = loader.model
    embed_tokens = model.model.language_model.embed_tokens
    inputs0 = build_inputs(loader, image, __import__(
        "diagnostics.test_gemma_kv_cache_reuse", fromlist=["PROMPT_0"]
    ).PROMPT_0, system_content)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
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
        max_new_tokens=n_tokens, charset_processor=charset_processor,
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
        max_new_tokens=n_tokens, charset_processor=charset_processor,
    )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return elapsed, peak


def _warm_reuse(loader, image, system_content, charset_processor, n_tokens):
    _timed_reuse(loader, image, system_content, charset_processor, n_tokens)


if __name__ == "__main__":
    main()
