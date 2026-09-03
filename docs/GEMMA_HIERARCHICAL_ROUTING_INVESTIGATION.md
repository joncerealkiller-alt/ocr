# Hierarchical In-Memory Prompt Routing — Investigation & Findings (2026-08-05)

Status: **Parked. Architecture sound, not currently worth building.** Kept
here so this doesn't get re-investigated from scratch later - see
"Recommendation" at the end for what would need to be true to revisit it.

## Bottom line

Architecturally sound, empirically not worth building today. The core
mechanism (one Gemma vision encoding reused across N narrow Decision-
Engine-driven prompts) is real and was proven to work - but every
implementation tried underperforms the current single-prompt classifier
design at the realistic operating point, and the one condition where it
wins hasn't been confirmed with enough statistical rigor to act on.
Recommend: do not implement; revisit only if a stronger decode
implementation or a different operating regime changes the runtime
picture.

---

## Background

Current Stage 5 classifier (`core/classifier.py`, `core/loaders/
gemma_loader.py`) does one large taxonomy-classification prompt per
image. Proposal under investigation: reuse the same Gemma visual tensor
across multiple progressively narrower prompts (e.g. processing-family ->
document layout -> tabular subtype -> census variant), with an external
Decision Engine controlling all branching - Gemma never decides its own
next question, only answers the one it's given.

## Phase 1 - research/feasibility (no code)

Established from reading `core/loaders/gemma_loader.py`,
`core/classifier.py`, and the installed `transformers` 5.12.1 Gemma4
model class (`Gemma4ForConditionalGeneration`) directly:

- Image encoding and text generation are not separated calls anywhere in
  this project's code today - one `processor(...)` + one `model.
  generate()` call does both internally, once per image, once per
  classify() call.
- Inside a single `generate()` call, HF's own decode loop already
  separates the two: vision only runs on that call's first internal
  step. That separation is per-call, not persistent - nothing exposes or
  caches a reusable encoding across separate top-level `generate()`
  calls today.
- No existing primitive in this codebase does what the proposal needs;
  building it is new engineering, not a reconfiguration of anything
  already there.

## Phase 2 - architecture design (no code)

Proposed a 3-5 level routing tree (processing family -> document layout
-> tabular subtype -> census variant, etc.), with:

- Decision Engine owns all branching logic; Gemma only answers isolated
  narrow questions with no memory of prior questions or the tree shape.
- Every edge has three exits: valid-and-branch, invalid-and-retry,
  exhausted-and-quarantine - no edge is allowed to silently force a
  branch on low confidence.
- Per-level abstention as a first-class outcome, not just a top-level
  hedge - since a bad early branch forecloses the correct leaf entirely
  (unlike today's flat classifier, where one hedge covers the whole
  decision).
- Taxonomy stored as a YAML routing graph, not code, so adding a new
  leaf (e.g. `tabular -> land_records`) is a data edit, not a Decision
  Engine code change.

This design is sound independent of the caching question - the
Decision-Engine-controls-workflow separation and per-level abstention
discipline are good practice regardless of whether tensor reuse ever
ships.

## Empirical validation (Phase 2 follow-up experiments)

All experiments are standalone scripts under `diagnostics/`, none modify
`core/loaders/gemma_loader.py`, `core/classifier.py`, or any prompt/
config file.

### Part A - naive `generate()` -> `generate()` cache handoff

Seeding a second top-level `generate()` call with the first call's
`past_key_values` was tested directly. **Structurally broken - hard
crash, every image**, confirmed both by static analysis and by running
it: `GenerationMixin._sample()` hard-codes `is_first_iteration=not
generation_config.is_assistant` for every top-level call's prefill step
(`is_assistant` is the speculative-decoding draft-model flag, unrelated
to whether a cache was pre-supplied) - there is no public kwarg to
override this for a normal call. The public `generate()` API has no
supported path for this.

Script: `diagnostics/test_gemma_kv_cache_reuse.py` (Part A).

### Part B - manual `get_image_features()` + hand-spliced embeddings + custom decode loop

Bypasses `generate()` entirely for prompt 2+: computes image features
once via `Gemma4Model.get_image_features()`, splices them into the text
embedding sequence by hand, and drives decoding via a hand-rolled greedy
loop calling `model.forward()` directly.

**Works.** Vision tower fires exactly once across both prompts, on all 5
test images, confirmed via forward-hook instrumentation (not assumed).
Answers match independent cold calls in content.

Two real Gemma4-internals bugs found and fixed along the way, both
informative for anyone building this for real:
1. `get_image_features()` returns a `BaseModelOutputWithPooling` - the
   LM-space-projected embedding is `.pooler_output` (1536-dim), not
   `.last_hidden_state` (768-dim, pre-projection, wrong shape for
   splicing into the text sequence).
2. Gemma4 has a second embedding pathway ("Per-Layer Embeddings") that
   requires real `input_ids` at every decode step. Passing `inputs_embeds`
   alone forces a reverse-search of the full embedding table to recover
   token identities - tried to allocate **127GB** and OOM'd immediately.
   Fix: always pass real `input_ids` alongside `inputs_embeds` (the
   merged prompt's ids for the prefill step, the greedily-sampled id for
   every step after).
3. `restrict_output_charset`'s constrained-decoding logits mask (`core/
   loaders/constrained_decoding.py`) was wired into the manual loop by
   hand (applied every decode step, matching what `generate()`'s
   `logits_processor=` kwarg does automatically) - confirmed it doesn't
   break the mechanism.

Scripts: `diagnostics/test_gemma_kv_cache_reuse.py` (Part B),
`diagnostics/test_gemma_kv_cache_latency_vram.py`,
`diagnostics/test_gemma_kv_cache_crossover_sweep.py`.

### Part C - cache reuse via standard `generate()` (hybrid)

Tested whether reusing the cache while still going through `generate()`
(instead of a custom loop) could get its compiled-decode-loop advantage
without giving up correctness.

- **Literal draft call shape** (`generate(input_ids=..., inputs_embeds=
  ..., ...)` together) fails worse than expected: not the clean
  `ValueError` the XOR check (`Gemma4Model.forward()`) would predict, but
  the same 127GB OOM Part B's bug #2 hit - something upstream in
  `generate()`'s own input-preparation path falls into the expensive
  reverse-embedding branch before `forward()`'s XOR check is ever
  reached.
- **Corrected shape** (new-suffix-only `input_ids` + a full-length
  `attention_mask` + reused `past_key_values`, call 1 done as a normal
  cold call) runs without erroring, and vision tower still fires exactly
  once across both calls (confirmed).
- **Answer fidelity dropped**: only 1/5 images matched the cold-call
  answer exactly (others showed a token-boundary mismatch not seen in
  Part B) - a new, unresolved correctness gap.
- **Latency: worse than everything**, including doing nothing:

  | Path | Latency (median, 8 tokens/prompt) | VRAM (median) | vs baseline |
  |---|---|---|---|
  | Baseline (2 cold `generate()` calls) | 922.9 ms | 9857.0 MB | 1.00x |
  | Part B (1 encode + custom loop) | 1003.0 ms | 10130.1 MB | 0.92x |
  | Part C (1 encode + hybrid `generate()`x2) | 1156.9 ms | 9849.6 MB | 0.80x |

**Verdict: rejected architecture branch. Do not invest further.** A
fresh top-level `generate()` call carries real per-call Python setup
cost (config validation, logits-processor-list assembly, cache/
stopping-criteria bookkeeping) that gets paid twice here on top of the
actual compute - at short outputs that overhead swamps whatever
advantage `generate()`'s compiled loop has over Part B's raw Python
loop, and it introduces a correctness regression Part B didn't have.

Script: `diagnostics/test_gemma_kv_cache_part_c_hybrid.py`.

---

## Runtime findings summary

| Question | Answer |
|---|---|
| Does vision-tower reuse actually save vision-encode cost? | Yes, confirmed (fires once, not N times) |
| Does that saving translate to a net latency win? | Only in a narrow band, and unconfirmed at rigor |
| At 32 output tokens/prompt | Reuse is 10% slower, ~270MB more VRAM |
| At the swept range 2-32 tokens | No clean monotonic trend; noisy between 4-32 |
| At 2 output tokens (closest to the real "YES / TABULAR / 1911"-style answers this architecture would actually use) | Reuse wins, 1.17x - but one data point from a small, noisy sweep (3 images, 2 trials), not yet validated with proper statistical power or GPU-event-based timing |

Where this leaves it: the real hierarchical-routing use case (short
categorical answers) sits closer to the one point where reuse shows
promise than to the 32-token point where it clearly loses - so the idea
is not falsified. But "one noisy data point suggests a possible win" is
not the same as "validated architecture," and every attempt to get there
via cleaner engineering (Part C) made things worse, not better.

## What would need to be true before building this for real

1. The n~2 crossover result needs to survive a properly powered rerun
   (more trials, more images, GPU-event timing instead of wall-clock)
   before being trusted as more than noise.
2. A decode implementation that's actually competitive with `generate()`'s
   per-token cost - the current custom loop and the `generate()`-hybrid
   are both worse than `generate()` alone at short outputs; neither is a
   viable production decode path as-is.
3. The taxonomy-graph-as-YAML idea and per-level abstention design are
   unimplemented - orthogonal to the caching question, still open work
   if this is ever picked back up.
4. Answer-fidelity between reused-cache and cold-call paths (the token-
   boundary mismatches seen in both Part B and Part C) needs a real
   explanation, not just "same category, close enough" - this matters
   more once real accuracy is at stake, not just a mechanism smoke test.

## Recommendation

Park this. The static-analysis premise from Phase 1 was correct, the
mechanism works, but nothing tested so far demonstrates a runtime win
robust enough to justify the engineering cost (custom decode loop, YAML
routing graph, Decision Engine rebuild) against the current single-
prompt classifier. If revisited later, start from item 1 above (confirm
or kill the n~2 result properly) before any further architecture work -
that's the cheapest remaining way to find out whether this is worth
resuming at all.

## Diagnostic scripts (standalone, no production code touched)

- `diagnostics/test_gemma_kv_cache_reuse.py` - Parts A and B, mechanism
  validation (vision-fire counting, answer correctness vs. cold calls).
- `diagnostics/test_gemma_kv_cache_latency_vram.py` - baseline vs. Part
  B latency/VRAM comparison at a fixed token count.
- `diagnostics/test_gemma_kv_cache_crossover_sweep.py` - sweeps
  `max_new_tokens` in [2, 4, 8, 16, 32] to find where (if anywhere) Part
  B's reuse mechanism beats the baseline.
- `diagnostics/test_gemma_kv_cache_part_c_hybrid.py` - Part C, the
  `generate()`-based hybrid reuse attempt and its three-way comparison
  against baseline and Part B.
