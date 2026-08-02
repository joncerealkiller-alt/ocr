# Benchmark 2.3 — E2B vs E4B Reference Model Comparison

**Status**: complete. Single independent variable: Gemma 4 E2B ->
Gemma 4 E4B. Every other variable held constant by construction (same
146-image stratified corpus, same source_path per image, same 8 tower
votes - all reused verbatim from the frozen `benchmark2_3_multi_tower_
routing_audit.py` output, never recomputed; same prompt file; same
sampling/token-budget/system-prompt config;
`config/pipeline.yaml`'s `classifier.model` was never read or touched -
production stays on E2B throughout). Script: `benchmark/
benchmark2_3_e2b_vs_e4b.py`.

## Setup verification (done before writing any code, per the brief)

- E2B and E4B confirmed to share the identical architecture class
  (`Gemma4ForConditionalGeneration`, `model_type: gemma4`), identical
  `processor_config.json`, and identical `chat_template.jinja` (matching
  sha256) - verified via each repo's real `config.json`, not the model
  card. `GemmaLoader` needed no code changes for the base architecture
  swap.
- Both E2B and E4B were re-classified FRESH in this script (not just
  E4B), specifically so wall-clock/VRAM timing is measured under
  identical current-session conditions - the original frozen multi-tower
  run never captured per-image timing. E2B's freshly-reproduced
  categories were verified against the frozen original as a determinism
  check: **clean, all 146 images match exactly** (greedy decoding, as
  expected) - confirming nothing else drifted before trusting the E4B
  comparison.

## Real problem found and fixed along the way: naive 8-bit quantization breaks E4B's vision path

E4B's raw bf16 weights (16.02GB) exceed this card's total VRAM
(15.93GB), so an 8-bit quantized load was tried first per Jon's
request. This surfaced two real bugs before landing on a working setup:

1. **bitsandbytes 8-bit, first attempt** (blanket `load_in_8bit=True`):
   agreement with E2B collapsed to 29.5% and mean confidence dropped
   from 0.98 to 0.59. Sampling raw outputs showed *coherent, not
   degenerate* text: E4B was reporting "the image is largely blank" /
   "obscured by a large gray block" on images that are clearly
   legible typed letters (visually confirmed by direct inspection) -
   a real hallucination, not honest hedging about genuinely poor
   image quality.
2. **Fix attempt 1** (`llm_int8_skip_modules=["vision_tower",
   "embed_vision", "audio_tower", "embed_audio"]`, later also adding
   `"lm_head"` after a `'Parameter' object has no attribute 'CB'`
   crash caused by an unrelated bug - supplying an explicit skip list
   REPLACES transformers' default auto-detected skip list rather than
   extending it): did NOT fix the hallucination. Same failure on the
   same images even with those modules confirmed kept in bf16.
3. **Full bf16 test** (no quantization at all, accepting CPU/disk
   offload): the SAME two images classified correctly
   (`printed_document`, confidence 0.99-1.0) - proving the
   hallucination was a quantization artifact, not a genuine E4B
   capability difference, and that the naive bitsandbytes blanket-8bit
   cast was damaging something beyond what was excluded.
4. **Resolution**: switched to `google/gemma-4-E4B-it-qat-mobile-transformers`
   - Google's own QAT-trained, deliberately mixed-precision release
   (`quant_method: "gemma"`, transformers' native `GemmaQuantizer`, not
   bitsandbytes). Ships already packed (3.56GB on disk, not a 16GB bf16
   download cast at load time). Its own embedded `modules_to_not_convert`
   protects `vision_tower.patch_embedder`, `audio_tower` components,
   `embed_audio`/`embed_vision` - baked in by Google's own training
   process, not guessed after the fact. Smoke test on the same two
   images: correct, high-confidence, coherent output. This checkpoint
   is what all numbers below use.

`config/models/gemma_e4b.yaml` documents this history in its own
comments. `core/loaders/gemma_loader.py` gained a generic, config-driven
`extra.load_in_8bit` option (unused by the final config, since this
checkpoint's quantization is self-describing) - kept since it's a real,
correctly-implemented capability, documented with the bugs that led to
discovering the `lm_head`/default-skip-list subtlety.

## Results

**Determinism check**: clean, all 146 E2B reproductions match the
frozen original exactly.

**Overall agreement: 111/146 (76.0%)**

**Of the 35 disagreements**:
```
E4B matches tower consensus (E2B didn't):    14
E4B leaves tower consensus (E2B matched it): 10
Neither matches tower consensus:             11
```
Net movement toward tower consensus: +14/-10 = **+4 images**, a real
but modest improvement, not a dramatic one.

**Performance on the existing 56 consensus-disagreement cases** (images
where E2B already disagreed with the 8-tower consensus, from the
frozen multi-tower audit): **E4B agrees with tower consensus on 14/56
(25%)** of those previously-hard cases. Meaningful partial improvement,
not a fix for most of them.

**Taxonomy-boundary canary case** (`c10264.767`, the image confirmed
dual-natured - handwritten content in a tabular structure, see
`docs/REFERENCE_CLASSIFIER_QUALIFICATION.md`): E2B still gives its
confirmed-wrong answer (`printed_document` - this image is definitely
not printed text). **E4B gives `dense_tabular_rows`** ("numerous
repeated rows structured into columns, characteristic of a manifest"),
matching the tower consensus and the earlier Gemma 12B Unified
finding on this same image. A real, concrete example of E4B improving
on a taxonomy-boundary case, not just a stylistic difference.

**Confidence: more confident AND somewhat more correct, not confidence
substituting for correctness**:
```
Mean confidence E2B: 0.9767
Mean confidence E4B: 0.9927

On the 35 disagreement images specifically:
  E2B mean confidence: 0.9654
  E4B mean confidence: 0.9900
```
E4B is more confident overall and even more so on the disagreement
images specifically - but this rise in confidence is accompanied by a
real (if modest) net gain in tower-consensus agreement (+4), not a
confidence increase divorced from correctness the way the broken
8-bit run showed (confidence collapse alongside content hallucination).

**Runtime / VRAM / throughput cost**:
```
gemma (E2B):     total=948.05s   mean/image=6.493s   peak_vram=10642.1 MB
gemma_e4b:       total=4592.67s  mean/image=31.457s  peak_vram=12524.4 MB
```
E4B is **~4.8x slower per image** and uses **~1.9GB more peak VRAM**,
despite being far smaller on disk (3.56GB QAT-packed vs E2B's
10.28GB bf16). This is a genuinely interesting finding, not a
contradiction: this checkpoint's mixed 2/4/8-bit scheme is tuned for
actual mobile/edge NPU hardware with native low-bit compute units, not
for this desktop GPU's generic PyTorch/bitsandbytes-style kernels -
dequantization overhead per layer during the forward pass makes it
slower here than E2B's full-precision compute, despite the smaller
footprint on disk.

## Answers to the original questions

- **How often do E2B and E4B produce the same routing decision?** 76.0% (111/146).
- **On which images do they disagree?** 35 images; full list in
  `data/outputs/benchmark2_3_e2b_vs_e4b/disagreements.json`, each with
  both models' category/confidence and the tower-consensus bucket for
  reference.
- **Does E4B improve performance on the Benchmark 2.3 disagreement
  cases?** Modestly yes - 14/56 (25%) of the pre-existing hard cases
  now agree with tower consensus; net +4 across all 35 E2B/E4B
  disagreements.
- **Does E4B improve subtype/taxonomy-boundary reasoning?** Yes on the
  one concrete canary case tested - correctly identifies tabular
  structure where E2B falsely claims printed text.
- **Does E4B simply become more confident, or more correct?** Both,
  together, not one substituting for the other - confidence rises
  (0.977 -> 0.993 overall) alongside a real net gain in tower-consensus
  agreement, unlike the broken 8-bit run where confidence collapsed
  alongside content hallucination.
- **What is the runtime/VRAM/throughput cost?** ~4.8x slower per image,
  ~1.9GB more peak VRAM, despite a much smaller on-disk footprint - a
  real cost of this specific mobile-optimized quantization scheme on
  desktop GPU hardware it wasn't tuned for.

## Evidence locations

- `data/outputs/benchmark2_3_e2b_vs_e4b/fresh_results.json` - full
  per-image record for both models (category, confidence, reason,
  per-image elapsed seconds).
- `data/outputs/benchmark2_3_e2b_vs_e4b/disagreements.json` - the 35
  disagreement images with both models' answers + tower consensus.
- `data/outputs/benchmark2_3_e2b_vs_e4b/summary.json` - run metadata
  and all headline numbers above.

## Not changed

- `config/pipeline.yaml` - production `classifier.model` stays `gemma` (E2B).
- `benchmark/benchmark2_3_multi_tower_routing_audit.py` - untouched,
  frozen; its output was reused, not regenerated.
- Corpus, prompt file, taxonomy, evaluation metrics, report structure -
  all identical to the frozen Benchmark 2.3 design.
