# Reference Classifier Qualification — Gemma 4 E2B vs alternative architectures

**Status**: complete. **Verdict: retain Gemma 4 E2B as the reference
classifier.** Tested against Gemma 4 12B Unified QAT (no promotion -
no meaningful improvement, broken confidence calibration), Fuyu-8B
(ruled out - base model, fails structured output twice), EVE-7B (set
aside - blocked by an upstream broken checkpoint across BAAI's whole
model family, not our setup), and Chameleon-7B (ruled out - complete
mode collapse to a single category across all 68 test images despite
clean surface-level formatting). E2B remains the reference classifier.
Proceeding to the Multi-Tower Routing Audit per Jon's pre-stated
decision rule.

## Section 1: Gemma 4 12B Unified QAT

**Objective** (as scoped): not "is 12B a better model in general" but
whether the architectural difference - Dense Unified, encoder-free
direct patch projection (12B) vs MoE + separate ~150M-param vision
encoder (E2B) - materially improves ROUTING behavior specifically, on
the 4 Gemma-input-qualification images and Benchmark 2.2's 64 real
Gemma-vs-tower disagreement images. Every other variable held constant:
same prompt file, same sampling config (do_sample=false, temp=0.1),
same image token budget (140), same system prompt, same native-
resolution input (no pre-resize, per Variable 1's qualified-safe
finding), sequential loading (2B fully, released, then 12B fully) to
avoid any concurrent-residency confound.

## Setup notes (real, load-bearing findings from getting this running)

- **Model**: `google/gemma-4-12B-it-qat-w4a16-ct` (compressed-tensors
  w4a16 format, 10.3GB download) - not the `-qat-q4_0-unquantized`
  bf16 master checkpoint (24GB) originally proposed. Loader:
  `core/loaders/gemma_unified_loader.py`'s `Gemma4UnifiedLoader`
  (subclasses `GemmaLoader`, overrides only
  `initialize_model_and_tokenizer` to use `AutoModelForMultimodalLM`
  instead of `AutoModelForCausalLM` - every other method, including
  prompt building, generation, and response parsing, is inherited
  unchanged and confirmed correct for this architecture too).
- **MEASURED**: this compressed-tensors checkpoint decompresses to
  full dense precision in memory before compute when loaded through
  plain `transformers` (confirmed by reading
  `compressed_tensors.compressors.model_compressors.model_compressor.
  ModelCompressor.decompress_model`'s own source - it iterates every
  quantized module and decompresses in place). The compressed 10.3GB
  download is NOT the real runtime VRAM footprint - actual usage after
  decompression is ~15.9GB, essentially the same ballpark as the
  original unquantized bf16 checkpoint would need. Google's own blog
  post confirms the fused-kernel, actually-4-bit-at-inference path for
  this format is vLLM-specific; plain `transformers` just gets the
  smaller download, then pays the full dense cost at load.
- **MEASURED**: this leaves extremely thin headroom on a 16GB card
  (~400MB free at peak). A first run attempt crashed
  (`RuntimeError: Tensor.item() cannot be called on meta tensors`)
  because the benchmark script called `loader.release()` alone between
  models - which is a no-op hook on both Gemma loaders, NOT the real
  VRAM-release sequence (null out model/processor/tokenizer + gc.collect
  + torch.cuda.empty_cache(), the pattern `core/row_extraction.py`'s own
  `_release_model()` uses). With 2B's memory never actually freed, the
  12B's decompression didn't have enough real headroom, and accelerate
  partially CPU-offloaded some modules as uninitialized meta tensors,
  which then crashed the compressed-tensors decompression hook. Fixed
  by adding the same `_release_model()` pattern locally to the
  benchmark script (module-private by this project's own convention,
  not a shared import).

## Qualification images

Ground truth is tracked with an explicit status, not silently
redefined by whichever model answers what
(`benchmark/reference_classifier_qualification.py`'s
`QUALIFICATION_IMAGES`):

| Image | Status | 2B | 12B |
|---|---|---|---|
| c10264.658 (control) | CONFIRMED, truth=printed_document | printed_document ✓ | printed_document ✓ |
| genealogy_chart_screenshot_color | CONFIRMED, truth=genealogy_chart | genealogy_chart ✓ | genealogy_chart ✓ |
| c10264.767 | UNDER_REVIEW (see below) | printed_document (CONFIRMED-wrong axis: no handwriting recognized at all) | dense_tabular_rows (a candidate under review, not proof) |
| c10301.601 | AMBIGUOUS, unscored | portrait_photo | mixed_text_image |

**c10264.767 is a taxonomy boundary case, not a resolved win for
either model.** Direct visual re-inspection (not the older single-axis
note) shows the image is genuinely dual-natured: handwritten cursive
content organized in a clear repeating 3-column table (Immigrant/
Farmer/P. Office, ~28 rows). 2B misses the handwriting axis entirely
(printed_document - independently confirmed wrong, it is not printed
text). 12B's dense_tabular_rows answer tracks the structural/layout
axis instead and is a defensible read (matches the image's original
production bucket assignment), but it is NOT evidence that
dense_tabular_rows is "the" correct answer - it's what prompted this
review, not what settles it. **INFERENCE, not measured**: this may
reflect the architectural difference itself - 12B's encoder-free
direct patch projection preserves more raw spatial/layout signal
(column alignment, row repetition) than E2B's separate vision encoder,
which abstracts pixels into semantic features (content/modality) before
the LLM ever sees them, plausibly at the cost of that structural detail.
Not confirmed by a dedicated follow-up test. This example is a strong
candidate for the planned subtype-interface work (bucket + subtype +
properties, rather than forcing one label to carry both axes) once
that phase starts.

On the confirmed-ground-truth images, both models score 2/2 - no
differentiator either way.

## Disagreement corpus (n=64) - directional signal, not correctness

`human_verdict` is still null for all 64 (that review is separately
paused) - this benchmark does NOT claim ground-truth correctness here.
What it measures is INFERENCE-tagged directional shift:

```
12B reproduces 2B's ORIGINAL bucket call:            33/64
12B matches TOWER's call instead (2B/tower disagreed): 16/64
12B lands on a third answer (neither):                15/64
```

**MEASURED, and the deciding finding**: 12B's self-reported confidence
was **exactly 1.0 on all 68 of 68 calls** (4 qualification + 64
disagreement images), with zero variance. 2B over the same 68 calls
used three distinct values (0.95 x23, 0.98 x38, 1.0 x7) - real,
meaningful variance. This isn't a parsing artifact: both loaders share
the identical `_parse_kv_block`/`_to_float` code path (Gemma4UnifiedLoader
inherits it unchanged from GemmaLoader), which correctly extracts 2B's
varied values from the same corpus, and 12B's own reasoning text is
coherent and content-aware even when confidence is saturated (e.g.
disagreement_038: "The image is a photograph of a train derailment and
does not fit any of the specified document or UI categories" ->
category=uncertain_review, confidence=1.0 - self-contradictory pairing:
maximum confidence while explicitly saying nothing fits). A classifier
whose confidence never varies is unusable for this pipeline's
confidence-gated review routing (`pipeline.yaml`'s `min_confidence`
threshold), regardless of category-level accuracy - it would silently
defeat the "route uncertain cases to a human" safety net this project
has repeatedly identified as the actual load-bearing mechanism (see
[[feedback_reduce_manual_touches_is_the_metric]] / [[feedback_abstention_is_a_feature]]
- honest hedging is the thing being protected here, and 12B does not
do it under this prompt).

Separately, category distribution across the 64 disagreement images
shows 12B substantially over-triggers `website_screenshot` (21/64,
~33%) versus 2B's 4/64 (~6%) on the identical images - a ~5x skew, not
an obviously-more-accurate read, and a plausible partial explanation
for the "third answer" 15/64 bucket above.

## Verdict

No meaningful improvement observed - if anything, a regression on the
one dimension (confidence calibration) this pipeline structurally
depends on, plus a real behavioral skew (website_screenshot
over-triggering) that isn't clearly an accuracy gain. Per the
pre-agreed decision rule: **retain Gemma 4 E2B as the reference
classifier.** `config/models/gemma_12b_unified.yaml` and
`core/loaders/gemma_unified_loader.py` are kept in the repo (real,
working, reusable if a future checkpoint/prompt combination is worth
re-testing) but are not wired into `config/pipeline.yaml`.

Proceeding to the Multi-Tower Routing Audit
(`docs/BENCHMARK2_METADATA_LAYER_QUALIFICATION.md`) with 2B unchanged,
per this task's original branch instruction.

## Candidate router replacement: Fuyu-8B - skipped (base model, not a router fit)

Broader context: Gemma was never actually compared against other
encoder-free/unified architectures as a ROUTER specifically - it was
chosen because it "worked" when the router was first built, and this
is the first systematic look at alternatives. Feasibility-checked 5
candidates (EVE-7B, Fuyu-8B, Chameleon-7B, VILA-U-7B, Show-o/Show-o2)
for real HF availability, native `transformers` support, and licensing
before downloading anything. Correction to the initial candidate list:
VILA-U is NOT actually encoder-free - its repo has an explicit
`vision_tower/` component (a unified VQ tokenizer fed by a vision
tower, not raw-patch projection like the others).

**Fuyu-8B (`adept/fuyu-8b`) tested and ruled out for the router role.**
Real encoder-free architecture, natively supported
(`FuyuForCausalLM`), no access gate - the cleanest candidate to test
first on paper. Its own model card is explicit that it's a **base
model**: "we expect you to need to finetune the model for specific use
cases like verbose captioning or multimodal chat" - no chat template,
no system-prompt concept, just raw `text_prompt + image -> continuation`.

- **Setup bug (real, fixed)**: `FuyuProcessor` returns `image_patches`
  as float32 regardless of the model's load dtype - crashes with
  `RuntimeError: mat1 and mat2 must have the same dtype` unless cast to
  `model.dtype` before `generate()`.
- **Zero-shot result (our full production prompt, unmodified)**: on
  `c10264.658` (the known-correct control - both 2B and 12B get this
  right easily), Fuyu returned category=`genealogy chart` (wrong, and
  wrong field-naming - no underscore), confidence=`0.0`, every boolean
  field (`handwriting`/`table_layout`/`faces`/`map_like`) defaulted to
  `true` regardless of actual content, and `reason: reason only` - a
  placeholder-like non-answer. ~230 of 256 max_new_tokens were then
  padded out as blank lines.
- **Few-shot result** (one text-only worked example prepended, using
  deliberately different field values than expected for the real
  image, specifically to catch echo-locking - see
  [[feedback_prompt_examples_get_locked_onto]]): worse, not better. The
  model abandoned the key:value format entirely and fell into a
  verbatim repetition loop - but notably quoting neither the real image
  nor the worked example, instead looping on a fragment lifted directly
  from the PROMPT'S OWN category-definition text for genealogy_chart.
  Classic base-model failure: no instruction/text distinction, so it
  locked onto the most template-like fragment in context and repeated
  it rather than reasoning about the image.
- **Verdict**: decisive negative on two independent attempts (zero-shot
  and few-shot), not worth further prompt engineering as a router
  candidate. No FuyuLoader/config/registry code was written (per this
  project's own discipline: don't invest in the full loader/config/
  registry wiring until a raw smoke test justifies it) - this was all
  done via inline scratch scripts, nothing permanent to clean up.
- **Jon's note for later**: revisit Fuyu specifically for an
  EXTRACTION role instead of routing, if this project ever pursues
  fine-tuning a dedicated extraction model - its model card's own
  strengths (UI/document QA, fine-grained localization, arbitrary
  resolution, fast inference) and explicit "responds well to
  fine-tuning" framing fit an extraction task much better than
  zero-shot classification. Not pursued now; a fine-tuning project is
  a different scope than this qualification.

## Candidate router replacement: EVE-7B - set aside (upstream broken checkpoint, not our bug)

`BAAI/EVE-7B-HD-v2.0` looked more promising than Fuyu on paper: the
model card explicitly says "we release the **instruction-tuned**
weights of EVEv2" (unlike Fuyu's raw base model), and it's Apache-2.0
(no commercial restriction, unlike Fuyu's CC-BY-NC-4.0). Not natively
loadable via `transformers.AutoModel*` - requires the model's own
`eve` package from `github.com/baaivision/EVE` (`eve.model.builder.
load_pretrained_model`), following the same isolated-venv approach as
`MoondreamLoader`/`SubprocessLoaderBase`.

**Setup progress (real, working, kept for a future attempt)**:
- `.venv_eve` created (`--system-site-packages`, reusing this project's
  torch/CUDA per the Moondream pattern).
- `flash_attn` confirmed NOT required for inference (verified by
  reading the actual vendored `eve/model/language_model/qwen2/
  modeling_qwen2.py` directly via a shallow git clone, not trusting
  WebFetch's summarization after it gave two mutually-inconsistent
  `load_pretrained_model()` signatures on two fetches of the same
  page): the `flash_attn` import is guarded behind
  `is_flash_attn_2_available()`, and the model class declares
  `_supports_sdpa = True` - PyTorch's built-in attention backend covers
  inference. `pyproject.toml`'s flat, unsegmented dependency list
  (`apex`, `deepspeed`, `flash_attn`, `xformers`, `wandb`, `gradio`, ...
  all in one `dependencies = [...]`, no real `[train]` extra despite a
  README example implying one) is misleading - those are training/demo
  only. Installed with `pip install --no-deps` + a manually-selected
  minimal runtime subset (accelerate, einops, einops-exts,
  sentencepiece, tiktoken, timm, shortuuid, tabulate, protobuf) instead
  of following the literal `pip install -e .` instruction.
- Real version incompatibility found and fixed (same category of
  problem `SubprocessLoaderBase` exists for): the vendored code imports
  `is_flax_available` from `transformers.utils`, removed in this
  project's main transformers (5.12.1). Pinned `transformers==4.49.0`
  *inside* `.venv_eve` specifically (already a known-good pin
  elsewhere in this project, for HunyuanOCR) - overriding the inherited
  system-site-packages copy for this one package works as intended;
  import succeeds cleanly after the pin.
- Model loads onto `cuda:0` successfully with explicit single-device
  placement (`device_map='cuda:0'`, not the function's own
  `device_map='auto'` default - avoiding the same accelerate-dispatch
  meta-tensor failure mode hit with the Gemma 12B compressed-tensors
  case)... up to the point described below.

**Blocker (confirmed upstream bug, not fixable by choosing a different
EVE variant)**: `EVEQwen2ForCausalLM.__init__` builds a `VisionTokenizer`
that calls `CLIPImageProcessor.from_pretrained(config.mm_vision_tower)`
purely to fetch standard preprocessing metadata (mean/std/resize) - NOT
a pretrained encoder's weights, consistent with the paper's
encoder-free design. `BAAI/EVE-7B-HD-v2.0`'s own `config.json` points
this at `openai/eve-anyratio-res1600-patch16`, which does not exist on
the Hub (404, and an `HfApi` search turns up nothing under any org).
**Checked all 4 published EVE checkpoints** (`EVE-7B-v1.0`,
`EVE-7B-Pretrain-v1.0`, `EVE-7B-HD-v1.0`, `EVE-7B-HD-v2.0`) - every
single one references a different, equally-dead `openai/eve-*` repo
(`eve-patch14-anypixel-672`, `-1344`, and the HD-v2.0 one above). This
is systematic across BAAI's entire released model family, not a
one-off mistake in a single checkpoint - switching to a different EVE
variant would not route around it.

**Verdict**: set aside for now (Jon's call). Not a dead end forced by
our own setup - the actual LLM weights are already fully cached locally
(28.33GB, confirmed via `huggingface_hub.scan_cache_dir()`, so no
re-download needed if revisited) and `.venv_eve`'s transformers pin +
package install already work. The one remaining piece is reconstructing
a `CLIPImageProcessor` manually with reasonable standard CLIP
normalization defaults instead of fetching the dead repo's config -
not attempted, since the exact resize/crop behavior intended for EVE's
"any-ratio" variable-resolution design is undocumented, and guessing
wrong could silently corrupt preprocessing without an obvious crash
(a worse outcome than a clean failure, since it could produce a
misleading result attributed to the model rather than the guess).

## Candidate router replacement: Chameleon-7B - tested, ruled out (complete mode collapse)

`facebook/chameleon-7b` (encoder-free, early-fusion VQ tokenization)
looked like the strongest candidate on paper - genuinely encoder-free
per its own paper, natively supported in `transformers`
(`ChameleonForConditionalGeneration`), and access was approved by Meta
(Chameleon Research License - a different, research-only license from
Fuyu's CC-BY-NC-4.0, confirmed with Jon that's fine for this
evaluation work). No chat template exists for this repo (confirmed
empty in `tokenizer_config.json`, no `chat_template.jinja`) - prompting
follows `transformers`' own documented pattern: a flat text string with
an inline `<image>` placeholder, no system role. New loader:
`core/loaders/chameleon_loader.py`'s `ChameleonLoader` (subclasses
`GemmaLoader` only for its already-generic key:value parsing, same
pattern as `Gemma4UnifiedLoader` - overrides `initialize_model_and_
tokenizer`/`_run_generate` completely).

**Setup bug (real, fixed)**: `token=False` (copied from `GemmaLoader`,
fine there since that repo isn't gated) broke loading here - Chameleon
IS gated, so `token=False` forces an anonymous request that fails even
with legitimate approved access. Removed for this loader; it now uses
whatever token `huggingface_hub` already has cached.

**Single-image smoke test looked promising and was misleading.** On
the known-correct control (`c10264.658`), Chameleon produced a clean,
well-formed 8-field key:value response with a real, non-saturated
confidence (0.9) - format-following on par with an instruction-tuned
model, a genuine step up from Fuyu's degenerate output. But the
category was wrong (`dense_tabular_rows` for a single typed
application form, with reasoning inventing "many rows... different
person" content that isn't in the image).

**Full 68-image run (4 qualification + 64 disagreement corpus)
revealed why: complete mode collapse, not a content-grounded
mistake.** Every single one of 68 calls returned `dense_tabular_rows` -
portraits, maps, website screenshots, genealogy charts, printed
documents, all of it. Confidence was flat at 0.9 on 67/68 calls (0.6 on
one). Only 25 of the 64 disagreement images even had
`dense_tabular_rows` as one of the two candidate answers (2B-original
or tower) already - the other 39 are cases where the fixed answer
doesn't coincidentally align with anything real either. This is the
same category of failure as Fuyu's all-booleans-true defaulting (a
model producing a generic template-shaped response disconnected from
actual image content) but manifesting as a single dominant category
instead of degenerate field values - strong, well-formatted output is
not the same thing as content-grounded output, and this is the second
confirmation of that lesson from this candidate search.

```
chameleon reproduces 2B's original bucket call: 10/64
chameleon matches tower's call instead:          15/64
chameleon lands on a third answer (neither):     39/64
```

**Verdict**: ruled out, same as Fuyu. Real, working `ChameleonLoader` +
`config/models/chameleon.yaml` kept in the repo (loading/prompting
mechanics are confirmed correct - the failure is the model's actual
classification behavior, not our integration) in case a different
prompting strategy or a future checkpoint is worth revisiting, but not
pursued further now.

## Open follow-up (queued, not yet started)

Jon's suggestion: test whether the 12B Unified model can output
positional anchors (bounding boxes / point coordinates) for downstream
use in `core/row_extraction.py`'s row/column cropping. The Gemma 4
model card's "Core Capabilities" section lists "Object detection...
and pointing" as native capabilities for the family, so a documented
output format likely already exists rather than needing to be
improvised. Deliberately not tested as part of this qualification
(would confound the reference-classifier comparison, which holds every
variable but the model itself constant) - queued as the next distinct
piece of work now that this qualification is closed.
