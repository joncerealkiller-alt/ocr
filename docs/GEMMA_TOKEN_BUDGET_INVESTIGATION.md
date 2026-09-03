# Gemma 4 E2B visual-token budget investigation

Date: 2026-08-04 (investigation), fix applied 2026-08-05  
Scope: read-only investigation while inference was active; no model inference was launched and no pipeline code was edited. Repository and installed-library files were copied to a temporary scratch directory before inspection.  
Installed source examined: `transformers==5.14.1`, including `models/gemma4/modeling_gemma4.py`, `processing_gemma4.py`, `image_processing_gemma4.py`, and base `processing_utils.py`.

## Fix applied, 2026-08-05

`core/loaders/gemma_loader.py`'s `_run_generate()` now passes
`images_kwargs={"max_soft_tokens": image_token_budget}` to
`self.processor(...)`, replacing the silently-ignored flat
`image_seq_length=` kwarg this document diagnosed below. Verified live
against the currently-installed package (`transformers==5.12.1` at fix
time - see version note at the end of this addendum): a real
`classify()` call with vision debug instrumentation on reproduced the
exact padded/pooled shapes this document's Phase 3/4 predicted -
`(1, 2520, 768)` encoder, `(273, 768)`/`(273, 1536)` pooled/projected
for `e001926997.png`, matching this doc's own Phase 4 table exactly.

**Direct instruction from Jon, overriding this document's own Phase 7
recommendation**: do not actually reduce the effective budget to the
originally-configured 140 - confirmed directly that dropping to 140
measurably reduces how much of the image Gemma can see. Instead,
`config/models/gemma.yaml` (and `gemma_12b_unified.yaml`/
`gemma_e4b.yaml`, which mirror it) were raised from `140` to `280`,
which is what those configs were **already actually getting** in
production the whole time via this bug - so this is not a new increase,
it's making the config value match the real, already-in-use behavior,
and making it load-bearing going forward (a future config change will
now genuinely take effect, instead of being silently ignored the way
`140` always was).

`_validate_vision_capture()`'s exact-equality check was also corrected
to an upper-bound check (`shape[-2] > expected_tokens`, not `!=`) per
this document's own Phase 4 finding that the real per-image count is
`<= budget`, not a fixed value - the old `!=` form would have falsely
flagged nearly every real image once the call-site bug was fixed.

`config/models/gemma_extract.yaml`'s `image_token_budget: 560` was left
unchanged, but is a real behavior-affecting side effect worth naming
explicitly: extraction was also silently capped at 280 by the same bug,
despite its config asking for 560. This fix means extraction will now,
for the first time, actually run at the higher detail level its own
config already specified - not a regression, but a real change in
Gemma's actual visual input for that role, worth knowing about before
attributing any future extraction-quality shift to something else.

**Version note**: this investigation's Phase 0-7 below cites installed
`transformers==5.14.1`; the fix was implemented and verified against
`transformers==5.12.1` (the version found actually installed in this
project's active environment at fix time). Both versions share the same
relevant API shape (`max_soft_tokens` as the real per-call control,
nested under `images_kwargs`; `image_seq_length` not read at call time)
- the live verification above confirms the fix works correctly against
the actual runtime environment, regardless of which version was live
during the original investigation.

## Executive conclusion

**[Source evidence]** `config/models/gemma.yaml:43-51` sets this pipeline's `extra.image_token_budget` to 140. `core/loaders/gemma_loader.py:406-418` reads that value and passes it to `processor(...)` as `image_seq_length=140`.

**[Source evidence]** In installed `processing_gemma4.py:34-42`, valid image-call arguments are described by `Gemma4ImageProcessorKwargs`; `image_seq_length` is not among them. Installed `image_processing_gemma4.py:119-132` defines the relevant argument as `max_soft_tokens`. Base `processing_utils.py:1611-1661` routes only registered arguments and emits “not a valid argument ... and will be ignored” for an unrecognized flat argument.

**[Interpretation]** The call-time `image_seq_length=140` is ignored by this installed processor version. It does not control image resizing, patch count, pooling length, or prompt placeholder count.

**[Source evidence]** Installed `Gemma4ImageProcessor` defaults to `max_soft_tokens=280`, `patch_size=16`, and `pooling_kernel_size=3` (`image_processing_gemma4.py:135-149`).

**[Interpretation]** The effective visual-token *cap* is therefore 280, not 140. Aspect-ratio-preserving rounding makes the actual count at or below that cap. Counts of 272 and 273 are expected consequences of preprocessing under the 280 cap.

**[Source evidence]** The warning compares the observed pooled/projected token dimension with `self.config.extra.image_token_budget` (`gemma_loader.py:245-265`). Gemma's model forward instead validates that the number of generated image features equals the number of image placeholder slots actually produced by the processor (`modeling_gemma4.py:2336-2347`). Inference evidently continues, which means Gemma's internal feature/placeholder check passes.

**[Interpretation]** Gemma is not internally expecting 140. Only pipeline instrumentation expects 140. The present warning detects a real configuration/API mismatch, but its wording incorrectly treats a configured cap as an exact, fixed output length.

## Phase 1 — Where 140 originates

### Repository-wide search results

The repository mirror was searched for `image_token_budget`, `expected_tokens`, and the standalone number `140`.

**[Source evidence]** The operative classifier value is in `config/models/gemma.yaml:43-51`. Its comment says the supported budgets are 70/140/280/560/1120 and selects 140 for classification.

**[Source evidence]** `core/loaders/gemma_loader.py:245` uses `self.config.extra.get("image_token_budget", 140)` as the instrumentation expectation. `gemma_loader.py:406` uses the same expression for the processor call. The literal 140 is also a local fallback.

**[Source evidence]** `config/models/gemma_12b_unified.yaml:38-41` and `config/models/gemma_e4b.yaml:58-63` independently carry the same local 140 setting. `config/models/gemma_extract.yaml:49-55` documents that extraction was raised from 140 to 560. Benchmarks explicitly test `[140, 280, 560]`.

**[Source evidence]** `docs/GEMMA_INSTRUMENTATION_AND_SENSOR_SURVEY.md:322-338` previously predicted `(1,140,768)` pooled output and approximately 1260 encoder patches from the pipeline setting. The same document cites Gemma 3 separately at lines 430-505.

**[Interpretation]** The number 140 is a pipeline configuration choice based on Gemma 4's advertised supported budgets, not a fixed expectation loaded from the E2B model config at runtime.

**[Interpretation]** It is not merely residue from Gemma 3 research. Gemma 3 is discussed in the older survey, but the operative value and API call were deliberately added for Gemma 4.

**[Hypothesis]** The original implementation followed an earlier model-card/API idiom in which `image_seq_length` appeared to select the image budget. In installed Transformers 5.14.1, `image_seq_length` remains a processor-constructor attribute with a default of 280 (`processing_gemma4.py:54-77`), while per-image computation is controlled by `max_soft_tokens`. This API split likely caused the stale call site.

## Phase 2 — Actual runtime path and shapes

Let:

- `B` = number of images in the processor batch (the classifier uses 1);
- `Pmax = max_soft_tokens × pooling_kernel_size²`;
- `Pvalid = patch_height × patch_width` after resize;
- `Tvalid = Pvalid / pooling_kernel_size²`;
- `Dv = vision_config.hidden_size` (the project survey records 768 for E2B);
- `Dt = text_config.hidden_size` (the project survey records 1536 for E2B);
- `patch_pixels = 16 × 16 × 3 = 768`.

For the currently effective defaults, `max_soft_tokens=280`, `pooling_kernel_size=3`, so `Pmax=2520`.

| Runtime point | Installed-source operation | Shape for one image |
|---|---|---|
| Original image | PIL image passed by `gemma_loader.py:396` | `(W, H)` |
| Resized image | aspect-ratio resize; dimensions are multiples of `3×16=48` | `(3, target_h, target_w)` |
| Patchification | `convert_image_to_patches`, `image_processing_gemma4.py:87-99` | `(Pvalid, 768)` |
| `processor(...)` → `pixel_values` | pad patches to `Pmax`, then stack, lines 260-272 | `(1, 2520, 768)` |
| `image_position_ids` | valid `(x,y)` positions plus `(-1,-1)` padding | `(1, 2520, 2)` |
| `vision_tower.patch_embedder` | linear projection into vision hidden width | `(1, 2520, Dv)` = `(1,2520,768)` |
| `vision_tower.encoder` | retains padded patch sequence | `(1, 2520, Dv)` = `(1,2520,768)` |
| `vision_tower.pooler` before stripping | 3×3 positional average to `2520/9=280` slots | `(1, 280, Dv)` |
| `vision_tower` return | `hidden_states[pooler_mask]` strips padded slots | `(Tvalid, Dv)`, e.g. `(272,768)` or `(273,768)` |
| `embed_vision` | RMS normalization and linear projection only | `(Tvalid, Dt)`, e.g. `(272,1536)` or `(273,1536)` |
| Decoder input merge | projected features replace exactly the processor-created image-token slots | full text sequence `(1, L, Dt)`, with `Tvalid` image positions |

**[Source evidence]** `image_processing_gemma4.py:219-272` computes `max_patches=max_soft_tokens×pooling_kernel_size²`, patchifies each resized image, records `patches.shape[0]//pooling_kernel_size²`, pads to `max_patches`, and returns `pixel_values`, `image_position_ids`, and `num_soft_tokens_per_image`.

**[Source evidence]** `modeling_gemma4.py:2041-2082` computes `output_length=pixel_values.shape[-2]//k²`, runs patch embedding and `self.encoder`, pools, and then applies `hidden_states[pooler_mask]`. That last advanced-indexing operation removes the batch dimension and returns all valid visual tokens as a two-dimensional tensor.

**[Source evidence]** `modeling_gemma4.py:2212-2229` passes the vision-tower result to `embed_vision`. `modeling_gemma4.py:2085-2109` shows that the embedder normalizes and linearly projects the final dimension; it does not alter sequence length.

**[Source evidence]** `modeling_gemma4.py:2336-2351` checks feature count against the processor-created image placeholder mask and uses `masked_scatter` to insert image features into decoder embeddings.

**[Interpretation]** The hook named `pooled` is attached to the whole `vision_tower`, not its internal pooler. It therefore sees the post-mask two-dimensional `(Tvalid,Dv)` result. The `projected` hook sees `(Tvalid,Dt)`. The code's use of `shape[-2]` happens to read the token count correctly for those two-dimensional tensors.

## Phase 3 — Mathematical derivation of 272 and 273

Installed preprocessing computes:

```text
max_patches = max_soft_tokens × k² = 280 × 9 = 2520
target_pixels = max_patches × patch_size² = 2520 × 16² = 645,120
scale = sqrt(target_pixels / (original_height × original_width))
ideal dimensions = original dimensions × scale
target dimensions = floor(each ideal dimension / 48) × 48
patch grid = (target_height/16) × (target_width/16)
pooled grid = (target_height/48) × (target_width/48)
visual tokens = product of pooled-grid dimensions
```

There are no added vision special tokens in this calculation. `<boi>` and `<eoi>` delimiters exist in the text prompt, but they are not outputs of `vision_tower` or `embed_vision`.

### Literal proof for 272

**[Source evidence]** Since target sides are multiples of 48, an aspect-ratio-rounded target of `768×816` produces a patch grid of `48×51`.

**[Interpretation]** The encoder receives the padded length 2520, of which `48×51=2448` patch positions are valid. The 3×3 pooler produces a valid `16×17` grid: `2448/9 = 272`. Its remaining `2520-2448=72` padded patch positions correspond to `8` pooled padding slots, which `hidden_states[pooler_mask]` removes: `280-8=272`.

### Literal proof for 273

**[Source evidence]** A target of `1008×624` produces a patch grid of `63×39 = 2457` valid patches.

**[Interpretation]** Pooling yields a `21×13` grid: `2457/9 = 273`. The padded encoder length is still 2520; `2520-2457=63` padded patch positions become 7 pooled padding slots, and `280-7=273` remain.

**[Hypothesis]** The observed 272 image has an aspect ratio that quantizes to a `16×17` (or rotated `17×16`) pooled grid. Capturing the processor's returned `image_position_ids` or original image dimensions for that exact warning would identify which orientation, but the installed source proves that 272 can only arise here from valid pooled positions after aspect-ratio quantization, not from a hidden fixed 272-token model setting.

## Phase 4 — Is the count dynamic?

No inference was started. Eight existing repository images were copied to scratch and their dimensions were passed through the installed resize formula. This is sufficient to test preprocessing dynamics without touching the running model.

| Image | Original resolution | Aspect ratio W/H | Resized | Patch grid | Encoder valid / padded | Pooled valid |
|---|---:|---:|---:|---:|---:|---:|
| `e001926997.png` | 7654×4724 | 1.6202 | 1008×624 | 63×39 | 2457 / 2520 | 273 |
| `e001928017.png` | 7654×4774 | 1.6033 | 1008×624 | 63×39 | 2457 / 2520 | 273 |
| `e001943201.png` | 7718×4696 | 1.6435 | 1008×624 | 63×39 | 2457 / 2520 | 273 |
| `e001946014.png` | 7422×4810 | 1.5430 | 960×624 | 60×39 | 2340 / 2520 | 260 |
| `e001946614.png` | 7484×4810 | 1.5559 | 960×624 | 60×39 | 2340 / 2520 | 260 |
| `e001946615.png` | 7462×4766 | 1.5657 | 960×624 | 60×39 | 2340 / 2520 | 260 |
| `e001946616.png` | 7462×4810 | 1.5514 | 960×624 | 60×39 | 2340 / 2520 | 260 |
| `e001946617.png` | 7484×4784 | 1.5644 | 960×624 | 60×39 | 2340 / 2520 | 260 |

**[Source evidence]** The real-image calculation produces different pooled counts (273 and 260) under the same 280 cap.

**[Interpretation]** The padded encoder length is fixed at 2520 for this processor setting, but its valid positions vary. The post-mask `vision_tower` and `embed_vision` token counts are dynamic and depend on aspect ratio and resize quantization.

**[Interpretation]** “280” is a maximum budget, not a promise that every image produces exactly 280 features. Likewise, after correcting the call to a 140 cap, actual post-mask counts can be below 140.

## Phase 5 — Verification against installed Transformers

**[Source evidence]** `modeling_gemma4.py:2026-2030` constructs `patch_embedder`, `encoder`, and `pooler` inside `Gemma4VisionModel`.

**[Source evidence]** `modeling_gemma4.py:2041-2082` is the exact location where pooled output is produced. `output_length` comes from the *padded* `pixel_values` length divided by 9; `_avg_pool_by_positions` pools to that length; `hidden_states[pooler_mask]` then removes aspect-ratio padding and creates 272/273/etc.

**[Source evidence]** `modeling_gemma4.py:637-662` derives `k`, verifies `k²×length=input_seq_len`, maps 2-D positions into pooling cells, averages by `k²`, and returns a validity mask.

**[Source evidence]** `modeling_gemma4.py:2212-2229` is the exact `vision_tower → embed_vision` path. `modeling_gemma4.py:2101-2109` verifies that `embed_vision` preserves token count.

**[Interpretation]** The model source contains no fixed 272 or 273 output setting. Those values are produced precisely at `hidden_states = hidden_states[pooler_mask]` from masks created by image preprocessing and positional pooling.

## Phase 6 — Is the warning valid?

### [A] Is Gemma internally expecting 140?

**[Source evidence]** No. Installed processor defaults select a 280 maximum, and the decoder checks features against dynamically generated prompt placeholders, not against 140.

**[Interpretation]** Gemma's internal contract is “feature count equals placeholder count.” Both are derived from actual preprocessing, so 272/273 is internally consistent.

### [B] Is only pipeline instrumentation expecting 140?

**[Source evidence]** Yes. `gemma_loader.py:245-265` obtains 140 from local pipeline config and compares it as an exact pooled/projected dimension.

**[Interpretation]** The warning is valuable because it surfaced that the intended 140 cap was not applied. Its exact-count comparison is nevertheless invalid for this dynamic processor.

## Phase 7 — Recommendation

### Recommendation: update pipeline configuration/API use and compute expected token count automatically

**[Interpretation]** Do not simply remove the warning. It found a genuine silent mismatch between intended and effective preprocessing.

**[Interpretation]** Do not keep the warning unchanged. Even when the intended cap is correctly applied, aspect-ratio rounding means pooled/projected length may be less than the cap.

**[Interpretation]** After the active inference run, change the processor call to the installed API's visual-budget argument, preferably an explicit modality-scoped form such as `images_kwargs={"max_soft_tokens": image_token_budget}` (or the verified equivalent accepted by the pinned Transformers version). This makes the intended cap 140.

**[Interpretation]** Compute the expected actual count from processor output, ideally `inputs["num_soft_tokens_per_image"]`, and validate:

1. each pooled/projected feature count equals the sum of actual per-image soft-token counts;
2. each actual per-image count is `<= configured image_token_budget`;
3. the encoder padded length equals `configured cap × pooling_kernel_size²` when that invariant applies;
4. Gemma's own feature/placeholder equality remains the final correctness contract.

**[Interpretation]** Rename `image_token_budget` in comments/documentation as a maximum/cap, not an exact expected output length. Update the older instrumentation survey's fixed `(1,140,...)` prediction and note the post-mask loss of the batch dimension.

**[Hypothesis]** Once `max_soft_tokens=140` is correctly passed, encoder padded length should become `140×9=1260`, while pooled/projected valid lengths will vary at or below 140. A small processor-only regression test across wide, tall, square, and near-square images can verify this without loading model weights.

## Final answer in one sentence

Gemma 4 E2B produces 272/273 visual tokens because Transformers 5.14.1 ignores the pipeline's call-time `image_seq_length=140`, preprocesses with the installed `max_soft_tokens=280` default, pads to 2520 patches, pools 3×3, and strips aspect-ratio padding to dynamic valid grids such as `16×17=272` or `21×13=273`; Gemma is consistent, while the pipeline warning compares against the wrong argument and treats a cap as an exact length.
