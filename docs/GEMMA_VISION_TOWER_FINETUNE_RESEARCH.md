# Fine-Tuning Gemma's Vision Tower for Archive Routing — Research & Feasibility (2026-08-08)

Status: **Research document only. No code written, no fine-tuning run, no
`GemmaLoader` changes.** Follows the hidden-state probe results in
`docs/GEMMA_HIDDEN_STATE_ROUTING_HEAD_DESIGN.md` and the multi-seed/MLP
follow-up (frozen `vision_encoder` + linear probe: 92.0 ± 0.5% test,
tied for best of 5 hook locations; shallow MLP on the same features:
92.6 ± 0.5%, no real gain over linear). Every claim below is tagged
**[A] confirmed** (verified by reading installed source or this project's
own measured results), **[B] interpretation** (a reasonable reading of [A]
evidence, not itself directly measured), or **[C] hypothesis** (plausible,
untested, would need its own experiment).

---

## 1. Feasibility

**[A]** Gemma4's vision tower is a real, separately-instantiable module —
confirmed by reading the installed `transformers` 5.12.1 source directly
(`transformers.models.gemma4.modeling_gemma4`), not assumed from another
Gemma generation or a generic VLM pattern:

- `Gemma4VisionModel` = `Gemma4VisionPatchEmbedder` → `Gemma4VisionEncoder`
  (a stack of `Gemma4VisionEncoderLayer`, each with `Gemma4VisionAttention`
  + `Gemma4VisionMLP`) → `Gemma4VisionPooler`.
- **`Gemma4VisionConfig` defaults**: `hidden_size=768`,
  `num_hidden_layers=16`, `num_attention_heads=12`, `head_dim=64`,
  `intermediate_size=3072`, `patch_size=16`, `pooling_kernel_size=3`,
  rotary position embeddings (`rope_theta=100.0`), no absolute position
  embedding table — this is a **native-resolution, packed-patch encoder**
  (accepts `pixel_position_ids` as explicit (x,y) patch coordinates, not a
  fixed-grid CLIP/SigLIP-style square input), closer in design lineage to
  NaViT/Pixtral-style variable-aspect-ratio vision towers than to a
  classic fixed-224×224 ViT. This matters directly for question 4 (image
  diversity) — the encoder already natively handles this corpus's mixed
  aspect ratios (portrait photos, wide ledger pages, etc.) without the
  padding/cropping tradeoffs a fixed-grid encoder would force.
- 16 encoder layers at 768-dim is a comparatively **small** vision tower —
  smaller than SigLIP-base (12 layers, but usually ~768-1024 dim depending
  on variant) or a typical CLIP ViT-L (24 layers, 1024-dim). Fewer layers
  to choose an unfreeze boundary from, and a cheaper full fine-tune than a
  large VLM's vision tower would be.
- `embed_vision` (the `projected`/hook-B location) is a **separate module**
  from the vision tower itself — a projection layer mapping the 768-dim
  vision-tower output into the 1536-dim text embedding space (confirmed
  dim from `docs/GEMMA_HIERARCHICAL_ROUTING_INVESTIGATION.md`'s direct
  reading of `get_image_features()`'s `.pooler_output`). It can be
  frozen, fine-tuned, or replaced independently of the vision tower proper.

**What would need to stay frozen**: the entire text decoder stack
(`Gemma4TextModel`/`Gemma4ForConditionalGeneration`'s language layers,
`lm_head`) — question 8's framing ("vision tower only, LM frozen") is
architecturally natural here precisely *because* the vision tower is a
structurally separate submodule with its own optimizer-addressable
parameter set (`model.model.vision_tower.*`), not entangled with the
decoder's weights the way, say, a fused early-fusion architecture might be.

**Which layers could realistically be unfrozen** — by analogy to this
project's own already-completed multi-architecture fine-tune benchmark
(`docs/MULTI_SOURCE_VOTING_CLASSIFIER_PROPOSAL.md`'s "isotropic ViT-style"
unfreeze pattern: last 2 `.blocks` + final norm + head, applied to
dinov2/beit/siglip), the same shape would translate directly to Gemma4's
`Gemma4VisionEncoder.layers` (a flat list, confirmed structurally
isotropic like those families, not hierarchical like Swin):

- **Last N encoder layers + pooler** (N=1-4 of 16) — closest analog to
  what this project already validated works well on this exact taxonomy.
- **Full vision tower** (all 16 layers) — feasible given the tower's small
  size (16 layers × 768-dim is modest compared to the fine-tuned siglip/
  beit/dinov2 towers already trained in this project, several of which are
  85-93M-parameter models with 12-24 layers) — VRAM cost estimated in
  question 7.
- **Projection layer (`embed_vision`) only** — the cheapest possible
  unfreeze, analogous to only ever training a linear adapter; given the
  probe results already show routing information degrades only slightly
  from `vision_encoder` (768-dim, pre-projection) to `vision_projected`
  (1536-dim, post-projection) — 92.0% vs 91.7% test, **[A]** measured,
  overlapping within std — the projection layer alone is unlikely to be
  where a meaningful gain lives; it's already close to information-
  preserving on this task.

**[A]** What the installed Transformers implementation actually allows:
standard PyTorch `.requires_grad_(False)`/`.requires_grad_(True)` on any
submodule works with no Gemma4-specific obstruction — nothing in the
`Gemma4VisionModel`/`Gemma4ForConditionalGeneration` source registers
custom parameter-freezing logic or blocks partial gradient flow. The
model is loaded today via `AutoModelForCausalLM.from_pretrained()`
(`core/loaders/gemma_loader.py`), which returns a standard `nn.Module`
tree — the same `model.model.vision_tower`/`model.model.embed_vision`
paths already used for hook registration (this project's own `--debug`
vision instrumentation, and this session's Stage 2 probe hooks) are the
exact paths a fine-tune script would call `.parameters()` on. **[B]** No
Gemma4-specific fine-tuning obstacle is expected based on this reading —
but this has not been empirically verified by actually running a
backward pass through the vision tower in this project yet, only by
static inspection of the module tree and forward-hook behavior.

## 2. Training strategy comparison

| Approach | What's trainable | Analogy in this project | Feasibility here |
|---|---|---|---|
| Linear probe (done) | One `nn.Linear` on frozen features | This session's Stage 1/2 work | **[A]** Done — 92.0 ± 0.5% |
| Vision tower + frozen LM | Vision tower (last-N or full) unfrozen, LM decoder frozen, task loss backpropagated through the vision tower only | Directly what question 1 scopes | **[B]** Architecturally clean given the vision tower's structural separation |
| Vision tower + projection layer | Vision tower + `embed_vision` both unfrozen, LM frozen | Natural extension of the above | **[B]** Same feasibility, marginal extra trainable params |
| LoRA on the vision tower | Low-rank adapters injected into `Gemma4VisionAttention`/`Gemma4VisionMLP` linear layers, base weights frozen | Not yet tried in this project on any vision tower (`training/` LoRA infra per [[project_workflow_gui_preview_and_checkpoints]] targets the LM/extraction side, not vision) | **[A]** `peft` 0.19.1 is installed in this environment (confirmed via `pip show`-equivalent import check) — LoRA is mechanically available; would need new wiring, not present today |
| Full vision fine-tuning | All 16 encoder layers + pooler + patch embedder | None of this project's fine-tunes have gone this far even on the larger CNN/ViT towers (all used last-N-layer unfreezing) | **[B]** Feasible given the tower's small size, but this project has no direct precedent for "unfreeze everything" on *any* vision encoder yet — every prior fine-tune (dinov2/beit/siglip/convnext/swin/vit21k) used partial unfreezing |
| Parameter-efficient alternatives (adapters, prompt-tuning on vision tokens, BitFit-style bias-only tuning) | Small inserted modules or bias terms only | Not tried in this project | **[C]** Plausible but unexplored territory here; LoRA is the most standard version of this family and already has `peft` available |

**[B] Recommendation for a first experiment, if one is run**: last-N-layer
unfreezing (N=2, matching this project's own validated isotropic-ViT
recipe) + frozen projection + frozen LM, evaluated against the probe
baseline. This is the closest analog to a method already proven on this
exact taxonomy (siglip/beit/dinov2 all gained 25-28pts with this exact
recipe — see question 3) and reuses the codebase's existing training
pattern (`diagnostics/vit_family_benchmark_common.py`'s
`configure_finetune_layers()`-equivalent logic) almost directly, rather
than inventing a new unfreeze strategy from scratch. LoRA is the natural
Stage 2 of this experiment if last-N full unfreezing overfits (small
dataset risk, see question 4) — not the first thing to reach for.

## 3. Expected gains — grounded in this project's own measured numbers, not just published literature

**[A] The single most relevant piece of evidence is already in this
codebase, not the published literature**: `docs/
MULTI_SOURCE_VOTING_CLASSIFIER_PROPOSAL.md`'s controlled multi-architecture
benchmark already measured exactly this kind of fine-tune — last-2-blocks
(or family-equivalent) unfreezing, frozen backbone otherwise, same task,
same taxonomy, same held-out split — across 6 independent vision
architectures:

| Model | Zero-shot | Fine-tuned | Gain |
|---|---|---|---|
| siglip | 64.2% | 92.5% | +28.3 pts |
| beit | 65.7% | 91.3% | +25.6 pts |
| dinov2 | 64.5% | 91.0% | +26.5 pts |
| convnext | 67.8% | 90.7% | +22.9 pts |
| swin | 68.4% | 89.8% | +21.4 pts |
| vit21k | 54.4%* | 89.8% | +35.4 pts* |
| mobilenetv2 | 59.6% | 76.5% | +16.9 pts |

**[A]** "Every architecture generalizes the core finding" per that doc —
domain-adaptation fine-tuning on this specific taxonomy reliably moves a
frozen-embedding baseline from a 54-68% zero-shot ceiling to an 89-93%
fine-tuned ceiling, consistently across CNN, hierarchical-transformer,
isotropic-transformer, and vision-language-pretrained (SigLIP) encoder
families — mobilenetv2 is the one outlier, and its smaller gain tracks
its much smaller unfrozen parameter count, not a failure of the method.

**[B] Why this doesn't transfer directly to "expect +25pts from fine-tuning
Gemma's vision tower"**: every model in that table started from a
**zero-shot embedding** baseline (54-68%) — a frozen, generic,
ImageNet/web-pretrained feature space with *no* task-specific linear
readout trained on it at all in the zero-shot condition (that table's
"zero-shot" numbers come from cosine-similarity-to-reference-embedding
classification, not even the equivalent of a trained linear probe).
Gemma's *frozen* vision tower, by contrast, **already reaches 92.0%
±0.5% with nothing more than a linear probe** on top of it. The apples-
to-apples comparison isn't "zero-shot embedding vs. fine-tuned tower" —
Gemma already skipped past the zero-shot-embedding stage entirely and
landed close to where those other models land only *after* fine-tuning.
**[B]** This strongly suggests most of the "domain adaptation gain" those
other encoders needed fine-tuning to capture, Gemma's vision tower may
have already captured through its own (much larger-scale, VLM-oriented)
pretraining — which is a plausible explanation for why the decoder
hidden-state probes and the shallow MLP experiment *also* found no
further gain: the representation may simply already be close to whatever
ceiling this taxonomy's visual signal supports, at least for a linear/
shallow readout.

**[C] Is moving from ~92% toward ~94% (matching full autoregressive
generation) a realistic expectation from vision-tower fine-tuning
specifically?** Genuinely uncertain, and arguably the *least* likely of
the levers considered so far to close that gap, precisely because of the
argument above: fine-tuning helps most when the frozen baseline is far
from ceiling (54-68%→90%+ pattern, [A]), and Gemma's frozen baseline is
already near ceiling (92%). The remaining ~2pt gap to full generation
could instead reflect information the *decoder*'s prompted reasoning adds
that isn't present in vision features at all (e.g. resolving ambiguous
cases using textual/layout reasoning conditioned on the specific prompt
wording) — in which case no amount of vision-tower fine-tuning would
close it, because the missing signal was never a vision-encoding problem.
This is a **testable claim**, not asserted as fact: Stage 5's design
(comparing the best hidden-state approach against full generation on
identical images, per `docs/GEMMA_HIDDEN_STATE_ROUTING_HEAD_DESIGN.md`)
would need to specifically inspect the ~7-8% of images the probe gets
wrong to see whether they look like "genuinely ambiguous, needs reasoning"
cases or "clean images the probe just misjudged" cases — that error
analysis hasn't been done yet and would be informative before greenlighting
a fine-tune run.

**Risks — catastrophic forgetting / overfitting**:
- **[A]** This project's own fine-tunes were all done on a fairly small
  corpus (the `vit_finetune_dataset.csv` split — 1568 train images this
  session's own probe reused). **[B]** A vision tower with 16 layers at
  768-dim, even partially unfrozen, has meaningfully more trainable
  parameters than a linear probe (768×8 ≈ 6K params) — last-2-layers
  unfreezing alone would be in the low-to-mid millions of parameters,
  comparable to what dinov2 (22M total, ~3-4M unfrozen per that
  benchmark's per-family breakdown) used. Overfitting risk on ~1568
  training images is real and needs the same held-out-val-checkpoint-
  selection discipline this project already uses (`train_linear_probe`'s
  best-val-accuracy checkpointing, this benchmark's `NUM_EPOCHS=8` cap).
- **[C]** Catastrophic forgetting of Gemma's general vision capability is
  a real concern *if* the fine-tuned tower were ever reused for anything
  beyond this narrow routing task (e.g. if a future stage wanted the same
  tower for extraction-quality visual grounding) — but for a
  routing-only, frozen-LM fine-tune whose only consumer is the linear/
  shallow head studied so far, this risk is largely moot: nothing downstream
  currently depends on Gemma's vision tower retaining general-purpose
  capability outside this pipeline.

## 4. Dataset suitability

**[A]** From this session's own split (`diagnostics/
vit_family_benchmark_common.py`'s `get_split()`, reused for every probe
experiment): 1568 train / 332 val / 332 test images across 8 taxonomy
classes, stratified. **[A]** Per-class support is uneven — the probe's
own confusion matrices (Stage 1/2 reports) show `printed_document` at
175 test images vs. `genealogy_chart`/`mixed_text_image` at 1 and 3
respectively. **[B]** This imbalance is almost certainly why
`mixed_text_image` was the noisiest class across every probe experiment
run so far (0/3 to 2/3 test accuracy, wildly unstable across seeds) —
a class with 3 held-out examples cannot produce a statistically
meaningful per-class accuracy number regardless of what fine-tuning does
to the underlying representation.

- **Dataset size**: 1568 training images is workable for last-N-layer
  unfreezing (this project's own siglip/beit/dinov2/convnext/swin fine-
  tunes used a comparably-sized split and succeeded), but full
  vision-tower fine-tuning (all 16 layers) on this few images is a
  materially higher overfitting risk than last-N unfreezing — **[C]**
  not tested at this scale in this project yet.
- **Taxonomy balance**: uneven, confirmed **[A]**. Per
  [[project_no_comprehensive_ground_truth]] (only sparse, biased
  review-queue labels exist and always will) this isn't a gap that gets
  fixed before an experiment — any fine-tune here needs class-weighted
  loss (matching this project's existing pattern) and results need to be
  read per-class, not as one aggregate accuracy number, especially for
  low-support classes.
- **Image diversity**: **[A]** the corpus spans genuinely different
  physical sources (microfilm scans, phone-camera photos, website
  screenshots, LAC census pulls across multiple years — 1906/1916/1921/
  1926/1931 per [[project_prairie_census_different_form]]) — real
  diversity within classes, which is generally good for generalization
  but also means some classes (`dense_tabular_rows` in particular, which
  spans structurally different census-year forms per that memory) may
  need more than a few hundred examples to be well-represented by
  fine-tuning.
- **Label quality**: **[A]** ground truth here comes from
  `manifest_final.csv`'s hand-reviewed labels — the same labels this
  session's every probe experiment already trusted implicitly. No new
  label-quality concern specific to fine-tuning beyond what already
  applies to every measurement in this document.
- **Would additional self-supervised pretraining help first?** **[C]**
  Speculative — this project has no precedent for self-supervised
  pretraining on its own corpus (every existing fine-tune in this
  codebase is supervised, starting from an already-pretrained public
  checkpoint). Given the corpus size (~1568-2233 labeled images total)
  is almost certainly too small to usefully self-supervised-pretrain a
  vision tower from scratch, and Gemma's tower already arrives with
  large-scale pretraining, this doesn't look like a promising lever
  compared to simply trying supervised last-N-layer fine-tuning first —
  but this is an untested judgment call, not a measured conclusion.

## 5. Training objective

- **Direct taxonomy-label supervision (cross-entropy)** — **[B]**
  the default, lowest-risk choice, and what every one of this project's
  existing fine-tunes already uses (class-weighted cross-entropy per
  `vit_family_benchmark_common.py`). Matches the linear-probe/MLP
  experiments already run this session, so a fine-tuned-tower result
  would be directly comparable without changing what's being optimized
  for, only what's frozen.
- **Contrastive / metric learning (e.g. supervised contrastive loss,
  pulling same-class embeddings together)** — **[C]** could plausibly
  produce a more separable embedding space than pure cross-entropy,
  especially useful if the eventual head stays a simple linear probe
  (contrastive pretraining objectives are often justified specifically
  when the downstream readout is meant to stay cheap) — but adds real
  complexity (batch construction, temperature tuning) for a benefit
  that's speculative given cross-entropy already gets other architectures
  in this project to 89-93%.
- **Triplet loss** — **[C]** same family as contrastive, generally harder
  to tune (hard-negative mining) than a modern contrastive loss (e.g.
  InfoNCE-style) for equivalent benefit — no obvious reason to prefer
  triplet loss over a supervised contrastive alternative here.
- **Recommendation**: **[B]** start with direct cross-entropy — it's the
  proven, low-risk choice this project's own fine-tune benchmark already
  validated across 6 architectures, and it keeps this experiment
  comparable to the linear-probe baseline already measured. Reach for
  contrastive/metric learning only if cross-entropy fine-tuning shows a
  real gain worth refining further, not as a first move.

## 6. Evaluation methodology

Reuse `diagnostics/vit_family_benchmark_common.py`'s `get_split()` **[A]
already the shared split for every experiment referenced in this
document** — the same train/val/test partition (seed 42, stratified) used
by Stage 1/2 probes, the multi-seed sweep, the MLP experiment, and the
6-architecture fine-tune benchmark. A fine-tuned-vision-tower experiment
MUST use this exact split, not a fresh one, for any of the comparisons
below to be valid.

Proposed comparison, all four evaluated on the identical 332-image held-out
test set:

| Approach | Status | Source |
|---|---|---|
| Frozen vision tower + linear probe (baseline) | **[A]** measured this session | 92.0 ± 0.5% (10-seed mean), `vision_encoder` |
| Fine-tuned vision tower + frozen LM + linear/shallow head | Not yet run | This proposal |
| Full Gemma generation | **[A]** measured | ~94%, `diagnostics/run_gemma_flat8_on_benchmark_holdout.py` |
| Existing ViT-21k routing pipeline | **[A]** measured | 89.8% held-out, `docs/MULTI_SOURCE_VOTING_CLASSIFIER_PROPOSAL.md` |

- **Statistical rigor**: per the multi-seed lesson learned this session
  (single-seed decoder_final wobbled 90.1%→92.5% across two otherwise-
  identical runs) — any fine-tuned-tower result MUST be reported as
  mean ± std over multiple seeds (10, matching this session's precedent
  in `gemma_hidden_state_probe_multiseed.py`), not a single run. Given
  fine-tuning itself is stochastic (weight init on the newly-unfrozen
  layers, data shuffling, potentially dropout), multi-seed reporting
  matters even more here than it did for the (cheaper, CPU-only) linear
  probe — full fine-tuning runs are expensive enough that 10 full
  fine-tunes may not be practical; at minimum, report the best-of-N *and*
  a smaller multi-seed sample (e.g. 3) to bound variance, rather than
  trusting one run.
  Use the same "gap larger than max(std)" separation check this session's
  `gemma_hidden_state_probe_multiseed.py` already implements — informal
  but consistent with how every other comparison in this document was
  actually decided (decoder_early vs. others being the one location that
  survived that check, for example).
- **Fairness**: same held-out set, same class-weighting scheme, same
  checkpoint-selection-on-val discipline as every existing fine-tune in
  this codebase — no result should be reported from a checkpoint selected
  using test-set feedback.

## 7. Compute requirements (RTX 5060 Ti, 16GB VRAM)

**[A]** From `core/loaders/gemma_loader.py`'s own comments: the raw bf16
Gemma E2B-it weights (16.02GB) **exceed** this card's usable VRAM
(15.93GB) on their own when loaded without quantization headroom — this
is why `gemma_e4b.yaml` (the larger variant) already needs
`load_in_8bit=True` in this project. E2B itself is smaller (repo id
`google/gemma-4-E2B-it`) and per that same file's context is loaded at
`dtype="auto"` (bf16) without quantization in current production use —
**[B]** implying E2B alone likely fits in ~16GB for *inference*, but with
limited headroom for anything else.

**[B] Training adds substantially more than inference**: even freezing
the LM decoder and only training the vision tower's last-N layers still
requires holding activations for backprop through the *entire forward
pass* up to that point (the frozen LM's forward pass still needs to run
to produce the loss, even though its weights don't get gradients) —
gradient memory is only saved on the frozen parameters' optimizer state,
not on the activation memory needed to backpropagate through them. This
means a full fine-tune attempt (even "vision-tower-only, LM frozen") on
this card is a real VRAM risk, not a straightforward fit, unless memory-
saving techniques are used:

- **Mixed precision (bf16, likely already default via `dtype="auto"`)**:
  **[B]** already effectively the loader's default for E2B — a real,
  low-cost saving already banked, not something to newly enable.
- **Gradient checkpointing**: **[B]** advisable — trades recomputation
  time for activation memory, standard for this VRAM class, and this
  project already has training infrastructure precedent
  (`training/` LoRA pipeline per [[project_workflow_gui_preview_and_checkpoints]])
  that would be the natural place to check for an existing pattern to
  reuse rather than reinventing.
- **QLoRA / 8-bit quantized base + LoRA adapters**: **[B]** the single
  most VRAM-conservative option, and directly compatible with the
  already-confirmed `BitsAndBytesConfig` quantization path
  (`gemma_loader.py`'s `load_in_8bit` mechanism, already proven to work
  on this exact model family with the `llm_int8_skip_modules` exclusion
  for `vision_tower`/`embed_vision`/`lm_head` — **[A]** confirmed by a
  real bug already found and fixed in this codebase for *inference*
  quantization). Note directly: that existing mechanism *excludes*
  `vision_tower`/`embed_vision` from 8-bit quantization specifically
  because naive int8 quantization of the vision/projector path was
  measured to produce coherent-but-wrong output — **[A]** a real,
  already-documented finding. This means a QLoRA-on-the-vision-tower
  setup can't simply reuse the existing `load_in_8bit` config as-is; it
  would need the *opposite* pattern (vision tower LoRA-adapted at full/
  near-full precision, LM backbone quantized for VRAM savings instead) —
  **[C]** untested combination, flagged as a real design detail to get
  right rather than a copy-paste of the existing config.
- **VRAM estimate**: **[C]** not measured — the design doc's own
  constraint against inventing timing numbers applies equally to VRAM
  figures. A real estimate needs either a documented calculation (param
  count × precision × optimizer-state multiplier + activation memory for
  the frozen forward pass) or, more reliably, an actual smoke-test
  forward+backward step (the same kind of pre-flight check
  `vit_family_benchmark_common.py`'s benchmark used before committing to
  its full run) before trusting any number here.
- **RAM (host memory)**: **[A] worth flagging directly given this
  session's own incident** — the hidden-state extraction scripts built
  this session leaked host RAM badly enough to require an emergency kill
  (~150MB/image RSS growth, unresolved root cause, mitigated only by
  running extraction in per-chunk subprocesses). A fine-tuning run would
  hold the model resident for the *entire* training duration (no
  per-chunk process boundary to reclaim memory), so if the same
  leak-prone code paths are involved (e.g. `device_map="auto"`/accelerate
  hooks, suspected but not confirmed as the source), a long-running
  fine-tune could be more exposed to this risk than the chunked
  extraction was, not less. This should be smoke-tested for memory
  stability over enough steps to see whether RSS climbs before trusting
  a multi-epoch run to complete safely on this machine.
- **Expected training time**: **[C]** not estimated — no comparable
  Gemma-vision-tower fine-tune has been run in this project to anchor an
  estimate; the closest analogs (siglip: 32.3min, beit: 32.2min,
  dinov2: 36.7min for 8 epochs each, per the multi-architecture
  benchmark) are for stand-alone `timm` models without a frozen 2B+
  parameter LM's forward pass riding along on every step — the frozen-LM
  forward pass alone will add real per-step latency this analogy doesn't
  capture. A real estimate needs a timed smoke-test (a handful of steps),
  not a guess from unrelated numbers.

## 8. Decision Engine implications

Per the explicit framing already established for the hidden-state probe
work (`docs/GEMMA_HIDDEN_STATE_ROUTING_HEAD_DESIGN.md` question 8) and
this project's standing architectural principle
([[feedback_reduce_manual_touches_is_the_metric]]-adjacent: every stage
has a fallback/quarantine by design) — **[B] recommendation, conditioned
on a fine-tune actually being tried and measured**:

- **Not a replacement for the current vision towers** — the existing
  6-architecture fine-tuned ensemble (siglip/beit/dinov2/convnext/swin/
  vit21k, 89.8-92.5% each) already provides genuine complementarity per
  the multi-source voting proposal's own pairwise-disagreement analysis;
  a fine-tuned Gemma vision tower would be a **7th** family to evaluate
  for complementarity the same way, not an automatic swap-in.
- **An additional Decision Engine sensor** is the natural fit,
  structurally identical to how the hidden-state linear-probe sensor was
  already scoped to integrate (`EvidenceRecord(sensor_name="gemma_vision_
  finetuned_head", ...)`, same `score_kind="softmax_probability"`,
  `calibrated=False` until measured, same pattern documented in the
  hidden-state design doc's question 8).
- **A first-pass routing stage before full Gemma generation** is the more
  interesting operational framing *if* a fine-tuned tower closes some of
  the ~92%→94% gap: since it reuses the same forward pass Gemma's own
  vision tower already computes (no extra vision encoding needed), a
  fast/cheap first-pass classification could gate whether the full
  generation call happens at all — but this framing only pays off if the
  gap actually closes with fine-tuning, which question 3 argues is the
  least certain part of this whole proposal. **[C]** Worth deciding only
  after a real fine-tune experiment, not designed further in the abstract
  now.

---

## Summary recommendation

**[B]** Fine-tuning Gemma's vision tower is architecturally feasible and
mechanically straightforward given this project's own already-proven
last-N-layer unfreezing recipe — but the *expected gain* is genuinely
uncertain and, based on the closest available evidence (this project's own
6-architecture fine-tune benchmark), plausibly small: those architectures'
large gains (+21-35pts) came from starting near a 54-68% zero-shot floor,
while Gemma's frozen vision tower already sits at 92% with nothing more
than a linear probe. The more informative next step before committing
compute to a fine-tune run is the **cheap one flagged in question 3**:
look directly at which images the frozen-tower probe gets wrong, and
whether those errors look like a vision-representation gap (fine-tuning
territory) or a reasoning/ambiguity gap (not fixable by vision-tower
fine-tuning at all). That error analysis is CPU-only, reuses the already-
cached Stage 2 features and predictions, and would meaningfully sharpen
whether this proposal is worth the real VRAM/RAM risk and unresolved
training-time estimate documented in question 7.
