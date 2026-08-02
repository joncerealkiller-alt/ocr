# Canonical Vision Representation (Vision IR) — Architecture Research

**Status**: research only, no implementation. Nothing in this document
proposes changing production code. Written against the pipeline as it
exists on the `automated-sidecar-generation` branch, 2026-07-31.

**Terminology note (2026-08-02)**: this document's "Stage A / Stage B /
Stage C" are RESEARCH EXPERIMENT PHASES for the Vision IR proposal
specifically (CV analyser -> encoder+projector prototype -> full
integration test) - a different concept from the pipeline's canonical
Stage 0-6 orchestration naming in `docs/PIPELINE_STAGE_TERMINOLOGY.md`.
Do not conflate them: this doc's "Stage A" roughly parallels the
pipeline's operational Stage 1 (Raw Sensor Capture) in subject matter,
but names an experiment phase, not a pipeline stage - left as-is
throughout the rest of this document rather than renamed, since
renaming would misrepresent what these labels actually mean.

**Scope note**: this assesses whether a Vision IR is worth building as
an *experimental branch*, not how to build it. Where the answer is
"yes, but," the "but" is the point — this corpus (archival scans,
hundreds of thousands eventually) and this loader roster (24 unrelated
VLM families, see below) are unusually hostile to a shared-encoder
architecture, more so than a green-field multimodal project would be.

---

## 0. What the pipeline actually does today (grounding)

Every loader (`core/loaders/*.py`) subclasses `BaseLoader`
(`core/loaders/base_loader.py:265`) and owns the full chain: load image
→ encode → project → prompt → generate → parse. `_run_generate(raw_image,
prompt)` is the one abstract method every model implements differently,
because there is no shared vision step to factor out today — e.g.
`GemmaLoader.initialize_model_and_tokenizer()` loads via plain
`AutoModelForCausalLM.from_pretrained()` (Gemma-3/4's vision tower is
internal to that single class, no separable encoder object exposed at
the loader level).

`config/models/*.yaml` currently configures **24 model profiles** across
genuinely unrelated architectures: Qwen2.5-VL / Qwen3-VL family (native
dynamic-resolution ViT), InternVL3 (InternViT-6B or 300M depending on
size), Florence-2 (DaViT), Pixtral (own vision encoder + 2D RoPE),
DeepSeek-VL2 (hybrid SigLIP + SAM-B), GLM-OCR, Granite Vision, SmolVLM2,
Moondream2, Hunyuan-OCR, LFM2-VL, Nanonets-OCR2, Chandra, GOT-OCR2,
olmOCR (a Qwen2.5-VL fine-tune), gemma. Several run in separate
subprocess venvs (`core/loaders/subprocess_loader_base.py`,
`_moondream_worker.py`, `_deepseek_vl2_worker.py`,
`_hunyuan_ocr_worker.py`) because their dependency trees conflict with
the main interpreter's.

Separately, `core/image_analysis.py` already implements a **pure
deterministic CV measurement stage** ("Stage A") that is architecturally
the deterministic half of what a Vision IR would contain: geometry
(aspect ratio, deskew angle, ruling-line counts/angles), per-region
tone/quality stats (Otsu threshold, ink fraction, polarity, stroke
width, text height, blur, noise), two ROI detectors (page boundary,
table boundary) with ordinal confidences. It explicitly emits numbers,
not categories, so that binning/policy logic (Stage B, not yet built)
can be retuned without re-measuring. This is a working precedent worth
building on rather than reinventing.

There is **no existing shared vision encoder, no cached embeddings, and
no "vision provider" abstraction** anywhere in the codebase today. A
Vision IR would be new architecture, not a refactor of something
partially built.

---

## 1. What belongs in a Vision Representation

### Deterministic (classical CV — cheap, reproducible, versionable by algorithm)

These already exist, in whole or in near-whole form, in
`core/image_analysis.py`:

- Image hash (perceptual + cryptographic — not yet in `image_analysis.py`,
  trivial to add; both are useful for different jobs: crypto hash for
  exact-duplicate/tamper detection, perceptual hash for near-duplicate
  scan detection)
- Blur / sharpness (`blur_laplacian_var`)
- Skew angle (`deskew_angle_deg`, `dominant_vertical_angle_deg` — the
  code map already documents that the former silently reads 0.00 on
  ~90% of one sub-corpus and the latter is the more trustworthy signal
  there; a Vision IR must carry both, not collapse them into one field)
- Contrast / illumination metrics, Otsu threshold, ink fraction, polarity
- Text density / text height, stroke width
- Edge / ruling-line statistics, page and table boundary boxes with
  confidence

**Why deterministic**: every one of these is a closed-form function of
pixels. Given the same image and the same algorithm version, the output
is bit-identical forever. That's the whole value proposition — no GPU,
no model drift, trivially cheap to compute for 100k+ images, and
auditable (a reviewer can literally recompute it by hand). The one
caveat already discovered in this codebase: "deterministic" does not
mean "context-free" — `image_analysis.py`'s tone fields are *only*
meaningful on a page-cropped image (the LAC microfilm finding: 43/218
pages misflagged as inverted-polarity before cropping the black film
surround). A deterministic field can still have an implicit dependency
on an upstream deterministic step (crop-before-tone is a real ordering
constraint, not a nice-to-have).

### Learned (vision-encoder output — semantic, model-dependent, not closed-form)

- Semantic layout / document structure ("this looks like a tabular
  form with N columns" as a *belief*, distinct from
  `table_boundary`'s pixel-geometric box)
- Document-type / subtype embedding or classification signal (what
  Stage 1/2 of this pipeline currently do with a full generative VLM
  call — a candidate for replacement by a cheap embedding + classifier
  head, discussed in §6)
- Duplicate/near-duplicate similarity via embedding distance (different
  from a perceptual hash: catches "same page, different scan/crop/skew,"
  not just "same bytes")
- Orientation confidence, photograph-vs-document-vs-map discrimination
- General "document understanding" embedding for downstream retrieval or
  clustering, if ever wanted

**Why learned**: none of these are computable in closed form — they
require a model that has seen enough documents to generalize "this is
laid out like a ledger" or "this photo is upside down" from pixel
statistics that don't reduce to an edge-detector rule. The corollary is
the risk: these outputs are only as stable as the encoder checkpoint
producing them (see §7 versioning), and they are not human-auditable
the way a Laplacian variance number is.

### The dividing line, stated plainly

If you can write the field's definition as a formula a person could
verify by hand on one image, it's deterministic and belongs in Stage A
territory. If the field's definition is "whatever this checkpoint's
training data taught it to represent," it's learned, and every
downstream consumer inherits that checkpoint's opinions and biases as a
silent dependency. Don't let "the vision encoder could also detect
blur" tempt you into moving a deterministic field into the learned
bucket — cheaper, more auditable, more reproducible always wins when
both routes are available, which is most of the fields above.

---

## 2. Shared vision encoder — current landscape

This project's loader roster is itself evidence here: **10+ different
vision-encoder families are already in production use** (Qwen's native
ViT, InternViT, DaViT/Florence, Pixtral's own tower, SigLIP+SAM in
DeepSeek-VL2, and others per-model). That heterogeneity is the central
fact any "one shared encoder" proposal has to survive.

Brief characterization of the named candidates:

- **CLIP** (OpenAI) — the original contrastive image-text encoder.
  Largely superseded as a *backbone* in 2024-2026-era VLMs by SigLIP
  variants, but still common as an off-the-shelf embedding model for
  retrieval/clustering tasks outside a VLM.
- **SigLIP / SigLIP 2** (Google) — sigmoid-loss contrastive encoder,
  now the most common "generic" vision backbone dropped into new VLMs
  (Gemma's own vision tower descends from this family). Good general
  semantic embeddings; not trained for document/OCR-dense layouts
  specifically, so text-in-image fidelity is weaker than an
  OCR-purpose-built encoder.
- **DINO / DINOv2** — self-supervised, no text supervision at all. Very
  strong for pure visual similarity/clustering (duplicate detection,
  layout clustering) precisely *because* it isn't entangled with a
  particular language decoder. This is the most promising candidate
  specifically for the "learned similarity/duplicate/layout" fields in
  §1, decoupled from any generation task.
- **Florence-2** — Microsoft's DaViT-based encoder, unusual in that the
  same checkpoint is trained across detection/captioning/OCR/grounding
  tasks jointly, so its representation is somewhat more task-general
  than a pure contrastive encoder. Already in this project's loader
  roster (`florence_loader.py`) — a real, running data point rather than
  a hypothetical.
- **InternViT** — the encoder behind InternVL, trained at very large
  scale with strong document/chart/OCR representation specifically
  called out in its own reporting. Also already in this project's
  roster (`internvl_loader.py`), at two sizes (8B, 2B variants
  configured).

**Feasibility read**: a single shared encoder *for embeddings used to
drive preprocessing/routing decisions* (duplicate detection, layout
classification, orientation) is realistic — DINOv2 or SigLIP2 run once
per image, independent of which generative VLM eventually reads the
document, is a genuinely separable, low-risk slice. A single shared
encoder that *replaces* the vision towers baked into Qwen-VL, InternVL,
Florence, Pixtral, DeepSeek-VL2, etc. for their own generation task is
not realistic — see §3.

---

## 3. Projection layers — can models share an encoder's output?

**No, not with today's off-the-shelf checkpoints, and this is the load-
bearing finding of this whole document.**

A VLM's projection layer (the MLP or cross-attention adapter mapping
vision-encoder patch embeddings into the language model's embedding
space) is trained jointly with — and only with — that specific
(encoder, language model) pair, on that lab's specific instruction-tuning
data. The projection layer isn't a generic adapter; it's calibrated to
the exact geometry and statistics of its own encoder's output and its
own decoder's input space. Swapping in a foreign encoder's embeddings
means feeding the projection layer inputs it has never seen an
in-distribution example of — there's no reason to expect coherent
output, and every public VLM's release process trains the projector
from scratch (or heavily fine-tunes it) whenever the encoder changes,
which is direct evidence the labs themselves don't treat this as a
solved swap.

Concretely, in this project's own roster: Qwen2.5-VL's projector was
never trained against InternViT features, and InternVL's projector was
never trained against Qwen's ViT features. There is no released
adapter bridging them. Building one would be a nontrivial fine-tuning
project in its own right (collect paired data, train a new
projection/adapter layer per target LM, validate it doesn't regress
that LM's instruction-following) — i.e., exactly the kind of
implementation work this document is explicitly not scoping.

**What *is* real and already productized**: architectures like
LLaVA-style "one encoder, swap the LM" setups exist, but only within a
single training lineage that deliberately designed for it (e.g. a lab
releasing several LM sizes against one frozen encoder + newly-trained
projector per size). That is "the same lab trained N projectors for one
encoder," not "any encoder's raw output is consumable by any model's
existing projector." It doesn't generalize to this project's
already-heterogeneous, independently-sourced loader roster.

**Practical implication for this pipeline**: a shared encoder can feed
*non-generative* consumers directly (embedding-distance duplicate
detection, a small trained classifier head for document-type routing)
because those consumers can be trained fresh against whatever encoder
you pick. It cannot feed the *existing* generative loaders' internal
projection layers without also retraining or fine-tuning each of
those — at which point you're no longer reusing those loaders as-is,
you're building 10+ new adapter-tuning projects.

---

## 4. Loader architecture implications (discussion only)

Given §3, the realistic shape is narrower than the prompt's "Vision
Provider → Loader (project, prompt, generate)" sketch implies for the
*generative* loaders — their internal encode+project step can't be
outsourced without retraining. Where a Vision Provider stage genuinely
slots in cleanly is **upstream of loader dispatch entirely**, alongside
(not replacing) `core/image_analysis.py`'s existing Stage A:

- A Vision Provider that runs once per image, producing IR fields
  (§1) consumed by *pipeline decisions* — which bucket to route to,
  whether to deskew, whether to flag as a likely duplicate — is
  additive to the current `BaseLoader` contract, not a change to it.
  Every existing loader keeps doing exactly what it does today.
- The IR could reasonably be added as a documented *input* a loader's
  prompt-builder is allowed to reference (e.g. "table_confidence: 0.9"
  folded into the prompt text as a hint), the same way
  `column_regions_approx` from `document_templates.py` already
  conditions per-field crops today. That's a prompt-engineering
  consumer, not an encoder-sharing one, and needs no projection-layer
  work at all.
- What the prompt's sketch would require to be literal (loaders
  consuming the *encoder's tensor output*, only doing project→prompt→
  generate themselves) is blocked by §3 for every current loader except
  ones you deliberately fine-tune a new projector for — a real,
  scoped, per-model research project, not a loader-abstraction change.

So: "Vision Provider produces an IR that downstream *decisions* consume"
is architecturally sound and roughly additive to what exists.
"Vision Provider produces tensors that downstream *models* consume in
place of their own encoders" is not achievable without per-model
retraining work this document is scoped to exclude.

---

## 5. Preprocessing decisions informed by semantic understanding

The prompt is explicit that semantic understanding should *inform*,
not *replace*, deterministic CV here — and this project's own measured
history supports that framing directly. From `docs/CODE_MAP.md`'s
recorded findings:

- The LAC microfilm corpus's black-film-surround problem was diagnosed
  and fixed with **pure deterministic CV** (`largest_ink_blob_frac`
  gating tone-driven policy) — no learned model was needed once the
  right deterministic signal was identified.
- Auto-dewarp's real finding was that **the corpus barely warps at
  all** (median 0.12–1.08% of page width on the labeled set) — the
  original assumption that a learned corner-detector was needed turned
  out to be solving a near-nonexistent problem; classical
  Canny/`approxPolyDP` quad detection already gets a defensible answer
  (14/16 detected, median 1.95% error).
- The column-registration investigation found the real bug was a
  **registration/anchor** problem (a 2-point affine fit fixes 1911/1921
  almost entirely), not a place where semantic understanding was
  missing — more learned signal would not have found this; more
  careful deterministic geometry did.

That history argues for a specific, narrower role for learned signal in
preprocessing than "detect everything": use it for the genuinely
semantic questions deterministic CV structurally cannot answer —
"is this a screenshot vs. a scan," "is this a map vs. a ledger vs. a
photograph" (already partly done today via the Stage 1/2 classification
VLM calls), "does this look like a duplicate of a page we've already
processed" — and keep the deterministic layer as the arbiter of
questions with a closed-form geometric answer (skew angle, crop
boundary, contrast). The `table_confidence` calibration finding (real
tables score 0.375+, most non-table pages score under 0.25) is exactly
the kind of thing a learned signal should defer to, not duplicate: it's
already solved deterministically and works.

Concretely, a Vision IR's learned fields could *gate* which
deterministic preprocessing step even runs (e.g., "orientation
confidence low → don't trust `deskew_angle_deg`, route to manual
review" rather than a learned model re-deriving the angle itself) —
consistent with how this pipeline already treats every stage as
producing a confidence that downstream logic gates on, not a silent
guess.

---

## 6. Architectural comparison

**A — Current (per-model CV + independent vision encoding)**
- Maintainability: fragmented but *isolated* — a bug in one loader's
  vision handling can't leak into another's (already a deliberate
  property per `CLAUDE.md`'s "don't run two models concurrently" and
  the load-once-release-fully lifecycle convention in `CODE_MAP.md`).
- Explainability: each model's behavior is self-contained; no shared
  failure mode across models.
- Reproducibility: high per-model (config hash traces every output to
  exact settings already — `GenerationConfig.content_hash()`).
- Extensibility: adding model #25 is pure addition, zero coupling to
  existing 24.
- Engineering complexity: low incremental cost per model, but O(n)
  duplicated "did I load/crop/normalize the image right" logic across
  loaders.
- Model independence: total — this is the current roster's proven
  strength (10+ unrelated architectures coexisting).

**B — Vision-only (encoder does everything, models consume its output
directly)**
- Blocked as stated by §3 for this project's actual loader roster.
  Would require retraining/fine-tuning a projector per target LM,
  which is a different, much larger project than "add a shared
  encoder."
- If pursued anyway (ignoring feasibility), maintainability and
  explainability get *worse*, not better — a shared-encoder bug now
  silently propagates into every downstream model instead of staying
  isolated, the opposite of what this pipeline's design has optimized
  for so far (see `feedback_reduce_manual_touches_is_the_metric` and
  the load-once-release discipline).

**C — Hybrid (CV → Vision → Canonical IR, consumed by decisions and
prompts, not by replacing encoders)**
- Maintainability: adds one new stage with a clear, narrow contract
  (produce IR fields, versioned) — comparable in shape to how Stage A
  (`image_analysis.py`) was added without disrupting the classifier or
  extraction stages.
- Explainability: improves *if* deterministic vs. learned fields stay
  clearly labeled (per §1's dividing line) — degrades if the IR becomes
  a grab-bag where callers can't tell which fields are formula-backed
  vs. checkpoint-opinion-backed.
- Reproducibility: deterministic fields stay perfectly reproducible;
  learned fields inherit standard ML-versioning obligations (§7).
- Extensibility: genuinely the strongest case for this option — new
  preprocessing/routing logic can consume the IR without re-touching
  loader code, matching this codebase's existing preference (per
  `docs/CODE_MAP.md`'s architectural principle) for "config/data owns
  behavior, code owns mechanics."
- Engineering complexity: moderate, additive — the IR sits *beside* the
  existing loaders, not inside their abstract contract, so no existing
  loader needs to change.
- Model independence: fully preserved for generation (every loader
  keeps its own encoder); a new, *separate* dependency is introduced on
  whichever encoder produces the learned IR fields.

**Recommendation**: **C, scoped narrowly** — build a Vision IR as an
upstream *decision-support* artifact (deterministic fields expanding on
`image_analysis.py`'s Stage A, learned fields from one general-purpose
encoder like DINOv2 or SigLIP2 for duplicate/layout/orientation
signals), consumed by preprocessing and routing logic and optionally
folded into prompts as hints. Do **not** attempt to route it through
loaders' own projection layers or treat it as a replacement for any
existing loader's internal vision encoding — that's architecture B in
disguise and hits the same wall documented in §3.

---

## 7. Risks

- **Embedding versioning**: a learned field's meaning is pinned to one
  encoder checkpoint. Any encoder upgrade invalidates every stored
  embedding for re-comparison purposes (duplicate detection breaks
  silently if half the corpus is embedded with checkpoint v1 and half
  with v2 — cosine distances aren't comparable across checkpoints).
  Mitigation direction: store an explicit encoder version/hash
  alongside every learned field, the same discipline this codebase
  already applies to generation config (`content_hash()`).
- **Cache invalidation**: unlike deterministic fields (safe to treat as
  eternally valid for a given algorithm version), learned fields need
  an explicit "recompute if encoder version changed" rule, and a
  decision about whether old embeddings get backfilled or left stale.
- **Encoder replacement**: swapping the shared encoder (e.g. DINOv2 →
  DINOv3) is not a drop-in — anything downstream trained against v2's
  embedding space (a classifier head, a similarity threshold) needs
  retraining or recalibration. This is a real, recurring maintenance
  event, not a one-time cost.
- **Compatibility with future models**: adding model #25 to the loader
  roster gains nothing automatically from the IR unless that model's
  own prompt-builder is deliberately wired to consume IR fields as
  hints — the IR does not make new models "free" to integrate, it adds
  one more optional input they can choose to use.
- **Projection layer coupling**: covered in depth in §3 — the central
  risk of over-scoping this project is assuming the IR can feed
  existing models' internal projectors. It can't, without per-model
  retraining this document explicitly excludes from scope.
- **Maintenance burden**: a second "vision understanding" surface
  alongside 24 independent loaders is more moving parts, not fewer,
  unless it demonstrably *removes* duplicated logic from those loaders
  (it currently wouldn't, per §4 — loaders keep their own encode step
  regardless).
- **Storage costs**: at "hundreds of thousands of archival images"
  scale, storing a full embedding vector (typically 768–1536 floats,
  i.e. 3–6KB) per image per encoder version is on the order of
  hundreds of MB to low GB per full-corpus pass — cheap in absolute
  terms, but multiplies with every encoder version kept for
  reproducibility, and dwarfs the deterministic fields (a few hundred
  bytes of JSON per image, as `image_analysis.py` already produces).
- **Reproducibility**: deterministic fields are reproducible by
  construction. Learned fields are only reproducible if the exact
  encoder checkpoint, preprocessing (resize/normalize), and inference
  settings are pinned and recorded — the same audit-trail discipline
  `GenerationConfig.content_hash()` already enforces for generation,
  which a Vision IR would need to replicate for its own learned stage.

---

## 8. Suggested experimental roadmap (if pursued)

Sequenced so each step is independently useful and cheaply abandon-able
if it doesn't pay off — no step commits to the next.

1. **Extend `image_analysis.py`'s deterministic fields** (image hash,
   perceptual hash) — zero new dependencies, zero risk, immediately
   useful for duplicate detection on its own, no encoder needed yet.
2. **Stand up one general-purpose encoder (DINOv2 or SigLIP2) as a
   standalone embedding step**, entirely separate from any loader —
   run once per image in the existing preprocessing stage, store the
   embedding + explicit checkpoint version. Evaluate purely on
   duplicate/near-duplicate detection accuracy against a small hand-
   labeled set, the same measurement discipline already used for
   `table_confidence` and the dewarp corner-error baseline.
3. **Only if step 2 shows real signal**: train a small classifier head
   (not a new VLM) on top of the frozen embeddings for one narrow task
   — e.g. orientation confidence, or document-type routing as a
   cheaper alternative to a full generative classification call at
   Stage 1. Compare against the current Gemma-based classifier's
   accuracy and cost, the same way `prompt_sweep.py` already compares
   model/prompt combos.
4. **Explicitly do not attempt** cross-model projection-layer sharing
   (architecture B) as part of this roadmap — §3's finding should be
   revisited only if a specific released adapter/projector becomes
   available that bridges two of this project's actual encoder
   families, not attempted as a from-scratch research effort under
   this initiative.
5. Any of the above should land in a genuinely separate experimental
   directory/branch, per your stated constraint, with the production
   pipeline (`core/`, `config/models/*.yaml`, existing loaders)
   untouched until the experiment independently demonstrates a
   reduction in manual review touches — the metric this project already
   uses to judge pipeline changes (`feedback_reduce_manual_touches_is_the_metric`),
   not raw accuracy or architectural elegance on its own.

---

## Addendum — follow-up research (2026-07-31)

Research only, same constraints as above. Extends §§3/7/8 with a
specific tooling question, an evaluation of a proposed staged
prototyping plan, and a literature check on Gemma-specific prior art.

### 9. Does `timm` provide anything for projector training?

**No — `timm` is scoped to the encoder/backbone half only, and doesn't
touch the alignment/projector half at all.** Concretely, per its own
documentation, `timm` provides: pretrained backbone architectures
(700+), layers/utilities, optimizers, schedulers, dataloaders,
augmentations, and training/validation scripts oriented around
reproducing **ImageNet-style classification training**. There is
nothing in `timm`'s scope — no dataset, no reference script, no
established recipe — for text-image contrastive alignment, projector
architectures, or VLM instruction-tuning data. That work lives in an
entirely separate ecosystem (LLaVA's own training code, `open_clip` for
contrastive alignment, or hand-rolled Trainer loops against HF
`transformers`' multimodal model classes).

Practical read for this project: if the §6 recommendation (a `timm`-
hosted general encoder like DINOv2, used standalone for the
decision-support IR) is pursued, `timm` is a clean, sufficient source
for that encoder — no alignment/projector work is needed there because
nothing downstream consumes its raw embedding through a trained
projector, only through embedding-distance comparison or a freshly-
trained classifier head (both of which `timm` also doesn't need to
provide — those are ordinary downstream-task training, well-covered by
plain PyTorch/sklearn). If the excluded route (§3/§4 — swapping a
loader's internal vision tower) were ever attempted anyway, `timm`
would supply the encoder and nothing else; the projector-alignment
data and training loop would need to be built from scratch against a
different toolchain, which is itself a nontrivial part of why that
route is excluded from scope.

### 10. Evaluating Jon's staged plan

**Stage A (Vision IR only, measure whether it improves routing/
classification accuracy) — sound in shape, under-specified on the
success criterion.** This matches the roadmap's steps 1–2 in §8
directly. Two gaps worth closing before running it:

- **"Accuracy" needs to be decomposed per decision**, not treated as
  one number. The IR's candidate fields (§1) inform genuinely different
  decisions — duplicate detection, orientation/deskew gating, and
  document-type/subtype routing are different tasks with different
  correct-answer definitions. A blanket "did accuracy improve" question
  will conflate them and produce an ambiguous result even if one field
  clearly helps and another clearly doesn't. Stage A should state, in
  advance, which specific field is hypothesized to help which specific
  decision, and what metric/threshold counts as "worked" for each pair.
- **A labeled evaluation set for classification/routing barely exists
  yet.** `data/outputs/ground_truth_log.jsonl` is field-extraction
  ground truth (~3 pages, ~750 field records per `docs/CODE_MAP.md`),
  not classification/routing ground truth — a different eval set. Stage
  A's "measure whether accuracy improves" step has no real denominator
  to measure against until that set exists (or an existing one — e.g.
  manual sessions already logged through `ui/classifier_validation_ui.py`
  — is repurposed for this). Building or confirming that set is a
  precondition for Stage A producing a real signal, not an optional
  nicety.

**Stage B (one encoder + one projector + Gemma, subset of corpus) —
correctly targets the highest-value loader, but is scoped as the
single least-precedented technique in this research, not a safe next
step.** Two things worth separating:

- **The premise is confirmed correct against the actual code.**
  `config/models/gemma.yaml` (`google/gemma-4-E2B-it`) and
  `core/loaders/gemma_loader.py`'s `_run_generate()` show Gemma is
  genuinely multimodal here — every call passes `images=raw_image`
  into the processor alongside text, and GemmaLoader is the loader
  behind both Stage 1 and Stage 2 classification, i.e. it really does
  see every single image that enters the pipeline. "Highest-value place
  to improve" is well-supported, not just asserted.
- **But this is exactly the case §3 already flagged as highest-risk.**
  Freezing an LM and retraining only a projector after swapping the
  vision tower is a real, demonstrated *general* VLM technique (see §11
  below) — but §11 found no confirmed successful precedent doing this
  specifically to a Gemma-family model, and the one direct practitioner
  attempt at almost exactly this (swapping SigLIP versions inside
  PaliGemma2) was reported as "quite difficult" with no completion
  reported. So Stage B isn't "apply a well-trodden recipe to our
  priority target" — it's "attempt the specific unproven case, on the
  model we care most about." That's a legitimate thing to prototype
  cheaply, but it should be scoped and budgeted as research with a real
  chance of a dead end, not as a routine validation step before Stage C.
- **A concrete unknown Stage B would need to resolve first**: this
  research (and the literature it drew on) is grounded in PaliGemma and
  Gemma 3's documented architectures (SigLIP-family encoder + a
  separate `MultiModalProjector` module). This pipeline's actual
  checkpoint, `google/gemma-4-E2B-it`, is a newer generation whose
  internal module structure (whether the vision tower and projector are
  still cleanly separable the same way) isn't confirmed by anything
  gathered here — that needs to be checked against the actual model
  class/config before Stage B's scope is even well-defined, let alone
  attempted.
- **Is Stage B "as scoped" a meaningful test of Stage C's premise?
  Only partially.** Training one projector on a corpus subset gets a
  cheap directional signal ("does this even produce coherent output"),
  but can't validate the two things Stage C's production premise
  actually depends on: (1) whether the recipe generalizes across this
  project's unusually heterogeneous corpus (handwritten ledgers,
  microfilm, maps, portraits — not what a general-purpose encoder/
  projector recipe is normally validated against), and (2) whether
  classification accuracy on *this project's specific taxonomy* matches
  or beats Gemma's native vision path, not just "the model produces
  plausible-looking text." Recommend Stage B define its own explicit
  gate in advance — e.g. matched-or-better accuracy on the same
  classification eval set Stage A should have built — rather than
  "training converged" or "outputs looked reasonable" as the bar for
  greenlighting Stage C.
- **Stage A and Stage B are testing different hypotheses, not
  sequential validation of one idea** — worth naming explicitly since
  the staging could be misread otherwise. Stage A tests "does more
  signal improve pipeline decisions" (the additive, architecture-C
  question this document already recommends). Stage B tests "can
  Gemma's vision tower be replaced" (the excluded, architecture-B
  question §3 flagged as unresolved). Stage A succeeding is not
  evidence that Stage B is likely to succeed — they're independent
  bets. Jon's plan already gates C on B specifically (not on A), so the
  sequencing is structurally fine; this is just flagging that "Stage A
  went well" shouldn't be read as momentum toward Stage B.

**Stage C (replace Gemma's vision tower for real, only if B succeeds)**
— correctly contingent, no issue with the gating itself. The overall
risk ordering (A cheap/reversible → B moderate-cost/subset-scale,
explicitly the risky unproven bet → C only on B's proof) matches the
"each step independently useful and abandon-able" principle from §8's
original roadmap, and is a sound shape for a staged plan. The
adjustments above are about sharpening what "success" means at each
gate, not about the ordering itself.

### 11. Prior art: has a Gemma-family encoder been swapped via
projector-only retraining with a frozen LLM?

Direct search, reporting what's demonstrated vs. theoretical:

- **PaliGemma** (SigLIP-So400m + Gemma-2B, linear projector) and
  **Gemma 3**'s multimodal architecture (a tailored SigLIP variant +
  a `MultiModalProjector` producing 256 fixed soft tokens) are both
  *co-designed and jointly pretrained* pairings released by Google —
  neither is an example of an encoder being swapped into an
  already-trained Gemma after the fact.
- **Closest lead, not confirmed**: "Visual Lexicon: Rich Image Features
  in Language Space" (arXiv 2412.06774, ViLex) reports its encoder
  "consistently improving vision-language model performance across 15
  benchmarks relative to a strong SigLIP baseline." This is the right
  shape of claim, but fetching the paper's abstract/metadata was not
  sufficient to confirm whether the downstream model in that comparison
  was specifically PaliGemma/Gemma-family with the LLM frozen and only
  a projector retrained, versus a different downstream LM or a
  fully-joint fine-tune. Flagged as a lead worth reading in full before
  relying on it, not as a confirmed precedent.
- **The general technique is real and published — on other model
  families.** Encoder-swap-plus-projector-only-retraining with a frozen
  LLM is a documented methodology: "Do VLMs Need Vision Transformers?
  Evaluating State Space Models as Vision Encoders" (arXiv 2603.19209,
  LLaVA-style controlled encoder swaps under a fixed recipe), "Encoder
  Winners Do Not Reliably Transfer Across VLA Backbone Scale" (arXiv
  2606.14153, explicitly projector-only training for a fixed step
  count while swapping encoder identity), and "Let ViT Speak" (arXiv
  2605.00809, encoder replacement + LM instruction-tuning). All three
  demonstrate the mechanism works *somewhere* — none of them target a
  Gemma-family LM.
- **One direct practitioner data point, on almost exactly this
  question**: a Hugging Face forum thread ("Custom VLM - Swapping a
  vision encoder from a VLM") has someone attempting to replace SigLIP1
  with SigLIP2 inside PaliGemma2 (and separately, pairing SigLIP2 with
  Qwen2.5-VL). The only response characterizes the task as "quite
  difficult" and points to generic HF custom-model documentation — no
  reported completion, benchmark, or working recipe.

**Verdict**: the general recipe (freeze LLM, replace encoder, retrain
only the projector) is real and demonstrated in VLM research broadly.
A confirmed, reported success doing this specifically to a Gemma-family
model was **not found** — the closest lead is unconfirmed at the
specificity needed, and the one direct practitioner attempt reported
difficulty rather than success. This reinforces rather than
contradicts §3's original risk assessment: theoretically well-grounded
as a general technique, practically unproven for this specific model
family. Since this pipeline's actual checkpoint (`gemma-4-E2B-it`) is
newer than anything covered by the sources above, the real difficulty
here is presently unknown in either direction — which is precisely
what makes Stage B a genuine experiment rather than a formality.

---

## Second addendum — evaluating the cross-model consensus reframing
(2026-07-31, same day, follow-up)

Jon ran the same question through Gemini, Grok, and Claude Opus
independently; their reports converged on a significant reframing —
skip encoder replacement entirely, first test whether Gemma's own
frozen vision encoder already separates the pipeline's 8 routing
buckets well enough for a lightweight linear/kNN classifier, before
touching the decoder or projector at all. Evaluated below against this
repo's actual code, not against the report's framing on faith.

### 12. Is this really an 8-bucket closed-set problem?

**Yes, confirmed** — `core/schema.py:29`'s `DocumentCategory` enum has
exactly 8 members (`dense_tabular_rows`, `handwritten_ledger`,
`printed_document`, `portrait_photo`, `map_land_record`,
`mixed_text_image`, `genealogy_chart`, `uncertain_review`), matching
`config/pipeline.yaml`'s `buckets:` section one-for-one. Gemma
(`GemmaLoader`, via `core/classifier.py`) is the only VLM call in this
routing step.

Also confirmed, and a genuine point in the report's favor: this
project's **other** classification step — subtype/template
identification (e.g. telling 1911 vs. 1921 vs. 1931 census forms
apart) — is already **not** a VLM call. `core/document_classification.py`
is deliberately rule-based CV (aspect ratio + ruling-line counts +
row-count estimate), explicitly built to run before any loader is
touched. So Gemma's classification role in this pipeline really is
scoped to exactly the one 8-way call the report assumes — that
narrowing is accurate, not an oversimplification.

**One thing the report's framing glosses over**: `ClassificationResult`
(`core/schema.py:40`) is not just a bucket label. It also carries
`confidence`, `text_density`, `handwriting`, `table_layout`, `faces`,
`map_like`, and a length-capped `reason` justification string — all
written to the bucket CSVs today (`core/classifier.py`'s `CSV_FIELDS`).
A linear probe naturally replaces the category decision only. Whether
those auxiliary fields are used by anything downstream wasn't checked
here, but whatever the answer, it needs to be an explicit decision when
scoping this experiment, not a silent side effect of swapping Gemma out.

### 13. Is a pooled embedding actually easy to extract from `GemmaLoader` as it stands?

**Not "for free" — this needs new code, and the report's own Stage 0 is
the right way to find out how much.** `GemmaLoader.initialize_model_and_tokenizer()`
(`core/loaders/gemma_loader.py:36`) loads via plain
`AutoModelForCausalLM.from_pretrained()`, not a dedicated
image-text-to-text/conditional-generation class. That's already an
unusual wrapper choice among typical VLM HF classes — PaliGemma's and
Gemma 3's official classes expose a `get_image_features()`-style
accessor precisely because they're built as conditional-generation
classes with a separable vision path. Whether `google/gemma-4-E2B-it`'s
`AutoModelForCausalLM`-compatible class exposes an equivalent clean
accessor, or requires a raw forward hook on an internal submodule to
capture pre-projection features, is **not confirmed by anything read in
this pipeline** — the loader code has no such accessor today; `_run_generate`
only ever returns decoded text.

This is exactly the unknown the cross-model consensus's own "Stage 0:
interface inspection before extracting embeddings" step is designed to
resolve, and it's well-justified for precisely this reason — endorse
it as a necessary first step, not a skippable formality. Until it runs,
"embedding extraction is easy" is an assumption, not a confirmed fact.

### 14. Does a labeled routing/validation dataset already exist?

**No — and the real gap is bigger than "needs light labeling."**
`ui/classifier_validation_ui.py` was deliberately built *without*
Correct/Incorrect verdict capture — per `docs/CODE_MAP.md`, its
`self.future_frame` is a reserved-but-empty placeholder for exactly
that feature, not yet built. There is no existing human-verified "was
this bucket assignment right" record anywhere in this repo. The bucket
CSVs (`data/buckets/*.csv`) are Gemma's **own** predictions, not ground
truth — training or evaluating a probe against them would just measure
agreement with Gemma, not correctness.

**Update (2026-07-31, same day, post-full-corpus run)**: the
71/1/183/15/29/4/20/0 counts cited above were from a partial run and
are now stale — noted here rather than silently corrected, since the
gap between the two readings is itself informative about how fast this
number moves. Re-counted directly against the current
`data/buckets/*.csv` files after Jon ran classification over the full
corpus: **385 / 12 / 220 / 32 / 87 / 103 / 1430 / 16**
(`dense_tabular_rows` / `handwritten_ledger` / `map_land_record` /
`mixed_text_image` / `genealogy_chart` / `portrait_photo` /
`printed_document` / `uncertain_review`), ~2,285 rows total. This is a
materially different, much more workable picture — `handwritten_ledger`
went from 1 example to 12, and even the smallest bucket now clears the
"can't train or evaluate at all" floor, though 12 and 16 are still thin
for a held-out test split on top of a training split.

**The circularity problem (§14, general point) is unchanged by volume
alone, though**: these are still Gemma's own predictions, not verified
ground truth — more rows doesn't fix that a probe trained/evaluated
against them would just measure agreement with Gemma. What actually
resolves it is the manual QA pass Jon is running right now, using the
review UI, flagging misclassification, bad deskew, and
needs-new-bucket cases per image — that pass is precisely the
Correct/Incorrect verdict capture `docs/CODE_MAP.md` noted as an unbuilt
placeholder in that UI. Once even a subset of buckets has been
manually confirmed/corrected this way, that subset (not the raw
per-bucket row counts) is the real number that determines whether
Stage 0.5 has enough labeled signal to run — re-check against whatever
that pass produces, not against either the old or the new raw
classifier-output counts above, before treating the validation-set
prerequisite as satisfied. The general lesson from the two readings
here: **any specific count in this document is a snapshot, not a
constant** — re-verify against the live bucket CSVs (and, once it
exists, the manual-verdict data) at decision time rather than citing
this file's numbers going forward.

### 15. Is the Opus reframe (skip the decoder, linear-probe pooled features) sound?

**The core instinct is right, and a genuine improvement over the
original encoder-replacement framing.** Linear probing a frozen
encoder is well-established practice elsewhere in the field (standard
for evaluating self-supervised/contrastive encoders), and matching
architecture complexity to problem complexity — not running a closed-set
8-way decision through a full generative pipeline — is sound
engineering judgment. It's also the correct instinct that three
independently-run models converging on it doesn't by itself confirm it
works here (see below) — the report itself frames it as a hypothesis,
not a result, which is the right level of confidence.

**Two gaps the convergence plausibly missed, both specific to this
project rather than generic VLM knowledge:**

1. **Pooling behavior on this corpus is not a solved detail.** This
   project's own measured findings (`document_classification.py`,
   `image_analysis.py`) show extreme, bimodal page aspect ratios
   (portrait manifests ~0.84–0.88 vs. landscape census ~1.57–1.64) and
   frequently tiny/degraded text (median 11px text height measured on
   the microfilm sub-corpus) — a very different distribution from the
   roughly-square, natural-image data most vision encoders' pooled/CLS
   representations are validated against. Several of the 8 buckets are
   *defined* by exactly this kind of structural signal (row/column
   density, aspect ratio, table presence), which a naive global-pool
   vector may or may not preserve. The report lists "whether global
   pooling loses document structure" as an open unknown, correctly —
   but doesn't connect it to this project's own already-measured
   aspect-ratio/scale extremity, which makes the risk concrete rather
   than generic and argues for testing spatial/patch-level pooling
   alongside global pooling from the start, not as a fallback.
2. **Parity gap with the current output contract** — covered in §12
   above (`ClassificationResult`'s five auxiliary fields plus `reason`)
   — not addressed anywhere in the report.

**One place the convergence is likely right, worth crediting rather
than just critiquing**: replacing today's self-reported LLM confidence
number (used only as a `min_confidence` threshold gate, per
`config/pipeline.yaml`) with a classifier-native margin/distance signal
(softmax margin, centroid distance, entropy) is a real, well-precedented
calibration improvement — self-reported LLM confidence is a known-weak
signal, whereas margin/distance-based abstention from a linear or kNN
classifier is standard and better-understood. This piece of the plan is
worth pursuing even independent of whether the rest of it pans out.

**A question the report doesn't ask, and should**: is Stage-1
bucket-routing accuracy actually a measured pain point in this project
today? Nothing gathered in this research, or in this project's own
recorded history (`docs/CODE_MAP.md`), documents a routing-accuracy
problem specifically — the real measured failures on record so far
cluster in Stage 2 subtype disambiguation (already non-VLM, unaffected
by this plan either way) and extraction-stage fabrication (a different
pipeline stage entirely). Convergence across three independently-run
models is evidence they each found the *reasoning* sound in the
abstract, not evidence this targets this project's actual current
bottleneck. Worth a cheap check — current bucket-CSV skew, the
`uncertain_review` rate — before treating this as the highest-priority
experiment rather than merely the most theoretically elegant one.

### 16. Verdict: is Stage 0/0.5 grounded, and what has to be true first?

**Grounded, and a real improvement in framing over the original
encoder-replacement research** — it correctly identifies Stage 1 as
genuinely closed-set (§12), correctly proposes inspecting the actual
interface before extracting anything from it (§13), and correctly
treats its own premise as a hypothesis rather than an established
result. That's a materially better-scoped starting point than Stage B
of the original three-model plan.

**"Highest-value next experiment" is plausible, not yet
established.** It's the cheapest option on the table and fails safely
— a negative result just falls through to the already-planned Stage A
— but "highest-value" is currently argued from cross-model agreement
and general ML reasoning, not from a documented problem in this
pipeline's own history (§15's last point).

**What needs to be true or built before it can run, in priority order:**

1. A real interface-inspection pass against the actual
   `google/gemma-4-E2B-it` class (Stage 0 as proposed) to confirm
   whether a clean pooled-feature accessor exists, or a forward hook on
   an internal submodule is needed. Not knowable from the loader code
   alone — genuinely unresolved until someone opens the model.
2. **A labeled, class-balanced validation set for the 8-bucket task —
   still does not exist today**, independent of raw row counts (see
   §14's update: the full-corpus run raised counts to 385/12/220/32/
   87/103/1430/16, but those are still unverified classifier output,
   not ground truth). What resolves this is Jon's in-progress manual QA
   pass over the review UI (flagging misclassification/bad deskew/
   needs-new-bucket) — that's the actual verdict data this experiment
   needs; re-check its coverage per bucket once that pass is further
   along, rather than assuming volume alone has closed the gap.
3. An explicit decision on what happens to `ClassificationResult`'s
   non-category fields (§12) if a probe replaces Gemma for routing —
   dropped, replicated via separate heads, or left to a different
   mechanism.
4. Some minimal grounding that Stage-1 routing accuracy is a real
   problem worth this investment (§15's last point) — cheapness is a
   reason to try an experiment, not a substitute for confirming it
   addresses an actual pain point in this pipeline.

---

## Third addendum — standalone vision tower as a downstream hint source
(2026-07-31, same day, follow-up)

Jon's latest framing: instead of replacing Gemma's vision encoder (the
excluded architecture-B route), run a separate, standalone vision
tower earlier in the pipeline and feed *its* embeddings to later
stages as semantic hints — Gemma keeps its own native vision path
untouched. Caveat as posed: "if the embeddings are able to be
understood correctly."

### 17. This is the document's existing recommendation, arrived at independently

This is architecture C from §6, and matches the "Vision Provider →
IR → consumed as hints, not as a replacement for any loader's internal
encoding" framing in §4 — worth saying plainly rather than re-deriving:
this idea converges with, rather than adds to, the recommendation
already on record. The useful new content here is the caveat itself —
what "understood correctly" actually requires, which the document
hasn't addressed concretely until now.

### 18. Two different ways to make an embedding "understood," with very different risk

- **Route 1 — translate the embedding into text or retrieval evidence,
  then inject that text into the prompt.** E.g. a nearest-neighbor
  lookup against known examples ("closest confirmed match: printed_document,
  similarity 0.91") or a small trained classifier's label+confidence,
  rendered as a plain-language hint. Gemma never touches the embedding
  itself — it only ever sees text, exactly like every other prompt
  input it already handles. **This does not reintroduce §3's
  projector-coupling problem at all**, because nothing is being fed
  into Gemma's internal vision-token space — it's ordinary text
  conditioning, the same mechanism `document_templates.py`'s
  `column_regions_approx` already uses to condition per-field crops.
- **Route 2 — feed the raw embedding tensor into a model's internals**
  (e.g. concatenated as extra soft tokens alongside Gemma's own visual
  tokens). This quietly re-imports §3's exact finding: an arbitrary
  encoder's embedding space is only "understood" by a decoder if
  something has been trained to translate it into that decoder's
  expected input distribution — i.e., a projector, trained per
  consuming model, the same unproven-for-Gemma technique §11's
  literature search covered. Relocating the injection point (alongside
  the native tower, rather than replacing it) doesn't remove that
  requirement, it just makes it easy to overlook.

**Recommendation: Route 1 only.** It gets the stated benefit (shared
semantic hint from a standalone tower) without touching any loader's
internals or requiring new projector training — the actual reason this
version of the idea is lower-risk than encoder replacement, and worth
stating as the reason, not just asserting the idea is safer in general.

### 19. Within Route 1, prefer retrieval/kNN hints over a trained classifier, at least first

A nearest-neighbor lookup against manually-verified anchor images
needs no trained classifier head at all — just the standalone
encoder's embeddings plus a labeled anchor set. That anchor set is
exactly what Jon's in-progress manual QA pass (flagging
misclassification/bad deskew/needs-new-bucket in the review UI) is
producing right now, per the second addendum's §14 update — direct
reuse, not throwaway work. It also produces a hint a human can audit
directly ("nearest confirmed match: X, similarity 0.91, from these N
verified examples") rather than an opaque probability from a trained
model's weights, which matters for the failure mode below.

### 20. A concrete, project-specific risk: hints can get "locked onto," not just wrong

This project has already measured this failure mode in a related
context — smaller/weaker models in this pipeline have been observed to
echo a strongly-worded worked example in a prompt instead of doing
independent extraction (`feedback_prompt_examples_get_locked_onto`
memory; also `gemma.yaml`'s own instruction-tuned, conservative
sampling settings suggest this model is tuned to follow instructions
literally). A hint phrased as a confident answer — *"Predicted
category: printed_document, confidence 0.95"* — risks the same
collapse: Gemma parrots the hint instead of weighing it against the
actual pixels, which would make the hint a silent single point of
failure rather than a genuine second signal. This risk is specific to
*how* the hint is worded, not to the retrieval/kNN mechanism itself —
phrasing it as contextual evidence rather than a pre-computed answer
("nearest confirmed similar pages: …, image also shows: …") is the
mitigation direction, not a fix that removes the need to test for it.

### 21. How to actually evaluate "understood correctly"

Because this is a hint, not a standalone decision, the right test is
paired, not aggregate: run the same images through Gemma with and
without the hint, and check two cases separately — **hint present and
correct** (does routing accuracy or confidence calibration improve),
and **hint present and wrong** (does Gemma still get the image right,
or does it get pulled toward the hint's wrong answer, i.e. is §20's
lock-on risk real in practice). A single "did accuracy go up" number
would hide the second case entirely, and the second case is the one
that would make the hint a net negative. Both trials need the same
labeled evaluation set §14's update already identified as the real
bottleneck — this idea doesn't remove that dependency, it just gives
a second, lower-risk use for the same anchor data once it exists.

---

## Fourth addendum — Architecture D: CPU-only analysis of frozen embeddings
(2026-07-31, same day, follow-up)

### 22. Where this sits relative to everything above

This is §18's Route 1, formalized and substantially expanded — the
embedding stops at the analysis stage, Gemma only ever sees text, so
none of §3's projector-coupling risk applies, exactly as claimed in the
"non-goals" list. What's new here is the breadth of CPU-side methods
(clustering, anomaly/density scoring, topology) and packaging the
output as a formal evidence contract rather than a single kNN hint.

Worth naming explicitly: this maps directly onto a pattern **already
established and working in this codebase** —
`core/image_analysis.py`'s own "Stage A: pure measurement, emits
numbers, decides nothing / binning is Stage B's job, done separately
so policy can be retuned without re-measuring" discipline. Recommend
reusing that exact split here rather than inventing a parallel scheme:
the CPU analyses below should emit raw scores as a permanent, versioned
record, with a separate (and separately revisable) binning/policy step
deciding what becomes prompt text.

### 23. What's extractable from frozen embeddings via deterministic CPU analysis (Q1)

Everything below is deterministic **given a fixed encoder checkpoint
and a fixed reference set** — both need to be pinned/versioned for any
of this to be reproducible (see §26). "Deterministic" here means the
same as `image_analysis.py`'s own fields: reproducible bit-for-bit
given the same inputs and same code, not "no neural network is
involved anywhere upstream."

- **Cosine similarity / kNN retrieval against a labeled reference
  set** — the strongest, cheapest, most interpretable method here.
  Gives "this image's nearest confirmed neighbors are X, Y, Z at
  similarity 0.9x" directly. Scales fine at this project's stated
  target (hundreds of thousands of images) with a simple flat index;
  approximate-NN libraries only become necessary well past that.
- **Duplicate / near-duplicate detection** — a special case of the
  above at a very high similarity threshold. Already flagged as
  high-value in §1/§6; reaffirmed here as one of the two or three best
  uses of this whole architecture.
- **Centroid distance per known bucket** (distance to the mean
  embedding of each of the 8 current `DocumentCategory` classes, or to
  each cluster from a clustering pass below) — a continuous,
  classifier-free confidence-like signal per candidate bucket, cheaper
  than training a probe (§14–16's Stage 0.5) and a natural companion to
  kNN rather than a replacement for it.
- **Clustering** (k-means for a fixed k, or HDBSCAN for
  density-based clusters that can naturally emit a "noise"/no-cluster
  label) — useful for discovering structure that doesn't map onto the
  current 8 buckets, i.e. a semi-automated version of exactly the
  "needs new bucket" flag Jon is raising by hand right now in the
  review UI. HDBSCAN in particular is worth preferring over k-means
  here specifically because its noise label is a built-in, free
  anomaly signal, rather than forcing every point into some cluster.
- **Anomaly / outlier scoring** (distance-to-k-th-nearest-neighbor,
  Local Outlier Factor, Isolation Forest, one-class SVM) — flags
  images unlike anything seen before, a proxy for "novel or
  out-of-distribution document." Distance-to-nearest-neighbor is the
  simplest and most auditable of these; LOF/Isolation Forest add
  modeling complexity without an obviously better payoff at this
  corpus's scale.
- **Density estimation** (kernel density estimate in embedding space,
  or the k-NN-distance proxy above) — a continuous version of the same
  signal as anomaly scoring; mainly useful for calibrating a
  percentile-based threshold (e.g. "flag the bottom 5% by local
  density") rather than as a value exposed on its own.
- **Topology analysis** (persistent homology, Mapper-style methods) —
  technically applicable, but the weakest fit here: heavy
  machinery, expensive at "hundreds of thousands of images" scale, and
  its output (topological features of the whole point cloud) doesn't
  reduce naturally to a per-image piece of evidence the way every
  method above does. Better suited to an offline, one-time corpus
  health check ("does this corpus have hidden substructure we're
  missing") than to this stage's per-image evidence-generation role.
- **Dimensionality reduction** (PCA, UMAP) — same verdict as topology:
  valuable for a human-facing diagnostic dashboard, not something that
  reduces to meaningful per-image prompt text on its own.

### 24. Which analyses are interpretable, verifiable, and prompt-appropriate (Q2)

The bar that matters: **can a human open the referenced evidence and
check it against the actual image**, not just trust a number.

- **Nearest-neighbor identity + similarity score** clears this bar
  cleanly — a person (or Gemma) can be told exactly which prior pages
  the current one resembles and by how much, and that claim is directly
  checkable. This is the strongest candidate for direct prompt
  inclusion.
- **Cluster membership** clears the bar **only after a human has
  inspected and named the cluster** — an arbitrary cluster ID
  (`cluster_17`) is not independently verifiable by a prompt reader
  without that mapping already established. Until that curation step
  happens, cluster membership is Q3's territory (internal diagnostic),
  not Q2's.
- **Anomaly/density scores and layout-similarity metrics** are only
  interpretable once **binned into calibrated ordinal categories**
  (low/moderate/high), exactly the way `image_analysis.py` already
  treats `table_confidence` and friends — a raw Isolation Forest score
  or KDE value means nothing to a reader without a calibration step
  behind it. The raw float should still be recorded (for tuning/audit,
  same as `image_analysis.py`'s numeric fields), but the prompt-facing
  value should be the calibrated category, not the number itself.
- A **`confidence` field, as in the example schema, needs its
  derivation stated**, not left implicit — "moderate" derived from a
  density percentile is a different claim than "moderate" derived from
  a nearest-neighbor margin, and collapsing them into one unlabeled
  field just recreates the same opacity problem already flagged with
  Gemma's own self-reported confidence in §15.

### 25. Which analyses should stay internal-only (Q3)

- Raw cluster IDs before human verification/naming (as above).
- PCA/UMAP coordinates, and any topology/TDA output — offline
  corpus-health diagnostics, not per-image evidence.
- Raw density/KDE values and raw LOF/Isolation-Forest scores, prior to
  calibration into an ordinal category.
- **Anything computed against a reference set that hasn't itself been
  through Jon's manual verification pass** — covered in depth in §26,
  since the example schema's own field name (`nearest_validated_pages`)
  makes a claim that needs to actually be true.

### 26. Evidence contract (Q4) — refined, with two risks specific to this project

Building on the schema in the prompt, with the additions this analysis
surfaces:

```
vision_evidence:
  schema_version: 1
  encoder:
    model_id: <encoder checkpoint id>
    checkpoint_hash: <sha256 or equivalent>
  reference_set:
    version: <id/hash of the anchor set used for kNN/clustering/centroids>
    source: "manually_verified_subset"   # see risk below — must be literally true

  nearest_validated_pages:              # Q2 — prompt-eligible
    - page_id: ...
      similarity: 0.94
  duplicate_flag: false                  # Q2 — prompt-eligible

  cluster:                               # Q3 until human-named, then Q2
    id: internal_cluster_17
    human_label: null                    # null = not yet curated, do NOT surface to prompt

  anomaly:
    raw_score: 0.06                      # Q3 — internal/audit only
    calibrated_level: "low"              # Q2 — prompt-eligible

  layout_similarity:
    raw_distance: 0.31                   # Q3
    calibrated_level: "high"             # Q2

  confidence:
    value: "moderate"
    derived_from: "nearest_neighbor_margin"   # explicit, not implicit
```

Two risks worth flagging specifically for this architecture, beyond
the general ones already in §7:

- **Reproducibility now depends on two pinned things, not one.**
  `image_analysis.py`'s deterministic fields only need their algorithm
  version pinned. This stage's outputs additionally depend on the
  reference set (anchors/cluster fit) used for kNN/clustering/centroid
  distance — adding, removing, or re-verifying anchors changes
  neighbor results even with the encoder held fixed. Both need
  versioning (as in the schema above), or "nearest similarity 0.94"
  becomes a number nobody can reproduce six months later.
- **The reference set's "validated" label needs to be literally true,
  not aspirational.** If `nearest_validated_pages` is built from the
  raw bucket CSVs (Gemma's own unverified predictions) rather than the
  subset Jon's manual QA pass has actually confirmed, this evidence
  stage silently reintroduces §14's circularity problem — the prompt
  would present Gemma's own past guesses back to Gemma, dressed up as
  independent evidence. The `reference_set.source` field above should
  gate this explicitly: don't populate `nearest_validated_pages` from
  anything broader than the confirmed subset, even though that means
  starting with far fewer anchors than the full bucket CSVs would
  offer.
- **Anomaly/density scoring needs to be scoped, not computed globally
  across the whole corpus** — this project has already hit and fixed
  exactly this class of bug once: `image_analysis.py`'s tone fields
  were meaningless until scoped per-region (the LAC microfilm
  black-surround finding, §1). A global anomaly score across this
  corpus's full, confirmed heterogeneity (microfilm reels vs. the
  clean census batch vs. portraits vs. maps) would likely just
  rediscover "this is microfilm vs. not," not genuinely novel or
  problematic pages, unless anomaly/density scoring is computed within
  a known document-family or cluster rather than against the whole
  corpus at once. Recommend scoping this the same way `table_confidence`
  was already forced to be region-scoped, rather than assuming a
  single global score is meaningful.
- **§20's lock-on risk still applies, regardless of which CPU method
  produced the evidence.** Whether the hint comes from a trained
  classifier (Stage 0.5) or from kNN/clustering/anomaly scoring
  (Architecture D), the same phrasing discipline and paired
  hint-correct/hint-wrong evaluation from §20/§21 is still required
  before trusting that Gemma weighs the evidence rather than parroting
  it.

**Overall verdict**: Architecture D is sound, and — because it never
lets an embedding leave the analysis stage — strictly lower-risk than
either the original encoder-replacement direction (§§2–3, 7, 11) or
Route 2 of the standalone-tower idea (§18). It should reuse this
codebase's existing deterministic/policy split
(`image_analysis.py`'s Stage A/B pattern) rather than inventing a new
one, and its two open dependencies are the same ones already on record:
a genuinely human-verified reference set (§14, §19) and a defined,
paired evaluation protocol for whatever gets surfaced into Gemma's
prompt (§21).

---

## Fifth addendum — reframing as Vision Analysis / Pipeline Context
(multi-consumer, not Gemma-specific) (2026-07-31, same day, follow-up)

### 27. This reframing returns to the document's original thesis — worth naming, not re-deriving

The very first research pass (§0's framing at the top of this document)
already posed the question this way: *"The Vision Representation should
become the canonical interface between image acquisition and
downstream reasoning... it should not be viewed simply as a cache."*
The intervening sections (§§17–26) narrowed to "hints for Gemma"
because that's where the concrete design questions (lock-on risk,
kNN vs. classifier, evidence-contract shape) were easiest to reason
about precisely — but that narrowing was a working simplification, not
a scope decision. This refinement is a correct course-correction back
to the original framing, not a new architecture. The genuinely new
content it requires is: (a) an organizing principle for *how* different
kinds of consumers should use the same artifact safely, and (b) a
consumer-by-consumer check of which proposed outputs are actually
derivable from a frozen embedding at all. Both below.

### 28. The organizing distinction this refinement needs: machine-consumed fields vs. text-hint fields

Broadening to multiple consumers surfaces a split that the
Gemma-only framing didn't force into the open: this pipeline-context
artifact has **two fundamentally different consumption channels**,
not one, and they carry different risk profiles:

- **Machine-consumed structured fields** — read directly by ordinary
  code (`if duplicate_candidate: skip_reprocessing()`, `if
  anomaly_level == "high": route_to_manual_review()`). These carry
  none of §20's lock-on risk, because there's no model being
  "persuaded" — a program either branches on the field or it doesn't.
  Preprocessing, retry policy, batching, and validation-queue ordering
  (§29) are naturally this kind of consumer.
- **Text-hint fields**, rendered into a prompt for a generative model
  (Gemma, or any future extraction-stage model) — these are the only
  consumers subject to §20's lock-on risk and §21's paired
  hint-correct/hint-wrong evaluation requirement.

This distinction matters because it changes *which* validation burden
applies to *which* consumer. A field like `duplicate_candidate: true`
needs to be **correct** (ordinary precision/recall) if a program acts
on it directly, but doesn't need the lock-on-risk evaluation §20/§21
describe — that evaluation is specific to feeding a claim into a
generative model's prompt. Recommend the evidence contract (§26)
explicitly tag each field with which channel it's meant for, so a
future consumer doesn't accidentally pipe a machine-only field into a
prompt (or vice versa) without triggering the right validation for that
channel.

### 29. Consumer-by-consumer check

- **Preprocessing / OpenCV stages** — the strongest legitimate use here
  is *not* re-deriving measurements `image_analysis.py` already makes
  directly from pixels (scale, geometry, contrast — see §30 for why
  re-deriving these from a pooled embedding is a step down in
  fidelity, not up). The genuine value-add is novelty/OOD signal: "this
  page looks unlike anything the CV calibration was tuned against."
  That's concretely useful — the LAC microfilm finding (§0/§1: 43/218
  pages misflagged as inverted-polarity before the black-film-surround
  problem was diagnosed) is exactly the kind of silent calibration
  failure an embedding-space novelty check, computed *before* trusting
  the CV pipeline's output, could plausibly have flagged early instead
  of being found the hard way. Machine-consumed channel.
- **Routing (Stage 1 Gemma classification)** — already covered in depth
  (§§17–26); the only consumer requiring the lock-on-risk evaluation.
  Text-hint channel.
- **OCR / extraction model selection** — a genuinely new idea in this
  round, distinct from Stage-1 routing: using retrieval not to pick a
  *bucket* but to pick a *model/config* based on which has historically
  performed best on visually similar pages. Plausible in principle, but
  currently ungrounded — it needs a reference dataset linking image
  clusters to measured extraction accuracy, which doesn't exist yet
  (the closest thing on record, `ground_truth_log.jsonl`, is small —
  ~3 pages — and isn't indexed by anything embedding-related). Worth
  flagging as a real future consumer, not a currently-actionable one.
- **Retry policy** — plausible and cheap: route a high-anomaly page
  straight to manual review instead of burning another expensive
  generative retry. Machine-consumed, threshold-driven, no
  interpretation risk.
- **Batching** — plausible (group similar-complexity pages for
  consistent generation settings) but a minor efficiency angle, not a
  correctness one — worth remembering the very first message in this
  research thread explicitly framed runtime reduction as "a bonus
  rather than the objective," so this shouldn't be prioritized over the
  consumers above.
- **Validation** — arguably the single strongest non-Gemma consumer
  identified across this whole research thread. `config/pipeline.yaml`'s
  existing `audit_sampling` is a flat `rate: 0.05` **random** sample.
  Replacing that with an anomaly/low-confidence-prioritized sample
  would directly serve this project's own stated success metric
  (`feedback_reduce_manual_touches_is_the_metric` — fewer manual
  touches per unit of accuracy) better than uniform random sampling
  does, using exactly the anomaly/density scoring already discussed in
  §23. Machine-consumed, no new risk category, concrete and immediately
  actionable once the underlying scoring exists.
- **Future pipeline stages** — nothing concrete to add beyond "the
  contract should be extensible," which §26's `schema_version` field
  already accounts for.

### 30. Checking the specific example outputs against what a pooled embedding can actually support

Several of the example fields in this round's proposal don't follow
from a **pooled, single-vector** embedding the way similarity/
clustering/anomaly scoring do — worth flagging before they're assumed
achievable:

- **`recommended_resolution` / `recommended_crop`** — resolution and
  crop recommendations are inherently *spatial* (crop needs a bounding
  box; resolution needs to know where the fine detail actually is), but
  a pooled embedding is a single vector for the whole image — it has
  already discarded the very spatial/scale information a crop or
  resolution recommendation would need. Getting either of these
  requires **patch-level embeddings** (a grid of per-region vectors, not
  one pooled vector) analyzed spatially — a materially heavier-weight
  approach than anything else in this section, closer to "train or
  adapt a lightweight spatial detector on patch features" than "run
  cosine similarity on a vector." Also, concretely: `gemma.yaml`'s
  `image_token_budget` (70/140/280/560/1120) is a Gemma-specific
  resolution knob — recommending a value for it from a *different*
  encoder's embedding would need its own empirical validation (does
  this signal actually predict which Gemma budget performs best),
  not just a generic CPU computation.
- **`estimated_text_scale` / `patch_density`-as-geometry** — this
  project already measures text scale directly and more reliably from
  pixels (`image_analysis.py`'s `text_height_px`, median 11px on the
  microfilm sub-corpus). Re-deriving the same thing from a pooled
  embedding is redundant at best and less trustworthy at worst — the
  encoder's value-add is answering what classical CV structurally
  *can't* (semantic similarity, novelty), not re-deriving what it
  already does directly from pixels. Same principle §5 already
  established for preprocessing decisions generally, now applying to
  this expanded consumer list too.
- **`layout_complexity`** — plausible, and a better fit than the two
  above, since "complexity" is closer to a semantic/gestalt judgment a
  pooled embedding could capture — but only after the same
  calibration-into-ordinal-category step §24 already requires for
  anomaly/layout-similarity scores. Not a raw distance value on its
  own.
- **`preprocessing_recommendation` (e.g. `preserve_original_scale`)** —
  this is a **policy decision**, not a measurement, and mixing it into
  the same emission as raw/calibrated measurements re-blurs the exact
  split `image_analysis.py` deliberately keeps separate ("binning
  cutoffs are policy and live in Stage B, so policy can be retuned
  without re-measuring"). Recommend this stage emit measurements
  (Stage-A-shaped: similarity, cluster membership, calibrated anomaly
  level) and leave recommendation-shaped decisions to a separate,
  explicitly-tunable policy step downstream — not because the
  recommendation is a bad idea, but because baking a decision into the
  measurement stage means retuning the policy requires re-touching the
  same code that does the expensive vision pass.
- **`confidence: high`** — same requirement as §24: state what it's
  `derived_from`. Generalizing the consumer list doesn't relax this;
  if anything it matters more once multiple stages might key off the
  same unlabeled field with different implicit assumptions about what
  it means.

### 31. Cost of broadening scope: versioning becomes a shared discipline, not a per-consumer one

§26 already flagged that this stage's reproducibility depends on two
pinned things (encoder checkpoint, reference set). Multiplying the
number of consumers multiplies the blast radius of getting that wrong:
if six pipeline stages key off this artifact and the encoder or
reference set changes, all six need coordinated re-validation, not
just whichever one prompted the change. This argues for a single
shared version stamp (`schema_version` plus the encoder/reference-set
hashes already in §26's contract) that every consumer checks against,
rather than each stage independently tracking whether the context it's
reading is still current — the multi-consumer framing makes this a
harder requirement to skip than it was in the Gemma-only framing,
where only one consumer needed to care.

### 32. Verdict

The reframing is sound and is the right long-term shape for this
work — it matches where this research started (§0) and correctly
separates "is this a good measurement" from "is this a good hint for
one specific model." The two things this round of planning adds that
weren't explicit before: **(1)** the machine-consumed vs. text-hint
distinction, which determines which risk (correctness vs. lock-on)
applies to which consumer, and **(2)** a genuine per-example check
showing that not everything in a wishlist of pipeline-context fields is
actually obtainable from a *pooled* embedding — several of the most
pipeline-relevant-sounding ones (resolution, crop) need patch-level
features and are a heavier lift than the similarity/clustering/anomaly
core this document has otherwise validated. Recommend scoping the near
term to the fields that are both cheaply derivable from a pooled vector
and cleanly assignable to one consumption channel — retrieval/
duplicate detection, calibrated anomaly level, and (once curated)
cluster membership — and treating spatial/patch-level analysis (crop,
resolution) as a distinct, separately-scoped follow-on question rather
than folding it into the same evidence contract from the start.

---

## Sixth addendum — `timm` capability inspection (empirical, first hands-on step)
(2026-07-31, same day — repo confirmed as the experimental branch, a
separate `...Main` copy preserved by Jon as the frozen baseline)

Installed `timm==1.0.28` (`torch==2.13.0+cu130`, CUDA already available)
and inspected its catalog/API directly — no pretrained weights
downloaded, `nvidia-smi` checked clear first per `CLAUDE.md`'s gate. This
upgrades §9's and §30's claims from "per public documentation/general
knowledge" to "confirmed against the actual installed package":

- **1,293 architectures registered.** Every candidate family discussed
  so far is present and real, not just plausible by name: DINOv2 (8
  variants), SigLIP/SigLIP2 (36, including `naflexvit_*` — a
  native-flexible-resolution variant worth a closer look given §30's
  finding that fixed-resolution pooling discards scale information),
  ConvNeXt (31), EVA (18), BEiT (11), Swin (39).
- **Pretrained weights are real and resolvable**, not bare
  architecture shells: `vit_base_patch14_dinov2.lvd142m`,
  `vit_large_patch14_dinov2.lvd142m`, `vit_base_patch16_siglip_224.v2_webli`,
  `convnext_base.fb_in22k`, and `eva02_base_patch14_224.mim_in22k` all
  resolve to real HF-hub-hosted checkpoints via `timm.get_pretrained_cfg()`.
- **§30's pooled-vs-patch distinction is now empirically confirmed, not
  just theoretical.** `create_model(name, pretrained=False, num_classes=0)`
  returns a pooled vector directly from `forward()` (384-dim for
  `vit_small_patch14_dinov2` at 518×518 input); `forward_features()` on
  the same model returns the pre-pool patch grid, `(1, 1370, 384)`. Both
  are one line of code apart on the identical model instance — getting
  patch-level access isn't an obscure or unsupported path.
- **A third access mode exists that §30 didn't anticipate**:
  `features_only=True` returns multi-scale spatial feature maps
  directly (confirmed on `convnext_tiny`: stages at 56×56, 28×28, 14×14,
  7×7 spatial resolution). This is a clean, documented, one-line API for
  exactly the spatial signal §30 said `recommended_crop`-style outputs
  would need — softens (doesn't remove) that section's "heavier lift,
  separate track" verdict on crop/resolution recommendations. Still a
  materially different analysis than pooled-vector cosine
  similarity/clustering, but the tooling to get the input it would need
  already exists off the shelf, which wasn't confirmed before this
  check.
- **§9's claim about `timm`'s scope is now directly verified, not
  inferred from documentation**: `timm.loss` does not exist as a
  submodule, and the entire top-level API surface is `create_model`,
  `list_models`/`list_pretrained`, `get_pretrained_cfg`, `data`
  (transforms/datasets), `layers`, `utils` — nothing for contrastive
  loss, text alignment, or projector training. Confirms `timm` is
  encoder-catalog-and-loader only, exactly as §9 concluded from its
  documentation, now checked against the real package.

**Natural next step, not yet taken**: picking 2–3 concrete candidates
(e.g. a DINOv2 size, a SigLIP2 size, possibly the `naflexvit` variant
given its native-resolution handling) and actually downloading their
pretrained weights to run forward passes against a handful of real
corpus images, comparing pooled-feature separability across this
project's own 8 buckets directly. That's a heavier action than this
inspection pass (real downloads, likely several hundred MB per
checkpoint) and a good next thing to scope explicitly before running,
rather than pulled in automatically here.

### Per-family capability matrix (what's available without modifying the model)

Follow-up check, same session: for one representative model per family
(all `pretrained=False`, zero downloads, CPU-only shape checks), tested
each of the six outputs a CPU-analysis stage could plausibly draw on.
The first pass had a real bug (`forward_intermediates()`'s default
return is `(final_output, [intermediates])`, not the list directly —
grabbing element `[0]` silently returned the single final tensor, which
is why every family first reported "1 stage") and an unfair test for
variable resolution (the probe offset wasn't aligned to each model's
own patch size, which made DINOv2/SigLIP/EVA look like they didn't
support it when the real constraint is narrower: they do, but only with
patch-size-aligned input). Both corrected below — noted rather than
silently fixed, same discipline as the bucket-count correction earlier
in this document.

| Family | Pooled embedding | Patch embeddings | Multi-depth intermediates | Attention maps | True multi-scale pyramid | Variable resolution |
|---|---|---|---|---|---|---|
| DINOv2 | YES (384-d) | YES (1370×384) | YES, arbitrary depths | NO by default (fused attention — see below) | NO, same-resolution depth stack only | YES, opt-in `dynamic_img_size=True`, patch(14)-aligned input required |
| SigLIP | YES (768-d) | YES (196×768) | YES | NO by default, same caveat | NO, same-resolution stack | YES, opt-in flag, patch(16)-aligned |
| SigLIP-NaFlex | YES (768-d) | YES (scales with input) | YES | NO by default, same caveat | NO, same-resolution stack | YES **natively, no flag** — its core design point |
| ConvNeXt | YES (768-d) | YES (768×7×7 spatial) | YES | N/A — pure conv, no attention | YES — genuine multi-scale (56→28→14→7) | YES natively |
| EVA-02 | YES (768-d) | YES (257×768) | YES | NO by default, same caveat | NO, same-resolution stack | YES, opt-in flag, patch(14)-aligned |
| BEiT | YES (768-d) | YES (197×768) | YES | NO by default, same caveat | NO, same-resolution stack | NO — `dynamic_img_size` kwarg not implemented in timm's BEiT class at all |
| Swin | YES (1024-d) | YES (7×7×1024 spatial) | YES | PARTIAL — mixed fused/eager flags across blocks | YES — genuine multi-scale (56→28→14→7) | NO — window-partitioning fails under timm's current implementation |

**Three findings worth carrying into the design, not just the table:**

- **Attention maps are not a zero-touch measurement anywhere in this
  set**, unlike the other five categories. Every transformer-family
  model here defaults to PyTorch's fused/flash attention path, which
  never materializes attention weights at all — recovering them needs
  `fused_attn=False` set at model creation (a real execution-path
  change, distinct from altering weights or architecture, but still not
  "available for free"), plus a forward hook on top of that. Worth
  flagging since attention maps weren't in Architecture D's original
  method list (§23) at all — if they're wanted, they're the one item
  here that isn't already a pure zero-touch measurement the way
  cosine-similarity/kNN/clustering/anomaly-scoring are.
- **"Feature pyramid" is not one capability, it's two, and they map to
  different families.** ConvNeXt and Swin give a genuine multi-scale
  pyramid — spatial resolution actually shrinks with depth, the classic
  detection/segmentation sense of the term, and the actual fit for
  §30's spatial/crop-recommendation discussion. DINOv2/SigLIP/EVA/BEiT's
  "stages" are the *same* spatial resolution at every depth — useful for
  comparing semantic depth, not a substitute for genuine multi-scale
  spatial analysis. Don't treat "feature_pyramid: available" as
  equivalent across rows of this table.
- **Variable-resolution support is real but uneven across the set**,
  which matters directly for §26/§30's earlier concern about whether an
  encoder discards scale information before a CPU analysis stage ever
  sees it. SigLIP-NaFlex is the only architecture here built for this
  natively. DINOv2/SigLIP/EVA support it as an opt-in flag with a real
  constraint (patch-size-aligned input only — not truly arbitrary).
  BEiT and Swin don't expose it through timm's public API for this
  purpose at all. If preserving scale/resolution signal matters for a
  chosen use case, this table is a real constraint on which family to
  pick, not an afterthought.

### Candidate selection (decided, not yet downloaded)

Jon's picks against the matrix above, one family per role rather than
one family for everything — sound, and each maps to a specific finding
from this addendum rather than a generic "this one's popular" choice:

- **Semantic** → DINOv2 (or SigLIP as a comparison candidate). DINOv2
  is the stronger theoretical fit specifically because it's
  self-supervised with no text entanglement (§2's original point) —
  the near-term Architecture D use cases (kNN retrieval, clustering,
  anomaly scoring, §23) are all pure visual-similarity tasks, which is
  exactly DINOv2's training objective. SigLIP remains a reasonable
  comparison point, or the pick if text-prompted zero-shot queries ever
  become a real requirement.
- **Variable-resolution** → SigLIP-NaFlex. Matches the matrix exactly —
  the only family here that's natively variable-resolution rather than
  needing a workaround. Confirmed real pretrained weights exist:
  `naflexvit_base_patch16_siglip.v2_webli` (not checked in the earlier
  pass, which only confirmed the architecture stub).
- **Spatial** → ConvNeXt over Swin. Both give genuine multi-scale
  pyramids per the matrix, but ConvNeXt is simpler (no windowing
  constraints) and supports native variable resolution as a bonus,
  where Swin's timm implementation supports neither dynamic resolution
  nor clean attention-map access.
- **Attention maps** → explicitly deferred, consistent with this
  addendum's finding that it's the one item on the list that isn't a
  zero-touch measurement in any family tested.

Concrete checkpoints and real sizes (params confirmed via
`pretrained=False` load, no download yet):

| Role | Checkpoint | Params | Approx. size (fp32) |
|---|---|---|---|
| Semantic | `vit_small_patch14_dinov2.lvd142m` | 22.1M | ~88MB |
| Semantic (comparison) | `vit_base_patch16_siglip_224.v2_webli` | 92.9M | ~372MB |
| Variable-res | `naflexvit_base_patch16_siglip.v2_webli` | 92.9M | ~372MB |
| Spatial | `convnext_tiny.fb_in22k` | 27.8M | ~111MB |

Minimal first set (DINOv2-small + NaFlex + ConvNeXt-tiny): ~571MB.
**Not yet downloaded** — this is a real external download, flagged
explicitly for confirmation before pulling, same discipline as the
"natural next step, not yet taken" note earlier in this addendum.

### Experiment protocol, defined before any download (2026-07-31, same day)

Jon's refinement: define question → metric → candidate → experiment
*before* downloading anything, rather than download-then-look. Sound,
and it directly targets this addendum's open question — whether the
candidates picked above are actually stable under the transforms this
pipeline already performs, since an embedding that moves sharply under
a 2° deskew is a poor retrieval key regardless of how well it separates
document types otherwise.

**Grounding the transform list against real repo code**, rather than
inventing generic perturbations — reusing `core/image_preprocessing.py`
and `core/row_segmentation.py`'s actual functions wherever they exist:

| Transform (as posed) | Real function to use | Note |
|---|---|---|
| Deskew ±2° | `apply_deskew_angle(image, angle)` | test ±2° *and* a smaller ±0.5° — real measured corrections on this corpus run far smaller (median 0.12–1.08% of page width, per the earlier dewarp-ground-truth finding), so ±2° alone would test a larger perturbation than this pipeline typically applies |
| Brightness | *(no named function exists yet)* | `image_preprocessing.py` has `enhance_contrast`/`autocontrast`/`sharpen`/`denoise`/`invert`/`grayscale`/`upscale` but no brightness function — this would need a plain `ImageEnhance.Brightness` call, flagged here as testing a hypothetical rather than an existing pipeline step |
| Contrast normalization | `enhance_contrast()` **and** `autocontrast()` separately | these are documented as mechanistically different (scales around midpoint vs. stretches histogram) — test both, don't collapse into one "contrast" transform |
| Denoise | `denoise()` / `median()` | same underlying `MedianFilter`, two call sites |
| Crop | `crop_region_from_source()` | bbox-driven, needs an existing sidecar/mask — only usable on the subset of images that already have one (the 9–16 manually-masked reference pages noted earlier); a generic "crop N% off each edge" proxy is needed for arbitrary corpus images without one |
| Border/frame removal | *(no isolated function — part of `image_analysis.py`'s ROI detection)* | this is the single most concretely-motivated transform on the list: the LAC microfilm black-film-surround finding (§1/§26) is a documented real failure of classical CV under exactly this condition. Worth prioritizing an image known to have this surround, if one exists in the corpus, over a synthetic border crop |
| Inversion | `invert()` | direct reuse |
| Different DPI | *(no DPI concept in this pipeline)* | proxy via `upscale()`/downscale — this pipeline doesn't re-DPI, it varies resolution |
| Different aspect ratio | *(not a transform — a cross-corpus property)* | do **not** synthetically stretch an image's aspect ratio, since nothing in this pipeline ever does that to a real page. Instead, sample real images that already span the documented range (0.42–1.60) and test whether stability holds up *across* that natural variation, not whether one image survives a distortion nothing would ever apply to it |

**The three experiments, refined with concrete metrics and pass/fail
criteria** — Jon's question→metric→candidate framing kept, with the
metric made decision-relevant (see rationale under Experiment 1):

**Experiment 1 — Question: do duplicates remain nearest neighbors
after preprocessing? Candidate: DINOv2.**
- Two metrics, not one: **cosine drift** (`1 - cosine_sim(original,
  transformed)`) is a cheap sanity check, but it's the wrong metric to
  gate a decision on — a similarity that drops from 0.99 to 0.93 could
  still be a perfectly fine retrieval key if every unrelated image in
  the reference set sits at 0.4. The metric that actually answers "is
  this stable enough for kNN retrieval" is **rank preservation**: given
  a reference set of N other distinct real images, does the transformed
  embedding still retrieve the untransformed original as its nearest
  neighbor (recall@1), ahead of every unrelated image?
- Protocol: ~15–20 real corpus images spanning several buckets (no
  labels needed — this doesn't depend on the in-progress manual QA
  pass at all, unlike the Stage 0.5 classifier work). For each image,
  apply each transform from the table above individually (not
  bundled — this project's own convention in
  `image_preprocessing.py`'s docstring is exactly "steps are
  independent and toggleable... a blind clean-everything transform
  could help one thing and hurt another"), embed, and check recall@1
  against the full untransformed reference set.
- Pass/fail: if recall@1 stays ≥ ~95% across the test set for a given
  transform, DINOv2 is stable enough under that transform without
  further normalization. If it drops for one specific transform (e.g.
  inversion), that's not necessarily a reason to discard DINOv2 — it's
  evidence that a normalization step (e.g. always de-invert before
  embedding) needs to happen before this stage, the same way
  `image_analysis.py`'s tone fields needed crop-before-tone ordering.

**Experiment 2 — Question: does variable resolution preserve
neighborhoods better? Candidate: SigLIP-NaFlex.**
- Metric: **cluster stability** (Jon's framing) — concretely, cluster
  the same image set's embeddings twice (once at native/original
  resolution, once after a resolution/aspect change), and compute
  **Adjusted Rand Index (ARI)** between the two clusterings. ARI is the
  standard metric for exactly this question (does clustering agreement
  survive a perturbation) and is bounded/interpretable (0 = no better
  than chance, 1 = identical), unlike raw cosine drift which doesn't
  have a natural pass/fail scale.
- A caveat worth stating up front: if cluster ground truth for this
  test comes from the current 8 `DocumentCategory` buckets, that's
  Gemma's own unverified labels, not confirmed clusters — fine for
  *this* experiment specifically, since ARI only needs the *same*
  clustering compared against itself under a transform, not a labeled
  ground truth. The circularity concern from §14 applies to *training a
  classifier against unverified labels*, not to this internal
  before/after comparison — worth noting explicitly so this distinction
  doesn't get flattened into "always need the QA pass first."
- Pass/fail: ARI ≥ ~0.8 across the resolution/aspect perturbations
  tested would support NaFlex's native-resolution handling actually
  paying off relative to a fixed-resolution encoder forced through the
  same transforms (run the identical test against fixed-resolution
  SigLIP as the comparison point — this experiment is only informative
  paired against that baseline, not in isolation).

**Experiment 3 — Question: can spatial features identify regions
classical CV misses? Candidate: ConvNeXt.**
- This is the softest of the three to make quantitative, and worth
  being honest about that rather than forcing a false-precision metric.
  Metric, first pass: **qualitative localization check** against
  already-known CV failure cases — this project has two documented,
  specific ones to reuse directly rather than inventing new test cases:
  (1) the LAC microfilm pages where `table_boundary` returns a
  spurious, near-global (~92%-of-frame) low-confidence box on 174/218
  pages (§1/earlier CODE_MAP finding), and (2) any image where
  `estimate_deskew_angle` silently reads exactly 0.00 on visibly skewed
  microfilm (the documented false-zero finding). For each, render
  ConvNeXt's spatial feature map (via `features_only=True`, per this
  addendum's matrix) as a heatmap overlay and check by eye whether it
  shows localized structure roughly where the true table/page region
  is, even where classical CV's own boundary detector failed.
- A more rigorous follow-on (IoU of a thresholded feature-map region
  against a manually-drawn ground-truth box) is possible but shouldn't
  be the first pass — it requires new manual annotation work, and the
  qualitative check is enough to decide whether pursuing that
  investment is worthwhile at all.
- Pass/fail: no numeric threshold for the first pass by design — the
  practical criterion is "does a human looking at the overlay agree the
  feature map is picking out real structure on the known CV-failure
  cases." If yes, that justifies investing in the IoU-based follow-on;
  if the feature map looks as confused as the CV detector did, ConvNeXt
  isn't obviously solving the problem it was picked for and the spatial
  track should be reconsidered before further investment.

**Fourth metric added: corpus locality / neighborhood preservation**
(Jon's addition, same day) — richer than Experiment 1's recall@1, and
catches a failure mode recall@1 alone misses: a transform could leave
an image's *self-match* intact while scrambling everything around it,
which would silently corrupt the retrieval evidence text §19/§23
already proposed (`nearest_validated_pages` could return a completely
different, wrong neighbor set post-transform, even while the image
still "finds itself" at rank 1).

- **Metric**: for each query image, take its top-*k* neighbors
  (*k*≈5, self excluded) against the fixed reference gallery *before*
  any transform. Transform only the query (the gallery stays embedded
  from the original, untransformed images — this mirrors the real
  deployment shape: a new incoming scan gets compared against an
  already-embedded gallery, not a gallery that moves too). Recompute
  the query's top-*k* neighbors after the transform, and compute
  **Jaccard overlap** between the two neighbor sets
  (`|before ∩ after| / |before ∪ after|`). Bounded and interpretable
  (0 = completely different neighbors, 1 = identical set), same
  reasoning as choosing ARI over raw distance for Experiment 2 — avoid
  a metric with no natural pass/fail scale.
- **This changes the image-sampling requirement for Experiment 1.**
  ~15–20 diverse-but-mostly-unrelated images (the earlier plan) can't
  test this meaningfully — Jon's own example needs multiple real
  examples of the *same* real-world document series in the set (several
  actual passenger-manifest pages, several actual census pages, etc.),
  not just one representative per bucket. Revise the sample to include
  a handful of same-type clusters (e.g. 3–5 real pages each from
  several buckets) specifically so "does A's neighborhood stay
  {B, C} and not {map, photo, letter}" is a well-posed question against
  real data.
- Pass/fail: high self-recall@1 with low neighborhood Jaccard would be
  a genuinely useful negative result this addendum's other metrics
  would have missed entirely — worth treating as a real pass/fail
  criterion in its own right, not a nice-to-have alongside recall@1.

**Fifth addition: operational metadata, captured regardless of
outcome** (Jon's addition) — embedding latency, memory, dimensionality,
patch count. Not correctness metrics, but exactly the kind of thing
this project already treats as worth recording permanently elsewhere
(`GenerationConfig.content_hash()` traces every generation call's exact
settings for the same reason). Worth noting explicitly: **dimensionality
and patch count are architecture properties, and latency/memory are
weight-independent** (a forward pass takes identical time/memory
whether the weights are random or trained — same tensor shapes, same
compute graph) — meaning all four can be measured right now, on the
already-installed `pretrained=False` models, with zero downloads. Ran
this immediately rather than deferring it:

Measured directly (random-init, batch=1, mean of 10 runs after 3
warmup, RTX 5060 Ti / same machine as the rest of this research):

| Candidate | Embedding dim | Patch count (at input res used) | CPU latency | GPU latency | GPU peak memory |
|---|---|---|---|---|---|
| `vit_small_patch14_dinov2` (518×518) | 384 | 1,370 | 172.1 ms | 14.0 ms | 153 MB |
| `vit_base_patch16_siglip_224` (224×224) | 768 | 196 | 73.2 ms | 6.8 ms | 413 MB |
| `naflexvit_base_patch16_siglip` (384×384) | 768 | 576 | 184.6 ms | 12.9 ms | 430 MB |
| `convnext_tiny` (224×224) | 768 | 49 (7×7 spatial) | 29.7 ms | 7.0 ms | 162 MB |

Read with the obvious caveat that these are architecture/compute-graph
properties, not accuracy signals — cheap and fast doesn't mean
"good embedding," it's purely a cost record for later comparison, which
is exactly the point Jon raised it for. Two things worth flagging now
rather than in six months: DINOv2 is being tested here at a much larger
input (518×518, its native config) than the other three (224–384px),
which accounts for most of its higher latency and patch count relative
to ConvNeXt/SigLIP — not an inherent DINOv2-vs-others cost difference,
an artifact of comparing at each model's own default resolution. A fair
head-to-head would re-run all four at one common input size. ConvNeXt
is the cheapest by a wide margin on both CPU and GPU, consistent with
being the smallest/simplest architecture of the four (pure conv, no
attention).

**Sixth addition: cross-representation consistency** (Jon's addition,
same day) — only became a well-posed question once this session's timm
work confirmed pooled, patch-level, and (for ConvNeXt/Swin) true
multi-scale feature-map outputs are all real, accessible outputs of
the same forward pass. Question: for the same image, do the three
representations agree on what's similar to what, or do they diverge?
Metric: pairwise cosine similarity structure across a small image set,
computed once per representation (pool the patch tokens for a
patch-vs-pooled comparison; use a spatially-pooled feature map for the
feature-map comparison), then compare *those similarity structures* to
each other (e.g. correlation between the pooled-similarity matrix and
the patch-mean-similarity matrix, across the same image pairs). If they
agree, that's reassurance the pooled vector isn't discarding anything
decision-relevant; if they diverge, that's a real signal about which
representation to trust for which downstream use. Note this needs more
than one checkpoint to run in full (true feature-map requires
ConvNeXt/Swin, per the earlier matrix) — the DINOv2-only round below
can answer the pooled-vs-patch half of it, not the full three-way
comparison.

**Research/design phase declared complete, per Jon's direction.**
Everything above this line is design work; from here, further
additions to this document should come from measured results, not
further architecture research or literature review. The operational
metadata table above is being kept permanently for exactly the reason
Jon gave — so "why did we reject X" has a recorded answer six months
from now instead of needing to be re-derived.

---

## Experiment 001: DINOv2-small qualification (execution begins)

Per Jon's sequencing — one checkpoint at a time, full battery, document,
stop, then decide whether a real gap justifies the next download. This
section is measured results, not design discussion, starting here.

**Setup**: `vit_small_patch14_dinov2.lvd142m` (real pretrained weights,
downloaded this round only), 24 real corpus images across 6 buckets (4
each: `dense_tabular_rows`, `printed_document`, `map_land_record`,
`portrait_photo`, `genealogy_chart`, `handwritten_ledger` — same-type
clusters per bucket, needed for the neighborhood-locality metric),
sampled directly from the real `data/buckets/*.csv` files, all
confirmed to exist on disk before running. Transform battery reuses
this pipeline's real functions (`core/image_preprocessing.py`,
`core/row_segmentation.py`) exactly as catalogued earlier in this
addendum. Implementation: `benchmark/vision_encoder_qualification.py`;
raw record appended to
`data/outputs/vision_encoder_qualification_log.jsonl`.

**Cross-representation consistency (pooled vs. patch-mean)**:
correlation = **0.862** across all 276 image-pair similarities. Strong
agreement, not identical — the two representations mostly tell the same
story about what's similar to what, but ~14% of the variance in one
doesn't track the other. Not alarming on its own, but a data point worth
keeping once ConvNeXt's true feature-map representation is available
for the full three-way comparison this metric was designed for.

**Transform battery**:

| transform | recall@1 | mean cosine drift | mean neighborhood Jaccard@5 |
|---|---|---|---|
| deskew +2.0° | 0.88 | 0.0745 | 0.834 |
| deskew −2.0° | 0.92 | 0.0715 | 0.841 |
| deskew +0.5° | **1.00** | 0.0291 | 0.855 |
| contrast (enhance 1.5×) | 1.00 | 0.0372 | 0.848 |
| contrast (autocontrast) | 1.00 | 0.0000 | 1.000 |
| denoise (median-3) | 1.00 | 0.0313 | 0.879 |
| **invert** | **0.92** | **0.1631** | **0.786** |
| downscale 0.5× | 1.00 | 0.0333 | 0.869 |
| upscale 2× | 1.00 | 0.0023 | 0.972 |
| border crop 5% | 1.00 | 0.0321 | 0.879 |

**Interpretation**:

- **DINOv2 is stable under the transforms this pipeline applies most
  routinely** — mild deskew (0.5°, close to this corpus's real measured
  correction magnitudes, 0.12–1.08% of page width per the earlier
  dewarp-ground-truth finding), both contrast mechanisms, denoise,
  resolution changes in either direction, and a mild border crop all
  show perfect recall@1 and low drift.
- **Two real, actionable exceptions**, not just noise:
  - **Inversion is the single largest disruption measured** — drift
    5× higher than any other transform, and the lowest neighborhood
    Jaccard. This isn't a synthetic edge case: `invert()` is a real
    pipeline function applied to real faded census pages. **If a kNN
    reference gallery ends up with a mix of inverted and non-inverted
    versions of visually similar documents, retrieval could genuinely
    misfire across that boundary** — worth treating as a required
    normalization step (e.g. always embed a canonical-polarity version)
    rather than an encoder-selection concern, the same conclusion §26
    already anticipated in the abstract, now backed by a real number.
  - **Deskew at ±2° (not ±0.5°) shows real degradation** — 2–3 of 24
    images lost self-retrieval entirely. Confirms Jon's original
    framing concern was directionally right, but with an important
    qualifier the raw concern didn't have: it's stable at the rotation
    magnitudes this pipeline actually produces, and only degrades at a
    larger synthetic perturbation past that range. Worth re-testing at
    intermediate angles (1.0°, 1.5°) before concluding exactly where the
    breakdown point sits, if this matters for a production threshold.
  - `autocontrast`'s exactly-zero drift is worth a note of caution
    rather than uncritical acceptance — plausibly these particular scans
    already span close to full dynamic range, making the operation
    near-identity on this sample, rather than proof the encoder is
    perfectly invariant to it. Not re-tested against a deliberately
    low-contrast sample this round.
  - The border-crop test used a mild, symmetric 5% crop — it does not
    test the more extreme, documented real case (the LAC microfilm
    black-film-surround, §1/§26), which would need an asymmetric crop
    against an image that actually has that surround. Left untested this
    round.

**Decision gate, per Jon's explicit rule** ("is there a question DINOv2
couldn't answer — if yes, download the next candidate"): **yes, two
specific ones**, not a general "let's see what else is out there":

1. This round only tested isotropic resolution changes (uniform up/down
   scale) — it did **not** test genuine aspect-ratio variation, which
   DINOv2 can't handle natively without the `dynamic_img_size` flag and
   even then only with patch-aligned input (§ capability matrix). The
   variable-resolution/neighborhood-stability question (Experiment 2)
   is only answerable with SigLIP-NaFlex.
2. The cross-representation consistency check only covered pooled vs.
   patch-mean — DINOv2's own "feature pyramid" is a same-resolution
   depth stack, not true multi-scale (per the capability matrix), so
   the spatial-localization question (Experiment 3) and the full
   three-way consistency check both require ConvNeXt specifically, not
   substitutable by anything DINOv2 can produce.

Both are real gaps, not a rationalization to keep downloading — flagged
here per Jon's sequencing rather than acted on automatically. Recommend
Round 2 = SigLIP-NaFlex or ConvNeXt next (whichever question matters
more to resolve first); holding for confirmation before pulling either.

### Protocol frozen: Vision Qualification Battery v1.0

Per Jon's direction, the battery itself (`benchmark/vision_encoder_qualification.py`)
is now frozen at `BATTERY_VERSION = "1.0"` — every future round runs the
identical `TRANSFORMS` set, `TOP_K`, and metric definitions, so results
stay directly comparable across candidates rather than each round
quietly testing something slightly different. Changing the protocol
itself is a version bump and an explicit note here, not an in-place
edit.

### Design decision arising directly from Round 1: canonical polarity

This is the headline result of Experiment 001 — not a benchmark score,
a design decision, exactly the kind of finding Architecture D was
built to surface: **inversion showed the highest drift, the lowest
neighborhood preservation, and real recall degradation, and `invert()`
is a real function this pipeline actually applies to real images.**
That means a kNN gallery mixing inverted and non-inverted versions of
visually similar documents is a genuine, live risk, not a hypothetical
one.

**Decision**: embedding galleries need a canonical polarity, enforced
one of two ways —

1. **Preferred**: `Input → normalize polarity → encoder → gallery`.
2. **Fallback**, where normalization can't be trusted: keep polarity as
   explicit metadata on every embedding and only compare like-with-like
   at retrieval time.

**This is immediately buildable, not just a good idea** — this project
already has the deterministic piece option 1 needs.
`core/image_analysis.py`'s region-scoped polarity detection
(`ink_is_dark_on_light`, gated on `largest_ink_blob_frac`) was already
measured at 0/218 false positives on the LAC microfilm corpus once
scoped to the page region (vs. 43/218 false positives when measured at
the frame level, per the earlier finding in this document) — i.e. this
pipeline already has a validated, working "is this page inverted"
signal. The natural implementation is: run that check before
embedding, apply `invert()` when it says the page is inverted relative
to the corpus's dominant convention, and only fall back to polarity
metadata (option 2) for cases the detector itself flags as low-
confidence — not a new detection problem, a new *use* of an
already-validated one.

**One nuance worth recording alongside the headline finding, not
instead of it**: it's tempting to read the inversion sensitivity as
proof DINOv2 has specifically learned "dark text on light background"
as a document-specific concept. That's plausible but not the only
explanation the data supports — DINOv2 is trained on general natural
imagery, where photographic negatives are rare; the same sensitivity
could just as easily reflect a generic "this isn't a naturally-lit
scene" prior rather than anything document/text-specific. Distinguishing
the two would need a follow-up (e.g. testing inversion sensitivity on
non-document natural images) — not run this round, and not necessary to
act on the polarity-normalization decision either way, since the
decision holds regardless of which explanation is correct. Worth
keeping the distinction on record rather than letting the more specific
(and more narratively satisfying) explanation get cited as confirmed.

**Operational stability vs. stress testing — a real distinction this
round surfaced**: the battery now separates "does this hold up under
what the pipeline actually does" (0.5° deskew, contrast, denoise,
resolution changes, mild crop — all stable) from "where does it
actually break" (±2° deskew, inversion). Both are useful, but they
answer different questions, and Round 1's headline result came from
correctly not stopping at the operational-stability numbers alone.

**Milestone note**: Experiment 001 has already produced engineering
value — a standalone vision encoder surfaced a real pipeline behavior
(polarity sensitivity) that neither the existing classical-CV stage nor
Gemma's own routing was positioned to reveal — before a single encoder
has been integrated into the production pipeline. That's the outcome
this whole research branch was speculating might be possible, now
demonstrated rather than argued for.

**Correctly not locked in yet** — Jon held off finalizing the canonical-
polarity decision specifically to test whether it generalizes beyond
DINOv2, per the "natural-image prior vs. document-specific concept"
nuance flagged above. Round 2 (below) tested exactly that.

---

## Experiment 001, Round 2: ConvNeXt-tiny (same frozen v1.0 battery)

Identical protocol, same 24 images, same `BATTERY_VERSION = "1.0"` —
`convnext_tiny.fb_in22k` downloaded, `embed_patch_mean()` generalized to
handle ConvNeXt's spatial `(B, C, H, W)` `forward_features()` output
(no CLS token, unlike DINOv2's token sequence — mean over spatial dims
instead of dropping a prefix token; same semantic metric, correct
handling per architecture, not a protocol change).

Cross-representation consistency (pooled vs. patch-mean): **0.894**
(DINOv2: 0.862) — similar middle-ground relationship on a completely
different architecture, which makes "strongly related, not identical"
look like a more general property of pooled-vs-patch representations
rather than a DINOv2-specific artifact.

| transform | recall@1 | mean cosine drift | mean neighborhood Jaccard@5 |
|---|---|---|---|
| deskew +2.0° | 1.00 | 0.0865 | 0.790 |
| deskew −2.0° | 1.00 | 0.0846 | 0.813 |
| deskew +0.5° | 1.00 | 0.0228 | 0.935 |
| contrast (enhance 1.5×) | 1.00 | 0.0626 | 0.786 |
| contrast (autocontrast) | 1.00 | 0.0000 | 1.000 |
| denoise (median-3) | 1.00 | 0.0288 | 0.907 |
| invert | 1.00 | **0.0814** | 0.813 |
| downscale 0.5× | 1.00 | 0.0341 | 0.885 |
| upscale 2× | 1.00 | 0.0021 | 0.948 |
| border crop 5% | 1.00 | 0.0753 | 0.804 |

**The headline result: inversion is not an outlier for ConvNeXt.**
DINOv2's invert drift (0.163) was more than double its own next-highest
transform. ConvNeXt's invert drift (0.081) is comparable to — and
lower than — its own deskew ±2° result (0.085/0.086), and recall@1
stays perfect (1.00) under inversion, unlike DINOv2's drop to 0.92.
**This supports the natural-image-prior explanation over the
document-specific-concept explanation** flagged as untested in Round 1
— polarity sensitivity looks substantially tied to DINOv2's (or
self-supervised ViT training's) particular inductive bias, not a
universal property every vision embedding space shares.

**A second finding, orthogonal to polarity**: ConvNeXt's raw cosine
drift is *higher* than DINOv2's on nearly every transform, including
the benign ones (e.g. denoise: 0.029 vs. DINOv2's 0.031 — comparable;
but border crop: 0.075 vs. 0.032 — notably higher) — yet ConvNeXt's
recall@1 stays perfect everywhere, including under the transforms where
DINOv2 degraded. **Raw cosine drift is not directly comparable across
encoders** — each architecture's embedding space has its own baseline
scale of "how far apart do genuinely different images sit," so the
same drift number means different things for different encoders. This
is a concrete confirmation that designing the battery around
rank-based metrics (recall@1, neighborhood Jaccard) rather than raw
drift was the right call, not just a design nicety — a comparison
based on drift alone would have wrongly suggested ConvNeXt is less
stable than DINOv2, when the rank-based metrics show the opposite.

**Autocontrast's exact-zero drift replicated identically on a
completely different architecture** (0.0000 for both DINOv2 and
ConvNeXt) — raises confidence this is a real property of the test
images (already near full dynamic range) rather than an encoder-specific
quirk, resolving Round 1's open caution on this point.

**Revised polarity-normalization recommendation**: the decision from
Round 1 should be **encoder-conditional, not a blanket architecture
policy** — real and load-bearing if a self-supervised ViT-style encoder
(DINOv2, and plausibly SigLIP/EVA/BEiT given similar training paradigms,
untested directly) ends up in the pipeline; apparently much less
critical for ConvNeXt specifically. If the eventual production choice
is ConvNeXt or another CNN-family encoder, canonical-polarity
enforcement may genuinely be unnecessary overhead rather than a
required safeguard — the right call is to defer finalizing this policy
until the actual encoder for each role is chosen, and re-check polarity
sensitivity specifically for whichever one is selected, rather than
applying Round 1's finding universally.

**Note**: this round tested the transform battery only — it did not
yet run Experiment 3's spatial/feature-map-localization check
(heatmap overlay against the known CV-failure cases, §26/§30) even
though ConvNeXt is now downloaded. That remains a separate, not-yet-run
piece of this round's original motivation for picking ConvNeXt.

### Reframing, after Round 2: qualifying capabilities, not encoders

Round 1 alone risked a premature architectural rule ("embeddings are
sensitive to inversion, therefore canonicalize polarity, universally").
Round 2 falsified that generalization rather than confirming it — the
real result is that this is a per-encoder property, not a universal
one. That reframes the policy shape itself: not "normalize polarity,"
but **"does this specific encoder require polarity normalization? →
if yes, enable the policy; if no, don't pay the complexity cost."**
A conditional policy gated on measured per-encoder behavior, not a
blanket architectural rule inferred from one data point.

That, in turn, reframes the underlying research question. This document
started by asking "should the pipeline use one vision encoder?" Two
rounds in, the more accurate framing is **qualifying capabilities, not
picking a single encoder**:

| Capability | Candidate | Status |
|---|---|---|
| Semantic retrieval | DINOv2 | qualified (Round 1) |
| Robust retrieval under polarity changes | ConvNeXt | qualified (Round 2) |
| Variable-resolution / aspect-ratio robustness | SigLIP-NaFlex | pending (Round 3) |
| Spatial localization | ConvNeXt | pending (Experiment 3, deferred until after Round 3) |

This document is accordingly closer to a **qualification of
computational vision sensors for archive processing** than a single
"which encoder should we adopt" research memo — it documents observable
properties of different sensing approaches under this pipeline's real
transforms, not an argument for one "best" encoder. Keeping the file at
its current path/name for link continuity, but this is the operative
framing going forward.

**Methodological note worth stating plainly**: neither Round 1 nor
Round 2 simply confirmed the prior round's plan — each one changed it.
That's the loop this research has actually been running (hypothesis →
experiment → unexpected result → architecture update → new experiment),
not a validation exercise. Sequential, one-checkpoint-at-a-time
qualification is what caught this before a local, single-encoder
observation became a global design rule.

---

## Experiment 001, Round 3: SigLIP-NaFlex (same frozen v1.0 battery)

**A real harness bug caught before this round ran, not after**:
`embed_patch_mean()` originally hardcoded dropping one leading token
(`feats[:, 1:, :]`), written against DINOv2's single CLS token. Checked
directly before running NaFlex rather than assumed: DINOv2
(`num_prefix_tokens=1`) matches that assumption, so Round 1 is
confirmed unaffected; ConvNeXt used the spatial-mean path and was never
exposed to this code at all. But SigLIP-family models
(`vit_base_patch16_siglip_224`, `naflexvit_base_patch16_siglip`) have
`num_prefix_tokens=0` — SigLIP pools via attention pooling, no CLS
token exists at all. The hardcoded slice would have silently dropped a
real patch token instead of a prefix token for NaFlex. Fixed to read
`model.num_prefix_tokens` generically before this round ran. This is a
harness correctness fix, not a change to `BATTERY_VERSION`'s protocol
(same transforms, same metrics) — noted here rather than silently
patched.

Cross-representation consistency (pooled vs. patch-mean): **0.761** —
notably lower than DINOv2 (0.862) or ConvNeXt (0.894). Worth an
interpretive caveat rather than taking the number at face value: SigLIP
pools via a learned attention-pooling head, not a simple average, so
naive mean-pooling its patch tokens is a bigger structural mismatch
from its *own* native pooling mechanism than for the other two
candidates — ConvNeXt's `num_classes=0` pooling is itself close to a
literal spatial mean, which plausibly explains why its correlation
came out highest almost by construction. A lower correlation for NaFlex
may partly reflect "attention pooling differs more from mean pooling"
rather than "NaFlex's patches carry more genuinely distinct information"
— the same kind of confound Round 2 surfaced for raw cosine drift.
Worth recording, not treating as clean signal either way.

| transform | recall@1 | mean cosine drift | mean neighborhood Jaccard@5 |
|---|---|---|---|
| deskew +2.0° | 0.96 | 0.0554 | 0.833 |
| deskew −2.0° | 1.00 | 0.0549 | 0.861 |
| deskew +0.5° | 1.00 | 0.0351 | 0.833 |
| contrast (enhance 1.5×) | 1.00 | 0.0282 | 0.917 |
| contrast (autocontrast) | 1.00 | 0.0000 | 1.000 |
| denoise (median-3) | 1.00 | 0.0307 | 0.851 |
| invert | 1.00 | 0.0574 | 0.796 |
| downscale 0.5× | 1.00 | 0.0269 | 0.845 |
| upscale 2× | 1.00 | 0.0035 | 0.958 |
| border crop 5% | 1.00 | 0.0531 | 0.778 |

**Inversion sensitivity, refined further**: NaFlex's invert drift
(0.057) is, like ConvNeXt's, unremarkable relative to its own other
transforms (comparable to deskew ±2° at 0.055) — not the DINOv2-style
outlier. This is a more informative data point than Round 2 alone: **NaFlex is also a ViT-family architecture**, like DINOv2, yet patterns
with ConvNeXt (a CNN) on this property rather than with DINOv2 (also a
ViT). That rules out "ViT vs. CNN" as the explanation and points more
specifically at **DINOv2's particular training objective** — pure
self-supervised visual distillation, with no text or class-label
supervision at all — as the more likely driver, rather than anything
inherent to transformer-based vision architectures generally. Both
SigLIP-family models (image-text contrastive) and ConvNeXt (supervised
classification) involve *some* form of external supervision; DINOv2
doesn't. Still only three data points — a real hypothesis refinement,
not a confirmed conclusion.

**Autocontrast's exact-zero drift held on a third, architecturally
distinct candidate.** DINOv2, ConvNeXt, and now NaFlex all show
0.0000 drift under `autocontrast()`. Three independent architectures
agreeing is strong evidence this is a genuine property of the test
images (already near full dynamic range at `cutoff=1.0`) rather than
an encoder-specific artifact — treat this as settled unless a
deliberately low-contrast test image contradicts it.

**Deskew robustness**: NaFlex handled ±2° close to perfectly (0.96/1.00
recall@1) — much closer to ConvNeXt's perfect result than to DINOv2's
0.88/0.92. Another point that doesn't split cleanly by architecture
family (NaFlex is a ViT, like DINOv2, but performs like ConvNeXt here
too), reinforcing the same "objective/training-driven, not
architecture-driven" pattern as the inversion result above.

**What this round did *not* test, despite being NaFlex**: the v1.0
battery's resolution transforms are isotropic upscale/downscale only —
no genuine aspect-ratio change. NaFlex's actual distinguishing
capability (native handling of varying resolution/aspect ratio,
confirmed architecturally in the capability matrix) remains untested by
this run. These clean results describe how NaFlex behaves under the
*same* battery every other candidate ran — useful for cross-round
comparability, but they say nothing yet about the specific reason
NaFlex was picked. Answering that still needs the paired comparison
against fixed-resolution SigLIP-base under real aspect-ratio variation,
flagged before this round ran and still not executed — a candidate
follow-up, not assumed to be a foregone "yes" just because this round's
numbers looked clean.

**Capability-qualification table, updated:**

| Capability | Candidate | Status |
|---|---|---|
| Semantic retrieval | DINOv2 | qualified (Round 1) |
| Robust retrieval under polarity changes | ConvNeXt, SigLIP-NaFlex | qualified (Rounds 2–3); DINOv2 does *not* qualify here |
| Variable-resolution / aspect-ratio robustness | SigLIP-NaFlex | Round 4 below — mechanism confirmed, practical benefit not yet shown |
| Spatial localization | ConvNeXt | pending at time of writing — see "Experiment 3 — CLOSED" below for the final outcome |

---

## Experiment 001, Round 4: NaFlex native resolution vs. fixed SigLIP

**A second real gap caught before this round could run on a wrong
assumption**: checked directly (not assumed) whether `timm`'s default
`create_transform()` actually exercises NaFlex's variable-resolution
capability. It does not — the resolved pipeline for
`naflexvit_base_patch16_siglip.v2_webli` is `Resize(384) →
CenterCrop(384,384)`, identical in effect to a fixed-resolution model.
**This means Rounds 1–3 never actually tested NaFlex's distinguishing
capability at all** — every prior round used NaFlex in its
squared-input fallback mode, confirming exactly the gap flagged at the
end of Round 3.

NaFlex's real native-resolution path requires a different call
convention entirely: `ResizeKeepRatioToSequence` (aspect-preserving
resize to a patch/sequence budget) → `patchify_image()` → `patches,
patch_coord, patch_valid` → `model.forward(patches, patch_coord=...,
patch_valid=...)`. Verified end-to-end against a real corpus image
before building the full experiment (4376×3328 original → 432×336
after aspect-preserving resize, aspect ratio correctly preserved to
within patch-size rounding; sane pooled output). Implementation:
`benchmark/vision_encoder_aspect_ratio_experiment.py`, deliberately
separate from the frozen v1.0 battery since this needs a genuinely
different pipeline, not a substitutable transform. Downloaded
`vit_base_patch16_siglip_224.v2_webli` (fixed-resolution baseline) for
this round.

**Two comparisons, reported honestly rather than as one clean result:**

**1. Same-model ablation (NaFlex native vs. squared, isolates the
mechanism from any other checkpoint difference)**: correlation between
each image's aspect-ratio extremity (`|ln(aspect_ratio)|`, symmetric
for portrait vs. landscape) and native-vs-squared embedding similarity
= **−0.789**. Strong, and exactly the expected shape — near-square
images barely differ between native and squared processing; extreme-
aspect images diverge sharply. **This confirms the native-resolution
pathway does something real, and it scales correctly with how extreme
the aspect ratio actually is.**

**2. Cross-model comparison (the practically relevant question — does
this translate into better retrieval)**, using each image's real
`DocumentCategory` bucket as a same/different anchor (Gemma's own
labels, not manually verified — same caveat as Experiment 2's design,
acceptable for an internal same-vs-different signal, not for training
anything against):

| | NaFlex-native | SigLIP-squared |
|---|---|---|
| mean same-bucket precision@5 | 0.408 | 0.400 |
| correlation(aspect-ratio extremity, precision) | −0.297 | −0.259 |

Both comfortably above this sample's random baseline (~0.13, given 24
images/4 per bucket) — both carry real signal — but **the two methods
are nearly identical to each other**, and both degrade similarly as
aspect ratio gets more extreme. **No clear retrieval-quality advantage
for NaFlex-native over fixed-resolution SigLIP was found at this
sample size.**

**Honest overall read, not rounded up in either direction**: the
mechanism is confirmed real (comparison 1) — NaFlex's native-resolution
path genuinely behaves differently, and specifically more so exactly
where aspect ratio is most extreme, which is the correct signature if
it's doing what it's supposed to. But that confirmed mechanistic
difference has **not yet been shown to produce a practical retrieval
benefit** (comparison 2) at this sample size, using bucket-label
agreement as the proxy metric. These are genuinely different questions
— divergence in embedding space isn't automatically better retrieval,
the same kind of distinction Round 2 surfaced between raw cosine drift
and rank-based metrics. Worth a larger, better-verified sample before
concluding either way on the practical question; recommend not treating
this round as a decisive win for NaFlex, and not as a decisive loss
either — it's the "mechanism confirmed, benefit unproven" outcome, an
honest result on its own terms.

**Capability-qualification update**: "Variable-resolution / aspect-
ratio robustness" moves from "untested" to "mechanism confirmed,
practical benefit not yet demonstrated" — a meaningfully different,
more precise status than either "qualified" or "disqualified."

### Scaled-up replication (n=87, 2026-08-01)

Same design, same two checkpoints (no new downloads), sample increased
from 24 to 87 real images (up to 15 per bucket; `handwritten_ledger`
capped at all 12 of its real files — the tightest constraint in the
corpus). `sample_real_images()` in the frozen battery module gained an
optional `images_per_cluster` parameter for this, defaulting to the
existing constant so **Rounds 1–8's frozen protocol is untouched** —
only this separate, already-non-frozen aspect-ratio experiment uses the
larger count.

- **Same-model ablation (mechanism): −0.803** (n=87) vs. −0.789 (n=24)
  — **replicates almost exactly at much higher statistical power.**
  The mechanism is now well-confirmed, not a small-sample artifact:
  NaFlex's native-resolution path genuinely diverges from its squared
  fallback, specifically scaling with how extreme the aspect ratio is.
- **Cross-model retrieval precision: NaFlex-native 0.798 vs.
  SigLIP-squared 0.809** (both roughly doubled from n=24's 0.408/0.400,
  as expected with larger same-bucket candidate pools) — **still
  essentially tied, and the tiny directional edge flipped** (NaFlex was
  marginally ahead at n=24; SigLIP is marginally ahead at n=87). With
  nearly 4× the data, the null result held rather than resolving in
  NaFlex's favor — **a more confident "no demonstrated retrieval
  benefit" than Round 4 alone could support**, not a reversal of it.
- **One modest, honest nuance**: NaFlex's precision degrades less with
  aspect-ratio extremity than SigLIP's (correlation −0.108 vs. −0.170)
  — directionally in NaFlex's favor, consistent with what its design
  would predict, but small enough in magnitude that it shouldn't be
  leaned on without a proper significance check. Recorded as a modest
  point in NaFlex's favor, not a demonstrated advantage.

**Capability-qualification update, revised**: "Variable-resolution /
aspect-ratio robustness" moves from "mechanism confirmed, practical
benefit not yet demonstrated" (small-sample, Round 4) to **"mechanism
confirmed at scale; practical retrieval benefit still not demonstrated
at scale either"** — a genuine strengthening of the null result, not
merely an unresolved question. If a production choice between NaFlex
and fixed-resolution SigLIP for the retrieval role is needed now,
this data does not support paying for NaFlex's added complexity on
retrieval-quality grounds alone — its value proposition (per the
capability-qualification framing) would need to rest on some other
axis (e.g. the mechanism itself mattering for a use case not tested
here), not on the bucket-retrieval precision measured in this
experiment.

---

## Systematic family expansion (Rounds 5–8) — cumulative comparison

Per Jon's direction: the goal shifts from opportunistic single-encoder
evaluation to systematically qualifying the remaining candidate
families under the *exact same, frozen* Vision Qualification Battery
v1.0, building a comparative evidence base rather than searching for a
"winner." **Battery unchanged — same `TRANSFORMS`, same `TOP_K`, same
metric definitions since Round 1.**

**Infrastructure built, not just more manual rounds**:
`benchmark/vision_encoder_qualification.py` now automatically captures
operational metadata (latency, memory, embedding dim, patch count,
prefix tokens, pooling method) as part of every round's JSONL record,
and `benchmark/vision_encoder_comparison_report.py` regenerates a full
comparison report from the raw log after every round — no
hand-transcribed tables. Auto-generated report:
`docs/VISION_ENCODER_COMPARISON_REPORT.md`. The interpretive analysis
below is written by hand, deliberately — pattern-finding across 7 data
points is a judgment call, not something a heuristic should produce.

**A correction made before running any of Rounds 5–8**: the comparison
report's first draft included a `TRANSFORM_HIGHLIGHTS` list, showing
only the 4 transforms that had already differentiated the first 3
candidates (inversion, large deskew, border crop, autocontrast). Jon
caught this directly: *"the benchmark exists to discover future
surprises, not confirm current hypotheses."* Correct catch — that
curation would have systematically under-displayed the other 6
transforms in every future round's report, exactly where a genuinely
new family might diverge in a way nothing so far had. Fixed before
Round 5 ran: the report's per-transform tables now cover all 10
transforms, every candidate, no filtering.

**Two real implementation bugs caught before accepting results**,
consistent with the discipline established in Rounds 1–4 — architecture-
specific issues are found and fixed *before* trusting a number, not
after:
- `embed_patch_mean()`'s 4D-tensor branch assumed NCHW layout (channel
  at dim 1, correct for ConvNeXt) — Swin's `forward_features()` is
  NHWC (channel at dim −1). Caught by testing directly before Round 8
  ran; fixed via `_spatial_layout()`, which detects the actual channel
  axis against the model's own `num_features` rather than assuming one
  architecture's convention generalizes.
- (Carried over from Round 3: the `num_prefix_tokens` fix for
  SigLIP-family models applies identically to EVA-02 and BEiT, both
  correctly read `num_prefix_tokens=1` since both use a real CLS
  token, confirmed via the same dynamic lookup rather than re-assumed.)

### Round 5: SigLIP-base (fixed-resolution)

Results closely track NaFlex's Round 3 numbers (invert drift 0.059 vs.
NaFlex's 0.057, deskew ±2° 0.96/0.96 vs. NaFlex's 0.96/1.00) — expected,
since NaFlex was only ever tested in its squared fallback mode through
Round 3 (per Round 4's finding), so the two were doing very similar
things architecturally in every prior round. Autocontrast: exact zero
drift again (4th replication).

### Round 6: EVA-02-base — a real falsification, not a confirmation

**Invert drift = 0.170, recall@1 = 0.92** — comparable to DINOv2's
outlier result (0.163), and clearly anomalous relative to EVA-02's own
other transforms (next-highest is deskew +2° at 0.057). **This directly
falsifies the Round 3 hypothesis** ("inversion sensitivity tracks
DINOv2's *lack* of external supervision, vs. SigLIP/ConvNeXt's *some*
form of supervision") — EVA-02 is trained with real external
supervision (masked-image-modeling with a frozen CLIP teacher's
features as the reconstruction target), yet shows the same heightened
sensitivity as DINOv2.

- **Previous hypothesis** (Round 3): self-supervised-only training
  (no text/class-label signal) predicts inversion sensitivity.
- **New evidence**: EVA-02 has real external supervision (CLIP-feature
  distillation) and still shows DINOv2-level sensitivity.
- **Revised interpretation** (tentative, see Round 7 below before
  treating as settled): the shared property between DINOv2 and EVA-02
  might not be "no supervision" but something about *what kind* of
  target each is trained to match.

### Round 7: BEiT-base — falsifies the very next hypothesis, same session

BEiT is also masked-image-modeling, the natural next candidate to test
"is it MIM specifically." **Invert drift = 0.064, recall@1 = 1.00** —
not an outlier at all (comparable to its own deskew ±2° at 0.076).
**This falsifies "masked-image-modeling objective" as the explanation**
just as quickly as the previous hypothesis fell.

- **Previous hypothesis** (after Round 6): masked-image-modeling
  objectives (patch-level reconstruction) predict inversion
  sensitivity.
- **New evidence**: BEiT is pure MIM (reconstructing discrete
  visual-token IDs from a fixed tokenizer) and shows no elevated
  sensitivity at all.
- **Revised interpretation**: "masked modeling" isn't the shared
  property either. What differs between EVA-02 (sensitive) and BEiT
  (not) within the "MIM" umbrella: EVA-02's reconstruction target is a
  frozen CLIP model's *continuous* feature vectors; BEiT's target is a
  fixed tokenizer's *discrete* codebook IDs. DINOv2's training (self-
  distillation, continuous teacher output, no discrete target
  anywhere) fits the same "continuous feature-matching" side as EVA-02.

### Round 8: Swin-base — consistent with the developing pattern, not yet confirming it

Swin is supervised classification, the same broad category as ConvNeXt
(already not sensitive). **Invert drift = 0.073, recall@1 = 1.00** —
not an outlier (comparable to or below its own deskew ±2° at
0.084/0.097). Consistent with ConvNeXt, both supervised-classification
architectures land on the "not sensitive" side.

### Current working hypothesis on inversion sensitivity — explicitly tentative

**Pattern across all 7 qualified candidates**: inversion sensitivity
(recall@1 degradation + drift far exceeding the candidate's own other
transforms) appears specifically in **DINOv2** and **EVA-02** — the two
candidates trained to match a *continuous* feature-space target
produced by another network (DINOv2: self-distillation from an EMA
teacher; EVA-02: distillation from a frozen CLIP encoder). It does
**not** appear in ConvNeXt or Swin (discrete class labels), SigLIP or
SigLIP-NaFlex (discrete-ish text-pair contrastive matching), or BEiT
(discrete tokenizer-ID target).

**This is a working hypothesis with exactly 2 positive examples,
explicitly not a confirmed encoder-family characteristic** — per Jon's
own stated discipline, this needs more than 2 qualified representatives
before being elevated beyond "worth testing further." The honest
sequence that got here matters more than the current answer: two prior
hypotheses were proposed and both were falsified within two rounds each
— a real demonstration of the qualification loop working as intended,
not evidence the current (third) hypothesis is correct just because
it hasn't been falsified yet. The natural next falsification test:
another continuous-feature-distillation-trained encoder (e.g. an
iBOT-only checkpoint, or a different CLIP-distilled model) — if it
also shows heightened inversion sensitivity, that's a third
confirming data point; if not, this hypothesis falls too.

### Other patterns scanned across all 10 transforms, not just inversion

Per the same discipline that caught the `TRANSFORM_HIGHLIGHTS` bias —
scanning every transform, not re-confirming the ones already known to
differentiate:

- **Autocontrast's exact-zero drift now replicated across all 7
  candidates** (7/7 — DINOv2, ConvNeXt, NaFlex, SigLIP-base, EVA-02,
  BEiT all exactly 0.0000; Swin 0.0001). This is now well-supported
  evidence — treat as settled that this is a property of these
  specific test images (already near full dynamic range) rather than
  any encoder's behavior, though still specific to *this* sample, not
  a general claim about `autocontrast()`.
- **Border crop (5%) shows a recall@1 dip (0.96) only for BEiT and
  Swin** — both otherwise-perfect candidates, both showing exactly one
  failure on this specific transform. Tentative, only 2 examples, no
  confident explanation offered — worth flagging as a real observation
  (not noise-fishing, since it's the same transform for both) but
  explicitly not elevated to a claim about window-attention or
  absolute-position-embedding architectures without more data.
- **Deskew ±2° shows no clean pattern** across training objective or
  architecture family — DINOv2 dips most (0.88/0.92), NaFlex/SigLIP-
  base/Swin dip slightly (0.96), ConvNeXt/EVA-02/BEiT stay perfect.
  Explicitly recording the absence of a pattern here, not just the
  presence of ones elsewhere.
- **Cross-representation consistency's full ranking (Swin 1.000 >
  EVA-02 0.969 > BEiT 0.938 > ConvNeXt 0.894 > DINOv2 0.862 > SigLIP-
  base 0.779 > NaFlex 0.761) is well explained by the confound already
  flagged in Round 3, not a new finding about information content.**
  Swin's own native pooling *is* a spatial mean (hence a perfect 1.000
  — patch-mean and pooled are computing nearly the same operation);
  the SigLIP family's attention-pooling head is structurally the most
  different from a naive mean (hence its scores are lowest). The
  ranking tracks "how similar is this architecture's own pooling
  mechanism to a plain average" more cleanly than it tracks anything
  about how much extra information patches carry — worth stating this
  explicitly rather than reading the ranking as a clean signal on its
  own terms.

See `docs/VISION_ENCODER_COMPARISON_REPORT.md` for the full
auto-generated operational and per-transform tables underlying this
analysis.

### Round 9: MAE ViT-Base — the third falsification test, and this time it holds

Per the recommendation flagged after Rounds 6–7: a genuinely new axis
to test the inversion-sensitivity hypothesis was a **pure pixel-
reconstruction** model — no teacher network (unlike DINOv2's self-
distillation or EVA-02's CLIP-feature distillation), no discrete
tokens (unlike BEiT), no text or class labels (unlike SigLIP/NaFlex/
ConvNeXt/Swin). `vit_base_patch16_224.mae` (the classic He et al.
MAE checkpoint, raw self-supervised encoder, no classification
fine-tuning) is exactly this: reconstructs masked patches' raw pixel
values directly, nothing more abstracted than that.

**Result: invert recall@1 = 0.21** — far more extreme than DINOv2's or
EVA-02's 0.92, and the neighborhood Jaccard collapses to 0.195 (vs.
0.79–0.94 for every other transform on this same model). **This is not
just an outlier — it's a near-total collapse.** Just as strikingly,
MAE shows *near-zero* drift on every other transform (0.0001–0.0024),
an order of magnitude below every other candidate's typical 0.02–0.09
range — MAE's embedding is almost completely insensitive to deskew,
contrast, denoise, and resolution changes, making the inversion
collapse even more dramatic by contrast.

- **Previous hypothesis** (after Rounds 6–7, tentative, n=2): inversion
  sensitivity tracks *continuous feature-distillation from another
  network* (DINOv2's self-distillation teacher, EVA-02's CLIP-feature
  target) — as opposed to discrete/labeled targets.
- **New evidence**: MAE has no teacher network at all — it isn't
  "continuous feature distillation" in any sense the previous
  hypothesis meant — yet shows the *most* extreme inversion sensitivity
  of any candidate tested, well beyond DINOv2 or EVA-02.
- **Revised interpretation**: the shared property isn't "matching
  another network's features" specifically — it's broader: **training
  objectives that require preserving or reconstructing something close
  to the raw, continuous visual signal** (MAE: literal pixel values;
  DINOv2/EVA-02: a continuous teacher embedding, one step removed from
  raw pixels but still not abstracted into a label/token/caption).
  Objectives that never require faithfully preserving raw appearance
  (ConvNeXt/Swin's class labels, SigLIP/NaFlex's text captions, BEiT's
  *discrete* tokenizer codes) show no elevated sensitivity. This is now
  supported by **3 positive representatives at graded strength** (MAE
  strongest at 0.21 recall, DINOv2/EVA-02 weaker but real at 0.92) and
  **5 negative ones** — crossing Jon's own bar of "more than one
  qualified representative" for the first time in this hypothesis
  chain, though still worth treating as a strong working pattern rather
  than a law, given the sample is still 8 candidates total.

**A secondary, unexplained finding worth recording rather than
smoothing over**: MAE's cross-representation consistency is **0.696**
— the lowest of all 8 candidates, lower even than SigLIP/NaFlex's
0.761–0.779. Unlike the SigLIP family, MAE uses a standard CLS token
(`num_prefix_tokens=1`, same mechanism as DINOv2/EVA-02/BEiT, all of
which score 0.86–0.97) — so the "attention-pooling differs from mean-
pooling" explanation that accounted for SigLIP's low scores does **not**
apply here. MAE's CLS token and its patch-token-mean genuinely diverge
more than any other CLS-token-based candidate's do. No confident
explanation offered — flagged as an open observation, not
force-fit into the existing confound story.

**Capability-qualification note**: this strengthens the case that
polarity-normalization policy (§ "Design decision arising directly from
Round 1") should be gated on the *specific* encoder chosen, with an
extra emphasis now that the effect size varies enormously even among
"sensitive" candidates (MAE's collapse is roughly 4× more severe than
DINOv2/EVA-02's) — a blanket "self-supervised encoders need polarity
normalization" rule would understate MAE's risk and overstate
DINOv2/EVA-02's by treating them as equivalent.

---

## Hypotheses tested — summary (capstone reference)

Jon's request: a single page answering "what did we actually establish"
without needing to re-read the round-by-round narrative. Verified each
line against the actual evidence above rather than transcribed as
given — wording tightened in three places where the one-line version
risked overstating what was found (noted below), and three additional
rows added for hypotheses that were also explicitly tested but didn't
make the original list. This table is the thing to re-read in six
months; the sections above are the audit trail behind it.

| # | Hypothesis | Outcome | Evidence | Confidence |
|---|---|---|---|---|
| 1 | ViTs are inversion-sensitive (as an architecture class) | ❌ Falsified | Rounds 3–9: BEiT, SigLIP, NaFlex (all ViT) show no elevated sensitivity; DINOv2, EVA-02, MAE (all ViT) do | High — spans both outcomes within the same architecture family |
| 2 | CNNs are inversion-robust (as an architecture class) | ⚠️ Not directly tested as a class | Rounds 2, 8: ConvNeXt, Swin both robust — but both are also supervised-classification-trained, so architecture and training objective are confounded with only 2 examples | Low — suggestive, not separable from hypothesis 4 with current data |
| 3 | Teacher distillation causes inversion sensitivity | ❌ Falsified | Round 9: MAE has no teacher network at all, yet shows the *most* extreme sensitivity of any candidate (recall@1 0.21 vs. DINOv2/EVA-02's 0.92) | High — a direct, clean counterexample |
| 4 | Continuous visual objectives (pixel or continuous-feature reconstruction) increase inversion sensitivity | ✅ Currently supported | Rounds 1, 6, 9 positive (DINOv2, EVA-02, MAE, graded strength); Rounds 2, 3, 5, 7, 8 negative (ConvNeXt, SigLIP, NaFlex, BEiT, Swin) | Moderate — 3 positive/5 negative, crosses the "more than one representative" bar but still only 8 candidates total |
| 5 | SigLIP-NaFlex's native resolution improves retrieval quality over fixed-resolution SigLIP | ❌ Not supported | Round 4 (n=24) and its scaled replication (n=87): retrieval precision essentially tied both times (0.798 vs. 0.809 at n=87); the underlying mechanism itself replicated cleanly (corr ≈ −0.80 both times) — mechanism real, practical benefit not demonstrated | High on "no demonstrated benefit"; the mechanism itself is separately well-confirmed |
| 6 | Single-cell argmax on a coarse feature map measures spatial localization | ❌ Falsified | Experiment 3.2: argmax location changes completely across stages/near-null crops while weighted centroid, spread, and entropy stay stable — the textbook measurement-artifact signature | High |
| 7 | ConvNeXt's unconditional channel-mean activation localizes document content | ❌ Not supported (tightened from a flat "falsified" — see note) | Experiment 3.2: activation distribution is close to uniform (normalized entropy 0.94–0.98) in every condition; 60–80% of mass was already on the document even *before* cropping, which falsifies the pilot's specific "border-dominated" claim, but the near-uniformity means genuine content-tracking was never demonstrated either | Moderate — the border-artifact explanation is cleanly falsified; the broader localization question is undetermined, not disproven |
| 8 | `autocontrast()` is a true no-op on this test image set | ✅ Confirmed | Exact 0.0000 drift replicated across 7 of 8 candidates spanning every architecture/objective tested (Swin: 0.0001) | High — the single most robustly replicated finding in this entire report |
| 9 | Deskew sensitivity (±2°) correlates with training objective or architecture family | ⚠️ No pattern found | Rounds 1–9: DINOv2 dips most, NaFlex/SigLIP/Swin dip slightly, ConvNeXt/EVA-02/BEiT/MAE stay near-perfect — no grouping by objective or architecture explains the split | Recorded explicitly as an absence, not a gap in analysis |
| 10 | Cross-representation consistency (pooled vs. patch-mean correlation) measures how much extra information patches carry | ❌ Confounded, not a clean signal | The full ranking (Swin 1.000 > EVA-02 > BEiT > ConvNeXt > DINOv2 > SigLIP/NaFlex > MAE 0.696) tracks each architecture's own native pooling mechanism's similarity to a plain mean far more cleanly than it tracks anything about patch information content (Swin's own pooling *is* a spatial mean) | Moderate — MAE's low score doesn't fit the pooling-mechanism story cleanly and is flagged as unexplained, not forced into it |
| 11 | Grad-CAM (task-conditioned attribution) shows real, content-relevant localization where unconditional pooling did not | ✅ Supported | Experiment 3.3: entropy dropped from 0.94–0.98 to 0.60–0.80, top-10% mass concentration roughly tripled, argmax varied meaningfully across images instead of universally centering, and the clean-image consistency check (which both prior methods failed) passed; image B showed a visually-confirmed shift from the reel-leader tag (87% of mass outside the document) to genuine document text after cropping | Moderate — n=3 images, and the classification target backpropagated from is semantically meaningless for this corpus even though the resulting spatial signal is not |

**Where this research continues**: `docs/BENCHMARK2_METADATA_LAYER_QUALIFICATION.md`
picks up from here — not evaluating encoders anymore (this document's
job), but designing the persistent metadata layer built on what this
qualification work established, including a real, measured
per-encoder-and-combined runtime/storage analysis and a concrete
encoder-responsibility recommendation grounded in the hypotheses table
above.

---

## Experiment 3.3: Grad-CAM (task-conditioned attribution) — the recommended replacement, executed

Per Experiment 3's closure: "future localization work should use
task-conditioned attribution (Grad-CAM or equivalent), not unconditional
feature-map pooling." Implemented and run. Same three test images, same
`page_boundary` crop source, same `characterize()` metrics as
Experiment 3.2 — imported directly, not reimplemented, so results are
on the same scale and directly comparable to the channel-mean baseline.

**Method**: `convnext_tiny.fb_in22k` loaded with its real ImageNet-22k
classification head (21,841 classes) — not the headless embedding
model used everywhere else in this research. For each image, forward
pass to the top predicted class, backpropagate that class's logit to
the same stage-3 feature map used throughout, weight channels by their
gradients (standard Grad-CAM), ReLU the result. **Caveat stated up
front**: the predicted class is an ImageNet-22k category, meaningless
for an archival document — that's fine for what this experiment
actually tests (does task-conditioned attribution behave differently
from unconditional pooling), which doesn't require the class label
itself to be correct.

### The result is a real, qualitative improvement over Experiment 3.2

- **Entropy dropped substantially**: 0.60–0.80 (Grad-CAM) vs. 0.94–0.98
  (channel-mean, Experiment 3.2) — the attention is genuinely
  concentrated, not close to uniform.
- **Top-10% mass concentration roughly tripled**: 0.35–0.72 vs.
  0.17–0.27 — a real, decisive peak, not a barely-distinguishable one.
- **Argmax varies meaningfully across images** — (0.21, 0.07), (0.93,
  0.21), (0.21, 0.36) for the three "before" conditions — not the same
  grid cell every time, unlike Experiment 3.2's near-universal
  `(0.5, 0.5)`. No evidence of the architecture-wide center-bias that
  undermined the previous method.
- **The clean-image consistency check — which both Experiment 3.1's
  argmax and Experiment 3.2's channel-mean approach failed — passes
  here.** Image C's near-null crop (99.8%→100% of frame) produced an
  essentially *identical* result before and after: centroid
  `(0.3957, 0.4068)` → `(0.3959, 0.4072)`, argmax exactly unchanged.
  This is the single strongest piece of evidence that Grad-CAM is a
  more trustworthy measurement than either prior method.

### Image B: a clean, visually-confirmed, decisive migration

Verified directly, not just from the numbers: in the "before" panel,
the argmax cell sits almost exactly on top of the bright reel-leader
tag itself. **`mass_inside_document_bbox = 0.130`** — 87% of Grad-CAM's
attention was on the border artifact, not the document, before
cropping. After cropping, argmax and centroid both move onto genuine
document text (the middle of the letter's body paragraph), distance to
document center drops from 0.461 to 0.043. This is exactly H1's
predicted pattern, on the same image the original Experiment 3 pilot
used, now with a decisive rather than ambiguous result. Worth an
honest caveat: this "before" prediction had very low confidence
(0.074, among 21,841 classes) — the underlying classification signal
itself was weak, even though the resulting attribution pattern is
clean.

### Image A: smaller, but consistent with the same pattern

`mass_inside_document_bbox` rises from 0.661 (before) to 1.0 (after,
trivially), distance to document center drops from 0.158 to 0.073 — a
real but more modest migration than image B. Checking the argmax cell
directly against the image: in "before," it sits in the black surround
just above the card, not on the reel-leader tag or the document; in
"after," the same relative grid position now falls on the document's
own header text, once the crop has removed everything else that used
to occupy that space. Consistent direction, smaller effect size than
image B.

### Verdict

**H1 (task-conditioned attribution shows real, content-relevant
localization, and cropping causally shifts it toward the document) is
supported** — with the same honest caveat as everywhere else in this
research: n=3 images, and the underlying classification target is
semantically meaningless for this corpus even though the resulting
spatial signal is not. This is a genuinely different outcome from
Experiments 3/3.1/3.2, not a rerun of the same ambiguous finding —
Grad-CAM passed a consistency check the previous methods failed, and
produced a visually confirmable, decisive result on the same test
image the original pilot used. The recommendation made when Experiment
3 was closed is now itself validated: unconditional feature-map pooling
was the wrong tool, and task-conditioned attribution is a materially
better one for this question.

**Notes on the three tightened labels** (1, 7, 10 above kept as given;
these are the ones where the evidence supports something slightly more
precise than the original one-line phrasing): hypothesis 7 is "not
supported" rather than a flat "falsified" because the pilot's *specific*
border-artifact claim did get falsified cleanly, while the broader
"does ConvNeXt localize content at all" question was left undetermined
by a near-uniform activation distribution, not disproven outright —
worth keeping those two claims distinguishable rather than collapsing
them into one verdict. Hypothesis 4 is marked "currently supported"
rather than "confirmed" deliberately, per Jon's own repeated framing
throughout this document: 8 candidates is not a large N, and this
hypothesis has already survived exactly one revision after new evidence
(Rounds 6→9) — the next falsification test would be another pure
pixel/continuous-reconstruction candidate that *doesn't* show
sensitivity, which hasn't been tried.

---

## Experiment 3 (deferred since Round 2): ConvNeXt spatial localization

**Question**: can ConvNeXt's genuine multi-scale spatial feature map
localize structure on real images where classical CV's own boundary
detector failed? Design from earlier in this document: no numeric
pass/fail threshold, deliberately — the practical criterion is whether
a human looking at the heatmap agrees it picks out real structure.
Implementation: `benchmark/experiment3_convnext_localization.py`.

**Test images, found by actually running the real detector, not
assumed**: scanned real LAC microfilm images in `data/working/` with
`core/image_analysis.py`'s own `analyze_image()` for the documented
failure signature (near-global, low-confidence `table_boundary` and a
falsely-zero `deskew_angle_deg`). Confirmed broadly present across this
batch — consistent with the original 174/218 finding. Selected two:
`oocihm.lac_reel_c10264.767.jpg` (table_confidence 0.125) and
`oocihm.lac_reel_c10295.1199.jpg` (table_confidence 0.125).

**A real visualization bug caught by directly looking at the first
result, not assumed to work**: the first version of the heatmap script
used a red-channel-boost color overlay — uninterpretable on this
specific test image, which turned out to be a naturally pink/rose-toned
photograph (not a grayscale scan), making the overlay indistinguishable
from the document's own inherent color cast. Fixed to a side-by-side
panel (original | grayscale activation map alone) before drawing any
conclusion from it.

**Finding, reported honestly rather than rounded toward success**: on
both test images, ConvNeXt's coarsest-stage activation map (a 7×7 grid,
per this session's capability matrix) shows one concentrated hotspot —
genuinely not the diffuse, near-global pattern the classical CV
detector produced on these same images, which is a real qualitative
difference worth recording. **But the hotspot lands in nearly the same
relative grid position in both images** (middle row, right-of-center),
which does not track where the actual document content sits (the two
images have very differently positioned documents) — it tracks much
more closely with where the bright microfilm reel-leader tag sits in
both frames. **The more likely explanation is that the coarsest stage
is responding to the single brightest, most salient rectangular object
in an otherwise near-black frame, not localizing genuine document/table
structure** — a materially less useful finding than what this
experiment set out to show, and reported here as inconclusive-leaning-
negative rather than a success, per the same discipline as every other
result in this document. Sent both heatmap images to Jon directly for
an independent look before treating this as settled.

**This is the same lesson this project already learned once, in a
different context** — worth naming the connection explicitly rather
than treating this as a new, unrelated problem. §1's LAC-microfilm
black-film-surround finding showed classical CV's tone measurements
were meaningless until scoped to the cropped page region, not the full
frame. This result suggests the same may be true here: testing on the
**uncropped** frame (black surround, reel-leader tag, and all) likely
confounds a spatial-feature-localization test the same way it
confounded polarity detection — the encoder has no reason to know the
reel-leader tag isn't the "interesting" object in the frame.

**Natural next step, not yet run**: re-crop both test images to the
actual document region (excluding the black surround and reel-leader
tag) before feeding them to ConvNeXt, and check whether genuine
document-structure localization emerges once that confound is removed.
Until that's done, Experiment 3's original question — can spatial
features find structure CV misses — remains genuinely open, not
answered either way by this round.

---

## Experiment 3.1: crop-confound test (redesigned per Jon's H0/H1 spec)

**Explicit framing, per Jon's direction**: the pilot is treated as
having identified a confound, not as evidence for or against ConvNeXt.
This experiment tests whether *border artifacts specifically* are the
causal factor, not whether ConvNeXt performs localization.

- **H0**: after removing non-document border artifacts, activation
  remains unrelated to document structure.
- **H1**: after removing non-document border artifacts, activation
  shifts toward meaningful document regions.

**Crop source (requirement #1 — no manual cropping)**: no dedicated
dewarp sidecar existed for these specific images, so the crop used is
`core/image_analysis.py`'s own `page_boundary` detector — a real,
already-built pipeline component, run exactly as the pipeline would run
it, not a hand-drawn box.

**Test set**: the same two difficult images from the pilot
(`oocihm.lac_reel_c10264.767`, page_confidence 0.806;
`oocihm.lac_reel_c10295.1199`, page_confidence 0.815 — both with
reel leader, archive label, and microfilm surround), plus one clean
page (`oocihm.lac_reel_c10301.529`, page_confidence 0.993, detected
boundary covers 99.8% of the frame — essentially a near-no-op crop,
included specifically as an internal consistency check per the
experiment design).

**A real bug caught before trusting the numbers**: the first run raised
a `TypeError` (numpy bool/float types aren't JSON-serializable) —
fixed by casting to native Python types before logging, not by
suppressing or ignoring the error.

### The consistency check failed, and that's the actual headline finding

The clean-page sanity check was included specifically to catch exactly
this: if before/after differ wildly despite an almost-null crop change
(99.8% → 100% of frame), that's a flag on the method, not a real
finding. **It failed.** The clean image's hotspot moved from
`(0.929, 0.214)` (before) to exactly `(0.5, 0.5)` (after) despite the
crop removing essentially nothing. Visually inspecting both panels
confirmed the two input images are nearly identical — the argmax
genuinely relocated based on a trivial input change.

### Re-examining the "before" condition numerically, not visually

The pilot's qualitative read (§ Experiment 3 above) concluded the
hotspot tracked the reel-leader tag. Checking this against real
coordinates rather than eyeballing a rendered image: image A's document
bbox center sits at pixel (1357, 2449); the full frame's *geometric*
center is (1664, 2188). **The "before" hotspot landed at exactly the
frame's geometric center — not the document's center, and not clearly
the reel tag's position either.** The same held for image B. This
undercuts the pilot's own visual interpretation — the apparent
"tracks the reel tag" pattern is at least as consistent with, and
arguably better explained by, a much more mundane effect: ConvNeXt's
coarsest stage gravitating toward the geometric center of whatever
image it's given, largely independent of content.

### The "after" condition does land on real text — but this doesn't cleanly resolve anything

Visually inspecting both cropped panels: the post-crop hotspot **does**
fall on genuine text content in both images (a row of names/columns for
image A, a paragraph of body text for image B) — not blank margin, not
a residual border scrap. That's consistent with H1. **But a tightly
cropped page naturally has its own real content filling the geometric
middle** (typical document layout: centered text block, roughly
symmetric margins) — which means "activation gravitates to image
center" and "activation tracks real content" produce visually
indistinguishable results once the input is basically just a document.
A single-argmax-cell metric on a 7×7 grid cannot separate these two
explanations with the data collected here.

### Classification (requirement #4)

| Image | Before | After | Migration |
|---|---|---|---|
| difficult_A (c10264.767) | border/frame-center-dominated (not clearly document- or reel-tag-specific) | document-dominated (lands on real text, confound above still applies) | shifted, cause ambiguous |
| difficult_B (c10295.1199) | border/frame-center-dominated (same caveat) | document-dominated (same caveat) | shifted, cause ambiguous |
| clean_C (c10301.529) | document-dominated (off-center, no clear artifact) | document-dominated (moved to center despite near-null crop) | **inconsistent** — flags the metric, not the hypothesis |

### Verdict: confounded / insufficient evidence — not supported, not falsified

Per Jon's explicit instruction not to round toward success: **this
experiment cannot cleanly support or falsify H1.** Two independent
problems, both real:

1. **A newly-identified confound**, distinct from the border-artifact
   one the pilot proposed: ConvNeXt-tiny's coarsest-stage (7×7)
   activation shows a strong pull toward the geometric center of
   *whatever* image is fed to the model — plausibly an artifact of the
   model's own resize/center-crop input transform, or of receptive-field/
   padding structure at that depth, not evidence of content-tracking.
2. **A metric-robustness failure**: the clean-page consistency check —
   built into the design specifically to catch this — failed. A
   near-null crop change (0.998 → 1.0 area fraction) flipped the argmax
   completely. Single-cell argmax on a 7×7 grid is demonstrably too
   fragile to trust as a localization measurement on this evidence.

Classified per Jon's four valid outcomes: **confounded, shading into
insufficient evidence** — not "hypothesis supported" (the after-crop
content-match is real but not attributable to border-removal
specifically) and not "falsified" either (a genuine shift did occur,
just not cleanly interpretable). The honest conclusion is that this
experiment, as designed, cannot yet distinguish "border artifacts cause
mislocalization" from "ConvNeXt's coarse stage has a content-independent
center bias that happens to coincide with typical document layout."

### Comparison against the original "crop before trust" finding

**Only partial, weak reinforcement — not a replication.** The original
`image_analysis.py` finding (§1) had a clean, well-diagnosed causal
mechanism: Otsu tone thresholding measurably broke on the black film
surround (43/218 real false positives), and cropping to the page
region measurably fixed it (0/218) — a single, well-understood
confound, cleanly removed. Experiment 3.1 does **not** replicate that
clean pattern. It surfaces a **different, murkier problem**: a
geometric center-bias that has no analog in the original finding (Otsu
thresholding has no "gravitates toward the middle" failure mode to
confuse with a real effect). The general methodological principle —
*measurements can be silently confounded by something other than the
variable you're testing, and this must be checked directly rather than
assumed* — is reinforced. The specific mechanism, and the clean
"crop fixes it" narrative, is not. Treat these as related in spirit,
not the same finding twice.

### Recommended next step, not yet run

Replace single-argmax-cell centroid with a weighted centroid (activation-
magnitude-weighted average position across all cells, not just the
single max) before attempting this test again — the current metric's
sensitivity to near-ties is very likely why the clean-page consistency
check failed. Until that's fixed, further rounds of this specific
experiment would just be re-measuring the same fragile metric.

---

## Experiment 3.2: robust spatial localization (measurement validation)

Explicit framing per Jon's direction: this experiment's primary goal is
validating the *metric*, not judging ConvNeXt. Same image set and crop
conditions as Experiment 3.1 (same `page_boundary` crops, no manual
cropping). Replaced single-cell argmax with a full characterization:
weighted centroid, activation spread (variance/covariance), normalized
entropy, top-10/20/50%-mass concentration, and centroid-to-argmax
distance — computed at all 4 ConvNeXt stages, not just the coarsest,
specifically to address where any center-bias comes from.
Implementation: `benchmark/experiment3_2_robust_localization.py`.

### The result is unusually clean, and it resolves in one clear direction

**Across all 3 images, both crop conditions, and all 4 stages, the
weighted centroid sits within ~0.05 of dead-center `(0.5, 0.5)` every
single time** — while the argmax bounces wildly: for image A's "before"
condition alone, argmax moves from `(0.87, 0.83)` at stage 0, to
`(0.16, 0.13)` at stage 1, to `(0.46, 0.75)` at stage 2, to `(0.5, 0.5)`
at stage 3 — four completely different locations across four stages of
the *same* forward pass on the *same* image. The centroid, meanwhile,
stays put: `(0.495, 0.500)`, `(0.493, 0.491)`, `(0.492, 0.494)`,
`(0.467, 0.512)` across those same four stages — remarkably stable
while argmax is essentially noise.

**This is precisely the "measurement artifact" signature Jon's own
requirement #6 described as an example**: *"the argmax changes
dramatically while the centroid and activation distribution remain
nearly unchanged."* That is exactly what the data shows. **H0
(argmax-reduction artifact) is well supported; the apparent
"localization" pattern from Experiment 3.1 was substantially a metric
artifact, not a robust representation property.**

### Full metrics (coarsest stage, primary view)

| Image | Cond. | Centroid | Argmax | Centroid–argmax dist | Spread | Entropy (norm.) | Top 10/20/50% mass | Dist. to doc. center | Mass inside doc. bbox |
|---|---|---|---|---|---|---|---|---|---|
| A | before | (0.467, 0.512) | (0.5, 0.5) | 0.035 | 0.135 | 0.957 | 0.22/0.37/0.70 | 0.076 | **0.600** |
| A | after | (0.498, 0.511) | (0.5, 0.5) | 0.011 | 0.129 | 0.945 | 0.27/0.40/0.71 | 0.011 | 1.000 |
| B | before | (0.454, 0.501) | (0.5, 0.5) | 0.046 | 0.143 | 0.975 | 0.17/0.31/0.63 | 0.085 | **0.801** |
| B | after | (0.517, 0.504) | (0.5, 0.5) | 0.017 | 0.127 | 0.965 | 0.21/0.37/0.67 | 0.017 | 1.000 |
| C | before | (0.501, 0.477) | (0.93, 0.21) | 0.502 | 0.162 | 0.972 | 0.21/0.33/0.63 | 0.022 | 1.000 |
| C | after | (0.487, 0.491) | (0.5, 0.5) | 0.015 | 0.157 | 0.981 | 0.17/0.30/0.60 | 0.015 | 1.000 |

Normalized entropy near 1.0 (max possible = fully uniform) across every
row, and top-10% mass fractions of only 0.17–0.27 (a perfectly uniform
49-cell map would give exactly 0.10) — **the activation distribution,
honestly characterized as a whole rather than reduced to its peak, is
close to uniform and only mildly concentrated anywhere.**

### The most decisive single finding: `mass_inside_document_bbox`

**Even in the "before" (uncropped) condition, 60–80% of total
activation mass already sits inside the true document region** — not
on the black surround, not on the reel-leader tag. This directly
contradicts the original Experiment 3 pilot's "border-dominated"
reading. Looking only at the single argmax cell made it look like
activation was centered on the frame (potentially the border/reel-tag
region); looking at the full distribution shows most of the mass was
on the document all along, even without cropping.

### Answering Jon's specific questions

- **Does cropping move the weighted centroid?** Barely — all six
  before/after centroids were already close to center; the shifts
  (0.02–0.06 in normalized units) are small and inconsistent in
  direction, not a decisive migration.
- **Does cropping reduce activation spread?** Yes, consistently, though
  modestly — every image showed a small spread decrease after cropping
  (A: 0.135→0.129, B: 0.143→0.127, C: 0.162→0.157). The clearest
  consistent before/after effect in the whole dataset, even if small in
  magnitude.
- **Does cropping increase concentration on document regions?** Only
  trivially — "after" is 100% inside the document by construction (the
  crop *is* the document). The real information is in "before":
  60–80% of mass was already inside the document region without any
  cropping at all.
- **Is the argmax stable while the centroid moves?** The opposite:
  **argmax is highly unstable (bounces across all four stages) while
  the centroid stays essentially fixed.** This is the clearest possible
  measurement-artifact signature.
- **Is center bias still present after replacing argmax?** Yes, the
  centroid is still consistently near-center — but this is far less
  alarming once combined with the entropy/mass findings: these
  documents' true `page_boundary` region is itself roughly centered and
  covers over half the frame, so a broadly-uniform, mildly-centered
  distribution is not distinguishable from "correctly weighted toward
  where the document actually is." The center-bias and the
  document-location are not in tension here the way the pilot implied.

### Where does the center-bias come from (requirement #7)?

Checked across all 4 stages, not assumed to be a coarsest-stage-only
effect: the weighted centroid sits near `(0.5, 0.5)` at **every stage**,
including stage 0 (56×56, minimal downsampling) — not something that
only emerges at the coarsest, most-downsampled stage. Spread is
similar in magnitude across stages too (stage 0: ~0.17, stage 3: ~0.13–0.16
— a mild decrease at the coarsest stage, not a qualitative jump). **This
argues against "a consequence of using the coarsest feature stage" or a
pure large-receptive-field/padding effect specific to depth**, and is
more consistent with either a **resize/center-crop preprocessing bias**
(the same transform feeds every stage) or a general **architecture-wide
positional prior** (plausibly inherited from ImageNet-style training,
where the subject is conventionally centered) — not definitively
distinguishable between those two from this data alone, but clearly not
a deep-stage-specific artifact.

### Verdict, per Jon's four valid outcomes — reported separately for the two questions this experiment actually answers

**On the measurement-validation question (this experiment's primary
objective): H0 supported.** The argmax-based "localization" signal from
Experiment 3.1 was substantially a reduction artifact — a noisy,
unstable statistic computed on top of a distribution that, properly
characterized, is close to uniform and centrally-symmetric. Confirmed,
not merely suspected.

**On the original Experiment 3 question (does ConvNeXt localize
document structure CV missed): falsified, specifically on the
"border-dominated" reading** — the pilot's central claim doesn't survive
contact with the full-distribution evidence (60–80% of mass was already
on the document pre-crop). But this is **not** evidence *for* genuine
spatial localization either — the near-uniform, high-entropy character
of the distribution across all conditions suggests this specific
measurement (channel-mean of a coarse stage, no class- or
task-conditioning) may simply not carry strong spatial signal at all,
independent of border considerations. A fair summary: **insufficient
evidence that ConvNeXt's plain coarse-stage activation performs
meaningful document localization, and clear evidence that the pilot's
specific "it's tracking the border artifact" explanation was wrong.**

### What would be needed to actually test spatial localization, if pursued further

The uniform, high-entropy character of even the "after" (correctly
cropped) distributions suggests plain channel-mean pooling at a coarse
stage isn't the right tool for this question at all — a genuine test
would need either a task-conditioned signal (e.g. Grad-CAM-style
gradient weighting toward a specific class/output, not an unconditional
channel mean) or a finer spatial stage examined with a method robust to
the near-uniformity problem surfaced here.

---

## Experiment 3 — CLOSED (2026-08-01)

**Status**: Closed.

**Outcome**: Original hypothesis not supported.

**Reason**: Localization metric failed qualification. Experiment 3.2
showed the single-cell argmax statistic underlying Experiment 3's
original finding was substantially a measurement artifact (argmax
unstable across stages while weighted centroid, spread, and entropy
remained stable) — the pilot's "border-dominated" reading does not
survive full-distribution scrutiny (60–80% of activation mass was
already on the document pre-crop), and the underlying activation
distribution is close to uniform throughout, giving no reliable signal
of genuine document-structure localization either way.

**Replacement recommendation**: future spatial-localization work on
this pipeline's candidates should use task-conditioned attribution
(Grad-CAM or equivalent) rather than unconditional feature-map pooling
— the method this round of investigation shows is not qualified to
answer the question it was built to answer.

**Capability-qualification table, final status**: "Spatial
localization | ConvNeXt" moves from *pending* to *closed, method
disqualified* — not "ConvNeXt cannot localize," but "this measurement
approach cannot tell us either way, and a properly qualified method is
required before revisiting this capability row at all."
