# Planning / queued work

Deferred tasks not being worked on right now - picked up when their
prerequisite is done or Jon says go. Not a how-to-run guide (see
`README.md`/`PIPELINE_WORKFLOW.md`) or a symbol index (see
`docs/CODE_MAP.md`) - just a queue.

---

## Queued: column-number-anchor / page-dewarp INTEGRATION (research phase closed, as of 2026-08-06)

**Status: Jon's call - research is substantially complete; what remains
is an engineering problem (integration, more ground truth), not an open
research question.** Full log (17 rounds, long): `docs/COLUMN_NUMBER_
ANCHOR_RESEARCH.md` - its own "Status & handoff (as of Round 17) - READ
THIS FIRST" section near the top is the authoritative catch-up point,
don't read the doc top-to-bottom cold. Results also folded into `docs/
MULTI_SOURCE_VOTING_CLASSIFIER_PROPOSAL.md`'s evidence-source inventory.

One-line summary: started as "can printed column numbers narrow the
search window for `core/auto_sidecar.py`'s `locate_columns()`," and this
was originally an EXTRACTION-quality investigation (row/column boundary
detection for Stage 6), not classification-first - the classification-
prior use (see the multi-source voting classifier entry below) was a
secondary benefit Jon identified after the refinement started working,
not the original goal. Don't conflate the two purposes when picking
either one back up.

Where it landed:
- 1901's column-detection problem turned out to be PHYSICAL, not a
  calibration/tuning problem: the source scans come from a bound volume
  with its own gutter crease, and the header row is measurably NOT level
  across the page width (~80px vertical drift left-to-right, visually
  confirmed, not inferred) - consistent with page curvature/scan skew a
  single global deskew angle can't fully correct.
- A gradient-based (Scharr) line detector was found to strictly dominate
  the existing dark-ridge/threshold detector on every year tested - a
  real, validated upgrade candidate for `_find_vertical_ruling_line()`,
  though it surfaced a "wrong-line lock-on" failure mode on tightly-
  packed columns that isn't resolved.
- `core/page_dewarp.py` (new, research-only module) - crease detection +
  per-column dewarp via `cv2.remap`. Includes one honestly-caught false
  start: an initial "67% match rate" claim turned out to be an
  extrapolation bug that corrupted the image while still moving the
  metric the right direction - caught by looking at the actual rendered
  output, not trusting the number.
- Genuinely unresolved: only 1 of 6 checked pages has real (not
  visually-guessed) ground truth for validating the crease-position
  prediction; blind detection (crease or table-border) has failed
  without a prior every time it's been tried, in several different
  shapes; nothing is wired into `core/auto_sidecar.py` or any production
  path anywhere.
- The specific problem Jon described when pausing: 1901's census spans
  2 photographed pages with a gutter down the middle, creating localized
  warping concentrated near the crease - the in-progress fix being built
  was slice-at-crease -> dewarp each half independently -> restitch, not
  yet completed/validated at the time work paused.

**Next session picking this up**: start at the doc's own Round 17
handoff section, not this summary - this stays a pointer, not a
replacement for the real log. Since research is now closed, the actual
next steps are engineering ones: (1) get real (not visually-guessed)
ground truth for the crease-position prediction on more than 1 of the 6
checked pages, (2) decide the integration shape into `locate_columns()`'s
existing resolution cascade, (3) swap the validated gradient-based line
detector in for the existing dark-ridge one, (4) resolve the "wrong-line
lock-on" failure mode on tightly-packed columns before trusting this in
production.

## Queued: multi-source voting classifier (added 2026-08-06)

**Status: research/plan complete, not implemented. Not blocked, but
should not be started without a further explicit go** - see the full
doc for real prerequisite gaps.

Full research + proposal: `docs/MULTI_SOURCE_VOTING_CLASSIFIER_PROPOSAL.md`.

One-line summary: Stage 5 routing currently trusts Gemma's single output
as final, despite `core/decision_engine.py` already computing an
8-vision-tower consensus signal (validated, `docs/BENCHMARK2_3_MULTI_
TOWER_ROUTING_AUDIT.md`) that never feeds back into the actual routing
decision - it's recorded and audited after the fact, never acted on.
Proposal doc lays out a tiered voting design (structural priors -> tower
consensus -> Gemma -> structural/extraction evidence) and, critically,
what's missing before it could be built: the 56 disagreement cases from
the original audit were never human-reviewed (highest-value next step,
~90% done already), DocLayout-YOLO's domain transfer to this corpus is
unmeasured, and no confidence-gating threshold has been chosen anywhere
in the system yet.

Also contains a cross-session status brief from the parallel YOLO
layout-detector training effort (verbatim, 2026-08-05/06) - so THAT work
can be picked back up cold too, not just the voting proposal. Includes a
notable convergent finding: two independent investigations the same day
(the YOLO session's baseline probe, and this session's single-shot
Gemma classifier test) both flagged the 1926 census pull as structurally
anomalous, corroborating the standing `[[project_prairie_census_
different_form]]` memory note. Worth treating 1926 (and likely 1906/1916)
as a genuinely different form family needing its own template/prompt
handling, not "census, generic."

## Queued: speculative decoding for slow VLM inference (added 2026-08-06)

**Status: deferred, not blocked - pick up after the manual-classification
UI and routing work below are settled.**

Raised while diagnosing why several models in the overnight VLM routing
batch (`diagnostics/test_all_vlms_prompt_tiering.py`) were painfully slow
(`qwen3vl2b_thinking` hit 35 minutes for a single image before being
killed). Jon's Pixel 9 Pro has a speculative-decoding toggle for its
on-device Gemma-4-E2B-it - wanted to know if the same trick applies here.

**Important scoping note for whoever picks this up**: speculative
decoding is a pure LATENCY optimization - a correctly-implemented version
produces outputs identical to normal greedy decoding (a small draft model
proposes tokens, the target model verifies them in a batched pass). It
does NOT fix accuracy or confidence calibration - do not reach for this
if the actual problem turns out to be calibration (it wasn't the fix for
the single-shot classifier prompt's broken confidence field, see
data/outputs/single_shot_classifier_test/log.txt - that needs a different
approach entirely).

If genuinely pursued for speed:
- HF `transformers`' `generate(assistant_model=...)` needs a draft model
  with a COMPATIBLE tokenizer/vocab, ideally same-family and smaller -
  no smaller Gemma-4 checkpoint exists in this project's current model
  roster (`config/models/*.yaml`) to use as one.
- `prompt_lookup_num_tokens=` (n-gram self-speculation, no second model
  needed) is a free, zero-setup alternative worth trying FIRST - but it
  only helps when output echoes text already in the prompt, which
  doesn't describe this project's short classification answers, so
  expect limited benefit before spending real time on it.
- The Pixel's toggle is Google's on-device AI Edge/MediaPipe
  implementation (bundled into the LiteRT `.task` package) - a
  DIFFERENT system from HF `transformers`, not directly portable here.

## Queued: Lightweight visual document classifier survey (added 2026-08-06)

**Status: BLOCKED - waiting on Jon to review the overnight VLM routing
test results first** (`data/outputs/overnight_vlm_routing_test/` -
`log.txt` + `summary.json`, the multi-model batch comparing 11+ VLMs
against the same v7 gated-binary routing tree Gemma uses, per
diagnostics/test_all_vlms_prompt_tiering.py). Do not start this survey
until that review has happened - its findings (which models handle the
routing task at all, which mode-collapse or degenerate, relative
accuracy) directly inform how to scope this survey, and starting it
blind would risk redundant work.

**This is research only - do not design a new architecture until the
survey is complete**, per Jon's explicit instruction.

### Task, as given by Jon (2026-08-06), verbatim

Conduct a comprehensive literature and open-source survey on lightweight visual document classification for historical records.

This is NOT an OCR project.

The objective is to classify a scanned historical document into a project taxonomy before OCR or extraction.

Current examples include:

- Canadian census schedules (multiple years)
- Passenger manifests / immigration lists
- Maps
- Photographs
- Certificates
- Newspapers
- Letters
- Books
- Blank forms
- Unknown / quarantine

Investigate:

1. Existing document classification models
   - Vision-language models trained for classification
   - Pure vision classifiers
   - Vision encoders with classifier heads
   - Hybrid approaches

2. Existing pretrained checkpoints
   Search Hugging Face, GitHub and published research for:
   - document classification
   - historical document classification
   - genealogy documents
   - census
   - passenger manifests
   - archival documents
   - RVL-CDIP
   - Tobacco3482
   - DocLayNet
   - DocILE
   - IIT-CDIP
   - FUNSD
   - CORD
   - PublayNet

3. Gemma 4 specific work
   Search for:
   - Gemma 4 E2B
   - Gemma 4 vision
   - Gemma classification
   - Gemma vision classifier
   - Gemma classifier heads
   - LoRA checkpoints
   - Fine-tuned document classifiers

4. Similar work for:
   - Qwen VL
   - InternVL
   - Pixtral
   - Florence
   - LFM2-VL
   - Moondream
   - SmolVLM
   - Chameleon

5. Research using frozen vision embeddings with lightweight classifier heads.

6. Research using pooled/projected vision embeddings instead of autoregressive generation.

7. Multi-encoder ensemble classification using multiple vision towers.

8. Small transformer classifier heads operating directly on vision tokens.

For every approach report:

- model
- parameter count
- license
- hardware requirements
- inference speed
- reported accuracy
- datasets used
- whether checkpoints are publicly available
- links

Finally compare every discovered approach against our current architecture:

Current pipeline:

Image
→ Tier 0
→ Tier 1
→ Tier 2
→ Tier 3
→ OCR

Current routing is prompt-based using general VLMs.

Evaluate whether replacing prompt-based routing with a dedicated classifier would likely improve:

- speed
- determinism
- confidence calibration
- maintainability
- extensibility
- accuracy

Do not design a new architecture until the survey is complete.

### Context for whoever picks this up

- The "Tier 0 -> Tier 1 -> Tier 2 -> Tier 3" pipeline referenced above is
  the gated-binary routing tree developed and validated 2026-08-05/06 -
  full history in diagnostics/test_gemma_prompt_tiering.py,
  test_gemma_prompt_tiering_variants.py (v1-v9 prompt-wording
  experiments, v7 adopted as current-best), and
  test_all_vlms_prompt_tiering.py (the overnight cross-model comparison
  this survey is blocked on).
- Separately, docs/GEMMA_HIERARCHICAL_ROUTING_INVESTIGATION.md covers a
  DIFFERENT, already-closed investigation (vision-tensor cache reuse
  across multiple prompts) - parked, not related to this survey's
  question (which model/architecture should do the classifying), don't
  conflate the two.
- Known related findings already on record, relevant background for this
  survey (not to be re-derived): [[project_chameleon_modecollapse]]
  (Chameleon predicts the same category regardless of image - a real
  prior finding, not this survey's job to rediscover), and the
  now-confirmed Gemma census/passenger-manifest confusion investigation
  (v1-v9 prompt tuning in the files above) as a concrete example of where
  prompt-based routing has shown real, hard-to-fix failure modes that a
  dedicated classifier might avoid.
