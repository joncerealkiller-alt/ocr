# Stage 4 Research — Error Analysis of the Remaining ~8% (2026-08-08)

Status: **Research and methodology design only. No code, no training, no
GPU inference.** Follows `docs/GEMMA_HIDDEN_STATE_ROUTING_HEAD_DESIGN.md`
(Stages 1-2, multi-seed, MLP) and `docs/GEMMA_VISION_TOWER_FINETUNE_
RESEARCH.md`'s own recommendation to do this error analysis *before*
spending compute on a vision-tower fine-tune. Every claim is tagged **[A]
confirmed** (verified against real cached data or installed-environment
facts), **[B] interpretation**, or **[C] hypothesis**.

## Starting facts (all [A], all already measured this session)

- Frozen `vision_encoder` + linear probe: 92.0 ± 0.5% test (10 seeds)
- `vision_projected`: 91.7 ± 0.4% test
- `decoder_early`/`decoder_mid`/`decoder_final`: statistically
  indistinguishable from the above except `decoder_early` (~1pt worse,
  the one location that actually separated from the pack)
- Shallow MLP on `vision_encoder`: 92.6 ± 0.5% (no real gain over linear)
- Full autoregressive generation: ~94%
- Held-out test set: 332 images, 8 taxonomy classes, uneven support
  (`printed_document` 175, `genealogy_chart` 1, `mixed_text_image` 3 —
  confirmed from this session's own confusion matrices)
- **[A] Real environment/data fact found while preparing this doc**: the
  merged feature cache (`data/outputs/gemma_hidden_state_probe/
  features_stage2.pt`) stores only `x_by_loc`/`y`/`classes` — **no image
  paths**. Paths exist only in the per-chunk shard files
  (`data/outputs/gemma_hidden_state_probe/shards_stage2/*.pt`, each with
  a `"paths"` field). Any error-analysis tool needs to either read paths
  from the shards directly (self-documenting, doesn't depend on reasoning
  about ordering) or reconstruct them from `get_split()`'s deterministic
  item order — the latter is verified safe for Stage 2's cache
  specifically (zero extraction failures occurred in that run, confirmed
  by grepping its report for "FAILED": none found), but **not safe in
  general** — Stage 1's run *did* drop one image
  (`e001211838.jpg`, a real `FileNotFoundError`), which would have
  silently shifted alignment between `get_split()`'s item list and the
  cached tensor rows if paths weren't tracked per-shard. **Recommendation:
  always resolve paths from the shard `"paths"` field, never assume
  index alignment with a fresh `get_split()` call**, precisely because
  this project has already had one real case where that assumption would
  have been wrong.

---

## 1. Failure taxonomy

**[B]** A single, mutually-exclusive "primary cause" per failure plus
optional secondary tags is the right shape — it matches the
`EvidenceRecord`/Decision Engine's existing preference for structured,
auditable evidence (`metadata` dict, explicit `score_kind`) over free-text
judgment calls, and it forces the reviewer to commit to one dominant
explanation instead of hedging on every case.

Proposed taxonomy (deliberately incomplete per the prompt's own
instruction — designed to be extended, not treated as exhaustive):

| Code | Primary cause | What it means | Distinguishing signal |
|---|---|---|---|
| `image_quality` | Poor scan/photo quality | Blur, low resolution, heavy noise | Visible in the raw image itself, independent of any prediction |
| `skew_crop` | Geometric/framing defect | Severe skew, incorrect crop, page edge cut off | Visible in the raw image; distinct from quality because a sharp, well-lit, badly-cropped image is a different problem than a blurry well-cropped one |
| `low_contrast` | Faded/low-contrast source | Text/content present but hard to distinguish from background | Often specific to microfilm-sourced images per this project's own corpus makeup |
| `multi_content` | Multiple document types in one frame | E.g. a photo of a ledger page next to a loose photograph | Genuinely ambiguous even to a human reviewer applying the taxonomy correctly |
| `ambiguous_taxonomy` | Category boundary itself is unclear | The image plausibly fits 2+ categories under the current taxonomy's written definitions | Distinguished from `multi_content` — this is a *definition* problem, not a *content* problem; a single, unambiguous image can still sit on a fuzzy boundary |
| `label_error` | Ground truth itself looks wrong | Reviewer, looking only at the image (not the model's prediction first), would assign the model's predicted class instead | See section 5 for how to check this without just trusting the model's own opinion |
| `rare_class` | Insufficient training examples | True class has very low train-split support (e.g. `genealogy_chart`: 1 test image total) | Read directly from class support counts, not subjective |
| `missing_visual_info` | Genuinely no signal in the image | A human, given only the image (no prior taxonomy knowledge, no external context), could not confidently assign a class either | The key operational test for a real vision limitation — see section 6 |
| `requires_reasoning` | Visual signal present, but classification needs interpretation beyond raw appearance | E.g. distinguishing a `handwritten_ledger` from `dense_tabular_rows` may hinge on subtle structural cues that need reasoning about function, not just appearance | See section 6 for the objective test distinguishing this from `missing_visual_info` |
| `preprocessing_artifact` | Pipeline preprocessing (deskew/crop/resize) introduced or amplified a problem not present in the original source | Requires comparing the version Gemma actually saw against an earlier pipeline stage's version of the same image | Only assignable if preprocessing intermediates are available (section 2) |
| `unknown` | None of the above fit, or reviewer genuinely can't tell | Escape valve — required so the taxonomy doesn't force a false-confidence answer, matching this project's own standing pattern of first-class abstention (`uncertain_review` in the classifier taxonomy, per-level abstention in the parked hierarchical-prompt investigation) | Explicitly not a failure of the taxonomy design; a taxonomy with zero `unknown` assignments after real review is a red flag that reviewers are being forced into false confidence |

**Secondary causes**: same list, multi-select, for cases like "the image
is genuinely low-contrast AND sits on an ambiguous taxonomy boundary" —
tracked separately from primary cause so aggregate statistics (section 4)
can report primary-cause distribution as the headline number without
losing the secondary detail.

## 2. Visual review workflow

**[B]** Given this project's own stated preference
([[feedback_manual_ui_direct_interaction]]: prefer click-to-set / direct
manipulation over coordinate entry for manual review tools) and existing
precedent (the Column Calibration UI, per [[project_column_calibration_ui]]),
a lightweight per-image review screen — not a coordinate-entry form — is
the right shape here too, even though this review task itself has no
spatial/coordinate component (it's closer to the classification review
already implicit in how `run_gemma_flat8_on_benchmark_holdout.py`'s
predictions get inspected today, just formalized into a UI instead of a
text report).

**Per-failure display** (everything listed in the prompt, all derivable
from data already produced this session with zero new inference):

- **Image** — resolved via the shard `"paths"` field (see the starting-
  facts note above), loaded directly from disk, no re-inference needed.
- **True label / predicted label** — from the cached `y` tensor and the
  probe's stored predictions (need to persist per-image predictions,
  which the existing `evaluate()` function computes internally but
  currently only aggregates into a confusion matrix and discards the
  per-image `preds` tensor — a genuinely new, small addition: save
  `preds`/`probs` alongside the confusion matrix rather than only
  printing summary stats).
- **Prediction confidence** — the probe's own softmax max-probability,
  already computed in `evaluate()` (`max_conf` in the existing script,
  currently only aggregated into "mean confidence, correct vs. wrong" —
  again just needs to be persisted per-image instead of discarded).
- **Probe score** — full softmax distribution over all 8 classes, useful
  for seeing whether a wrong prediction was a confident wrong answer or a
  near-tie between the true and predicted class (a near-tie is a very
  different failure mode than a confident wrong answer, directly relevant
  to section 6's vision-vs-reasoning question).
- **Embedding location** — which of the 5 hook locations (or all 5
  side-by-side) the prediction/analysis is being shown for; since Stage 2
  cached all 5 per image, a reviewer could toggle between them to see
  whether an error is location-specific (e.g. wrong on `decoder_final`
  but right on `vision_encoder`) or universal across all 5 — a genuinely
  useful signal for section 7's decoder-comparison question that a
  single-location review would miss entirely.
- **Optional preprocessing intermediates** — **[A]** this project already
  has a defined Stage 0-6 pipeline with real intermediate artifacts
  (deskew, crop, resize outputs per `core/manifest_pipeline.py` /
  `docs/PIPELINE_STAGE_TERMINOLOGY.md`); showing the pre-deskew /
  pre-crop version alongside the final version Gemma actually saw is the
  only way to ever assign the `preprocessing_artifact` cause with
  confidence rather than guessing. **[C]** not every image will have
  these intermediates retained on disk depending on pipeline
  configuration — needs checking against what's actually persisted for
  images in this specific held-out split before assuming it's always
  available.

**Workflow**: filtered to only the ~26-27 test-set images the probe got
wrong (332 × ~8% ≈ 26), a small enough set for one reviewer to work
through by hand in a single session — no need for batching/pagination
infrastructure beyond a simple "next/previous" control. Each screen: show
the data above, let the reviewer pick one primary cause (single-select)
+ zero or more secondary causes (multi-select) from the taxonomy in
section 1, plus a free-text note field for anything the fixed taxonomy
doesn't capture (feeding future taxonomy revisions, not meant to be
statistically summarized itself).

## 3. Embedding analysis (no new GPU inference)

**[A] real environment fact, checked before writing this section**:
`torch`, `numpy`, and `matplotlib` are available in this environment;
`scikit-learn` and `umap-learn` are **not currently installed**. This
matters concretely for which techniques are "free" (usable today, zero
new dependency) versus which require an install (a real action, needing
explicit approval, not silently added as a side effect of "just
research"):

| Technique | Needs a new dependency? | Notes |
|---|---|---|
| PCA | **No** — `torch.pca_lowrank()` (already available via installed `torch`) or a hand-rolled NumPy SVD both work with zero new installs | Cheapest, most interpretable first pass |
| t-SNE | **Yes** — standard implementation is `sklearn.manifold.TSNE`, not installed here | Would need `pip install scikit-learn` (or a from-scratch implementation, not worth it given scikit-learn is a small, standard, non-GPU dependency) |
| UMAP | **Yes** — `umap-learn`, not installed | Same install-approval consideration as t-SNE |

**What each could reveal, applied to the already-cached 5-location ×
train/val/test feature set**:

- **PCA** (**[B]** cheapest, most defensible first step, matching this
  document's own "cheapest useful experiment first" pattern already used
  for the linear-probe-before-MLP decision): projecting each location's
  features to 2-3 components and coloring by (a) true class, (b)
  correct/incorrect — reveals whether the 8 classes are linearly
  separated in a way roughly consistent with the probe's own linear
  decision boundary (since the probe **is** linear, PCA is a genuinely
  apt visualization here, not just a generic dimensionality-reduction
  habit) and whether misclassified points sit near class boundaries
  (expected if `ambiguous_taxonomy`/`requires_reasoning` dominates) or
  scattered as isolated outliers far from any class cluster (expected if
  `image_quality`/`missing_visual_info` dominates).
- **t-SNE / UMAP** (**[C]**, pending install approval): both preserve
  local neighborhood structure better than PCA at the cost of global
  distance meaning — useful specifically for the "are failures isolated
  outliers vs. clustered together" question, since PCA's linear
  projection can make genuinely well-separated non-linear clusters look
  falsely overlapping. UMAP is generally faster and better at preserving
  some global structure than t-SNE for this size of dataset (~2200
  images) — **[C]** a reasonable default preference if only one
  non-linear method is added, not a strong claim.
- **Specific questions this maps to**:
  - *Do failures cluster together?* — color failures distinctly in the
    2D projection; a tight failure cluster suggests a systematic cause
    (one bad sub-source, one confusable class pair); scattered failures
    suggest per-image idiosyncratic causes (image quality, individual
    mislabels).
  - *Are specific taxonomy buckets overlapping?* — check whether
    `mixed_text_image`/`printed_document` (the two classes most confused
    with each other in every confusion matrix this session produced)
    show genuine embedding overlap or whether the overlap is purely a
    classifier-boundary artifact despite well-separated embeddings (the
    latter would argue the vision representation is fine and the
    boundary/head is the issue — directly informs section 6).
  - *Are failures isolated outliers?* — distance from each failure point
    to its true class's centroid, compared to the distribution of
    correct-prediction distances for that class; an outlier failure (far
    from its own true class's cluster) is a different story than a
    failure sitting squarely inside a class cluster but on the wrong side
    of a nearby different class's boundary.
  - *Do certain document types naturally separate?* — visual inspection
    of the 2D projection's cluster shapes per class; classes that form
    tight, well-isolated clusters are unlikely to be contributing to the
    8% failure rate at all regardless of head architecture, narrowing
    where a future fine-tune (if pursued) should focus.

## 4. Confusion analysis

**[A]** Per-class confusion matrices already exist for every probe run
this session (Stage 1, Stage 2 × 5 locations, multi-seed × 5 locations ×
10 seeds) — this section proposes *aggregating* what's already been
produced rather than re-computing from scratch.

- **Per-class confusion**: already computed per run; aggregate across the
  10 multi-seed runs per location (not just Stage 2's single-seed
  matrices) to get a *stable* confusion pattern instead of one seed's
  noise — directly motivated by the same lesson that produced the
  multi-seed experiment in the first place (single-run decoder_final
  wobbled 90.1%→92.5%; a single-run confusion matrix is exactly as
  unreliable for the same reason).
- **Most common confusions**: from this session's own Stage 1/2 matrices,
  the recurring pattern is `printed_document` ↔ `dense_tabular_rows`/
  `handwritten_ledger`/`mixed_text_image` — **[A]** directly visible in
  every confusion matrix produced so far, not a new finding, but worth
  aggregating into one clean ranked list (confusion pair, count, % of
  total errors) rather than reading it back out of five separate text
  reports by eye.
- **Confusion directionality**: does `handwritten_ledger` get
  misclassified as `printed_document` more often than the reverse? **[A]**
  Stage 1's test confusion matrix shows exactly this asymmetry already
  (handwritten_ledger→printed_document: 1 case; printed_document→
  handwritten_ledger: 8 cases) — **[B]** consistent with `printed_document`
  being the largest, most visually diverse class (175 of 332 test images)
  acting as a "gravity well" default the probe falls back to under
  uncertainty, rather than a symmetric confusion between two similarly-
  defined categories. This directional asymmetry is itself evidence
  worth checking against class *base rate* directly (a naive "always
  predict the majority class" baseline would show exactly this kind of
  asymmetric error pattern) before attributing it to any deeper cause.
- **Confidence of incorrect predictions**: **[A]** already measured —
  Stage 1: correct preds mean confidence 0.986, wrong preds mean 0.808.
  **[B]** A wrong-but-confident prediction (softmax mass concentrated on
  the wrong class) is a stronger signal of a genuine representation gap
  than a wrong-but-uncertain one (softmax mass split near-evenly between
  true and predicted classes) — the aggregate mean (0.808) hides this
  distinction; the per-image review in section 2 should specifically flag
  cases where wrong-prediction confidence is *high* (e.g. >0.9) as the
  most interesting/concerning subset, since those are the cases least
  explicable by "reasonable ambiguity" and most likely to reflect either
  `label_error` or a genuine representation gap.
- **Random vs. systematic**: **[B]** the directional asymmetry and
  repeated confusion pairs across independent seeds (not just one run)
  is itself the test — if the same class pairs show up as the dominant
  confusion across all 10 seeds and multiple locations, that's systematic
  (a real, stable boundary problem); if the specific confused pairs vary
  seed-to-seed with only the aggregate error *rate* staying stable,
  that's closer to random noise on genuinely hard/borderline individual
  images. This project's own data already leans toward "systematic" — the
  same `printed_document`-involving confusions appear in every single
  report produced this session — but this should be confirmed by
  actually tabulating agreement across all 10 seeds, not asserted from
  eyeballing a few reports.

## 5. Label quality audit

**[B]** The explicit instruction not to assume labels are wrong is
important — the right framing is "which failures are *worth checking*
for a label problem," not "assume the model is right and the label is
wrong."

- **Questionable labels**: flag test images where the probe is
  wrong **and** confident (per section 4's high-confidence-wrong flag)
  **and** the confusion is *consistent across all 5 hook locations and
  all 10 seeds* (i.e., every independent measurement this session
  produced agrees on a different class than the label) — this is the
  strongest, most defensible signal for "worth a human second look,"
  precisely because it can't be explained by training noise (ruled out
  by the multi-seed/multi-location agreement) or by the specific vision
  representation being deficient (ruled out by the same class being
  called across *every* representation, including the ones that see
  fundamentally different information — raw vision-encoder tokens vs.
  a decoder layer that's already mixed in the prompt's text).
- **Inconsistent taxonomy usage**: **[C]** would need comparing multiple
  human-labeled examples of the same true class for visual consistency
  (e.g. do all `dense_tabular_rows` labels actually share the structural
  properties the taxonomy's `classifier_guidance` text describes?) —
  this is closer to an audit of the *labeling process* than of individual
  images, and per [[project_new_taxonomy_categories_guidance_deferred]]
  this project already knows some taxonomy categories have thin/absent
  guidance — worth cross-referencing which failure-prone classes also
  have weak `classifier_guidance` text in `core/taxonomy.py`, since a
  category the humans themselves had unclear written guidance for is a
  plausible source of labeling inconsistency, not just model error.
- **Mislabeled training images**: **[C]** the same "wrong + confident +
  consistent across every representation" test from above, but applied to
  the *training* split. Not evaluated by this session's probes (which
  only measured test-split accuracy) — would need a specific pass
  re-running the frozen probe's predictions against the train split too
  (cheap: reuses cached train features, no new inference) to see whether
  training labels show the same "every representation disagrees with the
  label" pattern anywhere.
- **Taxonomy definitions that overlap**: **[B]** directly testable via
  the confusion-pair analysis in section 4 combined with a manual read of
  each confused pair's `classifier_guidance` text — if two categories'
  written definitions genuinely describe overlapping visual criteria
  (not just "the model sometimes confuses them" but "a careful human
  reading only the definitions couldn't reliably assign every real
  example to one or the other"), that's a taxonomy problem, distinct
  from either a vision or reasoning limitation, and no amount of
  fine-tuning or better reasoning fixes it — only a taxonomy edit does.
  **`printed_document` vs. `dense_tabular_rows`/`handwritten_ledger`** is
  the strongest candidate to check first given this session's own
  confusion data.

## 6. Vision vs. reasoning — the key question, objective criteria

**[B]** The prompt's own framing is exactly right and maps directly onto
data this session already has: "information is absent from the image
representation" (vision limitation) vs. "representation appears
sufficient but needs higher-level interpretation" (reasoning limitation).
Three complementary, objective (not purely subjective) tests, each usable
without new GPU inference:

**Test 1 — cross-location agreement.** If a failure is wrong identically
across *all 5* hook locations (vision_encoder through decoder_final) —
including `vision_encoder`, which has had **zero** exposure to the
prompt/text/reasoning pathway at all, it's purely raw visual patch
features — that's strong evidence the information genuinely isn't
recoverable from the image alone by *any* representation this pipeline
produces, i.e. a real candidate for `missing_visual_info`. Conversely, if
`decoder_final`/`decoder_mid` get it right while `vision_encoder` gets it
wrong, that's evidence the *decoder's* processing (which has access to
the specific prompt text and more contextualized computation) recovered
something the raw vision features didn't have — a `requires_reasoning`
signal, or at minimum evidence that whatever the decoder adds is doing
real, non-trivial work rather than pure reformatting (directly answering
section 7's own question).

**Test 2 — full-generation cross-check.** For each hidden-state-probe
failure, check whether the full autoregressive Gemma classifier
(the ~94% generation-based `classify()` path) gets that *same* image
right. **[A]** the predictions for exactly this comparison already exist
on disk from `diagnostics/run_gemma_flat8_on_benchmark_holdout.py`'s
saved `gemma_flat8_test_predictions.json` (same held-out split, same
taxonomy) — this is a zero-new-inference, pure data-join analysis. If
full generation also gets it wrong, that's evidence the difficulty isn't
specific to the frozen-hidden-state approach at all (a genuinely hard/
ambiguous image, possibly `label_error` or `ambiguous_taxonomy`, not a
vision-vs-reasoning question specific to this experiment). If full
generation gets it *right* where every hidden-state probe gets it wrong,
that's the cleanest possible `requires_reasoning` signal available in
this dataset — it directly demonstrates that seeing the model's full
autoregressive reasoning process (not just any single hidden state)
recovers information no single-forward-pass hidden state captured.

**Test 3 — softmax margin at the true class.** For each failure, look at
where the true class ranked in the probe's own softmax distribution (not
just top-1). A failure where the true class was the probe's *second*
choice with a close margin (e.g. predicted 0.51 vs. true-class 0.43) is
categorically different from one where the true class ranked near the
bottom with negligible softmax mass — the former is consistent with a
representation that's *nearly* sufficient (arguing against a hard vision
limitation, more consistent with `ambiguous_taxonomy`/`requires_reasoning`
or a genuinely close judgment call), the latter is more consistent with
the representation lacking the relevant signal entirely
(`missing_visual_info`).

**[B] Combined decision rule** (each failure gets classified by running
all three tests, not picking one):

| Pattern | Interpretation |
|---|---|
| Wrong on all 5 locations + wrong on full generation too + true class ranks low in softmax | Likely NOT fixable by more vision capacity — check `label_error`/`ambiguous_taxonomy` first |
| Wrong on all 5 locations + full generation gets it RIGHT | Strongest available `requires_reasoning` signal |
| Wrong on vision_encoder but right on decoder_mid/final | Decoder is adding real information beyond reformatting — some reasoning benefit exists even within hidden states, informs section 7 |
| Wrong everywhere but true class is a close second in softmax | Borderline/close call — more consistent with `ambiguous_taxonomy` than a hard limitation of either kind |

This avoids the subjective-judgment trap the prompt explicitly warns
against: every cell in that table is derived from existing or
zero-new-inference data, not a reviewer's gut read of the image.

## 7. Decoder comparison — is the decoder adding information or just reformatting?

**[A]** All 5 locations' features for every image in the held-out split
are already cached (`features_stage2.pt`) — every analysis below is pure
CPU linear algebra on existing tensors, zero new inference.

- **Cosine similarity / distance between locations, per image**:
  compute cosine similarity between e.g. `vision_encoder`'s and
  `decoder_final`'s feature vector for the same image (note: these are
  different dimensionalities — 768 vs. 1536 — so a direct vector
  comparison needs a shared reduced space first, e.g. project both to
  the same PCA basis, or compare *relative* structure rather than raw
  vectors directly — a real methodological detail to get right, not a
  free lunch). More directly comparable: `vision_projected` (1536-dim)
  vs. `decoder_early`/`decoder_mid`/`decoder_final` (also 1536-dim) share
  the same space and can be compared with plain cosine similarity without
  a projection step.
- **Clustering behaviour**: run the same PCA/UMAP projection (section 3)
  independently per location and visually/quantitatively compare cluster
  tightness and class separation — if `decoder_final`'s clusters are
  measurably tighter/more separated than `vision_projected`'s for the
  *same* images, that's evidence of real added value from decoder
  processing; if the cluster geometry looks like a rotated/rescaled
  version of the same structure, that's evidence of reformatting without
  new information.
- **Neighbourhood preservation**: for each image, compare its k-nearest
  neighbors (by cosine distance) in `vision_projected`-space vs.
  `decoder_final`-space — if the same images are each other's neighbors
  in both spaces (high neighborhood overlap), the decoder is preserving
  the vision-tower's similarity structure (reformatting); if neighbor
  sets differ substantially, the decoder has reorganized the
  representation in a way that isn't just a relabeling of the same
  geometry (real transformation, though not necessarily *useful*
  transformation — that still needs the probe-accuracy comparison to
  confirm).
- **Representation drift across layers**: **[B]** plot cosine similarity
  between consecutive/distant layer pairs across the decoder stack for a
  sample of images — a large jump in similarity at some specific depth
  would suggest that's where task-relevant reorganization happens (a
  useful signal for *where*, if a future hidden-state approach wanted a
  location between the 5 already tested, though per the multi-seed
  results this session already found no location beats the others by a
  reliable margin, so this is more explanatory than actionable right now).
- **Can we determine whether the decoder is genuinely adding information
  or merely reformatting?** **[B]** The multi-seed probe-accuracy results
  already partially answer this at the aggregate level — since
  `decoder_final`/`decoder_mid` don't reliably outperform `vision_encoder`
  (91.0-92.3% vs. 92.0%, overlapping), the decoder isn't adding
  *linearly-exploitable* task-relevant information beyond what the raw
  vision tower already has, at least not information a linear probe can
  use. **[C]** It's still possible the decoder reorganizes the
  representation in a way that's more separable *non-linearly* (which a
  linear probe can't see, but the shallow MLP experiment argued against
  this too, at least on `vision_encoder`) — Test 1's per-image
  cross-location disagreement cases (section 6) are the more precise way
  to answer "does the decoder add information," since aggregate accuracy
  parity doesn't rule out the decoder helping on a *different* subset of
  images than the vision tower struggles with, even at equal overall
  accuracy.

## 8. Decision framework

**[B]** Objective, pre-committed thresholds — set *before* running the
analysis, per good practice, though the specific cutoffs below are
proposed starting points, not validated against any prior calibration
(flagged explicitly, matching this session's own standing discipline
around uncalibrated Decision Engine thresholds):

| If the error-analysis finds... | ...then the evidence favors |
|---|---|
| >60-70% of failures classified `missing_visual_info` (vision_encoder-level, confirmed by Test 1's cross-location agreement + Test 2's full-generation-also-wrong) | Vision-tower fine-tuning — per `GEMMA_VISION_TOWER_FINETUNE_RESEARCH.md`, but temper expectations given that doc's own finding that Gemma's frozen tower already starts near ceiling relative to the zero-shot baselines that produced big fine-tune gains elsewhere |
| >50% of failures classified `requires_reasoning` (Test 2: full generation gets it right where hidden-state probes don't) | Full-generation fallback for exactly this subset, NOT vision-tower fine-tuning (fine-tuning the vision tower can't add prompt-conditioned reasoning) — favors Decision Engine fusion (question 8 of the fine-tune doc) routing low-confidence hidden-state predictions to full generation |
| >30% of failures classified `image_quality`/`skew_crop`/`low_contrast`/`preprocessing_artifact` | Preprocessing improvements (Stage A per [[project_preprocessing_analyser_and_dewarp]]) — a pipeline-input fix, not a model-capacity fix at all; would apply equally to full generation and every hidden-state location |
| >20% of failures classified `ambiguous_taxonomy` or show genuine taxonomy-overlap per section 5's audit | Taxonomy refinement — no amount of additional training/fine-tuning fixes a boundary that's genuinely undefined; needs a `classifier_guidance`/category-definition edit |
| A meaningful fraction (no fixed threshold proposed — needs the actual audit) pass the section-5 "wrong + confident + consistent across every representation" label-quality test | Manual label review for that specific subset, not a modeling change at all |
| Failures spread roughly evenly across causes, no single category dominates | Decision Engine fusion (multiple independent sensors covering each other's blind spots) is the most robust response — matches this project's own standing philosophy ([[feedback_reduce_manual_touches_is_the_metric]]-adjacent: no single stage needs to be perfect if fusion covers the gaps) more than betting on any single fix |

**[C]** These percentage thresholds are proposed defaults, not measured —
the actual distribution across ~26-27 failures (with `rare_class` support
this thin, e.g. some categories at n=1) means percentages will be noisy
at this sample size; the review in section 2 should report raw counts
alongside percentages, and any decision this framework produces should be
treated as directional, not a hard statistical verdict, given how few
failure examples exist to classify in the first place.

---

## Summary

This methodology is designed so every step reuses data already produced
this session — cached embeddings, existing confusion matrices, saved
generation predictions — and needs **zero new GPU inference** except
possibly `pip install`-ing scikit-learn/umap-learn for the non-PCA
embedding visualizations (a real action requiring approval, flagged
explicitly in section 3, not assumed). The core deliverable is a ~26-27-
image manual review (section 2) cross-referenced against three objective,
already-computable tests (section 6) — small enough to actually complete,
concrete enough to produce a real answer to "is this a vision problem,"
before any further GPU compute is spent on the fine-tune proposal this
error analysis exists to gate.
