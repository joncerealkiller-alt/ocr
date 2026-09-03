# Gemma Hidden-State Routing Head — Research & Architecture (2026-08-07)

Status: **Design document only. No code written, no `GemmaLoader` changes.**
Answers the research questions as posed; ends with a staged benchmark plan.
Do not treat any stage below as authorized to start — it's a plan to review,
not a queue to run.

## Relationship to prior work

This is a different proposal from `docs/GEMMA_HIERARCHICAL_ROUTING_INVESTIGATION.md`
(parked 2026-08-05). That doc investigated reusing one Gemma vision
*encoding* across multiple narrow *generate()* calls (still autoregressive,
still text output, just amortizing the vision tower). This document
investigates skipping generation entirely — reading a hidden state once and
classifying it with a trained head, never calling `generate()`. The two are
not mutually exclusive but solve different bottlenecks; that doc's two
confirmed Gemma4-internals gotchas (`.pooler_output` vs `.last_hidden_state`
shape mismatch, and the 127GB OOM from `inputs_embeds` without real
`input_ids`) are only relevant here if a future stage does a manual forward
pass with spliced embeddings — a single vision-only or single-forward-pass
hook (this proposal's actual mechanism) doesn't touch the decode loop at
all, so those gotchas don't apply directly, but are worth knowing exist.

---

## 1. Which internal tensors are realistically available

`core/loaders/gemma_loader.py` already defines and uses three named vision
hook points (used today only for vision-tower diagnostics/logging, not
routing):

| Name | Path | What it is |
|---|---|---|
| `encoder` | `model.model.vision_tower.encoder` | Pre-pooling patch tokens, vision-hidden-dim |
| `pooled` | `model.model.vision_tower` | Post-pool, post-standardize vision embedding |
| `projected` | `model.model.embed_vision` | Vision features projected into LM embedding space (text-hidden-dim) |

Confirmed shapes/semantics from the parked investigation (reading the
installed `transformers` 5.12.1 `Gemma4Model` source directly, not assumed):
`get_image_features()` returns `.last_hidden_state` (768-dim, pre-projection
vision-tower output) and `.pooler_output` (1536-dim, LM-space-projected —
this is what `embed_vision`/`projected` corresponds to). Treat these two
numbers as this project's confirmed dims for that model/transformers
version; re-verify against whichever `gemma.yaml` variant is actually in
use if the exact model changes.

Beyond the three already-wired vision hooks, decoder hidden states are also
realistically available via **`output_hidden_states=True`** on a `forward()`
call (or `model.generate(..., output_hidden_states=True, return_dict_in_generate=True)`
for the *first* decode step) — this returns a tuple of one tensor per
decoder layer, each `(batch, seq_len, text_hidden_dim)`. None of this is
currently captured anywhere in `core/loaders/gemma_loader.py`; it would be
a new hook, not a rewiring of the existing three.

Practical accessibility ranking, cheapest to most invasive:
1. **`projected`/`pooled`/`encoder`** — already instrumented, zero new code
   needed to observe, just needs to persist instead of discard.
2. **Final decoder hidden state on one forward pass** — one extra
   `output_hidden_states=True` kwarg on a single `model(**inputs)` call, no
   `generate()` loop involved at all. Cheapest new hook to add.
3. **Early/middle decoder layers** — same call, just indexing a different
   position in the returned tuple. No extra compute over (2) since all
   layers are already computed by the same forward pass — the "cost" is
   purely which index you keep, not extra inference.
4. **Hidden state during multi-step `generate()`** — much more invasive:
   requires the same `is_first_iteration` decode-loop internals documented
   in the parked doc, only relevant if the head needs to see hidden states
   *after* several generated tokens rather than after the prompt-only
   prefill. Avoid unless a cheaper hook is proven insufficient — pulls in
   the same brittle internals that made the parked hierarchical-prompt work
   slower than baseline.

**Recommendation for research questions 1-2**: start from a single
`forward()` pass with `output_hidden_states=True`, prompt = whatever
`core/classifier.py` sends today minus the generation step. This gets every
decoder layer's hidden state at the prefill position in one call, at the
cost of one forward pass (no autoregressive loop) — the single cheapest way
to get comparative data across hook locations A-E.

## 2. Hook location comparison

| Hook | Visual richness | Semantic richness | Latency | Implementation complexity |
|---|---|---|---|---|
| A. Vision encoder output (`encoder`) | Highest — raw patch tokens, no projection loss | Lowest — no text-grounding at all | Lowest (vision tower only, no LM forward) | Low — already exposed |
| B. Projection layer output (`projected`) | High, but compressed into LM embedding space | Low-medium — positioned for the LM but not yet contextualized by it | Low (vision tower + one linear projection) | Low — already exposed |
| C. Early decoder layers | Diminishing visual fidelity as layers mix tokens | Low-medium — some cross-token attention, minimal task-specific abstraction yet | Medium (partial decoder forward) | Medium — new hook |
| D. Middle decoder layers | Further diminished, but this is where classification-relevant abstraction has empirically tended to concentrate in comparable transformer literature | Medium-high | Medium-high | Medium — new hook |
| E. Final decoder hidden state (pre-`lm_head`) | Most compressed/abstracted — visual detail is now folded into whatever concept the model is about to verbalize | Highest — this is the representation the text decision was actually generated from | Highest of the single-forward-pass options (full decoder stack) but still far cheaper than full autoregressive decode | Low-medium — new hook, straightforward |

Important framing per the explicit constraint: **do not assume E is
automatically best.** The counter-risk is real — Gemma's final layer is
shaped by next-token prediction pressure, not by an explicit classification
objective, so it may encode "what word comes next" rather than "what
category is this," and could be *more* entangled with the specific prompt
phrasing than a middle layer. Middle layers (D) are the standard place probes
find the most linearly-separable task-relevant structure in comparable
literature (an *expectation* going in, not something measured on this
model yet — treat it as a hypothesis Stage 2 exists to test, not a
conclusion). Vision-only hooks (A/B) skip the text/prompt machinery
entirely, which removes a large source of prompt-sensitivity but also
removes whatever signal Gemma's language modeling contributes to routing —
and the current classifier's 94% figure is presumably not achievable from
vision features alone, since e.g. distinguishing structurally-similar
document types may depend on prompt-conditioned reasoning, not just pixels.

## 3. Starting architecture: linear probe vs. shallow MLP vs. hierarchical head

**Start with a linear probe.** Justification:

- It directly answers the research question ("is routing information
  *linearly* separable") rather than answering a harder question ("can
  *some* function of these features separate routing classes") that would
  leave the original question unresolved if it works.
- If a linear probe on layer X gets meaningfully above chance and
  meaningfully close to the 94% ceiling, that's strong evidence the
  representation itself is doing the work, not the classifier head — the
  cheapest, most interpretable, fastest-to-train option.
- If a linear probe fails but a shallow MLP succeeds, that's evidence the
  information exists but is not linearly accessible — a different, weaker,
  still-useful finding, but only worth funding *after* the cheap check
  rules out the strong version.
- A hierarchical head should not be the first experiment at all — see
  question 4. Starting there would conflate "does hidden-state info exist"
  with "does hierarchical decomposition help," two separable questions.

Shallow MLP is the natural Stage 3 escalation only if the linear probe
underperforms; it should reuse the exact same frozen features and train/eval
split as the linear probe so the comparison isolates head capacity, not
data or feature differences.

## 4. Hierarchical classification — critical evaluation

Do not assume hierarchical is correct here either, and note the direct
project precedent for skepticism: the parked hierarchical-prompt investigation
(`docs/GEMMA_HIERARCHICAL_ROUTING_INVESTIGATION.md`) already built
per-level abstention specifically *because* irreversible early branching
was known to be dangerous, and even with that safeguard the whole approach
underperformed the flat single-prompt classifier at the realistic operating
point. A hidden-state hierarchical head inherits the same structural risk
(wrong early branch forecloses the correct leaf) without inheriting that
prior work's mitigation unless it's deliberately rebuilt.

Alternatives, ranked by how directly they avoid that failure mode:

- **Flat classifier** (single head, full class set): simplest, no
  compounding-error risk, but may struggle if some classes are only
  separable via features that get diluted by having to distinguish them
  from all other classes simultaneously. Right choice if Stage 1 already
  shows layer info is close to linearly complete — no need to buy
  hierarchical complexity for a problem the flat version already solves.
- **Shared backbone with independent heads** (one frozen feature vector,
  multiple parallel linear/MLP heads, each predicting one facet — e.g. one
  head for processing-family, one for document-layout, no head's output
  gates another): avoids irreversible gating entirely since all heads see
  the same evidence independently; the Decision Engine can fuse them the
  same way it already fuses tower evidence (`weighted_fusion` policy,
  `core/routing_decision_engine.py`). This is the most consistent with the
  project's existing "evidence, not gating" philosophy and question 8's
  integration ask.
- **Multi-task learning** (one backbone, multiple correlated losses trained
  jointly): plausible extension of the above if per-facet heads turn out to
  share useful structure, but adds training complexity for a benefit that's
  speculative until independent heads are tried first.
- **Soft hierarchy** (hierarchical structure retained, but each level
  outputs a distribution, not a hard branch, propagated downward as
  weights rather than a discrete decision): keeps whatever efficiency
  hierarchy offers (narrower per-level class sets) while formally avoiding
  the "wrong branch forecloses the leaf" failure — but this is real
  additional complexity (probability propagation across levels, calibration
  cross-level) and should only be pursued if a flat/independent-heads
  approach is measurably not good enough on some genuinely hierarchical
  class group.
- **Top-k routing** (keep top-k branches alive per level instead of one):
  reduces but doesn't eliminate the foreclosure risk, and multiplies
  downstream evaluation cost by k. Weakest of the group here — it's a
  patch on hierarchical's known failure mode rather than an architecture
  that avoids it.

**Recommendation**: treat hierarchical variants as something Stage 4 tests
empirically against a flat baseline (Stage 3's output), not something
assumed superior going in — consistent with the explicit constraint and
with this project's own prior finding that hierarchical underperformed once
already in a closely related setting.

## 5. Training strategy (frozen backbone, head-only training)

- **Feature caching**: since the backbone is frozen and the hook point is a
  single forward pass (not multi-step generation), features for the entire
  training/eval corpus can be computed once and cached to disk (e.g. one
  `.pt`/`.npy` per image per hook location, mirroring the existing
  per-image `.pt` capture pattern already used for vision-hook diagnostics
  in `gemma_loader.py`). This turns every subsequent head-architecture
  experiment into a fast CPU-only training loop over cached tensors — no
  GPU/model reload needed for Stages 1-4 once caching is done, and directly
  respects the CUDA-usage discipline in `CLAUDE.md` (no risk of a second
  concurrent inference process during head-iteration work).
- **Pooled representations**: decoder hidden states are per-token
  `(seq_len, hidden_dim)`; a classification head needs a fixed-size vector.
  Options: last-token position (natural for a decoder-only LM, matches
  "what the model was about to say"), mean-pool over the prompt tokens, or
  a dedicated pooling token if the prompt template reserves one. Last-token
  is the cheapest, most defensible default for Stage 1 — matches how
  `lm_head` itself reads the state to produce the next token.
- **Normalization**: standardize (zero-mean, unit-variance per-dimension)
  computed from the training split only, applied at both train and
  inference time — standard practice for linear probes, and cheap to get
  right or catastrophically easy to leak (never fit normalization stats on
  eval data).
- **Dimensionality reduction**: not needed for a linear probe (a linear
  layer already handles high-dimensional input directly and its weight
  vector *is* the interpretability artifact — reducing dimensions first
  would obscure exactly the signal the experiment is trying to observe).
  Worth revisiting only if the MLP stage shows evidence of overfitting on a
  small labeled set.
- **Loss function**: standard cross-entropy over the class set for a flat
  head; per-head cross-entropy per facet for independent heads (question 4).
  No need for anything more exotic at this stage.
- **Calibration**: do not assume the head's softmax output is calibrated
  confidence — the project has already measured this exact failure mode
  for Gemma's *own* self-reported confidence field ("repeatedly measured
  unreliable" per `core/routing_decision_engine.py`'s design notes) and for
  pre-generation yes/no token probabilities per this task's own background.
  A hidden-state head's raw softmax score should be treated as
  `score_kind="softmax_probability"` (an existing category in
  `DecisionConfig.score_semantics`) and calibrated against real measured
  accuracy before any AUTO_ACCEPT threshold is set — exactly the same
  calibration debt the Decision Engine already tracks explicitly for every
  other sensor, not a new problem this proposal introduces.

## 6. Inference strategy — what compute is actually removed

Do not invent timing numbers — reason from what computation exists in each
path instead:

**image → Gemma forward pass → hidden-state head**: one vision-tower pass +
one decoder forward pass (prefill only, processing the full prompt+image
token sequence once) + one small head (linear or shallow MLP) evaluation.

**image → Gemma → autoregressive decoding**: the same vision-tower pass +
prefill forward pass, **plus** N additional decoder forward passes, one per
generated output token (KV-cache-accelerated, so each is cheaper than the
prefill step, but still N sequential passes that cannot be parallelized
across tokens — each depends on the previous token's output).

What's removed by the hidden-state approach: every decode-step forward pass
after the prefill, i.e. all computation proportional to output sequence
length. What's *not* removed: the vision tower pass and the single prefill
forward pass — both approaches pay that cost identically, since the hook
point sits inside/after that same pass. The actual saving is therefore
proportional to how many tokens the current classifier prompt currently
generates (structured output — likely a short JSON/label string, not a long
free-text response) — this needs to be measured (count actual generated
tokens per call in the current pipeline), not assumed, before sizing the
expected win. If the current output is already very short (a few tokens),
the achievable saving is bounded and should be measured before this
proposal is prioritized over other latency work.

## 7. Staged benchmark plan

Each stage should reuse a single fixed, held-out labeled evaluation split
(not the training corpus) so results are comparable across stages — reuse
whatever existing split `core/routing_decision_engine.py`'s replay
diagnostics use if one already exists with sufficient labels, rather than
carving a new one, since [[project_no_comprehensive_ground_truth]] means
labeled data is scarce and every existing labeled split is valuable.

### Stage 1 — Linear probe, one hook location
- **Hypothesis**: routing-relevant information is linearly present at at
  least one accessible hidden-state location.
- **Location**: single most defensible default — final decoder hidden
  state, last-token position, prefill-only forward pass (cheapest to
  implement, and the most semantically "downstream" of the vision+prompt
  fusion without requiring generation).
- **Success criteria**: probe accuracy meaningfully above chance and worth
  continuing to invest in — the concrete bar should be set relative to the
  ~59.5% pre-generation-token-probability baseline already measured (this
  approach should clearly beat that, since it's a strictly richer signal)
  and read as a *fraction of the gap* to the 94% generation ceiling, not
  required to match 94% outright at this stage.
- **Metrics**: accuracy, per-class precision/recall (class imbalance is
  likely given the taxonomy), confusion matrix against the same classes
  the current classifier uses.
- **Risks**: single labeled eval split may be too small/biased to trust a
  single-point accuracy number — report a confidence interval or n, not a
  bare percentage. Last-token pooling choice may be suboptimal and get
  blamed as "hidden states don't work" when it's a pooling artifact.
- **Expected outcome**: genuinely uncertain — this is exactly the question
  being asked. A plausible failure mode worth naming going in: the final
  layer is optimized for next-token prediction of the *specific prompt's*
  expected continuation, not a clean class signal, so it may underperform
  a middle layer (motivating Stage 2 regardless of Stage 1's result).

### Stage 2 — Compare multiple hook locations
- **Hypothesis**: different hook locations trade off differently, and the
  best one is not assumed in advance (per explicit constraint).
- **Locations**: A (vision encoder), B (projected), C (early decoder), D
  (middle decoder), E (final decoder) — reuse Stage 1's cached-feature
  infrastructure, just cache additional locations from the same forward
  passes (no extra inference cost — all layers are already computed).
  Since caching all layers costs nothing extra over caching one, this stage
  is cheap; do not skip it in a rush to "pick one and move on."
  Same linear probe, same eval split, same metrics as Stage 1.
- **Success criteria**: identify whether any location clearly dominates,
  or whether results are close enough that other factors (implementation
  simplicity, latency) should decide.
  **Metrics**: same as Stage 1, per location, plus feature dimensionality
  (affects head parameter count / overfitting risk on a small labeled set).
- **Risks**: middle-layer indexing conventions vary by model family — get
  the actual layer count and indexing confirmed against the installed
  `transformers` Gemma4 source (per the parked doc's precedent of verifying
  internals directly rather than assuming).
- **Expected outcome**: hypothesis going in (to be falsified, not assumed):
  middle layers outperform both extremes, but Stage 1's risk note above
  means this needs to actually be checked.

### Stage 3 — Replace linear probe with shallow MLP
- **Hypothesis**: if Stage 1/2's best linear probe underperforms the 94%
  ceiling by a meaningful margin, some of that gap is due to head capacity
  (non-linear separability) rather than missing information.
- **Only run if Stage 1/2 doesn't already close the gap** — an MLP that
  merely matches a linear probe's accuracy is not informative and wastes
  the calibration/overfitting risk it introduces for nothing.
- **Success criteria**: MLP measurably closes the gap to 94% beyond the
  linear probe's result, on the same eval split.
- **Metrics**: same as above, plus overfitting checks (train vs. eval gap)
  given a frozen-backbone MLP has more capacity to memorize a small
  labeled set.
- **Risks**: small labeled corpus (per [[project_manual_review_scale]] —
  tens of pages per manifest scale) may not support a larger head without
  overfitting; needs explicit train/eval split discipline, possibly k-fold
  given scarce labels.
- **Expected outcome**: uncertain; the value of this stage is precisely in
  distinguishing "linear ceiling" from "information ceiling."

### Stage 4 — Evaluate hierarchical variants
- **Hypothesis**: per question 4, hierarchical is not assumed superior —
  this stage exists to test it against the flat/independent-heads
  baseline from Stage 3, not to justify a foregone conclusion.
- **Compare**: flat classifier (Stage 3's result) vs. shared-backbone
  independent heads vs. (only if independent heads show a real facet-
  specific weakness) soft hierarchy.
- **Success criteria**: hierarchical/independent-heads variant must beat
  the flat baseline by a margin that justifies its added complexity —
  matching, not beating, is a loss for hierarchical given complexity cost.
- **Metrics**: same as above, plus per-facet accuracy for the independent-
  heads variant (does splitting into facets recover cases the flat
  classifier's single objective muddles?).
- **Risks**: this is exactly the failure mode already observed once in the
  parked hierarchical-prompt investigation — a real chance of confirming
  the same negative result for a different reason (hidden-state gating
  instead of prompt gating). That would still be a useful, real finding,
  not a wasted stage.
- **Expected outcome**: given the project's prior negative result on a
  structurally similar hierarchical idea, mild expectation that a flat or
  independent-heads approach wins, but this is explicitly the stage meant
  to check that rather than assume it.

### Stage 5 — Compare against production Gemma routing
- **Hypothesis**: the best head from Stages 1-4, at whatever accuracy it
  reached, offers a real latency/accuracy tradeoff worth considering
  against the current 94%-accuracy generation-based classifier.
- **Success criteria**: not "beats 94%" — a head that reaches, say, 88-90%
  at a fraction of the latency may already be a legitimate production
  option for one tier of a multi-sensor Decision Engine (see question 8),
  not a strict win/lose against the incumbent.
- **Metrics**: accuracy delta vs. current classifier on the *same* eval
  images (not a different split — needs a true head-to-head), measured
  latency (real measurement, not estimated, addressing question 6's "don't
  invent numbers" constraint retroactively with real data), and — if this
  head is to be treated as Decision Engine evidence — its calibration
  curve (per question 5's calibration note).
- **Risks**: the current classifier's 94% figure needs to be confirmed as
  measured on a comparable/reproducible split, not assumed transferable to
  this stage's eval set.
- **Expected outcome**: genuinely the deciding stage — determines whether
  this becomes a new Decision Engine sensor (question 8) or a parked
  finding like the hierarchical-prompt investigation.

## 8. Integration with the existing Decision Engine

Per the explicit framing: **this is another sensor, not a Gemma
replacement.** `core/routing_decision_engine.py` already has the exact
shape for this — `EvidenceRecord(sensor_name=..., predicted_class=...,
raw_score=..., normalized_score=..., metadata={"score_kind": ...})`, fed
into `decide(evidence, config, policy=...)` alongside vision-tower and
(eventually) Gemma logit-margin evidence.

Concretely:
- `sensor_name="gemma_hidden_state_head"` (or similar), distinct from
  `"gemma"` (the existing full-generation classifier) — they can coexist as
  two independent evidence sources on the *same* image, and a policy like
  `weighted_fusion` can weigh them, exactly as it already family-weights
  correlated vision towers.
- `raw_score`/`normalized_score` populated from the head's softmax output,
  tagged `score_kind="softmax_probability"` — an existing, already-handled
  category in `DecisionConfig.score_semantics`, needing no new engine code.
- Per question 5's calibration point, this evidence should ship
  `calibrated=False` until measured against real held-out accuracy at each
  confidence level — same honesty standard the engine already applies to
  Gemma's self-reported confidence and the uncalibrated policy thresholds
  documented in `docs/CODE_MAP.md`'s Decision Engine entry.
- Because the engine already supports N independent sensors voting/
  weighing rather than one sensor gating another, a hidden-state head
  naturally sidesteps the irreversible-hierarchical-gating failure mode
  from question 4 at the *system* level even if its own internal
  architecture (question 4) turns out flat — the Decision Engine, not the
  head, is what already knows how to combine multiple partial-confidence
  signals without any one of them foreclosing the final decision.
- A realistic operating mode once measured: use the hidden-state head as a
  fast first pass, and only fall back to full Gemma generation (or route to
  `ACCEPT_WITH_CAUTION`/`QUARANTINE`) when the head's confidence is low —
  this is a policy-level decision for after Stage 5's numbers exist, not
  something to design now.

---

## Summary recommendation

Lowest-risk, highest-information first step: **Stage 1 alone** — one linear
probe, one hook location (final decoder hidden state, last-token pooling),
reusing cached forward-pass features, measured against the existing 59.5%
and 94% reference points already established. It answers the core
feasibility question with the least implementation surface, and every
later stage is conditioned on what it finds rather than committed to in
advance.
