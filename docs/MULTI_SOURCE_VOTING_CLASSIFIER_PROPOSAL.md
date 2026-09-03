# Evidence Fusion Classifier — Research & Proposal (2026-08-06)

Status: **research and planning only, not implemented.** Written after
Jon's observation that Stage 5 routing currently trusts Gemma's single
output as final, despite this project already gathering substantial
independent classification-relevant evidence that never feeds back into
the decision. Do not implement any of this without a further explicit
go - see "What's needed before this could actually be implemented" below
for the real gaps first.

**Revision history** (kept, not overwritten, since the reasoning behind
each correction matters for anyone extending this later): (1) initial
draft framed this as a "voting system" across roughly-equal sources;
(2) revised after a second-opinion review into an evidence-FUSION frame
- correlated tower families instead of independent votes, heterogeneous
native-format evidence instead of forced categorical votes, DocLayout-
YOLO demoted, the "defer to Gemma on split towers" rule retracted as
unvalidated, and a Gemma-escalation-sensor idea added with a concrete
counter-example attached; (3) additional uncatalogued evidence sources
found in the codebase (image_analysis.py's full field set, pipeline_
db.py metadata, auto_sidecar.py's structural detectors), plus a concrete
plan for expanding the human-validation UI's scope without slowing
normal labeling; (4) this pass - the 56-disagreement-case experiment
actually RUN (real results: Gemma beats tower consensus 2x on this set,
MAE confirmed zero unique value, a real map/portrait failure pattern
found) and the row/column-anchor research declared substantially
complete by Jon (remaining work is now engineering/integration, not
open research questions); (5) this pass - logit-based confidence tested
as a replacement for the broken self-reported confidence field, against
two independent gates (n=48 each): raw logit gap at the decision token
is real signal (wrong-case gaps are dramatically lower than the bulk of
correct answers on both runs), but the first run's clean zero-overlap
separation did NOT replicate on the second gate - revised conclusion:
this is an UNCERTAINTY flag (low gap = the model was genuinely torn,
whether it then landed right or wrong), not a strict correctness
predictor; (6) this pass - found `data/outputs/reference_pipeline_v2/
manifest_final.csv`, a real pre-automation whole-corpus human-reviewed
ground truth set (1975 rows, predates subtypes), used it to re-run the
logit-confidence test at real scale (n=368, 168 real positives) - the
larger sample REVISES (6) again: the signal is directionally real
(medians clearly differ) but not cleanly separable at this scale -
several errors were confidently wrong, not just uncertain. Also found a
separate, real, one-directional model bias on `dense_tabular_rows`
specifically (100% precision, 75% recall, all errors are false
negatives) - see the dedicated section below; (7) this pass - tested and
REJECTED two alternative hypotheses about where the "real" decision
happens: earlier output tokens don't secretly decide (confirmed via full
per-step trajectory dump), and neither does the model's state before
generation starts (a first-token reformulation scored WORSE on both
accuracy and confidence-signal strength than the current label-echo
format, and the pre-generation logit "lean" toward yes/no was barely
above chance and directionally biased) - the current gated-prompt
architecture is confirmed correct as built, not just incidentally so;
(8) this pass - tested MobileNetV2 as a candidate new tower (n=169
held-out, real `manifest_final.csv` ground truth): 54.4% overall,
mostly rejected as a general voter (much weaker than the existing 8
towers), but a genuine 100% `website_screenshot` result found - kept as
a candidate cheap pre-filter gate, not a full ensemble member; (9) this
pass - tested ViT-Base/16 ImageNet-21k (`vit_base_patch16_224.orig_
in21k`) with the identical methodology: 54.4% overall, statistically
identical aggregate to MobileNetV2 despite an 86M-param/21k-class model
vs. a 3.4M-param/1k-class one - same failure pattern (portrait_photo
worst, website_screenshot 100%), which reframes the earlier per-model
rejection as a class-of-approach rejection: generic ImageNet-object
pretraining looks like a dead end for this taxonomy regardless of scale,
not a MobileNetV2-specific weakness; (10) this pass - Teklia/pylaia-iam
(handwriting recognizer, misidentified by name as `Teklia/IAM-line`
which is actually the dataset) investigated as a candidate handwriting
sensor and blocked at install: requires `nnutils-pytorch-cuda`, a CUDA
extension with no published wheel on any platform - not evaluated, not
worth a source-build detour given the two ImageNet-tower results above;
(11) this pass - a genuinely different technique tried instead: FINE-
TUNED (not zero-shot) `vit_base_patch16_224.orig_in21k` on this
project's own 8-category taxonomy (manifest_final.csv + 484 hand-
reviewed real LAC census pages added to dense_tabular_rows; lora_dataset
and IAM both excluded as crop-scale mismatches vs. full-page images).
Result: 89.8% on a genuine held-out test set (never touched during
training/checkpoint-selection) - dramatically better than any zero-shot
tower tested so far (54.4% ceiling), confirmed not to be a val-overfit
artifact (test score nearly matches the val score used for selection).
Reframes the fusion-ensemble discussion: a fine-tuned classifier is a
plausible standalone/supplemental evidence source in its own right, not
just another minor voter - real caveats remain (no wholly
external test set, 3 categories still too small to validate); (12) this
pass - ran the ablation Jon requested to separate "fine-tuning helps"
from "more dense_tabular_rows data helps": removed the 484 extra census
images, kept everything else identical. Result: 86.5% overall held-out
(88.0% on dense_tabular_rows specifically) vs. 89.8% (92.8%) with the
extra data, vs. 54.4% (36.7%) zero-shot - fine-tuning itself is
overwhelmingly the dominant driver (+51 points from technique alone),
the extra data contributes a smaller, real, additional gain (+3-5
points) on top. Caveat (3) from entry (11) is now resolved; (13) this
pass - a full controlled multi-architecture benchmark (6 more families:
convnext, mobilenetv2, dinov2, beit, swin, siglip) confirmed the
domain-adaptation finding generalizes (every architecture jumped from a
54-68% zero-shot ceiling to 76-93% fine-tuned) and, more importantly,
ran the complementarity analysis Jon explicitly asked for (pairwise
agreement, error overlap, same-wrong-label overlap, unique-correct
counts on the shared held-out set) rather than just another accuracy
leaderboard. Finding: MobileNetV2 is genuinely complementary despite
lower accuracy (lowest agreement/error-overlap with everyone else, 3
unique-correct cases despite being the weakest model); siglip is the
standout within the high-accuracy cluster (highest accuracy AND
relatively distinct errors vs. convnext/dinov2/beit/swin, which mostly
echo each other's mistakes). Provisional two-sensor pick for further
consideration: siglip + mobilenetv2 - not yet a decision, no fusion
implementation exists; (14) this pass - the "Decision-Engine Sensor
Complementarity Analysis" Jon requested, explicitly framed as sensor
qualification not model selection. Extracted full per-image logit
vectors from the 7 already-trained checkpoints (no retraining), ran a
generic pairwise conditional-disagreement analysis across all 21 pairs,
a detailed siglip<->mobilenetv2 case-level breakdown (8 mobilenet-
rescue cases, 61 siglip-rescue cases, real per-file evidence preserved),
confidence/margin/entropy analysis (margin cleanly separates correct/
incorrect for every architecture, not just the strong ones), a full
codebase inventory of every other implemented vision sensor (DocLayout-
YOLO, YOLOv26-small research variant, image_analysis.py structural
measurements, auto_sidecar.py table/column/row detection, row_
segmentation.py number-anchor detection, the 8 frozen QUALIFIED_
ENCODERS, tower consensus), a Gemma comparator section preserving
distinct behavioral findings from earlier work (NOT re-run on this
split - explicitly flagged as a gap, not silently assumed), a
multi-dimension sensor-value framework (accuracy/complementarity/
redundancy/confidence-usefulness/category-specialization/cost/evidence-
type kept separate, not collapsed into one score), and a hierarchical-
vs-flat-voting discussion grounded in what was actually measured.
Headline findings: MobileNetV2 disagrees with the strong cluster ~3x
more often in every pairwise comparison (22-24% vs. 6-10%), consistent
with a cheap dissent-flag role rather than a peer voter; the 6 strong-
cluster models are largely redundant with each other (91-94% agreement);
the single highest-value next experiment is running Gemma on this
benchmark's actual held-out split, which has never been done - every
existing Gemma-vs-tower comparison in this doc used a different
sample/task. No fusion implemented, per explicit instruction; (15) same
day, later pass - ran that exact experiment. Jon asked whether the
original production classifier prompt would help; confirmed and used
`config/prompts/classifier_classify_v1.txt` (this project's current,
dynamically-taxonomy-templated version, verified via diff against the
sibling `genealogy_pipeline - Main` checkout's older hardcoded variant)
through the real `core/classifier.py` production loader/call path, on
the identical 332-image held-out split. Result: **Gemma scores 93.7%
(311/332) - the highest of every sensor tested in this entire
benchmark**, beating siglip's 92.5%. More importantly: Gemma wins
accuracy-conditional-on-disagreement against every one of the 7 towers
(50-78% vs. towers' 14-39%), has 6 unique-correct images no tower
catches (vs. only 2 combined the other direction), and shows low
same-wrong-label overlap with every tower - genuinely independent
errors, not redundant ones. This materially changes the doc's working
hypothesis: the earlier "Gemma as expensive escalation sensor" framing
(built when the only comparison was against FROZEN towers at ~54-68%)
does not hold up against the FINE-TUNED towers (89.8-92.5%) - Gemma
looks more like it should stay primary, with towers potentially useful
as a dissent check ON Gemma rather than the reverse. No fusion rule
implemented - this remains qualification data, with the next
recommended step being to test whether a tower-disagreement trigger
actually catches any of Gemma's 21 real errors on this split, not to
build a production fusion layer yet; (16) same day, immediate follow-up
- ran exactly that test. Found and separated out a real confound first:
7 of the 21 "errors" are Gemma predicting `casual_photo`, a category
added to the production taxonomy 2026-08-04 that doesn't exist in the
towers'/ground-truth's 8-class label space - not provably wrong, so
reported separately (14 genuine cross-category errors vs. 21 total, not
silently collapsed either direction). The proposed rule (Gemma predicts
X, SigLIP disagrees, MobileNet independently agrees with SigLIP's
alternative) **catches 50% of the 14 genuine errors (57% of all 21) at
a 1.9% false-positive rate** on Gemma's 311 correct predictions - a
real, usable arbitration signal, the first result in this analysis
phase that's a genuine building block rather than just a
characterization finding. Confirms the asymmetric architecture Jon
described with real evidence per role: Gemma=primary, SigLIP=dissent
trigger (first condition), MobileNet=corroborating confirmation (second
condition, not a standalone trigger - the rule needed both towers
agreeing against Gemma). Also found: 8/21 errors have unanimous
7-tower consensus against Gemma (easy cases), 3/21 have every tower
also wrong (a hard ceiling no tower-based rule could catch), and
Gemma's self-reported confidence still doesn't separate these error
cases from its normal correct-case confidence (0.956 vs 0.981 mean, a
third independent confirmation of the known weak-confidence-field
finding). Still not implemented as a production rule - n=21 errors is
small, and next steps (larger sample, taxonomy-consistency fix,
third-condition testing) are noted, not executed; (17) this pass -
reran Gemma restricted to a flat 8-bucket prompt (the exact categories
ground truth uses, no `casual_photo`/`photo_collage`/`cemetery_photo`),
per Jon's request, to remove the taxonomy-mismatch confound at the
source. Result: 94.0% (312/332), essentially flat vs. the 11-category
run's 93.7%. But a per-image diff found this ISN'T just "casual_photo
noise subtracted cleanly" - 6 brand-new errors appeared that have
nothing to do with casual_photo, most strikingly 3/4 handwritten_ledger
images flipping to map_land_record (a pairing with no obvious semantic
link). Net accuracy barely moved but the actual error SET changed
substantially (17/332 predictions flipped). Read as the same
prompt-structure-sensitivity finding this project already has strong
precedent for (the manifest/census gate-ordering effect), now
demonstrated at whole-taxonomy scale: the category list offered to the
model isn't a neutral filter, changing it shifts decision boundaries
for categories that were never touched. Flat-8 (94.0%) is the fairer
number to compare against the 8-category towers (92.5% best/siglip)
going forward. Also, separately: while sourcing ground truth for
casual_photo/cemetery_photo/photo_collage (found to have ~0-2 real
examples each across the whole project, a hard blocker for any
11-category tower fine-tune), built `data/outputs/new_taxonomy_ground_
truth.csv` with an explicit 3-way PROVENANCE tag per Jon's instruction
(naturally_occurring_corpus / algorithm_flagged_human_confirmed /
deliberately_sourced_reference) so a future evaluation can test against
only naturally-occurring images rather than accidentally measuring
recognition of hand-picked exemplars - documented in a companion
README for whichever session/model sources more examples next; (18)
this pass - began PRODUCTION IMPLEMENTATION of the Decision Engine per
Jon's explicit direction, using this doc as design basis without
hard-coding current findings as permanent. Built `core/routing_
decision_engine.py` (new module, distinct from the pre-existing
`core/decision_engine.py` - naming collision addressed explicitly in
both files' docstrings and CODE_MAP.md), `config/decision_engine.yaml`,
`tests/test_decision_engine.py` (21/21 passing, all 8 required
scenarios), and `diagnostics/replay_decision_engine.py` (first
milestone: real recorded evidence -> engine -> route/quarantine/audit,
no live sensor wiring yet). Four policies (gemma_primary/vision_primary/
weighted_fusion/class_specific_authority), proven config-switchable
with zero code changes (test 8, and a real-data policy comparison in
the replay). gemma_primary is today's default (Gemma's same-split
93.7-94.0% beats the best tower's 92.5%) but explicitly marked
provisional in the config file itself, not structural. Family-weighted
voting directly implements the complementarity finding (correlated
towers get one vote, not one-per-sensor). Every threshold marked
UNCALIBRATED. Real gap found by the replay: Gemma's production path
has no raw logit margin, only the known-unreliable confidence field -
recorded as unscored rather than faked, which means gemma_primary can
never reach AUTO_ACCEPT in this replay (always ACCEPT_WITH_CAUTION/
QUARANTINE), while vision_primary/weighted_fusion DO reach real
AUTO_ACCEPT (towers have genuine scores) at 96.7-97.6% accuracy within
that bucket - a concrete argument for building a live Gemma logit-
margin adapter next. No live sensor adapters, no pipeline_db/bucket-CSV
wiring, no QUARANTINE pipeline effect - explicitly out of scope for
this pass per instruction; (19) this pass - built that live adapter
(`core/gemma_logit_margin_adapter.py`, wraps GemmaLoader without
modifying it, real decision-token logit margin via output_scores=True).
100% decision-token location rate (332/332), correct-case median margin
19.12 vs. wrong-case median 1.75 - the cleanest separation this project
has measured yet. Fair 3-way policy comparison with real uncertainty on
both sides: gemma_primary 94.0% overall AND materially better-
calibrated trust triage than the alternatives - AUTO_ACCEPT+ACCEPT_
WITH_CAUTION cover 81% of images with ZERO combined errors (all 20
mistakes land in QUARANTINE, 68.3% accurate there), while vision_
primary (91.0%) and weighted_fusion (92.8%) both leak real errors into
their own AUTO_ACCEPT bucket (97.6%/96.7%, not 100%). Reinforces
gemma_primary as the current default on real evidence, not just
historical accuracy; (20) this pass - populated `class_specific_
authority` with real per-category accuracy (Gemma + 7 towers, same
split) instead of leaving it empty. Found two categories with genuine
vision-beats-Gemma evidence (portrait_photo: SigLIP 92% vs. Gemma 75%;
website_screenshot: 4 towers unanimous 100% vs. Gemma 96%), one
near-tie kept gemma-primary (map_land_record), 3 tiny-N categories
explicitly excluded rather than guessed. Real-data test of the
populated config scored 93.1%, slightly below gemma_primary's 94.0% -
website_screenshot improved to 26/26 as expected, but portrait_photo
under-performed SigLIP's own standalone 92% (reached only 8/12) because
of a genuine, now-empirically-confirmed limitation in the policy's
single-pass bootstrap design: a category's override only fires when the
whole-evidence plurality vote already lands on that category first,
which portrait_photo's generally-weak sensor agreement often prevents.
Traced via the audit trail, not just inferred from the aggregate number.
Not fixed in this pass - flagged as a real next design question
(iterative/fixed-point bootstrap, or per-category bootstrap-weight
adjustment), not silently patched over.

---

## The core finding

Every piece of a voting system already exists as evidence. Nothing
currently decides with it.

```
Stage 1 (sensors)          Stage 2 (decision_engine.py)       Stage 5 (classifier.py)
  8 tower embeddings   -->   consensus_category computed  -->   RECORDED, never read
  table_confidence (CV)      (unanimous/majority/split/          Gemma classifies alone  -->  FINAL
  layout_detections           complete_disagreement)              (zero upstream signal)      routing
    (uncalibrated)                    |                                    |
                                       v                                   v
                              written to DB + sidecar         audit_classification_against_
                              (observational only)             tower_consensus() compares
                                                                 AFTER the fact (human review
                                                                 paused, never completed)
```

Confirmed directly: `core/classifier.py` has zero references to
`decision_engine`, `consensus`, or `tower` anywhere - Gemma's one call
is genuinely the entire routing decision today.

## What's already gathered

**1. Eight independent vision-tower embeddings, per image, already
captured** (`core/baseline_embeddings.py`, `data/baseline_embeddings.json`)
- DINOv2, ConvNeXt, NaFlex-SigLIP, fixed-SigLIP, EVA02, BEiT, Swin, MAE.
Captured before preprocessing, once per image, for the ~1500-image
corpus already. Zero additional inference cost to reuse.

**2. A validated nearest-cluster + consensus mechanism already built**
(`core/vision_embeddings.py`'s `predict_nearest_bucket()` /
`classify_consensus()`, validated in the Multi-Tower Routing Audit,
`docs/BENCHMARK2_3_MULTI_TOWER_ROUTING_AUDIT.md`). Real results, 146
images, 9 voters (8 towers + Gemma):
- 86% of images (unanimous + majority) show real, stable consensus -
  only 1/146 was genuine chaos.
- Clear encoder-family clustering: ConvNeXt/BEiT/Swin agree tightly
  (0.84-0.89); NaFlex-SigLIP/fixed-SigLIP cluster (0.82). **MAE is a
  consistent outlier**, agreeing weakly with everything (0.49-0.64) -
  a known-lower-reliability voter, not equal-weight material.
- Multi-tower consensus disagrees with Gemma on 38% of images, down
  from a single tower's (DINOv2) 44% - the ensemble is measurably
  cleaner than any one tower alone.

**3. A Stage 2 module already computes this consensus**
(`core/decision_engine.py`), explicitly designed to run before Gemma
using zero new inference. Its own docstring: "this module does not act
on it, only records it."

**4. A post-hoc audit tool exists too**
(`audit_classification_against_tower_consensus()`) - compares Gemma's
actual Stage 5 output against tower consensus after the fact, writes a
sorted disagreement report. Its own docstring: "nothing here re-routes
or corrects a classification automatically." **The human-review pass
that would have closed this loop was flagged and paused, never
completed** - this is the single highest-value, most-nearly-done next
step (see below).

**5. A calibrated, independent CV structural signal**
(`core/image_analysis.py`'s `table_confidence`) - validated against 16
labeled census pages (real tables score >=0.375). A genuinely different
KIND of evidence than Gemma or the embedding towers: classical
structural measurement, not learned visual similarity.

**6. A pretrained general-purpose layout detector, integrated but
uncalibrated for this domain** (`core/layout_detector.py`,
DocLayout-YOLO/DocStructBench). Returns labeled bounding boxes, not an
embedding. Real caveat found during integration: a smoke test on a real
census page returned one `figure` box spanning nearly the whole page at
0.91 confidence - general/academic training distribution may not
transfer to handwritten/tabular genealogical documents. Not measured at
scale.

**7. A custom-trained YOLO, in a separate session** - see the dedicated
section below. Purpose-built for table/row/header STRUCTURAL detection
(extraction), not document-category classification - a different role
than items 1-6, corrected from an earlier draft of this proposal that
conflated the two.

## Proposed architecture (revised 2026-08-06 after review)

**REVISION NOTE**: the design below replaces an earlier "tiered voting"
draft, revised after a second-opinion review (Jon's own GPT scratchpad
session, used to pressure-test ideas before they reach this repo). Three
corrections were adopted outright, one idea was added with a caveat.
Kept for the record since the reasoning matters for anyone extending
this later:

1. **Towers are correlated families, not 8 independent votes.** The
   audit's own pairwise-agreement matrix proves this - ConvNeXt/BEiT/
   Swin (0.84-0.89) and the two SigLIPs (0.82) are each effectively ONE
   opinion measured three (or two) times, not three/two independent
   pieces of evidence. Counting them separately (even with MAE
   down-weighted) risks the correlated families outvoting a genuinely
   independent signal just by redundancy. Family consensus should be
   computed first, then families - not raw encoders - become the
   voting units.
2. **Evidence fusion, not sequential tiers.** The original "Tier 1
   narrows the field before Tier 2 votes" framing implies gating/
   suppression and forces every sensor to answer in the SAME categorical
   taxonomy terms, which fits Gemma and the embedding towers naturally
   but is awkward for structural detectors. YOLO shouldn't have to say
   "Census" - it should report "large table + 48 row boxes + header
   region" as raw structural observations, and a fusion layer
   translates heterogeneous evidence into a taxonomy decision, not each
   sensor forcing its own vote into that shape.
3. **Retracted: "split towers -> defer more to Gemma."** This was an
   unvalidated rule and shouldn't have been stated as a default even
   tentatively - tower disagreement means tower evidence is weak, not
   that Gemma is automatically more trustworthy on that image. A weird/
   novel document can confuse both. Whether split-tower cases should
   defer to Gemma or go straight to quarantine is an open question the
   human-labeled disagreement set (below) should answer, not something
   to hardcode ahead of that evidence.
4. **DocLayout-YOLO demoted to comparison baseline, not a voting
   participant** - the custom fine-tune now has real measured numbers on
   this project's own data (P=0.97/R=1.00/mAP50=0.995 table detection,
   `docs/LAYOUT_DETECTOR_BOOTSTRAP_TRAINING.md`) that clearly outrank
   DocLayout-YOLO's one bad smoke-test data point on the same kind of
   document. Once the custom detector's output is stable, it becomes
   the real structural-evidence source; DocLayout-YOLO stays around only
   as an experimental comparison, not something the fusion layer weighs.

**Evidence families feeding one fusion layer**, each reporting its own
native observation type rather than being forced into a category vote:

```
IMAGE
 |
 +-- Classical CV (core/image_analysis.py)
 |     table_confidence, line structure, ROI geometry
 |
 +-- Row/column grid-anchor detection (core/row_segmentation.py) - DUAL
 |     PURPOSE, not classification-first: originally built purely to
 |     improve EXTRACTION (row/column boundary detection for Stage 6),
 |     shelved earlier after a first bad attempt
 |     (`core/auto_sidecar.py`'s removed `_detect_header_number_blobs()`,
 |     see docs/COLUMN_NUMBER_ANCHOR_RESEARCH.md's "prior art" section).
 |     Today's refinement got it working well enough that Jon separately
 |     realized (2026-08-06) the SAME signal also serves as a
 |     classification prior - a realized secondary benefit discovered
 |     after the fact, not the original design goal. Both purposes are
 |     real and current: detect_row_number_centers() / detect_column_
 |     number_centers() + find_number_row_band() auto-calibration.
 |     "Does this page have a detectable, regular printed-number grid
 |     in both directions" is a direct, cheap proxy for "is this
 |     dense_tabular_rows" (census/manifest), independent of Gemma/
 |     towers/YOLO - see the dedicated entries below for full
 |     validation detail.
 |
 +-- Custom YOLO (training/yolo_assisted_auto_sidecar.py, once stable)
 |     table/row/header boxes, coverage, box counts
 |
 +-- Vision encoders (core/vision_embeddings.py, family-consensus first)
 |     structural-CNN-family vote (ConvNeXt/BEiT/Swin, collapsed to one)
 |     SigLIP-family vote (collapsed to one)
 |     DINOv2/EVA02 (semi-independent)
 |     MAE (outlier - keep as a distinct, separately-weighted signal,
 |          not folded into a "towers" average - see open question below)
 |
 +-- Gemma
 |     taxonomy choice, confidence, reason (semantic/text-grounded -
 |     the one source that reads actual header/content text)
 |
 +-- Metadata/context (core/pipeline_db.py's images table + provenance)
       source, known year (if any), template family, acquisition
       provenance, needs_manual_dewarp flag, source_type/page_number
       (page_number==1 from a multi-page source is a real, cheap prior
       toward "title/index card" - see the LAC title-card finding below)
             |
             v
      EVIDENCE FUSION (design TBD - depends on the disagreement-set
      analysis below; do not hardcode weights or rules ahead of that)
             |
      route / quarantine
```

**Additional evidence sources found this round (2026-08-06), not
previously catalogued in this doc:**

- **`core/image_analysis.py` computes far more than `table_confidence`
  alone**, all already sitting in per-image analysis sidecars at zero
  additional cost: whole-frame `aspect_ratio`, `deskew_angle_deg` (+ a
  `deskew_angle_clamped` flag - a clamped estimate means the true angle
  is AT LEAST the reported value, not a real measurement), `blur_
  laplacian_var`, `noise_residual_std`, `ruling_lines_vertical/horizontal`
  (counts + angles); per-region (frame/page/table): `luminance_mean/std`,
  `contrast_p5_p95_spread`, `otsu_threshold`, `ink_fraction`,
  `illumination_unevenness`, `text_height_px`, `stroke_width_px`,
  `component_count`, `largest_ink_blob_frac`, `page_confidence`/
  `page_method`. Category-relevant intuition: a census schedule's
  `text_height`/`ink_fraction`/`ruling_lines` distribution should look
  measurably different from a portrait photo's or a manifest's, purely
  structurally, independent of Gemma or the embedding towers - untested,
  but free to compute since it's already there.
- **`core/pipeline_db.py`'s `images` table**: `needs_manual_dewarp` (a
  flag - pages needing manual dewarp intervention may correlate with
  atypical/hard-to-classify documents) and `source_type`/`page_number` -
  the latter directly relevant to the LAC title-card finding below:
  `page_number == 1` (or similarly low) from a multi-page PDF/microfilm-
  reel source is a real, cheap prior toward "this might be a title/index
  card, not a content page."
- **`core/auto_sidecar.py` has its own independent structural
  detectors** - `locate_table_boundary()`, `locate_columns()`,
  `detect_data_rows()` - a THIRD structural-evidence source (distinct
  from `image_analysis.py`'s classical ROI detection and from either
  YOLO model), already computed during row extraction but not yet
  connected to anything else per `docs/CODE_MAP.md`. Currently runs
  DOWNSTREAM of routing, not before it - whether it should also inform
  Stage 5 is a real ordering question for the fusion design, not
  resolved here.
- **`core/row_segmentation.py`'s `detect_row_number_centers()` -
  reinforced/improved recently (2026-08-05/06, exact session
  uncertain)**: detects printed row-numbers as a stronger row anchor
  than ruling-line detection - validated on a real 1901 census page
  where ruling-line detection couldn't clear even a heavily-relaxed
  threshold, but row-number-center detection found 50/50 rows cleanly
  matching true (non-uniform) spacing. **Correction, 2026-08-06: this
  (and its column-axis companion below) is DUAL PURPOSE, not
  classification-first as an earlier draft of this doc overstated** -
  built originally to improve EXTRACTION (row/column boundary detection
  for Stage 6), shelved after a first bad attempt, revived and refined
  today - the classification use is a realized secondary benefit Jon
  identified after the refinement worked, not the original design goal.
  Both purposes are real and current: ROW-COUNT REGULARITY as a
  category prior - census and passenger manifests both have many
  regular, repeated rows; most other categories don't. This is a
  DIFFERENT signal than `table_confidence` (which measures "is there a
  table region," not "how many regular rows does it contain") - cheap,
  structural, independent of Gemma/towers, and specifically
  discriminating for exactly the two categories (`dense_tabular_rows`'s
  census/manifest subtypes) that have needed the most disambiguation
  work all session (see the census-vs-manifest Tier 2 prompt-hardening
  history, `diagnostics/test_gemma_prompt_tiering_variants.py`). Not
  yet measured at scale or wired into anything - a real candidate
  feature for the fusion layer, not a decided one.
- **Companion work, same session: `detect_column_number_centers()` +
  `nearest_number_candidate()`** (`core/row_segmentation.py`) - the same
  printed-number-anchor idea applied to the COLUMN axis, not just rows.
  `detect_column_number_centers()` is a pure sensor (raw number-blob
  positions in a band, no column-identity decision, no filtering -
  matches the "Stage A: measure, don't decide" discipline this project
  already follows elsewhere); `nearest_number_candidate()` is the
  filtering step, shared/reusable across both row and column axes.
  Validated: `diagnostics/test_column_number_anchor.py` against a real
  1911 reference page with independently human-confirmed column
  positions - 5/5 known columns matched, 2-5px deltas. Full research
  history (prior failed attempt, the y-band/dashed-line discovery, a
  falsified gap-bridging hypothesis, the filtering fix, results table,
  explicit next steps: cross-year validation, a calibration template
  schema, integration shape into `locate_columns()`'s existing cascade)
  in `docs/COLUMN_NUMBER_ANCHOR_RESEARCH.md`, cross-referenced in
  `docs/CODE_MAP.md`. Nothing touched in production -
  `core/auto_sidecar.py`'s `locate_columns()` is untouched. Combined
  with the row-anchor work above: BOTH axes of a census/manifest page's
  grid structure now have independently-validated printed-number anchor
  detection - strengthens the row-count-regularity prior above into a
  fuller "regular grid in both directions" structural signal, though
  still just as untested-at-fusion-scale as the row-only version.

  **Round 2 update (same doc, later the same day)**: full 41-column
  audit on the 1911 reference page (~83% clean hits; the two known
  failure modes - tight-packing touching-ink, page-edge warp
  degradation - characterized, not fixed); cross-year test on 1931
  (8/10 spot-checked columns 1.0-8.5px, including all 4 sub-lettered
  columns like "4a/4b/4c" - sub-lettering turned out NOT to be a real
  problem for this technique); an inversion experiment ruled out
  (negative result, Otsu adapts to polarity regardless, don't revisit).

  **`find_number_row_band()` - automated y-band calibration, and the
  one finding worth cross-referencing against the 1926 anomaly
  documented elsewhere in this file**: manual y-band calibration
  doesn't scale, so this sweeps/scores candidate bands automatically.
  Validated within ~1px of hand-calibration on 1911 and inside the
  hand-calibrated range on 1931. **On 1926 - the same form flagged as
  structurally anomalous by both the YOLO probe and the Gemma
  classifier test earlier this session - a manual y-band guess found
  only 2 of ~12 visible numbers, but the AUTOMATED search recovered
  8 of them cleanly.** This is a real, useful counterpoint to treating
  1926 as simply "hard": the underlying grid structure is still
  detectable on this form family, it just needs proper (automated)
  calibration rather than a naive assumption that 1911/1921's
  calibration transfers unchanged. Doesn't contradict the earlier
  recommendation to give 1926 its own template/prompt handling - if
  anything it strengthens the case that 1926 needs its OWN calibrated
  parameters specifically, which this auto-calibration function can now
  find without hand-tuning.

  **Final status (2026-08-06, Jon's call): research substantially
  complete, what remains is an engineering problem, not an open research
  question.** Extensive further work (17 rounds total, full log
  `docs/COLUMN_NUMBER_ANCHOR_RESEARCH.md` - its own "Status & handoff"
  section is the authoritative catch-up point, not this summary) found:
  - **1901's real problem was physical, not calibration**: source scans
    come from a bound volume with its own gutter crease, and the header
    row is measurably NOT level across the page width (~80px vertical
    drift left-to-right, visually confirmed). A new research-only module,
    `core/page_dewarp.py`, does crease detection + per-column dewarp to
    address this directly.
  - **A gradient-based (Scharr) line detector strictly dominates the
    existing dark-ridge/threshold detector** on every year tested -
    validated as a real upgrade candidate for `core/auto_sidecar.py`'s
    `_find_vertical_ruling_line()`, with one known unresolved failure
    mode (wrong-line lock-on on tightly-packed columns).
  - Cross-year validation is real, not a single-sample fluke: 1911,
    1926, and 1931 all confirmed with genuine ground truth; 1901's low
    match rate was root-caused (not just observed) down to the physical
    skew above, not left as an unexplained gap.
  - One false start was caught and corrected in the open, not hidden: an
    early "67% match rate" claim turned out to be an extrapolation bug
    that corrupted the image while still moving the metric the right
    direction - caught by inspecting the actual rendered output, not
    trusting the number. Worth remembering as a discipline example for
    any future validation work in this doc's orbit, not just for this
    one module.
  - **Still not done, and still nothing wired into production**: only 1
    of 6 checked pages has real (not visually-guessed) ground truth for
    the crease-position prediction specifically; blind detection without
    a prior has failed every time it's been tried, in multiple shapes;
    `core/auto_sidecar.py`'s `locate_columns()` remains untouched. "The
    remaining work is engineering" means the open questions are now
    "how do we integrate this" and "how do we get more ground truth,"
    not "does this technique work at all" - a real, meaningful shift in
    confidence level, not a claim that integration is trivial or that
    every remaining question is answered.
- **Correction, not a new finding**: the flagged-triage CSVs
  (`misclassifications.csv`, `flagged_bad_deskew.csv`,
  `flagged_for_pruning.csv`, `flagged_needs_new_bucket.csv`) were
  assumed to hold accumulated human signal - checked directly, **none
  currently exist** (empty/deleted as part of the ongoing restructuring).
  The mechanism is live (`classifier_validation_ui.py` still writes to
  them), but there's no populated data there yet - a FUTURE human-signal
  source once used going forward, not a currently-available one.

## Human validation UI - expand scope, don't replace

`ui/manual_classification_ui.py` (built 2026-08-06) currently validates
only the human's own taxonomy judgment. Realization this round: it
should become the calibration tool for the WHOLE evidence fusion system,
not stay Gemma-specific -

```
Image -> Evidence package (Gemma + towers + CV + YOLO + metadata) ->
Human validation -> Ground truth
```

**Discipline: anything derivable automatically from the human's chosen
taxonomy label stays automatic - never a manual click.** Comparing
Gemma's stored bucket, each tower's nearest-bucket prediction, and
family-consensus against the human's final label is pure post-hoc
computation, no UI change needed. Reserve manual input ONLY for
judgments a script genuinely cannot make:

- Does a YOLO/CV detection box actually overlap the real table/row/
  header - not just "was the confidence number high," which the
  Schedule-2 case above proves can be misleadingly high on the wrong
  document.
- Is Gemma's reasoning TEXT actually grounded in the image, independent
  of whether the final category happened to be right (a right answer
  for a hallucinated reason is a different failure mode than a wrong
  answer, and neither script-derivable).
- Metadata contamination (the 1926-folder-containing-a-1921-title-card
  case, found by direct image inspection this session).
- Needs new bucket (already built).

**Proposed shape for later** (not built now): a small row of OPTIONAL,
fast toggle chips next to the existing taxonomy buttons - unset by
default so normal high-speed batch labeling is completely unaffected,
available with one click each specifically when working through the
56-case disagreement review or similar targeted evidence-calibration
passes. The goal per Jon's framing: generate calibration data for every
evidence source without slowing normal labeling, not add friction to
every review pass regardless of purpose.

**A further idea worth designing toward, with a real caveat attached:**
Gemma is currently run on every image unconditionally. If CV +
structural-family + custom-YOLO evidence strongly and independently
agree, that may be sufficient to route WITHOUT ever loading/running
Gemma - extending this project's existing E2B/E4B escalation pattern
(`[[project_e2b_e4b_escalation_architecture]]`: E4B is a gated
second-stage reviewer, not a universal replacement) one level further -
Gemma becomes an escalation sensor for disagreement/ambiguity, not a
mandatory stage for every image. This would attack the pipeline's
single most expensive inference stage structurally, not just optimize
its decode speed.

**The caveat, backed by a concrete case found during this session's own
manual review** (see `docs/GEMMA_HIERARCHICAL_ROUTING_INVESTIGATION.md`-
adjacent image inspection, 2026-08-06): a real LAC 1901 pull turned out
to be Schedule No. 2 ("Buildings and Lands, Churches and Schools"), not
a population census page. That image would score high `table_confidence`,
get real YOLO table/row detections, and likely cluster near other
census-tabular images in embedding space - strong structural agreement
across every non-Gemma signal - yet it's the wrong category. The ONLY
thing that caught this was Gemma reading the actual header text. A
structural-agreement-only escalation gate would systematically route
exactly this class of error straight past Gemma with high confidence.
**Any Gemma-skip gate needs a semantic-ambiguity check, not just
structural agreement** - e.g. still escalating to Gemma whenever a page
matches a known "structurally similar but taxonomically distinct" cluster
(census Schedule 1 vs. 2/3/4, real census vs. manifest), even if every
structural signal agrees. What that check should actually look like is
itself an open research question, not designed here.

The manual classification UI (`ui/manual_classification_ui.py`, built
2026-08-06) is the natural destination for whatever this system flags
as disagreement or ambiguous - direct feedback loop from fusion-layer
uncertainty into human-verified ground truth.

## What's needed before this could actually be implemented

1. **The 56 disagreement cases from the original Multi-Tower Routing
   Audit were never human-reviewed** - this is the calibration data
   needed to know whether tower-consensus disagreement actually
   predicts a real Gemma error, or is mostly noise. Highest-value next
   step, already ~90% done (cases are archived, just not looked at).
   **Now specified concretely** (see experiment design below).
2. **DocLayout-YOLO's domain transfer to this corpus has one data
   point** (one smoke test, one wrong-looking result) - demoted to
   comparison baseline per the revision above; not worth further
   investment unless the custom YOLO's own results need a reference
   point.
3. **No confidence-gating threshold has been chosen anywhere in this
   system** - explicitly flagged as deferred in the audit doc. Depends
   on the disagreement-set analysis below, not decidable in the
   abstract.
4. **The custom YOLO's final output shape/classes aren't fully settled**
   (row detection still plateaued ~84.5% recall v3->v4) - design should
   accept it later without a redesign; table detection (P=0.97/R=1.00/
   mAP50=0.995) is solid enough to start using now if needed.

## The 56-disagreement-case experiment - RUN, with real results (2026-08-06)

**Status: complete for the 50 recoverable cases (6 of the original 56
had no surviving image anywhere, source or local archive copy - listed
in the queue-build step, excluded, not silently dropped).** Human
review done via `ui/manual_classification_ui.py`'s disagreement queue
(`data/logs/reviewed/disagreement_review_queue.csv`), joined against
each case's archived `record.json` via `diagnostics/
analyze_disagreement_review.py`. Full per-case detail: `data/logs/
reviewed/disagreement_analysis_report.txt` / `.json`.

### Headline result: on this set, Gemma beats tower consensus 2x

| Source | Correct (of 48 classified cases) |
|---|---|
| Gemma | 23/48 (47.9%) |
| Tower consensus (flat 8-way) | 11/48 (22.9%) |
| Family-collapsed consensus | 11/48 (22.9%) |

When Gemma and the towers disagree, Gemma is right roughly twice as
often as the towers are. **This is the opposite of what the original
"trust tower-consensus disagreement as an override signal" hope
implied** - real evidence against a blanket disagreement-triggers-
distrust-of-Gemma rule, not confirmation of it. ("Both right: 0" in the
raw breakdown is a structural artifact of the case-selection criterion
(Gemma-bucket != tower-bucket by construction for every case in this
set), not a separate finding - flagging so it isn't misread later.)

### MAE: confirmed dead weight on this sample, not orthogonal signal

5/48 (10.4%) individual accuracy - worst of all 8 towers - and it
uniquely rescued **0 of the 37 cases** where the rest of the towers
missed. Directly answers the open question below: on this sample, MAE
isn't catching anything the correlated majority misses. Real support
for dropping/heavily down-weighting it, with the standing caveat that
n=48 is a modest sample to hang a permanent architectural decision on.

### Family-collapsing: no net accuracy change, but real per-case shifts

11/48 correct either way (flat vs. family-collapsed) - the aggregate
number didn't move. Individual cases DID flip both directions (e.g.
`disagreement_037` went from wrong to right; others flipped the other
way), netting to the same total. The family-not-raw-encoders correction
wasn't wrong architecturally, it just isn't shown to matter for THIS
sample's top-line accuracy.

### A real, specific, actionable failure pattern found in the per-case data

Two visible clusters, not just aggregate noise:
- **8 cases where tower/family consensus said `dense_tabular_rows` but
  the correct answer was `map_land_record`** - Gemma got every one of
  these right. Towers show a systematic blind spot confusing land-record
  maps with dense tabular content.
- **A run of `portrait_photo` cases where Gemma consistently got it
  right and towers scattered across genealogy_chart/mixed_text_image/
  map_land_record** - towers specifically unreliable on portraits in
  this set.

### The important caveat, stated plainly

**This 48-case set is NOT a representative sample of overall Gemma or
tower accuracy - it's the hard subset where Gemma and towers already
disagreed**, selected adversarially from the towers' perspective too.
Don't generalize "towers are 23% accurate" beyond this specific hard
subset. Also: **zero of these 48 cases were census/manifest** - this
analysis says nothing about the census-vs-manifest disambiguation work
from earlier in the session; it's informative mainly for the
photo/map/screenshot-adjacent categories.

### What this means for the fusion design

Real evidence against treating tower-consensus disagreement as an
automatic override signal for Gemma. If anything, on this evidence,
Gemma should stay primary, with tower consensus narrowed to a much more
targeted role (e.g. only for categories where it actually shows signal,
like the map/portrait clusters above used as a targeted secondary check)
rather than a blanket "disagreement -> distrust Gemma" rule. This is a
meaningfully different conclusion than the proposal originally hoped to
validate - recorded as such, not softened.

### Original experiment spec (for reference - now answered above)

- human label (ground truth, from the new UI) - DONE
- Gemma's label, confidence, and reason text - DONE (confidence field
  itself wasn't analyzed for calibration this pass - a possible follow-up)
- each individual tower's prediction/similarity score - DONE
- family-collapsed consensus - DONE, no net accuracy change (see above)
- `table_confidence` and other `image_analysis.py` structural stats -
  NOT YET pulled into this analysis - a real follow-up, not done this
  pass
- custom YOLO detections where available - NOT YET available/applicable
  to this analysis (YOLO isn't a category-vote signal per the fusion
  correction earlier in this doc)
- metadata (source, known year if any, template family) - NOT YET
  pulled into this analysis

---

## Logit-based confidence - real fix for the broken self-reported confidence field (2026-08-06)

**Origin**: the single-shot classifier test and every gated-tree variant
this session confirmed Gemma's self-reported `confidence: x.xx` text
field is broken - either malformed (`-99.9`, `Can 1.0`, `Can't
determine.`) or, when well-formed, near-uniformly high regardless of
correctness. Jon's proposal: instead of trusting the model's own
narrated belief about its belief, read the model's ACTUAL next-token
probability distribution at the real decision token - a mechanistically
grounded signal, not a self-report.

### Mechanism, validated

`model.generate(output_scores=True, return_dict_in_generate=True)` - a
first-class HF feature, no manual decode-loop needed (unlike the earlier
KV-cache-reuse investigation, `docs/GEMMA_HIERARCHICAL_ROUTING_
INVESTIGATION.md`) - returns `.scores`, the raw per-step logits for
every generated token. Tested against the already-validated
`GATE_CENSUS_PROMPT` (single yes/no answer - ideal for this technique,
unlike a multi-token free-form category word):

- **The decision token ("yes"/"no") is reliably locatable** - decode
  each generated token individually, find the first one that strips/
  lowercases to exactly "yes" or "no". Confirmed at a STABLE index
  (always position 4) across all 10 ground-truth images tested.
- **Truncation-stability confirmed empirically, not just assumed from
  greedy-decoding theory**: re-ran at smaller `max_new_tokens` budgets
  (4/8/16) and confirmed the token sequence up to the decision point is
  byte-identical regardless of budget - the decision token's position
  never shifts based on how much more the model is allowed to generate
  afterward.

### First result: softmax probability is ALSO saturated (not just the self-report)

On 10 images the gate answered correctly, every softmax probability at
the decision token came back ~1.0000. This is a real, important finding
in its own right, separate from the self-report question: it's not that
Gemma was lying about confidence in text - the actual logits genuinely
are that peaked for a clear yes/no decision under greedy decoding.
Softmax near 1.0 has almost no resolution left near the top of the
scale to distinguish "very sure" from "extremely sure."

### The real signal: RAW LOGIT GAP, not softmax probability

Tested against 5 real KNOWN-WRONG cases (disagreement-review images
where Gemma's ORIGINAL wide classify prompt confidently mis-called
`dense_tabular_rows`, confirmed wrong by human ground truth) plus a
scaled run across the FULL 48-image disagreement-review set (all
confirmed non-`dense_tabular_rows`, so ground truth `is_census=False`
for every one) - both per Jon's direction to use existing ground truth
rather than construct synthetic test cases.

**Result, n=48, `gate_census` prompt:**

| | Mean logit gap | Range |
|---|---|---|
| Correct (47/48, said "no") | 26.82 | 22.2 - 30.0 |
| **Wrong (1/48, said "yes")** | **3.38** | - |

Clean, total separation on this run - the minimum correct-case gap
(22.2) is nearly 7x the one wrong case's gap (3.38), zero overlap.
Softmax probability barely moved on the wrong case (0.9669 vs a flat
1.0000) - the raw logit gap is where the real resolution lives,
invisible if you only look at the winning token's probability.

**Second run, n=48, `gate_manifest` prompt against the SAME 48 images**
(independent second source of wrong-case data, per Jon's direction -
"use them but present them with a wrong prompt... to try and capture
more No's"):

| | Mean logit gap | Range |
|---|---|---|
| Correct (47/48, said "no") | 25.34 | 5.6 - 30.9 |
| **Wrong (1/48, said "yes")** | **6.69** | - |

**This time the separation is NOT clean** - one CORRECT case
(Handwritten Ledger) had an even lower gap (5.63) than the wrong case's
6.69. Reported honestly rather than letting the first run's clean result
stand uncorrected.

### Revised conclusion after the second run: logit gap flags UNCERTAINTY, not correctness directly

The right reading of both runs together: a low logit gap most likely
means "the model was genuinely torn," not strictly "the model is about
to be wrong." A near-coin-flip decision can still land on the correct
side by chance (the Handwritten Ledger case) - it was RIGHT, but the
model was nearly as uncertain about it as it was on the case it actually
got wrong. Both are exactly the kind of case worth flagging for human
review regardless of which side the toss-up landed on.

Across both gates combined (2 real wrong answers, from 2 independent
gates, on 2 different images): wrong cases cluster low (3.4, 6.7) while
the bulk of correct answers sit well above 20 - a quarantine threshold
somewhere around 8-10 would catch both real errors AND the one shaky-
but-correct case, at the cost of some false quarantines on cases that
happen to be right despite low confidence. That's a normal, acceptable
tradeoff for a quarantine gate (better to over-flag borderline-but-
correct cases for review than to miss real errors), not a flaw in the
signal - it just means this isn't a perfect correctness classifier, it's
an uncertainty detector, which is actually the more useful and more
honest thing to have built.

**Honest limitation, stated plainly at the time**: only 2 wrong cases
total across both gates - not enough to know the real shape of the
wrong-case distribution. Resolved by the large-scale run below.

### Large-scale run (n=368) against the pre-automation whole-corpus ground truth - the picture gets more honest, not cleaner

`data/outputs/reference_pipeline_v2/manifest_final.csv` (found
2026-08-06, see "Cross-session context" further down for what it is) -
a real, human-reviewed, whole-corpus classification set from before
subtypes existed, giving REAL POSITIVE examples for the first time (the
48-case disagreement set had zero true `dense_tabular_rows` positives).
Built `GATE_DENSE_TABULAR_PROMPT` from the real, current `classifier_
guidance` text for `dense_tabular_rows` (top-level category, unchanged
between old and new taxonomy - only its subtypes are new, so this old
ground truth is directly valid for it, unlike a census/manifest-specific
gate). Ran against all 168 real positives still on disk + 200 randomly
sampled negatives (`diagnostics/test_gemma_logit_confidence_old_
taxonomy.py`).

**Finding 1 - a real, separate, one-directional model bias**: all 42
errors were FALSE NEGATIVES (real `dense_tabular_rows` images the gate
wrongly said "no" to). ZERO false positives across 200 real negatives.
Recall on true positives: 126/168 = 75%. Precision: 126/126 = 100%. The
model is very conservative on this category specifically - never claims
something is tabular when it isn't, but misses a real quarter of the
genuine cases.

**Finding 2 - the confidence signal is directionally real but NOT
cleanly separable at this scale**:

| | Median logit gap |
|---|---|
| Correct (326) | 30.41 |
| Wrong (42) | 13.56 |

Medians clearly differ - a real, useful population-level signal,
consistent with the smaller runs' direction. But the overlap is wide:
min correct-case gap 0.125, MAX WRONG-CASE GAP 30.625 - nearly
indistinguishable from the top of the correct-case range. **Several
false negatives were confidently wrong** (gap 25-30), not just
uncertain-and-wrong - the worst failure mode for a confidence signal,
and one the smaller (n=5, n=48x2) runs' clean-separation results didn't
surface. One case did land at the theoretical ideal (`gap=0.000,
P=0.500000`, a genuine coin-flip, and wrong) - the effect is real, just
not reliable enough per-case to trust as a standalone hard gate.

**Revised conclusion, superseding the "clean uncertainty flag" framing
above**: logit gap is real, useful signal - keep it - but at this
larger, more realistic scale it should be treated as ONE INPUT to a
threshold/fusion decision, not a standalone reliable quarantine trigger
on its own. A naive low-gap-only threshold would miss a meaningful
fraction of the confidently-wrong cases found here. The smaller runs'
apparent cleanliness was very likely a small-sample artifact, not a
property that held up - recorded here, not quietly replaced, so the
reasoning trail stays honest for whoever builds on this next.

### What this means for the fusion design

RAW LOGIT GAP AT THE DECISION TOKEN is real, worthwhile signal for the
fusion layer - but per the large-scale result above, it should be
combined with other evidence (structural/CV signals, tower consensus)
rather than used alone as a quarantine gate, since a meaningful fraction
of real errors are confidently wrong on this metric alone. Still solves
part of the confidence-gating gap flagged as unresolved everywhere else
in this doc, just not the whole thing by itself. Separately, the
one-directional false-negative bias found on `dense_tabular_rows`
specifically (conservative, never false-positive, misses 25% of real
positives) is itself useful evidence for the fusion design - it suggests
this gate's "no" answers deserve more scrutiny than its "yes" answers
when other evidence (CV table_confidence, row/column-anchor detection)
suggests a page IS tabular. Does NOT directly apply to free-form
multi-token answers (e.g. the census-year extraction) without more
design work - another reason the gated-binary tree structure (v7,
already adopted) is the right shape to build this on, not the earlier
free-form single-shot prompt.

Scripts: `diagnostics/test_gemma_logit_confidence.py` (mechanism
validation, decision-token location + truncation stability),
`diagnostics/test_gemma_logit_confidence_known_wrong.py` (5-case
known-wrong stress test), `diagnostics/test_gemma_logit_confidence_
scaled.py` (48-case `gate_census` run), `diagnostics/test_gemma_logit_
confidence_scaled_manifest.py` (same 48 images, `gate_manifest`),
`diagnostics/test_gemma_logit_confidence_old_taxonomy.py` (n=368
large-scale run against real pre-automation ground truth, the result
that superseded the smaller runs' clean-separation framing). Reports:
`data/logs/reviewed/logit_confidence_scaled_report.txt`,
`logit_confidence_scaled_manifest_report.txt`, `logit_confidence_
old_taxonomy_report.txt`.

### Does an earlier point in the sequence already "secretly" decide? No - confirmed three ways

Jon's follow-up question: since we only read the logits at the literal
decision token, is the real decision already baked in earlier - either
in prior OUTPUT tokens, or in the model's hidden state before generation
even starts? Tested three ways, all pointing to the same conclusion.

1. **Trajectory dump** (`diagnostics/test_gemma_logit_trajectory.py`):
   printed the full top-8 candidates at EVERY generated step, not just
   the decision step. Tokens 0-3 of the label-echo format
   (`is_census: `) are perfectly deterministic on every image checked -
   `p=1.000000`, next candidate rounds to `0.000000`. No competition at
   all, regardless of image content. The literal decision genuinely only
   happens at the token already being read (index 4) - no earlier OUTPUT
   token secretly decides anything.

2. **First-token reformulation** (`diagnostics/test_gemma_logit_
   confidence_first_token.py`): if those earlier tokens are truly
   decision-irrelevant, stripping them (reformatting the prompt so the
   answer word IS token 0) should be neutral or better. Tested against
   the SAME 368-image sample as the large-scale run above - **it was
   worse on both axes**: accuracy dropped from 88.6% to 79.9% (recall
   fell further, 75.0% -> 56.5%, and the previously-perfect precision
   cracked, 1 false positive appeared where there were 0 before), AND
   the confidence signal weakened (correct-case median gap dropped from
   30.4 to 14.8). **Conclusion: the "boring" format-echo tokens are
   functionally load-bearing, not filler** - each one gives the model
   another autoregressive forward-pass step to settle on an answer
   grounded in the image, the same reason a little chain-of-thought
   "thinking room" helps LLMs generally even when the intermediate
   tokens don't look like visible work.

3. **Pre-generation lean** (`diagnostics/test_gemma_logit_
   pregeneration_lean.py`): checked the logits at the LAST PROMPT
   POSITION (index N-1, before token 0 is generated - one forward pass,
   `generate(max_new_tokens=1)`'s `scores[0]`) for raw probability mass
   specifically on yes/no-like tokens, buried far under the dominant
   top-1 winner ("is", trivially, since the model is following the
   requested format). Same 368-image sample. **Result: 59.5% accuracy -
   barely above the 50% random-chance floor, and the wrong cases weren't
   random noise, they showed a real directional bias toward "yes"**
   regardless of ground truth (most `printed_document`/"no" cases leaned
   "yes" anyway). Whatever weak signal exists there (correct leans had
   ~2x stronger yes/no ratios than wrong ones, 11.04x vs 5.39x) is far
   too weak and biased to use.

**Combined conclusion**: there is no shortcut. The model's real,
reliable, image-grounded decision only crystallizes through its normal
autoregressive generation process, including the tokens that look
decision-irrelevant. The current gated-prompt architecture (full
label-echo format, read the logits at the actual decision token) is
correct as built, not just incidentally structured that way - confirmed
by directly testing and rejecting both alternatives (skip ahead to
token 0, or peek before generation starts) rather than assuming either
would work.

---

## Resource: pre-automation whole-corpus ground truth found (2026-08-06)

`data/outputs/reference_pipeline_v2/manifest_final.csv` - a real,
human-reviewed classification set for the WHOLE corpus, from before the
pipeline was automated (manifest went through the manual validation UI
directly). 1975 rows, columns `file_path, category, status, source_
original_path`. 1749/1975 (88.6%) of file_paths still exist on disk as
of 2026-08-06 - the rest were likely moved/pruned during later
reorganization work. Category distribution: printed_document 1228,
website_screenshot 278, dense_tabular_rows 193, map_land_record 105,
portrait_photo 90, handwritten_ledger 38, mixed_text_image 21,
genealogy_chart 19, uncertain_review 3 (a very low uncertain count,
consistent with a properly cleaned-up reviewed set, not a raw
automated pass).

**Important caveat**: predates subtypes (census/passenger_manifest under
`dense_tabular_rows`) - only has real ground truth at the TOP-LEVEL
category. Top-level category names are unchanged between old and new
taxonomy (only subtypes are new), so this is directly valid for any
top-level-category question, NOT for census-vs-manifest-specific
questions. This is exactly why the large-scale logit-confidence run
above targeted `dense_tabular_rows` (a real, unchanged top-level
category) rather than `gate_census`/`gate_manifest` (a subtype
distinction this ground truth cannot verify).

This is a much larger, real resource than the 48-56 case disagreement
set used throughout most of this doc - worth remembering as the
default ground-truth source for any future top-level-category
validation work, not just the one use above.

## Candidate tower tested and mostly rejected: MobileNetV2 (2026-08-07)

Tested `google/mobilenet_v2_1.0_224` (timm tag `mobilenetv2_100.ra_in1k`,
matching architecture/width/resolution) as a candidate NEW tower for the
fusion ensemble - NOT one of the 8 already-validated `QUALIFIED_ENCODERS`.
Reused the existing, already-validated `core/vision_embeddings.py`
machinery directly (`build_model_and_transform`/`embed_pooled`/
`predict_nearest_bucket` - same nearest-cluster-via-mean-cosine-
similarity methodology as the real Multi-Tower Routing Audit, not a
one-off evaluation invented for this test). Reference/test split against
the real `manifest_final.csv` ground truth (15 reference + up to 30
held-out test images per category).

**Result: 54.4% overall (92/169), unevenly distributed by category** -
meaningfully above chance (8 categories, ~12.5% floor) but well below
both the existing 8-tower set and Gemma:

| Category | Accuracy |
|---|---|
| website_screenshot | 100% (30/30) |
| mixed_text_image | 83.3% (5/6, small sample) |
| handwritten_ledger | 69.2% (9/13) |
| printed_document | 56.7% (17/30) |
| dense_tabular_rows | 43.3% (13/30) |
| map_land_record | 43.3% (13/30) |
| portrait_photo | 16.7% (5/30) |
| genealogy_chart | no held-out test data (only 6 images total on disk) |

**portrait_photo's failure is specific and real, not noise**: 19 of 30
real portraits were misclassified as `mixed_text_image` in the confusion
matrix - surprising on its face since ImageNet models are usually strong
on people/faces, most likely explained by these being old black-and-
white/sepia archival portraits, visually quite different from the
modern photography ImageNet was trained on.

**Verdict: not a strong general voter for this taxonomy.** Makes sense
given MobileNetV2 is much smaller (~3.4M params) and purely supervised-
ImageNet-object-trained, versus the modern, larger, self-supervised
towers already in use. **One real exception worth keeping**:
`website_screenshot` at 100% is a genuine, cheap, nearly-free signal -
digital-native screenshots have visual statistics different enough from
scanned paper documents that even a small generic model separates them
trivially. **Candidate use: a fast, low-cost `is_screenshot` pre-filter
gate ahead of more expensive evidence sources**, not a full voting
member of the fusion ensemble alongside the 8 qualified encoders.

Script: `diagnostics/test_mobilenetv2_classification.py`. Report:
`data/logs/reviewed/mobilenetv2_classification_report.txt`.

## Candidate tower tested and mostly rejected: ViT-Base/16, ImageNet-21k (2026-08-07)

Tested `google/vit-base-patch16-224` (timm tag `vit_base_patch16_224.orig_in21k`
specifically - Google's original checkpoint, ImageNet-21k pretrained only,
NOT timm's own "augreg" retrain and NOT fine-tuned down to 1k) as another
candidate new tower, using the identical methodology, same reference/test
split, and same seed (42) as the MobileNetV2 test above, for direct
comparability.

**Result: 54.4% overall (92/169) - statistically identical aggregate to
MobileNetV2**, despite ViT-21k being a much larger model (86M params) with
21,843 pretraining classes versus MobileNetV2's 1000:

| Category | ViT-21k | MobileNetV2 |
|---|---|---|
| website_screenshot | 100% (30/30) | 100% (30/30) |
| handwritten_ledger | 84.6% (11/13) | 69.2% (9/13) |
| printed_document | 70.0% (21/30) | 56.7% (17/30) |
| mixed_text_image | 66.7% (4/6, small sample) | 83.3% (5/6, small sample) |
| dense_tabular_rows | 36.7% (11/30) | 43.3% (13/30) |
| map_land_record | 36.7% (11/30) | 43.3% (13/30) |
| portrait_photo | 13.3% (4/30) | 16.7% (5/30) |
| genealogy_chart | no held-out test data (only 6 images total on disk) | (same) |

**Same failure pattern as MobileNetV2, not a different one**: portrait_photo
is still the worst category by far (this time scattered mostly into
mixed_text_image and website_screenshot rather than concentrated in one
bucket - 15 of 30 went to mixed_text_image, 10 of 30 to website_screenshot),
and map_land_record/dense_tabular_rows/handwritten_ledger still bleed into
each other. website_screenshot is again a clean 100%.

**Verdict: also not a strong general voter, and the near-identical
aggregate score across two very different architectures (small supervised
CNN vs. large ViT, 1k vs. 21k pretraining classes) is itself informative**
- it suggests the ~54% ceiling reflects a limit of generic ImageNet-style
object-classification pretraining applied to this document-type taxonomy,
not a weakness specific to either model. The already-qualified 8 towers
(self-supervised or document/vision-language-pretrained, per the Multi-
Tower Routing Audit) remain the right fusion voters; ImageNet-object
classifiers of this style are a dead end for that role. The
`website_screenshot`-at-100% pattern replicating on a second, architecturally
unrelated model strengthens the case for a cheap `is_screenshot` pre-filter
gate (either model would do; MobileNetV2 is cheaper to run for that narrow
purpose).

Script: `diagnostics/test_vit_base21k_classification.py`. Report:
`data/logs/reviewed/vit_base21k_classification_report.txt`.

## Candidate sensor investigated and blocked: Teklia/pylaia-iam handwriting recognizer (2026-08-07)

Jon asked about `Teklia/IAM-line` as a handwritten-detection sensor. Two
corrections found before any test could run: (1) `Teklia/IAM-line` is a
**dataset** on HuggingFace, not a model - the actual model is
`Teklia/pylaia-iam`, a PyLaia (CRNN + CTC) line-level handwriting
transcriber trained on the IAM English handwriting dataset (CER 8.44%,
WER 24.51% without a language model); (2) it's not an embedding tower
like the timm-based candidates above - using it as a "sensor" would mean
a fundamentally different signal (transcription confidence on line crops
as a proxy for "is this handwriting"), and it expects line-level crops,
not full pages.

**Blocked at install**: PyLaia requires `nnutils-pytorch-cuda`, a custom
CUDA C++ extension with **no published wheel for any platform** (pip
cannot resolve a distribution at all, not just a Windows gap) - it would
need to be compiled from source. Given the two ImageNet-pretrained
towers already tested both plateaued around 54% as general voters, this
wasn't judged worth a source-build detour. **Status: not evaluated,
blocked by environment** - revisit if a Linux/WSL environment becomes
available, or substitute a plain-`transformers`-compatible HTR model
(e.g. TrOCR) if a "confident-transcription-as-handwriting-signal" sensor
is wanted later.

## Fine-tuned ViT-21k full-page document classifier - a different approach from zero-shot voting (2026-08-07)

Prompted by the PyLaia dead-end, Jon proposed a different direction:
rather than testing MORE frozen pretrained embeddings as zero-shot
voters (ceiling ~54% across two very different architectures, see
above), actually FINE-TUNE `vit_base_patch16_224.orig_in21k`'s
classification head (+ last 2 transformer blocks) directly on this
project's own 8-category taxonomy. This is a genuinely different
technique from every other candidate tested in this doc so far - all
prior evidence sources are frozen/zero-shot; this is the first
trained-for-this-task model.

**Dataset build** (`diagnostics/build_vit_finetune_dataset.py`): started
from an idea to reuse `data/outputs/lora_dataset/` (750 images, built for
Gemma LoRA extraction fine-tuning) plus the IAM dataset, but investigation
found neither fit this task cleanly:
- `lora_dataset` is **per-FIELD cell crops** (Name/Age/Sex/Birthplace/
  Relationship, short text, unusual aspect ratios), not page-level images
  with a document-type label - every image in it is implicitly the same
  category (handwritten census field), so it can't teach class
  separation, and mixing crop-scale images with full-page images would
  let a classifier cheat on image-size/aspect-ratio rather than learn
  real content signal.
- IAM line crops have the identical crop-scale problem versus full pages.
  **Both were excluded from this run** on Jon's explicit call - candidates
  for a separate line-level/binary handwriting sensor later, not this
  full-page 8-way classifier.

Instead, Jon had a **different, page-level census image set** already
pulled from LAC (Library and Archives Canada): 522 raw images across 7
batch folders (1906/1921/1926/1931 + 3 undated batches), which Jon
hand-reviewed and marked 38 as blank scanner-bed captures or title cards
(`data/outputs/lac_census_pull/lac_pull ground_truth.txt`). The build
script excludes those 38, keeping **484 real census page images**, added
as additional `dense_tabular_rows` examples on top of
`manifest_final.csv`'s existing 168, alongside all 7 other categories
exactly as manifest_final.csv already has them. Final combined dataset
(`data/outputs/vit_finetune_dataset.csv`): 2233 images -
`printed_document` 1169, `dense_tabular_rows` 652, `website_screenshot`
176, `map_land_record` 95, `portrait_photo` 86, `handwritten_ledger` 28,
`mixed_text_image` 21, `genealogy_chart` 6 (severe imbalance is real and
unresolved, not a bug - addressed with inverse-frequency class weighting
in the loss, not oversampling).

**Training setup** (`diagnostics/train_vit21k_document_classifier.py`):
froze the full ViT-21k backbone except the last 2 transformer blocks +
final norm + a fresh 8-way head, trained 8 epochs, AdamW with a much
lower LR on the unfrozen backbone (1e-5) than the fresh head (1e-3),
class-weighted cross-entropy for the imbalance, light augmentation only
(occasional horizontal flip - deliberately conservative since these are
scanned documents, not natural photos where heavy color/rotation jitter
would help).

**First-pass result (val used for both model selection AND reported
accuracy - optimistic, flagged immediately)**: 90.4% (301/333), a large
jump over the frozen zero-shot version of the identical checkpoint
(54.4%) - actual fine-tuning clearly does something the cosine-similarity
voting approach can't. Per-category: `website_screenshot` 100%,
`map_land_record` 100%, `dense_tabular_rows` 93.8%, `printed_document`
88.0%, `portrait_photo` 83.3% - all read as real given reasonable val
counts (12-175 examples). `handwritten_ledger`/`mixed_text_image`/
`genealogy_chart` scores (75%/67%/50%) are NOT reliable - only 2-4 val
examples each, essentially noise.

**Held-out test pass** (`diagnostics/train_vit21k_document_classifier_
holdout.py`), run immediately after to get a cleaner number: identical
dataset/hyperparameters/architecture, but a genuine 3-way stratified
split - val used ONLY for picking the best checkpoint during training,
a separate test set never touched until one final evaluation pass on the
winning checkpoint only.

**Held-out test result: 89.8% (298/332)** - barely below the optimistic
90.4% val-only number, which is the important finding here: if the model
were meaningfully overfit to the val set used for checkpoint selection,
the genuinely untouched test set should have scored noticeably lower.
It didn't (a 0.6-point gap), so 90.4%/89.8% can both be trusted as
approximately the model's real performance, not a val-selection
artifact.

| Category | Held-out test | (earlier val-only) |
|---|---|---|
| website_screenshot | 100% (26/26) | 100% |
| map_land_record | 100% (14/14) | 100% |
| dense_tabular_rows | 92.8% (90/97) | 93.8% |
| printed_document | 87.4% (153/175) | 88.0% |
| handwritten_ledger | 75.0% (3/4, tiny) | 75.0% |
| portrait_photo | 75.0% (9/12) | 83.3% |
| mixed_text_image | 66.7% (2/3, tiny) | 66.7% |
| genealogy_chart | 100% (1/1, tiny) | 50.0% |

The big/well-sampled categories (`website_screenshot`, `map_land_record`,
`dense_tabular_rows`, `printed_document`) are consistent to within a
couple points between val and test - genuinely validated. The tiny
categories (`handwritten_ledger` n=4, `mixed_text_image` n=3,
`genealogy_chart` n=1) swing around between runs as expected from such
small samples - still not a reliable measurement, unchanged conclusion
from the first pass.

**Confusion matrix read**: `printed_document` is the main source of
cross-category leakage in both directions - 4 dense_tabular_rows and 4
handwritten_ledger images misclassified as printed_document, and
conversely printed_document itself loses 22 of 175 test images spread
across most other categories (4 to dense_tabular_rows, 4 to
handwritten_ledger, 2 to map_land_record, 5 to mixed_text_image, 1 to
portrait_photo, 6 to website_screenshot) - printed_document is the
taxonomy's catch-all/most-visually-diverse bucket, so this is the
expected failure mode, not a surprise.

**Overall verdict**: fine-tuning is a clearly different, and clearly
better, approach than every zero-shot embedding-tower candidate tested
in this doc (MobileNetV2 54.4%, ViT-21k zero-shot 54.4%, this same
checkpoint fine-tuned 89.8% held-out). This is not a fusion-ensemble
VOTER like the 8 qualified towers - it's a candidate standalone
classifier or a strong additional evidence source in its own right,
worth considering as a real alternative/supplement to trusting Gemma
alone for the top-level category, not just another minor voting input.
Caveats before treating 89.8% as production-ready: (1) no true test set
independent of `manifest_final.csv`/the LAC census pull - both are
project-internal, not a wholly separate blind set; (2) three categories
remain unvalidated due to tiny sample sizes; (3) [RESOLVED, see ablation
below] `dense_tabular_rows`'s strong score is partly explained by simply
having 4x more training data now (484 new images) - couldn't originally
separate "fine-tuning helps" from "more data helps" as the dominant
driver.

### Ablation: fine-tuning vs. extra data, which one is actually driving the result? (2026-08-07)

Ran the identical fine-tune technique (same hyperparameters, same 3-way
held-out split methodology, same seed) with the 484 LAC census pull
images REMOVED - `dense_tabular_rows` trained on `manifest_final.csv`'s
original 168 examples only, same as every other category gets no
supplement. Separate dataset CSV
(`data/outputs/vit_finetune_dataset_ablation_no_census_pull.csv`) and
separate checkpoint file (`ablation_no_census_pull.pt`) - the main run's
`best.pt`/`best_holdout_eval.pt` were explicitly left untouched per
Jon's instruction.

**Result: 86.5% overall held-out (225/260)**, `dense_tabular_rows`
specifically 88.0% (22/25) - vs. 92.8% (90/97) with the extra 484 images.

This cleanly separates the two effects:

| | dense_tabular_rows | Overall |
|---|---|---|
| Zero-shot (frozen, no fine-tune) | 36.7% | 54.4% |
| Fine-tuned, NO extra data (this ablation) | 88.0% | 86.5% |
| Fine-tuned, WITH 484 extra census pages | 92.8% | 89.8% |

**Fine-tuning itself is overwhelmingly the dominant driver** (+51.3
points on dense_tabular_rows from technique alone, zero-shot → fine-
tuned-no-extra-data). The extra 484 images contribute a real but much
smaller further gain on top (+4.8 points on that category, +3.3 points
overall) - worth keeping, not worth over-crediting. This resolves caveat
(3) above: the 89.8% full-data result is legitimately mostly a
fine-tuning effect, not a data-volume artifact.

Script: `diagnostics/train_vit21k_ablation_no_census_pull.py`. Report:
`data/logs/reviewed/vit21k_ablation_no_census_pull_report.txt`.
Checkpoint: `data/outputs/vit21k_doc_classifier_checkpoints/
ablation_no_census_pull.pt` (kept separate from the main-run
checkpoints, as instructed).

Scripts: `diagnostics/build_vit_finetune_dataset.py`,
`diagnostics/train_vit21k_document_classifier.py`,
`diagnostics/train_vit21k_document_classifier_holdout.py`. Reports:
`data/logs/reviewed/vit21k_finetune_report.txt`,
`data/logs/reviewed/vit21k_finetune_holdout_report.txt`. Checkpoints:
`data/outputs/vit21k_doc_classifier_checkpoints/best.pt` (val-only
selection run), `best_holdout_eval.pt` (3-way split run).

## Multi-architecture sensor-qualification benchmark (2026-08-07)

Following the ViT-21k fine-tune result and ablation (which established
that fine-tuning itself, not the extra 484 census images, was the
dominant driver of that model's jump from 54.4%→89.8%), Jon asked
whether this generalizes: does domain-adaptation fine-tuning help OTHER
architecture families as much, and - more importantly for the fusion-
engine goal - do different families retain DIFFERENT error patterns
after fine-tuning (the property that makes them complementary voters
rather than redundant ones)? Framed explicitly by Jon as sensor
qualification, not a leaderboard: "the goal is NOT simply to find the
highest accuracy... two models with similar accuracy but different
confusion matrices are more valuable to the fusion engine than two
models making identical mistakes."

**Controlled-benchmark design** (`diagnostics/vit_family_benchmark_
common.py`, `run_vit_family_benchmark.py`, `analyze_vit_family_
benchmark.py`): held IDENTICAL across every architecture - dataset
(`data/outputs/vit_finetune_dataset.csv`, the same file used for the
successful ViT-21k run), the exact 3-way stratified split (same seed 42,
same code, copied not re-derived), class-weighted cross-entropy loss,
augmentation (occasional flip only), epoch count (8), batch size,
optimizer family (AdamW, head LR 1e-3 / backbone LR 1e-5), and a
one-time-only held-out test evaluation on the val-selected checkpoint.
The only thing that legitimately varies per architecture is WHICH layers
get unfrozen, because the families are structurally different (verified
via real inspection before writing any training code, not guessed):
isotropic ViT-style models (dinov2/beit/siglip) unfreeze the last 2
`.blocks` + norm + head; Swin unfreezes its last hierarchical `.layers`
stage + norm + head (its "blocks" are nested inside stages, not a flat
list); ConvNeXt unfreezes its last CNN `.stages` + head; MobileNetV2
unfreezes its last CNN stage + conv_head + classifier. A pre-flight
smoke test (one forward+backward step per architecture) confirmed all
six build and train correctly before committing to the full multi-hour
run.

**Models tested** (one representative per family, matching the project's
own already-qualified `QUALIFIED_ENCODERS` tags in
`core/vision_embeddings.py` where possible): `convnext_tiny.fb_in22k`
(modern CNN), `mobilenetv2_100.ra_in1k` (lightweight CNN),
`vit_small_patch14_dinov2.lvd142m` (self-supervised transformer, DINOv2),
`beit_base_patch16_224.in22k_ft_in22k` (self-supervised transformer,
masked-image-modeling), `swin_base_patch4_window7_224.ms_in22k`
(hierarchical transformer), `vit_base_patch16_siglip_224.v2_webli`
(vision-language encoder). ViT-21k itself was NOT re-run (already fully
benchmarked in the immediately preceding work) - its known numbers are
folded into the comparison table, with the caveat that its zero-shot
number specifically comes from an earlier, differently-partitioned test
(noted explicitly, not silently treated as equivalent).

### Comparison table: zero-shot vs. fine-tuned

| Model | Zero-shot | Fine-tuned (held-out) | Abs. improvement | Params | Speed (img/s, batch=1) | Train time |
|---|---|---|---|---|---|---|
| siglip | 64.2% | **92.5%** | +28.3 pts | 92.9M | 91.7 | 32.3 min |
| beit | 65.7% | 91.3% | +25.6 pts | 85.8M | 106.5 | 32.2 min |
| dinov2 | 64.5% | 91.0% | +26.5 pts | 22.1M | 57.3 | 36.7 min |
| convnext | 67.8% | 90.7% | +22.9 pts | 27.8M | 98.5 | 29.6 min |
| swin | 68.4% | 89.8% | +21.4 pts | 86.8M | 34.1 | 32.6 min |
| vit21k | 54.4%* | 89.8% | +35.4 pts* | ~86M | n/a | n/a |
| mobilenetv2 | 59.6% | 76.5% | +16.9 pts | 2.2M | 128.1 | 31.6 min |

*vit21k's zero-shot number used a different train/test partition than
this benchmark's shared split - direction/magnitude is consistent with
the others but not a strictly apples-to-apples percentage-point
comparison; its fine-tuned number does share this benchmark's exact
split.

**Every architecture generalizes the core finding**: fine-tuning moved
every model from a 54-68% zero-shot ceiling to 76-93% - domain adaptation
is not a ViT-21k-specific effect, it's a property of fine-tuning itself
on this taxonomy, consistent across CNN, hierarchical-transformer,
isotropic-transformer, and vision-language-pretrained architectures.
**MobileNetV2 is the clear outlier, and not just numerically**: its
gain is real but far smaller (+16.9 pts vs. +21-35 pts for everyone
else), which lines up directly with it having by far the least unfrozen
capacity to adapt (0.9M trainable params vs. 3.6-27.3M for the others) -
not a flaw in the method, an expected consequence of its size.

`dense_tabular_rows` improved for every model (+6-9 points among the
newly-benchmarked six; vit21k's own +56-point jump on this category is
explained separately by the ablation - extra census data compounding
with fine-tuning, not directly comparable to this benchmark's numbers
since those six models trained on the SAME dense_tabular_rows data as
vit21k's full run, i.e. they should show a similar large jump too if the
ablation's finding generalizes - worth a follow-up note: these six
models' dense_tabular_rows numbers (85.6%→85.6-96.9% zero-shot-to-
fine-tuned) are noticeably smaller improvements than vit21k's, because
their zero-shot baselines here were computed on THIS benchmark's own
train/test partition with a 15-image reference set, not the standalone
zero-shot test's methodology - the absolute fine-tuned numbers land in
the same 90-97% range across all seven models regardless, which is the
more load-bearing comparison). `printed_document` improved substantially
for every model except vit21k (+17.4 pts) which improved far less than
the six newly-tested models (+25.7 to +42.9 pts) - printed_document
being the taxonomy's largest, most visually diverse bucket, this
suggests some architectures generalize across its internal diversity
better than others even before controlling for anything else.

### Complementarity analysis - the part that actually matters for fusion

**Prediction agreement matrix** (fraction of the 332 held-out test
images where both models predicted the same label, right or wrong):
mobilenetv2 stands apart from every other model (76.5-77.7% agreement
with the rest) while the other five form a tightly correlated cluster
(91.0-93.7% agreement with each other). This alone suggests mobilenetv2
is doing something structurally different, not just "worse" - and
that convnext/dinov2/beit/swin/siglip are largely seeing the same
signal and reaching the same conclusions.

**Error overlap** (of images where model A was wrong, fraction where
model B was ALSO wrong) confirms this directly: mobilenetv2's error
overlap with the other five is dramatically LOWER (21.8-30.8%) than
their overlap with each other (50.0-75.9%). In plain terms: when
mobilenetv2 gets something wrong, the other five models are usually
still right (69-78% of the time) - genuinely complementary failures, not
shared blind spots. The five high-accuracy models, by contrast, tend to
fail on the same images as each other (my_error → your_error roughly
half to three-quarters of the time) - closer to redundant.

**Same-wrong-label overlap** (not just "both wrong" but "predicted the
exact same incorrect category") tells a similar story with more
resolution: mobilenetv2 vs. the others is 10.3-15.4%, versus 32-68%
among the five correlated models. Within that correlated cluster, siglip
stands out as relatively more distinct (33.3-45.2% same-wrong-label
overlap with the others, on the low end of that cluster's range) despite
having the HIGHEST fine-tuned accuracy (92.5%) - a genuinely useful
combination: siglip isn't just "another strong model," it's a strong
model that fails somewhat differently from the rest of the cluster.

**Unique-correct counts** (held-out images this model alone got right,
every other model got wrong): siglip leads with 4, mobilenetv2 has 3
despite its much lower overall accuracy (76.5%), dinov2 and swin each
have 1, convnext and beit have 0. MobileNetV2 contributing 3 unique-
correct cases despite being the weakest model overall is exactly the
"modest accuracy, genuinely complementary" pattern Jon's framing was
looking for - raw accuracy alone would have suggested dropping it, but
it's catching real cases nothing else catches.

### What this benchmark answers

- **Which architectures benefit most from domain adaptation?** All of
  them, roughly equally (+21 to +35 points) except MobileNetV2 (+17
  points) - the effect is general, MobileNetV2's smaller capacity is
  the one real exception and it's explainable, not mysterious.
- **Which architectures remain complementary after training?**
  MobileNetV2 clearly - lowest agreement, lowest error overlap, and real
  unique-correct contributions despite modest accuracy. Within the
  high-accuracy cluster, siglip shows the most distinct error pattern
  (lowest same-wrong-label overlap combined with the highest accuracy) -
  a plausible second complementary pick, not just "the best one."
  convnext/dinov2/beit/swin largely echo each other's mistakes (high
  same-wrong-label overlap, 32-68%) - adding more than one or two of
  this correlated group to a fusion ensemble would likely add cost
  without much new signal.
- **Which models deserve to become permanent qualified sensors?**
  Provisional read, pending Jon's decision (nothing here is implemented
  or wired into the pipeline): siglip (highest accuracy + relatively
  distinct errors) and mobilenetv2 (cheap, fast, genuinely
  complementary failure pattern, real unique-correct contribution) look
  like the strongest two-sensor combination from this batch. Adding a
  third from the correlated cluster (dinov2/beit/convnext/swin) would
  need justification beyond raw accuracy, since they mostly fail
  together already.

**Caveats, stated plainly**: (1) six architectures is a first pass, not
exhaustive - other members of each family (e.g. a larger DINOv2, a
different Swin variant) weren't tested; (2) the complementarity analysis
covers only the 332-image held-out set from this project's own data -
generalization to genuinely new, unseen document scans is untested;
(3) MobileNetV2's fine-tune only unfroze 0.9M params - an open question
whether unfreezing more of it would close its accuracy gap while
preserving its complementary error pattern, or whether the complementary
pattern would disappear once it's given more capacity to match the
others; (4) this whole analysis is about PREDICTION diversity on
top-level category, not about whether combining these models via actual
fusion (not yet built) would improve on Gemma's already-strong solo
performance - that remains the real open question this doc's earlier
sections lay out, not answered by this benchmark alone.

Scripts: `diagnostics/vit_family_benchmark_common.py` (shared dataset/
split/training/profiling machinery), `diagnostics/run_vit_family_
benchmark.py` (trains all 6 architectures), `diagnostics/analyze_vit_
family_benchmark.py` (comparison table + pairwise agreement/error-
overlap analysis). Reports: `data/logs/reviewed/vit_family_benchmark_
<arch>_report.txt` (one per architecture) and `vit_family_benchmark_
analysis.txt` (cross-model comparison). Checkpoints: `data/outputs/
vit_family_benchmark/<arch>/<arch>_best_holdout_eval.pt` (six
architecture-specific folders, `data/outputs/vit21k_doc_classifier_
checkpoints/best.pt` and `best_holdout_eval.pt` untouched, per Jon's
instruction).

## Decision-Engine Sensor Complementarity Analysis (2026-08-07)

Jon's explicit framing for this stage: **not a model-selection
exercise.** The project's architecture is a multi-source decision
engine where different sensors contribute different KINDS of evidence,
not competing votes on the same question. This section characterizes
complementarity, disagreement behavior, confidence usefulness, and each
sensor's potential ROLE - not a ranking. Evidence boundaries are marked
throughout: **[A]** = measured result or existing implementation,
**[B]** = interpretation supported by measurements, **[C]** = hypothesis
requiring a further experiment.

### 1. Generic pairwise conditional-disagreement analysis [A]

Ran on all 21 pairs among the 7 fine-tuned models from the previous
benchmark, using the SAME 332-image held-out test set for every pair
(`diagnostics/analyze_sensor_pairwise_conditional.py`, reading full
per-image logit vectors extracted from the already-saved checkpoints via
`diagnostics/extract_vit_family_logits.py` - no retraining, no new
inference beyond one forward pass per image per model). For every pair:
both-correct / A-correct-B-wrong / A-wrong-B-correct / both-wrong (split
into same-predicted-label vs. different-predicted-label) /
disagreement rate / each model's accuracy conditional on disagreement,
plus a per-category breakdown for every category (tiny-N categories
reported, not hidden, explicitly marked `[UNRELIABLE - n<5]`).

**Headline pattern**: disagreement rate correlates strongly with which
"cluster" a pair spans. Pairs entirely within the high-accuracy cluster
(convnext/dinov2/beit/swin/siglip/vit21k) disagree only 6-10% of the
time. Any pair involving MobileNetV2 disagrees 22-24% of the time -
roughly 3x more often - consistent with the earlier agreement-matrix
finding, now confirmed at the individual-pair level with full
conditional detail. Full table: `data/logs/reviewed/sensor_pairwise_
conditional_analysis.txt` / `.json`.

**Within the high-accuracy cluster**, when two models DO disagree, the
higher-solo-accuracy model wins conditional-on-disagreement more often
but not overwhelmingly - e.g. siglip vs. vit21k: siglip 63.0% correct
on their 27 disagreement cases vs. vit21k 29.6%; swin vs. siglip: siglip
63.0% vs. swin 29.6%. This is weaker separation than raw accuracy alone
would suggest and is itself useful: when two strong models disagree,
naively trusting "whichever usually scores higher" is directionally
right but leaves real cases on the table either way.

### 2. Detailed SigLIP <-> MobileNetV2 analysis [A]

Requested as the starting pair. Of 332 held-out images:
- **MobileNet correct / SigLIP wrong: 8 cases**
- **SigLIP correct / MobileNet wrong: 61 cases**

The imbalance is expected given the 16-point accuracy gap (92.5% vs.
76.5%), but the 8 MobileNet-rescue cases are the actually interesting
ones. By ground truth: 3 `printed_document`, 2 `dense_tabular_rows`, 1
each `mixed_text_image`/`handwritten_ledger`/`website_screenshot` - no
single dominant category, spread thin. On those 8, SigLIP's mistaken
predictions were varied too (`printed_document`, `website_screenshot`
x2, `handwritten_ledger`, `portrait_photo`, `map_land_record`,
`dense_tabular_rows`) - **no evidence of a single systematic SigLIP
blind spot MobileNet is specifically catching**; the rescues look more
like scattered independent noise than a coherent pattern **[B]**. Full
per-case file listing (paths, ground truth, both models' predicted
label + margin + softmax + entropy for every one of the 69 disagreement
cases): `data/logs/reviewed/siglip_vs_mobilenet_disagreement_cases.txt`.

On the 61 cases where MobileNet is wrong and SigLIP is right, MobileNet's
mistaken predictions cluster heavily on `handwritten_ledger` (15),
`dense_tabular_rows` (14), and `printed_document`/`mixed_text_image`
(11 each) - **[B]** consistent with MobileNet's earlier-noted weak
capacity (only 0.9M trainable params after fine-tuning) causing it to
over-predict the taxonomy's more "textured/dense" categories broadly,
rather than making one specific, narrow, avoidable mistake.

**Does MobileNet's own margin distinguish its rare correct-override
cases from its ordinary wrong cases?** Mean margin on the 8 override
cases (2.463) is meaningfully higher than on its other 78 wrong
predictions (1.285) - and a simple threshold at the midpoint of the two
medians separates 6/8 override cases above it vs. only 28/78 ordinary-
wrong cases above it. **[B] Some real separability visible**, but n=8
is far too small to trust as a standalone rule - this is exactly the
kind of finding that needs a larger disagreement set before it could
justify anything like "let MobileNet override SigLIP when its margin
exceeds X" **[C]**.

### 3. Confidence / margin / entropy analysis [A]

For every model, correct vs. incorrect predictions separate cleanly on
ALL THREE metrics (top1-top2 logit margin, top1 softmax probability,
entropy) - e.g. siglip: correct-case margin mean 8.19 vs. incorrect-case
mean 1.94; entropy mean 0.09 (correct) vs. 0.63 (incorrect). This holds
for every architecture, not just the strong ones - even MobileNetV2
(correct margin mean 4.00 vs. incorrect 1.29) shows the same directional
separation, just compressed (smaller absolute gap, matching its smaller
overall unfrozen capacity). **[B] Raw logit margin is a real, usable
per-image confidence signal across every architecture tested**, echoing
the earlier finding for Gemma's own decision-token logit gap (see
Gemma comparator section below) - this appears to be a general property
of how these models express uncertainty, not something specific to one
architecture or to autoregressive generation.

Full per-model correct/incorrect statistics (mean/median/stdev for all
three metrics): `data/logs/reviewed/sensor_pairwise_conditional_
analysis.txt`.

### 4. Gemma comparator [A]

**UPDATE (2026-08-07, same day, later pass): the same-split run
happened.** The paragraph below was the original "not yet done, here's
why" framing - preserved for the reasoning trail, but superseded by the
real result immediately after it. Jon asked whether the production
prompt (`config/prompts/classifier_classify_v1.txt`, THIS project's
version - confirmed via diff against the sibling `genealogy_pipeline -
Main` checkout's older hardcoded variant that this one is current,
dynamically taxonomy-templated, and includes the website_screenshot
pre-check) would help; the answer was to use exactly that, since it's
what Stage 5 actually runs in production, not a research variant.
`diagnostics/run_gemma_on_benchmark_holdout.py` calls the real
`core/classifier.py` production path (`build_classifier_loader()` +
`loader.classify()`) on the identical 332-image held-out split the 7
vision towers were benchmarked on.

**[A] Gemma result: 93.7% (311/332) - the highest accuracy of every
sensor tested in this entire benchmark**, ahead of siglip's 92.5%.
Perfect on `dense_tabular_rows` (97/97), `handwritten_ledger` (4/4),
`map_land_record` (14/14), `genealogy_chart` (1/1). Weaker on
`portrait_photo` (50.0%, 6/12), `website_screenshot` (80.8%, 21/26),
`mixed_text_image` (66.7%, 2/3) - **a visibly different weakness profile
than the vision towers**, which were uniformly strong on
`website_screenshot` (88.5-100%) and weaker on `dense_tabular_rows`
(85.6-97% zero-shot-to-fine-tuned range, never a clean 100%). Cost:
6.43s/image - roughly 200-1500x slower than any vision tower's batch=1
throughput (34-128 img/s, i.e. <0.03s/image).

**[A] Direct answers to the four open questions this section originally
posed**, via `diagnostics/analyze_gemma_vs_towers.py` on the same
predictions:

- **When Gemma and a tower disagree, who's right more often?** Gemma,
  decisively, against EVERY one of the 7 towers - accuracy-conditional-
  on-disagreement ranges 50.0-78.4% for Gemma vs. 13.6-38.9% for the
  tower it's disagreeing with. Most extreme vs. mobilenetv2 (Gemma
  78.4% vs. mobilenetv2 13.6% on their 88 disagreement cases) but true
  even against siglip, the strongest tower (Gemma 50.0% vs. siglip
  38.9% on their 36 disagreement cases).
- **Does Gemma catch cases no tower catches?** Yes - **6 images where
  Gemma is correct and all 7 towers are wrong** (spread across
  `portrait_photo`, `dense_tabular_rows` x2, `mixed_text_image`,
  `printed_document` x2 - no single category pattern). Compare: across
  all 7 towers combined, only **2 images** exist where a tower is
  correct and Gemma AND every other tower are wrong (1 mobilenetv2, 1
  siglip). Gemma's unique-correct contribution is 3x larger than all 7
  towers' combined unique-correct contribution.
- **Are there cases where EVERYTHING fails?** Yes, 3 - all
  `printed_document` ground truth, each tower and Gemma predicting a
  different wrong category. `printed_document`'s already-noted role as
  the taxonomy's most diverse/catch-all bucket (see the "Confusion
  matrix read" note in the earlier fine-tune section) is confirmed here
  too, at the whole-ensemble level, not just per-model.
- **Does Gemma's cost buy genuinely independent errors, or does it fail
  on the same images the towers do?** Genuinely independent. `both_wrong
  same_label` (Gemma and the tower both wrong AND predicted the SAME
  incorrect category) is a small fraction of their combined wrong cases
  for every pairing (2-5 images per tower pair) - when Gemma is wrong,
  it's usually wrong in a DIFFERENT way than a given tower's wrong
  answer, not the same mistake twice.

**[A] Gemma's self-reported confidence field, checked directly against
correctness on this run**: correct-case mean 0.981 vs. incorrect-case
mean 0.956 - barely separated, confirming (now on a THIRD independent
sample, after the two disagreement-set/gate experiments earlier in this
doc) that the production confidence field is a weak signal, consistent
with this project's standing finding that raw decision-token logit gap
is the more useful metric - **[C] not directly measurable from the
production `classify()` call as built today** (would need custom
instrumentation inside that call path, not just this benchmark, to
capture logits the way the earlier standalone experiments did).

**[B] Bottom line**: on this benchmark's own held-out set, Gemma is not
just "also accurate" - it's the single best-performing sensor tested,
AND it demonstrably fails differently from the fine-tuned towers
(low same-wrong-label overlap, real unique-correct contribution 3x the
towers' combined total). This is real evidence FOR keeping Gemma as the
primary/production classifier rather than only an escalation sensor -
a genuinely different conclusion than the pre-fine-tuning-era framing
(built when towers were frozen and scored ~54-68%) which is why this
same-split experiment mattered. **[C] still open**: whether a fusion
layer combining Gemma with 1-2 complementary towers (e.g. as a dissent
check, per the hierarchical sketch below) would push past 93.7%, or
whether Gemma alone is already close enough to a practical ceiling on
this taxonomy that the added complexity isn't worth it - not answerable
without actually building and testing a fusion rule, which remains
explicitly out of scope for this analysis phase.

Script: `diagnostics/run_gemma_on_benchmark_holdout.py` (production-path
Gemma inference on the shared split), `diagnostics/analyze_gemma_vs_
towers.py` (pairwise conditional analysis + unique-correct + confidence
check). Predictions: `data/outputs/vit_family_benchmark/gemma/
gemma_test_predictions.json`. Reports: `data/logs/reviewed/gemma_on_
benchmark_holdout_report.txt`, `gemma_vs_towers_analysis.txt`.

---

**[Original framing, preserved for the reasoning trail]**: Gemma has NOT
been run on this exact 332-image held-out test set - doing so would mean
a real, GPU-heavy inference pass (Gemma inference is far slower than
these lightweight towers), and per Jon's instruction to reuse existing
data rather than rerun unnecessarily, that wasn't done as part of this
pass. Everything below is EXISTING, already-recorded Gemma behavior from
earlier work in this doc, on DIFFERENT samples/tasks - useful for
qualitative comparison and behavioral characterization, but NOT a
direct head-to-head on the same images.

**[A] What's already measured about Gemma, preserved as distinct
findings, not collapsed into "more/less accurate"**:
- On the 48-case tower/Gemma DISAGREEMENT set (cases specifically
  selected because Gemma and tower consensus disagreed - a hard,
  biased sample by construction, not representative of overall
  accuracy): Gemma correct 23/48 (47.9%), flat tower consensus 11/48
  (22.9%). Gemma beats tower consensus roughly 2x on the cases where
  they disagree.
- On a real, large-scale (n=368: 168 true positives + 200 negatives)
  binary `dense_tabular_rows` yes/no gate against
  `manifest_final.csv` ground truth: 88.6% overall (326/368), but with
  a real, ONE-DIRECTIONAL bias - 100% precision, 75% recall, every
  error a false negative (conservative: never wrongly claims tabular,
  misses a real quarter of genuine cases).
- Gemma's raw logit-gap-at-decision-token confidence signal is
  directionally real (median gap 30.41 correct vs. 13.56 wrong) but
  NOT cleanly separable at this scale - several false negatives were
  CONFIDENTLY wrong (gap 25-30), the worst failure mode for a
  confidence signal.
- Prompt architecture materially changes what Gemma gets right: asking
  "is this a manifest?" before "is this a census?" caused real census
  pages to get claimed as manifests; census-gate-first (the adopted
  v7 architecture) separated them better - a prompt-ordering effect
  with no equivalent in the vision towers tested here (they have no
  sequential-decision structure to order).
- Fixed output templates ("is_census: ") can encourage format-completion
  behavior; the first 3-4 tokens of that echo are perfectly
  deterministic (p=1.000000) - genuinely NOT part of the content
  decision, just format-following.
- Forcing the decision into generation token 0 (stripping the label-
  echo) made BOTH accuracy and the confidence signal WORSE - the
  "boring" echo tokens are functionally useful extra computation, not
  filler.
- Pre-generation (before any token is produced) yes/no tail-logit
  "lean" was only weakly predictive and directionally biased -
  essentially unusable as a cheap zero-generation shortcut.
- The useful decision crystallizes DURING autoregressive generation,
  not before it and not in early format-echo tokens - confirmed three
  separate ways (trajectory dump, first-token reformulation, pre-
  generation lean check).

**[B] What this means in comparison to the vision towers tested here**:
Gemma's failure mode (conservative, one-directional, real precision/
recall tradeoff on a specific category) is qualitatively DIFFERENT from
any vision tower's failure mode observed in this benchmark - the towers
don't show a comparable directional bias in their `dense_tabular_rows`
per-category numbers (all land in a tighter 85-97% band with errors
going both directions in the confusion matrices, not one-directional).
This is suggestive of genuinely different error character between
Gemma and the fine-tuned towers, which is exactly the property that
would make Gemma valuable as an escalation/dissent sensor rather than
redundant with them - **but this is inference from separately-measured
behavior on different samples, not a same-image comparison, so it stays
a [B] interpretation, not a [A] measured fact.**

**[C] Open questions this analysis surfaces but cannot answer without a
same-split experiment**: What does Gemma get right that the trained
towers get wrong, on the SAME images? What do the towers get right that
Gemma gets wrong? When they disagree, does any observable evidence (CV
structural signals, tower consensus, Gemma's own logit gap) predict who
is right? Does Gemma's much higher inference cost buy genuinely
independent errors worth escalating to, or would a same-split run show
its errors substantially overlap with the towers' after all? None of
these are answerable from existing data - they require running Gemma
on this benchmark's actual 332-image held-out set.

### 5. Sensor value framework - dimensions kept separate, not collapsed [B]

Per Jon's explicit instruction not to prematurely collapse these into
one weighted score. Populated where measured; left explicitly blank
where this benchmark doesn't cover that sensor/dimension yet.

| Sensor | Accuracy (solo) | Complementarity (correct when others wrong) | Redundancy (echoes another sensor) | Confidence usefulness | Category specialization | Cost | Evidence type |
|---|---|---|---|---|---|---|---|
| SigLIP (fine-tuned) | Highest of the 7 (92.5%) | Real but modest - 8 unique-ish rescues vs. MobileNet, lowest same-wrong-label overlap within the strong cluster | Moderate-high vs. convnext/dinov2/beit/swin (91-94% agreement) | Strong - margin/entropy cleanly separate correct/incorrect | No single standout category; broadly strong | 92.9M params, 91.7 img/s, moderate | Semantic document-type (vision-language pretrained) |
| MobileNetV2 (fine-tuned) | Lowest of the 7 (76.5%) | Highest measured - lowest agreement/error-overlap with every other model, 3-8 unique-correct cases despite weak solo accuracy | Low - genuinely the most independent sensor in this set | Directionally real (correct/incorrect margins separate) but small-n; some evidence its margin flags its own rare correct-override cases, unconfirmed at scale | Weak generally; disproportionately bad on `handwritten_ledger`/`portrait_photo` | 2.2M params, 128.1 img/s (fastest), smallest GPU footprint | Semantic document-type, but from a much smaller/cheaper model - candidate cheap early-stage prior or dissent flag, not a primary classifier |
| BEiT (fine-tuned) | High (91.3%) | Not specifically measured beyond the generic pairwise table | High vs. the other strong-cluster models | Strong (same pattern as siglip/dinov2) | **0/4 on held-out `handwritten_ledger`** despite high overall accuracy - a real, specific category weakness worth flagging even though n=4 is tiny | 85.8M params, 106.5 img/s | Semantic document-type (masked-image-modeling pretrained) |
| DINOv2 (fine-tuned) | High (91.0%) | Not specifically measured beyond the generic pairwise table | High vs. the other strong-cluster models | Strong | Notably strong `map_land_record` (100%) and `dense_tabular_rows` (96.9%) | 22.1M params, 57.3 img/s (slowest of the strong cluster; 518x518 input) | Semantic document-type (self-supervised pretrained) |
| ConvNeXt (fine-tuned) | High (90.7%) | Not specifically measured beyond the generic pairwise table | High vs. the other strong-cluster models | Strong | Balanced, no standout weakness or strength found | 27.8M params, 98.5 img/s | Semantic document-type (modern CNN) |
| Swin (fine-tuned) | High (89.8%) | Not specifically measured beyond the generic pairwise table | High vs. the other strong-cluster models | Strong | Balanced | 86.8M params, 34.1 img/s (slowest overall) | Semantic document-type (hierarchical transformer) |
| ViT-21k (fine-tuned) | High (89.8%) | Not specifically measured beyond the generic pairwise table | High vs. the other strong-cluster models | Strong | Largest ablation-confirmed `dense_tabular_rows` gain (+51 points from fine-tuning alone) | ~86M params | Semantic document-type (isotropic ViT, ImageNet-21k pretrained) |
| Gemma (Stage 5, current production classifier) | **Highest of every sensor tested: 93.7% (311/332), same held-out split, production prompt/loader** | **Real and large: beats every tower's accuracy-conditional-on-disagreement (50.0-78.4% vs. towers' 13.6-38.9%), 6 unique-correct images vs. towers' 2 combined** | **Low - "both wrong, same predicted label" is a small fraction of combined errors against every tower**, genuinely different failure modes, not overlapping | Self-reported confidence field weak (0.981 vs 0.956 mean, correct vs incorrect) - confirmed on a third independent sample; raw logit gap (not measurable via the production call path) remains the more useful metric per earlier experiments | Perfect on `dense_tabular_rows`/`handwritten_ledger`/`map_land_record`/`genealogy_chart`; weaker on `portrait_photo` (50%)/`website_screenshot` (80.8%) - opposite weak spot from the vision towers | Much higher than any vision tower - 6.43s/image vs. towers' <0.03s/image (~200-1500x), full generative LM inference per image, not a single forward pass | Semantic + reasoning/generative - fundamentally different mechanism (autoregressive decision) from every tower above |
| DocLayout-YOLO (`core/layout_detector.py`) | Not benchmarked against this taxonomy at all | Unmeasured | Unmeasured | N/A - returns detections with per-box confidence, not evaluated here | Unmeasured; one smoke-test detection on a real `dense_tabular_rows` page returned a near-page-spanning `figure` box at 0.91 confidence - domain-transfer concern noted, not resolved | Real inference cost, currently captured but not consumed downstream | Object/region evidence - structured labeled bounding boxes, not a whole-image category |
| YOLOv26-small (`core/layout_detector_v26.py`) | Not benchmarked, explicitly NOT wired into the pipeline (research-only per Jon's instruction) | Unmeasured | Unmeasured vs. DocLayout-YOLO (different training source, different 11-class label set - not directly comparable) | Unmeasured | Unmeasured; DocLayNet-trained (modern/academic documents), an even bigger domain gap than DocStructBench for this project's aged/handwritten corpus, by design not yet tested here | Real inference cost | Object/region evidence, same shape as DocLayout-YOLO but a different model/training source |
| 8 QUALIFIED_ENCODERS frozen towers (`core/vision_embeddings.py`, `core/decision_engine.py`'s tower consensus) | 22.9% on the 48-case hard disagreement set (same set Gemma scored 47.9% on) | Beaten by Gemma ~2x on that set | MAE confirmed dead weight (5/48, uniquely rescued 0 cases) - real evidence one of the 8 adds nothing | Consensus category (unanimous/majority/split/complete_disagreement) recorded but not yet validated as a decision signal | Unmeasured on this project's benchmark taxonomy specifically (frozen, not fine-tuned - this whole multi-architecture benchmark exists because the frozen version plateaued at ~54-68%) | Cheap - reuses embeddings already captured at Stage 1, zero new inference | Semantic document-type via frozen nearest-cluster cosine similarity - a fundamentally cheaper, less accurate version of the fine-tuned towers above |
| `core/image_analysis.py` structural measurements (Stage A) | N/A - not a classifier | N/A | N/A - measures geometry/ink/structure, nothing else in this table observes the same thing | `table_confidence` specifically calibrated: real tables score min 0.375/median 1.0 vs. non-tables median 0.125 - a real, validated threshold | `table_confidence` is inherently `dense_tabular_rows`/census-specific evidence | Very cheap - pure CV measurement, no model inference | Geometric/structural evidence - orthogonal to every semantic tower above, measures the page's physical layout not its semantic content |
| `core/auto_sidecar.py` table/column/row boundary detection | N/A - not a classifier | N/A | N/A | Unmeasured as a confidence signal | Tabular-structure-specific by construction | Cheap CV | Tabular-structure evidence - geometric, orthogonal to semantic classification |
| `core/row_segmentation.py` printed-number geometric anchors | N/A - not a classifier | N/A | N/A | Row-axis version validated (real project use); column-axis version research-only, not wired into production | Census/manifest form-specific (relies on printed row/column numbers existing) | Cheap, dependency-free (PIL+numpy only) | Geometric anchor evidence - a distinct sub-type of structural evidence from ruling-line detection, exploits printed numbering instead of line geometry |

### 6. Hierarchical vs. flat-voting architecture - what the evidence actually supports [B/C]

This document is titled "voting classifier" but the measurements here
don't support literal flat majority voting as the right shape:

- **[B] MobileNetV2 looks like a plausible cheap early-stage/dissent
  signal, not a peer voter.** Its low agreement with the strong cluster,
  its fast/cheap inference (128 img/s, 2.2M params, smallest GPU
  footprint of anything benchmarked), and its handful of genuine
  unique-correct rescues are consistent with a role like "flag when this
  cheap model disagrees with the primary classifier, worth a second
  look" rather than "count its vote equally with SigLIP's." This is
  the shape Jon's hierarchical sketch describes
  (cheap structural evidence → trained classifier(s) → disagreement
  detection → specialist/dissent sensors → Gemma escalation where
  justified) - the MobileNetV2 data is consistent with that sketch,
  but **[C] whether MobileNet-disagreement-as-a-trigger actually
  improves end-to-end accuracy (vs. just adding cost) is untested** -
  this benchmark measured prediction diversity, not a working
  escalation rule.
- **[B] The 5-6 strong-cluster towers (convnext/dinov2/beit/swin/siglip/
  vit21k) are largely redundant with each other** (91-94% pairwise
  agreement, 32-68% same-wrong-label overlap) - stacking more than one
  or two of them into a fusion layer would likely add cost without much
  new signal, an argument for picking one or two representatives
  (siglip for accuracy + distinct-within-cluster errors) rather than
  including all of them.
- **[C] Whether YOLO/layout/structural CV signals can provide strong
  class-specific evidence without acting as complete classifiers is
  plausible in principle** (`table_confidence`'s real, calibrated
  threshold is exactly this shape - cheap, category-specific, not a
  general classifier) **but genuinely untested for the layout-YOLO
  sensors specifically** - both DocLayout-YOLO and YOLOv26 are captured
  but not yet evaluated against this project's own taxonomy at any
  real scale (one smoke-test detection each, not a benchmark).
- **[A] RESOLVED - Gemma's role is NOT "expensive escalation sensor
  only"**: the same-split run (see Gemma comparator section above)
  shows Gemma outright winning on accuracy (93.7%, best of every
  sensor tested, including the fine-tuned towers) AND showing
  genuinely low error-overlap with them. The original hierarchical
  sketch's placement of Gemma at the "expensive escalation, used
  sparingly" end of the pipeline is **not what this evidence
  supports** - Gemma looks more like it should stay the PRIMARY
  classifier, with towers (especially a fast/cheap one like
  MobileNetV2) potentially useful as a cross-check or dissent flag
  ON Gemma's output, rather than the reverse framing (towers primary,
  Gemma reserved for hard cases) the original conceptual sketch
  implied. **[C] still untested**: whether adding a tower-based dissent
  check actually catches any of Gemma's 21 real errors on this split in
  practice, or whether it would just add cost without changing the
  outcome - that requires actually building and testing the check, not
  just noting the towers exist.

### 7. Per-image evidence - what's preserved, what schema exists already [A/C]

Every held-out prediction from this benchmark retains full per-image
detail, not just a final label, so nothing needs to be rerun for future
analysis: `data/outputs/vit_family_benchmark/<name>/<name>_test_logits.json`
per model contains, per image, `{gt, pred, correct, logits (full vector,
all 8 classes), softmax (full vector), top1_logit, top2_logit,
top1_top2_margin, top1_softmax, entropy}`. This is real per-image
evidence, not aggregate statistics only.

**[A] Existing project schema partially supports the sensor →
observation → evidence pattern already**: `core/pipeline_db.py`'s
`stage_outputs` table is generic per-(image, stage) evidence pointer
(`stage`, `sidecar_path`, `status`, `note`), and `core/decision_
engine.py`'s tower-consensus write already follows an "evidence in
sidecars, coarse category on the row" split (`consensus_category` as a
queryable column, full per-encoder votes in a `<name>_tower_
consensus.json` sidecar) - this is close to the sensor →
observation/prediction → confidence/margin → evidence type shape Jon
described, but **[C] no existing table has explicit `confidence`,
`margin`, or `evidence_type` columns** - today that detail lives inside
each stage's own sidecar JSON schema, not a unified cross-sensor
schema. Building that unified schema is future work, not attempted
here per the instruction not to build the production schema yet.

### Recommendations for the next experiment (not implemented)

In priority order, based on what this analysis surfaced as the biggest
open gaps:

1. **[DONE, same day]** ~~Run Gemma on this exact 332-image held-out
   set~~ - completed (see Gemma comparator section above). Result:
   Gemma wins outright (93.7%, best of every sensor) with genuinely
   low error-overlap against the towers - a materially different
   conclusion than the pre-fine-tuning-era framing this doc started
   from, where Gemma vs. frozen tower consensus (~54-68%) was the only
   comparison available.
2. **[NEW, replaces the old #1] Test whether a cheap tower-based
   dissent check actually catches any of Gemma's 21 real errors on this
   split** - now that Gemma is confirmed the stronger sensor, the real
   open question is whether MobileNetV2 (or another tower) disagreeing
   with Gemma is a usable trigger to flag those 21 specific error cases
   for review, not whether towers should replace or gate Gemma broadly.
3. **Scale up the MobileNet-margin-as-override-signal check** (currently
   n=8 override cases vs. siglip, suggestive but not reliable) - either
   against a larger held-out set, or by pooling MobileNet's
   disagreements with every other tower AND with Gemma to get a bigger
   sample.
4. **Benchmark DocLayout-YOLO and YOLOv26 against this project's own
   taxonomy at real scale** (beyond the one-image smoke tests already
   on record) - if `table_confidence`-style category-specific evidence
   generalizes to the layout detectors' box outputs, that's a
   genuinely different evidence type from every semantic tower tested
   here, not redundant with any of them by construction.
5. **Investigate BEiT's 0/4 `handwritten_ledger` result** at a larger
   sample size before treating it as a real category weakness rather
   than n=4 noise.
6. Do NOT implement a fusion rule yet - every finding in this section
   is qualification data. The Gemma same-split comparison is now done
   and changes the shape of what a sensible fusion design would even
   look like (Gemma-primary-with-dissent-checks, not towers-primary-
   with-Gemma-escalation) - a real architectural decision, but still a
   decision for Jon to make explicitly, not to infer from this data
   alone.

Scripts: `diagnostics/extract_vit_family_logits.py` (full logit
extraction from existing checkpoints), `diagnostics/analyze_sensor_
pairwise_conditional.py` (generic pairwise + siglip/mobilenet deep dive
+ confidence/margin analysis). Reports: `data/logs/reviewed/sensor_
pairwise_conditional_analysis.txt` / `.json`, `data/logs/reviewed/
siglip_vs_mobilenet_disagreement_cases.txt`. Per-image logit data:
`data/outputs/vit_family_benchmark/<name>/<name>_test_logits.json`
(7 models, includes vit21k in a newly-created subfolder for this
purpose - its original checkpoint location is untouched).

## Gemma error-dissent analysis: can a cheap sensor flag Gemma's mistakes without over-flagging? (2026-08-07)

Reframed by Jon after the same-split Gemma result: the useful question
isn't "can another classifier beat Gemma" (nothing tested does,
decisively) - it's whether a cheap dissent sensor can flag a meaningful
fraction of Gemma's 21 real errors on this held-out set WITHOUT also
flagging too many of its 311 correct predictions. This is an
arbitration-mechanism question. `diagnostics/analyze_gemma_errors_
dissent.py`.

### A real confound found and separated out before drawing conclusions [A]

**7 of Gemma's 21 "errors" predicted `casual_photo`** - a category that
does not exist in the 8-class taxonomy the ground truth (`manifest_
final.csv`, pre-subtypes) and all 7 vision towers use. `casual_photo`
was added to the production taxonomy 2026-08-04 (`core/schema.py`,
alongside `photo_collage`/`cemetery_photo`), specifically for candid/
non-portrait personal photos, AFTER the ground truth this benchmark
scores against was built. The towers CANNOT predict `casual_photo` -
it's not in their label space - so these 7 cases aren't really "the
towers caught something Gemma missed," they're Gemma using a real,
valid, more specific category the evaluation has no way to credit.
5 were `portrait_photo` ground truth -> `casual_photo` (plausibly
correct given the category's actual definition - "distinct from
portrait_photo... isn't a document"), 2 were `printed_document` ->
`casual_photo` (less obviously defensible, but still outside what this
comparison can fairly judge). **This means Gemma's true error rate
against genuinely wrong content classification is likely lower than the
93.7%/21-error headline suggests** - but not provably higher without
manually reviewing those 7 images, so the honest position is: **14/21
are genuine, verifiable cross-category errors** (e.g.
`website_screenshot` -> `printed_document`/`dense_tabular_rows`,
`printed_document` -> `dense_tabular_rows`/`map_land_record`); the other
7 are flagged as a taxonomy-versioning artifact, not silently folded
into either "real error" or "real correct." All analysis below is
reported BOTH ways (all 21, and the 14 genuine-only) so this confound
doesn't get lost in an aggregate number.

### 1. Are Gemma's errors concentrated or spread across categories? [A]

Spread across 4 of 8 categories, but heavily weighted toward two:
`portrait_photo` (6/12 = 50% error rate within category - 5 of those 6
are the `casual_photo` taxonomy-mismatch cases, so really 1/12 genuine)
and `website_screenshot` (5/26 = 19.2%, all genuine). `printed_document`
has the most raw error count (9) but the lowest rate (5.1% of 175) -
its earlier-noted role as the taxonomy's largest/most diverse bucket
shows up again. `dense_tabular_rows`, `handwritten_ledger`,
`map_land_record`, `genealogy_chart` had ZERO errors - Gemma is
completely reliable on these four in this sample.

### 2. Per-error tower breakdown - was there consensus against Gemma? [A]

For each of the 21 errors, how many of the 7 towers were independently
correct: `{0: 3, 1: 2, 2: 3, 3: 1, 4: 1, 5: 2, 6: 1, 7: 8}`. Reading
this directly: **8 of 21 errors have ALL 7 towers unanimously correct**
(strong consensus against Gemma - these are the cases where an
arbitration rule should have the easiest time), but **3 of 21 have
EVERY tower also wrong** (no tower-based dissent signal is possible on
these AT ALL, by construction - a hard ceiling on what any tower-based
rule could ever catch on this sample).

**MobileNet was correct on 12/21 of Gemma's errors. SigLIP was correct
on 14/21.** Both real, useful hit rates - not close to unanimous, but
neither is noise. Individually: convnext 14/21, mobilenetv2 12/21,
dinov2 11/21, beit 13/21, swin 11/21, siglip 14/21, vit21k 12/21 - no
tower dominates, consistent with the earlier finding that the strong
cluster is largely redundant with each other (when one strong tower
catches a Gemma error, most of the others usually do too, per the 8/21
unanimous-consensus figure above).

### 3. Tower margin/entropy and Gemma's own confidence on these 21 cases [A]

No tower shows a dramatically different margin/entropy signature on
these specific 21 images versus its general behavior - margins in the
2.0-3.1 range, entropy 0.48-0.72, roughly mid-pack rather than
extreme-low-confidence. **This means towers are not obviously "unsure"
on the cases where they're catching Gemma's mistakes - when they're
right here, they tend to be right the same confident way they usually
are**, which is a point in favor of a margin-based tower-confidence
gate being viable (a tower flagging disagreement with real margin, not
a coin-flip, is more often than not actually right on this evidence).

Gemma's own self-reported confidence on these 21 errors: mean 0.956,
barely below its overall correct-case mean (0.981) - **confirmed on a
third independent sample that Gemma's production confidence field does
NOT reliably signal when it's wrong**, consistent with every earlier
finding in this doc about that field vs. raw logit gap. The raw logit
gap itself was not extracted here (the production `classify()` path
doesn't expose it without custom instrumentation) - flagged as a real
gap, not silently assumed equivalent to the confidence field.

### 4. Testing the proposed rule: Gemma=X, SigLIP disagrees, MobileNet independently agrees with SigLIP [A]

**On all 21 errors: 12/21 caught (57.1%) at 6/311 false positives
(1.9%).** **On the 14 genuine (non-taxonomy-mismatch) errors: 7/14
caught (50.0%) at the same 1.9% false-positive rate.** Either way this
reads as a real, meaningful signal, not noise: half or better of
Gemma's real mistakes flagged, while wrongly flagging fewer than 1 in
50 of its correct predictions. Full per-case listing (all 18 flags,
which are genuine catches vs. taxonomy-mismatch catches vs. false
positives): `data/logs/reviewed/gemma_errors_dissent_analysis.txt`.

A stricter variant (only count SigLIP's dissent if its own margin is
above its overall median) flagged ZERO images - **[B] SigLIP's
"confident enough to matter" disagreements with Gemma are rare enough
in this 332-image sample that a stricter margin gate over-filters
rather than improves precision; the simple binary "disagrees + MobileNet
agrees" rule outperforms the margin-gated version at this sample size**,
though [C] this could flip with a larger sample where SigLIP has more
high-margin disagreement cases to learn a real threshold from.

### Verdict [B]

**This is a real, promising arbitration signal, not a hand-designed
guess.** A cheap two-tower agreement rule (SigLIP disagrees with Gemma,
MobileNet independently backs SigLIP's alternative) catches half to
just-over-half of Gemma's genuine errors at under 2% false-positive
cost - exactly the "meaningful fraction without too many unnecessary
escalations" bar Jon set. This is the first result in this whole
analysis phase that looks like a genuinely usable BUILDING BLOCK for an
arbitration mechanism, rather than just a characterization finding.

**[B] This also reframes the architecture in the asymmetric way Jon
described**, and the evidence now supports each named role specifically
(not just as a plausible-sounding sketch):
- **Gemma = semantic primary** - highest solo accuracy (93.7%,
  arguably higher once the `casual_photo` taxonomy artifacts are
  discounted), stays the default answer.
- **SigLIP = strong fast corroborator/dissent trigger** - highest
  accuracy among the towers (92.5%), correct on 14/21 of Gemma's real
  disagreement cases, its disagreement is the FIRST condition in the
  rule that worked.
- **MobileNet = cheap divergent confirmation signal** - lowest solo
  accuracy (76.5%) but genuinely independent error pattern (established
  in the earlier complementarity analysis); here it serves as the
  SECOND, corroborating condition, not the primary dissent trigger -
  the rule needed BOTH towers to agree with each other against Gemma,
  not either one alone (an ungated "SigLIP disagrees" rule alone would
  flag far more of Gemma's 311 correct predictions - not tested
  numerically here, but implied by SigLIP's own ~11% overall
  disagreement rate with Gemma vs. this rule's 18/332 = 5.4% flag rate).
- **YOLO/CV/layout = structural evidence, still unqualified for this
  specific role** - genuinely different evidence type (geometric, not
  semantic), but not yet benchmarked at real scale against this
  project's own taxonomy (see the earlier sensor-value table) - an open
  slot in this architecture, not a filled one.

**[C] Not yet established**: whether this specific rule generalizes
beyond n=21 errors / n=332 images (a genuinely small base rate to
calibrate any production threshold against), whether it would still
work if Gemma's prompt/taxonomy were made consistent with the towers'
8-class label space (removing the `casual_photo` confound at the
source rather than post-hoc filtering it), and whether a THIRD
condition (e.g. requiring Gemma's own confidence below some threshold,
once the raw logit gap is actually extractable from the production
path) would improve precision further. None of this is implemented -
still qualification/prototyping-signal data, not a production rule.

Script: `diagnostics/analyze_gemma_errors_dissent.py`. Report:
`data/logs/reviewed/gemma_errors_dissent_analysis.txt`.

## Flat 8-bucket Gemma re-run: taxonomy-narrowing shifts the ERROR SET, not just removes taxonomy-mismatch noise (2026-08-07)

Jon asked to rerun Gemma restricted to exactly the 8 categories
`manifest_final.csv`'s ground truth uses (plus `uncertain_review`, the
standing escape valve) - removing `casual_photo`/`photo_collage`/
`cemetery_photo`, the source of the taxonomy-mismatch confound found in
the prior 11-category run. Built by filtering `core/taxonomy.py`'s real
`categories_for_classifier_prompt()` output to the 9 allowed IDs and
rendering through the SAME production prompt template
(`config/prompts/classifier_classify_v1.txt`) - real tuned guidance
text, not hand-written descriptions.
(`diagnostics/run_gemma_flat8_on_benchmark_holdout.py`). The original
11-category run's predictions were archived first, unchanged, at
`data/outputs/vit_family_benchmark/gemma_11cat_production_prompt/` so
both remain directly comparable going forward.

**[A] Result: 94.0% (312/332)** - essentially flat against the
11-category run's 93.7% (311/332), a 1-image net difference. If
narrowing the taxonomy had simply removed the `casual_photo` confound
and left everything else untouched, that's roughly what you'd expect
(7 fewer artifact-errors, accuracy ticks up slightly). **What actually
happened is more interesting and worth taking seriously**:

| Category | 11-cat run | flat-8 run |
|---|---|---|
| dense_tabular_rows | 100.0% (97/97) | 99.0% (96/97) |
| genealogy_chart | 100.0% (1/1) | 100.0% (1/1) |
| **handwritten_ledger** | **100.0% (4/4)** | **25.0% (1/4)** |
| map_land_record | 100.0% (14/14) | 100.0% (14/14) |
| mixed_text_image | 66.7% (2/3) | 66.7% (2/3) |
| portrait_photo | 50.0% (6/12) | 75.0% (9/12) |
| printed_document | 94.9% (166/175) | 93.7% (164/175) |
| **website_screenshot** | **80.8% (21/26)** | **96.2% (25/26)** |

**[A] Direct per-image diff between the two runs** (17 of 332
predictions flipped):

- **7 cases flat-8 fixed** (was wrong in the 11-cat run, now right):
  5 of these are exactly the `casual_photo` taxonomy-artifact cases
  from the earlier analysis (`portrait_photo` ground truth, Gemma no
  longer has `casual_photo` available to pick) - the expected,
  intended effect. The other 2 are genuine `website_screenshot` fixes
  unrelated to `casual_photo` at all.
- **6 cases flat-8 BROKE** (was right in the 11-cat run, now wrong) -
  **none of these involve `casual_photo` in either direction**, meaning
  narrowing the category list didn't just subtract noise, it changed
  Gemma's decision boundary somewhere else too. Most strikingly:
  **3 of 4 `handwritten_ledger` images flipped to `map_land_record`** -
  a category pairing with no obvious semantic relationship (a
  handwritten ledger page and a map share little visually). One
  `printed_document` also flipped to `map_land_record`, one
  `printed_document` flipped to `handwritten_ledger`, and one
  `dense_tabular_rows` flipped to `website_screenshot`.
- **4 cases both wrong, but wrong differently**: all 4 were the
  `casual_photo`-in-11cat cases that were ALREADY wrong ground-truth-
  wise (2 `printed_document`, 2 `portrait_photo`) - removing
  `casual_photo` didn't fix them, it just redirected the wrong answer
  elsewhere (`mixed_text_image` x3, `portrait_photo` x1).

**[B] Interpretation**: this is the same class of finding this project
already has strong precedent for at the PROMPT-GATE level (asking
"is this a manifest?" before "is this a census?" changed which pages
got claimed as manifests) - now demonstrated at the WHOLE-TAXONOMY
scale. The set of categories offered to the model isn't a neutral
filter on its answer space; changing it measurably shifts decision
boundaries for categories that were never removed or added. The
`handwritten_ledger`->`map_land_record` regression in particular reads
as a genuine, surprising behavioral quirk worth a closer look on its
own - **[C] not explained by this pass alone** (hypothesis: with fewer
categories in the list, relative prompt-position/context effects shift
which category "wins" close calls, independent of any category's own
guidance text changing - untested).

**[B] For the Jon-vs-towers comparison specifically**: the flat-8
number (94.0%) is the fairer one to set against the 8-category vision
towers (92.5% best/siglip) since it removes the taxonomy-mismatch
confound at the SOURCE rather than requiring post-hoc filtering, as the
error-dissent analysis had to do for the 11-category run. Gemma's edge
over the best tower narrows slightly under this fairer framing (94.0%
vs. 92.5%, a 1.5-point gap) compared to the 11-category framing's
93.7% vs. 92.5% - but the conclusion doesn't change: Gemma remains the
single best-performing sensor on this held-out set either way.

**[C] Not yet done**: rerunning the error-dissent analysis (Gemma
error vs. tower agreement, the SigLIP+MobileNet arbitration rule) on
this flat-8 prediction set specifically - the error SET changed enough
(6 new errors, 5 fewer taxonomy-artifact ones) that the earlier rule's
50%/1.9% catch-rate numbers should be re-checked against this cleaner
run before being treated as final, not assumed to carry over unchanged.

Script: `diagnostics/run_gemma_flat8_on_benchmark_holdout.py`. Report:
`data/logs/reviewed/gemma_flat8_on_benchmark_holdout_report.txt`.
Predictions: `data/outputs/vit_family_benchmark/gemma_flat8/
gemma_flat8_test_predictions.json`. Archived original 11-category run:
`data/outputs/vit_family_benchmark/gemma_11cat_production_prompt/`.

## Decision Engine implementation begins (2026-08-07)

Jon's direction: begin PRODUCTION implementation of the routing
arbitration layer this doc's research has been building toward, using
this doc as the design basis but explicitly NOT hard-coding current
findings as permanent truths - the vision-tower benchmark is still
running, Gemma E2B fine-tuning feasibility on 16GB hardware is unknown,
and per-category authority may look different once more evidence
exists. This section records what was actually built, tagged [A]/[B]/[C]
per the doc's standing discipline.

**[A] Built**: `core/routing_decision_engine.py` - a NEW module,
distinct from the pre-existing `core/decision_engine.py` (Stage 2's
preprocessing-profile pass-through + tower-consensus recording, which
explicitly does not act on its own output - see CODE_MAP.md's naming-
collision note, added specifically because these two modules' names are
easy to confuse). This module CONSUMES already-recorded `EvidenceRecord`
objects (producer-agnostic - a live Gemma call, a replayed JSON
prediction, and a future CV sensor all produce the same shape) and
ARBITRATES a `DecisionResult` (selected class, trust state, full
human-readable audit trail, disagreements list). Zero inference imports
in this file by design - sensor acquisition and decision logic are kept
structurally separate, so either can evolve independently.

**[A] Four policies implemented, switchable via one config field with
zero code changes** (`config/decision_engine.yaml`'s `default_policy`):
`gemma_primary`, `vision_primary`, `weighted_fusion`, and
`class_specific_authority`. `gemma_primary` is TODAY's configured
default because Gemma is currently the single best-measured sensor on
the one same-split comparison that exists (93.7-94.0% vs. SigLIP's
92.5% - see the Gemma comparator sections above) - **this is a
configuration value, not a structural assumption**. The config file's
own comments state this provisionality explicitly, including that
Gemma E2B fine-tuning on 16GB hardware is unproven and the vision
benchmark may still improve. `class_specific_authority`'s per-category
table (Jon's `dense_tabular_rows: primary=fine_tuned_vit` /
`printed_document: primary=gemma` / `map_land_record: primary=
trained_vision_consensus` shape) is implemented and tested but left
EMPTY in the default config - no category has validated per-class
evidence yet to justify a specific override, so none was invented.

**[A] Family-weighted evidence aggregation**: `weighted_fusion`/
`vision_primary` group evidence by `sensor_family` before voting, so
correlated architectures don't outweigh independent ones - directly
implements this doc's own complementarity finding (convnext/dinov2/
beit/swin/siglip/vit21k pairwise agreement 91-94%, largely redundant;
mobilenetv2 agreement only 76-78%, genuinely independent). Config's
`sensor_families` map is the mechanism, editable without code changes.

**[A] Gemma-specific findings respected directly in the implementation**,
not just referenced: self-reported confidence is never used as a trust
signal (the engine's `score_semantics` mechanism only recognizes
`logit_margin`/`softmax_probability`/`ordinal_confidence` score kinds -
Gemma's confidence field has no defined semantics in this system on
purpose); low margin reduces trust state rather than flipping the
selected class outright (every "weak primary" path in `_arbitrate_
primary_supporting()` still returns the primary's OWN predicted class,
just with a lower trust state - verified directly in test 4, "selected
class stays gemma's pick even under QUARANTINE"); every threshold is
marked UNCALIBRATED in both code and config comments, with the real
measured numbers (median gap 30.41 correct / 13.56 wrong, n=368) cited
as the DIRECTIONAL basis, not asserted as validated cutoffs.

**[A] Tests**: `tests/test_decision_engine.py` (no pytest in this
environment - plain assert, `python tests/test_decision_engine.py`,
21/21 passing) covers all 8 scenarios Jon specified. **Test 8 is the
load-bearing one**: proves `gemma_primary` and `vision_primary` produce
genuinely DIFFERENT routing decisions on IDENTICAL evidence, driven
entirely by a config field, with zero code changes between the two
`decide()` calls.

**[A] First milestone - replay, not live wiring**: `diagnostics/
replay_decision_engine.py` feeds the ALREADY-RECORDED Gemma flat-8 run
+ 7-tower logit extraction (same 332-image held-out split) through the
engine - "recorded evidence JSON -> Decision Engine -> route/quarantine/
audit record," per Jon's explicit first-milestone instruction. Real
result: under `gemma_primary`, 231/332 (69.6%) land `ACCEPT_WITH_
CAUTION` (100% accurate within that bucket on this sample), 101/332
(30.4%) `QUARANTINE` (80.2% accurate - correctly the lower-trust
bucket), overall 312/332 (94.0%) matching the underlying Gemma flat-8
number exactly (the engine doesn't change WHICH class gets selected
under gemma_primary with unscored primary evidence, only the trust
state layered on top).

**[A] A real, honest gap surfaced by the replay, not smoothed over**:
Gemma's production `classify()` call doesn't expose a raw decision-
token logit margin (only the self-reported confidence field, measured
unreliable three separate times in this doc already) - so this replay
records Gemma's evidence as UNSCORED rather than substituting a known-
bad number. Consequence: `gemma_primary` policy can NEVER reach
AUTO_ACCEPT in this replay (always falls to ACCEPT_WITH_CAUTION or
QUARANTINE, since unscored primary evidence is treated conservatively
by design). Meanwhile `vision_primary` (91.0% overall, but 245/332
reach AUTO_ACCEPT at 97.6% accuracy within that bucket) and
`weighted_fusion` (92.8% overall, 305/332 AUTO_ACCEPT at 96.7%) DO
reach real AUTO_ACCEPT states, because the tower evidence has genuine
normalized scores. **[B] This is a concrete illustration of a real
next-step priority**: a live Gemma adapter that captures the actual
raw logit margin (the mechanism already exists and is validated -
`diagnostics/test_gemma_logit_confidence.py` - it just isn't wired into
the production `classify()` path or this replay) would likely let
`gemma_primary` reach AUTO_ACCEPT on its genuinely confident cases,
not just ACCEPT_WITH_CAUTION on all of them. **[C] Not yet tested**:
whether that would meaningfully change overall accuracy/trust-state
distribution, or just relabel the same decisions with more granular
trust states.

**Explicitly NOT done in this pass** (per Jon's "do not immediately
wire every production sensor into live inference" instruction): no live
sensor adapters (a real Gemma call, a real tower forward pass, a real
CV measurement -> EvidenceRecord in actual Stage 5 flow), no wiring into
`core/pipeline_db.py` or the bucket-CSV write path, no QUARANTINE
trust-state pipeline effect (routing to a review queue etc.), no
row-regularity/column-grid/layout-detector evidence adapters (the
detectors exist and are real - `core/row_segmentation.py`,
`core/layout_detector.py`/`_v26.py` - but no adapter turns their output
into an EvidenceRecord yet).

Files: `core/routing_decision_engine.py`, `config/decision_engine.yaml`,
`tests/test_decision_engine.py`, `diagnostics/replay_decision_engine.py`.
Report: `data/logs/reviewed/decision_engine_replay_report.txt`. See also
CODE_MAP.md's own entry for this module (added alongside this doc
section) for the naming-collision note and integration-status summary.

## Live Gemma logit-margin adapter + fair policy comparison (2026-08-07)

Direct follow-up to the Decision Engine implementation's honest gap
(Gemma's production `classify()` path exposed no raw logit margin, only
the unreliable self-reported confidence field, so the first replay
could never let `gemma_primary` reach `AUTO_ACCEPT`). Jon: "Build the
live Gemma decision-token logit-margin adapter, then replay/compare the
same corpus again... you can finally evaluate gemma_primary fairly
against the vision-primary and weighted policies using real uncertainty
from both sides."

**[A] Built**: `core/gemma_logit_margin_adapter.py` - wraps an
already-initialized `GemmaLoader` WITHOUT modifying `core/loaders/
gemma_loader.py` (same established convention as `core/classifier.py`'s
`_enable_raw_output_debug()`), replicating `_run_generate()`'s exact
input-construction logic but adding `output_scores=True`. Locates the
decision token differently than the earlier binary yes/no gate
experiments (`diagnostics/test_gemma_logit_confidence.py`), since the
production `classify()` prompt's answer is a multi-token free-form
category name after a "category: " label, not a single token -
`find_category_decision_token_index()` finds "category:" in the
decoded text and maps that character offset back to the FIRST token of
the value. **[C]** using only the first value token as "the decision"
is a real simplification (defensible because every category id in this
taxonomy starts with a distinct first word, but not independently
verified against every tokenizer-collision case).

**[A] Result: 100% decision-token location rate (332/332)**, and a
strong margin/correctness separation - **correct-case median margin
19.12, wrong-case median margin 1.75** (n=312 correct, n=20 wrong) -
even cleaner than the earlier smaller-sample standalone experiments
found. Same 94.0% accuracy as the unscored flat-8 run (expected -
greedy decoding is deterministic, adding `output_scores=True` doesn't
change what gets generated, only what's observable afterward).

**[A] Fair 3-way policy comparison** (`diagnostics/replay_decision_
engine_v2_real_gemma_margin.py`), same 332-image split, same 7 towers,
real uncertainty evidence on all sides:

| Policy | Overall accuracy | Trust-state distribution | AUTO_ACCEPT accuracy |
|---|---|---|---|
| gemma_primary | **94.0%** (312/332) | auto_accept 111, accept_with_caution 158, quarantine 63 | **100.0%** (111/111) |
| vision_primary | 91.0% (302/332) | auto_accept 245, accept_with_caution 76, quarantine 11 | 97.6% (239/245) |
| weighted_fusion | 92.8% (308/332) | auto_accept 305, accept_with_caution 21, quarantine 6 | 96.7% (295/305) |

**[B] gemma_primary doesn't just win on overall accuracy - it produces
materially better-CALIBRATED trust triage**, which is arguably the more
important property for a routing engine: `AUTO_ACCEPT` +
`ACCEPT_WITH_CAUTION` together cover 269/332 images (81%) with **ZERO
errors in either bucket combined** - all 20 real mistakes concentrate
into the 63 `QUARANTINE` cases (68.3% accuracy there, correctly the
riskiest bucket, exactly the behavior a trust-triage system should
have). `vision_primary` and `weighted_fusion`, by contrast, both leak
real errors into their own `AUTO_ACCEPT` bucket (6 and 10 wrong
predictions respectively land in what should be the most-trusted tier)
- their higher AUTO_ACCEPT COUNTS (245, 305 vs. gemma_primary's 111)
come at the cost of some of those being wrong, not free extra coverage.

**[B] This is the fair comparison this whole exercise was building
toward**, and it reinforces rather than overturns `gemma_primary` as
the current configured default - not because vision evidence is
unhelpful (it still contributes real value as supporting/cross-check
evidence within gemma_primary's own arbitration), but because Gemma's
real, live-extracted uncertainty signal turns out to be a genuinely
strong triage mechanism on its own, not merely "the model with the
highest raw accuracy." **[C] Not yet tested**: whether this finding
holds on a genuinely unseen (non-benchmark) sample, and whether
`class_specific_authority` overrides could beat `gemma_primary`'s
trust-calibration property specifically for categories where a vision
tower is known to be stronger (see the next section).

Files: `core/gemma_logit_margin_adapter.py`, `diagnostics/run_gemma_
logit_adapter_on_benchmark_holdout.py`, `diagnostics/replay_decision_
engine_v2_real_gemma_margin.py`. Predictions: `data/outputs/
vit_family_benchmark/gemma_flat8_logit_margin/gemma_flat8_logit_margin_
test_predictions.json`. Reports: `data/logs/reviewed/gemma_flat8_logit_
margin_report.txt`, `decision_engine_replay_v2_real_gemma_margin_
report.txt`.

## Class-specific authority table populated with real evidence (2026-08-07)

Direct follow-up per Jon: "After that, the tower benchmark can start
populating the class-specific authority table with actual evidence
instead of hypotheses." Computed real per-category accuracy for Gemma
(live logit-margin run) + all 7 towers on the shared 332-image held-out
split:

| Category | n | gemma | convnext | mobilenetv2 | dinov2 | beit | swin | siglip | vit21k |
|---|---|---|---|---|---|---|---|---|---|
| dense_tabular_rows | 97 | **0.99** | 0.95 | 0.91 | 0.97 | 0.95 | 0.94 | 0.96 | 0.93 |
| map_land_record | 14 | **1.00** | 0.86 | 0.57 | **1.00** | 0.71 | 0.79 | 0.93 | **1.00** |
| portrait_photo | 12 | 0.75 | 0.58 | 0.58 | 0.67 | 0.58 | 0.58 | **0.92** | 0.75 |
| printed_document | 175 | **0.94** | 0.92 | 0.73 | 0.90 | **0.94** | 0.89 | 0.93 | 0.87 |
| website_screenshot | 26 | 0.96 | 0.92 | 0.73 | **1.00** | **1.00** | **1.00** | 0.92 | **1.00** |
| genealogy_chart | 1 | (all 100% except mobilenetv2/dinov2 - n=1, not usable) |
| handwritten_ledger | 4 | 0.25 | 0.50 | 0.75 | 0.50 | 0.00 | **1.00** | 0.50 | 0.75 |
| mixed_text_image | 3 | 0.67 | 0.67 | 0.67 | 0.00 | 0.67 | 0.67 | 0.33 | 0.67 |

**[A] Two categories have REAL evidence of a vision tower beating
Gemma, not hypothesis**: `portrait_photo` (SigLIP 92% vs. Gemma's 75%,
every other sensor 58-75%) and `website_screenshot` (DINOv2/BEiT/Swin/
ViT21k all 100% vs. Gemma's 96%, a 4-tower-unanimous margin, not a
single-sensor fluke). `map_land_record` is a near 3-way tie at the
ceiling (gemma/dinov2/vit21k all 100%) - gemma kept as primary since
it's tied, not beaten, unlike the two categories above. `dense_tabular_
rows`/`printed_document` stay gemma-primary, consistent with both the
measured numbers and Jon's own original hypothesis for `printed_
document`. The 3 tiny-N categories (genealogy_chart n=1, handwritten_
ledger n=4, mixed_text_image n=3) were EXPLICITLY OMITTED from the
table, not given a guessed entry - any "best sensor" claim at those
sample sizes would be noise per this project's standing tiny-N
discipline, especially `handwritten_ledger`'s apparent Swin-100%/
BEiT-0% split, which is exactly the kind of number that looks dramatic
and is almost certainly n=4 noise, not a real finding.

**[A] Real-data test of the populated config, and a genuine limitation
found, not glossed over**: running `class_specific_authority` on the
same held-out set scored **93.1% (309/332)** - slightly BELOW `gemma_
primary`'s 94.0%. `website_screenshot` hit a clean 26/26 (100%, up from
Gemma's 25/26), confirming the override helped where it fired. But
`portrait_photo` only reached 8/12, well short of SigLIP's own 11/12 -
**traced directly via the audit trail**: for 3 of the 4 wrong `portrait_
photo` cases, the `portrait_photo` rule never fired AT ALL. The
policy's single-pass bootstrap design (family-weighted plurality vote
across ALL evidence first, THEN look up that candidate's rule) means a
category's override only applies when the whole-evidence plurality
vote already lands on that category - for `portrait_photo` specifically,
where most towers are weak (58-67%), the bootstrap vote frequently
lands on `mixed_text_image` or `printed_document` INSTEAD, and since
those categories have their own rules (or none), the `portrait_photo`
rule - and SigLIP's real strength on it - never gets consulted for
those images. This is exactly the caveat flagged in `_policy_class_
specific_authority()`'s own docstring when it was built ("a genuine
open design question... flagged here rather than silently resolved"),
now empirically confirmed rather than theoretical.

**[B] What this means**: populating the table with real evidence was
worth doing (`website_screenshot` genuinely improved), but the
single-pass bootstrap architecture measurably limits how much a
category override can help when that category's own sensors are
collectively unreliable at first-pass identification - the exact
situation (weak general agreement) where a specialist override would
help MOST is also the situation where the current bootstrap mechanism
is least likely to reach it. **[C] Not implemented, a real candidate
next step**: an iterative/fixed-point version (re-bootstrap using only
each rule's own designated primary+supporting sensors' votes, not the
full evidence set, possibly across 2 passes) might resolve this - or a
lower-level fix (giving portrait_photo-relevant sensors, i.e. SigLIP,
more bootstrap weight specifically) - neither attempted here, this
pass's job was populating real data and reporting what it revealed,
not redesigning the arbitration mechanism.

Config: `config/decision_engine.yaml`'s `class_specific_authority`
section (now populated, with inline comments citing the exact measured
numbers behind each entry).

## Cross-session context: YOLO layout-detector bootstrap (as of 2026-08-05/06)

Status brief received verbatim from the parallel session doing this
work, recorded here so any session can pick it up without re-deriving
it. **This is a DIFFERENT tool from `core/layout_detector.py`'s
DocLayout-YOLO** (item 6 above, off-the-shelf, general-purpose) - this
one is a custom fine-tune, purpose-built for this project's forms.

> **What it is**: Experimental YOLOv26-small fine-tune for table/row/
> header detection on census pages, used only as an optional assist
> inside `training/yolo_assisted_auto_sidecar.py` (conditional CV-first,
> YOLO-rescue-if-quarantined). **Not wired into production** -
> `core/auto_sidecar.py` doesn't call it.
>
> **Training history (v1->v4)**: v1 (13 img) -> v2 (103 img) -> v3 (203
> img, 1911-only) -> v4 (402 img, current best): 1911x302 + 1921x100,
> stratified train/val split. Checkpoint: `data/outputs/
> layout_bootstrap_train/runs/census_bootstrap_v4/weights/best.pt`.
>
> **Table detection**: P=0.97/R=1.00/mAP50=0.995 - solid. Row detection
> plateaued (~84.5% recall v3->v4). **Verdict**: diversifying by year
> beat pure scale-up - v4 fixed two real generalization failures (RG14
> narrow-box, 1939 Register fragmentation) that v3 couldn't, confirmed
> via a 21-image cross-year/cross-form probe.
>
> Full writeup: `docs/LAYOUT_DETECTOR_BOOTSTRAP_TRAINING.md`.
>
> **New data pulled but NOT yet used for training** (this session): 1901
> (28 img), 1906 (23 img), 1926 (50 img), 1931 (50 img) - all sourced via
> `central.bac-lac.gc.ca` redirect -> `data2.archives.ca` (permissive
> robots.txt, verified directly). 1911/1921 already had 402 combined.
> 1916: no confirmed working URL found, deferred.
>
> **Baseline probe of v4 against these 4 new years**: 0/151 total
> misses, but **1926 flagged heavily** (likely real - Prairie Provinces
> census is a genuinely different form family from the standard census,
> not just a different year).
>
> **Directly relevant to what the new session should pick up**:
> - Deskew bug fixed in `core/manifest_pipeline.py` (34/1750 corpus
>   images had a bad +-15 degree rotation applied; now correctly falls
>   back to 0 degrees) - affects any future Stage 0-3 corpus rerun,
>   independent of YOLO.
> - Built a new row-number-column detection system in
>   `core/row_segmentation.py` (finds printed row numbers as a stronger
>   anchor than faint ruling lines; left/right dual-column with
>   agreement/disagreement telemetry; row-level trust +
>   neighbor-inheritance for anomaly recovery) - validated end-to-end on
>   a real 1901 page, but not yet wired into `auto_sidecar.py`'s
>   production `fixed_periodic` dispatch.
> - `canada_census_1901.yaml` template recalibrated with real measured
>   geometry + row-number-column fractions. 1906/1926/1931 have no
>   templates yet.

### Convergent finding worth acting on: 1926 is real

**Two completely independent investigations, same day, flagged 1926 as
structurally anomalous, without either referencing the other:**

- The YOLO session's baseline probe (v4, trained only on 1911/1921)
  against the new pulls: 0/151 misses overall, but 1926 flagged heavily.
- The single-shot Gemma classifier test against the same LAC pulls
  (`data/outputs/single_shot_classifier_test/log.txt`, same day): 1926
  had by far the worst year-extraction accuracy (2/50, 4%) while
  category accuracy stayed high (96%) - consistent with a page that's
  recognizably "census-like" but whose year/header layout doesn't match
  what either system learned from the more common years.

Two unrelated methods (a fine-tuned object detector and a general-
purpose VLM prompt) converging on the same anomaly from entirely
different signals is strong evidence this is real, not noise in either
system. This also corroborates a standing hypothesis already on record
before either of today's investigations ran - see
`[[project_prairie_census_different_form]]` in Claude's memory (1906/
1916/1926 flagged as structurally different form family from 1901/1911/
1921/1931). **Recommendation for whichever session picks this up**:
treat 1926 (and likely 1906/1916 too, per the standing memory note) as
needing its own template/prompt handling, not as "census, generic" that
will improve with more undifferentiated training data.
