# Stage 1 vs Stage 4 investigation - final conclusion (2026-08-04)

**Status**: CLOSED for the 77-image flagged set. This is the closing
document for a chain of work spanning Stage 4's build, the before/after
comparison, a real physical-data bug found and fixed mid-investigation,
a sandboxed Gemma re-test, and - finally - real human ground truth. Each
step is summarized here with a pointer to its own detailed doc; this
document's job is to state what the WHOLE chain resolved to, now that
it no longer needs to caveat "no ground truth exists."

## The chain, in order

1. **Stage 4 (Validation Capture) built** - repeated Stage 1's semantic
   capture (`core/baseline_embeddings.py`, GPU-sharded k=4) and physical
   capture (`core/image_analysis.py`) against the corpus's current,
   post-Stage-3 pixels, writing to separate files so Stage 1's baseline
   stayed untouched. First time this project had a genuine before/after
   pixel-state pair for the full corpus.

2. **Before/after comparison run** (`docs/STAGE1_STAGE4_BEFORE_AFTER_
   COMPARISON.md`). Found a real bug mid-comparison: the physical "Stage
   1" data had actually been captured against already-Stage-3-processed
   pixels (same bug class as `docs/REFERENCE_PIPELINE_V1.md`, previously
   fixed on the semantic side, not yet on physical). Fixed it
   (`scripts/recapture_image_analysis_raw.py`) using the same raw
   reconstruction already built for the semantic fix. With correct data:
   semantic drift was large (90% of image/encoder pairs below the
   0.9999 threshold used elsewhere this session) and had real decision
   consequences - **581/1750 images (33%) changed tower-consensus
   bucket**, **1217/1750 (70%) had at least one encoder's vote flip**.

3. **Gained/lost agreement characterization** (`benchmark/gained_lost_
   agreement_characterization.py`). Cross-referenced the 1217 flips
   against Gemma's real production classification. 57 images "gained"
   agreement with Gemma (tower's post-processing prediction now matches
   Gemma, didn't before); 20 "lost" it (matched before, no longer does).
   Gained-agreement images showed larger-than-typical contrast
   correction (mean +39.3 vs. +34.7 control); lost-agreement images
   showed even larger contrast AND deskew corrections (some at the
   ±30° estimator clamp), suggesting aggressive correction sometimes
   overshot.

4. **Gemma raw-image sandbox test** (`docs/GEMMA_RAW_SANDBOX_
   RECLASSIFICATION.md`). Fully isolated (zero production writes,
   checksum-verified) reclassification of the 77 flagged images' RAW
   pixels using the production Gemma model. Found Gemma itself is
   stable across preprocessing (76/77) - unlike the towers, which
   flipped on all 77 by construction. For the gained group, 56/57
   matched Gemma-on-raw to the tower's post-processing read; for the
   lost group, 20/20 matched Gemma-on-raw to the tower's PRE-processing
   read instead - a clean split suggesting (not yet proving) the
   post-processing read was likely right for gained and wrong for lost.

5. **Human ground truth** (`data/misclassifications.csv`, labeled via
   the newly taxonomy-driven `debug_tools/review_uncertain.py`). Jon
   manually labeled 64/77 images (13 left unlabeled as subtype
   candidates for a later taxonomy pass; 5 marked not useful; 3 flagged
   as needing a new bucket entirely - none of which represent a
   right/wrong tower call, correctly excluded below).

## The result

| Group | n with ground truth | Tower PRE correct | Tower POST correct | Gemma correct |
|---|---|---|---|---|
| Gained agreement | 44/57 | **0/44 (0%)** | **44/44 (100%)** | 44/44 (100%) |
| Lost agreement | 12/20 | **12/12 (100%)** | **0/12 (0%)** | 12/12 (100%) |

Both groups resolve with **zero exceptions**. Every gained-agreement
image with ground truth was genuinely wrong before preprocessing and
genuinely right after. Every lost-agreement image was genuinely right
before and genuinely wrong after. Gemma matched ground truth on all 56
labeled images in both groups - not most, all.

## What this settles

**The step-4 sandbox finding (item 4 above) was not just suggestive -
it was correct**, for both groups, without exception. The n=11 pilot's
near-zero-drift result (`docs/BENCHMARK2_METADATA_LAYER_QUALIFICATION.md`)
undersold how much a real, full-scale preprocessing pass can move a
tower's classification for a meaningful minority of images: this project
now has direct, ground-truth-confirmed evidence that preprocessing can
fully flip a classification from wrong to right (57 real cases) and,
just as decisively, from right to wrong (20 real cases) - both at
non-trivial rates within the flipped subset.

**Gemma's real-world reliability, for this specific 77-image
adversarial sample, was total** (56/56 where labeled) - a stronger
result than the sandbox test alone could show, since that only measured
agreement between Gemma-on-raw and Gemma-on-postprocessed/tower reads,
never against real truth. This is the first time this project has
measured Gemma's accuracy against genuine human labels rather than only
against another model's output or itself.

## What this does NOT settle

- **Corpus-wide accuracy** remains unmeasured and unclaimed - this is a
  77-image sample deliberately selected because the tower's prediction
  changed, not a random sample (`project_no_comprehensive_ground_truth`
  memory still applies to the other ~1670 images with no ground truth).
- **Why** large contrast/deskew corrections sometimes help (gained) and
  sometimes hurt (lost) at a mechanistic level - the correlation was
  observed (item 3), not causally isolated. A follow-up could test
  whether a deskew-magnitude threshold predicts which outcome occurs.
- **Whether Stage 2's currently-gated profile-decision should change**
  as a result - that's a separate, not-yet-made decision; this document
  supplies evidence for it, not the decision itself.
- The 13 still-unlabeled (subtype-candidate) and 8 sentinel-labeled
  (ignored/needs-new-bucket) images have no correctness verdict and
  aren't claimed to.
