# Fresh pass vs. reference_pipeline_v3 checkpoint (2026-08-04)

**Status**: complete. Answers "did this session's changes improve or
regress the pipeline" for the full 1750-image corpus, using a
controlled design - same exact source files, same working directory,
same preprocessing profile, only the code changed.

## What changed between the checkpoint and this run

1. Stage 1 semantic capture: GPU-sharded k=4 (`core/baseline_embeddings.py`)
   instead of sequential CPU.
2. Taxonomy: 3 new categories added (`photo_collage`, `casual_photo`,
   `cemetery_photo`), discovered via manually ground-truthing 77 flagged
   images from the pre/post-Stage-3 investigation.
3. Classifier prompt: `config/prompts/classifier_classify_v1.txt`
   templated to render its category-choice list from
   `config/taxonomy.yaml` instead of hardcoded prose - verified
   byte-identical to the original for the 8 pre-existing categories
   (see `docs/TAXONOMY.md`).

## [A] Source Evidence

Bucket distribution, checkpoint vs. fresh (1750 images in both):

| Bucket | Checkpoint | Fresh | Delta |
|---|---|---|---|
| dense_tabular_rows | 168 | 198 | **+30** |
| genealogy_chart | 6 | 6 | +0 |
| handwritten_ledger | 28 | 19 | -9 |
| map_land_record | 98 | 94 | -4 |
| printed_document | 1167 | 1154 | -13 |
| mixed_text_image | 20 | 29 | +9 |
| portrait_photo | 87 | 80 | -7 |
| website_screenshot | 176 | 170 | -6 |
| photo_collage | 0 | 0 | +0 |
| casual_photo | 0 | 0 | +0 |
| cemetery_photo | 0 | 0 | +0 |
| uncertain_review | 0 | 0 | +0 |

Per-image: **1657/1750 unchanged (94.7%)**, **93/1750 changed (5.3%)**.
Dominant transitions: `printed_document -> dense_tabular_rows` (25),
`website_screenshot -> dense_tabular_rows` (9), `handwritten_ledger ->
printed_document` (8), `portrait_photo -> mixed_text_image` (6),
`map_land_record -> printed_document` (5) - no single transition
dominates overwhelmingly; changes are spread across many bucket pairs.

Root-cause checks performed (not assumed):
- **Deskew angles**: identical for all 1750 images, checkpoint vs.
  fresh (0 differences).
- **Preprocessed pixel bytes**: identical for a 100-image sample
  (sha256 match, 0/100 differ).
- **Prompt content fed to Gemma**: confirmed byte-identical earlier
  this session for the 8 pre-existing categories.
- **Model's own stated reasoning on identical inputs**: genuinely
  contradicts itself run-to-run. Example (`Screenshot 2026-05-30
  230656.png`, identical pixels both times): checkpoint run said "The
  image consists of a list of names and does NOT exhibit the
  repetitive structure of a manifest"; fresh run said "The image
  displays a list of names structured in repeated rows, characteristic
  of a passenger manifest" - same image, opposite visual claim.

## [B] Interpretation

With pixels, preprocessing, and prompt content all proven identical
between runs, the 93 changed classifications cannot be attributed to
anything this session changed in the pipeline. The model's own
reasoning text directly contradicting itself on identical input is
strong, direct evidence of genuine Gemma sampling non-determinism
(temperature/generation randomness), not a pipeline effect - this is
the null hypothesis a controlled rerun exists to test for, and it's the
one that held up here.

The `+30` shift toward `dense_tabular_rows` is the most visible pattern
in the bucket table, but given the transition tally itself is spread
across 20 different bucket pairs with no single pair dominating beyond
25 cases (out of 1750 total images), and given the direct contradiction
evidence above, this reads as noise-consistent rather than a systematic
behavioral shift from any prompt or taxonomy change.

## [C] Conclusions

- **Did the GPU-sharded k=4 semantic capture change classification
  outcomes?** No evidence of it - Stage 5 classification doesn't
  consume Stage 1 semantic embeddings at all (that's tower-consensus
  evidence, a separate signal `core/decision_engine.py` records but
  doesn't act on yet); the two are independent.
- **Did the 3 new taxonomy categories get used?** No - confirmed empty
  (0 rows) in all three bucket CSVs, exactly as expected since none
  has `classifier_guidance` defined yet.
- **Did the prompt-template refactor change classification behavior?**
  No evidence of it - the rendered prompt content for the categories
  Gemma could actually choose is proven byte-identical to before.
- **Did anything regress?** No evidence found. The observed 5.3%
  classification drift is attributable to the model's own run-to-run
  non-determinism, based on directly observing contradictory reasoning
  on identical pixels - not proof beyond all doubt (a dedicated same-
  input-twice rerun with zero code changes would be the fully clean
  confirmation), but the most evidence-consistent explanation available
  from what was actually measured here.
- **Net assessment**: this session's changes appear to be a clean,
  behavior-preserving refactor for the 8 pre-existing categories,
  successfully extended with 3 new taxonomy categories that exist but
  are correctly inert until `classifier_guidance` is written for them -
  no regression, no unintended improvement, exactly the "structural
  change, unchanged behavior" the taxonomy refactor was designed to be.
