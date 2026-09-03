# Reference Pipeline v3 — full pipeline-state checkpoint before the taxonomy/prompt-template pass

**Status**: preserved reference data, not a failed experiment. Do not
overwrite or delete `data/outputs/reference_pipeline_v3/`.

## What this is

A frozen snapshot of the entire live Stage 0-5 pipeline state for the
1750-image corpus, captured 2026-08-04, immediately before Jon starts a
fresh Stage 0-5 run to measure whether this session's changes improved
or regressed the pipeline:

- `manifest.csv`, `manifest_provenance.json` - Stage 0's acquisition
  manifest and provenance record.
- `pipeline.db` - the full orchestration database, including every
  correction/label recorded during this session's ground-truthing pass
  (`stage5_human_review` events, `bad_deskew`/`needs_new_bucket` status
  values) and the Stage 4 (Validation Capture) stage_outputs history.
- `baseline_embeddings.json` (genuinely pre-Stage-3, GPU-sharded k=4
  recapture) and `postprocessing_embeddings.json` (Stage 4, post-Stage-3)
  - the full before/after semantic snapshot pair this session's
    investigation was built on.
- `buckets/*.csv` - Stage 5's category CSVs, including this session's
  new `photo_collage`/`casual_photo`/`cemetery_photo` categories where
  populated.
- `misclassifications.csv` - the 77-image ground-truth set (67 real
  category labels + `bad_deskew`/`needs_new_bucket` sentinels),
  `flagged_needs_new_bucket.csv`, `flagged_bad_deskew.csv`.
- `working/` - every working-copy image plus its `*_analysis.json`
  (physical sensor, both the original and the raw-recaptured version)
  and `*_tower_consensus.json` sidecars.
- `raw_from_pdf/` (16 files) and `raw_stage0_recapture/` (1750 raw
  images + 1750 `*_analysis.json` sidecars from the physical
  raw-recapture pass, 3500 files total) - the reconstructed pre-Stage-3
  pixel state this session's whole before/after investigation depended on.
- `outputs/` - this pass's analysis artifacts: `image_analysis` (Stage
  1 physical, post-fix), `image_analysis_raw_recapture` (the raw
  recapture's own working files), `baseline_embeddings_raw_recapture`,
  `gpu_cpu_equivalence`, `gpu_normalize_full_corpus`,
  `stage4_before_after_comparison`, `stage1_gemma_stage4_flip_report`,
  `gained_lost_agreement_characterization`, `tower_consensus_audit`.

## Why this exists

This session made three categories of change that could each plausibly
affect classification behavior, and Jon wants to measure the combined
effect of all of them together with a genuine start-to-finish rerun,
not reason about each in isolation:

1. **Execution engine**: Stage 1's semantic capture now runs GPU-sharded
   k=4 (`core/baseline_embeddings.py`) instead of sequential CPU -
   already validated as numerically equivalent, included here for
   completeness, not expected to change classification outcomes.
2. **Taxonomy expansion**: `photo_collage`, `casual_photo`, and
   `cemetery_photo` added to `core/schema.py`'s `DocumentCategory` and
   `config/taxonomy.yaml`, discovered via manually ground-truthing the
   77-image flagged set - these are new possible Gemma outputs a fresh
   run could actually produce, unlike the engine change above.
3. **Prompt template refactor**: `config/prompts/classifier_classify_v1.txt`
   now renders its category-choice list from `config/taxonomy.yaml`'s
   `classifier_guidance` fields instead of hardcoded prose
   (`core/classifier.py`'s `render_classifier_prompt()`). Verified
   byte-for-byte (modulo one incidental line-wrap artifact in the
   original) against the pre-refactor prompt for the 9 categories it
   already covered - the 3 new categories above have no
   `classifier_guidance` yet, so they do NOT appear in the live prompt
   despite existing in the taxonomy.

A genuinely fresh run needs `data/manifest.csv`, `data/pipeline.db`,
`data/baseline_embeddings.json`, `data/buckets/*.csv`, and `data/working/`
all empty/absent, for the same reason `reference_pipeline_v2` gave: a
truly fresh corpus is easiest to reason about when comparing before/after.

## What was moved vs. left alone

**Moved** (verified file counts match before and after):
`manifest.csv`, `manifest_provenance.json`, `pipeline.db`,
`baseline_embeddings.json`, `postprocessing_embeddings.json`,
`misclassifications.csv`, `flagged_needs_new_bucket.csv`,
`flagged_bad_deskew.csv`, `buckets/` (9 CSVs, `working/` (5250 files),
`raw_stage0_recapture/` (3500 files), `raw_stage0_recapture_mapping.json`,
`raw_from_pdf/` (16 files), `pending_prune_review/` (609 files - an
active human-review workflow, not pipeline output, but moved per Jon's
explicit "move both to be sure"), `debug_model_inputs/` (curated
ground-truth text files, same reasoning), plus the 9 `data/outputs/`
subdirectories listed above.

**Deliberately left alone**: `data/reference_prerefactor/` (an
already-existing, older checkpoint from an earlier pass - a different
thing from this one), `data/archive_census_only_run/` (a separate,
already-named archive), `data/_test_fixtures/` (needed by the automated
test suite regardless of pipeline state), `data/automatedgenealogy_pull.csv`
(a static external source list, not pipeline output),
`data/outputs/reference_pipeline_v1/`, `v2/`, `prerefactor/` (already
frozen), and every other `data/outputs/` subdirectory (benchmark2*,
`*_lora_checkpoints`, `prompt_sweep*`, `vision_encoder_*`,
`experiment3_*`, etc. - unrelated research/training phases, not
regenerated by a Stage 0-5 run).

## Verification performed before/after moving

File counts confirmed matching post-move: `working/` 5250/5250,
`pending_prune_review/` 609/609, `buckets/` 9/9 CSVs, `raw_from_pdf/`
16/16, `raw_stage0_recapture/` 3500/3500 (1750 images + 1750 sidecars -
not a discrepancy, both halves of the physical raw-recapture pass live
there). `baseline_embeddings.json` byte size confirmed unchanged
(111,079,108 bytes) post-move.

## Do not

- Overwrite or delete anything under `data/outputs/reference_pipeline_v3/`.
- Confuse this with `data/reference_prerefactor/` or
  `data/outputs/reference_pipeline_v1/`/`v2/` - different, earlier
  checkpoints, still present and untouched.
- Assume the live files this was moved from still exist - they don't; a
  fresh Stage 0 run is expected to recreate `data/manifest.csv`,
  `data/pipeline.db`, `data/baseline_embeddings.json`,
  `data/buckets/*.csv`, and populate `data/working/` from scratch.
- Assume the 3 new taxonomy categories (`photo_collage`, `casual_photo`,
  `cemetery_photo`) will appear in the fresh run's classification output -
  they won't, since no `classifier_guidance` exists for them yet
  (see `docs/TAXONOMY.md`).
