# Reference Pipeline v1 — benchmark snapshot of the existing production pipeline

**Status**: preserved reference data, not a failed experiment. Do not
overwrite or delete `data/outputs/reference_pipeline_v1/`.

## What this is

A frozen snapshot of the full 1677-image working corpus's Stage 1
measurements (both physical and semantic sensors), captured 2026-08-02,
under `data/outputs/reference_pipeline_v1/`:

- `physical_sensor_measurements.csv` - Stage A / `core/image_analysis.py`'s
  full-corpus run (geometry, skew, blur, illumination, contrast, edge
  statistics - see `docs/PREPROCESSING_STAGE_NOTES.md` for the real
  findings from this run).
- `semantic_sensor_embeddings_raw.json` - `core/baseline_embeddings.py`'s
  full-corpus run (all 8 qualified vision towers' pooled embeddings per
  image), wrapped with an `archive_metadata` block explaining the
  correction below.
- `sidecars/` - all 1677 per-image `*_analysis.json` sidecars from the
  physical-sensor run (these also still live beside their images in
  `data/working/`; copied here too so a future re-processing pass that
  overwrites those in place can't destroy this reference copy).

## Why this exists — the architectural discovery that prompted it

While reviewing Stage A's output, a direct pixel comparison between a
working-copy image and its true source-archive original
(`oocihm.lac_reel_c10264.658.jpg`) showed they are NOT the same:
original histogram range 0-224, working copy 0-255 (a full-range
stretch - the exact signature of `autocontrast`), mean absolute pixel
difference 19.05 (not 0).

**This means the current `data/working` corpus had already been through
`core/manifest_pipeline.py`'s preprocessing step (Stage 0, at the time)
in an earlier session, before either of these two captures ran.** Both
the Stage A physical measurements and the semantic-sensor baseline
embeddings archived here were measured on **already-preprocessed**
images, not true raw/pre-preprocessing ones - despite the semantic
embeddings being tagged `preprocessing_stage: "pre_preprocessing"` (that
tag is correct as a description of where in the CODE the capture call
happened - before that specific function call's own preprocessing step -
but not correct as a description of the images' actual state, since
they'd already been processed once before, by an earlier run).

This discovery is exactly what prompted standardizing the pipeline's
stage terminology (`docs/PIPELINE_STAGE_TERMINOLOGY.md`) and explicitly
separating **Stage 1 (Raw Sensor Capture, no image modification)** from
**Stage 3 (Image Processing, execute preprocessing)** as genuinely
distinct, individually-orchestratable stages - the conflation of "copy
working files" + "capture baseline" + "preprocess in place" inside one
function (`build_working_manifest_from_paths()`) is what let this
mislabeling happen unnoticed.

## What this is good for, and what it isn't

**Good for**: a real, honest benchmark of "what does the vision system
perceive, and what does Stage A physically measure, on the corpus as
the EXISTING pipeline actually leaves it" - a legitimate reference
point, since this IS the real production state today, not a
hypothetical one.

**Not a substitute for**: a true pre-preprocessing baseline. Once the
Stage 0/1/3 separation is real (Raw Sensor Capture genuinely runs before
any pixel modification, on every new image, not just retroactively on
an already-processed corpus), a **second reference dataset**
(`reference_pipeline_v2` or similar, following the same archival
pattern) should be captured and compared directly against this one on:

- preprocessing effectiveness
- embedding drift (this reference's embeddings vs. the new pipeline's
  genuine pre/post pair)
- physical measurement changes
- routing decisions
- downstream OCR performance

## Do not

- Overwrite or delete anything under `data/outputs/reference_pipeline_v1/`.
- Treat `semantic_sensor_embeddings_raw.json`'s `preprocessing_stage`
  field as literally true for these specific 1677 images - read the
  `archive_metadata` block first.
- Assume the LIVE files this was copied from
  (`data/outputs/image_analysis/full_corpus_analysis_report.csv`,
  `data/baseline_embeddings.json`) will stay unchanged - those are
  working files a future session may legitimately re-run/overwrite once
  the pipeline changes; this archive is the permanent snapshot.
