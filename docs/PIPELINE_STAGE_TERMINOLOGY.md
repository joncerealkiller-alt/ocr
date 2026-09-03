# Pipeline stage terminology — canonical naming (2026-08-02)

**Purpose**: the single source of truth for pipeline stage names, per
Jon's direction to standardize terminology before the architecture
grows further. Rename ORCHESTRATION scripts/docs/config/workflow
references to this scheme. Reusable capability libraries
(`core/vision_embeddings.py`, `core/pdf_conversion.py`, `core/
source_expansion.py`, etc.) stay capability-named, not stage-named -
they're used BY multiple stages, not owned by one.

## The seven stages

| # | Name | What it does | Real modules today |
|---|---|---|---|
| 0 | **Source Acquisition** | Discovery, expansion, manifest writing | `core/manifest_pipeline.py` (discovery/manifest parts), `core/source_expansion.py`, `core/pdf_conversion.py` |
| 1 | **Raw Sensor Capture** | Semantic + physical measurement ONLY - no image modification | `core/baseline_embeddings.py` (semantic), `core/image_analysis.py` (physical) |
| 2 | **Decision Engine** | Determine required preprocessing operations from Stage 1 evidence | `core/decision_engine.py` (built 2026-08-02, architecture only - see below; previously called "Stage B") |
| 3 | **Image Processing** | Execute preprocessing ONLY | `core/manifest_pipeline.py`'s `preprocess_for_manifest()` (deskew + `core/image_preprocessing.py` profile) |
| 4 | **Validation Capture** | Repeat Stage 1's measurements after processing | NOT STARTED (previously called "postprocessing capture" in the Benchmark 2 metadata design doc) |
| 5 | **Document Routing** | Classification and routing | `core/classifier.py`, the Multi-Tower Routing Audit (`docs/BENCHMARK2_3_MULTI_TOWER_ROUTING_AUDIT.md`), the E2B/E4B escalation policy |
| 6 | **Specialized Extraction** | OCR, tables, handwriting, maps, etc. | `core/row_extraction.py`, `core/semantic_stages.py`, model loaders |

**Orchestration layer (2026-08-02)**: `core/pipeline_db.py`'s
`PipelineDatabase` (SQLite, `data/pipeline.db`) - the pipeline's system
of record for STATE (which stage an image is at, its routing outcome,
pointers to where each stage's real evidence lives) as opposed to the
evidence itself, which stays exactly where it already lives (JSON
sidecars, bucket CSVs, `baseline_embeddings.json`). See
`docs/PIPELINE_DATABASE.md` for the full schema/design/migration-risk
writeup. Initially built read-only via a one-time migration script
against the real 1677-image corpus; **as of the same day, Stage 0/1/2/3
now dual-write into it live** (`core/manifest_pipeline.py`'s
`stage0_acquire_and_copy_sources()`/`stage1_capture_baseline_
embeddings()`/`stage3_preprocess_manifest()`, plus the new
`core/decision_engine.py`'s `stage2_decide_profiles()`) - every existing
CSV/JSON output is UNCHANGED, this is additive. Stage 5 (`core/
classifier.py`) and later stages are NOT yet wired in - that's still
future, incremental work.

**Stage 2 (Decision Engine), what's actually built vs. not**:
`core/decision_engine.py` does two DIFFERENT things, deliberately kept
apart. (1) Preprocessing-profile decision: still a trivial pass-through
(every image gets `"autocontrast"`, exactly as before this stage
existed) - per `docs/PREPROCESSING_STAGE_NOTES.md`'s explicit gate,
there is no validated evidence yet that any Stage 1 signal should
change which profile an image gets, so this stage is the real,
DB-wired ARCHITECTURE for that future decision, not an invented rule.
(2) Tower-consensus classification: REAL, not gated the same way -
reuses the Multi-Tower Routing Audit's own validated nearest-cluster +
consensus logic (promoted to `core/vision_embeddings.py`'s
`predict_nearest_bucket()`/`classify_consensus()`), computed from
already-captured Stage 1 embeddings with ZERO new model inference,
against a reference set built from already-classified buckets (15
images/bucket, matching the audit's own validated scale). Recorded as
`images.tower_consensus_category` (coarse) + a `<name>_tower_
consensus.json` sidecar (full per-encoder votes) - evidence for
whatever later decides E2B/E4B escalation, not acted on by this module.

## Why Stage 1 and Stage 3 had to become genuinely separate

**Real bug this caught (2026-08-02)**: `core/manifest_pipeline.py`'s
`build_working_manifest_from_paths()` used to conflate what should be
three different stages - copy to working dir (Stage 0), capture
baseline embeddings (Stage 1), and preprocess in place (Stage 3) - all
inside one function, in that call order. Because the function ALREADY
existed and had ALREADY been run against the current 1677-image corpus
in an earlier session (applying deskew+autocontrast), a LATER capture
of "baseline embeddings" run directly against `data/working` (not
through that same function, in a fresh session) silently measured
already-preprocessed images while still labeling them
`preprocessing_stage: "pre_preprocessing"` - confirmed via direct pixel
comparison against the true source archive (histogram 0-224 in the
original vs. 0-255 in the working copy, mean abs diff 19.05). See
`docs/REFERENCE_PIPELINE_V1.md` for the full story and the preserved
snapshot this produced.

**The fix going forward**: Stage 1 (Raw Sensor Capture) and Stage 3
(Image Processing) must be genuinely separate, independently-callable
orchestration steps - not two things that happen to run in the right
order inside one shared function. That's what makes "capture before
this specific run's own preprocessing" different from "capture
genuinely before ANY preprocessing has ever touched this image."

## Real bug this caught, part 2: the split (2026-08-02)

`build_working_manifest_from_paths()` is now split into three genuinely
separate, independently-callable functions - `stage0_acquire_and_copy_
sources()`, `stage1_capture_baseline_embeddings()`, `stage3_preprocess_
manifest()` - each reading/writing only through a manifest_path's
"file_path" column, not an in-memory dict handed between them.
`build_working_manifest_from_paths()` itself is now a thin composition
wrapper (stage0 -> stage1 if capture_baseline -> stage3), kept so every
existing caller (`build_working_manifest()`, `scripts/
run_preprocessing.py`, `ui/build_manifest_ui.py`'s subprocess CLI) sees
identical behavior and identical printed progress lines (verified: the
GUI's `_parse_preprocessing_progress()` regex still matches). See
`core/manifest_pipeline.py`'s own module docstring "SPLIT (2026-08-02,
second restructuring pass)" section for the full design and the
deliberate ordering change (manifest.csv now written as part of Stage 0,
immediately after copying, rather than after preprocessing - contents
are identical either way, but this is what makes Stage 1/Stage 3
independently callable against an existing manifest without re-running
Stage 0). Verified with a real smoke test: stage0 called alone, then
stage3 called alone afterward against the manifest stage0 wrote - pixels
only changed on the stage3 call, manifest paths stayed stable throughout.

## Old -> new name mapping (for anything still using old terms)

- "Stage 0" (`core/manifest_pipeline.py`'s old all-in-one meaning) ->
  split across new Stage 0 (Source Acquisition) + Stage 1 (Raw Sensor
  Capture) + Stage 3 (Image Processing) - and, as of the second
  restructuring pass above, these are now three real separate functions,
  not just three doc-level names for one function's internal phases.
- "Stage A" (`core/image_analysis.py`) -> Stage 1, physical half.
- "Stage B" (not-yet-built profile selection) -> Stage 2 (Decision
  Engine).
- "Stage 2" (`core/manifest_pipeline.py`'s old finalize-after-
  classification meaning, `finalize_manifest()`) -> NOT canonical Stage
  2 (that's Decision Engine, `core/decision_engine.py` - a real
  collision, since both were historically called "Stage 2" for
  different reasons). Labeled "Stage 2b" in code comments/docstrings to
  disambiguate - it's actually post-Stage-5 bookkeeping (merging
  classification buckets + manual dewarp output). Whether it deserves
  its own canonical number is still not resolved, but as of 2026-08-02
  it's DB-query-based (see `docs/PIPELINE_DATABASE.md`'s "finalize_
  manifest() migration" section) rather than re-scanning bucket CSVs by
  hand - the naming ambiguity is a documentation gap, not a functional
  one anymore.

## Rollout status (2026-08-02) - what's done, what's pending

**Done this pass**:
- This terminology doc created as the canonical reference.
- `docs/REFERENCE_PIPELINE_V1.md` written, archive captured.
- `docs/PREPROCESSING_STAGE_NOTES.md` and `docs/CODE_MAP.md` updated to
  reference the new stage numbers (see those files' own 2026-08-02
  entries).

**Done (2026-08-02, second pass)**: `core/manifest_pipeline.py`,
`core/image_analysis.py`, `core/baseline_embeddings.py` docstrings
updated to reference new stage numbers (old "Stage A"/"Stage 0"/"Stage
B" names kept alongside as a bridge, not deleted).

**Done (2026-08-02, sixth pass)**: `build_working_manifest_from_paths()`
actually split into real, separately-callable `stage0_acquire_and_copy_
sources()` / `stage1_capture_baseline_embeddings()` / `stage3_
preprocess_manifest()` functions - see "Real bug this caught, part 2:
the split" above. This was the one item explicitly deferred in the
second pass; now done.

**Done (2026-08-02, third pass)**: `scripts/build_working_manifest.py`,
`scripts/run_preprocessing.py`, `scripts/finalize_manifest.py`,
`scripts/run_semantic_stages.py`, `ui/build_manifest_ui.py` - one-line
pointers added to each docstring (old names kept, not deleted).
`scripts/run_two_stage_extraction.py` checked and deliberately
SKIPPED - its "stage 1/stage 2" is an unrelated internal concept
(extraction sub-passes within one file, not the pipeline's Stage 0-6),
a naming collision, not something to rename. Filenames NOT renamed
(backward compat preserved - nothing broken, all touched files
syntax-checked).

**Done (2026-08-02, fourth pass)**: `core/classifier.py` (Stage 5,
old "Stage 2"), `core/row_extraction.py` (Stage 6), `core/
semantic_stages.py` (between Stage 5 and 6) - one-line pointers added,
old names kept, all import-verified.

**Done (2026-08-02, fifth/final pass)**: `docs/
BENCHMARK2_METADATA_LAYER_QUALIFICATION.md` - its 2 real pipeline-stage
references (Stage 0/preprocessing, Stage 1/classification, Stage A)
now cross-reference the new numbers inline. `docs/VISION_IR_RESEARCH.md`
- one disambiguation note added near the top; its many internal "Stage
A/B/C" mentions deliberately left AS-IS, since they name research
EXPERIMENT PHASES for that document's own proposal, a different concept
from the pipeline's operational stages, not a stale reference to
rename. `benchmark/models.py`, `benchmark/prompt_sweep.py`, `benchmark/
prompt_sweep_gui.py`, `scripts/run_two_stage_extraction.py` - checked,
all deliberately SKIPPED: their "Stage 1/2" means "OCR pass / structuring
pass" within one extraction call, an unrelated internal concept, not
this pipeline's terminology.

**Rollout is now complete** for every file that actually uses this
pipeline's Stage terminology. Remaining old-name mentions in the
codebase (if any turn up later) are either already-covered bridge
references (old name kept alongside new, by design) or belong to one
of the unrelated internal "stage" concepts documented above.

No renamed file has broken any import - every touched module still
works under its existing name. Renaming actual filenames (vs. just
in-code/doc terminology) was deliberately NOT done in this pass, per
"preserve backward compatibility where practical... until all
references are updated" - flip filenames only once every reference to
the old name has been found and updated, to avoid a half-migrated state
where some code imports the old name and some imports a new alias.
