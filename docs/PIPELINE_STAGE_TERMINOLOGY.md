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
| 2 | **Decision Engine** | Determine required preprocessing operations from Stage 1 evidence | NOT STARTED (previously called "Stage B") |
| 3 | **Image Processing** | Execute preprocessing ONLY | `core/manifest_pipeline.py`'s `preprocess_for_manifest()` (deskew + `core/image_preprocessing.py` profile) |
| 4 | **Validation Capture** | Repeat Stage 1's measurements after processing | NOT STARTED (previously called "postprocessing capture" in the Benchmark 2 metadata design doc) |
| 5 | **Document Routing** | Classification and routing | `core/classifier.py`, the Multi-Tower Routing Audit (`docs/BENCHMARK2_3_MULTI_TOWER_ROUTING_AUDIT.md`), the E2B/E4B escalation policy |
| 6 | **Specialized Extraction** | OCR, tables, handwriting, maps, etc. | `core/row_extraction.py`, `core/semantic_stages.py`, model loaders |

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

## Old -> new name mapping (for anything still using old terms)

- "Stage 0" (`core/manifest_pipeline.py`'s old all-in-one meaning) ->
  split across new Stage 0 (Source Acquisition) + Stage 1 (Raw Sensor
  Capture) + Stage 3 (Image Processing).
- "Stage A" (`core/image_analysis.py`) -> Stage 1, physical half.
- "Stage B" (not-yet-built profile selection) -> Stage 2 (Decision
  Engine).
- "Stage 2" (`core/manifest_pipeline.py`'s old finalize-after-
  classification meaning, `finalize_manifest()`) -> this was actually
  post-Stage-5 bookkeeping (merging classification buckets + manual
  dewarp output) - re-examine whether it belongs under Stage 5
  (Document Routing) or needs its own place once the full rollout
  happens; not resolved in this pass.

## Rollout status (2026-08-02) - what's done, what's pending

**Done this pass**:
- This terminology doc created as the canonical reference.
- `docs/REFERENCE_PIPELINE_V1.md` written, archive captured.
- `docs/PREPROCESSING_STAGE_NOTES.md` and `docs/CODE_MAP.md` updated to
  reference the new stage numbers (see those files' own 2026-08-02
  entries).

**NOT yet done - flagged so a future session doesn't assume complete**:
- `core/manifest_pipeline.py`, `core/image_analysis.py`, `core/
  baseline_embeddings.py` docstrings/comments still use old "Stage 0"/
  "Stage A" language in places - needs a pass to update in-code
  references to the new numbering, ideally alongside actually splitting
  `build_working_manifest_from_paths()` into real Stage 0 / Stage 1 /
  Stage 3 sub-functions (ⁿot done yet - would need care to avoid
  breaking the working pipeline; recommend doing this as its own
  focused task, not rushed).
- `scripts/build_working_manifest.py`, `scripts/run_preprocessing.py`,
  `scripts/finalize_manifest.py`, `scripts/run_semantic_stages.py`,
  `scripts/run_two_stage_extraction.py`, `ui/build_manifest_ui.py` -
  CLI/UI entry points, still using old terminology in help text/
  docstrings. Filenames NOT renamed yet (backward compat preserved by
  default - nothing broken).
- `core/classifier.py`, `core/row_extraction.py`, `core/semantic_stages.py`
  - Stage 5/6 modules, not yet updated to reference new numbering.
- `docs/BENCHMARK2_METADATA_LAYER_QUALIFICATION.md`, `docs/
  VISION_IR_RESEARCH.md`, `benchmark/models.py`, `benchmark/
  prompt_sweep.py`, `benchmark/prompt_sweep_gui.py` - other stage
  references not yet updated (lower priority - these are
  research/benchmark docs and tools, not the live orchestration path).

No renamed file has broken any import - every touched module still
works under its existing name. Renaming actual filenames (vs. just
in-code/doc terminology) was deliberately NOT done in this pass, per
"preserve backward compatibility where practical... until all
references are updated" - flip filenames only once every reference to
the old name has been found and updated, to avoid a half-migrated state
where some code imports the old name and some imports a new alias.
