# Handoff: Hashed Run-Directory Restructuring (2026-08-08)

**UPDATE (2026-08-08, later same day): Phase 1 implemented.** See
`docs/RUN_ARCHITECTURE.md` for the actual design (repo/workspace split,
`WorkspaceContext`/`RunContext`, the DB `run_id`+relative-path
invariant, what's still deferred to phase 2). This document's DO NOT
TOUCH findings below are still accurate and still apply unchanged — the
two protected directories were left exactly where they were, per the
plan's explicit scope. Everything below is now historical
context/blast-radius notes, not an open task.

Status: **Context/handoff document only.** Written so the session doing
this restructuring doesn't have to rediscover what this session already
found while doing unrelated work (Gemma hidden-state routing research,
error-analysis tooling, a taxonomy/config gap fix, and a review-tool
walkthrough). Nothing below is a plan for the restructure itself — that's
the other session's job — this is the blast-radius map and known-state
notes that should inform it.

## The stated goal (Jon's own framing, reproduced so it isn't lost)

> "each run through the pipeline requires a complete deletion of the
> prior run. I'm going to move to a hashed run hierarchy where the
> working directory is inside the hashed run folder itself, so all the
> images are stored with that specific run. this also allows the
> pipeline to be ran against a training set to see how it currently
> works as a baseline for how everything is classified"

Two goals bundled together: (1) stop clobbering `data/working/` on every
run, (2) make it possible to snapshot a full run (e.g. against a fixed
training set) as an immutable baseline for later comparison — directly
useful for the kind of before/after measurement this project already does
a lot of (fine-tune benchmarks, probe experiments, etc. all rely on a
stable, comparable snapshot).

## DO NOT TOUCH without a path-rewrite pass first

These are real, finished artifacts from this session's work (Gemma
hidden-state routing research) that a blanket "clear out `data/outputs/`
for a clean run" pass would destroy or silently orphan. They are content-
valid (the embeddings/stats are correct) but path-keyed, so a move without
updating the paths inside them makes them unresolvable, not obviously
wrong — the dangerous kind of breakage. Each directory below also has its
own `DO_NOT_TOUCH.md` marker file for visibility outside this doc.

- **`data/outputs/gemma_hidden_state_probe/`** (all of it — `shards_stage2/*.pt`,
  `features_stage1.pt`, `features_stage2.pt`, `shards/*.pt`) — the cached
  Gemma hidden-state embeddings backing every probe result in
  `docs/GEMMA_HIDDEN_STATE_ROUTING_HEAD_DESIGN.md`. Recomputing these means
  re-running Gemma inference over ~2200 images again (this is the exact
  extraction that hit the host-RAM leak and had to be run in chunked
  subprocesses — not a cheap redo).
- **`data/outputs/error_analysis/`** (all of it) — Phase 1-7 outputs
  (`review_table.csv`, `failure_annotations.json`, all `phaseN_*` stats/
  tables) from `diagnostics/error_analysis/*.py`, built directly on top
  of the directory above. `failure_annotations.json` in particular may
  gain real, non-reproducible manual review data (primary/secondary
  failure causes) once Jon works through it — that's hand-entered
  judgment, not something any script can regenerate.
- **`data/outputs/vit_family_benchmark/gemma_flat8/gemma_flat8_test_predictions.json`**
  — full-generation reference predictions this session's error-analysis
  pipeline joins against; predates this session but is load-bearing for
  the same reason.
- **`docs/GEMMA_HIDDEN_STATE_ROUTING_HEAD_DESIGN.md`,
  `docs/GEMMA_VISION_TOWER_FINETUNE_RESEARCH.md`,
  `docs/GEMMA_HIDDEN_STATE_ERROR_ANALYSIS_STAGE4_RESEARCH.md`** — research
  docs, not data, but reference the above paths by name throughout; a
  silent path rewrite elsewhere would make their own examples stale.

If the restructure needs to happen before this work is fully wrapped up,
the safe move is a **path-rewrite pass over the files above** (rewrite
the `paths`/`file_path` strings inside them to the new hashed-run
location), not a delete-and-regenerate — regenerating is possible but
expensive (a multi-hour Gemma run, per the timings in
`GEMMA_HIDDEN_STATE_ROUTING_HEAD_DESIGN.md`'s benchmark plan) and would
lose `failure_annotations.json`'s manual review work outright if any has
been done by then.

## Why this matters more than it looks like at first glance

Almost everything in this codebase joins data across files/tables using
the **absolute file path as the key**, not a stable ID. A folder
restructure that moves files without also rewriting every path reference
will silently orphan a large amount of existing work — not crash loudly,
just quietly stop matching.

## Structural path dependencies (verified this session, not guessed)

**The real database** — `core/pipeline_db.py`'s SQLite schema:
```sql
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY,
    ... source_path, working_path ...
)
CREATE TABLE IF NOT EXISTS stage_outputs (
    id INTEGER PRIMARY KEY,
    ... sidecar_path, lookup_key ...
)
```
`get_image_by_path()`/`get_or_create_image(source_path=..., working_path=...)`
key lookups on the literal path string. If images move without a
corresponding `working_path` update in this DB, `get_image_by_path()`
stops finding rows that logically still exist — this is the single
highest-value place to design the migration path through, not an
afterthought.

**Every bucket CSV** (`data/buckets/*.csv`) — `file_path` column, joined
against by `ui/classifier_validation_ui.py`, `debug_tools/
review_uncertain.py`, `scripts/reclassify_flagged_to_new_bucket.py`,
`scripts/move_flagged_for_pruning.py`, and `core/bucket_worklist.py`.

**The four flag/triage logs** (`data/misclassifications.csv`,
`data/flagged_bad_deskew.csv`, `data/flagged_for_pruning.csv`,
`data/flagged_needs_new_bucket.csv`) — all keyed on `file_path`, all
deduplicated on it too (`_load_flagged_file_paths()` in
`classifier_validation_ui.py`). **Currently these 4 files are already
deleted in the working tree** (confirmed with Jon this session — stale
from before, not from the last run, not intentional data to preserve,
he never revalidated across several quick-succession runs). Nothing to
migrate for these specifically, but any NEW ones created before this
restructure lands will need the same path-rewrite treatment.

**`data/manifest.csv`** (Stage 0 output, no category column) — read/
written by `core/manifest_pipeline.py`, `scripts/build_working_manifest.py`,
`scripts/migrate_manifest_to_db.py`, and others.

**This session's own new caches** (2026-08-07/08 — genuinely new, worth
knowing about even though unrelated to the restructure's original
motivation):
- `data/outputs/gemma_hidden_state_probe/shards_stage2/*.pt` — each shard
  stores a `"paths"` list of absolute `data/working/...` strings.
- `data/outputs/error_analysis/review_table.csv`,
  `embeddings_index.json`, `probe_softmax_probs.{pt,json}` — all keyed on
  the same absolute paths, joined against `gemma_flat8_test_predictions.json`
  by exact path match.
- **These embeddings themselves stay valid after a move** (they're
  computed from image content, not from the path) — only the path
  strings inside them go stale. If the restructure happens before any
  further work on the hidden-state-probe/error-analysis line, those
  caches either need a path-rewrite pass or a documented "these are
  pre-restructure, treat paths as historical" note — don't silently
  re-run extraction assuming the old caches are just gone, they're not,
  they're just unresolvable by path until fixed.

**Full list of files with hardcoded `data/working`, `data/buckets`,
`BUCKET_DIR`, or `data/manifest.csv` references** (grepped this session,
not exhaustively read — a starting point for the restructuring session's
own audit, not a substitute for it):

```
core/bucket_worklist.py            core/calibration_workspace.py
core/classifier.py                 core/decision_engine.py
core/extractor.py                  core/layout_detector.py
core/loaders/gemma_loader.py       core/manifest_pipeline.py
core/pipeline_db.py                core/review_modes.py
core/semantic_stages.py            debug_tools/workflow_gui.py
scripts/archive/build_manifest.py  scripts/build_flagged_images_ground_truth_sidecar.py
scripts/build_working_manifest.py  scripts/calibrate_deskew_anomaly_threshold.py
scripts/check_deterministic_rotation_signal.py
scripts/migrate_manifest_to_db.py  scripts/model_assessment.py
scripts/move_flagged_for_pruning.py scripts/pull_reference_images.py
scripts/recapture_image_analysis_raw.py
scripts/reclassify_flagged_to_new_bucket.py
scripts/reconstruct_raw_stage0.py  scripts/run_batch_auto_sidecar.py
scripts/run_semantic_stages.py     scripts/triage_flagged_deskew.py
ui/build_manifest_ui.py            ui/classifier_validation_ui.py
ui/column_calibration_ui.py        ui/dewarp_preprocessor_ui.py
ui/manual_classification_ui.py     ui/reference_pull_review_ui.py
diagnostics/test_column_boundary_projection_multiyear.py
diagnostics/test_line_detection_enhancement.py
```

## Recommended thread-through point

`core/manifest_pipeline.py` already owns Stage 0 (copy + deskew +
preprocess + merge) and is the earliest point images get a working path
assigned at all — per [[project_manifest_pipeline_stage0]] this is
already the established entry point for path-related pipeline behavior.
The hashed-run-root almost certainly belongs threaded through from here
downstream, rather than each consumer (UI tools, scripts, the DB layer)
inventing its own notion of "current run" independently.

## Other real findings from this session, tangentially relevant

- **Config-driven UI bucket lists**: `ui/classifier_validation_ui.py`
  reads its bucket list from `config/pipeline.yaml`'s `buckets:` mapping
  (deliberately, per that file's own docstring — never hardcode buckets
  in UI code). That mapping was just fixed this session to include
  `photo_collage`/`casual_photo`/`cemetery_photo` (previously missing
  since 2026-08-04's taxonomy addition never got mirrored into
  `pipeline.yaml`). If the restructuring session touches config loading
  at all, be aware this pattern exists and should be preserved, not
  bypassed.
- **Manual reclassification tool chain** (for context, not directly
  path-restructure-relevant): flag in `classifier_validation_ui.py` →
  label the correct answer in `debug_tools/review_uncertain.py --source
  misclassifications` → optionally move buckets via `scripts/
  reclassify_flagged_to_new_bucket.py`. All three are part of the same
  absolute-path-keyed chain described above.
- **A real, unresolved host-RAM leak** was found and worked around (not
  fixed) in this session's own new diagnostic scripts
  (`diagnostics/gemma_hidden_state_extract_worker*.py`) — ~150MB/image
  RSS growth during Gemma forward passes, root cause not isolated,
  mitigated by running extraction in short-lived subprocesses instead of
  one long-running process. Not obviously related to the folder
  restructure, but worth knowing about if the restructuring session ends
  up touching `core/loaders/gemma_loader.py` or anything that holds a
  Gemma model resident for a long-running batch job — a hashed-run batch
  process that iterates a whole training set through the pipeline in one
  process could hit the same wall.

## What this document is NOT

Not a design for the hashed-run hierarchy itself (naming scheme, hash
input, where the root lives, migration script design) — that's the
restructuring session's own work. This is scoped to "here's what already
depends on today's flat path structure, verified, so you don't have to
re-grep the whole codebase to find out."
