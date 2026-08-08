# Run Architecture: Repo/Workspace Split + Hashed Runs

Status: **Phase 1 of 2 implemented AND the real cutover has run**
(2026-08-08). `data/working/`, `data/raw_from_pdf/`, `data/buckets/`,
`data/manifest.csv`, `data/manifest_provenance.json`, and
`data/pipeline.db` have been migrated into
`genealogy_workspace/runs/legacy_pre_run_system/` and
`genealogy_workspace/pipeline.db`, and removed from `data/`. Verified
before removal: full checksum comparison against a pre-cutover baseline
(3528 files byte-for-byte identical, all 1750 DB image rows + all 5250
sidecar-bearing stage_outputs rows resolve), a real UI
(`ui/classifier_validation_ui.py`) launched successfully against the
migrated data, and a fresh new run created and confirmed fully isolated
from the legacy run. Phase 2 (reorganizing existing `data/outputs/`
research/dataset/checkpoint content into
`genealogy_workspace/{research,datasets,models}/`, UI reorg, root
cleanup) is planned but **not executed** - see "Phase 2 target mapping"
below. See also `docs/HASHED_RUN_RESTRUCTURE_HANDOFF.md` for the
original blast-radius/DO-NOT-TOUCH findings this work was scoped
against.

## Why

Before this change, every pipeline execution wrote into fixed, mutable
locations (`data/working/`, `data/raw_from_pdf/`, `data/manifest.csv`,
`data/buckets/`) inside the repo itself - a new run clobbered the
previous one's working images, manifest, and buckets. There was no way
to keep two runs side by side, resume an interrupted run, or run the
pipeline against a fixed training set as a comparable baseline. Every
join across files/tables (buckets, flag CSVs, `pipeline_db.py`) used the
absolute file path as the key, so generated data and source code were
also physically entangled in one git-tracked tree.

## The two-root split

```
J:\Genealogy\
├── genealogy_pipeline\      (this repo - code, config, docs, UI, tests)
└── genealogy_workspace\     (sibling dir - all generated/persistent data)
    ├── runs\
    │   └── <run_id>\
    │       ├── metadata.json
    │       ├── manifest\
    │       │   ├── manifest.csv
    │       │   └── manifest_provenance.json
    │       ├── raw_from_pdf\
    │       ├── working\
    │       │   ├── images\
    │       │   └── analysis\        (*_analysis.json / *_preprocess.json sidecars)
    │       ├── outputs\
    │       │   ├── buckets\
    │       │   ├── extraction\
    │       │   ├── routing\
    │       │   └── embeddings\
    │       ├── quarantine\
    │       ├── diagnostics\
    │       ├── reports\
    │       └── config_snapshot\
    ├── pipeline.db
    └── unmigrated\               (phase-2 target - not created yet)
```

Both roots are independently configurable, never hardcoded:

1. `GENEALOGY_PIPELINE_ROOT` / `GENEALOGY_WORKSPACE_ROOT` env vars
2. `config/workspace.yaml` (`pipeline_root:` / `workspace_root:` - ships
   with both `null`, deliberately never carrying a machine-specific
   absolute path; use `config/workspace.local.yaml`, gitignored, for that)
3. Fallback: `pipeline_root` = this repo's own root;
   `workspace_root` = `pipeline_root.parent / "genealogy_workspace"`

Implemented in `core/workspace_context.py`'s `WorkspaceContext`.

## Run IDs

`YYYYMMDDTHHMMSSffffff_<8-char-hash>` (`core/run_context.py`,
`RunContext.create()`):

- The microsecond timestamp makes the run_id itself unique (not just
  orderable) - two runs starting within the same second must not
  collide. `RunContext.create()` additionally checks for an existing
  directory and appends `_2`, `_3`, ... in the astronomically unlikely
  case of a real collision (clock rollback, mocked time in tests).
- The hash is `sha256(canonical_json({source_input, config file
  contents}))[:8]` - a content fingerprint of `pipeline.yaml` +
  `taxonomy.yaml` + `decision_engine.yaml` plus the source input
  identifier. Two runs against the same source with the same config get
  the *same* hash segment but different timestamps: the hash lets a
  human glance at two run_ids and see "these were functionally
  equivalent runs," it is not what makes the id unique.

`run_type` is one of `production, dataset_build, benchmark, research,
training, calibration, validation, diagnostic`, plus one **reserved**
internal value `legacy` used exclusively by
`scripts/migrate_pipeline_db_to_run_schema.py`'s synthetic
`legacy_pre_run_system` run - `RunContext.create()` rejects `legacy` if
a normal caller passes it.

Runs are immutable once `mark_completed()` is called: `RunContext.resume()`
raises on a completed run. A rerun always gets a new run_id;
"resuming" only applies to an in-progress run.

## The ownership model

Ask this question whenever a new artifact's location is unclear:

| Question | Answer | Home |
|---|---|---|
| "If I run the pipeline again tomorrow, does a new copy get produced?" | Yes | `genealogy_workspace/runs/<run_id>/...` |
| "Is this curated input meant to be reused across many runs?" (human-reviewed ground truth, reference pulls, training corpora) | Yes | `genealogy_workspace/datasets/...` (phase 2) |
| "Is this a trained model/checkpoint meant to be reused?" | Yes | `genealogy_workspace/models/...` (phase 2) |
| "Does this represent an experiment/benchmark/study, not a normal run?" | Yes | `genealogy_workspace/research/...` (phase 2) |
| "Is this a human-reviewed historical report spanning many runs?" | Yes | stays outside `runs/`, currently `data/logs/` |
| "Can this disappear without losing reproducibility or human work?" | Yes | disposable/temporary |

Concrete examples decided during this migration:

- `data/misclassifications.csv` (per `core/review_modes.py`'s own
  docstring: "a flat ground-truth sample, not a queue") → **persistent
  dataset**, not run-local. Left untouched.
- `data/buckets/uncertain_review.csv` → a routing bucket CSV, same
  lifecycle as every other `DocumentCategory` bucket CSV → **run-local**,
  now under `ctx.buckets`.
- `data/outputs/gemma_hidden_state_probe/`, `data/outputs/error_analysis/`
  (both have `DO_NOT_TOUCH.md` markers) → pre-run-system research
  artifacts, multi-hour-to-regenerate, contain hand-entered review data
  → **left exactly where they are**, indefinitely, until a dedicated
  path-rewrite pass is explicitly requested.
- **Baseline/postprocessing/layout embeddings are run-owned.** Every
  future run captures its own `outputs/embeddings/baseline_embeddings.json`
  / `postprocessing_embeddings.json` under `ctx.embeddings` -
  `core/manifest_pipeline.py`'s `stage1_capture_baseline_embeddings()`,
  `stage1_capture_layout_detections()`, and
  `stage4_capture_postprocessing_embeddings()` all route their
  `write_baseline_embeddings()`/`write_layout_detections()` output
  through `ctx.embeddings` when `ctx` is given, instead of
  `core/baseline_embeddings.py`'s fixed `DEFAULT_BASELINE_PATH`/
  `DEFAULT_POSTPROCESSING_PATH` (`data/baseline_embeddings.json`, the
  pre-migration 112MB file - that file remains the **legacy run's**
  captures, reachable only via the `legacy_pre_run_system` fallback
  path, not a special-cased shared global anymore). This also means
  `capture_baseline=True`/`capture_layout=True` are now safe to use in
  a `ctx`-based run (including throwaway/test runs) - they were
  previously flagged as dangerous specifically because they wrote into
  that one shared file; that risk no longer exists once `ctx` is
  passed.

## `RunContext` API (`core/run_context.py`)

```python
ctx = RunContext.create(workspace, run_type="production", source_input=str(folder), run_name="optional label")
ctx.run_id            # "20260808T193050927883_ec3556b6"
ctx.run_root           # workspace.runs_root / ctx.run_id
ctx.manifest_csv        # run_root/manifest/manifest.csv
ctx.raw_from_pdf        # run_root/raw_from_pdf
ctx.working_images      # run_root/working/images
ctx.working_analysis    # run_root/working/analysis
ctx.buckets             # run_root/outputs/buckets  (NOT a run-root sibling)
ctx.extraction / .routing / .embeddings   # run_root/outputs/*
ctx.reports             # run_root/reports
ctx.config_snapshot     # run_root/config_snapshot

ctx.to_relative(path)    # canonical DB-boundary normalization: absolute Path under
                          # ctx.run_root -> run-relative string. Raises ValueError
                          # for a path outside the run (e.g. an external source).
ctx.to_absolute(rel)     # inverse

ctx.mark_completed() / ctx.mark_failed(error)
RunContext.resume(workspace, run_id)
```

## Run naming

`run_name` is a human label layered on top of the immutable `run_id`/
`run_type` (e.g. "production", "test_run_2026-08-08",
"dataset_baseline_v1"). `core/manifest_pipeline.py`'s CLI entry point
(`python -m core.manifest_pipeline <source>`) prompts once for it
interactively (`Run name (optional, press Enter to skip): `) **only**
when stdin is a tty and `--run-name` wasn't passed - a scripted/
subprocess caller must pass `--run-name` explicitly or accept `None`,
never hits the prompt. `scripts/build_working_manifest.py --new-run`
has the same `--run-name`/`--run-type` flags.

## The DB identity contract (`core/pipeline_db.py`)

**The single invariant every row obeys, old or new:**
`resolve_path(run_id, relative_path) == workspace.runs_root / run_id /
relative_path`. `PipelineDatabase.resolve_path()` derives `runs_root`
from **its own `db_path`'s parent**, not a freshly re-resolved
`WorkspaceContext` - this keeps it correct even when the ambient env/
config `WorkspaceContext` differs from the one a given database instance
was actually opened against (tests, or a moved workspace).

- `images.run_id` (new column) tags every row with the run that
  produced it.
- `images.working_path` / `source_path` and `stage_outputs.sidecar_path`
  / `lookup_key` are stored **run-relative** for any run-owned artifact
  (an external `source_path`, e.g. `J:\Screenshots\...\foo.pdf`, is
  never relativized - `RunContext.to_relative()` raises if you try).
- Every call site normalizes through `ctx.to_relative()` before writing,
  and passes `run_id=ctx.run_id` on every read
  (`get_image_by_path`/`get_or_create_image`) so two different runs'
  same-named files never collide in a lookup.
- `manifest.csv`/bucket CSVs themselves still store **absolute** paths
  (by design - they're read directly, as real openable paths, by many
  tools that don't go through the DB or `RunContext` at all). Only the
  DB's own persisted identity is normalized. This is a deliberate,
  explicit decision, not an oversight: it avoids the DB persistence
  format depending on how the CSV happens to represent a path.

### Migration script

`scripts/migrate_pipeline_db_to_run_schema.py` moves the pre-run-system
`data/` layout into a real synthetic run, `legacy_pre_run_system`, and
rewrites `pipeline.db` so it obeys the same invariant as every future
run - no permanent special-casing of old data. Validated against real-
data **copies** before ever touching production files (1750/1750 images,
12250/12250 `stage_outputs` rows preserved, every path resolves). See
that script's own docstring for the exact copy-then-verify workflow and
CLI usage (`--data-root` vs `--source-data-root` for validation runs).

## Known gaps (deliberately out of scope this pass)

- **FIXED during cutover, not a remaining gap**: `ui/classifier_validation_ui.py`'s
  `MISCLASSIFICATION_LOG`/`BAD_DESKEW_LOG`/`PRUNE_LOG`/`NEEDS_BUCKET_LOG`
  were derived as `BUCKET_DIR.parent / "..."` - correct pre-migration
  (`BUCKET_DIR.parent` was `data/`), but once `core/classifier.py`'s
  `BUCKET_DIR` started resolving into a run directory this would have
  silently pointed all four persistent, cross-run logs at
  `<run>/outputs/` instead of `data/`. Caught during the real cutover's
  "launch a real UI" verification step, before it could bite -
  re-anchored to a fixed `data/` reference instead. Same class of bug
  as the `review_modes.py` ownership question flagged earlier in this
  session: a file being *reachable through* a run-owned constant isn't
  the same as the file *being* run-owned.
- **`core/image_analysis.py`'s per-image sidecar location**: currently
  colocated with the working image (`working/images/`), not split into
  `working/analysis/` the way the migration script's *target* layout
  does for legacy data. Cosmetic, not a correctness issue - paths still
  resolve fine via `ctx.to_relative()`.
- ~~`core/baseline_embeddings.py`'s shared-file risk~~ **RESOLVED**:
  embeddings are now run-owned (see above) whenever `ctx` is passed -
  `capture_baseline=True`/`capture_layout=True` are safe in a `ctx`-based
  run. The remaining caveat is narrower: `core/manifest_pipeline.py`'s
  `build_working_manifest()` (folder adapter) still has no
  `capture_baseline`/`capture_layout` override params of its own and
  always defaults both to `True` - fine when you pass `ctx=`, but a
  call **without** `ctx` still targets the real, shared
  `data/baseline_embeddings.json` (the pre-migration file, now
  reachable as the `legacy_pre_run_system` run's data) - a pre-existing
  property of that function, not new risk introduced here.
- **`core/calibration_workspace.py`**: explicitly deferred to phase 2 -
  a persistent evaluation workspace, not per-run state.
- **`sync_bucket_classifications()`'s internal lookup**: reads bucket
  CSVs' absolute `file_path` column and calls `get_image_by_path()`
  unscoped by `run_id` - for a run-based DB this won't match a
  run-scoped row. Acceptable for this pass (Jon: "CSV paths can be
  fixed later, this is a restructuring phase") - tracked as follow-up,
  not silently ignored.
- **3 GUI tools not yet ctx-aware**: `ui/build_manifest_ui.py`,
  `ui/classifier_validation_ui.py`, `ui/dewarp_preprocessor_ui.py`.
  Confirmed safe as-is (they keep working against the legacy run via
  the dynamic `DEFAULT_*` fallback constants in `manifest_pipeline.py`/
  `pipeline_db.py`/`classifier.py`), they just can't target a *new*
  hashed run from the GUI yet.
- **~28 scripts/diagnostics files** listed in
  `docs/HASHED_RUN_RESTRUCTURE_HANDOFF.md` - mostly one-off calibration/
  dataset/diagnostic tools operating on the flat legacy corpus (phase-2/
  dataset territory), confirmed safe unmodified via the same fallback.

## Phase 2 target mapping (planned, not executed)

| Category | Current location | Target |
|---|---|---|
| Research experiments (`benchmark2*`, `gemma_reasoning_consistency_audit/`, etc.) | `data/outputs/*` | `genealogy_workspace/research/experiments/` |
| Protected caches (`DO_NOT_TOUCH.md`) | `data/outputs/gemma_hidden_state_probe/`, `data/outputs/error_analysis/` | **Stay exactly where they are, indefinitely** |
| LoRA/model checkpoints | `data/outputs/*_lora_checkpoints/` | `genealogy_workspace/models/checkpoints/` |
| Dataset pulls / curated corpora | `lac_pull_*/`, `reference_pull/`, `lora_dataset/` | `genealogy_workspace/datasets/{reference,training,bootstrap}/` |
| Ground truth | `new_taxonomy_ground_truth.csv`, `data/misclassifications.csv`, `ground_truth_log.jsonl` | `genealogy_workspace/datasets/ground_truth/` |
| Persistent logs/reports | `data/logs/`, `reviewed_uncertain.csv` | `genealogy_workspace/logs/` |
| Root generated artifacts | `benchmark_results/`, `experiment3_*_outputs/`, `yolo26n.pt` | `genealogy_workspace/research/...`, `genealogy_workspace/models/` |
| UI mockups | `UI Mockups/` | `docs/ui_mockups/` (stays in repo - design artifact) |

## Unresolved ownership questions

- `data/subtype_review_queue.csv` - not yet fed by any production
  generator; could become a run-local queue or persistent ground truth
  depending on how the second Gemma pass it's designed for actually
  gets built. Left untouched, undecided.
- `column_calibration/`, `row_segmentation/` workspaces - persistent
  evaluation/training workspaces per the original plan, but individual
  items may turn out to be run-tied on closer inspection; needs a
  per-item ownership check before phase 2 moves them.
- `reference_pipeline_v1..v4`, `data/reference_prerefactor/` - look like
  frozen comparison snapshots, not experiments; phase 2 should confirm
  before choosing `research/reports/` vs. a dedicated `baselines/`.
