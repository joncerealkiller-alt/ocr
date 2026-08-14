# Run Architecture: Repo/Workspace Split + Hashed Runs

Status: **Both Phase 1 and Phase 2 have run for real** (2026-08-08).

**2026-08-13/14 provenance audit addendum** (repository-wide RunContext
reconciliation, verified live):

- **`RunContext.update_metadata()`** is the producer API for the four
  metadata fields that existed since day one but were null in every
  real run (`model_config`, `prompt_versions`, `preprocessing_config`,
  `artifact_summary`). Dict values merge (two stages contribute without
  clobbering); identity/lifecycle keys are rejected.
  `core/classifier.py::run(ctx=...)` is the first real producer -
  records the classifier's model_name/repo_id/**runtime**/loader_class/
  generation_config_hash + prompt version. Verified live (real Gemma
  classification, run `20260814T060340071232_af1dd7bf`).
- **Checkpoint + runtime is the provenance unit, not model name.**
  `GenerationConfig.runtime` is included in `content_hash()` (new
  hashes differ from historical ones for otherwise-identical settings -
  intended: the settings universe gained a dimension; the hash is a
  stamp, never a join key - verified no caller joins on it), and
  `core/genealogy_memory.py`'s `discoveries` gained an additive
  `runtime` column (historical rows stay NULL = genuinely unknown,
  never backfilled). `extract_fields_tool` passes it live (verified on
  WSL: real extraction recorded runtime="transformers").
- **Stray post-migration writer fixed**:
  `diagnostics/compare_v26_baseline_vs_census_checkpoint.py` was the
  only writer created AFTER the 2026-08-08 migration still targeting
  `data/outputs/`; its output (20 files, 226M) moved by same-volume
  rename to `genealogy_workspace/research/experiments/
  layout_v26_vs_census_checkpoint_comparison/` and the script
  repointed. (`data/outputs/manual_test_images/` is an INPUT image
  referenced by `scripts/real_image_negative_control_test.py` - a
  dataset-shaped item, left in place, flagged as an ownership
  question.)
- **Gemma E2B hidden-state linear probe closed** (Jon's direction):
  11 probe-only diagnostics scripts archived to
  `genealogy_workspace/research/experiments/gemma_hidden_state_probe_archive/`,
  the 153.2MB of feature caches deleted (131 files, per-file manifest
  in `deleted_caches_manifest.json` there; path keys were verified
  RESOLVABLE at deletion time - deleted because the experiment is
  closed, not because the data was broken). The live logit-margin
  adapter (`core/gemma_logit_margin_adapter.py`) is a DIFFERENT
  mechanism used by the Decision Engine and is untouched.
- **Historical conclusion corrected, raw data untouched**:
  `research/benchmarks/benchmark2_3_e2b_vs_e4b/CORRECTION_RUNTIME_CONTEXT.md`
  records that the "E4B 4.8x slower" finding describes a
  (mobile-QAT-checkpoint, transformers-runtime) combination, not the
  E4B model.
- **Intentionally separate stores confirmed, not migrated into
  RunContext** (boundary audit): `model_console/session_log.py`
  (conversation identity), `benchmark/benchmark_db.py` + prompt-sweep
  run ids (experiment semantics, [D] special-purpose),
  `core/genealogy_memory.py` (durable discoveries),
  `core/pipeline_db.py` (per-image pipeline state). Agent/chat tool
  invocations are session-scoped work, deliberately NOT pipeline runs -
  their provenance lives in session logs + genealogy memory
  (now with runtime), not in synthetic RunContexts.
- **`core/debug_dump.py`** was already RunContext-backed
  (run_type="diagnostic") with a documented flat-dir fallback - no
  change needed.

**Phase 1**: `data/working/`, `data/raw_from_pdf/`, `data/buckets/`,
`data/manifest.csv`, `data/manifest_provenance.json`, and
`data/pipeline.db` have been migrated into
`genealogy_workspace/runs/legacy_pre_run_system/` and
`genealogy_workspace/pipeline.db`, and removed from `data/`. Verified
before removal: full checksum comparison against a pre-cutover baseline
(3528 files byte-for-byte identical, all 1750 DB image rows + all 5250
sidecar-bearing stage_outputs rows resolve), a real UI
(`ui/classifier_validation_ui.py`) launched successfully against the
migrated data, and a fresh new run created and confirmed fully isolated
from the legacy run. A real end-to-end smoke test (7 real screenshots
through Stage 0-3 + Gemma classification) also caught and fixed a
run-lifecycle bug (premature `mark_completed()` blocking classification
from continuing a run) - see git history for `core/manifest_pipeline.py`
and `core/classifier.py`.

**Phase 2**: `data/outputs/` (44GB, 67 mapped items) has been
decomposed into `genealogy_workspace/{research,datasets,models}/` - see
"Phase 2 target mapping" below for the actual outcome (not a plan
anymore). See also `docs/HASHED_RUN_RESTRUCTURE_HANDOFF.md` for the
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
  Confirmed with Jon (2026-08-08): keep this hardcoded/absolute rather
  than relativizing it - a UI may load a manifest CSV directly, outside
  of any `RunContext` at all (no run_id in hand to resolve a relative
  path against), so the file needs to stay independently openable.
  Revisit only if/when every manifest-CSV consumer is guaranteed to
  have a `RunContext` available.

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
- ~~`sync_bucket_classifications()`'s internal lookup~~ **FIXED**
  (2026-08-08, revisited after the cutover before moving on to
  classification): now normalizes the bucket CSV's absolute `file_path`
  through `ctx.to_relative()` and scopes every lookup/write by
  `ctx.run_id`, matching every other stage's contract. Also fixed the
  same gap in `core/classifier.py`'s `_record_classification_db()`,
  which had been storing `lookup_key` absolute even in a `ctx`-based run
  - both `finalize_manifest()`'s and `classifier.py`'s calls into
  `sync_bucket_classifications()` now pass `ctx=ctx`. `find_stage_output()`
  stays unscoped by `run_id` by design (see its own docstring) - a
  same-named run-relative path across two different runs could in
  principle find a stale row from the wrong run, but the only
  consequence is skipping a redundant identical "failed" status
  re-write, never corrupting state.
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

## Phase 2 target mapping (executed, 2026-08-08)

`research/` ended up with four subcategories, not one flat
`experiments/` - `benchmark2*`/`vit_family_benchmark` are genuinely
benchmarks (comparing methods against each other), distinct from
one-off experiments, frozen comparison snapshots, and persistent
calibration workspaces:

```
genealogy_workspace/
    research/
        benchmarks/      # benchmark2*, vit_family_benchmark
        experiments/      # one-off investigations, logs
        baselines/         # reference_pipeline_v1-v4, _prerefactor (frozen snapshots)
        calibration/        # column_calibration, column_calibration_workspace
    models/
        checkpoints/        # all *_lora_checkpoints/, vit21k_doc_classifier_checkpoints
    datasets/
        reference/           # lac_pull_*, lac_census_pull, lac_new_years_samples
        bootstrap/            # layout_bootstrap_train (14G)
        training/              # lora_dataset (hook-protected, see below)
        ground_truth/           # ground_truth_log.jsonl, vit_finetune_dataset*.csv
    migration_manifest.json  # per-item record of what happened and why
    _workflow_gui_state.json  # tool state, not data
```

**Mechanism: `scripts/migrate_data_outputs_to_workspace.py`, driven by
a literal, pre-flight-validated mapping (67 entries, zero
pattern-matching).** 75 files across `training/`/`diagnostics/`/docs
hardcode a `data/outputs/<name>` path directly, with no shared constant
to patch centrally (unlike Phase 1's `BUCKET_DIR`-style fallback) -
rewriting all 75 was judged high-risk for low benefit, so instead: the
real data was copied and checksum-verified into
`genealogy_workspace/...`, the original at `data/outputs/<name>` was
removed, and an **NTFS reparse point** was left in its place -
`mklink /J` (junction) for a directory, `mklink /H` (hardlink) for a
loose file (junctions are directory-only and error on a file target).
Every hardcoded reference in those 75 files keeps working completely
unmodified; `open()`/`Path.exists()`/`Path.is_dir()` all follow the
link transparently. Verified post-migration: sample files read through
the OLD path are byte-identical to the new location, and a real
hardlink's `st_ino`/`st_dev` were confirmed identical to its target
(not a second copy).

**3 items got copy-only treatment, no link, original left in place**:
`row_segmentation/`, `lora_dataset/`, `scoring_reports/` are hard-blocked
from any scripted `rm -rf` by `.claude/protected_paths.txt`
(`.claude/hooks/block_protected_delete.py`, a `PreToolUse` hook - added
after a real incident where a careless cleanup command destroyed 25
manually-dewarped images). For these three, both the original
`data/outputs/<name>` and the verified copy under
`genealogy_workspace/...` coexist; removing the original and creating
the junction is a manual, explicit step for Jon to do himself whenever
ready - never a scripted one.

**`genealogy_workspace/migration_manifest.json`** records every
processed item: `old`/`new` path, `kind` (`dir`/`file`), `type`
(`junction`/`hardlink`/`copy_only`), size, file count, and (for
`copy_only` entries) `reason` - so "why is this a junction" or "why
wasn't this one deleted" never needs re-deriving from scratch.

**Benchmark convention going forward**: `research/benchmarks/<name>/`
is meant to become a container, with each future EXECUTION of that
benchmark getting its own timestamped child directory
(`research/benchmarks/vit_family_benchmark/20260808T204158/`) rather
than overwriting the previous run's output - benchmarks should stay
reproducible historical comparisons. The migrated benchmark dirs are
grandfathered in flat (each is already a single, distinct,
already-completed study, not a repeat run of the same one). This is a
target-structure decision only - no benchmark script was edited to
actually write timestamped subdirs; that's separate future work.

**Left untouched, explicitly out of scope this pass** (still in
`data/outputs/`):
- `gemma_hidden_state_probe/`, `error_analysis/` - `DO_NOT_TOUCH.md`,
  never touched without a dedicated separate ask.
- `reference_pull/`, `new_taxonomy_ground_truth.csv`/`_README.txt` -
  another session was actively writing to these during Phase 2 (last
  write ~5 min before recon); moving mid-write risks a torn copy. Move
  in a later phase 2.5 pass once that work concludes.
- `.lorabackup/` (53M, hidden/dot-prefixed) - a manual backup snapshot
  of old checkpoints/dataset/ground-truth, discovered during planning
  (missed by the initial recon since it's dot-prefixed). Not part of
  the approved mapping; left in place, undecided.
- The root `DO NOT AUTO DELETE ANYTHING HERE.txt` marker (0 bytes) -
  still protects what remains in `data/outputs/`.

## Unresolved ownership questions

- `data/subtype_review_queue.csv` - not yet fed by any production
  generator; could become a run-local queue or persistent ground truth
  depending on how the second Gemma pass it's designed for actually
  gets built. Left untouched, undecided.
- `.lorabackup/` - is this backup snapshot still worth keeping at all,
  and if so, does it deserve its own `genealogy_workspace/backups/`
  category? Deferred, see above.
- `reference_pull/`/`new_taxonomy_ground_truth.csv` - pending phase 2.5
  once the other session's active work concludes.
