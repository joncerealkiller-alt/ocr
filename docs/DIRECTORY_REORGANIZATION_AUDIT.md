# Directory reorganization audit (2026-08-02)

**Read-only audit. No files moved, no code changed.** Produced by
grepping the entire repository for filesystem-path references and
inspecting `data/outputs/`'s actual contents. Written up per Jon's
5-phase request; see that request for the phase definitions.

## Lead finding — read this before anything else

**`data/pipeline.db` and `data/baseline_embeddings.json` store absolute
filesystem paths as DATA, not just as code references.** Every row in
`pipeline.db`'s `images` table has a `working_path`/`source_path`
column holding a real absolute path string (e.g.
`J:\Genealogy\genealogy_pipeline\data\working\foo.jpg`) — `core/
pipeline_db.py`'s `get_image_by_path()` looks images up by exact string
match against these columns. `baseline_embeddings.json` similarly keys
each record by an `"image"` path string (already confirmed this session
to be inconsistently relative-vs-absolute in different records — see
`scripts/migrate_manifest_to_db.py`'s `_to_abs()` normalizer, added
after that exact ambiguity caused a real bug).

**Consequence: moving `data/working/` (or any directory a file's
`working_path` currently points into) is NOT a simple folder rename.**
No grep will catch this, because the "reference" isn't in any script —
it's baked into 1677+ rows of a SQLite file and a 106MB JSON file. A
plain `mv data/working data/somewhere/working` would leave every one of
those paths silently pointing at a location that no longer exists;
every `get_image_by_path()` call in Stage 0/1/2/3/5 and `finalize_
manifest()` would then either fail to find the image or (worse) find
nothing and silently skip it.

**This must be treated as its own migration step, not a side effect of
a folder move**: rewrite every `source_path`/`working_path` value in
`pipeline.db` and every `"image"` key in `baseline_embeddings.json` to
the new location, THEN move the folder, verified with the same
before/after diff discipline already used for `finalize_manifest()`'s
own migration (`docs/PIPELINE_DATABASE.md`) — never move first and fix
paths after.

Everything below covers the more conventional (but still real) code-reference risks.

## Phase 1 — Path reference inventory

Grepped for `PROJECT_ROOT / "..."` (the dominant convention - ~180
matches across the repo), plus targeted searches for env vars, config
YAML path references, and the LoRA-checkpoint naming pattern
specifically (called out in Jon's own example).

### The dominant, consistent pattern

Nearly every `.py` file that touches the filesystem defines
`PROJECT_ROOT = Path(__file__).resolve().parent.parent` at module top,
then builds constants like `BUCKET_DIR = PROJECT_ROOT / "data" /
"buckets"`. This is NOT centralized (each file redeclares its own copy
of the same constant - `BUCKET_DIR` alone is independently redefined in
at least 6 files: `core/classifier.py`, `debug_tools/
review_uncertain.py`, `core/decision_engine.py`, `core/
manifest_pipeline.py`, `core/pipeline_db.py`, `benchmark/
vision_encoder_qualification.py`, `benchmark/benchmark2_2_cross_
validator.py`, several `scripts/*.py`), but it IS consistent - every
redeclaration resolves to the identical real path. This is exactly what
makes a `core/paths.py` centralization low-risk: it's a mechanical
"replace N identical redeclarations with one import" change, not a
behavior change.

### Where `data/` gets referenced, by category

- **Core pipeline modules** (`core/manifest_pipeline.py`, `core/
  classifier.py`, `core/pipeline_db.py`, `core/decision_engine.py`,
  `core/baseline_embeddings.py`, `core/extractor.py`, `core/
  image_analysis.py`, `core/pdf_conversion.py`) - `data/working`,
  `data/buckets`, `data/manifest.csv`, `data/manifest_provenance.json`,
  `data/manifest_final.csv`, `data/pipeline.db`, `data/
  baseline_embeddings.json`, `data/outputs/image_analysis`, `data/
  raw_from_pdf`, `data/outputs` (general).
- **UI tools** (`ui/*.py`, `debug_tools/*.py`) - same core paths plus
  their own: `data/outputs/dewarped`, `data/outputs/row_segmentation`,
  `data/outputs/auto_row_segmentation`, `data/outputs/
  ground_truth_log.jsonl`, `data/outputs/_workflow_gui_state.json`,
  `data/misclassifications.csv`, `config/columns`.
- **Benchmark scripts** (`benchmark/*.py`, ~30 files) - each defines its
  own `OUT_DIR`/`RESULTS_LOG` under `data/outputs/<benchmark_name>/` -
  by far the largest single source of distinct path constants (one per
  benchmark script), and the least likely to ever be touched again once
  a benchmark is "frozen" (several are explicitly documented as frozen
  this session: Vision Qualification Battery v1.0, Multi-Tower Routing
  Audit).
- **Training scripts** (`training/*.py`) - `data/outputs/
  lora_dataset`, `data/outputs/<model>_lora_checkpoints/`, `data/
  outputs/ground_truth_log.jsonl`.
- **Diagnostics/scripts** (`diagnostics/*.py`, `scripts/*.py`) -
  overwhelmingly reference `data/outputs/dewarped`, `data/buckets`,
  `config/prompts`, `config/models` - genuinely diagnostic/one-off
  tools, all following the same `PROJECT_ROOT` convention.
- **Config files** - `config/pipeline.yaml:124` has one literal
  relative path (`log_file: data/audit_log.csv`) not anchored to
  anything - the one config-file path reference found.
  `config/document_templates/*.yaml` mention `data/outputs/
  row_segmentation/` only in COMMENTS (calibration provenance notes),
  not live paths.
- **Environment variables** - **none used for paths anywhere in this
  project's own code.** The only `os.environ`/`os.getenv` hits outside
  `.venv_*` vendored packages are `core/loaders/chandra_loader.py`
  (sets `MAX_OUTPUT_TOKENS`, a model config value) and `ui/
  build_manifest_ui.py` (sets `PYTHONUNBUFFERED` for a subprocess) -
  neither is path-related. This significantly simplifies centralization:
  there's no env-var override layer to reconcile.
- **Documentation** - `README.md`, `PIPELINE_WORKFLOW.md`, `docs/
  CODE_MAP.md`, `docs/PIPELINE_DATABASE.md` all reference the same
  real paths (`data/manifest.csv`, `data/buckets/*.csv`, `data/
  pipeline.db`, etc.) as worked examples/usage instructions - these are
  documentation, not runtime dependencies, but would need updating
  for consistency if paths move (cosmetic risk, not functional).

### Real inconsistencies found (worth fixing regardless of any move)

1. **`training/train_lora.py:252`** - `out_dir = Path(args.out_dir) if
   args.out_dir else Path(f"data/outputs/{args.model}_lora_checkpoints")`.
   This is the ONE place in the entire codebase that builds a data path
   as a **bare relative string**, not anchored to `PROJECT_ROOT`. If
   this script is ever invoked from a working directory other than the
   project root, it silently writes checkpoints to the wrong place -
   already a latent bug, independent of any reorganization.
2. **Four benchmark scripts write OUTSIDE `data/outputs/` entirely**,
   directly under the project root: `benchmark/experiment3_1_crop_
   confound_test.py` (`PROJECT_ROOT / "experiment3_1_outputs"`),
   `experiment3_2_robust_localization.py` (`experiment3_2_outputs`),
   `experiment3_3_gradcam_localization.py` (`experiment3_3_outputs`),
   and `benchmark/benchmark_db.py` (`PROJECT_ROOT / "benchmark_
   results"`). Every other benchmark script uses `data/outputs/
   <name>/` - these four are the exception, and exactly the kind of
   stray top-level clutter a centralization pass should fix.
3. **The `_lora_checkpoints` naming convention is duplicated, not
   shared**: `debug_tools/workflow_gui.py`'s `_scan_checkpoints()`
   builds `OUTPUTS_DIR / f"{model_name}_lora_checkpoints"` in 4 separate
   call sites within that one file; `training/train_lora.py` builds the
   same string independently (and, per #1, without even anchoring it).
   No shared function computes "the checkpoint directory for model X" -
   a real duplication risk if the naming convention ever needs to
   change.
4. **`.lorabackup`** (`data/outputs/.lorabackup/`) is referenced by
   **zero code** anywhere in the repo (confirmed by grep) - it's a
   manually-created backup snapshot (contains copies of `ground_truth_
   log.jsonl`, `lora_checkpoints`, `lora_checkpoints_smoketest`,
   `lora_dataset`), not a live pipeline path. Completely safe to
   move/rename - nothing will notice.

## Phase 2 — Classification of every `data/outputs/` directory

| Directory | Files | Classification | Why |
|---|---|---|---|
| `auto_row_segmentation_batch_test` | 49 | Temporary Working Data | Name says "test"; `scripts/run_batch_auto_sidecar.py`'s real output dir is `auto_row_segmentation` (no `_batch_test` suffix) - this looks like a one-off test run left behind. |
| `benchmark2` | 67 | Benchmark Artifact | `benchmark/benchmark2_pilot_isolated.py`'s `BENCHMARK2_DIR`. |
| `benchmark2_2` | 129 | Benchmark Artifact | `benchmark/benchmark2_2_cross_validator.py`'s `OUT_DIR`. |
| `benchmark2_3_e2b_vs_e4b` | 3 | Benchmark Artifact | Frozen result (`docs/BENCHMARK2_3_E2B_VS_E4B.md`). |
| `benchmark2_3_multi_tower` | 115 | Benchmark Artifact | Frozen result (Multi-Tower Routing Audit) - other benchmarks (`benchmark2_3_e2b_vs_e4b`, `benchmark2_4_selective_escalation`) read `all_results.json` FROM here, so it's a dependency for those, not just a leaf. |
| `benchmark2_4_selective_escalation` | 2 | Benchmark Artifact | Reads from `benchmark2_3_multi_tower` + `benchmark2_3_e2b_vs_e4b`. |
| `benchmark2_gemma_input_qualification` | 81 | Benchmark Artifact | Contrast/grayscale/sharpening/denoise/resolution/resize-algorithm sub-experiments, all frozen (`docs/BENCHMARK2_3_GEMMA_INPUT_QUALIFICATION.md`). |
| `reference_classifier_qualification` | 2 | Benchmark Artifact | `docs/REFERENCE_CLASSIFIER_QUALIFICATION.md`'s frozen result. |
| `granite_vision_2b_lora_checkpoints` | 0 | Model Checkpoint (empty) | Naming matches the convention; currently no epochs saved. |
| `internvl3_2b_lora_checkpoints` | 0 | Model Checkpoint (empty) | Same. |
| `qwen25_vl_7b_lora_checkpoints` | 0 | Model Checkpoint (empty) | Same. |
| `qwen2b_lora_checkpoints` | 0 | Model Checkpoint (empty) | Same. |
| `qwen3vl2b_lora_checkpoints` | 12 | Model Checkpoint | Real saved epochs (`training/train_lora.py`'s output). |
| `qwen3vl2b_thinking_lora_checkpoints` | 12 | Model Checkpoint | Same. |
| `qwen3vl4b_lora_checkpoints` | 12 | Model Checkpoint | Same. |
| `smolvlm_lora_checkpoints` | 47 | Model Checkpoint | Same (most epochs of any model). |
| `lora_dataset` | 752 | Dataset | `training/export_lora_dataset.py`'s `DEFAULT_OUT` - training examples exported from `ground_truth_log.jsonl`, consumed by `training/train_lora.py`. |
| `prompt_sweep_runs` | 168 | Benchmark Artifact | `benchmark/prompt_sweep.py`/`prompt_sweep_gui.py`'s output. |
| `scoring_reports` | 3 | Pipeline Output | `debug_tools/workflow_gui.py`'s two-stage extraction scoring - real per-page evaluation output, not a benchmark experiment. |
| `reference_pipeline_v1` | 1679 | Reference Archive | Explicitly frozen (`docs/REFERENCE_PIPELINE_V1.md`) - "do not overwrite or delete" per its own doc. |
| `reference_pipeline_prerefactor` | 83 | Reference Archive | Same class as above - a snapshot, not live data (predates this session's `data/reference_prerefactor/`, a DIFFERENT, top-level-`data/` snapshot from today's move - two similarly-named but distinct archives now exist, worth reconciling naming). |
| `.lorabackup` | (4 subdirs) | Reference Archive / Cache | Manual snapshot, zero code references (see Phase 1, finding #4). |
| *(top-level files)* `ground_truth_log.jsonl`, `*_log.jsonl` (7 files), `benchmark2_pilot_checkpoints.json` | - | Log | Append-only event/result logs, each written by exactly one script. |
| `_workflow_gui_state.json` | - | Cache | `debug_tools/workflow_gui.py`'s persisted UI state (last-used paths/selections) - regenerable, not data. |
| `DO NOT AUTO DELETE ANYTHING HERE.txt` | - | *(marker, not data)* | A standing instruction file - **preserve this marker's INTENT (nothing here should be auto-deleted) in whatever new location replaces this folder**, don't just discard it because it's not "real" data. |

**Not classified above but relevant**: `data/outputs/dewarped`,
`data/outputs/row_segmentation`, `data/outputs/auto_row_segmentation`,
`data/outputs/image_analysis`, `data/outputs/model_assessments` are
all REAL, referenced-by-code destinations (Phase 1) that don't
currently exist as populated directories in this corpus snapshot (no
dewarp has been run yet in production, e.g.) - they're Pipeline Output
locations that will be created on first use, not "missing" or
erroneous.

## Phase 3 — Migration table

| Current Location | Proposed Location | Purpose | Referenced By | Safe to Move? |
|---|---|---|---|---|
| `data/working/` | *(do not move without a path-rewrite step first)* | Working copies of every acquired image | `pipeline.db` (every row's `working_path`), `baseline_embeddings.json` (every record's `"image"` key), every `core/*.py` stage, all bucket CSVs (absolute paths), `manifest.csv`/`manifest_provenance.json` | **No — highest-risk item in this audit. See "Lead finding" above.** |
| `data/manifest.csv`, `data/manifest_provenance.json`, `data/buckets/*.csv` | `data/pipeline/manifest.csv` etc. (if reorganizing) | Stage 0-5 CSV artefacts, git-tracked | `core/manifest_pipeline.py`, `core/classifier.py`, `finalize_manifest()`, `debug_tools/review_uncertain.py`, `ui/dewarp_preprocessor_ui.py`, `ui/build_manifest_ui.py` | No — update every `DEFAULT_MANIFEST_PATH`/`BUCKET_DIR` constant first (mechanical, ~8 files, all via the same redeclared constant pattern) |
| `data/pipeline.db` | `data/pipeline.db` (recommend NOT moving) | Orchestration DB | `core/pipeline_db.py`'s `DEFAULT_DB_PATH`, every stage function's `db_path` default | Technically movable (one constant), but its ROWS reference `data/working/` paths - moving the DB alone doesn't help unless `data/working/` moves too, in lockstep, with a path-rewrite |
| `data/baseline_embeddings.json` | *(same caution as data/working/)* | Semantic sensor evidence, shared file | `core/baseline_embeddings.py`, `core/decision_engine.py`, `scripts/migrate_manifest_to_db.py` | No — same reason as `data/working/`: paths are DATA, not just a code reference |
| `data/outputs/*_lora_checkpoints/` (8 dirs) | `models/checkpoints/<model>/` | LoRA adapter checkpoints | `training/train_lora.py` (writes), `debug_tools/workflow_gui.py` (reads/lists), `training/test_lora_checkpoint.py`/`test_lora_on_page.py` (CLI args, not hardcoded) | No - update `train_lora.py`'s unanchored string (Phase 1 finding #1) AND `workflow_gui.py`'s 4 call sites together, ideally via one shared `checkpoint_dir_for(model)` helper instead of two separate fixes |
| `data/outputs/lora_dataset` | `datasets/lora_dataset` | Training dataset | `training/export_lora_dataset.py` (writes), `training/train_lora.py` (reads) | Yes, once both constants updated - no other consumers found |
| `data/outputs/benchmark2*`, `prompt_sweep_runs`, `reference_classifier_qualification` | `benchmarks/<name>/` | Frozen research results | Each benchmark's own script; `benchmark2_4_selective_escalation` cross-references `benchmark2_3_multi_tower`'s output path, so those two must move together or one script's constant breaks | Yes, low risk - not git-tracked (`data/outputs/` is entirely `.gitignore`d), no production-pipeline code reads these |
| `data/outputs/reference_pipeline_v1`, `reference_pipeline_prerefactor` | `benchmarks/references/<name>/` or `data/references/<name>/` | Frozen snapshots | Documentation only (`docs/REFERENCE_PIPELINE_V1.md`); no code reads them | Yes - update the doc's paths, nothing else |
| `data/outputs/scoring_reports` | `data/outputs/scoring_reports` (leave, or `data/pipeline_outputs/scoring_reports`) | Real per-page scoring output | `debug_tools/workflow_gui.py` (a user-editable `StringVar` default, not a hard dependency) | Yes - trivially, it's a GUI default the user can already override |
| `data/outputs/.lorabackup` | anywhere (e.g. `archive/lorabackup/`) | Manual backup, zero live references | None (Phase 1 finding #4) | Yes - completely safe, nothing will notice |
| `data/outputs/*_log.jsonl` (8 files), `benchmark2_pilot_checkpoints.json`, `_workflow_gui_state.json` | `logs/` | Append-only logs / cache | Each file's one producing script (`benchmark/*.py`, `debug_tools/workflow_gui.py`) | Yes, mechanically - one constant per file, all independent of each other |
| `experiment3_1_outputs`, `experiment3_2_outputs`, `experiment3_3_outputs` (top-level, NOT under `data/`) | `benchmarks/experiment3_*/` | Benchmark Artifact (Phase 1 finding #2) | Their own scripts only | Yes - these are stray already; moving them under `data/outputs/` or `benchmarks/` is a net improvement regardless |
| `benchmark_results` (top-level) | `benchmarks/` | Benchmark Artifact (Phase 1 finding #2) | `benchmark/benchmark_db.py` | Yes, same reasoning |
| `data/reference_prerefactor/`, `data/buckets/reference_prerefactor/` | *(Jon's own just-made snapshot, 2026-08-02)* | Reference Archive (production pre-refactor freeze) | Nothing yet - created outside any script, ahead of the current live run | N/A - already moved, by Jon, deliberately; not part of this audit's scope, but worth reconciling the near-identical name with `data/outputs/reference_pipeline_prerefactor` (see Phase 2) so a future reader isn't confused by two "prerefactor" archives from different sessions |

## Phase 4 — Recommended layout

Jon's sketched layout is close, but two things about THIS project argue
for a variant: (1) `data/pipeline.db` already exists as the real
system-of-record and must stay adjacent to whatever `data/working/`
becomes, not be treated as a generic "output"; (2) the benchmark
corpus is large and stable enough (30+ scripts, several explicitly
FROZEN per `docs/`) that it deserves its own top-level tree rather than
living inside `data/`.

```
data/
    working/            # unchanged in spirit - but see Lead Finding:
                         # moving this requires a path-rewrite migration,
                         # not a rename
    pipeline.db         # stays beside working/ - both are the live
                         # orchestration state, not "output"
    manifest.csv, manifest_provenance.json, buckets/
    baseline_embeddings.json
    references/         # data/reference_prerefactor, data/buckets/
                         # reference_prerefactor (Jon's 2026-08-02 snapshot),
                         # reconciled with docs/outputs' reference_pipeline_v1/
                         # reference_pipeline_prerefactor into ONE place
    logs/                # every *_log.jsonl + benchmark2_pilot_checkpoints.json
                         # + _workflow_gui_state.json

models/
    checkpoints/
        qwen3vl2b/, smolvlm/, granite_vision_2b/, ...   # one per model,
                         # replacing the *_lora_checkpoints naming
                         # convention with a directory-per-model instead
                         # of a suffix-per-model

datasets/
    lora_dataset/

benchmarks/
    benchmark2/, benchmark2_2/, benchmark2_3_multi_tower/, ...
    prompt_sweep_runs/
    scoring_reports/     # arguably belongs here too, or stays a
                         # pipeline output - it's real per-page
                         # evaluation, not throwaway experiment data;
                         # judgment call, not a clear-cut case

docs/                   # unchanged
```

**Deliberately NOT recommending** collapsing everything into the
generic `data/{working,outputs,references,logs,cache}` shape Jon
sketched, because `data/outputs/` today conflates three genuinely
different lifecycles (benchmark experiments that are frozen and never
touched again, model checkpoints that grow during active training, and
one real pipeline output/scoring_reports) under one folder - splitting
`models/`, `datasets/`, and `benchmarks/` out to their own top-level
trees makes each one's actual growth/retention policy visible from the
directory structure itself, rather than needing to read this audit to
know which of 20 `data/outputs/` subdirectories is safe to delete.

## Phase 5 — Refactor strategy

1. **Path-rewrite tooling for `data/working/`/`baseline_embeddings.json`
   FIRST**, before touching any folder - a small one-time script
   (mirroring `scripts/migrate_manifest_to_db.py`'s own careful,
   read-first-then-verify discipline) that rewrites every stored path
   in `pipeline.db` and every `"image"` key in `baseline_embeddings.json`
   to a new root, verified with a before/after diff exactly like
   `finalize_manifest()`'s migration was. This is the genuinely novel
   risk THIS project has that a generic "grep for references" audit
   wouldn't surface, and it must be solved before `data/working/` itself
   ever moves.
2. **Introduce `core/paths.py`** with one constant per real location
   (`WORKING_DIR`, `BUCKET_DIR`, `MANIFEST_PATH`, `PIPELINE_DB_PATH`,
   `CHECKPOINTS_DIR`, etc.) - given every file already independently
   redeclares the identical `PROJECT_ROOT / "..."` expression, this is a
   mechanical, low-risk, one-constant-at-a-time replacement, not a
   redesign.
3. **Fix the two real inconsistencies while centralizing, not after**:
   `training/train_lora.py`'s unanchored relative path (Phase 1 #1), and
   the duplicated `_lora_checkpoints` string-building logic (Phase 1
   #3) - both naturally resolved by having `core/paths.py` own a single
   `checkpoint_dir_for(model_name)` function instead of two independent
   constructions.
4. **Update all code to import from `core/paths.py`**, verified by
   re-running the exact verification pattern already proven this
   session: capture a "before" output from a real invocation, make the
   change, capture "after," diff as a SET not just a count.
5. **Move the low-risk folders first** (benchmarks/, datasets/,
   `.lorabackup`, the four stray top-level `experiment3_*_outputs`/
   `benchmark_results` dirs) - none are git-tracked, none feed the live
   pipeline, and Phase 3 already confirms nothing references them
   except their own already-updated constants.
6. **`data/working/` + `pipeline.db` + `baseline_embeddings.json` move
   together, last, as one atomic step** - path-rewrite (step 1) run
   immediately before the actual folder move, then `finalize_manifest()`
   and a handful of `get_image_by_path()` spot-checks re-verified
   against the new location before calling it done.
7. **Remove legacy path support** only after every script has been
   confirmed working against the new layout for at least one real run
   (matching this session's own "verify against real data before
   trusting it" discipline throughout the DB migration work).
