# Pipeline orchestration database (SQLite) — design + Phase 1/2 (2026-08-02)

**Status (2026-08-02): every live stage now dual-writes into the DB -
Stage 0/1/2/3/5, `debug_tools/review_uncertain.py`'s human corrections,
and `ui/dewarp_preprocessor_ui.py`'s live Save/Bypass - and
`finalize_manifest()` (Stage 2b) READS from the DB instead of scanning
bucket CSVs**, the first (and so far only) stage migrated past
dual-write into an actual read-path change. Every existing CSV/JSON
output any tool still depends on is UNCHANGED — `finalize_manifest()`'s
own output (`manifest_final.csv`) was verified row-for-row identical (as
a set) against its pre-migration implementation on the real 1677-image
corpus before this was trusted.

## Why

`data/manifest.csv` started as a one-column file list and grew into
something several stages lean on for state (`file_path`, and at the
now-unused `manifest_final.csv` design, `category`/`status`/
`source_original_path`). Provenance lives in a sidecar JSON, classification
lives in 8 separate per-bucket CSVs, dewarp results live in
`<bucket>_dewarped.csv`, and "what stage is this image at" has always
been *inferred* — by which CSV a path currently appears in
(`finalize_manifest()`'s status literals: `ready` / `ready_dewarped` /
`pending_manual_dewarp`) — rather than stored anywhere directly. This
file-scattered, path-string-joined state is exactly what produced a real
bug earlier this session: a Stage 1 (baseline embedding) capture, run
separately from Stage 0/3, had no record of whether the file at a given
path had already been through Stage 3 preprocessing — it silently
measured already-processed pixels and tagged them `"pre_preprocessing"`
(see `docs/REFERENCE_PIPELINE_V1.md`). This database replaces
join-by-path-string with a real system of record for pipeline *state*,
while every actual measurement/evidence artifact (embeddings, OCR,
physical/semantic measurements) stays exactly where it already lives —
on disk, as JSON, immutable once written — per this project's
established evidence-first philosophy.

**Design philosophy (Jon's framing, adopted directly)**: a database row
answers *"what is the current state of this image?"* A JSON sidecar
answers *"what did this stage observe?"* Two tables, not the more
normalized six-table shape an earlier design pass sketched.

## Schema

### `images` — current state, one row per logical image

| column | meaning |
|---|---|
| `id` | primary key |
| `source_path` | original file this came from (a PDF or plain image) |
| `source_type` | `"image"` \| `"pdf"` \| a future `ExpandedSource` tag |
| `page_number` | 1-indexed, `NULL` for a plain image |
| `working_path` | current on-disk location; **updated** when a stage produces a genuinely new file (e.g. dewarp output) |
| `identity_hash` | sha256 captured **once**, at Stage 0, on the raw acquired copy, before any preprocessing. Never recomputed. |
| `current_hash` | sha256 of `working_path`'s bytes **right now**. Recomputed whenever a stage modifies the file in place or produces a new `working_path`. |
| `current_stage` | 0-6, per `docs/PIPELINE_STAGE_TERMINOLOGY.md` |
| `status` | free-form per-stage status string (`"acquired"`, `"preprocessed"`, `"classified"`, `"ready_dewarped"`, `"uncertain_review"`, ...) |
| `bucket` | `DocumentCategory` value, once Stage 5 has run |
| `classifier_confidence` | float |
| `classifier_model` | model name string |
| `processing_profile` | e.g. `"autocontrast"` — Stage 3's applied profile |
| `created_at` / `updated_at` | UTC ISO timestamps |

Indexes: `working_path`, `source_path`, `(current_stage, status)`, `bucket`.

**Why two hash columns, not one** (the single deliberate deviation from
Jon's own sketch, which had one `sha256` field): a single hash recomputed
at every stage can't distinguish "the pipeline changed this on purpose"
(expected — deskew/autocontrast/dewarp) from "this drifted unexpectedly
between two runs" (a bug). `identity_hash` is the stable fingerprint
proving lineage; `current_hash` + `updated_at` answer "has anything
touched this file since I last looked." This directly closes the
ambiguity that produced the mislabeled `reference_pipeline_v1` data.

**Provenance is not a separate table.** The three cheap, deterministic,
always-present fields (`source_path`, `source_type`, `page_number`) are
plain columns for fast filtering. The open-ended, expander-specific part
(`ExpandedSource.metadata`, e.g. PDF DPI) stays in
`manifest_provenance.json` exactly as it already does today — referenced
via a `stage_outputs` row (`stage="stage0_acquire"`), never duplicated
into new columns.

### `stage_outputs` — append-only evidence pointer log

| column | meaning |
|---|---|
| `id` | primary key |
| `image_id` | FK -> `images.id` |
| `stage` | e.g. `"stage0_acquire"`, `"stage1_baseline_embeddings"`, `"stage1_image_analysis"`, `"stage3_preprocess"`, `"stage5_classify"`, `"stage5_human_review"`, `"stage2a_dewarp"`, `"auto_sidecar_generation"`, `"stage6_extract"` |
| `sidecar_path` | path to the JSON/CSV holding this stage's real output. `NULL` when the output lives in a file **shared** across many images (e.g. `data/baseline_embeddings.json`) rather than a dedicated per-image sidecar. |
| `lookup_key` | only set for the shared-file case above — the key (`identity_hash`, or a bucket CSV's `file_path`) needed to find this image's record inside that shared file. |
| `sha256` | hash of the sidecar's content at write time — lets a future check notice if a sidecar was altered after the DB recorded it. |
| `status` | `"done"` \| `"failed"` \| `"skipped"` |
| `note` | short free text — error message, reviewer note, etc. |
| `created_at` | never updated — one row per stage **run**, not per image |

Index: `(image_id, stage)`.

This single table is simultaneously the historical audit log (many rows
per image over time — mirrors this project's existing append-only
discipline for `ground_truth_log.jsonl`/`reviewed_uncertain.csv`) and the
sidecar location index. No separate `stage_history` or
`sidecar_locations` tables. `routing`/`pipeline_state` as distinct tables
are also dropped — their *current* values are columns on `images`; their
*history* is `stage_outputs` rows.

## How each real stage maps onto it

**Stage 0/1/2/3 are LIVE as of 2026-08-02** (dual-write - every existing
CSV/JSON output is unchanged, the DB write is additive). Stage 2a and
onward are still future work, described below as the target mapping.

- **Stage 0** (`core/manifest_pipeline.py: stage0_acquire_and_copy_sources`)
  — one `get_or_create_image()` per `ExpandedSource`; `identity_hash`
  computed here, the ONE place it's ever computed. `stage_outputs`
  row pointing at the provenance sidecar. **LIVE.**
- **Stage 1** (`stage1_capture_baseline_embeddings` +
  `core/image_analysis.py`) — no pixel change, no hash update.
  `stage1_capture_baseline_embeddings()` records `stage1_baseline_
  embeddings` (shared file, `lookup_key=identity_hash`) - **LIVE**.
  `core/image_analysis.py` itself is a separate, standalone script, not
  called from `core/manifest_pipeline.py`'s chain - its
  `stage1_image_analysis` stage_outputs row (per-image sidecar via
  `analysis_sidecar_path()`) is NOT yet wired in.
- **Stage 2** (`core/decision_engine.py: stage2_decide_profiles`) —
  **LIVE, architecture-only** (2026-08-02). Reads Stage 1's baseline
  embeddings; records a preprocessing-profile decision
  (`images.processing_profile`) that is currently a trivial pass-through
  (`"autocontrast"` for every image - no validated rule exists yet, per
  `docs/PREPROCESSING_STAGE_NOTES.md`'s explicit gate) plus a
  tower-consensus classification signal (`images.tower_consensus_
  category` + a `<name>_tower_consensus.json` sidecar) - this part IS
  real, reusing the Multi-Tower Routing Audit's own validated
  nearest-cluster logic (now in `core/vision_embeddings.py`'s
  `predict_nearest_bucket()`/`classify_consensus()`) with zero new model
  inference. See "Stage 2 in depth" below.
- **Stage 3** (`stage3_preprocess_manifest`) — recomputes `current_hash`.
  **LIVE.** Today's per-file output (deskew angle, profile) still only
  prints to stdout - a small `<name>_preprocess.json` sidecar would be a
  natural future addition, not yet built.
- **Stage 2a manual dewarp** (`ui/dewarp_preprocessor_ui.py`) — the one
  place `working_path` changes to a genuinely different file.
  `current_hash` recomputed against it, `status="ready_dewarped"`.
- **Stage 5** (`core/classifier.py`) — updates `bucket`/
  `classifier_confidence`/`classifier_model`. Exact placement of the
  richer per-row fields (`reason`, `text_density`, etc.) is an open call
  for the actual Phase 3 classifier migration, not resolved here.
- **Stage 5 human review** (`debug_tools/review_uncertain.py`) — updates
  `bucket`; `stage_outputs` row `stage5_human_review` mirrors
  `reviewed_uncertain.csv`'s reviewer/original-bucket/assigned-bucket fields.
- **Auto sidecar generation** (`scripts/run_batch_auto_sidecar.py` /
  `core/auto_sidecar.py`) — NOT canonical Stage 4 (that's "Validation
  Capture," NOT STARTED, per `docs/PIPELINE_STAGE_TERMINOLOGY.md`) — this
  tool sits between Stage 5 (routing/subtype classification) and Stage 6
  (extraction), the same bridge position as `core/semantic_stages.py`.
  `PIPELINE_WORKFLOW.md`'s own "Stage 4" label for it predates the
  canonical Stage 0-6 renaming and is a naming collision, not the same
  concept - use `stage="auto_sidecar_generation"`, no number, matching
  how this doc's stage_outputs examples above already do the same for
  Stage 2a. `stage_outputs.status` maps from that tool's own existing
  per-row outcomes (`ok` / `skipped_unknown_doc_type` / `skipped_not_
  dewarped` / `error`, already in `batch_run_summary.csv` today).
- **Stage 6** (`core/row_extraction.py`) — `stage_outputs` row pointing
  at the per-page results CSV/JSON `save_results_csv()`/`save_results_json()`
  already produce.

## Stage 2 in depth (`core/decision_engine.py`, added 2026-08-02)

Two DIFFERENT things, deliberately kept apart - see that module's own
docstring for the full reasoning:

1. **Preprocessing-profile decision** - trivial pass-through today.
   `docs/PREPROCESSING_STAGE_NOTES.md` is explicit: *"the semantic
   sensor layer is NOT used to make preprocessing decisions yet...
   Stage B profile selection (once evidence demonstrates predictive
   value - not inferred from embeddings without that evidence)"*, and
   there is currently exactly ONE calibrated Stage 1 signal
   (`table_confidence`, floor ~0.375) - no validated finding that any
   signal should change which profile an image gets. This module builds
   the real, DB-wired ARCHITECTURE for that future decision (Stage 1
   evidence in, a profile choice out, recorded per image) without
   inventing an unvalidated rule now - matching this project's
   Hypothesis -> Experiment -> Benchmark -> Evidence -> Production
   promotion rule.
2. **Tower-consensus classification** - real, not gated the same way.
   Jon's direction: *"the sensor tower should be used here to determine
   what an image needs to have done to it, while its gathering that
   information we should also be using its classification ability
   too."* The 8 qualified vision towers' nearest-cluster bucket
   prediction + consensus categorization is ALREADY validated
   production research (the Multi-Tower Routing Audit) - reusing it here
   isn't inventing a new rule, just running it earlier (right after
   Stage 1, before Stage 5/Gemma classification happens) using
   embeddings Stage 1 already captured, with **zero new model
   inference** - no encoder is loaded or run again.

**`build_reference_embeddings()`** builds the per-bucket reference set
from the first `N_REFERENCE_PER_BUCKET=15` already-classified images per
bucket (excluding `uncertain_review` - "never a tower candidate bucket",
same as the audit), pulling vectors straight from `data/
baseline_embeddings.json` - no new inference. 15/bucket matches
`benchmark2_2_cross_validator.BUCKET_PLAN`'s own validated scale, not an
invented size.

**Verified end-to-end** (2026-08-02) against a real production image
(from `dense_tabular_rows`) in an isolated scratch DB/manifest, with
reference embeddings sourced read-only from the real
`data/baseline_embeddings.json` + `data/buckets/*.csv`: reference set
sizes came out exactly as expected (15 for 7 of 8 buckets, 4 for the
small `handwritten_ledger`), per-bucket cosine-similarity scores were
real and differentiated (e.g. dinov2: `handwritten_ledger` 0.776 >
`printed_document` 0.712 > `dense_tabular_rows` 0.680 > ...), and the
full chain (`stage0_acquire_and_copy_sources` -> `stage1_capture_
baseline_embeddings` -> `stage2_decide_profiles` -> `stage3_preprocess_
manifest`) ran against a fresh synthetic image with `identity_hash`
staying frozen while `current_hash` changed after Stage 3 - confirming
the two-hash design works as intended in the live chain, not just in
isolated unit tests. Production `data/pipeline.db` (1677 rows) was
untouched by any of this testing (confirmed via row count + `git
status`) - every test ran against an isolated scratch `db_path`.

**Real finding, not a bug**: the test image's tower consensus predicted
`handwritten_ledger` (7/8 towers), not its own true bucket
(`dense_tabular_rows`) - confirmed correct by direct score inspection,
not assumed. `handwritten_ledger` only has 4 total images in the whole
corpus, so its reference set is thin; this is a genuine small-N/corpus-
composition artifact of real embedding-space geometry, consistent with
the Multi-Tower Routing Audit's own finding that Gemma and tower
consensus don't always agree. It does not indicate the code is wrong -
it's exactly why this signal is recorded as evidence, not acted on.

## What was actually built and verified this pass

- **`core/pipeline_db.py`** — `PipelineDatabase` class: schema creation,
  `get_or_create_image()`, `get_image_by_path()`, `get_image()`,
  `update_image_state()` (whitelisted columns only),
  `record_stage_output()`, `list_images()`, `get_stage_outputs()`.
  WAL mode, one connection per call (no held-open connection or
  transaction across calls — this project already runs multiple
  independent processes against the same `data/` directory). stdlib-only
  import (`sqlite3`/`hashlib`/`pathlib`/`datetime`) — no torch/timm,
  matching this project's lazy-import discipline for heavy deps.
  Default path `data/pipeline.db`.

- **`scripts/migrate_manifest_to_db.py`** — one-time, read-only,
  additive import: `data/manifest.csv` -> `images` rows, then layers on
  `data/manifest_provenance.json` (if present), per-image
  `*_analysis.json` sidecars, `data/baseline_embeddings.json`,
  every `data/buckets/<category>.csv`, and any `*_dewarped.csv`.
  `--verify` diffs DB counts against the source CSVs.

  **Real bug found and fixed while running this against the actual
  corpus**: `data/baseline_embeddings.json` stores each image's path
  **relative to the project root** (`"data\\working\\foo.jpg"`), while
  `manifest.csv` and every bucket CSV store **absolute** paths
  (`"J:\\Genealogy\\genealogy_pipeline\\data\\working\\foo.jpg"`) —
  confirmed by direct inspection, not assumed. The initial run matched
  0/1677 baseline-embedding records for exactly this reason. Fixed with
  a `_to_abs()` normalizer (resolves relative paths against
  `PROJECT_ROOT`, not the caller's cwd) applied on both sides of the
  comparison. Re-run matched 1677/1677.

  **Honest caveat about `identity_hash` for this specific migrated
  batch**: this project's real `data/working/` corpus was already
  through Stage 3 (deskew+autocontrast) *before* this session's Stage 1
  baseline-embedding capture ever ran (`docs/REFERENCE_PIPELINE_V1.md`).
  There is no surviving pre-preprocessing byte state to hash.
  `identity_hash` for these 1677 images is therefore a best-effort
  retroactive fingerprint computed from *current* (already-processed)
  bytes — printed as a warning by the migration script, not silently
  assumed to be a genuine Stage-0-before-anything hash. Going forward,
  once Phase 3 wires `stage0_acquire_and_copy_sources()` directly into
  this database, `identity_hash` will be captured correctly, before
  Stage 3 ever runs — this caveat applies only to this one retroactively
  -imported batch, not to the design going forward.

  Since this corpus's classification is already complete, the migration
  also records `current_stage=3, status="preprocessed"` as a baseline
  for every image (a documented fact, not a guess) before layering Stage
  5 classification on top from the bucket CSVs.

### Verified against the real corpus (2026-08-02)

```
manifest.csv rows: 1677  |  images in DB: 1677
dense_tabular_rows: bucket CSV=190  DB=190  [OK]
handwritten_ledger: bucket CSV=4  DB=4  [OK]
printed_document: bucket CSV=987  DB=987  [OK]
portrait_photo: bucket CSV=83  DB=83  [OK]
map_land_record: bucket CSV=186  DB=186  [OK]
mixed_text_image: bucket CSV=28  DB=28  [OK]
genealogy_chart: bucket CSV=49  DB=49  [OK]
uncertain_review: bucket CSV=2  DB=2  [OK]
website_screenshot: bucket CSV=148  DB=148  [OK]
verify PASSED
```

1677/1677 images also got a `stage1_image_analysis` sidecar pointer and a
`stage1_baseline_embeddings` pointer. `git status` confirmed zero files
under `data/` were modified by the migration — only the new
`data/pipeline.db` (2.4MB) appeared, untracked.

## `finalize_manifest()` migration (Stage 2b, 2026-08-02)

The first stage migrated past dual-write into an actual READ-path
change - it was flagged as the strongest candidate (see the "What's NOT
done" section below, pre-migration) since its old implementation was
exactly the fragile cross-CSV join pattern the DB exists to replace, and
its output (`manifest_final.csv`) has zero downstream consumers
(confirmed by grep) - nothing could break from changing what feeds it.

**New shape**: `finalize_manifest()` now (1) syncs the DB from Stage 5's
bucket CSVs and Stage 2a's dewarped-bucket CSVs via two new promoted
helpers in `core/pipeline_db.py` - `sync_bucket_classifications()` and
`sync_dewarp_results()` - then (2) builds every output row from
`db.list_images()`/`db.get_stage_outputs()`, not from re-reading the
CSVs a second time. Both sync helpers are the SAME logic
`scripts/migrate_manifest_to_db.py` used for its one-time import
(promoted, not duplicated - that script now calls these too), extended
to be safely re-run on every `finalize_manifest()` call:
- **Auto-registers** any bucket-CSV `file_path` not yet known to the DB
  (`get_or_create_image()`) instead of silently dropping it - a bucket
  CSV is allowed to reference files Stage 0 never registered.
- **Idempotent**: `sync_bucket_classifications()` compares against the
  DB's current `bucket`/`confidence`/`model` before writing; a genuinely
  moving target (`sync_dewarp_results()`, since `working_path` itself
  changes to the dewarped output) checks `find_stage_output()` by
  `lookup_key` instead, since the pre-dewarp path can no longer be
  looked up directly on a second call. Both verified: re-running
  `finalize_manifest()` twice in a row produces byte-identical output
  and zero `stage_outputs` growth.
- `source_original_path` (for `ready_dewarped` rows) is recovered from
  the `stage2a_dewarp` stage_outputs row's `lookup_key` - no new column
  needed.

**Verified two ways before trusting this**:
1. Against the real 1677-image corpus: captured the OLD implementation's
   output first, then compared it against the new implementation's
   output as a SET of `(file_path, category, status,
   source_original_path)` tuples - **identical, 0 rows different either
   direction**. `stage_outputs` count unchanged (5031 = 1677×3) since
   the corpus was already fully synced - confirming the idempotency
   check correctly made zero redundant writes.
2. Against a synthetic scratch corpus (since production has zero
   `*_dewarped.csv` files today - this branch was otherwise completely
   untested): one dense_tabular_rows image with a dewarp record, one
   without. Correctly produced `ready_dewarped` (file_path = dewarped
   output, source_original_path = pre-dewarp path) and
   `pending_manual_dewarp` respectively; re-running twice produced
   identical output and no `stage_outputs` growth.

## Stage 5 (`core/classifier.py`) wiring, 2026-08-02

Deliberately the SMALLEST possible change, not a parallel reimplementation:
the existing classify loop (model inference, bucket-CSV writing) is
completely untouched. `run()` just calls `sync_bucket_classifications()`
ONCE, after all bucket CSVs are written, reusing the exact same promoted
function `finalize_manifest()` and the migration script already use -
not a second, possibly-drifting implementation of "how does a bucket row
become DB state." A new `db_path` param (default `DEFAULT_DB_PATH`)
threads through to `main()`'s CLI as `--db-path`, matching every other
stage script's convention.

**A real regression caught during this pass, not shipped**: promoting
`import_bucket_classifications()` into `sync_bucket_classifications()`
(for the `finalize_manifest()` migration above) had silently dropped the
original migration script's "failed" `stage_outputs` recording for
error rows (a hard pipeline failure - malformed model output, etc. -
core/classifier.py's own distinction) - the promoted version just
`continue`d past them with no record at all. Never caught by the
finalize_manifest() diff test since the real corpus's 2 `uncertain_
review.csv` rows both happen to have an empty `error` field. Fixed by
re-adding the failed-row recording, now WITH an idempotency check
(`find_stage_output()` comparing status+note) so a persistently-erroring
row - which round-trips untouched in `uncertain_review.csv` forever,
per `debug_tools/review_uncertain.py`'s own design - doesn't grow
`stage_outputs` on every re-sync. Verified with a dedicated synthetic
test (a fake error row, synced twice): records exactly once, `bucket`
stays `NULL`, never treated as a real classification.

**Verified** (stubbed loader + monkeypatched `BUCKET_DIR`, no real model
load - see this session's transcript): a fake classification correctly
produced `current_stage=5`, `status="classified"`, `bucket`/
`classifier_confidence`/`classifier_model` set, with zero writes to
production `data/buckets/` or `data/pipeline.db` during the test.
Re-ran `finalize_manifest()` and `scripts/migrate_manifest_to_db.py
--verify` against the real corpus after the fix - both still pass,
`stage_outputs` count unchanged (5031 = 1677×3).

## `debug_tools/review_uncertain.py` wiring (2026-08-02)

In `--source uncertain` mode (the live queue - NOT `--source
misclassifications`, which labels a separate ground-truth sample in
place and never touches a real bucket CSV), `assign()` and
`mark_ignore()` each now call a new `_sync_db_correction()` method
immediately at the point of correction - a dedicated `"stage5_human_
review"` stage_outputs event, not left for the next passive
`sync_bucket_classifications()` call to notice the bucket CSV changed.
`assign(bucket)` sets `images.bucket=bucket`, `status="classified"`.
`mark_ignore()` sets `images.bucket=NULL` (mirrors what actually happens
on disk - the file leaves EVERY bucket CSV, `_write_bucket_row()` is
never called for this case) and `status="ignored"` - a genuinely new
terminal status string (the column is free-form TEXT, not a fixed
enum). Also fixed a real, unrelated naming collision found while
touching this file: its old docstring called itself "Stage 3 (human
gate)", which predates the canonical Stage 0-6 numbering and is NOT
canonical Stage 3 (Image Processing) - relabeled as the Stage 5
human-review gate.

**Verified** with a scratch test exercising BOTH corrections against an
isolated DB (module constants `BUCKET_DIR`/`REVIEWED_LOG`/`OUTPUT_DIR`
monkeypatched to scratch paths, `messagebox.askyesno` patched to avoid a
blocking confirmation dialog, no real Tkinter interaction needed): one
image assigned to `dense_tabular_rows` correctly got `bucket=
"dense_tabular_rows"`/`status="classified"` plus a `stage5_human_review`
stage_outputs row with the reviewer/original-bucket/assigned-bucket
note; one image marked ignore correctly got `bucket=NULL`/
`status="ignored"` plus its own stage_outputs row. Zero writes to
production `data/buckets/`, `data/outputs/`, or `data/pipeline.db`
during the test.

## `ui/dewarp_preprocessor_ui.py` wiring (2026-08-02)

The last remaining CSV-only write path. `apply_and_save()`/`bypass()`
both funnel through `_record_and_advance_if_worklist(status)` (already
the ONE place that writes to the preprocessed-bucket CSV) - it now also
calls `core/pipeline_db.py`'s new `sync_one_dewarp_result()` immediately
after the CSV write, in bucket-worklist mode only (single-file mode has
no bucket-CSV relationship to record, same gate as the CSV write
itself). `sync_one_dewarp_result()` is a refactor-out of
`sync_dewarp_results()`'s own per-row body - the batch function now just
calls it in a loop - so the live UI call and the batch
`finalize_manifest()` sync are the exact same logic, not two
implementations that could drift apart.

Confirms `images.status="ready_dewarped"` for BOTH a real dewarp AND a
bypass, matching this project's own pre-existing `finalize_manifest()`
semantics (it never distinguished the two either - a bypassed entry has
`output_file_path == source_file_path`, so this is a same-value
"update," not a special case).

**Verified** with a scratch worklist simulation (`DewarpApp` instantiated
directly, `bucket_csv_path`/`worklist`/`image_path`/`output_path` set
manually to avoid needing real corner-dragging or a file dialog - this
tests the RECORDING logic the change actually touches, not `dewarp_
quad()`'s own already-proven correctness): a Save correctly produced
`working_path` = the dewarped output, a recomputed `current_hash`, and
one `stage2a_dewarp` stage_outputs row with `lookup_key` = the pre-dewarp
path; a Bypass correctly produced `status="ready_dewarped"` with
`working_path` unchanged. Then ran `finalize_manifest()` against the
same DB/bucket dir afterward and confirmed its batch `sync_dewarp_
results()` correctly recognized both as already-synced (`stage2a_dewarp`
event count stayed at exactly 2, not 4) - the live write and the batch
sync agree, not just each one individually. Zero writes to production
`data/`, confirmed via `git status` and DB row counts (1677 images, 5031
stage_outputs, unchanged).

## Evidence-completeness pass (2026-08-03)

Two changes, both purely additive - no decision logic changed, no
threshold introduced, benchmark behavior unchanged (verified).

**1. Full bucket-score landscape recorded, not just the winner.**
Jon's framing: *"treat the vision towers as sensors - a sensor should
report everything it observed, not just its final winner... storage is
cheap, re-running eight vision towers across the corpus is expensive...
I do not want to begin adjusting thresholds yet - without the full
bucket scores, threshold tuning is guesswork."* `core/vision_
embeddings.py` gained `score_all_buckets()` (the per-bucket scoring
loop `predict_nearest_bucket()` already did internally, extracted);
`predict_nearest_bucket()` itself is now a thin wrapper over it with
its exact original `(winner, score)` return UNCHANGED - verified via
direct equality checks before touching anything, since `benchmark/
benchmark2_3_multi_tower_routing_audit.py` (frozen research code)
depends on that exact signature.

Each encoder's entry in the `<name>_tower_consensus.json` sidecar now
carries `winner`/`winner_score`/`runner_up`/`runner_up_score`/`margin`
(computed from the already-rounded scores, so margin always equals
what's displayed) plus the complete `scores` dict for every bucket that
had reference data, sorted highest-first. The `scores` dict is what
matters; the rest are convenience summaries of it.

**2. The physical sensor (`core/image_analysis.py`) is now wired in and
has been run for the first time against the current corpus.** This
sensor existed and was fully built, but had never been connected to
`core/pipeline_db.py` at all, and the "prepare for fresh run"
reorganization moved away the only sidecars that existed for the old
corpus - so the physical half of Stage 1 had ZERO evidence for any of
the current 1975 images until this pass. `analyze_manifest()` gained a
`db_path` parameter: records a `stage1_image_analysis` stage_outputs
event per image (pointing at the `*_analysis.json` sidecar it already
wrote), idempotent (skips images with an existing event via `get_
stage_outputs()`, not `find_stage_output()` - a real bug caught during
testing, since this stage never changes `working_path` the way dewarp
does, so there's no "path changed since" case to guard against; the
simpler by-image_id check is both correct and sufficient). Ran for
real: 1975/1975 measured, 0 failures.

**Verified before running for real**: scratch tests for both changes
(enhanced tower-consensus fields against real reference data; the
image_analysis DB wiring's success/failure/idempotency paths, including
the lookup_key bug caught and fixed via a clean re-test). Zero
production writes during any test.

**Storage cost, measured not estimated**: 1974 `*_tower_consensus.json`
files, 8.20MB total (4.4KB/image avg); 1974 `*_analysis.json` files,
4.00MB total (2.1KB/image avg) - 12.2MB of new evidence for the whole
1975-image corpus, against a `baseline_embeddings.json` that's already
106MB. `data/pipeline.db` grew to 5.4MB (was 3.3MB) - 17,775 stage_
outputs rows total, still just pointers/small state, no evidence stored
in the DB itself.

**New analyses this now supports without ever re-running a vision tower
or the physical sensor again**: for any image, read its `_tower_
consensus.json` and see all 8 buckets' scores per encoder - not just
which one won. Aggregate across the corpus: which bucket PAIRS have
the smallest average margin (candidates for a real confusability
study); which images have a near-zero margin on their winning encoder
(near-ties, worth a second look regardless of what the consensus
category says); which of the 8 encoders has the smallest average
margin overall (a "confidently uncertain" encoder, maybe worth
down-weighting in the vote); cross-referencing `_analysis.json`'s
`table_confidence` (the one already-calibrated physical signal) against
`dense_tabular_rows` tower-consensus scores, to see whether the
physical and semantic sensors agree on which images are really tables.
None of this requires new capture - it's all sitting in sidecars
already written, waiting to be queried.

## Architectural consolidation pass (2026-08-03)

Following a cross-stage duplication audit (every stage checked for
observations measured more than once, decisions made in more than one
place, and state reconstructed outside the DB instead of read from it),
two low-risk items were consolidated - deliberately structural only, no
new routing rule, threshold, or weighting introduced:

**1. `scripts/run_batch_auto_sidecar.py`'s dewarp-completion check** used
to glob `data/outputs/dewarped/<stem>_dewarped.*` by filename convention
- state `pipeline_db` already tracks unambiguously via `images.
working_path`/`status`. `find_dewarped()` now takes `(db, file_path)`
and resolves the dewarped output via the `stage2a_dewarp` stage_outputs
row's `lookup_key` (the pre-dewarp path, recorded at write time) ->
`image_id` -> current `working_path` - the same "identity survives a
path change" pattern `finalize_manifest()` already uses to recover
`source_original_path`. Verified against a throwaway DB in scratch:
not-yet-dewarped -> `None`, dewarped -> correct new path, unknown path
-> `None`, matching the old glob's contract exactly.

**2. "Does this `dense_tabular_rows` image need manual dewarp"** was
decided by a hardcoded stub (`_dense_tabular_needs_manual_dewarp()`,
always `True`) called inline inside `finalize_manifest()`, with no
durable record - a real duplicate-decision risk, since the dewarp
worklist tooling independently assumes the same answer for the whole
bucket rather than reading a shared source of truth. `images.
needs_manual_dewarp` (INTEGER, `NULL` where not applicable) now records
this decision the first time it's computed for a given image, still via
the exact same stub logic - this pass changes WHERE the answer lives,
not what it is. Idempotent (only writes when the stored value would
change). Verified behavior-preserving: `manifest_final.csv` byte-
identical across two consecutive runs against a scratch copy, then run
for real against the production DB - 193 `dense_tabular_rows` images
recorded (`NULL` -> `1`), 1782 correctly left `NULL`, output unchanged
(1975 files, 193 pending manual dewarp - same numbers as every prior
run of this function).

**Not yet done** (flagged, not silently skipped): the dewarp worklist
tooling itself (`ui/dewarp_preprocessor_ui.py` / `core/bucket_
worklist.py`) still assumes 100% of the bucket needs dewarp rather than
reading `images.needs_manual_dewarp` - wiring that in touches a live,
human-facing GUI tool and was judged out of scope for this low-risk
pass. Classification's write-authority (bucket CSVs are still the real
source, `pipeline_db` still a synced mirror via `sync_bucket_
classifications()`) remains the largest deferred item, correctly ranked
last in the consolidation order for its risk relative to how central
Stage 5 is.

**3. Deskew angle - two of three call sites consolidated, one
deliberately left alone.** Stage 3 (`core/manifest_pipeline.py`'s
`preprocess_for_manifest`) used to re-estimate deskew independently of
Stage 1's physical sensor, with no override on `angle_range` - silently
using `estimate_deskew_angle()`'s own narrower 5.0 default instead of
Stage 1's deliberately wider 15.0 (`core/image_analysis.py`'s
`DESKEW_ANGLE_RANGE`, promoted from a private constant to a shared
public one this pass). Measured against the real corpus before
touching anything: 32/1974 images (1.6%) have `|deskew_angle_deg| > 5`,
meaning Stage 3's old narrower search would have landed on a different,
silently-clamped angle than Stage 1 actually measured for those pages.

Jon's call on the resulting fork (adopt Stage 1's value everywhere vs.
defer): **adopt it**. `preprocess_for_manifest()` now calls a new
`_resolve_deskew_angle()` helper - reads Stage 1's already-computed
`deskew_angle_deg` from the `*_analysis.json` sidecar when one exists
(same pre-dewarp working-copy pixels Stage 1 measured, so this is a
genuine reuse of the same observation, not a different one), falling
back to a fresh estimate using `DESKEW_ANGLE_RANGE` (not the buggy 5.0
default) when no sidecar exists yet - the common case today, since the
physical sensor isn't wired into `build_working_manifest_from_paths()`'s
automatic chain. Verified against a real divergent case (a real image
measured at 9.5 deg by Stage 1): `_resolve_deskew_angle()` returns
exactly 9.5, not a re-derived/clamped value; full `preprocess_for_
manifest()` round-trip confirmed the angle is actually applied to
pixels.

`core/auto_sidecar.py`'s own deskew estimate was deliberately NOT
consolidated the same way, despite calling `estimate_deskew_angle()`
with the same missing override - it measures the DEWARPED output
(`data/outputs/dewarped/...`), a genuinely different image from the
pre-dewarp working copy Stage 1 measured, since perspective correction
can change residual skew. Reusing Stage 1's stored value there would
have been measuring the wrong thing, not eliminating a duplicate. It
was fixed to use `DESKEW_ANGLE_RANGE` explicitly (closing the same
range-inconsistency bug) while keeping its own independent, necessary
measurement.

**4. Auto-sidecar review state wired into `pipeline_db`.**
`scripts/run_batch_auto_sidecar.py` had ZERO DB wiring before this
(confirmed by grep) - its review-worthy outcomes (a CV-fallback-guessed
template, quarantined rows, an unknown doc_type) were visible only
inside that run's own CSVs (`batch_run_summary.csv`, `needs_manual_
classification.csv`), never queryable pipeline-wide the way Stage 5's
`uncertain_review` state already is. Every per-image outcome is now
also recorded as a `stage_outputs` event under `stage="stage4_auto_
sidecar"` - status values mirror this script's own existing outcomes
(`skipped_not_dewarped`, `error`) plus one new value, `needs_review`,
covering BOTH "no sidecar produced at all" and the previously-invisible
case where `summary["status"]=="ok"` (a sidecar WAS written) but the
page still has a CV-fallback-guessed template or quarantined rows -
without this, that nuance only existed inside the sidecar JSON itself.
`done` only when neither condition applies. New `_resolve_image()`
helper handles the same "working_path moved after dewarp" resolution
`find_dewarped()` already needed, shared rather than duplicated.

Deliberately NOT done: merging `needs_manual_classification.csv` into
`uncertain_review.csv`/`reviewed_uncertain.csv` as one literal file -
different schemas, different (and in this queue's case, not-yet-built)
consuming tools. "Unified" here means both paths now write to the same
underlying `stage_outputs` table, queryable together by `image_id`
without knowing which CSV to open - not a file merge, which would have
been a larger, riskier restructuring than this pass's scope.

Verified with a full 7-scenario scratch test (clean / not-yet-dewarped
/ unknown doc_type / CV-fallback / quarantined rows / no-sidecar-
produced / raised exception), `generate_auto_sidecar()` monkeypatched
so the test exercises this script's own branching logic in isolation
from that function's real behavior (unchanged by this pass). Not yet
run against real production data - the current corpus has no `dense_
tabular_rows_subtype.csv` or dewarped images yet (Stage 5.5/dewarp
haven't been run on it since the reorg), so there was nothing real to
run this against.

**5. Classification write-authority flipped, with an explicit atomic
transaction boundary.** The largest deferred item, tackled last as
ranked. Before this pass, `core/classifier.py` wrote ONLY bucket CSVs
during its classify loop; `sync_bucket_classifications()` ran once,
after the whole batch finished, to bring the DB up to date FROM those
CSVs - the bucket CSV, not the DB, was the real authoritative producer
of classification state, exactly the inversion the target architecture
(sensors -> decision -> Decision State (`pipeline.db`) -> execution)
argues against.

Jon's addition before starting this: a named "transaction boundary"
concept - not because SQLite itself needs one (it already has real
transactions), but because *conceptually* the classifier should either
finish successfully for an image or leave its previous state completely
untouched, "that makes recovery much cleaner." Folded into this same
pass rather than done separately, per his own call ("do it all in the
same pass, saves having to go over it twice").

**What changed**: `PipelineDatabase.record_classification()` (new) wraps
the `images` UPDATE (bucket/classifier_confidence/classifier_model/
current_stage=5/status) and the `stage5_classify` `stage_outputs` INSERT
in one explicit `BEGIN`/`COMMIT`/`ROLLBACK` transaction - the connection
's `isolation_level=None` means autocommit per statement by default (see
`_connect()`), so without an explicit transaction these would be two
independent atomic units, not one. `core/classifier.py`'s `run()` now
calls this directly, per image, immediately after that image's bucket
CSV row is written (CSV-then-DB, deliberately - see below), instead of
relying on the end-of-batch sync. `sync_bucket_classifications()` is
kept, still called at the end of `run()`, but reframed as a safety-net
reconciliation pass rather than the primary write path - idempotent and
fast (~2.3s over the real 1975-image corpus, measured), so keeping it
costs nothing even when it finds zero changes (confirmed in testing: it
always found 0 additional changes once the atomic per-image writes had
already run).

**Why CSV-then-DB, not DB-then-CSV**: the two writes can't be made
truly atomic with each other (one's a SQLite transaction, the other's a
plain file write - different systems). Given that, the order matters
for what a partial failure leaves behind. `sync_bucket_classifications()`
only ever flows CSV -> DB, never the reverse, so it can heal a
"CSV row written, DB write failed" gap but has no mechanism to recover a
"DB updated, CSV row lost" one - and the CSV holds real evidence (Gemma's
`reason`/`text_density`/etc.) the DB never stores at all, so losing that
row would be a genuine, unrecoverable loss, not just a state gap.
CSV-then-DB means the one failure mode that can happen is exactly the
one the existing reconciliation pass already fixes.

**Verified, not run against production yet**: `record_classification()`
tested directly against a throwaway DB - a success case (both tables
updated correctly in one call) and a REAL forced-failure case (a bogus
`image_id` violating the `stage_outputs` -> `images` foreign key,
`foreign_keys=ON` per `_connect()`) confirming the exception properly
triggers `ROLLBACK` and leaves every other row completely untouched -
not inferred, directly observed via before/after row comparison. Full
`core/classifier.py.run()` also tested end-to-end (loader mocked, no
GPU/model involved - `nvidia-smi` checked clean first per CLAUDE.md's
rule anyway) across 4 real scenarios: high-confidence classification,
confidence-threshold override to `uncertain_review`, a raised model
exception, and a second bucket - bucket CSV contents confirmed identical
in shape/values to the pre-change behavior, DB state correct in every
case, reconciliation sync a clean no-op afterward. Not yet run against
the real production corpus/model - this pass changed the write
plumbing, not the classification logic itself, so a real run is a
matter of when Stage 5 is next invoked for real, not a prerequisite for
trusting this change.

## Physical sensor wired into the automatic chain (2026-08-04)

`build_working_manifest_from_paths()` (`core/manifest_pipeline.py`) now
runs `core.image_analysis.analyze_manifest()` immediately after Stage
1's semantic half (`stage1_capture_baseline_embeddings()`) and before
Stage 2 - both halves of Stage 1 now genuinely complete before Stage 3
ever touches a pixel, matching `docs/PIPELINE_STAGE_TERMINOLOGY.md`'s
definition of Stage 1 as one logical stage. This closes a gap that had
been standing, acknowledged, and explicitly NOT committed to either way
since the original DB migration (see this doc's own earlier "NOT yet
wired in" note). New `capture_physical_sensors: bool = True` parameter,
same on/off pattern as the existing `capture_baseline` flag.

**Real, measured cost, not estimated**: ~3.7s/image on this machine,
pure CPU (no GPU, no model inference - confirmed via a 15-image timing
sample before committing to anything larger). For a corpus in the
thousands this is a genuine multi-hour addition to every future Stage
0-3 run, not a rounding error - this is exactly why the CPU-concurrency
prototype (`benchmark/stage1_concurrency_experiment.py`, same session)
exists: to find out whether that cost can be reduced before treating it
as fixed.

Verified with a real 5-image end-to-end run through the full composed
chain (Stage 0 → Stage 1 semantic → Stage 1 physical → Stage 2 → Stage
3): all 5 images got a physical-analysis sidecar, all 5 got a correctly
recorded `stage1_image_analysis` "done" stage_outputs event, and the
rest of the chain (tower consensus, preprocessing) ran unaffected.

**Real mistake caught during that test, not by inspection first**:
`stage1_capture_baseline_embeddings()` and `analyze_manifest()` both
write to hardcoded default paths (`DEFAULT_BASELINE_PATH`,
`DEFAULT_REPORT_PATH`) that the test's `working_dir`/`manifest_path`/
`db_path` overrides didn't cover - the 5 scratch-path test images
actually got merged into the real `data/baseline_embeddings.json`
(1751 → 1756) and overwrote `data/outputs/image_analysis/
analysis_report.csv` with only the 5 scratch rows. Caught immediately
by checking for `wiring_test`/`scratchpad` paths in the real files
right after the test rather than assuming isolation worked; both
cleaned (`baseline_embeddings.json` restored to exactly 1751 real
records, the untracked/gitignored report CSV removed outright since it
had no real prior content to preserve). `data/pipeline.db` was
correctly isolated (test used its own `db_path`) and confirmed
untouched (integrity check + row count). Worth remembering for the
NEXT function that gets scratch-tested this way: overriding the
obvious path parameters isn't enough if the function also has its own
hardcoded defaults underneath.

## What's NOT done (explicitly deferred)

**Every live stage now dual-writes into the DB** - Stage 0/1/2/3/5,
human review corrections, dewarp Save/Bypass, and (as of this pass) the
physical sensor. `finalize_manifest()` reads from the DB. Every existing
CSV/JSON output any tool still depends on is UNCHANGED.

**Remaining work** (lower priority, no strong candidate flagged):
- Actually USING the recorded evidence (margins, physical/semantic
  cross-reference) to inform a real Decision Engine rule - explicitly
  NOT done in this pass, per Jon's own instruction not to begin
  threshold work yet.
- Every other `ui/*.py` tool (`ui/row_segmentation_ui.py`,
  `ui/build_manifest_ui.py`, etc.) - none inspected yet for whether they
  have an equivalent "dedicated live event" opportunity; not assumed to
  need one just because the pattern fit `review_uncertain.py`/`dewarp_
  preprocessor_ui.py`.

## Migration risks (carried forward for Phase 3)

1. **Hash-semantics ambiguity recurring** — addressed via the two-hash
   design above; this was the single highest-risk item given it already
   caused a real bug this session.
2. **Path-as-identity vs. row-as-identity, mid-migration** — most
   existing code identifies "an image" by its file-path string
   (`core/bucket_worklist.load_bucket_filepaths`, `core/classifier.py`,
   every `ui/*.py`). `get_or_create_image(path)`/`get_image_by_path(path)`
   exist specifically so not-yet-migrated tools keep working against
   plain paths while migrated ones use `image_id`.
3. **SQLite + concurrent processes** — `ui/build_manifest_ui.py` shells
   stages out as subprocesses; several Tkinter tools hold long review
   sessions open. Mitigated with WAL mode + a connection-per-call
   pattern (never held open across calls) — a real concurrent-write
   smoke test is still needed before Phase 3 touches any GUI tool.
4. **Existing production data must not be silently orphaned** — the
   migration script is read-only against every source file by
   construction (verified: `git status` clean under `data/` after the
   real run).
5. **`reference_pipeline_v1` stays untouched** — it's a deliberately
   frozen snapshot (`docs/REFERENCE_PIPELINE_V1.md`); this migration
   does not ingest or reference it.
6. **Dual-write during Phase 3** — a migrated stage must keep writing
   its existing CSV/JSON *and* the DB during the transition, since
   not-yet-migrated tools downstream still read the old file directly.
