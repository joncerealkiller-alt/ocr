# Reference Pipeline v2 — full pipeline-state checkpoint before the folder-structure reorg

**Status**: preserved reference data, not a failed experiment. Do not
overwrite or delete `data/outputs/reference_pipeline_v2/`.

## What this is

A frozen snapshot of the ENTIRE live Stage 0-5 pipeline state for the
1975-image corpus, captured 2026-08-03, immediately before Jon began a
fresh run against a reorganized source-folder structure:

- `manifest.csv`, `manifest_final.csv`, `manifest_provenance.json` -
  Stage 0's acquisition manifest and provenance record.
- `pipeline.db` - the full orchestration database at the moment every
  consolidation item in this session's architecture-review pass had
  just been completed (write-authority flip + atomic transaction
  boundary for Stage 5, `find_dewarped()`/`needs_manual_dewarp`/
  auto-sidecar review-state consolidation - see `docs/
  PIPELINE_DATABASE.md`'s consolidation-pass section for the full
  list). 1975 images, 17775 stage_outputs rows, integrity-checked
  before and after this copy.
- `baseline_embeddings.json` - all 8 qualified vision towers' pooled
  embeddings for all 1975 images (Stage 1 semantic sensor).
- `buckets/*.csv` - Stage 5's 9 category CSVs (the classifier's real
  output for this corpus) - NOT `data/buckets/reference_prerefactor/`,
  which is a DIFFERENT, older checkpoint from an earlier pass, already
  present and untouched by this one.
- `working/` - all 5923 files: every working-copy image plus its
  `*_analysis.json` (physical sensor) and `*_tower_consensus.json`
  (semantic tower-consensus) sidecars.
- `raw_from_pdf/` - Stage 0's PDF-expansion intermediate output (16
  files) for whatever PDF sources fed this corpus.

Broader in scope than `reference_pipeline_v1` (`docs/
REFERENCE_PIPELINE_V1.md`), which was Stage 1 sensor measurements only.
This checkpoint captures the FULL pipeline state (manifest, DB, buckets,
embeddings, working corpus) because the reason for creating it is
different: v1 existed to benchmark sensor output; this one exists so a
clean, verifiable "before" state survives a fresh run against
reorganized source folders (Jon: "ill do a fresh run... backup the
current files from this existing pass as a reference checkpoint...
leaving everything in a fresh clean state for a new run").

That said, this checkpoint ALSO satisfies what v1's own doc explicitly
asked for: "a second reference dataset (`reference_pipeline_v2` or
similar, following the same archival pattern) should be captured and
compared directly against this one" - the Stage 0/1/3 separation v1
was waiting on is real now, and this genuinely is that comparison
point, just captured for a more immediate reason too.

## Why this exists

Two independent things converged at this moment: (1) the architecture
consolidation pass (duplicate observations/decisions eliminated,
Stage 5 write-authority flipped to the DB with an atomic transaction
boundary) had just finished, and (2) Jon is reorganizing his source
archive's folder structure (see the conversation this was captured
from - a mix of document-type folders like `Census`/`Land Grants` and
provenance-only folders like family surnames/Manitoba place names) and
wants a completely fresh Stage 0-5 run against the new layout, not an
incremental one.

A fresh run needs `data/manifest.csv`, `data/pipeline.db`, `data/
baseline_embeddings.json`, `data/buckets/*.csv`, and `data/working/`
all genuinely empty/absent - not because leaving old content there
would corrupt anything (each stage generally tolerates being re-run),
but because a truly fresh corpus is easiest to reason about, and it's
what was explicitly asked for.

## What was cleared vs. left alone

**Cleared** (backed up here first, verified byte-identical/integrity-
checked, THEN deleted from their live locations): `data/manifest.csv`,
`data/manifest_final.csv`, `data/manifest_provenance.json`, `data/
pipeline.db`, `data/baseline_embeddings.json`, `data/buckets/*.csv`
(9 files), `data/working/*` (5923 files - directory itself kept,
empty), `data/raw_from_pdf/*` (16 files - directory itself kept,
empty).

**Deliberately left alone** (not part of "this pass's" live pipeline
state, so not backed up here and not cleared): `data/
reference_prerefactor/` and `data/buckets/reference_prerefactor/` (an
already-existing, older checkpoint from an earlier pass - a different
thing from this one), `data/archive_census_only_run/` and `data/
pending_prune_review/` (unrelated to the Stage 0-5 corpus flow),
`data/outputs/` otherwise (mixed research/benchmark history, not
exclusively this corpus's state).

## Verification performed before clearing anything

File counts (`working/`: 5923/5923, `raw_from_pdf/`: 16/16) and exact
byte sizes matched for every individual file; all 9 bucket CSVs
confirmed byte-identical via sha256; the copied `pipeline.db` passed
`PRAGMA integrity_check` and its `images`/`stage_outputs` row counts
matched the live DB (1975 / 17775) before the live DB was deleted.
Copying was done with Python's `shutil` rather than the shell's `cp` -
this session found `cp` unreliable on this project's `J:` drive earlier
(silently incomplete copies), `shutil.copy2`/`copytree` did not
reproduce that problem.

## Do not

- Overwrite or delete anything under `data/outputs/reference_pipeline_v2/`.
- Confuse this with `data/reference_prerefactor/` - a different,
  earlier checkpoint, still present and untouched.
- Assume the live files this was copied from still exist - they don't;
  a fresh Stage 0 run is expected to recreate `data/manifest.csv`,
  `data/pipeline.db`, `data/baseline_embeddings.json`, `data/
  buckets/*.csv`, and populate `data/working/`/`data/raw_from_pdf/`
  from scratch against the reorganized source folders.
