# UI mockup integration notes (2026-08-03)

**Status**: investigation only. No code was written, no backend/API work
started. Captured here so a future session doesn't have to re-open and
re-inspect these mockups from scratch. This is future-stage work, not
scheduled - see "Recommendation" at the bottom for when it's reasonable
to pick up.

## What exists and where

`UI Mockups/` (project root - moved there from `J:\Genealogy\UI Mockups\`
by Jon during this investigation, so the in-app browser tool could
render them; browser tooling only fully renders files inside the
project folder) contains three Claude-Design-generated files:

- `OCR Pipeline - Developer Mockup.html`
- `OCR Pipeline - Basic User Mockup.html`
- `OCR Pipeline - Power User Mockup.html`

These are **Claude Artifact export bundles**, not plain HTML - each is
a self-contained React app packaged with a custom bundler format
(`<script type="__bundler/manifest">` / `<script type="__bundler/template">`
JSON payloads that reconstruct the real page client-side). They render
correctly in a browser; a plain text/grep read of the file only shows
the bundler's own unpacking JS, not the real UI - to inspect the actual
markup without a browser, extract and `json.loads()` the
`__bundler/template` script tag's text content, which contains the real
HTML as a string.

## What they actually are

All three share **one underlying data model** for a single pipeline
run, presented three different ways:

- **Developer mockup**: a live inspector panel with tabs - `Overview`,
  `Vision`, `Routing`, `OCR`, `Extraction`, `Performance`, `Logs` - plus
  a left-side stage stepper (`Image → Preprocess → Classify → Template
  → Extract → Validate → Output`) and a "now processing" card with a
  live confidence gauge.
- **Basic User mockup**: the identical underlying run state, relabeled
  in plain language with no jargon - `Preprocess`→"Preparing (cleaning
  up the scan)", `Classify`→"Identifying (figuring out the document
  type)", `Template`→"Layout (matching the right form)",
  `Extract`→"Reading (pulling out the details)",
  `Validate`→"Checking (quality review)", `Output`→"Finishing (saving
  to your family tree)". Same numbers (99.8% confidence, 1,241/1,751
  processed, etc.) as the Developer view.
- **Power User mockup**: a batch-run ops dashboard - per-stage status
  list (`Waiting`/`Running`/`Complete`/`Completed with warnings`/
  `Failed`/`Cancelled`), a resource-usage sidebar (GPU allocated/
  reserved, VRAM%, CPU%, RAM, disk I/O, live sparklines), and an
  explicit "reference - failure display pattern" showing what a failed
  stage looks like in the UI.

## The data model is not generic - it already matches real backend fields

This was the most useful finding of the investigation: several fields
shown in the mockups are near-exact matches to fields this project
already has, not placeholder/generic values:

- Developer mockup's "Vision" tab: `Deskew Angle: 0.8°`, `Page
  Confidence: 0.997`, `Polarity: Dark on light`, `Operations: deskew →
  dewarp → normalize` - these match `core/image_analysis.py`'s real
  `ImageAnalysis.deskew_angle_deg`, `.page_confidence`,
  `RegionStats.ink_is_dark_on_light` fields almost exactly.
- Developer mockup's "Routing" tab: `Routing Reason: Confidence above
  threshold` - matches `core/classifier.py`'s real `min_confidence`
  gate logic exactly (`pipeline.yaml`'s `classifier.min_confidence`,
  currently 0.6).
- Power User dashboard: "3 pages flagged for skew > 4°" as an
  `Image Analysis` stage warning - directly matches the deskew-angle
  consolidation work done earlier this same session
  (`core/manifest_pipeline.py`'s `_resolve_deskew_angle()`,
  `core/image_analysis.py`'s `DESKEW_ANGLE_RANGE`).
- Performance/resource panels (both Developer and Power User mockups):
  GPU VRAM allocated/reserved, GPU utilization %, CPU %, RAM, images/sec,
  per-stage timing (`Preprocessing: 0.3s/img`, `OCR: 1.9s/row`) - this
  is the exact shape of the CPU/GPU benchmarking methodology discussed
  earlier the same session (`psutil` for CPU/disk/images-per-sec,
  `torch.profiler` for operator-level breakdown - see the Stage 1
  scheduling research conversation this doc's own session covered).

## What's missing before this could actually be wired up

1. **No API or live-update layer exists anywhere in this project.**
   Every current UI tool (`ui/dewarp_preprocessor_ui.py`, `ui/
   classifier_validation_ui.py`, `ui/build_manifest_ui.py`, `debug_tools/
   review_uncertain.py`) is Tkinter calling Python functions in-process.
   Nothing serves state over HTTP or WebSocket. These mockups expect
   live-updating values (progress %, ETA, a scrolling log tail,
   resource sparklines over a rolling window) - that needs a real
   server layer (most likely FastAPI + WebSocket or polling) that does
   not exist in any form today. This is the actual size of the
   integration work, not the frontend design.
2. **No structured logging.** The Logs tab expects filterable `INFO`/
   `WARN` events with timestamps. Current stages `print()` to stdout -
   fine for a CLI, not queryable by a UI.
3. **No live resource telemetry capture.** The GPU/CPU/RAM/VRAM panels
   need real-time system monitoring - nothing in this project currently
   captures this (see the Stage 1 scheduling-research conversation for
   what's available: `psutil` already installed, `torch.profiler`
   available, true memory-bandwidth profiling genuinely hard to get on
   Windows without specialized tooling).
4. **Stage naming doesn't match the canonical Stage 0-6 taxonomy.** The
   mockups use `Image / Preprocess / Classify / Template / Extract /
   Validate / Output`; `docs/PIPELINE_STAGE_TERMINOLOGY.md`'s canonical
   names are `Source Acquisition / Raw Sensor Capture / Decision Engine
   / Image Processing / Validation Capture / Document Routing /
   Specialized Extraction`. Not a blocker, but a reconciliation
   decision to make before an API's field/endpoint names get chosen -
   picking one vocabulary (or an explicit mapping table) up front avoids
   the two taxonomies drifting further apart.

## Why this is lower-effort than a typical mockup-to-product gap

`core/pipeline_db.py` already being the single source of truth for
pipeline state (built earlier this session's consolidation pass) is
exactly the shape a REST/WebSocket API wants to sit on top of -
`db.list_images()` / `db.get_stage_outputs()` map naturally onto "query
current state" endpoints, and the existing evidence-vs-decision-state
split means an API layer wouldn't need to invent a new query model, just
expose the one that already exists. If the DB/consolidation work hadn't
happened, this would be a meaningfully bigger lift (either inventing a
query layer at API-build time, or reading scattered CSVs from a web
backend).

## Recommendation

Not blocking anything, but Phase 1 below was actually started while
Gemma's classification run was in progress (2026-08-04), reusing spare
time rather than waiting - see below.

## Phase 1 - read-only API (built 2026-08-04)

`api/main.py` (FastAPI) + `api/schemas.py` (Pydantic response models).
Five GET endpoints, no live updates anywhere (no WebSocket/SSE/polling
loop in this module) - every response is computed fresh from `core/
pipeline_db.py` at request time:

- `GET /images` - filterable (`bucket`, `status`, `current_stage`,
  matching `PipelineDatabase.list_images()`'s existing signature
  exactly), paginated (`limit`/`offset`, applied in this layer).
- `GET /image/{image_id}` - one image's current state, 404 if unknown.
- `GET /stage/{image_id}` - that IMAGE's full stage_outputs history
  (not a lookup by `stage_outputs.id` - see the endpoint's own
  docstring for why), optionally narrowed by `?stage=`.
- `GET /logs` - corpus-wide `stage_outputs`, most recent first. New
  `PipelineDatabase.list_stage_outputs()` method added for this (the
  only new DB-layer method Phase 1 needed - everything else already
  existed). Not the same shape as the mockups' human-readable INFO/WARN
  log lines - this project has no structured-logging system, so this
  exposes the real event log that already exists rather than
  fabricating a second one.
- `GET /performance` - a single point-in-time system snapshot, not
  live/streaming. CPU%/disk I/O rate sampled over a short (~0.3s)
  blocking window during the request (`psutil`); GPU via an
  `nvidia-smi` subprocess, deliberately NOT `torch.cuda` in-process -
  querying this process's own CUDA context would report only this
  process's allocation, not real system-wide GPU state, and would mean
  creating a CUDA context in the API process at all while a real
  inference run might be active elsewhere on the same GPU (CLAUDE.md's
  GPU-contention discipline - this module never imports torch).
  `recent_images_per_sec` is derived from real `stage_outputs`
  timestamps over a rolling window, not an instrumented live counter.

**Verified against the real, live database while Gemma's classification
run was actively in progress** (not a scratch/synthetic test): started
the server, hit all five endpoints, confirmed real responses - `/image/1`
showed a fully classified image (`dense_tabular_rows`, confidence 0.98)
with a complete `stage0→stage1→stage2→stage3→stage5` history via
`/stage/1`; `/logs` showed live classification events a few seconds
apart landing in `printed_document`; `/performance` captured Gemma's
actual live GPU usage (96% utilization, 12.2/16.3GB VRAM, correctly
identified as a separate process from the API server). This also
incidentally end-to-end-verified the Stage 5 write-authority/atomic-
transaction consolidation from earlier the same session, live, for
real, since `/image/1`'s complete state is exactly what that change was
supposed to produce. Test server (port 8123) was stopped cleanly
afterward via its own PID (found through `netstat`, unrelated to
Gemma's PID) - read-only the entire time, Gemma's run was never
touched.

**Explicitly not done (Phase 2+, per Jon's own scoping)**: live updates
of any kind (WebSocket/SSE/polling), the stage-naming reconciliation
between this API and the mockups' vocabulary, wiring an actual frontend
to these endpoints, structured (non-`stage_outputs`-derived) logging,
historical/rolling-window resource telemetry (only current-instant
snapshots exist).

### Running it

```
uvicorn api.main:app --reload --port 8000
```

Then `http://127.0.0.1:8000/docs` for the auto-generated Swagger UI.
