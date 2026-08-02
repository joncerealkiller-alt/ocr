# Pipeline workflow: raw image → sidecar → validation

This documents the actual current pipeline as of the 2026-07-27/28
"automated rebuilt stage" work (Stage 0 / auto-sidecar / semantic-stages
rebuild), reconstructed from the code since it was never written up.
Each stage below is a separate, manually-invoked CLI script chained by
convention — there is **no** one-click orchestrator yet.
`debug_tools/workflow_gui.py` predates this rebuild by ~2 weeks and does
**not** cover any of it (its tabs only wrap the older segmentation/
extraction/classification scripts).

## Stage 0 — Copy + deskew + preprocess ("working manifest")

```bash
python scripts/build_working_manifest.py --source-folder "J:\path\to\raw_scans"
```
(omit `--source-folder` for a folder-picker dialog)

- Calls `core/manifest_pipeline.py`'s `build_working_manifest()`.
- Input: a folder of raw scans (originals never touched, only read).
- Recursively collects images, copies them flat into `data/working/`
  (name collisions get a numeric suffix), then for every copy: deskews
  (`core/row_segmentation.py`) and applies the `autocontrast`
  preprocessing profile (`core/image_preprocessing.py`) **in place**,
  for every doc type, unconditionally.
- Output: `data/working/*.{ext}` + `data/manifest.csv` (`file_path` column).
- Fully automatic.

## Stage 1 — Bucket classification (unchanged, pre-existing)

```bash
python -m core.classifier data/manifest.csv
```

- Not touched by the rebuild.
- Input: `data/manifest.csv` from Stage 0.
- Output: `data/buckets/<category>.csv` per document category (e.g.
  `dense_tabular_rows.csv`), plus `uncertain_review.csv` for
  low-confidence/failed classifications.
- Fully automatic (model-based classification).

## Stage 2a — Manual dewarp (dense_tabular_rows only — human-in-the-loop)

```bash
python ui/dewarp_preprocessor_ui.py
```
Run in bucket-worklist mode pointed at `data/buckets/dense_tabular_rows.csv`.

- For each file, drag 4 corner handles onto the page's real edges to
  perspective-correct it, or click "Bypass / Use Original" if it isn't
  distorted.
- Output: `<name>_dewarped.jpg` under `data/outputs/dewarped/`, plus
  `data/buckets/dense_tabular_rows_dewarped.csv`.

**This step is still manual and mandatory for every dense_tabular_rows
file — nothing has automated it yet.** `core/warp_detection.py`
(`detect_warp()`, a ruling-line X-drift heuristic) exists and is real,
tested code, but calibration against 23 real image pairs failed twice
(v1 and v2), so it's **not wired into the pipeline anywhere**.
`core/manifest_pipeline.py`'s `_dense_tabular_needs_manual_dewarp()` —
the one insertion point for that automation — currently just
`return True` unconditionally. Every other bucket (single whole-image
model calls) skips this step because mild warp doesn't hurt them.

## Stage 2b — Finalize manifest (merge classification + dewarp)

```bash
python scripts/finalize_manifest.py
```

- Calls `core/manifest_pipeline.py`'s `finalize_manifest()`. Safe to
  re-run anytime — always rebuilds from scratch, never appends.
- Input: every `data/buckets/*.csv` bucket, plus (for
  `dense_tabular_rows`) `data/buckets/dense_tabular_rows_dewarped.csv`.
- Output: `data/manifest_final.csv` with columns
  `file_path, category, status, source_original_path`, where `status` is:
  - `ready` — non-dense_tabular_rows bucket, usable as-is
  - `ready_dewarped` — dense_tabular_rows, dewarp done; `file_path`
    points at the dewarped output
  - `pending_manual_dewarp` — dense_tabular_rows queued but not yet
    dewarped; **downstream stages should skip these rows**

## Stage 3 — Semantic sub-type classification (Gemma)

```bash
python scripts/run_semantic_stages.py
```

- Calls `core/semantic_stages.py`.
- Input: `data/buckets/dense_tabular_rows.csv` (the classifier's
  original bucket CSV, **not** `manifest_final.csv`).
- Prompt: `config/prompts/classifier_document_subtype_v2.txt` — use
  **v2**, not v1 (v2 fixed a Canada/UK 1901 misclassification bug).
- Loads Gemma once, classifies census year / printed-vs-handwritten /
  unknown per row, reading title text as a signal.
- Output: `data/buckets/dense_tabular_rows_subtype.csv`
  (`file_path, document_type, confidence, title_text_read, reason, error`).
- Fully automatic (one GPU load for the whole batch). Doesn't check
  dewarp status itself — ideally run after Stage 2a for the pages you
  care about, though it'll happily classify a not-yet-dewarped working
  copy too since it only needs a page image, not precise geometry.
- `config/prompts/classifier_region_anchors_v1.txt` exists but is
  referenced only in a **commented-out** future stage in this script —
  not currently active, don't treat it as a real step.

## Stage 4 — Automated sidecar (row/column geometry) generation

This is the step that produces the sidecar file the validation UIs consume.

### 4a. Single-file (ad hoc)
```bash
python scripts/auto_generate_sidecar.py <dewarped_image.jpg> [--out PATH] [--debug] [--force]
```
- Input: one already-dewarped image.
- Output: **`data/outputs/auto_row_segmentation/<stem>_sidecar.json`**
  (see the directory-gap warning below).
- No `doc_type_override` given, so it runs `core/document_classification.py`'s
  rule-based CV classifier (aspect ratio + ruling-line count + row-count
  estimate → best `config/document_templates/*.yaml` template) itself.

### 4b. Batch, Gemma-driven (preferred path)
```bash
python scripts/run_batch_auto_sidecar.py [--input data/buckets/dense_tabular_rows_subtype.csv] [--out-dir data/outputs/auto_row_segmentation] [--debug] [--force]
```
- Input: `data/buckets/dense_tabular_rows_subtype.csv` (Stage 3's
  output). Finds the matching dewarped image via
  `data/outputs/dewarped/<stem>_dewarped.*`. Rows with `document_type`
  in `("", "unknown")` are skipped; rows with no dewarped image found
  are skipped too.
- Passes Stage 3's `document_type` in as `doc_type_override`, which
  **skips** the CV classifier entirely and goes straight to the
  matching template — i.e. in this path, Gemma drives template
  selection, not the CV classifier (the CV classifier is retained,
  unmodified, only for the 4a standalone path).
- Output: `data/outputs/auto_row_segmentation/<stem>_dewarped_sidecar.json`
  per row, plus `batch_run_summary.csv` (`ok` /
  `skipped_unknown_doc_type` / `skipped_not_dewarped` /
  `skipped_already_exists` / `unknown_after_all` / `error`).
- Fully automatic, no model inference here (Gemma already ran in Stage 3).

### What `generate_auto_sidecar()` (`core/auto_sidecar.py`) actually does
1. Opens the image, estimates+applies deskew itself (used directly, no
   human confirmation step in this path).
2. Uses `doc_type_override` if given, otherwise calls
   `classify_document()` itself — it does **not** require classification
   to be resolved externally; it can run fully end-to-end from a raw
   dewarped image.
3. Loads the matching `DocumentTemplate` (`core/document_templates.py`,
   `config/document_templates/*.yaml`).
4. `locate_table_boundary()` / `locate_header_region()` refine the
   template's approximate regions against real detected ruling lines.
5. `detect_data_rows()` — fixed-layout templates (census years) locate
   row 1 and tile the rest arithmetically; `row_strategy: "detect"`
   templates (manifests) do real per-row band detection with a preset
   retry ladder (`default` → `lenient` → `lenient_alt`).
6. Builds the sidecar via `core/row_segmentation.py`'s existing
   `build_sidecar()` — **same schema** the older manual
   `ui/row_segmentation_ui.py` workflow produces, so downstream code
   needs no changes.
7. Quarantine thresholds apply at both the whole-page level
   (`PAGE_QUARANTINE_THRESHOLD = 0.50`) and per-row
   (`rows_needs_review`) — a page classified `"unknown"` (confidence
   < 0.5) yields **no sidecar at all**.

### ⚠️ Known gap: directory mismatch

`scripts/auto_generate_sidecar.py` / `run_batch_auto_sidecar.py` write
to **`data/outputs/auto_row_segmentation/`**. The validation UIs
(`ui/hint_validation_ui.py`, `ui/ground_truth_labeling_ui.py`) default
their file-picker to **`data/outputs/row_segmentation/`** — the older,
manual `ui/row_segmentation_ui.py` output directory. **Nothing currently
copies or promotes a sidecar from `auto_row_segmentation/` into
`row_segmentation/`.** Until this is resolved, either:
- manually copy the reviewed sidecar JSON across, or
- browse to `data/outputs/auto_row_segmentation/` directly in the
  validation UI's file picker (it accepts any path, the default is
  just a starting directory), or
- decide this promotion step needs to be built.

## Stage 5 — Validation / ground truth

```bash
python ui/hint_validation_ui.py       # CSV-hint-assisted confirm/deny + manual fallback
python ui/ground_truth_labeling_ui.py # fully manual, field-by-field
```

Load a Stage-4 sidecar (see the gap above for where to find it), a
`config/columns/*.txt` column list, and — for `hint_validation_ui.py`
only — a reference CSV (e.g. `data/automatedgenealogy_pull.csv`).

## Downstream (FYI, not part of getting to a sidecar)

- `scripts/run_row_extraction.py` — consumes a confirmed sidecar, runs
  OCR extraction per row (single-column, writing into
  `sidecar.columns[name]["results"]`, or legacy multi-column mode with
  a `columns.txt`). Outputs `<name>_extraction.csv`/`.json`.
- `scripts/run_two_stage_extraction.py` — two-model pipeline (raw OCR
  engine, then a structuring VLM/LLM) against a sidecar +
  `columns.txt`. Outputs `<name>_twostage_extraction.csv`/`.json`.

## Summary table

| # | Stage | Script | Automatic? |
|---|-------|--------|------------|
| 0 | Copy + deskew + preprocess | `scripts/build_working_manifest.py` | Yes |
| 1 | Bucket classification | `python -m core.classifier` | Yes |
| 2a | Manual dewarp (dense_tabular_rows) | `ui/dewarp_preprocessor_ui.py` | **No — manual, mandatory** |
| 2b | Merge manifest | `scripts/finalize_manifest.py` | Yes |
| 3 | Gemma sub-type classification | `scripts/run_semantic_stages.py` | Yes (model call) |
| 4 | Auto sidecar generation | `scripts/run_batch_auto_sidecar.py` (preferred) / `scripts/auto_generate_sidecar.py` | Yes |
| — | **Gap:** promote sidecar to `data/outputs/row_segmentation/` | Not implemented | Needs a decision |
| 5 | Validation | `ui/hint_validation_ui.py` / `ui/ground_truth_labeling_ui.py` | Manual review |
