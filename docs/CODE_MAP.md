# Code map — reusable functions, classes, and schemas

**Purpose**: this is NOT a how-to-run guide (see `README.md` and
`PIPELINE_WORKFLOW.md` for that) and NOT a directory-layout overview
(see `README.md`'s "Project layout"). This is a symbol-level index of
the reusable primitives already built — the stuff a session doing new
work would otherwise burn a lot of tool calls rediscovering (an Explore
agent + a dozen manual reads, in the case that prompted this file:
building `benchmark/prompt_sweep.py`, which turned out to need exactly
the primitives listed under "Row-level extraction" and "Ground truth &
scoring" below, all of which already existed).

**Upkeep is the whole point of this file.** A stale entry is worse than
no entry — it gets trusted without verification. Whenever a session does
real exploration to find how something already works (not just "I
remember roughly"), add or correct an entry here before finishing, the
same way `README.md`'s change log gets appended for structural changes.
Keep entries short: signature, one line of what it's for, one line of
who already calls it if that helps context. Point to the file for the
full docstring rather than duplicating it — this file goes stale fast if
it tries to be the documentation instead of an index into it.

---

## Architectural principle: config owns behavior, loaders own mechanics

**Shared loaders must remain task-agnostic. Task-specific behavior
(system prompts, token budgets, generation settings, reasoning toggles)
belongs in the model's YAML config, not hardcoded in loader Python.**
The loader's job is "read config → assemble request → call model →
return response" - never "if this model then that personality."

This isn't aspirational - it's already the pattern for
`image_token_budget`, `reasoning_enabled`, `restrict_output_charset`,
`prompt_text`: every one of those lives in `config/models/*.yaml`
(mostly under `extra:` for loader-specific knobs), read via
`self.config.*` at call time. A model config file fully determines that
model's behavior; the same loader class serving two different configs
(e.g. `gemma.yaml` vs `gemma_extract.yaml`, both `loader_class:
GemmaLoader`) should behave completely differently if their configs say
to, with zero code branching on which config it is.

**Real bug this caught (2026-07-30)**: `core/loaders/gemma_loader.py`'s
`_run_generate()` had a hardcoded system prompt ("You are a strict,
non-interpretive archival routing classifier.") applied unconditionally
to every call, regardless of task or which config was driving it. It
went unnoticed because `_build_prompt()` (used by the normal `.classify()`/
`.extract()` path) IS task-gated and fails fast for anything but
classification - but any caller that invokes `_run_generate()` directly
(e.g. `benchmark/prompt_sweep.py`, which bypasses the task machinery by
design so it can control prompts per sweep combo) skipped that gate
entirely and got the classifier's personality injected into an
extraction call. Fix: `system_prompt` moved into `extra:` in each
model's own YAML (empty/unset = no system message sent at all -
**deliberately no code-level fallback to any specific wording**, so a
new model config that forgets to set one gets generic behavior, not an
accidentally-inherited personality from whichever model happened to get
written first).

**When adding a new model config or a new loader capability**: ask "is
this a property of THIS model/task, or genuinely true for every model
this loader will ever serve?" If the former, it's a config field, not a
Python literal - even if only one config currently needs a non-default
value.

---

## Model loading & config

- **`core/loader_registry.py`**: `LOADER_REGISTRY: dict[str, type[BaseLoader]]`
  — maps a model YAML's `loader_class` string (e.g. `"QwenLoader"`,
  `"NanonetsOcr2Loader"`) to the loader class. Look a model up via
  `LOADER_REGISTRY.get(config.loader_class)`, not a hardcoded per-model
  if/else.
- **`core/loaders/base_loader.py`**:
  - `load_model_config(model_name: str) -> GenerationConfig` — reads
    `config/models/<model_name>.yaml`.
  - `GenerationConfig` (dataclass) — every YAML field (temperature,
    top_p, max_new_tokens, `prompt_text`, `restrict_output_charset`,
    `reasoning_enabled`, etc.) plus `.content_hash()` (sha256, for
    traceability) and `.build_max_memory_map()`.
  - `BaseLoader` (ABC) — the common interface every model loader
    implements: `initialize_model_and_tokenizer()` (expensive/stateful —
    see lifecycle note below), `_build_prompt(task)` (`"classify"`/
    `"extract"`), `_run_generate(image, prompt) -> str` (the actual
    inference call — same across every loader), `.classify()`/
    `.extract()` (higher-level wrappers), `.release()` (no-op by
    default, overridden for subprocess-backed loaders — Moondream,
    DeepSeek-VL2, Hunyuan), `.apply_checkpoint(path)` (LoRA adapter via
    `PeftModel.from_pretrained`).
- **Load-once-per-model lifecycle convention**: loading a model is
  expensive (GPU weights) and this project has hit a real, confirmed
  failure from two models resident/running concurrently (moondream2 —
  see `CLAUDE.md`). Every existing caller loads once, does its work,
  releases in a `finally`:
  - `core/extractor.py`'s `_loader_cache: dict[str, Any]` — keyed by
    **model_name only** (not model+prompt) — `get_or_build_loader()`
    swaps `loader.config.prompt_text` on an already-loaded instance
    rather than reloading. Caching by (model, prompt) was tried and
    caused VRAM to climb every bucket — don't reintroduce that.
  - `core/row_extraction.py`'s `_release_model(loader)` — null out
    model/processor/tokenizer, `gc.collect()`, `torch.cuda.empty_cache()`.
  - `run_two_stage_extraction()` loads stage-1, runs every row, releases,
    *then* loads stage-2 — never two models loaded at once.
  - **Before editing any loader file or launching inference**: check
    `nvidia-smi` first — see `CLAUDE.md`'s gate at the top of the repo.

## Prompt & column-list files

- **`config/prompts/*.txt`** — plain text, no shared loader abstraction.
  Every caller does `path.read_text(encoding="utf-8")` and assigns to
  `config.prompt_text` (or passes the string straight into a function
  param, e.g. `run_two_stage_extraction(..., ocr_prompt=...)`). Naming:
  `classifier_*`, `extractor_<bucket>_v{N}`, `ocr_stage1_*` (stage-1 OCR
  reading, often model/field-specific), `structuring_stage2_*` (stage-2
  templates, need `{raw_ocr_text}`/`{columns_str}`/`{example_lines}`
  placeholders — see `build_structuring_prompt` below).
- **`config/columns/*.txt`** — one column name per line, in OUTPUT order
  (not necessarily left-to-right form order) — the curated field list
  offered for manual masking in `ui/quarantine_review_ui.py`. As of
  2026-07-29 this is looked up via each doc type's own
  `config/document_templates/*.yaml` → `columns_file:` field (see
  `core/document_templates.py` below), not a hardcoded dict.
- **`core/document_templates.py`**: `DocumentTemplate` dataclass +
  `load_template(doc_type)` / `load_all_templates()` (auto-scans
  `config/document_templates/*.yaml` — adding a doc type needs no code
  change). Fields: `row_strategy`, `expected_row_count`,
  `expected_columns`, `regions_approx`, `column_regions_approx`,
  `number_row_approx`, `expected_row_height_frac`, `columns_file`.

## Row-level extraction (`core/row_extraction.py`, `core/row_segmentation.py`)

This is the pipeline's per-row/per-field extraction engine — one row is
one census person / one manifest line, cropped from the source image in
memory. Genuinely different schema from `core/schema.py`'s
`ExtractionResult` (that's for whole-document extraction).

- `RowFieldValue` (`value`, `confidence: ConfidenceLevel`) /
  `RowExtractionResult` (`row_index`, `bbox`, `fields: dict[str,
  RowFieldValue]`, `raw_output`, `model`, `runtime_seconds`,
  `schema_pass`, `schema_error`, `stage1_raw_output`).
- `build_row_prompt(column_names)` — builds the single-stage extraction
  prompt IN PYTHON from column names. **No override param** — not
  swappable without a code edit (unlike the two-stage prompts below).
- `build_structuring_prompt(raw_ocr_text, column_names,
  template_override=None)` — stage-2's prompt. `template_override` lets
  you swap the wording via a file instead of editing this module — see
  `config/prompts/structuring_stage2_default.txt` for a working
  template (`{raw_ocr_text}`, `{columns_str}`, `{example_lines}` required;
  `{num_columns}`, `{plural}`, `{field_hint}` optional).
- `parse_row_output(raw_output, column_names) -> dict[str, RowFieldValue]`
  — parses "ColumnName: value|confidence" lines, tolerant of several
  real near-miss shapes models have produced (see its docstring).
- `_resolve_column_field_mask(sidecar, column_state) -> (row_masks,
  mask_active, tight_crop_ranges)` — reads ONE column's own persisted
  mask (`sidecar["columns"][name]["mask_keep_ranges"]`) — NOT the
  unrelated `sidecar["columns"]["__multi__"]` scratch mask (unlabeled
  union, can't drive per-field crops). This is the shared logic behind
  every per-column crop in the codebase — reuse it, don't reinterpret
  "this column's boundary" a second way.
- `_field_bbox_from_keep_ranges(row_bbox, keep_ranges)` — full-image
  bbox spanning a column's keep_ranges, for debug metadata.
- `core/row_segmentation.py`: `load_sidecar(path)` / `update_sidecar(path,
  column_name, patch)`, `crop_region_from_source(source_image_path, bbox,
  deskew_angle, mask_ranges=None, tight_crop_keep_ranges=None,
  tight_crop_padding_px=20, upscale_target_height=None, ...)` — the
  primitive for "load the original source image and crop a region",
  re-applying deskew so stored (already-deskewed-space) coordinates line
  up. `compute_exclude_ranges()` turns keep-ranges into paint-white
  exclude-ranges.
- `_release_model(loader)` — see lifecycle note above.
- High-level entry points, all load-once/release internally:
  `run_row_extraction()`, `run_single_column_extraction()` (writes
  straight into the sidecar's persistent `columns[name]["results"]`),
  `run_two_stage_extraction()` (field-level: one OCR call + one
  structuring call PER SELECTED COLUMN per row, not one whole-row call —
  loads OCR model, runs every row, releases, then loads structuring
  model), `extract_page_header()`.
- `save_results_csv()` / `save_results_json()`.
- `ConfidenceLevel` (`core/schema.py`): `CONFIRMED` / `PARTIAL` /
  `UNCLEAR` / `NOT_PRESENT`.

## Ground truth & scoring

- **`data/outputs/ground_truth_log.jsonl`** — append-only JSONL, one
  record per (sidecar_path, row_index, column):
  `{"timestamp", "sidecar_path", "source_image_path", "row_index",
  "column", "status": "readable"|"partially_readable"|"illegible"|"blank",
  "value", "notes"}`. Written by `ui/ground_truth_labeling_ui.py`. As of
  2026-07-29, only 3 pages are labeled (~750 field records, 5 columns) —
  check this file's current size before assuming a bigger eval set exists.
- **`benchmark/score_two_stage_against_ground_truth.py`**:
  `score_results(results_path, ground_truth_path, sidecar_path,
  model_name=None) -> ScoringReport` — THE evaluator, reused by the CLI,
  `debug_tools/workflow_gui.py`'s `ScoringTab`, and
  `benchmark/prompt_sweep.py`; never reimplement this logic elsewhere.
  `_is_correct(predicted, expected)` — exact match after
  strip+casefold, EXCEPT abstention normalization: `expected=="blank"`
  matches predicted `""` or `"blank"`; `expected=="illegible"` matches
  predicted `"?"` or `"illegible"` (a real scoring bug this fixed — see
  its docstring). `_record_to_expected(record)` normalizes a
  ground-truth record to the expected string.
- **`training/test_lora_checkpoint.py`**: `_classify_mismatch(status,
  predicted) -> "honest_hedge" | "incorrectly_silent" | "confident_wrong"`
  + `ABSTAIN_STATUSES = {"illegible", "blank"}` — the three-way
  failure-mode taxonomy ("abstention is a feature, not a failure" — see
  `feedback_abstention_is_a_feature` memory). `honest_hedge` = contains
  `?` or silent on an abstain-status; `incorrectly_silent` = silent on a
  readable status; `confident_wrong` = asserted specific false content —
  the real hallucination-risk bucket. Reused by `benchmark/prompt_sweep.py`
  to report Correct/Abstained/Wrong.
- **`training/report_lora_evals.py`** — the pattern for "render an
  append-only JSONL eval log as one sortable comparison table" (headers/
  widths/`print_row`, `--sort` choices). Reused directly by
  `benchmark/prompt_sweep_report.py`.

## Classifier Validation UI (`ui/classifier_validation_ui.py`, built 2026-07-30)

Developer QA tool for rapidly eyeballing Gemma's routing decisions after
a batch run - NOT part of the production pipeline, purely read-only
(never opens a bucket CSV for writing, never reruns classification).

- **Real bug, found and root-caused 2026-07-30** — the image preview
  grew to fill the whole screen, pushing the metadata panel and nav
  buttons off-screen. First fix attempt (defense-in-depth: `root.
  maxsize()`, a forced `update_idletasks()` before the first render, a
  clamp on the computed thumbnail box) treated the symptom without
  reproducing the actual cause — verified as insufficient by direct
  evidence: Jon supplied a screen recording; extracting frames from it
  with `cv2.VideoCapture` (no ffmpeg on this machine) showed the window
  GROWING CONTINUOUSLY over several seconds, not jumping once - the
  signature of a feedback loop, not a single bad reading. Root cause:
  `_render_image()` sizes the thumbnail from `image_frame.winfo_width()/
  height()`, but by DEFAULT a Tk `Frame`'s own size can be pulled upward
  by its child's (the image `Label`'s) natural size once an image is
  assigned - if the rendered image comes out even marginally larger than
  the frame's true allocated space (thumbnail/LANCZOS rounding is
  enough), the frame grows to fit it, which fires a NEW `<Configure>`
  with a BIGGER box, which re-renders bigger again - an unbounded loop
  with no floor. **Fix: `self.image_frame.pack_propagate(False)`**
  right after creating the frame - makes its size depend ONLY on what
  its OWN `pack()` call inside `root` gives it, never on its child's
  content, breaking the loop at the source rather than capping the
  damage. Stress-tested directly (not just "window stays on screen"):
  30 repeated re-renders of the same image left `image_frame`'s
  `winfo_width()/height()` byte-identical every time, and firing
  synthetic `<Configure>` events with escalating fake sizes (up to
  1920×1080) directly at the frame had zero effect on its real
  dimensions. The three original defense-in-depth measures were kept
  (real ceiling, sane first-render size, guard against a degenerate
  reading) - they weren't wrong, just not the actual cause.
- **Buckets come from `config/pipeline.yaml`, not code** - Jon stopped
  the build specifically to require this: "they should be stored in a
  config file so if we add a bucket later its a config change rather
  than code rewrite." `_bucket_names()` returns
  `load_pipeline_config()["buckets"].keys()` (reusing
  `core/classifier.py`'s own loader, not a re-parse) - verified against
  the real file: 8 names, `dense_tabular_rows` first,
  `uncertain_review` last, matching insertion order with no re-sorting.
  Never imports `DocumentCategory` or hardcodes a bucket string anywhere.
- `_load_bucket_rows(bucket_name)` reads `BUCKET_DIR /
  f"{bucket_name}.csv"` via plain `csv.DictReader` - schema-agnostic, so
  `uncertain_review.csv`'s extra `error` column needs no special-casing
  and no column list is redeclared from `core/classifier.py`'s
  `CSV_FIELDS`/`UNCERTAIN_FIELDS` to drift out of sync with.
- **Lazy image loading** - only the currently-displayed row's image is
  ever decoded; navigating a 183-row bucket (real count, `printed_
  document`) doesn't load 183 images up front. `Image.thumbnail()`
  scales to fit the image frame, preserving aspect ratio, never
  cropping, never upscaling past the source resolution - re-fit on
  window resize via a `<Configure>` binding on the image frame, guarded
  against redundant re-renders when the box size hasn't actually
  changed.
- **A missing/unreadable file shows an error placeholder instead of
  crashing** - verified directly with a real bad path swapped into a
  live row; reviewing hundreds of images should survive one bad one.
- **Session stats ("Reviewed: N / M")** - Jon's addition to the spec,
  built now rather than deferred: `self._visited: dict[str, set[int]]`
  tracks visited row INDICES per bucket, independently, preserved when
  you switch buckets and back (verified: switching away and returning
  does not lose or double-count prior progress). Deliberately the
  natural precursor to the described future evolution ("Correct 21 /
  Incorrect 2 / Remaining 48 / Accuracy 91.3%") - swapping "visited" for
  "has a recorded verdict" is the only change needed later, not a
  layout redesign.
- **`self.future_frame`** - a real, empty, correctly-positioned `Frame`
  (between the metadata panel and the nav row) reserved for the
  explicitly-deferred features (Correct/Incorrect, Reassign, Notes,
  Export, confidence colouring/filters) - an architectural placeholder,
  not a visible one, so today's window matches the spec's mockup exactly.
- Changing bucket resets to image 1 (index 0); Previous/Next (and
  bound `<Left>`/`<Right>`) clamp at the bucket's own bounds rather than
  wrapping, with both buttons correctly disabled at each end and both
  disabled together on an empty bucket.
- Verified with a full headless drive against REAL project data (no
  synthetic fixtures): real bucket counts (71/1/183/15/29/4/20/0),
  real image decode on first load, navigation + button-state clamping,
  visited-set accuracy across revisits and bucket switches, the real
  empty `uncertain_review` bucket, and a real broken-path failure.

## `core/classifier.py` --debug flag (added 2026-07-30)

Built while a real classification batch was actively running, so it
deliberately does NOT touch anything under `core/loaders/` — CLAUDE.md's
"ask before editing loader code while a run is in progress" is
specifically about that directory, and a mid-run edit there could affect
an already-loaded process depending on what's cached. Editing
`core/classifier.py` itself is safe regardless (Python doesn't hot-reload
a running process's already-imported source), but the design stayed
loader-free anyway.

- `_enable_raw_output_debug(loader)` — monkey-patches `loader.
  _run_generate` on the INSTANCE (not the class), wrapping it to print
  the raw model text to stdout before returning it unchanged for
  `classify()`'s normal parsing. Instance-attribute assignment doesn't
  go through Python's descriptor/auto-bind protocol, so the replacement
  function is called with exactly `(raw_image, prompt)` — no `self`
  needed in its signature.
- `run(manifest_path, debug=False)` calls it when `debug=True`, before
  `open_bucket_writers()`.
- CLI gained real `argparse` (`--debug` flag) replacing the old manual
  `len(sys.argv) != 2` check — `python -m core.classifier
  data/manifest.csv` still works identically; `--debug` is additive.
  Verified via `--help` (exits before any model import) and a plain
  `import core.classifier` (confirmed no model load at import time) -
  not via an actual classify() call, since the GPU was busy with a real
  run when this was built.

## Pipeline stage terminology (2026-08-02)

See `docs/PIPELINE_STAGE_TERMINOLOGY.md` for the canonical Stage 0-6
naming (Source Acquisition / Raw Sensor Capture / Decision Engine /
Image Processing / Validation Capture / Document Routing / Specialized
Extraction) - the entries below still use each module's original
naming ("Stage 0", "Stage A") since the in-code rename is a
still-in-progress, tracked rollout (see that doc's "Rollout status"
section), not yet reflected in every docstring.

## Vision-tower embedding infrastructure (`core/vision_embeddings.py`, added 2026-08-02)

The generic, stateless embedding functions originally written for
`benchmark/vision_encoder_qualification.py` (the frozen Vision
Qualification Battery v1.0) - `build_model_and_transform(timm_tag)`,
`embed_pooled(model, transform, pil_image)`, `embed_patch_mean(...)`
(handles both isotropic-ViT token-sequence and hierarchical-CNN/
windowed-transformer 4D-spatial-map pooling correctly, including the
NCHW-vs-NHWC and prefix-token-count bugs already found and fixed this
session), `cosine_sim(a, b)`, `_spatial_layout(...)`. Moved to core/ so
`core/baseline_embeddings.py` (a real production consumer, not a
benchmark script) doesn't have to import from `benchmark/` - same
reasoning as the PDF-conversion move. `benchmark/vision_encoder_
qualification.py` now imports these from here instead of defining them
locally; every OTHER benchmark script that imports from that module
(benchmark2_2/2_3/sequential_runtime/concurrency_falsification_test/
aspect_ratio_experiment/the pilot scripts) needed zero changes - they
all import via that module's path, which just re-exports now.

`QUALIFIED_ENCODERS` here is the canonical `(name, timm_tag)` list for
the 8 encoders qualified in the frozen battery - previously redefined
ad hoc in `benchmark2_sequential_runtime.py`'s `CANDIDATES` and
`benchmark2_3_multi_tower_routing_audit.py`'s `TOWERS` (both untouched,
still work exactly as before - this is for NEW callers, not a forced
migration).

## Pre-preprocessing baseline embeddings (`core/baseline_embeddings.py`, added 2026-08-02)

Captures a per-image vision-tower embedding (all 8 `QUALIFIED_ENCODERS`)
BEFORE `core/manifest_pipeline.py`'s deskew/autocontrast step runs -
per `docs/BENCHMARK2_METADATA_LAYER_QUALIFICATION.md`'s Third/Fourth
extension design (one stable, non-drifting reference embedding per
image) and Jon's 2026-08-02 direction to capture this datapoint before
preprocessing happens, specifically so it can eventually inform a
not-yet-built per-image preprocessing-profile decision (Stage B -
still NOT STARTED, see `docs/PREPROCESSING_STAGE_NOTES.md`) instead of
today's fixed default profile applied to every image regardless of
what it actually needs.

- `capture_baseline_embeddings(image_paths, encoders=QUALIFIED_ENCODERS)
  -> list[dict]` - loads each encoder ONCE and embeds every image with
  it (not the reverse - avoids reloading 8 models per image).
- `write_baseline_embeddings(image_paths, output_path=DEFAULT_BASELINE_PATH)`
  - append-only per image (latest capture wins for that image path, old
  entries for OTHER images are preserved, not overwritten) - persists to
  `data/baseline_embeddings.json`, one file for the whole corpus (NOT
  per-image sidecars - measured cost is 34.6MB / 16.8 min compute for
  the entire ~1500-image corpus at both baseline+postprocessing stages
  combined, all 8 encoders, so a single consolidated file is both
  simpler and nowhere near a size needing per-image files).
- Called from `core/manifest_pipeline.py`'s `build_working_manifest_
  from_paths()` via `capture_baseline: bool = True` (default on) -
  runs immediately after `copy_to_working_dir()`, before
  `preprocess_for_manifest()`. Lazy-imported inside that function (not
  at module top) so importing `core.manifest_pipeline` doesn't force
  every caller to pay torch/timm's import cost even when
  `capture_baseline=False`.
- **Real gotcha hit while testing this**: `write_baseline_embeddings`'s
  `output_path` parameter defaults to the module-level
  `DEFAULT_BASELINE_PATH` - reassigning that module attribute at
  runtime (`core.baseline_embeddings.DEFAULT_BASELINE_PATH = ...`) does
  NOT redirect an already-imported function's default (Python binds
  default argument values at function-definition time, not call time).
  Pass `output_path=` explicitly to redirect; don't try to monkeypatch
  the module constant.
- Schema per image record: `image` (path str), `image_hash` (sha256),
  `preprocessing_stage` (`"pre_preprocessing"` today - a second,
  post-preprocessing capture point is designed but not built, see the
  Fourth extension doc section), `timestamp`, `library_versions`
  (timm/torch), `embeddings.<encoder_name>.{checkpoint, vector}`.
- Full production corpus (1677 images) captured 2026-08-02.

## Source expansion layer (`core/source_expansion.py`, added 2026-08-02)

**Generic Stage 0 ingestion layer, not PDF-specific code bolted onto
the manifest builder.** Went through two design passes same-day: PDF
handling was first added directly into `core/manifest_pipeline.py`
(`_expand_pdf_sources()`), then generalized into this module per Jon's
explicit direction - "the objective is not simply to add PDF support -
the objective is to introduce a generic source expansion stage into
the manifest pipeline." `core/manifest_pipeline.py` now calls exactly
one function from this module (`expand_source_paths(source_paths)`)
and has ZERO knowledge of PDFs, output directories, DPI, or any other
expander-specific config - see this module's own docstring and
`core/manifest_pipeline.py`'s updated one for the full ingestion shape:
`Source Discovery -> Source Expansion -> Manifest Construction -> ...`.

- **`ExpandedSource`** (dataclass) - the canonical interchange object:
  `source_path` (original), `expanded_path` (the real image path
  downstream consumes), `source_type` ("image"/"pdf"/future tag),
  `page_number` (1-indexed, `None` for a plain image), `metadata: dict`
  (open-ended, format-specific info a future expander wants to attach
  without ever needing a new dataclass field for it). A plain image is
  a degenerate one-to-one expansion (`source_path == expanded_path`),
  not a special case callers branch on.
- **`SourceExpander`** (ABC) + **`PdfExpander`** - each expander is a
  self-contained object that declares its own `extensions` and owns
  ALL of its own configuration internally (`PdfExpander.__init__(self,
  output_dir=..., dpi=...)` - these never appear as parameters
  anywhere in `core/manifest_pipeline.py`). `register_expander(instance)`
  populates the module-level `SOURCE_EXPANDERS` dict; `EXPANDABLE_EXTENSIONS`
  is derived from it (`collect_image_paths()` walks `IMAGE_EXTENSIONS |
  EXPANDABLE_EXTENSIONS`). Adding a future format (multipage TIFF,
  JPEG2000, a ZIP archive) means one new `SourceExpander` subclass + one
  `register_expander()` call - zero changes anywhere else.
- **`core/pdf_conversion.py`** - the actual PDF rasterization logic
  (PyMuPDF/fitz, `convert_pdf(pdf_path, out_dir, dpi=300, ext="png",
  force=False) -> list[Path]`, idempotent - skips an existing rendered
  page unless `force=True`) plus `PDF_EXTENSIONS`/`DEFAULT_PDF_OUTPUT_DIR`
  (PDF-specific constants deliberately kept out of the generic modules).
  Moved here from `scripts/convert_pdf_to_image.py` so core modules
  don't import from `scripts/` (this project's convention is the
  reverse - scripts are thin CLI adapters over core engines).
  `scripts/convert_pdf_to_image.py` is now a thin wrapper over this for
  standalone manual conversion outside the Stage 0 flow.
- **Provenance sidecar**: `build_working_manifest_from_paths()` writes
  `<manifest_stem>_provenance.json` alongside the manifest CSV (e.g.
  `data/manifest_provenance.json`) - every `ExpandedSource` as a dict
  plus its final `working_path`. The manifest CSV itself stays exactly
  the single `file_path` column `core/classifier.py` already expects -
  provenance is additive, for diagnostics/UI/future traceability, never
  consumed by anything downstream today.
- `ui/build_manifest_ui.py`'s "Add File" dialog and "Add Folder" walk
  both pick up PDFs automatically (`_IMAGE_FILETYPES` includes a PDF
  pattern, imported from `core.pdf_conversion` not `core.manifest_
  pipeline`; "Add Folder" already delegated to `collect_image_paths()`
  so it needed no changes at all).
- Verified end-to-end with a synthetic single-page test PDF through
  `build_working_manifest()` (throwaway working_dir/manifest_path, not
  touching production `data/manifest.csv`/`data/working`) after EACH
  design pass, not just the final one - not just assumed to work from
  reading the code.

## Build Manifest UI (`ui/build_manifest_ui.py`, built 2026-07-30)

A Pipeline Launcher front end — the window shows exactly ONE stage's
view at a time, swapping forward on an explicit user action, not a
wizard and not one window that accumulates every stage's controls
forever. **Contains zero manifest-building/preprocessing/classification
logic itself** — every real decision lives in the real modules it shells
out to (`core/manifest_pipeline.py`, `scripts/run_preprocessing.py`,
`core/classifier.py`), imported/invoked unmodified.

`scripts/build_manifest.py` (the ORIGINAL engine this UI was built
against) is now RETIRED and ARCHIVED (`scripts/archive/build_manifest.py`,
moved 2026-07-30 via `git mv` after a full dependency audit found zero
live references anywhere in the repo) — not imported anywhere in this
file. See "scripts/build_manifest.py retired" below for the full story. This
entry describes the current, post-retirement state only; earlier
single-stage/pre-retirement design notes were consolidated out of this
entry (still in git history if needed) rather than left contradicting
the current description.

- **Model**: `self.paths: dict[Path, None]`, keyed on `Path.resolve()` —
  a single ordered dict gives dedup (key lookup) + stable insertion order
  + O(1) lookup all at once in Python 3.7+, replacing the set+list pair
  the original spec sketched (written for a language without ordered
  dicts). Re-adding an already-queued file is a no-op that does NOT move
  its position.
- **List widget**: `ttk.Treeview`, `selectmode="extended"`, **iid = the
  resolved path string itself** (verified safe directly — no reserved
  characters, no separate id↔Path lookup table needed). Right-click opens
  a `Menu` (currently one entry, "Remove") — the deliberate extension
  point for future per-file actions (Reveal in Explorer, Open, Retry,
  Mark ignored) without a layout change.
- **No popups for routine dedup** (Jon's explicit direction) — a status
  line above the list reports the outcome of the last add
  ("142 files added (5 already present)"); `messagebox` is reserved for
  real problems (empty queue on Build click, GPU-busy confirmation).
- **Canvas-swap Pipeline Launcher, three views, IMPLEMENTED 2026-07-30**
  — `manifest_frame` → `preprocessing_frame` → `classification_frame`,
  exactly one visible at a time via `_show_frame()`, under a persistent
  header (title, description, Power User checkbox). The button that
  LAUNCHES the next stage always lives on the CURRENT view, not the
  next one (Jon's direct instruction on the swap mechanics).
  - **`manifest_frame`**: Add File/Add Folder/file list/"Build Manifest
    →". Clicking it is the swap trigger to Preprocessing.
  - **`preprocessing_frame`**: status/progress/timing/log for
    `scripts/run_preprocessing.py` (Stage 0, CPU-only, `device="CPU"` so
    no GPU dialog ever fires), and — once it finishes successfully —
    "Run Classification →", the swap trigger to Classification.
  - **`classification_frame`**: status/progress/log for
    `core/classifier.py`, and — once it finishes successfully — the
    Routing Summary (`routing_frame`, all 8 `DocumentCategory` bucket
    counts from `_bucket_counts()`) IN PLACE as a child of this frame.
    No "proceed to next stage" button yet — no defined next workflow
    exists (see `project_pipeline_launcher_view_architecture` memory).
  - **WIDGET CONSTRUCTION IS TWO-PHASE**: all three views' widgets
    (including each stage's own log/status/progress widgets) are built
    FIRST with no button commands wired, THEN `_wire_stages()`
    constructs the `Stage` objects referencing those already-real
    widgets and wires every `.config(command=...)`. An earlier version
    tried building each `Stage` inline as its owning view was
    constructed and had to create-then-destroy placeholder widgets to
    make the ordering work — replaced because it was fragile, not
    because it was broken.
  - **`Stage` now owns its own log widgets** (`log_toggle_btn`,
    `log_frame`, `log_text`, `log_expanded`) — necessary once a SECOND
    subprocess-running view (Preprocessing) existed; a single shared
    `self.log_text` from the Classification-only design stopped making
    sense. `_toggle_log`/`_set_log_expanded` now take the `stage` they
    apply to.
  - **The swap is gated on `on_launch`, not the button click itself** —
    `_run_named_stage(stage, on_launch=..., on_done=...)` only fires
    `on_launch` AFTER any GPU-busy confirmation is accepted (Classification
    only; Preprocessing has no such gate). Verified directly: declining
    the GPU dialog leaves the canvas on the CURRENT view rather than
    stranding the user on a view with no way to retry.
  - Content updating in place (a summary appearing) is explicitly NOT
    the same as the canvas swapping. No "back" affordance between
    views — forward-only, matching "not a wizard".

- **Device label / timing / ETA, ADDED 2026-07-30** (Jon's QOL request,
  built once the GPU was clear again) — `Stage.uses_gpu: bool` was
  REPLACED with `Stage.device: str` ("CPU", "GPU (RTX 5060 Ti)", ...)
  per Jon's explicit refinement: a free-form label "leaves room for
  future flexibility" (Apple MPS, DirectML, Remote API) without the UI
  changing shape later. `_detect_gpu_name()` queries
  `nvidia-smi --query-gpu=name` ONCE at `_wire_stages()` time for the
  real card name (verified: `"GPU (NVIDIA GeForce RTX 5060 Ti)"` on this
  dev machine) — display only, never a capability check, falls back to
  a bare `"GPU"` on failure. The GPU-busy gate now keys off
  `stage.device != "CPU"` instead of a separate bool.
  - `Stage` gained `timing_var: StringVar`, `start_monotonic: float |
    None`, `start_wall: str | None`, `last_progress: tuple[int, int] |
    None`. `_update_timing(stage, include_eta=True)` renders
    `"Device: ... · Started HH:MM:SS · Elapsed M:SS"` plus, once at
    least one progress line has arrived, `"· ETA ~M:SS remaining"` — a
    plain straight-line `(elapsed / done-so-far) * remaining` estimate.
    Called every ~100ms poll tick (elapsed visibly counts up between log
    lines, not just when one arrives) and once more on `"done"`/`"error"`
    with `include_eta=False` so the frozen final line doesn't show a
    stale ETA next to "Done". `_build_stage_status_block()` grew a third
    label for it, shared by both stage views.
  - **Real bug found and fixed while testing this**: a synthetic
    multi-line fake-command test showed progress/ETA never updating
    live — `bufsize=1` on `Popen` only governs how the PARENT reads the
    pipe, not whether the CHILD process buffers before writing, and
    CPython fully block-buffers stdout whenever it isn't a real TTY
    (always true for a subprocess pipe). Without a fix, a child script's
    `print()` calls could sit unflushed until the OS pipe buffer fills
    or the process exits — silently defeating live progress for EVERY
    stage, not just the new timing feature, though it likely went
    unnoticed earlier since short/bursty test runs looked the same
    either way. Fixed by passing `env=dict(os.environ,
    PYTHONUNBUFFERED="1")` to `Popen` in `_stage_worker()`. Re-verified
    after the fix: ETA now genuinely appears mid-run and disappears on
    the frozen final display, across multiple live ticks in a timed test.
  - `classification_stage.supports_debug` flipped `False` → `True` now
    that `core/classifier.py` has a real `--debug` flag (see below) —
    the Power User checkbox now does something on Classification, not
    just Preprocessing (which still has none).
  - Verified with a full headless drive, including a REAL (not faked)
    Preprocessing subprocess run via `scripts/run_preprocessing.py`
    against SAFE SCRATCH paths, confirming real deskew output, correct
    progress-line parsing, and correct working-manifest shape.

- **`scripts/build_manifest.py` retired from this UI, then ARCHIVED to
  `scripts/archive/build_manifest.py` project-wide, 2026-07-30** — Jon:
  "build_manifest was the initial starting module. its legacy now and
  wont be used going forward." A full dependency audit (imports,
  subprocess launches, docs, CI — none exists in this repo) found and
  fixed the one live reference (`debug_tools/workflow_gui.py`'s Tools
  tab launcher, migrated to `scripts/build_working_manifest.py`) before
  the file was moved via `git mv` (history preserved). The UI no longer
  imports it at all; `collect_image_paths`/`IMAGE_EXTENSIONS` now come from
  `core/manifest_pipeline.py` (identical definitions, confirmed) instead.

- **`scripts/run_preprocessing.py`, NEW 2026-07-30** — the CSV-input
  adapter `core/manifest_pipeline.py`'s own docstring said could be
  "added the same way whenever something needs it." Reads a `file_path`
  CSV column via `core/bucket_worklist.py`'s `load_bucket_filepaths()`
  (the project-wide standard reader, not a bespoke format) and calls
  `build_working_manifest_from_paths()`. Independent of
  `scripts/build_working_manifest.py` (the folder-walking CLI) — a
  second adapter over the same engine, not a replacement.
  `ui/build_manifest_ui.py`'s `_build_preprocessing_command()` writes
  the UI's queued paths to a THROWAWAY temp CSV (not a durable pipeline
  artefact, deleted after the run) as this script's input — there is
  now only ONE real manifest artefact this UI produces,
  `data/manifest.csv` via `core/manifest_pipeline.py`'s
  `DEFAULT_MANIFEST_PATH`, resolving the earlier "Manifest means two
  things" ambiguity by retiring the writer that caused it.

- **TEST-SAFETY LESSON, learned the hard way 2026-07-30** — a headless
  test that launches a REAL subprocess must pass safe/scratch paths
  as actual CLI ARGUMENTS to that subprocess. Monkeypatching a Python
  module attribute (e.g. `core.manifest_pipeline.DEFAULT_MANIFEST_PATH`)
  in the TEST'S OWN process has ZERO effect on a child subprocess, which
  re-imports that module fresh in its own interpreter and sees only the
  real on-disk defaults. A test that did this actually overwrote the
  real `data/manifest.csv` (114 real file paths → 1 test path) and
  created a stray file in the real `data/working/` — caught immediately
  via `git status`/`git diff`, fully recovered via `git checkout --
  data/manifest.csv` (the file was tracked, so HEAD had the real
  content) plus manually deleting the one stray working-copy. The
  correct pattern for testing UI code that launches a subprocess
  targeting real default paths with no override: swap
  `stage.build_base_command` to a harmless fake command (e.g.
  `["python", "-c", "print(...)"]`) to test the LAUNCH/PROGRESS/REVEAL
  machinery, and verify the real underlying script separately, in
  isolation, with explicit scratch-path CLI arguments (already how
  `scripts/run_preprocessing.py` itself was verified). Never assume a
  parent-process patch reaches a child process.

## GUI building blocks (tkinter — the whole repo's GUI convention, no PyQt/web)

- **`debug_tools/workflow_gui.py`**:
  - `CommandTab` — the queue+thread+`after()`-polling pattern for
    running a long CLI subprocess without freezing the UI:
    `threading.Thread(target=self._run_worker, ...)`, worker pushes
    `("line", text)`/`("done", exit_code)`/`("error", msg)` into a
    `queue.Queue`, `self.frame.after(100, self._poll_log_queue)` drains
    it on the Tk main loop. Reused (adapted, not subclassed — this class
    is tab-system-coupled) by `benchmark/prompt_sweep_gui.py`.
  - `ScoringTab` — the in-process (no subprocess) pattern for a cheap
    computation + sortable `ttk.Treeview` + headline-stats label grid:
    click a column header to sort (`_sort_by`, re-binds itself with
    `reverse` flipped each click).
  - `_open_folder(path)` — cross-platform (`os.startfile` / `open` /
    `xdg-open`) "open this folder in Explorer/Finder" helper.
- **`scripts/model_assessment.py`**: `list_model_profiles()` (sorted
  `config/models/*.yaml` stems), `list_prompt_files()`,
  `list_bucket_profiles()`, `load_bucket_profile()`,
  `OVERRIDABLE_FIELDS`. Isolation contract worth copying for any new
  dev tool: never imports `core/classifier.py`/`core/extractor.py`,
  never writes to `config/pipeline.yaml`/bucket CSVs/the manifest.
- **Gotcha (cost real debugging time 2026-07-29)**: `tkinter.Frame`'s
  CONSTRUCTOR does not accept a tuple for `padx`/`pady` (raises
  `_tkinter.TclError: bad screen distance`) — tuple padding
  (`padx=(10, 0)`) is only valid on `.pack()`/`.grid()` calls, not the
  widget constructor itself.
- **Headless-testing a tkinter app**: `messagebox.show*` opens a REAL
  modal dialog that blocks forever with no one to click it — stub
  `tkinter.messagebox.showwarning`/`showerror`/`showinfo` before running
  any automated check that might hit a validation path. A full
  integration test (click a button programmatically, then pump the
  event loop with repeated `root.update()` calls in a loop instead of
  `root.mainloop()`) can drive a real subprocess + real widgets end to
  end without a human present.

## Batch model/prompt benchmarking (`benchmark/prompt_sweep*.py`, built 2026-07-29)

Answers "which model+prompt combo extracts this best?" (vs. the pipeline,
which just answers "can we extract this at all?"). See
`project_prompt_sweep_tool` memory for the full history/decisions.

- `benchmark/prompt_sweep.py` — engine + CLI. `discover_examples()`
  (ground-truth-backed examples from `ground_truth_log.jsonl`),
  `run_sweep()` (two-phase: load OCR model once → every OCR-prompt
  variant × every example; load structuring model once → every combo ×
  every example), `_emit(event, **kwargs)` — prints `"@@PROGRESS <json>"`
  lines (plan/model_load_start/stage1_step/combo_done/etc.) alongside
  normal output, for the GUI to parse without scraping human-readable text.
  `_load_prompt_variants(..., glob_pattern=...)` — when given
  `--ocr-prompt-dir`/`--structuring-prompt-dir`, only globs
  `ocr_stage1_*.txt` / `structuring_stage2_*.txt` by default (2026-07-30
  fix, both flags override this) - `config/prompts/` is a flat folder
  holding every pipeline stage's prompts, not a per-stage subfolder, so
  pointing a sweep straight at it used to also sweep `classifier_*`/
  `extractor_*`/`loader_*` files that aren't candidates for either stage.
- `benchmark/prompt_sweep_report.py` — renders the full
  `data/outputs/prompt_sweep_log.jsonl` history as one sortable table.
- `benchmark/prompt_sweep_gui.py` — tkinter picker wrapping the CLI as a
  subprocess (`CommandTab`-style), live queue/current-run panels driven
  by the `@@PROGRESS` events, sortable report `Treeview` on completion.

## Image analysis / preprocessing decisions (`core/image_analysis.py`, built 2026-07-29)

**OpenCV 5.0.0 IS available** on the main interpreter (`C:\Python314`,
which is what `core/` runs on — the `.venv_*` dirs are per-model
subprocess-loader envs only). Earlier CV work here assumed PIL+numpy
only, which is why `core/image_preprocessing.py` hand-writes `clahe()`
and `adaptive_threshold()` and why auto-dewarp was recorded as
intractable. Don't re-derive numpy implementations of things cv2 does in
one call. Note `cv2.HoughLinesP` returns `(N, 4)` in 5.x, not `(N, 1, 4)`
as in 4.x — reshape rather than unpacking a fixed shape.

- `core/image_analysis.py` — **Stage A: pure measurement, modifies
  nothing, decides nothing.** `analyze_image(image_or_path) ->
  ImageAnalysis`. Emits NUMBERS not categories on purpose — binning
  cutoffs are policy and live in Stage B, so policy can be retuned
  without re-measuring. `None` means NOT MEASURABLE, not zero and not OK.
  `save_analysis`/`load_analysis`/`analysis_sidecar_path` (sibling
  `<name>_analysis.json`), `analyze_manifest(csv) -> rolled-up CSV`.
  CLI: `python -m core.image_analysis <image|manifest.csv>`.
- **Two-tier schema** (Jon's design, 2026-07-29 — the corpus stopped being
  homogeneous once receipts/letters/microfilm joined census+manifests):
  geometry-invariant fields at the top level (`aspect_ratio`,
  `deskew_angle_deg`, `blur_laplacian_var`, `noise_residual_std`,
  ruling-line counts + angles), and ROI-dependent fields in per-region
  `RegionStats` blocks under `.regions` — `"frame"` always, plus
  `"page"`/`"table"` when detected. Otsu threshold, ink fraction,
  polarity, stroke width, text height, illumination and contrast are ALL
  region-scoped, because they are meaningless without saying which region.
- **Two ROI detectors, no document-specific assumptions**:
  `page_boundary`/`page_confidence`/`page_method` and
  `table_boundary`/`table_confidence`. Both always run; **Stage B picks**
  (census/manifest → table is the ROI; letter/receipt → page is the ROI).
  Confidences are ordinal signals, NOT probabilities.
  `measure_region()` is public so a future better detector can get the
  same measurement block without reimplementing it.
- **The ROI split is empirically validated, not just tidy**: frame-level
  polarity flagged 46/218 microfilm pages as inverted (two inspected, both
  wrong); page-level polarity flagged **0/218**. Scoping the region fixed
  the measurement.
- **First real calibration — `table_confidence` separates real tables**
  (measured 2026-07-29): the 16 labelled census pages score min 0.375,
  median 1.0; the 218 microfilm pages (letters/receipts, mostly no tables)
  score median 0.125, p90 0.25, with only 13/218 reaching the census
  minimum. So a confidence floor around ~0.375 is a defensible starting
  point for "is there really a table" — but Stage B MUST gate on it: the
  detector returns a boundary for 174/218 microfilm pages, nearly all
  spurious low-confidence boxes spanning ~92% of the frame.
- Its `page_boundary` / `ruling_lines_*` / `dominant_*_angle_deg` fields
  are deliberately shared with the auto-dewarp cascade (Tier 1 wants the
  quad, Tier 2 the line geometry) so the two don't drift apart.
- **Known gaps, do not mistake for working**: (1) the `bright_region`
  page-detection fallback returned the white "PUBLIC ARCHIVES" label board
  instead of the document on `oocihm.lac_reel_c10639.733.jpg` — "paper is
  the brightest large thing" fails on aged mid-grey microfilm paper;
  confidence is capped at 0.55 but that is not a fix. (2)
  `estimate_deskew_angle` returns **exactly 0.00 on ~90% of the microfilm
  corpus and 15/16 census pages** — correct for the pre-deskewed census
  files, but a silent zero (indistinguishable from "genuinely straight")
  on microfilm, where `dominant_vertical_angle_deg` independently measures
  ~1° median. Prefer the ruling-line angle as the skew signal there.
- Its ruling-line counts use morphological opening and are **NOT
  comparable** to `document_classification.py`'s
  `_count_vertical_ruling_lines()` unbroken-run counts — never feed them
  to a template's `column_count_range`.

### Microfilm corpus finding — crop before you trust tone (measured 2026-07-29)

Ran the analyser over all 218 jpgs in
`J:\Screenshots\Knott_Ancestry\Archive Microfilms` (LAC microfilm reel
scans, `oocihm.lac_reel_*` — provenance completely separate from the
`e0019466xx` census batch). 218/218 measured, 0 failures. Findings:

- **The unexposed black film surround dominates 215/218 frames** (median
  88% of all ink in one connected component, max 99%). It saturates
  `contrast_p5_p95_spread` (med 217/255), inflates
  `illumination_unevenness` (med 0.65 vs 0.12 on a cropped census page),
  and breaks Otsu: the threshold splits PAPER from SURROUND instead of
  INK from PAPER. Consequence — `otsu_threshold`, `ink_fraction` and
  `ink_is_dark_on_light` are all unreliable on uncropped frames. 43 pages
  were flagged inverted-polarity; the two most extreme were inspected and
  **both were ordinary dark-on-light documents**. Applying `invert` off
  that flag would have damaged them. Gate any tone-driven policy on
  `largest_ink_blob_frac`.
- **This makes cropping a prerequisite for the analyser, not a
  convenience.** Stage A's tone fields only become meaningful on a
  page-cropped image, so the boundary-crop stage has to run first — the
  two stages are ordered, not independent.
- **Real skew exists here, unlike the labelled batch**: median dominant
  vertical angle 1.04°, p90 2.88°, max 12.57°; horizontal angles span
  -9.7° to +10.0°. So "there is no warp in this project's images" is true
  of the `e0019466xx` batch ONLY — do not generalize it.
- **Text is small**: median `text_height_px` 11px (analysis-space, i.e.
  longest side 1600), 137/218 under 12px → upscaling is broadly relevant.
- Aspect ratios span 0.42–1.60 (med 0.84) and `page_quad` was found on
  only 122/218 with median area fraction 0.56 — the page genuinely IS
  about half the frame on these, so a low area fraction here is often
  correct rather than a detection failure.

## Manual-classification queue (`run_batch_auto_sidecar.py`, 2026-07-30)

Every vision stage in this project has a fallback/quarantine, and the goal
is to reduce manual touches — so a page the CV path cannot be trusted on is
now routed OUT to a human instead of silently guessed.

- `AutoSidecarResult.used_cv_fallback: bool` (`core/auto_sidecar.py`) —
  True when no `doc_type_override` was passed and the CV classifier picked
  the template itself. Also mirrored at `diagnostics["used_cv_fallback"]`,
  and prepends a loud warning to `result.warnings`. **A True here means the
  template choice was a GUESS** — see that field's docstring for the
  measured evidence (2/5 on fresh LAC scans, 13–43 line spread within a
  single census year).
- `scripts/run_batch_auto_sidecar.py` now writes a second CSV beside the
  run summary: **`<out-dir>/needs_manual_classification.csv`**, the
  worklist for the (not-yet-built) manual classifier. Columns:
  `file_path, image_path, stem, reason, gemma_doc_type, gemma_confidence,
  cv_guess, cv_confidence, manual_doc_type`.
  - `image_path` is the DEWARPED image when one exists (what a UI should
    display), falling back to the source; `file_path` is always the source.
  - `manual_doc_type` is written deliberately EMPTY — it is the column the
    human/UI fills in, so the same file serves as queue and completed
    record. Nothing in the pipeline writes to it. Valid values are in
    `KNOWN_DOC_TYPES` in that script.
  - `reason` is written verbatim so a reviewer sees WHY the page reached
    them. Four routes in: upstream `unknown`/empty doc_type, no sidecar
    produced, CV-fallback-chosen template (sidecar IS still written — the
    page is queued for confirmation, not discarded), and hard errors.
  - Always written, header-only when empty, so a downstream tool can read
    it unconditionally.
  - Verified 2026-07-30 on the real 16-page batch: 13 sidecars generated,
    3 pages queued (the same 3 `unknown`s that prompt v3 fixes).

## Column registration fix — IMPLEMENTED 2026-07-30, evidence-driven cascade

`locate_columns()` in `core/auto_sidecar.py` resolves each column EDGE
(not each column - the two edges can land in different tiers) through a
4-tier cascade, evidence-driven rather than fixed-precedence (Jon's
design, refined twice during the same session - see the function's own
"RESOLUTION CASCADE" docstring for the full reasoning):

1. **`measured`** - a real ruling line was found by the per-edge search
   AND it agrees with the page's own registration affine (or no affine
   exists to check it against).
2. **`affine_override`** / **`affine_predicted`** - a page-level
   scale+shift affine (`_fit_registration_affine`, `cv2.fitLine` with
   `cv2.DIST_HUBER` - robust, not ordinary least squares, since a
   "confident" match is only the closest candidate in its search window,
   not a correctness guarantee) fitted from every confidently-matched
   edge in the table. `_override` when a found edge DISAGREED with the
   affine beyond an adaptive tolerance (`_affine_residual_tolerance` -
   3x the affine's own median fit residual, floored at 3px) and got
   overruled; `_predicted` when no line was found at all.
3. **`locally_interpolated`** - no page-level affine (too few/too
   clustered confident points), but this edge's expected position falls
   BETWEEN two confidently-matched edges elsewhere in the table
   (possibly including this same column's OTHER edge) -
   `_interpolate_from_neighbors()`, bracketed linear interpolation only,
   never extrapolates past the last known-good anchor on either side.
4. **`template`** - none of the above; the raw template fraction, same
   safety net every other boundary in this module has.

**The sanity-check tier (2) is doing real work, not just theory**:
measured 2026-07-30, 5 real `affine_override` cases on the 9 reference
pages, and inspecting them shows the mechanism catching a genuine failure
mode - e.g. `1931_174-e011707164`'s "Age" column had `raw_search=[1330,
1330]`, BOTH edges snapped to the same physical line (an actual
degenerate collapse the old code accepted silently), corrected via the
affine to `[1330, 1381]`.

**Falls back to the exact original per-edge-only behavior, byte for
byte, whenever no affine can be fit** - no regression risk on pages
where a fit can't be trusted.

Measured on the 9 real manually-masked pages (correct doc_type per page,
not the CV-fallback guess): mean column-edge error **2.85% → 1.94%**,
median **1.80% → 0.78%** (essentially unchanged from the simpler
always-prefer-affine version tried first in the same session, but now
with a mechanism-level fix for the same-line-collapse failure mode, and
honest per-tier provenance instead of a single blanket "affine" label).
`locally_interpolated` fired 0 times on this sample set - checked, not a
bug: the two candidate pages' confident points are spatially clustered on
one side of the table (e.g. `e001926997`'s only sit below x≈1112, while
every un-found column sits at x≈1465-2686), so there's genuinely no
bracket to interpolate from and it correctly refuses to extrapolate.
`z000017634` (the pre-existing, documented aspect-ratio outlier) is
unaffected, as expected.

`locate_columns()`'s diagnostics dict gained a `_registration_affine` key
(`{used, scale, shift, confident_points, residual_tolerance_px}`)
alongside the per-column entries - leading underscore so callers
iterating column names (e.g. `flagged_columns` lookup in
`generate_auto_sidecar()`) don't mistake it for one. Existing fields
(`expected`, `refined`, `width_ratio`, `width_implausible`,
`left_from_ruling_line`, `right_from_ruling_line`) are unchanged in name
and meaning; `refined` is the FINAL chosen [x0, x1]. **`source` is now a
2-element list `[x0_source, x1_source]`, not a single string** (each
edge's own tier) - anything reading a prior single-string `source` field
needs updating. `raw_search` still records the uncorrected per-edge
search result for debugging.

## Auto-sidecar column location — why it under-detects (measured 2026-07-30)

Investigating "rows and columns aren't detected fully on census forms"
(`core/auto_sidecar.py`). Ground truth used: the 9 manual sidecars in
`data/outputs/row_segmentation/` that carry real `mask_keep_ranges`
(43 comparable column instances). **Rows are largely fine; columns are
the problem.** All numbers below are mean absolute column-edge error as a
percentage of table width.

- **Table boundary works, column dividers don't**: table edges refine
  against a real ruling line 30/32 = **94%** of the time; column edges only
  69/230 = **30%**. When `_find_vertical_ruling_line()` fails it silently
  falls back to the template `x_frac`, so ~70% of column positions are one
  calibration sample's proportions rescaled, not measured.
- **Tuning the search does NOT help accuracy.** Swept ratio × radius:
  hit rate rises 30% → 60% (ratio 0.25 + radius 3%), but accuracy is FLAT
  (median 1.36–1.89%, mean 2.83–3.00%, max always 17.5% on the known
  `z000017634/Sex` outlier). Finding more lines just finds more chances to
  snap to a neighbouring divider.
- **The refinement contributes ~nothing at all**: template `x_frac` with NO
  refinement scores median 1.80% / mean 2.90%; the current refined output
  scores median 1.80% / mean 2.85%. Where refinement fires (21/43) it
  improves the mean by 0.11pp and makes 3 cases WORSE. Note the template
  baseline is optimistically biased (templates were calibrated from some of
  these same sidecars), which makes this worse than it looks.
- **ROOT CAUSE — it is a table-ANCHOR registration error, not column
  drift.** (Corrected 2026-07-30, same day: an earlier version of this
  entry concluded "real column positions move between pages", which the
  affine test below disproves for 1911/1921. Jon's hypothesis — "if the
  table edge can be detected the columns should land near where they
  actually are" — is the correct one.) Fitting a single global
  scale+shift between two pages' normalized column positions collapses the
  apparent spread almost entirely:

  | doc type | raw mean | after affine fit | fitted scale / shift |
  |---|---|---|---|
  | 1911 | 4.33pp | **0.03pp** (max 0.04) | 1.109 / -6.82pp |
  | 1921 | 4.73pp | **0.91pp** (max 2.26) | 1.010 / +4.40pp |
  | 1931 | 7.19pp | 3.50pp | 0.831 / -2.33pp |
  | 1931 | 7.62pp | 5.31pp | 0.791 / -1.61pp |

  1911's residual of 0.03pp means the columns are **rigid relative to each
  other** — the template fractions are already essentially exact, and 100%
  of the error is a wrong per-page scale+offset. The detected table span is
  what's inconsistent: measured table width runs 0.940–0.997 of page width
  across samples, and `left/right_from_ruling_line` is sometimes True/False
  on the same edge across pages, i.e. the locator is snapping to
  DIFFERENT physical lines (outer page border vs first inner rule) on
  different pages. Every column inherits that error, scaled.
- **Therefore the high-leverage fix is registration, not column search**:
  anchor columns to a 2-point affine derived from reliably-detected
  structure (two confident dividers, or a properly consistent table span)
  instead of the raw table bbox. Threshold/radius tuning is proved useless
  above, and rebuilding column detection is not needed for 1911/1921 —
  their fractions are already right once registered. Not implemented;
  needs Jon's go-ahead since it changes a deliberately locked-in function.
- **The "1931 residual" was a CLASSIFICATION bug, not bad masks**
  (resolved 2026-07-30; an earlier version of this entry wrongly suspected
  the manual masks). `http___data2...1911_jpg_e001946617.jpg_dewarped` and
  `e078_e001946617_dewarped` are two copies of the SAME page — the header
  reads "FIFTH CENSUS OF CANADA, 1911" — but they classify as 1931 and 1911
  respectively. Regrouping by TRUE year, all three real 1911 pages fit each
  other to **0.03–0.04pp** after an affine fit, including the misclassified
  one. The masks were never the problem.
- **Why it misclassifies — `vertical_ruling_lines` tracks scan SHARPNESS,
  not just layout.** The two copies are tonally near-identical (luminance
  205 vs 207, Otsu 148 vs 150, ink fraction 0.174 both) but differ in
  acutance: `blur_laplacian_var` 9793 vs 14775, stroke width 2.74px vs
  1.91px. In the softer copy faint sub-rules fade below threshold and break
  up; in the sharper copy they survive as long unbroken runs. Count goes
  21 → 39, which crosses the entire 1911 → 1931 decision boundary, since
  that count is `classify_document()`'s PRIMARY year discriminator.
- **A more robust counter already exists**: on the same two copies,
  `core/image_analysis.py`'s morphological-opening ruling counter moves only
  10v → 12v where the unbroken-run counter moves 21 → 39. An opening
  tolerates the small breaks that sharpening/softening introduce. Worth
  considering as the classifier's discriminator — but re-calibrating every
  template's `column_count_range` against it is required, NOT optional (see
  `document_classification.py`'s own threshold-history warning).
- `z000017634` remains a genuine outlier: 5.31pp residual even within true
  1931, aspect ratio 1.238 vs 1.589 for the other 1931 page, and the lowest
  classification confidence in the set (0.615). Already documented as an
  outlier in `canada_census_1911.yaml`.
- **Suspected data problem worth eyeballing**: 1931's `Sex` (12.99–31.59%)
  and `Relationship to Head` (16.72–28.45%) OVERLAP across samples, which
  looks like the manual masks were placed on different physical columns on
  different pages. Any calibration derived from those masks inherits the
  inconsistency.
- **Python gotcha, live in this file**:
  `_find_vertical_ruling_line(..., run_ratio_threshold=_RULING_LINE_RUN_RATIO)`
  binds its default at DEFINITION time, so editing the module-level
  `_RULING_LINE_RUN_RATIO` (line 83) does **not** change the column search.
  Anyone retuning that constant will see no effect. Also note this file uses
  0.5 while `document_classification.py` uses 0.35 for the same measurement
  and documents why 0.5 was wrong — that fix never propagated here.
- Unrelated-but-verified: a page reporting 0 rows (e.g. `e001946619`) is NOT
  a crash — it is `_quarantine_whole_page()` moving all 50 rows to
  `rows_needs_review` on an ambiguous `table_top`. Working as designed.
- Row-1 refinement falls back to the theoretical row height on 5/16 pages,
  and always in the SAME direction (measures 41–43px vs 55–60px expected,
  ~30% short) — a systematic bias suggesting it snaps to a sub-row feature
  (ruled sub-line or text baseline), not random noise. Not investigated.

## Dewarp ground truth — the labelled corner set (measured 2026-07-29)

`data/outputs/dewarped/` holds **16** `*_dewarped.json` sidecars with
human-placed 4-corner quads (`save_dewarp_sidecar` format), all 16
`source_file_path`s still present in `data/working/`. The other ~40
`.jpg` outputs there predate sidecars and have **no** labels. This is
the benchmark for any auto-corner-detection work — don't rebuild it.

**The measured finding that reframes auto-dewarp**: those human
corrections are 2 pure axis-aligned crops + 14 quads whose keystone
deviation is only **0.12%–1.08% of page width** (max 74.6px on ~7000px
pages). The manual dewarp step is therefore doing ~95% *cropping* and
almost no perspective correction on this corpus. That also explains
`core/warp_detection.py`'s two failed calibrations: it was hunting a
sub-1%-of-width signal, i.e. an essentially absent one, not using a
flawed technique. Baseline for comparison: `image_analysis.py`'s Canny/
`approxPolyDP` Tier 1 quad detector scores 14/16 detected, **median
corner error 1.95% of width, worst 4.69%**, with errors clustering
bimodally (~1.2% and ~4.5% groups — a systematic wrong-boundary pick,
not noise). Read that against the 1% warp being corrected before
concluding anything about Tier 1's usefulness.

## Known conventions worth not re-learning

- **Windows console encoding**: `sys.stdout.reconfigure(encoding="utf-8",
  errors="replace")` near the top of any module that might print raw
  model output — cp1252 (Windows default) can't represent arbitrary
  Unicode a model might hallucinate, and this has caused real mid-run
  crashes twice (`core/row_extraction.py`, `training/test_lora_checkpoint.py`).
- **Abstention convention mismatch** (source of a real scoring bug, see
  `_is_correct` above): LoRA training targets use the literal words
  `"blank"`/`"illegible"`; LIVE extraction prompts use an empty string
  for blank and literal `"?"` for illegible. Don't compare one
  convention's output against the other's expectation directly.
- **`CLAUDE.md`'s GPU-check gate** — `nvidia-smi` before editing loader
  code or launching any inference, every session, not just once.
- **`_release_model()` frees allocated memory but not reserved memory,
  once real generation has actually run** (confirmed via isolated repro,
  2026-07-30 - see `project_prompt_sweep_tool` memory for the full
  investigation): `torch.cuda.memory_allocated()` correctly drops to
  ~0 after release (the null-out + `gc.collect()` + `empty_cache()` in
  `_release_model()` works exactly as intended at the Python level) -
  but `torch.cuda.memory_reserved()` stays pinned at the released
  model's peak, because PyTorch's caching allocator doesn't return
  fragmented-by-generation memory blocks to the driver, even though
  nothing is still allocated within them. A bare load-then-release with
  NO inference in between releases perfectly clean (reserved drops to
  0 too) - this only shows up once `_run_generate()`/`generate()` has
  actually executed, which is why a quick "does release work" smoke
  test without real calls can miss it entirely. Practically bounded,
  not a runaway leak: the next model's allocation gets reused INSIDE
  the still-reserved pool rather than growing beyond it, as long as
  the next model's real need fits inside the previous model's peak -
  confirmed by watching `qwen3vl2b` (needs 3.99GB) load into a 9.51GB
  pool left behind by `gemma_extract` without reserved climbing to
  13.5GB. Could still matter on a card with less headroom, or in a
  sweep where a LATER model needs more than an EARLIER model's peak
  reserved - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is an
  untested-but-plausible mitigation if it ever becomes a real problem,
  not yet applied since it hasn't caused an actual failure.
  `nvidia-smi`'s reading (system-wide, not per-process, on WDDM) will
  look consistent with this the whole time a Phase-A-then-B sweep runs
  - don't mistake it for a different model still being resident.
