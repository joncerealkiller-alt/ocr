# Genealogy Extraction Pipeline

Custom vision-model pipeline for archival document classification and
extraction, built to replace VectorDB-Plugin's vision tab. Structural
fixes for fabrication/loop failure modes found in extensive prior
testing: per-model tuned generation configs, bounded/confidence-tagged
output schema, human review gate before extraction, isolated model
assessment tool for A/B testing before committing a model to a bucket.

**Keep this file updated whenever a new dependency is added.**

---

## Dependencies

### Python version
Python 3.10+ (uses modern type hints like `list[str]`, `dict[str, Any]`)

### Install commands

Run these in order. GPU/CUDA build for `torch` must be installed
*before* anything that depends on it, and must match your actual CUDA
driver version — check with `nvidia-smi` (top-right corner shows
supported CUDA version) before picking the index URL below.

```bash
# 1. Core data/validation libraries
pip install pydantic pyyaml

# 2. Image handling
pip install pillow

# 3. PyTorch — CUDA build, NOT the default CPU-only build.
#    Confirm your CUDA version with `nvidia-smi` first, then match
#    the index URL below (cu130 = CUDA 13.0; adjust if different).
#    A CPU-only torch install (`torch==X.X.X+cpu`) will silently work
#    but run everything on CPU with near-zero GPU usage — this is a
#    known failure mode we hit during setup, always verify after
#    installing (see "Verifying your install" below).
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 4. Hugging Face Transformers + device mapping support
pip install transformers accelerate

# 5. Qwen-specific vision preprocessing (required by core/loaders/qwen_loader.py)
pip install qwen-vl-utils

# 6. Gemma-4's image processor needs torchvision's v2 transforms
#    (already installed in step 3, listed here as a reminder of why
#    it's required — Gemma4ImageProcessor imports from
#    torchvision.transforms.v2)

# 7. Required for PixtralLoader's 4-bit quantization (BitsAndBytesConfig)
#    - confirmed via a real test (2026-07-14): older bitsandbytes
#    versions fail here, 0.46.1+ required.
pip install -U "bitsandbytes>=0.46.1"

# 8. Required for ChandraLoader (core/loaders/chandra_loader.py) - not a
#    plain transformers call, uses this package's own generate_hf()/
#    BatchInputItem/parse_markdown API internally.
pip install "chandra-ocr[hf]"

# 9. Required for OlmOcrLoader (core/loaders/olmocr_loader.py) - supplies
#    the RL-trained prompt builder (build_no_anchoring_v4_yaml_prompt)
#    that loader deliberately always uses, ignoring any configured
#    prompt file.
pip install olmocr
```

### tkinter (no install needed, but required)
`build_manifest.py`, `review_uncertain.py`, and `model_assessment.py`
all use `tkinter` for their UI (folder picker / review queue / test
harness). It ships with standard Python on Windows and macOS. On
Linux, if missing: `sudo apt install python3-tk` (Debian/Ubuntu) or
equivalent for your distro.

---

## Verifying your install

**Always run this after installing/reinstalling torch**, before
running any pipeline script:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count()); print('Torch version:', torch.__version__)"
```

Expected output on a working GPU setup:
```
CUDA available: True
Device count: 1
Torch version: 2.x.x+cu130   (or whatever CUDA tag you installed)
```

If `CUDA available: False` or the version string ends in `+cpu`,
torch installed the CPU-only build — reinstall per step 3 above with
the correct index URL for your CUDA version.

---

## Full dependency list (for requirements.txt / quick reference)

```
pydantic
pyyaml
pillow
torch          # install via CUDA index URL, not plain `pip install torch`
torchvision    # same
transformers
accelerate
qwen-vl-utils
bitsandbytes>=0.46.1   # PixtralLoader's 4-bit quantization - older
                         # versions fail here (confirmed 2026-07-14)
chandra-ocr[hf]         # ChandraLoader only - own generate_hf() API,
                         # not a plain transformers call
olmocr                  # OlmOcrLoader only - supplies its RL-trained
                         # prompt builder
```

(`tkinter` not listed — stdlib, not pip-installable)

---

## Pipeline stages & scripts

Run in this order:

1. **`build_manifest.py`**
   Folder picker → walks folder for image files → writes
   `data/manifest.csv`.

2. **`core/classifier.py`**
   ```
   python -m core.classifier data/manifest.csv
   ```
   Runs the configured classifier model (currently Gemma) over every
   image in the manifest, routes each into one of 8 bucket CSVs under
   `data/buckets/`. Never transcribes content — routing decision only.

3. **`review_uncertain.py`**
   ```
   python review_uncertain.py
   ```
   Human review queue for anything the classifier routed to
   `uncertain_review.csv`. Reassigns to the correct bucket, logs every
   correction to `data/outputs/reviewed_uncertain.csv` (permanent
   record, never deleted). Must be run until the queue is empty —
   `core/extractor.py` will refuse to run otherwise.

4. **`core/extractor.py`**
   ```
   python -m core.extractor
   ```
   Per bucket, runs the model/prompt configured in
   `config/pipeline.yaml`, validates every result against the schema
   in `core/schema.py`, runs anomaly checks (n-gram overlap,
   length-outlier, repeated-entry), writes:
   - `data/outputs/extracted.csv` — successful records
   - `data/outputs/failed_extraction.csv` — parse/schema/OOM failures,
     or buckets skipped due to a missing prompt file
   - `data/outputs/anomaly_flags.csv` — flagged (not dropped) records

### Separate tool — does not touch pipeline data

**`model_assessment.py`**
```
python model_assessment.py
```
Isolated single-image testing: pick image, bucket, prompt, model
profile, optionally override any generation setting. Never writes to
bucket CSVs, `extracted.csv`, or `pipeline.yaml` — only writes JSON
reports to `data/outputs/model_assessments/`. Use this to A/B test
model/prompt/settings combinations before assigning a model to a
bucket in `config/pipeline.yaml`.

---

## Config files

- `config/pipeline.yaml` — bucket → model/prompt routing table,
  anomaly check settings, audit sampling rate
- `config/models/*.yaml` — per-model generation settings
  (temperature, repetition_penalty, resolution ceilings,
  vram_headroom_gb, cpu_offload_limit_gb, reasoning toggle, etc.)
- `config/prompts/*.txt` — extraction/classification prompts, one per
  bucket type plus the classifier prompt

---

## Row-level segmentation & extraction pipeline (separate from the stages above)

A second pipeline, for dense tabular records (census pages etc.) where
whole-document extraction isn't the right unit of work - isolates and
extracts ONE ROW, and within a row optionally ONE COLUMN, at a time.
Does not touch `data/manifest.csv`, bucket CSVs, or anything from the
document-classification pipeline above.

1. **`row_segmentation_ui.py`**
   ```
   python row_segmentation_ui.py
   ```
   Interactive tkinter tool: load a page image, deskew, detect row
   bands (periodic or uniform-tile mode), confirm table/header/
   metadata bounds, then mask individual columns by clicking to keep
   just that column's x-range (everything else painted white on the
   crop sent downstream - fixes cross-column contamination that a
   plain left/right crop boundary can't, since a wanted column can sit
   between two unwanted ones).

   **Persistent per-image sidecar** (2026-07-21/22 redesign): one
   `<name>_sidecar.json` per source image holds page geometry (rows,
   table/header/metadata bboxes) PLUS a `columns` dict keyed by column
   name, each with its own mask, extraction results, and status
   (`pending`/`in_progress`/`done`), plus `active_column` and
   `progress`. Saves are atomic merges (`core.row_segmentation.
   update_sidecar` - temp file + `os.replace`), never a blind
   overwrite, so masking or extracting one column can't destroy
   another's saved work. "Load columns file..." loads the full
   ordered column list (same one-name-per-line format as
   `run_row_extraction.py`'s columns.txt); "Mark column done → Next"
   and "Extract active column" both auto-advance to the next pending
   column and restore its previously-saved mask if one exists - no
   manual remasking between columns. Reopening an image with an
   existing sidecar resumes exactly where it left off (column
   progress, masks, active column) without re-running row detection.
   The preview draws every defined column's mask simultaneously in a
   stable per-column color, with the active column highlighted.

2. **`run_row_extraction.py`**
   ```
   # Single-column mode (matches the UI's persistent-sidecar workflow) -
   # extracts the sidecar's active_column (or --column NAME) using that
   # column's own stored mask, writes results into the sidecar itself:
   python run_row_extraction.py <sidecar.json> --model qwen3vl2b [--column NAME]

   # Legacy multi-column mode - all columns in one prompt per row,
   # against the sidecar's old single global mask, output to CSV/JSON
   # only (not written back into the sidecar):
   python run_row_extraction.py <sidecar.json> <columns.txt> --model qwen3vl2b \
       [--header-fields <fields.txt>]
   ```
   Core logic in `core/row_extraction.py` (`run_single_column_extraction`,
   `run_row_extraction`, `parse_row_output`, prompt builders). Column
   names supplied explicitly (never auto-OCR'd from the header) -
   header text is often dense/bilingual, exactly the kind of thing
   this project's evidence says needs human confirmation.

3. **`ground_truth_labeling_ui.py`**
   ```
   python ground_truth_labeling_ui.py
   ```
   Human-verified per-field labeling, walking a sidecar row by row,
   field by field, showing the EXACT same crop (same mask) the model
   sees. Output: `data/outputs/ground_truth_log.jsonl` (append-only,
   resumable, skips already-labeled fields on reload). Explicitly
   captures genuine illegibility/blankness as correct answers, not
   failures to push past - this is a finding aid, not a source of
   truth.

4. **`export_lora_dataset.py`**
   ```
   python export_lora_dataset.py [--log-file PATH] [--out-dir PATH] \
       [--min-examples-per-status N]
   ```
   Converts `ground_truth_log.jsonl` into image+target-text LoRA
   training pairs (`train.jsonl` + per-row crop images + a coverage
   report). Targets are literal single-field values (including the
   literal strings `"illegible"`/`"blank"`), NOT the pipe-delimited
   `value|confidence` structuring convention used elsewhere - the
   first LoRA's job is cursive recognition and honest abstention, not
   confident-looking structured output.

---

## Change log

Keep brief entries here when dependencies or major structure change,
so it's clear why a version was pinned/changed later.

- Fixed a serious bug affecting both real extraction AND LoRA training
  data: apply_column_mask() only PAINTS outside a kept column range
  white, it never narrows the image - so every single-column
  extraction and every exported training image was still the FULL row
  width (~3800px on a real census page), with real content in as
  little as ~9% of it. Worse in export_lora_dataset.py specifically:
  it was reading the sidecar's legacy top-level mask fields (empty
  under the per-column architecture), so masking was never applied to
  exported images AT ALL - confirmed via a real export where two
  different columns' images for the same row were byte-identical,
  meaning the same image was being paired with contradictory target
  texts depending on which column a label came from. Added
  core.row_segmentation.tight_crop_to_ranges() (opt-in, backward
  compatible - existing callers see zero behavior change) and wired it
  into run_single_column_extraction and export_lora_dataset.py, with
  configurable padding (--tight-crop-padding-px / --tight-crop-
  padding-pct) and a regression test (test_tight_crop.py). Sidecar row
  geometry is untouched - only the in-memory image actually sent to
  the model/exporter is tightened. Extraction results now record
  columns[name].extraction_meta.tight_crop_applied so pre-fix and
  post-fix results are distinguishable: the FIRST smolvlm2 quality
  test batch (Name/Age columns, 2026-07-22 early session) predates
  this fix and should be treated as baseline-only, not compared
  directly against later results without checking this flag

- Initial scaffold: pydantic schema, base loader, Gemma classifier
- Added Qwen extraction loader (initially 2B, switched to 3B —
  official model card only exists for 3B/other sizes at time of
  writing; 2B is architecturally identical but had no dedicated card)
- Corrected Qwen loader to official pattern: `Qwen2_5_VLForConditionalGeneration`
  + `qwen_vl_utils.process_vision_info()`, replacing initial manual
  chat-template string construction
- Added `genealogy_chart` as an 8th classification bucket after
  family-tree-diagram images were misrouted to `portrait_photo`
- Added VRAM headroom reservation + CPU-offload fallback + per-image
  OOM retry, so a single oversized image can't crash a full batch run
- Added execution-mode logging (device_map, max_memory, oom_recovered)
  to every extraction result and assessment report, for traceability
- **Discovered torch was installed as CPU-only build (`+cpu`), causing
  near-zero GPU usage despite `device_map="auto"` — always verify with
  the CUDA check above after any torch install/reinstall**
- Added PixtralLoader, GotOcr2Loader (stage-1 extraction candidates) —
  requires `bitsandbytes>=0.46.1` (older versions fail on Pixtral's
  4-bit quantization, confirmed via a real test 2026-07-14)
- Added ChandraLoader, OlmOcrLoader — each needs its own package
  (`chandra-ocr[hf]`, `olmocr`) beyond plain transformers; both were
  missing from this README until 2026-07-14 despite being added
  earlier — real documentation gap, not new as of this entry
- Added `ground_truth_labeling_ui.py` (human-verified per-field
  labeling, walking a sidecar row/column, honest illegible/blank
  capture) and `export_lora_dataset.py` (converts that log into
  image+literal-text LoRA training pairs — deliberately NOT the
  pipe-delimited value|confidence convention, to avoid un-teaching
  abstention behavior)
- Redesigned the row-segmentation sidecar from one-mask-per-file
  (`row_segmentation_ui.py`'s Save overwrote the whole sidecar on
  every pass, requiring manual remask + resave per column) to a
  persistent per-image state file: `core/row_segmentation.py` gained
  `update_sidecar`/`init_column_state`/`advance_column` (atomic merge
  writes, per-column masks/results/status, never a blind overwrite);
  `row_segmentation_ui.py` gained per-column mask tracking (stable
  color per column in the preview), a loaded column list, auto-advance
  with mask restoration, a background-threaded "Extract active column"
  action, and resume-on-reopen without re-running row detection;
  `core/row_extraction.py` gained `run_single_column_extraction`,
  which writes results into the sidecar itself instead of only a
  separate CSV; `run_row_extraction.py`'s CLI gained this as its
  default mode, columns.txt still selects the original legacy
  multi-column mode
