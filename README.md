# Genealogy Extraction Pipeline

Custom vision-model pipeline for archival document classification and
extraction, built to replace VectorDB-Plugin's vision tab. Structural
fixes for fabrication/loop failure modes found in extensive prior
testing: per-model tuned generation configs, bounded/confidence-tagged
output schema, human review gate before extraction, isolated model
assessment tool for A/B testing before committing a model to a bucket.

**Keep this file updated whenever a new dependency is added.**

---

## Quick start

```bash
python main.py
```

`main.py` is the one command to know regardless of what's actually
running underneath it. Right now it's a minimal stub that launches
`debug_tools/workflow_gui.py` (the current dev/debug workspace, not a
polished end-user UI yet) - once the real UI exists, `main.py` becomes
its actual entry point instead, so this command never has to change
again. See "Project layout" below for what lives where, and "Pipeline
stages & scripts" for running individual steps directly instead of
through the GUI.

## Project layout

Reorganized 2026-07-25 (was previously a flat directory of scripts at
repo root) so future modules get built directly inside their intended
folder instead of needing another repo-wide move later:

```
main.py            ← application entry point (see "Quick start")
README.md
CLAUDE.md

config/            ← pipeline.yaml, per-model YAML, prompt .txt files
core/              ← the reusable framework: loaders, schema, row
                       segmentation/extraction, classifier, debug_dump
data/              ← manifest.csv, buckets/, outputs/, and
                       debug_model_inputs/ (--debug-model-inputs captures)
ui/                ← real interactive tools a user runs to do
                       segmentation/labeling work (row_segmentation_ui.py,
                       ground_truth_labeling_ui.py)
debug_tools/       ← dev/debug workspace, not the end-user app
                       (workflow_gui.py, review_uncertain.py)
benchmark/         ← accuracy scoring against ground truth
                       (score_two_stage_against_ground_truth.py)
training/          ← LoRA dataset export + training
                       (export_lora_dataset.py, train_lora.py)
scripts/           ← operational CLI tools, not part of the reusable
                       framework (build_manifest.py, run_row_extraction.py,
                       run_two_stage_extraction.py, model_assessment.py)
diagnostics/       ← standalone manual diagnostic scripts, not pytest
                       unit tests (test_row_segmentation.py,
                       test_native_prompt.py, test_tight_crop.py,
                       test_stage2_isolated.py)
```

Every script that computes its own project root from `__file__` (or
imports `core.*`/`benchmark.*` directly) was updated as part of this
move, not just relocated - see the "Repo-root reorganization" changelog
entry below for exactly what that involved and why a plain file move
wasn't safe here.

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

### Subprocess-backed loaders — a model whose deps conflict with the main env

Some models' `trust_remote_code` pins a `transformers` version that
conflicts with this project's main environment and can't be resolved by
picking one version for everything — e.g. `vikhyatk/moondream2` only
produces correct output on `transformers==4.52.4` (its own declared
version); on this project's main `transformers` (5.12.1), generation
silently degenerates to garbage (`"1"` followed by hundreds of
blank-token lines), confirmed by a direct A/B test, not a guess. A real
inventory of other candidate models' own `config.json` files confirmed
this isn't a one-off: moondream3-preview wants yet another version
(4.51.1), and HunyuanOCR's own subfolders disagree with each other.
Since two `transformers` versions can't coexist in one Python process,
these models run in their own dedicated venv instead, driven by
`core/loaders/subprocess_loader_base.py` (`SubprocessLoaderBase`) — the
reusable venv-resolution / spawn+READY-handshake / JSON-I/O / cleanup
plumbing shared by every such loader, paired with
`core/loaders/_subprocess_worker_common.py` (`run_worker()`) on the
worker-script side. A new subprocess-backed loader only needs to supply
`VENV_DIR`, `WORKER_SCRIPT`, and a worker script that knows how to load
and call that specific model — not reimplement the plumbing.

`MoondreamLoader` (`core/loaders/moondream_loader.py`) is the first
(and currently only) real user of this pattern — it doesn't load the
model itself, it spawns `core/loaders/_moondream_worker.py` under
`.venv_moondream` and talks to it over stdin/stdout JSON. Create that
venv once, from the project root:

```bash
# --system-site-packages reuses this environment's already-installed
# torch/CUDA build instead of re-downloading it (multi-GB torch+cuda
# wheel) — do not omit this flag.
python -m venv --system-site-packages .venv_moondream
.venv_moondream/Scripts/python.exe -m pip install transformers==4.52.4
```

`.venv_moondream/` is local environment state, not source — don't
commit it, and don't add it to `requirements.txt`. If `MoondreamLoader`
can't find `.venv_moondream/Scripts/python.exe` (or `bin/python` on
non-Windows), it raises with this same setup command rather than
silently falling back to the main environment's (broken, for this
model) transformers.

Model weights: `vikhyatk/moondream2` (revision pinned in
`config/models/moondream2.yaml`) must already be present in the normal
Hugging Face hub cache (`~/.cache/huggingface/hub/`,
`models--vikhyatk--moondream2/...`) — the loader always passes
`local_files_only=True`, so it will not attempt an on-demand download
even from inside the worker venv, and instead should be moved/downloaded
into the hub cache ahead of time. This is also a deliberate cutoff of
moondream2's LoRA-variant-download feature (a bare `urlopen()` against
`api.moondream.ai` in the model's own remote code, only reachable if a
`variant`/`settings` kwarg is ever passed to `query()`) — both the
loader and the worker never pass that kwarg, and the worker additionally
monkeypatches the relevant function to raise instead of opening a
socket, so this stays blocked even if that changes accidentally later.

Subprocess cleanup is deterministic, not left to garbage collection —
`BaseLoader.release()` (default no-op, zero behavior change for every
in-process loader) is called explicitly by both `core/row_extraction.py`
and `model_assessment.py`'s own `_release_model()` helpers, before the
next model loads. `SubprocessLoaderBase.release()` terminates the
worker (`terminate()`, 5s wait, `kill()` if it doesn't exit) — needed
because a worker subprocess holds its own separate CUDA context in a
different venv/process entirely, invisible to the parent process's own
`torch.cuda.empty_cache()`. Without this, moondream2's worker could
still be resident in VRAM at the exact moment a second model tried to
load, undermining `run_two_stage_extraction()`'s documented guarantee
of never having two models resident simultaneously. `__del__` remains
as a last-resort safety net only.

`config/models/moondream2.yaml` has `reasoning_enabled: true` (flipped
from `false`) — `query()`'s own documented default (per moondream3-
preview's README, same API lineage) is `reasoning=True`, recommending
`False` only as a speed/cost optimization for trivial questions; reading
ambiguous handwritten cursive isn't trivial by that framing, so this
project had been running the opposite of the model's own recommendation
for exactly the task that needs the most help. Untested in practice as
of this writing — flip back if it doesn't help. Also added
`config/prompts/ocr_stage1_moondream_query.txt`, phrased as a direct
question ("What does the text in this image say?") rather than an
imperative instruction — `query()`'s documented usage is short direct
questions, and a real stage-2 test showed moondream2 echoing a long
multi-rule instruction block back verbatim instead of following it (a
task-shape mismatch, not a wording problem every other `ocr_stage1_*.txt`
file's imperative phrasing doesn't share with this model).

### tkinter (no install needed, but required)
`scripts/build_manifest.py`, `debug_tools/review_uncertain.py`, and
`scripts/model_assessment.py` all use `tkinter` for their UI (folder
picker / review queue / test harness). It ships with standard Python
on Windows and macOS. On Linux, if missing: `sudo apt install
python3-tk` (Debian/Ubuntu) or equivalent for your distro.

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

MoondreamLoader is deliberately NOT in this list — it runs entirely
inside `.venv_moondream`, a separate venv with `transformers==4.52.4`
pinned (see "MoondreamLoader — separate venv required" above), not
inside the main environment described by this list.

(`tkinter` not listed — stdlib, not pip-installable)

---

## Pipeline stages & scripts

Run in this order:

1. **`scripts/build_manifest.py`**
   Folder picker → walks folder for image files → writes
   `data/manifest.csv`.

2. **`core/classifier.py`**
   ```
   python -m core.classifier data/manifest.csv
   ```
   Runs the configured classifier model (currently Gemma) over every
   image in the manifest, routes each into one of 8 bucket CSVs under
   `data/buckets/`. Never transcribes content — routing decision only.

3. **`debug_tools/review_uncertain.py`**
   ```
   python debug_tools/review_uncertain.py
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

**`scripts/model_assessment.py`**
```
python scripts/model_assessment.py
```
Isolated single-image testing: pick image, bucket, prompt, model
profile, optionally override any generation setting. Never writes to
bucket CSVs, `extracted.csv`, or `pipeline.yaml` — only writes JSON
reports to `data/outputs/model_assessments/`. Use this to A/B test
model/prompt/settings combinations before assigning a model to a
bucket in `config/pipeline.yaml`.

---

## Constrained decoding — preventing off-script token fallback (`restrict_output_charset`)

Real evidence motivating this (2026-07-24): multiple test runs produced
garbage output containing unexpected-script characters under low visual
confidence — e.g. a genuine Cyrillic fragment ("Малы", "Туанилова")
where English handwriting was expected (qwen3vl4b, saved in
`debug_model_inputs/20260724T054950676627Z/row_0006/column_05_stage1/`).
`core/row_extraction.py`'s existing `_scrub_example_leakage()` (in
`parse_row_output()`) is a **post-hoc** safety net — it can only detect
already-generated bad output and force it to `?`/unclear, never recover
a correct answer the model could have produced instead. Constrained
decoding is complementary, not a replacement: it masks generation
**at the token level**, forbidding the model from ever picking a token
whose decoded text falls outside an expected character set, so it's
forced to pick its best answer from what's actually plausible rather
than falling back to a high-probability-but-wrong-script token.

`core/loaders/constrained_decoding.py`'s `AllowedCharsLogitsProcessor`
(a real `transformers.LogitsProcessor`) does this: given a tokenizer,
it precomputes (once, cached per tokenizer — see below) a boolean mask
over the full vocabulary, then sets every disallowed token's logits to
`-inf` before each sampling step. **Not bare ASCII** — this project's
data is Western/European-origin genealogical records where accented
Latin names ("François", "Québec") are routine, not edge cases, so the
default allowed set is the full Latin alphabet (bare + accented),
ASCII digits, and this project's own output-format punctuation
(`: | ? , . -` quotes, whitespace, plus `{}[]` for OlmOcrLoader's
genuine JSON and FlorenceLoader's `json.dumps()`-wrapped output).
Cyrillic/CJK/Arabic/Hebrew/etc. are excluded entirely — that's what
actually blocks the observed failure mode. **Special tokens (EOS/BOS/
PAD/every id in `tokenizer.all_special_ids`) are always allowed**
regardless of decoded text — masking EOS in particular would make
generation unable to ever stop.

Opt-in per model, `config/models/*.yaml`: `restrict_output_charset:
false` (default, same pattern as `reasoning_enabled` — a deliberate
behavior change, not silent). Wired via a shared
`BaseLoader._maybe_add_charset_logits_processor()` helper every
qualifying loader calls right before its own `model.generate()` call.

**Scope — confirmed by a real audit of every `core/loaders/*.py`
file's own `_run_generate()`, not assumed:**
- **Qualifies (11):** FlorenceLoader, GemmaLoader, GlmOcrLoader,
  GotOcr2Loader, GraniteVisionLoader, InternVLLoader, OlmOcrLoader,
  PixtralLoader, Qwen3VLLoader, QwenLoader, SmolVLM2Loader — all call
  `self.model.generate(**inputs, **gen_kwargs)` directly, a real
  `logits_processor=` kwarg path. (Fixed a real gap found while wiring
  this: `GemmaLoader`/`QwenLoader` never set `self.tokenizer` at all,
  unlike every other loader — brought in line, not special-cased.)
- **Does not qualify — ChandraLoader:** calls the external
  `chandra-ocr` package's own `generate_hf(batch, self.model)` — no
  `logits_processor` kwarg exposed at that call site.
- **Does not qualify (for now) — MoondreamLoader:** runs in a separate
  venv via `SubprocessLoaderBase`; its public `query()`/`caption()` API
  takes no token-level parameters. **Confirmed possible if ever
  needed**, not assumed impossible: moondream2 runs fully in-process
  (not a cloud API — that's a different model, moondream3-preview,
  which this project doesn't use) via a hand-rolled decode loop
  (`MoondreamModel._generate_answer()`) that already masks specific
  token ids to `-inf` before sampling — a charset mask would be a
  small, mechanically similar addition. Not built because it would
  require monkeypatching a **private, undocumented** method inside
  moondream2's own remote code (less stable than every other loader's
  public, versioned `logits_processor=` kwarg), and no real off-script
  failure has actually been observed from moondream2 specifically so
  far (the observed cases were other VLMs).

Recorded in `extraction_meta` (`run_single_column_extraction`'s sidecar
patch) as `restrict_output_charset`, same distinguishability principle
as `tight_crop_applied` — results with and without it active stay
comparable/auditable later. Also part of `GenerationConfig.
content_hash()`'s payload for document-level `ExtractionResult`s.

**Verified, not assumed** (2026-07-24): mask correctness spot-checked
against two real tokenizers — SmolVLM2's (49,280 tokens: `?`/digits/
`:`/`|`/letters/`{}[]`/accented Latin `é ç ñ ü É` all correctly
allowed; Cyrillic `М а л ы` and CJK `中 文` all correctly excluded;
every real special token — `<|im_start|>`, `<end_of_utterance>`,
`<image>`, etc. — correctly allowed regardless) and Qwen3-VL-4B's
(151,669 tokens). **Reproduced the actual saved Cyrillic-fallback
incident**: same source crop, same prompt, same model
(qwen3vl4b), `restrict_output_charset=True` — output changed from
containing repeated Cyrillic ("Туанилова") to zero Cyrillic characters
(regex-verified). The model's final guess still wasn't the correct
answer either way (ground truth for that field is "Manitoba"; it
guessed "Hamiltonova" with the mask on) and it's still rambling
through multiple hallucinated attempts — this mechanism fixes the
specific off-script failure mode it targets, not general answer
quality or verbosity, which remain separate, unsolved problems.
**Cost, measured honestly**: mask computation is a real one-time cost
(tens of thousands of `tokenizer.decode()` calls) — 0.55s for
SmolVLM2's ~49K-token vocab, 1.41s for Qwen3-VL-4B's ~152K-token vocab
— cached per tokenizer (keyed by `name_or_path` + vocab length, so
reused across separate loader instances loading the same model repo,
e.g. two-stage extraction's sequential stage 1/stage 2 loads) so this
cost is paid once, not once per generation call. Per-step cost once
the mask exists is a single `masked_fill` over the vocab dimension —
negligible next to one forward pass of a multi-billion-parameter model.

**Known caveats, not silently glossed over:** FlorenceLoader's
`<OCR_WITH_REGION>` task emits location tokens (`<loc_123>`) containing
`<`/`>`/`_`, none of which are in the default allowed set — this would
break bbox-preserving output under that specific task token (plain
`<OCR>`, this loader's actual documented role in this project, is
unaffected).

**Real bug found and fixed on first live use (2026-07-24):**
`len(tokenizer)` and the model's actual `lm_head` output width are not
always equal — a live InternVL run crashed every single generation call
with `"The size of tensor a (151675) must match the size of tensor b
(151674)"` the moment this flag was turned on. `AllowedCharsLogitsProcessor.
__call__()` only handled the mask being SMALLER than `scores` (some
models pad the lm_head to a rounder number for hardware efficiency —
handled by padding the mask with `True`), not the mask being LARGER
(InternVL's real case: `len(tokenizer)` is 151675, the model's actual
output dimension is 151674 — a genuine tokenizer/model vocab-size
mismatch, not hardware padding). Fixed by also truncating the mask down
to `scores.shape[-1]` when it's the larger side — safe, not a data-loss
workaround, since the model literally cannot produce logits for a token
id past its own output layer's width regardless of what the mask says.
Verified against the real InternVL3-2B tokenizer (confirmed `len() ==
151675`, reproducing the exact reported numbers) — generation now
completes cleanly instead of crashing.

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

1. **`ui/row_segmentation_ui.py`**
   ```
   python ui/row_segmentation_ui.py
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

2. **`scripts/run_row_extraction.py`**
   ```
   # Single-column mode (matches the UI's persistent-sidecar workflow) -
   # extracts the sidecar's active_column (or --column NAME) using that
   # column's own stored mask, writes results into the sidecar itself:
   python scripts/run_row_extraction.py <sidecar.json> --model qwen3vl2b [--column NAME]

   # Legacy multi-column mode - all columns in one prompt per row,
   # against the sidecar's old single global mask, output to CSV/JSON
   # only (not written back into the sidecar):
   python scripts/run_row_extraction.py <sidecar.json> <columns.txt> --model qwen3vl2b \
       [--header-fields <fields.txt>]
   ```
   Core logic in `core/row_extraction.py` (`run_single_column_extraction`,
   `run_row_extraction`, `parse_row_output`, prompt builders). Column
   names supplied explicitly (never auto-OCR'd from the header) -
   header text is often dense/bilingual, exactly the kind of thing
   this project's evidence says needs human confirmation.

   **Resolution fix (2026-07-23)** — tight-cropped single-column images
   (see the tight-crop fix below) could be as small as **79×36 pixels**
   in absolute terms (a real measured "Sex" column) — nowhere near
   enough detail for a vision encoder regardless of model capability.
   Some earlier model-comparison results from this project may partly
   reflect this resolution starvation rather than genuine
   capability differences, since it went unchecked until now.
   `core/row_segmentation.py` gained `upscale_to_target_height()` —
   aspect-preserving LANCZOS upscale to a target height (never shrinks
   an already-tall-enough crop), with `max_width` capping runaway
   aspect ratios on very wide/short masks. Wired into
   `crop_region_from_source()` and `_extract_region()`, with a new
   `--upscale-target-height` / `--upscale-max-width` CLI flag pair on
   `run_row_extraction.py` and `export_lora_dataset.py`. **Default is
   ON (target height 160px)** for `run_single_column_extraction` and
   `export_lora_dataset.py` (deliberately matched, so LoRA training
   images reflect the same input distribution real extraction actually
   sends the model). Every extraction result's `extraction_meta`
   records whether upscaling was applied and at what settings, same
   distinguishability principle as `tight_crop_applied` — results from
   before this fix should be treated as a resolution-starved baseline,
   not compared directly against later results without checking this
   flag.

3. **`scripts/run_two_stage_extraction.py`**
   ```
   python scripts/run_two_stage_extraction.py <sidecar.json> <columns.txt> \
       --ocr-model chandra --structure-model qwen3vl4b [--max-rows N]
   ```
   Stage 1 (a fixed-task or general OCR model) reads each row; stage 2
   (an instruction-following model) structures that reading, plus the
   row image, into the actual columns.

   **Field-level stage 1 redesign (2026-07-24)** — fixes a real bug
   `--debug-model-inputs` caught directly: stage 1 previously ran ONE
   OCR call per row against the FULL row crop (e.g. 3155×38px) with
   unwanted columns painted white, asking the model to find the wanted
   columns itself inside that wide, mostly-blank image — confirmed via
   a debug capture showing `original_crop_dimensions ==
   final_crop_dimensions` (no cropping/upscaling had run at all) and a
   `model_input.png` only a few dozen pixels tall. Stage 1 now makes
   **one OCR call per selected column**, each against that column's own
   tightly-cropped, independently-upscaled field image — the same crop
   `run_single_column_extraction` sends the model for one column, and
   the same crop `ground_truth_labeling_ui.py` shows a human labeler.
   The per-field readings are combined (one line per column) into the
   text stage 2 receives. **Stage 2's own image is now ALSO tightened**
   (a second instance of the exact same bug, caught and fixed
   immediately after the first when a real debug capture showed the
   same `original_crop_dimensions == final_crop_dimensions` signature
   for `row_NNNN/stage2` too) — it used to only PAINT unwanted columns
   white via `__multi__` without ever narrowing the image, so it was
   still the full row width regardless of how few columns were
   selected. It now crops to a padded box around the UNION of every
   selected column's own range, built from the SAME validated
   per-column masks stage 1 uses (not `__multi__` at all anymore, for
   either stage).

   **Important — each column now needs its OWN saved mask**, not just
   membership in the `__multi__` multi-select: `sidecar["columns"]
   ["__multi__"]["mask_keep_ranges"]` (populated by
   `row_segmentation_ui.py`'s "Select column to keep" multi-mask mode)
   is a single flat, UNLABELED union of x-ranges with no per-column
   identity — it can tell you *where* the kept areas are, not *which
   range is which column*, so it cannot drive per-field crops. Only
   `sidecar["columns"][name]` (the persistent per-column architecture
   masking one NAMED column at a time — the same masks
   `run_single_column_extraction` and `ground_truth_labeling_ui.py`
   already use) carries a real name → boundary mapping. Before running
   this script, mask every column named in your `columns.txt`
   **individually by name** in `row_segmentation_ui.py` (not via
   `__multi__`) — running without this now fails fast with a clear
   error listing exactly which columns are missing a mask, rather than
   silently falling back to the old, wrong full-row behavior.

   New flags: `--stage1-upscale-target-height` / `--stage1-upscale-max-width`
   (default 160/4096 — stage 1's field crops now need the same
   upscale-by-default treatment as `run_single_column_extraction`, real
   evidence justified this), `--stage2-upscale-target-height` /
   `--stage2-upscale-max-width` (default off — stage 2's crop size
   hasn't been directly measured yet the way the single-column 79×36px
   evidence was, so defaulting upscale on would be a guess rather than
   evidence — set explicitly once measured), `--tight-crop-padding-px` /
   `--tight-crop-padding-pct` (padding around the union of selected
   columns for stage 2, and around each individual column for stage 1 —
   same meaning as `run_row_extraction.py`'s identical flags).

4. **`ui/ground_truth_labeling_ui.py`**
   ```
   python ui/ground_truth_labeling_ui.py
   ```
   Human-verified per-field labeling, walking a sidecar row by row,
   field by field, showing the EXACT same crop (same mask) the model
   sees. Output: `data/outputs/ground_truth_log.jsonl` (append-only,
   resumable, skips already-labeled fields on reload). Explicitly
   captures genuine illegibility/blankness as correct answers, not
   failures to push past - this is a finding aid, not a source of
   truth.

5. **`training/export_lora_dataset.py`**
   ```
   python training/export_lora_dataset.py [--log-file PATH] [--out-dir PATH] \
       [--min-examples-per-status N]
   ```
   Converts `ground_truth_log.jsonl` into image+target-text LoRA
   training pairs (`train.jsonl` + per-row crop images + a coverage
   report). Targets are literal single-field values (including the
   literal strings `"illegible"`/`"blank"`), NOT the pipe-delimited
   `value|confidence` structuring convention used elsewhere - the
   first LoRA's job is cursive recognition and honest abstention, not
   confident-looking structured output.

### Debug flag — see exactly what image a model received (`--debug-model-inputs`)

Available on `scripts/run_row_extraction.py` (both single-column and
legacy multi-column mode) and `scripts/run_two_stage_extraction.py`.
Off by default, zero effect on extraction results or timing when
omitted — answers one specific diagnostic question: *when a model
returns blank output or describes "only lines," what exact image did
it actually receive?* Inferring that from bbox coordinates and config
values means trusting every preprocessing step ran correctly —
exactly the thing usually being debugged — so this saves the real
in-memory image object instead.

```bash
python scripts/run_row_extraction.py <sidecar.json> --model moondream2 --debug-model-inputs
python scripts/run_two_stage_extraction.py <sidecar.json> <columns.txt> \
    --ocr-model chandra --structure-model qwen3vl4b --debug-model-inputs
```

Also exposed as a **"Debug model inputs"** checkbox (unchecked by
default) in every GUI that can trigger one of these calls:
`ui/row_segmentation_ui.py`'s "Extract active column" panel
(single-column mode — passes a `DebugModelInputRecorder` straight to
`run_single_column_extraction()` in-process, since that UI calls the
extraction function directly rather than shelling out) and
`debug_tools/workflow_gui.py`'s Row Extraction and Two-Stage Extraction
tabs (append `--debug-model-inputs` to the subprocess command line they
already build for `scripts/run_row_extraction.py`/
`scripts/run_two_stage_extraction.py`). Same off-by-default,
zero-effect-when-unchecked behavior as the CLI flag — the checkbox
state is included in `workflow_gui.py`'s saved tab state, so it
persists across sessions like every other option there.

Writes to `data/debug_model_inputs/<run_id>/` (timestamped — never
overwrites a prior run; override the base directory with `--debug-dir`).
One subdirectory per model call:

```
data/debug_model_inputs/
└── 20260724T151203123456Z/
    ├── run_metadata.json         # list of every item captured this run
    └── row_0001/
        │  # scripts/run_row_extraction.py: one item, "row_0001" itself.
        │  # scripts/run_two_stage_extraction.py (field-level for BOTH
        │  # stages as of 2026-07-25 - stage 2 was field-level-union
        │  # briefly on 2026-07-24, then redesigned per-field the next day
        │  # after real evidence showed the union crop still forced stage 2
        │  # to solve "which region belongs to which column" on top of
        │  # reading the handwriting): ONE item per SELECTED COLUMN for
        │  # EACH stage - column_NN comes from the field's position in
        │  # sidecar["column_order"] when available (matching the real
        │  # page's own column numbering), else its position in the given
        │  # column list:
        ├── column_03_stage1/
        │   ├── original_crop.png     # full row bbox crop, before this
        │   │                         # column's own tight-crop/upscale
        │   ├── model_input.png       # EXACT per-field image object passed
        │   │                         # to loader._run_generate() for THIS
        │   │                         # column - tightly cropped to its own
        │   │                         # kept range and upscaled
        │   ├── prompt.txt
        │   ├── output.txt            # this column's raw OCR reading
        │   └── metadata.json         # includes column_index, column_name,
        │                             # row_bbox, field_bbox on top of the
        │                             # usual dimensions/preprocessing/
        │                             # model/runtime/empty_output fields
        ├── column_11_stage1/
        ├── column_12_stage1/
        ├── column_03_stage2/         # SAME field_bbox/crop dimensions as
        │                             # column_03_stage1 above - stage 2 now
        │                             # sees the identical per-field image,
        │                             # just re-reading/confirming it against
        │                             # that one field's own stage-1 hint,
        │                             # not a combined multi-column image
        ├── column_11_stage2/
        └── column_12_stage2/
```

Implemented once, in the shared path every extraction mode already
calls through (`crop_region_from_source()`'s `debug_stage_callback`
parameter, `core/row_segmentation.py`) — not duplicated per loader or
per caller. `core/debug_dump.py`'s `DebugModelInputRecorder` is
constructed unconditionally by both CLIs; every method is a no-op when
`enabled=False`, so callers never need an `if debug:` branch. A
debug-save failure (disk full, permissions) is logged and swallowed,
never allowed to abort a real extraction run. `training/
export_lora_dataset.py` and `scripts/model_assessment.py` don't route
through this flag — the former never calls a model (it only re-crops
already-labeled ground truth), and the latter operates on whole
documents rather than `crop_region_from_source()`'s row/column crops.

---

## Change log

Keep brief entries here when dependencies or major structure change,
so it's clear why a version was pinned/changed later.

- **Repo-root reorganization** (2026-07-25, per Jon's direction, done
  proactively while the architecture is still small rather than
  deferred until more modules exist and need moving too): every
  standalone script moved out of repo root into a purpose-named
  subfolder (`ui/`, `debug_tools/`, `benchmark/`, `training/`,
  `scripts/`, `diagnostics/` - see "Project layout" above for what
  lives where and why). Treated as a real refactor, not a bulk `git
  mv`: every moved file that computed its own project root via
  `Path(__file__).resolve().parent` had that changed to `.parent.parent`
  (one directory deeper now), and every file importing `core.*` (or, for
  `debug_tools/workflow_gui.py`'s lazy scoring-tab import,
  `benchmark.*`) gained an explicit `sys.path.insert(0, str(repo_root))`
  bootstrap - without it, running e.g. `python scripts/build_manifest.py`
  would set `sys.path[0]` to `scripts/`, not repo root, breaking every
  `from core.xxx import yyy` with `ModuleNotFoundError`. `benchmark/`
  gained an `__init__.py` so it can be imported as a package.
  `debug_tools/workflow_gui.py`'s 8 subprocess launch commands (for the
  other tools it runs as separate processes) and its scoring-tab import
  were updated to the new relative paths. `debug_model_inputs/` moved to
  `data/debug_model_inputs/` (generated evidence, same category as
  `data/outputs/`, not code) - `core/debug_dump.py`'s default `base_dir`
  and both CLI scripts' `--debug-dir` defaults updated to match. Added
  `main.py` as a minimal stub entry point (launches
  `debug_tools/workflow_gui.py` as a subprocess for now) so `python
  main.py` is the one launch command that never has to change again,
  even once the real end-user UI replaces `workflow_gui.py` underneath
  it. Every moved file's own usage docstrings/help text updated to the
  new paths; this README's live usage instructions updated to match
  (historical changelog entries below are left as-is - they're accurate
  narrations of what was true when they were written, not live
  documentation).

- Fixed a real crash in `AllowedCharsLogitsProcessor.__call__()`
  (constrained decoding, below) found on its first live use: it only
  handled `scores` being WIDER than the cached mask (hardware-padded
  lm_head), not narrower - InternVL's real `len(tokenizer)` (151675) is
  actually LARGER than its model's true output dimension (151674), a
  genuine tokenizer/model vocab-size mismatch, which crashed every
  generation call with a tensor shape-mismatch error the moment
  `restrict_output_charset` was turned on. Fixed by also truncating the
  mask when it's the larger side - verified against the real
  InternVL3-2B tokenizer, reproducing the exact reported numbers, then
  confirming generation completes cleanly with the fix.

- Added constrained decoding (`core/loaders/constrained_decoding.py`'s
  `AllowedCharsLogitsProcessor`, opt-in via `restrict_output_charset` in
  `config/models/*.yaml`, default false) to prevent off-script token
  fallback (real evidence: multiple runs produced stray Cyrillic
  fragments under low visual confidence) - masks generation at the
  token level to Latin/accented-Latin letters, digits, and this
  project's own output-format punctuation, complementing (not
  replacing) the existing post-hoc `_scrub_example_leakage()` backstop,
  since masking can recover a correct answer the scrub can only ever
  detect-and-discard. Wired into the 11 loaders confirmed (by auditing
  every loader's own `_run_generate()`) to call `transformers.generate()`
  directly - ChandraLoader and MoondreamLoader don't qualify (see the
  "Constrained decoding" section above for exactly why, including a
  real investigation into whether MoondreamLoader could support this
  later despite running in a separate subprocess). Fixed a real gap
  found while wiring this: `GemmaLoader`/`QwenLoader` never set
  `self.tokenizer` at all, unlike every other loader. Verified against
  two real tokenizers (correct allow/deny on digits, punctuation,
  accented Latin, Cyrillic, CJK, and every real special token) and by
  reproducing the actual saved Cyrillic-fallback incident - output
  changed from repeated Cyrillic to zero Cyrillic characters on the
  identical crop/prompt/model. Mask computation cost measured honestly
  (0.55-1.4s one-time per tokenizer, cached, negligible per-step cost
  after that) rather than assumed free.

- Fixed a real row-level parser gap found via a live two-stage run with
  qwen3vl4b as stage 2, cross-checked against newly added ground-truth
  files (`debug_model_inputs/Ground_truth/*.txt`, first 6 rows per
  source image): its output used "uncLEAR", "uncleared", "uncleard",
  and "unc lear"/"unc lea r" (mixed case, extra letters, stray internal
  spaces) everywhere it meant the documented confidence word
  "unclear". `CONFIDENCE_MAP.get(conf_part.strip().lower())` in
  `parse_row_output()` (`core/row_extraction.py`) only matched exact
  dict keys - every one of those variants except the mixed-case one
  failed the lookup and got silently DROPPED (not even recorded as
  "?"/unclear - just missing, surfacing as "Missing/dropped fields" in
  `schema_error`), a parser bug masquerading as a model-quality
  problem. Added `_normalize_confidence_word()`: same two-step
  tolerance `_normalize_key()` already uses for column names (strip ALL
  whitespace, not just ends, then lowercase) plus a prefix match
  against each of the three canonical words - "uncleared"/"uncleard"
  both start with "unclear" once whitespace-stripped, so they now
  resolve correctly. Deliberately narrow: "uncertain" does NOT start
  with "unclear" and still returns no match (this is spelling/spacing
  tolerance for the three DOCUMENTED words, not a guess at unrelated
  ones). Wired into both places `parse_row_output()` looks up a
  confidence word. Verified directly against the real problematic
  `raw_output` strings from the run that surfaced this.

- Fixed `config/models/smolvlm2_2b.yaml` having BOTH anti-repetition
  levers fully disabled (`repetition_penalty: 1.0` is a no-op,
  `no_repeat_ngram_size: null` is unset) - found via a live two-stage
  run where smolvlm2_2b (as stage 2) repeated its entire answer block
  5x verbatim, including hallucinating an extra `Occupation:` field
  never asked for, before stopping at `max_new_tokens` - roughly
  tripling runtime on the affected rows. Changed to
  `repetition_penalty: 1.15` / `no_repeat_ngram_size: 5`, the exact
  combination already proved out and settled on for
  qwen3vl2b/qwen3vl4b/gemma_extract/got_ocr2/granite_vision_2b/
  pixtral_12b/qwen25_vl_7b after real repetition-loop testing -
  smolvlm2_2b was simply never brought in line with that established
  result, not a case of trying something new.

- Fixed a real prompt-format bug found via a live two-stage run using
  FlorenceLoader as stage 1: `FlorenceLoader._run_generate()` always
  returns `json.dumps(...)` (e.g. `{"<OCR>": "actual text"}` - by
  design, so the assessment tool can show region/bbox data for
  `<OCR_WITH_REGION>` too), but `run_two_stage_extraction()` was
  embedding that raw JSON VERBATIM into stage 2's combined-reading
  text - with the literal token `<OCR>` sitting right next to the real
  content. Stage 2 (smolvlm2_2b) echoed `"<OCR>"` back as its own
  answer for every field on the affected row, rather than the actual
  transcribed text. Not a repeat of the cropping/plumbing bugs already
  fixed - confirmed via that same run's other rows and via
  `--debug-model-inputs` that the crops themselves were correct; this
  was purely a text-formatting mismatch between one specific loader's
  output shape and what gets fed into the next model's prompt. Fixed
  with a new `_plain_text_reading()` helper (`core/row_extraction.py`)
  that unwraps a Florence-shaped `{"<token>": "text"}` reading down to
  the plain string before it's combined for stage 2 - deliberately
  narrow (only a dict with exactly one `<...>`-shaped key and a plain
  string value; a nested `<OCR_WITH_REGION>` value is left untouched
  rather than half-unwrapped) and a no-op for every other loader's
  plain-text output, which fails `json.loads()` and passes through
  unchanged. The debug capture's `output.txt` still shows the true,
  unmodified raw model output - only the copy combined into stage 2's
  prompt is unwrapped.

- Added a "Debug model inputs" checkbox (unchecked by default) to
  every GUI panel that can trigger a `--debug-model-inputs`-capable
  extraction: `row_segmentation_ui.py`'s "Extract active column"
  panel (passes a `DebugModelInputRecorder` directly to
  `run_single_column_extraction()`, since that call happens in-process
  on a background thread rather than via subprocess) and
  `workflow_gui.py`'s Row Extraction and Two-Stage Extraction tabs
  (appends `--debug-model-inputs` to the CLI command those tabs
  already build). Same off-by-default, zero-effect-when-unchecked
  guarantee as the CLI flag itself; `workflow_gui.py`'s checkbox state
  persists across sessions via the same capture_state/restore_state
  mechanism every other tab option already uses.

- Fixed a second instance of the exact same bug in stage 2's own
  image, found immediately after fixing stage 1 (below) via a real
  `--debug-model-inputs` capture on `row_NNNN/stage2` showing the same
  `original_crop_dimensions == final_crop_dimensions` signature: stage
  2 used `sidecar["columns"]["__multi__"]` to PAINT unwanted columns
  white without ever narrowing the image, so it stayed the full row
  width regardless of how few columns were selected. Now builds its
  crop from the SAME validated per-column masks stage 1 uses (the
  union of every selected column's own range, tightened + padded via
  `tight_crop_to_ranges()`) instead of `__multi__` - stage 2 no longer
  depends on `__multi__` at all. New `--tight-crop-padding-px`/`--pct`
  flags apply to both stages now (stage 1: padding around each
  individual column; stage 2: padding around the combined union).
  `--stage2-upscale-target-height` intentionally still defaults to off
  - this crop's typical size (union of several columns) hasn't been
  directly measured yet, so defaulting upscale on would be a guess,
  not evidence, unlike stage 1's single-column-derived 160px default.

- Redesigned `run_two_stage_extraction()`'s stage 1 to run per-column,
  not per-row — a real bug `--debug-model-inputs` caught directly on
  its first real use: stage 1 was making ONE OCR call per row against
  the FULL row crop (e.g. 3155×38px) with unwanted columns painted
  white, asking the model to locate the wanted columns itself inside
  that wide, mostly-blank image, rather than being shown one tightly-
  cropped image per selected column. Confirmed via a debug capture
  showing `original_crop_dimensions == final_crop_dimensions` (no
  tight-crop/upscale had run) and a `model_input.png` only a few dozen
  pixels tall — proof the "legacy multi-column path doesn't tight-crop"
  assumption behind the prior `run_two_stage_extraction()` upscale
  default (see the entry below) had calcified into the wrong behavior
  for what stage 1 actually needed. Fixed by extracting
  `run_single_column_extraction`'s per-column mask logic into a shared
  `_resolve_column_field_mask()` (`core/row_extraction.py`) and reusing
  it in stage 1's loop: one OCR call per selected column, each against
  that column's own tightly-cropped, independently-upscaled field crop
  — the same crop `run_single_column_extraction` and
  `ground_truth_labeling_ui.py` already use, not a second, possibly-
  diverging interpretation. Per-field readings are combined into the
  text stage 2 receives; stage 2's own whole-row image is unchanged.
  Surfaced a real, separate data-model gap while implementing this:
  `sidecar["columns"]["__multi__"]["mask_keep_ranges"]` (the multi-
  select mask `row_segmentation_ui.py`'s "Select column to keep" mode
  writes to) is a single flat, UNLABELED union of x-ranges with no
  per-column identity — it cannot drive per-field crops on its own.
  Only `sidecar["columns"][name]` (masking one named column at a time)
  carries a real name → boundary mapping, so `run_two_stage_extraction`
  now validates upfront that every requested column has its own saved,
  row-applied mask, failing fast with a message naming exactly which
  columns need masking (and how) rather than silently falling back to
  the old full-row behavior. `--upscale-target-height`/`--upscale-max-
  width` split into separate `--stage1-*`/`--stage2-*` flag pairs
  (stage 1 now defaults to upscale ON at 160px, matching
  `run_single_column_extraction`'s evidence-based default; stage 2
  keeps the old off-by-default, matching its own unchanged whole-row
  image). Verified with a real sidecar/source image and stub loaders
  (no GPU needed for this check): 5 selected columns on 1 row now
  produce 5 separate stage-1 OCR calls (not 1), each field crop
  confirmed upscaled from its raw tiny height, 5 separate debug items
  created (`column_NN_stage1/`, `NN` from the sidecar's real
  `column_order` when available) each with `column_index`/
  `column_name`/`row_bbox`/`field_bbox` in its metadata, the combined
  per-field readings confirmed present in stage 2's actual prompt, and
  the new upfront validation confirmed to raise a clear, actionable
  error (not a silent fallback) against a real sidecar whose columns
  only had `__multi__` masks, not per-column ones.

- Added `--debug-model-inputs` (new `core/debug_dump.py`, wired into
  `run_row_extraction.py` and `run_two_stage_extraction.py`) — saves
  the exact original bbox crop, the exact final image object handed to
  `loader._run_generate()` (after masking/tight-crop/upscale, matching
  what the model actually receives), the exact prompt, raw output, and
  per-item metadata (bbox, before/after dimensions, preprocessing
  settings, model name, reasoning/config flags, runtime, empty-output
  flag, exception+traceback) to `debug_model_inputs/<timestamped_run_id>/`.
  Built to answer one specific question directly rather than by
  inference: when a model returns blank output or says it sees only
  lines, what exact image did it receive? Implemented as a single
  instrumented point in `crop_region_from_source()`
  (`core/row_segmentation.py`'s `debug_stage_callback` parameter) so
  every extraction path shares the same capture logic rather than
  duplicating it per loader. Off by default, verified to have zero
  effect on extraction results/behavior when disabled and to create no
  files at all; a debug-save failure is logged and swallowed, never
  allowed to abort a real extraction run. Verified end-to-end against a
  real sidecar/source image with a stub loader (no GPU required for
  this check): debug directory creation, both crop stages written,
  prompt/output/metadata saved, the error path captured without
  aborting extraction, the saved `model_input.png` confirmed
  pixel-for-pixel identical to the actual image object passed to
  `_run_generate()` (not just visually similar), and the disabled/
  no-recorder-arg cases confirmed to create zero files.

- Flipped `config/models/moondream2.yaml`'s `reasoning_enabled` from
  `false` to `true`, matching `query()`'s own documented default
  (confirmed via moondream3-preview's README, same API lineage — not a
  100%-confirmed identical default for moondream2 specifically, but a
  strong same-family signal). The docs frame `reasoning=False` as a
  speed/cost optimization for trivial questions only, keeping it on for
  anything harder — this project had been running the opposite of that
  for exactly the task (ambiguous handwritten cursive) least like a
  trivial question. Untested in practice as of this entry; a real
  before/after comparison is still needed, and this is an
  accuracy-vs-speed tradeoff, not a strict win.

- Added `config/prompts/ocr_stage1_moondream_query.txt`, phrased as a
  direct question rather than an imperative instruction — moondream2's
  `query()` API is documented and built for short direct questions, and
  a real stage-2 test showed it echoing a long multi-rule instruction
  block back verbatim instead of following it. Every other
  `ocr_stage1_*.txt` file's imperative phrasing is fine for the
  instruction-following VLMs those target, but not the shape moondream2
  expects.

- Generalized `MoondreamLoader`'s subprocess pattern into a reusable
  `SubprocessLoaderBase` (`core/loaders/subprocess_loader_base.py`) +
  `_subprocess_worker_common.py`, after an inventory of other candidate
  models' own `config.json` files confirmed the venv-per-model-version
  problem isn't moondream-specific (moondream3-preview and HunyuanOCR
  both want different pinned `transformers` versions again).
  `MoondreamLoader` itself shrank from 264 to ~115 lines as the direct
  result of this extraction. Also fixed a real resource leak this
  refactor surfaced: `_release_model()`'s `del loader` only dropped its
  own local binding, never guaranteed to run before the caller's frame
  exited, so a moondream2 worker subprocess (a genuinely separate CUDA
  context in a different venv/process, invisible to the parent's
  `torch.cuda.empty_cache()`) could still be resident in VRAM when a
  second model tried to load — undermining
  `run_two_stage_extraction()`'s documented "never two models resident
  at once" guarantee. Fixed via a new `BaseLoader.release()` hook
  (default no-op) called explicitly and deterministically by both
  `_release_model()` copies before clearing model attributes;
  `SubprocessLoaderBase.release()` terminates the worker
  (`terminate()`, 5s wait, `kill()` fallback) instead of relying solely
  on non-deterministic `__del__` timing.

- Fixed a real resolution-starvation confound found via a testing-
  methodology brainstorm: the tight-crop fix below correctly solved
  "mostly blank, full-row-width" crops, but never checked whether what
  was left was big enough to actually read — real sidecars measured
  tightened column crops as small as 79×36 pixels (a "Sex" column), not
  enough detail for most vision encoders regardless of model
  capability. This means some earlier model-comparison conclusions in
  this project may partly reflect resolution starvation rather than
  genuine capability differences — a confound that had gone unchecked.
  Added `core/row_segmentation.py`'s `upscale_to_target_height()`
  (aspect-preserving LANCZOS upscale to a target height, never shrinks
  an already-adequate crop, `max_width`-capped against runaway aspect
  ratios) and wired it through `crop_region_from_source()`,
  `_extract_region()`, `run_single_column_extraction()` (new default:
  ON, target height 160px — real evidence justified this default, not
  a guess), `run_two_stage_extraction()` (default OFF — the 79×36
  evidence was measured on tight-cropped single-column images
  specifically, and this legacy multi-column path doesn't tight-crop),
  and `export_lora_dataset.py` (default matches
  `run_single_column_extraction`'s 160, deliberately, so LoRA training
  images reflect the same input distribution real extraction sends the
  model). New `--upscale-target-height` / `--upscale-max-width` CLI
  flags on `run_row_extraction.py`, `run_two_stage_extraction.py`, and
  `export_lora_dataset.py`. `extraction_meta` now records the
  preprocessing settings actually used, same distinguishability
  principle as `tight_crop_applied` — pre-fix results (including this
  project's first smolvlm2 quality test batch) should be treated as a
  resolution-starved baseline, not compared directly against later
  results without checking this flag.

- Added MoondreamLoader (`core/loaders/moondream_loader.py`,
  `vikhyatk/moondream2`) as an extraction-stage candidate. Its
  `trust_remote_code` model produces garbage output on this project's
  main `transformers` (5.12.1) — confirmed via direct A/B test, not
  assumed — but correct output on `transformers==4.52.4`, its own
  declared version. Since one process can't run two `transformers`
  versions, the loader now spawns `core/loaders/_moondream_worker.py`
  as a subprocess under a separate `.venv_moondream` venv
  (`--system-site-packages`, so it reuses the main env's torch/CUDA
  rather than a second multi-GB download) and talks to it over
  stdin/stdout JSON — see the "Subprocess-backed loaders" section above
  for the one-time setup command. Also closes
  off moondream2's LoRA-variant-download path (a bare `urlopen()`
  against `api.moondream.ai` in the model's own remote code) at two
  layers: neither the loader nor the worker ever pass the
  `variant`/`settings` kwarg that triggers it, and the worker
  additionally monkeypatches the download function to raise instead of
  opening a socket.

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
