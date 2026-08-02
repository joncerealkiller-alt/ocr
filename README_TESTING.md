# Hint Validation UI — testing notes (TEMPORARY, delete once verified)

## ⚠️ PICK UP HERE (2026-07-29, session 2) — open issue, not yet resolved

**Table boundary auto-detection may not match the manually-confirmed
one for the same page**, discovered while sanity-testing the column
x_frac fix below. Needs investigation before trusting Stage 4's column
masks on real batch output.

What happened: regenerated `data/outputs/auto_row_segmentation/
e001926997_dewarped_sidecar.json` (`python scripts/auto_generate_sidecar.py
data/outputs/dewarped/e001926997_dewarped.png --debug --force`) after
the column-frac fix below, to sanity-check it against your manually-
confirmed sidecar (`data/outputs/row_segmentation/e001926997_sidecar.json`).
The two don't agree on `table_bbox` at all:

| | table_bbox `[left, top, right, bottom]` | source image |
|---|---|---|
| Your manual sidecar | `[413, 1408, 7312, 4308]` | `data/working/e001926997.png` (deskewed only, **pre-dewarp**) |
| Auto-regenerated sidecar | `[153, 1237, 6786, 4100]` | `data/outputs/dewarped/e001926997_dewarped.png` (**post-dewarp**) |

**These are two different images of the same page** — your manual
masking (the numbers I used to calibrate `column_regions_approx`
below) was done against the pre-dewarp working copy, not the dewarped
image Stage 4 actually consumes. If perspective correction shifted the
table's position/size non-trivially for this page, the calibration
data itself may not transfer cleanly to what the batch pipeline sees —
distinct from (and potentially compounding) the page-vs-table-relative
bug already fixed below.

Symptom in the regenerated sidecar: `Family Number` and `Age` both got
flagged `needs_review` with implausibly narrow located widths (~10px),
consistent with the auto-detected table boundary being different
enough that the (now table-relative, but still calibrated-on-a-
different-image) expected positions missed their real ruling lines.

**Next steps to figure out**: is `locate_table_boundary()` itself
inaccurate on the dewarped image, or is the dewarp step genuinely
shifting the table enough that pre-dewarp calibration doesn't apply
post-dewarp? Worth re-masking `Family Number` (and maybe re-confirming
Name/Age/Sex/etc.) by hand directly against a **dewarped** image next
time, not the working copy, so calibration and the real pipeline input
are the same image.

---


What got built this session: a fast Yes/No validation pass that uses the
automatedgenealogy.com CSV pull as ground-truth hints, so confirming a
field against the image is a click instead of retyping it.

## Files touched/created

| File | What it is |
|---|---|
| `ui/hint_validation_ui.py` | The new UI. Two phases in one window: Phase 1 confirm/deny CSV hints, Phase 2 manual entry for anything rejected or unhinted. |
| `core/csv_hint_source.py` | Matches a sidecar's page/row to the CSV and looks up the hint value for a given column name. |
| `config/columns/validation_1911_census.txt` | The column list this UI walks, in sidecar column order. `#`-prefixed lines are skipped entirely. |
| `data/automatedgenealogy_pull.csv` | The 716-row reference CSV scraped earlier this session — header renamed and `links` dropped to match the columns file. |

## How to run

```bash
python ui/hint_validation_ui.py
```

Then, in any order:
1. **Load sidecar...** — a segmentation sidecar JSON from `data/outputs/row_segmentation/`.
2. **Load columns file...** — `config/columns/validation_1911_census.txt`.
3. **Load reference CSV...** — `data/automatedgenealogy_pull.csv`.

Once all three are loaded, `_build_queues()` splits every un-labeled
(row, column) pair into:
- **Phase 1 (review queue)** — pairs where the CSV has a hint.
- **Phase 2 (manual queue)** — pairs with no hint (unmatched page, or a
  `#`-skipped column) or that were rejected in Phase 1.

## Phase 1 — Confirm/Deny

For each field: shows the same crop the model would see (via
`crop_region_from_source`, identical logic to the existing ground-truth
UI) plus the CSV hint text in a big box.

- **Yes (green button / `Y` / Enter)** — writes a record straight to
  `data/outputs/ground_truth_log.jsonl` with `status: "readable"`,
  `notes: "confirmed_from_reference_csv"`.
- **No (red button / `N` / Backspace)** — nothing written; the field is
  pushed onto the Phase 2 queue instead.

## Phase 2 — Manual entry

Same controls as `ui/ground_truth_labeling_ui.py` (status radios,
value entry, quick-fill buttons, notes, Save & Next) for whatever
landed in the manual queue. Writes to the same JSONL log.

Both phases share the **same resume key** as the existing ground-truth
tool — `(sidecar_path, row_index, column)` — so this UI and the
original one can be run interchangeably across sessions without
double-labeling a field.

## Key functions to know if something looks wrong

**`core/csv_hint_source.py`**
- `_page_identifier(source_pdf_url)` — `.../e001946614.pdf` → `e001946614`.
- `_image_matches_identifier(image_path, identifier)` — page match is a
  **suffix check**: does the sidecar's image filename stem end with
  the CSV row's PDF identifier? This is a placeholder until the real
  auto-pipeline output naming scheme is confirmed — **check this first**
  if a sidecar isn't finding its CSV page.
- `rows_for_image(all_rows, source_image_path)` — returns
  `{line_num: csv_row}` for the matched page, or `{}` if nothing matches.
- `get_hint(csv_row, column_name)` — normalizes `column_name`, looks it
  up in `CSV_COLUMN_MAP`, returns the joined value or `None`.
- `CSV_COLUMN_MAP` — the name translation table. Add entries here if
  you rename columns again in the `.txt` file.

**`ui/hint_validation_ui.py`**
- `_build_queues()` — the row_index↔line_num join point (direct 1:1,
  per Jon: sidecar rows are assigned by walking the table the same way
  the CSV was transcribed).
- `_refresh_csv_page_rows()` — re-filters the CSV to the current
  sidecar's page; call chain is triggered from both `load_sidecar_file`
  and `load_csv_file` so load order doesn't matter.
- `confirm_hint()` / `reject_hint()` — Phase 1 actions.
- `save_manual_and_next()` — Phase 2 action, copy-pasted logic from
  `ground_truth_labeling_ui.py`'s `save_and_next()`.

## Status as of end of session 2 (2026-07-29)

Resolved this session:
1. ~~Page-matching never run against real output~~ — run for real,
   found `_image_matches_identifier` was checking `endswith()` when the
   PDF identifier is actually a **prefix** of the output filename
   (`e001926997_dewarped.png`, not `..._e001926997.png`). Fixed to
   `startswith()`.
2. ~~Row-index-to-line_num alignment unconfirmed~~ — confirmed correct
   in practice (direct 1:1 join, no issues seen).
3. `config/columns/validation_1911_census.txt` — populated for real:
   `Family Number, Name, Sex, Relationship to Head, Marital Status,
   Month of Birth, Year, Age` (Dwelling Number/Address/Links skipped
   via `#` — Address's transcription data is condensed/re-numbered and
   not trustworthy as a hint, Links doesn't need tracking).
4. CSV header renamed to match (`household_num`→`family_number`,
   `birth_month`→`month_of_birth`, `birth_year`→`year`), `links` column
   dropped.
5. Full pipeline run end-to-end for real (16 LAC PDFs → images →
   working manifest → classified → dewarped → finalized → Gemma
   subtype → 13 auto-sidecars). Found and fixed a real bug along the
   way: `scripts/run_batch_auto_sidecar.py`'s `find_dewarped()` glob
   was matching the dewarp tool's `.json` metadata file instead of the
   image (alphabetically `json` < `png`) — every file was silently
   failing to load. Fixed by filtering to `IMAGE_EXTENSIONS`.
6. Found and fixed a real column-cropping bug: `locate_columns()` in
   `core/auto_sidecar.py` computed column x-positions as a fraction of
   the full PAGE width, when it should be relative to the TABLE's own
   width — confirmed as the actual cause of an apparent ~4pp
   "disagreement" between reference samples (it was a dewarp/scan
   margin difference, not a real column-position difference). Fixed in
   `locate_columns()` + `_COLUMN_SEARCH_RADIUS_MIN_FRAC`, and all 4
   templates' `column_regions_approx` re-derived table-relative
   (`canada_census_1911/1921/1931.yaml`, `printed_manifest.yaml`).
   Also added previously-missing `Family Number`/`Marital Status`/
   `Month of Birth`/`Year` calibration to `canada_census_1911.yaml`
   (provisional — one real reference sample so far, not the averaged
   2-3 samples the other columns have).

**Not yet resolved** — see "PICK UP HERE" at the top of this file: a
manual-vs-auto `table_bbox` mismatch on `e001926997` that may mean the
new Family Number/Marital Status/Month of Birth/Year calibration
(measured on the pre-dewarp working copy) doesn't cleanly transfer to
the dewarped image Stage 4 actually processes.

**Still not run**: `ui/hint_validation_ui.py` itself hasn't completed a
full successful pass yet (was mid-testing when the page-matching bug
was found and fixed) — re-run it against a Stage-4 sidecar to confirm
Phase 1 now populates hints correctly.
