# Auto-preprocessing + auto-dewarp — running status

**Purpose**: pick-up-where-we-left-off notes for the automatic CV
preprocessing analyser and the automated dewarp/crop stage. This is a
STATUS log (what's done, what's known, what's next). Symbol-level API
details live in `docs/CODE_MAP.md`; how-to-run lives in `README.md` /
`PIPELINE_WORKFLOW.md`.

Last updated: **2026-08-02**.

**Terminology note**: this doc predates the 2026-08-02 stage-numbering
standardization - see `docs/PIPELINE_STAGE_TERMINOLOGY.md` for the
canonical Stage 0-6 names. "Stage A" below = new **Stage 1 (Raw Sensor
Capture)**'s physical half; "Stage B" = new **Stage 2 (Decision
Engine)**. Also see `docs/REFERENCE_PIPELINE_V1.md` for why the
existing corpus's captures turned out to be post-preprocessing, not
pre-preprocessing as originally labeled.

---

## Conceptual model (Jon, 2026-08-02): two complementary sensor systems

The pipeline has two independent measurement systems feeding evidence
to (eventual) downstream decisions - neither replaces the other:

- **Physical sensors (Stage A / `core/image_analysis.py`)** - measure
  properties of the image ITSELF: geometry, skew, blur, illumination,
  contrast, edge statistics, other interpretable CV measurements.
- **Semantic sensors (the 8 qualified vision towers / `core/
  baseline_embeddings.py`)** - measure HOW different vision
  architectures perceive the document: layout, visual structure,
  semantic similarity, content representation, architecture-specific
  perception.

The captured embedding vectors are **measurements produced by
sensors**, not simply "stored embeddings" - the objective is a
measurable perception layer for the pipeline, not a cache.

**Current pipeline shape** (production research baseline):

    Source Discovery
            v
    Source Expansion
            v
    Manifest
            v
    Raw Sensor Capture (Semantic)   <- core/baseline_embeddings.py, DONE, all 1677 images
            v
    CV Analysis (Physical)          <- core/image_analysis.py, DONE but not wired into manifest_pipeline.py
            v
    Profile Selection (future Stage B)   <- NOT STARTED
            v
    Preprocessing                   <- fixed "autocontrast" for every image, today
            v
    Post-preprocessing Sensor Capture (future)   <- NOT STARTED
            v
    Vision Routing                  <- Multi-Tower Routing Audit, DONE (docs/BENCHMARK2_3_MULTI_TOWER_ROUTING_AUDIT.md)
            v
    OCR

**Immediate goal, explicit**: the semantic sensor layer is NOT used to
make preprocessing decisions yet. The baseline capture is immutable
reference data - "what did the vision system perceive before the
pipeline modified the image." Once post-preprocessing capture exists,
the delta between the two observations becomes semantic telemetry
about what preprocessing actually did to each image - a measurable
transformation, not just an embedding-distance comparison.

**Already-supported downstream uses of the baseline data as-is** (no
further capture needed): nearest-neighbour search, duplicate detection,
corpus clustering, semantic corpus QA, bucket contamination detection,
retrieval. **Future work, not yet built, do not assume complete just
because capture infrastructure exists**: preprocessing drift
measurement, preprocessing profile evaluation, semantic stability
measurement, routing diagnostics, preprocessing policy research, Stage
B profile selection (once evidence demonstrates predictive value - not
inferred from embeddings without that evidence).

---

## The agreed plan (Jon, 2026-07-29)

Split the preprocessing decision into two stages so the measurements
outlive any given policy — *"if in six months you replace your
preprocessing policy, you don't have to rewrite the measurements."*

    Stage A  image analysis   -> pure measurement, modifies nothing
    Stage B  policy           -> maps measurements onto a named profile

Build order:

1. **Image analyser** (cheap measurements, no image modification) — **DONE**
2. **Automatic quad/boundary detection**, benchmarked against the labelled
   corner set — **NOT STARTED** (target revised, see below)
3. **Profile selection** driven by analyser output (Stage B) — NOT STARTED
4. **Automatic dewarp in the UI** with manual adjustment — NOT STARTED

Two refinements Jon added on 2026-07-29, both now reflected in the code:

- **Two ROI detectors, not one.** The corpus is no longer homogeneous
  (census schedules, passenger manifests, receipts, typescript on backing
  board, microfilm captures with huge borders). For census/manifest the ROI
  is the **table**; for a letter/receipt it's the **paper**. Stage A
  measures both and reports confidences; **Stage B chooses**. That keeps
  document-specific assumptions out of Stage A.
- **ROI detection is a prerequisite for *some* measurements, not all.**
  Geometry-invariant: aspect ratio, skew, ruling-line angles. ROI-dependent:
  Otsu threshold, ink fraction, polarity, stroke width, illumination.

## What exists now

`core/image_analysis.py` — Stage A. Two-tier schema: geometry-invariant
fields at top level; ROI-dependent fields in per-region `RegionStats`
blocks under `.regions` (`"frame"` always, `"page"`/`"table"` when
detected). `analyze_image()`, `measure_region()` (public on purpose),
`save_analysis`/`load_analysis` (sidecar round-trip verified identical, so
Stage B can re-run off stored JSON without touching pixels),
`analyze_manifest()` → flat CSV with `frame_*`/`page_*`/`table_*` columns.

    python -m core.image_analysis <image|manifest.csv> [--report PATH] [--no-sidecars]

Stage A's own `image_analysis.py` is still not wired into
`core/manifest_pipeline.py` — Stage 0 still applies a fixed
`"autocontrast"` to every image regardless of what Stage A measures.

**2026-08-02, a related but separate capability WAS wired in**:
`core/baseline_embeddings.py` captures a pre-preprocessing vision-tower
embedding (all 8 qualified encoders) for every image, called from
`build_working_manifest_from_paths()` immediately after
`copy_to_working_dir()` and before `preprocess_for_manifest()` -
persisted to `data/baseline_embeddings.json` (image_hash, all 8
encoders' vectors, `preprocessing_stage: "pre_preprocessing"`,
library_versions). This is Jon's proposal to have a real per-image
datapoint available before deciding what preprocessing that image
needs, matching `docs/BENCHMARK2_METADATA_LAYER_QUALIFICATION.md`'s
Third/Fourth extension design. **It only captures and persists - it
does NOT decide anything yet.** Stage B (profile selection, using
either Stage A's CV measurements, this baseline-embedding data, or
both) is still NOT STARTED - this just makes sure the raw material for
that future decision exists per-image once, rather than needing a
retroactive recompute later. Full production corpus (1677 images)
captured 2026-08-02.

## Key findings (all measured, not assumed)

**OpenCV 5.0.0 is available** on the main interpreter (`C:\Python314`; the
`.venv_*` dirs are per-model subprocess loader envs only). Every earlier
"auto-dewarp is intractable" note in `core/warp_detection.py` and
`core/manifest_pipeline.py` was written against a PIL+numpy toolbox and
should be re-read in that light. Note cv2 5.x's `HoughLinesP` returns
`(N, 4)`, not 4.x's `(N, 1, 4)`.

**The labelled dewarp ground truth is 16 pages, not 28.**
`data/outputs/dewarped/` has 16 `*_dewarped.json` sidecars with
human-placed corners (all 16 sources still present in `data/working/`);
the other ~40 `.jpg` outputs predate sidecars and have no labels.

**Those human corrections are ~all crop, almost no keystone**: 2 pure
axis-aligned crops + 14 quads deviating only 0.12%–1.08% of page width
(max 74.6px on ~7000px pages). This explains `core/warp_detection.py`'s
two failed calibrations — it was hunting a sub-1%-of-width signal, i.e. an
essentially absent one. Not a flawed technique; a mis-specified target.
**Hence step 2's target was revised from perspective-first to
boundary-cropping-first, with keystone as a small correction on top.**

**Real skew DOES exist elsewhere in the corpus** — the microfilm reels
measured dominant vertical ruling angles ~1° median, max 12.57°. So "this
project's images aren't warped" is true of the `e0019466xx` batch ONLY.

**The ROI split is empirically validated**: frame-level polarity flagged
46/218 microfilm pages as inverted (two inspected directly — both were
ordinary dark-on-light documents, so both flags were wrong); page-level
polarity flagged **0/218**, which is the correct answer. Root cause of the
frame-level failure: with a large dark surround, a single Otsu threshold
splits PAPER from SURROUND rather than INK from PAPER.

**First real calibration — `table_confidence` discriminates**: the 16
labelled census pages score min 0.375 / median 1.0; the 218 microfilm
pages (letters, receipts) score median 0.125 / p90 0.25, with only 13/218
reaching the census minimum. A floor around **~0.375** is a defensible
starting point for "is there really a table".

## Corpora used

- `data/working/` + `data/outputs/dewarped/*.json` — 16 census pages with
  human corner labels. The only labelled geometry ground truth.
- `J:\Screenshots\Knott_Ancestry\Archive Microfilms` — 218 jpgs
  (`oocihm.lac_reel_*` LAC microfilm) + 17 pdfs (16 are the PDF originals
  of the labelled census pages). Analyser run 218/218, 0 failures. Report:
  `data/outputs/image_analysis/microfilm_analysis_report.csv`. Run with
  `--no-sidecars` — **do not write into that source archive.**

## Known gaps — do NOT mistake these for working

1. **`bright_region` page detection is wrong on a real case.** On
   `oocihm.lac_reel_c10639.733.jpg` it returns the white "PUBLIC ARCHIVES"
   label board (bbox y 3117–3727 of 3840) instead of the document above it.
   "Paper is the brightest large thing" fails on aged mid-grey microfilm
   paper next to a modern white label. Confidence capped at 0.55; a cap is
   not a fix. **This is the weak link, and it's step 2's real work.**
2. **`table_boundary` returns spurious boxes.** 174/218 microfilm pages get
   a boundary, nearly all low-confidence ~92%-of-frame junk. Edge exclusion
   cut them from 99.94% → 92% but the film border isn't flush with the frame
   edge. **Stage B must gate on confidence, not presence**; the
   `min 2 lines per axis` gate is too permissive alone.
3. **`estimate_deskew_angle` returns exactly 0.00 on ~90% of microfilm** and
   15/16 census pages. Correct for the pre-deskewed census files, but a
   silent zero on microfilm — indistinguishable from "genuinely straight" —
   while `dominant_vertical_angle_deg` measures ~1° median on the same
   images. It needs strong horizontal text lines and finds none in a
   dark-surround frame. **Prefer the ruling-line angle as the skew signal
   there.** Widening the search range to ±15° was treating the wrong problem
   (only 1/218 was actually clamped) and it TRIPLED runtime — the full 218
   now takes >10 min, dominated by the rotation search. Narrowing it back is
   probably both faster and more accurate, but it touches a function the
   census path already depends on, so **ask Jon before changing
   `row_segmentation.estimate_deskew_angle` itself.**
4. Every threshold except the `table_confidence` floor above is
   **UNCALIBRATED**. Confidences are ordinal signals, not probabilities.

## Open questions for Jon

- Should `blur_laplacian_var` / `noise_residual_std` move into the
  per-region blocks too? Currently top-level, reasoning: both are
  properties of the *capture* (focus, film grain, JPEG) computed from local
  operators, so a uniform surround dilutes them predictably rather than
  corrupting them the way a global histogram statistic gets corrupted.
  One-line move if he'd rather have them scoped.
- Step 2 needs **labelled page boundaries for the non-table case** — the 16
  existing labels only cover table-bearing census pages, and the page
  detector is exactly what's weakest. A handful of labelled boundaries on
  the microfilm set would fix that.
- Are any of the 13 microfilm pages scoring ≥0.375 `table_confidence`
  genuine forms/manifests? Not inspected.

## Side investigation, 2026-07-30 — auto_sidecar column detection

Jon asked why rows/columns aren't detected fully on census forms
(`core/auto_sidecar.py`). **Diagnosed, nothing changed yet.** Full numbers
in `docs/CODE_MAP.md` under "Auto-sidecar column location". Summary:

- Rows are largely fine. **Columns are the problem.**
- Table edges hit a real ruling line 94% of the time; column edges only
  **30%**, silently falling back to the template fraction.
- Sweeping the threshold and search radius raises the hit rate to 60% but
  leaves accuracy FLAT — and the refinement as a whole beats the raw
  template fraction by only 0.05pp mean. It is doing ~nothing.
- **Root cause = table-ANCHOR registration error** (Jon's hypothesis,
  confirmed by measurement). Fitting one global scale+shift between two
  pages collapses 1911's apparent 4.33pp spread to **0.03pp** and 1921's
  4.73pp to 0.91pp — the columns are rigid relative to each other and the
  template fractions are already essentially exact. What varies is the
  detected table span (0.940–0.997 of page width across samples; the same
  edge sometimes snaps to the outer border and sometimes to the first inner
  rule). Every column inherits that error, scaled.
- **Proposed fix (needs Jon's go-ahead)**: anchor columns to a 2-point
  affine from reliably-detected structure rather than the raw table bbox.
  Do NOT rebuild column detection for 1911/1921 — their fractions are right
  once registered. Changes a deliberately "LOCKED IN" function, so not done
  unilaterally.
- **The apparent "1931 problem" was a CLASSIFICATION bug, not bad masks.**
  Two copies of the same 1911 page (`e078_e001946617_dewarped` and
  `http___data2...e001946617.jpg_dewarped`, header reads "FIFTH CENSUS OF
  CANADA, 1911") classify as 1911 and 1931 respectively. Regrouped by true
  year, all three real 1911 pages fit to **0.03–0.04pp**. Cause: the
  classifier's primary year discriminator (`vertical_ruling_lines`, an
  unbroken-run count) tracks scan SHARPNESS — the two copies differ in
  `blur_laplacian_var` (9793 vs 14775) and stroke width (2.74 vs 1.91px),
  moving the count 21 → 39 and crossing the 1911/1931 boundary.
  `core/image_analysis.py`'s morphological counter moves only 10 → 12 on
  the same pair and is a candidate replacement (requires re-calibrating
  every template's `column_count_range`).
- Incidental find: `_RULING_LINE_RUN_RATIO` (line 83) is dead as a tuning
  knob because of a Python default-argument binding gotcha.

## Column registration fix — IMPLEMENTED 2026-07-30, evidence-driven cascade

Done, in two passes. First pass: a page-level affine always overrode a
found edge when available. Jon's refinement: don't let the two compete —
use the affine as a SANITY CHECK on the measurement instead, only
overruling it when they disagree beyond an adaptive tolerance. Final
design is a 4-tier cascade per EDGE (not per column):
`measured` (agrees with page geometry, or nothing to check against) →
`affine_override`/`affine_predicted` → `locally_interpolated` (bracketed
between two other confident matches, possibly the same column's other
edge) → `template`. Full mechanism, constants, and the real
`affine_override` catch (a genuine same-line-collapse bug on
`1931_174-e011707164`'s "Age" column) are in `docs/CODE_MAP.md` under
"Column registration fix".

Headline numbers (9 real manually-masked reference pages, each scored
against its CORRECT doc_type): mean column-edge error 2.85% → **1.94%**,
median 1.80% → **0.78%** — essentially the same aggregate as the simpler
first pass, but now with the collapse bug actually fixed rather than
coincidentally-not-triggered, and full per-edge provenance instead of one
page-wide label. `locally_interpolated` didn't fire on this sample (the
two candidate pages' confident points are one-sided, so there's nothing
to bracket) — confirmed as correct refusal-to-extrapolate, not a gap.
`z000017634` remains the documented outlier, untouched by design.

**`source` in `column_locating` diagnostics changed shape**: it's now
`[x0_source, x1_source]` (two tier labels, one per edge), not a single
string — anything written against the single-value version from earlier
today needs updating.

## Doc-type classification — status (2026-07-30, Jon: "not sure we finished the doctypes")

**Gemma (`core/semantic_stages.py` + `config/prompts/
classifier_document_subtype_v2.txt`) is the pipeline driver**, not
`core/document_classification.py`'s CV classifier — `run_batch_auto_sidecar
.py` passes `doc_type_override` from `data/buckets/
dense_tabular_rows_subtype.csv`. Note `generate_auto_sidecar()` called
WITHOUT an override silently falls back to the CV classifier, which is how
an earlier pass in this session mistook a CV failure for a pipeline one.

- **Gemma's output on the 16-page batch**: 13 × `canada_census_1911`,
  3 × `unknown`. No 1921 or 1931 pages at all — so those two templates have
  **no validated samples in this batch**, and their calibration rests on one
  reference page each.
- **The CV fallback is not viable as a fallback.** It disagrees with Gemma
  on **6/16**. Worse, across the 13 pages Gemma confirms as 1911, its
  primary discriminator (`vertical_ruling_lines`) spans **13–43**, which
  covers the 1921 range [6,16], the 1911 range [16,28] AND the 1931 range
  [30,55]. Within-class spread exceeds between-class separation, so
  re-calibrating those ranges cannot fix it.
- The morphological counter in `core/image_analysis.py` is ~2–3x tighter on
  the same pages (12 of 13 land in 9–12 vs 13–23) but shares the same single
  outlier page. Better, still not separable. Low priority while Gemma drives.
- **The 3 `unknown`s look like a fixable PROMPT problem, not a model
  limitation.** Gemma read the CENTRE heading each time — "SCHEDULE No. 16
  Population", "SCHEDULE TABLE II. No. 213 Population by Name, Personal
  Description, Etc." — but the year and country are NOT there. On these
  forms they are printed small in the TOP CORNERS: "FIFTH CENSUS OF CANADA,
  1911" top-left and "CINQUIÈME RECENSEMENT DU CANADA, 1911" top-right
  (confirmed by eye on a rendered page). The prompt's CRITICAL RULE requires
  the word "Canada", so the model correctly returned `unknown` after reading
  a heading that could never contain it. **Suggested fix: tell the prompt
  WHERE the identifying text lives (top-left/top-right corners, bilingual
  EN/FR), and that the centre "SCHEDULE No. N" line carries the schedule
  number, not the year.** Not yet applied — needs a prompt-sweep run to
  confirm (`benchmark/prompt_sweep.py`).
- **The `confidence` field carries no information**: 15 of 16 rows are
  exactly 0.95 (the other 0.98), including all three `unknown`s. A 0.95
  confidence attached to "I can't tell" means it is not being used as a real
  signal — don't gate anything on it until that changes.
- **New reference pages, 2026-07-30**: `J:\Screenshots\Census` — 5 raw LAC
  pages, confirmed BY EYE from their printed corner titles: `e002879426`,
  `e002879427`, `e002879428` = **1921** (FORM 1.); `e011670449`,
  `e011670457` = **1931** (FORM 1A.). These are the first real 1921/1931
  samples beyond the single reference page each template was calibrated on.
  Not yet dewarped/segmented. The CV fallback scored **2/5** on them (called
  a 1921 page `handwritten_manifest` at confidence 1.00, and both 1931s
  `1921`) — more evidence it is unusable.
- **The corner-title layout is consistent across ALL THREE years**
  (verified on real pages): top-left `"<ORDINAL> CENSUS OF CANADA, <YEAR>"`
  (FIFTH=1911, SIXTH=1921, SEVENTH=1931), top-right the French
  `"<ORDINAL> RECENSEMENT DU CANADA, <YEAR>"`; the large CENTRE heading is
  `DOMINION BUREAU OF STATISTICS`/`POPULATION` (1921/1931) or
  `SCHEDULE No. N` (1911) and contains NO year or country.
- **v2 vs v3 SWEPT on GPU 2026-07-30** via
  `diagnostics/compare_document_subtype_prompts.py` (19 images, ground truth
  read by eye from each page's printed corner title, one model load):

  | group | v2 | v3 |
  |---|---|---|
  | FORMERLY_UNKNOWN (the 3 target pages) | 0/3 | **3/3** |
  | NEW_1921_1931 | 5/5 | 5/5 |
  | REGRESSION | 6/10 | 8/10 |
  | UK_GUARD (UK page must stay `unknown`) | 1/1 | 1/1 |
  | **TOTAL** | 12/19 | **17/19** |

  - The corner-text diagnosis was correct: all 3 formerly-`unknown` pages
    now classify as 1911, and v3's `title_text_read` shows it reading
    "FIFTH CENSUS OF CANADA, 1911" instead of the centre "SCHEDULE No." line.
  - **Gemma was ALREADY fine on 1921/1931** (5/5 under both prompts) — the
    new LAC pages did not need the prompt fix, they needed to exist.
  - **v2 has an unrecorded FORMAT failure mode**: on 4 manifest pages it
    emitted a bare `- handwritten_manifest` with no `document_type:` key.
    Production's `GemmaLoader._parse_kv_block()` requires `key: value`, so
    these become empty `document_type` + "missing fields" error, and
    `run_batch_auto_sidecar.py` then skips the row. Semantically v2 was
    RIGHT on 3 of those 4 — it is a schema-compliance failure, not a
    classification one. It has not bitten production yet only because
    manifests aren't in the `dense_tabular_rows` bucket. v3 emitted the key
    correctly on all 19.
  - **One genuine v3 regression**: `IMCANQC1865_T4821-00622`
    handwritten_manifest → `unknown`, with `title_text_read` =
    "DOMINION BUREAU OF STATISTICS..." on a page that has no such text. The
    corner guidance made it hunt for census corner text on a manifest. It
    self-reported confidence **0.1**, so the failure is visible rather than
    silent. A v4 should scope the corner instruction to census-shaped pages.
  - `S3HY-6LL2-RG` (v3: handwritten_manifest 0.95 vs ground truth
    `unknown`) is a taxonomy gap, not really a miss — it IS a manifest,
    just an out-of-list cabin-class one.
  - **The confidence fix works**: v3 produced 3 distinct values (0.1–0.98)
    and emitted 0.1 on exactly the two hardest cases (the UK guard page and
    its own miss). v2 was pinned at 0.95/0.98 across all 19. v3's
    confidence is worth gating on; v2's was not.
- **v4 was tried and REJECTED (2026-07-30)** — batched 3-way re-run of
  v2/v3/v4, one model load:

  | group | v2 | v3 | v4 |
  |---|---|---|---|
  | FORMERLY_UNKNOWN | 0/3 | **3/3** | 3/3 |
  | NEW_1921_1931 | 5/5 | 5/5 | 5/5 |
  | REGRESSION | 6/10 | **8/10** | 4/10 |
  | UK_GUARD | 1/1 | 1/1 | 1/1 |
  | **TOTAL** | 12/19 | **17/19** | 13/19 |

  v4 tried to fix v3's one manifest regression by scoping the corner rule
  to census-shaped pages, adding a manifest off-ramp, and telling the model
  NEVER to report title text it cannot see. It did the opposite:
  - **All 5 handwritten manifests → `unknown`, `title_text_read="not
    legible"`** (v3 got 4/5 right). The off-ramp made it abstain on
    manifests rather than route them.
  - **`30807_A000676-00099(1)` (a printed manifest) → `canada_census_1911`,
    title "FIFTH CENSUS OF CANADA, 1911"** — text that is not on the page.
  - It did NOT fix the `IMCANQC` case it was written for.
  - Census pages were unaffected (all still correct), so the damage is
    entirely on the manifest side.

  **Why it backfired — this is [[feedback_prompt_examples_get_locked_onto]]
  again.** v4's negative instruction literally quoted the strings
  "DOMINION BUREAU OF STATISTICS" and "CENSUS OF CANADA" as examples of what
  NOT to write, and the model then wrote one of them on a printed manifest.
  Naming a wrong answer in a prompt makes it MORE likely, not less. Any
  future v5 must express this as a positive instruction ("report the
  heading you can actually read") with no counter-example strings.

- **ADOPT v3.** It is the best of the three on every group, its numbers
  REPRODUCED EXACTLY across two independent runs (17/19 both times,
  including the same single `IMCANQC` miss), and v2 reproduced at 12/19 —
  so the run-to-run noise concern is resolved. v3's one regression fails
  loudly at confidence 0.1 into the manual-classification queue.
  - **Not yet switched** — `scripts/run_semantic_stages.py` still points at
    v2, awaiting Jon's go-ahead.

- **Manual classification routing is still TO BE BUILT** (Jon, 2026-07-30).
  Today `run_batch_auto_sidecar.py` just skips rows with `document_type` in
  `("", "unknown")` or a parse error, recording
  `status="skipped_unknown_doc_type"` — that skip list is the natural seam
  for the future manual route. Counting what each prompt would send there:
  **v2 routes 8/19 (6 needlessly), v3 routes 2/19 (1 needlessly)** — v2's
  needless six include 3 manifests it classified CORRECTLY but emitted in a
  format the parser drops. **v3's confidence is what makes a real gate
  possible** ("route if unknown OR confidence < 0.5"); v2's pinned
  0.95/0.98 cannot support any threshold.
- **`config/prompts/classifier_document_subtype_v3.txt` drafted**: v2 plus an explicit "read the top corners first"
  section, the ordinal→year mapping as a cross-check, an instruction not to
  answer `unknown` merely because the centre heading lacks a year, and a
  confidence field that must actually vary (v2's is pinned at 0.95).
  **Validate with `benchmark/prompt_sweep.py` v2 vs v3 before adopting.**
- **Duplicate copies**: `e001946617` exists twice (LAC `http___data2...`
  and `e078_...`). They are NOT a high-res/low-res pair — 3563×2170 vs
  3549×2168, essentially identical. The real difference is SHARPENING
  (`blur_laplacian_var` 14775 vs 9793, stroke width 1.91 vs 2.74px). If
  deduplicating, choose on sharpness/provenance, not resolution.

## Next action when picking this up

Step 2: build the boundary detector + a scoring harness against the 16
labelled quads (median corner error as % of page width is the metric
already used). Baseline to beat, measured 2026-07-29 with the Canny/
`approxPolyDP` Tier 1 detector: **14/16 detected, median corner error
1.95% of width, worst 4.69%**, errors clustering bimodally (~1.2% and
~4.5% — a systematic wrong-boundary pick, not noise). Note that 1.95% is
still LARGER than the ~1% keystone the humans were correcting, so accuracy
has to improve before this can drive an unsupervised correction.
