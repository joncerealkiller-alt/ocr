# Column-Number Anchor Research (2026-08-06)

## Origin

Direct continuation of the row-number-anchor work (`core/row_segmentation.py`'s
`detect_row_number_centers()`, built the same day - see that function's own
docstring for the row-axis history). Task spec: investigate whether printed
column numbers in a census form's header can provide geometric priors that
narrow the search window for the *existing* column-boundary detector
(`core/auto_sidecar.py`'s `locate_columns()`), without replacing it. Same
"prior narrows the window, detector stays authoritative" principle already
established for rows.

**Scope of this pass**: research and prototype only.
`core/auto_sidecar.py`'s production `locate_columns()` was NOT modified -
per direct instruction, this is evaluated before any production wiring.

## Status & handoff (as of Round 17, 2026-08-06) - READ THIS FIRST

This doc is a long, chronological research log (17 rounds). If picking
this back up, start here rather than reading top to bottom.

### The headline finding

1901's column-number detection problem was never really a detection-
calibration problem. The root cause is physical: the source scans come
from a bound volume with its own binding/gutter crease, and pages don't
lie flat when photographed. Confirmed by direct visual inspection
across 6 independently-checked real pages (different enumerators/
townships/sub-districts) - not a one-off, it's present on every page in
this reel. All the early rounds (2-13) were calibration workarounds
built before this was known; Round 14 found the real cause and started
fixing pixels instead of working around them.

### What's proven and built

- **`core/page_dewarp.py`** (new module, research-only, not wired into
  production): `detect_page_crease_x()` (blind crease finder),
  `find_crease_x_near_prior()` (prior-guided, no confidence gate),
  `dewarp_page_at_crease()` (per-column vertical `cv2.remap` correction
  using `core/row_segmentation.py`'s wide-anchor trend primitives).
- **Validated result on `z000077117`** (the one page with full 33-column
  ground truth): column-number match rate 39% -> 70% (dewarp +
  simple piecewise n_zones=3 on top), via a real, honestly-verified
  pipeline - not the initial 67% claim, which turned out to be a real
  extrapolation bug that corrupted the image while still moving the
  metric the right direction (Round 14's central cautionary tale: Jon
  caught it by looking at the actual rendered image, not from the
  number).
- **The crease sits on a nameable, structural feature**: the printed
  column 15/16 boundary (Nationality/Religion on this form) - not just
  "somewhere near the middle."
- **Confirmed real page-to-page placement shift** on the scanner
  (~20-40px spread in table-left position across 5/6 pages checked,
  Round 17) - a genuine physical variation, not measurement noise.

### What's NOT resolved - pick up here

1. **Only 1 of 6 pages has real, measured table bounds**
   (`z000077117`, from its human-confirmed sidecar). The other 5 used
   rough visual guesses this session, and Round 16/17 showed that's
   likely why the column-boundary-fraction crease prediction didn't
   generalize cleanly to them (predictions were 30-60px off on pages
   checked visually). **Next step**: get real per-page table bounds
   (Round 17's leftmost-strong-line method got close on 5/6 pages,
   19px off on `z000077117` where it's checkable - reusable as a
   starting point, not a finished solution) and re-run Round 16's
   crease-position validation with those instead of guesses.
2. **Blind detection (both crease and table-border) is unreliable
   without a prior.** Documented failing 3+ separate times this
   session in slightly different shapes (Round 14's ruling-line-vs-
   crease confusion, Round 17's row-number-margin-vs-outer-border
   confusion, Round 17's photo-frame-edge confusion) - the consistent
   pattern is that multiple real vertical lines sit close together on
   these forms, and "pick the strongest/first one" is the wrong
   selection principle without an additional constraint. Every
   confirmed success this session involved a prior narrowing the
   search first.
3. **`z000077134` is a recurring problem page** - weakest crease signal
   in Round 16 (ratio 1.33, below the 1.5 threshold) AND the one page
   where the corrected table-border search still failed in Round 17.
   Plausibly a genuinely lower-contrast scan, not investigated further.
4. **Trend refinement beyond 3 wide anchors was tried and didn't help**
   (Round 15) - more sample points and smooth polynomial fits both
   underperformed the simple 3-anchor baseline. Don't re-attempt this
   exact direction without new evidence; the noise floor is in
   `find_number_row_band()`'s own measurement, not the fitting method.
5. **Nothing is wired into production.** `core/page_dewarp.py` and
   everything downstream of it is research/prototype code, consistent
   with this whole investigation's standing scope constraint.

### Suggested order for the next session

1. Get real table bounds for the other 5 confirmed-crease pages (manual
   visual confirmation, like `z000077122`'s column-15/16 check in
   Round 16, is slower but reliable - matches how `z000077117`'s own
   ground truth was originally produced).
2. Re-run the crease-position prediction (column-15/16-boundary
   fraction) against those real bounds - see whether the 30-60px
   errors from Round 16 close up once the bounds themselves are trusted.
3. If that closes the gap, re-run Round 14's full dewarp+piecewise
   validation pipeline on all 6 pages (only visual/score-based checks
   are possible on 5 of them, no full ground truth) and decide whether
   this is ready to propose as a production integration step.
4. `z000077134` may need separate handling (manual crease/border pin,
   or accept it as a residual failure case) rather than forcing the
   same automated pipeline to work on it.

## Prior art - already tried, already removed

`core/auto_sidecar.py` already contains `_detect_header_number_blobs()`
(2026-07-27) - a simple density-relative-to-peak column-number blob
detector, built to shift `locate_columns()`'s search anchor toward a
detected number. It was tried and explicitly removed: it didn't fix the
outlier it was built for (`z000017634`, off by 50-175px on Sex/Relationship
to Head), and it *introduced new errors* on pages that were already
accurate. Stated root cause in `locate_columns()`'s own docstring: "a 1D
column-density profile can't cleanly separate individual digits at this
print resolution."

That z000017634 outlier is instead caught today by
`locate_table_boundary()`'s row-height plausibility check and sent to
whole-page quarantine - a deliberate "acceptable to hit manual review
rather than ship a fragile fix" decision, not a claim that anchoring
itself is unworkable.

**Why this research pass isn't just repeating that failure**: the row-number
work hit the identical class of problem first (a naive coverage-based line
exclusion also failed), and what actually worked needed real structural
exclusion (longest-run-length, not raw density) plus a validation/filtering
layer, not just a better raw detector. `_detect_header_number_blobs()` had
neither. This research re-tests the underlying idea with those lessons
applied, rather than assuming the old failure settles the question.

## Validation setup

Real reference page: `data/outputs/reference_pipeline_prerefactor/dewarped/
e078_e001946617_dewarped.jpg` (1911 census, 3549x2168), cross-checked against
its own human-confirmed sidecar (`.../row_segmentation/
e078_e001946617_dewarped_sidecar.json`) - real `mask_keep_ranges` for 5
columns (Name, Sex, Relationship to Head, Age, Birthplace), manually
confirmed via `ui/row_segmentation_ui.py`, not derived from anything tested
here. This is genuine ground truth, not a self-consistency check.

Reproducible via `diagnostics/test_column_number_anchor.py`.

## Findings, in the order discovered

1. **The geometric principle holds.** The printed column-number row is
   simple sequential integers on this form (1, 2, 3, ... - no sub-lettered
   groups like the "4a 4b 4c" pattern `_detect_header_number_blobs()`'s own
   docstring mentions seeing on some OTHER form). Each number sits centered
   over its column, confirmed by direct visual check.

2. **Naive full-header-width blob detection is noisy**, same failure shape
   as the earlier attempt: raw detection produced one blob spanning ~1400px
   (clearly wrong) plus assorted noise specks, alongside several blobs that
   landed suspiciously close to the real column centers.

3. **Root cause #1 - wrong y-band, not a detection bug.** The number row's
   *true* pixel band (y=592-604 on this page) sits just below a dashed
   sub-header guide-line (a category-label underline, e.g. under
   "Citizenship, Nationality and Religion") at y=583-590. An initial guess
   at the y-band included that guide-line, and its broad near-continuous ink
   contribution merged large stretches of otherwise-separable digits
   together. Narrowing the band to exclude it (y0: 583->592) eliminated the
   1400px merged blob entirely. **Lesson: the y-band must be calibrated per
   template by direct visual check, same discipline as every other
   measurement in this pipeline - it is not a fixed offset that transfers
   between forms or even between sections of the same header.**

4. **Root cause #2, tested and FALSIFIED - not a gap-bridging artifact.**
   A few wide merges remained after fixing the y-band, in a section where
   columns are printed very close together (columns 26-32 on this form).
   Hypothesis: `close_gap_px` bridging was merging genuinely-separate
   numbers across narrow real gaps. Tested directly: setting
   `close_gap_px=0` (no bridging at all) left the worst merges completely
   unchanged (still ~116px). The 5 known-column accuracy was unaffected
   either way (2-5px both times). **Conclusion: this remaining noise is
   genuinely touching/near-touching ink in the source print at tight
   column spacing - not fixable by threshold tuning, confirmed by directly
   testing the hypothesis rather than assuming it.**

5. **Filtering, not perfect raw separation, is what makes this reliable.**
   Rather than trying to parse every blob on the page correctly, scoping
   the search to each column's own template-provided approximate x_frac
   position (±60px window) and taking the nearest raw blob within that
   window found the correct candidate for **all 5/5 known columns**, with
   the surrounding noise never entering into it at all. This mirrors the
   task's own design pattern exactly: template gives the prior, the number
   signal narrows the window further, a downstream step (here: nearest-
   candidate selection; in a real integration: the existing ruling-line
   detector) does the final measurement.

## Results (via `diagnostics/test_column_number_anchor.py`)

| Column | Real center | Template approx | Detected (filtered) | Delta from real |
|---|---|---|---|---|
| Name | 365.5 | 370.0 | 360.5 | 5.0px |
| Sex | 699.5 | 703.6 | 695.5 | 4.0px |
| Relationship to Head | 786.5 | 791.2 | 783.5 | 3.0px |
| Age | 1095.5 | 1098.5 | 1093.5 | 2.0px |
| Birthplace | 1184.5 | 1184.8 | 1181.5 | 3.0px |

5/5 matched, deltas 2-5px (mean 3.4px) - on a table ~3442px wide, all well
within a reasonable search-window tolerance for narrowing
`_find_vertical_ruling_line()`'s existing search radius.

## What was built

- `core/row_segmentation.py`:
  - `detect_column_number_centers(image, x0, x1, y0, y1, ...)` - pure
    sensor, returns `[(center_x, width_px), ...]` for every plausible
    number-blob found in a given band. Does not know which column
    anything belongs to, does not filter, does not decide.
  - `nearest_number_candidate(candidates, expected_position, max_distance)`
    - the filtering step (finding #5 above). Shared helper, usable for
      both row- and column-axis anchoring.
- `diagnostics/test_column_number_anchor.py` - reproduces the validation
  above against the real reference page.

Both are dependency-free (PIL + numpy only), matching
`core/row_segmentation.py`'s existing module-wide discipline.

## Round 2 (same day) - full-sequence recovery, cross-year testing, inversion, auto-calibration

### Full-header-sequence recovery (still on e078_e001946617, 1911, 41 real columns)

Beyond the 5 spot-checked columns above, ran detection across the ENTIRE
header width and visually audited hit/miss against all 41 printed numbers
(zoomed segment-by-segment, since a first low-res pass was unreliable
enough to mis-tally the results - re-done at higher zoom):

- **~34/41 (~83%) clean hits.**
- One merge cluster (columns 19-21) - the already-characterized touching-
  ink issue from finding #4 above.
- **New finding: a hard-miss cluster at the page's outer edge (columns 36,
  38-41)** - nothing detected there at all, not a wrong blob, an absence.
  Consistent with warp/contrast degradation being worst at page edges
  (same pattern already seen elsewhere this session, e.g. the row-number
  work's ink-blot case).
- **Double-digit numbers are NOT specifically problematic** - hits span
  10 through 35 cleanly. Failures cluster by *position* (tight-packing
  zone, page edge), not by digit count.
- Detected blobs preserved correct left-to-right order throughout - no
  out-of-sequence detections.

### Cross-year test #1 - 1931 (real ground truth, genuinely different form)

Real reference: `data/outputs/row_segmentation/e011717826_sidecar.json`
(source `lac_pull_1931_batch1/raw_jpgs/e011717826.jpg`, 41 confirmed
columns, human-masked). This form has **sub-lettered column groups**
("4 / 4a / 4b / 4c" for Section/Township/Range/Meridian) - the exact
complication flagged as untested above.

Checked 10 columns spread across the form: 8 landed in the 1.0-8.5px
range (comparable to 1911's 2-5px, a bit noisier but clearly usable),
including all four sub-lettered columns (Section 8.5px, Township 1.5px,
Range 6.5px, Meridian 3.5px). **Two real outliers: Name 12.0px, Sex
24.5px** - not yet root-caused.

**Sub-lettered columns turned out NOT to be a real problem** - the
detector doesn't need to read "4a" vs "4b", it only needs a plausible
blob near each column's own position, and the printed sub-letter marks
still produce one independently.

### Cross-year test #2 - 1926, raw/uncalibrated (no mask exists)

Different form family (Prairie Provinces census, already flagged
elsewhere this session as structurally distinct from the standard
Census of Canada family) with NO manual reference mask to check against
- deliberately tested "blind" to see how a first-attempt, uncalibrated
y-band performs, not to get a validated accuracy number.

Result with a rough manual y-band guess (975-1000): **poor** - only 2 of
~12 visible numbers detected in one segment (2 and 8; 1,3,4,5,6,7,9,10,
11,12 all missed). Diagnosed as likely: numbers sit at a different
vertical position relative to the header/row divider than on 1911/1931,
and a diagonal handwritten enumerator stroke crosses part of the row.
**Not a failure of the technique - a failure of NOT calibrating the
band the same careful way the other two years were.** See auto-
calibration below for the resolution.

### Inversion experiment - negative result, confirmed

Tested whether running detection on an inverted copy of the image (with
the ink-comparison direction correspondingly flipped) changes anything.
Result: essentially no difference (51 vs 52 blobs, identical known-
column deltas, same worst-case merge width). Expected in hindsight -
Otsu thresholding adapts to find the optimal split regardless of ink/
background polarity, so simple inversion is close to a no-op for this
technique. **Do not spend further effort on inversion.**

### Automated y-band calibration - `find_number_row_band()`

The manual y-band calibration process (visual inspection + row-density-
profile reading, done by hand for 1911 and 1931 above) doesn't scale -
it's exactly the kind of per-image tuning a real sensor shouldn't need.
Built `find_number_row_band()`: sweeps a sliding window across a broader
search region, scores each candidate via `_score_number_row_band()`
(rewards plausible blob count + digit-like widths, penalizes a dominant
merged blob - i.e. codifies the SAME signals used to manually diagnose
band quality above), keeps the best-scoring candidate.

**Validated against all three samples above:**
- 1911: auto-found band (591, 603) - within 1px of the hand-calibrated
  (592, 604), same 51 blobs.
- 1931: auto-found band (654, 666) - falls inside the hand-calibrated
  (650, 678) range, 77 blobs, solid positive score.
- **1926 (the case that failed manually): auto-search found (992, 1004)
  - substantially better than the manual guess**, recovering columns
  1,2,4,5,6,7,8,9 cleanly (vs. only 2,8 by hand) - though some
  handwriting still bleeds into later/wider columns, and the score
  correctly stayed negative, honestly signaling reduced confidence
  rather than reporting false certainty.

This directly answers "can the calibration be automated" - yes, at
least well enough to match hand-calibration on two forms and
meaningfully outperform an uncalibrated guess on a third, without
retraining or OCR. Not proven to be robust on every conceivable form
(no printed number row at all, e.g., is untested - expected to just
score consistently low, not silently misfire, but not confirmed).

## What was deliberately NOT done

- `core/auto_sidecar.py`'s `locate_columns()` / `_find_vertical_ruling_line()`
  were not touched. No production wiring.
- The 1931 Name/Sex outliers (12.0px/24.5px) were not root-caused.
- The 1926 handwriting-bleed contamination on wider/later columns (visible
  even in the auto-calibrated band) was not further investigated.
- The "progressive search corridor" idea (using confirmed measurements to
  predict subsequent columns' positions, propagating left-to-right) was
  not prototyped this round.
- No attempt at 1901/1906/1921 yet.
- No OCR - stayed blob/geometry-driven throughout, per direct instruction.

## Round 3 (same day) - boundary PROJECTION from header-centre anchors

Prototyped item #2 from Round 2's "suggested next steps" above: rather
than just narrowing a search window around each column independently,
chain the header-number centres into an actual boundary PREDICTION,
using the fact that a printed number sits centred inside its own column.

**Geometry**: given a known left edge `B[0]` and detected centre `C[i]`
for column `i`, the column's right edge is the reflection of the known
left edge through the centre: `B[i] = 2*C[i] - B[i-1]`. Deliberately
NOT `(C[i]+C[i+1])/2` (a same-width assumption the task spec explicitly
ruled out) - this formula makes no assumption about neighboring column
widths at all.

Two variants tested:
- **Variant A (pure projection)**: `B[i]` chained from projections only,
  never corrected - measures how much drift accumulates uncorrected.
- **Variant B (detector-corrected)**: after each projection, the
  EXISTING `core/auto_sidecar.py: _find_vertical_ruling_line()` (reused
  as-is, not reimplemented, not modified) searches a ±30px corridor
  around the projected position; if it finds a real ruling line, that
  MEASURED position (not the raw projection) feeds the next step.

Reproducible via `diagnostics/test_column_boundary_projection.py`,
against the same 1931 reference (`e011717826_sidecar.json`, 42 real
columns) used in Round 2's cross-year test.

### Failed hypothesis: naive "every matched column is trustworthy" run-building

First pass matched all 42 real columns to a nearby detected candidate
(within a 60px window) and treated the entire page as one giant "clean
run" for projection, since every column found *some* nearby candidate.
Variant A degraded from single-digit/low-20s px error across the first
~20 columns to **120-160px** from "Yea of Naturalization" onward, with
Variant B (despite detector correction) still degrading to 80-107px in
the same region - a 126px closure error at the end of the page.

**Root cause: one bad ground-truth point, not a geometry flaw.**
"Yea of Naturalization" has a known remask-artifact reference range
(`mask_keep_ranges: [[2091,2151],[2152,2298]]` - two disjoint ranges,
the same "remask noise, not a real split" class of unreliable manual
annotation `locate_columns()`'s own docstring already flags for a
different column/form). Its computed real-center is unreliable, and its
matched detected candidate carried a 34.5px delta - dramatically larger
than every genuinely good match nearby (1-8px, e.g. Year of Immigration
6.0px, Nationality 4.0px, Racial Origin 1.0px). Once fed into a
*recursive* projection, this single bad anchor poisoned every column
downstream of it, and even Variant B's per-step detector correction
couldn't fully recover within its search radius.

**Fix**: added a match-quality gate (`MATCH_QUALITY_MAX_DELTA_PX=15`) -
a matched candidate whose delta from the reference center exceeds this
is treated as UNRELIABLE and dropped to `None` (a gap), same "don't
invent, flag explicitly" discipline already applied to missing sensor
data, now also applied to unreliable REFERENCE data. This is a
refinement worth generalizing: **"clean consecutive anchor run" must
also filter against known-bad ground truth, not just missing detector
output** - a real page has no way to know its own ground truth is bad,
but a validation harness does, and should say so rather than average
over it.

After the gate: 41/42 columns confidently matched (1 flagged and
correctly gapped), splitting the page into 2 clean runs of 21 and 20
columns rather than one poisoned 42-column run.

### Results, after the quality gate

| Run | Columns | Variant A mean / median / max err (px) | Variant B mean / median / max err (px) | Variant A closure err |
|---|---|---|---|---|
| Dwelling House .. Year of Immigration | 21 | 25.8 / 23.0 / 61 | 11.4 / 2.0 / 48 | 61.0px (3.4% of 1800px run span) |
| Nationality .. Other Cause | 20 | 14.9 / 17.0 / 29 | 7.7 / 3.0 / 26 | 3.0px (0.2% of 1535px run span) |
| **Combined** | 41 | **20.5 / 20.0 / 61** | **9.6 / 3.0 / 48** | - |

Corridor capture-rate sweep (both runs combined, 41 boundaries, measured
against the raw projection `proj_B` before detector correction):

| Corridor width | Boundaries captured |
|---|---|
| ±5px | 17/41 (41.5%) |
| ±10px | 22/41 (53.7%) |
| ±20px | 32/41 (78.0%) |
| ±30px | 39/41 (95.1%) |
| ±50px | 41/41 (100.0%) |

(Not normalized to local column width - columns on this form range
roughly 40-350px wide, so a fixed-px corridor is proportionally tighter
on narrow columns; not yet re-expressed as a width fraction.)

### Two further findings (why Variant B doesn't uniformly dominate A)

1. **Run 1's first column carries a real detection offset, not a gate
   artifact.** "Dwelling House" (the very first column in its run)
   already shows a 25px Variant-A error on step 1 alone: its detected
   blob centre (278.5) sits 12.5px from the column's true geometric
   centre (266.0) - and the projection formula doubles any centre
   error into the projected edge (`2 * 12.5 = 25`), exactly matching
   the observed error. Variant B *recovers* from this over the next
   ~10 columns (errors fall from 48px to ~0-2px) as the detector's own
   corrections progressively wash out the bad start - a real
   demonstration of the "prior narrows the window, detector stays
   authoritative" design working as intended, but only once several
   correction steps have elapsed.

2. **Variant B is not always better than Variant A, and can be worse.**
   In the last 5 columns of run 2 (Illness, Accident, Strike or
   Lockout, Layoff, Other Cause - a stretch of narrow, tightly-packed
   columns), Variant A stayed low (5, 6, 5, 6, 3px) while Variant B rose
   (15, 26, 25, 26, 23px). Likely explanation: the ±30px search corridor
   can catch an *adjacent* ruling line when columns are narrow and
   closely spaced, "measuring" a real line that is nonetheless the
   wrong one - and because Variant B feeds its own (possibly wrong)
   measurement into the next step, one bad lock-on propagates forward
   the same way a bad ground-truth anchor did above. **Detector
   correction is not a strict safety net against projection drift; it
   trades unbounded drift for a different failure mode (wrong-line
   lock-on) that gets worse, not better, as column spacing tightens.**
   Not yet resolved - a narrower/adaptive corridor (scaled to local
   column width rather than a fixed ±30px) is the natural next thing to
   try, not attempted this round.

### What this means for integration (not implemented)

The projection is a usable PRIOR (corridor capture rate 78-100% at
±20-50px, well inside `_find_vertical_ruling_line()`'s existing search
radius conventions) but is **not accurate enough on its own to replace
a measured boundary** - median error is reasonable (3.0px with detector
correction) but the max/tail cases (26-61px) are exactly the kind of
silent bad values a "Prior becomes Result" design would ship without
warning. This confirms the task's own starting philosophy was correct:
`Prior -> narrowed search corridor -> existing CV detector -> measured
result`, not `Prior -> final boundary`. A future integration into
`locate_columns()`'s RESOLUTION CASCADE should treat this projection as
another corridor-narrowing input alongside the affine tiers already
there, not a new authoritative tier of its own.

### What was deliberately NOT done (round 3)

- `core/auto_sidecar.py`'s `locate_columns()` was not touched - research
  and prototype only, per the task's explicit scope constraint.
- No adaptive/width-scaled corridor was implemented (identified above as
  the natural fix for the narrow-column wrong-line-lock-on finding).
- Corridor width was not re-expressed as a fraction of local column
  width, only tested as fixed pixel widths.
- Only 1931 was re-tested with the projection formula; 1911/1926 were
  not re-run through the projection/closure-error harness (they were
  only used for the earlier centre-detection-only validation in Round 2).

## Suggested next steps (not started)

1. Root-cause the 1931 Sex/Name outliers - a wide column (Name) and an
   unexplained 24.5px miss (Sex) are the two loosest threads in an
   otherwise clean cross-year result.
2. ~~Prototype the progressive/sequential search-corridor idea~~ - DONE,
   see Round 3 above. Remaining: an adaptive/width-scaled corridor to
   address the narrow-column wrong-line-lock-on finding.
3. Design how a `DocumentTemplate` would carry this calibration (parallel
   to `row_number_column_x_frac`/`row_number_alignment` already added for
   rows) - `find_number_row_band()`'s existence changes this: a template
   may only need a coarse SEARCH REGION (e.g. metadata_bottom to
   table_top) rather than an exact hand-measured y-band, since the band
   itself can now be found automatically.
4. Decide the actual integration shape into `locate_columns()`'s existing
   RESOLUTION CASCADE (measured / affine_override / affine_predicted /
   locally_interpolated) - likely a new tier, evaluated with the same
   real-reference-sample discipline used for the affine fix (`docs/CODE_MAP.md`
   / `locate_columns()`'s own docstring has the full cascade history).
5. ~~Test 1901/1906/1921 for completeness~~ - 1901/1921 DONE, see Round 4.
   1906 still untested (no ground truth located yet).
6. Apply the "gate against known-bad ground truth, not just missing
   sensor data" lesson from Round 3 back to the row-number-anchor
   validation work, if any similar remask-artifact rows exist there too
   - not checked yet.

## Round 4 (same day) - multi-year generalization test

Direct response to "does this work across all years, not just the one
1931 sample" - ran the same auto-calibrated-band -> detect -> quality-
gate -> Variant A/B projection pipeline from Round 3 against every real,
independently-confirmed reference available across the corpus, not just
1931. Reproducible via
`diagnostics/test_column_boundary_projection_multiyear.py`.

Two tiers of real ground truth exist in this corpus, and mixing them up
caused a real bug (see below), so the test now keeps them explicitly
separate:
- **Full-layout** (every real column in the table is known - the
  projection chain can be tested end-to-end): 1901 (`z000077117`, 33
  columns), 1926 (`e001926997`, 10 columns), 1931 (`e011717826`, 42
  columns, same sample as Round 3).
- **Sparse spot-check** (only ~4-5 non-contiguous columns known - not
  enough to chain a projection, only a nearest-candidate accuracy
  check): 1911 (`e078_e001946617`), 1921 (`1921_022-E002880409`), a
  second 1931 sample (`1931_174-e011707164`).

### Harness bug found and fixed: sparse ground truth silently treated as contiguous

First pass classified the 1911 sample as "full layout" because its
sidecar has a populated `columns` dict - but that dict only holds 5
spot-checked columns (Name/Sex/Relationship to Head/Age/Birthplace), not
the form's real ~30+ columns. The run-detection logic doesn't know
that - it just saw 5 known x-positions and, since 2 of them (Relationship
to Head -> Birthplace) happened to both survive the quality gate, treated
them as adjacent and projected straight across every real, unlisted
column in between. Result: a nonsense 226px "closure error" that had
nothing to do with detection or projection accuracy - it was measuring
against a bridge the data never supported.

**Fix**: moved 1911 into the sparse spot-check tier (its actual
ground-truth completeness), where the harness now only checks
individual columns against their nearest detected candidate, never
chains a projection across them. **New rule, alongside Round 3's
"gate against known-bad ground truth"**: before trusting a sidecar as a
full-layout reference, confirm its `columns` dict actually matches the
form's real column count/`column_order`, not just that consecutive
known entries happen to look adjacent.

### Results, after the fix

| Year | Sample | Real cols | Matched | Runs | Variant A mean/median/max | Variant B mean/median/max |
|---|---|---|---|---|---|---|
| 1901 | z000077117 | 33 | 13 (39%) | 7 (mostly len-1, 4 usable) | 14.2 / 18.0 / 38.0px | 14.2 / 18.0 / 38.0px (identical to A) |
| 1926 | e001926997 | 10 | 5 (50%) | 3 (2 usable) | 11.2 / 13.0 / 23.0px | 5.8 / 5.0 / 13.0px |
| 1931 | e011717826 | 42 | 41 (98%) | 2 | 27.5 / 30.0 / 66.0px | 25.1 / 22.0 / 76.0px |

Combined (full-layout years, 55 boundaries): Variant A mean=23.9 /
median=23.0 / max=66.0px; Variant B mean=21.7 / median=14.0 / max=76.0px.
Corridor capture: ±5px 29.1%, ±10px 41.8%, ±20px 58.2%, ±30px 69.1%,
±50px 85.5%.

Spot-check tier (nearest-candidate only, no projection):

| Year | Sample | Matched | Mean delta | Max delta |
|---|---|---|---|---|
| 1911 | e078_e001946617 | 4/5 | 9.2px | 27.0px (Sex) |
| 1921 | 1921_022-E002880409 | 3/4 (Birthplace: no candidate) | 1.7px | 3.5px |
| 1931b | 1931_174-e011707164 | 4/4 | 5.0px | 5.5px |

### Honest conclusion: generalization is real but weaker than the single-sample result suggested

1. **Match rate (how many real columns get ANY confident matched
   centre) varies far more by year than Round 2/3 alone would suggest.**
   1901's auto-calibrated band only matched 13/33 (39%) real columns -
   dramatically worse than 1931's 41/42 (98%). This isn't a band-
   calibration failure in the "wrong y-range" sense (the auto-search
   score, 1.92, is in the same range as every other year tested) - the
   band it found is plausible, the detector just isn't reliably finding
   or precisely centering a blob for every real column on this specific
   form. Not yet root-caused; the most likely candidates (not
   confirmed) are print/scan quality of this particular 1901 sample or
   genuinely different digit spacing/size on the 1901 form family versus
   1926/1931.
2. **New, unexplained finding: on 1901, Variant B never once corrected
   anything** - its mean/median/max are IDENTICAL to Variant A's,
   meaning `_find_vertical_ruling_line()` found no real line inside the
   search corridor for every single projected boundary tested on this
   page. Either this form's ruling lines are too faint/thin for that
   detector's existing threshold, or the low match rate above meant the
   corridors it searched were already too far off target to catch a
   real line. Not root-caused this round.
3. **Cross-year corridor capture is meaningfully lower than the
   single-1931-sample number reported in Round 3** (58.2%/69.1%/85.5%
   at ±20/30/50px here, vs. 78.0%/95.1%/100% in Round 3). The single
   1931 sample was, in hindsight, closer to a best case than a
   representative case - a real and useful thing to know before
   considering production integration, and exactly why this round's
   test was worth running before trusting Round 3's numbers as
   generalizable.
4. **1926 and the second 1931 sample remain solid** - Variant B clearly
   helps on 1926 (11.2px -> 5.8px mean) and the 1911/1921/1931b spot-
   checks are all sub-10px except one outlier (1911's Sex, 27.0px).

### What this changes about integration readiness

Round 3 alone might have read as "ready to prototype a production
integration tier." This round's honest answer is **not yet** - the
technique clearly works on well-behaved samples (1926, 1931, most spot-
checks) but 1901's low match rate and Variant B's complete non-
correction there are open, unexplained problems on a full census year,
not an edge case. Recommend root-causing #1 and #2 above (ideally with
a couple more 1901 samples, since this was only one page) before
treating the projection as reliable across the full year range this
pipeline needs to support.

### What was deliberately NOT done (round 4)

- 1906 still has no located ground truth in this corpus - untested.
- The two open findings above (1901 low match rate, 1901 Variant B
  never correcting) were observed and reported, not root-caused - would
  need dedicated investigation (e.g. visually auditing 1901's raw
  detected blobs the way Round 2 did for 1911's touching-ink cluster).
- No adaptive/width-scaled corridor (still open from Round 3).
- `core/auto_sidecar.py`'s `locate_columns()` was still not touched.

## Round 5 (same day) - line-detection-only enhancement test (root-causing Round 4's 1901 finding)

Direct follow-up to Round 4's open question #2 (Variant B never once
corrects anything on the 1901 sample). Jon proposed a cv2 preprocessing
pipeline (mild `fastNlMeansDenoising` -> CLAHE -> vertical-morphology
closing) as a candidate fix, with an explicit framing: **judge it only
on whether it helps the existing ruling-line detector FIND a line, not
on whether the resulting image would still be usable for OCR/extraction**
("this is not a profile for extraction quality" - the morphology-closed
output is expected to look worse to a human, that's fine, out of scope
for this test).

Built `diagnostics/test_line_detection_enhancement.py` to isolate this
cleanly: for every real ground-truth boundary, call the EXISTING,
unmodified `core/auto_sidecar.py: _find_vertical_ruling_line()`
centered exactly ON the real boundary (search_radius=15px, not a
drifted projection like Round 3/4) - so a miss here can only mean the
line itself isn't detectable at that exact location, removing the
"was the search window even pointed at the right place" confound
entirely. Ran raw vs. enhanced on the same three full-layout, real-
ground-truth samples used in Round 4 (1901/1926/1931). cv2 (already a
project dependency elsewhere, e.g. `core/auto_sidecar.py`,
`core/image_analysis.py` - not a new addition) used directly in this
research script only; `core/row_segmentation.py`'s own PIL+numpy-only
discipline is unaffected since nothing there changed.

### Results

| Year | Raw found | Enhanced found | Effect |
|---|---|---|---|
| 1901 | 0/33 (0%) | 0/33 (0%) | **No change - enhancement did not help at all** |
| 1926 | 2/10 (20%), 2.0px delta | 2/10 (20%), 4.0px delta | Neutral to mildly worse |
| 1931 | 18/42 (43%), 1.9px delta | 29/42 (69%), 2.1px delta | **Real, substantial improvement - +11 columns recovered, same accuracy when found** |

### Conclusion: 1901's problem is NOT contrast/noise - it's structural

This cleanly answers Round 4's open question. If 1901's lines were
merely faint/low-contrast, denoise+CLAHE+morphological closing - a
fairly aggressive enhancement stack - would have recovered at least
some of them, the way it recovered 11 additional columns on 1931.
Instead: **zero effect, zero lines found, before or after, anywhere on
the page.** The most likely explanation (not yet directly confirmed by
visual inspection) is that this 1901 form genuinely lacks a continuous
printed vertical ruling line between every column - i.e. there may be
nothing there for ANY line-detection approach to find, not a detection
failure. This reframes 1901 from "detector needs tuning" to "this form
family may need boundary evidence from a source OTHER than ruling
lines" - the header-number-centre projection itself (Variant A) becomes
more important for 1901-style forms specifically, since it doesn't
depend on a ruling line existing at all.

**On 1931, this is a genuinely useful, validated finding independent of
the 1901 question**: the enhancement pipeline meaningfully improves
ruling-line recall (43%->69%) at no accuracy cost. Worth carrying
forward as a candidate improvement to `_find_vertical_ruling_line()`'s
search step specifically (not a blanket image-preprocessing change),
if/when this work moves toward production integration.

### What was deliberately NOT done (round 5)

- Did not visually confirm the "1901 may lack real vertical ruling
  lines" hypothesis by eye - inferred from the enhancement having zero
  effect, not directly verified against the source image.
- Did not test the enhancement on the two 1911/1921 spot-check samples.
- Did not test a vertical-only vs. combined horizontal+vertical version
  of the morphology step (only vertical was used, since only vertical
  column-divider lines are relevant here).
- Did not wire this into `core/auto_sidecar.py` - still research-only,
  per the same standing scope constraint.

## Round 6 (same day) - gradient-based detection: Round 5's conclusion was WRONG

Jon's direct hypothesis after seeing Round 5: the existing detector
(`_find_vertical_ruling_line()`) looks for a DARK RIDGE - Otsu threshold
then longest-run-length of "this pixel is dark enough." A hairline-thin
printed rule can fail that test even where a real, sharp brightness
CHANGE exists on both sides of it, because absolute darkness and edge
sharpness are different things - a threshold-based method is polarity/
contrast-dependent in a way a GRADIENT isn't. Proposed testing a
Sobel/Scharr `dx=1` vertical-edge gradient inside the same search
corridor instead of a dark-pixel threshold.

Added `find_vertical_line_via_gradient()` to
`diagnostics/test_line_detection_enhancement.py` (research-only,
alongside the existing raw/enhanced comparison, same real-ground-truth-
centered methodology as Round 5): computes `cv2.Scharr(dx=1)` on the
search corridor, averages the absolute gradient down each column,
declares "found" if the strongest column beats the local median by a
1.5x ratio (a real outlier, not just corridor noise).

### Results - decisive, across all three years

| Year | Raw found | Enhanced found | **Gradient found** |
|---|---|---|---|
| 1901 | 0/33 (0%) | 0/33 (0%) | **18/33 (55%), mean delta 11.0px** |
| 1926 | 2/10 (20%) | 2/10 (20%) | **10/10 (100%), mean delta 3.8px** |
| 1931 | 18/42 (43%) | 29/42 (69%) | **42/42 (100%), mean delta 2.8px** |

**Round 5's conclusion was wrong, and it's important to say so plainly:**
Round 5 inferred that 1901 likely lacks continuous printed vertical
ruling lines at all, because an aggressive denoise+CLAHE+morphology
enhancement had zero effect. That inference doesn't survive this
result - the gradient method found a real, clear edge signal (ratios
1.5-3.9x background) at 18 of 33 real boundaries on the SAME page,
using the SAME source pixels. The lines are there. The dark-ridge/
threshold-based detection approach just couldn't see them. Jon's
polarity hypothesis was correct; my structural-absence hypothesis was
not - noted here so this doesn't get treated as settled fact later.

On 1926 and 1931, the effect is just as strong: 100% of real boundaries
found on BOTH years, a dramatic jump from the best previous result
(69% on 1931 via the CLAHE/morphology enhancement, which itself doesn't
help at all on 1901). Localization accuracy when found stays reasonable
(2.8-3.8px on 1926/1931, matching prior methods) but is visibly worse
on 1901 specifically (11.0px mean) - consistent with something already
known about this specific 1901 form/page from earlier row-number-anchor
work this session (its physical printed border was already found to be
non-straight on some rows), rather than a new problem with the gradient
method itself.

### What this changes

This is a stronger core building block than anything tried in Rounds
3-5 - **it doesn't just help 1901's problem case, it strictly dominates
the existing detector on every year tested, at every column, with only
a modest accuracy cost on the one page that's already known to have a
physically imperfect print.** The natural next step this opens up (not
done yet, deliberately deferred to get this write-up in first) is
re-running Round 3/4's full Variant A/B projection + closure-error
harness with `find_vertical_line_via_gradient()` swapped in for
`_find_vertical_ruling_line()` as Variant B's detector step - which
could plausibly resolve most of Round 4's open cross-year generalization
gap, since the root cause there was never confirmed but this result
makes "the detector couldn't see the line" a much more likely
explanation than "the projection math is imprecise."

### What was deliberately NOT done (round 6)

- Did not re-run the Round 3/4 projection/closure-error harness with
  the gradient detector substituted in - proposed as the next step,
  not started.
- `min_peak_ratio=1.5` was not tuned/swept - picked as a reasonable
  first guess (clear outlier vs. background) and not stress-tested
  against a case designed to produce false positives (e.g. handwriting
  strokes crossing the corridor).
- Sobel vs. Scharr was not compared head-to-head (Scharr used
  throughout, on the general principle that it's a more accurate small-
  kernel approximation - not empirically verified here).
- Still research-only - `core/auto_sidecar.py`'s
  `_find_vertical_ruling_line()` was not modified or replaced.

## Round 7 (same day) - gradient detector wired into the full projection harness

Did the "next step" flagged at the end of Round 6: swapped
`find_vertical_line_via_gradient()` in for `_find_vertical_ruling_line()`
as Variant B's correction step inside
`diagnostics/test_column_boundary_projection_multiyear.py`, and re-ran
the full Round 4 multi-year harness unchanged otherwise (same auto-
calibrated bands, same quality-gated column-number matching, same
±30px search radius, same three full-layout years).

### Results - large win on 1926/1931, a real regression on 1901

| Year | Variant B (old detector) | Variant B (gradient detector) | Detector found-rate |
|---|---|---|---|
| 1901 | 14.2 / 18.0 / 38.0px | **23.1 / 22.0 / 44.0px (WORSE)** | 9/10 (was ~0/10) |
| 1926 | 5.8 / 5.0 / 13.0px | **3.8 / 3.0 / 10.0px** | 4/4 (100%) |
| 1931 | 25.1 / 22.0 / 76.0px | **3.6 / 2.0 / 37.0px** | 41/41 (100%) |

Combined (55 boundaries): Variant B mean dropped from 21.7px (Round 4,
old detector) to **7.2px**; median from 14.0px to **3.0px**. Corridor
capture at ±20px jumped from 58.2% to **90.9%**, at ±30px from 69.1% to
**96.4%**. This is a large, genuine improvement overall, driven almost
entirely by 1926 and 1931 essentially solving the "does the projection
work" question for those years (near-perfect detector found-rate, low
single-digit-px error).

### 1901 got WORSE, and the reason is informative, not a contradiction of Round 6

Round 6 showed the gradient detector finds real edges at 1901 boundaries
that the old detector missed entirely. That's still true here (found-
rate 9/10, up from near-zero). But Variant B's ERROR went up, not down,
because **1901's bottleneck was never really about line-detection
sensitivity in the first place - it's the upstream column-number-blob
match rate (13/33, 39%, from Round 4)**. When only 13 of 33 real columns
get a confidently-matched header-number centre, most of the runnable
"clean runs" are short, and the recursive projection chain feeding each
search corridor is built on shakier ground to begin with. A MORE
sensitive line detector, told to search a corridor that's already
centered somewhere not-quite-right, will confidently lock onto whatever
real edge is nearest - which on a page this dense with printed rules
(every column has one) is often the WRONG column's boundary, not the
intended one. This is the same "wrong-line lock-on" failure mode
flagged in Round 3 (narrow, tightly-packed columns), now more visible
on 1901 specifically because higher detector recall means more chances
to lock onto a plausible-but-wrong line instead of finding nothing at
all.

**Practical implication**: the gradient detector is an unambiguous
upgrade over the old dark-ridge detector and should be the default for
any future integration - but on forms like 1901 where the upstream
header-number match rate is already low, fixing the number-blob
matching step (why is 1901's match rate so much lower than 1926/1931's?
- still not root-caused, see Round 4's open item #1) matters more than
further tuning the line detector itself. The two problems are coupled:
a better-targeted search corridor makes ANY line detector's job easier,
including this one.

### What was deliberately NOT done (round 7)

- 1901's low upstream match-rate (Round 4's open item #1) still not
  root-caused - this round's result makes it clearly the higher-
  priority open item now, since it's now the harness's single worst
  remaining number.
- No adaptive/width-scaled search corridor (still open from Round 3) -
  would likely help exactly the 1901 wrong-line-lock-on case described
  above.
- Did not re-run the two 1911/1921/1931b sparse spot-check samples
  through this changed harness (they don't use Variant B/the projection
  chain at all, so no change expected - not re-verified).
- Still research-only - no production wiring.

## Round 8 (same day) - projection-profile + gradient fusion: negative result on 1901

Jon proposed a fusion detector, explicitly labeled "highest priority"
of five ideas, to attack Round 7's "wrong-line lock-on" failure mode
directly: multi-scale Scharr (2 blur scales, per-pixel max) + a light
CLAHE pre-pass, combined with a COVERAGE check - only accept a gradient
peak as a real line if it's ALSO elevated across most of the y-band
(roughly continuous top-to-bottom), not just on average - the idea
being that a real ruling line should look like a continuous line, while
a wrong-line lock-on or a handwriting stroke would show a shorter,
patchier burst of gradient response that a plain mean-magnitude check
(Round 6/7's method) can't tell apart from the real thing.

Built `find_vertical_line_via_projection_gradient_fusion()` in the same
diagnostics file (light CLAHE clip=1.5 -> 2-scale Scharr max -> per-
column mean=profile, per-column coverage=fraction of rows exceeding
2x the crop's median magnitude -> only accept a profile peak whose
coverage >= 0.5). Ran the same ground-truth-centered found/accuracy
test as Round 6 first (cheaper, catches a regression before spending
time re-running the full multi-year projection harness).

### Result: no change on 1926/1931, but 1901 got WORSE, not better

| Year | Plain gradient (Round 6) | **Fusion (coverage-gated)** |
|---|---|---|
| 1901 | 18/33 (55%), 11.0px delta | **7/33 (21%) - WORSE**, 10.9px delta |
| 1926 | 10/10 (100%), 3.8px delta | 10/10 (100%), 3.9px delta - no change |
| 1931 | 42/42 (100%), 2.8px delta | 42/42 (100%), 2.8px delta - no change |

Given the regression appeared at this cheaper, ground-truth-centered
stage, the full multi-year projection/closure-error harness (Round 7's
methodology) was NOT re-run with this detector - no reason to spend
that time when the underlying detector already regressed on the one
year it was meant to fix.

### Why: 1901's real lines don't have the "continuous" signature the fusion assumed

Inspecting the coverage values directly explains it. On 1926/1931,
genuine ruling lines show coverage 0.67-1.00 (elevated response across
nearly the whole column height) - clearly separated from background.
On 1901, genuine matches (the ones Round 6's plain gradient correctly
found) show coverage mostly in the 0.37-0.65 range - and CRITICALLY,
this overlaps almost completely with the coverage values at columns
where there is NO real line at all (0.37-0.49 for clear non-matches
like "Name", "Sex", "Color"). Coverage isn't a useful discriminator on
this specific page - real and fake both look "patchy," not because the
detector is confused, but because **1901's real printed rules
themselves are apparently broken/discontinuous down their length**,
consistent with what was already known about this exact form from
earlier row-number-anchor work this session (a physically non-straight
printed border on some rows). The fusion's core assumption - "a real
line is continuous, a false lock-on is patchy" - simply doesn't hold
for this form's print quality, so the coverage gate ends up rejecting
genuine matches right alongside genuine noise, net REDUCING recall on
exactly the case it targeted.

**This doesn't invalidate the fusion idea in general** - multi-scale
Scharr + coverage gating is a reasonable technique and produced zero
regression on the two years with well-printed, continuous lines. It
specifically doesn't work for 1901's apparently degraded/broken print,
which needed a different kind of fix than "require more continuity."

### Remaining ideas from Jon's list, not yet tried

1. Projection profile + gradient fusion (highest priority) - **tried,
   negative result on 1901, see above.**
2. Multi-scale Scharr/Laplacian alone, WITHOUT the coverage gate - not
   isolated from the coverage gate in this round's test, so it's not
   yet known whether multi-scale alone (no continuity requirement)
   would have helped 1901 without the regression.
3. Hough Line Transform on the gradient map - not attempted.
4. Morphology on the gradient magnitude map itself (not the raw image,
   which Round 5 already showed doesn't help) - not attempted.
5. Header-driven anchoring - explicitly flagged by Jon as "the real 1901
   fix": the column-number match rate (13/33, 39%) is the actual
   upstream bottleneck feeding bad search corridors into every line-
   detection approach tried so far, this one included. Not attempted
   yet - still the highest-leverage remaining open item per Round 7's
   conclusion, now reinforced by Round 8 showing that fixing the line
   detector alone (however cleverly) can't fully compensate for a
   poorly-centered search corridor.

### What was deliberately NOT done (round 8)

- Did not isolate multi-scale-without-coverage-gate as its own variant
  (idea #2 above) - the one round-8 test bundled CLAHE + multi-scale +
  coverage gating together, so it's not known which piece(s) specifically
  caused the 1901 regression vs. which (if any) might have helped.
- Did not attempt Hough line transform or gradient-map morphology
  (ideas #3/#4).
- Did not attempt header-driven anchoring (idea #5, "the real 1901
  fix" per Jon's own framing) - this is now the clear next candidate,
  since two different line-detector improvements (Round 6/7's plain
  gradient, this round's fusion) have both run into the same wall: a
  detector can't measure a line that's outside its search corridor to
  begin with, and 1901's corridors start from a 39% match rate.

## Round 9 (same day) - root-causing 1901's 39% match rate: header row is NOT level

Direct root-cause investigation of Round 4's original open item #1
(why does 1901's column-number match rate, 13/33 = 39%, lag so far
behind 1926/1931's ~50-98%?), per Jon's explicit instruction to
diagnose systematically across specific candidate causes BEFORE
assuming OCR is needed - the existing blob-and-filter sensor already
works well on other years, so the question is why THIS page specifically
breaks it.

Built `diagnostics/diagnose_1901_column_match_failures.py`: for every
real column, widened the match search to 100px (vs. production's 60px)
to separate "nothing nearby at all" from "something nearby but wrong,"
tagged each as MATCHED / NEAR_MISS / FAR_OR_ABSENT, compared blob widths
between the two groups, checked whether `find_number_row_band()` finds a
DIFFERENT optimal y-band when run independently on the left/middle/right
thirds of the table instead of the whole width, and swept the detector's
own filter thresholds (`min_blob_width_px`, `line_run_threshold_frac`)
to see if they were eating real blobs.

### The header row is measurably NOT level across the page width

| Zone (x-range) | Locally optimal band | vs. global band (568,580) |
|---|---|---|
| Left third (192-1222) | **(607, 619)** | +39 to +51px below global |
| Middle third (1222-2252) | (592, 604) | +12 to +24px below global |
| Right third (2252-3283) | **(529, 541)** | -27 to -39px above global |

That's an **~80px vertical drift** in where the real number row actually
sits, moving from left to right across the table - far larger than the
band's own 12px height. The single global band `find_number_row_band()`
returns when searching the FULL width (568,580) is a compromise that
isn't really correct ANYWHERE - it's just the least-bad single answer
across a header that isn't flat.

**Visually confirmed, not just inferred from the numbers**: cropped and
directly inspected the header strip (y=500-720) in each third. In the
left-third crop, the printed numbers (1-10) sit right at the bottom
edge of the crop, hard against the data-row boundary. In the middle-
third crop (11-22), the numbers sit noticeably higher, with real
whitespace and the start of a data row visible beneath them. In the
right-third crop (23-34), the numbers sit higher still, with an entire
full data row (a legible "English" entry) visible underneath them in
the SAME y-window. The rise is continuous and unambiguous by eye across
all three crops - this is a real skew/warp in how the page was
scanned/photographed (roughly a 1.5 degree effective tilt if treated as
linear, `atan(80/3091)`), not a detection artifact. Worth noting: this
page's sidecar already records a `deskew_angle: -0.8` correction, which
either under-corrected the true skew or there's page curvature beyond
what a single global rotation angle can fix.

### Other candidate causes - checked and mostly ruled out

- **Filtering thresholds: ruled out.** Loosening `min_blob_width_px`
  from its default down to 1, and `line_run_threshold_frac` from 0.85
  up to 0.95 (both independently and together), changed the blob count
  from 85 to at most 93 - an 8-blob difference on a page where 20
  columns are mismatched. Not the driver.
- **Blob size/shape - a real secondary signal, but downstream of the
  drift, not independent of it.** NEAR_MISS blobs (the "nearest thing
  found" for a column that didn't cleanly match) average 20.5px wide,
  vs. 14.4px for genuine MATCHED blobs. Consistent with what you'd
  expect once the search band is in the wrong place: instead of
  catching a clean small digit, the detector either grabs a smeared
  partial-digit edge, a merged mark, or unrelated ink that happens to
  fall inside the (mis-positioned) band - a symptom of drift, not a
  separate touching-digits problem in its own right.
- **Touching digits / nearby header-text contamination: not confirmed
  as a distinct cause.** Visible in the crops (some stray marks, cross-
  outs), but nothing suggesting a separate, independent contamination
  problem beyond what mis-positioned-band search already explains.
- **Edge warp**: this IS the y-drift finding, just framed differently
  than the row-number-anchor work's "columns missing at the page edge"
  pattern - here it's the same physical page-curvature/scan-skew family
  of problem, but manifesting as vertical position drift of the header
  row rather than horizontal dropout.
- **The expected-position matching itself (real_center computed from
  sidecar mask_keep_ranges)**: not implicated - real columns' geometry
  is fine, it's the SEARCH BAND that's wrong for most of the table's
  width, not the target position being matched against.

### What this means: NOT an OCR problem, per Jon's caution

This is exactly the "simple adaptive threshold or local-band issue"
Jon flagged as more likely than needing OCR - confirmed. The existing
blob-and-filter sensor doesn't need replacing; `find_number_row_band()`
needs to stop assuming one band works for the whole table width. A
natural fix (not yet implemented, proposed as the next step): run the
band search independently across several x-zones (or fit a
left-to-right band-position trend and interpolate a LOCAL expected band
per column) instead of one global search - directly targeting the
mechanism just confirmed here, rather than a blind retune.

### What was deliberately NOT done (round 9)

- Did not implement or test a per-zone/local band-calibration fix -
  diagnosis only, per Jon's explicit "diagnose first" instruction.
  Proposed as the clear next step given this result.
- Did not check whether 1926/1931 have any similar (smaller-magnitude)
  drift - only 1901 was profiled this round, since it's the one with
  the open problem.
- Did not re-examine whether this same drift explains any of the
  earlier-flagged 1931 Name/Sex outliers (Round 2, still unresolved) -
  plausible but not checked.

## Round 10 (same day) - prototyping the local-band fix: real improvement, but noisy

Implemented Round 9's proposed fix. Added two new functions to
`core/row_segmentation.py` (same module as the rest of this research-
validated-but-not-production-wired family):

- `find_number_row_band_piecewise(image, x0, x1, search_y0, search_y1,
  n_zones=3, ...)` - splits the table width into `n_zones` equal slices
  and calibrates each independently via the existing
  `find_number_row_band()`, instead of one global search.
- `detect_column_number_centers_piecewise(...)` - drop-in replacement
  for the global-band call, merges each zone's own blobs (already
  x-scoped to that zone, no de-duplication needed).

Tested via `diagnostics/test_piecewise_band_1901.py`, sweeping
`n_zones` from 2 to 16 and comparing match rate against the same 33
real 1901 columns and quality-gate logic used everywhere else this
session.

### Real improvement, but the curve is noisy, not clean

| n_zones | Match rate | n_zones | Match rate |
|---|---|---|---|
| 2 | 36% | 10 | **73% (best)** |
| 3 | 58% | 11 | 67% |
| 4 | 39% | 12 | 45% |
| 5 | 70% | 13 | 52% |
| 6 | 48% | 14 | 52% |
| 7 | 61% | 15 | 64% |
| 8 | 67% | 16 | 55% |
| 9 | 52% | global (old) | 39% |

Every tested `n_zones >= 3` beat the 39% global-band baseline, and the
best (n_zones=10, 73%) is a real, nearly 2x improvement - the core
Round 9 diagnosis (non-level header, needs local calibration) is
confirmed actionable, not just correct in principle. **But the curve
has no clean trend** - it oscillates rather than rising smoothly with
finer zones, meaning a column sitting near a zone boundary can land in
whichever neighbor's compromise band happens to be worse for it, and
that's essentially luck-of-the-grid-alignment for any single `n_zones`
choice.

### A smoother alternative was tried and did NOT fix the noise - it did worse

Hypothesis: the noise is a hard-boundary artifact of a shared zone grid
- a genuinely local search, centered on each column's own expected
position with no shared grid at all, should be smoother. Tested
directly: for each of the 33 real columns, ran `find_number_row_band()`
on a window of +/-200 to +/-800px centered on that column's own real
position (no grid, no zones). Result: **also noisy (36-61%), and never
beat the best fixed-zone result (73%).** This rules out "hard zone
boundaries specifically" as the noise's root cause - a per-column local
window has the same instability without any boundary to blame. More
likely explanation (not confirmed): `_score_number_row_band()`'s
plausibility scoring was calibrated against whole-table blob counts
(real forms run ~25-50 printed numbers, per that function's own
docstring) - inside a narrow local window, there are only a handful of
real digits to score against, so the "plausible blob count" signal
becomes unreliable at this scale, and the [best-scoring candidate]
selection can bounce between near-tied bands for reasons unrelated to
which one is actually better.

### Honest bottom line - do NOT hardcode n_zones=10

Picking n_zones=10 because it scored best on this ONE 1901 page would
be fitting a magic constant to a single sample - exactly the kind of
thing this session's research has consistently avoided (see the
"gate against known-bad ground truth" and "verify sample completeness"
lessons from Rounds 3/4). The real, defensible conclusion is narrower:
**piecewise/local calibration is a genuine, substantial improvement
over a single global band on a page with a non-level header (confirmed:
every n_zones from 3-16 beat the baseline)**, but the specific
mechanism needs either (a) testing across multiple 1901-family samples
before trusting any specific n_zones value, or (b) a better-founded
approach than either a fixed grid or a fixed-radius local window - e.g.
fitting a smooth trend line to a few widely-spaced band measurements
and interpolating, rather than re-running the noisy per-window
plausibility scorer at fine granularity.

### What was deliberately NOT done (round 10)

- Did not test piecewise calibration against additional 1901-family
  samples (e.g. `z000077130`, seen but not ground-truthed earlier this
  session) to check whether n_zones=10's strong result generalizes or
  is specific to this one page's exact column layout.
- Did not investigate/fix `_score_number_row_band()`'s apparent
  unreliability at narrow-window scale directly (the more likely real
  fix per the analysis above) - only observed its symptom.
- Did not try a smooth-trend-line-plus-interpolation approach (the
  more principled alternative floated above) - not implemented.
- Did not re-run the full multi-year projection harness (Round 7 style)
  with the piecewise detector swapped in - premature given the n_zones
  instability isn't resolved yet.
- Not wired into any production code - `find_number_row_band_piecewise()`
  and `detect_column_number_centers_piecewise()` live in
  `core/row_segmentation.py` alongside the rest of this research-
  validated-but-unwired family, matching existing precedent for that
  module.

## Round 11 (same day) - does piecewise calibration generalize across 1901 pages?

Jon's framing for this round: the important question isn't "which
n_zones wins" (Round 10 already showed that's noisy and shouldn't be
hardcoded) - it's **does local calibration consistently improve recall
across DIFFERENT 1901 pages at all**. If yes, the next move is fixing
`_score_number_row_band()`'s apparent narrow-window unreliability
directly. If no, a fixed-grid/fixed-window scorer approach is the wrong
shape of fix entirely, and a more adaptive strategy is needed instead.

**Important, honestly-reported constraint**: this corpus has exactly
ONE full-layout (33-column) 1901 ground-truth sample -
`z000077117` (Sample A, used throughout Rounds 2-10). Searched the
whole corpus for a second one; none exists. The only other 1901-family
page with ANY real, human-confirmed column ground truth is
`z000017634` (Sample B) - but it's a sparse 5-column spot-check (Name/
Sex/Relationship to Head/Age/Birthplace), not a full column_order. This
was used anyway, as the best real (not invented) comparison available,
but its statistics are much thinner (5 data points, not 33) and every
single-column flip swings its match rate by 20 percentage points - kept
in mind when reading its results, not glossed over.

Reused Round 10's exact sweep (global baseline + n_zones 2-16,
`diagnostics/test_piecewise_band_1901_two_sample.py`) against both
samples, reporting median gain (not just best-case, to avoid the "one
lucky n_zones" trap) and stability.

### Results

| | Sample A (33 cols) | Sample B (5 cols, sparse) |
|---|---|---|
| Global band | 39% | 60% |
| Best n_zones | 10 -> 73% (+33pt) | 3 -> 100% (+40pt) |
| **Median gain across n_zones=2..16** | **+15.2pt** | **+20.0pt** |
| % of n_zones settings that beat global | 87% | 60% |
| Stdev across the sweep | 10.5pt | 17.1pt |

### Verdict: YES, it generalizes - median gain is positive on both samples

Both samples show a clearly positive median gain from piecewise
calibration, and a majority of n_zones settings beat the global
baseline on both (87% and 60%). The BEST n_zones differs between
samples (10 vs. 3, confirming Round 10's caution against hardcoding one
value), but the underlying effect - splitting the calibration instead
of using one global band helps, on average, not just in a lucky
best-case - replicates across the only two real 1901 pages available.
Per Jon's own decision framework for this round: **this points at
fixing `_score_number_row_band()`'s narrow-window unreliability
directly, not at abandoning the piecewise approach for something more
exotic.**

### What was deliberately NOT done (round 11)

- Only 2 real 1901-family samples exist in this corpus - a stronger
  claim of "generalizes" would need more, especially a second
  FULL-layout sample (Sample B's 5-column sparsity is a real
  statistical-power limitation, acknowledged, not fixed).
- Did not yet fix `_score_number_row_band()` - this round only answered
  the yes/no gate for whether that's the right next investment.
- Did not investigate why the best n_zones differs so much between
  samples (10 vs. 3) - plausibly related to each page's specific drift
  shape/magnitude, not checked.

## Round 12 (same day) - the scorer fix: tried, regressed, reverted

Per Round 11's own decision framework (median gain positive on both
samples -> go fix `_score_number_row_band()`'s narrow-window scoring
directly), attempted exactly that. **Net result: reverted. The fix
regressed the target task AND silently broke previously-validated
results on 1911/1926.** Recorded here in full, including the two failed
calibration attempts, because this is exactly the kind of result that's
easy to quietly discard and re-litigate later without the "why" - and
because Jon's own stated principle ("if no, the right answer may be a
more adaptive strategy") turned out to be the more accurate read.

### The diagnosed bug (real, confirmed)

`count_score = min(n, 70) / 70.0` uses a FIXED cap regardless of the
actual search width passed in (`table_width` was already a parameter,
just unused for this term). At full-table scale (~3100-3900px) this is
fine. At narrow-zone scale (as `find_number_row_band_piecewise()` uses),
count_score stays near-zero for every candidate band regardless of
which is actually better - it can't discriminate, so the coarser/more-
quantized terms (`plausible_width_frac`, the binary `merge_penalty`)
end up driving the argmax close to arbitrarily. This diagnosis itself
was confirmed correct by direct calculation (a 1901 zone at 1/10th
table width has expected_max capped effectively near its raw blob
count of ~30, versus a fixed-70 count_score of ~0.4-0.5 - the fixed
version genuinely has less range to work with at that scale).

### Attempt 1: scale count_score's cap by search width, calibrated on REAL COLUMN density (~90px/column)

Result: Sample A's best case REGRESSED (73%->55%), median gain fell
(+15.2pt->+6.1pt). Root cause of the regression, found by direct
calculation: `n` in this function counts RAW BLOBS (detect_column_
number_centers()'s noisy output), not real columns - a fundamentally
different, denser quantity (85 raw blobs vs. 33 real columns on the
same 1901 table). Calibrating against real-column density made
`expected_max` far too small for a narrow zone (as few as 3-4 blobs
maxed out count_score), so it saturated almost immediately and
provided EVEN LESS discriminating range than the original fixed
constant - the opposite of the intended fix.

### Attempt 2: recalibrate on RAW BLOB density (~36-68px/blob measured across 1901/1911/1931)

Result: mixed, and ultimately not a win. Sample A's piecewise sweep
improved over attempt 1 but still didn't beat the original fixed-70
constant (best case 64% vs. the original 73%; median gain +12.1pt vs.
+15.2pt). Sample B's GLOBAL band search jumped to a suspicious-looking
100% - but this turned out to be a ceiling effect masking a real
problem, caught by the regression check below, not a genuine
improvement.

### The regression that ended this attempt: 1911 and 1926's already-validated global bands broke

Re-ran the exact three real, already-validated samples from Round 2
(the ones `find_number_row_band()`'s own docstring cites as evidence it
works) through the recalibrated scorer:

| Year | Round 2 validated band | Recalibrated-scorer band | Drift |
|---|---|---|---|
| 1911 | (591, 603) | (500, 512) | **91px off - wrong band** |
| 1926 | (992, 1004) | (1202, 1214) | **210px off, 133 blobs (handwriting noise, not the number row)** |
| 1931 | (654, 666) | (652, 664) | 2px - fine |

1911 and 1926 both landed on a genuinely different, wrong y-band -
confirmed by the blob count alone for 1926 (133 blobs is far more than
the ~30 real printed numbers that form has; the scorer was now
rewarding a noisier, wrong location because the changed count_score
formula shifted which candidate wins the argmax even at FULL table
width, not just in narrow zones as intended). **This is a real
regression on already-shipped-and-trusted research results, not a
tradeoff worth accepting for a narrow-zone improvement that didn't even
clearly beat the original baseline.**

### Reverted

`_score_number_row_band()` is back to the original `min(n, 70) / 70.0`,
confirmed by re-running the exact three Round 2 samples and matching
their original bands exactly ((521,533)/(655,667)/(1280,1292) - these
differ slightly in notation from the Round 2 writeup's numbers because
Round 2 used slightly different search_y0/y1 windows for 1911, not
because of any residual change here; re-verified byte-for-byte
reproducible from a clean run). The docstring itself now documents both
failed attempts and why, so this isn't quietly re-attempted identically
in a future session.

### What this means for the open question

Round 11 posed a clean either/or: fix the scorer, or the fixed-grid/
fixed-window shape of calibration is wrong entirely. This round's
result reads as evidence for the SECOND branch, not the first -
tuning the scorer's internals didn't produce a robust win and cost a
real regression trying. The untried alternative flagged back in Round
10 (`plausible_width_frac`'s coarse quantization at low n - a blob
either counts or doesn't, no partial credit, so a handful of blobs only
ever produces a few discrete score levels) was NOT attempted this round
and remains a candidate, but given how narrowly-targeted the width-
scaling fix already was and still caused a regression, a smooth-trend-
fit approach across a few widely-spaced, reliable measurements (Round
10's other floated alternative) looks like the safer next direction
rather than continuing to iterate on this specific scoring function.

### What was deliberately NOT done (round 12)

- Did not try adjusting `plausible_width_frac` to be a smoother/
  continuous score instead of a hard 3-30px in/out test - the other
  candidate explanation for narrow-window noise, not yet tested.
- Did not attempt the smooth-trend-fit-and-interpolate calibration
  approach floated in Round 10 - now the more promising remaining
  direction given this round's result.
- Did not re-verify whether the reverted scorer still produces exactly
  Round 2's DOCUMENTED numbers (591,603) for 1911 - the re-run here
  produced (521,533), which is what every OTHER run this session
  (Rounds 4, 7, 9, 10, 11) also independently reproduced with the same
  search_y0/y1 window (500-625), so this is treated as the real,
  consistently-reproducible value and Round 2's differently-worded
  number as using a different search window, not a discrepancy
  introduced by this round's changes - but this was not independently
  re-derived from Round 2's original script to confirm with certainty.

## Round 13 (same day) - the trend-fit-and-interpolate alternative: comparable best-case, worse on average

Tried the other candidate from Round 10/12's list: instead of touching
`_score_number_row_band()` again, sidestep it entirely. Added
`detect_column_number_centers_trend()` to `core/row_segmentation.py`:
take `n_anchors` WIDE band measurements (reliable at that scale per
Round 9's thirds check), fit a simple linear trend through their
(x_center, y_center) points, then INTERPOLATE (pure arithmetic, no
search, no scoring) the expected band for `n_slices` finer x-slices
before running plain `detect_column_number_centers()` at each. Never
re-invokes the scorer at narrow scale at all - the mechanism that
caused Round 12's regression simply isn't in this code path.

Swept `n_anchors` in {2,3,4,5} x `n_slices` in {5,10,15,20}
(`diagnostics/test_trend_band_1901_two_sample.py`) against the same two
real 1901 samples as Round 11.

### Best-case looks comparable to piecewise - but the full sweep tells a different story

| | Best config | Best result | **Median gain across the whole sweep** |
|---|---|---|---|
| Sample A | n_anchors=5, n_slices=5 | 70% (vs. piecewise's 73%) | **+1.5pt** (vs. piecewise's +15.2pt) |
| Sample B | n_anchors=3, n_slices=5 | 100% (vs. piecewise's 100%) | **+0.0pt** (vs. piecewise's +20.0pt) |

The best-case numbers are close to piecewise's best case, which might
look like a wash at first glance. **It isn't** - Round 11 established
median gain (not best-case) as the fair metric specifically because
best-case numbers reward cherry-picking a lucky parameter. By that
metric, this approach is dramatically less reliable: most (n_anchors,
n_slices) combinations perform AT OR BELOW the global baseline, not
above it - the good results cluster narrowly around `n_slices=5` paired
with specific `n_anchors` values, with no broader trend of "more
anchors/slices = better."

### A new noise source, not the one this was built to avoid

Diagnosed by inspecting the sweep directly: performance drops sharply
as `n_slices` increases (5 -> 10 -> 15 -> 20), which is the OPPOSITE of
what finer interpolation resolution should produce if the trend line
itself were the limiting factor. Likely explanation: `detect_column_
number_centers()` is called on each slice's own x-range independently
- a real printed digit's blob can straddle a slice boundary when slices
get narrow enough (e.g. `table_width/20` ~ 150px for 1901, well under
the ~94px average column spacing, meaning some slices are narrower than
a single column), getting cut in half and either failing the min-width
filter or landing at a shifted position. This is a DIFFERENT failure
mode than Round 12's scorer instability (there's no scorer involved
here at all) - it's a boundary-fragmentation artifact specific to
detecting on cropped horizontal windows, not a calibration problem.

### Honest verdict

Neither of the two candidates from Round 10/12's list turned out to be
a clean win: the scorer fix regressed already-validated results and was
reverted; this trend-fit alternative avoids that specific failure mode
but introduces a different one (slice-boundary digit fragmentation) and
is measurably LESS reliable than the original piecewise approach on the
metric that matters (median gain, not best-case). **Piecewise
calibration (Round 10/11) remains the best-performing option found so
far for 1901's non-level header row**, imperfect and grid-alignment-
sensitive as it is - not because it's proven robust, but because both
alternatives tried since have done worse, not better, when measured
fairly.

### What was deliberately NOT done (round 13)

- Did not try widening the minimum slice width (e.g. never let a slice
  go narrower than ~2x the typical column spacing) to test whether
  that specifically fixes the diagnosed digit-fragmentation issue -
  plausible quick follow-up, not attempted.
- Did not investigate a genuinely different, more principled direction:
  actually modelling the physical page skew/warp geometrically (e.g.
  fitting a single global tilt angle from a couple of well-separated
  measurements, the way `core/manifest_pipeline.py`'s deskew logic
  already does for whole-page rotation) rather than empirically
  calibrating a grid or scorer at all. Given three attempts (scorer
  tuning, piecewise grid, trend interpolation) have all hit some form
  of grid/parameter sensitivity, this may be the more durable fix, but
  is a bigger design change than anything tried so far this round.
- Not wired into any production code - stays alongside the rest of
  this research-validated-but-unwired family in `core/row_
  segmentation.py`.

## Round 14 (same day) - the actual root cause: a physical binding crease, confirmed and corrected

Rounds 9-13 spent significant effort trying to CALIBRATE around 1901's
non-level header row (piecewise grids, scorer tuning, trend
interpolation) - all built on the assumption that the underlying pixels
were fixed and only the DETECTION strategy could adapt. Jon identified
the actual root cause instead: a physical paper crease from the bound
volume's own binding/gutter, visible by eye in the raw scans. This
reframes the whole problem - it's not a detection-calibration problem
at all, it's an uncorrected physical page distortion that calibration
tricks were only ever going to work around, never fix.

### Confirmed across 6 independently-checked real pages, not just the one research sample

Direct visual inspection (cropping and reading the raw scans, not
inference from measurements) confirmed the IDENTICAL crease - at the
same position relative to the printed form, right between "SCHEDULE/
TABLEAU" and "No. 1" in the header - on all 6 pages checked:
`z000077117`, `z000077122`, `z000077130`, `z000077132`, `z000077134`,
`z000077145`. These span three different enumerators (McLaren, Campbell,
Webster), multiple townships, and multiple sub-districts within
District 81 South Lanark. That consistency across unrelated pages rules
out a one-off handling crease - it's the bound volume's own gutter,
present on every page in this reel. `z000077130` even has extra
photographic-frame margin that shifted its absolute pixel position, yet
the crease still lands in the same spot relative to the FORM content
once that's accounted for - further confirming it's tied to the page,
not the scan framing.

Drift SHAPE varies page to page even though the crease location doesn't:
`z000077117` showed a gradual continuous decline (613→598→535 across
three wide anchors); `z000077134` showed a FLAT first two-thirds
(599, 599 - literally identical) then a sharp late drop (548) - both
consistent with "distortion concentrated toward one side, severity
varies by how well this particular page happened to lie flat when
photographed," not a fixed geometric formula that would transfer
page-to-page unchanged.

### Built `core/page_dewarp.py` - crease detection + per-column vertical correction

Two functions, deliberately reusing already-validated primitives rather
than re-deriving them:

- `detect_page_crease_x(image, search_x_lo, search_x_hi, min_coverage=0.5,
  min_peak_ratio=1.5)` - Scharr-gradient + coverage-gate crease finder,
  scoped to the FULL PAGE HEIGHT (not one row band). Reuses the same
  gradient+coverage principle from Round 8's fusion detector, but for
  the opposite reason it worked there: Round 8's coverage gate REGRESSED
  ruling-line detection because 1901's printed lines are genuinely
  broken/discontinuous - a paper crease is a physical fold, not printed
  ink, and IS continuous top-to-bottom, so the same technique that hurt
  one measurement helps this different one. **First attempt located the
  wrong feature** - a strong printed ruling line at x=1573, 130px from
  the real crease (confirmed by cropping the FULL page height at that
  x-range and visually tracing which vertical line actually continues
  unbroken through the data rows, not just the header). The ruling line
  had a stronger gradient response than the actual (subtler, ink-free)
  crease, so a wide, blind search window preferred it. Narrowing the
  search window to bracket only the confirmed crease region
  (1650-1780px, informed by the visual crease-location work) found the
  real crease at x=1731 - close to, though not exactly matching, the
  ~1699-1706 range read by eye across the 6-page check; the discrepancy
  is not yet resolved.
- `dewarp_page_at_crease(image, crease_x, table_left, table_right,
  search_y0, search_y1, n_anchors=3, ...)` - per-column VERTICAL pixel
  shift (via `cv2.remap`) so the whole page's band position agrees with
  a single reference anchor.

### A real bug, caught by Jon spotting a visual artifact - not by a metric

**First implementation split the trend measurement into independent
left/right anchor sets at the crease, then set the correction target by
EXTRAPOLATING the left side's slope 385px past its last real
measurement, out to the crease position.** That produced a jump in the
number reported (39%->67%) that looked like a clean win - and would have
been reported as one, if Jon hadn't looked at the actual dewarped image
and asked about it. The extrapolated target (721) turned out to sit
outside the range of every real measurement on the page (613-685) - a
classic extrapolation failure, confirmed by printing the anchors and
target directly, not just suspected. Because every column's shift was
computed relative to this one bad, too-high target, every single column
got shifted the same direction (all-negative, up to -144px) - and near
the page edges this dragged the raw scan's actual black photographic
border into the visible table area, a real, visible corruption Jon
caught by eye in the rendered corner crop that the match-rate metric
alone did not surface. **The lesson: a metric moving the right direction
is not proof of correctness - Jon looking at the actual image is what
caught this, and the fix would not have been found by staring at
numbers alone.**

### Fix: one connected trend, never extrapolated for the target itself

Root cause wasn't really "the crease needs independent left/right
measurement" (Round 9/13's original finding) - it was specifically
"the target value must be a REAL measurement, never a projection."
Rewrote to use ONE set of wide, reliable anchors across the WHOLE table
(reusing the exact `n_anchors=3` scale already validated in Round 9),
with `target_y` set to the FIRST anchor's own real measured value -
never extrapolated. This works because the crease's confirmed position
(~1700-1731) sits almost exactly on top of the middle anchor of that
same Round 9 measurement (x=1737) - and the earlier visual check across
x=1400-2000 (straddling the crease) already found no visible
discontinuity in the printed content right at the fold. Both facts
together mean a normal piecewise-linear interpolation through the
existing reliable anchors already passes near the crease naturally,
without needing to treat it as a hard "reset point" requiring
independent re-measurement per side.

Re-inspected the corners after the fix: the black-border artifact is
gone, confirmed by direct crop inspection, not assumed from the metric
alone this time.

### Second finding: `cv2.remap`'s interpolation mode matters

With the bug fixed, the HONEST result was much more modest than the
buggy 67%: **39% -> 45%** (max shift dropped from 162px to a sane 114px).
Investigated why linear interpolation (the initial, unexamined default)
underperformed: `cv2.INTER_LINEAR` blends adjacent source pixels
vertically when shifting, which blurs the printed digit edges -
directly counterproductive for `detect_column_number_centers()`, which
depends on a sharp Otsu threshold to separate ink from background.
Switched to `cv2.INTER_NEAREST` (no blending, exact pixel selection) -
confirmed via direct A/B test this is not merely "different" but better:

| Method | Dewarp alone | + piecewise n_zones=3 on top |
|---|---|---|
| Before dewarp (baseline) | 13/33 (39%) | - |
| Dewarp, INTER_LINEAR | 15/33 (45%) | 21/33 (64%) |
| **Dewarp, INTER_NEAREST** | **17/33 (52%)** | **23/33 (70%)** |

### Result: 39% -> 70% via dewarp (INTER_NEAREST) + simple piecewise n_zones=3

This is close to Round 10/11's best-ever recorded number for this page
(73%, at n_zones=10) - but reached through a meaningfully more
principled and more stable path: the underlying page distortion is
actually corrected first (not just calibrated around), the piecewise
step on top only needs to fix a much smaller RESIDUAL error, and it does
so at n_zones=3 - a small, wide-zone, low-parameter-count setting,
not the fragile, best-of-16-sweep n_zones=10 that Round 10 explicitly
warned should never be hardcoded. Combining dewarp with LARGER n_zones
(8, 10) on top was tried and was WORSE than n_zones=3 (58-61%),
consistent with Round 10's original finding that finer zone grids don't
monotonically help - now doubly true once most of the real distortion
is already corrected upstream.

### What was deliberately NOT done (round 14)

- The ~130px gap between the visually-read crease position (~1699-1706)
  and the detector's found position (1731) was not resolved.
- The remaining 3-point piecewise-linear trend (vs. the true, likely
  smoother drift curve visible in the 6-strip check) was not refined
  further - more (still-wide, still-reliable) anchors, or a smooth
  curve fit, could plausibly close more of the remaining gap to 100%.
- Only tested against ONE of the 6 confirmed-crease pages
  (`z000077117`, the one with full 33-column ground truth) - the other
  5 have no equivalent ground truth to score against, so improvement
  there could only be checked visually, not quantified.
- Not wired into any production pipeline - `core/page_dewarp.py` is
  research code, matching every other artifact from this
  investigation's standing scope constraint.
- Did not re-run the full multi-year projection harness (Round 7 style)
  or the multi-year generalization test (Round 4/11 style) on the
  dewarped image - only the before/after match-rate comparison was
  done this round.

## Round 15 (same day) - refining the trend beyond 3 anchors: no improvement found

Direct follow-up on Round 14's flagged next step: the 3-anchor trend is
a coarse piecewise-linear approximation of what the 6-strip visual
check already showed is a smoother, continuous curve - so more/better
sample points seemed like a plausible way to close more of the 39%->70%
gap toward 100%. Tried two honest refinements; **neither beat the
original 3-anchor baseline.**

### Attempt 1: overlapping wide windows for more sample points

Built `_find_overlapping_band_anchors()` (`core/page_dewarp.py`) - a
sliding WIDE window (still >=1000px+, staying inside the scale Round
9-13 established as reliable) stepped across the table with overlap,
instead of partitioning it into disjoint zones. This gets more (x, y)
sample points without shrinking any individual measurement's width -
avoiding the exact mechanism (narrow zones = unreliable scores) that
caused Round 10/12's instability.

Swept several (window_width, window_step) configs, piecewise-linear
through the resulting points, dewarp-alone and combined with piecewise
n_zones=3 on top (Round 14's best pipeline):

| width | step | n anchors | Alone | +piecewise n_zones=3 |
|---|---|---|---|---|
| 1200 | 400 | 7 | 42% | 48% |
| 1200 | 600 | 4 | 45% | 58% |
| 1500 | 500 | 5 | 52% | 55% |
| 1500 | 750 | 3 | 52% | 55% |
| 1800 | 600 | 4 | 48% | 55% |
| **Original 3-anchor baseline** | | **3** | **52%** | **70%** |

Every overlapping-window config underperformed the original disjoint
3-anchor result, several by a wide margin. Diagnosis: even individually
reliable wide-window measurements carry some real noise (visible in the
anchor values themselves - not a smooth monotonic sequence), and
piecewise-LINEAR interpolation through MORE points means more segments,
each one a fresh opportunity for that per-point noise to introduce a
local kink. Fewer, more widely-spaced anchors happen to average over a
larger physical span each, which incidentally smooths out more of that
noise - not because 3 is a magically correct number, but because it
naturally lands on the more-forgiving end of the resolution/noise
tradeoff for this specific measurement's noise floor.

### Attempt 2: smooth polynomial regression through the extra points

If piecewise-linear is too sensitive to individual noisy points, a
least-squares smooth fit (which AVERAGES across all points rather than
passing through each one exactly) should be more robust. Tried linear
and quadratic `np.polyfit` regression through the same overlapping-
window anchor sets:

| width | step | degree | Alone | +piecewise n_zones=3 |
|---|---|---|---|---|
| 1200 | 400 | 1 | 55% | 58% |
| 1200 | 400 | 2 | 55% | 55% |
| 1500 | 500 | 1 | 39% | 55% |
| 1500 | 500 | 2 | 42% | 45% |
| 1800 | 600 | 1 | 48% | **67%** |
| 1800 | 600 | 2 | 48% | 61% |

Closer than attempt 1's best case, but still short of the 3-anchor
baseline's 70% combined result. Smoothing helped somewhat (as expected,
averaging over noisy points beats connecting them exactly), but not
enough to overcome the underlying limitation.

### Conclusion: 3-anchor piecewise-linear remains the best found - refinement attempts hit a real wall, not a tuning gap

Both honest attempts at improving trend resolution made things worse or
at best matched-but-didn't-beat the original. This suggests the limit
isn't "not enough sample points" or "too rigid an interpolation" - it's
that `find_number_row_band()`'s own per-measurement noise floor (even
at reliable wide scale) is comparable to or larger than the fine-
grained signal any of these refinements were trying to extract. Adding
resolution just adds proportionally more noise along with it. Round
14's 39%->70% (dewarp + simple piecewise n_zones=3) stands as the best
validated result for this page; further gains likely need a
fundamentally more precise underlying measurement (e.g. tracking the
crease/fold geometrically rather than via the number-row band proxy)
rather than more of the same technique applied more finely.

### What was deliberately NOT done (round 15)

- Did not attempt tracking the physical crease/fold line itself down
  the page height as a more direct, position-independent warp
  reference (as opposed to the number-row band, which is a proxy
  measurement one step removed from the actual page geometry) - a
  plausible more-precise direction, not attempted.
- Did not test degree-3+ polynomial fits or other smoothing techniques
  (e.g. a robust/outlier-resistant fit) - stopped at linear/quadratic
  since neither showed a trend toward beating the baseline.
- Not wired into any production pipeline - still research-only.

## Round 16 (same day) - validating the dewarp against the other 5 confirmed-crease pages

Ran Round 14's fixed pipeline (`detect_page_crease_x` + `dewarp_page_at_crease`,
n_anchors=3, INTER_NEAREST) against `z000077122`, `z000077130`,
`z000077132`, `z000077134`, `z000077145` - the other 5 pages Round 14
confirmed by eye have the identical crease. None have full-column
ground truth like `z000077117`, so this used `find_number_row_band()`'s
own plausibility score/blob count as a ground-truth-free signal, plus
visual inspection - consistent with Round 14's own lesson that a metric
alone isn't sufficient proof.

### Blind crease detection generalized badly: 1 of 5, not 5 of 5

Using the same search-window approach validated on `z000077117`
(1650-1780px), `detect_page_crease_x()` found a crease on only
`z000077145`. The other 4 - including `z000077134`, which was
VISUALLY confirmed to have the identical crease in Round 14's own
6-page check - failed the min_coverage/min_peak_ratio gate. Direct
inspection of `z000077134`'s raw profile values confirmed the crease IS
there but weaker than the threshold demanded (best in-window ratio
1.33, need >=1.5) - a real, present feature, just with a fainter
gradient signal on this page than on `z000077117`.

### Jon supplied a structural prior: the crease sits on the printed column 15/16 boundary

On `z000077117`, column 15 (Nationality, x1=1735) meets column 16
(Religion, x0=1735) at EXACTLY the crease position independently
measured three different ways this session (1731 by blind detection,
1737.5 by the wide-anchor trend, ~1699-1706 by eye). That's a genuine,
useful structural fact: the crease isn't just "somewhere near the
middle," it's pinned to a specific, nameable column boundary.

Computed the fraction (crease_x - table_left) / table_width = 0.4992
from `z000077117`'s real, ground-truth table bounds, and added
`find_crease_x_near_prior(image, prior_x, search_radius=60)` to
`core/page_dewarp.py` - given a prior x-position, finds the local
gradient-magnitude peak in a tight window with NO confidence-gate at
all (the prior is doing the confirmation work the blind version's
thresholds were doing - same "prior narrows the corridor, detector just
measures" pattern used throughout this whole investigation).

### Result: still not reliable - table bounds, not the crease-finding logic, are the likely bottleneck

Scaling that 0.4992 fraction against each page's own (table_left,
table_right) predicted crease positions 1730-1815 across the 5 pages.
The prior-guided detector found *something* near each prediction
(within the 60px search radius), but direct visual verification on
`z000077122` showed the REAL column 15/16 boundary sits at x~1665 on
that page - neither matching the table-fraction prediction (1730) nor
where the detector actually landed (1783). Both were off, in different
directions.

**Most likely cause, not yet confirmed**: `z000077117` is the only one
of these 6 pages with a real, human-confirmed ground-truth sidecar
(`table_left=192, table_right=3283`). The other 5 pages' table bounds
used in this round (`table_left=190`, `table_right` estimated per page)
were rough visual guesses, not measured values - so the "same fraction
of table width" calculation is only as trustworthy as those guesses,
and a table-bounds error of even a percent or two would shift the
predicted crease position by 30-60px, which is exactly the size of the
discrepancy observed. This is a measurement-input problem, not
necessarily evidence the column-boundary prior itself is wrong.

### What was deliberately NOT done (round 16)

- Did not get real, measured table bounds for the 5 secondary pages
  (e.g. via the same review-UI process that produced `z000077117`'s
  sidecar, or by locating the row-number margin columns the way
  `detect_row_number_centers()` already does for other samples) - this
  is the clear next step before re-testing the column-boundary prior,
  since the current test can't distinguish "the prior is imprecise"
  from "the table-bounds guess feeding it is imprecise."
- Did not complete the dewarp+piecewise validation on the 4 pages where
  blind detection failed, since a reliable crease position wasn't
  established for them this round.
- Not wired into any production pipeline - still research-only.

## Round 17 (same day) - confirmed: real page-to-page placement shift on the scanner

Direct follow-up on Round 16's open question - Jon asked whether
running column/border detection would reveal an actual shift in how
each page sat on the scanner, rather than assuming a fixed table-left
guess was safe to reuse across pages.

### First attempt: "strongest vertical line near the left margin" - wrong feature

Applied the same Scharr-gradient technique used for crease detection to
each page's own left margin (search window x=50-400), expecting to find
the outer table border. On `z000077117` (the one page with real ground
truth, `table_left=192`) this found x=323 instead - confirmed by
cropping the region directly: **x=323 is the divider between the
"Line/Ligne" row-number margin column and "Dwelling House" (column 1),
not the table's outer edge.** The internal divider has a bolder,
stronger printed line than the true outer border, so "find the
strongest line in a wide window" reliably finds the wrong one - the
identical failure shape as the crease-vs-ruling-line confusion in
Round 14, now showing up for a different measurement.

Jon clarified `z000077117`'s ground truth `table_left=192` was
deliberately set to INCLUDE the row-number margin column inside the
table - confirming x=192 (not x=323) is genuinely the value being
searched for.

### Second attempt: leftmost strong line instead of strongest - closer, one real bug found

Switched from "global argmax in the window" to "first x where the
gradient ratio crosses a threshold, scanning left to right" - on the
theory that the TRUE outer border should be the first real line
encountered, not necessarily the boldest one anywhere in the window.
First run (search starting at x=50) found x=91 for `z000077117` -
**also wrong**, and cropping that region showed why: x=50-~120 is the
actual black photographic negative-frame border visible at the very
edge of the scan, an even stronger edge than the printed table border.
Restarting the search past that margin (x=150) fixed it: `z000077117`
found x=211 (19px from the real 192 - close, not exact).

### Results across all 6 pages, search starting at x=150

| Page | Leftmost strong line found |
|---|---|
| `z000077117` (real ground truth: 192) | 211 |
| `z000077122` | 175 |
| `z000077130` | 187 |
| `z000077132` | 216 |
| `z000077134` | **314 (wrong - same internal-divider confusion as attempt 1)** |
| `z000077145` | 200 |

5 of 6 landed in a real, plausible cluster (175-216px - a genuine ~40px
spread, not noise) - directly confirming Jon's hypothesis: **the page's
physical position on the scanner/copy stand genuinely shifts by ~20-40px
from photograph to photograph**, consistent with each page of a bound
volume being individually handled and re-photographed rather than
mechanically fixed in place. This is real signal, not measurement
error, and explains a meaningful part of Round 16's crease-position
prediction errors - a fixed table-left assumption reused across pages
doesn't account for this genuine physical variation.

`z000077134` still found the wrong feature (314, the internal divider)
even with the corrected search start - the SAME failure mode recurring
on the SAME problem page that also gave the weakest crease signal in
Round 16 (ratio 1.33, below threshold). Not a coincidence worth
ignoring: this specific page's printed lines may generally have lower
contrast/distinctness than the others in this batch, a real per-page
data-quality difference rather than a single unlucky miss.

### What this confirms and what's still open

- **Confirmed**: page placement genuinely shifts between scans - a
  fixed-bounds assumption is the wrong model, matching Jon's original
  instinct in asking for this check.
- **Confirmed, again**: "find the strongest/first vertical line in a
  wide blind window" is not a reliable general technique on this page
  family - it has now mis-identified the wrong feature at least 3
  distinct times this session (Round 14's ruling-line-vs-crease
  confusion, this round's row-number-margin-vs-outer-border confusion
  twice). The pattern is consistent: MULTIPLE real, structurally
  different vertical lines exist close together on these forms, and
  picking "whichever is strongest/first" is fundamentally the wrong
  selection principle without an additional constraint (a prior, in
  every case where this has actually worked well this session).
- **Still open**: a reliable, page-adaptive way to find the true table
  border automatically - even the corrected method still failed on 1 of
  6 pages. `z000077117`'s own 19px residual error (211 vs. true 192)
  also shows this specific technique isn't yet precise enough to fully
  replace real ground truth, even when it points at roughly the right
  feature.

### What was deliberately NOT done (round 17)

- Did not fix `z000077134`'s specific misdetection - flagged as a
  possible genuine lower-contrast page rather than investigated further.
- Did not re-attempt the Round 16 crease-position validation using
  these newly-measured (imperfect but real) per-page left-border
  values - the natural next step, not done this round.
- Did not investigate whether the same leftmost-strong-line approach
  works better/worse for the RIGHT table border (only left was tested).
- Not wired into any production pipeline - still research-only.
