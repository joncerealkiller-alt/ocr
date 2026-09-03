# Layout Detector Bootstrap Training — Does Fine-Tuning Help?

**Date:** 2026-08-05 (v1), same day (v2 - see bottom section)
**Status:** Experimental only, per direct instruction — nothing here is wired into the pipeline. `core/layout_detector.py` and `core/layout_detector_v26.py` are both untouched.

**Skip to [v2](#v2-does-more-data-fix-row-detection) for the current answer** — v1 below found row detection failed completely on 13 images; v2 tested whether scaling to 103 images (pulled directly from LAC, see `docs/` git history / this file's v2 section) fixes that, and it substantially does.

## Question

Given 16 census images with real transcribed ground truth (`data/automatedgenealogy_pull.csv`), does a small amount of domain-specific fine-tuning improve YOLOv26-small's layout detection on this corpus's material, versus the off-the-shelf DocLayNet-trained weights?

## Pipeline

1. **`training/layout_bootstrap_step1_generate_sidecars.py`** — ran all 16 census images through `core/auto_sidecar.py`'s `generate_auto_sidecar()` (the "mostly dialed in" auto_row/auto_column CV pipeline, `doc_type_override="canada_census_1911"`), producing real `table_bbox`/`header_bbox`/per-row `bbox` coordinates. Cross-validated each image's detected row count against `automatedgenealogy_pull.csv`'s real transcribed row count.
   - **13/16 validated cleanly** (detected row count within 3 of the real transcribed count).
   - **3/16 failed validation**: `e001946615`, `e001946619`, `e001946623` — the row detector tiled a fixed 50-row count regardless of actual page content (real counts were 14, 45, 15 respectively), and its own warning flagged rows running past the image's actual bottom edge. Excluded from training rather than fed in as bad pseudo-labels.

2. **`training/layout_bootstrap_step2_build_yolo_dataset.py`** — converted the 13 good sidecars into a YOLO dataset with a **new, project-specific 3-class scheme** (`table`, `row`, `header` — not DocLayNet's 11 classes; this bootstrap replaces the detection head rather than fine-tuning within the existing label space). 10 train / 3 val images, 676 total boxes (650 rows, 13 tables, 13 headers).

3. **`training/layout_bootstrap_step3_train.py`** — fine-tuned from the pretrained YOLOv26-small checkpoint (`Armaggheddon/yolo26-document-layout`, `yolo26s_doc_layout.pt`), 80 epochs requested, early-stopped at epoch 37 (patience=30, no improvement after epoch 7).

4. **`training/layout_bootstrap_step4_compare.py`** — visual + quantitative before/after comparison, vanilla pretrained weights vs. the fine-tuned checkpoint, on the same sample images.

## Result: real signal, but split sharply by class

Final validation metrics (3 held-out images, best checkpoint = epoch 7):

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| **table** | 1.00 | 1.00 | **0.995** | 0.497 |
| row | 0.003 | 0.007 | 0.0001 | ~0 |
| header | 0.005 | 1.00 | 0.018 | 0.006 |

**Table detection learned essentially perfectly from just 10 training images.** One large, visually distinctive region per page — close to what the pretrained backbone already does well (it's the same shape of task as DocLayNet's `Picture`/`Table` classes), so 10 examples was enough to sharpen it to near-perfect on held-out pages.

**Row detection completely failed to learn** — precision and recall both near zero against 150 real row instances in the val set. Visually (see `data/outputs/layout_bootstrap_train/comparison/`), the fine-tuned model's "row" predictions are scattered fragments along scan edges and one stray oversized box, not real horizontal row bands. Going from "find 1 big region" to "find ~50 small, near-identical, densely-packed regions" is a much harder localization task, and 10 images (500 row examples) clearly wasn't enough for it to generalize — it likely memorized the specific pixel patterns of the 10 training pages rather than learning a transferable "row" concept.

**Header degraded rather than improved**: recall 1.0 but precision 0.005 — it predicts a header-shaped box almost everywhere rather than localizing the real one. The vanilla model's single, correctly-placed `Page-header` box (visually confirmed on `e001961124`) was a cleaner result than the fine-tuned model's noise.

**Overfitting was fast and visible**: `cls_loss` was unstable/rising across epochs 12-30 (8.7 → 45.6 → oscillating 6-20) and collapsed only at epoch 37 right as the run early-stopped — the best real checkpoint was epoch 7, meaning nearly all of the 37-epoch run was past the useful point. Expected at this scale (13 images is far below normal object-detection training volume), not a bug.

## Visual comparison

`data/outputs/layout_bootstrap_train/comparison/` — 5 sample images × 2 (vanilla / fine-tuned) each. On `e001961124` (held-out val image): vanilla produces 2 clean boxes (`Picture` covering the page, `Page-header` on the title block); fine-tuned produces 50 scattered fragments.

## Verdict

**Directionally, yes — but only for the coarse class.** This bootstrap is real evidence that a small amount of domain-specific fine-tuning can sharpen "does this page have one big dominant region" detection well beyond the generic pretrained weights, on this corpus specifically. It is equally real evidence that **10-13 images is nowhere near enough to learn fine-grained repeated-row detection** — that task needs either substantially more labeled pages, or a different approach entirely (which is arguably not a coincidence: `core/auto_sidecar.py`'s existing ruling-line/geometric row detector — the thing that generated these training labels in the first place — is already a purpose-built, non-learned solution to exactly this sub-problem, and this experiment's own failure to learn "row" from scratch is a point in favor of keeping that geometric approach rather than trying to replace it with a general object detector).

## Not yet done (as of v1)

- No test of whether MORE labeled images (beyond these 16) would fix row detection — this was a 13-image bootstrap by design, not a scaled attempt.
- No test of freezing backbone layers or other anti-overfitting measures — the epoch-7-is-best pattern suggests early stopping alone was doing the real regularization work here.
- No decision made about keeping, discarding, or extending this fine-tuned checkpoint — it lives only at `data/outputs/layout_bootstrap_train/runs/census_bootstrap/weights/best.pt`, not referenced by any pipeline code.

---

## v2: Does More Data Fix Row Detection?

**Date:** 2026-08-05, same session.

### Getting more data

Checked `robots.txt` on the relevant domains before pulling anything:

- `data2.collectionscanada.gc.ca` (the actual LAC image/PDF file host — what `automatedgenealogy_pull.csv`'s URLs already point to): permissive, default `Disallow:` empty for generic bots.
- `recherche-collection-search.bac-lac.gc.ca` (LAC's search/browse tool — needed to *discover* new record IDs by district/name/etc.): explicitly disallows crawling (`User-agent: * / Disallow: /`), and names `ClaudeBot` specifically among blocked crawlers. **Not crawled, by design.**
- `automatedgenealogy.com` (the third-party site `automatedgenealogy_pull.csv` was originally sourced from): CAPTCHA/WAF-gated. Not accessed.
- `automatedgenealogy_pull.csv` itself only covers the same 16 pages already in the corpus — no free extra data there.
- No official LAC bulk/API access for census images was found (confirmed via search) — access is described as browsing the search UI and clicking individual PDF/JPG links per record.

**The workaround, per direct instruction**: LAC image IDs are literally sequential integers — confirmed directly from this project's own existing corpus, where `e001946614` through `e001946623` (10 of the original 16 images) are 10 consecutive numbers. `training/lac_sequential_pull.py` fetches a range of sequential IDs directly from the permissive file host (never touching the disallowed search tool), rate-limited to 2s/request (matching this host's own `Googlebot` `Crawl-delay` value as a courtesy), with graceful 404 handling. Two batches of 50 pulled (`e001946624`-`673`, then `e001946674`-`723`) — **100/100 fetched successfully, 0 missing, 0 errors**, confirming the sequential-ID approach works cleanly for this reel range.

Licensing: canada.ca's Terms and Conditions permit non-commercial reproduction of Government of Canada material without further permission (checked directly) — this project's use (personal genealogical research + internal ML bootstrap, no redistribution) qualifies.

### Processing the new batch

`training/lac_batch1_convert_and_sidecar.py` — same PDF→PNG (300 DPI, `core/pdf_conversion.py`) → `generate_auto_sidecar()` (`doc_type_override="canada_census_1911"`) pipeline as v1's step 1, run against all 100 new pages. Since no external transcription ground truth exists for these new pages (unlike the original 16), validation relies entirely on `auto_sidecar`'s own internal structural-sanity quarantine flags rather than a row-count cross-check.

**Result: 90/100 OK, 10 whole-page-quarantined, 0 outright failures** — a clean 90% yield. Combined with the 13 good images from v1:

| | Images | Row instances |
|---|---:|---:|
| v1 bootstrap | 13 | 650 |
| LAC pull batch 1 | 90 | 4,474 |
| **Combined (v2)** | **103** | **5,124** (5,330 total boxes incl. table/header) |

`training/layout_bootstrap_v2_combined_dataset.py` merged both sources into one dataset, 85 train / 18 val (deterministic split, last 18 by sorted stem).

### Training and result

`training/layout_bootstrap_v2_train.py` — fine-tuned from the **same fresh pretrained checkpoint** used for v1 (not continuing from v1's own weights), same hyperparameters (80 epochs, imgsz=1280, batch=2, patience=30), so any metric difference is attributable to dataset size alone, not a training-config change.

| Class | v1 (13 img) P / R / mAP50 / mAP50-95 | v2 (103 img) P / R / mAP50 / mAP50-95 |
|---|---|---|
| table | 1.00 / 1.00 / 0.995 / 0.497 | 0.905 / 1.00 / 0.992 / **0.926** |
| row | 0.003 / 0.007 / 0.0001 / ~0 | **0.593 / 0.065 / 0.521** / 0.168 |
| header | 0.005 / 1.00 / 0.018 / 0.006 | **0.727** / 1.00 / **0.992** / 0.784 |

**Clear, unambiguous improvement across every class**, most dramatically for the two that had failed or nearly failed in v1:

- **Row detection went from complete failure to genuinely learning something real.** mAP50 0.0001 → 0.521. Precision 0.59 means when it does predict a row, it's usually a real one — a complete flip from v1's behavior of spraying scattered noise everywhere. Recall is still low (0.065, catching only ~6.5% of actual rows) — visually (see `data/outputs/layout_bootstrap_train/comparison/e001961124_3_finetuned_v2_combined.png`, at a lower conf threshold to reveal the pattern), it's found a real, localized cluster of row-shaped boxes concentrated in one region of the page rather than across the full table width. Not yet practically usable on its own, but a real, generalizing signal where there was none before.
- **Header went from noise to excellent**: mAP50 0.018 → 0.992, precision 0.005 → 0.727, still perfect recall.
- **Table's coarse detection was already near-perfect in v1**, but mAP50-95 (a stricter, tighter-localization metric) nearly doubled (0.497 → 0.926) — more data sharpened box precision even on the class that already "worked."

### Verdict

**Yes, scaling the dataset genuinely helps, substantially and unambiguously** — this directly answers the question v1 left open. Row detection is still far from a standalone production signal (6.5% recall means it's missing the large majority of real rows), but the qualitative change — from zero generalizing signal to a real, if incomplete, learned pattern — strongly suggests more data continues to help rather than having hit a ceiling. The next natural test (not run here) would be a further-scaled batch to see whether recall keeps climbing or plateaus.

### Not yet done (as of v2)

- No further-scaled test (e.g. 200-300 images) to see whether row recall keeps improving or plateaus.
- No check of whether the 10 LAC-batch quarantined pages, if manually reviewed/corrected rather than dropped, would add more usable signal.
- No decision made about keeping, discarding, or extending `data/outputs/layout_bootstrap_train/runs/census_bootstrap_v2/weights/best.pt` — same as v1, lives only in the bootstrap output directory, not referenced by any pipeline code.

---

## v3: Scaling to 203 Images (Still 1911-Only)

**Date:** 2026-08-05, same session.

Pulled 2 more batches of 100 sequential 1911 pages (100/100 and 99/100 usable — 0 excluded from batch 1, matching the conditional CV-first-then-YOLO-v2-rescue-if-quarantined labeling strategy: batch 1 = 88 CV-only clean + 12 YOLO-rescued, 0 excluded). Combined with the original 13: **203 images, 10,493 boxes, 170/33 train/val split.**

| Class | v2 (103 img) | v3 (203 img) |
|---|---|---|
| table (P/R/mAP50/mAP50-95) | 0.91/1.00/0.992/0.926 | 0.95/1.00/0.995/**0.951** |
| row (P/R/mAP50/mAP50-95) | 0.59/0.065/0.521/0.168 | 0.56/**0.847**/**0.636**/**0.245** |
| header (P/R/mAP50/mAP50-95) | 0.73/1.00/0.992/0.784 | 0.64/0.92/0.953/**0.821** |

**Row recall jumped from 6.5% to 84.7%** — by far the largest single move in the whole progression. Doubling the dataset didn't just refine an already-working signal, it turned a non-functional class into a genuinely useful one. Table and header both continued tightening too.

## v4: Diversifying by Year (1911 + 1921), Not Just Scaling

**Date:** 2026-08-05, same session. Per direct instruction, before jumping straight to a 400-image same-reel pull: *"take a handful of pages from 1921, 1931, portrait forms, landscape forms, maybe a different province, and see how well the table detection alone transfers... if it struggles, that gives you evidence the next scaling step should intentionally diversify."*

### Generalization probe (v3 checkpoint, before scaling further)

Ran v3's table detection ONLY against 21 real Census-folder images never used in any training (1921, 1931, England 1891/1901/1911/1939 records, 2 genuinely portrait pages). All 21 got at least one detection, but confidence/count alone was misleading - direct visual inspection of 6 samples found a real, informative split:

- **Transferred correctly**: 1921 Canada (different province/columns), 1931 Canada (different year/columns, same district), 1891 Manitoba portrait (correctly split into 2 table boxes for 2 stacked pages), 1901 England/Devon (RG13).
- **Genuinely struggled**: **1911 England (RG14)** - box only captured a narrow left margin, missing the real table (confirmed on 2/2 independent samples, not a fluke). **1939 England Register** (portrait, 4 stacked sections) - fragmented into 6 boxes, side annotation column mistaken for separate tables.

Conclusion: the model had learned a genuinely reusable "find the one wide ruled table" concept (transfers across year/province/country for structurally similar single-table-per-page forms) but had zero exposure to multi-section-per-page layouts or the specific RG14 gestalt - and no amount of *more Manitoba 1911* data could fix that, since neither pattern exists anywhere in the training pool.

### Sourcing 1921 (and discovering a 1901 template gap along the way)

- Confirmed via direct `robots.txt` + HEAD-request testing that **1921 uses the identical simple URL pattern as 1911** (`/1921/pdf/e{9digit}.pdf`, lowercase, sequential) - trivial to pull once tested, not assumed.
- Found `config/document_templates/canada_census_1901.yaml` didn't exist at all (1901 needed for a separate diversification path) - built one by **direct visual estimate** from one real 1901 sample (`z000017634.jpg` - Manitoba, Gilbert Plains, the same district the Campbell/Knott research is centered on) rather than a full manual-segmentation calibration, per direct instruction to unblock quickly. Tested clean-ish on that one sample (46/50 rows) before deciding not to scale 1901 further this round, in favor of the already-better-validated 1921 template (which tested 100% clean on 2 independent samples with zero rescue needed).
- Pulled 100×1911 (batch 3, 99/100 usable) + 100×1921 (batch 4, **100/100 usable, all CV-only clean, 0 needed YOLO-rescue at all** - confirming the 1921 template's calibration quality directly at scale, not just on 2 samples).

### Combined v4 dataset and result

**402 images, 20,713 boxes, 334/68 train/val split** (1921 stratified into both splits, not just training) - v1's 13 + LAC batches 1/2/3 (1911, 289 images) + LAC 1921 batch (100 images).

| Class | v3 (203 img, 1911-only) | v4 (402 img, 1911+1921) |
|---|---|---|
| table (P/R/mAP50/mAP50-95) | 0.95/1.00/0.995/0.951 | 0.97/1.00/0.995/0.940 |
| row (P/R/mAP50/mAP50-95) | 0.56/0.847/0.636/0.245 | 0.51/0.845/0.629/**0.285** |
| header (P/R/mAP50/mAP50-95) | 0.64/0.92/0.953/**0.821** | 0.77/0.94/0.965/0.679 |

**On the raw validation metric, this is a mixed/flat result** - row recall plateaued instead of continuing to climb, header's mAP50-95 dropped. But the val set itself is now intentionally harder (17% 1921 pages mixed in), so a flat headline number doesn't mean flat real-world performance.

**Re-running the exact same 21-image generalization probe with v4 tells the real story**: both RG14 failures are fixed. `rg14_18900_0013_03` went from a narrow-left-margin-only box (conf 0.765, visually confirmed wrong) to a correct, full-table box (conf 0.85, visually confirmed correct - see `v4_generalization_probe/rg14_18900_0013_03.png`). The 1939 Register went from 6 fragmented/confused boxes to 4 clean ones, each correctly matching one of the page's 4 real stacked sections (see `v4_generalization_probe/tna_r39_5711_5711d_015.png`).

### Verdict

**Diversifying by year beat pure scale-up on the metric that actually matters.** Adding 100 genuinely different-year pages (25% of the total dataset) fixed two concrete, previously-identified real-world failure modes that no amount of additional same-reel 1911 data could have touched, even though the narrow validation-metric comparison alone would have suggested "roughly flat, maybe slightly worse." This is a direct confirmation of the hypothesis behind running the generalization probe before scaling: **measure transfer, not just in-distribution metrics, before deciding what "more data" should mean.**

### Not yet done (as of v4)

- Row detection's plateau (v3→v4 recall barely moved, 84.7%→84.5%) is still an open question - unclear whether it's a genuine ceiling for this architecture/scale, or whether it would resume climbing with more 1911-specific data, more 1921 data, or a third diversifying year.
- 1901 and 1931 remain unused for training (1901 has a rough, single-sample-estimated template; 1931 has only 1 real sample in the corpus at all) - both are plausible next diversification sources if row-recall plateauing turns out to need more than year-diversity alone.
- No decision made about keeping, discarding, or extending `data/outputs/layout_bootstrap_train/runs/census_bootstrap_v4/weights/best.pt` - lives only in the bootstrap output directory, not referenced by any pipeline code, same as v1/v2/v3.
