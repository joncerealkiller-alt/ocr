# YOLOv26-Small Layout Detector — Test Comparison (Not Wired In)

**Date:** 2026-08-05
**Status:** Experimental only. `core/layout_detector_v26.py` is a standalone module, never imported by `core/manifest_pipeline.py` or any pipeline stage. All output isolated to `data/outputs/layout_detection_yolo26_test/` — confirmed `data/baseline_embeddings.json` and `data/pipeline.db` mtimes unchanged before/after this run.

## Setup

- Model: `Armaggheddon/yolo26-document-layout`, `yolo26s_doc_layout.pt` (small variant, per direct instruction — nano/medium also available).
- Package: `ultralytics==8.4.115` (newly installed, `pip check` clean, no conflicts with the existing environment).
- Trained on **DocLayNet v1.2** (11 classes: Caption, Footnote, Formula, List-item, Page-footer, Page-header, Picture, Section-header, Table, Text, Title) — a *different* dataset and label set from the current sensor's DocStructBench training (10 classes). Class names are not directly comparable between the two.
- License: MIT (code) / Apache-2.0 (project) — resolves the AGPL-3.0 concern attached to the current `doclayout_yolo` package.
- Sample set: all 23 images from the `Census` source folder (the same set manually verified against the current model earlier this session) + 5 extra spread images (handwritten_ledger, map_land_record ×2, portrait_photo, genealogy_chart) — 28 total, chosen specifically so results are directly comparable to what's already been visually reviewed.

## Result: detection counts, current model vs. v26-small

| Image | Current (DocStructBench) | v26-small |
|---|---:|---:|
| 14765_0304.jpg (the multi-panel outlier) | 72 | 45 |
| 1921_022-E002880409.jpg | 6 | 1 |
| rg14_18900_0013_03.jpg | 5 | 2 |
| rg14_18900_0087_03.jpg | 4 | 2 |
| rg14_04575_0097_03.jpg | 4 | **0** |
| Screenshot 2026-05-08 081916.png | 6 | 3 |
| tna_r39_5711_5711d_015.jpg | 6 | 8 |
| z000017634.jpg | 3 | 1 |
| 30953_148096-00296.jpg | 1 | **0** |
| 31228_4363955-00089.jpg | 1 | **0** |
| (remaining 19 images) | mostly 1-4 each | mostly 1-4 each, roughly comparable |

Full per-image data: `data/outputs/layout_detection_yolo26_test/detections.json`. Overlay PNGs for all 28: `data/outputs/layout_detection_yolo26_test/overlays/`.

## Notable findings

1. **The multi-panel outlier stayed an outlier.** `14765_0304.jpg` (the complex cover+instructions+example-table spread that produced 72 fine-grained detections from the current model) also produced the most detections from v26-small (45) — both models agree this page is structurally complex, even though the exact counts differ. Consistent behavior, not a contradiction.

2. **Three real misses**: `30953_148096-00296.jpg`, `31228_4363955-00089.jpg`, and `rg14_04575_0097_03.jpg` — v26-small returned **zero** detections at `conf=0.2` on all three, where the current model found at least one region on each (including a title+text+table+figure set on `rg14_04575_0097_03.jpg`). Current-model overlays for these three are saved separately at `data/outputs/layout_detection_yolo26_test/current_model_for_comparison/*_CURRENT_MODEL.png` so they're easy to eyeball against the (empty) v26 overlays in the main `overlays/` folder.

3. **v26-small generally returns fewer, more consolidated boxes** on the images where both models found something — e.g. 6→1, 5→2, 4→2. Whether that's "cleaner, less fragmented" or "under-detecting" isn't determined by this pass — no ground truth was checked against either model's output here, only the two models against each other.

## Not yet done

- No quantitative accuracy comparison (no ground truth for either model's boxes on this corpus).
- No test of the nano or medium v26 variants.
- No investigation into *why* the three zero-detection images failed — worth a closer look before drawing conclusions about which sensor is actually better for this corpus.
- No decision made about whether to keep, swap, or drop either sensor — this is a data-gathering pass only, per the explicit "test against, don't wire in" instruction.
