# Stage 1 vs Stage 4 before/after comparison (2026-08-04)

**Status**: measurement only, per Stage 4's own scope. No claim about
whether any observed drift is "correct" or "wrong" - no ground truth
exists to judge that against (see `project_no_comprehensive_ground_
truth` memory). Reports deltas and flip counts.

## A real bug found and fixed mid-comparison

The first run of this comparison found the physical dataset showed
**exactly zero delta on every single field** for all 1710 common images
- not small, literally `0.0000` with `n_nonzero=0` everywhere. That result
contradicted the semantic comparison (large, real drift on the same
corpus) and was investigated rather than reported at face value.

Direct verification: `analyze_image()` was run against three images
where genuinely-raw and current-working pixels differ (found via
`data/raw_stage0_recapture/`, the hash-verified raw reconstruction from
the earlier semantic-embeddings fix). The **stored "Stage 1" physical
report matched the working-copy (post-Stage-3) values exactly, not the
raw ones**:

| Image | Raw blur | Working blur | Stored "Stage 1 pre" blur |
|---|---|---|---|
| e001928017.png | 11523.02 | 11525.74 | 11525.74 (matches working) |
| e001946014.png | 13001.67 | 13388.4 | 13388.4 (matches working) |
| e001946614.png | 12365.02 | 12762.51 | 12762.51 (matches working) |

**Root cause**: `data/outputs/image_analysis/analysis_report.csv` (and
every `<name>_analysis.json` sidecar) was captured by a standalone call
to `analyze_manifest()` against `data/working` *after* Stage 3 (and
classification) had already run against this corpus - the exact same
bug class as `docs/REFERENCE_PIPELINE_V1.md` and the semantic-embeddings
incident fixed earlier this session, just never caught on the physical
side until this comparison surfaced it.

**Fix** (`scripts/recapture_image_analysis_raw.py`): reused
`analyze_manifest()` unchanged (same 8-worker/4-cv2-thread engine)
against the 1750 hash-verified raw copies. Added one small,
backward-compatible parameter to `analyze_manifest()` itself - `stage:
str = "stage1_image_analysis"` - so this recapture's own idempotency
scope doesn't collide with the (correct) Stage 4 physical capture's
`"stage4_validation_capture_physical"` records; every existing caller's
behavior is unchanged since the default matches what was hardcoded
before. Sidecars and the report CSV were written against the raw
copies, then promoted (file_path/`file_path` field rewritten from
raw-copy path to working path) into their real production locations,
overwriting the mislabeled data. The old mislabeled report was backed
up first, at `data/outputs/image_analysis_raw_recapture/
analysis_report_PRE_FIX_mislabeled_backup.csv`. DB: 1750 new
`stage1_image_analysis` stage_outputs rows recorded with an explicit
note distinguishing them from the earlier mislabeled capture (the
existing wrong rows were not deleted - append-only log, per this
project's established discipline).

**Sanity check the fix worked**: the raw pass produced real nonzero
deskew angles (e.g. -1.25°, -11.00°) for some images - the previous
(broken) capture showed `+0.00` for virtually everything, since it was
measuring already-deskewed pixels.

## [A] Source Evidence

### Semantic (`baseline_embeddings.json` vs `postprocessing_embeddings.json`)

1750 images, 8 encoders, 14000 pairs.

| Encoder | min | p01 | p50 | mean | max |
|---|---|---|---|---|---|
| dinov2 | 0.3665 | 0.7932 | 0.9953 | 0.9844 | 1.0000 |
| convnext | 0.2859 | 0.6585 | 0.9893 | 0.9684 | 1.0000 |
| naflex_siglip | 0.7126 | 0.8557 | 0.9912 | 0.9810 | 1.0000 |
| siglip_fixed | 0.6827 | 0.8862 | 0.9929 | 0.9854 | 1.0000 |
| eva02 | 0.6552 | 0.7898 | 0.9912 | 0.9792 | 1.0000 |
| beit | 0.2929 | 0.7003 | 0.9890 | 0.9691 | 1.0000 |
| swin | 0.3103 | 0.6496 | 0.9864 | 0.9671 | 1.0000 |
| mae | 0.6703 | 0.7655 | 0.9985 | 0.9843 | 1.0000 |

Overall: min=0.2859, mean=0.9774, max=1.0000. **12,587/14,000 pairs
(89.9%) fall below the 0.9999 threshold** used throughout this
session's equivalence work.

Bucket-prediction flip check (self-consistent: each snapshot scored
against a reference set built from that same snapshot, matching the
n=11 pilot's Checkpoint A/B design):
- consensus category or top voted bucket changed: **581/1750 images
  (33.2%)**
- per-encoder winner flips: 1788/14000 (12.8%)
- images with at least one encoder's vote flipped: **1217/1750
  (69.5%)**

### Physical (`analysis_report.csv` vs `analysis_report_postprocessing.csv`, post-fix)

1750 common rows.

| Field | mean delta | abs mean delta | min | max | n nonzero |
|---|---|---|---|---|---|
| deskew_angle_deg | +0.13 | 0.76 | -30.00 | +30.00 | 239/1750 |
| blur_laplacian_var | +2577.89 | 2627.94 | -12427.48 | +14917.71 | 1633/1750 |
| noise_residual_std | +3.55 | 3.71 | -9.03 | +17.96 | 1632/1750 |
| page_contrast_p5_p95_spread | +33.22 | 33.83 | -98.00 | +179.00 | 1584/1745 |
| page_luminance_mean | +1.68 | 11.51 | -98.68 | +88.94 | 1624/1745 |
| page_text_height_px | +0.08 | 0.15 | -4.00 | +6.00 | 189/1743 |
| page_ink_fraction | +0.005 | 0.010 | -0.42 | +0.66 | 1353/1745 |
| table_contrast_p5_p95_spread | +29.44 | 30.51 | -62.00 | +188.00 | 464/538 |

`page_method` (boundary-detection algorithm outcome) changed for
**117/1750 images (6.7%)**. `table_boundary` presence (detected vs not)
changed for **156/1750 images (8.9%)**.

## [B] Interpretation

The physical measurements are consistent with what Stage 3's two
operations are documented to do: `page_contrast_p5_p95_spread` and
`table_contrast_p5_p95_spread` both widened substantially on average
(+33.2, +29.4) - the expected direction and rough magnitude for an
autocontrast pass. Deskew changed the angle for 239/1750 images
(13.7%), consistent with most of this corpus being already
close-to-level scans and a real minority needing correction.

`blur_laplacian_var`'s large positive mean shift (+2577.89, i.e. pixels
read as measurably "sharper" post-processing on this metric) is a real,
measured side effect worth noting without over-interpreting: autocontrast
stretches the intensity histogram, which mechanically increases
edge-gradient magnitude (what Laplacian variance measures) even without
any actual focus change - this is a plausible mechanism, not a confirmed
one, and is offered as interpretation, not fact. The correlated increase
in `noise_residual_std` is consistent with the same mechanism (contrast
stretching amplifying both real edges and noise together).

The semantic drift is substantially larger than the earlier n=11 pilot
suggested (that pilot found near-zero drift, 0.0001-0.0024, with deskew
measuring exactly 0.000° for all 11 images due to a known false-zero
limit on this microfilm sub-corpus). At full-corpus scale, real deskew
correction is now measured for a meaningful minority of images (239),
and the resulting semantic drift is large for a large minority of
(image, encoder) pairs - 90% falling below a threshold that, in this
session's separate GPU-preprocessing work, was used to reject GPU resize
as numerically unsafe. The same statistical signature (large minority
divergence, small majority near-identical) appearing here is not
evidence of a bug the way it was there - Stage 3 is a real, intentional
pixel-changing operation, unlike GPU resize's supposed no-op - but it
does mean pre/post embedding drift from real preprocessing is
substantially larger, and more consequential (33% consensus-bucket
changes), than the small pilot indicated.

## [C] Conclusions

**Is the physical dataset now trustworthy for comparison?** Yes -
verified via the raw-vs-working spot check and the reappearance of
nonzero deskew angles in the corrected data. The original mislabeled
data is preserved (backed up, not deleted) for anyone who needs to see
what the bug produced.

**Does Stage 3 preprocessing do real, measurable, directionally-sensible
work?** Yes, on the evidence here - contrast spread widened as
expected from autocontrast, deskew corrected a real minority of skewed
images, and boundary/table detection outcomes shifted for a real
fraction of the corpus as a downstream consequence.

**Does that preprocessing change what the vision towers perceive?**
Substantially, for a large minority of images - 90% of (image, encoder)
pairs show cosine similarity below 0.9999, and roughly a third of images
have their tower-consensus bucket prediction change entirely. This is
the headline finding of the semantic side: preprocessing is not a
negligible perturbation for this corpus at full scale, contrary to what
the earlier small pilot suggested.

**What this does NOT establish**: which prediction (pre or post) is more
correct, whether the flipped 33% represent improvements or regressions,
or whether Stage 2's currently-gated profile-decision should change as
a result. Those require ground truth or a downstream decision this
document deliberately does not make.
