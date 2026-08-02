# Benchmark 2.3 — Gemma Input Qualification

**Status**: complete — all 6 variables qualified, see summary table at
the bottom of this document. Purpose: qualify the image presented to Gemma
(resolution, resize algorithm, color, contrast, sharpening, denoising)
before expanding to the full multi-encoder Routing Audit — so that
audit compares encoders against a known-good, evidence-based Gemma
baseline rather than an unexamined one. One variable changes per
experiment; everything else stays identical. Evidence tagged
MEASURED / INFERENCE / PROPOSED throughout, per established discipline.

**Relationship to other documents**: this follows directly from
`docs/BENCHMARK2_METADATA_LAYER_QUALIFICATION.md`'s Routing Audit work
— the multi-encoder comparison is paused until this qualification
completes and the image pipeline is frozen.

**Mechanistic finding, established before any classification testing
began** (read-only inspection of `Gemma4ImageProcessor`, no inference):
`pixel_values` shape is identical — `(1, 2520, 768)` — whether the
input image is pre-resized to 224×224, 1400×1400, or left at native
resolution, for a fixed `image_seq_length`. The processor always
resizes internally to a fixed patch grid; **resolution/resize-algorithm
experiments are not testing "how much Gemma sees"** (already fixed by
token budget, separately qualified and ruled out) — **they're testing
whether compounding two resize passes (mine, then the processor's own
fixed bicubic resize) changes the classification outcome**, since a
small enough intermediate step can discard detail the second resize
can't recover.

---

## Variable 1: pre-resize resolution (longest edge)

```yaml
pre_resize_resolution:
  status: qualified
  result: threshold_effect
  values_tested: [224, 448, 896, 1400, native]
  images_tested: 3  # 2 known disagreements, 1 known-correct control
  finding: >
    2 of 3 test images stayed flat across the entire range. 1 image
    (c10301.601, the genuinely ambiguous mixed-content case) flipped
    category at a sharp threshold: printed_document at 224px, portrait_photo
    at 448px and every larger value tested including native. Not a
    gradual drift - a clean step change, then stable above it.
  safe_region:
    longest_edge_px: ">=448"
  production_risk: low
  production_status: already_compliant
  production_status_note: >
    GemmaLoader._run_generate() passes the native-resolution PIL image
    directly to the processor with no intermediate pre-resize step
    today - this failure mode is not live in the current pipeline. It
    is a real, latent risk for any FUTURE preprocessing step
    (thumbnailing, caching) that might pre-resize below 448px longest
    edge before Gemma sees the image.
  mechanism: INFERENCE - plausible, not independently confirmed
  mechanism_detail: >
    Downsampling to a small intermediate size (224px) discards detail
    that the processor's own subsequent fixed-size resize cannot
    recover, regardless of what happens after. 448px+ apparently
    retains enough real detail that the processor's resize produces an
    equivalent-quality result to using the native image directly.
  evidence_location: data/outputs/benchmark2_gemma_input_qualification/resolution/
```

---

## Variable 2: resize algorithm (LANCZOS / BICUBIC / BILINEAR)

Resolution held fixed at 896px longest edge (confirmed safe region from
Variable 1) so any effect found here is attributable to algorithm
choice alone, not confounded with the resolution threshold. Same 3 test
images.

```yaml
resize_algorithm:
  status: qualified
  result: no_effect
  values_tested: [LANCZOS, BICUBIC, BILINEAR]
  fixed_at: {longest_edge_px: 896}
  images_tested: 3
  finding: >
    Byte-identical category and confidence across all 3 algorithms,
    for all 3 test images (9 calls total). A clean, flat qualified
    negative per the stop-condition rule - no further tuning of this
    variable, proceed to the next one.
  safe_region: any  # no measurable preference among the 3 tested
  production_risk: none
  production_status: not_applicable  # production doesn't pre-resize at all (see Variable 1)
  evidence_location: data/outputs/benchmark2_gemma_input_qualification/resize_algorithm/
```

---

## Variable 3: contrast normalization

Resolution held fixed at 896px longest edge (Variable 1's safe region),
resize algorithm fixed at LANCZOS (Variable 2 showed no preference), so
any effect found here is attributable to contrast treatment alone. Same
3 test images. Jon's original 4 proposed conditions (None / Autocontrast
/ CLAHE / "existing pipeline normalization") collapse to 3 real distinct
conditions: `core/image_preprocessing.py`'s `PREPROCESSING_PROFILES["autocontrast"]`
**is** the existing production pipeline's normalization
(`DEFAULT_PREPROCESSING_PROFILE = "autocontrast"`), so it was tested
once, not twice.

```yaml
contrast_normalization:
  status: qualified
  result: no_effect
  values_tested: [none, autocontrast_production_default, clahe]
  fixed_at: {longest_edge_px: 896, resize_algorithm: LANCZOS}
  images_tested: 3
  finding: >
    Byte-identical category and confidence across all 3 contrast
    conditions, for all 3 test images (9 calls total). Verified this
    is a genuine negative and not a silent no-op bug: measured actual
    pixel-level deltas between conditions before trusting the flat
    classification result - CLAHE produced a real mean absolute
    per-pixel difference of ~8-9/255 versus the untreated image
    (autocontrast ~0.8-5.5/255, smaller since it's a global stretch on
    already near-full-range scans). The contrast treatments demonstrably
    changed the image; Gemma's classification was unmoved regardless.
  safe_region: any  # no measurable preference among the 3 tested
  production_risk: none
  production_status: already_compliant  # production already applies autocontrast by default; shown here to be a no-op for classification either way
  evidence_location: data/outputs/benchmark2_gemma_input_qualification/contrast/
```

---

## Variable 4: grayscale vs RGB

Resolution/algorithm/contrast held at their qualified-safe settings
(896px, LANCZOS, no contrast treatment - simplest baseline). Two
conditions: untouched RGB vs. `core.image_preprocessing.grayscale()`
(desaturate, convert back to 3-channel RGB so tensor shape is
unaffected).

**Confound caught before trusting the result**: the original 3 test
images are microfilm scans, and turned out to be monochrome-in-RGB
already (R=G=B, measured channel spread = 0.000 for all 3) - `grayscale()`
is a no-op on them. Testing only on these would have produced a
meaningless "no effect" (nothing was different to have an effect).
Added a 4th test image with real, measured color content (a genealogy
chart screenshot, channel_spread=28.3, the highest of 3 candidate
screenshots checked) so the variable is actually exercised at least once.

```yaml
grayscale_vs_rgb:
  status: qualified
  result: no_effect
  values_tested: [rgb, grayscale]
  fixed_at: {longest_edge_px: 896, resize_algorithm: LANCZOS, contrast: none}
  images_tested: 4  # 3 microfilm scans (monochrome-in-RGB already, confound - see below) + 1 real-color screenshot added specifically for this variable
  finding: >
    Byte-identical category and confidence across both conditions for
    all 4 images including the genuinely colorful one. A real
    qualified negative, not a vacuous one - confirmed the color-bearing
    test image actually lost its color channel information under the
    grayscale condition before trusting the flat classification result.
  safe_region: any
  production_risk: none
  production_status: not_applicable  # production doesn't grayscale-convert today
  evidence_location: data/outputs/benchmark2_gemma_input_qualification/grayscale/
```

---

## Variable 5: sharpening

Resolution/algorithm/contrast/color held at their qualified-safe
settings (896px, LANCZOS, no contrast treatment, RGB). Three
conditions, two genuinely distinct mechanisms already in
`core/image_preprocessing.py`: `sharpen()` (global `ImageEnhance.Sharpness`
scalar, factor=2.0) and `unsharp_mask()` (blurred-copy-subtraction edge
filter, radius=1.5, amount=1.2 - a different mechanism, documented in
the source as looking less artificial on scanned handwriting than a
flat global boost). Same 3 test images (all monochrome-in-RGB
microfilm scans - sharpening operates on luminance/edges regardless of
color channels, so the Variable 4 color confound doesn't apply here).

```yaml
sharpening:
  status: qualified
  result: no_effect
  values_tested: [none, sharpen_global, unsharp_mask]
  fixed_at: {longest_edge_px: 896, resize_algorithm: LANCZOS, contrast: none, color: rgb}
  images_tested: 3
  finding: >
    Byte-identical category and confidence across all 3 conditions, for
    all 3 test images (9 calls total). Verified genuine: measured real
    pixel-level deltas before trusting the flat result - unsharp_mask
    produced mean absolute per-pixel difference of ~4.6-7.1/255 vs.
    untreated (sharpen_global ~2.3-3.9/255, smaller as expected since a
    global scalar is a gentler effect than an edge-targeted filter).
  safe_region: any
  production_risk: none
  production_status: not_applicable  # production doesn't sharpen today
  evidence_location: data/outputs/benchmark2_gemma_input_qualification/sharpening/
```

---

## Variable 6 (final): denoising

Resolution/algorithm/contrast/color/sharpening all held at their
qualified-safe settings. Two conditions: untouched vs.
`core.image_preprocessing.denoise()` (gentle `MedianFilter(size=3)`;
its own docstring already flags aggressive denoising as a handwriting-
detail risk, which is why this stays at the deliberately mild default
rather than testing a stronger radius).

```yaml
denoising:
  status: qualified
  result: minor_effect
  values_tested: [none, denoise_median3]
  fixed_at: {longest_edge_px: 896, resize_algorithm: LANCZOS, contrast: none, color: rgb, sharpening: none}
  images_tested: 3
  finding: >
    Not a clean flat negative like Variables 2-5 - recorded honestly
    rather than rounded to "no_effect". Category never flipped on any
    of the 3 images. But c10301.601 (the same genuinely-ambiguous image
    that flipped category outright in Variable 1) showed a real
    confidence dip under denoising: 0.98 -> 0.95, category unchanged
    (portrait_photo both times). The other 2 images were fully flat
    (identical category and confidence). Verified the treatment itself
    is real: mean absolute per-pixel difference ~2.1-4.9/255 vs.
    untreated, comparable in magnitude to sharpening's effect on pixels
    (Variable 5) despite producing a confidence wobble there where
    sharpening produced none - the two mechanisms are not equivalent
    in effect even at similar pixel-delta magnitude.
  safe_region: any  # no category flips, but treat the ambiguous-image confidence sensitivity as a known minor residual, not zero
  production_risk: low
  production_status: not_applicable  # production doesn't denoise today
  mechanism: INFERENCE - not independently confirmed
  mechanism_detail: >
    Median filtering can soften fine edge detail exactly on the kind of
    genuinely mixed/ambiguous content this image already represents
    (per Variable 1's finding it sits at the printed_document/
    portrait_photo boundary) - plausible the confidence dip reflects
    reduced boundary-relevant detail, not measurement noise, but this
    is inference, not confirmed by a dedicated follow-up test.
  evidence_location: data/outputs/benchmark2_gemma_input_qualification/denoise/
```

---

## Gemma Input Qualification — frozen (all 6 variables complete)

All planned variables (resolution, resize algorithm, contrast,
grayscale/color, sharpening, denoising) are now qualified. Summary:

| Variable | Result | Production risk |
|---|---|---|
| Resolution (pre-resize longest edge) | threshold_effect (<448px unsafe) | low, already compliant |
| Resize algorithm | no_effect | none |
| Contrast normalization | no_effect | none, already compliant |
| Grayscale vs RGB | no_effect | none |
| Sharpening | no_effect | none |
| Denoising | minor_effect (confidence only, one image) | low |

Only the resolution threshold represents a real production-relevant
finding, and production is already on the safe side of it (native
resolution passed directly, no pre-resize step exists today). The
Routing Audit (`docs/BENCHMARK2_METADATA_LAYER_QUALIFICATION.md`) can
now resume against this evidence-based Gemma baseline instead of an
unexamined one.
