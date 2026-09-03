# Fresh Pass vs. `reference_pipeline_v4` Checkpoint Comparison

**Date:** 2026-08-05
**Script:** `benchmark/fresh_pass_vs_v4_checkpoint_comparison.py`
**Output:** `data/outputs/fresh_pass_vs_v4_checkpoint_comparison/changed_classifications.csv` (empty — see below)

## Setup

`data/outputs/reference_pipeline_v4/` was checkpointed 2026-08-04
~01:53, immediately before a fresh full-corpus run that completed and
was confirmed current as of 2026-08-05 ~01:19. Unlike the earlier
`v3`-vs-fresh comparison, **no classifier code, prompt, or taxonomy
change happened between the `v4` checkpoint and this run** — the token-
budget fix and the layout-detector addition both landed in the repo
*after* this run completed. So this comparison is a same-conditions
determinism check, not a before/after-a-change comparison.

## Result

**0/1750 (0.0%) differences** — identical bucket, identical confidence,
identical reason text for every one of the 1750 common images (all
1750 images present on both sides; nothing only-in-current or
only-in-checkpoint).

| Bucket | v4 | current | delta |
|---|---|---|---|
| printed_document | 1154 | 1154 | +0 |
| dense_tabular_rows | 198 | 198 | +0 |
| website_screenshot | 170 | 170 | +0 |
| map_land_record | 94 | 94 | +0 |
| portrait_photo | 80 | 80 | +0 |
| mixed_text_image | 29 | 29 | +0 |
| handwritten_ledger | 19 | 19 | +0 |
| genealogy_chart | 6 | 6 | +0 |
| photo_collage / casual_photo / cemetery_photo / uncertain_review | 0 each | 0 each | +0 |

## Interpretation

This extends the [Gemma Stability Audit](GEMMA_STABILITY_AUDIT.md)'s
683-image determinism finding (byte-identical output across two
independent passes under fixed conditions) to the **full 1750-image
corpus**, under real production conditions rather than an isolated
sandbox. Combined, both results say the same thing: Gemma's classify()
output is fully reproducible given identical pixels, prompt, and code —
any observed drift in this pipeline's history (e.g. the 93/1750
`v3`-vs-fresh changes) has always come from a genuine difference in
conditions between runs, never from run-to-run model noise. See the
Gemma Stability Audit's corrected explanation of the `v3` drift (a
real prompt line-wrapping/formatting difference, not new categories or
instability) for the concrete example of what such a "genuine
difference in conditions" actually looks like in practice.

**The `photo_collage`/`casual_photo`/`cemetery_photo` buckets are empty
in both snapshots** because those 3 taxonomy categories have no
`classifier_guidance` text defined (`config/taxonomy.yaml`) — they were
added for the ground-truth review UI but were never wired into the
classifier prompt (`categories_for_classifier_prompt()` filters on
`classifier_guidance` being set), so Gemma has never had them as
selectable options in any run. Not a bug in this comparison; a real,
still-open gap if those categories are meant to eventually route
production images.
