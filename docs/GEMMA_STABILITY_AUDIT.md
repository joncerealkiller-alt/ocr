# Gemma Stability Audit (Second-Pass Classification)

**Date:** 2026-08-04
**Sandbox:** `benchmark/gemma_stability_sandbox.py` (candidate selection + 2-pass reclassification), `benchmark/gemma_stability_analysis.py` (analysis)
**Output data:** `data/outputs/gemma_stability_sandbox/` (`candidate_set.json`, `pass1.json`, `pass2.json`, `unstable_pairs.csv`)
**Isolation:** confirmed clean — checksum diff of `pipeline.db` and every bucket CSV showed zero changes before/after the run; all 683×2 = 1366 classify calls returned `error=None`.

## Headline result

**0/683 (0.0%) category changes between pass 1 and pass 2.** Every single
classification — category, confidence, reason text, and the captured
category-token logit distribution — was **byte-identical** across both
passes, for all 683 images, under fixed weights and greedy decoding.

This is a stronger result than "stable decisions": it is full
determinism. Given the same pixels, the same rendered prompt, and the
same model, Gemma reproduced its own output exactly, down to the
generated text and the softmax distribution at the category token.

## Correction to the record: `do_sample: false`

`config/models/gemma.yaml` sets `do_sample: false` — Gemma runs greedy
decoding here, not sampling. `temperature`/`top_p`/`top_k` are
configured but never applied (`_run_generate()`'s
`if self.config.do_sample:` gate never executes). This audit confirms
the practical consequence directly: with sampling off, there is no
observed run-to-run variation at all, including no GPU floating-point
non-determinism materializing as a decision flip across 683 images × 2
independent passes. Any instability in this pipeline's Gemma
classifications is not coming from the decoding step itself.

## A second, more consequential correction (twice-corrected — see update below)

Earlier this session, the [Fresh Pass vs. Checkpoint Comparison](FRESH_PASS_VS_CHECKPOINT_COMPARISON.md)
found 93/1750 (5.3%) classification changes between the `reference_pipeline_v3`
checkpoint and a fresh end-to-end rerun, with pixels, preprocessing, and
model config held byte-identical, and concluded this was attributable to
**"Gemma's own instability."** This audit shows that conclusion does not
hold: Gemma is provably 100% deterministic under identical conditions,
now confirmed at two different scales (683 images here, and the full
1750-image corpus — see [Fresh Pass vs. v4 Checkpoint Comparison](FRESH_PASS_VS_V4_CHECKPOINT_COMPARISON.md),
0 differences in bucket, confidence, or reason text). Something about
the *conditions* differed between the `v3` checkpoint and the fresh run
that produced the 93 changes, not the model's behavior under fixed
conditions.

**First attempted explanation (also wrong, corrected here)**: this
section originally claimed the fresh run's rendered prompt included 3
new categories (`photo_collage`/`casual_photo`/`cemetery_photo`, 12
categories vs. the checkpoint's 9) as the residual explanation, since
none of them were individually chosen. That claim doesn't hold up either
— directly checked `config/taxonomy.yaml`: all 3 of those categories
have `classifier_guidance: null`, and `categories_for_classifier_prompt()`
filters on `classifier_guidance` being set. Confirmed by actually
rendering the prompt (`Taxonomy.categories_for_classifier_prompt()`):
**exactly 9 categories**, matching the checkpoint, not 12. Those 3
categories were added to the taxonomy for the ground-truth review UI
but were never given prompt-guidance text, so the classifier has never
been able to select them — this also fully explains why every
`photo_collage`/`casual_photo`/`cemetery_photo` bucket has stayed at 0
across every run this session, including the [E] human-ground-truth
misses above.

**Real explanation, confirmed via `git diff` on `config/prompts/classifier_classify_v1.txt`**:
the checkpoint's original prompt manually line-wrapped each category's
guidance text at ~72 characters, with a 2-space continuation indent —
e.g. `- website_screenshot: a screenshot of a genealogy or archive
website's\n  user interface - search results, ...` (real newlines, real
indentation, one bullet spanning 6-9 lines). `render_classifier_category_block()`
emits the identical words but as **one long unwrapped line per
category** — no embedded newlines or continuation indent at all. Same
9 categories, same wording, substantially different token sequence
(dozens of newline/indentation tokens removed per category, times 9).
This is a real, structural prompt-content difference between the
checkpoint and fresh runs — not "one line-wrap artifact" as the earlier,
too-casual phrasing put it, but a full reformatting of every category's
guidance text. **Conclusion: the 5.3% checkpoint-vs-fresh drift is best
explained by this line-wrapping/whitespace difference in the rendered
prompt, not by new categories and not by intrinsic model instability.**
This is now the third and (pending any further evidence) final
explanation in this chain of corrections — recorded honestly rather than
silently replacing the previous wrong one.

## [A] Stability Summary

| | Count | % |
|---|---|---|
| Identical category (stable) | 683 | 100.0% |
| Changed category (unstable) | 0 | 0.0% |

## [B] Confidence Analysis

All 683 pairs: mean confidence delta = **+0.0000**, median = **+0.0000**,
variance = **0.0**. Confidence values were literally identical between
passes for every image (see [B] note below) — a direct consequence of
full determinism, not a separate finding.

## [C] Reason Stability

Not applicable in the 4-way form specified (all four tiers presuppose at
least one changed decision or reasoning variant to classify). With 0
unstable pairs, and reason text byte-identical for all 683 stable pairs,
every case is trivially "Tier 1: same category + identical reasoning" —
stronger than "similar," since the text matches character-for-character.
`data/outputs/gemma_stability_sandbox/unstable_pairs.csv` was generated
for this step and is empty (header only).

## [D] Tower Correlation

Since [A] found 0 unstable images, "does instability correlate with
tower uncertainty" has no population to test. Reporting the tower
context of the (stable) candidate set for reference instead:

Tower `consensus_category` distribution across the 683 candidates:

| Tower consensus | Count | % |
|---|---|---|
| majority | 437 | 66.5% |
| split | 179 | 27.2% |
| unanimous | 40 | 6.1% |
| complete_disagreement | 1 | 0.2% |

96.2% of the candidate set disagrees with the tower consensus by
construction (most of the 683 were drawn from the tower-disagreement
list). The finding this enables: **Gemma's disagreement with the towers
is a stable, reproducible disagreement** — not classifier noise that
happens to land on one side or the other. Whatever the Decision Engine
does with tower/Gemma conflicts, it's resolving a real, repeatable
difference of "opinion," not averaging out randomness.

## [E] Human Ground Truth

22/683 candidates have a real human label (excluding `ignored` /
`needs_new_bucket` / `bad_deskew` sentinels). Per the task's own
instruction, this is not extrapolated to the full corpus — it's a tiny,
non-representative slice drawn specifically from hard/flagged cases.

- **14/22 (63.6%) match** the human label, both passes identically (full
  agreement between pass 1 and pass 2 confirms these aren't
  borderline-flip cases either).
- **8/22 (36.4%) miss**, and the misses show a clean pattern: **all 8**
  are cases where the human label is one of the newer/finer-grained
  categories (`photo_collage`, `casual_photo`, `cemetery_photo` — 5
  cases — or a `map_land_record` vs. `dense_tabular_rows`/
  `handwritten_ledger` boundary call — 3 cases) and Gemma instead
  produced the older, coarser category (`portrait_photo`,
  `printed_document`, `dense_tabular_rows`, `handwritten_ledger`).

This reads as **taxonomy-adoption lag, not instability**: Gemma isn't
failing to be consistent, it's consistently defaulting to the coarser
pre-existing bucket on cases that need one of the newer, more specific
categories. Since every miss reproduced identically in both passes, this
is a targeting/prompt-guidance problem to address separately (e.g.
`classifier_guidance` wording for the 3 new categories, or fine-tuning),
not something this audit's scope covers.

## [F] Pre-Reasoning Confidence (logit-based)

`logit_stats` (top-1 probability, top-2 margin, entropy at the
`category:` token) captured for all 683/683 images, in both passes,
identically (consistent with [A]'s full-determinism finding).

Distribution across the 683 images (pass 1 == pass 2 in every case):

| Percentile | top1_prob |
|---|---|
| p5 | 0.7546 |
| p10 | 0.9046 |
| p25 | 0.9979 |
| p50 | 1.0000 |
| p75 | 1.0000 |
| p90 | 1.0000 |

- 489/683 (71.6%) images have top1_prob ≥ 0.999 — near-total certainty at the token level.
- 61/683 (8.9%) are below 0.90; 23/683 (3.4%) below 0.70; 3/683 (0.4%) below 0.50.
- margin: p5=0.5347, median=1.0000. entropy: median=0.0000, p90=0.3152, p99=0.7414, max=1.0960.

This is a wider, more usable range than the 4-image feasibility sample
suggested (0.998–1.0) — at n=683 there's a real low-confidence tail worth
having. But since [A] found zero decision instability, this signal
**cannot be validated against instability** in this dataset (there's
nothing to discriminate against) — it would need to be checked against
something else, e.g. correlation with human-label misses in [E] or with
tower disagreement, as a separate follow-up.

**Availability in production**: this signal does not exist anywhere in
the current inference stack. It required a local, non-invasive copy of
`GemmaLoader._run_generate()`'s call to
`self.model.generate(**inputs, **gen_kwargs)`
(`core/loaders/gemma_loader.py`) with `output_scores=True,
return_dict_in_generate=True` added — this only changes what `generate()`
*returns* (the per-step score tensors), not what it generates, since
`do_sample=False` means the returned sequence is unaffected. To expose
this for production Decision Engine use, the same two kwargs would need
to be added to `core/loaders/gemma_loader.py`'s own `_run_generate()`
call, plus the token-position-location and softmax-extraction logic
demonstrated in `benchmark/gemma_stability_sandbox.py`'s
`_generate_with_scores()`.

## Implication for the Decision Engine

The original open question was whether tower/Gemma conflicts represent
"stable but difficult decisions or intrinsic instability." This audit
answers it cleanly: **stable**. Every one of the 683 flagged/disagreement/
contradictory-reasoning/human-corrected images produced an identical
Gemma decision twice, under identical conditions. There is no
run-to-run noise for a Decision Engine to average out or be robust
against — a Gemma/tower conflict is a real, reproducible disagreement
between two different evidence sources, not classifier jitter on one
side. That reframes the design question from "how do we handle Gemma's
instability" to "how do we adjudicate two internally-consistent but
disagreeing sources" — the pre-reasoning confidence tail in [F] and the
taxonomy-adoption-lag pattern in [E] are the two concrete signals this
audit surfaced that could feed that adjudication, both flagged here as
follow-up work rather than resolved by this audit itself.
