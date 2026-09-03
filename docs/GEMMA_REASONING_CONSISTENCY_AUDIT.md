# Gemma classification-vs-reasoning consistency audit (2026-08-04)

**Status**: internal consistency audit, not an accuracy benchmark. Covers
the full 1750-image fresh-run corpus for [A]/[C]/Special Investigation,
plus a fully-read (not sampled) analysis of the 21 mechanically-flagged
Contradictory cases and all 93 checkpoint-vs-fresh classification
changes for [B]/reasoning-stability.

**Methodology, stated plainly**: a mechanical keyword-marker pass
covers all 1750 reasons (reproducible, auditable, but not true semantic
judgment). It was iteratively corrected against my own direct reading
of its output - not assumed accurate. Two real false-positive sources
were found and fixed (negation blindness, e.g. "without a tabular
structure"; and `mixed_text_image`'s own definition legitimately
including portrait/document-shaped language for its photo/text
components). After both fixes, I read every one of the 21 remaining
mechanically-flagged Contradictory cases directly and re-tiered them by
hand - those are the final, human-verified numbers below, not the raw
mechanical output.

## [A] Counts by consistency tier

Mechanical pass, post-fix (1750 images):

| Tier | Count | % |
|---|---|---|
| Consistent | 1513 | 86.5% |
| Weak | 189 | 10.8% |
| Contradictory (mechanical) | 21 | 1.2% |
| Ambiguous | 27 | 1.5% |
| Self-Contradictory | 0 | 0.0% |

**After reading all 21 mechanically-flagged Contradictory cases directly**,
the true breakdown of that group is:

| True tier (of the 21) | Count |
|---|---|
| Genuinely Contradictory | 5 |
| Actually Ambiguous (self-acknowledged hedging in Gemma's own text) | 11 |
| Actually Consistent (remaining mechanical false positives) | 5 |

**Corrected full-corpus estimate**: Contradictory ≈ **5/1750 (0.29%)**,
Ambiguous ≈ 27 + 11 = 38/1750 (2.2%), Consistent ≈ 1513 + 5 = 1518/1750
(86.7%). Self-Contradictory is genuinely near-zero, consistent with the
field's own one-sentence, 40-word cap (`core/schema.py`'s
`ClassificationResult.reason`) leaving little room for a true internal
reversal within a single response.

The Weak and Consistent tiers beyond the 21 re-read cases were spot-
checked (12 each) but not individually re-verified at the same depth -
the spot check found 0/24 miscategorized, but this is a smaller,
lower-confidence check than the full re-read given to the Contradictory
tier.

## [B] Representative examples

**Genuinely Contradictory** (5, human-verified):
- `Screenshot 2026-05-31 224650.png` (predicted `printed_document`):
  *"The image displays a collection of fried food items, which is a
  physical object rather than a document type."* - confirmed real error
  (human ground truth: `casual_photo`).
- 4x cemetery-scene images (predicted `portrait_photo`), e.g.
  `Screenshot 2026-06-19 232858.png`: *"The image is a single photograph
  primarily depicting a cemetery scene with crosses and trees."* -
  `portrait_photo` requires a person as primary subject; none of these
  four mention one. One confirmed by human ground truth
  (`cemetery_photo`).

**Genuinely Ambiguous** (self-acknowledged hedging, 11):
- *"The image is a scan of a document containing prose and handwriting,
  fitting the description of a printed or handwritten document."*
  (predicted `printed_document`) - the reasoning itself proposes two
  categories before settling on one.
- *"...resembling a catalog entry rather than a repeating data table"*
  / *"...typical of an archival finding aid or catalog entry"*
  (predicted `printed_document`, 3 cases) - these consistently describe
  archival index/finding-aid content that doesn't cleanly fit ANY
  current bucket; `printed_document` is the least-bad default. This
  looks like a genuine taxonomy gap, not a reasoning failure - flagged
  here, not solved (out of this audit's scope).

**Mechanical false positives, corrected** (5): mostly "without a
repeating tabular structure or user interface elements" - a negated
comparison whose scope extended past my fixed detection window; on
direct reading these are correctly-reasoned `printed_document` calls.

## Special Investigation: off-category concept mentions

The same marker system doubles as the requested keyword search. One
additional, real, systematic pattern surfaced beyond the Contradictory
tier: **28/1154 `printed_document` classifications (2.4%) explicitly
mention "handwritten" in their own reasoning.** On direct reading this
group is a genuine mix, not uniformly an error - a typed letter with a
handwritten signature or annotation is legitimately still
`printed_document`; a few others describe content as primarily
handwritten while still landing in `printed_document`, which is a real
boundary miss. Not further split by hand at this pass; flagged as the
next place to look if this needs tightening.

## [C] Correlations with disagreement

**By bucket** (Contradictory, corrected count of 5): 1 in
`printed_document` (of 1154), 4 in `portrait_photo` (of 80, 5.0% -
notably concentrated relative to its size). `mixed_text_image` initially
showed 11/29 (37.9%) contradictory before the fix - confirmed on direct
reading to be **zero** genuine contradictions; that entire signal was
the mechanical pass mismeasuring `mixed_text_image`'s own correct,
expected photo+text component language.

**Confidence**: contradictory-tier cases average confidence 0.951 vs.
0.974 corpus-wide - slightly lower, but not a clean, reliable
discriminator (0.95-0.98 is a narrow band; both groups sit inside it).

**Tower agreement - the strongest correlation found**: tower-consensus
disagrees with Gemma in **85.7% (18/21) of mechanically-flagged
Contradictory cases**, vs. a 37.5% baseline disagreement rate across all
1750 images. This holds up as real, independent corroboration - the
vision towers, using completely separate evidence (embeddings, not
text), flag likely-wrong classifications at more than double the
baseline rate specifically where Gemma's own reasoning looks
inconsistent with its answer.

**Human ground truth**: 2/21 contradictory cases have a human label,
both confirming genuine errors (the food and cemetery-scene cases).
Too small a sample to compute a real precision/recall number, but zero
counter-examples.

## Reason-stability across the 93 checkpoint-vs-fresh classification changes

Read every one of the 93 pairs directly (not sampled). The question the
task posed: does the reasoning change WITH the category (internally
coherent per-response, even if the underlying judgment flip-flops
across runs), or does the SAME reasoning text get attached to a
different category (a much more troubling failure mode)?

**Result: 0/93 pairs show identical reasoning text.** Every single
changed classification came with genuinely different reasoning content
that tracks its own predicted category - e.g. `S3HY-65JC-4X.jpg`:
checkpoint said *"dense, handwritten records **without**... repeating
tabular rows"* (predicted `handwritten_ledger`); fresh said *"numerous
**repeated rows** with fixed columns"* (predicted `dense_tabular_rows`)
- a direct reversal of the same structural judgment (repeating vs. not)
about the identical pixels, but each response is internally coherent
with its own conclusion. This is the "at least internally coherent"
failure mode the task distinguished, confirmed as the only one observed
- never the "exact same text, different category" mode.

**The drift is concentrated, not uniform.** Undirected bucket-pair swap
counts across the 93:

| Pair | Count | % of 93 |
|---|---|---|
| `dense_tabular_rows` <-> `printed_document` | 29 | 31.2% |
| `dense_tabular_rows` <-> `website_screenshot` | 12 | 12.9% |
| `handwritten_ledger` <-> `printed_document` | 10 | 10.8% |
| `map_land_record` <-> `printed_document` | 8 | 8.6% |
| `printed_document` <-> `website_screenshot` | 8 | 8.6% |
| `mixed_text_image` <-> `printed_document` | 7 | 7.5% |
| `mixed_text_image` <-> `portrait_photo` | 7 | 7.5% |

Nearly a third of all drift is the single `dense_tabular_rows` <->
`printed_document` boundary - exactly the distinction
`config/taxonomy.yaml`'s own `classifier_guidance` text calls out as
needing careful judgment ("does this page repeat the same columns for
many different people/rows, or does it state a handful of facts about
one person/entity once?"). This concentration is itself informative:
the 5.3% overall drift rate isn't spread evenly across all category
pairs as generic noise - it's disproportionately one specific, already-
acknowledged-as-hard boundary.

**A small number of pairs show a clear regression, not neutral noise**:
`R21W-044.jpg` and `R22W-060.jpg` - checkpoint correctly identified
*"a printed map showing land divisions and geographic features"* /
*"a plan of township and range"* (`map_land_record`); fresh moved both
to `printed_document`, one with reasoning that explicitly says *"not...
a map"* while describing what checkpoint had already correctly
identified as one. Not the majority pattern, but a real example that
"noise" isn't always neutral - it can occasionally discard a clearly
better-supported answer.

## [D] Is Gemma's reasoning a reliable explanatory signal, or only diagnostic text?

**Diagnostic, not fully reliable as an explanation** - with real,
usable value once corrected against a few known false-positive patterns
this audit found:

- At face value, reasoning is genuinely informative: 86.7% of the
  corpus's classifications have reasoning that directly and correctly
  supports the predicted category, and true contradictions are rare
  (~0.3%, corrected).
- **Contradictory reasoning is a real, corroborated signal, not noise
  to ignore**: it's strongly correlated with independent tower
  disagreement (85.7% vs. 37.5% baseline) and, in the small sample with
  human labels, was right both times about flagging a genuine error.
  Worth building into a monitoring signal.
- **A naive keyword-based reading of "does the reason mention another
  category" is NOT reliable on its own** - this audit's own mechanical
  first pass overstated the true contradiction rate by roughly 4-7x
  (105 -> 35 -> 21 -> 5 true positives, across three successive
  corrections) before direct reading corrected it. Any future automated
  version of this check needs negation-awareness and category-specific
  exceptions (like `mixed_text_image`'s legitimate photo+text language)
  built in from the start, not bolted on after the fact.
- **The reasoning does NOT reliably indicate when Gemma's underlying
  judgment is stable.** The 93-pair analysis shows reasoning is always
  internally coherent with whatever category was picked THAT run - it
  gives no signal, on its own, that the same image was called something
  else moments/runs earlier. Reasoning consistency and classification
  stability are different properties; this audit measured both and
  found the first is generally good while the second (established
  earlier this session) is not.
