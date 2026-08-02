# Benchmark 2.4 — Selective E4B Escalation Analysis

**Status**: complete. Purely observational - no models were run, no
new inference happened. Every number here is computed directly from
already-frozen data: `data/outputs/benchmark2_3_multi_tower/
all_results.json` (corpus, tower votes, consensus) and `data/outputs/
benchmark2_3_e2b_vs_e4b/fresh_results.json` (E2B and E4B predictions,
per-image timing). Script: `benchmark/benchmark2_4_selective_e4b_
escalation.py`. Nothing written to production manifests; benchmark
corpus, tower embeddings, tower predictions, and consensus
calculations are unchanged.

## 1. Method

Read the two frozen result sets directly, join on image name, and
compute escalation membership + outcomes without touching any model,
image file, or bucket CSV. Corpus size (146), tower votes, and
consensus categorization are exactly as computed in Benchmark 2.3 - not
recomputed here.

## 2. Selection criteria

Escalate an image when **both**:
1. Multi-tower consensus is not unanimous (`consensus_category !=
   "unanimous"`).
2. E2B disagrees with the tower-majority prediction
   (`gemma_agrees_with_tower_consensus == False`).

**Real finding, checked rather than assumed**: these two conditions are
**mathematically equivalent in practice** given how `consensus_category`
was computed in Benchmark 2.3 - "unanimous" requires all 9 voters (8
towers + Gemma) to agree, so if Gemma (E2B) disagrees with the tower
plurality, unanimity across all 9 is already impossible. Verified
directly: condition 2 alone selects exactly the same 56 images as both
conditions together. This isn't a redundant instruction - it's a
genuine property of the escalation rule worth knowing before reusing
this pattern elsewhere (e.g. if a "majority-only, excluding split/
complete_disagreement" version of condition 1 were wanted instead of
"just not unanimous," the two conditions would no longer collapse into
one).

## 3. Results

**Escalation set: 56/146 images (38.4%)**.

Within the escalated set:
```
E4B moves the image to match tower consensus (correction captured): 14
E4B still disagrees with tower consensus after escalation:          42
Regressions (E2B matched consensus, escalated anyway):                0
```

The 0 regressions is a **structural property of the gate**, not a
measured coincidence: every escalated image already has E2B
disagreeing with tower consensus, so there is no image in this set
where escalation could turn an already-correct-by-proxy answer into a
wrong one. Verified directly (not assumed) by checking no escalation
record has `e2b_prediction == tower_consensus_bucket`.

**A second, non-obvious safety property, found while checking whether
escalation misses any of E4B's real behavior**: of the 90
non-escalated images (E2B already matches tower consensus), running
E4B on them would have introduced a **new** disagreement with E2B on
**10 of the 90** - cases the always-E4B run actually hit ("E4B leaves
tower consensus" in `docs/BENCHMARK2_3_E2B_VS_E4B.md`). Selective
escalation never runs E4B on these 90 images at all, so it structurally
avoids ever triggering those 10 potential regressions, on top of the 0
possible within the escalated set itself.

**Tower consensus strength within the escalated set** spans the full
range, including 9 images where all 8 towers unanimously agree with
each other yet E2B still picks something different:
```
2/8: 1   3/8: 7   4/8: 11   5/8: 7   6/8: 9   7/8: 12   8/8: 9
```
The 9 images at 8/8 tower agreement are arguably the strongest
evidence-against-E2B subset in the whole escalated set - full tower
consensus disagreeing with E2B is a stronger signal than a narrow 3/8
plurality would be.

**Known taxonomy-boundary canary** (`oocihm.lac_reel_c10264.767.jpg`,
the dual-natured handwritten/tabular image from `docs/
REFERENCE_CLASSIFIER_QUALIFICATION.md`): confirmed present in the
escalation set, as expected.

## 4. Runtime analysis

Using the measured per-image timing from Benchmark 2.3's E2B-vs-E4B
comparison (E2B: 6.493s/image, E4B: 31.457s/image), modeling E4B as a
genuine second-stage reviewer (E2B always runs first; E4B is an
*additional* pass only on the escalated subset, not a replacement):

```
Always E2B:            948.0s  (15.8 min)
Always E4B:            4592.7s (76.5 min)
Selective escalation:  2709.6s (45.2 min)

Reduction vs always-E4B:  41.0%
Overhead vs always-E2B:  +185.8%
```

Selective escalation cuts total E4B-related runtime by 41% compared to
running E4B on the entire corpus, while capturing exactly the same 14
corrections (the escalation set is the complete set of images where
E4B could possibly move anything toward consensus - see the
equivalence finding in Section 2). Nothing is left on the table by
escalating a subset rather than everything: **100% of E4B's real
benefit is captured at 59% of its total classification cost.**

It is still substantially slower than E2B alone (+186%), since the
baseline E2B pass is unavoidable and 38.4% of the corpus also pays
E4B's much higher per-image cost (4.8x E2B's).

## 5. Benefit analysis

```
                    Always E2B   Always E4B   Selective escalation
Corrections gained       -            14                 14
Regressions introduced   -            10                  0
Net improvement          -             4                 14
Runtime                948.0s      4592.7s             2709.6s
```

Selective escalation strictly dominates always-E4B on this evidence:
**same corrections (14), zero regressions instead of 10, at 59% of
the runtime.** The "net +4" figure reported for always-E4B in
Benchmark 2.3 understates E4B's real corrective power - it was net
of 10 self-inflicted regressions on cases E2B already had right. Gating
E4B behind tower disagreement removes those 10 regressions entirely by
construction, not through any additional correctness logic.

The remaining open question is accuracy in absolute terms, not just
agreement with the tower-consensus proxy: 42 of the 56 escalated images
still don't resolve to tower consensus even after paying for E4B, and
without independent ground truth for most of these, "regression-free
relative to the proxy" is not the same claim as "correct." The
tower-consensus proxy is itself imperfect (see the known
taxonomy-boundary canary, where consensus itself reflects a genuine,
still-open taxonomy question rather than settled ground truth).

**Optional trigger comparison** (observation only, not adopted): E2B's
own confidence score is a poor standalone escalation trigger at this
corpus's confidence distribution - thresholds of 0.90 and 0.95 select
zero images (E2B's confidence is almost always ≥0.95 here, including on
images where it's wrong), and even a very permissive 0.97 threshold
selects 32 images with only 75% overlap (24/32) with the real
56-image tower-disagreement escalation set - it would miss most of the
genuine hard cases while flagging 8 unnecessary ones. The tower-
consensus-based trigger is meaningfully more precise than a
confidence-based one for this corpus.

## 6. Recommendation

**Yes - the evidence supports using E4B as a targeted second-stage
reviewer rather than a universal replacement for E2B.**

Selective escalation (tower-disagreement gated) captures 100% of E4B's
measured corrections (14/146), introduces 0 regressions instead of the
10 always-E4B introduces, and does so at 59% of always-E4B's total
classification runtime. This is a strict improvement over "always E4B"
on every dimension measured here - not a trade-off between accuracy and
cost, but a better position on both axes simultaneously, because the
escalation gate happens to filter out exactly the cases where E4B was
actively hurting (the 10 regressions) while preserving every case where
it was helping.

This does not mean selective escalation is free or that E4B is
unambiguously "better" - it remains ~186% slower than E2B alone, still
leaves 42/56 hard cases unresolved by the tower-consensus proxy, and
that proxy itself is not ground truth. But relative to the two
alternatives actually measured (always E2B, always E4B), gated
escalation is the only one of the three that avoids introducing new
regressions while still capturing real corrections.

**No pipeline changes made.** This experiment is analysis only, per
scope - implementing selective escalation in production would be a
separate, later decision.
