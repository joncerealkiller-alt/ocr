# Auto-accept detector study — 2026-09-02

**What this is:** the first systematic characterization of the two-stage
pipeline's auto-accept detector (the `field_agreement` gate), run over
549 GT-scored cells on 3 fully-labeled reference pages x both engine
pairs, followed by three deterministic hardening layers - each driven
by a measured false accept, each preserved as a separate commit so the
before/after is auditable. Prior to this study the detector's safety
rested on one n=30 validation (15/0, 2026-08-15); this study shows that
result was partly sampling luck (true FP rate was ~1-2%).

**Decision frame (Jon):** the corpus is small relative to review
capacity, so speed gains fund human review of everything routed there.
The quantity being minimized is FALSE AUTO-ACCEPTS (errors that bypass
the human), not review volume and not raw accuracy. Auto-accept volume
is deliberately spent to buy FP reduction.

## Fixture set

3 pages x 50 rows x 5 columns, `ground_truth_log.jsonl` (latest label
per cell, status=readable only): `1921_022-E002880409` (82 scorable
cells - heavily blank/illegible page), `31228_4363955-00089` (232),
`1931_174-e011707164` (235). Runs: `genealogy_workspace/research/
experiments/gguf_two_stage_20260902/full_{gguf,vllm}_pair/`. GGUF pair
= minicpm_v_gguf + gemma_12b_qat_gguf (llamacpp); vLLM pair =
minicpm_v_gptq + gemma_12b_w4a16. Same prompts (terse no-think stage 1,
hint-free independent stage 2), same sidecars.

## Headline evolution (FP = false auto-accepts, same 549 cells)

| Detector | GGUF pair FP (precision) | vLLM pair FP (precision) |
|---|---|---|
| Original (as shipped 2026-08-15) | 13 (0.935) | 7 (0.966) |
| + layer-2 hardening (16c6636) | 6-7 (0.962-0.967) | 3-4 (0.977-0.983) |
| + layer-3 page context (bb19c9e) | 6 (0.967) | 3 (0.983) |
| after GT-artifact adjudication* | **5 (0.973)** | **2 (0.989)** |

*The `Geneviève` FP in both pairs is a GT-label artifact (stray leading
colon, missing accent - the model read is very likely correct). GT
cells needing human adjudication: `1931_174 r7 Name (": Genevieve")`,
`1921_022 r20 Birthplace ("Eng 2")`, `1921_022 r37 Name ("F?ingthy")`.

Cost: ~18 auto-accepts pushed to review across the fixture set
(deliberate, per the decision frame). Wall-clock context: GGUF pair ran
the 3 pages in 9.7 min of model time vs the vLLM pair's 32.2 min plus
~6 engine relaunches.

## The four FP mechanisms found (and which layer answers each)

1. **Comparator defects** (layer 2, `fields_agree()` hardening):
   spaced-digit ambiguity ("5 1 4" auto-accepted "4", GT 14), CoT
   contamination ("<think>" treated as a reading), substring mangle
   ("Umanitoba" matched inside "Manitoba"). All engine-independent.
2. **Malformed values / column spill** (layer 2, `column_schema_valid()`
   + `config/column_schemas.yaml`): "Eng &" in Birthplace, "m2" in Sex,
   "brother 3" in Relationship. Schema is deliberately wide - it
   catches garbage, never second-guesses plausible values.
3. **Page-convention anomalies** (layer 3, `apply_page_context_vetoes()`):
   well-formed values inconsistent with the page's own established
   granularity - "Hamilton" (city) and "L.A." on province/country-level
   pages. Derived from the page's own trusted cells, GT never
   consulted; strong-evidence thresholds (>=8 trusted, >=80%
   in-vocabulary) make low-evidence/mixed pages abstain entirely -
   measured necessity, since 2 of the 3 fixture pages yield only 1
   trusted birthplace cell and would otherwise be false-rejection
   factories. Veto routes to review; NEVER substitutes a value.
4. **Correlated plausible misreads** (irreducible by deterministic
   layers): `Brother`/mother, `Hadie`/Katie, off-by-one-digit ages
   (12/17, 42/40, 62/52), `Timothy`/uncertain-GT. Well-formed,
   in-convention, agreed by both readers. Only error diversity between
   readers or a human catches these. The 62/52 cell fooled BOTH engine
   pairs - some cells defeat any reader pair.

## Error-diversity finding (the ensemble question)

6 of the original FP cells were shared by both pairs (same wrong value)
- reader-pair-independent. The same-engine GGUF pair added ~7 unique
FPs vs the mixed pair's 1, BUT most of that gap was mechanisms 1-3;
after hardening the residual engine-attributable diversity cost is
~3 cells in 549 (~0.6%). Conclusion: two individually-good readers
don't automatically make a good agreement detector - the quantity that
matters is accuracy + error INDEPENDENCE - but the measured
independence penalty of the same-engine pair is small once the
detector itself is sound. Deliberate decorrelation engineering
(different per-stage preprocessing/quants) remains NOT justified by
this data; revisit only if same-wrong agreements grow on new pages.

## The abbreviation no-op (46331f6) — infrastructure, NOT benchmark optimization

The two-letter/dotted province-abbreviation widening (`BC`, `N.S.`,
`P.E.I.`, `Man.` forms in the Birthplace pattern + vocabulary)
produced **zero delta on this fixture set - deliberately kept anyway.**
Recorded here so future readers don't see a no-delta commit and wonder
why it exists:

- It fixes a FALSE-REJECTION class, not a false-acceptance class: the
  old 3-char syntax floor schema-rejected legitimate enumerator forms
  before the vocabulary could accept them. This fixture set simply
  contained no correct 2-char/dotted reads to rescue (its abbreviated
  GT values - "Man", "Ont", "Eng" - are 3-char forms that already
  passed).
- The layers feed each other: schema breadth determines how many cells
  become TRUSTED, and trusted mass is the activation condition for the
  page-context veto. Abbreviation-heavy pages (1921_022's GT is
  man/ont/eng throughout) can only accumulate enough trusted evidence
  for layer 3 to protect them if their abbreviation style passes
  layer 2. Without this change, better reader accuracy on those pages
  would never translate into context-veto coverage.

## Current detector stack (as of 46331f6)

```
stage1/stage2 independent-reader agreement   (fields_agree - hardened)
  -> per-column value schema                 (column_schema_valid, config/column_schemas.yaml)
  -> page-context convention consistency     (apply_page_context_vetoes - strong-evidence-only veto)
  -> AUTO-ACCEPT; any failure -> human review
```

All layers deterministic, config-driven, and covered by
`tests/test_field_agreement.py` (49 cases, every hardening case a
measured shape from this study). Commits: 16c6636 (layer 2), bb19c9e
(layer 3), 46331f6 (abbreviations). Scoring methodology caveat: the
layer-2/3 numbers are offline recomputes over stored run JSONs (the
stage-1 per-column reparse approximates the pipeline's own
field_reading); a live rerun would firm them up but the FP-level
conclusions were verified cell-by-cell.

## Open items

- Human adjudication of the 3 GT cells listed above.
- The irreducible correlated-misread class (~0.4-1% of cells) is the
  standing FP floor; page-count growth will show whether it stays flat.
- Per-form-year schema overrides: build only when a year actually
  needs different shapes (multi-schedule lesson).
- census_pairing GT-vs-detector note: suite-level scoring treats
  GT-alternation ("Hyzie|Huzie") loosely - keep suite scoring and
  detector scoring distinct.
