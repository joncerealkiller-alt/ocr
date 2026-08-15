# Proposal: Hint-Free Two-Stage Extraction (independent reads + downstream comparison)

**Status: VALIDATED LIVE, NOT YET APPLIED TO PRODUCTION** — drafted and then
tested same-day, 2026-08-15 (Fable session). The production pipeline is
unchanged; `config/prompts/structuring_stage2_independent.txt` remains inert
until explicitly selected via the existing `--structuring-prompt-file` flag.
See "Live validation" at the bottom for the head-to-head result.

## The evidence this responds to

All measured live on 2026-08-15, persisted in the benchmark store:

1. **`census_hint_anchoring_v1`** (real `build_structuring_prompt()` output, correct
   vs wrong-but-plausible hints, same 1931 GT cells as the no-hint control runs):
   - Hard name fields: wrong-plausible hints **parroted verbatim 7/8** across
     E2B + Qwen. Correct hints passed 8/8 (including cells no model reads
     unaided). Stage 2 is a hint *pass-through* on hard fields, not a verifier —
     hint quality in, answer quality out, in both directions.
   - The decisive case: Qwen is the only model that reads "Hyzie Bertha"
     correctly unhinted — given the hint "Huzie Bertha" it echoed
     `Huzie Bertha|confirmed`. **The hint destroyed the only correct read any
     model produces, and stamped it confirmed.**
   - Easy birthplace fields: wrong hints rejected 4/4 — the ink wins where the
     model can independently contradict the hint. (Matches the pre-existing
     anchoring finding; now quantified.)
   - Confidence tags are uninformative under hints: parroted wrong answers
     came back `confirmed`.
2. **Pre-refactor two-stage scored run** (`benchmark_results.csv`, smolvlm2 →
   qwen3vl4b): the hint design's one real measured benefit was fabrication
   filtering (stage1 hallucinations 9 → 1 after stage2). That benefit comes
   from the model *rejecting implausible* hints — and is reproducible by plain
   downstream string comparison, without giving the hint a chance to overwrite
   perception on plausible-but-wrong cases (6 cells got worse than their hint;
   "output the same name as stage 1, but missed the Ditto marks").
3. **Three-model no-hint runs** (`census_fieldcrops_v1`): zero hallucinations in
   48 independent per-field outputs; complementary failure modes (Qwen best on
   names, Gemma E2B best on tiny numeric cells, honest "?" abstentions).

## The design

Replace "Stage 1 hint feeds Stage 2's prompt" with "two independent reads,
compared downstream":

```
today:   stage1 (fast OCR) --hint--> stage2 (sees image + hint) --> value
proposed: stage1 (fast OCR) ---------------------------\
          stage2 (sees image ONLY) --------------------+--> compare
                                                            |
                                        agree -> accept stage2 value, high trust
                                        disagree/abstain -> review queue
                                          (with the +/-2-row context crop,
                                           which Jon demonstrated resolves
                                           ambiguity for a human reviewer)
```

## What it takes — surprisingly little

### 1. Template (DONE, inert): `config/prompts/structuring_stage2_independent.txt`

Same structured `Field: value|confidence` output contract, same
`{field_hint}`/`{example_lines}` machinery, **no `{raw_ocr_text}` block and no
check-or-correct framing**. Because `str.format()` ignores unused keyword
arguments, `build_structuring_prompt()` needs **zero changes** to render it —
and `run_two_stage_extraction.py --structuring-prompt-template
structuring_stage2_independent` selects it today. The hint-free *experiment*
is therefore already runnable with no code edits at all.

### 2. Agreement computation (the only new code)

In `core/row_extraction.py`, the stage-2 field loop (~line 1136–1181) already
has both strings in hand: `field_reading` (stage 1) and `parsed[column_name]`
(stage 2). Sketch:

```python
# after: parsed = parse_row_output(raw_output, [column_name])
stage2_value = parsed.get(column_name, "")
agreement = _normalize(field_reading) == _normalize(stage2_value)
field_agreement[column_name] = agreement
```

- `_normalize`: reuse/adapt `benchmark/scorers._normalize` (lowercase, collapse
  whitespace, strip trailing punctuation). Consider stripping the `|confidence`
  suffix and `?` handling: either read being `?` = automatic disagreement
  (review), matching the abstention-is-a-feature ethos.
- `RowExtractionResult` gains an additive `field_agreement: dict[str, bool]`
  field (default empty — old callers unaffected), persisted the same way
  `stage1_raw_output` already is.
- Routing: disagreements do NOT overwrite the value — stage 2's independent
  read stays the recorded value, with the field flagged for the review queue.
  This is deliberately the same quarantine-not-guess shape as the rest of the
  pipeline.

### 3. Retroactive analysis is free

`RowExtractionResult.stage1_raw_output` is already persisted per row, so the
agreement metric can be computed **post-hoc on existing hint-era outputs** to
estimate review-queue volume before committing to anything.

### 4. Optional follow-ups (not part of the minimal change)

- Model split per the 2026-08-15 evidence: Qwen (`qwen25_vl_7b_awq`) for Name
  columns, E2B for numeric/short fields — both stages under vLLM. Complementary
  failure modes make disagreement a stronger signal than same-model double-reads.
- Review UI: show the ±2-row same-column context crop (generation code exists
  in the census_name_context suite tooling) instead of the bare cell — Jon
  demonstrated live that writer-calibration from neighbors resolves cells the
  models can't (the Hyzie "u-has-no-tail" comparison against Burton).
- If the widecrop red-box prompt is used with non-Gemma models anywhere, add a
  "reply with only the transcription" line (12B and Qwen both wrapped answers
  without it; Gemma didn't).

## What this deliberately does NOT do

- No change to Stage 1 (it still runs; its value shifts from "hint" to
  "independent second reading for the comparator").
- No LLM-as-judge, no third model call — comparison is a string operation.
- No automatic acceptance of agreements on `?`/abstentions.
- No removal of the hint template (`structuring_stage2_default.txt` stays for
  history/comparison; suite `census_hint_anchoring_v1` documents why it lost).

## Predicted costs (to verify, not assume)

- Review-queue volume rises where the two models genuinely disagree — on the
  2026-08-15 data that's concentrated exactly where it should be (hard
  handwriting like Knott/Hyzie, i.e. ink-limited cells a human should see
  anyway). Quantify with the post-hoc analysis in §3 before rollout.
- Hint-parroting currently *hides* disagreement by collapsing stage 2 onto
  stage 1 — expect measured agreement rates to drop when the hint is removed.
  That is the metric becoming honest, not the pipeline getting worse.


## Live validation (2026-08-15, same day)

Both conditions were run through the REAL pipeline
(`scripts/run_two_stage_extraction.py`, smolvlm2_2b Stage 1 with its
deliberate empty prompt -> qwen3vl4b Stage 2), rows 1-6 of the
1931_174-e011707164 page, identical crops/models/settings - the ONLY
difference was the structuring template. Scored against the merged ground
truth (the v3 Ground_truth transcript + ground_truth_log.jsonl; the two
human-GT conflicts, Huzie/Hyzie and Wilson/Nelson, were accepted either way).
Artifacts: `genealogy_workspace/research/experiments/hint_free_two_stage_20260815/`.

| | Hint (production template) | Hint-free (this proposal) |
|---|---:|---:|
| Stage 2 accuracy | 18/30 (60%) | 21/30 (70%) |
| Stage1<->2 containment-agreement | 11/30 | 9/30 |
| Agreements that were WRONG | 2 | 0 |
| Agreement precision as a trust signal | 82% | 100% |

Findings:

1. **Hint-free was +10 points more accurate on identical inputs.** Concrete
   contamination in the hinted leg: Age "1" (anchored on Stage 1's garbled
   "5,1"; hint-free correctly read "14"), Birthplace "Maniototo" (Manitoba
   blended with hint noise; hint-free: clean "Manitoba"), Relationship "F"
   (from hint "Langlahta"; hint-free: correct "Daughter").
2. **The auto-accept-on-agreement requirement held: every hint-free agreement
   was correct (9/9).** Hinted agreement is a contaminated signal (2/11 wrong
   - the hint echoing back). This is the property the routing design depends
   on, confirmed rather than assumed.
3. **Comparator design lesson**: exact-string agreement is useless against
   real Stage-1 output (0/30 both legs - Stage 1 emits noisy strings like
   "3. Manitoba", "The answer is 18."). The agreement metric must be
   normalized CONTAINMENT (one read contains the other's normalized value),
   which is what the numbers above use. §2's sketch should be updated
   accordingly when implemented.
4. Caveats: n=30 fields, one page, one (legacy) model pair. The proposed
   Qwen-7B/E2B split should raise both accuracy and agreement; quantify on
   more pages before rollout. Review-queue volume at this quality level:
   21/30 routed to review (12 of which were actually correct) - the queue is
   real, and shrinks as the models improve.
