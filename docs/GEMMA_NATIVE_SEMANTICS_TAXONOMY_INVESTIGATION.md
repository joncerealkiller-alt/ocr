# Gemma Native-Semantics vs. Taxonomy Investigation (2026-08-11)

## Status: measurement only. No production code, config, or prompt changes made.

## Origin

Jon ran a comparison of our production classifier's Gemma model against a
separate on-device (phone) Gemma instance, asking it to define each taxonomy
label in its own words with no prompt engineering. An external report (via
ChatGPT) analyzing those phone outputs flagged 4 of our 11 taxonomy labels as
possible "semantic mismatch" candidates: `handwritten_ledger`,
`photo_collage`, `casual_photo`, `cemetery_photo`.

This prompted a multi-stage experiment, run against our actual production
Gemma model/settings (not the phone model), to find out whether Gemma's own
native understanding of our labels - and its own native description of real
images from our corpus - lines up with how we've defined the taxonomy.

Goal: figure out whether some of our real classification errors are caused
by asking Gemma to route into a taxonomy that doesn't match concepts it
already has, rather than by weak prompting or a weak model.

## Stage 1 - Semantic self-report, production system prompt in place

Script: `scripts/gemma_semantic_definition_experiment.py`

Asked Gemma, once per taxonomy label (11 labels), with the real production
system prompt ("You are a strict, non-interpretive archival routing
classifier.") and real production model settings (`do_sample: false`, i.e.
greedy decoding), to define what each label means to it - no images, no
examples, no hints at our intended definition. Run 5 independent times per
label to check for drift.

**Finding: zero drift.** All 5 runs per label were byte-identical, for all
11 labels (55/55 generations). This is expected under greedy decoding but
was confirmed directly rather than assumed, per Jon's explicit design intent
("we needed to prove there was no drift at all and the model output was a
true representation of its semantic classification").

Run: `20260810T210951208121_5837a4bd`, results at
`genealogy_workspace/runs/20260810T210951208121_5837a4bd/reports/results.jsonl`

## Stage 2 - Same test, system prompt removed (attempt 1: corrupted)

Rerun with `config.extra["system_prompt"]` cleared in memory (production
`config/models/gemma.yaml` untouched), everything else identical, 1 run per
label (settings are deterministic, so 5 runs would be redundant here).

**Finding (negative, but real): catastrophic degeneration.** 6/11 labels
showed heavy `?`-character flooding (8-19% of the response); the other 5/11
truncated after one sentence into an empty `---` loop. Root cause: without
the system prompt's short/plain framing, Gemma's natural instinct for
open-ended prose is heavy markdown (bold headers, numbered sections), and
production's `restrict_output_charset` charset mask doesn't include the
punctuation that formatting needs - every masked attempt at formatting
collapsed generation.

## Stage 2b - System prompt removed, charset mask also disabled

Same as Stage 2 but with `config.restrict_output_charset` also overridden
`False` in memory. Confirmed clean: 0-2 stray `?` characters across all 11
labels, 3700-4900 chars each, no truncation.

Run: `20260811T000423839974_77d138c5`, results at
`genealogy_workspace/runs/20260811T000423839974_77d138c5/reports/results.jsonl`

**Cross-check finding:** for the two labels checked directly against Stage 1
(`handwritten_ledger`, `cemetery_photo`), the underlying semantic content was
the same with or without the system prompt - only formatting/tone changed.
This means the semantic mismatch (where found) is baked into the label
token/definition itself, not induced by the archival-router persona.

## Stage 3 - Minimal-prompt image classification ablation (the main finding)

Script: `scripts/gemma_minimal_prompt_classification_ablation.py`

Per Jon's direction: rather than asking Gemma to *define* labels in the
abstract, ask it to classify **real images from our corpus** with
essentially no schema - prompt is literally `"Classify this image."`, no
system prompt, no taxonomy provided, `restrict_output_charset` off (same
proven fix as Stage 2b), `max_new_tokens` raised to 512. Calls
`GemmaLoader._run_generate()` directly, bypassing `classify()` and its
strict `category:/confidence:/reason:` schema parser entirely - the point is
to see what Gemma says with zero schema imposed.

### Run A - 20 known confident-wrong images (`data/misclassifications.csv`)

Run: `20260811T005250134203_7890bcaf`, results at
`genealogy_workspace/runs/20260811T005250134203_7890bcaf/reports/results.jsonl`

Included the real Kemper memorial-plaque images that production had called
`printed_document` at 0.98 confidence. All 3 plaque images in this set came
back from the minimal prompt using language like "photograph of a memorial
plaque or headstone," "commemorative plaque or memorial," and one response
volunteered its own unprompted classification: **"Classification:
Memorial/Plaque/Historical Artifact."** None used "document" language.

Everything else in the 20-image set (text-block screenshots, tabular
records) held up fine under the minimal prompt - descriptions matched the
existing production category, no broad taxonomy drift.

### Run B - full 487-image corpus (`data/outputs/kemper_ancestry_review.csv`)

Run: `20260811T010309141535_76c5fb5d`, results at
`genealogy_workspace/runs/20260811T010309141535_76c5fb5d/reports/results.jsonl`

Searched all 487 raw responses for plaque/memorial/headstone/monument
language in the first ~400 characters. 13 hits total; 3 were false positives
(website screenshots of genealogy/cemetery-memorial *websites*, correctly
classified as `website_screenshot` - "memorial" was just part of the page
title). The remaining **10 are real physical plaque/memorial-marker images**,
currently misrouted:

- 9 currently classified `printed_document`
- 1 currently classified `casual_photo`

That's 10/487 (~2.1%) of the corpus. All 10 converge on the same natural
vocabulary ("plaque," "memorial," "commemorative," "headstone") regardless
of which wrong production bucket they landed in - a stable, non-trivial
cluster, not noise.

## Stage 1 addendum - persona-triggered early termination on 3 labels (found 2026-08-11, reviewing raw logs)

Not in the original write-up; found by rereading Stage 1's raw
`results.jsonl` directly rather than only the two labels spot-checked
against Stage 2b at the time.

**Finding**: in Stage 1 (system prompt present), 3 of the 11 labels -
`website_screenshot`, `dense_tabular_rows`, `mixed_text_image` -
generated dramatically short responses (224-283 characters, stopping
right after "1." with no list content following) versus every other
label's 3000-5000+ characters. `run_metadata.json` confirms
`max_new_tokens: 2048` for this run, so this is NOT a token-budget
truncation - Gemma itself emitted an end-of-generation signal almost
immediately, only for these 3 labels.

**First verified this wasn't drift/noise**: confirmed directly (not
assumed) that all 5 runs/label are byte-identical across all 11 labels,
55/55 - matches the doc's existing Stage 1 drift-check claim exactly,
re-verified independently.

**Then checked whether it was persona-dependent**: Stage 2b (system
prompt removed) response lengths for the same 3 labels: `website_
screenshot` 4154 chars, `dense_tabular_rows` 4606 chars, `mixed_text_
image` 3773 chars - fully in line with every other label once the
persona is removed. **The early termination is specific to the
persona-present condition, not a general instability in these 3
labels.**

**Interpretation**: the "strict, non-interpretive archival routing
classifier" persona appears to prime Gemma toward a terse, low-
elaboration answer specifically for the labels that most obviously fit
that self-concept already - `website_screenshot`/`dense_tabular_rows`/
`mixed_text_image` read as the most archetypally "document classifier"
categories in this taxonomy, and Gemma seems to treat them as requiring
little further unpacking once already framed as an archival router.
Strip the persona and it elaborates equally on everything, matched or
mismatched. This is a distinct effect from the doc's existing Stage
1-vs-2b cross-check finding (persona changes tone/formatting, not
semantic content, checked on `handwritten_ledger`/`cemetery_photo`) -
this shows persona can ALSO change how much Gemma says, correlated with
how well the label matches its own self-framing, without necessarily
changing the semantic content once it does elaborate (not independently
re-verified here whether the eventual content differs for these 3
specific labels beyond the length difference itself - the 2 labels
originally cross-checked for content-parity were not among these 3).

No production code, config, or prompt changes made - measurement only,
same as the rest of this document.

## Stage 4 - Cross-generation replication on Gemma 3 (on-device), plus a candidate replacement label (2026-08-11)

Source: a separate session ran this stage directly against Jon's on-device
Gemma 3 (not our production transformers Gemma), specifically to test
whether the semantic mismatches found above are a Gemma-4-checkpoint
peculiarity or a general property of how these labels read in ordinary
English. Reported here verbatim-summarized so the finding isn't siloed in
that session.

**All 4 labels flagged by the original phone-Gemma ChatGPT report were
independently retested cold on Gemma 3, and all 4 mismatches replicated:**

| Label | Gemma's natural semantic attractor | Result |
|---|---|---|
| `cemetery_photo` | photograph/scene of a cemetery (gravesites, tombstones, monuments, solemn atmosphere) vs. our operational focus on individual grave-marker/headstone photos | Replicated |
| `casual_photo` | candid/informal/unstaged photographic *style* vs. our broader "ordinary archival or personal photo regardless of pose" | Replicated |
| `photo_collage` | correct core morphology (multiple photos arranged together) but with added connotations of deliberate/artistic/thematic intent we don't require | Partial mismatch replicated |
| `handwritten_ledger` | financial/accounting ledger (receipts, invoices, balances, bank statements) vs. our broader structured-handwritten-record intent | Strong mismatch replicated |

This is independent evidence, from a different Gemma checkpoint/runtime,
converging on the same 4 labels this document already flagged via the
text-definition method (Stage 1/2b) - materially weakens the theory that
these are peculiarities of one checkpoint. Models appear to interpret these
taxonomy labels via their ordinary pretrained English semantics; the
mismatch is in labels whose *intended* project meaning diverges from that.

### Candidate replacement for `handwritten_ledger`: `structured_handwritten_record`

During discussion, Gemma 3 itself proposed the umbrella concept "Structured
Handwritten Records," covering census sheets, enumeration tables, parish
registers, land registers, court rolls, civil registry rolls, inventories,
apprenticeship registers, military registers, and financial records -
generalizing across record *purpose* while keeping the common handwritten
structured *morphology*.

This candidate was then tested (cold, in a fresh context) rather than just
proposed:

- **Cold definition test**: asked from scratch what `structured_handwritten_
  record` means, with no prior framing. Definition centered on handwritten
  text + predefined/defined fields, gridlines, consistent layout, aligned
  data, abbreviations/codes - financial examples remained but shifted from
  *defining* the category to merely being examples within it.
- **Positive test - Canadian census**: a handwritten census page (rows of
  people, columns of demographic fields) was accepted as a clear "yes"
  without needing to reinterpret it as a financial ledger.
- **Positive test - passenger manifest**: accepted under the same broader
  concept, another non-financial generalization passing.
- **Positive test - real marriage-record image**: shown an actual historical
  structured handwritten image (not a hypothetical description). Gemma
  classified it as `structured_handwritten_record` (correct top-level
  morphology) and additionally guessed it was likely a marriage certificate/
  record. Jon confirmed the source image was in fact a marriage entry -
  meaning Gemma got both the routing-level morphology AND the deeper
  semantic subtype right simultaneously.

**One unwanted condition found**: Gemma's cold definition included that
handwriting should be "sufficiently legible to interpret." The project's
visual routing category should not require successful transcription - an
illegible/degraded structured handwritten record should still belong to the
category. This needs a negative-control test (see below) before being
treated as resolved either way.

### Unresolved negative controls (not yet run)

Before this label is a real candidate for adoption, these should still be
tested:

1. A handwritten personal letter - should NOT qualify merely because
   handwriting appears on ruled lines (tests free-form prose vs. structured
   fields).
2. A printed (not handwritten) census/table - should NOT qualify merely
   because rows/columns/structured fields are present (tests the
   handwritten requirement holds independently of structure).
3. A degraded/poorly legible handwritten structured form - SHOULD still
   qualify despite transcription difficulty (directly tests the unwanted
   "legible enough" condition found above).

Desired boundary, not yet verified: `handwritten + structured fields → YES`,
`handwritten + free-form prose → NO`, `printed + structured fields → NO`.

### Separate behavioral note (Gemma 3 on-device, methodology caution)

Instructing Gemma conversationally to "forget prior conversational state"
did **not** actually clear context - it caused near-empty/near-instant
subsequent generations instead, until the bare system prompt was changed
to an explicit "reply to the user" instruction. **For future cold-semantic
tests on this or any chat-wrapped runtime, use an actual new-session/
context reset, not a conversational instruction to forget** - a bare/no
system prompt and a minimal-instruction system prompt are not necessarily
behaviorally equivalent. The same repetition-loop failure mode seen
elsewhere in this doc's Gemma testing also reappeared during the
`handwritten_ledger` cold-definition run and was handled the same way
(manually stopped once the semantic answer was already evident, since
further looped tokens added no evidence).

### Status

`handwritten_ledger` -> semantic alignment: **FAIL** (now confirmed on two
separate Gemma checkpoints/runtimes).

`structured_handwritten_record` -> **PROVISIONAL/STRONG replacement
candidate**, pending the 3 negative controls above. **No taxonomy change
has been made** - this is still measurement, consistent with the rest of
this document and with Jon's standing instruction to hold off on taxonomy
changes.

## Stage 5 - Automated local cross-model battery (4 models, 48 generations, 2026-08-11)

Script: `scripts/cross_model_taxonomy_semantic_audit.py`

Extends Stage 4's cross-checkpoint idea into a controlled, automated,
one-shot battery run locally (not on-device/phone) against 4 different
text-only-capable loaders already in this repo, reusing `model_console`'s
`ChatBackendAdapter` so every model gets identical treatment (fresh
messages per call, no history carried, same in-memory-only override
discipline as every other stage here). GPU was verified idle
(`nvidia-smi`) before launch; `model_console.app`'s resident Gemma
instance was released cleanly (its own `WM_DELETE_WINDOW` handler, which
calls `loader.release()`) before this run, so the two never collided.

**Models** (all have an empirically-confirmed `text_only_supported: true`
in their `config/models/*.yaml` - never assumed from architecture):
`gemma` (production, `google/gemma-4-E2B-it`), `qwen3b` (`Qwen/Qwen2.5-VL-
3B-Instruct` - same checkpoint family Jon tested on-device in Stage 4,
direct comparison point), `qwen3vl4b` (`Qwen/Qwen3-VL-4B-Instruct` -
different Qwen generation), `internvl3_8b` (`OpenGVLab/InternVL3-8B-hf` -
different model family entirely).

**Settings**: each model's own tested `do_sample`/`temperature`/`top_p`/
`top_k`/`repetition_penalty`/`no_repeat_ngram_size` left untouched (all
greedy/deterministic already); only `system_prompt` (cleared) and
`restrict_output_charset` (disabled) overridden in memory, identically
across all 4 models, per the same proven-necessary fix from Stage 2/2b.
`max_new_tokens` raised per test set (2048 for cold-definitions, 768/512
for the shorter elicitation sets) - the only setting that varies, and it
varies identically across models.

Battery = 12 prompts x 4 models = 48 one-shot generations, **48/48
succeeded, zero errors**. Run: `20260811T193849340938_6695bf7d`, results
at `genealogy_workspace/runs/20260811T193849340938_6695bf7d/reports/
results.jsonl`. Gemma's output was specifically checked for the Stage 2
degeneration signature (`?`-flooding, empty `---` loops) per Jon's "if
Gemma comes back weird looking, use the gemma_extract config" caution -
clean, 0-6 stray `?` per response (legitimate rhetorical questions/markdown
rules, not flooding), no fallback needed.

### Test Set A (existing labels) - 4-way convergence confirmed

All 4 models, independently, one-shot, no cross-contamination, produced
the same semantic attractor for all 4 previously-flagged labels:

| Label | All 4 models' shared attractor |
|---|---|
| `cemetery_photo` | photo of/in a cemetery generally (gravestones, monuments, mausoleums, cemetery grounds) - not narrowed to individual grave-marker photos |
| `casual_photo` | informal/candid/unposed/spontaneous photographic *style*, not "ordinary personal photo regardless of pose" |
| `photo_collage` | correct core morphology (multiple photos arranged together) but all 4 added artistic/thematic/narrative intent language unprompted |
| `handwritten_ledger` | financial/accounting/bookkeeping record - all 4 explicitly said "financial transactions" or "accounting" in the first 2 sentences |

This is now a 6-checkpoint replication in total across this document
(Gemma 4 prod x2 stages, Gemma 3 on-device, Qwen2.5 on-device, plus these
4 local models, 2 of which are Gemma/Qwen and 2 of which - `qwen3vl4b`,
`internvl3_8b` - are checkpoints/families not tested anywhere else in this
document). The theory that this is a single-checkpoint quirk is no longer
tenable.

### Test Set B (candidate labels) - `structured_handwritten_record` / `handwritten_tabular_record`

All 4 models' cold definitions centered on handwriting + structured/
tabular layout (columns, rows, headers, grids) as the *defining* feature,
with financial examples appearing only as illustrative instances, not as
the core concept - consistent with Stage 4's Qwen2.5 finding. No model
independently volunteered the Stage 4-flagged "must be legible" condition
in this run's cold definitions (not a contradiction of Stage 4 - Jon's
earlier phone test found that condition too, just not reproduced as
prominently here; still an open item, not resolved either way).

### Test Set C (unanchored morphology) - negative controls PASS here, unlike Stage 4's anchored version

This is the most useful new result. Stage 4 found that direct "Would X
belong to category Y?" membership questions caused Qwen2.5 to rationalize
memberships it shouldn't (a personal letter accepted into
`handwritten_tabular_record` by inventing fake tabular structure). This
run used the **same unanchored, no-category-name-supplied elicitation**
Stage 4 later switched to for exactly that reason - and across all 4
models, the negative controls now hold cleanly:

- **Personal letter** (continuous cursive prose, lined paper): all 4
  models independently proposed names like "Handwritten Text Document" /
  "Handwritten Prose" / "Handwritten_Cursive_Letter" / "CursiveProseLetter"
  - none conflated it with a tabular/structured category.
- **Printed table** (rows/columns, fixed headings, no handwriting): all 4
  models independently proposed names like "Tabular Data Form" / "Tabular
  Data" / "Machine-Readable Table" / "TabularDocument," and 3 of 4
  explicitly wrote "no handwritten" or "without handwritten elements" in
  their own definition, unprompted.
- Census/marriage-register/passenger-manifest (the 3 positive cases) all
  got handwritten + tabular/structured names, distinct from both negative
  controls.

**Methodological finding**: the failure Stage 4 found is specific to the
anchored membership-question format, not to unanchored classification
itself. When asked to independently name/define rather than confirm/deny a
supplied label, all 4 models correctly separated "handwritten + structured
fields" from both "handwritten + free prose" and "printed + structured
fields" without any accommodation bias. **Future boundary testing on this
or any model should prefer this unanchored form over direct membership
questions.**

### Test Set D (unanchored superclass discovery)

Given the same 3-positive/2-negative exemplar set as Stage 4 (no prior
answer exposed to any model), single-word/short-phrase answers:

| Model | Proposed superclass name |
|---|---|
| `gemma` | "Tabular Record" |
| `qwen3b` | "Handwritten Tabular Records" |
| `qwen3vl4b` | "handwritten_repeated_table" |
| `internvl3_8b` | "StructuredRecordTable" (with reasoning: "repeated records in rows and columns under fixed headings") |

3 of 4 (`qwen3b`, `qwen3vl4b`, and Stage 4's external Qwen2.5 result
"Handwritten Document Classification: Row-Column Layout") explicitly
encode **both** "handwritten" and "tabular/table/row-column" in the name
itself. `gemma` and `internvl3_8b` produced names encoding the tabular/
structured half but dropped "handwritten" from the short name itself
(though `internvl3_8b`'s accompanying reasoning still correctly excluded
the letter and printed-table negatives). None of the 4 chose "ledger,"
"financial," "census," "marriage," or "manifest" - matching Stage 4's
finding that models converge on morphology-based naming, not
subject-matter or the existing production vocabulary, when given a free
choice.

**Overall Stage 5 conclusion**: the cross-model convergence on all 4
originally-flagged labels is now solid (6 checkpoints total, 4 distinct
model families). The `structured_handwritten_record`/`handwritten_tabular_
record` candidate concept continues to hold up, and this run additionally
shows the "handwritten + row/column structure" superclass is discoverable
across 4 different local models using a clean unanchored-elicitation
method, not just the phone Qwen2.5 instance from Stage 4. No taxonomy or
production config changed.

## Stage 6 - Real-image negative-control test (closes Stage 5's open gap, 2026-08-11)

Script: `scripts/real_image_negative_control_test.py`

Stage 5 flagged its own limitation: its negative controls (personal letter,
printed table) were text-DESCRIBED hypotheticals, not real images. This
stage closes that gap using 3 real images and the same 4 models
(`gemma`, `qwen3b`, `qwen3vl4b`, `internvl3_8b`), same unanchored
elicitation method (no candidate label supplied), same in-memory-only
overrides (`system_prompt` cleared, `restrict_output_charset` disabled).
12/12 generations succeeded, run `20260811T210124204603_b804b3fe`, results
at `genealogy_workspace/runs/20260811T210124204603_b804b3fe/reports/
results.jsonl`.

**Images used** (all real, all verified by direct visual inspection before
use, not just filename/manifest trust):

- **Positive** (handwritten + tabular structure): `J:\Screenshots\Kemper_
  Ancestry\Screenshot 2026-04-30 003700.png` - a real cursive German
  parish-register/baptismal entry, ruled columns, currently classified
  `dense_tabular_rows` in production.
- **Negative** (handwritten, continuous free prose, NOT tabular):
  `data/outputs/manual_test_images/kirk_session_minutes_1707_nrscotland.png`
  - a real 1707 Church of Scotland Kirk Session minute page (source:
    National Records of Scotland's blog, `blog.nrscotland.gov.uk`,
    screenshotted and supplied by Jon, cropped locally to remove phone/
    browser UI chrome before use). This was the missing case: no genuine
    handwritten personal-letter/prose image existed anywhere in the
    already-classified corpus (checked exhaustively - see the "no
    suitable image found" discussion earlier in this investigation's
    session log), and the 8 automated "kirk session notes" reference-pull
    images in `data/outputs/reference_pull/handwritten_ledger/` were all
    individually opened and confirmed to be book spines/printed title
    pages, not manuscript content - a real, separate bug in `scripts/
    pull_reference_images.py`'s LOC IIIF page-index selection (the 8 bad
    pulls all request IIIF page `:0001` for `service:rbc:...` book
    sequences, which is the cover for that source type, not an interior
    page; the 126 *good* entries in the same manifest, from a different
    query, correctly request `:0003`/`:0005`+). That bug is unfixed and
    out of scope here - flagged for separate follow-up, not touched.
- **Negative** (printed/typeset, tabular, NOT handwritten): `J:\
  Screenshots\Kemper_Ancestry\Screenshot 2026-05-02 152719.png` - a real
  Birmingham, England electoral register page (1937-1938), typeset,
  clean rows/columns, currently classified `dense_tabular_rows` in
  production.

### Result: the boundary holds on real images for 3 of 4 models; Gemma partially

| Model | Positive (name given) | Negative-prose (name given) | Negative-printed (name given) | 3-way distinct? |
|---|---|---|---|---|
| `gemma` | "Handwritten Text Document" | "Handwritten Text Document" (**same name**) | "Printed Document/Text Page" | No - reused identical name for both handwritten cases |
| `qwen3b` | "Historical Document Entry" | "Historical Document Logbook" | "Historical Electoral Register" | Yes |
| `qwen3vl4b` | "structured_table" | "handwritten_text_document" | "TEXT_DENSE_TABLE" | Yes - cleanest 3-way split |
| `internvl3_8b` | "HistoricalDocument-Table" | "HistoricalDocument-Textual" | "HistoricalDocument-TextList" | Yes |

`qwen3b`, `qwen3vl4b`, and `internvl3_8b` each independently gave **three
distinct category names** across the three real images, correctly:
including tabular/structural language ("columns", "table", "grid") only
for the two structured images, and correctly omitting it for the free-
prose Kirk Session page while still correctly identifying that page as
handwritten. None of the 4 models fell into the false-positive failure
Stage 4 found with anchored membership questions (inventing fake tabular
structure in a letter) - confirming again that the unanchored method holds
up, now on real images, not just hypotheticals.

**Gemma-specific finding, worth flagging on its own**: `gemma` gave the
*identical* category name ("Handwritten Text Document") to both the
tabular positive and the free-prose negative image - its category
*naming* did not distinguish them, even though its own accompanying
**definition** text did (the positive definition explicitly says "tabular
or columnar format"; the negative definition says "organized into lines
of text, suggesting... a letter, manuscript, or historical document" with
no tabular language at all). So Gemma's underlying visual discrimination
between the two cases is intact and correct - the gap is specifically in
how coarse its short category-name output is, not in what it actually
perceives. This is directly relevant to production, since Gemma is the
model actually running the classifier: **a real replacement taxonomy
label needs to be specific enough that Gemma can't collapse two
genuinely-different cases into the same short name**, even when its
underlying reasoning already tells them apart correctly.

**Overall Stage 6 conclusion**: the `structured_handwritten_record`/
`handwritten_tabular_record` candidate concept survives its first
real-image negative-control test. The positive/negative boundary
(handwritten+structured vs. handwritten+prose vs. printed+structured)
holds on real images for 3 of 4 models tested, and even the 4th (Gemma)
gets it right at the reasoning/definition level, just not in its shortest
possible label. No taxonomy or production config changed.

## Interpretation

The mechanism appears to be: our schema-heavy production prompt forces
Gemma to squeeze what it already perceives correctly (a physical memorial
plaque/marker) into the nearest available label in our current taxonomy,
and it picks `printed_document` because "engraved text on a surface"
pattern-matches to document more than to anything else offered. This is a
**forced-choice-with-no-good-option failure, not a perception failure** -
Gemma's own descriptions are accurate; our label set just doesn't have a
bucket that fits.

This reframes at least this subset of misclassifications: they are likely
not fixable by better prompting within the current taxonomy. They would
need an actual new category (something like `memorial_marker` or `plaque`)
that gives Gemma a bucket matching what it already sees.

## What this does NOT establish

- Whether `cemetery_photo`, `casual_photo`, and `photo_collage` (confirmed
  mismatched via text-definition testing on 6 checkpoints across 4 model
  families, Stage 1/2b/4/5) also show a stable mismatch under the
  **image**-ablation method (Stage 3) the way `printed_document`/plaque
  did - not yet tested; Stage 3 was only run against the existing corpus,
  whose flagged errors happened to be plaque/memorial cases, not
  cemetery/casual/collage cases.
- ~~Whether `structured_handwritten_record`/`handwritten_tabular_record`
  survive a real-image negative control~~ - **closed by Stage 6**: yes,
  for 3 of 4 models tested; Gemma's short category *name* conflated the
  two handwritten cases even though its definition text correctly told
  them apart (see Stage 6's Gemma-specific finding).
- Whether the "must be legible" unwanted condition (found once in Stage 4)
  is a real, reproducible property of the candidate label, or a one-off -
  not reproduced or specifically retested in Stage 5's cold definitions.
- Whether replacement labels for `cemetery_photo`, `casual_photo`, or
  `photo_collage` exist and would pass similar cold-definition + positive/
  negative testing - not yet attempted for those 3.
- Whether adding any new category actually improves end-to-end
  classification accuracy or manual-review load - no taxonomy or prompt
  change has been made or tested.
- Whether this generalizes beyond the Kemper corpus.

## Explicitly deferred (do not start unprompted)

- Changing the production taxonomy (`config/taxonomy.yaml`) - Jon: "I'd
  resist changing the production taxonomy just yet."
- The classification-accuracy ablation test (production schema-heavy prompt
  vs. bare label vs. concise semantic phrases, scored against real ground
  truth) - Jon: "keep it as a future task for now."
- Applying the same "let the model tell us how to present information"
  principle to extraction prompts (not just classification) - raised by Jon
  as a hypothesis, not yet requested as work.

## Key files

- `scripts/gemma_semantic_definition_experiment.py` - Stage 1/2/2b (text-only
  label self-report)
- `scripts/gemma_minimal_prompt_classification_ablation.py` - Stage 3 (real
  image, minimal prompt)
- `scripts/cross_model_taxonomy_semantic_audit.py` - Stage 5 (automated
  one-shot battery, 4 local models, reuses `model_console.adapter.
  ChatBackendAdapter`)
- `scripts/real_image_negative_control_test.py` - Stage 6 (same method,
  real images instead of text-described hypotheticals)
- `data/outputs/manual_test_images/kirk_session_minutes_1707_nrscotland.png`
  - the real handwritten free-prose negative-control image (Stage 6),
    manually sourced since none existed in the corpus already available
- `data/misclassifications.csv` - 20 known confident-wrong production
  classifications, used as Run A source
- `data/outputs/kemper_ancestry_review.csv` - 487-image full corpus, used as
  Run B source
- All run outputs under `genealogy_workspace/runs/<run_id>/reports/` (see
  run IDs above) - raw JSONL, one line per generation, plus
  `run_metadata.json` recording exact settings/hashes for each run
