# Ground Truth Reference — Knott/O'Connor Marriage Certificate

**Source image:** `Screenshot 2026-05-03 134309.png`
**Confirmed:** 2026-07-11, direct visual read of the source document
**Status:** [A] primary source, confirmed by direct inspection — not model output

This document was used as the primary test image across the 2026-07-10/11
model assessment session (Qwen3-VL-2B Instruct/Thinking, SmolVLM2-2.2B,
Granite Vision 3.2-2b). This reference exists because a large share of
what looked like model fabrication during that session turned out, on
direct inspection, to be real document content that was either correctly
read or misfiled into the wrong field — and separating those categories
changes how each model's results should be read. Build on this rather
than re-deriving it from memory next session.

---

## Confirmed document content (ground truth)

**Document type:** Certificate of Marriage (Ontario, Canada), form serial
number **010073** (printed top-right corner)

**Bridegroom:** George Pervis Campbell Knott, age 25
- Residence: 909 Logan Ave, Toronto
- Place of birth: Gilbert Plains, Manitoba
- Status: Bachelor
- Occupation: Machine Shop Inspector
- Religious denomination: Catholic
- Father: Alfred William Knott
- Mother: Eleanor Campbell

**Bride:** Mary Cecilia O'Connor, age 31
- Residence: 860 Carlaw Ave, Toronto
- Place of birth: Enniscorne(?), Ontario — genuinely hard to read even
  on direct inspection; an uncommon Irish-sounding place name
- Status: Spinster
- Occupation: Saleslady & Stenographer
- Religious denomination: Catholic
- Father: Stephen O'Connor
- Mother: Cecilia Condon

**Witnesses:**
- Donald Bissett, 66 Alfred St, Brantford
- Irene O'Brien, 204 Wolsey St, Peterborough

**Marriage details:**
- Married 28 September 1942, City of Toronto, County of York, Province
  of Ontario
- Officiant: Rev. John Lawlor McKenna, Holy Name Church (Catholic),
  address 71 Gough Ave, Toronto
- Minister's registration number: 12698
- Both bride and groom recorded as able to read and write ("Yes" to both
  questions)

**Registrar stamp (top area of document):** "REGISTRAR GENERAL RECEIVED
SEP 30 1942 ONT-RM" — a registration-received date **three days after**
the marriage date, genuinely a second, distinct, real date on the
document. Not model drift or duplication when both 28 Sep and 30 Sep
1942 appear in a result.

---

## NOT on the physical document (confirmed external/UI content)

- **"1826-1943"** and **"Marriages"** — this is archive-website
  navigation breadcrumb text (Ancestry.com/FamilySearch-style: "Ontario,
  Canada, Marriages, 1826-1943 › York › 1942"), visible in some source
  scans/screenshots alongside the document image, but not part of the
  physical 1942 certificate itself. Confirmed via a Gemma-4-E2B-it LM
  Studio test that showed the breadcrumb in the source thumbnail.
  Multiple models (Granite, Gemma) independently read this text
  correctly — it was genuinely visible in their input — but it does not
  belong in extracted fields for this document. If future test images
  are cropped from archive-site screenshots, expect this same category
  of confound and crop it out before testing if possible.

---

## How to read each model's session results against this ground truth

### Confirmed correct (multiple models, multiple runs)
Mary Cecilia O'Connor · Alfred William Knott · Eleanor Campbell ·
Stephen O'Connor · Toronto · York · Ontario · September 1942 · Catholic ·
"Machine Shop Inspector" · Gilbert Plains

### Real content, consistently MISFILED (not fabrication)
- **"Wolsey St, Peterboro"** repeatedly extracted as if it were a
  person's name — it's actually part of witness Irene O'Brien's address.
  Models reading this were seeing real text; the failure is
  field-classification (address fragment → personal_names instead of
  place_names/dropped), not invention. "Irene O'Brien" herself was
  dropped by every model that made this error.
- **"Cecilia Condon"** merged/garbled with "Eleanor Campbell" in several
  runs — these are two adjacent mother's-maiden-name fields on the form
  (groom's mother, bride's mother) that got concatenated into one
  string, not a fabricated name.
- **"010073"** (Granite, dumped into an unrequested `other_fields`
  block) — this is the real printed form serial number. A good catch by
  the model, just outside the requested schema; not a fabricated price
  or invented number.
- **Groom's name variants** (Perce/Percy/Perkins/Pearson/Bowie Campbell
  Knott) — actual name is "Pervis." Every variant is a phonetically
  plausible misread of a genuinely unusual middle name, not random
  invention. No model got this exactly right all session.
- **"Environne, Ontario"** (multiple models independently) vs. actual
  "Enniscorne(?), Ontario" — multiple models converging on the *same*
  wrong reading, rather than different wrong readings, is itself a
  useful signal: it usually indicates a genuinely difficult source
  rather than independent fabrication. Worth treating repeated
  cross-model convergence-on-an-error this way in future evidence
  review, distinct from single-model one-off errors.

### Genuine fabrication (not explained by document content or UI chrome)
No connection found to anything on this document or its source
scan/thumbnail:
- Places: Newark, New Brunswick, Montreal, Quebec, New York City, London
- Dates: 1873, 1946, 1950, 1958 (anything outside 28 Sep / 30 Sep 1942)

These remain genuine hallucination findings and should NOT be
reinterpreted in light of the misfiling/UI-chrome discoveries above —
they don't connect to any real content on or around this document.

---

## Process note for future sessions

A meaningful fraction of one evening's "fabrication" findings turned out
to be real content that was either (a) genuinely on the document but
misfiled into the wrong schema field, or (b) genuinely visible in the
model's input but belonging to archive-site UI chrome rather than the
document itself. Before logging a model result as fabrication in future
sessions, worth checking the actual source image directly (the
assessment tool's "Open full image" button exists for exactly this) —
plausible-but-wrong content and genuinely-real-but-misplaced content can
look identical from raw model output alone.
