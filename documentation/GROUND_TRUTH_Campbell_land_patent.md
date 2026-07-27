# Ground Truth Reference — Robert Campbell Land Patent (Dominion Lands)

**Source image:** Land grant to Robert Campbell, Dominion Lands Act patent
**Confirmed:** 2026-07-12, direct visual read of the source document
**Status:** [A] primary source, confirmed by direct inspection — not model output

Second ground-truth reference built this session, alongside
`GROUND_TRUTH_Knott_marriage_cert.md`. Exists for the same reason: to
separate confirmed-correct model output from real errors and real
omissions, so this doesn't need to be re-derived from memory next
session, and to give the map_land_record bucket a real comparison
baseline the way the marriage certificate gives printed_document /
census_or_form_text.

Tested against: olmOCR-2-7B-1025-FP8, greedy decoding, no preprocessing,
338.46s runtime. This is a first-pass reference (n=1 model, n=1 run) —
not yet cross-validated against a second model the way the marriage
certificate was.

---

## Confirmed document content (ground truth)

**Document type:** Dominion Lands Act patent ("Short Form"), Canada,
patent no. **110** (printed top-right)

**Grantee:** Robert Campbell, of Holland, Province of Manitoba, Farmer

**Land description:**
- Seventh Township, Eleventh Range, West of the Principal Meridian,
  Province of Manitoba
- The North East quarter of section Twenty-nine of the said Township
- Containing by **admeasurement** one hundred and sixty (160) acres,
  more or less

**Reservations:** Standard Dominion Lands Act reservations for
navigable waters, fishery/fishing rights, and boat landing/mooring
access — full boilerplate paragraph present and accurately transcribed.

**Signatories / officials:**
- Witness: John Joseph McGee Esquire
- Sir Gilbert John **Elliot**, Earl of Minto and Viscount Melgund of
  Melgund, County of Forfar; Baron Minto of Minto, County of Roxburgh;
  Baronet of Nova Scotia; Knight Grand Cross of the Order of Saint
  Michael and Saint George — Governor General of Canada
- Date: Ottawa, the twenty-third day of June, one thousand nine hundred
  and three (1903), third year of the Reign
- By Command: P. Pelletier, Acting Under-Secretary of State
- T.G. Rothwell, Acting Deputy of the Minister of the Interior

**Marginal/administrative content, printed on the document but separate
from the main granting text:**
- Plat No. **95726** (bottom left)
- Left-margin recording block: "Recorded in the Department of the
  Interior... Liber [33?]... folio [110?]... 1903 June" (partially
  illegible even on direct inspection - dates/numbers approximate)
- "SHORT FORM." octagonal stamp/seal, top left

---

## How olmOCR's first test result compares

### Confirmed correct
Full legal boilerplate paragraph (navigable waters/fishery reservations,
transcribed accurately and completely) · Robert Campbell, Holland,
Manitoba, Farmer · Seventh Township / Eleventh Range / West of Principal
Meridian · North East quarter of section Twenty-nine · 160 acres ·
"John Joseph McGee Esquire" (correctly resolved from cursive signature,
apparently by cross-referencing the document's own printed header "JOHN
J. McGEE / DEPUTY GOVERNOR" - worth noting as a real capability, not
luck) · Ottawa, 23 June 1903, third year of the Reign · P. Pelletier ·
T.G. Rothwell · both officials' full titles

### Real transcription error
"containing by **advertisement**" - should be "**admeasurement**" (a
real surveying/legal term meaning "as measured"). Same mechanism as
other substitution errors seen this session (an unusual-but-correct
word overridden by a more common one) - see
GROUND_TRUTH_Knott_marriage_cert.md for the York→New York, Brantford→
Brampton pattern this matches.

### Real proper-name error
"Sir Gilbert John **Ellis**, Earl of Minto" - document clearly prints
"**Elliot**." Not a plausible alternate reading here; a genuine miss on
an unusual surname, same general category as this session's repeated
struggles with "Pervis" (marriage certificate) but a clean single
substitution, not scattered guesses.

### Real content omissions (not errors - simply not transcribed)
- Patent number **"110"** (top right)
- **"Plat No 95726"** (bottom left)
- Left-margin recording/filing metadata block (Liber/folio/date detail)

These are exactly the kind of unique reference/citation numbers that
matter for genealogical sourcing specifically, even though the main
granting text came through cleanly. Worth flagging as a distinct
failure shape from the marriage certificate test, which had no
comparable omissions - margin/stamp content on this document type may
need particular attention in future testing, not just narrative
transcription accuracy.

---

## Process note

This document type (dense archaic legal narrative, no table structure,
marginal reference numbers separate from main text) stresses different
things than the marriage certificate (structured form, no dense
narrative prose, no marginal filing metadata). Both are needed as
reference points precisely because they don't share the same failure
modes - a model doing well on one doesn't guarantee doing well on the
other, and this document already demonstrated that (strong on
connected prose, weaker on marginal/stamped identifying data).
