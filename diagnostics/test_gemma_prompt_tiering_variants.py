"""
ADOPTED (2026-08-06): v7_gated_census_first is the current-best routing
tree - census gate asked BEFORE manifest gate, original (non-differential)
gate wording. Matches the pre-hardening baseline's 7/10 with zero manifest
regressions, and is the simplest of the three variants (v7/v8/v9) that
fixed the v3-v6 regression on e078_e001946617 - v9's added differential
"household grouping" cue content scored identically to v7's plain order
flip, so the extra complexity wasn't earning its keep. GATE_CENSUS_PROMPT
+ GATE_MANIFEST_PROMPT below, asked in that order via
route_with_gated_binary(first="census", ...), is what
diagnostics/test_all_vlms_prompt_tiering.py reuses as "the" tree when
testing other VLMs against this same task. S3HY-6LL2-RG (the deliberately
out-of-taxonomy case) is UNSOLVED by any variant tested here (v1 through
v9 all fail it identically) - that's a separate, still-open investigation
outside Tier 2 entirely (see the batch run's final comparison table).

Batch-compares multiple Tier 2 (record_type) prompt candidates against
the same 10-image ground-truth set, in ONE model load - per Jon's
request to chain prompt variants back to back instead of paying the
~19s weight-load cost (plus a fresh Python process) for every prompt
edit, which is what running diagnostics/test_gemma_prompt_tiering.py
repeatedly was costing.

Further optimization beyond just "one load": Tier 0 (family) and Tier 1
(layout) prompts don't change between variants - only Tier 2 (and,
conditionally, Tier 3) do. So Tier 0/1 are run ONCE per image and
cached, not re-run once per variant. Tier 3 (census year) is also
cached per image the first time any variant reaches it, since the
census-year question and the image are the same regardless of which
Tier 2 wording routed there - only genuinely re-run if a variant's
Tier 2 result routes to census for the first time.

Variants tested (see TIER2_VARIANTS below):
  - v1_original: the first Tier 2 wording from the original design
    (before Jon's hardening pass) - baseline.
  - v2_census_hardened: Jon's "dense tabular rows" tightening, verified
    via 4 manual LM Studio A/B tests (fixed a narrative-text false
    positive without breaking a real census page) but never run
    against the full ground-truth batch until now.
  - v3_manifest_hardened: Jon's full schema + "INSTRUCTIONS TO PURSERS"
    passenger_manifest hardening - already batch-tested once (6/10,
    regressed vs v1's 8/10 - a real census image, e078_e001946617,
    started getting misclassified as passenger_manifest).
  - v4_softened_manifest: same structural manifest cues as v3, minus
    the "almost always... do not call it unknown" absolutism that's
    the suspected cause of v3's regression - untested until now.

Usage:
    python diagnostics/test_gemma_prompt_tiering_variants.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from core.loaders.base_loader import load_model_config
from core.loader_registry import LOADER_REGISTRY
from diagnostics.test_gemma_prompt_tiering import (
    TEST_CASES, WORKING_DIR, TIER0_PROMPT, TIER1_PROMPT, TIER3_PROMPT,
    parse_fields, parse_confidence, MIN_CONFIDENCE,
)

TIER2_VARIANTS = {
    "v1_original": """This document has document-record layout. Classify its RECORD TYPE.
Answer with exactly two lines: record_type: <census|passenger_manifest|unknown>
- census: a government population census return or schedule.
- passenger_manifest: a ship's passenger list or immigration manifest.
- unknown: neither of the above, or genuinely unclear.
Also give a confidence score for this answer: confidence: <a number from 0.0 to 1.0>
0.0 means a pure guess, 1.0 means completely certain. Score your ACTUAL certainty about
THIS SPECIFIC image, not a default value - a genuinely ambiguous or unfamiliar image should
score low even if you still have to pick one of the listed options.
Answer only with the two lines above, nothing else.""",

    "v2_census_hardened": """This document has document-record layout. Classify its RECORD TYPE.
Answer with exactly two lines: record_type: <census|passenger_manifest|unknown>
- census: a government population census form with dense tabular rows
- passenger_manifest: a ship's passenger list or immigration manifest.
- unknown: neither of the above, or genuinely unclear.
Also give a confidence score for this answer: confidence: <a number from 0.0 to 1.0>
0.0 means a pure guess, 1.0 means completely certain. Score your ACTUAL certainty about
THIS SPECIFIC image, not a default value - a genuinely ambiguous or unfamiliar image should
score low even if you still have to pick one of the listed options.
Answer only with the two lines above, nothing else.""",

    "v3_manifest_hardened": """Classify this historical form image.

record_type must be one of: census | passenger_manifest | unknown

passenger_manifest = any ship's passenger list or immigration manifest.
Look for: "INSTRUCTIONS TO PURSERS", "Page No.", name columns, age/sex/occupation columns, rows of passenger names.
Even partially filled or sparse pages are passenger_manifest. The printed form structure decides it.

census = government census schedule with household groups and relationship columns.

unknown = only if it is clearly neither.

IMPORTANT: This image type with "INSTRUCTIONS TO PURSERS" at the top is almost always passenger_manifest. Do not call it unknown or instruction page.

Output EXACTLY these lines, nothing else:

record_type: <census|passenger_manifest|unknown>
confidence: <0.0-1.0>
text_density: <0.0-1.0>
handwriting: <true|false>
table_layout: <true|false>
faces: <true|false>
map_like: <true|false>
reason: <≤30 words>""",

    "v4_softened_manifest": """Classify this historical form image.

record_type must be one of: census | passenger_manifest | unknown

passenger_manifest = any ship's passenger list or immigration manifest.
Look for actual visible evidence: "INSTRUCTIONS TO PURSERS" text, "Page No." text, name columns,
age/sex/occupation columns, rows of passenger names. Even partially filled or sparse pages can
still be passenger_manifest if this structure is actually visible.

census = a government population census schedule - typically dense tabular rows with household
groups and relationship-to-head-of-household columns, NOT passenger-list-style columns.

unknown = only if it is clearly neither, or genuinely unclear which.

Base your answer on what is ACTUALLY VISIBLE in THIS image, not on a general assumption about
which category is more common - a real census page must still be classified as census even if it
has some structural similarity to a manifest.

Output EXACTLY these lines, nothing else:

record_type: <census|passenger_manifest|unknown>
confidence: <0.0-1.0>
text_density: <0.0-1.0>
handwriting: <true|false>
table_layout: <true|false>
faces: <true|false>
map_like: <true|false>
reason: <≤30 words>""",

    "v5_balanced_features": """Classify this historical form image.

record_type must be one of:
- census
- passenger_manifest
- unknown

Definitions

passenger_manifest
A ship's passenger list or immigration manifest.

Identify it primarily by its PRINTED FORM LAYOUT, even if mostly blank.

Common features include one or more of:
- repeated passenger rows
- name columns
- age / sex / occupation columns
- nationality, destination, last residence, race, or birthplace
- vessel or voyage information
- "Page No."
- "INSTRUCTIONS TO PURSERS"
- "LIST OF ALIENS PASSENGERS"
- "Manifest"
- immigration or customs wording

The printed form determines the type.
Blank, sparse, or partially completed manifest forms are still passenger_manifest.

census
A government census schedule organized into households or families.

Typical features:
- relationship to head
- marital status
- dwelling or family number
- household grouping
- census schedule headings

unknown
Only if the form clearly matches neither definition.

Do not classify by amount of handwriting.
Do not classify by how full the page is.
Classify by the printed document design.

Output EXACTLY:

record_type: <census|passenger_manifest|unknown>
confidence: <0.0-1.0>
text_density: <0.0-1.0>
handwriting: <true|false>
table_layout: <true|false>
faces: <true|false>
map_like: <true|false>
reason: <≤30 words>""",
}

# Gated binary approach (Jon's hypothesis, 2026-08-06): every v1-v5
# variant above asks the model to hold BOTH category definitions in mind
# at once and pick between them - and every variant that enriched the
# manifest side with structural cues caused the SAME regression on a
# real census page (e078_e001946617), regardless of how the wording was
# softened. Hypothesis: the two categories are similar enough (both
# dense tabular forms with name/data columns) that a single simultaneous
# choice lets the manifest cue list "leak" into census pages. Splitting
# into two ISOLATED yes/no gates - each only ever reasoning about ONE
# category's definition, never comparing both at once - tests whether
# that removes the cross-contamination. Manifest gate asked first
# (existing variants all showed a manifest-ward bias, so gating it out
# first is the more conservative order); census gate only asked if the
# manifest gate said no.
GATE_MANIFEST_PROMPT = """Look ONLY at this question: is this image a ship's passenger list or immigration manifest form?

Identify it by its PRINTED FORM LAYOUT, even if mostly blank. Look for one or more of:
- repeated passenger rows
- name columns
- age / sex / occupation columns
- nationality, destination, last residence, race, or birthplace columns
- vessel or voyage information
- "Page No."
- "INSTRUCTIONS TO PURSERS"
- "LIST OF ALIENS PASSENGERS"
- "Manifest"
- immigration or customs wording

Blank, sparse, or partially completed manifest forms still count as yes.
Do not consider whether it might instead be a census form - answer only this one question.

Output EXACTLY these two lines, nothing else:
is_manifest: <yes|no>
confidence: <0.0-1.0>"""

GATE_CENSUS_PROMPT = """Look ONLY at this question: is this image a government population census schedule?

Identify it by its PRINTED FORM LAYOUT. Look for one or more of:
- relationship to head of household column
- marital status column
- dwelling or family number
- household or family grouping
- census schedule headings or titles
- dense tabular rows organized by household

Do not consider whether it might instead be a passenger manifest - answer only this one question.

Output EXACTLY these two lines, nothing else:
is_census: <yes|no>
confidence: <0.0-1.0>"""

# v7 (2026-08-06): pure order flip of v6 - SAME gate prompts, census asked
# first instead of manifest. Isolates "does order alone matter" from "does
# the cue-list content matter" - v6 already showed content matters (both
# gate orders in v6 failed the same way when manifest went first); this
# checks whether asking the household-grouping question BEFORE the
# manifest cue list ever gets introduced changes anything.

# v8/v9 (2026-08-06): differential framing - v3 through v6 all only gave
# each category a list of features it HAS. None told the model what a
# real distinguishing feature LACKS in the other category. Census schedules
# group people into households (relationship-to-head-of-household column,
# dwelling/family number) - manifests do not. This is the one structural
# difference between the two forms that isn't just "more of the same kind
# of column," so it's used here as the explicit deciding feature, in both
# a single three-way prompt (v8) and a gated-binary pair (v9, census-first
# per v7's order).
TIER2_VARIANTS["v8_differential_single"] = """Classify this historical form image.

record_type must be one of: census | passenger_manifest | unknown

The key distinguishing feature: census schedules group people into HOUSEHOLDS with a
"relationship to head of household" column (e.g. wife, son, daughter, lodger) and a
dwelling/family number. Passenger manifests do NOT have this - they list individual
passengers row by row with NO household-relationship structure, even though both may have
similar-looking name/age/occupation columns.

census = has household grouping AND a relationship-to-head-of-household column (or equivalent
family/dwelling number). This is the deciding feature - if present, it is census even if the
form also has passenger-list-like columns.

passenger_manifest = row-by-row passenger list WITHOUT household-relationship structure. May
include: "Page No.", "INSTRUCTIONS TO PURSERS", nationality/destination/last residence columns,
vessel or voyage information. Blank or sparse manifest forms still count.

unknown = only if the form clearly matches neither definition.

First check specifically: does this form have a relationship-to-head-of-household or
family/dwelling grouping column? If yes, it is census regardless of any other similarity to a
manifest layout.

Output EXACTLY:

record_type: <census|passenger_manifest|unknown>
confidence: <0.0-1.0>
text_density: <0.0-1.0>
handwriting: <true|false>
table_layout: <true|false>
faces: <true|false>
map_like: <true|false>
reason: <≤30 words>"""

GATE_CENSUS_DIFFERENTIAL_PROMPT = """Look ONLY at this question: does this image show a government population census schedule?

The deciding feature is a HOUSEHOLD/FAMILY GROUPING structure: a "relationship to head of
household" column (e.g. wife, son, daughter, lodger, boarder) and/or a dwelling or family
number grouping people into households. Passenger manifests do NOT have this feature, even
though both document types can have similar-looking name/age/occupation columns - so column
similarity alone is NOT enough to answer yes here.

Answer yes ONLY if you can see actual household/family relationship structure. Otherwise
answer no, even if the page has dense tabular rows.

Do not consider whether it might instead be a passenger manifest - answer only this one question.

Output EXACTLY these two lines, nothing else:
is_census: <yes|no>
confidence: <0.0-1.0>"""

GATE_MANIFEST_DIFFERENTIAL_PROMPT = """Look ONLY at this question: is this image a ship's passenger list or immigration manifest form?

Identify it by its PRINTED FORM LAYOUT, even if mostly blank. Look for one or more of:
- repeated passenger rows WITHOUT household/family relationship structure
- name columns
- age / sex / occupation columns
- nationality, destination, last residence, race, or birthplace columns
- vessel or voyage information
- "Page No."
- "INSTRUCTIONS TO PURSERS"
- "LIST OF ALIENS PASSENGERS"
- "Manifest"
- immigration or customs wording

If the form groups people into households with a relationship-to-head-of-household column, it is
NOT a manifest even if it also has some of the features above - answer no in that case.

Blank, sparse, or partially completed manifest forms still count as yes, as long as no
household-grouping structure is present.

Do not consider whether it might instead be a census form - answer only this one question.

Output EXACTLY these two lines, nothing else:
is_manifest: <yes|no>
confidence: <0.0-1.0>"""


def load_gemma():
    print("Loading gemma (config/models/gemma.yaml)...")
    model_cfg = load_model_config("gemma")
    model_cfg.prompt_text = ""
    loader_cls = LOADER_REGISTRY.get(model_cfg.loader_class)
    loader = loader_cls(model_cfg)
    t0 = time.time()
    loader.initialize_model_and_tokenizer()
    print(f"Loaded in {time.time() - t0:.1f}s\n")
    return loader


def precompute_tier01(loader, images: dict) -> dict:
    """Runs Tier 0 + Tier 1 once per image - identical across every Tier 2
    variant, so paying for it once instead of once-per-variant is a real,
    not just cosmetic, saving (2 of the tree's up-to-4 calls, for every
    image, times however many variants would otherwise repeat them)."""
    cache = {}
    for stem, image in images.items():
        raw0 = loader._run_generate(image, TIER0_PROMPT)
        f0 = parse_fields(raw0)
        family = f0.get("family", "<unparsed>")
        conf0 = parse_confidence(f0)

        entry = {"family": family, "conf0": conf0, "raw0": raw0,
                 "layout": None, "conf1": None, "raw1": None}

        if conf0 is not None and conf0 >= MIN_CONFIDENCE and family == "document_content":
            raw1 = loader._run_generate(image, TIER1_PROMPT)
            f1 = parse_fields(raw1)
            entry["layout"] = f1.get("layout", "<unparsed>")
            entry["conf1"] = parse_confidence(f1)
            entry["raw1"] = raw1

        cache[stem] = entry
    return cache


def route_with_variant(loader, image, tier01_entry, tier2_prompt, tier3_cache, stem):
    """Tier 0/1 already resolved (tier01_entry) - only Tier 2 (this
    variant's prompt) and, conditionally, Tier 3 (cached across variants
    per image) run here."""
    path = [
        ("tier0_family", tier01_entry["family"], tier01_entry["conf0"], tier01_entry["raw0"]),
    ]
    conf0 = tier01_entry["conf0"]
    family = tier01_entry["family"]
    if conf0 is None or conf0 < MIN_CONFIDENCE:
        return {"leaf": "quarantine_low_confidence", "path": path}
    if family != "document_content":
        return {"leaf": family, "path": path}

    layout = tier01_entry["layout"]
    conf1 = tier01_entry["conf1"]
    path.append(("tier1_layout", layout, conf1, tier01_entry["raw1"]))
    if conf1 is None or conf1 < MIN_CONFIDENCE:
        return {"leaf": "quarantine_low_confidence", "path": path}
    if layout in ("narrative", "uncertain", "<unparsed>"):
        return {"leaf": f"layout_{layout}", "path": path}

    raw2 = loader._run_generate(image, tier2_prompt)
    f2 = parse_fields(raw2)
    record_type = f2.get("record_type", "<unparsed>")
    conf2 = parse_confidence(f2)
    path.append(("tier2_record_type", record_type, conf2, raw2))
    if conf2 is None or conf2 < MIN_CONFIDENCE:
        return {"leaf": "quarantine_low_confidence", "path": path}
    if record_type != "census":
        return {"leaf": record_type, "path": path}

    if stem not in tier3_cache:
        raw3 = loader._run_generate(image, TIER3_PROMPT)
        f3 = parse_fields(raw3)
        tier3_cache[stem] = {
            "year": f3.get("census_year", "<unparsed>"),
            "conf3": parse_confidence(f3),
            "raw3": raw3,
        }
    t3 = tier3_cache[stem]
    path.append(("tier3_census_year", t3["year"], t3["conf3"], t3["raw3"]))
    if t3["conf3"] is None or t3["conf3"] < MIN_CONFIDENCE:
        return {"leaf": "quarantine_low_confidence", "path": path}
    return {"leaf": f"census_{t3['year']}", "path": path}


def route_with_gated_binary(loader, image, tier01_entry, tier3_cache, stem,
                             first="manifest", manifest_prompt=GATE_MANIFEST_PROMPT,
                             census_prompt=GATE_CENSUS_PROMPT):
    """Generalized version of the gated-binary approach: two ISOLATED
    binary gates instead of route_with_variant()'s single three-way Tier 2
    call, never asking the model to weigh both category definitions in
    the same call. `first` picks which gate runs first ("manifest" or
    "census" - v6 used manifest-first, v7/v9 test census-first).
    `manifest_prompt`/`census_prompt` let the SAME order logic be reused
    with either the original gate wording (v6/v7) or the differential-cue
    wording (v9) - keeps order and content as independently testable
    variables instead of conflating them in separate hand-written
    functions. Tier 0/1 still shared/cached exactly as in
    route_with_variant()."""
    path = [
        ("tier0_family", tier01_entry["family"], tier01_entry["conf0"], tier01_entry["raw0"]),
    ]
    conf0 = tier01_entry["conf0"]
    family = tier01_entry["family"]
    if conf0 is None or conf0 < MIN_CONFIDENCE:
        return {"leaf": "quarantine_low_confidence", "path": path}
    if family != "document_content":
        return {"leaf": family, "path": path}

    layout = tier01_entry["layout"]
    conf1 = tier01_entry["conf1"]
    path.append(("tier1_layout", layout, conf1, tier01_entry["raw1"]))
    if conf1 is None or conf1 < MIN_CONFIDENCE:
        return {"leaf": "quarantine_low_confidence", "path": path}
    if layout in ("narrative", "uncertain", "<unparsed>"):
        return {"leaf": f"layout_{layout}", "path": path}

    def ask_manifest_gate():
        raw = loader._run_generate(image, manifest_prompt)
        f = parse_fields(raw)
        val = f.get("manifest", f.get("is_manifest", "<unparsed>"))
        conf = parse_confidence(f)
        path.append(("gate_manifest", val, conf, raw))
        return val, conf

    def ask_census_gate():
        raw = loader._run_generate(image, census_prompt)
        f = parse_fields(raw)
        val = f.get("census", f.get("is_census", "<unparsed>"))
        conf = parse_confidence(f)
        path.append(("gate_census", val, conf, raw))
        return val, conf

    def resolve_census_leaf():
        if stem not in tier3_cache:
            raw3 = loader._run_generate(image, TIER3_PROMPT)
            f3 = parse_fields(raw3)
            tier3_cache[stem] = {
                "year": f3.get("census_year", "<unparsed>"),
                "conf3": parse_confidence(f3),
                "raw3": raw3,
            }
        t3 = tier3_cache[stem]
        path.append(("tier3_census_year", t3["year"], t3["conf3"], t3["raw3"]))
        if t3["conf3"] is None or t3["conf3"] < MIN_CONFIDENCE:
            return "quarantine_low_confidence"
        return f"census_{t3['year']}"

    if first == "manifest":
        val, conf = ask_manifest_gate()
        if conf is None or conf < MIN_CONFIDENCE:
            return {"leaf": "quarantine_low_confidence", "path": path}
        if val == "yes":
            return {"leaf": "passenger_manifest", "path": path}
        val2, conf2 = ask_census_gate()
        if conf2 is None or conf2 < MIN_CONFIDENCE:
            return {"leaf": "quarantine_low_confidence", "path": path}
        if val2 != "yes":
            return {"leaf": "unknown", "path": path}
        return {"leaf": resolve_census_leaf(), "path": path}
    else:
        val, conf = ask_census_gate()
        if conf is None or conf < MIN_CONFIDENCE:
            return {"leaf": "quarantine_low_confidence", "path": path}
        if val == "yes":
            return {"leaf": resolve_census_leaf(), "path": path}
        val2, conf2 = ask_manifest_gate()
        if conf2 is None or conf2 < MIN_CONFIDENCE:
            return {"leaf": "quarantine_low_confidence", "path": path}
        if val2 != "yes":
            return {"leaf": "unknown", "path": path}
        return {"leaf": "passenger_manifest", "path": path}


def is_correct(leaf: str, ground_truth: str) -> bool:
    return leaf == ground_truth or (
        ground_truth == "unknown" and leaf == "quarantine_low_confidence"
    )


def main():
    loader = load_gemma()

    images = {}
    for stem, _ in TEST_CASES:
        p = WORKING_DIR / f"{stem}.jpg"
        if p.exists():
            images[stem] = Image.open(p).convert("RGB")
        else:
            print(f"SKIP {stem}: not found")

    print("Precomputing Tier 0 + Tier 1 once per image (shared across all variants)...")
    t0 = time.time()
    tier01 = precompute_tier01(loader, images)
    print(f"Done in {time.time() - t0:.1f}s\n")

    tier3_cache: dict = {}
    variant_results = {}

    for variant_name, tier2_prompt in TIER2_VARIANTS.items():
        print(f"=== Variant: {variant_name} ===")
        correct = 0
        total = 0
        rows = []
        for stem, ground_truth in TEST_CASES:
            if stem not in images:
                continue
            result = route_with_variant(
                loader, images[stem], tier01[stem], tier2_prompt, tier3_cache, stem,
            )
            leaf = result["leaf"]
            ok = is_correct(leaf, ground_truth)
            total += 1
            correct += int(ok)
            rows.append((stem, ground_truth, leaf, ok))
            mark = "OK  " if ok else "MISS"
            print(f"  [{mark}] {stem}: ground_truth={ground_truth!r} leaf={leaf!r}")

        print(f"  -> {correct}/{total}\n")
        variant_results[variant_name] = {"correct": correct, "total": total, "rows": rows}

    # (name, first-gate, manifest_prompt, census_prompt)
    # v6: original gate wording, manifest first (already run/known: 6/10)
    # v7: SAME gate wording as v6, order flipped to census-first - isolates
    #     whether order alone matters (v6 already showed cue-list CONTENT
    #     matters; this checks order independent of that)
    # v9: differential-cue gate wording (explicit household-grouping
    #     discriminator), census-first per v7's order
    gated_configs = [
        ("v6_gated_binary", "manifest", GATE_MANIFEST_PROMPT, GATE_CENSUS_PROMPT),
        ("v7_gated_census_first", "census", GATE_MANIFEST_PROMPT, GATE_CENSUS_PROMPT),
        ("v9_gated_differential", "census", GATE_MANIFEST_DIFFERENTIAL_PROMPT, GATE_CENSUS_DIFFERENTIAL_PROMPT),
    ]
    for name, first, manifest_prompt, census_prompt in gated_configs:
        print(f"=== Variant: {name} ===")
        correct = 0
        total = 0
        rows = []
        for stem, ground_truth in TEST_CASES:
            if stem not in images:
                continue
            result = route_with_gated_binary(
                loader, images[stem], tier01[stem], tier3_cache, stem,
                first=first, manifest_prompt=manifest_prompt, census_prompt=census_prompt,
            )
            leaf = result["leaf"]
            ok = is_correct(leaf, ground_truth)
            total += 1
            correct += int(ok)
            rows.append((stem, ground_truth, leaf, ok))
            mark = "OK  " if ok else "MISS"
            print(f"  [{mark}] {stem}: ground_truth={ground_truth!r} leaf={leaf!r}")
        print(f"  -> {correct}/{total}\n")
        variant_results[name] = {"correct": correct, "total": total, "rows": rows}

    print("=== COMPARISON TABLE ===")
    print(f"{'variant':<24}{'score':>8}")
    for name, r in variant_results.items():
        print(f"{name:<24}{r['correct']}/{r['total']:>6}")

    print("\n=== The two cases that matter most (regression watch) ===")
    for stem in ("e078_e001946617", "S3HY-6LL2-RG"):
        gt = dict(TEST_CASES)[stem]
        print(f"\n{stem} (ground_truth={gt!r}):")
        for name, r in variant_results.items():
            row = next((x for x in r["rows"] if x[0] == stem), None)
            if row:
                _, _, leaf, ok = row
                print(f"  {name:<24} leaf={leaf!r}  {'OK' if ok else 'MISS'}")


if __name__ == "__main__":
    main()
