"""
Prompt-tiering research (2026-08-05) - separate question from the cache-
reuse mechanism investigated in docs/GEMMA_HIERARCHICAL_ROUTING_
INVESTIGATION.md (parked: not worth building). THIS script deliberately
does NOT reuse the vision tensor - every tier is a fully independent
cold call through the real, production `GemmaLoader._run_generate()`
path (same system prompt handling, same restrict_output_charset gate,
same image_token_budget - the only difference from core/classifier.py's
own classify() is that _parse_classification()'s flat category/
confidence/reason schema doesn't fit a multi-tier prompt, so this script
does its own lightweight key:value parsing per tier instead, same
pattern diagnostics/test_gemma_document_subtype.py already uses).

Question under test: does decomposing one large taxonomy prompt into a
small Decision-Engine-routed tree of narrow prompts (per the Phase 2
architecture design) actually route real images to the correct leaf -
not "is caching worth it" (settled, parked), but "is the DECOMPOSITION
ITSELF sound" - does narrowing help, does an early wrong turn foreclose
the right answer the way the design doc's risk section predicted, does
per-tier abstention work.

Ground truth: reuses the 10 hand-verified cases from diagnostics/
test_gemma_document_subtype.py (real pixel-level inspection, not
assumed) - covers 3 census years, 4 handwritten manifests (including a
known-hard case, e003566165), 1 printed manifest, and 1 genuinely out-
of-taxonomy case (S3HY-6LL2-RG, a portrait Cabin-class manifest) to see
what an early tier does with something that doesn't fit the tree at all.

Tree (Decision Engine logic below `route_one_image()`), matching the
Phase 2 design: family -> layout -> record_type -> census_year. Layout's
tabular/printed/handwritten split doesn't cleanly separate from record-
type in the real taxonomy (a passenger manifest can be printed OR
handwritten AND is inherently row/column structured) - this script
deliberately continues to record_type from printed/handwritten/tabular
alike (only "narrative"/"uncertain" terminate early), rather than
forcing the original design sketch's stricter tabular-only edge, since
that stricter edge would reject real manifests before they ever reach
the record_type question. Recording this adjustment here so it isn't
silently lost - the original 4-way layout split doesn't nest as cleanly
under real taxonomy as the initial sketch assumed.

Usage:
    python diagnostics/test_gemma_prompt_tiering.py
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

WORKING_DIR = PROJECT_ROOT / "data" / "working"

# (stem, ground_truth_leaf) - ground truth is the FINAL leaf this tree
# should reach, not a document_type string from the old flat taxonomy.
# "unknown" ground truth = deliberately out-of-taxonomy, included to see
# what early tiers do with something that shouldn't reach a clean leaf.
TEST_CASES = [
    ("1921_022-E002880409", "census_1921"),
    ("1931_174-e011707164", "census_1931"),
    ("e078_e001946617", "census_1911"),
    ("30807_A000676-00099(1)", "passenger_manifest"),
    ("CANIMM1913PLIST_2000908421-00487", "passenger_manifest"),
    ("e003559198", "passenger_manifest"),
    ("e003558130", "passenger_manifest"),
    ("IMCANQC1865_T4821-00622", "passenger_manifest"),
    ("e003566165", "passenger_manifest"),  # known-hard case, flagged before
    ("S3HY-6LL2-RG", "unknown"),  # out-of-taxonomy on purpose
]

CONFIDENCE_INSTRUCTION = """Also give a confidence score for this answer: confidence: <a number from 0.0 to 1.0>
0.0 means a pure guess, 1.0 means completely certain. Score your ACTUAL certainty about
THIS SPECIFIC image, not a default value - a genuinely ambiguous or unfamiliar image should
score low even if you still have to pick one of the listed options."""

TIER0_PROMPT = f"""Look at this image and classify its PROCESSING FAMILY.
Answer with exactly two lines: family: <visual_content|document_content|mixed_uncertain>
- visual_content: mostly a photograph, portrait, landscape, or map with little or no readable text structure.
- document_content: mostly structured readable text or writing (forms, ledgers, letters, printed pages).
- mixed_uncertain: genuinely unclear which of the above applies, or a roughly even mix.
{CONFIDENCE_INSTRUCTION}
Answer only with the two lines above, nothing else."""

TIER1_PROMPT = f"""This image has already been identified as document content. Classify its LAYOUT.
Answer with exactly two lines: layout: <printed|handwritten|tabular|narrative|uncertain>
- printed: printed or typeset text, not organized in a row/column grid.
- handwritten: primarily handwritten text, not organized in a row/column grid.
- tabular: organized in rows and columns (a form, table, ledger, or list), whether printed or handwritten.
- narrative: continuous prose text (a letter, document body), not row/column structured.
- uncertain: genuinely cannot tell.
{CONFIDENCE_INSTRUCTION}
Answer only with the two lines above, nothing else."""

# v2 (2026-08-06) - hardened by Jon via manual LM Studio A/B testing:
# fixed a confirmed false-positive (a plain narrative-text page was
# confidently (1.0) misclassified as census by the v1 wording) without
# breaking the true positive (a real census schedule still correctly
# classifies as census) - verified 4/4 across narrative text, a
# newspaper notice, a death certificate, and a real census page before
# being brought into this script. Full schema (not just record_type)
# so this doubles as a test of whether Tier 2 could directly emit the
# same field set core/classifier.py's ClassificationResult expects.
TIER2_PROMPT = """Classify this historical form image.

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
reason: <≤30 words>"""

TIER3_PROMPT = f"""This document has been identified as a census return. Identify its CENSUS YEAR from the form's title, headers, or printed year markings.
Answer with exactly two lines: census_year: <1901|1906|1911|1916|1921|1926|1931|unknown>
{CONFIDENCE_INSTRUCTION}
Answer only with the two lines above, nothing else."""

# Same threshold core/classifier.py's own min_confidence already uses
# (config/pipeline.yaml's classifier.min_confidence) - reusing it here
# rather than picking a fresh number, so this experiment's quarantine
# behavior is directly comparable to production's existing bar, not an
# arbitrarily different one.
MIN_CONFIDENCE = 0.6


def parse_fields(raw_output: str) -> dict[str, str]:
    fields = {}
    for line in raw_output.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lstrip("-*").strip().lower()
        if key:
            fields[key.split()[-1] if " " in key else key] = value.strip().lower()
    return fields


def parse_confidence(fields: dict[str, str]) -> float | None:
    raw = fields.get("confidence")
    if raw is None:
        return None
    try:
        return float(raw.split()[0])
    except (ValueError, IndexError):
        return None


def route_one_image(loader, image) -> dict:
    """Runs the tiered tree against one image, one tier at a time, each a
    fully independent loader._run_generate() call (real production call
    shape - system prompt, charset mask, image_token_budget all applied
    exactly as core/classifier.py's own classify() would). Decision
    Engine logic (which tier to ask next, when to stop) lives here, not
    in any prompt - matches the Phase 2 design's "Gemma only answers
    questions" requirement.

    v2 (2026-08-05): quarantines on a NUMERIC confidence field below
    MIN_CONFIDENCE at ANY tier, checked BEFORE branching on that tier's
    category answer - not on the model voluntarily picking an "uncertain"/
    "unknown" enum value. v1 of this script relied on the enum alone and
    the model never once self-selected it, even on a deliberately out-of-
    taxonomy test case - this is the fix for that gap, mirroring how
    core/classifier.py's own min_confidence threshold already works
    against a numeric field, not an enum choice."""
    path = []

    raw0 = loader._run_generate(image, TIER0_PROMPT)
    f0 = parse_fields(raw0)
    family = f0.get("family", "<unparsed>")
    conf0 = parse_confidence(f0)
    path.append(("tier0_family", family, conf0, raw0))
    if conf0 is None or conf0 < MIN_CONFIDENCE:
        return {"leaf": "quarantine_low_confidence", "path": path}
    if family != "document_content":
        return {"leaf": family, "path": path}

    raw1 = loader._run_generate(image, TIER1_PROMPT)
    f1 = parse_fields(raw1)
    layout = f1.get("layout", "<unparsed>")
    conf1 = parse_confidence(f1)
    path.append(("tier1_layout", layout, conf1, raw1))
    if conf1 is None or conf1 < MIN_CONFIDENCE:
        return {"leaf": "quarantine_low_confidence", "path": path}
    if layout in ("narrative", "uncertain", "<unparsed>"):
        return {"leaf": f"layout_{layout}", "path": path}

    raw2 = loader._run_generate(image, TIER2_PROMPT)
    f2 = parse_fields(raw2)
    record_type = f2.get("record_type", "<unparsed>")
    conf2 = parse_confidence(f2)
    path.append(("tier2_record_type", record_type, conf2, raw2))
    if conf2 is None or conf2 < MIN_CONFIDENCE:
        return {"leaf": "quarantine_low_confidence", "path": path}
    if record_type != "census":
        return {"leaf": record_type, "path": path}

    raw3 = loader._run_generate(image, TIER3_PROMPT)
    f3 = parse_fields(raw3)
    year = f3.get("census_year", "<unparsed>")
    conf3 = parse_confidence(f3)
    path.append(("tier3_census_year", year, conf3, raw3))
    if conf3 is None or conf3 < MIN_CONFIDENCE:
        return {"leaf": "quarantine_low_confidence", "path": path}
    return {"leaf": f"census_{year}", "path": path}


def main():
    print("Loading gemma (config/models/gemma.yaml, same config the bucket classifier uses)...")
    model_cfg = load_model_config("gemma")
    model_cfg.prompt_text = ""  # unused - this script bypasses _build_prompt() entirely
    loader_cls = LOADER_REGISTRY.get(model_cfg.loader_class)
    loader = loader_cls(model_cfg)
    t0 = time.time()
    loader.initialize_model_and_tokenizer()
    print(f"Loaded in {time.time() - t0:.1f}s\n")

    correct = 0
    total = 0
    total_prompts_issued = 0
    results = []

    for stem, ground_truth in TEST_CASES:
        path_img = WORKING_DIR / f"{stem}.jpg"
        if not path_img.exists():
            print(f"SKIP {stem}: not found at {path_img}")
            continue

        image = Image.open(path_img).convert("RGB")
        t0 = time.time()
        result = route_one_image(loader, image)
        elapsed = time.time() - t0

        leaf = result["leaf"]
        depth = len(result["path"])
        total_prompts_issued += depth
        # A quarantine on the deliberately out-of-taxonomy case is the
        # CORRECT outcome now (that's the whole point of adding the
        # confidence field) - not a miss just because the leaf string
        # doesn't literally equal "unknown".
        is_correct = leaf == ground_truth or (
            ground_truth == "unknown" and leaf == "quarantine_low_confidence"
        )
        total += 1
        correct += int(is_correct)
        mark = "OK  " if is_correct else "MISS"

        print(f"[{mark}] {stem}  ({elapsed:.1f}s, {depth} prompt{'s' if depth != 1 else ''})")
        print(f"       ground_truth={ground_truth!r}  leaf={leaf!r}")
        for tier_name, value, conf, raw in result["path"]:
            print(f"       {tier_name}: {value!r}  confidence={conf}")
            print(f"           raw: {raw!r}")
        print()

        results.append({
            "stem": stem, "ground_truth": ground_truth, "leaf": leaf,
            "correct": is_correct, "depth": depth, "elapsed": elapsed,
            "path": result["path"],
        })

    print(f"=== {correct}/{total} correct ===")
    print(f"Average tree depth: {total_prompts_issued / total:.2f} prompts/image "
          f"(vs. 1 prompt/image for today's flat classifier)")

    # Failure breakdown - WHERE in the tree did each miss happen, since
    # an early wrong turn forecloses the right leaf entirely (the risk
    # flagged in the Phase 2 design doc's abstention discussion).
    misses = [r for r in results if not r["correct"]]
    if misses:
        print(f"\n=== {len(misses)} miss(es) - tier-by-tier breakdown ===")
        for r in misses:
            print(f"{r['stem']}: ground_truth={r['ground_truth']!r} leaf={r['leaf']!r}")
            for tier_name, value, conf, _ in r["path"]:
                print(f"    {tier_name} -> {value!r} (confidence={conf})")


if __name__ == "__main__":
    main()
