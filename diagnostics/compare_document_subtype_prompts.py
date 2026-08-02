"""
A/B comparison of document-subtype classifier prompts (v2 vs v3),
loading Gemma ONCE and running every prompt against every test image in
a single process - same "one load, many prompts" pattern
core/semantic_stages.py already uses, and the same reason
diagnostics/test_gemma_document_subtype.py gives for doing it this way
(matches how it really runs in production, and CLAUDE.md's GPU-
contention rule means one process, not one per prompt).

Built 2026-07-30 to answer a specific, diagnosed failure rather than to
tune blind. On the 16-page batch, Gemma returned `unknown` for 3 real
1911 census pages. Reading their `title_text_read` showed it had read
the CENTRE heading each time ("SCHEDULE No. 16 Population", "SCHEDULE
TABLE II. No. 213 ..."), which on these forms contains neither the year
nor the word "Canada" - while the prompt's CRITICAL RULE requires
"Canada" before any canada_census_* answer. So it abstained correctly
from the wrong piece of text.

Verified by eye on real pages (2026-07-30), consistent across ALL THREE
census years: the identifying text is printed small in the TOP CORNERS,
bilingual -
    top-left   "<ORDINAL> CENSUS OF CANADA, <YEAR>"
    top-right  "<ORDINAL> RECENSEMENT DU CANADA, <YEAR>"
    centre     "DOMINION BUREAU OF STATISTICS"/"POPULATION" (1921/1931)
               or "SCHEDULE No. N" (1911) - no year, no country.
v3 adds that location guidance plus the ordinal->year cross-check
(FIFTH=1911, SIXTH=1921, SEVENTH=1931).

GROUND TRUTH BELOW WAS CONFIRMED BY DIRECT VISUAL INSPECTION of each
page's printed corner title in this session - not assumed, not carried
over from a previous classifier's output. The three FORMERLY_UNKNOWN
cases and the five NEW_1921_1931 cases were each rendered and read.

Usage:
    python diagnostics/compare_document_subtype_prompts.py
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

PROMPTS_DIR = PROJECT_ROOT / "config" / "prompts"
DEWARPED = PROJECT_ROOT / "data" / "outputs" / "dewarped"
WORKING = PROJECT_ROOT / "data" / "working"
NEW_CENSUS = Path(r"J:\Screenshots\Census")

PROMPTS = {
    "v2": PROMPTS_DIR / "classifier_document_subtype_v2.txt",
    "v3": PROMPTS_DIR / "classifier_document_subtype_v3.txt",
    "v4": PROMPTS_DIR / "classifier_document_subtype_v4.txt",
}

# (path, ground_truth, group). Group labels let the summary report
# regression separately from the cases v3 is meant to fix - a prompt
# that fixes 3 pages and breaks 5 others is not an improvement.
TEST_CASES = [
    # The 3 real 1911 pages v2 returned "unknown" for - the target.
    (WORKING / "e001946615.png", "canada_census_1911", "FORMERLY_UNKNOWN"),
    (WORKING / "e001946620.png", "canada_census_1911", "FORMERLY_UNKNOWN"),
    (WORKING / "e002101688.png", "canada_census_1911", "FORMERLY_UNKNOWN"),

    # First real 1921/1931 samples (J:\Screenshots\Census, raw LAC).
    (NEW_CENSUS / "e002879426.jpg", "canada_census_1921", "NEW_1921_1931"),
    (NEW_CENSUS / "e002879427.jpg", "canada_census_1921", "NEW_1921_1931"),
    (NEW_CENSUS / "e002879428.jpg", "canada_census_1921", "NEW_1921_1931"),
    (NEW_CENSUS / "e011670449.jpg", "canada_census_1931", "NEW_1921_1931"),
    (NEW_CENSUS / "e011670457.jpg", "canada_census_1931", "NEW_1921_1931"),

    # Regression set - diagnostics/test_gemma_document_subtype.py's own
    # TEST_CASES, which v2 already passes 9/10. v3 must not break these.
    (DEWARPED / "1921_022-E002880409_dewarped.jpg", "canada_census_1921", "REGRESSION"),
    (DEWARPED / "1931_174-e011707164_dewarped.jpg", "canada_census_1931", "REGRESSION"),
    (DEWARPED / "e078_e001946617_dewarped.jpg", "canada_census_1911", "REGRESSION"),
    (DEWARPED / "30807_A000676-00099(1)_dewarped.jpg", "printed_manifest", "REGRESSION"),
    (DEWARPED / "CANIMM1913PLIST_2000908421-00487_dewarped.jpg", "handwritten_manifest", "REGRESSION"),
    (DEWARPED / "e003559198_dewarped.jpg", "handwritten_manifest", "REGRESSION"),
    (DEWARPED / "e003558130_dewarped.jpg", "handwritten_manifest", "REGRESSION"),
    (DEWARPED / "IMCANQC1865_T4821-00622_dewarped.jpg", "handwritten_manifest", "REGRESSION"),
    (DEWARPED / "e003566165_dewarped.jpg", "handwritten_manifest", "REGRESSION"),
    (DEWARPED / "S3HY-6LL2-RG_dewarped.jpg", "unknown", "REGRESSION"),

    # The UK page v2 was specifically built to stop misclassifying as
    # canada_census_1911 (see run_semantic_stages.py's own comment). The
    # single most important non-regression: v3 tells the model WHERE to
    # find "Canada", which must not become a licence to assume it.
    (DEWARPED / "WARRG13_2948_2949-0078_dewarped.jpg", "unknown", "UK_GUARD"),
]


def parse_fields(raw: str) -> dict[str, str]:
    out = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        k = k.strip().lstrip("-*").strip().lower()
        if k:
            out[k] = v.strip()
    return out


def main() -> None:
    cases = [(p, gt, g) for p, gt, g in TEST_CASES if p.exists()]
    missing = [p.name for p, _, _ in TEST_CASES if not p.exists()]
    if missing:
        print(f"WARNING: {len(missing)} test image(s) not found, skipped: {missing}\n")

    print("Loading gemma (config/models/gemma.yaml)...")
    cfg = load_model_config("gemma")
    loader = LOADER_REGISTRY.get(cfg.loader_class)(cfg)
    t0 = time.time()
    loader.initialize_model_and_tokenizer()
    print(f"Loaded in {time.time() - t0:.1f}s. {len(cases)} image(s) x {len(PROMPTS)} prompt(s).\n")

    prompt_text = {k: p.read_text(encoding="utf-8") for k, p in PROMPTS.items()}
    results: dict[str, dict[str, dict]] = {k: {} for k in PROMPTS}

    for i, (path, truth, group) in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {path.name}  (truth: {truth}, {group})")
        with Image.open(path) as im:
            image = im.convert("RGB")
        for key in PROMPTS:
            raw = loader._run_generate(image, prompt_text[key])
            f = parse_fields(raw)
            got = f.get("document_type", "<no field>")
            results[key][str(path)] = {
                "got": got, "truth": truth, "group": group,
                "confidence": f.get("confidence", ""),
                "title": f.get("title_text_read", ""),
                "reason": f.get("reason", ""),
            }
            mark = "OK " if got == truth else "MISS"
            print(f"    {key}: {mark} {got:<22} conf={f.get('confidence','?'):<6} "
                  f"title={f.get('title_text_read','')[:60]!r}")
        print()

    print("=" * 78)
    groups = ["FORMERLY_UNKNOWN", "NEW_1921_1931", "REGRESSION", "UK_GUARD"]
    print(f"{'group':<20} " + "  ".join(f"{k:>10}" for k in PROMPTS))
    for g in groups:
        line = f"{g:<20} "
        for key in PROMPTS:
            rows = [r for r in results[key].values() if r["group"] == g]
            if not rows:
                line += f"{'-':>10}  "
                continue
            ok = sum(1 for r in rows if r["got"] == r["truth"])
            line += f"{f'{ok}/{len(rows)}':>10}  "
        print(line)
    line = f"{'TOTAL':<20} "
    for key in PROMPTS:
        rows = list(results[key].values())
        ok = sum(1 for r in rows if r["got"] == r["truth"])
        line += f"{f'{ok}/{len(rows)}':>10}  "
    print(line)

    keys = list(PROMPTS)
    print(f"\nPer-image results where the prompts disagree ({' vs '.join(keys)}):")
    any_diff = False
    for path in results[keys[0]]:
        got = [results[k][path]["got"] for k in keys]
        if len(set(got)) > 1:
            any_diff = True
            truth = results[keys[0]][path]["truth"]
            marks = "  ".join(f"{k}={g}{'*' if g == truth else ''}" for k, g in zip(keys, got))
            print(f"  {Path(path).name:<44} truth={truth:<22} {marks}")
    if not any_diff:
        print("  (none)")
    else:
        print("  (* = matches ground truth)")

    # Confidence spread - v2's was pinned at 0.95 on 15/16 real rows,
    # including its "unknown"s, i.e. carrying no information at all.
    print("\nConfidence spread:")
    for key in PROMPTS:
        vals = []
        for r in results[key].values():
            try:
                vals.append(float(r["confidence"]))
            except (TypeError, ValueError):
                pass
        if vals:
            print(f"  {key}: {len(set(vals))} distinct value(s) over {len(vals)} rows, "
                  f"min {min(vals)}, max {max(vals)}")



if __name__ == "__main__":
    main()
