"""
Single-shot classifier prompt test (2026-08-06) - Jon's own prompt,
developed and hand-tested on-device (LM Studio-equivalent mobile app,
Gemma-4-E2B-it) against 9 real/adversarial images, then confirmed clean
at temperature 0 (matches this project's do_sample=false everywhere).
Notable phone results before this script existed: correctly read BOTH
a 1906 and a 1931 census year from header text in a single call - 1931
is the exact year our own 4-tier gated-binary tree (diagnostics/
test_gemma_prompt_tiering_variants.py's v7, adopted 2026-08-06) got
WRONG in every one of 9 variants tested (always predicted 1911 instead).

This script formalizes that phone testing: SAME prompt, SAME model
config (config/models/gemma.yaml, do_sample=false already), run via the
real GemmaLoader._run_generate() path (system prompt, charset mask,
image_token_budget all applied exactly as production does) against:

  1. The same 10-image ground-truth set used throughout the gated-tree
     work (diagnostics/test_gemma_prompt_tiering.py's TEST_CASES) - if
     this single flat prompt handles that set "ok" (Jon's own bar - see
     PASS_THRESHOLD below, set to match-or-beat the gated tree's known
     7/10), the script proceeds automatically to:
  2. The newly-acquired LAC (Library and Archives Canada) pulls under
     data/outputs/lac_pull_*_batch1/ - five of these are already split
     into per-year folders (1901/1906/1921/1926/1931), which IS ground
     truth for those (folder name = real census year, category = census
     for all of them since these are census-specific pulls). data/
     outputs/lac_new_years_samples/ has no folder-encoded ground truth -
     logged for Jon's manual review, not auto-scored.

Does NOT modify core/loaders/gemma_loader.py, core/classifier.py, or
any prompt/config file - standalone, same "don't touch loader code
without a real reason" discipline as every other diagnostic this
session.

Usage:
    python diagnostics/test_gemma_single_shot_classifier.py
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from core.loaders.base_loader import load_model_config
from core.loader_registry import LOADER_REGISTRY
from diagnostics.test_gemma_prompt_tiering import TEST_CASES, WORKING_DIR

OUT_DIR = PROJECT_ROOT / "data" / "outputs" / "single_shot_classifier_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT_DIR / "log.txt"

# Jon's own bar for "passes ok" before expanding to the LAC pulls -
# match-or-beat v7's known 7/10 on this same 10-image set, not a new
# arbitrary threshold.
PASS_THRESHOLD = 7

SINGLE_SHOT_PROMPT = """You are a document routing classifier.

Your ONLY task is to classify the image into exactly ONE of these categories:

- Census pages
- Passenger Manifests
- Unknown

Do not transcribe, summarize, or describe the document beyond what is necessary for classification.

If there is insufficient evidence to distinguish between the categories, output Unknown.
Do not guess.

If a year is explicitly visible in the document header, output it.
Do not infer or estimate the year.

Output EXACTLY in the following format:

<confidence x.xx>
<Category>
<Year> (only if explicitly visible)

Reason: <maximum 60 words>

The confidence score represents your confidence in the selected category,
including Unknown if the image clearly does not belong to the supported document types."""

CATEGORY_TOKENS = ["census pages", "passenger manifests", "unknown"]

# Ground-truth category mapping from test_gemma_prompt_tiering.py's
# taxonomy (census_YYYY / passenger_manifest / unknown) onto THIS
# prompt's taxonomy (Census pages / Passenger Manifests / Unknown).
def gt_category(ground_truth: str) -> str:
    if ground_truth.startswith("census_"):
        return "census pages"
    if ground_truth == "passenger_manifest":
        return "passenger manifests"
    return "unknown"


def gt_year(ground_truth: str) -> str | None:
    if ground_truth.startswith("census_") and ground_truth != "census_unknown":
        year = ground_truth.split("_", 1)[1]
        return year if year.isdigit() else None
    return None


def log(f, msg: str) -> None:
    print(msg)
    print(msg, file=f)
    f.flush()


def parse_response(raw: str) -> dict:
    """Robust to the format variance seen in Jon's phone testing (single-
    line vs multi-line, occasional duplicated category token like "Census
    pages Unknown" under sampling - not expected at do_sample=false, but
    handled defensively anyway): confidence via regex anywhere in the
    text; category = whichever of the three valid tokens appears
    EARLIEST (leftmost) in the raw text, case-insensitive - naturally
    picks the intended first-mentioned category even if a duplicate
    trails it; year = first 1900-1999 4-digit number found anywhere."""
    conf_match = re.search(r"confidence[^\d]*([\d.]+)", raw, re.I)
    confidence = float(conf_match.group(1)) if conf_match else None

    lowered = raw.lower()
    best_idx = None
    category = None
    for token in CATEGORY_TOKENS:
        idx = lowered.find(token)
        if idx != -1 and (best_idx is None or idx < best_idx):
            best_idx = idx
            category = token

    year_match = re.search(r"\b(19\d{2})\b", raw)
    year = year_match.group(1) if year_match else None

    return {"confidence": confidence, "category": category, "year": year, "raw": raw}


def classify_one(loader, image) -> dict:
    raw = loader._run_generate(image, SINGLE_SHOT_PROMPT)
    return parse_response(raw)


def load_gemma():
    print("Loading gemma (config/models/gemma.yaml, do_sample already false)...")
    model_cfg = load_model_config("gemma")
    model_cfg.prompt_text = ""
    loader_cls = LOADER_REGISTRY.get(model_cfg.loader_class)
    loader = loader_cls(model_cfg)
    t0 = time.time()
    loader.initialize_model_and_tokenizer()
    print(f"Loaded in {time.time() - t0:.1f}s (do_sample={loader.config.do_sample})\n")
    return loader


def run_ground_truth_set(f, loader) -> tuple[int, int]:
    log(f, "=" * 70)
    log(f, "=== Phase 1: existing 10-image ground-truth set ===")
    log(f, "=" * 70)
    correct = 0
    total = 0
    for stem, ground_truth in TEST_CASES:
        img_path = WORKING_DIR / f"{stem}.jpg"
        if not img_path.exists():
            log(f, f"SKIP {stem}: not found")
            continue
        image = Image.open(img_path).convert("RGB")
        t0 = time.time()
        result = classify_one(loader, image)
        elapsed = time.time() - t0

        want_cat = gt_category(ground_truth)
        want_year = gt_year(ground_truth)
        cat_ok = result["category"] == want_cat
        year_ok = (want_year is None) or (result["year"] == want_year)
        ok = cat_ok and year_ok
        total += 1
        correct += int(ok)
        mark = "OK  " if ok else "MISS"
        log(f, f"[{mark}] {stem}  ({elapsed:.1f}s)  ground_truth=({want_cat!r}, year={want_year!r})  "
               f"got=({result['category']!r}, year={result['year']!r}, conf={result['confidence']})")
        log(f, f"       raw: {result['raw']!r}")
    log(f, f"\n-> {correct}/{total}\n")
    return correct, total


def run_lac_pulls(f, loader) -> None:
    log(f, "=" * 70)
    log(f, "=== Phase 2: LAC pulls (year-labeled folders = ground truth) ===")
    log(f, "=" * 70)

    year_folders = [
        ("1901", PROJECT_ROOT / "data/outputs/lac_pull_1901_batch1/raw_jpgs"),
        ("1906", PROJECT_ROOT / "data/outputs/lac_pull_1906_batch1/raw_jpgs"),
        ("1921", PROJECT_ROOT / "data/outputs/lac_pull_1921_batch1/images"),
        ("1926", PROJECT_ROOT / "data/outputs/lac_pull_1926_batch1/raw_jpgs"),
        ("1931", PROJECT_ROOT / "data/outputs/lac_pull_1931_batch1/raw_jpgs"),
    ]

    grand_cat_correct = 0
    grand_year_correct = 0
    grand_total = 0

    for year, folder in year_folders:
        if not folder.exists():
            log(f, f"SKIP {year}: {folder} not found")
            continue
        images = sorted(
            p for p in folder.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        log(f, f"\n--- {year} ({len(images)} images, folder={folder}) ---")
        cat_correct = 0
        year_correct = 0
        for img_path in images:
            try:
                image = Image.open(img_path).convert("RGB")
                t0 = time.time()
                result = classify_one(loader, image)
                elapsed = time.time() - t0
                cat_ok = result["category"] == "census pages"
                year_ok = result["year"] == year
                cat_correct += int(cat_ok)
                year_correct += int(year_ok)
                mark = "OK  " if (cat_ok and year_ok) else ("CAT " if cat_ok else "MISS")
                log(f, f"  [{mark}] {img_path.name}  ({elapsed:.1f}s)  "
                       f"got=({result['category']!r}, year={result['year']!r}, conf={result['confidence']})")
            except Exception as e:
                log(f, f"  [ERROR] {img_path.name}: {type(e).__name__}: {e}")
        total = len(images)
        log(f, f"  -> category correct: {cat_correct}/{total}   year correct: {year_correct}/{total}")
        grand_cat_correct += cat_correct
        grand_year_correct += year_correct
        grand_total += total

    log(f, f"\n=== LAC pulls TOTAL: category {grand_cat_correct}/{grand_total}, "
           f"year {grand_year_correct}/{grand_total} ===")

    # lac_new_years_samples - no folder-encoded ground truth, logged raw
    # for Jon's manual review only, not scored.
    samples_dir = PROJECT_ROOT / "data/outputs/lac_new_years_samples"
    if samples_dir.exists():
        log(f, f"\n--- lac_new_years_samples (no ground truth - for manual review) ---")
        images = sorted(
            p for p in samples_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        for img_path in images:
            try:
                image = Image.open(img_path).convert("RGB")
                result = classify_one(loader, image)
                log(f, f"  {img_path.name}: category={result['category']!r} year={result['year']!r} "
                       f"conf={result['confidence']}")
                log(f, f"       raw: {result['raw']!r}")
            except Exception as e:
                log(f, f"  [ERROR] {img_path.name}: {type(e).__name__}: {e}")


def main():
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        from datetime import datetime
        log(f, f"\n\n########## SINGLE-SHOT CLASSIFIER TEST - started {datetime.now().isoformat()} ##########")
        log(f, f"Prompt (verbatim, Jon's):\n{SINGLE_SHOT_PROMPT}\n")

        loader = load_gemma()
        correct, total = run_ground_truth_set(f, loader)

        if correct >= PASS_THRESHOLD:
            log(f, f"PASSED ({correct}/{total} >= threshold {PASS_THRESHOLD}) - proceeding to LAC pulls.\n")
            run_lac_pulls(f, loader)
        else:
            log(f, f"DID NOT MEET THRESHOLD ({correct}/{total} < {PASS_THRESHOLD}) - "
                   f"stopping here per Jon's instruction, NOT running against LAC pulls. "
                   f"Review the 10-image results above before deciding how to proceed.")

        log(f, f"\nDone. Full log: {LOG_PATH}")


if __name__ == "__main__":
    main()
