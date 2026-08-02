"""
A/B comparison of Stage 2 routing prompts (classifier_classify_v1 vs
v2), loading Gemma ONCE and running every prompt against every test
image in a single process - same "one load, many prompts" pattern as
diagnostics/compare_document_subtype_prompts.py.

Built 2026-07-30 to test a specific fix: v2 adds map-vs-grid guidance
so a township/section survey plat (R22W-064.jpg) routes to
map_land_record instead of dense_tabular_rows (user-flagged, "this one
needs a stage 1 fix though"). The HBC post-report screenshot
(Screenshot 2026-05-05 233029.png) is a DIFFERENT misclassification the
user explicitly said belongs to the semantic subtype stage, not here -
included as a case that must NOT change between v1 and v2.

Test set spans all 8 buckets currently populated with real files, for
general stability/regression coverage of v2's prompt changes.

Usage:
    python diagnostics/compare_classify_prompts.py
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
from core.loaders.gemma_loader import GemmaLoader

PROMPTS_DIR = PROJECT_ROOT / "config" / "prompts"
WORKING = PROJECT_ROOT / "data" / "working"

PROMPTS = {
    "v1": PROMPTS_DIR / "classifier_classify_v1.txt",
    "v2": PROMPTS_DIR / "classifier_classify_v2.txt",
}

# (path, ground_truth, group)
TEST_CASES = [
    # The actual fix target: map misrouted to dense_tabular_rows under v1.
    (WORKING / "R22W-064.jpg", "map_land_record", "FIX_TARGET"),

    # Must NOT change - user said this is a semantic-stage problem, not
    # a Stage 2 bug. Both v1 and v2 are expected to route it the same
    # way (dense_tabular_rows), even though that's not its "true" type.
    (WORKING / "Screenshot 2026-05-05 233029.png", "dense_tabular_rows", "MUST_NOT_CHANGE"),

    # Known-correct dense_tabular_rows (census/manifest) - regression check.
    (WORKING / "1921_158-E003219273.jpg", "dense_tabular_rows", "REGRESSION_TABULAR"),
    (WORKING / "e003558130.jpg", "dense_tabular_rows", "REGRESSION_TABULAR"),
    (WORKING / "CANIMM1913PLIST_2000908421-00487.jpg", "dense_tabular_rows", "REGRESSION_TABULAR"),

    # Known-correct map_land_record already - v2's strengthened map
    # language must not cause drift elsewhere.
]

MAP_BUCKET = PROJECT_ROOT / "data" / "buckets" / "map_land_record.csv"
PRINTED_BUCKET = PROJECT_ROOT / "data" / "buckets" / "printed_document.csv"
PORTRAIT_BUCKET = PROJECT_ROOT / "data" / "buckets" / "portrait_photo.csv"
MIXED_BUCKET = PROJECT_ROOT / "data" / "buckets" / "mixed_text_image.csv"
CHART_BUCKET = PROJECT_ROOT / "data" / "buckets" / "genealogy_chart.csv"
LEDGER_BUCKET = PROJECT_ROOT / "data" / "buckets" / "handwritten_ledger.csv"


def _first_n_paths(csv_path: Path, n: int) -> list[Path]:
    import csv
    if not csv_path.exists():
        return []
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [Path(r["file_path"]) for r in rows[:n] if Path(r["file_path"]).exists()]


def _extend_test_cases() -> None:
    for p in _first_n_paths(MAP_BUCKET, 3):
        if p.name != "R22W-064.jpg":
            TEST_CASES.append((p, "map_land_record", "REGRESSION_MAP"))
    for p in _first_n_paths(PRINTED_BUCKET, 2):
        TEST_CASES.append((p, "printed_document", "REGRESSION_OTHER"))
    for p in _first_n_paths(PORTRAIT_BUCKET, 2):
        TEST_CASES.append((p, "portrait_photo", "REGRESSION_OTHER"))
    for p in _first_n_paths(MIXED_BUCKET, 2):
        TEST_CASES.append((p, "mixed_text_image", "REGRESSION_OTHER"))
    for p in _first_n_paths(CHART_BUCKET, 2):
        TEST_CASES.append((p, "genealogy_chart", "REGRESSION_OTHER"))
    for p in _first_n_paths(LEDGER_BUCKET, 1):
        TEST_CASES.append((p, "handwritten_ledger", "REGRESSION_OTHER"))


def main() -> None:
    _extend_test_cases()
    cases = [(p, gt, g) for p, gt, g in TEST_CASES if p.exists()]
    missing = [str(p) for p, _, _ in TEST_CASES if not p.exists()]
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
            f = GemmaLoader._parse_kv_block(raw)
            got = f.get("category", "<no field>")
            results[key][str(path)] = {
                "got": got, "truth": truth, "group": group,
                "confidence": f.get("confidence", ""),
                "reason": f.get("reason", ""),
            }
            mark = "OK " if got == truth else "MISS"
            print(f"    {key}: {mark} {got:<20} conf={f.get('confidence', '?'):<6} "
                  f"reason={f.get('reason', '')[:70]!r}")
        print()

    print("=" * 78)
    groups = sorted({g for _, _, g in cases})
    print(f"{'group':<22} " + "  ".join(f"{k:>10}" for k in PROMPTS))
    for g in groups:
        line = f"{g:<22} "
        for key in PROMPTS:
            rows = [r for r in results[key].values() if r["group"] == g]
            if not rows:
                line += f"{'-':>10}  "
                continue
            ok = sum(1 for r in rows if r["got"] == r["truth"])
            line += f"{f'{ok}/{len(rows)}':>10}  "
        print(line)
    line = f"{'TOTAL':<22} "
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
            print(f"  {Path(path).name:<44} truth={truth:<20} {marks}")
    if not any_diff:
        print("  (none)")
    else:
        print("  (* = matches ground truth)")


if __name__ == "__main__":
    main()
