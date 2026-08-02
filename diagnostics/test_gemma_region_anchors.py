"""
One-off test: can Gemma produce usable APPROXIMATE region anchors
(metadata_bbox/header_bbox/table_bbox) - Jon's Phase 4 idea - reliably
enough to be worth building out, or should that stay pure CV?

Built 2026-07-27. Loads the model once (matching the "stay resident in
VRAM" design from Phase 1-3) and tests TWO image_token_budget levels
per sample: 140 (the budget the existing bucket/subtype classifier
already uses - if this works at 140, Phase 4 can ride the exact same
low-cost pass) and 560 (a heavier budget - if 140 fails but 560 works,
that's real information: bbox grounding needs more visual detail than
content classification does, and Phase 4 would need its own separate,
more expensive call, not a free addition to the existing pass).

Ground truth bboxes below come from THIS SESSION's own CV pipeline
output (core/auto_sidecar.py), each one visually verified against the
real page earlier in this session (not assumed) - see data/outputs/
auto_row_segmentation/*_sidecar.json for the source values.

Usage:
    python diagnostics/test_gemma_region_anchors.py
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

DEWARPED_DIR = PROJECT_ROOT / "data" / "outputs" / "dewarped"
PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "classifier_region_anchors_v1.txt"
TOLERANCE_PX = 40  # Jon's stated acceptable tolerance, upper end (~20-40px)

# (filename stem, ground truth bboxes) - from this session's own
# CV-verified sidecar output, NOT re-derived here.
TEST_CASES = [
    ("1921_022-E002880409", {
        "metadata_bbox": [0, 0, 3244, 322],
        "header_bbox": [22, 318, 3203, 646],
        "table_bbox": [22, 642, 3203, 2066],
    }),
    ("1931_174-e011707164", {
        "metadata_bbox": [0, 0, 3988, 182],
        "header_bbox": [10, 178, 3926, 571],
        "table_bbox": [10, 567, 3926, 2291],
    }),
    ("30807_A000676-00099(1)", {
        "metadata_bbox": [0, 0, 1604, 163],
        "header_bbox": [3, 159, 1599, 361],
        "table_bbox": [3, 357, 1599, 1818],
    }),
]

BUDGETS_TO_TEST = [140, 560]


def parse_bbox_fields(raw_output: str) -> dict:
    fields = {}
    for line in raw_output.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lstrip("-*").strip().lower()
        value = value.strip()
        if key in ("metadata_bbox", "header_bbox", "table_bbox"):
            nums = [int(n) for n in re.findall(r"-?\d+", value)]
            if len(nums) == 4:
                fields[key] = nums
        elif key:
            fields[key] = value
    return fields


def bbox_error(predicted: list[int] | None, ground_truth: list[int]) -> str:
    if predicted is None:
        return "MISSING"
    diffs = [abs(p - g) for p, g in zip(predicted, ground_truth)]
    max_diff = max(diffs)
    verdict = "OK" if max_diff <= TOLERANCE_PX else "OUT OF TOLERANCE"
    return f"predicted={predicted} truth={ground_truth} per-edge-error={diffs} max={max_diff}px [{verdict}]"


def main():
    print("Loading gemma (config/models/gemma.yaml)...")
    model_cfg = load_model_config("gemma")
    base_prompt = PROMPT_PATH.read_text(encoding="utf-8")

    loader_cls = LOADER_REGISTRY.get(model_cfg.loader_class)
    loader = loader_cls(model_cfg)
    t0 = time.time()
    loader.initialize_model_and_tokenizer()
    print(f"Loaded in {time.time() - t0:.1f}s\n")

    for budget in BUDGETS_TO_TEST:
        loader.config.extra["image_token_budget"] = budget
        print(f"\n{'=' * 20} image_token_budget={budget} {'=' * 20}\n")

        for stem, truth in TEST_CASES:
            path = DEWARPED_DIR / f"{stem}_dewarped.jpg"
            if not path.exists():
                print(f"SKIP {stem}: not found")
                continue

            image = Image.open(path).convert("RGB")
            w, h = image.size
            prompt = base_prompt.format(width=w, height=h)

            t0 = time.time()
            raw = loader._run_generate(image, prompt)
            elapsed = time.time() - t0
            fields = parse_bbox_fields(raw)

            print(f"[{stem}] size={w}x{h}  ({elapsed:.1f}s)  confidence={fields.get('confidence', '?')}")
            for region in ("metadata_bbox", "header_bbox", "table_bbox"):
                print(f"  {region}: {bbox_error(fields.get(region), truth[region])}")
            print(f"  reason: {fields.get('reason', '?')}")
            if not fields:
                print(f"  RAW (unparsed): {raw[:300]!r}")
            print()


if __name__ == "__main__":
    main()
