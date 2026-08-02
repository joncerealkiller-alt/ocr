"""
Smoke test for LiquidAI/LFM2-VL-1.6B (2026-07-28 prep task) - loads the
model ONCE via the real loader path (core/loaders/lfm2_vl_loader.py,
in-process, no subprocess venv needed) and runs it against a handful of
real project sample images, so Jon can pick up real evaluation from
here. Not wired into config/pipeline.yaml, does not touch any bucket
CSV or extracted.csv - same isolation guarantee as model_assessment.py.

Known finding from the initial sanity check (see config/models/
lfm2_vl_1_6b.yaml's own comment): on the one sample tried, this model
got the real document type and date right and transcribed real column
headers, but then confidently invented a long tail of fields not
present on the document at all - fabrication, not a hedge. Worth
watching for on every sample, not just assuming it was a one-off.

Usage:
    python diagnostics/test_lfm2_vl_smoke.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
from pydantic import ValidationError

from core.loaders.base_loader import load_model_config
from core.loader_registry import LOADER_REGISTRY

DEWARPED_DIR = PROJECT_ROOT / "data" / "outputs" / "dewarped"
PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "extractor_printed_v2.txt"

TEST_STEMS = [
    "30807_A000676-00099(1)",   # printed manifest
    "e078_e001946617",          # 1911 census
    "e003558130",                # handwritten manifest
]


def main():
    print("Loading lfm2_vl_1_6b (config/models/lfm2_vl_1_6b.yaml)...")
    model_cfg = load_model_config("lfm2_vl_1_6b")
    model_cfg.prompt_text = PROMPT_PATH.read_text(encoding="utf-8")

    loader_cls = LOADER_REGISTRY.get(model_cfg.loader_class)
    loader = loader_cls(model_cfg)
    t0 = time.time()
    loader.initialize_model_and_tokenizer()
    print(f"Loaded in {time.time() - t0:.1f}s\n")

    for stem in TEST_STEMS:
        path = DEWARPED_DIR / f"{stem}_dewarped.jpg"
        if not path.exists():
            print(f"SKIP {stem}: dewarped file not found at {path}")
            continue

        with Image.open(path) as image:
            image = image.convert("RGB")
            t0 = time.time()
            raw = loader._run_generate(image, model_cfg.prompt_text)
            elapsed = time.time() - t0

        print(f"=== {stem} ({elapsed:.1f}s) ===")
        print(raw)
        try:
            result = loader._parse_extraction(str(path), "dense_tabular_rows", raw)
            print(f"[schema PASS] {result.model_dump_json(indent=2)}")
        except (ValueError, ValidationError) as e:
            print(f"[schema FAIL] {e}")
        print()


if __name__ == "__main__":
    main()
