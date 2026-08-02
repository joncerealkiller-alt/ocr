"""
Smoke test for deepseek-vl2-tiny (2026-07-28 prep task) - loads the
model ONCE via the real loader/subprocess-venv path (core/loaders/
deepseek_vl2_loader.py -> .venv_deepseek_vl2) and runs it against a
handful of real project sample images, so Jon can pick up real
evaluation from here. Not wired into config/pipeline.yaml, does not
touch any bucket CSV or extracted.csv - same isolation guarantee as
model_assessment.py.

First-run note: initializing the model spawns the .venv_deepseek_vl2
worker subprocess, which itself imports the vendored DeepSeek-VL2-src
package - see core/loaders/_deepseek_vl2_worker.py's module docstring
for the full setup story and what it depends on already existing on
disk (the venv, the git clone, the patched tokenizers pin).

Usage:
    python diagnostics/test_deepseek_vl2_smoke.py
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

# A small, varied sample - not the full dataset (that's explicitly
# Jon's own thorough-testing pass, not this prep task's job).
TEST_STEMS = [
    "30807_A000676-00099(1)",   # printed manifest
    "e078_e001946617",          # 1911 census
    "e003558130",                # handwritten manifest
]


def main():
    print("Loading deepseek_vl2_tiny (config/models/deepseek_vl2_tiny.yaml)...")
    model_cfg = load_model_config("deepseek_vl2_tiny")
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

    loader.release()


if __name__ == "__main__":
    main()
