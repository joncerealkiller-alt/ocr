"""
Candidate Qualification smoke test for nanonets/Nanonets-OCR2-3B
(2026-07-28) - loads the model ONCE via the real loader path
(core/loaders/qwen_loader.py - no new loader needed, this is a
Qwen2.5-VL-3B-Instruct fine-tune) and runs it against a real project
sample image. Not wired into config/pipeline.yaml, does not touch any
bucket CSV or extracted.csv.

Purpose (see models.md's methodology note): verify load, measure VRAM/
load time, confirm basic prompt-following, detect catastrophic failure.
NOT a Production Evaluation - this uses a whole-page prompt against a
whole document image, which is not how this project's real pipeline
handles dense tabular documents (core/row_extraction.py). Read results
as Candidate Qualification only.

Usage:
    python diagnostics/test_nanonets_ocr2_smoke.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Windows console default codepage can't display arbitrary Unicode a model
# might emit - reconfigure rather than let a crash discard completed work.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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
    print("Loading nanonets_ocr2_3b (config/models/nanonets_ocr2_3b.yaml)...")
    model_cfg = load_model_config("nanonets_ocr2_3b")
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
