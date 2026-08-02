"""
One-off test: can Gemma (the SAME model/config already used for bucket
routing, config/models/gemma.yaml) identify the SPECIFIC document
sub-type (census year, printed vs. handwritten manifest) directly from
the image, as a VLM alternative/complement to core/document_
classification.py's rule-based CV classifier?

Built 2026-07-27 per Jon's direction - loads the model ONCE and runs it
against every test image in one process, matching how this would
actually run in production (folded into the same model load as the
existing bucket classifier, not a separate load/unload pass).

Deliberately calls loader._run_generate() directly rather than
loader.classify() - classify()'s parsing (_parse_classification) is
built for classifier_classify_v1.txt's bucket-routing schema
(category/text_density/handwriting/etc.), not this prompt's different
field set. This script does its own lightweight parsing instead.

Ground truth labels below come from this session's own manual
verification (real pixel-level inspection + visual confirmation) of
each file, not assumed.

Usage:
    python diagnostics/test_gemma_document_subtype.py
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

DEWARPED_DIR = PROJECT_ROOT / "data" / "outputs" / "dewarped"
PROMPT_PATH = PROJECT_ROOT / "config" / "prompts" / "classifier_document_subtype_v2.txt"

# (filename stem, ground truth doc_type) - "unknown" for the cabin-class
# manifest cluster since it's not in the prompt's taxonomy at all yet;
# included specifically to see what Gemma does with an out-of-list case.
TEST_CASES = [
    ("1921_022-E002880409", "canada_census_1921"),
    ("1931_174-e011707164", "canada_census_1931"),
    ("e078_e001946617", "canada_census_1911"),
    ("30807_A000676-00099(1)", "printed_manifest"),
    ("CANIMM1913PLIST_2000908421-00487", "handwritten_manifest"),
    ("e003559198", "handwritten_manifest"),
    ("e003558130", "handwritten_manifest"),
    ("IMCANQC1865_T4821-00622", "handwritten_manifest"),
    ("e003566165", "handwritten_manifest"),  # the CV classifier's own known miss
    ("S3HY-6LL2-RG", "unknown"),  # portrait Cabin-class manifest - out of taxonomy
]


def parse_fields(raw_output: str) -> dict[str, str]:
    fields = {}
    for line in raw_output.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lstrip("-*").strip().lower()
        if key:
            fields[key] = value.strip()
    return fields


def main():
    print("Loading gemma (config/models/gemma.yaml, same config the bucket classifier uses)...")
    model_cfg = load_model_config("gemma")
    model_cfg.prompt_text = PROMPT_PATH.read_text(encoding="utf-8")

    loader_cls = LOADER_REGISTRY.get(model_cfg.loader_class)
    loader = loader_cls(model_cfg)
    t0 = time.time()
    loader.initialize_model_and_tokenizer()
    print(f"Loaded in {time.time() - t0:.1f}s\n")

    correct = 0
    total = 0
    for stem, ground_truth in TEST_CASES:
        path = DEWARPED_DIR / f"{stem}_dewarped.jpg"
        if not path.exists():
            print(f"SKIP {stem}: dewarped file not found at {path}")
            continue

        image = Image.open(path).convert("RGB")
        t0 = time.time()
        raw = loader._run_generate(image, model_cfg.prompt_text)
        elapsed = time.time() - t0

        fields = parse_fields(raw)
        predicted = fields.get("document_type", "<missing>")
        confidence = fields.get("confidence", "?")
        title_text = fields.get("title_text_read", "?")
        reason = fields.get("reason", "?")

        is_correct = predicted == ground_truth
        total += 1
        correct += int(is_correct)
        mark = "OK  " if is_correct else "MISS"

        print(f"[{mark}] {stem}")
        print(f"       ground_truth={ground_truth!r}  predicted={predicted!r}  confidence={confidence}  ({elapsed:.1f}s)")
        print(f"       title_text_read: {title_text}")
        print(f"       reason: {reason}")
        if not fields:
            print(f"       RAW (unparsed): {raw[:300]!r}")
        print()

    print(f"=== {correct}/{total} correct ===")


if __name__ == "__main__":
    main()
