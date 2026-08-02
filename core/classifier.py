"""
Stage 2 of the pipeline: classify every image in the manifest and
route it to the appropriate bucket CSV.

Usage:
    python -m core.classifier data/manifest.csv

Behavior on failure (this is deliberate, not a bug to silence):
  - If the loader raises during classification (malformed output,
    unrecognized category, missing fields), the file is routed to
    uncertain_review.csv with the error recorded, NOT retried with a
    looser parse. A classification failure should surface, not be
    papered over.
  - If confidence is below pipeline.yaml's min_confidence threshold,
    the file is routed to uncertain_review.csv even if the category
    parsed cleanly.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from core.loaders.base_loader import load_model_config
from core.loader_registry import LOADER_REGISTRY
from core.schema import DocumentCategory, ClassificationResult

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUCKET_DIR = PROJECT_ROOT / "data" / "buckets"

CSV_FIELDS = [
    "file_path", "category", "confidence", "text_density",
    "handwriting", "table_layout", "faces", "map_like",
    "reason", "model", "prompt_version",
]

UNCERTAIN_FIELDS = CSV_FIELDS + ["error"]


def load_pipeline_config() -> dict:
    with open(PROJECT_ROOT / "config" / "pipeline.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_classifier_loader(pipeline_cfg: dict) -> GemmaLoader:
    model_name = pipeline_cfg["classifier"]["model"]
    model_cfg = load_model_config(model_name)

    prompt_path = PROJECT_ROOT / pipeline_cfg["classifier"]["prompt_file"]
    model_cfg.prompt_text = prompt_path.read_text(encoding="utf-8")

    loader_cls = LOADER_REGISTRY.get(model_cfg.loader_class)
    if loader_cls is None:
        raise ValueError(
            f"No loader registered for loader_class={model_cfg.loader_class!r}. "
            f"Known loaders: {list(LOADER_REGISTRY.keys())}"
        )

    loader = loader_cls(model_cfg)
    loader.initialize_model_and_tokenizer()
    return loader


def open_bucket_writers() -> dict[str, tuple[csv.DictWriter, Any]]:
    BUCKET_DIR.mkdir(parents=True, exist_ok=True)
    writers = {}
    for category in DocumentCategory:
        path = BUCKET_DIR / f"{category.value}.csv"
        is_new = not path.exists()
        f = open(path, "a", newline="", encoding="utf-8")
        fields = UNCERTAIN_FIELDS if category == DocumentCategory.UNCERTAIN else CSV_FIELDS
        writer = csv.DictWriter(f, fieldnames=fields)
        if is_new:
            writer.writeheader()
        writers[category.value] = (writer, f)
    return writers


def result_to_row(result: ClassificationResult) -> dict:
    row = result.model_dump()
    row["category"] = result.category.value
    return row


def _enable_raw_output_debug(loader) -> None:
    """
    Wraps loader._run_generate so --debug prints each call's raw model
    text to stdout before core/loaders/base_loader.py's classify()
    parses it into a ClassificationResult. Deliberately does NOT touch
    any file under core/loaders/ - added 2026-07-30 while a real
    classification batch was actively running, and CLAUDE.md's rule
    ("ask before editing loader code while a run is in progress") is
    specifically about that directory; wrapping at this call site
    instead means the flag never needs to touch it, live run or not.

    Assigns a plain function to the INSTANCE (not the class) - Python's
    descriptor protocol only auto-binds `self` for methods looked up on
    the class, so an instance attribute like this is called with
    exactly the two args it's defined to take (raw_image, prompt), no
    `self` involved. Standard, safe pattern for patching one instance
    without touching the class/module it came from.
    """
    original_run_generate = loader._run_generate

    def _debug_run_generate(raw_image, prompt):
        raw_output = original_run_generate(raw_image, prompt)
        print(f"  [raw model output]\n{raw_output}\n  [end raw output]")
        return raw_output

    loader._run_generate = _debug_run_generate


def run(manifest_path: Path, debug: bool = False) -> None:
    pipeline_cfg = load_pipeline_config()
    min_confidence = pipeline_cfg["classifier"]["min_confidence"]

    loader = build_classifier_loader(pipeline_cfg)
    if debug:
        _enable_raw_output_debug(loader)
    writers = open_bucket_writers()

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Classifying {len(rows)} images...")

    for i, row in enumerate(rows, 1):
        file_path = row["file_path"]
        print(f"[{i}/{len(rows)}] {file_path}")

        try:
            with Image.open(file_path) as raw_image:
                result = loader.classify(file_path, raw_image)
        except Exception as e:
            writer, _ = writers[DocumentCategory.UNCERTAIN.value]
            writer.writerow({
                "file_path": file_path,
                "category": "",
                "confidence": "",
                "text_density": "", "handwriting": "", "table_layout": "",
                "faces": "", "map_like": "", "reason": "",
                "model": loader.config.model_name,
                "prompt_version": loader.config.prompt_version,
                "error": str(e)[:300],
            })
            print(f"  -> uncertain_review (error: {e})")
            continue

        target_category = result.category
        if result.confidence < min_confidence:
            target_category = DocumentCategory.UNCERTAIN
            print(f"  -> uncertain_review (low confidence: {result.confidence:.2f}, "
                  f"originally classified as {result.category.value})")
        else:
            print(f"  -> {target_category.value} (confidence: {result.confidence:.2f})")

        writer, _ = writers[target_category.value]
        row_out = result_to_row(result)
        if target_category == DocumentCategory.UNCERTAIN:
            row_out["error"] = ""
        writer.writerow(row_out)

    for _, f in writers.values():
        f.close()

    print(f"\nDone. Bucket CSVs written to {BUCKET_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2: classify every image in the manifest, route into bucket CSVs.")
    parser.add_argument("manifest", help="Path to manifest.csv")
    parser.add_argument(
        "--debug", action="store_true",
        help="Print each image's raw model output to stdout before it's parsed.",
    )
    args = parser.parse_args()
    run(Path(args.manifest), debug=args.debug)


if __name__ == "__main__":
    main()
