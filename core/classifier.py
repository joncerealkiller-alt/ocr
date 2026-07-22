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

import csv
import sys
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


def run(manifest_path: Path) -> None:
    pipeline_cfg = load_pipeline_config()
    min_confidence = pipeline_cfg["classifier"]["min_confidence"]

    loader = build_classifier_loader(pipeline_cfg)
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


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m core.classifier <manifest.csv>")
        sys.exit(1)
    run(Path(sys.argv[1]))
