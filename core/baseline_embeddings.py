"""
Captures a pre-preprocessing baseline vision-tower embedding for each
image, per docs/BENCHMARK2_METADATA_LAYER_QUALIFICATION.md's Third/
Fourth extension design (Jon, 2026-07-29/30): one canonical embedding
per image, captured once - before deskew/contrast/autocontrast modify
it - so every corpus-relative comparison (nearest-neighbor, drift
detection, a future per-image preprocessing-profile decision) has a
stable, non-drifting reference point. All 8 encoders qualified in the
frozen Vision Qualification Battery v1.0 are captured per image (Jon,
2026-08-02 - matches the Multi-Tower Routing Audit's own architecture),
using core/vision_embeddings.py's already-validated embedding functions.

WHERE this runs: called from core/manifest_pipeline.py's
build_working_manifest_from_paths(), immediately after
copy_to_working_dir() and BEFORE preprocess_for_manifest() - i.e. on
the raw working copy, untouched by deskew/autocontrast. Deliberately
AFTER Source Expansion (a PDF page has already been rasterized by that
point) - rasterization is how the image came to exist at all, not a
"preprocessing" step in the deskew/contrast sense this baseline is
meant to precede.

Immediate purpose (2026-08-02, Jon): a real per-image datapoint, not
yet a decision rule - "the data can help decide what preprocessing
needs to be done per image" is Stage B (profile selection), which is
explicitly NOT STARTED yet (docs/PREPROCESSING_STAGE_NOTES.md). This
module only captures and persists; it does not decide anything.

Storage: one JSON file per manifest-building run (sidecar next to the
manifest, same pattern as core/source_expansion.py's provenance file),
not one file per image - 8 encoders x ~1700 images is measured at
34.6MB / 16.8 min compute for the FULL corpus at both pipeline stages
combined (see the design doc above), so a single consolidated file is
both simpler to consume and nowhere near a size that needs per-image
sidecars.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import timm
import torch
from PIL import Image

from core.vision_embeddings import QUALIFIED_ENCODERS, build_model_and_transform, embed_pooled

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE_PATH = PROJECT_ROOT / "data" / "baseline_embeddings.json"

PREPROCESSING_STAGE = "pre_preprocessing"


def _image_hash(image_path: Path) -> str:
    return hashlib.sha256(image_path.read_bytes()).hexdigest()


def capture_baseline_embeddings(
    image_paths: list[Path],
    encoders: list[tuple[str, str]] = QUALIFIED_ENCODERS,
    on_encoder_done: "Callable[[list[dict]], None] | None" = None,
) -> list[dict]:
    """
    Loads each encoder ONCE and embeds every image with it (not the
    reverse) - avoids reloading all 8 models per image. Returns one
    record per image, matching the schema written to disk by
    write_baseline_embeddings() below.

    Prints per-encoder start/finish and a periodic per-image tick - a
    silent multi-hour run with zero progress output is a real gap found
    the hard way (the first full-corpus run gave no visibility into
    whether it was progressing, stuck, or crashed). on_encoder_done, if
    given, is called with the current full record list after EACH
    encoder finishes - write_baseline_embeddings() uses this to persist
    partial progress incrementally rather than only at the very end, so
    a kill/crash partway through doesn't lose everything already
    computed.
    """
    per_image: dict[Path, dict] = {
        p: {
            "image": str(p),
            "image_hash": _image_hash(p),
            "preprocessing_stage": PREPROCESSING_STAGE,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "library_versions": {"timm": timm.__version__, "torch": torch.__version__},
            "embeddings": {},
        }
        for p in image_paths
    }

    n_images = len(image_paths)
    tick = max(1, n_images // 10)
    for encoder_i, (encoder_name, checkpoint) in enumerate(encoders, 1):
        print(f"[encoder {encoder_i}/{len(encoders)}] {encoder_name} ({checkpoint}): "
              f"embedding {n_images} image(s)...")
        model, transform = build_model_and_transform(checkpoint)
        for img_i, p in enumerate(image_paths, 1):
            with Image.open(p) as img:
                vector = embed_pooled(model, transform, img.convert("RGB"))
            per_image[p]["embeddings"][encoder_name] = {
                "checkpoint": checkpoint,
                "vector": [round(float(v), 6) for v in vector],
            }
            if img_i % tick == 0 or img_i == n_images:
                print(f"    {img_i}/{n_images}")
        del model, transform
        print(f"  done: {encoder_name}")
        if on_encoder_done is not None:
            on_encoder_done(list(per_image.values()))

    return list(per_image.values())


def _merge_and_save(records: list[dict], output_path: Path) -> dict[str, dict]:
    existing: list[dict] = []
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
    existing_by_image = {r["image"]: r for r in existing}
    for record in records:
        existing_by_image[record["image"]] = record  # append-only per image, latest wins for that image
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(list(existing_by_image.values()), f)
    return existing_by_image


def write_baseline_embeddings(
    image_paths: list[Path],
    output_path: Path = DEFAULT_BASELINE_PATH,
    encoders: list[tuple[str, str]] = QUALIFIED_ENCODERS,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _save_partial(records: list[dict]) -> None:
        merged = _merge_and_save(records, output_path)
        print(f"  (incremental save: {output_path}, {len(merged)} image(s) on file so far)")

    records = capture_baseline_embeddings(image_paths, encoders, on_encoder_done=_save_partial)
    final = _merge_and_save(records, output_path)

    print(f"Baseline embeddings written to {output_path} "
          f"({len(records)} image(s) this run, {len(final)} total on file).")
    return output_path
