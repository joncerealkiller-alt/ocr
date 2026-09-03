"""
Stage 1 (Raw Sensor Capture, semantic half) per docs/PIPELINE_STAGE_
TERMINOLOGY.md's canonical Stage 0-6 naming (2026-08-02). See that doc
and docs/REFERENCE_PIPELINE_V1.md for why this module's name stays
"baseline_embeddings" (capability-named, not stage-named, per Jon's
explicit instruction) even though its ROLE is Stage 1 - and for the
real mislabeling bug this module's data uncovered (existing-corpus
captures tagged "pre_preprocessing" that were actually post-Stage-3).

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
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import timm
import torch
from PIL import Image

from core.vision_embeddings import QUALIFIED_ENCODERS, build_model_and_transform, embed_pooled

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE_PATH = PROJECT_ROOT / "data" / "baseline_embeddings.json"
# Stage 4 (Validation Capture, 2026-08-04): the post-Stage-3 counterpart
# to DEFAULT_BASELINE_PATH, written by calling write_baseline_embeddings()
# with this as output_path - same function, same schema, separate file.
# DEFAULT_BASELINE_PATH itself is never touched by that call.
DEFAULT_POSTPROCESSING_PATH = PROJECT_ROOT / "data" / "postprocessing_embeddings.json"

PREPROCESSING_STAGE = "pre_preprocessing"

# Execution strategy validated 2026-08-04 (benchmark/gpu_sharded_towers_
# full_corpus.py, benchmark/gpu_images_in_flight_sweep.py): 8 encoders
# resident on GPU simultaneously, one CUDA stream per encoder per
# in-flight image, k=4 concurrent images (measured plateau - gain below,
# CPU-preprocessing-bound above). Preprocessing (resize/crop/totensor/
# normalize) stays on CPU, unchanged - GPU normalize was separately
# tested and found to add nothing once inference is already GPU-resident.
# k=4 is a measured optimum on THIS machine's GPU (RTX 5060 Ti) - a
# default worth revisiting on different hardware, not a fixed law.
DEFAULT_GPU_FLIGHT_LEVEL = 4


def resolve_baseline_image_path(path_str: str) -> str:
    """
    Normalizes a path string to an absolute, resolved form anchored at
    PROJECT_ROOT - NOT the caller's cwd. Needed because records in this
    module's own output file are captured with paths RELATIVE to
    PROJECT_ROOT (e.g. "data\\working\\foo.jpg" - confirmed by direct
    inspection of the real data/baseline_embeddings.json), while most of
    the rest of this project (manifest.csv, bucket CSVs) uses absolute
    paths. Any reader that needs to match a baseline_embeddings.json
    record against a path from elsewhere must normalize both sides the
    same way - promoted here 2026-08-02 after scripts/migrate_manifest_
    to_db.py hit this as a real bug (0/1677 baseline records matched
    until both sides were normalized) so core/decision_engine.py doesn't
    have to rediscover/reimplement the same fix a second time.
    """
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def _image_hash(image_path: Path) -> str:
    return hashlib.sha256(image_path.read_bytes()).hexdigest()


@torch.no_grad()
def _embed_one_image_gpu(p: Path, models: dict, stream_set: dict) -> dict:
    """One image, all 8 encoders, each on its own dedicated CUDA stream
    from this call's own stream_set - never a stream shared with another
    concurrently in-flight image (that's what makes k-images-in-flight
    safe). Preprocessing (transform, including normalize) still runs on
    CPU exactly as build_model_and_transform() already defines it - only
    the .to("cuda") move and inference are on GPU."""
    with Image.open(p) as img:
        rgb = img.convert("RGB")
        vectors = {}
        for encoder_name, (model, transform) in models.items():
            stream = stream_set[encoder_name]
            with torch.cuda.stream(stream):
                x = transform(rgb).unsqueeze(0).to("cuda", non_blocking=True)
                vectors[encoder_name] = model(x).squeeze(0)
        for stream in stream_set.values():
            stream.synchronize()
        return {name: t.cpu().numpy() for name, t in vectors.items()}


def capture_baseline_embeddings(
    image_paths: list[Path],
    encoders: list[tuple[str, str]] = QUALIFIED_ENCODERS,
    on_encoder_done: "Callable[[list[dict]], None] | None" = None,
) -> list[dict]:
    """
    Returns one record per image, matching the schema written to disk by
    write_baseline_embeddings() below.

    Execution engine (2026-08-04): GPU-sharded, k=4-images-in-flight when
    CUDA is available (see DEFAULT_GPU_FLIGHT_LEVEL) - all 8 encoders
    load ONCE, resident on GPU for the whole call; up to
    DEFAULT_GPU_FLIGHT_LEVEL images are embedded concurrently, each on
    its own dedicated set of 8 CUDA streams. Falls back to the original
    sequential CPU loop (one encoder loaded at a time, across all images)
    when CUDA isn't available. Preprocessing and the output schema are
    identical either way - only where inference runs, and how many
    images are in flight, changed.

    Prints periodic progress - a silent multi-hour run with zero output
    is a real gap found the hard way. on_encoder_done, if given, is
    called with the current full record list whenever enough progress
    has accumulated since the last call - after each encoder finishes on
    the CPU path (there, "an encoder finished" and "enough progress
    accumulated" are the same event); on an images-completed interval on
    the GPU path, since encoders don't finish independently there. Either
    way the contract seen by callers is unchanged: get called often
    enough that a crash doesn't lose much, not tied to any particular
    cause. write_baseline_embeddings() uses this to persist partial
    progress incrementally.
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
    t0 = time.time()

    if torch.cuda.is_available():
        flight_level = DEFAULT_GPU_FLIGHT_LEVEL
        print(f"[GPU sharded, k={flight_level}] loading {len(encoders)} encoders resident on GPU...")
        models = {}
        for encoder_name, checkpoint in encoders:
            model, transform = build_model_and_transform(checkpoint)
            models[encoder_name] = (model.cuda(), transform)
        print(f"  resident. VRAM: {torch.cuda.memory_allocated()/1e6:.1f}MB. "
              f"Embedding {n_images} image(s)...")

        stream_sets = [
            {encoder_name: torch.cuda.Stream() for encoder_name, _ in encoders}
            for _ in range(flight_level)
        ]
        lock = threading.Lock()
        completed = 0
        last_checkpoint = 0
        n_checkpoints = 0

        def _process(i: int, p: Path) -> None:
            nonlocal completed, last_checkpoint, n_checkpoints
            vectors = _embed_one_image_gpu(p, models, stream_sets[i % flight_level])
            for encoder_name, vec in vectors.items():
                checkpoint = next(cp for name, cp in encoders if name == encoder_name)
                per_image[p]["embeddings"][encoder_name] = {
                    "checkpoint": checkpoint,
                    "vector": [round(float(v), 6) for v in vec],
                }
            with lock:
                completed += 1
                if completed % tick == 0 or completed == n_images:
                    print(f"    {completed}/{n_images}")
                # Checkpointing decoupled from any notion of "an encoder
                # finished" (there's no such event in this execution
                # engine - every image passes through all 8 encoders
                # together): save whenever enough NEW images have
                # completed since the last save, regardless of what
                # triggered it. The caller only cares that progress is
                # saved often enough that a crash doesn't lose much.
                if on_encoder_done is not None and completed - last_checkpoint >= tick:
                    on_encoder_done(list(per_image.values()))
                    last_checkpoint = completed
                    n_checkpoints += 1

        with ThreadPoolExecutor(max_workers=flight_level) as executor:
            list(executor.map(lambda args: _process(*args), enumerate(image_paths)))

        for model, _transform in models.values():
            del model
        torch.cuda.empty_cache()
        elapsed = time.time() - t0
        print(f"[GPU sharded, k={flight_level}] done: {n_images} image(s) in {elapsed:.1f}s "
              f"({n_images/elapsed:.2f} img/s), {n_checkpoints} checkpoint(s) saved, 0 failure(s).")
        return list(per_image.values())

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

    elapsed = time.time() - t0
    cpu_checkpoints = len(encoders) if on_encoder_done is not None else 0
    print(f"[CPU, sequential] done: {n_images} image(s) x {len(encoders)} encoder(s) in "
          f"{elapsed:.1f}s ({n_images/elapsed:.2f} img/s), "
          f"{cpu_checkpoints} checkpoint(s) saved, 0 failure(s).")
    return list(per_image.values())


def merge_sensor_records(records: list[dict], output_path: Path) -> dict[str, dict]:
    """
    Merges new sensor-capture records into the on-disk file, one record
    per image, keyed by "image" - a SHALLOW dict update per image
    (existing top-level keys not present in the new record are
    preserved), not a full per-image replace.

    Changed 2026-08-04 (was a full-record replace, `existing_by_image
    [record["image"]] = record`) specifically so two DIFFERENT sensor
    capture passes writing to the SAME file can coexist on the same
    image's record without one wiping out the other's data - e.g.
    capture_layout_detections()'s "layout_detections" key and
    capture_baseline_embeddings()'s "embeddings" key on the same image.
    Still "latest wins" for any KEY present in both the existing and new
    record - e.g. re-running capture_baseline_embeddings() replaces that
    image's "embeddings"/"timestamp"/etc. wholesale, unchanged from the
    original behavior, since that function always writes every one of
    its own keys on every call; the only behavior change is that keys
    the NEW record doesn't mention (e.g. a different sensor's data) are
    no longer discarded.
    """
    existing: list[dict] = []
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
    existing_by_image = {r["image"]: r for r in existing}
    for record in records:
        image_key = record["image"]
        if image_key in existing_by_image:
            existing_by_image[image_key].update(record)
        else:
            existing_by_image[image_key] = record
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
        merged = merge_sensor_records(records, output_path)
        print(f"  (incremental save: {output_path}, {len(merged)} image(s) on file so far)")

    records = capture_baseline_embeddings(image_paths, encoders, on_encoder_done=_save_partial)
    final = merge_sensor_records(records, output_path)

    print(f"Baseline embeddings written to {output_path} "
          f"({len(records)} image(s) this run, {len(final)} total on file).")
    return output_path


# ---------------------------------------------------------------------------
# Layout-detection sensor (core/layout_detector.py, added 2026-08-04) -
# recorded into the SAME per-image file/record as the embeddings above
# (via merge_sensor_records()), under a "layout_detections" key instead of
# "embeddings" - a different sensor TYPE (variable-length labeled boxes,
# not a fixed-length vector), same capture/persist discipline: pure
# measurement, no decision made here about which detections to trust.
# ---------------------------------------------------------------------------

LAYOUT_DETECTOR_NAME = "doclayout_yolo"


def capture_layout_detections(
    image_paths: list[Path],
    model=None,
    on_progress: "Callable[[list[dict]], None] | None" = None,
) -> list[dict]:
    """
    Same per-image record shape/discipline as capture_baseline_embeddings()
    (image/image_hash/timestamp/library_versions), but under a
    "layout_detections" key instead of "embeddings" - kept as its own
    top-level key (not folded into "embeddings") because the payload
    shape is genuinely different (a variable-length list of labeled
    boxes, not a fixed-length vector); see merge_sensor_records() for how
    the two coexist in the same file/record without either overwriting
    the other.

    model: pass an already-loaded model (core.layout_detector.
    build_layout_model()) to avoid reloading across repeated calls, same
    reasoning as capture_baseline_embeddings() loading each encoder once
    for the whole batch rather than once per image. Built fresh if not
    given.

    Sequential, one image at a time (no GPU-sharded/k-in-flight path like
    capture_baseline_embeddings() has) - a single YOLO forward pass is
    already cheap relative to the 8-encoder embedding battery, and
    concurrency for this specific model hasn't been measured/validated,
    unlike the embeddings path's benchmarked k=4 setup. Revisit only if
    this turns out to be a real bottleneck at full-corpus scale.
    """
    from core.layout_detector import (
        DOCSTRUCTBENCH_FILENAME, DOCSTRUCTBENCH_REPO_ID, DEFAULT_CONF,
        DEFAULT_IMGSZ, build_layout_model, detect_layout,
    )

    model = model or build_layout_model()
    checkpoint = f"{DOCSTRUCTBENCH_REPO_ID}/{DOCSTRUCTBENCH_FILENAME}"

    n_images = len(image_paths)
    tick = max(1, n_images // 10)
    t0 = time.time()
    records = []

    print(f"[{LAYOUT_DETECTOR_NAME}] detecting layout regions in {n_images} image(s)...")
    for i, p in enumerate(image_paths, 1):
        with Image.open(p) as img:
            detections = detect_layout(model, img.convert("RGB"), imgsz=DEFAULT_IMGSZ, conf=DEFAULT_CONF)
        records.append({
            "image": str(p),
            "image_hash": _image_hash(p),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "library_versions": {"torch": torch.__version__},
            "layout_detections": {
                LAYOUT_DETECTOR_NAME: {
                    "checkpoint": checkpoint,
                    "imgsz": DEFAULT_IMGSZ,
                    "conf_threshold": DEFAULT_CONF,
                    "detections": detections,
                },
            },
        })
        if i % tick == 0 or i == n_images:
            print(f"    {i}/{n_images}")
            if on_progress is not None:
                on_progress(records)

    elapsed = time.time() - t0
    print(f"[{LAYOUT_DETECTOR_NAME}] done: {n_images} image(s) in {elapsed:.1f}s "
          f"({n_images / elapsed:.2f} img/s), 0 failure(s).")
    return records


def write_layout_detections(
    image_paths: list[Path],
    output_path: Path = DEFAULT_BASELINE_PATH,
    model=None,
) -> Path:
    """
    Writes into DEFAULT_BASELINE_PATH by default (data/baseline_
    embeddings.json) - the SAME file capture_baseline_embeddings() writes
    to, on purpose: "record it the same as the other sensor tower data"
    means one consolidated per-image sensor record, not a second parallel
    file to keep in sync. merge_sensor_records() (shallow per-image key
    update) is what makes this safe - writing layout data for an image
    that already has embeddings adds the "layout_detections" key without
    touching "embeddings", and vice versa.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _save_partial(records: list[dict]) -> None:
        merged = merge_sensor_records(records, output_path)
        print(f"  (incremental save: {output_path}, {len(merged)} image(s) on file so far)")

    records = capture_layout_detections(image_paths, model=model, on_progress=_save_partial)
    final = merge_sensor_records(records, output_path)

    print(f"Layout detections written to {output_path} "
          f"({len(records)} image(s) this run, {len(final)} total on file).")
    return output_path
