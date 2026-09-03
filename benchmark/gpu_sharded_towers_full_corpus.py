"""
Full-corpus run of the sharded GPU tower inference validated in
benchmark/gpu_sharded_towers_experiment.py (2026-08-04) - 8 encoders
resident simultaneously, one CUDA stream per encoder, dispatched
concurrently per image. Same logic as that prototype (not
reimplemented differently), scaled to the full corpus with progress
output and metadata recording matching the Phase A/B pattern
(docs/GPU_CPU_EQUIVALENCE_REPORT.md).

Prototype validated: 2.34x speedup vs sequential GPU on a 40-image
sample, 0/320 correctness mismatches (cosine similarity 1.0 exactly).
This run is what actually happens at full scale, not an extrapolation.

Output: baseline_embeddings_gpu_sharded.json + companion .meta.json.
core/vision_embeddings.py NOT modified - device/stream-parameterized
copies of its two functions live only here, same as every other
experimental script this session.

Usage:
    python -m benchmark.gpu_sharded_towers_full_corpus
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image

from core.baseline_embeddings import PREPROCESSING_STAGE
from core.bucket_worklist import load_bucket_filepaths
from core.vision_embeddings import QUALIFIED_ENCODERS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "outputs" / "gpu_cpu_equivalence" / "baseline_embeddings_gpu_sharded.json"
META_PATH = PROJECT_ROOT / "data" / "outputs" / "gpu_cpu_equivalence" / "baseline_embeddings_gpu_sharded.meta.json"


def _image_hash(image_path: Path) -> str:
    return hashlib.sha256(image_path.read_bytes()).hexdigest()


def _build_model_and_transform(candidate: str):
    model = timm.create_model(candidate, pretrained=True, num_classes=0)
    model.eval()
    model.to("cuda")
    cfg = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**cfg)
    return model, transform


def _gpu_info() -> dict:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        name = result.stdout.strip() if result.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        name = None
    return {"gpu_name": name, "cuda_version": torch.version.cuda, "torch_version": torch.__version__}


@torch.no_grad()
def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available - aborting.")
        return

    manifest_path = PROJECT_ROOT / "data" / "manifest.csv"
    image_paths = [Path(p) for p in load_bucket_filepaths(manifest_path)]
    n = len(image_paths)
    print(f"Sharded GPU capture: {n} images, 8 encoders resident simultaneously, "
          f"per-encoder CUDA streams.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading all 8 encoders resident on GPU...")
    models = {}
    streams = {}
    for encoder_name, checkpoint in QUALIFIED_ENCODERS:
        model, transform = _build_model_and_transform(checkpoint)
        models[encoder_name] = (model, transform)
        streams[encoder_name] = torch.cuda.Stream()
    print(f"All 8 resident. VRAM after loading: {torch.cuda.memory_allocated()/1e6:.1f}MB")

    torch.cuda.reset_peak_memory_stats()
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

    t0 = time.time()
    tick = max(1, n // 40)
    for img_i, p in enumerate(image_paths, 1):
        with Image.open(p) as img:
            rgb = img.convert("RGB")
            results = {}
            for encoder_name, (model, transform) in models.items():
                with torch.cuda.stream(streams[encoder_name]):
                    x = transform(rgb).unsqueeze(0).to("cuda", non_blocking=True)
                    results[encoder_name] = model(x).squeeze(0)
            torch.cuda.synchronize()  # wait for every stream before reading any result
            for encoder_name, tensor in results.items():
                vector = tensor.cpu().numpy()
                per_image[p]["embeddings"][encoder_name] = {
                    "checkpoint": dict(QUALIFIED_ENCODERS)[encoder_name],
                    "vector": [round(float(v), 6) for v in vector],
                }
        if img_i % tick == 0 or img_i == n:
            print(f"  {img_i}/{n}")

    elapsed = time.time() - t0
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6

    records = list(per_image.values())
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f)

    meta = {
        "phase": "gpu_sharded_full_corpus",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_images": n,
        "elapsed_seconds": round(elapsed, 1),
        "images_per_second": round(n / elapsed, 4) if elapsed else None,
        "peak_vram_mb": round(peak_vram_mb, 1),
        **_gpu_info(),
        "embedding_format_version": "baseline_embeddings_v1 (per-record: image, image_hash, "
                                     "preprocessing_stage, timestamp, library_versions, embeddings)",
        "output_path": str(OUTPUT_PATH),
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone: {n} images, {elapsed:.1f}s ({elapsed/60:.1f} min), "
          f"{n/elapsed:.3f} img/s, peak VRAM={peak_vram_mb:.1f}MB")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Meta:   {META_PATH}")


if __name__ == "__main__":
    main()
