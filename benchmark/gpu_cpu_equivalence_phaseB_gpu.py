"""
Phase B of the full-corpus CPU vs GPU equivalence characterization
(2026-08-04). Generates the IDENTICAL corpus's embeddings using an
EXPERIMENTAL GPU implementation - device-parameterized copies of
core/vision_embeddings.py's build_model_and_transform()/embed_pooled(),
local to this file only. core/vision_embeddings.py itself is NOT
modified and NOT imported for these two functions - per this
characterization's explicit constraint ("treat the GPU path as an
experimental implementation only... do not replace core/vision_
embeddings.py"), this is evidence-gathering, not a migration.

Must be run AFTER Phase A completes, never concurrently with it - both
phases would contend for the same CPU cores (Phase A) or GPU (nothing
else should be using it during Phase B, per CLAUDE.md's GPU-contention
discipline - check nvidia-smi first) and contaminate both timing
measurements.

Output: baseline_embeddings_gpu.json + baseline_embeddings_gpu.meta.json
(GPU model, CUDA version, torch version, peak VRAM, elapsed, img/s).

Usage:
    python -m benchmark.gpu_cpu_equivalence_phaseB_gpu
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

from core.vision_embeddings import QUALIFIED_ENCODERS
from core.bucket_worklist import load_bucket_filepaths
from core.baseline_embeddings import PREPROCESSING_STAGE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "outputs" / "gpu_cpu_equivalence" / "baseline_embeddings_gpu.json"
META_PATH = PROJECT_ROOT / "data" / "outputs" / "gpu_cpu_equivalence" / "baseline_embeddings_gpu.meta.json"


def _build_model_and_transform(candidate: str):
    model = timm.create_model(candidate, pretrained=True, num_classes=0)
    model.eval()
    model.to("cuda")
    cfg = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**cfg)
    return model, transform


@torch.no_grad()
def _embed_pooled(model, transform, pil_image: Image.Image) -> np.ndarray:
    x = transform(pil_image.convert("RGB")).unsqueeze(0).to("cuda")
    return model(x).squeeze(0).cpu().numpy()


def _image_hash(image_path: Path) -> str:
    return hashlib.sha256(image_path.read_bytes()).hexdigest()


def _gpu_info() -> dict:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        name = result.stdout.strip() if result.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        name = None
    return {
        "gpu_name": name,
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
    }


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available - aborting Phase B.")
        return

    manifest_path = PROJECT_ROOT / "data" / "manifest.csv"
    image_paths = [Path(p) for p in load_bucket_filepaths(manifest_path)]
    n = len(image_paths)
    print(f"Phase B: capturing GPU baseline embeddings for {n} images (experimental path, "
          f"core/vision_embeddings.py NOT modified/used for these two functions).")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

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
    tick = max(1, n // 20)
    for encoder_i, (encoder_name, checkpoint) in enumerate(QUALIFIED_ENCODERS, 1):
        print(f"[encoder {encoder_i}/{len(QUALIFIED_ENCODERS)}] {encoder_name} ({checkpoint}): "
              f"embedding {n} image(s) on GPU...")
        model, transform = _build_model_and_transform(checkpoint)
        for img_i, p in enumerate(image_paths, 1):
            with Image.open(p) as img:
                vector = _embed_pooled(model, transform, img.convert("RGB"))
            per_image[p]["embeddings"][encoder_name] = {
                "checkpoint": checkpoint,
                "vector": [round(float(v), 6) for v in vector],
            }
            if img_i % tick == 0 or img_i == n:
                print(f"    {img_i}/{n}")
        del model, transform
        torch.cuda.empty_cache()
        print(f"  done: {encoder_name}")

    elapsed = time.time() - t0
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6

    records = list(per_image.values())
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f)

    meta = {
        "phase": "B_gpu",
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
