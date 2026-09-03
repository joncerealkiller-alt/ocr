"""
GPU "sharded" tower inference prototype (2026-08-04) - the "all encoders
resident, dispatched concurrently" architecture from this session's very
first Stage 1 scheduling research pass, now actually measured instead of
just reasoned about. "Shard" here means: all 8 encoder models loaded
resident on the GPU simultaneously (one CUDA stream lane per encoder),
rather than the current sequential pattern (one full model, one full
pass over every image, unload, next model).

Memory feasibility, answered empirically: Phase B (benchmark/gpu_cpu_
equivalence_phaseB_gpu.py) measured peak VRAM at 429.9MB for ONE
resident model. 8 models resident simultaneously is estimated ~3.4GB,
against 16.3GB total on this GPU - plenty of headroom, but this script
measures the REAL peak, not the estimate.

Correctness, not just speed: stream-based concurrent dispatch is
verified against the SAME sequential-GPU computation on the same
images (cosine similarity, matching the same rigor as the CPU-vs-GPU
equivalence pass) - a "faster but wrong" result would not be useful and
CUDA streams are exactly the kind of change where a synchronization
mistake could silently produce that.

Small sample first, deliberately - this is a prototype check, not a
full-corpus commitment. core/vision_embeddings.py is NOT modified;
device/stream-parameterized copies of its two functions live only here.

Usage:
    python -m benchmark.gpu_sharded_towers_experiment
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image

from core.vision_embeddings import QUALIFIED_ENCODERS, cosine_sim

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_SIZE = 40


def _sample_images(n: int) -> list[Path]:
    all_images = sorted(
        p for p in (PROJECT_ROOT / "data" / "working").iterdir()
        if p.is_file() and not p.name.endswith(".json")
    )
    step = max(1, len(all_images) // n)
    return all_images[::step][:n]


def _build_model_and_transform(candidate: str):
    model = timm.create_model(candidate, pretrained=True, num_classes=0)
    model.eval()
    model.to("cuda")
    cfg = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**cfg)
    return model, transform


@torch.no_grad()
def _run_sequential_gpu(images: list[Path]) -> tuple[dict, float]:
    """Baseline - matches Phase B's real production pattern exactly:
    one model resident at a time, full pass over every image, unload."""
    t0 = time.time()
    out: dict[str, dict[str, np.ndarray]] = {str(p): {} for p in images}
    for encoder_name, checkpoint in QUALIFIED_ENCODERS:
        model, transform = _build_model_and_transform(checkpoint)
        for p in images:
            with Image.open(p) as img:
                x = transform(img.convert("RGB")).unsqueeze(0).to("cuda")
                vec = model(x).squeeze(0).cpu().numpy()
            out[str(p)][encoder_name] = vec
        del model, transform
        torch.cuda.empty_cache()
    return out, time.time() - t0


@torch.no_grad()
def _run_sharded_gpu(images: list[Path]) -> tuple[dict, float, float]:
    """All 8 encoders loaded resident simultaneously, one CUDA stream per
    encoder. Per image: dispatch all 8 forward passes on their own
    streams (launches return immediately, don't block), then synchronize
    ALL streams before reading any result - this is the part a mistake
    here would most likely get wrong (reading a result before its
    stream's kernel actually finished)."""
    torch.cuda.reset_peak_memory_stats()
    print("Loading all 8 encoders resident on GPU...")
    models = {}
    streams = {}
    for encoder_name, checkpoint in QUALIFIED_ENCODERS:
        model, transform = _build_model_and_transform(checkpoint)
        models[encoder_name] = (model, transform)
        streams[encoder_name] = torch.cuda.Stream()
    resident_vram_mb = torch.cuda.memory_allocated() / 1e6
    print(f"All 8 resident. VRAM after loading: {resident_vram_mb:.1f}MB")

    t0 = time.time()
    out: dict[str, dict[str, np.ndarray]] = {str(p): {} for p in images}
    for p in images:
        with Image.open(p) as img:
            rgb = img.convert("RGB")
            results = {}
            for encoder_name, (model, transform) in models.items():
                with torch.cuda.stream(streams[encoder_name]):
                    x = transform(rgb).unsqueeze(0).to("cuda", non_blocking=True)
                    results[encoder_name] = model(x).squeeze(0)
            torch.cuda.synchronize()  # wait for EVERY stream before reading any result
            for encoder_name, tensor in results.items():
                out[str(p)][encoder_name] = tensor.cpu().numpy()

    elapsed = time.time() - t0
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6
    for model, transform in models.values():
        del model, transform
    torch.cuda.empty_cache()
    return out, elapsed, peak_vram_mb


def _compare(seq_out: dict, sharded_out: dict) -> None:
    sims = []
    for path_str, seq_vecs in seq_out.items():
        sharded_vecs = sharded_out[path_str]
        for encoder_name, seq_vec in seq_vecs.items():
            sharded_vec = sharded_vecs[encoder_name]
            sims.append(cosine_sim(seq_vec.astype(np.float64), sharded_vec.astype(np.float64)))
    sims = np.array(sims)
    print(f"\n--- Correctness check: sequential-GPU vs sharded-GPU cosine similarity ({len(sims)} pairs) ---")
    print(f"  min={sims.min():.8f}  mean={sims.mean():.8f}  max={sims.max():.8f}")
    n_bad = int((sims < 0.999999).sum())
    print(f"  pairs with cosine similarity < 0.999999: {n_bad} / {len(sims)}")


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    images = _sample_images(SAMPLE_SIZE)
    n = len(images)
    print(f"Testing {n} images.\n")

    seq_out, t_seq = _run_sequential_gpu(images)
    print(f"sequential GPU (current production pattern): {t_seq:.1f}s  ({n/t_seq:.3f} img/s)")

    sharded_out, t_sharded, peak_vram = _run_sharded_gpu(images)
    print(f"sharded GPU (8 resident, per-encoder streams): {t_sharded:.1f}s  ({n/t_sharded:.3f} img/s)")
    print(f"peak VRAM with all 8 resident: {peak_vram:.1f}MB")
    print(f"\nSpeedup, sharded vs sequential GPU: {t_seq/t_sharded:.2f}x")

    _compare(seq_out, sharded_out)


if __name__ == "__main__":
    main()
