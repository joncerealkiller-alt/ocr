"""
Semantic embedding capture (core/baseline_embeddings.py) concurrency +
CPU-vs-GPU prototype (2026-08-04) - forced by a real incident this
session (a mistaken test write destroyed the corpus's real baseline_
embeddings.json, requiring full recomputation) turned into an
opportunity to measure two open questions instead of just blindly
redoing the same sequential CPU run:

1. Does the same "more images concurrently, fewer threads per call"
   pattern that gave core/image_analysis.py a 4.5x speedup also apply
   here? DIFFERENT starting point than that case: torch.get_num_threads()
   already defaults to 16 on this machine (confirmed, not assumed) -
   PyTorch's intra-op parallelism for a Base-scale ViT's matmuls is a
   real, substantial per-call cost unlike OpenCV's classical CV ops,
   which barely benefited from their own intra-op threading at all. So
   this needs its OWN measurement, not an assumption transferred from
   the physical-sensor result.

2. GPU vs CPU: this project's own Stage 1 GPU-scheduling research
   (earlier this session) found core/vision_embeddings.py never calls
   .cuda() anywhere - CPU-only by default despite CUDA being available -
   and flagged that moving to GPU is NOT "zero behavioral change" by
   default, since CPU (MKL/oneDNN) and GPU (cuDNN/cuBLAS) don't
   guarantee bit-identical floating-point results. This script measures
   BOTH wall-clock speed AND actual embedding-vector cosine similarity
   between CPU- and GPU-computed vectors for the SAME images - the
   equivalence question, not just the speed question. GPU code paths
   here are LOCAL to this benchmark (device-parameterized copies of
   build_model_and_transform()/embed_pooled()) - core/vision_
   embeddings.py itself is NOT modified, matching this project's
   "prototype first, promote only once validated" discipline.

Read-only against the corpus (measures, never writes sidecars or
touches data/baseline_embeddings.json).

Usage:
    python -m benchmark.baseline_embeddings_concurrency_experiment
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import timm
from PIL import Image

from core.vision_embeddings import QUALIFIED_ENCODERS, cosine_sim

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_SIZE = 16  # small - this workload is far more expensive per image
                  # than the physical sensor (real neural net forward
                  # passes x 8 encoders, not classical CV ops)


def _sample_images(n: int) -> list[Path]:
    all_images = sorted(
        p for p in (PROJECT_ROOT / "data" / "working").iterdir()
        if p.is_file() and not p.name.endswith(".json")
    )
    step = max(1, len(all_images) // n)
    return all_images[::step][:n]


def _build_model_and_transform(candidate: str, device: str):
    model = timm.create_model(candidate, pretrained=True, num_classes=0)
    model.eval()
    model.to(device)
    cfg = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**cfg)
    return model, transform


@torch.no_grad()
def _embed_pooled(model, transform, pil_image: Image.Image, device: str) -> np.ndarray:
    x = transform(pil_image.convert("RGB")).unsqueeze(0).to(device)
    return model(x).squeeze(0).cpu().numpy()


def _run_sequential_cpu(images: list[Path], intra_op_threads: int) -> tuple[dict, float]:
    torch.set_num_threads(intra_op_threads)
    t0 = time.time()
    out: dict[str, dict[str, np.ndarray]] = {str(p): {} for p in images}
    for encoder_name, checkpoint in QUALIFIED_ENCODERS:
        model, transform = _build_model_and_transform(checkpoint, "cpu")
        for p in images:
            with Image.open(p) as img:
                out[str(p)][encoder_name] = _embed_pooled(model, transform, img, "cpu")
        del model, transform
    return out, time.time() - t0


def _run_threaded_cpu(images: list[Path], n_workers: int, intra_op_threads: int) -> tuple[dict, float]:
    torch.set_num_threads(intra_op_threads)
    t0 = time.time()
    out: dict[str, dict[str, np.ndarray]] = {str(p): {} for p in images}
    for encoder_name, checkpoint in QUALIFIED_ENCODERS:
        model, transform = _build_model_and_transform(checkpoint, "cpu")

        def _one(p: Path):
            with Image.open(p) as img:
                return str(p), _embed_pooled(model, transform, img, "cpu")

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for path_str, vec in ex.map(_one, images):
                out[path_str][encoder_name] = vec
        del model, transform
    return out, time.time() - t0


def _run_sequential_gpu(images: list[Path]) -> tuple[dict, float]:
    t0 = time.time()
    out: dict[str, dict[str, np.ndarray]] = {str(p): {} for p in images}
    for encoder_name, checkpoint in QUALIFIED_ENCODERS:
        model, transform = _build_model_and_transform(checkpoint, "cuda")
        for p in images:
            with Image.open(p) as img:
                out[str(p)][encoder_name] = _embed_pooled(model, transform, img, "cuda")
        del model, transform
        torch.cuda.empty_cache()
    return out, time.time() - t0


def _compare_embeddings(cpu_out: dict, gpu_out: dict) -> None:
    """Cosine similarity between CPU- and GPU-computed vectors for every
    (image, encoder) pair - the direct equivalence measurement, not
    inferred from timing. 1.0 = identical direction; this project's
    downstream logic rounds cosine similarities to 4 decimals, so
    deviations below ~1e-4 are very unlikely to change any recorded
    score, and larger ones are worth knowing about explicitly."""
    sims = []
    for path_str, cpu_vecs in cpu_out.items():
        gpu_vecs = gpu_out[path_str]
        for encoder_name, cpu_vec in cpu_vecs.items():
            gpu_vec = gpu_vecs[encoder_name]
            sim = cosine_sim(cpu_vec.astype(np.float64), gpu_vec.astype(np.float64))
            sims.append(sim)
    sims = np.array(sims)
    print(f"\n--- CPU vs GPU embedding-vector cosine similarity ({len(sims)} (image, encoder) pairs) ---")
    print(f"  min={sims.min():.8f}  mean={sims.mean():.8f}  max={sims.max():.8f}")
    n_not_1 = int((sims < 0.999999).sum())
    print(f"  pairs with cosine similarity < 0.999999: {n_not_1} / {len(sims)}")


def main() -> None:
    images = _sample_images(SAMPLE_SIZE)
    n = len(images)
    print(f"Benchmarking {n} images per configuration (fixed sample, identical across all configs).\n")

    print(f"{'configuration':<45}{'elapsed(s)':<12}{'img/s':<10}")

    cpu_out, t_seq = _run_sequential_cpu(images, intra_op_threads=16)
    print(f"{'sequential CPU, 16 intra-op threads (default)':<45}{t_seq:<12.1f}{n/t_seq:<10.3f}")

    _, t_thread4 = _run_threaded_cpu(images, n_workers=4, intra_op_threads=4)
    print(f"{'threaded CPU x4, 4 intra-op threads/worker':<45}{t_thread4:<12.1f}{n/t_thread4:<10.3f}")

    _, t_thread2 = _run_threaded_cpu(images, n_workers=2, intra_op_threads=8)
    print(f"{'threaded CPU x2, 8 intra-op threads/worker':<45}{t_thread2:<12.1f}{n/t_thread2:<10.3f}")

    if torch.cuda.is_available():
        gpu_out, t_gpu = _run_sequential_gpu(images)
        print(f"{'sequential GPU':<45}{t_gpu:<12.1f}{n/t_gpu:<10.3f}")
        print(f"\nSpeedup GPU vs sequential CPU baseline: {t_seq/t_gpu:.2f}x")
        _compare_embeddings(cpu_out, gpu_out)
    else:
        print("CUDA not available - skipping GPU configuration.")

    print(f"\nSpeedup vs sequential CPU baseline:")
    print(f"  threaded CPU x4: {t_seq/t_thread4:.2f}x")
    print(f"  threaded CPU x2: {t_seq/t_thread2:.2f}x")


if __name__ == "__main__":
    main()
