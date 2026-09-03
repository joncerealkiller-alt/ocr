"""
Full-corpus confirmation of benchmark/gpu_normalize_flight_sweep.py's
sample result (60 images: gpu_normalize showed no consistent gain over
cpu_normalize at any flight level - best case 1.02x noise, real 0.94x
regression at k=4/k=6 - with perfect correctness at every level, 0/480
pairs below threshold each time). Runs both configs once each, at
k=4 - the plateau concurrency level found by the earlier images-in-
flight sweep (same session) - across the full corpus, to confirm the
null/negative result holds at real scale rather than just a 60-image
sample.

core/vision_embeddings.py is NOT modified. Read-only against the real
corpus (data/working/) - writes only benchmark output files under
data/outputs/gpu_normalize_full_corpus/, nothing production.

Usage:
    python -m benchmark.gpu_normalize_full_corpus
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from benchmark.gpu_normalize_flight_sweep import (
    _build_model_and_transform,
    _process_one_image_cpu_normalize,
    _process_one_image_gpu_normalize,
    _run_config,
)
from core.vision_embeddings import QUALIFIED_ENCODERS, cosine_sim

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FLIGHT_LEVEL = 4
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "gpu_normalize_full_corpus"


def _all_images() -> list[Path]:
    return sorted(
        p for p in (PROJECT_ROOT / "data" / "working").iterdir()
        if p.is_file() and not p.name.endswith(".json")
    )


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    images = _all_images()
    n = len(images)
    print(f"Full-corpus gpu_normalize vs cpu_normalize confirmation: {n} images, k={FLIGHT_LEVEL}.\n")

    print("Loading all 8 encoders resident on GPU...")
    models = {}
    stages_by_encoder = {}
    mean_std_by_encoder = {}
    for encoder_name, checkpoint in QUALIFIED_ENCODERS:
        model, transform = _build_model_and_transform(checkpoint)
        models[encoder_name] = (model, transform)
        stages = list(transform.transforms)
        stages_by_encoder[encoder_name] = stages
        mean_t = stages[3].mean.view(1, 3, 1, 1).cuda()
        std_t = stages[3].std.view(1, 3, 1, 1).cuda()
        mean_std_by_encoder[encoder_name] = (mean_t, std_t)
    print(f"Resident. VRAM: {torch.cuda.memory_allocated()/1e6:.1f}MB\n")

    print(f"Running cpu_normalize (k={FLIGHT_LEVEL}) over {n} images...")
    cpu_out, cpu_elapsed, cpu_gpu_stats, cpu_cpu_stats = _run_config(
        _process_one_image_cpu_normalize, images, models, (), FLIGHT_LEVEL)
    cpu_img_s = n / cpu_elapsed if cpu_elapsed else 0
    print(f"  done: {cpu_elapsed:.1f}s, {cpu_img_s:.3f} img/s, "
          f"GPU util mean {cpu_gpu_stats['mean']}% max {cpu_gpu_stats['max']}%, "
          f"CPU util mean {cpu_cpu_stats['mean']}% max {cpu_cpu_stats['max']}%\n")

    print(f"Running gpu_normalize (k={FLIGHT_LEVEL}) over {n} images...")
    gpu_out, gpu_elapsed, gpu_gpu_stats, gpu_cpu_stats = _run_config(
        _process_one_image_gpu_normalize, images, models,
        (stages_by_encoder, mean_std_by_encoder), FLIGHT_LEVEL)
    gpu_img_s = n / gpu_elapsed if gpu_elapsed else 0
    print(f"  done: {gpu_elapsed:.1f}s, {gpu_img_s:.3f} img/s, "
          f"GPU util mean {gpu_gpu_stats['mean']}% max {gpu_gpu_stats['max']}%, "
          f"CPU util mean {gpu_cpu_stats['mean']}% max {gpu_cpu_stats['max']}%\n")

    print("Comparing correctness across full corpus (1750 x 8 = 14000 pairs)...")
    sims = []
    for path_str, ref_vecs in cpu_out.items():
        for encoder_name, ref_vec in ref_vecs.items():
            sims.append(cosine_sim(
                ref_vec.astype(np.float64), gpu_out[path_str][encoder_name].astype(np.float64)))
    sims = np.array(sims)
    n_bad = int((sims < 0.9999).sum())
    speedup = cpu_elapsed / gpu_elapsed if gpu_elapsed else 0

    print(f"\n=== Full-corpus result (k={FLIGHT_LEVEL}, n={n} images, {len(sims)} pairs) ===")
    print(f"cpu_normalize: {cpu_elapsed:.1f}s  {cpu_img_s:.3f} img/s")
    print(f"gpu_normalize: {gpu_elapsed:.1f}s  {gpu_img_s:.3f} img/s")
    print(f"speedup: {speedup:.3f}x")
    print(f"correctness: cosine sim min={sims.min():.8f} mean={sims.mean():.8f} max={sims.max():.8f}")
    print(f"pairs below 0.9999 threshold: {n_bad}/{len(sims)}")
    print(f"GPU util: cpu_normalize mean {cpu_gpu_stats['mean']}% -> gpu_normalize mean {gpu_gpu_stats['mean']}%")
    print(f"CPU util: cpu_normalize mean {cpu_cpu_stats['mean']}% -> gpu_normalize mean {gpu_cpu_stats['mean']}%")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "n_images": n,
        "flight_level": FLIGHT_LEVEL,
        "cpu_normalize": {"elapsed_s": cpu_elapsed, "img_per_sec": cpu_img_s,
                           "gpu_util": cpu_gpu_stats, "cpu_util": cpu_cpu_stats},
        "gpu_normalize": {"elapsed_s": gpu_elapsed, "img_per_sec": gpu_img_s,
                           "gpu_util": gpu_gpu_stats, "cpu_util": gpu_cpu_stats},
        "speedup": speedup,
        "correctness": {
            "cosine_sim_min": float(sims.min()), "cosine_sim_mean": float(sims.mean()),
            "cosine_sim_max": float(sims.max()), "n_pairs": len(sims), "n_below_threshold": n_bad,
        },
    }
    out_path = OUTPUT_DIR / "result.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
