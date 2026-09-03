"""
GPU-normalize x images-in-flight sweep (2026-08-04) - follow-up to both
benchmark/gpu_images_in_flight_sweep.py (found the k=4 plateau, CPU-
preprocessing-bound) and benchmark/gpu_preprocessing_stage34_prototypes.py
(found gpu_normalize is numerically safe - 0/240 pairs failing 0.9999 -
but its measured 3.39x speedup there was confounded with also moving
inference to GPU, which the images-in-flight sweep ALREADY does for its
"baseline"). This isolates normalize as the only NEW variable on top of
the already-GPU-resident-inference images-in-flight pipeline, at every
flight level, to see whether it buys anything once inference is already
on GPU - not assumed either way.

Two configs, run back-to-back at each flight level so GPU/CPU load
conditions are as comparable as possible:

  cpu_normalize - current images-in-flight behavior unchanged: full
                  transform (resize/crop/totensor/normalize) on CPU,
                  tensor moved to GPU, inference on GPU.
  gpu_normalize - resize/crop/totensor stay CPU; normalize moves to
                  GPU (elementwise, pre-validated numerically safe);
                  inference stays GPU.

Correctness: gpu_normalize's output compared against cpu_normalize's
output AT THE SAME flight level (not just k=1), since this checks
whether the per-image dedicated-stream concurrency design has any
cross-image interference for the new GPU-side normalize step specifically.

core/vision_embeddings.py is NOT modified. Sample only (SAMPLE_SIZE=60,
matching the earlier sweep) - full-corpus run is a separate follow-up
script once this is validated.

Usage:
    python -m benchmark.gpu_normalize_flight_sweep
"""
from __future__ import annotations

import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import psutil
import timm
import torch
from PIL import Image

from core.vision_embeddings import QUALIFIED_ENCODERS, cosine_sim

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_SIZE = 60
FLIGHT_LEVELS = [1, 2, 3, 4, 6, 8]


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


class _GpuUtilMonitor:
    def __init__(self, interval: float = 0.5):
        self._interval = interval
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2,
                )
                if result.returncode == 0 and result.stdout.strip():
                    self._samples.append(float(result.stdout.strip()))
            except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
                pass
            self._stop.wait(self._interval)

    def start(self):
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        self._thread.join(timeout=2)
        if not self._samples:
            return {"mean": None, "max": None}
        return {"mean": round(sum(self._samples) / len(self._samples), 1), "max": round(max(self._samples), 1)}


class _CpuUtilMonitor:
    def __init__(self, interval: float = 0.5):
        self._interval = interval
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self._samples.append(psutil.cpu_percent(interval=self._interval))

    def start(self):
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        self._thread.join(timeout=2)
        if not self._samples:
            return {"mean": None, "max": None}
        return {"mean": round(sum(self._samples) / len(self._samples), 1), "max": round(max(self._samples), 1)}


@torch.no_grad()
def _process_one_image_cpu_normalize(p: Path, models: dict, stream_set: dict) -> dict:
    """Unchanged from the original images-in-flight sweep: full CPU
    transform including normalize, then GPU inference."""
    with Image.open(p) as img:
        rgb = img.convert("RGB")
        results = {}
        for encoder_name, (model, transform) in models.items():
            stream = stream_set[encoder_name]
            with torch.cuda.stream(stream):
                x = transform(rgb).unsqueeze(0).to("cuda", non_blocking=True)
                results[encoder_name] = model(x).squeeze(0)
        for stream in stream_set.values():
            stream.synchronize()
        return {name: t.cpu().numpy() for name, t in results.items()}


@torch.no_grad()
def _process_one_image_gpu_normalize(p: Path, models: dict, stages_by_encoder: dict,
                                      mean_std_by_encoder: dict, stream_set: dict) -> dict:
    """resize/crop/totensor stay CPU; normalize moves to GPU inside the
    same per-encoder stream as inference - isolates normalize as the
    only new variable versus _process_one_image_cpu_normalize."""
    with Image.open(p) as img:
        rgb = img.convert("RGB")
        results = {}
        for encoder_name, (model, _transform) in models.items():
            resize_t, crop_t, totensor_t, _normalize_t = stages_by_encoder[encoder_name]
            mean_t, std_t = mean_std_by_encoder[encoder_name]
            stream = stream_set[encoder_name]
            with torch.cuda.stream(stream):
                x = totensor_t(crop_t(resize_t(rgb))).unsqueeze(0).to("cuda", non_blocking=True)
                x = (x - mean_t) / std_t
                results[encoder_name] = model(x).squeeze(0)
        for stream in stream_set.values():
            stream.synchronize()
        return {name: t.cpu().numpy() for name, t in results.items()}


def _run_config(fn, images, models, extra_args, k) -> tuple[dict, float, dict, dict]:
    stream_sets = [
        {encoder_name: torch.cuda.Stream() for encoder_name, _ in QUALIFIED_ENCODERS}
        for _ in range(k)
    ]
    gpu_monitor = _GpuUtilMonitor()
    cpu_monitor = _CpuUtilMonitor()
    gpu_monitor.start()
    cpu_monitor.start()
    t0 = time.time()
    out: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=k) as executor:
        futures = {
            executor.submit(fn, p, models, *extra_args, stream_sets[i % k]): p
            for i, p in enumerate(images)
        }
        for future in futures:
            p = futures[future]
            out[str(p)] = future.result()
    elapsed = time.time() - t0
    gpu_stats = gpu_monitor.stop()
    cpu_stats = cpu_monitor.stop()
    return out, elapsed, gpu_stats, cpu_stats


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    images = _sample_images(SAMPLE_SIZE)
    n = len(images)
    print(f"GPU-normalize x images-in-flight sweep over {n} images.\n")

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

    header = (f"{'k':<4}{'config':<16}{'elapsed(s)':<12}{'img/s':<9}"
              f"{'GPU mean%':<11}{'GPU max%':<10}{'CPU mean%':<11}{'CPU max%':<10}")
    print(header)
    rows = []

    for k in FLIGHT_LEVELS:
        cpu_out, cpu_elapsed, cpu_gpu_stats, cpu_cpu_stats = _run_config(
            _process_one_image_cpu_normalize, images, models, (), k)
        cpu_img_s = n / cpu_elapsed if cpu_elapsed else 0
        print(f"{k:<4}{'cpu_normalize':<16}{cpu_elapsed:<12.1f}{cpu_img_s:<9.3f}"
              f"{str(cpu_gpu_stats['mean']):<11}{str(cpu_gpu_stats['max']):<10}"
              f"{str(cpu_cpu_stats['mean']):<11}{str(cpu_cpu_stats['max']):<10}")

        gpu_out, gpu_elapsed, gpu_gpu_stats, gpu_cpu_stats = _run_config(
            _process_one_image_gpu_normalize, images, models,
            (stages_by_encoder, mean_std_by_encoder), k)
        gpu_img_s = n / gpu_elapsed if gpu_elapsed else 0
        print(f"{k:<4}{'gpu_normalize':<16}{gpu_elapsed:<12.1f}{gpu_img_s:<9.3f}"
              f"{str(gpu_gpu_stats['mean']):<11}{str(gpu_gpu_stats['max']):<10}"
              f"{str(gpu_cpu_stats['mean']):<11}{str(gpu_cpu_stats['max']):<10}")

        sims = []
        for path_str, ref_vecs in cpu_out.items():
            for encoder_name, ref_vec in ref_vecs.items():
                sims.append(cosine_sim(
                    ref_vec.astype(np.float64), gpu_out[path_str][encoder_name].astype(np.float64)))
        sims = np.array(sims)
        n_bad = int((sims < 0.9999).sum())
        speedup = cpu_elapsed / gpu_elapsed if gpu_elapsed else 0
        print(f"     correctness (gpu_normalize vs cpu_normalize @ k={k}): "
              f"mean cos_sim={sims.mean():.8f}, {n_bad}/{len(sims)} pairs below 0.9999  "
              f"| speedup {speedup:.2f}x\n")

        rows.append({
            "k": k, "cpu_elapsed": cpu_elapsed, "cpu_img_s": cpu_img_s,
            "gpu_elapsed": gpu_elapsed, "gpu_img_s": gpu_img_s,
            "speedup": speedup, "cos_sim_mean": float(sims.mean()), "n_bad": n_bad,
            "cpu_gpu_util_mean": cpu_gpu_stats["mean"], "gpu_gpu_util_mean": gpu_gpu_stats["mean"],
            "cpu_cpu_util_mean": cpu_cpu_stats["mean"], "gpu_cpu_util_mean": gpu_cpu_stats["mean"],
        })

    print("=== Summary: gpu_normalize vs cpu_normalize speedup, by flight level ===")
    for r in rows:
        verdict = "GAIN" if r["speedup"] > 1.02 else ("REGRESSION" if r["speedup"] < 0.98 else "NO CHANGE")
        print(f"  k={r['k']}: {r['speedup']:.2f}x  ({verdict})  "
              f"GPU util {r['cpu_gpu_util_mean']}%->{r['gpu_gpu_util_mean']}%  "
              f"CPU util {r['cpu_cpu_util_mean']}%->{r['gpu_cpu_util_mean']}%  "
              f"correctness: {r['n_bad']}/{n*8} pairs below threshold")


if __name__ == "__main__":
    main()
