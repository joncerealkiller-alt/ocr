"""
Images-in-flight sweep (2026-08-04) - the follow-up to benchmark/
gpu_sharded_towers_experiment.py's validated "8 encoders resident,
per-encoder streams" result. That version is still strictly serial
ACROSS images: dispatch image N's 8 streams, synchronize, THEN start
decoding image N+1 - CPU-side decode/transform for the next image never
overlaps with the GPU still finishing the current one. This sweeps how
many images are allowed in flight simultaneously (1/2/3/4...) and
measures BOTH images/sec AND real GPU utilization (nvidia-smi, sampled
throughout each configuration, not assumed) to find the plateau - the
point where adding another concurrent image stops buying real
throughput, per Jon's own framing of what this measurement is for.

CORRECTNESS-CRITICAL DESIGN NOTE: each concurrent image gets its OWN
dedicated set of 8 CUDA streams, synchronized individually via
stream.synchronize() (not the global torch.cuda.synchronize(), which
would block a thread on every OTHER in-flight image's streams too, not
just its own - a real correctness/performance trap for this specific
experiment that a naive port of the single-image prototype would fall
into). Models are shared read-only across worker threads (safe - no
weight mutation, no .train() mode).

Small sample, not full corpus - this is exploratory tuning to find the
plateau, matching this project's own "measure on a sample, decide, then
scale" discipline used throughout this session.

Usage:
    python -m benchmark.gpu_images_in_flight_sweep
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
    """nvidia-smi subprocess sampling, not torch.cuda in-process (same
    reasoning as the Phase B/Phase-performance telemetry earlier this
    session: reports real device-wide utilization, and never risks a
    second CUDA context)."""
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
            return {"mean": None, "max": None, "n_samples": 0}
        return {
            "mean": round(sum(self._samples) / len(self._samples), 1),
            "max": round(max(self._samples), 1),
            "n_samples": len(self._samples),
        }


class _CpuUtilMonitor:
    """psutil.cpu_percent(interval=...) sampled on its own thread,
    alongside the GPU monitor - answers whether CPU-side decode/
    transform work (which scales with images-in-flight, unlike the GPU
    compute itself) becomes the limiting factor at higher concurrency,
    not just whether the GPU is saturated."""
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
            return {"mean": None, "max": None, "n_samples": 0}
        return {
            "mean": round(sum(self._samples) / len(self._samples), 1),
            "max": round(max(self._samples), 1),
            "n_samples": len(self._samples),
        }


@torch.no_grad()
def _process_one_image(p: Path, models: dict, stream_set: dict) -> dict:
    """Uses ONLY this call's own dedicated stream_set - never the global
    torch.cuda.synchronize(), so concurrent calls (different images, on
    their own stream_sets) don't block on each other's work."""
    with Image.open(p) as img:
        rgb = img.convert("RGB")
        results = {}
        for encoder_name, (model, transform) in models.items():
            stream = stream_set[encoder_name]
            with torch.cuda.stream(stream):
                x = transform(rgb).unsqueeze(0).to("cuda", non_blocking=True)
                results[encoder_name] = model(x).squeeze(0)
        for stream in stream_set.values():
            stream.synchronize()  # only THIS image's streams, not the whole device
        return {name: t.cpu().numpy() for name, t in results.items()}


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    images = _sample_images(SAMPLE_SIZE)
    n = len(images)
    print(f"Sweeping images-in-flight over {n} images.\n")

    print("Loading all 8 encoders resident on GPU (shared across all flight levels)...")
    models = {}
    for encoder_name, checkpoint in QUALIFIED_ENCODERS:
        models[encoder_name] = _build_model_and_transform(checkpoint)
    print(f"Resident. VRAM: {torch.cuda.memory_allocated()/1e6:.1f}MB\n")

    reference_out = None
    print(f"{'in-flight':<11}{'elapsed(s)':<12}{'img/s':<9}"
          f"{'GPU mean%':<11}{'GPU max%':<10}{'CPU mean%':<11}{'CPU max%':<10}")
    results_table = []

    for k in FLIGHT_LEVELS:
        # One dedicated stream-set PER concurrent slot, reused round-robin
        # across images at that concurrency level - avoids creating
        # thousands of stream objects while still giving every
        # simultaneously-in-flight image its own isolated set.
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
                executor.submit(_process_one_image, p, models, stream_sets[i % k]): p
                for i, p in enumerate(images)
            }
            for future in futures:
                p = futures[future]
                out[str(p)] = future.result()
        elapsed = time.time() - t0
        gpu_stats = gpu_monitor.stop()
        cpu_stats = cpu_monitor.stop()

        img_per_sec = n / elapsed if elapsed else 0
        print(f"{k:<11}{elapsed:<12.1f}{img_per_sec:<9.3f}"
              f"{str(gpu_stats['mean']):<11}{str(gpu_stats['max']):<10}"
              f"{str(cpu_stats['mean']):<11}{str(cpu_stats['max']):<10}")
        results_table.append((k, elapsed, img_per_sec, gpu_stats, cpu_stats))

        if k == 1:
            reference_out = out
        else:
            sims = []
            for path_str, ref_vecs in reference_out.items():
                for encoder_name, ref_vec in ref_vecs.items():
                    sims.append(cosine_sim(
                        ref_vec.astype(np.float64), out[path_str][encoder_name].astype(np.float64)))
            sims = np.array(sims)
            n_bad = int((sims < 0.999999).sum())
            print(f"    correctness vs k=1: mean cos_sim={sims.mean():.8f}, "
                  f"{n_bad}/{len(sims)} pairs below 0.999999")

    print("\nSpeedup vs images-in-flight=1:")
    base_elapsed = results_table[0][1]
    for k, elapsed, img_per_sec, gpu_stats, cpu_stats in results_table:
        print(f"  k={k}: {base_elapsed/elapsed:.2f}x  "
              f"(GPU util mean {gpu_stats['mean']}%, CPU util mean {cpu_stats['mean']}%)")


if __name__ == "__main__":
    main()
