"""
Phase A of the full-corpus CPU vs GPU equivalence characterization
(2026-08-04) - see docs/GPU_CPU_EQUIVALENCE_REPORT.md once written for
the full context. Generates a COMPLETE embedding set for the current
corpus using the existing, unmodified CPU COMPUTATION (core/vision_
embeddings.py's own build_model_and_transform()/embed_pooled(), called
exactly as production does - nothing about the model, weights, or
per-call precision differs). The only thing this changes vs. core/
baseline_embeddings.py's own capture_baseline_embeddings() is
SCHEDULING - threaded x4, 4 intra-op threads/worker, the config this
same session's benchmark/baseline_embeddings_concurrency_experiment.py
measured (1.31x speedup, 128/128 CPU-vs-GPU embedding pairs at cosine
similarity 1.0 in that smaller-sample check). Scheduling-only changes
don't alter computed values - same principle already validated for the
physical sensor earlier this session - so this remains a fair "CPU
implementation" baseline for the equivalence study, not a different
implementation.

Output: baseline_embeddings_cpu.json (NOT data/baseline_embeddings.json
- the production default path is never touched by this experiment) plus
a companion baseline_embeddings_cpu.meta.json with run-level stats
(elapsed, images/sec, CPU utilization, library versions, concurrency
config) that don't belong in the per-record embedding schema itself.

Read-only against the corpus otherwise - no sidecars written, no DB
writes, core/vision_embeddings.py is imported and used exactly as it
exists in production (not modified, not reimplemented).

Usage:
    python -m benchmark.gpu_cpu_equivalence_phaseA_cpu
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import psutil
import timm
import torch
from PIL import Image

from core.baseline_embeddings import PREPROCESSING_STAGE
from core.bucket_worklist import load_bucket_filepaths
from core.vision_embeddings import QUALIFIED_ENCODERS, build_model_and_transform, embed_pooled

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "outputs" / "gpu_cpu_equivalence" / "baseline_embeddings_cpu.json"
META_PATH = PROJECT_ROOT / "data" / "outputs" / "gpu_cpu_equivalence" / "baseline_embeddings_cpu.meta.json"

CONCURRENCY_WORKERS = 4
INTRA_OP_THREADS_PER_WORKER = 4


def _image_hash(image_path: Path) -> str:
    return hashlib.sha256(image_path.read_bytes()).hexdigest()


class _CpuMonitor:
    """Lightweight background sampler - psutil.cpu_percent(interval=1)
    once per second for the run's duration, so Phase A's own metadata
    reports a real measured mean/max rather than a single before/after
    snapshot that could miss transient spikes."""
    def __init__(self):
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self._samples.append(psutil.cpu_percent(interval=1))

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


def main() -> None:
    manifest_path = PROJECT_ROOT / "data" / "manifest.csv"
    image_paths = [Path(p) for p in load_bucket_filepaths(manifest_path)]
    n = len(image_paths)
    print(f"Phase A: capturing CPU baseline embeddings for {n} images "
          f"(threaded x{CONCURRENCY_WORKERS}, {INTRA_OP_THREADS_PER_WORKER} intra-op threads/worker - "
          f"same computation as production, scheduling-only change).")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(INTRA_OP_THREADS_PER_WORKER)

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

    monitor = _CpuMonitor()
    monitor.start()
    t0 = time.time()
    tick = max(1, n // 20)

    for encoder_i, (encoder_name, checkpoint) in enumerate(QUALIFIED_ENCODERS, 1):
        print(f"[encoder {encoder_i}/{len(QUALIFIED_ENCODERS)}] {encoder_name} ({checkpoint}): "
              f"embedding {n} image(s) (threaded)...")
        model, transform = build_model_and_transform(checkpoint)

        def _one(p: Path):
            with Image.open(p) as img:
                return p, embed_pooled(model, transform, img.convert("RGB"))

        with ThreadPoolExecutor(max_workers=CONCURRENCY_WORKERS) as executor:
            for img_i, (p, vector) in enumerate(executor.map(_one, image_paths), 1):
                per_image[p]["embeddings"][encoder_name] = {
                    "checkpoint": checkpoint,
                    "vector": [round(float(v), 6) for v in vector],
                }
                if img_i % tick == 0 or img_i == n:
                    print(f"    {img_i}/{n}")
        del model, transform
        print(f"  done: {encoder_name}")

    elapsed = time.time() - t0
    cpu_stats = monitor.stop()

    records = list(per_image.values())
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f)

    meta = {
        "phase": "A_cpu",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_images": n,
        "elapsed_seconds": round(elapsed, 1),
        "images_per_second": round(n / elapsed, 4) if elapsed else None,
        "cpu_utilization_percent": cpu_stats,
        "concurrency": {"workers": CONCURRENCY_WORKERS, "intra_op_threads_per_worker": INTRA_OP_THREADS_PER_WORKER},
        "torch_version": torch.__version__,
        "timm_version": timm.__version__,
        "embedding_format_version": "baseline_embeddings_v1 (per-record: image, image_hash, "
                                     "preprocessing_stage, timestamp, library_versions, embeddings)",
        "output_path": str(OUTPUT_PATH),
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone: {n} images, {elapsed:.1f}s ({elapsed/60:.1f} min), "
          f"{n/elapsed:.3f} img/s, CPU util mean={cpu_stats['mean']}% max={cpu_stats['max']}%")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Meta:   {META_PATH}")


if __name__ == "__main__":
    main()
