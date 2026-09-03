"""
Stage 1 physical-sensor concurrency prototype (2026-08-04) - the
"images in flight" experiment discussed earlier this session, now
actually measured against core/image_analysis.py's real
analyze_image(), the first Stage 1 workload with real, measured cost
data to prototype against (~3.7s/image, pure CPU, confirmed via direct
timing before this file existed - see docs/PIPELINE_DATABASE.md's
"Physical sensor wired into the automatic chain" section).

DIFFERENT threading model from the earlier vision-tower discussion:
that was PyTorch (MKL/oneDNN intra-op parallelism, torch.get_num_
threads()). This is OpenCV (cv2.setNumThreads(), a "Concurrency"
framework backend on this machine - confirmed via cv2.getBuildInfo(),
not assumed). Same underlying tuning TENSION (intra-op thread count per
call vs. how many calls run at once can oversubscribe the same cores),
different API to control it - conflating the two would be a real
mistake, not just a style difference.

Measures wall-clock throughput (images/sec) AND CPU utilization
(psutil) per configuration - elapsed time alone can hide whether a
"faster" config is actually using more cores efficiently or just
avoiding idle time that intra-op parallelism already claimed, per the
benchmarking-methodology discussion earlier this session.

No production code changed by this file. Read-only against real corpus
images (analyze_image() never modifies pixels or writes anything by
itself - only save_analysis() would, and this script never calls it).

Usage:
    python -m benchmark.stage1_concurrency_experiment
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import cv2
import psutil

from core.image_analysis import analyze_image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_SIZE = 24  # per configuration - keeps total runtime reasonable
                  # (~24 x 3.7s baseline ~= 90s/config x ~6 configs ~= 9 min)


def _sample_images(n: int) -> list[Path]:
    all_images = sorted(
        p for p in (PROJECT_ROOT / "data" / "working").iterdir()
        if p.is_file() and not p.name.endswith(".json")
    )
    # Fixed stride sample (not random) so every configuration measures
    # the EXACT same images - eliminates "one config got easier images"
    # as a confound between runs.
    step = max(1, len(all_images) // n)
    return all_images[::step][:n]


def _run_sequential(images: list[Path], cv2_threads: int) -> dict:
    cv2.setNumThreads(cv2_threads)
    cpu_before = psutil.cpu_percent(interval=None)  # prime, discard
    t0 = time.time()
    for p in images:
        try:
            analyze_image(p)
        except Exception:
            pass
    elapsed = time.time() - t0
    cpu = psutil.cpu_percent(interval=None)
    return {"elapsed": elapsed, "cpu_percent": cpu}


def _analyze_one(path_str: str, cv2_threads: int) -> bool:
    """Module-level (picklable) worker for ProcessPoolExecutor - each
    subprocess is a fresh interpreter, so cv2_threads must be set
    INSIDE the worker, not inherited from the parent's setting."""
    cv2.setNumThreads(cv2_threads)
    try:
        analyze_image(Path(path_str))
        return True
    except Exception:
        return False


def _run_threaded(images: list[Path], n_workers: int, cv2_threads_per_worker: int) -> dict:
    cv2.setNumThreads(cv2_threads_per_worker)  # process-global setting, shared by all threads
    cpu_before = psutil.cpu_percent(interval=None)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        list(ex.map(lambda p: analyze_image(p), images))
    elapsed = time.time() - t0
    cpu = psutil.cpu_percent(interval=None)
    return {"elapsed": elapsed, "cpu_percent": cpu}


def _run_multiprocess(images: list[Path], n_workers: int, cv2_threads_per_worker: int) -> dict:
    cpu_before = psutil.cpu_percent(interval=None)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        list(ex.map(_analyze_one, [str(p) for p in images], [cv2_threads_per_worker] * len(images)))
    elapsed = time.time() - t0
    cpu = psutil.cpu_percent(interval=None)
    return {"elapsed": elapsed, "cpu_percent": cpu}


def main() -> None:
    images = _sample_images(SAMPLE_SIZE)
    n = len(images)
    print(f"Benchmarking {n} images per configuration (fixed sample, identical across all configs).\n")

    logical_cpus = psutil.cpu_count(logical=True)
    print(f"Logical CPUs: {logical_cpus}, cv2 default thread count: {cv2.getNumThreads()}\n")

    configs = [
        ("sequential (baseline, current production behavior)",
         lambda: _run_sequential(images, cv2_threads=logical_cpus)),
        ("sequential, cv2 threads=1 (intra-op OFF entirely)",
         lambda: _run_sequential(images, cv2_threads=1)),
        ("threaded x4, cv2 threads=8/worker",
         lambda: _run_threaded(images, n_workers=4, cv2_threads_per_worker=logical_cpus // 4)),
        ("threaded x8, cv2 threads=4/worker",
         lambda: _run_threaded(images, n_workers=8, cv2_threads_per_worker=logical_cpus // 8)),
        ("threaded x8, cv2 threads=1/worker (max inter-op)",
         lambda: _run_threaded(images, n_workers=8, cv2_threads_per_worker=1)),
        ("multiprocess x4, cv2 threads=8/worker",
         lambda: _run_multiprocess(images, n_workers=4, cv2_threads_per_worker=logical_cpus // 4)),
        ("multiprocess x8, cv2 threads=4/worker",
         lambda: _run_multiprocess(images, n_workers=8, cv2_threads_per_worker=logical_cpus // 8)),
    ]

    print(f"{'configuration':<50}{'elapsed(s)':<12}{'img/s':<10}{'cpu%':<8}")
    results = []
    for name, fn in configs:
        r = fn()
        img_per_sec = n / r["elapsed"] if r["elapsed"] else 0
        results.append((name, r["elapsed"], img_per_sec, r["cpu_percent"]))
        print(f"{name:<50}{r['elapsed']:<12.1f}{img_per_sec:<10.2f}{r['cpu_percent']:<8.1f}")

    baseline_elapsed = results[0][1]
    print("\nSpeedup vs. sequential baseline:")
    for name, elapsed, img_per_sec, cpu in results:
        print(f"  {name:<50} {baseline_elapsed/elapsed:.2f}x")


if __name__ == "__main__":
    main()
