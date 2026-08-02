"""
Benchmark 2 (Metadata Layer Qualification) - real sequential runtime
measurement across all 8 candidates qualified in Benchmark 1. Answers
the one empirical question Benchmark 1's per-encoder operational
metadata didn't: what does it actually cost to run every qualified
encoder, one after another (never concurrently, per CLAUDE.md's GPU
discipline), over the same real image set, including model LOAD time,
not just per-image inference?

Reuses the same 24-image sample (6 buckets x 4) used throughout Rounds
1-9 for continuity with Benchmark 1's own numbers.

Usage:
    python -m benchmark.benchmark2_sequential_runtime
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import timm
from PIL import Image

from benchmark.vision_encoder_qualification import sample_real_images, embed_pooled

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_LOG = PROJECT_ROOT / "data" / "outputs" / "benchmark2_sequential_runtime_log.jsonl"

CANDIDATES = [
    ("dinov2", "vit_small_patch14_dinov2.lvd142m"),
    ("convnext", "convnext_tiny.fb_in22k"),
    ("naflex_siglip", "naflexvit_base_patch16_siglip.v2_webli"),
    ("siglip_fixed", "vit_base_patch16_siglip_224.v2_webli"),
    ("eva02", "eva02_base_patch14_224.mim_in22k"),
    ("beit", "beit_base_patch16_224.in22k_ft_in22k"),
    ("swin", "swin_base_patch4_window7_224.ms_in22k"),
    ("mae", "vit_base_patch16_224.mae"),
]


def main():
    clusters = sample_real_images()
    images = []
    for bucket, paths in clusters.items():
        for p in paths:
            images.append(Image.open(p).convert("RGB"))
    n_images = len(images)
    print(f"Sequential run: {len(CANDIDATES)} encoders x {n_images} real images\n")

    per_encoder_results = []
    total_wall_start = time.perf_counter()

    for name, tag in CANDIDATES:
        load_start = time.perf_counter()
        model = timm.create_model(tag, pretrained=True, num_classes=0)
        model.eval()
        cfg = timm.data.resolve_data_config({}, model=model)
        transform = timm.data.create_transform(**cfg)
        if torch.cuda.is_available():
            model = model.cuda()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        load_time = time.perf_counter() - load_start

        device = "cuda" if torch.cuda.is_available() else "cpu"
        infer_start = time.perf_counter()
        for img in images:
            x = transform(img).unsqueeze(0)
            if device == "cuda":
                x = x.cuda()
            with torch.no_grad():
                model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        infer_time = time.perf_counter() - infer_start

        peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else None
        images_per_sec = n_images / infer_time if infer_time > 0 else None

        result = {
            "candidate": name, "tag": tag,
            "load_time_s": round(load_time, 3),
            "inference_time_s_for_batch": round(infer_time, 3),
            "images_per_sec": round(images_per_sec, 2) if images_per_sec else None,
            "peak_gpu_memory_mb": round(peak_mem_mb, 1) if peak_mem_mb else None,
        }
        per_encoder_results.append(result)
        print(f"{name:<15} load={load_time:>6.2f}s  infer({n_images}imgs)={infer_time:>6.2f}s  "
              f"({images_per_sec:.1f} img/s)  peak_mem={peak_mem_mb:.0f}MB" if peak_mem_mb else
              f"{name:<15} load={load_time:>6.2f}s  infer({n_images}imgs)={infer_time:>6.2f}s")

        # release before loading the next model - never concurrent, matching
        # this project's established GPU discipline (CLAUDE.md)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_wall_time = time.perf_counter() - total_wall_start
    total_load_time = sum(r["load_time_s"] for r in per_encoder_results)
    total_infer_time = sum(r["inference_time_s_for_batch"] for r in per_encoder_results)
    per_image_all_encoders_s = total_infer_time / n_images
    max_peak_mem = max((r["peak_gpu_memory_mb"] for r in per_encoder_results if r["peak_gpu_memory_mb"]), default=None)

    print(f"\n=== TOTALS (all {len(CANDIDATES)} encoders, sequential, {n_images} images) ===")
    print(f"Total load time (one-time, all 8 models): {total_load_time:.2f}s")
    print(f"Total inference time (all 8 x {n_images} images): {total_infer_time:.2f}s")
    print(f"Per-image cost across all 8 encoders combined: {per_image_all_encoders_s:.3f}s/image")
    print(f"Peak GPU memory (max across encoders, sequential so never summed): {max_peak_mem:.0f}MB")
    print(f"Total wall-clock time for this whole run: {total_wall_time:.2f}s")

    RESULTS_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment": "benchmark2_sequential_runtime", "n_images": n_images,
        "per_encoder": per_encoder_results,
        "total_load_time_s": round(total_load_time, 2),
        "total_inference_time_s": round(total_infer_time, 2),
        "per_image_cost_all_encoders_s": round(per_image_all_encoders_s, 4),
        "peak_gpu_memory_mb_max_across_encoders": round(max_peak_mem, 1) if max_peak_mem else None,
        "total_wall_clock_s": round(total_wall_time, 2),
    }
    with open(RESULTS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"\nAppended full record to {RESULTS_LOG}")


if __name__ == "__main__":
    main()
