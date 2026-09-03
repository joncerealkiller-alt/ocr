"""
Stage 1 of the GPU-preprocessing characterization (2026-08-04):
per-operation timing of the REAL preprocessing pipeline, not an
assumed one. Every one of the 8 QUALIFIED_ENCODERS' timm-built
transforms was directly inspected (torchvision Compose repr) before
writing this script - all 8 are structurally identical 4-step
pipelines (Resize -> CenterCrop -> MaybeToTensor -> Normalize), only
differing in target size/interpolation and mean/std. Decode (Image.
open + convert("RGB")) happens ONCE per image in the current sharded
implementation (core/auto_sidecar... no - this measures core/
vision_embeddings.py's embed_pooled() pipeline as actually used by
benchmark/gpu_sharded_towers_experiment.py); the transform itself runs
ONCE PER ENCODER PER IMAGE (8x), since target size/normalization
differ per encoder - confirmed by direct inspection this session, not
assumed.

Read-only, no model inference beyond what's needed to build each
encoder's real transform object (timm.create_model is required to
call resolve_data_config/create_transform - no forward pass here).

Usage:
    python -m benchmark.gpu_preprocessing_stage1_characterization
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from torchvision import transforms as T

from core.vision_embeddings import QUALIFIED_ENCODERS, build_model_and_transform

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_SIZE = 30
N_TIMING_REPS = 5  # repeat each op this many times per image, report mean


def _sample_images(n: int) -> list[Path]:
    all_images = sorted(
        p for p in (PROJECT_ROOT / "data" / "working").iterdir()
        if p.is_file() and not p.name.endswith(".json")
    )
    step = max(1, len(all_images) // n)
    return all_images[::step][:n]


def _time_op(fn, reps: int) -> float:
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def main() -> None:
    images = _sample_images(SAMPLE_SIZE)
    n = len(images)
    print(f"Characterizing preprocessing over {n} images.\n")

    # -- Decode: once per image, shared across all 8 encoders --
    decode_times = []
    decoded_cache = {}
    for p in images:
        t0 = time.perf_counter()
        with Image.open(p) as img:
            rgb = img.convert("RGB")
            rgb.load()  # force full decode, not lazy
        decode_times.append(time.perf_counter() - t0)
        with Image.open(p) as img:
            decoded_cache[p] = img.convert("RGB")
    print(f"Decode (Image.open + convert('RGB')), PIL, CPU, PER IMAGE (once, shared across encoders):")
    print(f"  mean={np.mean(decode_times)*1000:.2f}ms  median={np.median(decode_times)*1000:.2f}ms  "
          f"max={np.max(decode_times)*1000:.2f}ms")
    print(f"  total for {n} images: {sum(decode_times)*1000:.1f}ms\n")

    # -- Per-encoder transform sub-steps --
    print(f"{'encoder':<15}{'resize(ms)':<13}{'crop(ms)':<11}{'to_tensor(ms)':<15}{'normalize(ms)':<15}{'total(ms)':<12}")
    per_encoder_totals = {}
    grand_decode_total = sum(decode_times)
    grand_transform_total = 0.0

    for encoder_name, checkpoint in QUALIFIED_ENCODERS:
        model, transform = build_model_and_transform(checkpoint)
        # transform is a torchvision Compose - pull out its individual stages
        stages = list(transform.transforms)
        resize_t, crop_t, totensor_t, normalize_t = stages[0], stages[1], stages[2], stages[3]

        resize_times, crop_times, totensor_times, normalize_times = [], [], [], []
        for p in images:
            rgb = decoded_cache[p]
            resize_times.append(_time_op(lambda: resize_t(rgb), N_TIMING_REPS))
            resized = resize_t(rgb)
            crop_times.append(_time_op(lambda: crop_t(resized), N_TIMING_REPS))
            cropped = crop_t(resized)
            totensor_times.append(_time_op(lambda: totensor_t(cropped), N_TIMING_REPS))
            tensor = totensor_t(cropped)
            normalize_times.append(_time_op(lambda: normalize_t(tensor.clone()), N_TIMING_REPS))

        r, c, tt, nm = np.mean(resize_times), np.mean(crop_times), np.mean(totensor_times), np.mean(normalize_times)
        total = r + c + tt + nm
        per_encoder_totals[encoder_name] = total
        grand_transform_total += total * n
        print(f"{encoder_name:<15}{r*1000:<13.3f}{c*1000:<11.3f}{tt*1000:<15.3f}{nm*1000:<15.3f}{total*1000:<12.3f}")
        del model, transform

    print(f"\nTotal decode time for {n} images (once, shared): {grand_decode_total*1000:.1f}ms")
    print(f"Total transform time for {n} images x 8 encoders: {grand_transform_total*1000:.1f}ms")
    print(f"Ratio - transform-work : decode-work = {grand_transform_total/grand_decode_total:.1f} : 1")

    print("\n=== Operation table ===")
    print(f"{'Operation':<20}{'Device':<8}{'Library':<14}{'Per-image':<11}{'Per-encoder':<13}{'GPU candidate?'}")
    print(f"{'Decode':<20}{'CPU':<8}{'PIL':<14}{'yes':<11}{'no (shared)':<13}{'no - excluded per scope'}")
    print(f"{'Resize':<20}{'CPU':<8}{'torchvision':<14}{'no':<11}{'yes':<13}{'yes'}")
    print(f"{'CenterCrop':<20}{'CPU':<8}{'torchvision':<14}{'no':<11}{'yes':<13}{'yes (trivial, fused w/ resize candidate)'}")
    print(f"{'MaybeToTensor':<20}{'CPU':<8}{'torchvision':<14}{'no':<11}{'yes':<13}{'yes'}")
    print(f"{'Normalize':<20}{'CPU':<8}{'torchvision':<14}{'no':<11}{'yes':<13}{'yes'}")


if __name__ == "__main__":
    main()
