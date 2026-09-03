"""
Stages 3-5 of the GPU-preprocessing characterization (2026-08-04):
individual prototypes, each compared against the current CPU-only
production pipeline at the EMBEDDING level (not just raw pixels) -
Stage 1 found Resize is ~95% of transform cost; a standalone pixel-level
check found PIL-CPU-bicubic and PyTorch-GPU-bicubic are NOT numerically
identical (mean abs diff 0.31, max 12.16 in 0-255 space). This measures
whether that pixel difference survives into the actual embeddings and
downstream decisions, or gets absorbed by the network - not assumed
either way.

Three configurations, each isolated (never combined, per the explicit
instruction not to test multiple changes at once):

  baseline        - current production pipeline exactly (PIL resize/
                    crop/totensor/normalize, all CPU)
  gpu_normalize   - ONLY normalize moves to GPU (near-zero pixel risk -
                    pure elementwise arithmetic, no interpolation)
  gpu_resize_crop - ONLY resize+crop move to GPU (structurally coupled -
                    crop must happen wherever resize's output already
                    lives; normalize stays CPU in this config)

core/vision_embeddings.py is NOT modified - all GPU code is local to
this file.

Usage:
    python -m benchmark.gpu_preprocessing_stage34_prototypes
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image

from core.vision_embeddings import QUALIFIED_ENCODERS, build_model_and_transform, cosine_sim

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_SIZE = 30


def _sample_images(n: int) -> list[Path]:
    all_images = sorted(
        p for p in (PROJECT_ROOT / "data" / "working").iterdir()
        if p.is_file() and not p.name.endswith(".json")
    )
    step = max(1, len(all_images) // n)
    return all_images[::step][:n]


@torch.no_grad()
def _embed_baseline(model, transform, rgb: Image.Image) -> np.ndarray:
    """Exactly the current production path - core/vision_embeddings.py's
    embed_pooled(), reproduced here (not imported) only so its timing
    can be isolated per-stage inside this comparison; the computation
    itself is identical."""
    x = transform(rgb).unsqueeze(0)
    return model(x).squeeze(0).numpy()


@torch.no_grad()
def _embed_gpu_normalize(model_gpu, stages: list, rgb: Image.Image, mean_t, std_t) -> np.ndarray:
    """resize/crop/totensor stay CPU (PIL/torchvision, unchanged);
    ONLY normalize happens on GPU."""
    resize_t, crop_t, totensor_t, _normalize_t = stages
    x = totensor_t(crop_t(resize_t(rgb))).unsqueeze(0).cuda()
    x = (x - mean_t) / std_t
    return model_gpu(x).squeeze(0).cpu().numpy()


@torch.no_grad()
def _embed_gpu_resize_crop(model_cpu, stages: list, rgb: Image.Image, crop_size: int) -> np.ndarray:
    """Isolates JUST resize+crop on GPU, per the "one variable at a
    time" rule - model inference and normalize both stay exactly where
    the baseline has them (CPU), so model_cpu is a CPU-resident model,
    never moved. Decode stays CPU; full-size image tensorized on CPU
    (uint8->float, HWC->CHW - unavoidable to get it onto the GPU at
    all), THEN resize+crop happen on GPU, THEN the result comes back to
    CPU for normalize+inference exactly as the baseline does them."""
    _resize_t, _crop_t, _totensor_t, normalize_t = stages
    arr = np.array(rgb, dtype=np.float32)  # HWC, 0-255
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).cuda() / 255.0
    target = _resize_t.size if isinstance(_resize_t.size, int) else _resize_t.size[0]
    x = F.interpolate(x, size=(target, target), mode="bicubic", align_corners=False, antialias=True)
    # center crop on GPU
    _, _, h, w = x.shape
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    x = x[:, :, top:top + crop_size, left:left + crop_size]
    x = x.cpu()
    x = normalize_t(x.squeeze(0)).unsqueeze(0)
    return model_cpu(x).squeeze(0).numpy()


def _compare(name: str, baseline_vecs: dict, other_vecs: dict) -> dict:
    sims, l2s, maxabs = [], [], []
    for key, base_v in baseline_vecs.items():
        other_v = other_vecs[key]
        b, o = base_v.astype(np.float64), other_v.astype(np.float64)
        sims.append(cosine_sim(b, o))
        l2s.append(float(np.linalg.norm(b - o)))
        maxabs.append(float(np.max(np.abs(b - o))))
    sims, l2s, maxabs = np.array(sims), np.array(l2s), np.array(maxabs)
    print(f"\n--- {name} vs baseline: embedding-level equivalence ({len(sims)} pairs) ---")
    print(f"  cosine similarity: min={sims.min():.8f} mean={sims.mean():.8f} max={sims.max():.8f}")
    print(f"  L2 distance:       min={l2s.min():.6f} mean={l2s.mean():.6f} max={l2s.max():.6f}")
    print(f"  max abs elem diff: min={maxabs.min():.6f} mean={maxabs.mean():.6f} max={maxabs.max():.6f}")
    n_outliers = int((sims < 0.9999).sum())
    print(f"  pairs with cosine similarity < 0.9999: {n_outliers} / {len(sims)}")
    return {"cosine_sim": sims, "l2": l2s, "max_abs": maxabs}


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    images = _sample_images(SAMPLE_SIZE)
    n = len(images)
    print(f"Testing {n} images x {len(QUALIFIED_ENCODERS)} encoders = {n*len(QUALIFIED_ENCODERS)} pairs.\n")

    decoded = {}
    for p in images:
        with Image.open(p) as img:
            decoded[p] = img.convert("RGB")

    baseline_vecs, gpu_norm_vecs, gpu_resize_vecs = {}, {}, {}
    t_baseline = t_gpu_norm = t_gpu_resize = 0.0

    for encoder_name, checkpoint in QUALIFIED_ENCODERS:
        print(f"[{encoder_name}]")
        model_cpu, transform = build_model_and_transform(checkpoint)
        stages = list(transform.transforms)
        mean_t = stages[3].mean.view(1, 3, 1, 1).cuda()
        std_t = stages[3].std.view(1, 3, 1, 1).cuda()
        crop_size = stages[1].size[0]

        t0 = time.time()
        for p in images:
            baseline_vecs[(str(p), encoder_name)] = _embed_baseline(model_cpu, transform, decoded[p])
        t_baseline += time.time() - t0

        model_gpu = build_model_and_transform(checkpoint)[0].cuda()
        t0 = time.time()
        for p in images:
            gpu_norm_vecs[(str(p), encoder_name)] = _embed_gpu_normalize(model_gpu, stages, decoded[p], mean_t, std_t)
        t_gpu_norm += time.time() - t0
        del model_gpu

        t0 = time.time()
        for p in images:
            gpu_resize_vecs[(str(p), encoder_name)] = _embed_gpu_resize_crop(model_cpu, stages, decoded[p], crop_size)
        t_gpu_resize += time.time() - t0

        del model_cpu, transform
        torch.cuda.empty_cache()

    print(f"\n=== Timing (total across all {n} images x 8 encoders) ===")
    print(f"baseline (all CPU):        {t_baseline:.1f}s  ({n*len(QUALIFIED_ENCODERS)/t_baseline:.2f} pairs/s)")
    print(f"gpu_normalize:              {t_gpu_norm:.1f}s  ({n*len(QUALIFIED_ENCODERS)/t_gpu_norm:.2f} pairs/s)  "
          f"speedup {t_baseline/t_gpu_norm:.2f}x")
    print(f"gpu_resize_crop:            {t_gpu_resize:.1f}s  ({n*len(QUALIFIED_ENCODERS)/t_gpu_resize:.2f} pairs/s)  "
          f"speedup {t_baseline/t_gpu_resize:.2f}x")

    _compare("gpu_normalize", baseline_vecs, gpu_norm_vecs)
    _compare("gpu_resize_crop", baseline_vecs, gpu_resize_vecs)


if __name__ == "__main__":
    main()
