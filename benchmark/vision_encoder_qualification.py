"""
Vision encoder qualification battery (Architecture D research,
see docs/VISION_IR_RESEARCH.md's "Experiment protocol" section for the
full design rationale). One encoder per round, per Jon's sequencing
(2026-07-31) - CANDIDATE is the only thing that changes between rounds.

Reuses this pipeline's own real preprocessing functions
(core/image_preprocessing.py, core/row_segmentation.py) as the
transform battery, not synthetic approximations of what the pipeline
does.

BATTERY_VERSION is frozen as of Round 1 (2026-07-31, DINOv2-small) -
per Jon's direction, every candidate from here on runs the IDENTICAL
protocol (same TRANSFORMS, same TOP_K, same metrics) so results stay
directly comparable across rounds. Changing TRANSFORMS, TOP_K, or the
metric definitions is a breaking change to the battery itself, not a
tweak - bump BATTERY_VERSION and note the change explicitly in
docs/VISION_IR_RESEARCH.md if it's ever genuinely necessary, rather
than editing this file's protocol in place and losing comparability
with Round 1's numbers.

Usage:
    python -m benchmark.vision_encoder_qualification
"""
from __future__ import annotations

BATTERY_VERSION = "1.0"

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import timm
from PIL import Image

from core.image_preprocessing import enhance_contrast, autocontrast, denoise, invert, upscale
from core.row_segmentation import apply_deskew_angle
from core.vision_embeddings import (
    build_model_and_transform, _spatial_layout, embed_pooled, embed_patch_mean, cosine_sim,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUCKET_DIR = PROJECT_ROOT / "data" / "buckets"
RESULTS_LOG = PROJECT_ROOT / "data" / "outputs" / "vision_encoder_qualification_log.jsonl"

CANDIDATE = "vit_base_patch16_224.mae"   # Round 9
ROUND_LABEL = "round_9_mae_vit_base"

CLUSTER_BUCKETS = [
    "dense_tabular_rows", "printed_document", "map_land_record",
    "portrait_photo", "genealogy_chart", "handwritten_ledger",
]
IMAGES_PER_CLUSTER = 4
TOP_K = 5


def _crop_border_pct(img: Image.Image, pct: float) -> Image.Image:
    w, h = img.size
    dx, dy = int(w * pct), int(h * pct)
    return img.crop((dx, dy, w - dx, h - dy))


TRANSFORMS = {
    "deskew_+2.0deg": lambda img: apply_deskew_angle(img, 2.0),
    "deskew_-2.0deg": lambda img: apply_deskew_angle(img, -2.0),
    "deskew_+0.5deg": lambda img: apply_deskew_angle(img, 0.5),
    "contrast_enhance_1.5x": lambda img: enhance_contrast(img, 1.5),
    "contrast_autocontrast": lambda img: autocontrast(img, 1.0),
    "denoise_median3": lambda img: denoise(img),
    "invert": lambda img: invert(img),
    "downscale_0.5x": lambda img: img.resize((max(1, img.width // 2), max(1, img.height // 2)), Image.LANCZOS),
    "upscale_2x": lambda img: upscale(img, 2.0),
    "border_crop_5pct": lambda img: _crop_border_pct(img, 0.05),
}


def sample_real_images(images_per_cluster: int | None = None) -> dict[str, list[Path]]:
    """Real file_path values straight from the bucket CSVs this pipeline
    already produced - not synthetic test fixtures.

    images_per_cluster defaults to the frozen v1.0 battery's own
    IMAGES_PER_CLUSTER (do not change that module constant - it's part
    of the frozen protocol). Callers outside the frozen battery (e.g.
    the aspect-ratio experiment's larger sample) can pass a different
    count here without touching Rounds 1-8's sample size at all."""
    n = images_per_cluster if images_per_cluster is not None else IMAGES_PER_CLUSTER
    clusters: dict[str, list[Path]] = {}
    for bucket in CLUSTER_BUCKETS:
        csv_path = BUCKET_DIR / f"{bucket}.csv"
        paths = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                p = Path(row["file_path"])
                if p.exists():
                    paths.append(p)
                if len(paths) >= n:
                    break
        clusters[bucket] = paths
    return clusters


def measure_operational_metadata(candidate: str, sample_img: Image.Image) -> dict:
    """
    Additive to the frozen v1.0 protocol, not part of it - this measures
    architecture/compute-graph properties (latency, memory, dim, patch
    count, prefix tokens), not qualification metrics. Weight-independent
    for latency/memory (same shapes regardless of pretrained vs random
    init), included here so every round automatically produces this
    without a separate manual inspection pass, per the comparison-report
    requirement.
    """
    model = timm.create_model(candidate, pretrained=True, num_classes=0)
    model.eval()
    cfg = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**cfg)
    x = transform(sample_img.convert("RGB")).unsqueeze(0)

    n_params = sum(p.numel() for p in model.parameters())
    with torch.no_grad():
        pooled_out = model(x)
        feats = model.forward_features(x)
    embedding_dim = int(pooled_out.shape[-1])
    if feats.dim() == 4:
        layout = _spatial_layout(feats, embedding_dim)
        patch_count = (feats.shape[2] * feats.shape[3]) if layout == "NCHW" else (feats.shape[1] * feats.shape[2])
        pooling_method = f"spatial_mean (conv/hierarchical, {layout})"
        num_prefix_tokens = 0
    else:
        patch_count = feats.shape[1]
        num_prefix_tokens = getattr(model, "num_prefix_tokens", None)
        pooling_method = "cls_token" if num_prefix_tokens == 1 else (
            "attention_pooling (no prefix token)" if num_prefix_tokens == 0 else "unknown"
        )

    def bench(dev, n=10, warmup=3):
        m = model.to(dev)
        xi = x.to(dev)
        with torch.no_grad():
            for _ in range(warmup):
                m(xi)
            if dev == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            for _ in range(n):
                m(xi)
            if dev == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
        mean_ms = (t1 - t0) / n * 1000
        peak_mb = torch.cuda.max_memory_allocated() / 1e6 if dev == "cuda" else None
        return mean_ms, peak_mb

    cpu_ms, _ = bench("cpu")
    gpu_ms, gpu_peak_mb = (None, None)
    if torch.cuda.is_available():
        gpu_ms, gpu_peak_mb = bench("cuda")

    return {
        "params_millions": round(n_params / 1e6, 2),
        "embedding_dim": embedding_dim,
        "patch_count": int(patch_count),
        "num_prefix_tokens": num_prefix_tokens,
        "pooling_method": pooling_method,
        "input_size": list(cfg.get("input_size", (3, 224, 224))),
        "cpu_latency_ms": round(cpu_ms, 1),
        "gpu_latency_ms": round(gpu_ms, 1) if gpu_ms is not None else None,
        "gpu_peak_memory_mb": round(gpu_peak_mb, 1) if gpu_peak_mb is not None else None,
    }


def topk_neighbors(query: np.ndarray, gallery: dict[str, np.ndarray], k: int, exclude: set[str]) -> list[str]:
    sims = [(name, cosine_sim(query, vec)) for name, vec in gallery.items() if name not in exclude]
    sims.sort(key=lambda t: t[1], reverse=True)
    return [name for name, _ in sims[:k]]


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    union = sa | sb
    return len(sa & sb) / len(union) if union else 1.0


def main():
    print(f"Round: {ROUND_LABEL}  candidate: {CANDIDATE}\n")

    clusters = sample_real_images()
    all_images: dict[str, tuple[str, Path]] = {}
    for bucket, paths in clusters.items():
        for p in paths:
            all_images[str(p)] = (bucket, p)
        print(f"  {bucket}: {len(paths)} real images")
    print(f"  total: {len(all_images)} images\n")

    model, transform = build_model_and_transform(CANDIDATE)

    # --- baseline embeddings (originals) ---
    pooled_orig: dict[str, np.ndarray] = {}
    patchmean_orig: dict[str, np.ndarray] = {}
    pil_cache: dict[str, Image.Image] = {}
    for name, (bucket, path) in all_images.items():
        img = Image.open(path).convert("RGB")
        pil_cache[name] = img
        pooled_orig[name] = embed_pooled(model, transform, img)
        patchmean_orig[name] = embed_patch_mean(model, transform, img)

    # --- cross-representation consistency (pooled vs patch-mean) ---
    names = list(all_images.keys())
    pooled_sims, patchmean_sims = [], []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pooled_sims.append(cosine_sim(pooled_orig[names[i]], pooled_orig[names[j]]))
            patchmean_sims.append(cosine_sim(patchmean_orig[names[i]], patchmean_orig[names[j]]))
    pooled_sims, patchmean_sims = np.array(pooled_sims), np.array(patchmean_sims)
    consistency_corr = float(np.corrcoef(pooled_sims, patchmean_sims)[0, 1])

    # --- baseline neighborhoods (self excluded) ---
    baseline_neighbors = {
        name: topk_neighbors(pooled_orig[name], pooled_orig, TOP_K, exclude={name})
        for name in names
    }

    # --- transform battery: recall@1, cosine drift, neighborhood Jaccard ---
    per_transform_results = {}
    for tname, tfunc in TRANSFORMS.items():
        recall_hits, drifts, jaccards = 0, [], []
        per_image_failures = []
        for name in names:
            try:
                transformed_img = tfunc(pil_cache[name])
            except Exception as e:
                per_image_failures.append((name, f"transform error: {e}"))
                continue
            t_pooled = embed_pooled(model, transform, transformed_img)

            drift = 1.0 - cosine_sim(pooled_orig[name], t_pooled)
            drifts.append(drift)

            nn = topk_neighbors(t_pooled, pooled_orig, 1, exclude=set())
            if nn and nn[0] == name:
                recall_hits += 1

            after_neighbors = topk_neighbors(t_pooled, pooled_orig, TOP_K, exclude={name})
            jaccards.append(jaccard(baseline_neighbors[name], after_neighbors))

        n = len(names)
        per_transform_results[tname] = {
            "recall_at_1": recall_hits / n if n else None,
            "mean_cosine_drift": float(np.mean(drifts)) if drifts else None,
            "mean_neighborhood_jaccard": float(np.mean(jaccards)) if jaccards else None,
            "failures": per_image_failures,
        }

    # --- report ---
    print(f"Cross-representation consistency (pooled vs. patch-mean similarity structure): "
          f"corr={consistency_corr:.3f}\n")
    print(f"{'transform':<24} {'recall@1':>9} {'mean_drift':>11} {'mean_jaccard@'+str(TOP_K):>15}")
    for tname, r in per_transform_results.items():
        print(f"{tname:<24} {r['recall_at_1']:>9.2f} {r['mean_cosine_drift']:>11.4f} {r['mean_neighborhood_jaccard']:>15.3f}")
        if r["failures"]:
            for name, err in r["failures"]:
                print(f"    FAILED on {name}: {err}")

    operational = measure_operational_metadata(CANDIDATE, next(iter(pil_cache.values())))
    print(f"\nOperational metadata: {operational}")

    RESULTS_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "battery_version": BATTERY_VERSION,
        "round": ROUND_LABEL,
        "candidate": CANDIDATE,
        "num_images": len(names),
        "clusters": {b: [str(p) for p in paths] for b, paths in clusters.items()},
        "cross_representation_consistency_corr": consistency_corr,
        "transform_battery": per_transform_results,
        "operational_metadata": operational,
    }
    with open(RESULTS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"\nAppended full record to {RESULTS_LOG}")


if __name__ == "__main__":
    main()
