"""
Generic vision-tower embedding infrastructure - the pure, stateless
functions originally written for benchmark/vision_encoder_qualification.py
(the frozen Vision Qualification Battery v1.0, see docs/VISION_IR_RESEARCH.md),
moved here 2026-08-02 so production code (core/baseline_embeddings.py)
can use the exact same, already-validated embedding logic without a
core module importing from benchmark/ (this project's convention is
the reverse: benchmark/scripts are consumers of core engines, not the
other way around - same reasoning as core/pdf_conversion.py's move
earlier this session).

benchmark/vision_encoder_qualification.py now imports these from here
instead of defining them locally - every other benchmark script that
imports from vision_encoder_qualification (benchmark2_2_cross_validator,
benchmark2_3_multi_tower_routing_audit, benchmark2_sequential_runtime,
concurrency_falsification_test, vision_encoder_aspect_ratio_experiment,
the pilot scripts) needed ZERO changes - they all import from that
module's path, which just re-exports these now.

QUALIFIED_ENCODERS is the canonical (name, timm_tag) list for the 8
encoders qualified in the frozen battery - previously redefined ad hoc
in benchmark2_sequential_runtime.py's CANDIDATES and benchmark2_3_
multi_tower_routing_audit.py's TOWERS; consolidated here as the single
source of truth for any NEW consumer (those two benchmark scripts are
untouched, per the "don't touch already-validated scripts" discipline -
this is for future callers, not a forced migration of working code).

predict_nearest_bucket()/classify_consensus() (added 2026-08-02) -
promoted from benchmark/benchmark2_3_multi_tower_routing_audit.py's
local predict_bucket()/classify_consensus() (previously a closure over
one run_tower() call) for the same reason as the move above: Stage 2
(core/decision_engine.py) needs this EXACT validated nearest-cluster/
consensus logic in production, and core/ modules must never import from
benchmark/ (this project's established dependency direction). The
benchmark script itself now imports these instead of defining them
locally - same relocate-don't-reinvent pattern as QUALIFIED_ENCODERS.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch
import timm
from PIL import Image

QUALIFIED_ENCODERS = [
    ("dinov2", "vit_small_patch14_dinov2.lvd142m"),
    ("convnext", "convnext_tiny.fb_in22k"),
    ("naflex_siglip", "naflexvit_base_patch16_siglip.v2_webli"),
    ("siglip_fixed", "vit_base_patch16_siglip_224.v2_webli"),
    ("eva02", "eva02_base_patch14_224.mim_in22k"),
    ("beit", "beit_base_patch16_224.in22k_ft_in22k"),
    ("swin", "swin_base_patch4_window7_224.ms_in22k"),
    ("mae", "vit_base_patch16_224.mae"),
]


def build_model_and_transform(candidate: str):
    model = timm.create_model(candidate, pretrained=True, num_classes=0)
    model.eval()
    cfg = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**cfg)
    return model, transform


def _spatial_layout(feats: torch.Tensor, embedding_dim: int) -> str:
    """
    4D spatial feature maps are NOT all the same axis order - real bug
    caught before Round 8 (Swin) ran: ConvNeXt's forward_features() is
    NCHW (channel at dim 1), but Swin's is NHWC (channel at dim -1).
    Blindly assuming one layout silently averages over the wrong axes
    for the other. Detected by checking which axis actually matches the
    model's own known embedding_dim, not assumed from one architecture.
    """
    if feats.shape[1] == embedding_dim:
        return "NCHW"
    if feats.shape[-1] == embedding_dim:
        return "NHWC"
    raise ValueError(
        f"Could not determine spatial layout: feats.shape={tuple(feats.shape)}, "
        f"embedding_dim={embedding_dim} matches neither dim 1 nor dim -1."
    )


@torch.no_grad()
def embed_pooled(model, transform, pil_image: Image.Image) -> np.ndarray:
    x = transform(pil_image.convert("RGB")).unsqueeze(0)
    return model(x).squeeze(0).numpy()


@torch.no_grad()
def embed_patch_mean(model, transform, pil_image: Image.Image) -> np.ndarray:
    """
    Mean-pool the pre-pool representation - same semantic metric across
    architecture families, but the tensor shape convention genuinely
    differs and both must be handled correctly:
      - isotropic ViT families: forward_features() returns a (B, N, D)
        token sequence, with model.num_prefix_tokens leading tokens
        (CLS and/or register tokens) to drop before averaging - NOT a
        hardcoded "1" (DINOv2 has num_prefix_tokens=1, but SigLIP-family
        models have 0 - no CLS token at all, attention pooling instead).
      - hierarchical CNNs/windowed transformers: forward_features()
        returns a spatial 4D map, but NOT always the same axis order -
        ConvNeXt is NCHW, Swin is NHWC - detected via _spatial_layout()
        against the model's own known embedding dim, never assumed.
    """
    x = transform(pil_image.convert("RGB")).unsqueeze(0)
    feats = model.forward_features(x)
    if feats.dim() == 4:
        embedding_dim = getattr(model, "num_features", None)
        if embedding_dim is None:
            raise ValueError(
                f"{type(model).__name__} has no num_features attribute - "
                "can't safely determine spatial layout (NCHW vs NHWC) for "
                "patch-mean pooling. Check manually before trusting this metric."
            )
        layout = _spatial_layout(feats, embedding_dim)
        dims = (2, 3) if layout == "NCHW" else (1, 2)
        return feats.mean(dim=dims).squeeze(0).numpy()
    # (B, N, D) token sequence - drop model's actual prefix-token count
    n_prefix = getattr(model, "num_prefix_tokens", 1)
    patch_tokens = feats[:, n_prefix:, :] if n_prefix else feats
    return patch_tokens.mean(dim=1).squeeze(0).numpy()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def score_all_buckets(
    emb: np.ndarray,
    reference_embeddings: dict[str, list[tuple[str, np.ndarray]]],
    exclude_key: str | None = None,
) -> dict[str, float]:
    """
    Mean cosine similarity to EVERY bucket's reference embeddings, not
    just the winner - the full per-bucket score distribution
    predict_nearest_bucket() below computes internally anyway, extracted
    (2026-08-03, Jon: "record the scoring per image for all buckets, so
    we can see where images are closely tied in confidence first, then
    target with adjusting thresholds - doing that without all the
    bucket confidence values would be guesswork") so a caller that needs
    the full picture - not just "which bucket won" - doesn't have to
    re-derive it.

    reference_embeddings: {bucket_name: [(key, vector), ...]}. exclude_key
    drops one reference entry by its key before scoring, same reasoning
    as predict_nearest_bucket() below. A bucket with zero reference
    entries after exclusion is simply ABSENT from the returned dict, not
    scored as 0.0 - there's no meaningful "zero similarity" claim to make
    when there was nothing to compare against.
    """
    scores = {}
    for bucket, entries in reference_embeddings.items():
        filtered = [v for (k, v) in entries if k != exclude_key]
        if filtered:
            scores[bucket] = float(np.mean([cosine_sim(emb, v) for v in filtered]))
    return scores


def predict_nearest_bucket(
    emb: np.ndarray,
    reference_embeddings: dict[str, list[tuple[str, np.ndarray]]],
    exclude_key: str | None = None,
) -> tuple[str, float]:
    """
    Nearest-cluster bucket prediction via MEAN cosine similarity to each
    bucket's reference embeddings (not cosine-to-centroid - deliberately
    the same "mean of per-item similarities" method the Multi-Tower
    Routing Audit validated, not a cheaper approximation). Thin wrapper
    over score_all_buckets() (2026-08-03) - unchanged behavior/signature,
    still just the single winning bucket + its score, since this exact
    function is what benchmark/benchmark2_3_multi_tower_routing_audit.py
    (frozen research code) already depends on.

    reference_embeddings: {bucket_name: [(key, vector), ...]}.
    exclude_key drops one reference entry by its key before scoring -
    e.g. an image scoring against a reference set that happens to
    already include itself would trivially win via self-similarity;
    the audit's own run_tower() used this to exclude the test image's
    name from its own bucket's reference list for exactly this reason.
    Raises if every bucket ends up with zero reference entries after
    exclusion (nothing to compare against) rather than silently
    returning a meaningless best-of-empty pick.
    """
    scores = score_all_buckets(emb, reference_embeddings, exclude_key)
    if not scores:
        raise ValueError(
            "predict_nearest_bucket(): every bucket had zero reference "
            "embeddings after exclusion - nothing to compare against."
        )
    best = max(scores, key=scores.get)
    return best, scores[best]


def classify_consensus(votes: list[str]) -> str:
    """
    Categorizes agreement among a list of votes (e.g. one bucket
    prediction per tower) - the exact method from the Multi-Tower
    Routing Audit: "unanimous" (all agree), "majority" (>= half + 1
    agree on one value), "complete_disagreement" (top vote has <= 2
    supporters), else "split".
    """
    counts = Counter(votes)
    top_bucket, top_n = counts.most_common(1)[0]
    n = len(votes)
    if top_n == n:
        return "unanimous"
    if top_n >= (n // 2) + 1:
        return "majority"
    if top_n <= 2:
        return "complete_disagreement"
    return "split"
