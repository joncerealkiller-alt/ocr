"""
Benchmark 2.3 - Multi-Tower Routing Audit.

Extends Benchmark 2.2's single-tower (DINOv2 only) cross-validator to
all 8 encoders qualified in the frozen Vision Qualification Battery
v1.0 (docs/VISION_IR_RESEARCH.md), on the IDENTICAL stratified 146-image
corpus (same BUCKET_PLAN, same deterministic bucket-CSV slicing,
imported directly from benchmark2_2_cross_validator rather than
re-declared, so this stays a true extension of that baseline, not a
different sample that happens to be the same size).

Real gap in Benchmark 2.2 this closes: that script only ever saved the
64 DISAGREEMENT cases to disk (data/outputs/benchmark2_2/disagreements/)
- the other 82 agreement cases' Gemma confidence/reason were never
persisted, and only one tower (DINOv2) was ever run. Reusing "cached"
Gemma results wasn't actually possible from that data alone. This
script re-runs Gemma once fresh (146 images, cheap) so it always has a
complete, consistent per-image record for every image and every
encoder, not a mix of fresh-vs-reconstructed data.

GPU sequencing: the 8 timm tower encoders run on CPU (embed_pooled()
never moves anything to cuda - confirmed by reading vision_encoder_
qualification.py directly), so there is zero GPU contention concern
computing all 8 before Gemma ever loads. Gemma loads once, at the end,
classifies all 146 images, releases. No concurrent-residency question
to resolve here (unlike the reference-classifier-qualification work),
since the towers never touch the GPU at all in this qualification-style
embedding usage.

What this produces that Benchmark 2.2 didn't:
  - A full per-image record for all 146 images (not just 64
    disagreements): every tower's predicted bucket + score, Gemma's
    bucket + confidence + reason, sample_bucket (stratified ground
    truth), and a consensus classification.
  - Pairwise agreement matrix among all 8 encoders + Gemma (9x9).
  - Consensus categorization per image: unanimous (all 9 agree),
    majority (>=5 of 9 agree on one bucket), split (multiple buckets
    with real support, no majority), complete_disagreement (no bucket
    has more than 1-2 votes).
  - A merged disagreement dataset: images where Gemma disagrees with
    the TOWER CONSENSUS (the majority/plurality bucket among the 8
    towers), not just with one arbitrary tower - a stronger signal
    than Benchmark 2.2's single-tower-vs-Gemma disagreement.

Entirely observational, same as every Benchmark 1/2 script before it:
reads data/working images directly, writes nothing back to
data/buckets/*.csv or any production manifest.

Usage:
    python -m benchmark.benchmark2_3_multi_tower_routing_audit
"""
from __future__ import annotations

import gc
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.benchmark2_2_cross_validator import (
    BUCKET_PLAN, UNCERTAIN_REVIEW_TEST_COUNT, load_existing_paths,
)
from benchmark.vision_encoder_qualification import build_model_and_transform, embed_pooled
from core.vision_embeddings import predict_nearest_bucket, classify_consensus
from core.classifier import build_classifier_loader, load_pipeline_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "outputs" / "benchmark2_3_multi_tower"
DISAGREEMENTS_DIR = OUT_DIR / "consensus_disagreements"

# Same 8 encoders as benchmark2_sequential_runtime.py's qualified set
# (docs/VISION_IR_RESEARCH.md's frozen Vision Qualification Battery v1.0)
TOWERS = [
    ("dinov2", "vit_small_patch14_dinov2.lvd142m"),
    ("convnext", "convnext_tiny.fb_in22k"),
    ("naflex_siglip", "naflexvit_base_patch16_siglip.v2_webli"),
    ("siglip_fixed", "vit_base_patch16_siglip_224.v2_webli"),
    ("eva02", "eva02_base_patch14_224.mim_in22k"),
    ("beit", "beit_base_patch16_224.in22k_ft_in22k"),
    ("swin", "swin_base_patch4_window7_224.ms_in22k"),
    ("mae", "vit_base_patch16_224.mae"),
]


def build_corpus():
    """Identical selection logic to benchmark2_2_cross_validator.main() -
    same bucket CSVs, same slicing - so this stays directly comparable
    to that baseline, not a different sample of the same size."""
    test_items = []  # (name, path, sample_bucket)
    ref_paths_by_bucket = {}
    for bucket, test_n, ref_n in BUCKET_PLAN:
        all_paths = load_existing_paths(bucket)
        test_paths = all_paths[:test_n]
        ref_paths = test_paths if ref_n == 0 else all_paths[test_n:test_n + ref_n]
        for p in test_paths:
            test_items.append((p.name, p, bucket))
        ref_paths_by_bucket[bucket] = ref_paths

    ur_paths = load_existing_paths("uncertain_review")[:UNCERTAIN_REVIEW_TEST_COUNT]
    for p in ur_paths:
        test_items.append((p.name, p, "uncertain_review"))

    return test_items, ref_paths_by_bucket


def run_tower(tag: str, test_items, ref_paths_by_bucket) -> dict[str, dict]:
    model, transform = build_model_and_transform(tag)

    reference_embeddings: dict[str, list[tuple[str, np.ndarray]]] = {}
    for bucket, ref_paths in ref_paths_by_bucket.items():
        embs = []
        for p in ref_paths:
            try:
                embs.append((p.name, embed_pooled(model, transform, Image.open(p).convert("RGB"))))
            except Exception as e:
                print(f"    reference embed failed for {p.name}: {e}")
        reference_embeddings[bucket] = embs

    # predict_nearest_bucket() - promoted to core/vision_embeddings.py
    # 2026-08-02 (Stage 2/Decision Engine needs this exact validated
    # logic in production) - identical method, just relocated.
    results = {}
    for name, path, sample_bucket in test_items:
        emb = embed_pooled(model, transform, Image.open(path).convert("RGB"))
        bucket, score = predict_nearest_bucket(emb, reference_embeddings, exclude_key=name)
        results[name] = {"bucket": bucket, "score": score}

    del model, transform
    gc.collect()
    return results


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DISAGREEMENTS_DIR.mkdir(parents=True, exist_ok=True)

    test_items, ref_paths_by_bucket = build_corpus()
    print(f"Corpus: {len(test_items)} test images (identical to Benchmark 2.2's stratified sample)\n")

    all_tower_results = {}  # tower_name -> {image_name: {bucket, score}}
    for tower_name, tag in TOWERS:
        print(f"=== Running tower: {tower_name} ({tag}) ===")
        all_tower_results[tower_name] = run_tower(tag, test_items, ref_paths_by_bucket)
        print(f"  done, {len(all_tower_results[tower_name])} predictions\n")

    print("=== Loading Gemma for real classification (observational, not written to production) ===")
    pipeline_cfg = load_pipeline_config()
    loader = build_classifier_loader(pipeline_cfg)
    gemma_results = {}
    try:
        for i, (name, path, sample_bucket) in enumerate(test_items, 1):
            with Image.open(path) as raw_image:
                result = loader.classify(str(path), raw_image)
            gemma_results[name] = {
                "bucket": result.category.value, "confidence": result.confidence, "reason": result.reason,
            }
            print(f"[{i}/{len(test_items)}] {name:<45} gemma={result.category.value}")
    finally:
        loader.release()

    # --- assemble full per-image records ---
    tower_names = [t[0] for t in TOWERS]
    all_voters = tower_names + ["gemma"]
    records = []
    for name, path, sample_bucket in test_items:
        votes = {t: all_tower_results[t][name]["bucket"] for t in tower_names}
        votes["gemma"] = gemma_results[name]["bucket"]
        consensus = classify_consensus(list(votes.values()))
        tower_only_votes = Counter(votes[t] for t in tower_names)
        tower_consensus_bucket, tower_consensus_n = tower_only_votes.most_common(1)[0]

        records.append({
            "image": name, "source_path": str(path), "sample_bucket": sample_bucket,
            "towers": {t: all_tower_results[t][name] for t in tower_names},
            "gemma": gemma_results[name],
            "consensus_category": consensus,
            "tower_consensus_bucket": tower_consensus_bucket,
            "tower_consensus_strength": f"{tower_consensus_n}/{len(tower_names)}",
            "gemma_agrees_with_tower_consensus": gemma_results[name]["bucket"] == tower_consensus_bucket,
        })

    with open(OUT_DIR / "all_results.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    # --- pairwise agreement matrix (9x9: 8 towers + gemma) ---
    pairwise = {a: {b: 0 for b in all_voters} for a in all_voters}
    for r in records:
        votes = {t: r["towers"][t]["bucket"] for t in tower_names}
        votes["gemma"] = r["gemma"]["bucket"]
        for a in all_voters:
            for b in all_voters:
                if votes[a] == votes[b]:
                    pairwise[a][b] += 1
    n = len(records)
    pairwise_rates = {a: {b: round(pairwise[a][b] / n, 3) for b in all_voters} for a in all_voters}

    with open(OUT_DIR / "pairwise_agreement.json", "w", encoding="utf-8") as f:
        json.dump(pairwise_rates, f, indent=2)

    print("\n=== Pairwise agreement matrix ===")
    header = "".join(f"{v:>15}" for v in all_voters)
    print(f"{'':>15}{header}")
    for a in all_voters:
        row = "".join(f"{pairwise_rates[a][b]:>15.2f}" for b in all_voters)
        print(f"{a:>15}{row}")

    # --- consensus distribution ---
    consensus_counts = Counter(r["consensus_category"] for r in records)
    print(f"\n=== Consensus distribution (n={n}) ===")
    for cat in ["unanimous", "majority", "split", "complete_disagreement"]:
        print(f"  {cat:<22} {consensus_counts.get(cat, 0)}")

    # --- merged disagreement dataset: Gemma vs TOWER CONSENSUS (not one tower) ---
    consensus_disagreements = [r for r in records if not r["gemma_agrees_with_tower_consensus"]]
    print(f"\n=== Gemma vs tower-consensus disagreements: {len(consensus_disagreements)}/{n} ===")
    for i, r in enumerate(consensus_disagreements, 1):
        d = DISAGREEMENTS_DIR / f"disagreement_{i:03d}"
        d.mkdir(exist_ok=True)
        src = Path(r["source_path"])
        shutil.copy2(src, d / f"image{src.suffix}")
        record = dict(r)
        record["timestamp"] = datetime.now(timezone.utc).isoformat()
        record["human_verdict"] = None
        record["human_verdict_notes"] = None
        with open(d / "record.json", "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark": "2.3_multi_tower_routing_audit",
        "n_total": n,
        "towers": tower_names,
        "consensus_distribution": dict(consensus_counts),
        "n_gemma_vs_tower_consensus_disagreements": len(consensus_disagreements),
        "note": "Observational only - nothing written to production manifests or bucket CSVs.",
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nFull results: {OUT_DIR / 'all_results.json'}")
    print(f"Pairwise agreement: {OUT_DIR / 'pairwise_agreement.json'}")
    print(f"Consensus disagreements: {DISAGREEMENTS_DIR}/ ({len(consensus_disagreements)} folders)")
    print(f"Summary: {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
