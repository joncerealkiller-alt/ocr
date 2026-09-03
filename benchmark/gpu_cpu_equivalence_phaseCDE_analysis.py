"""
Phases C, D, E of the full-corpus CPU vs GPU equivalence
characterization (2026-08-04). Pure analysis over the two embedding
sets Phase A/B already produced - no model inference here, reuses
core/decision_engine.py's compute_tower_consensus_for_image() and
build_reference_embeddings() exactly as they exist in production (not
reimplemented), so this measures the SAME computation the real pipeline
would do, just fed CPU- vs GPU-computed embeddings.

Phase C: raw embedding-vector equivalence (cosine similarity, L2
distance, max abs element diff) for every (image, encoder) pair.

Phase D: operational equivalence - each embedding set used
INDEPENDENTLY (its own vectors as both query and reference pool,
exactly like the real audit) to recompute nearest-neighbour prediction,
tower vote, consensus category/strength per image; CPU-based vs
GPU-based results compared directly.

Phase E: downstream equivalence - reproduces the tower-consensus-audit
disagreement classification (vs Gemma's real bucket, from pipeline_db)
independently for CPU-based and GPU-based results, compared against
each other AND against the real audit's previously-reported 692 total /
47 unanimous disagreements. IMPORTANT CAVEAT (stated here, not just in
the report): the ORIGINAL baseline_embeddings.json that produced those
692/47 numbers was destroyed (see this session's own incident record) -
Phase A's CPU regeneration is compared against those PREVIOUSLY-REPORTED
counts as an implicit determinism check (CPU inference in eval mode has
no dropout/randomness, so a fresh run should reproduce them exactly),
not a literal byte-for-byte re-comparison against the lost file.

Usage:
    python -m benchmark.gpu_cpu_equivalence_phaseCDE_analysis
"""
from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

from core.baseline_embeddings import resolve_baseline_image_path
from core.decision_engine import BUCKET_DIR, build_reference_embeddings, compute_tower_consensus_for_image
from core.pipeline_db import DEFAULT_DB_PATH, PipelineDatabase
from core.vision_embeddings import QUALIFIED_ENCODERS, cosine_sim

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "outputs" / "gpu_cpu_equivalence"
CPU_PATH = OUT_DIR / "baseline_embeddings_cpu.json"
GPU_PATH = OUT_DIR / "baseline_embeddings_gpu.json"

PREVIOUSLY_REPORTED_TOTAL_DISAGREEMENTS = 692
PREVIOUSLY_REPORTED_UNANIMOUS_DISAGREEMENTS = 47

OUTLIER_COSINE_THRESHOLD = 0.9999


def load_records(path: Path) -> dict[str, dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {resolve_baseline_image_path(r["image"]): r for r in raw}


def _stats(values: list[float]) -> dict:
    return {
        "min": min(values), "max": max(values),
        "mean": statistics.mean(values), "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def phase_c(cpu_records: dict, gpu_records: dict, common_paths: list[str]) -> dict:
    print("=" * 70)
    print("[Phase C] Full corpus embedding-vector equivalence")
    print("=" * 70)

    cos_sims, l2_dists, max_abs_diffs = [], [], []
    outliers = []
    for path in common_paths:
        cpu_rec, gpu_rec = cpu_records[path], gpu_records[path]
        for encoder_name, _ in QUALIFIED_ENCODERS:
            cpu_emb = cpu_rec["embeddings"].get(encoder_name)
            gpu_emb = gpu_rec["embeddings"].get(encoder_name)
            if cpu_emb is None or gpu_emb is None:
                continue
            cv = np.array(cpu_emb["vector"], dtype=np.float64)
            gv = np.array(gpu_emb["vector"], dtype=np.float64)
            sim = cosine_sim(cv, gv)
            l2 = float(np.linalg.norm(cv - gv))
            max_abs = float(np.max(np.abs(cv - gv)))
            cos_sims.append(sim)
            l2_dists.append(l2)
            max_abs_diffs.append(max_abs)
            if sim < OUTLIER_COSINE_THRESHOLD:
                outliers.append({"image": path, "encoder": encoder_name, "cosine_similarity": sim,
                                  "l2_distance": l2, "max_abs_diff": max_abs})

    print(f"images compared: {len(common_paths)}")
    print(f"embedding pairs: {len(cos_sims)}")
    print(f"cosine similarity:  {_stats(cos_sims)}")
    print(f"L2 distance:        {_stats(l2_dists)}")
    print(f"max abs elem diff:  {_stats(max_abs_diffs)}")
    print(f"outliers (cosine similarity < {OUTLIER_COSINE_THRESHOLD}): {len(outliers)}")
    for o in sorted(outliers, key=lambda x: x["cosine_similarity"])[:20]:
        print(f"  {o}")

    return {
        "images_compared": len(common_paths), "embedding_pairs": len(cos_sims),
        "cosine_similarity": _stats(cos_sims), "l2_distance": _stats(l2_dists),
        "max_abs_diff": _stats(max_abs_diffs), "outliers": outliers,
    }


def phase_d(cpu_records: dict, gpu_records: dict, common_paths: list[str]) -> tuple[dict, dict, dict]:
    print("\n" + "=" * 70)
    print("[Phase D] Operational equivalence (independent per-set replay)")
    print("=" * 70)

    cpu_refs = build_reference_embeddings(bucket_dir=BUCKET_DIR, baseline_path=CPU_PATH)
    gpu_refs = build_reference_embeddings(bucket_dir=BUCKET_DIR, baseline_path=GPU_PATH)

    cpu_results, gpu_results = {}, {}
    for path in common_paths:
        cpu_cat, cpu_top, cpu_per_enc = compute_tower_consensus_for_image(cpu_records[path], cpu_refs, exclude_key=path)
        gpu_cat, gpu_top, gpu_per_enc = compute_tower_consensus_for_image(gpu_records[path], gpu_refs, exclude_key=path)
        cpu_results[path] = {"category": cpu_cat, "top_bucket": cpu_top, "per_encoder": cpu_per_enc}
        gpu_results[path] = {"category": gpu_cat, "top_bucket": gpu_top, "per_encoder": gpu_per_enc}

    nn_changes = 0          # per (image, encoder) winner changes
    n_pairs = 0
    vote_count_changes = 0  # per-image: did the vote-count distribution shape change
    consensus_changes = 0   # per-image: did the consensus CATEGORY change
    top_bucket_changes = 0  # per-image: did the overall winning bucket change
    margin_deltas = []
    changed_images: list[dict] = []

    for path in common_paths:
        cr, gr = cpu_results[path], gpu_results[path]
        image_changed = False

        if cr["category"] != gr["category"]:
            consensus_changes += 1
            image_changed = True
        if cr["top_bucket"] != gr["top_bucket"]:
            top_bucket_changes += 1
            image_changed = True

        cpu_votes = Counter(t["winner"] for t in cr["per_encoder"].values())
        gpu_votes = Counter(t["winner"] for t in gr["per_encoder"].values())
        if cpu_votes != gpu_votes:
            vote_count_changes += 1
            image_changed = True

        for encoder_name in cr["per_encoder"]:
            if encoder_name not in gr["per_encoder"]:
                continue
            n_pairs += 1
            cpu_t, gpu_t = cr["per_encoder"][encoder_name], gr["per_encoder"][encoder_name]
            if cpu_t["winner"] != gpu_t["winner"]:
                nn_changes += 1
                image_changed = True
            if cpu_t["margin"] is not None and gpu_t["margin"] is not None:
                margin_deltas.append(abs(cpu_t["margin"] - gpu_t["margin"]))

        if image_changed:
            changed_images.append({
                "image": path,
                "cpu_category": cr["category"], "gpu_category": gr["category"],
                "cpu_top_bucket": cr["top_bucket"], "gpu_top_bucket": gr["top_bucket"],
            })

    print(f"Images compared: {len(common_paths)}")
    print(f"Embedding pairs: {n_pairs}")
    print(f"Nearest-neighbour (per-encoder winner) changes: {nn_changes} / {n_pairs}")
    print(f"Tower vote changes: {nn_changes} / {n_pairs}  (same measurement as nearest-neighbour above)")
    print(f"Consensus category changes: {consensus_changes} / {len(common_paths)} images")
    print(f"Vote-count distribution changes: {vote_count_changes} / {len(common_paths)} images")
    print(f"Overall winning-bucket changes: {top_bucket_changes} / {len(common_paths)} images")
    if margin_deltas:
        print(f"Margin changes (abs delta): {_stats(margin_deltas)}")
    print(f"Images with ANY change: {len(changed_images)}")
    for c in changed_images[:30]:
        print(f"  {c}")

    summary = {
        "images_compared": len(common_paths), "embedding_pairs": n_pairs,
        "nearest_neighbour_changes": nn_changes, "tower_vote_changes": nn_changes,
        "consensus_category_changes": consensus_changes,
        "vote_count_distribution_changes": vote_count_changes,
        "overall_winning_bucket_changes": top_bucket_changes,
        "margin_delta_stats": _stats(margin_deltas) if margin_deltas else None,
        "changed_images": changed_images,
    }
    return summary, cpu_results, gpu_results


def phase_e(cpu_results: dict, gpu_results: dict, common_paths: list[str]) -> dict:
    print("\n" + "=" * 70)
    print("[Phase E] Downstream equivalence (disagreement-audit reproduction)")
    print("=" * 70)

    db = PipelineDatabase(DEFAULT_DB_PATH)
    gemma_bucket_by_path = {img["working_path"]: img["bucket"] for img in db.list_images()}

    def classify(results: dict) -> dict:
        disagreements = []
        by_category = Counter()
        for path in common_paths:
            gemma_bucket = gemma_bucket_by_path.get(path)
            if gemma_bucket is None:
                continue
            r = results[path]
            by_category[r["category"]] += 1
            if r["top_bucket"] != gemma_bucket:
                disagreements.append({"image": path, "gemma_bucket": gemma_bucket,
                                       "tower_top_bucket": r["top_bucket"], "consensus_category": r["category"]})
        by_cat_disagree = Counter(d["consensus_category"] for d in disagreements)
        return {"disagreements": disagreements, "consensus_distribution": dict(by_category),
                "disagreements_by_category": dict(by_cat_disagree)}

    cpu_audit = classify(cpu_results)
    gpu_audit = classify(gpu_results)

    cpu_total = len(cpu_audit["disagreements"])
    gpu_total = len(gpu_audit["disagreements"])
    cpu_unanimous = cpu_audit["disagreements_by_category"].get("unanimous", 0)
    gpu_unanimous = gpu_audit["disagreements_by_category"].get("unanimous", 0)

    print(f"CPU-based: {cpu_total} total disagreements, {cpu_unanimous} unanimous "
          f"({cpu_audit['disagreements_by_category']})")
    print(f"GPU-based: {gpu_total} total disagreements, {gpu_unanimous} unanimous "
          f"({gpu_audit['disagreements_by_category']})")
    print(f"\nPreviously reported (now-destroyed original embeddings): "
          f"{PREVIOUSLY_REPORTED_TOTAL_DISAGREEMENTS} total, {PREVIOUSLY_REPORTED_UNANIMOUS_DISAGREEMENTS} unanimous")
    print(f"  CPU regeneration matches previously-reported totals: "
          f"{cpu_total == PREVIOUSLY_REPORTED_TOTAL_DISAGREEMENTS and cpu_unanimous == PREVIOUSLY_REPORTED_UNANIMOUS_DISAGREEMENTS}")

    cpu_disagree_set = {d["image"] for d in cpu_audit["disagreements"]}
    gpu_disagree_set = {d["image"] for d in gpu_audit["disagreements"]}
    only_cpu = cpu_disagree_set - gpu_disagree_set
    only_gpu = gpu_disagree_set - cpu_disagree_set

    print(f"\nDisagreement-set differences: {len(only_cpu)} images disagree under CPU only, "
          f"{len(only_gpu)} images disagree under GPU only")
    for path in sorted(only_cpu)[:20]:
        print(f"  CPU-only disagreement: {path}")
    for path in sorted(only_gpu)[:20]:
        print(f"  GPU-only disagreement: {path}")

    return {
        "cpu_total_disagreements": cpu_total, "gpu_total_disagreements": gpu_total,
        "cpu_disagreements_by_category": cpu_audit["disagreements_by_category"],
        "gpu_disagreements_by_category": gpu_audit["disagreements_by_category"],
        "previously_reported_total": PREVIOUSLY_REPORTED_TOTAL_DISAGREEMENTS,
        "previously_reported_unanimous": PREVIOUSLY_REPORTED_UNANIMOUS_DISAGREEMENTS,
        "cpu_matches_previously_reported": (
            cpu_total == PREVIOUSLY_REPORTED_TOTAL_DISAGREEMENTS
            and cpu_unanimous == PREVIOUSLY_REPORTED_UNANIMOUS_DISAGREEMENTS
        ),
        "images_disagreeing_cpu_only": sorted(only_cpu),
        "images_disagreeing_gpu_only": sorted(only_gpu),
    }


def main() -> None:
    if not CPU_PATH.exists() or not GPU_PATH.exists():
        print(f"Missing input(s) - CPU exists={CPU_PATH.exists()}, GPU exists={GPU_PATH.exists()}")
        return

    cpu_records = load_records(CPU_PATH)
    gpu_records = load_records(GPU_PATH)
    common_paths = sorted(set(cpu_records) & set(gpu_records))
    missing_in_gpu = sorted(set(cpu_records) - set(gpu_records))
    missing_in_cpu = sorted(set(gpu_records) - set(cpu_records))
    if missing_in_gpu or missing_in_cpu:
        print(f"WARNING: {len(missing_in_gpu)} image(s) in CPU set but not GPU, "
              f"{len(missing_in_cpu)} vice versa - these are excluded from the comparison below.")

    phase_c_results = phase_c(cpu_records, gpu_records, common_paths)
    phase_d_results, cpu_consensus, gpu_consensus = phase_d(cpu_records, gpu_records, common_paths)
    phase_e_results = phase_e(cpu_consensus, gpu_consensus, common_paths)

    full_report = {
        "phase_c": phase_c_results, "phase_d": phase_d_results, "phase_e": phase_e_results,
        "missing_in_gpu": missing_in_gpu, "missing_in_cpu": missing_in_cpu,
    }
    out_path = OUT_DIR / "phaseCDE_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, default=str)
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
