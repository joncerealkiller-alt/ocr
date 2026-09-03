"""
Phase 4 of the Stage-4 error-analysis pipeline (2026-08-08): embedding
statistics from CACHED embeddings only. Uses torch.pca_lowrank() ONLY
(already available via the installed torch, zero new dependency) - per
the handoff prompt's explicit instruction, t-SNE/UMAP (which would need
scikit-learn/umap-learn, confirmed NOT installed in this environment
while researching docs/GEMMA_HIDDEN_STATE_ERROR_ANALYSIS_STAGE4_
RESEARCH.md) are deferred until explicitly approved.

Per location (test split, the split every accuracy number in this
project's probe work has been measured on):
    - PCA projection to 2D (for later plotting) + explained variance
    - per-class centroid (mean embedding)
    - within-class variance (mean squared distance to own centroid)
    - between-class distance (pairwise centroid distances)
    - nearest-neighbour label agreement (does each point's nearest
      neighbour share its true class?)
    - cosine similarity matrix between class centroids
    - cluster compactness (within-class variance / between-class distance
      ratio - lower is more compact/separated)

Output: data/outputs/error_analysis/phase4_embedding_stats/<location>.json
        data/outputs/error_analysis/phase4_embedding_stats/<location>_pca2d.csv
          (review_id, gt, pc1, pc2 - for later plotting without recomputing PCA)

Usage:
    python diagnostics/error_analysis/phase4_embedding_stats.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from diagnostics.error_analysis.common import LOCATIONS, OUT_DIR, load_all_shards

STATS_DIR = OUT_DIR / "phase4_embedding_stats"


def compute_pca(x: torch.Tensor, n_components: int = 2):
    """torch.pca_lowrank - zero new dependency (torch already installed).
    Returns (projection, explained_variance_ratio)."""
    x_centered = x - x.mean(dim=0, keepdim=True)
    q = min(n_components + 5, x.shape[0], x.shape[1])  # small oversampling per torch's own recommendation
    U, S, V = torch.pca_lowrank(x_centered, q=q)
    projection = x_centered @ V[:, :n_components]
    total_var = (x_centered ** 2).sum()
    # Variance explained by each of the q computed components (S are singular values of the centered data)
    component_var = S ** 2
    explained_ratio = (component_var[:n_components] / total_var).tolist()
    return projection, explained_ratio


def main():
    STATS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading shards to assemble per-location test-split embeddings...")
    shards = load_all_shards()
    test_shards = sorted([s for s in shards if s["split"] == "test"], key=lambda s: s["start"])
    classes = test_shards[0]["classes"]

    y_all = torch.cat([s["y"] for s in test_shards], dim=0)
    paths_all = [p for s in test_shards for p in s["paths"]]
    review_ids = [f"test_{i:05d}" for i in range(len(paths_all))]

    for loc in LOCATIONS:
        x = torch.cat([s["x_by_loc"][loc] for s in test_shards], dim=0)
        print(f"\n{loc}: x={tuple(x.shape)}")

        projection, explained_ratio = compute_pca(x, n_components=2)

        # Per-class centroid + within-class variance.
        centroids = {}
        within_class_var = {}
        for ci, c in enumerate(classes):
            mask = (y_all == ci)
            n = mask.sum().item()
            if n == 0:
                centroids[c] = None
                within_class_var[c] = None
                continue
            class_x = x[mask]
            centroid = class_x.mean(dim=0)
            centroids[c] = centroid
            within_class_var[c] = ((class_x - centroid) ** 2).sum(dim=1).mean().item()

        # Between-class centroid distances + cosine similarity matrix.
        present_classes = [c for c in classes if centroids[c] is not None]
        between_class_dist = {}
        cosine_sim = {}
        for a in present_classes:
            between_class_dist[a] = {}
            cosine_sim[a] = {}
            for b in present_classes:
                dist = torch.norm(centroids[a] - centroids[b]).item()
                cos = torch.nn.functional.cosine_similarity(centroids[a].unsqueeze(0), centroids[b].unsqueeze(0)).item()
                between_class_dist[a][b] = round(dist, 4)
                cosine_sim[a][b] = round(cos, 4)

        # Cluster compactness: mean within-class variance / mean between-class distance.
        mean_within = sum(v for v in within_class_var.values() if v is not None) / len(present_classes)
        off_diag_dists = [between_class_dist[a][b] for a in present_classes for b in present_classes if a != b]
        mean_between = sum(off_diag_dists) / len(off_diag_dists) if off_diag_dists else None
        compactness_ratio = mean_within / mean_between if mean_between else None

        # Nearest-neighbour label agreement - pairwise cosine distance,
        # excluding self, O(n^2) but n=332 for the test split, trivial.
        x_norm = torch.nn.functional.normalize(x, dim=1)
        sim_matrix = x_norm @ x_norm.T
        sim_matrix.fill_diagonal_(-2.0)  # exclude self as its own neighbour
        nn_idx = sim_matrix.argmax(dim=1)
        nn_agree = (y_all[nn_idx] == y_all).float().mean().item()

        # Per-class NN agreement too - a class that's well-separated
        # should show near-100% NN label agreement even if overall
        # probe accuracy for that class is lower (a separability check
        # independent of the linear probe's own decision boundary).
        nn_agree_per_class = {}
        for ci, c in enumerate(classes):
            mask = (y_all == ci)
            if mask.sum().item() == 0:
                nn_agree_per_class[c] = None
                continue
            nn_agree_per_class[c] = (y_all[nn_idx][mask] == y_all[mask]).float().mean().item()

        payload = {
            "location": loc,
            "n": x.shape[0],
            "dim": x.shape[1],
            "pca_explained_variance_ratio_top2": [round(v, 4) for v in explained_ratio],
            "within_class_variance": {c: round(v, 4) if v is not None else None for c, v in within_class_var.items()},
            "between_class_centroid_distance": between_class_dist,
            "cosine_similarity_centroids": cosine_sim,
            "cluster_compactness_ratio": round(compactness_ratio, 4) if compactness_ratio else None,
            "nearest_neighbour_label_agreement_overall": round(nn_agree, 4),
            "nearest_neighbour_label_agreement_per_class": {
                c: round(v, 4) if v is not None else None for c, v in nn_agree_per_class.items()
            },
            "note": ("cluster_compactness_ratio = mean within-class variance / mean between-class distance - "
                     "LOWER means tighter, more separated clusters. nearest_neighbour_label_agreement is a "
                     "separability check independent of the linear probe's own decision boundary."),
        }
        out_path = STATS_DIR / f"{loc}.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        pca_csv_path = STATS_DIR / f"{loc}_pca2d.csv"
        with open(pca_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["review_id", "path", "gt", "pc1", "pc2"])
            for i in range(x.shape[0]):
                writer.writerow([review_ids[i], paths_all[i], classes[y_all[i].item()],
                                  round(projection[i, 0].item(), 4), round(projection[i, 1].item(), 4)])

        print(f"  PCA top-2 explained variance: {[round(v,3) for v in explained_ratio]}")
        print(f"  cluster compactness ratio: {round(compactness_ratio, 4) if compactness_ratio else None}")
        print(f"  NN label agreement (overall): {round(nn_agree, 4)}")

    print(f"\nPer-location stats + 2D PCA projections written to {STATS_DIR}/")


if __name__ == "__main__":
    main()
