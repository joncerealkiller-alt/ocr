"""
Phase 7 of the Stage-4 error-analysis pipeline (2026-08-08):
reproducibility manifest. Every phase (1-6) already writes CSV/JSON only
(no pickle, no framework-specific formats) - this script just walks
data/outputs/error_analysis/, records what exists, its format, row/item
counts where cheap to compute, and which phase produced it, so a later
GPU-inference phase knows exactly what it can append to without
re-deriving anything from scratch.

Usage:
    python diagnostics/error_analysis/phase7_manifest.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics.error_analysis.common import OUT_DIR

MANIFEST_PATH = OUT_DIR / "manifest.json"

# (relative path pattern, phase, description) - hand-maintained rather
# than inferred from filenames, so the manifest's descriptions stay
# accurate even if a future phase's internal naming changes slightly.
KNOWN_OUTPUTS = [
    ("review_table.csv", 1, "One row per image (all splits): path, gt, per-location pred/conf/correct, gemma_gen cross-reference."),
    ("embeddings_index.json", 1, "review_id -> {shard_file, row_idx} for looking up raw embeddings without duplicating tensors."),
    ("probe_softmax_probs.pt", 1, "review_id -> {location: full softmax distribution} (torch convenience copy; see .json for the CSV/JSON-preferred version)."),
    ("probe_softmax_probs.json", 1, "Same content as probe_softmax_probs.pt, plain JSON."),
    ("phase1_inconsistency_report.json", 1, "Every shard-consistency anomaly found (duplicates, drops, gaps, row-count mismatches)."),
    ("failure_taxonomy_schema.json", 2, "Machine-readable primary/secondary cause definitions + pending_review sentinel."),
    ("failure_annotations.json", 2, "One entry per (test-split failure, location) - starts pending_review, filled in by manual review."),
    ("phase3_prediction_stats/", 3, "Per (split, location): confusion matrix, precision/recall/F1, confidence histogram, calibration, class imbalance."),
    ("phase3_summary.csv", 3, "One row per (split, location): accuracy, macro-F1, mean confidence correct/incorrect."),
    ("phase4_embedding_stats/", 4, "Per location: PCA explained variance, centroids, within/between-class distance, NN label agreement, cosine similarity."),
    ("phase5_cross_layer_agreement/", 5, "Pairwise agreement rates, per-image behavior across all locations + gemma_gen, disagreement sets (test split)."),
    ("phase6_decision_rules.json", 6, "Proposed (unvalidated) thresholds mapping failure-cause distribution to recommended actions."),
    ("phase6_decision_report.json", 6, "Rule verdicts evaluated against current failure_annotations.json review progress."),
]


def count_rows(path: Path) -> int | None:
    if path.suffix == ".csv":
        with open(path, encoding="utf-8") as f:
            return sum(1 for _ in f) - 1  # minus header
    if path.suffix == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return len(data)
            if isinstance(data, dict):
                return len(data)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
    return None


def main():
    entries = []
    for rel_pattern, phase, description in KNOWN_OUTPUTS:
        target = OUT_DIR / rel_pattern
        if rel_pattern.endswith("/"):
            if not target.exists():
                entries.append({"path": rel_pattern, "phase": phase, "description": description,
                                 "exists": False, "files": []})
                continue
            files = sorted(p.name for p in target.iterdir() if p.is_file())
            entries.append({
                "path": rel_pattern, "phase": phase, "description": description,
                "exists": True, "n_files": len(files), "files": files,
            })
        else:
            if not target.exists():
                entries.append({"path": rel_pattern, "phase": phase, "description": description, "exists": False})
                continue
            entries.append({
                "path": rel_pattern, "phase": phase, "description": description,
                "exists": True, "size_bytes": target.stat().st_size,
                "format": target.suffix.lstrip("."),
                "n_rows_or_items": count_rows(target),
                "not_csv_json": target.suffix not in (".csv", ".json"),
            })

    manifest = {
        "generated_by": "diagnostics/error_analysis/phase7_manifest.py",
        "out_dir": str(OUT_DIR),
        "phases_completed": sorted(set(e["phase"] for e in entries if e["exists"])),
        "phases_missing": sorted(set(e["phase"] for e in entries if not e["exists"])),
        "outputs": entries,
        "deferred_to_gpu_phase": [
            "new Gemma inference", "generation fallback testing", "per-image reasoning runs",
            "decoder probing requiring new forward passes", "softmax recomputation on new data",
            "new embedding generation", "vision encoder execution",
            "installation of optional visualization packages (scikit-learn TSNE, umap-learn)",
        ],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Manifest written to {MANIFEST_PATH}\n")
    print(f"Phases completed: {manifest['phases_completed']}")
    print(f"Phases missing:   {manifest['phases_missing']}\n")
    for e in entries:
        status = "OK" if e["exists"] else "MISSING"
        note = " (NOT CSV/JSON - torch tensor file)" if e.get("not_csv_json") else ""
        print(f"  [{status:>7s}] phase {e['phase']}  {e['path']}{note}")


if __name__ == "__main__":
    main()
