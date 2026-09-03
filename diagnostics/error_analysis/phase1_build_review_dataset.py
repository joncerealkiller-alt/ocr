"""
Phase 1 of the Stage-4 error-analysis pipeline (2026-08-08): build a
validated, per-image review table from cached artifacts only. No Gemma/
vision inference - resolves image paths from the Stage 2 SHARD files
(shards_stage2/*.pt), never assumes index alignment with a fresh
get_split() call, per the real gap found while writing docs/
GEMMA_HIDDEN_STATE_ERROR_ANALYSIS_STAGE4_RESEARCH.md (Stage 1's run
dropped one image to a FileNotFoundError - if paths weren't tracked
per-shard, that would have silently shifted index alignment for every
downstream table).

"Predictions" here come from a linear probe trained fresh on the cached
train-split embeddings (see common.py's docstring for why this doesn't
count as "new model inference" under the handoff prompt's scope) - one
probe per hook location, so the review table can show all 5 locations'
predictions side by side.

Outputs (all under data/outputs/error_analysis/):
    review_table.csv           - one row per image, wide format
    embeddings_index.json      - review_id -> {shard_file, row_idx} for
                                  later embedding lookups without
                                  duplicating tensors into the CSV
    probe_softmax_probs.pt     - {review_id: {location: [n_classes floats]}}
    phase1_inconsistency_report.json - every anomaly found, not silently fixed

Usage:
    python diagnostics/error_analysis/phase1_build_review_dataset.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from diagnostics.error_analysis.common import (
    LOCATIONS, OUT_DIR, GEMMA_GEN_PREDICTIONS_PATH,
    load_all_shards, train_probe_and_predict_all,
)

REVIEW_TABLE_PATH = OUT_DIR / "review_table.csv"
EMBEDDINGS_INDEX_PATH = OUT_DIR / "embeddings_index.json"
PROBE_PROBS_PATH = OUT_DIR / "probe_softmax_probs.pt"
INCONSISTENCY_REPORT_PATH = OUT_DIR / "phase1_inconsistency_report.json"


def check_shard_consistency(shards: list[dict]) -> dict:
    """
    Every anomaly is REPORTED, not silently corrected - a review table
    built on top of a silently-patched inconsistency is worse than one
    that flags the problem, per this project's own "stale entry is worse
    than no entry" principle (docs/CODE_MAP.md's own stated philosophy,
    applied here to data integrity instead of documentation).
    """
    issues = {
        "dropped_samples": [],       # shard covers fewer rows than its filename range implies
        "duplicate_paths": [],       # same image path appears more than once (within or across shards)
        "overlapping_ranges": [],    # two shards for the same split claim overlapping index ranges
        "gaps_in_ranges": [],        # a split's shard ranges don't tile [0, max) with no gaps
        "row_count_mismatches": [],  # len(paths) != len(y) != x_by_loc[loc].shape[0] within one shard
        "missing_locations": [],     # a shard doesn't have all 5 expected locations
    }

    path_to_shards: dict[str, list[str]] = defaultdict(list)
    ranges_by_split: dict[str, list[tuple[int, int, str]]] = defaultdict(list)

    for shard in shards:
        expected_n = shard["end"] - shard["start"]
        actual_n = len(shard["paths"])
        if actual_n != expected_n:
            issues["dropped_samples"].append({
                "shard_file": shard["shard_file"], "split": shard["split"],
                "expected": expected_n, "actual": actual_n,
                "dropped": expected_n - actual_n,
            })

        n_y = shard["y"].numel()
        loc_counts = {loc: shard["x_by_loc"][loc].shape[0] for loc in LOCATIONS if loc in shard["x_by_loc"]}
        if not (actual_n == n_y == (loc_counts.get(LOCATIONS[0], -1))) or len(set(loc_counts.values())) > 1:
            issues["row_count_mismatches"].append({
                "shard_file": shard["shard_file"], "n_paths": actual_n, "n_y": n_y, "loc_counts": loc_counts,
            })

        missing_locs = [loc for loc in LOCATIONS if loc not in shard["x_by_loc"]]
        if missing_locs:
            issues["missing_locations"].append({"shard_file": shard["shard_file"], "missing": missing_locs})

        for p in shard["paths"]:
            path_to_shards[p].append(shard["shard_file"])

        ranges_by_split[shard["split"]].append((shard["start"], shard["end"], shard["shard_file"]))

    for p, shard_files in path_to_shards.items():
        if len(shard_files) > 1:
            issues["duplicate_paths"].append({"path": p, "shard_files": shard_files})

    for split, ranges in ranges_by_split.items():
        ranges_sorted = sorted(ranges, key=lambda r: r[0])
        cursor = 0
        for start, end, shard_file in ranges_sorted:
            if start < cursor:
                issues["overlapping_ranges"].append({
                    "split": split, "shard_file": shard_file, "start": start, "cursor_at_time": cursor,
                })
            elif start > cursor:
                issues["gaps_in_ranges"].append({
                    "split": split, "gap_start": cursor, "gap_end": start, "before_shard_file": shard_file,
                })
            cursor = max(cursor, end)

    return issues


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading all Stage 2 shards...")
    shards = load_all_shards()
    print(f"Loaded {len(shards)} shards.")

    print("Checking shard consistency (duplicates, drops, ordering, row-count mismatches)...")
    issues = check_shard_consistency(shards)
    n_issues = sum(len(v) for v in issues.values())
    print(f"Found {n_issues} total anomalies across {len(issues)} categories:")
    for k, v in issues.items():
        print(f"  {k}: {len(v)}")
    INCONSISTENCY_REPORT_PATH.write_text(json.dumps(issues, indent=2), encoding="utf-8")
    print(f"Full inconsistency report: {INCONSISTENCY_REPORT_PATH}\n")

    classes = shards[0]["classes"]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # Build per-split concatenated tensors, in shard-start order (NOT
    # assumed contiguous - just sorted, matching whatever the shards
    # actually contain, drops and all, since check_shard_consistency()
    # already surfaced any gaps/drops rather than papering over them).
    shards_by_split = defaultdict(list)
    for s in shards:
        shards_by_split[s["split"]].append(s)
    for split in shards_by_split:
        shards_by_split[split].sort(key=lambda s: s["start"])

    x_by_loc_by_split = {}
    y_by_split = {}
    paths_by_split = {}
    review_id_by_split = {}  # review_id -> (shard_file, row_idx_in_shard)
    rid_counter = 0

    for split, split_shards in shards_by_split.items():
        x_by_loc_by_split[split] = {loc: [] for loc in LOCATIONS}
        ys, paths, rids = [], [], []
        for shard in split_shards:
            n = len(shard["paths"])
            for i in range(n):
                rids.append((shard["shard_file"], i))
            for loc in LOCATIONS:
                x_by_loc_by_split[split][loc].append(shard["x_by_loc"][loc])
            ys.append(shard["y"])
            paths.extend(shard["paths"])
        x_by_loc_by_split[split] = {loc: torch.cat(v, dim=0) for loc, v in x_by_loc_by_split[split].items()}
        y_by_split[split] = torch.cat(ys, dim=0)
        paths_by_split[split] = paths
        review_id_by_split[split] = rids

    for split in ("train", "val", "test"):
        if split not in y_by_split:
            raise ValueError(f"No shards found for split={split!r} - cannot build a complete review table.")

    print(f"train={len(paths_by_split['train'])} val={len(paths_by_split['val'])} "
          f"test={len(paths_by_split['test'])} (post-dedup-check, reflects actual cached rows)\n")

    # Train one probe per location, predict on ALL splits (including
    # train, for the label-quality-audit use case docs/
    # GEMMA_HIDDEN_STATE_ERROR_ANALYSIS_STAGE4_RESEARCH.md section 5
    # describes - flagged as in-sample in the output table, not treated
    # as a held-out accuracy number).
    print("Training one linear probe per location on cached train features (CPU only, no Gemma)...")
    predictions_by_loc = {}
    for loc in LOCATIONS:
        eval_x_by_split = {
            "train": x_by_loc_by_split["train"][loc],
            "val": x_by_loc_by_split["val"][loc],
            "test": x_by_loc_by_split["test"][loc],
        }
        results, probe, mean, std = train_probe_and_predict_all(
            x_by_loc_by_split["train"][loc], y_by_split["train"], y_by_split["val"],
            eval_x_by_split, len(classes), seed=0,
        )
        predictions_by_loc[loc] = results
        print(f"  {loc}: done")
    print()

    # Load full-generation reference predictions (real cached data, keyed
    # by absolute path) for the cross-check join.
    gemma_gen = {}
    if GEMMA_GEN_PREDICTIONS_PATH.exists():
        gemma_gen_raw = json.loads(GEMMA_GEN_PREDICTIONS_PATH.read_text(encoding="utf-8"))
        gemma_gen = gemma_gen_raw["predictions"]
        print(f"Loaded {len(gemma_gen)} full-generation reference predictions from {GEMMA_GEN_PREDICTIONS_PATH}")
    else:
        print(f"WARNING: {GEMMA_GEN_PREDICTIONS_PATH} not found - gemma_gen_* columns will be empty.")

    # Assemble the wide review table.
    rows = []
    embeddings_index = {}
    probe_probs_out = {}
    duplicate_paths_seen = set(d["path"] for d in issues["duplicate_paths"])

    for split in ("train", "val", "test"):
        paths = paths_by_split[split]
        ys = y_by_split[split]
        rids = review_id_by_split[split]
        for i, path in enumerate(paths):
            review_id = f"{split}_{i:05d}"
            gt_idx = ys[i].item()
            gt_class = classes[gt_idx]
            shard_file, row_idx = rids[i]

            row = {
                "review_id": review_id,
                "path": path,
                "split": split,
                "gt": gt_class,
                "shard_file": shard_file,
                "shard_row_idx": row_idx,
                "duplicate_path": path in duplicate_paths_seen,
            }
            probe_probs_out[review_id] = {}
            for loc in LOCATIONS:
                pred_idx = predictions_by_loc[loc][split]["preds"][i].item()
                conf = predictions_by_loc[loc][split]["confidences"][i].item()
                probs = predictions_by_loc[loc][split]["probs"][i].tolist()
                pred_class = classes[pred_idx]
                row[f"pred_{loc}"] = pred_class
                row[f"conf_{loc}"] = round(conf, 4)
                row[f"correct_{loc}"] = (pred_class == gt_class)
                probe_probs_out[review_id][loc] = probs

            gg = gemma_gen.get(path)
            if gg is not None:
                row["gemma_gen_pred"] = gg.get("pred")
                row["gemma_gen_conf"] = gg.get("confidence")
                row["gemma_gen_correct"] = gg.get("correct")
                row["in_gemma_gen_set"] = True
            else:
                row["gemma_gen_pred"] = ""
                row["gemma_gen_conf"] = ""
                row["gemma_gen_correct"] = ""
                row["in_gemma_gen_set"] = False

            rows.append(row)
            embeddings_index[review_id] = {"shard_file": shard_file, "row_idx": row_idx, "locations": LOCATIONS}

    fieldnames = list(rows[0].keys())
    with open(REVIEW_TABLE_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {REVIEW_TABLE_PATH}")

    EMBEDDINGS_INDEX_PATH.write_text(json.dumps(embeddings_index, indent=2), encoding="utf-8")
    print(f"Wrote embeddings index to {EMBEDDINGS_INDEX_PATH}")

    torch.save(probe_probs_out, PROBE_PROBS_PATH)
    # Phase 7's reproducibility requirement prefers CSV/JSON over
    # framework-specific formats - the .pt file above is kept too (more
    # convenient for a later PyTorch-based consumer), but a plain-JSON
    # export ensures every Phase 1 output is readable without torch
    # installed at all.
    probe_probs_json_path = PROBE_PROBS_PATH.with_suffix(".json")
    probe_probs_json_path.write_text(json.dumps(probe_probs_out, indent=2), encoding="utf-8")
    print(f"Wrote per-image softmax probabilities to {PROBE_PROBS_PATH} and {probe_probs_json_path}")

    # Quick summary so this phase's own output is auditable at a glance.
    n_in_gemma_gen = sum(1 for r in rows if r["in_gemma_gen_set"])
    print(f"\n{n_in_gemma_gen}/{len(rows)} rows have a matching full-generation reference prediction "
          "(gemma_flat8 only covers the test split, so this is expected to be << total row count).")
    test_rows = [r for r in rows if r["split"] == "test"]
    for loc in LOCATIONS:
        n_correct = sum(1 for r in test_rows if r[f"correct_{loc}"])
        print(f"  test accuracy sanity check, {loc}: {n_correct}/{len(test_rows)} "
              f"({100*n_correct/len(test_rows):.1f}%) - single-seed, expect it near but not "
              "identical to the multi-seed means already measured.")


if __name__ == "__main__":
    main()
