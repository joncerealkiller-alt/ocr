"""
Shared machinery for the Stage-4 error-analysis pipeline (all phases in
this directory). CPU-only, no Gemma/vision model execution anywhere in
this module - every function here operates on tensors already cached to
disk by diagnostics/gemma_hidden_state_probe_stage2.py (shards_stage2/*.pt)
and diagnostics/run_gemma_flat8_on_benchmark_holdout.py
(gemma_flat8_test_predictions.json).

"Prediction" in this pipeline means running the already-trained-shape
LINEAR PROBE (a single nn.Linear, trained fresh each call on cached
features via diagnostics/gemma_hidden_state_linear_probe_stage1.py's
train_linear_probe()) forward on cached embeddings - a few milliseconds
of CPU matrix multiplication, not a new Gemma/vision-model forward pass.
This is deliberately distinct from "new model inference" in the sense the
handoff prompt defers (no Gemma, no vision tower, no new embeddings) -
the embeddings themselves are 100% reused from the existing cache.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Single source of truth for where this legacy corpus's images live,
# post genealogy_workspace migration (docs/RUN_ARCHITECTURE.md, 2026-08-08).
# Phase 1 of that migration moved data/working/ into
# genealogy_workspace/runs/legacy_pre_run_system/working/images/ with NO
# junction left behind (unlike Phase 2's data/outputs/ treatment) - every
# cached artifact from this session's probe/error-analysis work that
# stored an absolute "...\data\working\<filename>" path went stale as a
# result (confirmed: 1749/2232 review_table.csv paths, 78.4%, failed
# Path.exists() before diagnostics/error_analysis/rewrite_legacy_paths.py
# fixed them). ANY future code that needs to resolve one of THESE
# specific legacy paths, or extend this corpus, should import
# LEGACY_WORKING_IMAGES_DIR from here rather than hardcoding
# "data/working" again (which no longer exists) or re-deriving the new
# location from scratch.
LEGACY_WORKSPACE_ROOT = PROJECT_ROOT.parent / "genealogy_workspace"
LEGACY_RUN_ID = "legacy_pre_run_system"
LEGACY_WORKING_IMAGES_DIR = LEGACY_WORKSPACE_ROOT / "runs" / LEGACY_RUN_ID / "working" / "images"
OLD_WORKING_PREFIX = str(PROJECT_ROOT / "data" / "working")

SHARD_DIR = PROJECT_ROOT / "data" / "outputs" / "gemma_hidden_state_probe" / "shards_stage2"
FEATURES_PATH = PROJECT_ROOT / "data" / "outputs" / "gemma_hidden_state_probe" / "features_stage2.pt"
GEMMA_GEN_PREDICTIONS_PATH = (
    PROJECT_ROOT / "data" / "outputs" / "vit_family_benchmark" / "gemma_flat8" / "gemma_flat8_test_predictions.json"
)
OUT_DIR = PROJECT_ROOT / "data" / "outputs" / "error_analysis"

LOCATIONS = ["vision_encoder", "vision_projected", "decoder_early", "decoder_mid", "decoder_final"]

SHARD_FILENAME_RE = re.compile(r"^(train|val|test)_(\d{5})_(\d{5})\.pt$")


def parse_shard_filename(path: Path) -> tuple[str, int, int]:
    """Extracts (split, start, end) from a shard filename like
    'test_00080_00120.pt' - the split/index-range encoding
    gemma_hidden_state_probe_stage2.py's extract_via_subprocess_chunks()
    uses. Raises ValueError on an unrecognized filename rather than
    guessing, since silently misparsing a shard's split would corrupt
    every downstream table."""
    m = SHARD_FILENAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Shard filename doesn't match expected 'SPLIT_START_END.pt' pattern: {path.name}")
    split, start, end = m.group(1), int(m.group(2)), int(m.group(3))
    return split, start, end


def load_all_shards() -> list[dict[str, Any]]:
    """Loads every shard in SHARD_DIR, attaching parsed (split, start, end,
    shard_file) metadata to each. Does NOT assume shards are complete,
    contiguous, or in any particular order - Phase 1's consistency checks
    are what actually verify that; this function just loads raw data."""
    if not SHARD_DIR.exists():
        raise FileNotFoundError(
            f"{SHARD_DIR} not found - run diagnostics/gemma_hidden_state_probe_stage2.py first "
            "(this pipeline only reads its cached shards, never re-runs inference)."
        )
    shard_paths = sorted(SHARD_DIR.glob("*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard files found in {SHARD_DIR}.")

    shards = []
    for shard_path in shard_paths:
        split, start, end = parse_shard_filename(shard_path)
        data = torch.load(shard_path)
        shards.append({
            "shard_file": shard_path.name,
            "split": split,
            "start": start,
            "end": end,
            "paths": data["paths"],
            "y": data["y"],
            "x_by_loc": data["x_by_loc"],
            "classes": data["classes"],
        })
    return shards


def train_probe_and_predict_all(train_x, train_y, val_y, eval_x_by_split: dict[str, torch.Tensor],
                                 n_classes: int, seed: int = 0):
    """
    Trains ONE linear probe (seed fixed for reproducibility - this
    pipeline's job is building infrastructure/tables, not re-litigating
    the multi-seed variance question already answered by
    gemma_hidden_state_probe_multiseed.py) on train_x/train_y, then
    returns (preds, confidences, full_softmax_probs) for every split in
    eval_x_by_split, e.g. {"train": ..., "val": ..., "test": ...}.

    Reuses diagnostics/gemma_hidden_state_linear_probe_stage1.py's
    train_linear_probe() as-is (same standardization/class-weighting/
    checkpoint-selection logic) rather than reimplementing it - only new
    code here is exposing PER-IMAGE predictions/probabilities, which that
    script's evaluate() computes internally but only aggregates and
    discards.
    """
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from diagnostics.gemma_hidden_state_linear_probe_stage1 import train_linear_probe

    # Stage1's train_linear_probe() needs a val split for checkpoint
    # selection - reuse whichever eval split is named "val" if present,
    # else fall back to holding out the last 15% of train (keeps this
    # function usable even if a caller only has train+test).
    if "val" not in eval_x_by_split:
        raise ValueError("train_probe_and_predict_all() requires a 'val' split for checkpoint selection.")
    val_x = eval_x_by_split["val"]

    torch.manual_seed(seed)

    def _quiet_log(s=""):
        pass

    probe, mean, std = train_linear_probe(train_x, train_y, val_x, val_y, n_classes, _quiet_log)

    results = {}
    for split_name, x in eval_x_by_split.items():
        x_n = (x - mean) / std
        probe.eval()
        with torch.no_grad():
            logits = probe(x_n)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            confs = probs.max(dim=1).values
        results[split_name] = {"preds": preds, "confidences": confs, "probs": probs}
    return results, probe, mean, std
