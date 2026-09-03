"""
Runs the production capture_baseline_embeddings() (GPU-sharded k=4,
see core/baseline_embeddings.py) against the reconstructed true-raw
copies in data/raw_stage0_recapture/ (see scripts/reconstruct_raw_
stage0.py - every copy hash-verified against pipeline.db's identity_hash
before this script trusts it).

Writes to a SCRATCH output path first, never directly to the production
data/baseline_embeddings.json - that file was destroyed twice earlier
this session (see docs/GPU_CPU_EQUIVALENCE_REPORT.md) and is not to be
written to again without an explicit verified-then-promote step.

capture_baseline_embeddings() records "image" as whatever path it was
given - since it was given the RAW COPY paths (not the working_path the
rest of the pipeline looks records up by), this script remaps each
record's "image" field to its corresponding working_path afterward,
using the mapping scripts/reconstruct_raw_stage0.py already wrote.

Usage:
    python -m scripts.recapture_baseline_embeddings_raw
"""
from __future__ import annotations

import json
from pathlib import Path

from core.baseline_embeddings import write_baseline_embeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAPPING_PATH = PROJECT_ROOT / "data" / "raw_stage0_recapture_mapping.json"
SCRATCH_OUTPUT = PROJECT_ROOT / "data" / "outputs" / "baseline_embeddings_raw_recapture" / "raw_paths.json"
FINAL_SCRATCH_OUTPUT = PROJECT_ROOT / "data" / "outputs" / "baseline_embeddings_raw_recapture" / "remapped.json"


def main() -> None:
    mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))  # raw_copy_path -> working_path
    raw_paths = [Path(p) for p in mapping.keys()]
    print(f"Capturing baseline embeddings for {len(raw_paths)} TRUE RAW images "
          f"(reconstructed from original sources)...")

    SCRATCH_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    write_baseline_embeddings(raw_paths, output_path=SCRATCH_OUTPUT)

    print(f"\nRemapping 'image' field: raw copy path -> working_path...")
    records = json.loads(SCRATCH_OUTPUT.read_text(encoding="utf-8"))
    remapped = []
    unmatched = 0
    for r in records:
        working_path = mapping.get(r["image"])
        if working_path is None:
            unmatched += 1
            continue
        r["image"] = working_path
        remapped.append(r)

    print(f"Remapped: {len(remapped)}/{len(records)} (unmatched: {unmatched})")
    FINAL_SCRATCH_OUTPUT.write_text(json.dumps(remapped), encoding="utf-8")
    print(f"Written: {FINAL_SCRATCH_OUTPUT}")


if __name__ == "__main__":
    main()
