"""
Fresh Stage 0-3 pass (2026-08-04) - controlled rerun against the EXACT
same 1750 source files as the checkpointed reference_pipeline_v3 corpus
(data/outputs/reference_pipeline_v3/source_paths_for_fresh_run.json),
to measure whether this session's changes (GPU-sharded k=4 semantic
capture, the 3 new taxonomy categories, the taxonomy-driven prompt
template) improved or regressed the pipeline - holding the input corpus
constant is what makes this a controlled comparison rather than a
different-corpus comparison.

Usage:
    python -m scripts.run_fresh_pass_stage0to3
"""
from __future__ import annotations

import json
from pathlib import Path

from core.manifest_pipeline import build_working_manifest_from_paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_PATHS_JSON = PROJECT_ROOT / "data" / "outputs" / "reference_pipeline_v3" / "source_paths_for_fresh_run.json"


def main() -> None:
    paths = [Path(p) for p in json.loads(SOURCE_PATHS_JSON.read_text(encoding="utf-8"))]
    print(f"Running Stage 0-3 against {len(paths)} source files (same corpus as reference_pipeline_v3)...")
    build_working_manifest_from_paths(paths)
    print("\nStage 0-3 complete.")


if __name__ == "__main__":
    main()
