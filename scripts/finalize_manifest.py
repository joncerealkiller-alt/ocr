"""
Stage 2 CLI: merges core/classifier.py's bucket CSVs (and, for
dense_tabular_rows, ui/dewarp_preprocessor_ui.py's own dewarped-bucket
CSV) into one final manifest recording each file's real, ready-to-use
image path - see core/manifest_pipeline.py's module docstring for the
full pipeline-stage story and status meanings.

Safe to re-run anytime (e.g. after running more pages through the dewarp
UI) - always rebuilds from the current state of every bucket CSV, never
accumulates/appends.

Usage:
    python scripts/finalize_manifest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.manifest_pipeline import finalize_manifest


def main():
    finalize_manifest()


if __name__ == "__main__":
    main()
