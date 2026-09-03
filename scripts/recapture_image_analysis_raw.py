"""
Fixes the physical-sensor ("Stage 1 image_analysis") mislabeling bug
found 2026-08-04 while comparing Stage 1 vs Stage 4: data/outputs/
image_analysis/analysis_report.csv (and every <name>_analysis.json
sidecar in data/working/) was captured against ALREADY-Stage-3-processed
pixels, mislabeled as pre-processing - confirmed by direct comparison
(stored blur_laplacian_var for e001928017.png/e001946014.png/
e001946614.png matches data/working's CURRENT pixels exactly, not the
reconstructed raw ones - same bug class as docs/REFERENCE_PIPELINE_V1.md,
previously fixed for the semantic embeddings this session, not yet for
physical).

Reuses core/image_analysis.py's analyze_manifest() UNCHANGED (same
8-worker/4-cv2-thread concurrency engine already validated for Stage 1)
against data/raw_stage0_recapture/ (1750 images, hash-verified against
pipeline.db's identity_hash by scripts/reconstruct_raw_stage0.py earlier
this session - reused here, not rebuilt).

write_sidecars=True is used so analyze_manifest() writes its sidecars
next to the RAW COPIES (data/raw_stage0_recapture/<name>_analysis.json) -
deliberately NOT next to the real working images, to avoid a half-
written intermediate state at the real Stage 1 location. This script
then does the two promotion steps explicitly, after verifying the run
succeeded:
  1. copies each sidecar to data/working/<name>_analysis.json (the real
     Stage 1 location), rewriting its own "file_path" field from the
     raw-copy path to the working path first, so the sidecar's own
     content is honest about which image it identifies as, even though
     the pixels it measured were the raw ones.
  2. remaps the report CSV's file_path column the same way and
     overwrites data/outputs/image_analysis/analysis_report.csv (Stage
     1's canonical report).

db writes: analyze_manifest()'s own DB wiring does nothing during the
measurement pass (raw-copy paths are never registered in pipeline.db -
db.get_image_by_path() returns None for all of them, and every DB write
in that function is gated on that lookup succeeding) - this script
records the DB stage_outputs event itself afterward, using the REMAPPED
working_path, with an explicit note distinguishing this from the
original (mislabeled) capture rather than leaving the two
indistinguishable in the append-only stage_outputs log.

Usage:
    python -m scripts.recapture_image_analysis_raw
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from core.image_analysis import analyze_manifest, DEFAULT_REPORT_PATH, analysis_sidecar_path
from core.pipeline_db import PipelineDatabase, DEFAULT_DB_PATH

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAPPING_PATH = PROJECT_ROOT / "data" / "raw_stage0_recapture_mapping.json"
RAW_DIR = PROJECT_ROOT / "data" / "raw_stage0_recapture"
SCRATCH_MANIFEST = PROJECT_ROOT / "data" / "outputs" / "image_analysis_raw_recapture" / "raw_manifest.csv"
SCRATCH_REPORT = PROJECT_ROOT / "data" / "outputs" / "image_analysis_raw_recapture" / "raw_report.csv"
FINAL_REMAPPED_REPORT = PROJECT_ROOT / "data" / "outputs" / "image_analysis_raw_recapture" / "remapped_report.csv"


def main() -> None:
    mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))  # raw_copy_path -> working_path
    print(f"{len(mapping)} raw copies available (from scripts/reconstruct_raw_stage0.py).")

    SCRATCH_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(SCRATCH_MANIFEST, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_path"])
        for raw_path in mapping:
            writer.writerow([raw_path])

    print("Running analyze_manifest() against TRUE RAW images "
          "(same 8-worker/4-cv2-thread engine as Stage 1)...")
    analyze_manifest(
        SCRATCH_MANIFEST,
        report_path=SCRATCH_REPORT,
        write_sidecars=True,
        db_path=DEFAULT_DB_PATH,
        stage="stage1_image_analysis_raw_recapture_pass",  # irrelevant: no DB writes occur (paths unregistered)
    )

    print("\nPromoting sidecars: raw-copy location -> working-image location, remapping file_path field...")
    n_sidecars = 0
    for raw_path_str, working_path_str in mapping.items():
        raw_sidecar = analysis_sidecar_path(raw_path_str)
        if not raw_sidecar.exists():
            continue
        record = json.loads(raw_sidecar.read_text(encoding="utf-8"))
        record["file_path"] = working_path_str
        working_sidecar = analysis_sidecar_path(working_path_str)
        working_sidecar.write_text(json.dumps(record, indent=2), encoding="utf-8")
        n_sidecars += 1
    print(f"  {n_sidecars}/{len(mapping)} sidecars promoted to data/working/.")

    print("\nRemapping report CSV file_path column, then promoting to Stage 1's canonical report...")
    with open(SCRATCH_REPORT, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    n_remapped = 0
    for row in rows:
        working_path = mapping.get(row["file_path"])
        if working_path is not None:
            row["file_path"] = working_path
            n_remapped += 1
    with open(FINAL_REMAPPED_REPORT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {n_remapped}/{len(rows)} rows remapped.")

    shutil.copyfile(FINAL_REMAPPED_REPORT, DEFAULT_REPORT_PATH)
    print(f"  Promoted to {DEFAULT_REPORT_PATH}")

    print("\nRecording stage_outputs events (stage='stage1_image_analysis', using the REMAPPED working_path)...")
    db = PipelineDatabase(DEFAULT_DB_PATH)
    recorded, skipped = 0, 0
    for row in rows:
        working_path = row["file_path"]
        image = db.get_image_by_path(working_path)
        if image is None:
            skipped += 1
            continue
        db.record_stage_output(
            image["id"], stage="stage1_image_analysis",
            sidecar_path=str(analysis_sidecar_path(working_path)),
            status="done" if not row.get("error") else "failed",
            note="raw recapture 2026-08-04 - corrects earlier stage1_image_analysis capture, "
                 "which was inadvertently measured against already-Stage-3-processed pixels",
        )
        recorded += 1
    print(f"Recorded {recorded} corrected stage_outputs event(s) ({skipped} skipped - not in DB).")


if __name__ == "__main__":
    main()
