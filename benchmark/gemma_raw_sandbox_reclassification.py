"""
Full-sandbox Gemma reclassification, 2026-08-04, against the RAW
(pre-Stage-3) versions of the 77 "flagged" images from benchmark/
gained_lost_agreement_characterization.py (57 gained-agreement + 20
lost-agreement). Production only ever classifies POST-processed images
(core/classifier.py's run() reads data/working) - Gemma's own decision
on the RAW pixels for these specific images has never been measured.
This answers: is Gemma's own classification stable across preprocessing
for the images where the TOWER's classification was not, or does Gemma
flip too.

HARD ISOLATION - verified by construction, not just intent:
  - Input images are COPIED from data/raw_stage0_recapture/ (already
    hash-verified against pipeline.db's identity_hash) into a sandbox
    directory OUTSIDE the repo entirely (the session scratchpad) -
    the classifier never reads anything under data/working or the raw
    reconstruction directory directly.
  - Only core/classifier.py's build_classifier_loader() (model loading,
    read-only against config/pipeline.yaml and the prompt file) and
    loader.classify(path, image) (pure inference, no I/O beyond reading
    the image already in memory) are reused - NOT run(), NOT
    open_bucket_writers(), NOT _record_classification_db(), NOT
    PipelineDatabase. Those are the only functions in core/classifier.py
    that ever write anything, and none of them are called here.
  - All output (per-image classification, combined report) is written
    ONLY under the sandbox directory - never under data/.

Usage:
    python -m benchmark.gemma_raw_sandbox_reclassification
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from PIL import Image

from core.classifier import build_classifier_loader, load_pipeline_config, result_to_row

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw_stage0_recapture"
MAPPING_PATH = PROJECT_ROOT / "data" / "raw_stage0_recapture_mapping.json"
CHAR_DIR = PROJECT_ROOT / "data" / "outputs" / "gained_lost_agreement_characterization"

# Sandbox lives OUTSIDE the repo entirely - the session scratchpad, per
# this project's own convention for temporary/sandboxed work.
SANDBOX_DIR = Path(
    r"C:\Users\jonny\AppData\Local\Temp\claude\J--Genealogy-genealogy-pipeline"
    r"\9cd19cdb-4b6e-40e0-912e-730a45b5ec51\scratchpad\gemma_raw_sandbox"
)
SANDBOX_IMAGES_DIR = SANDBOX_DIR / "images"


def _load_flagged_images() -> list[dict]:
    rows = []
    for name in ("gained_agreement_detail.csv", "lost_agreement_detail.csv"):
        with open(CHAR_DIR / name, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r["_source_group"] = name.replace("_agreement_detail.csv", "")
                rows.append(r)
    return rows


def main() -> None:
    flagged = _load_flagged_images()
    print(f"{len(flagged)} flagged images (gained + lost agreement).")

    working_to_raw = {
        v: k for k, v in json.loads(MAPPING_PATH.read_text(encoding="utf-8")).items()
    }

    SANDBOX_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Sandbox directory: {SANDBOX_DIR}")

    copied = []
    for row in flagged:
        working_path = row["image"]
        raw_path = working_to_raw.get(working_path)
        if raw_path is None:
            print(f"  WARNING: no raw copy found for {working_path} - skipping")
            continue
        dest = SANDBOX_IMAGES_DIR / Path(raw_path).name
        shutil.copyfile(raw_path, dest)
        copied.append({"sandbox_path": dest, "working_path": working_path, "row": row})
    print(f"Copied {len(copied)}/{len(flagged)} raw images into the sandbox.")

    print("\nLoading Gemma (production model/prompt, read-only config access)...")
    pipeline_cfg = load_pipeline_config()
    loader = build_classifier_loader(pipeline_cfg)
    print("Loaded. Classifying sandboxed RAW images (no DB, no bucket CSV writes)...\n")

    results = []
    for i, item in enumerate(copied, 1):
        sandbox_path = item["sandbox_path"]
        print(f"[{i}/{len(copied)}] {sandbox_path.name}", end=" ")
        try:
            with Image.open(sandbox_path) as raw_image:
                result = loader.classify(str(sandbox_path), raw_image)
            row_out = result_to_row(result)
            print(f"-> {row_out['category']} (confidence={row_out['confidence']:.2f})")
        except Exception as e:
            row_out = {"category": None, "confidence": None, "error": f"{type(e).__name__}: {e}"}
            print(f"-> FAILED ({e})")

        source_row = item["row"]
        results.append({
            "image": item["working_path"],
            "group": source_row["_source_group"],
            "tower_top_bucket_pre": source_row["tower_top_bucket_pre"],
            "tower_top_bucket_post": source_row["tower_top_bucket_post"],
            "gemma_bucket_on_postprocessed_PRODUCTION": source_row["gemma_bucket"],
            "gemma_bucket_on_RAW_sandbox": row_out.get("category"),
            "gemma_confidence_on_RAW_sandbox": row_out.get("confidence"),
            "gemma_reason_on_RAW_sandbox": row_out.get("reason"),
            "error": row_out.get("error"),
        })

    print(f"\nWritten: nothing under data/ - all output below is sandbox-only.")
    out_csv = SANDBOX_DIR / "gemma_raw_reclassification_results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()) if results else [])
        writer.writeheader()
        writer.writerows(results)
    print(f"Results: {out_csv}")

    out_json = SANDBOX_DIR / "gemma_raw_reclassification_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Results (json): {out_json}")

    # Quick summary
    n_matches_raw_to_postprocessed_gemma = sum(
        1 for r in results
        if r["gemma_bucket_on_RAW_sandbox"] == r["gemma_bucket_on_postprocessed_PRODUCTION"]
    )
    n_matches_raw_to_tower_pre = sum(
        1 for r in results
        if r["gemma_bucket_on_RAW_sandbox"] == r["tower_top_bucket_pre"]
    )
    n_matches_raw_to_tower_post = sum(
        1 for r in results
        if r["gemma_bucket_on_RAW_sandbox"] == r["tower_top_bucket_post"]
    )
    print(f"\n=== Summary ({len(results)} images) ===")
    print(f"Gemma-on-raw matches Gemma-on-postprocessed (production): "
          f"{n_matches_raw_to_postprocessed_gemma}/{len(results)}")
    print(f"Gemma-on-raw matches tower's PRE-processing prediction: "
          f"{n_matches_raw_to_tower_pre}/{len(results)}")
    print(f"Gemma-on-raw matches tower's POST-processing prediction: "
          f"{n_matches_raw_to_tower_post}/{len(results)}")


if __name__ == "__main__":
    main()
