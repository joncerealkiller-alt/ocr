"""
[Runs after new Stage 5 (Document Routing) - see docs/
PIPELINE_STAGE_TERMINOLOGY.md; "Stage 2" below is the old name]
Runs the post-bucket-classification semantic stage pipeline (Jon's
Phase 1/3 design, 2026-07-27): loads Gemma ONCE, then runs each stage
in STAGES below against it in order, before unloading.

Stage 2 (core/classifier.py, bucket routing) is a SEPARATE, existing
script - untouched, run first, on its own:
    python -m core.classifier data/manifest.csv

THIS script picks up from its output (data/buckets/dense_tabular_rows.csv)
and never writes back to it - see core/semantic_stages.py's own
docstring for why that's true structurally, not just by convention.

Currently one stage: document sub-type classification (census year /
printed vs. handwritten manifest / unknown), reading
data/buckets/dense_tabular_rows.csv and writing
data/buckets/dense_tabular_rows_subtype.csv. That output CSV is meant
to become the input manifest for the CV geometry pass (core/
auto_sidecar.py) - each row's document_type mapping directly to which
template to load - though that wiring isn't built yet (a separate,
explicit next step, not implied by this script existing).

Adding a future stage: append one more SemanticStage(...) to STAGES
below. No other code changes needed - core/semantic_stages.py's
run_stage() is generic across any prompt/input/output combination.

Usage:
    python scripts/run_semantic_stages.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.semantic_stages import SemanticStage, load_gemma_loader, run_stage

BUCKET_DIR = PROJECT_ROOT / "data" / "buckets"
PROMPTS_DIR = PROJECT_ROOT / "config" / "prompts"

STAGES = [
    SemanticStage(
        name="document_subtype",
        input_csv=BUCKET_DIR / "dense_tabular_rows.csv",
        output_csv=BUCKET_DIR / "dense_tabular_rows_subtype.csv",
        # v3 (2026-07-30), not v2: v2 required the word "Canada" to be
        # visible (fixing the UK-page misclassification, see below), but
        # on real 1911/1921/1931 pages the year and "Canada" are printed
        # small in the TOP CORNERS, bilingual - never in the big centred
        # heading ("SCHEDULE No. N" / "DOMINION BUREAU OF STATISTICS").
        # v2 was reading the centre heading, finding no "Canada" there,
        # and correctly-but-uselessly abstaining: 3/16 real 1911 pages in
        # one batch came back "unknown" this way. v3 adds explicit corner-
        # reading guidance plus an ordinal->year cross-check (FIFTH=1911,
        # SIXTH=1921, SEVENTH=1931).
        #
        # Verified via diagnostics/compare_document_subtype_prompts.py
        # (19 images, ground truth read by eye from each page's printed
        # corner title, reproduced identically across two independent
        # runs): v2 12/19, v3 17/19. All 3 formerly-"unknown" 1911 pages
        # now correct; all 5 real 1921/1931 pages correct under both
        # (Gemma was already fine there - those pages just hadn't existed
        # yet). v3's one regression (IMCANQC1865_T4821-00622,
        # handwritten_manifest -> unknown) self-reports confidence 0.1,
        # i.e. fails loudly into the manual-classification queue rather
        # than silently. The UK guard page (WARRG13_2948_2949-0078,
        # v2 was built to fix this) still correctly reads "unknown" under
        # v3 - the corner guidance did not become a license to assume
        # "Canada" is present.
        #
        # A v4 attempt to also fix the IMCANQC regression (scoping the
        # corner rule to census-shaped pages, adding a manifest off-ramp)
        # was tried and REJECTED - it broke 5/5 real manifests (all
        # "unknown", title_text_read="not legible") without fixing the
        # case it targeted. Cause: v4's negative instruction quoted the
        # exact strings "DOMINION BUREAU OF STATISTICS" and "CENSUS OF
        # CANADA" as what NOT to write, and the model then wrote one of
        # them onto a printed manifest - see [[feedback_prompt_examples_
        # get_locked_onto]]. classifier_document_subtype_v4.txt is kept
        # for the record but must not be used. Full numbers in
        # docs/PREPROCESSING_STAGE_NOTES.md.
        #
        # v3's confidence field also actually varies (0.1-0.98, vs v2's
        # pinned 0.95/0.98 across every row including its own "unknown"s)
        # - needed for any future "route to manual review if confidence
        # < X" gate; v2's confidence could not support one.
        prompt_path=PROMPTS_DIR / "classifier_document_subtype_v3.txt",
        output_fields=["document_type", "confidence", "title_text_read", "reason"],
    ),
    # Future stages append here, e.g.:
    # SemanticStage(
    #     name="region_anchors",
    #     input_csv=BUCKET_DIR / "dense_tabular_rows_subtype.csv",
    #     output_csv=BUCKET_DIR / "dense_tabular_rows_regions.csv",
    #     prompt_path=PROMPTS_DIR / "classifier_region_anchors_v1.txt",
    #     output_fields=["metadata_bbox", "header_bbox", "table_bbox", "confidence", "reason"],
    # ),
]


def main():
    print("Loading gemma (shared across every stage below - one load, no reload between stages)...")
    loader = load_gemma_loader()

    for stage in STAGES:
        run_stage(loader, stage)

    print("\nAll stages complete.")


if __name__ == "__main__":
    main()
