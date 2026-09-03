"""
Phase 2 of the Stage-4 error-analysis pipeline (2026-08-08): failure
taxonomy infrastructure - definitions, serialization, validation, and
reporting support ONLY. Does NOT classify any image - per the handoff
prompt's explicit instruction, manual review is still required before any
image gets a real primary_cause assigned. This module produces the
SCHEMA and a template annotation file with every review-worthy failure
pre-populated as "pending_review" (a distinct state from the taxonomy's
own "unknown" cause - pending_review means nobody has looked yet; unknown
means a reviewer looked and genuinely couldn't tell).

Taxonomy source: docs/GEMMA_HIDDEN_STATE_ERROR_ANALYSIS_STAGE4_RESEARCH.md
section 1 - reproduced here as data, not re-derived.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
OUT_DIR = PROJECT_ROOT / "data" / "outputs" / "error_analysis"
ANNOTATIONS_PATH = OUT_DIR / "failure_annotations.json"
TAXONOMY_SCHEMA_PATH = OUT_DIR / "failure_taxonomy_schema.json"

# Primary causes - see docs/GEMMA_HIDDEN_STATE_ERROR_ANALYSIS_STAGE4_
# RESEARCH.md section 1 for the full rationale behind each. Deliberately
# NOT exhaustive - designed to be extended (see EXTEND_TAXONOMY note
# below) rather than treated as final.
PRIMARY_CAUSES = {
    "image_quality": "Poor scan/photo quality - blur, low resolution, heavy noise.",
    "skew_crop": "Severe skew, incorrect crop, or page edge cut off.",
    "low_contrast": "Faded/low-contrast source; content present but hard to distinguish from background.",
    "multi_content": "Multiple document types present in one frame (genuinely ambiguous even to a careful human).",
    "ambiguous_taxonomy": "Image plausibly fits 2+ categories under the taxonomy's own written definitions.",
    "label_error": "Ground truth itself looks wrong on inspection, independent of the model's prediction.",
    "rare_class": "True class has very low train-split support - failure may reflect data scarcity, not a real limitation.",
    "missing_visual_info": "A human given only the image (no prior taxonomy knowledge) could not confidently classify it either.",
    "requires_reasoning": "Visual signal present, but classification needs interpretation beyond raw appearance.",
    "preprocessing_artifact": "Pipeline preprocessing (deskew/crop/resize) introduced or amplified a problem not present in the original source.",
    "unknown": "Reviewed, but none of the above causes clearly apply and the reviewer genuinely can't tell.",
}

# Distinct from every PRIMARY_CAUSES entry - this is the "nobody has
# looked yet" state, never a valid final answer, only a placeholder.
PENDING_REVIEW = "pending_review"

# Secondary causes reuse the same vocabulary as primary causes (any
# primary cause can also be tagged as a secondary cause on a DIFFERENT
# failure - e.g. a failure whose primary cause is `ambiguous_taxonomy`
# might also carry `low_contrast` as a secondary tag).
SECONDARY_CAUSE_CHOICES = set(PRIMARY_CAUSES.keys())


@dataclass
class FailureAnnotation:
    review_id: str
    path: str
    gt: str
    location: str  # which hook location's prediction this annotation is scoped to (a failure can differ per location)
    pred: str
    primary_cause: str = PENDING_REVIEW
    secondary_causes: list[str] = field(default_factory=list)
    reviewer_note: str = ""
    reviewed: bool = False

    def validate(self) -> list[str]:
        """Returns a list of validation errors (empty list = valid). Does
        NOT raise - callers decide whether an invalid annotation blocks
        anything, since a batch-load of partially-reviewed annotations is
        an expected, normal state, not an error state."""
        errors = []
        if self.primary_cause != PENDING_REVIEW and self.primary_cause not in PRIMARY_CAUSES:
            errors.append(f"primary_cause {self.primary_cause!r} is not a recognized taxonomy entry "
                          f"(known: {sorted(PRIMARY_CAUSES)} or {PENDING_REVIEW!r}).")
        for c in self.secondary_causes:
            if c not in SECONDARY_CAUSE_CHOICES:
                errors.append(f"secondary cause {c!r} is not a recognized taxonomy entry.")
        if self.primary_cause in self.secondary_causes:
            errors.append(f"primary_cause {self.primary_cause!r} duplicated in secondary_causes - "
                          "a cause should be recorded once, as primary, not both.")
        if self.reviewed and self.primary_cause == PENDING_REVIEW:
            errors.append("reviewed=True but primary_cause is still pending_review - inconsistent state.")
        if not self.reviewed and self.primary_cause != PENDING_REVIEW:
            errors.append("primary_cause was set but reviewed=False - mark reviewed=True once a cause is assigned.")
        return errors


def save_annotations(annotations: list[FailureAnnotation], path: Path = ANNOTATIONS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(a) for a in annotations], indent=2), encoding="utf-8")


def load_annotations(path: Path = ANNOTATIONS_PATH) -> list[FailureAnnotation]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [FailureAnnotation(**r) for r in raw]


def validate_annotations(annotations: list[FailureAnnotation]) -> dict:
    """Batch validation report - counts, not just a pass/fail, so Phase 6
    can report review PROGRESS (how many pending vs. reviewed) alongside
    correctness."""
    all_errors = {}
    n_pending, n_reviewed = 0, 0
    for a in annotations:
        errs = a.validate()
        if errs:
            all_errors[a.review_id] = errs
        if a.primary_cause == PENDING_REVIEW:
            n_pending += 1
        else:
            n_reviewed += 1
    return {
        "total": len(annotations),
        "reviewed": n_reviewed,
        "pending_review": n_pending,
        "validation_errors": all_errors,
        "valid": len(all_errors) == 0,
    }


def cause_distribution_report(annotations: list[FailureAnnotation]) -> dict:
    """Primary-cause counts + secondary-cause counts, reviewed items only
    (pending_review items are excluded from the distribution - including
    them would silently understate how incomplete the review is)."""
    from collections import Counter
    reviewed = [a for a in annotations if a.primary_cause != PENDING_REVIEW]
    primary_counts = Counter(a.primary_cause for a in reviewed)
    secondary_counts = Counter(c for a in reviewed for c in a.secondary_causes)
    total_reviewed = len(reviewed)
    return {
        "n_reviewed": total_reviewed,
        "n_total": len(annotations),
        "primary_cause_counts": dict(primary_counts),
        "primary_cause_percentages": {
            k: round(100 * v / total_reviewed, 1) if total_reviewed else None for k, v in primary_counts.items()
        },
        "secondary_cause_counts": dict(secondary_counts),
    }


def build_template_annotations(failure_rows: list[dict], location: str) -> list[FailureAnnotation]:
    """One FailureAnnotation per (review_id, location) failure, all
    starting at pending_review - the infrastructure this phase is
    responsible for building, not the review itself."""
    return [
        FailureAnnotation(
            review_id=row["review_id"], path=row["path"], gt=row["gt"],
            location=location, pred=row[f"pred_{location}"],
        )
        for row in failure_rows
    ]


def write_schema_doc(path: Path = TAXONOMY_SCHEMA_PATH) -> None:
    """Machine-readable taxonomy definition - lets a review UI (or a
    future automated pass) load valid causes/descriptions without
    hardcoding them a second time."""
    schema = {
        "primary_causes": PRIMARY_CAUSES,
        "pending_review_sentinel": PENDING_REVIEW,
        "secondary_cause_choices": sorted(SECONDARY_CAUSE_CHOICES),
        "annotation_fields": list(FailureAnnotation.__dataclass_fields__.keys()),
        "extension_note": (
            "This taxonomy is deliberately incomplete per the design doc's own instruction. "
            "To add a new primary cause: add an entry to PRIMARY_CAUSES in this file with a one-line "
            "description, re-run write_schema_doc(), and re-validate any existing annotations "
            "(new causes never invalidate old annotations, since PRIMARY_CAUSES only grows)."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema, indent=2), encoding="utf-8")


def main():
    """
    Builds the template annotation file for review: one entry per
    (failure, location) pair, using Phase 1's review_table.csv as the
    source of which rows are failures. Does NOT assign any real cause -
    every entry starts at PENDING_REVIEW, per this phase's explicit scope.
    """
    review_table_path = OUT_DIR / "review_table.csv"
    if not review_table_path.exists():
        raise FileNotFoundError(f"{review_table_path} not found - run phase1_build_review_dataset.py first.")

    with open(review_table_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    from diagnostics.error_analysis.common import LOCATIONS

    all_annotations = []
    for loc in LOCATIONS:
        failure_rows = [r for r in rows if r[f"correct_{loc}"] == "False" and r["split"] == "test"]
        all_annotations.extend(build_template_annotations(failure_rows, loc))
        print(f"  {loc}: {len(failure_rows)} test-split failures templated for review")

    write_schema_doc()
    print(f"\nTaxonomy schema written to {TAXONOMY_SCHEMA_PATH}")

    # Don't overwrite existing progress if annotations already exist and
    # have real review work done - a re-run of this script should never
    # silently discard a reviewer's completed work.
    existing = load_annotations()
    existing_reviewed = {(a.review_id, a.location) for a in existing if a.primary_cause != PENDING_REVIEW}
    if existing_reviewed:
        print(f"\n{len(existing_reviewed)} annotations already have real review progress in "
              f"{ANNOTATIONS_PATH} - merging rather than overwriting.")
        existing_by_key = {(a.review_id, a.location): a for a in existing}
        merged = []
        for a in all_annotations:
            key = (a.review_id, a.location)
            merged.append(existing_by_key.get(key, a))
        all_annotations = merged

    save_annotations(all_annotations)
    print(f"Wrote {len(all_annotations)} template annotations to {ANNOTATIONS_PATH} "
          f"(all pending_review except any merged-in prior progress).")

    report = validate_annotations(all_annotations)
    print(f"\nValidation: {report['reviewed']} reviewed, {report['pending_review']} pending, "
          f"{len(report['validation_errors'])} validation errors.")


if __name__ == "__main__":
    main()
