"""
Review-mode abstraction for debug_tools/review_uncertain.py (and any
future taxonomy-driven review tool - see docs/TAXONOMY.md). Extracted
2026-08-04 from that tool's own inline `if self.source == "uncertain":
... else: ...` branches, which had grown throughout DUAL-SOURCE MODE
(2026-07-31) into logic mixed directly into the Tk app class. Splitting
it out here does two things at once: (1) makes the UI class itself
mode-agnostic - it calls ReviewMode methods, never branches on a source
string - and (2) makes adding a THIRD mode (Subtype Annotation, for the
planned second Gemma pass) a matter of writing one new class here, not
touching the UI's button-generation or event-handling code at all.

Three concrete modes, one shared ReviewMode interface:

  ProductionReviewMode      - reads data/buckets/uncertain_review.csv
                               (a live queue). Assigning WRITES the row
                               into a real bucket CSV, REMOVES it from
                               the queue, logs to reviewed_uncertain.csv,
                               and syncs core/pipeline_db.py (a real
                               production correction).
  ResearchGroundTruthMode   - reads data/misclassifications.csv (a flat
                               ground-truth sample, not a queue).
                               Assigning fills in that row's own
                               correct_category column IN PLACE - no
                               file move, no DB write, freely
                               re-correctable.
  SubtypeAnnotationMode     - reads a subtype-review queue CSV (file_path,
                               bucket, subtype) - bucket is an ALREADY-
                               assigned primary category; get_options()
                               returns that category's SUBTYPES (core/
                               taxonomy.py's subtypes_for()), not the
                               top-level categories the other two modes
                               use. Not yet fed by any production
                               generator (the second Gemma pass this
                               exists for isn't built) - included fully
                               functional, not a stub, as the concrete
                               proof this architecture doesn't need a
                               redesign when that pass exists: it needs
                               a queue file and nothing else.

Every mode gets its buttons from core/taxonomy.py's load_taxonomy() -
none of the three hardcodes a bucket/subtype list.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from core.pipeline_db import PipelineDatabase, DEFAULT_DB_PATH, hash_file
from core.taxonomy import Taxonomy, Category, Subtype, load_taxonomy
from core.workspace_context import WorkspaceContext

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "outputs"

# Legacy fallback - see core/manifest_pipeline.py's identical comment.
# uncertain_review.csv is run-owned (a routing bucket CSV); the other
# files below (REVIEWED_LOG, MISCLASSIFICATIONS_CSV, etc.) are
# deliberately NOT migrated - see ProductionReviewMode's own docstring.
_legacy_bucket_dir = WorkspaceContext.resolve().runs_root / "legacy_pre_run_system" / "outputs" / "buckets"
BUCKET_DIR = _legacy_bucket_dir if _legacy_bucket_dir.exists() else DATA_DIR / "buckets"

UNCERTAIN_CSV = BUCKET_DIR / "uncertain_review.csv"
REVIEWED_LOG = OUTPUT_DIR / "reviewed_uncertain.csv"
MISCLASSIFICATIONS_CSV = DATA_DIR / "misclassifications.csv"
SUBTYPE_QUEUE_CSV = DATA_DIR / "subtype_review_queue.csv"

# Kept in sync with ui/classifier_validation_ui.py's MISCLASSIFICATION_LOG/
# MISCLASSIFICATION_FIELDS (the "Flag Misclassified" button writes this
# file) - see debug_tools/review_uncertain.py's original module docstring
# for why these constants are re-declared here rather than imported from
# that module (it builds a full Tk app at import time).
MISCLASSIFICATION_FIELDS = [
    "bucket", "file_path", "category", "confidence", "reason",
    "model", "prompt_version", "correct_category",
]
SUBTYPE_QUEUE_FIELDS = ["file_path", "bucket", "subtype"]

BUCKET_CSV_FIELDS = [
    "file_path", "category", "confidence", "text_density",
    "handwriting", "table_layout", "faces", "map_like",
    "reason", "model", "prompt_version",
]
REVIEWED_LOG_FIELDS = [
    "file_path", "original_bucket", "assigned_bucket",
    "reviewer", "timestamp", "note",
]

# Same running "doesn't fit any existing bucket" triage log ui/
# classifier_validation_ui.py's "Needs New Bucket" button already writes
# (added 2026-07-31) - one shared running list for the later
# taxonomy-design pass, not a second file.
NEEDS_NEW_CATEGORY_LOG = DATA_DIR / "flagged_needs_new_bucket.csv"
# Same shared log ui/classifier_validation_ui.py's "Mark Bad Deskew"
# button writes (added 2026-07-31, Jon: "some images are getting
# deskewed when they didn't need it, making them MORE skewed than the
# source - a preprocessing bug, not a classification one"). A different
# problem from "doesn't fit any bucket" - these images DO fit a bucket,
# their PIXELS are just wrong (Stage 3 made them worse), so they're kept
# in a separate list rather than conflated with taxonomy gaps.
BAD_DESKEW_LOG = DATA_DIR / "flagged_bad_deskew.csv"
# Distinct file for subtype-level taxonomy gaps ("this image's primary
# bucket is right, but none of ITS subtypes fit") - a different kind of
# feedback from "no top-level bucket fits at all", so kept as its own
# list rather than conflated into the category-level one above.
NEEDS_NEW_SUBTYPE_LOG = DATA_DIR / "flagged_needs_new_subtype.csv"
FLAG_LOG_FIELDS = ["bucket", "file_path", "category", "reason"]
SUBTYPE_FLAG_LOG_FIELDS = ["bucket", "file_path", "reason"]


def _append_csv_row(path: Path, fieldnames: list[str], row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


@dataclass
class Option:
    """Uniform shape the UI renders a button from, regardless of whether
    it came from Taxonomy.assignable_categories() (Category) or
    Taxonomy.subtypes_for() (Subtype) - the whole point of this
    conversion is that debug_tools/review_uncertain.py's button-building
    code never needs to know which one it's looking at."""
    id: str
    display_name: str
    color: str | None = None


def _category_option(c: Category) -> Option:
    return Option(id=c.id, display_name=c.display_name, color=c.color)


def _subtype_option(s: Subtype) -> Option:
    return Option(id=s.id, display_name=s.display_name, color=s.color)


class ReviewApp(Protocol):
    """The subset of debug_tools/review_uncertain.py's ReviewApp that
    mode methods below are allowed to touch - documented here so a mode
    implementation's contract with the UI class is explicit, not just
    convention. Modes read/mutate app.rows/app.index directly (UI state
    the app class owns) and call back into app.load_current()/app._advance()
    for the shared "move to the next row" UI refresh."""
    rows: list[dict]
    error_rows: list[dict]
    index: int
    reviewer_name: object  # tkinter.StringVar

    def load_current(self) -> None: ...
    def _advance(self) -> None: ...


class ReviewMode:
    """Base interface. Every method here is called BY debug_tools/
    review_uncertain.py's ReviewApp - that class never branches on which
    mode it has, it just calls these."""

    #: Shown in the window title and status line.
    label: str = "Review"
    #: Whether assign() should show a destructive-action confirm dialog
    #: (true for production - a real file move; false for research/
    #: subtype - a lightweight, freely re-correctable in-place label).
    requires_confirmation: bool = False
    #: Label above the option-button column - mode-specific wording
    #: ("bucket" vs "subtype") without the UI needing to know why.
    options_label: str = "Disagree? Pick the correct option:"

    def __init__(self, taxonomy: Taxonomy):
        self.taxonomy = taxonomy

    def load(self) -> tuple[list[dict], list[dict]]:
        """Returns (reviewable_rows, error_rows)."""
        raise NotImplementedError

    def status_text(self, rows: list[dict], index: int) -> str:
        raise NotImplementedError

    def get_options(self, row: dict) -> list[Option]:
        """Which buttons to offer for this row - categories for
        production/research, subtypes-of-this-row's-bucket for subtype
        annotation. This is the method that makes the whole UI
        taxonomy-driven: it is the ONLY place button content comes from."""
        raise NotImplementedError

    def predicted_id(self, row: dict) -> str | None:
        """The id to preselect/highlight as 'predicted', for the Accept
        button - None if there's nothing valid to accept (Accept button
        is disabled in that case)."""
        raise NotImplementedError

    def accept_button_text(self, predicted_display_name: str) -> str:
        return f"✓ Accept: {predicted_display_name}"

    def assign(self, app: ReviewApp, row: dict, choice_id: str) -> None:
        raise NotImplementedError

    def mark_ignore(self, app: ReviewApp, row: dict) -> None:
        raise NotImplementedError

    def needs_new_button_text(self) -> str:
        return "Doesn't fit any option (needs a new one)"

    def mark_needs_new(self, app: ReviewApp, row: dict) -> None:
        raise NotImplementedError

    def mark_bad_deskew(self, app: ReviewApp, row: dict) -> None:
        """A different problem from mark_needs_new(): this image fits a
        taxonomy option fine, but Stage 3's deskew made its PIXELS worse
        (more skewed than the original), so it isn't a real usable
        example right now regardless of which bucket it'd go in - a
        preprocessing bug, not a classification/taxonomy one. Not every
        mode needs to override this (only production/research do
        anything useful with it today); the base raises so a caller
        finds out immediately if it's wired to a mode that hasn't
        implemented it, rather than silently doing nothing."""
        raise NotImplementedError

    def save(self, app: ReviewApp) -> None:
        raise NotImplementedError


class ProductionReviewMode(ReviewMode):
    """--source uncertain (default, unchanged behavior from before this
    refactor): data/buckets/uncertain_review.csv is a live queue - every
    terminal action (assign/ignore/needs-new) removes the row from it,
    writes the real consequence (bucket CSV row, reviewed_uncertain.csv
    entry, pipeline_db.py correction), and error rows round-trip through
    untouched (see load()'s docstring for the real bug this preserves
    the fix for)."""

    label = "Uncertain Review"
    requires_confirmation = True
    options_label = "Disagree? Pick the correct bucket:"

    def __init__(self, taxonomy: Taxonomy, db_path: Path = DEFAULT_DB_PATH, ctx=None):
        """
        ctx (hashed-run migration, typed loosely to avoid an import
        cycle - see docs/RUN_ARCHITECTURE.md): when given a RunContext,
        overrides db_path/bucket_dir with ctx.workspace.pipeline_db_path/
        ctx.buckets. ONLY uncertain_review.csv (a routing bucket CSV,
        same directory/lifecycle as every other DocumentCategory bucket
        CSV) is run-owned this way - REVIEWED_LOG, MISCLASSIFICATIONS_CSV,
        SUBTYPE_QUEUE_CSV, and the three flag logs below are deliberately
        NOT touched by ctx: per this module's own docstring they're
        either ground-truth accumulation (MISCLASSIFICATIONS_CSV, "a
        flat ground-truth sample, not a queue") or running/cumulative
        logs meant to persist across many runs, not per-run routing
        state. Classified per-file by actual ownership, not by "this
        file happens to be read by a UI tool" - see docs/RUN_ARCHITECTURE.md.
        """
        super().__init__(taxonomy)
        if ctx is not None:
            db_path = ctx.workspace.pipeline_db_path
        self.db = PipelineDatabase(db_path)
        self.ctx = ctx
        self.bucket_dir = ctx.buckets if ctx is not None else BUCKET_DIR
        self.uncertain_csv = self.bucket_dir / "uncertain_review.csv"

    def load(self) -> tuple[list[dict], list[dict]]:
        """
        REAL BUG FIXED HERE ORIGINALLY (2026-07-25, found via a live
        run): this used to return ONLY the reviewable rows, discarding
        error_rows (hard pipeline failures, e.g. a classifier parse
        error) outright at load time. save() rewrites uncertain_review.csv
        from the in-memory row list on every close - since error rows
        were never loaded into that list, simply opening this tool,
        doing anything at all, and closing it silently and permanently
        deleted every hard-error row from the CSV, with no logging and
        no recovery path. error_rows must be carried through untouched
        and written back by save(), not just excluded from the
        reviewable list - preserved exactly through this refactor.
        """
        if not self.uncertain_csv.exists():
            return [], []
        with open(self.uncertain_csv, "r", encoding="utf-8") as f:
            all_rows = [row for row in csv.DictReader(f) if row.get("file_path")]
        reviewable = [row for row in all_rows if not row.get("error")]
        error_rows = [row for row in all_rows if row.get("error")]
        return reviewable, error_rows

    def status_text(self, rows: list[dict], index: int) -> str:
        return f"{len(rows) - index} remaining in uncertain_review"

    def get_options(self, row: dict) -> list[Option]:
        return [_category_option(c) for c in self.taxonomy.assignable_categories()]

    def predicted_id(self, row: dict) -> str | None:
        predicted = (row.get("category") or "").strip()
        valid_ids = {o.id for o in self.get_options(row)}
        return predicted if predicted in valid_ids else None

    def _write_bucket_row(self, bucket: str, row: dict) -> None:
        path = self.bucket_dir / f"{bucket}.csv"
        is_new = not path.exists() or path.stat().st_size == 0
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=BUCKET_CSV_FIELDS)
            if is_new:
                writer.writeheader()
            out_row = {k: row.get(k, "") for k in BUCKET_CSV_FIELDS}
            out_row["category"] = bucket
            writer.writerow(out_row)

    def _log_review(self, app: ReviewApp, row: dict, assigned_bucket: str, note: str = "") -> None:
        _append_csv_row(REVIEWED_LOG, REVIEWED_LOG_FIELDS, {
            "file_path": row["file_path"],
            "original_bucket": "uncertain_review",
            "assigned_bucket": assigned_bucket,
            "reviewer": app.reviewer_name.get(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "note": note,
        })

    def _sync_db_correction(self, app: ReviewApp, row: dict, assigned_bucket: str) -> None:
        """
        assigned_bucket in ("ignored", "needs_new_bucket"): these are
        sentinel outcome strings, not real DocumentCategory/bucket CSV
        values - bucket is cleared to NULL rather than stored as a fake
        bucket, and status carries the distinct terminal outcome
        ("ignored" / "needs_new_bucket", both genuinely new terminal
        status strings - images.status is free-form TEXT, not a fixed
        enum). Auto-registers the image if the DB doesn't already know
        it, same reasoning as core/pipeline_db.py's
        sync_bucket_classifications(): a review session must never
        silently fail to record a real correction.
        """
        file_path = row["file_path"]
        lookup = self.ctx.to_relative(file_path) if self.ctx is not None else file_path
        run_id = self.ctx.run_id if self.ctx is not None else None
        image = self.db.get_image_by_path(lookup, run_id=run_id)
        if image is None:
            image_id = self.db.get_or_create_image(
                source_path=file_path, working_path=lookup, run_id=run_id,
                identity_hash=hash_file(file_path) if self.ctx is not None else None,
            )
            image = self.db.get_image(image_id)

        original_bucket = row.get("category") or "uncertain_review"
        if assigned_bucket in ("ignored", "needs_new_bucket"):
            self.db.update_image_state(image["id"], bucket=None, current_stage=5, status=assigned_bucket)
        else:
            self.db.update_image_state(image["id"], bucket=assigned_bucket, current_stage=5, status="classified")
        self.db.record_stage_output(
            image["id"], stage="stage5_human_review",
            sidecar_path=str(REVIEWED_LOG), lookup_key=file_path, status="done",
            note=f"reviewer={app.reviewer_name.get()!r} "
                 f"original_bucket={original_bucket!r} assigned_bucket={assigned_bucket!r}",
        )

    def assign(self, app: ReviewApp, row: dict, choice_id: str) -> None:
        self._write_bucket_row(choice_id, row)
        self._log_review(app, row, assigned_bucket=choice_id)
        self._sync_db_correction(app, row, assigned_bucket=choice_id)
        app._advance()

    def mark_ignore(self, app: ReviewApp, row: dict) -> None:
        self._log_review(app, row, assigned_bucket="ignored", note="Marked not useful during review")
        self._sync_db_correction(app, row, assigned_bucket="ignored")
        app._advance()

    def needs_new_button_text(self) -> str:
        return "Doesn't fit any bucket (needs new one)"

    def mark_needs_new(self, app: ReviewApp, row: dict) -> None:
        _append_csv_row(NEEDS_NEW_CATEGORY_LOG, FLAG_LOG_FIELDS, {
            "bucket": row.get("category") or "uncertain_review",
            "file_path": row["file_path"],
            "category": row.get("category", ""),
            "reason": row.get("reason", ""),
        })
        self._log_review(app, row, assigned_bucket="needs_new_bucket",
                          note="Doesn't fit any existing bucket - logged for taxonomy review")
        self._sync_db_correction(app, row, assigned_bucket="needs_new_bucket")
        app._advance()

    def mark_bad_deskew(self, app: ReviewApp, row: dict) -> None:
        _append_csv_row(BAD_DESKEW_LOG, FLAG_LOG_FIELDS, {
            "bucket": row.get("category") or "uncertain_review",
            "file_path": row["file_path"],
            "category": row.get("category", ""),
            "reason": row.get("reason", ""),
        })
        self._log_review(app, row, assigned_bucket="bad_deskew",
                          note="Deskew made this image worse - needs Stage 3 re-processing")
        self._sync_db_correction(app, row, assigned_bucket="bad_deskew")
        app._advance()

    def save(self, app: ReviewApp) -> None:
        """Rewrites uncertain_review.csv with whatever's left (assigned/
        ignored/needs-new/bad-deskew rows already removed by _advance(),
        skipped rows retained) PLUS every error_rows entry, unchanged."""
        with open(UNCERTAIN_CSV, "w", newline="", encoding="utf-8") as f:
            fieldnames = BUCKET_CSV_FIELDS + ["error"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in app.error_rows + app.rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})


class ResearchGroundTruthMode(ReviewMode):
    """--source misclassifications (unchanged behavior from before this
    refactor): data/misclassifications.csv is a flat ground-truth
    sample, not a queue - rows are never removed or moved to a bucket
    CSV. Assigning fills in correct_category IN PLACE; the row stays in
    the file either way. No reviewed_uncertain.csv entry, no DB write -
    this labels a research sample, not a real pipeline classification."""

    label = "Uncertain Review — Misclassification Labeling"
    requires_confirmation = False
    options_label = "Disagree? Pick the correct bucket:"

    def load(self) -> tuple[list[dict], list[dict]]:
        if not MISCLASSIFICATIONS_CSV.exists():
            return [], []
        with open(MISCLASSIFICATIONS_CSV, "r", encoding="utf-8") as f:
            return [row for row in csv.DictReader(f) if row.get("file_path")], []

    def status_text(self, rows: list[dict], index: int) -> str:
        already = sum(1 for r in rows if (r.get("correct_category") or "").strip())
        return f"{len(rows) - index} remaining to label   |   {already}/{len(rows)} labeled so far"

    def get_options(self, row: dict) -> list[Option]:
        return [_category_option(c) for c in self.taxonomy.assignable_categories()]

    def predicted_id(self, row: dict) -> str | None:
        predicted = (row.get("category") or "").strip()
        valid_ids = {o.id for o in self.get_options(row)}
        return predicted if predicted in valid_ids else None

    def accept_button_text(self, predicted_display_name: str) -> str:
        # Rows here were flagged BECAUSE they looked wrong - phrased to
        # not imply "accept" is the expected outcome the way it is for a
        # genuinely uncertain (not yet judged) row in production mode.
        return f"Gemma's original call was actually correct: {predicted_display_name}"

    def assign(self, app: ReviewApp, row: dict, choice_id: str) -> None:
        row["correct_category"] = choice_id
        app.index += 1
        app.load_current()

    def mark_ignore(self, app: ReviewApp, row: dict) -> None:
        row["correct_category"] = "ignored"
        app.index += 1
        app.load_current()

    def needs_new_button_text(self) -> str:
        return "Doesn't fit any bucket (needs new one)"

    def mark_needs_new(self, app: ReviewApp, row: dict) -> None:
        _append_csv_row(NEEDS_NEW_CATEGORY_LOG, FLAG_LOG_FIELDS, {
            "bucket": row.get("category") or "",
            "file_path": row["file_path"],
            "category": row.get("category", ""),
            "reason": row.get("reason", ""),
        })
        row["correct_category"] = "needs_new_bucket"
        app.index += 1
        app.load_current()

    def mark_bad_deskew(self, app: ReviewApp, row: dict) -> None:
        _append_csv_row(BAD_DESKEW_LOG, FLAG_LOG_FIELDS, {
            "bucket": row.get("category") or "",
            "file_path": row["file_path"],
            "category": row.get("category", ""),
            "reason": row.get("reason", ""),
        })
        row["correct_category"] = "bad_deskew"
        app.index += 1
        app.load_current()

    def save(self, app: ReviewApp) -> None:
        """Rewrites misclassifications.csv with every row (nothing is
        ever removed in this mode), carrying forward each row's own
        correct_category value - labeled or still blank, so a
        partially-worked session is never lost."""
        with open(MISCLASSIFICATIONS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=MISCLASSIFICATION_FIELDS)
            writer.writeheader()
            for row in app.rows:
                writer.writerow({k: row.get(k, "") for k in MISCLASSIFICATION_FIELDS})


class SubtypeAnnotationMode(ReviewMode):
    """
    Mode 3 (planned second Gemma pass, 2026-08-04) - fully functional,
    not a stub, as the concrete proof that adding this mode required no
    changes to debug_tools/review_uncertain.py's UI/event-handling code,
    only this one class. Reads data/subtype_review_queue.csv (file_path,
    bucket, subtype) - "bucket" is an ALREADY-assigned primary category
    (from Mode 1/2 or production classification); this mode's buttons
    are core/taxonomy.py's subtypes_for(row["bucket"]), NOT the top-level
    categories the other two modes use - the one place get_options()
    genuinely differs in WHAT it draws from, still via the same Taxonomy
    object.

    Not yet wired into any real production generator - no code writes
    data/subtype_review_queue.csv yet, since the second Gemma pass this
    supports doesn't exist. A row for a bucket with no defined subtypes
    (supports_subtypes=false, or none listed) legitimately gets zero
    options - the UI's existing "no valid prediction" defensive handling
    already covers an empty options list without needing new logic.
    """

    label = "Subtype Annotation"
    requires_confirmation = False
    options_label = "Pick the correct subtype:"

    def load(self) -> tuple[list[dict], list[dict]]:
        if not SUBTYPE_QUEUE_CSV.exists():
            return [], []
        with open(SUBTYPE_QUEUE_CSV, "r", encoding="utf-8") as f:
            return [row for row in csv.DictReader(f) if row.get("file_path")], []

    def status_text(self, rows: list[dict], index: int) -> str:
        already = sum(1 for r in rows if (r.get("subtype") or "").strip())
        return f"{len(rows) - index} remaining to subtype-label   |   {already}/{len(rows)} labeled so far"

    def get_options(self, row: dict) -> list[Option]:
        bucket = row.get("bucket", "")
        return [_subtype_option(s) for s in self.taxonomy.subtypes_for(bucket)]

    def predicted_id(self, row: dict) -> str | None:
        predicted = (row.get("subtype") or "").strip()
        valid_ids = {o.id for o in self.get_options(row)}
        return predicted if predicted in valid_ids else None

    def accept_button_text(self, predicted_display_name: str) -> str:
        return f"✓ Accept: {predicted_display_name}"

    def assign(self, app: ReviewApp, row: dict, choice_id: str) -> None:
        row["subtype"] = choice_id
        app.index += 1
        app.load_current()

    def mark_ignore(self, app: ReviewApp, row: dict) -> None:
        row["subtype"] = "ignored"
        app.index += 1
        app.load_current()

    def needs_new_button_text(self) -> str:
        return "Doesn't fit any subtype (needs new one)"

    def mark_needs_new(self, app: ReviewApp, row: dict) -> None:
        _append_csv_row(NEEDS_NEW_SUBTYPE_LOG, SUBTYPE_FLAG_LOG_FIELDS, {
            "bucket": row.get("bucket", ""),
            "file_path": row["file_path"],
            "reason": "",
        })
        row["subtype"] = "needs_new_subtype"
        app.index += 1
        app.load_current()

    def mark_bad_deskew(self, app: ReviewApp, row: dict) -> None:
        _append_csv_row(BAD_DESKEW_LOG, FLAG_LOG_FIELDS, {
            "bucket": row.get("bucket", ""),
            "file_path": row["file_path"],
            "category": row.get("bucket", ""),
            "reason": "",
        })
        row["subtype"] = "bad_deskew"
        app.index += 1
        app.load_current()

    def save(self, app: ReviewApp) -> None:
        with open(SUBTYPE_QUEUE_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SUBTYPE_QUEUE_FIELDS)
            writer.writeheader()
            for row in app.rows:
                writer.writerow({k: row.get(k, "") for k in SUBTYPE_QUEUE_FIELDS})


MODE_REGISTRY = {
    "uncertain": ProductionReviewMode,
    "misclassifications": ResearchGroundTruthMode,
    "subtype": SubtypeAnnotationMode,
}


def build_mode(source: str, db_path: Path = DEFAULT_DB_PATH, taxonomy: Taxonomy | None = None, ctx=None) -> ReviewMode:
    """ctx: passed through to ProductionReviewMode only - the other two
    modes' files are deliberately not run-owned, see ProductionReviewMode
    .__init__()'s docstring."""
    taxonomy = taxonomy or load_taxonomy()
    mode_cls = MODE_REGISTRY[source]
    if mode_cls is ProductionReviewMode:
        return mode_cls(taxonomy, db_path=db_path, ctx=ctx)
    return mode_cls(taxonomy)
