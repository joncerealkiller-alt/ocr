"""
Stage 3 (human gate) of the pipeline: review uncertain_review.csv one
image at a time, assign each to the correct bucket, and log the
correction for later classifier improvement.

Usage:
    python review_uncertain.py
    python review_uncertain.py --source misclassifications

Design per project discussion:
  - Original uncertain entry is never silently deleted - every
    reassignment is logged to data/outputs/reviewed_uncertain.csv
    with the original bucket, assigned bucket, and a timestamp, so
    the classifier's mistakes are a durable, inspectable record.
  - uncertain_review.csv itself IS mutated (the row is removed once
    assigned) since it's meant to function as a live queue, not an
    archive - the archive is reviewed_uncertain.csv.
  - Pipeline gate: core/extractor.py should not be run until this
    queue is empty (or you've deliberately decided to leave some
    rows unreviewed and accept they won't be extracted this pass).
    This script prints a warning count on exit if the queue isn't
    empty yet.

DUAL-SOURCE MODE (added 2026-07-31, per Jon's explicit direction: "its
core use should stay untouched, just change what it does dependant on
its source file"). The click-a-bucket review interaction is identical
either way - only what happens on submit changes, selected by
--source:

  - --source uncertain (default, UNCHANGED behavior): reads
    data/buckets/uncertain_review.csv, a live QUEUE. Assigning a bucket
    WRITES the row into that bucket's CSV, REMOVES it from the queue,
    and appends a correction record to reviewed_uncertain.csv. Error
    rows (hard pipeline failures) are excluded from review and
    round-tripped untouched, exactly as before.

  - --source misclassifications: reads data/misclassifications.csv (the
    flat triage log ui/classifier_validation_ui.py's "Flag
    Misclassified" button appends to). This file is NOT a queue - it's
    Jon's running ground-truth sample, so rows are never deleted or
    moved to a bucket CSV. Assigning a bucket instead fills in that
    row's own correct_category column, IN PLACE, and the row stays in
    misclassifications.csv either way. No reviewed_uncertain.csv entry
    is written in this mode - the label lives on the row itself.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import platform
from datetime import datetime, timezone
from pathlib import Path

from tkinter import Tk, Frame, Label, Button, StringVar, messagebox
from PIL import Image, ImageTk

# Moved into debug_tools/ (2026-07-25) - one directory deeper than repo root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUCKET_DIR = PROJECT_ROOT / "data" / "buckets"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"

UNCERTAIN_CSV = BUCKET_DIR / "uncertain_review.csv"
REVIEWED_LOG = OUTPUT_DIR / "reviewed_uncertain.csv"

# Kept in sync with ui/classifier_validation_ui.py's MISCLASSIFICATION_LOG/
# MISCLASSIFICATION_FIELDS (the "Flag Misclassified" button writes this
# file) - not imported from there since that module builds a full Tk app
# at import time and this script has no need for it beyond these two
# constants, plus the one extra column this tool adds on top.
MISCLASSIFICATIONS_CSV = PROJECT_ROOT / "data" / "misclassifications.csv"
MISCLASSIFICATION_FIELDS = [
    "bucket", "file_path", "category", "confidence", "reason",
    "model", "prompt_version", "correct_category",
]

ASSIGNABLE_BUCKETS = [
    "dense_tabular_rows",
    "genealogy_chart",
    "handwritten_ledger",
    "map_land_record",
    "printed_document",
    "mixed_text_image",
    "portrait_photo",
]

# Distinct hue per bucket, paired with the text label (never colour-only -
# a colorblind reviewer, or anyone doing a quick pass without full
# attention, still needs the label to read correctly). This is a speed
# aid for the common case, not the sole signal.
BUCKET_COLOURS = {
    "dense_tabular_rows": "#a8e6a1",
    "genealogy_chart": "#f0e08a",
    "handwritten_ledger": "#d1a875",
    "map_land_record": "#f0a0a0",
    "printed_document": "#a0c8f0",
    "mixed_text_image": "#c9a0e0",
    "portrait_photo": "#f0c8a0",
}
ACCEPT_COLOUR = "#4caf50"  # distinct "agree" green, independent of bucket
                           # identity colours above - this button's colour
                           # signals the action (accept), not which bucket

BUCKET_CSV_FIELDS = [
    "file_path", "category", "confidence", "text_density",
    "handwriting", "table_layout", "faces", "map_like",
    "reason", "model", "prompt_version",
]

REVIEWED_LOG_FIELDS = [
    "file_path", "original_bucket", "assigned_bucket",
    "reviewer", "timestamp", "note",
]

MAX_PREVIEW_SIZE = (500, 650)


class ReviewApp:
    def __init__(self, root: Tk, rows: list[dict], error_rows: list[dict] | None = None,
                 source: str = "uncertain"):
        self.root = root
        self.rows = rows
        # Hard pipeline-failure rows (e.g. classifier parse errors) -
        # never shown in this review UI and never mutated by accept/
        # skip/ignore, but must round-trip back into uncertain_review.
        # csv unchanged via save_remaining() - see load_uncertain_rows()'s
        # docstring for the real data-loss bug this fixes. Always empty
        # in "misclassifications" source mode - that file has no
        # equivalent hard-error concept.
        self.error_rows = error_rows or []
        self.index = 0
        self.reviewer_name = StringVar(value="jonny")
        # "uncertain" (default, original behavior) or "misclassifications"
        # - see module docstring's DUAL-SOURCE MODE section for exactly
        # what differs between the two.
        self.source = source

        root.title("Uncertain Review" if source == "uncertain" else "Uncertain Review — Misclassification Labeling")
        root.geometry("1150x820")

        # -- top strip: status/path/prediction/reason, full width --------
        self.status_label = Label(root, text="", font=("Segoe UI", 10))
        self.status_label.pack(pady=(10, 0))

        self.path_label = Label(root, text="", font=("Segoe UI", 9), wraplength=1100, justify="left")
        self.path_label.pack(pady=(2, 4))

        self.open_image_button = Button(
            root, text="Open full image", command=self.open_image_in_viewer
        )
        self.open_image_button.pack(pady=(0, 4))

        self.prediction_label = Label(root, text="", font=("Segoe UI", 10, "bold"), fg="#225")
        self.prediction_label.pack(pady=(4, 0))

        self.reason_label = Label(root, text="", font=("Segoe UI", 9, "italic"),
                                   wraplength=1100, justify="left", fg="#444")
        self.reason_label.pack(pady=(0, 8))

        # -- main split: image preview (left) / all action controls (right) --
        main_frame = Frame(root)
        main_frame.pack(fill="both", expand=True, padx=12, pady=(4, 8))

        self.image_frame = Frame(main_frame)
        self.image_frame.pack(side="left", fill="both", expand=True)
        self.image_label = Label(self.image_frame)
        self.image_label.pack(anchor="n")
        self._tk_image = None  # keep a reference, tkinter needs this

        controls = Frame(main_frame, width=320)
        controls.pack(side="left", fill="y", padx=(16, 0))
        controls.pack_propagate(False)  # keep a stable button-column width
        # regardless of image size, rather than stretching to fit content

        # -- accept Gemma's prediction: own block, own colour, separate
        # from the override list below - common/fast path, styled
        # distinctly from "I disagree, pick a different bucket".
        self.accept_button = Button(
            controls, text="", bg=ACCEPT_COLOUR, fg="white", wraplength=290,
            font=("Segoe UI", 10, "bold"), command=self.accept_prediction,
        )
        self.accept_button.pack(fill="x", pady=(0, 14))

        Label(controls, text="Disagree? Pick the correct bucket:",
              font=("Segoe UI", 9)).pack(anchor="w", pady=(0, 6))

        # Single column of bucket assignment buttons, colour-keyed per
        # BUCKET_COLOURS - text label always present alongside colour.
        for bucket in ASSIGNABLE_BUCKETS:
            Button(
                controls, text=f"Assign: {bucket}",
                bg=BUCKET_COLOURS.get(bucket, "#ddd"),
                command=lambda b=bucket: self.assign(b),
            ).pack(fill="x", pady=3)

        Frame(controls, height=16).pack()  # spacer

        Button(controls, text="Mark ignore / not useful",
               command=self.mark_ignore, fg="#a33").pack(fill="x", pady=(0, 4))
        Button(controls, text="Skip for now",
               command=self.skip).pack(fill="x")

        self.load_current()

    def load_current(self):
        if self.index >= len(self.rows):
            self.status_label.config(text="Queue empty. Close this window.")
            self.path_label.config(text="")
            self.prediction_label.config(text="")
            self.reason_label.config(text="")
            self.image_label.config(image="")
            self.accept_button.config(text="", state="disabled")
            self.open_image_button.config(state="disabled")
            return

        row = self.rows[self.index]
        remaining = len(self.rows) - self.index
        if self.source == "uncertain":
            self.status_label.config(text=f"{remaining} remaining in uncertain_review")
        else:
            already = sum(1 for r in self.rows if (r.get("correct_category") or "").strip())
            self.status_label.config(
                text=f"{remaining} remaining to label   |   {already}/{len(self.rows)} labeled so far")
        self.path_label.config(text=row["file_path"])
        self.reason_label.config(text=f"Classifier reason: {row.get('reason', '')}")
        self.open_image_button.config(state="normal")

        predicted = (row.get("category") or "").strip()
        confidence = row.get("confidence", "")
        if predicted and predicted in ASSIGNABLE_BUCKETS:
            self.prediction_label.config(
                text=f"Gemma predicted: {predicted}   (confidence: {confidence})"
            )
            accept_text = (
                f"\u2713 Accept: {predicted}" if self.source == "uncertain"
                # Rows in misclassifications.csv were flagged BECAUSE they
                # looked wrong - this button still exists for the case
                # where a second look says Gemma's original call was
                # actually fine after all, just phrased to not imply
                # "accept" is the expected outcome here the way it is for
                # a genuinely uncertain (not yet judged) row.
                else f"Gemma's original call was actually correct: {predicted}"
            )
            self.accept_button.config(text=accept_text, state="normal")
        else:
            # Defensive - per core/loaders/gemma_loader.py, category should
            # always be a real DocumentCategory value, never empty or
            # literally "uncertain_review". If this fires, something
            # upstream changed; don't offer a one-click accept onto a
            # bucket we can't confirm is real.
            self.prediction_label.config(
                text=f"Gemma predicted: (missing/invalid category: {predicted!r})"
            )
            self.accept_button.config(text="(no valid prediction to accept)", state="disabled")

        try:
            img = Image.open(row["file_path"])
            img.thumbnail(MAX_PREVIEW_SIZE)
            self._tk_image = ImageTk.PhotoImage(img)
            self.image_label.config(image=self._tk_image)
        except Exception as e:
            self.image_label.config(image="", text=f"[Could not load image: {e}]")

    def open_image_in_viewer(self):
        """Opens the full-resolution source image in the OS default viewer -
        mirrors the same helper in model_assessment.py, for the same reason:
        the capped thumbnail often isn't enough to judge dense/handwritten
        content confidently."""
        if self.index >= len(self.rows):
            return
        path = self.rows[self.index]["file_path"]
        try:
            system = platform.system()
            if system == "Windows":
                import os
                os.startfile(path)  # noqa: S606 - local trusted path only
            elif system == "Darwin":
                subprocess.run(["open", path], check=True)
            else:
                subprocess.run(["xdg-open", path], check=True)
        except Exception as e:
            messagebox.showerror("Could not open image", str(e))

    def accept_prediction(self):
        """One-click accept of Gemma's original predicted category - routes
        through the same assign() method (and therefore the same
        confirmation dialog) as the override buttons, so the safety gate
        applies uniformly regardless of which path committed the bucket."""
        row = self.rows[self.index]
        predicted = (row.get("category") or "").strip()
        if predicted not in ASSIGNABLE_BUCKETS:
            return  # button should be disabled in this case, but guard anyway
        self.assign(predicted)

    def _write_bucket_row(self, bucket: str, row: dict):
        path = BUCKET_DIR / f"{bucket}.csv"
        is_new = not path.exists() or path.stat().st_size == 0
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=BUCKET_CSV_FIELDS)
            if is_new:
                writer.writeheader()
            out_row = {k: row.get(k, "") for k in BUCKET_CSV_FIELDS}
            out_row["category"] = bucket
            writer.writerow(out_row)

    def _log_review(self, row: dict, assigned_bucket: str, note: str = ""):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        is_new = not REVIEWED_LOG.exists() or REVIEWED_LOG.stat().st_size == 0
        with open(REVIEWED_LOG, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=REVIEWED_LOG_FIELDS)
            if is_new:
                writer.writeheader()
            writer.writerow({
                "file_path": row["file_path"],
                "original_bucket": "uncertain_review",
                "assigned_bucket": assigned_bucket,
                "reviewer": self.reviewer_name.get(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "note": note,
            })

    def assign(self, bucket: str):
        row = self.rows[self.index]
        if self.source == "uncertain":
            confirmed = messagebox.askyesno(
                "Confirm bucket assignment",
                f"Move this image to:\n\n{bucket}\n\n"
                f"File: {Path(row['file_path']).name}\n\n"
                "This will add it to that bucket's CSV and remove it from "
                "the review queue. This cannot be undone from this screen.",
                icon="question",
            )
            if not confirmed:
                return  # stays on the same image, no state change
            self._write_bucket_row(bucket, row)
            self._log_review(row, assigned_bucket=bucket)
            self._advance()
        else:
            # misclassifications.csv is a ground-truth log, not a queue -
            # no move, no confirmation dialog (labeling is a lightweight,
            # freely-correctable action, not a destructive one). Just fill
            # in this row's correct_category and step forward; the row
            # stays in the file either way, updated in place on save.
            row["correct_category"] = bucket
            self.index += 1
            self.load_current()

    def mark_ignore(self):
        row = self.rows[self.index]
        if self.source == "uncertain":
            self._log_review(row, assigned_bucket="ignored", note="Marked not useful during review")
            self._advance()
        else:
            row["correct_category"] = "ignored"
            self.index += 1
            self.load_current()

    def skip(self):
        # Leaves the row in the queue for next session - just advance
        # the in-memory pointer without removing/logging it.
        self.index += 1
        self.load_current()

    def _advance(self):
        # Row handled - remove it from the in-memory list so it won't
        # be rewritten back to uncertain_review.csv on save. Only used in
        # "uncertain" source mode - misclassifications.csv rows are never
        # removed, just labeled in place (see assign()/mark_ignore()).
        del self.rows[self.index]
        self.load_current()

    def save_remaining(self):
        """
        "uncertain" source: rewrites uncertain_review.csv with whatever's
        left (assigned/ignored rows removed, skipped rows retained) PLUS
        every error_rows entry, unchanged - error rows are never shown or
        mutated by this review UI, but must still round-trip back into
        the file rather than being dropped (real bug, fixed 2026-07-25 -
        see load_uncertain_rows()'s docstring).

        "misclassifications" source: rewrites misclassifications.csv with
        every row (nothing is ever removed in this mode), carrying
        forward each row's own correct_category value - labeled or still
        blank, so a partially-worked session is never lost.
        """
        if self.source == "uncertain":
            with open(UNCERTAIN_CSV, "w", newline="", encoding="utf-8") as f:
                fieldnames = BUCKET_CSV_FIELDS + ["error"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in self.error_rows + self.rows:
                    out_row = {k: row.get(k, "") for k in fieldnames}
                    writer.writerow(out_row)
        else:
            with open(MISCLASSIFICATIONS_CSV, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=MISCLASSIFICATION_FIELDS)
                writer.writeheader()
                for row in self.rows:
                    out_row = {k: row.get(k, "") for k in MISCLASSIFICATION_FIELDS}
                    writer.writerow(out_row)


def load_uncertain_rows() -> tuple[list[dict], list[dict]]:
    """
    Returns (reviewable_rows, error_rows) - split, not filtered down to
    one list. error_rows (hard pipeline failures, e.g. a classifier
    parse error - see core/loaders/gemma_loader.py's _parse_kv_block)
    aren't classification-ambiguity calls for a human to adjudicate
    here (core/extractor.py's own gate check applies the identical
    `not row.get("error")` split and deliberately does NOT block
    extraction on them, for the same reason), so they're excluded from
    the accept/reject review UI.

    REAL BUG FIXED HERE (2026-07-25, found via a live run): this used
    to return ONLY the reviewable rows, discarding error_rows outright
    at load time. save_remaining() rewrites uncertain_review.csv from
    the in-memory row list on every close - since error rows were never
    loaded into that list, simply opening this tool, doing anything at
    all (even just Skip on an unrelated row), and closing it silently
    and permanently deleted every hard-error row from the CSV, with no
    logging and no recovery path - confirmed by reproducing exactly
    this against a real run's uncertain_review.csv. error_rows must be
    carried through untouched and written back by save_remaining(),
    not just excluded from the reviewable list.
    """
    if not UNCERTAIN_CSV.exists():
        return [], []
    with open(UNCERTAIN_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = [row for row in reader if row.get("file_path")]
    reviewable = [row for row in all_rows if not row.get("error")]
    error_rows = [row for row in all_rows if row.get("error")]
    return reviewable, error_rows


def load_misclassification_rows() -> list[dict]:
    """
    All rows of data/misclassifications.csv, in file order - the file
    ui/classifier_validation_ui.py's "Flag Misclassified" button appends
    to. No error_rows split (this file has no hard-pipeline-error
    concept, unlike uncertain_review.csv); rows already labeled
    (correct_category set) are still returned - session.index just
    starts wherever load_current() naturally lands, and the "N/M labeled
    so far" status line makes it obvious progress already exists.
    """
    if not MISCLASSIFICATIONS_CSV.exists():
        return []
    with open(MISCLASSIFICATIONS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader if row.get("file_path")]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source", choices=["uncertain", "misclassifications"], default="uncertain",
        help="uncertain (default): review the live uncertain_review.csv queue, moving each "
             "image into its assigned bucket. misclassifications: label "
             "data/misclassifications.csv's correct_category column in place, without "
             "moving anything - see this file's module docstring for the full distinction.",
    )
    args = parser.parse_args()

    if args.source == "uncertain":
        rows, error_rows = load_uncertain_rows()
        source_label = "uncertain_review.csv"
    else:
        rows, error_rows = load_misclassification_rows(), []
        source_label = "misclassifications.csv"

    if error_rows:
        print(f"\n{len(error_rows)} row(s) in uncertain_review.csv recorded a hard "
              f"pipeline error (not reviewable here - these need the underlying bug "
              f"fixed, not a bucket reassignment; left untouched, not extraction-"
              f"gate-blocking either, matching core/extractor.py's own gate check):")
        for row in error_rows:
            print(f"  - {row.get('file_path')}: {row.get('error', '')[:150]}")

    if not rows:
        if not error_rows:
            print(f"{source_label} is empty or contains no reviewable rows. Nothing to do.")
        return

    root = Tk()
    app = ReviewApp(root, rows, error_rows, source=args.source)

    def on_close():
        app.save_remaining()
        if args.source == "uncertain":
            remaining = len(app.rows)
            if remaining > 0:
                print(f"\n{remaining} row(s) still in uncertain_review.csv - "
                      f"extraction gate NOT clear yet.")
            else:
                print("\nuncertain_review.csv is now empty - extraction gate clear.")
        else:
            labeled = sum(1 for r in app.rows if (r.get("correct_category") or "").strip())
            print(f"\n{labeled}/{len(app.rows)} row(s) in misclassifications.csv now have a "
                  f"correct_category label.")
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
