"""
Human-review gate for Stage 5 (Document Routing) per docs/PIPELINE_
STAGE_TERMINOLOGY.md's canonical Stage 0-6 naming. Reviews images one at
a time and assigns each to the correct taxonomy option, logging the
correction for later classifier improvement.

TAXONOMY-DRIVEN (2026-08-04 refactor - see docs/TAXONOMY.md): this file
contains NO hardcoded bucket list and NO hardcoded mode-specific logic.
Every button shown is generated from core/taxonomy.py's load_taxonomy()
via whichever core/review_modes.py ReviewMode is active; what happens on
submit is entirely delegated to that mode object. This class only knows
how to display an image, render whatever options the active mode hands
it, and call back into the mode on every action - it never branches on
a source string itself. See core/review_modes.py's own module docstring
for the three modes (production/research/subtype) and exactly what each
one does differently.

Usage:
    python review_uncertain.py
    python review_uncertain.py --source misclassifications
    python review_uncertain.py --source subtype

Design per project discussion (behavior unchanged by this refactor):
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
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path

from tkinter import Tk, Frame, Label, Button, StringVar, messagebox
from PIL import Image, ImageTk

# Moved into debug_tools/ (2026-07-25) - one directory deeper than repo root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.pipeline_db import DEFAULT_DB_PATH
from core.taxonomy import load_taxonomy
from core.review_modes import ReviewMode, build_mode, MODE_REGISTRY

ACCEPT_COLOUR = "#4caf50"  # distinct "agree" green, independent of option
                           # identity colours - this button's colour
                           # signals the action (accept), not which option
DEFAULT_OPTION_COLOUR = "#ddd"

MAX_PREVIEW_SIZE = (500, 650)


class ReviewApp:
    def __init__(self, root: Tk, mode: ReviewMode, rows: list[dict], error_rows: list[dict] | None = None):
        self.root = root
        self.mode = mode
        self.rows = rows
        # Hard pipeline-failure rows (e.g. classifier parse errors) -
        # never shown in this review UI and never mutated by accept/
        # skip/ignore, but must round-trip back into their source file
        # unchanged via mode.save() - see ProductionReviewMode.load()'s
        # docstring for the real data-loss bug this fixes. Always empty
        # for research/subtype modes - neither has an equivalent
        # hard-error concept.
        self.error_rows = error_rows or []
        self.index = 0
        self.reviewer_name = StringVar(value="jonny")

        root.title(mode.label)
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

        # -- accept the mode's predicted option: own block, own colour,
        # separate from the override list below - common/fast path,
        # styled distinctly from "I disagree, pick something else".
        self.accept_button = Button(
            controls, text="", bg=ACCEPT_COLOUR, fg="white", wraplength=290,
            font=("Segoe UI", 10, "bold"), command=self.accept_prediction,
        )
        self.accept_button.pack(fill="x", pady=(0, 14))

        self.options_label_widget = Label(controls, text=mode.options_label, font=("Segoe UI", 9))
        self.options_label_widget.pack(anchor="w", pady=(0, 6))

        # Taxonomy-driven option buttons live in their own frame, rebuilt
        # on every load_current() call - the option set can differ per
        # ROW (Subtype mode: different rows have different primary
        # buckets, so different valid subtypes), not just per mode, so
        # these can't be built once in __init__ the way a hardcoded list
        # could be.
        self.options_frame = Frame(controls)
        self.options_frame.pack(fill="x")

        Frame(controls, height=16).pack()  # spacer

        self.needs_new_button = Button(
            controls, text=mode.needs_new_button_text(),
            command=self.mark_needs_new, fg="#a33",
        )
        self.needs_new_button.pack(fill="x", pady=(0, 4))
        Button(controls, text="Bad deskew (preprocessing made it worse)",
               command=self.mark_bad_deskew, fg="#a33").pack(fill="x", pady=(0, 4))
        Button(controls, text="Mark ignore / not useful",
               command=self.mark_ignore, fg="#a33").pack(fill="x", pady=(0, 4))
        Button(controls, text="Skip for now",
               command=self.skip).pack(fill="x")

        self.load_current()

    def _rebuild_option_buttons(self, row: dict) -> None:
        for child in self.options_frame.winfo_children():
            child.destroy()
        for option in self.mode.get_options(row):
            Button(
                self.options_frame, text=f"Assign: {option.display_name}",
                bg=option.color or DEFAULT_OPTION_COLOUR,
                command=lambda oid=option.id: self.assign(oid),
            ).pack(fill="x", pady=3)

    def load_current(self):
        if self.index >= len(self.rows):
            self.status_label.config(text="Queue empty. Close this window.")
            self.path_label.config(text="")
            self.prediction_label.config(text="")
            self.reason_label.config(text="")
            self.image_label.config(image="")
            self.accept_button.config(text="", state="disabled")
            self.open_image_button.config(state="disabled")
            for child in self.options_frame.winfo_children():
                child.destroy()
            return

        row = self.rows[self.index]
        self.status_label.config(text=self.mode.status_text(self.rows, self.index))
        self.path_label.config(text=row["file_path"])
        self.reason_label.config(text=f"Classifier reason: {row.get('reason', '')}")
        self.open_image_button.config(state="normal")

        self._rebuild_option_buttons(row)

        options = self.mode.get_options(row)
        options_by_id = {o.id: o for o in options}
        predicted_id = self.mode.predicted_id(row)
        if predicted_id is not None and predicted_id in options_by_id:
            predicted = options_by_id[predicted_id]
            confidence = row.get("confidence", "")
            self.prediction_label.config(
                text=f"Predicted: {predicted.display_name}   (confidence: {confidence})"
            )
            self.accept_button.config(
                text=self.mode.accept_button_text(predicted.display_name), state="normal",
            )
        else:
            # Defensive - a genuinely invalid/missing prediction (or, for
            # Subtype mode, a bucket with zero defined subtypes) offers
            # nothing to one-click accept, rather than accepting onto an
            # option that isn't actually valid right now.
            raw_predicted = row.get("category") or row.get("bucket") or row.get("subtype") or ""
            self.prediction_label.config(
                text=f"Predicted: (missing/invalid: {raw_predicted!r})" if raw_predicted
                else "Predicted: (none)"
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
        """One-click accept of the mode's predicted option - routes
        through the same assign() method (and therefore the same
        confirmation gate) as the override buttons, so the safety
        behavior applies uniformly regardless of which path committed
        the choice."""
        row = self.rows[self.index]
        predicted_id = self.mode.predicted_id(row)
        if predicted_id is None:
            return  # button should be disabled in this case, but guard anyway
        self.assign(predicted_id)

    def assign(self, choice_id: str):
        row = self.rows[self.index]
        if self.mode.requires_confirmation:
            confirmed = messagebox.askyesno(
                "Confirm assignment",
                f"Assign this image to:\n\n{choice_id}\n\n"
                f"File: {Path(row['file_path']).name}\n\n"
                "This will add it to that bucket's CSV and remove it from "
                "the review queue. This cannot be undone from this screen.",
                icon="question",
            )
            if not confirmed:
                return  # stays on the same image, no state change
        self.mode.assign(self, row, choice_id)

    def mark_ignore(self):
        row = self.rows[self.index]
        self.mode.mark_ignore(self, row)

    def mark_needs_new(self):
        row = self.rows[self.index]
        self.mode.mark_needs_new(self, row)

    def mark_bad_deskew(self):
        row = self.rows[self.index]
        self.mode.mark_bad_deskew(self, row)

    def skip(self):
        # Leaves the row in place for next session - just advance the
        # in-memory pointer without removing/logging it.
        self.index += 1
        self.load_current()

    def _advance(self):
        # Row handled - remove it from the in-memory list so it won't
        # be rewritten back to its source file on save(). Only ever
        # called by modes whose semantics are "row leaves the queue"
        # (ProductionReviewMode) - research/subtype modes relabel rows
        # in place instead and never call this.
        del self.rows[self.index]
        self.load_current()

    def save(self):
        self.mode.save(self)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source", choices=list(MODE_REGISTRY.keys()), default="uncertain",
        help="uncertain (default): review the live uncertain_review.csv queue, moving each "
             "image into its assigned bucket. misclassifications: label "
             "data/misclassifications.csv's correct_category column in place, without "
             "moving anything. subtype: label data/subtype_review_queue.csv's subtype "
             "column in place, using the row's already-assigned bucket to determine which "
             "subtype options apply (core/taxonomy.py's subtypes_for()) - see "
             "core/review_modes.py's module docstring for the full distinction between modes.",
    )
    parser.add_argument(
        "--db-path", default=str(DEFAULT_DB_PATH),
        help=f"core/pipeline_db.py database path (default: {DEFAULT_DB_PATH})",
    )
    args = parser.parse_args()

    taxonomy = load_taxonomy()
    mode = build_mode(args.source, db_path=Path(args.db_path), taxonomy=taxonomy)
    rows, error_rows = mode.load()

    if error_rows:
        print(f"\n{len(error_rows)} row(s) recorded a hard pipeline error (not reviewable "
              f"here - these need the underlying bug fixed, not a bucket reassignment; left "
              f"untouched, not extraction-gate-blocking either, matching core/extractor.py's "
              f"own gate check):")
        for row in error_rows:
            print(f"  - {row.get('file_path')}: {row.get('error', '')[:150]}")

    if not rows:
        if not error_rows:
            print(f"Nothing to review for --source {args.source}.")
        return

    root = Tk()
    app = ReviewApp(root, mode, rows, error_rows)

    def on_close():
        app.save()
        if args.source == "uncertain":
            remaining = len(app.rows)
            if remaining > 0:
                print(f"\n{remaining} row(s) still in uncertain_review.csv - "
                      f"extraction gate NOT clear yet.")
            else:
                print("\nuncertain_review.csv is now empty - extraction gate clear.")
        else:
            label_field = "correct_category" if args.source == "misclassifications" else "subtype"
            labeled = sum(1 for r in app.rows if (r.get(label_field) or "").strip())
            print(f"\n{labeled}/{len(app.rows)} row(s) now have a {label_field} label.")
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
