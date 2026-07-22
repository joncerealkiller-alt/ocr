"""
Ground Truth Labeling UI — builds a human-verified per-field training/
evaluation set, walking a segmentation sidecar row by row, field by
field.

Consistent with this project's core principle: the OCR pipeline is a
FINDING AID, not the source of truth. This tool's job is to capture
what a human can ACTUALLY read in the crop, including genuine
illegibility - it does not aim to produce a complete, polished
transcription. Marking a field "illegible" is a correct, valuable
answer, not a failure to push past.

Shows EXACTLY the same crop the model sees (same bbox, same column
mask applied via core.row_segmentation.crop_region_from_source) - not
a wider context crop - so labels are directly comparable to what a
model was actually asked to read.

Usage:
    python ground_truth_labeling_ui.py

Output: appends one JSON record per field to
data/outputs/ground_truth_log.jsonl (JSONL, matching
test_stage2_isolated.py's --log-file convention) - never overwrites,
so labeling can be paused/resumed across sessions without losing prior
work. A field already labeled for a given (sidecar, row, column) is
skipped on reload, not re-asked.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from tkinter import (
    Tk, Frame, Label, Button, Entry, StringVar, filedialog, messagebox,
    END, DISABLED, NORMAL, Radiobutton,
)
from PIL import Image, ImageTk

from core.row_segmentation import load_sidecar, crop_region_from_source, compute_exclude_ranges

PROJECT_ROOT = Path(__file__).resolve().parent
LOG_PATH = PROJECT_ROOT / "data" / "outputs" / "ground_truth_log.jsonl"
PREVIEW_SIZE = (900, 300)  # single row crop is wide and short - not a
                            # square thumbnail budget like other tools

STATUS_OPTIONS = [
    ("readable", "Readable — type the exact value below"),
    ("partially_readable", "Partially readable — type what you can read, mark the rest with ?"),
    ("illegible", "Illegible — genuinely cannot be read (not a guess)"),
    ("blank", "Genuinely blank on the form (not illegible — nothing was written)"),
]


def _load_existing_keys(log_path: Path) -> set[tuple]:
    """
    Returns the set of (sidecar_path, row_index, column) already
    labeled, so a resumed session skips fields already done rather than
    re-asking - per the project's discipline of never silently
    discarding prior work.
    """
    keys = set()
    if not log_path.exists():
        return keys
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            keys.add((rec.get("sidecar_path"), rec.get("row_index"), rec.get("column")))
    return keys


class LabelingApp:
    def __init__(self, root: Tk):
        self.root = root
        root.title("Ground Truth Labeling — finding-aid honesty, not duplication accuracy")
        root.geometry("1000x760")
        root.minsize(900, 650)

        self.sidecar_path: str | None = None
        self.sidecar: dict | None = None
        self.column_names: list[str] = []
        self.queue: list[tuple[int, str]] = []  # (row_index, column) pairs remaining
        self.queue_pos = 0
        self.existing_keys: set[tuple] = set()
        self._tk_image = None
        self.status_var = StringVar(value="readable")

        warning = Label(
            root,
            text="This is a finding aid. The goal is honest capture of what's actually "
                 "legible, not a complete-looking transcription. 'Illegible' and 'blank' "
                 "are correct answers, not failures.",
            fg="#a33", font=("Segoe UI", 9, "bold"), wraplength=950, justify="left",
        )
        warning.pack(pady=(8, 4), padx=12)

        top = Frame(root, padx=12)
        top.pack(fill="x")

        load_row = Frame(top)
        load_row.pack(fill="x", pady=4)
        Button(load_row, text="Load sidecar...", command=self.load_sidecar_file).pack(side="left")
        Button(load_row, text="Load columns file...", command=self.load_columns_file).pack(
            side="left", padx=(6, 0))
        self.sidecar_label = Label(load_row, text="(no sidecar loaded)", fg="#666")
        self.sidecar_label.pack(side="left", padx=10)

        self.progress_label = Label(top, text="", font=("Segoe UI", 9, "bold"))
        self.progress_label.pack(anchor="w", pady=(4, 0))

        self.field_label = Label(top, text="", font=("Segoe UI", 13, "bold"), fg="#225")
        self.field_label.pack(anchor="w", pady=(8, 4))

        # -- image preview --------------------------------------------------
        preview_frame = Frame(root, padx=12)
        preview_frame.pack(fill="x")
        self.image_label = Label(preview_frame, bg="#ddd")
        self.image_label.pack(anchor="w")

        # -- status radio buttons --------------------------------------------
        status_frame = Frame(root, padx=12)
        status_frame.pack(fill="x", pady=(10, 4))
        Label(status_frame, text="What can you actually read here?",
              font=("Segoe UI", 9, "bold")).pack(anchor="w")
        for value, label in STATUS_OPTIONS:
            Radiobutton(
                status_frame, text=label, variable=self.status_var, value=value,
                command=self._on_status_change,
            ).pack(anchor="w")

        # -- value entry ---------------------------------------------------
        entry_frame = Frame(root, padx=12)
        entry_frame.pack(fill="x", pady=(6, 4))
        Label(entry_frame, text="Exact value (leave blank if illegible/blank above):",
              font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.value_var = StringVar(value="")
        self.value_entry = Entry(entry_frame, textvariable=self.value_var,
                                  font=("Consolas", 12), width=60)
        self.value_entry.pack(anchor="w", pady=(2, 0), fill="x")
        Label(entry_frame, text="Use ? for individual illegible characters within an "
                                 "otherwise-readable value (e.g. \"J?hn\").",
              font=("Segoe UI", 8), fg="#666").pack(anchor="w", pady=(2, 0))

        # -- notes -----------------------------------------------------------
        notes_frame = Frame(root, padx=12)
        notes_frame.pack(fill="x", pady=(6, 4))
        Label(notes_frame, text="Notes (optional — e.g. \"ditto mark\", \"crossed out\", "
                                 "\"stray mark, not a real character\"):").pack(anchor="w")
        self.notes_var = StringVar(value="")
        Entry(notes_frame, textvariable=self.notes_var, width=80).pack(anchor="w", pady=(2, 0))

        # -- nav / save ---------------------------------------------------
        nav_frame = Frame(root, padx=12, pady=10)
        nav_frame.pack(fill="x")
        self.save_next_button = Button(
            nav_frame, text="Save & Next \u2192", command=self.save_and_next,
            bg="#2a6", fg="white", font=("Segoe UI", 10, "bold"), state=DISABLED,
        )
        self.save_next_button.pack(side="left")
        Button(nav_frame, text="Skip (don't save, come back later)",
               command=self.skip_field).pack(side="left", padx=(8, 0))
        self.status_label = Label(nav_frame, text="", fg="#444")
        self.status_label.pack(side="left", padx=12)

        # Enter key in the value box acts as Save & Next, for fast
        # keyboard-only labeling once a rhythm is established.
        self.value_entry.bind("<Return>", lambda e: self.save_and_next())

        self._on_status_change()  # set initial entry enabled/disabled state

    # -- loading ------------------------------------------------------------

    def load_sidecar_file(self):
        path = filedialog.askopenfilename(
            title="Select segmentation sidecar JSON",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        try:
            self.sidecar = load_sidecar(path)
        except Exception as e:
            messagebox.showerror("Could not load sidecar", str(e))
            return
        self.sidecar_path = path
        self.sidecar_label.config(text=Path(path).name, fg="black")
        if self.column_names:
            self._build_queue()

    def load_columns_file(self):
        path = filedialog.askopenfilename(
            title="Select column names file",
            filetypes=[("Text", "*.txt")],
        )
        if not path:
            return
        names = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        if not names:
            messagebox.showerror("Empty file", "Column file contained no names.")
            return
        self.column_names = names
        if self.sidecar is not None:
            self._build_queue()

    def _build_queue(self):
        """
        Builds the (row_index, column) work queue across every row in
        the sidecar x every column - skipping any pair already present
        in the existing log, so resuming a session doesn't re-ask
        already-labeled fields. Queue order is row-major (all columns
        for row 1, then row 2, ...) rather than column-major, matching
        how a person would naturally work through a physical page.
        """
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.existing_keys = _load_existing_keys(LOG_PATH)

        self.queue = []
        for row in self.sidecar["rows"]:
            for col in self.column_names:
                key = (self.sidecar_path, row["index"], col)
                if key not in self.existing_keys:
                    self.queue.append((row["index"], col))
        self.queue_pos = 0

        total_possible = len(self.sidecar["rows"]) * len(self.column_names)
        already_done = total_possible - len(self.queue)
        self.status_label.config(
            text=f"{already_done}/{total_possible} already labeled, resuming.")

        if not self.queue:
            messagebox.showinfo("Nothing to label",
                                 "Every (row, column) pair in this sidecar with these "
                                 "columns has already been labeled.")
            self.save_next_button.config(state=DISABLED)
            return

        self.save_next_button.config(state=NORMAL)
        self._show_current()

    # -- display ------------------------------------------------------------

    def _current_pair(self) -> tuple[int, str] | None:
        if self.queue_pos >= len(self.queue):
            return None
        return self.queue[self.queue_pos]

    def _show_current(self):
        pair = self._current_pair()
        if pair is None:
            self.progress_label.config(text="Queue complete for this sidecar.")
            self.field_label.config(text="")
            self.image_label.config(image="", text="(done)")
            self.save_next_button.config(state=DISABLED)
            return

        row_index, column = pair
        self.progress_label.config(
            text=f"Field {self.queue_pos + 1} of {len(self.queue)} remaining "
                 f"(row {row_index}, column {column!r})")
        self.field_label.config(text=f"{column}  —  Row {row_index}")

        row = next(r for r in self.sidecar["rows"] if r["index"] == row_index)
        source_path = self.sidecar["source_image_path"]
        deskew_angle = self.sidecar["deskew_angle"]

        # Same mask logic core.row_extraction.run_single_column_extraction
        # actually uses (2026-07-22 sidecar redesign: masks now live per-
        # column under sidecar["columns"][name], not as one global top-
        # level field) - the crop shown here must match what the model
        # actually saw for THIS column, or a label built against a
        # differently-masked crop isn't a valid comparison point. Falls
        # back to the old top-level mask_keep_ranges/mask_apply_rows only
        # for a legacy sidecar that predates the per-column redesign and
        # has no "columns" entry at all.
        column_state = self.sidecar.get("columns", {}).get(column)
        if column_state is not None:
            keep_ranges = [tuple(k) for k in column_state.get("mask_keep_ranges", [])]
            apply_rows = column_state.get("mask_apply_rows", True)
        else:
            keep_ranges = [tuple(k) for k in self.sidecar.get("mask_keep_ranges", [])]
            apply_rows = self.sidecar.get("mask_apply_rows", False)
        width = self.sidecar["deskewed_image_size"][0]
        row_masks = (
            compute_exclude_ranges(keep_ranges, width) if keep_ranges and apply_rows else []
        )

        try:
            crop = crop_region_from_source(source_path, row["bbox"], deskew_angle, row_masks)
        except Exception as e:
            messagebox.showerror("Could not load crop", str(e))
            return

        preview = crop.copy()
        preview.thumbnail(PREVIEW_SIZE)
        self._tk_image = ImageTk.PhotoImage(preview)
        self.image_label.config(image=self._tk_image, text="")

        # Reset entry state for the new field - never carry over the
        # previous field's typed value.
        self.status_var.set("readable")
        self.value_var.set("")
        self.notes_var.set("")
        self._on_status_change()
        self.value_entry.focus_set()

    def _on_status_change(self):
        """
        Disables/clears the value entry for illegible/blank statuses -
        prevents accidentally leaving a stray typed value attached to a
        field marked as unreadable, which would silently contradict the
        status and risk being read later as a real value.
        """
        status = self.status_var.get()
        if status in ("illegible", "blank"):
            self.value_var.set("")
            self.value_entry.config(state=DISABLED)
        else:
            self.value_entry.config(state=NORMAL)

    # -- save / navigation ----------------------------------------------

    def save_and_next(self):
        pair = self._current_pair()
        if pair is None:
            return
        row_index, column = pair
        status = self.status_var.get()
        value = self.value_var.get() if status in ("readable", "partially_readable") else ""

        if status == "readable" and not value.strip():
            messagebox.showwarning(
                "Empty value",
                "Status is 'Readable' but no value was entered. Type the value, or "
                "change the status to Illegible/Blank if that's actually the case.")
            return

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sidecar_path": self.sidecar_path,
            "source_image_path": self.sidecar["source_image_path"],
            "row_index": row_index,
            "column": column,
            "status": status,
            "value": value,
            "notes": self.notes_var.get().strip(),
        }
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        self.existing_keys.add((self.sidecar_path, row_index, column))
        self.queue_pos += 1
        self._show_current()

    def skip_field(self):
        """
        Advances without writing a record - the field stays unlabeled
        and will be re-offered next time this sidecar+columns
        combination is loaded (skip is NOT the same as 'blank' or
        'illegible', which ARE real recorded answers)."""
        self.queue_pos += 1
        self._show_current()


def main():
    root = Tk()
    LabelingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()