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
import sys
from datetime import datetime, timezone
from pathlib import Path

# Moved into ui/ (2026-07-25) - one directory deeper than repo root, so
# repo root must be put back on sys.path before any `core.*` import
# below will resolve.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tkinter import (
    Tk, Frame, Label, Button, Entry, StringVar, BooleanVar, filedialog, messagebox,
    END, DISABLED, NORMAL, Radiobutton, Checkbutton, Canvas, Scrollbar, HORIZONTAL,
)
from PIL import Image, ImageDraw, ImageTk

from core.row_segmentation import load_sidecar, crop_region_from_source, compute_exclude_ranges
LOG_PATH = PROJECT_ROOT / "data" / "outputs" / "ground_truth_log.jsonl"
PREVIEW_SIZE = (900, 300)  # single row crop is wide and short - not a
                            # square thumbnail budget like other tools
ROW_CONTEXT_HEIGHT = 90     # full-row context panel - short and wide,
                            # scrolled horizontally rather than shrunk to fit

STATUS_OPTIONS = [
    ("readable", "Readable — type the exact value below"),
    ("partially_readable", "Partially readable — type what you can read, mark the rest with ?"),
    ("illegible", "Illegible — genuinely cannot be read (not a guess)"),
    ("blank", "Genuinely blank on the form (not illegible — nothing was written)"),
]

# Quick-entry phrases for the Notes field (2026-07-26, Jon's direction) -
# each checkbox adds/removes its own phrase in the comma-joined Notes
# text rather than replacing it, so several can be combined (e.g.
# "ditto mark, blurry") and freeform typing still works alongside them.
NOTE_PHRASES = [
    "ditto mark",
    "crossed out",
    "blurry",
    "faded",
    "overwritten",
    "stray mark, not a real character",
    "ink bleed",
    "faded ink",
    "enumerator notes",
]

# Quick-fill buttons for the Exact value entry, keyed by column name
# (2026-07-26, Jon's direction) - these are for a HUMAN who has already
# looked at the image and decided what it says, purely to save typing;
# NOT the same as model-prompt field hints (core/row_extraction.py's
# _FIELD_TYPE_HINTS), which deliberately omit concrete candidate values
# to avoid seeding a model's guess - that risk doesn't apply here since
# the human forms their own judgment first and the button only fills in
# what they already decided. Extend this dict for more columns as needed.
QUICK_FILL_VALUES = {
    "Sex": ["M", "F"],
    "Relationship to Head": [
        "Head", "Wife", "Husband", "Son", "Daughter", "Servant", "Domestic",
        "Boarder", "Lodger", "Mother", "Father", "Brother", "Sister",
    ],
    "Birthplace": ["Ont", "Man", "Ontario", "Manitoba", "Eng", "England", "Scotland", "USA", "Austria", "Ukraine",],
}


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

        # -- full row context preview (2026-07-26, Jon's direction) --------
        # Purely contextual evidence for difficult handwriting - shows the
        # WHOLE row (unmasked, every column's real content visible, unlike
        # the labeling crop below which paints everything outside the
        # active column white), with the field actually being labeled
        # boxed in lime. Never affects what gets labeled or saved - see
        # save_and_next(), unchanged.
        row_context_frame = Frame(root, padx=12)
        row_context_frame.pack(fill="x", pady=(6, 0))
        Label(row_context_frame,
              text="Full census row (context only - lime box is the field being labeled):",
              font=("Segoe UI", 8), fg="#666").pack(anchor="w")
        canvas_wrap = Frame(row_context_frame)
        canvas_wrap.pack(fill="x")
        self.row_canvas = Canvas(canvas_wrap, bg="#ddd", height=ROW_CONTEXT_HEIGHT + 4,
                                  width=PREVIEW_SIZE[0], highlightthickness=0)
        row_scroll = Scrollbar(canvas_wrap, orient=HORIZONTAL, command=self.row_canvas.xview)
        self.row_canvas.configure(xscrollcommand=row_scroll.set)
        self.row_canvas.pack(side="top", fill="x")
        row_scroll.pack(side="top", fill="x")
        self._tk_row_image = None

        # -- image preview (current field, zoomed) ---------------------------
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
        value_row = Frame(entry_frame)
        value_row.pack(fill="x", pady=(2, 0))

        self.value_var = StringVar(value="")
        self.value_entry = Entry(
        value_row,
        textvariable=self.value_var,
        font=("Consolas", 12),
        width=60,
        )
        self.value_entry.pack(fill="x", expand=True)
        
        # Quick-fill buttons (2026-07-26) - rebuilt per-field in
        # _show_current() for whichever column is currently active;
        # empty for any column not listed in QUICK_FILL_VALUES.
        self.quick_fill_frame = Frame(entry_frame)
        self.quick_fill_frame.pack(anchor="w", pady=(4, 0))

        self._quick_fill_buttons: list[Button] = []
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

        # Quick-entry checkboxes (2026-07-26) - each toggles its own
        # phrase in/out of the comma-joined Notes text above, without
        # disturbing any other phrase or freeform text already there.
        note_check_row = Frame(notes_frame)
        note_check_row.pack(anchor="w", pady=(4, 0))
        self.note_phrase_vars: dict[str, BooleanVar] = {}
        for phrase in NOTE_PHRASES:
            var = BooleanVar(value=False)
            self.note_phrase_vars[phrase] = var
            Checkbutton(
                note_check_row, text=phrase, variable=var,
                command=lambda p=phrase, v=var: self._toggle_note_phrase(p, v),
            ).pack(side="left", padx=(0, 10))

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
        # Fixed default (2026-07-26) - every sidecar this project produces
        # lands in data/outputs/row_segmentation/ (row_segmentation_ui.py's
        # own hardcoded OUTPUT_DIR, not user-configurable there), so unlike
        # Select image (which can point anywhere, hence "remember last
        # used" in ui/row_segmentation_ui.py) this one has a single real
        # canonical location, same as Load columns file's fix above.
        sidecar_dir = PROJECT_ROOT / "data" / "outputs" / "row_segmentation"
        path = filedialog.askopenfilename(
            title="Select segmentation sidecar JSON",
            filetypes=[("JSON", "*.json")],
            initialdir=str(sidecar_dir) if sidecar_dir.is_dir() else None,
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
        # Fixed default (2026-07-26, matching ui/row_segmentation_ui.py's
        # same fix) - config/columns/ is the actual canonical location
        # every real column list lives in.
        columns_dir = PROJECT_ROOT / "config" / "columns"
        path = filedialog.askopenfilename(
            title="Select column names file",
            filetypes=[("Text", "*.txt")],
            initialdir=str(columns_dir) if columns_dir.is_dir() else None,
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

        self._render_row_context(row, source_path, deskew_angle, keep_ranges)

        preview = crop.copy()
        # apply_column_mask() PAINTS everything outside the kept
        # range(s) white - it does NOT narrow the image's actual pixel
        # dimensions. So a masked column crop is still the FULL row
        # width (often 3000-4000px), with only a narrow sliver of real
        # content and the rest blank. Scaling that whole thing to fit
        # the preview budget made the sliver nearly invisible (confirmed
        # via real screenshots, 2026-07-22) - upscaling alone couldn't
        # fix this because the problem wasn't that the crop was small,
        # it was that 90%+ of it is blank margin.
        #
        # For DISPLAY ONLY, tighten to a padded region around the union
        # of kept ranges (converted from full-image x-coordinates to
        # this crop's LOCAL coordinates by subtracting the row bbox's
        # x0). This does NOT change what's sent to the model - that
        # extraction path uses the untouched `crop` from
        # crop_region_from_source() above, exactly as before. Falls
        # back to the full crop unchanged when there's no mask (nothing
        # to tighten around).
        if keep_ranges:
            crop_x0 = row["bbox"][0]
            local_ranges = [(x0 - crop_x0, x1 - crop_x0) for x0, x1 in keep_ranges]
            pad = 20
            left = max(0, min(r[0] for r in local_ranges) - pad)
            right = min(preview.width, max(r[1] for r in local_ranges) + pad)
            if right > left:
                preview = preview.crop((left, 0, right, preview.height))

        # Image.thumbnail() only ever SHRINKS - now that preview is
        # tightened to roughly the actual content (not a huge mostly-
        # blank row), it's typically far smaller than PREVIEW_SIZE, so
        # scale UP to fill the budget (capped so an extremely tiny crop
        # doesn't blow up into a blurry mess); only fall back to
        # shrinking for a crop that's still larger than the budget (an
        # unmasked full row, or a very wide kept range).
        MAX_UPSCALE = 6.0
        scale = min(PREVIEW_SIZE[0] / preview.width, PREVIEW_SIZE[1] / preview.height)
        if scale > 1.0:
            scale = min(scale, MAX_UPSCALE)
            new_size = (max(1, round(preview.width * scale)), max(1, round(preview.height * scale)))
            preview = preview.resize(new_size, Image.LANCZOS)
        else:
            preview.thumbnail(PREVIEW_SIZE)
        self._tk_image = ImageTk.PhotoImage(preview)
        self.image_label.config(image=self._tk_image, text="")

        # Reset entry state for the new field - never carry over the
        # previous field's typed value.
        self.status_var.set("readable")
        self.value_var.set("")
        self.notes_var.set("")
        for var in self.note_phrase_vars.values():
            var.set(False)
        self._rebuild_quick_fill_buttons(column)
        self._on_status_change()
        self.value_entry.focus_set()

    def _render_row_context(self, row: dict, source_path: str, deskew_angle: float,
                             keep_ranges: list[tuple[int, int]]):
        """
        Full, UNMASKED row crop (every column's real handwriting visible,
        not painted white outside the active column) with the current
        field boxed in lime - purely contextual evidence for judging
        difficult handwriting (a hard-to-read given name is often clearer
        once you've seen the Sex/Age columns, a ditto mark elsewhere on
        the page, or the enumerator's general letter shapes). Does not
        affect what gets labeled or saved - save_and_next() is untouched.
        """
        try:
            full_row = crop_region_from_source(source_path, row["bbox"], deskew_angle, [])
        except Exception:
            self.row_canvas.delete("all")
            return
        if full_row.mode != "RGB":
            full_row = full_row.convert("RGB")

        scale = ROW_CONTEXT_HEIGHT / full_row.height
        scaled = full_row.resize(
            (max(1, round(full_row.width * scale)), ROW_CONTEXT_HEIGHT), Image.LANCZOS
        )
        draw = ImageDraw.Draw(scaled)
        crop_x0 = row["bbox"][0]
        for x0, x1 in keep_ranges:
            lx0 = (x0 - crop_x0) * scale
            lx1 = (x1 - crop_x0) * scale
            draw.rectangle([lx0, 1, lx1, scaled.height - 2], outline="lime", width=3)

        self._tk_row_image = ImageTk.PhotoImage(scaled)
        self.row_canvas.delete("all")
        self.row_canvas.create_image(0, 0, anchor="nw", image=self._tk_row_image)
        self.row_canvas.configure(scrollregion=(0, 0, scaled.width, ROW_CONTEXT_HEIGHT))

        # Auto-scroll so the highlighted field is centered - the user
        # shouldn't have to manually hunt for it in a wide row every time.
        if keep_ranges:
            self.row_canvas.update_idletasks()
            center_x = ((min(r[0] for r in keep_ranges) + max(r[1] for r in keep_ranges)) / 2
                        - crop_x0) * scale
            visible_w = self.row_canvas.winfo_width() or PREVIEW_SIZE[0]
            denom = max(1, scaled.width - visible_w)
            frac = max(0.0, min(1.0, (center_x - visible_w / 2) / denom))
            self.row_canvas.xview_moveto(frac)
        else:
            self.row_canvas.xview_moveto(0)

    def _toggle_note_phrase(self, phrase: str, var: "BooleanVar"):
        """Adds/removes exactly this phrase from the comma-joined Notes
        text, preserving any other phrase or freeform text already
        there - never a blind overwrite."""
        current = [p.strip() for p in self.notes_var.get().split(",") if p.strip()]
        if var.get():
            if phrase not in current:
                current.append(phrase)
        else:
            current = [p for p in current if p != phrase]
        self.notes_var.set(", ".join(current))

    def _set_quick_fill_value(self, value: str):
        self.value_var.set(value)

    def _rebuild_quick_fill_buttons(self, column: str):
        """Rebuilds the quick-fill button row for whichever column is now
        active - QUICK_FILL_VALUES has no entry for columns too variable
        for a fixed button list (Name, Age), in which case this simply
        clears out the previous column's buttons. Easy to extend for any
        column with a genuinely common, closed-ish value set - see
        Birthplace's own entry above for a recurring-value example
        beyond the original Sex/Relationship pair."""
        for b in self._quick_fill_buttons:
            b.destroy()
        self._quick_fill_buttons = []
        for value in QUICK_FILL_VALUES.get(column, []):
            # No fixed width (2026-07-26 fix) - a width sized for "Son"/
            # "Head" clipped longer labels like "Husband"/"Daughter"/
            # "Domestic" into unreadable fragments ("Iusban", "aughte").
            # Buttons auto-size to their own label text by default.
            b = Button(self.quick_fill_frame, text=value,
                       command=lambda v=value: self._set_quick_fill_value(v))
            b.pack(side="left", padx=(0, 2))
            self._quick_fill_buttons.append(b)

    def _on_status_change(self):
        """
        Disables/clears the value entry (and quick-fill buttons) for
        illegible/blank statuses - prevents accidentally leaving a stray
        typed value attached to a field marked as unreadable, which would
        silently contradict the status and risk being read later as a
        real value.
        """
        status = self.status_var.get()
        if status in ("illegible", "blank"):
            self.value_var.set("")
            self.value_entry.config(state=DISABLED)
            for b in self._quick_fill_buttons:
                b.config(state=DISABLED)
        else:
            self.value_entry.config(state=NORMAL)
            for b in self._quick_fill_buttons:
                b.config(state=NORMAL)

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