"""
Hint Validation UI - fast Yes/No confirmation pass over auto-extracted
field values, to speed up building the ground-truth set.

Two phases, one continuous run:

  Phase 1 (Validate): for every (row, column) that has a hint from the
  loaded reference CSV (a pre-transcribed source like
  data/automatedgenealogy_pull.csv - see core/csv_hint_source.py for
  the row/page/column matching rules), shows the SAME crop the model
  would see plus that hint text, and asks a single question - does the
  hint match the image? Yes (green) accepts the hint as ground truth
  outright. No (red) queues the field for Phase 2. A field with no CSV
  hint available (no matching page, or that column isn't in the CSV)
  skips straight to the Phase 2 queue, since there's nothing to confirm.

  Phase 2 (Manual entry): once every field has been reviewed, switches
  in-place to the same manual entry controls as
  ui/ground_truth_labeling_ui.py (status radios, value entry, quick-fill,
  notes) for just the fields that were rejected or had no hint - this is
  the "not fully automated yet" fallback Jon asked for, not a separate
  tool to launch later.

Consistent with this project's core principle: the OCR pipeline is a
FINDING AID, not the source of truth. Accepting a hint in Phase 1 is a
human asserting "I checked this against the image and it's correct" -
not a rubber stamp. When in doubt, reject and use Phase 2.

Output: appends to the SAME log as the ground-truth UI,
data/outputs/ground_truth_log.jsonl (JSONL, one record per field),
using the same (sidecar_path, row_index, column) resume/skip key - a
field already labeled by either tool is not re-asked here.

Usage:
    python ui/hint_validation_ui.py
    Then: Load sidecar... / Load columns file... / Load reference CSV...
    (any order - the CSV is re-matched to the sidecar's page whenever
    either changes; see core/csv_hint_source.py for the matching rules,
    and note it still needs a columns file whose names match Jon's new
    10-column census header list before Phase 1 has anything to offer).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tkinter import (
    Tk, Frame, Label, Button, Entry, StringVar, BooleanVar, filedialog, messagebox,
    END, DISABLED, NORMAL, Radiobutton, Checkbutton, Canvas, Scrollbar, HORIZONTAL,
)
from PIL import Image, ImageDraw, ImageTk

from core.row_segmentation import load_sidecar, crop_region_from_source, compute_exclude_ranges
from core.csv_hint_source import load_csv_rows, rows_for_image, get_hint as csv_get_hint
from core import extraction_hint_source

LOG_PATH = PROJECT_ROOT / "data" / "outputs" / "ground_truth_log.jsonl"
PREVIEW_SIZE = (900, 300)
ROW_CONTEXT_HEIGHT = 90

STATUS_OPTIONS = [
    ("readable", "Readable — type the exact value below"),
    ("partially_readable", "Partially readable — type what you can read, mark the rest with ?"),
    ("illegible", "Illegible — genuinely cannot be read (not a guess)"),
    ("blank", "Genuinely blank on the form (not illegible — nothing was written)"),
]

# Same quick-fill set as ground_truth_labeling_ui.py, kept in sync by
# hand since Phase 2 here is meant to feel identical to that tool.
QUICK_FILL_VALUES = {
    "Sex": ["M", "F"],
    "Relationship to Head": [
        "Head", "Wife", "Husband", "Son", "Daughter", "Servant", "Domestic",
        "Boarder", "Lodger", "Mother", "Father", "Brother", "Sister",
    ],
    "Birthplace": ["Ont", "Man", "Ontario", "Manitoba", "Eng", "England", "Scotland", "USA", "Austria", "Ukraine",],
}

NOTE_PHRASES = [
    "ditto mark", "crossed out", "blurry", "faded", "overwritten",
    "stray mark, not a real character", "ink bleed", "faded ink", "enumerator notes",
]


def _load_existing_keys(log_path: Path) -> set[tuple]:
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


class HintValidationApp:
    def __init__(self, root: Tk):
        self.root = root
        root.title("Hint Validation — confirm auto-extracted values against the image")
        root.geometry("1000x800")
        root.minsize(900, 680)

        self.sidecar_path: str | None = None
        self.sidecar: dict | None = None
        self.column_names: list[str] = []
        self.existing_keys: set[tuple] = set()

        # Reference CSV state (core/csv_hint_source.py) - all_rows is
        # every row from the loaded CSV; page_rows is that filtered down
        # to {line_num: csv_row} for whichever page matches the current
        # sidecar's source image, rebuilt whenever either changes.
        self.csv_path: str | None = None
        self.csv_all_rows: list[dict[str, str]] = []
        self.csv_page_rows: dict[int, dict[str, str]] = {}

        # Two-stage extraction results as a hint source (2026-08-15,
        # field_agreement wiring - core/extraction_hint_source.py).
        # When loaded, takes precedence over the CSV as the Phase 1
        # hint source; disagreed fields queue FIRST and display stage
        # 1's independent reading alongside stage 2's value.
        self.extraction_path: str | None = None
        self.extraction_rows: dict[int, dict] = {}

        # Phase 1 queue: (row_index, column, hint, source, agreement,
        # stage1_reading) for fields with a hint to confirm - source is
        # "extraction" or "csv"; agreement/stage1_reading are None for
        # csv-sourced hints. Phase 2 queue: (row_index, column) for fields that
        # were rejected in Phase 1 or never had a hint - built up as
        # Phase 1 runs, then walked once Phase 1 is empty.
        self.review_queue: list[tuple[int, str, str, str, object, object]] = []
        self.review_pos = 0
        self.manual_queue: list[tuple[int, str]] = []
        self.manual_pos = 0
        self.phase = "review"  # "review" | "manual" | "done"

        self._tk_image = None
        self._tk_row_image = None
        self.status_var = StringVar(value="readable")

        # -- load controls ---------------------------------------------------
        top = Frame(root, padx=12)
        top.pack(fill="x", pady=(8, 0))
        load_row = Frame(top)
        load_row.pack(fill="x", pady=4)
        Button(load_row, text="Load sidecar...", command=self.load_sidecar_file).pack(side="left")
        Button(load_row, text="Load columns file...", command=self.load_columns_file).pack(
            side="left", padx=(6, 0))
        Button(load_row, text="Load reference CSV...", command=self.load_csv_file).pack(
            side="left", padx=(6, 0))
        Button(load_row, text="Load extraction JSON...", command=self.load_extraction_file).pack(
            side="left", padx=(6, 0))
        self.sidecar_label = Label(load_row, text="(no sidecar loaded)", fg="#666")
        self.sidecar_label.pack(side="left", padx=10)
        self.csv_label = Label(load_row, text="(no reference CSV loaded)", fg="#666")
        self.csv_label.pack(side="left", padx=10)
        self.extraction_label = Label(load_row, text="(no extraction JSON loaded)", fg="#666")
        self.extraction_label.pack(side="left", padx=10)

        self.phase_label = Label(top, text="", font=("Segoe UI", 10, "bold"), fg="#225")
        self.phase_label.pack(anchor="w", pady=(4, 0))
        self.progress_label = Label(top, text="", font=("Segoe UI", 9, "bold"))
        self.progress_label.pack(anchor="w")
        self.field_label = Label(top, text="", font=("Segoe UI", 13, "bold"), fg="#225")
        self.field_label.pack(anchor="w", pady=(8, 4))

        # -- row context (shared by both phases) ------------------------------
        row_context_frame = Frame(root, padx=12)
        row_context_frame.pack(fill="x", pady=(6, 0))
        Label(row_context_frame,
              text="Full census row (context only - lime box is the field being reviewed):",
              font=("Segoe UI", 8), fg="#666").pack(anchor="w")
        canvas_wrap = Frame(row_context_frame)
        canvas_wrap.pack(fill="x")
        self.row_canvas = Canvas(canvas_wrap, bg="#ddd", height=ROW_CONTEXT_HEIGHT + 4,
                                  width=PREVIEW_SIZE[0], highlightthickness=0)
        row_scroll = Scrollbar(canvas_wrap, orient=HORIZONTAL, command=self.row_canvas.xview)
        self.row_canvas.configure(xscrollcommand=row_scroll.set)
        self.row_canvas.pack(side="top", fill="x")
        row_scroll.pack(side="top", fill="x")

        # -- field crop (shared) ----------------------------------------------
        preview_frame = Frame(root, padx=12)
        preview_frame.pack(fill="x")
        self.image_label = Label(preview_frame, bg="#ddd")
        self.image_label.pack(anchor="w")

        # ==================== PHASE 1: review widgets ========================
        self.review_frame = Frame(root, padx=12, pady=10)
        Label(self.review_frame, text="Auto-extracted hint:",
              font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.hint_label = Label(self.review_frame, text="", font=("Consolas", 16, "bold"),
                                 fg="#003", bg="#eef", anchor="w", padx=10, pady=6)
        self.hint_label.pack(fill="x", pady=(2, 2))
        # field_agreement banner (2026-08-15): green = both independent
        # model reads agreed (fast Yes - measured zero false agreements
        # across every hint-free validation run); red = they disagreed,
        # with stage 1's own reading shown because review is exactly
        # where a correct stage-1/wrong-stage-2 case gets recovered.
        # Empty/hidden for CSV-sourced hints.
        self.agreement_label = Label(self.review_frame, text="", font=("Segoe UI", 10, "bold"),
                                      anchor="w", padx=10, pady=4)
        self.agreement_label.pack(fill="x", pady=(0, 8))

        yn_row = Frame(self.review_frame)
        yn_row.pack(fill="x")
        self.yes_button = Button(
            yn_row, text="✓  Yes, matches (Y / Enter)", command=self.confirm_hint,
            bg="#2a6", fg="white", font=("Segoe UI", 11, "bold"), width=26, height=2,
        )
        self.yes_button.pack(side="left", padx=(0, 8))
        self.no_button = Button(
            yn_row, text="✗  No, reject (N / Backspace)", command=self.reject_hint,
            bg="#c33", fg="white", font=("Segoe UI", 11, "bold"), width=26, height=2,
        )
        self.no_button.pack(side="left")

        # ==================== PHASE 2: manual entry widgets ===================
        self.manual_frame = Frame(root, padx=12, pady=10)

        status_frame = Frame(self.manual_frame)
        status_frame.pack(fill="x", pady=(0, 4))
        Label(status_frame, text="What can you actually read here?",
              font=("Segoe UI", 9, "bold")).pack(anchor="w")
        for value, label in STATUS_OPTIONS:
            Radiobutton(
                status_frame, text=label, variable=self.status_var, value=value,
                command=self._on_status_change,
            ).pack(anchor="w")

        entry_frame = Frame(self.manual_frame)
        entry_frame.pack(fill="x", pady=(6, 4))
        Label(entry_frame, text="Exact value (leave blank if illegible/blank above):",
              font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.value_var = StringVar(value="")
        self.value_entry = Entry(entry_frame, textvariable=self.value_var,
                                  font=("Consolas", 12), width=60)
        self.value_entry.pack(fill="x", expand=True, pady=(2, 0))

        self.quick_fill_frame = Frame(entry_frame)
        self.quick_fill_frame.pack(anchor="w", pady=(4, 0))
        self._quick_fill_buttons: list[Button] = []

        notes_frame = Frame(self.manual_frame)
        notes_frame.pack(fill="x", pady=(6, 4))
        Label(notes_frame, text="Notes (optional):").pack(anchor="w")
        self.notes_var = StringVar(value="")
        Entry(notes_frame, textvariable=self.notes_var, width=80).pack(anchor="w", pady=(2, 0))
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

        manual_nav = Frame(self.manual_frame, pady=6)
        manual_nav.pack(fill="x")
        self.save_next_button = Button(
            manual_nav, text="Save & Next →", command=self.save_manual_and_next,
            bg="#2a6", fg="white", font=("Segoe UI", 10, "bold"), state=DISABLED,
        )
        self.save_next_button.pack(side="left")
        Button(manual_nav, text="Skip (don't save, come back later)",
               command=self.skip_manual).pack(side="left", padx=(8, 0))

        self.status_label = Label(root, text="", fg="#444", padx=12)
        self.status_label.pack(anchor="w")

        # Keybindings - reused across both phases; harmless when the
        # inactive phase's frame isn't packed.
        root.bind("y", lambda e: self.phase == "review" and self.confirm_hint())
        root.bind("Y", lambda e: self.phase == "review" and self.confirm_hint())
        root.bind("n", lambda e: self.phase == "review" and self.reject_hint())
        root.bind("N", lambda e: self.phase == "review" and self.reject_hint())
        root.bind("<Return>", self._on_return_key)
        self.value_entry.bind("<Return>", lambda e: (self.save_manual_and_next(), "break"))

        self._on_status_change()

    # -- loading --------------------------------------------------------------

    def load_sidecar_file(self):
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
        self._refresh_csv_page_rows()
        if self.column_names:
            self._build_queues()

    def load_csv_file(self):
        # Fixed default (matching Load sidecar's own fix above) - the
        # reference CSVs this project pulls in land in data/, same as
        # data/automatedgenealogy_pull.csv.
        data_dir = PROJECT_ROOT / "data"
        path = filedialog.askopenfilename(
            title="Select reference transcription CSV",
            filetypes=[("CSV", "*.csv")],
            initialdir=str(data_dir) if data_dir.is_dir() else None,
        )
        if not path:
            return
        try:
            rows = load_csv_rows(path)
        except Exception as e:
            messagebox.showerror("Could not load CSV", str(e))
            return
        if not rows:
            messagebox.showerror("Empty CSV", "Reference CSV contained no rows.")
            return
        self.csv_path = path
        self.csv_all_rows = rows
        self.csv_label.config(text=Path(path).name, fg="black")
        self._refresh_csv_page_rows()
        if self.column_names:
            self._build_queues()

    def _refresh_csv_page_rows(self):
        """
        Re-filters the loaded CSV down to the page matching the current
        sidecar's source image (core.csv_hint_source.rows_for_image) -
        call this whenever either the sidecar or the CSV changes. Left
        empty (not an error) if no CSV is loaded yet or no page in it
        matches this sidecar's image - Phase 1 simply has nothing to
        offer and every field falls through to Phase 2.
        """
        if self.sidecar is None or not self.csv_all_rows:
            self.csv_page_rows = {}
            return
        self.csv_page_rows = rows_for_image(
            self.csv_all_rows, self.sidecar["source_image_path"])

    def load_extraction_file(self):
        """Loads a two-stage extraction output JSON (scripts/
        run_two_stage_extraction.py --out) as the Phase 1 hint source -
        core/extraction_hint_source.py. Sanity-checks row bboxes
        against the loaded sidecar so a mispaired file fails loudly."""
        path = filedialog.askopenfilename(
            title="Select two-stage extraction output JSON",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        try:
            rows = extraction_hint_source.load_extraction_results(path)
        except Exception as e:
            messagebox.showerror("Could not load extraction JSON", str(e))
            return
        if self.sidecar is not None:
            warning = extraction_hint_source.match_check(rows, self.sidecar)
            if warning:
                if not messagebox.askyesno(
                        "Possible mismatch",
                        warning + "\n\nLoad it anyway?"):
                    return
        self.extraction_path = path
        self.extraction_rows = rows
        self.extraction_label.config(text=Path(path).name, fg="black")
        if self.sidecar is not None and self.column_names:
            self._build_queues()

    def load_columns_file(self):
        columns_dir = PROJECT_ROOT / "config" / "columns"
        path = filedialog.askopenfilename(
            title="Select column names file",
            filetypes=[("Text", "*.txt")],
            initialdir=str(columns_dir) if columns_dir.is_dir() else None,
        )
        if not path:
            return
        # '#' prefix (Jon, convention introduced 2026-07-29) marks a
        # column that exists on the sidecar/form but should be skipped
        # entirely here - either it isn't captured by the transcription
        # at all, or (per Jon: the Address column) the transcription DOES
        # have data in that slot but it's a condensed/re-numbered field
        # that doesn't actually correspond to what's really in that form
        # column, so treating it as a hint source would be actively
        # wrong, not just unavailable. These lines are dropped entirely,
        # not passed through as literal (and therefore unmatched) column
        # names.
        names = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()
                 if line.strip() and not line.strip().startswith("#")]
        if not names:
            messagebox.showerror("Empty file", "Column file contained no usable (non-#) names.")
            return
        self.column_names = names
        if self.sidecar is not None:
            self._build_queues()

    def _build_queues(self):
        """
        Splits every not-yet-labeled (row, column) pair into the Phase 1
        review queue (has a hint to confirm) or straight into the Phase
        2 manual queue (no hint exists yet, nothing to validate).
        """
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.existing_keys = _load_existing_keys(LOG_PATH)

        self.review_queue = []
        self.manual_queue = []
        disagreed: list[tuple] = []
        agreed: list[tuple] = []
        csv_hinted: list[tuple] = []
        for row in self.sidecar["rows"]:
            # Row matching is a direct row_index == line_num join (Jon,
            # 2026-07-29) - the sidecar's row index is assigned by
            # walking the table the same way the CSV's line_num was
            # transcribed, so no fuzzy matching is needed here. The
            # extraction JSON shares the same row_index space by
            # construction (it was produced FROM a sidecar).
            csv_row = self.csv_page_rows.get(row["index"])
            ext_row = self.extraction_rows.get(row["index"])
            for col in self.column_names:
                key = (self.sidecar_path, row["index"], col)
                if key in self.existing_keys:
                    continue
                # Extraction hints take precedence over CSV when both
                # are loaded - reviewing a fresh extraction is the live
                # use-case; the CSV remains the fallback source.
                ext_hint = extraction_hint_source.get_hint(ext_row, col)
                if ext_hint is not None:
                    agree = extraction_hint_source.agreement(ext_row, col)
                    s1 = extraction_hint_source.stage1_reading(ext_row, col)
                    entry = (row["index"], col, ext_hint, "extraction", agree, s1)
                    (agreed if agree else disagreed).append(entry)
                    continue
                hint = csv_get_hint(csv_row, col) if csv_row is not None else None
                if hint is not None:
                    csv_hinted.append((row["index"], col, hint, "csv", None, None))
                else:
                    self.manual_queue.append((row["index"], col))
        # Disagreements FIRST (the fields the review queue exists for),
        # then agreed fast-confirms, then CSV-sourced hints.
        self.review_queue = disagreed + agreed + csv_hinted
        self.review_pos = 0
        self.manual_pos = 0

        total_possible = len(self.sidecar["rows"]) * len(self.column_names)
        already_done = total_possible - len(self.review_queue) - len(self.manual_queue)
        csv_note = "" if self.csv_page_rows else " (no matching CSV page found for this sidecar's image)"
        n_disagreed = sum(1 for e in self.review_queue if e[3] == "extraction" and not e[4])
        disagree_note = f" ({n_disagreed} model DISAGREEMENTS queued first)" if n_disagreed else ""
        self.status_label.config(
            text=f"{already_done}/{total_possible} already labeled. "
                 f"{len(self.review_queue)} to confirm{disagree_note}, "
                 f"{len(self.manual_queue)} need manual entry (no hint available){csv_note}.")

        if not self.review_queue and not self.manual_queue:
            messagebox.showinfo("Nothing to do",
                                 "Every (row, column) pair in this sidecar with these "
                                 "columns has already been labeled.")
            self.phase = "done"
            self._show_current()
            return

        self.phase = "review" if self.review_queue else "manual"
        self._show_current()

    # -- display ----------------------------------------------------------

    def _show_current(self):
        if self.phase == "review":
            self.manual_frame.pack_forget()
            self.review_frame.pack(fill="x")
            self._show_review_current()
        elif self.phase == "manual":
            self.review_frame.pack_forget()
            self.manual_frame.pack(fill="both", expand=True)
            self._show_manual_current()
        else:
            self.review_frame.pack_forget()
            self.manual_frame.pack_forget()
            self.phase_label.config(text="All fields reviewed for this sidecar.")
            self.progress_label.config(text="")
            self.field_label.config(text="")
            self.image_label.config(image="", text="(done)")

    def _current_row(self, row_index: int) -> dict:
        return next(r for r in self.sidecar["rows"] if r["index"] == row_index)

    def _column_mask(self, column: str):
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
        return keep_ranges, row_masks

    def _render_field(self, row_index: int, column: str):
        """Shared crop + row-context rendering for whichever field is
        currently active, in either phase - identical logic to
        ground_truth_labeling_ui.py so the crop is guaranteed to match
        what the model was actually shown."""
        row = self._current_row(row_index)
        source_path = self.sidecar["source_image_path"]
        deskew_angle = self.sidecar["deskew_angle"]
        keep_ranges, row_masks = self._column_mask(column)

        try:
            crop = crop_region_from_source(source_path, row["bbox"], deskew_angle, row_masks)
        except Exception as e:
            messagebox.showerror("Could not load crop", str(e))
            return

        self._render_row_context(row, source_path, deskew_angle, keep_ranges)

        preview = crop.copy()
        if keep_ranges:
            crop_x0 = row["bbox"][0]
            local_ranges = [(x0 - crop_x0, x1 - crop_x0) for x0, x1 in keep_ranges]
            pad = 20
            left = max(0, min(r[0] for r in local_ranges) - pad)
            right = min(preview.width, max(r[1] for r in local_ranges) + pad)
            if right > left:
                preview = preview.crop((left, 0, right, preview.height))

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

    def _render_row_context(self, row: dict, source_path: str, deskew_angle: float,
                             keep_ranges: list[tuple[int, int]]):
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

    # -- Phase 1: review ----------------------------------------------------

    def _show_review_current(self):
        if self.review_pos >= len(self.review_queue):
            # Review queue exhausted - fall through to manual phase (may
            # itself be empty, in which case _show_current handles "done").
            self.phase = "manual" if self.manual_queue else "done"
            self._show_current()
            return

        row_index, column, hint, source, agree, s1 = self.review_queue[self.review_pos]
        self.phase_label.config(text="Phase 1 of 2 — Confirm auto-extracted hints against the image")
        self.progress_label.config(
            text=f"Field {self.review_pos + 1} of {len(self.review_queue)} to confirm "
                 f"({len(self.manual_queue)} queued for manual entry so far)")
        self.field_label.config(text=f"{column}  —  Row {row_index}")
        self.hint_label.config(text=hint if hint else "(empty value)")
        if source == "extraction":
            if agree:
                self.agreement_label.config(
                    text="✓ both independent model reads AGREE",
                    fg="#0a5c1f", bg="#e4f5e7")
            else:
                s1_display = s1 if (s1 or "").strip() else "(no stage-1 reading)"
                self.agreement_label.config(
                    text=f"✗ MODELS DISAGREE — stage 1 independently read: {s1_display}",
                    fg="#8a1621", bg="#fbe6e8")
        else:
            self.agreement_label.config(text="(reference CSV hint)", fg="#666", bg=self.review_frame.cget("bg"))
        self._render_field(row_index, column)

    def confirm_hint(self):
        if self.phase != "review" or self.review_pos >= len(self.review_queue):
            return
        row_index, column, hint, source, agree, _s1 = self.review_queue[self.review_pos]
        if source == "extraction":
            note = ("confirmed_from_extraction_agreed" if agree
                    else "confirmed_from_extraction_disagreed")
        else:
            note = "confirmed_from_reference_csv"
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sidecar_path": self.sidecar_path,
            "source_image_path": self.sidecar["source_image_path"],
            "row_index": row_index,
            "column": column,
            "status": "readable",
            "value": hint,
            "notes": note,
        }
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.existing_keys.add((self.sidecar_path, row_index, column))
        self.review_pos += 1
        self._show_review_current()

    def reject_hint(self):
        if self.phase != "review" or self.review_pos >= len(self.review_queue):
            return
        row_index, column, _hint, _source, _agree, _s1 = self.review_queue[self.review_pos]
        self.manual_queue.append((row_index, column))
        self.review_pos += 1
        self._show_review_current()

    def _on_return_key(self, event):
        # Return in the review phase = Yes, unless focus is in the value
        # entry (Phase 2), which has its own dedicated binding above.
        if self.phase == "review" and self.root.focus_get() is not self.value_entry:
            self.confirm_hint()

    # -- Phase 2: manual entry (mirrors ground_truth_labeling_ui.py) --------

    def _show_manual_current(self):
        if self.manual_pos >= len(self.manual_queue):
            self.phase = "done"
            self._show_current()
            return

        row_index, column = self.manual_queue[self.manual_pos]
        self.phase_label.config(text="Phase 2 of 2 — Manual entry (rejected or no auto hint)")
        self.progress_label.config(
            text=f"Field {self.manual_pos + 1} of {len(self.manual_queue)} remaining")
        self.field_label.config(text=f"{column}  —  Row {row_index}")
        self._render_field(row_index, column)

        self.status_var.set("readable")
        self.value_var.set("")
        self.notes_var.set("")
        for var in self.note_phrase_vars.values():
            var.set(False)
        self._rebuild_quick_fill_buttons(column)
        self._on_status_change()
        self.save_next_button.config(state=NORMAL)
        self.value_entry.focus_set()

    def _toggle_note_phrase(self, phrase: str, var: "BooleanVar"):
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
        for b in self._quick_fill_buttons:
            b.destroy()
        self._quick_fill_buttons = []
        for value in QUICK_FILL_VALUES.get(column, []):
            b = Button(self.quick_fill_frame, text=value,
                       command=lambda v=value: self._set_quick_fill_value(v))
            b.pack(side="left", padx=(0, 2))
            self._quick_fill_buttons.append(b)

    def _on_status_change(self):
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

    def save_manual_and_next(self):
        if self.phase != "manual" or self.manual_pos >= len(self.manual_queue):
            return
        row_index, column = self.manual_queue[self.manual_pos]
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
        self.manual_pos += 1
        self._show_manual_current()

    def skip_manual(self):
        self.manual_pos += 1
        self._show_manual_current()


def main():
    root = Tk()
    HintValidationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
