"""
Manual Classification UI - built 2026-08-06 per Jon's spec, for building
a large, human-verified ground-truth set by rapidly hand-classifying
images loaded from an existing bucket. Distinct purpose from every other
review tool in this project:

  - `ui/classifier_validation_ui.py` is READ-ONLY QA (eyeball Gemma's
    decisions, flag triage logs) - never commits a classification.
  - `debug_tools/review_uncertain.py` drains the LIVE uncertain_review.csv
    queue, moving rows between production bucket CSVs.
  - THIS tool: browse any bucket, hand-classify each image against the
    FULL taxonomy tree (not just accept/override one prediction), for
    TRAINING ground truth - deliberately NOT wired into the pipeline
    (no bucket CSV writes, no pipeline_db.py, no manifest.csv). Output
    is a standalone CSV (see OUTPUT_LOG below), kept in a NEW location
    (data/logs/reviewed/) rather than data/outputs/ (where ground_truth_
    log.jsonl already lives) - Jon's explicit call, to avoid this needing
    to move again once the planned folder restructuring actually happens.

TAXONOMY-DRIVEN, two-level (category -> subtype), reusing core/taxonomy.py
directly (Taxonomy.assignable_categories() / subtypes_for()) - a new
config/taxonomy.yaml entry populates buttons with zero code changes here,
same guarantee debug_tools/review_uncertain.py already has.

BUTTON FLOW per Jon's spec:
  - Top level: one button per assignable category, PLUS "Skip" and
    "Needs New Bucket" always present at this level (not nested under
    any category).
  - Clicking a category that has subtypes REDRAWS the button panel to
    show ONLY that category's subtype buttons plus a "Back" button
    (returns to the top-level view without committing anything).
  - Clicking a category with NO subtypes commits immediately (subtype
    left blank) - no pointless empty intermediate screen.
  - Clicking a subtype commits (category + subtype) and advances.

ZOOM/PAN (Jon's spec: "scroll mouse wheel for zoom level, click to drag
image around") - a Canvas, not a Label (Label has no native zoom/pan).
The full-resolution source image is decoded once per row and kept in
memory only for the CURRENTLY-displayed row (same lazy-loading discipline
every other review tool in this project already has); zoom/pan resets to
a fit-to-frame view on every new image, tracked as (zoom_level, pan_x,
pan_y) and re-rendered from the retained full-res PIL image on every
wheel/drag event, not re-decoded from disk each time.

RESUMABILITY - this is meant to run across many sessions building up a
large ground-truth set, unlike review_uncertain.py's "drain a small live
queue" framing. On load, file_paths already present in OUTPUT_LOG are
filtered OUT of the bucket's row list before showing anything, so
reopening the tool never re-shows already-classified images.

Usage:
    python ui/manual_classification_ui.py
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tkinter import Tk, Frame, Label, Button, StringVar, Canvas, DISABLED, NORMAL
from tkinter import ttk

from PIL import Image, ImageTk

from core.classifier import load_pipeline_config, BUCKET_DIR
from core.taxonomy import load_taxonomy, Taxonomy, Category

# NEW location, deliberately not data/outputs/ (see module docstring) -
# standalone training ground truth, never read by any pipeline stage.
OUTPUT_DIR = PROJECT_ROOT / "data" / "logs" / "reviewed"
OUTPUT_LOG = OUTPUT_DIR / "manual_classification_log.csv"
OUTPUT_FIELDS = [
    "timestamp", "bucket", "file_path",
    "category_id", "category_display", "subtype_id", "subtype_display",
    "verdict",  # "classified" | "skip" | "needs_bucket"
]

DEFAULT_BUTTON_COLOUR = "#ddd"
SKIP_COLOUR = "#eee"
NEEDS_BUCKET_COLOUR = "#f5c6a5"
BACK_COLOUR = "#ccc"

MIN_ZOOM = 0.1
MAX_ZOOM = 8.0
ZOOM_STEP = 1.15  # multiplicative per wheel notch


# Extra reviewable sources beyond real production buckets - added
# 2026-08-06 for the tower-consensus disagreement review (see docs/
# MULTI_SOURCE_VOTING_CLASSIFIER_PROPOSAL.md's "56 disagreement cases"
# experiment). Deliberately NOT added as a config/pipeline.yaml bucket
# entry - those entries carry real per-bucket workflow/model config real
# classification code depends on (see that file's own dense_tabular_rows
# comment), and this is a one-off review queue, not a production
# routing destination. Name used as the "bucket" field in OUTPUT_LOG
# rows, so these are clearly distinguishable from real bucket reviews
# later, not silently mixed in.
EXTRA_QUEUES: dict[str, Path] = {
    "tower_consensus_disagreements": OUTPUT_DIR / "disagreement_review_queue.csv",
}


def _load_file_paths_from_csv(path: Path) -> list[str]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return [row["file_path"] for row in csv.DictReader(f) if row.get("file_path")]


def _all_sources() -> dict[str, Path]:
    """Every loadable source for the bucket dropdown: real production
    buckets (name -> data/buckets/<name>.csv) plus EXTRA_QUEUES above."""
    pipeline_cfg = load_pipeline_config()
    sources = {name: BUCKET_DIR / f"{name}.csv" for name in pipeline_cfg["buckets"].keys()}
    sources.update(EXTRA_QUEUES)
    return sources


def _load_already_reviewed_paths() -> set[str]:
    if not OUTPUT_LOG.exists():
        return set()
    with open(OUTPUT_LOG, "r", encoding="utf-8", newline="") as f:
        return {row["file_path"] for row in csv.DictReader(f)}


def _append_output_row(row: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    is_new = not OUTPUT_LOG.exists()
    with open(OUTPUT_LOG, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


class ManualClassificationApp:
    def __init__(self, root: Tk, taxonomy: Taxonomy):
        self.root = root
        self.taxonomy = taxonomy
        root.title("Manual Classification")
        root.geometry("1400x900")
        root.minsize(900, 600)
        # Same real ceiling classifier_validation_ui.py's own bug-fix
        # relies on - see that file's docstring for the growth-loop this
        # prevents; kept here even though the Canvas-based image area
        # below doesn't have a Label's "grows to fit content" failure
        # mode, since it costs nothing and this file otherwise shares
        # the same window-sizing code path.
        root.update_idletasks()
        screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.maxsize(max(1400, screen_w - 80), max(900, screen_h - 80))

        self.rows: list[str] = []  # file_paths not yet reviewed, current bucket
        self.index: int = 0
        self._current_bucket: str | None = None
        self._reviewed_paths: set[str] = _load_already_reviewed_paths()

        # Full-res source image for the CURRENT row only - decoded once
        # per row, re-rendered from this on every zoom/pan event rather
        # than re-reading the file from disk each time.
        self._pil_image: Image.Image | None = None
        self._tk_image: ImageTk.PhotoImage | None = None
        self._zoom: float = 1.0
        self._pan_x: float = 0.0
        self._pan_y: float = 0.0
        self._drag_start: tuple[int, int] | None = None
        self._drag_pan_start: tuple[float, float] | None = None

        # -- Header: bucket selector + position --
        header = Frame(root, padx=12, pady=10)
        header.pack(fill="x")
        Label(header, text="Bucket:", font=("Segoe UI", 9, "bold")).grid(
            row=0, column=0, sticky="w")
        self.bucket_var = StringVar(value="")
        self._bucket_display_to_name: dict[str, str] = {}
        self.bucket_combo = ttk.Combobox(
            header, textvariable=self.bucket_var, state="readonly", width=32)
        self.bucket_combo.grid(row=0, column=1, sticky="w", padx=(6, 0))
        self.bucket_combo.bind("<<ComboboxSelected>>", self._on_bucket_change)

        self.position_var = StringVar(value="Image 0 / 0")
        Label(header, textvariable=self.position_var, font=("Segoe UI", 10, "bold")).grid(
            row=0, column=2, sticky="e", padx=(24, 0))
        header.columnconfigure(2, weight=1)

        self.filename_var = StringVar(value="")
        Label(header, textvariable=self.filename_var, font=("Consolas", 9), fg="#444").grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(4, 0))

        # -- Main split: zoomable canvas (left) / taxonomy buttons (right, edge) --
        main_frame = Frame(root)
        main_frame.pack(fill="both", expand=True, padx=12, pady=(4, 8))

        canvas_frame = Frame(main_frame, bg="#111")
        canvas_frame.pack(side="left", fill="both", expand=True)
        self.canvas = Canvas(canvas_frame, bg="#111", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        # Windows/macOS wheel event; <Button-4>/<Button-5> is the X11
        # equivalent (no <MouseWheel> there) - bound too so this doesn't
        # silently do nothing on Linux.
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", lambda e: self._zoom_at(e.x, e.y, ZOOM_STEP))
        self.canvas.bind("<Button-5>", lambda e: self._zoom_at(e.x, e.y, 1 / ZOOM_STEP))
        self.canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self.canvas.bind("<B1-Motion>", self._on_drag_motion)

        self.button_panel = Frame(main_frame, width=260)
        self.button_panel.pack(side="left", fill="y", padx=(16, 0))
        self.button_panel.pack_propagate(False)

        self._render_top_level_buttons()

        root.update_idletasks()
        self._populate_bucket_list()

    # -- bucket list / selection ------------------------------------------

    def _populate_bucket_list(self) -> None:
        display_values = []
        # EXTRA_QUEUES first (e.g. the disagreement review queue) so the
        # thing Jon's actually working on next isn't buried below 9 real
        # buckets in the dropdown - real buckets still follow, ordered as
        # pipeline.yaml lists them, unchanged.
        for name, path in _all_sources().items():
            paths = _load_file_paths_from_csv(path)
            remaining = sum(1 for p in paths if p not in self._reviewed_paths)
            marker = "⚠ " if name in EXTRA_QUEUES else ""
            display = f"{marker}{name} ({remaining} left of {len(paths)})"
            self._bucket_display_to_name[display] = name
            display_values.append(display)
        display_values = (
            [d for d in display_values if d.startswith("⚠ ")]
            + [d for d in display_values if not d.startswith("⚠ ")]
        )

        self.bucket_combo["values"] = display_values
        if display_values:
            self.bucket_combo.current(0)
            self._on_bucket_change()

    def _on_bucket_change(self, _event=None) -> None:
        display = self.bucket_var.get()
        bucket_name = self._bucket_display_to_name.get(display)
        if bucket_name is None:
            return
        self._current_bucket = bucket_name
        all_paths = _load_file_paths_from_csv(_all_sources()[bucket_name])
        self.rows = [p for p in all_paths if p not in self._reviewed_paths]
        self.index = 0
        self._render_top_level_buttons()
        self._show_current()

    # -- taxonomy button panel ---------------------------------------------

    def _clear_button_panel(self) -> None:
        for child in self.button_panel.winfo_children():
            child.destroy()

    def _render_top_level_buttons(self) -> None:
        """Category buttons + Skip/Needs New Bucket, always present at
        this level per Jon's spec - not nested under any category."""
        self._clear_button_panel()
        Label(self.button_panel, text="Classify as:", font=("Segoe UI", 10, "bold")).pack(
            anchor="w", pady=(0, 6))
        for category in self.taxonomy.assignable_categories():
            Button(
                self.button_panel, text=category.display_name,
                bg=category.color or DEFAULT_BUTTON_COLOUR,
                wraplength=230, justify="left", anchor="w",
                command=lambda c=category: self._on_category_click(c),
            ).pack(fill="x", pady=2)

        Frame(self.button_panel, height=16).pack()  # spacer
        Button(self.button_panel, text="Skip", bg=SKIP_COLOUR,
               command=self._on_skip).pack(fill="x", pady=2)
        Button(self.button_panel, text="Needs New Bucket", bg=NEEDS_BUCKET_COLOUR,
               command=self._on_needs_bucket).pack(fill="x", pady=2)

    def _render_subtype_buttons(self, category: Category) -> None:
        """Redraws the panel to show ONLY this category's subtypes + a
        Back button - per Jon's spec ("redraw the buttons to only show
        the newer ones and a back button")."""
        self._clear_button_panel()
        Label(self.button_panel, text=f"{category.display_name} → subtype:",
              font=("Segoe UI", 10, "bold"), wraplength=230, justify="left").pack(
            anchor="w", pady=(0, 6))
        for subtype in self.taxonomy.subtypes_for(category.id):
            Button(
                self.button_panel, text=subtype.display_name,
                bg=subtype.color or DEFAULT_BUTTON_COLOUR,
                wraplength=230, justify="left", anchor="w",
                command=lambda c=category, s=subtype: self._on_commit(c, s),
            ).pack(fill="x", pady=2)

        Frame(self.button_panel, height=16).pack()  # spacer
        Button(self.button_panel, text="◀ Back", bg=BACK_COLOUR,
               command=self._render_top_level_buttons).pack(fill="x", pady=2)

    def _on_category_click(self, category: Category) -> None:
        subtypes = self.taxonomy.subtypes_for(category.id)
        if subtypes:
            self._render_subtype_buttons(category)
        else:
            self._on_commit(category, None)

    # -- committing a verdict, advancing ------------------------------------

    def _current_file_path(self) -> str | None:
        if 0 <= self.index < len(self.rows):
            return self.rows[self.index]
        return None

    def _write_verdict(self, verdict: str, category: Category | None = None,
                        subtype=None) -> None:
        file_path = self._current_file_path()
        if file_path is None or self._current_bucket is None:
            return
        _append_output_row({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "bucket": self._current_bucket,
            "file_path": file_path,
            "category_id": category.id if category else "",
            "category_display": category.display_name if category else "",
            "subtype_id": subtype.id if subtype else "",
            "subtype_display": subtype.display_name if subtype else "",
            "verdict": verdict,
        })
        self._reviewed_paths.add(file_path)
        self.index += 1
        self._render_top_level_buttons()
        self._show_current()

    def _on_commit(self, category: Category, subtype) -> None:
        self._write_verdict("classified", category, subtype)

    def _on_skip(self) -> None:
        self._write_verdict("skip")

    def _on_needs_bucket(self) -> None:
        self._write_verdict("needs_bucket")

    # -- image display / zoom / pan -----------------------------------------

    def _show_current(self) -> None:
        total = len(self.rows)
        if total == 0:
            self.position_var.set("Image 0 / 0 - queue empty")
            self.filename_var.set("")
            self._pil_image = None
            self.canvas.delete("all")
            return

        self.position_var.set(f"Image {self.index + 1} / {total}")
        file_path = self.rows[self.index]
        self.filename_var.set(file_path)

        try:
            self._pil_image = Image.open(file_path).convert("RGB")
        except Exception as e:
            self._pil_image = None
            self.canvas.delete("all")
            self.canvas.create_text(
                20, 20, anchor="nw", fill="#f66",
                text=f"Could not load image:\n{file_path}\n\n{type(e).__name__}: {e}",
            )
            return

        self._fit_to_frame()
        self._render_canvas()

    def _fit_to_frame(self) -> None:
        """Resets zoom/pan to fit-to-frame - called on every new image,
        per Jon's spec that zoom is a per-image interaction, not
        something that should carry over and surprise the reviewer on
        the next image."""
        if self._pil_image is None:
            return
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        iw, ih = self._pil_image.size
        self._zoom = min(cw / iw, ch / ih, 1.0)  # never upscale past source on fit
        self._pan_x = 0.0
        self._pan_y = 0.0

    def _render_canvas(self) -> None:
        if self._pil_image is None:
            return
        iw, ih = self._pil_image.size
        disp_w, disp_h = max(1, round(iw * self._zoom)), max(1, round(ih * self._zoom))
        # Resample from the ORIGINAL full-res image every render (not a
        # cached scaled copy) - keeps zoom-in quality correct at any
        # level rather than compounding resampling loss across repeated
        # zoom operations.
        scaled = self._pil_image.resize((disp_w, disp_h), Image.LANCZOS)
        self._tk_image = ImageTk.PhotoImage(scaled)

        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        self.canvas.delete("all")
        # Centered by default (pan offset 0,0), shifted by accumulated
        # drag - anchor="center" so pan deltas map directly to screen
        # pixels dragged, matching direct-manipulation expectations.
        self.canvas.create_image(
            cw / 2 + self._pan_x, ch / 2 + self._pan_y,
            anchor="center", image=self._tk_image,
        )

    def _on_canvas_resize(self, _event) -> None:
        if self._pil_image is not None and self.rows:
            # Only re-fit if this is effectively a fresh view (no manual
            # zoom/pan applied yet) - otherwise a window resize mid-review
            # would silently discard the reviewer's zoom/pan state.
            pass
        self._render_canvas()

    def _on_mousewheel(self, event) -> None:
        # Windows delta is +-120 per notch; normalize to one step.
        direction = ZOOM_STEP if event.delta > 0 else 1 / ZOOM_STEP
        self._zoom_at(event.x, event.y, direction)

    def _zoom_at(self, x: int, y: int, factor: float) -> None:
        if self._pil_image is None:
            return
        new_zoom = max(MIN_ZOOM, min(MAX_ZOOM, self._zoom * factor))
        if new_zoom == self._zoom:
            return
        self._zoom = new_zoom
        self._render_canvas()

    def _on_drag_start(self, event) -> None:
        self._drag_start = (event.x, event.y)
        self._drag_pan_start = (self._pan_x, self._pan_y)

    def _on_drag_motion(self, event) -> None:
        if self._drag_start is None or self._drag_pan_start is None:
            return
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        self._pan_x = self._drag_pan_start[0] + dx
        self._pan_y = self._drag_pan_start[1] + dy
        self._render_canvas()


def main() -> None:
    taxonomy = load_taxonomy()
    root = Tk()
    ManualClassificationApp(root, taxonomy)
    root.mainloop()


if __name__ == "__main__":
    main()
