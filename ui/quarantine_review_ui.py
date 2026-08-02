"""
Quarantine review UI - manual geometry correction for pages that failed
automated validation in core/auto_sidecar.py, per Jon's 2026-07-28
"Quarantine UI Design Scope" spec and 2026-07-28 follow-up direction.

Purpose: NOT an image editor. Its only job is confirming/correcting page
GEOMETRY (row bounds + column edges) for pages the automated pipeline
routed to manual review, so every sidecar reaching the OCR pipeline -
automatic or manually corrected - is geometrically verified. Does not
care WHY a page was quarantined - it only reads status == "needs_review"
(whole-page: rows_needs_review non-empty with no surviving rows; or any
column flagged needs_review) and queues it.

Two review workflows, chosen automatically from the page's OWN
template.row_strategy (already known - the doc type was classified
before this page ever reached quarantine, so there's no reason to ask
again):

  - "fixed_periodic" (census forms, regular row spacing): click row 1's
    top edge and the LAST row's bottom edge, then table left/right - the
    span divides evenly by the template's own expected_row_count (from
    its yaml), tiled by pure arithmetic. No per-row clicking, matching
    the original spec's "one question at a time, <10-20s/page" goal -
    appropriate ONLY because these forms have a known, fixed row count.

  - "detect" (passenger manifests, and - per Jon's 2026-07-28 note -
    occasional UK census pages with irregular spacing that the
    autodetect may misclassify into this bucket): row spacing isn't
    regular and row COUNT isn't known ahead of time, so arithmetic
    tiling doesn't apply. Instead: click table left/right, then click
    each row's top and bottom edge one at a time until "Finish Rows" is
    pressed - the direct, no-guessing equivalent for irregular pages.

Verification-before-accept (2026-07-28, per Jon's direction: "the
output from this should go back through the CV pipeline to verify
before being accepted... if confidence is still below threshold it gets
reviewed for further refinement"): after a page's rows are manually
defined (either workflow), row heights are run back through
core/auto_sidecar.py's OWN quarantine-rate gate (_find_quarantine_
positions + PAGE_QUARANTINE_THRESHOLD - the exact same check the
automated pipeline itself uses to decide "is this trustworthy",
not a new invented threshold). If the rate is still at/above threshold,
the page is NOT accepted - it's reset for another pass at defining rows,
same as the automated pipeline would refuse to trust its own
low-confidence output. Below threshold: any still-anomalous individual
rows are flagged into rows_needs_review (normal, per the "quarantine
some rows, keep the rest" philosophy) and the page is saved and
accepted.

Reuses core/row_segmentation.py's own save/load helpers (save_sidecar,
load_sidecar) for I/O, and core/auto_sidecar.py's existing
locate_header_region() for the header/metadata boundary. Never touches
ui/row_segmentation_ui.py, core/classifier.py, or
core/document_classification.py - separate, new file entirely, per
CLAUDE.md's explicit preservation list.

Column masking here writes a SINGLE keep-range per column (left click,
right click) - by design, matching core/auto_sidecar.py's own
locate_columns() single-range assumption. A column needing a more
complex multi-range mask still goes through the full manual tool
(ui/row_segmentation_ui.py) afterward, same as any other column.

Usage:
    python ui/quarantine_review_ui.py [--sidecar-dir data/outputs/auto_row_segmentation]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tkinter import Tk, Frame, Label, Button, Canvas, Scrollbar, messagebox

from PIL import Image, ImageDraw, ImageTk

from core.auto_sidecar import (
    PAGE_QUARANTINE_THRESHOLD,
    _DETECT_MAX_HEIGHT_RATIO,
    _DETECT_MIN_HEIGHT_RATIO,
    _PERIODIC_MAX_HEIGHT_RATIO,
    _PERIODIC_MIN_HEIGHT_RATIO,
    _find_quarantine_positions,
    _quarantine_anomalous_rows,
    locate_header_region,
)
from core.document_templates import DocumentTemplate, load_template
from core.row_segmentation import (
    init_column_state,
    load_sidecar,
    save_sidecar,
    segment_rows_uniform_tile,
)

DEFAULT_SIDECAR_DIR = PROJECT_ROOT / "data" / "outputs" / "auto_row_segmentation"
DEWARPED_DIR = PROJECT_ROOT / "data" / "outputs" / "dewarped"
COLUMNS_DIR = PROJECT_ROOT / "config" / "columns"

# Magnifier loupe (2026-07-28, per Jon's direction: the downscaled full-
# page preview isn't sharp enough to read column-header text or place
# row/column edges precisely - "columns i had to guess from locations
# memorized from other census pages"). Samples a small crop straight
# from the FULL-RESOLUTION source image (not the already-downscaled
# preview) around the cursor, zooms it up, and masks it into a circle -
# a real magnifying glass, not just a bigger preview.
LOUPE_DIAMETER = 260   # on-screen pixels
LOUPE_ZOOM = 5         # magnification factor applied to the source crop

# Per Jon's 2026-07-28 direction: column masking here must use the SAME
# curated column list ui/row_segmentation_ui.py's "load columns file"
# feature already draws from (config/columns/*.txt, one name per line,
# in the file's own order) - NOT core/auto_sidecar.py's
# template.expected_columns, which lists every column the CV detector
# can locate (e.g. every census template's "Sex", already known to be
# the one hard-to-locate column - see auto_sidecar.py's
# _COLUMN_WIDTH_PLAUSIBLE_RATIO_RANGE comment). Only the fields actually
# wanted in the sidecar, in the desired OUTPUT order (not left-to-right
# scan order - config/columns/census_core_fields.txt reorders "Age"
# ahead of "Sex"/"Relationship to Head" relative to the physical form),
# should ever be offered for manual masking here.
#
# 2026-07-29: the doc_type -> filename mapping itself used to live here
# as a hardcoded dict (DEFAULT_COLUMNS_FILE_BY_DOC_TYPE) - the one part
# of this pipeline that wasn't pure config. Moved to each template's
# own columns_file field (config/document_templates/*.yaml, see
# core/document_templates.py's DocumentTemplate.columns_file comment)
# so a new doc_type no longer needs a Python edit here.


def load_column_list(template: DocumentTemplate, override_path: str | None = None) -> list[str] | None:
    """
    Returns the ordered column names to mask for this template, or None
    if it has no columns_file set (and no CLI override) - the caller
    falls back to template.expected_columns (the full CV-detectable set)
    in that case, with a visible warning, rather than silently guessing
    a file name.
    """
    if override_path:
        path = Path(override_path)
    else:
        if template.columns_file is None:
            return None
        path = COLUMNS_DIR / template.columns_file
    if not path.exists():
        return None
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return names or None


def find_quarantined_sidecars(sidecar_dir: Path) -> list[Path]:
    """
    A page qualifies for this UI if it hits one of the two triggers this
    workflow actually handles - whole-page quarantine (every row moved
    to rows_needs_review, none left in "rows") or a column flagged
    needs_review by core/auto_sidecar.py's width-plausibility check.
    Partial per-row quarantine (some rows kept, a few flagged) is
    deliberately OUT of scope here - that's an individual-row issue, not
    a page-geometry problem, and doesn't need a full row/column redo.
    """
    candidates = []
    for path in sorted(sidecar_dir.glob("*_sidecar.json")):
        try:
            sidecar = load_sidecar(path)
        except Exception:
            continue
        whole_page_quarantined = (
            not sidecar.get("rows") and sidecar.get("rows_needs_review")
        )
        flagged_columns = [
            name for name, state in sidecar.get("columns", {}).items()
            if state.get("status") == "needs_review"
        ]
        if whole_page_quarantined or flagged_columns:
            candidates.append(path)
    return candidates


def find_source_image(sidecar: dict, stem: str) -> Path | None:
    recorded = sidecar.get("source_image_path")
    if recorded and Path(recorded).exists():
        return Path(recorded)
    matches = list(DEWARPED_DIR.glob(f"{stem}_dewarped.*"))
    return matches[0] if matches else None


class QuarantineReviewApp:
    """
    Per-page phase machine: "bounds" (table left/right, + for fixed_
    periodic templates also row-1-top/last-row-bottom) -> "rows" (detect-
    strategy templates only: click each row's top/bottom one at a time)
    -> "columns" (one column at a time, left/right edge) -> "done"
    (Accept enabled). Undo only steps backward within the current phase
    (a no-op at a phase's very first click) - crossing back into an
    earlier phase isn't needed in practice, since a bad early click is
    caught immediately by the live overlay, well before later clicks
    depend on it.
    """

    def __init__(self, root: Tk, sidecar_paths: list[Path], columns_file_override: str | None = None):
        self.root = root
        self.root.title("Quarantine Review")
        self.sidecar_paths = sidecar_paths
        self.page_index = 0
        self.columns_file_override = columns_file_override

        self.image: Image.Image | None = None
        self.template = None
        self.sidecar: dict | None = None
        self.mode = "fixed"  # "fixed" or "detect", set per page from template.row_strategy
        self.active_columns: list[str] = []  # the curated, ordered column list for THIS page
        self._preview_scale = 1.0

        top = Frame(root)
        top.pack(fill="x", padx=8, pady=4)
        self.progress_label = Label(top, text="", font=("Segoe UI", 11, "bold"), anchor="w")
        self.progress_label.pack(side="left")

        prompt = Frame(root)
        prompt.pack(fill="x", padx=8)
        self.prompt_label = Label(prompt, text="", font=("Segoe UI", 13), fg="#0a4")
        self.prompt_label.pack(side="left")

        canvas_frame = Frame(root)
        canvas_frame.pack(fill="both", expand=True, padx=8, pady=4)
        yscroll = Scrollbar(canvas_frame, orient="vertical")
        xscroll = Scrollbar(canvas_frame, orient="horizontal")
        self.canvas = Canvas(
            canvas_frame, bg="#222",
            yscrollcommand=yscroll.set, xscrollcommand=xscroll.set,
        )
        yscroll.config(command=self.canvas.yview)
        xscroll.config(command=self.canvas.xview)
        yscroll.pack(side="right", fill="y")
        xscroll.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", self._on_leave)
        self._loupe_photo = None

        controls = Frame(root)
        controls.pack(fill="x", padx=8, pady=8)
        Button(controls, text="Undo", width=12, command=self.undo).pack(side="left", padx=4)
        Button(controls, text="Skip Page", width=12, command=self.skip_page).pack(side="left", padx=4)
        self.finish_rows_button = Button(
            controls, text="Finish Rows", width=14, command=self.finish_rows, state="disabled")
        self.finish_rows_button.pack(side="left", padx=4)
        self.accept_button = Button(controls, text="Accept", width=12, command=self.accept_page, state="disabled")
        self.accept_button.pack(side="left", padx=4)

        self._load_page()

    # ---------------------------------------------------------- page load

    def _load_page(self):
        if self.page_index >= len(self.sidecar_paths):
            messagebox.showinfo("Quarantine Review", "No more quarantined pages - all done.")
            self.root.quit()
            return

        path = self.sidecar_paths[self.page_index]
        self.sidecar_path = path
        self.sidecar = load_sidecar(path)
        stem = path.name.removesuffix("_sidecar.json")

        image_path = find_source_image(self.sidecar, stem)
        if image_path is None:
            messagebox.showwarning(
                "Quarantine Review",
                f"Could not find the source image for {stem} - skipping.",
            )
            self.page_index += 1
            self._load_page()
            return

        doc_type = self.sidecar.get("parameters", {}).get("doc_type")
        try:
            self.template = load_template(doc_type)
        except FileNotFoundError as e:
            messagebox.showwarning("Quarantine Review", f"{stem}: {e} - skipping.")
            self.page_index += 1
            self._load_page()
            return

        self.image = Image.open(image_path).convert("RGB")
        self.mode = "fixed" if self.template.row_strategy == "fixed_periodic" else "detect"

        columns = load_column_list(self.template, override_path=self.columns_file_override)
        if columns is None:
            columns = self.template.expected_columns
            print(f"  [{stem}] no config/columns/*.txt mapping for doc_type={doc_type!r} - "
                  f"falling back to the template's full expected_columns list.")
        self.active_columns = columns
        self.column_steps = [
            step for col in self.active_columns for step in (f"col:{col}:left", f"col:{col}:right")
        ]
        self._reset_state()
        self._render()

    def _reset_state(self):
        """Blank slate for the current page's geometry - used on first load and when a manual pass fails re-verification."""
        if self.mode == "fixed":
            self.bounds_steps = ["row1_top", "last_row_bottom", "table_left", "table_right"]
        else:
            self.bounds_steps = ["table_left", "table_right"]
        self.bounds_step_index = 0
        self.bounds = {}
        self.manual_rows: list[tuple[int, int]] = []   # detect mode only, sorted top-to-bottom
        self.pending_row_top = None                     # detect mode only
        self.col_step_index = 0
        self.col_clicks: dict[str, int] = {}
        self.phase = "bounds"
        self.accept_button.config(state="disabled")
        self.finish_rows_button.config(state="disabled")

    # --------------------------------------------------------- rendering

    def _render(self):
        w, h = self.image.size
        max_display_w = 1500
        scale = min(1.0, max_display_w / w)
        self._preview_scale = scale
        disp_w, disp_h = int(w * scale), int(h * scale)

        overlay = self.image.copy()
        draw = ImageDraw.Draw(overlay)

        l = self.bounds.get("table_left")
        r = self.bounds.get("table_right")
        if l is not None:
            draw.line([(l, 0), (l, h)], fill="cyan", width=3)
        if r is not None:
            draw.line([(r, 0), (r, h)], fill="cyan", width=3)

        if self.mode == "fixed":
            t = self.bounds.get("row1_top")
            b = self.bounds.get("last_row_bottom")
            if t is not None:
                draw.line([(0, t), (w, t)], fill="lime", width=3)
            if b is not None:
                draw.line([(0, b), (w, b)], fill="lime", width=3)
            if None not in (t, b, l, r):
                draw.rectangle([l, t, r, b], outline="yellow", width=2)
                n = self.template.expected_row_count
                row_h = (b - t) / n
                for i in range(1, n):
                    y = int(t + i * row_h)
                    draw.line([(l, y), (r, y)], fill=(255, 255, 0, 128), width=1)
        else:
            for i, (rt, rb) in enumerate(self.manual_rows):
                x0 = l if l is not None else 0
                x1 = r if r is not None else w
                draw.rectangle([x0, rt, x1, rb], outline="yellow", width=2)
                draw.text((x0 + 4, rt + 2), str(i + 1), fill="yellow")
            if self.pending_row_top is not None:
                draw.line([(0, self.pending_row_top), (w, self.pending_row_top)], fill="lime", width=3)

        if self.phase == "columns" or self.phase == "done":
            top = self._rows_top()
            bot = self._rows_bottom()
            for col in self.active_columns:
                cl = self.col_clicks.get(f"col:{col}:left")
                cr = self.col_clicks.get(f"col:{col}:right")
                if cl is not None:
                    draw.line([(cl, top), (cl, bot)], fill="magenta", width=2)
                if cr is not None:
                    draw.line([(cr, top), (cr, bot)], fill="magenta", width=2)
                if cl is not None and cr is not None:
                    draw.rectangle([cl, top, cr, bot], outline="orange", width=1)

        display_img = overlay.resize((disp_w, disp_h), Image.LANCZOS) if scale != 1.0 else overlay
        self._photo = ImageTk.PhotoImage(display_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        self.canvas.config(scrollregion=(0, 0, disp_w, disp_h))

        self._update_prompt()

    def _rows_top(self) -> int:
        if self.mode == "fixed":
            return self.bounds.get("row1_top", 0)
        return self.manual_rows[0][0] if self.manual_rows else 0

    def _rows_bottom(self) -> int:
        if self.mode == "fixed":
            return self.bounds.get("last_row_bottom", self.image.height)
        return self.manual_rows[-1][1] if self.manual_rows else self.image.height

    def _update_prompt(self):
        stem = self.sidecar_path.name.removesuffix("_sidecar.json")
        mode_label = "fixed-spacing" if self.mode == "fixed" else "irregular-spacing (row-by-row)"
        self.progress_label.config(
            text=f"Page {self.page_index + 1} of {len(self.sidecar_paths)}   |   {stem}   |   "
                 f"{mode_label}   |   Phase: {self.phase}"
        )

        if self.phase == "bounds":
            step = self.bounds_steps[self.bounds_step_index]
            prompts = {
                "row1_top": f"Click the TOP of row 1 (of {self.template.expected_row_count} rows)",
                "last_row_bottom": f"Click the BOTTOM of row {self.template.expected_row_count} (the last row)",
                "table_left": "Click the LEFT edge of the table",
                "table_right": "Click the RIGHT edge of the table",
            }
            self.prompt_label.config(text=prompts[step])
        elif self.phase == "rows":
            n = len(self.manual_rows) + 1
            if self.pending_row_top is None:
                self.prompt_label.config(text=f"Row {n}: click the TOP edge")
            else:
                self.prompt_label.config(text=f"Row {n}: click the BOTTOM edge")
            self.finish_rows_button.config(
                state="normal" if (self.pending_row_top is None and self.manual_rows) else "disabled")
        elif self.phase == "columns":
            step = self.column_steps[self.col_step_index]
            _, col, side = step.split(":")
            self.prompt_label.config(text=f"Column: {col}   ->   click the {side.upper()} edge")
        else:
            self.prompt_label.config(text="All geometry defined - click Accept, or Undo to correct.")
            self.accept_button.config(state="normal")

    # ------------------------------------------------------------ clicks

    def _on_click(self, event):
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        image_x = int(canvas_x / self._preview_scale)
        image_y = int(canvas_y / self._preview_scale)

        if self.phase == "bounds":
            step = self.bounds_steps[self.bounds_step_index]
            value = image_y if step in ("row1_top", "last_row_bottom") else image_x
            self.bounds[step] = value
            self.bounds_step_index += 1
            if self.bounds_step_index >= len(self.bounds_steps):
                self.phase = "rows" if self.mode == "detect" else "columns"
        elif self.phase == "rows":
            if self.pending_row_top is None:
                self.pending_row_top = image_y
            else:
                top, bottom = self.pending_row_top, image_y
                if bottom < top:
                    top, bottom = bottom, top
                self.manual_rows.append((top, bottom))
                self.manual_rows.sort()
                self.pending_row_top = None
        elif self.phase == "columns":
            step = self.column_steps[self.col_step_index]
            self.col_clicks[step] = image_x
            self.col_step_index += 1
            if self.col_step_index >= len(self.column_steps):
                self.phase = "done"
        else:
            return
        self._render()

    def _on_motion(self, event):
        if self.image is None or self.phase == "done":
            self.canvas.delete("loupe")
            return

        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        image_x = canvas_x / self._preview_scale
        image_y = canvas_y / self._preview_scale

        w, h = self.image.size
        src_half = (LOUPE_DIAMETER / LOUPE_ZOOM) / 2
        box = (image_x - src_half, image_y - src_half, image_x + src_half, image_y + src_half)
        # PIL crop() handles a box that runs off the image edges by
        # padding with black - acceptable near page borders, and avoids
        # a separate clamping path for a rare edge case.
        crop = self.image.crop((int(box[0]), int(box[1]), int(box[2]), int(box[3])))
        zoomed = crop.resize((LOUPE_DIAMETER, LOUPE_DIAMETER), Image.LANCZOS).convert("RGBA")

        mask = Image.new("L", (LOUPE_DIAMETER, LOUPE_DIAMETER), 0)
        ImageDraw.Draw(mask).ellipse([0, 0, LOUPE_DIAMETER - 1, LOUPE_DIAMETER - 1], fill=255)
        zoomed.putalpha(mask)

        draw = ImageDraw.Draw(zoomed)
        draw.ellipse([1, 1, LOUPE_DIAMETER - 2, LOUPE_DIAMETER - 2], outline=(255, 255, 0, 255), width=3)
        c = LOUPE_DIAMETER // 2
        draw.line([(c - 12, c), (c + 12, c)], fill=(255, 0, 0, 255), width=2)
        draw.line([(c, c - 12), (c, c + 12)], fill=(255, 0, 0, 255), width=2)

        self._loupe_photo = ImageTk.PhotoImage(zoomed)
        # Offset up-and-right of the cursor by default so the loupe
        # itself never covers the exact point being aimed at - but
        # column headers sit near the TOP of the page, where "up" runs
        # off the visible viewport entirely (Jon's 2026-07-28 report).
        # Flip to whichever side of the cursor actually has room, based
        # on the canvas's own CURRENT scroll position, not just the
        # image's full size.
        visible_left = self.canvas.canvasx(0)
        visible_top = self.canvas.canvasy(0)
        visible_right = self.canvas.canvasx(self.canvas.winfo_width())
        margin = 10

        pos_x = canvas_x + 30
        if pos_x + LOUPE_DIAMETER > visible_right - margin:
            pos_x = canvas_x - 30 - LOUPE_DIAMETER

        pos_y = canvas_y - 30 - LOUPE_DIAMETER
        if pos_y < visible_top + margin:
            pos_y = canvas_y + 30  # not enough room above - drop it below the cursor instead

        self.canvas.delete("loupe")
        self.canvas.create_image(pos_x, pos_y, anchor="nw", image=self._loupe_photo, tags="loupe")

    def _on_leave(self, _event):
        self.canvas.delete("loupe")

    def undo(self):
        if self.phase == "bounds":
            if self.bounds_step_index == 0:
                return
            self.bounds_step_index -= 1
            self.bounds.pop(self.bounds_steps[self.bounds_step_index], None)
        elif self.phase == "rows":
            if self.pending_row_top is not None:
                self.pending_row_top = None
            elif self.manual_rows:
                top, _bottom = self.manual_rows.pop()
                self.pending_row_top = top
        elif self.phase == "columns":
            if self.col_step_index == 0:
                return
            self.col_step_index -= 1
            self.col_clicks.pop(self.column_steps[self.col_step_index], None)
        elif self.phase == "done":
            self.phase = "columns"
            self.col_step_index -= 1
            self.col_clicks.pop(self.column_steps[self.col_step_index], None)
        self._render()

    def finish_rows(self):
        if self.phase != "rows" or self.pending_row_top is not None or not self.manual_rows:
            return
        self.phase = "columns"
        self._render()

    def skip_page(self):
        self.page_index += 1
        self._load_page()

    # ----------------------------------------------------------- accept

    def accept_page(self):
        l, r = self.bounds["table_left"], self.bounds["table_right"]
        if r <= l:
            l, r = min(l, r), max(l, r)

        if self.mode == "fixed":
            # Per Jon's 2026-07-28 direction: rather than re-searching
            # for row 1's own boundary (what core/auto_sidecar.py's
            # detect_data_rows() does, and exactly the kind of automated
            # guess that put this page in quarantine in the first
            # place), divide the human-confirmed row-1-top -> last-row-
            # bottom span evenly by the template's own expected_row_count
            # and tile from that - pure arithmetic, no search.
            t, b = self.bounds["row1_top"], self.bounds["last_row_bottom"]
            if b <= t:
                t, b = min(t, b), max(t, b)
            row_count = self.template.expected_row_count
            row_height = (b - t) / row_count
            table_bbox = (l, t, r, b)
            deskew_angle = self.sidecar.get("deskew_angle", 0.0)
            header_bbox, metadata_bottom, _ = locate_header_region(self.image, self.template, table_bbox)
            try:
                result, _row_crops, _header_crop, _overlay = segment_rows_uniform_tile(
                    self.image,
                    row_count=row_count,
                    first_row_top=float(t), first_row_bottom=float(t + row_height),
                    table_left=l, table_right=r,
                    header_row_count=1,
                    deskew_angle=deskew_angle,
                    header_box_top=metadata_bottom, header_box_bottom=t,
                )
            except Exception as e:
                messagebox.showerror("Quarantine Review", f"Row generation failed: {e}")
                return
            # result.bands is list[(y0, y1)] only (core/row_segmentation
            # .py's RowDetectionResult - no x-extent, since row detection
            # is vertical-only) - the sidecar schema's row "bbox" needs
            # the full [x0, y0, x1, y1], so the table's own left/right
            # fill in x here rather than being dropped.
            rows = [{"index": i + 1, "bbox": [l, y0, r, y1]} for i, (y0, y1) in enumerate(result.bands)]
            min_ratio, max_ratio = _PERIODIC_MIN_HEIGHT_RATIO, _PERIODIC_MAX_HEIGHT_RATIO
        else:
            # Irregular spacing (manifests, and occasional misrouted UK
            # census pages per Jon's note): rows came directly from the
            # person's own top/bottom clicks, one per row - no tiling,
            # no search, just what was clicked.
            t, b = self.manual_rows[0][0], self.manual_rows[-1][1]
            table_bbox = (l, t, r, b)
            deskew_angle = self.sidecar.get("deskew_angle", 0.0)
            header_bbox, metadata_bottom, _ = locate_header_region(self.image, self.template, table_bbox)
            rows = [{"index": i + 1, "bbox": [l, rt, r, rb]} for i, (rt, rb) in enumerate(self.manual_rows)]
            min_ratio, max_ratio = _DETECT_MIN_HEIGHT_RATIO, _DETECT_MAX_HEIGHT_RATIO

        # Verification-before-accept (2026-07-28, per Jon's direction):
        # run the manually-defined rows back through core/auto_sidecar
        # .py's OWN quarantine-rate gate before trusting them - the exact
        # same check (_find_quarantine_positions + PAGE_QUARANTINE_
        # THRESHOLD) the automated pipeline itself uses. If too many rows
        # would still be quarantined, this pass isn't confident enough to
        # accept - reset the page for another attempt rather than saving
        # a low-confidence result.
        heights = [row["bbox"][3] - row["bbox"][1] for row in rows]
        quarantine_positions = _find_quarantine_positions(heights, min_ratio, max_ratio)
        quarantine_rate = (len(quarantine_positions) / len(heights)) if heights else 1.0

        if quarantine_rate >= PAGE_QUARANTINE_THRESHOLD:
            messagebox.showwarning(
                "Quarantine Review",
                f"{len(quarantine_positions)}/{len(heights)} rows ({quarantine_rate:.0%}) still look "
                f"structurally implausible - at/above the {PAGE_QUARANTINE_THRESHOLD:.0%} confidence "
                f"threshold. Not accepting yet - redefine this page's rows.",
            )
            self._reset_state()
            self._render()
            return

        self.sidecar["table_bbox"] = list(table_bbox)
        self.sidecar["header_bbox"] = list(header_bbox) if header_bbox else None
        self.sidecar["rows"] = rows
        self.sidecar["rows_needs_review"] = []
        self.sidecar.setdefault("warnings", []).append(
            f"Rows manually confirmed via ui/quarantine_review_ui.py ({self.mode} mode) and "
            f"re-verified against the automated quarantine-rate gate ({quarantine_rate:.0%})."
        )
        _quarantine_anomalous_rows(self.sidecar)

        # Discard whatever column scaffold the earlier automated attempt
        # left behind (it was built from template.expected_columns - the
        # FULL CV-detectable set, e.g. every census template's "Sex" -
        # not the curated config/columns/*.txt list) and rebuild fresh
        # from self.active_columns only, per Jon's 2026-07-28 direction:
        # "only what's defined in the current column definition."
        self.sidecar["columns"] = {}
        self.sidecar.pop("active_column", None)
        self.sidecar.pop("progress", None)
        init_column_state(self.sidecar, self.active_columns)
        for col in self.active_columns:
            x0 = self.col_clicks[f"col:{col}:left"]
            x1 = self.col_clicks[f"col:{col}:right"]
            if x1 < x0:
                x0, x1 = x1, x0
            self.sidecar["columns"][col]["mask_keep_ranges"] = [[x0, x1]]
            self.sidecar["columns"][col]["status"] = "done"

        save_sidecar(self.sidecar, self.sidecar_path)
        self.page_index += 1
        self._load_page()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sidecar-dir", type=str, default=str(DEFAULT_SIDECAR_DIR),
                         help="Directory of auto-generated sidecars to scan for quarantined pages.")
    parser.add_argument("--columns-file", type=str, default=None,
                         help="Force a specific config/columns/*.txt file for every page in this run, "
                              "overriding each page's own template's columns_file default.")
    args = parser.parse_args()

    sidecar_dir = Path(args.sidecar_dir)
    if not sidecar_dir.exists():
        print(f"ERROR: sidecar dir not found: {sidecar_dir}")
        sys.exit(1)

    candidates = find_quarantined_sidecars(sidecar_dir)
    if not candidates:
        print(f"No quarantined pages found in {sidecar_dir}")
        return

    print(f"Found {len(candidates)} quarantined page(s).")
    root = Tk()
    QuarantineReviewApp(root, candidates, columns_file_override=args.columns_file)
    root.mainloop()


if __name__ == "__main__":
    main()
