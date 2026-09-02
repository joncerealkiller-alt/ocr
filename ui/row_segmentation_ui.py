"""
Row Segmentation UI — standalone tool, zero model/VRAM dependency.

Rebuilt 2026-07-13 per Jon's direction: this is NOT just "expose the
script with fields" - it's a visual adjustment workflow. Deskew angle
and table bounds are CHEAP, USER-CONTROLLED steps with instant preview
feedback (rotate + draw bounds, no detection at all - fast enough for
every nudge-button click); the EXPENSIVE step (ruling-line detection +
periodic row refinement) only runs when explicitly requested via
"Refine rows", using whatever angle/bounds the user has confirmed by
that point. Auto-estimated deskew is a starting SUGGESTION, never
authoritative - the user can always see, override, and fine-tune it
before anything downstream depends on it.

    Load image
    -> adjust deskew (auto-suggest, then nudge/edit until table lines
       actually look horizontal in the preview)
    -> set/refine table bounds (top/bottom/left/right, visually)
    -> Refine rows (the one expensive step - locked-in angle + bounds)
    -> tweak row count / search radius and re-refine cheaply (angle/
       bounds already locked, no need to redo deskew/detection from
       scratch)
    -> Save (writes the confirmed spatial contract as sidecar JSON)

Separate standalone tool, not a tab inside model_assessment.py - zero
model/VRAM dependency, mirrors review_uncertain.py's separation.
"""

from __future__ import annotations

import json
import platform
import queue
import subprocess
import sys
import threading
from pathlib import Path

# Moved into ui/ (2026-07-25) - one directory deeper than repo root, so
# repo root must be put back on sys.path before any `core.*` import
# below will resolve. See PROJECT_ROOT's own reuse further down for
# path-building.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tkinter import (
    Tk, Frame, Label, Button, Entry, StringVar, OptionMenu, Text,
    Checkbutton, BooleanVar, filedialog, messagebox, END, NORMAL, DISABLED,
    Canvas, Scrollbar, VERTICAL, HORIZONTAL,
)

from PIL import Image, ImageTk, ImageDraw

from core.row_segmentation import (
    segment_rows, segment_rows_periodic, segment_rows_uniform_tile,
    estimate_table_extent, estimate_deskew_angle, apply_deskew_angle,
    build_sidecar, save_sidecar, load_sidecar, update_sidecar,
    update_sidecar_preprocessing,
)
from core.image_preprocessing import apply_pipeline
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "row_segmentation"
PREVIEW_SIZE = (760, 920)

ANGLE_NUDGE_STEPS = [-0.25, -0.05, 0.05, 0.25]

UI_STATE_PATH = OUTPUT_DIR / "_ui_state.json"

# Fields genuinely specific to ONE exact scan - blindly carrying these
# over to a different image would be actively wrong (this session's
# whole lesson: different pages/forms need different table_top etc.).
# Saved/restored per-image, keyed by resolved file path.
PER_IMAGE_FIELDS = [
    "angle_var", "table_top_var", "table_bottom_var", "table_left_var",
    "table_right_var", "metadata_bottom_var", "header_box_top_var",
    "header_box_bottom_var", "row1_top_var", "row1_bottom_var",
    "mask_apply_header_var", "mask_apply_rows_var",
]

# Fields more likely to transfer usefully across different pages of the
# same form type - saved as a single "last used" fallback, applied on
# startup and for any image with no per-image entry of its own.
GENERAL_FIELDS = [
    "mode_var", "row_count_var", "search_radius_var", "header_rows_var",
    "debug_crops_var", "padding_var", "padding_pct_var", "zoom_var",
    # Preprocessing (2026-07-25) - a "how do I want to view any scan"
    # choice, more like zoom than per-image geometry, so it belongs
    # here rather than in PER_IMAGE_FIELDS.
    "preproc_grayscale_var", "preproc_invert_var",
    "preproc_clahe_var", "preproc_clahe_clip_var", "preproc_clahe_tile_var",
    "preproc_median_var", "preproc_median_size_var",
    "preproc_unsharp_var", "preproc_unsharp_radius_var", "preproc_unsharp_amount_var",
    "preproc_upscale_var", "preproc_upscale_factor_var",
    "preproc_adaptive_threshold_var", "preproc_at_block_size_var", "preproc_at_c_var",
]


def _load_ui_state() -> dict:
    if not UI_STATE_PATH.exists():
        return {"per_image": {}, "last_general": {}}
    try:
        with open(UI_STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"per_image": {}, "last_general": {}}


def _save_ui_state(state: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(UI_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


class RowSegmentationApp:
    # Click-to-set (2026-07-26, Jon's direction): these bounds are all
    # "identify a visual location on the page" values - same direct-
    # manipulation interaction style as ui/dewarp_preprocessor_ui.py's
    # corner clicking, instead of estimate-type-refresh-repeat. Row 1
    # top/bottom deliberately have NO entry here - Jon's explicit
    # direction that those stay precision/manual-refinement-only (a 1px
    # error there compounds across row_count tiled rows, unlike every
    # field below, which is set once).
    _BORDER_CLICK_BUTTON_LABELS = {
        "table_top": "Set Top",
        "table_bottom": "Set Bottom",
        "table_left": "Set Left",
        "table_right": "Set Right",
        "metadata_bottom": "Set Metadata Bottom",
        "header_box_top": "Set Header Top",
        "header_box_bottom": "Set Header Bottom",
    }
    _BORDER_CLICK_ORIENTATION = {
        "table_top": "h", "table_bottom": "h",
        "table_left": "v", "table_right": "v",
        "metadata_bottom": "h",
        "header_box_top": "h", "header_box_bottom": "h",
    }

    def __init__(self, root: Tk):
        self.root = root
        root.title("Row Segmentation - visual adjustment workflow (no model calls)")
        root.geometry("1500x1080")
        root.minsize(1300, 780)

        self.image_path: str | None = None
        self.original_image: Image.Image | None = None
        self.last_result = None          # RowDetectionResult from last "Refine rows"
        self.last_row_crops = None
        self._last_overlay_path: str | None = None
        self._tk_preview = None
        self._preview_scale = 1.0
        self._preview_render_size: tuple[int, int] = (0, 0)

        # Click-to-set border mode (2026-07-26): None when idle, else one
        # of _BORDER_CLICK_BUTTON_LABELS's keys - see _start_border_click_
        # mode()/_handle_border_click() below.
        self._border_click_mode: str | None = None
        self._border_guide_line_id: int | None = None

        # Column masking (2026-07-15, per Jon's direction, corrected):
        # real testing showed Age contamination pulling from DIFFERENT
        # nearby columns on different rows (dwelling numbers on one row,
        # section/township/range on another) - a spatial-counting
        # problem, not a labeling one. User clicks the ONE column to
        # KEEP (e.g. just Age's real column) - everything OUTSIDE it
        # gets masked white. This is the inverse of "click each unwanted
        # column" - much less manual work when isolating one narrow
        # target column in a wide, dense row.
        #
        # Stored as keep_ranges (what was actually selected, in full
        # deskewed-image coordinates) rather than pre-computed exclude
        # ranges - more transparent/debuggable, and the complement
        # (core.row_segmentation.compute_exclude_ranges) is computed
        # fresh wherever the actual image width is known.
        #
        # Scope flags: header and row-data columns don't share the same
        # layout (header has page-level fields like Province/District
        # spanning wide; rows have narrow per-person data columns), so
        # a keep-range tuned for isolating one row column would be
        # WRONG to also apply to the header crop - these let the user
        # decide independently.
        # column_masks (2026-07-21): keep-ranges grouped by column name
        # instead of one flat anonymous list - a prerequisite for the
        # persistent per-column sidecar workflow, which needs to know
        # WHICH column a given mask belongs to, not just that some mask
        # was clicked. self.active_column_var selects which entry
        # _on_canvas_click/_clear_masks operate on; ranges for a column
        # not yet in the dict are created on first click via
        # self._active_ranges().
        self.column_masks: dict[str, list[tuple[int, int]]] = {}
        # Full ordered column list, loaded upfront via "Load columns
        # file..." (same one-name-per-line format ground_truth_labeling_
        # ui.py's Load columns file uses). Without this, update_sidecar
        # only ever knows about columns typed so far, so there's nothing
        # to auto-advance TO beyond the current one - loading this list
        # is what makes "Next column" actually walk the whole form
        # instead of stopping after whatever's been named manually.
        self.column_order: list[str] = []
        self.active_column_var = StringVar(value="")
        self._mask_click_start: int | None = None
        self.mask_apply_header_var = BooleanVar(value=False)
        self.mask_apply_rows_var = BooleanVar(value=True)

        # Preprocessing (2026-07-25, per Jon's direction): applied to the
        # PREVIEW only, on top of the existing deskew step - never
        # touches the original source image on disk. Checkboxes only
        # ever enable/disable a step; the actual execution order is
        # fixed internally by core.image_preprocessing.PIPELINE_ORDER,
        # not by which order the user happens to check boxes in (Jon's
        # explicit requirement - e.g. running threshold before CLAHE is
        # rarely useful, so that ordering mistake isn't offered as an
        # option at all). Saved to the sidecar's own "preprocessing" key
        # via update_sidecar_preprocessing() - a separate, page-level
        # setting alongside deskew_angle, not folded into any column's
        # mask/results state.
        self.preproc_grayscale_var = BooleanVar(value=True)
        self.preproc_invert_var = BooleanVar(value=False)
        self.preproc_clahe_var = BooleanVar(value=True)
        self.preproc_clahe_clip_var = StringVar(value="2.0")
        self.preproc_clahe_tile_var = StringVar(value="8")
        self.preproc_median_var = BooleanVar(value=False)
        self.preproc_median_size_var = StringVar(value="3")
        self.preproc_unsharp_var = BooleanVar(value=True)
        self.preproc_unsharp_radius_var = StringVar(value="1.5")
        self.preproc_unsharp_amount_var = StringVar(value="1.2")
        self.preproc_upscale_var = BooleanVar(value=True)
        self.preproc_upscale_factor_var = StringVar(value="2.0")
        self.preproc_adaptive_threshold_var = BooleanVar(value=False)
        self.preproc_at_block_size_var = StringVar(value="11")
        self.preproc_at_c_var = StringVar(value="2.0")

        top = Frame(root, padx=10, pady=10)
        top.pack(fill="both", expand=True)

        # -- preprocessing section (2026-07-25) --------------------------
        # Preview-only - never touches the original source image. Fixed
        # execution order (core.image_preprocessing.PIPELINE_ORDER), not
        # whatever order these checkboxes happen to be toggled in.
        preproc_frame = Frame(top, relief="groove", borderwidth=1, padx=8, pady=4)
        preproc_frame.pack(fill="x", pady=(0, 3))
        Label(preproc_frame,
              text="Preprocessing (preview only, saved to sidecar) + Deskew "
                   "(auto-estimate is a SUGGESTION) - confirm everything visually:",
              font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="w")

        simple_row = Frame(preproc_frame)
        simple_row.grid(row=1, column=0, sticky="w", pady=(4, 0))
        Checkbutton(simple_row, text="Grayscale", variable=self.preproc_grayscale_var,
                    command=self.update_preview).pack(side="left")
        Checkbutton(simple_row, text="Invert", variable=self.preproc_invert_var,
                    command=self.update_preview).pack(side="left", padx=(10, 0))
        Checkbutton(simple_row, text="Median", variable=self.preproc_median_var,
                    command=self.update_preview).pack(side="left", padx=(10, 0))
        Label(simple_row, text="Size:").pack(side="left", padx=(4, 2))
        Entry(simple_row, textvariable=self.preproc_median_size_var, width=4).pack(side="left")
        Checkbutton(simple_row, text="Upscale", variable=self.preproc_upscale_var,
                    command=self.update_preview).pack(side="left", padx=(10, 0))
        Label(simple_row, text="Factor:").pack(side="left", padx=(4, 2))
        Entry(simple_row, textvariable=self.preproc_upscale_factor_var, width=4).pack(side="left")

        # Expandable filters (CLAHE / Unsharp / Adaptive Threshold) -
        # most users never touch the underlying parameters, so they
        # start collapsed regardless of whether the filter itself is
        # enabled - the checkbox controls "does this run", the arrow
        # controls "are the knobs visible", independently.
        expand_row = Frame(preproc_frame)
        expand_row.grid(row=2, column=0, sticky="w", pady=(4, 0))

        Checkbutton(expand_row, text="CLAHE", variable=self.preproc_clahe_var,
                    command=self.update_preview).pack(side="left")
        clahe_toggle = Button(expand_row, text="▶", width=2)
        clahe_toggle.pack(side="left", padx=(2, 0))
        clahe_params = Frame(expand_row)
        Label(clahe_params, text="Clip:").pack(side="left", padx=(6, 2))
        Entry(clahe_params, textvariable=self.preproc_clahe_clip_var, width=5).pack(side="left")
        Label(clahe_params, text="Tile:").pack(side="left", padx=(6, 2))
        Entry(clahe_params, textvariable=self.preproc_clahe_tile_var, width=5).pack(side="left")
        clahe_toggle.config(command=lambda: self._toggle_filter_params(clahe_params, clahe_toggle))

        Checkbutton(expand_row, text="Unsharp", variable=self.preproc_unsharp_var,
                    command=self.update_preview).pack(side="left", padx=(16, 0))
        unsharp_toggle = Button(expand_row, text="▶", width=2)
        unsharp_toggle.pack(side="left", padx=(2, 0))
        unsharp_params = Frame(expand_row)
        Label(unsharp_params, text="Radius:").pack(side="left", padx=(6, 2))
        Entry(unsharp_params, textvariable=self.preproc_unsharp_radius_var, width=5).pack(side="left")
        Label(unsharp_params, text="Amount:").pack(side="left", padx=(6, 2))
        Entry(unsharp_params, textvariable=self.preproc_unsharp_amount_var, width=5).pack(side="left")
        unsharp_toggle.config(
            command=lambda: self._toggle_filter_params(unsharp_params, unsharp_toggle))

        Checkbutton(expand_row, text="Adaptive Threshold",
                    variable=self.preproc_adaptive_threshold_var,
                    command=self.update_preview).pack(side="left", padx=(16, 0))
        at_toggle = Button(expand_row, text="▶", width=2)
        at_toggle.pack(side="left", padx=(2, 0))
        at_params = Frame(expand_row)
        Label(at_params, text="Block:").pack(side="left", padx=(6, 2))
        Entry(at_params, textvariable=self.preproc_at_block_size_var, width=5).pack(side="left")
        Label(at_params, text="C:").pack(side="left", padx=(6, 2))
        Entry(at_params, textvariable=self.preproc_at_c_var, width=5).pack(side="left")
        at_toggle.config(command=lambda: self._toggle_filter_params(at_params, at_toggle))

        # -- deskew controls, same box now -----------------------------------
        deskew_row = Frame(preproc_frame)
        deskew_row.grid(row=3, column=0, sticky="w", pady=(6, 0))
        Label(deskew_row, text="Deskew:").pack(side="left")
        Button(deskew_row, text="Auto-estimate", command=self.auto_estimate_angle
               ).pack(side="left", padx=(6, 0))
        self.angle_var = StringVar(value="0.0")
        Entry(deskew_row, textvariable=self.angle_var, width=6).pack(side="left", padx=(6, 10))
        for step in ANGLE_NUDGE_STEPS:
            label = f"{step:+.2f}°"
            Button(deskew_row, text=label, width=6,
                   command=lambda s=step: self.nudge_angle(s)).pack(side="left", padx=1)
        Button(deskew_row, text="Reset to 0°", command=self.reset_angle
               ).pack(side="left", padx=(10, 0))

        bottom_row = Frame(preproc_frame)
        bottom_row.grid(row=4, column=0, sticky="w", pady=(6, 0))
        Button(bottom_row, text="Refresh preview", command=self.update_preview,
               bg="#68a", fg="white").pack(side="left")
        Label(bottom_row, text="Any edit above needs Refresh preview to apply.",
              font=("Segoe UI", 8), fg="#666").pack(side="left", padx=(10, 0))

        # -- image selection ---------------------------------------------
        img_row = Frame(top)
        img_row.pack(fill="x", pady=4)
        Button(img_row, text="Select image...", command=self.select_image).pack(side="left")
        self.open_overlay_button = Button(
            img_row, text="Open full-resolution overlay",
            command=self.open_overlay_in_viewer, state=DISABLED,
        )
        self.open_overlay_button.pack(side="left", padx=(6, 0))
        self.image_path_label = Label(img_row, text="(no image selected)", fg="#666")
        self.image_path_label.pack(side="left", padx=8)

        # -- table bounds section -------------------------------------------
        bounds_frame = Frame(top, relief="groove", borderwidth=1, padx=8, pady=4)
        bounds_frame.pack(fill="x", pady=3)
        Label(bounds_frame, text="Table bounds (pixels, in the DESKEWED image) - "
                                  "confirm visually via the preview:",
              font=("Segoe UI", 9, "bold")).grid(row=0, column=0, columnspan=6, sticky="w")

        # COMPACTED 2026-07-15 per Jon's direction: the original version
        # (built across several additions through 2026-07-13) gave every
        # field its own multi-line wrapped paragraph on a separate row -
        # reasonable when each was new and needed real explanation, but
        # cumulatively pushed the actual preview panel off the bottom of
        # the window, confirmed via a real screenshot. Hints are now one
        # short line each, placed INLINE next to their fields rather than
        # as a separate row underneath - cuts this section from 11 rows
        # to 5. Full explanations remain in this file's git history /
        # earlier comments for anyone who needs the long version again.

        Label(bounds_frame, text="Top:").grid(row=1, column=0, sticky="w", pady=(3, 0))
        self.table_top_var = StringVar(value="")
        Entry(bounds_frame, textvariable=self.table_top_var, width=8).grid(
            row=1, column=1, sticky="w", padx=(2, 16), pady=(3, 0))
        Label(bounds_frame, text="Bottom:").grid(row=1, column=2, sticky="w", pady=(3, 0))
        self.table_bottom_var = StringVar(value="")
        Entry(bounds_frame, textvariable=self.table_bottom_var, width=8).grid(
            row=1, column=3, sticky="w", padx=(2, 16), pady=(3, 0))
        Label(bounds_frame, text="Left:").grid(row=1, column=4, sticky="w", pady=(3, 0))
        self.table_left_var = StringVar(value="")
        Entry(bounds_frame, textvariable=self.table_left_var, width=8).grid(
            row=1, column=5, sticky="w", padx=(2, 16), pady=(3, 0))
        Label(bounds_frame, text="Right:").grid(row=1, column=6, sticky="w", pady=(3, 0))
        self.table_right_var = StringVar(value="")
        Entry(bounds_frame, textvariable=self.table_right_var, width=8).grid(
            row=1, column=7, sticky="w", padx=(2, 0), pady=(3, 0))

        click_row = Frame(bounds_frame)
        click_row.grid(row=2, column=0, columnspan=8, sticky="w", pady=(4, 0))
        Label(click_row, text="Click-set:", font=("Segoe UI", 8)).pack(side="left")
        for _key in ("table_top", "table_bottom", "table_left", "table_right",
                     "metadata_bottom", "header_box_top", "header_box_bottom"):
            Button(click_row, text=self._BORDER_CLICK_BUTTON_LABELS[_key], width=14,
                   command=lambda k=_key: self._start_border_click_mode(k)
                   ).pack(side="left", padx=(6, 0))
        self.border_click_status_label = Label(click_row, text="", font=("Segoe UI", 8), fg="#a04")
        self.border_click_status_label.pack(side="left", padx=(10, 0))

        Button(bounds_frame, text="Auto-estimate top/bottom",
               command=self.auto_estimate_extent).grid(row=3, column=0, columnspan=2,
                                                          sticky="w", pady=(4, 3))
        Label(bounds_frame, text="(top/bottom only - set left/right by eye)",
              font=("Segoe UI", 8), fg="#666").grid(row=3, column=2, columnspan=4,
                                                      sticky="w", pady=(4, 3))

        Label(bounds_frame, text="Metadata bottom:").grid(row=4, column=0, sticky="w", pady=(3, 0))
        self.metadata_bottom_var = StringVar(value="")
        Entry(bounds_frame, textvariable=self.metadata_bottom_var, width=8).grid(
            row=4, column=1, sticky="w", padx=(2, 8), pady=(3, 0))
        Label(bounds_frame, text="splits above Top into metadata/headings - cyan line, optional",
              font=("Segoe UI", 8), fg="#666").grid(row=4, column=2, columnspan=6,
                                                      sticky="w", pady=(3, 0))

        Label(bounds_frame, text="Header box top:").grid(row=5, column=0, sticky="w", pady=(3, 0))
        self.header_box_top_var = StringVar(value="")
        Entry(bounds_frame, textvariable=self.header_box_top_var, width=8).grid(
            row=5, column=1, sticky="w", padx=(2, 16), pady=(3, 0))
        Label(bounds_frame, text="Header box bottom:").grid(row=5, column=2, sticky="w", pady=(3, 0))
        self.header_box_bottom_var = StringVar(value="")
        Entry(bounds_frame, textvariable=self.header_box_bottom_var, width=8).grid(
            row=5, column=3, sticky="w", padx=(2, 8), pady=(3, 0))
        Label(bounds_frame, text="exact strip prepended to every row - magenta box, overrides Metadata bottom",
              font=("Segoe UI", 8), fg="#666").grid(row=5, column=4, columnspan=4,
                                                      sticky="w", pady=(3, 0))

        Label(bounds_frame, text="[uniform_tile] Row 1 top:").grid(
            row=6, column=0, sticky="w", pady=(3, 3))
        self.row1_top_var = StringVar(value="")
        Entry(bounds_frame, textvariable=self.row1_top_var, width=8).grid(
            row=6, column=1, sticky="w", padx=(2, 16), pady=(3, 3))
        Label(bounds_frame, text="Row 1 bottom:").grid(row=6, column=2, sticky="w", pady=(3, 3))
        self.row1_bottom_var = StringVar(value="")
        Entry(bounds_frame, textvariable=self.row1_bottom_var, width=8).grid(
            row=6, column=3, sticky="w", padx=(2, 8), pady=(3, 3))
        Label(bounds_frame, text="confirm via Refresh preview - yellow box - then tiled row_count times "
                                  "(precision field - manual only, no click-set)",
              font=("Segoe UI", 8), fg="#666").grid(row=6, column=4, columnspan=4,
                                                      sticky="w", pady=(3, 3))

        # -- periodic-mode parameters (row count / search radius) -----------
        params_frame = Frame(top, relief="groove", borderwidth=1, padx=8, pady=4)
        params_frame.pack(fill="x", pady=4)
        Label(params_frame, text="Refinement parameters:",
              font=("Segoe UI", 9, "bold")).grid(row=0, column=0, columnspan=6, sticky="w")

        Label(params_frame, text="Mode:").grid(row=1, column=0, sticky="w", pady=(3, 0))
        self.mode_var = StringVar(value="periodic")
        OptionMenu(params_frame, self.mode_var, "detect", "periodic", "uniform_tile").grid(
            row=1, column=1, sticky="w", padx=(2, 12), pady=(3, 0))

        Label(params_frame, text="Row count:").grid(row=1, column=2, sticky="w", pady=(3, 0))
        self.row_count_var = StringVar(value="50")
        Entry(params_frame, textvariable=self.row_count_var, width=6).grid(
            row=1, column=3, sticky="w", padx=(2, 12), pady=(3, 0))

        Label(params_frame, text="Search radius ratio:").grid(row=1, column=4, sticky="w", pady=(3, 0))
        self.search_radius_var = StringVar(value="0.3")
        Entry(params_frame, textvariable=self.search_radius_var, width=6).grid(
            row=1, column=5, sticky="w", padx=(2, 0), pady=(3, 0))

        Label(params_frame, text="Header rows:").grid(row=2, column=0, sticky="w", pady=(3, 0))
        self.header_rows_var = StringVar(value="0")
        Entry(params_frame, textvariable=self.header_rows_var, width=6).grid(
            row=2, column=1, sticky="w", padx=(2, 12), pady=(3, 0))

        Label(params_frame, text="Row padding (px):").grid(row=2, column=2, sticky="w", pady=(3, 0))
        self.padding_var = StringVar(value="4")
        Entry(params_frame, textvariable=self.padding_var, width=6).grid(
            row=2, column=3, sticky="w", padx=(2, 12), pady=(3, 0))

        Label(params_frame, text="or %:").grid(row=2, column=4, sticky="w", pady=(3, 0))
        self.padding_pct_var = StringVar(value="")
        Entry(params_frame, textvariable=self.padding_pct_var, width=6).grid(
            row=2, column=5, sticky="w", padx=(2, 0), pady=(3, 0))

        self.debug_crops_var = BooleanVar(value=False)
        Checkbutton(params_frame, text="Also save individual row PNGs (debug)",
                    variable=self.debug_crops_var).grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(3, 0))
        Label(params_frame, text="oversample beyond the line; %, if set, overrides px "
                                  "(~20%+ risks the adjacent row)",
              font=("Segoe UI", 8), fg="#666").grid(row=3, column=3, columnspan=3,
                                                      sticky="w", pady=(3, 0))

        # -- action buttons ---------------------------------------------------
        action_row = Frame(top)
        action_row.pack(fill="x", pady=5)
        Button(action_row, text="Refine rows (expensive - runs detection)",
               command=self.refine_rows, bg="#4a7", fg="white",
               font=("Segoe UI", 10, "bold")).pack(side="left")
        self.save_button = Button(action_row, text="Save sidecar JSON",
                                   command=self.save_result, state=DISABLED)
        self.save_button.pack(side="left", padx=(8, 0))
        self.next_column_button = Button(
            action_row, text="Mark column done \u2192 Next",
            command=self.next_column, state=DISABLED, bg="#68a", fg="white")
        self.next_column_button.pack(side="left", padx=(8, 0))
        self.status_label = Label(action_row, text="", fg="#444")
        self.status_label.pack(side="left", padx=12)

        self.column_progress_label = Label(top, text="", font=("Segoe UI", 9), fg="#333",
                                            justify="left", anchor="w")
        self.column_progress_label.pack(fill="x", pady=(0, 4))

        # -- extraction (runs core.row_extraction against the active column) --
        extract_row = Frame(top)
        extract_row.pack(fill="x", pady=(0, 5))
        Label(extract_row, text="Model profile:", font=("Segoe UI", 8)).pack(side="left")
        self.model_profile_var = StringVar(value="")
        Entry(extract_row, textvariable=self.model_profile_var, width=16).pack(
            side="left", padx=(2, 8))
        self.extract_button = Button(
            extract_row, text="Extract active column",
            command=self.extract_active_column, state=DISABLED, bg="#a62", fg="white")
        self.extract_button.pack(side="left")
        # --debug-model-inputs (2026-07-24): same flag as run_row_extraction.py's
        # CLI, exposed here so a UI-driven extraction can be debugged the
        # same way - see core/debug_dump.py. Off by default, no effect on
        # extraction results when unchecked.
        self.debug_model_inputs_var = BooleanVar(value=False)
        Checkbutton(extract_row, text="Debug model inputs",
                    variable=self.debug_model_inputs_var).pack(side="left", padx=(8, 0))
        self.extract_status_label = Label(extract_row, text="", fg="#444")
        self.extract_status_label.pack(side="left", padx=(10, 0))

        # Background-extraction plumbing: core.row_extraction pulls in
        # the full model-loading stack (torch/transformers etc.) via
        # core.loader_registry - a HEAVY, GPU-relevant import this UI
        # otherwise has no reason to need at startup (someone doing
        # pure masking work shouldn't need torch installed just to open
        # this window). Imported lazily, inside the worker thread only,
        # the first time extraction is actually run. Extraction itself
        # also genuinely blocks for a while (model load + N rows), so
        # it runs on a background thread with results handed back
        # through a thread-safe queue and polled via root.after -
        # tkinter widgets must only be touched from the main thread.
        self._extraction_queue: queue.Queue = queue.Queue()
        self._extraction_running = False

        # -- preview + report, side by side ------------------------------------
        main = Frame(top)
        main.pack(fill="both", expand=True, pady=(4, 0))

        preview_frame = Frame(main)
        preview_frame.pack(side="left", fill="both", expand=True)

        preview_header = Frame(preview_frame)
        preview_header.pack(fill="x")
        Label(preview_header, text="Preview (purple = table bounds, blue = header, "
                                    "red/green = refined rows once available):",
              font=("Segoe UI", 9)).pack(side="left")
        Label(preview_header, text="  Zoom:").pack(side="left", padx=(12, 0))
        self.zoom_var = StringVar(value="Fit")
        OptionMenu(preview_header, self.zoom_var, "Fit", "2x", "4x", "8x",
                   command=lambda _: self.update_preview()).pack(side="left")
        Label(preview_header, text="(scroll freely at any zoom level - "
                                    "position is preserved across refreshes)",
              font=("Segoe UI", 8), fg="#666").pack(side="left", padx=(6, 0))

        mask_row = Frame(preview_frame)
        mask_row.pack(fill="x", pady=(3, 0))
        Label(mask_row, text="Column:", font=("Segoe UI", 8)).pack(side="left")
        self.active_column_entry = Entry(mask_row, textvariable=self.active_column_var, width=16)
        self.active_column_entry.pack(side="left", padx=(2, 4))
        Button(mask_row, text="Load columns file...",
               command=self.load_columns_file).pack(side="left", padx=(0, 4))
        Button(mask_row, text="Use two-stage mask (__multi__)",
               command=self.use_multi_mask).pack(side="left", padx=(0, 8))
        self.mask_mode_var = BooleanVar(value=False)
        Checkbutton(mask_row, text="Select column to keep", variable=self.mask_mode_var,
                    command=self._on_mask_mode_toggle).pack(side="left")
        self.mask_instruction_label = Label(
            mask_row, text="", font=("Segoe UI", 8), fg="#a04")
        self.mask_instruction_label.pack(side="left", padx=(6, 0))
        Button(mask_row, text="Clear this column's mask", command=self._clear_masks).pack(
            side="left", padx=(10, 0))
        self.mask_count_label = Label(mask_row, text="(no column selected)",
                                       fg="#444")
        self.mask_count_label.pack(side="left", padx=(6, 0))
        Label(mask_row, text="  Apply to:", font=("Segoe UI", 8)).pack(side="left", padx=(10, 2))
        Checkbutton(mask_row, text="Header", variable=self.mask_apply_header_var,
                    command=self.update_preview).pack(side="left")
        Checkbutton(mask_row, text="Rows", variable=self.mask_apply_rows_var,
                    command=self.update_preview).pack(side="left")

        canvas_frame = Frame(preview_frame)
        canvas_frame.pack(fill="both", expand=True, pady=(4, 0))
        self.preview_canvas = Canvas(canvas_frame, bg="#ddd")
        v_scroll = Scrollbar(canvas_frame, orient=VERTICAL, command=self.preview_canvas.yview)
        h_scroll = Scrollbar(canvas_frame, orient=HORIZONTAL, command=self.preview_canvas.xview)
        self.preview_canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        self.preview_canvas.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        self.preview_canvas.bind("<Button-1>", self._on_canvas_click)
        self.preview_canvas.bind("<Motion>", self._on_canvas_motion)
        root.bind("<Escape>", lambda e: self._cancel_border_click_mode())

        report_frame = Frame(main, width=320)
        report_frame.pack(side="left", fill="y", padx=(12, 0))
        report_frame.pack_propagate(False)
        Label(report_frame, text="Report:", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.report_text = Text(report_frame, wrap="word", width=38, font=("Consolas", 9))
        self.report_text.pack(fill="both", expand=True, pady=(4, 0))

        # Restore persisted settings (2026-07-13) - general fields only
        # at this point, since no image is selected yet. Per-image
        # geometry is restored in select_image() once a specific image
        # (and thus a lookup key) exists.
        self._ui_state = _load_ui_state()
        self._apply_general_settings(self._ui_state.get("last_general", {}))
        # Not a Var (see select_image()), so restored directly here rather
        # than through _apply_general_settings()'s Var-only mechanism.
        self._last_image_dir = self._ui_state.get("last_general", {}).get("last_image_dir", "")

    # -- persistence ----------------------------------------------------------

    def _capture_field_values(self, field_names: list) -> dict:
        """Reads current values from the given list of Var attribute
        names into a plain dict, ready for JSON serialization."""
        values = {}
        for name in field_names:
            var = getattr(self, name)
            values[name] = var.get()
        return values

    def _apply_general_settings(self, values: dict) -> None:
        """Restores GENERAL_FIELDS from a saved dict - used on startup
        and whenever a new (never-before-seen) image is selected. Does
        NOT touch per-image geometry fields - those stay blank/default
        for a genuinely new image, matching the rest of this tool's
        "don't guess geometry, confirm visually" philosophy."""
        for name in GENERAL_FIELDS:
            if name in values:
                getattr(self, name).set(values[name])

    def _apply_per_image_settings(self, values: dict) -> None:
        """Restores PER_IMAGE_FIELDS from a saved dict - used only when
        the exact same image (by resolved path) was previously worked
        on, so this geometry is known to actually apply to it."""
        for name in PER_IMAGE_FIELDS:
            if name in values:
                getattr(self, name).set(values[name])
        if "column_masks" in values:
            # JSON round-trips tuples as lists - convert back. Older
            # saved state may still have the flat "keep_ranges" key from
            # before column_masks existed; migrate it once under a
            # placeholder name rather than silently dropping it.
            self.column_masks = {
                name: [tuple(r) for r in ranges]
                for name, ranges in values["column_masks"].items()
            }
            self._refresh_mask_count_label()
        elif "keep_ranges" in values and values["keep_ranges"]:
            self.column_masks = {"_unnamed": [tuple(r) for r in values["keep_ranges"]]}
            self.active_column_var.set("_unnamed")
            self._refresh_mask_count_label()

    def _save_current_settings(self) -> None:
        """
        Persists current field values (2026-07-13, so relaunching the
        UI doesn't mean re-entering everything from scratch) - called
        after every successful 'Refine rows', not just on final Save,
        so work-in-progress survives a crash or accidental close.
        Per-image geometry is saved keyed by this image's resolved
        path; general settings are saved as a single "last used"
        fallback applied to any future image with no entry of its own.
        """
        if not self.image_path:
            return
        key = str(Path(self.image_path).resolve())
        all_values = self._capture_field_values(PER_IMAGE_FIELDS + GENERAL_FIELDS)
        # column_masks is a plain dict, not a Var - doesn't fit the
        # generic Var-based capture mechanism, stored as its own key.
        all_values["column_masks"] = self.column_masks
        self._ui_state.setdefault("per_image", {})[key] = all_values
        self._ui_state["last_general"] = self._capture_field_values(GENERAL_FIELDS)
        _save_ui_state(self._ui_state)

    # -- helpers ------------------------------------------------------------

    def _parse_optional_int(self, var: StringVar) -> int | None:
        raw = var.get().strip()
        return int(raw) if raw else None

    def _parse_optional_float(self, var: StringVar) -> float | None:
        """
        Used for Row 1 top/bottom specifically (2026-07-13, Jon's
        direction) - at row_count=50 a 1px rounding error in row height
        compounds to 50px of accumulated drift by the last row (see
        segment_rows_uniform_tile's docstring for the confirmed
        before/after numbers). Every other bounds field stays int-only
        since they're each used once, not multiplied by row_count.
        """
        raw = var.get().strip()
        return float(raw) if raw else None

    def _toggle_filter_params(self, param_frame: Frame, button: Button) -> None:
        """
        Shows/hides one expandable filter's parameter group (CLAHE/
        Unsharp/Adaptive Threshold) - independent of whether the
        filter itself is enabled, so a user can inspect/tune a
        currently-disabled filter's settings before turning it on.

        Compacted layout (2026-07-25): param_frame now packs INLINE in
        the same row as its checkbox/toggle button (not a separate grid
        row below), so re-showing must use pack(after=button) rather
        than a bare pack() - Tkinter's pack manager otherwise appends
        newly (re-)packed widgets to the END of their parent's packing
        order, which would misplace it after whichever OTHER filter's
        toggle happens to be expanded at the time, not back next to its
        own button.
        """
        if param_frame.winfo_ismapped():
            param_frame.pack_forget()
            button.config(text="▶")  # collapsed
        else:
            param_frame.pack(side="left", after=button)
            button.config(text="▼")  # expanded

    def _safe_float(self, var: StringVar, default: float) -> float:
        """Like _parse_optional_float, but a genuinely invalid (not
        just blank) value falls back to `default` instead of raising -
        used only for preprocessing params, where a live preview
        refresh shouldn't crash over a field mid-edit (e.g. a stray
        "2." typed on the way to "2.5")."""
        raw = var.get().strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    def _safe_int(self, var: StringVar, default: int) -> int:
        raw = var.get().strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            return default

    def _current_preprocessing_config(self) -> dict:
        """
        Reads the Preprocessing section's current checkbox/parameter
        state into the shape core.image_preprocessing.apply_pipeline()
        and core.row_segmentation.update_sidecar_preprocessing() both
        expect. A blank/invalid parameter field falls back to that
        filter's own default rather than raising - a preview refresh
        shouldn't crash over a field the user hasn't finished typing
        into yet.
        """
        return {
            "grayscale": {"enabled": self.preproc_grayscale_var.get()},
            "invert": {"enabled": self.preproc_invert_var.get()},
            "clahe": {
                "enabled": self.preproc_clahe_var.get(),
                "clip_limit": self._safe_float(self.preproc_clahe_clip_var, 2.0),
                "tile_size": self._safe_int(self.preproc_clahe_tile_var, 8),
            },
            "median": {
                "enabled": self.preproc_median_var.get(),
                "size": self._safe_int(self.preproc_median_size_var, 3),
            },
            "unsharp": {
                "enabled": self.preproc_unsharp_var.get(),
                "radius": self._safe_float(self.preproc_unsharp_radius_var, 1.5),
                "amount": self._safe_float(self.preproc_unsharp_amount_var, 1.2),
            },
            "upscale": {
                "enabled": self.preproc_upscale_var.get(),
                "factor": self._safe_float(self.preproc_upscale_factor_var, 2.0),
            },
            "adaptive_threshold": {
                "enabled": self.preproc_adaptive_threshold_var.get(),
                "block_size": self._safe_int(self.preproc_at_block_size_var, 11),
                "C": self._safe_float(self.preproc_at_c_var, 2.0),
            },
        }

    def _current_padding_kwargs(self) -> dict:
        """
        Row padding (2026-07-13, per Jon/GPT's suggestion) - fixed px
        always has a value (default 4), padding_pct overrides it when
        the field is non-empty (see _compute_padding()'s docstring for
        precedence).
        """
        try:
            padding = int(self.padding_var.get().strip() or 4)
        except ValueError:
            padding = 4
        padding_pct = self._parse_optional_float(self.padding_pct_var)
        return {"padding": padding, "padding_pct": padding_pct}

    def _current_header_box_kwargs(self) -> dict:
        """
        Explicit header crop box (2026-07-13) - see the UI field's own
        help text for why this exists (a real, multi-tier bilingual
        header block made the automatic Metadata-to-Top span too
        imprecise to blindly prepend to every row on a real census
        form). Both must be set to take effect; segment_rows_periodic/
        segment_rows_uniform_tile fall back to the automatic metadata_
        bottom-based span if either is missing.
        """
        return {
            "header_box_top": self._parse_optional_int(self.header_box_top_var),
            "header_box_bottom": self._parse_optional_int(self.header_box_bottom_var),
        }

    def _current_angle(self) -> float:
        try:
            return float(self.angle_var.get())
        except ValueError:
            return 0.0

    # -- image / deskew handlers ----------------------------------------------

    def select_image(self):
        path = filedialog.askopenfilename(
            title="Select census/table image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp")],
            initialdir=self._last_image_dir or None,
        )
        if not path:
            return
        # Remembered across sessions (2026-07-26, Jon's direction) - so
        # "Select image..." reopens in the same folder every time instead
        # of always starting wherever the OS/tkinter default happens to
        # land, since real usage is always working through one bucket/
        # folder of scans at a time. Persisted immediately, independent
        # of _save_current_settings() (which only runs once an image/
        # geometry actually exists) so it survives even a session where
        # nothing else gets saved.
        self._last_image_dir = str(Path(path).resolve().parent)
        self._ui_state.setdefault("last_general", {})["last_image_dir"] = self._last_image_dir
        _save_ui_state(self._ui_state)

        self.image_path = path
        self.original_image = Image.open(path)
        self.image_path_label.config(text=path, fg="black")
        self.angle_var.set("0.0")
        self.table_top_var.set("")
        self.table_bottom_var.set("")
        self.table_left_var.set("")
        self.table_right_var.set("")
        self.metadata_bottom_var.set("")
        self.row1_top_var.set("")
        self.row1_bottom_var.set("")
        self.header_box_top_var.set("")
        self.header_box_bottom_var.set("")
        self.column_masks = {}
        self.active_column_var.set("")
        self._mask_click_start = None
        self._cancel_border_click_mode()
        self._refresh_mask_count_label()
        self.last_result = None
        self.save_button.config(state=DISABLED)
        self.next_column_button.config(state=DISABLED)
        self.extract_button.config(state=DISABLED)
        self.column_progress_label.config(text="")

        # Restore per-image geometry if this exact image was worked on
        # before (2026-07-13) - only then, since blindly carrying over
        # geometry from a DIFFERENT image would be actively wrong (this
        # session's whole lesson). Fields above are already reset to
        # blank as the safe default; this only overrides them if a real
        # match exists.
        key = str(Path(path).resolve())
        saved = self._ui_state.get("per_image", {}).get(key)
        if saved:
            self._apply_per_image_settings(saved)
            self.status_label.config(
                text="Image loaded - restored previous settings for this exact image.")
        else:
            self.status_label.config(
                text="Image loaded. Try 'Auto-estimate' for a starting deskew angle, "
                     "then confirm visually.")

        # Resume support (2026-07-22): if a sidecar with real per-column
        # progress already exists for this image, restore it - the
        # actual authoritative record of what's done is the sidecar
        # itself, not the local ui_state cache above (which only ever
        # covers geometry FIELD VALUES/masks, not columns/progress/
        # active_column). This is what lets reopening an interrupted
        # image pick up exactly where it left off, per the resume
        # requirement, without forcing a re-run of 'Refine rows' just
        # to keep masking/extracting forward (see _write_sidecar_state's
        # skip_geometry_rewrite path).
        sidecar_path = OUTPUT_DIR / f"{Path(path).stem}_sidecar.json"
        if sidecar_path.exists():
            try:
                sidecar = load_sidecar(sidecar_path)
            except Exception:
                sidecar = None
            if sidecar and sidecar.get("rows"):
                self.column_order = sidecar.get("column_order", [])
                for name, state in sidecar.get("columns", {}).items():
                    self.column_masks[name] = [tuple(r) for r in state.get("mask_keep_ranges", [])]
                active = sidecar.get("active_column")
                if active:
                    self.active_column_var.set(active)
                    active_state = sidecar["columns"][active]
                    self.mask_apply_header_var.set(active_state.get("mask_apply_header", False))
                    self.mask_apply_rows_var.set(active_state.get("mask_apply_rows", True))
                self._refresh_mask_count_label()
                self._refresh_column_progress_display(sidecar_path)
                # Geometry already exists on disk for this image, so
                # masking/Next/Extract can proceed without a fresh
                # Refine - see _write_sidecar_state's resume path.
                self.save_button.config(state=NORMAL)
                self.next_column_button.config(state=NORMAL)
                self.extract_button.config(state=NORMAL)
                self.status_label.config(
                    text="Image loaded - resumed prior sidecar progress "
                         f"({sidecar.get('progress', {}).get('completed', 0)}/"
                         f"{sidecar.get('progress', {}).get('total', 0)} columns done).")

        # New image - scroll to origin rather than carrying over wherever
        # the PREVIOUS image happened to be scrolled to.
        self.preview_canvas.xview_moveto(0)
        self.preview_canvas.yview_moveto(0)
        self.update_preview()

    def auto_estimate_angle(self):
        if not self.original_image:
            messagebox.showwarning("No image", "Select an image first.")
            return
        angle = estimate_deskew_angle(self.original_image)
        self.angle_var.set(f"{angle:.2f}")
        self.status_label.config(
            text=f"Auto-suggested angle: {angle:.2f}\u00b0. This is a SUGGESTION - "
                 f"check the preview and nudge/edit if the table lines don't look "
                 f"horizontal."
        )
        self.update_preview()

    def nudge_angle(self, step: float):
        new_angle = self._current_angle() + step
        self.angle_var.set(f"{new_angle:.2f}")
        self.update_preview()

    def reset_angle(self):
        self.angle_var.set("0.0")
        self.update_preview()

    def auto_estimate_extent(self):
        if not self.original_image:
            messagebox.showwarning("No image", "Select an image first.")
            return
        deskewed = apply_deskew_angle(self.original_image, self._current_angle())
        x0 = self._parse_optional_int(self.table_left_var)
        x1 = self._parse_optional_int(self.table_right_var)
        top, bottom = estimate_table_extent(deskewed, x0=x0, x1=x1)
        self.table_top_var.set(str(top))
        self.table_bottom_var.set(str(bottom))
        self.status_label.config(
            text=f"Auto-estimated top/bottom: ({top}, {bottom}). Coarse heuristic - "
                 f"check the preview and adjust by hand if it looks wrong."
        )
        self.update_preview()

    def _on_mask_mode_toggle(self):
        self._mask_click_start = None
        if self.mask_mode_var.get():
            self._cancel_border_click_mode()
            self.mask_instruction_label.config(
                text="Click the LEFT edge of the column to keep, then the RIGHT edge.")
        else:
            self.mask_instruction_label.config(text="")

    def load_columns_file(self):
        """
        Loads a one-name-per-line column list (same format as
        run_row_extraction.py's columns.txt / ground_truth_labeling_ui.
        py's "Load columns file...") to populate self.column_order -
        without this, "Next column" has no defined destination beyond
        whatever name is currently typed. If the active image already
        has a saved sidecar, existing per-column state (masks, done
        status) for names in this list is pulled in immediately so
        loading the list doesn't look like it reset progress.
        """
        # Fixed default (2026-07-26, Jon's direction) - config/columns/ is
        # the actual canonical location every real column list lives in
        # (census_1931.txt, census_standard.txt, etc.), unlike Select
        # image's "remember last used" behavior - this one should always
        # start in the same real place, not wherever was last browsed to.
        columns_dir = PROJECT_ROOT / "config" / "columns"
        path = filedialog.askopenfilename(
            title="Select column names file", filetypes=[("Text", "*.txt")],
            initialdir=str(columns_dir) if columns_dir.is_dir() else None,
        )
        if not path:
            return
        names = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        if not names:
            messagebox.showerror("Empty file", "Column file contained no names.")
            return
        self.column_order = names
        if not self.active_column_var.get().strip():
            self.active_column_var.set(names[0])

        if self.image_path:
            sidecar_path = OUTPUT_DIR / f"{Path(self.image_path).stem}_sidecar.json"
            if sidecar_path.exists():
                try:
                    sidecar = load_sidecar(sidecar_path)
                except Exception:
                    sidecar = None
                if sidecar and "columns" in sidecar:
                    for name in names:
                        state = sidecar["columns"].get(name)
                        if state:
                            self.column_masks[name] = [
                                tuple(r) for r in state.get("mask_keep_ranges", [])]
                self._refresh_column_progress_display(sidecar_path)

        self._refresh_mask_count_label()
        self.status_label.config(text=f"Loaded {len(names)} column(s) from {Path(path).name}")

    # Distinct enough at small width/against a mostly-grayscale scan
    # background, and deliberately excludes lime/green (reserved for
    # the active column, so it always reads as visually distinct from
    # every already-defined one).
    _MASK_COLOR_PALETTE = [
        "cyan", "yellow", "orange", "magenta", "#66f", "#f66",
        "#fc0", "#0cf", "#f0c", "white",
    ]

    def _color_for_column(self, name: str) -> str:
        """
        Stable per-column color, keyed by position in column_order when
        the column is part of the loaded list (so it doesn't shift
        around as other columns are added/finished mid-session).
        Columns masked before a columns file was ever loaded (no
        column_order entry) fall back to a hash of the name - still
        stable for that name across redraws, just not coordinated with
        list position.
        """
        if name in self.column_order:
            idx = self.column_order.index(name)
        else:
            idx = abs(hash(name))
        return self._MASK_COLOR_PALETTE[idx % len(self._MASK_COLOR_PALETTE)]

    def _active_column_name(self) -> str | None:
        name = self.active_column_var.get().strip()
        return name or None

    def _active_ranges(self) -> list[tuple[int, int]]:
        """Ranges list for the currently-named column, created empty on
        first access. Returns [] (not stored) if no column is named yet
        - callers that would mutate it should check _active_column_name()
        first and refuse rather than silently masking under an empty key."""
        name = self._active_column_name()
        if name is None:
            return []
        return self.column_masks.setdefault(name, [])

    def _refresh_mask_count_label(self):
        name = self._active_column_name()
        if name is None:
            self.mask_count_label.config(text="(no column selected)")
            return
        n = len(self.column_masks.get(name, []))
        self.mask_count_label.config(
            text=f"{name!r}: {n} kept range(s)" if n
                 else f"{name!r}: 0 kept ranges (everything shown)")

    def _clear_masks(self):
        name = self._active_column_name()
        if name is None:
            messagebox.showwarning("No column selected", "Type a column name first.")
            return
        self.column_masks[name] = []
        self._mask_click_start = None
        self._refresh_mask_count_label()
        self.update_preview()

    # -- click-to-set table bounds (2026-07-26) ------------------------------

    def _start_border_click_mode(self, key: str) -> None:
        """
        Arms crosshair mode for one border field (`key` is one of
        _BORDER_CLICK_BUTTON_LABELS's keys) - the NEXT click on the
        preview canvas sets that field directly from the clicked pixel,
        instead of estimate/type/refresh/repeat. Mutually exclusive
        with column-mask click mode (turning one on turns the other
        off) so a click always has one unambiguous meaning.
        """
        if not self.original_image:
            messagebox.showwarning("No image", "Select an image first.")
            return
        if self.mask_mode_var.get():
            self.mask_mode_var.set(False)
            self._on_mask_mode_toggle()
        self._border_click_mode = key
        self.preview_canvas.config(cursor="crosshair")
        self.border_click_status_label.config(
            text=f"Click the preview to {self._BORDER_CLICK_BUTTON_LABELS[key].lower()} "
                 f"(Esc to cancel)."
        )

    def _cancel_border_click_mode(self) -> None:
        if self._border_click_mode is None:
            return
        self._border_click_mode = None
        self.preview_canvas.config(cursor="")
        self._clear_border_guide()
        self.border_click_status_label.config(text="")

    def _clear_border_guide(self) -> None:
        if self._border_guide_line_id is not None:
            self.preview_canvas.delete(self._border_guide_line_id)
            self._border_guide_line_id = None

    def _on_canvas_motion(self, event) -> None:
        """Live guide line following the mouse while a border-click mode
        is armed - vertical for Left/Right, horizontal for Top/Bottom/
        Metadata bottom/Header box top/bottom. No-op otherwise."""
        if self._border_click_mode is None:
            return
        self._clear_border_guide()
        canvas_x = self.preview_canvas.canvasx(event.x)
        canvas_y = self.preview_canvas.canvasy(event.y)
        w, h = self._preview_render_size
        if self._BORDER_CLICK_ORIENTATION[self._border_click_mode] == "v":
            self._border_guide_line_id = self.preview_canvas.create_line(
                canvas_x, 0, canvas_x, h, fill="red", dash=(4, 2))
        else:
            self._border_guide_line_id = self.preview_canvas.create_line(
                0, canvas_y, w, canvas_y, fill="red", dash=(4, 2))

    def _handle_border_click(self, event) -> None:
        """Reads the clicked pixel back into full deskewed-image
        coordinates (same canvasx/canvasy + self._preview_scale
        conversion _on_canvas_click's mask-click path already uses),
        writes it straight into the target field's StringVar, and
        refreshes the preview - the textbox stays live/editable
        afterward for manual fine-adjustment."""
        canvas_x = self.preview_canvas.canvasx(event.x)
        canvas_y = self.preview_canvas.canvasy(event.y)
        scale = self._preview_scale or 1.0
        key = self._border_click_mode
        if self._BORDER_CLICK_ORIENTATION[key] == "v":
            value = int(round(canvas_x / scale))
        else:
            value = int(round(canvas_y / scale))
        getattr(self, f"{key}_var").set(str(value))
        label = self._BORDER_CLICK_BUTTON_LABELS[key]
        self._cancel_border_click_mode()
        self.status_label.config(text=f"{label} → {value} (click-set). Preview refreshed.")
        self.update_preview()

    def _on_canvas_click(self, event):
        """
        Click-to-keep: first click marks the left edge of a column to
        keep, second click marks the right edge - the pair is appended
        to column_masks[active column name] (in FULL deskewed-image
        coordinates, via self._preview_scale which _render_preview()
        keeps up to date for exactly this purpose, and canvas.canvasx()/
        canvasy() which correctly account for the current scroll
        position, not just the click's raw widget-relative position).
        """
        if self._border_click_mode is not None:
            self._handle_border_click(event)
            return
        if not self.mask_mode_var.get():
            return
        if self._active_column_name() is None:
            messagebox.showwarning("No column selected",
                                    "Type a column name in the 'Column:' box before clicking.")
            self.mask_mode_var.set(False)
            self._on_mask_mode_toggle()
            return
        canvas_x = self.preview_canvas.canvasx(event.x)
        image_x = int(canvas_x / self._preview_scale) if self._preview_scale else int(canvas_x)

        if self._mask_click_start is None:
            self._mask_click_start = image_x
            self.mask_instruction_label.config(
                text=f"Left edge set at x={image_x}. Click the RIGHT edge now.")
        else:
            x0, x1 = sorted([self._mask_click_start, image_x])
            if x1 > x0:
                self._active_ranges().append((x0, x1))
                self._refresh_mask_count_label()
            self._mask_click_start = None
            self.mask_instruction_label.config(
                text="Click the LEFT edge of the next range to keep, "
                     "or uncheck to stop.")
            self.update_preview()

    def update_preview(self):
        """
        CHEAP preview: rotate at the current angle + draw the table/
        header bounds as simple rectangles. No Otsu, no ruling-line
        detection, no periodic refinement - fast enough to call on
        every nudge-button click or bounds-field edit. This is the core
        of the interactive workflow: see the effect of an adjustment
        immediately, without waiting for a full detection run.
        """
        if not self.original_image:
            return
        deskewed = apply_deskew_angle(self.original_image, self._current_angle())

        # Preprocessing (2026-07-25): every filter EXCEPT upscale changes
        # pixel VALUES only, so it's safe to apply before the overlay
        # annotations below, which are drawn using bound-field
        # coordinates in the ORIGINAL deskewed image's pixel space.
        # Upscale changes canvas DIMENSIONS - applying it here too would
        # misalign every rectangle/line drawn below (they'd need their
        # own separate coordinate scaling). Applied instead AFTER
        # annotations are drawn, at the very end, so the whole annotated
        # overlay scales together and stays aligned with zero extra math.
        preproc_config = self._current_preprocessing_config()
        upscale_config = preproc_config.pop("upscale")
        content = apply_pipeline(deskewed, preproc_config)
        overlay = content.convert("RGB").copy()
        # Captured BEFORE the optional upscale step below reassigns
        # `overlay` - every bounds field (table_top_var etc.) and every
        # column mask range is defined in THIS pre-upscale pixel space,
        # so _render_preview must compute self._preview_scale relative
        # to this width, not whatever (possibly 2x/4x larger) width the
        # image ends up at after upscaling. See _render_preview's
        # reference_width param docstring for the bug this fixes.
        reference_width = overlay.width
        draw = ImageDraw.Draw(overlay)

        x0 = self._parse_optional_int(self.table_left_var)
        x1 = self._parse_optional_int(self.table_right_var)
        top = self._parse_optional_int(self.table_top_var)
        bottom = self._parse_optional_int(self.table_bottom_var)
        left = x0 if x0 is not None else 0
        right = x1 if x1 is not None else overlay.width

        if x0 is not None or x1 is not None:
            draw.rectangle([left, 0, right, overlay.height], outline="purple", width=2)
        if top is not None and bottom is not None:
            draw.rectangle([left, top, right, bottom], outline="orange", width=3)

        metadata_bottom = self._parse_optional_int(self.metadata_bottom_var)
        if metadata_bottom is not None:
            draw.line([0, metadata_bottom, overlay.width, metadata_bottom],
                      fill="cyan", width=3)
            draw.text((5, max(0, metadata_bottom - 16)), "METADATA / HEADINGS SPLIT",
                      fill="cyan")

        row1_top = self._parse_optional_float(self.row1_top_var)
        row1_bottom = self._parse_optional_float(self.row1_bottom_var)
        if row1_top is not None and row1_bottom is not None:
            draw.rectangle([left, round(row1_top), right, round(row1_bottom)],
                            outline="yellow", width=3)
            draw.text((left + 5, round(row1_top) + 2), "ROW 1 (confirm this, then tile)",
                      fill="yellow")

        header_box_top = self._parse_optional_int(self.header_box_top_var)
        header_box_bottom = self._parse_optional_int(self.header_box_bottom_var)
        if header_box_top is not None and header_box_bottom is not None:
            draw.rectangle([left, header_box_top, right, header_box_bottom],
                            outline="magenta", width=3)
            draw.text((left + 5, header_box_top + 2),
                      "HEADER BOX (exact strip prepended to every row)",
                      fill="magenta")

        # Kept-column ranges (2026-07-15, extended 2026-07-22 to draw
        # EVERY defined column at once, not just the active one) -
        # drawn as vertical bands spanning the full image height, so
        # it's visually obvious these apply across both the header area
        # and the row area regardless of which scope checkboxes are
        # currently on (scope only affects what gets PASSED to
        # extraction, not what's shown here).
        #
        # Each column gets a STABLE color (keyed by its position in
        # column_order when known, so the same column keeps the same
        # color across a whole session even as others are added/
        # finished - a color that kept shifting around would make the
        # progress readout and the preview disagree about identity).
        # Non-active columns are drawn thin/dim first; the active
        # column is drawn last, in bright lime with its scope label, so
        # it's unambiguous which one clicking will currently edit -
        # this was a real gap in the single-column-only version (no
        # way to see previously-masked columns while working on a new
        # one, easy to accidentally re-mask over something already
        # done).
        scope_bits = []
        if self.mask_apply_header_var.get():
            scope_bits.append("header")
        if self.mask_apply_rows_var.get():
            scope_bits.append("rows")
        scope_label = "+".join(scope_bits) if scope_bits else "NEITHER SCOPE ON - no effect"
        active_name = self._active_column_name()

        for col_name, ranges in self.column_masks.items():
            if col_name == active_name or not ranges:
                continue
            color = self._color_for_column(col_name)
            for kx0, kx1 in ranges:
                draw.rectangle([kx0, 0, kx1, overlay.height], outline=color, width=1)
            # One label per column (at its first range) rather than per
            # range - avoids label clutter when a column has several
            # kept ranges, while the colored rectangles still mark each
            # range individually.
            first_x0 = ranges[0][0]
            draw.text((first_x0 + 3, 4), col_name, fill=color)

        if active_name is not None:
            for i, (kx0, kx1) in enumerate(self.column_masks.get(active_name, [])):
                draw.rectangle([kx0, 0, kx1, overlay.height], outline="lime", width=2)
                draw.text((kx0 + 3, 4), f"{active_name} {i+1} ({scope_label}) [ACTIVE]",
                          fill="lime")
        if self._mask_click_start is not None:
            draw.line([self._mask_click_start, 0, self._mask_click_start, overlay.height],
                       fill="lime", width=2)

        if upscale_config.get("enabled"):
            from core.image_preprocessing import upscale as _upscale_step
            overlay = _upscale_step(overlay, factor=upscale_config.get("factor", 2.0))

        self._render_preview(overlay, reference_width=reference_width)

    def _render_preview(self, overlay: Image.Image, reference_width: int | None = None):
        """
        Shared rendering for both the cheap live preview and the post-
        refine full overlay.

        `reference_width` (2026-07-26 bugfix): the pixel-space width
        that bounds fields/mask ranges/click-set coordinates actually
        mean. Defaults to overlay.width (correct for refine_rows' call
        site below, whose overlay is never upscaled). update_preview()'s
        call site passes the PRE-upscale width explicitly, because when
        Upscale preprocessing is on (the default), `overlay` here is
        ALREADY the upscaled image - using its width would make
        self._preview_scale (and therefore every click-derived
        coordinate: border click-set, column-mask clicking, the guide
        line) silently inflated by the upscale factor relative to what
        the textboxes/segment_rows_* functions actually operate on. This
        is exactly the bug Jon hit: Bottom/Right values came out ~2x too
        large after clicking with the (default-on) 2.0x Upscale enabled.

        REWRITTEN 2026-07-13 - the original version cropped a narrow
        strip auto-centered on whatever boundary field seemed relevant
        and reset the canvas to it on every refresh. That's a locked,
        repeatedly-resetting viewport, not free scrolling - it fought
        the actual use case (looking at whatever part of the page you
        actually want, not just the one field currently being edited)
        and any interaction that triggered a refresh yanked the view
        back, making it impossible to manually scroll elsewhere and
        stay there. Fixed properly: zoom now renders the WHOLE page
        scaled up (not a cropped strip), and scroll position is saved
        before re-rendering and restored after, so refreshing a field
        no longer resets where you've scrolled to.
        """
        if reference_width is None:
            reference_width = overlay.width

        zoom = self.zoom_var.get()
        if zoom == "Fit":
            preview = overlay.copy()
            preview.thumbnail(PREVIEW_SIZE)
        else:
            factor = int(zoom.rstrip("x"))
            # Whole page, not a cropped strip - NEAREST keeps pixel/line
            # edges crisp rather than blurring them, useful for
            # precisely judging where a printed line actually sits.
            preview = overlay.resize(
                (overlay.width * factor, overlay.height * factor), Image.NEAREST
            )

        # Tracked for click-to-mask coordinate inversion (2026-07-15) -
        # works uniformly for both Fit (thumbnail) and zoomed (explicit
        # factor) modes, since both just produce SOME preview.width
        # relative to reference_width (the PRE-upscale width - see this
        # method's docstring).
        self._preview_scale = preview.width / reference_width if reference_width else 1.0
        self._preview_render_size = (preview.width, preview.height)

        # Save current scroll position (fractional) before tearing down
        # canvas content, restore it after - this is what actually
        # fixes the "locks/resets" complaint.
        try:
            x_frac = self.preview_canvas.xview()[0]
            y_frac = self.preview_canvas.yview()[0]
        except Exception:
            x_frac = y_frac = 0.0

        self._tk_preview = ImageTk.PhotoImage(preview)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(0, 0, anchor="nw", image=self._tk_preview)
        self.preview_canvas.configure(scrollregion=(0, 0, preview.width, preview.height))
        self.preview_canvas.xview_moveto(x_frac)
        self.preview_canvas.yview_moveto(y_frac)

    # -- the expensive step ---------------------------------------------------

    def refine_rows(self):
        """
        The one EXPENSIVE step (ruling-line detection + periodic
        refinement, or general detection), using whatever angle/bounds
        the user has confirmed via the preview by this point. Locks in
        the current angle (deskew_angle=...) rather than letting this
        step silently re-estimate its own - see segment_rows_periodic's
        docstring for why that distinction matters.
        """
        if not self.image_path or not self.original_image:
            messagebox.showwarning("No image", "Select an image first.")
            return

        try:
            header_rows = int(self.header_rows_var.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Header rows must be an integer.")
            return

        angle = self._current_angle()
        mode = self.mode_var.get()
        self.status_label.config(text="Running detection (no model calls)...")
        self.root.update_idletasks()

        try:
            if mode == "periodic":
                try:
                    row_count = int(self.row_count_var.get())
                    search_radius_ratio = float(self.search_radius_var.get())
                except ValueError:
                    messagebox.showerror("Invalid input",
                                          "Row count must be an integer, search radius "
                                          "ratio must be a number.")
                    return
                table_top = self._parse_optional_int(self.table_top_var)
                table_bottom = self._parse_optional_int(self.table_bottom_var)
                table_left = self._parse_optional_int(self.table_left_var)
                table_right = self._parse_optional_int(self.table_right_var)

                result, row_crops, header_crop, overlay = segment_rows_periodic(
                    self.original_image, row_count=row_count,
                    table_top=table_top, table_bottom=table_bottom,
                    table_left=table_left, table_right=table_right,
                    search_radius_ratio=search_radius_ratio,
                    header_row_count=header_rows, deskew_angle=angle,
                    metadata_bottom=self._parse_optional_int(self.metadata_bottom_var),
                    **self._current_header_box_kwargs(),
                    **self._current_padding_kwargs(),
                )
                # Write back the actual extent used, same UX principle as
                # before - if auto-estimated, the confirmed value becomes
                # visible/editable rather than staying hidden.
                if table_top is None or table_bottom is None:
                    deskewed = apply_deskew_angle(self.original_image, angle)
                    auto_top, auto_bottom = estimate_table_extent(
                        deskewed, x0=table_left, x1=table_right)
                    self.table_top_var.set(str(table_top if table_top is not None else auto_top))
                    self.table_bottom_var.set(
                        str(table_bottom if table_bottom is not None else auto_bottom))
            elif mode == "uniform_tile":
                try:
                    row_count = int(self.row_count_var.get())
                except ValueError:
                    messagebox.showerror("Invalid input", "Row count must be an integer.")
                    return
                row1_top = self._parse_optional_float(self.row1_top_var)
                row1_bottom = self._parse_optional_float(self.row1_bottom_var)
                if row1_top is None or row1_bottom is None:
                    messagebox.showerror(
                        "Row 1 not confirmed",
                        "uniform_tile mode needs Row 1 top/bottom set - use "
                        "Refresh preview to confirm the yellow box lines up "
                        "with the first real row before running this.")
                    return
                table_left = self._parse_optional_int(self.table_left_var)
                table_right = self._parse_optional_int(self.table_right_var)

                result, row_crops, header_crop, overlay = segment_rows_uniform_tile(
                    self.original_image, row_count=row_count,
                    first_row_top=row1_top, first_row_bottom=row1_bottom,
                    table_left=table_left, table_right=table_right,
                    header_row_count=header_rows, deskew_angle=angle,
                    metadata_bottom=self._parse_optional_int(self.metadata_bottom_var),
                    **self._current_header_box_kwargs(),
                    **self._current_padding_kwargs(),
                )
            else:
                table_top = self._parse_optional_int(self.table_top_var)
                table_bottom = self._parse_optional_int(self.table_bottom_var)
                table_left = self._parse_optional_int(self.table_left_var)
                table_right = self._parse_optional_int(self.table_right_var)

                result, row_crops, header_crop, overlay = segment_rows(
                    self.original_image, header_row_count=header_rows,
                    table_top=table_top, table_bottom=table_bottom,
                    table_left=table_left, table_right=table_right,
                    deskew_angle=angle,
                    metadata_bottom=self._parse_optional_int(self.metadata_bottom_var),
                    **self._current_header_box_kwargs(),
                    **self._current_padding_kwargs(),
                )
                # Same write-back UX as periodic mode - if table_top/bottom
                # were auto-estimated (left blank), surface the confirmed
                # value rather than leaving it hidden.
                if table_top is None or table_bottom is None:
                    deskewed = apply_deskew_angle(self.original_image, angle)
                    auto_top, auto_bottom = estimate_table_extent(
                        deskewed, x0=table_left, x1=table_right)
                    self.table_top_var.set(str(table_top if table_top is not None else auto_top))
                    self.table_bottom_var.set(
                        str(table_bottom if table_bottom is not None else auto_bottom))
        except Exception as e:
            self.status_label.config(text=f"ERROR: {e}")
            messagebox.showerror("Segmentation failed", str(e))
            return

        self.last_result = result
        self.last_row_crops = row_crops
        self.save_button.config(state=NORMAL)
        self.next_column_button.config(state=NORMAL)
        self.extract_button.config(state=NORMAL)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        name = Path(self.image_path).stem
        overlay_path = OUTPUT_DIR / f"{name}_debug_overlay.png"
        overlay.save(overlay_path)
        self._last_overlay_path = str(overlay_path)
        self.open_overlay_button.config(state=NORMAL)

        self._render_preview(overlay)

        if self.debug_crops_var.get():
            for i, crop in enumerate(row_crops, start=1):
                crop.save(OUTPUT_DIR / f"{name}_row_{i:03d}.png")

        self.report_text.delete("1.0", END)
        self.report_text.insert(END, f"Mode: {mode}\n")
        self.report_text.insert(END, f"Rows detected: {len(result.bands)}\n")
        if result.header_band:
            self.report_text.insert(END, "Header region: used\n")
        self.report_text.insert(END, f"Dropped bands: {len(result.dropped_bands)}\n\n")
        self.report_text.insert(END, "Warnings / diagnostics:\n")
        for w in result.warnings:
            self.report_text.insert(END, f"  - {w}\n")

        # Persist settings (2026-07-13) after every successful refine,
        # not just final Save - work-in-progress survives a crash or
        # accidental close, not just a completed sidecar save.
        self._save_current_settings()

        self.status_label.config(
            text=f"Refined: {len(result.bands)} rows. Not saved yet - click "
                 f"'Save sidecar JSON' once you're happy with this result."
        )

    def _safe_column_order_seed(self, active_name: str) -> list[str]:
        """
        What to pass as update_sidecar()'s column_order= argument.

        Normally self.column_order (the real loaded list) if available,
        else [active_name] - a reasonable seed for a genuinely new
        sidecar's first-ever column.

        CRITICAL EXCEPTION for "__multi__" (2026-07-22): that name is a
        pseudo-column, a mask-definition scratchpad for two-stage/
        legacy multi-column extraction - it is NEVER a real progression
        step. If self.column_order were empty and this fell through to
        [active_name] = ["__multi__"], update_sidecar()'s
        init_column_state() would overwrite the sidecar's REAL
        column_order (if one already exists on disk from a prior
        session that this UI session just hasn't reloaded) down to
        that single bogus entry - a real, silent, destructive bug
        distinct from the one this whole __multi__ feature exists to
        fix. Returning [] here instead means update_sidecar() only
        touches columns["__multi__"] itself (via its own setdefault)
        without ever touching column_order - safe regardless of
        whether the real column list happens to be loaded in THIS
        session or not. If the sidecar has no "columns" scaffold at
        all yet (a truly brand-new sidecar), update_sidecar() will
        raise a clear error asking for a real column_order first -
        correct behavior, since there's no real progression list to
        infer from a mask-only pseudo-column on a sidecar with nothing
        else in it yet.
        """
        if active_name == "__multi__":
            return self.column_order or []
        return self.column_order or [active_name]

    def use_multi_mask(self):
        """Convenience for the two-stage/legacy multi-column mask -
        sets the active column name to the reserved "__multi__"
        pseudo-column rather than requiring it to be typed by hand
        (a typo here, e.g. "__muti__", would silently create a
        pointless new real-looking column instead of touching the
        one core.row_extraction._compute_scoped_masks() actually
        looks for)."""
        self.active_column_var.set("__multi__")
        name = self._active_column_name()
        if name and name not in self.column_masks:
            self.column_masks[name] = []
        self._refresh_mask_count_label()
        self.update_preview()

    def _write_sidecar_state(self, active_name: str | None) -> Path | None:
        """
        Shared by save_result()/next_column()/extract_active_column().
        Two paths:

        1. A fresh 'Refine rows' result exists (self.last_result) -
           rebuilds page-level geometry from it and merges into any
           existing sidecar (preserving per-column state), same as
           before.
        2. RESUME case (2026-07-22): no fresh refine this session, but
           image_path is set and a sidecar for it already exists on
           disk with real geometry ("rows" present) - e.g. reopening an
           image from a prior session per the resume-support
           requirement. Rather than reconstructing a RowDetectionResult
           from the saved bboxes and re-running build_sidecar() (which
           would re-apply row padding on top of ALREADY-padded stored
           bboxes - a real double-padding bug, not just redundant work),
           geometry is left untouched entirely and only the per-column
           patch below is written. This means resuming to mask/extract
           further columns never requires re-running detection just to
           move forward - only changing the geometry itself does.

        Returns the sidecar path, or None if neither path applies
        (caller has already shown the relevant warning in that case).
        """
        if not self.image_path:
            messagebox.showwarning("Nothing to save", "Load an image first.")
            return None

        skip_geometry_rewrite = False
        if not self.last_result:
            name = Path(self.image_path).stem
            candidate = OUTPUT_DIR / f"{name}_sidecar.json"
            if candidate.exists():
                try:
                    existing_check = load_sidecar(candidate)
                except Exception:
                    existing_check = {}
                if existing_check.get("rows"):
                    skip_geometry_rewrite = True
                    sidecar_path = candidate
            if not skip_geometry_rewrite:
                messagebox.showwarning(
                    "Nothing to save",
                    "Run 'Refine rows' first (no fresh geometry this session, and no "
                    "existing sidecar with geometry was found to resume from).")
                return None

        if skip_geometry_rewrite:
            if active_name is not None:
                update_sidecar(
                    str(sidecar_path), active_name,
                    {
                        "status": "in_progress",
                        "mask_keep_ranges": [list(r) for r in self.column_masks.get(active_name, [])],
                        "mask_apply_header": self.mask_apply_header_var.get(),
                        "mask_apply_rows": self.mask_apply_rows_var.get(),
                    },
                    column_order=self._safe_column_order_seed(active_name),
                )
            update_sidecar_preprocessing(str(sidecar_path), self._current_preprocessing_config())
            return sidecar_path

        mode = self.mode_var.get()
        if mode == "periodic":
            parameters = {
                "row_count": int(self.row_count_var.get()),
                "table_top": self._parse_optional_int(self.table_top_var),
                "table_bottom": self._parse_optional_int(self.table_bottom_var),
                "table_left": self._parse_optional_int(self.table_left_var),
                "table_right": self._parse_optional_int(self.table_right_var),
                "metadata_bottom": self._parse_optional_int(self.metadata_bottom_var),
                "search_radius_ratio": float(self.search_radius_var.get()),
                "header_row_count": int(self.header_rows_var.get()),
            }
            sidecar_table_top = self._parse_optional_int(self.table_top_var)
            sidecar_table_bottom = self._parse_optional_int(self.table_bottom_var)
        elif mode == "uniform_tile":
            row1_top = self._parse_optional_float(self.row1_top_var)
            row1_bottom = self._parse_optional_float(self.row1_bottom_var)
            row_count = int(self.row_count_var.get())
            parameters = {
                "row_count": row_count,
                "first_row_top": row1_top,
                "first_row_bottom": row1_bottom,
                "table_left": self._parse_optional_int(self.table_left_var),
                "table_right": self._parse_optional_int(self.table_right_var),
                "metadata_bottom": self._parse_optional_int(self.metadata_bottom_var),
                "header_row_count": int(self.header_rows_var.get()),
            }
            # table_bbox needs to span the FULL tiled table (not just row 1)
            # so header extraction's table_bbox[1] usage still works
            # correctly - computed from the same tiling math as
            # segment_rows_uniform_tile itself.
            if row1_top is not None and row1_bottom is not None:
                row_height = row1_bottom - row1_top
                sidecar_table_top = round(row1_top)
                sidecar_table_bottom = round(row1_top + row_count * row_height)
            else:
                sidecar_table_top = sidecar_table_bottom = None
        else:
            parameters = {"header_row_count": int(self.header_rows_var.get())}
            sidecar_table_top = self._parse_optional_int(self.table_top_var)
            sidecar_table_bottom = self._parse_optional_int(self.table_bottom_var)

        sidecar_geometry = build_sidecar(
            self.last_result, source_image_path=str(Path(self.image_path).resolve()),
            mode=mode, parameters=parameters,
            table_top=sidecar_table_top,
            table_bottom=sidecar_table_bottom,
            x0=self._parse_optional_int(self.table_left_var),
            x1=self._parse_optional_int(self.table_right_var),
            metadata_bottom=self._parse_optional_int(self.metadata_bottom_var),
            # These top-level mask_* fields are legacy/unused going
            # forward - real per-column masks live under
            # sidecar["columns"][name], written below via
            # update_sidecar(). Left at defaults here only so
            # build_sidecar()'s signature doesn't need to change.
            **self._current_padding_kwargs(),
        )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        name = Path(self.image_path).stem
        sidecar_path = OUTPUT_DIR / f"{name}_sidecar.json"

        # Geometry (rows/table_bbox/etc.) legitimately gets replaced by
        # every confirmed 'Refine rows' + Save - that's a real new
        # spatial contract, not incremental column progress. But if a
        # sidecar already exists here with per-column state (columns/
        # column_order/active_column/progress), that state must survive
        # the geometry refresh - only the page-level keys from the
        # fresh build_sidecar() output get applied on top.
        if sidecar_path.exists():
            try:
                existing = load_sidecar(sidecar_path)
            except Exception:
                existing = {}
            merged = dict(existing)
            merged.update(sidecar_geometry)  # page-level keys only overwrite page-level keys
            for preserved_key in ("columns", "column_order", "active_column", "progress"):
                if preserved_key in existing:
                    merged[preserved_key] = existing[preserved_key]
            save_sidecar(merged, sidecar_path)
        else:
            save_sidecar(sidecar_geometry, sidecar_path)

        if active_name is not None:
            update_sidecar(
                str(sidecar_path), active_name,
                {
                    "status": "in_progress",
                    "mask_keep_ranges": [list(r) for r in self.column_masks.get(active_name, [])],
                    "mask_apply_header": self.mask_apply_header_var.get(),
                    "mask_apply_rows": self.mask_apply_rows_var.get(),
                },
                # Prefer the full loaded list (so auto-advance has
                # somewhere real to go); fall back to just this one
                # name if no columns file was ever loaded for this
                # session.
                column_order=self._safe_column_order_seed(active_name),
            )

        update_sidecar_preprocessing(str(sidecar_path), self._current_preprocessing_config())
        return sidecar_path

    def save_result(self):
        """
        Saves the CONFIRMED spatial contract - only reachable after at
        least one 'Refine rows' run, so a sidecar is never written from
        unconfirmed/never-run parameters.
        """
        active_name = self._active_column_name()
        if active_name is None and self.column_masks:
            messagebox.showwarning(
                "No active column selected",
                "Type a column name in the 'Column:' box (matching one already masked, "
                "or a new one) before saving, so this mask is attributed correctly.")
            return

        sidecar_path = self._write_sidecar_state(active_name)
        if sidecar_path is None:
            return

        self._refresh_column_progress_display(sidecar_path)
        self.status_label.config(text=f"Saved: {sidecar_path}")
        messagebox.showinfo("Saved", f"Sidecar JSON saved:\n{sidecar_path}")

    def next_column(self):
        """
        Saves the active column's current mask (marking it "done"),
        auto-advances to the next pending column in column_order, and -
        if that next column already has stored mask geometry from a
        prior session - loads it back into column_masks so the operator
        sees it immediately instead of starting from an empty mask.
        This is the step that removes the manual remask cycle: masking
        is only required again if the operator wants to ADJUST what was
        auto-restored, not just to move forward.
        """
        active_name = self._active_column_name()
        if active_name is None:
            messagebox.showwarning(
                "No active column selected",
                "Type the column name you're currently masking before advancing.")
            return
        if active_name == "__multi__":
            messagebox.showwarning(
                "Not a real column",
                "\"__multi__\" is the mask-definition slot for two-stage/legacy "
                "multi-column extraction, not a real extraction target - it has no "
                "place in the column progression. Use 'Save sidecar JSON' to save "
                "its mask, then switch back to a real column name to continue.")
            return
        if not self.column_masks.get(active_name):
            proceed = messagebox.askyesno(
                "No mask defined",
                f"No kept ranges are defined for {active_name!r} yet - the whole row "
                f"will be extracted unmasked. Mark it done and advance anyway?")
            if not proceed:
                return

        sidecar_path = self._write_sidecar_state(active_name)
        if sidecar_path is None:
            return

        sidecar = update_sidecar(str(sidecar_path), active_name, {"status": "done"})
        self._sync_ui_to_active_column(sidecar, sidecar_path, prior_name=active_name)

    def _sync_ui_to_active_column(self, sidecar: dict, sidecar_path, prior_name: str | None = None) -> None:
        """
        Shared by next_column() and the post-extraction handler: given a
        freshly-written sidecar dict, moves the UI's active-column state
        (name entry, column_masks, scope checkboxes, preview, progress
        readout) to match whatever the sidecar now says active_column
        is - restoring that column's stored mask if it already has one,
        same behavior regardless of whether the advance was triggered by
        masking or by a completed extraction pass.
        """
        next_name = sidecar.get("active_column")
        self._refresh_column_progress_display(sidecar_path)

        if next_name is None:
            # BUG FIX (2026-07-22, found via a real reverted column):
            # previously left self.active_column_var showing whatever
            # column had JUST been completed. A stray extra click on
            # Save afterward would then silently re-patch that column
            # with status: "in_progress", reverting it - confirmed
            # exactly this sequence happened on a real sidecar (Next
            # completed the last column -> Save clicked again out of
            # habit -> that column's status flipped back). Clearing the
            # field is enough to make that impossible: every write
            # handler (save_result/next_column/extract_active_column)
            # already warns-and-refuses on an empty active column name,
            # so this alone closes the gap without needing to disable
            # the buttons - typing a column name back in to
            # intentionally redo one still works exactly as before.
            self.active_column_var.set("")
            self.status_label.config(text="All configured columns are done.")
            messagebox.showinfo("Processing complete",
                                 "Every configured column for this image is marked done.")
            return

        self.active_column_var.set(next_name)
        next_state = sidecar["columns"][next_name]
        self.column_masks[next_name] = [tuple(r) for r in next_state.get("mask_keep_ranges", [])]
        self.mask_apply_header_var.set(next_state.get("mask_apply_header", False))
        self.mask_apply_rows_var.set(next_state.get("mask_apply_rows", True))
        self._mask_click_start = None
        self._refresh_mask_count_label()
        self.update_preview()
        prefix = f"{prior_name!r} marked done. " if prior_name else ""
        self.status_label.config(
            text=f"{prefix}Now on {next_name!r}"
                 + (" (restored prior mask)." if next_state.get("mask_keep_ranges") else "."))

    def extract_active_column(self):
        """
        Runs core.row_extraction.run_single_column_extraction for the
        active column, on a background thread (model load + N rows of
        inference genuinely takes a while - blocking the Tk mainloop
        for that long would freeze the whole window). The extraction
        module is imported lazily INSIDE the worker thread, not at file
        top, so this UI still opens fine on a machine with no torch/
        model stack installed - that's only needed at the moment
        extraction is actually run, not to do masking work.
        """
        if self._extraction_running:
            messagebox.showinfo("Already running", "An extraction is already in progress.")
            return

        active_name = self._active_column_name()
        if active_name is None:
            messagebox.showwarning("No active column selected",
                                    "Type/select the column to extract first.")
            return
        if active_name == "__multi__":
            messagebox.showwarning(
                "Not a real column",
                "\"__multi__\" is a mask-definition slot, not a real extraction "
                "target - run_single_column_extraction would ask the model for a "
                "field literally named '__multi__', which doesn't exist. Use "
                "run_two_stage_extraction.py (or workflow_gui.py's Two-Stage "
                "Extraction panel) to actually run multi-column extraction using "
                "this mask.")
            return
        model_name = self.model_profile_var.get().strip()
        if not model_name:
            messagebox.showwarning("No model profile", "Enter a model profile name "
                                    "(matching a config/models/ entry) first.")
            return

        # Save current mask state first, same as a manual Save, so the
        # extraction runs against exactly what's on screen right now,
        # not stale sidecar state from before this masking session.
        sidecar_path = self._write_sidecar_state(active_name)
        if sidecar_path is None:
            return

        self._extraction_running = True
        self.extract_button.config(state=DISABLED)
        self.next_column_button.config(state=DISABLED)
        self.extract_status_label.config(text=f"Extracting {active_name!r}... (model loading)")

        debug_model_inputs = self.debug_model_inputs_var.get()
        thread = threading.Thread(
            target=self._extraction_worker,
            args=(str(sidecar_path), model_name, active_name, debug_model_inputs),
            daemon=True,
        )
        thread.start()
        self.root.after(200, self._poll_extraction_queue)

    def _extraction_worker(
        self, sidecar_path: str, model_name: str, column_name: str,
        debug_model_inputs: bool = False,
    ) -> None:
        """Runs off the main thread - must not touch any Tk widget
        directly, only put results on the queue for the main thread
        (_poll_extraction_queue) to apply."""
        try:
            from core.row_extraction import run_single_column_extraction
            from core.debug_dump import DebugModelInputRecorder
            debug_recorder = DebugModelInputRecorder(
                enabled=debug_model_inputs, source_input=sidecar_path,
            )
            results = run_single_column_extraction(
                sidecar_path, model_name, column_name=column_name, mark_done=True,
                debug_recorder=debug_recorder)
            if debug_recorder.enabled:
                print(f"Debug model-input capture: ON -> {debug_recorder.run_dir}")
            debug_recorder.close()
            self._extraction_queue.put(("ok", sidecar_path, column_name, len(results)))
        except Exception as e:
            self._extraction_queue.put(("error", sidecar_path, column_name, str(e)))

    def _poll_extraction_queue(self):
        try:
            outcome = self._extraction_queue.get_nowait()
        except queue.Empty:
            self.root.after(200, self._poll_extraction_queue)
            return

        status, sidecar_path, column_name, payload = outcome
        self._extraction_running = False
        self.extract_button.config(state=NORMAL)
        self.next_column_button.config(state=NORMAL)

        if status == "error":
            self.extract_status_label.config(text=f"Extraction of {column_name!r} FAILED.")
            messagebox.showerror("Extraction failed", payload)
            return

        self.extract_status_label.config(
            text=f"Extraction of {column_name!r} complete ({payload} rows).")
        sidecar = load_sidecar(sidecar_path)
        # run_single_column_extraction already marked the column "done"
        # and (via update_sidecar's auto-advance) moved active_column
        # forward - reuse the same sync path Next-column uses so the UI
        # ends up in the identical state either way.
        self._sync_ui_to_active_column(sidecar, sidecar_path, prior_name=column_name)

    def _refresh_column_progress_display(self, sidecar_path) -> None:
        """Renders the '\u2713 Surname \u2713 Age \u25b6 Occupation \u25a1 Birthplace...'
        line from the sidecar's column_order/columns/active_column -
        purely a readout of what's on disk, no separate state kept."""
        try:
            sidecar = load_sidecar(sidecar_path)
        except Exception:
            return
        order = sidecar.get("column_order", [])
        if not order:
            self.column_progress_label.config(text="")
            return
        active = sidecar.get("active_column")
        columns = sidecar.get("columns", {})
        parts = []
        for name in order:
            status = columns.get(name, {}).get("status", "pending")
            if name == active:
                parts.append(f"\u25b6 {name}")
            elif status == "done":
                parts.append(f"\u2713 {name}")
            else:
                parts.append(f"\u25a1 {name}")
        progress = sidecar.get("progress", {})
        self.column_progress_label.config(
            text=" ".join(parts) +
                 f"   ({progress.get('completed', 0)}/{progress.get('total', len(order))} done)")

    def open_overlay_in_viewer(self):
        if not self._last_overlay_path:
            return
        try:
            system = platform.system()
            if system == "Windows":
                import os
                os.startfile(self._last_overlay_path)  # noqa: S606
            elif system == "Darwin":
                subprocess.run(["open", self._last_overlay_path], check=True)
            else:
                subprocess.run(["xdg-open", self._last_overlay_path], check=True)
        except Exception as e:
            messagebox.showerror("Could not open overlay", str(e))


if __name__ == "__main__":
    root = Tk()
    app = RowSegmentationApp(root)
    root.mainloop()
