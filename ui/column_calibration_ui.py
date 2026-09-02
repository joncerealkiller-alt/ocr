"""
Column Calibration UI - correct COLUMN geometry only, per Jon's 2026-08-07
"Column Calibration / Sidecar Correction UI" spec.

Deliberately does NOT touch or replace ui/row_segmentation_ui.py (the full
manual sidecar tool - requires table bounds/rows/regions/column names
before it will save anything) or ui/quarantine_review_ui.py (whole-page
re-review after automated validation failure). This tool exists because
those are too heavyweight for a column-only calibration pass:
core.auto_sidecar.locate_columns() already proposes column geometry for
most pages, and the residual-error research this tool feeds (docs/
COLUMN_NUMBER_ANCHOR_RESEARCH.md, locate_columns()'s own resolution
cascade) only needs a human confirming/correcting that proposal, not
re-deriving table bounds/rows/column names from scratch.

Out of scope, always: table bounds, rows, header/metadata boxes. Those
come from the loaded sidecar (core/auto_sidecar.py's output) and are
never re-entered here. Column NAMES normally come the same way (via the
doc_type's template, core.document_templates.load_template) but CAN be
overridden per-session via "Load columns file..." (added 2026-08-07,
per Jon) - for a page whose real doc_type has no template YAML yet (a
document layout this pipeline doesn't have a full template for), this
lets a human pick the correct config/columns/*.txt directly instead of
being stuck with whatever column list the CV-classifier's best-guess
template happened to use. Still never re-derives table bounds/rows -
only which NAMES apply to the existing table_bbox changes.

Boundary model (see core/column_calibration.py's module docstring for the
full reasoning): columns are ordered left-to-right and share DIVIDING
LINES - N-1 interior dividers for N columns. The table's own outer edges
are drawn for context only and are not interactive here.

Never overwrites the source sidecar. Saves a separate correction record
via core.column_calibration (data/outputs/column_calibration/
{stem}_correction.json). "Export merged sidecar" writes a NEW file at a
user-chosen path - it never touches the original sidecar either.

Header-number anchor: manual placement/nudge only in this version. If a
data/outputs/column_number_anchors/{stem}_anchors.json cache exists
(produced by some future, separate offline script - none exists in this
pipeline today), its detected centers are shown as reference tick marks
along the anchor line. This UI never calls detect_column_number_centers()
or any variant itself - see core.column_calibration.try_load_anchor_
cache()'s own docstring.

Row-number margin anchors (added 2026-08-07, per Jon): a SEPARATE pair of
single-x calibration lines for the printed VERTICAL row-number margin(s)
(the "1, 2, 3..." numbering down the page edge - see DocumentTemplate.
row_number_column_x_frac/row_number_column2_x_frac). Gated behind "Define
left/right row number centers" - click LEFT margin center, click RIGHT
margin center, mode auto-deactivates. Re-arming the mode after both are
already set clears both immediately, so the next click starts a fresh
placement (the redo mechanism, per spec - no separate per-line clear).

Usage:
    python ui/column_calibration_ui.py
        (no args - opens a folder picker, added 2026-08-07)
    python ui/column_calibration_ui.py --image-dir J:\\path\\to\\images
    python ui/column_calibration_ui.py --images img1.jpg img2.jpg ...
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tkinter import (
    Tk, Toplevel, Frame, Label, Button, Canvas, Scrollbar, Listbox, Text,
    StringVar, OptionMenu, messagebox, filedialog,
)

from PIL import Image, ImageDraw, ImageTk

from core.column_calibration import (
    load_or_build_correction_record, load_correction_record, save_correction_record, correction_path_for,
    apply_divider_move, confirm_divider_as_is, clear_divider,
    apply_table_edge_move, confirm_table_edge_as_is, clear_table_edge, apply_left_crop, apply_top_crop,
    merge_column_corrections, project_remaining_dividers, try_load_anchor_cache,
    PROVENANCE_UNREVIEWED, PROVENANCE_CONFIRMED, PROVENANCE_PLACED,
)
from core.rotation_refinement import refine_rotation, BlobCandidate
from core.document_templates import load_template, load_all_templates, TEMPLATES_DIR
from core.row_segmentation import load_sidecar, save_sidecar, detect_row_number_centers
from core.manifest_pipeline import preprocess_for_manifest, DEFAULT_PREPROCESSING_PROFILE
from core.auto_sidecar import generate_auto_sidecar, estimate_deskew_angle
from core.image_analysis import DESKEW_ANGLE_RANGE
from core.template_refinement import refine_template_from_calibration, format_refinement_report
from core.workspace_context import WorkspaceContext
from ui.quarantine_review_ui import find_source_image, load_column_list

# row_segmentation is NOT repointed like the calibration paths below -
# core/row_segmentation.py's own output dir is a "copy-only" path from
# the workspace migration (both data/outputs/row_segmentation/ and
# genealogy_workspace's copy genuinely coexist with independent
# content, protected via .claude/protected_paths.txt - unlike column_
# calibration/column_calibration_workspace, there's no junction here to
# stop riding on, and repointing this one risks silently reading a
# DIFFERENT, incomplete copy instead of the real one).
DEFAULT_SIDECAR_DIR = PROJECT_ROOT / "data" / "outputs" / "row_segmentation"

# Per-dialog remembered directories (2026-08-09, per Jon: "file picker
# for input, can we set that to remember last directory, and same for
# export sidecar, its currently using whatever was last used for either
# function"). Without an explicit initialdir, Windows' native picker
# (which tkinter's filedialog delegates to) falls back to its own
# single shared MRU bucket across every dialog in the process that
# doesn't set one - so the raw-folder import picker and the export-
# merged-sidecar dialog were bleeding into each other's last-used
# location instead of each remembering its own. Small standalone JSON
# file storing {dialog_key: last_directory}, persisted across separate
# tool launches, not just within one run. Placed under the real
# calibration workspace root (2026-08-09, per Jon - see core.
# calibration_workspace.WORKSPACE_ROOT's own comment for the full
# "why") rather than the old data/outputs/ convention - a genuinely
# new file with no legacy junction to worry about, so no migration
# concern here, just consistent placement with the rest of this
# tool's real data.
_UI_PREFS_PATH = (
    WorkspaceContext.resolve().workspace_root / "research" / "calibration"
    / "column_calibration_ui_prefs.json"
)


def _load_ui_prefs() -> dict:
    try:
        return json.loads(_UI_PREFS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _remember_dialog_dir(key: str, path_str: str) -> None:
    if not path_str:
        return
    prefs = _load_ui_prefs()
    prefs[key] = str(Path(path_str).parent if Path(path_str).suffix else path_str)
    try:
        _UI_PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _UI_PREFS_PATH.write_text(json.dumps(prefs, indent=2), encoding="utf-8")
    except OSError:
        pass  # best-effort convenience state - never worth failing the actual dialog action over


def _last_dialog_dir(key: str) -> str | None:
    value = _load_ui_prefs().get(key)
    return value if value and Path(value).is_dir() else None

ZOOM_OPTIONS = {
    "Fit": None, "1.25x": 1.25, "1.5x": 1.5, "1.75x": 1.75, "2x": 2.0, "4x": 4.0, "8x": 8.0,
}
PREVIEW_SIZE = (1400, 1000)  # Fit-mode thumbnail bound

DIVIDER_HIT_RADIUS_PX = 6  # canvas-space click tolerance, divided by scale for image space

# Rendering convention - no existing auto-vs-human dichotomy anywhere in
# this codebase to copy (row_segmentation_ui.py's colors are per boundary
# TYPE, not per provenance), so this is a new one, deliberately distinct
# in both color and line style:
COLOR_AUTO = "#5b8fd4"        # thin dashed, muted - unconfirmed auto proposal
COLOR_HUMAN = "#39ff6a"       # solid bold - human-confirmed/placed
COLOR_PROJECTED = "#e0a92e"   # dotted - advisory projection, not yet accepted
COLOR_TABLE_EDGE = "#888888"  # faint, non-interactive context only (table_bbox)
COLOR_ANCHOR = "#ff5bd0"      # header-number anchor line + detected centers
COLOR_ROW_NUMBER = "#00e5ff"  # left/right row-number margin center lines
COLOR_GUIDE_LINE = "#ffff00"  # cursor-following guide crosshair - bright yellow,
                               # deliberately unused elsewhere in this palette so it
                               # stays visible against both light scan backgrounds and
                               # dark table rulings (white was hard to see on a white
                               # page background per Jon, 2026-08-09)
# Expected-row-N-end overlay (2026-08-09, per Jon: "a visual overlay
# showing where the template thinks row 40 should end... if that line is
# obviously hanging off the page, you know immediately the pitch
# calibration is wrong"). Two colors, not one - the whole POINT is that
# this line's color itself is the sanity check: orange when it lands
# within the visible page (plausible), red when it overshoots past the
# actual image bottom (the exact 1906 pitch bug, made visually obvious
# instead of only discoverable by running generate_auto_sidecar()
# afterward and reading a warning in the log).
COLOR_EXPECTED_ROW_END_OK = "#ffa500"
COLOR_EXPECTED_ROW_END_BAD = "#ff2020"

# Crop-left projected suggestion (2026-08-09) - see apply_left_crop()'s
# own docstring for the real, measured safe-margin validation this
# offset is based on. Distinct color from every other overlay in this
# module (COLOR_GUIDE_LINE is yellow but only appears while actively
# armed/moving the mouse - this one is static once table_left exists).
COLOR_CROP_LEFT_SUGGESTED = "#00bfff"
_CROP_LEFT_SUGGESTED_OFFSET_PX = 50
# Crop-top projected suggestion (2026-08-09, same day) - see
# apply_top_crop()'s own docstring for the real, measured safe-margin
# validation (table_top-to-blob gap 661-701px across 14 confirmed 1906
# samples) this offset is based on. 600, not the full safe-margin
# ceiling (~650) - per Jon, moved 50px closer to table_top after
# reviewing the tightest-margin overlay, still comfortably inside the
# validated safe range.
COLOR_CROP_TOP_SUGGESTED = "#00bfff"
_CROP_TOP_SUGGESTED_OFFSET_PX = 600

# Pixel-ruler tool (2026-08-10) - see _arm_ruler_mode()'s own docstring.
# Bright orange: distinct from every other overlay color already used in
# this module (crop suggestions are blue, guide crosshair is yellow).
COLOR_RULER = "#ff8c00"

PROVENANCE_MARKER = {
    PROVENANCE_UNREVIEWED: " ",
    PROVENANCE_CONFIRMED: "\u2713",
    PROVENANCE_PLACED: "+",
}


def _draw_dashed_vline(draw: ImageDraw.ImageDraw, x: int, y0: int, y1: int, color: str,
                        width: int = 1, dash: int = 5, gap: int = 4) -> None:
    y = y0
    while y < y1:
        y_end = min(y + dash, y1)
        draw.line([(x, y), (x, y_end)], fill=color, width=width)
        y += dash + gap


def _draw_dotted_vline(draw: ImageDraw.ImageDraw, x: int, y0: int, y1: int, color: str,
                        width: int = 2, dot: int = 2, gap: int = 5) -> None:
    y = y0
    while y < y1:
        y_end = min(y + dot, y1)
        draw.line([(x, y), (x, y_end)], fill=color, width=width)
        y += dot + gap


class ColumnCalibrationApp:
    def __init__(self, root: Tk, image_paths: list[Path], sidecar_dir: Path,
                 columns_file_override: str | None):
        self.root = root
        self.root.title("Column Calibration")
        self.image_paths = image_paths
        self.sidecar_dir = sidecar_dir
        self.columns_file_override = columns_file_override
        self.page_index = 0

        self.original_image: Image.Image | None = None
        self.sidecar: dict | None = None
        self.sidecar_path: Path | None = None
        self.correction_record: dict | None = None
        self.correction_path: Path | None = None
        self.template = None
        self.column_order: list[str] = []

        self.undo_stack: list[dict] = []
        self.redo_stack: list[dict] = []

        self._preview_scale = 1.0
        self._preview_render_size = (0, 0)
        self._tk_preview = None
        self._drag_index: int | None = None
        self._drag_before: dict | None = None
        self._selected_index: int | None = None
        self._anchor_drag_armed = False
        self._row_number_mode_armed = False
        self._row_number_click_stage = 0  # 0 = waiting for left click, 1 = waiting for right
        self._dirty = False  # unsaved changes since last save, for the nav guard
        # Display-only extra rotation from a "Test rotation refinement"
        # run (see _on_test_rotation_refinement()) - never written to any
        # sidecar/correction record, purely a visual preview. Reset to
        # 0.0 on every page load. ALSO settable manually via the Deskew
        # -/+ buttons (2026-08-08, per Jon: "dont autosplit, have it as
        # an option inside the ui... and also manual deskew + and -
        # buttons") - same variable, same render path, either source.
        # Still never auto-written to a sidecar/correction record on its
        # own; export_merged_sidecar() is the one place a nonzero value
        # actually gets persisted, same "only Export writes real output"
        # rule every other correction in this tool already follows.
        self._preview_rotation_delta: float = 0.0
        # Interactive split-point mode (2026-08-08, per Jon - replaces
        # relying solely on core.calibration_workspace's folder-wide
        # split.split marker file with a per-page manual alternative):
        # armed via "Set split point..." below, mutually exclusive with
        # the anchor/row-number click modes. self._working_image_path is
        # the actual working-copy file _load_page() opened (needed here
        # because self.original_image is a PIL Image already in memory,
        # not something a fresh split can re-crop from disk without
        # re-opening the source file, and because a raw source path
        # isn't otherwise tracked - see _load_page()).
        self._split_mode_armed = False
        self._working_image_path: Path | None = None
        # Table outer-edge mode (2026-08-08, per Jon: "the ui changing to
        # enable me to mark those lines" - the table's left/right outer
        # edges were previously drawn for context only, non-interactive,
        # which is exactly why the FIRST/LAST column's own outer boundary
        # never had real calibration data - no divider can exist there,
        # see core/column_calibration.py's table_edges docstring). Same
        # 2-click-then-done pattern as row-number mode: click LEFT table
        # edge, click RIGHT table edge, mode auto-deactivates.
        self._table_edge_mode_armed = False
        self._table_edge_click_stage = 0  # 0 = waiting for left click, 1 = waiting for right
        # Table TOP mode (2026-08-09, per Jon: running generate_auto_
        # sidecar() standalone in a separate session found table_top
        # detection wrong on ~half a real batch). Single-click, same
        # pattern as the header-number anchor line (_arm_anchor_mode) -
        # a Y-axis boundary, unlike left/right which are X, so it
        # doesn't fit the 2-click table-edge mode above.
        self._table_top_mode_armed = False
        # Table BOTTOM mode (2026-08-09, per Jon: "would it be easier to
        # just mark the table bottom for templating?" - marking real
        # table_bottom lets row pitch be computed exactly per-page
        # ((bottom-top)/(row_count-1)) instead of an averaged template
        # constant. Same single-click Y-axis pattern as table top; NOT
        # yet wired into auto_sidecar's tiling logic, per Jon's own
        # sequencing (UI first, then inspect real samples, then wire in).
        self._table_bottom_mode_armed = False
        # Row-pitch MEASURE mode (2026-08-09, per Jon: after marking a
        # correct table_top by hand, the expected-row-N-end overlay still
        # didn't match row N's real position on a specific page - meaning
        # the mismatch was the TEMPLATE's averaged pitch not matching
        # THIS page's real spacing, not a table_top bug. This mode lets
        # him click row 1's top then row N's top directly on any page and
        # see that page's OWN real pitch next to the template's assumed
        # one, to tell real page-to-page variance from an actual bug at a
        # glance. Purely a diagnostic readout - never written to the
        # correction record or sidecar, resets on every page load/re-arm.
        self._pitch_measure_armed = False
        self._pitch_measure_first_y: float | None = None
        self._pitch_measure_result: str = ""
        # Crop-left mode (2026-08-09, per Jon's "crop_left_of_marker"
        # design - a real, physical camera-mask blob measured on almost
        # every 1906 "_L" split-half sample, ~3.5-4.4% of page width,
        # too position-inconsistent relative to the blob itself to
        # auto-crop safely - see core.column_calibration.apply_left_
        # crop()'s own docstring). A dashed projected line is drawn at
        # table_bbox[0] - _CROP_LEFT_SUGGESTED_OFFSET_PX (only once
        # table_left is human-confirmed) as a rough starting SUGGESTION
        # - real validated safe margin across 19 real 1906 "_L" samples
        # (row-number-anchor gap 17-31px, all comfortably cleared by a
        # 50px offset) - but the human still clicks the real position
        # every time; this tool never auto-applies the projection.
        # Permanently crops the working image file AND shifts every
        # stored X-coordinate in the correction record to match (see
        # apply_left_crop()) - unlike the deskew bake, there's no way to
        # undo this short of re-deriving the page from the raw source,
        # so it's gated behind an explicit confirm dialog, not silently
        # folded into Save like the deskew bake was.
        self._crop_left_mode_armed = False
        # Crop-top mode (2026-08-09, same day, per Jon: "we need a way
        # to crop the top too" - a real, near-universal damaged/torn top
        # border found across the 1906 batch, ~3.5-3.9% of page height
        # on 40/42 samples, one genuine outlier (e001211831_L/_R) where
        # it extends much deeper). Mirrors crop-left exactly: dashed
        # projected line at table_bbox[1] - _CROP_TOP_SUGGESTED_OFFSET_PX
        # (only once table_top is human-confirmed), offset validated
        # against real measured table_top-to-blob gaps across 14
        # confirmed samples (661-701px) - human still clicks the real
        # position every time. Permanently crops the working image file
        # AND shifts every stored Y-coordinate to match (see apply_top_
        # crop()) - same explicit-confirm-dialog gating as crop-left,
        # never folded into Save.
        self._crop_top_mode_armed = False
        # Pixel-ruler tool (2026-08-10, per Jon: "click a spot and it
        # applies a px ruler to that clicked location to help with
        # measuring and a toggle to flip axis between x and y" - built
        # after several rounds of manually scripting one-off ruler-
        # annotated crops (see e.g. the crop_top_suggested_offset_px
        # measurement work) to visually verify table_top/metadata
        # positions; this puts the same tick-marked-distance-from-a-
        # point idea directly in the UI instead of a throwaway script
        # each time. Non-destructive and, unlike the crop tools, stays
        # armed across clicks - re-clicking just repositions the ruler,
        # since repeated quick re-measurement (not a single one-shot
        # commit) is the whole point.
        self._ruler_mode_armed = False
        self._ruler_origin = None  # (image_x, image_y) of the last ruler click, or None
        self._ruler_axis = "y"  # "y" (default - most measurements here are vertical) or "x"
        # Live guide-line crosshair (2026-08-09, per Jon: "a guide line
        # across the preview to help with making sure the line is
        # straight before apply the lines") - the LAST canvas coordinates
        # seen by _on_canvas_motion(), used to draw a cheap canvas-native
        # line item (never re-renders the full PIL image) that follows
        # the cursor whenever a click would place/move something.
        self._guide_line_id: int | None = None

        self._build_widgets()
        self._load_page()

    # ------------------------------------------------------------ widgets
    def _build_widgets(self) -> None:
        top = Frame(self.root)
        top.pack(fill="x", padx=8, pady=4)
        self.title_label = Label(top, text="", font=("Segoe UI", 11, "bold"), anchor="w")
        self.title_label.pack(side="left")

        nav = Frame(self.root)
        nav.pack(fill="x", padx=8, pady=2)
        Button(nav, text="<< Prev", command=self.prev_page).pack(side="left")
        Button(nav, text="Next >>", command=self.next_page).pack(side="left")
        Button(nav, text="Save / Confirm Page", command=self.save_page).pack(side="left", padx=(12, 0))
        Button(nav, text="Undo", command=self.undo).pack(side="left", padx=(12, 0))
        Button(nav, text="Redo", command=self.redo).pack(side="left")
        Button(nav, text="Export merged sidecar...", command=self.export_merged_sidecar).pack(
            side="left", padx=(12, 0))
        Button(nav, text="Save overlay PNG...", command=self.save_overlay_png).pack(
            side="left", padx=(4, 0))

        self.zoom_var = StringVar(value="Fit")
        Label(nav, text="  Zoom:").pack(side="left", padx=(12, 0))
        OptionMenu(nav, self.zoom_var, *ZOOM_OPTIONS.keys(), command=lambda _v: self._render()).pack(side="left")

        nav2 = Frame(self.root)
        nav2.pack(fill="x", padx=8, pady=2)
        Button(nav2, text="Load columns file...", command=self._on_load_columns_file).pack(side="left")
        Button(nav2, text="Regenerate sidecars (this folder)...",
               command=self._on_regenerate_sidecars).pack(side="left", padx=(8, 0))
        Button(nav2, text="Preview template refinement...",
               command=self._on_preview_template_refinement).pack(side="left", padx=(8, 0))
        Button(nav2, text="Confirm selected as-is", command=self.confirm_selected).pack(
            side="left", padx=(20, 0))
        Button(nav2, text="Accept projected value", command=self.accept_projected).pack(side="left", padx=(8, 0))
        Button(nav2, text="Project from confirmed anchors", command=self.project_remaining).pack(
            side="left", padx=(8, 0))
        Button(nav2, text="Set header-number anchor line", command=self._arm_anchor_mode).pack(
            side="left", padx=(20, 0))
        Button(nav2, text="Define left/right row number centers", command=self._toggle_row_number_mode).pack(
            side="left", padx=(8, 0))
        Button(nav2, text="Mark table left/right edges", command=self._toggle_table_edge_mode).pack(
            side="left", padx=(8, 0))
        Button(nav2, text="Mark table top", command=self._arm_table_top_mode).pack(
            side="left", padx=(8, 0))
        Button(nav2, text="Mark table bottom", command=self._arm_table_bottom_mode).pack(
            side="left", padx=(8, 0))
        Button(nav2, text="Measure row pitch...", command=self._arm_pitch_measure_mode).pack(
            side="left", padx=(8, 0))
        # Rotation-refinement TEST row (added 2026-08-08, per Jon: "test
        # this first in column_calibration_ui... have a button to trigger
        # the process and update the image on screen, not write a new
        # image completely yet"). Deliberately its own row, separate from
        # the real column-calibration controls above - this is an
        # exploratory preview of core/rotation_refinement.py against the
        # row-number-center detector, not a real calibration action, and
        # never writes anything to disk (no sidecar/correction-record
        # mutation) - purely an on-screen rotated preview + a status-line
        # readout of the evidence dict, so Jon can visually judge the
        # result while he's already masking a page, before this module
        # gets wired into any live pipeline stage.
        nav3 = Frame(self.root)
        nav3.pack(fill="x", padx=8, pady=2)
        Button(nav3, text="Test rotation refinement (row-number band)",
               command=self._on_test_rotation_refinement).pack(side="left")
        Button(nav3, text="Clear rotation preview", command=self._clear_rotation_preview).pack(
            side="left", padx=(8, 0))
        # Manual deskew nudge (2026-08-08, per Jon: "manual deskew + and
        # - buttons") - same self._preview_rotation_delta the auto-test
        # above sets, so either source ends up in the same place: an
        # on-screen preview only until Export merged sidecar applies it.
        # +/-0.1 deg per click (fine enough for a nudge-correction, not
        # a from-scratch angle search); "Clear rotation preview" above
        # doubles as the revert-to-original action for this too - no
        # separate reset button needed for the same variable.
        Label(nav3, text="  Deskew:").pack(side="left", padx=(12, 0))
        Button(nav3, text="− 0.1°", width=6, command=lambda: self._nudge_deskew(-0.1)).pack(
            side="left", padx=(4, 0))
        Button(nav3, text="+ 0.1°", width=6, command=lambda: self._nudge_deskew(0.1)).pack(
            side="left", padx=(2, 0))
        self.rotation_test_status_var = StringVar(value="")
        Label(nav3, textvariable=self.rotation_test_status_var, font=("Segoe UI", 9), fg="#0a4").pack(
            side="left", padx=(12, 0))

        # Bulk horizontal shift (2026-08-09, per Jon: redoing an older
        # page's manual deskew with a corrected angle changes where the
        # content lands pixel-wise - real spacing between already-placed
        # dividers/edges stays correct (deskew is a small-angle rotation,
        # not a re-scale), but the whole set can end up shifted sideways
        # from the freshly re-deskewed image by a near-constant offset.
        # Re-clicking every divider individually just to correct a
        # uniform offset would be wasted work - this shifts every
        # already-placed X-axis position (divider human_x, table left/
        # right edges, row-number left/right anchors) by the same delta
        # in one action, preserving relative spacing exactly. Vertical
        # geometry (table top/bottom, header-number anchor band_y) is
        # untouched - a horizontal deskew-realignment offset has no
        # reason to move those.
        Label(nav3, text="  Shift all lines:").pack(side="left", padx=(12, 0))
        Button(nav3, text="◄ 1px", width=6, command=lambda: self._shift_all_x(-1)).pack(
            side="left", padx=(4, 0))
        Button(nav3, text="1px ►", width=6, command=lambda: self._shift_all_x(1)).pack(
            side="left", padx=(2, 0))

        # Interactive split point (2026-08-08, per Jon: "dont autosplit,
        # have it as an option inside the ui, i click where to split
        # then" - replacing sole reliance on core.calibration_workspace's
        # folder-wide split.split marker file, which applies one fixed
        # fraction to every image in a folder with no per-page override).
        nav4 = Frame(self.root)
        nav4.pack(fill="x", padx=8, pady=2)
        Button(nav4, text="Set split point...", command=self._arm_split_mode).pack(side="left")
        # Red fg/bg (2026-08-09, per Jon: the button was getting squeezed
        # illegibly narrow in nav2 on a fullscreen layout - moved here,
        # next to the other "starts an armed click-mode" button it's
        # conceptually closest to - AND colored red, since unlike every
        # other button in this row this one PERMANENTLY mutates the
        # working image file with no undo (see apply_left_crop()'s own
        # docstring) - a real "dangerous action" distinct from every
        # other reversible-via-Undo/Redo correction this tool makes.
        Button(nav4, text="Crop left edge...", command=self._arm_crop_left_mode,
               fg="white", bg="#c0392b", activebackground="#a93226", activeforeground="white").pack(
            side="left", padx=(12, 0))
        Button(nav4, text="Crop top edge...", command=self._arm_crop_top_mode,
               fg="white", bg="#c0392b", activebackground="#a93226", activeforeground="white").pack(
            side="left", padx=(4, 0))
        # Pixel-ruler tool (2026-08-10) - see _arm_ruler_mode()'s own
        # docstring. Non-destructive (unlike the two red crop buttons
        # above), so plain default styling - the color-coding in this
        # row is reserved for "this click permanently mutates the file."
        Button(nav4, text="Ruler", command=self._arm_ruler_mode).pack(side="left", padx=(12, 0))
        self.ruler_axis_btn_var = StringVar(value=f"Ruler axis: {self._ruler_axis.upper()}")
        Button(nav4, textvariable=self.ruler_axis_btn_var, command=self._toggle_ruler_axis, width=14).pack(
            side="left", padx=(4, 0))
        # Open/Reload Config (2026-08-10) - see _open_config()/_reload_
        # config()'s own docstrings. Generic over doc_type, no per-
        # template special-casing - part of the shift toward this being
        # a document-preparation tool where config tweaks happen inline.
        Button(nav4, text="Open Config", command=self._open_config).pack(side="left", padx=(12, 0))
        Button(nav4, text="Reload Config", command=self._reload_config).pack(side="left", padx=(4, 0))
        self.split_status_var = StringVar(value="")
        Label(nav4, textvariable=self.split_status_var, font=("Segoe UI", 9), fg="#a40").pack(
            side="left", padx=(12, 0))

        self.status_label = Label(self.root, text="", anchor="w", font=("Segoe UI", 9))
        self.status_label.pack(fill="x", padx=8)

        body = Frame(self.root)
        body.pack(fill="both", expand=True, padx=8, pady=4)

        canvas_frame = Frame(body)
        canvas_frame.pack(side="left", fill="both", expand=True)
        hbar = Scrollbar(canvas_frame, orient="horizontal")
        vbar = Scrollbar(canvas_frame, orient="vertical")
        # highlightthickness=0, bd=0 (2026-08-07, found via real-widget
        # testing against a real image/sidecar, not caught by the earlier
        # synthetic-dict tests): the default highlightthickness=2 focus
        # border shifts canvasx()/canvasy()'s coordinate mapping by a
        # constant -2 canvas-space units once a scrollregion is active -
        # confirmed by isolating it (canvasx(0) == -2.0 with defaults,
        # == 0.0 with these two options set). At Fit-zoom on a 3352px-wide
        # page (preview_scale ~0.38) that's a ~5px image-space drag
        # error - real precision loss for a tool whose whole purpose is
        # precise pixel calibration. ui/row_segmentation_ui.py's own
        # Canvas has the same default (no highlightthickness/bd override)
        # and likely inherits the same offset - flagged separately as an
        # out-of-scope fix for that file, not applied here.
        self.canvas = Canvas(canvas_frame, bg="black", highlightthickness=0, bd=0,
                              xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        hbar.config(command=self.canvas.xview)
        vbar.config(command=self.canvas.yview)
        hbar.pack(side="bottom", fill="x")
        vbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Button-3>", self._on_right_click)
        # Guide-line crosshair (2026-08-09, per Jon: "a guide line across
        # the preview to help with making sure the line is straight
        # before apply the lines") - a plain canvas line item, NOT a full
        # re-render on every mouse move (would be far too slow on the
        # large 1921 census scans, ~7000px wide). <Motion> fires on any
        # mouse movement over the canvas, independent of whether a
        # button is held - <B1-Motion> above only fires during an active
        # drag, which is a different, already-handled case.
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        self.canvas.bind("<Leave>", self._on_canvas_leave)

        side = Frame(body, width=280)
        side.pack(side="left", fill="y", padx=(8, 0))

        # Row-pitch sanity readout (2026-08-09, per Jon: "Table height...
        # Expected row pitch... Expected row 40: y=..." live in the
        # sidebar - "exactly the sort of sanity check that would have
        # exposed the original pitch bug before it ever reached
        # AutoSidecar"). Plain text, updated by _update_row_pitch_info()
        # from _render() - no live computation logic lives in the widget
        # itself.
        self.row_pitch_info_var = StringVar(value="")
        Label(side, textvariable=self.row_pitch_info_var, anchor="w", justify="left",
              font=("Consolas", 9), fg="#a40", wraplength=270).pack(fill="x", pady=(0, 6))

        Label(side, text="Dividers (left | right)", anchor="w", font=("Segoe UI", 9, "bold")).pack(fill="x")
        self.divider_listbox = Listbox(side, font=("Consolas", 9), width=40)
        self.divider_listbox.pack(fill="both", expand=True)
        self.divider_listbox.bind("<<ListboxSelect>>", self._on_listbox_select)

        self.root.bind("<Delete>", self._on_delete_key)
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<Control-y>", lambda e: self.redo())

    # -------------------------------------------------------------- page
    def _load_page(self) -> None:
        if not (0 <= self.page_index < len(self.image_paths)):
            messagebox.showinfo("Column Calibration", "No more pages.")
            return

        launch_path = self.image_paths[self.page_index]
        stem = launch_path.stem

        self.sidecar_path = self._find_sidecar_for(stem)
        self.sidecar = load_sidecar(self.sidecar_path) if self.sidecar_path else None

        # Resolve the ACTUAL image to display via the sidecar's own
        # source_image_path (same resolution find_source_image() already
        # does for ui/quarantine_review_ui.py) - not just the launch-time
        # path, since that may point at a raw scan rather than the
        # deskewed working copy the sidecar's pixel coordinates are
        # relative to (coordinate_space: "deskewed_image").
        if self.sidecar is not None:
            resolved = find_source_image(self.sidecar, stem) or launch_path
        else:
            resolved = launch_path
        self._working_image_path = Path(resolved)
        self.original_image = Image.open(resolved).convert("RGB")
        self._split_mode_armed = False

        doc_type = (self.sidecar or {}).get("parameters", {}).get("doc_type")
        self.template = None
        columns_file_name = None
        if doc_type:
            try:
                self.template = load_template(doc_type)
            except FileNotFoundError as e:
                messagebox.showwarning("Column Calibration", f"{stem}: {e}")

        if self.columns_file_override:
            self.column_order = [
                line.strip() for line in Path(self.columns_file_override).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        elif self.template is not None:
            cols = load_column_list(self.template, override_path=None)
            self.column_order = cols if cols is not None else self.template.expected_columns
            columns_file_name = self.template.columns_file
        elif self.sidecar is not None:
            self.column_order = self.sidecar.get("column_order", [])
        else:
            self.column_order = []

        self.correction_path = correction_path_for(stem)
        if self.sidecar is not None and self.sidecar.get("table_bbox") and len(self.column_order) >= 2:
            self.correction_record = load_or_build_correction_record(
                self.sidecar, self.sidecar_path, self.correction_path,
                doc_type, self.column_order, columns_file_name,
            )
        else:
            self.correction_record = None
            if self.sidecar is None:
                self.status_label.config(
                    text=f"{stem}: no auto sidecar found - opening image only, nothing to calibrate yet.")
            elif not self.sidecar.get("table_bbox"):
                self.status_label.config(
                    text=f"{stem}: sidecar has no table_bbox yet - nothing to calibrate.")
            elif len(self.column_order) < 2:
                self.status_label.config(
                    text=f"{stem}: fewer than 2 columns known - no dividers to calibrate.")

        detected = try_load_anchor_cache(stem)
        if self.correction_record is not None and detected is not None:
            self.correction_record["header_number_anchor"]["detected_centers"] = [list(c) for c in detected]

        self.undo_stack = []
        self.redo_stack = []
        # Auto-select the FIRST unreviewed divider (added 2026-08-07, per
        # Jon: the click-list-click-image-click-list dance was the real
        # friction, not any single step) - so a fresh page is immediately
        # ready for "click click click" left-to-right, no listbox touch
        # needed before the very first placement either.
        self._selected_index = None
        if self.correction_record is not None:
            for i, d in enumerate(self.correction_record["dividers"]):
                if d["provenance"] == PROVENANCE_UNREVIEWED:
                    self._selected_index = i
                    break
        self._dirty = False
        # Restore any previously-saved manual deskew correction (2026-08-09,
        # per Jon: "each reload has to rederive the deskew manually and it
        # can affect the masked line placement by shifting them slightly
        # after a fresh manual deskew") - was unconditionally 0.0 before,
        # discarding a real correction on every reopen even though the
        # dividers were originally placed against the ROTATED preview.
        self._preview_rotation_delta = (
            (self.correction_record or {}).get("manual_deskew_delta_deg", 0.0) or 0.0)
        if self._preview_rotation_delta != 0.0:
            self.rotation_test_status_var.set(
                f"Restored saved manual deskew: {self._preview_rotation_delta:+.2f} deg.")
        else:
            self.rotation_test_status_var.set("")
        self._pitch_measure_armed = False
        self._pitch_measure_first_y = None
        self._pitch_measure_result = ""
        self._crop_left_mode_armed = False
        self._crop_top_mode_armed = False
        self._ruler_mode_armed = False
        self._ruler_origin = None

        review_status = (self.correction_record or {}).get("review_status", "n/a")
        self.title_label.config(
            text=f"Page {self.page_index + 1} of {len(self.image_paths)}   |   {stem}   |   "
                 f"review_status: {review_status}")

        self._render()

    def _open_config(self) -> None:
        """
        Opens the CURRENT page's active template YAML in the system's
        default editor (2026-08-10, per Jon: a "document preparation
        workflow" needs config edits to be a normal part of the loop,
        not a context-switch out to a file browser/IDE every time -
        this and _reload_config() below are the pair that makes that
        practical). Generic over doc_type - there is no per-template
        special-casing here, it just resolves core.document_templates.
        TEMPLATES_DIR / f"{doc_type}.yaml", the exact same path load_
        template() itself reads.
        """
        if self.template is None:
            messagebox.showinfo(
                "Column Calibration",
                "No template loaded for the current page - open a page with a known doc_type first.")
            return
        path = TEMPLATES_DIR / f"{self.template.doc_type}.yaml"
        if not path.exists():
            messagebox.showwarning("Column Calibration", f"Template file not found: {path}")
            return
        try:
            if sys.platform == "win32":
                os.startfile(str(path))  # noqa: S606 - opening a local config file, not user-controlled input
            elif sys.platform == "darwin":
                subprocess.run(["open", str(path)], check=False)
            else:
                subprocess.run(["xdg-open", str(path)], check=False)
        except OSError as e:
            messagebox.showwarning("Column Calibration", f"Could not open {path}:\n{e}")

    def _reload_config(self) -> None:
        """
        Re-reads the current page's template YAML from disk and
        refreshes every template-derived piece of UI/state - WITHOUT
        restarting the tool or touching self.correction_record's
        already-CONFIRMED divider positions (2026-08-10, per Jon: this
        is the other half of the Open Config / Reload Config pair - edit
        the YAML in Open Config, tweak a crop offset or column_regions_
        approx band, then see it reflected immediately here).

        core.document_templates.load_template() has no caching (plain
        function, re-parses the YAML every call - confirmed by reading
        it, not assumed) so simply calling it again already picks up
        any on-disk edit; the real work here is re-deriving column_order
        and reconciling self.correction_record against it the SAME way
        _load_page() does on a fresh load (via load_or_build_correction_
        record(), which is specifically built to carry forward already-
        reviewed divider positions across a column_order/template
        change - see that function's own docstring). Undo/redo history
        is cleared: if column_order's real length/order changed, old
        undo/redo entries reference divider INDICES that may no longer
        mean the same thing, and silently replaying one against the
        wrong divider would be worse than losing undo history.
        """
        if self.template is None:
            messagebox.showinfo(
                "Column Calibration",
                "No template loaded for the current page - nothing to reload.")
            return
        doc_type = self.template.doc_type
        try:
            new_template = load_template(doc_type)
        except FileNotFoundError as e:
            messagebox.showwarning("Column Calibration", f"Reload failed: {e}")
            return

        columns_file_name = None
        if self.columns_file_override:
            new_column_order = [
                line.strip() for line in Path(self.columns_file_override).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            cols = load_column_list(new_template, override_path=None)
            new_column_order = cols if cols is not None else new_template.expected_columns
            columns_file_name = new_template.columns_file

        self.template = new_template
        self.column_order = new_column_order

        if (self.correction_record is not None and self.sidecar is not None
                and self.sidecar_path is not None and len(new_column_order) >= 2):
            self.correction_record = load_or_build_correction_record(
                self.sidecar, self.sidecar_path, self.correction_path,
                doc_type, new_column_order, columns_file_name,
            )
            self.undo_stack = []
            self.redo_stack = []
            self._selected_index = None
            for i, d in enumerate(self.correction_record["dividers"]):
                if d["provenance"] == PROVENANCE_UNREVIEWED:
                    self._selected_index = i
                    break

        self.status_label.config(text=f"Reloaded template config for {doc_type} from disk.")
        self._render()

    def _find_sidecar_for(self, stem: str) -> Path | None:
        path = self.sidecar_dir / f"{stem}_sidecar.json"
        return path if path.exists() else None

    def _on_load_columns_file(self) -> None:
        """
        Added 2026-08-07, per Jon: pages whose real doc_type has no
        config/document_templates/*.yaml yet (e.g. an 1906 census page,
        a structurally different Prairie-census form from the 4 existing
        templates) still get a sidecar with a real table_bbox from
        generate_auto_sidecar()'s CV-classifier fallback - just against
        the WRONG template's column list. This lets a human pick the
        RIGHT column list directly, same "Load columns file..." pattern
        ui/row_segmentation_ui.py already has, without needing a new
        template YAML built first just to get dividers to click.

        Reuses the EXISTING self.columns_file_override mechanism
        _load_page() already checks first (originally a --columns-file
        CLI-only option) rather than duplicating its column-order-
        building logic here - setting it and reloading the current page
        is the whole implementation. Because _load_page() re-checks this
        attribute on every page load, the picked file stays applied for
        every subsequent page in this session too, matching row_
        segmentation_ui.py's own "load once, applies going forward"
        behavior - not just a one-page-only override.
        """
        if self.sidecar is None or not self.sidecar.get("table_bbox"):
            messagebox.showinfo(
                "Column Calibration",
                "This page has no sidecar with a table region yet - a column list alone "
                "can't place dividers without a table_bbox to draw them inside.")
            return
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
        self.columns_file_override = path
        self._load_page()
        self.status_label.config(
            text=f"Loaded {len(names)} column(s) from {Path(path).name} - now applied to this "
                 f"and all subsequent pages this session.")

    def _prompt_doc_type_override(self) -> tuple[bool, str | None]:
        """
        Small modal dialog shown before a "Regenerate sidecars" batch
        run (2026-08-11, per Jon: pages regenerated over and over stayed
        stuck at a wrong doc_type - canada_census_1931 - because that
        button deliberately reads each page's EXISTING sidecar's own
        doc_type rather than reclassifying, and there was no way to
        force a specific one). Generic over doc_type - lists whatever
        core.document_templates.load_all_templates() finds in config/
        document_templates/*.yaml, no hardcoded year list to keep in
        sync.

        Returns (proceed, doc_type_override): proceed=False means the
        user cancelled (caller should abort the whole regenerate run).
        doc_type_override=None means "(keep existing per-image
        doc_type)" was chosen - the normal/default behavior, unchanged
        from before this dialog existed. A specific doc_type FORCES
        every image in this regenerate pass to that template, including
        fixing up any existing correction record's own stale doc_type/
        column_order - see _on_regenerate_sidecars()'s own reconciliation
        block for what that entails.
        """
        KEEP_EXISTING = "(keep existing per-image doc_type)"
        try:
            doc_types = sorted(load_all_templates().keys())
        except Exception:
            doc_types = []
        options = [KEEP_EXISTING] + doc_types
        choice_var = StringVar(value=KEEP_EXISTING)
        outcome = {"proceed": False}

        win = Toplevel(self.root)
        win.title("Regenerate sidecars")
        win.transient(self.root)
        win.grab_set()
        Label(win, text=f"Re-run auto-sidecar detection for all {len(self.image_paths)} "
                         f"image(s) in this folder.", justify="left").pack(padx=12, pady=(12, 8), anchor="w")
        Label(win, text="doc_type for this batch:").pack(padx=12, anchor="w")
        OptionMenu(win, choice_var, *options).pack(padx=12, pady=(2, 4), fill="x")
        Label(win,
              text="'(keep existing...)' preserves each page's own already-classified doc_type\n"
                   "(the normal default - never reclassifies). Pick a specific doc_type only to\n"
                   "FORCE every page in this folder onto that template - use this for pages stuck\n"
                   "on a wrong doc_type from before their columns_<doc_type>.txt override existed.\n"
                   "Any existing correction record is reconciled against the forced template\n"
                   "(carrying forward already-reviewed positions), not just blindly re-merged.",
              justify="left", font=("Segoe UI", 8), fg="#666").pack(padx=12, pady=(0, 8), anchor="w")

        def on_ok():
            outcome["proceed"] = True
            win.destroy()

        def on_cancel():
            win.destroy()

        btn_row = Frame(win)
        btn_row.pack(padx=12, pady=(0, 12), fill="x")
        Button(btn_row, text="Continue", command=on_ok).pack(side="right")
        Button(btn_row, text="Cancel", command=on_cancel).pack(side="right", padx=(0, 6))

        win.wait_window()
        chosen = choice_var.get()
        override = None if chosen == KEEP_EXISTING else chosen
        return outcome["proceed"], override

    def _on_regenerate_sidecars(self) -> None:
        """
        2026-08-09, per Jon - built after discovering a real stale-data
        bug the hard way: core.calibration_workspace.prepare_evaluation_
        workspace() runs generate_auto_sidecar() exactly ONCE, at the
        moment a raw image is first copied into a workspace, using
        whatever the doc_type template looked like at that instant.
        processed_sources.json then marks it done forever - a later
        template fix (e.g. the 2026-08-09 canada_census_1921.yaml row-
        pitch correction) silently never reaches already-imported images
        unless something forces a redo. That produced real corrupted
        data here: a row bbox clamped to the exact page-bottom pixel,
        table_bottom deltas swinging wildly from -81px to +398px across
        6 otherwise-identical samples - all just stale detections from
        the pre-fix template, not real per-page variance.

        Deliberately operates ONLY on self.image_paths - the ALREADY
        split/deskewed working images this session has open - never
        raw_dir/prepare_evaluation_workspace(). Per Jon's explicit
        instruction: re-running against the raw source folder would
        re-trigger copy_to_working_dir()/_split_image() from scratch,
        risking the exact "2 census forms on 1 image" dual-page-split
        problem that motivated building interactive per-page split in
        the first place. This only re-runs the DETECTION step against
        images that already physically exist as single-page working
        copies on disk.

        For each image: re-reads its EXISTING sidecar's own doc_type (so
        a folder with several doc_types, or one that used the CV
        fallback, keeps whatever classification a human already
        implicitly accepted by working the page - never re-classifies),
        re-runs generate_auto_sidecar() fresh against the current
        template, overwrites the base {stem}_sidecar.json, and - only if
        a correction record already exists for that page (a human has
        started/finished reviewing it) - re-merges it against the fresh
        sidecar and overwrites {stem}_merged_sidecar.json too. The
        correction record ITSELF (placed dividers, table edges,
        review_status) is never touched - only the auto-detected
        baseline underneath it, exactly like a template fix should
        propagate.
        """
        if not self.image_paths:
            return
        proceed, doc_type_override_choice = self._prompt_doc_type_override()
        if not proceed:
            return

        ok_count = 0
        failed_count = 0
        for path in self.image_paths:
            stem = path.stem
            existing_sidecar_path = self._find_sidecar_for(stem)
            existing_sidecar = load_sidecar(existing_sidecar_path) if existing_sidecar_path else None
            doc_type = doc_type_override_choice or (existing_sidecar or {}).get("parameters", {}).get("doc_type")
            working_image = None
            if existing_sidecar is not None:
                working_image = find_source_image(existing_sidecar, stem)
            working_image = Path(working_image) if working_image else path

            self.status_label.config(text=f"Regenerating {stem} ({ok_count + failed_count + 1}/"
                                           f"{len(self.image_paths)})...")
            self.root.update_idletasks()

            # Apply any saved manual deskew correction BEFORE this fresh
            # detection pass (2026-08-09, real bug - Jon: "these images
            # are skewed even though deskew angle was applied in the
            # masking stage"). generate_auto_sidecar() always re-
            # estimates deskew from scratch and had no way to accept a
            # verified correction until deskew_angle_override was added
            # - without this, every regenerate silently threw away a
            # human-confirmed deskew fix. Computes the SAME fresh auto-
            # estimate generate_auto_sidecar() would use internally, then
            # adds the saved delta on top - mirrors export_merged_
            # sidecar()'s own "sidecar's own deskew_angle + preview
            # delta" formula exactly, just applied at regeneration time
            # instead of export time.
            corr_path = correction_path_for(stem)
            correction_record = load_correction_record(corr_path) if corr_path.exists() else None
            manual_delta = (correction_record or {}).get("manual_deskew_delta_deg", 0.0) or 0.0
            deskew_override = None
            if manual_delta:
                try:
                    original_image = Image.open(str(working_image)).convert("RGB")
                    base_angle = estimate_deskew_angle(original_image, angle_range=DESKEW_ANGLE_RANGE)
                    deskew_override = base_angle + manual_delta
                except Exception:
                    deskew_override = None  # fall back to a fresh auto-estimate rather than fail the page

            try:
                result = generate_auto_sidecar(
                    str(working_image), debug=False, doc_type_override=doc_type,
                    deskew_angle_override=deskew_override)
            except Exception as e:
                self.status_label.config(text=f"{stem}: regeneration FAILED - {e}")
                failed_count += 1
                continue
            if result.sidecar is None:
                failed_count += 1
                continue

            out_sidecar_path = self.sidecar_dir / f"{stem}_sidecar.json"
            save_sidecar(result.sidecar, out_sidecar_path)
            ok_count += 1

            if correction_record is not None:
                # Forced doc_type differs from what this record was last
                # built against (2026-08-11, per Jon - real stuck-at-
                # wrong-doc_type pages found: a page first processed
                # before its columns_<doc_type>.txt override existed
                # gets a wrong doc_type baked into BOTH its sidecar and
                # correction record forever, since the normal regenerate
                # path above deliberately never re-classifies - see this
                # method's own docstring). A blind merge_column_
                # corrections() here would merge the fresh (correct)
                # sidecar against a correction record still built for
                # the WRONG column set/count - instead, fix the record's
                # own doc_type and rebuild it via load_or_build_
                # correction_record(), the same column_order-reconcile-
                # and-carry-forward-reviewed-positions logic _load_page()
                # itself uses whenever a template change is detected.
                if doc_type_override_choice and correction_record.get("doc_type") != doc_type_override_choice:
                    try:
                        forced_template = load_template(doc_type_override_choice)
                        forced_cols = (load_column_list(forced_template, override_path=None)
                                       or forced_template.expected_columns)
                    except FileNotFoundError:
                        forced_cols = None
                    if forced_cols:
                        # corr_path was already resolved above (used for
                        # the manual-deskew-delta lookup) - same path.
                        correction_record = load_or_build_correction_record(
                            result.sidecar, out_sidecar_path, corr_path,
                            doc_type_override_choice, forced_cols, forced_template.columns_file,
                        )
                        # load_or_build_correction_record() only rebuilds
                        # (and thus only sets doc_type fresh) when
                        # column_order actually changed - a record whose
                        # column_order happens to ALREADY match (real
                        # case found 2026-08-11: 3 records stuck with
                        # doc_type="canada_census_1931" despite already
                        # holding the correct 1911 column_order) hits its
                        # early-return path unchanged, doc_type included.
                        # Set it explicitly so the force-override always
                        # takes, regardless of which path fired.
                        correction_record["doc_type"] = doc_type_override_choice
                        save_correction_record(correction_record, corr_path)
                merged = merge_column_corrections(result.sidecar, correction_record)
                merged_path = self.sidecar_dir / f"{stem}_merged_sidecar.json"
                with open(merged_path, "w", encoding="utf-8") as f:
                    json.dump(merged, f, indent=2)

        self.status_label.config(
            text=f"Regenerated {ok_count}/{len(self.image_paths)} sidecar(s) "
                 f"({failed_count} failed) against the current template.")
        self._load_page()

    def _on_preview_template_refinement(self) -> None:
        """
        2026-08-09, per Jon's own spec: "Regenerate sidecars -> mark
        geometry -> mark Done -> Preview refinement -> inspect spreads
        -> manually/apply update." PREVIEW ONLY - computes suggested
        template values from real, human-reviewed correction records
        (core.template_refinement.refine_template_from_calibration())
        and shows them in a scrollable read-only dialog; never writes
        the template YAML itself (see that module's own docstring for
        why - a later "Apply to template" step is deliberately deferred
        until this preview is trusted across more than one document
        year).
        """
        if self.template is None:
            messagebox.showinfo("Column Calibration",
                                 "No template loaded for the current page - nothing to refine.")
            return
        result = refine_template_from_calibration(self.template.doc_type)
        report = format_refinement_report(result)

        win = Toplevel(self.root)
        win.title(f"Template refinement preview - {self.template.doc_type}")
        win.geometry("640x720")
        text = Text(win, font=("Consolas", 10), wrap="none")
        vbar = Scrollbar(win, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=vbar.set)
        text.pack(side="left", fill="both", expand=True)
        vbar.pack(side="right", fill="y")
        text.insert("1.0", report)
        text.configure(state="disabled")

    # ------------------------------------------------------------ render
    def _render(self) -> None:
        if self.original_image is None:
            return
        if self._preview_rotation_delta != 0.0:
            # TEST-ONLY preview rotation from "Test rotation refinement"
            # (see _on_test_rotation_refinement()) - expand=False (fixed
            # canvas/coordinate frame, center pivot) so the divider
            # overlay's existing calibrated pixel coordinates stay valid
            # to draw on top without any per-delta coordinate transform.
            # Never written to disk - purely a visual "does this look
            # more aligned" check while Jon keeps working on this page.
            fill = (255, 255, 255) if self.original_image.mode == "RGB" else 255
            overlay = self.original_image.rotate(
                self._preview_rotation_delta, resample=Image.BICUBIC, expand=False, fillcolor=fill)
        else:
            overlay = self.original_image.copy()
        draw = ImageDraw.Draw(overlay)
        h = overlay.height

        table_bbox = (self.correction_record or {}).get("table_bbox") or (self.sidecar or {}).get("table_bbox")
        if table_bbox:
            x0, y0, x1, y1 = table_bbox
            w = overlay.width
            # Human-touched edges (added 2026-08-08, "Mark table left/
            # right edges"; "top" added 2026-08-09, "bottom" added
            # 2026-08-09 same day - per Jon: "would it be easier to just
            # mark the table bottom for templating?") render like a
            # confirmed divider - solid, bold, COLOR_HUMAN - so it's
            # visually obvious which edges are real calibration data vs.
            # still just the auto-detected frame boundary (COLOR_TABLE_
            # EDGE, faint, unchanged).
            table_edges = (self.correction_record or {}).get("table_edges", {})
            left_touched = table_edges.get("left", {}).get("provenance") != PROVENANCE_UNREVIEWED
            right_touched = table_edges.get("right", {}).get("provenance") != PROVENANCE_UNREVIEWED
            top_touched = table_edges.get("top", {}).get("provenance") != PROVENANCE_UNREVIEWED
            bottom_touched = table_edges.get("bottom", {}).get("provenance") != PROVENANCE_UNREVIEWED
            draw.line([(x0, 0), (x0, h)], fill=COLOR_HUMAN if left_touched else COLOR_TABLE_EDGE,
                      width=3 if left_touched else 1)
            draw.line([(x1, 0), (x1, h)], fill=COLOR_HUMAN if right_touched else COLOR_TABLE_EDGE,
                      width=3 if right_touched else 1)
            draw.line([(0, y0), (w, y0)], fill=COLOR_HUMAN if top_touched else COLOR_TABLE_EDGE,
                      width=3 if top_touched else 1)
            draw.line([(0, y1), (w, y1)], fill=COLOR_HUMAN if bottom_touched else COLOR_TABLE_EDGE,
                      width=3 if bottom_touched else 1)

            # Crop-left projected suggestion (2026-08-09, per Jon's
            # "crop_left_of_marker" design) - only shown once table_left
            # itself is human-confirmed (left_touched), since the
            # suggestion is defined relative to it. Purely advisory -
            # see apply_left_crop()'s docstring/the "Crop left edge..."
            # tool for why this is never auto-applied.
            if left_touched:
                # Per-template override (2026-08-10, per Jon: "config
                # driven per year so this doesn't have to dig through
                # lines of code") - see DocumentTemplate.
                # crop_left_suggested_offset_px's own docstring. None
                # (no per-template value set) falls back to this
                # module's shared default.
                crop_left_offset_px = (
                    (self.template.crop_left_suggested_offset_px if self.template else None)
                    or _CROP_LEFT_SUGGESTED_OFFSET_PX
                )
                crop_suggest_x = max(0, x0 - crop_left_offset_px)
                dash = 8
                yy = 0
                while yy < h:
                    draw.line([(crop_suggest_x, yy), (crop_suggest_x, min(yy + dash, h))],
                              fill=COLOR_CROP_LEFT_SUGGESTED, width=2)
                    yy += dash * 2
                draw.text((crop_suggest_x + 4, h // 2), f"suggested crop (-{crop_left_offset_px}px)",
                          fill=COLOR_CROP_LEFT_SUGGESTED)

            # Crop-top projected suggestion (2026-08-09, same day) -
            # mirrors the crop-left one above exactly, only shown once
            # table_top is human-confirmed (top_touched).
            if top_touched:
                # Per-template override - see the crop-left block above
                # for the same pattern/rationale. Added 2026-08-10 after
                # 1906's validated 600px value turned out unsafe for
                # 1911 (landed inside 1911's real metadata block - see
                # canada_census_1911.yaml's own crop_top_suggested_
                # offset_px comment for the measured evidence).
                crop_top_offset_px = (
                    (self.template.crop_top_suggested_offset_px if self.template else None)
                    or _CROP_TOP_SUGGESTED_OFFSET_PX
                )
                crop_suggest_y = max(0, y0 - crop_top_offset_px)
                dash = 8
                xx = 0
                while xx < w:
                    draw.line([(xx, crop_suggest_y), (min(xx + dash, w), crop_suggest_y)],
                              fill=COLOR_CROP_TOP_SUGGESTED, width=2)
                    xx += dash * 2
                draw.text((w // 2, crop_suggest_y + 4), f"suggested crop (-{crop_top_offset_px}px)",
                          fill=COLOR_CROP_TOP_SUGGESTED)

            # Pixel ruler (2026-08-10) - see _arm_ruler_mode()'s own
            # docstring. Draws a full-width/height ruler anchored at the
            # last clicked point, ticked every RULER_TICK_PX with a
            # signed px-distance-from-origin label every RULER_LABEL_
            # EVERY ticks - deliberately signed (+/-), not absolute,
            # since "how far and which direction from where I clicked"
            # is the actual question this tool answers (e.g. "is the
            # rule 700 or 800px above table_top" - see the crop_top_
            # suggested_offset_px measurement work this tool replaces).
            if self._ruler_origin is not None:
                RULER_TICK_PX = 25
                RULER_LABEL_EVERY = 4  # label every 4th tick = every 100px
                ox, oy = self._ruler_origin
                # small crosshair at the origin itself, always shown
                draw.line([(ox - 12, oy), (ox + 12, oy)], fill=COLOR_RULER, width=2)
                draw.line([(ox, oy - 12), (ox, oy + 12)], fill=COLOR_RULER, width=2)
                if self._ruler_axis == "y":
                    y = oy % RULER_TICK_PX
                    tick_i = -(oy // RULER_TICK_PX)
                    while y < h:
                        is_label_tick = (tick_i % RULER_LABEL_EVERY == 0)
                        tick_len = 26 if is_label_tick else 14
                        draw.line([(0, y), (tick_len, y)], fill=COLOR_RULER, width=2)
                        draw.line([(w - tick_len, y), (w, y)], fill=COLOR_RULER, width=2)
                        if is_label_tick:
                            dist = int(round(y - oy))
                            draw.text((tick_len + 4, y - 6), f"{dist:+d}", fill=COLOR_RULER)
                        y += RULER_TICK_PX
                        tick_i += 1
                else:
                    x = ox % RULER_TICK_PX
                    tick_i = -(ox // RULER_TICK_PX)
                    while x < w:
                        is_label_tick = (tick_i % RULER_LABEL_EVERY == 0)
                        tick_len = 26 if is_label_tick else 14
                        draw.line([(x, 0), (x, tick_len)], fill=COLOR_RULER, width=2)
                        draw.line([(x, h - tick_len), (x, h)], fill=COLOR_RULER, width=2)
                        if is_label_tick:
                            dist = int(round(x - ox))
                            draw.text((x + 4, tick_len + 4), f"{dist:+d}", fill=COLOR_RULER)
                        x += RULER_TICK_PX
                        tick_i += 1

            # Expected-row-N-end overlay (2026-08-09, per Jon - see this
            # module's COLOR_EXPECTED_ROW_END_OK/_BAD constants for the
            # full rationale). Purely advisory, never written anywhere -
            # a live sanity check against THIS template's own expected_
            # row_height_frac/expected_row_count, using table_top (y0,
            # already resolved above - reflects a human "Mark table top"
            # edit if one was made) and the actual page height.
            expected_y = self._expected_row_bottom_y(y0, h)
            if expected_y is not None:
                color = COLOR_EXPECTED_ROW_END_OK if expected_y <= h else COLOR_EXPECTED_ROW_END_BAD
                clamped_y = min(expected_y, h - 2)  # keep the line/label drawable even off-page
                dash = 10
                x = 0
                while x < w:
                    draw.line([(x, clamped_y), (min(x + dash, w), clamped_y)], fill=color, width=3)
                    x += dash * 2
                row_n = self.template.expected_row_count if self.template else "?"
                label = f"expected row {row_n} end (y={expected_y:.0f})"
                if expected_y > h:
                    label += f" - {expected_y - h:.0f}px PAST PAGE BOTTOM"
                draw.text((4, clamped_y - 16), label, fill=color)

        if self.correction_record is not None:
            for i, d in enumerate(self.correction_record["dividers"]):
                selected = (i == self._selected_index)
                if d["provenance"] in (PROVENANCE_CONFIRMED, PROVENANCE_PLACED) and d["human_x"] is not None:
                    x = d["human_x"]
                    draw.line([(x, 0), (x, h)], fill=COLOR_HUMAN, width=4 if selected else 3)
                elif d["human_x"] is not None:
                    x = d["human_x"]
                    _draw_dashed_vline(draw, x, 0, h, COLOR_AUTO, width=2 if selected else 1)
                if d.get("projected_x") is not None and d["provenance"] == PROVENANCE_UNREVIEWED:
                    _draw_dotted_vline(draw, d["projected_x"], 0, h, COLOR_PROJECTED, width=2 if selected else 1)

                label_x = d["human_x"] if d["human_x"] is not None else d.get("projected_x")
                if label_x is not None:
                    label_color = COLOR_HUMAN if d["provenance"] != PROVENANCE_UNREVIEWED else COLOR_AUTO
                    draw.text((label_x + 3, 4), str(i), fill=label_color)

            anchor = self.correction_record["header_number_anchor"]
            if anchor.get("band_y") is not None:
                y = anchor["band_y"]
                draw.line([(0, y), (overlay.width, y)], fill=COLOR_ANCHOR, width=2)
                for cx, _cw in (anchor.get("detected_centers") or []):
                    draw.ellipse([cx - 3, y - 3, cx + 3, y + 3], outline=COLOR_ANCHOR, width=1)

            row_number = self.correction_record["row_number_anchor"]
            if row_number.get("left_x") is not None:
                x = row_number["left_x"]
                draw.line([(x, 0), (x, h)], fill=COLOR_ROW_NUMBER, width=2)
                draw.text((x + 3, h - 16), "row-# left", fill=COLOR_ROW_NUMBER)
            if row_number.get("right_x") is not None:
                x = row_number["right_x"]
                draw.line([(x, 0), (x, h)], fill=COLOR_ROW_NUMBER, width=2)
                draw.text((x + 3, h - 16), "row-# right", fill=COLOR_ROW_NUMBER)

        # Kept for "Save overlay PNG..." (2026-09-02, per Jon: "makes
        # canary testing fast and easy") - the exact full-resolution
        # composed overlay the canvas shows, before any zoom scaling.
        self._last_overlay = overlay
        self._render_preview(overlay)
        self._refresh_divider_list()
        self._update_row_pitch_info()

    def save_overlay_png(self) -> None:
        """
        Writes the CURRENT rendered overlay (full resolution, every
        drawn element: row bands, dividers, human-touched table edges,
        row-number anchors, preview rotation if active) to a PNG -
        added 2026-09-02 for geometry canary checks: the volunteer-GT
        1911 experiment burned two full extraction sweeps on crops that
        a 10-second look at an overlay would have condemned (see
        experiments/volunteer_gt_1911_20260902/PROVENANCE.md and the
        crop-validation pre-flight gate it establishes). Pure export of
        what is already on screen - draws nothing new, changes no
        state, same "only an explicit action writes real output"
        discipline as Export merged sidecar.
        """
        if getattr(self, "_last_overlay", None) is None:
            messagebox.showinfo("Save overlay", "Nothing rendered yet.")
            return
        stem = ""
        try:
            stem = Path(self._working_image_path).stem
        except Exception:
            pass
        out_path = filedialog.asksaveasfilename(
            title="Save calibration overlay PNG",
            defaultextension=".png",
            initialfile=f"{stem}_calibration_overlay.png" if stem else "calibration_overlay.png",
            filetypes=[("PNG image", "*.png")],
        )
        if not out_path:
            return
        self._last_overlay.save(out_path)
        messagebox.showinfo("Save overlay", f"Overlay saved:\n{out_path}")

    def _render_preview(self, overlay: Image.Image) -> None:
        zoom = ZOOM_OPTIONS[self.zoom_var.get()]
        reference_width = overlay.width
        if zoom is None:
            preview = overlay.copy()
            preview.thumbnail(PREVIEW_SIZE, Image.LANCZOS)
        else:
            preview = overlay.resize((int(overlay.width * zoom), int(overlay.height * zoom)), Image.NEAREST)

        self._preview_scale = preview.width / reference_width if reference_width else 1.0
        self._preview_render_size = (preview.width, preview.height)

        try:
            x_frac = self.canvas.xview()[0]
            y_frac = self.canvas.yview()[0]
        except Exception:
            x_frac = y_frac = 0.0

        self._tk_preview = ImageTk.PhotoImage(preview)
        self.canvas.delete("all")
        # delete("all") above also wipes the guide-line item (see
        # _on_canvas_motion()) - its old ID is now invalid, so drop the
        # reference and let the next mouse-motion event recreate it
        # fresh rather than risk a stale-ID error from canvas.coords().
        self._guide_line_id = None
        self.canvas.create_image(0, 0, anchor="nw", image=self._tk_preview)
        self.canvas.configure(scrollregion=(0, 0, preview.width, preview.height))
        self.canvas.xview_moveto(x_frac)
        self.canvas.yview_moveto(y_frac)

    def _expected_row_bottom_y(self, table_top: float, page_height: int) -> float | None:
        """
        Where this template's OWN expected_row_height_frac x
        expected_row_count says the last row should end, in image-space
        y - the pure arithmetic core.row_segmentation.
        segment_rows_uniform_tile() itself does at generation time,
        computed here ahead of time as a sanity check (2026-08-09, per
        Jon). table_top is deliberately a PARAMETER, not re-read from
        self.correction_record here, so the SAME call in _render() can
        feed it the just-resolved table_bbox value (reflecting a human
        "Mark table top" edit) without a second, possibly-inconsistent
        lookup. Returns None if there's no template loaded, or the
        template has no expected_row_height_frac/expected_row_count set
        (both optional fields - a template without them simply has
        nothing for this check to compare against, same as before this
        feature existed).
        """
        if self.template is None:
            return None
        row_frac = self.template.expected_row_height_frac
        row_count = self.template.expected_row_count
        if row_frac is None or not row_count:
            return None
        return table_top + (row_frac * page_height * row_count)

    def _update_row_pitch_info(self) -> None:
        """Live sidebar readout (2026-08-09, per Jon: "Table height...
        Expected row pitch... Expected row 40: y=..." - "exactly the
        sort of sanity check that would have exposed the original pitch
        bug before it ever reached AutoSidecar"). Pure display - all the
        real computation lives in _expected_row_bottom_y(), reused
        as-is rather than duplicated here."""
        table_bbox = (self.correction_record or {}).get("table_bbox") or (self.sidecar or {}).get("table_bbox")
        if not table_bbox or self.original_image is None or self.template is None:
            self.row_pitch_info_var.set("")
            return
        _x0, y0, _x1, y1 = table_bbox
        page_height = self.original_image.height
        row_frac = self.template.expected_row_height_frac
        row_count = self.template.expected_row_count
        if row_frac is None or not row_count:
            self.row_pitch_info_var.set("")
            return
        pitch_px = row_frac * page_height
        expected_end = y0 + pitch_px * row_count
        lines = [
            f"Table height (auto bottom): {y1 - y0:.0f} px",
            f"Expected row pitch: {pitch_px:.1f} px",
            f"Expected row {row_count} end: y={expected_end:.0f}",
        ]
        if expected_end > page_height:
            lines.append(f"** {expected_end - page_height:.0f}px PAST PAGE BOTTOM **")
        if self._pitch_measure_result:
            lines.append("")
            lines.append(self._pitch_measure_result)
        self.row_pitch_info_var.set("\n".join(lines))

    def _refresh_divider_list(self) -> None:
        self.divider_listbox.delete(0, "end")
        if self.correction_record is None:
            return
        for i, d in enumerate(self.correction_record["dividers"]):
            marker = PROVENANCE_MARKER.get(d["provenance"], "?")
            pos = d["human_x"] if d["human_x"] is not None else d.get("projected_x")
            pos_label = str(pos) if pos is not None else "-"
            self.divider_listbox.insert(
                "end", f"[{marker}] {i}: {d['left_column']} | {d['right_column']}  x={pos_label}")
        if self._selected_index is not None:
            self.divider_listbox.selection_set(self._selected_index)

    # --------------------------------------------------- coordinate math
    def _image_x(self, event) -> int:
        canvas_x = self.canvas.canvasx(event.x)
        return int(round(canvas_x / (self._preview_scale or 1.0)))

    def _image_y(self, event) -> int:
        canvas_y = self.canvas.canvasy(event.y)
        return int(round(canvas_y / (self._preview_scale or 1.0)))

    def _hit_radius_image_px(self) -> float:
        return DIVIDER_HIT_RADIUS_PX / (self._preview_scale or 1.0)

    def _current_placement_axis(self) -> str | None:
        """
        Returns "x" (vertical guide line) if the NEXT click would place
        something along the x-axis, "y" (horizontal guide line) if the
        y-axis, or None if a click right now wouldn't place anything
        (matches _on_click()'s own dispatch order/logic exactly, so the
        guide line only ever appears when it's telling the truth about
        what a click would do).
        """
        if self._ruler_mode_armed:
            # Ruler mode's own axis toggle decides the guide, not the
            # fixed y/x split every other mode below uses - it's the
            # one mode where the user picks the axis themselves.
            return self._ruler_axis
        if (self._table_top_mode_armed or self._table_bottom_mode_armed
                or self._anchor_drag_armed or self._pitch_measure_armed
                or self._crop_top_mode_armed):
            # "bottom" (2026-08-09) and pitch-measure (2026-08-09) were
            # both added after this function was first written and never
            # wired in here - a real gap Jon caught by feel ("it doesn't
            # use the guide for the later added click to mark functions
            # though"), not by design. Both are single-Y-axis-click
            # placements same as table-top/anchor, so they belong in the
            # same branch.
            return "y"
        if (self._table_edge_mode_armed or self._row_number_mode_armed or self._split_mode_armed
                or self._crop_left_mode_armed):
            return "x"
        if (self.correction_record is not None and self._selected_index is not None
                and 0 <= self._selected_index < len(self.correction_record["dividers"])):
            # Covers BOTH the awaiting-placement case (selected + still
            # unreviewed) and the already-reviewed-selected-divider
            # reposition fallback - both are vertical, and both are only
            # reachable via a click landing in EMPTY space (a click on an
            # existing line instead just selects/drags it - see
            # _on_click()'s own "clicked directly on an existing line"
            # branch) - the guide line is still a reasonable hint in that
            # case too, not misleading.
            return "x"
        return None

    def _on_canvas_motion(self, event) -> None:
        axis = self._current_placement_axis()
        if axis is None:
            self._on_canvas_leave(event)
            return
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        w, h = self._preview_render_size
        coords = (canvas_x, 0, canvas_x, h) if axis == "x" else (0, canvas_y, w, canvas_y)
        if self._guide_line_id is None:
            self._guide_line_id = self.canvas.create_line(
                *coords, fill=COLOR_GUIDE_LINE, width=1, dash=(4, 2))
        else:
            self.canvas.coords(self._guide_line_id, *coords)
            self.canvas.tag_raise(self._guide_line_id)

    def _on_canvas_leave(self, _event) -> None:
        if self._guide_line_id is not None:
            self.canvas.delete(self._guide_line_id)
            self._guide_line_id = None

    def _nearest_divider(self, image_x: int) -> int | None:
        if self.correction_record is None:
            return None
        best_i, best_dist = None, None
        for i, d in enumerate(self.correction_record["dividers"]):
            x = d["human_x"] if d["human_x"] is not None else d.get("projected_x")
            if x is None:
                continue
            dist = abs(x - image_x)
            if dist <= self._hit_radius_image_px() and (best_dist is None or dist < best_dist):
                best_i, best_dist = i, dist
        return best_i

    def _select_divider(self, idx: int) -> None:
        self._selected_index = idx
        self._render()

    def _advance_to_next_unreviewed(self, from_index: int) -> None:
        """
        Selects the next divider AFTER from_index (left-to-right, matching
        column_order's own physical order) whose provenance is still
        auto_accepted_unreviewed - added 2026-08-07, per Jon: "once a line
        is placed the list should progress so i can move left to right and
        go click click click click." Called after every action that
        reviews a divider (click-to-place, drag, Confirm selected as-is,
        Accept projected value), so the whole page can be worked start to
        finish without ever touching the side listbox. Deliberately does
        NOT wrap around back to index 0 - "left to right" is a directional
        workflow, not a cycle; a divider skipped earlier (or one out of
        order) is still reachable via a manual listbox click same as
        before, this is purely a forward-progress shortcut. Leaves
        selection on the current divider (does nothing) if none remain
        unreviewed further right - the natural "you're done" state.
        """
        if self.correction_record is None:
            return
        dividers = self.correction_record["dividers"]
        for idx in range(from_index + 1, len(dividers)):
            if dividers[idx]["provenance"] == PROVENANCE_UNREVIEWED:
                self._select_divider(idx)
                return

    # ------------------------------------------------------- undo/redo
    def _push_undo(self, index: int, before: dict, after: dict) -> None:
        self.undo_stack.append({"index": index, "before": before, "after": after})
        self.redo_stack = []

    def undo(self) -> None:
        if not self.undo_stack or self.correction_record is None:
            return
        entry = self.undo_stack.pop()
        self.correction_record["dividers"][entry["index"]].update(entry["before"])
        self.redo_stack.append(entry)
        self._dirty = True
        self._render()

    def redo(self) -> None:
        if not self.redo_stack or self.correction_record is None:
            return
        entry = self.redo_stack.pop()
        self.correction_record["dividers"][entry["index"]].update(entry["after"])
        self.undo_stack.append(entry)
        self._dirty = True
        self._render()

    # ------------------------------------------------------ mouse events
    def _on_click(self, event) -> None:
        if self._split_mode_armed:
            self._handle_split_click(event)
            return
        if self._table_edge_mode_armed:
            self._handle_table_edge_click(event)
            return
        if self._table_top_mode_armed:
            self._set_table_top_from_event(event)
            return
        if self._table_bottom_mode_armed:
            self._set_table_bottom_from_event(event)
            return
        if self._pitch_measure_armed:
            self._handle_pitch_measure_click(event)
            return
        if self._crop_left_mode_armed:
            self._handle_crop_left_click(event)
            return
        if self._crop_top_mode_armed:
            self._handle_crop_top_click(event)
            return
        if self._ruler_mode_armed:
            self._handle_ruler_click(event)
            return
        if self._row_number_mode_armed:
            self._handle_row_number_click(event)
            return
        if self._anchor_drag_armed:
            self._set_anchor_from_event(event)
            return
        if self.correction_record is None:
            return

        image_x = self._image_x(event)

        # AWAITING-PLACEMENT PRIORITY (2026-08-08, real bug found via a
        # real page with multiple closely-spaced auto/projected lines
        # already visible - Jon: "it skips 2 dividers on the list and
        # places that click 2 dividers down"). When the currently
        # selected divider is still UNREVIEWED, this click's whole
        # purpose is to place THAT divider - checked and handled FIRST,
        # before any proximity search, so a click meant for the selected
        # divider can never be silently hijacked by landing within hit-
        # radius of some OTHER, unrelated divider's pre-existing auto_x/
        # projected_x line (exactly what was happening: the click landed
        # near a later divider's already-visible proposed position,
        # _nearest_divider() found THAT one instead, and the resulting
        # select+drag+release confirmed the wrong divider while the
        # actually-selected one silently stayed unplaced). The
        # proximity-based "click directly on an existing line to grab
        # it" behavior below still applies once a divider has already
        # been reviewed (adjusting a past placement), just never takes
        # priority over an armed, not-yet-placed selection.
        if (self._selected_index is not None
                and self.correction_record["dividers"][self._selected_index]["provenance"] == PROVENANCE_UNREVIEWED):
            idx = self._selected_index
            before = dict(self.correction_record["dividers"][idx])
            apply_divider_move(self.correction_record, idx, image_x)
            after = dict(self.correction_record["dividers"][idx])
            self._push_undo(idx, before, after)
            self._dirty = True
            self._advance_to_next_unreviewed(idx)
            return

        idx = self._nearest_divider(image_x)
        if idx is not None:
            # Clicked directly on an existing line (auto or human) -
            # select it and arm a drag. If the mouse never actually
            # moves before release, _on_release still commits at the
            # same position, which is a deliberate "click confirms in
            # place" shortcut alongside the explicit Confirm button.
            self._select_divider(idx)
            self._drag_index = idx
            self._drag_before = dict(self.correction_record["dividers"][idx])
            return

        if self._selected_index is not None:
            # Empty-space click with an already-reviewed divider selected
            # (re-placing/adjusting a past placement) - place/move THAT
            # divider here. Avoids guessing which of several undefined
            # dividers a click was meant for.
            idx = self._selected_index
            before = dict(self.correction_record["dividers"][idx])
            apply_divider_move(self.correction_record, idx, image_x)
            after = dict(self.correction_record["dividers"][idx])
            self._push_undo(idx, before, after)
            self._dirty = True
            self._advance_to_next_unreviewed(idx)

    def _on_drag(self, event) -> None:
        if self._drag_index is None or self.correction_record is None:
            return
        image_x = self._image_x(event)
        self.correction_record["dividers"][self._drag_index]["human_x"] = image_x
        self._render()

    def _on_release(self, event) -> None:
        if self._drag_index is None or self.correction_record is None:
            self._drag_index = None
            return
        idx = self._drag_index
        image_x = self._image_x(event)
        before = self._drag_before
        apply_divider_move(self.correction_record, idx, image_x)
        after = dict(self.correction_record["dividers"][idx])
        self._push_undo(idx, before, after)
        self._dirty = True
        self._drag_index = None
        self._drag_before = None
        self._advance_to_next_unreviewed(idx)

    def _on_right_click(self, event) -> None:
        if self.correction_record is None:
            return
        image_x = self._image_x(event)
        idx = self._nearest_divider(image_x)
        if idx is not None:
            self._clear_and_record(idx)
            return
        # Table-edge clear (2026-08-09, per Jon - a real damaged-scan
        # case where a guessed edge became the biggest outlier in a
        # refinement preview). Only reachable when the click ISN'T near
        # a divider (checked first, above) - mirrors that same "right-
        # click an existing line to clear it" gesture at table-edge
        # granularity, never moves table_bbox itself.
        side = self._nearest_table_edge(event)
        if side is not None:
            clear_table_edge(self.correction_record, side)
            self._dirty = True
            self.status_label.config(text=f"Cleared table {side} edge back to unreviewed.")
            self._render()

    def _nearest_table_edge(self, event) -> str | None:
        table_bbox = (self.correction_record or {}).get("table_bbox")
        if not table_bbox:
            return None
        x0, y0, x1, y1 = table_bbox
        image_x = self._image_x(event)
        image_y = self._image_y(event)
        radius = self._hit_radius_image_px()
        candidates = {
            "left": abs(image_x - x0),
            "right": abs(image_x - x1),
            "top": abs(image_y - y0),
            "bottom": abs(image_y - y1),
        }
        best_side, best_dist = None, None
        for side, dist in candidates.items():
            if dist <= radius and (best_dist is None or dist < best_dist):
                best_side, best_dist = side, dist
        return best_side

    def _on_delete_key(self, _event) -> None:
        if self._selected_index is not None and self.correction_record is not None:
            self._clear_and_record(self._selected_index)

    def _clear_and_record(self, idx: int) -> None:
        before = dict(self.correction_record["dividers"][idx])
        clear_divider(self.correction_record, idx)
        after = dict(self.correction_record["dividers"][idx])
        self._push_undo(idx, before, after)
        self._dirty = True
        self._render()

    def _on_listbox_select(self, _event) -> None:
        sel = self.divider_listbox.curselection()
        if sel:
            self._select_divider(sel[0])

    # ------------------------------------------------------ anchor line
    def _arm_anchor_mode(self) -> None:
        if self.correction_record is None:
            messagebox.showinfo("Column Calibration", "Load a page with a sidecar first.")
            return
        self._row_number_mode_armed = False  # mutually exclusive with the other click-modes
        self._split_mode_armed = False
        self._table_edge_mode_armed = False
        self._table_top_mode_armed = False
        self._table_bottom_mode_armed = False
        self._pitch_measure_armed = False
        self._crop_left_mode_armed = False
        self._crop_top_mode_armed = False
        self._ruler_mode_armed = False
        self._anchor_drag_armed = True
        self.status_label.config(text="Click on the printed column-number row to set the anchor line.")

    def _arm_table_top_mode(self) -> None:
        """
        Single-click table-TOP mode (2026-08-09, per Jon - see this
        class's own __init__ comment for the real motivating finding).
        Same "click again to cancel" toggle as the other armed modes,
        mirroring _arm_anchor_mode()'s exact structure (also a single
        Y-axis click) rather than the 2-click left/right table-edge
        pattern.
        """
        if self.correction_record is None:
            messagebox.showinfo("Column Calibration", "Load a page with a sidecar first.")
            return
        if self._table_top_mode_armed:
            self._table_top_mode_armed = False
            self.status_label.config(text="Table top mode cancelled.")
            return
        self._anchor_drag_armed = False  # mutually exclusive with the other click-modes
        self._row_number_mode_armed = False
        self._split_mode_armed = False
        self._table_edge_mode_armed = False
        self._table_bottom_mode_armed = False
        self._pitch_measure_armed = False
        self._crop_left_mode_armed = False
        self._crop_top_mode_armed = False
        self._ruler_mode_armed = False
        self._table_top_mode_armed = True
        self.status_label.config(text="Click the TRUE table top edge.")

    def _set_table_top_from_event(self, event) -> None:
        self._table_top_mode_armed = False
        if self.correction_record is None:
            return
        image_y = self._image_y(event)
        apply_table_edge_move(self.correction_record, "top", image_y)
        self._dirty = True
        self.status_label.config(text=f"Table top set at y={image_y}.")
        self._render()

    def _arm_table_bottom_mode(self) -> None:
        """
        Single-click table-BOTTOM mode (2026-08-09, per Jon: "would it be
        easier to just mark the table bottom for templating?"). Mirrors
        _arm_table_top_mode() exactly - a real marked bottom (table_bbox[3])
        lets row pitch be computed exactly per-page from (bottom-top)/
        (row_count-1) instead of an averaged template constant, which is
        the real source of the cross-page variance the row-pitch sanity
        check and "Measure row pitch" tool were built to surface. This is
        data capture only - core/auto_sidecar.py's tiling doesn't consume
        it yet (that's the deliberate next step after real samples are
        inspected).
        """
        if self.correction_record is None:
            messagebox.showinfo("Column Calibration", "Load a page with a sidecar first.")
            return
        if self._table_bottom_mode_armed:
            self._table_bottom_mode_armed = False
            self.status_label.config(text="Table bottom mode cancelled.")
            return
        self._anchor_drag_armed = False  # mutually exclusive with the other click-modes
        self._row_number_mode_armed = False
        self._split_mode_armed = False
        self._table_edge_mode_armed = False
        self._table_top_mode_armed = False
        self._pitch_measure_armed = False
        self._crop_left_mode_armed = False
        self._crop_top_mode_armed = False
        self._ruler_mode_armed = False
        self._table_bottom_mode_armed = True
        self.status_label.config(text="Click the TRUE table bottom edge.")

    def _set_table_bottom_from_event(self, event) -> None:
        self._table_bottom_mode_armed = False
        if self.correction_record is None:
            return
        image_y = self._image_y(event)
        apply_table_edge_move(self.correction_record, "bottom", image_y)
        self._dirty = True
        self.status_label.config(text=f"Table bottom set at y={image_y}.")
        self._render()

    # -------------------------------------------------------- pitch measure
    def _arm_pitch_measure_mode(self) -> None:
        """
        2-click direct pitch measurement (2026-08-09, per Jon: after
        confirming table_top was correctly marked on a real page, the
        expected-row-N-end overlay STILL didn't match row N's actual
        position - proving the mismatch was the template's averaged
        pitch, not a table_top bug. Click row 1's real top, then row N's
        real top (same row_count the template expects), and this shows
        the page's OWN real pitch next to the template's assumed one -
        lets him tell ordinary page-to-page registration variance
        (already documented for column dividers, ~5% on 1906) from an
        actual problem at a glance, instead of eyeballing overlay gaps.
        """
        if self.correction_record is None:
            messagebox.showinfo("Column Calibration", "Load a page with a sidecar first.")
            return
        if self._pitch_measure_armed:
            self._pitch_measure_armed = False
            self._pitch_measure_first_y = None
            self.status_label.config(text="Row pitch measurement cancelled.")
            return
        self._anchor_drag_armed = False  # mutually exclusive with the other click-modes
        self._row_number_mode_armed = False
        self._split_mode_armed = False
        self._table_edge_mode_armed = False
        self._table_top_mode_armed = False
        self._table_bottom_mode_armed = False
        self._crop_left_mode_armed = False
        self._crop_top_mode_armed = False
        self._ruler_mode_armed = False
        self._pitch_measure_armed = True
        self._pitch_measure_first_y = None
        row_count = self.template.expected_row_count if self.template else "N"
        self.status_label.config(text=f"Click row 1's real top edge (then row {row_count}'s).")

    def _handle_pitch_measure_click(self, event) -> None:
        image_y = self._image_y(event)
        if self._pitch_measure_first_y is None:
            self._pitch_measure_first_y = image_y
            row_count = self.template.expected_row_count if self.template else "N"
            self.status_label.config(text=f"Row 1 top set at y={image_y:.0f} - now click row {row_count}'s top.")
            return

        row1_y = self._pitch_measure_first_y
        rowN_y = image_y
        self._pitch_measure_armed = False
        self._pitch_measure_first_y = None

        row_count = self.template.expected_row_count if self.template else None
        if not row_count or row_count < 2:
            self.status_label.config(text="Template has no usable expected_row_count - can't compute pitch.")
            return
        real_pitch = (rowN_y - row1_y) / (row_count - 1)

        lines = [f"Real measured pitch (row 1-{row_count}): {real_pitch:.1f} px"]
        if self.template.expected_row_height_frac is not None and self.original_image is not None:
            template_pitch = self.template.expected_row_height_frac * self.original_image.height
            delta_pct = (real_pitch - template_pitch) / template_pitch * 100 if template_pitch else 0.0
            lines.append(f"Template assumed pitch: {template_pitch:.1f} px ({delta_pct:+.1f}%)")
        self._pitch_measure_result = "\n".join(lines)
        self.status_label.config(text="Row pitch measured - see sidebar.")

        # PERSIST into the correction record (2026-08-09, per Jon - see
        # core/column_calibration.py's SCHEMA_VERSION=6 comment for the
        # full "why": once a template's expected_row_height_frac is set,
        # core/auto_sidecar.py's fixed_periodic tiling skips its own
        # per-page search entirely and just multiplies the template
        # value by page height - so a regenerated sidecar's own rows can
        # no longer supply an INDEPENDENT measurement. This 2-click tool
        # is the one remaining source of real, template-independent
        # per-page pitch - core/template_refinement.py's refine_
        # template_from_calibration() reads THIS field, not sidecar
        # rows, for its ROW PITCH statistic. Only meaningful once saved
        # via the normal Save/Confirm Page button - not auto-saved here,
        # same "export is the only real output" discipline as every
        # other correction-record field in this tool.
        if self.correction_record is not None and self.original_image is not None:
            page_height = self.original_image.height
            self.correction_record["measured_row_pitch"] = {
                "fraction": real_pitch / page_height,
                "pixels": real_pitch,
                "page_height": page_height,
                "row_start": 1,
                "row_end": row_count,
                "y_start": row1_y,
                "y_end": rowN_y,
                "sample_count": row_count - 1,
                "provenance": "human_measured",
                "measured_at": datetime.now(timezone.utc).isoformat(),
            }
            self._dirty = True

        self._update_row_pitch_info()

    def _set_anchor_from_event(self, event) -> None:
        self._anchor_drag_armed = False
        if self.correction_record is None:
            return
        image_y = self._image_y(event)
        self.correction_record["header_number_anchor"]["band_y"] = image_y
        self.correction_record["header_number_anchor"]["band_y_source"] = "manual"
        self._dirty = True
        self.status_label.config(text=f"Header-number anchor set at y={image_y}.")
        self._render()

    def _toggle_row_number_mode(self) -> None:
        """
        Per Jon's exact spec (2026-08-07): a single button arms a 2-click
        mode - first click sets the LEFT row-number margin center, second
        click sets the RIGHT one and auto-deactivates the mode (no need
        to press the button again to exit). Re-arming the mode after both
        are already set clears both immediately, so the very next click
        starts a fresh left/right placement rather than adding a stray
        third value - this is the "redo if one is wrong" mechanism; no
        separate per-line clear affordance was asked for or built.
        """
        if self.correction_record is None:
            messagebox.showinfo("Column Calibration", "Load a page with a sidecar first.")
            return
        if self._row_number_mode_armed:
            # Pressed again mid-mode - cancel without clearing anything
            # already placed this session.
            self._row_number_mode_armed = False
            self._row_number_click_stage = 0
            self.status_label.config(text="Row-number center mode cancelled.")
            return

        self._anchor_drag_armed = False  # mutually exclusive with the other click-modes
        self._split_mode_armed = False
        self._table_edge_mode_armed = False
        self._table_top_mode_armed = False
        self._table_bottom_mode_armed = False
        self._pitch_measure_armed = False
        self._crop_left_mode_armed = False
        self._crop_top_mode_armed = False
        rn = self.correction_record["row_number_anchor"]
        if rn["left_x"] is not None and rn["right_x"] is not None:
            rn["left_x"] = None
            rn["right_x"] = None
            self._dirty = True
            self._render()
        self._row_number_mode_armed = True
        self._row_number_click_stage = 0
        self.status_label.config(text="Click the LEFT row-number margin center.")

    def _handle_row_number_click(self, event) -> None:
        if self.correction_record is None:
            self._row_number_mode_armed = False
            return
        image_x = self._image_x(event)
        rn = self.correction_record["row_number_anchor"]
        self._dirty = True
        if self._row_number_click_stage == 0:
            rn["left_x"] = image_x
            self._row_number_click_stage = 1
            self.status_label.config(text=f"Left set at x={image_x}. Click the RIGHT row-number margin center.")
        else:
            rn["right_x"] = image_x
            self._row_number_mode_armed = False
            self._row_number_click_stage = 0
            self.status_label.config(text=f"Row-number centers set: left={rn['left_x']}, right={rn['right_x']}.")
        self._render()

    # ------------------------------------------------------ table edges
    def _toggle_table_edge_mode(self) -> None:
        """
        Same 2-click-then-done pattern as _toggle_row_number_mode() -
        click LEFT table edge, click RIGHT table edge, mode auto-
        deactivates. Re-arming after both are already touched does NOT
        clear them first (unlike row-number mode) - a table edge click
        always just moves table_bbox[0]/[2] directly to wherever's
        clicked next, so there's nothing stale to clear first; the
        "redo if wrong" story is simply "click it again."
        """
        if self.correction_record is None:
            messagebox.showinfo("Column Calibration", "Load a page with a sidecar first.")
            return
        if self._table_edge_mode_armed:
            self._table_edge_mode_armed = False
            self._table_edge_click_stage = 0
            self.status_label.config(text="Table edge mode cancelled.")
            return

        self._anchor_drag_armed = False  # mutually exclusive with the other click-modes
        self._row_number_mode_armed = False
        self._split_mode_armed = False
        self._table_top_mode_armed = False
        self._table_bottom_mode_armed = False
        self._pitch_measure_armed = False
        self._crop_left_mode_armed = False
        self._crop_top_mode_armed = False
        self._ruler_mode_armed = False
        self._table_edge_mode_armed = True
        self._table_edge_click_stage = 0
        self.status_label.config(text="Click the LEFT table edge (true content boundary, not just the frame).")

    def _handle_table_edge_click(self, event) -> None:
        if self.correction_record is None:
            self._table_edge_mode_armed = False
            return
        image_x = self._image_x(event)
        self._dirty = True
        if self._table_edge_click_stage == 0:
            apply_table_edge_move(self.correction_record, "left", image_x)
            self._table_edge_click_stage = 1
            self.status_label.config(text=f"Left table edge set at x={image_x}. Click the RIGHT table edge.")
        else:
            apply_table_edge_move(self.correction_record, "right", image_x)
            self._table_edge_mode_armed = False
            self._table_edge_click_stage = 0
            tb = self.correction_record["table_bbox"]
            self.status_label.config(text=f"Table edges set: left={tb[0]}, right={tb[2]}.")
        self._render()

    # -------------------------------------------------------- crop left
    def _arm_crop_left_mode(self) -> None:
        """
        Single-click, PERMANENT left-edge crop (2026-08-09, per Jon's
        "crop_left_of_marker" design). Unlike every other mode in this
        tool, this one genuinely mutates the working image file - see
        _handle_crop_left_click()'s confirm dialog and apply_left_crop()'s
        own docstring for why it needs its own explicit action rather
        than folding into Save like the deskew bake did.
        """
        if self.correction_record is None or self.original_image is None:
            messagebox.showinfo("Column Calibration", "Load a page with a sidecar first.")
            return
        if self._crop_left_mode_armed:
            self._crop_left_mode_armed = False
            self.status_label.config(text="Crop-left mode cancelled.")
            return
        self._anchor_drag_armed = False  # mutually exclusive with the other click-modes
        self._row_number_mode_armed = False
        self._split_mode_armed = False
        self._table_edge_mode_armed = False
        self._table_top_mode_armed = False
        self._table_bottom_mode_armed = False
        self._pitch_measure_armed = False
        self._crop_top_mode_armed = False
        self._ruler_mode_armed = False
        self._crop_left_mode_armed = True
        self.status_label.config(
            text="Click where the LEFT edge should be cropped (permanent - see the dashed "
                 "blue suggestion line if table_left is already marked).")

    def _handle_crop_left_click(self, event) -> None:
        self._crop_left_mode_armed = False
        if self.correction_record is None or self.original_image is None or self._working_image_path is None:
            return
        crop_px = int(round(self._image_x(event)))
        if crop_px <= 0:
            self.status_label.config(text="Crop position must be right of the image's own left edge - cancelled.")
            return
        if crop_px >= self.original_image.width:
            self.status_label.config(text="Crop position is past the image's right edge - cancelled.")
            return

        if not messagebox.askyesno(
            "Crop left edge",
            f"Permanently crop {crop_px}px off the LEFT edge of this working image?\n\n"
            f"This cannot be undone (short of re-deriving the page from the raw source again). "
            f"Every already-placed divider/anchor/table-edge position in this page's correction "
            f"record will be shifted left by {crop_px}px to match - nothing will visually move on "
            f"screen once this completes.\n\nContinue?"):
            self.status_label.config(text="Crop-left cancelled.")
            return

        cropped = self.original_image.crop((crop_px, 0, self.original_image.width, self.original_image.height))
        cropped.save(self._working_image_path)
        self.original_image = cropped
        apply_left_crop(self.correction_record, crop_px)
        self._dirty = True
        self.status_label.config(text=f"Cropped {crop_px}px off the left edge - correction record updated to match.")
        self._render()

    def _arm_crop_top_mode(self) -> None:
        """
        Single-click, PERMANENT top-edge crop (2026-08-09, same day -
        per Jon: "we need a way to crop the top too"). Mirrors _arm_
        crop_left_mode() exactly, on the Y axis - see apply_top_crop()'s
        own docstring for the real measured safe-margin validation.
        """
        if self.correction_record is None or self.original_image is None:
            messagebox.showinfo("Column Calibration", "Load a page with a sidecar first.")
            return
        if self._crop_top_mode_armed:
            self._crop_top_mode_armed = False
            self.status_label.config(text="Crop-top mode cancelled.")
            return
        self._anchor_drag_armed = False  # mutually exclusive with the other click-modes
        self._row_number_mode_armed = False
        self._split_mode_armed = False
        self._table_edge_mode_armed = False
        self._table_top_mode_armed = False
        self._table_bottom_mode_armed = False
        self._pitch_measure_armed = False
        self._crop_left_mode_armed = False
        self._crop_top_mode_armed = True
        self.status_label.config(
            text="Click where the TOP edge should be cropped (permanent - see the dashed "
                 "blue suggestion line if table_top is already marked).")

    def _handle_crop_top_click(self, event) -> None:
        self._crop_top_mode_armed = False
        if self.correction_record is None or self.original_image is None or self._working_image_path is None:
            return
        crop_px = int(round(self._image_y(event)))
        if crop_px <= 0:
            self.status_label.config(text="Crop position must be below the image's own top edge - cancelled.")
            return
        if crop_px >= self.original_image.height:
            self.status_label.config(text="Crop position is past the image's bottom edge - cancelled.")
            return

        if not messagebox.askyesno(
            "Crop top edge",
            f"Permanently crop {crop_px}px off the TOP edge of this working image?\n\n"
            f"This cannot be undone (short of re-deriving the page from the raw source again). "
            f"Every already-placed table-edge/anchor/measured-pitch Y position in this page's "
            f"correction record will be shifted up by {crop_px}px to match - nothing will "
            f"visually move on screen once this completes.\n\nContinue?"):
            self.status_label.config(text="Crop-top cancelled.")
            return

        cropped = self.original_image.crop((0, crop_px, self.original_image.width, self.original_image.height))
        cropped.save(self._working_image_path)
        self.original_image = cropped
        apply_top_crop(self.correction_record, crop_px)
        self._dirty = True
        self.status_label.config(text=f"Cropped {crop_px}px off the top edge - correction record updated to match.")
        self._render()

    def _arm_ruler_mode(self) -> None:
        """
        Pixel-ruler tool (2026-08-10, per Jon - see the state-init
        comment next to self._ruler_mode_armed for the full backstory).
        Unlike the crop tools, clicking again while armed does NOT
        disarm - it just repositions the ruler, since re-measuring
        against a new reference point without re-arming every time is
        the whole point of a measuring tool. The toolbar button toggles
        the mode itself off; a click never does.
        """
        if self.correction_record is None or self.original_image is None:
            messagebox.showinfo("Column Calibration", "Load a page with a sidecar first.")
            return
        if self._ruler_mode_armed:
            self._ruler_mode_armed = False
            self.status_label.config(text="Ruler mode off.")
            self._render()
            return
        self._anchor_drag_armed = False  # mutually exclusive with the other click-modes
        self._row_number_mode_armed = False
        self._split_mode_armed = False
        self._table_edge_mode_armed = False
        self._table_top_mode_armed = False
        self._table_bottom_mode_armed = False
        self._pitch_measure_armed = False
        self._crop_left_mode_armed = False
        self._crop_top_mode_armed = False
        self._ruler_mode_armed = True
        self.status_label.config(
            text=f"Ruler mode ({self._ruler_axis.upper()}-axis) - click a point to measure "
                 f"distances from it. Click 'Ruler' again to turn off.")

    def _toggle_ruler_axis(self) -> None:
        self._ruler_axis = "x" if self._ruler_axis == "y" else "y"
        self.ruler_axis_btn_var.set(f"Ruler axis: {self._ruler_axis.upper()}")
        if self._ruler_mode_armed:
            self.status_label.config(text=f"Ruler mode ({self._ruler_axis.upper()}-axis).")
        self._render()

    def _handle_ruler_click(self, event) -> None:
        self._ruler_origin = (self._image_x(event), self._image_y(event))
        self.status_label.config(
            text=f"Ruler origin set at ({self._ruler_origin[0]}, {self._ruler_origin[1]}) - "
                 f"measuring along the {self._ruler_axis.upper()}-axis.")
        self._render()

    # ---------------------------------------------------------- actions
    def confirm_selected(self) -> None:
        if self.correction_record is None or self._selected_index is None:
            return
        idx = self._selected_index
        before = dict(self.correction_record["dividers"][idx])
        try:
            confirm_divider_as_is(self.correction_record, idx)
        except ValueError as e:
            messagebox.showwarning("Column Calibration", str(e))
            return
        after = dict(self.correction_record["dividers"][idx])
        self._push_undo(idx, before, after)
        self._dirty = True
        self._advance_to_next_unreviewed(idx)

    def accept_projected(self) -> None:
        if self.correction_record is None or self._selected_index is None:
            return
        idx = self._selected_index
        d = self.correction_record["dividers"][idx]
        if d.get("projected_x") is None:
            messagebox.showinfo(
                "Column Calibration",
                "Selected divider has no projected value - run 'Project from confirmed anchors' first.")
            return
        before = dict(d)
        apply_divider_move(self.correction_record, idx, d["projected_x"])
        after = dict(self.correction_record["dividers"][idx])
        self._push_undo(idx, before, after)
        self._dirty = True
        self._advance_to_next_unreviewed(idx)

    def project_remaining(self) -> None:
        if self.correction_record is None or self.template is None:
            messagebox.showinfo("Column Calibration", "Need a loaded sidecar + template first.")
            return
        # Advisory only - project_remaining_dividers() mutates only
        # projected_x, never human_x/provenance, so no undo entry is
        # needed: re-running it or clearing a divider both cleanly
        # supersede/clear the projection without losing any real edit.
        message = project_remaining_dividers(self.correction_record, self.template)
        self._dirty = True
        self.status_label.config(text=message)
        self._render()

    def save_page(self) -> None:
        if self.correction_record is None:
            messagebox.showinfo("Column Calibration", "Nothing to save for this page.")
            return

        # BAKE any confirmed manual deskew into the working image file
        # itself (2026-08-09, per Jon: "have the ui actually deskew the
        # working image when saving, then any downstream process is
        # working with the same image and not having to deskew then").
        # Only ever touches THIS calibration workspace's own working
        # copy (self._working_image_path) - never the raw source image,
        # and never any file real production extraction reads (that's a
        # completely separate corpus/path - see core.calibration_
        # workspace's own "persistent evaluation workspace, not a
        # production run" scoping). Uses the SAME expand=False, center-
        # pivot rotation _render()'s own preview already uses, so every
        # divider/anchor pixel coordinate already placed against that
        # preview stays valid on the baked file with NO coordinate
        # transform needed - baking is safe to do after calibration
        # work has already started, not just before.
        baked_note = ""
        if self._preview_rotation_delta != 0.0 and self.original_image is not None and self._working_image_path is not None:
            fill = (255, 255, 255) if self.original_image.mode == "RGB" else 255
            baked = self.original_image.rotate(
                self._preview_rotation_delta, resample=Image.BICUBIC, expand=False, fillcolor=fill)
            baked.save(self._working_image_path)
            self.original_image = baked
            baked_note = f"  (baked {self._preview_rotation_delta:+.2f} deg deskew into the working image)"
            self._preview_rotation_delta = 0.0
            self.correction_record["manual_deskew_delta_deg"] = 0.0
            self.rotation_test_status_var.set("")

        all_reviewed = all(
            d["provenance"] in (PROVENANCE_CONFIRMED, PROVENANCE_PLACED)
            for d in self.correction_record["dividers"]
        )
        self.correction_record["review_status"] = "done" if all_reviewed else "in_progress"
        save_correction_record(self.correction_record, self.correction_path)
        self.undo_stack = []
        self.redo_stack = []
        self._dirty = False
        self.status_label.config(
            text=f"Saved {self.correction_path.name}  "
                 f"(review_status: {self.correction_record['review_status']}){baked_note}")
        if baked_note:
            self._render()

    # -------------------------------------------- rotation refinement test
    def _on_test_rotation_refinement(self) -> None:
        """
        Interactive TEST of core.rotation_refinement's shared "validate
        then refine" utility against THIS page's row-number band, per
        Jon's request (2026-08-08): a button to run it and see the
        result on screen while he's already masking, before it's wired
        into any real pipeline stage. Writes NOTHING to disk - no
        sidecar/correction-record mutation, purely a preview rotation
        (self._preview_rotation_delta) plus a status-line readout of the
        evidence dict.

        Uses expand=False rotation (center-pivot, fixed canvas/
        coordinate frame) for BOTH the detector closure and the on-
        screen preview - deliberately simpler than the real deskew
        pipeline's apply_deskew_angle() (expand=True, which would need
        every anchor coordinate re-transformed per delta to stay
        correct). Reasonable for THIS narrow use case (a small +/-2 deg
        local search on top of an already-mostly-correct baseline, not a
        full-page deskew from scratch) - corner-clipping risk from
        expand=False is minimal at this angle range. Whichever expand
        convention a real pipeline wiring eventually uses is a decision
        for that later step, not this test tool.
        """
        if self.correction_record is None or self.original_image is None:
            messagebox.showinfo("Rotation Refinement Test", "Load a page with a sidecar first.")
            return
        row_number = self.correction_record["row_number_anchor"]
        left_x = row_number.get("left_x")
        if left_x is None:
            messagebox.showinfo(
                "Rotation Refinement Test",
                "No row-number left-margin center defined yet - use 'Define left/right "
                "row number centers' first, so this has a real band to search.")
            return

        table_bbox = self.correction_record.get("table_bbox")
        if not table_bbox:
            messagebox.showinfo("Rotation Refinement Test", "No table_bbox known for this page.")
            return
        _tl, table_top, _tr, table_bottom = table_bbox

        band_half_width = 20  # px either side of the calibrated row-number center
        x0 = int(left_x - band_half_width)
        x1 = int(left_x + band_half_width)

        expected_row_count = self.template.expected_row_count if self.template else None
        base_image = self.original_image

        def detect_at_delta(delta: float) -> list[BlobCandidate]:
            rotated = base_image if delta == 0.0 else base_image.rotate(
                delta, resample=Image.BICUBIC, expand=False,
                fillcolor=(255, 255, 255) if base_image.mode == "RGB" else 255,
            )
            centers = detect_row_number_centers(rotated, x0, x1, int(table_top), int(table_bottom))
            return [BlobCandidate(position=c, confidence=1.0) for c in centers]

        result = refine_rotation(
            baseline_angle=0.0, detect_at_delta=detect_at_delta,
            min_count_to_trust_baseline=max(1, int((expected_row_count or 40) * 0.8)),
            expected_count=expected_row_count,
            search_range_deg=2.0, search_step_deg=0.25, min_score_margin=0.15,
            log=print,
        )

        ev = result.evidence
        if result.used_refinement:
            self._preview_rotation_delta = result.delta_deg
            msg = (f"REFINED: delta={result.delta_deg:+.2f} deg  "
                   f"(baseline score {ev['baseline_score']:.3f} -> {ev['refined_score']:.3f}) "
                   f"- preview updated below.")
        elif result.quarantine:
            self._preview_rotation_delta = 0.0
            best_score = ev["refined_score"] if ev["refined_score"] is not None else ev["baseline_score"]
            msg = (f"QUARANTINE: no angle in +/-2 deg cleared the acceptance bar "
                   f"(baseline {ev['baseline_score']:.3f}, best tried {best_score:.3f}) - "
                   f"keeping original, no preview change.")
        else:
            self._preview_rotation_delta = 0.0
            msg = f"Baseline already confident ({result.baseline_score.count} candidate(s) found) - no search needed."

        self._save_preview_rotation_delta()
        self.rotation_test_status_var.set(msg)
        self._render()

    def _save_preview_rotation_delta(self) -> None:
        """
        Persists self._preview_rotation_delta into the correction record
        (2026-08-09, per Jon - see core/column_calibration.py's
        SCHEMA_VERSION=7 comment for the full "why"). Called after every
        site that changes the delta with real intent, so a Save/Confirm
        Page afterward carries it forward and _load_page() can restore
        it on the next reopen instead of always starting at 0.0.
        """
        if self.correction_record is not None:
            self.correction_record["manual_deskew_delta_deg"] = self._preview_rotation_delta
            self._dirty = True

    def _clear_rotation_preview(self) -> None:
        self._preview_rotation_delta = 0.0
        self._save_preview_rotation_delta()
        self.rotation_test_status_var.set("")
        self._render()

    def _nudge_deskew(self, delta: float) -> None:
        """
        Manual deskew +/- button handler. Clamped to +/-5 deg - beyond
        that this preview mechanism's expand=False, fixed-canvas
        rotation (see _on_test_rotation_refinement()'s docstring for why
        that's an acceptable simplification for a SMALL nudge) starts
        clipping page corners visibly, which would be actively
        misleading rather than just imprecise. Use "Clear rotation
        preview" to reset to 0.0 - no separate revert action needed.
        """
        if self.original_image is None:
            return
        self._preview_rotation_delta = max(-5.0, min(5.0, self._preview_rotation_delta + delta))
        self._save_preview_rotation_delta()
        self.rotation_test_status_var.set(
            f"Manual deskew nudge: {self._preview_rotation_delta:+.2f} deg (saved with this page).")
        self._render()

    def _shift_all_x(self, delta_px: int) -> None:
        """
        Shifts every already-placed X-axis position by delta_px in one
        action (2026-08-09, per Jon - see the "Shift all lines" button
        block's own comment for the full motivation). Only touches
        positions that are already SET (human_x/left_x/right_x/table
        edges not None, or a human-touched table edge) - an unplaced
        divider (human_x is None, nothing clicked yet) has nothing
        meaningful to shift and is left alone, same as every other
        placement action in this tool.

        Deliberately NOT undo-stack tracked (self.undo_stack assumes a
        single-divider before/after pair - see _push_undo()'s own
        signature) - this is a bulk operation across many fields at
        once. Re-running with the opposite-sign delta reverses it
        exactly (integer px, no rounding drift either direction), which
        is the intended "undo" path here.
        """
        if self.correction_record is None:
            return
        moved = 0
        for d in self.correction_record["dividers"]:
            if d["human_x"] is not None:
                d["human_x"] = int(round(d["human_x"] + delta_px))
                moved += 1

        table_bbox = self.correction_record.get("table_bbox")
        table_edges = self.correction_record.get("table_edges", {})
        if table_bbox:
            if table_edges.get("left", {}).get("provenance") != PROVENANCE_UNREVIEWED:
                table_bbox[0] = int(round(table_bbox[0] + delta_px))
                moved += 1
            if table_edges.get("right", {}).get("provenance") != PROVENANCE_UNREVIEWED:
                table_bbox[2] = int(round(table_bbox[2] + delta_px))
                moved += 1

        rn = self.correction_record["row_number_anchor"]
        if rn.get("left_x") is not None:
            rn["left_x"] = int(round(rn["left_x"] + delta_px))
            moved += 1
        if rn.get("right_x") is not None:
            rn["right_x"] = int(round(rn["right_x"] + delta_px))
            moved += 1

        if moved == 0:
            self.status_label.config(text="Nothing placed yet on this page - nothing to shift.")
            return
        self._dirty = True
        self.status_label.config(text=f"Shifted {moved} placed line(s) by {delta_px:+d}px.")
        self._render()

    # ------------------------------------------------------ split point
    def _arm_split_mode(self) -> None:
        """
        Arms a single click on the canvas to become this page's split
        point, replacing sole reliance on core.calibration_workspace's
        folder-wide split.split marker (one fixed fraction applied
        blindly to every image in a folder - see that module's own
        DUAL-PAGE SPLIT MARKER docstring for why an automatic brightness-
        dip detector was tried and rejected as unreliable). Mutually
        exclusive with the anchor/row-number click modes, same pattern
        as _arm_anchor_mode()/_toggle_row_number_mode().
        """
        if self.original_image is None or self._working_image_path is None:
            messagebox.showinfo("Column Calibration", "Load a page first.")
            return
        if self._split_mode_armed:
            self._split_mode_armed = False
            self.split_status_var.set("Split mode cancelled.")
            return
        self._anchor_drag_armed = False
        self._row_number_mode_armed = False
        self._table_edge_mode_armed = False
        self._table_top_mode_armed = False
        self._table_bottom_mode_armed = False
        self._pitch_measure_armed = False
        self._crop_left_mode_armed = False
        self._crop_top_mode_armed = False
        self._ruler_mode_armed = False
        self._split_mode_armed = True
        self.split_status_var.set(
            "Click where this page should be split into left/right halves.")

    def _handle_split_click(self, event) -> None:
        self._split_mode_armed = False
        if self.original_image is None or self._working_image_path is None:
            return

        image_x = self._image_x(event)
        width = self.original_image.width
        fraction = image_x / width

        if not (0.05 < fraction < 0.95):
            messagebox.showwarning(
                "Column Calibration",
                f"Split point at x={image_x} (fraction {fraction:.3f}) is too close to "
                f"an edge to be a real two-page split - cancelled, nothing changed.")
            self.split_status_var.set("")
            return

        if not messagebox.askyesno(
            "Confirm split",
            f"Split this page at fraction {fraction:.3f} (x={image_x} of {width}px) into "
            f"two new working pages, deskew/preprocess/auto-sidecar each half, replace this "
            f"page in the navigation list with the two halves, and DELETE the original "
            f"whole-page working copy (image + sidecar) once at least one half is usable?\n\n"
            f"The RAW SOURCE image is untouched either way - the working copy is a preprocessed "
            f"COPY of it (see core/calibration_workspace.py), so this never touches your "
            f"original input. This cannot be undone within this session once you click Yes."
        ):
            self.split_status_var.set("")
            return

        working_path = self._working_image_path
        doc_type_override = (self.sidecar or {}).get("parameters", {}).get("doc_type")

        try:
            with Image.open(working_path) as im:
                im = im.convert("RGB")
                w, h = im.size
                split_x = int(w * fraction)
                left = im.crop((0, 0, split_x, h))
                right = im.crop((split_x, 0, w, h))

            left_path = working_path.parent / f"{working_path.stem}_L{working_path.suffix}"
            right_path = working_path.parent / f"{working_path.stem}_R{working_path.suffix}"
            left.save(left_path)
            right.save(right_path)

            new_pages: list[Path] = []
            for half_path in (left_path, right_path):
                self.split_status_var.set(f"Processing {half_path.name}...")
                self.root.update_idletasks()
                preprocess_for_manifest(half_path, profile_name=DEFAULT_PREPROCESSING_PROFILE)
                result = generate_auto_sidecar(
                    str(half_path), debug=False, doc_type_override=doc_type_override)
                if result.sidecar is None:
                    messagebox.showwarning(
                        "Column Calibration",
                        f"{half_path.name}: no sidecar could be generated - "
                        f"{'; '.join(result.warnings)[:200]}\n\nThis half was still saved as "
                        f"an image but has no sidecar yet; it won't appear usable in this "
                        f"tool until one exists.")
                    continue
                sidecar_path = self.sidecar_dir / f"{half_path.stem}_sidecar.json"
                save_sidecar(result.sidecar, sidecar_path)
                new_pages.append(half_path)
        except Exception as e:
            messagebox.showerror(
                "Split failed",
                f"Splitting/reprocessing failed: {type(e).__name__}: {e}\n\n"
                f"The original page (index {self.page_index}) is unchanged in the "
                f"navigation list.")
            self.split_status_var.set("")
            return

        if not new_pages:
            self.split_status_var.set("")
            messagebox.showerror(
                "Column Calibration",
                "Neither half produced a usable sidecar - navigation list unchanged, "
                "original working file left in place.")
            return

        # DELETE the original whole-page working copy (2026-08-08, per
        # Jon: "the old image persists too" - now that at least one half
        # is confirmed usable, the original is redundant clutter, not a
        # safety net anyone needs). Only the WORKING copy - a preprocessed
        # COPY the raw source folder never sees, per core/calibration_
        # workspace.py's own "raw source image != production input"
        # design - the actual raw source file is never touched by this
        # or any other action in this tool. Best-effort: a delete failure
        # (e.g. file locked/already gone) is logged to the status line,
        # not raised - the split itself already fully succeeded by this
        # point and shouldn't be reported as failed over cleanup alone.
        for stale_path in (
            working_path,
            self.sidecar_dir / f"{working_path.stem}_sidecar.json",
            working_path.parent / f"{working_path.stem}_preprocess.json",
        ):
            try:
                stale_path.unlink(missing_ok=True)
            except OSError as e:
                print(f"  (could not remove stale {stale_path.name}: {e})")

        self.image_paths[self.page_index:self.page_index + 1] = new_pages
        self.split_status_var.set(
            f"Split into {len(new_pages)} page(s): {', '.join(p.name for p in new_pages)} - "
            f"original whole-page working copy removed.")
        self._load_page()

    def export_merged_sidecar(self) -> None:
        if self.correction_record is None or self.sidecar is None:
            messagebox.showinfo("Column Calibration", "Nothing to merge for this page.")
            return
        merged = merge_column_corrections(self.sidecar, self.correction_record)
        # Apply any accumulated preview rotation delta (manual +/- nudge
        # buttons, or a "Test rotation refinement" run - same variable,
        # see _nudge_deskew()'s docstring) - this is the ONE place it
        # actually persists, same "Export is the only real output" rule
        # every column correction in this tool already follows. Safe to
        # simply add rather than needing any incremental-rotation
        # reasoning: core.row_segmentation.crop_region_from_source()
        # always re-applies deskew_angle from scratch against the
        # UNTOUCHED original image (expand=True) - it never composes
        # with whatever this preview's simplified expand=False on-screen
        # rendering showed, so the stored angle NUMBER is all that
        # matters downstream, not how the preview got there.
        if self._preview_rotation_delta != 0.0:
            merged["deskew_angle"] = merged.get("deskew_angle", 0.0) + self._preview_rotation_delta
        default_name = f"{self.image_paths[self.page_index].stem}_merged_sidecar.json"
        out_path = filedialog.asksaveasfilename(
            title="Save merged sidecar as (never overwrites the original)",
            initialfile=default_name, defaultextension=".json", filetypes=[("JSON", "*.json")],
            initialdir=_last_dialog_dir("export_merged_sidecar"))
        if not out_path:
            return
        _remember_dialog_dir("export_merged_sidecar", out_path)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)
        messagebox.showinfo("Column Calibration", f"Merged sidecar written to:\n{out_path}")

    # -------------------------------------------------------- navigation
    def _save_if_dirty(self) -> None:
        """
        Auto-saves before navigating away (2026-08-09, per Jon: "moving
        between images and going back to a previous one the deskew isn't
        persisting"). Previously next_page()/prev_page() only asked
        "leave without saving?" and DISCARDED everything in memory on
        confirmation - including a manual deskew nudge, since that only
        ever lived in self._preview_rotation_delta/self.correction_record
        until an explicit "Save / Confirm Page" click wrote it to disk.
        The discard dialog's own wording ("unsaved column corrections")
        didn't even mention deskew, making it easy to click through
        without realizing it would be lost. save_page() is safe to call
        unconditionally here - it computes review_status from actual
        per-divider completeness (never forces "done" on a partial
        page), so auto-saving on navigation can't silently mark
        incomplete work as finished.
        """
        if self._dirty:
            self.save_page()

    def next_page(self) -> None:
        self._save_if_dirty()
        if self.page_index < len(self.image_paths) - 1:
            self.page_index += 1
            self._load_page()
        else:
            messagebox.showinfo("Column Calibration", "This is the last page.")

    def prev_page(self) -> None:
        self._save_if_dirty()
        if self.page_index > 0:
            self.page_index -= 1
            self._load_page()
        else:
            messagebox.showinfo("Column Calibration", "This is the first page.")


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _collect_images(args) -> tuple[list[Path], Path | None, Path | None]:
    """
    Returns (image_paths, sidecar_dir_override, columns_file_override).
    Both overrides are None for the two explicit-args paths below
    (main() falls back to --sidecar-dir/--columns-file/DEFAULT_SIDECAR_
    DIR, unchanged from before) - only the no-args folder-picker path
    below routes through an isolated evaluation workspace and can return
    real overrides for both (columns_file_override only when the raw
    folder had a columns_<doc_type>.txt - see
    core.calibration_workspace's own docstring).
    """
    if args.images:
        return [Path(p) for p in args.images], None, None
    if args.image_dir:
        d = Path(args.image_dir)
        return sorted(p for p in d.iterdir() if p.suffix.lower() in _IMAGE_EXTS), None, None

    # No CLI args (added 2026-08-07, per Jon: launching with no arguments
    # should never just dead-end at a terminal error - "UI is supposed to
    # reduce friction, not add it"). Falls back to a folder picker, same
    # filedialog module this file already uses for "Export merged
    # sidecar..." - a real Tk root is needed to host the dialog even
    # though the app's own root isn't built yet at this point in main(),
    # so a throwaway one is created and destroyed immediately after.
    #
    # The picked folder is treated as RAW SOURCE, not production input
    # (2026-08-07, per Jon: "raw source image != production input... the
    # folder picker should really mean 'Import this folder into a
    # temporary evaluation/calibration workspace'") - routed through
    # core.calibration_workspace.prepare_evaluation_workspace() rather
    # than opened directly, so this never implies the picked folder is
    # already part of the production pipeline (data/working/,
    # data/manifest.csv, pipeline_db are never touched by this path).
    picker_root = Tk()
    picker_root.withdraw()
    chosen = filedialog.askdirectory(
        title="Column Calibration - choose a RAW image folder to import for calibration",
        initialdir=_last_dialog_dir("import_raw_folder"))
    picker_root.destroy()
    if not chosen:
        raise SystemExit("No folder chosen - nothing to review.")
    _remember_dialog_dir("import_raw_folder", chosen)

    from core.calibration_workspace import prepare_evaluation_workspace
    image_paths, sidecar_dir, columns_override = prepare_evaluation_workspace(
        Path(chosen), regenerate=args.regenerate)
    return image_paths, sidecar_dir, columns_override


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-dir", type=str, default=None,
                         help="Folder of images to review, in directory order.")
    parser.add_argument("--images", nargs="+", default=None,
                         help="Explicit list of image paths, in review order.")
    parser.add_argument("--sidecar-dir", type=str, default=str(DEFAULT_SIDECAR_DIR),
                         help=f"Directory of auto-generated sidecars to look up per image "
                              f"(default: {DEFAULT_SIDECAR_DIR}). Ignored when no --image-dir/"
                              f"--images is given - the no-args folder-picker path always uses "
                              f"its own evaluation workspace's sidecar dir instead.")
    parser.add_argument("--columns-file", type=str, default=None,
                         help="Force a specific config/columns/*.txt for every page, "
                              "overriding each page's own template.columns_file.")
    parser.add_argument("--regenerate", action="store_true",
                         help="No-args folder-picker path only: re-run deskew/preprocess/auto_sidecar "
                              "for every image even if this folder was already imported before, "
                              "instead of reusing the existing evaluation-workspace artifacts.")
    args = parser.parse_args()

    image_paths, sidecar_dir_override, columns_file_override = _collect_images(args)
    if not image_paths:
        print("No images found.")
        return
    sidecar_dir = sidecar_dir_override if sidecar_dir_override is not None else Path(args.sidecar_dir)
    columns_file = str(columns_file_override) if columns_file_override is not None else args.columns_file

    root = Tk()
    ColumnCalibrationApp(root, image_paths, sidecar_dir, columns_file)
    root.mainloop()


if __name__ == "__main__":
    main()
