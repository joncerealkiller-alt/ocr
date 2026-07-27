"""
Dewarp Preprocessor UI - standalone 4-point perspective-correction tool,
built 2026-07-25 after real evidence that some LoRA training crops
exported from warped source pages were illegible even to a human
reviewer - a distorted source page is a ceiling no downstream model,
prompt, or preprocessing-filter work can fix (see ui/row_segmentation_
ui.py's own Preprocessing section for the filters that DO help once
the page geometry itself is sound).

Usage:
    python ui/dewarp_preprocessor_ui.py

Workflow: Select image -> drag the 4 corner handles onto the page's
real (possibly skewed/curled) outer edges -> "Preview Flattened Image"
to inspect the result -> "Apply & Save Dewarped Copy" (writes a NEW
file, e.g. <name>_dewarped.jpg under data/outputs/dewarped/ - the
original source file is never modified) or "Bypass / Use Original" if
the page isn't actually distorted.

NOT auto-chained to any other tool - there is no automated "download
from a bucket, then dewarp, then segment" pipeline anywhere in this
project (confirmed 2026-07-25: every "bucket" in this codebase means
core/classifier.py's local document-type routing CSVs under
data/buckets/, not cloud storage; the real current workflow is three
separate, manually-run steps - scripts/build_manifest.py, core.
classifier, then a human opens ui/row_segmentation_ui.py and picks a
file). This tool matches that same manual-handoff pattern for its
single-file mode: it shows the saved (or bypassed) path clearly and
offers a convenience button to launch ui/row_segmentation_ui.py, but
the human still picks the file there themselves via its own "Select
image..." dialog, exactly like every other step already works.

ARCHITECTURE (Jon's direction, 2026-07-25): every module in this
project is currently standalone by design - isolates development/
testing/debugging while the pipeline is still evolving. End-to-end
automation is planned for later, once individual modules stabilize,
not now. This tool stays fully standalone and human-in-the-loop (one
image at a time, human confirms/adjusts before Save), but exposes
clean extension points for that later orchestration:
  - dewarp_and_save()/bypass_passthrough() below are plain, GUI-free
    functions - the "Apply & Save"/"Bypass" buttons just call them.
    A future orchestrator can call these directly without driving
    the Tkinter UI at all.
  - "Load bucket CSV..." mode reads an EXISTING core/classifier.py
    bucket CSV (data/buckets/<category>.csv) as an ordered worklist
    of file_paths (see core/bucket_worklist.py) and, on every Save/
    Bypass, appends one record to a sibling "preprocessed bucket"
    (data/buckets/<category>_dewarped.csv: source_file_path,
    output_file_path, status, timestamp) before auto-advancing to the
    next undone entry - resume-friendly (already-recorded entries are
    skipped on reopen), same per-item-progress spirit as ui/row_
    segmentation_ui.py's per-column advance. A later automated stage
    can pick up straight from that output CSV instead of the original
    classification bucket. This is still NOT unattended batch
    processing - a human still drives every single Save/Bypass click.
  - See the TODO(orchestrator) comment on _confirm_current() below for
    the one seam that's a real placeholder for later automation, not
    yet built.

Default corner placement is a plain inset margin (5% of image
dimensions) - NOT real page-edge detection. Detecting real page/table
edges reliably on a still-warped source is a much harder problem than
this tool takes on; the 4 handles are meant to be dragged into place
by a human looking at the actual scan, same "auto-estimate is a
suggestion, confirm visually" philosophy as ui/row_segmentation_ui.py's
own deskew angle.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tkinter import (
    Tk, Frame, Label, Button, StringVar, BooleanVar, OptionMenu,
    filedialog, messagebox, Canvas, Scrollbar, VERTICAL, HORIZONTAL, Entry,
)
from PIL import Image, ImageTk, ImageDraw

from core.dewarp import dewarp_quad, save_dewarp_sidecar, regenerate_from_sidecar, dewarp_sidecar_path
from core.bucket_worklist import (
    load_bucket_filepaths, preprocessed_bucket_path, load_processed_sources,
    load_processed_records, append_preprocessed_record,
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "dewarped"
PREVIEW_SIZE = (900, 700)
PYTHON = sys.executable
WORKLIST_STAGE_NAME = "dewarped"

# Corner handle hit-test radius, in CANVAS pixels (not source-image
# pixels) - independent of zoom level, matching how a mouse actually
# interacts with the rendered preview regardless of how zoomed in/out
# the source image currently is.
HANDLE_HIT_RADIUS = 12
HANDLE_LABELS = ["TL", "TR", "BR", "BL"]
HANDLE_COLOR = "#39ff14"
QUAD_LINE_COLOR = "#ff3030"


def dewarp_and_save(
    source_path: str, corners: list[tuple[float, float]], out_dir: Path = OUTPUT_DIR,
) -> Path:
    """
    Standalone entry point, deliberately GUI-free (2026-07-25, per
    Jon's direction: keep this tool fully standalone/manual for now,
    but expose a clean callable interface a future central-workflow
    orchestrator could call directly, without needing to drive the
    Tkinter UI). Loads source_path fresh from disk rather than
    assuming a caller already has it open in memory, so this is a real
    standalone entry point - not just an internal refactor of a button
    handler that secretly depends on GUI state.

    Also writes a corner sidecar next to the output file (2026-07-26,
    real incident: an output file got deleted by an unrelated cleanup
    command, and there was no way to regenerate it short of re-dragging
    all 4 corners from scratch - see core/dewarp.py's module docstring
    and save_dewarp_sidecar()). The sidecar is what makes that
    recoverable going forward without any human effort.
    """
    image = Image.open(source_path).convert("RGB")
    flattened = dewarp_quad(image, corners)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(source_path)
    suffix = src.suffix if src.suffix else ".jpg"
    out_path = out_dir / f"{src.stem}_dewarped{suffix}"
    flattened.save(out_path)
    save_dewarp_sidecar(source_path, out_path, corners)
    return out_path


def bypass_passthrough(source_path: str) -> str:
    """
    Trivial, but a real documented entry point rather than inline UI
    logic buried in a button handler - "bypass" means forward the
    ORIGINAL path unchanged. Having this as a real function is what
    makes "bypass" a checkable contract (see the worklist-mode tests)
    instead of an implicit assumption.
    """
    return source_path


class DewarpApp:
    def __init__(self, root: Tk):
        self.root = root
        root.title("Dewarp Preprocessor - 4-point perspective correction (no model calls)")
        root.geometry("1300x900")
        root.minsize(1000, 700)

        self.image_path: str | None = None
        self.original_image: Image.Image | None = None
        self.corners: list[tuple[float, float]] | None = None  # [TL, TR, BR, BL], source coords
        self._dragging_idx: int | None = None
        self._preview_scale = 1.0
        self._tk_preview = None
        self.output_path: str | None = None

        # Bucket-worklist mode (2026-07-25, per Jon's direction) - None/
        # empty means plain single-file mode (unchanged from before this
        # feature existed). When active, bucket_csv_path/preprocessed_
        # csv_path/worklist/worklist_index drive which image is loaded
        # and where Save/Bypass results get recorded - see
        # _load_bucket_csv()/_advance_worklist().
        self.bucket_csv_path: str | None = None
        self.preprocessed_csv_path: str | None = None
        self.worklist: list[str] = []
        self.worklist_index: int | None = None

        self.show_flattened_var = BooleanVar(value=False)
        self.zoom_var = StringVar(value="Fit")
        self.output_path_var = StringVar(value="")
        self.worklist_progress_var = StringVar(value="")

        top = Frame(root, padx=10, pady=10)
        top.pack(fill="both", expand=True)

        # -- image selection ----------------------------------------------
        img_row = Frame(top)
        img_row.pack(fill="x", pady=4)
        Button(img_row, text="Select image...", command=self.select_image).pack(side="left")
        Button(img_row, text="Load bucket CSV...", command=self._load_bucket_csv
               ).pack(side="left", padx=(6, 0))
        self.image_path_label = Label(img_row, text="(no image selected)", fg="#666")
        self.image_path_label.pack(side="left", padx=8)
        Label(img_row, textvariable=self.worklist_progress_var, fg="#06c",
              font=("Segoe UI", 9, "bold")).pack(side="left", padx=8)

        # -- view controls --------------------------------------------------
        view_row = Frame(top, relief="groove", borderwidth=1, padx=8, pady=4)
        view_row.pack(fill="x", pady=(4, 3))
        Label(view_row, text="Drag the 4 corner handles onto the page's real edges - "
                              "default position is just a plain inset margin, not detection:",
              font=("Segoe UI", 9, "bold")).pack(anchor="w")
        controls = Frame(view_row)
        controls.pack(fill="x", pady=(4, 0))
        Label(controls, text="Zoom:").pack(side="left")
        OptionMenu(controls, self.zoom_var, "Fit", "1x", "2x", "4x",
                   command=lambda _v: self._render()).pack(side="left", padx=(4, 16))
        self.flatten_toggle_button = Button(
            controls, text="Preview Flattened Image", command=self._toggle_flattened_view)
        self.flatten_toggle_button.pack(side="left", padx=(0, 16))
        Button(controls, text="Reset corners to margin", command=self._reset_corners
               ).pack(side="left")

        # -- canvas -----------------------------------------------------------
        canvas_frame = Frame(top)
        canvas_frame.pack(fill="both", expand=True, pady=(4, 4))
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)

        self.preview_canvas = Canvas(canvas_frame, bg="#ddd")
        v_scroll = Scrollbar(canvas_frame, orient=VERTICAL, command=self.preview_canvas.yview)
        h_scroll = Scrollbar(canvas_frame, orient=HORIZONTAL, command=self.preview_canvas.xview)
        self.preview_canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        self.preview_canvas.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")

        self.preview_canvas.bind("<ButtonPress-1>", self._on_handle_press)
        self.preview_canvas.bind("<B1-Motion>", self._on_handle_drag)
        self.preview_canvas.bind("<ButtonRelease-1>", self._on_handle_release)

        # -- actions ------------------------------------------------------------
        action_row = Frame(top)
        action_row.pack(fill="x", pady=(0, 4))
        Button(action_row, text="Apply & Save Dewarped Copy (Enter)",
               command=self.apply_and_save, bg="#3a8f3a", fg="white",
               font=("Segoe UI", 9, "bold")).pack(side="left")
        Button(action_row, text="Bypass / Use Original (Esc or Space)",
               command=self.bypass).pack(side="left", padx=(8, 0))
        self.open_row_seg_button = Button(
            action_row, text="Open in Row Segmentation UI", state="disabled",
            command=self._launch_row_segmentation_ui)
        self.open_row_seg_button.pack(side="left", padx=(8, 0))

        output_row = Frame(top)
        output_row.pack(fill="x")
        Label(output_row, text="Output path:").pack(side="left")
        Entry(output_row, textvariable=self.output_path_var, state="readonly",
              width=90).pack(side="left", padx=(6, 0), fill="x", expand=True)

        self.status_label = Label(top, text="Select an image to begin.", fg="#444", anchor="w")
        self.status_label.pack(fill="x", pady=(4, 0))

        # Keyboard shortcuts (2026-07-25 - the first shortcuts anywhere in
        # this project's Tkinter tools; ui/row_segmentation_ui.py has none
        # to match conventions from). Bound at root level for this simple,
        # single-purpose utility rather than per-widget, since there's no
        # free-text field here whose typing these keys would collide with.
        root.bind("<Return>", lambda e: self.apply_and_save())
        root.bind("<Escape>", lambda e: self.bypass())
        root.bind("<space>", lambda e: self.bypass())

    # -- image loading --------------------------------------------------------

    def select_image(self):
        path = filedialog.askopenfilename(
            title="Select raw census page scan",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp")],
        )
        if not path:
            return
        # Manual single-file selection always exits worklist mode - the
        # two input modes are mutually exclusive (a manually-picked file
        # isn't tracked in any bucket CSV, so there's nothing to advance
        # to or record a result against).
        self.bucket_csv_path = None
        self.preprocessed_csv_path = None
        self.worklist = []
        self.worklist_index = None
        self.worklist_progress_var.set("")
        self._load_image_path(path)
        self.status_label.config(
            text="Image loaded. Drag the 4 green handles onto the page's real edges.")

    def _load_image_path(self, path: str) -> None:
        """Shared image-loading logic for both single-file selection and
        worklist mode - resets everything that's specific to the
        PREVIOUS image (corners, output path, flattened-view toggle)."""
        self.image_path = path
        self.original_image = Image.open(path).convert("RGB")
        self.image_path_label.config(text=path, fg="black")
        self.output_path = None
        self.output_path_var.set("")
        self.open_row_seg_button.config(state="disabled")
        self.show_flattened_var.set(False)
        self.flatten_toggle_button.config(text="Preview Flattened Image")
        self._reset_corners()

    def _load_bucket_csv(self):
        """
        Worklist mode entry point (2026-07-25, per Jon's direction):
        reads an existing core/classifier.py bucket CSV as an ordered
        list of file_paths, skips anything already recorded in the
        sibling preprocessed-bucket CSV (resume support), and loads the
        first undone entry. See core/bucket_worklist.py's module
        docstring for the full input/output CSV contract.
        """
        path = filedialog.askopenfilename(
            title="Select a classification bucket CSV (data/buckets/<category>.csv)",
            filetypes=[("CSV files", "*.csv")],
            initialdir=str(PROJECT_ROOT / "data" / "buckets"),
        )
        if not path:
            return
        try:
            all_paths = load_bucket_filepaths(path)
        except Exception as e:
            messagebox.showerror("Could not read bucket CSV", str(e))
            return
        if not all_paths:
            messagebox.showwarning("Empty bucket", "That bucket CSV has no file_path rows.")
            return

        self.bucket_csv_path = path
        self.preprocessed_csv_path = str(preprocessed_bucket_path(path, WORKLIST_STAGE_NAME))
        done = self._verify_and_self_heal(self.preprocessed_csv_path)
        self.worklist = [p for p in all_paths if p not in done]

        if not self.worklist:
            messagebox.showinfo(
                "Nothing to do",
                f"Every file in {Path(path).name} already has a recorded "
                f"{WORKLIST_STAGE_NAME}/bypassed entry in "
                f"{Path(self.preprocessed_csv_path).name}.")
            self.bucket_csv_path = None
            self.preprocessed_csv_path = None
            return

        self.worklist_index = 0
        self._load_worklist_current()

    def _verify_and_self_heal(self, preprocessed_csv_path: str) -> set[str]:
        """
        Real incident this fixes (2026-07-26): a recorded "dewarped"
        entry's output file was deleted by an unrelated cleanup
        command run against the wrong directory. Before this existed,
        reopening the bucket worklist would have silently treated that
        entry as done anyway (load_processed_sources() only checks the
        CSV, never the actual file), permanently hiding the fact that
        real work was lost.

        Now: every recorded "dewarped" entry's output_file_path is
        checked against disk. If missing but its corner sidecar (and
        the original source) still exist, the output is regenerated
        automatically - the whole point of persisting corners as
        durable metadata instead of throwaway state. If it truly can't
        be healed (no sidecar, or the source is ALSO gone), the entry
        is dropped from the returned "done" set so it gets correctly
        re-offered to a human instead of silently skipped. "bypassed"
        entries are checked against the source file only (there's
        nothing to regenerate - the output IS the source).
        """
        done = set()
        healed, lost = [], []
        for row in load_processed_records(preprocessed_csv_path):
            source = row.get("source_file_path")
            output = row.get("output_file_path")
            status = row.get("status")
            if not source or not output:
                continue
            if Path(output).exists():
                done.add(source)
                continue
            if status == WORKLIST_STAGE_NAME:
                sidecar = dewarp_sidecar_path(output)
                try:
                    regenerate_from_sidecar(sidecar)
                    healed.append(source)
                    done.add(source)
                    continue
                except FileNotFoundError:
                    pass
            lost.append(source)

        if healed:
            print(f"Regenerated {len(healed)} previously-missing dewarped file(s) "
                  f"from saved corner coordinates:")
            for s in healed:
                print(f"  - {s}")
        if lost:
            messagebox.showwarning(
                "Some recorded results are missing",
                f"{len(lost)} file(s) were recorded as done in "
                f"{Path(preprocessed_csv_path).name} but their output is missing "
                f"and could not be regenerated (no corner sidecar, or the original "
                f"source is also gone). They'll be re-offered in this worklist:\n\n"
                + "\n".join(lost[:10]) + ("\n..." if len(lost) > 10 else ""))
        return done

    def _load_worklist_current(self):
        """Loads self.worklist[self.worklist_index] and updates the
        progress label - shared by _load_bucket_csv() (first entry) and
        _advance_worklist() (every entry after)."""
        path = self.worklist[self.worklist_index]
        total = len(self.worklist)
        try:
            self._load_image_path(path)
        except Exception as e:
            messagebox.showerror("Could not load image", f"{path}\n\n{e}")
            return
        self.worklist_progress_var.set(
            f"Bucket worklist: {self.worklist_index + 1}/{total} "
            f"({Path(self.bucket_csv_path).name})")
        self.status_label.config(
            text=f"Image {self.worklist_index + 1}/{total} loaded from bucket worklist. "
                 f"Drag the 4 green handles onto the page's real edges.")

    def _advance_worklist(self):
        """Moves to the next entry after a Save/Bypass in worklist mode,
        or reports completion once every entry has been recorded."""
        self.worklist_index += 1
        if self.worklist_index >= len(self.worklist):
            self.worklist_progress_var.set(
                f"Bucket worklist complete ({len(self.worklist)}/{len(self.worklist)}).")
            self.status_label.config(
                text=f"All images in this bucket worklist are done - results recorded in "
                     f"{self.preprocessed_csv_path}")
            self.bucket_csv_path = None
            self.preprocessed_csv_path = None
            self.worklist = []
            self.worklist_index = None
            return
        self._load_worklist_current()

    def _reset_corners(self):
        """
        Starting quad: a plain 5% inset margin from each edge - NOT
        real page-edge detection (see module docstring). Just a
        reasonable starting rectangle to drag inward/outward from,
        same "auto-estimate is a suggestion" spirit as ui/row_
        segmentation_ui.py's deskew angle, minus any actual estimation
        since reliable edge detection on a still-warped source is a
        substantially harder problem this tool doesn't take on.
        """
        if not self.original_image:
            return
        w, h = self.original_image.size
        mx, my = round(w * 0.05), round(h * 0.05)
        self.corners = [
            (mx, my), (w - mx, my), (w - mx, h - my), (mx, h - my),
        ]
        self._render()

    # -- rendering --------------------------------------------------------------

    def _render(self):
        if not self.original_image:
            return
        if self.show_flattened_var.get():
            try:
                base = dewarp_quad(self.original_image, self.corners)
            except ValueError as e:
                messagebox.showerror("Cannot flatten", str(e))
                self.show_flattened_var.set(False)
                base = self.original_image.copy()
        else:
            base = self.original_image.copy()
            draw = ImageDraw.Draw(base)
            quad = self.corners + [self.corners[0]]
            draw.line(quad, fill=QUAD_LINE_COLOR, width=max(2, round(base.width / 400)))
            r = max(6, round(base.width / 150))
            for (x, y), label in zip(self.corners, HANDLE_LABELS):
                draw.ellipse([x - r, y - r, x + r, y + r], fill=HANDLE_COLOR, outline="black")
                draw.text((x + r + 2, y - r), label, fill=HANDLE_COLOR)

        self._render_preview(base)

    def _render_preview(self, image: Image.Image):
        """
        Same zoom/scroll-preservation pattern as ui/row_segmentation_
        ui.py's _render_preview() - whole image scaled up (not a
        cropped strip), scroll position saved before re-render and
        restored after, so adjusting a handle or toggling the
        flattened view doesn't yank the viewport back to the origin.
        """
        zoom = self.zoom_var.get()
        if zoom == "Fit":
            preview = image.copy()
            preview.thumbnail(PREVIEW_SIZE)
        else:
            factor = int(zoom.rstrip("x"))
            preview = image.resize((image.width * factor, image.height * factor), Image.NEAREST)

        self._preview_scale = preview.width / image.width if image.width else 1.0

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

    def _toggle_flattened_view(self):
        if not self.original_image:
            return
        self.show_flattened_var.set(not self.show_flattened_var.get())
        self.flatten_toggle_button.config(
            text="Show Original / Edit Corners" if self.show_flattened_var.get()
            else "Preview Flattened Image")
        self._render()

    # -- corner dragging ----------------------------------------------------------

    def _canvas_to_source(self, event) -> tuple[float, float]:
        cx = self.preview_canvas.canvasx(event.x)
        cy = self.preview_canvas.canvasy(event.y)
        return cx / self._preview_scale, cy / self._preview_scale

    def _on_handle_press(self, event):
        if not self.corners or self.show_flattened_var.get():
            return
        cx = self.preview_canvas.canvasx(event.x)
        cy = self.preview_canvas.canvasy(event.y)
        hit_radius_src = HANDLE_HIT_RADIUS / self._preview_scale
        for i, (x, y) in enumerate(self.corners):
            hx, hy = x * self._preview_scale, y * self._preview_scale
            if (hx - cx) ** 2 + (hy - cy) ** 2 <= HANDLE_HIT_RADIUS ** 2 * 4:
                # generous hit box (2x visual handle radius) - small
                # handles on a large, zoomed-out page are hard to hit
                # exactly otherwise.
                self._dragging_idx = i
                return
        self._dragging_idx = None

    def _on_handle_drag(self, event):
        if self._dragging_idx is None or not self.original_image:
            return
        x, y = self._canvas_to_source(event)
        w, h = self.original_image.size
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        self.corners[self._dragging_idx] = (x, y)
        self._render()

    def _on_handle_release(self, event):
        self._dragging_idx = None

    # -- actions --------------------------------------------------------------------

    # TODO(orchestrator): apply_and_save()/bypass() are the one seam in
    # this tool that's a real placeholder for later automation, not yet
    # built (Jon's direction, 2026-07-25: individual modules stay fully
    # standalone/human-in-the-loop until the pipeline stabilizes, then
    # a central workflow UI gets built on top without needing to
    # refactor each stage). Today, a human always decides which button
    # to click for every single image. A future orchestrator wanting to
    # run this stage unattended would need its own policy for THAT
    # decision (e.g. an automatic warp-severity heuristic deciding
    # dewarp-vs-bypass) - it should call dewarp_and_save()/
    # bypass_passthrough() directly (both are plain, GUI-free
    # functions already) rather than trying to drive these Tkinter
    # button handlers.

    def apply_and_save(self):
        if not self.original_image or not self.image_path:
            messagebox.showwarning("No image", "Select an image first.")
            return
        try:
            out_path = dewarp_and_save(self.image_path, self.corners)
        except ValueError as e:
            messagebox.showerror("Dewarp failed", str(e))
            return

        self.output_path = str(out_path)
        self.output_path_var.set(self.output_path)
        self.open_row_seg_button.config(state="normal")
        self.status_label.config(
            text=f"Saved dewarped copy: {out_path} - original source file untouched.")
        self._record_and_advance_if_worklist(status=WORKLIST_STAGE_NAME)

    def bypass(self):
        if not self.image_path:
            messagebox.showwarning("No image", "Select an image first.")
            return
        self.output_path = bypass_passthrough(self.image_path)
        self.output_path_var.set(self.output_path)
        self.open_row_seg_button.config(state="normal")
        self.status_label.config(text="Bypassed - forwarding the original, unmodified image.")
        self._record_and_advance_if_worklist(status="bypassed")

    def _record_and_advance_if_worklist(self, status: str) -> None:
        """
        No-op in single-file mode (bucket_csv_path is None). In
        worklist mode, appends one result row to the preprocessed
        bucket CSV (source path -> self.output_path, which Save vs.
        Bypass already set correctly above) and auto-advances to the
        next undone entry - see core/bucket_worklist.py and
        _advance_worklist().
        """
        if not self.bucket_csv_path:
            return
        append_preprocessed_record(
            self.preprocessed_csv_path, source_file_path=self.image_path,
            output_file_path=self.output_path, status=status,
        )
        self._advance_worklist()

    def _launch_row_segmentation_ui(self):
        """
        Convenience only - matches debug_tools/workflow_gui.py's own
        "_launch()" pattern (subprocess.Popen a real entry point, don't
        reimplement it). Does NOT auto-load self.output_path into the
        launched tool - ui/row_segmentation_ui.py has no CLI argument
        for a startup image path, and there is no automated hand-off
        chain anywhere else in this project to model this on (see
        module docstring) - the path is shown/selectable in the Output
        path field above for the human to pick via that tool's own
        "Select image..." dialog, same manual pattern every other step
        in this pipeline already uses.
        """
        try:
            subprocess.Popen([PYTHON, str(PROJECT_ROOT / "ui" / "row_segmentation_ui.py")])
        except Exception as e:
            messagebox.showerror("Could not launch", str(e))


if __name__ == "__main__":
    root = Tk()
    app = DewarpApp(root)
    root.mainloop()
