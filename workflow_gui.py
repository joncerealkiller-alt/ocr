"""
Workflow GUI — one tab per CLI workflow step, built 2026-07-15 per
Jon's request: a way to run the pipeline's CLI tools without needing
to keep the README open for exact flag names.

ARCHITECTURE: this GUI calls the REAL CLI entry points as subprocesses
(python test_row_segmentation.py ..., python run_row_extraction.py ...,
etc.) - it does NOT reimplement any of their processing logic. Every
option shown here maps to an actual argparse flag in the real script,
cross-checked directly against each script's add_argument() calls
before this GUI was built, not guessed or half-remembered. If a CLI
script's options change, this GUI's comboboxes may go stale, but the
actual command that runs is always exactly what the real script defines
- there's no separate pipeline implementation here to drift out of sync
with the one that matters.

Standalone GUI tools already built this session (model_assessment.py,
review_uncertain.py, row_segmentation_ui.py, build_manifest.py) are NOT
re-wrapped as option-driven tabs - they're full self-contained apps, so
this GUI just launches them as separate processes via simple buttons,
same "call the real entry point" principle applied to a different shape
of tool.
"""

from __future__ import annotations

import json
import platform
import queue
import subprocess
import sys
import threading
from pathlib import Path
from tkinter import (
    Tk, Frame, Label, Button, Entry, StringVar, IntVar, BooleanVar,
    OptionMenu, Text, Checkbutton, filedialog, messagebox, END, NORMAL,
    DISABLED, ttk,
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "config" / "models"
PROMPTS_DIR = PROJECT_ROOT / "config" / "prompts"
COLUMNS_DIR = PROJECT_ROOT / "config" / "columns"
SIDECAR_DIR = PROJECT_ROOT / "data" / "outputs" / "row_segmentation"
DATA_DIR = PROJECT_ROOT / "data"
STATE_PATH = PROJECT_ROOT / "data" / "outputs" / "_workflow_gui_state.json"

PYTHON = sys.executable  # same interpreter this GUI is running under,
                          # not a bare "python" that might resolve to a
                          # different environment


# -- shared helpers -----------------------------------------------------

def _scan(directory: Path, pattern: str) -> list[str]:
    """Same convention already established in model_assessment.py
    (sorted glob) - reused here rather than inventing a different
    scanning style for this GUI specifically."""
    if not directory.exists():
        return []
    return sorted(p.name for p in directory.glob(pattern))


def _scan_stems(directory: Path, pattern: str) -> list[str]:
    if not directory.exists():
        return []
    return sorted(p.stem for p in directory.glob(pattern))


def _relative_or_absolute(directory: Path, value: str) -> str:
    """
    Builds a path for the command - RELATIVE to PROJECT_ROOT when the
    value is just a filename from one of our own project folders (the
    subprocess already runs with cwd=PROJECT_ROOT, so this is exactly
    equivalent to an absolute path, just far shorter and matching the
    clean "python script.py data/outputs/.../foo.json" style every CLI
    script's own --help/usage text and the README use). Confirmed
    necessary (2026-07-14, real test): the original absolute-path
    version produced commands cluttered with the full install path
    repeated for every argument, not matching expected CLI usage at all.

    If the value is ALREADY an absolute path (e.g. from a Browse...
    file dialog, which could point anywhere on disk, not just inside
    this project), it's kept as-is - relativizing something outside
    PROJECT_ROOT would either fail outright or produce a nonsensical
    "../../.." path.
    """
    if not value:
        return ""
    p = Path(value)
    if p.is_absolute():
        return value
    full = directory / value
    try:
        return str(full.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(full)


def _load_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _open_folder(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    system = platform.system()
    try:
        if system == "Windows":
            import os
            os.startfile(str(path))  # noqa: S606
        elif system == "Darwin":
            subprocess.run(["open", str(path)], check=True)
        else:
            subprocess.run(["xdg-open", str(path)], check=True)
    except Exception as e:
        messagebox.showerror("Could not open folder", str(e))


class CommandTab:
    """
    Shared scaffolding for every rich, option-driven tab (segmentation,
    row extraction, two-stage extraction, native-prompt test). Each
    subclass builds its own option widgets and command-building logic;
    this base class handles the parts that are identical across all of
    them: command preview, Run button + threaded subprocess streaming,
    log panel, open-folder buttons, and state persistence.
    """

    def __init__(self, parent, app: "WorkflowApp", tab_key: str,
                 title: str, description: str):
        self.app = app
        self.tab_key = tab_key
        self.frame = Frame(parent, padx=10, pady=10)
        self.log_queue: queue.Queue = queue.Queue()
        self.process: subprocess.Popen | None = None

        Label(self.frame, text=title, font=("Segoe UI", 11, "bold")).pack(anchor="w")
        Label(self.frame, text=description, font=("Segoe UI", 9), fg="#444",
              wraplength=1000, justify="left").pack(anchor="w", pady=(2, 10))

        self.options_frame = Frame(self.frame)
        self.options_frame.pack(fill="x")

        action_row = Frame(self.frame)
        action_row.pack(fill="x", pady=(10, 4))
        self.run_button = Button(action_row, text="Run", command=self._on_run,
                                  bg="#4a7", fg="white", font=("Segoe UI", 10, "bold"))
        self.run_button.pack(side="left")
        Button(action_row, text="Open input folder",
               command=self._open_input_folder).pack(side="left", padx=(8, 0))
        Button(action_row, text="Open output folder",
               command=self._open_output_folder).pack(side="left", padx=(8, 0))
        self.status_label = Label(action_row, text="", fg="#444")
        self.status_label.pack(side="left", padx=12)

        Label(self.frame, text="Command preview:", font=("Segoe UI", 9, "bold")).pack(
            anchor="w", pady=(8, 0))
        self.preview_text = Text(self.frame, height=2, font=("Consolas", 9), wrap="word",
                                  bg="#f0f0f0")
        self.preview_text.pack(fill="x")
        self.preview_text.config(state=DISABLED)

        Label(self.frame, text="Output log:", font=("Segoe UI", 9, "bold")).pack(
            anchor="w", pady=(8, 0))
        log_frame = Frame(self.frame)
        log_frame.pack(fill="both", expand=True)
        self.log_text = Text(log_frame, font=("Consolas", 9), wrap="word", bg="#111", fg="#ddd")
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")

    # -- subclasses implement these ----------------------------------

    def build_command(self, silent: bool = False) -> list[str] | None:
        """Returns the full command as a list, or None (with a message
        box already shown, unless silent) if required fields are
        missing. MUST return exactly what will be run - this list IS
        both the preview text and the actual subprocess argv, never
        two separate things that could drift apart."""
        raise NotImplementedError

    def input_folder(self) -> Path:
        raise NotImplementedError

    def output_folder(self) -> Path:
        raise NotImplementedError

    def capture_state(self) -> dict:
        """Subclasses return their current field values for persistence."""
        return {}

    def restore_state(self, values: dict) -> None:
        """Subclasses restore field values from a saved dict."""
        pass

    # -- shared behavior ------------------------------------------------

    def update_preview(self, *_):
        cmd = self.build_command(silent=True)
        self.preview_text.config(state=NORMAL)
        self.preview_text.delete("1.0", END)
        if cmd:
            self.preview_text.insert(END, " ".join(cmd))
        else:
            self.preview_text.insert(END, "(fill in required fields)")
        self.preview_text.config(state=DISABLED)

    def _open_input_folder(self):
        _open_folder(self.input_folder())

    def _open_output_folder(self):
        _open_folder(self.output_folder())

    def _on_run(self):
        cmd = self.build_command(silent=False)
        if cmd is None:
            return
        self.app.save_all_state()
        self.run_button.config(state=DISABLED)
        self.status_label.config(text="Running...")
        self.log_text.delete("1.0", END)
        self.log_text.insert(END, f"$ {' '.join(cmd)}\n\n")

        thread = threading.Thread(target=self._run_worker, args=(cmd,), daemon=True)
        thread.start()
        self.frame.after(100, self._poll_log_queue)

    def _run_worker(self, cmd: list[str]):
        try:
            # encoding/errors explicit here (2026-07-16) - without this,
            # Popen(text=True) falls back to the OS locale's default
            # encoding (cp1252 on Windows), which cannot decode the
            # UTF-8 bytes the child process now explicitly writes to
            # its own stdout (see core/row_extraction.py's matching
            # fix). Same root cause as that crash, other side of the
            # same pipe - fixing only the child's encoding wasn't
            # enough, since the parent reading it had its own separate,
            # unfixed mismatch.
            self.process = subprocess.Popen(
                cmd, cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
                encoding="utf-8", errors="replace",
            )
            for line in self.process.stdout:
                self.log_queue.put(("line", line))
            exit_code = self.process.wait()
            self.log_queue.put(("done", exit_code))
        except Exception as e:
            self.log_queue.put(("error", str(e)))

    def _poll_log_queue(self):
        try:
            while True:
                kind, payload = self.log_queue.get_nowait()
                if kind == "line":
                    self.log_text.insert(END, payload)
                    self.log_text.see(END)
                elif kind == "done":
                    self.log_text.insert(END, f"\n[exit code {payload}]\n")
                    self.log_text.see(END)
                    self.run_button.config(state=NORMAL)
                    self.status_label.config(
                        text="Done" if payload == 0 else f"Failed (exit {payload})")
                    return
                elif kind == "error":
                    self.log_text.insert(END, f"\n[ERROR launching process: {payload}]\n")
                    self.run_button.config(state=NORMAL)
                    self.status_label.config(text="Error")
                    return
        except queue.Empty:
            pass
        self.frame.after(100, self._poll_log_queue)


class SegmentationTab(CommandTab):
    def __init__(self, parent, app):
        super().__init__(
            parent, app, "segmentation",
            "1. Row Segmentation (test_row_segmentation.py)",
            "Non-interactive CLI version of the row-segmentation workflow - "
            "detects/tiles rows on a census-style image and writes a sidecar "
            "JSON + debug overlay. For visual, interactive boundary tuning, "
            "use the 'Row Segmentation UI' button in the Tools tab instead - "
            "this CLI is for re-running with already-known settings.",
        )
        f = self.options_frame

        Label(f, text="Image:").grid(row=0, column=0, sticky="w")
        self.image_path_var = StringVar(value="")
        Entry(f, textvariable=self.image_path_var, width=50).grid(row=0, column=1, sticky="w")
        Button(f, text="Browse...", command=self._browse_image).grid(row=0, column=2, padx=(4, 0))

        Label(f, text="Mode:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.mode_var = StringVar(value="periodic")
        OptionMenu(f, self.mode_var, "detect", "periodic",
                   command=self.update_preview).grid(row=1, column=1, sticky="w", pady=(6, 0))

        Label(f, text="Row count:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.row_count_var = IntVar(value=50)
        Entry(f, textvariable=self.row_count_var, width=8).grid(row=2, column=1, sticky="w", pady=(6, 0))

        for i, (label, attr) in enumerate([
            ("Table top:", "table_top_var"), ("Table bottom:", "table_bottom_var"),
            ("Metadata bottom:", "metadata_bottom_var"), ("Table left:", "table_left_var"),
            ("Table right:", "table_right_var"),
        ]):
            row = 3 + i // 3
            col = (i % 3) * 2
            Label(f, text=label).grid(row=row, column=col, sticky="w", pady=(6, 0))
            var = StringVar(value="")
            setattr(self, attr, var)
            Entry(f, textvariable=var, width=8).grid(row=row, column=col + 1, sticky="w",
                                                        padx=(2, 12), pady=(6, 0))

        Label(f, text="Deskew angle:").grid(row=6, column=0, sticky="w", pady=(6, 0))
        self.deskew_angle_var = StringVar(value="")
        Entry(f, textvariable=self.deskew_angle_var, width=8).grid(row=6, column=1, sticky="w", pady=(6, 0))

        Label(f, text="Header rows:").grid(row=6, column=2, sticky="w", pady=(6, 0))
        self.header_rows_var = IntVar(value=1)
        Entry(f, textvariable=self.header_rows_var, width=8).grid(row=6, column=3, sticky="w", pady=(6, 0))

        Label(f, text="Search radius ratio:").grid(row=7, column=0, sticky="w", pady=(6, 0))
        self.search_radius_var = StringVar(value="0.3")
        Entry(f, textvariable=self.search_radius_var, width=8).grid(row=7, column=1, sticky="w", pady=(6, 0))

        Label(f, text="Padding (px):").grid(row=7, column=2, sticky="w", pady=(6, 0))
        self.padding_var = StringVar(value="4")
        Entry(f, textvariable=self.padding_var, width=8).grid(row=7, column=3, sticky="w", pady=(6, 0))

        Label(f, text="Padding %:").grid(row=8, column=0, sticky="w", pady=(6, 0))
        self.padding_pct_var = StringVar(value="")
        Entry(f, textvariable=self.padding_pct_var, width=8).grid(row=8, column=1, sticky="w", pady=(6, 0))

        self.debug_crops_var = BooleanVar(value=False)
        Checkbutton(f, text="Save individual row PNGs (--debug-crops)",
                    variable=self.debug_crops_var, command=self.update_preview).grid(
            row=8, column=2, columnspan=2, sticky="w", pady=(6, 0))

        Label(f, text="Output dir:").grid(row=9, column=0, sticky="w", pady=(6, 0))
        self.out_var = StringVar(value="data/outputs/row_segmentation")
        Entry(f, textvariable=self.out_var, width=40).grid(row=9, column=1, columnspan=3,
                                                              sticky="w", pady=(6, 0))

        for var in [self.image_path_var, self.row_count_var, self.table_top_var,
                    self.table_bottom_var, self.metadata_bottom_var, self.table_left_var,
                    self.table_right_var, self.deskew_angle_var, self.header_rows_var,
                    self.search_radius_var, self.padding_var, self.padding_pct_var,
                    self.out_var]:
            var.trace_add("write", self.update_preview)
        self.update_preview()

    def _browse_image(self):
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp")])
        if path:
            self.image_path_var.set(path)

    def build_command(self, silent: bool = False) -> list[str] | None:
        image_path = self.image_path_var.get().strip()
        if not image_path:
            if not silent:
                messagebox.showwarning("Missing input", "Select an image first.")
            return None
        cmd = [PYTHON, "test_row_segmentation.py", image_path,
               "--mode", self.mode_var.get(),
               "--row-count", str(self.row_count_var.get()),
               "--header-rows", str(self.header_rows_var.get()),
               "--search-radius-ratio", self.search_radius_var.get() or "0.3",
               "--padding", self.padding_var.get() or "4",
               "--out", self.out_var.get() or "data/outputs/row_segmentation"]
        for flag, var in [("--table-top", self.table_top_var),
                           ("--table-bottom", self.table_bottom_var),
                           ("--metadata-bottom", self.metadata_bottom_var),
                           ("--table-left", self.table_left_var),
                           ("--table-right", self.table_right_var),
                           ("--deskew-angle", self.deskew_angle_var),
                           ("--padding-pct", self.padding_pct_var)]:
            val = var.get().strip()
            if val:
                cmd.extend([flag, val])
        if self.debug_crops_var.get():
            cmd.append("--debug-crops")
        return cmd

    def input_folder(self) -> Path:
        p = self.image_path_var.get().strip()
        return Path(p).parent if p else PROJECT_ROOT

    def output_folder(self) -> Path:
        return PROJECT_ROOT / (self.out_var.get() or "data/outputs/row_segmentation")

    def capture_state(self) -> dict:
        return {
            "mode": self.mode_var.get(), "row_count": self.row_count_var.get(),
            "header_rows": self.header_rows_var.get(),
            "search_radius": self.search_radius_var.get(),
            "padding": self.padding_var.get(), "out": self.out_var.get(),
        }

    def restore_state(self, values: dict) -> None:
        if "mode" in values:
            self.mode_var.set(values["mode"])
        if "row_count" in values:
            self.row_count_var.set(values["row_count"])
        if "header_rows" in values:
            self.header_rows_var.set(values["header_rows"])
        if "search_radius" in values:
            self.search_radius_var.set(values["search_radius"])
        if "padding" in values:
            self.padding_var.set(values["padding"])
        if "out" in values:
            self.out_var.set(values["out"])


class RowExtractionTab(CommandTab):
    def __init__(self, parent, app):
        super().__init__(
            parent, app, "row_extraction",
            "2. Row Extraction (run_row_extraction.py)",
            "Single-stage extraction: runs one model against every row in a "
            "sidecar, structuring output against a column list. Optionally "
            "also extracts the page header/metadata block using the same "
            "model, sharing one load.",
        )
        f = self.options_frame

        Label(f, text="Sidecar:").grid(row=0, column=0, sticky="w")
        self.sidecar_var = StringVar(value="")
        self.sidecar_combo = ttk.Combobox(f, textvariable=self.sidecar_var, width=47,
                                           values=_scan(SIDECAR_DIR, "*_sidecar.json"))
        self.sidecar_combo.grid(row=0, column=1, sticky="w")
        Button(f, text="Refresh", command=self._refresh_sidecars).grid(row=0, column=2, padx=(4, 0))

        Label(f, text="Columns file:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.columns_var = StringVar(value="")
        ttk.Combobox(f, textvariable=self.columns_var, width=47,
                     values=_scan(COLUMNS_DIR, "*.txt")).grid(row=1, column=1, sticky="w", pady=(6, 0))

        Label(f, text="Model:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.model_var = StringVar(value="")
        ttk.Combobox(f, textvariable=self.model_var, width=47,
                     values=_scan_stems(MODELS_DIR, "*.yaml")).grid(row=2, column=1, sticky="w", pady=(6, 0))

        Label(f, text="Header fields file (optional):").grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.header_fields_var = StringVar(value="")
        ttk.Combobox(f, textvariable=self.header_fields_var, width=47,
                     values=[""] + _scan(COLUMNS_DIR, "*.txt")).grid(row=3, column=1, sticky="w", pady=(6, 0))

        Label(f, text="Max rows (blank = all):").grid(row=4, column=0, sticky="w", pady=(6, 0))
        self.max_rows_var = StringVar(value="")
        Entry(f, textvariable=self.max_rows_var, width=8).grid(row=4, column=1, sticky="w", pady=(6, 0))

        Label(f, text="Output dir (blank = sidecar's folder):").grid(row=5, column=0, sticky="w", pady=(6, 0))
        self.out_var = StringVar(value="")
        Entry(f, textvariable=self.out_var, width=47).grid(row=5, column=1, sticky="w", pady=(6, 0))

        for var in [self.sidecar_var, self.columns_var, self.model_var,
                    self.header_fields_var, self.max_rows_var, self.out_var]:
            var.trace_add("write", self.update_preview)
        self.update_preview()

    def _refresh_sidecars(self):
        self.sidecar_combo["values"] = _scan(SIDECAR_DIR, "*_sidecar.json")

    def _sidecar_path(self) -> str:
        v = self.sidecar_var.get().strip()
        if not v:
            return ""
        return _relative_or_absolute(SIDECAR_DIR, v)

    def _columns_path(self) -> str:
        v = self.columns_var.get().strip()
        if not v:
            return ""
        return _relative_or_absolute(COLUMNS_DIR, v)

    def build_command(self, silent: bool = False) -> list[str] | None:
        sidecar = self._sidecar_path()
        columns = self._columns_path()
        model = self.model_var.get().strip()
        if not sidecar or not columns or not model:
            if not silent:
                messagebox.showwarning(
                    "Missing input", "Sidecar, columns file, and model are all required.")
            return None
        cmd = [PYTHON, "run_row_extraction.py", sidecar, columns, "--model", model]
        max_rows = self.max_rows_var.get().strip()
        if max_rows:
            cmd.extend(["--max-rows", max_rows])
        header_fields = self.header_fields_var.get().strip()
        if header_fields:
            hf_path = _relative_or_absolute(COLUMNS_DIR, header_fields)
            if hf_path:
                cmd.extend(["--header-fields", hf_path])
        out = self.out_var.get().strip()
        if out:
            cmd.extend(["--out", out])
        return cmd

    def input_folder(self) -> Path:
        return SIDECAR_DIR

    def output_folder(self) -> Path:
        out = self.out_var.get().strip()
        return Path(out) if out else SIDECAR_DIR

    def capture_state(self) -> dict:
        return {"model": self.model_var.get(), "header_fields": self.header_fields_var.get(),
                "max_rows": self.max_rows_var.get()}

    def restore_state(self, values: dict) -> None:
        if "model" in values:
            self.model_var.set(values["model"])
        if "header_fields" in values:
            self.header_fields_var.set(values["header_fields"])
        if "max_rows" in values:
            self.max_rows_var.set(values["max_rows"])


class TwoStageTab(CommandTab):
    def __init__(self, parent, app):
        super().__init__(
            parent, app, "two_stage",
            "3. Two-Stage Extraction (run_two_stage_extraction.py)",
            "Stage 1 (a raw OCR engine) reads each row; stage 2 (an "
            "instruction-following model) structures that reading, plus the "
            "row image itself, against the column list.",
        )
        f = self.options_frame

        Label(f, text="Sidecar:").grid(row=0, column=0, sticky="w")
        self.sidecar_var = StringVar(value="")
        self.sidecar_combo = ttk.Combobox(f, textvariable=self.sidecar_var, width=47,
                                           values=_scan(SIDECAR_DIR, "*_sidecar.json"))
        self.sidecar_combo.grid(row=0, column=1, sticky="w")
        Button(f, text="Refresh all", command=self._refresh_all).grid(row=0, column=2, padx=(4, 0))

        Label(f, text="Columns file:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.columns_var = StringVar(value="")
        self.columns_combo = ttk.Combobox(f, textvariable=self.columns_var, width=47,
                     values=_scan(COLUMNS_DIR, "*.txt"))
        self.columns_combo.grid(row=1, column=1, sticky="w", pady=(6, 0))

        Label(f, text="Stage 1 (OCR) model:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.ocr_model_var = StringVar(value="")
        self.ocr_model_combo = ttk.Combobox(f, textvariable=self.ocr_model_var, width=47,
                     values=_scan_stems(MODELS_DIR, "*.yaml"))
        self.ocr_model_combo.grid(row=2, column=1, sticky="w", pady=(6, 0))

        Label(f, text="Stage 2 (structure) model:").grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.structure_model_var = StringVar(value="")
        self.structure_model_combo = ttk.Combobox(f, textvariable=self.structure_model_var, width=47,
                     values=_scan_stems(MODELS_DIR, "*.yaml"))
        self.structure_model_combo.grid(row=3, column=1, sticky="w", pady=(6, 0))

        Label(f, text="Stage 1 prompt file (optional):").grid(row=4, column=0, sticky="w", pady=(6, 0))
        self.ocr_prompt_var = StringVar(value="")
        self.ocr_prompt_combo = ttk.Combobox(f, textvariable=self.ocr_prompt_var, width=47,
                     values=[""] + _scan(PROMPTS_DIR, "ocr_stage1_*.txt"))
        self.ocr_prompt_combo.grid(row=4, column=1, sticky="w", pady=(6, 0))

        Label(f, text="Stage 2 prompt template (optional):").grid(row=5, column=0, sticky="w", pady=(6, 0))
        self.structuring_prompt_var = StringVar(value="")
        self.structuring_prompt_combo = ttk.Combobox(f, textvariable=self.structuring_prompt_var, width=47,
                     values=[""] + _scan(PROMPTS_DIR, "structuring_stage2_*.txt"))
        self.structuring_prompt_combo.grid(row=5, column=1, sticky="w", pady=(6, 0))

        Label(f, text="Max rows (blank = all):").grid(row=6, column=0, sticky="w", pady=(6, 0))
        self.max_rows_var = StringVar(value="")
        Entry(f, textvariable=self.max_rows_var, width=8).grid(row=6, column=1, sticky="w", pady=(6, 0))

        Label(f, text="Output dir (blank = sidecar's folder):").grid(row=7, column=0, sticky="w", pady=(6, 0))
        self.out_var = StringVar(value="")
        Entry(f, textvariable=self.out_var, width=47).grid(row=7, column=1, sticky="w", pady=(6, 0))

        for var in [self.sidecar_var, self.columns_var, self.ocr_model_var,
                    self.structure_model_var, self.ocr_prompt_var, self.structuring_prompt_var,
                    self.max_rows_var, self.out_var]:
            var.trace_add("write", self.update_preview)
        self.update_preview()

    def _refresh_all(self):
        """
        Re-scans every source folder this tab draws from, not just
        sidecars (2026-07-16, per Jon's direction) - new column files,
        model configs, or stage-1 prompt files created after this GUI
        launched (e.g. a new prompt file created mid-session while a
        test was running) previously needed a full app restart to show
        up anywhere except the sidecar dropdown.
        """
        self.sidecar_combo["values"] = _scan(SIDECAR_DIR, "*_sidecar.json")
        self.columns_combo["values"] = _scan(COLUMNS_DIR, "*.txt")
        self.ocr_model_combo["values"] = _scan_stems(MODELS_DIR, "*.yaml")
        self.structure_model_combo["values"] = _scan_stems(MODELS_DIR, "*.yaml")
        self.ocr_prompt_combo["values"] = [""] + _scan(PROMPTS_DIR, "ocr_stage1_*.txt")
        self.structuring_prompt_combo["values"] = [""] + _scan(PROMPTS_DIR, "structuring_stage2_*.txt")

    def _sidecar_path(self) -> str:
        v = self.sidecar_var.get().strip()
        if not v:
            return ""
        return _relative_or_absolute(SIDECAR_DIR, v)

    def _columns_path(self) -> str:
        v = self.columns_var.get().strip()
        if not v:
            return ""
        return _relative_or_absolute(COLUMNS_DIR, v)

    def build_command(self, silent: bool = False) -> list[str] | None:
        sidecar = self._sidecar_path()
        columns = self._columns_path()
        ocr_model = self.ocr_model_var.get().strip()
        structure_model = self.structure_model_var.get().strip()
        if not sidecar or not columns or not ocr_model or not structure_model:
            if not silent:
                messagebox.showwarning(
                    "Missing input",
                    "Sidecar, columns file, stage-1 model, and stage-2 model are all required.")
            return None
        cmd = [PYTHON, "run_two_stage_extraction.py", sidecar, columns,
               "--ocr-model", ocr_model, "--structure-model", structure_model]
        max_rows = self.max_rows_var.get().strip()
        if max_rows:
            cmd.extend(["--max-rows", max_rows])
        ocr_prompt = self.ocr_prompt_var.get().strip()
        if ocr_prompt:
            op_path = _relative_or_absolute(PROMPTS_DIR, ocr_prompt)
            if op_path:
                cmd.extend(["--ocr-prompt-file", op_path])
        structuring_prompt = self.structuring_prompt_var.get().strip()
        if structuring_prompt:
            sp_path = _relative_or_absolute(PROMPTS_DIR, structuring_prompt)
            if sp_path:
                cmd.extend(["--structuring-prompt-file", sp_path])
        out = self.out_var.get().strip()
        if out:
            cmd.extend(["--out", out])
        return cmd

    def input_folder(self) -> Path:
        return SIDECAR_DIR

    def output_folder(self) -> Path:
        out = self.out_var.get().strip()
        return Path(out) if out else SIDECAR_DIR

    def capture_state(self) -> dict:
        return {"ocr_model": self.ocr_model_var.get(),
                "structure_model": self.structure_model_var.get(),
                "ocr_prompt": self.ocr_prompt_var.get(),
                "structuring_prompt": self.structuring_prompt_var.get(),
                "max_rows": self.max_rows_var.get()}

    def restore_state(self, values: dict) -> None:
        if "ocr_model" in values:
            self.ocr_model_var.set(values["ocr_model"])
        if "structure_model" in values:
            self.structure_model_var.set(values["structure_model"])
        if "ocr_prompt" in values:
            self.ocr_prompt_var.set(values["ocr_prompt"])
        if "structuring_prompt" in values:
            self.structuring_prompt_var.set(values["structuring_prompt"])
        if "max_rows" in values:
            self.max_rows_var.set(values["max_rows"])


class NativePromptTab(CommandTab):
    def __init__(self, parent, app):
        super().__init__(
            parent, app, "native_prompt",
            "4. Native Prompt Test (test_native_prompt.py)",
            "Runs a model's OWN built-in prompt (not our column format) "
            "against one row or the header region - raw output only, no "
            "parsing. Only meaningful for loaders with a real native prompt "
            "(e.g. olmocr_2_7b); most models just use their configured "
            "prompt_text, which is usually empty.",
        )
        f = self.options_frame

        Label(f, text="Sidecar:").grid(row=0, column=0, sticky="w")
        self.sidecar_var = StringVar(value="")
        self.sidecar_combo = ttk.Combobox(f, textvariable=self.sidecar_var, width=47,
                                           values=_scan(SIDECAR_DIR, "*_sidecar.json"))
        self.sidecar_combo.grid(row=0, column=1, sticky="w")
        Button(f, text="Refresh", command=self._refresh_sidecars).grid(row=0, column=2, padx=(4, 0))

        Label(f, text="Model:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.model_var = StringVar(value="")
        ttk.Combobox(f, textvariable=self.model_var, width=47,
                     values=_scan_stems(MODELS_DIR, "*.yaml")).grid(row=1, column=1, sticky="w", pady=(6, 0))

        self.target_var = StringVar(value="row")
        Label(f, text="Target:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        target_frame = Frame(f)
        target_frame.grid(row=2, column=1, sticky="w", pady=(6, 0))
        for label, val in [("Row #", "row"), ("Header region", "header")]:
            Button(target_frame, text=label,
                   command=lambda v=val: self._set_target(v)).pack(side="left")
        Label(f, text="Row number (if Target=row):").grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.row_var = StringVar(value="1")
        Entry(f, textvariable=self.row_var, width=8).grid(row=3, column=1, sticky="w", pady=(6, 0))

        for var in [self.sidecar_var, self.model_var, self.row_var]:
            var.trace_add("write", self.update_preview)
        self.update_preview()

    def _set_target(self, val):
        self.target_var.set(val)
        self.update_preview()

    def _refresh_sidecars(self):
        self.sidecar_combo["values"] = _scan(SIDECAR_DIR, "*_sidecar.json")

    def _sidecar_path(self) -> str:
        v = self.sidecar_var.get().strip()
        if not v:
            return ""
        return _relative_or_absolute(SIDECAR_DIR, v)

    def build_command(self, silent: bool = False) -> list[str] | None:
        sidecar = self._sidecar_path()
        model = self.model_var.get().strip()
        if not sidecar or not model:
            if not silent:
                messagebox.showwarning("Missing input", "Sidecar and model are both required.")
            return None
        cmd = [PYTHON, "test_native_prompt.py", sidecar, "--model", model]
        if self.target_var.get() == "header":
            cmd.append("--header")
        else:
            row = self.row_var.get().strip() or "1"
            cmd.extend(["--row", row])
        return cmd

    def input_folder(self) -> Path:
        return SIDECAR_DIR

    def output_folder(self) -> Path:
        return SIDECAR_DIR

    def capture_state(self) -> dict:
        return {"model": self.model_var.get(), "target": self.target_var.get(),
                "row": self.row_var.get()}

    def restore_state(self, values: dict) -> None:
        if "model" in values:
            self.model_var.set(values["model"])
        if "target" in values:
            self.target_var.set(values["target"])
        if "row" in values:
            self.row_var.set(values["row"])


class ClassifierTab(CommandTab):
    def __init__(self, parent, app):
        super().__init__(
            parent, app, "classifier",
            "5. Classification (python -m core.classifier)",
            "Runs the Gemma classifier against a manifest CSV, writing "
            "per-bucket CSVs to data/outputs/. This CLI is intentionally "
            "minimal (one positional argument, no flags) - use 'Build "
            "Manifest' in the Tools tab first to create a manifest.csv "
            "from a folder of images.",
        )
        f = self.options_frame

        Label(f, text="Manifest CSV:").grid(row=0, column=0, sticky="w")
        self.manifest_var = StringVar(value="data/manifest.csv")
        Entry(f, textvariable=self.manifest_var, width=50).grid(row=0, column=1, sticky="w")
        Button(f, text="Browse...", command=self._browse_manifest).grid(row=0, column=2, padx=(4, 0))

        self.manifest_var.trace_add("write", self.update_preview)
        self.update_preview()

    def _browse_manifest(self):
        path = filedialog.askopenfilename(title="Select manifest CSV",
                                           filetypes=[("CSV", "*.csv")])
        if path:
            self.manifest_var.set(path)

    def build_command(self, silent: bool = False) -> list[str] | None:
        manifest = self.manifest_var.get().strip()
        if not manifest:
            if not silent:
                messagebox.showwarning("Missing input", "Select a manifest CSV first.")
            return None
        return [PYTHON, "-m", "core.classifier", manifest]

    def input_folder(self) -> Path:
        m = self.manifest_var.get().strip()
        return Path(m).parent if m else DATA_DIR

    def output_folder(self) -> Path:
        return DATA_DIR / "outputs"

    def capture_state(self) -> dict:
        return {"manifest": self.manifest_var.get()}

    def restore_state(self, values: dict) -> None:
        if "manifest" in values:
            self.manifest_var.set(values["manifest"])


class ScoringTab:
    """
    Wraps score_two_stage_against_ground_truth.py's score_results() -
    built 2026-07-22 after enough real testing sessions where getting a
    number meant re-typing four file paths on the CLI every single run.
    Remembers each path across sessions (same _load_state/_save_state
    mechanism every other tab uses) and shows a real sortable results
    table instead of parsed console text.

    DELIBERATELY DOES NOT SUBCLASS CommandTab: every other tab shells
    out to a CLI script as a subprocess and streams raw stdout - fine
    for a log, useless for a sortable table, since there's no
    structured data to sort once it's already been formatted as text.
    This tab imports score_two_stage_against_ground_truth.score_results()
    directly and calls it in-process instead - CPU-cheap (no model
    loading, just JSONL parsing and dict comparisons), so there's no
    real cost to running it on the main thread synchronously, unlike
    every other tab's subprocess+thread+queue plumbing which exists
    specifically because THOSE commands can run for minutes.

    STILL a thin wrapper in the sense that matters: score_results() is
    the ONLY place the actual scoring logic lives. This tab and the
    CLI's main() both call it and would produce byte-for-byte identical
    numbers given the same inputs - verified directly (see this
    session's commit history) by running both against the same real
    files.

    Explicitly deferred from the original spec (real added complexity,
    confirm priority before building): character error rate, double-
    click-to-open-crop-image, drag-and-drop, and named/saved run
    profiles beyond basic remembered-last-used paths.
    """

    def __init__(self, parent, app: "WorkflowApp"):
        self.app = app
        self.tab_key = "scoring"
        self.frame = Frame(parent, padx=10, pady=10)
        self.last_report = None  # ScoringReport | None - kept for "save report"

        Label(self.frame, text="Score extraction results against ground truth",
              font=("Segoe UI", 11, "bold")).pack(anchor="w")
        Label(self.frame,
              text="Wraps score_two_stage_against_ground_truth.score_results() directly "
                   "(in-process, not a subprocess) - same evaluator the CLI uses, so "
                   "numbers here and on the command line always match exactly.",
              font=("Segoe UI", 9), fg="#444", wraplength=1000, justify="left").pack(
            anchor="w", pady=(2, 10))

        inputs = Frame(self.frame)
        inputs.pack(fill="x")

        self.results_var = StringVar(value="")
        self.gt_log_var = StringVar(value="")
        self.sidecar_var = StringVar(value="")
        self.out_dir_var = StringVar(value=str(DATA_DIR / "outputs" / "scoring_reports"))
        self.model_name_var = StringVar(value="")

        def _row(label_text, var, r, browse_kind):
            Label(inputs, text=label_text).grid(row=r, column=0, sticky="w", pady=(4, 0))
            Entry(inputs, textvariable=var, width=70).grid(
                row=r, column=1, sticky="w", pady=(4, 0))
            Button(inputs, text="Browse...",
                   command=lambda: self._browse(var, browse_kind)).grid(
                row=r, column=2, padx=(4, 0), pady=(4, 0))

        _row("Prediction/results JSON:", self.results_var, 0, "file")
        _row("Ground-truth log (.jsonl):", self.gt_log_var, 1, "file")
        _row("Sidecar (.json) for this page:", self.sidecar_var, 2, "file")
        _row("Output report folder:", self.out_dir_var, 3, "dir")

        Label(inputs, text="Model/run name (optional label):").grid(
            row=4, column=0, sticky="w", pady=(4, 0))
        Entry(inputs, textvariable=self.model_name_var, width=40).grid(
            row=4, column=1, sticky="w", pady=(4, 0))

        run_config = Frame(self.frame)
        run_config.pack(fill="x", pady=(8, 0))
        self.save_report_var = BooleanVar(value=True)
        Checkbutton(run_config, text="Save detailed mismatch report to output folder",
                    variable=self.save_report_var).pack(side="left")
        self.open_after_var = BooleanVar(value=False)
        Checkbutton(run_config, text="Open report after completion",
                    variable=self.open_after_var).pack(side="left", padx=(12, 0))

        action_row = Frame(self.frame)
        action_row.pack(fill="x", pady=(8, 6))
        Button(action_row, text="Score", command=self._on_score,
               bg="#4a7", fg="white", font=("Segoe UI", 10, "bold")).pack(side="left")
        self.status_label = Label(action_row, text="", fg="#444")
        self.status_label.pack(side="left", padx=12)

        # -- headline results --------------------------------------------
        headline = Frame(self.frame, relief="groove", borderwidth=1, padx=8, pady=6)
        headline.pack(fill="x", pady=(4, 8))
        self.headline_labels = {}
        headline_fields = [
            "Overall accuracy", "False confidence", "Schema failures",
            "Matched / total records", "Rows in results", "Total runtime (extraction)",
        ]
        for i, name in enumerate(headline_fields):
            Label(headline, text=f"{name}:", font=("Segoe UI", 9, "bold")).grid(
                row=i // 3, column=(i % 3) * 2, sticky="w", padx=(0, 4), pady=2)
            lbl = Label(headline, text="-", font=("Segoe UI", 9))
            lbl.grid(row=i // 3, column=(i % 3) * 2 + 1, sticky="w", padx=(0, 20), pady=2)
            self.headline_labels[name] = lbl

        # -- sortable failure table ---------------------------------------
        Label(self.frame, text="Mismatches (click a column header to sort):",
              font=("Segoe UI", 9, "bold")).pack(anchor="w")
        table_frame = Frame(self.frame)
        table_frame.pack(fill="both", expand=True)
        columns = ("row", "column", "status", "expected", "predicted", "confidence", "flag")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=12)
        headers = {
            "row": "Row", "column": "Column", "status": "GT Status",
            "expected": "Ground Truth", "predicted": "Prediction",
            "confidence": "Pred. Confidence", "flag": "Flag",
        }
        widths = {"row": 50, "column": 130, "status": 90, "expected": 140,
                  "predicted": 140, "confidence": 100, "flag": 120}
        for col in columns:
            self.tree.heading(col, text=headers[col],
                               command=lambda c=col: self._sort_by(c, False))
            self.tree.column(col, width=widths[col], anchor="w")
        tree_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="right", fill="y")

    def _browse(self, var: StringVar, kind: str):
        if kind == "file":
            path = filedialog.askopenfilename(
                initialdir=str(Path(var.get()).parent) if var.get() else str(DATA_DIR))
        else:
            path = filedialog.askdirectory(
                initialdir=var.get() if var.get() else str(DATA_DIR))
        if path:
            var.set(path)

    def _sort_by(self, col: str, reverse: bool):
        items = [(self.tree.set(k, col), k) for k in self.tree.get_children("")]
        try:
            items.sort(key=lambda t: (int(t[0]) if t[0].lstrip("-").isdigit() else t[0]),
                       reverse=reverse)
        except (ValueError, TypeError):
            items.sort(key=lambda t: t[0], reverse=reverse)
        for index, (_, k) in enumerate(items):
            self.tree.move(k, "", index)
        # clicking the same header again reverses order next time
        self.tree.heading(col, command=lambda: self._sort_by(col, not reverse))

    def _on_score(self):
        results_path = self.results_var.get().strip()
        gt_path = self.gt_log_var.get().strip()
        sidecar_path = self.sidecar_var.get().strip()
        if not results_path or not gt_path or not sidecar_path:
            messagebox.showwarning(
                "Missing input",
                "Prediction JSON, ground-truth log, and sidecar path are all required.")
            return

        self.app.save_all_state()
        self.status_label.config(text="Scoring...")
        self.frame.update_idletasks()

        # Imported here, not at module top - keeps this GUI's startup
        # dependency-light (matches this file's own stated principle of
        # only importing what a given tab's Run action actually needs),
        # and this scorer module has zero heavy dependencies anyway
        # (pure stdlib), so this is cheap either way.
        from score_two_stage_against_ground_truth import score_results

        report = score_results(results_path, gt_path, sidecar_path,
                                model_name=self.model_name_var.get().strip() or None)
        self.last_report = report

        if report.error:
            self.status_label.config(text="Error")
            msg = report.error
            if report.available_sidecar_paths:
                msg += "\n\nAvailable sidecar_paths in this log:\n" + "\n".join(
                    report.available_sidecar_paths)
            messagebox.showerror("Scoring failed", msg)
            return

        self._render_report(report)
        self.status_label.config(text="Done")

        if self.save_report_var.get():
            out_path = self._save_report(report)
            if self.open_after_var.get() and out_path:
                _open_folder(out_path.parent)

    def _render_report(self, report):
        matched = sum(c["total"] - c["missing"] for c in report.per_column.values())
        total = sum(c["total"] for c in report.per_column.values())
        total_runtime = None
        try:
            with open(report.results_path, "r", encoding="utf-8") as f:
                rows = json.load(f)
            total_runtime = sum(r.get("runtime_seconds", 0) for r in rows)
        except Exception:
            pass

        self.headline_labels["Overall accuracy"].config(
            text=f"{report.overall_correct}/{report.overall_total} "
                 f"({report.overall_accuracy:.0%})")
        fc = report.false_confidence
        self.headline_labels["False confidence"].config(
            text=(f"{len(fc)}/{len(report.mismatches)} wrong answers "
                  f"({report.false_confidence_rate:.0%})") if report.mismatches
            else "none")
        self.headline_labels["Schema failures"].config(
            text=f"{report.schema_failed}/{report.total_rows_in_results}")
        self.headline_labels["Matched / total records"].config(text=f"{matched}/{total}")
        self.headline_labels["Rows in results"].config(text=str(report.total_rows_in_results))
        self.headline_labels["Total runtime (extraction)"].config(
            text=f"{total_runtime:.1f}s" if total_runtime is not None else "unknown")

        self.tree.delete(*self.tree.get_children())
        for m in report.mismatches:
            flag = "FALSE CONFIDENCE" if m["predicted_confidence"] == "confirmed" else ""
            self.tree.insert("", "end", values=(
                m["row"], m["column"], m["status"], m["expected"], m["predicted"],
                m["predicted_confidence"], flag,
            ))

    def _save_report(self, report) -> Path | None:
        out_dir = Path(self.out_dir_var.get().strip() or str(DATA_DIR / "outputs" / "scoring_reports"))
        out_dir.mkdir(parents=True, exist_ok=True)
        name = report.model_name or Path(report.results_path).stem
        safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
        out_path = out_dir / f"{safe_name}_scoring_report.json"
        try:
            from dataclasses import asdict
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(asdict(report), f, indent=2)
            return out_path
        except Exception as e:
            messagebox.showwarning("Could not save report", str(e))
            return None

    # -- CommandTab-compatible interface (WorkflowApp calls these on
    # every tab uniformly, regardless of subclass) -----------------------

    def update_preview(self, *_):
        pass  # no command preview - this tab doesn't shell out to a CLI

    def capture_state(self) -> dict:
        return {
            "results": self.results_var.get(), "gt_log": self.gt_log_var.get(),
            "sidecar": self.sidecar_var.get(), "out_dir": self.out_dir_var.get(),
            "model_name": self.model_name_var.get(),
            "save_report": self.save_report_var.get(), "open_after": self.open_after_var.get(),
        }

    def restore_state(self, values: dict) -> None:
        if "results" in values:
            self.results_var.set(values["results"])
        if "gt_log" in values:
            self.gt_log_var.set(values["gt_log"])
        if "sidecar" in values:
            self.sidecar_var.set(values["sidecar"])
        if "out_dir" in values:
            self.out_dir_var.set(values["out_dir"])
        if "model_name" in values:
            self.model_name_var.set(values["model_name"])
        if "save_report" in values:
            self.save_report_var.set(values["save_report"])
        if "open_after" in values:
            self.open_after_var.set(values["open_after"])


class ToolsTab:
    """
    Launch buttons for the standalone GUI apps already built this
    session - these are full self-contained tools, not option-driven
    CLI scripts, so they get a simple "launch as a separate process"
    button rather than a comboboxes-and-preview treatment. Same "call
    the real entry point, don't duplicate logic" principle as every
    other tab, just applied to a different shape of tool.
    """

    def __init__(self, parent, app):
        self.app = app
        self.frame = Frame(parent, padx=10, pady=10)
        Label(self.frame, text="6. Other Tools", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        Label(self.frame, text="These are full standalone GUI apps, launched as separate "
                                "processes - not wrapped with options here.",
              font=("Segoe UI", 9), fg="#444").pack(anchor="w", pady=(2, 12))

        tools = [
            ("Row Segmentation UI", "row_segmentation_ui.py",
             "Interactive, visual deskew/bounds/row confirmation - the "
             "primary tool for producing a sidecar JSON before running "
             "any extraction step."),
            ("Model Assessment", "model_assessment.py",
             "Test a single model/prompt/bucket-profile combination "
             "against one image, with a preprocessing dropdown and raw-"
             "output review."),
            ("Review Uncertain", "review_uncertain.py",
             "Manually assign a final bucket to images the classifier "
             "couldn't confidently place."),
            ("Build Manifest", "build_manifest.py",
             "Pick a folder of images, write data/manifest.csv - the "
             "input the Classification tab needs."),
        ]
        for i, (label, script, desc) in enumerate(tools):
            row_frame = Frame(self.frame, relief="groove", borderwidth=1, padx=8, pady=6)
            row_frame.pack(fill="x", pady=4)
            Label(row_frame, text=label, font=("Segoe UI", 10, "bold")).pack(anchor="w")
            Label(row_frame, text=desc, font=("Segoe UI", 9), fg="#444",
                  wraplength=900, justify="left").pack(anchor="w", pady=(2, 6))
            Button(row_frame, text=f"Launch {label}",
                   command=lambda s=script: self._launch(s)).pack(anchor="w")

    def _launch(self, script: str):
        try:
            subprocess.Popen([PYTHON, script], cwd=str(PROJECT_ROOT))
        except Exception as e:
            messagebox.showerror("Could not launch", str(e))


class WorkflowApp:
    def __init__(self, root: Tk):
        self.root = root
        root.title("Pipeline Workflow - CLI wrapper (calls real entry points, no duplicated logic)")
        root.geometry("1150x850")
        root.minsize(950, 650)

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True)

        self.tabs: list = []
        for TabClass, label in [
            (SegmentationTab, "1. Segmentation"),
            (RowExtractionTab, "2. Row Extraction"),
            (TwoStageTab, "3. Two-Stage Extraction"),
            (NativePromptTab, "4. Native Prompt Test"),
            (ClassifierTab, "5. Classification"),
        ]:
            tab = TabClass(notebook, self)
            notebook.add(tab.frame, text=label)
            self.tabs.append(tab)

        scoring_tab = ScoringTab(notebook, self)
        notebook.add(scoring_tab.frame, text="6. Scoring")
        self.tabs.append(scoring_tab)

        tools_tab = ToolsTab(notebook, self)
        notebook.add(tools_tab.frame, text="7. Tools")

        self._restore_all_state()

    def save_all_state(self):
        state = _load_state()
        for tab in self.tabs:
            state[tab.tab_key] = tab.capture_state()
        _save_state(state)

    def _restore_all_state(self):
        state = _load_state()
        for tab in self.tabs:
            if tab.tab_key in state:
                tab.restore_state(state[tab.tab_key])
                tab.update_preview()


if __name__ == "__main__":
    root = Tk()
    app = WorkflowApp(root)
    root.mainloop()
