"""
Build Manifest UI - a Pipeline Launcher front end: the window shows ONE
stage's view at a time (Manifest -> Preprocessing -> Classification),
and swaps to the next view on an explicit user action - not a wizard
with Next/Back buttons, and not one window that accumulates every
stage's controls forever.

LEGACY MODULE RETIRED FROM THIS UI, 2026-07-30 - scripts/build_manifest.py
(Jon: "build_manifest was the initial starting module. its legacy now
and wont be used going forward") is NOT imported here. What used to be
this UI's "Build Manifest" step (write a raw file list straight to CSV,
no processing) is now real work spanning new Stage 0/1/3 (see docs/
PIPELINE_STAGE_TERMINOLOGY.md - not yet split apart): copying queued
files into data/working/, deskewing, and preprocessing them, via
core/manifest_pipeline.py's build_working_manifest_from_paths() - the
ACTUAL pipeline engine, run through a small new CSV-input adapter CLI,
scripts/run_preprocessing.py. There is only ONE manifest artefact this
UI produces now (data/manifest.csv, via core/manifest_pipeline.py's
DEFAULT_MANIFEST_PATH) - the WORKING manifest, pointing at processed
copies. The earlier "Selection Manifest vs Working Manifest" naming
ambiguity (see the project_manifest_pipeline_stage0 memory) doesn't
apply to this UI's own output anymore because the raw-list writer that
caused it is gone; the UI's in-memory `self.paths` queue IS the
selection, and it's never itself serialized as a standalone pipeline
artefact - only handed via a throwaway temp CSV to the preprocessing
subprocess as its input.

THE GUI CONTAINS NO MANIFEST-BUILDING (OR PREPROCESSING, OR
CLASSIFICATION, OR ANY OTHER STAGE'S) LOGIC - every real decision lives
in the real modules/scripts this UI shells out to, imported/invoked
unmodified.

CANVAS SWAP MODEL - the window is `header` (always visible: title,
description, Power User checkbox) + `content_area` (shows exactly ONE
of `manifest_frame` / `preprocessing_frame` / `classification_frame` at
a time, via `_show_frame()`).

  - MANIFEST view: Add File/Add Folder, the file list, and "Build
    Manifest ->" - clicking it is the swap trigger to Preprocessing.
  - PREPROCESSING view: status/progress/log for
    scripts/run_preprocessing.py, and - once it finishes successfully -
    "Run Classification ->", the swap trigger to Classification.
  - CLASSIFICATION view: status/progress/log for core/classifier.py,
    and - once it finishes successfully - the Routing Summary IN PLACE
    on the same view. No "proceed to next stage" button yet - no
    defined next workflow exists (see the
    project_pipeline_launcher_view_architecture memory).

The pattern established across all three views: the button that
LAUNCHES the next stage lives on the CURRENT view, not the next one.
Content updating in place (a summary appearing once a run finishes) is
NOT the same thing as the canvas swapping - a view can react to
pipeline events on its own, but only advances to the NEXT view on a
deliberate click. Swaps are gated on `on_launch`, which fires only once
a stage is ACTUALLY about to run (after any GPU-busy confirmation is
accepted) - not on the button click itself. This matters concretely for
Classification (device is a real GPU name, not "CPU"): swapping
unconditionally on click, then having the user decline the GPU-busy
dialog, would strand them on classification_frame with no way back to
the only button that starts a run. Preprocessing (device="CPU") never
shows that dialog, so its swap effectively happens immediately on click.

There is deliberately no "back" affordance between views - progression
is forward-only; starting over means restarting the app.

WIDGET CONSTRUCTION ORDER (2026-07-30) - each view builds ALL of its own
widgets, including its status/progress/log widgets, FIRST, as plain
instance attributes with no command wired yet. Only after all three
views exist does `_wire_stages()` construct the `Stage` objects
(referencing those already-built real widgets) and `.config(command=...)`
each button. This two-phase split (build widgets, then wire stages) is
deliberate - an earlier version tried building each Stage inline as its
owning view was constructed, which required a stage's log widgets to
exist before the view meant to hold them did, and ended up creating
throwaway placeholder widgets just to destroy and replace them a few
lines later. Building everything first and wiring second has no such
ordering problem.

INTERNAL MODEL - a single ORDERED DICT, not a set+list pair. In Python
3.7+ a dict already IS both a fast-lookup set (via its keys) and a
stable-order sequence (iteration order = insertion order) in one
structure - `self.paths: dict[Path, None]` gives no-duplicates +
stable-ordering + O(1)-lookup. The dict key is the RESOLVED path
(Path.resolve()) - two different strings naming the same file on disk
collapse to one entry.

TREEVIEW, NOT A TEXTBOX. Selecting a row exposes actions via a
right-click context menu (currently just Remove) rather than a fixed
toolbar button, so a future action (Reveal in Explorer, Open,
Properties, Retry, Mark ignored) is one more menu entry, not a layout
change. Each row's iid IS the resolved path string.

DUPLICATE HANDLING - no popups. A single status line above the list
reports the outcome of the last Add File/Add Folder action; the running
total below the list is the persistent count.

STAGE ABSTRACTION - each pipeline stage this UI can launch is one
`Stage` object declaring its own command, whether it accepts --debug,
what device it runs on, AND (2026-07-30) its own log widgets - never an
`if stage_name == "classification"` special case scattered through the
launcher. Log ownership moved onto Stage itself once a SECOND
subprocess-running view (Preprocessing) existed - each view's log is
independent (only relevant while viewing that stage), so a single
shared self.log_text from the Classification-only design no longer
made sense.

  Stage(
      name=..., build_base_command=lambda: [...],
      supports_debug=bool, device=str,
      button=<widget>, status_var=..., progress_var=..., timing_var=...,
      log_toggle_btn=<widget>, log_frame=<widget>, log_text=<widget>,
      parse_progress=<optional per-line progress parser>,
  )

POWER USER CHECKBOX - a persistent, UI-wide setting, shown in the
always-visible header. `_build_command()` is the SINGLE place --debug
gets appended, gated on `stage.supports_debug`. core/classifier.py
gained a real --debug flag 2026-07-30 (prints each image's raw model
output to stdout before parsing - see that module's own argparse setup)
- Classification now declares supports_debug=True. scripts/run_
preprocessing.py still has no --debug flag, so Preprocessing stays
supports_debug=False.

GPU SAFETY GATE - a Stage whose `device != "CPU"` gets a best-effort
nvidia-smi check before launch (_check_gpu_busy()) - CLAUDE.md's own
written rule made into code. Excludes a short allowlist of known-benign
desktop/driver/UI processes, flags anything else holding more than
_GPU_SUSPICIOUS_MEMORY_MB. Does NOT hard-block - asks for CONFIRMATION
and proceeds/cancels on the answer.

DEVICE LABEL, START TIME, ELAPSED, ETA (added 2026-07-30, Jon's QOL
request - "something to add after this run" once the GPU was clear
again). `Stage.device` is a free-form label ("CPU", "GPU (RTX 5060
Ti)"), not a bool, per Jon's explicit refinement ("leaves room for
future flexibility... the label stays the same regardless of where the
computation runs") - `_detect_gpu_name()` queries `nvidia-smi
--query-gpu=name` once at wiring time for the real card name, falling
back to a bare "GPU" if that fails; this is a DISPLAY label only, never
a capability check. `stage.timing_var` (own line under progress, same
per-stage-widget pattern as status/progress/log) shows
"Device: ... · Started HH:MM:SS · Elapsed M:SS" while idle/running, plus
"· ETA ~M:SS remaining" once at least one progress line has arrived -
`_update_timing()` computes ETA as a straight-line
`(elapsed / done-so-far) * remaining-count` estimate, nothing fancier,
and is deliberately excluded on the frozen final display (`_run_stage`'s
"done"/"error" handling calls it with `include_eta=False`) so a stale
ETA doesn't sit next to "Done". Ticks every ~100ms poll cycle
regardless of log activity, so elapsed visibly counts up even between
log lines rather than jumping only when a new one arrives.

LIVE PROGRESS - `Stage.parse_progress`, given one line of streamed
stdout, returns (current, total, label) or None.
_parse_classifier_progress() matches core/classifier.py's own
`print(f"[{i}/{len(rows)}] {file_path}")` exactly.
_parse_preprocessing_progress() matches core/manifest_pipeline.py's
build_working_manifest_from_paths()'s own
`print(f"[{i}/{len(mapping)}] Preprocessing {working_path.name}...")`
exactly - both verified against the literal source lines, not guessed.
The same (current, total) feeds both the progress label AND the ETA
estimate above - `stage.last_progress` is set right alongside
`stage.progress_var`.

ROUTING SUMMARY - once Classification finishes successfully, shows how
many files landed in each of the 8 real DocumentCategory buckets, read
straight from core/classifier.py's own output CSVs. Shows ALL 8
buckets. A missing bucket CSV reads as 0, not an error. A FAILED run
neither refreshes nor reveals it. Re-running refreshes counts in place.

Usage:
    python ui/build_manifest_ui.py
"""

from __future__ import annotations

import csv
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tkinter import (
    Tk, Frame, Label, Button, Checkbutton, Menu, Text, StringVar, BooleanVar,
    filedialog, messagebox, END, DISABLED, NORMAL,
)
from tkinter import ttk

from core.manifest_pipeline import collect_image_paths, IMAGE_EXTENSIONS, DEFAULT_MANIFEST_PATH
from core.pdf_conversion import PDF_EXTENSIONS
from core.classifier import BUCKET_DIR
from core.schema import DocumentCategory

PYTHON = sys.executable  # same interpreter this GUI runs under
RUN_PREPROCESSING_SCRIPT = PROJECT_ROOT / "scripts" / "run_preprocessing.py"

# filedialog's filetypes format: (label, space-separated "*.ext" patterns).
# Only affects what the picker SHOWS by default for "Add File" - unlike
# "Add Folder" (a blind recursive scan that MUST filter, via
# collect_image_paths()), an explicit single-file pick is respected even
# outside this list if the user switches the dialog to "All files".
_IMAGE_FILETYPES = [
    ("Image and PDF files", " ".join(f"*{ext}" for ext in sorted(IMAGE_EXTENSIONS | PDF_EXTENSIONS))),
    ("Image files", " ".join(f"*{ext}" for ext in sorted(IMAGE_EXTENSIONS))),
    ("PDF files", " ".join(f"*{ext}" for ext in sorted(PDF_EXTENSIONS))),
    ("All files", "*.*"),
]

# -- GPU safety gate ---------------------------------------------------

# Known desktop/OS/driver-UI processes that routinely hold a little GPU
# memory but are never doing model inference - CLAUDE.md's own
# "background driver/OS processes" distinction, made concrete against
# real nvidia-smi output observed on this project's own dev machine.
_GPU_BENIGN_PROCESS_NAMES = {
    "explorer.exe", "dwm.exe", "shellhost.exe", "nvidia overlay.exe",
    "nvidia app.exe", "armourycrate.exe", "discord.exe",
    "phoneexperiencehost.exe", "crossdeviceresume.exe",
}
# A benign process below this holds a small amount of GPU memory; real
# model inference does not. Not calibrated against this project's own
# loaders specifically - a starting value, adjust if it proves noisy.
_GPU_SUSPICIOUS_MEMORY_MB = 500.0

# core/classifier.py's own progress line, unchanged: print(f"[{i}/{len(rows)}] {file_path}")
_CLASSIFIER_PROGRESS_RE = re.compile(r"^\[(\d+)/(\d+)\]\s+(.+?)\s*$")

# core/manifest_pipeline.py's build_working_manifest_from_paths()'s own
# progress line, unchanged: print(f"[{i}/{len(mapping)}] Preprocessing {working_path.name}...")
_PREPROCESSING_PROGRESS_RE = re.compile(r"^\[(\d+)/(\d+)\] Preprocessing (.+?)\.\.\.\s*$")


def _bucket_counts() -> dict[str, int]:
    """
    Row count of each real DocumentCategory's own bucket CSV
    (core/classifier.py's output). A bucket with no CSV yet reads as 0,
    not an error - same tolerance every other "maybe hasn't run yet"
    reader in this project already has. Iterates DocumentCategory in its
    own declared order (dense_tabular_rows first, uncertain_review
    last) - not re-sorted, since that order already reads sensibly.
    """
    counts: dict[str, int] = {}
    for category in DocumentCategory:
        path = BUCKET_DIR / f"{category.value}.csv"
        if not path.exists():
            counts[category.value] = 0
            continue
        with open(path, "r", encoding="utf-8", newline="") as f:
            counts[category.value] = sum(1 for _ in csv.DictReader(f))
    return counts


def _parse_classifier_progress(line: str) -> tuple[int, int, str] | None:
    match = _CLASSIFIER_PROGRESS_RE.match(line)
    if not match:
        return None
    current, total, file_path = match.groups()
    return int(current), int(total), Path(file_path).name


def _parse_preprocessing_progress(line: str) -> tuple[int, int, str] | None:
    match = _PREPROCESSING_PROGRESS_RE.match(line)
    if not match:
        return None
    current, total, filename = match.groups()
    return int(current), int(total), filename


def _detect_gpu_name() -> str:
    """
    Best-effort real GPU name for the device label (Jon, 2026-07-30:
    use "Device" rather than a hardcoded CPU/GPU binary, so the label
    reads e.g. "GPU (RTX 5060 Ti)", "GPU (CUDA:0)", "Apple MPS", "Remote
    API" without the UI itself needing to change shape later). This is a
    DISPLAY label only, not a capability check - failure here must never
    block anything, so it degrades to a bare "GPU" if nvidia-smi isn't
    available or returns nothing usable.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
            if lines:
                return f"GPU ({lines[0]})"
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return "GPU"


def _format_duration(seconds: float) -> str:
    """mm:ss, or h:mm:ss once an hour is crossed - used for both the
    elapsed-time and ETA displays."""
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _check_gpu_busy() -> list[str]:
    """
    Best-effort answer to "is something already using the GPU for real
    inference" - see module docstring's GPU SAFETY GATE section. Returns
    a list of "process_name (N MiB)" strings, empty if the GPU looks
    clear (or nvidia-smi can't be queried at all).
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=process_name,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []
    if result.returncode != 0:
        return []

    suspicious = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        name, mem_str = parts
        try:
            mem_mb = float(mem_str)
        except ValueError:
            continue
        if name.lower() in _GPU_BENIGN_PROCESS_NAMES:
            continue
        if mem_mb >= _GPU_SUSPICIOUS_MEMORY_MB:
            suspicious.append(f"{name} ({mem_mb:.0f} MiB)")
    return suspicious


@dataclass
class Stage:
    """One launchable pipeline stage - see module docstring's STAGE
    ABSTRACTION section. All widget fields are already-built real
    widgets by the time a Stage is constructed (see WIDGET CONSTRUCTION
    ORDER) - a Stage never owns widget CREATION, only references them.
    `log_expanded`/`start_monotonic`/`start_wall`/`last_progress` are
    per-stage mutable state - each view's log/timing tracks
    independently.

    `device` (2026-07-30, Jon's refinement) replaced a plain
    `uses_gpu: bool` - a string label ("CPU", "GPU (RTX 5060 Ti)",
    "Apple MPS", "Remote API", ...) names what's actually running the
    work instead of a binary, and doubles as the GPU-busy gate's
    condition (`device != "CPU"`) so there's one field instead of two
    that could drift apart if a non-GPU, non-CPU device ever shows up.
    """
    name: str
    build_base_command: Callable[[], list[str] | None]
    supports_debug: bool
    device: str
    button: Button
    status_var: StringVar
    progress_var: StringVar
    timing_var: StringVar
    log_toggle_btn: Button
    log_frame: Frame
    log_text: Text
    parse_progress: Callable[[str], tuple[int, int, str] | None] | None = None
    log_expanded: bool = False
    start_monotonic: float | None = None
    start_wall: str | None = None
    last_progress: tuple[int, int] | None = None


def _build_stage_status_block(
    parent: Frame, status_var: StringVar, progress_var: StringVar, timing_var: StringVar,
) -> None:
    """Shared status/progress/timing label layout - identical on the
    Preprocessing and Classification views, factored out so the two
    don't silently drift apart in appearance. `timing_var` (2026-07-30)
    shows device + start time + elapsed + (once progress exists) ETA -
    see _update_timing()."""
    status_row = Frame(parent)
    status_row.pack(anchor="w", padx=12, pady=(0, 2))
    Label(status_row, textvariable=status_var, font=("Segoe UI", 10, "bold"), fg="#444").pack(
        side="left")
    Label(parent, textvariable=progress_var, font=("Consolas", 9), fg="#555").pack(
        anchor="w", padx=12, pady=(0, 2))
    Label(parent, textvariable=timing_var, font=("Consolas", 9), fg="#777").pack(
        anchor="w", padx=12, pady=(0, 8))


def _build_stage_log_widgets(parent: Frame) -> tuple[Button, Frame, Text]:
    """
    Shared collapsible-log construction - returns (toggle_btn, log_frame,
    log_text), all built with no command wired to the toggle button yet
    (see module docstring's WIDGET CONSTRUCTION ORDER - _wire_stages()
    does that once the owning Stage exists to reference). log_frame is
    deliberately NOT packed here - every stage's log starts collapsed.
    """
    log_toggle_btn = Button(parent, text="Output ▶", anchor="w", relief="flat",
                             font=("Segoe UI", 9, "bold"))
    log_toggle_btn.pack(anchor="w", padx=12)

    log_frame = Frame(parent)
    log_text = Text(log_frame, height=12, font=("Consolas", 9), wrap="word", bg="#111", fg="#ddd")
    log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=log_text.yview)
    log_text.configure(yscrollcommand=log_scroll.set)
    log_text.pack(side="left", fill="both", expand=True)
    log_scroll.pack(side="right", fill="y")

    return log_toggle_btn, log_frame, log_text


class BuildManifestApp:
    def __init__(self, root: Tk):
        self.root = root
        root.title("Build Manifest")
        root.geometry("780x780")

        self.paths: dict[Path, None] = {}

        self.log_queue: queue.Queue = queue.Queue()
        self.process: subprocess.Popen | None = None
        self._active_stage: Stage | None = None
        self._active_frame: Frame | None = None
        self._run_classification_revealed = False
        self._routing_revealed = False
        self._preprocessing_input_csv: Path | None = None

        # -- Persistent header - session-wide, not scoped to any one view --
        Label(root, text="Build Manifest", font=("Segoe UI", 13, "bold")).pack(
            anchor="w", padx=12, pady=(12, 0))
        Label(
            root,
            text="Assembles a manifest, preprocesses it, then classifies it - the "
                 "pipeline stages that would otherwise be run by hand from the CLI. "
                 "A thin front end - the real modules do the work.",
            font=("Segoe UI", 9), fg="#444", wraplength=730, justify="left",
        ).pack(anchor="w", padx=12, pady=(2, 8))

        self.power_user_var = BooleanVar(value=False)
        Checkbutton(
            root, variable=self.power_user_var,
            text="Power User (pass --debug to downstream stages that support it)",
            font=("Segoe UI", 9),
        ).pack(anchor="w", padx=12, pady=(0, 8))

        # -- Swappable content area - shows exactly one view at a time --
        self.content_area = Frame(root)
        self.content_area.pack(fill="both", expand=True)

        # Phase 1: build every view's widgets (no Stage objects, no
        # button commands yet). Phase 2: wire Stage objects + commands
        # now that every real widget exists. See module docstring.
        self._build_manifest_view()
        self._build_preprocessing_view()
        self._build_classification_view()
        self._wire_stages()

        self._show_frame(self.manifest_frame)

    # -- view construction (phase 1: widgets only) -------------------------

    def _build_manifest_view(self) -> None:
        self.manifest_frame = Frame(self.content_area)

        button_row = Frame(self.manifest_frame)
        button_row.pack(anchor="w", padx=12, pady=(8, 4))
        Button(button_row, text="Add File", width=12, command=self._on_add_file).pack(
            side="left")
        Button(button_row, text="Add Folder", width=12, command=self._on_add_folder).pack(
            side="left", padx=(8, 0))

        self.status_var = StringVar(value="")
        Label(self.manifest_frame, textvariable=self.status_var,
              font=("Segoe UI", 9), fg="#256029").pack(anchor="w", padx=12, pady=(0, 6))

        Label(self.manifest_frame, text="Files queued for manifest",
              font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=12)

        list_frame = Frame(self.manifest_frame)
        list_frame.pack(fill="both", expand=True, padx=12, pady=(2, 6))

        self.tree = ttk.Treeview(
            list_frame, columns=("idx", "path"), show="headings", selectmode="extended",
        )
        self.tree.heading("idx", text="#")
        self.tree.heading("path", text="Path")
        self.tree.column("idx", width=50, anchor="e", stretch=False)
        self.tree.column("path", width=650, anchor="w", stretch=True)

        vscroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vscroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vscroll.pack(side="right", fill="y")

        # Right-click context menu - see module docstring for why this
        # (not a fixed toolbar) is the extension point for future
        # per-file actions.
        self.context_menu = Menu(self.tree, tearoff=0)
        self.context_menu.add_command(label="Remove", command=self._on_remove_selected)
        self.tree.bind("<Button-3>", self._on_right_click)
        self.tree.bind("<Delete>", lambda _e: self._on_remove_selected())

        self.count_var = StringVar(value="0 files")
        Label(self.manifest_frame, textvariable=self.count_var, font=("Segoe UI", 10)).pack(
            pady=(0, 4))

        # "Build Manifest ->" is the swap trigger to Preprocessing - see
        # module docstring's CANVAS SWAP MODEL. Command wired in
        # _wire_stages() once preprocessing_stage exists.
        self.build_manifest_btn = Button(
            self.manifest_frame, text="Build Manifest →", font=("Segoe UI", 10, "bold"),
            bg="#4a7", fg="white",
        )
        self.build_manifest_btn.pack(pady=(0, 12))

    def _build_preprocessing_view(self) -> None:
        self.preprocessing_frame = Frame(self.content_area)

        Label(self.preprocessing_frame, text="Preprocessing",
              font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=12, pady=(8, 0))
        Label(
            self.preprocessing_frame,
            text="Copies the queued files into data/working/, deskews and "
                 "preprocesses each copy (core/manifest_pipeline.py, Stage 0), and "
                 "writes the working manifest classification will run against. "
                 "CPU-only - no model, no GPU wait.",
            font=("Segoe UI", 9), fg="#444", wraplength=730, justify="left",
        ).pack(anchor="w", padx=12, pady=(2, 6))

        self.preprocessing_status_var = StringVar(value="Ready")
        self.preprocessing_progress_var = StringVar(value="")
        self.preprocessing_timing_var = StringVar(value="")
        _build_stage_status_block(
            self.preprocessing_frame, self.preprocessing_status_var,
            self.preprocessing_progress_var, self.preprocessing_timing_var)

        self.preprocessing_log_toggle_btn, self.preprocessing_log_frame, self.preprocessing_log_text = \
            _build_stage_log_widgets(self.preprocessing_frame)

        # "Run Classification ->" lives HERE - built now but not packed
        # until Preprocessing finishes successfully. Command wired in
        # _wire_stages() once classification_stage exists.
        self.run_classification_row = Frame(self.preprocessing_frame)
        self.run_classification_btn = Button(
            self.run_classification_row, text="Run Classification →",
            font=("Segoe UI", 10, "bold"), bg="#4a7", fg="white",
        )
        self.run_classification_btn.pack(side="left")

    def _build_classification_view(self) -> None:
        self.classification_frame = Frame(self.content_area)

        Label(self.classification_frame, text="Classification",
              font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=12, pady=(8, 0))
        Label(
            self.classification_frame,
            text="Runs python -m core.classifier against the working manifest - "
                 "Stage 2 of the pipeline (bucket routing). Loads a real model; may "
                 "take a while for a large manifest.",
            font=("Segoe UI", 9), fg="#444", wraplength=730, justify="left",
        ).pack(anchor="w", padx=12, pady=(2, 6))

        self.classification_status_var = StringVar(value="Ready")
        self.classification_progress_var = StringVar(value="")
        self.classification_timing_var = StringVar(value="")
        _build_stage_status_block(
            self.classification_frame, self.classification_status_var,
            self.classification_progress_var, self.classification_timing_var)

        self.classification_log_toggle_btn, self.classification_log_frame, self.classification_log_text = \
            _build_stage_log_widgets(self.classification_frame)

        # Routing Summary - built now but not packed; revealed IN PLACE
        # on this same view once Classification finishes successfully.
        # No "proceed to next stage" button yet - no defined next
        # workflow exists to point it at.
        self.routing_frame = Frame(self.classification_frame)

        Label(self.routing_frame, text="Routing Summary",
              font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=12, pady=(8, 0))
        Label(
            self.routing_frame,
            text="How classification sorted the manifest's files across buckets "
                 "(data/buckets/*.csv). Only dense_tabular_rows has a built "
                 "extraction workflow so far - the rest are counted, not yet actionable.",
            font=("Segoe UI", 9), fg="#444", wraplength=730, justify="left",
        ).pack(anchor="w", padx=12, pady=(2, 6))

        routing_grid = Frame(self.routing_frame)
        routing_grid.pack(anchor="w", padx=12, pady=(0, 4))

        self._routing_count_vars: dict[str, StringVar] = {}
        for row, category in enumerate(DocumentCategory):
            Label(routing_grid, text=category.value, font=("Consolas", 9)).grid(
                row=row, column=0, sticky="w", padx=(0, 24), pady=1)
            var = StringVar(value="0")
            self._routing_count_vars[category.value] = var
            Label(routing_grid, textvariable=var, font=("Consolas", 9)).grid(
                row=row, column=1, sticky="e", pady=1)

        total_row = len(DocumentCategory)
        Label(routing_grid, text="Total", font=("Consolas", 9, "bold")).grid(
            row=total_row, column=0, sticky="w", padx=(0, 24), pady=(4, 0))
        self._routing_total_var = StringVar(value="0")
        Label(routing_grid, textvariable=self._routing_total_var,
              font=("Consolas", 9, "bold")).grid(
            row=total_row, column=1, sticky="e", pady=(4, 0))

    # -- stage wiring (phase 2: Stage objects + button commands) ----------

    def _wire_stages(self) -> None:
        self.preprocessing_stage = Stage(
            name="Preprocessing",
            build_base_command=self._build_preprocessing_command,
            # scripts/run_preprocessing.py has no --debug flag today.
            supports_debug=False,
            device="CPU",  # deskew/autocontrast is pure CPU/PIL work
            button=self.build_manifest_btn,
            status_var=self.preprocessing_status_var,
            progress_var=self.preprocessing_progress_var,
            timing_var=self.preprocessing_timing_var,
            log_toggle_btn=self.preprocessing_log_toggle_btn,
            log_frame=self.preprocessing_log_frame,
            log_text=self.preprocessing_log_text,
            parse_progress=_parse_preprocessing_progress,
        )
        self.build_manifest_btn.config(command=self._on_click_build_manifest)
        self.preprocessing_log_toggle_btn.config(
            command=lambda: self._toggle_log(self.preprocessing_stage))
        self.preprocessing_timing_var.set(f"Device: {self.preprocessing_stage.device}")

        self.classification_stage = Stage(
            name="Classification",
            build_base_command=lambda: [PYTHON, "-m", "core.classifier", str(DEFAULT_MANIFEST_PATH)],
            # core/classifier.py gained --debug 2026-07-30 (see that
            # module's own argparse setup) - True is deliberate now,
            # not an oversight; Power User actually does something here.
            supports_debug=True,
            device=_detect_gpu_name(),  # loads Gemma - the GPU safety gate applies
            button=self.run_classification_btn,
            status_var=self.classification_status_var,
            progress_var=self.classification_progress_var,
            timing_var=self.classification_timing_var,
            log_toggle_btn=self.classification_log_toggle_btn,
            log_frame=self.classification_log_frame,
            log_text=self.classification_log_text,
            parse_progress=_parse_classifier_progress,
        )
        self.run_classification_btn.config(command=self._on_click_run_classification)
        self.classification_log_toggle_btn.config(
            command=lambda: self._toggle_log(self.classification_stage))
        self.classification_timing_var.set(f"Device: {self.classification_stage.device}")

    # -- canvas swap -------------------------------------------------------

    def _show_frame(self, frame: Frame) -> None:
        """The whole swap mechanism. Called once at startup (manifest
        view) and once more per stage launch (see _run_named_stage's
        on_launch)."""
        if self._active_frame is frame:
            return
        if self._active_frame is not None:
            self._active_frame.pack_forget()
        self._active_frame = frame
        frame.pack(fill="both", expand=True)

    # -- path collection -----------------------------------------------

    def _add_paths(self, new_paths: list[Path]) -> None:
        """
        The only place self.paths is mutated. Resolves each path (the
        real de-duplication key - see module docstring), skips anything
        already present WITHOUT moving its existing position, and
        reports the outcome as a status line rather than a popup.
        """
        added = 0
        skipped = 0
        for p in new_paths:
            resolved = p.resolve()
            if resolved in self.paths:
                skipped += 1
                continue
            self.paths[resolved] = None
            added += 1

        if added or skipped:
            self._refresh_tree()
        self._set_add_status(added, skipped)

    def _set_add_status(self, added: int, skipped: int) -> None:
        if added == 0 and skipped == 0:
            self.status_var.set("No image files found.")
        elif skipped == 0:
            self.status_var.set(f"{added} file{'s' if added != 1 else ''} added")
        elif added == 0:
            self.status_var.set(f"0 added ({skipped} already present)")
        else:
            self.status_var.set(f"{added} file{'s' if added != 1 else ''} added "
                                 f"({skipped} already present)")

    def _refresh_tree(self) -> None:
        self.tree.delete(*self.tree.get_children())
        for i, p in enumerate(self.paths, start=1):
            self.tree.insert("", END, iid=str(p), values=(i, str(p)))
        self.count_var.set(f"{len(self.paths)} file{'s' if len(self.paths) != 1 else ''}")

    # -- button/menu handlers -------------------------------------------

    def _on_add_file(self) -> None:
        picked = filedialog.askopenfilenames(
            title="Select image or PDF file(s)", filetypes=_IMAGE_FILETYPES,
        )
        if not picked:
            return
        self._add_paths([Path(p) for p in picked])

    def _on_add_folder(self) -> None:
        picked = filedialog.askdirectory(title="Select a folder of images (scanned recursively)")
        if not picked:
            return
        folder = Path(picked)
        # collect_image_paths() IS the folder-scan logic, unmodified from
        # core/manifest_pipeline.py - see module docstring.
        self._add_paths(collect_image_paths(folder))

    def _on_right_click(self, event) -> None:
        # Right-clicking a row that isn't already part of the selection
        # replaces the selection with just that row - the usual file-
        # manager convention.
        row_iid = self.tree.identify_row(event.y)
        if row_iid and row_iid not in self.tree.selection():
            self.tree.selection_set(row_iid)
        if self.tree.selection():
            self.context_menu.tk_popup(event.x_root, event.y_root)

    def _on_remove_selected(self) -> None:
        selected = self.tree.selection()
        if not selected:
            return
        for iid in selected:
            self.paths.pop(Path(iid), None)
        self._refresh_tree()
        self.status_var.set(f"Removed {len(selected)} file{'s' if len(selected) != 1 else ''}")

    def _build_preprocessing_command(self) -> list[str] | None:
        """
        scripts/run_preprocessing.py needs a "file_path" CSV as input
        (see that script's own docstring for why - the CSV-input adapter
        core/manifest_pipeline.py's engine was extracted to support).
        Writes the UI's current queued paths to a throwaway temp CSV -
        NOT data/manifest.csv, and NOT a durable pipeline artefact; it
        exists only to hand this one subprocess its input list, and is
        deleted once that subprocess finishes (see
        _on_preprocessing_done).
        """
        if not self.paths:
            return None
        fd, temp_path_str = tempfile.mkstemp(suffix=".csv", prefix="build_manifest_ui_selection_")
        temp_path = Path(temp_path_str)
        with open(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["file_path"])
            for p in self.paths:
                writer.writerow([str(p)])
        self._preprocessing_input_csv = temp_path
        return [PYTHON, str(RUN_PREPROCESSING_SCRIPT), str(temp_path)]

    def _on_click_build_manifest(self) -> None:
        if not self.paths:
            messagebox.showwarning("Build Manifest", "No files queued - add some first.")
            return
        self._run_named_stage(
            self.preprocessing_stage,
            on_launch=lambda: self._show_frame(self.preprocessing_frame),
            on_done=self._on_preprocessing_done,
        )

    def _on_preprocessing_done(self, exit_code: int) -> None:
        """Wired as Preprocessing's on_done. Cleans up the throwaway
        input CSV regardless of outcome (best-effort - its survival
        isn't load-bearing for anything). A failed run reveals nothing
        - there's no working manifest to classify."""
        if self._preprocessing_input_csv is not None:
            try:
                self._preprocessing_input_csv.unlink(missing_ok=True)
            except OSError:
                pass
            self._preprocessing_input_csv = None

        if exit_code != 0:
            return
        self._reveal_run_classification_button()

    def _reveal_run_classification_button(self) -> None:
        """Idempotent - a second successful Preprocessing run must not
        re-pack an already-visible button."""
        if self._run_classification_revealed:
            return
        self._run_classification_revealed = True
        self.run_classification_row.pack(anchor="w", padx=12, pady=(0, 12))

    def _on_click_run_classification(self) -> None:
        self._run_named_stage(
            self.classification_stage,
            on_launch=lambda: self._show_frame(self.classification_frame),
            on_done=self._on_classification_done,
        )

    def _on_classification_done(self, exit_code: int) -> None:
        """Wired as Classification's on_done - see _run_named_stage()/
        _poll_log_queue(). A failed run has nothing meaningful to
        summarize, so the section is neither refreshed nor revealed."""
        if exit_code != 0:
            return
        self._refresh_routing_summary()
        self._reveal_routing_summary_section()

    def _refresh_routing_summary(self) -> None:
        counts = _bucket_counts()
        for category_value, var in self._routing_count_vars.items():
            var.set(str(counts.get(category_value, 0)))
        self._routing_total_var.set(str(sum(counts.values())))

    def _reveal_routing_summary_section(self) -> None:
        """Idempotent - re-running Classification refreshes the COUNTS
        (see _on_classification_done) but must not re-pack the section a
        second time. Packed with no explicit ordering needed - it's a
        child of classification_frame, revealed after that view's other
        (already-packed) widgets, so it naturally lands at the end."""
        if self._routing_revealed:
            return
        self._routing_revealed = True
        self.routing_frame.pack(fill="both", expand=True)

    def _toggle_log(self, stage: Stage) -> None:
        self._set_log_expanded(stage, not stage.log_expanded)

    def _set_log_expanded(self, stage: Stage, expanded: bool) -> None:
        if expanded == stage.log_expanded:
            return
        stage.log_expanded = expanded
        if expanded:
            stage.log_toggle_btn.config(text="Output ▼")
            stage.log_frame.pack(fill="both", expand=True, padx=12, pady=(2, 12))
        else:
            stage.log_toggle_btn.config(text="Output ▶")
            stage.log_frame.pack_forget()

    # -- stage launcher --------------------------------------------------

    def _build_command(self, base: list[str], supports_debug: bool) -> list[str]:
        """
        Appends --debug to `base` iff Power User is checked AND this
        particular stage declares supports_debug=True. See module
        docstring's POWER USER CHECKBOX section for why this is the only
        place that decision is made.
        """
        cmd = list(base)
        if supports_debug and self.power_user_var.get():
            cmd.append("--debug")
        return cmd

    def _run_named_stage(
        self, stage: Stage,
        on_launch: Callable[[], None] | None = None,
        on_done: Callable[[int], None] | None = None,
    ) -> None:
        """
        Public entry point a stage's button command wires to. Handles
        the GPU confirmation gate, builds the final command (Power User
        flag included), and launches it. `on_launch` fires ONLY once the
        stage is actually about to run (after any GPU-busy confirmation
        is accepted) - this is what the canvas swap hooks into, so a
        declined confirmation leaves the caller's current view alone
        instead of stranding them on a view with no way to retry.
        `on_done(exit_code)` fires once the process exits.
        """
        if self._active_stage is not None:
            return  # a stage is already running - its own button is
                     # disabled, this is just a re-entrancy guard

        base = stage.build_base_command()
        if base is None:
            return

        if stage.device != "CPU":
            busy = _check_gpu_busy()
            if busy:
                proceed = messagebox.askyesno(
                    "GPU already busy",
                    "The GPU already appears to be running something:\n\n"
                    + "\n".join(f"  • {p}" for p in busy)
                    + "\n\nRunning two models on the GPU at once has already caused a "
                      "real, confirmed failure in this project (see CLAUDE.md).\n\n"
                      f"Run {stage.name} anyway?",
                )
                if not proceed:
                    stage.status_var.set("Cancelled (GPU busy)")
                    return

        if on_launch is not None:
            on_launch()

        cmd = self._build_command(base, stage.supports_debug)
        self._set_log_expanded(stage, True)
        self._run_stage(cmd, stage, on_done)

    def _run_stage(self, cmd: list[str], stage: Stage, on_done: Callable[[int], None] | None) -> None:
        self._active_stage = stage
        stage.button.config(state=DISABLED)
        stage.status_var.set("Running...")
        stage.progress_var.set("")
        stage.start_monotonic = time.monotonic()
        stage.start_wall = datetime.now().strftime("%H:%M:%S")
        stage.last_progress = None
        self._update_timing(stage)
        stage.log_text.delete("1.0", END)
        stage.log_text.insert(END, f"$ {' '.join(cmd)}\n\n")

        threading.Thread(target=self._stage_worker, args=(cmd,), daemon=True).start()
        self.root.after(100, lambda: self._poll_log_queue(stage, on_done))

    def _update_timing(self, stage: Stage, include_eta: bool = True) -> None:
        """
        Refreshes stage.timing_var: "Device: ... · Started HH:MM:SS ·
        Elapsed M:SS" and, once at least one progress line has been
        seen, "· ETA ~M:SS remaining" - a straight-line estimate from
        (elapsed / done-so-far) * remaining-count, nothing fancier.
        Called every poll tick while a stage runs (so elapsed ticks up
        smoothly even between log lines) and once more on completion
        with include_eta=False, so the frozen final line doesn't show a
        stale "ETA" next to "Done".
        """
        if stage.start_monotonic is None:
            return
        elapsed = time.monotonic() - stage.start_monotonic
        text = (f"Device: {stage.device}  ·  Started {stage.start_wall}  ·  "
                f"Elapsed {_format_duration(elapsed)}")
        if include_eta and stage.last_progress is not None:
            current, total = stage.last_progress
            if current > 0 and elapsed > 0:
                remaining = max(0.0, (total - current) * (elapsed / current))
                text += f"  ·  ETA {_format_duration(remaining)} remaining"
        stage.timing_var.set(text)

    def _stage_worker(self, cmd: list[str]) -> None:
        try:
            # encoding/errors explicit here, matching debug_tools/
            # workflow_gui.py's CommandTab fix (2026-07-16):
            # Popen(text=True) otherwise falls back to the OS locale's
            # default encoding (cp1252 on Windows), which cannot decode
            # the UTF-8 this pipeline's scripts explicitly write to
            # their own stdout.
            #
            # PYTHONUNBUFFERED=1 (added 2026-07-30, found while testing
            # the timing/ETA feature) - bufsize=1 above only controls how
            # THIS process reads the pipe; it does nothing about whether
            # the CHILD process buffers before writing. CPython
            # block-buffers stdout by default whenever it isn't a real
            # TTY, which a subprocess pipe never is - without this, a
            # child script's print() calls (core/classifier.py's,
            # core/manifest_pipeline.py's) could sit unflushed until the
            # OS pipe buffer fills or the process exits, silently
            # defeating live progress/ETA (they'd all arrive in one
            # burst at the end instead of streaming per line).
            env = dict(os.environ, PYTHONUNBUFFERED="1")
            self.process = subprocess.Popen(
                cmd, cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
                encoding="utf-8", errors="replace", env=env,
            )
            for line in self.process.stdout:
                self.log_queue.put(("line", line))
            exit_code = self.process.wait()
            self.log_queue.put(("done", exit_code))
        except Exception as e:
            self.log_queue.put(("error", str(e)))

    def _poll_log_queue(self, stage: Stage, on_done: Callable[[int], None] | None) -> None:
        self._update_timing(stage)  # every ~100ms tick, so elapsed visibly
                                     # ticks up even between log lines
        try:
            while True:
                kind, payload = self.log_queue.get_nowait()
                if kind == "line":
                    stage.log_text.insert(END, payload)
                    stage.log_text.see(END)
                    if stage.parse_progress is not None:
                        progress = stage.parse_progress(payload.rstrip("\n"))
                        if progress is not None:
                            current, total, label = progress
                            stage.progress_var.set(f"{current} / {total}  —  {label}")
                            stage.last_progress = (current, total)
                            self._update_timing(stage)
                elif kind == "done":
                    stage.log_text.insert(END, f"\n[exit code {payload}]\n")
                    stage.log_text.see(END)
                    stage.button.config(state=NORMAL)
                    stage.status_var.set("Done" if payload == 0 else f"Failed (exit {payload})")
                    self._update_timing(stage, include_eta=False)
                    self._active_stage = None
                    if on_done is not None:
                        on_done(payload)
                    return
                elif kind == "error":
                    stage.log_text.insert(END, f"\n[ERROR launching process: {payload}]\n")
                    stage.button.config(state=NORMAL)
                    stage.status_var.set("Error")
                    self._update_timing(stage, include_eta=False)
                    self._active_stage = None
                    return
        except queue.Empty:
            pass
        self.root.after(100, lambda: self._poll_log_queue(stage, on_done))


def main() -> None:
    root = Tk()
    BuildManifestApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
