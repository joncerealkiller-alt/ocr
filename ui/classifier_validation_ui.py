"""
Classifier Validation UI - a developer QA tool for rapidly reviewing
Gemma's classification routing decisions after a batch run. NOT part of
the production pipeline: never reruns classification, never writes
anything back to a bucket CSV - the one write this tool does is
appending to a separate flat triage log (see "FLAG MISCLASSIFIED"
below), not a bucket CSV. Built 2026-07-30 per Jon's spec.

BUCKETS COME FROM CONFIG, NOT CODE - Jon's explicit correction before
this was built: "stopped you before we hardcode in the buckets. they
should be stored in a config file so if we add a bucket later its a
config change rather than code rewrite." The bucket list is read from
`config/pipeline.yaml`'s own `buckets:` mapping via
`core/classifier.py`'s `load_pipeline_config()` (reused directly, not
re-parsed) - the SAME config that already decides which model/prompt
each bucket uses. This file never hardcodes `DocumentCategory` or any
bucket name string; adding a bucket later is a `pipeline.yaml` edit,
exactly like it already is for the rest of the pipeline. PyYAML/Python
3.7+ dicts preserve mapping order, so buckets display in the same order
`pipeline.yaml` lists them (dense_tabular_rows first, uncertain_review
last) with no re-sorting needed.

ARCHITECTURE - "follow the existing launcher style... do not duplicate
pipeline code... read existing bucket CSVs only... do not rerun
classification" (Jon's spec). This tool imports `load_pipeline_config`
and `BUCKET_DIR` from `core/classifier.py` (the single source of truth
for where bucket CSVs live and what buckets exist) but calls no
model/loader code whatsoever - there is no "engine" here to avoid
duplicating beyond those two reused constants. Bucket CSVs are read via
plain `csv.DictReader`, which is schema-agnostic - `dense_tabular_rows.
csv` (11 columns) and `uncertain_review.csv` (12, the extra `error`
column) are read identically, with no column list re-declared from
`core/classifier.py`'s own `CSV_FIELDS`/`UNCERTAIN_FIELDS` to drift out
of sync with.

Per this project's established "every module is standalone by design"
principle (see e.g. `ui/dewarp_preprocessor_ui.py`'s own docstring),
this tool does NOT import from `ui/build_manifest_ui.py` even though
both do their own small amount of bucket-CSV counting - two real,
independent, genuinely different uses (one drives a pipeline launcher,
this one drives a review queue) don't yet justify a shared module for a
few lines of CSV counting.

LAZY IMAGE LOADING - only the currently-displayed image is ever decoded;
navigating a 71-image bucket does not load 71 images into memory up
front. A missing/unreadable file shows an error placeholder in the image
area rather than crashing the tool - reviewing hundreds of images should
survive one bad path.

SESSION STATS ("Reviewed: N / M") - Jon's addition to the spec, built
now (not deferred with the other future-features list) since it's
genuinely useful today even without Correct/Incorrect buttons: tracks
which row INDICES have been visited per bucket this session
(`self._visited: dict[str, set[int]]`), reset per bucket (each bucket
tracks its own progress independently, preserved if you switch away and
back). Deliberately the natural precursor to the validation-buttons
version Jon described ("Correct 21 / Incorrect 2 / Remaining 48 /
Accuracy 91.3%") - swapping "visited" for "has a recorded verdict" later
is the only change needed, not a redesign.

RESERVED SPACE FOR FUTURE FEATURES (Jon's spec: "do NOT implement yet") -
Correct/Incorrect buttons, Reassign bucket, Notes, Export corrections,
validation keyboard shortcuts, confidence colouring, confidence/prompt-
version filters. `self.future_frame` is a real Frame sitting in the
correct position in the layout (between the metadata panel and the nav
row), with those features still deferred and this remains their
ready-made home.

THREE FLAG BUTTONS in `future_frame`, deliberately NOT the deferred
verdict-button system above - all three are plain triage logs, not a
verdict system, appended-to via the shared `_flag_current()`/
`_load_flagged_file_paths()`/`_append_flagged_row()` helpers, each
deduplicated on `file_path` so re-flagging the same image (including
across bucket switches) is a no-op:

  - "Flag Misclassified" (added 2026-07-31, Jon's explicit request) ->
    `data/misclassifications.csv` (bucket, file_path, category,
    confidence, reason, model, prompt_version). Point: build up a
    representative sample of real misclassifications - taxonomy gaps,
    prompt problems, CV/layout problems, genuinely ambiguous images -
    before designing any fix for them.

  - "Mark Bad Deskew" (added 2026-07-31, Jon's finding: some images are
    getting deskewed when they didn't need it, coming out MORE skewed
    than the source) -> `data/flagged_bad_deskew.csv` (bucket,
    file_path, category, reason). Point: isolate a batch of real
    examples to re-run through CV analysis with deskew skipped from the
    original, for comparison - a preprocessing bug, not a classification
    one, tracked here only because this is where images are already
    being looked at one at a time.

  - "Mark for Pruning" (added 2026-07-31) -> `data/flagged_for_pruning.
    csv` (bucket, file_path, category, reason). Point: images to be
    dropped from the corpus entirely - a separate list from bad-deskew,
    since "wrongly deskewed" and "not worth keeping at all" are
    different actions on the same data.

  - "Needs New Bucket" (added 2026-07-31, Jon: "not that we may need it,
    but incase we do... better to have and not need it than get 1000
    images in and find we do need it and have to redo from the start
    again") -> `data/flagged_needs_new_bucket.csv` (bucket, file_path,
    category, reason). Point: a running list of images that don't fit
    any existing bucket well, for the same later taxonomy-design pass as
    misclassifications.csv - not acted on now.

Usage:
    python ui/classifier_validation_ui.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tkinter import Tk, Frame, Label, Button, StringVar, DISABLED, NORMAL
from tkinter import ttk

from PIL import Image, ImageTk

from core.classifier import load_pipeline_config, BUCKET_DIR

# Persistent, cross-run triage/ground-truth logs - deliberately anchored
# to the fixed data/ directory, NOT derived from BUCKET_DIR.parent.
# BUCKET_DIR now resolves into whichever run produced the current bucket
# CSVs (see core/classifier.py's dynamic legacy-run fallback), but these
# four logs are NOT run-owned - misclassifications.csv is explicitly "a
# flat ground-truth sample, not a queue" (see core/review_modes.py's own
# docstring), and the other three are "running lists" meant to persist
# across many runs for a later taxonomy-design pass. Deriving them from
# BUCKET_DIR.parent would have silently pointed them at
# <run>/outputs/ instead of data/ once BUCKET_DIR started resolving into
# a run directory - see docs/RUN_ARCHITECTURE.md's ownership model.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"

MISCLASSIFICATION_LOG = _DATA_DIR / "misclassifications.csv"
MISCLASSIFICATION_FIELDS = [
    "bucket", "file_path", "category", "confidence", "reason",
    "model", "prompt_version",
]

# Same flat-log-with-dedup pattern as MISCLASSIFICATION_LOG above, added
# 2026-07-31 per Jon's finding: some images are getting deskewed when
# they didn't need it, making them MORE skewed than the source - a
# preprocessing bug, not a classification one. These two logs exist so
# he can isolate a batch of real examples (same "collect the sample
# first" approach as the misclassification triage) rather than guessing
# at a fix blind:
#   - BAD_DESKEW_LOG: images to re-run through CV analysis/deskew with
#     the deskew step SKIPPED from the original, to compare against.
#   - PRUNE_LOG: images to be dropped from the corpus entirely (e.g.
#     unusable scans) - a separate list, since "skewed wrongly" and
#     "not worth keeping at all" are different actions on the same data.
BAD_DESKEW_LOG = _DATA_DIR / "flagged_bad_deskew.csv"
PRUNE_LOG = _DATA_DIR / "flagged_for_pruning.csv"
# Added 2026-07-31 alongside the two above, per Jon: "not that we may
# need it, but incase we do. better to have and not need it than get
# 1000 images in and find we do need it and have to redo from the start
# again." Same pattern - a plain triage log of images that don't fit any
# existing bucket well (the non-portrait-photo gap, modern-UI-screenshot
# gap, etc. already surfaced during review), collected for the same
# later taxonomy-design pass as misclassifications.csv, not acted on now.
NEEDS_BUCKET_LOG = _DATA_DIR / "flagged_needs_new_bucket.csv"
# Shared field list - these two logs only ever need "which bucket/row was
# this seen from", not the full classification metadata the
# misclassification log tracks, so they're intentionally smaller.
FLAG_LOG_FIELDS = ["bucket", "file_path", "category", "reason"]

# Metadata fields shown per Jon's spec, in that order. Read via .get() on
# each row dict, so a bucket CSV missing a column (or the extra `error`
# column uncertain_review.csv alone has) never raises - just shows blank.
_METADATA_FIELDS = [
    ("Category", "category"),
    ("Confidence", "confidence"),
    ("Reason", "reason"),
    ("Model", "model"),
    ("Prompt Version", "prompt_version"),
]


def _bucket_names() -> list[str]:
    """The ordered bucket list, straight from config/pipeline.yaml's own
    buckets: mapping - see module docstring for why this is never
    hardcoded here."""
    pipeline_cfg = load_pipeline_config()
    return list(pipeline_cfg["buckets"].keys())


def _load_flagged_file_paths(log_path: Path) -> set[str]:
    """file_paths already present in the given flat log, so a flag
    button can skip appending a duplicate. Keyed on file_path alone (not
    bucket+file_path) - a given image only needs flagging once regardless
    of which bucket it was reviewed from. Shared by all three flag logs
    (misclassification, bad-deskew, prune) - they're all the same shape
    of "append once, dedupe on file_path" log."""
    if not log_path.exists():
        return set()
    with open(log_path, "r", encoding="utf-8", newline="") as f:
        return {row["file_path"] for row in csv.DictReader(f)}


def _append_flagged_row(log_path: Path, fields: list[str], row: dict) -> None:
    is_new = not log_path.exists()
    with open(log_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def _load_bucket_rows(bucket_name: str) -> list[dict]:
    """
    Every row of data/buckets/<bucket_name>.csv, in file order. Missing
    file (that bucket never got any images, or classification hasn't
    run yet) returns [] - same "maybe hasn't run yet" tolerance every
    other bucket-CSV reader in this project already has, not an error.
    """
    path = BUCKET_DIR / f"{bucket_name}.csv"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


class ClassifierValidationApp:
    def __init__(self, root: Tk):
        self.root = root
        root.title("Classifier Validation")
        root.geometry("1000x900")
        root.minsize(700, 600)
        # Hard cap, ENFORCED by Tk regardless of what any child widget's
        # content demands - added 2026-07-30 after a real bug: the image
        # preview grew to fill the whole screen on open, pushing the
        # metadata panel and nav buttons off-screen. geometry()/minsize()
        # alone don't prevent a Toplevel from growing past its requested
        # size if a child's natural/requested size (e.g. a Label sized to
        # a several-thousand-pixel source image, however that happened)
        # exceeds it - maxsize() is the one constraint Tk actually
        # refuses to violate, so this is a real ceiling, not a request.
        root.update_idletasks()
        screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.maxsize(max(1000, screen_w - 80), max(900, screen_h - 80))

        self.rows: list[dict] = []
        self.index: int = 0
        self._visited: dict[str, set[int]] = {}
        self._tk_image: ImageTk.PhotoImage | None = None
        self._current_bucket: str | None = None
        self._flagged_paths: set[str] = _load_flagged_file_paths(MISCLASSIFICATION_LOG)
        self._bad_deskew_paths: set[str] = _load_flagged_file_paths(BAD_DESKEW_LOG)
        self._prune_paths: set[str] = _load_flagged_file_paths(PRUNE_LOG)
        self._needs_bucket_paths: set[str] = _load_flagged_file_paths(NEEDS_BUCKET_LOG)

        # -- Header: bucket selector + session stats --
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

        self.reviewed_var = StringVar(value="Reviewed: 0 / 0")
        Label(header, textvariable=self.reviewed_var, font=("Segoe UI", 9), fg="#444").grid(
            row=0, column=2, sticky="e", padx=(24, 0))
        header.columnconfigure(2, weight=1)

        self.position_var = StringVar(value="Image 0 / 0")
        Label(header, textvariable=self.position_var, font=("Segoe UI", 10, "bold")).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

        # -- Image preview - the majority of the window, per spec --
        self.image_frame = Frame(root, bg="#111")
        self.image_frame.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        # ROOT CAUSE FIX (2026-07-30, found from a screen recording of the
        # actual bug - the window grew CONTINUOUSLY over several seconds,
        # not in one jump, which is the signature of a feedback loop, not
        # a one-off bad reading): pack_propagate(False) makes image_frame's
        # size depend ONLY on what ITS OWN pack() call (fill=both,
        # expand=True inside root) gives it - NEVER on its child's
        # (image_label's) content. Without this, the loop is: render an
        # image sized to fit image_frame -> the Label's own natural size
        # tracks that image -> if it's even marginally larger than the
        # frame's true allocated space (LANCZOS/thumbnail rounding is
        # enough), the frame grows to fit the Label -> a NEW <Configure>
        # fires with a BIGGER box -> re-render bigger -> repeat, each lap
        # growing the window further with no built-in floor to stop it.
        # This breaks the loop at its source rather than only capping the
        # damage (root.maxsize() / the box clamp in _render_image() stay
        # in place too, as real defense in depth, not because they were
        # wrong - they just weren't the actual cause).
        self.image_frame.pack_propagate(False)
        self.image_label = Label(self.image_frame, bg="#111", fg="#888",
                                  font=("Segoe UI", 10))
        self.image_label.pack(fill="both", expand=True)
        # Re-fit the current image whenever the available area actually
        # changes size (window resize) - not on every Configure event
        # regardless of size, to avoid redundant re-decoding.
        self._last_render_box: tuple[int, int] | None = None
        self.image_frame.bind("<Configure>", self._on_image_frame_resize)

        # -- Metadata panel --
        meta_frame = Frame(root, padx=12)
        meta_frame.pack(fill="x")
        Label(meta_frame, text="Gemma Decision", font=("Segoe UI", 11, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))

        self._metadata_vars: dict[str, StringVar] = {}
        for i, (label_text, _field) in enumerate(_METADATA_FIELDS, start=1):
            Label(meta_frame, text=f"{label_text}:", font=("Segoe UI", 9, "bold"),
                  anchor="nw").grid(row=i, column=0, sticky="nw", pady=1, padx=(0, 8))
            var = StringVar(value="")
            self._metadata_vars[label_text] = var
            Label(meta_frame, textvariable=var, font=("Segoe UI", 9), fg="#333",
                  anchor="nw", justify="left", wraplength=900).grid(
                row=i, column=1, sticky="nw", pady=1)

        filename_row = len(_METADATA_FIELDS) + 1
        Label(meta_frame, text="Filename:", font=("Segoe UI", 9, "bold"), anchor="nw").grid(
            row=filename_row, column=0, sticky="nw", pady=1, padx=(0, 8))
        self.filename_var = StringVar(value="")
        Label(meta_frame, textvariable=self.filename_var, font=("Consolas", 9), fg="#333",
              anchor="nw", justify="left", wraplength=900).grid(
            row=filename_row, column=1, sticky="nw", pady=1)

        # Reserved for future validation controls (Correct/Incorrect,
        # Reassign, Notes, Export) - see module docstring. Real, empty,
        # correctly-positioned Frame; deliberately not populated yet.
        # The "Flag Misclassified" button below is an exception, added
        # 2026-07-31 per Jon's explicit request - it lives in this frame
        # (not the nav row) so it stays visually separate from
        # Previous/Next, but it is intentionally NOT one of the deferred
        # Correct/Incorrect verdict buttons: it just appends the current
        # row to a standalone flat log (misclassifications.csv) for later
        # triage into taxonomy/prompt/CV/ambiguous buckets, per Jon's plan
        # to collect a representative sample before designing any fix.
        self.future_frame = Frame(root, padx=12)
        self.future_frame.pack(fill="x")
        self.flag_btn = Button(self.future_frame, text="Flag Misclassified",
                                command=self._on_flag_misclassified)
        self.flag_btn.pack(side="left")
        self.flag_status_var = StringVar(value="")
        Label(self.future_frame, textvariable=self.flag_status_var,
              font=("Segoe UI", 9), fg="#a00").pack(side="left", padx=(10, 18))

        # Two more flag buttons, added 2026-07-31 per Jon's finding that
        # some preprocessing deskew is making images WORSE, not better -
        # see BAD_DESKEW_LOG/PRUNE_LOG comment above for what each is for.
        self.bad_deskew_btn = Button(self.future_frame, text="Mark Bad Deskew",
                                      command=self._on_flag_bad_deskew)
        self.bad_deskew_btn.pack(side="left")
        self.bad_deskew_status_var = StringVar(value="")
        Label(self.future_frame, textvariable=self.bad_deskew_status_var,
              font=("Segoe UI", 9), fg="#a00").pack(side="left", padx=(10, 18))

        self.prune_btn = Button(self.future_frame, text="Mark for Pruning",
                                 command=self._on_flag_prune)
        self.prune_btn.pack(side="left")
        self.prune_status_var = StringVar(value="")
        Label(self.future_frame, textvariable=self.prune_status_var,
              font=("Segoe UI", 9), fg="#a00").pack(side="left", padx=(10, 18))

        self.needs_bucket_btn = Button(self.future_frame, text="Needs New Bucket",
                                        command=self._on_flag_needs_bucket)
        self.needs_bucket_btn.pack(side="left")
        self.needs_bucket_status_var = StringVar(value="")
        Label(self.future_frame, textvariable=self.needs_bucket_status_var,
              font=("Segoe UI", 9), fg="#a00").pack(side="left", padx=(10, 0))

        # -- Navigation --
        nav = Frame(root, padx=12, pady=10)
        nav.pack(fill="x")
        self.prev_btn = Button(nav, text="◀ Previous", width=12, command=self._on_previous)
        self.prev_btn.pack(side="left")
        self.next_btn = Button(nav, text="Next ▶", width=12, command=self._on_next)
        self.next_btn.pack(side="left", padx=(8, 0))

        root.bind("<Left>", lambda _e: self._on_previous())
        root.bind("<Right>", lambda _e: self._on_next())

        # Force one real layout pass NOW, before the first image render -
        # without this, image_frame.winfo_width()/height() in
        # _render_image() reports Tk's "1x1, never been drawn"
        # placeholder on the very first call (confirmed directly:
        # winfo_width()/height() stay at 1 until every sibling is packed
        # AND a layout pass has run). update_idletasks() processes
        # pending geometry work immediately - no visible redraw, no wait
        # for mainloop.
        root.update_idletasks()

        self._populate_bucket_list()

    # -- bucket list / selection ------------------------------------------

    def _populate_bucket_list(self) -> None:
        display_values = []
        for name in _bucket_names():
            count = len(_load_bucket_rows(name))
            display = f"{name} ({count})"
            self._bucket_display_to_name[display] = name
            display_values.append(display)

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
        self.rows = _load_bucket_rows(bucket_name)
        self.index = 0  # per spec: changing bucket resets to image 1
        self._visited.setdefault(bucket_name, set())
        self._show_current()

    # -- navigation ---------------------------------------------------------

    def _on_previous(self) -> None:
        if self.index > 0:
            self.index -= 1
            self._show_current()

    def _on_next(self) -> None:
        if self.index < len(self.rows) - 1:
            self.index += 1
            self._show_current()

    # -- display --------------------------------------------------------

    def _show_current(self) -> None:
        total = len(self.rows)
        if total == 0:
            self.position_var.set("Image 0 / 0")
            self._set_metadata(None)
            self._render_image(None)
            self.prev_btn.config(state=DISABLED)
            self.next_btn.config(state=DISABLED)
            self._update_reviewed_label()
            return

        self.position_var.set(f"Image {self.index + 1} / {total}")
        row = self.rows[self.index]
        self._set_metadata(row)
        self._render_image(row.get("file_path"))

        self.prev_btn.config(state=NORMAL if self.index > 0 else DISABLED)
        self.next_btn.config(state=NORMAL if self.index < total - 1 else DISABLED)

        if self._current_bucket is not None:
            self._visited[self._current_bucket].add(self.index)
        self._update_reviewed_label()
        self._update_flag_status()

    def _update_flag_status(self) -> None:
        if not self.rows:
            self.flag_status_var.set("")
            self.bad_deskew_status_var.set("")
            self.prune_status_var.set("")
            self.needs_bucket_status_var.set("")
            return
        file_path = self.rows[self.index].get("file_path", "")
        self.flag_status_var.set("✓ Flagged" if file_path in self._flagged_paths else "")
        self.bad_deskew_status_var.set("✓ Flagged" if file_path in self._bad_deskew_paths else "")
        self.prune_status_var.set("✓ Flagged" if file_path in self._prune_paths else "")
        self.needs_bucket_status_var.set("✓ Flagged" if file_path in self._needs_bucket_paths else "")

    def _flag_current(self, log_path: Path, fields: list[str], flagged_paths: set[str],
                       status_var: StringVar, extra_fields: dict | None = None) -> None:
        """
        Shared body for all three flag buttons - appends the current
        row to `log_path` (deduplicated on file_path) and updates
        `status_var`. `extra_fields` lets the misclassification log carry
        its extra columns (confidence/model/prompt_version) that the
        bad-deskew/prune logs don't need.
        """
        if not self.rows or self._current_bucket is None:
            return
        row = self.rows[self.index]
        file_path = row.get("file_path", "")
        if not file_path:
            return
        if file_path in flagged_paths:
            status_var.set("✓ Flagged (already logged)")
            return
        out_row = {
            "bucket": self._current_bucket,
            "file_path": file_path,
            "category": row.get("category", ""),
            "reason": row.get("reason", ""),
        }
        out_row.update(extra_fields or {})
        _append_flagged_row(log_path, fields, out_row)
        flagged_paths.add(file_path)
        status_var.set("✓ Flagged")

    def _on_flag_misclassified(self) -> None:
        self._flag_current(
            MISCLASSIFICATION_LOG, MISCLASSIFICATION_FIELDS, self._flagged_paths,
            self.flag_status_var,
            extra_fields={
                "confidence": self.rows[self.index].get("confidence", "") if self.rows else "",
                "model": self.rows[self.index].get("model", "") if self.rows else "",
                "prompt_version": self.rows[self.index].get("prompt_version", "") if self.rows else "",
            },
        )

    def _on_flag_bad_deskew(self) -> None:
        self._flag_current(BAD_DESKEW_LOG, FLAG_LOG_FIELDS, self._bad_deskew_paths,
                            self.bad_deskew_status_var)

    def _on_flag_prune(self) -> None:
        self._flag_current(PRUNE_LOG, FLAG_LOG_FIELDS, self._prune_paths,
                            self.prune_status_var)

    def _on_flag_needs_bucket(self) -> None:
        self._flag_current(NEEDS_BUCKET_LOG, FLAG_LOG_FIELDS, self._needs_bucket_paths,
                            self.needs_bucket_status_var)

    def _update_reviewed_label(self) -> None:
        if self._current_bucket is None:
            self.reviewed_var.set("Reviewed: 0 / 0")
            return
        visited = len(self._visited.get(self._current_bucket, ()))
        self.reviewed_var.set(f"Reviewed: {visited} / {len(self.rows)}")

    def _set_metadata(self, row: dict | None) -> None:
        for label_text, field in _METADATA_FIELDS:
            self._metadata_vars[label_text].set((row or {}).get(field, "") or "")
        file_path = (row or {}).get("file_path", "")
        self.filename_var.set(Path(file_path).name if file_path else "")

    # -- image rendering --------------------------------------------------

    def _on_image_frame_resize(self, event) -> None:
        box = (event.width, event.height)
        if box == self._last_render_box:
            return
        self._last_render_box = box
        if self.rows:
            self._render_image(self.rows[self.index].get("file_path"))

    def _render_image(self, file_path: str | None) -> None:
        if not file_path:
            self._tk_image = None
            self.image_label.config(image="", text="No images in this bucket")
            return

        # Clamped both directions, not just floored - a widget that
        # hasn't been through a real layout pass yet can report Tk's
        # "1x1, never drawn" placeholder (too small) OR, if this is ever
        # reached before root.maxsize() has taken effect or from some
        # other unanticipated timing, an unbounded value derived from
        # native screen resolution (too large) - this is the same real
        # bug's second line of defense, see root.maxsize() in __init__
        # for the first. Falls back to a fixed, sane box in either case
        # rather than trusting winfo_width()/height() blindly.
        raw_w, raw_h = self.image_frame.winfo_width(), self.image_frame.winfo_height()
        screen_w, screen_h = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        if 50 <= raw_w <= screen_w and 50 <= raw_h <= screen_h:
            box_w, box_h = raw_w, raw_h
        else:
            box_w, box_h = 900, 650

        try:
            with Image.open(file_path) as im:
                preview = im.convert("RGB")
                # thumbnail() scales down to fit within the box, preserves
                # aspect ratio, never crops, and never upscales past the
                # source's own resolution - matches the spec exactly.
                preview.thumbnail((box_w, box_h), Image.LANCZOS)
        except Exception as e:
            self._tk_image = None
            self.image_label.config(
                image="", text=f"Could not load image:\n{file_path}\n\n{type(e).__name__}: {e}")
            return

        self._tk_image = ImageTk.PhotoImage(preview)
        self.image_label.config(image=self._tk_image, text="")


def main() -> None:
    root = Tk()
    ClassifierValidationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
