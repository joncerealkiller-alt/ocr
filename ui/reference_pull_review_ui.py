"""
Manual triage UI for scripts/pull_reference_images.py's output
(data/outputs/reference_pull/<category>/*.jpg + manifest.csv) - built
2026-08-08 per Jon's explicit request: he already has ui/classifier_
validation_ui.py, but that tool exists to check GEMMA'S predictions
against real bucket CSVs (read-only, never mutates anything) - this is
a genuinely different job: these are raw pulled reference images with
NO classifier prediction attached at all, and Jon needs to actually
KEEP, REJECT (permanently delete), or RECLASSIFY (move to a different
category) each one - real, deliberate mutations, not flags.

Deliberately does NOT touch ui/classifier_validation_ui.py or any bucket
CSV - this operates ONLY on data/outputs/reference_pull/, a curation
dataset Jon is actively building, never on data/buckets/ or any
production sidecar/manifest.

REJECT IS A REAL, PERMANENT DELETE (unlike every other flag/mark
mechanism built so far in this project, which are all non-destructive
logs) - deliberate, since Jon's own words were "reject completely" and
this is curation-stage data he's actively pruning, not production
output. Confirmed via a dialog before every delete, same safety gate
debug_tools/review_uncertain.py already uses for its own real bucket-CSV
mutations.

RECLASSIFY moves the file to the target category's own folder and
transplants its manifest row (category field updated to match) - never
copies, so an image only ever lives in one category's manifest/folder at
a time.

MOVE-TARGET LIST IS THE FULL 11-CATEGORY TAXONOMY (added 2026-08-08, per
Jon: a pull only targets a handful of categories at a time, but a wrongly
routed image found during review can genuinely belong to ANY of the
project's real categories - e.g. a portrait pulled under a casual_photo
search query - and discarding still-useful training data just because
this particular pull run didn't target that category "doesn't make
sense"). Reuses core.taxonomy.load_taxonomy() - the SAME canonical
source core/schema.py's DocumentCategory is validated against - rather
than hardcoding a second copy of the category list here (the exact class
of drift core/taxonomy.py's own docstring exists to prevent). The
routing sentinel (uncertain_review) is excluded - it's a pipeline
routing state, never a real destination category to file a reference
image under. A target category with no prior pull yet (no existing
folder/manifest) is created fresh on first move - _load_manifest()
already tolerates a missing manifest.csv (returns []), same as every
other reader in this project.

Moving a file this way is LOCAL DATASET ORGANIZATION ONLY - still
entirely within data/outputs/reference_pull/, never data/buckets/,
data/manifest.csv, or any other production path. It does not enroll
anything into the real pipeline.

Usage:
    python ui/reference_pull_review_ui.py
"""
from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tkinter import Tk, Frame, Label, Button, StringVar, DISABLED, NORMAL, messagebox
from tkinter import ttk

from PIL import Image, ImageTk

from core.taxonomy import load_taxonomy, ROUTING_SENTINEL_ID

REFERENCE_PULL_DIR = PROJECT_ROOT / "data" / "outputs" / "reference_pull"
MANIFEST_FIELDS = [
    "file_path", "category", "source", "source_id", "source_url",
    "query", "license", "provenance",
]

_METADATA_FIELDS = [
    ("Source", "source"),
    ("License", "license"),
    ("Query", "query"),
    ("Source URL", "source_url"),
]


def _all_taxonomy_category_ids() -> list[str]:
    """The full real taxonomy (all 11 project categories), minus the
    uncertain_review routing sentinel - see this module's own docstring
    (MOVE-TARGET LIST IS THE FULL 11-CATEGORY TAXONOMY) for why the move
    dropdown uses this instead of _discover_categories() below."""
    taxonomy = load_taxonomy()
    return [c.id for c in taxonomy.categories if c.id != ROUTING_SENTINEL_ID]


def _discover_categories() -> list[str]:
    """Any subdirectory of data/outputs/reference_pull/ with its own
    manifest.csv - discovered from disk, not a hardcoded list, same
    reasoning as classifier_validation_ui.py's bucket discovery (a new
    category pulled later needs no code change here)."""
    if not REFERENCE_PULL_DIR.is_dir():
        return []
    return sorted(
        p.name for p in REFERENCE_PULL_DIR.iterdir()
        if p.is_dir() and (p / "manifest.csv").exists()
    )


def _load_manifest(category: str) -> list[dict]:
    path = REFERENCE_PULL_DIR / category / "manifest.csv"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _save_manifest(category: str, rows: list[dict]) -> None:
    path = REFERENCE_PULL_DIR / category / "manifest.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})


class ReferencePullReviewApp:
    def __init__(self, root: Tk):
        self.root = root
        root.title("Reference Pull Review")
        root.geometry("1050x900")
        root.minsize(750, 600)

        self.rows: list[dict] = []
        self.index: int = 0
        self._current_category: str | None = None
        self._visited: dict[str, set[int]] = {}
        self._tk_image: ImageTk.PhotoImage | None = None

        header = Frame(root, padx=12, pady=10)
        header.pack(fill="x")
        Label(header, text="Category:", font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="w")
        self.category_var = StringVar(value="")
        self._category_display_to_name: dict[str, str] = {}
        self.category_combo = ttk.Combobox(
            header, textvariable=self.category_var, state="readonly", width=32)
        self.category_combo.grid(row=0, column=1, sticky="w", padx=(6, 0))
        self.category_combo.bind("<<ComboboxSelected>>", self._on_category_change)

        self.reviewed_var = StringVar(value="Reviewed: 0 / 0")
        Label(header, textvariable=self.reviewed_var, font=("Segoe UI", 9), fg="#444").grid(
            row=0, column=2, sticky="e", padx=(24, 0))
        header.columnconfigure(2, weight=1)

        self.position_var = StringVar(value="Image 0 / 0")
        Label(header, textvariable=self.position_var, font=("Segoe UI", 10, "bold")).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

        self.image_frame = Frame(root, bg="#111")
        self.image_frame.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        # Same fix as classifier_validation_ui.py's own real bug (frame/
        # child feedback loop growing the window unbounded) - see that
        # file for the full root-cause writeup.
        self.image_frame.pack_propagate(False)
        self.image_label = Label(self.image_frame, bg="#111", fg="#888", font=("Segoe UI", 10))
        self.image_label.pack(fill="both", expand=True)
        self._last_render_box: tuple[int, int] | None = None
        self.image_frame.bind("<Configure>", self._on_image_frame_resize)

        meta_frame = Frame(root, padx=12)
        meta_frame.pack(fill="x")
        Label(meta_frame, text="Pull Metadata", font=("Segoe UI", 11, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))
        self._metadata_vars: dict[str, StringVar] = {}
        for i, (label_text, _field) in enumerate(_METADATA_FIELDS, start=1):
            Label(meta_frame, text=f"{label_text}:", font=("Segoe UI", 9, "bold"), anchor="nw").grid(
                row=i, column=0, sticky="nw", pady=1, padx=(0, 8))
            var = StringVar(value="")
            self._metadata_vars[label_text] = var
            Label(meta_frame, textvariable=var, font=("Segoe UI", 9), fg="#333",
                  anchor="nw", justify="left", wraplength=900).grid(row=i, column=1, sticky="nw", pady=1)
        filename_row = len(_METADATA_FIELDS) + 1
        Label(meta_frame, text="Filename:", font=("Segoe UI", 9, "bold"), anchor="nw").grid(
            row=filename_row, column=0, sticky="nw", pady=1, padx=(0, 8))
        self.filename_var = StringVar(value="")
        Label(meta_frame, textvariable=self.filename_var, font=("Consolas", 9), fg="#333",
              anchor="nw", justify="left", wraplength=900).grid(row=filename_row, column=1, sticky="nw", pady=1)

        # -- actions --
        actions = Frame(root, padx=12, pady=8)
        actions.pack(fill="x")
        self.keep_btn = Button(actions, text="✓ Keep", width=14, bg="#4caf50", fg="white",
                                font=("Segoe UI", 10, "bold"), command=self.keep_current)
        self.keep_btn.pack(side="left")
        self.reject_btn = Button(actions, text="✗ Reject (delete)", width=18, bg="#c0392b", fg="white",
                                  font=("Segoe UI", 10, "bold"), command=self.reject_current)
        self.reject_btn.pack(side="left", padx=(8, 0))

        Label(actions, text="  Move to:", font=("Segoe UI", 9)).pack(side="left", padx=(20, 4))
        self.move_target_var = StringVar(value="")
        self.move_target_combo = ttk.Combobox(
            actions, textvariable=self.move_target_var, state="readonly", width=24)
        self.move_target_combo.pack(side="left")
        self.move_btn = Button(actions, text="Move", width=10, command=self.move_current)
        self.move_btn.pack(side="left", padx=(6, 0))

        # -- navigation --
        nav = Frame(root, padx=12, pady=10)
        nav.pack(fill="x")
        self.prev_btn = Button(nav, text="◀ Previous", width=12, command=self._on_previous)
        self.prev_btn.pack(side="left")
        self.next_btn = Button(nav, text="Next ▶", width=12, command=self._on_next)
        self.next_btn.pack(side="left", padx=(8, 0))
        root.bind("<Left>", lambda _e: self._on_previous())
        root.bind("<Right>", lambda _e: self._on_next())

        root.update_idletasks()
        self._populate_categories()

    # -- category list -----------------------------------------------------

    def _populate_categories(self) -> None:
        categories = _discover_categories()
        display_values = []
        for name in categories:
            count = len(_load_manifest(name))
            display = f"{name} ({count})"
            self._category_display_to_name[display] = name
            display_values.append(display)

        self.category_combo["values"] = display_values
        self.move_target_combo["values"] = _all_taxonomy_category_ids()
        if display_values:
            self.category_combo.current(0)
            self._on_category_change()
        else:
            self.status_message("No categories found in data/outputs/reference_pull/.")

    def status_message(self, text: str) -> None:
        self.filename_var.set(text)

    def _on_category_change(self, _event=None) -> None:
        display = self.category_var.get()
        category = self._category_display_to_name.get(display)
        if category is None:
            return
        self._current_category = category
        self.rows = _load_manifest(category)
        self.index = 0
        self._visited.setdefault(category, set())
        # Default the move-target dropdown to something other than the
        # currently-open category - moving a file "to" its own category
        # is never a meaningful action.
        other_categories = [c for c in self.move_target_combo["values"] if c != category]
        if other_categories:
            self.move_target_var.set(other_categories[0])
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
            self.keep_btn.config(state=DISABLED)
            self.reject_btn.config(state=DISABLED)
            self.move_btn.config(state=DISABLED)
            self._update_reviewed_label()
            return

        self.keep_btn.config(state=NORMAL)
        self.reject_btn.config(state=NORMAL)
        self.move_btn.config(state=NORMAL)
        self.position_var.set(f"Image {self.index + 1} / {total}")
        row = self.rows[self.index]
        self._set_metadata(row)
        self._render_image(row.get("file_path"))

        self.prev_btn.config(state=NORMAL if self.index > 0 else DISABLED)
        self.next_btn.config(state=NORMAL if self.index < total - 1 else DISABLED)

        if self._current_category is not None:
            self._visited[self._current_category].add(self.index)
        self._update_reviewed_label()

    def _update_reviewed_label(self) -> None:
        if self._current_category is None:
            self.reviewed_var.set("Reviewed: 0 / 0")
            return
        visited = len(self._visited.get(self._current_category, ()))
        self.reviewed_var.set(f"Reviewed: {visited} / {len(self.rows)}")

    def _set_metadata(self, row: dict | None) -> None:
        for label_text, field in _METADATA_FIELDS:
            self._metadata_vars[label_text].set((row or {}).get(field, "") or "")
        file_path = (row or {}).get("file_path", "")
        self.filename_var.set(Path(file_path).name if file_path else "")

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
            self.image_label.config(image="", text="No images in this category")
            return

        raw_w, raw_h = self.image_frame.winfo_width(), self.image_frame.winfo_height()
        screen_w, screen_h = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        if 50 <= raw_w <= screen_w and 50 <= raw_h <= screen_h:
            box_w, box_h = raw_w, raw_h
        else:
            box_w, box_h = 900, 600

        try:
            with Image.open(file_path) as im:
                preview = im.convert("RGB")
                preview.thumbnail((box_w, box_h), Image.LANCZOS)
        except Exception as e:
            self._tk_image = None
            self.image_label.config(
                image="", text=f"Could not load image:\n{file_path}\n\n{type(e).__name__}: {e}")
            return

        self._tk_image = ImageTk.PhotoImage(preview)
        self.image_label.config(image=self._tk_image, text="")

    # -- actions ----------------------------------------------------------

    def _advance_after_removal(self) -> None:
        """Row at self.index was just removed from self.rows - stay at
        the same index (now pointing at the NEXT row), unless that was
        the last row, in which case step back one."""
        if self.index >= len(self.rows):
            self.index = max(0, len(self.rows) - 1)
        self._show_current()

    def keep_current(self) -> None:
        """No mutation - this image is fine as-is. Just advances, same
        as clicking Next, but reads as a deliberate decision in the
        review flow rather than a skip."""
        self._on_next()

    def reject_current(self) -> None:
        if not self.rows or self._current_category is None:
            return
        row = self.rows[self.index]
        file_path = Path(row["file_path"])
        confirmed = messagebox.askyesno(
            "Confirm permanent delete",
            f"Permanently delete this image and remove it from the manifest?\n\n"
            f"File: {file_path.name}\nCategory: {self._current_category}\n\n"
            "This cannot be undone from this screen.",
            icon="warning",
        )
        if not confirmed:
            return
        try:
            file_path.unlink(missing_ok=True)
        except OSError as e:
            messagebox.showerror("Delete failed", f"Could not delete {file_path}:\n{e}")
            return
        del self.rows[self.index]
        _save_manifest(self._current_category, self.rows)
        self._advance_after_removal()

    def move_current(self) -> None:
        if not self.rows or self._current_category is None:
            return
        target = self.move_target_var.get()
        if not target:
            messagebox.showinfo("Reference Pull Review", "Choose a target category first.")
            return
        if target == self._current_category:
            messagebox.showinfo("Reference Pull Review", "Already in that category.")
            return

        row = self.rows[self.index]
        source_path = Path(row["file_path"])
        target_dir = REFERENCE_PULL_DIR / target
        target_dir.mkdir(parents=True, exist_ok=True)
        dest_path = target_dir / source_path.name
        if dest_path.exists():
            messagebox.showerror(
                "Move failed",
                f"{dest_path.name} already exists in {target} - resolve the name collision "
                "manually before moving this one.")
            return

        try:
            shutil.move(str(source_path), str(dest_path))
        except OSError as e:
            messagebox.showerror("Move failed", f"Could not move {source_path}:\n{e}")
            return

        # Remove from the source category's manifest.
        del self.rows[self.index]
        _save_manifest(self._current_category, self.rows)

        # Append (with category updated) to the target category's manifest.
        moved_row = dict(row)
        moved_row["category"] = target
        moved_row["file_path"] = str(dest_path)
        target_rows = _load_manifest(target)
        target_rows.append(moved_row)
        _save_manifest(target, target_rows)

        self._advance_after_removal()


def main() -> None:
    root = Tk()
    ReferencePullReviewApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
