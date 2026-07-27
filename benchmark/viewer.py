"""
ItemViewer - a Tkinter Frame that displays ONE BenchmarkItem's full
evidence (prompt, raw output, both crop images, stage1's reading when
viewing a stage2 item) and the scoring controls for it (OCRScore,
per-error-type checkboxes, notes).

Plain tkinter throughout (Jon's direction, 2026-07-25 planning) - no
extra toolkit dependency for a tool this small. Deliberately dumb about
storage: it only ever builds a ScoreRecord and hands it to whatever
on_save callback the owning gui.py wired up - it never touches
benchmark_db.py or a run's files directly, so this module stays
reusable/testable without a real run loaded.
"""

from __future__ import annotations

import sys
from pathlib import Path
from tkinter import (
    Frame, Label, Button, Text, StringVar, BooleanVar, Checkbutton,
    END, NORMAL, DISABLED, LEFT, TOP, X, BOTH, Y, RIGHT,
)
from tkinter import ttk
from typing import Callable, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image, ImageTk

from benchmark.models import BenchmarkItem, ERROR_TYPE_FIELDS, OCRScore, ScoreRecord, Stage

IMAGE_PREVIEW_MAX = (520, 140)


class ItemViewer(Frame):
    def __init__(self, parent, on_save: Callable[[ScoreRecord], None]):
        super().__init__(parent)
        self._on_save = on_save
        self._current_item: Optional[BenchmarkItem] = None
        self._benchmark_session: str = ""
        self._tk_images: list[ImageTk.PhotoImage] = []  # keep refs alive

        self._build_widgets()

    # -- layout -----------------------------------------------------------

    def _build_widgets(self) -> None:
        header = Frame(self)
        header.pack(side=TOP, fill=X, pady=(0, 4))
        self.header_label = Label(header, text="(no item loaded)",
                                   font=("Segoe UI", 11, "bold"))
        self.header_label.pack(side=LEFT)
        self.meta_label = Label(header, text="", fg="#555")
        self.meta_label.pack(side=LEFT, padx=(12, 0))

        images_row = Frame(self)
        images_row.pack(side=TOP, fill=X, pady=(0, 4))
        self.original_crop_label = Label(images_row, text="(original crop)", bg="#ddd")
        self.original_crop_label.pack(side=LEFT, padx=(0, 8))
        self.model_input_label = Label(images_row, text="(model input)", bg="#ddd")
        self.model_input_label.pack(side=LEFT)

        stage1_row = Frame(self)
        stage1_row.pack(side=TOP, fill=X, pady=(0, 4))
        self.stage1_reading_label = Label(stage1_row, text="", fg="#555")
        self.stage1_reading_label.pack(side=LEFT)

        text_frame = Frame(self)
        text_frame.pack(side=TOP, fill=BOTH, expand=True, pady=(0, 4))

        prompt_col = Frame(text_frame)
        prompt_col.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 6))
        Label(prompt_col, text="Prompt:", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.prompt_text = Text(prompt_col, wrap="word", height=10, font=("Consolas", 9))
        self.prompt_text.pack(fill=BOTH, expand=True)

        output_col = Frame(text_frame)
        output_col.pack(side=LEFT, fill=BOTH, expand=True)
        Label(output_col, text="Raw output:", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.output_text = Text(output_col, wrap="word", height=10, font=("Consolas", 9))
        self.output_text.pack(fill=BOTH, expand=True)

        value_row = Frame(self)
        value_row.pack(side=TOP, fill=X, pady=(0, 8))
        Label(value_row, text="Parsed value:", font=("Segoe UI", 9, "bold")).pack(side=LEFT)
        self.value_label = Label(value_row, text="", font=("Consolas", 10))
        self.value_label.pack(side=LEFT, padx=(4, 16))
        Label(value_row, text="Confidence:", font=("Segoe UI", 9, "bold")).pack(side=LEFT)
        self.confidence_label = Label(value_row, text="", font=("Consolas", 10))
        self.confidence_label.pack(side=LEFT, padx=(4, 0))

        # -- scoring controls --------------------------------------------
        score_frame = Frame(self, relief="groove", borderwidth=1, padx=8, pady=6)
        score_frame.pack(side=TOP, fill=X)

        score_row = Frame(score_frame)
        score_row.pack(fill=X)
        Label(score_row, text="OCR score:", font=("Segoe UI", 9, "bold")).pack(side=LEFT)
        self.ocr_score_var = StringVar(value=OCRScore.PERFECT.value)
        ttk.Combobox(score_row, textvariable=self.ocr_score_var,
                     values=[s.value for s in OCRScore], state="readonly", width=16
                     ).pack(side=LEFT, padx=(6, 0))

        errors_row = Frame(score_frame)
        errors_row.pack(fill=X, pady=(6, 0))
        Label(errors_row, text="Error types:", font=("Segoe UI", 9, "bold")).pack(
            side=LEFT, anchor="n")
        self._error_vars: dict[str, BooleanVar] = {}
        checks_frame = Frame(errors_row)
        checks_frame.pack(side=LEFT, padx=(6, 0))
        for i, (field_name, label) in enumerate(ERROR_TYPE_FIELDS):
            var = BooleanVar(value=False)
            self._error_vars[field_name] = var
            Checkbutton(checks_frame, text=label, variable=var).grid(
                row=i // 4, column=i % 4, sticky="w", padx=(0, 10))

        notes_row = Frame(score_frame)
        notes_row.pack(fill=X, pady=(6, 0))
        Label(notes_row, text="Notes:", font=("Segoe UI", 9, "bold")).pack(side=LEFT, anchor="n")
        self.notes_text = Text(notes_row, height=2, width=60, font=("Segoe UI", 9))
        self.notes_text.pack(side=LEFT, padx=(6, 0), fill=X, expand=True)

        save_row = Frame(score_frame)
        save_row.pack(fill=X, pady=(6, 0))
        self.save_button = Button(save_row, text="Save score", bg="#4a7", fg="white",
                                   font=("Segoe UI", 10, "bold"), command=self._handle_save,
                                   state=DISABLED)
        self.save_button.pack(side=LEFT)
        self.save_status_label = Label(save_row, text="", fg="#444")
        self.save_status_label.pack(side=LEFT, padx=(10, 0))

    # -- image helpers ------------------------------------------------------

    def _load_preview(self, label: Label, path) -> None:
        if not path:
            label.config(image="", text="(not captured)")
            return
        try:
            img = Image.open(path)
            img.thumbnail(IMAGE_PREVIEW_MAX)
            tk_img = ImageTk.PhotoImage(img)
        except Exception as e:
            label.config(image="", text=f"(failed to load: {e})")
            return
        self._tk_images.append(tk_img)
        label.config(image=tk_img, text="")

    # -- public API -----------------------------------------------------------

    def set_benchmark_session(self, session_id: str) -> None:
        self._benchmark_session = session_id

    def load_item(self, item: BenchmarkItem, existing_score: Optional[ScoreRecord] = None) -> None:
        """Populates every widget from `item`, and pre-fills the scoring
        controls from `existing_score` (the latest saved score for this
        item_id, if any) so reopening an already-scored item shows what
        was previously recorded rather than resetting to defaults."""
        self._current_item = item
        self._tk_images.clear()

        self.header_label.config(
            text=f"Row {item.row_index} — {item.column_name} ({item.stage.value})")
        meta_bits = [f"model={item.model}"]
        if item.runtime_seconds is not None:
            meta_bits.append(f"runtime={item.runtime_seconds:.2f}s")
        if item.reasoning_enabled is not None:
            meta_bits.append(f"reasoning={item.reasoning_enabled}")
        if item.generation_config_hash:
            meta_bits.append(f"config={item.generation_config_hash}")
        self.meta_label.config(text=" | ".join(meta_bits))

        self._load_preview(self.original_crop_label, item.original_crop_path)
        self._load_preview(self.model_input_label, item.model_input_path)

        if item.stage is Stage.STAGE2 and item.stage1_raw_reading is not None:
            self.stage1_reading_label.config(
                text=f"Stage 1 raw reading (input to this stage2 call): {item.stage1_raw_reading!r}")
        else:
            self.stage1_reading_label.config(text="")

        self.prompt_text.delete("1.0", END)
        self.prompt_text.insert(END, item.prompt_text)

        self.output_text.delete("1.0", END)
        self.output_text.insert(END, item.output_text)
        if item.exception:
            self.output_text.insert(END, f"\n\n[EXCEPTION] {item.exception}")

        self.value_label.config(text=item.value or "(empty)")
        self.confidence_label.config(text=item.confidence or "—")

        if existing_score is not None:
            self.ocr_score_var.set(existing_score.ocr_score)
            for field_name, var in self._error_vars.items():
                var.set(getattr(existing_score, field_name, False))
            self.notes_text.delete("1.0", END)
            self.notes_text.insert(END, existing_score.notes)
            self.save_status_label.config(
                text=f"Previously scored at {existing_score.timestamp} - saving again will "
                     f"add a NEW history entry, not overwrite it.")
        else:
            self.ocr_score_var.set(OCRScore.PERFECT.value)
            for var in self._error_vars.values():
                var.set(False)
            self.notes_text.delete("1.0", END)
            self.save_status_label.config(text="Not yet scored.")

        self.save_button.config(state=NORMAL)

    def clear(self) -> None:
        self._current_item = None
        self.header_label.config(text="(no item loaded)")
        self.meta_label.config(text="")
        self.save_button.config(state=DISABLED)

    def _handle_save(self) -> None:
        item = self._current_item
        if item is None:
            return
        from datetime import datetime, timezone
        record = ScoreRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            benchmark_session=self._benchmark_session,
            item_id=item.item_id,
            run_id=item.run_id,
            stage=item.stage.value,
            row=str(item.row_index),
            column=str(item.column_index),
            column_name=item.column_name,
            model=item.model or "",
            runtime=f"{item.runtime_seconds:.3f}" if item.runtime_seconds is not None else "",
            generation_config_hash=item.generation_config_hash or "",
            prompt_hash=item.prompt_hash,
            ocr_score=self.ocr_score_var.get(),
            confidence=item.confidence or "",
            notes=self.notes_text.get("1.0", END).strip(),
            **{name: var.get() for name, var in self._error_vars.items()},
        )
        self._on_save(record)
        self.save_status_label.config(text=f"Saved at {record.timestamp}.")
