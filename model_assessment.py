"""
Model Assessment tool — isolated model+prompt+settings testing.

HARD RULE, enforced structurally not just by convention: this script
never imports or calls anything from core/classifier.py or
core/extractor.py, never writes to data/buckets/*.csv, never writes to
data/outputs/extracted.csv, and never touches config/pipeline.yaml.
It only ever writes JSON reports to data/outputs/model_assessments/.
This is what makes it safe to experiment freely - nothing here can
contaminate the real pipeline's data.

Usage:
    python model_assessment.py

Workflow: pick an image, pick a bucket type (for schema validation
context, since ExtractionResult needs a category), pick a prompt file,
pick a model profile (YAML from config/models/), optionally override
any generation setting, run, inspect raw/parsed output + schema
validation + anomaly flags + runtime, then save a report.

Settings changed in the UI are recorded as "temporary_overrides"
separately from the loaded profile's base settings, per design -
this lets a report show exactly what deviated from the saved profile
without silently mutating the profile file itself.
"""

from __future__ import annotations

import json
import time
import copy
import subprocess
import sys
import platform
import gc
import yaml
import torch
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from tkinter import (
    Tk, Frame, Label, Button, StringVar, BooleanVar,
    Entry, OptionMenu, Text, filedialog, messagebox, END, DISABLED, NORMAL,
)
from PIL import Image, ImageTk
from pydantic import ValidationError

from core.loaders.base_loader import GenerationConfig, load_model_config
from core.loader_registry import LOADER_REGISTRY
from core.schema import DocumentCategory, ConfidenceLevel
from core.image_preprocessing import PREPROCESSING_PROFILES, apply_profile

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "config" / "models"
PROMPTS_DIR = PROJECT_ROOT / "config" / "prompts"
BUCKET_PROFILES_DIR = PROJECT_ROOT / "config" / "bucket_profiles"
ASSESSMENT_DIR = PROJECT_ROOT / "data" / "outputs" / "model_assessments"

OVERRIDABLE_FIELDS = [
    "temperature", "top_p", "top_k", "max_new_tokens",
    "do_sample", "repetition_penalty", "presence_penalty", "no_repeat_ngram_size",
    "min_pixels", "max_pixels", "num_beams", "length_penalty", "early_stopping",
]

MAX_PREVIEW_SIZE = (420, 560)


def list_model_profiles() -> list[str]:
    return sorted(p.stem for p in MODELS_DIR.glob("*.yaml"))


def list_prompt_files() -> list[str]:
    return sorted(p.name for p in PROMPTS_DIR.glob("*.txt"))


def list_bucket_profiles() -> list[str]:
    return sorted(p.stem for p in BUCKET_PROFILES_DIR.glob("*.yaml"))


def load_bucket_profile(bucket_name: str) -> dict:
    """
    Reads a bucket profile YAML and returns only the fields that are
    real GenerationConfig fields, so callers can apply them via
    setattr() the same way UI overrides are applied. Missing file or
    an empty/comment-only file both return {} (no overrides) rather
    than erroring - bucket profiles are optional by design (see the
    placeholder files for untested buckets). Unknown keys are ignored
    with a console warning rather than silently absorbed, since a
    typo'd field name here should be visible, not invisible.
    """
    path = BUCKET_PROFILES_DIR / f"{bucket_name}.yaml"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    known = {k: v for k, v in data.items() if k in GenerationConfig.__dataclass_fields__}
    unknown = set(data.keys()) - set(known.keys())
    if unknown:
        print(f"[bucket_profiles] Warning: {path.name} has unrecognized field(s) "
              f"{unknown} - not a real GenerationConfig field, ignored. Check for typos.")
    return known


def run_repeated_entry_check(personal_names: list) -> list[dict]:
    """
    Single-image subset of the batch anomaly checks in core/extractor.py.
    no_ngram_overlap and length_outlier need a batch to be meaningful -
    those are intentionally NOT run here, since a one-image test has
    no batch to compare against. repeated_entry (same name 3+ times
    within one record) is meaningful even for a single image, so it's
    included.
    """
    flags = []
    seen: dict[str, int] = {}
    for n in personal_names:
        seen[n.value] = seen.get(n.value, 0) + 1
    for value, count in seen.items():
        if count >= 3:
            flags.append({
                "flag_type": "repeated_entry",
                "detail": f"Name {value!r} appears {count} times within this record - "
                          f"possible loop degeneration.",
                "severity": "high",
            })
    return flags


class AssessmentApp:
    def __init__(self, root: Tk):
        self.root = root
        root.title("Model Assessment (isolated - does not touch pipeline data)")
        root.geometry("1050x1250")
        root.minsize(1000, 850)  # can shrink for smaller screens, but not
        # so far that the save button becomes unreachable without manual
        # resizing - the actual bug reported was the fixed 900px default
        # no longer fitting everything added since that height was set.

        self.image_path: str | None = None
        self._last_preprocessed_path: str | None = None
        self.last_result: dict | None = None

        warning = Label(
            root,
            text="Isolated test only. Never writes to bucket CSVs, extracted.csv, "
                 "or pipeline.yaml. Reports go to data/outputs/model_assessments/ only.",
            fg="#a33", font=("Segoe UI", 9, "bold"),
        )
        warning.pack(pady=(8, 4))

        top = Frame(root)
        top.pack(fill="x", padx=12)

        # -- image selection ---------------------------------------------
        img_row = Frame(top)
        img_row.pack(fill="x", pady=4)
        Button(img_row, text="Select image...", command=self.select_image).pack(side="left")
        self.open_image_button = Button(
            img_row, text="Open image", command=self.open_image_in_viewer, state=DISABLED
        )
        self.open_image_button.pack(side="left", padx=(6, 0))
        self.open_preprocessed_button = Button(
            img_row, text="Open preprocessed copy",
            command=self.open_preprocessed_in_viewer, state=DISABLED,
        )
        self.open_preprocessed_button.pack(side="left", padx=(6, 0))
        self.image_path_label = Label(img_row, text="(no image selected)", fg="#666")
        self.image_path_label.pack(side="left", padx=8)

        # -- bucket / prompt / profile selection --------------------------
        select_row = Frame(top)
        select_row.pack(fill="x", pady=4)

        Label(select_row, text="Bucket type:").grid(row=0, column=0, sticky="w")
        self.bucket_var = StringVar(value=list(DocumentCategory)[0].value)
        OptionMenu(select_row, self.bucket_var, *[c.value for c in DocumentCategory
                                                    if c != DocumentCategory.UNCERTAIN]
                   ).grid(row=0, column=1, sticky="w", padx=(4, 20))

        Label(select_row, text="Prompt file:").grid(row=0, column=2, sticky="w")
        prompt_files = list_prompt_files() or ["(none found)"]
        self.prompt_var = StringVar(value=prompt_files[0])
        self.prompt_menu = OptionMenu(
            select_row, self.prompt_var, *prompt_files, command=self._on_prompt_file_change
        )
        self.prompt_menu.grid(row=0, column=3, sticky="w", padx=4)
        Button(select_row, text="\u21bb", width=2, command=self.reload_prompt_files,
               ).grid(row=0, column=4, sticky="w")

        Label(select_row, text="Bucket config:").grid(row=0, column=5, sticky="w", padx=(16, 0))
        bucket_profiles = ["(none)"] + list_bucket_profiles()
        self.bucket_profile_var = StringVar(value="(none)")
        self.bucket_profile_menu = OptionMenu(
            select_row, self.bucket_profile_var, *bucket_profiles
        )
        self.bucket_profile_menu.grid(row=0, column=6, sticky="w", padx=4)
        Button(select_row, text="\u21bb", width=2, command=self.reload_bucket_profiles,
               ).grid(row=0, column=7, sticky="w")

        Label(select_row, text="Model profile:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        profiles = list_model_profiles() or ["(none found)"]
        self.profile_var = StringVar(value=profiles[0])
        self.profile_menu = OptionMenu(
            select_row, self.profile_var, *profiles, command=self._on_profile_change
        )
        self.profile_menu.grid(row=1, column=1, sticky="w", padx=(4, 20), pady=(6, 0))
        Button(select_row, text="\u21bb", width=2, command=self.reload_model_profiles,
               ).grid(row=1, column=2, sticky="w", pady=(6, 0))

        Label(select_row, text="Preprocessing:").grid(row=1, column=3, sticky="w", pady=(6, 0))
        preprocessing_choices = list(PREPROCESSING_PROFILES.keys())
        self.preprocessing_var = StringVar(value="none")
        self.preprocessing_menu = OptionMenu(
            select_row, self.preprocessing_var, *preprocessing_choices
        )
        self.preprocessing_menu.grid(row=1, column=4, sticky="w", padx=4, pady=(6, 0))
        # No reload button here (unlike prompts/buckets/models) - this
        # dropdown is defined in-code (core/image_preprocessing.py), not
        # discovered from a directory of files, so there's nothing on
        # disk to rescan.

        # -- prompt text override: pre-filled from the selected file, ----
        # editable in place, applies to THIS RUN ONLY - never written back
        # to the .txt file on disk. Exists because iterating on prompt
        # wording mid-session previously required creating a new versioned
        # file for every small wording tweak, even one-off experiments
        # that didn't warrant a permanent v4/v5/etc.
        prompt_override_frame = Frame(top)
        prompt_override_frame.pack(fill="x", pady=(8, 4))
        override_header = Frame(prompt_override_frame)
        override_header.pack(fill="x")
        Label(override_header, text="Prompt text (editable - overrides the file above for "
                                     "this run only, never saved back to disk):",
              font=("Segoe UI", 9, "bold")).pack(side="left")
        Button(override_header, text="Reload from file",
               command=self.reload_prompt_text_box).pack(side="left", padx=(12, 0))
        self.prompt_text_box = Text(prompt_override_frame, wrap="word", height=8,
                                     font=("Consolas", 9))
        self.prompt_text_box.pack(fill="x", pady=(4, 0))
        self._loaded_prompt_file_text = ""  # tracks on-disk content, to detect edits
        self.reload_prompt_text_box()  # populate box from the initially selected file

        # -- override fields ------------------------------------------------
        override_frame = Frame(top)
        override_frame.pack(fill="x", pady=(10, 4))
        Label(override_frame, text="Overrides (blank = use profile default):",
              font=("Segoe UI", 9, "bold")).grid(row=0, column=0, columnspan=4, sticky="w")

        self.override_vars: dict[str, StringVar] = {}
        for i, field in enumerate(OVERRIDABLE_FIELDS):
            row, col = divmod(i, 3)
            Label(override_frame, text=field).grid(row=row + 1, column=col * 2, sticky="w", padx=(0, 4))
            var = StringVar(value="")
            Entry(override_frame, textvariable=var, width=12).grid(
                row=row + 1, column=col * 2 + 1, sticky="w", padx=(0, 12), pady=2
            )
            self.override_vars[field] = var

        # -- prior result comparison (optional) ---------------------------------
        prior_frame = Frame(top)
        prior_frame.pack(fill="x", pady=(6, 4))
        Label(prior_frame, text="Prior result (optional, for before/after comparison):",
              font=("Segoe UI", 9, "bold")).grid(row=0, column=0, columnspan=4, sticky="w")

        Label(prior_frame, text="Prior source:").grid(row=1, column=0, sticky="w")
        self.prior_source_var = StringVar(value="")
        Entry(prior_frame, textvariable=self.prior_source_var, width=20).grid(
            row=1, column=1, sticky="w", padx=(4, 20)
        )
        Label(prior_frame, text="Prior outcome:").grid(row=1, column=2, sticky="w")
        self.prior_outcome_var = StringVar(value="")
        Entry(prior_frame, textvariable=self.prior_outcome_var, width=30).grid(
            row=1, column=3, sticky="w", padx=4
        )

        Label(prior_frame, text="Change reason (why might this run differ?):").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )
        self.change_reason_var = StringVar(value="")
        Entry(prior_frame, textvariable=self.change_reason_var, width=60).grid(
            row=2, column=2, columnspan=2, sticky="w", padx=4, pady=(4, 0)
        )

        # -- run button + status ---------------------------------------------
        run_row = Frame(top)
        run_row.pack(fill="x", pady=8)
        self.run_button = Button(run_row, text="Run test", command=self.run_test,
                                  bg="#2a6", fg="white", font=("Segoe UI", 10, "bold"))
        self.run_button.pack(side="left")
        self.status_label = Label(run_row, text="", fg="#444")
        self.status_label.pack(side="left", padx=12)

        # -- results display ---------------------------------------------
        results_frame = Frame(root)
        results_frame.pack(fill="both", expand=True, padx=12, pady=(4, 4))

        left = Frame(results_frame)
        left.pack(side="left", fill="both", expand=True)
        Label(left, text="Preview").pack(anchor="w")
        self.image_label = Label(left)
        self.image_label.pack(anchor="w")
        self._tk_image = None

        right = Frame(results_frame)
        right.pack(side="left", fill="both", expand=True, padx=(12, 0))
        Label(right, text="Result").pack(anchor="w")
        self.result_text = Text(right, wrap="word", height=24, font=("Consolas", 9))
        self.result_text.pack(fill="both", expand=True)

        # -- notes + save ---------------------------------------------
        bottom = Frame(root)
        bottom.pack(fill="x", padx=12, pady=8)
        Label(bottom, text="Reviewer notes:").pack(anchor="w")
        self.notes_entry = Entry(bottom, width=100)
        self.notes_entry.pack(fill="x", pady=(2, 6))
        self.save_button = Button(bottom, text="Save assessment report",
                                   command=self.save_report, state=DISABLED)
        self.save_button.pack(anchor="w")

    def _on_profile_change(self, *_):
        pass  # placeholder hook if we want to auto-populate override defaults later

    @staticmethod
    def _refresh_optionmenu(option_menu: OptionMenu, string_var: StringVar,
                             choices: list[str], command=None):
        """
        tkinter's OptionMenu has no built-in "update the choice list"
        method - the menu has to be cleared and repopulated directly.
        Preserves the current selection if it's still valid in the new
        list; falls back to the first available choice otherwise (e.g.
        the previously selected file was deleted from disk).
        """
        menu = option_menu["menu"]
        menu.delete(0, "end")
        for choice in choices:
            def _select(v=choice):
                string_var.set(v)
                if command:
                    command(v)
            menu.add_command(label=choice, command=_select)
        if string_var.get() not in choices and choices:
            string_var.set(choices[0])
            if command:
                command(choices[0])

    def reload_prompt_files(self):
        choices = list_prompt_files() or ["(none found)"]
        self._refresh_optionmenu(self.prompt_menu, self.prompt_var, choices,
                                  command=self._on_prompt_file_change)

    def reload_bucket_profiles(self):
        choices = ["(none)"] + list_bucket_profiles()
        self._refresh_optionmenu(self.bucket_profile_menu, self.bucket_profile_var, choices)

    def reload_model_profiles(self):
        choices = list_model_profiles() or ["(none found)"]
        self._refresh_optionmenu(self.profile_menu, self.profile_var, choices)

    def _on_prompt_file_change(self, *_):
        # User picked a different prompt file from the dropdown - reload
        # the override box to match it. Any unsaved edits to the previous
        # file's text are discarded here, same as switching away from an
        # unsaved document; there's no draft-per-file tracking.
        self.reload_prompt_text_box()

    def reload_prompt_text_box(self):
        prompt_file = self.prompt_var.get()
        prompt_path = PROMPTS_DIR / prompt_file
        try:
            text = prompt_path.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError) as e:
            text = f"[Could not read {prompt_file}: {e}]"
        self._loaded_prompt_file_text = text
        self.prompt_text_box.delete("1.0", END)
        self.prompt_text_box.insert("1.0", text)

    def select_image(self):
        path = filedialog.askopenfilename(
            title="Select test image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp")],
        )
        if not path:
            return
        self.image_path = path
        self.image_path_label.config(text=path, fg="black")
        self.open_image_button.config(state=NORMAL)
        try:
            img = Image.open(path)
            img.thumbnail(MAX_PREVIEW_SIZE)
            self._tk_image = ImageTk.PhotoImage(img)
            self.image_label.config(image=self._tk_image)
        except Exception as e:
            messagebox.showerror("Image load error", str(e))

    def open_image_in_viewer(self):
        """
        Opens the full-resolution source image in the OS default viewer -
        the preview pane thumbnail (capped at MAX_PREVIEW_SIZE) is often
        too small to actually read dense handwriting/small print, which
        is exactly the content this tool spends most of its time testing.
        """
        if not self.image_path:
            return
        try:
            system = platform.system()
            if system == "Windows":
                import os
                os.startfile(self.image_path)  # noqa: S606 - local trusted path only
            elif system == "Darwin":
                subprocess.run(["open", self.image_path], check=True)
            else:
                subprocess.run(["xdg-open", self.image_path], check=True)
        except Exception as e:
            messagebox.showerror("Could not open image", str(e))

    def open_preprocessed_in_viewer(self):
        """
        Opens the preprocessed copy from the most recent run - lets the
        person visually compare it against the original (via the
        adjacent "Open image" button) to judge whether a preprocessing
        step actually improved legibility before trusting any change in
        model output to it. Disabled until a run has actually produced
        one (preprocessing_var != "none"); see run_test().
        """
        if not self._last_preprocessed_path:
            return
        try:
            system = platform.system()
            if system == "Windows":
                import os
                os.startfile(self._last_preprocessed_path)  # noqa: S606
            elif system == "Darwin":
                subprocess.run(["open", self._last_preprocessed_path], check=True)
            else:
                subprocess.run(["xdg-open", self._last_preprocessed_path], check=True)
        except Exception as e:
            messagebox.showerror("Could not open preprocessed image", str(e))

    def _collect_overrides(self) -> dict:
        overrides = {}
        for field, var in self.override_vars.items():
            raw = var.get().strip()
            if raw == "":
                continue
            if field in ("do_sample", "early_stopping"):
                overrides[field] = raw.lower() in ("true", "1", "yes")
            elif field in ("top_k", "max_new_tokens", "no_repeat_ngram_size",
                           "min_pixels", "max_pixels", "num_beams"):
                try:
                    overrides[field] = int(raw)
                except ValueError:
                    raise ValueError(f"'{field}' must be an integer, got {raw!r}")
            else:
                try:
                    overrides[field] = float(raw)
                except ValueError:
                    raise ValueError(f"'{field}' must be a number, got {raw!r}")
        return overrides

    def run_test(self):
        if not self.image_path:
            messagebox.showwarning("No image", "Select an image first.")
            return
        if self.prompt_var.get() == "(none found)" or self.profile_var.get() == "(none found)":
            messagebox.showwarning("Missing config", "No prompt files or model profiles found.")
            return

        try:
            overrides = self._collect_overrides()
        except ValueError as e:
            messagebox.showerror("Invalid override", str(e))
            return

        self.status_label.config(text="Loading model and running... (see console)")
        self.result_text.delete("1.0", END)
        self.root.update_idletasks()

        profile_name = self.profile_var.get()
        prompt_file = self.prompt_var.get()
        bucket = self.bucket_var.get()
        bucket_profile_name = self.bucket_profile_var.get()

        loader = None  # set inside the try; checked in finally to release
        # the model even if something fails before or after it's created
        try:
            base_config = load_model_config(profile_name)
            base_settings_snapshot = copy.deepcopy(asdict(base_config))

            # Apply overrides to an in-memory copy only - the YAML file
            # on disk is never touched by this tool.
            test_config = load_model_config(profile_name)

            # Precedence: model defaults (just loaded) -> bucket profile
            # overrides -> assessment UI overrides (applied last, so a
            # one-off UI change always wins over both persisted layers).
            bucket_profile_applied = {}
            if bucket_profile_name != "(none)":
                bucket_profile_applied = load_bucket_profile(bucket_profile_name)
                for field, value in bucket_profile_applied.items():
                    setattr(test_config, field, value)

            for field, value in overrides.items():
                setattr(test_config, field, value)

            # Use the editable prompt box's current content, not a fresh
            # file read - the box is pre-filled from prompt_file when
            # selected, but may have been hand-edited for this run only
            # (never written back to disk). Compare against the tracked
            # on-disk text to know whether an override is actually active.
            current_prompt_text = self.prompt_text_box.get("1.0", "end-1c")
            prompt_override_active = current_prompt_text != self._loaded_prompt_file_text
            test_config.prompt_text = current_prompt_text

            loader_cls = LOADER_REGISTRY.get(test_config.loader_class)
            if loader_cls is None:
                raise ValueError(f"No loader registered for {test_config.loader_class!r}")

            loader = loader_cls(test_config)
            loader.initialize_model_and_tokenizer()

            with Image.open(self.image_path) as original_image:
                start = time.time()
                task = "classify" if prompt_file.startswith("classify") else "extract"
                prompt = loader._build_prompt(task=task)

                preprocessing_profile = self.preprocessing_var.get()
                if preprocessing_profile != "none":
                    # apply_profile always returns a NEW image and never
                    # mutates original_image - the source file on disk and
                    # the preview pane (built from a separate Image.open()
                    # call in select_image) are both untouched regardless
                    # of what happens here.
                    raw_image = apply_profile(original_image, preprocessing_profile)
                    ASSESSMENT_DIR.mkdir(parents=True, exist_ok=True)
                    preprocessed_path = ASSESSMENT_DIR / (
                        f"preprocessed_{preprocessing_profile}_"
                        f"{Path(self.image_path).stem}.png"
                    )
                    raw_image.save(preprocessed_path)
                    self._last_preprocessed_path = str(preprocessed_path)
                    self.open_preprocessed_button.config(state=NORMAL)
                else:
                    raw_image = original_image
                    self._last_preprocessed_path = None
                    self.open_preprocessed_button.config(state=DISABLED)

                oom_recovered = False
                try:
                    raw_output = loader._run_generate(raw_image, prompt)
                except Exception as e:
                    if "out of memory" in str(e).lower():
                        torch.cuda.empty_cache()
                        oom_recovered = True
                        raw_output = loader._run_generate(raw_image, prompt)
                    else:
                        raise

                runtime = time.time() - start

                schema_pass = True
                schema_errors = ""
                parsed_output_display = ""
                anomaly_flags = []

                try:
                    if task == "classify":
                        result = loader._parse_classification(self.image_path, raw_output)
                        parsed_output_display = result.model_dump_json(indent=2)
                    else:
                        result = loader._parse_extraction(self.image_path, bucket, raw_output)
                        parsed_output_display = result.model_dump_json(indent=2)
                        anomaly_flags = run_repeated_entry_check(result.personal_names)
                except (ValueError, ValidationError, NotImplementedError) as e:
                    schema_pass = False
                    schema_errors = str(e)

            # Derive a short current-outcome label from the actual result,
            # so the before/after comparison isn't just a raw dump but a
            # comparable classification (e.g. "Looped" vs "Schema pass").
            if not schema_pass:
                current_outcome = "Schema fail"
            elif any(f["severity"] == "high" for f in anomaly_flags):
                current_outcome = "Anomaly flagged (high severity - likely loop/repetition)"
            elif anomaly_flags:
                current_outcome = "Schema pass, anomaly flagged (low/medium)"
            else:
                current_outcome = "Schema pass, no anomalies"

            self.last_result = {
                "image_path": self.image_path,
                "bucket": bucket,
                "prompt_file": prompt_file,
                "model_id": base_config.repo_id,
                "profile_name": profile_name,
                "profile_settings": base_settings_snapshot,
                "bucket_profile_name": bucket_profile_name,
                "bucket_profile_applied": bucket_profile_applied,
                "prompt_override_active": prompt_override_active,
                "preprocessing_profile": preprocessing_profile,
                "preprocessed_image_path": self._last_preprocessed_path,
                "prompt_text_used": current_prompt_text if prompt_override_active else None,
                "temporary_overrides": overrides,
                "raw_output": raw_output,
                "parsed_output": parsed_output_display,
                "schema_pass": schema_pass,
                "schema_errors": schema_errors,
                "anomaly_flags": anomaly_flags,
                "runtime_seconds": round(runtime, 2),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reviewer_notes": "",
                "device_map": getattr(loader, "_execution_meta", {}).get("device_map"),
                "max_memory": getattr(loader, "_execution_meta", {}).get("max_memory"),
                "vram_headroom_gb": test_config.vram_headroom_gb,
                "cpu_offload_limit_gb": test_config.cpu_offload_limit_gb,
                "oom_recovered": oom_recovered,
                "previous_result": {
                    "source": self.prior_source_var.get().strip(),
                    "outcome": self.prior_outcome_var.get().strip(),
                } if self.prior_source_var.get().strip() or self.prior_outcome_var.get().strip() else None,
                "current_result": {
                    "source": "Pipeline v2",
                    "outcome": current_outcome,
                },
                "change_reason": self.change_reason_var.get().strip() or None,
            }

            self._display_result(self.last_result)
            self.status_label.config(text=f"Done in {runtime:.1f}s. "
                                           f"Schema: {'PASS' if schema_pass else 'FAIL'}")
            self.save_button.config(state=NORMAL)

        except Exception as e:
            self.status_label.config(text=f"ERROR: {e}")
            self.result_text.insert(END, f"Exception during test run:\n{e}")
            messagebox.showerror("Test run failed", str(e))

        finally:
            # This is the actual VRAM-accumulation fix. Previously,
            # torch.cuda.empty_cache() only ran on the OOM-recovery path -
            # a normal successful run never called it. PyTorch's CUDA
            # allocator doesn't return freed memory to the OS on its own;
            # it keeps it cached inside the process for reuse. Each run's
            # model gets dereferenced by Python fine, but that memory
            # stayed parked in PyTorch's pool rather than actually being
            # released, so nvidia-smi/Task Manager usage climbed across a
            # session and only dropped when the whole app was closed -
            # not because models were literally stacked simultaneously,
            # but because nothing was ever handing the memory back.
            self._release_model(loader)

    def _release_model(self, loader):
        if loader is None:
            return
        before_mb = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
        try:
            loader.model = None
            loader.processor = None
            loader.tokenizer = None
        except Exception:
            pass  # loader may not have gotten far enough to set these
        del loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            after_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            print(f"[model_assessment] VRAM released: {before_mb:.0f} MB -> "
                  f"{after_mb:.0f} MB allocated after cleanup")

    def _display_result(self, result: dict):
        self.result_text.delete("1.0", END)
        lines = [
            f"Model: {result['model_id']} (profile: {result['profile_name']})",
            f"Prompt: {result['prompt_file']}   Bucket: {result['bucket']}",
            f"Bucket config: {result['bucket_profile_name']}"
            + (f"  -> applied {result['bucket_profile_applied']}"
               if result['bucket_profile_applied'] else ""),
            f"Prompt text edited for this run: "
            f"{'YES (differs from ' + result['prompt_file'] + ' on disk)' if result['prompt_override_active'] else 'no (using file as-is)'}",
            f"Image preprocessing: {result['preprocessing_profile']}"
            + (f"  -> {result['preprocessed_image_path']}"
               if result.get('preprocessed_image_path') else ""),
            f"Runtime: {result['runtime_seconds']}s   "
            f"Schema: {'PASS' if result['schema_pass'] else 'FAIL'}",
            f"Overrides applied: {result['temporary_overrides'] or '(none)'}",
        ]
        if result.get("previous_result"):
            lines += [
                "",
                "--- BEFORE / AFTER COMPARISON ---",
                f"  Previous ({result['previous_result']['source']}): "
                f"{result['previous_result']['outcome']}",
                f"  Current  ({result['current_result']['source']}): "
                f"{result['current_result']['outcome']}",
            ]
            if result.get("change_reason"):
                lines.append(f"  Reason for difference: {result['change_reason']}")
        lines += [
            "",
            "--- RAW OUTPUT ---",
            result["raw_output"],
            "",
            "--- PARSED OUTPUT ---",
            result["parsed_output"] or "(parse failed, see schema_errors)",
        ]
        if not result["schema_pass"]:
            lines += ["", "--- SCHEMA ERRORS ---", result["schema_errors"]]
        if result["anomaly_flags"]:
            lines += ["", "--- ANOMALY FLAGS ---"]
            for flag in result["anomaly_flags"]:
                lines.append(f"  [{flag['severity']}] {flag['flag_type']}: {flag['detail']}")
        self.result_text.insert(END, "\n".join(lines))

    def save_report(self):
        if not self.last_result:
            return
        self.last_result["reviewer_notes"] = self.notes_entry.get().strip()

        ASSESSMENT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp_safe = self.last_result["timestamp"].replace(":", "-")
        image_stem = Path(self.last_result["image_path"]).stem
        out_path = ASSESSMENT_DIR / f"{timestamp_safe}_{image_stem}_{self.last_result['profile_name']}.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.last_result, f, indent=2, default=str)

        messagebox.showinfo("Saved", f"Report saved to:\n{out_path}")


def main():
    root = Tk()
    AssessmentApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
