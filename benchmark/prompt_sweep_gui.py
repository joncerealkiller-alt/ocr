"""
GUI for benchmark/prompt_sweep.py - built 2026-07-29 per Jon's direction,
AFTER the CLI engine was built and smoke-tested end-to-end (real
inference, real ground-truth scoring, history log confirmed accumulating
across runs). The point of this GUI, in Jon's own words: "removing
friction... not have to copy/paste filepaths for every variable" - it's
a picker over the same CLI, not a second implementation of the sweep
logic.

ARCHITECTURE: matches debug_tools/workflow_gui.py's own stated principle
- this shells out to the REAL CLI entry point (python -u benchmark/
prompt_sweep.py ...) as a subprocess and streams its stdout; it does NOT
reimplement discover_examples()/run_sweep()/scoring. The command built
in _build_command() is both the literal subprocess argv AND the command-
preview text - never two things that could drift apart.

LIVE PROGRESS: prompt_sweep.py additionally prints "@@PROGRESS <json>"
lines (see that module's _emit()) alongside its normal human-readable
output - added specifically to drive this GUI's queue/current-run panels
without fragile text-scraping of the human-oriented log lines. Those
lines are parsed here and stripped from the visible log (they're for
this GUI, not for a person reading the log). -u is passed explicitly to
the subprocess so those lines arrive promptly instead of sitting in a
block buffer - debug_tools/workflow_gui.py's own CommandTab doesn't need
this (it only ever displays raw text, buffering delay doesn't matter
there), but the responsiveness of the queue/right-panel here does.

ISOLATION: this file only ever launches benchmark/prompt_sweep.py as a
subprocess and reads config/models/*.yaml + data/outputs/
ground_truth_log.jsonl (both read-only) to populate its pickers - same
contract as the CLI it wraps (see that module's own docstring).
"""

from __future__ import annotations

import json
import platform
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from tkinter import (
    Tk, Frame, Label, Button, StringVar, Entry, Listbox, Text, Scrollbar,
    filedialog, messagebox, END, DISABLED, NORMAL, EXTENDED, BOTH, X, Y, LEFT, RIGHT, TOP, BOTTOM, ttk,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.prompt_sweep_report import _load_records, _rate  # noqa: E402

MODELS_DIR = PROJECT_ROOT / "config" / "models"
DEFAULT_GT_LOG = PROJECT_ROOT / "data" / "outputs" / "ground_truth_log.jsonl"
DEFAULT_LOG_FILE = PROJECT_ROOT / "data" / "outputs" / "prompt_sweep_log.jsonl"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "outputs" / "prompt_sweep_runs"
SWEEP_SCRIPT = PROJECT_ROOT / "benchmark" / "prompt_sweep.py"

PROGRESS_PREFIX = "@@PROGRESS "


def list_model_profiles() -> list[str]:
    return sorted(p.stem for p in MODELS_DIR.glob("*.yaml"))


def _read_gt_records(gt_log_path: str) -> list[dict]:
    path = Path(gt_log_path)
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def discover_gt_pages(gt_log_path: str) -> list[str]:
    records = _read_gt_records(gt_log_path)
    return sorted({r.get("sidecar_path") for r in records if r.get("sidecar_path")})


def discover_gt_columns(gt_log_path: str, pages: list[str] | None = None) -> list[str]:
    records = _read_gt_records(gt_log_path)
    if pages:
        records = [r for r in records if r.get("sidecar_path") in pages]
    return sorted({r.get("column") for r in records if r.get("column")})


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


class PromptSweepApp:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("Prompt/Model Sweep")
        self.root.geometry("1360x900")

        self.process: subprocess.Popen | None = None
        self.log_queue: queue.Queue = queue.Queue()
        self.queue_items: dict[tuple[str, str], str] = {}  # (ocr_prompt, structuring_prompt) -> tree item id
        self.combo_results: list[dict] = []
        self.run_start_time: float | None = None
        self.current_run_id: str | None = None
        self.current_out_dir: Path | None = None

        self._build_widgets()
        self._refresh_models()
        self._refresh_gt_pages()

    # ------------------------------------------------------------ layout

    def _build_widgets(self):
        left = Frame(self.root, padx=10, pady=10, width=360)
        left.pack(side=LEFT, fill=Y)
        left.pack_propagate(False)
        self._build_left_panel(left)

        right_of_left = Frame(self.root)
        right_of_left.pack(side=LEFT, fill=BOTH, expand=True)

        middle_row = Frame(right_of_left, padx=10, pady=10)
        middle_row.pack(side=TOP, fill=BOTH, expand=True)
        self._build_queue_panel(middle_row)
        self._build_current_run_panel(middle_row)

        self._build_log_panel(right_of_left)
        self._build_report_panel(right_of_left)

    def _build_left_panel(self, parent):
        Label(parent, text="Sweep setup", font=("Segoe UI", 11, "bold")).pack(anchor="w")

        def section(text):
            Label(parent, text=text, font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(10, 2))

        def path_row(label_text, var, allow_dir=True, allow_file=True):
            Label(parent, text=label_text, font=("Segoe UI", 9)).pack(anchor="w")
            row = Frame(parent)
            row.pack(fill=X, pady=(0, 4))
            Entry(row, textvariable=var).pack(side=LEFT, fill=X, expand=True)
            if allow_file:
                Button(row, text="File...", width=6,
                       command=lambda: self._browse(var, "file")).pack(side=LEFT, padx=(2, 0))
            if allow_dir:
                Button(row, text="Folder...", width=7,
                       command=lambda: self._browse(var, "dir")).pack(side=LEFT, padx=(2, 0))

        section("Stage 1 - OCR reading")
        Label(parent, text="Model:", font=("Segoe UI", 9)).pack(anchor="w")
        self.ocr_model_var = StringVar()
        self.ocr_model_menu = ttk.Combobox(parent, textvariable=self.ocr_model_var, state="readonly")
        self.ocr_model_menu.pack(fill=X, pady=(0, 4))
        Label(parent, text="Prompt (blank=default; a FOLDER sweeps its ocr_stage1_*.txt files):",
              font=("Segoe UI", 8), fg="#555", wraplength=340, justify="left").pack(anchor="w")
        self.ocr_prompt_var = StringVar()
        path_row("", self.ocr_prompt_var)

        section("Stage 2 - structuring")
        Label(parent, text="Model:", font=("Segoe UI", 9)).pack(anchor="w")
        self.structure_model_var = StringVar()
        self.structure_model_menu = ttk.Combobox(parent, textvariable=self.structure_model_var, state="readonly")
        self.structure_model_menu.pack(fill=X, pady=(0, 4))
        Label(parent, text="Prompt template (blank=built-in default; a FOLDER sweeps its "
                           "structuring_stage2_*.txt files):",
              font=("Segoe UI", 8), fg="#555", wraplength=340, justify="left").pack(anchor="w")
        self.structuring_prompt_var = StringVar()
        path_row("", self.structuring_prompt_var)

        section("Ground truth")
        self.gt_log_var = StringVar(value=str(DEFAULT_GT_LOG))
        path_row("Log file:", self.gt_log_var, allow_dir=False)
        Button(parent, text="Refresh pages/columns from log", command=self._refresh_gt_pages).pack(
            anchor="w", pady=(0, 4))

        Label(parent, text="Pages (none selected = every page in the log):",
              font=("Segoe UI", 9)).pack(anchor="w")
        pages_frame = Frame(parent)
        pages_frame.pack(fill=X, pady=(0, 4))
        self.pages_listbox = Listbox(pages_frame, selectmode=EXTENDED, height=4, exportselection=False)
        pages_scroll = Scrollbar(pages_frame, orient="vertical", command=self.pages_listbox.yview)
        self.pages_listbox.configure(yscrollcommand=pages_scroll.set)
        self.pages_listbox.pack(side=LEFT, fill=X, expand=True)
        pages_scroll.pack(side=RIGHT, fill=Y)
        self.pages_listbox.bind("<<ListboxSelect>>", self._on_pages_selection_changed)

        Label(parent, text="Columns (none selected = every column in the log):",
              font=("Segoe UI", 9)).pack(anchor="w")
        cols_frame = Frame(parent)
        cols_frame.pack(fill=X, pady=(0, 4))
        self.columns_listbox = Listbox(cols_frame, selectmode=EXTENDED, height=4, exportselection=False)
        cols_scroll = Scrollbar(cols_frame, orient="vertical", command=self.columns_listbox.yview)
        self.columns_listbox.configure(yscrollcommand=cols_scroll.set)
        self.columns_listbox.pack(side=LEFT, fill=X, expand=True)
        cols_scroll.pack(side=RIGHT, fill=Y)

        section("Run options")
        row = Frame(parent)
        row.pack(fill=X, pady=(0, 4))
        Label(row, text="Max rows/page:", width=16, anchor="w").pack(side=LEFT)
        self.max_rows_var = StringVar()
        Entry(row, textvariable=self.max_rows_var, width=8).pack(side=LEFT)

        row = Frame(parent)
        row.pack(fill=X, pady=(0, 4))
        Label(row, text="Run ID:", width=16, anchor="w").pack(side=LEFT)
        self.run_id_var = StringVar()
        Entry(row, textvariable=self.run_id_var).pack(side=LEFT, fill=X, expand=True)

        self.log_file_var = StringVar(value=str(DEFAULT_LOG_FILE))
        path_row("History log (blank = don't log):", self.log_file_var, allow_dir=False)
        self.out_dir_var = StringVar(value=str(DEFAULT_OUT_DIR))
        path_row("Detail output folder:", self.out_dir_var, allow_file=False)

        action_row = Frame(parent)
        action_row.pack(fill=X, pady=(10, 4))
        self.run_button = Button(action_row, text="Run sweep", command=self._on_run,
                                  bg="#4a7", fg="white", font=("Segoe UI", 10, "bold"))
        self.run_button.pack(side=LEFT)
        self.stop_button = Button(action_row, text="Stop", command=self._on_stop, state=DISABLED)
        self.stop_button.pack(side=LEFT, padx=(6, 0))
        Button(action_row, text="Open detail folder", command=self._open_detail_folder).pack(
            side=LEFT, padx=(6, 0))

        Label(parent, text="Command preview:", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(8, 0))
        self.preview_text = Text(parent, height=4, font=("Consolas", 8), wrap="word", bg="#f0f0f0")
        self.preview_text.pack(fill=X)
        self.preview_text.config(state=DISABLED)

    def _build_queue_panel(self, parent):
        col = Frame(parent)
        col.pack(side=LEFT, fill=BOTH, expand=True)
        Label(col, text="Benchmark queue", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.queue_tree = ttk.Treeview(
            col, columns=("ocr", "struct", "status"), show="headings", height=14)
        self.queue_tree.heading("ocr", text="OCR Prompt")
        self.queue_tree.heading("struct", text="Struct Prompt")
        self.queue_tree.heading("status", text="Status")
        self.queue_tree.column("ocr", width=180, anchor="w")
        self.queue_tree.column("struct", width=180, anchor="w")
        self.queue_tree.column("status", width=140, anchor="w")
        self.queue_tree.tag_configure("running", background="#fff3cd")
        self.queue_tree.tag_configure("done", background="#d4edda")
        self.queue_tree.pack(fill=BOTH, expand=True)

    def _build_current_run_panel(self, parent):
        col = Frame(parent, width=280)
        col.pack(side=LEFT, fill=Y, padx=(10, 0))
        col.pack_propagate(False)
        Label(col, text="Current run", font=("Segoe UI", 10, "bold")).pack(anchor="w")

        self.status_label = Label(col, text="Idle", font=("Segoe UI", 9, "bold"), fg="#444")
        self.status_label.pack(anchor="w", pady=(4, 8))

        self.right_labels: dict[str, Label] = {}
        fields = ["OCR Model", "Struct Model", "Stage", "OCR Prompt",
                  "Struct Prompt", "Row / Column", "Progress", "Elapsed"]
        for name in fields:
            Label(col, text=f"{name}:", font=("Segoe UI", 8, "bold"), fg="#333").pack(anchor="w", pady=(4, 0))
            lbl = Label(col, text="-", font=("Segoe UI", 9), wraplength=260, justify="left")
            lbl.pack(anchor="w")
            self.right_labels[name] = lbl

    def _build_log_panel(self, parent):
        frame = Frame(parent, padx=10)
        frame.pack(side=TOP, fill=X)
        Label(frame, text="Log:", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        inner = Frame(frame)
        inner.pack(fill=X)
        self.log_text = Text(inner, height=8, font=("Consolas", 9), wrap="word", bg="#111", fg="#ddd")
        log_scroll = Scrollbar(inner, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side=LEFT, fill=X, expand=True)
        log_scroll.pack(side=RIGHT, fill=Y)

    def _build_report_panel(self, parent):
        frame = Frame(parent, padx=10)
        frame.pack(side=TOP, fill=BOTH, expand=True, pady=(6, 10))
        Label(frame, text="Report (click a column header to sort):",
              font=("Segoe UI", 9, "bold")).pack(anchor="w")
        columns = ("ocr", "struct", "n", "correct", "abstained", "wrong", "avg_time")
        self.report_tree = ttk.Treeview(frame, columns=columns, show="headings", height=8)
        headers = {"ocr": "OCR Prompt", "struct": "Struct Prompt", "n": "N", "correct": "Correct",
                   "abstained": "Abstained", "wrong": "Wrong", "avg_time": "Avg Time"}
        widths = {"ocr": 180, "struct": 180, "n": 50, "correct": 80,
                  "abstained": 80, "wrong": 90, "avg_time": 80}
        for c in columns:
            self.report_tree.heading(c, text=headers[c], command=lambda c=c: self._sort_report_by(c, False))
            self.report_tree.column(c, width=widths[c], anchor="w")
        report_scroll = Scrollbar(frame, orient="vertical", command=self.report_tree.yview)
        self.report_tree.configure(yscrollcommand=report_scroll.set)
        self.report_tree.pack(side=LEFT, fill=BOTH, expand=True)
        report_scroll.pack(side=RIGHT, fill=Y)

    # ---------------------------------------------------------- pickers

    def _refresh_models(self):
        names = list_model_profiles()
        self.ocr_model_menu["values"] = names
        self.structure_model_menu["values"] = names
        if names:
            self.ocr_model_var.set("qwen3vl4b" if "qwen3vl4b" in names else names[0])
            self.structure_model_var.set("qwen3vl4b" if "qwen3vl4b" in names else names[0])

    def _refresh_gt_pages(self):
        gt_log = self.gt_log_var.get().strip()
        pages = discover_gt_pages(gt_log)
        self.pages_listbox.delete(0, END)
        for p in pages:
            self.pages_listbox.insert(END, p)
        self._refresh_gt_columns()

    def _on_pages_selection_changed(self, _event=None):
        self._refresh_gt_columns()

    def _refresh_gt_columns(self):
        gt_log = self.gt_log_var.get().strip()
        selected_pages = [self.pages_listbox.get(i) for i in self.pages_listbox.curselection()]
        columns = discover_gt_columns(gt_log, selected_pages or None)
        self.columns_listbox.delete(0, END)
        for c in columns:
            self.columns_listbox.insert(END, c)

    def _browse(self, var: StringVar, kind: str):
        initial = str(Path(var.get()).parent) if var.get() else str(PROJECT_ROOT / "config" / "prompts")
        if kind == "file":
            path = filedialog.askopenfilename(initialdir=initial, filetypes=[("Text", "*.txt"), ("All", "*.*")])
        else:
            path = filedialog.askdirectory(initialdir=initial)
        if path:
            var.set(path)

    # -------------------------------------------------------------- run

    def _build_command(self) -> list[str] | None:
        ocr_model = self.ocr_model_var.get().strip()
        struct_model = self.structure_model_var.get().strip()
        if not ocr_model or not struct_model:
            messagebox.showwarning("Missing model", "Pick both an OCR model and a structuring model.")
            return None

        cmd = [sys.executable, "-u", str(SWEEP_SCRIPT),
               "--ocr-model", ocr_model, "--structure-model", struct_model]

        ocr_prompt_path = self.ocr_prompt_var.get().strip()
        struct_prompt_path = self.structuring_prompt_var.get().strip()
        if not ocr_prompt_path and not struct_prompt_path:
            messagebox.showwarning(
                "Nothing to sweep",
                "Give an OCR prompt and/or a structuring prompt path (file or folder) - "
                "at least one must be swept, otherwise there's nothing to compare.")
            return None

        for path_str, dir_flag, file_flag, label in (
            (ocr_prompt_path, "--ocr-prompt-dir", "--ocr-prompt-file", "OCR prompt"),
            (struct_prompt_path, "--structuring-prompt-dir", "--structuring-prompt-file", "Structuring prompt"),
        ):
            if not path_str:
                continue
            p = Path(path_str)
            if p.is_dir():
                cmd += [dir_flag, str(p)]
            elif p.is_file():
                cmd += [file_flag, str(p)]
            else:
                messagebox.showerror("Invalid path", f"{label} path does not exist: {p}")
                return None

        for i in self.pages_listbox.curselection():
            cmd += ["--sidecar-path", self.pages_listbox.get(i)]
        for i in self.columns_listbox.curselection():
            cmd += ["--columns", self.columns_listbox.get(i)]

        gt_log = self.gt_log_var.get().strip()
        if gt_log:
            cmd += ["--ground-truth-log", gt_log]

        max_rows = self.max_rows_var.get().strip()
        if max_rows:
            if not max_rows.isdigit():
                messagebox.showerror("Invalid value", "Max rows/page must be a whole number.")
                return None
            cmd += ["--max-rows-per-page", max_rows]

        run_id = self.run_id_var.get().strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        cmd += ["--run-id", run_id]
        cmd += ["--log-file", self.log_file_var.get().strip()]
        out_dir = self.out_dir_var.get().strip()
        if out_dir:
            cmd += ["--out-dir", out_dir]

        self._pending_run_id = run_id
        self._pending_out_dir = Path(out_dir) if out_dir else DEFAULT_OUT_DIR
        return cmd

    def _update_preview(self, cmd: list[str] | None):
        self.preview_text.config(state=NORMAL)
        self.preview_text.delete("1.0", END)
        self.preview_text.insert(END, " ".join(cmd) if cmd else "(fill in required fields)")
        self.preview_text.config(state=DISABLED)

    def _on_run(self):
        if self.process is not None:
            messagebox.showinfo("Already running", "A sweep is already in progress.")
            return
        cmd = self._build_command()
        self._update_preview(cmd)
        if cmd is None:
            return

        self.current_run_id = self._pending_run_id
        self.current_out_dir = self._pending_out_dir / self.current_run_id
        self.combo_results = []
        self.queue_items = {}
        self.queue_tree.delete(*self.queue_tree.get_children())
        self.report_tree.delete(*self.report_tree.get_children())
        self.log_text.delete("1.0", END)
        self.log_text.insert(END, f"$ {' '.join(cmd)}\n\n")
        for lbl in self.right_labels.values():
            lbl.config(text="-")

        self.run_start_time = time.time()
        self.run_button.config(state=DISABLED)
        self.stop_button.config(state=NORMAL)
        self.status_label.config(text="Starting...", fg="#a60")

        thread = threading.Thread(target=self._run_worker, args=(cmd,), daemon=True)
        thread.start()
        self.root.after(100, self._poll_log_queue)
        self.root.after(500, self._tick_elapsed)

    def _on_stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.status_label.config(text="Stopping...", fg="#a60")

    def _open_detail_folder(self):
        _open_folder(self.current_out_dir or DEFAULT_OUT_DIR)

    def _run_worker(self, cmd: list[str]):
        try:
            self.process = subprocess.Popen(
                cmd, cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, encoding="utf-8", errors="replace",
            )
            for line in self.process.stdout:
                self.log_queue.put(("line", line))
            exit_code = self.process.wait()
            self.log_queue.put(("done", exit_code))
        except Exception as e:
            self.log_queue.put(("error", str(e)))

    # ------------------------------------------------------- log polling

    def _tick_elapsed(self):
        if self.process is None or self.process.poll() is not None:
            return
        if self.run_start_time is not None:
            self.right_labels["Elapsed"].config(text=f"{time.time() - self.run_start_time:.0f}s")
        self.root.after(500, self._tick_elapsed)

    def _poll_log_queue(self):
        try:
            while True:
                kind, payload = self.log_queue.get_nowait()
                if kind == "line":
                    if payload.startswith(PROGRESS_PREFIX):
                        self._handle_progress_line(payload[len(PROGRESS_PREFIX):])
                    else:
                        self.log_text.insert(END, payload)
                        self.log_text.see(END)
                elif kind == "done":
                    self._on_process_done(payload)
                    return
                elif kind == "error":
                    self.log_text.insert(END, f"\n[ERROR launching process: {payload}]\n")
                    self.status_label.config(text="Error", fg="#c00")
                    self.run_button.config(state=NORMAL)
                    self.stop_button.config(state=DISABLED)
                    self.process = None
                    return
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log_queue)

    def _on_process_done(self, exit_code: int):
        self.process = None
        self.run_button.config(state=NORMAL)
        self.stop_button.config(state=DISABLED)
        if exit_code == 0:
            self.status_label.config(text="Done", fg="#080")
            self._render_report()
        else:
            self.status_label.config(text=f"Failed (exit {exit_code})", fg="#c00")

    def _handle_progress_line(self, raw_json: str):
        try:
            event = json.loads(raw_json)
        except json.JSONDecodeError:
            return
        kind = event.get("event")

        if kind == "plan":
            self.right_labels["OCR Model"].config(text=event.get("ocr_model", "-"))
            self.right_labels["Struct Model"].config(text=event.get("structuring_model", "-"))
            ocr_variants = event.get("ocr_variants", [])
            struct_variants = event.get("structuring_variants", [])
            n = event.get("n_examples", 0)
            self.status_label.config(
                text=f"Running - {n} example(s), {len(ocr_variants)}x{len(struct_variants)} combo(s)",
                fg="#a60")
            for ocr_name in ocr_variants:
                for struct_name in struct_variants:
                    item = self.queue_tree.insert("", END, values=(ocr_name, struct_name, "Pending"))
                    self.queue_items[(ocr_name, struct_name)] = item

        elif kind == "model_load_start":
            self.right_labels["Stage"].config(text=f"Loading {event['stage']} model ({event['model']})...")
        elif kind == "model_load_done":
            self.right_labels["Stage"].config(text=f"{event['stage']} model loaded")

        elif kind == "stage1_variant_start":
            self.right_labels["Stage"].config(text="Stage 1 - OCR reading")
            self.right_labels["OCR Prompt"].config(text=event["ocr_prompt"])
            self.right_labels["Struct Prompt"].config(text="-")
        elif kind == "stage1_step":
            self.right_labels["Row / Column"].config(text=f"row {event['row_index']} [{event['column']}]")
            self.right_labels["Progress"].config(text=f"{event['i']} / {event['n']}")

        elif kind == "combo_start":
            self.right_labels["Stage"].config(text="Stage 2 - structuring")
            self.right_labels["OCR Prompt"].config(text=event["ocr_prompt"])
            self.right_labels["Struct Prompt"].config(text=event["structuring_prompt"])
            key = (event["ocr_prompt"], event["structuring_prompt"])
            item = self.queue_items.get(key)
            if item:
                self.queue_tree.set(item, "status", "Running...")
                self.queue_tree.item(item, tags=("running",))
        elif kind == "stage2_step":
            self.right_labels["Row / Column"].config(text=f"row {event['row_index']} [{event['column']}]")
            self.right_labels["Progress"].config(text=f"{event['i']} / {event['n']}")
        elif kind == "combo_done":
            self.combo_results.append(event)
            key = (event["ocr_prompt"], event["structuring_prompt"])
            item = self.queue_items.get(key)
            if item:
                total = event.get("overall_total", 0)
                correct = event.get("overall_correct", 0)
                self.queue_tree.set(item, "status", f"Done ({correct}/{total})")
                self.queue_tree.item(item, tags=("done",))

        elif kind == "error":
            self.log_text.insert(END, f"\n[ERROR] {event.get('message', '?')}\n")
            self.log_text.see(END)

    # ------------------------------------------------------------ report

    def _render_report(self):
        self.report_tree.delete(*self.report_tree.get_children())
        for r in self.combo_results:
            total = r.get("overall_total", 0)
            correct_rate = _rate(r.get("overall_correct", 0), total)
            abstained_rate = _rate(r.get("abstained", 0), total)
            wrong_rate = _rate(r.get("wrong", 0), total)
            self.report_tree.insert("", END, values=(
                r["ocr_prompt"], r["structuring_prompt"], total,
                f"{correct_rate:.0%}", f"{abstained_rate:.0%}", f"{wrong_rate:.0%}",
                f"{r.get('avg_total_seconds', 0):.1f}s",
            ))

    def _sort_report_by(self, col: str, reverse: bool):
        items = [(self.report_tree.set(k, col), k) for k in self.report_tree.get_children("")]

        def sort_key(pair):
            value = pair[0]
            if value.endswith("%"):
                try:
                    return float(value[:-1])
                except ValueError:
                    return value
            if value.endswith("s") and value[:-1].replace(".", "", 1).isdigit():
                return float(value[:-1])
            if value.isdigit():
                return int(value)
            return value

        items.sort(key=sort_key, reverse=reverse)
        for index, (_, k) in enumerate(items):
            self.report_tree.move(k, "", index)
        self.report_tree.heading(col, command=lambda: self._sort_report_by(col, not reverse))


def main():
    root = Tk()
    PromptSweepApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
