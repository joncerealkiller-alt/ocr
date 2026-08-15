"""
Benchmark tab - repeatable, predefined model tests against the shared
config-driven model registry (2026-08-15).

Reuses, does not duplicate:
  - model list: model_console.adapter.list_console_models() (same
    config/pipeline.yaml console.allowed_models the Chat tab uses - no
    second model whitelist).
  - inference/telemetry: benchmark.console_runner.run_suite(), which
    itself only calls ChatBackendAdapter.ensure_loaded()/send_turn() -
    the exact same calls chat_tab.py's _worker() makes for a normal chat
    turn. This tab adds zero new low-level model-calling code.
  - threading pattern: background worker thread + queue.Queue +
    root.after() polling, identical shape to chat_tab.py's
    _on_send()/_worker()/_poll_result_queue() (see that file's own
    module docstring on why - Tk must never block on a multi-second/
    -minute model call).

A suite selection determines model eligibility (task #1) - the model
dropdown is rebuilt every time the suite changes, filtered through
benchmark.console_runner.eligible_for_suite() on top of the same
enabled-models list the Chat tab already reads. This tab never
maintains its own model list or capability logic.
"""

from __future__ import annotations

import queue
import threading
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from tkinter import (
    Frame, Label, Button, StringVar, Text, Scrollbar, PanedWindow,
    END, DISABLED, NORMAL, WORD, BOTH, X, Y, LEFT, RIGHT, TOP, BOTTOM, VERTICAL, HORIZONTAL,
    messagebox,
)
from tkinter import ttk
from PIL import Image, ImageTk

from benchmark.console_runner import eligible_for_suite, run_suite
from benchmark.run_result import BenchmarkRunResult
from benchmark.suite_schema import BenchmarkSuite, list_suites, PROJECT_ROOT as SUITE_PROJECT_ROOT
from model_console import benchmark_store
from model_console.adapter import ChatBackendAdapter, list_console_models

STATUS_COLORS = {
    "pass": "#1a7f37", "fail": "#b42318", "error": "#b42318",
    "unscored": "#6b6b6b", "cancelled": "#a86a00",
}

THUMBNAIL_SIZE = (220, 220)


class BenchmarkTab(Frame):
    def __init__(self, master, adapter: Optional[ChatBackendAdapter] = None):
        super().__init__(master)
        # A DEDICATED adapter/residency user, same as ChatTab's own
        # self.adapter - deliberately not shared with ChatTab's instance
        # even though both ultimately route through the same process-
        # wide core.model_residency.residency singleton, so a benchmark
        # run and a manual chat turn can never be confused about whose
        # in-flight request is whose. Passing an existing adapter in is
        # supported for tests/composition; app.py doesn't need to.
        self.adapter = adapter or ChatBackendAdapter()

        self._suites: list[BenchmarkSuite] = list_suites()
        self._suites_by_label: dict[str, BenchmarkSuite] = {
            f"{s.display_name} ({s.qualified_id})": s for s in self._suites
        }
        self._result_queue: "queue.Queue" = queue.Queue()
        self._cancel_event = threading.Event()
        self._running = False
        self._current_run: Optional[BenchmarkRunResult] = None
        self._current_case_results_by_id: dict[str, object] = {}
        self._thumbnail_ref = None  # keep a live reference so Tk doesn't GC the PhotoImage

        self._build_ui()
        self._on_suite_change()
        self._refresh_history()

    # ---------------------------------------------------------------- UI

    def _build_ui(self) -> None:
        top = Frame(self)
        top.pack(side=TOP, fill=X, padx=8, pady=6)

        Label(top, text="Suite:").grid(row=0, column=0, sticky="w")
        self.suite_var = StringVar(value=next(iter(self._suites_by_label), ""))
        self.suite_menu = ttk.Combobox(
            top, textvariable=self.suite_var, values=list(self._suites_by_label),
            state="readonly", width=40,
        )
        self.suite_menu.grid(row=0, column=1, sticky="w", padx=(4, 16))
        self.suite_menu.bind("<<ComboboxSelected>>", lambda e: self._on_suite_change())

        Label(top, text="Model:").grid(row=0, column=2, sticky="w")
        self.model_var = StringVar()
        self.model_menu = ttk.Combobox(top, textvariable=self.model_var, state="readonly", width=28)
        self.model_menu.grid(row=0, column=3, sticky="w", padx=(4, 16))

        self.run_button = Button(top, text="Run", command=self._on_run, width=10)
        self.run_button.grid(row=0, column=4, padx=(0, 4))
        self.cancel_button = Button(top, text="Cancel", command=self._on_cancel, width=10, state=DISABLED)
        self.cancel_button.grid(row=0, column=5)

        self.suite_desc_label = Label(self, text="", justify=LEFT, anchor="w", wraplength=900, fg="#444")
        self.suite_desc_label.pack(side=TOP, fill=X, padx=8)

        self.progress_label = Label(self, text="", anchor="w")
        self.progress_label.pack(side=TOP, fill=X, padx=8, pady=(2, 6))

        self.summary_label = Label(self, text="", anchor="w", justify=LEFT)
        self.summary_label.pack(side=TOP, fill=X, padx=8, pady=(0, 6))

        main = PanedWindow(self, orient=HORIZONTAL, sashwidth=6)
        main.pack(side=TOP, fill=BOTH, expand=True, padx=8, pady=(0, 8))

        # Left: results table for the current/selected run.
        left = Frame(main)
        columns = ("case_id", "category", "status", "score", "runtime")
        self.results_tree = ttk.Treeview(left, columns=columns, show="headings", height=14)
        for col, width in zip(columns, (170, 130, 80, 70, 80)):
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=width, stretch=(col == "case_id"))
        self.results_tree.pack(side=TOP, fill=BOTH, expand=True)
        self.results_tree.bind("<<TreeviewSelect>>", lambda e: self._on_case_selected())
        main.add(left, minsize=420)

        # Right: detail pane for the selected case.
        right = Frame(main)
        self.detail_image_label = Label(right)
        self.detail_image_label.pack(side=TOP, anchor="w", pady=(0, 4))
        self.detail_text = Text(right, wrap=WORD, height=20, state=DISABLED)
        detail_scroll = Scrollbar(right, command=self.detail_text.yview)
        self.detail_text.configure(yscrollcommand=detail_scroll.set)
        self.detail_text.pack(side=LEFT, fill=BOTH, expand=True)
        detail_scroll.pack(side=RIGHT, fill=Y)
        main.add(right, minsize=380)

        # Bottom: run history + compare.
        bottom = Frame(self)
        bottom.pack(side=BOTTOM, fill=X, padx=8, pady=(0, 8))
        Label(bottom, text="Run history:").pack(side=TOP, anchor="w")
        hist_columns = ("run_id", "model_name", "suite_qualified_id", "status", "started_at", "score")
        self.history_tree = ttk.Treeview(bottom, columns=hist_columns, show="headings", height=6)
        for col, width in zip(hist_columns, (170, 140, 170, 90, 170, 110)):
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=width)
        self.history_tree.pack(side=TOP, fill=X)
        self.history_tree.bind("<<TreeviewSelect>>", lambda e: self._on_history_selected())

        compare_row = Frame(bottom)
        compare_row.pack(side=TOP, fill=X, pady=(4, 0))
        Button(compare_row, text="Compare selected two runs", command=self._on_compare).pack(side=LEFT)
        self.compare_label = Label(compare_row, text="", anchor="w", fg="#444")
        self.compare_label.pack(side=LEFT, padx=(10, 0))

    # ---------------------------------------------------------- suite/model

    def _on_suite_change(self) -> None:
        suite = self._suites_by_label.get(self.suite_var.get())
        if suite is None:
            self.suite_desc_label.config(text="No benchmark suites found under config/benchmark_suites/.")
            self.model_menu.configure(values=[])
            self.model_var.set("")
            return
        self.suite_desc_label.config(
            text=f"{suite.description.strip()}  [required_capability={suite.required_capability}, "
                 f"{len(suite.cases)} cases]"
        )
        eligible = []
        for name in list_console_models():
            ok, _ = eligible_for_suite(name, suite)
            if ok:
                eligible.append(name)
        self.model_menu.configure(values=eligible)
        if eligible:
            self.model_var.set(eligible[0])
        else:
            self.model_var.set("")
            self.suite_desc_label.config(
                text=self.suite_desc_label.cget("text") +
                     "  (no enabled console model currently meets this suite's capability requirement)"
            )

    # ------------------------------------------------------------------ run

    def _on_run(self) -> None:
        if self._running:
            return
        suite = self._suites_by_label.get(self.suite_var.get())
        model_name = self.model_var.get()
        if suite is None or not model_name:
            messagebox.showwarning("Benchmark", "Pick a suite and a model first.")
            return
        ok, reason = eligible_for_suite(model_name, suite)
        if not ok:
            messagebox.showerror("Invalid configuration", reason)
            return

        self._running = True
        self._cancel_event.clear()
        self.run_button.config(state=DISABLED, text="Running...")
        self.cancel_button.config(state=NORMAL)
        self.progress_label.config(text=f"Starting {suite.qualified_id} on {model_name}...")
        self._clear_results()

        thread = threading.Thread(target=self._worker, args=(model_name, suite), daemon=True)
        thread.start()
        self.after(150, self._poll_result_queue)

    def _on_cancel(self) -> None:
        if self._running:
            self._cancel_event.set()
            self.cancel_button.config(state=DISABLED, text="Cancelling...")

    def _worker(self, model_name: str, suite: BenchmarkSuite) -> None:
        def on_progress(index: int, total: int, case_id: str) -> None:
            self._result_queue.put(("progress", (index, total, case_id), None))

        try:
            run = run_suite(
                self.adapter, model_name, suite,
                on_progress=on_progress,
                cancel_check=self._cancel_event.is_set,
            )
            benchmark_store.save_run(run)
            self._result_queue.put(("done", run, None))
        except Exception as e:
            self._result_queue.put(("error", str(e), traceback.format_exc()))

    def _poll_result_queue(self) -> None:
        try:
            kind, payload, extra = self._result_queue.get_nowait()
        except queue.Empty:
            if self._running:
                self.after(150, self._poll_result_queue)
            return

        if kind == "progress":
            index, total, case_id = payload
            self.progress_label.config(text=f"Running case {index + 1}/{total}: {case_id}")
            self.after(150, self._poll_result_queue)
            return

        self._running = False
        self.run_button.config(state=NORMAL, text="Run")
        self.cancel_button.config(state=DISABLED, text="Cancel")

        if kind == "error":
            self.progress_label.config(text="Run failed to start.")
            messagebox.showerror("Benchmark error", f"{payload}\n\n{extra}")
            return

        run: BenchmarkRunResult = payload
        self._show_run(run)
        self._refresh_history()
        if run.status != "completed":
            self.progress_label.config(text=f"Run {run.status}: {run.abort_reason or ''}")
        else:
            self.progress_label.config(text=f"Run {run.run_id} complete.")

    # -------------------------------------------------------------- display

    def _clear_results(self) -> None:
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        self._current_case_results_by_id = {}
        self._set_detail_text("")
        self.detail_image_label.config(image="")
        self.summary_label.config(text="")

    def _show_run(self, run: BenchmarkRunResult) -> None:
        self._current_run = run
        self._clear_results()
        for case in run.case_results:
            self._current_case_results_by_id[case.case_id] = case
            score_display = f"{case.score:.2f}" if case.score is not None else "-"
            item = self.results_tree.insert(
                "", END,
                values=(case.case_id, case.category, case.status, score_display,
                        f"{case.runtime_seconds:.2f}s" if case.runtime_seconds else "-"),
            )
            self.results_tree.tag_configure(case.status, foreground=STATUS_COLORS.get(case.status, "black"))
            self.results_tree.item(item, tags=(case.status,))

        counts = run.counts
        parts = [f"{k}={v}" for k, v in counts.items() if v]
        pass_rate = run.scored_pass_rate
        pass_rate_str = f"{pass_rate * 100:.0f}%" if pass_rate is not None else "n/a"
        tps = run.mean_tokens_per_sec
        vram = run.peak_vram_mb
        self.summary_label.config(text=(
            f"Model: {run.model_name}   Suite: {run.suite_qualified_id}   "
            f"Status: {run.status}   Counts: {', '.join(parts) or 'none'}   "
            f"Scored pass rate: {pass_rate_str}   "
            f"Runtime: {run.total_runtime_seconds:.1f}s   "
            f"Mean tok/s: {f'{tps:.1f}' if tps else 'n/a'}   "
            f"Peak VRAM: {f'{vram:.0f}MB' if vram else 'n/a'}   "
            f"Backend: {run.backend}   "
            f"Model already resident: {run.was_resident_before_run}"
        ))
        if run.case_results:
            self.results_tree.selection_set(self.results_tree.get_children()[0])
            self._on_case_selected()

    def _on_case_selected(self) -> None:
        selection = self.results_tree.selection()
        if not selection:
            return
        case_id = self.results_tree.item(selection[0], "values")[0]
        case = self._current_case_results_by_id.get(case_id)
        if case is None:
            return

        lines = [
            f"case_id: {case.case_id}",
            f"category: {case.category}",
            f"status: {case.status}    score: {case.score}",
            "",
            "prompt:",
            case.prompt,
            "",
            f"expected: {case.expected_display}",
            "",
            "raw_output:",
            case.raw_output if case.raw_output is not None else "(none - inference failed)",
            "",
            "explanation:",
            case.explanation,
        ]
        if case.error:
            lines += ["", "error:", case.error]
        if case.telemetry:
            lines += ["", "telemetry:", _format_telemetry(case.telemetry)]
        self._set_detail_text("\n".join(lines))

        self.detail_image_label.config(image="")
        self._thumbnail_ref = None
        if case.image_path:
            try:
                image_path = SUITE_PROJECT_ROOT / case.image_path
                img = Image.open(image_path)
                img.thumbnail(THUMBNAIL_SIZE)
                self._thumbnail_ref = ImageTk.PhotoImage(img)
                self.detail_image_label.config(image=self._thumbnail_ref)
            except Exception:
                pass  # thumbnail is a display nicety, never blocks showing the rest of the case

    def _set_detail_text(self, text: str) -> None:
        self.detail_text.config(state=NORMAL)
        self.detail_text.delete("1.0", END)
        self.detail_text.insert("1.0", text)
        self.detail_text.config(state=DISABLED)

    # -------------------------------------------------------------- history

    def _refresh_history(self) -> None:
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        for summary in benchmark_store.list_runs():
            counts = summary.get("counts") or {}
            scored = counts.get("pass", 0) + counts.get("fail", 0)
            score_display = f"{counts.get('pass', 0)}/{scored}" if scored else "n/a"
            self.history_tree.insert("", END, iid=summary["run_id"], values=(
                summary["run_id"], summary["model_name"], summary["suite_qualified_id"],
                summary["status"], summary["started_at"], score_display,
            ))

    def _on_history_selected(self) -> None:
        selection = self.history_tree.selection()
        if not selection:
            return
        run = benchmark_store.load_run(selection[0])
        if run is not None:
            self._show_run(run)
            self.progress_label.config(text=f"Viewing saved run {run.run_id} (read-only).")

    def _on_compare(self) -> None:
        selection = self.history_tree.selection()
        if len(selection) != 2:
            messagebox.showinfo("Compare", "Ctrl/Shift-click exactly two runs in the history table to compare.")
            return
        run_a = benchmark_store.load_run(selection[0])
        run_b = benchmark_store.load_run(selection[1])
        if run_a is None or run_b is None:
            messagebox.showerror("Compare", "Could not load one or both selected runs.")
            return
        if run_a.suite_qualified_id != run_b.suite_qualified_id:
            self.compare_label.config(
                text=f"NOT COMPARABLE - different suite versions "
                     f"({run_a.suite_qualified_id} vs {run_b.suite_qualified_id})",
                fg="#b42318",
            )
            return

        counts_a, counts_b = run_a.counts, run_b.counts
        rate_a = run_a.scored_pass_rate
        rate_b = run_b.scored_pass_rate
        tps_a, tps_b = run_a.mean_tokens_per_sec, run_b.mean_tokens_per_sec
        vram_a, vram_b = run_a.peak_vram_mb, run_b.peak_vram_mb
        self.compare_label.config(fg="#1a4d2e", text=(
            f"{run_a.suite_qualified_id}  |  "
            f"{run_a.model_name}: pass={counts_a.get('pass', 0)}/fail={counts_a.get('fail', 0)} "
            f"rate={f'{rate_a*100:.0f}%' if rate_a is not None else 'n/a'} "
            f"tok/s={f'{tps_a:.1f}' if tps_a else 'n/a'} vram={f'{vram_a:.0f}MB' if vram_a else 'n/a'}  "
            f"vs  "
            f"{run_b.model_name}: pass={counts_b.get('pass', 0)}/fail={counts_b.get('fail', 0)} "
            f"rate={f'{rate_b*100:.0f}%' if rate_b is not None else 'n/a'} "
            f"tok/s={f'{tps_b:.1f}' if tps_b else 'n/a'} vram={f'{vram_b:.0f}MB' if vram_b else 'n/a'}"
        ))


def _format_telemetry(telemetry: dict) -> str:
    lines = []
    for phase in ("load", "generate"):
        block = telemetry.get(phase)
        if block:
            lines.append(f"  {phase}: " + ", ".join(f"{k}={v}" for k, v in block.items()))
    return "\n".join(lines) if lines else "  (none captured)"
