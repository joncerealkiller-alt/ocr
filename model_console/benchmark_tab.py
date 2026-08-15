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

from benchmark.console_runner import (
    eligible_for_suite, find_cross_engine_pairs, inference_engine_for_model,
    required_backend_transport, run_suite,
)
from benchmark.run_result import BenchmarkRunResult
from benchmark.suite_schema import BenchmarkSuite, list_suites, PROJECT_ROOT as SUITE_PROJECT_ROOT
from model_console import benchmark_store
from model_console.adapter import ChatBackendAdapter, list_console_models

# Generation-setting keys worth flagging in a comparison if they differ
# between two runs' generation_config_snapshot (task #11: "do not permit
# an apparently apples-to-apples comparison without clearly warning when
# ... effective generation settings materially differ"). Deliberately a
# curated subset, not every GenerationConfig field - fields like
# `extra`/`prompt_text` differ harmlessly by construction and would just
# add noise to every comparison.
_MATERIAL_SETTING_KEYS = [
    "do_sample", "temperature", "top_p", "top_k", "repetition_penalty",
    "no_repeat_ngram_size", "max_new_tokens", "min_pixels", "max_pixels",
    "context_length", "runtime",
]

STATUS_COLORS = {
    "pass": "#1a7f37", "fail": "#b42318", "error": "#b42318",
    "unscored": "#6b6b6b", "cancelled": "#a86a00",
}

THUMBNAIL_SIZE = (220, 220)


class BenchmarkTab(Frame):
    def __init__(self, master):
        super().__init__(master)
        # No single self.adapter (removed 2026-08-16 - was a real bug
        # source, see _worker()/_backend_comparison_worker()'s own
        # comments): a fixed-transport adapter shared across every run
        # is WRONG the moment two runs in the same session need
        # different backends (transformers=local vs vllm=remote) - a
        # real error hit live the first time this tab ran a vllm-runtime
        # model. Each run/leg now builds its OWN adapter via
        # required_backend_transport(model_name), a dedicated instance
        # per call rather than one shared across the tab's lifetime -
        # still never shared with ChatTab's own adapter (same "never
        # confuse whose in-flight request is whose" reasoning as before,
        # just per-call now instead of per-tab).

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
        self.model_menu.bind("<<ComboboxSelected>>", lambda e: self._on_model_change())

        # Engine is READ-ONLY, derived from the selected model's own
        # config.runtime (task #4/#11) - not a picker, because no single
        # registered model entry supports being run under both engines
        # today (see console_runner.inference_engine_for_model()'s
        # docstring). Offering a toggle here would imply a combination
        # that doesn't exist for ~95% of models.
        self.engine_label = Label(top, text="", width=14, fg="#444")
        self.engine_label.grid(row=0, column=4, sticky="w", padx=(0, 12))

        self.run_button = Button(top, text="Run", command=self._on_run, width=10)
        self.run_button.grid(row=0, column=5, padx=(0, 4))
        self.cancel_button = Button(top, text="Cancel", command=self._on_cancel, width=10, state=DISABLED)
        self.cancel_button.grid(row=0, column=6, padx=(0, 12))

        self.backend_compare_button = Button(
            top, text="Run Backend Comparison", command=self._on_run_backend_comparison,
        )
        self.backend_compare_button.grid(row=0, column=7, padx=(0, 12))

        # Real gap found live (2026-08-16): nothing in this tab ever
        # released a model after a run - Chat tab has an explicit Eject
        # button (chat_tab.py's _on_eject()) plus release-on-close
        # (shutdown()); this tab had neither, so a completed run (single
        # OR the last leg of a backend comparison) left the model/vLLM
        # subprocess resident indefinitely, confirmed live as a vLLM
        # server still holding ~16GB after a finished run. Mirrors that
        # same convention now - see _on_eject()/shutdown() below.
        self.eject_button = Button(top, text="Eject Model", command=self._on_eject)
        self.eject_button.grid(row=0, column=8)

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
        Label(bottom, text="Run history (ctrl/shift-click two rows, then Compare):").pack(side=TOP, anchor="w")
        hist_columns = ("run_id", "model_name", "engine", "suite_qualified_id", "status", "started_at", "score")
        self.history_tree = ttk.Treeview(bottom, columns=hist_columns, show="headings", height=6, selectmode="extended")
        for col, width in zip(hist_columns, (170, 140, 90, 170, 90, 170, 110)):
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=width)
        self.history_tree.pack(side=TOP, fill=X)
        self.history_tree.bind("<<TreeviewSelect>>", lambda e: self._on_history_selected())

        compare_row = Frame(bottom)
        compare_row.pack(side=TOP, fill=X, pady=(4, 0))
        Button(compare_row, text="Compare selected two runs", command=self._on_compare).pack(side=LEFT)

        self.compare_text = Text(bottom, wrap=WORD, height=8, state=DISABLED, font=("Consolas", 9))
        self.compare_text.pack(side=TOP, fill=X, pady=(4, 0))

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
        self._on_model_change()

    def _on_model_change(self) -> None:
        model_name = self.model_var.get()
        if not model_name:
            self.engine_label.config(text="")
            self.backend_compare_button.config(state=DISABLED)
            return
        engine = inference_engine_for_model(model_name)
        transport = required_backend_transport(model_name)
        self.engine_label.config(text=f"engine: {engine} ({transport})")
        pair = self._find_pair_for(model_name)
        self.backend_compare_button.config(state=(NORMAL if pair else DISABLED))

    def _find_pair_for(self, model_name: str) -> Optional[str]:
        """The cross-engine partner for model_name, if the registry has
        one (see console_runner.find_cross_engine_pairs()) - None means
        no weight-identical counterpart is registered under the other
        engine, so "Run Backend Comparison" has nothing valid to do."""
        for a, b in find_cross_engine_pairs():
            if a == model_name:
                return b
            if b == model_name:
                return a
        return None

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
        self.backend_compare_button.config(state=DISABLED)
        self.cancel_button.config(state=NORMAL)
        self.progress_label.config(text=f"Starting {suite.qualified_id} on {model_name}...")
        self._clear_results()

        thread = threading.Thread(target=self._worker, args=(model_name, suite), daemon=True)
        thread.start()
        self.after(150, self._poll_result_queue)

    def _on_run_backend_comparison(self) -> None:
        """
        Task #12: run the currently selected model's suite under
        transformers, release, run its cross-engine partner under vLLM,
        release, persist both, show the comparison - sequentially,
        never concurrently on the same GPU (each leg goes through the
        normal ensure_loaded() -> residency/vllm_runtime.acquire() path,
        which already claims the GPU exclusively via gpu_coordinator and
        respects core/resource_guard.py's pre-flight check; this method
        adds an explicit release() between legs on top of that so the
        second leg never finds the first leg's model still resident).
        """
        if self._running:
            return
        suite = self._suites_by_label.get(self.suite_var.get())
        model_name = self.model_var.get()
        if suite is None or not model_name:
            messagebox.showwarning("Benchmark", "Pick a suite and a model first.")
            return
        partner = self._find_pair_for(model_name)
        if partner is None:
            messagebox.showinfo(
                "Run Backend Comparison",
                f"{model_name!r} has no registered cross-engine partner (same repo_id, "
                "different config.runtime) - see console_runner.find_cross_engine_pairs(). "
                "Only one such pair exists in this registry today: gemma_12b_unified "
                "(transformers) / gemma_12b_w4a16 (vllm).",
            )
            return
        ok, reason = eligible_for_suite(partner, suite)
        if not ok:
            messagebox.showerror("Invalid configuration", f"Partner model {partner!r} is not eligible: {reason}")
            return

        self._running = True
        self._cancel_event.clear()
        self.run_button.config(state=DISABLED)
        self.backend_compare_button.config(state=DISABLED, text="Running comparison...")
        self.cancel_button.config(state=NORMAL)
        self.progress_label.config(text=f"Backend comparison: {model_name} vs {partner} on {suite.qualified_id}...")
        self._clear_results()

        thread = threading.Thread(
            target=self._backend_comparison_worker, args=(model_name, partner, suite), daemon=True,
        )
        thread.start()
        self.after(150, self._poll_result_queue)

    def _backend_comparison_worker(self, model_a: str, model_b: str, suite: BenchmarkSuite) -> None:
        """
        Real bug fixed 2026-08-16 (found live: 'gemma_extract_vllm
        requires backend=remote, but the given adapter is backend=local'):
        this used to run BOTH legs through self.adapter, a single
        fixed-transport instance created once in __init__ - correct for
        a same-engine comparison, WRONG the moment the two legs need
        different transports (exactly the vLLM-vs-transformers case this
        button exists for). Each leg now gets its OWN adapter, built for
        that specific model's required_backend_transport() - the same
        logic run_suite_auto() already encapsulates, inlined here (not
        called directly) only so this method keeps a handle to call
        .release() on the correct adapter between legs.
        """
        def on_progress(index: int, total: int, case_id: str) -> None:
            self._result_queue.put(("progress", (index, total, f"[{model_a}] {case_id}"), None))

        try:
            adapter_a = ChatBackendAdapter(backend=required_backend_transport(model_a))
            run_a = run_suite(adapter_a, model_a, suite, on_progress=on_progress,
                               cancel_check=self._cancel_event.is_set)
            benchmark_store.save_run(run_a)
            if run_a.status == "cancelled" or self._cancel_event.is_set():
                self._result_queue.put(("batch_partial", run_a, None))
                return

            self._result_queue.put(("progress", (0, 1, f"releasing {model_a} before {model_b}..."), None))
            adapter_a.release()

            def on_progress_b(index: int, total: int, case_id: str) -> None:
                self._result_queue.put(("progress", (index, total, f"[{model_b}] {case_id}"), None))

            adapter_b = ChatBackendAdapter(backend=required_backend_transport(model_b))
            run_b = run_suite(adapter_b, model_b, suite, on_progress=on_progress_b,
                               cancel_check=self._cancel_event.is_set)
            benchmark_store.save_run(run_b)
            self._result_queue.put(("batch_done", (run_a, run_b), None))
        except Exception as e:
            self._result_queue.put(("error", str(e), traceback.format_exc()))

    def _on_cancel(self) -> None:
        if self._running:
            self._cancel_event.set()
            self.cancel_button.config(state=DISABLED, text="Cancelling...")

    def _on_eject(self) -> None:
        """
        Releases whatever is resident on EITHER transport, unconditionally
        - this tab has no single persistent adapter to ask "is anything
        loaded?" the way chat_tab.py's _on_eject() can (each run/suite
        builds its own short-lived adapter for whichever transport that
        model needs, see _worker()/_backend_comparison_worker()), so
        there's no single instance whose .resident_model_name is
        trustworthy here. Releasing both is a safe no-op wherever nothing
        is actually resident - matches core/model_residency.py's
        release_all() and core/vllm_runtime.py's release(), both already
        idempotent no-ops when empty.
        """
        if self._running:
            messagebox.showinfo("Busy", "Can't eject while a benchmark run is in progress.")
            return
        self.progress_label.config(text="Ejecting (releasing local + remote residency)...")
        self.update_idletasks()
        errors = []
        for transport in ("local", "remote"):
            try:
                ChatBackendAdapter(backend=transport).release()
            except Exception as e:
                errors.append(f"{transport}: {type(e).__name__}: {e}")
        if errors:
            self.progress_label.config(text="Eject completed with errors: " + "; ".join(errors))
        else:
            self.progress_label.config(text="Ejected - local and remote residency both released.")

    def shutdown(self) -> None:
        """Called on window close (see model_console/app.py's on_close())
        - same "don't leave GPU memory held by a closed window" discipline
        as ChatTab.shutdown(). Swallows errors (the app is closing either
        way; a release failure here shouldn't block that)."""
        for transport in ("local", "remote"):
            try:
                ChatBackendAdapter(backend=transport).release()
            except Exception:
                pass

    def _worker(self, model_name: str, suite: BenchmarkSuite) -> None:
        """
        Same fix as _backend_comparison_worker (2026-08-16): builds an
        adapter for model_name's OWN required_backend_transport() rather
        than reusing self.adapter's fixed transport - a plain single Run
        against a vllm-only model (e.g. gemma_extract_vllm) hit the exact
        same 'requires backend=remote, given adapter is backend=local'
        error this fixes for the comparison path.
        """
        def on_progress(index: int, total: int, case_id: str) -> None:
            self._result_queue.put(("progress", (index, total, case_id), None))

        try:
            adapter = ChatBackendAdapter(backend=required_backend_transport(model_name))
            run = run_suite(
                adapter, model_name, suite,
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
        self.backend_compare_button.config(text="Run Backend Comparison")
        self.cancel_button.config(state=DISABLED, text="Cancel")
        self._on_model_change()  # re-enables backend_compare_button iff a pair still exists

        if kind == "error":
            self.progress_label.config(text="Run failed to start.")
            messagebox.showerror("Benchmark error", f"{payload}\n\n{extra}")
            return

        if kind == "batch_partial":
            run_a: BenchmarkRunResult = payload
            self._show_run(run_a)
            self._refresh_history()
            self.progress_label.config(text=f"Backend comparison cancelled after {run_a.model_name} ({run_a.status}).")
            return

        if kind == "batch_done":
            run_a, run_b = payload
            self._refresh_history()
            self._show_run(run_b)
            self._render_compare(run_a, run_b)
            self.progress_label.config(
                text=f"Backend comparison complete: {run_a.model_name} ({run_a.inference_engine}) "
                     f"vs {run_b.model_name} ({run_b.inference_engine}). See comparison table below history."
            )
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
        resident_vram = run.resident_vram_mb
        load_s = (run.load_telemetry or {}).get("load_time_s")
        cold_warm = "cold (loaded this run)" if load_s is not None else (
            "warm (already resident)" if run.was_resident_before_run else "unknown"
        )
        self.summary_label.config(text=(
            f"Model: {run.model_name}   Engine: {run.inference_engine}   "
            f"Suite: {run.suite_qualified_id}   "
            f"Status: {run.status}   Counts: {', '.join(parts) or 'none'}   "
            f"Scored pass rate: {pass_rate_str}\n"
            f"Runtime: {run.total_runtime_seconds:.1f}s   "
            f"Load/startup: {f'{load_s:.1f}s' if load_s is not None else 'n/a (already resident)'}   "
            f"[{cold_warm}]   "
            f"Mean tok/s: {f'{tps:.1f}' if tps else 'n/a'}   "
            f"Peak VRAM: {f'{vram:.0f}MB' if vram else 'n/a (n/a for vllm - see resident VRAM)'}   "
            f"Resident VRAM: {f'{resident_vram:.0f}MB' if resident_vram else 'n/a'}   "
            f"Backend: {run.backend}"
        ))
        if run.settings_translation_notes:
            self.summary_label.config(
                text=self.summary_label.cget("text") +
                     f"\nSettings translation notes: {'; '.join(run.settings_translation_notes)}"
            )
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
                summary["run_id"], summary["model_name"], summary.get("inference_engine", "transformers"),
                summary["suite_qualified_id"], summary["status"], summary["started_at"], score_display,
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
        self._render_compare(run_a, run_b)

    def _render_compare(self, run_a: BenchmarkRunResult, run_b: BenchmarkRunResult) -> None:
        warnings = []
        if run_a.suite_qualified_id != run_b.suite_qualified_id:
            warnings.append(
                f"SUITE VERSION DIFFERS: {run_a.suite_qualified_id} vs {run_b.suite_qualified_id} "
                "- per-case results are NOT comparable (different cases/scoring may apply)."
            )
        if run_a.repo_id and run_b.repo_id and run_a.repo_id != run_b.repo_id:
            warnings.append(
                f"MODEL WEIGHTS DIFFER: repo_id {run_a.repo_id!r} vs {run_b.repo_id!r} - "
                "this is a model/quantization comparison, not the same checkpoint under two engines."
            )
        if run_a.model_name == run_b.model_name and run_a.inference_engine == run_b.inference_engine:
            warnings.append("Same model AND same engine - this is two runs of one configuration, not a backend comparison.")
        setting_diffs = _diff_material_settings(run_a.generation_config_snapshot, run_b.generation_config_snapshot)
        if setting_diffs:
            warnings.append("EFFECTIVE GENERATION SETTINGS DIFFER: " + "; ".join(setting_diffs))

        def fmt_row(label: str, a: str, b: str) -> str:
            return f"{label:<22} {a:>28} {b:>28}"

        def vram_str(run: BenchmarkRunResult) -> str:
            if run.inference_engine == "vllm":
                return f"{run.resident_vram_mb:.0f}MB (nvidia-smi snapshot)" if run.resident_vram_mb else "n/a"
            return f"{run.peak_vram_mb:.0f}MB peak" if run.peak_vram_mb else "n/a"

        def rate_str(run: BenchmarkRunResult) -> str:
            r = run.scored_pass_rate
            return f"{r * 100:.0f}%" if r is not None else "n/a"

        def load_str(run: BenchmarkRunResult) -> str:
            s = (run.load_telemetry or {}).get("load_time_s")
            if s is None:
                return "n/a (already resident)" if run.was_resident_before_run else "n/a"
            return f"{s:.1f}s"

        lines = [
            fmt_row("Model", run_a.model_name, run_b.model_name),
            fmt_row("Engine", run_a.inference_engine, run_b.inference_engine),
            fmt_row("Suite", run_a.suite_qualified_id, run_b.suite_qualified_id),
            fmt_row("Scored pass rate", rate_str(run_a), rate_str(run_b)),
            fmt_row("Cold load/startup", load_str(run_a), load_str(run_b)),
            fmt_row("Total runtime", f"{run_a.total_runtime_seconds:.1f}s", f"{run_b.total_runtime_seconds:.1f}s"),
            fmt_row("Mean tokens/sec", f"{run_a.mean_tokens_per_sec:.1f}" if run_a.mean_tokens_per_sec else "n/a",
                    f"{run_b.mean_tokens_per_sec:.1f}" if run_b.mean_tokens_per_sec else "n/a"),
            fmt_row("VRAM", vram_str(run_a), vram_str(run_b)),
            fmt_row("Errors", str(run_a.counts.get("error", 0)), str(run_b.counts.get("error", 0))),
        ]
        text = "\n".join(lines)
        if warnings:
            text += "\n\n" + "\n".join(f"WARNING: {w}" for w in warnings)

        text += "\n\nPer-case (A / B):\n"
        cases_a = {c.case_id: c for c in run_a.case_results}
        cases_b = {c.case_id: c for c in run_b.case_results}
        for case_id in dict.fromkeys(list(cases_a) + list(cases_b)):
            ca, cb = cases_a.get(case_id), cases_b.get(case_id)
            text += (f"  {case_id:<32} {ca.status if ca else 'n/a':>10} / "
                     f"{cb.status if cb else 'n/a':<10}\n")

        self.compare_text.config(state=NORMAL)
        self.compare_text.delete("1.0", END)
        self.compare_text.insert("1.0", text)
        self.compare_text.config(state=DISABLED)


def _diff_material_settings(a: Optional[dict], b: Optional[dict]) -> list[str]:
    """Compares two generation_config_snapshot dicts on the curated
    _MATERIAL_SETTING_KEYS list, returning one "key: a_val vs b_val"
    string per differing key. Missing dicts (e.g. an aborted run with
    no snapshot) produce no diffs - nothing to compare, not a false
    "everything differs.\""""
    if not a or not b:
        return []
    diffs = []
    for key in _MATERIAL_SETTING_KEYS:
        va, vb = a.get(key), b.get(key)
        if va != vb:
            diffs.append(f"{key}: {va!r} vs {vb!r}")
    return diffs


def _format_telemetry(telemetry: dict) -> str:
    lines = []
    for phase in ("load", "generate"):
        block = telemetry.get(phase)
        if block:
            lines.append(f"  {phase}: " + ", ".join(f"{k}={v}" for k, v in block.items()))
    return "\n".join(lines) if lines else "  (none captured)"
