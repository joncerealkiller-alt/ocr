"""
OCR Benchmark Review GUI - standalone tool, same "every module stays
independently launchable" convention as ui/row_segmentation_ui.py and
ui/dewarp_preprocessor_ui.py (run directly: `python benchmark/gui.py`).

Ties together:
    loader.py       - reads a data/debug_model_inputs/<run_id>/ run
    viewer.py       - displays one item + scoring controls
    benchmark_db.py - append-only benchmark_results/benchmark_results.csv

Navigator (left): a run picker, then a tree of Row -> Column -> stage1/
stage2 leaves. A leaf shows a "check" prefix once it has a saved score
(read from benchmark_db.load_latest_scores() - the CSV's append-only,
so this always reflects the LAST score entered for that item_id, not
just "has it ever been scored"). Selecting a leaf loads that item into
the viewer on the right.

Saving a score auto-advances to the next leaf in the tree - reviewing
hundreds of items one at a time should need one click per item
(Save-then-look-at-the-next-one), not a click to save plus a separate
click to move on.
"""

from __future__ import annotations

import sys
from pathlib import Path
from tkinter import Tk, Frame, Label, Button, StringVar, ttk
from tkinter import LEFT, TOP, X, Y, BOTH, END

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark import benchmark_db
from benchmark.benchmark_db import append_score, load_latest_scores, new_benchmark_session_id
from benchmark.loader import discover_runs, load_run
from benchmark.models import BenchmarkItem, BenchmarkRun, RunSummary, ScoreRecord, Stage
from benchmark.viewer import ItemViewer


def _run_display_label(summary: RunSummary) -> str:
    source = Path(summary.source_image_path).name if summary.source_image_path else "?"
    return f"{summary.run_id}  —  {source}  ({summary.num_rows} rows, {summary.num_items} items)"


class BenchmarkGUI:
    def __init__(self, root: Tk, results_csv_path: Path | str | None = None):
        """
        results_csv_path defaults to benchmark_db.DEFAULT_RESULTS_CSV -
        read HERE, at call time, not captured as a Python default
        parameter value (which would bind at function-DEFINITION/import
        time and silently ignore any later reassignment of the module
        attribute - a real bug this exact class hit during its own
        smoke test, writing a stray row into the real benchmark_results/
        benchmark_results.csv despite the test believing it had
        redirected storage elsewhere). Passing an explicit path here is
        also what lets a test point at a scratch file safely.
        """
        self.root = root
        root.title("OCR Benchmark Review")
        root.geometry("1500x950")
        root.minsize(1200, 700)

        self._results_csv_path = Path(results_csv_path) if results_csv_path is not None \
            else benchmark_db.DEFAULT_RESULTS_CSV
        self.benchmark_session = new_benchmark_session_id()
        self._runs: list[RunSummary] = []
        self._current_run: BenchmarkRun | None = None
        self._latest_scores: dict[str, ScoreRecord] = {}
        # tree item id -> item_id (only leaves carry this)
        self._tree_item_ids: dict[str, str] = {}

        self._build_widgets()
        self._refresh_run_list()

    # -- layout -----------------------------------------------------------

    def _build_widgets(self) -> None:
        top_bar = Frame(self.root, padx=8, pady=6)
        top_bar.pack(side=TOP, fill=X)
        Label(top_bar, text="Run:", font=("Segoe UI", 9, "bold")).pack(side=LEFT)
        self.run_var = StringVar(value="")
        self.run_combo = ttk.Combobox(top_bar, textvariable=self.run_var, state="readonly",
                                       width=90)
        self.run_combo.pack(side=LEFT, padx=(6, 0))
        self.run_combo.bind("<<ComboboxSelected>>", lambda e: self._on_run_selected())
        Button(top_bar, text="Refresh run list", command=self._refresh_run_list).pack(
            side=LEFT, padx=(8, 0))
        self.progress_label = Label(top_bar, text="", fg="#444")
        self.progress_label.pack(side=LEFT, padx=(16, 0))

        main = Frame(self.root, padx=8, pady=4)
        main.pack(side=TOP, fill=BOTH, expand=True)

        nav_frame = Frame(main, width=340)
        nav_frame.pack(side=LEFT, fill=Y)
        nav_frame.pack_propagate(False)
        Label(nav_frame, text="Rows / fields:", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.tree = ttk.Treeview(nav_frame, show="tree")
        self.tree.pack(fill=BOTH, expand=True, pady=(4, 0))
        self.tree.bind("<<TreeviewSelect>>", lambda e: self._on_tree_select())

        viewer_frame = Frame(main)
        viewer_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=(10, 0))
        self.viewer = ItemViewer(viewer_frame, on_save=self._handle_save)
        self.viewer.pack(fill=BOTH, expand=True)
        self.viewer.set_benchmark_session(self.benchmark_session)

    # -- run loading --------------------------------------------------------

    def _refresh_run_list(self) -> None:
        self._runs = discover_runs()
        labels = [_run_display_label(r) for r in self._runs]
        self.run_combo.configure(values=labels)
        if labels and not self.run_var.get():
            self.run_var.set(labels[0])
            self._on_run_selected()

    def _on_run_selected(self) -> None:
        label = self.run_var.get()
        idx = self.run_combo["values"].index(label) if label in self.run_combo["values"] else -1
        if idx < 0:
            return
        summary = self._runs[idx]
        self._current_run = load_run(summary.run_dir)
        self._latest_scores = load_latest_scores(self._results_csv_path)
        self.viewer.clear()
        self._build_navigator()
        self._refresh_progress_label()

    # -- navigator ------------------------------------------------------------

    def _leaf_text(self, item: BenchmarkItem) -> str:
        prefix = "✓ " if item.item_id in self._latest_scores else "• "
        return f"{prefix}{item.stage.value}"

    def _build_navigator(self) -> None:
        self.tree.delete(*self.tree.get_children())
        self._tree_item_ids.clear()
        if self._current_run is None:
            return
        for row in self._current_run.rows:
            row_node = self.tree.insert("", END, text=f"Row {row.row_index}", open=False)
            for column_name in row.column_names():
                col_node = self.tree.insert(row_node, END, text=column_name, open=False)
                for stage in (Stage.STAGE1, Stage.STAGE2):
                    item = row.item_for(column_name, stage)
                    if item is None:
                        continue
                    leaf = self.tree.insert(col_node, END, text=self._leaf_text(item))
                    self._tree_item_ids[leaf] = item.item_id

    def _refresh_progress_label(self) -> None:
        if self._current_run is None:
            self.progress_label.config(text="")
            return
        total = len(self._current_run.all_items())
        scored = sum(1 for it in self._current_run.all_items() if it.item_id in self._latest_scores)
        self.progress_label.config(text=f"Scored: {scored}/{total}")

    def _on_tree_select(self) -> None:
        selection = self.tree.selection()
        if not selection or self._current_run is None:
            return
        node = selection[0]
        item_id = self._tree_item_ids.get(node)
        if item_id is None:
            return  # a Row/Column grouping node, not a leaf - nothing to load
        item = self._current_run.item_by_id(item_id)
        if item is None:
            return
        self.viewer.load_item(item, self._latest_scores.get(item_id))

    def _advance_to_next_leaf(self) -> None:
        """After a save, moves the tree selection to the next leaf in
        display order (depth-first) - so reviewing a run is click-Save,
        look, click-Save, look, ... without a separate step to move on."""
        selection = self.tree.selection()
        if not selection:
            return
        current = selection[0]
        node = self.tree.next(current)
        if not node:
            # No sibling left - walk up and take the parent's next sibling,
            # descending back into its first leaf.
            parent = self.tree.parent(current)
            while parent:
                sibling = self.tree.next(parent)
                if sibling:
                    node = sibling
                    break
                parent = self.tree.parent(parent)
        while node and node not in self._tree_item_ids:
            children = self.tree.get_children(node)
            if not children:
                break
            node = children[0]
        if node and node in self._tree_item_ids:
            self.tree.selection_set(node)
            self.tree.see(node)

    # -- scoring --------------------------------------------------------------

    def _handle_save(self, record: ScoreRecord) -> None:
        append_score(record, csv_path=self._results_csv_path)
        self._latest_scores[record.item_id] = record
        for node, item_id in self._tree_item_ids.items():
            if item_id == record.item_id:
                item = self._current_run.item_by_id(item_id) if self._current_run else None
                if item is not None:
                    self.tree.item(node, text=self._leaf_text(item))
                break
        self._refresh_progress_label()
        self._advance_to_next_leaf()


def main() -> None:
    root = Tk()
    BenchmarkGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
