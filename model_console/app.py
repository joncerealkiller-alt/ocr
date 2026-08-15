"""
model_console entry point. Run as:

    python -m model_console.app

Deliberately its own process, never imported into core/extractor.py's
process - see the plan's Architectural Risks #1: two separate
model-residency caches (this one and core.extractor's pipeline cache)
coexisting in the same process would have no coordination and could
both try to hold GPU memory simultaneously.

2026-08-13: PYTORCH_CUDA_ALLOC_CONF is set below, before anything in
this process imports torch - same underlying bug as api/agent_main.py
(see that module's docstring and docs/WSL_COMPUTE_BACKEND_BASELINE.md's
"Real bug found + fixed: VRAM not released" section for the full
root-cause writeup): PyTorch's default CUDA allocator can strand an
entire multi-GB memory segment un-freeable after a real, correct
release (core/model_residency.py's _release_current() - unchanged,
not at fault) if one small persistent per-process allocation (a
cuBLAS/cuDNN workspace) happens to share that segment.

DIFFERENT VALUE than the WSL fix, confirmed live on THIS platform, not
copied blind: `expandable_segments:True` (the WSL/Linux fix) silently
NO-OPS on Windows - PyTorch prints "expandable_segments not supported
on this platform" and reserved memory stays pinned exactly as before
(tested and confirmed here). `backend:cudaMallocAsync` - a different,
also-real PyTorch allocator mode that delegates directly to the CUDA
driver's own stream-ordered allocator instead of PyTorch's segment-
based caching allocator - IS supported on Windows and confirmed live to
fix the same symptom: after a load -> generate -> release cycle,
reserved memory dropped from 10234.1MB to 67.1MB (vs. staying at
~10.2GB with either the default allocator or the no-op expandable_
segments setting on this platform). This process (model_console) is
the ONLY Windows-side client of core.model_residency.residency -
core/extractor.py's separate batch pipeline always runs as its own
process and was never implicated - so this is the one place the
Windows-side fix belongs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from tkinter import Tk
from tkinter import ttk

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

# Mirrors scripts/model_assessment.py's own sys.path fix - this module
# lives one directory below repo root, so repo root must be back on
# sys.path before `core.*` / `model_console.*` absolute imports resolve
# when launched directly (python model_console/app.py) rather than via
# `python -m model_console.app`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.agent_status_server import start_status_server  # noqa: E402
from model_console.chat_tab import ChatTab  # noqa: E402
from model_console.benchmark_tab import BenchmarkTab  # noqa: E402


def main() -> None:
    # Started once per process, before the Tk mainloop - a daemon
    # thread, so it never blocks app shutdown. Failure to bind (e.g. a
    # PREVIOUS model_console instance's server still holding the port
    # after this process was killed rather than closed cleanly) must
    # not prevent the actual chat app from starting - the dashboard
    # integration is observability, not a hard dependency of the app it
    # observes. See core/agent_status_server.py's own module docstring.
    try:
        start_status_server()
    except OSError as e:
        print(f"[model_console.app] WARNING: could not start the agent "
              f"status server ({e}) - continuing without dashboard "
              f"integration for this session.")

    root = Tk()
    root.title("Model Console")
    root.geometry("900x700")

    # Notebook added 2026-08-15 for the Benchmark tab, alongside the
    # existing Chat tab - each tab owns its own ChatBackendAdapter
    # instance (see BenchmarkTab's own docstring on why it doesn't share
    # ChatTab's), so nothing about ChatTab's behavior changes by being
    # inside a Notebook instead of packed directly into root.
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    tab = ChatTab(notebook)
    notebook.add(tab, text="Chat")

    benchmark_tab = BenchmarkTab(notebook)
    notebook.add(benchmark_tab, text="Benchmark")

    def on_close():
        tab.shutdown()
        benchmark_tab.shutdown()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
