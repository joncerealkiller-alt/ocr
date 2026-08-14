"""
Cross-runtime GPU ownership arbiter - the one place that knows WHICH
execution runtime (transformers-in-process vs. a vLLM subprocess)
currently owns the physical GPU, so the two can never hold VRAM at the
same time on this 16GB card.

Built 2026-08-13 for the multi-runtime integration (see the plan's
"Multi-Runtime Model Integration" addendum). Deliberately a THIRD, tiny
module rather than mutual release calls between core/model_residency.py
and core/vllm_runtime.py - inspecting the real call graph showed that
design would be a genuine Python import cycle (model_residency already
imports loader machinery; vllm_runtime imports subprocess/HTTP
machinery; each would need the other), plus split responsibility for
"who owns the GPU right now." This module imports NEITHER runtime -
each runtime registers its own release callback here at import time,
and calls claim() before taking the GPU.

Same module-level singleton pattern as core.model_residency.residency /
core.network_gate.network_gate: one process-wide instance, do not
construct a second.

Scope note (mirrors model_residency.py's own): this is process-local
state plus whatever the registered callbacks actually control. The
vLLM callback controls a real OS subprocess, so ITS effect is cross-
process - but the bookkeeping here still lives in the one backend
process (api/agent_main.py's) that launches everything. There is no
lock file / IPC; CLAUDE.md's check-nvidia-smi-first discipline still
covers humans starting unrelated GPU processes by hand.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional


class GpuCoordinator:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._current_owner: Optional[str] = None
        self._release_callbacks: dict[str, Callable[[], None]] = {}

    @property
    def current_owner(self) -> Optional[str]:
        with self._lock:
            return self._current_owner

    def register_runtime(self, runtime_name: str, release_callback: Callable[[], None]) -> None:
        """
        Called once per runtime at ITS module's import time (see
        core/model_residency.py and core/vllm_runtime.py). Re-registering
        the same name just replaces the callback - harmless, supports
        module reloads in dev.
        """
        with self._lock:
            self._release_callbacks[runtime_name] = release_callback

    def claim(self, runtime_name: str) -> None:
        """
        Ensures `runtime_name` may take the GPU: releases every OTHER
        registered runtime's holdings first (each release callback must
        itself be a no-op when that runtime holds nothing - both
        runtimes' release paths already are), then records the new
        owner. Claiming while already the owner is a cheap no-op apart
        from the loop of other-runtime no-op releases.

        RLock (not Lock) on purpose: a release callback may consult
        this module (e.g. to null the owner via release_noted()) from
        within the claim that triggered it.
        """
        with self._lock:
            for name, release in self._release_callbacks.items():
                if name != runtime_name:
                    release()
            self._current_owner = runtime_name

    def release_noted(self, runtime_name: str) -> None:
        """
        A runtime calls this from its own release path to record that it
        no longer holds the GPU - only clears ownership if that runtime
        was in fact the recorded owner (a stale note from a runtime that
        never owned, or was already displaced, must not clobber the
        actual owner's claim).
        """
        with self._lock:
            if self._current_owner == runtime_name:
                self._current_owner = None


# The one shared instance. Do not construct GpuCoordinator() anywhere else.
gpu_coordinator = GpuCoordinator()
