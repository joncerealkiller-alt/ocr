"""
Single GPU-residency authority WITHIN ONE OS PROCESS - specifically,
whichever process hosts model_console/agent mode. This is deliberately
NOT a repo-wide singleton across every process that ever loads a model
in this codebase, and should never be extended to try to become one.
core/extractor.py's standalone batch pipeline keeps its own, entirely
separate _loader_cache in ITS OWN process (see "Scope" below) - that
remains correct and untouched. "Single authority" here means: within
the model_console process, both of that process's GPU-touching clients
(chat + agent tools) now agree on one resident model instead of each
independently owning a cache. Built 2026-08-13 after a real
architectural gap was found while planning agent-mode hardware behavior
on a 16GB card already running near its ceiling in normal chat use.

SCOPE, STATED PLAINLY: this module's `residency` singleton is a
process-local Python object - like any module-level state, importing it
from a different OS process (e.g. core/extractor.py::run() launched as
its own script) gets that process its OWN separate instance, not a
connection to this one. There is no cross-process coordination here (no
lock file, no IPC, no shared memory) and none is needed: CLAUDE.md's
GPU-safety rule already requires a human to check `nvidia-smi` before
starting a second heavyweight inference process by hand, and the two
processes that matter here (model_console and extractor.py's batch
`run()`) are never launched concurrently as a matter of existing
project practice. If a future use case DOES need true cross-process GPU
coordination, that is a different, larger problem than this module
solves - do not assume this singleton already covers it.

ROOT CAUSE: model_console/adapter.py's ChatBackendAdapter and
core/extractor.py's get_or_build_loader() are the only two
CROSS-CALL-PERSISTENT loader caches in this codebase (every other
LOADER_REGISTRY call site - core/classifier.py, core/row_extraction.py,
benchmark/*, scripts/model_assessment.py - loads per-call inside its
own standalone OS process, so no collision is possible there - see
model_console/app.py's own docstring on why it's deliberately a
separate process from core/extractor.py's batch pipeline). Those two
caches were never a problem in isolation. The collision is new and
specific: core/agent_tools/tools.py's extract_fields_tool called
core.extractor.get_or_build_loader() FROM INSIDE the model_console
process (agent mode) - the one place both caches could end up holding
an independently-loaded model on the SAME GPU at the SAME time. GPU
execution being serialized (core/agent_tools/dispatcher.py's
cost_hint=GPU handling) does NOT prevent this - serialization only
stops two things from running inference AT THE SAME INSTANT, it says
nothing about both being simultaneously RESIDENT in VRAM, which is what
actually causes an OOM on tight hardware.

core/extractor.py's OWN _loader_cache is UNCHANGED and remains correct
for its own use - core/extractor.py::run() always executes as its own
standalone process, so it never shares an address space with this
module's clients. This module does not touch extractor.py at all.

This module's only two clients are:
  - model_console/adapter.py's ChatBackendAdapter (the chat model)
  - core/agent_tools/tools.py's extract_fields_tool (a GPU tool)
Both now ask THIS module for a loader instead of owning one
independently - see each file's own docstring for how it delegates.

GPU EXECUTION SERIALIZATION vs GPU MODEL RESIDENCY are deliberately
kept as two separate concerns, not collapsed into one mechanism: the
dispatcher decides WHEN a GPU call may run relative to other GPU calls;
this module decides WHICH model is physically loaded when that call
actually executes. Serialization would still matter even for a single
huge model (avoiding two concurrent forward passes); residency swapping
would still matter even for perfectly serialized calls to two DIFFERENT
models. Both layers stay in place, independently.

NO HIDDEN STATE SURVIVES AN UNLOAD, BY DESIGN: releasing a model here
tears it down completely (weights, KV cache, everything) - "restoring"
a previous model means fully reloading it from scratch, not resuming
some suspended session. This is safe because logical state (prompt
text, conversation history, tool results, generation config) already
lives OUTSIDE any loader - in ChatSession, GenerationConfig, and the
agent's own scratchpad - and gets rebuilt fresh on every call regardless
of whether a reload happened in between. See core/agent_tools/planner.py
and model_console/session.py - neither ever assumes a loader's internal
state persists across calls.
"""

from __future__ import annotations

import ctypes
import gc
import time
from dataclasses import dataclass
from typing import Any, Optional

from core.agent_status import status_hub
from core.gpu_coordinator import gpu_coordinator
from core.loader_registry import LOADER_REGISTRY
from core.loaders.base_loader import BaseLoader, GenerationConfig
from core.resource_guard import check_resources_or_raise, log_resource_trend as _log_resource_trend

try:
    import torch
except ImportError:  # pragma: no cover - torch is a hard runtime dependency
    torch = None  # noqa: N816

try:
    _libc = ctypes.CDLL("libc.so.6")
except OSError:  # pragma: no cover - non-glibc/non-Linux host
    _libc = None


def _trim_host_memory() -> None:
    """
    Returns freed-but-not-yet-returned glibc heap arenas back to the OS
    (2026-08-15, found live: the backend process held ~8.65GB RSS with
    NO model resident and GPU VRAM already idle - a completely separate
    leak from the earlier CUDA-allocator VRAM issue). gc.collect()
    correctly drops the Python references, but glibc's ptmalloc doesn't
    automatically unmap the underlying pages back to the kernel after a
    large allocate-then-free cycle (loading a multi-GB model's weight
    tensors into host memory during from_pretrained(), even when they're
    then moved to GPU, leaves the heap fragmented) - this is the same
    class of problem torch.cuda.empty_cache() solves for VRAM, just at
    the OS/glibc layer instead of CUDA's. malloc_trim(0) asks glibc to
    release every trimmable arena; called AFTER gc.collect() so nothing
    still-referenced gets in the way. No-op (not an error) on a non-
    glibc host (e.g. native Windows, if this module is ever imported
    there) - _libc is None there, guarded below.
    """
    if _libc is None:
        return
    try:
        _libc.malloc_trim(0)
    except Exception:
        pass


def _physical_identity(config: GenerationConfig) -> tuple:
    """
    The fields that determine whether two GenerationConfigs can share
    ONE resident loader instance without a reload. Deliberately NOT the
    same test as "same model_name": in this codebase model_name already
    uniquely determines a config/models/<name>.yaml file
    (load_model_config()), so two configs loaded under the same
    model_name are identical by construction - the risk this guards
    against is a resident loader's config having silently drifted from
    what a fresh load of the same model_name would now produce (e.g. a
    YAML hot-edited mid-session), not two different names secretly
    colliding.

    Deliberately EXCLUDES prompt_text, sampling params (temperature/
    top_p/top_k/do_sample/repetition_penalty/etc), max_new_tokens, and
    every other field that's legitimately swappable on an already-
    resident loader without a reload - that's exactly the behavior both
    prior independent caches already trusted before this module existed
    (core/extractor.py's get_or_build_loader() swaps prompt_text on a
    cached instance across buckets sharing a model; model_console's
    ChatBackendAdapter.ensure_loaded() swapped config wholesale on a
    same-model_name reuse). This function generalizes that same trust
    boundary, it doesn't tighten or loosen it.
    """
    return (
        config.loader_class,
        config.repo_id,
        config.processor_repo_id or config.repo_id,
        config.attn_implementation,
    )


@dataclass
class _Residency:
    model_name: str
    config: GenerationConfig
    loader: BaseLoader
    identity: tuple


class ModelResidencyManager:
    """
    Holds AT MOST ONE resident GPU loader at a time for this process.
    Use the module-level `residency` singleton below - constructing a
    second instance would just recreate the exact bug this module
    exists to close.
    """

    def __init__(self) -> None:
        self._current: Optional[_Residency] = None
        # Phase 3 inference telemetry (2026-08-12) - load-side metrics
        # (baseline/after-load VRAM, load time) for whichever load()
        # most recently actually happened (a same-model config-swap
        # reuse in acquire() does NOT update this, since no load
        # occurred). None until the first real load in this process.
        # model_console/adapter.py reads this into send_turn()'s meta
        # dict alongside GemmaLoader's own generate-side telemetry.
        self._last_load_telemetry: Optional[dict[str, Any]] = None

    @property
    def resident_model_name(self) -> Optional[str]:
        return self._current.model_name if self._current else None

    @property
    def resident_loader(self) -> Optional[BaseLoader]:
        return self._current.loader if self._current else None

    @property
    def last_load_telemetry(self) -> Optional[dict[str, Any]]:
        return self._last_load_telemetry

    def acquire(self, model_name: str, config: GenerationConfig) -> BaseLoader:
        """
        Ensures `model_name` is the resident model and returns its
        loader. Reuses the resident loader IN PLACE (config swapped, no
        reload) when model_name matches AND physical identity still
        agrees. Logs a loud warning and forces a reload if model_name
        matches but identity has drifted, rather than silently trusting
        a stale assumption ("loud error over silent wrong behavior" -
        this project's existing discipline, see core/extractor.py's
        _check_header_matches for the same principle applied elsewhere).
        Releases whatever else was resident first if a genuinely
        different model is requested - never holds two loaders at once.
        """
        if self._current is not None and self._current.model_name == model_name:
            if self._current.identity == _physical_identity(config):
                self._current.loader.config = config
                self._current.config = config
                print(f"[core.model_residency] reuse: {model_name!r} already resident, config swapped.")
                status_hub.update(
                    resident_model_name=model_name,
                    resident_loader_class=config.loader_class,
                    resident_repo_id=config.repo_id,
                )
                return self._current.loader
            print(
                f"[core.model_residency] WARNING: {model_name!r} is resident but its "
                f"physical identity changed since it was loaded (was "
                f"{self._current.identity}, now {_physical_identity(config)}) - "
                f"forcing a reload rather than trusting the name alone."
            )

        if self._current is not None:
            self._release_current()

        # OOM-hardening pre-flight check (2026-08-15) - refuses to start
        # a load when host RAM or free VRAM is already critically low,
        # same placement discipline as the gpu_coordinator claim below
        # (AFTER the reuse-in-place fast path, since that path consumes
        # no new memory and must never be blocked by this). Raises
        # ResourceExhaustedError, deliberately NOT caught here - the
        # caller (api/agent_main.py) turns it into a clean 503; letting
        # it propagate raw is correct for other callers (agent tool
        # borrows) too, per this project's loud-error discipline.
        check_resources_or_raise(context=f"loading {model_name!r}")

        # Cross-runtime GPU claim (2026-08-13, multi-runtime integration)
        # - evicts a running vLLM subprocess (or any other registered
        # runtime's holdings) before this process loads transformers
        # weights. Deliberately AFTER the reuse-in-place fast path above:
        # if the requested model is already resident, transformers
        # already owns the GPU and there is nothing to arbitrate. See
        # core/gpu_coordinator.py's module docstring for why this is a
        # third module rather than a direct vllm_runtime import (import
        # cycle avoidance + single ownership authority).
        gpu_coordinator.claim("transformers")

        loader_cls = LOADER_REGISTRY.get(config.loader_class)
        if loader_cls is None:
            raise ValueError(
                f"No loader registered for loader_class={config.loader_class!r}. "
                f"Known loaders: {list(LOADER_REGISTRY.keys())}"
            )
        print(f"[core.model_residency] load: {model_name!r} (loader_class={config.loader_class!r})")

        baseline_vram_mb = None
        if torch is not None and torch.cuda.is_available():
            baseline_vram_mb = round(torch.cuda.memory_allocated() / 1e6, 1)
        status_hub.emit(
            "model_loading",
            operation="loading",
            resident_model_name=model_name,
            resident_loader_class=config.loader_class,
            resident_repo_id=config.repo_id,
            vram_idle_mb=baseline_vram_mb,
        )
        load_start = time.perf_counter()

        loader = loader_cls(config)
        loader.initialize_model_and_tokenizer()

        load_time_s = round(time.perf_counter() - load_start, 3)
        after_load_vram_mb = None
        after_load_reserved_mb = None
        if torch is not None and torch.cuda.is_available():
            after_load_vram_mb = round(torch.cuda.memory_allocated() / 1e6, 1)
            after_load_reserved_mb = round(torch.cuda.memory_reserved() / 1e6, 1)
        self._last_load_telemetry = {
            "model_name": model_name,
            "baseline_vram_mb": baseline_vram_mb,
            "after_load_vram_mb": after_load_vram_mb,
            "after_load_reserved_mb": after_load_reserved_mb,
            "load_time_s": load_time_s,
        }
        status_hub.emit(
            "model_loaded",
            operation="idle",
            vram_current_mb=after_load_vram_mb,
        )

        self._current = _Residency(model_name, config, loader, _physical_identity(config))
        return loader

    def _release_current(self) -> None:
        if self._current is None:
            return
        print(f"[core.model_residency] release: {self._current.model_name!r}")
        loader = self._current.loader
        try:
            loader.release()
        except Exception:
            pass
        try:
            loader.model = None
            loader.processor = None
            loader.tokenizer = None
        except Exception:
            pass
        self._current = None
        del loader
        gc.collect()
        _trim_host_memory()
        vram_after_release_mb = None
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            vram_after_release_mb = round(torch.cuda.memory_allocated() / 1e6, 1)
        status_hub.emit(
            "model_released",
            resident_model_name=None,
            resident_loader_class=None,
            resident_repo_id=None,
            vram_current_mb=vram_after_release_mb,
            vram_idle_mb=vram_after_release_mb,
        )
        gpu_coordinator.release_noted("transformers")
        _log_resource_trend("model_residency")

    def release_all(self) -> None:
        """Full teardown - nothing left resident. Safe to call when
        nothing is resident (no-op)."""
        self._release_current()

    def borrow(self, model_name: str, config: GenerationConfig) -> "_BorrowContext":
        """
        Context manager: acquires `model_name`, yields its loader, and
        on exit restores whatever model (if any) was resident BEFORE
        this call - "serialize originating state -> release -> load
        tool model -> execute -> release tool model -> restore
        originating model." If nothing was resident before (a cold
        start) or the requested model was ALREADY the resident one
        (no swap needed at all), nothing is torn down on exit - a
        newly-loaded model is left resident rather than pointlessly
        released, since the next caller likely wants it again.

        Exception-safe: restoration happens in `finally`, so a failure
        during the borrowed tool's own execution still leaves residency
        in a recoverable, known state - either the tool's model (if
        restore itself is what failed) or the restored original -
        rather than an ambiguous one. The exception itself always
        propagates unchanged; this only guarantees residency state,
        never swallows the caller's error.
        """
        return _BorrowContext(self, model_name, config)


class _BorrowContext:
    def __init__(self, manager: ModelResidencyManager, model_name: str, config: GenerationConfig):
        self._manager = manager
        self._model_name = model_name
        self._config = config
        self._previous: Optional[_Residency] = None

    def __enter__(self) -> BaseLoader:
        current = self._manager._current
        # Only remember a "previous" to restore if it's genuinely a
        # DIFFERENT model - borrowing the model that's already resident
        # (e.g. a tool happens to use the same model as the current chat
        # session) is a no-op swap with nothing to restore afterward.
        self._previous = current if (current is not None and current.model_name != self._model_name) else None
        if self._previous is not None:
            # originating_model_name is set here (not by adapter.py) so
            # it reflects the model actually being displaced, not just
            # whatever model_console's UI has selected - correct even
            # if a future caller borrows without going through the chat
            # adapter at all.
            status_hub.update(
                originating_model_name=self._previous.model_name,
                borrowed_model_name=self._model_name,
            )
        loader = self._manager.acquire(self._model_name, self._config)
        return loader

    def __exit__(self, exc_type, exc, tb) -> None:
        # Runs whether the `with` block raised or not (Python guarantees
        # __exit__ is called on the way out either way) - restoration
        # happens here unconditionally, and returning None (not True)
        # means any exception from the block always propagates normally,
        # never swallowed.
        if self._previous is not None:
            print(f"[core.model_residency] restore: {self._previous.model_name!r} "
                  f"(after borrowing {self._model_name!r})")
            status_hub.emit("model_restoring", operation="restoring", borrowed_model_name=None)
            self._manager.acquire(self._previous.model_name, self._previous.config)
            status_hub.update(originating_model_name=None)


# The single shared instance every client uses. Do not construct
# ModelResidencyManager() anywhere else - that would just recreate the
# two-independent-owners bug this module exists to close.
residency = ModelResidencyManager()

# Cross-runtime registration (2026-08-13): lets core/gpu_coordinator.py
# evict this runtime's resident model when another runtime (vLLM
# subprocess) claims the GPU - see gpu_coordinator's module docstring.
# release_all() is already a safe no-op when nothing is resident.
gpu_coordinator.register_runtime("transformers", residency.release_all)
