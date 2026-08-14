"""
Single authoritative runtime-state hub for the model_console agent
process. Built 2026-08-13 so an external dashboard can OBSERVE what
ModelResidencyManager/Dispatcher/Planner/telemetry are actually doing,
without becoming a second, independently-guessing owner of that state
(Jon's explicit design constraint: "don't let the dashboard become
another owner of model state - it should observe the residency
manager, not independently ask loaders what's happening").

This module is that one owner. core/model_residency.py,
core/agent_tools/dispatcher.py, core/agent_tools/planner.py, and
model_console/adapter.py all call INTO this hub as they do their real
work (a few extra lines each, no behavior change) - none of them
reads status back out of it, and nothing outside this file mutates
AgentStatus directly. core/agent_status_server.py is a thin HTTP/SSE
transport wrapped around this hub's snapshot()/subscribe() - it has no
state of its own either.

Deliberately stdlib-only (dataclasses/threading/queue) - no new
runtime dependency for something this small.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from queue import Full, Queue
from typing import Any, Optional


@dataclass
class AgentStatus:
    # --- Residency (core/model_residency.py) ---
    resident_model_name: Optional[str] = None
    resident_loader_class: Optional[str] = None
    resident_repo_id: Optional[str] = None
    # idle | loading | generating | extracting | restoring
    operation: str = "idle"
    # The model the CURRENT chat session is configured for (model_console's
    # ChatBackendAdapter's own selection) - stays set even while a tool
    # has temporarily borrowed the GPU for a different model, so a
    # dashboard can show "Gemma (borrowed by Qwen3VL)" rather than just
    # whatever's resident this instant.
    originating_model_name: Optional[str] = None
    # Non-None only while core.model_residency.residency.borrow() has
    # displaced the originating model for a GPU tool call.
    borrowed_model_name: Optional[str] = None

    # --- Agent / planner (core/agent_tools/planner.py) ---
    active_step: Optional[str] = None          # e.g. "2/4"
    current_tool_name: Optional[str] = None
    current_tool_cost_hint: Optional[str] = None  # cheap | gpu | slow
    last_tool_name: Optional[str] = None
    last_tool_success: Optional[bool] = None
    last_tool_error: Optional[str] = None

    # --- Telemetry (core/model_residency.py load side + loader generate side) ---
    vram_idle_mb: Optional[float] = None     # last-seen VRAM while operation was "idle"
    vram_current_mb: Optional[float] = None
    vram_peak_mb: Optional[float] = None
    prompt_tokens: Optional[int] = None
    generated_tokens: Optional[int] = None
    tokens_per_sec: Optional[float] = None

    # --- Conversation context (model_console/adapter.py) ---
    context_enabled: Optional[bool] = None
    context_tokens: Optional[int] = None

    # --- Run identity ---
    # NOT populated yet, honestly: agent-mode chat sessions don't
    # construct a core.run_context.RunContext today (that's the offline
    # batch pipeline's concept - manifest_pipeline.py etc). Left as None
    # rather than faking a value; wire this up if/when agent mode ever
    # gets a real RunContext of its own.
    run_context_hash: Optional[str] = None

    # --- Dispatcher (core/agent_tools/dispatcher.py) ---
    queue_depth: int = 0
    gpu_locked: bool = False

    updated_at: float = field(default_factory=time.time)


class AgentStatusHub:
    """
    Process-wide singleton (use the module-level `status_hub` below,
    same convention as core/model_residency.py's `residency` and
    core/genealogy_memory.py's `memory`). Holds ONE AgentStatus, and a
    set of subscriber queues for push (SSE) consumers.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._status = AgentStatus()
        self._subscribers: list[Queue] = []

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return asdict(self._status)

    def update(self, **fields: Any) -> dict[str, Any]:
        """Mutates the shared status in place and returns the new
        snapshot. Does NOT emit an event - use emit() when a state
        change is significant enough for a dashboard to react to
        immediately; use update() alone for telemetry fields that
        just ride along with the next real event."""
        with self._lock:
            for key, value in fields.items():
                if not hasattr(self._status, key):
                    raise AttributeError(
                        f"AgentStatus has no field {key!r} - typo, or a field "
                        f"that needs to be added to the dataclass first "
                        f"(not silently accepted, per this project's own "
                        f"'loud error over silent wrong behavior' discipline)."
                    )
                setattr(self._status, key, value)
            self._status.updated_at = time.time()
            return asdict(self._status)

    def emit(self, event: str, **fields: Any) -> None:
        """Updates status (if any fields given) and pushes {event,
        status, ts} to every subscriber. A full subscriber queue (a
        dashboard that stopped reading) drops the event for THAT
        subscriber rather than blocking the whole agent process on a
        slow/dead consumer - observers must never be able to stall
        real work."""
        snapshot = self.update(**fields) if fields else self.snapshot()
        payload = {"event": event, "status": snapshot, "ts": time.time()}
        with self._lock:
            subscribers = list(self._subscribers)
        for q in subscribers:
            try:
                q.put_nowait(payload)
            except Full:
                pass

    def subscribe(self) -> Queue:
        q: Queue = Queue(maxsize=200)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: Queue) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)


# The one shared instance every client (residency/dispatcher/planner/
# adapter) calls into, and the one instance the HTTP/SSE server reads
# from. Do not construct AgentStatusHub() anywhere else.
status_hub = AgentStatusHub()
