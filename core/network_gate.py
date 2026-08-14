"""
Single process-wide toggle for whether agent tools may reach the
network at all. Built 2026-08-13 for Phase 7's web_search tool - Jon's
explicit requirement: "these will need a toggle for internet access."

OFF by default, matching this project's whole local-first stance
(CLAUDE.md, README) - every other tool built so far (classify_document,
extract_fields, genealogy_framework_lookup) is local-only by
construction; web_search is the FIRST one that can leave the machine at
all, so it gets an explicit, visible, off-by-default gate rather than
just working the moment it's registered.

Same module-level singleton pattern as core.model_residency.residency /
core.genealogy_memory.memory / core.agent_status.status_hub - one
process-wide instance (`network_gate` below), read live at tool-call
time (not snapshotted per-turn), so flipping the UI checkbox takes
effect on the very next tool call, not just future conversations.
"""

from __future__ import annotations

import threading


class NetworkGate:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._enabled = False

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def set_enabled(self, value: bool) -> None:
        with self._lock:
            self._enabled = value
        print(f"[core.network_gate] internet access {'ENABLED' if value else 'disabled'}")


# The one shared instance every network-touching tool checks. Do not
# construct NetworkGate() anywhere else.
network_gate = NetworkGate()
