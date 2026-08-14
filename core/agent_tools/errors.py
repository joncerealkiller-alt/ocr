"""
Typed tool errors. A tool failure is surfaced back to the agent as a
ToolResultBlock(success=False, error=...) rather than a bare exception,
so the planner can decide whether to retry, pick a different tool, or
ask the user — instead of the turn crashing outright.
"""

from __future__ import annotations

from enum import Enum


class ToolErrorClass(str, Enum):
    INVALID_ARGS = "invalid_args"
    NOT_FOUND = "not_found"
    TIMEOUT = "timeout"
    MODEL_OOM = "model_oom"
    LOW_CONFIDENCE = "low_confidence"
    EXECUTION_FAILED = "execution_failed"


class ToolError(Exception):
    """Raised by a tool implementation; caught by the dispatcher and
    turned into a ToolResultBlock(success=False, error=this)."""

    def __init__(self, error_class: ToolErrorClass, message: str):
        self.error_class = error_class
        self.message = message
        super().__init__(f"[{error_class.value}] {message}")

    def to_dict(self) -> dict:
        return {"error_class": self.error_class.value, "message": self.message}
