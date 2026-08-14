"""
Tool contracts. A ToolSpec is pure metadata + a callable — the registry
and dispatcher never know anything tool-specific beyond this shape.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel


class SideEffect(str, Enum):
    NONE = "none"    # pure read of already-in-memory/already-on-disk data
    READ = "read"    # touches disk/DB but doesn't mutate
    WRITE = "write"  # mutates disk/DB/network state


class CostHint(str, Enum):
    CHEAP = "cheap"  # CPU-only, sub-second
    GPU = "gpu"      # touches a resident model — must serialize against
                      # other GPU tool calls (see dispatcher.py)
    SLOW = "slow"    # cheap on GPU but slow wall-clock (e.g. network)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: type[BaseModel]
    output_schema: type[BaseModel]
    fn: Callable[[BaseModel], BaseModel]
    side_effects: SideEffect = SideEffect.READ
    cost_hint: CostHint = CostHint.CHEAP


@dataclass
class ToolCallBlock:
    tool_name: str
    args: dict[str, Any]
    call_id: str = ""


@dataclass
class ToolResultBlock:
    call_id: str
    tool_name: str
    success: bool
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    latency_s: float = 0.0
    started_at: float = field(default_factory=time.time)
