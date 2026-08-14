"""
Tool Registry — same config-owns-behavior principle as
core/loader_registry.py's LOADER_REGISTRY: this dict is mechanics
(name -> ToolSpec), never decision logic. The planner picks a tool
name; the registry just resolves it.
"""

from __future__ import annotations

from core.agent_tools.schema import CostHint, SideEffect, ToolSpec

TOOL_REGISTRY: dict[str, ToolSpec] = {}


def register_tool(
    name: str,
    description: str,
    input_schema,
    output_schema,
    side_effects: SideEffect = SideEffect.READ,
    cost_hint: CostHint = CostHint.CHEAP,
):
    """Decorator: @register_tool(...) def my_tool(args: InputModel) -> OutputModel"""

    def decorator(fn):
        if name in TOOL_REGISTRY:
            raise ValueError(f"Tool {name!r} is already registered.")
        TOOL_REGISTRY[name] = ToolSpec(
            name=name,
            description=description,
            input_schema=input_schema,
            output_schema=output_schema,
            fn=fn,
            side_effects=side_effects,
            cost_hint=cost_hint,
        )
        return fn

    return decorator
