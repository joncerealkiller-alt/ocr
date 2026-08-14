"""
Tool Dispatcher — looks up a ToolCallBlock's tool in the registry,
validates args, executes, validates the result, and always returns a
ToolResultBlock (never raises to the caller — failures are data).

dispatch_many() runs independent calls concurrently, but serializes any
calls flagged cost_hint=GPU against each other and against everything
else, per CLAUDE.md: running two models against the GPU concurrently
has already caused a real, confirmed failure in this project
(moondream2). Non-GPU calls run in a thread pool since they're mostly
I/O/CPU bound, not truly parallel-CPU workloads.
"""

from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from pydantic import ValidationError

from core.agent_status import status_hub
from core.agent_tools.errors import ToolError, ToolErrorClass
from core.agent_tools.registry import TOOL_REGISTRY
from core.agent_tools.schema import CostHint, ToolCallBlock, ToolResultBlock


def dispatch(call: ToolCallBlock) -> ToolResultBlock:
    """
    Thin status-emitting wrapper around _dispatch_inner() - every code
    path through the real dispatch logic below (tool not found, bad
    args, execution failure, output-schema failure, success) funnels
    through here exactly once, so tool_started/tool_finished always
    fire as a matched pair regardless of which return path was taken,
    without duplicating emit calls at each of _dispatch_inner()'s five
    return statements.
    """
    spec = TOOL_REGISTRY.get(call.tool_name)
    is_gpu = spec is not None and spec.cost_hint == CostHint.GPU
    status_hub.emit(
        "tool_started",
        operation="extracting" if is_gpu else "generating",
        current_tool_name=call.tool_name,
        current_tool_cost_hint=spec.cost_hint.value if spec is not None else None,
        gpu_locked=is_gpu,
    )
    result = _dispatch_inner(call)
    status_hub.emit(
        "tool_finished",
        operation="idle",
        current_tool_name=None,
        current_tool_cost_hint=None,
        last_tool_name=call.tool_name,
        last_tool_success=result.success,
        last_tool_error=result.error["message"] if result.error else None,
        gpu_locked=False,
    )
    return result


def _dispatch_inner(call: ToolCallBlock) -> ToolResultBlock:
    call_id = call.call_id or str(uuid.uuid4())
    started = time.time()

    spec = TOOL_REGISTRY.get(call.tool_name)
    if spec is None:
        return ToolResultBlock(
            call_id=call_id,
            tool_name=call.tool_name,
            success=False,
            error=ToolError(
                ToolErrorClass.NOT_FOUND, f"No tool registered as {call.tool_name!r}."
            ).to_dict(),
            latency_s=time.time() - started,
        )

    try:
        args = spec.input_schema.model_validate(call.args)
    except ValidationError as e:
        return ToolResultBlock(
            call_id=call_id,
            tool_name=call.tool_name,
            success=False,
            error=ToolError(ToolErrorClass.INVALID_ARGS, str(e)).to_dict(),
            latency_s=time.time() - started,
        )

    try:
        result = spec.fn(args)
    except ToolError as e:
        return ToolResultBlock(
            call_id=call_id,
            tool_name=call.tool_name,
            success=False,
            error=e.to_dict(),
            latency_s=time.time() - started,
        )
    except Exception as e:  # noqa: BLE001 - genuinely unknown failure, surface it
        return ToolResultBlock(
            call_id=call_id,
            tool_name=call.tool_name,
            success=False,
            error=ToolError(ToolErrorClass.EXECUTION_FAILED, str(e)).to_dict(),
            latency_s=time.time() - started,
        )

    try:
        validated = spec.output_schema.model_validate(
            result.model_dump() if hasattr(result, "model_dump") else result
        )
    except ValidationError as e:
        return ToolResultBlock(
            call_id=call_id,
            tool_name=call.tool_name,
            success=False,
            error=ToolError(
                ToolErrorClass.EXECUTION_FAILED,
                f"Tool {call.tool_name!r} returned a result that failed its own "
                f"output schema: {e}",
            ).to_dict(),
            latency_s=time.time() - started,
        )

    return ToolResultBlock(
        call_id=call_id,
        tool_name=call.tool_name,
        success=True,
        result=validated.model_dump(),
        latency_s=time.time() - started,
    )


def dispatch_many(calls: list[ToolCallBlock]) -> list[ToolResultBlock]:
    """Runs independent tool calls concurrently where safe. GPU-flagged
    tools always run serially, in call order, relative to each other and
    to nothing else running at the same time (see module docstring)."""
    for c in calls:
        if not c.call_id:
            c.call_id = str(uuid.uuid4())

    gpu_calls = [c for c in calls if _cost_hint(c) == CostHint.GPU]
    other_calls = [c for c in calls if _cost_hint(c) != CostHint.GPU]

    status_hub.update(queue_depth=len(calls))
    results: dict[str, ToolResultBlock] = {}

    if other_calls:
        with ThreadPoolExecutor(max_workers=min(8, len(other_calls))) as pool:
            for r in pool.map(dispatch, other_calls):
                results[r.call_id] = r
                status_hub.update(queue_depth=len(calls) - len(results))

    for c in gpu_calls:
        results[c.call_id] = dispatch(c)
        status_hub.update(queue_depth=len(calls) - len(results))

    return [results[c.call_id] for c in calls]


def _cost_hint(call: ToolCallBlock) -> CostHint:
    spec = TOOL_REGISTRY.get(call.tool_name)
    return spec.cost_hint if spec else CostHint.CHEAP
