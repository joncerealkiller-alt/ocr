"""
Thin bridge between model_console's ChatBackendAdapter and
core.agent_tools.planner.run_agent_turn - the only file allowed to
import both sides, so neither package has to know about the other
directly (core/ must not import model_console/, per adapter.py's own
import-boundary discipline; the planner instead takes a plain
send_turn_fn callable).

Wired into chat_tab.py's UI (the agent-mode checkbox calls
run_agent_chat_turn(self.adapter, ...) directly - see chat_tab.py's
_worker()).

2026-08-13 UPDATE (HTTP-client migration, see adapter.py's own
docstring): this is also the ONE place that branches on
adapter.backend. core.agent_tools.planner.run_agent_turn()'s loop
(cpu_planner -> dispatcher -> tools, all in-process) can't sensibly
run half on Windows and half on WSL, so a remote-backed adapter never
runs that loop locally at all - it sends the WHOLE turn to the WSL
backend in one POST /chat/turn (use_agent=True) via
adapter.send_agent_turn(), and this module reconstructs the real
AgentTurnResult/AgentStep/PlanStep/ToolResultBlock dataclass instances
from the JSON response, so chat_tab.py's rendering code (attribute
access like result.steps[i].plan.action, step.tool_result.success)
needed zero changes - it has no idea whether the turn ran locally or
over HTTP.
"""

from __future__ import annotations

from typing import Any, Optional

from PIL import Image

from core.agent_tools.planner import AgentStep, AgentTurnResult, PlanStep, run_agent_turn
from core.agent_tools.schema import ToolResultBlock
from model_console.adapter import ChatBackendAdapter
from model_console.session import ChatTurn


def run_agent_chat_turn(
    adapter: ChatBackendAdapter,
    user_text: str,
    image: Optional[Image.Image],
    system_prompt: str,
    history: Optional[list[ChatTurn]] = None,
    image_path: Optional[str] = None,
) -> AgentTurnResult:
    """
    Runs one agent turn (plan -> optional tool call(s), chained up to
    MAX_STEPS -> answer - see core/agent_tools/planner.py's module
    docstring for the Phase 4 multi-step loop) against whatever model is
    currently resident on `adapter`
    (adapter.ensure_loaded() must already have been called, same
    precondition as adapter.send_turn()).

    history is passed straight through to adapter.send_turn() unchanged
    - history=None keeps the one-shot path (see adapter.py's send_turn
    docstring on why that must stay byte-identical to pre-context-mode
    behavior); history=[...] takes the context-aware path. The agent's
    OWN planning/final-answer turns are not themselves added to
    `history` - only the caller's ChatSession decides what becomes a
    persisted ChatTurn.

    backend="remote": the entire loop runs server-side (see this
    module's docstring) - one HTTP call, then the response is
    reconstructed into the same AgentTurnResult shape the local path
    returns.
    """
    if getattr(adapter, "backend", "local") == "remote":
        data = adapter.send_agent_turn(user_text, image, system_prompt,
                                        history=history, image_path=image_path)
        agent_result = data.get("agent_result")
        if agent_result is None:
            raise RuntimeError(
                "Remote /chat/turn returned no agent_result for use_agent=True - "
                "server-side run_agent_chat_turn() should always populate this field."
            )
        return _agent_turn_result_from_dict(agent_result)

    def send_turn_fn(prompt_text: str, image, system_prompt: str, history):
        return adapter.send_turn(prompt_text, image, system_prompt, history)

    return run_agent_turn(send_turn_fn, user_text, image, system_prompt, history, image_path=image_path)


def _agent_turn_result_from_dict(data: dict[str, Any]) -> AgentTurnResult:
    """
    Reconstructs the real dataclasses chat_tab.py expects (result.
    steps[i].plan.action, step.tool_result.success, etc - attribute
    access, not dict lookups) from the plain JSON dict a remote
    /chat/turn response carries. The server built that JSON via
    dataclasses.asdict(AgentTurnResult) (see api/agent_main.py's
    chat_turn()), so field names match these dataclasses' own field
    names exactly - this just re-nests them as real instances instead
    of dicts.
    """
    steps = []
    for step in data["steps"]:
        plan = PlanStep(**step["plan"])
        tool_result_data = step.get("tool_result")
        tool_result = ToolResultBlock(**tool_result_data) if tool_result_data is not None else None
        steps.append(AgentStep(plan=plan, tool_result=tool_result))
    return AgentTurnResult(
        steps=steps,
        final_answer=data["final_answer"],
        planner_raw_last=data["planner_raw_last"],
        planner_meta_last=data.get("planner_meta_last") or {},
        answer_meta=data.get("answer_meta"),
    )
