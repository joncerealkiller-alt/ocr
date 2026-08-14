"""
Planner (Phase 2 built single-step-only; Phase 4, 2026-08-13, adds
multi-step chaining). Each step decides one action (answer_directly /
call_tool / ask_user_clarification); call_tool steps loop - the result
of one tool call is shown to the NEXT planning call via a scratchpad, so
the model can chain tools (e.g. classify_document -> extract_fields)
without needing a redesign, per Jon's framing: "you don't have to
redesign anything, just add more tools." Capped at MAX_STEPS so a
confused model can't loop forever - a forced answer beats a hang.

Deliberately decoupled from model_console.ChatBackendAdapter by duck
typing (send_turn_fn), not import - core/ must not depend on
model_console/, only the reverse (see model_console/adapter.py's own
module docstring on the import boundary this mirrors).

Hybrid planner architecture (2026-08-13, Jon's design): the PLANNING
decision (which action/tool to take next) no longer goes through
send_turn_fn/Gemma at all - it's decided by a resident CPU model
(core/cpu_planner.py, Qwen2.5-1.5B-Instruct) with deterministic
Python pre-checks (core/decision_heuristics.py, network_gate) handling
anything code already knows for certain. send_turn_fn/Gemma is now
used only for the FINAL answer synthesis and the no-tool-evidence
direct-answer path, both still genuinely vision/GPU-shaped tasks. This
replaced an earlier all-Gemma-planning design after repeated live
testing showed prompt-only policy tightening (agent_planning_v2/v3)
didn't reliably change small-model judgment calls - see
feedback_planner_needs_web_search_decision_policy memory note for the
confirmed failure that motivated the redesign.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from PIL import Image

from core import cpu_planner
from core.agent_status import status_hub
from core.agent_tools import research_llm
from core.agent_tools.dispatcher import dispatch
from core.agent_tools.web_research_agent import run_web_research
from core.agent_tools.errors import ToolError, ToolErrorClass
from core.agent_tools.registry import TOOL_REGISTRY
from core.agent_tools.schema import ToolCallBlock, ToolResultBlock
from core.decision_heuristics import (
    derive_lookup_query,
    detect_explicit_web_search_request,
    detect_unspecified_recurring_event,
    extract_years,
)
from core.extraction_parsing import parse_kv_block
from core.network_gate import network_gate

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "config" / "prompts"

_VALID_ACTIONS = {"answer_directly", "call_tool", "ask_user_clarification"}

# Hard cap on call_tool steps per turn. Not tuned against real chains
# yet (no tool chain longer than 2 has been exercised live) - 4 leaves
# real headroom for a classify -> extract -> lookup -> lookup shape
# without allowing an unbounded loop if the model keeps re-requesting
# tools the scratchpad already answered.
MAX_STEPS = 4

# Confirmed live (2026-08-13, hybrid CPU planner testing): after a
# SUCCESSFUL web_search or genealogy_framework_lookup call, the CPU
# planner does not reliably recognize "I already have this evidence" -
# it re-calls the same tool with a slightly reworded query instead of
# answering, burning real network calls (web_search) and producing a
# final synthesis fed several near-duplicate results, which visibly
# degraded answer quality (garbled numbers from conflated duplicate
# snippets). The existing circuit breaker only catches repeated
# FAILURES with IDENTICAL args - it doesn't catch this, since a
# reworded query is neither identical nor a failure. These two tools
# are read-only external/lookup calls where a second query within the
# same turn essentially never adds real value over refining the first
# one's evidence in the final-answer step - capped to one successful
# call per turn. extract_fields/classify_document are deliberately NOT
# in this set - a real classify -> extract chain legitimately calls
# different tools once each, and there's no confirmed case yet of
# either of those needing this same guard.
_SINGLE_SHOT_TOOLS = frozenset({"web_search", "genealogy_framework_lookup"})

# SendTurnFn matches model_console.adapter.ChatBackendAdapter.send_turn's
# signature: (prompt_text, image, system_prompt, history) -> (raw_output, meta)
SendTurnFn = Callable[..., tuple[str, dict[str, Any]]]


@dataclass
class PlanStep:
    action: str
    tool_name: Optional[str] = None
    args: dict[str, Any] | None = None
    rationale: str = ""


@dataclass
class AgentStep:
    plan: PlanStep
    tool_result: Optional[ToolResultBlock]


@dataclass
class AgentTurnResult:
    steps: list[AgentStep]
    final_answer: str
    planner_raw_last: str
    planner_meta_last: dict[str, Any] = field(default_factory=dict)
    answer_meta: Optional[dict[str, Any]] = None

    @property
    def plan(self) -> Optional[PlanStep]:
        """Convenience accessor for the FIRST step's plan - kept for
        callers (chat_tab.py) that only care about "what did it decide
        to do", not the full chain. None if the turn ended before any
        step ran (shouldn't happen in practice - even a single
        answer_directly/ask_user_clarification turn produces one step)."""
        return self.steps[0].plan if self.steps else None

    @property
    def tool_result(self) -> Optional[ToolResultBlock]:
        """Convenience accessor for the LAST step's tool result - see
        plan's docstring. None if no call_tool step ran this turn."""
        for step in reversed(self.steps):
            if step.tool_result is not None:
                return step.tool_result
        return None


def _normalize_tool_name(raw: str) -> str:
    """
    Strips whitespace and leading/trailing underscores/dashes before
    matching a candidate tool name against TOOL_REGISTRY. Added
    2026-08-13 (confirmed live): a model wrote `tool_name:_web_search`
    (no space after the colon, then a literal leading underscore before
    the real name - a tokenization/formatting tic, not a deliberate
    different name). Safe to strip generally: no real registered tool
    name in this codebase starts or ends with '_'/'-', so this can
    never cause a false match, only recover an otherwise-lost real one.
    """
    return raw.strip().strip("_- \t")


def _tool_catalog() -> str:
    """
    web_search is structurally hidden from the catalog whenever
    network_gate is off (2026-08-13, hybrid planner design) - the
    model can't choose what it isn't shown, which is a stronger
    guarantee than a prompt instruction telling it not to. Read live
    (not snapshotted) so flipping the UI checkbox mid-turn is reflected
    on the very next planning call, matching network_gate's own
    documented contract.
    """
    lines = []
    for spec in TOOL_REGISTRY.values():
        if spec.name == "web_search" and not network_gate.enabled:
            continue
        lines.append(f"- {spec.name}: {spec.description}")
    return "\n".join(lines) if lines else "(no tools registered)"


def _load_prompt(filename: str, **substitutions: str) -> str:
    text = (PROMPTS_DIR / filename).read_text(encoding="utf-8")
    for key, value in substitutions.items():
        text = text.replace(f"{{{{{key}}}}}", value)
    return text


def _tool_result_text(tool_result: Optional[ToolResultBlock]) -> str:
    if tool_result is None:
        return "(no result)"
    if not tool_result.success:
        return f"FAILED ({tool_result.error['error_class']}): {tool_result.error['message']}"
    return json.dumps(tool_result.result, indent=2)


_EXTRACT_FIELDS_CATEGORIES = frozenset({
    "dense_tabular_rows", "handwritten_ledger", "printed_document", "portrait_photo",
    "map_land_record", "mixed_text_image", "genealogy_chart", "website_screenshot",
    "photo_collage", "casual_photo", "cemetery_photo",
})


def _classify_category_for_extraction(
    send_turn_fn: SendTurnFn,
    image: Optional[Image.Image],
    system_prompt: str,
    history: Optional[list],
) -> Optional[str]:
    """
    A narrow, isolated vision call to pick extract_fields' required
    'category' argument - added 2026-08-13 alongside the CPU-planner
    hybrid design. The CPU planner is text-only (no vision), so it
    cannot do what extract_fields' own description asks the caller to
    do ("judge it yourself from what you can see in the image") -
    confirmed live: the CPU planner correctly decided extract_fields
    was the right tool but left args={} since it genuinely can't see
    the image at all. Rather than let that dispatch fail category-
    missing and force a wasted retry loop, run ONE small, isolated
    vision classification call (Gemma, GPU) - this is exactly the kind
    of task that genuinely needs vision, so it stays on the GPU model,
    just as a narrow single-purpose call rather than folded into the
    old do-everything planning call.
    """
    prompt = _load_prompt("agent_category_classify_v1.txt")
    raw, _meta = send_turn_fn(prompt_text=prompt, image=image, system_prompt=system_prompt, history=history)
    candidate = raw.strip().lower().strip(".").split()[0] if raw.strip() else ""
    if candidate in _EXTRACT_FIELDS_CATEGORIES:
        return candidate
    print(f"[core.agent_tools.planner] category classification call did not return a valid "
          f"category; raw output was: {raw!r}")
    return None


def _readable_evidence_summary(steps: list[AgentStep]) -> str:
    """
    Human-readable fallback text for
    _generate_final_answer_with_year_provenance()'s last-resort case -
    pulls just the 'evidence'/'sources' fields tool results already
    expose (web_search's structured payload, see
    core/agent_tools/web_research_agent.py) rather than dumping the
    full JSON scratchpad at the user, which is technically non-
    fabricated but unreadable. Falls back to the raw tool_result dict
    only for a tool shape that doesn't have an 'evidence' field (a
    genealogy_framework_lookup match list, say) - still correct, just
    less polished for those cases.
    """
    parts = []
    for step in steps:
        if step.tool_result is None or not step.tool_result.success:
            continue
        result = step.tool_result.result or {}
        if "evidence" in result:
            block = result["evidence"]
            if result.get("sources"):
                block += f"\n\nSources: {result['sources']}"
            parts.append(block)
        else:
            parts.append(json.dumps(result, indent=2))
    return "\n\n---\n\n".join(parts) if parts else "(no evidence available)"


def _generate_final_answer_with_year_provenance(
    generate_fn: Callable[[str], tuple[str, dict[str, Any]]],
    final_input: str,
    evidence_text: str,
    readable_evidence: Optional[str] = None,
) -> tuple[str, dict[str, Any]]:
    """
    Confirmed live 2026-08-13: even fed a clean, provenance-verified
    evidence bundle (core/agent_tools/web_research_agent.py's own
    quarantine, applied one layer earlier, was already working
    correctly), the OUTER final-answer synthesis step still fabricated
    years on its own - "1926" became "11926"/"19226"/other wrong
    4-digit strings, confirmed NOT explained by restrict_output_charset
    or repetition_penalty (A/B/C/D decoding-config comparison against
    the identical clean input all fabricated different wrong years) -
    a genuine model-capability limit on free-form numeric paraphrase,
    not a config problem. Same "quarantine, don't silently pass
    through" principle as the research-bundle check, applied here:

    1. Generate normally.
    2. If the answer mentions a year not present anywhere in the
       evidence it was given, retry ONCE with an explicit correction
       naming the specific wrong year(s) - a retry-with-correction is
       given one real chance, since it's cheap and sometimes works, but
       is NOT trusted as the guarantee (this project's repeated finding
       this session: prompt-only self-correction doesn't reliably
       hold).
    3. If the retry still fabricates a year, don't trust a THIRD
       paraphrase - fall back to showing the raw evidence text
       directly rather than any further free-form rewording.

    `generate_fn` is model-agnostic by design (2026-08-13, Jon's
    principle: "keep Gemma as the VLM front end, but if it's a research
    task that's text only, no vision required, handing back to Gemma
    doesn't make sense for synthesis") - the caller decides which model
    answers based on whether the turn actually has an image (see the
    two call-site branches in _run_agent_turn_inner), this function
    itself has no opinion on that, it only needs a single generate(str)
    -> (str, meta) callable, whichever model backs it.
    """
    answer_raw, answer_meta = generate_fn(final_input)
    evidence_years = extract_years(evidence_text)
    unverified = extract_years(answer_raw) - evidence_years
    if not unverified:
        return answer_raw, answer_meta

    print(f"[core.agent_tools.planner] final answer mentioned year(s) {sorted(unverified)} "
          f"not present in the tool evidence - retrying once with an explicit correction. "
          f"Raw answer was: {answer_raw!r}")
    correction_input = (
        f"{final_input}\n\nCORRECTION: your previous answer mentioned year(s) "
        f"{sorted(unverified)} that do not appear anywhere in the evidence above - "
        f"that is a fabrication, not a fact from the sources. Rewrite your answer "
        f"using ONLY years that literally appear in the evidence above. If you are "
        f"not sure of an exact year, say so plainly instead of guessing one."
    )
    retry_raw, retry_meta = generate_fn(correction_input)
    if not (extract_years(retry_raw) - evidence_years):
        return retry_raw, retry_meta

    print(f"[core.agent_tools.planner] retry still fabricated year(s) "
          f"{sorted(extract_years(retry_raw) - evidence_years)} - not trusting a third "
          f"paraphrase, falling back to the raw evidence text directly. Retry answer "
          f"was: {retry_raw!r}")
    fallback = (
        "I found relevant information, but I can't reliably restate the exact year(s) "
        "in my own words without risking a transcription error, so here is the "
        "evidence directly from the source(s):\n\n" + (readable_evidence or evidence_text)
    )
    return fallback, retry_meta


def _scratchpad_text(steps: list[AgentStep]) -> str:
    if not steps:
        return "(none yet - this is the first step)"
    lines = []
    for i, step in enumerate(steps, 1):
        lines.append(
            f"Step {i}: called {step.plan.tool_name!r} "
            f"(rationale: {step.plan.rationale})\nResult:\n{_tool_result_text(step.tool_result)}"
        )
    return "\n\n".join(lines)


def parse_plan_step(raw_output: str) -> PlanStep:
    """
    Parses the planner prompt's 'field: value' output via the same
    parse_kv_block() every extraction loader already uses - one parsing
    contract for this style of output, not a second implementation.
    Falls back to answer_directly (never crashes the turn) if the model's
    output doesn't parse or names an unregistered/malformed action -
    an unparseable plan is treated as "nothing decisive to do", not a
    hard failure, matching this project's abstain-over-guess discipline.
    """
    try:
        fields = parse_kv_block(raw_output)
    except ValueError as e:
        # Printed, not just returned in the rationale - the fallback
        # rationale alone was found (2026-08-13, live agent-mode test)
        # to be useless for debugging WHY a small model's output didn't
        # parse, since the raw text itself was never persisted anywhere
        # (session log only keeps the FINAL answer, not intermediate
        # planning calls). This is diagnostic-only - never raises, never
        # changes the fallback behavior below.
        print(f"[core.agent_tools.planner] planner output failed to parse ({e}); "
              f"raw output was:\n{raw_output!r}")
        return PlanStep(action="answer_directly", rationale="Planner output malformed; falling back.")

    action = fields.get("action", "").strip().lower()
    tool_name = _normalize_tool_name(fields.get("tool_name", "")) or None

    if action not in _VALID_ACTIONS:
        # Recovery, not just a fallback (2026-08-13, confirmed live):
        # a model can write the TOOL NAME into the action field itself
        # ("action: web_search" instead of "action: call_tool" /
        # "tool_name: web_search") - especially likely for a tool whose
        # name reads like a verb phrase. Rather than discard a
        # perfectly clear intent because of a field-placement mixup,
        # check whether the invalid action value is itself a real
        # registered tool name and recover as call_tool - the same
        # "recover the model's clear intent structurally rather than
        # discard it over a formatting slip" principle already applied
        # to multi-line JSON args parsing above.
        normalized_action = _normalize_tool_name(action)
        if normalized_action in TOOL_REGISTRY:
            print(f"[core.agent_tools.planner] planner wrote tool name {action!r} in the "
                  f"action field - recovering as call_tool.")
            tool_name = tool_name or normalized_action
            action = "call_tool"
        elif normalized_action.startswith("call_tool") and normalized_action != "call_tool":
            # Confirmed live with the CPU planner (2026-08-13): "action:
            # call_tool extract_fields" - the valid action word plus the
            # tool name appended directly onto the same field instead of
            # using tool_name separately. Same recovery principle as the
            # branch above (structural slip, not a discarded intent) -
            # strip the "call_tool" prefix and check what's left against
            # the registry.
            suffix = _normalize_tool_name(normalized_action[len("call_tool"):])
            if suffix in TOOL_REGISTRY:
                print(f"[core.agent_tools.planner] planner appended tool name to the "
                      f"action field ({action!r}) - recovering as call_tool.")
                tool_name = tool_name or suffix
                action = "call_tool"
            else:
                print(f"[core.agent_tools.planner] planner named unrecognized action {action!r}; "
                      f"raw output was:\n{raw_output!r}")
                return PlanStep(action="answer_directly", rationale="Planner named an unrecognized action; falling back.")
        else:
            print(f"[core.agent_tools.planner] planner named unrecognized action {action!r}; "
                  f"raw output was:\n{raw_output!r}")
            return PlanStep(action="answer_directly", rationale="Planner named an unrecognized action; falling back.")
    # NOT fields.get("args") - parse_kv_block() is line-oriented (one
    # "key: value" per line), so a model that pretty-prints its args as
    # multi-line JSON (confirmed live, 2026-08-13 - "args: {\n  ...\n}")
    # gets truncated to just "{" by parse_kv_block, since the closing
    # "}" and inner lines don't look like "key: value" lines and get
    # silently dropped. Re-extract args directly from raw_output instead,
    # capturing everything between "args:" and the next known field
    # label (or end of string) so multi-line JSON survives intact - the
    # single-line case (the common one) still works identically, this
    # only adds tolerance for a well-formed model that spans lines.
    # [ \t]* (not \s*) right after "args:" - must NOT consume the
    # newline itself, or a genuinely blank args field ("args: \n
    # rationale: ...") swallows "rationale: ..." into args_raw instead
    # of matching the terminator on the very next line (caught by a
    # regression test before this shipped - see test_planner_args_parsing.py).
    args_match = re.search(r"args:[ \t]*(.*?)(?:\n\s*rationale:|\Z)", raw_output, re.DOTALL | re.IGNORECASE)
    args_raw = args_match.group(1).strip() if args_match else ""
    # Blank args is valid (and expected for tools whose only argument is
    # file_path - see cpu_planner_v1.txt: the model is told to omit
    # file_path since it can't know the real path) - {} means "no known
    # args", distinct from args=None which means "failed to parse".
    args: dict[str, Any] | None = {} if not args_raw else None
    if args_raw:
        try:
            args = json.loads(args_raw)
        except json.JSONDecodeError:
            # Windows file paths contain raw backslashes, which are not
            # valid JSON escapes ('\S' etc.) - a model emitting
            # {"file_path": "J:\Screenshots\..."} produces technically
            # invalid JSON despite being an entirely reasonable answer
            # on this platform. Retry once with backslashes doubled
            # before giving up, rather than falling back on every
            # Windows-path tool call.
            try:
                args = json.loads(re.sub(r"\\(?!\\)", r"\\\\", args_raw))
            except json.JSONDecodeError:
                print(f"[core.agent_tools.planner] planner's args field did not parse as "
                      f"JSON (even after the Windows-backslash retry) for tool "
                      f"{tool_name!r}; raw args text was:\n{args_raw!r}\nfull raw "
                      f"output was:\n{raw_output!r}")
                args = None

    if action == "call_tool":
        if not tool_name or tool_name not in TOOL_REGISTRY:
            # Recovery, generalized (2026-08-13, confirmed live): a
            # model can invent a near-miss FIELD NAME for the tool_name
            # field itself ("tool_name_call: web_search" instead of
            # "tool_name: web_search") - a different mixup than the
            # action-field-holds-a-tool-name case above, but the same
            # underlying pattern (the model's INTENT is clear, only the
            # exact field label is wrong). Rather than hardcode a narrow
            # alias for this one exact typo (this project's own
            # precedent for that, core/extraction_parsing.py's
            # _KEY_ALIASES, requires TWO independent confirmed
            # observations before adding one - this is only one so
            # far), scan every parsed field for ANY value that happens
            # to be a real registered tool name and recover with that -
            # covers this typo and any other future near-miss field
            # name for free, without needing to enumerate them.
            recovered = next(
                (_normalize_tool_name(v) for k, v in fields.items()
                 if k != "action" and _normalize_tool_name(v) in TOOL_REGISTRY),
                None,
            )
            if recovered is not None:
                print(f"[core.agent_tools.planner] planner used a non-standard field name "
                      f"for tool_name (raw fields: {fields!r}) - recovered {recovered!r} "
                      f"from another field's value.")
                tool_name = recovered
            else:
                # Missing this diagnostic was itself a real gap
                # (2026-08-13, confirmed live) - this is a 4th distinct
                # parse-fallback branch, and the other three already had
                # a print() here; this one didn't, so a real live
                # occurrence of it produced no raw text to debug from at
                # all.
                print(f"[core.agent_tools.planner] planner requested unknown/missing tool "
                      f"{tool_name!r} with action=call_tool; raw output was:\n{raw_output!r}")
                return PlanStep(
                    action="answer_directly",
                    rationale=f"Planner requested unknown tool {tool_name!r}; falling back.",
                )
        if args is None:
            return PlanStep(
                action="answer_directly",
                rationale="Planner's tool args did not parse as JSON; falling back.",
            )

    return PlanStep(
        action=action,
        tool_name=tool_name,
        args=args,
        rationale=fields.get("rationale", "").strip(),
    )


def run_agent_turn(
    send_turn_fn: SendTurnFn,
    user_text: str,
    image: Optional[Image.Image],
    system_prompt: str,
    history: Optional[list] = None,
    image_path: Optional[str] = None,
) -> AgentTurnResult:
    """Thin status-clearing wrapper around _run_agent_turn_inner() - a
    try/finally here guarantees active_step/operation reset to idle on
    EVERY exit path (there are three: ask_user_clarification, the
    no-tool-evidence answer, and the tool-evidence final answer) without
    duplicating the reset at each individual return statement."""
    try:
        return _run_agent_turn_inner(
            send_turn_fn, user_text, image, system_prompt, history, image_path,
        )
    finally:
        status_hub.update(active_step=None, operation="idle")


def _run_agent_turn_inner(
    send_turn_fn: SendTurnFn,
    user_text: str,
    image: Optional[Image.Image],
    system_prompt: str,
    history: Optional[list] = None,
    image_path: Optional[str] = None,
) -> AgentTurnResult:
    """
    One full agent turn: loop (plan -> act)* -> answer, up to MAX_STEPS
    call_tool steps. Each planning call sees a scratchpad of every step
    already taken THIS turn, so it can chain tools (classify_document ->
    extract_fields -> ...) or stop once it has enough. Raises nothing
    tool-related - tool failures surface inside a step's
    tool_result.success=False and still let the loop continue (the
    model sees the failure and can react to it).

    image_path: the CURRENTLY-ATTACHED image's real on-disk path, when
    known. The planner prompt never tells the model this path (the model
    only sees pixels, not a filesystem location - see the vision
    architecture's image-reference design), so a model-guessed
    "file_path" arg is not trustworthy. When a tool's input schema wants
    a "file_path" field, this always overrides whatever the model put
    there with the real path rather than trusting the guess - confirmed
    necessary 2026-08-13 (first live agent-mode run: classify_document
    failed because the model had no way to know the real path and
    fabricated a plausible-looking one).
    """
    steps: list[AgentStep] = []
    planner_raw = ""
    planner_meta: dict[str, Any] = {}
    image_status = "YES, an image is attached" if image is not None else "no image attached"

    while len(steps) < MAX_STEPS:
        # Deterministic bypass (2026-08-13, hybrid planner design): an
        # explicit "search online"/"google it" style request is
        # unambiguous enough that no model judgment is needed at all -
        # skip the CPU planner call entirely and go straight to
        # web_search. Only applies on the FIRST step of a turn (a later
        # step's scratchpad context matters more than the raw wording
        # by then) and only when internet access is actually on -
        # otherwise fall through to the CPU planner as normal, which
        # will find web_search absent from its catalog and has to
        # decide something else.
        if (
            len(steps) == 0
            and network_gate.enabled
            and detect_explicit_web_search_request(user_text)
        ):
            plan = PlanStep(
                action="call_tool",
                tool_name="web_search",
                args={"query": user_text},
                rationale="Deterministic: user's wording explicitly requested a web/online search.",
            )
            planner_raw = "(deterministic bypass - explicit web search request, no model call)"
            planner_meta = {"model": "deterministic", "device": "n/a", "latency_s": 0.0}
        elif len(steps) == 0 and detect_unspecified_recurring_event(user_text):
            # Second deterministic bypass (2026-08-13): confirmed live
            # that even the 1.5B CPU planner missed this exact
            # ambiguity pattern once the full 5-tool production catalog
            # (longer, more emphatic descriptions) replaced the smaller
            # catalog it was first tuned against - see
            # decision_heuristics.detect_unspecified_recurring_event's
            # own docstring. Narrow and high-precision by construction,
            # so routing straight to ask_user_clarification here is
            # safe rather than gambling on further prompt tuning.
            plan = PlanStep(
                action="ask_user_clarification",
                rationale=(
                    "Deterministic: this asks for THE year/date of something that "
                    "happens repeatedly, without saying which instance is meant."
                ),
            )
            planner_raw = "(deterministic bypass - unspecified recurring event, no model call)"
            planner_meta = {"model": "deterministic", "device": "n/a", "latency_s": 0.0}
        else:
            status_hub.emit(
                "generation_started", operation="planning_cpu", active_step=f"{len(steps) + 1}/{MAX_STEPS}",
            )
            planning_prompt = _load_prompt(
                "cpu_planner_v1.txt",
                TOOL_CATALOG=_tool_catalog(),
                IMAGE_STATUS=image_status,
                SCRATCHPAD=_scratchpad_text(steps),
            )
            planner_input = f"{planning_prompt}\n\nUser message: {user_text}"
            planner_raw, planner_meta = cpu_planner.generate_plan(planner_input)
            status_hub.emit("generation_finished", operation="idle")
            plan = parse_plan_step(planner_raw)

        if plan.action == "ask_user_clarification":
            question = plan.rationale or "Could you clarify which document or person you mean?"
            steps.append(AgentStep(plan, None))
            return AgentTurnResult(steps, question, planner_raw, planner_meta)

        if plan.action == "answer_directly":
            steps.append(AgentStep(plan, None))
            break

        # action == "call_tool"
        #
        # Circuit breaker (2026-08-13): a code-level guarantee, not just
        # the "don't repeat a failed call" instruction in the planning
        # prompt - that instruction alone was confirmed live to NOT
        # reliably stop a small model from re-issuing the
        # EXACT SAME (tool_name, args) pair after it already failed
        # identically. For a GPU tool this is expensive, not just
        # wasteful: core/model_residency.py's borrow() releases the chat
        # model and loads the tool's model on every attempt, so a blind
        # retry loop forces a full swap-out/swap-back cycle each time
        # for zero new information. Comparing on the model's OWN
        # pre-injection args (not the post-file_path-injection call_args)
        # is sufficient - file_path injection is deterministic per turn
        # (same image_path every time), so identical tool_name+args here
        # always means identical post-injection args too.
        already_failed_identically = any(
            s.plan.tool_name == plan.tool_name and s.plan.args == plan.args
            and s.tool_result is not None and not s.tool_result.success
            for s in steps
        )
        if already_failed_identically:
            plan = PlanStep(
                action="answer_directly",
                rationale=(
                    f"Refusing to repeat the identical failed call to "
                    f"{plan.tool_name!r} with the same arguments - see the "
                    f"scratchpad for why it failed; retrying without changing "
                    f"anything would not help."
                ),
            )
            steps.append(AgentStep(plan, None))
            break

        already_succeeded_this_turn = plan.tool_name in _SINGLE_SHOT_TOOLS and any(
            s.plan.tool_name == plan.tool_name and s.tool_result is not None and s.tool_result.success
            for s in steps
        )
        if already_succeeded_this_turn:
            plan = PlanStep(
                action="answer_directly",
                rationale=(
                    f"Already have a successful {plan.tool_name!r} result from earlier "
                    f"this turn - answering from that evidence instead of searching again."
                ),
            )
            steps.append(AgentStep(plan, None))
            break

        call_args = dict(plan.args or {})
        tool_spec = TOOL_REGISTRY.get(plan.tool_name)
        if image_path and tool_spec is not None and "file_path" in tool_spec.input_schema.model_fields:
            call_args["file_path"] = image_path
        if plan.tool_name == "genealogy_framework_lookup" and not call_args.get("query"):
            # Confirmed live (2026-08-13): CPU planner picks this tool
            # correctly but sometimes leaves args={} - see
            # decision_heuristics.derive_lookup_query's docstring.
            call_args["query"] = derive_lookup_query(user_text)
        if plan.tool_name == "extract_fields" and not call_args.get("category"):
            # The CPU planner has no vision - see
            # _classify_category_for_extraction's docstring. Only runs
            # when the model didn't already supply one (a real image-
            # bearing categorization it happened to get right shouldn't
            # be second-guessed by a redundant extra call).
            status_hub.emit("generation_started", operation="generating", active_step="classify_category")
            category = _classify_category_for_extraction(send_turn_fn, image, system_prompt, history)
            status_hub.emit("generation_finished", operation="idle")
            if category:
                call_args["category"] = category
        if plan.tool_name == "web_search":
            # Skill handoff (2026-08-13, Jon's design): web_search is no
            # longer a flat one-shot call - control passes to
            # core.agent_tools.web_research_agent's own bounded loop (up
            # to MAX_SEARCHES real searches, refine/stop judgment,
            # evidence-vs-inference synthesis - now on a dedicated
            # text-only research LLM, not the chat model - see that
            # module's docstring), and returns ONE ToolResultBlock
            # exactly as before. The CPU planner still only invokes this
            # once per turn (the single-shot circuit breaker above still
            # applies, now as a second, redundant safeguard rather than
            # the only one) - all the actual multi-search bounding
            # happens inside the skill.
            tool_result = run_web_research(
                question=user_text, initial_query=call_args.get("query", user_text),
            )
        else:
            call = ToolCallBlock(tool_name=plan.tool_name, args=call_args)
            tool_result = dispatch(call)
        steps.append(AgentStep(plan, tool_result))
    else:
        # Hit MAX_STEPS without the model choosing answer_directly - force
        # a stop rather than looping indefinitely. This is a real, visible
        # outcome (not silently truncated) - the final-answer prompt below
        # still sees every step's result, it just never got an explicit
        # "I'm done" signal from the planner.
        pass

    tool_steps = [s for s in steps if s.tool_result is not None]
    if not tool_steps:
        # Every step was answer_directly on the first pass (the common,
        # cheap case) - no tool evidence to fold in, answer straight from
        # the conversation exactly as Phase 2 did.
        status_hub.emit("generation_started", operation="generating", active_step="answer")
        answer_raw, answer_meta = send_turn_fn(
            prompt_text=user_text, image=image, system_prompt=system_prompt, history=history,
        )
        status_hub.emit("generation_finished", operation="idle")
        return AgentTurnResult(steps, answer_raw, planner_raw, planner_meta, answer_meta)

    scratchpad_text = _scratchpad_text(steps)
    final_prompt = _load_prompt("agent_final_answer_v2.txt", STEPS_TEXT=scratchpad_text)
    final_input = f"{final_prompt}\n\nUser's message: {user_text}"
    readable_evidence = _readable_evidence_summary(steps)
    status_hub.emit("generation_started", operation="generating", active_step="answer")

    if image is None:
        # Text-only synthesis - dedicated research LLM, not the VLM
        # (2026-08-13, Jon's principle: "keep Gemma as the VLM front
        # end, but if it's a research task that's text only, no vision
        # required, handing back to Gemma doesn't make sense for
        # synthesis"). One borrow covers both the initial attempt and
        # the correction retry (research_llm.borrowed()'s own
        # efficiency guarantee - see its docstring), so this costs at
        # most one GPU residency swap regardless of whether a retry
        # happens.
        with research_llm.borrowed() as research_generate:
            def generate_fn(prompt_text: str) -> tuple[str, dict[str, Any]]:
                return research_generate(prompt_text), {"model": research_llm.MODEL_NAME}
            answer_raw, answer_meta = _generate_final_answer_with_year_provenance(
                generate_fn, final_input, scratchpad_text, readable_evidence=readable_evidence,
            )
    else:
        # An image IS attached this turn - the final answer may need to
        # reference it (e.g. cross-checking a tool result against what's
        # actually visible), so this genuinely needs the VLM.
        def generate_fn(prompt_text: str) -> tuple[str, dict[str, Any]]:
            return send_turn_fn(
                prompt_text=prompt_text, image=image, system_prompt=system_prompt, history=history,
            )
        answer_raw, answer_meta = _generate_final_answer_with_year_provenance(
            generate_fn, final_input, scratchpad_text, readable_evidence=readable_evidence,
        )

    status_hub.emit("generation_finished", operation="idle")
    return AgentTurnResult(steps, answer_raw, planner_raw, planner_meta, answer_meta)
