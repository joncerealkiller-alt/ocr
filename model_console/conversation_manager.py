"""
Rolling conversation summarization (Phase 3 of the agent architecture
plan). Sits above ChatBackendAdapter/ChatSession, same layer as
agent_bridge.py - imports both sides, core/ knows nothing about this.

Policy (matches the architecture doc's §1 token-growth strategy):
  - Below SUMMARY_TRIGGER_TOKENS of unsummarized conversation AND below
    VRAM_PRESSURE_TRIGGER fraction of GPU memory in use: do nothing,
    pass raw turns through exactly as build_history_for_context already
    does.
  - Above EITHER trigger: summarize every unsummarized turn EXCEPT the
    most recent KEEP_RECENT_TURNS, via one extra model call using
    config/prompts/agent_summarize_v1.txt. The summary EXTENDS any
    prior summary (never re-derived from scratch - see ChatSession.
    summary_text's own docstring), and session.summarized_up_to_turn_id
    advances to the last turn folded in.
  - The summary text is injected into the SYSTEM PROMPT for that call
    (not as a fake user/assistant history turn), since it's a
    compressed fact-sheet, not something either party "said".

VRAM trigger (2026-08-13, added after a live test on a 16GB card): a
fixed token count alone doesn't know what card it's running on - a
budget tuned for a 16GB card is needlessly conservative on a 24GB one,
and (the case that actually motivated this) can be too LOOSE on a card
already near its ceiling, since KV-cache memory pressure and token
count don't move in lockstep once other things (a large recent
image-heavy turn, a bigger max_new_tokens reservation) are also eating
VRAM. Checking live GPU memory pressure directly, alongside the token
count rather than instead of it, makes the trigger adapt to whatever
hardware it's actually running on. Confirmed live: a session sat at
~96% VRAM used while its real (tokenizer-counted, not the char/4
estimate) unsummarized-chunk token count stayed under
SUMMARY_TRIGGER_TOKENS the whole time - the token-only trigger would
never have fired before an OOM became a real risk.

This module mutates the ChatSession it's given (summary_text/
summarized_up_to_turn_id) - callers should persist the session
afterward (session_log.write_session_meta) the same way any other
session-state change is persisted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from model_console.adapter import ChatBackendAdapter
from model_console.session import ChatSession, ChatTurn

try:
    import torch
except ImportError:  # pragma: no cover - torch is a hard runtime dependency
    torch = None  # noqa: N816

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "config" / "prompts"

# Token thresholds are deliberately conservative relative to any single
# model's real context window (see adapter.py's _resolve_context_length)
# - summarization firing a bit early costs one extra small model call;
# firing too late risks build_history_for_context() silently dropping
# whole turns instead of compressing them. Not tied to a specific
# model's budget on purpose - Phase 3 scope is "summarize before things
# get dropped", not per-model tuning.
SUMMARY_TRIGGER_TOKENS = 3000
KEEP_RECENT_TURNS = 6

# Fraction of TOTAL device memory in use (not just this process's
# allocation - torch.cuda.mem_get_info() reports true free/total, so
# this also accounts for anything else sharing the GPU) above which
# summarization fires regardless of estimated token count. 0.85 leaves
# real headroom for the summarization call's own generation plus
# whatever the next real turn needs - not cut so close that the safety
# margin is itself consumed by the act of checking it.
VRAM_PRESSURE_TRIGGER = 0.85


def _vram_pressure_fraction() -> Optional[float]:
    """Returns used/total GPU memory as a fraction, or None if no CUDA
    device is available (e.g. CPU-only dev environment) - callers must
    treat None as "can't assess VRAM pressure", not as 0.0."""
    if torch is None or not torch.cuda.is_available():
        return None
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return (total_bytes - free_bytes) / total_bytes
    except Exception:
        return None


def _load_prompt(filename: str, **substitutions: str) -> str:
    text = (PROMPTS_DIR / filename).read_text(encoding="utf-8")
    for key, value in substitutions.items():
        text = text.replace(f"{{{{{key}}}}}", value)
    return text


def _turns_to_text(turns: list[ChatTurn]) -> str:
    lines = []
    for t in turns:
        text = t.text or ""
        if t.image is not None:
            text = f"{text}\n[an image was attached to this message]" if text else "[an image was attached to this message]"
        lines.append(f"{t.role}: {text}")
    return "\n".join(lines)


def maybe_summarize_session(
    adapter: ChatBackendAdapter, session: ChatSession, turns: list[ChatTurn],
) -> bool:
    """
    Checks whether `turns` (a caller-supplied snapshot - see
    session.filter_unsummarized()'s docstring for why this must be a
    snapshot, not live session.turns) has enough unsummarized content to
    exceed the trigger budget and, if so, runs one summarization call
    and advances session.summary_text/summarized_up_to_turn_id in place.
    Returns True if a summarization pass actually ran (False = under
    budget, no-op).

    Requires adapter.ensure_loaded() to already have been called - uses
    the same resident model as the rest of the turn, one-shot (history=
    None), since the summarizer only ever needs the turns it's given,
    never the full session's own history.
    """
    unsummarized = session.filter_unsummarized(turns)
    if len(unsummarized) <= KEEP_RECENT_TURNS:
        return False  # nothing old enough to be worth summarizing yet

    to_summarize = unsummarized[:-KEEP_RECENT_TURNS]
    turns_text = _turns_to_text(to_summarize)
    total_tokens = adapter.estimate_token_count(turns_text)
    vram_pressure = _vram_pressure_fraction()

    over_token_budget = total_tokens >= SUMMARY_TRIGGER_TOKENS
    over_vram_budget = vram_pressure is not None and vram_pressure >= VRAM_PRESSURE_TRIGGER
    if not (over_token_budget or over_vram_budget):
        return False

    prompt = _load_prompt(
        "agent_summarize_v1.txt",
        PRIOR_SUMMARY=session.summary_text or "(none yet)",
        TURNS_TEXT=turns_text,
    )
    try:
        new_summary, _meta = adapter.send_turn(
            prompt_text=prompt, image=None, system_prompt="", history=None,
        )
    except Exception as e:
        # A VRAM-pressure-triggered summarization pass is, by definition,
        # attempted when the GPU is already close to its ceiling - the
        # summarization call's own generation pass can itself OOM. Don't
        # propagate: the user is still waiting on their REAL answer this
        # turn, and that turn's own OOM-retry-once logic (adapter.py's
        # send_turn) is a separate, more important safety net than this
        # one. Skip this pass; the next turn re-checks both triggers.
        if torch is not None and torch.cuda.is_available() and "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return False
        raise
    session.summary_text = new_summary.strip()
    session.summarized_up_to_turn_id = to_summarize[-1].turn_id
    return True


def get_effective_context(
    session: ChatSession, turns: list[ChatTurn], system_prompt: str,
) -> tuple[str, list[ChatTurn]]:
    """
    Returns (augmented_system_prompt, recent_raw_turns) - what a caller
    should actually pass to adapter.send_turn()/run_agent_chat_turn() as
    system_prompt and history, after accounting for any summarization
    already done. `turns` is the same caller-supplied snapshot passed to
    maybe_summarize_session() - call that first if a fresh pass might be
    needed; this function itself has no side effects.
    """
    recent = session.filter_unsummarized(turns)
    if not session.summary_text:
        return system_prompt, recent

    summary_block = (
        "\n\n[Summary of earlier conversation, provided as background - "
        "not something the user just said]:\n" + session.summary_text
    )
    return system_prompt + summary_block, recent
