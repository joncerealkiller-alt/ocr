"""
On-disk persistence for model_console chat sessions - raw prompts,
settings, attached image metadata, model/profile used, and raw
responses, written incrementally so a crash mid-session doesn't lose
prior turns.

Path resolved directly off WorkspaceContext, following the live
precedent set by core/calibration_workspace.py (updated 2026-08-09 for
exactly this reason - "why does every workspace path get hardcoded
separately"). WorkspaceContext itself has no research_dir() accessor;
research/{benchmarks,experiments,baselines,calibration} is a directory
convention from the Phase 2 migration (docs/RUN_ARCHITECTURE.md), not
a code-backed API, so a new persistent/non-run module composes its own
subdirectory the same way calibration_workspace.py does. No
data/outputs/ reference anywhere in this module - model_console is a
new subsystem with no legacy path to stay backward-compatible with.

Save-failure discipline mirrors core/debug_dump.py: every public save
method catches its own exceptions, prints a WARNING, and returns -
never raises. A session-log write failure must not be able to abort a
live chat.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from core.workspace_context import WorkspaceContext
from model_console.session import ChatSession, ChatTurn

SESSIONS_ROOT = WorkspaceContext.resolve().workspace_root / "research" / "model_console" / "chat_sessions"


def _iso(value: Any) -> Any:
    return value.isoformat() if isinstance(value, datetime) else value


def session_dir(session_id: str) -> Path:
    return SESSIONS_ROOT / session_id


def _turn_basename(index: int, turn: ChatTurn) -> str:
    return f"{index:04d}_{turn.role}"


def write_session_meta(session: ChatSession) -> None:
    """Rewritten after every turn (small, cheap) so a UI session
    browser can show live sessions - turn count/updated_at - without
    parsing every turn file."""
    try:
        _write_session_meta_unsafe(session)
    except Exception as e:
        print(f"[model_console.session_log] WARNING: failed to write session "
              f"metadata for {session.session_id!r}: {type(e).__name__}: {e}")


def _write_session_meta_unsafe(session: ChatSession) -> None:
    out_dir = session_dir(session.session_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "session_id": session.session_id,
        "model_name": session.model_name,
        "profile_name": session.profile_name,
        "created_at": _iso(session.created_at),
        "updated_at": _iso(datetime.now(session.created_at.tzinfo)),
        "system_prompt": session.system_prompt,
        "generation_overrides": session.generation_overrides,
        "turn_count": len(session.turns),
        "summary_text": session.summary_text,
        "summarized_up_to_turn_id": session.summarized_up_to_turn_id,
    }
    with open(out_dir / "session.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_turn(session: ChatSession, turn: ChatTurn) -> None:
    """
    Writes one turn's JSON (and its image, if attached) immediately
    after the turn completes - not buffered to end-of-session - then
    refreshes session.json. `turn` must already be appended to
    session.turns (its index in that list determines the on-disk
    filename ordering).
    """
    try:
        _write_turn_unsafe(session, turn)
    except Exception as e:
        print(f"[model_console.session_log] WARNING: failed to save turn "
              f"{turn.turn_id!r} for session {session.session_id!r}: "
              f"{type(e).__name__}: {e}")
        return
    write_session_meta(session)


def _write_turn_unsafe(session: ChatSession, turn: ChatTurn) -> None:
    index = session.turns.index(turn)
    out_dir = session_dir(session.session_id) / "turns"
    out_dir.mkdir(parents=True, exist_ok=True)
    basename = _turn_basename(index, turn)

    image_ref = None
    if turn.image is not None:
        image_ref = f"{basename}_image.png"
        turn.image.pil_image.save(out_dir / image_ref)

    payload: dict[str, Any] = {
        "turn_id": turn.turn_id,
        "role": turn.role,
        "text": turn.text,
        "timestamp": _iso(turn.timestamp),
        "image_ref": image_ref,
        "image_display_name": turn.image.display_name if turn.image else None,
        "image_sha256": turn.image.sha256 if turn.image else None,
    }
    if turn.role == "assistant":
        payload.update({
            "raw_output": turn.raw_output,
            "generation_config_snapshot": turn.generation_config_snapshot,
            "model_name": turn.model_name,
            "profile_name": turn.profile_name,
            "runtime_seconds": turn.runtime_seconds,
            "error": turn.error,
            # Conversation-context provenance (2026-08-11) - answers
            # "what did the model actually receive," not just "was
            # context on." context_enabled=None means context mode
            # wasn't in play for this turn (pre-context-mode / one-shot
            # default); False/True only appear once context mode exists.
            "context_enabled": turn.context_enabled,
            "history_turn_ids": turn.history_turn_ids,
            "history_turn_count": turn.history_turn_count,
            "approx_input_tokens": turn.approx_input_tokens,
            "dropped_turn_ids": turn.dropped_turn_ids,
            "telemetry": turn.telemetry,
        })

    with open(out_dir / f"{basename}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
