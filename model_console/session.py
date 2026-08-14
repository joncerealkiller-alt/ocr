"""
Pure data model for model_console chat sessions - no tkinter, no torch,
no imports from core.loaders/core.loader_registry. Only adapter.py and
the UI layer are allowed to know those exist (see docs/CODE_MAP.md's
"config owns behavior, loaders own mechanics" split, which this package
is deliberately built not to cross).

ChatSession/ChatTurn mirror scripts/model_assessment.py's own shape
("profile settings snapshot + overrides applied last, on an in-memory
copy"), generalized from a single test run to a multi-turn transcript.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from PIL import Image


def _new_id() -> str:
    return uuid.uuid4().hex


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class ChatImage:
    """An image attached to one chat turn, already prepped for a loader
    call (see image_prep.prep_for_chat - not built here)."""

    pil_image: Image.Image
    display_name: str
    width: int
    height: int
    sha256: str
    source_path: Optional[Path] = None

    @classmethod
    def from_pil(cls, pil_image: Image.Image, display_name: str,
                 source_path: Optional[Path] = None) -> "ChatImage":
        """Computes sha256 once at construction time, off the encoded
        bytes actually held in memory - used for session-log correlation
        and "same image already attached" detection in the UI, not a
        security control."""
        digest = hashlib.sha256(pil_image.tobytes()).hexdigest()
        return cls(
            pil_image=pil_image,
            display_name=display_name,
            width=pil_image.width,
            height=pil_image.height,
            sha256=digest,
            source_path=source_path,
        )


@dataclass
class ChatTurn:
    role: str  # "user" | "assistant"
    text: str
    turn_id: str = field(default_factory=_new_id)
    image: Optional[ChatImage] = None
    timestamp: datetime = field(default_factory=_utcnow)

    # Assistant-only fields - None on every user turn.
    raw_output: Optional[str] = None
    generation_config_snapshot: Optional[dict[str, Any]] = None
    model_name: Optional[str] = None
    profile_name: Optional[str] = None
    runtime_seconds: Optional[float] = None
    error: Optional[str] = None

    # Conversation-context provenance (2026-08-11) - set only on assistant
    # turns generated while the context checkbox was on. None means
    # "context mode wasn't in play for this turn" (the one-shot default),
    # distinct from context_enabled=True with an empty history list (context
    # was on but this was the first turn, nothing to supply yet). Exists so
    # a session log can answer "what did the model actually receive," not
    # just "was context on" - see model_console/adapter.py's
    # build_history_for_context().
    context_enabled: Optional[bool] = None
    history_turn_ids: Optional[list[str]] = None
    history_turn_count: Optional[int] = None
    approx_input_tokens: Optional[int] = None
    dropped_turn_ids: Optional[list[str]] = None

    # Phase 3 inference telemetry (2026-08-12) - {"load": {...}, "generate":
    # {...}} per model_console/adapter.py's send_turn(), or None when the
    # resident loader doesn't populate BaseLoader.last_inference_telemetry
    # (only GemmaLoader does today). Assistant-only, same as the fields
    # above.
    telemetry: Optional[dict[str, Any]] = None


@dataclass
class ChatSession:
    model_name: str
    session_id: str = field(default_factory=_new_id)
    created_at: datetime = field(default_factory=_utcnow)
    profile_name: Optional[str] = None  # None = plain direct chat, no profile
    system_prompt: str = ""
    generation_overrides: dict[str, Any] = field(default_factory=dict)
    turns: list[ChatTurn] = field(default_factory=list)

    # Rolling summarization state (Phase 3, 2026-08-13 - see
    # model_console/conversation_manager.py). summary_text accumulates
    # every summarization pass so far (a summary of a summary, when a
    # long session gets summarized more than once - NOT re-derived from
    # scratch each time, since that would risk losing facts a prior pass
    # already distilled). summarized_up_to_turn_id is the turn_id of the
    # last raw turn folded into summary_text - every conversational turn
    # AFTER it (in turns list order) is still raw and eligible to be
    # summarized next time the budget is exceeded. Both empty/None means
    # "never summarized yet", the common case for most sessions.
    summary_text: str = ""
    summarized_up_to_turn_id: Optional[str] = None

    def add_turn(self, turn: ChatTurn) -> None:
        self.turns.append(turn)

    def conversational_turns(self) -> list[ChatTurn]:
        """
        Turns eligible to be fed back to the model as conversation
        history: real user/assistant turns only. Excludes assistant
        turns that errored (turn.error is not None) - a failed
        generation must not become context for the next turn. System/
        diagnostic transcript notes (see chat_tab.py's intro message)
        never reach session.turns in the first place - they're appended
        to the visible transcript widget directly, not added via
        add_turn() - so no separate filtering is needed for those here.
        """
        return [t for t in self.turns if t.error is None]

    def unsummarized_conversational_turns(self) -> list[ChatTurn]:
        """
        conversational_turns() filtered to just the ones NOT already
        folded into summary_text - i.e. what conversation_manager.py
        still has to either pass raw or summarize next. If nothing has
        been summarized yet (the common case), this is identical to
        conversational_turns().
        """
        return self.filter_unsummarized(self.conversational_turns())

    def filter_unsummarized(self, turns: list[ChatTurn]) -> list[ChatTurn]:
        """
        Same filtering as unsummarized_conversational_turns(), but
        against an EXTERNALLY-SNAPSHOTTED turn list rather than live
        session.turns. Needed because chat_tab.py snapshots history
        BEFORE appending the current user turn to session.turns (so the
        message being sent doesn't count as its own history) - by the
        time a background worker thread gets around to summarizing,
        session.turns already includes that new turn, so filtering
        against session.turns directly would wrongly fold the
        in-flight message into "already summarized" bookkeeping.
        """
        if self.summarized_up_to_turn_id is None:
            return turns
        for i, t in enumerate(turns):
            if t.turn_id == self.summarized_up_to_turn_id:
                return turns[i + 1:]
        # summarized_up_to_turn_id no longer present in this particular
        # list (e.g. it was summarized after this snapshot was taken) -
        # fail safe by treating everything in the snapshot as
        # unsummarized rather than silently dropping history.
        return turns
