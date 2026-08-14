"""
Deterministic, code-level pre-checks for the planner (2026-08-13,
Jon's hybrid design) - things Python already knows for certain and
should never re-ask a model to (re)decide. Pure functions, no model
calls, no side effects.

Kept deliberately narrow: only patterns confirmed unambiguous enough to
act on WITHOUT a model in the loop at all. A broader/fuzzier ambiguity
detector was considered and rejected for now - false positives here
would short-circuit a call the CPU planner might have gotten right, and
there is no live-traffic evidence yet for tuning a wider pattern. This
module grows only from confirmed real cases, same discipline as
core/extraction_parsing.py's _KEY_ALIASES.
"""

from __future__ import annotations

import re

# Deliberately narrower than a bare "look up" / "search" - those are
# also legitimate ways to phrase a genealogy_framework_lookup request
# ("look up whether we've seen this name before"), so matching on them
# alone would misroute a local-lookup request as a forced web search.
# Requires an explicit online/web/internet qualifier or "google" itself.
_EXPLICIT_WEB_SEARCH_PATTERN = re.compile(
    r"\b(search\s+(the\s+)?(web|online|internet)|"
    r"look\s+(it|this|that)?\s*up\s+online|"
    r"google\s+(it|this|that)?\b|"
    r"check\s+online|"
    r"search\s+.{0,40}\bonline\b)",
    re.IGNORECASE,
)


def detect_explicit_web_search_request(user_text: str) -> bool:
    """
    True only when the user's own wording explicitly names the web/
    online/internet as the place to look - not just "look up" or
    "search" alone (see module docstring). Used by
    core/agent_tools/planner.py to skip the CPU planner call entirely
    on the first step of a turn and go straight to web_search - a
    genuine "we don't need a model for this" case, and a nice side
    effect: one fewer CPU planner round-trip for the clearest case.
    """
    return bool(_EXPLICIT_WEB_SEARCH_PATTERN.search(user_text))


# Deliberately narrow word list: only nouns that are STRUCTURALLY
# periodic/recurring by their nature (a census, an election, the
# Olympics - there have been many), never a nameable one-off event
# ("the Treaty of Versailles", "World War 1") that only sounds
# historical-and-vague. Confirmed 2026-08-13: even Qwen2.5-1.5B-
# Instruct, correctly discriminating tools everywhere else, missed
# this exact pattern once the full 5-tool production catalog (with its
# longer, more emphatic tool descriptions) replaced a smaller test
# catalog - a real, reproducible reliability gap, not a fluke - so this
# one narrow case gets a deterministic gate rather than depending on
# further prompt tuning to hold.
_RECURRING_EVENT_PATTERN = re.compile(
    r"\b(what|which)\s+year\b.{0,40}\b(census|election|olympics|world\s*cup)\b"
    r"|\bwhen\b.{0,40}\b(census|election|olympics|world\s*cup)\b.{0,20}\btaken|held|happen",
    re.IGNORECASE,
)
_YEAR_PATTERN = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")


def extract_years(text: str) -> set[str]:
    """
    All 4-digit year-shaped substrings (1500-2099) in `text`. Shared by
    core/agent_tools/web_research_agent.py (research-bundle provenance
    check) and core/agent_tools/planner.py (final-answer provenance
    check) - both confirmed live 2026-08-13 to need the identical
    "is this year actually traceable to the evidence" check, just
    applied at different points in the pipeline.
    """
    return set(_YEAR_PATTERN.findall(text))


# A qualifier naming WHICH instance is meant resolves the ambiguity -
# "the last election" or "the most recent census" isn't ambiguous even
# without a literal year.
_INSTANCE_QUALIFIER_PATTERN = re.compile(
    r"\b(first|last|next|latest|most\s+recent|earliest|current|upcoming|\d+(?:st|nd|rd|th))\b",
    re.IGNORECASE,
)


_LOOKUP_SCAFFOLD_PATTERN = re.compile(
    r"\b(have\s+we\s+seen|has\s+|come\s+up\s+before|is\s+|already\s+recorded|"
    r"did\s+we\s+(record|find)|do\s+we\s+have\s+(a\s+)?record\s+of|"
    r"before(\s+in\s+this\s+project)?|the\s+name|the\s+person|the\s+place|"
    r"a\s+(place|person)\s+called|someone\s+named|somewhere\s+called|"
    r"in\s+this\s+project)\b",
    re.IGNORECASE,
)


def derive_lookup_query(user_text: str) -> str:
    """
    Best-effort fallback query for genealogy_framework_lookup when the
    planner picked the right tool but left 'query' unfilled - confirmed
    live 2026-08-13: the CPU planner correctly chose
    genealogy_framework_lookup for "have we seen the name Alistair
    MacKenzie before in this project" but returned args={}, which fails
    schema validation (query is required) and forces an otherwise-
    avoidable answer_directly fallback via the circuit breaker.

    genealogy_framework_lookup does a plain SQL substring match
    (core/genealogy_memory.py::search(), 'value LIKE %query%'), not
    fuzzy matching - so defaulting to the full raw sentence as query
    would silently search for an absurdly long substring and just
    return zero matches, which LOOKS like "checked, not found" but is
    actually "query was malformed" - a silent-wrong-answer risk this
    project's discipline (core/genealogy_memory.py's own docstring)
    explicitly avoids elsewhere. Stripping common question-scaffolding
    phrases and keeping the remainder is a real improvement over that,
    not a guess - falls back to the full text only if stripping leaves
    nothing (never worse than the naive default).
    """
    stripped = _LOOKUP_SCAFFOLD_PATTERN.sub(" ", user_text)
    stripped = re.sub(r"\s+", " ", stripped).strip(" ?.,!")
    return stripped or user_text


def detect_unspecified_recurring_event(user_text: str) -> bool:
    """
    True when the question asks for THE year/date of something that
    happens repeatedly (a census, an election, the Olympics) without
    naming a specific year or an instance-resolving qualifier ("the
    last one", "the 1911 one"). This is a code-level gate, not a model
    judgment call - see the confirmed real example in the module
    docstring above. Used by core/agent_tools/planner.py to route
    straight to ask_user_clarification, bypassing the CPU planner for
    this one narrow, high-precision pattern.
    """
    if not _RECURRING_EVENT_PATTERN.search(user_text):
        return False
    if _YEAR_PATTERN.search(user_text):
        return False
    if _INSTANCE_QUALIFIER_PATTERN.search(user_text):
        return False
    return True
