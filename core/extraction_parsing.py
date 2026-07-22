"""
Shared parsing logic for models using the standard extraction prompt
contract (document_type:/visible_dates:/personal_names:/place_names:/
subject_keywords: key:value block format, each list field using
"value|confidence; value|confidence; ..." pipe-delimited entries).

Pulled out of Qwen3VLLoader into its own module rather than copy-pasted
per-loader, since this parsing contract is tied to the PROMPT format,
not to any specific model - every extraction-role loader should share
one implementation so a fix (like the duplicate-block degeneration
guard below) only needs to be made once.
"""

from __future__ import annotations

import re

from core.schema import ConfidenceLevel

CONFIDENCE_MAP = {
    "confirmed": ConfidenceLevel.CONFIRMED,
    "partial": ConfidenceLevel.PARTIAL,
    "unclear": ConfidenceLevel.UNCLEAR,
    "not_present": ConfidenceLevel.NOT_PRESENT,
}

REQUIRED_KEYS = {"document_type", "personal_names", "place_names"}

_KV_LINE = re.compile(r"^\s*([a-zA-Z_][a-zA-Z_ ]*?)\s*[:;]\s*(.*)$")
_KEYWORD_SPLIT = re.compile(r"[,;]")

# Narrow, evidence-based aliases for confirmed-stable model-specific field
# name corruptions - NOT a general fuzzy-matching mechanism. Each entry
# here should be backed by the same corruption being observed at least
# twice, independently, before being added - silently remapping arbitrary
# "creative" field names would mask real problems instead of catching a
# known, reproducible quirk. "invisible_dates" confirmed twice from
# Granite Vision 3.2-2b across two different prompt versions
# (2026-07-11) - not a formatting variant of "visible_dates", a
# different word entirely, but consistently substituted in its place.
_KEY_ALIASES = {
    "invisible_dates": "visible_dates",
}


_KNOWN_FIELDS = ["document_type", "visible_dates", "personal_names", "place_names", "subject_keywords"]
_INLINE_FIELD_SPLIT = re.compile(
    r"(?i)\b(" + "|".join(_KNOWN_FIELDS) + r")\s*[:;]\s*"
)


def parse_kv_block(raw_output: str) -> dict[str, str]:
    """
    Parses 'key: value' lines into a dict. Raises if any recognized
    field key appears more than once - a repeated document_type:/
    personal_names:/etc. block is a structural signal of generation
    degeneration (the model re-emitting the whole field set instead
    of stopping), independent of whether the repeat block's content
    happens to be empty, garbage, or plausible-looking. Silently
    keeping "the last occurrence" would let a degenerate repeat
    overwrite a correct first pass and still validate cleanly against
    the schema - this closes that gap. (Confirmed against a real
    Qwen3-VL failure case where a 3x-repeated block passed schema
    validation on the empty second pass while discarding correct data
    from the first pass - see project history 2026-07-10.)
    """
    result: dict[str, str] = {}
    seen_keys: dict[str, int] = {}
    for line in raw_output.splitlines():
        match = _KV_LINE.match(line)
        if not match:
            continue
        # Some models (confirmed: SmolVLM2) insert a stray space after the
        # underscore in multi-word field names - "personal_ names:" instead
        # of "personal_names:". Without normalizing this, the line simply
        # fails to match a known field and gets silently dropped entirely
        # (not partially parsed) - the field then reads as "missing" rather
        # than "malformed", which is a confusing error to debug from the
        # schema-validation side alone. Stripping internal whitespace from
        # the key (not the value) fixes this without weakening the
        # duplicate-block degeneration guard below, since that guard keys
        # off the normalized name either way.
        key = match.group(1).replace(" ", "").lower()
        key = _KEY_ALIASES.get(key, key)
        seen_keys[key] = seen_keys.get(key, 0) + 1
        result[key] = match.group(2).strip()

    duplicated = {k: n for k, n in seen_keys.items() if n > 1}
    if duplicated:
        raise ValueError(
            f"Output contains repeated field block(s): {duplicated} - "
            f"this indicates the model re-emitted the full field set "
            f"instead of stopping after one pass (generation "
            f"degeneration), not a single clean extraction. Raw output "
            f"head: {raw_output[:200]!r}."
        )

    # Fallback: some models (confirmed: InternVL3-2B, 2026-07-12) emit
    # the entire multi-field response on ONE line with no real line
    # breaks at all - semicolon-delimited throughout. Line-based parsing
    # above has no way to terminate a value without a newline, so it
    # swallows everything after the first field into that field's value,
    # and every other field silently never becomes its own key (reads as
    # "missing" downstream, not "malformed" - confusing to debug from
    # the schema-validation error alone). Only engages when the inline
    # scan finds MORE known fields than line-based parsing did, so a
    # properly newline-separated response (everything tested successfully
    # this session) is completely unaffected - this is strictly a
    # fallback for a parsing failure, not a change to normal behavior.
    known_found_by_lines = len(set(result.keys()) & set(_KNOWN_FIELDS))
    inline_result = _inline_field_split(raw_output)
    known_found_inline = len(set(inline_result.keys()) & set(_KNOWN_FIELDS))
    if known_found_inline > known_found_by_lines:
        return inline_result

    return result


def _inline_field_split(raw_output: str) -> dict[str, str]:
    """
    Finds known field names as inline anchors (not requiring line-start
    position), using their positions to slice the string into per-field
    value segments. See parse_kv_block's fallback comment above for when
    and why this is used.
    """
    matches = list(_INLINE_FIELD_SPLIT.finditer(raw_output))
    if len(matches) < 2:
        return {}
    result: dict[str, str] = {}
    seen_keys: dict[str, int] = {}
    for i, m in enumerate(matches):
        key = m.group(1).replace(" ", "").lower()
        key = _KEY_ALIASES.get(key, key)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_output)
        value = raw_output[start:end].strip().rstrip(";").strip()
        seen_keys[key] = seen_keys.get(key, 0) + 1
        result[key] = value
    duplicated = {k: n for k, n in seen_keys.items() if n > 1}
    if duplicated:
        raise ValueError(
            f"Output contains repeated field block(s) (inline-collapsed "
            f"format): {duplicated} - generation degeneration, not a "
            f"single clean extraction. Raw output head: {raw_output[:200]!r}."
        )
    return result


def parse_pipe_entries(value: str) -> list[tuple[str, ConfidenceLevel]]:
    """
    Parses 'value|confidence; value|confidence; ...' into tuples.
    Entries missing a confidence tag, or with an unrecognized
    confidence word, are dropped rather than guessed - a malformed
    entry should not silently become CONFIRMED (or any other level)
    by default, since that defeats the point of mandatory per-field
    confidence tagging.
    """
    entries: list[tuple[str, ConfidenceLevel]] = []
    if not value.strip():
        return entries
    for chunk in value.split(";"):
        chunk = chunk.strip()
        if not chunk or "|" not in chunk:
            continue
        name_part, _, conf_part = chunk.rpartition("|")
        name_part = name_part.strip()
        conf_key = conf_part.strip().lower()
        confidence = CONFIDENCE_MAP.get(conf_key)
        if not name_part or confidence is None:
            continue
        entries.append((name_part, confidence))
    return entries


def parse_keyword_list(value: str, max_len: int = 60) -> list[str]:
    """
    Splits subject_keywords on EITHER comma or semicolon - not just one.
    The original convention (extract_census_v1/v2/v3.txt) used commas;
    later prompts (extract_printed_v2.txt and others, both Jon's edits
    and Claude's proposed versions independently) drifted to semicolons,
    matching the delimiter every other list field already uses. Rather
    than pick one and silently break whichever prompts use the other -
    exactly the kind of silent, undetected corruption this project has
    hit before (see project history 2026-07-11, Granite Vision) - both
    are accepted here.
    """
    if not value.strip():
        return []
    return [k.strip()[:max_len] for k in _KEYWORD_SPLIT.split(value) if k.strip()]


def parse_json_schema_output(raw_output: str) -> dict:
    """
    Parses raw JSON output from models with a native structured-extraction
    mode (confirmed: GLM-OCR's "Information Extraction" prompt scenario),
    as opposed to the pipe-delimited key:value text convention every other
    loader in this project uses. Distinct parser because the output shape
    is genuinely different, not a formatting variant of the same contract.

    Strips markdown code fences (```json ... ```) if present - common LLM
    JSON-output behavior even when a prompt asks for raw JSON only, worth
    handling defensively rather than letting json.loads() fail on a fence
    that wasn't actually requested.

    Raises json.JSONDecodeError (uncaught, propagates to caller) if the
    output isn't valid JSON at all - that's a real, informative failure
    signal (the model didn't follow the schema format), not something to
    paper over with a fallback.
    """
    import json
    import re as _re

    text = raw_output.strip()
    # Not end-anchored - some models (confirmed: GLM-OCR) can emit a
    # stray token after the closing fence (e.g. a next-turn role marker
    # like "<|user|>" leaking in), which would prevent an end-anchored
    # pattern from matching at all and silently fall through to trying
    # to parse the whole raw string, prefix and all.
    fence_match = _re.match(r"^```(?:json)?\s*(.*?)\s*```", text, _re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    return json.loads(text)
