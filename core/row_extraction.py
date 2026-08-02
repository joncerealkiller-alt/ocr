"""
Stage 6 (Specialized Extraction) per docs/PIPELINE_STAGE_TERMINOLOGY.md.
Row-level extraction stage - consumes a segmentation sidecar JSON (see
core/row_segmentation.py) and runs structured extraction on each row,
cropped from the ORIGINAL source image in memory per row, not from
pre-saved crop files.

Built 2026-07-13, the step the whole row-segmentation build (deskew,
periodic anchoring, the visual adjustment UI, three real-page
validations) was working toward: isolating single census rows before
extraction was specifically motivated by olmOCR's whole-page failure
(fabricating an entire household, silently, on a real page - see
project history) - a single row has none of the repeated-similar-
content structure that drove every major degeneration/fabrication
failure this session.

GENUINELY DIFFERENT SCHEMA from core/schema.py's ExtractionResult: that
schema was built for whole-DOCUMENT extraction (multiple names/places/
dates per document, a fixed 5-field structure). A single census row is
one person with N COLUMNS (name, age, relationship, birthplace,
occupation...) matching whatever this specific form's header row says -
forcing that through the document-level schema would be a poor fit.
This module defines its own row-level result type instead.

Column names are supplied EXPLICITLY (not auto-OCR'd from the header
crop) - same reasoning that already justified treating row_count=50 as
known-in-advance for a given census form type: column layout is
genuinely fixed per census year/form, and auto-reading tiny, sometimes
bilingual, sometimes densely-packed header text is exactly the kind of
detection this project's own evidence says needs human confirmation,
not blind trust. Supply the list once per form type, reuse across every
page of that type.
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import BaseModel

from core.row_segmentation import (
    load_sidecar, crop_region_from_source, compute_exclude_ranges, update_sidecar,
)
from core.loader_registry import LOADER_REGISTRY
from core.loaders.base_loader import load_model_config
from core.schema import ConfidenceLevel
from core.debug_dump import DebugModelInputRecorder, NOOP_RECORDER

# Confirmed real bug (2026-07-16): Windows' default console encoding
# (cp1252, a legacy Western-European codepage) cannot represent
# arbitrary Unicode. Models occasionally hallucinate non-Latin
# characters into raw output (confirmed tonight - an actual CJK glyph
# appeared in raw Age output on one row) - printing that raw text
# crashed an entire 50-row run with UnicodeEncodeError, taking down
# work that had already completed successfully rather than just
# skipping the one unprintable line. Reconfiguring stdout to replace
# unprintable characters instead of raising fixes this for every print
# in this module, not just one call site - errors='replace' means a
# genuinely un-encodable character becomes "?" in the console (the
# SAVED JSON/CSV are unaffected either way, since those are written
# with explicit UTF-8 encoding, not through stdout).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass  # stdout may not support reconfigure (e.g. redirected/piped in
          # some contexts) - never let this safety fix itself crash.

CONFIDENCE_MAP = {
    "confirmed": ConfidenceLevel.CONFIRMED,
    "partial": ConfidenceLevel.PARTIAL,
    "unclear": ConfidenceLevel.UNCLEAR,
}


def _normalize_confidence_word(raw: str) -> ConfidenceLevel | None:
    """
    Tolerant lookup for the confidence word a model wrote, not just an
    exact CONFIDENCE_MAP hit - real bug found 2026-07-24: qwen3vl4b's
    stage-2 output on a real run used "uncLEAR" (mixed case - the naive
    .lower() lookup DOES catch this one), "uncleared" and "uncleard"
    (an extra/wrong trailing letter), and "unc lear"/"unc lea r" (stray
    internal spaces) for what was clearly meant to be "unclear" every
    time. CONFIDENCE_MAP.get() with only .lower() applied matched NONE
    of the space-containing or suffix variants, so parse_row_output
    silently DROPPED those fields entirely (not even "?"/unclear - just
    missing, showing up as "Missing/dropped fields" in schema_error) -
    a parser gap masquerading as a model-quality problem.

    Same two-step tolerance _normalize_key already uses for column
    names (strip ALL whitespace, not just leading/trailing, then
    lowercase), plus a prefix match against each canonical word as a
    second pass: "uncleared"/"uncleard" both start with "unclear" once
    whitespace-stripped, so they resolve correctly. Deliberately does
    NOT fuzzy-match unrelated words (e.g. "uncertain" does not start
    with "unclear" and stays unmatched, returning None here) - this is
    tolerance for near-miss SPELLING/SPACING of the three DOCUMENTED
    confidence words, not a guess at what different words might mean.
    """
    cleaned = "".join(raw.split()).lower()
    if cleaned in CONFIDENCE_MAP:
        return CONFIDENCE_MAP[cleaned]
    for word, level in CONFIDENCE_MAP.items():
        if cleaned.startswith(word):
            return level
    return None


class RowFieldValue(BaseModel):
    """One column's value for one row - same confidence-tagging
    discipline as the document-level schema (proven valuable all
    session for telling genuine reads apart from fabrication), applied
    per-column instead of per-entry."""
    value: str
    confidence: ConfidenceLevel


class RowExtractionResult(BaseModel):
    """
    Result for ONE row. Deliberately NOT core.schema.ExtractionResult -
    see module docstring for why a document-level multi-entry schema is
    a poor fit for a single row's fixed-column structure.
    """
    row_index: int
    bbox: list[int]
    fields: dict[str, RowFieldValue]
    raw_output: str
    model: str
    runtime_seconds: float
    schema_pass: bool
    schema_error: str | None = None
    stage1_raw_output: str | None = None


def build_row_prompt(column_names: list[str]) -> str:
    """
    Builds an extraction prompt targeting this form's ACTUAL columns
    (supplied explicitly, not guessed) - one line per column, pipe-
    delimited confidence tag, same convention as every document-level
    extraction prompt this session, applied to whatever columns this
    specific form really has instead of a fixed 5-field schema.

    Uses a CONCRETE WORKED EXAMPLE, not an abstract bracket-placeholder
    template - confirmed necessary through TWO separate real failures
    (2026-07-13): the original "<value>|<confidence>" angle-bracket
    template got echoed back literally on some rows ("Name: <NAME>|
    confirmed"). Switching to "[the value written]|[confidence word]"
    square-bracket wording was meant to fix that, but a later real row
    showed the SAME underlying behavior in a different shape - the
    model wrote "[Ricardo Burton] [confirmed]" (brackets AND a space,
    no pipe at all), which fails parsing just as completely (zero pipe
    characters -> every field silently dropped). The common thread
    across both failures isn't which bracket character was used - it's
    that ANY abstract placeholder template invites this model to echo
    structure back literally rather than substitute real content. A
    concrete, fully-filled-in example (fictional but complete, no
    placeholder syntax anywhere) sidesteps that failure mode entirely,
    since there's no ambiguous template for the model to reproduce -
    it infers the pattern from a genuine worked instance instead.
    """
    columns_str = ", ".join(column_names)
    example_lines = "\n".join(
        f"{name}: {_EXAMPLE_VALUES.get(name, 'Zzyx')}|confirmed"
        for name in column_names
    )
    return f"""This is ONE ROW from a census/tabular record. Copy exactly what is written in this row.

The columns for this row, in order, are: {columns_str}

For each column, write the column name, a colon, the value you actually read, a pipe character, then a confidence word. Here is a complete worked example using invented data (not from your image - just showing the format):

{example_lines}

Now do the same thing for the REAL columns of the row in the image ({columns_str}), using the values actually visible there - not the example values above.

Confidence must be exactly one of: confirmed, partial, unclear
- confirmed: every character is clearly readable with no doubt
- partial: mostly readable, some uncertainty
- unclear: not legible enough to read with confidence

If a column is genuinely blank for this row, write the column name and colon followed immediately by the pipe and confirmed, with nothing in between - for example: Occupation:|confirmed

Do not invent a value that is not visible. Do not add commentary, brackets, or any punctuation not shown in the example. Output ONLY {len(column_names)} lines, one per column, then stop."""


_EXAMPLE_VALUES = {
    # REPLACED 2026-07-22 (real, conclusive evidence): every non-empty
    # row across a genuine 6-row masked test (glm_ocr+smolvlm2_2b) came
    # back a 100% byte-exact match to this dict's OLD values (John
    # Smith/45/M/Boarder/Ontario), across 5 DIFFERENT real people on
    # the actual census page. Not "the model guessed a plausible
    # record" - it was echoing the worked example verbatim whenever its
    # real reading confidence was low, exactly the same failure
    # mechanism as the earlier "Head" self-reference bug below, just
    # not fully eliminated by that fix - swapping which specific wrong
    # value leaks isn't the same as stopping the leak.
    #
    # These are now DELIBERATELY ABSURD/IMPOSSIBLE - still a complete,
    # concrete, fully-filled-in example (preserving the original fix
    # that stopped BRACKET/PLACEHOLDER echoing - see build_row_prompt's
    # docstring), but an echoed value can no longer pass as real data
    # to a scorer OR a human reviewer. "Boarder"/"Ontario" were real,
    # plausible values, so a leak was invisible unless you happened to
    # already know the true answer; "Interstellar Cousin"/"Atlantis"
    # cannot be mistaken for a genuine reading under any circumstance.
    "Name": "Zzyx Q. Bleeth", "Age": "999", "Sex": "X",
    "Relationship to Head": "Interstellar Cousin", "Birthplace": "Atlantis",
    "Occupation": "Dragon Tamer", "Province": "Narnia",
}


_FIELD_TYPE_HINTS: dict[str, dict[str, Any]] = {
    # Semantic context for build_structuring_prompt()'s {field_hint}
    # placeholder (2026-07-25, per Jon's direction: give the model a
    # sense of what KIND of value a field holds, without forcing it to
    # a fixed vocabulary). REWORKED same day, second pass: the first
    # version of this dict gave concrete candidate words per field
    # (e.g. "Head, Wife, Son, Daughter, Boarder" for Relationship to
    # Head) - Jon's call: that still risks seeding an answer, since a
    # concrete word sitting right there in the prompt is a much easier
    # thing for the model to reach for under low confidence than
    # actually reading faint handwriting, no matter how the surrounding
    # wording disclaims it. This version describes the CATEGORY and
    # SHAPE of an acceptable answer (expected_content, typical_format)
    # instead of ever naming a candidate value - "a family or household
    # relationship, one or two words" tells the model what kind of
    # thing to look for without handing it anything it could copy
    # verbatim. No leakage-risk tradeoff to track here the way the
    # examples-based version had.
    "Name": {
        "expected_content": "A personal name - given name and/or surname - as written.",
        "typical_format": "One or more words.",
        "notes": [
            "May appear as a ditto mark or \"Do.\" meaning the same "
            "surname as the row above - transcribe those literally, "
            "do not expand them into a name.",
        ],
    },
    "Age": {
        "expected_content": "A numeric age.",
        "typical_format": "A number.",
        "notes": ["Infants may be recorded as fractions (e.g. \"2/12\" for two months old)."],
    },
    "Sex": {
        "expected_content": "A sex/gender abbreviation as recorded on the form.",
        "typical_format": "A single character.",
        "notes": [],
    },
    "Relationship to Head": {
        "expected_content": "A family or household relationship.",
        "typical_format": "One or two words.",
        "notes": [
            "Do not infer the relationship from age, sex, or name. Only "
            "transcribe what is visibly written. Return ? if unreadable.",
        ],
    },
    "Birthplace": {
        "expected_content": "A place name.",
        "typical_format": "Town, county, province, state, or country.",
        "notes": ["Preserve abbreviations exactly."],
    },
    "Occupation": {
        "expected_content": "An occupation or trade.",
        "typical_format": "One or a few words.",
        "notes": [],
    },
    "Province": {
        "expected_content": "A province or country name.",
        "typical_format": "One or a few words.",
        "notes": ["Preserve abbreviations exactly."],
    },
}


def _format_field_hint(column_name: str) -> str:
    """
    Builds the optional semantic-context block for one field, or ""
    when column_name has no entry in _FIELD_TYPE_HINTS (e.g. a form's
    own column not yet covered here) - build_structuring_prompt()
    degrades gracefully, this is never a hard requirement.

    Deliberately never names a candidate VALUE (see _FIELD_TYPE_HINTS'
    docstring comment) - only the category (expected_content), the
    structural shape (typical_format), and any field-specific
    transcription reminders (notes).
    """
    hint = _FIELD_TYPE_HINTS.get(column_name)
    if not hint:
        return ""
    lines = [f"Expected content: {hint['expected_content']}"]
    if hint.get("typical_format"):
        lines.append(f"Typical format: {hint['typical_format']}")
    lines.extend(hint.get("notes", []))
    return "\n".join(lines)


_ROW_FIELD_LINE = re.compile(r"^\s*(.+?)\s*:\s*(.*)$")


def _normalize_key(s: str) -> str:
    """Lowercase + strip ALL whitespace (not just leading/trailing) for
    column-name matching - confirmed necessary (2026-07-13, real row
    result) after a model wrote "RelationshiptoHead" (spaces dropped)
    for "Relationship to Head", which .strip().lower() alone doesn't
    catch (that only trims the ends, not internal spaces) - the whole
    line was silently rejected at the key-matching step even though the
    value itself was a real, usable answer."""
    return "".join(s.split()).lower()


def parse_row_output(raw_output: str, column_names: list[str]) -> dict[str, RowFieldValue]:
    """
    Parses the flat per-column output. Matches lines against the KNOWN
    column list (case- and whitespace-tolerant - see _normalize_key)
    rather than accepting any "key: value" line blindly - a garbled/
    misnamed line from the model shouldn't silently become a new,
    unexpected field. Missing confidence tag or unrecognized confidence
    word -> that column is dropped (not guessed), same rule as every
    other parser this session.
    """
    normalized_columns = {_normalize_key(c): c for c in column_names}
    result: dict[str, RowFieldValue] = {}

    for line in raw_output.splitlines():
        match = _ROW_FIELD_LINE.match(line)
        if not match:
            continue
        key_raw, value_raw = _normalize_key(match.group(1)), match.group(2).strip()
        if key_raw not in normalized_columns:
            continue
        real_column = normalized_columns[key_raw]

        if "|" not in value_raw:
            # Tolerate "ColumnName::confidence" (2026-07-20, real case:
            # qwen25_vl_7b's output on a simulated-wrong-input test) -
            # looks like a near-miss attempt at the documented blank-
            # value convention ("Occupation:|confirmed") with a stray
            # colon instead of the pipe. GENERALIZED 2026-07-21 after a
            # second, different variant showed up (granite_vision_2b:
            # "Relationship to Head:, confirmed" - a comma instead of
            # either a colon or pipe) - rather than adding a third
            # narrow special case, strip ANY leading run of punctuation
            # (colon, comma, space, combinations) and check if what's
            # left is EXACTLY a recognized confidence word. Still only
            # matches when the remainder is nothing but a bare
            # confidence word - a genuine value that happens to start
            # with punctuation, or contains a confidence word as part
            # of real content, won't match this.
            bare_conf = value_raw.lstrip(" :,;.-").strip()
            bare_conf_level = _normalize_confidence_word(bare_conf)
            if bare_conf_level is not None:
                result[real_column] = RowFieldValue(
                    value="", confidence=bare_conf_level)
                continue

            # Second confirmed near-miss shape (2026-07-21, granite_
            # vision_2b, seen twice on real Daughter-input tests):
            # punctuation then the REAL VALUE itself, with no
            # confidence tag at all - e.g. "Relationship to Head:,
            # Daughter". No stated confidence to read, so this
            # defaults to unclear (never invents a confidence level
            # the model didn't actually give). Length-guarded (<=5
            # words) so this doesn't swallow rambling non-answer text
            # as if it were a genuine short value - real values for
            # this kind of field are supposed to be short per the
            # prompt's own instructions.
            bare_value = value_raw.lstrip(" :,;.-").strip()
            if bare_value and len(bare_value.split()) <= 5:
                result[real_column] = RowFieldValue(
                    value=bare_value, confidence=CONFIDENCE_MAP["unclear"])
            continue
        value_part, _, conf_part = value_raw.rpartition("|")
        confidence = _normalize_confidence_word(conf_part.strip())
        if confidence is None:
            continue
        result[real_column] = RowFieldValue(value=value_part.strip(), confidence=confidence)

    return _scrub_example_leakage(result)


def _scrub_example_leakage(fields: dict[str, RowFieldValue]) -> dict[str, RowFieldValue]:
    """
    Deterministic safety net (2026-07-22) - added after real evidence
    showed prompt wording ALONE isn't reliable enough to stop this:
    a genuine 6-row test with the (already-fixed, deliberately absurd)
    worked example still came back with EVERY non-empty row a 100%
    byte-exact match to _EXAMPLE_VALUES, across different real rows.
    Two separate prompt-content changes (plausible values, then absurd
    values) both failed to suppress the underlying behavior - the
    model copies the worked example verbatim under low confidence
    regardless of what that example's content actually IS. That's a
    model-level tendency prompt wording alone hasn't been able to
    override, so this catches it in code instead: if a field's value
    is an EXACT match to what that column's worked-example value would
    have been, it's forced back to "?"/unclear rather than trusted -
    100% reliable regardless of model, prompt, or how convincing future
    leaked content might look, unlike relying on the model to follow an
    instruction.

    This does NOT fix why the leakage happens (still worth continued
    prompt iteration - see build_structuring_prompt's template_override)
    - it's a backstop that guarantees a leaked value can never silently
    reach saved output as if it were a real reading, layered on top of
    whatever prompt-level mitigation is in place, not a replacement for
    it.

    Deliberately exact-match only (not fuzzy/substring) - a real value
    that happens to legitimately match by coincidence (someone actually
    named "Zzyx Q. Bleeth" is not a real risk given these are
    deliberately absurd/fictional; a real Age of "999" or Birthplace of
    "Atlantis" is not realistically possible on a real census record) -
    so this has effectively zero false-positive risk against genuine
    data.
    """
    for column, field_value in list(fields.items()):
        expected_leak_value = _EXAMPLE_VALUES.get(column, "Zzyx")
        if field_value.value == expected_leak_value:
            print(f"WARNING: {column!r} value exactly matched the worked-example "
                  f"placeholder ({expected_leak_value!r}) - this is prompt-example "
                  f"leakage, not a real reading. Forcing to unclear.")
            fields[column] = RowFieldValue(value="?", confidence=ConfidenceLevel.UNCLEAR)
    return fields


def _compute_scoped_masks(sidecar: dict) -> tuple[list, list]:
    """
    Computes the actual paint-white exclude ranges separately for row
    crops and the header crop.

    2026-07-22: prefers sidecar["columns"]["__multi__"] - a dedicated
    pseudo-column reserved specifically for the two-stage/legacy
    multi-column extraction mask, added after discovering the per-
    column sidecar redesign had silently broken masking for this path
    entirely. row_segmentation_ui.py's _write_sidecar_state() stopped
    writing to the top-level mask_keep_ranges field once per-column
    masks existed (columns[name]["mask_keep_ranges"]) - but this
    function still only read that now-permanently-empty top-level
    field, so run_two_stage_extraction/run_row_extraction's legacy
    multi-column mode/extract_page_header were ALWAYS running fully
    unmasked with no way for the operator to change that, a real
    regression Jon confirmed mattered (both for output quality AND
    compute time - the whole point of masking is not sending the model
    pixels it doesn't need to look at).

    __multi__ is NOT part of column_order (it isn't a real extraction
    target, just a mask definition scratchpad) and is never touched by
    the single-column auto-advance/Extract flow - see
    row_segmentation_ui.py's set_multi_mask_mode()/_write_sidecar_state()
    for how it's populated.

    Falls back to the legacy top-level fields when no __multi__ entry
    exists (a sidecar that predates this, or genuinely has no mask
    defined for either path) - same convention as every other per-
    column-with-legacy-fallback fix made this session.

    Returns (row_mask_ranges, header_mask_ranges) - either may be []
    if no keep_ranges are set, or if that particular scope's checkbox
    was off when the mask was saved.
    """
    multi_state = sidecar.get("columns", {}).get("__multi__")
    if multi_state is not None:
        keep_ranges = [tuple(k) for k in multi_state.get("mask_keep_ranges", [])]
        apply_rows = multi_state.get("mask_apply_rows", False)
        apply_header = multi_state.get("mask_apply_header", False)
    else:
        keep_ranges = [tuple(k) for k in sidecar.get("mask_keep_ranges", [])]
        apply_rows = sidecar.get("mask_apply_rows", False)
        apply_header = sidecar.get("mask_apply_header", False)

    width = sidecar["deskewed_image_size"][0]
    row_masks = (
        compute_exclude_ranges(keep_ranges, width) if keep_ranges and apply_rows else []
    )
    header_masks = (
        compute_exclude_ranges(keep_ranges, width) if keep_ranges and apply_header else []
    )
    return row_masks, header_masks


def _extract_region(
    loader, source_path: str, deskew_angle: float, bbox: list[int],
    field_names: list[str], row_index: int, model_profile_name: str,
    mask_ranges: list[tuple[int, int]] | None = None,
    stage1_raw_output: str | None = None,
    tight_crop_keep_ranges: list[tuple[int, int]] | None = None,
    tight_crop_padding_px: int = 20,
    tight_crop_padding_pct: float | None = None,
    upscale_target_height: int | None = None,
    upscale_max_width: int = 4096,
    debug_recorder: DebugModelInputRecorder | None = None,
    debug_item_id: str | None = None,
) -> RowExtractionResult:
    """
    Shared extraction logic for ONE region (a person row OR the page
    header block) against an ALREADY-LOADED loader - no model init/
    release here, that's the caller's responsibility. Both
    run_row_extraction()'s per-row loop and extract_page_header() call
    this, so header+rows extracted together share one model load rather
    than paying the load/unload cost twice.

    mask_ranges (2026-07-15): column-mask ranges from the sidecar,
    passed straight through to crop_region_from_source() - see that
    function's docstring. Painting unwanted columns white rather than
    just relying on a narrower crop, since real testing showed Age
    contamination pulling from DIFFERENT nearby columns on different
    rows (dwelling numbers on one row, section/township/range on
    another) - a spatial-counting problem a single left/right crop
    boundary can't isolate when the wanted column sits between two
    different unwanted ones, not at either edge.

    tight_crop_keep_ranges (2026-07-22): defaults to None, meaning NO
    behavior change for existing callers (run_row_extraction's
    multi-column pass, extract_page_header) - neither passes this, so
    both keep sending the model the FULL row width with unwanted
    columns painted white, unchanged. Only run_single_column_extraction
    passes this, since tightening to one column's kept range is only
    correct when exactly one column is being isolated - the legacy
    multi-column path can have SEVERAL kept ranges spread across one
    row, and tightening to their combined span would still be mostly
    blank, or could cut off ranges depending on padding.

    stage1_raw_output (2026-07-16, real gap found by Jon): the two-
    stage pipeline previously only ever printed a CHARACTER COUNT for
    stage 1's reading ("Stage 1 (OCR) row 1: 99 chars"), never the
    actual text, and never saved it anywhere - made it impossible to
    tell whether a lost value (e.g. an "m" for months, a "?" for an
    illegible digit) was dropped by stage 1 itself or by stage 2's
    structuring pass. Passed through here so it can be attached to the
    saved result for exactly this kind of diagnosis.

    debug_recorder / debug_item_id (2026-07-24, --debug-model-inputs):
    see core/debug_dump.py. debug_recorder defaults to None, which
    core.debug_dump.DebugModelInputRecorder(enabled=False) also
    satisfies (its .new_item() returns a no-op recorder) - either way,
    every debug_* call below becomes a no-op and nothing is written to
    disk unless a caller explicitly passes an ENABLED recorder.
    debug_item_id defaults to a "row_NNNN" / "header" name derived from
    row_index when not given explicitly.
    """
    start = time.time()
    prompt = build_row_prompt(field_names)

    item_id = debug_item_id or (f"row_{row_index:04d}" if row_index else "header")
    debug_item = (debug_recorder or NOOP_RECORDER).new_item(item_id)

    region_image = crop_region_from_source(
        source_path, bbox, deskew_angle, mask_ranges,
        tight_crop_keep_ranges=tight_crop_keep_ranges,
        tight_crop_padding_px=tight_crop_padding_px,
        tight_crop_padding_pct=tight_crop_padding_pct,
        upscale_target_height=upscale_target_height,
        upscale_max_width=upscale_max_width,
        debug_stage_callback=debug_item.stage_callback(),
    )
    if region_image.mode != "RGB":
        region_image = region_image.convert("RGB")

    debug_item.set_prompt(prompt)
    debug_item.set_meta(
        source_image_path=source_path,
        row_index=row_index,
        bbox=bbox,
        model=model_profile_name,
        reasoning_enabled=getattr(loader.config, "reasoning_enabled", None),
        generation_config_hash=(
            loader.config.content_hash() if hasattr(loader.config, "content_hash") else None
        ),
        preprocessing={
            "tight_crop_keep_ranges": tight_crop_keep_ranges,
            "tight_crop_padding_px": tight_crop_padding_px,
            "tight_crop_padding_pct": tight_crop_padding_pct,
            "upscale_target_height": upscale_target_height,
            "upscale_max_width": upscale_max_width,
        },
    )

    try:
        raw_output = loader._run_generate(region_image, prompt)
        fields = parse_row_output(raw_output, field_names)
        missing = set(field_names) - set(fields.keys())
        schema_pass = len(missing) == 0
        schema_error = f"Missing/dropped fields: {missing}" if missing else None
        debug_item.finalize(raw_output=raw_output)
    except Exception as e:
        raw_output = f"[ERROR: {e}]"
        fields = {}
        schema_pass = False
        schema_error = str(e)
        debug_item.finalize(raw_output=None, error=e)

    return RowExtractionResult(
        row_index=row_index, bbox=bbox, fields=fields, raw_output=raw_output,
        model=model_profile_name, runtime_seconds=time.time() - start,
        schema_pass=schema_pass, schema_error=schema_error,
        stage1_raw_output=stage1_raw_output,
    )


def _release_model(loader) -> None:
    """
    Same VRAM-release discipline as model_assessment.py's
    _release_model - see that file's history for why this matters
    (torch.cuda.empty_cache() only firing on an OOM path silently let
    VRAM climb across a whole session). Restored 2026-07-13 - this
    function was called from run_row_extraction()/extract_page_header()
    but its actual definition was accidentally dropped during an
    earlier refactor (the _extract_region split), leaving only
    docstring/comment references to it - a real bug, caught by an
    actual live run raising NameError, not by any test in this sandbox
    (no torch available here to exercise this code path directly).

    UPDATED 2026-07-22 to call loader.release() first - found via
    MoondreamLoader (its worker runs as a genuinely separate OS
    subprocess with its OWN CUDA context in a different venv/process
    entirely). This function's own `del loader` only removes ITS
    internal local binding, not the CALLER's - the caller's own
    `loader` variable keeps the object alive until the caller's frame
    itself exits, so relying on MoondreamLoader.__del__ to terminate
    the subprocess meant it could keep running (and holding VRAM in
    its own process) well past the point run_two_stage_extraction()
    intended it to be freed before loading the SECOND model - directly
    undermining the "avoid two models resident in VRAM simultaneously"
    guarantee that function's docstring promises. release() is a no-op
    for every ordinary in-process loader (default on BaseLoader), so
    this change is a no-op for them too - only a loader that overrides
    release() (currently just MoondreamLoader) behaves any differently.
    """
    import gc
    import torch
    try:
        loader.release()
    except Exception:
        pass
    try:
        loader.model = None
        loader.processor = None
        loader.tokenizer = None
    except Exception:
        pass
    del loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


_DEFAULT_STRUCTURING_TEMPLATE = """Below is a raw OCR reading of ONE FIELD - a single column, cropped from one row of a census/tabular record - produced by a different tool. The image shown to you is that SAME field crop, not the whole row and not any other column.

Raw OCR reading of this field:
\"\"\"
{raw_ocr_text}
\"\"\"

Your job: report this ONE field's real value, using the actual image as the authoritative source - check or correct the raw OCR reading where it looks wrong, incomplete, or unrelated to what the image shows. Do not just copy the raw reading blindly if the image shows something different, and do not treat a garbled, rambling, or off-topic raw reading as if it describes this field at all - that kind of text is noise, not evidence.

The field is: {columns_str}
{field_hint}

Write the field name, a colon, the value, a pipe character, then a confidence word. Here is a complete worked example using invented data (not from your image - just showing the format):

{example_lines}

Now do the same thing for the REAL field shown in the image ({columns_str}), using the value actually visible there - not the example value.

Confidence must be exactly one of: confirmed, partial, unclear
- confirmed: every character is clearly readable with no doubt
- partial: mostly readable, some uncertainty
- unclear: not legible enough to read with confidence

If the field is genuinely blank on the form, write the field name and colon followed immediately by the pipe and confirmed, with nothing in between - for example: Occupation:|confirmed

If the raw OCR reading above is empty, garbled, or describes something other than a form field (e.g. an unrelated sentence, a description of a different kind of image entirely), ignore it completely - it is not evidence about what's on the form. Look at the image directly and read it yourself; if you genuinely cannot determine a value from the image either, write ? as the value with confidence unclear - never guess a plausible-sounding value that is not actually supported by what you can see.

Do not invent a value that is not visible in the image. Do not add commentary, brackets, or any punctuation not shown in the example. Output ONLY {num_columns} line{plural}, nothing else, then stop."""


def build_structuring_prompt(
    raw_ocr_text: str, column_names: list[str], template_override: str | None = None
) -> str:
    """
    Stage 2 of the two-stage pipeline (2026-07-13, redesigned per-FIELD
    2026-07-25): takes ONE column's raw OCR reading (from a fixed-task
    engine like Chandra, which can't produce our column schema
    natively) and asks an instruction-following model to confirm or
    correct it into that one column's real value. That column's own
    tightly-cropped, upscaled field IMAGE is passed alongside this
    prompt too (not text-only) - see run_two_stage_extraction() - so
    this stage can cross-reference the source directly if the raw OCR
    reading looks incomplete, ambiguous, or (a real observed failure,
    2026-07-25) entirely unrelated to a form field at all, rather than
    being blind to everything except stage 1's text.

    column_names is still accepted as a list (not a single str) for
    build_row_prompt-style flexibility and test_stage2_isolated.py's
    multi-column experiments, but run_two_stage_extraction() always
    calls this with exactly one column now - the template wording below
    is written for that single-field case first.

    Uses the same concrete-worked-example format proven necessary for
    single-stage extraction (see build_row_prompt's docstring - two
    separate real failures with abstract bracket-placeholder templates
    established this, not re-derived here).

    template_override (2026-07-22): lets an operator swap the
    INSTRUCTIONAL wording without editing this file's Python source -
    added directly in response to the prompt-example-leakage bug this
    same session, which required a code edit + redeploy just to test a
    wording change. If given, must be a str.format()-style template
    containing at minimum {raw_ocr_text}, {columns_str}, and
    {example_lines} placeholders (num_columns, plural, and field_hint
    are also available, matching the default template's own usage) -
    see config/prompts/structuring_stage2_default.txt for a real,
    working starting point (the exact template below, extracted to a
    file so editing it doesn't require touching this module). Falls
    back to the hardcoded default when None (unchanged behavior from
    before this parameter existed).

    field_hint (2026-07-25, per Jon's direction, reworked same day to
    drop candidate words entirely - see _FIELD_TYPE_HINTS' docstring
    comment): optional semantic context for the single field being
    asked about - expected content and typical format, never a
    candidate VALUE, from _FIELD_TYPE_HINTS/_format_field_hint() above.
    Only populated when column_names has exactly one entry (the only
    case a per-field hint makes sense for) and that column has a
    registered hint; otherwise "".

    A malformed template (missing a required placeholder, or a stray
    single brace) raises a clear ValueError naming the problem rather
    than a bare KeyError/IndexError from str.format() - the operator
    testing a new template mid-iteration needs to know WHAT broke, not
    just that something did.
    """
    columns_str = ", ".join(column_names)
    example_lines = "\n".join(
        f"{name}: {_EXAMPLE_VALUES.get(name, 'Zzyx')}|confirmed"
        for name in column_names
    )
    field_hint = _format_field_hint(column_names[0]) if len(column_names) == 1 else ""
    template = template_override if template_override is not None else _DEFAULT_STRUCTURING_TEMPLATE
    try:
        return template.format(
            raw_ocr_text=raw_ocr_text, columns_str=columns_str, example_lines=example_lines,
            num_columns=len(column_names), plural="s" if len(column_names) != 1 else "",
            field_hint=field_hint,
        )
    except KeyError as e:
        raise ValueError(
            f"Structuring prompt template references an unknown placeholder {e} - "
            f"available placeholders are: raw_ocr_text, columns_str, example_lines, "
            f"num_columns, plural, field_hint.") from e
    except (IndexError, ValueError) as e:
        raise ValueError(
            f"Structuring prompt template has malformed {{}} syntax (a stray single "
            f"brace? use {{{{ and }}}} for a literal brace): {e}") from e


def _field_bbox_from_keep_ranges(
    row_bbox: list[int], keep_ranges: list[tuple[int, int]],
) -> list[int]:
    """Full-image-coordinate bbox spanning the union of keep_ranges (x)
    and the row's own y-range - a real bbox for debug metadata, not just
    the raw range list, since keep_ranges alone doesn't carry the row's
    y0/y1 or make the union explicit."""
    xs = [x for r in keep_ranges for x in r]
    return [min(xs), row_bbox[1], max(xs), row_bbox[3]]


def _plain_text_reading(raw_reading: str) -> str:
    """
    Unwraps FlorenceLoader's task-token JSON format (e.g.
    '{"<OCR>": "actual text"}', see core/loaders/florence_loader.py's
    _run_generate - it always returns json.dumps(parsed, ...), by
    design, so the assessment tool can show region/bbox data for
    <OCR_WITH_REGION> too) down to the plain text value, when a stage-1
    OCR reading looks like that shape.

    Real bug found 2026-07-24: without this, a raw Florence reading like
    {"<OCR>": "Hawthana fume"} got embedded VERBATIM into stage 2's
    combined_reading/prompt as "the raw OCR text" - with the literal
    token "<OCR>" sitting right next to the real content. Stage 2
    (smolvlm2_2b) then echoed "<OCR>" back as its own answer for every
    field on that row, rather than the actual transcribed text - a
    prompt-format mismatch, not a cropping/plumbing bug (that layer was
    already fixed and confirmed correct via --debug-model-inputs).

    Deliberately narrow and defensive: only unwraps a dict with EXACTLY
    ONE key that looks like a Florence task token ("<...>"), and only
    when its value is a plain string (not a nested dict/list, as
    <OCR_WITH_REGION>'s value would be - unwrapping that into "plain
    text" would silently discard the region data, and produce a
    confusing stringified-dict fragment instead of an error, so it's
    left untouched instead - stage 2 seeing an odd dict-shaped string in
    that case is still more informative than a HALF-unwrapped one).
    Every other loader's plain-text output fails json.loads() and is
    returned completely unchanged - this has zero effect on any stage-1
    model except Florence.
    """
    try:
        parsed = json.loads(raw_reading)
    except (json.JSONDecodeError, TypeError):
        return raw_reading
    if (
        isinstance(parsed, dict) and len(parsed) == 1
        and isinstance(value := next(iter(parsed.values())), str)
        and next(iter(parsed.keys())).startswith("<") and next(iter(parsed.keys())).endswith(">")
    ):
        return value
    return raw_reading


def run_two_stage_extraction(
    sidecar_path: str,
    ocr_model_profile_name: str,
    structuring_model_profile_name: str,
    column_names: list[str],
    max_rows: int | None = None,
    ocr_prompt: str = "",
    structuring_prompt_template: str | None = None,
    stage1_upscale_target_height: int | None = 160,
    stage1_upscale_max_width: int = 4096,
    stage2_upscale_target_height: int | None = 160,
    stage2_upscale_max_width: int = 4096,
    tight_crop_padding_px: int = 20,
    tight_crop_padding_pct: float | None = None,
    debug_recorder: DebugModelInputRecorder | None = None,
    ocr_checkpoint: str | None = None,
    structure_checkpoint: str | None = None,
) -> list[RowExtractionResult]:
    """
    ocr_checkpoint / structure_checkpoint (2026-07-28): optional path to
    a saved LoRA adapter dir (data/outputs/<model>_lora_checkpoints/
    epoch_N/, see training/train_lora.py) applied on top of the stage-1/
    stage-2 base model respectively, via BaseLoader.apply_checkpoint()
    (core/loaders/base_loader.py) - same PeftModel.from_pretrained()
    mechanism training/test_lora_checkpoint.py already uses. Applied
    AFTER initialize_model_and_tokenizer(), before any generation calls.
    Only meaningful for in-process loaders (self.model is a real HF
    object) - raises clearly if pointed at a subprocess-backed loader
    (MoondreamLoader, DeepseekVL2Loader, HunyuanOcrLoader), since no
    LoRA training has targeted those and apply_checkpoint() has no way
    to reach a separate worker process's own model.

    Two-stage pipeline (2026-07-13, per Jon's direction): stage 1 runs
    a fixed-task OCR engine (e.g. Chandra) that can't follow our column
    schema natively; stage 2 runs an instruction-following model (e.g.
    Qwen3-VL-4B, Gemma) given BOTH a row image and stage 1's raw
    reading(s), structuring the result into our actual columns.

    FIELD-LEVEL STAGE 1 (redesigned 2026-07-24, replacing a real bug):
    stage 1 previously ran ONE OCR call per row against the FULL row
    crop (e.g. 3155x38px) with unwanted columns painted white, asking
    the model to find the wanted columns itself within that wide, mostly
    -blank image - confirmed via --debug-model-inputs output showing
    original_crop_dimensions == final_crop_dimensions (no cropping/
    upscaling ever applied) and a model_input.png only a few dozen
    pixels tall. That is NOT the intended pipeline: stage 1 must instead
    make ONE OCR call PER SELECTED COLUMN, each against that column's
    OWN tightly-cropped, independently-upscaled field image - exactly
    the same per-field crop shown to a human labeler in
    ground_truth_labeling_ui.py and the same crop run_single_column_
    extraction sends the model when running one column at a time. This
    function now reuses that identical logic via
    _resolve_column_field_mask() rather than reinventing it - see that
    function's docstring for why sidecar["columns"][name] (a REAL name
    -> boundary mapping) is the correct source, and NOT sidecar
    ["columns"]["__multi__"] (a single flat, unlabeled union of
    x-ranges with no per-column identity - can't drive per-field crops
    at all, and is no longer used anywhere in this function - see
    stage 2 below, also fixed 2026-07-24 to stop depending on it).

    Each column named in column_names MUST already have its own mask
    saved in the sidecar (row_segmentation_ui.py: select that column,
    NOT "__multi__", drag to define its kept range, Save or Next
    column) - checked upfront, before any model loads, so a missing
    mask fails fast with an actionable message rather than partway
    through a long run or (worse) silently falling back to a full-row
    OCR call.

    ocr_prompt (2026-07-13, per Jon's direction to compare stage-1
    prompts): passed to EVERY per-field stage-1 call as-is. Default ""
    preserves the ORIGINAL behavior - a real finding from testing
    (Jon, same date): a general instruction-following VLM used for
    stage 1 with NO prompt at all still organized fields more usefully
    for stage 2 than Chandra's raw fixed-task markdown did. Note: fixed
    -task engines like ChandraLoader IGNORE this entirely regardless of
    what's passed (see that loader's _build_prompt docstring).

    structuring_prompt_template (2026-07-22): passed straight through
    to build_structuring_prompt()'s template_override - see that
    function's docstring.

    stage1_upscale_target_height / stage1_upscale_max_width (2026-07-24,
    replaces the old single shared upscale_target_height param): stage 1's
    field crops are now tightly-cropped exactly like run_single_column_
    extraction's, so they need the SAME upscale-by-default treatment and
    the SAME justification (real crops measured as small as 79x36px -
    see run_single_column_extraction's docstring) - defaults to 160,
    not None.

    stage2_upscale_target_height / stage2_upscale_max_width: stage 2 is
    now FIELD-LEVEL, same as stage 1 (2026-07-25, replacing the
    2026-07-24 "tighten to the padded UNION of every selected column"
    design) - real evidence (Jon, same date) showed that even a
    tightened union crop still contains every selected column's own
    inter-column gutter whitespace, leaving stage 2 to solve two
    problems at once: reading the handwriting, AND figuring out which
    region of one shared image corresponds to which of up to 5 output
    columns. Stage 2 now gets the EXACT SAME per-field crop stage 1
    gets for that column (same tight_crop_ranges from
    column_field_masks), just upscaled with stage 2's own params
    instead of stage 1's - so it needs the same upscale-by-default
    treatment and the same justification as stage 1's default (real
    crops measured as small as 79x36px). Defaults to 160, not None.

    (For context: between 2026-07-24 and 2026-07-25 stage 2 briefly
    used a single per-row union-of-columns image instead of per-field
    crops, and for part of that window defaulted this upscale to None/
    off - real --debug-model-inputs captures from that period showed
    the union crop coming out at the same un-upscaled native row height
    as stage 1's crops used to before ITS upscale-by-default fix, e.g.
    original_crop_dimensions [3155, 38] -> final_crop_dimensions
    [1339, 38], and stage 2 falling back to "?|unclear" on nearly every
    field as a result. Both the union-crop design and its missing
    upscale are gone now, superseded by the field-level redesign above.)

    tight_crop_padding_px / tight_crop_padding_pct: padding around each
    field's own kept range, same meaning and defaults as run_single_
    column_extraction's identically-named parameters (passed straight
    through to the same tight_crop_to_ranges() underneath).

    Loads BOTH models for the duration of the run (not one at a time
    per row) - stage 1 processes every row (now: every column of every
    row) first, then stage 1's model is released before stage 2 loads,
    avoiding having two models resident in VRAM simultaneously.

    debug_recorder (2026-07-24, --debug-model-inputs; stage 2 naming
    updated 2026-07-25 for the field-level redesign above): see
    core/debug_dump.py. BOTH stages now get ONE debug item PER FIELD -
    "row_NNNN/column_NN_stage1" and "row_NNNN/column_NN_stage2"
    (column_NN from the field's position in sidecar["column_order"]
    when available, matching the real page layout's own column
    numbering, else its position within column_names) - not one per
    row. Each item's metadata includes column_index, column_name,
    row_bbox, and field_bbox (the field's own full-image-coordinate
    bounding box), on top of the fields every debug item already
    carries.
    """
    sidecar = load_sidecar(sidecar_path)
    source_path = sidecar["source_image_path"]
    deskew_angle = sidecar["deskew_angle"]
    rows = sidecar["rows"]
    if max_rows is not None:
        rows = rows[:max_rows]

    # Fail fast, before loading either model: every requested column
    # must have its OWN per-column mask (sidecar["columns"][name]), not
    # just be present in the unrelated "__multi__" scratch mask - see
    # this function's docstring for why those are not interchangeable.
    column_order = sidecar.get("column_order", [])
    sidecar_columns = sidecar.get("columns", {})
    missing = [
        name for name in column_names
        if not sidecar_columns.get(name, {}).get("mask_keep_ranges")
    ]
    # Also required: mask_apply_rows must actually be on for each column,
    # or _resolve_column_field_mask() below returns tight_crop_ranges=None
    # despite mask_keep_ranges being non-empty (mask_apply_rows=False
    # means "this mask isn't applied to rows" - a valid state for
    # run_single_column_extraction's more general use, but one this
    # mandatory-field-level path can't proceed with, since every column
    # here MUST tighten to its own range).
    disabled = [
        name for name in column_names
        if name not in missing and not sidecar_columns[name].get("mask_apply_rows", True)
    ]
    if missing or disabled:
        problems = []
        if missing:
            problems.append(f"no saved mask: {missing}")
        if disabled:
            problems.append(f"mask saved but \"apply to rows\" is off: {disabled}")
        raise ValueError(
            f"Field-level stage 1 requires each column to have its OWN saved, "
            f"row-applied mask - {'; '.join(problems)}. Mask each of these "
            f"individually in row_segmentation_ui.py (select the column BY "
            f"NAME - not \"__multi__\" - drag to define its kept range, check "
            f"\"Rows\", then Save or Next column) before running two-stage "
            f"extraction. \"__multi__\" masks (used for stage 2's own "
            f"whole-row image) do not carry per-column identity and cannot "
            f"be used here."
        )
    column_field_masks = {
        name: _resolve_column_field_mask(sidecar, sidecar_columns[name])
        for name in column_names
    }
    column_index_map = {
        name: (column_order.index(name) + 1 if name in column_order
               else i + 1)
        for i, name in enumerate(column_names)
    }

    # Stage 1: one OCR call PER SELECTED COLUMN per row, using a FRESH
    # loader instance, released before stage 2 loads (see docstring -
    # avoid two models resident in VRAM at once).
    ocr_config = load_model_config(ocr_model_profile_name)
    ocr_loader_cls = LOADER_REGISTRY.get(ocr_config.loader_class)
    if ocr_loader_cls is None:
        raise ValueError(f"No loader registered for {ocr_config.loader_class!r}")
    ocr_loader = ocr_loader_cls(ocr_config)

    # row_index -> {column_name: raw_reading}
    raw_readings: dict[int, dict[str, str]] = {}
    try:
        ocr_loader.initialize_model_and_tokenizer()
        if ocr_checkpoint:
            ocr_loader.apply_checkpoint(ocr_checkpoint)
        for row in rows:
            raw_readings[row["index"]] = {}
            for column_name in column_names:
                _row_masks_unused, _mask_active, tight_crop_ranges = column_field_masks[column_name]
                col_idx = column_index_map[column_name]
                debug_item = (debug_recorder or NOOP_RECORDER).new_item(
                    f"row_{row['index']:04d}/column_{col_idx:02d}_stage1"
                )
                field_image = crop_region_from_source(
                    source_path, row["bbox"], deskew_angle,
                    tight_crop_keep_ranges=tight_crop_ranges,
                    tight_crop_padding_px=tight_crop_padding_px,
                    tight_crop_padding_pct=tight_crop_padding_pct,
                    upscale_target_height=stage1_upscale_target_height,
                    upscale_max_width=stage1_upscale_max_width,
                    debug_stage_callback=debug_item.stage_callback(),
                )
                if field_image.mode != "RGB":
                    field_image = field_image.convert("RGB")
                debug_item.set_prompt(ocr_prompt)
                debug_item.set_meta(
                    source_image_path=source_path, row_index=row["index"],
                    column_index=col_idx, column_name=column_name,
                    row_bbox=row["bbox"],
                    field_bbox=_field_bbox_from_keep_ranges(row["bbox"], tight_crop_ranges),
                    model=ocr_model_profile_name,
                    reasoning_enabled=getattr(ocr_config, "reasoning_enabled", None),
                    generation_config_hash=(
                        ocr_config.content_hash() if hasattr(ocr_config, "content_hash") else None
                    ),
                    preprocessing={
                        "tight_crop_padding_px": tight_crop_padding_px,
                        "tight_crop_padding_pct": tight_crop_padding_pct,
                        "upscale_target_height": stage1_upscale_target_height,
                        "upscale_max_width": stage1_upscale_max_width,
                    },
                    stage="stage1_ocr_field",
                )
                try:
                    reading = ocr_loader._run_generate(field_image, ocr_prompt)
                    # Stored (and combined into stage 2's prompt) as
                    # PLAIN TEXT - the debug item still records the true,
                    # unmodified raw model output below, for anyone who
                    # needs to see exactly what came back. See
                    # _plain_text_reading()'s docstring for why this
                    # unwrap matters (a real bug, not a defensive guess).
                    raw_readings[row["index"]][column_name] = _plain_text_reading(reading)
                    debug_item.finalize(raw_output=reading)
                except Exception as e:
                    raw_readings[row["index"]][column_name] = f"[STAGE 1 ERROR: {e}]"
                    debug_item.finalize(raw_output=None, error=e)
                print(f"Stage 1 (OCR) row {row['index']} [{column_name}]: "
                      f"{raw_readings[row['index']][column_name]!r}")
    finally:
        _release_model(ocr_loader)

    # Stage 2: structure each FIELD separately (2026-07-25, replacing the
    # 2026-07-24 "whole selected-columns union" design) - real evidence
    # (Jon, same date) showed the union crop, even tightened, still left
    # every selected column's own inter-column gutter whitespace sitting
    # inside a single image, forcing stage 2 to both (a) read the
    # handwriting AND (b) figure out which region of that one image
    # belongs to which of the up-to-5 columns it was asked to fill in -
    # a second task stage 1 doesn't have to do at all, since stage 1
    # already gets ONE column's own tightly-cropped, upscaled field
    # image per call. Stage 2 now gets the EXACT SAME per-field image
    # (same column_field_masks[name] tight_crop_ranges, same
    # stage2_upscale_target_height/max_width in place of stage 1's own
    # upscale params) plus ONLY that field's own stage-1 raw reading as
    # the OCR hint - not all 5 fields' readings, which would reintroduce
    # the same "which text belongs to this image" ambiguity from the
    # hint side even with a single-column image. build_structuring_
    # prompt() and parse_row_output() already generalize correctly to a
    # single-column column_names list (verified: example_lines/
    # {num_columns}/{plural} all degrade correctly to one line), so no
    # separate single-field prompt builder was needed.
    #
    # Debug items are now "row_NNNN/column_NN_stage2", mirroring stage
    # 1's "row_NNNN/column_NN_stage1" naming exactly - stage 2 is
    # field-level evidence on disk now too, not row-level.
    struct_config = load_model_config(structuring_model_profile_name)
    struct_loader_cls = LOADER_REGISTRY.get(struct_config.loader_class)
    if struct_loader_cls is None:
        raise ValueError(f"No loader registered for {struct_config.loader_class!r}")
    struct_loader = struct_loader_cls(struct_config)

    results: list[RowExtractionResult] = []
    try:
        struct_loader.initialize_model_and_tokenizer()
        if structure_checkpoint:
            struct_loader.apply_checkpoint(structure_checkpoint)
        for row in rows:
            start = time.time()
            fields: dict[str, RowFieldValue] = {}
            per_field_raw_output: dict[str, str] = {}
            for column_name in column_names:
                _row_masks_unused, _mask_active, tight_crop_ranges = column_field_masks[column_name]
                col_idx = column_index_map[column_name]
                debug_item = (debug_recorder or NOOP_RECORDER).new_item(
                    f"row_{row['index']:04d}/column_{col_idx:02d}_stage2"
                )
                field_image = crop_region_from_source(
                    source_path, row["bbox"], deskew_angle,
                    tight_crop_keep_ranges=tight_crop_ranges,
                    tight_crop_padding_px=tight_crop_padding_px,
                    tight_crop_padding_pct=tight_crop_padding_pct,
                    upscale_target_height=stage2_upscale_target_height,
                    upscale_max_width=stage2_upscale_max_width,
                    debug_stage_callback=debug_item.stage_callback(),
                )
                if field_image.mode != "RGB":
                    field_image = field_image.convert("RGB")

                field_reading = raw_readings[row["index"]].get(column_name, "")
                # No "ColumnName: " label prefix (2026-07-26, per Jon's
                # direction) - a leftover from the pre-field-level design,
                # where stage2 saw a combined multi-field raw OCR block and
                # needed each line labeled to tell fields apart. Now stage2
                # gets exactly one field's own crop + prompt (which already
                # states "Field: {name}" explicitly), so relabeling the raw
                # OCR hint here is pure redundancy - and the same class of
                # risk as the "FieldName" literal-label bug already fixed
                # in the templates: an extra label sitting right next to
                # the value invites the model to echo it back unnecessarily
                # instead of just reporting the field's real value.
                prompt = build_structuring_prompt(
                    field_reading, [column_name],
                    template_override=structuring_prompt_template)
                debug_item.set_prompt(prompt)
                debug_item.set_meta(
                    source_image_path=source_path, row_index=row["index"],
                    column_index=col_idx, column_name=column_name,
                    row_bbox=row["bbox"],
                    field_bbox=_field_bbox_from_keep_ranges(row["bbox"], tight_crop_ranges),
                    model=structuring_model_profile_name,
                    reasoning_enabled=getattr(struct_config, "reasoning_enabled", None),
                    generation_config_hash=(
                        struct_config.content_hash() if hasattr(struct_config, "content_hash") else None
                    ),
                    preprocessing={
                        "tight_crop_padding_px": tight_crop_padding_px,
                        "tight_crop_padding_pct": tight_crop_padding_pct,
                        "upscale_target_height": stage2_upscale_target_height,
                        "upscale_max_width": stage2_upscale_max_width,
                    },
                    stage="stage2_structure_field",
                    stage1_raw_reading=field_reading,
                )
                try:
                    raw_output = struct_loader._run_generate(field_image, prompt)
                    parsed = parse_row_output(raw_output, [column_name])
                    fields.update(parsed)
                    per_field_raw_output[column_name] = raw_output
                    debug_item.finalize(raw_output=raw_output)
                except Exception as e:
                    per_field_raw_output[column_name] = f"[STAGE 2 ERROR: {e}]"
                    debug_item.finalize(raw_output=None, error=e)
                print(f"Stage 2 (structure) row {row['index']} [{column_name}]: "
                      f"{fields.get(column_name)!r}")

            combined_reading = "\n".join(
                f"{name}: {raw_readings[row['index']].get(name, '')}"
                for name in column_names
            )
            raw_output = "\n".join(
                f"{name}: {per_field_raw_output.get(name, '')}" for name in column_names
            )
            missing_fields = set(column_names) - set(fields.keys())
            schema_pass = len(missing_fields) == 0
            schema_error = f"Missing/dropped fields: {missing_fields}" if missing_fields else None

            ocr_label = f"{ocr_model_profile_name}[{ocr_checkpoint}]" if ocr_checkpoint else ocr_model_profile_name
            struct_label = (f"{structuring_model_profile_name}[{structure_checkpoint}]"
                             if structure_checkpoint else structuring_model_profile_name)
            results.append(RowExtractionResult(
                row_index=row["index"], bbox=row["bbox"], fields=fields,
                raw_output=raw_output, model=f"{ocr_label}+{struct_label}",
                runtime_seconds=time.time() - start,
                schema_pass=schema_pass, schema_error=schema_error,
                stage1_raw_output=combined_reading,
            ))
            print(f"Stage 2 (structure) row {row['index']}: "
                  f"{'OK' if schema_pass else 'INCOMPLETE'} "
                  f"({len(fields)}/{len(column_names)} columns) - "
                  f"{results[-1].runtime_seconds:.1f}s")
    finally:
        _release_model(struct_loader)

    return results


def run_row_extraction(
    sidecar_path: str,
    model_profile_name: str,
    column_names: list[str],
    bucket_config_overrides: dict | None = None,
    max_rows: int | None = None,
    header_field_names: list[str] | None = None,
    debug_recorder: DebugModelInputRecorder | None = None,
    checkpoint: str | None = None,
) -> tuple[RowExtractionResult | None, list[RowExtractionResult]]:
    """
    checkpoint (2026-07-28): optional path to a saved LoRA adapter dir
    (data/outputs/<model>_lora_checkpoints/epoch_N/, see training/
    train_lora.py) applied on top of model_profile_name via
    BaseLoader.apply_checkpoint() (core/loaders/base_loader.py) - same
    mechanism core.row_extraction.run_two_stage_extraction()'s own
    ocr_checkpoint/structure_checkpoint params use. Applied immediately
    after initialize_model_and_tokenizer(), before header or row
    extraction. Must have been trained from model_profile_name
    specifically - see apply_checkpoint()'s own docstring.

    Main orchestration: loads the sidecar, loads the model ONCE (not
    once per row - same VRAM-lifecycle discipline as model_assessment.py,
    see _release_model there), crops each row from the ORIGINAL source
    image in memory (crop_region_from_source - never from pre-saved row
    files, per the sidecar architecture's whole point), runs extraction,
    releases the model at the end regardless of success/failure.

    column_names: supplied explicitly per module docstring - not
    auto-detected.

    header_field_names (2026-07-13, Jon's direction: the page's
    district/sub-district/province/enumerator block is "required
    keywords for the context of the data following"): if given, the
    header region (0,0,width,table_top) is extracted FIRST, using this
    same loaded model, before the row loop - avoids loading the model
    twice when header+rows are wanted together (the common case). Pass
    None to skip header extraction entirely (row-only run).

    max_rows: optional cap for a quick test run (e.g. first 5 rows)
    before committing to a full 50-row pass.

    debug_recorder (2026-07-24, --debug-model-inputs): see
    core/debug_dump.py. None (default) means no debug capture - passed
    straight through to _extract_region() for both the header call and
    every row.

    Returns (header_result_or_None, row_results).
    """
    sidecar = load_sidecar(sidecar_path)
    source_path = sidecar["source_image_path"]
    deskew_angle = sidecar["deskew_angle"]
    rows = sidecar["rows"]
    row_masks, header_masks = _compute_scoped_masks(sidecar)
    if max_rows is not None:
        rows = rows[:max_rows]

    config = load_model_config(model_profile_name)
    if bucket_config_overrides:
        for field_name, value in bucket_config_overrides.items():
            setattr(config, field_name, value)

    loader_cls = LOADER_REGISTRY.get(config.loader_class)
    if loader_cls is None:
        raise ValueError(f"No loader registered for {config.loader_class!r}")

    loader = loader_cls(config)
    header_result: RowExtractionResult | None = None
    results: list[RowExtractionResult] = []

    try:
        loader.initialize_model_and_tokenizer()
        if checkpoint:
            loader.apply_checkpoint(checkpoint)
        # Audit-trail label (see RowExtractionResult.model) - same
        # "model[checkpoint]" convention run_two_stage_extraction() uses,
        # passed as model_profile_name to _extract_region() below since
        # that's the exact string it stores verbatim as .model - the
        # loader itself was already built from the real model_profile_name
        # above, so relabeling here doesn't affect what actually runs.
        model_label = f"{model_profile_name}[{checkpoint}]" if checkpoint else model_profile_name

        if header_field_names:
            table_bbox = sidecar.get("table_bbox")
            metadata_bbox = sidecar.get("metadata_bbox")
            if metadata_bbox is not None:
                header_bbox = metadata_bbox
            elif table_bbox is not None:
                width = sidecar["deskewed_image_size"][0]
                header_bbox = [0, 0, width, table_bbox[1]]
                print("NOTE: sidecar has no metadata_bbox - using the full "
                      "0-to-table_top block, which includes column headings/ "
                      "instructions text as well as page metadata. Re-save "
                      "the sidecar with metadata_bottom set (via "
                      "row_segmentation_ui.py) for a cleaner metadata-only "
                      "extraction.")
            else:
                header_bbox = None
                print("WARNING: header_field_names given but sidecar has "
                      "neither metadata_bbox nor table_bbox - skipping "
                      "header extraction.")

            if header_bbox is not None:
                header_result = _extract_region(
                    loader, source_path, deskew_angle, header_bbox,
                    header_field_names, row_index=0,
                    model_profile_name=model_label,
                    mask_ranges=header_masks,
                    debug_recorder=debug_recorder, debug_item_id="header",
                )
                print(f"Header: {'OK' if header_result.schema_pass else 'INCOMPLETE'} "
                      f"({len(header_result.fields)}/{len(header_field_names)} fields) - "
                      f"{header_result.runtime_seconds:.1f}s")

        for row in rows:
            result = _extract_region(
                loader, source_path, deskew_angle, row["bbox"], column_names,
                row_index=row["index"], model_profile_name=model_label,
                mask_ranges=row_masks,
                debug_recorder=debug_recorder,
            )
            results.append(result)
            print(f"Row {row['index']}: {'OK' if result.schema_pass else 'INCOMPLETE'} "
                  f"({len(result.fields)}/{len(column_names)} columns) - "
                  f"{result.runtime_seconds:.1f}s")

    finally:
        _release_model(loader)

    return header_result, results


def _column_state_or_raise(sidecar: dict, sidecar_path: str, column_name: str) -> dict:
    columns = sidecar.get("columns")
    if not columns:
        raise ValueError(
            f"Sidecar at {sidecar_path} has no per-column state yet - mask at least "
            f"one column in row_segmentation_ui.py (Save or Next column) before running "
            f"single-column extraction.")
    state = columns.get(column_name)
    if state is None:
        raise ValueError(
            f"Column {column_name!r} not found in sidecar's columns "
            f"({list(columns.keys())}). Mask it first in row_segmentation_ui.py.")
    return state


def _resolve_column_field_mask(
    sidecar: dict, column_state: dict,
) -> tuple[list[tuple[int, int]], bool, list[tuple[int, int]] | None]:
    """
    Returns (row_masks, mask_active, tight_crop_ranges) for ONE column,
    reading that column's OWN persisted mask - sidecar["columns"]
    [column_name]["mask_keep_ranges"] - the same per-column, NAME-TAGGED
    boundary data run_single_column_extraction and
    ground_truth_labeling_ui.py already read to show "the exact crop
    the model/a human labeler sees" for a given field.

    2026-07-24: extracted out of run_single_column_extraction (which
    had this inline) so run_two_stage_extraction's field-level stage-1
    pass (see that function's docstring) can reuse the IDENTICAL logic
    instead of a second, possibly-diverging interpretation of what "this
    column's boundary" means - a real risk flagged directly: this
    project already has a SEPARATE, differently-shaped mask concept
    (sidecar["columns"]["__multi__"]["mask_keep_ranges"]) that must NOT
    be confused with this one. __multi__ is a single flat, UNLABELED
    union of x-ranges (row_segmentation_ui.py's "Select column to keep"
    multi-mask mode appends every click-drag to one shared list with no
    per-column identity retained) - useful for PAINTING several wanted
    columns' worth of area in one whole-row image, but it cannot tell
    you which sub-range belongs to which named column, so it cannot
    drive per-field cropping. Only sidecar["columns"][name] (the
    persistent per-column architecture from the 2026-07-21/22 sidecar
    redesign) carries a real name -> boundary mapping - that's what
    this function reads, and it is a required input (not an optional
    enhancement) for real field-level extraction.

    row_masks: exclude-ranges suitable for apply_column_mask() (paint
    everything outside this column's range white) - not used for the
    field-level crop itself (tight_crop_ranges narrows the image
    instead, so there's no "everything else" left to paint), but kept
    for callers that want a masked-not-narrowed view (e.g. showing the
    same field within a wider row image).
    """
    width = sidecar["deskewed_image_size"][0]
    keep_ranges = [tuple(k) for k in column_state.get("mask_keep_ranges", [])]
    mask_active = bool(keep_ranges) and column_state.get("mask_apply_rows", True)
    row_masks = compute_exclude_ranges(keep_ranges, width) if mask_active else []
    # Only tighten when a mask is actually active - with no mask, the
    # whole row is the intended input and there's nothing to tighten to.
    tight_crop_ranges = keep_ranges if mask_active else None
    return row_masks, mask_active, tight_crop_ranges


def run_single_column_extraction(
    sidecar_path: str,
    model_profile_name: str,
    column_name: str | None = None,
    max_rows: int | None = None,
    mark_done: bool = True,
    tight_crop_padding_px: int = 20,
    tight_crop_padding_pct: float | None = None,
    upscale_target_height: int | None = 160,
    upscale_max_width: int = 4096,
    debug_recorder: DebugModelInputRecorder | None = None,
    checkpoint: str | None = None,
) -> list[RowExtractionResult]:
    """
    checkpoint (2026-07-28): optional LoRA adapter path applied on top
    of model_profile_name - same convention as run_row_extraction()'s
    own checkpoint param, see that function's docstring.

    Extracts ONE column (using that column's OWN stored mask, not a
    global one) across every row, and writes the results straight into
    the sidecar's persistent columns[column_name]["results"] via
    update_sidecar() - a single merge-write at the end of the run, not
    one write per row, so an atomic sidecar write doesn't happen 50+
    times for one column pass. This is the piece that was missing:
    run_row_extraction() above only ever produced a separate CSV/JSON,
    never fed results back into the sidecar the persistent-column
    workflow (row_segmentation_ui.py's mask -> Next -> mask -> Next
    cycle) depends on for showing real progress, not just mask
    completion.

    column_name: defaults to the sidecar's current active_column - the
    natural thing to run right after masking it in the UI. Can be
    overridden to re-run a specific already-done column.

    mark_done: if True (default), the column's status is set "done"
    after a successful pass (this IS the "processing complete for this
    column" signal, same as the UI's "Next column" button, so running
    this from the CLI has the same effect on the sidecar's progress
    state as doing it by hand). Set False for a quick test pass you
    don't want counted as final.

    tight_crop_padding_px / tight_crop_padding_pct (2026-07-22): the
    image actually sent to the model is tightened to a padded box
    around this column's kept mask range - see
    core.row_segmentation.tight_crop_to_ranges()'s docstring for why
    (the untightened crop was the FULL row width, ~90% blank, for
    every single-column extraction before this fix). padding_pct, if
    given, overrides padding_px with a percentage of the kept range's
    own width instead of a fixed pixel margin. Only applied when the
    column actually has an active mask (mask_apply_rows AND a non-
    empty mask_keep_ranges) - a column with no mask has no range to
    tighten to, and sends the full unmasked row as before.

    upscale_target_height (2026-07-23, DEFAULT CHANGED from prior
    behavior - real evidence, not a guess): defaults to 160 now, not
    None. A real inventory this session found tightened column crops
    as small as 79x36 pixels - not enough resolution for most vision
    encoders to have real detail to work with, regardless of model
    capability. Unlike tight-cropping (a strict improvement with no
    tradeoff), upscaling doesn't add real information and its
    interaction with any given model's own internal preprocessing is
    untested per-model - but given how small these crops measured in
    practice, defaulting to upscale-on is the more defensible choice
    than continuing to silently under-resource every model tested. Set
    to None to reproduce EXACT prior behavior for a specific controlled
    comparison. Recorded in extraction_meta's "preprocessing" field
    either way, so results from before/after this default changed stay
    distinguishable - same principle as tight_crop_applied.

    Still writes a CSV/JSON alongside (via save_results_csv/json in the
    caller, same as run_row_extraction) for tooling that reads those
    directly - this doesn't replace that, it adds the sidecar as a
    second, persistent destination for the same results.

    debug_recorder (2026-07-24, --debug-model-inputs): see
    core/debug_dump.py. None (default) means no debug capture.
    """
    sidecar = load_sidecar(sidecar_path)
    if column_name is None:
        column_name = sidecar.get("active_column")
        if column_name is None:
            raise ValueError(
                f"No column_name given and sidecar at {sidecar_path} has no "
                f"active_column set (either all columns are done, or none has "
                f"been masked yet).")

    column_state = _column_state_or_raise(sidecar, sidecar_path, column_name)

    source_path = sidecar["source_image_path"]
    deskew_angle = sidecar["deskew_angle"]
    rows = sidecar["rows"]
    if max_rows is not None:
        rows = rows[:max_rows]

    row_masks, mask_active, tight_crop_ranges = _resolve_column_field_mask(sidecar, column_state)

    config = load_model_config(model_profile_name)
    loader_cls = LOADER_REGISTRY.get(config.loader_class)
    if loader_cls is None:
        raise ValueError(f"No loader registered for {config.loader_class!r}")

    loader = loader_cls(config)
    results: list[RowExtractionResult] = []

    try:
        loader.initialize_model_and_tokenizer()
        if checkpoint:
            loader.apply_checkpoint(checkpoint)
        model_label = f"{model_profile_name}[{checkpoint}]" if checkpoint else model_profile_name
        for row in rows:
            result = _extract_region(
                loader, source_path, deskew_angle, row["bbox"], [column_name],
                row_index=row["index"], model_profile_name=model_label,
                mask_ranges=row_masks,
                tight_crop_keep_ranges=tight_crop_ranges,
                tight_crop_padding_px=tight_crop_padding_px,
                tight_crop_padding_pct=tight_crop_padding_pct,
                upscale_target_height=upscale_target_height,
                upscale_max_width=upscale_max_width,
                debug_recorder=debug_recorder,
            )
            results.append(result)
            field = result.fields.get(column_name)
            status = "OK" if field is not None else "MISSING/DROPPED"
            print(f"Row {row['index']} [{column_name}]: {status} - "
                  f"{result.runtime_seconds:.1f}s")
    finally:
        _release_model(loader)

    row_results = {}
    for r in results:
        field = r.fields.get(column_name)
        row_results[str(r.row_index)] = {
            "value": field.value if field else None,
            "confidence": field.confidence.value if field else None,
            "schema_pass": r.schema_pass,
            "raw_output": r.raw_output,
        }

    patch = {
        "results": row_results,
        "extraction_meta": {
            "model": model_profile_name,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "row_count": len(results),
            # 2026-07-22: distinguishes results extracted with the
            # tight-crop fix active from earlier "baseline" runs that
            # sent the model a full-row-width, mostly-blank image
            # (Name/Age from the first smolvlm2 test batch predate
            # this). Compare model-quality results across runs only
            # when this flag matches - the input distribution genuinely
            # changed, not just the model or column.
            "tight_crop_applied": tight_crop_ranges is not None,
            # 2026-07-23: same idea, for the upscale-to-target-height
            # default change - see run_single_column_extraction's
            # upscale_target_height docstring. None here means this run
            # used the pre-2026-07-23 tiny-crop input (e.g. the real
            # 79x36px 'Sex' column crops found this session); a number
            # means every result in this batch was upscaled to at least
            # that height before the model ever saw it.
            "preprocessing": {
                "upscale_target_height": upscale_target_height,
                "upscale_max_width": upscale_max_width if upscale_target_height else None,
            },
            # 2026-07-24: same distinguishability principle as
            # tight_crop_applied/preprocessing above - whether
            # constrained decoding (core/loaders/constrained_decoding.py)
            # was active for this batch, so results with and without it
            # stay comparable/auditable later. Unlike the document-level
            # ExtractionResult (core/schema.py), RowExtractionResult
            # carries no generation_config_hash field at all - this
            # sidecar-level extraction_meta flag is the ONLY record of
            # this setting for row-level results, not a redundant copy
            # of something already traceable elsewhere.
            "restrict_output_charset": config.restrict_output_charset,
        },
    }
    if mark_done:
        patch["status"] = "done"
    update_sidecar(sidecar_path, column_name, patch)

    return results


def extract_page_header(
    sidecar_path: str,
    model_profile_name: str,
    header_field_names: list[str],
) -> RowExtractionResult:
    """
    Standalone header-only extraction - for a quick test of just the
    header block without running any rows. For the common case (header
    + rows together), use run_row_extraction(..., header_field_names=...)
    instead, which shares one model load rather than paying the load/
    unload cost twice.

    Extracts the page's own administrative/metadata block (district,
    sub-district, province, enumerator name, page number, etc.) -
    prefers the sidecar's metadata_bbox (the narrower 0-to-metadata_
    bottom region, excluding column-heading/instructions text) if
    present, falling back to the full 0-to-table_top block otherwise.
    Deliberately excluded from every person row (2026-07-13, Jon's
    direction: "required keywords for the context of the data
    following" - provenance/context for the rows below it, not a person
    entry itself - see project history: table_top correction fixed a
    real bug where this block was being read AS row 1 on one page).

    Uses FULL image width (not table_bbox's x0/x1, even if those were
    narrowed to exclude table margins) - census page headers (province/
    district/enumerator info) commonly span wider than the data table
    itself, so narrowing to the table's x-range risks cutting off real
    header content that starts further left/right than the table body.
    """
    sidecar = load_sidecar(sidecar_path)
    source_path = sidecar["source_image_path"]
    deskew_angle = sidecar["deskew_angle"]
    width = sidecar["deskewed_image_size"][0]
    _row_masks_unused, header_masks = _compute_scoped_masks(sidecar)

    metadata_bbox = sidecar.get("metadata_bbox")
    table_bbox = sidecar.get("table_bbox")
    if metadata_bbox is not None:
        header_bbox = metadata_bbox
    elif table_bbox is not None:
        header_bbox = [0, 0, width, table_bbox[1]]
        print("NOTE: sidecar has no metadata_bbox - using the full "
              "0-to-table_top block, which includes column headings/ "
              "instructions text as well as page metadata.")
    else:
        raise ValueError(
            "Sidecar has neither metadata_bbox nor table_bbox - cannot "
            "determine where the header region ends. Re-save the sidecar "
            "with table_top (and ideally metadata_bottom) set via "
            "row_segmentation_ui.py before extracting the header."
        )

    config = load_model_config(model_profile_name)
    loader_cls = LOADER_REGISTRY.get(config.loader_class)
    if loader_cls is None:
        raise ValueError(f"No loader registered for {config.loader_class!r}")

    loader = loader_cls(config)
    try:
        loader.initialize_model_and_tokenizer()
        result = _extract_region(
            loader, source_path, deskew_angle, header_bbox, header_field_names,
            row_index=0, model_profile_name=model_profile_name,
            mask_ranges=header_masks,
        )
        print(f"Header: {'OK' if result.schema_pass else 'INCOMPLETE'} "
              f"({len(result.fields)}/{len(header_field_names)} fields) - "
              f"{result.runtime_seconds:.1f}s")
    finally:
        _release_model(loader)

    return result


def save_results_csv(results: list[RowExtractionResult], path, column_names: list[str]) -> None:
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["row_index", "schema_pass", "runtime_seconds"]
        for col in column_names:
            header.append(f"{col}_value")
            header.append(f"{col}_confidence")
        header.append("stage1_raw_output")
        writer.writerow(header)
        for r in results:
            row = [r.row_index, r.schema_pass, f"{r.runtime_seconds:.2f}"]
            for col in column_names:
                if col in r.fields:
                    row.append(r.fields[col].value)
                    row.append(r.fields[col].confidence.value)
                else:
                    row.append("")
                    row.append("")
            row.append(r.stage1_raw_output or "")
            writer.writerow(row)


def save_results_json(results: list[RowExtractionResult], path) -> None:
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in results], f, indent=2, default=str)
