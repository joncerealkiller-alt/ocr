"""
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

import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from pydantic import BaseModel

from core.row_segmentation import (
    load_sidecar, crop_region_from_source, compute_exclude_ranges, update_sidecar,
)
from core.loader_registry import LOADER_REGISTRY
from core.loaders.base_loader import load_model_config
from core.schema import ConfidenceLevel

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
        f"{name}: {_EXAMPLE_VALUES.get(name, 'Smith')}|confirmed"
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
    "Name": "John Smith", "Age": "45", "Sex": "M",
    # "Relationship to Head" FIXED 2026-07-16 (real bug, found via a
    # second-model handoff report + confirmed by inspecting this dict
    # directly): was "Head" - the field's own NAME doubling as its own
    # worked-example VALUE, meaning every single stage-2 prompt for
    # this column literally showed the model "Relationship to Head:
    # Head|confirmed" as the example to follow. Directly explains a
    # real, reproduced-across-multiple-stage-1-models pattern: stage 2
    # defaulting to "Head" whenever stage 1's reading was ambiguous,
    # a list, or garbage (e.g. got_ocr2 row 6: stage1_raw_output was
    # "29.44", a number with zero relationship-word content, and stage
    # 2 still output "Head" - coincidentally correct for that specific
    # row, but not earned, and would silently mask bad stage-1 output
    # as a correct extraction on any row where it wasn't). Replaced
    # with "Boarder" - a real value but NOT the modal/most-common
    # answer on real census pages (unlike "Son", which would just make
    # a future version of the same bias less visible by blending into
    # an already-frequently-correct guess instead of eliminating the
    # self-reference problem).
    "Relationship to Head": "Boarder", "Birthplace": "Ontario",
    "Occupation": "Farmer", "Province": "Manitoba",
}


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
            bare_conf = value_raw.lstrip(" :,;.-").strip().lower()
            if bare_conf in CONFIDENCE_MAP:
                result[real_column] = RowFieldValue(
                    value="", confidence=CONFIDENCE_MAP[bare_conf])
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
        confidence = CONFIDENCE_MAP.get(conf_part.strip().lower())
        if confidence is None:
            continue
        result[real_column] = RowFieldValue(value=value_part.strip(), confidence=confidence)

    return result


def _compute_scoped_masks(sidecar: dict) -> tuple[list, list]:
    """
    Computes the actual paint-white exclude ranges separately for row
    crops and the header crop, from the sidecar's stored keep_ranges +
    scope flags (2026-07-15, per Jon's corrected design: click the
    column to KEEP, everything else masked; header and row columns
    don't share the same layout, so scope is independent per region).

    Returns (row_mask_ranges, header_mask_ranges) - either may be []
    if no keep_ranges are set, or if that particular scope's checkbox
    was off when the sidecar was saved.
    """
    keep_ranges = [tuple(k) for k in sidecar.get("mask_keep_ranges", [])]
    width = sidecar["deskewed_image_size"][0]
    row_masks = (
        compute_exclude_ranges(keep_ranges, width)
        if keep_ranges and sidecar.get("mask_apply_rows", False) else []
    )
    header_masks = (
        compute_exclude_ranges(keep_ranges, width)
        if keep_ranges and sidecar.get("mask_apply_header", False) else []
    )
    return row_masks, header_masks


def _extract_region(
    loader, source_path: str, deskew_angle: float, bbox: list[int],
    field_names: list[str], row_index: int, model_profile_name: str,
    mask_ranges: list[tuple[int, int]] | None = None,
    stage1_raw_output: str | None = None,
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

    stage1_raw_output (2026-07-16, real gap found by Jon): the two-
    stage pipeline previously only ever printed a CHARACTER COUNT for
    stage 1's reading ("Stage 1 (OCR) row 1: 99 chars"), never the
    actual text, and never saved it anywhere - made it impossible to
    tell whether a lost value (e.g. an "m" for months, a "?" for an
    illegible digit) was dropped by stage 1 itself or by stage 2's
    structuring pass. Passed through here so it can be attached to the
    saved result for exactly this kind of diagnosis.
    """
    start = time.time()
    prompt = build_row_prompt(field_names)
    region_image = crop_region_from_source(source_path, bbox, deskew_angle, mask_ranges)
    if region_image.mode != "RGB":
        region_image = region_image.convert("RGB")

    try:
        raw_output = loader._run_generate(region_image, prompt)
        fields = parse_row_output(raw_output, field_names)
        missing = set(field_names) - set(fields.keys())
        schema_pass = len(missing) == 0
        schema_error = f"Missing/dropped fields: {missing}" if missing else None
    except Exception as e:
        raw_output = f"[ERROR: {e}]"
        fields = {}
        schema_pass = False
        schema_error = str(e)

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
    """
    import gc
    import torch
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


def build_structuring_prompt(raw_ocr_text: str, column_names: list[str]) -> str:
    """
    Stage 2 of the two-stage pipeline (2026-07-13): takes a raw OCR
    reading (from a fixed-task engine like Chandra, which can't produce
    our column schema natively) and asks an instruction-following model
    to organize it into our actual columns. The row IMAGE is passed
    alongside this prompt too (not text-only) - see run_two_stage_
    extraction() - so this stage can cross-reference the source
    directly if the raw OCR reading looks incomplete or ambiguous,
    rather than being blind to everything except stage 1's text.

    Uses the same concrete-worked-example format proven necessary for
    single-stage extraction (see build_row_prompt's docstring - two
    separate real failures with abstract bracket-placeholder templates
    established this, not re-derived here).
    """
    columns_str = ", ".join(column_names)
    example_lines = "\n".join(
        f"{name}: {_EXAMPLE_VALUES.get(name, 'Smith')}|confirmed"
        for name in column_names
    )
    return f"""Below is a raw OCR reading of ONE ROW from a census/tabular record, produced by a different tool. The image of that same row is also provided to you directly.

Raw OCR reading:
\"\"\"
{raw_ocr_text}
\"\"\"

Your job: organize the information above into the columns listed below, using the actual image to check or correct the raw OCR reading where it looks wrong, incomplete, or ambiguous - do not just copy the raw reading blindly if the image shows something different.

The columns, in order, are: {columns_str}

For each column, write the column name, a colon, the value, a pipe character, then a confidence word. Here is a complete worked example using invented data (not from your image - just showing the format):

{example_lines}

Now do the same thing for the REAL columns of this row, using the values actually visible in the image and supported by the raw OCR reading above - not the example values.

Confidence must be exactly one of: confirmed, partial, unclear
- confirmed: every character is clearly readable with no doubt
- partial: mostly readable, some uncertainty
- unclear: not legible enough to read with confidence

If a column is genuinely blank for this row, write the column name and colon followed immediately by the pipe and confirmed, with nothing in between - for example: Occupation:|confirmed

If the raw OCR reading above is empty, garbled, or clearly unrelated to this field, do NOT treat that as evidence the field is blank on the form. Look at the image directly and read it yourself; if you genuinely cannot determine a value from the image either, write ? as the value with confidence unclear - never guess a plausible-sounding value that is not actually supported by what you can see.

Do not invent a value that is not visible in the image. Do not add commentary, brackets, or any punctuation not shown in the example. Output ONLY {len(column_names)} line{'s' if len(column_names) != 1 else ''}, one per column, then stop."""


def run_two_stage_extraction(
    sidecar_path: str,
    ocr_model_profile_name: str,
    structuring_model_profile_name: str,
    column_names: list[str],
    max_rows: int | None = None,
    ocr_prompt: str = "",
) -> list[RowExtractionResult]:
    """
    Two-stage pipeline (2026-07-13, per Jon's direction): stage 1 runs
    a fixed-task OCR engine (e.g. Chandra) that can't follow our column
    schema natively, producing raw text; stage 2 runs an instruction-
    following model (e.g. Qwen3-VL-4B, Gemma) given BOTH the row image
    and stage 1's raw text, structuring the result into our actual
    columns.

    ocr_prompt (2026-07-13, per Jon's direction to compare stage-1
    prompts): passed to stage 1's _run_generate() as-is. Default ""
    preserves the ORIGINAL behavior - a real finding from testing
    (Jon, same date): a general instruction-following VLM used for
    stage 1 with NO prompt at all still organized fields more usefully
    for stage 2 than Chandra's raw fixed-task markdown did, suggesting
    even minimal guidance might help further. Note: fixed-task engines
    like ChandraLoader IGNORE this entirely regardless of what's passed
    (see that loader's _build_prompt docstring) - this parameter only
    has an effect when ocr_model_profile_name points to a genuine
    instruction-following loader.

    Loads BOTH models for the duration of the run (not one at a time
    per row) - stage 1 processes every row first, then stage 1's model
    is released before stage 2 loads, avoiding having two models
    resident in VRAM simultaneously on top of everything else tested
    this session (most of it was already tight on a single model).
    """
    sidecar = load_sidecar(sidecar_path)
    source_path = sidecar["source_image_path"]
    deskew_angle = sidecar["deskew_angle"]
    rows = sidecar["rows"]
    row_masks, _header_masks_unused = _compute_scoped_masks(sidecar)
    if max_rows is not None:
        rows = rows[:max_rows]

    # Stage 1: raw OCR reading per row, using a FRESH loader instance,
    # released before stage 2 loads (see docstring - avoid two models
    # resident in VRAM at once).
    ocr_config = load_model_config(ocr_model_profile_name)
    ocr_loader_cls = LOADER_REGISTRY.get(ocr_config.loader_class)
    if ocr_loader_cls is None:
        raise ValueError(f"No loader registered for {ocr_config.loader_class!r}")
    ocr_loader = ocr_loader_cls(ocr_config)

    raw_readings: dict[int, str] = {}
    try:
        ocr_loader.initialize_model_and_tokenizer()
        for row in rows:
            row_image = crop_region_from_source(source_path, row["bbox"], deskew_angle, row_masks)
            if row_image.mode != "RGB":
                row_image = row_image.convert("RGB")
            try:
                raw_readings[row["index"]] = ocr_loader._run_generate(row_image, ocr_prompt)
            except Exception as e:
                raw_readings[row["index"]] = f"[STAGE 1 ERROR: {e}]"
            print(f"Stage 1 (OCR) row {row['index']}: {raw_readings[row['index']]!r}")
    finally:
        _release_model(ocr_loader)

    # Stage 2: structure each row's raw reading + image into our columns.
    struct_config = load_model_config(structuring_model_profile_name)
    struct_loader_cls = LOADER_REGISTRY.get(struct_config.loader_class)
    if struct_loader_cls is None:
        raise ValueError(f"No loader registered for {struct_config.loader_class!r}")
    struct_loader = struct_loader_cls(struct_config)

    results: list[RowExtractionResult] = []
    try:
        struct_loader.initialize_model_and_tokenizer()
        for row in rows:
            start = time.time()
            row_image = crop_region_from_source(source_path, row["bbox"], deskew_angle, row_masks)
            if row_image.mode != "RGB":
                row_image = row_image.convert("RGB")

            prompt = build_structuring_prompt(raw_readings[row["index"]], column_names)
            try:
                raw_output = struct_loader._run_generate(row_image, prompt)
                fields = parse_row_output(raw_output, column_names)
                missing = set(column_names) - set(fields.keys())
                schema_pass = len(missing) == 0
                schema_error = f"Missing/dropped fields: {missing}" if missing else None
            except Exception as e:
                raw_output = f"[STAGE 2 ERROR: {e}]"
                fields = {}
                schema_pass = False
                schema_error = str(e)

            results.append(RowExtractionResult(
                row_index=row["index"], bbox=row["bbox"], fields=fields,
                raw_output=raw_output, model=(
                    f"{ocr_model_profile_name}+{structuring_model_profile_name}"
                ),
                runtime_seconds=time.time() - start,
                schema_pass=schema_pass, schema_error=schema_error,
                stage1_raw_output=raw_readings[row["index"]],
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
) -> tuple[RowExtractionResult | None, list[RowExtractionResult]]:
    """
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
                    model_profile_name=model_profile_name,
                    mask_ranges=header_masks,
                )
                print(f"Header: {'OK' if header_result.schema_pass else 'INCOMPLETE'} "
                      f"({len(header_result.fields)}/{len(header_field_names)} fields) - "
                      f"{header_result.runtime_seconds:.1f}s")

        for row in rows:
            result = _extract_region(
                loader, source_path, deskew_angle, row["bbox"], column_names,
                row_index=row["index"], model_profile_name=model_profile_name,
                mask_ranges=row_masks,
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


def run_single_column_extraction(
    sidecar_path: str,
    model_profile_name: str,
    column_name: str | None = None,
    max_rows: int | None = None,
    mark_done: bool = True,
) -> list[RowExtractionResult]:
    """
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

    Still writes a CSV/JSON alongside (via save_results_csv/json in the
    caller, same as run_row_extraction) for tooling that reads those
    directly - this doesn't replace that, it adds the sidecar as a
    second, persistent destination for the same results.
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

    width = sidecar["deskewed_image_size"][0]
    keep_ranges = [tuple(k) for k in column_state.get("mask_keep_ranges", [])]
    row_masks = (
        compute_exclude_ranges(keep_ranges, width)
        if keep_ranges and column_state.get("mask_apply_rows", True) else []
    )

    config = load_model_config(model_profile_name)
    loader_cls = LOADER_REGISTRY.get(config.loader_class)
    if loader_cls is None:
        raise ValueError(f"No loader registered for {config.loader_class!r}")

    loader = loader_cls(config)
    results: list[RowExtractionResult] = []

    try:
        loader.initialize_model_and_tokenizer()
        for row in rows:
            result = _extract_region(
                loader, source_path, deskew_angle, row["bbox"], [column_name],
                row_index=row["index"], model_profile_name=model_profile_name,
                mask_ranges=row_masks,
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
