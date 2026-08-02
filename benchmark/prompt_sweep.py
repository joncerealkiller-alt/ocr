"""
Batch model/prompt sweep tool - answers "which model+prompt combo
extracts this best?", as distinct from the production pipeline (which
just answers "can we extract this at all?" using whichever combo has
already proven best). Built 2026-07-29 per Jon's direction, after the
model registry grew a batch of new OCR loaders (deepseek_vl2,
granite_vision_4_1, hunyuan_ocr, lfm2_vl, nanonets_ocr2) with no
systematic way yet to compare them - or their stage-1 OCR prompts -
against each other.

STAYS A SEPARATE DEVELOPER TOOL, same isolation contract as
scripts/model_assessment.py: never imports core/classifier.py or
core/extractor.py, never writes to config/pipeline.yaml, bucket CSVs,
or the manifest. Only reads config/models/*.yaml, config/prompts/*.txt,
data/outputs/ground_truth_log.jsonl, and existing sidecar JSONs
(read-only); only writes to data/outputs/prompt_sweep_log.jsonl and
data/outputs/prompt_sweep_runs/.

WHAT GETS SWEPT: the two-stage extraction pipeline
(core/row_extraction.py's run_two_stage_extraction) already has two
independently swappable prompts - stage 1's OCR reading prompt (a
plain string passed straight to the OCR loader's _run_generate) and
stage 2's structuring prompt template (build_structuring_prompt's
template_override). Both are swept here by pointing --ocr-prompt-dir
and/or --structuring-prompt-dir at a folder of candidate .txt files -
whichever one(s) you give get swept; the other stays fixed. Give both
for a full cross-product sweep. The single-stage path
(run_single_column_extraction) builds its prompt internally with no
override param and is out of scope here.

config/prompts/ holds candidates for EVERY pipeline stage in one flat
folder (classifier_*, extractor_*, loader_*, ...), not a per-stage
subfolder - pointing --ocr-prompt-dir straight at it only picks up
ocr_stage1_*.txt files (--ocr-prompt-glob's default), and
--structuring-prompt-dir only picks up structuring_stage2_*.txt
(--structuring-prompt-glob's default), so a sweep doesn't accidentally
run every other stage's prompts through this stage too (a real mistake
caught 2026-07-30 - the queue showed classifier_*/extractor_*/loader_*
files where only ocr_stage1_* candidates were wanted). Override the
glob flags for a curated subfolder that doesn't follow this naming
convention.

WHY THIS DUPLICATES run_two_stage_extraction()'S LOOP INSTEAD OF
CALLING IT: that function loads+releases BOTH models on every single
call - correct for the pipeline's one-shot use, wrong for a sweep,
which wants each model to load EXACTLY ONCE and then run every prompt
variant of the stage it's assigned to. This module restructures the
same two-stage logic (same field-level crop/prompt/parse primitives,
imported directly - see below) into two batched phases instead:
Phase A loads the OCR model once and runs every OCR-prompt variant
across every example; Phase B loads the structuring model once and
runs every (OCR variant x structuring-prompt variant) combination
across every example. Never two models resident in VRAM at once - same
discipline run_two_stage_extraction already follows, for the same
reason (see that function's own docstring).

SCORING: reuses benchmark.score_two_stage_against_ground_truth's exact
_is_correct/_record_to_expected abstention-aware comparison (never
duplicate that logic - see that module's own docstring) plus
training.test_lora_checkpoint's _classify_mismatch three-way failure-
mode taxonomy for every non-exact-match: honest_hedge (self-flagged
uncertain, e.g. contains "?", or silent on a genuinely illegible/blank
field) and incorrectly_silent (silent on a field with a real answer)
are reported together as "Abstained" - a safe, non-hallucinating miss;
confident_wrong (asserted specific false content with no self-flagged
doubt) is reported as "Wrong" - the real hallucination-risk failure
mode. This is deliberately the same three-way split test_lora_
checkpoint.py already uses for LoRA eval, applied here to model/prompt
combos instead of checkpoints.

Usage (single-axis sweep, stage-1 OCR prompts, structuring held fixed):
    python benchmark/prompt_sweep.py \\
        --ocr-model lfm2_vl_1_6b --ocr-prompt-dir config/prompts/ocr_stage1_candidates \\
        --structure-model qwen3vl4b

Usage (stage-2 structuring prompts swept instead, OCR held fixed):
    python benchmark/prompt_sweep.py \\
        --ocr-model chandra --structure-model qwen3vl4b \\
        --structuring-prompt-dir config/prompts/structuring_candidates

See benchmark/prompt_sweep_report.py to render the accumulated history
log (every sweep ever run, not just the latest) as one sortable table.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Windows console/redirected-file default codepage can't represent
# arbitrary model output - same real crash this project has already
# hit twice (core/row_extraction.py, training/test_lora_checkpoint.py).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from core.loader_registry import LOADER_REGISTRY
from core.loaders.base_loader import load_model_config
from core.row_extraction import (
    _field_bbox_from_keep_ranges, _release_model, _resolve_column_field_mask,
    build_structuring_prompt, parse_row_output,
)
from core.row_segmentation import crop_region_from_source, load_sidecar
from benchmark.score_two_stage_against_ground_truth import _is_correct, _record_to_expected
from training.test_lora_checkpoint import ABSTAIN_STATUSES, _classify_mismatch

DEFAULT_GROUND_TRUTH_LOG = PROJECT_ROOT / "data" / "outputs" / "ground_truth_log.jsonl"
DEFAULT_LOG_FILE = PROJECT_ROOT / "data" / "outputs" / "prompt_sweep_log.jsonl"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "outputs" / "prompt_sweep_runs"

# config/prompts/ holds candidates for EVERY pipeline stage (classifier_*,
# extractor_*, loader_*, ...), not a per-stage subfolder - these defaults
# scope a --ocr-prompt-dir/--structuring-prompt-dir sweep to just this
# stage's own naming convention, so pointing at the shared folder
# directly doesn't also sweep every other stage's prompts (see
# _load_prompt_variants' docstring).
DEFAULT_OCR_PROMPT_GLOB = "ocr_stage1_*.txt"
DEFAULT_STRUCTURING_PROMPT_GLOB = "structuring_stage2_*.txt"

# Same defaults run_two_stage_extraction() itself uses - not exposed as
# CLI flags in V1, matching scripts/model_assessment.py's "isolated,
# simple" philosophy (see this module's docstring). Revisit if a real
# sweep needs to vary these too.
TIGHT_CROP_PADDING_PX = 20
UPSCALE_TARGET_HEIGHT = 160
UPSCALE_MAX_WIDTH = 4096


@dataclass
class Example:
    """One ground-truth-backed (page, row, column) to score every combo
    against - the row bbox and this column's own tight-crop range are
    resolved once up front (discover_examples), not re-derived per
    prompt variant."""
    sidecar_path: str
    source_image_path: str
    deskew_angle: float
    row_index: int
    row_bbox: list[int]
    column: str
    tight_crop_ranges: list[tuple[int, int]] | None
    expected: str
    status: str


@dataclass
class PromptVariant:
    """A candidate prompt for a stage - name is the filename for a
    swept variant, or a fixed label when that stage isn't being swept."""
    name: str
    text: str | None  # None for structuring's "built-in default template"


def _emit(event: str, **kwargs) -> None:
    """
    Machine-readable progress line for benchmark/prompt_sweep_gui.py
    (2026-07-29) - a single "@@PROGRESS <json>" line per event, printed
    ADDITIONALLY alongside (never replacing) the existing human-readable
    prints throughout this module, so CLI usage is completely unchanged.
    "@@PROGRESS " is a prefix no real prompt/model/path content is going
    to start a line with, so the GUI can filter these out of its visible
    log and parse them separately without any ambiguity. flush=True since
    the GUI's subprocess is launched with -u anyway, but explicit here
    too so this helper is correct even if a future caller isn't.
    """
    print("@@PROGRESS " + json.dumps({"event": event, **kwargs}, ensure_ascii=False), flush=True)


def _load_prompt_variants(prompt_dir: str | None, prompt_file: str | None,
                           fixed_label: str, fixed_text: str | None,
                           glob_pattern: str = "*.txt") -> list[PromptVariant]:
    """
    glob_pattern (2026-07-30, per Jon's direction after a real run swept
    every file in config/prompts/ - classifier_*, extractor_*, loader_*,
    the OTHER stage's own templates - not just this stage's candidates,
    since that directory holds prompts for every pipeline stage, not a
    per-stage subfolder): defaults to "*.txt" here, but main() below
    passes a stage-specific default (ocr_stage1_*.txt / structuring_
    stage2_*.txt, matching this project's existing naming convention -
    see docs/CODE_MAP.md's "Prompt & column-list files" section) so
    pointing --ocr-prompt-dir/--structuring-prompt-dir straight at the
    shared config/prompts/ folder only picks up that stage's own
    candidates. Override via --ocr-prompt-glob/--structuring-prompt-glob
    if a curated subfolder doesn't follow that naming convention.
    """
    if prompt_dir and prompt_file:
        raise ValueError("Pass at most one of --{dir} / --{file} for the same stage.")
    if prompt_dir:
        paths = sorted(Path(prompt_dir).glob(glob_pattern))
        if not paths:
            raise ValueError(
                f"No files matching {glob_pattern!r} found in {prompt_dir} - if this "
                f"folder's prompt files don't follow the ocr_stage1_*/structuring_stage2_* "
                f"naming convention, override with --ocr-prompt-glob/--structuring-prompt-glob.")
        return [PromptVariant(name=p.name, text=p.read_text(encoding="utf-8"))
                for p in paths]
    if prompt_file:
        path = Path(prompt_file)
        if not path.exists():
            raise ValueError(f"Prompt file not found: {path}")
        return [PromptVariant(name=path.name, text=path.read_text(encoding="utf-8"))]
    return [PromptVariant(name=fixed_label, text=fixed_text)]


def discover_examples(
    sidecar_paths: list[str] | None,
    columns: list[str] | None,
    ground_truth_log: str,
    max_rows_per_page: int | None = None,
) -> list[Example]:
    """
    Loads every requested (or, if none given, every distinct) sidecar
    referenced in ground_truth_log.jsonl, cross-references against the
    filtered ground-truth records, and returns one Example per (page,
    row, column) with the row bbox + this column's own tight-crop range
    already resolved via _resolve_column_field_mask - the same per-
    column, name-tagged mask run_two_stage_extraction reads, not the
    unrelated "__multi__" scratch mask (see that function's own
    docstring for why those aren't interchangeable).

    Raises ValueError up front (before any model loads) if a requested
    column has no saved, row-applied mask on some page - same fail-fast
    discipline run_two_stage_extraction already applies, so a sweep
    doesn't burn GPU time before discovering a masking gap.
    """
    gt_path = Path(ground_truth_log)
    if not gt_path.exists():
        raise ValueError(f"Ground-truth log not found: {gt_path}")
    records = []
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    all_sidecar_paths = sorted({r.get("sidecar_path") for r in records if r.get("sidecar_path")})
    target_sidecars = sidecar_paths if sidecar_paths else all_sidecar_paths
    if not target_sidecars:
        raise ValueError(f"No sidecar_path values found in {gt_path}")

    examples: list[Example] = []
    for sidecar_path in target_sidecars:
        page_records = [r for r in records if r.get("sidecar_path") == sidecar_path]
        if columns:
            page_records = [r for r in page_records if r.get("column") in columns]
        if not page_records:
            continue

        sidecar = load_sidecar(sidecar_path)
        source_image_path = sidecar["source_image_path"]
        deskew_angle = sidecar["deskew_angle"]
        rows_by_index = {row["index"]: row for row in sidecar["rows"]}
        sidecar_columns = sidecar.get("columns", {})

        page_columns = sorted({r["column"] for r in page_records})
        missing = [c for c in page_columns if not sidecar_columns.get(c, {}).get("mask_keep_ranges")]
        disabled = [c for c in page_columns
                    if c not in missing and not sidecar_columns[c].get("mask_apply_rows", True)]
        if missing or disabled:
            problems = []
            if missing:
                problems.append(f"no saved mask: {missing}")
            if disabled:
                problems.append(f"mask saved but \"apply to rows\" is off: {disabled}")
            raise ValueError(
                f"{sidecar_path}: every swept column needs its own saved, row-applied "
                f"mask - {'; '.join(problems)}. Mask each in row_segmentation_ui.py first."
            )

        tight_crop_by_column = {
            c: _resolve_column_field_mask(sidecar, sidecar_columns[c])[2] for c in page_columns
        }

        seen_rows: set[int] = set()
        for record in page_records:
            expected = _record_to_expected(record)
            if expected is None:
                continue
            row_index = record.get("row_index")
            if max_rows_per_page is not None:
                if row_index not in seen_rows and len(seen_rows) >= max_rows_per_page:
                    continue
                seen_rows.add(row_index)
            row = rows_by_index.get(row_index)
            if row is None:
                continue
            column = record["column"]
            examples.append(Example(
                sidecar_path=sidecar_path, source_image_path=source_image_path,
                deskew_angle=deskew_angle, row_index=row_index, row_bbox=row["bbox"],
                column=column, tight_crop_ranges=tight_crop_by_column[column],
                expected=expected, status=record.get("status", "unknown"),
            ))

    if not examples:
        raise ValueError("No ground-truth examples matched the given sidecar/column filters.")
    return examples


def run_sweep(
    ocr_model: str,
    structure_model: str,
    examples: list[Example],
    ocr_variants: list[PromptVariant],
    structuring_variants: list[PromptVariant],
    run_id: str,
) -> list[dict]:
    """
    Phase A: load the OCR model once, run every OCR-prompt variant
    across every example, cache each (variant, example) -> raw reading
    + runtime. Phase B: load the structuring model once, run every
    (ocr_variant, structuring_variant) combination across every
    example using Phase A's cached reading, parse, and score.

    Returns one result dict per (ocr_variant, structuring_variant)
    combo - the unit a row in the comparison table/history log
    represents.
    """
    ocr_config = load_model_config(ocr_model)
    ocr_loader_cls = LOADER_REGISTRY.get(ocr_config.loader_class)
    if ocr_loader_cls is None:
        raise ValueError(f"No loader registered for {ocr_config.loader_class!r}")

    # (ocr_variant.name, example_index) -> (raw_reading, seconds)
    readings: dict[tuple[str, int], tuple[str, float]] = {}
    ocr_loader = ocr_loader_cls(ocr_config)
    _emit("model_load_start", stage="ocr", model=ocr_model)
    try:
        ocr_loader.initialize_model_and_tokenizer()
        _emit("model_load_done", stage="ocr", model=ocr_model)
        for variant in ocr_variants:
            _emit("stage1_variant_start", ocr_prompt=variant.name)
            for i, ex in enumerate(examples):
                start = time.time()
                field_image = crop_region_from_source(
                    ex.source_image_path, ex.row_bbox, ex.deskew_angle,
                    tight_crop_keep_ranges=ex.tight_crop_ranges,
                    tight_crop_padding_px=TIGHT_CROP_PADDING_PX,
                    upscale_target_height=UPSCALE_TARGET_HEIGHT,
                    upscale_max_width=UPSCALE_MAX_WIDTH,
                )
                if field_image.mode != "RGB":
                    field_image = field_image.convert("RGB")
                try:
                    reading = ocr_loader._run_generate(field_image, variant.text or "")
                except Exception as e:
                    reading = f"[STAGE 1 ERROR: {e}]"
                readings[(variant.name, i)] = (reading, time.time() - start)
                print(f"[stage1 {variant.name}] row {ex.row_index} [{ex.column}]: {reading!r}")
                _emit("stage1_step", ocr_prompt=variant.name, i=i + 1, n=len(examples),
                      row_index=ex.row_index, column=ex.column)
            _emit("stage1_variant_done", ocr_prompt=variant.name)
    finally:
        _release_model(ocr_loader)

    struct_config = load_model_config(structure_model)
    struct_loader_cls = LOADER_REGISTRY.get(struct_config.loader_class)
    if struct_loader_cls is None:
        raise ValueError(f"No loader registered for {struct_config.loader_class!r}")

    combos: list[dict] = []
    struct_loader = struct_loader_cls(struct_config)
    _emit("model_load_start", stage="structuring", model=structure_model)
    try:
        struct_loader.initialize_model_and_tokenizer()
        _emit("model_load_done", stage="structuring", model=structure_model)
        for ocr_variant in ocr_variants:
            for struct_variant in structuring_variants:
                _emit("combo_start", ocr_prompt=ocr_variant.name, structuring_prompt=struct_variant.name)
                combo = _run_one_combo(
                    struct_loader, examples, ocr_variant, struct_variant, readings)
                combos.append(combo)
                _emit("combo_done", ocr_prompt=ocr_variant.name, structuring_prompt=struct_variant.name,
                      overall_correct=combo["overall_correct"], overall_total=combo["overall_total"],
                      abstained=combo["abstained"], wrong=combo["wrong"],
                      avg_total_seconds=combo["avg_total_seconds"])
    finally:
        _release_model(struct_loader)

    return combos


def _run_one_combo(
    struct_loader, examples: list[Example], ocr_variant: PromptVariant,
    struct_variant: PromptVariant, readings: dict[tuple[str, int], tuple[str, float]],
) -> dict:
    per_example: list[dict] = []
    correct = 0
    failure_modes: dict[str, int] = defaultdict(int)
    ocr_seconds_total = 0.0
    struct_seconds_total = 0.0

    for i, ex in enumerate(examples):
        raw_reading, ocr_seconds = readings[(ocr_variant.name, i)]
        start = time.time()
        field_image = crop_region_from_source(
            ex.source_image_path, ex.row_bbox, ex.deskew_angle,
            tight_crop_keep_ranges=ex.tight_crop_ranges,
            tight_crop_padding_px=TIGHT_CROP_PADDING_PX,
            upscale_target_height=UPSCALE_TARGET_HEIGHT,
            upscale_max_width=UPSCALE_MAX_WIDTH,
        )
        if field_image.mode != "RGB":
            field_image = field_image.convert("RGB")
        prompt = build_structuring_prompt(raw_reading, [ex.column], template_override=struct_variant.text)
        try:
            raw_output = struct_loader._run_generate(field_image, prompt)
            parsed = parse_row_output(raw_output, [ex.column])
            field = parsed.get(ex.column)
            predicted = field.value if field else ""
            predicted_confidence = field.confidence.value if field else ""
        except Exception as e:
            raw_output = f"[STAGE 2 ERROR: {e}]"
            predicted = ""
            predicted_confidence = ""
        struct_seconds = time.time() - start
        ocr_seconds_total += ocr_seconds
        struct_seconds_total += struct_seconds

        is_correct = _is_correct(predicted, ex.expected)
        failure_mode = None
        if is_correct:
            correct += 1
        else:
            failure_mode = _classify_mismatch(ex.status, predicted)
            failure_modes[failure_mode] += 1

        per_example.append({
            "sidecar_path": ex.sidecar_path, "row_index": ex.row_index, "column": ex.column,
            "status": ex.status, "expected": ex.expected, "predicted": predicted,
            "predicted_confidence": predicted_confidence, "correct": is_correct,
            "failure_mode": failure_mode, "ocr_raw_reading": raw_reading,
            "structuring_raw_output": raw_output, "ocr_seconds": round(ocr_seconds, 3),
            "structuring_seconds": round(struct_seconds, 3),
        })
        print(f"[stage2 {ocr_variant.name}+{struct_variant.name}] row {ex.row_index} "
              f"[{ex.column}]: {'OK' if is_correct else failure_mode} "
              f"predicted={predicted!r} expected={ex.expected!r}")
        _emit("stage2_step", ocr_prompt=ocr_variant.name, structuring_prompt=struct_variant.name,
              i=i + 1, n=len(examples), row_index=ex.row_index, column=ex.column)

    n = len(examples)
    honest_hedge = failure_modes.get("honest_hedge", 0)
    incorrectly_silent = failure_modes.get("incorrectly_silent", 0)
    confident_wrong = failure_modes.get("confident_wrong", 0)

    return {
        "ocr_prompt": ocr_variant.name, "structuring_prompt": struct_variant.name,
        "n": n, "overall_correct": correct, "overall_total": n,
        "failure_modes_total": {
            "honest_hedge": honest_hedge, "incorrectly_silent": incorrectly_silent,
            "confident_wrong": confident_wrong,
        },
        "abstained": honest_hedge + incorrectly_silent, "wrong": confident_wrong,
        "avg_ocr_seconds": round(ocr_seconds_total / n, 3) if n else 0.0,
        "avg_structuring_seconds": round(struct_seconds_total / n, 3) if n else 0.0,
        "avg_total_seconds": round((ocr_seconds_total + struct_seconds_total) / n, 3) if n else 0.0,
        "per_example": per_example,
    }


def print_comparison_table(combos: list[dict], ocr_model: str, structure_model: str) -> None:
    headers = ["OCR-Prompt", "Struct-Prompt", "N", "Correct", "Abstained", "Wrong", "Avg-Time"]
    rows = []
    for c in combos:
        acc = c["overall_correct"] / c["overall_total"] if c["overall_total"] else 0.0
        abst = c["abstained"] / c["overall_total"] if c["overall_total"] else 0.0
        wrong = c["wrong"] / c["overall_total"] if c["overall_total"] else 0.0
        rows.append([
            c["ocr_prompt"], c["structuring_prompt"], str(c["n"]),
            f"{acc:.0%}", f"{abst:.0%}", f"{wrong:.0%}", f"{c['avg_total_seconds']:.1f}s",
        ])
    widths = [max(len(headers[i]), max((len(r[i]) for r in rows), default=0)) for i in range(len(headers))]

    def print_row(cells):
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)))

    print(f"\nSweep results (OCR model={ocr_model}, structuring model={structure_model}):\n")
    print_row(headers)
    print_row(["-" * w for w in widths])
    for r in rows:
        print_row(r)
    print("\nAbstained = honest '?' hedge or safe silence on a real field (not counted against "
          "the prompt). Wrong = confident, specific, false content - the real hallucination-risk rate.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ocr-model", type=str, required=True, help="Stage 1 model profile.")
    parser.add_argument("--structure-model", type=str, required=True, help="Stage 2 model profile.")
    parser.add_argument("--ocr-prompt-dir", type=str, default=None,
                         help="Sweep every file matching --ocr-prompt-glob in this dir as a "
                              "stage-1 OCR prompt candidate.")
    parser.add_argument("--ocr-prompt-file", type=str, default=None,
                         help="Use a single fixed stage-1 prompt file (mutually exclusive with --ocr-prompt-dir).")
    parser.add_argument("--ocr-prompt-glob", type=str, default=DEFAULT_OCR_PROMPT_GLOB,
                         help=f"Glob pattern applied within --ocr-prompt-dir (default: "
                              f"{DEFAULT_OCR_PROMPT_GLOB!r}, matching this project's naming "
                              f"convention - override if a curated subfolder doesn't follow it).")
    parser.add_argument("--structuring-prompt-dir", type=str, default=None,
                         help="Sweep every file matching --structuring-prompt-glob in this dir "
                              "as a stage-2 structuring prompt candidate.")
    parser.add_argument("--structuring-prompt-file", type=str, default=None,
                         help="Use a single fixed stage-2 template file (mutually exclusive with --structuring-prompt-dir).")
    parser.add_argument("--structuring-prompt-glob", type=str, default=DEFAULT_STRUCTURING_PROMPT_GLOB,
                         help=f"Glob pattern applied within --structuring-prompt-dir (default: "
                              f"{DEFAULT_STRUCTURING_PROMPT_GLOB!r}).")
    parser.add_argument("--sidecar-path", type=str, action="append", default=None,
                         help="Restrict to this page (repeatable). Default: every page in the ground-truth log.")
    parser.add_argument("--columns", type=str, action="append", default=None,
                         help="Restrict to this column (repeatable). Default: every column in the ground-truth log.")
    parser.add_argument("--ground-truth-log", type=str, default=str(DEFAULT_GROUND_TRUTH_LOG))
    parser.add_argument("--max-rows-per-page", type=int, default=None,
                         help="Cap rows scored per page (smoke-test / quick-iteration knob).")
    parser.add_argument("--run-id", type=str, default=None, help="Default: a timestamp.")
    parser.add_argument("--log-file", type=str, default=str(DEFAULT_LOG_FILE),
                         help="Append-only JSONL history log. Pass '' to skip logging.")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR),
                         help="Per-combo detailed per-example JSON goes here.")
    args = parser.parse_args()

    if not args.ocr_prompt_dir and not args.structuring_prompt_dir:
        msg = ("give --ocr-prompt-dir and/or --structuring-prompt-dir - at least one "
               "must be swept, otherwise there's nothing to compare.")
        print(f"ERROR: {msg}")
        _emit("error", message=msg)
        sys.exit(1)

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    try:
        examples = discover_examples(
            args.sidecar_path, args.columns, args.ground_truth_log, args.max_rows_per_page)
        ocr_variants = _load_prompt_variants(
            args.ocr_prompt_dir, args.ocr_prompt_file, "(fixed/empty)", "",
            glob_pattern=args.ocr_prompt_glob)
        structuring_variants = _load_prompt_variants(
            args.structuring_prompt_dir, args.structuring_prompt_file, "(built-in default)", None,
            glob_pattern=args.structuring_prompt_glob)
    except ValueError as e:
        print(f"ERROR: {e}")
        _emit("error", message=str(e))
        sys.exit(1)

    print(f"Run {run_id}: {len(examples)} example(s), {len(ocr_variants)} OCR-prompt variant(s) "
          f"x {len(structuring_variants)} structuring-prompt variant(s) = "
          f"{len(ocr_variants) * len(structuring_variants)} combo(s).")
    _emit("plan", run_id=run_id, ocr_model=args.ocr_model, structuring_model=args.structure_model,
          ocr_variants=[v.name for v in ocr_variants],
          structuring_variants=[v.name for v in structuring_variants],
          n_examples=len(examples))

    combos = run_sweep(args.ocr_model, args.structure_model, examples, ocr_variants, structuring_variants, run_id)

    print_comparison_table(combos, args.ocr_model, args.structure_model)

    out_dir = Path(args.out_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    log_path = Path(args.log_file) if args.log_file else None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    for c in combos:
        detail_name = f"{c['ocr_prompt']}__{c['structuring_prompt']}.json".replace("/", "_")
        detail_path = out_dir / detail_name
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump(c["per_example"], f, indent=2, ensure_ascii=False)

        if log_path:
            record = {
                "run_id": run_id, "timestamp": timestamp,
                "ocr_model": args.ocr_model, "ocr_prompt": c["ocr_prompt"],
                "structuring_model": args.structure_model, "structuring_prompt": c["structuring_prompt"],
                "sidecar_paths": sorted({e.sidecar_path for e in examples}),
                "columns": sorted({e.column for e in examples}),
                "n": c["n"], "overall_correct": c["overall_correct"], "overall_total": c["overall_total"],
                "failure_modes_total": c["failure_modes_total"],
                "avg_ocr_seconds": c["avg_ocr_seconds"], "avg_structuring_seconds": c["avg_structuring_seconds"],
                "avg_total_seconds": c["avg_total_seconds"],
                "detail_path": str(detail_path),
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nDetail JSON written to {out_dir}/")
    if log_path:
        print(f"Appended {len(combos)} record(s) to {log_path}")
    _emit("run_done", run_id=run_id, log_file=str(log_path) if log_path else None)


if __name__ == "__main__":
    main()
