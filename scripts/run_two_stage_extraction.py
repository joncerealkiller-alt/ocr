"""
Two-stage extraction CLI: stage 1 runs a fixed-task raw-OCR engine
(e.g. Chandra) that can't follow our column schema natively; stage 2
runs an instruction-following model (e.g. Qwen3-VL-4B, Gemma) given
BOTH the row image and stage 1's raw reading, structuring the result
into the actual census columns.

Built 2026-07-13 per Jon's direction: "even if we have to use a
non-prompt OCR engine... use a VLM or LLM to combine them after
extraction."

Usage:
    python scripts/run_two_stage_extraction.py <sidecar.json> <columns.txt> \\
        --ocr-model chandra --structure-model qwen3vl4b [--max-rows N]

Outputs to --out (default: same directory as the sidecar):
    <name>_twostage_extraction.csv
    <name>_twostage_extraction.json   (includes raw_output showing the
                                        stage-2 model's actual response;
                                        stage 1's raw OCR reading is
                                        embedded in the structuring
                                        prompt sent to stage 2, not
                                        saved separately here)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Moved into scripts/ (2026-07-25) - one directory deeper than repo
# root, so repo root must be put back on sys.path before the `core.*`
# imports below will resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.row_extraction import run_two_stage_extraction, save_results_csv, save_results_json
from core.debug_dump import DebugModelInputRecorder


def _load_field_list(path: Path, label: str) -> list[str]:
    if not path.exists():
        print(f"ERROR: {label} file not found: {path}")
        sys.exit(1)
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        print(f"ERROR: {path} contained no {label} names")
        sys.exit(1)
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sidecar_path", type=str, help="Path to a segmentation sidecar JSON")
    parser.add_argument("columns_path", type=str,
                         help="Text file with column names, one per line, in "
                              "left-to-right form order")
    parser.add_argument("--ocr-model", type=str, required=True,
                         help="Stage 1 model profile - the raw OCR engine (e.g. chandra).")
    parser.add_argument("--structure-model", type=str, required=True,
                         help="Stage 2 model profile - the instruction-following "
                              "model that structures stage 1's output (e.g. qwen3vl4b).")
    parser.add_argument("--max-rows", type=int, default=None,
                         help="Only process the first N rows.")
    parser.add_argument("--ocr-prompt-file", type=str, default=None,
                         help="Prompt file for stage 1 (OCR model). Default: none "
                              "(empty prompt - a real finding, 2026-07-13: even a "
                              "general VLM given NO prompt at all organized fields "
                              "more usefully for stage 2 than a fixed-task OCR "
                              "engine's raw markdown). Only affects instruction-"
                              "following stage-1 models - fixed-task engines like "
                              "chandra ignore this regardless. See "
                              "config/prompts/ocr_stage1_*.txt for starting options.")
    parser.add_argument("--structuring-prompt-file", type=str, default=None,
                         help="Template file for stage 2's structuring prompt. "
                              "Default: none (uses the built-in default template, "
                              "identical to config/prompts/structuring_stage2_"
                              "default.txt). Must be a str.format()-style template "
                              "using {raw_ocr_text}, {columns_str}, {example_lines} "
                              "(and optionally {num_columns}, {plural}) - copy "
                              "structuring_stage2_default.txt as a starting point. "
                              "Added 2026-07-22 so stage 2's instructional wording "
                              "can be iterated on without editing core/"
                              "row_extraction.py.")
    parser.add_argument("--tight-crop-padding-px", type=int, default=20,
                         help="Fixed pixel padding around each field's own kept "
                              "mask range for stage 1's per-column crop (default: "
                              "20). Ignored if --tight-crop-padding-pct is given. "
                              "Same meaning as run_row_extraction.py's identically-"
                              "named flag.")
    parser.add_argument("--tight-crop-padding-pct", type=float, default=None,
                         help="Padding as a fraction of each field's own kept "
                              "range width, instead of a fixed pixel margin. "
                              "Overrides --tight-crop-padding-px if given.")
    parser.add_argument("--stage1-upscale-target-height", type=int, default=160,
                         help="Upscales each PER-COLUMN field crop (aspect-"
                              "preserving, LANCZOS) so its height reaches at least "
                              "this many pixels before stage 1 (OCR) sees it. "
                              "Default: 160 - stage 1 now runs one OCR call per "
                              "selected column against that column's own tightly-"
                              "cropped image (2026-07-24 fix - it previously ran "
                              "one call against the full, mostly-blank row), so it "
                              "needs the same upscale-by-default treatment as "
                              "single-column extraction (real crops measured as "
                              "small as 79x36px). Pass 0 to disable.")
    parser.add_argument("--stage1-upscale-max-width", type=int, default=4096,
                         help="Caps stage 1's upscaled field crop width (default: "
                              "4096). Only relevant if --stage1-upscale-target-"
                              "height is set.")
    parser.add_argument("--stage2-upscale-target-height", type=int, default=160,
                         help="Upscales each PER-COLUMN field crop stage 2 sees "
                              "(aspect-preserving, LANCZOS) so its height reaches "
                              "at least this many pixels. Default: 160, matching "
                              "stage 1 - stage 2 now runs one structuring call per "
                              "selected column against that SAME column's own "
                              "tightly-cropped image stage 1 used (2026-07-25 fix: "
                              "previously one call against all selected columns "
                              "unioned into one image, which still left inter-"
                              "column gutter whitespace for the model to sort "
                              "through). Pass 0 to disable.")
    parser.add_argument("--stage2-upscale-max-width", type=int, default=4096,
                         help="Caps stage 2's upscaled field crop width (default: "
                              "4096). Only relevant if --stage2-upscale-target-"
                              "height is set.")
    parser.add_argument("--out", type=str, default=None,
                         help="Output directory. Default: same directory as the sidecar.")
    parser.add_argument("--debug-model-inputs", action="store_true",
                         help="Save the exact image crop, prompt, and raw output for "
                              "every stage-1/stage-2 model call to "
                              "data/debug_model_inputs/<run_id>/ - see "
                              "scripts/run_row_extraction.py --help for the full "
                              "explanation. Off by default; has no effect on "
                              "extraction results when omitted.")
    parser.add_argument("--debug-dir", type=str, default="data/debug_model_inputs",
                         help="Base directory for --debug-model-inputs output "
                              "(default: data/debug_model_inputs/).")
    parser.add_argument("--ocr-checkpoint", type=str, default=None,
                         help="Optional path to a saved LoRA adapter dir (data/outputs/"
                              "<model>_lora_checkpoints/epoch_N/, see training/train_lora.py) "
                              "applied on top of --ocr-model. Must have been trained from "
                              "--ocr-model specifically - loading a checkpoint trained "
                              "against a different base model will fail or silently "
                              "produce garbage.")
    parser.add_argument("--structure-checkpoint", type=str, default=None,
                         help="Same as --ocr-checkpoint, applied to --structure-model instead.")
    args = parser.parse_args()

    sidecar_path = Path(args.sidecar_path)
    if not sidecar_path.exists():
        print(f"ERROR: sidecar not found: {sidecar_path}")
        sys.exit(1)

    column_names = _load_field_list(Path(args.columns_path), "column")

    ocr_prompt = ""
    if args.ocr_prompt_file:
        ocr_prompt_path = Path(args.ocr_prompt_file)
        if not ocr_prompt_path.exists():
            print(f"ERROR: OCR prompt file not found: {ocr_prompt_path}")
            sys.exit(1)
        ocr_prompt = ocr_prompt_path.read_text(encoding="utf-8").strip()

    structuring_prompt_template = None
    if args.structuring_prompt_file:
        structuring_prompt_path = Path(args.structuring_prompt_file)
        if not structuring_prompt_path.exists():
            print(f"ERROR: structuring prompt file not found: {structuring_prompt_path}")
            sys.exit(1)
        structuring_prompt_template = structuring_prompt_path.read_text(encoding="utf-8")

    out_dir = Path(args.out) if args.out else sidecar_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    name = sidecar_path.stem.replace("_sidecar", "")

    debug_recorder = DebugModelInputRecorder(
        enabled=args.debug_model_inputs, base_dir=args.debug_dir,
    )
    if debug_recorder.enabled:
        print(f"Debug model-input capture: ON -> {debug_recorder.run_dir}")

    print(f"Sidecar: {sidecar_path}")
    print(f"Columns ({len(column_names)}): {', '.join(column_names)}")
    print(f"Stage 1 (OCR): {args.ocr_model}"
          + (f"  [checkpoint: {args.ocr_checkpoint}]" if args.ocr_checkpoint else ""))
    print(f"Stage 1 prompt: {args.ocr_prompt_file or '(none - empty)'}")
    print(f"Stage 2 (structure): {args.structure_model}"
          + (f"  [checkpoint: {args.structure_checkpoint}]" if args.structure_checkpoint else ""))
    print(f"Stage 2 prompt template: {args.structuring_prompt_file or '(none - built-in default)'}")
    print(f"Stage 1 upscale target height (per-field crop): {args.stage1_upscale_target_height or '(off)'}")
    print(f"Stage 2 upscale target height (per-field crop): {args.stage2_upscale_target_height or '(off)'}")
    print(f"{'='*60}")

    results = run_two_stage_extraction(
        str(sidecar_path), args.ocr_model, args.structure_model,
        column_names, max_rows=args.max_rows, ocr_prompt=ocr_prompt,
        structuring_prompt_template=structuring_prompt_template,
        stage1_upscale_target_height=args.stage1_upscale_target_height or None,
        stage1_upscale_max_width=args.stage1_upscale_max_width,
        stage2_upscale_target_height=args.stage2_upscale_target_height or None,
        stage2_upscale_max_width=args.stage2_upscale_max_width,
        tight_crop_padding_px=args.tight_crop_padding_px,
        tight_crop_padding_pct=args.tight_crop_padding_pct,
        debug_recorder=debug_recorder,
        ocr_checkpoint=args.ocr_checkpoint,
        structure_checkpoint=args.structure_checkpoint,
    )

    csv_path = out_dir / f"{name}_twostage_extraction.csv"
    json_path = out_dir / f"{name}_twostage_extraction.json"
    save_results_csv(results, csv_path, column_names)
    save_results_json(results, json_path)

    passed = sum(1 for r in results if r.schema_pass)
    print(f"{'='*60}")
    print(f"Done: {len(results)} rows processed, {passed}/{len(results)} fully complete")
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
