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
    python run_two_stage_extraction.py <sidecar.json> <columns.txt> \\
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

from core.row_extraction import run_two_stage_extraction, save_results_csv, save_results_json


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
    parser.add_argument("--out", type=str, default=None,
                         help="Output directory. Default: same directory as the sidecar.")
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

    out_dir = Path(args.out) if args.out else sidecar_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    name = sidecar_path.stem.replace("_sidecar", "")

    print(f"Sidecar: {sidecar_path}")
    print(f"Columns ({len(column_names)}): {', '.join(column_names)}")
    print(f"Stage 1 (OCR): {args.ocr_model}")
    print(f"Stage 1 prompt: {args.ocr_prompt_file or '(none - empty)'}")
    print(f"Stage 2 (structure): {args.structure_model}")
    print(f"{'='*60}")

    results = run_two_stage_extraction(
        str(sidecar_path), args.ocr_model, args.structure_model,
        column_names, max_rows=args.max_rows, ocr_prompt=ocr_prompt,
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
