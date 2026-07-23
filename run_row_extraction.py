"""
Row-level extraction CLI - the OCR-integration step that consumes a
segmentation sidecar JSON (from test_row_segmentation.py or
row_segmentation_ui.py) and runs structured extraction on each row.

Two modes:

1. SINGLE-COLUMN (2026-07-22, matches row_segmentation_ui.py's
   mask -> Next -> mask -> Next persistent-sidecar workflow): extracts
   just ONE column, using that column's own stored mask from the
   sidecar's columns[name], and writes results straight into
   sidecar.columns[name]["results"] (merged, not overwritten - see
   core.row_segmentation.update_sidecar). This is the default when no
   columns.txt is given - the sidecar's own active_column is used.

       python run_row_extraction.py <sidecar.json> --model qwen3vl2b \
           [--column NAME] [--max-rows N] [--no-mark-done]

2. LEGACY MULTI-COLUMN (original behavior, unchanged): all columns
   extracted together in one prompt per row, against the sidecar's
   OLD single global mask_keep_ranges field (not per-column state) -
   still useful for a whole-row pass where isolating one column isn't
   needed. Triggered by passing a columns.txt.

       python run_row_extraction.py <sidecar.json> <columns.txt> \
           --model qwen3vl2b [--max-rows N] [--header-fields <fields.txt>]

Example columns.txt for a standard census form:
    Name
    Age
    Sex
    Relationship to Head
    Birthplace
    Occupation

--header-fields (2026-07-13, Jon's direction, multi-column mode only):
extracts the page's own district/sub-district/province/enumerator
metadata block - "required keywords for the context of the data
following" - using the SAME model load as the row pass. See
config/columns/census_page_header.txt for a starting template. Omit
this flag to skip header extraction entirely (row-only run).

Outputs to --out (default: same directory as the sidecar):
    <name>_extraction.csv     - one row per census line, one column pair
                                 (value/confidence) per field extracted
                                 this run
    <name>_extraction.json    - full structured row results including
                                 raw model output, for debugging
    <name>_header.json        - page header/context fields, if
                                 --header-fields was given (multi-column
                                 mode only)
    (single-column mode also writes results into the sidecar itself -
    no separate flag needed, that's the point of the mode)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from core.row_extraction import (
    run_row_extraction, run_single_column_extraction, save_results_csv, save_results_json,
)
from core.row_segmentation import load_sidecar


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
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sidecar_path", type=str, help="Path to a segmentation sidecar JSON")
    parser.add_argument("columns_path", type=str, nargs="?", default=None,
                         help="Text file with column names, one per line, in "
                              "left-to-right form order. OMIT this to run in "
                              "single-column mode against the sidecar's own "
                              "active_column (or --column).")
    parser.add_argument("--column", type=str, default=None,
                         help="Single-column mode only: column name to extract. "
                              "Defaults to the sidecar's active_column. Ignored "
                              "if columns_path is given (multi-column mode).")
    parser.add_argument("--no-mark-done", action="store_true",
                         help="Single-column mode only: don't flip the column's "
                              "status to 'done' after this run (e.g. a quick "
                              "--max-rows test pass you don't want counted as final).")
    parser.add_argument("--tight-crop-padding-px", type=int, default=20,
                         help="Single-column mode only: fixed pixel padding around "
                              "the kept mask range for the image actually sent to "
                              "the model (default: 20). Ignored if --tight-crop-"
                              "padding-pct is given.")
    parser.add_argument("--tight-crop-padding-pct", type=float, default=None,
                         help="Single-column mode only: padding as a fraction of "
                              "the kept range's own width (e.g. 0.1 = 10%%), instead "
                              "of a fixed pixel margin - useful when column widths "
                              "vary a lot across the page. Overrides --tight-crop-"
                              "padding-px if given.")
    parser.add_argument("--upscale-target-height", type=int, default=160,
                         help="Single-column mode only: upscales the crop (aspect-"
                              "preserving, LANCZOS) so its height reaches at least "
                              "this many pixels before sending it to the model. "
                              "Default: 160 - real evidence found tightly-cropped "
                              "column images as small as 79x36px, not enough "
                              "resolution for most vision encoders. Pass 0 to "
                              "disable and reproduce exact pre-2026-07-23 behavior "
                              "for a controlled comparison.")
    parser.add_argument("--upscale-max-width", type=int, default=4096,
                         help="Caps the upscaled image's width (default: 4096) - "
                              "prevents a very wide, very short mask from producing "
                              "an absurdly large image that most model processors "
                              "would just resize back down internally anyway.")
    parser.add_argument("--model", type=str, required=True,
                         help="Model profile name (e.g. qwen3vl2b) - must exist in "
                              "config/models/")
    parser.add_argument("--max-rows", type=int, default=None,
                         help="Only process the first N rows - useful for a quick "
                              "test before committing to a full page.")
    parser.add_argument("--header-fields", type=str, default=None,
                         help="Multi-column mode only. Text file with page-header "
                              "field names, one per line (e.g. Province, District, "
                              "Sub-District Number, Enumerator). If given, extracts "
                              "the header block (0,0,width,table_top) using the "
                              "same model load as the row pass.")
    parser.add_argument("--out", type=str, default=None,
                         help="Output directory. Default: same directory as the sidecar.")
    args = parser.parse_args()

    sidecar_path = Path(args.sidecar_path)
    if not sidecar_path.exists():
        print(f"ERROR: sidecar not found: {sidecar_path}")
        sys.exit(1)

    out_dir = Path(args.out) if args.out else sidecar_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    name = sidecar_path.stem.replace("_sidecar", "")

    if args.columns_path is None:
        # -- single-column mode -------------------------------------------
        sidecar = load_sidecar(str(sidecar_path))
        column_name = args.column or sidecar.get("active_column")
        if column_name is None:
            print("ERROR: no --column given and sidecar has no active_column set "
                  "(mask a column in row_segmentation_ui.py first, or pass "
                  "--column NAME, or pass a columns.txt for legacy multi-column mode).")
            sys.exit(1)

        print(f"Sidecar: {sidecar_path}")
        print("Mode: single-column")
        print(f"Column: {column_name}")
        print(f"Model: {args.model}")
        print(f"{'='*60}")

        results = run_single_column_extraction(
            str(sidecar_path), args.model, column_name=column_name,
            max_rows=args.max_rows, mark_done=not args.no_mark_done,
            tight_crop_padding_px=args.tight_crop_padding_px,
            tight_crop_padding_pct=args.tight_crop_padding_pct,
            upscale_target_height=args.upscale_target_height or None,
            upscale_max_width=args.upscale_max_width,
        )

        csv_path = out_dir / f"{name}_{column_name}_extraction.csv"
        json_path = out_dir / f"{name}_{column_name}_extraction.json"
        save_results_csv(results, csv_path, [column_name])
        save_results_json(results, json_path)

        passed = sum(1 for r in results if r.schema_pass)
        print(f"{'='*60}")
        print(f"Done: {len(results)} rows processed, {passed}/{len(results)} fully complete")
        print(f"Results written into sidecar: {sidecar_path} (columns.{column_name}.results)")
        print(f"CSV:  {csv_path}")
        print(f"JSON: {json_path}")
        if not args.no_mark_done:
            updated = load_sidecar(str(sidecar_path))
            next_active = updated.get("active_column")
            progress = updated.get("progress", {})
            print(f"Column marked done. Progress: {progress.get('completed', '?')}/"
                  f"{progress.get('total', '?')}. Next active column: {next_active}")
        return

    # -- legacy multi-column mode ------------------------------------------
    column_names = _load_field_list(Path(args.columns_path), "column")
    header_field_names = (
        _load_field_list(Path(args.header_fields), "header field")
        if args.header_fields else None
    )

    print(f"Sidecar: {sidecar_path}")
    print("Mode: multi-column (legacy)")
    print(f"Columns ({len(column_names)}): {', '.join(column_names)}")
    if header_field_names:
        print(f"Header fields ({len(header_field_names)}): {', '.join(header_field_names)}")
    print(f"Model: {args.model}")
    print(f"{'='*60}")

    header_result, results = run_row_extraction(
        str(sidecar_path), args.model, column_names,
        max_rows=args.max_rows, header_field_names=header_field_names,
    )

    csv_path = out_dir / f"{name}_extraction.csv"
    json_path = out_dir / f"{name}_extraction.json"
    save_results_csv(results, csv_path, column_names)
    save_results_json(results, json_path)

    if header_result:
        header_path = out_dir / f"{name}_header.json"
        with open(header_path, "w", encoding="utf-8") as f:
            json.dump(header_result.model_dump(), f, indent=2, default=str)
        print(f"Header JSON: {header_path}")

    passed = sum(1 for r in results if r.schema_pass)
    print(f"{'='*60}")
    print(f"Done: {len(results)} rows processed, {passed}/{len(results)} fully complete")
    if header_result:
        print(f"Header: {'complete' if header_result.schema_pass else 'incomplete'}")
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
