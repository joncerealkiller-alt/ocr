"""
Standalone test harness for core/row_segmentation.py - deliberately has
NO model/OCR dependency. Per the agreed build order (2026-07-12): prove
the segmentation layer works on real census pages before adding any
model-call complexity on top.

Usage:
    python test_row_segmentation.py <image_path> [options]

Normal output, written to --out (default: data/outputs/row_segmentation/):
    <name>_debug_overlay.png   - full page with detected bands drawn on it
    <name>_sidecar.json        - deskew angle, table/header/row bounding
                                  boxes - the coordinates a downstream OCR
                                  stage needs to crop each row from the
                                  ORIGINAL source image in memory (see
                                  core.row_segmentation.crop_region_from_
                                  source). Primary output as of 2026-07-13
                                  (Jon's suggestion, schema per his spec)
                                  - individual row PNGs are no longer
                                  saved by default.
    <name>_report.txt          - warnings/diagnostics from sanity checks

Pass --debug-crops to ALSO save individual row PNGs for direct visual
inspection - useful while first tuning a new document type.

For genuinely interactive deskew/bounds adjustment (nudge buttons, live
preview), use row_segmentation_ui.py instead - this CLI script is for
non-interactive/batch use once settings are already confirmed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

from core.row_segmentation import (
    segment_rows, segment_rows_periodic, build_sidecar, save_sidecar,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_path", type=str, help="Path to the census/table image to test")
    parser.add_argument("--header-rows", type=int, default=1,
                         help="Number of detected bands (from the top) to treat as "
                              "the header block. Default: 1. Set to 0 to disable.")
    parser.add_argument("--out", type=str, default="data/outputs/row_segmentation",
                         help="Output directory for debug overlay, sidecar JSON, and report.")
    parser.add_argument("--mode", type=str, choices=["detect", "periodic"], default="detect",
                         help="'detect' (default): general-purpose blind row detection. "
                              "'periodic': for document types with a KNOWN fixed row count "
                              "and near-uniform spacing (e.g. standard census forms).")
    parser.add_argument("--row-count", type=int, default=50,
                         help="[periodic mode only] Known number of data rows. Default: 50.")
    parser.add_argument("--table-top", type=int, default=None,
                         help="[periodic mode only] Pixel y-coordinate where the table "
                              "body starts. If omitted, estimated automatically.")
    parser.add_argument("--table-bottom", type=int, default=None,
                         help="[periodic mode only] Pixel y-coordinate where the table "
                              "body ends.")
    parser.add_argument("--metadata-bottom", type=int, default=None,
                         help="[periodic mode only] Pixel y-coordinate splitting page "
                              "metadata [0..this] from column headings [this..table-top]. "
                              "Required for --header-rows > 0 to actually capture the "
                              "real printed column headers rather than sacrificing a real "
                              "data row (fixed 2026-07-13 - see core/row_segmentation.py).")
    parser.add_argument("--table-left", type=int, default=None,
                         help="[periodic mode only] Pixel x-coordinate of the table's "
                              "left edge. Default: full image width (0).")
    parser.add_argument("--table-right", type=int, default=None,
                         help="[periodic mode only] Pixel x-coordinate of the table's "
                              "right edge. Default: full image width.")
    parser.add_argument("--deskew-angle", type=float, default=None,
                         help="Use this EXACT deskew angle (degrees) instead of "
                              "auto-estimating. Read off row_segmentation_ui.py's "
                              "preview once you've confirmed it visually.")
    parser.add_argument("--search-radius-ratio", type=float, default=0.3,
                         help="[periodic mode only] Local refinement window size as a "
                              "fraction of the expected row height. Default: 0.3.")
    parser.add_argument("--padding", type=int, default=4,
                         help="Fixed-pixel row-crop padding (oversample slightly beyond "
                              "the detected line so descenders/ascenders aren't clipped). "
                              "Default: 4. Ignored if --padding-pct is set.")
    parser.add_argument("--padding-pct", type=float, default=None,
                         help="Row-crop padding as a fraction of each row's own height "
                              "(e.g. 0.08 = 8%%) - overrides --padding when set. Values "
                              "above ~0.20 risk reading into the adjacent row.")
    parser.add_argument("--debug-crops", action="store_true",
                         help="ALSO save individual row PNGs. Off by default.")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"ERROR: image not found: {image_path}")
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = image_path.stem

    print(f"Loading {image_path} ...")
    image = Image.open(image_path)

    if args.mode == "periodic":
        print(f"Running periodic-anchor segmentation (row_count={args.row_count}, "
              f"table_top={args.table_top}, table_bottom={args.table_bottom}, "
              f"table_left={args.table_left}, table_right={args.table_right}, "
              f"deskew_angle={args.deskew_angle}) - no OCR, no model calls ...")
        result, row_crops, header_crop, debug_overlay = segment_rows_periodic(
            image,
            row_count=args.row_count,
            table_top=args.table_top,
            table_bottom=args.table_bottom,
            table_left=args.table_left,
            table_right=args.table_right,
            search_radius_ratio=args.search_radius_ratio,
            header_row_count=args.header_rows,
            deskew_angle=args.deskew_angle,
            metadata_bottom=args.metadata_bottom,
            padding=args.padding,
            padding_pct=args.padding_pct,
        )
        parameters = {
            "row_count": args.row_count, "table_top": args.table_top,
            "table_bottom": args.table_bottom, "table_left": args.table_left,
            "table_right": args.table_right,
            "search_radius_ratio": args.search_radius_ratio,
            "header_row_count": args.header_rows,
        }
        sidecar_table_top, sidecar_table_bottom = args.table_top, args.table_bottom
    else:
        print("Running deskew + row detection (no OCR, no model calls) ...")
        result, row_crops, header_crop, debug_overlay = segment_rows(
            image, header_row_count=args.header_rows,
            padding=args.padding, padding_pct=args.padding_pct,
        )
        parameters = {"header_row_count": args.header_rows}
        sidecar_table_top, sidecar_table_bottom = None, None

    overlay_path = out_dir / f"{name}_debug_overlay.png"
    debug_overlay.save(overlay_path)
    print(f"\nDebug overlay saved: {overlay_path}")
    print("  -> LOOK AT THIS FIRST. Red/green = kept rows, blue = header, "
          "purple = table left/right bounds, gray+X = dropped as noise.")

    sidecar = build_sidecar(
        result, source_image_path=str(image_path.resolve()),
        mode=args.mode, parameters=parameters,
        table_top=sidecar_table_top, table_bottom=sidecar_table_bottom,
        x0=args.table_left, x1=args.table_right,
        padding=args.padding, padding_pct=args.padding_pct,
    )
    sidecar_path = out_dir / f"{name}_sidecar.json"
    save_sidecar(sidecar, sidecar_path)
    print(f"\nSidecar JSON saved: {sidecar_path}")
    print(f"  -> {len(sidecar['rows'])} row bbox(es), deskew_angle="
          f"{sidecar['deskew_angle']:.2f} degrees. A downstream OCR "
          f"stage should load the ORIGINAL image and use "
          f"core.row_segmentation.crop_region_from_source() with these "
          f"coordinates, not crop the working copy directly.")

    if args.debug_crops:
        for i, crop in enumerate(row_crops, start=1):
            crop.save(out_dir / f"{name}_row_{i:03d}.png")
        print(f"\n--debug-crops: {len(row_crops)} individual row PNGs also saved to {out_dir}/")

    report_path = out_dir / f"{name}_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Row segmentation report for {image_path}\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(f"Detected {len(result.bands)} data row(s)")
        if result.header_band:
            f.write(f" + 1 header region\n")
        else:
            f.write(f", no header region used\n")
        f.write(f"Dropped {len(result.dropped_bands)} band(s) as implausible noise\n\n")
        f.write("Warnings / diagnostics:\n")
        for w in result.warnings:
            f.write(f"  - {w}\n")
        if not result.warnings:
            f.write("  (none)\n")
    print(f"\nReport saved: {report_path}")

    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"  Rows detected: {len(result.bands)}")
    print(f"  Bands dropped as noise: {len(result.dropped_bands)}")
    print(f"  Warnings raised: {len(result.warnings)}")
    print(f"{'=' * 60}")
    print("\nNo OCR was run - this only tests row detection. Review the debug "
          "overlay before trusting the sidecar coordinates are usable.")


if __name__ == "__main__":
    main()
