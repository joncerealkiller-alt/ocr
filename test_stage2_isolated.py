"""
Isolated stage-2 test - built 2026-07-20 per Jon's direction, after a
real, deterministic negative result: an explicit new instruction was
added to build_structuring_prompt() telling it what to do when stage
1's reading is empty/garbled ("look at the image directly; if you
still can't determine a value, output ? / unclear - never guess"), and
a real two-stage re-run showed the model still guessing plausible-
sounding words with no visible connection to the actual handwriting
(row 5: "Donor" where the real value is "Domestic"). Since do_sample:
false confirmed deterministic decoding, that wasn't noise - the new
instruction genuinely didn't change the behavior it targeted.

The open question that result couldn't answer: is stage 2 looking at
the image at all when stage 1 gives it nothing, and just guessing
badly anyway - or is it not really using the image in this failure
case, guessing from column-name association alone ("Relationship to
Head" -> some plausible relationship word) with the image contributing
nothing? A real two-stage run can't isolate this, because stage 1's
OWN reading is itself a variable - you can't tell whether a bad final
answer came from a bad stage-1 reading, stage 2 ignoring a good one, or
stage 2 ignoring the image on top of a bad one.

This script removes stage 1 entirely. It builds the REAL row crop
using the REAL mask logic (crop_region_from_source + compute_
exclude_ranges, mirroring core.row_extraction._compute_scoped_masks
exactly - not a reimplementation that could subtly differ and
invalidate the isolation) - so the image shown to stage 2 is IDENTICAL
to what it would see in a genuine two-stage run. The only variable
under the tester's direct control is the simulated stage-1 text.

Usage:
    # Simulate an empty stage-1 reading (the exact failure case):
    python test_stage2_isolated.py <sidecar.json> config/columns/relationship_only.txt --row 5 --structure-model qwen3vl4b --simulated-ocr ""

    # Simulate garbled/unrelated stage-1 text:
    python test_stage2_isolated.py <sidecar.json> config/columns/relationship_only.txt --row 5 --structure-model qwen3vl4b --simulated-ocr "4.2.2019"

    # Read the simulated text from a file instead (for longer/special content):
    python test_stage2_isolated.py <sidecar.json> config/columns/relationship_only.txt --row 5 --structure-model qwen3vl4b --simulated-ocr-file some_text.txt

What to look for in the result: if the output still guesses a
plausible-but-wrong word with no visible connection to the real
handwriting, that's real evidence stage 2 isn't meaningfully using the
image in this failure case - a deeper architectural gap (the image
isn't actually being weighted against a strong text-based guessing
prior), not a wording problem in the prompt.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from core.row_segmentation import load_sidecar, crop_region_from_source, compute_exclude_ranges
from core.loader_registry import LOADER_REGISTRY
from core.loaders.base_loader import load_model_config
from core.row_extraction import build_structuring_prompt, parse_row_output

# Same fix as core/row_extraction.py and test_native_prompt.py
# (2026-07-16) - Windows cp1252 console can't represent arbitrary
# Unicode a model might output.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


def _load_field_list(path: Path, kind: str) -> list[str]:
    if not path.exists():
        print(f"ERROR: {kind} file not found: {path}")
        sys.exit(1)
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if not names:
        print(f"ERROR: {kind} file is empty: {path}")
        sys.exit(1)
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sidecar_path", type=str, help="Path to a segmentation sidecar JSON")
    parser.add_argument("columns_path", type=str,
                         help="Text file with column names, one per line.")
    parser.add_argument("--row", type=int, required=True,
                         help="Row index to test (1-based, matching the sidecar's row numbering).")
    parser.add_argument("--structure-model", type=str, required=True,
                         help="Model profile name for the structuring pass.")
    sim_group = parser.add_mutually_exclusive_group(required=True)
    sim_group.add_argument("--simulated-ocr", type=str,
                            help="Simulated stage-1 reading, as a literal string. Use \"\" "
                                 "for an empty reading (the real failure case being tested).")
    sim_group.add_argument("--simulated-ocr-file", type=str,
                            help="Path to a text file containing the simulated stage-1 "
                                 "reading, for longer or special-character content.")
    args = parser.parse_args()

    sidecar_path = Path(args.sidecar_path)
    if not sidecar_path.exists():
        print(f"ERROR: sidecar not found: {sidecar_path}")
        sys.exit(1)

    column_names = _load_field_list(Path(args.columns_path), "column")

    if args.simulated_ocr_file:
        sim_path = Path(args.simulated_ocr_file)
        if not sim_path.exists():
            print(f"ERROR: simulated-ocr file not found: {sim_path}")
            sys.exit(1)
        simulated_ocr = sim_path.read_text(encoding="utf-8")
    else:
        simulated_ocr = args.simulated_ocr

    sidecar = load_sidecar(str(sidecar_path))
    source_path = sidecar["source_image_path"]
    deskew_angle = sidecar["deskew_angle"]

    row = next((r for r in sidecar["rows"] if r["index"] == args.row), None)
    if row is None:
        print(f"ERROR: row {args.row} not found in sidecar "
              f"(available: 1-{len(sidecar['rows'])})")
        sys.exit(1)

    # Mirror core.row_extraction._compute_scoped_masks exactly (not
    # reimplemented differently) - so the image is IDENTICAL to what
    # stage 2 would see in a genuine two-stage run. This is what makes
    # the isolation valid: only simulated_ocr is a free variable.
    keep_ranges = [tuple(k) for k in sidecar.get("mask_keep_ranges", [])]
    width = sidecar["deskewed_image_size"][0]
    row_masks = (
        compute_exclude_ranges(keep_ranges, width)
        if keep_ranges and sidecar.get("mask_apply_rows", False) else []
    )

    print(f"Testing row {args.row}, bbox={row['bbox']}")
    print(f"Columns: {', '.join(column_names)}")
    print(f"Structure model: {args.structure_model}")
    print(f"Simulated stage-1 reading: {simulated_ocr!r}")
    print(f"Row mask ranges applied: {row_masks}")
    print(f"{'='*60}")

    config = load_model_config(args.structure_model)
    loader_cls = LOADER_REGISTRY.get(config.loader_class)
    if loader_cls is None:
        print(f"ERROR: no loader registered for {config.loader_class!r}")
        sys.exit(1)

    loader = loader_cls(config)
    try:
        loader.initialize_model_and_tokenizer()

        region_image = crop_region_from_source(source_path, row["bbox"], deskew_angle, row_masks)
        if region_image.mode != "RGB":
            region_image = region_image.convert("RGB")

        prompt = build_structuring_prompt(simulated_ocr, column_names)

        start = time.time()
        raw_output = loader._run_generate(region_image, prompt)
        runtime = time.time() - start

        fields = parse_row_output(raw_output, column_names)
        missing = set(column_names) - set(fields.keys())

        print(f"Runtime: {runtime:.1f}s")
        print(f"{'='*60}")
        print("RAW OUTPUT:")
        print(raw_output)
        print(f"{'='*60}")
        print(f"Parsed fields: {fields}")
        if missing:
            print(f"Missing/dropped: {missing}")
    finally:
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


if __name__ == "__main__":
    main()
