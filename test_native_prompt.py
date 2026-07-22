"""
Native-prompt comparison test - runs a model's OWN _build_prompt()
output (its trained/expected prompt format) against a single row crop,
instead of our generic column-based build_row_prompt(). Prints raw
output only - makes NO attempt to parse it into fields.

Built 2026-07-13 specifically to test olmOCR's native RL-trained YAML
prompt (build_no_anchoring_v4_yaml_prompt) against row 6 of the 1931
census page after its generic-prompt run showed a real, confirmed
problem: every field tagged "confirmed" with a 67% error rate on
birthplace specifically (4/6 wrong, including two independently
fabricated "New York" answers that don't correspond to anything
visible - McRoberts Burton C's actual birthplace is Manitoba, not
"New York" as reported).

Why this needs to be a SEPARATE path, not a flag on run_row_extraction:
olmOCR's native prompt produces its own YAML front-matter + freeform
text structure (confirmed via its whole-page tests earlier this
session), not our "ColumnName: value|confidence" convention -
parse_row_output() would fail against it regardless of read quality,
since the output shapes are fundamentally different, not because the
native prompt is worse. Comparing native-prompt quality means reading
the raw output directly, the same "read the raw pane" approach already
established for Florence-2 and olmOCR's own whole-page testing.

Usage:
    python test_native_prompt.py <sidecar.json> --model olmocr_2_7b --row 6
    python test_native_prompt.py <sidecar.json> --model olmocr_2_7b --header

Only meaningful for loaders with a real native-prompt override
(currently: OlmOcrLoader). For every other loader, _build_prompt(task=
"extract") just returns config.prompt_text (usually empty in these
configs, since row extraction normally builds its own prompt) - running
this against a loader with no real native prompt will likely produce a
near-empty or generic response, which is itself informative (confirms
that loader has no trained prompt of its own to compare against) but
isn't the comparison this script exists for.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from core.row_segmentation import load_sidecar, crop_region_from_source
from core.loader_registry import LOADER_REGISTRY
from core.loaders.base_loader import load_model_config

# Same fix as core/row_extraction.py (2026-07-16) - this script prints
# raw model output directly and doesn't import that module, so it
# wouldn't inherit the fix automatically. See that module's comment for
# the full explanation (Windows cp1252 console can't represent
# arbitrary Unicode a model might hallucinate into raw output).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sidecar_path", type=str, help="Path to a segmentation sidecar JSON")
    parser.add_argument("--model", type=str, required=True,
                         help="Model profile name - must have a real native-prompt "
                              "override to be a meaningful test (currently: olmocr_2_7b).")
    parser.add_argument("--row", type=int, default=None,
                         help="Row index to test (1-based, matching the sidecar's "
                              "row numbering).")
    parser.add_argument("--header", action="store_true",
                         help="Test the header/metadata region instead of a row.")
    args = parser.parse_args()

    if args.row is None and not args.header:
        print("ERROR: specify either --row N or --header")
        sys.exit(1)

    sidecar_path = Path(args.sidecar_path)
    if not sidecar_path.exists():
        print(f"ERROR: sidecar not found: {sidecar_path}")
        sys.exit(1)

    sidecar = load_sidecar(str(sidecar_path))
    source_path = sidecar["source_image_path"]
    deskew_angle = sidecar["deskew_angle"]

    if args.header:
        bbox = sidecar.get("metadata_bbox") or sidecar.get("table_bbox")
        if bbox is None:
            print("ERROR: sidecar has neither metadata_bbox nor table_bbox")
            sys.exit(1)
        if sidecar.get("metadata_bbox") is None:
            width = sidecar["deskewed_image_size"][0]
            bbox = [0, 0, width, sidecar["table_bbox"][1]]
        label = "header/metadata region"
    else:
        row = next((r for r in sidecar["rows"] if r["index"] == args.row), None)
        if row is None:
            print(f"ERROR: row {args.row} not found in sidecar "
                  f"(available: 1-{len(sidecar['rows'])})")
            sys.exit(1)
        bbox = row["bbox"]
        label = f"row {args.row}"

    print(f"Testing {label}, bbox={bbox}")
    print(f"Model: {args.model}")
    print(f"{'='*60}")

    config = load_model_config(args.model)
    loader_cls = LOADER_REGISTRY.get(config.loader_class)
    if loader_cls is None:
        print(f"ERROR: no loader registered for {config.loader_class!r}")
        sys.exit(1)

    loader = loader_cls(config)
    try:
        loader.initialize_model_and_tokenizer()

        # THE key difference from run_row_extraction: uses the loader's
        # OWN _build_prompt(), not our generic build_row_prompt(). For
        # OlmOcrLoader this returns build_no_anchoring_v4_yaml_prompt()
        # regardless of the "extract" task argument - see that loader's
        # _build_prompt docstring.
        prompt = loader._build_prompt(task="extract")
        print(f"\nPrompt used (first 200 chars): {prompt[:200]!r}")
        print(f"{'='*60}\n")

        region_image = crop_region_from_source(source_path, bbox, deskew_angle)
        if region_image.mode != "RGB":
            region_image = region_image.convert("RGB")

        start = time.time()
        raw_output = loader._run_generate(region_image, prompt)
        runtime = time.time() - start

        print(f"Runtime: {runtime:.1f}s")
        print(f"{'='*60}")
        print("RAW OUTPUT (no parsing attempted - native prompt output "
              "doesn't match our column format, read this directly):")
        print(f"{'='*60}")
        print(raw_output)

    finally:
        import gc
        try:
            loader.model = None
            loader.processor = None
            loader.tokenizer = None
        except Exception:
            pass
        del loader
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    main()
