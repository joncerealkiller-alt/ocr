"""
CLI wrapper for standalone, manual PDF-to-image conversion. The real
logic lives in core/pdf_conversion.py's convert_pdf() - this script
just adds a folder-or-file CLI and per-file progress printing on top of
it (core/manifest_pipeline.py now calls convert_pdf() directly for
PDFs found during manifest building, so this script is for manual use
outside that flow, e.g. pre-converting a batch before Stage 0 runs).

Usage:
    python scripts/convert_pdf_to_image.py <pdf_file_or_folder> \\
        [--out-dir data/raw_from_pdf] [--dpi 300] [--ext png] [--force]

--out-dir defaults to data/raw_from_pdf/ - point Stage 0's
--source-folder at this directory (or wherever --out-dir was set to)
once conversion is done. (As of 2026-08-02, Stage 0 also converts PDFs
found directly in its own source folder automatically - this manual
route is for converting a batch ahead of time instead.)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.pdf_conversion import convert_pdf, DEFAULT_DPI  # noqa: E402

DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "raw_from_pdf"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="A single .pdf file, or a folder of .pdf files.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                         help=f"Where images are written (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI,
                         help=f"Render resolution (default: {DEFAULT_DPI})")
    parser.add_argument("--ext", default="png", choices=["png", "jpg", "jpeg"],
                         help="Output image format (default: png)")
    parser.add_argument("--force", action="store_true",
                         help="Overwrite an existing output image (default: skip it).")
    args = parser.parse_args()

    try:
        import fitz  # noqa: F401
    except ImportError:
        print("PyMuPDF is not installed. Run: pip install PyMuPDF", file=sys.stderr)
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if input_path.is_dir():
        pdf_paths = sorted(input_path.glob("*.pdf"))
        if not pdf_paths:
            print(f"No .pdf files found in {input_path}", file=sys.stderr)
            sys.exit(1)
    else:
        if input_path.suffix.lower() != ".pdf":
            print(f"Not a .pdf file: {input_path}", file=sys.stderr)
            sys.exit(1)
        pdf_paths = [input_path]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_written = 0
    for pdf_path in pdf_paths:
        print(f"{pdf_path.name}:")
        before = set(out_dir.glob(f"{pdf_path.stem}*.{args.ext}"))
        images = convert_pdf(pdf_path, out_dir, args.dpi, args.ext, args.force)
        for img in images:
            tag = "wrote" if img not in before or args.force else "skip (exists)"
            print(f"  {tag}: {img.name}")
        total_written += len(images)

    print(f"\nDone. {total_written} image(s) available in {out_dir}")


if __name__ == "__main__":
    main()
