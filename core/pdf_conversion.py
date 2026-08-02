"""
Rasterizes single-page-scan PDFs (e.g. LAC census PDFs like
https://data2.collectionscanada.gc.ca/1911/pdf/e001946614.pdf) into
image files - the PDF-specific piece behind core/source_expansion.py's
generic expansion layer (see that module for how Stage 0 actually
consumes this; core/manifest_pipeline.py never imports this module
directly, deliberately - see source_expansion.py's own docstring for
why the split exists).

Moved here from scripts/convert_pdf_to_image.py 2026-08-02 so core
modules can call this directly without importing from scripts/ (this
project's convention is the reverse: scripts/ are thin CLI adapters
over core/ engines). scripts/convert_pdf_to_image.py now just wraps
convert_pdf() for standalone manual use; the logic itself lives in
exactly one place.

Uses PyMuPDF (pip install PyMuPDF) to rasterize each page at a chosen
DPI. A PDF with more than one page (not the normal LAC case, but
handled defensively) writes one image per page, suffixed _p1, _p2,
...; a single-page PDF writes just <stem>.<ext>, no suffix.

PDF_EXTENSIONS and DEFAULT_PDF_OUTPUT_DIR live here, not in core/
manifest_pipeline.py or core/source_expansion.py - they're PDF-specific
configuration, and the whole point of the expansion-layer refactor
(2026-08-02) is that neither of those generic modules needs to know
PDF-specific details. ui/build_manifest_ui.py imports PDF_EXTENSIONS
from here for its file-picker's PDF-specific filter entry.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DPI = 300
PDF_EXTENSIONS = {".pdf"}
DEFAULT_PDF_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw_from_pdf"


def convert_pdf(pdf_path: Path, out_dir: Path, dpi: int = DEFAULT_DPI,
                 ext: str = "png", force: bool = False) -> list[Path]:
    import fitz  # PyMuPDF

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    doc = fitz.open(str(pdf_path))
    try:
        multi_page = doc.page_count > 1
        for page_index in range(doc.page_count):
            suffix = f"_p{page_index + 1}" if multi_page else ""
            out_path = out_dir / f"{pdf_path.stem}{suffix}.{ext}"
            if out_path.exists() and not force:
                written.append(out_path)
                continue
            page = doc.load_page(page_index)
            pix = page.get_pixmap(dpi=dpi)
            pix.save(str(out_path))
            written.append(out_path)
    finally:
        doc.close()
    return written
