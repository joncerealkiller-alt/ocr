"""
Generic source-expansion layer for Stage 0 ingestion, added 2026-08-02
per Jon's direction to generalize what started as PDF-specific handling
in core/manifest_pipeline.py: "the objective is not simply to add PDF
support - the objective is to introduce a generic source expansion
stage into the manifest pipeline."

Ingestion pipeline shape this implements:
    Source Discovery -> Source Expansion -> Manifest Construction -> ...

core/manifest_pipeline.py (the manifest builder) only ever calls
expand_source_paths(source_paths) and consumes the resulting
ExpandedSource objects' expanded_path - it has NO knowledge of which
expander produced them, where any temporary/rendered files were
written, or what configuration that expander needed. Every one of
those details is owned entirely by the expander itself (a SourceExpander
subclass, e.g. PdfExpander), never by this module's registry/dispatch
code and never by the manifest builder.

Adding a future expandable format (multipage TIFF, JPEG2000, a ZIP
archive, ...) means writing one SourceExpander subclass and calling
register_expander() with an instance of it - core/manifest_pipeline.py
needs zero changes, and this module's own dispatch code needs zero
changes either.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from core.pdf_conversion import convert_pdf, PDF_EXTENSIONS, DEFAULT_PDF_OUTPUT_DIR, DEFAULT_DPI


@dataclass
class ExpandedSource:
    """
    THE canonical interchange object between source expansion and
    manifest construction. For a plain image, source_path ==
    expanded_path, source_type="image", page_number=None, metadata={} -
    a normal image is a degenerate one-to-one expansion, not a special
    case any caller needs to branch on.

    metadata is deliberately open-ended (format-specific info a future
    expander wants to attach - e.g. a ZIP expander's original archive
    member name, a TIFF expander's page count) without ever needing to
    add a new field to this dataclass for it.
    """
    source_path: Path             # the original source (an image, a PDF, a future archive/etc.)
    expanded_path: Path           # the real image path - what the manifest builder actually consumes
    source_type: str              # "image", "pdf", or a future expander's own tag
    page_number: int | None = None  # 1-indexed, only meaningful for multi-page sources
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["source_path"] = str(self.source_path)
        d["expanded_path"] = str(self.expanded_path)
        return d


class SourceExpander(ABC):
    """
    Base for a self-contained expander: declares which extensions it
    handles and owns ALL of its own configuration (output directory,
    render resolution, whatever else) internally - nothing about that
    configuration is ever exposed through core/manifest_pipeline.py's
    function signatures.
    """
    extensions: set[str]

    @abstractmethod
    def expand(self, source_path: Path) -> list[ExpandedSource]:
        ...


class PdfExpander(SourceExpander):
    extensions = PDF_EXTENSIONS

    def __init__(self, output_dir: Path = DEFAULT_PDF_OUTPUT_DIR, dpi: int = DEFAULT_DPI):
        self.output_dir = output_dir
        self.dpi = dpi

    def expand(self, source_path: Path) -> list[ExpandedSource]:
        """
        convert_pdf() is already idempotent (skips an existing rendered
        page unless force=True) and already reuses previously-rendered
        pages - both preserved unchanged by routing through it here
        rather than reimplementing anything.
        """
        images = convert_pdf(source_path, self.output_dir, dpi=self.dpi)
        return [
            ExpandedSource(
                source_path=source_path, expanded_path=image, source_type="pdf",
                page_number=i + 1, metadata={"dpi": self.dpi},
            )
            for i, image in enumerate(images)
        ]


# Registry: extension -> the expander instance that owns it. Populated
# via register_expander(), not edited directly, so registering a future
# expander is always "write the class, call register_expander(MyExpander())"
# - one line, one place, no dispatch code to touch.
SOURCE_EXPANDERS: dict[str, SourceExpander] = {}


def register_expander(expander: SourceExpander) -> None:
    for ext in expander.extensions:
        SOURCE_EXPANDERS[ext] = expander


register_expander(PdfExpander())

# What collect_image_paths() (core/manifest_pipeline.py) walks for, IN
# ADDITION to plain image extensions - anything with a registered
# expander is a valid Stage 0 source even though it isn't an image yet.
EXPANDABLE_EXTENSIONS: set[str] = set(SOURCE_EXPANDERS.keys())


def expand_source_paths(source_paths: list[Path]) -> list[ExpandedSource]:
    """
    THE generic entry point, and the only function core/manifest_
    pipeline.py calls into this module. Every input path becomes one or
    more ExpandedSource records - a plain image expands to itself, and
    anything with a registered expander (currently just .pdf) expands
    via that expander's own expand() instead. Order is preserved for
    non-expanding entries; an expanding entry's pages appear in page
    order at that entry's position.
    """
    expanded: list[ExpandedSource] = []
    for source_path in source_paths:
        expander = SOURCE_EXPANDERS.get(source_path.suffix.lower())
        if expander is not None:
            expanded.extend(expander.expand(source_path))
        else:
            expanded.append(ExpandedSource(
                source_path=source_path, expanded_path=source_path,
                source_type="image", page_number=None, metadata={},
            ))
    return expanded
