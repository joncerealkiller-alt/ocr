"""
CSV hint source - feeds Phase 1 of ui/hint_validation_ui.py from a
pre-transcribed reference CSV (e.g. data/automatedgenealogy_pull.csv,
pulled from automatedgenealogy.com's human transcriptions) instead of
a model extraction pass. Those pages already have a professional
transcription; there's no reason to make a model guess first when the
answer is already known and just needs a human to confirm it against
the image.

Row matching (per Jon, 2026-07-29): sidecar row `index` is assigned by
walking column-by-column across the page and starting a new row each
time the first bbox column repeats - i.e. row N in the sidecar IS row
N of the physical table on the page, same as the CSV's `line_num`
column for that page. So matching is a direct row_index == line_num
join, not fuzzy.

Page matching (per Jon, 2026-07-29): the auto-pipeline's output image
filename scheme isn't finalized yet, so match on a PREFIX check - the
CSV's source_pdf (e.g. ".../e001946614.pdf") reduces to an identifier
stem ("e001946614"), and any sidecar source_image_path whose filename
stem STARTS WITH that identifier is treated as the same page (every
pipeline stage appends its own suffix - _dewarped, _sidecar, etc. -
onto the original PDF stem, it's never prepended). Confirmed against a
real run, 2026-07-29 - this was originally written as an endswith()
check and silently matched nothing. Adjust
_page_identifier()/_image_matches_identifier() here first if the real
naming scheme turns out to need something smarter.

Column mapping: the CSV's fixed schema (see scripts that built
data/automatedgenealogy_pull.csv) doesn't share column names with
whatever config/columns/*.txt file is loaded in the UI - CSV_COLUMN_MAP
below is the translation, keyed by the column NAME as it appears in
that columns file (case/spacing-insensitive). A sidecar will likely
have more bbox columns than this CSV covers (e.g. Birthplace, which
these 10 numbered census columns don't include) - any column with no
entry here simply yields no hint and falls through to manual entry in
Phase 2, same as a never-extracted field. Edit this map, not the UI,
once Jon's new columns file names are finalized.
"""

from __future__ import annotations

import csv
from pathlib import Path

# column name (normalized) -> CSV field name, or tuple of CSV field
# names to join with a space (for composite hints like a birth date
# split across two source columns).
CSV_COLUMN_MAP: dict[str, str | tuple[str, ...]] = {
    "name": "surname_given",
    "full name": "surname_given",
    "household number": "family_number",
    "household no": "family_number",
    "household": "family_number",
    "family number": "family_number",
    # "Address"/"Dwelling Number" deliberately NOT mapped (Jon,
    # 2026-07-29): the CSV's occupation_col slot does hold data on some
    # pages (e.g. Victoria, BC street addresses), but the transcription
    # is condensed/re-numbered relative to the real form, so that data
    # does not reliably correspond to whatever the sidecar's actual
    # Address/Dwelling Number column is - offering it as a hint would
    # be actively wrong, not just occasionally missing. Jon marks these
    # with a leading '#' in the columns file, which skips them before
    # this map is ever consulted; left commented out here too as a
    # trap for anyone tempted to wire it back up without re-verifying.
    "sex": "sex",
    "relationship to head": "relation_to_head",
    "relation to head": "relation_to_head",
    "marital status": "marital_status",
    "birth month": "month_of_birth",
    "month of birth": "month_of_birth",
    "birth year": "year",
    "year": "year",
    "date of birth": ("month_of_birth", "year"),
    "age": "age",
    # "Links" deliberately NOT mapped (Jon, 2026-07-29): doesn't need
    # ground-truth tracking at all, and the CSV no longer even carries a
    # links column (dropped from data/automatedgenealogy_pull.csv). Jon
    # marks it with a leading '#' in the columns file, same as
    # Address/Dwelling Number above.
}


def _normalize(name: str) -> str:
    return " ".join(name.strip().lower().split())


def load_csv_rows(csv_path) -> list[dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _page_identifier(source_pdf_url: str) -> str:
    """'.../e001946614.pdf' -> 'e001946614'."""
    return Path(source_pdf_url.strip()).stem


def _image_matches_identifier(image_path: str, identifier: str) -> bool:
    # Was endswith() - wrong direction. Every stage in this pipeline
    # APPENDS a suffix to the original stem (e001926997 -> ..._dewarped
    # -> ..._dewarped_sidecar, etc.), so the PDF identifier is always a
    # PREFIX of the final image filename, never a suffix. Confirmed
    # against a real run, 2026-07-29 - endswith() silently matched
    # nothing and every field fell through to manual entry.
    return Path(image_path).stem.startswith(identifier)


def rows_for_image(all_rows: list[dict[str, str]], source_image_path: str) -> dict[int, dict[str, str]]:
    """
    Returns {line_num: csv_row} for whichever CSV page (source_pdf)
    matches this sidecar's source image, or {} if no page matches -
    the caller should treat that as "no CSV hints available", not an
    error, since not every sidecar necessarily has a corresponding
    scraped page.
    """
    by_identifier: dict[str, list[dict[str, str]]] = {}
    for row in all_rows:
        by_identifier.setdefault(_page_identifier(row["source_pdf"]), []).append(row)

    for identifier, page_rows in by_identifier.items():
        if _image_matches_identifier(source_image_path, identifier):
            result = {}
            for row in page_rows:
                try:
                    line_num = int(row["line_num"])
                except (KeyError, ValueError):
                    continue
                result[line_num] = row
            return result
    return {}


def get_hint(csv_row: dict[str, str], column_name: str) -> str | None:
    """
    Looks up the hint value for `column_name` (as named in whatever
    config/columns/*.txt file the UI loaded) from a single matched CSV
    row. Returns None (not "") when the column has no mapping or the
    source field(s) were blank in the CSV - callers should treat None
    as "nothing to confirm here", same as a never-extracted sidecar
    field, and route straight to manual entry.
    """
    mapped = CSV_COLUMN_MAP.get(_normalize(column_name))
    if mapped is None:
        return None
    fields = (mapped,) if isinstance(mapped, str) else mapped
    parts = [csv_row.get(f, "").strip() for f in fields]
    parts = [p for p in parts if p]
    if not parts:
        return None
    return " ".join(parts)
