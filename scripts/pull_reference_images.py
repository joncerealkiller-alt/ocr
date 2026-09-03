"""
Pulls a reference set (target ~200-300 images per taxonomy category) of
open-access, public-domain / permissively-licensed images for the
categories the review sessions have flagged as thin or missing in the
real corpus (casual_photo, cemetery_photo, photo_collage,
genealogy_chart, mixed_text_image, handwritten_ledger) - a rework of
training/lac_sequential_pull.py's structure (same conventions: rate-
limited, writes ONLY to --out-dir, never touches data/working/,
data/manifest.csv, or any production bucket CSV; a later, separate step
decides what/if anything gets promoted into the real corpus) but pulling
from category/keyword-searchable open archives instead of a known
sequential ID range, since these categories have no such ID scheme.

Three sources, chosen because all three are scrapable without an API
key (checked directly against each host, 2026-07-31 - see per-function
docstrings for the exact endpoints and response shapes verified):

  - Wikimedia Commons (MediaWiki API) - category-based, e.g.
    Category:Cemeteries. Best coverage for cemetery_photo,
    genealogy_chart, photo_collage.
  - Internet Archive (archive.org Advanced Search + item metadata API) -
    free-text/subject query based. Best coverage for photo_collage,
    mixed_text_image, handwritten_ledger.
  - Library of Congress (loc.gov/photos JSON API) - free-text query
    based. Decent secondary coverage for casual_photo, cemetery_photo,
    handwritten_ledger; US Government works are public domain by
    default.

Flickr Commons is NOT included - its API requires a registered key
(api_key param on every call), which this script doesn't have and
won't silently work around; add it later with FLICKR_API_KEY as an env
var if wanted (flickr.photos.search, license filter, same shape as the
other three fetch functions here).

PROVENANCE TAGGING: every manifest row gets provenance =
"deliberately_sourced_reference" (never "naturally_occurring_corpus") -
per Jon's note, so downstream analysis (accuracy comparisons, bucket
composition stats) never accidentally treats these as if they arrived
the same way as the real, naturally-collected corpus.

LICENSE IS RECORDED, NOT FILTERED: Wikimedia Commons in particular
returns a real mix (many CC-BY-SA 4.0, some PD, some other CC
variants) - every row's `license` column records exactly what the
source reported, and pull_summary.csv prints the distinct licenses
seen per category on exit, so Jon can decide per-category/per-license
what's fair to actually use, rather than this script silently making
that call. LOC results are near-certain US Government public domain
(not independently re-verified per item) - recorded as "loc-photos
(likely PD, US Government work)", not asserted as a hard guarantee.

Usage:
    python -m scripts.pull_reference_images --category cemetery_photo --target-count 250
    python -m scripts.pull_reference_images --category genealogy_chart --target-count 300 --out-dir data/outputs/reference_pull
    python -m scripts.pull_reference_images --list-categories
"""
from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "reference_pull"
DEFAULT_DELAY_SECONDS = 2.0  # same courtesy value training/lac_sequential_pull.py uses
USER_AGENT = "GenealogyPipelineResearchBot/1.0 (personal genealogical research; non-commercial)"
MIN_BYTES = 8_000  # below this is almost always an icon/placeholder, not a real reference photo
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

MANIFEST_FIELDS = [
    "file_path", "category", "source", "source_id", "source_url",
    "query", "license", "provenance",
]


@dataclass
class Candidate:
    source: str
    source_id: str       # unique WITHIN source - dedup key is (source, source_id)
    url: str              # direct, fetchable image URL
    license: str
    query: str             # the search/category term that surfaced this candidate


def _get_with_backoff(session: requests.Session, url: str, delay: float,
                       max_retries: int = 3) -> requests.Response | None:
    """
    Image DOWNLOADS (not the JSON API calls, which have their own
    per-source delay already) - specifically upload.wikimedia.org, a
    separate host from commons.wikimedia.org's API with its own,
    stricter, undocumented rate limit - hit real HTTP 429s during
    testing (2026-07-31) even at the same 1-2s delay the API itself was
    fine with. Backs off using the response's own Retry-After header
    when present, otherwise an increasing fixed wait (delay * 4, 8, 16),
    up to max_retries attempts before giving up on this one candidate
    (never blocks the whole run indefinitely on one stuck URL). Returns
    None (caller skips) if every retry still comes back 429/error.
    """
    for attempt in range(max_retries + 1):
        try:
            resp = session.get(url, timeout=30)
        except requests.RequestException as e:
            print(f"    ERROR fetching {url}: {e}")
            time.sleep(delay)
            return None
        if resp.status_code != 429:
            time.sleep(delay)
            return resp
        if attempt == max_retries:
            time.sleep(delay)
            return resp  # let the caller's normal non-200 handling log/skip it
        wait = float(resp.headers.get("Retry-After", delay * (2 ** (attempt + 2))))
        print(f"    429 from {urlparse(url).netloc} - backing off {wait:.0f}s (attempt {attempt + 1}/{max_retries})")
        time.sleep(wait)
    return None  # unreachable, but keeps type-checkers happy


# ---------------------------------------------------------------- sources

def wikimedia_candidates(category_titles: list[str], session: requests.Session,
                          delay: float) -> Iterator[Candidate]:
    """
    MediaWiki API, action=query. Two calls per batch of files:
      1. list=categorymembers&cmtype=file&cmlimit=500 (paginated via
         cmcontinue) - just gets File: titles, no image data yet.
      2. prop=imageinfo&iiprop=url|extmetadata, up to 50 titles per
         request (pipe-joined) - the real URL + LicenseShortName come
         from here. Batching this step (rather than one request per
         file) is what keeps this source's request count sane against
         a 200-300 image target.
    Verified directly against commons.wikimedia.org, 2026-07-31 - see
    module docstring.
    """
    api = "https://commons.wikimedia.org/w/api.php"
    for cat in category_titles:
        titles_buffer: list[str] = []
        cmcontinue = None
        while True:
            params = {
                "action": "query", "list": "categorymembers", "cmtitle": cat,
                "cmtype": "file", "cmlimit": 100, "format": "json",
            }
            if cmcontinue:
                params["cmcontinue"] = cmcontinue
            resp = session.get(api, params=params, timeout=30)
            time.sleep(delay)
            if resp.status_code != 200:
                print(f"    [wikimedia] {cat}: HTTP {resp.status_code} - skipping rest of this category")
                break
            data = resp.json()
            members = data.get("query", {}).get("categorymembers", [])
            titles_buffer.extend(m["title"] for m in members)

            # Flush in batches of 50 (the API's own titles-per-request cap)
            while len(titles_buffer) >= 50:
                batch, titles_buffer = titles_buffer[:50], titles_buffer[50:]
                yield from _wikimedia_imageinfo_batch(batch, cat, api, session, delay)

            cmcontinue = data.get("continue", {}).get("cmcontinue")
            if not cmcontinue:
                break
        if titles_buffer:
            yield from _wikimedia_imageinfo_batch(titles_buffer, cat, api, session, delay)


def _wikimedia_imageinfo_batch(titles: list[str], cat: str, api: str,
                                session: requests.Session, delay: float) -> Iterator[Candidate]:
    resp = session.get(api, params={
        "action": "query", "titles": "|".join(titles),
        "prop": "imageinfo", "iiprop": "url|extmetadata", "format": "json",
    }, timeout=30)
    time.sleep(delay)
    if resp.status_code != 200:
        return
    pages = resp.json().get("query", {}).get("pages", {})
    for page in pages.values():
        infos = page.get("imageinfo") or []
        if not infos:
            continue
        info = infos[0]
        url = info.get("url", "")
        if not url or Path(urlparse(url).path).suffix.lower() not in ALLOWED_EXTENSIONS:
            continue
        license_name = info.get("extmetadata", {}).get("LicenseShortName", {}).get("value", "unknown")
        yield Candidate(
            source="wikimedia_commons", source_id=str(page.get("pageid", page.get("title"))),
            url=url, license=license_name, query=cat,
        )


def internet_archive_candidates(queries: list[str], session: requests.Session,
                                 delay: float, rows_per_query: int = 200) -> Iterator[Candidate]:
    """
    archive.org Advanced Search API (no key) for identifiers, then one
    metadata call per identifier (archive.org/metadata/{id}) to find its
    largest real JPEG/PNG file - IA items are collections of files, not
    single images, so this second call is unavoidable per candidate.
    Verified directly, 2026-07-31 - see module docstring.
    """
    search_api = "https://archive.org/advancedsearch.php"
    for query in queries:
        resp = session.get(search_api, params={
            "q": query, "fl[]": "identifier", "rows": rows_per_query, "output": "json",
        }, timeout=30)
        time.sleep(delay)
        if resp.status_code != 200:
            print(f"    [internet_archive] {query!r}: HTTP {resp.status_code} - skipping")
            continue
        docs = resp.json().get("response", {}).get("docs", [])
        for doc in docs:
            identifier = doc.get("identifier")
            if not identifier:
                continue
            meta_resp = session.get(f"https://archive.org/metadata/{identifier}", timeout=30)
            time.sleep(delay)
            if meta_resp.status_code != 200:
                continue
            files = meta_resp.json().get("files", [])
            # Prefer the original, largest real image file - skip
            # derivative thumbnails/tile previews (format names like
            # "JPEG Thumb", "Item Tile") which are real files IA lists
            # but not useful as reference images.
            image_files = [
                f for f in files
                if f.get("source") == "original"
                and Path(f.get("name", "")).suffix.lower() in ALLOWED_EXTENSIONS
            ]
            if not image_files:
                continue
            best = max(image_files, key=lambda f: int(f.get("size", 0) or 0))
            url = f"https://archive.org/download/{identifier}/{best['name']}"
            yield Candidate(
                source="internet_archive", source_id=f"{identifier}/{best['name']}",
                url=url, license="unspecified (see item page)", query=query,
            )


def loc_candidates(queries: list[str], session: requests.Session,
                    delay: float, pages_per_query: int = 5) -> Iterator[Candidate]:
    """
    loc.gov/photos/ JSON API (no key), c=100 results/page, sp=page
    number. image_url is a list of available sizes for that record - the
    LAST entry is used (largest available; not always guaranteed to be
    more than a thumbnail for older/restricted items, per direct
    inspection 2026-07-31 - see module docstring). US Government
    Prints & Photographs Division holdings are public domain by
    default, not independently re-verified per item here.
    """
    api = "https://www.loc.gov/photos/"
    for query in queries:
        for page in range(1, pages_per_query + 1):
            resp = session.get(api, params={"q": query, "fo": "json", "c": 100, "sp": page}, timeout=30)
            time.sleep(delay)
            if resp.status_code != 200:
                print(f"    [loc] {query!r} page {page}: HTTP {resp.status_code} - stopping this query")
                break
            results = resp.json().get("results", [])
            if not results:
                break
            for r in results:
                image_urls = r.get("image_url") or []
                if not image_urls:
                    continue
                url = image_urls[-1]
                item_id = r.get("id", url)
                yield Candidate(
                    source="loc_photos", source_id=str(item_id), url=url,
                    license="loc-photos (likely PD, US Government work)", query=query,
                )


def openverse_candidates(queries: list[str], session: requests.Session,
                          delay: float, pages_per_query: int = 3) -> Iterator[Candidate]:
    """
    api.openverse.org (no key) - a CC-licensed image SEARCH ENGINE that
    aggregates many providers (Wikimedia Commons, PICRYL, Flickr Commons,
    museum collections, etc.) behind one query/response shape, added
    2026-07-31 per Jon's suggestion. license_type=commercial,modification
    restricts results to the CC/PD tiers that are actually free to use
    for this project's internal training-data purpose (excludes
    NonCommercial/NoDerivatives-only results) - filtering at the QUERY
    level here, not after download, since Openverse's API supports it
    directly. Verified directly, 2026-07-31 - 'url' is already a direct,
    fetchable image URL (no second per-item call needed, unlike
    Internet Archive).

    page_size CAPPED AT 20 (2026-08-09, real bug found - every
    handwritten_ledger query was silently 401ing on every run this
    session): originally requested page_size=100 "to cut request
    count," but Openverse anonymous (no-API-key) requests are hard-
    capped at page_size<=20 - confirmed directly against the real API,
    error body: "page_size may not exceed 20 for anonymous requests".
    100 has never actually worked without a registered API key; every
    call this source has ever made was failing, not just recently. Per
    Jon: "just drop it to 20 for now" - registering a free Openverse
    API key (they offer one for higher limits) would let this go back
    up, not pursued yet. NOTE: pages_per_query's default (3) was NOT
    changed here even though each page now returns 5x fewer results -
    if this source's total yield looks thin in a future run, that's
    the first knob to raise (more pages), not page_size (fixed by the
    anonymous-tier cap regardless).
    """
    api = "https://api.openverse.org/v1/images/"
    for query in queries:
        for page in range(1, pages_per_query + 1):
            resp = session.get(api, params={
                "q": query, "license_type": "commercial,modification",
                "page_size": 20, "page": page,
            }, timeout=30)
            time.sleep(delay)
            if resp.status_code != 200:
                print(f"    [openverse] {query!r} page {page}: HTTP {resp.status_code} - stopping this query")
                break
            results = resp.json().get("results", [])
            if not results:
                break
            for r in results:
                url = r.get("url")
                if not url or Path(urlparse(url).path).suffix.lower() not in ALLOWED_EXTENSIONS:
                    continue
                yield Candidate(
                    source="openverse", source_id=str(r.get("id", url)), url=url,
                    license=f"{r.get('license', 'unknown')} {r.get('license_version', '')}".strip(),
                    query=query,
                )


# ------------------------------------------------------------- category config

CATEGORY_QUERIES: dict[str, dict[str, list[str]]] = {
    # Expanded 2026-07-31 with Jon's per-category source review - added
    # "openverse" as a 4th query key (see openverse_candidates() - an
    # aggregator that itself surfaces PICRYL/Flickr Commons/other
    # providers, which is why those aren't separate hand-rolled sources
    # here) and additional wikimedia/loc terms he named directly.
    "casual_photo": {
        "wikimedia": ["Category:Snapshot photography", "Category:Candid photographs"],
        "internet_archive": ['subject:"family photographs" AND mediatype:image'],
        "loc": ["candid snapshot family life"],
        "openverse": ["everyday life snapshot", "candid photograph", "street scene photograph"],
    },
    "cemetery_photo": {
        # Jon: "Wikimedia Commons is by far the strongest... no living
        # people as subjects" - kept as the lead source; openverse added
        # as a second real source rather than the single-source risk of
        # relying on Commons alone if its rate limit bites hard on a
        # full-target run.
        "wikimedia": ["Category:Cemeteries", "Category:Headstones", "Category:Graveyards"],
        "internet_archive": ['subject:cemetery AND mediatype:image', 'subject:gravestone AND mediatype:image'],
        "loc": ["cemetery headstone", "grave marker monument"],
        "openverse": ["cemetery headstone", "graveyard tombstone"],
    },
    "photo_collage": {
        "wikimedia": ["Category:Photo collages", "Category:Photograph albums", "Category:Scrapbooks"],
        "internet_archive": ['subject:"photo album" AND mediatype:image', 'subject:scrapbook AND mediatype:image'],
        "loc": [],
        "openverse": ["scrapbook page", "photo album page", "vintage photo collage"],
    },
    "genealogy_chart": {
        # Jon: LOC's "Free to Use Genealogy" set is the cleanest source -
        # no separate collection-filtered API for that curated set was
        # found, so this stays a free-text query against the same
        # loc.gov/photos/ endpoint as everything else here; still LOC,
        # still likely to surface that same material.
        "wikimedia": ["Category:Family trees", "Category:Pedigree charts"],
        "internet_archive": ['subject:"pedigree chart" AND mediatype:image', 'subject:"family tree" AND mediatype:image'],
        "loc": ["family tree chart genealogy", "genealogy family register"],
        "openverse": ["family tree chart", "pedigree chart genealogy"],
    },
    "mixed_text_image": {
        "wikimedia": ["Category:Manuscripts"],
        "internet_archive": ['subject:"book illustration" AND mediatype:image'],
        "loc": [],
        "openverse": ["historical document page illustration"],
    },
    "handwritten_ledger": {
        "wikimedia": ["Category:Ledgers", "Category:Account books"],
        "internet_archive": ['subject:"account books" AND mediatype:image', 'subject:ledgers AND mediatype:image'],
        # "kirk session notes" added 2026-08-09, per Jon, after this
        # category fell short of target (124/250 - the configured
        # queries ran out of new candidates). Checked availability
        # directly before adding (per Jon: "if they are available") -
        # loc.gov returned 5 real results on a 1-page test (real Scottish
        # kirk-session record images, e.g. loc.gov/item/19008438);
        # internet_archive (both a subject:"kirk session" filter and a
        # plain free-text query) and a wikimedia "Category:Kirk sessions"
        # guess both returned 0 - only added here, not as a dead query on
        # those other two sources.
        "loc": ["account book ledger manuscript", "kirk session notes"],
        "openverse": ["handwritten ledger register", "parish register manuscript"],
    },
}


# ------------------------------------------------------------------- fetch loop

def fetch_category(category: str, target_count: int, out_dir: Path, delay: float) -> None:
    if category not in CATEGORY_QUERIES:
        raise SystemExit(f"Unknown category {category!r}. Known: {', '.join(CATEGORY_QUERIES)}")

    cfg = CATEGORY_QUERIES[category]
    cat_dir = out_dir / category
    cat_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cat_dir / "manifest.csv"

    seen: set[tuple[str, str]] = set()
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                seen.add((row["source"], row["source_id"]))
        print(f"  Resuming - {len(seen)} already pulled for {category!r}.")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    def candidate_stream() -> Iterator[Candidate]:
        if cfg["wikimedia"]:
            yield from wikimedia_candidates(cfg["wikimedia"], session, delay)
        if cfg["internet_archive"]:
            yield from internet_archive_candidates(cfg["internet_archive"], session, delay)
        if cfg["loc"]:
            yield from loc_candidates(cfg["loc"], session, delay)
        if cfg.get("openverse"):
            yield from openverse_candidates(cfg["openverse"], session, delay)

    is_new_manifest = not manifest_path.exists()
    manifest_f = open(manifest_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(manifest_f, fieldnames=MANIFEST_FIELDS)
    if is_new_manifest:
        writer.writeheader()

    fetched = len(seen)
    licenses_seen: set[str] = set()
    try:
        for cand in candidate_stream():
            if fetched >= target_count:
                break
            key = (cand.source, cand.source_id)
            if key in seen:
                continue
            seen.add(key)

            ext = Path(urlparse(cand.url).path).suffix.lower() or ".jpg"
            safe_id = "".join(c if c.isalnum() else "_" for c in cand.source_id)[:120]
            file_path = cat_dir / f"{cand.source}_{safe_id}{ext}"

            resp = _get_with_backoff(session, cand.url, delay)
            if resp is None:
                print(f"    skip {cand.source}/{cand.source_id}: gave up after repeated 429s")
                continue

            if resp.status_code != 200 or len(resp.content) < MIN_BYTES:
                print(f"    skip {cand.source}/{cand.source_id}: "
                      f"HTTP {resp.status_code}, {len(resp.content)} bytes")
                continue

            file_path.write_bytes(resp.content)
            writer.writerow({
                "file_path": str(file_path), "category": category, "source": cand.source,
                "source_id": cand.source_id, "source_url": cand.url, "query": cand.query,
                "license": cand.license, "provenance": "deliberately_sourced_reference",
            })
            manifest_f.flush()
            licenses_seen.add(cand.license)
            fetched += 1
            print(f"    [{fetched}/{target_count}] {cand.source}: {file_path.name}")
    finally:
        manifest_f.close()

    print(f"\n{category}: {fetched}/{target_count} images. Manifest: {manifest_path}")
    if licenses_seen:
        print(f"  Licenses seen this run: {sorted(licenses_seen)}")
    if fetched < target_count:
        print(f"  Short of target - configured queries ran out of new candidates. "
              f"Add more CATEGORY_QUERIES entries for {category!r} to pull further.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category", type=str, default=None,
                         help="One of: " + ", ".join(CATEGORY_QUERIES))
    parser.add_argument("--target-count", type=int, default=250)
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY_SECONDS)
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--list-categories", action="store_true",
                         help="Print configured categories and their query sources, then exit.")
    args = parser.parse_args()

    if args.list_categories:
        for cat, cfg in CATEGORY_QUERIES.items():
            sources = [s for s in ("wikimedia", "internet_archive", "loc", "openverse") if cfg.get(s)]
            print(f"  {cat}: {', '.join(sources)}")
        return

    if not args.category:
        raise SystemExit("--category is required (or use --list-categories to see options).")

    out_dir = Path(args.out_dir)
    print(f"Pulling up to {args.target_count} reference image(s) for {args.category!r} -> {out_dir / args.category}")
    fetch_category(args.category, args.target_count, out_dir, args.delay)


if __name__ == "__main__":
    main()
