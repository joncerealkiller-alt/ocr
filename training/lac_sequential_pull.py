"""
Pulls a batch of sequential 1911 Canada census page images directly from
data2.collectionscanada.gc.ca (the LAC image/PDF file host) - NOT from
the search/browse tool (recherche-collection-search.bac-lac.gc.ca),
whose robots.txt explicitly disallows crawling (names ClaudeBot
specifically). data2.collectionscanada.gc.ca's own robots.txt is
permissive (default Disallow: empty for generic bots) - checked
directly, 2026-08-05.

ID scheme confirmed from this project's own existing corpus: known IDs
like e001946614 through e001946623 (10 of the 16 census images already
in data/working/) are LITERALLY CONSECUTIVE integers in the numeric
suffix - per direct instruction, this script continues that exact
sequence forward rather than attempting any search/discovery.

Rate-limited (DEFAULT_DELAY_SECONDS between requests) as a courtesy to
a government archive server, even though the wildcard robots.txt entry
carries no explicit Crawl-delay (only the Googlebot/Googlebot-Image
groups do, at 2s - used as the reference value here).

404s / missing IDs are expected and handled gracefully (not every
numeric ID in a reel's range is guaranteed to be a valid census
schedule page) - logged and skipped, not treated as fatal.

Non-commercial reproduction of Government of Canada material is
permitted without further permission under canada.ca's Terms and
Conditions (checked directly, 2026-08-05), provided source is
attributed - this project's own use (personal genealogical research +
internal ML training-data bootstrap, no redistribution) qualifies.

Writes to --out-dir (default data/outputs/lac_pull_batch1/) - never
touches data/working/, data/manifest.csv, or any production corpus
file. A later, separate step decides whether/how to promote any of
this into the real corpus.

Multi-year support (2026-08-05): --url-template lets this same script
pull other census years, once their URL pattern is confirmed by direct
HEAD-request testing (never guessed/assumed) - e.g. 1921 uses the exact
same flat "/{year}/pdf/e{id:09d}.pdf" shape as 1911, just a different
year segment and confirmed independently; 1901 uses a genuinely
different shape ("/1901/z/z001/pdf/z{id:09d}.pdf", a letter/reel
subfolder) and needs its own --url-template value rather than this
script's default.

New host support (2026-08-05): central.bac-lac.gc.ca/.item/?app=Census{year}&op=img&id={id}
307-redirects to data2.archives.ca (confirmed permissive robots.txt,
same shape as data2.collectionscanada.gc.ca's), serving raw .jpg pages
directly (no PDF wrapper) - confirmed working for 1901/1906/1911/1921/
1926/1931 via direct GET request, 2026-08-05. requests follows the
redirect transparently, so --url-template can point straight at either
the central.bac-lac.gc.ca redirector or the data2.archives.ca target
directly. Use --ext jpg for these (default remains pdf for the
original data2.collectionscanada.gc.ca census-PDF pulls).

Usage:
    python -m training.lac_sequential_pull --start 1946824 --count 100
    python -m training.lac_sequential_pull --start 2869157 --count 100 \\
        --url-template "https://data2.collectionscanada.gc.ca/1921/pdf/e{id:09d}.pdf" \\
        --stem-prefix e --out-dir data/outputs/lac_pull_1921_batch1
    python -m training.lac_sequential_pull --start 11252626 --count 50 --ext jpg \\
        --url-template "https://data2.archives.ca/1926/jpg/e{id:09d}.jpg" \\
        --stem-prefix e --out-dir data/outputs/lac_pull_1926_batch1
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "lac_pull_batch1"
DEFAULT_BASE_URL = "https://data2.collectionscanada.gc.ca/1911/pdf/e{id:09d}.pdf"

DEFAULT_START_ID = 1946624  # continues the proven-consecutive e001946614-623 run
DEFAULT_COUNT = 50
DEFAULT_DELAY_SECONDS = 2.0  # matches this host's own Googlebot Crawl-delay value
USER_AGENT = "GenealogyPipelineResearchBot/1.0 (personal genealogical research; non-commercial)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=int, default=DEFAULT_START_ID)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY_SECONDS)
    parser.add_argument("--url-template", type=str, default=DEFAULT_BASE_URL,
                         help="Must contain {id:09d} or equivalent - format-string compatible with a plain int.")
    parser.add_argument("--stem-prefix", type=str, default="e",
                         help="Filename prefix for saved files, e.g. 'e' -> e001946824.pdf, 'z' -> z000017635.pdf.")
    parser.add_argument("--ext", type=str, default="pdf",
                         help="File extension to save as, e.g. 'pdf' (data2.collectionscanada.gc.ca) "
                              "or 'jpg' (data2.archives.ca / central.bac-lac.gc.ca redirector).")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.out_dir)
    pdf_dir = output_dir / f"raw_{args.ext}s"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    fetched, missing, errors = [], [], []
    for i, numeric_id in enumerate(range(args.start, args.start + args.count), 1):
        stem = f"{args.stem_prefix}{numeric_id:09d}"
        url = args.url_template.format(id=numeric_id)
        out_path = pdf_dir / f"{stem}.{args.ext}"

        if out_path.exists():
            fetched.append(stem)
            print(f"[{i}/{args.count}] {stem}: already have it, skipping fetch")
            continue

        try:
            resp = session.get(url, timeout=30)
        except requests.RequestException as e:
            errors.append((stem, str(e)))
            print(f"[{i}/{args.count}] {stem}: ERROR {e}")
            time.sleep(args.delay)
            continue

        if resp.status_code == 200 and resp.content:
            out_path.write_bytes(resp.content)
            fetched.append(stem)
            print(f"[{i}/{args.count}] {stem}: OK ({len(resp.content)} bytes)")
        elif resp.status_code == 404:
            missing.append(stem)
            print(f"[{i}/{args.count}] {stem}: 404 (no page at this ID)")
        else:
            errors.append((stem, f"HTTP {resp.status_code}"))
            print(f"[{i}/{args.count}] {stem}: HTTP {resp.status_code}")

        time.sleep(args.delay)

    print(f"\nFetched: {len(fetched)}, missing (404): {len(missing)}, errors: {len(errors)}")
    if missing:
        print(f"Missing IDs: {missing}")
    if errors:
        print(f"Errors: {errors}")
    print(f"PDFs saved to {pdf_dir}")


if __name__ == "__main__":
    main()
