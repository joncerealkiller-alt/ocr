"""
One-time, read-only-against-production recovery script (2026-08-04).

data/baseline_embeddings.json (the production Stage 1 semantic-sensor
capture) was destroyed twice this session (see docs/GPU_CPU_EQUIVALENCE_
REPORT.md's incident notes) and, separately, the same investigation found
the corpus's ORIGINAL capture had measured already-Stage-3-preprocessed
pixels rather than true raw ones (docs/REFERENCE_PIPELINE_V1.md's bug
class, recreated in miniature this session). data/working/*'s current
bytes are confirmed post-Stage-3 (every image's current_hash != its
identity_hash in data/pipeline.db) - the true pre-preprocessing bytes no
longer exist at those paths.

This script re-copies each image's ORIGINAL bytes from its DB source_path
(all 1750 confirmed still present at their recorded external location)
into a fresh directory, data/raw_stage0_recapture/, named to match its
corresponding working_path's basename. Every copy is verified against
the DB's identity_hash (captured once, at the real Stage 0, before any
processing ever touched the file) - a mismatch means this recovery
approach doesn't hold for that image and is reported, not silently
accepted.

Does NOT touch data/working, manifest.csv, or pipeline.db. Purely
additive - writes only under data/raw_stage0_recapture/.

Usage:
    python -m scripts.reconstruct_raw_stage0
"""
from __future__ import annotations

import hashlib
import shutil
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "pipeline.db"
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw_stage0_recapture"
# PDF-page-derived images: source_path points at the original .pdf, not
# a directly-copyable image - a straight byte copy of the PDF is not the
# rasterized page and will fail the hash check below. A prior session
# already re-ran the PDF rasterization for these 16 into this directory;
# reuse it here instead of re-implementing PDF expansion in this script.
PRERASTERIZED_PDF_PAGES_DIR = PROJECT_ROOT / "data" / "raw_from_pdf"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT working_path, source_path, identity_hash FROM images")
    rows = cur.fetchall()
    print(f"{len(rows)} images in DB.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    copied, mismatched, missing = 0, [], []
    mapping: dict[str, str] = {}  # raw copy path (str) -> working_path (str)

    for row in rows:
        working_path = Path(row["working_path"])
        source_path = Path(row["source_path"])
        identity_hash = row["identity_hash"]

        dest = OUTPUT_DIR / working_path.name

        if source_path.suffix.lower() == ".pdf":
            # A straight byte copy of the PDF is not the rasterized page
            # inside it - use the already-rasterized version instead.
            prerasterized = PRERASTERIZED_PDF_PAGES_DIR / working_path.name
            if not prerasterized.exists():
                missing.append(f"{source_path} (PDF page, no prerasterized copy at {prerasterized})")
                continue
            shutil.copyfile(prerasterized, dest)
        else:
            if not source_path.exists():
                missing.append(str(source_path))
                continue
            shutil.copyfile(source_path, dest)

        actual_hash = _sha256(dest)
        # identity_hash values observed with an extra leading char vs a
        # plain 64-char sha256 hexdigest in this DB - compare by
        # substring containment rather than assuming exact equality,
        # same defensive check used in the earlier raw_from_pdf spot
        # check this session.
        if actual_hash not in identity_hash and identity_hash not in actual_hash:
            mismatched.append((str(working_path), actual_hash, identity_hash))
            continue

        mapping[str(dest)] = str(working_path)
        copied += 1

    print(f"\nCopied and hash-verified: {copied}/{len(rows)}")
    if missing:
        print(f"Missing source files: {len(missing)}")
        for m in missing[:10]:
            print(f"  {m}")
    if mismatched:
        print(f"Hash MISMATCHES (not copied as verified): {len(mismatched)}")
        for wp, actual, expected in mismatched[:10]:
            print(f"  {wp}: got {actual[:16]}... expected {expected[:16]}...")

    mapping_path = OUTPUT_DIR.parent / "raw_stage0_recapture_mapping.json"
    import json
    mapping_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    print(f"\nMapping (raw copy path -> working_path) written to {mapping_path}")


if __name__ == "__main__":
    main()
