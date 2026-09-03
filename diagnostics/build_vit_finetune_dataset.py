"""
Builds the combined training manifest for the ViT-21k full-page document-
type fine-tune, per Jon's direction (2026-08-07): supplement
`manifest_final.csv`'s existing `dense_tabular_rows` examples with the
~484 real LAC census page images Jon pulled and hand-marked (blank scans
and title cards excluded per his own review file), keeping every other
category exactly as manifest_final.csv already has it. IAM line crops are
explicitly EXCLUDED from this run (Jon's call: crop-scale mismatch vs.
full-page images would confound a full 8-way classifier - revisit for a
separate line-level/binary handwriting sensor later).

Ground-truth-exclusions file (Jon's manual review, verbatim):
    data/outputs/lac_census_pull/lac_pull ground_truth.txt
Format: a source folder path on its own line, then zero or more
"<filename>\t<reason>" lines naming files to EXCLUDE from that folder
(blank scans / title cards), blank line separates folders.

Usage:
    python diagnostics/build_vit_finetune_dataset.py
Writes: data/outputs/vit_finetune_dataset.csv (file_path, category)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MANIFEST_PATH = PROJECT_ROOT / "data" / "outputs" / "reference_pipeline_v2" / "manifest_final.csv"
EXCLUSIONS_TXT = PROJECT_ROOT / "data" / "outputs" / "lac_census_pull" / "lac_pull ground_truth.txt"
OUT_CSV = PROJECT_ROOT / "data" / "outputs" / "vit_finetune_dataset.csv"

CENSUS_CATEGORY = "dense_tabular_rows"
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def parse_exclusions(txt_path: Path) -> dict[str, set[str]]:
    """Returns {folder_path_str: {excluded_filename, ...}}."""
    folders: dict[str, set[str]] = {}
    current_folder = None
    for raw_line in txt_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        # a folder header line looks like a bare Windows path with no tab-separated reason
        if "\t" not in line and (stripped.startswith("J:\\") or stripped.startswith("J:/")):
            current_folder = stripped.rstrip("\\/")
            folders.setdefault(current_folder, set())
            continue
        if current_folder is not None and "\t" in line:
            fname = line.split("\t", 1)[0].strip()
            if fname:
                folders[current_folder].add(fname)
    return folders


def collect_census_pages(folders: dict[str, set[str]]) -> list[str]:
    kept: list[str] = []
    for folder_str, excluded in folders.items():
        folder = Path(folder_str)
        if not folder.is_dir():
            print(f"  WARNING: folder missing, skipping: {folder}")
            continue
        n_kept = 0
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() not in IMAGE_EXTS:
                continue
            if f.name in excluded:
                continue
            kept.append(str(f))
            n_kept += 1
        print(f"  {folder}: {n_kept} kept, {len(excluded)} excluded")
    return kept


def load_manifest_final() -> list[tuple[str, str]]:
    rows = []
    with open(MANIFEST_PATH, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["category"] == "uncertain_review":
                continue
            if Path(r["file_path"]).exists():
                rows.append((r["file_path"], r["category"]))
    return rows


def main():
    print(f"Parsing exclusions from: {EXCLUSIONS_TXT}")
    folders = parse_exclusions(EXCLUSIONS_TXT)
    for folder, excluded in folders.items():
        print(f"  folder={folder} excluded_count={len(excluded)}")
    print("\nCollecting real census page images...")
    census_pages = collect_census_pages(folders)
    print(f"Total real census pages kept: {len(census_pages)}\n")

    print(f"Loading manifest_final.csv (existing taxonomy ground truth)...")
    manifest_rows = load_manifest_final()
    print(f"  {len(manifest_rows)} rows loaded (existing files only)\n")

    combined: list[tuple[str, str]] = list(manifest_rows)
    combined.extend((p, CENSUS_CATEGORY) for p in census_pages)

    # de-dupe by file_path in case of overlap between the two sources
    seen = set()
    deduped = []
    for path, cat in combined:
        if path in seen:
            continue
        seen.add(path)
        deduped.append((path, cat))

    from collections import Counter
    counts = Counter(cat for _, cat in deduped)
    print("Final combined dataset category counts:")
    for cat, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {cat:<20s} {n}")
    print(f"\nTotal: {len(deduped)} images ({len(combined) - len(deduped)} duplicates removed)")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file_path", "category"])
        w.writerows(deduped)
    print(f"\nWritten: {OUT_CSV}")


if __name__ == "__main__":
    main()
