"""
Stage 0 of the pipeline: copy raw scans into a working directory, apply
deskew + preprocessing, and write a manifest.csv in the exact shape
core/classifier.py already expects (one "file_path" column) - pointing
at the WORKING COPIES, not the originals, so classification (and every
stage after it) sees corrected images instead of raw ones.

Built 2026-07-28 per Jon's direction: "build manifest should go into the
CV deskew and preprocessor. then each stage is getting the same image
thats been processed" - and separately, "the images should be copied to
a working directory and not actually touch the originals." core/
classifier.py itself needs ZERO changes for this - it already just reads
whatever manifest.csv it's given, so pointing it at a working-directory
manifest instead of scripts/build_manifest.py's raw-file manifest is the
entire integration; no code downstream of Stage 0 changes.

STAGE ORDER (Jon's direction, 2026-07-28 - narrowed after warp-detection
calibration didn't hold up, see core/warp_detection.py's docstring for
that story):
  1. copy_to_working_dir() - never touches originals
  2. preprocess_for_manifest() - deskew (core/row_segmentation.py's
     already-proven estimate_deskew_angle/apply_deskew_angle) + a
     preprocessing profile (core/image_preprocessing.py) - applies to
     EVERY image, improves classification input quality regardless of
     document type
  3. (separate step, unchanged) core/classifier.py classifies from the
     manifest this module writes
  4. ONLY for the dense_tabular_rows bucket: manual dewarp via the
     EXISTING ui/dewarp_preprocessor_ui.py in its bucket-worklist mode,
     pointed at data/buckets/dense_tabular_rows.csv - no new code needed,
     that mode already exists. Every other bucket skips this entirely -
     a single whole-image model call tolerates mild residual warp fine,
     only the tight per-field crop pipeline (core/row_extraction.py)
     actually depends on precise geometry.
  5. finalize_manifest() (this module) - merges classification buckets +
     the dewarp UI's own output CSV into one final manifest recording
     each file's real, ready-to-use image path.

INGESTION SHAPE (2026-08-02, generalized from PDF-specific handling):
    Source Discovery -> Source Expansion -> Working Manifest -> ...
This module (Working Manifest onward) only ever deals in plain image
paths - it has NO knowledge of PDFs, TIFFs, ZIPs, or any other
expandable source format. That normalization is core/source_expansion.
py's job entirely (expand_source_paths()/EXPANDABLE_EXTENSIONS) -
see that module's docstring for why the split exists and how to add a
future expandable format without touching this file at all.

WARP-DETECTION INSERTION POINT: dense_tabular_rows currently goes to
manual dewarp for EVERY file, unconditionally - see
_dense_tabular_needs_manual_dewarp() below. core/warp_detection.py's
detect_warp() exists and is real, tested code, but calibration against
23 real image pairs from this project didn't hold up (two different
techniques tried, both failed - see that module's docstring for the
full story), so it is NOT called here. When/if a reliable version of
that heuristic exists, _dense_tabular_needs_manual_dewarp() is the ONLY
function that needs to change - swap its unconditional `return True` for
a real detect_warp(image).needs_dewarp check. Nothing else in this
module, core/classifier.py, or ui/dewarp_preprocessor_ui.py needs to
change for that to work, since the manual-dewarp path already goes
through the EXISTING bucket-worklist UI regardless of how a file ends up
queued for it.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from PIL import Image

from core.row_segmentation import estimate_deskew_angle, apply_deskew_angle
from core.image_preprocessing import apply_profile
from core.source_expansion import expand_source_paths, EXPANDABLE_EXTENSIONS
from core.bucket_worklist import (
    load_bucket_filepaths, preprocessed_bucket_path, load_processed_records,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORKING_DIR = PROJECT_ROOT / "data" / "working"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.csv"
BUCKET_DIR = PROJECT_ROOT / "data" / "buckets"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}

# Applied to EVERY image at Stage 0, before classification ever sees it -
# a general legibility improvement (autocontrast + mild sharpening), not
# tuned per document type. Deliberately conservative (not e.g. adaptive_
# threshold or invert, which help specific known-bad cases but can hurt
# otherwise-fine scans) - see core/image_preprocessing.py's own
# PREPROCESSING_PROFILES for the full menu if a different default proves
# better once real before/after classification accuracy is measured
# (matching [[feedback_context_and_preprocessing_improve_labeling]] -
# preprocessing already proven to help labeling quality once before,
# this is applying that same lesson one stage earlier).
DEFAULT_PREPROCESSING_PROFILE = "autocontrast"


def collect_image_paths(folder: Path) -> list[Path]:
    """
    Walks folder for both plain images AND anything with a registered
    expander (core/source_expansion.py's EXPANDABLE_EXTENSIONS -
    currently just .pdf, but this function doesn't know or care which
    formats those are). This function's job stays "find candidate
    Stage 0 source files" - actually normalizing an expandable source
    into image path(s) happens later, in expand_source_paths() (called
    from build_working_manifest_from_paths()), never here.
    """
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS | EXPANDABLE_EXTENSIONS
    )


def copy_to_working_dir(source_paths: list[Path], working_dir: Path) -> dict[Path, Path]:
    """
    Copies every source_paths entry into working_dir, flat (no
    subdirectory structure preserved - collect_image_paths() already
    walked recursively, and a flat working dir keeps every later stage's
    own path handling simple). A name collision (two source files from
    different subfolders sharing a filename) gets a numeric suffix
    rather than silently overwriting one - never lose a source file's
    own working copy to another file's name.

    Returns {source_path: working_copy_path}. The ORIGINAL file at
    source_path is only ever read here, never opened for writing -
    shutil.copy2 preserves metadata but is a read-then-write-elsewhere
    operation, not an in-place modification.
    """
    working_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[Path, Path] = {}
    used_names: set[str] = set()

    for source_path in source_paths:
        dest_name = source_path.name
        stem, suffix = source_path.stem, source_path.suffix
        counter = 1
        while dest_name in used_names:
            dest_name = f"{stem}_{counter}{suffix}"
            counter += 1
        used_names.add(dest_name)

        dest_path = working_dir / dest_name
        shutil.copy2(source_path, dest_path)
        mapping[source_path] = dest_path

    return mapping


def preprocess_for_manifest(
    image_path: Path, profile_name: str = DEFAULT_PREPROCESSING_PROFILE,
) -> tuple[float, str]:
    """
    Deskews + preprocesses the image AT image_path IN PLACE (this is a
    working-directory copy already, per copy_to_working_dir() above -
    never called against an original). Returns (deskew_angle_used,
    profile_name_applied) for the manifest row's own record of what
    happened to each file - same audit-trail discipline as every model
    loader's generation_config_hash.
    """
    with Image.open(image_path) as original:
        image = original.convert("RGB")
        angle = estimate_deskew_angle(image)
        deskewed = apply_deskew_angle(image, angle)
        processed = apply_profile(deskewed, profile_name)
    processed.save(image_path)
    return angle, profile_name


def build_working_manifest_from_paths(
    source_paths: list[Path],
    working_dir: Path = DEFAULT_WORKING_DIR,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    preprocessing_profile: str = DEFAULT_PREPROCESSING_PROFILE,
    capture_baseline: bool = True,
) -> Path:
    """
    Stage 0 end to end, from an EXPLICIT list of source paths - THE real
    engine. Normalizes source_paths through core/source_expansion.py's
    expand_source_paths() first (2026-08-02 - this function has NO
    knowledge of PDFs or any other expandable format; that's the whole
    point of the expansion-layer split, see that module's docstring),
    copies every resulting image into working_dir, deskews + preprocesses
    each working copy, writes manifest_path in the exact shape
    core/classifier.py expects (one "file_path" column, pointing at the
    WORKING copies). Originals are only ever read, never modified -
    true for plain images and for whatever an expander reads too (e.g.
    a PDF).

    capture_baseline (2026-08-02, Jon): if True (default), captures each
    working copy's pre-preprocessing vision-tower baseline embedding
    (core/baseline_embeddings.py, all 8 qualified encoders) BEFORE
    preprocess_for_manifest() runs - i.e. on the raw working copy,
    untouched by deskew/autocontrast. Per docs/BENCHMARK2_METADATA_
    LAYER_QUALIFICATION.md's Third/Fourth extension design: one stable,
    non-drifting reference embedding per image, meant to eventually
    inform a not-yet-built per-image preprocessing-profile decision
    (Stage B) rather than today's fixed default profile for everyone.
    This function only captures and persists (data/baseline_embeddings.
    json) - it does not decide anything yet. Set False to skip (e.g. a
    quick throwaway test run where the extra ~8-model pass isn't wanted).

    Also writes a provenance sidecar (`<manifest stem>_provenance.json`,
    e.g. manifest.csv -> manifest_provenance.json) recording each working
    copy's real origin - every ExpandedSource field (source_path,
    expanded_path, source_type, page_number, metadata) plus the final
    working_path. Not consumed by core/classifier.py or anything else
    downstream (that still reads manifest_path's plain file_path column,
    unchanged) - kept for diagnostics, logging, and future UI features,
    per Jon's explicit direction that this should remain available even
    where nothing consumes it yet.

    Takes a plain list of paths, not a folder or a CSV (Jon's design,
    2026-07-30: "the CSV becomes just one adapter, not part of the
    preprocessing engine itself... the GUI can pass its queued paths
    directly, future tools can construct lists however they like").
    build_working_manifest() below (folder input) is now a thin adapter
    over this, not a separate copy of the same logic - see its own
    docstring. A CSV-input adapter (read a "file_path" column, call this)
    can be added the same way whenever something needs it; none exists
    yet since nothing has asked for one.
    """
    if not source_paths:
        raise ValueError("build_working_manifest_from_paths() got an empty path list.")

    expanded_sources = expand_source_paths(source_paths)
    n_expanded = sum(1 for e in expanded_sources if e.source_type != "image")
    if n_expanded:
        print(f"Expanded {n_expanded} non-image source page(s) "
              f"({', '.join(sorted({e.source_type for e in expanded_sources if e.source_type != 'image'}))})")

    expanded_paths = [e.expanded_path for e in expanded_sources]
    mapping = copy_to_working_dir(expanded_paths, working_dir)
    print(f"Copied to working directory: {working_dir}")

    if capture_baseline:
        from core.baseline_embeddings import write_baseline_embeddings
        print(f"Capturing pre-preprocessing baseline embeddings for {len(mapping)} image(s) "
              f"(8 encoders each - this may take a while)...")
        write_baseline_embeddings(list(mapping.values()))

    for i, (source_path, working_path) in enumerate(mapping.items(), 1):
        print(f"[{i}/{len(mapping)}] Preprocessing {working_path.name}...")
        try:
            angle, profile = preprocess_for_manifest(working_path, preprocessing_profile)
            print(f"  deskew_angle={angle:+.2f}  profile={profile!r}")
        except Exception as e:
            print(f"  WARNING: preprocessing failed ({type(e).__name__}: {e}) - "
                  f"working copy left as an unprocessed straight copy of the original.")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_path"])
        for working_path in mapping.values():
            writer.writerow([str(working_path)])

    provenance_path = manifest_path.with_name(f"{manifest_path.stem}_provenance.json")
    provenance_records = []
    for expanded_source in expanded_sources:
        record = expanded_source.to_dict()
        record["working_path"] = str(mapping[expanded_source.expanded_path])
        provenance_records.append(record)
    with open(provenance_path, "w", encoding="utf-8") as f:
        json.dump(provenance_records, f, indent=2)

    print(f"\nWorking manifest written to {manifest_path} ({len(mapping)} file(s)).")
    print(f"Provenance written to {provenance_path}")
    print("Next step (unchanged):")
    print(f"  python -m core.classifier {manifest_path}")
    return manifest_path


def build_working_manifest(
    source_folder: Path,
    working_dir: Path = DEFAULT_WORKING_DIR,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    preprocessing_profile: str = DEFAULT_PREPROCESSING_PROFILE,
) -> Path:
    """
    Stage 0 from a FOLDER - a thin adapter over
    build_working_manifest_from_paths(): walks source_folder
    (collect_image_paths()) and delegates. This is what
    scripts/build_working_manifest.py's CLI still calls; kept as its own
    function rather than removed since that script and its docstring are
    folder-oriented, not because it does anything build_working_manifest_
    from_paths() doesn't already do.
    """
    source_paths = collect_image_paths(source_folder)
    if not source_paths:
        raise FileNotFoundError(
            f"No image or expandable source files found in {source_folder} "
            f"(looked for {sorted(IMAGE_EXTENSIONS | EXPANDABLE_EXTENSIONS)})"
        )
    n_expandable = sum(1 for p in source_paths if p.suffix.lower() in EXPANDABLE_EXTENSIONS)
    print(f"Found {len(source_paths)} source file(s) in {source_folder}"
          + (f" ({n_expandable} will be expanded)" if n_expandable else ""))
    return build_working_manifest_from_paths(
        source_paths, working_dir=working_dir, manifest_path=manifest_path,
        preprocessing_profile=preprocessing_profile,
    )


# -- Stage 2: final merge, after classification + (for dense_tabular_rows) manual dewarp --

FINAL_MANIFEST_FIELDS = ["file_path", "category", "status", "source_original_path"]


def _dense_tabular_needs_manual_dewarp(file_path: str) -> bool:
    """
    THE WARP-DETECTION INSERTION POINT - see module docstring. Always
    True today: every dense_tabular_rows file is queued for manual
    dewarp, unconditionally, because core/warp_detection.py's automated
    signal did not survive real calibration (two techniques tried, both
    failed against 23 real image pairs - see that module's docstring).

    file_path is accepted (not just ignored) so a future real
    implementation can open the image and call
    core.warp_detection.detect_warp() without this function's
    signature needing to change - only the body does.
    """
    return True


def finalize_manifest(
    bucket_dir: Path = BUCKET_DIR,
    output_path: Path | None = None,
) -> Path:
    """
    Stage 2: reads every data/buckets/<category>.csv (core/classifier.py's
    output) and, for dense_tabular_rows specifically, that bucket's own
    dewarped-preprocessed-bucket CSV (written by ui/dewarp_preprocessor_ui
    .py's EXISTING bucket-worklist mode - no new code produces this file,
    see module docstring) to build one final manifest recording each
    file's real, ready-to-use image path.

    Three possible statuses per row:
      "ready" - not dense_tabular_rows, or dense_tabular_rows with no
                manual-dewarp requirement (see _dense_tabular_needs_
                manual_dewarp - currently unreachable, always True today,
                kept for when that changes) - file_path is already the
                final, usable image.
      "ready_dewarped" - dense_tabular_rows, manual dewarp completed -
                file_path is the dewarped output from ui/dewarp_
                preprocessor_ui.py's own preprocessed-bucket CSV.
      "pending_manual_dewarp" - dense_tabular_rows, queued for manual
                dewarp, not done yet - file_path is still the working-
                directory (deskewed+preprocessed but not dewarped) copy;
                downstream stages should NOT consume rows in this status.

    Never raises on a missing bucket CSV or missing dewarped-bucket CSV -
    both are legitimate "nothing done in that bucket/stage yet" states,
    not errors (same reasoning as bucket_worklist.py's own
    load_processed_records() returning [] for a not-yet-existing file).
    """
    if output_path is None:
        output_path = PROJECT_ROOT / "data" / "manifest_final.csv"

    rows: list[dict] = []

    for bucket_csv in sorted(bucket_dir.glob("*.csv")):
        if bucket_csv.stem.endswith("_dewarped") or bucket_csv.stem.endswith("_final"):
            continue  # a stage's own OUTPUT csv, not a classifier bucket
        category = bucket_csv.stem

        try:
            file_paths = load_bucket_filepaths(bucket_csv)
        except (FileNotFoundError, OSError):
            continue

        if category != "dense_tabular_rows":
            for fp in file_paths:
                rows.append({
                    "file_path": fp, "category": category,
                    "status": "ready", "source_original_path": "",
                })
            continue

        dewarped_csv = preprocessed_bucket_path(bucket_csv, "dewarped")
        dewarped_records = {
            r["source_file_path"]: r for r in load_processed_records(dewarped_csv)
        }

        for fp in file_paths:
            record = dewarped_records.get(fp)
            if record is not None:
                rows.append({
                    "file_path": record["output_file_path"], "category": category,
                    "status": "ready_dewarped", "source_original_path": fp,
                })
            elif _dense_tabular_needs_manual_dewarp(fp):
                rows.append({
                    "file_path": fp, "category": category,
                    "status": "pending_manual_dewarp", "source_original_path": "",
                })
            else:
                rows.append({
                    "file_path": fp, "category": category,
                    "status": "ready", "source_original_path": "",
                })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FINAL_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    pending = sum(1 for r in rows if r["status"] == "pending_manual_dewarp")
    print(f"Final manifest written to {output_path} ({len(rows)} file(s), "
          f"{pending} pending manual dewarp).")
    if pending:
        print(
            f"  {pending} dense_tabular_rows file(s) need manual dewarp before they're "
            f"ready - run: python ui/dewarp_preprocessor_ui.py "
            f"(Load bucket CSV... -> {bucket_dir / 'dense_tabular_rows.csv'}), "
            f"then re-run this finalize step."
        )
    return output_path
