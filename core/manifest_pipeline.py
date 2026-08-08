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

STAGE TERMINOLOGY (2026-08-02): this module spans THREE stages per
docs/PIPELINE_STAGE_TERMINOLOGY.md's canonical Stage 0-6 naming -
Stage 0 (Source Acquisition), Stage 1 (Raw Sensor Capture, semantic
half), Stage 3 (Image Processing). Originally all three were conflated
inside one function (build_working_manifest_from_paths()), which is
exactly what caused the pre/post-preprocessing mislabeling documented
in docs/REFERENCE_PIPELINE_V1.md - a fresh Stage 1 capture run,
separately, against an already-Stage-3'd corpus silently measured
processed images while tagging them "pre_preprocessing".

SPLIT (2026-08-02, second restructuring pass): each stage is now its
own independently-callable function, reading/writing only through
manifest_path's "file_path" column rather than an in-memory dict
handed between them:
  - stage0_acquire_and_copy_sources() - expansion + copy + writes
    manifest.csv + provenance sidecar. No pixel modification.
  - stage1_capture_baseline_embeddings() - reads manifest_path, embeds
    whatever pixels are AT those paths right now via core/
    baseline_embeddings.py. Does not enforce call order - it measures
    whatever's there when called.
  - stage3_preprocess_manifest() - reads manifest_path, deskews +
    preprocesses each file IN PLACE.
build_working_manifest_from_paths() is now a thin composition wrapper
(stage0 -> stage1 if capture_baseline -> stage3) kept for backward
compatibility - every existing caller (build_working_manifest(),
scripts/run_preprocessing.py, ui/build_manifest_ui.py's subprocess CLI)
still gets identical behavior and identical printed progress lines.
The real fix this unlocks: stage1/stage3 can now be called directly,
independently, against any manifest_path - e.g. capturing Stage 1
against a manifest Stage 0 already built in an earlier process, without
re-running Stage 0, which is what "genuinely separate" required and
what one bundled function's fixed call order could not offer.

A deliberate ordering change from the pre-split version: manifest.csv
is now written as part of Stage 0, immediately after copying (before
Stage 1 or Stage 3 ever run) rather than at the very end after
preprocessing. The file's CONTENTS are identical either way (working
copy paths don't change when preprocess_for_manifest() rewrites pixels
at those same paths) - but writing it immediately makes the manifest
Stage 0's actual deliverable, which is what stage1/stage3 need to be
callable against it later on their own.

DB WIRING + STAGE 2 (2026-08-02, third restructuring pass): stage0/1/3
now DUAL-WRITE into core/pipeline_db.py's PipelineDatabase (data/
pipeline.db) alongside every existing CSV/JSON output - see
docs/PIPELINE_DATABASE.md. This is additive, not a replacement: nothing
downstream has been migrated to READ from the DB instead of its CSV
yet, so every existing consumer is unaffected. build_working_manifest_
from_paths() now also composes in core/decision_engine.py's
stage2_decide_profiles() (Stage 2, Decision Engine) between Stage 1 and
Stage 3, by default (run_decision_engine: bool = True) - see that
module's own docstring for what it does and does NOT decide yet.

That normalization (PDFs, TIFFs, ZIPs, or any other expandable source
format) is core/source_expansion.py's job entirely
(expand_source_paths()/EXPANDABLE_EXTENSIONS) - see that module's
docstring for why the split exists and how to add a future expandable
format without touching this file at all.

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

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Optional

from PIL import Image

from core.row_segmentation import estimate_deskew_angle, apply_deskew_angle
from core.image_preprocessing import apply_profile
from core.source_expansion import expand_source_paths, EXPANDABLE_EXTENSIONS
from core.bucket_worklist import load_bucket_filepaths
from core.pipeline_db import (
    PipelineDatabase, hash_file, DEFAULT_DB_PATH,
    sync_bucket_classifications, sync_dewarp_results,
)
from core.workspace_context import WorkspaceContext
from core.run_context import RUN_TYPES, RunContext
# load_bucket_filepaths() (reads a plain "file_path" CSV column) is what
# lets stage1_capture_baseline_embeddings()/stage3_preprocess_manifest()
# below take just a manifest_path rather than an in-memory path mapping -
# the same reader every other stage in this project already uses.
#
# core.pipeline_db is stdlib-only (sqlite3/hashlib/pathlib/datetime) - a
# safe, cheap top-level import, unlike core.baseline_embeddings (torch/
# timm) which stays lazily imported inside stage1_capture_baseline_
# embeddings() below. DB writes here are DUAL-WRITE, additive alongside
# every existing CSV/JSON output - see docs/PIPELINE_DATABASE.md's
# "Phase 3" migration risk #6 (not-yet-migrated tools still read the old
# files directly, so those files can't stop being written).

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _legacy_run_root() -> Path:
    """Resolves to the legacy_pre_run_system synthetic run (see
    scripts/migrate_pipeline_db_to_run_schema.py) if the workspace
    migration has run; otherwise falls back to the pre-migration
    PROJECT_ROOT/data/ layout. This indirection exists ONLY for the
    legacy default-path constants below - every new call site should
    pass ctx= (a real RunContext) instead of relying on these."""
    ws = WorkspaceContext.resolve()
    legacy_root = ws.runs_root / "legacy_pre_run_system"
    if legacy_root.exists():
        return legacy_root
    return None


_legacy_root = _legacy_run_root()

# Legacy fallbacks ONLY - used when a caller doesn't pass a RunContext
# (ctx=). Every call site in this repo should be migrating to pass ctx=
# explicitly (see RunContext.create()/resume() in core/run_context.py).
# Resolve into the legacy_pre_run_system run once it exists (post-
# migration); until then, fall back to the pre-migration data/ layout
# so nothing breaks mid-transition.
if _legacy_root is not None:
    DEFAULT_WORKING_DIR = _legacy_root / "working" / "images"
    DEFAULT_MANIFEST_PATH = _legacy_root / "manifest" / "manifest.csv"
    BUCKET_DIR = _legacy_root / "outputs" / "buckets"
else:
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


def preprocess_sidecar_path(image_path: Path) -> Path:
    """Stage 3's own evidence sidecar - <name>_preprocess.json, alongside
    Stage 1's <name>_analysis.json (core/image_analysis.py's
    analysis_sidecar_path()). Records what Stage 3 actually DID (angle
    applied, why, which profile) - previously only ever printed to
    stdout and lost. Added 2026-08-05 alongside the deskew_angle_clamped
    fix specifically so "how often does the clamped-zeroed case happen,
    and for which document types" can be answered later by querying
    these sidecars (or their stage_outputs DB rows) - never touching
    pixels again."""
    return image_path.with_name(f"{image_path.stem}_preprocess.json")


def _resolve_deskew_angle(image_path: Path) -> tuple[float, str, float | None]:
    """
    Reuses Stage 1's already-measured deskew_angle_deg (core/
    image_analysis.py's *_analysis.json sidecar) instead of
    re-estimating - 2026-08-03 consolidation pass. Valid ONLY because
    this is called from preprocess_for_manifest() below, which runs
    BEFORE any dewarp - i.e. against the exact same working-copy pixels
    Stage 1's physical sensor measured, not a later derivative (unlike
    core/auto_sidecar.py's own deskew estimate, which runs against the
    DEWARPED output - a genuinely different image with its own residual
    skew, and deliberately NOT consolidated here for that reason).

    Falls back to a fresh estimate when no sidecar exists yet - the
    common case today, since core/image_analysis.py's physical sensor
    is not yet wired into build_working_manifest_from_paths()'s
    automatic Stage 0->1->2->3 chain and must currently be run as a
    separate step. The fallback uses image_analysis.DESKEW_ANGLE_RANGE
    (15.0), NOT estimate_deskew_angle()'s own narrower 5.0 default -
    using two different ranges for the same physical property was the
    actual bug this pass closes (confirmed against the real corpus:
    32/1974 images, 1.6%, have a true skew Stage 3's old 5.0-range
    estimate would have silently clamped).

    Clamped-estimate override (2026-08-05): deskew_angle_clamped=True was
    documented as "the real angle is AT LEAST this, a floor not a
    measurement" - true for the microfilm page that comment was written
    about, but confirmed FALSE for a real, non-trivial slice of the
    corpus (34/1750 working images, mostly Screenshots - map_land_record/
    genealogy_chart/website_screenshot buckets). Those images lack the
    periodic text-row structure estimate_deskew_angle()'s projection-
    profile search assumes, so the search has no real interior peak and
    just climbs to the +/-15 boundary rather than reflecting a genuine
    15-degree skew.

    Deliberately does NOT try to recover a corrected angle from
    core/image_analysis.py's independent ruling-line Hough measurement
    (dominant_horizontal/vertical_angle_deg) - tried that first (three
    escalating guards: a minimum ruling-line count, then also requiring
    table_confidence==1.0), and direct visual round-trip checks against
    all 34 affected images' raw pre-rotation sources (2026-08-05) kept
    finding new ways for it to fail even after each guard: a torn page
    corner mistaken for a ruling line (1 line, no real guard needed
    since it was caught by the count check); two axes independently
    agreeing on a wrong angle because BOTH were fooled by the same real
    diagonal artifact in the image content, not genuine skew (caught by
    nothing - cross-axis agreement isn't proof); and, decisively, two
    Google-Maps-style screenshots where a real, confidently-detected
    straight road (table_confidence=1.0, the strongest gate tried) was
    measured accurately but is irrelevant to the SCREENSHOT's own frame
    orientation, since a road's angle in a top-down satellite photo
    reflects real-world geography, not page skew. Of the 6 cases that
    survived every guard, 5 were visually confirmed WRONG this way; only
    "Screenshot 2026-05-03 182722" (a genealogy-chart screenshot with
    real card-grid structure) was confirmed correct. That's not a
    reliable enough signal to generalize from - 0.0 (no rotation) was
    the correct answer for every other clamped case checked, including
    every one where the search clamped at the full +/-15 boundary, so
    it's the safe default across the board. A future, better fix would
    need a genuine table/document-vs-photo signal (e.g. bucket
    classification) gating which images even attempt a ruling-line
    fallback - not implemented here.

    Returns (angle, status, raw_estimate_deg) - status is one of
    "normal" (sidecar existed, not clamped, used as measured),
    "clamped_zeroed" (sidecar existed and was clamped, angle forced to
    0.0 per the above), or "no_sidecar_fresh_estimate" (no Stage 1
    sidecar existed at all - the common case today per the fallback
    paragraph above - a fresh angle_range=15 estimate was made instead,
    itself never checked for clamping). raw_estimate_deg is the
    pre-override deskew_angle_deg for "clamped_zeroed" rows (None
    otherwise) - kept so telemetry can show what the runaway search
    actually returned, for later analysis of whether a future algorithm
    fix would have handled it differently.
    """
    from core.image_analysis import analysis_sidecar_path, load_analysis, DESKEW_ANGLE_RANGE

    sidecar_path = analysis_sidecar_path(image_path)
    if sidecar_path.exists():
        try:
            analysis = load_analysis(sidecar_path)
            if analysis.deskew_angle_clamped:
                return 0.0, "clamped_zeroed", analysis.deskew_angle_deg
            return analysis.deskew_angle_deg, "normal", None
        except Exception:
            pass  # schema mismatch or unreadable sidecar - fall through and measure fresh

    with Image.open(image_path) as original:
        angle = float(estimate_deskew_angle(original.convert("RGB"), angle_range=DESKEW_ANGLE_RANGE))
    return angle, "no_sidecar_fresh_estimate", None


def preprocess_for_manifest(
    image_path: Path, profile_name: str = DEFAULT_PREPROCESSING_PROFILE,
) -> tuple[float, str]:
    """
    Deskews + preprocesses the image AT image_path IN PLACE (this is a
    working-directory copy already, per copy_to_working_dir() above -
    never called against an original). Returns (deskew_angle_used,
    profile_name_applied) for the manifest row's own record of what
    happened to each file - same audit-trail discipline as every model
    loader's generation_config_hash. Public return contract kept exactly
    as-is (benchmark/benchmark2_pilot_isolated.py unpacks this 2-tuple
    directly) even though _resolve_deskew_angle() now reports a third
    value (status) - that richer detail is written to
    preprocess_sidecar_path(image_path) as a side effect instead
    (2026-08-05, alongside the deskew_angle_clamped telemetry request),
    not threaded through the return value.
    """
    angle, status, raw_estimate = _resolve_deskew_angle(image_path)
    with Image.open(image_path) as original:
        image = original.convert("RGB")
        deskewed = apply_deskew_angle(image, angle)
        processed = apply_profile(deskewed, profile_name)
    processed.save(image_path)

    preprocess_sidecar_path(image_path).write_text(
        json.dumps({
            "deskew_angle_applied": angle,
            "deskew_status": status,
            "raw_deskew_estimate_deg": raw_estimate,
            "profile_applied": profile_name,
        }, indent=2),
        encoding="utf-8",
    )
    return angle, profile_name


def stage0_acquire_and_copy_sources(
    source_paths: list[Path],
    working_dir: Path = DEFAULT_WORKING_DIR,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    db_path: Path = DEFAULT_DB_PATH,
    ctx: Optional[RunContext] = None,
) -> Path:
    """
    ctx (preferred over the explicit path args above, added for the
    hashed-run migration - see docs/RUN_ARCHITECTURE.md): when given, a
    RunContext, overrides working_dir/manifest_path/db_path with
    ctx.working_images / ctx.manifest_csv / ctx.workspace.pipeline_db_path
    so this run's output is isolated under its own run directory instead
    of the shared legacy default location. The explicit path args remain
    for callers not yet passing ctx.

    Stage 0 (Source Acquisition) ONLY. Normalizes source_paths through
    core/source_expansion.py's expand_source_paths() first (this
    function has NO knowledge of PDFs or any other expandable format;
    that's the whole point of the expansion-layer split, see that
    module's docstring), copies every resulting image into working_dir,
    writes manifest_path in the exact shape core/classifier.py expects
    (one "file_path" column, pointing at the WORKING copies - not yet
    touched by any preprocessing), and a provenance sidecar (`<manifest
    stem>_provenance.json`, e.g. manifest.csv -> manifest_provenance.json)
    recording each working copy's real origin - every ExpandedSource
    field (source_path, expanded_path, source_type, page_number,
    metadata) plus the final working_path. Not consumed by
    core/classifier.py or anything else downstream (that still reads
    manifest_path's plain file_path column, unchanged) - kept for
    diagnostics, logging, and future UI features.

    Originals are only ever read, never modified - true for plain
    images and for whatever an expander reads too (e.g. a PDF).

    Does NOT capture embeddings and does NOT preprocess/modify a single
    pixel - call stage1_capture_baseline_embeddings(manifest_path)
    and/or stage3_preprocess_manifest(manifest_path) afterward for
    those, independently, in whichever order/combination you need. See
    the module docstring's "SPLIT (2026-08-02...)" section for why
    these are now separate functions instead of one bundled call.

    db_path (2026-08-02): DUAL-WRITE into core/pipeline_db.py's
    PipelineDatabase alongside manifest.csv/the provenance sidecar - one
    get_or_create_image() per working copy (identity_hash computed HERE,
    on the freshly-copied, not-yet-preprocessed file - the one place
    it's ever computed, see docs/PIPELINE_DATABASE.md), plus one
    stage_outputs row pointing at the provenance sidecar. This does NOT
    replace manifest.csv/the sidecar - every existing reader of those
    files is untouched; the DB is an additional, queryable record of the
    same event, not yet the source of truth for anything downstream.

    Returns manifest_path.
    """
    if not source_paths:
        raise ValueError("stage0_acquire_and_copy_sources() got an empty path list.")

    if ctx is not None:
        working_dir = ctx.working_images
        manifest_path = ctx.manifest_csv
        db_path = ctx.workspace.pipeline_db_path

    expanded_sources = expand_source_paths(
        source_paths, pdf_output_dir=ctx.raw_from_pdf if ctx is not None else None
    )
    n_expanded = sum(1 for e in expanded_sources if e.source_type != "image")
    if n_expanded:
        print(f"Expanded {n_expanded} non-image source page(s) "
              f"({', '.join(sorted({e.source_type for e in expanded_sources if e.source_type != 'image'}))})")

    expanded_paths = [e.expanded_path for e in expanded_sources]
    mapping = copy_to_working_dir(expanded_paths, working_dir)
    print(f"Copied to working directory: {working_dir}")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_path"])
        for working_path in mapping.values():
            writer.writerow([str(working_path)])

    provenance_path = ctx.manifest_provenance if ctx is not None else (
        manifest_path.with_name(f"{manifest_path.stem}_provenance.json")
    )
    provenance_records = []
    for expanded_source in expanded_sources:
        record = expanded_source.to_dict()
        record["working_path"] = str(mapping[expanded_source.expanded_path])
        provenance_records.append(record)
    with open(provenance_path, "w", encoding="utf-8") as f:
        json.dump(provenance_records, f, indent=2)

    print(f"Stage 0 manifest written to {manifest_path} ({len(mapping)} file(s)).")
    print(f"Provenance written to {provenance_path}")

    db = PipelineDatabase(db_path)
    for expanded_source in expanded_sources:
        working_path = mapping[expanded_source.expanded_path]
        # DB identity is normalized through ctx (RunContext.to_relative())
        # regardless of what manifest.csv itself stores - see
        # docs/RUN_ARCHITECTURE.md. working_path/sidecar_path are
        # run-owned artifacts, so they're persisted run-relative;
        # source_path is the external original (outside the run root)
        # and is never relativized.
        db_working_path = ctx.to_relative(working_path) if ctx is not None else str(working_path)
        db_sidecar_path = ctx.to_relative(provenance_path) if ctx is not None else str(provenance_path)
        image_id = db.get_or_create_image(
            source_path=str(expanded_source.source_path),
            working_path=db_working_path,
            source_type=expanded_source.source_type,
            page_number=expanded_source.page_number,
            identity_hash=hash_file(working_path) if ctx is not None else None,
            run_id=ctx.run_id if ctx is not None else None,
        )
        db.record_stage_output(
            image_id, stage="stage0_acquire",
            sidecar_path=db_sidecar_path, status="done",
        )
    print(f"Recorded {len(expanded_sources)} image(s) in {db.db_path}")

    return manifest_path


def stage1_capture_baseline_embeddings(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    db_path: Path = DEFAULT_DB_PATH,
    ctx: Optional[RunContext] = None,
) -> Path:
    """
    ctx (hashed-run migration): when given, db_path is overridden with
    ctx.workspace.pipeline_db_path, and every DB lookup below is
    normalized through ctx.to_relative()/scoped by ctx.run_id instead of
    matching manifest_path's absolute file_path strings directly against
    the DB's (now run-relative) working_path column - see
    docs/RUN_ARCHITECTURE.md.

    Stage 1 (Raw Sensor Capture, semantic half) ONLY. Reads
    manifest_path's "file_path" column and captures each listed image's
    vision-tower baseline embedding (core/baseline_embeddings.py, all 8
    qualified encoders) - for whatever pixels are AT THOSE PATHS RIGHT
    NOW. Per docs/BENCHMARK2_METADATA_LAYER_QUALIFICATION.md's Third/
    Fourth extension design: one stable, non-drifting reference
    embedding per image, meant to eventually inform a not-yet-built
    per-image preprocessing-profile decision (Stage 2, Decision Engine)
    rather than today's fixed default profile for everyone. This
    function only captures and persists (data/baseline_embeddings.json)
    - it does not decide anything yet.

    Deliberately does NOT check or enforce ordering relative to
    stage3_preprocess_manifest() - it measures whatever's there when
    called. Call this BEFORE stage3_preprocess_manifest() if genuine
    pre-preprocessing embeddings are wanted (the normal case) - calling
    it after Stage 3 has already run is exactly the caller-ordering
    mistake documented in docs/REFERENCE_PIPELINE_V1.md (an
    already-preprocessed corpus measured and mislabeled
    "pre_preprocessing"). Splitting this out of one bundled function is
    what makes that mistake visible/avoidable at the call site instead
    of hidden inside fixed internal call order.

    db_path (2026-08-02): for each image, records current_stage=1,
    status="sensed" in the DB, plus a stage_outputs row (lookup_key=
    the image's own identity_hash, since data/baseline_embeddings.json
    is ONE file shared across the whole corpus, not a per-image
    sidecar - see docs/PIPELINE_DATABASE.md). A path not already known
    to the DB (e.g. this function called standalone against a manifest
    Stage 0 never registered) is skipped with a printed warning rather
    than silently creating a partial row - get_or_create_image() is
    Stage 0's job, not this function's.

    Returns manifest_path unchanged (embeddings are written to
    core/baseline_embeddings.py's own DEFAULT_BASELINE_PATH, a separate
    file - this function never modifies the manifest itself).
    """
    from core.baseline_embeddings import write_baseline_embeddings

    if ctx is not None:
        db_path = ctx.workspace.pipeline_db_path

    paths = [Path(p) for p in load_bucket_filepaths(manifest_path)]
    print(f"Capturing baseline embeddings for {len(paths)} image(s) "
          f"(8 encoders each - this may take a while)...")
    if ctx is not None:
        write_baseline_embeddings(paths, output_path=ctx.embeddings / "baseline_embeddings.json")
    else:
        write_baseline_embeddings(paths)

    db = PipelineDatabase(db_path)
    recorded, skipped = 0, 0
    for p in paths:
        lookup = ctx.to_relative(p) if ctx is not None else str(p)
        image = db.get_image_by_path(lookup, run_id=ctx.run_id if ctx is not None else None)
        if image is None:
            skipped += 1
            continue
        db.update_image_state(image["id"], current_stage=1, status="sensed")
        db.record_stage_output(
            image["id"], stage="stage1_baseline_embeddings",
            lookup_key=image["identity_hash"], status="done",
        )
        recorded += 1
    print(f"Recorded {recorded} image(s) in {db.db_path}"
          + (f" ({skipped} not found in DB - skipped)" if skipped else ""))

    return manifest_path


def stage1_capture_layout_detections(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    db_path: Path = DEFAULT_DB_PATH,
    ctx: Optional[RunContext] = None,
) -> Path:
    """
    ctx: same DB-identity-normalization override as stage1_capture_
    baseline_embeddings() - see that function's docstring.

    Stage 1 (Raw Sensor Capture, semantic half) - the layout-detection
    sensor added 2026-08-04 (core/layout_detector.py, DocLayout-YOLO/
    DocStructBench), alongside stage1_capture_baseline_embeddings()'s
    8-encoder vision-tower battery. Reads manifest_path's "file_path"
    column and detects each listed image's structured layout regions
    (title/text/table/figure/... bounding boxes) - for whatever pixels
    are AT THOSE PATHS RIGHT NOW, same "measures, doesn't decide"
    contract as every other Stage 1 function. Written into the SAME file
    stage1_capture_baseline_embeddings() writes to (data/baseline_
    embeddings.json, via core.baseline_embeddings.merge_sensor_records())
    under a "layout_detections" key per image, not a separate file - "any
    information we can gather that could assist the CV stages
    (auto_rows/auto_columns) should be recorded the same as the other
    sensor tower data" was the explicit direction this followed.

    Genuinely independent of stage1_capture_baseline_embeddings() -
    neither calls the other, and build_working_manifest_from_paths()
    below gates them with separate flags (capture_baseline,
    capture_layout) so either can run without the other.

    db_path: same DB-recording contract as stage1_capture_baseline_
    embeddings() (current_stage=1, status="sensed", a stage_outputs row
    keyed by identity_hash) under stage="stage1_layout_detections" - a
    DISTINCT stage_outputs row from "stage1_baseline_embeddings" for the
    same image, since these are two independent sensor captures, not one
    combined stage. A path not already known to the DB is skipped with a
    printed warning, same reasoning as stage1_capture_baseline_
    embeddings().

    Returns manifest_path unchanged (layout detections are written to
    core/baseline_embeddings.py's DEFAULT_BASELINE_PATH - this function
    never modifies the manifest itself).
    """
    from core.baseline_embeddings import write_layout_detections

    if ctx is not None:
        db_path = ctx.workspace.pipeline_db_path

    paths = [Path(p) for p in load_bucket_filepaths(manifest_path)]
    print(f"Capturing layout detections for {len(paths)} image(s) "
          f"(doclayout_yolo - this may take a while)...")
    if ctx is not None:
        # Same per-run file stage1_capture_baseline_embeddings() writes
        # to when ctx is set - merge_sensor_records() merges into it
        # under a "layout_detections" key per image, same convention as
        # the shared-file (non-ctx) default.
        write_layout_detections(paths, output_path=ctx.embeddings / "baseline_embeddings.json")
    else:
        write_layout_detections(paths)

    db = PipelineDatabase(db_path)
    recorded, skipped = 0, 0
    for p in paths:
        lookup = ctx.to_relative(p) if ctx is not None else str(p)
        image = db.get_image_by_path(lookup, run_id=ctx.run_id if ctx is not None else None)
        if image is None:
            skipped += 1
            continue
        db.update_image_state(image["id"], current_stage=1, status="sensed")
        db.record_stage_output(
            image["id"], stage="stage1_layout_detections",
            lookup_key=image["identity_hash"], status="done",
        )
        recorded += 1
    print(f"Recorded {recorded} image(s) in {db.db_path}"
          + (f" ({skipped} not found in DB - skipped)" if skipped else ""))

    return manifest_path


def stage3_preprocess_manifest(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    preprocessing_profile: str = DEFAULT_PREPROCESSING_PROFILE,
    db_path: Path = DEFAULT_DB_PATH,
    ctx: Optional[RunContext] = None,
) -> Path:
    """
    ctx: same DB-identity-normalization override as stage1_capture_
    baseline_embeddings() - lookups AND the sidecar_path written into
    stage_outputs both go through ctx.to_relative()/ctx.run_id when set.

    Stage 3 (Image Processing) ONLY. Reads manifest_path's "file_path"
    column and deskews + preprocesses each listed image IN PLACE (see
    preprocess_for_manifest() above). Paths don't change - preprocessing
    rewrites the pixels at each existing path, it doesn't move or rename
    anything - so this never touches the manifest CSV itself.

    A failure on one file is a printed WARNING, not a raised exception -
    that file's working copy is simply left as an unprocessed straight
    copy of the original, and the loop continues (matching the
    pre-split behavior; one bad image shouldn't abort an entire batch).

    db_path (2026-08-02): on success, recomputes current_hash (pixels
    genuinely changed) and records current_stage=3, status=
    "preprocessed", processing_profile=preprocessing_profile, plus a
    stage_outputs row whose status mirrors preprocess_sidecar_path()'s
    deskew_status (2026-08-05) - "done" for the normal case, or
    "clamped_zeroed" when _resolve_deskew_angle() had to override a
    runaway projection-profile estimate. This is what makes the deskew-
    clamp failure mode queryable later without touching any pixels: how
    often it happens (COUNT stage_outputs WHERE stage='stage3_preprocess'
    AND status='clamped_zeroed'), which document types trigger it (JOIN
    to images.bucket), and whether a future estimator fix resolves it
    (compare the count across a re-run). sidecar_path points at the new
    per-image *_preprocess.json evidence file. On failure, records a
    "failed" stage_outputs row (note=the error) and leaves the image's
    state exactly as Stage 1 left it - the working copy itself was
    already left untouched on failure too, so DB state and file state
    agree. A path not already known to the DB is skipped with a printed
    warning, same reasoning as stage1_capture_baseline_embeddings().

    Returns manifest_path unchanged.
    """
    if ctx is not None:
        db_path = ctx.workspace.pipeline_db_path

    db = PipelineDatabase(db_path)
    paths = [Path(p) for p in load_bucket_filepaths(manifest_path)]
    recorded, clamped_zeroed, skipped = 0, 0, 0
    for i, working_path in enumerate(paths, 1):
        print(f"[{i}/{len(paths)}] Preprocessing {working_path.name}...")
        lookup = ctx.to_relative(working_path) if ctx is not None else str(working_path)
        image = db.get_image_by_path(lookup, run_id=ctx.run_id if ctx is not None else None)
        if image is None:
            skipped += 1

        try:
            angle, profile = preprocess_for_manifest(working_path, preprocessing_profile)
            print(f"  deskew_angle={angle:+.2f}  profile={profile!r}")
            if image is not None:
                new_hash = hash_file(working_path)
                db.update_image_state(
                    image["id"], current_hash=new_hash, current_stage=3,
                    status="preprocessed", processing_profile=preprocessing_profile,
                )
                sidecar_path = preprocess_sidecar_path(working_path)
                db_sidecar_path = ctx.to_relative(sidecar_path) if ctx is not None else str(sidecar_path)
                deskew_status = "done"
                note = None
                try:
                    telemetry = json.loads(sidecar_path.read_text(encoding="utf-8"))
                    if telemetry.get("deskew_status") == "clamped_zeroed":
                        deskew_status = "clamped_zeroed"
                        note = f"raw_deskew_estimate_deg={telemetry.get('raw_deskew_estimate_deg')}"
                        clamped_zeroed += 1
                except Exception:
                    pass  # telemetry sidecar missing/unreadable - fall back to plain "done"
                db.record_stage_output(
                    image["id"], stage="stage3_preprocess", status=deskew_status,
                    sidecar_path=db_sidecar_path, note=note,
                )
                recorded += 1
        except Exception as e:
            print(f"  WARNING: preprocessing failed ({type(e).__name__}: {e}) - "
                  f"working copy left as an unprocessed straight copy of the original.")
            if image is not None:
                db.record_stage_output(
                    image["id"], stage="stage3_preprocess",
                    status="failed", note=str(e)[:300],
                )
    print(f"Recorded {recorded} image(s) in {db.db_path}"
          + (f" ({skipped} not found in DB - skipped)" if skipped else "")
          + (f" ({clamped_zeroed} had a clamped deskew estimate, zeroed)" if clamped_zeroed else ""))
    return manifest_path


def stage4_capture_postprocessing_embeddings(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    db_path: Path = DEFAULT_DB_PATH,
    ctx: Optional[RunContext] = None,
) -> Path:
    """
    ctx: same DB-identity-normalization override as stage1_capture_
    baseline_embeddings() - see that function's docstring.

    Stage 4 (Validation Capture), semantic half. A measurement stage,
    not another preprocessing stage - repeats Stage 1's semantic capture
    (core/baseline_embeddings.py's write_baseline_embeddings(), UNCHANGED
    - same GPU-sharded k=4 execution engine, same 8 encoders, same
    checkpointing, same JSON schema) against whatever pixels are AT THE
    MANIFEST'S PATHS RIGHT NOW - i.e. the current working copies, already
    through Stage 3 (and, for a corpus already classified, Stage 5 too).
    Does NOT reconstruct Stage 0 and does NOT modify the working images -
    captures them exactly as they currently exist.

    Writes to core/baseline_embeddings.py's DEFAULT_POSTPROCESSING_PATH -
    a separate file from Stage 1's DEFAULT_BASELINE_PATH, which this
    function never touches - so the two snapshots can later be compared
    image-for-image without either one overwriting the other.

    Known caveat, not fixed here (deliberately - reusing the capture
    function exactly, not creating a second implementation with a
    parameter Stage 1 doesn't have): each record's own
    "preprocessing_stage" field will still read "pre_preprocessing",
    the same literal value Stage 1 writes - it is NOT auto-corrected to
    reflect that this call captured post-processing pixels. Which
    snapshot a record belongs to is determined by which FILE it's in
    (this function's output vs. Stage 1's), not by that field's value.

    db_path: does NOT bump images.current_stage (these images are
    already past Stage 3, often past Stage 5 too - Stage 4 is a
    supplementary measurement running alongside the main 0->3->5
    progression, not a further step in it; changing current_stage here
    would misrepresent how far each image has actually progressed).
    Records one additive stage_outputs row per image instead
    (stage="stage4_validation_capture", lookup_key=identity_hash, same
    shared-file convention Stage 1 uses for baseline_embeddings.json).
    A path not already known to the DB is skipped with a printed count,
    same tolerance as stage1_capture_baseline_embeddings().

    Returns manifest_path unchanged.
    """
    from core.baseline_embeddings import write_baseline_embeddings, DEFAULT_POSTPROCESSING_PATH

    if ctx is not None:
        db_path = ctx.workspace.pipeline_db_path

    paths = [Path(p) for p in load_bucket_filepaths(manifest_path)]
    print(f"Capturing POSTPROCESSING embeddings (Stage 4) for {len(paths)} image(s) "
          f"(8 encoders each - this may take a while)...")
    postprocessing_output_path = (
        ctx.embeddings / "postprocessing_embeddings.json" if ctx is not None else DEFAULT_POSTPROCESSING_PATH
    )
    write_baseline_embeddings(paths, output_path=postprocessing_output_path)

    db = PipelineDatabase(db_path)
    recorded, skipped = 0, 0
    for p in paths:
        lookup = ctx.to_relative(p) if ctx is not None else str(p)
        image = db.get_image_by_path(lookup, run_id=ctx.run_id if ctx is not None else None)
        if image is None:
            skipped += 1
            continue
        db.record_stage_output(
            image["id"], stage="stage4_validation_capture",
            lookup_key=image["identity_hash"], status="done",
        )
        recorded += 1
    print(f"Recorded {recorded} image(s) in {db.db_path}"
          + (f" ({skipped} not found in DB - skipped)" if skipped else ""))

    return manifest_path


def stage4_capture_postprocessing_physical(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    db_path: Path = DEFAULT_DB_PATH,
    ctx: Optional[RunContext] = None,
) -> Path:
    """
    ctx: passed straight through to analyze_manifest() (db_path is also
    overridden with ctx.workspace.pipeline_db_path) - see that
    function's own docstring for the DB-identity-normalization contract.

    Stage 4 (Validation Capture), physical half - the counterpart to
    stage4_capture_postprocessing_embeddings() above (semantic half),
    mirroring it exactly: a measurement stage, not another preprocessing
    stage. Repeats Stage 1's physical capture (core/image_analysis.py's
    analyze_manifest(), same 8-worker/4-cv2-thread concurrent execution
    engine, same per-image analysis logic - see that function's own
    _ANALYSIS_CONCURRENCY_WORKERS comment for the benchmark behind it)
    against whatever pixels are AT THE MANIFEST'S PATHS RIGHT NOW.

    Stage 1's per-image sidecars (`<name>_analysis.json`, colocated with
    each working image) are NOT reused for this - analyze_manifest()'s
    write_sidecars=False is passed here specifically so this run cannot
    silently overwrite Stage 1's own sidecars with post-processing
    measurements. DEFAULT_POSTPROCESSING_REPORT_PATH's rolled-up CSV is
    Stage 4 physical's complete, separate evidence file instead - Stage
    1's sidecars and report CSV are both left untouched.

    stage="stage4_validation_capture_physical" (not the same string the
    semantic half above uses, "stage4_validation_capture") - kept
    distinct so a given image's semantic-Stage-4-done and physical-
    Stage-4-done events stay independently queryable in stage_outputs,
    the same way Stage 1's own two halves already use two different
    stage strings ("stage1_baseline_embeddings" / "stage1_image_analysis").
    Passing this into analyze_manifest()'s stage= parameter also gives
    this call its OWN idempotency scope - re-running it only measures
    images that don't already have a stage4_validation_capture_physical
    record, independent of Stage 1's own "stage1_image_analysis" history
    for the same image.

    db_path: does NOT bump images.current_stage, same reasoning as the
    semantic half above.

    Returns manifest_path unchanged.
    """
    from core.image_analysis import analyze_manifest, DEFAULT_POSTPROCESSING_REPORT_PATH

    if ctx is not None:
        db_path = ctx.workspace.pipeline_db_path

    print(f"Capturing POSTPROCESSING physical measurements (Stage 4) "
          f"from manifest {manifest_path}...")
    analyze_manifest(
        manifest_path,
        report_path=DEFAULT_POSTPROCESSING_REPORT_PATH,
        write_sidecars=False,
        db_path=db_path,
        stage="stage4_validation_capture_physical",
        ctx=ctx,
    )
    return manifest_path


def build_working_manifest_from_paths(
    source_paths: list[Path],
    working_dir: Path = DEFAULT_WORKING_DIR,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    preprocessing_profile: str = DEFAULT_PREPROCESSING_PROFILE,
    capture_baseline: bool = True,
    capture_physical_sensors: bool = True,
    capture_layout: bool = True,
    run_decision_engine: bool = True,
    db_path: Path = DEFAULT_DB_PATH,
    ctx: Optional[RunContext] = None,
) -> Path:
    """
    ctx (hashed-run migration - see docs/RUN_ARCHITECTURE.md): when
    given, threaded through to stage0_acquire_and_copy_sources() so this
    entire Stage 0-3 run isolates its output under ctx.run_root instead
    of the shared legacy default location. working_dir/manifest_path/
    db_path are ignored in favor of ctx's own paths when ctx is set.

    Stage 0 + 1 + 2 + 3 end to end, from an EXPLICIT list of source
    paths - THE convenience/orchestration wrapper every existing caller
    uses (build_working_manifest() below, scripts/run_preprocessing.py,
    ui/build_manifest_ui.py's subprocess CLI). Composes, in order:
    stage0_acquire_and_copy_sources() -> stage1_capture_baseline_
    embeddings() (only if capture_baseline; the semantic half of Stage 1)
    -> core.image_analysis.analyze_manifest() (only if capture_physical_
    sensors, 2026-08-04; the physical half) -> stage1_capture_layout_
    detections() (only if capture_layout, 2026-08-04; a third Stage 1
    sensor, default ON as of 2026-08-05 - see that flag's own docstring
    below) -> stage2_decide_profiles() (only if run_decision_engine, 2026-08-02)
    -> stage3_preprocess_manifest(). Behavior and every printed progress
    line callers already
    depend on (e.g. ui/build_manifest_ui.py's `_parse_preprocessing_
    progress()` regex) are UNCHANGED from before the 2026-08-02 stage
    split - only the internal implementation moved into separate,
    independently-callable functions. See the module docstring's "SPLIT"
    section for the ordering-change note (manifest.csv is now written as
    part of Stage 0, immediately, rather than after preprocessing).

    capture_baseline: if True (default), runs Stage 1's semantic half
    between Stage 0 and Stage 3 - i.e. on the raw working copy, before
    deskew/autocontrast ever touches it. Set False to skip (e.g. a quick
    throwaway test run where the extra ~8-model pass isn't wanted).

    capture_physical_sensors (2026-08-04): if True (default), runs Stage
    1's physical half (core/image_analysis.py's analyze_manifest())
    right after the semantic half - both halves of Stage 1 now run
    before Stage 3 ever touches a pixel, matching docs/PIPELINE_STAGE_
    TERMINOLOGY.md's definition of Stage 1 as one logical stage, not two
    - this closes a gap that was standing, acknowledged, and
    undocumented-as-permanent in docs/PIPELINE_DATABASE.md ("core/
    image_analysis.py... NOT yet wired in") since this project's DB
    migration. MEASURED COST, not estimated: ~3.7s/image on this
    machine, pure CPU (no GPU, no model inference) - for a corpus in the
    thousands this is a real, multi-hour addition to every future Stage
    0-3 run, not a rounding error. Set False for a throwaway/test run
    where that cost isn't wanted, same reasoning as capture_baseline.
    See the CPU-concurrency prototype (benchmark/stage1_concurrency_
    experiment.py, same session) for whether this cost can be reduced
    before treating it as fixed.

    capture_layout (2026-08-04, default flipped to True 2026-08-05): runs
    the layout-detection sensor (core/layout_detector.py, DocLayout-YOLO/
    DocStructBench). Originally shipped DEFAULT FALSE (brand-new,
    not-yet-validated-on-this-corpus sensor, plus its Python package
    dependency is AGPL-3.0 licensed - see core/layout_detector.py's
    module docstring) - discovered 2026-08-05 that this made it
    practically unreachable, since nothing in scripts/ or ui/ ever passed
    capture_layout=True either (the UI's subprocess CLI chain had no flag
    for it at any layer). Direct instruction: this data is useful
    downstream and should be collected regardless, same as every other
    Stage 1 sensor - flipped to default True. The AGPL-3.0 dependency
    note above still stands as a real fact about the package (relevant if
    this code is ever redistributed), it's just no longer treated as a
    reason to gate capture off. Set False to skip (e.g. a throwaway test
    run), same reasoning as capture_baseline/capture_physical_sensors.
    Writes into the SAME file capture_baseline writes to (data/baseline_
    embeddings.json, see core.baseline_embeddings.merge_sensor_records()),
    under a "layout_detections" key per image, not a separate artifact.

    run_decision_engine (2026-08-02): if True (default), runs Stage 2
    (core/decision_engine.py) between Stage 1 and Stage 3 - reads Stage
    1's baseline embeddings, records a (currently pass-through)
    preprocessing-profile decision and a tower-consensus classification
    signal per image. Has no effect on which profile actually gets
    applied (still preprocessing_profile, unconditionally - see
    core/decision_engine.py's own docstring for why) - this flag exists
    so a throwaway test run (or a run with capture_baseline=False, which
    leaves Stage 2 nothing to compute tower consensus from anyway) can
    skip it, same reasoning as capture_baseline.

    Call the stage0_/stage1_/stage2_/stage3_ functions directly instead
    of this one when you need to run only a subset, interleave other
    work between them, or run a stage against a manifest Stage 0 already
    built in an earlier process (e.g. a future reference_pipeline_v2
    capture) - that independent callability is the whole point of the
    split.

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
    if ctx is not None:
        manifest_path = ctx.manifest_csv
        db_path = ctx.workspace.pipeline_db_path
    manifest_path = stage0_acquire_and_copy_sources(source_paths, working_dir, manifest_path, db_path, ctx=ctx)

    if capture_baseline:
        stage1_capture_baseline_embeddings(manifest_path, db_path, ctx=ctx)

    if capture_physical_sensors:
        from core.image_analysis import analyze_manifest
        analyze_manifest(manifest_path, db_path=db_path, ctx=ctx)

    if capture_layout:
        stage1_capture_layout_detections(manifest_path, db_path, ctx=ctx)

    if run_decision_engine:
        from core.decision_engine import stage2_decide_profiles
        stage2_decide_profiles(manifest_path, db_path, preprocessing_profile, ctx=ctx)

    stage3_preprocess_manifest(manifest_path, preprocessing_profile, db_path, ctx=ctx)

    print("Next step (unchanged):")
    print(f"  python -m core.classifier {manifest_path}")
    return manifest_path


def build_working_manifest(
    source_folder: Path,
    working_dir: Path = DEFAULT_WORKING_DIR,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    preprocessing_profile: str = DEFAULT_PREPROCESSING_PROFILE,
    ctx: Optional[RunContext] = None,
) -> Path:
    """
    Stage 0 from a FOLDER - a thin adapter over
    build_working_manifest_from_paths(): walks source_folder
    (collect_image_paths()) and delegates. This is what
    scripts/build_working_manifest.py's CLI still calls; kept as its own
    function rather than removed since that script and its docstring are
    folder-oriented, not because it does anything build_working_manifest_
    from_paths() doesn't already do.

    ctx: passed straight through to build_working_manifest_from_paths()
    - see that function's own docstring.
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
        preprocessing_profile=preprocessing_profile, ctx=ctx,
    )


# -- Stage 2b (old numbering - NOT canonical Stage 2/Decision Engine, see
# core/decision_engine.py for that): final merge, after classification +
# (for dense_tabular_rows) manual dewarp --

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

    THE SOLE decision-making call site (2026-08-03 consolidation pass -
    verified by grep, only finalize_manifest() below calls this). Its
    result is now RECORDED into images.needs_manual_dewarp the first
    time it's computed for a given image (see finalize_manifest()),
    making this the single authoritative, durable answer to "does this
    image need manual dewarp" - any future consumer (e.g. the dewarp
    worklist tooling, which today independently assumes the whole
    dense_tabular_rows bucket needs dewarp rather than reading this)
    should read that column instead of re-deriving its own assumption or
    calling this function a second time.
    """
    return True


def finalize_manifest(
    bucket_dir: Path = BUCKET_DIR,
    output_path: Path | None = None,
    db_path: Path = DEFAULT_DB_PATH,
    ctx: Optional[RunContext] = None,
) -> Path:
    """
    ctx: when given, overrides bucket_dir/db_path/output_path with
    ctx.buckets / ctx.workspace.pipeline_db_path / (ctx.manifest_dir /
    "manifest_final.csv") - see docs/RUN_ARCHITECTURE.md.

    Stage 2b: DB-QUERY-BASED as of 2026-08-02 (migrated from scanning
    every bucket CSV by hand - see docs/PIPELINE_DATABASE.md's "What's
    NOT done" section, which flagged this as the strongest candidate for
    exactly this reason: its old implementation re-derived status by
    joining several CSVs by file-path string on every call, which is the
    precise fragility core/pipeline_db.py exists to replace, and its
    output has ZERO downstream consumers (confirmed by grep) - nothing
    could break from changing what feeds it.

    Two-step: first SYNCS the DB from core/classifier.py's bucket CSVs
    and ui/dewarp_preprocessor_ui.py's dewarped-bucket CSVs (core/
    pipeline_db.py's sync_bucket_classifications()/sync_dewarp_results(),
    the exact promoted logic scripts/migrate_manifest_to_db.py's one-time
    import used, now reused rather than duplicated - auto-registering any
    file_path the DB doesn't already know, so a bucket CSV entry can
    never be silently dropped just because Stage 0 didn't run for it).
    Then builds every output row from db.list_images()/get_stage_outputs()
    - NOT from re-reading the CSVs a second time.

    Three possible statuses per row (UNCHANGED meaning from before this
    migration):
      "ready" - not dense_tabular_rows, or dense_tabular_rows with no
                manual-dewarp requirement (see _dense_tabular_needs_
                manual_dewarp - currently unreachable, always True today,
                kept for when that changes) - file_path is already the
                final, usable image.
      "ready_dewarped" - dense_tabular_rows, manual dewarp completed -
                file_path is images.working_path (already updated to the
                dewarped output by sync_dewarp_results());
                source_original_path is recovered from the "stage2a_dewarp"
                stage_outputs row's lookup_key (the pre-dewarp path) -
                no new column needed for this.
      "pending_manual_dewarp" - dense_tabular_rows, queued for manual
                dewarp, not done yet - file_path is still the working-
                directory (deskewed+preprocessed but not dewarped) copy;
                downstream stages should NOT consume rows in this status.

    RECORDS the manual-dewarp decision (2026-08-03 consolidation pass):
    for every dense_tabular_rows image, images.needs_manual_dewarp is set
    from _dense_tabular_needs_manual_dewarp()'s result - the SAME value
    this function already used to pick a status, now also persisted so
    it's queryable DB state rather than a value that only ever existed
    inside this function call. Output columns/values are UNCHANGED by
    this - see the verified-identical note below, re-confirmed after this
    change.

    Verified behavior-preserving (2026-08-02): row-for-row identical SET
    of (file_path, category, status, source_original_path) tuples against
    the real 1677-image production corpus, compared against this
    function's pre-migration output - see docs/PIPELINE_DATABASE.md.
    Row ORDER may differ (DB insertion order rather than bucket-CSV scan
    order) - immaterial since nothing consumes this file's row order (or
    the file at all, today).

    Never raises on a missing bucket CSV or missing dewarped-bucket CSV -
    both are legitimate "nothing done in that bucket/stage yet" states,
    not errors (unchanged from before).
    """
    if ctx is not None:
        bucket_dir = ctx.buckets
        db_path = ctx.workspace.pipeline_db_path
        if output_path is None:
            output_path = ctx.manifest_dir / "manifest_final.csv"
    if output_path is None:
        output_path = PROJECT_ROOT / "data" / "manifest_final.csv"

    db = PipelineDatabase(db_path)
    n_classified = sync_bucket_classifications(db, bucket_dir)
    n_dewarped = sync_dewarp_results(db, bucket_dir)
    if n_classified or n_dewarped:
        print(f"Synced {n_classified} new/changed classification(s), "
              f"{n_dewarped} new dewarp result(s) into {db.db_path}")

    rows: list[dict] = []
    for image in db.list_images():
        category = image["bucket"]
        if category is None:
            continue  # not yet classified - matches pre-migration behavior
                       # (a path only ever appeared in manifest_final.csv
                       # if it was in SOME bucket CSV)

        if category != "dense_tabular_rows":
            rows.append({
                "file_path": image["working_path"], "category": category,
                "status": "ready", "source_original_path": "",
            })
            continue

        # Compute + RECORD the manual-dewarp decision once here (still the
        # only call site - see that function's own docstring), regardless
        # of whether dewarp has already happened, so the decision becomes
        # durable DB state instead of a value that only ever existed for
        # the lifetime of this function call. Idempotent: only writes when
        # the stored value would actually change, matching this project's
        # established sync-function discipline (core/pipeline_db.py's
        # sync_bucket_classifications()) rather than rewriting every row
        # on every finalize_manifest() run.
        needs_dewarp = _dense_tabular_needs_manual_dewarp(image["working_path"])
        needs_dewarp_int = 1 if needs_dewarp else 0
        if image["needs_manual_dewarp"] != needs_dewarp_int:
            db.update_image_state(image["id"], needs_manual_dewarp=needs_dewarp_int)

        dewarp_outputs = db.get_stage_outputs(image["id"], stage="stage2a_dewarp")
        if dewarp_outputs:
            rows.append({
                "file_path": image["working_path"], "category": category,
                "status": "ready_dewarped",
                "source_original_path": dewarp_outputs[-1]["lookup_key"] or "",
            })
        elif needs_dewarp:
            rows.append({
                "file_path": image["working_path"], "category": category,
                "status": "pending_manual_dewarp", "source_original_path": "",
            })
        else:
            rows.append({
                "file_path": image["working_path"], "category": category,
                "status": "ready", "source_original_path": "",
            })

    rows.sort(key=lambda r: r["category"])  # group by category, matching
                                             # the old sorted(bucket_dir.glob(...))
                                             # scan order at the category level

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


def _prompt_for_run_name() -> Optional[str]:
    """Interactive-only: prompts once for an optional human-readable run
    name (per docs/RUN_ARCHITECTURE.md's run naming design - distinguishes
    e.g. a production run from a test run or a dataset baseline run,
    layered on top of the immutable run_id/run_type). Empty input -> None.
    Only called when stdin is a tty AND --run-name wasn't passed on the
    CLI - scripted/subprocess callers must pass --run-name explicitly (or
    accept the None default) and never hit this prompt."""
    try:
        answer = input("Run name (optional, press Enter to skip): ").strip()
    except EOFError:
        return None
    return answer or None


def _cli_main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point (`python -m core.manifest_pipeline <source>`):
    builds a RunContext for a fresh Stage 0-3 run against a source folder
    or explicit source paths, then runs build_working_manifest_from_paths()
    against it. --run-name/--run-type bypass the interactive run-name
    prompt for non-interactive/scripted invocations."""
    parser = argparse.ArgumentParser(
        description="Stage 0-3: acquire sources, build a working manifest, preprocess."
    )
    parser.add_argument("source", type=Path, help="Source folder or a single file (PDF/image).")
    parser.add_argument("--run-name", default=None, help="Optional human-readable run name.")
    parser.add_argument(
        "--run-type", default="production", choices=sorted(RUN_TYPES),
        help="What kind of run this is (default: production).",
    )
    parser.add_argument(
        "--preprocessing-profile", default=DEFAULT_PREPROCESSING_PROFILE,
    )
    args = parser.parse_args(argv)

    run_name = args.run_name
    if run_name is None and sys.stdin.isatty():
        run_name = _prompt_for_run_name()

    workspace = WorkspaceContext.resolve()
    ctx = RunContext.create(
        workspace, run_type=args.run_type, source_input=str(args.source), run_name=run_name,
    )
    print(f"Run: {ctx.run_id}" + (f" ({run_name})" if run_name else ""))
    print(f"Run root: {ctx.run_root}")

    try:
        if args.source.is_dir():
            build_working_manifest(
                args.source, preprocessing_profile=args.preprocessing_profile, ctx=ctx,
            )
        else:
            build_working_manifest_from_paths(
                [args.source], preprocessing_profile=args.preprocessing_profile, ctx=ctx,
            )
    except Exception as e:
        ctx.mark_failed(str(e))
        raise
    ctx.mark_completed()
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_main())
