"""
Prepares an ISOLATED evaluation/calibration workspace for a raw image
folder, per Jon's 2026-08-07 direction: "raw source image != production
input... the folder picker should really mean 'Import this folder into
a temporary evaluation/calibration workspace,' not 'Open images and
hope someone already made sidecars.'"

Reuses the SAME building blocks the real Stage 0/3/4 pipeline already
uses (core/manifest_pipeline.py's copy_to_working_dir/
preprocess_for_manifest, core/auto_sidecar.py's generate_auto_sidecar)
rather than a parallel reimplementation - "the same CV preprocessing
stack used for measurement," per Jon's own wording. The only thing this
module adds is the workspace isolation/reuse logic around those calls;
it never touches data/working/, data/manifest.csv, pipeline_db, or any
bucket CSV - a raw folder picked here is NEVER enrolled into the
production pipeline just by being opened for calibration.

Workspace layout, per source folder:
    data/outputs/column_calibration_workspace/<folder_key>/
        images/    - working copies, deskewed + preprocessed in place
                     (same operations Stage 3 applies, just run here
                     instead of against data/working/)
        sidecars/  - provisional sidecars from generate_auto_sidecar()
                     (CV-classifier doc_type guess, since there's no
                     upstream Gemma classification for an arbitrary
                     folder - same fallback path
                     scripts/run_batch_auto_sidecar.py already treats as
                     "untrusted, needs human confirmation")

<folder_key> = the source folder's own name + an 8-char hash of its
resolved absolute path, so two different folders that happen to share a
name (e.g. two different "batch1" folders) never collide.

REUSE, NOT RE-RUN (per Jon: "if a folder has already been processed, it
should reuse the existing evaluation-side artifacts rather than
rerunning CV unless you explicitly request regeneration"): a raw image
is skipped if its working copy AND sidecar already exist in the
workspace, unless regenerate=True. This is a per-file check, not an
all-or-nothing one - adding new images to a previously-processed folder
only processes the new ones.

DOC_TYPE/COLUMNS OVERRIDE FILE (added 2026-08-07, per Jon: "if i place a
columns<X>.txt into the source folder before preprocessing starts, can
we have it load that as the ground truth... and override the
classifier's decision"). If raw_dir contains exactly one file matching
`columns_<doc_type>.txt` (e.g. columns_canada_census_1911.txt), its
<doc_type> is passed as generate_auto_sidecar()'s doc_type_override for
EVERY image in this folder - the CV classifier is skipped entirely, so
the CORRECT template's table/row-detection geometry is used instead of
whatever the classifier would have guessed. Its contents (one column
name per line) are returned as columns_override_path so a caller (the
UI) can auto-apply that exact list instead of requiring a manual "Load
columns file..." click per import.

IMPORTANT LIMITATION, not solved by this mechanism: doc_type_override
still requires a REAL config/document_templates/<doc_type>.yaml to
exist - core.auto_sidecar.generate_auto_sidecar() looks up the template
unconditionally even with an override, and raises if none exists. This
only fixes MISCLASSIFICATION among the templates that already exist
(e.g. a 1911 page wrongly guessed as handwritten_manifest); it does NOT
create a template for a genuinely new doc_type (e.g. 1906 census, no
template as of this writing) - naming a doc_type with no template here
produces NO sidecar at all for that image (logged, not silently wrong),
which is worse than the CV fallback's wrong-but-present guess. Building
a real new template is a separate, larger task, out of scope for this
file-convention mechanism.

DUAL-PAGE SPLIT MARKER (added 2026-08-07, per Jon: several 1906 census
scans are a bound book photographed open, TWO separate census forms per
image, roughly split at the frame's horizontal midpoint - except the
first/last page of a reel, which are genuinely single-form). A REAL
brightness-dip auto-detector was tried first and rejected: measured
directly against a real two-form scan (e001211810), the binding crease
only produced a ~7% brightness dip, easily confused with handwriting/
ruling-line noise, and landed at 65% of width rather than near center -
not reliable enough to trust unsupervised, and a bad auto-split would
silently mangle the single-form exception pages (which must NOT be
split). Jon's direction: a plain marker file instead, no per-image
detection at all - if raw_dir contains a file named exactly
`split.split`, EVERY image in that folder is cropped in half at a fixed
fraction (the marker's own content, e.g. "0.5" - blank/unparseable
falls back to 0.5) before deskew/preprocess/auto_sidecar, each half
proceeding through the identical single-page pipeline as its own
working image ({stem}_L.ext / {stem}_R.ext). No marker file - or an
unrelated raw_dir without it (e.g. a folder holding only the known
single-form exception pages) - means no split, unchanged behavior.

PROCESSED-SOURCE STATE FILE (added 2026-08-08, real bug found by Jon:
"reloaded its pulling all the originals back in after i deleted them").
The resume check used to be PURELY file-existence-based (does this raw
image's working copy + sidecar still exist in the workspace) - which
cannot tell "never processed yet" apart from "was processed, then its
output was deliberately removed" (e.g. ui/column_calibration_ui.py's
split-then-delete-the-original workflow, or a human manually clearing
out a blank/rejected page). Reloading a folder after either kind of
deletion would silently re-derive the raw image from scratch, resurrec-
ting exactly what was just removed. Fixed with a small persistent state
file, `<workspace>/processed_sources.json` - a flat {raw_image_name:
{"processed_at": ..., "outcome": "ok"|"no_sidecar"|"failed"}} map. A raw
image is skipped once it's in this file, REGARDLESS of whether its
output still exists on disk, unless regenerate=True. Every outcome
(including a failure or "no sidecar produced") gets recorded, not just
success - an image that can't get a usable sidecar shouldn't be
silently retried forever on every reload either; --regenerate remains
the explicit, deliberate way to force a redo.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from core.auto_sidecar import generate_auto_sidecar
from core.manifest_pipeline import (
    collect_image_paths, copy_to_working_dir, preprocess_for_manifest,
    DEFAULT_PREPROCESSING_PROFILE,
)
from core.row_segmentation import save_sidecar
from core.workspace_context import WorkspaceContext

SPLIT_MARKER_NAME = "split.split"
DEFAULT_SPLIT_FRACTION = 0.5
PROCESSED_STATE_FILENAME = "processed_sources.json"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Resolved via WorkspaceContext (2026-08-09, per Jon: "why does every
# session still want to output to the old data location?" - this used
# to hardcode PROJECT_ROOT/data/outputs/column_calibration_workspace,
# which only worked because of an NTFS junction transparently
# redirecting it to genealogy_workspace/research/calibration/
# column_calibration_workspace (see docs/RUN_ARCHITECTURE.md's phase-2
# migration) - real data, same files, just via an indirection every
# new session/tool would naturally keep recreating since the CODE
# itself never stopped pointing at the old path. This resolves the
# real location directly - same folder name, same existing data,
# no functional change, just no longer riding on the junction).
WORKSPACE_ROOT = WorkspaceContext.resolve().workspace_root / "research" / "calibration" / "column_calibration_workspace"


def workspace_dir_for(raw_dir: Path) -> Path:
    resolved = str(raw_dir.resolve())
    digest = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:8]
    safe_name = "".join(c if c.isalnum() else "_" for c in raw_dir.name) or "folder"
    return WORKSPACE_ROOT / f"{safe_name}_{digest}"


def _find_columns_override(raw_dir: Path, log) -> tuple[str, Path] | None:
    """
    Returns (doc_type, columns_file_path) from a raw_dir file matching
    `columns_<doc_type>.txt`, or None if no such file exists. More than
    one match is ambiguous (which doc_type applies to the whole folder?)
    so it's treated as "no override" with a loud warning rather than
    silently picking one - a folder-wide setting silently resolving to
    the wrong one of several candidates would be worse than requiring
    the human to remove the extras.
    """
    matches = sorted(raw_dir.glob("columns_*.txt"))
    if not matches:
        return None
    if len(matches) > 1:
        log(f"  WARNING: {len(matches)} columns_*.txt files found in {raw_dir} - "
            f"ambiguous, ignoring all of them: {[m.name for m in matches]}")
        return None
    path = matches[0]
    doc_type = path.stem[len("columns_"):]
    if not doc_type:
        log(f"  WARNING: {path.name} has no doc_type after 'columns_' - ignoring.")
        return None
    return doc_type, path


def _find_split_fraction(raw_dir: Path, log) -> float | None:
    """
    Returns the split fraction if raw_dir/split.split exists, else None -
    see this module's own DUAL-PAGE SPLIT MARKER docstring section for
    the full mechanism/rationale.
    """
    marker = raw_dir / SPLIT_MARKER_NAME
    if not marker.exists():
        return None
    content = marker.read_text(encoding="utf-8").strip()
    if not content:
        return DEFAULT_SPLIT_FRACTION
    try:
        fraction = float(content)
    except ValueError:
        log(f"  WARNING: {SPLIT_MARKER_NAME} content {content!r} isn't a number - "
            f"using default {DEFAULT_SPLIT_FRACTION}.")
        return DEFAULT_SPLIT_FRACTION
    if not (0.05 < fraction < 0.95):
        log(f"  WARNING: {SPLIT_MARKER_NAME} fraction {fraction} looks implausible - "
            f"using default {DEFAULT_SPLIT_FRACTION}.")
        return DEFAULT_SPLIT_FRACTION
    return fraction


def _load_processed_state(ws: Path, raw_images: list[Path], log) -> dict:
    """
    Real bug hit in production (2026-08-08, Jon): a workspace that
    already had real work done BEFORE processed_sources.json existed
    (this file was added after several sessions of real calibration
    work) came back with an empty state on first load - every raw image
    looked "unprocessed" despite having real output already, and
    prepare_evaluation_workspace() started RE-DERIVING all of them,
    resurrecting flat originals ui/column_calibration_ui.py's split-
    then-delete workflow had deliberately removed.

    Self-healing fix: if the state file doesn't exist yet AND the
    images/ directory already has real content, BACKFILL state from
    what's actually there before ever treating anything as "new" - any
    raw stem with an existing derived file (flat <stem>.ext, a split
    half <stem>_L.ext/_R.ext, or a crop-via-split double-suffix name
    like <stem>_L_R.ext - matched via a startswith prefix check, not an
    exact name list, since the exact derived filename shape varies) is
    marked processed. This can't perfectly recover which OUTCOME each
    one originally had (ok/no_sidecar/failed - that distinction was
    never persisted before this fix existed), so a backfilled entry is
    always outcome "ok" with backfilled=True - not fully equivalent
    to a normal entry, but sufficient for its one real purpose (stop
    treating real existing work as new).
    """
    path = ws / PROCESSED_STATE_FILENAME
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass  # corrupt/unreadable - fall through to backfill/empty below

    images_dir = ws / "images"
    if images_dir.is_dir():
        existing_stems = [p.stem for p in images_dir.iterdir() if p.is_file()]
        if existing_stems:
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            backfilled = {}
            for raw_image in raw_images:
                stem = raw_image.stem
                if any(s == stem or s.startswith(stem + "_") for s in existing_stems):
                    backfilled[raw_image.name] = {
                        "processed_at": now, "outcome": "ok", "backfilled": True,
                    }
            if backfilled:
                log(f"  {PROCESSED_STATE_FILENAME} missing but {len(backfilled)} raw image(s) "
                    f"already have real output in this workspace - backfilling state instead "
                    f"of treating them as new (see this function's own docstring).")
                _save_processed_state(ws, backfilled)
                return backfilled

    return {}


def _save_processed_state(ws: Path, state: dict) -> None:
    (ws / PROCESSED_STATE_FILENAME).write_text(json.dumps(state, indent=2), encoding="utf-8")


def _mark_processed(state: dict, raw_image: Path, outcome: str) -> None:
    state[raw_image.name] = {
        "processed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "outcome": outcome,
    }


def _split_image(raw_image: Path, images_dir: Path, fraction: float) -> list[Path]:
    """Crops raw_image into left/right halves at `fraction` of its width,
    saves both as new working copies (never touches the raw file), and
    returns [left_path, right_path]."""
    with Image.open(raw_image) as im:
        im = im.convert("RGB")
        w, h = im.size
        split_x = int(w * fraction)
        left = im.crop((0, 0, split_x, h))
        right = im.crop((split_x, 0, w, h))

    left_path = images_dir / f"{raw_image.stem}_L{raw_image.suffix}"
    right_path = images_dir / f"{raw_image.stem}_R{raw_image.suffix}"
    left.save(left_path)
    right.save(right_path)
    return [left_path, right_path]


def prepare_evaluation_workspace(
    raw_dir: Path, regenerate: bool = False,
    profile_name: str = DEFAULT_PREPROCESSING_PROFILE,
    log=print,
) -> tuple[list[Path], Path, Path | None]:
    """
    Returns (working_image_paths, sidecar_dir, columns_override_path) for
    the calibration UI to open DIRECTLY - never the raw_dir images
    themselves. Every image in the returned list has already been
    through deskew/preprocess, and has (or was attempted to have) a
    provisional sidecar written to sidecar_dir alongside it, matching
    ui/column_calibration_ui.py's own "{stem}_sidecar.json" lookup
    convention exactly. columns_override_path is the raw_dir's own
    columns_<doc_type>.txt file (see _find_columns_override()) if one was
    found and used, else None - callers pass this straight through as
    ColumnCalibrationApp's columns_file_override so the override applies
    automatically, no manual "Load columns file..." click needed.

    A per-file failure (generate_auto_sidecar() unable to produce a
    sidecar at all - e.g. a genuinely undetectable page, OR a
    columns_<doc_type>.txt naming a doc_type with no template - see this
    module's own docstring's IMPORTANT LIMITATION) does not raise; it's
    logged and that one file is left without a sidecar, matching the
    calibration UI's own existing "no auto sidecar found" message for a
    page rather than aborting the whole folder's import over one bad
    image.
    """
    raw_images = collect_image_paths(raw_dir)
    if not raw_images:
        raise SystemExit(f"No images found in {raw_dir}")

    override = _find_columns_override(raw_dir, log)
    doc_type_override, columns_override_path = override if override else (None, None)
    if doc_type_override:
        log(f"  Using doc_type override {doc_type_override!r} from {columns_override_path.name} "
            f"for every image in this folder - CV classifier skipped.")

    split_fraction = _find_split_fraction(raw_dir, log)
    if split_fraction is not None:
        log(f"  {SPLIT_MARKER_NAME} found - splitting every image in this folder into "
            f"left/right halves at fraction {split_fraction}.")

    ws = workspace_dir_for(raw_dir)
    images_dir = ws / "images"
    sidecars_dir = ws / "sidecars"
    images_dir.mkdir(parents=True, exist_ok=True)
    sidecars_dir.mkdir(parents=True, exist_ok=True)

    state = _load_processed_state(ws, raw_images, log)
    to_process = [
        raw_image for raw_image in raw_images
        if regenerate or raw_image.name not in state
    ]

    if to_process:
        log(f"Evaluation workspace: {ws.name} - processing {len(to_process)}/{len(raw_images)} "
            f"new/changed image(s) (regenerate={regenerate})...")
        for raw_image in to_process:
            if split_fraction is not None:
                log(f"  {raw_image.name}: splitting at fraction {split_fraction}...")
                working_copies = _split_image(raw_image, images_dir, split_fraction)
            else:
                mapping = copy_to_working_dir([raw_image], images_dir)
                working_copies = [mapping[raw_image]]

            outcome = "ok"
            for working_copy in working_copies:
                log(f"  {working_copy.name}: deskew + preprocess...")
                preprocess_for_manifest(working_copy, profile_name=profile_name)

                sidecar_path = sidecars_dir / f"{working_copy.stem}_sidecar.json"
                try:
                    result = generate_auto_sidecar(str(working_copy), debug=False,
                                                    doc_type_override=doc_type_override)
                except Exception as e:
                    log(f"  {working_copy.name}: auto_sidecar FAILED - {e}")
                    outcome = "failed"
                    continue
                if result.sidecar is None:
                    log(f"  {working_copy.name}: no sidecar produced - {'; '.join(result.warnings)[:150]}")
                    outcome = "no_sidecar" if outcome == "ok" else outcome
                    continue
                save_sidecar(result.sidecar, sidecar_path)
                fallback_note = " (CV-guessed doc_type, unconfirmed)" if result.used_cv_fallback else ""
                log(f"  {working_copy.name}: OK{fallback_note}")

            # Recorded regardless of outcome (see PROCESSED-SOURCE STATE
            # FILE docstring above) - a raw image that failed or produced
            # no usable sidecar is still "handled," not silently retried
            # forever on every reload. This is the fix for the real bug
            # Jon hit: state persists independent of whether images_dir/
            # sidecars_dir still contain the output (e.g. after a
            # deliberate deletion via the UI's split-then-delete or a
            # manual cleanup) - only --regenerate forces a redo now.
            _mark_processed(state, raw_image, outcome)
            _save_processed_state(ws, state)
    else:
        log(f"Evaluation workspace: {ws.name} - reusing {len(raw_images)} already-processed image(s) "
            f"(per {PROCESSED_STATE_FILENAME}).")

    working_images = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    )
    return working_images, sidecars_dir, columns_override_path
