"""
Stage 5 (Document Routing) per docs/PIPELINE_STAGE_TERMINOLOGY.md's
canonical Stage 0-6 naming (2026-08-02). Classifies every image in the
manifest and routes it to the appropriate bucket CSV.

Usage:
    python -m core.classifier data/manifest.csv

Behavior on failure (this is deliberate, not a bug to silence):
  - If the loader raises during classification (malformed output,
    unrecognized category, missing fields), the file is routed to
    uncertain_review.csv with the error recorded, NOT retried with a
    looser parse. A classification failure should surface, not be
    papered over.
  - If confidence is below pipeline.yaml's min_confidence threshold,
    the file is routed to uncertain_review.csv even if the category
    parsed cleanly.

DB WRITE-AUTHORITY (2026-08-03 consolidation pass): run() now calls
core/pipeline_db.py's PipelineDatabase.record_classification() directly,
per image, as soon as that image's classification (or failure) is
known - the DB is the authoritative write for Stage 5 as of this pass,
not a later mirror of the bucket CSVs. Per image: the bucket CSV row is
written FIRST, then record_classification() commits the images UPDATE
and the "stage5_classify" stage_outputs INSERT together as ONE atomic
transaction (see that method's own docstring). CSV-then-DB is the
deliberate order, not DB-then-CSV: sync_bucket_classifications() (kept,
now a safety-net reconciliation pass at the end of run() rather than
the primary write path) can only heal a CSV-row-written-but-DB-write-
failed gap, because it only ever reads CSV -> writes DB, never the
reverse - so if the DB write is the one that's ever at risk of being
lost, the existing reconciliation tool already covers it. A CSV write
failure, by contrast, means nothing happened for that image on EITHER
side, a clean stopping point without any invented DB->CSV repair path.
This directly satisfies "the classifier should either finish
successfully or leave the previous state untouched" per image, since a
DB-write failure (which its own explicit transaction guarantees can
never land only half of) never re-triggers a corrective step until it
does - and reconciliation is exactly what already exists for that. See
docs/PIPELINE_DATABASE.md for the full design.

Behavior UNCHANGED from before this pass: which bucket a file lands in,
the min_confidence override, error handling, and every bucket CSV's
contents/column set are identical - only WHERE the DB write happens
(per-image, atomically, immediately) and WHY (authoritative, not a
later mirror) changed.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from core.loaders.base_loader import load_model_config
from core.loader_registry import LOADER_REGISTRY
from core.schema import DocumentCategory, ClassificationResult
from core.pipeline_db import PipelineDatabase, DEFAULT_DB_PATH, sync_bucket_classifications, hash_file
from core.workspace_context import WorkspaceContext

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Legacy fallback - see core/manifest_pipeline.py's identical comment.
# Resolves into the legacy_pre_run_system run once the workspace
# migration has run; falls back to the pre-migration data/buckets/
# layout until then. New callers should pass ctx to run()/open_bucket_
# writers() instead of relying on this module-level constant.
_legacy_bucket_dir = WorkspaceContext.resolve().runs_root / "legacy_pre_run_system" / "outputs" / "buckets"
BUCKET_DIR = _legacy_bucket_dir if _legacy_bucket_dir.exists() else PROJECT_ROOT / "data" / "buckets"

CSV_FIELDS = [
    "file_path", "category", "confidence", "text_density",
    "handwriting", "table_layout", "faces", "map_like",
    "reason", "model", "prompt_version",
]

UNCERTAIN_FIELDS = CSV_FIELDS + ["error"]


def load_pipeline_config() -> dict:
    with open(PROJECT_ROOT / "config" / "pipeline.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CLASSIFIER_CATEGORY_PLACEHOLDER = "{{CLASSIFIER_CATEGORY_CHOICES}}"


def render_classifier_prompt(prompt_text: str, taxonomy: "Taxonomy | None" = None) -> str:
    """
    Substitutes CLASSIFIER_CATEGORY_PLACEHOLDER with core/taxonomy.py's
    render_classifier_category_block() (2026-08-04) - the mechanical
    "choose exactly one of: ..." bullet list, generated from config/
    taxonomy.yaml's classifier_guidance fields instead of being
    hardcoded prose in the prompt file. Verified byte-for-byte (modulo
    line-wrapping) against the original hardcoded prompt before this
    was wired in - see docs/TAXONOMY.md.

    A prompt file with no placeholder is returned unchanged - this
    function is safe to call unconditionally on every prompt, not just
    ones that opt into templating.
    """
    if CLASSIFIER_CATEGORY_PLACEHOLDER not in prompt_text:
        return prompt_text
    from core.taxonomy import load_taxonomy
    taxonomy = taxonomy or load_taxonomy()
    return prompt_text.replace(
        CLASSIFIER_CATEGORY_PLACEHOLDER, taxonomy.render_classifier_category_block()
    )


def build_classifier_loader(pipeline_cfg: dict, debug: bool = False) -> GemmaLoader:
    model_name = pipeline_cfg["classifier"]["model"]
    model_cfg = load_model_config(model_name)

    prompt_path = PROJECT_ROOT / pipeline_cfg["classifier"]["prompt_file"]
    model_cfg.prompt_text = render_classifier_prompt(prompt_path.read_text(encoding="utf-8"))

    loader_cls = LOADER_REGISTRY.get(model_cfg.loader_class)
    if loader_cls is None:
        raise ValueError(
            f"No loader registered for loader_class={model_cfg.loader_class!r}. "
            f"Known loaders: {list(LOADER_REGISTRY.keys())}"
        )

    loader = loader_cls(model_cfg)
    # Read by GemmaLoader.initialize_model_and_tokenizer() to decide
    # whether to register vision-instrumentation hooks (docs/GEMMA_
    # INSTRUMENTATION_AND_SENSOR_SURVEY.md Part 1) - set on the instance
    # BEFORE load, not after, so hook registration happens as part of
    # model load itself rather than as a separate post-load step. A
    # harmless, unused attribute for any loader class that doesn't check
    # it (every loader except GemmaLoader today), same "no-op unless
    # opted into" shape as every other debug-only mechanism in this file.
    loader._debug_mode = debug
    loader.initialize_model_and_tokenizer()
    return loader


def open_bucket_writers(bucket_dir: Path = BUCKET_DIR) -> dict[str, tuple[csv.DictWriter, Any]]:
    bucket_dir.mkdir(parents=True, exist_ok=True)
    writers = {}
    for category in DocumentCategory:
        path = bucket_dir / f"{category.value}.csv"
        is_new = not path.exists()
        f = open(path, "a", newline="", encoding="utf-8")
        fields = UNCERTAIN_FIELDS if category == DocumentCategory.UNCERTAIN else CSV_FIELDS
        writer = csv.DictWriter(f, fieldnames=fields)
        if is_new:
            writer.writeheader()
        writers[category.value] = (writer, f)
    return writers


def result_to_row(result: ClassificationResult) -> dict:
    row = result.model_dump()
    row["category"] = result.category.value
    return row


def _enable_raw_output_debug(loader) -> None:
    """
    Wraps loader._run_generate so --debug prints each call's raw model
    text to stdout before core/loaders/base_loader.py's classify()
    parses it into a ClassificationResult. Deliberately does NOT touch
    any file under core/loaders/ - added 2026-07-30 while a real
    classification batch was actively running, and CLAUDE.md's rule
    ("ask before editing loader code while a run is in progress") is
    specifically about that directory; wrapping at this call site
    instead means the flag never needs to touch it, live run or not.

    Assigns a plain function to the INSTANCE (not the class) - Python's
    descriptor protocol only auto-binds `self` for methods looked up on
    the class, so an instance attribute like this is called with
    exactly the two args it's defined to take (raw_image, prompt), no
    `self` involved. Standard, safe pattern for patching one instance
    without touching the class/module it came from.
    """
    original_run_generate = loader._run_generate

    def _debug_run_generate(raw_image, prompt):
        raw_output = original_run_generate(raw_image, prompt)
        print(f"  [raw model output]\n{raw_output}\n  [end raw output]")
        return raw_output

    loader._run_generate = _debug_run_generate


def _record_classification_db(
    db: PipelineDatabase, image_id: int | None, file_path: str,
    bucket: str, confidence: float | None, model: str,
    status: str, stage_output_status: str, note: str | None,
    bucket_dir: Path = BUCKET_DIR,
) -> None:
    """
    Wraps PipelineDatabase.record_classification() with the same
    per-image fault tolerance every other stage's DB wiring in this
    project already has: a DB write failure (locked file, disk full)
    prints a warning and lets the batch continue, rather than aborting
    an expensive, already-running GPU inference run over a DB problem.
    The bucket CSV row was already written by the caller before this is
    called, so nothing is lost - sync_bucket_classifications() (still
    called at the end of run()) picks up exactly this kind of gap.
    image_id=None (get_or_create_image() itself failed) skips the DB
    write entirely, same reasoning.
    """
    if image_id is None:
        return
    bucket_csv_path = bucket_dir / f"{bucket}.csv"
    try:
        db.record_classification(
            image_id, bucket=bucket, confidence=confidence, model=model,
            status=status, sidecar_path=str(bucket_csv_path), lookup_key=file_path,
            stage_output_status=stage_output_status, note=note,
        )
    except Exception as e:
        print(f"  WARNING: DB write failed for this image ({type(e).__name__}: {e}) - "
              f"bucket CSV row was already written; the end-of-run reconciliation "
              f"sync will pick this up.")


def run(manifest_path: Path, debug: bool = False, db_path: Path = DEFAULT_DB_PATH, ctx=None) -> None:
    """
    ctx (hashed-run migration, typed loosely to avoid an import cycle -
    see docs/RUN_ARCHITECTURE.md): when given a RunContext, overrides
    db_path/bucket_dir with ctx.workspace.pipeline_db_path/ctx.buckets,
    and every DB identity lookup/write below is normalized through
    ctx.to_relative()/scoped by ctx.run_id.
    """
    pipeline_cfg = load_pipeline_config()
    min_confidence = pipeline_cfg["classifier"]["min_confidence"]

    if ctx is not None:
        db_path = ctx.workspace.pipeline_db_path
    bucket_dir = ctx.buckets if ctx is not None else BUCKET_DIR

    loader = build_classifier_loader(pipeline_cfg, debug=debug)
    if debug:
        _enable_raw_output_debug(loader)
    writers = open_bucket_writers(bucket_dir)
    db = PipelineDatabase(db_path)

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Classifying {len(rows)} images...")

    for i, row in enumerate(rows, 1):
        file_path = row["file_path"]
        print(f"[{i}/{len(rows)}] {file_path}")

        try:
            lookup = ctx.to_relative(file_path) if ctx is not None else file_path
            image = db.get_image_by_path(lookup, run_id=ctx.run_id if ctx is not None else None)
            image_id = image["id"] if image is not None else db.get_or_create_image(
                source_path=file_path, working_path=lookup,
                run_id=ctx.run_id if ctx is not None else None,
                identity_hash=hash_file(file_path) if ctx is not None else None,
            )
        except Exception as e:
            image_id = None
            print(f"  WARNING: could not resolve/register this image in the DB "
                  f"({type(e).__name__}: {e}) - classification proceeds, DB write skipped.")

        try:
            with Image.open(file_path) as raw_image:
                result = loader.classify(file_path, raw_image)
        except Exception as e:
            error_note = str(e)[:300]
            writer, _ = writers[DocumentCategory.UNCERTAIN.value]
            writer.writerow({
                "file_path": file_path,
                "category": "",
                "confidence": "",
                "text_density": "", "handwriting": "", "table_layout": "",
                "faces": "", "map_like": "", "reason": "",
                "model": loader.config.model_name,
                "prompt_version": loader.config.prompt_version,
                "error": error_note,
            })
            print(f"  -> uncertain_review (error: {e})")
            _record_classification_db(
                db, image_id, file_path, bucket=DocumentCategory.UNCERTAIN.value,
                confidence=None, model=loader.config.model_name,
                status="uncertain_review", stage_output_status="failed", note=error_note,
                bucket_dir=bucket_dir,
            )
            continue

        target_category = result.category
        if result.confidence < min_confidence:
            target_category = DocumentCategory.UNCERTAIN
            print(f"  -> uncertain_review (low confidence: {result.confidence:.2f}, "
                  f"originally classified as {result.category.value})")
        else:
            print(f"  -> {target_category.value} (confidence: {result.confidence:.2f})")

        writer, _ = writers[target_category.value]
        row_out = result_to_row(result)
        if target_category == DocumentCategory.UNCERTAIN:
            row_out["error"] = ""
        writer.writerow(row_out)

        _record_classification_db(
            db, image_id, file_path, bucket=target_category.value,
            confidence=result.confidence, model=loader.config.model_name,
            status="uncertain_review" if target_category == DocumentCategory.UNCERTAIN else "classified",
            stage_output_status="done", note=None,
            bucket_dir=bucket_dir,
        )

    for _, f in writers.values():
        f.close()

    print(f"\nDone. Bucket CSVs written to {bucket_dir}")

    # Safety-net reconciliation, not the primary write path anymore (see
    # module docstring) - catches any gap left by a DB write failure
    # above, and any OTHER writer of these bucket CSVs. Idempotent and
    # fast (a few seconds over the whole corpus, measured), so keeping
    # this costs nothing even when it finds zero changes.
    updated = sync_bucket_classifications(db, bucket_dir)
    print(f"Reconciliation sync: {updated} additional change(s) into {db.db_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 5: classify every image in the manifest, route into bucket CSVs.")
    parser.add_argument("manifest", help="Path to manifest.csv")
    parser.add_argument(
        "--debug", action="store_true",
        help="Print each image's raw model output to stdout before it's parsed, "
             "and (Gemma only) register vision-instrumentation forward hooks - "
             "see docs/GEMMA_INSTRUMENTATION_AND_SENSOR_SURVEY.md Part 1.",
    )
    parser.add_argument(
        "--db-path", default=str(DEFAULT_DB_PATH),
        help=f"core/pipeline_db.py database path (default: {DEFAULT_DB_PATH})",
    )
    args = parser.parse_args()
    run(Path(args.manifest), debug=args.debug, db_path=Path(args.db_path))


if __name__ == "__main__":
    main()
