"""
Stage 2 (Decision Engine) per docs/PIPELINE_STAGE_TERMINOLOGY.md.
"Determine required preprocessing operations from Stage 1 evidence" -
runs between Stage 1 (Raw Sensor Capture) and Stage 3 (Image Processing).

TWO SEPARATE THINGS, deliberately kept apart:

1. PREPROCESSING PROFILE DECISION - trivial pass-through today
   (everyone gets DEFAULT_PREPROCESSING_PROFILE, exactly like before
   this stage existed). Per docs/PREPROCESSING_STAGE_NOTES.md's explicit
   gate ("the semantic sensor layer is NOT used to make preprocessing
   decisions yet... once evidence demonstrates predictive value - not
   inferred from embeddings without that evidence") and this project's
   Hypothesis -> Experiment -> Benchmark -> Evidence -> Production
   promotion rule: there is currently exactly ONE calibrated Stage 1
   signal (`table_confidence`, floor ~0.375) and no validated finding
   that any signal should change which profile an image gets. This
   module builds the real, DB-wired ARCHITECTURE for that decision -
   Stage 1 evidence in, a profile choice out, recorded per image - so a
   real rule can be promoted into stage2_decide_profiles() later without
   restructuring anything, but does not invent an unvalidated rule now.

2. TOWER-CONSENSUS CLASSIFICATION - real, reused, NOT gated the same
   way. Jon's direction (2026-08-02): "the sensor tower should be used
   here to determine what an image needs to have done to it, while its
   gathering that information we should also be using its
   classification ability too." The 8 qualified vision towers'
   nearest-cluster bucket prediction + consensus categorization is
   ALREADY validated production research (the Multi-Tower Routing
   Audit, docs/BENCHMARK2_3_MULTI_TOWER_ROUTING_AUDIT.md) - reusing it
   here is not inventing a new rule, just running it earlier (right
   after Stage 1, before Stage 5/Gemma classification even happens)
   using embeddings Stage 1 already captured, with ZERO new model
   inference (no encoder is loaded or run again - every vector this
   module touches was already computed and persisted to data/
   baseline_embeddings.json by core/baseline_embeddings.py). The
   consensus_category this produces is evidence for whatever later
   decides E2B/E4B escalation or cross-checks Stage 5's real
   classification - this module does not act on it, only records it.

REFERENCE EMBEDDINGS come from the corpus's OWN already-classified
bucket CSVs, not a separate labeled set - build_reference_embeddings()
takes the first N_REFERENCE_PER_BUCKET already-classified images per
bucket (excluding uncertain_review, same as the audit: "never a tower
candidate bucket") and pulls their vectors straight out of baseline_
embeddings.json. N_REFERENCE_PER_BUCKET=15 matches benchmark2_2_cross_
validator.BUCKET_PLAN's own validated scale (15 per bucket for most
categories, 8 for mixed_text_image, 0/skipped for handwritten_ledger) -
not a new, unvalidated reference-set size.

EVIDENCE VS. STATE SPLIT (matches core/pipeline_db.py's own design):
the coarse consensus_category ("unanimous"/"majority"/"split"/
"complete_disagreement") is small and queryable, so it's a column on
images. The full per-encoder votes/scores are NOT - they go into a
small `<name>_tower_consensus.json` sidecar, referenced via a
stage_outputs row, same evidence-in-sidecars discipline as every other
stage.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from core.vision_embeddings import QUALIFIED_ENCODERS, score_all_buckets, classify_consensus
from core.baseline_embeddings import DEFAULT_BASELINE_PATH, resolve_baseline_image_path
from core.bucket_worklist import load_bucket_filepaths
from core.pipeline_db import PipelineDatabase, DEFAULT_DB_PATH
from core.schema import DocumentCategory
from core.workspace_context import WorkspaceContext

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Legacy fallbacks - see core/manifest_pipeline.py's identical comment.
_legacy_root = WorkspaceContext.resolve().runs_root / "legacy_pre_run_system"
if _legacy_root.exists():
    DEFAULT_MANIFEST_PATH = _legacy_root / "manifest" / "manifest.csv"
    BUCKET_DIR = _legacy_root / "outputs" / "buckets"
else:
    DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.csv"
    BUCKET_DIR = PROJECT_ROOT / "data" / "buckets"

DEFAULT_PREPROCESSING_PROFILE = "autocontrast"

# Matches benchmark2_2_cross_validator.BUCKET_PLAN's own validated
# reference-set scale (15/bucket for most categories) - see module
# docstring. Buckets with fewer already-classified images than this
# simply contribute fewer references (or none), same graceful behavior
# as the original audit's mixed_text_image (8) / handwritten_ledger (0).
N_REFERENCE_PER_BUCKET = 15


def build_reference_embeddings(
    bucket_dir: Path = BUCKET_DIR,
    baseline_path: Path = DEFAULT_BASELINE_PATH,
    n_per_bucket: int = N_REFERENCE_PER_BUCKET,
    encoders: list[tuple[str, str]] = QUALIFIED_ENCODERS,
) -> dict[str, dict[str, list[tuple[str, np.ndarray]]]]:
    """
    Returns {encoder_name: {bucket: [(path, vector), ...]}}. Zero new
    model inference - every vector comes straight out of an already-
    captured baseline_embeddings.json record. Returns an empty dict if
    that file doesn't exist yet (nothing to build references from).
    """
    if not baseline_path.exists():
        return {}
    records = json.loads(baseline_path.read_text(encoding="utf-8"))
    by_path = {resolve_baseline_image_path(r["image"]): r for r in records}

    result: dict[str, dict[str, list[tuple[str, np.ndarray]]]] = {
        encoder_name: {} for encoder_name, _ in encoders
    }
    for category in DocumentCategory:
        if category == DocumentCategory.UNCERTAIN:
            continue  # not a real document type - never a reference candidate
        bucket_csv = bucket_dir / f"{category.value}.csv"
        if not bucket_csv.exists():
            continue
        file_paths = load_bucket_filepaths(bucket_csv)[:n_per_bucket]

        for encoder_name, _ in encoders:
            entries: list[tuple[str, np.ndarray]] = []
            for fp in file_paths:
                record = by_path.get(resolve_baseline_image_path(fp))
                if record is None:
                    continue
                emb_entry = record["embeddings"].get(encoder_name)
                if emb_entry is None:
                    continue
                entries.append((fp, np.array(emb_entry["vector"], dtype=np.float64)))
            result[encoder_name][category.value] = entries
    return result


def compute_tower_consensus_for_image(
    record: dict, reference_embeddings: dict, exclude_key: str,
) -> tuple[str | None, str | None, dict]:
    """
    Given one image's baseline_embeddings.json record and a reference_
    embeddings set (build_reference_embeddings()'s return shape),
    computes each of the 8 towers' nearest-bucket vote and the
    consensus category across those votes. Shared by stage2_decide_
    profiles() (runs BEFORE Stage 5/Gemma classification, using
    whatever reference data exists at that point - often none, for a
    freshly-acquired corpus) and audit_classification_against_tower_
    consensus() (runs AFTER, once Gemma's real classifications exist to
    build a real reference set from) - both need the IDENTICAL
    computation, just at different points in time.

    RECORDS THE COMPLETE BUCKET-SCORE LANDSCAPE FOR EVERY ENCODER, not
    just the winner (2026-08-03, Jon: "treat the vision towers as
    sensors - a sensor should report everything it observed, not just
    its final winner... storage is cheap, re-running eight vision
    towers across the corpus is expensive... I do not want to begin
    adjusting thresholds yet - without the full bucket scores, threshold
    tuning is guesswork"). Uses core/vision_embeddings.py's
    score_all_buckets() instead of predict_nearest_bucket() for exactly
    this reason. Each encoder's per_encoder entry carries:
      - "winner"/"winner_score" - the bucket that won and its score
      - "runner_up"/"runner_up_score" - the second-place bucket/score
        (None if fewer than 2 buckets had any reference data to score
        against - nothing to be a runner-up)
      - "margin" - winner_score - runner_up_score (None alongside
        runner_up when there isn't one) - a near-zero margin is a near
        tie; a large margin is a confident call. This is a convenience
        summary computed FROM "scores", not new information.
      - "scores" - the complete per-bucket dict, sorted highest-first.
        THIS is the field that matters - the others are summaries of
        it, kept only because they're what a human scanning the sidecar
        actually wants to see first.
    This is deliberately still evidence (goes into the sidecar JSON
    only, via _write_tower_consensus_sidecar() below), not a new DB
    column - see core/pipeline_db.py's own evidence-vs-state split. No
    decision logic or threshold reads this yet; it is recorded so that
    a FUTURE analysis (which buckets are consistently close together,
    which images are near-ties, which encoders are consistently
    uncertain) can be done without ever re-running the 8 vision towers.

    Returns (consensus_category, top_voted_bucket, per_encoder_votes).
    consensus_category/top_voted_bucket are both None if no encoder had
    both an embedding AND a non-empty reference set to compare against
    (no votes cast at all) - a normal "no evidence yet" state, not an
    error. The CONSENSUS VOTE ITSELF is UNCHANGED by this pass - each
    encoder still votes with its single winning bucket, exactly as
    before; only what gets RECORDED alongside that vote has grown.
    """
    votes = []
    per_encoder: dict[str, dict] = {}
    for encoder_name, _ in QUALIFIED_ENCODERS:
        emb_entry = record["embeddings"].get(encoder_name)
        refs = reference_embeddings.get(encoder_name, {})
        if emb_entry is None or not refs:
            continue
        emb = np.array(emb_entry["vector"], dtype=np.float64)
        scores = score_all_buckets(emb, refs, exclude_key=exclude_key)
        if not scores:
            continue  # every bucket had zero references after exclusion

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        winner, winner_score = ranked[0]
        winner_score = round(winner_score, 4)
        if len(ranked) >= 2:
            runner_up, runner_up_score = ranked[1]
            runner_up_score = round(runner_up_score, 4)
            # Computed from the ALREADY-ROUNDED scores, not the raw
            # ones - so margin always equals winner_score - runner_up_
            # score exactly as displayed, not a slightly different
            # number from rounding each operand separately afterward.
            margin = round(winner_score - runner_up_score, 4)
        else:
            runner_up, runner_up_score, margin = None, None, None

        votes.append(winner)
        per_encoder[encoder_name] = {
            "winner": winner,
            "winner_score": winner_score,
            "runner_up": runner_up,
            "runner_up_score": runner_up_score,
            "margin": margin,
            "scores": {b: round(s, 4) for b, s in ranked},
        }

    if not votes:
        return None, None, per_encoder
    consensus_category = classify_consensus(votes)
    top_bucket = Counter(votes).most_common(1)[0][0]
    return consensus_category, top_bucket, per_encoder


def _write_tower_consensus_sidecar(
    working_path: Path, per_encoder: dict, consensus_category: str,
) -> Path:
    """Sibling `<name>_tower_consensus.json` - same sidecar-beside-its-
    subject convention as core/image_analysis.py's analysis_sidecar_path()
    and core/dewarp.py's dewarp_sidecar_path()."""
    sidecar_path = working_path.with_name(f"{working_path.stem}_tower_consensus.json")
    payload = {
        "image": str(working_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "consensus_category": consensus_category,
        "towers": per_encoder,
    }
    sidecar_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return sidecar_path


def stage2_decide_profiles(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    db_path: Path = DEFAULT_DB_PATH,
    preprocessing_profile: str = DEFAULT_PREPROCESSING_PROFILE,
    baseline_path: Path = DEFAULT_BASELINE_PATH,
    bucket_dir: Path = BUCKET_DIR,
    ctx=None,
) -> Path:
    """
    ctx (hashed-run migration, typed loosely to avoid an import cycle -
    see docs/RUN_ARCHITECTURE.md): when given a RunContext, overrides
    db_path/bucket_dir and normalizes every DB identity lookup/write
    (get_image_by_path, the stage2_tower_consensus sidecar_path) through
    ctx.to_relative()/ctx.run_id instead of matching absolute paths.

    Stage 2 (Decision Engine) ONLY. Reads manifest_path's "file_path"
    column; for each image already known to the DB (from Stage 0):

    - Preprocessing profile: trivial pass-through (see module
      docstring's gate) - always preprocessing_profile, recorded on
      images.processing_profile. NOT a final decision from Stage 1
      evidence yet - this is the architecture, ready for a real rule
      once one is validated.
    - Tower consensus: if this image has a Stage 1 baseline-embedding
      record AND at least one bucket has reference embeddings, computes
      each of the 8 towers' nearest-bucket vote (compute_tower_consensus_
      for_image(), using score_all_buckets() - the same nearest-cluster
      method the Multi-Tower Routing Audit validated) and the
      consensus category across those 8 votes (classify_consensus,
      same). Written to images.tower_consensus_category (coarse
      summary) plus a full per-encoder sidecar (evidence, not state).

    An image with no baseline-embedding record, or where no bucket has
    any reference embeddings yet (e.g. a brand-new project with nothing
    classified yet), still gets a profile decision - just no tower
    consensus (tower_consensus_category stays NULL). This is a normal,
    expected state, not an error.

    A path not already known to the DB is skipped with a printed
    warning, same reasoning as stage1_capture_baseline_embeddings()/
    stage3_preprocess_manifest() in core/manifest_pipeline.py.

    Returns manifest_path unchanged.
    """
    if ctx is not None:
        db_path = ctx.workspace.pipeline_db_path
        bucket_dir = ctx.buckets

    db = PipelineDatabase(db_path)
    paths = [Path(p) for p in load_bucket_filepaths(manifest_path)]

    baseline_records: dict[str, dict] = {}
    if baseline_path.exists():
        raw = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline_records = {resolve_baseline_image_path(r["image"]): r for r in raw}
    else:
        print(f"No baseline embeddings at {baseline_path} - profile decisions will "
              f"still be recorded, but with no tower-consensus evidence to compute.")

    reference_embeddings = build_reference_embeddings(bucket_dir, baseline_path)

    decided, skipped, no_evidence = 0, 0, 0
    for i, working_path in enumerate(paths, 1):
        print(f"[{i}/{len(paths)}] Deciding profile for {working_path.name}...")
        lookup = ctx.to_relative(working_path) if ctx is not None else str(working_path)
        image = db.get_image_by_path(lookup, run_id=ctx.run_id if ctx is not None else None)
        if image is None:
            skipped += 1
            continue

        record = baseline_records.get(resolve_baseline_image_path(str(working_path)))
        consensus_category, _top_bucket, per_encoder = (None, None, {})
        if record is not None:
            consensus_category, _top_bucket, per_encoder = compute_tower_consensus_for_image(
                record, reference_embeddings, exclude_key=str(working_path),
            )

        db.update_image_state(
            image["id"], current_stage=2, status="decided",
            processing_profile=preprocessing_profile,
            tower_consensus_category=consensus_category,
        )
        db.record_stage_output(
            image["id"], stage="stage2_decide_profile", status="done",
            note="pass-through - no validated preprocessing-profile rule yet, "
                 "see docs/PREPROCESSING_STAGE_NOTES.md's explicit gate",
        )
        if consensus_category is not None:
            sidecar_path = _write_tower_consensus_sidecar(working_path, per_encoder, consensus_category)
            db_sidecar_path = ctx.to_relative(sidecar_path) if ctx is not None else str(sidecar_path)
            db.record_stage_output(
                image["id"], stage="stage2_tower_consensus",
                sidecar_path=db_sidecar_path, status="done",
            )
            print(f"  profile={preprocessing_profile!r}  tower_consensus={consensus_category}")
        else:
            no_evidence += 1
            print(f"  profile={preprocessing_profile!r}  tower_consensus=(no evidence)")
        decided += 1

    print(f"\nStage 2 decided {decided} image(s)"
          + (f", {skipped} not found in DB" if skipped else "")
          + (f", {no_evidence} without tower-consensus evidence" if no_evidence else "")
          + f". Recorded in {db.db_path}")
    return manifest_path


def audit_classification_against_tower_consensus(
    db_path: Path = DEFAULT_DB_PATH,
    baseline_path: Path = DEFAULT_BASELINE_PATH,
    bucket_dir: Path = BUCKET_DIR,
    output_dir: Path | None = None,
) -> Path | None:
    """
    Post-classification audit (2026-08-02, Jon's direction: "we didnt
    have the sensor towers consensus before gemma could classify... run
    that on new files after gemma finishes to get that data then
    compare again to figure out if it was correct"). NOT part of the
    normal Stage 0-6 forward flow - stage2_decide_profiles() runs
    BEFORE Stage 5/Gemma classification, using whatever reference data
    happens to exist at that point (often NONE, for a freshly-acquired
    batch, since nothing's been classified yet - exactly what happened
    on this project's first real production run). This function
    deliberately runs AFTER, once Gemma's real classifications exist to
    build a genuine reference set from, using the IDENTICAL nearest-
    cluster computation (compute_tower_consensus_for_image(), shared
    with stage2_decide_profiles() - not a second implementation).

    Scope per run - ONLY "new" images ("run that on new files"): images
    where `bucket IS NOT NULL` (Gemma has classified them) AND
    `tower_consensus_category IS NULL` (not yet audited). Re-running
    this after a later classification batch only costs work
    proportional to what's genuinely new - already-audited images are
    skipped, matching this project's established idempotent-sync
    pattern (core/pipeline_db.py's sync_bucket_classifications()/
    sync_dewarp_results()).

    For each candidate: updates ONLY images.tower_consensus_category -
    current_stage/status/processing_profile are left untouched, since
    this image already finished Stage 5 and does not "go back" to
    Stage 2 just because its tower-consensus evidence is being computed
    late. Records a stage_outputs row under stage="tower_consensus_audit"
    (deliberately NOT "stage2_tower_consensus" - that stage name means
    "computed before classification"; this is a distinct, later event,
    same reasoning as "auto_sidecar_generation" getting its own
    non-numbered stage label per docs/PIPELINE_DATABASE.md).

    Writes a disagreement report to output_dir (default: `data/outputs/
    tower_consensus_audit/<UTC timestamp>/`) - `summary.json` (counts)
    and `disagreements.csv` (one row per image where Gemma's bucket
    does NOT match the tower-consensus top-voted bucket), sorted so
    "unanimous"/"majority" disagreements - the strongest signal Gemma
    might actually be wrong - sort before weaker "split"/
    "complete_disagreement" ones, where the towers don't even agree
    among themselves. This is evidence for a human to look at, same as
    the Multi-Tower Routing Audit's own disagreement folders - nothing
    here re-routes or corrects a classification automatically.

    Returns output_dir, or None if there was nothing new to audit.
    """
    db = PipelineDatabase(db_path)
    candidates = [
        img for img in db.list_images()
        if img["bucket"] is not None and img["tower_consensus_category"] is None
    ]
    print(f"Auditing {len(candidates)} classified image(s) not yet tower-consensus-audited...")
    if not candidates:
        print("Nothing new to audit.")
        return None

    baseline_records: dict[str, dict] = {}
    if baseline_path.exists():
        raw = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline_records = {resolve_baseline_image_path(r["image"]): r for r in raw}
    else:
        print(f"No baseline embeddings at {baseline_path} - nothing to audit against.")
        return None

    reference_embeddings = build_reference_embeddings(bucket_dir, baseline_path)

    if output_dir is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = PROJECT_ROOT / "data" / "outputs" / "tower_consensus_audit" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    disagreements: list[dict] = []
    audited, no_evidence = 0, 0
    consensus_counts: Counter = Counter()

    for i, image in enumerate(candidates, 1):
        working_path = Path(image["working_path"])
        record = baseline_records.get(resolve_baseline_image_path(str(working_path)))
        if record is None:
            no_evidence += 1
            continue

        consensus_category, top_bucket, per_encoder = compute_tower_consensus_for_image(
            record, reference_embeddings, exclude_key=str(working_path),
        )
        if consensus_category is None:
            no_evidence += 1
            continue

        db.update_image_state(image["id"], tower_consensus_category=consensus_category)
        sidecar_path = _write_tower_consensus_sidecar(working_path, per_encoder, consensus_category)
        db.record_stage_output(
            image["id"], stage="tower_consensus_audit",
            sidecar_path=str(sidecar_path), status="done",
            note=f"gemma_bucket={image['bucket']!r} tower_top_bucket={top_bucket!r}",
        )
        consensus_counts[consensus_category] += 1
        audited += 1

        if top_bucket != image["bucket"]:
            disagreements.append({
                "working_path": str(working_path),
                "gemma_bucket": image["bucket"],
                "gemma_confidence": image["classifier_confidence"],
                "tower_top_bucket": top_bucket,
                "consensus_category": consensus_category,
            })

        if i % 100 == 0 or i == len(candidates):
            print(f"  {i}/{len(candidates)}")

    _SORT_ORDER = {"unanimous": 0, "majority": 1, "split": 2, "complete_disagreement": 3}
    disagreements.sort(key=lambda d: _SORT_ORDER.get(d["consensus_category"], 4))

    with open(output_dir / "disagreements.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "working_path", "gemma_bucket", "gemma_confidence",
            "tower_top_bucket", "consensus_category",
        ])
        writer.writeheader()
        writer.writerows(disagreements)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "candidates": len(candidates),
        "audited": audited,
        "no_evidence": no_evidence,
        "consensus_distribution": dict(consensus_counts),
        "disagreements": len(disagreements),
        "disagreements_by_consensus_category": dict(
            Counter(d["consensus_category"] for d in disagreements)
        ),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAudited {audited} image(s), {no_evidence} without evidence, "
          f"{len(disagreements)} disagreement(s) vs Gemma's classification.")
    print(f"Report: {output_dir}")
    return output_dir


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Post-classification tower-consensus audit: compares Gemma's "
                    "real classification against a freshly-computed tower-consensus "
                    "vote for every classified-but-not-yet-audited image.")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH),
                         help=f"core/pipeline_db.py database path (default: {DEFAULT_DB_PATH})")
    parser.add_argument("--baseline-path", default=str(DEFAULT_BASELINE_PATH),
                         help=f"data/baseline_embeddings.json path (default: {DEFAULT_BASELINE_PATH})")
    parser.add_argument("--bucket-dir", default=str(BUCKET_DIR),
                         help=f"data/buckets directory (default: {BUCKET_DIR})")
    parser.add_argument("--output-dir", default=None,
                         help="Report output directory (default: data/outputs/"
                              "tower_consensus_audit/<UTC timestamp>/)")
    args = parser.parse_args()
    audit_classification_against_tower_consensus(
        db_path=Path(args.db_path),
        baseline_path=Path(args.baseline_path),
        bucket_dir=Path(args.bucket_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )


if __name__ == "__main__":
    main()
