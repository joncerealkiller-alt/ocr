"""
The pipeline's orchestration system of record (2026-08-02) - per Jon's
explicit design, this database answers "what is the current state of
this image?" and nothing else. It never stores embeddings, OCR output,
physical/semantic measurements, or any other large or evidence-shaped
artefact - those stay exactly where they already live today (per-image
JSON sidecars, data/baseline_embeddings.json, bucket CSVs' eventual
sidecar equivalents), on disk, immutable once written. See
docs/PIPELINE_DATABASE.md for the full design rationale, the schema's
"why" per table, and how each of Stage 0-6 is meant to interact with it.

Two tables only, deliberately not the more normalized 6-table shape an
earlier design pass sketched:

  images        - one row per logical image, current state (small, wide,
                  fast to query - source/working path, both hashes,
                  current_stage, status, routing outcome).
  stage_outputs - one row per (image, stage-run) - an append-only log
                  that ALSO doubles as the sidecar location index
                  (sidecar_path/lookup_key). Never the evidence itself,
                  only a pointer to it plus enough metadata (sha256,
                  status, note) to know whether that evidence is still
                  trustworthy without opening it.

TWO HASH COLUMNS, not one - this is the one deliberate deviation from
Jon's own sketch (which had a single `sha256` field), added specifically
because of a real bug this session: baseline embeddings were captured
against an already-preprocessed working corpus and silently mislabeled
"pre_preprocessing" (see docs/REFERENCE_PIPELINE_V1.md) - there was no
stored fact anywhere that could have caught this at the time. identity_
hash is captured ONCE, at Stage 0, before any preprocessing, and never
recomputed - it is the stable proof "this is genuinely the same source
image" independent of whatever a later stage legitimately does to its
pixels. current_hash is recomputed by whichever stage last touched
working_path, and answers a different question: "has anything changed
since the DB last looked." Conflating these into one field is exactly
what made the original bug possible.

STDLIB ONLY (sqlite3/hashlib/pathlib/datetime) - no torch/timm or any
other heavy dependency, matching this project's existing lazy-import
discipline (e.g. core/manifest_pipeline.py's stage1_capture_baseline_
embeddings() only imports core.baseline_embeddings, itself torch/timm-
heavy, inside the function body, not at module top) so importing this
module never forces a heavy import cost on a caller that only wants
state, not evidence.

WAL mode + short-lived connections: every public method opens its own
connection, does its work in one transaction, and closes - no
connection is held open across calls. This project already runs
multiple independent processes against the same data/ directory
(ui/build_manifest_ui.py shells stages out as subprocesses; several
Tkinter tools hold long review sessions open) - WAL mode plus
short transactions is the standard safe pattern for that, not one
long-lived shared connection.
"""

from __future__ import annotations

import csv
import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from core.workspace_context import WorkspaceContext

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Legacy fallbacks - see core/manifest_pipeline.py's identical comment.
# New callers should resolve these via WorkspaceContext/RunContext
# (ws.pipeline_db_path, ctx.buckets) instead. Once the workspace-root
# pipeline.db exists (post scripts/migrate_pipeline_db_to_run_schema.py),
# it takes precedence over the pre-migration data/pipeline.db path.
_ws = WorkspaceContext.resolve()
if _ws.pipeline_db_path.exists():
    DEFAULT_DB_PATH = _ws.pipeline_db_path
else:
    DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "pipeline.db"

_legacy_bucket_dir = _ws.runs_root / "legacy_pre_run_system" / "outputs" / "buckets"
BUCKET_DIR = _legacy_bucket_dir if _legacy_bucket_dir.exists() else PROJECT_ROOT / "data" / "buckets"

# Columns update_image_state() is allowed to write - a deliberate
# whitelist (not **kwargs passed straight into SQL) so a typo'd field
# name fails loudly instead of silently doing nothing or, worse, being
# interpolated into a query string.
_UPDATABLE_IMAGE_FIELDS = {
    "source_path", "source_type", "page_number", "working_path",
    "identity_hash", "current_hash", "current_stage", "status",
    "bucket", "classifier_confidence", "classifier_model",
    "processing_profile", "tower_consensus_category", "needs_manual_dewarp",
    "run_id",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id                      INTEGER PRIMARY KEY,
    source_path             TEXT NOT NULL,
    source_type             TEXT NOT NULL DEFAULT 'image',
    page_number             INTEGER,
    working_path            TEXT NOT NULL,
    identity_hash           TEXT NOT NULL,
    current_hash            TEXT,
    current_stage           INTEGER NOT NULL DEFAULT 0,
    status                  TEXT NOT NULL DEFAULT 'acquired',
    bucket                  TEXT,
    classifier_confidence   REAL,
    classifier_model        TEXT,
    processing_profile      TEXT,
    created_at              TEXT NOT NULL,
    updated_at              TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_images_working_path ON images(working_path);
CREATE INDEX IF NOT EXISTS ix_images_source_path  ON images(source_path);
CREATE INDEX IF NOT EXISTS ix_images_stage_status ON images(current_stage, status);
CREATE INDEX IF NOT EXISTS ix_images_bucket       ON images(bucket);

CREATE TABLE IF NOT EXISTS stage_outputs (
    id            INTEGER PRIMARY KEY,
    image_id      INTEGER NOT NULL REFERENCES images(id),
    stage         TEXT NOT NULL,
    sidecar_path  TEXT,
    lookup_key    TEXT,
    sha256        TEXT,
    status        TEXT NOT NULL,
    note          TEXT,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_stage_outputs_image ON stage_outputs(image_id, stage);
"""

# Columns added AFTER the original schema shipped (2026-08-02, Stage 2/
# Decision Engine) - CREATE TABLE IF NOT EXISTS above is a no-op against
# an already-existing images table, so a new column needs an explicit
# ALTER TABLE migration, applied only if genuinely missing (checked via
# PRAGMA table_info, not just attempted-and-caught) so re-running this
# against the same database file is always safe and never touches the
# existing rows. Add future new columns here, not to the CREATE TABLE
# above alone - that only helps a brand-new database file.
_COLUMN_MIGRATIONS = [
    ("images", "tower_consensus_category", "TEXT"),
    # Hashed-run migration (see docs/RUN_ARCHITECTURE.md and
    # scripts/migrate_pipeline_db_to_run_schema.py): from this point on,
    # source_path/working_path/sidecar_path/lookup_key are stored
    # RUN-RELATIVE (e.g. "working/images/foo.jpg"), not absolute -
    # resolve_path(run_id, relative_path) reconstructs the real
    # filesystem path. NULL for any row not yet migrated (the migration
    # script backfills every pre-existing row with
    # run_id="legacy_pre_run_system"); NOT NULL is enforced at the
    # application layer (get_or_create_image() requires it going
    # forward), not via a DB-level constraint, so this ALTER TABLE stays
    # safe to run against an existing populated table.
    ("images", "run_id", "TEXT"),
    # 2026-08-03 consolidation pass: "does this dense_tabular_rows image
    # need manual dewarp" was being decided by a hardcoded stub
    # (core/manifest_pipeline.py's _dense_tabular_needs_manual_dewarp())
    # called inline inside finalize_manifest(), with no durable record -
    # a second consumer (the dewarp worklist tooling) independently
    # assumed the same answer for the whole bucket rather than reading a
    # shared decision. This column makes the decision durable, queryable
    # state instead of an ephemeral function call whose result only ever
    # reached manifest_final.csv. INTEGER 0/1, NULL for any image the
    # question doesn't apply to (not dense_tabular_rows) - same "None
    # means not applicable" convention as core/image_analysis.py's
    # sensor fields, not a third boolean-ish value invented here.
    ("images", "needs_manual_dewarp", "INTEGER"),
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_file(path: str | Path) -> str:
    """
    sha256 of a file's current bytes - the one shared primitive both
    identity_hash (Stage 0, once) and current_hash (whichever stage last
    touched the file) are computed with. Matches core/baseline_
    embeddings.py's own _image_hash() (same algorithm, same "hash the
    whole file" approach) - not reimplemented differently here by
    accident.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PipelineDatabase:
    """
    Thin wrapper over one SQLite file. Every public method is a
    complete, self-contained unit of work: open a connection, do
    exactly one thing, commit, close. No method holds a connection or
    a transaction open across a return to the caller - see module
    docstring for why.
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA)
            for table, column, coltype in _COLUMN_MIGRATIONS:
                existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
                if column not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")
            conn.execute("CREATE INDEX IF NOT EXISTS ix_images_run_working ON images(run_id, working_path)")
        finally:
            conn.close()

    # -- run-relative path resolution ----------------------------------

    def resolve_path(self, run_id: str, relative_path: str) -> Path:
        """Reconstructs a real filesystem path from a run_id + the
        RUN-RELATIVE path string stored in a DB row (working_path/
        sidecar_path/lookup_key for any run-owned artifact - see
        docs/RUN_ARCHITECTURE.md). The single invariant every row, old
        (legacy_pre_run_system, post-migration) or new, resolves
        through - no special case for pre-migration data, and no
        permanent absolute-path fallback. Callers are responsible for
        normalizing a real filesystem Path into this relative form
        before writing it (see RunContext.to_relative()).

        Derives runs_root from THIS database's own location
        (self.db_path.parent / "runs"), not a freshly re-resolved
        WorkspaceContext - self.db_path is always workspace_root/
        pipeline.db (see WorkspaceContext.pipeline_db_path), so this
        stays correct even when the ambient env/config WorkspaceContext
        differs from the one this particular database instance was
        opened against (e.g. tests pointed at an isolated workspace)."""
        return self.db_path.parent / "runs" / run_id / relative_path

    def resolve_image_path(self, image: dict, field: str = "working_path") -> Path:
        """Convenience: resolve_path() using an image row's own run_id +
        the given path field (default working_path)."""
        return self.resolve_path(image["run_id"], image[field])

    # -- images --------------------------------------------------------

    def get_or_create_image(
        self,
        source_path: str,
        working_path: str,
        source_type: str = "image",
        page_number: int | None = None,
        identity_hash: str | None = None,
        run_id: str | None = None,
    ) -> int:
        """
        Looks up an existing row by working_path first (the common case
        once an image has been acquired), then source_path (covers a
        re-run against the same source before a working copy exists at
        that exact path). If neither matches, inserts a new row.

        run_id (hashed-run migration): scopes the working_path/
        source_path lookup to this run - two different runs may
        legitimately have the same-named file at the same run-relative
        path (e.g. "working/images/foo.jpg" in both), and without this
        scope they'd collide. When run_id is None (a caller not yet
        migrated to pass it, or absolute legacy paths), lookup is
        unscoped, matching the pre-migration behavior. New callers
        should always pass run_id and a RUN-RELATIVE working_path/
        source_path (e.g. from ctx.working_images/... relative to
        ctx.run_root) - see docs/RUN_ARCHITECTURE.md.

        identity_hash: if not given, computed from working_path right
        now via hash_file() - the normal case, called once at Stage 0
        acquisition time, before any later stage has touched the file.
        Passing it explicitly is for callers (e.g. the migration
        script) that already know the hash from elsewhere and shouldn't
        pay to recompute it. NOTE: when working_path is run-relative,
        hash_file() cannot resolve it directly - such callers must
        always pass identity_hash explicitly (resolve the real path via
        resolve_path() first).
        """
        conn = self._connect()
        try:
            if run_id is not None:
                row = conn.execute(
                    "SELECT id FROM images WHERE run_id = ? AND (working_path = ? OR source_path = ?) "
                    "ORDER BY (working_path = ?) DESC LIMIT 1",
                    (run_id, working_path, source_path, working_path),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT id FROM images WHERE working_path = ? OR source_path = ? "
                    "ORDER BY (working_path = ?) DESC LIMIT 1",
                    (working_path, source_path, working_path),
                ).fetchone()
            if row is not None:
                return row["id"]

            if identity_hash is None:
                identity_hash = hash_file(working_path)
            now = _now()
            cur = conn.execute(
                "INSERT INTO images "
                "(source_path, source_type, page_number, working_path, "
                " identity_hash, current_hash, current_stage, status, "
                " run_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 0, 'acquired', ?, ?, ?)",
                (source_path, source_type, page_number, working_path,
                 identity_hash, identity_hash, run_id, now, now),
            )
            return cur.lastrowid
        finally:
            conn.close()

    def get_image_by_path(self, path: str, run_id: str | None = None) -> dict | None:
        """Checks both working_path and source_path - a caller with a
        plain file-path string (the norm throughout this project today)
        shouldn't need to know which column it currently lives under.

        run_id (hashed-run migration): scopes the match to this run -
        see get_or_create_image()'s docstring for why. Unscoped (None)
        by default for backward compatibility with not-yet-migrated
        callers passing absolute legacy paths, which stay globally
        unique in practice."""
        conn = self._connect()
        try:
            if run_id is not None:
                row = conn.execute(
                    "SELECT * FROM images WHERE run_id = ? AND (working_path = ? OR source_path = ?) "
                    "ORDER BY (working_path = ?) DESC LIMIT 1",
                    (run_id, path, path, path),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM images WHERE working_path = ? OR source_path = ? "
                    "ORDER BY (working_path = ?) DESC LIMIT 1",
                    (path, path, path),
                ).fetchone()
            return dict(row) if row is not None else None
        finally:
            conn.close()

    def get_image(self, image_id: int) -> dict | None:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
            return dict(row) if row is not None else None
        finally:
            conn.close()

    def update_image_state(self, image_id: int, **fields) -> None:
        """
        Whitelisted column updater - see _UPDATABLE_IMAGE_FIELDS. Always
        bumps updated_at, regardless of which fields were passed, since
        "something about this image's state changed" is true any time
        this is called at all.
        """
        if not fields:
            return
        unknown = set(fields) - _UPDATABLE_IMAGE_FIELDS
        if unknown:
            raise ValueError(
                f"update_image_state() got unknown field(s) {sorted(unknown)} - "
                f"not in the whitelisted set {sorted(_UPDATABLE_IMAGE_FIELDS)}. "
                f"If this is a genuinely new column, add it to the schema and "
                f"the whitelist together, not just here."
            )
        fields = dict(fields, updated_at=_now())
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        conn = self._connect()
        try:
            conn.execute(
                f"UPDATE images SET {set_clause} WHERE id = ?",
                (*fields.values(), image_id),
            )
        finally:
            conn.close()

    def list_images(
        self,
        current_stage: int | None = None,
        status: str | None = None,
        bucket: str | None = None,
    ) -> list[dict]:
        clauses, params = [], []
        if current_stage is not None:
            clauses.append("current_stage = ?")
            params.append(current_stage)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if bucket is not None:
            clauses.append("bucket = ?")
            params.append(bucket)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        conn = self._connect()
        try:
            rows = conn.execute(f"SELECT * FROM images {where} ORDER BY id", params).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # -- stage_outputs ---------------------------------------------------

    def record_stage_output(
        self,
        image_id: int,
        stage: str,
        sidecar_path: str | None = None,
        lookup_key: str | None = None,
        sha256: str | None = None,
        status: str = "done",
        note: str | None = None,
    ) -> int:
        """
        Appends one row - never updates or deletes a prior stage_outputs
        row, matching this project's existing append-only discipline for
        historical logs (ground_truth_log.jsonl, reviewed_uncertain.csv).
        A stage that runs again on the same image (e.g. re-classification
        after a correction) gets a SECOND row, not an overwritten one -
        the full history stays inspectable.
        """
        conn = self._connect()
        try:
            cur = conn.execute(
                "INSERT INTO stage_outputs "
                "(image_id, stage, sidecar_path, lookup_key, sha256, status, note, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (image_id, stage, sidecar_path, lookup_key, sha256, status, note, _now()),
            )
            return cur.lastrowid
        finally:
            conn.close()

    def get_stage_outputs(self, image_id: int, stage: str | None = None) -> list[dict]:
        conn = self._connect()
        try:
            if stage is None:
                rows = conn.execute(
                    "SELECT * FROM stage_outputs WHERE image_id = ? ORDER BY id",
                    (image_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM stage_outputs WHERE image_id = ? AND stage = ? ORDER BY id",
                    (image_id, stage),
                ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def list_stage_outputs(
        self, stage: str | None = None, status: str | None = None, limit: int = 100,
    ) -> list[dict]:
        """
        Corpus-wide stage_outputs, most recent first - unlike get_stage_
        outputs() (scoped to one image_id), this is for a global "what's
        been happening across the whole run" view. Added 2026-08-03 for
        the read-only API's /logs endpoint (docs/UI_MOCKUPS_INTEGRATION_
        NOTES.md) - stage_outputs is already a structured, timestamped,
        append-only event log (image_id, stage, status, note, created_at)
        in everything but name, so this exposes it directly rather than
        building a second, parallel logging system.
        """
        clauses, params = [], []
        if stage is not None:
            clauses.append("stage = ?")
            params.append(stage)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT * FROM stage_outputs {where} ORDER BY id DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def find_stage_output(self, stage: str, lookup_key: str) -> dict | None:
        """
        Finds the most recent stage_outputs row by (stage, lookup_key)
        ACROSS ALL IMAGES, regardless of image_id - needed when the
        image's own identifying path may have moved on since that stage
        ran (e.g. a Stage 2a dewarp updates images.working_path to a new
        file, so a later idempotency check can no longer find the row
        via get_image_by_path(old_path), but can still find the
        historical record via lookup_key, which stays whatever it was
        recorded as at write time). Returns None if never recorded.
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM stage_outputs WHERE stage = ? AND lookup_key = ? "
                "ORDER BY id DESC LIMIT 1",
                (stage, lookup_key),
            ).fetchone()
            return dict(row) if row is not None else None
        finally:
            conn.close()

    def record_classification(
        self,
        image_id: int,
        bucket: str,
        confidence: float | None,
        model: str | None,
        status: str,
        sidecar_path: str | None = None,
        lookup_key: str | None = None,
        stage_output_status: str = "done",
        note: str | None = None,
    ) -> None:
        """
        Atomically records ONE classification outcome - the images
        UPDATE (bucket/classifier_confidence/classifier_model/
        current_stage=5/status) and the "stage5_classify" stage_outputs
        INSERT together, in a single explicit SQLite transaction. This
        connection's isolation_level=None means autocommit per statement
        by default (see _connect()) - the explicit BEGIN/COMMIT here is
        what makes these two statements one atomic unit instead of two
        independent ones that could land only one of; ROLLBACK on any
        exception leaves both tables exactly as they were before this
        call, never a bucket change with no matching audit-log row or
        vice versa.

        2026-08-03 consolidation pass, Jon's framing: "the classifier
        should either finish successfully or leave the previous state
        untouched" for a given image, "that makes recovery much
        cleaner." This is THE authoritative write for Stage 5
        classification as of this pass - core/classifier.py calls this
        directly, per image, as soon as that image's model call
        returns/fails, rather than writing only a bucket CSV row and
        relying on a later, separate sync_bucket_classifications() pass
        to bring the DB up to date after the fact. Bucket CSVs are still
        written first, by the caller, as a derived export (other tools
        still read them directly - ui/dewarp_preprocessor_ui.py's bucket
        worklists, debug_tools/review_uncertain.py) - see core/
        classifier.py's own docstring for why CSV-then-DB is the safer
        order given what sync_bucket_classifications() can and can't
        recover from a partial failure.
        """
        conn = self._connect()
        try:
            conn.execute("BEGIN")
            now = _now()
            conn.execute(
                "UPDATE images SET bucket = ?, classifier_confidence = ?, "
                "classifier_model = ?, current_stage = 5, status = ?, updated_at = ? "
                "WHERE id = ?",
                (bucket, confidence, model, status, now, image_id),
            )
            conn.execute(
                "INSERT INTO stage_outputs "
                "(image_id, stage, sidecar_path, lookup_key, sha256, status, note, created_at) "
                "VALUES (?, 'stage5_classify', ?, ?, NULL, ?, ?, ?)",
                (image_id, sidecar_path, lookup_key, stage_output_status, note, now),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()


# -- Sync helpers: bring the DB up to date from Stage 5/2a's own CSVs -------
#
# Promoted 2026-08-02 from scripts/migrate_manifest_to_db.py's one-time
# import_bucket_classifications()/import_dewarp_results() - core/
# manifest_pipeline.py's finalize_manifest() needs the SAME logic (reused,
# not duplicated) to keep the DB in sync every time it runs, not just once.
# Module-level functions, not PipelineDatabase methods - they compose
# several PipelineDatabase calls plus real business logic (DocumentCategory,
# bucket file-naming conventions), which is a different kind of thing than
# the class's own thin per-row primitives.


def sync_bucket_classifications(
    db: "PipelineDatabase", bucket_dir: Path = BUCKET_DIR, ctx=None,
) -> int:
    """
    ctx (hashed-run migration): when given a RunContext, overrides
    bucket_dir with ctx.buckets, AND every DB identity lookup/write
    below (get_image_by_path, get_or_create_image, find_stage_output/
    record_stage_output's lookup_key) is normalized through
    ctx.to_relative()/scoped by ctx.run_id instead of matching the
    bucket CSV's absolute file_path column directly - the same
    normalization every other stage already does, fixed here while
    revisiting this function after the migration (previously this was a
    documented, deliberately deferred gap - see docs/RUN_ARCHITECTURE.md).
    Typed loosely (not `RunContext`) to avoid a core.run_context <->
    core.pipeline_db import cycle; core.manifest_pipeline/core.classifier
    (which import both) are the expected callers to pass this.

    Reads every data/buckets/<category>.csv and brings the DB's
    images.bucket/classifier_confidence/classifier_model up to date.

    Auto-registers a file_path not already known to the DB
    (get_or_create_image()) rather than skipping it - a bucket CSV is
    allowed to reference files Stage 0 never registered (an older
    workflow, e.g. scripts/archive/build_manifest.py's legacy path), and
    a caller like finalize_manifest() must never silently drop a real
    classified file just because Stage 0 didn't run for it. A path that
    no longer exists ON DISK is skipped with a printed warning instead -
    there is nothing to hash, so nothing safe to register.

    IDEMPOTENT: only updates + records a new stage_outputs row when this
    image's classification has genuinely changed (bucket, confidence, or
    model differs from the DB's current record) since the last sync -
    repeated calls (e.g. every finalize_manifest() re-run) don't grow
    stage_outputs with duplicate no-op entries. A row with a non-empty
    "error" column (a hard pipeline failure, not a real classification -
    core/classifier.py's own distinction) never updates bucket/routing.

    Returns the number of images actually updated (0 on a fully-synced
    re-run).
    """
    from core.schema import DocumentCategory

    if ctx is not None:
        bucket_dir = ctx.buckets

    updated = 0
    for category in [c.value for c in DocumentCategory]:
        bucket_csv = bucket_dir / f"{category}.csv"
        if not bucket_csv.exists():
            continue
        with open(bucket_csv, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

        for row in rows:
            file_path = row.get("file_path")
            if not file_path:
                continue

            # DB identity is normalized through ctx (RunContext.to_relative())
            # when given - the bucket CSV's own file_path column stays
            # absolute (manifest/bucket CSVs are meant to be directly
            # openable without a RunContext in hand), but the DB's
            # working_path/lookup_key are run-relative, so a lookup by
            # the raw absolute file_path would never match a ctx-based
            # row. See docs/RUN_ARCHITECTURE.md.
            lookup_key = ctx.to_relative(file_path) if ctx is not None else file_path
            run_id = ctx.run_id if ctx is not None else None

            image = db.get_image_by_path(lookup_key, run_id=run_id)
            if image is None:
                if not Path(file_path).exists():
                    print(f"WARNING: {file_path!r} in {bucket_csv.name} not registered "
                          f"in the DB and no longer exists on disk - skipped.")
                    continue
                image_id = db.get_or_create_image(
                    source_path=file_path, working_path=lookup_key, run_id=run_id,
                    identity_hash=hash_file(file_path) if ctx is not None else None,
                )
                image = db.get_image(image_id)

            error = row.get("error", "")
            if error:
                # A hard pipeline failure (core/classifier.py's own
                # distinction) - never a real classification, never
                # updates bucket/routing. IDEMPOTENT the same way as the
                # main path below: an error row round-trips untouched in
                # uncertain_review.csv indefinitely (debug_tools/review_
                # uncertain.py deliberately never mutates it), so without
                # this check every re-sync would re-append an identical
                # "failed" stage_outputs row forever.
                note = error[:300]
                # find_stage_output() is unscoped by run_id (searches
                # ACROSS ALL IMAGES by lookup_key alone - see its own
                # docstring), but that's still correct here: lookup_key
                # is now the run-relative path when ctx is set, and two
                # DIFFERENT runs can share the exact same run-relative
                # string (e.g. "working/images/foo.jpg" in both) - so
                # this could in principle find a stale row from a
                # DIFFERENT run with the same relative path. Acceptable
                # for this pass since a false match only skips a
                # redundant re-write of an identical "failed" status/note
                # pair, never corrupts state - see docs/RUN_ARCHITECTURE.md's
                # "Known gaps" if this needs tightening later.
                existing = db.find_stage_output("stage5_classify", lookup_key)
                if existing is None or existing["status"] != "failed" or existing["note"] != note:
                    db.record_stage_output(
                        image["id"], stage="stage5_classify",
                        sidecar_path=ctx.to_relative(bucket_csv) if ctx is not None else str(bucket_csv),
                        lookup_key=lookup_key,
                        status="failed", note=note,
                    )
                    updated += 1
                continue

            confidence = row.get("confidence")
            confidence_val = float(confidence) if confidence not in (None, "") else None
            model = row.get("model") or None

            if (image["bucket"] == category
                    and image["classifier_confidence"] == confidence_val
                    and image["classifier_model"] == model):
                continue  # already in sync

            db.update_image_state(
                image["id"], bucket=category,
                classifier_confidence=confidence_val, classifier_model=model,
                current_stage=5,
                status="uncertain_review" if category == "uncertain_review" else "classified",
            )
            db.record_stage_output(
                image["id"], stage="stage5_classify",
                sidecar_path=ctx.to_relative(bucket_csv) if ctx is not None else str(bucket_csv),
                lookup_key=lookup_key, status="done",
            )
            updated += 1
    return updated


def sync_one_dewarp_result(
    db: "PipelineDatabase",
    source_file_path: str,
    output_file_path: str,
    dewarped_csv_path: Path,
) -> bool:
    """
    Records ONE dewarp/bypass result into the DB - the shared unit both
    sync_dewarp_results() (batch, reads a whole existing *_dewarped.csv)
    and ui/dewarp_preprocessor_ui.py (live, called once per Save/Bypass
    as it happens, 2026-08-02) call, so both stay the exact same logic
    rather than two parallel implementations that could drift apart.

    Updates images.working_path to output_file_path (recomputing
    current_hash), status="ready_dewarped" - true whether this row's
    original CSV status was "dewarped" or "bypassed" (matching this
    project's pre-existing finalize_manifest() semantics, which never
    distinguished the two either - a bypassed entry has output_file_path
    == source_file_path, so this is a same-value "update"). Records a
    stage_outputs "stage2a_dewarp" row with lookup_key=the PRE-dewarp/
    bypass path, so finalize_manifest() can recover "what was this file
    before" without a dedicated column.

    IDEMPOTENT VIA STAGE_OUTPUTS HISTORY, not current state - once
    working_path moves, get_image_by_path(source_file_path) can no
    longer find the row by its old path (that's the whole point of
    updating working_path), so a naive "does current state already
    match" check would break on a second call. Checking
    find_stage_output(lookup_key=source_file_path) instead stays correct
    whether called once from the live UI or again later from a batch
    sync over the same CSV row.

    Returns True if this call actually wrote anything (False if this
    source_file_path was already synced - a normal, expected case for
    the live UI call site, not an error, since a batch sync may have
    already picked up a row the live UI is also reporting).
    """
    if db.find_stage_output("stage2a_dewarp", source_file_path) is not None:
        return False

    image = db.get_image_by_path(source_file_path)
    if image is None:
        print(f"WARNING: {source_file_path!r} not found in DB - skipped.")
        return False

    new_hash = hash_file(output_file_path) if Path(output_file_path).exists() else None
    db.update_image_state(
        image["id"], working_path=output_file_path,
        current_hash=new_hash, status="ready_dewarped",
    )
    db.record_stage_output(
        image["id"], stage="stage2a_dewarp",
        sidecar_path=str(dewarped_csv_path), lookup_key=source_file_path, status="done",
    )
    return True


def sync_dewarp_results(db: "PipelineDatabase", bucket_dir: Path = BUCKET_DIR, ctx=None) -> int:
    """
    Reads every data/buckets/<category>_dewarped.csv (ui/dewarp_
    preprocessor_ui.py's bucket-worklist output) and calls
    sync_one_dewarp_result() per row - see that function for the actual
    update/idempotency logic, shared with the live UI call site.

    ctx: same override as sync_bucket_classifications() - see its
    docstring.
    """
    if ctx is not None:
        bucket_dir = ctx.buckets
    updated = 0
    for dewarped_csv in sorted(bucket_dir.glob("*_dewarped.csv")):
        with open(dewarped_csv, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

        for row in rows:
            source_file_path = row.get("source_file_path")
            output_file_path = row.get("output_file_path")
            if not source_file_path or not output_file_path:
                continue
            if sync_one_dewarp_result(db, source_file_path, output_file_path, dewarped_csv):
                updated += 1
    return updated
