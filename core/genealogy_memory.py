"""
Phase 5 of the agent architecture plan: persistent genealogy
discoveries - the durable payoff layer described in the plan's §7
Memory table ("Genealogy discoveries... structured store, not prose").

Deliberately separate from core/pipeline_db.py, not an extra table
bolted onto it - pipeline_db.py's own docstring is explicit that it
tracks per-image PIPELINE STATE only and never stores evidence-shaped
artifacts (see core/model_residency.py and model_console/
conversation_manager.py's own build notes for the same boundary
already respected once this session, when conversation history hit the
same question). A person/place/date discovery is exactly the kind of
evidence-shaped artifact pipeline_db.py says it will never hold.

One table, deliberately not a normalized person/place/event/relation
schema: `discoveries`, one row per extracted entity (a name, a place, a
date), with its own confidence level and provenance (source file,
category, model). This is intentionally the SMALLEST useful shape - a
flat, searchable fact table - not a family-tree graph. Building real
relationship modeling (who is whose parent, a family tree graph per
the plan's "family_tree_lookup" tool) is real, separate work that needs
actual relationship-extraction logic this project doesn't have yet;
inventing a graph schema now with no real relationship data to put in
it would be exactly the kind of premature architecture this project's
own conventions warn against. `genealogy_framework_lookup` here is a
flat substring/entity-type search over recorded discoveries - useful on
its own (has this name come up before, in what document, how
confident), and a natural foundation a future relationship layer could
build on without a rewrite.

WAL mode + short-lived connections, mirroring core/pipeline_db.py's own
documented reasoning: multiple processes/tools may touch this file
independently (model_console's agent tools, any future batch backfill),
so no connection is held open across a call.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from core.workspace_context import WorkspaceContext

_ws = WorkspaceContext.resolve()
DEFAULT_DB_PATH = _ws.workspace_root / "genealogy_memory.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS discoveries (
    id                INTEGER PRIMARY KEY,
    entity_type       TEXT NOT NULL,      -- 'personal_name' | 'place_name' | 'visible_date'
    value             TEXT NOT NULL,
    confidence        TEXT NOT NULL,      -- core.schema.ConfidenceLevel value
    source_file_path  TEXT NOT NULL,
    source_category   TEXT,               -- core.schema.DocumentCategory value
    model             TEXT,
    prompt_version    TEXT,
    created_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_discoveries_value ON discoveries(value);
CREATE INDEX IF NOT EXISTS ix_discoveries_type ON discoveries(entity_type);
CREATE INDEX IF NOT EXISTS ix_discoveries_source ON discoveries(source_file_path);
"""

# Additive columns applied after CREATE TABLE IF NOT EXISTS (same
# convention as core/pipeline_db.py's _ADDITIVE_COLUMNS): existing DBs
# gain the column with NULL for every historical row - which is the
# CORRECT provenance for those rows (their runtime genuinely wasn't
# recorded at the time; NULL = unknown, never backfilled with a guess,
# per the 2026-08-13 provenance audit's evidence rule).
_ADDITIVE_COLUMNS = (
    # runtime: which execution backend produced this discovery
    # (transformers / vllm / ...) - a result is identified by
    # CHECKPOINT + RUNTIME, not model name alone (multi-runtime lesson,
    # see GenerationConfig.runtime).
    ("discoveries", "runtime", "TEXT"),
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class GenealogyMemory:
    """Thin wrapper over one SQLite file - same "open, do one thing,
    close" discipline as core/pipeline_db.py's PipelineDatabase, for
    the same reason (no long-lived shared connection across processes/
    tools)."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA)
            for table, column, decl in _ADDITIVE_COLUMNS:
                cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
                if column not in cols:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
        finally:
            conn.close()

    def record_extraction_result(self, result, runtime: Optional[str] = None) -> int:
        """
        Records every entity in a core.schema.ExtractionResult as one
        discoveries row each. `result` is typed loosely (not
        `ExtractionResult`) to avoid a hard import-time dependency on
        core.schema for callers that only ever pass one through - see
        core/agent_tools/tools.py's extract_fields_tool, the only
        current caller, which already imports core.schema itself and
        constructs a real ExtractionResult before calling this.

        Returns the number of rows inserted. Deliberately does NOT
        de-duplicate - the same name appearing in two different source
        documents is two genuine, separately-provenanced discoveries,
        not a duplicate to collapse. A future lookup can group by value
        itself; this layer stays a plain append-only fact log, matching
        this project's existing append-only discipline elsewhere
        (core/pipeline_db.py's stage_outputs, ground_truth_log.jsonl).
        """
        conn = self._connect()
        inserted = 0
        try:
            conn.execute("BEGIN")
            now = _now()
            for name in result.personal_names:
                conn.execute(
                    "INSERT INTO discoveries (entity_type, value, confidence, "
                    "source_file_path, source_category, model, prompt_version, created_at, runtime) "
                    "VALUES ('personal_name', ?, ?, ?, ?, ?, ?, ?, ?)",
                    (name.value, name.confidence.value, result.file_path,
                     result.category.value, result.model, result.prompt_version, now, runtime),
                )
                inserted += 1
            for place in result.place_names:
                conn.execute(
                    "INSERT INTO discoveries (entity_type, value, confidence, "
                    "source_file_path, source_category, model, prompt_version, created_at, runtime) "
                    "VALUES ('place_name', ?, ?, ?, ?, ?, ?, ?, ?)",
                    (place.value, place.confidence.value, result.file_path,
                     result.category.value, result.model, result.prompt_version, now, runtime),
                )
                inserted += 1
            for date in result.visible_dates:
                conn.execute(
                    "INSERT INTO discoveries (entity_type, value, confidence, "
                    "source_file_path, source_category, model, prompt_version, created_at, runtime) "
                    "VALUES ('visible_date', ?, ?, ?, ?, ?, ?, ?, ?)",
                    (date.value, date.confidence.value, result.file_path,
                     result.category.value, result.model, result.prompt_version, now, runtime),
                )
                inserted += 1
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()
        return inserted

    def search(
        self, query: str, entity_type: Optional[str] = None, limit: int = 25,
    ) -> list[dict]:
        """
        Case-insensitive substring search over `value`, most recent
        first. `entity_type`, if given, must be one of 'personal_name'/
        'place_name'/'visible_date' - filters, doesn't fuzzy-match (a
        typo'd entity_type is a caller bug, not something to guess
        around, matching this project's "loud error over silent wrong
        behavior" discipline elsewhere).
        """
        conn = self._connect()
        try:
            clauses = ["value LIKE ? COLLATE NOCASE"]
            params: list = [f"%{query}%"]
            if entity_type is not None:
                clauses.append("entity_type = ?")
                params.append(entity_type)
            where = " AND ".join(clauses)
            rows = conn.execute(
                f"SELECT * FROM discoveries WHERE {where} ORDER BY id DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


# Process-wide default instance, same convenience pattern as
# core/model_residency.py's module-level `residency` - callers needing
# a non-default db_path (tests) construct GenealogyMemory() directly
# instead of using this.
memory = GenealogyMemory()
